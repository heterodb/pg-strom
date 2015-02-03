/*
 * gpuscan.c
 *
 * Sequential scan accelerated by GPU processors
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "access/sysattr.h"
#include "catalog/pg_type.h"
#include "catalog/pg_namespace.h"
#include "commands/explain.h"
#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/execnodes.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
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
#include "utils/syscache.h"
#include "utils/tqual.h"
#include "pg_strom.h"
#include "device_gpuscan.h"

static set_rel_pathlist_hook_type	set_rel_pathlist_next;
static CustomPathMethods		gpuscan_path_methods;
static CustomScanMethods		gpuscan_plan_methods;
static PGStromExecMethods		gpuscan_exec_methods;
static bool						enable_gpuscan;

/*
 * Path information of GpuScan
 */
typedef struct {
	CustomPath	cpath;
	List	   *host_quals;		/* RestrictInfo run on host */
	List	   *dev_quals;		/* RestrictInfo run on device */
} GpuScanPath;

/*
 * form/deform interface of private field of CustomScan(GpuScan)
 */
typedef struct {
	char	   *kern_source;	/* source of opencl kernel */
	int32		extra_flags;	/* extra libraries to be included */
	List	   *used_params;	/* list of Const/Param in use */
	List	   *used_vars;		/* list of Var in use */
	List	   *dev_quals;		/* qualifiers to be run on device */
} GpuScanInfo;

static inline void
form_gpuscan_info(CustomScan *cscan, GpuScanInfo *gscan_info)
{
	cscan->custom_private =
		list_make2(makeString(gscan_info->kern_source),
				   makeInteger(gscan_info->extra_flags));
	cscan->custom_exprs =
		list_make3(gscan_info->used_params,
				   gscan_info->used_vars,
				   gscan_info->dev_quals);
}

static GpuScanInfo *
deform_gpuscan_info(Plan *plan)
{
	GpuScanInfo *result = palloc0(sizeof(GpuScanInfo));
	CustomScan	*cscan = (CustomScan *) plan;

	Assert(IsA(cscan, CustomScan));
	result->kern_source = strVal(linitial(cscan->custom_private));
	result->extra_flags = intVal(lsecond(cscan->custom_private));
	result->used_params = linitial(cscan->custom_exprs);
	result->used_vars = lsecond(cscan->custom_exprs);
	result->dev_quals = lthird(cscan->custom_exprs);

	return result;
}

typedef struct {
	CustomScanState	css;
	GpuTaskState	gts;

	BlockNumber		curr_blknum;
	BlockNumber		last_blknum;
	HeapTupleData	scan_tuple;
	List		   *dev_quals;

	const char	   *kern_source;
	int32			extra_flags;
	kern_parambuf  *kparams;

	pgstrom_gpuscan *curr_chunk;
	uint32			curr_index;
	cl_uint			num_rechecked;
	cl_uint			num_running;
	pgstrom_perfmon	pfm;	/* sum of performance counter */
} GpuScanState;

typedef struct
{
	GpuTask			task;
	CUdeviceptr		m_gpuscan;
	CUdeviceptr		m_kds;
	CUfunction		kern_qual;
	CUevent 		ev_dma_send_start;
	CUevent			ev_dma_send_stop;	/* also, start kernel exec */
	CUevent			ev_dma_recv_start;	/* also, stop kernel exec */
	CUevent			ev_dma_recv_stop;
	pgstrom_data_store *pds;
	kern_gpuscan	kern;
} pgstrom_gpuscan;

/* static functions */
static void clserv_process_gpuscan(pgstrom_message *msg);

/*
 * cost_gpuscan
 *
 * cost estimation for GpuScan
 */
static void
cost_gpuscan(CustomPath *pathnode, PlannerInfo *root,
			 RelOptInfo *baserel, ParamPathInfo *param_info,
			 List *host_quals, List *dev_quals, bool is_bulkload)
{
	Path	   *path = &pathnode->path;
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
	/* Disk costs */
    run_cost += spc_seq_page_cost * baserel->pages;

	/* GPU costs */
	cost_qual_eval(&dev_cost, dev_quals, root);
	dev_sel = clauselist_selectivity(root, dev_quals, 0, JOIN_INNER, NULL);
	dev_cost.startup += pgstrom_gpu_setup_cost;
	if (cpu_tuple_cost > 0.0)
		dev_cost.per_tuple *= pgstrom_gpu_tuple_cost / cpu_tuple_cost;
	else
		dev_cost.per_tuple += disable_cost;

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
	if (!is_bulkload)
		cpu_per_tuple = host_cost.per_tuple + cpu_tuple_cost;
	else
		cpu_per_tuple = host_cost.per_tuple;
	gpu_per_tuple = dev_cost.per_tuple;
	run_cost += (gpu_per_tuple * baserel->tuples +
				 cpu_per_tuple * dev_sel * baserel->tuples);

	path->startup_cost = startup_cost;
    path->total_cost = startup_cost + run_cost;
}

static void
gpuscan_add_scan_path(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Index rtindex,
					  RangeTblEntry *rte)
{
	CustomPath	   *pathnode;
	List		   *dev_quals = NIL;
	List		   *host_quals = NIL;
	ListCell	   *cell;
	codegen_context	context;

	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);

	/* nothing to do, if either PG-Strom or GpuScan is not enabled */
	if (!pgstrom_enabled() || !enable_gpuscan)
		return;

	/* only base relation we can handle */
	if (baserel->rtekind != RTE_RELATION || baserel->relid == 0)
		return;

	/* system catalog is not supported */
	if (get_rel_namespace(rte->relid) == PG_CATALOG_NAMESPACE)
		return;

	/*
	 * check whether the qualifier can run on GPU device, or not
	 */
	pgstrom_init_codegen_context(&context);
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_codegen_available_expression(rinfo->clause))
			dev_quals = lappend(dev_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}

	/*
	 * FIXME: needs to pay attention for projection cost.
	 */

	/*
	 * Construction of a custom-plan node.
	 */
	pathnode = makeNode(CustomPath);
	pathnode->path.pathtype = T_CustomScan;
	pathnode->path.parent = baserel;
	pathnode->path.param_info
		= get_baserel_parampathinfo(root, baserel, baserel->lateral_relids);
	pathnode->path.pathkeys = NIL;	/* unsorted result */
	pathnode->flags = 0;
	pathnode->custom_private = NIL;	/* we don't use private field */
	pathnode->methods = &gpuscan_path_methods;

	/* cost estimation */
	cost_gpuscan(pathnode, root, baserel,
				 pathnode->path.param_info,
				 host_quals, dev_quals, false);

	add_path(baserel, &pathnode->path);
}

/*
 * gpuscan_try_replace_relscan
 *
 * It tries to replace the supplied SeqScan plan by GpuScan, if it is
 * enough simple. Even though seq_path didn't involve any qualifiers,
 * it makes sense if parent path is managed by PG-Strom because of bulk-
 * loading functionality.
 */
Plan *
gpuscan_try_replace_relscan(Plan *plan,
							List *range_table,
							Bitmapset *attr_refs,
							List **p_upper_quals)
{
	CustomScan	   *cscan;
	GpuScanInfo		gs_info;
	RangeTblEntry  *rte;
	Relation		rel;
	Index			scanrelid;
	ListCell	   *lc;
	BlockNumber		num_pages;
	double			num_tuples;
	double			allvisfrac;
	double			spc_seq_page_cost;
	Oid				tablespace_oid;

	if (!enable_gpuscan)
		return NULL;

	if (!IsA(plan, SeqScan) && !pgstrom_plan_is_gpuscan(plan))
		return NULL;

	scanrelid = ((Scan *) plan)->scanrelid;
	rte = rt_fetch(scanrelid, range_table);
	if (rte->rtekind != RTE_RELATION)
		return NULL;	/* usually, shouldn't happen */
	if (rte->relkind != RELKIND_RELATION &&
		rte->relkind != RELKIND_TOASTVALUE &&
		rte->relkind != RELKIND_MATVIEW)
		return NULL;	/* usually, shouldn't happen */

	/*
	 * All the referenced target-entry must be constructable on the
	 * device side, even if not a simple var reference.
	 */
	foreach (lc, plan->targetlist)
	{
		TargetEntry *tle = lfirst(lc);
		int		x = tle->resno - FirstLowInvalidHeapAttributeNumber;

		if (bms_is_member(x, attr_refs))
		{
			if (!pgstrom_codegen_available_expression(tle->expr))
				return NULL;
		}
	}

	/*
     * Check whether the plan qualifiers can be executable on device
     */
	if (IsA(plan, SeqScan))
	{
		if (!pgstrom_codegen_available_expression((Expr *)plan->qual))
			return NULL;
		*p_upper_quals = copyObject(plan->qual);
	}
	else if (pgstrom_plan_is_gpuscan(plan))
	{
		GpuScanInfo	   *temp;

		/* unable run bulk-loading with host qualifiers */
		if (plan->qual != NIL)
			return NULL;

		temp = deform_gpuscan_info(plan);
		if (temp->dev_quals == NIL)
		{
			*p_upper_quals = NIL;
			return plan;	/* available to use bulk-loading as-is */
		}
		*p_upper_quals = copyObject(temp->dev_quals);
	}
	else
		elog(ERROR, "Bug? unexpected plan node: %s", nodeToString(plan));

	/*
	 * OK, it was SeqScan with all device executable (or no) qualifiers,
	 * or, GpuScan without host qualifiers.
	 */
	cscan = makeNode(CustomScan);
    cscan->scan.plan.plan_width = plan->plan_width;
    cscan->scan.plan.targetlist = copyObject(plan->targetlist);
    cscan->scan.plan.qual = NIL;
    cscan->scan.scanrelid = scanrelid;
    cscan->flags = 0;
	memset(&gs_info, 0, sizeof(GpuScanInfo));
	form_gpuscan_info(cscan, &gs_info);
	cscan->methods = &gpuscan_plan_methods;

	/*
	 * Rebuild the cost estimation of the new plan. Overall logic is same
	 * with estimate_rel_size(), although integration with cost_gpuhashjoin()
	 * is more preferable for consistency....
	 */
	rel = heap_open(rte->relid, NoLock);
	estimate_rel_size(rel, NULL, &num_pages, &num_tuples, &allvisfrac);
	tablespace_oid = RelationGetForm(rel)->reltablespace;
	get_tablespace_page_costs(tablespace_oid, NULL, &spc_seq_page_cost);

	cscan->scan.plan.startup_cost = pgstrom_gpu_setup_cost;
	cscan->scan.plan.total_cost = cscan->scan.plan.startup_cost
		+ spc_seq_page_cost * num_pages;
	cscan->scan.plan.plan_rows = num_tuples;
    cscan->scan.plan.plan_width = plan->plan_width;

	heap_close(rel, NoLock);

	return &cscan->scan.plan;
}

/*
 * OpenCL code generation that can run on GPU/MIC device
 */
static char *
gpuscan_codegen_quals(PlannerInfo *root,
					  List *dev_quals, codegen_context *context)
{
	StringInfoData	str;
	StringInfoData	decl;
	char		   *expr_code;

	pgstrom_init_codegen_context(context);
	if (dev_quals == NIL)
		return NULL;

	/* OK, let's walk on the device expression tree */
	expr_code = pgstrom_codegen_expression((Node *)dev_quals, context);
	Assert(expr_code != NULL);

	initStringInfo(&decl);
	initStringInfo(&str);

	/*
	 * make declarations of var and param references
	 */
	appendStringInfo(&str, "%s\n", pgstrom_codegen_func_declarations(context));
	appendStringInfo(&decl, "%s%s\n",
					 pgstrom_codegen_param_declarations(context),
					 pgstrom_codegen_var_declarations(context));

	/* qualifier definition with row-store */
	appendStringInfo(
		&str,
		"static pg_bool_t\n"
		"gpuscan_qual_eval(__private cl_int *errcode,\n"
		"                  __global kern_parambuf *kparams,\n"
		"                  __global kern_data_store *kds,\n"
		"                  __global kern_data_store *ktoast,\n"
		"                  size_t kds_index)\n"
		"{\n"
		"%s"
		"  return %s;\n"
		"}\n", decl.data, expr_code);
	return str.data;
}

static Plan *
create_gpuscan_plan(PlannerInfo *root,
					RelOptInfo *rel,
					CustomPath *best_path,
					List *tlist,
					List *clauses)
{
	CustomScan	   *cscan;
	GpuScanInfo		gs_info;
	List		   *host_quals = NIL;
	List		   *dev_quals = NIL;
	ListCell	   *cell;
	char		   *kern_source;
	codegen_context	context;

	/* It should be a base relation */
	Assert(rel->relid > 0);
	Assert(rel->rtekind == RTE_RELATION);

	/*
	 * Distribution of clauses into device executable and others.
	 *
	 * NOTE: Why we don't sort out on Path construction stage is,
	 * create_scan_plan() may add parameterized scan clause, thus
	 * we have to delay the final decision until this point.
	 */
	foreach (cell, clauses)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (!pgstrom_codegen_available_expression(rinfo->clause))
			host_quals = lappend(host_quals, rinfo);
		else
			dev_quals = lappend(dev_quals, rinfo);
	}
	/* Reduce RestrictInfo list to bare expressions; ignore pseudoconstants */
	host_quals = extract_actual_clauses(host_quals, false);
    dev_quals = extract_actual_clauses(dev_quals, false);

	/*
	 * Construct OpenCL kernel code
	 */
	kern_source = gpuscan_codegen_quals(root, dev_quals, &context);

	/*
	 * Construction of GpuScanPlan node; on top of CustomPlan node
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = host_quals;
	cscan->scan.plan.lefttree = NULL;
	cscan->scan.plan.righttree = NULL;
	cscan->scan.scanrelid = rel->relid;

	gs_info.kern_source = kern_source;
	gs_info.extra_flags = context.extra_flags |
		DEVKERNEL_NEEDS_GPUSCAN |
		(!devprog_enable_optimize ? DEVKERNEL_DISABLE_OPTIMIZE : 0);
	gs_info.used_params = context.used_params;
	gs_info.used_vars = context.used_vars;
	gs_info.dev_quals = dev_quals;
	form_gpuscan_info(cscan, &gs_info);
	cscan->methods = &gpuscan_plan_methods;

	return &cscan->scan.plan;
}

/*
 * pgstrom_path_is_gpuscan
 *
 * It returns true, if supplied path node is gpuscan.
 */
bool
pgstrom_path_is_gpuscan(const Path *path)
{
	if (IsA(path, CustomPath) &&
		path->pathtype == T_CustomScan &&
		((CustomPath *) path)->methods == &gpuscan_path_methods)
		return true;
	return false;
}

/*
 * pgstrom_plan_is_gpuscan
 *
 * It returns true, if supplied plan node is gpuscan.
 */
bool
pgstrom_plan_is_gpuscan(const Plan *plan)
{
	CustomScan	   *cscan = (CustomScan *) plan;

	if (IsA(cscan, CustomScan) &&
		cscan->methods == &gpuscan_plan_methods)
		return true;
	return false;
}

/*
 * pgstrom_gpuscan_setup_bulkslot
 *
 * It setup tuple-slot for bulk-loading and projection-info to transform
 * the tuple into expected form.
 * (Once CustomPlan become CustomScan, no need to be a API)
 */
void
pgstrom_gpuscan_setup_bulkslot(PlanState *outer_ps,
							   ProjectionInfo **p_bulk_proj,
							   TupleTableSlot **p_bulk_slot)
{
	GpuScanState   *gss = (GpuScanState *) outer_ps;

	if (!IsA(gss, CustomScanState) ||
		gss->css.methods != &gpuscan_exec_methods.c)
		elog(ERROR, "Bug? PlanState node is not GpuScanState");

	*p_bulk_proj = gss->css.ss.ps.ps_ProjInfo;
	*p_bulk_slot = gss->css.ss.ss_ScanTupleSlot;
}

/*
 * gpuscan_create_scan_state
 *
 * allocation of GpuScanState, rather than CustomScanState
 */
static Node *
gpuscan_create_scan_state(CustomScan *cscan)
{
	GpuScanState   *gss = palloc0(sizeof(GpuScanState));

	NodeSetTag(gss, T_CustomScanState);
	gss->css.methods = &gpuscan_exec_methods.c;

	return (Node *)gss;
}

static void
gpuscan_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuContext	   *gcontext = pgstrom_get_gpucontext();
	Relation		scan_rel = node->ss.ss_currentRelation;
	GpuScanState   *gss = (GpuScanState *) node;
	GpuScanInfo	   *gs_info = deform_gpuscan_info(node->ss.ps.plan);
	CUresult		rc;

	BlockNumber		relpages;
	double			reltuples;
	double			allvisfrac;

	/* gpuscan should not have inner/outer plan right now */
	Assert(outerPlan(node) == NULL);
	Assert(innerPlan(node) == NULL);

	/*
	 * Setup GpuTaskState
	 */
	gss->gts.gcontext = gcontext;
	if (gs_info->kern_source)
	{
		if (!pgstrom_get_cuda_program(&gss->gts,
									  gs_info->kern_source,
									  gs_info->extra_flags))
			elog(INFO, "no built binary, so kick nvcc");
	}
	SpinLockInit(&gss->gts.lock);
	dlist_init(&gss->gts.running_tasks);
	dlist_init(&gss->gts.pending_tasks);
	dlist_init(&gss->gts.completed_tasks);
	gss->gts.cb_cleanup = gpuscan_end_cleanup;

	/* initialize the start/end position */
	gss->curr_blknum = 0;
	gss->last_blknum = RelationGetNumberOfBlocks(scan_rel);
	/* initialize device qualifiers also, for fallback */
	gss->dev_quals = (List *)
		ExecInitExpr((Expr *) gsinfo->dev_quals, &gss->css.ss.ps);
	/* 'tableoid' should not change during relation scan */
	gss->scan_tuple.t_tableOid = RelationGetRelid(scan_rel);

	/* save the kernel source/extra params */
	gss->kern_source = gsinfo->kern_source;
	gss->extra_params = gsinfo->extra_params;
	/* kernel constant parameter buffer */
	gss->kparams = pgstrom_create_kern_parambuf(gs_info->used_params,
												gss->css.ss.ps.ps_ExprContext);
	/* other run-time parameters */
	gss->curr_chunk = NULL;
	gss->curr_index = 0;
    gss->num_rechecked = 0;

	/* Is perfmon needed? */
	gss->pfm.enabled = pgstrom_perfmon_enabled;
}

/*
 * pgstrom_release_gpuscan
 *
 * Callback handler when reference counter of pgstrom_gpuscan object
 * reached to zero, due to pgstrom_put_message.
 * It also unlinks associated device program and release row-store.
 * Note that this callback shall never be invoked under the OpenCL
 * server context, because some resources (like shared-buffer) are
 * assumed to be released by the backend process.
 */
static void
pgstrom_release_gpuscan(GpuTask *gputask)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *) gputask;
	CUresult			rc;

	if (gpuscan->pds)
		pgstrom_release_data_store(gpuscan->pds);

	if (gputask->stream)
	{
		rc = cuStreamDestroy(gputask->stream);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuStreamDestroy: %s", cuda_strerror(rc));
	}

	if (gpuscan->m_gpuscan)
	{
		rc = cuMemFree(gpuscan->m_gpuscan);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemFree: %s", cuda_strerror(rc));
	}

	if (gpuscan->m_kds)
	{
		rc = cuMemFree(gpuscan->m_kds);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemFree: %s", cuda_strerror(rc));
	}

	if (gpuscan->ev_dma_send_start)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", cuda_strerror(rc));
	}

	if (gpuscan->ev_dma_send_stop)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", cuda_strerror(rc));
	}

	if (gpuscan->ev_dma_recv_start)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", cuda_strerror(rc));
	}

	if (gpuscan->ev_dma_recv_stop)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", cuda_strerror(rc));
	}
	pfree(gpuscan);
}

static pgstrom_gpuscan *
pgstrom_create_gpuscan(GpuScanState *gss, pgstrom_data_store *pds)
{
	GpuContext		   *gcontext = gss->gts.gcontext;
	pgstrom_gpuscan    *gpuscan;
	kern_resultbuf	   *kresults;
	kern_data_store	   *kds = pds->kds;
	cl_uint				nitems = kds->nitems;
	Size				length;

	length = (STROMALIGN(offsetof(pgstrom_gpuscan, kern.kparams)) +
			  STROMALIGN(gss->kparams->length));
	if (!gss->kern_source)
		length += STROMALIGN(offsetof(kern_resultbuf, results[0]));
	else
		length += STROMALIGN(offsetof(kern_resultbuf, results[nitems]));

	gpuscan = MemoryContextAllocZero(gcontext->memcxt, length);
	memset(gpuscan, 0, sizeof(pgstrom_gpuscan));

	/* setting up */
	gpuscan->task.gts = &gss->gts;
	pgstrom_assign_stream(&gpuscan->task);
	gpuscan->task.errcode = 0;
	gpuscan->task.cb_process = pgstrom_process_gpuscan;
	gpuscan->task.cb_release = pgstrom_release_gpuscan;
	gpuscan->pds = pds;
	/* copy kern_parambuf */
	Assert(gss->kparams->length == STROMALIGN(gss->kparams->length));
	memcpy(KERN_GPUSCAN_PARAMBUF(&gpuscan->kern),
		   gss->kparams,
		   gss->kparams->length);
	/* setting up resultbuf */
	kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = 1;
	kresults->nrooms = nitems;

	/* If GpuScan does not kick GPU Kernek execution, we treat
	 * this chunk as all-visible one, without reference to
	 * results[] row-index.
	 */
	if (!gss->kern_source)
	{
		kresults->all_visible = true;
        kresults->nitems = nitems;
	}
	return gpuscan;
}

static pgstrom_gpuscan *
pgstrom_load_gpuscan(GpuScanState *gss)
{
	pgstrom_gpuscan	   *gpuscan = NULL;
	Relation			rel = gss->css.ss.ss_currentRelation;
	TupleDesc			tupdesc = RelationGetDescr(rel);
	Snapshot			snapshot = gss->css.ss.ps.state->es_snapshot;
	bool				end_of_scan = false;
	pgstrom_data_store *pds;
	struct timeval tv1, tv2;

	/* no more blocks to read */
	if (gss->curr_blknum > gss->last_blknum)
		return NULL;

	if (gss->pfm.enabled)
		gettimeofday(&tv1, NULL);

	while (!gpuscan && !end_of_scan)
	{
		pds = pgstrom_create_data_store_row(gss->gts.gcontext,
											tupdesc,
											pgstrom_chunk_size << 20,
											NULL);
		/* fill up this data-store */
		while (gss->curr_blknum < gss->last_blknum &&
			   pgstrom_data_store_insert_block(pds, rel,
											   gss->curr_blknum,
											   snapshot, true) >= 0)
			gss->curr_blknum++;

		if (pds->kds->nitems > 0)
			gpuscan = pgstrom_create_gpuscan(gss, pds);
		else
		{
			pgstrom_release_data_store(pds);

			/* NOTE: In case when it scans on a large hole (that is
			 * continuous blocks contain invisible tuples only; may
			 * be created by DELETE with relaxed condition),
			 * pgstrom_data_store_insert_block() may return negative
			 * value without valid tuples, even though we don't reach
			 * either end of relation or chunk.
			 * So, we need to check whether we actually touched on
			 * the end-of-relation. If not, retry scanning.
			 *
			 * XXX - Is the above behavior still right?
			 */
			if (gss->curr_blknum >= gss->last_blknum)
				end_of_scan = true;
		}
	}

	/* update perfmon statistics */
	if (gss->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gss->pfm.time_outer_load += timeval_diff(&tv1, &tv2);
	}
	return gpuscan;
}

static TupleTableSlot *
gpuscan_next_tuple(GpuScanState *gss)
{
	pgstrom_gpuscan	   *gpuscan = gss->curr_chunk;
	kern_resultbuf	   *kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	TupleTableSlot	   *slot = NULL;
	cl_int				i_result;
	bool				do_recheck = false;
	struct timeval		tv1, tv2;

	if (!gpuscan)
		return false;

	if (gss->pfm.enabled)
		gettimeofday(&tv1, NULL);

   	while (gss->curr_index < kresults->nitems)
	{
		pgstrom_data_store *pds = gpuscan->pds;

		if (kresults->all_visible)
			i_result = ++gss->curr_index;
		else
		{
			i_result = kresults->results[gss->curr_index++];
			if (i_result < 0)
			{
				i_result = -i_result;
				do_recheck = true;
				gss->num_rechecked++;
			}
		}
		Assert(i_result > 0);

		slot = gss->css.ss.ss_ScanTupleSlot;
		if (!pgstrom_fetch_data_store(slot, pds, i_result - 1,
									  &gss->scan_tuple))
			elog(ERROR, "failed to fetch a record from pds: %d", i_result);
		Assert(slot->tts_tuple == &gss->scan_tuple);

		if (do_recheck)
		{
			ExprContext *econtext = gss->css.ss.ps.ps_ExprContext;

			Assert(gss->dev_quals != NULL);
			econtext->ecxt_scantuple = slot;
			if (!ExecQual(gss->dev_quals, econtext, false))
			{
				slot = NULL;
				continue;
			}
		}
		break;
	}
	if (gss->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gss->pfm.time_materialize += timeval_diff(&tv1, &tv2);
	}
	return slot;
}

/*
 * pgstrom_fetch_gpuscan
 *
 * It loads a chunk from the target relation, then enqueue the GpuScan
 * chunk to be processed by OpenCL devices if valid device kernel was
 * constructed. Elsewhere, it works as a wrapper of pgstrom_load_gpuscan,
 */
static pgstrom_gpuscan *
pgstrom_fetch_gpuscan(GpuScanState *gss)
{
	pgstrom_message	   *msg;
	pgstrom_gpuscan	   *gpuscan;

	/*
	 * In case when no device code will be executed, we don't need to have
	 * asynchronous execution. So, just return a chunk with synchronous
	 * manner.
	 */
	if (!gss->kern_source)
		return pgstrom_load_gpuscan(gss);

	/* A valid device code shall have message queue */
	Assert(gss->mqueue != NULL);

	/*
	 * Try to keep number of gpuscan chunks being asynchronously executed
	 * larger than minimum multiplicity, unless it does not exceed
	 * maximum one and OpenCL server does not return a new response.
	 */
	while (gss->num_running <= pgstrom_max_async_chunks)
	{
		pgstrom_gpuscan	*gpuscan = pgstrom_load_gpuscan(gss);

		if (!gpuscan)
			break;	/* scan reached end of the relation */

		if (!pgstrom_enqueue_message(&gpuscan->msg))
		{
			pgstrom_put_message(&gpuscan->msg);
			elog(ERROR, "failed to enqueue pgstrom_gpuscan message");
		}
		gss->num_running++;

		if (gss->num_running > pgstrom_min_async_chunks &&
			(msg = pgstrom_try_dequeue_message(gss->mqueue)) != NULL)
		{
			gss->num_running--;
			dlist_push_tail(&gss->ready_chunks, &msg->chain);
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
			return NULL;

		/* Synchronization, if needed */
		msg = pgstrom_dequeue_message(gss->mqueue);
		if (!msg)
			elog(ERROR, "message queue wait timeout");
		gss->num_running--;
		dlist_push_tail(&gss->ready_chunks, &msg->chain);
	}

	/*
	 * Picks up next available chunks if any
	 */
	Assert(!dlist_is_empty(&gss->ready_chunks));
	gpuscan = dlist_container(pgstrom_gpuscan, msg.chain,
							  dlist_pop_head_node(&gss->ready_chunks));
	Assert(StromTagIs(gpuscan, GpuScan));

	/*
	 * Raise an error, if any error was reported
	 */
	if (gpuscan->msg.errcode != StromError_Success)
	{
		if (gpuscan->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
		{
			const char *buildlog
				= pgstrom_get_devprog_errmsg(gpuscan->dprog_key);

			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
							pgstrom_strerror(gpuscan->msg.errcode),
							gss->kern_source),
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
	return gpuscan;
}

static TupleTableSlot *
gpuscan_fetch_tuple(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *) node;
	TupleTableSlot *slot = gss->css.ss.ss_ScanTupleSlot;

	ExecClearTuple(slot);

	while (!gss->curr_chunk || !(slot = gpuscan_next_tuple(gss)))
	{
		pgstrom_gpuscan	   *gpuscan = gss->curr_chunk;

		/*
		 * Release the current gpuscan chunk being already scanned
		 */
		if (gpuscan)
		{
			dlist_delete(&gpuscan->chain);
			if (gpuscan->task.pfm.enabled)
				pgstrom_perfmon_add(&gss->gts.pfm, &gpuscan->task.pfm);
			pgstrom_release_gpuscan(gpuscan);
			gss->curr_chunk = NULL;
            gss->curr_index = 0;
		}
		gpuscan = pgstrom_fetch_gpuscan(gss);
		if (!gpuscan)
			break;
		gss->curr_chunk = gpuscan;
		gss->curr_index = 0;
	}
	return slot;
}

static bool
gpuscan_recheck(CustomScanState *node, TupleTableSlot *slot)
{
	/* There are no access-method-specific conditions to recheck. */
	return true;
}

static TupleTableSlot *
gpuscan_exec(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) gpuscan_fetch_tuple,
					(ExecScanRecheckMtd) gpuscan_recheck);
}

static void *
gpuscan_exec_bulk(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *) node;
	List			   *host_qual = node->ss.ps.qual;
	pgstrom_gpuscan	   *gpuscan;
	kern_resultbuf	   *kresults;
	pgstrom_bulkslot   *bulk;
	cl_uint				i, j, nitems;
	cl_int			   *rindex;
	HeapTupleData		tuple;

	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStartNode(node->ss.ps.instrument);

	while (true)
	{
		gpuscan = pgstrom_fetch_gpuscan(gss);
		if (!gpuscan)
		{
			if (node->ss.ps.instrument)
				InstrStopNode(node->ss.ps.instrument, (double) 0.0);
			return NULL;
		}

		/* update perfmon info */
		if (gpuscan->msg.pfm.enabled)
			pgstrom_perfmon_add(&gss->pfm, &gpuscan->msg.pfm);

		/*
		 * Make a bulk-slot according to the result
		 */
		kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
		bulk = palloc0(offsetof(pgstrom_bulkslot, rindex[kresults->nitems]));
		bulk->pds = pgstrom_get_data_store(gpuscan->pds);
		bulk->nvalids = 0;	/* to be set later */
		pgstrom_track_object(&bulk->pds->sobj, 0);

		/* No longer gpuscan is referenced any more. The associated
		 * data-store is not actually released because its refcnt is
		 * already incremented above.
		 */
		pgstrom_untrack_object(&gpuscan->msg.sobj);
		pgstrom_put_message(&gpuscan->msg);

		/*
		 * If any, it may take host side checks
		 * - Recheck of device qualifier, if result is nagative
		 * - Host qualifier checks, if any.
		 */
		if (kresults->all_visible)
		{
			if (!host_qual)
			{
				bulk->nvalids = -1;	/* all the rows are valid */
				break;
			}
			nitems = bulk->pds->kds->nitems;
			rindex = NULL;
			Assert(nitems <= kresults->nitems);
		}
		else
		{
			nitems = kresults->nitems;
			rindex = kresults->results;
		}

		for (i=0, j=0; i < nitems; i++)
		{
			cl_uint		row_index = (!rindex ? i + 1 : rindex[i]);
			bool		do_recheck = false;

			Assert(row_index != 0);
			if (row_index > 0)
				row_index--;
			else
			{
				row_index = -row_index - 1;
				do_recheck = true;
			}

			if (host_qual || do_recheck)
			{
				ExprContext	   *econtext = gss->css.ss.ps.ps_ExprContext;
				TupleTableSlot *slot = gss->css.ss.ss_ScanTupleSlot;

				if (!pgstrom_fetch_data_store(slot,
											  bulk->pds,
											  row_index,
											  &tuple))
					elog(ERROR, "Bug? invalid row-index was in the result");
				econtext->ecxt_scantuple = slot;

				/* Recheck of device qualifier, if needed */
				if (do_recheck)
				{
					Assert(gss->dev_quals != NULL);
					if (!ExecQual(gss->dev_quals, econtext, false))
						continue;
				}
				/* Check of host qualifier, if needed */
				if (host_qual)
				{
					if (!ExecQual(host_qual, econtext, false))
						continue;
				}
			}
			bulk->rindex[j++] = row_index;
		}
		bulk->nvalids = j;

		if (bulk->nvalids > 0)
			break;

		/* If this chunk has no valid items, it does not make sense to
		 * return upper level this chunk.
		 */
		pgstrom_untrack_object(&bulk->pds->sobj);
		pgstrom_put_data_store(bulk->pds);
		pfree(bulk);
	}
	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStopNode(node->ss.ps.instrument,
					  bulk->nvalids < 0 ?
					  (double) bulk->pds->kds->nitems :
					  (double) bulk->nvalids);
	return bulk;
}

static void
gpuscan_end(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *)node;
	pgstrom_gpuscan	   *gpuscan;

	if (gss->curr_chunk)
	{
		gpuscan = gss->curr_chunk;

		pgstrom_untrack_object(&gpuscan->msg.sobj);
		pgstrom_put_message(&gpuscan->msg);
		gss->curr_chunk = NULL;
	}

	while (gss->num_running > 0)
	{
		gpuscan = (pgstrom_gpuscan *)pgstrom_dequeue_message(gss->mqueue);
		if (!gpuscan)
			elog(ERROR, "message queue wait timeout");
		pgstrom_untrack_object(&gpuscan->msg.sobj);
		pgstrom_put_message(&gpuscan->msg);
		gss->num_running--;
	}

	if (gss->num_rechecked > 0)
		elog(NOTICE, "GpuScan: %u records were re-checked by CPU",
			 gss->num_rechecked);

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
}

static void
gpuscan_rescan(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *) node;
	pgstrom_message	   *msg;
	pgstrom_gpuscan	   *gpuscan;
	dlist_mutable_iter	iter;

	/*
	 * If asynchronous requests are still running, we need to synchronize
	 * their completion first.
	 *
	 * TODO: if we track all the pgstrom_gpuscan objects being enqueued
	 * locally, all we need to do should be just decrements reference
	 * counter. It may be a future enhancement.
	 */
	gpuscan = gss->curr_chunk;
	if (gpuscan)
	{
		pgstrom_untrack_object(&gpuscan->msg.sobj);
		pgstrom_put_message(&gpuscan->msg);
		gss->curr_chunk = NULL;
		gss->curr_index = 0;
	}

	while (gss->num_running > 0)
	{
		msg = pgstrom_dequeue_message(gss->mqueue);
		if (!msg)
			elog(ERROR, "message queue wait timeout");
		gss->num_running--;
		dlist_push_tail(&gss->ready_chunks, &msg->chain);
	}

	dlist_foreach_modify (iter, &gss->ready_chunks)
	{
		gpuscan = dlist_container(pgstrom_gpuscan, msg.chain, iter.cur);
		Assert(StromTagIs(gpuscan, GpuScan));

		dlist_delete(&gpuscan->msg.chain);
		if (gpuscan->msg.errcode != StromError_Success)
		{
			if (gpuscan->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
			{
				const char *buildlog
					= pgstrom_get_devprog_errmsg(gpuscan->dprog_key);

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
								pgstrom_strerror(gpuscan->msg.errcode),
								gss->kern_source),
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
		pgstrom_untrack_object(&gpuscan->msg.sobj);
		pgstrom_put_message(&gpuscan->msg);
	}

	/*
	 * OK, asynchronous jobs were cleared. revert scan state to the head.
	 */
	gss->curr_blknum = 0;
}

static void
gpuscan_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *) node;
	GpuScanInfo	   *gsinfo = deform_gpuscan_info(gss->css.ss.ps.plan);

	if (gsinfo->dev_quals != NIL)
	{
		show_scan_qual(gsinfo->dev_quals, "Device Filter",
					   &gss->css.ss.ps, ancestors, es);
		show_instrumentation_count("Rows Removed by Device Fileter",
								   2, &gss->css.ss.ps, es);
	}
	show_device_kernel(gss->dprog_key, es);

	if (es->analyze && gss->pfm.enabled)
		pgstrom_perfmon_explain(&gss->pfm, es);
}

void
pgstrom_init_gpuscan(void)
{
	/* enable_gpuscan */
	DefineCustomBoolVariable("enable_gpuscan",
							 "Enables the use of GPU accelerated full-scan",
							 NULL,
							 &enable_gpuscan,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* setup path methods */
	gpuscan_path_methods.CustomName			= "GpuScan";
	gpuscan_path_methods.PlanCustomPath		= create_gpuscan_plan;
	gpuscan_path_methods.TextOutCustomPath	= NULL;

	/* setup plan methods */
	gpuscan_plan_methods.CustomName			= "GpuScan";
	gpuscan_plan_methods.CreateCustomScanState = gpuscan_create_scan_state;
	gpuscan_plan_methods.TextOutCustomScan	= NULL;

	/* setup exec methods */
	gpuscan_exec_methods.c.CustomName         = "GpuScan";
	gpuscan_exec_methods.c.BeginCustomScan    = gpuscan_begin;
	gpuscan_exec_methods.c.ExecCustomScan     = gpuscan_exec;
	gpuscan_exec_methods.c.EndCustomScan      = gpuscan_end;
	gpuscan_exec_methods.c.ReScanCustomScan   = gpuscan_rescan;
	gpuscan_exec_methods.c.MarkPosCustomScan  = NULL;
	gpuscan_exec_methods.c.RestrPosCustomScan = NULL;
	gpuscan_exec_methods.c.ExplainCustomScan  = gpuscan_explain;
	gpuscan_exec_methods.ExecCustomBulk       = gpuscan_exec_bulk;

	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = gpuscan_add_scan_path;
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
	cl_mem			m_dstore;
	cl_mem			m_ktoast;
	cl_uint			ev_index;
	cl_event		events[20];
} clstate_gpuscan;

static void
clserv_respond_gpuscan(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpuscan	   *clgss = private;
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *)clgss->msg;
	kern_resultbuf	   *kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	cl_int				status;
	cl_int				rc;

	/* put error code */
	if (ev_status != CL_COMPLETE)
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpuscan->msg.errcode = StromError_OpenCLInternal;
	}
	else
	{
		gpuscan->msg.errcode = kresults->errcode;
	}

	/*
	 * collect performance statistics
	 */
	if (gpuscan->msg.pfm.enabled)
	{
		cl_ulong    dma_send_begin;
		cl_ulong    dma_send_end;
		cl_ulong    kern_exec_begin;
		cl_ulong    kern_exec_end;
		cl_ulong    dma_recv_begin;
		cl_ulong    dma_recv_end;
		cl_ulong	temp;
		cl_int		i, n;

		n = clgss->ev_index - 2;
		dma_send_begin = (cl_ulong)(~0);
		dma_send_end = (cl_ulong) 0;
		for (i=0; i < n; i++)
		{
			rc = clGetEventProfilingInfo(clgss->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			if (i==0 || dma_send_begin > temp)
				dma_send_begin = temp;

			rc = clGetEventProfilingInfo(clgss->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			if (i==0 || dma_send_end < temp)
				dma_send_end = temp;
		}

		rc = clGetEventProfilingInfo(clgss->events[clgss->ev_index - 2],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &kern_exec_begin,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgss->events[clgss->ev_index - 2],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &kern_exec_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgss->events[clgss->ev_index - 1],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &dma_recv_begin,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgss->events[clgss->ev_index - 1],
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
			gpuscan->msg.pfm.enabled = false;
		}
	}

	rc = clGetEventInfo(clgss->events[clgss->ev_index - 2],
						CL_EVENT_COMMAND_EXECUTION_STATUS,
						sizeof(cl_int),
						&status,
						NULL);
	Assert(rc == CL_SUCCESS);

	/* release opencl objects */
	while (clgss->ev_index > 0)
		clReleaseEvent(clgss->events[--clgss->ev_index]);
	if (clgss->m_ktoast)
		clReleaseMemObject(clgss->m_ktoast);
	clReleaseMemObject(clgss->m_dstore);
	clReleaseMemObject(clgss->m_gpuscan);
	clReleaseKernel(clgss->kernel);
	clReleaseProgram(clgss->program);
	free(clgss);

	/* respond to the backend side */
	pgstrom_reply_message(&gpuscan->msg);
}

/*
 * clserv_process_gpuscan
 *
 * entrypoint of kernel gpuscan implementation
 */
static void
clserv_process_gpuscan(pgstrom_message *msg)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *) msg;
	pgstrom_perfmon	   *pfm = &gpuscan->msg.pfm;
	pgstrom_data_store *pds = gpuscan->pds;
	kern_data_store	   *kds = pds->kds;
	clstate_gpuscan	   *clgss = NULL;
	cl_program			program;
	cl_command_queue	kcmdq;
	kern_resultbuf	   *kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	int					dindex;
	cl_int				rc;
	size_t				length;
	size_t				offset;
	size_t				gwork_sz;
	size_t				lwork_sz;

	/* sanity checks */
	Assert(StromTagIs(gpuscan, GpuScan));
	Assert(kds->format == KDS_FORMAT_ROW);
	Assert(kresults->nrels == 1);
	if (kds->nitems == 0)
	{
		msg->errcode = StromError_BadRequestMessage;
		pgstrom_reply_message(msg);
		return;
	}

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
	program = clserv_lookup_device_program(gpuscan->dprog_key,
										   &gpuscan->msg);
	if (!program)
		return;		/* message is in waitq, retry it! */
	if (program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error;
	}

	/*
	 * create a state object
	 */
	clgss = calloc(1, offsetof(clstate_gpuscan, events[20 + kds->nblocks]));
	if (!clgss)
	{
		clReleaseProgram(program);
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error;
	}
	clgss->msg = msg;
	clgss->program = program;

	/*
	 * lookup kernel function for gpuscan
	 */
	clgss->kernel = clCreateKernel(clgss->program, "gpuscan_qual", &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * choose a device to execute this kernel, and compute an optimal
	 * workgroup-size of this kernel
	 */
	dindex = pgstrom_opencl_device_schedule(&gpuscan->msg);
	kcmdq = opencl_cmdq[dindex];	
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clgss->kernel, dindex,
									   false,	/* smaller WG-sz is better */
									   kds->nitems, sizeof(cl_uint)))
		goto error;

	/* allocation of device memory for kern_gpuscan argument */
	clgss->m_gpuscan = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  KERN_GPUSCAN_LENGTH(&gpuscan->kern),
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/* allocation of device memory for kern_data_store argument */
	clgss->m_dstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 KERN_DATA_STORE_LENGTH(kds),
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * allocation of device memory for toast buffer, but never required
	 */
	clgss->m_ktoast = NULL;

	/*
	 * OK, all the device memory and kernel objects acquired.
	 * Let's prepare kernel invocation.
	 *
	 * The kernel call is:
	 * pg_bool_t
	 * gpuscan_qual_eval(__global kern_gpuscan *kgpuscan,
	 *                   __global kern_data_store *kds,
	 *                   __global kern_data_store *ktoast,
	 *                   __local void *local_workbuf)
	 */
	rc = clSetKernelArg(clgss->kernel,
						0,		/* kern_gpuscan */
						sizeof(cl_mem),
						&clgss->m_gpuscan);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clgss->kernel,
						1,		/* kern_data_store */
						sizeof(cl_mem),
						&clgss->m_dstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clgss->kernel,
						2,		/* kds of toast buffer; always NULL */
						sizeof(cl_mem),
						&clgss->m_ktoast);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clgss->kernel,
						3,		/* local_workmem */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	/*
     * OK, enqueue DMA transfer, kernel execution, then DMA writeback.
     *
	 * (1) gpuscan (incl. kparams) - DMA send
	 * (2) kern_data_store - DMA send
	 * (3) execution of kernel function
	 * (4) write back vrelation - DMA recv
	 */

	/* kern_gpuscan */
	offset = KERN_GPUSCAN_DMASEND_OFFSET(&gpuscan->kern);
	length = KERN_GPUSCAN_DMASEND_LENGTH(&gpuscan->kern);
	rc = clEnqueueWriteBuffer(kcmdq,
							  clgss->m_gpuscan,
							  CL_FALSE,
							  offset,
							  length,
							  &gpuscan->kern,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	clgss->ev_index++;
	pfm->bytes_dma_send += length;
	pfm->num_dma_send++;

	/* kern_data_store, via common routine */
	rc = clserv_dmasend_data_store(pds,
								   kcmdq,
								   clgss->m_dstore,
								   clgss->m_ktoast,
								   0,
								   NULL,
								   &clgss->ev_index,
								   clgss->events,
								   pfm);
	if (rc != CL_SUCCESS)
		goto error;

	/* execution of kernel function */
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
		goto error;
	}
	clgss->ev_index++;
	pfm->num_kern_exec++;

	/* write back result vrelation */
	offset = KERN_GPUSCAN_DMARECV_OFFSET(&gpuscan->kern);
	length = KERN_GPUSCAN_DMARECV_LENGTH(&gpuscan->kern);
	rc = clEnqueueReadBuffer(kcmdq,
							 clgss->m_gpuscan,
							 CL_FALSE,
							 offset,
							 length,
							 kresults,
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
	pfm->bytes_dma_recv += length;
	pfm->num_dma_recv++;

	/*
	 * Last, registers a callback routine that replies the message
	 * to the backend
	 */
	rc = clSetEventCallback(clgss->events[clgss->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpuscan,
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
			clWaitForEvents(clgss->ev_index, clgss->events);
		if (clgss->m_ktoast)
			clReleaseMemObject(clgss->m_ktoast);
		if (clgss->m_dstore)
			clReleaseMemObject(clgss->m_dstore);
		if (clgss->m_gpuscan)
			clReleaseMemObject(clgss->m_gpuscan);
		if (clgss->kernel)
			clReleaseKernel(clgss->kernel);
		if (clgss->program)
			clReleaseProgram(clgss->program);
		free(clgss);
	}
	gpuscan->msg.errcode = rc;
	pgstrom_reply_message(&gpuscan->msg);
}
