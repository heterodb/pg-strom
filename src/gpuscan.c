/*
 * gpuscan.c
 *
 * Sequential scan accelerated by GPU processors
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
#include "storage/latch.h"
#include "storage/proc.h"
#include "storage/procsignal.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/spccache.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
#include "pg_strom.h"
#include "cuda_gpuscan.h"

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

typedef struct
{
	GpuTask			task;
	dlist_node		chain;
	CUfunction		kern_qual;
	void		   *kern_qual_args[4];
	CUdeviceptr		m_gpuscan;
	CUdeviceptr		m_kds;
	CUevent 		ev_dma_send_start;
	CUevent			ev_dma_send_stop;	/* also, start kernel exec */
	CUevent			ev_dma_recv_start;	/* also, stop kernel exec */
	CUevent			ev_dma_recv_stop;
	pgstrom_data_store *pds;
	kern_gpuscan	kern;
} pgstrom_gpuscan;

typedef struct {
	CustomScanState	css;
	GpuTaskState	gts;

	BlockNumber		curr_blknum;
	BlockNumber		last_blknum;
	HeapTupleData	scan_tuple;
	List		   *dev_quals;

	kern_parambuf  *kparams;

	pgstrom_gpuscan *curr_chunk;
	uint32			curr_index;
	cl_uint			num_rechecked;
	cl_uint			num_running;
	pgstrom_perfmon	pfm;	/* sum of performance counter */
} GpuScanState;

/* forward declarations */
static bool pgstrom_process_gpuscan(GpuTask *task);

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

	/* Adjustment for param info */
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

	/* check bulk-load capability */
	if (host_quals == NIL)
	{
		bool		support_bulkload = true;

		foreach (cell, baserel->reltargetlist)
		{
			Expr	   *expr = lfirst(cell);

			if (!IsA(expr, Var) &&
				!pgstrom_codegen_available_expression(expr))
			{
				support_bulkload = false;
				break;
			}
		}
		if (support_bulkload)
			pathnode->flags |= CUSTOMPATH_SUPPORT_BULKLOAD;
	}
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
gpuscan_try_replace_seqscan(SeqScan *seqscan,
							List *range_tables,
							List **pullup_quals)
{
	CustomScan	   *cscan;
	GpuScanInfo		gs_info;
	RangeTblEntry  *rte;
	Relation		rel;
	ListCell	   *lc;
	BlockNumber		num_pages;
	double			num_tuples;
	double			allvisfrac;
	double			spc_seq_page_cost;
	Oid				tablespace_oid;

	if (!enable_gpuscan)
		return NULL;

	Assert(IsA(seqscan, SeqScan));
	rte = rt_fetch(seqscan->scanrelid, range_tables);
	if (rte->rtekind != RTE_RELATION)
		return NULL;	/* usually, shouldn't happen */
	if (rte->relkind != RELKIND_RELATION &&
		rte->relkind != RELKIND_TOASTVALUE &&
		rte->relkind != RELKIND_MATVIEW)
		return NULL;	/* usually, shouldn't happen */

	/*
	 * Target-entry must be a simle varnode or device executable
	 * expression because it shall be calculated on device-side
	 * if it takes projection
	 */
	foreach (lc, seqscan->plan.targetlist)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (!IsA(tle->expr, Var) &&
			!pgstrom_codegen_available_expression(tle->expr))
			return NULL;
	}

	/*
	 * Check whether the plan qualifiers is executable on device.
	 * Any host-only qualifier prevents bulk-loading.
	 */
	if (!pgstrom_codegen_available_expression((Expr *) seqscan->plan.qual))
		return NULL;
	*pullup_quals = copyObject(seqscan->plan.qual);

	/*
	 * OK, SeqScan with all device executable (or no) qualifiers, and
	 * no projection problems. So, GpuScan with bulk-load will ba a better
	 * choice than SeqScan.
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.plan_width = seqscan->plan.plan_width;
	cscan->scan.plan.targetlist = copyObject(seqscan->plan.targetlist);
	cscan->scan.plan.qual = NIL;
	cscan->scan.plan.extParam = bms_copy(seqscan->plan.extParam);
	cscan->scan.plan.allParam = bms_copy(seqscan->plan.allParam);
	cscan->scan.scanrelid = seqscan->scanrelid;
	cscan->flags = CUSTOMPATH_SUPPORT_BULKLOAD;
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
    cscan->scan.plan.plan_width = seqscan->plan.plan_width;

	heap_close(rel, NoLock);

	return &cscan->scan.plan;
}

/*
 * gpuscan_pullup_devquals - construct an equivalen GpuScan node, but
 * no device qualifiers which is pulled-up. In case of bulk-loading,
 * it is more reasonable to run device qualifier on upper node, than
 * individually.
 */
Plan *
gpuscan_pullup_devquals(Plan *plannode, List **pullup_quals)
{
	CustomScan	   *cscan_old = (CustomScan *) plannode;
	CustomScan	   *cscan_new;
	GpuScanInfo		*gs_info;

	Assert(pgstrom_plan_is_gpuscan(plannode));
	gs_info = deform_gpuscan_info(cscan_old);

	/* in case of nothing to be changed */
	if (gs_info->dev_quals == NIL)
	{
		Assert(gs_info->kern_source == NULL);
		*pullup_quals = NULL;
		return &cscan_old->scan.plan;
	}
	*pullup_quals = copyObject(gs_info->dev_quals);
	cscan_new = copyObject(cscan_old);
	memset(gs_info, 0, sizeof(GpuScanInfo));
	form_gpuscan_info(cscan_new, gs_info);

	return &cscan_new->scan.plan;
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
		"__device__ cl_bool\n"
		"gpuscan_qual_eval(cl_int *errcode,\n"
		"                  kern_parambuf *kparams,\n"
		"                  kern_data_store *kds,\n"
		"                  kern_data_store *ktoast,\n"
		"                  size_t kds_index)\n"
		"{\n"
		"%s"
		"  return EVAL(%s);\n"
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
	gs_info.extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUSCAN;
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
	GpuContext	   *gcontext = pgstrom_get_gpucontext();
	GpuScanState   *gss = MemoryContextAllocZero(gcontext->memcxt,
												 sizeof(GpuScanState));
	/* Set tag and executor callbacks */
	NodeSetTag(gss, T_CustomScanState);
	gss->css.methods = &gpuscan_exec_methods.c;
	/* GpuTaskState setup */
	pgstrom_init_gputaststate(gcontext, &gss->gts, NULL);

	return (Node *) gss;
}

static void
gpuscan_begin(CustomScanState *node, EState *estate, int eflags)
{
	Relation		scan_rel = node->ss.ss_currentRelation;
	GpuScanState   *gss = (GpuScanState *) node;
	GpuScanInfo	   *gs_info = deform_gpuscan_info(node->ss.ps.plan);

	/* gpuscan should not have inner/outer plan right now */
	Assert(outerPlan(node) == NULL);
	Assert(innerPlan(node) == NULL);

	/* initialize the start/end position */
	gss->curr_blknum = 0;
	gss->last_blknum = RelationGetNumberOfBlocks(scan_rel);
	/* initialize device qualifiers also, for fallback */
	gss->dev_quals = (List *)
		ExecInitExpr((Expr *) gs_info->dev_quals, &gss->css.ss.ps);
	/* 'tableoid' should not change during relation scan */
	gss->scan_tuple.t_tableOid = RelationGetRelid(scan_rel);
	/* assign kernel source and flags */
	gss->gts.kern_source = gs_info->kern_source;
	gss->gts.extra_flags = gs_info->extra_flags;
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
__pgstrom_release_gpuscan(pgstrom_gpuscan *gpuscan)
{
	CUresult	rc;

	if (gpuscan->ev_dma_recv_stop)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_recv_stop);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->ev_dma_recv_start)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_recv_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->ev_dma_send_stop)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_stop);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->ev_dma_send_start)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->m_kds)
	{
		rc = cuMemFree(gpuscan->m_kds);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemFree: %s", errorText(rc));
	}

	if (gpuscan->m_gpuscan)
	{
		rc = cuMemFree(gpuscan->m_gpuscan);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemFree: %s", errorText(rc));
	}

	/* ensure pointers being NULL */
	gpuscan->kern_qual = NULL;
	gpuscan->m_gpuscan = 0UL;
	gpuscan->m_kds = 0UL;
	gpuscan->ev_dma_send_start = NULL;
	gpuscan->ev_dma_send_stop  = NULL;
	gpuscan->ev_dma_recv_start = NULL;
	gpuscan->ev_dma_recv_stop  = NULL;
}

static void
pgstrom_release_gpuscan(GpuTask *gputask)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *) gputask;

	if (gpuscan->pds)
		pgstrom_release_data_store(gpuscan->pds);

	__pgstrom_release_gpuscan(gpuscan);

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
	if (!gss->gts.kern_source)
		length += STROMALIGN(offsetof(kern_resultbuf, results[0]));
	else
		length += STROMALIGN(offsetof(kern_resultbuf, results[nitems]));

	gpuscan = MemoryContextAllocZero(gcontext->memcxt, length);
	/* setting up */
	pgstrom_init_gputask(&gss->gts, &gpuscan->task,
						 pgstrom_process_gpuscan,
						 pgstrom_release_gpuscan);
	gpuscan->pds = pds;
	/* setting up kern_parambuf */
	Assert(gss->kparams->length == STROMALIGN(gss->kparams->length));
    memcpy(KERN_GPUSCAN_PARAMBUF(&gpuscan->kern),
           gss->kparams,
           gss->kparams->length);
	/* setting up kern_resultbuf */
	kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
    memset(kresults, 0, sizeof(kern_resultbuf));
    kresults->nrels = 1;
    kresults->nrooms = nitems;

	/* If GpuScan does not kick GPU Kernek execution, we treat
	 * this chunk as all-visible one, without reference to
	 * results[] row-index.
	 */
	if (!gss->gts.kern_source)
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
											pgstrom_chunk_size(),
											false);
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
		gpuscan->task.pfm.time_outer_load += timeval_diff(&tv1, &tv2);
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

static void
pgstrom_accum_gpuscan_perfmon(GpuScanState *gss, pgstrom_gpuscan *gpuscan)
{
	float		elapsed;
	CUresult	rc;

	/* Time for DMA send */
	rc = cuEventElapsedTime(&elapsed,
							gpuscan->ev_dma_send_start,
							gpuscan->ev_dma_send_stop);
	if (rc != CUDA_SUCCESS)
		goto error;
	gpuscan->task.pfm.time_dma_send += elapsed;

	/* Time for kernel exec */
	rc = cuEventElapsedTime(&elapsed,
							gpuscan->ev_dma_send_stop,
							gpuscan->ev_dma_recv_start);
	if (rc != CUDA_SUCCESS)
		goto error;
	gpuscan->task.pfm.time_kern_qual += elapsed;

	/* Time for DMA recv*/
	rc = cuEventElapsedTime(&elapsed,
							gpuscan->ev_dma_recv_start,
							gpuscan->ev_dma_recv_stop);
	if (rc != CUDA_SUCCESS)
		goto error;
	gpuscan->task.pfm.time_dma_recv += elapsed;

	/* accumulate them */
	pgstrom_accum_perfmon(&gss->gts.pfm_accum, &gpuscan->task.pfm);
	return;

error:
	elog(WARNING, "failed on cuEventElapsedTime: %s", errorText(rc));
	gss->gts.pfm_accum.enabled = false;
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
	pgstrom_gpuscan *gpuscan;
	dlist_node	   *dnode;
	bool			save_set_latch_on_sigusr1;
	int				rc;

	/*
	 * In case when no device code will be executed, we don't need to have
	 * asynchronous execution. So, just return a chunk with synchronous
	 * manner.
	 */
	if (!gss->gts.kern_source)
		return pgstrom_load_gpuscan(gss);

	save_set_latch_on_sigusr1 = set_latch_on_sigusr1;
	PG_TRY();
	{
	retry:
		SpinLockAcquire(&gss->gts.lock);
		while (pgstrom_max_async_chunks > (gss->gts.num_running_tasks +
										   gss->gts.num_pending_tasks +
										   gss->gts.num_completed_tasks))
		{
			/* no urgent reason why to make the scan progress */
			if (!dlist_is_empty(&gss->gts.completed_tasks) &&
				pgstrom_max_async_chunks < (gss->gts.num_running_tasks +
											gss->gts.num_pending_tasks))
				break;
			SpinLockRelease(&gss->gts.lock);

			gpuscan = pgstrom_load_gpuscan(gss);

			SpinLockAcquire(&gss->gts.lock);
			if (!gpuscan)
				break;
			dlist_push_tail(&gss->gts.pending_tasks, &gpuscan->task.chain);
			gss->gts.num_pending_tasks++;
			SpinLockRelease(&gss->gts.lock);

			pgstrom_launch_pending_tasks(&gss->gts);

			SpinLockAcquire(&gss->gts.lock);
		}

		/*
		 * Wait for the response from asynchronous task, if not completed
		 */
		if (dlist_is_empty(&gss->gts.completed_tasks))
		{
			SpinLockRelease(&gss->gts.lock);

			rc = WaitLatch(&MyProc->procLatch,
						   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
						   5 * 1000L);
			ResetLatch(&MyProc->procLatch);
			if (rc & WL_POSTMASTER_DEATH)
				elog(ERROR, "Emergency bail out because of Postmaster crash");
			/* flush pending tasks first */
			pgstrom_launch_pending_tasks(&gss->gts);
			goto retry;
		}
	}
	PG_CATCH();
	{
		set_latch_on_sigusr1 = save_set_latch_on_sigusr1;
		PG_RE_THROW();
	}
	PG_END_TRY();

	/*
	 * Picks up next available chunk if any
	 */
	dnode = dlist_pop_head_node(&gss->gts.completed_tasks);
	gpuscan = dlist_container(pgstrom_gpuscan, task.chain, dnode);
	memset(&gpuscan->task.chain, 0, sizeof(dlist_node));
	SpinLockRelease(&gss->gts.lock);

	/*
	 *
	 *
	 */
	if (gss->gts.pfm_accum.enabled)
		pgstrom_accum_gpuscan_perfmon(gss, gpuscan);

	/* release CUDA resources */
	__pgstrom_release_gpuscan(gpuscan);

	/*
	 * Raise an error, if any run-time error was reported
	 */
	if (gpuscan->task.errcode != StromError_Success)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("PG-Strom: CUDA execution error (%s)",
						errorText(gpuscan->task.errcode))));

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

		/* Release the current gpuscan chunk which is already scanned */
		if (gpuscan)
		{
			SpinLockAcquire(&gss->gts.lock);
			dlist_delete(&gpuscan->task.tracker);
			SpinLockRelease(&gss->gts.lock);
			pgstrom_release_gpuscan(&gpuscan->task);
			gss->curr_chunk = NULL;
			gss->curr_index = 0;
		}
		/* Fetch next chunk to be scanned */
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

static pgstrom_data_store *
gpuscan_exec_bulk(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *) node;
<<<<<<< HEAD
	Relation			rel = node->ss.ss_currentRelation;
	TupleTableSlot	   *slot = node->ss.ss_ScanTupleSlot;
	TupleDesc			tupdesc = slot->tts_tupleDescriptor;
	Snapshot			snapshot = node->ss.ps.state->es_snapshot;
	pgstrom_data_store *pds;
	struct timeval		tv1, tv2;

	/* note: bulkload shall not be chosen with qualifier */
	Assert(!gss->css.ss.ps.qual && !gss->dev_quals);

	/* no more blocks to read */
	if (gss->curr_blknum > gss->last_blknum)
		return NULL;

	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStartNode(node->ss.ps.instrument);
=======
	pgstrom_gpuscan	   *gpuscan;
	kern_resultbuf	   *kresults;
	pgstrom_bulkslot   *bulk = NULL;
>>>>>>> 66562309c0afc393580a884b3c231ab668008fd9

	if (gss->pfm.enabled)
		gettimeofday(&tv1, NULL);

	while (true)
	{
<<<<<<< HEAD
		pds = pgstrom_create_data_store_row(gss->gts.gcontext,
											tupdesc,
											pgstrom_chunk_size(),
											false);
		/* fill up this data store */
		while (gss->curr_blknum < gss->last_blknum &&
			   pgstrom_data_store_insert_block(pds, rel,
											   gss->curr_blknum,
											   snapshot, true) >= 0)
			gss->curr_blknum++;

		if (pds->kds->nitems > 0)
			break;
		pgstrom_release_data_store(pds);
        pds = NULL;
	}
	/* update perfmon statistics */
	if (gss->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gss->pfm.time_outer_load += timeval_diff(&tv1, &tv2);
	}

	if (node->ss.ps.instrument)
        InstrStopNode(node->ss.ps.instrument,
					  !pds ? 0.0 : (double) pds->kds->nitems);
	return pds;
=======
		gpuscan = pgstrom_fetch_gpuscan(gss);
		if (!gpuscan)
			return NULL;

		/* update perfmon info */
		if (gpuscan->msg.pfm.enabled)
			pgstrom_perfmon_add(&gss->pfm, &gpuscan->msg.pfm);

		/*
		 * Make a bulk-slot according to the result
		 */
		kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
		bulk = palloc0(offsetof(pgstrom_bulkslot, rindex[kresults->nitems]));
		bulk->pds = pgstrom_get_data_store(gpuscan->pds);
		bulk->nvalids = -1;		/* only -1 is valid */
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
		Assert(kresults->all_visible);
		Assert(!node->ss.ps.qual);

		if (bulk->pds->kds->nitems > 0)
			break;

		/*
		 * If this chunk has no valid items, it does not make sense to
		 * return this chunk to upper execution node.
		 */
		pgstrom_untrack_object(&bulk->pds->sobj);
        pgstrom_put_data_store(bulk->pds);
        pfree(bulk);
		bulk = NULL;
	}
	return bulk;
>>>>>>> 66562309c0afc393580a884b3c231ab668008fd9
}

static void
gpuscan_end(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *)node;

	pgstrom_release_gputaskstate(&gss->gts);
}

static void
gpuscan_rescan(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *) node;

	/* clean-up and release any concurrent tasks */
    pgstrom_cleanup_gputaskstate(&gss->gts);

	/* OK, rewind the position to read */
	gss->curr_blknum = 0;
	gss->curr_index = 0;
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
	pgstrom_explain_custom_flags(&gss->css, es);
	pgstrom_explain_kernel_source(&gss->gts, es);

	if (es->analyze && gss->pfm.enabled)
		pgstrom_explain_perfmon(&gss->pfm, es);
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

static void
gpuscan_cleanup_cuda_resources(pgstrom_gpuscan *gpuscan)
{
	CUresult	rc;

	if (gpuscan->ev_dma_recv_stop)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_recv_stop);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->ev_dma_recv_start)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_recv_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->ev_dma_send_stop)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_stop);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->ev_dma_send_start)
	{
		rc = cuEventDestroy(gpuscan->ev_dma_send_start);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
	}

	if (gpuscan->m_kds)
	{
		rc = cuMemFree(gpuscan->m_kds);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemFree: %s", errorText(rc));
	}

	if (gpuscan->m_gpuscan)
	{
		rc = cuMemFree(gpuscan->m_gpuscan);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemFree: %s", errorText(rc));
	}

	/* ensure pointers being NULL */
	gpuscan->kern_qual = NULL;
	gpuscan->m_gpuscan = 0UL;
	gpuscan->m_kds = 0UL;
	gpuscan->ev_dma_send_start = NULL;
	gpuscan->ev_dma_send_stop  = NULL;
	gpuscan->ev_dma_recv_start = NULL;
	gpuscan->ev_dma_recv_stop  = NULL;
}

static void
pgstrom_respond_gpuscan(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpuscan	   *gpuscan = private;
	kern_resultbuf	   *kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	GpuTaskState	   *gts = gpuscan->task.gts;

	SpinLockAcquire(&gts->lock);
	if (status != CUDA_SUCCESS)
		gpuscan->task.errcode = status;
	else
		gpuscan->task.errcode = kresults->errcode;

	if (gpuscan->task.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &gpuscan->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &gpuscan->task.chain);
	gts->num_completed_tasks++;
	SpinLockRelease(&gts->lock);
}

static bool
__pgstrom_process_gpuscan(pgstrom_gpuscan *gpuscan)
{
	kern_resultbuf	   *kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	pgstrom_data_store *pds = gpuscan->pds;
	kern_data_store	   *kds = pds->kds;
	size_t				offset;
	size_t				length;
	size_t				grid_size;
	size_t				block_size;
	CUresult			rc;

	/*
	 * Switch CUDA context
	 */
	rc = cuCtxSetCurrent(gpuscan->task.cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxSetCurrent: %s", errorText(rc));

	/*
	 * Kernel function lookup
	 */
	rc = cuModuleGetFunction(&gpuscan->kern_qual,
							 gpuscan->task.cuda_module,
							 "gpuscan_qual");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s",
			 errorText(rc));

	/*
	 * Allocation of device memory
	 */
	length = KERN_GPUSCAN_LENGTH(&gpuscan->kern);
	rc = cuMemAlloc(&gpuscan->m_gpuscan, length);
	if (rc != CUDA_SUCCESS)
	{
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
			goto out_of_resource;
		elog(ERROR, "failed on cuMemAlloc: %s", errorText(rc));
	}

	length = KERN_DATA_STORE_LENGTH(pds->kds);
	rc = cuMemAlloc(&gpuscan->m_kds, length);
	if (rc != CUDA_SUCCESS)
	{
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
			goto out_of_resource;
		elog(ERROR, "failed on cuMemAlloc: %s", errorText(rc));
	}

	/*
	 * Creation of event objects
	 */
	rc = cuEventCreate(&gpuscan->ev_dma_send_start, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

	rc = cuEventCreate(&gpuscan->ev_dma_send_stop, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

	rc = cuEventCreate(&gpuscan->ev_dma_recv_start, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

	rc = cuEventCreate(&gpuscan->ev_dma_recv_stop, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

	/*
	 * OK, enqueue a series of requests
	 */
	offset = KERN_GPUSCAN_DMASEND_OFFSET(&gpuscan->kern);
	length = KERN_GPUSCAN_DMASEND_LENGTH(&gpuscan->kern);
	rc = cuMemcpyHtoDAsync(gpuscan->m_gpuscan,
						   (char *)&gpuscan->kern + offset,
						   length,
						   gpuscan->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpuscan->task.pfm.bytes_dma_send += length;
	gpuscan->task.pfm.num_dma_send++;

	rc = cuMemcpyHtoDAsync(gpuscan->m_kds,
						   kds,
						   kds->length,
						   gpuscan->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpuscan->task.pfm.bytes_dma_send += kds->length;
    gpuscan->task.pfm.num_dma_send++;

	/*
	 * Launch kernel function
	 */
	pgstrom_compute_workgroup_size(&grid_size,
								   &block_size,
								   gpuscan->kern_qual,
								   gpuscan->task.cuda_device,
								   false,
								   kds->nitems,
								   sizeof(cl_uint));
	gpuscan->kern_qual_args[0] = &gpuscan->m_gpuscan;
	gpuscan->kern_qual_args[1] = &gpuscan->m_kds;

	rc = cuLaunchKernel(gpuscan->kern_qual,
						grid_size, 1, 1,
						block_size, 1, 1,
						sizeof(uint) * block_size,
						gpuscan->task.cuda_stream,
						gpuscan->kern_qual_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	gpuscan->task.pfm.num_kern_qual++;

	/*
	 * Recv DMA call
	 */
	offset = KERN_GPUSCAN_DMARECV_OFFSET(&gpuscan->kern);
	length = KERN_GPUSCAN_DMARECV_LENGTH(&gpuscan->kern);
	rc = cuMemcpyDtoHAsync(kresults,
						   gpuscan->m_gpuscan + offset,
						   length,
						   gpuscan->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));

	/*
	 * Register callback
	 */
	rc = cuStreamAddCallback(gpuscan->task.cuda_stream,
							 pgstrom_respond_gpuscan,
							 gpuscan, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));

	return true;

out_of_resource:
	gpuscan_cleanup_cuda_resources(gpuscan);
	return false;
}

/*
 * clserv_process_gpuscan
 *
 * entrypoint of kernel gpuscan implementation
 */
static bool
pgstrom_process_gpuscan(GpuTask *task)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *) task;
	GpuTaskState	   *gts = gpuscan->task.gts;
	bool				status;
	struct timeval		tv1, tv2;

	if (task->pfm.enabled)
		gettimeofday(&tv1, NULL);

	PG_TRY();
	{
		status = __pgstrom_process_gpuscan(gpuscan);
		SpinLockAcquire(&gts->lock);
		if (status)
		{
			dlist_push_tail(&gts->running_tasks, &gpuscan->chain);
			gts->num_running_tasks++;
		}
		else
		{
			dlist_push_head(&gts->pending_tasks, &gpuscan->chain);
			gts->num_pending_tasks++;
		}
		SpinLockRelease(&gts->lock);
	}
	PG_CATCH();
	{
		gpuscan_cleanup_cuda_resources(gpuscan);
		PG_RE_THROW();
	}
	PG_END_TRY();

	if (task->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		task->pfm.time_launch_cuda = timeval_diff(&tv1, &tv2);
	}
	return status;
}
