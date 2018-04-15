/*
 * gpuscan.c
 *
 * Sequential scan accelerated by GPU processors
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
#include "access/relscan.h"
#include "access/sysattr.h"
#include "access/xact.h"
#include "catalog/heap.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_type.h"
#include "executor/nodeCustom.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/plancat.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/ruleutils.h"
#include "utils/spccache.h"
#include "pg_strom.h"
#include "cuda_numeric.h"
#include "cuda_gpuscan.h"

static set_rel_pathlist_hook_type	set_rel_pathlist_next;
static CustomPathMethods	gpuscan_path_methods;
static CustomScanMethods	gpuscan_plan_methods;
static CustomExecMethods	gpuscan_exec_methods;
static bool					enable_gpuscan;
static bool					enable_pullup_outer_scan;

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
	List	   *func_defs;		/* list of declared functions */
	List	   *expr_defs;		/* list of special expression in use */
	List	   *used_params;	/* list of Const/Param in use */
	List	   *used_vars;		/* list of Var in use */
	List	   *dev_quals;		/* qualifiers to be run on device */
	cl_bool		force_row_format; /* true, if row format is required */
	cl_int      base_fixed_width; /* width of fixed fields on base rel */
    cl_int      proj_fixed_width; /* width of fixed fields on projection */
    cl_int      proj_extra_width; /* width of extra buffer on projection */
} GpuScanInfo;

static inline void
form_gpuscan_info(CustomScan *cscan, GpuScanInfo *gs_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, makeString(gs_info->kern_source ?
									  pstrdup(gs_info->kern_source) :
									  NULL));
	privs = lappend(privs, makeInteger(gs_info->extra_flags));
	privs = lappend(privs, gs_info->func_defs);
	privs = lappend(privs, gs_info->expr_defs);
	exprs = lappend(exprs, gs_info->used_params);
	exprs = lappend(exprs, gs_info->used_vars);
	exprs = lappend(exprs, gs_info->dev_quals);
	privs = lappend(privs, makeInteger(gs_info->force_row_format));
	privs = lappend(privs, makeInteger(gs_info->base_fixed_width));
	privs = lappend(privs, makeInteger(gs_info->proj_fixed_width));
	privs = lappend(privs, makeInteger(gs_info->proj_extra_width));

	cscan->custom_private = privs;
    cscan->custom_exprs = exprs;
}

static GpuScanInfo *
deform_gpuscan_info(CustomScan *cscan)
{
	GpuScanInfo	   *gs_info = palloc0(sizeof(GpuScanInfo));
	List		   *privs = cscan->custom_private;
	List		   *exprs = cscan->custom_exprs;
	int				pindex = 0;
	int				eindex = 0;

	Assert(IsA(cscan, CustomScan));
	gs_info->kern_source = strVal(list_nth(privs, pindex++));
	gs_info->extra_flags = intVal(list_nth(privs, pindex++));
	gs_info->func_defs = list_nth(privs, pindex++);
	gs_info->expr_defs = list_nth(privs, pindex++);
	gs_info->used_params = list_nth(exprs, eindex++);
	gs_info->used_vars = list_nth(exprs, eindex++);
	gs_info->dev_quals = list_nth(exprs, eindex++);
	gs_info->force_row_format = intVal(list_nth(privs, pindex++));
	gs_info->base_fixed_width = intVal(list_nth(privs, pindex++));
	gs_info->proj_fixed_width = intVal(list_nth(privs, pindex++));
	gs_info->proj_extra_width = intVal(list_nth(privs, pindex++));

	return gs_info;
}

typedef struct
{
	GpuTask			task;
	dlist_node		chain;
	CUfunction		kern_exec_quals;
	CUfunction		kern_dev_proj;
	CUdeviceptr		m_gpuscan;
	CUdeviceptr		m_kds_src;
	CUdeviceptr		m_kds_dst;
	CUevent 		ev_dma_send_start;
	CUevent			ev_dma_send_stop;
	CUevent			ev_kern_exec_quals;
	CUevent			ev_dma_recv_start;
	CUevent			ev_dma_recv_stop;
	pgstrom_data_store *pds_src;
	pgstrom_data_store *pds_dst;
	kern_resultbuf *kresults;
	kern_gpuscan	kern;
} pgstrom_gpuscan;

typedef struct {
	GpuTaskState	gts;

	HeapTupleData	scan_tuple;		/* buffer to fetch tuple */
	List		   *dev_tlist;		/* tlist to be returned from the device */
	List		   *dev_quals;		/* quals to be run on the device */
	bool			dev_projection;	/* true, if device projection is valid */
	cl_int			base_fixed_width; /* width of fixed fields on base rel */
	cl_int			proj_fixed_width; /* width of fixed fields on projection */
	cl_int			proj_extra_width; /* width of extra buffer on projection */
	/* resource for CPU fallback */
	TupleTableSlot *base_slot;
	ProjectionInfo *base_proj;
} GpuScanState;

/* forward declarations */
static bool pgstrom_process_gpuscan(GpuTask *gtask);
static bool pgstrom_complete_gpuscan(GpuTask *gtask);
static void pgstrom_release_gpuscan(GpuTask *gtask);
static GpuTask *gpuscan_next_chunk(GpuTaskState *gts);
static TupleTableSlot *gpuscan_next_tuple(GpuTaskState *gts);

/* dummy kernel source if no device qualifier is given */
#define GPUSCAN_KERN_SOURCE_NO_DEVQUAL					\
	"STATIC_FUNCTION(cl_bool)\n"						\
	"gpuscan_quals_eval(kern_context *kcxt,\n"			\
	"                   kern_data_store *kds,\n"		\
	"                   size_t kds_index)\n"			\
	"{\n"												\
	"  return true;\n"									\
	"}\n"

#if 0
/*
 * cost_gpuscan
 *
 * Cost estimation of GpuScan - Note that we may need to call this function
 * on the planner_hook context.
 */
static void
cost_gpuscan(AttrNumber ncols,		/* number of columns */
			 Oid tablespace_oid,	/* OID of tablespace */
			 BlockNumber num_pages,	/* number of base blocks */
			 double num_tuples,		/* number of base tuples */
			 double num_rows,		/* number of returned rows */
			 Selectivity dev_sel,	/* selectivity by dev_quals */
			 List *dev_quals,		/* List of device quals */
			 List *host_quals,		/* List of host quals */
			 List *ppi_quals,		/* List of PPI quals */
			 Cost *p_startup_cost,	/* out: startup_cost */
			 Cost *p_total_cost)	/* out: total_cost */
{
	Cost		startup_cost = pgstrom_gpu_setup_cost;
	Cost		run_cost = 0.0;
	Cost		startup_delay = 0.0;
	Cost		gpu_per_tuple;
	Cost		cpu_per_tuple;
	QualCost	dev_cost;
	QualCost	host_cost;
	QualCost	ppi_cost;
	double		spc_seq_page_cost;
	double		heap_size;
	cl_uint		htup_size;
	cl_uint		num_chunks;

	/* estimate number of chunks */
	heap_size = (double)(BLCKSZ - SizeOfPageHeaderData) * num_pages;
	htup_size = (MAXALIGN(offsetof(HeapTupleHeaderData,
								   t_bits[BITMAPLEN(ncols)])) +
				 MAXALIGN(heap_size / Max(num_tuples, 1.0) -
						  sizeof(ItemIdData) - SizeofHeapTupleHeader));
	num_chunks = (cl_uint)
		((double)(htup_size + sizeof(cl_int)) * num_tuples /
		 (double)(pgstrom_chunk_size() -
				  STROMALIGN(offsetof(kern_data_store, colmeta[ncols]))));
	num_chunks = Max(num_chunks, 1);

	/* fetch estimated page cost for tablespace containing the table */
	get_tablespace_page_costs(tablespace_oid, NULL, &spc_seq_page_cost);

	/* Disk costs */
	run_cost += spc_seq_page_cost * num_pages;

	/* GPU costs */
	cost_qual_eval(&dev_cost, dev_quals, NULL);
	if (cpu_tuple_cost > 0.0)
		dev_cost.per_tuple *= pgstrom_gpu_operator_cost / cpu_operator_cost;
	else
		dev_cost.per_tuple += disable_cost;

	/* CPU costs */
	cost_qual_eval(&host_cost, host_quals, NULL);

	/* PPI costs (as a part of host quals, if any) */
	cost_qual_eval(&ppi_cost, ppi_quals, NULL);
	host_cost.startup += ppi_cost.startup;
	host_cost.per_tuple += ppi_cost.per_tuple;

	/* total path cost */
	startup_cost += dev_cost.startup + host_cost.startup;
	cpu_per_tuple = host_cost.per_tuple + cpu_tuple_cost;
	gpu_per_tuple = dev_cost.per_tuple;

	run_cost += gpu_per_tuple * num_tuples;
	if (dev_quals != NIL)
		startup_delay = run_cost * (1.0 / (double)num_chunks);
	run_cost += cpu_per_tuple * dev_sel * num_tuples;

	*p_startup_cost = startup_cost + startup_delay;
	*p_total_cost = startup_cost + run_cost;
}
#endif

static void
cost_gpuscan_path(PlannerInfo *root, CustomPath *pathnode,
				  List *final_tlist, List *dev_quals, List *host_quals)
{
	RelOptInfo	   *baserel = pathnode->path.parent;
	ParamPathInfo  *param_info = pathnode->path.param_info;
	List		   *ppi_quals = param_info ? param_info->ppi_clauses : NIL;
	Cost			startup_cost = pgstrom_gpu_setup_cost;
	Cost			run_cost = 0.0;
	Cost			startup_delay = 0.0;
	Cost			cpu_per_tuple = 0.0;
	double			selectivity;
	double			heap_size;
	Size			htup_size;
	Size			num_chunks;
	QualCost		qcost;
	double			spc_seq_page_cost;
	double			ntuples = baserel->tuples;
	double			gpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;

	pathnode->path.rows = (param_info
						   ? param_info->ppi_rows
						   : baserel->rows);
	/* estimate selectivity */
	selectivity = clauselist_selectivity(root,
										 dev_quals,
										 baserel->relid,
										 JOIN_INNER,
										 NULL);
	/* estimate number of chunks */
	heap_size = (double)(BLCKSZ - SizeOfPageHeaderData) * baserel->pages;
	htup_size = (MAXALIGN(offsetof(HeapTupleHeaderData,
								   t_bits[BITMAPLEN(baserel->max_attr)])) +
				 MAXALIGN(heap_size / Max(baserel->tuples, 1.0) -
						  sizeof(ItemIdData) - SizeofHeapTupleHeader));
	num_chunks = (Size)
		(((double)(offsetof(kern_tupitem, htup) + htup_size +
				   sizeof(cl_uint)) * Max(baserel->tuples, 1.0)) /
		 ((double)(pgstrom_chunk_size() -
				   KDS_CALCULATE_HEAD_LENGTH(baserel->max_attr))));
	num_chunks = Max(num_chunks, 1);

	/* fetch estimated page cost for tablespace containing the table */
	get_tablespace_page_costs(baserel->reltablespace,
							  NULL, &spc_seq_page_cost);
	/* Disk costs */
	run_cost += spc_seq_page_cost * (double)baserel->pages;

	/* Cost for GPU qualifiers */
	cost_qual_eval(&qcost, dev_quals, root);
	startup_cost += qcost.startup;
	run_cost += qcost.per_tuple * gpu_ratio * ntuples;
	ntuples *= selectivity;

	/* Cost for CPU qualifiers */
	cost_qual_eval(&qcost, host_quals, root);
	startup_cost += qcost.startup;
	cpu_per_tuple += qcost.per_tuple;

	/* PPI costs (as a part of host quals, if any) */
	cost_qual_eval(&qcost, ppi_quals, root);
	startup_cost += qcost.startup;
	cpu_per_tuple += qcost.per_tuple;

	run_cost += (cpu_per_tuple + cpu_tuple_cost) * ntuples;

	/* Cost for DMA transfer */
	run_cost += pgstrom_gpu_dma_cost * (double) num_chunks;

	/* Cost discount it GPU Projection off-loads CPU workloads */
	if (final_tlist != NIL)
	{
		Cost		discount_per_tuple = 0.0;
		Cost		discount_total;
		cl_uint		num_vars = 0;
		ListCell   *lc;
		Bitmapset  *varattnos = NULL;

		foreach (lc, final_tlist)
		{
			TargetEntry	   *tle = lfirst(lc);

			if (IsA(tle->expr, Var))
				varattnos = bms_add_member(varattnos,
										   ((Var *)tle->expr)->varno -
										   FirstLowInvalidHeapAttributeNumber);
			else if (IsA(tle->expr, Const) || IsA(tle->expr, Param))
				num_vars++;
			else if (pgstrom_device_expression(tle->expr))
			{
				cost_qual_eval_node(&qcost, (Node *)tle->expr, root);
				discount_per_tuple += (qcost.per_tuple *
									   Max(1.0 - gpu_ratio, 0.0) / 10.0);
				num_vars++;
			}
			else
			{
				pull_varattnos((Node *)tle->expr, baserel->relid, &varattnos);
			}
		}
		if (num_vars > baserel->max_attr)
			discount_per_tuple -= cpu_tuple_cost *
				(double)(num_vars - baserel->max_attr);
		discount_total = discount_per_tuple * baserel->rows;

		run_cost = Max(run_cost - discount_total, 0.0);
	}

	/* Latency to get the first chunk */
	startup_delay = run_cost * (1.0 / (double) num_chunks);

	pathnode->path.startup_cost = startup_cost + startup_delay;
	pathnode->path.total_cost = startup_cost + run_cost;
}

static void
gpuscan_add_scan_path(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Index rtindex,
					  RangeTblEntry *rte)
{
	CustomPath	   *pathnode;
	List		   *final_tlist = NIL;
	List		   *dev_quals = NIL;
	List		   *host_quals = NIL;
	ListCell	   *cell;
	codegen_context	context;

	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);

	/* nothing to do, if either PG-Strom or GpuScan is not enabled */
	if (!pgstrom_enabled || !enable_gpuscan)
		return;

	/* We already proved the relation empty, so nothing more to do */
	if (IS_DUMMY_REL(baserel))
		return;

	/* It is the role of built-in Append node */
	if (rte->inh)
		return;

	/* only base relation we can handle */
	if (baserel->rtekind != RTE_RELATION || baserel->relid == 0)
		return;

	/* system catalog is not supported */
	if (get_rel_namespace(rte->relid) == PG_CATALOG_NAMESPACE)
		return;

	/* Is this path responsible to evaluation of final target-list? */
	if (bms_equal(root->all_baserels, baserel->relids))
	{
		foreach (cell, root->parse->targetList)
		{
			TargetEntry	   *tle = lfirst(cell);

			if (!IsA(tle->expr, Var) &&
				!IsA(tle->expr, Const) &&
				!IsA(tle->expr, Param) &&
				pgstrom_device_expression(tle->expr))
			{
				final_tlist = root->parse->targetList;
				break;
			}
		}
	}

	/*
	 * check whether the qualifier can run on GPU device, or not
	 */
	pgstrom_init_codegen_context(&context);
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_device_expression(rinfo->clause))
			dev_quals = lappend(dev_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}

	/*
	 * GpuScan does not make sense if neither final_tlist nor dev_quals valid.
	 * It is almost equivalent to SeqScan.
	 */
	if (final_tlist == NIL && dev_quals == NIL)
		return;

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
	cost_gpuscan_path(root, pathnode, final_tlist, dev_quals, host_quals);

	/*
	 * FIXME: needs to pay attention for projection cost?
	 */
	add_path(baserel, &pathnode->path);
}

/*
 * pgstrom_pullup_outer_scan
 *
 * It tries to pull up underlying SeqScan or GpuScan node if it is mergeable
 * to the upper node.
 */
bool
pgstrom_pullup_outer_scan(Plan *plannode,
						  bool allow_expression,
						  List **p_outer_quals)
{
	List		   *outer_quals;
	ListCell	   *lc;

	if (!enable_pullup_outer_scan)
		return false;

	if (IsA(plannode, SeqScan))
	{
		SeqScan	   *seqscan = (SeqScan *) plannode;

		outer_quals = seqscan->plan.qual;
		/* Scan-quals must be entirely device executable */
		if (!pgstrom_device_expression((Expr *) outer_quals))
			return false;
	}
	else if (pgstrom_plan_is_gpuscan(plannode))
	{
		CustomScan	   *cscan = (CustomScan *) plannode;
		GpuScanInfo	   *gs_info = deform_gpuscan_info(cscan);

		/* Scan-quals must be entirely device executable */
		if (cscan->scan.plan.qual != NIL)
			return false;
		outer_quals = gs_info->dev_quals;
		Assert(pgstrom_device_expression((Expr *) outer_quals));
	}
	else
	{
		/* quick bailout if unavailable to pull-up */
		return false;
	}

	/*
	 * Check whether target-list of the outer scan node is legal.
	 * GpuJoin expects tlist of underlying node has only Var-nodes;
	 * that makes projection simple. GpuPreAgg may have expression
	 * node on the tlist.
	 *
	 * XXX - Can we share the device projection code of GpuScan
	 * for GpuPreAgg with expression input? One major difference
	 * is GpuPreAgg needs expression as input, but GpuScan builds
	 * CPU accessible output.
	 */
	foreach (lc, plannode->targetlist)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;
			/* we cannot support whole row reference */
			if (var->varattno == InvalidAttrNumber)
				return false;
			continue;
		}
		if (allow_expression &&
			pgstrom_device_expression(tle->expr))
			continue;

		return false;
	}

	/*
	 * return the properties
	 */
	*p_outer_quals = copyObject(outer_quals);

	return true;
}

/*
 * Code generator for GpuScan's qualifier
 */
static char *
gpuscan_codegen_exec_quals(codegen_context *context, List *dev_quals)
{
	StringInfoData	body;
	char		   *expr_code;

	if (dev_quals == NULL)
		return GPUSCAN_KERN_SOURCE_NO_DEVQUAL;

	/* Let's walk on the device expression tree */
	expr_code = pgstrom_codegen_expression((Node *)dev_quals, context);

	initStringInfo(&body);
	appendStringInfo(
		&body,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpuscan_quals_eval(kern_context *kcxt,\n"
		"                   kern_data_store *kds,\n"
		"                   size_t kds_index)\n"
		"{\n");
	/* add parameter declarations */
	pgstrom_codegen_param_declarations(&body, context);
	/* add variables declarations */
	pgstrom_codegen_var_declarations(&body, context);

	appendStringInfo(
		&body,
		"\n"
		"  return EVAL(%s);\n"
		"}\n",
		expr_code);

	return body.data;
}

static Plan *
create_gpuscan_plan(PlannerInfo *root,
					RelOptInfo *rel,
					CustomPath *best_path,
					List *tlist,
					List *clauses,
					List *custom_children)
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
	Assert(custom_children == NIL);

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

		if (!pgstrom_device_expression(rinfo->clause))
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
	pgstrom_init_codegen_context(&context);
	kern_source = gpuscan_codegen_exec_quals(&context, dev_quals);

	/*
	 * Construction of GpuScanPlan node; on top of CustomPlan node
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = host_quals;
	cscan->scan.plan.lefttree = NULL;
	cscan->scan.plan.righttree = NULL;
	cscan->scan.scanrelid = rel->relid;

	memset(&gs_info, 0, sizeof(GpuScanInfo));
	gs_info.kern_source = kern_source;
	gs_info.extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUSCAN;
	gs_info.func_defs = context.func_defs;
	gs_info.expr_defs = context.expr_defs;
	gs_info.used_params = context.used_params;
	gs_info.used_vars = context.used_vars;
	gs_info.dev_quals = dev_quals;
	form_gpuscan_info(cscan, &gs_info);
	cscan->flags = best_path->flags;
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

	if (IsA(cscan, CustomScan) && cscan->methods == &gpuscan_plan_methods)
		return true;
	return false;
}

/*
 * codegen_device_projection
 *
 * It makes GPU code to perform device projection to extract a valid
 * kern_tupitem into tup_values/tup_isnull form.
 */
static void
codegen_device_projection(StringInfo source,
						  codegen_context *context,
						  Index scanrelid,
						  TupleDesc tupdesc,
						  List *tlist_dev)
{
	AttrNumber	   *varremaps;
	Bitmapset	   *varattnos;
	ListCell	   *lc;
	int				prev;
	int				i, j, k;
	bool			needs_vlbuf = false;
	devtype_info   *dtype;
	StringInfoData	decl;
	StringInfoData	body;
	StringInfoData	temp;

	initStringInfo(&decl);
	initStringInfo(&body);
	initStringInfo(&temp);

	/*
	 * step.1 - declaration of function and KVAR_xx for expressions
	 */
	appendStringInfoString(
		&decl,
		"STATIC_FUNCTION(void)\n"
		"gpuscan_projection(kern_context *kcxt,\n"
		"					kern_data_store *kds_src,\n"
		"                   kern_tupitem *tupitem,\n"
		"                   kern_data_store *kds_dst,\n"
		"                   cl_uint dst_nitems,\n"
		"                   Datum *tup_values,\n"
		"                   cl_bool *tup_isnull,\n"
		"                   cl_bool *tup_internal)\n"
		"{\n"
		"  HeapTupleHeaderData *htup;\n");

	varremaps = palloc0(sizeof(AttrNumber) * list_length(tlist_dev));
	varattnos = NULL;
	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		/*
		 * NOTE: If expression of TargetEntry is a simple Var-node,
		 * we can load the value into tup_values[]/tup_isnull[]
		 * array regardless of the data type. We have to track which
		 * column is the source of this TargetEntry.
		 * Elsewhere, we will construct device side expression using
		 * KVAR_xx variables.
		 */
		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;

			Assert(var->varno == scanrelid);
			Assert(var->varattno > FirstLowInvalidHeapAttributeNumber &&
				   var->varattno != InvalidAttrNumber &&
				   var->varattno <= tupdesc->natts);
			varremaps[tle->resno - 1] = var->varattno;
		}
		else
		{
			pull_varattnos((Node *) tle->expr, scanrelid, &varattnos);
		}
	}

	prev = -1;
	while ((prev = bms_next_member(varattnos, prev)) >= 0)
	{
		Form_pg_attribute attr;
		AttrNumber	anum = prev + FirstLowInvalidHeapAttributeNumber;

		/* system column should not appear within device expression */
		Assert(anum > 0);
		attr = tupdesc->attrs[anum - 1];

		dtype = pgstrom_devtype_lookup(attr->atttypid);
		if (!dtype)
			elog(ERROR, "Bug? failed to lookup device supported type: %s",
				 format_type_be(attr->atttypid));
		appendStringInfo(&decl,
						 "  pg_%s_t KVAR_%u;\n",
						 dtype->type_name, anum);
	}

	/*
	 * step.2 - extract tuples and load values to KVAR or values/isnull array
	 * (only if tupitem_src is valid, of course)
	 */
	appendStringInfo(
		&body,
		"  htup = (!tupitem ? NULL : &tupitem->htup);\n"
		"  if (htup)\n"
		"  {\n"
		"    char    *curr = (char *)htup + htup->t_hoff;\n"
		"    cl_bool  heap_hasnull = ((htup->t_infomask & HEAP_HASNULL)!=0);\n"
		"    kern_colmeta cmeta;\n"
		"\n"
		"    assert((devptr_t)curr == MAXALIGN(curr));\n");

	/* system columns reference if any */
	for (j=0; j < list_length(tlist_dev); j++)
	{
		if (varremaps[j] < 0)
		{
			Form_pg_attribute	attr
				= SystemAttributeDefinition(varremaps[j], true);

			appendStringInfo(
				&body,
				"    /* %s system column */\n"
				"    tup_isnull[%d] = false;\n"
				"    tup_values[%d] = kern_getsysatt_%s(kds_src, htup);\n",
				NameStr(attr->attname),
				j, j,
				NameStr(attr->attname));
		}
	}

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];
		bool		referenced = false;

		dtype = pgstrom_devtype_lookup(attr->atttypid);
		k = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		appendStringInfo(
			&temp,
			"    /* attribute %d */\n"
			"    cmeta = kds_src->colmeta[%d];\n"
			"    if (!heap_hasnull || !att_isnull(%d, htup->t_bits))\n"
			"    {\n",
			attr->attnum,
			attr->attnum - 1,
			attr->attnum - 1);

		/* current offset alignment */
		if (attr->attlen > 0)
			appendStringInfoString(
				&temp,
				"      curr = (char *)TYPEALIGN(cmeta.attalign, curr);\n");
		else
			appendStringInfoString(
				&temp,
				"      if (!VARATT_NOT_PAD_BYTE(curr))\n"
				"        curr = (char *)TYPEALIGN(cmeta.attalign, curr);\n");

		/* Put values on tup_values/tup_isnull and KVAR_xx if referenced */
		for (j=0; j < list_length(tlist_dev); j++)
		{
			if (varremaps[j] != attr->attnum)
				continue;

			appendStringInfo(
				&temp,
				"      tup_isnull[%d] = false;\n", j);
			if (attr->attbyval)
			{
				appendStringInfo(
					&temp,
					"      tup_values[%d] = *((%s *) curr);\n",
					j,
					(attr->attlen == sizeof(cl_long) ? "cl_long"
					 : attr->attlen == sizeof(cl_int) ? "cl_int"
					 : attr->attlen == sizeof(cl_short) ? "cl_short"
					 : "cl_char"));
			}
			else
			{
				appendStringInfo(
					&temp,
					"      tup_values[%d] = devptr_to_host(kds_src,curr);\n",
					j);
			}
			referenced = true;
		}

		k = attr->attnum - FirstLowInvalidHeapAttributeNumber;
		if (bms_is_member(k, varattnos))
		{
			appendStringInfo(
				&temp,
				"      KVAR_%u = pg_%s_datum_ref(kcxt, curr, false);\n",
				attr->attnum,
				dtype->type_name);
			referenced = true;
		}
		/* make advance the offset */
		appendStringInfoString(
			&temp,
			"      curr += (cmeta.attlen > 0 ?\n"
			"               cmeta.attlen :\n"
			"               VARSIZE_ANY(curr));\n"
			"    }\n");

		/* Put NULL on tup_isnull and KVAR_xx, if needed */
		if (referenced)
		{
			appendStringInfo(
				&temp,
				"    else\n"
				"    {\n");

			for (j=0; j < list_length(tlist_dev); j++)
			{
				if (varremaps[j] == attr->attnum)
					appendStringInfo(
						&temp,
						"      tup_isnull[%d] = true;\n", j);
			}
			k = attr->attnum - FirstLowInvalidHeapAttributeNumber;
			if (bms_is_member(k, varattnos))
			{
				appendStringInfo(
					&temp,
					"      KVAR_%u = pg_%s_datum_ref(kcxt, NULL, false);\n",
					attr->attnum,
					dtype->type_name);
			}
			appendStringInfo(
				&temp,
				"    }\n");
			/* we have to walk on until this column at least */
			appendStringInfoString(&body, temp.data);
			resetStringInfo(&temp);
		}
	}

	/*
	 * step.3 - execute expression node, then store the result onto KVAR_xx
	 */
    foreach (lc, tlist_dev)
    {
        TargetEntry    *tle = lfirst(lc);
		Oid				type_oid;

		if (IsA(tle->expr, Var))
			continue;

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "Bug? device supported type is missing: %u", type_oid);

		appendStringInfo(
			&decl,
			"  pg_%s_t temp_%u_v;\n",
			dtype->type_name,
			tle->resno);
		appendStringInfo(
			&body,
			"    temp_%u_v = %s;\n",
			tle->resno,
			pgstrom_codegen_expression((Node *) tle->expr, context));
	}

	appendStringInfo(
		&body,
		"  }\n"
		"\n");

	/*
	 * step.4 (only KDS_FORMAT_SLOT) - We have to acquire variable length
	 * buffer for indirect or numeric data type.
	 */
	appendStringInfo(
		&body,
		"  if (kds_dst->format == KDS_FORMAT_SLOT)\n"
		"  {\n");

	resetStringInfo(&temp);
	appendStringInfo(
		&temp,
		"    cl_uint vl_len = 0;\n"
		"    char   *vl_buf = NULL;\n"
		"    cl_uint offset;\n"
		"    cl_uint count;\n"
		"    cl_uint __shared__ base;\n"
		"\n"
		"    if (htup)\n"
		"    {\n");

	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid;

		if (IsA(tle->expr, Var) ||
			IsA(tle->expr, Const) ||
			IsA(tle->expr, Param))
			continue;

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "Bug? device supported type is missing: %u", type_oid);

		if (type_oid == NUMERICOID)
		{
			appendStringInfo(
				&temp,
				"      if (!temp_%u_v.isnull)\n"
				"        vl_len = TYPEALIGN(sizeof(cl_uint), vl_len)\n"
				"               + pg_numeric_to_varlena(kcxt,NULL,\n"
				"                                       temp_%u_v.value,\n"
				"                                       temp_%u_v.isnull);\n",
				tle->resno,
				tle->resno,
				tle->resno);
			needs_vlbuf = true;
		}
		else if (!dtype->type_byval)
		{
			/* varlena is not supported yet */
			Assert(dtype->type_length > 0);

			appendStringInfo(
				&temp,
				"      if (!temp_%u_v.isnull)\n"
				"        vl_len = TYPEALIGN(%u, vl_len) + %u;\n",
				tle->resno,
				dtype->type_align,
				dtype->type_length);
			needs_vlbuf = true;
		}
	}

	appendStringInfo(
		&temp,
		"    }\n"
		"\n"
		"    /* allocation of variable length buffer */\n"
		"    vl_len = MAXALIGN(vl_len);\n"
		"    offset = pgstromStairlikeSum(vl_len, &count);\n"
		"    if (get_local_id() == 0)\n"
		"    {\n"
		"      if (count > 0)\n"
		"        base = atomicAdd(&kds_dst->usage, count);\n"
		"      else\n"
		"        base = 0;\n"
		"    }\n"
		"    __syncthreads();\n"
		"\n"
		"    if (KERN_DATA_STORE_SLOT_LENGTH(kds_dst, dst_nitems) +\n"
		"        base + count > kds_dst->length)\n"
		"    {\n"
		"      STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);\n"
		"      return;\n"
		"    }\n"
		"    vl_buf = (char *)kds_dst + kds_dst->length\n"
		"           - (base + offset + vl_len);\n");

	if (needs_vlbuf)
		appendStringInfo(&body, "%s", temp.data);

	/*
	 * step.5 (only FDW_FORMAT_SLOT) - Store the KVAR_xx on the slot.
	 * pointer types must be host pointer
	 */
	appendStringInfo(
		&body,
		"    if (htup)\n"
		"    {\n");

	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid;

		if (IsA(tle->expr, Var))
			continue;

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "Bug? device supported type is missing: %u", type_oid);

		appendStringInfo(
			&body,
			"      tup_isnull[%d] = temp_%u_v.isnull;\n",
			tle->resno - 1, tle->resno);

		if (type_oid == NUMERICOID)
		{
			appendStringInfo(
				&body,
				"      if (!temp_%u_v.isnull)\n"
				"      {\n"
				"        vl_buf = (char *)TYPEALIGN(sizeof(cl_int), vl_buf);\n"
				"        tup_values[%d] = devptr_to_host(kds_dst, vl_buf);\n"
				"        vl_buf += pg_numeric_to_varlena(kcxt, vl_buf,\n"
				"                                        temp_%u_v.value,\n"
				"                                        temp_%u_v.isnull);\n"
				"       }\n",
				tle->resno,
				tle->resno - 1,
				tle->resno,
				tle->resno);
		}
		else if (dtype->type_byval)
		{
			appendStringInfo(
				&body,
				"      if (!temp_%u_v.isnull)\n"
				"        tup_values[%d] = pg_%s_to_datum(temp_%u_v.value);\n",
				tle->resno,
				tle->resno - 1,
				dtype->type_name,
				tle->resno);
		}
		else if (dtype->type_length > 0)
		{
			appendStringInfo(
				&body,
				"      if (!temp_%u_v.isnull)\n"
				"      {\n"
				"        vl_buf = (char *)TYPEALIGN(%u, vl_buf);\n"
				"        tup_values[%d] = devptr_to_host(kds_dst, vl_buf);\n"
				"        memcpy(vl_buf, &temp_%u_v.value, %d);\n"
				"        vl_buf += %d;\n"
				"      }\n",
				tle->resno,
				dtype->type_align,
				tle->resno - 1,
				tle->resno, dtype->type_length,
				dtype->type_length);
		}
		else
		{
			Assert(IsA(tle->expr, Const) || IsA(tle->expr, Param));
			appendStringInfo(
				&body,
				"      if (!temp_%u_v.isnull)\n"
				"      {\n"
				"        tup_values[%d] = devptr_to_host(kcxt->kparams,\n"
				"                                        temp_%u_v.value);\n"
				"      }\n",
				tle->resno,
				tle->resno - 1, tle->resno);
		}
	}
	appendStringInfo(
		&body,
		"    }\n"
		"  }\n");

	/*
	 * step.6 (only FDW_FORMAT_ROW) - Stora the KVAR_xx on the slot.
	 * pointer types must be device pointer.
	 */
	appendStringInfo(
		&body,
		"  else\n"
		"  {\n"
		"    if (htup)\n"
		"    {\n");

	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid;

		if (IsA(tle->expr, Var))
			continue;

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "Bug? device supported type is missing: %u", type_oid);

		appendStringInfo(
			&body,
			"      tup_isnull[%d] = temp_%u_v.isnull;\n",
			tle->resno - 1, tle->resno);

		if (type_oid == NUMERICOID)
		{
			appendStringInfo(
				&body,
				"      tup_internal[%d] = true;\n"
				"      if (!temp_%u_v.isnull)\n"
				"        tup_values[%d] = temp_%u_v.value;\n",
				tle->resno - 1,
				tle->resno,
				tle->resno - 1, tle->resno);
		}
		else if (dtype->type_byval)
		{
			appendStringInfo(
				&body,
				"      if (!temp_%u_v.isnull)\n"
				"        tup_values[%d] = temp_%u_v.value;\n",
				tle->resno,
				tle->resno - 1, tle->resno);
		}
		else if (dtype->type_length > 0)
		{
			appendStringInfo(
				&body,
				"      if (!temp_%u_v.isnull)\n"
				"      {\n"
				"        vl_buf = (char *)TYPEALIGN(%u, vl_buf);\n"
				"        tup_values[%d] = devptr_to_host(kds_dst, vl_buf);\n"
				"        memcpy(vl_buf, &temp_%u_v.value, %u);\n"
				"        vl_buf += %u;\n"
				"      }\n",
				tle->resno,
				dtype->type_align,
				tle->resno - 1,
				tle->resno, dtype->type_length,
				dtype->type_length);
		}
		else
		{
			Assert(IsA(tle->expr, Const) || IsA(tle->expr, Param));
			appendStringInfo(
				&body,
				"      if (!temp_%u_v.isnull)\n"
				"      {\n"
				"        tup_values[%d] = devptr_to_host(kcxt->kparams,\n"
				"                                        temp_%u_v.value);\n"
				"      }\n",
				tle->resno,
				tle->resno - 1, tle->resno);
		}
	}
	appendStringInfo(
		&body,
		"    }\n"
		"  }\n"
		"}\n");

	/* parameter references */
	pgstrom_codegen_param_declarations(&decl, context);

	/* OK, write back the kernel source */
	appendStringInfo(source, "%s\n%s", decl.data, body.data);
	pfree(decl.data);
	pfree(body.data);
}

/*
 * replace_varnode_with_tlist_dev - replaces the Var node in the supplied
 * expression node to reference tlist_dev, instead of the original.
 */
Node *
replace_varnode_with_tlist_dev(Node *node, List *tlist_dev)
{
	if (node == NULL)
		return NULL;

	if (IsA(node, Var))
	{
		Var		   *var = (Var *) node;
		ListCell   *lc;

		Assert(var->varlevelsup == 0);
		foreach (lc, tlist_dev)
		{
			TargetEntry	   *tle = lfirst(lc);
			Var			   *var_dev;
			Var			   *var_new;

			if (!IsA(tle->expr, Var))
				continue;
			var_dev = (Var *) tle->expr;

			Assert(var_dev->varno == var->varno);
			if (var_dev->varattno == var->varattno)
			{
				Assert(var_dev->vartype == var->vartype &&
					   var_dev->vartypmod == var->vartypmod &&
					   var_dev->varcollid == var->varcollid);

				var_new = makeVar(INDEX_VAR,
								  tle->resno,
								  var->vartype,
								  var->vartypmod,
								  var->varcollid,
								  0);
				var_new->varnoold = var->varno;
				var_new->varoattno = var->varattno;

				return (Node *) var_new;
			}
		}
		elog(ERROR, "Bug? referenced Var-node (%s) not in tlist_dev: %s",
			 nodeToString(var),
			 nodeToString(tlist_dev));
	}
	return expression_tree_mutator(node, replace_varnode_with_tlist_dev,
								   tlist_dev);
}

/*
 * add_unique_expression - adds an expression node on the supplied
 * target-list, then returns the varattno to reference the entry.
 */
AttrNumber
add_unique_expression(Expr *expr, List **p_targetlist, bool resjunk)
{
	TargetEntry	   *tle;
	ListCell	   *lc;
	AttrNumber		resno;

	foreach (lc, *p_targetlist)
	{
		tle = (TargetEntry *) lfirst(lc);
		if (equal(expr, tle->expr))
			return tle->resno;
	}
	/* Not found, so add this expression */
	resno = list_length(*p_targetlist) + 1;
	tle = makeTargetEntry(copyObject(expr), resno, NULL, resjunk);
	*p_targetlist = lappend(*p_targetlist, tle);

	return resno;
}

/*
 * build_device_projection
 *
 * It divides the original target-list into two portions; tlist_dev
 * that shall be executed on the device side, and tlist_new that shall
 * be executed on the host side on top of tlist_dev.
 * So, it expects performance improvement if target-list contains
 * complicated formula but executable on the device, because all host
 * side has to do is just reference the preliminary computed result.
 *
 * 'scanrelid'		rtindex of the base relation to be scanned
 * 'tupdesc'		Descriptor of the base relation to be scanned
 * 'tlist_old'		The original target-list
 * 'allow_hostonly' True, if caller allows host-only expression in
 *					the tlist_new.
 * 'p_tlist_new'	The tlist_new shall be set on success
 * 'p_tlist_dev'	The tlist_dev shall be set on success
 * 'p_tlist_compatible' True, if tlist_dev is compatible with the base
 *						relation, thus no actual projection is needed
 */
static bool
build_device_projection(Index scanrelid,
						TupleDesc tupdesc,
						List *tlist_old,
						bool allow_hostonly,
						List **p_tlist_new,
						List **p_tlist_dev,
						bool *p_tlist_compatible,
						bool *p_tlist_has_syscols)
{
	AttrNumber	attnum;
	ListCell   *lc;
	List	   *tlist_new = NIL;
	List	   *tlist_dev = NIL;
	bool		tlist_compatible = true;
	bool		tlist_has_syscols = false;

	attnum = 1;
	foreach (lc, tlist_old)
	{
		TargetEntry	   *tle = lfirst(lc);
		TargetEntry	   *tle_new;
		AttrNumber		varattno;

		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;

			/* if these Asserts fail, planner messed up */
			Assert(var->varno == scanrelid);
			Assert(var->varlevelsup == 0);

			/* GPU projection cannot contain whole-row reference */
			if (var->varattno == InvalidAttrNumber)
				return false;
			else if (var->varattno < 0)
				tlist_has_syscols = true;

			/*
			 * check whether the original tlist matches the physical layout
			 * of the base relation. GPU can reorder the var reference
			 * regardless of the data-type support.
			 */
			if (var->varattno != attnum || attnum > tupdesc->natts)
				tlist_compatible = false;
			else
			{
				Form_pg_attribute	attr = tupdesc->attrs[attnum - 1];

				/* How to reference the dropped column? */
				Assert(!attr->attisdropped);
				/* See the logic in tlist_matches_tupdesc */
				if (var->vartype != attr->atttypid ||
					(var->vartypmod != attr->atttypmod &&
					 var->vartypmod != -1))
					tlist_compatible = false;
			}
			/* add primitive Var-node on the tlist_dev */
			varattno = add_unique_expression((Expr *) var, &tlist_dev, false);

			/* add pseudo Var-node on the tlist_new */
			tle_new = makeTargetEntry((Expr *) makeVar(INDEX_VAR,
													   varattno,
													   var->vartype,
													   var->vartypmod,
													   var->varcollid,
													   0),
									  list_length(tlist_new) + 1,
									  tle->resname,
									  tle->resjunk);
			tlist_new = lappend(tlist_new, tle_new);
		}
		else if (pgstrom_device_expression(tle->expr))
		{
			Oid		type_oid = exprType((Node *)tle->expr);
			Oid		type_mod = exprTypmod((Node *)tle->expr);
			Oid		coll_oid = exprCollation((Node *)tle->expr);

			/* Add device executable expression onto the tlist_dev */
			varattno = add_unique_expression(tle->expr, &tlist_dev, false);

			/* Then, CPU just referenced the calculation result */
			tle_new = makeTargetEntry((Expr *) makeVar(INDEX_VAR,
													   varattno,
													   type_oid,
													   type_mod,
													   coll_oid,
													   0),
									  list_length(tlist_new) + 1,
									  tle->resname,
									  tle->resjunk);
			tlist_new = lappend(tlist_new, tle_new);

			/* obviously, we need GPU projection */
			tlist_compatible = false;
		}
		else
		{
			/* Elsewhere, expression is not device executable */
			Bitmapset  *varattnos = NULL;
			int			prev = -1;
			Node	   *expr_new;

			/* Does caller allow host-only expression? */
			if (!allow_hostonly)
				return false;

			/* 1. Pull varnodes within the expression*/
			pull_varattnos((Node *) tle->expr, scanrelid, &varattnos);

			/* 2. Add varnodes to tlist_dev */
			while ((prev = bms_next_member(varattnos, prev)) >= 0)
			{
				Form_pg_attribute	attr;
				int		anum = prev + FirstLowInvalidHeapAttributeNumber;

				/* GPU projection cannot contain whole-row reference */
				if (anum == InvalidAttrNumber)
					return false;

				/* Add varnodes to tlist_dev */
				if (anum > 0)
					attr = tupdesc->attrs[anum - 1];
				else
				{
					attr = SystemAttributeDefinition(anum, true);
					tlist_has_syscols = true;
				}
				add_unique_expression((Expr *) makeVar(scanrelid,
													   attr->attnum,
													   attr->atttypid,
													   attr->atttypmod,
													   attr->attcollation,
													   0),
									  &tlist_dev,
									  false);
			}
			/* 3. replace varnode of the expression */
			expr_new = replace_varnode_with_tlist_dev((Node *) tle->expr,
													  tlist_dev);
			/* 4. add modified expression to tlist_new */
			tle_new = makeTargetEntry((Expr *)expr_new,
									  list_length(tlist_new) + 1,
									  tle->resname,
									  tle->resjunk);
			tlist_new = lappend(tlist_new, tle_new);

			/* obviously, we need GPU projection */
			tlist_compatible = false;
		}
		attnum++;
	}

	/*
	 * tlist_old has smaller number of attributes than base relation's
	 * definition.
	 */
	attnum--;
	if (attnum != tupdesc->natts)
		tlist_compatible = false;

	/*
	 * NOTE: The reason why we don't need to add target-entries with
	 * resjunk==true is, all the varnodes in host-/device-qualifiers
	 * references the base relation, thus we don't need special
	 * translation for them.
	 */

	/* put results */
	*p_tlist_new = tlist_new;
	*p_tlist_dev = tlist_dev;
	*p_tlist_compatible = tlist_compatible;
	*p_tlist_has_syscols = tlist_has_syscols;

	return true;
}

/*
 * pgstrom_post_planner_gpuscan
 *
 * Applies projection of GpuScan if needed.
 */
void
pgstrom_post_planner_gpuscan(PlannedStmt *pstmt, Plan **p_curr_plan)
{
	Plan		   *plan = *p_curr_plan;
	CustomScan	   *cscan = (CustomScan *) (plan);
	GpuScanInfo	   *gs_info;
	RangeTblEntry  *rte;
	Relation		baserel;
	TupleDesc		tupdesc;
	codegen_context	context;
	List		   *tlist_new = NIL;
	List		   *tlist_dev = NIL;
	bool			tlist_compatible;
	bool			tlist_has_syscols;
	bool			force_row_format = false;
	ListCell	   *lc;
	QualCost		qcost;
	Cost			actual_old_cost = 0.0;
	Cost			actual_new_cost = 0.0;
	AttrNumber		attnum;

	/* quick bailout if not supported anyway */
	if (!pgstrom_plan_is_gpuscan(plan) && !IsA(plan, SeqScan))
		return;

	/* compute 'actual' cost that includes evaluation of targetlist */
	cost_qual_eval(&qcost, plan->targetlist, NULL);
	actual_old_cost = plan->total_cost
		+ qcost.startup
		+ qcost.per_tuple * plan->plan_rows;

	/*
	 * This plan is at least either of GpuScan or SeqScan.
	 * For more optimal plan by surgery, we try to open the relation
	 * and tries to replace SeqScan by GpuScan if reasonable.
	 */
	rte = rt_fetch(cscan->scan.scanrelid, pstmt->rtable);
	Assert(rte->rtekind == RTE_RELATION);
	baserel = heap_open(rte->relid, NoLock);
	tupdesc = RelationGetDescr(baserel);

	/* extract private fields */
	gs_info = deform_gpuscan_info(cscan);

	pgstrom_init_codegen_context(&context);
	context.extra_flags = gs_info->extra_flags;	/* restore */
	context.func_defs = gs_info->func_defs;		/* restore */
	context.expr_defs = gs_info->expr_defs;		/* restore */
	context.used_params = gs_info->used_params;	/* restore */
	context.used_vars = gs_info->used_vars;		/* restore */

	if (build_device_projection(cscan->scan.scanrelid,
								tupdesc,
								cscan->scan.plan.targetlist,
								true,	/* allow host-only */
								&tlist_new,
								&tlist_dev,
								&tlist_compatible,
								&tlist_has_syscols) &&
		!tlist_compatible)		/* device projection does not make much
								 * sense on the compatible base relation */
	{
		cl_int		base_fixed_width = 0;
		cl_int		proj_fixed_width = 0;
		cl_int		proj_extra_width = 0;

		/*
		 * Count width of the fixed-length fields for simple/indirect types,
		 * and width of the fixed-length fields of the base relation also.
		 */
		for (attnum = 1; attnum < tupdesc->natts; attnum++)
		{
			Form_pg_attribute	attr = tupdesc->attrs[attnum - 1];

			if (attr->attlen >= 0)
				base_fixed_width += attr->attlen;
		}

		foreach (lc, tlist_dev)
		{
			TargetEntry	   *tle = lfirst(lc);
			Oid				type_oid = exprType((Node *) tle->expr);
			int16			type_len;
			bool			type_byval;

			if (type_oid == NUMERICOID)
			{
				proj_fixed_width += 32;	// XXX - TO BE FIXED LATER
				proj_extra_width += 32;	// XXX - TO BE FIXED LATER
			}
			else
			{
				get_typlenbyval(type_oid, &type_len, &type_byval);
				if (type_len < 0)
					continue;
				proj_fixed_width += type_len;
				if (!type_byval)
					proj_extra_width += type_len;
			}
		}

		/*
		 * compute 'actual' cost based on device projection
		 */
		actual_new_cost = cscan->scan.plan.total_cost;
		/* cost for device projection */
		cost_qual_eval(&qcost, tlist_dev, NULL);
		qcost.per_tuple *= pgstrom_gpu_operator_cost / cpu_operator_cost;
		actual_new_cost += (qcost.startup +
							qcost.per_tuple * plan->plan_rows);
		/* cost for device projection */
		cost_qual_eval(&qcost, tlist_new, NULL);
		actual_new_cost += (qcost.startup +
							qcost.per_tuple * plan->plan_rows);

		/* if GpuScan is actually cheaper than original, replace it */
		if (actual_new_cost < actual_old_cost)
		{
			StringInfoData	kern;

			/* update GpuScanInfo to support device projection */
			initStringInfo(&kern);
			appendStringInfoString(&kern, gs_info->kern_source);
			codegen_device_projection(&kern,
									  &context,
									  cscan->scan.scanrelid,
									  tupdesc,
									  tlist_dev);
			gs_info->kern_source = kern.data;
			gs_info->base_fixed_width = base_fixed_width;
			gs_info->proj_fixed_width = proj_fixed_width;
			gs_info->proj_extra_width = proj_extra_width;

			/* update targetlist */
			cscan->scan.plan.targetlist = tlist_new;
			cscan->custom_scan_tlist = tlist_dev;

			if (*p_curr_plan != &cscan->scan.plan)
				*p_curr_plan = &cscan->scan.plan;
		}
		else if (tlist_has_syscols)
			force_row_format = true;
	}
	else
		force_row_format = true;

	/*
	 * Declaration of functions/special expressions
	 */
	if (context.func_defs != NULL || context.expr_defs != NULL)
	{
		StringInfoData	kern;

		initStringInfo(&kern);
		pgstrom_codegen_func_declarations(&kern, &context);
		pgstrom_codegen_expr_declarations(&kern, &context);
		appendStringInfoString(&kern, gs_info->kern_source);
		gs_info->kern_source = kern.data;
	}
	gs_info->extra_flags = context.extra_flags;
	gs_info->func_defs = context.func_defs;
	gs_info->expr_defs = context.expr_defs;
	gs_info->used_params = context.used_params;
	gs_info->used_vars = context.used_vars;
	gs_info->force_row_format = force_row_format;
	form_gpuscan_info(cscan, gs_info);

	heap_close(baserel, NoLock);
}

/*
 * assign_gpuscan_session_info
 *
 * Gives some definitions to the static portion of GpuScan implementation
 */
void
assign_gpuscan_session_info(StringInfo buf, GpuTaskState *gts)
{
	CustomScan *cscan = (CustomScan *)gts->css.ss.ps.plan;

	Assert(pgstrom_plan_is_gpuscan((Plan *) cscan));

	if (cscan->custom_scan_tlist != NIL)
	{
		appendStringInfo(
			buf,
			"#define GPUSCAN_DEVICE_PROJECTION          1\n"
			"#define GPUSCAN_DEVICE_PROJECTION_NFIELDS  %d\n\n",
			list_length(cscan->custom_scan_tlist));
	}
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

	/* Set tag and executor callbacks */
	NodeSetTag(gss, T_CustomScanState);
	gss->gts.css.flags = cscan->flags;
	if (cscan->methods == &gpuscan_plan_methods)
		gss->gts.css.methods = &gpuscan_exec_methods;
	else
		elog(ERROR, "Bug? unexpected CustomPlanMethods");

	return (Node *) gss;
}

static void
gpuscan_begin(CustomScanState *node, EState *estate, int eflags)
{
	Relation		scan_rel = node->ss.ss_currentRelation;
	GpuContext	   *gcontext = NULL;
	GpuScanState   *gss = (GpuScanState *) node;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	GpuScanInfo	   *gs_info = deform_gpuscan_info(cscan);

	/* gpuscan should not have inner/outer plan right now */
	Assert(outerPlan(node) == NULL);
	Assert(innerPlan(node) == NULL);

	/* activate GpuContext for device execution */
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		gcontext = pgstrom_get_gpucontext();
	/* setup common GpuTaskState fields */
	pgstrom_init_gputaskstate(gcontext, &gss->gts, estate);
	gss->gts.cb_task_process = pgstrom_process_gpuscan;
	gss->gts.cb_task_complete = pgstrom_complete_gpuscan;
	gss->gts.cb_task_release = pgstrom_release_gpuscan;
	gss->gts.cb_next_chunk = gpuscan_next_chunk;
	gss->gts.cb_next_tuple = gpuscan_next_tuple;

	/* Per chunk execution supported? */
	if (pgstrom_bulkexec_enabled &&
		gss->gts.css.ss.ps.qual == NIL &&
		gss->gts.css.ss.ps.ps_ProjInfo == NULL)
		gss->gts.cb_bulk_exec = pgstrom_exec_chunk_gputask;

	/* Is 'row' format required? */
	if (gs_info->force_row_format)
		gss->gts.be_row_format = true;

	/* initialize device tlist for CPU fallback */
	gss->dev_tlist = (List *)
		ExecInitExpr((Expr *) cscan->custom_scan_tlist, &gss->gts.css.ss.ps);
	/* initialize device qualifiers also, for fallback */
	gss->dev_quals = (List *)
		ExecInitExpr((Expr *) gs_info->dev_quals, &gss->gts.css.ss.ps);
	/* true, if device projection is needed */
	gss->dev_projection = (cscan->custom_scan_tlist != NIL);
	/* device projection related resource consumption */
	gss->base_fixed_width = gs_info->base_fixed_width;
	gss->proj_fixed_width = gs_info->proj_fixed_width;
	gss->proj_extra_width = gs_info->proj_extra_width;
	/* 'tableoid' should not change during relation scan */
	gss->scan_tuple.t_tableOid = RelationGetRelid(scan_rel);
	/* assign kernel source and flags */
	pgstrom_assign_cuda_program(&gss->gts,
								gs_info->used_params,
								gs_info->kern_source,
								gs_info->extra_flags);
	/* preload the CUDA program, if actually executed */
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		pgstrom_load_cuda_program(&gss->gts, true);
	/* initialize resource for CPU fallback */
	gss->base_slot = MakeSingleTupleTableSlot(RelationGetDescr(scan_rel));
	if (gss->dev_projection)
	{
		ExprContext	   *econtext = gss->gts.css.ss.ps.ps_ExprContext;
		TupleTableSlot *scan_slot = gss->gts.css.ss.ss_ScanTupleSlot;

		gss->base_proj = ExecBuildProjectionInfo(gss->dev_tlist,
												 econtext,
												 scan_slot,
												 RelationGetDescr(scan_rel));
	}
	else
		gss->base_proj = NULL;
	/* init perfmon */
	pgstrom_init_perfmon(&gss->gts);
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

	if (gpuscan->pds_src)
		PDS_release(gpuscan->pds_src);
	if (gpuscan->pds_dst)
		PDS_release(gpuscan->pds_dst);
	pgstrom_complete_gpuscan(&gpuscan->task);

	pfree(gpuscan);
}

static pgstrom_gpuscan *
create_pgstrom_gpuscan_task(GpuScanState *gss, pgstrom_data_store *pds_src)
{
	TupleDesc			scan_tupdesc = GTS_GET_SCAN_TUPDESC(gss);
	GpuContext		   *gcontext = gss->gts.gcontext;
	pgstrom_gpuscan    *gpuscan;
	kern_resultbuf	   *kresults;
	kern_data_store	   *kds_src = pds_src->kds;
	pgstrom_data_store *pds_dst;
	Size				length;

	/*
	 * allocation of the destination buffer
	 */
	if (gss->gts.be_row_format)
	{
		/*
		 * NOTE: When we have no device projection and row-format
		 * is required, we don't need to have destination buffer.
		 * kern_resultbuf will have offset of the visible rows,
		 * so we can reference pds_src as original PG-Strom did.
		 */
		if (!gss->dev_projection)
			pds_dst = NULL;
		else
		{
			length = (kds_src->length +
					  Max(gss->proj_fixed_width -
						  gss->base_fixed_width, 0) * kds_src->nitems);
			pds_dst = PDS_create_row(gcontext,
									 scan_tupdesc,
									 length);
		}
	}
	else
	{
		length = gss->proj_extra_width * kds_src->nitems;
		pds_dst = PDS_create_slot(gcontext,
								  scan_tupdesc,
								  kds_src->nitems,
								  length,
								  false);
	}

	/*
	 * allocation of pgstrom_gpuscan
	 */
	length = (STROMALIGN(offsetof(pgstrom_gpuscan, kern.kparams)) +
			  STROMALIGN(gss->gts.kern_params->length) +
			  STROMALIGN(offsetof(kern_resultbuf,
								  results[pds_dst ? 0 : kds_src->nitems])));
	gpuscan = MemoryContextAllocZero(gcontext->memcxt, length);
	/* setting up */
	pgstrom_init_gputask(&gss->gts, &gpuscan->task);

	gpuscan->pds_src = pds_src;
	gpuscan->pds_dst = pds_dst;

	/* setting up kern_parambuf */
	memcpy(KERN_GPUSCAN_PARAMBUF(&gpuscan->kern),
		   gss->gts.kern_params,
		   gss->gts.kern_params->length);
	/* setting up kern_resultbuf */
	kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
    memset(kresults, 0, sizeof(kern_resultbuf));
    kresults->nrels = 1;
	if (gss->dev_quals != NIL)
		kresults->nrooms = kds_src->nitems;
	else
		kresults->all_visible = true;
	gpuscan->kresults = kresults;

	return gpuscan;
}

/*
 * pgstrom_exec_scan_chunk
 *
 * It makes advance the scan pointer of the relation.
 */
pgstrom_data_store *
pgstrom_exec_scan_chunk(GpuTaskState *gts, Size chunk_length)
{
	Relation		base_rel = gts->css.ss.ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(base_rel);
	HeapScanDesc	scan = gts->css.ss.ss_currentScanDesc;
	pgstrom_data_store *pds = NULL;
	bool			finished = false;
	struct timeval	tv1, tv2;

	/* return NULL if relation is empty */
	if (scan->rs_nblocks == 0 || scan->rs_numblocks == 0)
		return NULL;

	if (scan->rs_cblock == InvalidBlockNumber)
		scan->rs_cblock = scan->rs_startblock;
	else if (scan->rs_cblock == scan->rs_startblock)
		return NULL;	/* already goes around the relation */
	Assert(scan->rs_cblock < scan->rs_nblocks);

	InstrStartNode(&gts->outer_instrument);
	PERFMON_BEGIN(&gts->pfm, &tv1);
	pds = PDS_create_row(gts->gcontext,
						 tupdesc,
						 chunk_length);
	pds->kds->table_oid = RelationGetRelid(base_rel);

	/*
	 * TODO: We have to stop block insert if and when device projection
	 * will increase the buffer consumption than threshold.
	 * OR,
	 * specify smaller chunk by caller. GpuScan may become wise using
	 * adaptive buffer size control by row selevtivity on run-time.
	 */

	/* fill up this data-store */
	while (!finished)
	{
		if (PDS_insert_block(pds, base_rel,
							 scan->rs_cblock,
							 scan->rs_snapshot,
							 scan->rs_strategy) < 0)
			break;

		/* move to the next block */
		scan->rs_cblock++;
		if (scan->rs_cblock >= scan->rs_nblocks)
			scan->rs_cblock = 0;
		if (scan->rs_syncscan)
			ss_report_location(scan->rs_rd, scan->rs_cblock);
		/* end of the scan? */
		if (scan->rs_cblock == scan->rs_startblock ||
			(scan->rs_numblocks != InvalidBlockNumber &&
			 --scan->rs_numblocks == 0))
			break;
	}

	if (pds->kds->nitems == 0)
	{
		PDS_release(pds);
		pds = NULL;
	}
	PERFMON_END(&gts->pfm, time_outer_load, &tv1, &tv2);
	InstrStopNode(&gts->outer_instrument,
				  !pds ? 0.0 : (double)pds->kds->nitems);
	return pds;
}

/*
 * pgstrom_rewind_scan_chunk - rewind the position to read
 */
void
pgstrom_rewind_scan_chunk(GpuTaskState *gts)
{
	InstrEndLoop(&gts->outer_instrument);
	Assert(gts->css.ss.ss_currentRelation != NULL);
	heap_rescan(gts->css.ss.ss_currentScanDesc, NULL);
	ExecScanReScan(&gts->css.ss);
}

static GpuTask *
gpuscan_next_chunk(GpuTaskState *gts)
{
	GpuScanState	   *gss = (GpuScanState *) gts;
	pgstrom_gpuscan	   *gpuscan;
	pgstrom_data_store *pds;

	pds = pgstrom_exec_scan_chunk(gts, pgstrom_chunk_size());
	if (!pds)
		return NULL;

	gpuscan = create_pgstrom_gpuscan_task(gss, pds);
	return &gpuscan->task;
}

static TupleTableSlot *
gpuscan_next_tuple(GpuTaskState *gts)
{
	GpuScanState	   *gss = (GpuScanState *) gts;
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *) gts->curr_task;
	TupleTableSlot	   *slot = NULL;
	struct timeval		tv1, tv2;

	PERFMON_BEGIN(&gss->gts.pfm, &tv1);
	if (!gpuscan->task.cpu_fallback)
	{
		if (gpuscan->pds_dst)
		{
			pgstrom_data_store *pds_dst = gpuscan->pds_dst;

			if (gss->gts.curr_index < pds_dst->kds->nitems)
			{
				slot = gss->gts.css.ss.ss_ScanTupleSlot;
				ExecClearTuple(slot);
				if (!pgstrom_fetch_data_store(slot, pds_dst,
											  gss->gts.curr_index++,
											  &gss->scan_tuple))
					elog(ERROR, "failed to fetch a record from pds");
			}
		}
		else
		{
			pgstrom_data_store *pds_src = gpuscan->pds_src;
			kern_resultbuf	   *kresults = gpuscan->kresults;

			/*
			 * We should not inject GpuScan for all-visible with no device
			 * projection; GPU has no actual works in other words.
			 * NOTE: kresults->results[] keeps offset from the head of
			 * kds_src.
			 */
			Assert(!kresults->all_visible);
			if (gss->gts.curr_index < kresults->nitems)
			{
				HeapTuple		tuple = &gss->scan_tuple;
				kern_tupitem   *tupitem = (kern_tupitem *)
					((char *)pds_src->kds +
					 kresults->results[gss->gts.curr_index++]);

				slot = gss->gts.css.ss.ss_ScanTupleSlot;
				tuple->t_len = tupitem->t_len;
				tuple->t_self = tupitem->t_self;
				tuple->t_data = &tupitem->htup;
				ExecStoreTuple(tuple, slot, InvalidBuffer, false);
			}
		}
	}
	else
	{
		/*
		 * If GPU kernel returned StromError_CpuReCheck, we have to
		 * evaluate dev_quals by ourselves, then adjust tuple format
		 * according to custom_scan_tlist.
		 */
		pgstrom_data_store *pds_src = gpuscan->pds_src;

		while (gss->gts.curr_index < pds_src->kds->nitems)
		{
			cl_uint			index = gss->gts.curr_index++;
			ExprContext	   *econtext = gss->gts.css.ss.ps.ps_ExprContext;
			ExprDoneCond	is_done;

			ExecClearTuple(gss->base_slot);
			if (!pgstrom_fetch_data_store(gss->base_slot, pds_src, index,
										  &gss->scan_tuple))
				elog(ERROR, "failed to fetch a record from pds");

			ResetExprContext(econtext);
			econtext->ecxt_scantuple = gss->base_slot;

			/*
			 * step.1 - evaluate dev_quals if any
			 */
			if (gss->dev_quals != NIL)
			{
				if (!ExecQual(gss->dev_quals, econtext, false))
					continue;
			}

			/*
			 * step.2 - makes a projection if any
			 */
			if (gss->base_proj == NULL)
				slot = gss->base_slot;
			else
			{
				slot = ExecProject(gss->base_proj, &is_done);
				if (is_done == ExprEndResult)
				{
					/* tuple fails qual, so free per-tuple memory and try
					 * again.
					 * XXX - Is logic really right? needs to be checked */
					ResetExprContext(econtext);
					slot = NULL;
					continue;
				}
			}
			break;
		}
	}
	PERFMON_END(&gss->gts.pfm, time_materialize, &tv1, &tv2);

	return slot;
}

/*
 * gpuscan_exec_recheck
 *
 * Routine of EPQ recheck on GpuScan. If any, HostQual shall be checked
 * on ExecScan(), all we have to do here is recheck of device qualifier.
 */
static bool
gpuscan_exec_recheck(CustomScanState *node, TupleTableSlot *slot)
{
	GpuScanState   *gss = (GpuScanState *) node;
	ExprContext	   *econtext = node->ss.ps.ps_ExprContext;
	HeapTuple		tuple = slot->tts_tuple;
	TupleTableSlot *scan_slot	__attribute__((unused));
	ExprDoneCond	is_done;

	/*
	 * Does the tuple meet the device qual condition?
	 * Please note that we should not use the supplied 'slot' as is,
	 * because it may not be compatible with relations's definition
	 * if device projection is valid.
	 */
	ExecStoreTuple(tuple, gss->base_slot, InvalidBuffer, false);
	econtext->ecxt_scantuple = gss->base_slot;
	ResetExprContext(econtext);

	if (!ExecQual(gss->dev_quals, econtext, false))
		return false;

	if (gss->base_proj)
	{
		/*
		 * NOTE: If device projection is valid, we have to adjust the
		 * supplied tuple (that follows the base relation's definition)
		 * into ss_ScanTupleSlot, to fit tuple descriptor of the supplied
		 * 'slot'.
		 */
		Assert(!slot->tts_shouldFree);
		ExecClearTuple(slot);

		scan_slot = ExecProject(gss->base_proj, &is_done);
		Assert(scan_slot == slot);
	}
	return true;
}

static TupleTableSlot *
gpuscan_exec(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstrom_exec_gputask,
					(ExecScanRecheckMtd) gpuscan_exec_recheck);
}

static void
gpuscan_end(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *)node;

	/* reset fallback resources */
	if (gss->base_slot)
		ExecDropSingleTupleTableSlot(gss->base_slot);
	pgstrom_release_gputaskstate(&gss->gts);
}

static void
gpuscan_rescan(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *) node;

	/* activate GpuTaskState first, not to release pinned memory */
	pgstrom_activate_gputaskstate(&gss->gts);
	/* clean-up and release any concurrent tasks */
    pgstrom_cleanup_gputaskstate(&gss->gts);
	/* OK, rewind the position to read */
	pgstrom_rewind_scan_chunk(&gss->gts);
}

static void
gpuscan_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *) node;
	CustomScan	   *cscan = (CustomScan *) gss->gts.css.ss.ps.plan;
	GpuScanInfo	   *gsinfo = deform_gpuscan_info(cscan);
	List		   *context;
	List		   *dev_proj = NIL;
	ListCell	   *lc;

	/* Set up deparsing context */
	context = set_deparse_context_planstate(es->deparse_cxt,
											(Node *)&gss->gts.css.ss.ps,
											ancestors);
	/* Show device projection */
	foreach (lc, cscan->custom_scan_tlist)
		dev_proj = lappend(dev_proj, ((TargetEntry *) lfirst(lc))->expr);
	pgstrom_explain_expression(dev_proj, "GPU Projection",
							   &gss->gts.css.ss.ps, context,
							   ancestors, es, false, false);
	/* Show device filter */
	pgstrom_explain_expression(gsinfo->dev_quals, "GPU Filter",
							   &gss->gts.css.ss.ps, context,
                               ancestors, es, false, true);
	// TODO: Add number of rows filtered by the device side

	pgstrom_explain_gputaskstate(&gss->gts, es);
}

void
pgstrom_init_gpuscan(void)
{
	/* pg_strom.enable_gpuscan */
	DefineCustomBoolVariable("pg_strom.enable_gpuscan",
							 "Enables the use of GPU accelerated full-scan",
							 NULL,
							 &enable_gpuscan,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.pullup_outer_scan */
	DefineCustomBoolVariable("pg_strom.pullup_outer_scan",
							 "Enables to pull up simple outer scan",
							 NULL,
							 &enable_pullup_outer_scan,
							 true,
							 PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);

	/* setup path methods */
	memset(&gpuscan_path_methods, 0, sizeof(gpuscan_path_methods));
	gpuscan_path_methods.CustomName			= "GpuScan";
	gpuscan_path_methods.PlanCustomPath		= create_gpuscan_plan;

	/* setup plan methods */
	memset(&gpuscan_plan_methods, 0, sizeof(gpuscan_plan_methods));
	gpuscan_plan_methods.CustomName			= "GpuScan";
	gpuscan_plan_methods.CreateCustomScanState = gpuscan_create_scan_state;

	/* setup exec methods */
	memset(&gpuscan_exec_methods, 0, sizeof(gpuscan_exec_methods));
	gpuscan_exec_methods.CustomName         = "GpuScan";
	gpuscan_exec_methods.BeginCustomScan    = gpuscan_begin;
	gpuscan_exec_methods.ExecCustomScan     = gpuscan_exec;
	gpuscan_exec_methods.EndCustomScan      = gpuscan_end;
	gpuscan_exec_methods.ReScanCustomScan   = gpuscan_rescan;
	gpuscan_exec_methods.ExplainCustomScan  = gpuscan_explain;

	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = gpuscan_add_scan_path;
}

static void
gpuscan_cleanup_cuda_resources(pgstrom_gpuscan *gpuscan)
{
	CUDA_EVENT_DESTROY(gpuscan,ev_dma_recv_stop);
	CUDA_EVENT_DESTROY(gpuscan,ev_dma_recv_start);
	CUDA_EVENT_DESTROY(gpuscan,ev_kern_exec_quals);
	CUDA_EVENT_DESTROY(gpuscan,ev_dma_send_stop);
	CUDA_EVENT_DESTROY(gpuscan,ev_dma_send_start);

	if (gpuscan->m_gpuscan)
		gpuMemFree(&gpuscan->task, gpuscan->m_gpuscan);

	/* ensure pointers being NULL */
	gpuscan->kern_exec_quals = NULL;
	gpuscan->kern_dev_proj = NULL;
	gpuscan->m_gpuscan = 0UL;
	gpuscan->m_kds_src = 0UL;
	gpuscan->m_kds_dst = 0UL;
}

/*
 * pgstrom_complete_gpuscan
 *
 *
 *
 *
 */
static bool
pgstrom_complete_gpuscan(GpuTask *gtask)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *) gtask;
	GpuTaskState	   *gts = gtask->gts;

	if (gts->pfm.enabled)
	{
		gts->pfm.num_tasks++;
		CUDA_EVENT_ELAPSED(gpuscan,time_dma_send,
						   gpuscan->ev_dma_send_start,
						   gpuscan->ev_dma_send_stop,
						   skip);
		CUDA_EVENT_ELAPSED(gpuscan, gscan.tv_kern_exec_quals,
						   gpuscan->ev_dma_send_stop,
						   gpuscan->ev_kern_exec_quals,
						   skip);
		CUDA_EVENT_ELAPSED(gpuscan, gscan.tv_kern_projection,
						   gpuscan->ev_kern_exec_quals,
						   gpuscan->ev_dma_recv_start,
						   skip);
		CUDA_EVENT_ELAPSED(gpuscan, time_dma_recv,
						   gpuscan->ev_dma_recv_start,
						   gpuscan->ev_dma_recv_stop,
						   skip);
	}
skip:
	gpuscan_cleanup_cuda_resources(gpuscan);

	return true;
}

static void
pgstrom_respond_gpuscan(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpuscan	   *gpuscan = private;
	GpuTaskState	   *gts = gpuscan->task.gts;

	/*
	 * NOTE: We need to pay careful attention for invocation timing of
	 * the callback registered via cuStreamAddCallback(). This routine
	 * shall be called on the non-master thread which is managed by CUDA
	 * runtime, so here is no guarantee resources are available.
	 * Once a transaction gets aborted, PostgreSQL backend takes a long-
	 * junk to the point where sigsetjmp(), then releases resources that
	 * is allocated for each transaction.
	 * Per-query memory context (estate->es_query_cxt) shall be released
	 * during AbortTransaction(), then CUDA context shall be also destroyed
	 * on the ResourceReleaseCallback().
	 * It means, this respond callback may be kicked, by CUDA runtime,
	 * concurrently, however, either/both of GpuTaskState or/and CUDA context
	 * may be already gone.
	 * So, prior to touch these resources, we need to ensure the resources
	 * are still valid.
	 *
	 * FIXME: Once IsTransactionState() returned 'true', transaction may be
	 * aborted during the rest of tasks. We need more investigation to
	 * ensure GpuTaskState is not released here...
	 *
	 * If CUDA runtime gives CUDA_ERROR_INVALID_CONTEXT, it implies CUDA
	 * context is already released. So, we should bail-out immediately.
	 * Also, once transaction state gets turned off from TRANS_INPROGRESS,
	 * it implies per-query memory context will be released very soon.
	 * So, we also need to bail-out immediately.
	 */
	if (status == CUDA_ERROR_INVALID_CONTEXT || !IsTransactionState())
		return;

	/* OK, routine is called back in the usual context */
	if (status == CUDA_SUCCESS)
	{
		gpuscan->task.kerror = gpuscan->kern.kerror;
		if (pgstrom_cpu_fallback_enabled &&
			(gpuscan->task.kerror.errcode == StromError_CpuReCheck ||
			 gpuscan->task.kerror.errcode == StromError_DataStoreNoSpace))
		{
			/* clear the error instead of the CPU fallback */
			gpuscan->task.kerror.errcode = StromError_Success;
			gpuscan->task.cpu_fallback = true;
		}
	}
	else
	{
		gpuscan->task.kerror.errcode = status;
		gpuscan->task.kerror.kernel = StromKernel_CudaRuntime;
		gpuscan->task.kerror.lineno = 0;
	}

	/*
	 * Remove the GpuTask from the running_tasks list, and attach it
	 * on the completed_tasks list again. Note that this routine may
	 * be called by CUDA runtime, prior to attachment of GpuTask on
	 * the running_tasks by cuda_control.c.
	 */
	SpinLockAcquire(&gts->lock);
	if (gpuscan->task.chain.prev && gpuscan->task.chain.next)
	{
		dlist_delete(&gpuscan->task.chain);
		gts->num_running_tasks--;
	}
	if (gpuscan->task.kerror.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &gpuscan->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &gpuscan->task.chain);
	gts->num_completed_tasks++;
	SpinLockRelease(&gts->lock);

	SetLatch(&MyProc->procLatch);
}

static bool
__pgstrom_process_gpuscan(pgstrom_gpuscan *gpuscan)
{
	GpuScanState	   *gss = (GpuScanState *) gpuscan->task.gts;
	pgstrom_data_store *pds_src = gpuscan->pds_src;
	pgstrom_data_store *pds_dst = gpuscan->pds_dst;
	cl_uint				src_nitems = pds_src->kds->nitems;
	void			   *kern_args[5];
	size_t				offset;
	size_t				length;
	size_t				grid_size;
	size_t				block_size;
	CUresult			rc;

	/*
	 * GPU kernel function lookup
	 */
	rc = cuModuleGetFunction(&gpuscan->kern_exec_quals,
							 gpuscan->task.cuda_module,
							 "gpuscan_exec_quals");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/* we don't need projection kernel without destination buffer */
	if (pds_dst != NULL)
	{
		rc = cuModuleGetFunction(&gpuscan->kern_dev_proj,
								 gpuscan->task.cuda_module,
								 gss->gts.be_row_format
								 ? "gpuscan_projection_row"
								 : "gpuscan_projection_slot");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));
	}

	/*
	 * Allocation of device memory
	 */
	length = (GPUMEMALIGN(KERN_GPUSCAN_LENGTH(&gpuscan->kern)) +
			  GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_src->kds)));
	if (pds_dst)
		length += GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_dst->kds));

	gpuscan->m_gpuscan = gpuMemAlloc(&gpuscan->task, length);
	if (!gpuscan->m_gpuscan)
		goto out_of_resource;

	gpuscan->m_kds_src = gpuscan->m_gpuscan +
		GPUMEMALIGN(KERN_GPUSCAN_LENGTH(&gpuscan->kern));

	if (pds_dst)
		gpuscan->m_kds_dst = gpuscan->m_kds_src +
			GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_src->kds));
	else
		gpuscan->m_kds_dst = 0UL;

	/*
	 * Creation of event objects, if any
	 */
	CUDA_EVENT_CREATE(gpuscan, ev_dma_send_start);
	CUDA_EVENT_CREATE(gpuscan, ev_dma_send_stop);
	CUDA_EVENT_CREATE(gpuscan, ev_kern_exec_quals);
	CUDA_EVENT_CREATE(gpuscan, ev_dma_recv_start);
	CUDA_EVENT_CREATE(gpuscan, ev_dma_recv_stop);

	/*
	 * OK, enqueue a series of requests
	 */
	CUDA_EVENT_RECORD(gpuscan, ev_dma_send_start);

	offset = KERN_GPUSCAN_DMASEND_OFFSET(&gpuscan->kern);
	length = KERN_GPUSCAN_DMASEND_LENGTH(&gpuscan->kern);
	rc = cuMemcpyHtoDAsync(gpuscan->m_gpuscan,
						   (char *)&gpuscan->kern + offset,
						   length,
						   gpuscan->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gss->gts.pfm.bytes_dma_send += length;
	gss->gts.pfm.num_dma_send++;

	/* kern_data_store *kds_src */
	length = KERN_DATA_STORE_LENGTH(pds_src->kds);
	rc = cuMemcpyHtoDAsync(gpuscan->m_kds_src,
						   pds_src->kds,
						   length,
						   gpuscan->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gss->gts.pfm.bytes_dma_send += length;
	gss->gts.pfm.num_dma_send++;

	/* kern_data_store *kds_dst, if any */
	if (pds_dst)
	{
		length = KERN_DATA_STORE_HEAD_LENGTH(pds_dst->kds);
		rc = cuMemcpyHtoDAsync(gpuscan->m_kds_dst,
							   pds_dst->kds,
							   length,
							   gpuscan->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		gss->gts.pfm.bytes_dma_send += length;
		gss->gts.pfm.num_dma_send++;
	}
	CUDA_EVENT_RECORD(gpuscan, ev_dma_send_stop);

	/*
	 * Launch kernel function
	 */
	if (gss->dev_quals != NIL)
	{
		optimal_workgroup_size(&grid_size,
							   &block_size,
							   gpuscan->kern_exec_quals,
							   gpuscan->task.cuda_device,
							   src_nitems,
							   0, sizeof(kern_errorbuf));
		kern_args[0] = &gpuscan->m_gpuscan;
		kern_args[1] = &gpuscan->m_kds_src;

		rc = cuLaunchKernel(gpuscan->kern_exec_quals,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(kern_errorbuf) * block_size,
							gpuscan->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		gss->gts.pfm.gscan.num_kern_exec_quals++;
	}
	else
	{
		/* no device qualifiers, thus, all rows are visible to projection */
		Assert(KERN_GPUSCAN_RESULTBUF(&gpuscan->kern)->all_visible);
	}
	CUDA_EVENT_RECORD(gpuscan, ev_kern_exec_quals);

	if (pds_dst != NULL)
	{
		optimal_workgroup_size(&grid_size,
							   &block_size,
							   gpuscan->kern_dev_proj, 
							   gpuscan->task.cuda_device,
							   src_nitems,
							   0, sizeof(kern_errorbuf));
		kern_args[0] = &gpuscan->m_gpuscan;
		kern_args[1] = &gpuscan->m_kds_src;
		kern_args[2] = &gpuscan->m_kds_dst;

		rc = cuLaunchKernel(gpuscan->kern_dev_proj,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(kern_errorbuf) * block_size,
							gpuscan->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		gss->gts.pfm.gscan.num_kern_projection++;
	}

	/*
	 * Recv DMA call
	 */
	CUDA_EVENT_RECORD(gpuscan, ev_dma_recv_start);

	offset = KERN_GPUSCAN_DMARECV_OFFSET(&gpuscan->kern);
	length = KERN_GPUSCAN_DMARECV_LENGTH(&gpuscan->kern,
										 pds_dst ? 0 : pds_src->kds->nitems);
	rc = cuMemcpyDtoHAsync((char *)&gpuscan->kern + offset,
						   gpuscan->m_gpuscan + offset,
						   length,
						   gpuscan->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
	gss->gts.pfm.bytes_dma_recv += length;
	gss->gts.pfm.num_dma_recv++;

	if (pds_dst)
	{
		length = KERN_DATA_STORE_LENGTH(pds_dst->kds);
		rc = cuMemcpyDtoHAsync(pds_dst->kds,
							   gpuscan->m_kds_dst,
							   length,
							   gpuscan->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
		gss->gts.pfm.bytes_dma_recv += length;
		gss->gts.pfm.num_dma_recv++;
	}
	CUDA_EVENT_RECORD(gpuscan, ev_dma_recv_stop);

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
	bool				status;
	CUresult			rc;

	/* Switch CUDA Context */
	rc = cuCtxPushCurrent(gpuscan->task.cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

	PG_TRY();
	{
		status = __pgstrom_process_gpuscan(gpuscan);
	}
	PG_CATCH();
	{
		gpuscan_cleanup_cuda_resources(gpuscan);
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
