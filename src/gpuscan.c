/*
 * gpuscan.c
 *
 * Sequential scan accelerated by GPU processors
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "optimizer/tlist.h"
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
 * form/deform interface of private field of CustomScan(GpuScan)
 */
typedef struct {
	ExtensibleNode	ex;
	char	   *kern_source;	/* source of the CUDA kernel */
	cl_uint		extra_flags;	/* extra libraries to be included */
	cl_uint		proj_row_extra;	/* extra requirements if row format */
	cl_uint		proj_slot_extra;/* extra requirements if slot format */
	cl_uint		nrows_per_block;/* estimated tuple density per block */
	List	   *used_params;
	List	   *dev_quals;
} GpuScanInfo;

static inline void
form_gpuscan_info(CustomScan *cscan, GpuScanInfo *gs_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, makeString(gs_info->kern_source));
	privs = lappend(privs, makeInteger(gs_info->extra_flags));
	privs = lappend(privs, makeInteger(gs_info->proj_row_extra));
	privs = lappend(privs, makeInteger(gs_info->proj_slot_extra));
	privs = lappend(privs, makeInteger(gs_info->nrows_per_block));
	exprs = lappend(exprs, gs_info->used_params);
	exprs = lappend(exprs, gs_info->dev_quals);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static inline GpuScanInfo *
deform_gpuscan_info(CustomScan *cscan)
{
	GpuScanInfo *gs_info = palloc0(sizeof(GpuScanInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;

	gs_info->kern_source = strVal(list_nth(privs, pindex++));
	gs_info->extra_flags = intVal(list_nth(privs, pindex++));
	gs_info->proj_row_extra = intVal(list_nth(privs, pindex++));
	gs_info->proj_slot_extra = intVal(list_nth(privs, pindex++));
	gs_info->nrows_per_block = intVal(list_nth(privs, pindex++));
	gs_info->used_params = list_nth(exprs, eindex++);
	gs_info->dev_quals = list_nth(exprs, eindex++);

	return gs_info;
}

typedef struct
{
	GpuTask_v2			task;
	bool				dev_projection;		/* true, if device projection */
	bool				with_nvme_strom;
	/* CUDA resources (only server side) */
	CUfunction			kern_gpuscan_main;
	CUdeviceptr			m_gpuscan;
	CUdeviceptr			m_kds_src;
	CUdeviceptr			m_kds_dst;
	CUevent				ev_dma_send_start;
	CUevent				ev_dma_send_stop;
	CUevent				ev_dma_recv_start;
	CUevent				ev_dma_recv_stop;
	/* Performance counters */
	cl_uint				num_kern_main;
	cl_double			tv_kern_main;
	cl_double			tv_kern_exec_quals;
	cl_double			tv_kern_projection;
	cl_uint				num_dma_send;
	cl_uint				num_dma_recv;
	Size				bytes_dma_send;
	Size				bytes_dma_recv;
	cl_double			time_dma_send;
	cl_double			time_dma_recv;
	/* DMA buffers */
	pgstrom_data_store *pds_src;
	pgstrom_data_store *pds_dst;
	kern_resultbuf	   *kresults;
	kern_gpuscan		kern;
} GpuScanTask;

typedef struct {
	GpuTaskState_v2	gts;

	HeapTupleData	scan_tuple;		/* buffer to fetch tuple */
	List		   *dev_tlist;		/* tlist to be returned from the device */
	List		   *dev_quals;		/* quals to be run on the device */
	bool			dev_projection;	/* true, if device projection is valid */
	cl_uint			proj_row_extra;
	cl_uint			proj_slot_extra;
//	cl_uint			nrows_per_block;
	/* resource for CPU fallback */
	TupleTableSlot *base_slot;
	ProjectionInfo *base_proj;
} GpuScanState;

/* shared state of GpuScan for CPU parallel */
typedef struct
{
	pgstromWorkerStatistics *worker_stat;
	ParallelHeapScanDescData pscan;	/* flexible length */
} GpuScanParallelDSM;

/*
 * static functions
 */
static GpuTask_v2  *gpuscan_next_task(GpuTaskState_v2 *gts);
static TupleTableSlot *gpuscan_next_tuple(GpuTaskState_v2 *gts);
static void gpuscan_switch_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);

/*
 * cost_discount_gpu_projection
 *
 * Because of the current optimizer's design of PostgreSQL, an exact
 * target-list is not informed during path consideration.
 * It shall be attached prior to the plan creation stage once entire
 * path gets determined based on the estimated cost.
 * If GpuProjection does not make sense, it returns false,
 *
 * Note that it is just a cost reduction factor, don't set complex
 * expression on the rel->reltarget. Right now, PostgreSQL does not
 * expect such an intelligence.
 */
bool
cost_discount_gpu_projection(PlannerInfo *root, RelOptInfo *rel,
							 Cost *p_discount_per_tuple)
{
	Query	   *parse = root->parse;
	bool		have_grouping = false;
	bool		may_gpu_projection = false;
	List	   *proj_var_list = NIL;
	List	   *proj_phv_list = NIL;
	cl_uint		proj_num_attrs = 0;
	cl_uint		normal_num_attrs = 0;
	Cost		discount_per_tuple = 0.0;
	double		gpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	ListCell   *lc;

	/* GpuProjection makes sense only if top-level of scan/join */
	if (!bms_equal(root->all_baserels, rel->relids))
		return false;

	/*
	 * In case when this scan/join path is underlying other grouping
	 * clauses, or aggregations, scan/join will generate expressions
	 * only if it is grouping/sorting keys. Other expressions shall
	 * be broken down into Var nodes, then calculated in the later
	 * stage.
	 */
	if (parse->groupClause || parse->groupingSets ||
		parse->hasAggs || root->hasHavingQual)
		have_grouping = true;

	/*
	 * Walk on the prospective final target list.
	 */
	foreach (lc, root->processed_tlist)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (IsA(tle->expr, Var))
		{
			if (!list_member(proj_var_list, tle->expr))
				proj_var_list = lappend(proj_var_list, tle->expr);
			normal_num_attrs++;
		}
		else if (IsA(tle->expr, PlaceHolderVar))
		{
			if (!list_member(proj_phv_list, tle->expr))
				proj_phv_list = lappend(proj_phv_list, tle->expr);
			normal_num_attrs++;
		}
		else if (IsA(tle->expr, Const) || IsA(tle->expr, Param))
		{
			proj_num_attrs++;
			normal_num_attrs++;
		}
		else if ((!have_grouping ||
				  (tle->ressortgroupref &&
				   parse->groupClause &&
				   get_sortgroupref_clause_noerr(tle->ressortgroupref,
												 parse->groupClause) != NULL))
				 && pgstrom_device_expression(tle->expr))
		{
			QualCost	qcost;

			cost_qual_eval_node(&qcost, (Node *)tle->expr, root);
			discount_per_tuple += (qcost.per_tuple *
								   Max(1.0 - gpu_ratio, 0.0) / 8.0);
			proj_num_attrs++;
			normal_num_attrs++;
			may_gpu_projection = true;
		}
		else
		{
			List	   *temp_vars;
			ListCell   *temp_lc;

			temp_vars = pull_var_clause((Node *)tle->expr,
										PVC_RECURSE_AGGREGATES |
										PVC_RECURSE_WINDOWFUNCS |
										PVC_INCLUDE_PLACEHOLDERS);
			foreach (temp_lc, temp_vars)
			{
				Expr   *temp_expr = lfirst(temp_lc);

				if (IsA(temp_expr, Var))
				{
					if (!list_member(proj_var_list, temp_expr))
						proj_var_list = lappend(proj_var_list, temp_expr);
				}
				else if (IsA(temp_expr, PlaceHolderVar))
				{
					if (!list_member(proj_phv_list, temp_expr))
						proj_phv_list = lappend(proj_phv_list, temp_expr);
				}
				else
					elog(ERROR, "Bug? unexpected node: %s",
						 nodeToString(temp_expr));
			}
			normal_num_attrs++;
		}
	}

	proj_num_attrs += (list_length(proj_var_list) +
					   list_length(proj_phv_list));
	if (proj_num_attrs > normal_num_attrs)
		discount_per_tuple -= cpu_tuple_cost *
			(double)(proj_num_attrs - normal_num_attrs);

	list_free(proj_var_list);
	list_free(proj_phv_list);

	*p_discount_per_tuple = (may_gpu_projection ? discount_per_tuple : 0.0);

	return may_gpu_projection;
}

/*
 * cost_gpuscan_common - common part of cost estimation for GpuScan
 * 
 * Once a simple scan path is pulled up to upper node, this node takes over
 * the jobs of relation scan and execution of outer qualifiers instead of
 * execution of GpuScan node. So, its cost needs to be added to the upper
 * node.
 */
void
cost_gpuscan_common(PlannerInfo *root,
					RelOptInfo *scan_rel,
					List *scan_quals,
					int parallel_workers,
					double *p_parallel_divisor,
					double *p_scan_ntuples,
					double *p_scan_nchunks,
					cl_uint *p_nrows_per_block,
					Cost *p_startup_cost,
					Cost *p_run_cost)
{
	Cost		startup_cost = 0.0;
	Cost		run_cost = 0.0;
	double		gpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	double		parallel_divisor = (double) parallel_workers;
	double		ntuples = scan_rel->tuples;
	double		nchunks;
	double		selectivity;
	double		spc_seq_page_cost;
	cl_uint		nrows_per_block;
	Size		heap_size;
	Size		htup_size;
	QualCost	qcost;

	Assert(scan_rel->reloptkind == RELOPT_BASEREL &&
		   scan_rel->relid > 0 &&
		   scan_rel->relid < root->simple_rel_array_size);

	/* selectivity of device executable qualifiers */
	selectivity = clauselist_selectivity(root,
										 scan_quals,
										 scan_rel->relid,
										 JOIN_INNER,
										 NULL);

	/* fetch estimated page cost for tablespace containing the table */
	/*
	 * TODO: we may need to discount page cost if NVMe-Strom is capable
	 */
	get_tablespace_page_costs(scan_rel->reltablespace,
							  NULL, &spc_seq_page_cost);

	/*
	 * Discount page scan cost if NVMe-Strom is capable
	 *
	 * XXX - acceleration ratio depends on number of SSDs configured
	 * as MD0-RAID volume, number of parallel workers and so on.
	 * Once NVMe-Strom driver supports hardware configuration info,
	 * we follow it.
	 */
	if (ScanPathWillUseNvmeStrom(root, scan_rel))
	{
		/* FIXME: discount 50% if NVMe-Strom is ready */
		spc_seq_page_cost /= 1.5;
		/*
		 * FIXME: i/o concurrency will effective throughput according
		 * to the number of parallel workers
		 */
		if (parallel_workers > 0)
			spc_seq_page_cost /= (Cost)(1 + Min(parallel_workers, 4));
	}

	/*
	 * Disk i/o cost; we may add special treatment for NVMe-Strom.
	 * On the other hands, planner usually choose PG-Strom's path
	 * for large scale of data.
	 */
	run_cost += spc_seq_page_cost * (double)scan_rel->pages;

	/*
	 * Cost adjustment by CPU parallelism, if used.
	 * (overall logic is equivalent to cost_seqscan())
	 */
	if (parallel_workers > 0)
	{
		double		leader_contribution;

		/* How much leader process can contribute query execution? */
		leader_contribution = 1.0 - (0.3 * (double)parallel_workers);
		if (leader_contribution > 0)
			parallel_divisor += leader_contribution;

		/* number of tuples to be actually processed */
		ntuples  = clamp_row_est(ntuples / parallel_divisor);

		/*
		 * After the v2.0, pg_strom.gpu_setup_cost represents the cost for
		 * run-time code build by NVRTC. Once binary is constructed, it can
		 * be shared with all the worker process, so we can discount the
		 * cost by parallel_divisor.
		 */
		startup_cost += pgstrom_gpu_setup_cost / parallel_divisor;
	}
	else
	{
		parallel_divisor = 1.0;
		startup_cost += pgstrom_gpu_setup_cost;
	}

	/* estimation for number of chunks (assume KDS_FORMAT_ROW) */
	heap_size = (double)(BLCKSZ - SizeOfPageHeaderData) * scan_rel->pages;
	htup_size = (MAXALIGN(offsetof(HeapTupleHeaderData,
								   t_bits[BITMAPLEN(scan_rel->max_attr)])) +
				 MAXALIGN(heap_size / Max(scan_rel->tuples, 1.0) -
						  sizeof(ItemIdData) - SizeofHeapTupleHeader));
	nchunks =  (((double)(offsetof(kern_tupitem, htup) + htup_size +
						  sizeof(cl_uint)) * Max(ntuples, 1.0)) /
				((double)(pgstrom_chunk_size() -
						  KDS_CALCULATE_HEAD_LENGTH(scan_rel->max_attr))));
	nchunks = Max(nchunks, 1);

	/*
	 * estimation of the tuple density per block - this logic follows
	 * the manner in estimate_rel_size()
	 */
	if (scan_rel->pages > 0)
		nrows_per_block = ceil(scan_rel->tuples / (double)scan_rel->pages);
	else
	{
		RangeTblEntry *rte = root->simple_rte_array[scan_rel->relid];
		size_t		tuple_width = get_relation_data_width(rte->relid, NULL);

		tuple_width += MAXALIGN(SizeofHeapTupleHeader);
		tuple_width += sizeof(ItemIdData);
		/* note: integer division is intentional here */
		nrows_per_block = (BLCKSZ - SizeOfPageHeaderData) / tuple_width;
	}

	/* Cost for GPU qualifiers */
	cost_qual_eval(&qcost, scan_quals, root);
	startup_cost += qcost.startup;
	run_cost += qcost.per_tuple * gpu_ratio * ntuples;
	ntuples *= selectivity;

	/* Cost for DMA transfer */
	run_cost += pgstrom_gpu_dma_cost * nchunks;

	*p_parallel_divisor = parallel_divisor;
	*p_scan_ntuples = ntuples;
	*p_scan_nchunks = nchunks;
	*p_nrows_per_block = nrows_per_block;
	*p_startup_cost = startup_cost;
	*p_run_cost = run_cost;
}

/*
 * create_gpuscan_path - constructor of CustomPath(GpuScan) node
 */
static Path *
create_gpuscan_path(PlannerInfo *root,
					RelOptInfo *baserel,
					List *dev_quals,
					List *host_quals,
					Cost discount_per_tuple,
					int parallel_nworkers)
{
	GpuScanInfo	   *gs_info = palloc0(sizeof(GpuScanInfo));
	CustomPath	   *cpath;
	ParamPathInfo  *param_info;
	List		   *ppi_quals;
	Cost			startup_cost;
	Cost			run_cost;
	Cost			startup_delay;
	QualCost		qcost;
	double			parallel_divisor;
	double			scan_ntuples;
	double			scan_nchunks;
	double			cpu_per_tuple = 0.0;

	/* cost for disk i/o + GPU qualifiers */
	cost_gpuscan_common(root, baserel, dev_quals,
						parallel_nworkers,
						&parallel_divisor,
						&scan_ntuples,
						&scan_nchunks,
						&gs_info->nrows_per_block,
						&startup_cost,
						&run_cost);

	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	cpath = makeNode(CustomPath);
	cpath->path.pathtype = T_CustomScan;
	cpath->path.parent = baserel;
	cpath->path.pathtarget = baserel->reltarget;
	cpath->path.param_info = param_info;
	cpath->path.parallel_aware = parallel_nworkers > 0 ? true : false;
	cpath->path.parallel_safe = baserel->consider_parallel;
	cpath->path.parallel_workers = parallel_nworkers;
	cpath->path.rows = (param_info
						? param_info->ppi_rows
						: baserel->rows) / parallel_divisor;
	/* cost for CPU qualifiers */
	cost_qual_eval(&qcost, host_quals, root);
	startup_cost += qcost.startup;
	cpu_per_tuple += qcost.per_tuple;

	/* PPI costs (as a part of host quals, if any) */
	ppi_quals = (param_info ? param_info->ppi_clauses : NIL);
	cost_qual_eval(&qcost, ppi_quals, root);
	startup_cost += qcost.startup;
	cpu_per_tuple += qcost.per_tuple;
	run_cost += (cpu_per_tuple + cpu_tuple_cost) * scan_ntuples;

	/* Cost discount by GPU projection */
	run_cost = Max(run_cost - discount_per_tuple * scan_ntuples, 0.0);

	/* Latency to get the first chunk */
	startup_delay = run_cost * (1.0 / scan_nchunks);

	cpath->path.startup_cost = startup_cost + startup_delay;
	cpath->path.total_cost = startup_cost + run_cost;
	cpath->path.pathkeys = NIL;	/* unsorted results */
	cpath->flags = 0;
	cpath->custom_paths = NIL;
	cpath->custom_private = list_make1(gs_info);
	cpath->methods = &gpuscan_path_methods;

	return &cpath->path;
}

/*
 * gpuscan_add_scan_path - entrypoint of the set_rel_pathlist_hook
 */
static void
gpuscan_add_scan_path(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Index rtindex,
					  RangeTblEntry *rte)
{
	Path	   *pathnode;
	List	   *dev_quals = NIL;
	List	   *host_quals = NIL;
	ListCell   *lc;
	Cost		discount_per_tuple;

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

	/* Check whether the qualifier can run on GPU device */
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(lc);

		if (pgstrom_device_expression(rinfo->clause))
			dev_quals = lappend(dev_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}

	/*
	 * Check whether the GPU Projection may be available
	 */
	if (!cost_discount_gpu_projection(root, baserel,
									  &discount_per_tuple))
	{
		/*
		 * GpuScan does not make sense if neither qualifier nor target-
		 * list are runnable on GPU device.
		 */
		if (dev_quals == NIL)
			return;
	}

	/* add GpuScan path in single process */
	pathnode = create_gpuscan_path(root, baserel,
								   dev_quals, host_quals,
								   discount_per_tuple,
								   0);
	add_path(baserel, pathnode);

	/* If appropriate, consider parallel GpuScan */
	if (baserel->consider_parallel && baserel->lateral_relids == NULL)
	{
		int		parallel_nworkers;
		int		parallel_threshold;

		/*
		 * We follow the logic of create_plain_partial_paths to determine
		 * the number of parallel workers as baseline. Then, it shall be
		 * adjusted according to the PG-Strom configuration.
		 */
		if (baserel->rel_parallel_workers != -1)
			parallel_nworkers = baserel->rel_parallel_workers;
		else
		{
			/* relation is too small for parallel execution? */
			if (baserel->pages < (BlockNumber) min_parallel_relation_size &&
				baserel->reloptkind == RELOPT_BASEREL)
				return;

			/*
			 * select the number of workers based on the log of the size of
			 * the relation.
			 */
			parallel_nworkers = 1;
			parallel_threshold = Max(min_parallel_relation_size, 1);
			while (baserel->pages >= (BlockNumber) (parallel_threshold * 3))
			{
				parallel_nworkers++;
				parallel_threshold *= 3;
				if (parallel_threshold > INT_MAX / 3)
					break;			/* avoid overflow */
			}
		}
		
		/*
		 * TODO: Put something specific to GpuScan to adjust parallel_workers
		 */

		/* max_parallel_workers_per_gather is the upper limit  */
		parallel_nworkers = Min4(parallel_nworkers,
								 4 * numDevAttrs,
								 max_parallel_workers_per_gather,
								 numGpuServerProcesses() - 1);
		if (parallel_nworkers <= 0)
			return;

		/* add GpuScan path performing on parallel workers */
		pathnode = create_gpuscan_path(root, baserel,
									   copyObject(dev_quals),
									   copyObject(host_quals),
									   discount_per_tuple,
									   parallel_nworkers);
		add_partial_path(baserel, pathnode);

		/* then, potentially generate Gather + GpuScan path */
		generate_gather_paths(root, baserel);

		foreach (lc, baserel->pathlist)
		{
			pathnode = lfirst(lc);
		}
	}
}

/*
 * Code generator for GpuScan's qualifier
 */
void
codegen_gpuscan_quals(StringInfo kern, codegen_context *context,
					  Index scanrelid, List *dev_quals)
{
	devtype_info   *dtype;
	Var			   *var;
	char		   *expr_code;
	ListCell	   *lc;

	appendStringInfoString(
		kern,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpuscan_quals_eval(kern_context *kcxt,\n"
		"                   kern_data_store *kds,\n"
		"                   ItemPointerData *t_self,\n"
		"                   HeapTupleHeaderData *htup)\n");
	if (dev_quals == NIL)
	{
		appendStringInfoString(kern,
							   "{\n"
							   "  return true;\n"
							   "}\n\n");
		return;
	}

	/* Let's walk on the device expression tree */
	expr_code = pgstrom_codegen_expression((Node *)dev_quals, context);
	appendStringInfoString(kern, "{\n");
	/* Const/Param declarations */
	pgstrom_codegen_param_declarations(kern, context);
	/* Sanity check of used_vars */
	foreach (lc, context->used_vars)
	{
		var = lfirst(lc);
		if (var->varno != scanrelid)
			elog(ERROR, "unexpected var-node reference: %s expected %u",
				 nodeToString(var), scanrelid);
		if (var->varattno <= InvalidAttrNumber)
			elog(ERROR, "cannot reference system column or whole-row on GPU");
		dtype = pgstrom_devtype_lookup(var->vartype);
		if (!dtype)
			elog(ERROR, "failed to lookup device type: %s",
				 format_type_be(var->vartype));
	}

	/*
	 * Var declarations - if qualifier uses only one variables (like x > 0),
	 * the pg_xxxx_vref() service routine is more efficient because it may
	 * use attcacheoff to skip walking on tuple attributes.
	 */
	if (list_length(context->used_vars) < 2)
	{
		foreach (lc, context->used_vars)
		{
			var = lfirst(lc);

			/* we don't support system columns in expression now */
			Assert(var->varattno > 0);

			dtype = pgstrom_devtype_lookup(var->vartype);
			appendStringInfo(
				kern,
				"  void *addr = kern_get_datum_tuple(kds->colmeta,htup,%u);\n"
				"  pg_%s_t %s_%u = pg_%s_datum_ref(kcxt,addr,false);\n",
				var->varattno - 1,
				dtype->type_name,
				context->var_label,
				var->varattno,
				dtype->type_name);
		}
	}
	else
	{
		AttrNumber		anum, varattno_max = 0;

		/* declarations */
		foreach (lc, context->used_vars)
		{
			var = lfirst(lc);
			dtype = pgstrom_devtype_lookup(var->vartype);

			appendStringInfo(
				kern,
				"  pg_%s_t %s_%u;\n",
				dtype->type_name,
                context->var_label,
				var->varattno);
			varattno_max = Max(varattno_max, var->varattno);
		}

		/* walking on the HeapTuple */
		appendStringInfoString(
			kern,
			"  char *addr;\n"
			"\n"
			"  assert(htup != NULL);\n"
			"  EXTRACT_HEAP_TUPLE_BEGIN(addr, kds, htup);\n");

		for (anum=1; anum <= varattno_max; anum++)
		{
			foreach (lc, context->used_vars)
			{
				var = lfirst(lc);

				if (var->varattno == anum)
				{
					dtype = pgstrom_devtype_lookup(var->vartype);

					appendStringInfo(
						kern,
						"  %s_%u = pg_%s_datum_ref(kcxt, addr, false);\n",
						context->var_label,
						var->varattno,
						dtype->type_name);
					break;	/* no need to read same value twice */
				}
			}

			if (anum < varattno_max)
				appendStringInfoString(
					kern,
					"  EXTRACT_HEAP_TUPLE_NEXT(addr);\n");
		}
		appendStringInfoString(
			kern,
			"  EXTRACT_HEAP_TUPLE_END();\n");
	}
	appendStringInfo(
		kern,
		"\n"
		"  return EVAL(%s);\n"
		"}\n\n",
		expr_code);
}

/*
 * Code generator for GpuScan's projection
 */
static void
codegen_gpuscan_projection(StringInfo kern, codegen_context *context,
						   Index scanrelid, Relation relation,
						   List *__tlist_dev)
{
	TupleDesc		tupdesc = RelationGetDescr(relation);
	List		   *tlist_dev = NIL;
	AttrNumber	   *varremaps;
	Bitmapset	   *varattnos;
	ListCell	   *lc;
	int				prev;
	int				i, j, k;
	bool			needs_vlbuf;
	devtype_info   *dtype;
	StringInfoData	decl;
	StringInfoData	body;
	StringInfoData	temp;

	initStringInfo(&decl);
	initStringInfo(&body);
	initStringInfo(&temp);
	/*
	 * step.0 - extract non-junk attributes
	 */
	foreach (lc, __tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (!tle->resjunk)
			tlist_dev = lappend(tlist_dev, tle);
	}

	/*
	 * step.1 - declaration of functions and KVAR_xx for expressions
	 */
	appendStringInfoString(
		&decl,
		"STATIC_FUNCTION(void)\n"
		"gpuscan_projection(kern_context *kcxt,\n"
		"                   kern_data_store *kds_src,\n"
		"                   HeapTupleHeaderData *htup,\n"
		"                   ItemPointerData *t_self,\n"
		"                   kern_data_store *kds_dst,\n"
		"                   cl_uint dst_nitems,\n"
		"                   Datum *tup_values,\n"
		"                   cl_bool *tup_isnull,\n"
		"                   cl_bool *tup_internal)\n"
		"{\n"
		"  cl_bool is_slot_format __attribute__((unused));\n"
		"  char *curr;\n");

	varremaps = palloc0(sizeof(AttrNumber) * tupdesc->natts);
	varattnos = NULL;
	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		Assert(tle->resno > 0);
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
			pull_varattnos((Node *)tle->expr, scanrelid, &varattnos);
		}
	}

	prev = -1;
	while ((prev = bms_next_member(varattnos, prev)) >= 0)
	{
		Form_pg_attribute attr;
		AttrNumber		anum = prev + FirstLowInvalidHeapAttributeNumber;

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
	 * step.2 - extract tuples and load values to KVAR or values/isnull
	 * array (only if tupitem_src is valid, of course)
	 */
	appendStringInfoString(
		&body,
		"  is_slot_format = (kds_dst->format == KDS_FORMAT_SLOT);\n");

	/*
	 * System columns reference if any
	 */
	for (j=0; j < list_length(tlist_dev); j++)
	{
		Form_pg_attribute	attr;

		if (varremaps[j] >= 0)
			continue;

		attr = SystemAttributeDefinition(varremaps[j], true);
		if (attr->attbyval)
			appendStringInfo(
				&body,
				"  /* %s system column */\n"
				"  tup_isnull[%d] = !htup;\n"
				"  if (htup)\n"
				"    tup_values[%d] = kern_getsysatt_%s(kds_src,htup,t_self);\n",
				NameStr(attr->attname),
				j,
				j, NameStr(attr->attname));
		else
			appendStringInfo(
				&body,
				"  /* %s system column */\n"
				"  tup_isnull[%d] = !htup;\n"
				"  if (htup)\n"
				"  {\n"
				"    void *__ptr = kern_getsysatt_%s(kds_src,htup,t_self);\n"
				"    if (is_slot_format)\n"
				"      __ptr = devptr_to_host(kds_src, __ptr);\n"
				"    tup_values[%d] = PointerGetDatum(__ptr);\n"
				"  }\n",
				NameStr(attr->attname),
				j,
				NameStr(attr->attname),
				j);
	}

	/*
	 * Extract regular tuples
	 */
	appendStringInfoString(
		&temp,
		"  EXTRACT_HEAP_TUPLE_BEGIN(curr, kds_src, htup);\n");

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];
		bool		referenced = false;

		dtype = pgstrom_devtype_lookup(attr->atttypid);
		k = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		/* Put values on tup_values/tup_isnull if referenced */
		for (j=0; j < list_length(tlist_dev); j++)
		{
			if (varremaps[j] != attr->attnum)
				continue;

			appendStringInfo(
				&temp,
				"  tup_isnull[%d] = !curr;\n"
				"  if (curr)\n"
				"  {\n",
				j);
			if (attr->attbyval)
			{
				appendStringInfo(
					&temp,
					"    tup_values[%d] = *((%s *) curr);\n",
					j,
					(attr->attlen == sizeof(cl_long) ? "cl_long" :
					 attr->attlen == sizeof(cl_int)  ? "cl_int"  :
					 attr->attlen == sizeof(cl_short) ? "cl_short" :
					 "cl_char"));
			}
			else
			{
				/* KDS_FORMAT_SLOT needs host pointer */
				appendStringInfo(
					&temp,
					"    tup_values[%d] = (is_slot_format\n"
					"                      ? devptr_to_host(kds_src, curr)\n"
					"                      : PointerGetDatum(curr));\n",
					j);
			}
			appendStringInfo(
				&temp,
				"  }\n");
			referenced = true;
		}
		/* Load values to KVAR_xx */
		k = attr->attnum - FirstLowInvalidHeapAttributeNumber;
		if (bms_is_member(k, varattnos))
		{
			appendStringInfo(
				&temp,
				"  KVAR_%u = pg_%s_datum_ref(kcxt, curr, false);\n",
				attr->attnum,
				dtype->type_name);
			referenced = true;
		}

		if (referenced)
		{
			appendStringInfoString(&body, temp.data);
			resetStringInfo(&temp);
		}
		appendStringInfoString(
			&temp,
			"  EXTRACT_HEAP_TUPLE_NEXT(curr);\n");
	}
	appendStringInfoString(
		&body,
		"  EXTRACT_HEAP_TUPLE_END();\n"
		"\n");

	/*
	 * step.3 - execute expression node, then store the result onto KVAR_xx
	 */
    foreach (lc, tlist_dev)
    {
        TargetEntry    *tle = lfirst(lc);
		Oid				type_oid;

		if (IsA(tle->expr, Var))
			continue;
		/* NOTE: Const/Param are once loaded to expr_%u variable. */

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "Bug? device supported type is missing: %s",
				 format_type_be(type_oid));
		appendStringInfo(
			&decl,
			"  pg_%s_t expr_%u_v;\n",
			dtype->type_name,
			tle->resno);
		appendStringInfo(
			&body,
			"  expr_%u_v = %s;\n",
			tle->resno,
			pgstrom_codegen_expression((Node *) tle->expr, context));
	}
	appendStringInfoChar(&body, '\n');

	/*
	 * step.4 (only KDS_FORMAT_SLOT)
	 *
	 * We have to allocate extra buffer for indirect or numeric data type.
	 * Also, any pointer values have to be fixed up to the host pointer.
	 */
	resetStringInfo(&temp);
	appendStringInfo(
		&temp,
		"  if (kds_dst->format == KDS_FORMAT_SLOT)\n"
        "  {\n"
		"    cl_uint vl_len = 0;\n"
		"    cl_uint offset;\n"
		"    cl_uint count;\n"
		"    cl_uint __shared__ base;\n"
		"\n"
		"    if (htup)\n"
		"    {\n");

	needs_vlbuf = false;
	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid;

		if (IsA(tle->expr, Var) ||		/* just reference to kds_src */
			IsA(tle->expr, Const) ||	/* just reference to kparams */
			IsA(tle->expr, Param))		/* just reference to kparams */
			continue;

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "Bug? device supported type is missing: %s",
				 format_type_be(type_oid));

		if (type_oid == NUMERICOID)
		{
			appendStringInfo(
				&temp,
				"      if (!expr_%u_v.isnull)\n"
				"        vl_len = TYPEALIGN(sizeof(cl_uint), vl_len)\n"
				"               + pg_numeric_to_varlena(kcxt,NULL,\n"
				"                                       expr_%u_v.value,\n"
				"                                       expr_%u_v.isnull);\n",
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
				"      if (!expr_%u_v.isnull)\n"
				"        vl_len = TYPEALIGN(%u, vl_len) + %u;\n",
				tle->resno,
				dtype->type_align,
				dtype->type_length);
			needs_vlbuf = true;
		}
	}

	if (needs_vlbuf)
	{
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
			"           - (base + offset + vl_len);\n"
			"  }\n\n");
		appendStringInfoString(&decl, "  char *vl_buf = NULL;\n");
		appendStringInfoString(&body, temp.data);
	}

	/*
	 * step.5 - Store the expressions on the slot.
	 * If FDW_FORMAT_SLOT, any pointer type must be adjusted to the host-
	 * pointer. Elsewhere, the caller expects device pointer.
	 */
	resetStringInfo(&temp);
	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid;

		/* host pointer should be already set */
		if (varremaps[tle->resno-1])
		{
			Assert(IsA(tle->expr, Var));
			continue;
		}

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "Bug? device supported type is missing: %u", type_oid);

		appendStringInfo(
			&temp,
			"    tup_isnull[%d] = expr_%u_v.isnull;\n",
			tle->resno - 1, tle->resno);

		if (type_oid == NUMERICOID)
		{
			appendStringInfo(
				&temp,
				"    if (!expr_%u_v.isnull)\n"
				"    {\n"
				"      if (is_slot_format)\n"
				"      {\n"
				"        vl_buf = (char *)TYPEALIGN(sizeof(cl_int), vl_buf);\n"
				"        tup_values[%d] = devptr_to_host(kds_dst, vl_buf);\n"
				"        vl_buf += pg_numeric_to_varlena(kcxt, vl_buf,\n"
				"                                        expr_%u_v.value,\n"
				"                                        expr_%u_v.isnull);\n"
				"      }\n"
				"      else\n"
				"      {\n"
				"        tup_values[%d] = expr_%u_v.value;\n"
				"      }\n"
				"    }\n",
				tle->resno,
                tle->resno - 1,
				tle->resno,
				tle->resno,
				tle->resno - 1,
				tle->resno);
		}
		else if (dtype->type_byval)
		{
			appendStringInfo(
				&temp,
				"    if (!expr_%u_v.isnull)\n"
				"      tup_values[%d] = pg_%s_to_datum(expr_%u_v.value);\n",
				tle->resno,
				tle->resno - 1,
				dtype->type_name,
				tle->resno);
		}
		else if (IsA(tle->expr, Const) || IsA(tle->expr, Param))
		{
			/*
             * Const/Param shall be stored in kparams, thus, we don't need
             * to allocate extra buffer again. Just referemce it.
             */
            appendStringInfo(
                &temp,
                "    if (!expr_%u_v.isnull)\n"
                "      tup_values[%d] = (is_slot_format"
				"                 ? devptr_to_host(kcxt->kparams,\n"
				"                                  expr_%u_v.value)\n"
				"                 : PointerGetDatum(expr_%u_v.value));\n",
                tle->resno,
                tle->resno - 1,
                tle->resno,
				tle->resno);
		}
		else
		{
			Assert(dtype->type_length > 0);
			appendStringInfo(
				&temp,
				"    if (!expr_%u_v.isnull)\n"
				"    {\n"
				"      if (is_slot_format)\n"
				"      {\n"
				"        vl_buf = (char *)TYPEALIGN(%u, vl_buf);\n"
				"        tup_values[%d] = devptr_to_host(kds_dst, vl_buf);\n"
				"        memcpy(vl_buf, &expr_%u_v.value, %d);\n"
				"        vl_buf += %d;\n"
				"      }\n"
				"      else\n"
				"      {\n"
				"        tup_values[%d] = PointerGetDatum(expr_%u_v.value);\n"
				"      }\n"
				"    }\n",
				tle->resno,
				dtype->type_align,
				tle->resno - 1,
				tle->resno, dtype->type_length,
				dtype->type_length,
				tle->resno - 1, tle->resno);
		}
	}

	if (temp.len > 0)
		appendStringInfo(
			&body,
			"  if (htup != NULL)\n"
			"  {\n"
			"%s"
			"  }\n", temp.data);
	appendStringInfo(
		&body,
		"}\n");

	/* parameter references */
	pgstrom_codegen_param_declarations(&decl, context);

	/* OK, write back the kernel source */
	appendStringInfo(kern, "%s\n%s\n", decl.data, body.data);
	appendStringInfoString(kern, "#define CUDA_GPUSCAN_HAS_PROJECTION 1\n");
	list_free(tlist_dev);
	pfree(temp.data);
	pfree(decl.data);
	pfree(body.data);
}

/*
 * add_unique_expression - adds an expression node on the supplied
 * target-list, then returns true, if new target-entry was added.
 */
bool
add_unique_expression(Expr *expr, List **p_targetlist, bool resjunk)
{
	TargetEntry	   *tle;
	ListCell	   *lc;
	AttrNumber		resno;

	foreach (lc, *p_targetlist)
	{
		tle = (TargetEntry *) lfirst(lc);
		if (equal(expr, tle->expr))
			return false;
	}
	/* Not found, so add this expression */
	resno = list_length(*p_targetlist) + 1;
	tle = makeTargetEntry(copyObject(expr), resno, NULL, resjunk);
	*p_targetlist = lappend(*p_targetlist, tle);

	return true;
}

/*
 * build_gpuscan_projection
 *
 * It checks whether the GpuProjection of GpuScan makes sense.
 * If executor may require the physically compatible tuple as result,
 * we don't need to have a projection in GPU side.
 */
static List *
build_gpuscan_projection(Index scanrelid,
						 Relation relation,
						 List *tlist,
						 List *host_quals,
						 List *dev_quals)
{
	TupleDesc	tupdesc = RelationGetDescr(relation);
	List	   *tlist_dev = NIL;
	AttrNumber	attnum = 1;
	bool		compatible_tlist = true;
	ListCell   *lc;

	foreach (lc, tlist)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;

			/* if these Asserts fail, planner messed up */
			Assert(var->varno == scanrelid);
			Assert(var->varlevelsup == 0);

			/* GPU projection cannot contain whole-row var */
			if (var->varattno == InvalidAttrNumber)
				return NIL;

			/*
			 * check whether the original tlist matches the physical layout
			 * of the base relation. GPU can reorder the var reference
			 * regardless of the data-type support.
			 */
			if (var->varattno != attnum || attnum > tupdesc->natts)
				compatible_tlist = false;
			else
			{
				Form_pg_attribute	attr = tupdesc->attrs[attnum-1];

				/* should not be a reference to dropped columns */
				Assert(!attr->attisdropped);
				/* See the logic in tlist_matches_tupdesc */
				if (var->vartype != attr->atttypid ||
					(var->vartypmod != attr->atttypmod &&
					 var->vartypmod != -1))
					compatible_tlist = false;
			}
			/* add a primitive var-node on the tlist_dev */
			if (!add_unique_expression((Expr *) var, &tlist_dev, false))
				compatible_tlist = false;
		}
		else if (pgstrom_device_expression(tle->expr))
		{
			/* add device executable expression onto the tlist_dev */
			add_unique_expression(tle->expr, &tlist_dev, false);
			/* of course, it is not a physically compatible tlist */
			compatible_tlist = false;
		}
		else
		{
			/*
			 * Elsewhere, expression is not device executable
			 *
			 * MEMO: We may be able to process Const/Param but no data-type
			 * support on the device side, as long as its length is small
			 * enought. However, we don't think it has frequent use cases
			 * right now.
			 */
			List	   *vars_list = pull_vars_of_level((Node *)tle->expr, 0);
			ListCell   *cell;

			foreach (cell, vars_list)
			{
				Var	   *var = lfirst(cell);
				if (var->varattno == InvalidAttrNumber)
					return NIL;		/* no whole-row support */
				add_unique_expression((Expr *)var, &tlist_dev, false);
			}
			list_free(vars_list);
			/* of course, it is not a physically compatible tlist */
			compatible_tlist = false;
		}
		attnum++;
	}

	/* Is the tlist shorter than relation's definition? */
	if (RelationGetNumberOfAttributes(relation) != attnum)
		compatible_tlist = false;

	/*
	 * Host quals needs 
	 */
	if (host_quals)
	{
		List	   *vars_list = pull_vars_of_level((Node *)host_quals, 0);
		ListCell   *cell;

		foreach (cell, vars_list)
		{
			Var	   *var = lfirst(cell);
			if (var->varattno == InvalidAttrNumber)
				return NIL;		/* no whole-row support */
			add_unique_expression((Expr *)var, &tlist_dev, false);
		}
		list_free(vars_list);
	}

	/*
	 * Device quals need junk var-nodes
	 */
	if (dev_quals)
	{
		List	   *vars_list = pull_vars_of_level((Node *)dev_quals, 0);
		ListCell   *cell;

		foreach (cell, vars_list)
		{
			Var	   *var = lfirst(cell);
			if (var->varattno == InvalidAttrNumber)
				return NIL;		/* no whole-row support */
			add_unique_expression((Expr *)var, &tlist_dev, true);
		}
		list_free(vars_list);
	}

	/*
	 * At this point, device projection is "executable".
	 * However, if compatible_tlist == true, it implies the upper node
	 * expects physically compatible tuple, thus, it is uncertain whether
	 * we should run GpuProjection for this GpuScan.
	 */
	if (compatible_tlist)
		return NIL;
	return tlist_dev;
}

/*
 * bufsz_estimate_gpuscan_projection - GPU Projection may need larger
 * destination buffer than the source buffer. 
 */
static void
bufsz_estimate_gpuscan_projection(RelOptInfo *baserel, Relation relation,
								  List *tlist_dev,
								  cl_int *p_proj_row_extra,
								  cl_int *p_proj_slot_extra)
{
	TupleDesc	tupdesc = RelationGetDescr(relation);
	cl_int		proj_src_extra;
	cl_int		proj_row_extra;
	cl_int		proj_slot_extra;
	AttrNumber	anum;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	ListCell   *lc;

	proj_row_extra = offsetof(HeapTupleHeaderData,
							  t_bits[BITMAPLEN(list_length(tlist_dev))]);
	proj_slot_extra = 0;

	foreach (lc, tlist_dev)
	{
		TargetEntry *tle = lfirst(lc);
		Oid		type_oid = exprType((Node *)tle->expr);
		int32	type_mod = exprTypmod((Node *)tle->expr);

		/* alignment */
		get_typlenbyvalalign(type_oid, &typlen, &typbyval, &typalign);
		proj_row_extra = att_align_nominal(proj_row_extra, typalign);

		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;

			Assert(var->vartype == type_oid &&
				   var->vartypmod == type_mod);
			Assert(var->varno == baserel->relid &&
				   var->varattno >= baserel->min_attr &&
				   var->varattno <= baserel->max_attr);
			proj_row_extra += baserel->attr_widths[var->varattno -
												   baserel->min_attr];
		}
		else if (IsA(tle->expr, Const))
		{
			Const  *con = (Const *) tle->expr;

			/* raw-data is the most reliable information source :) */
			if (!con->constisnull)
			{
				proj_row_extra += (con->constlen > 0
								   ? con->constlen
								   : VARSIZE_ANY(con->constvalue));
			}
		}
		else
		{
			proj_row_extra = att_align_nominal(proj_row_extra, typalign);
			proj_row_extra += get_typavgwidth(type_oid, type_mod);

			/*
			 * In case of KDS_FORMAT_SLOT, it needs extra buffer only when
			 * expression has data-type (a) with internal format (like
			 * NUMERIC right now), or (b) with fixed-length but indirect
			 * references.
			 */
			if (type_oid == NUMERICOID)
				proj_slot_extra += 32;	/* enough space for internal format */
			else if (typlen > 0 && !typbyval)
				proj_slot_extra += MAXALIGN(typlen);
		}
	}
	proj_row_extra = MAXALIGN(proj_row_extra);

	/*
	 * Length of the source relation
	 */
	proj_src_extra = offsetof(HeapTupleHeaderData,
							  t_bits[BITMAPLEN(baserel->max_attr)]);
	for (anum = 1; anum <= baserel->max_attr; anum++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[anum - 1];

		proj_src_extra = att_align_nominal(proj_src_extra, attr->attalign);
		proj_src_extra += baserel->attr_widths[anum - baserel->min_attr];
	}
	proj_src_extra = MAXALIGN(proj_src_extra);

	*p_proj_row_extra = (proj_row_extra > proj_src_extra
						 ? proj_row_extra - proj_src_extra : 0);
	*p_proj_slot_extra = proj_slot_extra;
}

/*
 * PlanGpuScanPath - construction of a new GpuScan plan node
 */
static Plan *
PlanGpuScanPath(PlannerInfo *root,
				RelOptInfo *baserel,
				CustomPath *best_path,
				List *tlist,
				List *clauses,
				List *custom_children)
{
	GpuScanInfo	   *gs_info = linitial(best_path->custom_private);
	CustomScan	   *cscan;
	RangeTblEntry  *rte;
	Relation		relation;
	List		   *host_quals = NIL;
	List		   *dev_quals = NIL;
	List		   *tlist_dev = NIL;
	ListCell	   *cell;
	cl_int			proj_row_extra = 0;
	cl_int			proj_slot_extra = 0;
	StringInfoData	kern;
	StringInfoData	source;
	codegen_context	context;

	/* It should be a base relation */
	Assert(baserel->relid > 0);
	Assert(baserel->rtekind == RTE_RELATION);
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
	 * Code construction for the CUDA kernel code
	 */
	rte = planner_rt_fetch(baserel->relid, root);
	relation = heap_open(rte->relid, NoLock);

	initStringInfo(&kern);
	initStringInfo(&source);
	pgstrom_init_codegen_context(&context);
	codegen_gpuscan_quals(&kern, &context, baserel->relid, dev_quals);

	tlist_dev = build_gpuscan_projection(baserel->relid, relation,
										 tlist, host_quals, dev_quals);
	if (tlist_dev != NIL)
	{
		bufsz_estimate_gpuscan_projection(baserel, relation, tlist_dev,
										  &proj_row_extra,
										  &proj_slot_extra);
		context.param_refs = NULL;
		codegen_gpuscan_projection(&kern, &context, baserel->relid,
								   relation, tlist_dev);
	}
	heap_close(relation, NoLock);

	pgstrom_codegen_func_declarations(&source, &context);
	pgstrom_codegen_expr_declarations(&source, &context);
	appendStringInfoString(&source, kern.data);
	pfree(kern.data);

	/*
	 * Construction of GpuScanPlan node; on top of CustomPlan node
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = host_quals;
	cscan->scan.plan.lefttree = NULL;
	cscan->scan.plan.righttree = NULL;
	cscan->scan.scanrelid = baserel->relid;
	cscan->flags = best_path->flags;
	cscan->methods = &gpuscan_plan_methods;
	cscan->custom_plans = NIL;
	cscan->custom_scan_tlist = tlist_dev;

	gs_info->kern_source = source.data;
	gs_info->extra_flags = context.extra_flags |
		DEVKERNEL_NEEDS_DYNPARA | DEVKERNEL_NEEDS_GPUSCAN;
	gs_info->proj_row_extra = proj_row_extra;
	gs_info->proj_slot_extra = proj_slot_extra;
	gs_info->used_params = context.used_params;
	gs_info->dev_quals = dev_quals;
	form_gpuscan_info(cscan, gs_info);

	return &cscan->scan.plan;
}

/*
 * pgstrom_pullup_outer_scan - pull up outer_path if it is a simple relation
 * scan with device executable qualifiers.
 */
bool
pgstrom_pullup_outer_scan(const Path *outer_path,
						  Index *p_outer_relid,
						  List **p_outer_quals,
						  bool *p_parallel_aware)
{
	RelOptInfo *baserel = outer_path->parent;
	PathTarget *outer_target = outer_path->pathtarget;
	List	   *outer_quals = NIL;
	ListCell   *lc;

	if (!enable_pullup_outer_scan)
		return false;

	for (;;)
	{
		if (outer_path->pathtype == T_SeqScan)
			break;	/* OK */
		if (pgstrom_path_is_gpuscan(outer_path))
			break;	/* OK, only if GpuScan */
		if (outer_path->pathtype == T_Result)
		{
			ProjectionPath *ppath = (ProjectionPath *) outer_path;

			if (ppath->dummypp)
			{
				outer_path = ppath->subpath;
				continue;	/* Dive into one more deep level */
			}
		}
		return false;	/* Elsewhere, we cannot pull-up the scan path */
	}

	/* qualifier has to be device executable */
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(lc);

		if (!pgstrom_device_expression(rinfo->clause))
			return false;
		outer_quals = lappend(outer_quals, rinfo->clause);
	}

	/* target entry has to be */
	foreach (lc, outer_target->exprs)
	{
		Expr   *expr = (Expr *) lfirst(lc);

		if (IsA(expr, Var))
		{
			Var	   *var = (Var *) expr;

			/* we don't support whole-row reference */
			if (var->varattno == InvalidAttrNumber)
				return false;
		}
		else if (!pgstrom_device_expression(expr))
			return false;
	}
	*p_outer_relid = baserel->relid;
	*p_outer_quals = outer_quals;
	*p_parallel_aware = outer_path->parallel_aware;
	return true;
}

#if 1
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
#endif

/*
 * assign_gpuscan_session_info
 *
 * Gives some definitions to the static portion of GpuScan implementation
 */
void
assign_gpuscan_session_info(StringInfo buf, GpuTaskState_v2 *gts)
{
	CustomScan *cscan = (CustomScan *)gts->css.ss.ps.plan;

	Assert(pgstrom_plan_is_gpuscan((Plan *) cscan) ||
		   pgstrom_plan_is_gpujoin((Plan *) cscan) ||
		   pgstrom_plan_is_gpupreagg((Plan *) cscan));

	if (cscan->custom_scan_tlist != NIL)
	{
		TupleTableSlot *slot = gts->css.ss.ss_ScanTupleSlot;
		TupleDesc       tupdesc = slot->tts_tupleDescriptor;

		appendStringInfo(
			buf,
			"#define GPUSCAN_DEVICE_PROJECTION          1\n"
			"#define GPUSCAN_DEVICE_PROJECTION_NFIELDS  %d\n\n",
			tupdesc->natts);
	}
}

/*
 * gpuscan_create_scan_state - allocation of GpuScanState
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

/*
 * ExecInitGpuScan
 */
static void
ExecInitGpuScan(CustomScanState *node, EState *estate, int eflags)
{
	Relation		scan_rel = node->ss.ss_currentRelation;
	GpuContext_v2  *gcontext = NULL;
	GpuScanState   *gss = (GpuScanState *) node;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	GpuScanInfo	   *gs_info = deform_gpuscan_info(cscan);
	char		   *kern_define;
	ProgramId		program_id;
	bool			with_connection = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0);

	/* gpuscan should not have inner/outer plan right now */
	Assert(outerPlan(node) == NULL);
	Assert(innerPlan(node) == NULL);

	/* activate a GpuContext for CUDA kernel execution */
	gcontext = AllocGpuContext(with_connection);

	/*
	 * Re-initialization of scan tuple-descriptor and projection-info,
	 * because commit 1a8a4e5cde2b7755e11bde2ea7897bd650622d3e of
	 * PostgreSQL makes to assign result of ExecTypeFromTL() instead
	 * of ExecCleanTypeFromTL; that leads incorrect projection.
	 * So, we try to remove junk attributes from the scan-descriptor.
	 */
	if (cscan->custom_scan_tlist != NIL)
	{
		TupleDesc		scan_tupdesc;

		scan_tupdesc = ExecCleanTypeFromTL(cscan->custom_scan_tlist, false);
		ExecAssignScanType(&gss->gts.css.ss, scan_tupdesc);
		ExecAssignScanProjectionInfoWithVarno(&gss->gts.css.ss, INDEX_VAR);
	}

	/* setup common GpuTaskState fields */
	pgstromInitGpuTaskState(&gss->gts,
							gcontext,
							GpuTaskKind_GpuScan,
							gs_info->used_params,
							estate);
	gss->gts.cb_next_task   = gpuscan_next_task;
	gss->gts.cb_next_tuple  = gpuscan_next_tuple;
	gss->gts.cb_switch_task = gpuscan_switch_task;
	if (pgstrom_bulkexec_enabled &&
		gss->gts.css.ss.ps.qual == NIL &&		/* no host quals */
		gss->gts.css.ss.ps.ps_ProjInfo == NULL)	/* no host projection */
		gss->gts.cb_bulk_exec = pgstromBulkExecGpuTaskState;
	else
		gss->gts.cb_bulk_exec = NULL;	/* BulkExec not supported */
	/* estimated number of rows per block */
	gss->gts.outer_nrows_per_block = gs_info->nrows_per_block;

	/* initialize device tlist for CPU fallback */
	//FIXME: dev_tlist should not contain junk attribute
	gss->dev_tlist = (List *)
		ExecInitExpr((Expr *) cscan->custom_scan_tlist, &gss->gts.css.ss.ps);
	/* initialize device qualifiers also, for CPU fallback */
	elog(INFO, "dev_quals => %s", nodeToString(gs_info->dev_quals));
	//FIXME: dev_quals MUST reference raw tuple!
	gss->dev_quals = (List *)
		ExecInitExpr((Expr *) gs_info->dev_quals, &gss->gts.css.ss.ps);
	/* true, if device projection is needed */
	gss->dev_projection = (cscan->custom_scan_tlist != NIL);
	/* device projection related resource consumption */
	gss->proj_row_extra = gs_info->proj_row_extra;
	gss->proj_slot_extra = gs_info->proj_slot_extra;
	/* 'tableoid' should not change during relation scan */
	gss->scan_tuple.t_tableOid = RelationGetRelid(scan_rel);
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

	/* Get CUDA program and async build if any */
	kern_define = pgstrom_build_session_info(gs_info->extra_flags, &gss->gts);
	program_id = pgstrom_create_cuda_program(gcontext,
											 gs_info->extra_flags,
											 gs_info->kern_source,
											 kern_define,
											 false);
	gss->gts.program_id = program_id;
}

/*
 * gpuscan_exec_recheck
 *
 * Routine of EPQ recheck on GpuScan. If any, HostQual shall be checked
 * on ExecScan(), all we have to do here is recheck of device qualifier.
 */
static bool
ExecReCheckGpuScan(CustomScanState *node, TupleTableSlot *slot)
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

/*
 * ExecGpuScan
 */
static TupleTableSlot *
ExecGpuScan(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecGpuTaskState,
					(ExecScanRecheckMtd) ExecReCheckGpuScan);
}

/*
 * ExecEndGpuScan
 */
static void
ExecEndGpuScan(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *)node;

	/* inform status of GpuScan to the master backend */
    gss->gts.outer_instrument.nloops = 1;

	/* reset fallback resources */
	if (gss->base_slot)
		ExecDropSingleTupleTableSlot(gss->base_slot);
	pgstromReleaseGpuTaskState(&gss->gts);
}

/*
 * ExecReScanGpuScan
 */
static void
ExecReScanGpuScan(CustomScanState *node)
{
	GpuScanState	   *gss = (GpuScanState *) node;

	/* common rescan handling */
	pgstromRescanGpuTaskState(&gss->gts);
	/* rewind the position to read */
	gpuscanRewindScanChunk(&gss->gts);
}

/*
 * ExecGpuScanEstimateDSM - return required size of shared memory
 */
Size
ExecGpuScanEstimateDSM(CustomScanState *node,
					   ParallelContext *pcxt)
{
	EState	   *estate = node->ss.ps.state;
	Size		length = offsetof(GpuScanParallelDSM, pscan);

	if (node->ss.ss_currentRelation)
		length += heap_parallelscan_estimate(estate->es_snapshot);

	return length;
}

/*
 * ExecGpuScanInitDSM - initialize the coordinate memory on the master backend
 */
void
ExecGpuScanInitDSM(CustomScanState *node,
				   ParallelContext *pcxt,
				   void *coordinate)
{
	GpuTaskState_v2	   *gts = (GpuTaskState_v2 *) node;
	EState			   *estate = node->ss.ps.state;
	GpuScanParallelDSM *gpdsm = coordinate;
	pgstrom_perfmon	   *worker_pfm;
	int					instr_options = 0;

	if (node->ss.ps.instrument)
	{
		if (node->ss.ps.instrument->need_timer)
			instr_options |= INSTRUMENT_TIMER;
		if (node->ss.ps.instrument->need_bufusage)
			instr_options |= INSTRUMENT_BUFFERS;
	}

	/*
	 * setup of shared performance counter
	 *
	 * NOTE: DSM segment shall be released prior to the ExecEnd callback,
	 * so we have to allocate another shared memory segment at v9.6.
	 */
	if (!gts->worker_stat)
	{
		Size		len = offsetof(pgstromWorkerStatistics, gpujoin[0]);

		gts->worker_stat = dmaBufferAlloc(gts->gcontext, len);
		if (!gts->worker_stat)
			elog(ERROR, "out of shared memory");
		memset(gts->worker_stat, 0, len);
	}
	SpinLockInit(&gts->worker_stat->lock);
	InstrInit(&gts->worker_stat->worker_instrument, instr_options);
	worker_pfm = &gts->worker_stat->worker_pfm;
	worker_pfm->enabled = gts->pfm.enabled;
	worker_pfm->prime_in_gpucontext = gts->pfm.prime_in_gpucontext;
	worker_pfm->task_kind = gts->pfm.task_kind;

	gpdsm->worker_stat = gts->worker_stat;

	if (gts->css.ss.ss_currentRelation)
	{
		/* setup of parallel scan descriptor */
		heap_parallelscan_initialize(&gpdsm->pscan,
									 gts->css.ss.ss_currentRelation,
									 estate->es_snapshot);
		node->ss.ss_currentScanDesc =
			heap_beginscan_parallel(gts->css.ss.ss_currentRelation,
									&gpdsm->pscan);
		/* Try to choose NVMe-Strom, if available */
		PDS_init_heapscan_state(gts, gts->outer_nrows_per_block);
	}
}

/*
 * ExecGpuScanInitWorker - initialize GpuScan on the backend worker process
 */
void
ExecGpuScanInitWorker(CustomScanState *node,
					  shm_toc *toc,
					  void *coordinate)
{
	GpuTaskState_v2	   *gts = (GpuTaskState_v2 *) node;
	GpuScanParallelDSM *gpdsm = coordinate;

	gts->worker_stat = gpdsm->worker_stat;
	if (gts->css.ss.ss_currentRelation)
	{
		/* begin parallel sequential scan */
		node->ss.ss_currentScanDesc =
			heap_beginscan_parallel(gts->css.ss.ss_currentRelation,
									&gpdsm->pscan);
		/* Try to choose NVMe-Strom, if available */
		PDS_init_heapscan_state(gts, gts->outer_nrows_per_block);
	}
}

/*
 * ExplainGpuScan - EXPLAIN callback
 */
static void
ExplainGpuScan(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *) node;
	CustomScan	   *cscan = (CustomScan *) gss->gts.css.ss.ps.plan;
	GpuScanInfo	   *gs_info = deform_gpuscan_info(cscan);
	List		   *dcontext;
	List		   *dev_proj = NIL;
	char		   *exprstr;
	ListCell	   *lc;

	/* Set up deparsing context */
	dcontext = set_deparse_context_planstate(es->deparse_cxt,
											 (Node *)&gss->gts.css.ss.ps,
											 ancestors);
	/* Show device projection */
	foreach (lc, cscan->custom_scan_tlist)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (!tle->resjunk)
			dev_proj = lappend(dev_proj, tle->expr);
	}

	if (dev_proj != NIL)
	{
		exprstr = deparse_expression((Node *)dev_proj, dcontext,
									 es->verbose, false);
		ExplainPropertyText("GPU Projection", exprstr, es);
	}

	/* Show device filters */
	if (gs_info->dev_quals != NIL)
	{
		Node   *dev_quals = (Node *)make_ands_explicit(gs_info->dev_quals);

		exprstr = deparse_expression(dev_quals, dcontext,
									 es->verbose, false);
		ExplainPropertyText("GPU Filter", exprstr, es);
		if (gss->gts.outer_instrument.nfiltered1 > 0.0)
			ExplainPropertyFloat("Rows Removed by GPU Filter",
								 gss->gts.outer_instrument.nfiltered1 /
								 (gss->gts.outer_instrument.nloops + 1),
								 0, es);
	}
	/* common portion of EXPLAIN */
	pgstromExplainGpuTaskState(&gss->gts, es);
}

/*
 * gpuscan_create_task - constructor of GpuScanTask
 */
static GpuScanTask *
gpuscan_create_task(GpuScanState *gss,
					pgstrom_data_store *pds_src, int file_desc)
{
	TupleDesc			scan_tupdesc = GTS_GET_SCAN_TUPDESC(gss);
	GpuContext_v2	   *gcontext = gss->gts.gcontext;
	kern_resultbuf	   *kresults;
	pgstrom_data_store *pds_dst;
	GpuScanTask		   *gscan;
	Size				ntuples;
	Size				length;

	if (pds_src->kds.format != KDS_FORMAT_BLOCK)
		ntuples = pds_src->kds.nitems;
	else
	{
		/* we cannot know exact number of tuples unless scans block actually */
		ntuples = (pgstrom_chunk_size_margin *
				   (double)pds_src->kds.nrooms *
				   (double)pds_src->kds.nrows_per_block);
	}

	/*
	 * allocation of the destination buffer
	 */
	if (gss->gts.row_format)
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
			pds_dst = PDS_create_row(gcontext,
									 scan_tupdesc,
									 pds_src->kds.length +
									 gss->proj_row_extra * ntuples);
		}
	}
	else
	{
		pds_dst = PDS_create_slot(gcontext,
								  scan_tupdesc,
								  ntuples,
								  gss->proj_slot_extra * ntuples,
								  false);
	}

	/*
	 * allocation of pgstrom_gpuscan
	 */
	length = (STROMALIGN(offsetof(GpuScanTask, kern.kparams)) +
			  STROMALIGN(gss->gts.kern_params->length) +
			  STROMALIGN(offsetof(kern_resultbuf,
								  results[pds_dst ? 0 : ntuples])));
	gscan = dmaBufferAlloc(gcontext, length);
	memset(gscan, 0, (offsetof(GpuScanTask, kern) +
					  offsetof(kern_gpuscan, kparams)));
	pgstromInitGpuTask(&gss->gts, &gscan->task);
	gscan->task.file_desc = file_desc;
	gscan->dev_projection = gss->dev_projection;
	gscan->with_nvme_strom = (pds_src->kds.format == KDS_FORMAT_BLOCK &&
							  pds_src->nblocks_uncached > 0);
	gscan->pds_src = pds_src;
	gscan->pds_dst = pds_dst;

	/* kern_parambuf */
	memcpy(KERN_GPUSCAN_PARAMBUF(&gscan->kern),
		   gss->gts.kern_params,
		   gss->gts.kern_params->length);
	/* kern_resultbuf */
	kresults = KERN_GPUSCAN_RESULTBUF(&gscan->kern);
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = 1;
	if (gss->dev_quals != NIL || pds_src->kds.format == KDS_FORMAT_BLOCK)
		kresults->nrooms = ntuples;
	else
		kresults->all_visible = true;
	gscan->kresults = kresults;

	return gscan;
}

/*
 * heap_parallelscan_nextpage - see access/heap/heapam.c
 */
static BlockNumber
heap_parallelscan_nextpage(HeapScanDesc scan, cl_uint *p_nblocks_atonce)
{
	BlockNumber		page = InvalidBlockNumber;
	BlockNumber		sync_startpage = InvalidBlockNumber;
	BlockNumber		report_page = InvalidBlockNumber;
	ParallelHeapScanDesc parallel_scan;

	Assert(scan->rs_parallel);
	parallel_scan = scan->rs_parallel;

retry:
	/* Grab the spinlock. */
	SpinLockAcquire(&parallel_scan->phs_mutex);

	/*
	 * If the scan's startblock has not yet been initialized, we must do so
	 * now.  If this is not a synchronized scan, we just start at block 0, but
	 * if it is a synchronized scan, we must get the starting position from
	 * the synchronized scan machinery.  We can't hold the spinlock while
	 * doing that, though, so release the spinlock, get the information we
	 * need, and retry.  If nobody else has initialized the scan in the
	 * meantime, we'll fill in the value we fetched on the second time
	 * through.
	 */
	if (parallel_scan->phs_startblock == InvalidBlockNumber)
	{
		if (!parallel_scan->phs_syncscan)
			parallel_scan->phs_startblock = 0;
		else if (sync_startpage != InvalidBlockNumber)
			parallel_scan->phs_startblock = sync_startpage;
		else
		{
			SpinLockRelease(&parallel_scan->phs_mutex);
			sync_startpage = ss_get_location(scan->rs_rd, scan->rs_nblocks);
			goto retry;
		}
		parallel_scan->phs_cblock = parallel_scan->phs_startblock;
	}

	/*
	 * The current block number is the next one that needs to be scanned,
	 * unless it's InvalidBlockNumber already, in which case there are no more
	 * blocks to scan.  After remembering the current value, we must advance
	 * it so that the next call to this function returns the next block to be
	 * scanned.
	 */
	page = parallel_scan->phs_cblock;
	if (page != InvalidBlockNumber)
	{
		BlockNumber	phs_cblock = parallel_scan->phs_cblock;
		cl_uint		nr_blocks = (!p_nblocks_atonce ? 1 : *p_nblocks_atonce);

		Assert(phs_cblock < scan->rs_nblocks);
		Assert(nr_blocks > 0 && nr_blocks < RELSEG_SIZE);
		/* multiple blocks read never goes across segment boundary */
		if ((phs_cblock / RELSEG_SIZE) !=
			(phs_cblock + nr_blocks - 1) / RELSEG_SIZE)
			nr_blocks = RELSEG_SIZE - (phs_cblock % RELSEG_SIZE);
		/* stop multiple blocks reads if end of the relation */
		if (phs_cblock + nr_blocks > scan->rs_nblocks)
			nr_blocks = scan->rs_nblocks - phs_cblock;
		/* check end of the relation scan */
		if (phs_cblock + nr_blocks == scan->rs_nblocks &&
			parallel_scan->phs_startblock == 0)
		{
			parallel_scan->phs_cblock = InvalidBlockNumber;
			report_page = parallel_scan->phs_startblock;
		}
		else if (phs_cblock < parallel_scan->phs_startblock &&
				 phs_cblock + nr_blocks >= parallel_scan->phs_startblock)
		{
			nr_blocks = parallel_scan->phs_startblock - phs_cblock;
			parallel_scan->phs_cblock = InvalidBlockNumber;
			report_page = parallel_scan->phs_startblock;
		}
		else
		{
			parallel_scan->phs_cblock =
				(parallel_scan->phs_cblock + nr_blocks) % scan->rs_nblocks;
		}
		if (p_nblocks_atonce)
			*p_nblocks_atonce = nr_blocks;
	}

	/* Release the lock. */
	SpinLockRelease(&parallel_scan->phs_mutex);

	/*
	 * Report scan location.  Normally, we report the current page number.
	 * When we reach the end of the scan, though, we report the starting page,
	 * not the ending page, just so the starting positions for later scans
	 * doesn't slew backwards.  We only report the position at the end of the
	 * scan once, though: subsequent callers will have report nothing, since
	 * they will have page == InvalidBlockNumber.
	 */
	if (scan->rs_syncscan)
	{
		if (report_page == InvalidBlockNumber)
			report_page = page;
		if (report_page != InvalidBlockNumber)
			ss_report_location(scan->rs_rd, report_page);
	}
	return page;
}

/*
 * gpuscanExecScanChunk - read the relation by one chunk
 */
pgstrom_data_store *
gpuscanExecScanChunk(GpuTaskState_v2 *gts, int *p_filedesc)
{
	Relation		base_rel = gts->css.ss.ss_currentRelation;
	HeapScanDesc	scan;
	pgstrom_data_store *pds = NULL;
	int				filedesc = -1;
	struct timeval	tv1, tv2;

	/*
	 * Setup scan-descriptor, if the scan is not parallel, of if we're
	 * executing a scan that was intended to be parallel serially.
	 */
	if (!gts->css.ss.ss_currentScanDesc)
	{
		EState	   *estate = gts->css.ss.ps.state;

		gts->css.ss.ss_currentScanDesc = heap_beginscan(base_rel,
														estate->es_snapshot,
														0, NULL);
		/*
		 * Try to choose NVMe-Strom, if relation is deployed on the supported
		 * tablespace and expected total i/o size is enough large than cache-
		 * only scan.
		 */
		PDS_init_heapscan_state(gts, gts->outer_nrows_per_block);
	}
	scan = gts->css.ss.ss_currentScanDesc;

	InstrStartNode(&gts->outer_instrument);
	PFMON_BEGIN(&gts->pfm, &tv1);

	/* return NULL immediately if relation is empty */
	if (!scan->rs_inited &&
		(scan->rs_nblocks == 0 || scan->rs_numblocks == 0))
		goto out;

	/*
	 * MEMO: A key of i/o performance is consolidation of continuous block
	 * reads with a small number of system-call invocation.
	 * The default one-by-one block read logic tend to generate i/o request
	 * fragmentation, and it will increase submit of i/o request and slow
	 * down the performance. So, in case of NVMe-Strom under CPU parallel,
	 * we make advance the @scan->rs_cblock pointer by multiple blocks at
	 * once. It ensures block numbers to read are continuous, thus, i/o
	 * stack can load data blocks with minimum number of DMA requests.
	 */
	if (scan->rs_parallel && gts->nvme_sstate)
	{
		NVMEScanState  *nvme_sstate = gts->nvme_sstate;
		cl_uint			nblocks_atonce = nvme_sstate->nblocks_per_chunk;

		scan->rs_cblock = heap_parallelscan_nextpage(scan, &nblocks_atonce);
		/* already done? */
		if (!BlockNumberIsValid(scan->rs_cblock) || nblocks_atonce == 0)
			goto out;

		/* PDS allocation */
		pds = PDS_create_block(gts->gcontext,
							   RelationGetDescr(base_rel),
							   nvme_sstate);
		/* Scan nblocks */
		while (nblocks_atonce-- > 0)
		{
			if (!PDS_exec_heapscan(gts, pds, &filedesc))
				elog(ERROR, "Bug? failed to read block-%u", scan->rs_cblock);
			/* move to the next block */
			scan->rs_cblock++;
			if (scan->rs_cblock >= scan->rs_nblocks)
				scan->rs_cblock = 0;
			if (scan->rs_syncscan)
				ss_report_location(scan->rs_rd, scan->rs_cblock);
			/* end of the scan? */
			if (scan->rs_cblock == scan->rs_startblock ||
				(BlockNumberIsValid(scan->rs_numblocks) &&
				 --scan->rs_numblocks == 0))
				scan->rs_cblock = InvalidBlockNumber;
		}
	}
	else
	{
		/*
		 * Elsewhere, we will fetch blocks one-by-one manner.
		 */
		if (!scan->rs_inited)
		{
			if (scan->rs_parallel)
				scan->rs_cblock = heap_parallelscan_nextpage(scan, NULL);
			else
				scan->rs_cblock = scan->rs_startblock;
			scan->rs_inited = true;
		}

		/* already done? */
		if (!BlockNumberIsValid(scan->rs_cblock))
			goto out;

		if (gts->nvme_sstate)
			pds = PDS_create_block(gts->gcontext,
								   RelationGetDescr(base_rel),
								   gts->nvme_sstate);
		else
			pds = PDS_create_row(gts->gcontext,
								 RelationGetDescr(base_rel),
								 pgstrom_chunk_size());
		pds->kds.table_oid = RelationGetRelid(base_rel);

		/*
		 * TODO: We have to stop block insert if and when device projection
		 * will increase the buffer consumption than threshold.
		 * OR,
		 * specify smaller chunk by caller. GpuScan may become wise using
		 * adaptive buffer size control by row selevtivity on run-time.
		 */
		while (BlockNumberIsValid(scan->rs_cblock))
		{
			/* try to load scan->rs_cblock */
			Assert(scan->rs_cblock < scan->rs_nblocks);
			if (!PDS_exec_heapscan(gts, pds, &filedesc))
				break;

			/* move to the next block */
			if (scan->rs_parallel)
				scan->rs_cblock = heap_parallelscan_nextpage(scan, NULL);
			else
			{
				scan->rs_cblock++;
				if (scan->rs_cblock >= scan->rs_nblocks)
					scan->rs_cblock = 0;
				if (scan->rs_syncscan)
					ss_report_location(scan->rs_rd, scan->rs_cblock);
				/* end of the scan? */
				if (scan->rs_cblock == scan->rs_startblock ||
					(BlockNumberIsValid(scan->rs_numblocks) &&
					 --scan->rs_numblocks == 0))
					scan->rs_cblock = InvalidBlockNumber;
			}
		}
	}

	if (pds->kds.nitems == 0)
	{
		Assert(!BlockNumberIsValid(scan->rs_cblock));
		PDS_release(pds);
		pds = NULL;
	}
	else if (pds->kds.format == KDS_FORMAT_BLOCK &&
			 pds->kds.nitems < pds->kds.nrooms &&
			 pds->nblocks_uncached > 0)
	{
		/*
		 * MEMO: Special case handling if KDS_FORMAT_BLOCK was not filled up
		 * entirely. KDS_FORMAT_BLOCK has an array of block-number to support
		 * "ctid" system column, located on next to the KDS-head.
		 * Block-numbers of pre-loaded blocks (hit on shared buffer) are
		 * used from the head, and others (to be read from the file) are
		 * used from the tail. If nitems < nrooms, this array has a hole
		 * on the middle of array.
		 * So, we have to move later half of the array to close the hole
		 * and make a flat array.
		 */
		BlockNumber	   *block_nums
			= (BlockNumber *)KERN_DATA_STORE_BODY(&pds->kds);

		memmove(block_nums + (pds->kds.nitems - pds->nblocks_uncached),
				block_nums + (pds->kds.nrooms - pds->nblocks_uncached),
				sizeof(BlockNumber) * pds->nblocks_uncached);
	}
out:
	*p_filedesc = filedesc;
	PFMON_END(&gts->pfm, time_outer_load, &tv1, &tv2);
	InstrStopNode(&gts->outer_instrument,
				  !pds ? 0.0 : (double)pds->kds.nitems);
	return pds;
}

static void
gpuscan_switch_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{
	GpuScanTask		   *gscan = (GpuScanTask *) gtask;
	pgstrom_data_store *pds_src = gscan->pds_src;

	if (pds_src->nblocks_uncached > 0)
	{
		Assert(pds_src->kds.format == KDS_FORMAT_BLOCK);
		PDS_fillup_blocks(pds_src, gscan->task.file_desc);
	}

	if (!gscan->task.cpu_fallback)
		gts->outer_instrument.nfiltered1 += gscan->kern.nitems_filtered;

	/* move the server side perfmon counter to local pfm */
	if (gscan->task.perfmon)
	{
		gts->pfm.num_dma_send	+= gscan->num_dma_send;
		gts->pfm.num_dma_recv	+= gscan->num_dma_recv;
		gts->pfm.bytes_dma_send	+= gscan->bytes_dma_send;
		gts->pfm.bytes_dma_recv	+= gscan->bytes_dma_recv;
		gts->pfm.time_dma_send	+= gscan->time_dma_send;
		gts->pfm.time_dma_recv	+= gscan->time_dma_recv;

		gts->pfm.gscan.num_kern_main	+= gscan->num_kern_main;
		gts->pfm.gscan.tv_kern_main		+= gscan->tv_kern_main;
		gts->pfm.gscan.tv_kern_exec_quals += gscan->tv_kern_exec_quals;
		gts->pfm.gscan.tv_kern_projection += gscan->tv_kern_projection;
	}
}

/*
 * gpuscan_next_task
 */
static GpuTask_v2 *
gpuscan_next_task(GpuTaskState_v2 *gts)
{
	GpuScanState	   *gss = (GpuScanState *) gts;
	GpuScanTask		   *gscan;
	pgstrom_data_store *pds;
	int					filedesc;

	pds = gpuscanExecScanChunk(gts, &filedesc);
	if (!pds)
		return NULL;
	gscan = gpuscan_create_task(gss, pds, filedesc);

	return &gscan->task;
}

/*
 * gpuscan_next_tuple_fallback - GPU fallback case
 */
static TupleTableSlot *
gpuscan_next_tuple_fallback(GpuScanState *gss, GpuScanTask *gscan)
{
	pgstrom_data_store *pds_src = gscan->pds_src;
	ExprContext		   *econtext = gss->gts.css.ss.ps.ps_ExprContext;
	TupleTableSlot	   *slot = NULL;
	ExprDoneCond		is_done;

retry_next:
	ExecClearTuple(gss->base_slot);
	if (!PDS_fetch_tuple(gss->base_slot, pds_src, &gss->gts))
		return NULL;

	ResetExprContext(econtext);
	econtext->ecxt_scantuple = gss->base_slot;

	/*
	 * (1) - Evaluation of dev_quals if any
	 */
	if (gss->dev_quals != NIL)
	{
		if (!ExecQual(gss->dev_quals, econtext, false))
		{
			gss->gts.outer_instrument.nfiltered1 += 1.0;
			goto retry_next;
		}
	}

	/*
	 * (2) - Makes a projection if any
	 */
	if (!gss->base_proj)
		slot = gss->base_slot;
	else
	{
		slot = ExecProject(gss->base_proj, &is_done);
		if (is_done == ExprMultipleResult)
			gss->gts.css.ss.ps.ps_TupFromTlist = true;
		else if (is_done != ExprEndResult)
			gss->gts.css.ss.ps.ps_TupFromTlist = false;
	}
	return slot;
}

/*
 * gpuscan_next_tuple
 */
static TupleTableSlot *
gpuscan_next_tuple(GpuTaskState_v2 *gts)
{
	GpuScanState	   *gss = (GpuScanState *) gts;
	GpuScanTask		   *gscan = (GpuScanTask *) gts->curr_task;
	TupleTableSlot	   *slot = NULL;
	struct timeval		tv1, tv2;

	PFMON_BEGIN(&gss->gts.pfm, &tv1);
	if (gscan->task.cpu_fallback)
		slot = gpuscan_next_tuple_fallback(gss, gscan);
	else if (gscan->pds_dst)
	{
		pgstrom_data_store *pds_dst = gscan->pds_dst;

		slot = gss->gts.css.ss.ss_ScanTupleSlot;
		ExecClearTuple(slot);
		if (!PDS_fetch_tuple(slot, pds_dst, &gss->gts))
			slot = NULL;
	}
	else
	{
		pgstrom_data_store *pds_src = gscan->pds_src;
		kern_resultbuf	   *kresults = gscan->kresults;

		/*
		 * We should not inject GpuScan for all-visible with no device
		 * projection; GPU has no actual works in other words.
		 * NOTE: kresults->results[] keeps offset from the head of
		 * kds_src.
		 */
		Assert(!kresults->all_visible);
		if (gss->gts.curr_index < kresults->nitems)
		{
			HeapTuple	tuple = &gss->scan_tuple;
			cl_uint		kds_offset;

			kds_offset = kresults->results[gss->gts.curr_index++];
			if (pds_src->kds.format == KDS_FORMAT_ROW)
			{
				tuple->t_data = KDS_ROW_REF_HTUP(&pds_src->kds,
												 kds_offset,
												 &tuple->t_self,
												 &tuple->t_len);
			}
			else
			{
				tuple->t_data = KDS_BLOCK_REF_HTUP(&pds_src->kds,
												   kds_offset,
												   &tuple->t_self,
												   &tuple->t_len);
			}
			slot = gss->gts.css.ss.ss_ScanTupleSlot;
			ExecStoreTuple(tuple, slot, InvalidBuffer, false);
		}
	}
	PFMON_END(&gss->gts.pfm, time_materialize, &tv1, &tv2);

	return slot;
}

/*
 * gpuscanRewindScanChunk
 */
void
gpuscanRewindScanChunk(GpuTaskState_v2 *gts)
{
	InstrEndLoop(&gts->outer_instrument);
	Assert(gts->css.ss.ss_currentRelation != NULL);
	heap_rescan(gts->css.ss.ss_currentScanDesc, NULL);
	ExecScanReScan(&gts->css.ss);
}

/*
 * gpuscan_cleanup_cuda_resources
 */
static void
gpuscan_cleanup_cuda_resources(GpuScanTask *gscan)
{
	CUresult	rc;

	PFMON_EVENT_DESTROY(gscan, ev_dma_send_start);
	PFMON_EVENT_DESTROY(gscan, ev_dma_send_stop);
	PFMON_EVENT_DESTROY(gscan, ev_dma_recv_start);
	PFMON_EVENT_DESTROY(gscan, ev_dma_recv_stop);

	if (gscan->m_gpuscan)
	{
		rc = gpuMemFree_v2(gscan->task.gcontext, gscan->m_gpuscan);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on gpuMemFree: %s", errorText(rc));
	}

	if (gscan->with_nvme_strom && gscan->m_kds_src)
	{
		rc = gpuMemFreeIOMap(gscan->task.gcontext, gscan->m_kds_src);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on gpuMemFreeIOMap: %s", errorText(rc));
	}
	/* ensure pointers are NULL */
	gscan->kern_gpuscan_main = NULL;
	gscan->m_gpuscan = 0UL;
	gscan->m_kds_src = 0UL;
	gscan->m_kds_dst = 0UL;
}

static void
gpuscan_respond_task(CUstream stream, CUresult status, void *private)
{
	GpuScanTask	   *gscan = private;
	bool			is_urgent = false;

	/* OK, routine is called back in the usual context */
	if (status == CUDA_SUCCESS)
	{
		if (gscan->kern.kerror.errcode != StromError_Success)
		{
			if (pgstrom_cpu_fallback_enabled &&
				(gscan->kern.kerror.errcode == StromError_CpuReCheck ||
				 gscan->kern.kerror.errcode == StromError_DataStoreNoSpace))
			{
				gscan->task.cpu_fallback = true;
			}
			else if (!gscan->task.kerror.errcode)
			{
				gscan->task.kerror = gscan->kern.kerror;
			}
			is_urgent = true;
		}
	}
	else
	{
		if (!gscan->task.kerror.errcode)
		{
			gscan->task.kerror.errcode = status;
			gscan->task.kerror.kernel = StromKernel_CudaRuntime;
			gscan->task.kerror.lineno = 0;
		}
		is_urgent = true;
	}
	gpuservCompleteGpuTask(&gscan->task, is_urgent);
}

/*
 * gpuscan_process_task
 */
int
gpuscan_process_task(GpuTask_v2 *gtask,
					 CUmodule cuda_module,
					 CUstream cuda_stream)
{
	GpuScanTask	   *gscan = (GpuScanTask *) gtask;
	pgstrom_data_store *pds_src = gscan->pds_src;
	pgstrom_data_store *pds_dst = gscan->pds_dst;
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(&gscan->kern);
	cl_uint			ntuples;
	void		   *kern_args[5];
	size_t			offset;
	size_t			length;
	CUresult		rc;

	/*
	 * Lookup GPU kernel functions
	 */
	rc = cuModuleGetFunction(&gscan->kern_gpuscan_main,
							 cuda_module,
							 "gpuscan_main");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * Allocation of device memory
	 *
	 * MEMO: NVMe-Strom requires the DMA destination address is mapped to
	 * PCI BAR area, but it is usually a small window thus easy to run out.
	 * So, if we cannot allocate i/o mapped device memory, we try to read
	 * the blocks synchronously then kicks usual RAM->GPU DMA.
	 */
	if (gscan->with_nvme_strom)
	{
		rc = gpuMemAllocIOMap(gtask->gcontext,
							  &gscan->m_kds_src,
							  pds_src->kds.length);
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		{
			PDS_fillup_blocks(pds_src, gtask->peer_fdesc);
			gscan->m_kds_src = 0UL;
			gscan->with_nvme_strom = false;
		}
		else if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocIOMap: %s", errorText(rc));
	}

	length = offset = GPUMEMALIGN(KERN_GPUSCAN_LENGTH(&gscan->kern));
	if (!gscan->with_nvme_strom)
		length += GPUMEMALIGN(pds_src->kds.length);
	if (pds_dst)
		length += GPUMEMALIGN(pds_dst->kds.length);
	rc = gpuMemAlloc_v2(gtask->gcontext, &gscan->m_gpuscan, length);
	if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		goto out_of_resource;
	else if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAlloc: %s", errorText(rc));

	if (!gscan->with_nvme_strom)
	{
		Assert(!gscan->m_kds_src);
		gscan->m_kds_src = gscan->m_gpuscan + offset;
		offset += GPUMEMALIGN(pds_src->kds.length);
	}

	if (pds_dst)
		gscan->m_kds_dst = gscan->m_gpuscan + offset;
	else
		gscan->m_kds_dst = 0UL;

	/*
	 * Creation of event objects, if needed
	 */
	PFMON_EVENT_CREATE(gscan, ev_dma_send_start);
	PFMON_EVENT_CREATE(gscan, ev_dma_send_stop);
	PFMON_EVENT_CREATE(gscan, ev_dma_recv_start);
	PFMON_EVENT_CREATE(gscan, ev_dma_recv_stop);

	/*
	 * OK, enqueue a series of requests
	 */
	PFMON_EVENT_RECORD(gscan, ev_dma_send_start, cuda_stream);

	offset = KERN_GPUSCAN_DMASEND_OFFSET(&gscan->kern);
	length = KERN_GPUSCAN_DMASEND_LENGTH(&gscan->kern);
	rc = cuMemcpyHtoDAsync(gscan->m_gpuscan,
						   (char *)&gscan->kern + offset,
						   length,
                           cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gscan->bytes_dma_send += length;
    gscan->num_dma_send++;

	/*  kern_data_store *kds_src */
	if (pds_src->kds.format == KDS_FORMAT_ROW || !gscan->with_nvme_strom)
	{
		length = pds_src->kds.length;
		rc = cuMemcpyHtoDAsync(gscan->m_kds_src,
							   &pds_src->kds,
							   length,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		gscan->bytes_dma_send += length;
		gscan->num_dma_send++;
	}
	else
	{
		gpuMemCopyFromSSDAsync(&gscan->task,
							   gscan->m_kds_src,
							   pds_src,
							   cuda_stream);
		gpuMemCopyFromSSDWait(&gscan->task,
							  cuda_stream);
	}

	/* kern_data_store *kds_dst, if any */
	if (pds_dst)
	{
		length = KERN_DATA_STORE_HEAD_LENGTH(&pds_dst->kds);
		rc = cuMemcpyHtoDAsync(gscan->m_kds_dst,
							   &pds_dst->kds,
							   length,
                               cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		gscan->bytes_dma_send += length;
		gscan->num_dma_send++;
	}
	PFMON_EVENT_RECORD(gscan, ev_dma_send_stop, cuda_stream);

	/*
	 * KERNEL_FUNCTION(void)
	 * gpuscan_main(kern_gpuscan *kgpuscan,
	 *              kern_data_store *kds_src,
	 *              kern_data_store *kds_dst)
	 */
	kern_args[0] = &gscan->m_gpuscan;
	kern_args[1] = &gscan->m_kds_src;
	kern_args[2] = &gscan->m_kds_dst;

	rc = cuLaunchKernel(gscan->kern_gpuscan_main,
						1, 1, 1,
						1, 1, 1,
						0,
						cuda_stream,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	gscan->num_kern_main++;

	/*
	 * Recv DMA call
	 */
	PFMON_EVENT_RECORD(gscan, ev_dma_recv_start, cuda_stream);
	if (pds_src->kds.format != KDS_FORMAT_BLOCK)
		ntuples = pds_src->kds.nitems;
	else
		ntuples = kresults->nrooms;
	offset = KERN_GPUSCAN_DMARECV_OFFSET(&gscan->kern);
	length = KERN_GPUSCAN_DMARECV_LENGTH(&gscan->kern, pds_dst ? 0 : ntuples);
	rc = cuMemcpyDtoHAsync((char *)&gscan->kern + offset,
						   gscan->m_gpuscan + offset,
						   length,
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
    gscan->bytes_dma_recv += length;
    gscan->num_dma_recv++;

	/*
	 * NOTE: varlena variables in the result references pds_src as buffer
	 * of variable length datum. So, if and when all the blocks are NOT
	 * yet loaded to the pds_src and pds_dst may contain varlena variables,
	 * we need to write back blocks unread from GPU to CPU/RAM.
	 */
	if (pds_src->kds.format == KDS_FORMAT_BLOCK &&
		pds_src->nblocks_uncached > 0 &&
		(!pds_dst
		 ? pds_src->kds.has_notbyval
		 : (pds_dst->kds.has_notbyval &&
			pds_dst->kds.format == KDS_FORMAT_SLOT)))
	{
		cl_uint	nr_loaded = pds_src->kds.nitems - pds_src->nblocks_uncached;

		offset = ((char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds_src->kds,
													   nr_loaded) -
				  (char *)&pds_src->kds);
		length = pds_src->nblocks_uncached * BLCKSZ;
		rc = cuMemcpyDtoHAsync((char *)&pds_src->kds + offset,
							   gscan->m_kds_src + offset,
							   length,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		gscan->bytes_dma_recv += length;
		gscan->num_dma_recv++;
		/*
		 * NOTE: Once GPU-to-GPU DMA gets completed, "uncached" blocks are
		 * no longer uncached, so we clear the @nblocks_uncached not to
		 * write back GPU RAM twice even if CPU fallback.
		 */
		pds_src->nblocks_uncached = 0;
	}

	if (pds_dst)
	{
		Assert(pds_dst->kds.format == KDS_FORMAT_ROW ||
			   pds_dst->kds.format == KDS_FORMAT_SLOT);
		rc = cuMemcpyDtoHAsync(&pds_dst->kds,
							   gscan->m_kds_dst,
							   pds_dst->kds.length,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
		gscan->bytes_dma_recv += pds_dst->kds.length;
		gscan->num_dma_recv++;
	}
	PFMON_EVENT_RECORD(gscan, ev_dma_recv_stop, cuda_stream);

	/*
	 * register the callback
	 */
	rc = cuStreamAddCallback(cuda_stream,
							 gpuscan_respond_task,
							 gscan, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));
	return 0;

out_of_resource:
	fprintf(stderr, "out of resource\n");
	gpuscan_cleanup_cuda_resources(gscan);
	return 1;
}

/*
 * gpuscan_complete_task
 */
int
gpuscan_complete_task(GpuTask_v2 *gtask)
{
	GpuScanTask	   *gscan = (GpuScanTask *) gtask;

	/* raise any kernel internal error */
	Assert(gtask->kerror.errcode == StromError_Success ||
		   gtask->kerror.errcode == StromError_CpuReCheck);
	if (gtask->kerror.errcode != StromError_Success)
		elog(ERROR, "GpuScan kernel internal error: %s",
			 errorTextKernel(&gtask->kerror));

	PFMON_EVENT_ELAPSED(gscan, time_dma_send,
						gscan->ev_dma_send_start,
						gscan->ev_dma_send_stop);
	PFMON_EVENT_ELAPSED(gscan, tv_kern_main,
						gscan->ev_dma_send_stop,
						gscan->ev_dma_recv_start);
	PFMON_EVENT_ELAPSED(gscan, time_dma_recv,
						gscan->ev_dma_recv_start,
						gscan->ev_dma_recv_stop);
	gscan->tv_kern_exec_quals = gscan->kern.pfm.tv_kern_exec_quals;
	gscan->tv_kern_projection = gscan->kern.pfm.tv_kern_projection;
skip_perfmon:
	gpuscan_cleanup_cuda_resources(gscan);

	return 0;
}

/*
 * gpuscan_release_task
 */
void
gpuscan_release_task(GpuTask_v2 *gtask)
{
	GpuScanTask	   *gscan = (GpuScanTask *) gtask;

	if (gscan->pds_src)
		PDS_release(gscan->pds_src);
	if (gscan->pds_dst)
		PDS_release(gscan->pds_dst);
	dmaBufferFree(gscan);
}

/*
 * pgstrom_init_gpuscan
 */
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
	gpuscan_path_methods.PlanCustomPath		= PlanGpuScanPath;

	/* setup plan methods */
	memset(&gpuscan_plan_methods, 0, sizeof(gpuscan_plan_methods));
	gpuscan_plan_methods.CustomName			= "GpuScan";
	gpuscan_plan_methods.CreateCustomScanState = gpuscan_create_scan_state;
	RegisterCustomScanMethods(&gpuscan_plan_methods);

	/* setup exec methods */
	memset(&gpuscan_exec_methods, 0, sizeof(gpuscan_exec_methods));
	gpuscan_exec_methods.CustomName         = "GpuScan";
	gpuscan_exec_methods.BeginCustomScan    = ExecInitGpuScan;
	gpuscan_exec_methods.ExecCustomScan     = ExecGpuScan;
	gpuscan_exec_methods.EndCustomScan      = ExecEndGpuScan;
	gpuscan_exec_methods.ReScanCustomScan   = ExecReScanGpuScan;
	gpuscan_exec_methods.EstimateDSMCustomScan = ExecGpuScanEstimateDSM;
	gpuscan_exec_methods.InitializeDSMCustomScan = ExecGpuScanInitDSM;
	gpuscan_exec_methods.InitializeWorkerCustomScan = ExecGpuScanInitWorker;
	gpuscan_exec_methods.ExplainCustomScan  = ExplainGpuScan;

	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = gpuscan_add_scan_path;
}
