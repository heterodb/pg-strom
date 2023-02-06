/*
 * gpu_scan.c
 *
 * Sequential scan accelerated with GPU processors
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"

/* static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
static CustomPathMethods	gpuscan_path_methods;
static CustomScanMethods	gpuscan_plan_methods;
static CustomExecMethods	gpuscan_exec_methods;
static bool					enable_gpuscan;		/* GUC */
static bool					enable_pullup_outer_scan;	/* GUC */

/*
 * form_gpuscan_info
 *
 * GpuScanInfo --> custom_private/custom_exprs
 */
void
form_gpuscan_info(CustomScan *cscan, GpuScanInfo *gs_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, bms_to_pglist(gs_info->gpu_cache_devs));
	privs = lappend(privs, bms_to_pglist(gs_info->gpu_direct_devs));
	privs = lappend(privs, __makeByteaConst(gs_info->kexp_kvars_load));
	privs = lappend(privs, __makeByteaConst(gs_info->kexp_scan_quals));
	privs = lappend(privs, __makeByteaConst(gs_info->kexp_projection));
	privs = lappend(privs, gs_info->kvars_depth);
	privs = lappend(privs, gs_info->kvars_resno);
	privs = lappend(privs, makeInteger(gs_info->extra_flags));
	privs = lappend(privs, makeInteger(gs_info->extra_bufsz));
	privs = lappend(privs, bms_to_pglist(gs_info->outer_refs));
	exprs = lappend(exprs, gs_info->used_params);
	exprs = lappend(exprs, gs_info->dev_quals);
	privs = lappend(privs, __makeFloat(gs_info->scan_tuples));
	privs = lappend(privs, __makeFloat(gs_info->scan_rows));
	privs = lappend(privs, makeInteger(gs_info->index_oid));
	privs = lappend(privs, gs_info->index_conds);
	exprs = lappend(exprs, gs_info->index_quals);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

/*
 * deform_gpuscan_info
 *
 * custom_private/custom_exprs -> GpuScanInfo
 */
void
deform_gpuscan_info(GpuScanInfo *gs_info, CustomScan *cscan)
{
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;

	gs_info->gpu_cache_devs  = bms_from_pglist(list_nth(privs, pindex++));
	gs_info->gpu_direct_devs = bms_from_pglist(list_nth(privs, pindex++));
	gs_info->kexp_kvars_load = __getByteaConst(list_nth(privs, pindex++));
	gs_info->kexp_scan_quals = __getByteaConst(list_nth(privs, pindex++));
	gs_info->kexp_projection = __getByteaConst(list_nth(privs, pindex++));
	gs_info->kvars_depth     = list_nth(privs, pindex++);
	gs_info->kvars_resno     = list_nth(privs, pindex++);
	gs_info->extra_flags     = intVal(list_nth(privs, pindex++));
	gs_info->extra_bufsz     = intVal(list_nth(privs, pindex++));
	gs_info->outer_refs      = bms_from_pglist(list_nth(privs, pindex++));
	gs_info->used_params     = list_nth(exprs, eindex++);
	gs_info->dev_quals       = list_nth(exprs, eindex++);
	gs_info->scan_tuples     = floatVal(list_nth(privs, pindex++));
	gs_info->scan_rows       = floatVal(list_nth(privs, pindex++));
	gs_info->index_oid       = intVal(list_nth(privs, pindex++));
	gs_info->index_conds     = list_nth(privs, pindex++);
	gs_info->index_quals     = list_nth(exprs, eindex++);
}

/*
 * GpuScanSharedState
 */
typedef struct
{
	/* for arrow_fdw file scan */
	pg_atomic_uint32	af_rbatch_index;
	pg_atomic_uint32	af_rbatch_nload;	/* # of loaded record-batches */
	pg_atomic_uint32	af_rbatch_nskip;	/* # of skipped record-batches */
	/* for gpu_cache cache scan */
	pg_atomic_uint32	gc_fetch_count;
	/* for block-based regular table scan */
	BlockNumber			pbs_nblocks;		/* # blocks in relation at start of scan */
	slock_t				pbs_mutex;			/* lock of the fields below */
	BlockNumber			pbs_startblock;		/* starting block number */
	BlockNumber			pbs_nallocated;		/* # of blocks allocated to workers */
	/* common parallel table scan descriptor */
	ParallelTableScanDescData phscan;
} GpuScanSharedState;

/*
 * GpuScanState
 */
typedef struct
{
	pgstromTaskState	pts;
	GpuScanInfo			gs_info;
	XpuCommand		   *xcmd_req;	/* request command buffer */
	size_t				xcmd_len;
} GpuScanState;

/*
 * xpuOperatorCostRatio
 */
double
xpuOperatorCostRatio(uint32_t devkind)
{
	double	xpu_ratio;

	switch (devkind)
	{
		case DEVKIND__NVIDIA_GPU:
			/* GPU computation cost */
			if (cpu_operator_cost > 0.0)
				xpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
			else if (pgstrom_gpu_operator_cost == 0.0)
				xpu_ratio = 1.0;
			else
				xpu_ratio = disable_cost;	/* very large but still finite */
			break;

		case DEVKIND__NVIDIA_DPU:
			/* DPU computation cost */
			if (cpu_operator_cost > 0.0)
				xpu_ratio = pgstrom_dpu_operator_cost / cpu_operator_cost;
			else if (pgstrom_dpu_operator_cost == 0.0)
				xpu_ratio = 1.0;
			else
				xpu_ratio = disable_cost;	/* very large but still finite */
			break;

		default:
			xpu_ratio = 1.0;
			break;
	}
	return xpu_ratio;
}

/*
 * xpuTupleCost
 */
Cost
xpuTupleCost(uint32_t devkind)
{
	switch (devkind)
	{
		case DEVKIND__NVIDIA_GPU:
			return pgstrom_gpu_tuple_cost;
		case DEVKIND__NVIDIA_DPU:
			return pgstrom_dpu_tuple_cost;
		default:
			break;
	}
	return 0.0;		/* CPU don't need xPU-->Host DMA */
}

/*
 * considerXpuScanPathParams
 */
bool
considerXpuScanPathParams(PlannerInfo *root,
						  RelOptInfo  *baserel,
						  uint32_t devkind,
						  bool parallel_aware,
						  List *dev_quals,
						  List *host_quals,
						  int  *p_parallel_nworkers,
						  Oid  *p_brin_index_oid,
						  List **p_brin_index_conds,
						  List **p_brin_index_quals,
						  Cost *p_startup_cost,
						  Cost *p_run_cost,
						  Cost *p_final_cost,
						  const Bitmapset **p_gpu_cache_devs,
						  const Bitmapset **p_gpu_direct_devs,
						  const DpuStorageEntry **p_ds_entry)
{
	RangeTblEntry  *rte = root->simple_rte_array[baserel->relid];
	IndexOptInfo   *indexOpt = NULL;
	List		   *indexConds = NIL;
	List		   *indexQuals = NIL;
	int64_t			indexNBlocks = 0;
	int				parallel_nworkers = 0;
	double			parallel_divisor = 1.0;
	const Bitmapset *gpu_cache_devs = NULL;
	const Bitmapset *gpu_direct_devs = NULL;
	const DpuStorageEntry *ds_entry = NULL;
	Cost			startup_cost = 0.0;
	Cost			disk_cost = 0.0;
	Cost			run_cost = 0.0;
	Cost			final_cost = 0.0;
	double			avg_seq_page_cost;
	double			spc_seq_page_cost;
	double			spc_rand_page_cost;
	double			xpu_ratio = xpuOperatorCostRatio(devkind);
	double			xpu_tuple_cost = xpuTupleCost(devkind);
	QualCost		qcost;
	double			ntuples;
	double			selectivity;

	/*
	 * Brief check towards the supplied baserel
	 */
	Assert(IS_SIMPLE_REL(baserel));
	Assert(devkind == DEVKIND__NVIDIA_GPU || devkind == DEVKIND__NVIDIA_DPU);
	switch (rte->relkind)
	{
		case RELKIND_RELATION:
		case RELKIND_MATVIEW:
			if (get_relation_am(rte->relid, true) == HEAP_TABLE_AM_OID)
				break;
			return false;
		case RELKIND_FOREIGN_TABLE:
			if (baseRelIsArrowFdw(baserel))
				break;
			return false;
		default:
			return false;
	}

	/*
	 * CPU Parallel parameters
	 */
	if (parallel_aware)
	{
		double	leader_contribution;

		parallel_nworkers = compute_parallel_worker(baserel,
													baserel->pages, -1,
													max_parallel_workers_per_gather);
		if (parallel_nworkers <= 0)
			return false;
		parallel_divisor = (double)parallel_nworkers;
		if (parallel_leader_participation)
		{
			leader_contribution = 1.0 - (0.3 * (double)parallel_nworkers);
			if (leader_contribution > 0.0)
				parallel_divisor += leader_contribution;
		}
	}

	/*
	 * Check device special disk-scan mode
	 */
	get_tablespace_page_costs(baserel->reltablespace,
							  &spc_rand_page_cost,
							  &spc_seq_page_cost);	
	switch (devkind)
	{
		case DEVKIND__NVIDIA_GPU:
			startup_cost += pgstrom_gpu_setup_cost;
			/* Is GPU-Cache available? */
			//gpu_cache_devs = baseRelHasGpuCache(root, baserel);
			/* Is GPU-Direct SQL available? */
			gpu_direct_devs = GetOptimalGpuForBaseRel(root, baserel);
			if (gpu_cache_devs)
				avg_seq_page_cost = 0;
			else if (gpu_direct_devs)
				avg_seq_page_cost = spc_seq_page_cost * (1.0 - baserel->allvisfrac) +
					pgstrom_gpu_direct_seq_page_cost * baserel->allvisfrac;
			else
				avg_seq_page_cost = spc_seq_page_cost;
			break;

		case DEVKIND__NVIDIA_DPU:
			startup_cost += pgstrom_dpu_setup_cost;
			/* Is DPU-attached Storage available? */
			ds_entry = GetOptimalDpuForBaseRel(root, baserel);
			if (!ds_entry)
				return false;
			avg_seq_page_cost = (spc_seq_page_cost * (1.0 - baserel->allvisfrac) +
								 pgstrom_dpu_seq_page_cost * baserel->allvisfrac);
			break;

		default:
			/* should not happen */
			avg_seq_page_cost = spc_seq_page_cost;
			break;
	}
	/*
	 * NOTE: ArrowGetForeignRelSize() already discount baserel->pages according
	 * to the referenced columns, to adjust total amount of disk i/o.
	 * So, we have nothing special to do here.
	 */
	disk_cost = avg_seq_page_cost * baserel->pages;
	ntuples =  baserel->tuples;

	/*
	 * Is BRIN-index available?
	 */
	indexOpt = pgstromTryFindBrinIndex(root, baserel,
									   &indexConds,
									   &indexQuals,
									   &indexNBlocks);
	if (indexOpt)
	{
		Cost	index_disk_cost = (cost_brin_bitmap_build(root,
														  baserel,
														  indexOpt,
														  indexQuals) +
								   avg_seq_page_cost * indexNBlocks);
		if (disk_cost > index_disk_cost)
		{
			disk_cost = index_disk_cost;
			if (baserel->pages > 0)
				ntuples *= (double)indexNBlocks / (double)baserel->pages;
		}
		else
			indexOpt = NULL;	/* disables BRIN-index if no benefit */
	}
	run_cost += disk_cost;

	/*
	 * Cost for xPU qualifiers
	 */
	if (dev_quals != NIL)
	{
		cost_qual_eval_node(&qcost, (Node *)dev_quals, root);
		startup_cost += qcost.startup;
		run_cost += qcost.per_tuple * xpu_ratio * ntuples / parallel_divisor;

		selectivity = clauselist_selectivity(root,
											 dev_quals,
											 baserel->relid,
											 JOIN_INNER,
											 NULL);
		ntuples *= selectivity;		/* rows after dev_quals */
	}

	/*
	 * Cost for DMA receive (xPU-->Host)
	 */
	final_cost += xpu_tuple_cost * ntuples;
	
	/*
	 * Cost for host qualifiers
	 */
	if (host_quals != NIL)
	{
		cost_qual_eval_node(&qcost, (Node *)dev_quals, root);
		startup_cost += qcost.startup;
		final_cost += qcost.per_tuple * ntuples / parallel_divisor;

		selectivity = clauselist_selectivity(root,
											 host_quals,
											 baserel->relid,
											 JOIN_INNER,
											 NULL);
		ntuples *= selectivity;		/* rows after host_quals */
	}
	/*
	 * Cost for host projection
	 */
	startup_cost += baserel->reltarget->cost.startup;
	final_cost += baserel->reltarget->cost.per_tuple * baserel->rows;

	/* Write back the result */
	if (p_parallel_nworkers)
		*p_parallel_nworkers = parallel_nworkers;
	if (p_brin_index_oid)
		*p_brin_index_oid = (indexOpt ? indexOpt->indexoid : InvalidOid);
	if (p_brin_index_conds)
		*p_brin_index_conds = (indexOpt ? indexConds : NIL);
	if (p_brin_index_quals)
		*p_brin_index_quals = (indexOpt ? indexQuals : NIL);
	if (p_startup_cost)
		*p_startup_cost = startup_cost;
	if (p_run_cost)
		*p_run_cost = run_cost;
	if (p_final_cost)
		*p_final_cost = final_cost;
	if (p_gpu_cache_devs)
		*p_gpu_cache_devs = gpu_cache_devs;
	if (p_gpu_direct_devs)
		*p_gpu_direct_devs = gpu_direct_devs;
	if (p_ds_entry)
		*p_ds_entry = ds_entry;

	return true;
}

/*
 * GpuScanAddScanPath
 */
static void
GpuScanAddScanPath(PlannerInfo *root,
				   RelOptInfo *baserel,
				   Index rtindex,
				   RangeTblEntry *rte)
{
	List	   *input_rels_tlist;
	List	   *dev_quals = NIL;
	List	   *host_quals = NIL;
	ParamPathInfo *param_info;
	ListCell   *lc;

	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);
	/* nothing to do, if either PG-Strom or GpuScan is not enabled */
	if (!pgstrom_enabled || !enable_gpuscan)
		return;
	/* We already proved the relation empty, so nothing more to do */
	if (is_dummy_rel(baserel))
		return;
	/* It is the role of built-in Append node */
	if (rte->inh)
		return;
	/*
	 * check whether the qualifier can run on GPU device
	 */
	input_rels_tlist = list_make1(makeInteger(baserel->relid));
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo *rinfo = lfirst(lc);

		if (pgstrom_gpu_expression(rinfo->clause,
								   input_rels_tlist,
								   NULL))
			dev_quals = lappend(dev_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}
	/*
	 * check parametalized qualifiers
	 */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		foreach (lc, param_info->ppi_clauses)
		{
			RestrictInfo *rinfo = lfirst(lc);

			if (pgstrom_gpu_expression(rinfo->clause,
									   input_rels_tlist,
									   NULL))
				dev_quals = lappend(dev_quals, rinfo);
			else
				host_quals = lappend(host_quals, rinfo);
		}
	}
	
	/* Creation of GpuScan path */
	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		GpuScanInfo		gs_data;
		GpuScanInfo	   *gs_info;
		CustomPath	   *cpath;
		ParamPathInfo  *param_info = NULL;
		int				parallel_nworkers = 0;
		Cost			startup_cost = 0.0;
		Cost			run_cost = 0.0;
		Cost			final_cost = 0.0;

		memset(&gs_data, 0, sizeof(GpuScanInfo));
		if (!considerXpuScanPathParams(root,
									   baserel,
									   DEVKIND__NVIDIA_GPU,
									   try_parallel > 0,	/* parallel_aware */
									   dev_quals,
									   host_quals,
									   &parallel_nworkers,
									   &gs_data.index_oid,
									   &gs_data.index_conds,
									   &gs_data.index_quals,
									   &startup_cost,
									   &run_cost,
									   &final_cost,
									   &gs_data.gpu_cache_devs,
									   &gs_data.gpu_direct_devs,
									   NULL))
			return;

		/* setup GpuScanInfo (Path phase) */
		gs_info = pmemdup(&gs_data, sizeof(GpuScanInfo));
		cpath = makeNode(CustomPath);
		cpath->path.pathtype = T_CustomScan;
		cpath->path.parent = baserel;
		cpath->path.pathtarget = baserel->reltarget;
		cpath->path.param_info = param_info;
		cpath->path.parallel_aware = (try_parallel > 0);
		cpath->path.parallel_safe = baserel->consider_parallel;
		cpath->path.parallel_workers = parallel_nworkers;
		cpath->path.rows = (param_info ? param_info->ppi_rows : baserel->rows);
		cpath->path.startup_cost = startup_cost;
		cpath->path.total_cost = startup_cost + run_cost + final_cost;
		cpath->path.pathkeys = NIL; /* unsorted results */
		cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
		cpath->custom_paths = NIL;
		cpath->custom_private = list_make1(gs_info);
		cpath->methods = &gpuscan_path_methods;

		if (custom_path_remember(root, baserel, (try_parallel > 0), cpath))
		{
			if (try_parallel == 0)
				add_path(baserel, &cpath->path);
			else
				add_partial_path(baserel, &cpath->path);
		}
	}
}

/*
 * gpuscan_build_projection - make custom_scan_tlist
 */
typedef struct
{
	List	   *tlist_dev;
	List	   *input_rels_tlist;
	bool		resjunk;
} build_projection_context;

static bool
__gpuscan_build_projection_walker(Node *node, void *__priv)
{
	build_projection_context *context = __priv;
	ListCell   *lc;

	if (!node)
		return false;
	foreach (lc, context->tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (equal(node, tle->expr))
			return false;
	}
	if (IsA(node, Var) ||
		pgstrom_gpu_expression((Expr *)node, context->input_rels_tlist, NULL))
	{
		AttrNumber		resno = list_length(context->tlist_dev) + 1;
		TargetEntry	   *tle = makeTargetEntry((Expr *)node,
											  resno,
											  NULL,
											  context->resjunk);
		context->tlist_dev = lappend(context->tlist_dev, tle);
		return false;
	}
	return expression_tree_walker(node, __gpuscan_build_projection_walker, __priv);
}

static List *
gpuscan_build_projection(RelOptInfo *baserel,
						 List *tlist,
						 List *host_quals,
						 List *dev_quals,
						 List *input_rels_tlist)
{
	build_projection_context context;
	List	   *vars_list;
	ListCell   *lc;

	memset(&context, 0, sizeof(build_projection_context));
	context.input_rels_tlist = input_rels_tlist;

	if (tlist != NIL)
	{
		foreach (lc, tlist)
		{
			TargetEntry *tle = lfirst(lc);

			if (IsA(tle->expr, Const) || IsA(tle->expr, Param))
				continue;
			__gpuscan_build_projection_walker((Node *)tle->expr, &context);
		}
	}
	else
	{
		/*
		 * When ProjectionPath is on CustomPath(GpuScan), it always assigns
		 * the result of build_path_tlist() and calls PlanCustomPath method
		 * with tlist == NIL.
		 * So, if GPU projection wants to make something valuable, we need
		 * to check path-target.
		 * Also don't forget all the Var-nodes to be added must exist at
		 * the custom_scan_tlist because setrefs.c references this list.
		 */
		foreach (lc, baserel->reltarget->exprs)
		{
			Node   *node = lfirst(lc);

			if (IsA(node, Const) || IsA(node, Param))
				continue;
			__gpuscan_build_projection_walker(node, &context);
		}
	}
	vars_list = pull_vars_of_level((Node *)host_quals, 0);
	foreach (lc, vars_list)
		__gpuscan_build_projection_walker((Node *)lfirst(lc), &context);

	context.resjunk = true;
	vars_list = pull_vars_of_level((Node *)dev_quals, 0);
	foreach (lc, vars_list)
		__gpuscan_build_projection_walker((Node *)lfirst(lc), &context);

	return context.tlist_dev;
}

/*
 * sort_device_qualifiers
 */
void
sort_device_qualifiers(List *dev_quals_list, List *dev_costs_list)
{
	int			nitems = list_length(dev_quals_list);
	ListCell  **dev_quals = alloca(sizeof(ListCell *) * nitems);
	int		   *dev_costs = alloca(sizeof(int) * nitems);
	int			i, j, k;
	ListCell   *lc1, *lc2;

	i = 0;
	forboth (lc1, dev_quals_list,
			 lc2, dev_costs_list)
	{
		dev_quals[i] = lc1;
		dev_costs[i] = lfirst_int(lc2);
		i++;
	}
	Assert(i == nitems);

	for (i=0; i < nitems; i++)
	{
		int		dcost = dev_costs[i];
		void   *dqual = dev_quals[i]->ptr_value;

		k = i;
		for (j=i+1; j < nitems; j++)
		{
			if (dcost > dev_costs[j])
			{
				dcost = dev_costs[j];
				dqual = dev_quals[j]->ptr_value;
				k = j;
			}
		}

		if (i != k)
		{
			dev_costs[k] = dev_costs[i];
			dev_costs[i] = dcost;
			dev_quals[k]->ptr_value = dev_quals[i]->ptr_value;
			dev_quals[i]->ptr_value = dqual;
		}
	}
}

/*
 * PlanXpuScanPathCommon
 */
CustomScan *
PlanXpuScanPathCommon(PlannerInfo *root,
					  RelOptInfo  *baserel,
					  CustomPath  *best_path,
					  List        *tlist,
					  List        *clauses,
					  GpuScanInfo *gs_info,
					  const CustomScanMethods *xpuscan_plan_methods)
{
	codegen_context context;
	CustomScan	   *cscan;
	List		   *host_quals = NIL;
	List		   *dev_quals = NIL;
	List		   *dev_costs = NIL;
	List		   *tlist_dev = NIL;
	List		   *input_rels_tlist = list_make1(makeInteger(baserel->relid));
	Bitmapset	   *outer_refs = NULL;
	ListCell	   *cell;

	/*
	 * Distribution of clauses into device executable and others.
	 */
	foreach (cell, clauses)
	{
		RestrictInfo *rinfo = lfirst(cell);
		int		devcost;

		Assert(exprType((Node *)rinfo->clause) == BOOLOID);
		if (pgstrom_gpu_expression(rinfo->clause, input_rels_tlist, &devcost))
		{
			dev_quals = lappend(dev_quals, rinfo);
			dev_costs = lappend_int(dev_costs, devcost);
		}
		else
		{
			host_quals = lappend(host_quals, rinfo);
		}
		pull_varattnos((Node *)rinfo->clause, baserel->relid, &outer_refs);
	}
	if (dev_quals == NIL)
		elog(ERROR, "GpuScan: Bug? no device executable qualifiers are given");
	sort_device_qualifiers(dev_quals, dev_costs);
	dev_quals = extract_actual_clauses(dev_quals, false);
	host_quals = extract_actual_clauses(host_quals, false);
	/* pickup referenced attributes */
	outer_refs = pickup_outer_referenced(root, baserel, outer_refs);
	/* code generation for WHERE-clause */
	codegen_context_init(&context, DEVKIND__NVIDIA_GPU);
	context.input_rels_tlist = input_rels_tlist;
	gs_info->kexp_scan_quals = codegen_build_scan_quals(&context, dev_quals);
	/* code generation for the Projection */
	tlist_dev = gpuscan_build_projection(baserel,
										 tlist,
										 host_quals,
										 dev_quals,
										 input_rels_tlist);
	gs_info->kexp_projection = codegen_build_projection(&context, tlist_dev);
	gs_info->kexp_kvars_load = codegen_build_scan_loadvars(&context);
	gs_info->kvars_depth = context.kvars_depth;
	gs_info->kvars_resno = context.kvars_resno;
	gs_info->extra_flags = context.extra_flags;
	gs_info->extra_bufsz = context.extra_bufsz;
	gs_info->used_params = context.used_params;
	gs_info->dev_quals = dev_quals;
	gs_info->outer_refs = outer_refs;
	gs_info->scan_tuples = baserel->tuples;
	gs_info->scan_rows = baserel->rows;

	/*
	 * Build CustomScan(GpuScan) node
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = host_quals;
	cscan->scan.scanrelid = baserel->relid;
	cscan->flags = best_path->flags;
	cscan->methods = xpuscan_plan_methods;
	cscan->custom_plans = NIL;
	cscan->custom_scan_tlist = tlist_dev;

	return cscan;
}

/*
 * PlanGpuScanPath
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

	/* sanity checks */
	Assert(baserel->relid > 0 &&
		   baserel->rtekind == RTE_RELATION &&
		   custom_children == NIL);

	cscan = PlanXpuScanPathCommon(root,
								  baserel,
								  best_path,
								  tlist,
								  clauses,
								  gs_info,
								  &gpuscan_plan_methods);
	form_gpuscan_info(cscan, gs_info);
	return &cscan->scan.plan;
}

/*
 * CreateGpuScanState
 */
static Node *
CreateGpuScanState(CustomScan *cscan)
{
	GpuScanState   *gss = palloc0(sizeof(GpuScanState));

	Assert(cscan->methods == &gpuscan_plan_methods);

	/* Set tag and executor callbacks */
	NodeSetTag(gss, T_CustomScanState);
	gss->pts.devkind = DEVKIND__NVIDIA_GPU;
	gss->pts.css.flags = cscan->flags;
	gss->pts.css.methods = &gpuscan_exec_methods;
	deform_gpuscan_info(&gss->gs_info, cscan);

	return (Node *)gss;
}

/*
 * ExecInitGpuScan
 */
static void
ExecInitGpuScan(CustomScanState *node, EState *estate, int eflags)
{
	GpuScanState   *gss = (GpuScanState *)node;

	/* sanity checks */
	Assert(node->ss.ss_currentRelation != NULL &&
		   outerPlanState(node) == NULL &&
		   innerPlanState(node) == NULL);
	pgstromExecInitTaskState(&gss->pts,
							 DEVKIND__NVIDIA_GPU,
							 gss->gs_info.dev_quals,
							 gss->gs_info.outer_refs,
							 gss->gs_info.index_oid,
							 gss->gs_info.index_conds,
							 gss->gs_info.index_quals);
	gss->pts.cb_cpu_fallback = ExecFallbackCpuScan;
}

/*
 * GpuScanReCheckTuple
 */
static bool
GpuScanReCheckTuple(GpuScanState *gss, TupleTableSlot *epq_slot)
{
	/*
	 * NOTE: Only immutable operators/functions are executable
	 * on the GPU devices, so its decision will never changed.
	 */
	return true;
}

/*
 * ExecGpuScan
 */
static TupleTableSlot *
ExecGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	if (!gss->pts.ps_state)
		pgstromSharedStateInitDSM(&gss->pts, NULL, NULL);
	if (!gss->pts.conn)
	{
		const XpuCommand *session;

		session = pgstromBuildSessionInfo(&gss->pts,
										  gss->gs_info.used_params,
										  gss->gs_info.extra_bufsz,
										  gss->gs_info.kvars_depth,
										  gss->gs_info.kvars_resno,
										  gss->gs_info.kexp_kvars_load,
										  gss->gs_info.kexp_scan_quals,
										  NULL,		/* join-load-vars */
										  NULL,		/* join-quals */
										  NULL,		/* hash-values */
										  NULL,		/* gist-join */
										  gss->gs_info.kexp_projection,
										  0);
		gpuClientOpenSession(&gss->pts, gss->pts.optimal_gpus, session);
	}
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecTaskState,
					(ExecScanRecheckMtd) GpuScanReCheckTuple);
}

/*
 * ExecEndGpuScan
 */
static void
ExecEndGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	pgstromExecEndTaskState(&gss->pts);
}

/*
 * ExecReScanGpuScan
 */
static void
ExecReScanGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	pgstromExecResetTaskState(&gss->pts);
}

/*
 * EstimateGpuScanDSM
 */
static Size
EstimateGpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return pgstromSharedStateEstimateDSM((pgstromTaskState *)node);
}

/*
 * InitializeGpuScanDSM
 */
static void
InitializeGpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *dsm_addr)
{
	pgstromSharedStateInitDSM((pgstromTaskState *)node, pcxt, dsm_addr);
}

/*
 * InitGpuScanWorker
 */
static void
InitGpuScanWorker(CustomScanState *node, shm_toc *toc, void *dsm_addr)
{
	pgstromSharedStateAttachDSM((pgstromTaskState *)node, dsm_addr);
}

/*
 * ExecShutdownGpuScan
 */
static void
ExecShutdownGpuScan(CustomScanState *node)
{
	pgstromSharedStateShutdownDSM((pgstromTaskState *)node);
}

/*
 * ExplainGpuScan
 */
static void
ExplainGpuScan(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *) node;
	GpuScanInfo	   *gs_info = &gss->gs_info;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	List		   *dcontext;

	/* setup deparsing context */
	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										ancestors);
	pgstromExplainScanState(&gss->pts, es,
							dcontext,
							cscan->custom_scan_tlist,
							gs_info->dev_quals,
							gs_info->scan_tuples,
                            gs_info->scan_rows);
	/* XPU Code (if verbose) */
	pgstrom_explain_xpucode(&gss->pts.css, es, dcontext,
							"Scan Var-Loads Code",
							gs_info->kexp_kvars_load);
	pgstrom_explain_xpucode(&gss->pts.css, es, dcontext,
							"Scan Quals Code",
							gs_info->kexp_scan_quals);
	pgstrom_explain_xpucode(&gss->pts.css, es, dcontext,
							"DPU Projection Code",
							gs_info->kexp_projection);

	pgstromExplainTaskState(&gss->pts, es, dcontext);
}

/*
 * ExecFallbackCpuScan
 */
void
ExecFallbackCpuScan(pgstromTaskState *pts, HeapTuple tuple)
{
	TupleTableSlot *scan_slot = pts->base_slot;
	bool			should_free = false;

	ExecForceStoreHeapTuple(tuple, scan_slot, false);
	/* check WHERE-clause if any */
	if (pts->base_quals)
	{
		ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;

		econtext->ecxt_scantuple = scan_slot;
		ResetExprContext(econtext);
		if (!ExecQual(pts->base_quals, econtext))
			return;
	}
	/* apply Projection if any */
	if (pts->base_proj)
	{
		TupleTableSlot *proj_slot = ExecProject(pts->base_proj);

		tuple = ExecFetchSlotHeapTuple(proj_slot, false, &should_free);
	}
	/* save the tuple on the fallback buffer */
	pgstromStoreFallbackTuple(pts, tuple);
	if (should_free)
		pfree(tuple);
}

/*
 * Handle GpuScan Commands
 */
void
gpuservHandleGpuScanExec(gpuClient *gclient, XpuCommand *xcmd)
{
	kern_session_info *session = gclient->session;
	kern_gpuscan	*kgscan = NULL;
	const char		*kds_src_pathname = NULL;
	strom_io_vector *kds_src_iovec = NULL;
	kern_data_store *kds_src = NULL;
	kern_data_extra	*kds_extra = NULL;
	kern_data_store *kds_dst = NULL;
	kern_data_store *kds_dst_head = NULL;
	kern_data_store **kds_dst_array = NULL;
	int				kds_dst_nrooms = 0;
	int				kds_dst_nitems = 0;
	int				n_kvars_slots = session->kvars_slot_width;
	CUfunction		f_kern_gpuscan;
	const gpuMemChunk *chunk = NULL;
	CUdeviceptr		m_kds_src = 0UL;
	CUdeviceptr		dptr;
	CUresult		rc;
	int				grid_sz;
	int				block_sz;
	unsigned int	shmem_sz;
	unsigned int	n_warps;
	size_t			sz;
	void		   *kern_args[5];

	if (xcmd->u.scan.kds_src_pathname)
		kds_src_pathname = (char *)xcmd + xcmd->u.scan.kds_src_pathname;
	if (xcmd->u.scan.kds_src_iovec)
		kds_src_iovec = (strom_io_vector *)((char *)xcmd + xcmd->u.scan.kds_src_iovec);
	if (xcmd->u.scan.kds_src_offset)
		kds_src = (kern_data_store *)((char *)xcmd + xcmd->u.scan.kds_src_offset);
	if (xcmd->u.scan.kds_dst_offset)
		kds_dst_head = (kern_data_store *)((char *)xcmd + xcmd->u.scan.kds_dst_offset);

	if (!kds_src)
	{
		gpuClientELog(gclient, "KDS_FORMAT_COLUMN is not yet implemented");
		return;
	}
	else if (kds_src->format == KDS_FORMAT_ROW)
	{
		m_kds_src = (CUdeviceptr)kds_src;
	}
	else if (kds_src->format == KDS_FORMAT_BLOCK)
	{
		if (kds_src_pathname && kds_src_iovec)
		{
			chunk = gpuservLoadKdsBlock(gclient,
										kds_src,
										kds_src_pathname,
										kds_src_iovec);
			if (!chunk)
				return;
			m_kds_src = chunk->m_devptr;
		}
		else
		{
			Assert(kds_src->block_nloaded == kds_src->nitems);
			m_kds_src = (CUdeviceptr)kds_src;
		}
	}
	else if (kds_src->format == KDS_FORMAT_ARROW)
	{
		if (kds_src_iovec->nr_chunks == 0)
			m_kds_src = (CUdeviceptr)kds_src;
		else
		{
			if (!kds_src_pathname)
			{
				gpuClientELog(gclient, "GpuScan: arrow file is missing");
				return;
			}
			chunk = gpuservLoadKdsArrow(gclient,
										kds_src,
										kds_src_pathname,
										kds_src_iovec);
			if (!chunk)
				return;
			m_kds_src = chunk->m_devptr;
		}
	}
	else
	{
		gpuClientELog(gclient, "unknown GpuScan Source format (%c)",
					  kds_src->format);
		return;
	}

	rc = cuModuleGetFunction(&f_kern_gpuscan,
							 gclient->cuda_module,
							 "kern_gpuscan_main");
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuModuleGetFunction: %s",
					   cuStrError(rc));
		goto bailout;
	}

	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 &shmem_sz,
							 f_kern_gpuscan,
							 0,
							 __KERN_WARP_CONTEXT_UNITSZ_BASE(0));
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on gpuOptimalBlockSize: %s",
					   cuStrError(rc));
		goto bailout;
	}
	
	/*
	 * Allocation of the control structure
	 */
	grid_sz = Min(grid_sz, (kds_src->nitems + block_sz - 1) / block_sz);
//	block_sz = 64;
//	grid_sz = 1;
	n_warps = grid_sz * (block_sz / WARPSIZE);

	sz = offsetof(kern_gpuscan, data) +
		KERN_WARP_CONTEXT_UNITSZ(0, n_kvars_slots) * n_warps;
	rc = cuMemAllocManaged(&dptr, sz, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuMemAllocManaged(%lu): %s",
					   sz, cuStrError(rc));
		goto bailout;
	}
	kgscan = (kern_gpuscan *)dptr;
	memset(kgscan, 0, offsetof(kern_gpuscan, data));
	kgscan->grid_sz  = grid_sz;
	kgscan->block_sz = block_sz;

	/* prefetch source KDS, if managed memory */
	if (!chunk)
	{
		rc = cuMemPrefetchAsync((CUdeviceptr)kds_src,
								kds_src->length,
								CU_DEVICE_PER_THREAD,
								CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientFatal(gclient, "failed on cuMemPrefetchAsync: %s",
						   cuStrError(rc));
			goto bailout;
		}
	}

	/*
	 * Allocation of the destination buffer
	 */
resume_kernel:
	sz = KDS_HEAD_LENGTH(kds_dst_head) + PGSTROM_CHUNK_SIZE;
	rc = cuMemAllocManaged(&dptr, sz, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuMemAllocManaged(%lu): %s",
					   sz, cuStrError(rc));
		goto bailout;
	}
	kds_dst = (kern_data_store *)dptr;
	memcpy(kds_dst, kds_dst_head, KDS_HEAD_LENGTH(kds_dst_head));
	kds_dst->length = sz;
	if (kds_dst_nitems >= kds_dst_nrooms)
	{
		kern_data_store **kds_dst_temp;

		kds_dst_nrooms = 2 * kds_dst_nrooms + 10;
		kds_dst_temp = alloca(sizeof(kern_data_store *) * kds_dst_nrooms);
		if (kds_dst_nitems > 0)
			memcpy(kds_dst_temp,
				   kds_dst_array,
				   sizeof(kern_data_store *) * kds_dst_nitems);
		kds_dst_array = kds_dst_temp;
	}
	kds_dst_array[kds_dst_nitems++] = kds_dst;
	
	/*
	 * Launch kernel
	 */
	kern_args[0] = &gclient->session;
	kern_args[1] = &kgscan;
	kern_args[2] = &m_kds_src;
	kern_args[3] = &kds_extra;
	kern_args[4] = &kds_dst;

	rc = cuLaunchKernel(f_kern_gpuscan,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						shmem_sz,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuLaunchKernel: %s", cuStrError(rc));
		goto bailout;
	}

	rc = cuEventRecord(CU_EVENT_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuEventRecord: %s", cuStrError(rc));
		goto bailout;
	}

	/* point of synchronization */
	rc = cuEventSynchronize(CU_EVENT_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuEventSynchronize: %s", cuStrError(rc));
		goto bailout;
	}

	/* status check */
	if (kgscan->kerror.errcode == ERRCODE_STROM_SUCCESS)
	{
		XpuCommand	resp;

		if (kgscan->suspend_count > 0)
		{
			if (gpuServiceGoingTerminate())
			{
				gpuClientFatal(gclient, "GpuService is going to terminate during GpuScan kernel suspend/resume");
				goto bailout;
			}
			/* restore warp context from the previous state */
			kgscan->resume_context = true;
			kgscan->suspend_count = 0;
			fprintf(stderr, "suspend / resume happen\n");
			goto resume_kernel;
		}
		/* send back status and kds_dst */
		memset(&resp, 0, offsetof(XpuCommand, u.results));
		resp.magic = XpuCommandMagicNumber;
		resp.tag   = XpuCommandTag__Success;
		resp.u.results.chunks_nitems = kds_dst_nitems;
		resp.u.results.chunks_offset = offsetof(XpuCommand, u.results.stats.scan.data);
		resp.u.results.stats.scan.nitems_in = kgscan->nitems_in;
		resp.u.results.stats.scan.nitems_out = kgscan->nitems_out;
		gpuClientWriteBack(gclient,
						   &resp, resp.u.results.chunks_offset,
						   kds_dst_nitems, kds_dst_array);
	}
	else if (kgscan->kerror.errcode == ERRCODE_CPU_FALLBACK)
	{
		XpuCommand	resp;

		/* send back kds_src with XpuCommandTag__CPUFallback */
		memset(&resp, 0, offsetof(XpuCommand, u.results));
		resp.magic = XpuCommandMagicNumber;
		resp.tag   = XpuCommandTag__CPUFallback;
		resp.u.results.chunks_nitems = 1;
		resp.u.results.chunks_offset = offsetof(XpuCommand, u.results.stats.scan.data);
		gpuClientWriteBack(gclient,
						   &resp, resp.u.results.chunks_offset,
						   1, &kds_src);
	}
	else
	{
		/* send back error status */
		__gpuClientELogRaw(gclient, &kgscan->kerror);
	}
bailout:
	if (kgscan)
	{
		rc = cuMemFree((CUdeviceptr)kgscan);
		if (rc != CUDA_SUCCESS)
			fprintf(stderr, "warning: failed on cuMemFree: %s\n", cuStrError(rc));
	}
	if (chunk)
		gpuMemFree(chunk);
	while (kds_dst_nitems > 0)
	{
		kds_dst = kds_dst_array[--kds_dst_nitems];

		rc = cuMemFree((CUdeviceptr)kds_dst);
		if (rc != CUDA_SUCCESS)
			fprintf(stderr, "warning: failed on cuMemFree: %s\n", cuStrError(rc));
	}
}

/*
 * pgstrom_init_gpuscan
 */
void
pgstrom_init_gpu_scan(void)
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
    gpuscan_plan_methods.CreateCustomScanState = CreateGpuScanState;
    RegisterCustomScanMethods(&gpuscan_plan_methods);

    /* setup exec methods */
    memset(&gpuscan_exec_methods, 0, sizeof(gpuscan_exec_methods));
    gpuscan_exec_methods.CustomName			= "GpuScan";
    gpuscan_exec_methods.BeginCustomScan	= ExecInitGpuScan;
    gpuscan_exec_methods.ExecCustomScan		= ExecGpuScan;
    gpuscan_exec_methods.EndCustomScan		= ExecEndGpuScan;
    gpuscan_exec_methods.ReScanCustomScan	= ExecReScanGpuScan;
    gpuscan_exec_methods.EstimateDSMCustomScan = EstimateGpuScanDSM;
    gpuscan_exec_methods.InitializeDSMCustomScan = InitializeGpuScanDSM;
    gpuscan_exec_methods.InitializeWorkerCustomScan = InitGpuScanWorker;
    gpuscan_exec_methods.ShutdownCustomScan	= ExecShutdownGpuScan;
    gpuscan_exec_methods.ExplainCustomScan	= ExplainGpuScan;

	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = GpuScanAddScanPath;
}
