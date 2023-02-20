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

/* static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
static CustomPathMethods	gpuscan_path_methods;
static CustomScanMethods	gpuscan_plan_methods;
static CustomExecMethods	gpuscan_exec_methods;
static bool					enable_gpuscan;		/* GUC */
static bool					enable_pullup_outer_scan;	/* GUC */

/*
 * consider_xpuscan_path_params
 */
bool
consider_xpuscan_path_params(PlannerInfo *root,
							 RelOptInfo  *baserel,
							 uint32_t task_kind,
							 List *dev_quals,
							 List *host_quals,
							 bool parallel_aware,
							 int *p_parallel_nworkers,
							 Cost *p_startup_cost,
							 Cost *p_run_cost,
							 pgstromPlanInfo *pp_info)
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
	double			xpu_ratio;
	double			xpu_tuple_cost;
	QualCost		qcost;
	double			ntuples;
	double			selectivity;

	/*
	 * Brief check towards the supplied baserel
	 */
	Assert(IS_SIMPLE_REL(baserel));
	Assert((task_kind & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU ||
		   (task_kind & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU);
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
	if ((task_kind & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU)
	{
		xpu_ratio = pgstrom_gpu_operator_ratio();
		xpu_tuple_cost = pgstrom_gpu_tuple_cost;
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
	}
	else if ((task_kind & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU)
	{
		xpu_ratio = pgstrom_dpu_operator_ratio();
		xpu_tuple_cost = pgstrom_dpu_tuple_cost;
		startup_cost += pgstrom_dpu_setup_cost;
		/* Is DPU-attached Storage available? */
		if (rte->relkind == RELKIND_FOREIGN_TABLE)
			ds_entry = GetOptimalDpuForArrowFdw(root, baserel);
		else
			ds_entry = GetOptimalDpuForBaseRel(root, baserel);
		if (!ds_entry)
			return false;
		avg_seq_page_cost = (spc_seq_page_cost * (1.0 - baserel->allvisfrac) +
							 pgstrom_dpu_seq_page_cost * baserel->allvisfrac);
	}
	else
	{
		elog(ERROR, "Bug? unsupported task_kind: %08x", task_kind);
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
	if (p_startup_cost)
		*p_startup_cost = startup_cost;
	if (p_run_cost)
		*p_run_cost = run_cost;
	if (pp_info)
	{
		pp_info->task_kind = task_kind;
		pp_info->gpu_cache_devs = gpu_cache_devs;
		pp_info->gpu_direct_devs = gpu_direct_devs;
		pp_info->ds_entry = ds_entry;
		if (indexOpt)
		{
			pp_info->brin_index_oid = indexOpt->indexoid;
			pp_info->brin_index_conds = indexConds;
			pp_info->brin_index_quals = indexQuals;
		}
		else
		{
			pp_info->brin_index_oid = InvalidOid;
			pp_info->brin_index_conds = NIL;
			pp_info->brin_index_quals = NIL;
		}
		pp_info->parallel_divisor = parallel_divisor;
		pp_info->final_cost = final_cost;
	}
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
		pgstromPlanInfo	pp_data;
		pgstromPlanInfo	*pp_info;
		CustomPath	   *cpath;
		ParamPathInfo  *param_info = NULL;
		int				parallel_nworkers = 0;
		Cost			startup_cost = 0.0;
		Cost			run_cost = 0.0;

		memset(&pp_data, 0, sizeof(pgstromPlanInfo));
		if (!consider_xpuscan_path_params(root,
										  baserel,
										  TASK_KIND__GPUSCAN,
										  dev_quals,
										  host_quals,
										  try_parallel > 0,
										  &parallel_nworkers,
										  &startup_cost,
										  &run_cost,
										  &pp_data))
			return;

		/* setup GpuScanInfo (Path phase) */
		pp_info = pmemdup(&pp_data, sizeof(pgstromPlanInfo));
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
		cpath->path.total_cost = startup_cost + run_cost + pp_info->final_cost;
		cpath->path.pathkeys = NIL; /* unsorted results */
		cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
		cpath->custom_paths = NIL;
		cpath->custom_private = list_make1(pp_info);
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
					  pgstromPlanInfo *pp_info,
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
	codegen_context_init(&context, pp_info->task_kind);
	context.input_rels_tlist = input_rels_tlist;
	pp_info->kexp_scan_quals = codegen_build_scan_quals(&context, dev_quals);
	/* code generation for the Projection */
	tlist_dev = gpuscan_build_projection(baserel,
										 tlist,
										 host_quals,
										 dev_quals,
										 input_rels_tlist);
	pp_info->kexp_projection = codegen_build_projection(&context, tlist_dev);
	pp_info->kexp_scan_kvars_load = codegen_build_scan_loadvars(&context);
	pp_info->kvars_depth = context.kvars_depth;
	pp_info->kvars_resno = context.kvars_resno;
	pp_info->extra_flags = context.extra_flags;
	pp_info->extra_bufsz = context.extra_bufsz;
	pp_info->used_params = context.used_params;
	pp_info->scan_quals = dev_quals;
	pp_info->outer_refs = outer_refs;
	pp_info->scan_tuples = baserel->tuples;
	pp_info->scan_rows = baserel->rows;

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
	pgstromPlanInfo *pp_info = linitial(best_path->custom_private);
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
								  pp_info,
								  &gpuscan_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
}

/*
 * CreateGpuScanState
 */
static Node *
CreateGpuScanState(CustomScan *cscan)
{
	pgstromTaskState *pts = palloc0(sizeof(pgstromTaskState));

	Assert(cscan->methods == &gpuscan_plan_methods);
	/* Set tag and executor callbacks */
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &gpuscan_exec_methods;
	pts->task_kind = TASK_KIND__GPUSCAN;
	pts->pp_info = deform_pgstrom_plan_info(cscan);
	Assert(pts->task_kind == pts->pp_info->task_kind);

	return (Node *)pts;
}

/*
 * ExecGpuScan
 */
static TupleTableSlot *
ExecGpuScan(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	if (!pts->ps_state)
		pgstromSharedStateInitDSM(&pts->css, NULL, NULL);
	if (!pts->conn)
	{
		const XpuCommand *session
			= pgstromBuildSessionInfo(pts, 0);
		gpuClientOpenSession(pts, pts->optimal_gpus, session);
	}
	return pgstromExecTaskState(pts);
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
    gpuscan_exec_methods.BeginCustomScan	= pgstromExecInitTaskState;
    gpuscan_exec_methods.ExecCustomScan		= ExecGpuScan;
    gpuscan_exec_methods.EndCustomScan		= pgstromExecEndTaskState;
    gpuscan_exec_methods.ReScanCustomScan	= pgstromExecResetTaskState;
    gpuscan_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
    gpuscan_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
    gpuscan_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
    gpuscan_exec_methods.ShutdownCustomScan	= pgstromSharedStateShutdownDSM;
    gpuscan_exec_methods.ExplainCustomScan	= pgstromExplainTaskState;

	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = GpuScanAddScanPath;
}
