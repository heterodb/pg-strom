/*
 * gpu_scan.c
 *
 * Sequential scan accelerated with GPU processors
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
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
static bool					enable_gpuscan = false;		/* GUC */
static CustomPathMethods	dpuscan_path_methods;
static CustomScanMethods	dpuscan_plan_methods;
static CustomExecMethods	dpuscan_exec_methods;
static bool					enable_dpuscan = false;		/* GUC */

/*
 * pgstrom_is_gpuscan_path
 */
bool
pgstrom_is_gpuscan_path(const Path *path)
{
	if (IsA(path, CustomPath))
	{
		const CustomPath *cpath = (const CustomPath *)path;

		if (cpath->methods == &gpuscan_path_methods)
			return true;
	}
	return false;
}

/*
 * pgstrom_is_gpuscan_plan
 */
bool
pgstrom_is_gpuscan_plan(const Plan *plan)
{
	if (IsA(plan, CustomScan))
	{
		const CustomScan *cscan = (const CustomScan *)cscan;

		if (cscan->methods == &gpuscan_plan_methods)
			return true;
	}
	return false;
}

/*
 * pgstrom_is_gpuscan_state
 */
bool
pgstrom_is_gpuscan_state(const PlanState *ps)
{
	if (IsA(ps, CustomScanState))
	{
		const CustomScanState *css = (const CustomScanState *)ps;

		if (css->methods == &gpuscan_exec_methods)
			return true;
	}
	return false;
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
 * buildSimpleScanPlanInfo
 */
static pgstromPlanInfo *
__buildSimpleScanPlanInfo(PlannerInfo *root,
						  RelOptInfo *baserel,
						  uint32_t xpu_task_flags,
						  bool parallel_path,
						  List *dev_quals,
						  List *host_quals,
						  Cardinality scan_nrows)
{
	RangeTblEntry  *rte = root->simple_rte_array[baserel->relid];
	pgstromPlanInfo *pp_info;
	const DpuStorageEntry *ds_entry = NULL;
	Bitmapset	   *outer_refs = NULL;
	IndexOptInfo   *indexOpt = NULL;
	List		   *indexConds = NIL;
	List		   *indexQuals = NIL;
	int64_t			indexNBlocks = 0;
	int				parallel_nworkers = 0;
	double			parallel_divisor = 1.0;
	double			spc_seq_page_cost;
	double			spc_rand_page_cost;
	Cost			startup_cost = 0.0;
	Cost			disk_cost = 0.0;
	Cost			run_cost = 0.0;
	Cost			final_cost = 0.0;
	double			avg_seq_page_cost;
	double			xpu_ratio;
	double			xpu_tuple_cost;
	QualCost		qcost;
	double			ntuples = baserel->tuples;
	double			selectivity;

	/*
	 * CPU Parallel parameters
	 */
	if (parallel_path)
	{
		double	leader_contribution;

		parallel_nworkers = compute_parallel_worker(baserel,
													baserel->pages, -1,
													max_parallel_workers_per_gather);
		if (parallel_nworkers <= 0)
			return NULL;
		parallel_divisor = (double)parallel_nworkers;
		if (parallel_leader_participation)
		{
			leader_contribution = 1.0 - (0.3 * (double)parallel_nworkers);
			if (leader_contribution > 0.0)
				parallel_divisor += leader_contribution;
		}
		/* discount # of rows to be produced per backend */
		ntuples    /= parallel_divisor;
		scan_nrows /= parallel_divisor;
	}

	/*
	 * Check device special disk-scan mode
	 */
	get_tablespace_page_costs(baserel->reltablespace,
							  &spc_rand_page_cost,
							  &spc_seq_page_cost);
	if ((xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU)
	{
		xpu_ratio = pgstrom_gpu_operator_ratio();
		xpu_tuple_cost = pgstrom_gpu_tuple_cost;
		startup_cost += pgstrom_gpu_setup_cost;

		if (baseRelHasGpuCache(root, baserel) >= 0)
		{
			/* assume GPU-Cache is available */
			avg_seq_page_cost = 0;
		}
		else if (GetOptimalGpuForBaseRel(root, baserel) != 0UL)
		{
			/* assume GPU-Direct SQL is available */
			avg_seq_page_cost = spc_seq_page_cost * (1.0 - baserel->allvisfrac) +
				pgstrom_gpu_direct_seq_page_cost * baserel->allvisfrac;
		}
		else
		{
			/* elsewhere, use PostgreSQL's storage layer */
			avg_seq_page_cost = spc_seq_page_cost;
		}
	}
	else if ((xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU)
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
			return NULL;
		avg_seq_page_cost = (spc_seq_page_cost * (1.0 - baserel->allvisfrac) +
							 pgstrom_dpu_seq_page_cost * baserel->allvisfrac);
	}
	else
	{
		elog(ERROR, "Bug? unsupported xpu_task_flags: %08x", xpu_task_flags);
	}

	/*
	 * NOTE: ArrowGetForeignRelSize() already discount baserel->pages according
	 * to the referenced columns, to adjust total amount of disk i/o.
	 * So, we have nothing special to do here.
	 */
	disk_cost = avg_seq_page_cost * baserel->pages;
	if (parallel_path)
		disk_cost /= parallel_divisor;

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
		cost_qual_eval_node(&qcost, (Node *)host_quals, root);
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
	final_cost += (baserel->reltarget->cost.startup +
				   baserel->reltarget->cost.per_tuple * scan_nrows);

	/* Setup the result */
	pp_info = palloc0(sizeof(pgstromPlanInfo));
	pp_info->xpu_task_flags = xpu_task_flags;
	pp_info->ds_entry = ds_entry;
	pp_info->scan_relid = baserel->relid;
	pp_info->host_quals = extract_actual_clauses(host_quals, false);
	pp_info->scan_quals = extract_actual_clauses(dev_quals, false);
	pp_info->scan_tuples = baserel->tuples;
	pp_info->scan_nrows = clamp_row_est(scan_nrows);
	pp_info->parallel_nworkers = parallel_nworkers;
	pp_info->parallel_divisor = parallel_divisor;
	pp_info->startup_cost = startup_cost;
	pp_info->run_cost = run_cost;
	pp_info->final_cost = final_cost;
	pp_info->final_nrows = baserel->rows;
	if (indexOpt)
	{
		pp_info->brin_index_oid = indexOpt->indexoid;
		pp_info->brin_index_conds = indexConds;
		pp_info->brin_index_quals = indexQuals;
	}
	outer_refs = pickup_outer_referenced(root, baserel, outer_refs);
	pull_varattnos((Node *)pp_info->host_quals,
				   baserel->relid, &outer_refs);
	pull_varattnos((Node *)pp_info->scan_quals,
				   baserel->relid, &outer_refs);
	pp_info->outer_refs = outer_refs;
	pp_info->sibling_param_id = -1;
	return pp_info;
}


static pgstromOuterPathLeafInfo *
buildSimpleScanPlanInfo(PlannerInfo *root,
						RelOptInfo *baserel,
						uint32_t xpu_task_flags,
						bool parallel_path)
{
	pgstromOuterPathLeafInfo *op_leaf;
	pgstromPlanInfo *pp_info;
	ParamPathInfo *param_info;
	List	   *dev_quals = NIL;
	List	   *dev_costs = NIL;
	List	   *host_quals = NIL;
	ListCell   *lc;
	Cardinality	scan_nrows = baserel->rows;

	Assert((xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU ||
		   (xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU);
	/* fetch device/host qualifiers */
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo *rinfo = lfirst(lc);
		int		devcost;

		if (pgstrom_xpu_expression(rinfo->clause,
								   xpu_task_flags,
								   baserel->relid,
								   NIL,
								   &devcost))
		{
			dev_quals = lappend(dev_quals, rinfo);
			dev_costs = lappend_int(dev_costs, devcost);
		}
		else
		{
			host_quals = lappend(host_quals, rinfo);
		}
	}
	/* also checks parametalized qualifiers */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		foreach (lc, param_info->ppi_clauses)
		{
			RestrictInfo *rinfo = lfirst(lc);
			int		devcost;

			if (pgstrom_xpu_expression(rinfo->clause,
									   xpu_task_flags,
									   baserel->relid,
									   NIL,
									   &devcost))
			{
				dev_quals = lappend(dev_quals, rinfo);
				dev_costs = lappend_int(dev_costs, devcost);
			}
			else
			{
				host_quals = lappend(host_quals, rinfo);
			}
		}
		scan_nrows = param_info->ppi_rows;
	}
	sort_device_qualifiers(dev_quals, dev_costs);

	pp_info = __buildSimpleScanPlanInfo(root,
										baserel,
										xpu_task_flags,
										parallel_path,
										dev_quals,
										host_quals,
										scan_nrows);
	if (!pp_info)
		return NULL;
	/* setup pgstromOuterPathLeafInfo */
	op_leaf = palloc0(sizeof(pgstromOuterPathLeafInfo));
	op_leaf->pp_info = pp_info;
	op_leaf->leaf_rel = baserel;
	op_leaf->leaf_param = param_info;
	op_leaf->leaf_nrows = pp_info->scan_nrows;
	op_leaf->leaf_cost = (pp_info->startup_cost +
						  pp_info->run_cost +
						  pp_info->final_cost);
	op_leaf->inner_paths_list = NIL;

	return op_leaf;
}

/*
 * try_add_simple_scan_path
 */
static void
try_add_simple_scan_path(PlannerInfo *root,
						 RelOptInfo *baserel,
						 RangeTblEntry *rte,
						 uint32_t xpu_task_flags,
						 bool be_parallel,
						 bool allow_host_quals,
						 bool allow_no_device_quals,
						 const CustomPathMethods *xpuscan_path_methods)
{
	pgstromOuterPathLeafInfo *op_leaf = NULL;

	if (rte->relkind == RELKIND_RELATION ||
		rte->relkind == RELKIND_MATVIEW)
	{
		if (rte->rtekind == RTE_RELATION &&
			get_relation_am(rte->relid, true) == HEAP_TABLE_AM_OID)
		{
			op_leaf = buildSimpleScanPlanInfo(root,
											  baserel,
											  xpu_task_flags,
											  be_parallel);
		}
	}
	else if (rte->relkind == RELKIND_FOREIGN_TABLE)
	{
		if (baseRelIsArrowFdw(baserel))
		{
			op_leaf = buildSimpleScanPlanInfo(root,
											  baserel,
											  xpu_task_flags,
											  be_parallel);
		}
	}

	if (op_leaf)
	{
		pgstromPlanInfo *pp_info = op_leaf->pp_info;

		if (pp_info->scan_quals != NIL)
		{
			CustomPath *cpath = makeNode(CustomPath);

			cpath->path.pathtype    = T_CustomScan;
			cpath->path.parent      = baserel;
			cpath->path.pathtarget  = baserel->reltarget;
			cpath->path.param_info  = op_leaf->leaf_param;
			cpath->path.parallel_aware = (pp_info->parallel_nworkers > 0);
			cpath->path.parallel_safe = baserel->consider_parallel;
			cpath->path.parallel_workers = pp_info->parallel_nworkers;
			cpath->path.rows        = pp_info->scan_nrows;
			Assert(pp_info->inner_cost == 0.0);
			cpath->path.startup_cost = pp_info->startup_cost;
			cpath->path.total_cost  = (pp_info->startup_cost +
									   pp_info->run_cost +
									   pp_info->final_cost);
			cpath->path.pathkeys    = NIL;	/* unsorted results */
			cpath->flags            = CUSTOMPATH_SUPPORT_PROJECTION;
			cpath->custom_paths     = NIL;
			cpath->custom_private   = list_make1(pp_info);
			cpath->methods			= xpuscan_path_methods;
			/* try attach GPU-Sorted version */
			try_add_sorted_gpujoin_path(root, baserel, cpath, be_parallel);
			if (be_parallel == 0)
				add_path(baserel, &cpath->path);
			else
				add_partial_path(baserel, &cpath->path);
		}
		/*
		 * unable pullup the scan path with host-quals
		 */
		if (pp_info->host_quals == NIL)
		{
			pgstrom_remember_op_normal(root,
									   baserel,
									   op_leaf,
									   be_parallel);
		}
	}
}

/*
 * try_add_partitioned_scan_path
 */
static List *
__try_add_partitioned_scan_path(PlannerInfo *root,
								RelOptInfo *baserel,
								uint32_t xpu_task_flags,
								bool be_parallel)
{
	List   *results = NIL;

	for (int k=0; k < baserel->nparts; k++)
	{
		if (bms_is_member(k, baserel->live_parts))
		{
			RelOptInfo *leaf_rel = baserel->part_rels[k];
			RangeTblEntry *rte = root->simple_rte_array[leaf_rel->relid];

			if (!rte->inh)
			{
				pgstromOuterPathLeafInfo *op_leaf;

				op_leaf = buildSimpleScanPlanInfo(root,
												  leaf_rel,
												  xpu_task_flags,
												  be_parallel);
				if (!op_leaf)
					return NIL;
				/* unable to register scan path with host quals */
				if (op_leaf->pp_info->host_quals != NIL)
					return NIL;
				results = lappend(results, op_leaf);
			}
			else if (rte->relkind == RELKIND_PARTITIONED_TABLE)
			{
				List   *temp;

				temp = __try_add_partitioned_scan_path(root,
													   leaf_rel,
													   xpu_task_flags,
													   be_parallel);
				if (temp == NIL)
					return NIL;
				results = list_concat(results, temp);
			}
		}
	}
	return results;
}

static void
try_add_partitioned_scan_path(PlannerInfo *root,
							  RelOptInfo *baserel,
							  uint32_t xpu_task_flags,
							  bool be_parallel)
{
	List   *results = __try_add_partitioned_scan_path(root,
													  baserel,
													  xpu_task_flags,
													  be_parallel);
	if (results != NIL)
		pgstrom_remember_op_leafs(root, baserel, results, be_parallel);
}

/*
 * XpuScanAddScanPath
 */
static void
__xpuScanAddScanPathCommon(PlannerInfo *root,
						   RelOptInfo *baserel,
						   Index rtindex,
						   RangeTblEntry *rte,
						   uint32_t xpu_task_flags,
						   const CustomPathMethods *xpuscan_path_methods)
{
	/* We already proved the relation empty, so nothing more to do */
	if (is_dummy_rel(baserel))
		return;
	/* Creation of GpuScan path */
	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		if (!rte->inh)
		{
			try_add_simple_scan_path(root,
									 baserel,
									 rte,
									 xpu_task_flags,
									 (try_parallel > 0),
									 true,	/* allow host quals */
									 false,	/* disallow no device quals*/
									 xpuscan_path_methods);
		}
		else if (rte->relkind == RELKIND_PARTITIONED_TABLE)
		{
			try_add_partitioned_scan_path(root,
										  baserel,
										  xpu_task_flags,
										  (try_parallel > 0));
		}
		if (!baserel->consider_parallel)
			break;
	}
}

static void
XpuScanAddScanPath(PlannerInfo *root,
				   RelOptInfo *baserel,
				   Index rtindex,
				   RangeTblEntry *rte)
{
	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);

	if (pgstrom_enabled())
	{
		if (enable_gpuscan && gpuserv_ready_accept())
			__xpuScanAddScanPathCommon(root, baserel, rtindex, rte,
									   TASK_KIND__GPUSCAN,
									   &gpuscan_path_methods);
		if (enable_dpuscan)
			__xpuScanAddScanPathCommon(root, baserel, rtindex, rte,
									   TASK_KIND__DPUSCAN,
									   &dpuscan_path_methods);
	}
}

/*
 * try_fetch_xpuscan_planinfo
 */
pgstromPlanInfo *
try_fetch_xpuscan_planinfo(const Path *__path)
{
	const CustomPath   *cpath = (const CustomPath *)__path;

	if (IsA(cpath, CustomPath) &&
		(cpath->methods == &gpuscan_path_methods ||
		 cpath->methods == &dpuscan_path_methods))
		return (pgstromPlanInfo *)linitial(cpath->custom_private);
	return NULL;
}

/*
 * gpuscan_build_projection - make custom_scan_tlist
 */
static List *
__gpuscan_build_projection_expr(List *tlist_dev,
								Node *node,
								uint32_t xpu_task_flags,
								Index scan_relid,
								bool resjunk)
{
	ListCell   *lc;

	if (!node || tlist_member((Expr *)node, tlist_dev))
		return tlist_dev;

	if (IsA(node, Var) ||
		pgstrom_xpu_expression((Expr *)node,
							   xpu_task_flags,
							   scan_relid,
							   NIL,
							   NULL))
	{
		AttrNumber	resno = list_length(tlist_dev) + 1;

		tlist_dev = lappend(tlist_dev,
							makeTargetEntry((Expr *)node,
											resno,
											NULL,
											resjunk));
	}
	else
	{
		List	*vars_list = pull_vars_of_level(node, 0);

		foreach (lc, vars_list)
			tlist_dev = __gpuscan_build_projection_expr(tlist_dev,
														lfirst(lc),
														xpu_task_flags,
														scan_relid,
														resjunk);
	}
	return tlist_dev;
}

static List *
gpuscan_build_projection(RelOptInfo *baserel,
						 pgstromPlanInfo *pp_info,
						 List *tlist)
{
	List	   *tlist_dev = NIL;
	List	   *vars_list;
	ListCell   *lc;

	if (tlist != NIL)
	{
		foreach (lc, tlist)
		{
			TargetEntry *tle = lfirst(lc);

			if (IsA(tle->expr, Const) || IsA(tle->expr, Param))
				continue;
			tlist_dev = __gpuscan_build_projection_expr(tlist_dev,
														(Node *)tle->expr,
														pp_info->xpu_task_flags,
														baserel->relid,
														false);
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
			tlist_dev = __gpuscan_build_projection_expr(tlist_dev,
														node,
														pp_info->xpu_task_flags,
														baserel->relid,
														false);
		}
	}
	vars_list = pull_vars_of_level((Node *)pp_info->host_quals, 0);
	foreach (lc, vars_list)
		tlist_dev = __gpuscan_build_projection_expr(tlist_dev,
													(Node *)lfirst(lc),
													pp_info->xpu_task_flags,
													baserel->relid,
													false);

	vars_list = pull_vars_of_level((Node *)pp_info->scan_quals, 0);
	foreach (lc, vars_list)
		tlist_dev = __gpuscan_build_projection_expr(tlist_dev,
													(Node *)lfirst(lc),
													pp_info->xpu_task_flags,
													baserel->relid,
													true);
	return tlist_dev;
}

/*
 * __build_explain_tlist_junks
 */
static void
__build_explain_tlist_junks(PlannerInfo *root,
							RelOptInfo *baserel,
							codegen_context *context)
{
	RangeTblEntry  *rte = root->simple_rte_array[baserel->relid];

	Assert(IS_SIMPLE_REL(baserel) && rte->rtekind == RTE_RELATION);
	for (int j=baserel->min_attr; j <= baserel->max_attr; j++)
	{
		Form_pg_attribute attr;
		HeapTuple	htup;
		Var		   *var;
		ListCell   *lc;
		TargetEntry *tle;

		if (bms_is_empty(baserel->attr_needed[j-baserel->min_attr]))
			continue;
		htup = SearchSysCache2(ATTNUM,
							   ObjectIdGetDatum(rte->relid),
							   Int16GetDatum(j));
		if (!HeapTupleIsValid(htup))
			elog(ERROR, "cache lookup failed for attribute %d of relation %u",
				 j, rte->relid);
		attr = (Form_pg_attribute) GETSTRUCT(htup);
		var = makeVar(baserel->relid,
					  attr->attnum,
					  attr->atttypid,
					  attr->atttypmod,
					  attr->attcollation,
					  0);
		foreach (lc, context->tlist_dev)
		{
			tle = lfirst(lc);
			if (equal(tle->expr, var))
				break;
		}
		if (lc)
		{
			/* found */
			pfree(var);
		}
		else
		{
			/* not found, append a junk */
			tle = makeTargetEntry((Expr *)var,
								  list_length(context->tlist_dev)+1,
								  pstrdup(NameStr(attr->attname)),
								  true);
			context->tlist_dev = lappend(context->tlist_dev, tle);
		}
		ReleaseSysCache(htup);
	}
}

/*
 * assign_custom_cscan_tlist
 */
List *
assign_custom_cscan_tlist(List *tlist_dev, pgstromPlanInfo *pp_info)
{
	ListCell   *lc1, *lc2;

	/* clear kv_fallback */
	foreach (lc1, pp_info->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc1);

		kvdef->kv_fallback = -1;
	}

	foreach (lc1, tlist_dev)
	{
		TargetEntry *tle = lfirst(lc1);

		foreach (lc2, pp_info->kvars_deflist)
		{
			codegen_kvar_defitem *kvdef = lfirst(lc2);

			if (kvdef->kv_depth >= 0 &&
				kvdef->kv_depth <= pp_info->num_rels &&
				kvdef->kv_resno != InvalidAttrNumber &&
				equal(tle->expr, kvdef->kv_expr))
			{
				kvdef->kv_fallback = tle->resno;
				tle->resorigtbl = (Oid)kvdef->kv_depth;
				tle->resorigcol = kvdef->kv_resno;
				break;
			}
		}
		if (!lc2)
		{
			tle->resorigtbl = (Oid)UINT_MAX;
			tle->resorigcol = -1;
		}
	}
	return tlist_dev;
}

/*
 * planxpuscanpathcommon
 */
static CustomScan *
PlanXpuScanPathCommon(PlannerInfo *root,
					  RelOptInfo  *baserel,
					  CustomPath  *best_path,
					  List        *tlist,
					  List        *clauses,
					  pgstromPlanInfo *pp_info,
					  const CustomScanMethods *xpuscan_plan_methods)
{
	codegen_context *context;
	CustomScan *cscan;
	List	   *proj_hash = pp_info->projection_hashkeys;

	context = create_codegen_context(root, best_path, pp_info);
	/* code generation for WHERE-clause */
	pp_info->kexp_scan_quals = codegen_build_scan_quals(context, pp_info->scan_quals);
	/* code generation for the Projection */
	context->tlist_dev = gpuscan_build_projection(baserel, pp_info, tlist);
	pp_info->kexp_projection = codegen_build_projection(context,
														proj_hash);
	/* code generation for GPU-Sort */
	pp_info->kexp_gpusort_keydesc = codegen_build_gpusort_keydesc(context, pp_info);
	/* VarLoads for each depth */
	codegen_build_packed_kvars_load(context, pp_info);
	/* VarMoves for each depth (only GPUs) */
	codegen_build_packed_kvars_move(context, pp_info);
	/* xpu_task_flags should not be cleared in codege.c */
	Assert((context->xpu_task_flags &
			pp_info->xpu_task_flags) == pp_info->xpu_task_flags);
	pp_info->kvars_deflist = context->kvars_deflist;
	pp_info->xpu_task_flags = context->xpu_task_flags;
	pp_info->extra_bufsz = context->extra_bufsz;
	pp_info->used_params = context->used_params;
	pp_info->cuda_stack_size = estimate_cuda_stack_size(context);
	__build_explain_tlist_junks(root, baserel, context);

	/*
	 * Build CustomScan(GpuScan) node
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = pp_info->host_quals;
	cscan->scan.scanrelid = baserel->relid;
	cscan->flags = best_path->flags;
	cscan->methods = xpuscan_plan_methods;
	cscan->custom_plans = NIL;
	cscan->custom_scan_tlist = assign_custom_cscan_tlist(context->tlist_dev,
														 pp_info);
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
 * PlanDpuScanPath
 */
static Plan *
PlanDpuScanPath(PlannerInfo *root,
                RelOptInfo *baserel,
                CustomPath *best_path,
                List *tlist,
                List *clauses,
                List *custom_children)
{
	pgstromPlanInfo *pp_info = linitial(best_path->custom_private);
	CustomScan *cscan;

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
								  &dpuscan_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);

	return &cscan->scan.plan;
}

/*
 * CreateGpuScanState
 */
static Node *
CreateGpuScanState(CustomScan *cscan)
{
	Assert(cscan->methods == &gpuscan_plan_methods);
	return pgstromCreateTaskState(cscan, &gpuscan_exec_methods);
}

/*
 * CreateDpuScanState
 */
static Node *
CreateDpuScanState(CustomScan *cscan)
{
	Assert(cscan->methods == &dpuscan_plan_methods);
	return pgstromCreateTaskState(cscan, &dpuscan_exec_methods);
}

/*
 * __pgstrom_init_xpuscan_common
 */
static void
__pgstrom_init_xpuscan_common(void)
{
	static bool	xpuscan_common_initialized = false;

	if (!xpuscan_common_initialized)
	{
		/* hook registration */
		set_rel_pathlist_next = set_rel_pathlist_hook;
		set_rel_pathlist_hook = XpuScanAddScanPath;

		xpuscan_common_initialized = true;
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
    gpuscan_exec_methods.ExecCustomScan		= pgstromExecTaskState;
    gpuscan_exec_methods.EndCustomScan		= pgstromExecEndTaskState;
    gpuscan_exec_methods.ReScanCustomScan	= pgstromExecResetTaskState;
    gpuscan_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
    gpuscan_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
    gpuscan_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
    gpuscan_exec_methods.ShutdownCustomScan	= pgstromSharedStateShutdownDSM;
    gpuscan_exec_methods.ExplainCustomScan	= pgstromExplainTaskState;
	/* common portion */
	__pgstrom_init_xpuscan_common();
}

/*
 * pgstrom_init_dpu_scan
 */
void
pgstrom_init_dpu_scan(void)
{
	/* pg_strom.enable_dpuscan */
	DefineCustomBoolVariable("pg_strom.enable_dpuscan",
							 "Enables the use of DPU accelerated full-scan",
							 NULL,
							 &enable_dpuscan,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	memset(&dpuscan_path_methods, 0, sizeof(dpuscan_path_methods));
	dpuscan_path_methods.CustomName			= "DpuScan";
	dpuscan_path_methods.PlanCustomPath		= PlanDpuScanPath;

	/* setup plan methods */
	memset(&dpuscan_plan_methods, 0, sizeof(dpuscan_plan_methods));
	dpuscan_plan_methods.CustomName			= "DpuScan";
	dpuscan_plan_methods.CreateCustomScanState = CreateDpuScanState;
	RegisterCustomScanMethods(&dpuscan_plan_methods);

	/* setup exec methods */
	memset(&dpuscan_exec_methods, 0, sizeof(dpuscan_exec_methods));
	dpuscan_exec_methods.CustomName			= "DpuScan";
	dpuscan_exec_methods.BeginCustomScan	= pgstromExecInitTaskState;
	dpuscan_exec_methods.ExecCustomScan		= pgstromExecTaskState;
	dpuscan_exec_methods.EndCustomScan		= pgstromExecEndTaskState;
	dpuscan_exec_methods.ReScanCustomScan	= pgstromExecResetTaskState;
	dpuscan_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
	dpuscan_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
	dpuscan_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
	dpuscan_exec_methods.ShutdownCustomScan = pgstromSharedStateShutdownDSM;
	dpuscan_exec_methods.ExplainCustomScan	= pgstromExplainTaskState;
	/* common portion */
	__pgstrom_init_xpuscan_common();
}
