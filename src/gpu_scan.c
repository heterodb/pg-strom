/*
 * gpu_scan.c
 *
 * Sequential scan accelerated with GPU processors
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
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
static void
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
 * __gpuScanBuildPlanInfo
 */
static pgstromPlanInfo *
__gpuScanBuildPlanInfo(PlannerInfo *root,
					   RelOptInfo *baserel,
					   RangeTblEntry *rte,
					   ParamPathInfo *param_info,
					   List *scan_rels,
					   List *dev_quals,
					   List *host_quals,
					   Bitmapset *outer_refs,
					   int parallel_nworkers)
{
	double		parallel_divisor = 1.0;
	Cost		startup_cost = pgstrom_gpu_setup_cost;
	Cost		run_cost = 0.0;
	Cost		final_cost = 0.0;
	double		total_ntuples = 0.0;
	double		total_npages = 0.0;
	double		curr_ntuples;
	List	   *scan_rels_list = NIL;
	ListCell   *lc;
	pgstromPlanInfo *pp_info;


	if (parallel_nworkers > 0)
	{
		double	leader_contribution;

		parallel_divisor = (double)parallel_nworkers;
		if (parallel_leader_participation)
		{
			leader_contribution = 1.0 - (0.3 * (double)parallel_nworkers);
			if (leader_contribution > 0.0)
				parallel_divisor += leader_contribution;
		}
	}

	/*
	 * count up for each relation's scan cost
	 */
	foreach (lc, scan_rels)
	{
		pgstromPlanScanInfo *pp_scan = palloc0(sizeof(pgstromPlanScanInfo));
		RelOptInfo *__rel = lfirst(lc);
		double		__ntuples = __rel->tuples;
		double		__npages = __rel->pages;
		double		spc_rand_page_cost;
		double		spc_seq_page_cost;
		double		avg_page_cost;
		double		disk_cost;
		IndexOptInfo *indexOpt;
		List	   *indexConds;
		List	   *indexQuals;
		int64_t		indexNBlocks;

		get_tablespace_page_costs(__rel->reltablespace,
								  &spc_rand_page_cost,
								  &spc_seq_page_cost);
		if (baseRelIsArrowFdw(__rel))
		{
			/*
			 * NOTE: ArrowGetForeignRelSize() already discount baserel->pages
			 * according to the referenced columns, to adjust total amount of
			 * disk i/o.
			 * So, we have nothing special to do here.
			 */
			avg_page_cost = spc_seq_page_cost * (1.0 - __rel->allvisfrac) +
				pgstrom_gpu_direct_seq_page_cost * __rel->allvisfrac;
		}
		else if (GetOptimalGpuForBaseRel(root, __rel) != 0UL)
		{
			/* assume GPU-Direct SQL is available */
			avg_page_cost = spc_seq_page_cost * (1.0 - __rel->allvisfrac) +
				pgstrom_gpu_direct_seq_page_cost * __rel->allvisfrac;
		}
		else
		{
			/* elsewhere, use PostgreSQL's storage layer */
			avg_page_cost = spc_seq_page_cost;
		}
		disk_cost = avg_page_cost * (double)__rel->pages;

		/*
		 * Is BRIN-index available?
		 */
		indexOpt = pgstromTryFindBrinIndex(root, __rel,
										   &indexConds,
										   &indexQuals,
										   &indexNBlocks);
		if (indexOpt)
		{
			Cost	brin_ratio = ((double)indexNBlocks / (double)__rel->pages);
			Cost	brin_disk_cost = brin_ratio * disk_cost;

			brin_disk_cost += cost_brin_bitmap_build(root,
													 __rel,
													 indexOpt,
													 indexQuals);
			if (disk_cost > brin_disk_cost)
			{
				disk_cost = brin_disk_cost;
				__ntuples *= brin_ratio;
				__npages = indexNBlocks;
			}
			else
			{
				/* disable BRIN-index if no benefit */
				indexOpt = NULL;
			}
		}
		total_ntuples += __ntuples;
		total_npages += __npages;
		/* discount disk cost if parallel scan */
		if (parallel_nworkers > 0)
			disk_cost /= parallel_divisor;
		run_cost += disk_cost;
		/* RTI to be scanned */
		pp_scan->scan_relid = __rel->relid;
		pp_scan->plan_ntuples_raw = __ntuples;
		pp_scan->plan_ntuples_in = __rel->rows;
		if (indexOpt)
		{
			pp_scan->brin_oid = indexOpt->indexoid;
			pp_scan->brin_conds = indexConds;
			pp_scan->brin_quals = indexQuals;
		}
		scan_rels_list = lappend(scan_rels_list, pp_scan);
	}
	curr_ntuples = total_ntuples;

	/*
	 * Cost for GPU qualifiers
	 */
	if (dev_quals != NIL)
	{
		QualCost	qcost;
		double		selectivity;

		cost_qual_eval_node(&qcost, (Node *)dev_quals, root);
		startup_cost += qcost.startup;
		run_cost += (qcost.per_tuple *
					 pgstrom_gpu_operator_ratio() *
					 curr_ntuples) / parallel_divisor;

		if (host_quals == NIL)
			curr_ntuples = baserel->rows;
		else
		{
			selectivity = clauselist_selectivity(root,
												 dev_quals,
												 baserel->relid,
												 JOIN_INNER,
												 NULL);
			curr_ntuples *= selectivity;
		}
		/* cost for DMA transfer (GPU->Host) */
		final_cost += pgstrom_gpu_tuple_cost * curr_ntuples;
	}

	/*
	 * Cost for host qualifiers
	 */
	if (host_quals != NIL)
	{
		QualCost	qcost;

		cost_qual_eval_node(&qcost, (Node *)host_quals, root);
		startup_cost += qcost.startup;
		final_cost += qcost.per_tuple * curr_ntuples / parallel_divisor;
	}

	/*
	 * Cost for host projection
	 */
	final_cost += (baserel->reltarget->cost.startup +
				   baserel->reltarget->cost.per_tuple * baserel->rows);
	/*
	 * Setup the result PlanInfo
	 */
	pp_info = palloc0(sizeof(pgstromPlanInfo));
	pp_info->xpu_task_flags = TASK_KIND__GPUSCAN;
	if (rte->relkind == RELKIND_PARTITIONED_TABLE)
		pp_info->xpu_task_flags |= DEVTASK__IS_PARTITION_WISE;
	pp_info->host_quals = extract_actual_clauses(host_quals, false);
	pp_info->base_relid = baserel->relid;
	pp_info->scan_quals = extract_actual_clauses(dev_quals, false);
	pp_info->scan_npages = total_npages;
	pp_info->scan_tuples = total_ntuples;
	pp_info->scan_nrows = (param_info ? param_info->ppi_rows : baserel->rows);
	pp_info->parallel_nworkers = parallel_nworkers;
    pp_info->parallel_divisor = parallel_divisor;
	pp_info->startup_cost = startup_cost;
	pp_info->run_cost = run_cost;
	pp_info->final_cost = final_cost;
	pp_info->final_nrows = pp_info->scan_nrows;
	pp_info->outer_refs = outer_refs;
	pp_info->scan_rels_list = scan_rels_list;
	pp_info->sibling_param_id = -1;		//to be deprecated
	return pp_info;
}

/*
 * __gpuScanAddCustomPathOne
 */
static void
__gpuScanAddCustomPathOne(PlannerInfo *root,
						  RelOptInfo *baserel,
						  RangeTblEntry *rte,
						  ParamPathInfo *param_info,
						  List *scan_rels,
						  List *dev_quals,
						  List *host_quals,
						  Bitmapset *outer_refs,
						  int *p_parallel_nworkers)
{
	int			parallel_nworkers = *p_parallel_nworkers;
	CustomPath *cpath;
	pgstromPlanInfo *pp_info;

	pp_info = __gpuScanBuildPlanInfo(root,
									 baserel,
									 rte,
									 param_info,
									 scan_rels,
									 dev_quals,
									 host_quals,
									 outer_refs,
									 parallel_nworkers);
	assert(pp_info->parallel_nworkers == parallel_nworkers);

	/*
	 * build a CustomPath(GpuScan)
	 */
	cpath = makeNode(CustomPath);
	cpath->path.pathtype    = T_CustomScan;
	cpath->path.parent      = baserel;
	cpath->path.pathtarget  = baserel->reltarget;
	cpath->path.param_info  = param_info;
	cpath->path.parallel_aware = (parallel_nworkers > 0);
	cpath->path.parallel_safe = baserel->consider_parallel;
	cpath->path.parallel_workers = parallel_nworkers;
	cpath->path.rows        = pp_info->final_nrows;
	Assert(pp_info->inner_cost == 0.0);
	cpath->path.startup_cost = pp_info->startup_cost;
	cpath->path.total_cost  = (pp_info->startup_cost +
							   pp_info->run_cost +
							   pp_info->final_cost);
	cpath->path.pathkeys    = NIL;  /* unsorted results */
	cpath->flags            = CUSTOMPATH_SUPPORT_PROJECTION;
	cpath->custom_paths     = NIL;
	cpath->custom_private   = list_make1(pp_info);
	cpath->methods          = &gpuscan_path_methods;
	/* try attach GPU-Sorted version */
	try_add_sorted_gpujoin_path(root, baserel, cpath, (parallel_nworkers > 0));

	/*
	 * GpuScan is executable on the device if it contains device-qualifiers
	 * because it works to reduce number of rows, and makes sense.
	 * On the other hands, it can be the source of upper GPU-Join or -PreAgg,
	 * however, host-only qualifier prevents to pull up.
	 *
	 * Note that pgstrom_remember_custom_path() must be prior to add_path()
	 * because optimizer may release CustomPath if it is lesser.
	 */
	if (pp_info->host_quals == NIL)
		pgstrom_remember_custom_path(root, baserel, cpath, parallel_nworkers > 0);
	if (pp_info->scan_quals != NIL)
	{
		if (parallel_nworkers == 0)
			add_path(baserel, &cpath->path);
		else
			add_partial_path(baserel, &cpath->path);
	}
	/* suggest parallel_nworkers next time */
	parallel_nworkers = compute_parallel_worker(baserel,
												pp_info->scan_npages,
												-1.0,
												max_parallel_workers_per_gather);
	*p_parallel_nworkers = parallel_nworkers;
}

/*
 * __gpuScanAddCustomPath - main portion of GpuScanPath
 */
static void
__gpuScanAddCustomPath(PlannerInfo *root,
					   RelOptInfo *baserel,
					   Index rtindex,
					   RangeTblEntry *rte,
					   List *scan_rels)
{
	ParamPathInfo *param_info;
	List	   *dev_quals = NIL;
	List	   *dev_costs = NIL;
	List	   *host_quals = NIL;
	ListCell   *lc;
	Bitmapset  *outer_refs;
	int			parallel_nworkers = 0;

	outer_refs = pickup_outer_referenced(root, baserel);
	/* Fetch device/host qualifiers */
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo *rinfo = lfirst(lc);
		int		devcost;

		if (pgstrom_xpu_expression(rinfo->clause,
								   DEVKIND__NVIDIA_GPU,
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
		pull_varattnos((Node *)rinfo->clause, baserel->relid, &outer_refs);
	}
	/* Also checks parametalized qualifiers */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		foreach (lc, param_info->ppi_clauses)
		{
			RestrictInfo *rinfo = lfirst(lc);
			int		devcost;

			if (pgstrom_xpu_expression(rinfo->clause,
									   DEVKIND__NVIDIA_GPU,
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
			pull_varattnos((Node *)rinfo->clause, baserel->relid, &outer_refs);
		}
	}
	sort_device_qualifiers(dev_quals, dev_costs);

	/*
	 * Single GpuScanPath
	 */
	__gpuScanAddCustomPathOne(root,
							  baserel,
							  rte,
							  param_info,
							  scan_rels,
							  dev_quals,
							  host_quals,
							  outer_refs,
							  &parallel_nworkers);
	/*
	 * Parallel GpuScanPath
	 */
	if (baserel->consider_parallel && parallel_nworkers > 0)
	{
		__gpuScanAddCustomPathOne(root,
								  baserel,
								  rte,
								  param_info,
								  scan_rels,
								  dev_quals,
								  host_quals,
								  outer_refs,
								  &parallel_nworkers);
	}
}

/*
 * check_compatible_relations
 */
static bool
check_compatible_relations(PlannerInfo *root,
						   RelOptInfo *parent_rel,
						   RelOptInfo *child_rel)
{
	AppendRelInfo *appinfo = root->append_rel_array[child_rel->relid];
	ListCell   *lc;
	AttrNumber	attnum = 1;

	if (!appinfo ||
		appinfo->parent_relid != parent_rel->relid ||
		appinfo->child_relid  != child_rel->relid)
	{
		return false;
	}
	foreach (lc, appinfo->translated_vars)
	{
		Var	   *var = lfirst(lc);

		if (var->varattno != attnum)
		{
			RangeTblEntry *p_rte = planner_rt_fetch(parent_rel->relid, root);
			RangeTblEntry *c_rte = planner_rt_fetch(child_rel->relid, root);

			elog(DEBUG2, "relation %s does not have compatible table layout at attnum=%d to the partition parent %s, so not supported right now",
				 c_rte->eref->aliasname,
				 attnum,
				 p_rte->eref->aliasname);
			return false;
		}
		attnum++;
	}
	return true;
}

/*
 * extract_partitioned_scan_rels
 */
static bool
extract_partitioned_scan_rels(PlannerInfo *root,
							  RelOptInfo *baserel,
							  List **p_results)
{
	for (int k=0; k < baserel->nparts; k++)
	{
		if (bms_is_member(k, baserel->live_parts))
		{
			RelOptInfo	   *leaf_rel = baserel->part_rels[k];
			RangeTblEntry  *rte = root->simple_rte_array[leaf_rel->relid];

			switch (rte->relkind)
			{
				case RELKIND_RELATION:
				case RELKIND_MATVIEW:
					if (!rte->inh &&
						rte->rtekind == RTE_RELATION &&
						get_relation_am(rte->relid, true) == HEAP_TABLE_AM_OID &&
						check_compatible_relations(root, baserel, leaf_rel))
					{
						*p_results = lappend(*p_results, leaf_rel);
						break;
					}
					return false;
				case RELKIND_FOREIGN_TABLE:
					if (!rte->inh &&
						baseRelIsArrowFdw(leaf_rel) &&
						check_compatible_relations(root, baserel, leaf_rel))
					{
						*p_results = lappend(*p_results, leaf_rel);
						break;
					}
					return false;
				case RELKIND_PARTITIONED_TABLE:
					if (check_compatible_relations(root, baserel, leaf_rel) &&
						extract_partitioned_scan_rels(root, leaf_rel, p_results))
						break;
					return false;
				default:
					return false;
			}
		}
	}
	return true;
}

static void
GpuScanAddScanPath(PlannerInfo *root,
				   RelOptInfo *baserel,
				   Index rtindex,
				   RangeTblEntry *rte)
{
	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);

	if (pgstrom_enabled() &&
		gpuserv_ready_accept() &&
		enable_gpuscan &&
		!is_dummy_rel(baserel))
	{
		List   *scan_rels = NIL;

		switch (rte->relkind)
		{
			case RELKIND_RELATION:
			case RELKIND_MATVIEW:
				if (!rte->inh &&
					rte->rtekind == RTE_RELATION &&
					get_relation_am(rte->relid, true) == HEAP_TABLE_AM_OID)
					scan_rels = list_make1(baserel);
				break;
			case RELKIND_FOREIGN_TABLE:
				if (!rte->inh &&
					baseRelIsArrowFdw(baserel))
					scan_rels = list_make1(baserel);
				break;
			case RELKIND_PARTITIONED_TABLE:
				if (!extract_partitioned_scan_rels(root,
												   baserel,
												   &scan_rels))
					return;
				break;
			default:
				/* not supported */
				break;
		}
		/*
		 * Creation of GpuScan paths
		 */
		if (scan_rels != NIL)
		{
			__gpuScanAddCustomPath(root,
								   baserel,
								   rtindex,
								   rte,
								   scan_rels);
		}
	}
}

/*
 * try_fetch_xpuscan_planinfo
 */
pgstromPlanInfo *
try_fetch_xpuscan_planinfo(const Path *__path)
{
	const CustomPath   *cpath = (const CustomPath *)__path;

	if (IsA(cpath, CustomPath) && cpath->methods == &gpuscan_path_methods)
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
				kvdef->kv_depth <= pp_info->num_inner_rels &&
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
 * PlanGpuScanPathCommon
 */
static CustomScan *
PlanGpuScanPathCommon(PlannerInfo *root,
					  RelOptInfo  *baserel,
					  CustomPath  *best_path,
					  List        *tlist,
					  List        *clauses,
					  pgstromPlanInfo *pp_info)
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
	cscan->methods = &gpuscan_plan_methods;
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
	cscan = PlanGpuScanPathCommon(root,
								  baserel,
								  best_path,
								  tlist,
								  clauses,
								  pp_info);
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
	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = GpuScanAddScanPath;
}
