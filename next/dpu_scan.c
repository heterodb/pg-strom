/*
 * dpu_scan.c
 *
 * Sequential scan accelerated with DPU processors
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
static CustomPathMethods	dpuscan_path_methods;
static CustomScanMethods	dpuscan_plan_methods;
static CustomExecMethods	dpuscan_exec_methods;
static bool					enable_dpuscan;		/* GUC */

/*
 * DpuScanSharedState
 */
typedef struct
{
	/* for arrow_fdw file scan */
	pg_atomic_uint32	af_rbatch_index;
	pg_atomic_uint32	af_rbatch_nload;	/* # of loaded record-batches */
	pg_atomic_uint32	af_rbatch_nskip;	/* # of skipped record-batches */
	/* for block-based regular table scan */
	BlockNumber			pbs_nblocks;		/* # blocks in relation at start of scan */
	slock_t				pbs_mutex;			/* lock of the fields below */
	BlockNumber			pbs_startblock;		/* starting block number */
	BlockNumber			pbs_nallocated;		/* # of blocks allocated to workers */
	/* common parallel table scan descriptor */
	ParallelTableScanDescData phscan;
} DpuScanSharedState;

/*
 * DpuScanState
 */
typedef struct
{
	pgstromTaskState	pts;
	DpuScanInfo			ds_info;
	XpuCommand		   *xcmd_req;
	size_t				xcmd_len;
} DpuScanState;

/*
 * create_dpuscan_path
 */
static CustomPath *
create_dpuscan_path(PlannerInfo *root,
					RelOptInfo *baserel,
					List *dev_quals,
					List *host_quals,
					bool parallel_aware,	/* for parallel-scan */
					IndexOptInfo *indexOpt,	/* for BRIN-index */
					List *indexConds,		/* for BRIN-index */
					List *indexQuals,		/* for BRIN-index */
					int64_t indexNBlocks)	/* for BRIN-index */
{
	DpuScanInfo	   *ds_info = palloc0(sizeof(DpuScanInfo));
	CustomPath	   *cpath = makeNode(CustomPath);
	ParamPathInfo  *param_info;
	int				parallel_nworkers = 0;
	double			parallel_divisor = 1.0;
	Cost			startup_cost = pgstrom_dpu_setup_cost;
	Cost			run_cost = 0.0;
	Cost			disk_cost;
	Cost			comp_cost;
	double			ntuples;
	double			dpu_ratio;
	double			selectivity;
	double			avg_seq_page_cost;
	double			spc_seq_page_cost;
	double			spc_rand_page_cost;
	QualCost		qcost;

	/* CPU parallel parameters */
	if (parallel_aware)
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
	}
	/* cost of full-disk scan */
	get_tablespace_page_costs(baserel->reltablespace,
							  &spc_rand_page_cost,
							  &spc_seq_page_cost);
	avg_seq_page_cost = (spc_seq_page_cost * (1.0 - baserel->allvisfrac) +
						 pgstrom_dpu_seq_page_cost * baserel->allvisfrac);
	disk_cost = avg_seq_page_cost * baserel->pages;

	/* consideration of BRIN-index, if any */
	ntuples = baserel->tuples;
	if (indexOpt)
	{
		Cost	index_disk_cost;

		index_disk_cost = cost_brin_bitmap_build(root,
												 baserel,
												 indexOpt,
												 indexQuals);
		index_disk_cost += avg_seq_page_cost * indexNBlocks;
		if (disk_cost > index_disk_cost)
		{
			if (baserel->pages > 0)
				ntuples *= (double)indexNBlocks / (double)baserel->pages;
			disk_cost = index_disk_cost;
		}
		else
			indexOpt = NULL;	/* disables BRIN-index if no benefit */
	}

	/* Cost for CPU/DPU qualifiers */
	if (cpu_operator_cost > 0.0)
		dpu_ratio = pgstrom_dpu_operator_cost / cpu_operator_cost;
	else if (pgstrom_dpu_operator_cost == 0.0)
		dpu_ratio = 1.0;
	else
		dpu_ratio = disable_cost;	/* very large value but finite */

	cost_qual_eval_node(&qcost, (Node *)dev_quals, root);
	startup_cost += qcost.startup;
	comp_cost = qcost.per_tuple *
		(ntuples * (1.0 - baserel->allvisfrac) +		/* by CPU */
		 ntuples * baserel->allvisfrac * dpu_ratio);	/* by DPU */

	/* Cost for DMA transfers (DPU-->Host) */
	selectivity = clauselist_selectivity(root,
										 dev_quals,
										 baserel->relid,
										 JOIN_INNER,
										 NULL);
	ntuples *= selectivity;		/* # tuples after dev_quals */
	run_cost += pgstrom_dpu_tuple_cost * ntuples * baserel->allvisfrac;

	/* Cost for Host-only qualifiers */
	cost_qual_eval(&qcost, host_quals, root);
	startup_cost += qcost.startup;
	comp_cost += qcost.per_tuple * ntuples;

	/* PPI costs (as a part of host quals, if any) */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		List   *ppi_quals = param_info->ppi_clauses;

		cost_qual_eval(&qcost, ppi_quals, root);
		startup_cost += qcost.startup;
		comp_cost += qcost.per_tuple * ntuples;
	}

	/* Cost for Projection */
	startup_cost += baserel->reltarget->cost.startup;
	comp_cost += baserel->reltarget->cost.per_tuple * baserel->rows;

	run_cost += disk_cost + (comp_cost / parallel_divisor);

	/* setup DpuScanInfo */
	if (indexOpt)
	{
		ds_info->index_oid = indexOpt->indexoid;
		ds_info->index_conds = indexConds;
		ds_info->index_quals = indexQuals;
	}
	/* setup CustomPath */
	cpath->path.pathtype = T_CustomScan;
	cpath->path.parent = baserel;
	cpath->path.pathtarget = baserel->reltarget;
	cpath->path.param_info = param_info;
	cpath->path.parallel_aware = parallel_aware;
	cpath->path.parallel_safe = baserel->consider_parallel;
	cpath->path.parallel_workers = parallel_nworkers;
	cpath->path.rows = (param_info
						? param_info->ppi_rows
						: baserel->rows) / parallel_divisor;
	cpath->path.startup_cost = startup_cost;
	cpath->path.total_cost = startup_cost + run_cost;
	cpath->path.pathkeys = NIL; /* unsorted results */
	cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
	cpath->custom_paths = NIL;
	cpath->custom_private = list_make1(ds_info);
	cpath->methods = &dpuscan_path_methods;

	return cpath;
}

/*
 * DpuScanAddScanPath
 */
static void
DpuScanAddScanPath(PlannerInfo *root,
				   RelOptInfo *baserel,
				   Index rtindex,
				   RangeTblEntry *rte)
{
	CustomPath *cpath;
	List	   *dev_quals = NIL;
	List	   *host_quals = NIL;
	IndexOptInfo *indexOpt;
	List	   *indexConds;
	List	   *indexQuals;
	int64_t		indexNBlocks;
	ListCell   *lc;

	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);
	/* nothing to do, if either PG-Strom or DpuScan is not enabled */
	if (!pgstrom_enabled || !enable_dpuscan)
		return;
	/* We already proved the relation empty, so nothing more to do */
	if (is_dummy_rel(baserel))
		return;
	/* It is the role of built-in Append node */
	if (rte->inh)
		return;
	/*
	 * DpuScan can run on only base relations or foreign table (arrow_fdw)
	 */
	switch (rte->relkind)
	{
		case RELKIND_RELATION:
		case RELKIND_MATVIEW:
			if (get_relation_am(rte->relid, true) == HEAP_TABLE_AM_OID &&
				GetOptimalDpuForTablespace(baserel->reltablespace) != NULL)
				break;
			return;
		case RELKIND_FOREIGN_TABLE:
#if 0
			// MEMO: directory --> tablespace mapping is necessary
			if (baseRelIsArrowFdw(baserel))
				break;
			return;
#endif
		default:
			/* not supported */
			return;
	}

	/* check whether the qualifier can run on DPU device */
	foreach (lc, baserel->baserestrictinfo)
    {
		RestrictInfo *rinfo = lfirst(lc);
		List	   *input_rels_tlist = list_make1(makeInteger(baserel->relid));

		if (pgstrom_dpu_expression(rinfo->clause, input_rels_tlist, NULL))
			dev_quals = lappend(dev_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}
	if (dev_quals == NIL)
		return;

	/* Check availability of DpuScan+BRIN Index */
	indexOpt = pgstromTryFindBrinIndex(root, baserel,
									   &indexConds,
									   &indexQuals,
									   &indexNBlocks);
	/* add DpuScan path in single process */
	cpath = create_dpuscan_path(root,
								baserel,
								dev_quals,
								host_quals,
								false,
								indexOpt,
								indexConds,
								indexQuals,
								indexNBlocks);
	if (cpath && custom_path_remember(root, baserel, false, false, cpath))
		add_path(baserel, &cpath->path);
	/* if appropriate, consider parallel DpuScan */
	if (baserel->consider_parallel && baserel->lateral_relids == NULL)
	{
		cpath = create_dpuscan_path(root,
									baserel,
									dev_quals,
									host_quals,
									true,
									indexOpt,
									indexConds,
									indexQuals,
									indexNBlocks);
		if (cpath && custom_path_remember(root, baserel, true, false, cpath))
			add_partial_path(baserel, &cpath->path);
	}
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
	DpuScanInfo	   *ds_info = linitial(best_path->custom_private);
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
								  ds_info,
								  &dpuscan_plan_methods);
	form_gpuscan_info(cscan, ds_info);

	return &cscan->scan.plan;
}

/*
 * CreateDpuScanState
 */
static Node *
CreateDpuScanState(CustomScan *cscan)
{
	DpuScanState   *dss = palloc0(sizeof(DpuScanState));

	NodeSetTag(dss, T_CustomScanState);
    dss->pts.css.flags = cscan->flags;
    Assert(cscan->methods == &dpuscan_plan_methods);
    dss->pts.css.methods = &dpuscan_exec_methods;
    deform_dpuscan_info(&dss->ds_info, cscan);

	return (Node *)dss;
}

/*
 * ExecInitDpuScan
 */
static void
ExecInitDpuScan(CustomScanState *node, EState *estate, int eflags)
{
	DpuScanState   *dss = (DpuScanState *)node;
	Relation		relation = node->ss.ss_currentRelation;

	 /* sanity checks */
    Assert(relation != NULL &&
           outerPlanState(node) == NULL &&
           innerPlanState(node) == NULL);
	dss->pts.ds_entry = GetOptimalDpuForRelation(relation);
	if (!dss->pts.ds_entry)
		elog(ERROR, "No DPU is installed on the relation: %s",
			 RelationGetRelationName(relation));

	pgstromBrinIndexExecBegin(&dss->pts,
                              dss->ds_info.index_oid,
                              dss->ds_info.index_conds,
                              dss->ds_info.index_quals);

	pgstromExecInitTaskState(&dss->pts, dss->ds_info.dev_quals);
	dss->pts.cb_cpu_fallback = ExecFallbackCpuScan;
}

/*
 * DpuScanReCheckTuple
 */
static bool
DpuScanReCheckTuple(DpuScanState *dss, TupleTableSlot *epq_slot)
{
	/*
	 * NOTE: Only immutable operators/functions are executable
	 * on DPU devices, so its decision will never changed.
	 */
	return true;
}

/*
 * ExecDpuScan
 */
static TupleTableSlot *
ExecDpuScan(CustomScanState *node)
{
	DpuScanState   *dss = (DpuScanState *)node;

	if (!dss->pts.ps_state)
		pgstromSharedStateInitDSM(&dss->pts, NULL);
	if (!dss->pts.conn)
	{
		const XpuCommand *session;

		session = pgstromBuildSessionInfo(&dss->pts.css.ss.ps,
										  dss->ds_info.used_params,
										  dss->ds_info.extra_bufsz,
										  dss->ds_info.kvars_nslots,
										  dss->ds_info.kern_quals,
										  dss->ds_info.kern_projs);
		DpuClientOpenSession(&dss->pts, session);
	}
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecTaskState,
					(ExecScanRecheckMtd) DpuScanReCheckTuple);
}

/*
 * ExecEndDpuScan
 */
static void
ExecEndDpuScan(CustomScanState *node)
{
	pgstromExecEndTaskState((pgstromTaskState *)node);
}

/*
 * ExecReScanDpuScan
 */
static void
ExecReScanDpuScan(CustomScanState *node)
{
	pgstromExecResetTaskState((pgstromTaskState *)node);
}

/*
 * EstimateDpuScanDSM
 */
static Size
EstimateDpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return pgstromSharedStateEstimateDSM((pgstromTaskState *)node);
}

/*
 * InitializeDpuScanDSM
 */
static void
InitializeDpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *dsm_addr)
{
	pgstromSharedStateInitDSM((pgstromTaskState *)node, dsm_addr);
}

/*
 * ReInitializeDpuScanDSM
 */
static void
ReInitializeDpuScanDSM(CustomScanState *node,
					   ParallelContext *pcxt,
					   void *dsm_addr)
{
	pgstromSharedStateReInitDSM((pgstromTaskState *)node);
}

/*
 * InitDpuScanWorker
 */
static void
InitDpuScanWorker(CustomScanState *node, shm_toc *toc, void *dsm_addr)
{
	pgstromSharedStateAttachDSM((pgstromTaskState *)node, dsm_addr);
}

/*
 * ExecShutdownDpuScan
 */
static void
ExecShutdownDpuScan(CustomScanState *node)
{
	pgstromSharedStateShutdownDSM((pgstromTaskState *)node);
}

/*
 * ExplainDpuScan
 */
static void
ExplainDpuScan(CustomScanState *node,
               List *ancestors,
               ExplainState *es)
{
	DpuScanState   *dss = (DpuScanState *) node;
	StringInfoData	temp;

	initStringInfo(&temp);
	pgstrom_explain_xpucode(&temp,
							dss->ds_info.kern_quals,
							&dss->pts.css,
							es, ancestors);
	ExplainPropertyText("DPU Quals", temp.data, es);

	resetStringInfo(&temp);
    pgstrom_explain_xpucode(&temp,
                            dss->ds_info.kern_projs,
                            &dss->pts.css,
                            es, ancestors);
    ExplainPropertyText("DPU Projection", temp.data, es);
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
    dpuscan_exec_methods.BeginCustomScan	= ExecInitDpuScan;
    dpuscan_exec_methods.ExecCustomScan		= ExecDpuScan;
    dpuscan_exec_methods.EndCustomScan		= ExecEndDpuScan;
    dpuscan_exec_methods.ReScanCustomScan	= ExecReScanDpuScan;
    dpuscan_exec_methods.EstimateDSMCustomScan = EstimateDpuScanDSM;
    dpuscan_exec_methods.InitializeDSMCustomScan = InitializeDpuScanDSM;
    dpuscan_exec_methods.InitializeWorkerCustomScan = InitDpuScanWorker;
    dpuscan_exec_methods.ReInitializeDSMCustomScan = ReInitializeDpuScanDSM;
    dpuscan_exec_methods.ShutdownCustomScan = ExecShutdownDpuScan;
    dpuscan_exec_methods.ExplainCustomScan	= ExplainDpuScan;

    /* hook registration */
    set_rel_pathlist_next = set_rel_pathlist_hook;
    set_rel_pathlist_hook = DpuScanAddScanPath;
}
