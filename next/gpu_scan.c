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
 * form/deform private information of CustomScan(GpuScan)
 */
typedef struct
{
	const Bitmapset *optimal_gpus;	/* optimal GPUs (if any) */
	bytea		   *kern_quals;		/* device qualifiers */
	bytea		   *kern_projs;		/* device projection */
	uint32_t		extra_flags;
	uint32_t		extra_bufsz;
	uint32_t		nrows_per_block;	/* average # rows per block */
	const Bitmapset *outer_refs;	/* referenced columns */
	List		   *used_params;	/* Param list in use */
	List		   *dev_quals;		/* Device qualifiers */
	Oid				index_oid;		/* OID of BRIN-index, if any */
	List		   *index_conds;	/* BRIN-index key conditions */
	List		   *index_quals;	/* Original BRIN-index qualifier*/
} GpuScanInfo;

static void
form_gpuscan_info(CustomScan *cscan, GpuScanInfo *gs_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, bms_to_pglist(gs_info->optimal_gpus));
	privs = lappend(privs, makeConst(BYTEAOID,
									 -1,
									 InvalidOid,
									 -1,
									 PointerGetDatum(gs_info->kern_quals),
									 (gs_info->kern_quals == NULL),
									 false));
	privs = lappend(privs, makeConst(BYTEAOID,
									 -1,
									 InvalidOid,
									 -1,
									 PointerGetDatum(gs_info->kern_projs),
									 (gs_info->kern_projs == NULL),
									 false));
	privs = lappend(privs, makeInteger(gs_info->extra_flags));
	privs = lappend(privs, makeInteger(gs_info->extra_bufsz));
	privs = lappend(privs, makeInteger(gs_info->nrows_per_block));
	privs = lappend(privs, bms_to_pglist(gs_info->outer_refs));
	exprs = lappend(exprs, gs_info->used_params);
	exprs = lappend(exprs, gs_info->dev_quals);
	privs = lappend(privs, makeInteger(gs_info->index_oid));
	exprs = lappend(exprs, gs_info->index_conds);
	exprs = lappend(exprs, gs_info->index_quals);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static GpuScanInfo *
deform_gpuscan_info(CustomScan *cscan)
{
	GpuScanInfo *gs_info = palloc0(sizeof(GpuScanInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;
	Const	   *con;

	gs_info->optimal_gpus	= bms_from_pglist(list_nth(privs, pindex++));
	con = list_nth(privs, pindex++);
	if (!con->constisnull)
		gs_info->kern_quals	= DatumGetByteaP(con->constvalue);
	con = list_nth(privs, pindex++);
	if (!con->constisnull)
		gs_info->kern_projs	= DatumGetByteaP(con->constvalue);
	gs_info->extra_flags	= intVal(list_nth(privs, pindex++));
	gs_info->extra_bufsz	= intVal(list_nth(privs, pindex++));
	gs_info->nrows_per_block = intVal(list_nth(privs, pindex++));
	gs_info->outer_refs		= bms_from_pglist(list_nth(privs, pindex++));
	gs_info->used_params	= list_nth(exprs, eindex++);
	gs_info->dev_quals		= list_nth(exprs, eindex++);
	gs_info->index_oid		= intVal(list_nth(privs, pindex++));
	gs_info->index_conds	= list_nth(exprs, eindex++);
	gs_info->index_quals	= list_nth(exprs, eindex++);

	return gs_info;
}

/*
 *
 */
static void
GpuScanAddScanPath(PlannerInfo *root,
				   RelOptInfo *baserel,
				   Index rtindex,
				   RangeTblEntry *rte)
{
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);
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
	return NULL;
}

/*
 * CreateGpuScanState
 */
static Node *
CreateGpuScanState(CustomScan *cscan)
{
	return NULL;
}

/*
 * ExecInitGpuScan
 */
static void
ExecInitGpuScan(CustomScanState *node, EState *estate, int eflags)
{}

/*
 * ExecGpuScan
 */
static TupleTableSlot *
ExecGpuScan(CustomScanState *node)
{
	return NULL;
}

/*
 * ExecEndGpuScan
 */
static void
ExecEndGpuScan(CustomScanState *node)
{}

/*
 * ExecReScanGpuScan
 */
static void
ExecReScanGpuScan(CustomScanState *node)
{}

/*
 * EstimateGpuScanDSM
 */
static Size
EstimateGpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return 0;
}

/*
 * InitializeGpuScanDSM
 */
static void
InitializeGpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *coordinate)
{}

/*
 * ReInitializeGpuScanDSM
 */
static void
ReInitializeGpuScanDSM(CustomScanState *node,
					   ParallelContext *pcxt,
					   void *coordinate)
{}

/*
 * InitGpuScanWorker
 */
static void
InitGpuScanWorker(CustomScanState *node, shm_toc *toc, void *coordinate)
{}

/*
 * ExecShutdownGpuScan
 */
static void
ExecShutdownGpuScan(CustomScanState *node)
{}

/*
 * ExplainGpuScan
 */
static void
ExplainGpuScan(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{}

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
    gpuscan_exec_methods.ReInitializeDSMCustomScan = ReInitializeGpuScanDSM;
    gpuscan_exec_methods.ShutdownCustomScan	= ExecShutdownGpuScan;
    gpuscan_exec_methods.ExplainCustomScan	= ExplainGpuScan;

	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = GpuScanAddScanPath;
}
