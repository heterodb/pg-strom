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
 * form/deform private information of CustomScan(DpuScan)
 */
typedef struct
{
	bytea	   *kern_quals;		/* device qualifiers */
	bytea	   *kern_projs;		/* device projection */
	
	
} DpuScanInfo;

static void
form_dpuscan_info(CustomScan *cscan, DpuScanInfo *ds_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, makeConst(BYTEAOID,
									 -1,
									 InvalidOid,
									 -1,
									 PointerGetDatum(ds_info->kern_quals),
									 (ds_info->kern_quals == NULL),
									 false));
	privs = lappend(privs, makeConst(BYTEAOID,
									 -1,
									 InvalidOid,
									 -1,
									 PointerGetDatum(ds_info->kern_projs),
									 (ds_info->kern_projs == NULL),
									 false));
	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static void
deform_gpuscan_info(DpuScanInfo *ds_info, CustomScan *cscan)
{
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;
	Const	   *con;

	con = list_nth(privs, pindex++);
	if (!con->constisnull)
		ds_info->kern_quals = DatumGetByteaP(con->constvalue);
	con = list_nth(privs, pindex++);
	if (!con->constisnull)
		ds_info->kern_projs = DatumGetByteaP(con->constvalue);
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
{}

/*
 * CreateDpuScanState
 */
static Node *
CreateDpuScanState(CustomScan *cscan)
{
	return NULL;
}

/*
 * ExecInitDpuScan
 */
static void
ExecInitDpuScan(CustomScanState *node, EState *estate, int eflags)
{
}

/*
 * ExecDpuScan
 */
static TupleTableSlot *
ExecDpuScan(CustomScanState *node)
{
	return NULL;
}

/*
 * ExecEndDpuScan
 */
static void
ExecEndDpuScan(CustomScanState *node)
{}

/*
 * ExecReScanDpuScan
 */
static void
ExecReScanDpuScan(CustomScanState *node)
{
}

/*
 * EstimateDpuScanDSM
 */
static Size
EstimateDpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return 0;
}

/*
 * InitializeDpuScanDSM
 */
static void
InitializeDpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *dsm_addr)
{
}

/*
 * ReInitializeDpuScanDSM
 */
static void
ReInitializeDpuScanDSM(CustomScanState *node,
					   ParallelContext *pcxt,
					   void *dsm_addr)
{}

/*
 * InitDpuScanWorker
 */
static void
InitDpuScanWorker(CustomScanState *node, shm_toc *toc, void *dsm_addr)
{}

/*
 * ExecShutdownDpuScan
 */
static void
ExecShutdownDpuScan(CustomScanState *node)
{
}

/*
 * ExplainDpuScan
 */
static void
ExplainDpuScan(CustomScanState *node,
               List *ancestors,
               ExplainState *es)
{}

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
