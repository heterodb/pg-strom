/*
 * arrow_fdw.c
 *
 * Routines to map Apache Arrow files as PG's Foreign-Table.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#include "pg_strom.h"
#include "arrow_defs.h"

/* ---------- static variables ---------- */
static FdwRoutine		pgstrom_arrow_fdw_routine;
/* ---------- static functions ---------- */
Datum	pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS);
Datum	pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS);

/*
 * ArrowGetForeignRelSize
 */
static void
ArrowGetForeignRelSize(PlannerInfo *root,
					   RelOptInfo *baserel,
					   Oid foreigntableid)
{}

/*
 * ArrowGetForeignPaths
 */
static void
ArrowGetForeignPaths(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid foreigntableid)
{}

/*
 * ArrowGetForeignPlan
 */
static ForeignScan *
ArrowGetForeignPlan(PlannerInfo *root,
					RelOptInfo *baserel,
					Oid foreigntableid,
					ForeignPath *best_path,
					List *tlist,
					List *scan_clauses,
					Plan *outer_plan)
{
	return NULL;
}

/*
 * ArrowBeginForeignScan
 */
static void
ArrowBeginForeignScan(ForeignScanState *node, int eflags)
{}

/*
 * ArrowIterateForeignScan
 */
static TupleTableSlot *
ArrowIterateForeignScan(ForeignScanState *node)
{
	return NULL;
}

/*
 * ArrowReScanForeignScan
 */
static void
ArrowReScanForeignScan(ForeignScanState *node)
{}

/*
 * ArrowEndForeignScan
 */
static void
ArrowEndForeignScan(ForeignScanState *node)
{}

/*
 * ArrowExplainForeignScan 
 */
static void
ArrowExplainForeignScan(ForeignScanState *node, ExplainState *es)
{}

/*
 * ArrowImportForeignSchema
 */
static List *
ArrowImportForeignSchema(ImportForeignSchemaStmt *stmt, Oid serverOid)
{
	return NIL;
}

/*
 * ArrowIsForeignScanParallelSafe
 */
static bool
ArrowIsForeignScanParallelSafe(PlannerInfo *root,
							   RelOptInfo *rel,
							   RangeTblEntry *rte)
{
	return false;
}

/*
 * ArrowEstimateDSMForeignScan 
 */
static Size
ArrowEstimateDSMForeignScan(ForeignScanState *node,
							ParallelContext *pcxt)
{
	return 0;
}

/*
 * ArrowInitializeDSMForeignScan
 */
static void
ArrowInitializeDSMForeignScan(ForeignScanState *node,
							  ParallelContext *pcxt,
							  void *coordinate)
{}

/*
 * ArrowReInitializeDSMForeignScan
 */
static void
ArrowReInitializeDSMForeignScan(ForeignScanState *node,
								ParallelContext *pcxt,
								void *coordinate)
{}

/*
 * ArrowInitializeWorkerForeignScan
 */
static void
ArrowInitializeWorkerForeignScan(ForeignScanState *node,
								 shm_toc *toc,
								 void *coordinate)
{}

/*
 * ArrowShutdownForeignScan
 */
static void
ArrowShutdownForeignScan(ForeignScanState *node)
{}

/*
 * handler of Arrow_Fdw
 */
Datum
pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&pgstrom_arrow_fdw_routine);
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_handler);

static List *
arrow_fdw_extract_files_list(List *options_list)
{
	ListCell   *lc;
	List	   *filesList = NIL;
	char	   *dir_path = NULL;
	char	   *dir_suffix = NULL;

	foreach (lc, options_list)
	{
		DefElem	   *defel = lfirst(lc);

		Assert(IsA(defel->arg, String));
		if (strcmp(defel->defname, "file") == 0)
		{
			char   *temp = strVal(defel->arg);
			filesList = lappend(filesList, makeString(pstrdup(temp)));
		}
		else if (strcmp(defel->defname, "files") == 0)
		{
			char   *temp = pstrdup(strVal(defel->arg));
			char   *saveptr;
			char   *tok, *pos;

			while ((tok = strtok_r(temp, ",", &saveptr)) != NULL)
			{
				while (isspace(*tok))
					tok++;
				pos = tok + strlen(tok) - 1;
				while (pos >= tok && isspace(*pos))
					*pos-- = '\0';

				filesList = lappend(filesList, makeString(pstrdup(tok)));

				temp = NULL;
			}
		}
		else if (strcmp(defel->defname, "dir") == 0)
		{
			dir_path = strVal(defel->arg);
		}
		else if (strcmp(defel->defname, "suffix") == 0)
		{
			dir_suffix = strVal(defel->arg);
		}
	}
	if (dir_suffix && !dir_path)
		elog(ERROR, "arrow: cannot use 'suffix' option without 'dir'");

	if (dir_path)
	{
		struct dirent *dentry;
		DIR	   *dir;
		char   *temp;

		dir = AllocateDir(dir_path);
		while ((dentry = ReadDir(dir, dir_path)) != NULL)
		{
			if (strcmp(dentry->d_name, ".") == 0 ||
				strcmp(dentry->d_name, "..") == 0)
				continue;
			if (dir_suffix)
			{
				int		dlen = strlen(dentry->d_name);
				int		slen = strlen(dir_suffix);
				int		diff;

				if (dlen < 2 + slen)
					continue;
				diff = dlen - slen;
				if (dentry->d_name[diff-1] != '.' ||
					strcmp(dentry->d_name + diff, dir_suffix) != 0)
					continue;
			}
			temp = psprintf("%s/%s", dir_path, dentry->d_name);
			filesList = lappend(filesList, makeString(temp));
		}
		FreeDir(dir);
	}

	if (filesList == NIL)
		elog(ERROR, "arrow: no files are on behalf of the foreign table");
	foreach (lc, filesList)
	{
		Value  *val = lfirst(lc);

		if (access(strVal(val), F_OK | R_OK) != 0)
			elog(ERROR, "arrow: permission error at '%s': %m", strVal(val));
	}
	return filesList;
}

/*
 * validator of Arrow_Fdw
 */
Datum
pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS)
{
	List	   *options_list = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid			catalog = PG_GETARG_OID(1);

	if (catalog == ForeignTableRelationId)
	{
		List	   *filesList = arrow_fdw_extract_files_list(options_list);
		ListCell   *lc;

		foreach (lc, filesList)
		{
			ArrowFileInfo	af_info;
			const char	   *fname = strVal(lfirst(lc));

			readArrowFile(fname, &af_info);
			//dump...
			//TODO: make cache item here
		}
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_validator);



/*
 * pgstrom_init_arrow_fdw
 */
void
pgstrom_init_arrow_fdw(void)
{
	FdwRoutine *r = &pgstrom_arrow_fdw_routine;

	memset(r, 0, sizeof(FdwRoutine));
	NodeSetTag(r, T_FdwRoutine);
	/* SCAN support */
	r->GetForeignRelSize			= ArrowGetForeignRelSize;
	r->GetForeignPaths				= ArrowGetForeignPaths;
	r->GetForeignPlan				= ArrowGetForeignPlan;
	r->BeginForeignScan				= ArrowBeginForeignScan;
	r->IterateForeignScan			= ArrowIterateForeignScan;
	r->ReScanForeignScan			= ArrowReScanForeignScan;
	r->EndForeignScan				= ArrowEndForeignScan;
	/* EXPLAIN support */
	r->ExplainForeignScan			= ArrowExplainForeignScan;
	/* IMPORT FOREIGN SCHEMA support */
	r->ImportForeignSchema			= ArrowImportForeignSchema;
	/* CPU Parallel support (not yet) */
	r->IsForeignScanParallelSafe	= ArrowIsForeignScanParallelSafe;
	r->EstimateDSMForeignScan		= ArrowEstimateDSMForeignScan;
	r->InitializeDSMForeignScan		= ArrowInitializeDSMForeignScan;
	r->ReInitializeDSMForeignScan	= ArrowReInitializeDSMForeignScan;
	r->InitializeWorkerForeignScan	= ArrowInitializeWorkerForeignScan;
	r->ShutdownForeignScan			= ArrowShutdownForeignScan;
}
