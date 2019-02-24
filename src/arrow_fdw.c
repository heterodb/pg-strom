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
static kern_tupdesc	   *arrowSchemaToKernTupdesc(ArrowSchema *schema);
static List			   *arrowFdwExtractFilesList(List *options_list);

Datum	pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS);
Datum	pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS);

/*
 * ArrowGetForeignRelSize
 */
static void
ArrowGetForeignRelSize(PlannerInfo *root,
					   RelOptInfo *baserel,
					   Oid foreigntableid)
{
	ForeignTable   *ft = GetForeignTable(foreigntableid);
	List		   *filesList = arrowFdwExtractFilesList(ft->options);
	kern_tupdesc   *ktdesc = NULL;
	BlockNumber		npages = 0;
	double			ntuples = 0.0;
	ListCell	   *lc1, *lc2;

	foreach (lc1, filesList)
	{
		ArrowFileInfo	af_info;
		const char	   *fname = strVal(lfirst(lc1));
		kern_tupdesc   *temp;
		struct stat		st_buf;

		if (stat(fname, &st_buf) != 0)
			elog(ERROR, "failed on stat('%s') on behalf of foreign table: %s",
				 fname, get_rel_name(foreigntableid));
		npages += (st_buf.st_size + BLCKSZ - 1) / BLCKSZ;

		readArrowFile(fname, &af_info);
		foreach (lc2, af_info.recordBatches)
			ntuples += ((ArrowRecordBatch *)lfirst(lc2))->length;

		temp = arrowSchemaToKernTupdesc(&af_info.footer.schema);
		if (!ktdesc)
			ktdesc = temp;
		else if (!kern_tupdesc_equal(ktdesc, temp))
			elog(ERROR, "file '%s' has incompatible schema definition from other files on behalf of foreign table: %s",
				 fname, get_rel_name(foreigntableid));
	}
	baserel->fdw_private = (void *)ktdesc;
	baserel->pages = npages;
	baserel->tuples = ntuples;
	baserel->rows = ntuples *
		clauselist_selectivity(root,
							   baserel->baserestrictinfo,
							   0,
							   JOIN_INNER,
							   NULL);
}

static void
cost_arrow_fdw_seqscan(Path *path,
					   PlannerInfo *root,
					   RelOptInfo *baserel,
					   ParamPathInfo *param_info,
					   int parallel_nworkers)
{
	Cost		startup_cost = 0.0;
	Cost		disk_run_cost = 0.0;
	Cost		cpu_run_cost = 0.0;
	QualCost	qcost;
	double		nrows;
	size_t		width_all = 0;
	size_t		width_read = 0;
	double		width_ratio;
	int			i;
	double		spc_seq_page_cost;
	bool		whole_row_refs = false;

	if (param_info)
		nrows = param_info->ppi_rows;
	else
		nrows = baserel->rows;

	/* Storage costs */
	for (i=baserel->min_attr; i <= baserel->max_attr; i++)
	{
		if (i < 0)
			continue;
		else if (i == 0)
			whole_row_refs = true;
		else if (baserel->attr_needed[i])
			width_read += baserel->attr_widths[i];

		width_all += baserel->attr_widths[i];
	}
	if (whole_row_refs)
		width_read = width_all;
	width_ratio = (double)width_read / (double)width_all;

	get_tablespace_page_costs(baserel->reltablespace,
							  NULL,
							  &spc_seq_page_cost);
	disk_run_cost += spc_seq_page_cost * width_ratio * baserel->pages;

	/* CPU costs */
	if (param_info)
	{
		cost_qual_eval(&qcost, param_info->ppi_clauses, root);
		qcost.startup += baserel->baserestrictcost.startup;
        qcost.per_tuple += baserel->baserestrictcost.per_tuple;
	}
	else
		qcost = baserel->baserestrictcost;
	startup_cost += qcost.startup;
	cpu_run_cost = (cpu_tuple_cost + qcost.per_tuple) * baserel->tuples;

	/* tlist evaluation costs */
	startup_cost += path->pathtarget->cost.startup;
	cpu_run_cost += path->pathtarget->cost.per_tuple * path->rows;

	/* adjust cost for CPU parallelism */
	if (parallel_nworkers > 0)
	{
		double		leader_contribution;
		double		parallel_divisor = (double) parallel_nworkers;

		/* see get_parallel_divisor() */
		leader_contribution = 1.0 - (0.3 * (double)parallel_nworkers);
		parallel_divisor += Max(leader_contribution, 0.0);

		/* The CPU cost is divided among all the workers. */
		cpu_run_cost /= parallel_divisor;

		/* Estimated row count per background worker process */
		nrows = clamp_row_est(nrows / parallel_divisor);
	}
	path->rows = nrows;
	path->startup_cost = startup_cost;
	path->total_cost = startup_cost + cpu_run_cost + disk_run_cost;
}

/*
 * ArrowGetForeignPaths
 */
static void
ArrowGetForeignPaths(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid foreigntableid)
{
	ForeignPath	   *fpath;
	ParamPathInfo  *param_info;
	Relids			required_outer = baserel->lateral_relids;

	fpath = create_foreignscan_path(root, baserel,
									NULL,	/* default pathtarget */
									1000,	/* set below */
									1234.0,	/* set below */
									2345.0,	/* set below */
									NIL,	/* no pathkeys */
									required_outer,
									NULL,	/* no extra plan */
									NIL);	/* no particular private */
	param_info = get_baserel_parampathinfo(root, baserel, required_outer);
	/* update nrows, startup_cost and total_cost */
	cost_arrow_fdw_seqscan(&fpath->path, root, baserel, param_info, 0);

	add_path(baserel, (Path *)fpath);
}

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



static kern_tupdesc *
arrowSchemaToKernTupdesc(ArrowSchema *schema)
{
	kern_tupdesc   *result = NULL;
	int				i, nattrs = schema->_num_fields;

	result = palloc0(offsetof(kern_tupdesc, colmeta[nattrs]));
	result->nattrs = nattrs;
	for (i=0; i < nattrs; i++)
	{
		ArrowField	   *field = &schema->fields[i];
		kern_colmeta   *cmeta = &result->colmeta[i];
		Oid				type_oid = InvalidOid;
		int32			type_mod = -1;
		Oid				nsp_id;
		int16			typlen;
		bool			typbyval;
		char			typalign;

		switch (field->type.tag)
		{
			case ArrowNodeTag__Int:
				switch (field->type.Int.bitWidth)
				{
					case 16:
						type_oid = INT2OID;
						break;
					case 32:
						type_oid = INT4OID;
						break;
					case 64:
						type_oid = INT8OID;
						break;
					default:
						elog(ERROR, "cannot map %s%d of Apache Arrow on PostgreSQL type",
							 field->type.Int.is_signed ? "Int" : "Uint",
							 field->type.Int.bitWidth);
						break;
				}
				break;
			case ArrowNodeTag__FloatingPoint:
				switch (field->type.FloatingPoint.precision)
				{
					case ArrowPrecision__Half:
						nsp_id = get_namespace_oid(PGSTROM_SCHEMA_NAME, false);
						type_oid = GetSysCacheOid2(TYPENAMENSP,
												   PointerGetDatum("float2"),
												   ObjectIdGetDatum(nsp_id));
						if (!OidIsValid(type_oid))
							elog(ERROR, "float2 is not defined at PostgreSQL");
						break;
					case ArrowPrecision__Single:
						type_oid = FLOAT4OID;
						break;
					case ArrowPrecision__Double:
						type_oid = FLOAT8OID;
						break;
					default:
						elog(ERROR, "unknown floating-point precision");
						break;
				}
				break;
			case ArrowNodeTag__Utf8:
				type_oid = TEXTOID;
				break;
			case ArrowNodeTag__Binary:
			case ArrowNodeTag__FixedSizeBinary:
				type_oid = BYTEAOID;
				break;
			case ArrowNodeTag__Bool:
				type_oid = BOOLOID;
				break;
			case ArrowNodeTag__Decimal:
				type_oid = NUMERICOID;
				break;
			case ArrowNodeTag__Date:
				type_oid = DATEOID;
				break;
			case ArrowNodeTag__Time:
				type_oid = TIMEOID;
				break;
			case ArrowNodeTag__Timestamp:
				if (field->type.Timestamp.timezone)
					elog(ERROR, "Timestamp with timezone of Apache Arrow is not supported, right now");
				type_oid = TIMESTAMPOID;
				break;
			case ArrowNodeTag__Interval:
				type_oid = INTERVALOID;
				break;
			default:
				elog(ERROR, "Not a supported Apache Arrow type");
				break;
		}
		get_typlenbyvalalign(type_oid, &typlen, &typbyval, &typalign);

		cmeta->attbyval		= typbyval;
		cmeta->attalign		= typealign_get_width(typalign);
		cmeta->attlen		= typlen;
		cmeta->attnum		= i+1;
		cmeta->attcacheoff	= -1;
		cmeta->atttypid		= type_oid;
		cmeta->atttypmod	= type_mod;
	}
	return result;
}




static List *
arrowFdwExtractFilesList(List *options_list)
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
		List	   *filesList = arrowFdwExtractFilesList(options_list);
		ListCell   *lc;

		foreach (lc, filesList)
		{
			ArrowFileInfo	af_info;
			const char	   *fname = strVal(lfirst(lc));

			readArrowFile(fname, &af_info);
			elog(INFO, "%s", dumpArrowNode((ArrowNode *)&af_info.footer));
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
