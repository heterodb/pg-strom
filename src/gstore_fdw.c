/*
 * gstore_fdw.c
 *
 * On GPU column based data store as FDW provider.
 * ----
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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

/*
 * gstoreScanState - state object for scan/insert/update/delete
 */
typedef struct
{
	GpuStoreBuffer *gs_buffer;
	cl_ulong		gs_index;
	AttrNumber		ctid_anum;	/* only UPDATE or DELETE */
} GpuStoreExecState;

/* ---- static variables ---- */
static Oid				reggstore_type_oid = InvalidOid;

Datum pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_in(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_out(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_recv(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_send(PG_FUNCTION_ARGS);

/*
 * gstoreGetForeignRelSize
 */
static void
gstoreGetForeignRelSize(PlannerInfo *root,
						RelOptInfo *baserel,
						Oid ftable_oid)
{
	Snapshot	snapshot;
	Size		rawsize;
	Size		nitems;

	snapshot = RegisterSnapshot(GetTransactionSnapshot());
	GpuStoreBufferGetSize(ftable_oid, snapshot, &rawsize, &nitems);
	UnregisterSnapshot(snapshot);

	baserel->rows	= nitems;
	baserel->pages	= (rawsize + BLCKSZ - 1) / BLCKSZ;
}

/*
 * gstoreGetForeignPaths
 */
static void
gstoreGetForeignPaths(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Oid foreigntableid)
{
	ParamPathInfo *param_info;
	ForeignPath *fpath;
	Cost		startup_cost = baserel->baserestrictcost.startup;
	Cost		per_tuple = baserel->baserestrictcost.per_tuple;
	Cost		run_cost;
	QualCost	qcost;

	param_info = get_baserel_parampathinfo(root, baserel, NULL);
	if (param_info)
	{
		cost_qual_eval(&qcost, param_info->ppi_clauses, root);
		startup_cost += qcost.startup;
		per_tuple += qcost.per_tuple;
	}
	run_cost = per_tuple * baserel->rows;

	fpath = create_foreignscan_path(root,
									baserel,
									NULL,	/* default pathtarget */
									baserel->rows,
									startup_cost,
									startup_cost + run_cost,
									NIL,	/* no pathkeys */
									NULL,	/* no outer rel either */
									NULL,	/* no extra plan */
									NIL);	/* no fdw_private */
	add_path(baserel, (Path *) fpath);
}

/*
 * gstoreGetForeignPlan
 */
static ForeignScan *
gstoreGetForeignPlan(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid foreigntableid,
					 ForeignPath *best_path,
					 List *tlist,
					 List *scan_clauses,
					 Plan *outer_plan)
{
	List	   *scan_quals = NIL;
	ListCell   *lc;

	foreach (lc, scan_clauses)
	{
		RestrictInfo *rinfo = (RestrictInfo *) lfirst(lc);

		Assert(IsA(rinfo, RestrictInfo));
		if (rinfo->pseudoconstant)
			continue;
		scan_quals = lappend(scan_quals, rinfo->clause);
	}

	return make_foreignscan(tlist,
							scan_quals,
							baserel->relid,
							NIL,		/* fdw_exprs */
							NIL,		/* fdw_private */
							NIL,		/* fdw_scan_tlist */
							NIL,		/* fdw_recheck_quals */
							NULL);		/* outer_plan */
}

/*
 * gstoreAddForeignUpdateTargets
 */
static void
gstoreAddForeignUpdateTargets(Query *parsetree,
							  RangeTblEntry *target_rte,
							  Relation target_relation)
{
	Var			*var;
	TargetEntry *tle;

	/*
	 * We carry row_index as ctid system column
	 */

	/* Make a Var representing the desired value */
	var = makeVar(parsetree->resultRelation,
				  SelfItemPointerAttributeNumber,
				  TIDOID,
				  -1,
				  InvalidOid,
				  0);

	/* Wrap it in a resjunk TLE with the right name ... */
	tle = makeTargetEntry((Expr *) var,
						  list_length(parsetree->targetList) + 1,
						  "ctid",
						  true);

	/* ... and add it to the query's targetlist */
	parsetree->targetList = lappend(parsetree->targetList, tle);
}

/*
 * gstoreBeginForeignScan
 */
static void
gstoreBeginForeignScan(ForeignScanState *node, int eflags)
{
	EState	   *estate = node->ss.ps.state;
	GpuStoreExecState *gstate;

	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;

	if (!IsMVCCSnapshot(estate->es_snapshot))
		elog(ERROR, "cannot scan gstore_fdw table without MVCC snapshot");

	gstate = palloc0(sizeof(GpuStoreExecState));
	node->fdw_state = (void *) gstate;
}

/*
 * gstoreIterateForeignScan
 */
static TupleTableSlot *
gstoreIterateForeignScan(ForeignScanState *node)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) node->fdw_state;
	Relation		frel = node->ss.ss_currentRelation;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	EState		   *estate = node->ss.ps.state;
	ForeignScan	   *fscan = (ForeignScan *)node->ss.ps.plan;

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, estate->es_snapshot);
	if (GpuStoreBufferGetNext(frel,
							  estate->es_snapshot,
							  slot,
							  gstate->gs_buffer,
							  &gstate->gs_index,
							  fscan->fsSystemCol))
		return slot;

	return NULL;
}

/*
 * gstoreReScanForeignScan
 */
static void
gstoreReScanForeignScan(ForeignScanState *node)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) node->fdw_state;

	gstate->gs_index = 0;
}

/*
 * gstoreEndForeignScan
 */
static void
gstoreEndForeignScan(ForeignScanState *node)
{
	//GpuStoreExecState  *gstate = (GpuStoreExecState *) node->fdw_state;
}

/*
 * gstorePlanForeignModify
 */
static List *
gstorePlanForeignModify(PlannerInfo *root,
						ModifyTable *plan,
						Index resultRelation,
						int subplan_index)
{
	CmdType		operation = plan->operation;

	if (operation != CMD_INSERT &&
		operation != CMD_UPDATE &&
		operation != CMD_DELETE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("gstore_fdw: not a supported operation")));
	return NIL;
}

/*
 * gstoreBeginForeignModify
 */
static void
gstoreBeginForeignModify(ModifyTableState *mtstate,
						 ResultRelInfo *rrinfo,
						 List *fdw_private,
						 int subplan_index,
						 int eflags)
{
	GpuStoreExecState *gstate = palloc0(sizeof(GpuStoreExecState));
	Relation	frel = rrinfo->ri_RelationDesc;
	CmdType		operation = mtstate->operation;

	/*
	 * NOTE: gstore_fdw does not support update operations by multiple
	 * concurrent transactions. So, we require stronger lock than usual
	 * INSERT/UPDATE/DELETE operations. It may lead unexpected deadlock,
	 * in spite of the per-tuple update capability.
	 */
	LockRelationOid(RelationGetRelid(frel), ShareUpdateExclusiveLock);

	/* Find the ctid resjunk column in the subplan's result */
	if (operation == CMD_UPDATE || operation == CMD_DELETE)
	{
		Plan	   *subplan = mtstate->mt_plans[subplan_index]->plan;
		AttrNumber	ctid_anum;

		ctid_anum = ExecFindJunkAttributeInTlist(subplan->targetlist, "ctid");
		if (!AttributeNumberIsValid(ctid_anum))
			elog(ERROR, "could not find junk ctid column");
		gstate->ctid_anum = ctid_anum;
	}
	rrinfo->ri_FdwState = gstate;
}

/*
 * gstoreExecForeignInsert
 */
static TupleTableSlot *
gstoreExecForeignInsert(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Snapshot		snapshot = estate->es_snapshot;
	Relation		frel = rrinfo->ri_RelationDesc;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, snapshot);

	GpuStoreBufferAppendRow(gstate->gs_buffer,
							RelationGetDescr(frel),
							snapshot,
							slot);
	return slot;
}

/*
 * gstoreExecForeignUpdate
 */
static TupleTableSlot *
gstoreExecForeignUpdate(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Relation		frel = rrinfo->ri_RelationDesc;
	Snapshot		snapshot = estate->es_snapshot;
	Datum			datum;
	bool			isnull;
	ItemPointer		t_self;
	size_t			old_index;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, snapshot);

	/* remove old version of the row */
	datum = ExecGetJunkAttribute(planSlot,
								 gstate->ctid_anum,
								 &isnull);
	if (isnull)
		elog(ERROR, "gstore_fdw: ctid is null");
	t_self = (ItemPointer)DatumGetPointer(datum);
	old_index = ((cl_ulong)t_self->ip_blkid.bi_hi << 32 |
				 (cl_ulong)t_self->ip_blkid.bi_lo << 16 |
				 (cl_ulong)t_self->ip_posid);
	GpuStoreBufferRemoveRow(gstate->gs_buffer,
							RelationGetDescr(frel),
							snapshot,
							old_index);

	/* insert new version of the row */
	GpuStoreBufferAppendRow(gstate->gs_buffer,
                            RelationGetDescr(frel),
							snapshot,
                            slot);
	return slot;
}

/*
 * gstoreExecForeignDelete
 */
static TupleTableSlot *
gstoreExecForeignDelete(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Relation		frel = rrinfo->ri_RelationDesc;
	Snapshot		snapshot = estate->es_snapshot;
	Datum			datum;
	bool			isnull;
	ItemPointer		t_self;
	size_t			old_index;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = GpuStoreBufferCreate(frel, snapshot);

	/* remove old version of the row */
	datum = ExecGetJunkAttribute(planSlot,
								 gstate->ctid_anum,
								 &isnull);
	if (isnull)
		elog(ERROR, "gstore_fdw: ctid is null");
	t_self = (ItemPointer)DatumGetPointer(datum);
	old_index = ((cl_ulong)t_self->ip_blkid.bi_hi << 32 |
				 (cl_ulong)t_self->ip_blkid.bi_lo << 16 |
				 (cl_ulong)t_self->ip_posid);
	GpuStoreBufferRemoveRow(gstate->gs_buffer,
							RelationGetDescr(frel),
							snapshot,
							old_index);
	return slot;
}

/*
 * gstoreEndForeignModify
 */
static void
gstoreEndForeignModify(EState *estate,
					   ResultRelInfo *rrinfo)
{
	//GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
}

/*
 * relation_is_gstore_fdw
 */
bool
relation_is_gstore_fdw(Oid table_oid)
{
	HeapTuple	tup;
	Oid			fserv_oid;
	Oid			fdw_oid;
	Oid			handler_oid;
	PGFunction	handler_fn;
	Datum		datum;
	char	   *prosrc;
	char	   *probin;
	bool		isnull;
	/* it should be foreign table, of course */
	if (get_rel_relkind(table_oid) != RELKIND_FOREIGN_TABLE)
		return false;
	/* pull OID of foreign-server */
	tup = SearchSysCache1(FOREIGNTABLEREL, ObjectIdGetDatum(table_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign table %u", table_oid);
	fserv_oid = ((Form_pg_foreign_table) GETSTRUCT(tup))->ftserver;
	ReleaseSysCache(tup);

	/* pull OID of foreign-data-wrapper */
	tup = SearchSysCache1(FOREIGNSERVEROID, ObjectIdGetDatum(fserv_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "foreign server with OID %u does not exist", fserv_oid);
	fdw_oid = ((Form_pg_foreign_server) GETSTRUCT(tup))->srvfdw;
	ReleaseSysCache(tup);

	/* pull OID of FDW handler function */
	tup = SearchSysCache1(FOREIGNDATAWRAPPEROID, ObjectIdGetDatum(fdw_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign-data wrapper %u",fdw_oid);
	handler_oid = ((Form_pg_foreign_data_wrapper) GETSTRUCT(tup))->fdwhandler;
	ReleaseSysCache(tup);
	/* pull library path & function name */
	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(handler_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", handler_oid);
	if (((Form_pg_proc) GETSTRUCT(tup))->prolang != ClanguageId)
		elog(ERROR, "FDW handler function is not written with C-language");

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "null prosrc for C function %u", handler_oid);
	prosrc = TextDatumGetCString(datum);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_probin, &isnull);
	if (isnull)
		elog(ERROR, "null probin for C function %u", handler_oid);
	probin = TextDatumGetCString(datum);
	ReleaseSysCache(tup);
	/* check whether function pointer is identical */
	handler_fn = load_external_function(probin, prosrc, true, NULL);
	if (handler_fn != pgstrom_gstore_fdw_handler)
		return false;
	/* OK, it is GpuStore foreign table */
	return true;
}

/*
 * gstore_fdw_table_options
 */
static void
__gstore_fdw_table_options(List *options,
						   int *p_pinning,
						   int *p_format)
{
	ListCell   *lc;
	int			pinning = -1;
	int			format = -1;

	foreach (lc, options)
	{
		DefElem	   *defel = lfirst(lc);

		if (strcmp(defel->defname, "pinning") == 0)
		{
			if (pinning >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"pinning\" option appears twice")));
			pinning = atoi(defGetString(defel));
			if (pinning < 0 || pinning >= numDevAttrs)
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("\"pinning\" on unavailable GPU device")));
		}
		else if (strcmp(defel->defname, "format") == 0)
		{
			char   *format_name;

			if (format >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"format\" option appears twice")));
			format_name = defGetString(defel);
			if (strcmp(format_name, "pgstrom") == 0 ||
				strcmp(format_name, "default") == 0)
				format = GSTORE_FDW_FORMAT__PGSTROM;
			else
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("gstore_fdw: format \"%s\" is unknown",
								format_name)));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_SYNTAX_ERROR),
					 errmsg("gstore_fdw: unknown option \"%s\"",
							defel->defname)));
		}
	}
	if (pinning < 0)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("gstore_fdw: No pinning GPU device"),
				 errhint("use 'pinning' option to specify GPU device")));

	/* put default if not specified */
	if (format < 0)
		format = GSTORE_FDW_FORMAT__PGSTROM;
	/* set results */
	if (p_pinning)
		*p_pinning = pinning;
	if (p_format)
		*p_format = format;
}

void
gstore_fdw_table_options(Oid gstore_oid, int *p_pinning, int *p_format)
{
	HeapTuple	tup;
	Datum		datum;
	bool		isnull;
	List	   *options = NIL;

	tup = SearchSysCache1(FOREIGNTABLEREL, ObjectIdGetDatum(gstore_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign table %u", gstore_oid);
	datum = SysCacheGetAttr(FOREIGNTABLEREL, tup,
							Anum_pg_foreign_table_ftoptions,
							&isnull);
	if (!isnull)
		options = untransformRelOptions(datum);
	__gstore_fdw_table_options(options, p_pinning, p_format);
	ReleaseSysCache(tup);
}

/*
 * gstore_fdw_column_options
 */
static void
__gstore_fdw_column_options(List *options, int *p_compression)
{
	ListCell   *lc;
	char	   *temp;
	int			compression = -1;

	foreach (lc, options)
	{
		DefElem	   *defel = lfirst(lc);

		if (strcmp(defel->defname, "compression") == 0)
		{
			if (compression >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"compression\" option appears twice")));
			temp = defGetString(defel);
			if (pg_strcasecmp(temp, "none") == 0)
				compression = GSTORE_COMPRESSION__NONE;
			else if (pg_strcasecmp(temp, "pglz") == 0)
				compression = GSTORE_COMPRESSION__PGLZ;
			else
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("unknown compression logic: %s", temp)));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_SYNTAX_ERROR),
					 errmsg("gstore_fdw: unknown option \"%s\"",
							defel->defname)));
		}
	}
	/* set default, if no valid options were supplied */
	if (compression < 0)
		compression = GSTORE_COMPRESSION__NONE;
	/* set results */
	if (p_compression)
		*p_compression = compression;
}

void
gstore_fdw_column_options(Oid gstore_oid, AttrNumber attnum,
						  int *p_compression)
{
	List	   *options = GetForeignColumnOptions(gstore_oid, attnum);

	__gstore_fdw_column_options(options, p_compression);
}

/*
 * pgstrom_gstore_fdw_validator
 */
Datum
pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS)
{
	List	   *options = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid			catalog = PG_GETARG_OID(1);

	switch (catalog)
	{
		case ForeignTableRelationId:
			__gstore_fdw_table_options(options, NULL, NULL);
			break;

		case AttributeRelationId:
			__gstore_fdw_column_options(options, NULL);
			break;

		case ForeignServerRelationId:
			if (options)
				elog(ERROR, "gstore_fdw: no options are supported on SERVER");
			break;

		case ForeignDataWrapperRelationId:
			if (options)
				elog(ERROR, "gstore_fdw: no options are supported on FOREIGN DATA WRAPPER");
			break;

		default:
			elog(ERROR, "gstore_fdw: no options are supported on catalog %s",
				 get_rel_name(catalog));
			break;
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_validator);

/*
 * pgstrom_gstore_fdw_handler
 */
Datum
pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS)
{
	FdwRoutine *routine = makeNode(FdwRoutine);

	/* functions for scanning foreign tables */
	routine->GetForeignRelSize	= gstoreGetForeignRelSize;
	routine->GetForeignPaths	= gstoreGetForeignPaths;
	routine->GetForeignPlan		= gstoreGetForeignPlan;
	routine->AddForeignUpdateTargets = gstoreAddForeignUpdateTargets;
	routine->BeginForeignScan	= gstoreBeginForeignScan;
	routine->IterateForeignScan	= gstoreIterateForeignScan;
	routine->ReScanForeignScan	= gstoreReScanForeignScan;
	routine->EndForeignScan		= gstoreEndForeignScan;

	/* functions for INSERT/UPDATE/DELETE foreign tables */

	routine->PlanForeignModify	= gstorePlanForeignModify;
	routine->BeginForeignModify	= gstoreBeginForeignModify;
	routine->ExecForeignInsert	= gstoreExecForeignInsert;
	routine->ExecForeignUpdate  = gstoreExecForeignUpdate;
	routine->ExecForeignDelete	= gstoreExecForeignDelete;
	routine->EndForeignModify	= gstoreEndForeignModify;

	PG_RETURN_POINTER(routine);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_handler);

/*
 * pgstrom_reggstore_in
 */
Datum
pgstrom_reggstore_in(PG_FUNCTION_ARGS)
{
	Datum	datum = regclassin(fcinfo);

	if (!relation_is_gstore_fdw(DatumGetObjectId(datum)))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						DatumGetObjectId(datum))));
	PG_RETURN_DATUM(datum);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_in);

/*
 * pgstrom_reggstore_out
 */
Datum
pgstrom_reggstore_out(PG_FUNCTION_ARGS)
{
	Oid		relid = PG_GETARG_OID(0);

	if (!relation_is_gstore_fdw(relid))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						relid)));
	return regclassout(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_out);

/*
 * pgstrom_reggstore_recv
 */
Datum
pgstrom_reggstore_recv(PG_FUNCTION_ARGS)
{
	/* exactly the same as oidrecv, so share code */
	Datum	datum = oidrecv(fcinfo);

	if (!relation_is_gstore_fdw(DatumGetObjectId(datum)))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						DatumGetObjectId(datum))));
	PG_RETURN_DATUM(datum);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_recv);

/*
 * pgstrom_reggstore_send
 */
Datum
pgstrom_reggstore_send(PG_FUNCTION_ARGS)
{
	Oid		relid = PG_GETARG_OID(0);

	if (!relation_is_gstore_fdw(relid))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						relid)));
	/* Exactly the same as oidsend, so share code */
	return oidsend(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_send);

/*
 * get_reggstore_type_oid
 */
Oid
get_reggstore_type_oid(void)
{
	if (!OidIsValid(reggstore_type_oid))
	{
		Oid		temp_oid;

		temp_oid = GetSysCacheOid2(TYPENAMENSP,
								   CStringGetDatum("reggstore"),
								   ObjectIdGetDatum(PG_PUBLIC_NAMESPACE));
		if (!OidIsValid(temp_oid) ||
			!type_is_reggstore(temp_oid))
			elog(ERROR, "type \"reggstore\" is not defined");
		reggstore_type_oid = temp_oid;
	}
	return reggstore_type_oid;
}

/*
 * reset_reggstore_type_oid
 */
static void
reset_reggstore_type_oid(Datum arg, int cacheid, uint32 hashvalue)
{
	reggstore_type_oid = InvalidOid;
}

/*
 * type_is_reggstore
 */
bool
type_is_reggstore(Oid type_oid)
{
	Oid			typinput;
	HeapTuple	tup;
	char	   *prosrc;
	char	   *probin;
	Datum		datum;
	bool		isnull;
	PGFunction	handler_fn;

	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typinput = ((Form_pg_type) GETSTRUCT(tup))->typinput;
	ReleaseSysCache(tup);

	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(typinput));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", typinput);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "null prosrc for C function %u", typinput);
	prosrc = TextDatumGetCString(datum);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_probin, &isnull);
	if (isnull)
		elog(ERROR, "null probin for C function %u", typinput);
	probin = TextDatumGetCString(datum);
	ReleaseSysCache(tup);

	/* check whether function pointer is identical */
	handler_fn = load_external_function(probin, prosrc, true, NULL);
	if (handler_fn != pgstrom_reggstore_in)
		return false;
	/* ok, it is reggstore type */
	return true;
}

/*
 * pgstrom_init_gstore_fdw
 */
void
pgstrom_init_gstore_fdw(void)
{
	/* invalidation of reggstore_oid variable */
	CacheRegisterSyscacheCallback(TYPEOID, reset_reggstore_type_oid, 0);
}
