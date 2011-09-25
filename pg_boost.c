/*
 * pg_boost.c
 *
 * Entrypoint of the pg_boost module
 *
 * Copyright 2011 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "postgres.h"
#include "access/reloptions.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "utils/guc.h"
#include "pg_boost.h"

PG_MODULE_MAGIC;

/*
 * pgboost_fdw_validator
 *
 * It validate options of foreign tables/columns.
 *
 * Right now, following options are supported:
 *  - relation_id <Oid>     : Oid of underlying relation
 *  - scan_chunk_size <int> : Size of vector_t in scan_chunk
 */
Datum
pgboost_fdw_validator(PG_FUNCTION_ARGS)
{
	List   *options_list = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid		catalog = DatumGetObjectId(PG_GETARG_OID(1));
	List   *cell;
	Oid		relation_id = InvalidOid;
	int		scan_chunk_size = 0;

	foreach (cell, options_list)
	{
		DefElem	   *def = (DefElem *) lfirst(cell);

		if (catalog == ForeignTableRelationId &&
			strcmp(def->defname, "relation_id") == 0)
		{
			if (OidIsValid(relation_id))
				goto duplicate_error;
			relation_id = defGetInt64(def);
		}
		else if (catalog == ForeignTableRelationId &&
				 strcmp(def->defname, "scan_chunk_size") == 0)
		{
			if (scan_chunk_size != 0)
				goto duplicate_error;
			scan_chunk_size = defGetInt64(def);
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
					 errmsg("invalid option \"%s\"", def->defname)));
		}
	}
	return;

duplicate_error:
	ereport(ERROR,
			(errcode(ERRCODE_SYNTAX_ERROR),
			 errmsg("conflicting or redundant options")));
}
PG_FUNCTION_INFO_V1(pgboost_fdw_validator);

/*
 * pgboost_fdw_handler
 *
 * Protocol handler of foreign data wrapper. It returns a set of function
 * pointers to be invoked from the (core) backend.
 */
Datum
pgboost_fdw_handler(PG_FUNCTION_ARGS)
{
	FdwRoutine *fdwroutine = makeNode(FdwRoutine);

	fdwroutine->PlanForeignScan = pgboost_plan_foreign_scan;
	fdwroutine->ExplainForeignScan = pgboost_explain_foreign_scan;
	fdwroutine->BeginForeignScan = pgboost_begin_foreign_scan;
	fdwroutine->IterateForeignScan = pgboost_iterate_foreign_scan;
	fdwroutine->ReScanForeignScan = pgboost_rescan_foreign_scan;
	fdwroutine->EndForeignScan = pgboost_end_foreign_scan;

	PG_RETURN_POINTER(fdwroutine);
}
PG_FUNCTION_INFO_V1(pgboost_fdw_handler);

/*
 * Entrypoint of the pg_boost module
 */
void
_PG_init(void)
{
}
