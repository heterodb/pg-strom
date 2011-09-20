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
 *
 *
 *
 *
 */
Datum
pgboost_fdw_validator(PG_FUNCTION_ARGS)
{
	List   *options_list = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid		catalog = DatumGetObjectId(PG_GETARG_OID(1));
	List   *cell;

	foreach (cell, options_list)
	{
		/*
		 * Currently no options are supported
		 */
		ereport(ERROR,
				(errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
				 errmsg("invalid option \"%s\"", def->defname)));
	}
}
PG_FUNCTION_INFO_V1(pgboost_fdw_validator);

/*
 *
 *
 *
 *
 *
 */
Datum
pgboost_fdw_handler(PG_FUNCTION_ARGS)
{
	FdwRoutine *fdwroutine = makeNode(FdwRoutine);

	fdwroutine->PlanForeignScan = PgBoostPlanForeignScan;
	fdwroutine->ExplainForeignScan = PgBoostExplainForeignScan;
	fdwroutine->BeginForeignScan = PgBoostBeginForeignScan;
	fdwroutine->IterateForeignScan = PgBoostIterateForeignScan;
	fdwroutine->ReScanForeignScan = PgBoostReScanForeignScan;
	fdwroutine->EndForeignScan = PgBoostEndForeignScan;

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
