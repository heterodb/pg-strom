/*
 * pg_strom.c
 *
 * Entrypoint of the pg_strom module
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "catalog/pg_type.h"
#include "foreign/fdwapi.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * Local declarations
 */
void	_PG_init(void);

FdwRoutine	pgstromFdwHandlerData = {
	.type				= T_FdwRoutine,
	.PlanForeignScan	= pgstrom_plan_foreign_scan,
	.ExplainForeignScan	= pgstrom_explain_foreign_scan,
	.BeginForeignScan	= pgstrom_begin_foreign_scan,
	.IterateForeignScan	= pgstrom_iterate_foreign_scan,
	.ReScanForeignScan	= pgstrom_rescan_foreign_scan,
	.EndForeignScan		= pgstrom_end_foreign_scan,
};

/*
 * pgstrom_fdw_handler
 *
 * FDW Handler function of pg_strom
 */
Datum
pgstrom_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&pgstromFdwHandlerData);
}
PG_FUNCTION_INFO_V1(pgstrom_fdw_handler);

/****/
Datum
pgstrom_fdw_validator(PG_FUNCTION_ARGS)
{
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_fdw_validator);

/*
 * pgstrom_debug_info
 *
 * shows user's debug information
 */
Datum
pgstrom_debug_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	   *fncxt;
	MemoryContext		oldcxt;
	ListCell   *cell;
	DefElem	   *defel;
	Datum		values[2];
	bool		isnull[2];
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc	tupdesc;
		List	   *debug_info_list = NIL;

		fncxt = SRF_FIRSTCALL_INIT();

		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(2, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "key",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "value",
						   TEXTOID, -1, 0);

		debug_info_list = pgstrom_scan_debug_info(debug_info_list);

		fncxt->user_fctx = (void *) debug_info_list;
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();
	cell = list_head((List *)fncxt->user_fctx);
	if (!cell)
		SRF_RETURN_DONE(fncxt);

	defel = lfirst(cell);
	Assert(IsA(defel, DefElem));

	memset(isnull, false, sizeof(isnull));
	values[0] = CStringGetTextDatum(defel->defname);
	values[1] = CStringGetTextDatum(strVal(defel->arg));

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	fncxt->user_fctx = list_delete_ptr((List *)fncxt->user_fctx,
									   lfirst(cell));
	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_debug_info);

/*
 * Entrypoint of the pg_strom module
 */
void
_PG_init(void)
{
	/*
	 * pg_strom has to be loaded using shared_preload_libraries setting.
	 */
	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
		errmsg("pg_strom must be loaded via shared_preload_libraries")));

	/* Register Hooks of PostgreSQL */
	pgstrom_utilcmds_init();

	/* Collect properties of GPU devices */
	pgstrom_devinfo_init();

	/* Initialize stuff related to scan.c */
	pgstrom_scan_init();

	/* Initialize stuff related to just-in-time compile */
	pgstrom_nvcc_init();
}
