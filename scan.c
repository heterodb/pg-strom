/*
 * scan.c
 *
 * Routines to scan column based data store with stream processing
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "pg_strom.h"

void
pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags)
{
}

TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss)
{
	return NULL;
}

void
pgboost_rescan_foreign_scan(ForeignScanState *fss)
{
}

void
pgboost_end_foreign_scan(ForeignScanState *fss)
{
}
