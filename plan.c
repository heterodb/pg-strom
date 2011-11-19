/*
 * plan.c
 *
 * Routines to plan streamed query execution.
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

FdwPlan *
pgstrom_plan_foreign_scan(Oid foreignTblOid,
						  PlannerInfo *root,
						  RelOptInfo *baserel)
{
	FdwPlan	   *fdwplan;

	fdwplan = makeNode(FdwPlan);

	/*
	 * XXX - we need actual cost estimation
	 */
	fdwplan->startup_cost = 1.0;
	fdwplan->startup_cost = 2.0;

	return fdwplan;
}

void
pgstrom_explain_foreign_scan(ForeignScanState *node,
							 ExplainState *es)
{
	ExplainPropertyText("Foreign File", "pg_strom (-o-;)", es);
}
