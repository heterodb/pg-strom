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
	List	   *private = NIL;
	DefElem	   *defel;
	AttrNumber	i;

	/*
	 * Save the referenced columns
	 */
	for (i = baserel->min_attr; i <= baserel->max_attr; i++)
	{
		if (bms_is_empty(baserel->attr_needed[i - baserel->min_attr]))
			continue;

		defel = makeDefElem("cols_needed",
							(Node *) makeInteger(i - baserel->min_attr));
		private = lappend(private, (Node *) defel);
	}

	/*
	 * Set up FdwPlan
	 */
	fdwplan = makeNode(FdwPlan);
	fdwplan->startup_cost = 1.0;
	fdwplan->startup_cost = 2.0;
	fdwplan->fdw_private = private;

	return fdwplan;
}

void
pgstrom_explain_foreign_scan(ForeignScanState *node,
							 ExplainState *es)
{
	Relation	rel = node->ss.ss_currentRelation;

	ExplainPropertyText("Stream Relation", RelationGetRelationName(rel), es);
}
