/*
 * plan.c
 *
 * routines to plan vectorized query execution.
 *
 * Copyright (C) 2011 - 2012 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "postgres.h"


FdwPlan *
PgBoostPlanForeignScan(Oid foreignTableOid,
					   PlannerInfo *root,
					   RelOptInfo *baserel)
{
	/*
	 * 1. check underlying table's attribute types.
	 * 2. check whether the qualifiers are vectorizable, or not
	 */


}

void
PgBoostExplainForeignScan(ForeignScanState *node,
						  ExplainState *es)
{
	ExplainPropertyText("Foreign File", "pg_boost (-o-;)", es);

}


List *
PgBoostPlanDescPack(PgBoostPlanDesc *plandesc)
{
	List   *result = NIL;

	result = lappend(result, makeDefElem("relationId",
					(Node *)makeInteger(plandesc->relationId)));
	return result;
}

PgBoostPlanDesc *
PgBoostPlanDescUnpack(List *packed)
{
	PgBoostPlanDesc *result;
	ListCell   *cell;

	result = palloc(sizeof(PgBoostPlanDesc));

	foreach (cell, packed)
	{
		DefElem	   *def = (DefElem *)cell;

		Assert(IsA(def, DefElem));

		if (strcmp(def->defname, "relationId") == 0)
			result->relationId = intVal(def->arg);
		else
			elog(ERROR, "pg_boost: unexpected packed PgBoostPlanDesc");
	}
	return result;
}

