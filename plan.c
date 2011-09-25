/*
 * plan.c
 *
 * routines to plan vectorized query execution.
 *
 * Copyright (C) 2011 - 2012 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "postgres.h"
#include "pg_boost.h"

void
pgboost_extract_plan_info(scan_state_t *scan_state,
						  List *plan_info)
{
	ListCell   *cell;
	List	   *vquals = NIL;
	AttrNumber	min_attr = InvalidAttrNumber;
	AttrNumber	max_attr = InvalidAttrNumber;
	BitmapSet  *referenced = NULL;

	foreach (cell, plan_info)
	{
		DefElem	   *def = lfirst(cell);

		Assert(IsA(def, DefElem));

		if (strcmp(def->defname, "vqual") == 0)
		{
			vquals = lappend(vquals, def->arg);
		}
		else if (strcmp(def->defname, "min_attr") == 0)
		{
			min_attr = defGetInt64(def);
		}
		else if (strcmp(def->defname, "max_attr") == 0)
		{
			max_attr = defGetInt64(def);
		}
		else if (strcmp(def->defname, "referenced") == 0)
		{
			// XXX - defGetString does not handle T_BitString
			const char *temp = strVal(def->arg);
			const char *tail = temp + strlen(temp) - 1;
			int			code, index;

			for (index = 0; (code = tail[-index]) != 'b'; index++)
			{
				if (code == '1')
					referenced = bms_add_member(referenced, index);
				else if (code != '0')
					elog(ERROR, "pgboost: Bug? planInfo corrupted");
			}
			Assert(temp == tail);
		}
		else
		{
			elog(ERROR, "pgboost: Bug? unexpected DefElem: %s",
				 nodeToString(def));
		}
	}
	if (min_attr == InvalidAttrNumber ||
		max_attr == InvalidAttrNumber ||
		referenced == NULL)
		elog(ERROR, "pgboost: Bug? plan_info has corruption");

	scan_state->vquals = vquals;
	scan_state->min_attr = min_attr;
	scan_state->max_attr = max_attr;
	scan_state->referenced = referenced;
}

FdwPlan *
pgboost_plan_foreign_scan(Oid foreignTableOid,
						  PlannerInfo *root,
						  RelOptInfo *baserel)
{
	FdwPlan	   *fdwplan;
	List	   *planInfo = NIL;
	Node	   *node;
	ListCell   *cell;
	ListCell   *prev;
	char	   *temp;
	int			i, j;

	fdwplan = makeNode(FdwPlan);

	/*
	 * XXX - we need actual cost estimation
	 */
	fdwplan->startup_cost = 1.0;
	fdwplan->startup_cost = 2.0;

	/*
	 * We remove qualifiers being executable by vectorized operations
	 * from the list of restrictinfo. Remained qualifiers shall be
	 * executed sequentially at the upper level .
	 */
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (qual_is_vectorizable(rinfo->clause))
		{
			list_delete_cell(baserel->baserestrictinfo, cell, prev);

			node = (Node *)makeDefElem("vqual", (Node *)rinfo->clause);
			planInfo = lappend(planInfo, node);

			cell = prev;
		}
		else
			prev = cell;
	}

	/*
	 * Also put min_attr/max_attr
	 */
	node = (Node *)makeDefElem("min_attr",
							   (Node *)makeInteger(baserel->min_attr));
	planInfo = lappend(planInfo, node);

	node = (Node *)makeDefElem("max_attr",
							   (Node *)makeInteger(baserel->max_attr));
	planInfo = lappend(planInfo, node);

	/*
	 * Bitmap of referenced attributes
	 */
	temp = palloc(baserel->max_attr - baserel->min_attr + 2);
	temp[0] = 'b';
	for (i = baserel->max_attr, j=1; i >= baserel->min_attr; i--)
	{
		temp[j++] = (!baserel->attr_needed[i - baserel->min_attr] ? '0' : '1');
	}
	temp[j] = '\0';

	node = (Node *)makeDefElem("referenced",
							   (Node *)makeBitString(temp));
	privList = lappend(planInfo, node);

	/*
	 * Save the private list
	 */
	fdwplan->fdw_private = qualList;

	return fdwplan;
}

void
pgboost_explain_foreign_scan(ForeignScanState *node,
							 ExplainState *es)
{
	ExplainPropertyText("Foreign File", "pg_boost (-o-;)", es);
}
