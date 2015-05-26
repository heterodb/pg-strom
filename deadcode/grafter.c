/*
 * grafter.c
 *
 * Routines to modify plan tree once constructed.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "postgres.h"
#include "optimizer/planner.h"
#include "pg_strom.h"

static planner_hook_type	planner_hook_next;

static void
grafter_try_replace_recurse(PlannedStmt *pstmt, Plan **p_curr_plan)
{
	Plan	   *plan = *p_curr_plan;
	ListCell   *lc;

	Assert(plan != NULL);

	switch (nodeTag(plan))
	{
		case T_Agg:
			/*
			 * Try to inject GpuPreAgg plan if cost of the aggregate plan
			 * is enough expensive to justify preprocess by GPU.
			 */
			pgstrom_try_insert_gpupreagg(pstmt, (Agg *) plan);
			break;

		case T_SubqueryScan:
			{
				SubqueryScan   *subquery = (SubqueryScan *) plan;
				Plan		  **p_subplan = &subquery->subplan;
				grafter_try_replace_recurse(pstmt, p_subplan);
			}
			break;
		case T_ModifyTable:
			{
				ModifyTable *mtplan = (ModifyTable *) plan;

				foreach (lc, mtplan->plans)
				{
					Plan  **p_subplan = (Plan **) &lc->data.ptr_value;
					grafter_try_replace_recurse(pstmt, p_subplan);
				}
			}
			break;
		case T_Append:
			{
				Append *aplan = (Append *) plan;

				foreach (lc, aplan->appendplans)
				{
					Plan  **p_subplan = (Plan **) &lc->data.ptr_value;
					grafter_try_replace_recurse(pstmt, p_subplan);
				}
			}
			break;
		case T_MergeAppend:
			{
				MergeAppend *maplan = (MergeAppend *) plan;

				foreach (lc, maplan->mergeplans)
				{
					Plan  **p_subplan = (Plan **) &lc->data.ptr_value;
					grafter_try_replace_recurse(pstmt, p_subplan);
				}
			}
			break;
		case T_BitmapAnd:
			{
				BitmapAnd  *baplan = (BitmapAnd *) plan;

				foreach (lc, baplan->bitmapplans)
				{
					Plan  **p_subplan = (Plan **) &lc->data.ptr_value;
					grafter_try_replace_recurse(pstmt, p_subplan);
				}
			}
			break;
		case T_BitmapOr:
			{
				BitmapOr   *boplan = (BitmapOr *) plan;

				foreach (lc, boplan->bitmapplans)
				{
					Plan  **p_subplan = (Plan **) &lc->data.ptr_value;
					grafter_try_replace_recurse(pstmt, p_subplan);
				}
			}
			break;
		default:
			/* nothing to do, keep existgin one */
			break;
	}

	/* also walk down left and right child plan sub-tree, if any */
	if (plan->lefttree)
		grafter_try_replace_recurse(pstmt, &plan->lefttree);
	if (plan->righttree)
		grafter_try_replace_recurse(pstmt, &plan->righttree);

	switch (nodeTag(plan))
	{
		case T_Sort:
			/* Try to replace Sort node by GpuSort node if cost of
			 * the alternative plan is enough reasonable to replace.
			 */
			pgstrom_try_insert_gpusort(pstmt, p_curr_plan);
			break;

		default:
			/* nothing to do, keep existing one */
			break;
	}
}

static PlannedStmt *
pgstrom_grafter_entrypoint(Query *parse,
						   int cursorOptions,
						   ParamListInfo boundParams)
{
	PlannedStmt	*result;

	if (planner_hook_next)
		result = planner_hook_next(parse, cursorOptions, boundParams);
	else
		result = standard_planner(parse, cursorOptions, boundParams);

	if (pgstrom_enabled)
	{
		ListCell   *cell;

		Assert(result->planTree != NULL);
		grafter_try_replace_recurse(result, &result->planTree);

		foreach (cell, result->subplans)
		{
			Plan  **p_subplan = (Plan **) &cell->data.ptr_value;
			grafter_try_replace_recurse(result, p_subplan);
		}
	}
	return result;
}

void
pgstrom_init_grafter(void)
{
	/* hook registration */
	planner_hook_next = planner_hook;
	planner_hook = pgstrom_grafter_entrypoint;
}
