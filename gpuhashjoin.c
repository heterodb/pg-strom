/*
 * gpuhashjoin.c
 *
 * Hash-Join acceleration by GPU processors
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"

#include "optimizer/paths.h"

#include "pg_strom.h"

/* static variables */
static add_join_path_hook_type	add_join_path_next;
static CustomPathMethods		gpuhashjoin_path_methods;
static CustomPlanMethods		gpuhashjoin_plan_methods;

typedef struct
{
	CustomPath		cpath;
} GpuHashJoinPath;

typedef struct
{
	CustomPlan		cplan;

} GpuHashJoin;

typedef struct
{
	CustomPlanState	cps;

} GpuHashJoinState;



static void
gpuhashjoin_add_join_path(PlannerInfo *root,
						  RelOptInfo *joinrel,
						  RelOptInfo *outerrel,
						  RelOptInfo *innerrel,
						  JoinType jointype,
						  SpecialJoinInfo *sjinfo,
						  List *restrictlist,
						  Relids param_source_rels,
						  Relids extra_lateral_rels)
{}

static CustomPlan *
gpuhashjoin_create_plan(PlannerInfo *root, CustomPath *best_path)
{
	return NULL;
}

static void
gpuhashjoin_textout_path(StringInfo str, Node *node)
{}

static void
gpuhashjoin_set_plan_ref(PlannerInfo *root,
						 CustomPlan *custom_plan,
						 int rtoffset)
{}

static void
gpuhashjoin_finalize_plan(PlannerInfo *root,
						  CustomPlan *custom_plan,
						  Bitmapset **paramids,
						  Bitmapset **valid_params,
						  Bitmapset **scan_params)
{}

static CustomPlanState *
gpuhashjoin_begin(CustomPlan *custom_plan, EState *estate, int eflags)
{
	return NULL;
}

static TupleTableSlot *
gpuhashjoin_exec(CustomPlanState *node)
{
	return NULL;
}

static Node *
gpuhashjoin_exec_multi(CustomPlanState *node)
{
	elog(ERROR, "not implemented yet");
	return NULL;
}

static void
gpuhashjoin_end(CustomPlanState *node)
{}

static void
gpuhashjoin_rescan(CustomPlanState *node)
{}

static void
gpuhashjoin_explain_rel(CustomPlanState *node, ExplainState *es)
{}

static void
gpuhashjoin_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{}

static Bitmapset *
gpuhashjoin_get_relids(CustomPlanState *node)
{
	return NULL;
}

static Node *
gpuhashjoin_get_special_var(CustomPlanState *node, Var *varnode)
{
	return NULL;
}

static void
gpuhashjoin_textout_plan(StringInfo str, const CustomPlan *node)
{}

static CustomPlan *
gpuhashjoin_copy_plan(const CustomPlan *from)
{
	GpuHashJoin		   *newnode = palloc(sizeof(GpuHashJoin));

	CopyCustomPlanCommon((Node *)from, (Node *)newnode);

	return &newnode->cplan;
}

void
pgstrom_init_gpuhashjoin(void)
{
	/* setup path methods */
	gpuhashjoin_path_methods.CustomName = "GpuHashJoin";
	gpuhashjoin_path_methods.CreateCustomPlan	= gpuhashjoin_create_plan;
	gpuhashjoin_path_methods.TextOutCustomPath	= gpuhashjoin_textout_path;

	/* setup plan methods */
	gpuhashjoin_plan_methods.CustomName = "GpuHashJoin";
	gpuhashjoin_plan_methods.SetCustomPlanRef	= gpuhashjoin_set_plan_ref;
	gpuhashjoin_plan_methods.SupportBackwardScan= NULL;
	gpuhashjoin_plan_methods.FinalizeCustomPlan	= gpuhashjoin_finalize_plan;
	gpuhashjoin_plan_methods.BeginCustomPlan	= gpuhashjoin_begin;
	gpuhashjoin_plan_methods.ExecCustomPlan		= gpuhashjoin_exec;
	gpuhashjoin_plan_methods.MultiExecCustomPlan= gpuhashjoin_exec_multi;
	gpuhashjoin_plan_methods.EndCustomPlan		= gpuhashjoin_end;
	gpuhashjoin_plan_methods.ReScanCustomPlan	= gpuhashjoin_rescan;
	gpuhashjoin_plan_methods.ExplainCustomPlanTargetRel
		= gpuhashjoin_explain_rel;
	gpuhashjoin_plan_methods.ExplainCustomPlan	= gpuhashjoin_explain;
	gpuhashjoin_plan_methods.GetRelidsCustomPlan= gpuhashjoin_get_relids;
	gpuhashjoin_plan_methods.GetSpecialCustomVar= gpuhashjoin_get_special_var;
	gpuhashjoin_plan_methods.TextOutCustomPlan	= gpuhashjoin_textout_plan;
	gpuhashjoin_plan_methods.CopyCustomPlan		= gpuhashjoin_copy_plan;

	/* hook registration */
	add_join_path_next = add_join_path_hook;
	add_join_path_hook = gpuhashjoin_add_join_path;
}
