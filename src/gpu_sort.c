/*
 * gpu_sort.c
 *
 * GPU-Sorting on top of GPU-Projectin / GPU-PreAgg
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* static variables */
static bool			pgstrom_enable_gpusort = true;

/*
 * try_add_sorted_groupby_path
 */
void
try_add_sorted_groupby_path(PlannerInfo *root,
							RelOptInfo *group_rel,
							Path *sub_path,
							CustomPath *preagg_path)
{
	if (!pgstrom_enable_gpusort)
		return;
#if 0
	elog(INFO, "query_pathkeys %p -> %s", root->query_pathkeys, nodeToString(root->query_pathkeys));
	elog(INFO, "window_pathkeys %p -> %s", root->window_pathkeys, nodeToString(root->window_pathkeys));
	elog(INFO, "distinct_pathkeys %p -> %s", root->distinct_pathkeys, nodeToString(root->distinct_pathkeys));
	elog(INFO, "sort_pathkeys %p -> %s", root->sort_pathkeys, nodeToString(root->sort_pathkeys));
	elog(INFO, "targetlist -> %s", nodeToString(sub_path->pathtarget));
#endif



}

/*
 * try_add_sorted_gpujoin_path
 */
void
try_add_sorted_gpujoin_path(PlannerInfo *root,
							RelOptInfo *join_rel,
							CustomPath *join_path,
							bool be_parallel)
{
	if (!pgstrom_enable_gpusort)
		return;
	
}

/*
 * pgstrom_init_gpu_sort
 */
void
pgstrom_init_gpu_sort(void)
{
	/* turn on/off GPU-Sort */
	DefineCustomBoolVariable("pg_strom.enable_gpusort",
							 "Enables to use GPU-Sort on top of GPU-Projection/PreAgg",
							 NULL,
							 &pgstrom_enable_gpusort,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}
