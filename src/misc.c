/*
 * misc.c
 *
 * miscellaneous and uncategorized routines but usefull for multiple subsystems
 * of PG-Strom.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#include "pg_strom.h"

/*
 * make_flat_ands_expr - similar to make_ands_explicit but it pulls up
 * underlying and-clause
 */
Expr *
make_flat_ands_explicit(List *andclauses)
{
	List	   *args = NIL;
	ListCell   *lc;

	if (andclauses == NIL)
		return (Expr *) makeBoolConst(true, false);
	else if (list_length(andclauses) == 1)
		return (Expr *) linitial(andclauses);

	foreach (lc, andclauses)
	{
		Expr   *expr = lfirst(lc);

		Assert(exprType((Node *)expr) == BOOLOID);
		if (IsA(expr, BoolExpr) &&
			((BoolExpr *)expr)->boolop == AND_EXPR)
			args = list_concat(args, ((BoolExpr *) expr)->args);
		else
			args = lappend(args, expr);
	}
	Assert(list_length(args) > 1);
	return make_andclause(args);
}

/*
 * __find_appinfos_by_relids - almost equivalent to find_appinfos_by_relids
 * that is added at PG11, but ignores relations that are not partition leafs.
 */
AppendRelInfo **
__find_appinfos_by_relids(PlannerInfo *root, Relids relids, int *nappinfos)
{
	AppendRelInfo **appinfos;
	ListCell   *lc;
	int			nrooms = bms_num_members(relids);
	int			nitems = 0;

	appinfos = palloc0(sizeof(AppendRelInfo *) * nrooms);
	foreach (lc, root->append_rel_list)
	{
		AppendRelInfo *apinfo = lfirst(lc);

		if (bms_is_member(apinfo->child_relid, relids))
			appinfos[nitems++] = apinfo;
	}
	Assert(nitems <= nrooms);
	*nappinfos = nitems;

	return appinfos;
}

#if PG_VERSION_NUM < 100000
/*
 * compute_parallel_worker at optimizer/path/allpaths.c
 * was newly added at PG10.x
 *
 * Compute the number of parallel workers that should be used to scan a
 * relation.  We compute the parallel workers based on the size of the heap to
 * be scanned and the size of the index to be scanned, then choose a minimum
 * of those.
 *
 * "heap_pages" is the number of pages from the table that we expect to scan,
 *  or -1 if we don't expect to scan any.
 *
 * "index_pages" is the number of pages from the index that we expect to scan,
 *  or -1 if we don't expect to scan any.
 */
int
__compute_parallel_worker(RelOptInfo *rel,
						  double heap_pages,
						  double index_pages)
{
	int			parallel_workers = 0;

	/*
	 * If the user has set the parallel_workers reloption, use that; otherwise
	 * select a default number of workers.
	 */
	if (rel->rel_parallel_workers != -1)
		parallel_workers = rel->rel_parallel_workers;
	else
	{
		/*
		 * If the number of pages being scanned is insufficient to justify a
		 * parallel scan, just return zero ... unless it's an inheritance
		 * child. In that case, we want to generate a parallel path here
		 * anyway.  It might not be worthwhile just for this relation, but
		 * when combined with all of its inheritance siblings it may well pay
		 * off.
		 */
		if (rel->reloptkind == RELOPT_BASEREL &&
			(heap_pages >= 0 && heap_pages < min_parallel_relation_size))
			return 0;

		if (heap_pages >= 0)
		{
			int			heap_parallel_threshold;
			int			heap_parallel_workers = 1;

			/*
			 * Select the number of workers based on the log of the size of
			 * the relation.  This probably needs to be a good deal more
			 * sophisticated, but we need something here for now.  Note that
			 * the upper limit of the min_parallel_relation_size GUC is
			 * chosen to prevent overflow here.
			 */
			heap_parallel_threshold = Max(min_parallel_relation_size, 1);
			while (heap_pages >= (BlockNumber) (heap_parallel_threshold * 3))
			{
				heap_parallel_workers++;
				heap_parallel_threshold *= 3;
				if (heap_parallel_threshold > INT_MAX / 3)
					break;		/* avoid overflow */
			}

			parallel_workers = heap_parallel_workers;
		}
		/*
		 * NOTE: PG9.6 does not pay attention for # of index pages
		 * for decision of parallel execution.
		 */
	}

	/*
	 * In no case use more than max_parallel_workers_per_gather workers.
	 */
	parallel_workers = Min(parallel_workers, max_parallel_workers_per_gather);

	return parallel_workers;
}
#endif		/* < PG10 */

/*
 * get_parallel_divisor - Estimate the fraction of the work that each worker
 * will do given the number of workers budgeted for the path.
 */
double
get_parallel_divisor(Path *path)
{
	double		parallel_divisor = path->parallel_workers;

#if PG_VERSION_NUM >= 110000
	if (parallel_leader_participation)
#endif
	{
		double	leader_contribution;

		leader_contribution = 1.0 - (0.3 * path->parallel_workers);
		if (leader_contribution > 0)
			parallel_divisor += leader_contribution;
	}
	return parallel_divisor;
}

/*
 * Usefulll wrapper routines like lsyscache.c
 */
#if PG_VERSION_NUM < 110000
char
get_func_prokind(Oid funcid)
{
	HeapTuple	tup;
	Form_pg_proc procForm;
	char		prokind;

	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(funcid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", funcid);
	procForm = (Form_pg_proc) GETSTRUCT(tup);
	if (procForm->proisagg)
	{
		Assert(!procForm->proiswindow);
		prokind = PROKIND_AGGREGATE;
	}
	else if (procForm->proiswindow)
	{
		Assert(!procForm->proisagg);
		prokind = PROKIND_WINDOW;
	}
	else
	{
		prokind = PROKIND_FUNCTION;
	}
	ReleaseSysCache(tup);

	return prokind;
}
#endif		/* <PG11 */

int
get_relnatts(Oid relid)
{
	HeapTuple	tup;
	int			relnatts = -1;

	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(relid));
	if (HeapTupleIsValid(tup))
	{
        Form_pg_class reltup = (Form_pg_class) GETSTRUCT(tup);

        relnatts = reltup->relnatts;
        ReleaseSysCache(tup);
	}
	return relnatts;
}

/*
 * bms_to_cstring - human readable Bitmapset
 */
char *
bms_to_cstring(Bitmapset *x)
{
	StringInfoData buf;
	int			i;

	initStringInfo(&buf);
	appendStringInfo(&buf, "{");
	for (i=0; i < x->nwords; i++)
		appendStringInfo(&buf, "%s %08x", i==0 ? "" : ",", x->words[i]);
	appendStringInfo(&buf, " }");

	return buf.data;
}

/*
 * pathnode_tree_walker
 */
static bool
pathnode_tree_walker(Path *node,
					 bool (*walker)(),
					 void *context)
{
	ListCell   *lc;

	if (!node)
		return false;

	check_stack_depth();
	switch (nodeTag(node))
	{
		case T_Path:
		case T_IndexPath:
		case T_BitmapHeapPath:
		case T_BitmapAndPath:
		case T_BitmapOrPath:
		case T_TidPath:
		case T_ResultPath:
		case T_MinMaxAggPath:
			/* primitive path nodes */
			break;
		case T_SubqueryScanPath:
			if (walker(((SubqueryScanPath *)node)->subpath, context))
				return true;
			break;
		case T_ForeignPath:
			if (walker(((ForeignPath *)node)->fdw_outerpath, context))
				return true;
			break;
		case T_CustomPath:
			foreach (lc, ((CustomPath *)node)->custom_paths)
			{
				if (walker((Path *)lfirst(lc), context))
					return true;
			}
			break;
		case T_NestPath:
		case T_MergePath:
		case T_HashPath:
			if (walker(((JoinPath *)node)->outerjoinpath, context))
				return true;
			if (walker(((JoinPath *)node)->innerjoinpath, context))
				return true;
			break;
		case T_AppendPath:
			foreach (lc, ((AppendPath *)node)->subpaths)
			{
				if (walker((Path *)lfirst(lc), context))
					return true;
			}
			break;
		case T_MergeAppendPath:
			foreach (lc, ((MergeAppendPath *)node)->subpaths)
			{
				if (walker((Path *)lfirst(lc), context))
					return true;
			}
			break;
		case T_MaterialPath:
			if (walker(((MaterialPath *)node)->subpath, context))
				return true;
			break;
		case T_UniquePath:
			if (walker(((UniquePath *)node)->subpath, context))
				return true;
			break;
		case T_GatherPath:
			if (walker(((GatherPath *)node)->subpath, context))
				return true;
			break;
#if PG_VERSION_NUM >= 100000
		case T_GatherMergePath:
			if (walker(((GatherMergePath *)node)->subpath, context))
				return true;
			break;
#endif		/* >= PG10 */
		case T_ProjectionPath:
			if (walker(((ProjectionPath *)node)->subpath, context))
				return true;
			break;
#if PG_VERSION_NUM >= 100000
		case T_ProjectSetPath:
			if (walker(((ProjectSetPath *)node)->subpath, context))
				return true;
			break;
#endif		/* >= PG10 */
		case T_SortPath:
			if (walker(((SortPath *)node)->subpath, context))
				return true;
			break;
		case T_GroupPath:
			if (walker(((GroupPath *)node)->subpath, context))
				return true;
			break;
		case T_UpperUniquePath:
			if (walker(((UpperUniquePath *)node)->subpath, context))
				return true;
			break;
		case T_AggPath:
			if (walker(((AggPath *)node)->subpath, context))
				return true;
			break;
		case T_GroupingSetsPath:
			if (walker(((GroupingSetsPath *)node)->subpath, context))
				return true;
			break;
		case T_WindowAggPath:
			if (walker(((WindowAggPath *)node)->subpath, context))
				return true;
			break;
		case T_SetOpPath:
			if (walker(((SetOpPath *)node)->subpath, context))
				return true;
			break;
		case T_RecursiveUnionPath:
			if (walker(((RecursiveUnionPath *)node)->leftpath, context))
				return true;
			if (walker(((RecursiveUnionPath *)node)->rightpath, context))
				return true;
			break;
		case T_LockRowsPath:
			if (walker(((LockRowsPath *)node)->subpath, context))
				return true;
			break;
		case T_ModifyTablePath:
			foreach (lc, ((ModifyTablePath *)node)->subpaths)
			{
				if (walker((Path *)lfirst(lc), context))
					return true;
			}
			break;
		case T_LimitPath:
			if (walker(((LimitPath *)node)->subpath, context))
				return true;
			break;
		default:
			elog(ERROR, "unrecognized path-node type: %d",
				 (int) nodeTag(node));
			break;
	}
	return false;
}

static bool
__pathtree_has_gpupath(Path *node, void *context)
{
	if (!node)
		return false;
	if (pgstrom_path_is_gpuscan(node) ||
		pgstrom_path_is_gpujoin(node) ||
		pgstrom_path_is_gpupreagg(node))
		return true;
	return pathnode_tree_walker(node, __pathtree_has_gpupath, context);
}

bool
pathtree_has_gpupath(Path *node)
{
	return __pathtree_has_gpupath(node, NULL);
}

/*
 * pgstrom_copy_pathnode
 *
 * add_path() / add_partial_path() may reject path-nodes that are already
 * registered and referenced by upper path nodes, like GpuJoin.
 * To avoid the problem, we use copy of path-nodes that are (potentially)
 * released by another ones. However, it is not a full-copy. add_path() will
 * never release fields of individual path-nodes, so this function tries
 * to make a copy of path-node itself and child path-nodes only.
 */
Path *
pgstrom_copy_pathnode(const Path *pathnode)
{
	switch (nodeTag(pathnode))
	{
		case T_Path:
			return pmemdup(pathnode, sizeof(Path));
		case T_IndexPath:
			return pmemdup(pathnode, sizeof(IndexPath));
		case T_BitmapHeapPath:
			return pmemdup(pathnode, sizeof(BitmapHeapPath));
		case T_BitmapAndPath:
			return pmemdup(pathnode, sizeof(BitmapAndPath));
		case T_BitmapOrPath:
			return pmemdup(pathnode, sizeof(BitmapOrPath));
		case T_TidPath:
			return pmemdup(pathnode, sizeof(TidPath));
		case T_SubqueryScanPath:
			{
				SubqueryScanPath *a = (SubqueryScanPath *)pathnode;
				SubqueryScanPath *b = pmemdup(a, sizeof(SubqueryScanPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ForeignPath:
			{
				ForeignPath	   *a = (ForeignPath *)pathnode;
				ForeignPath	   *b = pmemdup(a, sizeof(ForeignPath));
				b->fdw_outerpath = pgstrom_copy_pathnode(a->fdw_outerpath);
				return &b->path;
			}
		case T_CustomPath:
			if (pgstrom_path_is_gpuscan(pathnode))
				return pgstrom_copy_gpuscan_path(pathnode);
			else if (pgstrom_path_is_gpujoin(pathnode))
				return pgstrom_copy_gpujoin_path(pathnode);
			else if (pgstrom_path_is_gpupreagg(pathnode))
				return pgstrom_copy_gpupreagg_path(pathnode);
			else
			{
				CustomPath	   *a = (CustomPath *)pathnode;
				CustomPath	   *b = pmemdup(a, sizeof(CustomPath));
				List		   *subpaths = NIL;
				ListCell	   *lc;
				foreach (lc, a->custom_paths)
					subpaths = lappend(subpaths,
									   pgstrom_copy_pathnode(lfirst(lc)));
				b->custom_paths = subpaths;
				return &b->path;
			}
		case T_NestPath:
			{
				NestPath   *a = (NestPath *)pathnode;
				NestPath   *b = pmemdup(a, sizeof(NestPath));
				b->outerjoinpath = pgstrom_copy_pathnode(a->outerjoinpath);
				b->innerjoinpath = pgstrom_copy_pathnode(a->innerjoinpath);
				return &b->path;
			}
		case T_MergePath:
			{
				MergePath  *a = (MergePath *)pathnode;
				MergePath  *b = pmemdup(a, sizeof(MergePath));
				b->jpath.outerjoinpath =
					pgstrom_copy_pathnode(a->jpath.outerjoinpath);
				b->jpath.innerjoinpath =
					pgstrom_copy_pathnode(a->jpath.innerjoinpath);
				return &b->jpath.path;
			}
		case T_HashPath:
			{
				HashPath   *a = (HashPath *)pathnode;
				HashPath   *b = pmemdup(a, sizeof(HashPath));
				b->jpath.outerjoinpath =
					pgstrom_copy_pathnode(a->jpath.outerjoinpath);
				b->jpath.innerjoinpath =
					pgstrom_copy_pathnode(a->jpath.innerjoinpath);
				return &b->jpath.path;
			}
		case T_AppendPath:
			{
				AppendPath *a = (AppendPath *)pathnode;
				AppendPath *b = pmemdup(a, sizeof(AppendPath));
				List	   *subpaths = NIL;
				ListCell   *lc;

				foreach (lc, a->subpaths)
					subpaths = lappend(subpaths,
									   pgstrom_copy_pathnode(lfirst(lc)));
				b->subpaths = subpaths;
				return &b->path;
			}
		case T_MergeAppendPath:
			{
				MergeAppendPath *a = (MergeAppendPath *)pathnode;
				MergeAppendPath *b = pmemdup(a, sizeof(MergeAppendPath));
				List	   *subpaths = NIL;
				ListCell   *lc;

				foreach (lc, a->subpaths)
					subpaths = lappend(subpaths,
									   pgstrom_copy_pathnode(lfirst(lc)));
				b->subpaths = subpaths;
				return &b->path;
			}
		case T_ResultPath:
			return pmemdup(pathnode, sizeof(ResultPath));
		case T_MaterialPath:
			{
				MaterialPath   *a = (MaterialPath *)pathnode;
				MaterialPath   *b = pmemdup(a, sizeof(MaterialPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_UniquePath:
			{
				UniquePath	   *a = (UniquePath *)pathnode;
				UniquePath	   *b = pmemdup(a, sizeof(UniquePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_GatherPath:
			{
				GatherPath	   *a = (GatherPath *)pathnode;
				GatherPath	   *b = pmemdup(a, sizeof(GatherPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_GatherMergePath:
			{
				GatherMergePath *a = (GatherMergePath *)pathnode;
				GatherMergePath *b = pmemdup(a, sizeof(GatherMergePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ProjectionPath:
			{
				ProjectionPath *a = (ProjectionPath *)pathnode;
				ProjectionPath *b = pmemdup(a, sizeof(ProjectionPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ProjectSetPath:
			{
				ProjectSetPath *a = (ProjectSetPath *)pathnode;
				ProjectSetPath *b = pmemdup(a, sizeof(ProjectSetPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_SortPath:
			{
				SortPath	   *a = (SortPath *)pathnode;
				SortPath	   *b = pmemdup(a, sizeof(SortPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_GroupPath:
			{
				GroupPath	   *a = (GroupPath *)pathnode;
				GroupPath	   *b = pmemdup(a, sizeof(GroupPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_UpperUniquePath:
			{
				UpperUniquePath *a = (UpperUniquePath *)pathnode;
				UpperUniquePath *b = pmemdup(a, sizeof(UpperUniquePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_AggPath:
			{
				AggPath		   *a = (AggPath *)pathnode;
				AggPath		   *b = pmemdup(a, sizeof(AggPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_GroupingSetsPath:
			{
				GroupingSetsPath *a = (GroupingSetsPath *)pathnode;
				GroupingSetsPath *b = pmemdup(a, sizeof(GroupingSetsPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
                return &b->path;
			}
		case T_MinMaxAggPath:
			return pmemdup(pathnode, sizeof(MinMaxAggPath));
		case T_WindowAggPath:
			{
				WindowAggPath  *a = (WindowAggPath *)pathnode;
				WindowAggPath  *b = pmemdup(a, sizeof(WindowAggPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_SetOpPath:
			{
				SetOpPath	   *a = (SetOpPath *)pathnode;
				SetOpPath	   *b = pmemdup(a, sizeof(SetOpPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_RecursiveUnionPath:
			{
				RecursiveUnionPath *a = (RecursiveUnionPath *)pathnode;
				RecursiveUnionPath *b = pmemdup(a, sizeof(RecursiveUnionPath));
				b->leftpath = pgstrom_copy_pathnode(a->leftpath);
				b->rightpath = pgstrom_copy_pathnode(a->rightpath);
				return &b->path;
			}
		case T_LockRowsPath:
			{
				LockRowsPath   *a = (LockRowsPath *)pathnode;
				LockRowsPath   *b = pmemdup(a, sizeof(LockRowsPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ModifyTablePath:
			{
				ModifyTablePath *a = (ModifyTablePath *)pathnode;
				ModifyTablePath *b = pmemdup(a, sizeof(ModifyTablePath));
				List	   *subpaths = NIL;
				ListCell   *lc;
				foreach (lc, a->subpaths)
					subpaths = lappend(subpaths,
									   pgstrom_copy_pathnode(lfirst(lc)));
				b->subpaths = subpaths;
				return &b->path;
			}
		case T_LimitPath:
			{
				LimitPath  *a = (LimitPath *)pathnode;
				LimitPath  *b = pmemdup(a, sizeof(LimitPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		default:
			elog(ERROR, "Bug? unknown path-node: %s", nodeToString(pathnode));
	}
	return NULL;
}

/*
 * errorText - string form of the error code
 */
const char *
errorText(int errcode)
{
	static __thread char buffer[160];
	const char *error_name;
	const char *error_desc;

	if (cuGetErrorName(errcode, &error_name) == CUDA_SUCCESS &&
		cuGetErrorString(errcode, &error_desc) == CUDA_SUCCESS)
	{
		snprintf(buffer, sizeof(buffer), "%s - %s",
				 error_name, error_desc);
	}
	else
	{
		snprintf(buffer, sizeof(buffer), "%d - unknown", errcode);
	}
	return buffer;
}

/*
 * ----------------------------------------------------------------
 *
 * SQL functions to support regression test
 *
 * ----------------------------------------------------------------
 */
Datum pgstrom_random_int(PG_FUNCTION_ARGS);
Datum pgstrom_random_float(PG_FUNCTION_ARGS);
Datum pgstrom_random_date(PG_FUNCTION_ARGS);
Datum pgstrom_random_time(PG_FUNCTION_ARGS);
Datum pgstrom_random_timetz(PG_FUNCTION_ARGS);
Datum pgstrom_random_timestamp(PG_FUNCTION_ARGS);
Datum pgstrom_random_timestamptz(PG_FUNCTION_ARGS);
Datum pgstrom_random_interval(PG_FUNCTION_ARGS);
Datum pgstrom_random_macaddr(PG_FUNCTION_ARGS);
Datum pgstrom_random_inet(PG_FUNCTION_ARGS);
Datum pgstrom_random_text(PG_FUNCTION_ARGS);
Datum pgstrom_random_text_length(PG_FUNCTION_ARGS);
Datum pgstrom_random_int4range(PG_FUNCTION_ARGS);
Datum pgstrom_random_int8range(PG_FUNCTION_ARGS);
Datum pgstrom_random_tsrange(PG_FUNCTION_ARGS);
Datum pgstrom_random_tstzrange(PG_FUNCTION_ARGS);
Datum pgstrom_random_daterange(PG_FUNCTION_ARGS);
Datum pgstrom_abort_if(PG_FUNCTION_ARGS);

static inline bool
generate_null(double ratio)
{
	if (ratio <= 0.0)
		return false;
	if (100.0 * drand48() < ratio)
		return true;
	return false;
}

Datum
pgstrom_random_int(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int64		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT64(1) : 0);
	int64		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT64(2) : INT_MAX);
	cl_ulong	v;

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_INT64(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_INT64(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_int);

Datum
pgstrom_random_float(PG_FUNCTION_ARGS)
{
	float8	ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	float8	lower = (!PG_ARGISNULL(1) ? PG_GETARG_FLOAT8(1) : 0.0);
	float8	upper = (!PG_ARGISNULL(2) ? PG_GETARG_FLOAT8(2) : 1.0);

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_FLOAT8(lower);

	PG_RETURN_FLOAT8((upper - lower) * drand48() + lower);
}
PG_FUNCTION_INFO_V1(pgstrom_random_float);

Datum
pgstrom_random_date(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	DateADT		lower;
	DateADT		upper;
	cl_ulong	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_DATEADT(1);
	else
		lower = date2j(2015, 1, 1) - POSTGRES_EPOCH_JDATE;
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_DATEADT(2);
	else
		upper = date2j(2025, 12, 31) - POSTGRES_EPOCH_JDATE;

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_DATEADT(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_DATEADT(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_date);

Datum
pgstrom_random_time(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	TimeADT		lower = 0;
	TimeADT		upper = HOURS_PER_DAY * USECS_PER_HOUR - 1;
	cl_ulong	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMEADT(1);
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMEADT(2);
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_TIMEADT(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_TIMEADT(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_time);

Datum
pgstrom_random_timetz(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	TimeADT		lower = 0;
	TimeADT		upper = HOURS_PER_DAY * USECS_PER_HOUR - 1;
	TimeTzADT  *temp;
	cl_ulong	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMEADT(1);
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMEADT(2);
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	temp = palloc(sizeof(TimeTzADT));
	temp->zone = (random() % 23 - 11) * USECS_PER_HOUR;
	if (upper == lower)
		temp->time = lower;
	else
	{
		v = ((cl_ulong)random() << 31) | (cl_ulong)random();
		temp->time = lower + v % (upper - lower);
	}
	PG_RETURN_TIMETZADT_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_timetz);

Datum
pgstrom_random_timestamp(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	cl_ulong	v;
	struct pg_tm tm;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_TIMEADT(lower);
	v = ((cl_ulong)random() << 31) | (cl_ulong)random();

	PG_RETURN_TIMESTAMP(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_timestamp);

Datum
pgstrom_random_macaddr(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	macaddr	   *temp;
	cl_ulong	lower;
	cl_ulong	upper;
	cl_ulong	v, x;

	if (PG_ARGISNULL(1))
		lower = 0xabcd00000000UL;
	else
	{
		temp = PG_GETARG_MACADDR_P(1);
		lower = (((cl_ulong)temp->a << 40) | ((cl_ulong)temp->b << 32) |
				 ((cl_ulong)temp->c << 24) | ((cl_ulong)temp->d << 16) |
				 ((cl_ulong)temp->e <<  8) | ((cl_ulong)temp->f));
	}

	if (PG_ARGISNULL(2))
		upper = 0xabcdffffffffUL;
	else
	{
		temp = PG_GETARG_MACADDR_P(2);
		upper = (((cl_ulong)temp->a << 40) | ((cl_ulong)temp->b << 32) |
				 ((cl_ulong)temp->c << 24) | ((cl_ulong)temp->d << 16) |
				 ((cl_ulong)temp->e <<  8) | ((cl_ulong)temp->f));
	}

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		x = lower;
	else
	{
		v = ((cl_ulong)random() << 31) | (cl_ulong)random();
		x = lower + v % (upper - lower);
	}
	temp = palloc(sizeof(macaddr));
	temp->a = (x >> 40) & 0x00ff;
	temp->b = (x >> 32) & 0x00ff;
	temp->c = (x >> 24) & 0x00ff;
	temp->d = (x >> 16) & 0x00ff;
	temp->e = (x >>  8) & 0x00ff;
	temp->f = (x      ) & 0x00ff;
	PG_RETURN_MACADDR_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_macaddr);

Datum
pgstrom_random_inet(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	inet	   *temp;
	int			i, j, bits;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();

	if (!PG_ARGISNULL(1))
		temp = (inet *)PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(1));
	else
	{
		/* default template 192.168.0.1/16 */
		temp = palloc(sizeof(inet));
		temp->inet_data.family = PGSQL_AF_INET;
		temp->inet_data.bits = 16;
		temp->inet_data.ipaddr[0] = 0xc0;
		temp->inet_data.ipaddr[1] = 0xa8;
		temp->inet_data.ipaddr[2] = 0x01;
		temp->inet_data.ipaddr[3] = 0x00;
		SET_VARSIZE(temp, sizeof(inet));
	}
	bits = ip_bits(temp);
	i = ip_maxbits(temp) / 8 - 1;
	j = v = 0;
	while (bits > 0)
	{
		if (j < 8)
		{
			v |= (cl_ulong)random() << j;
			j += 31;	/* note: only 31b of random() are valid */
		}
		if (bits >= 8)
			temp->inet_data.ipaddr[i--] = (v & 0xff);
		else
		{
			cl_uint		mask = (1 << bits) - 1;

			temp->inet_data.ipaddr[i] &= ~(mask);
			temp->inet_data.ipaddr[i] |= (v & mask);
			i--;
		}
		bits -= 8;
		v >>= 8;
	}
	ip_bits(temp) = ip_maxbits(temp);
	PG_RETURN_INET_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_inet);

Datum
pgstrom_random_text(PG_FUNCTION_ARGS)
{
	static const char *base32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	text	   *temp;
	char	   *pos;
	int			i, j, n;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();

	if (PG_ARGISNULL(1))
		temp = cstring_to_text("test_**");
	else
		temp = PG_GETARG_TEXT_P_COPY(1);

	n = VARSIZE_ANY_EXHDR(temp);
	pos = VARDATA_ANY(temp);
	for (i=0, j=0, v=0; i < n; i++, pos++)
	{
		if (*pos == '*')
		{
			if (j < 5)
			{
				v |= (cl_ulong)random() << j;
				j += 31;
			}
			*pos = base32[v & 0x1f];
			v >>= 5;
			j -= 5;
		}
	}
	PG_RETURN_TEXT_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_text);

Datum
pgstrom_random_text_length(PG_FUNCTION_ARGS)
{
	static const char *base64 =
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz"
		"0123456789+/";
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	cl_int		maxlen;
	text	   *temp;
	char	   *pos;
	int			i, j, n;
	cl_ulong	v = 0;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	maxlen = (PG_ARGISNULL(1) ? 10 : PG_GETARG_INT32(1));
	if (maxlen < 1 || maxlen > BLCKSZ)
		elog(ERROR, "%s: max length too much", __FUNCTION__);
	n = 1 + random() % maxlen;

	temp = palloc(VARHDRSZ + n);
	SET_VARSIZE(temp, VARHDRSZ + n);
	pos = VARDATA(temp);
	for (i=0, j=0; i < n; i++, pos++)
	{
		if (j < 6)
		{
			v |= (cl_ulong)random() << j;
			j += 31;
		}
		*pos = base64[v & 0x3f];
		v >>= 6;
		j -= 6;
	}
	PG_RETURN_TEXT_P(temp);
}
PG_FUNCTION_INFO_V1(pgstrom_random_text_length);

static Datum
simple_make_range(TypeCacheEntry *typcache, Datum x_val, Datum y_val)
{
	RangeBound	x, y;
	RangeType  *range;

	memset(&x, 0, sizeof(RangeBound));
	x.val = x_val;
	x.infinite = generate_null(0.5);
	x.inclusive = generate_null(25.0);

	memset(&y, 0, sizeof(RangeBound));
	y.val = y_val;
	y.infinite = generate_null(0.5);
	y.inclusive = generate_null(25.0);

	if (x.infinite || y.infinite || x.val <= y.val)
	{
		x.lower = true;
		range = make_range(typcache, &x, &y, false);
	}
	else
	{
		y.lower = true;
		range = make_range(typcache, &y, &x, false);
	}
	return PointerGetDatum(range);
}

Datum
pgstrom_random_int4range(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int32		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT32(1) : 0);
	int32		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT32(2) : INT_MAX);
	int32		x, y;
	Oid			type_oid;
	TypeCacheEntry *typcache;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("int4range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + random() % (upper - lower);
	y = lower + random() % (upper - lower);
	return simple_make_range(typcache,
							 Int32GetDatum(x),
							 Int32GetDatum(y));
}
PG_FUNCTION_INFO_V1(pgstrom_random_int4range);

Datum
pgstrom_random_int8range(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int64		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT64(1) : 0);
	int64		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT64(2) : LONG_MAX);
	TypeCacheEntry *typcache;
	Oid			type_oid;
	int64		x, y, v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("int8range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	v = ((int64)random() << 31) | (int64)random();
	x = lower + v % (upper - lower);
	v = ((int64)random() << 31) | (int64)random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 Int64GetDatum(x),
							 Int64GetDatum(y));
}
PG_FUNCTION_INFO_V1(pgstrom_random_int8range);

Datum
pgstrom_random_tsrange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	struct pg_tm tm;
	TypeCacheEntry *typcache;
	Oid			type_oid;
	Timestamp	x, y;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);

	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("tsrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	v = ((cl_ulong)random() << 31) | random();
	x = lower + v % (upper - lower);
	v = ((cl_ulong)random() << 31) | random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 TimestampGetDatum(x),
							 TimestampGetDatum(y));	
}
PG_FUNCTION_INFO_V1(pgstrom_random_tsrange);

Datum
pgstrom_random_tstzrange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	struct pg_tm tm;
	TypeCacheEntry *typcache;
	Oid			type_oid;
	Timestamp	x, y;
	cl_ulong	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);

	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("tstzrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	v = ((cl_ulong)random() << 31) | random();
	x = lower + v % (upper - lower);
	v = ((cl_ulong)random() << 31) | random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 TimestampTzGetDatum(x),
							 TimestampTzGetDatum(y));	
}
PG_FUNCTION_INFO_V1(pgstrom_random_tstzrange);

Datum
pgstrom_random_daterange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	DateADT		lower;
	DateADT		upper;
	DateADT		x, y;
	TypeCacheEntry *typcache;
	Oid			type_oid;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_DATEADT(1);
	else
		lower = date2j(2015, 1, 1) - POSTGRES_EPOCH_JDATE;
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_DATEADT(2);
	else
		upper = date2j(2025, 12, 31) - POSTGRES_EPOCH_JDATE;
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);

	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   CStringGetDatum("daterange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + random() % (upper - lower);
	y = lower + random() % (upper - lower);
	return simple_make_range(typcache,
							 DateADTGetDatum(x),
							 DateADTGetDatum(y));
}
PG_FUNCTION_INFO_V1(pgstrom_random_daterange);

Datum
pgstrom_abort_if(PG_FUNCTION_ARGS)
{
	bool		cond = PG_GETARG_BOOL(0);

	if (cond)
		elog(ERROR, "abort transaction");

	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_abort_if);
