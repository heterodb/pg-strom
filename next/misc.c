/*
 * misc.c
 *
 * miscellaneous and uncategorized routines but usefull for multiple subsystems
 * of PG-Strom.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/*
 * fixup_varnode_to_origin
 */
Node *
fixup_varnode_to_origin(Node *node, List *cscan_tlist)
{
	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var	   *varnode = (Var *)node;
		TargetEntry *tle;

		if (cscan_tlist != NIL)
		{
			Assert(varnode->varno == INDEX_VAR);
			Assert(varnode->varattno >= 1 &&
				   varnode->varattno <= list_length(cscan_tlist));
			tle = list_nth(cscan_tlist, varnode->varattno - 1);
			return (Node *)copyObject(tle->expr);
		}
		Assert(!IS_SPECIAL_VARNO(varnode->varno));
	}
	return expression_tree_mutator(node, fixup_varnode_to_origin,
								   (void *)cscan_tlist);
}

#if 0
/*
 * find_appinfos_by_relids_nofail
 *
 * It is almost equivalent to find_appinfos_by_relids(), but ignores
 * relations that are not partition leafs, instead of ereport().
 * In addition, it tries to solve multi-leve parent-child relations.
 */
static AppendRelInfo *
build_multilevel_appinfos(PlannerInfo *root,
						  AppendRelInfo **appstack, int nlevels)
{
	AppendRelInfo *apinfo = appstack[nlevels-1];
	AppendRelInfo *apleaf = appstack[0];
	AppendRelInfo *result;
	ListCell   *lc;
	int			i;

	foreach (lc, root->append_rel_list)
	{
		AppendRelInfo *aptemp = lfirst(lc);

		if (aptemp->child_relid == apinfo->parent_relid)
		{
			appstack[nlevels] = aptemp;
			return build_multilevel_appinfos(root, appstack, nlevels+1);
		}
	}
	/* shortcut if a simple single-level relationship */
	if (nlevels == 1)
		return apinfo;

	result = makeNode(AppendRelInfo);
	result->parent_relid = apinfo->parent_relid;
	result->child_relid = apleaf->child_relid;
	result->parent_reltype = apinfo->parent_reltype;
	result->child_reltype = apleaf->child_reltype;
	foreach (lc, apinfo->translated_vars)
	{
		Var	   *var = lfirst(lc);

		for (i=nlevels-1; i>=0; i--)
		{
			AppendRelInfo *apcurr = appstack[i];
			Var	   *temp;

			if (var->varattno > list_length(apcurr->translated_vars))
				elog(ERROR, "attribute %d of relation \"%s\" does not exist",
					 var->varattno, get_rel_name(apcurr->parent_reloid));
			temp = list_nth(apcurr->translated_vars, var->varattno - 1);
			if (!temp)
				elog(ERROR, "attribute %d of relation \"%s\" does not exist",
					 var->varattno, get_rel_name(apcurr->parent_reloid));
			var = temp;
		}
		result->translated_vars = lappend(result->translated_vars, var);
	}
	result->parent_reloid = apinfo->parent_reloid;

	return result;
}

AppendRelInfo **
find_appinfos_by_relids_nofail(PlannerInfo *root,
							   Relids relids,
							   int *nappinfos)
{
	AppendRelInfo **appstack;
	AppendRelInfo **appinfos;
	ListCell   *lc;
	int			nrooms = bms_num_members(relids);
	int			nitems = 0;

	appinfos = palloc0(sizeof(AppendRelInfo *) * nrooms);
	appstack = alloca(sizeof(AppendRelInfo *) * root->simple_rel_array_size);
	foreach (lc, root->append_rel_list)
	{
		AppendRelInfo *apinfo = lfirst(lc);

		if (bms_is_member(apinfo->child_relid, relids))
		{
			appstack[0] = apinfo;
			appinfos[nitems++] = build_multilevel_appinfos(root, appstack, 1);
		}
	}
	Assert(nitems <= nrooms);
	*nappinfos = nitems;

	return appinfos;
}

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
#endif

/*
 * append a binary chunk at the aligned block
 */
int
__appendBinaryStringInfo(StringInfo buf, const void *data, int datalen)
{
	static uint64_t __zero = 0;
	int		padding = (MAXALIGN(buf->len) - buf->len);
	int		pos;

	if (padding > 0)
		appendBinaryStringInfo(buf, (char *)&__zero, padding);
	pos = buf->len;
	appendBinaryStringInfo(buf, data, datalen);
	return pos;
}

int
__appendZeroStringInfo(StringInfo buf, int nbytes)
{
	static uint64_t __zero = 0;
	int		padding = (MAXALIGN(buf->len) - buf->len);
	int		pos;

	if (padding > 0)
		appendBinaryStringInfo(buf, (char *)&__zero, padding);
	pos = buf->len;
	enlargeStringInfo(buf, nbytes);
	memset(buf->data + pos, 0, nbytes);
	buf->len += nbytes;

	return pos;
}

/*
 * get_type_name
 */
char *
get_type_name(Oid type_oid, bool missing_ok)
{
	HeapTuple	tup;
	char	   *retval;

	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tup))
	{
		if (!missing_ok)
			elog(ERROR, "cache lookup failed for type %u", type_oid);
		return NULL;
	}
	retval = pstrdup(NameStr(((Form_pg_type) GETSTRUCT(tup))->typname));
	ReleaseSysCache(tup);

	return retval;
}

/*
 * get_relation_am
 */
Oid
get_relation_am(Oid rel_oid, bool missing_ok)
{
	HeapTuple	tup;
	Oid			relam;

	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(rel_oid));
	if (!HeapTupleIsValid(tup))
	{
		if (!missing_ok)
			elog(ERROR, "cache lookup failed for relation %u", rel_oid);
		return InvalidOid;
	}
	relam = ((Form_pg_class) GETSTRUCT(tup))->relam;
	ReleaseSysCache(tup);

	return relam;
}

/*
 * Bitmapset <-> numeric List transition
 */
List *
bms_to_pglist(const Bitmapset *bms)
{
	List   *pglist = NIL;
	int		k;

	for (k = bms_next_member(bms, -1);
		 k >= 0;
		 k = bms_next_member(bms, k))
	{
		pglist = lappend_int(pglist, k);
	}
	return pglist;
}

Bitmapset *
bms_from_pglist(List *pglist)
{
	Bitmapset  *bms = NULL;
	ListCell   *lc;

	foreach (lc, pglist)
	{
		bms = bms_add_member(bms, lfirst_int(lc));
	}
	return bms;
}

#if 0
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
#if PG_VERSION_NUM < 120000
		case T_ResultPath:
#else
		case T_GroupResultPath:
#endif
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
		case T_GatherMergePath:
			if (walker(((GatherMergePath *)node)->subpath, context))
				return true;
			break;
		case T_ProjectionPath:
			if (walker(((ProjectionPath *)node)->subpath, context))
				return true;
			break;
		case T_ProjectSetPath:
			if (walker(((ProjectSetPath *)node)->subpath, context))
				return true;
			break;
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
#if PG_VERSION_NUM < 140000
			foreach (lc, ((ModifyTablePath *)node)->subpaths)
			{
				if (walker((Path *)lfirst(lc), context))
					return true;
			}
#else
			if (walker(((ModifyTablePath *)node)->subpath))
				return true;
#endif
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

static bool
__pathtree_has_parallel_aware(Path *path, void *context)
{
	bool	rv = path->parallel_aware;

	if (!rv)
		rv = pathnode_tree_walker(path, __pathtree_has_parallel_aware, context);
	return rv;
}

bool
pathtree_has_parallel_aware(Path *node)
{
	return __pathtree_has_parallel_aware(node, NULL);
}
#endif

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
static Path *
__pgstrom_copy_joinpath(const JoinPath *a, size_t sz)
{
	JoinPath   *b = pmemdup(a, sz);

	b->outerjoinpath = pgstrom_copy_pathnode(a->outerjoinpath);
	b->innerjoinpath = pgstrom_copy_pathnode(a->innerjoinpath);

	return &b->path;
}

Path *
pgstrom_copy_pathnode(const Path *pathnode)
{
	if (!pathnode)
		return NULL;

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
			{
				CustomPath	   *a = (CustomPath *)pathnode;
				CustomPath	   *b = pmemdup(a, sizeof(CustomPath));
				List		   *subpaths = NIL;
				ListCell	   *lc;

				foreach (lc, a->custom_paths)
				{
					Path	   *sp = pgstrom_copy_pathnode(lfirst(lc));
					subpaths = lappend(subpaths, sp);
				}
				b->custom_paths = subpaths;
				return &b->path;
			}
		case T_NestPath:
			return __pgstrom_copy_joinpath((JoinPath *)pathnode, sizeof(NestPath));
		case T_MergePath:
			return __pgstrom_copy_joinpath((JoinPath *)pathnode, sizeof(MergePath));
		case T_HashPath:
			return __pgstrom_copy_joinpath((JoinPath *)pathnode, sizeof(HashPath));
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
		case T_GroupResultPath:
			return pmemdup(pathnode, sizeof(GroupResultPath));
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
				b->subpath = pgstrom_copy_pathnode(a->subpath);
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
 * pgstrom_define_shell_type - A wrapper for TypeShellMake with a particular OID
 */
PG_FUNCTION_INFO_V1(pgstrom_define_shell_type);
Datum
pgstrom_define_shell_type(PG_FUNCTION_ARGS)
{
	char   *type_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
	Oid		type_oid = PG_GETARG_OID(1);
	Oid		type_namespace = PG_GETARG_OID(2);
	bool	__IsBinaryUpgrade = IsBinaryUpgrade;
	Oid		__binary_upgrade_next_pg_type_oid = binary_upgrade_next_pg_type_oid;

	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser to create a shell type")));
	PG_TRY();
	{
		IsBinaryUpgrade = true;
		binary_upgrade_next_pg_type_oid = type_oid;

		TypeShellMake(type_name, type_namespace, GetUserId());
	}
	PG_CATCH();
	{
		IsBinaryUpgrade = __IsBinaryUpgrade;
		binary_upgrade_next_pg_type_oid = __binary_upgrade_next_pg_type_oid;
		PG_RE_THROW();
	}
	PG_END_TRY();
	IsBinaryUpgrade = __IsBinaryUpgrade;
	binary_upgrade_next_pg_type_oid = __binary_upgrade_next_pg_type_oid;

	PG_RETURN_OID(type_oid);
}

/*
 * ----------------------------------------------------------------
 *
 * SQL functions to support regression test
 *
 * ----------------------------------------------------------------
 */
Datum pgstrom_random_setseed(PG_FUNCTION_ARGS);
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

static unsigned int		pgstrom_random_seed = 0;
static bool				pgstrom_random_seed_set = false;

Datum
pgstrom_random_setseed(PG_FUNCTION_ARGS)
{
	unsigned int	seed = PG_GETARG_UINT32(0);

	pgstrom_random_seed = seed ^ 0xdeadbeafU;
	pgstrom_random_seed_set = true;

	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_random_setseed);

static int64_t
__random(void)
{
	if (!pgstrom_random_seed_set)
	{
		pgstrom_random_seed = (unsigned int)MyProcPid ^ 0xdeadbeafU;
		pgstrom_random_seed_set = true;
	}
	return (uint64_t)rand_r(&pgstrom_random_seed);
}

static inline double
__drand48(void)
{
	return (double)__random() / (double)RAND_MAX;
}

static inline bool
generate_null(double ratio)
{
	if (ratio <= 0.0)
		return false;
	if (100.0 * __drand48() < ratio)
		return true;
	return false;
}

Datum
pgstrom_random_int(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int64		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT64(1) : 0);
	int64		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT64(2) : INT_MAX);
	uint64_t	v;

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_INT64(lower);
	v = (__random() << 31) | __random();

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

	PG_RETURN_FLOAT8((upper - lower) * __drand48() + lower);
}
PG_FUNCTION_INFO_V1(pgstrom_random_float);

Datum
pgstrom_random_date(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	DateADT		lower;
	DateADT		upper;
	uint64_t	v;

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
	v = (__random() << 31) | __random();

	PG_RETURN_DATEADT(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_date);

Datum
pgstrom_random_time(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	TimeADT		lower = 0;
	TimeADT		upper = HOURS_PER_DAY * USECS_PER_HOUR - 1;
	uint64_t	v;

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
	v = (__random() << 31) | __random();

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
	uint64_t	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMEADT(1);
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMEADT(2);
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	temp = palloc(sizeof(TimeTzADT));
	temp->zone = (__random() % 23 - 11) * USECS_PER_HOUR;
	if (upper == lower)
		temp->time = lower;
	else
	{
		v = (__random() << 31) | __random();
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
	uint64_t	v;
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
	v = (__random() << 31) | __random();

	PG_RETURN_TIMESTAMP(lower + v % (upper - lower));
}
PG_FUNCTION_INFO_V1(pgstrom_random_timestamp);

Datum
pgstrom_random_macaddr(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	macaddr	   *temp;
	uint64_t	lower;
	uint64_t	upper;
	uint64_t	v, x;

	if (PG_ARGISNULL(1))
		lower = 0xabcd00000000UL;
	else
	{
		temp = PG_GETARG_MACADDR_P(1);
		lower = (((uint64_t)temp->a << 40) | ((uint64_t)temp->b << 32) |
				 ((uint64_t)temp->c << 24) | ((uint64_t)temp->d << 16) |
				 ((uint64_t)temp->e <<  8) | ((uint64_t)temp->f));
	}

	if (PG_ARGISNULL(2))
		upper = 0xabcdffffffffUL;
	else
	{
		temp = PG_GETARG_MACADDR_P(2);
		upper = (((uint64_t)temp->a << 40) | ((uint64_t)temp->b << 32) |
				 ((uint64_t)temp->c << 24) | ((uint64_t)temp->d << 16) |
				 ((uint64_t)temp->e <<  8) | ((uint64_t)temp->f));
	}

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		x = lower;
	else
	{
		v = (__random() << 31) | __random();
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
	uint64_t	v;

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
			v |= __random() << j;
			j += 31;	/* note: only 31b of random() are valid */
		}
		if (bits >= 8)
			temp->inet_data.ipaddr[i--] = (v & 0xff);
		else
		{
			uint32_t		mask = (1 << bits) - 1;

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
	uint64_t	v;

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
				v |= __random() << j;
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
	int32_t		maxlen;
	text	   *temp;
	char	   *pos;
	int			i, j, n;
	uint64_t	v = 0;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	maxlen = (PG_ARGISNULL(1) ? 10 : PG_GETARG_INT32(1));
	if (maxlen < 1 || maxlen > BLCKSZ)
		elog(ERROR, "%s: max length too much", __FUNCTION__);
	n = 1 + __random() % maxlen;

	temp = palloc(VARHDRSZ + n);
	SET_VARSIZE(temp, VARHDRSZ + n);
	pos = VARDATA(temp);
	for (i=0, j=0; i < n; i++, pos++)
	{
		if (j < 6)
		{
			v |= __random() << j;
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
	x.lower = true;

	memset(&y, 0, sizeof(RangeBound));
	y.val = y_val;
	y.infinite = generate_null(0.5);
	y.inclusive = generate_null(25.0);
	y.lower = false;

	range = make_range(typcache, &x, &y, false);

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
							   Anum_pg_type_oid,
							   CStringGetDatum("int4range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'int4range' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + __random() % (upper - lower);
	y = lower + __random() % (upper - lower);
	return simple_make_range(typcache,
							 Int32GetDatum(Min(x,y)),
							 Int32GetDatum(Max(x,y)));
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
							   Anum_pg_type_oid,
							   CStringGetDatum("int8range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'int8range' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	v = (__random() << 31) | __random();
	x = lower + v % (upper - lower);
	v = (__random() << 31) | __random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 Int64GetDatum(Min(x,y)),
							 Int64GetDatum(Max(x,y)));
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
	uint64_t	v;

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
							   Anum_pg_type_oid,
							   CStringGetDatum("tsrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'tsrange' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	v = (__random() << 31) | __random();
	x = lower + v % (upper - lower);
	v = (__random() << 31) | __random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 TimestampGetDatum(Min(x,y)),
							 TimestampGetDatum(Max(x,y)));	
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
	uint64_t	v;

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
							   Anum_pg_type_oid,
							   CStringGetDatum("tstzrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'tstzrange' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	v = (__random() << 31) | __random();
	x = lower + v % (upper - lower);
	v = (__random() << 31) | __random();
	y = lower + v % (upper - lower);
	return simple_make_range(typcache,
							 TimestampTzGetDatum(Min(x,y)),
							 TimestampTzGetDatum(Max(x,y)));	
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
							   Anum_pg_type_oid,
							   CStringGetDatum("daterange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'daterange' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + __random() % (upper - lower);
	y = lower + __random() % (upper - lower);
	return simple_make_range(typcache,
							 DateADTGetDatum(Min(x,y)),
							 DateADTGetDatum(Max(x,y)));
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

/*
 * Simple wrapper for read(2) and write(2) to ensure full-buffer read and
 * write, regardless of i/o-size and signal interrupts.
 */
ssize_t
__readFile(int fdesc, void *buffer, size_t nbytes)
{
	ssize_t		rv, count = 0;

	do {
		rv = read(fdesc, (char *)buffer + count, nbytes - count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

ssize_t
__preadFile(int fdesc, void *buffer, size_t nbytes, off_t f_pos)
{
	ssize_t		rv, count = 0;

	do {
		rv = pread(fdesc, (char *)buffer + count, nbytes - count, f_pos + count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

ssize_t
__writeFile(int fdesc, const void *buffer, size_t nbytes)
{
	ssize_t		rv, count = 0;

	do {
		rv = write(fdesc, (const char *)buffer + count, nbytes - count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

ssize_t
__pwriteFile(int fdesc, const void *buffer, size_t nbytes, off_t f_pos)
{
	ssize_t		rv, count = 0;

	do {
		rv = pwrite(fdesc, (const char *)buffer + count, nbytes - count, f_pos + count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

/*
 * mmap/munmap wrapper that is automatically unmapped on regarding to
 * the resource-owner.
 */
typedef struct
{
	void	   *mmap_addr;
	size_t		mmap_size;
	int			mmap_prot;
	int			mmap_flags;
	int			mmap_fdesc;
	ResourceOwner owner;
} mmapEntry;

static HTAB	   *mmap_tracker_htab = NULL;

static void
cleanup_mmap_chunks(ResourceReleasePhase phase,
					bool isCommit,
					bool isTopLevel,
					void *arg)
{
	if (mmap_tracker_htab &&
		hash_get_num_entries(mmap_tracker_htab) > 0)
	{
		HASH_SEQ_STATUS	seq;
		mmapEntry	   *entry;

		hash_seq_init(&seq, mmap_tracker_htab);
		while ((entry = hash_seq_search(&seq)) != NULL)
		{
			if (entry->owner != CurrentResourceOwner)
				continue;
			if (isCommit)
				elog(WARNING, "mmap (%p-%p; sz=%zu) leaks, and still mapped",
					 (char *)entry->mmap_addr,
					 (char *)entry->mmap_addr + entry->mmap_size,
					 entry->mmap_size);

			
			if (munmap(entry->mmap_addr, entry->mmap_size) != 0)
				elog(WARNING, "failed on munmap(%p, %zu): %m",
					 entry->mmap_addr, entry->mmap_size);
			if (close(entry->mmap_fdesc) != 0)
				elog(WARNING, "failed on close(%d): %m", entry->mmap_fdesc);
			hash_search(mmap_tracker_htab,
						&entry->mmap_addr,
						HASH_REMOVE,
						NULL);
		}
	}
}

void *
__mmapFile(void *addr, size_t length,
		   int prot, int flags, int fdesc, off_t offset)
{
	void	   *mmap_addr;
	size_t		mmap_size = TYPEALIGN(PAGE_SIZE, length);
	mmapEntry  *entry;
	bool		found;

	if (!mmap_tracker_htab)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(void *);
		hctl.entrysize = sizeof(mmapEntry);
		hctl.hcxt = CacheMemoryContext;
		mmap_tracker_htab = hash_create("mmap_tracker_htab",
										256,
										&hctl,
										HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
		RegisterResourceReleaseCallback(cleanup_mmap_chunks, 0);
	}
	mmap_addr = mmap(addr, mmap_size, prot, flags, fdesc, offset);
	if (mmap_addr == MAP_FAILED)
		elog(ERROR, "failed on mmap(%zu): %m", mmap_size);
	PG_TRY();
	{
		entry = hash_search(mmap_tracker_htab,
							&mmap_addr,
							HASH_ENTER,
							&found);
		if (found)
			elog(ERROR, "Bug? duplicated mmap entry");
		Assert(entry->mmap_addr == mmap_addr);
		entry->mmap_size = mmap_size;
		entry->mmap_prot = prot;
		entry->mmap_flags = flags;
		entry->mmap_fdesc = fdesc;
		entry->owner = CurrentResourceOwner;
	}
	PG_CATCH();
	{
		if (munmap(mmap_addr, mmap_size) != 0)
			elog(WARNING, "failed on munmap(%p, %zu): %m",
				 mmap_addr, mmap_size);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return mmap_addr;
}

bool
__munmapFile(void *mmap_addr)
{
	mmapEntry  *entry;

	if (mmap_tracker_htab &&
		(entry = hash_search(mmap_tracker_htab,
							 &mmap_addr,
							 HASH_REMOVE,
							 NULL)) != NULL)
	{
		if (munmap(entry->mmap_addr,
				   entry->mmap_size) != 0)
			elog(WARNING, "failed on munmap(%p, %zu): %m",
				 entry->mmap_addr,
				 entry->mmap_size);
		if (close(entry->mmap_fdesc) != 0)
			elog(WARNING, "failed on close(%d): %m",
				 entry->mmap_fdesc);
		return true;
	}
	elog(WARNING, "addr=%p looks not memory-mapped", mmap_addr);
	return false;
}

void *
__mremapFile(void *mmap_addr, size_t new_size)
{
	mmapEntry  *entry;
	void	   *addr;

	if (!mmap_tracker_htab ||
		!(entry = hash_search(mmap_tracker_htab,
							  &mmap_addr, HASH_FIND, NULL)))
		return NULL;	/* not found */

	/* nothing to do? */
	if (new_size <= entry->mmap_size)
		return entry->mmap_addr;
	addr = mremap(entry->mmap_addr,
				  entry->mmap_size,
				  new_size,
				  MREMAP_MAYMOVE);
	if (addr == MAP_FAILED)
	{
		elog(WARNING, "failed on mremap(%p, %zu, %zu): %m",
			 entry->mmap_addr,
			 entry->mmap_size,
			 new_size);
		return MAP_FAILED;
	}
	entry->mmap_addr = addr;
	entry->mmap_size = new_size;
	return addr;
}

void *
__mmapShmem(size_t length)
{
	static uint	my_random_seed = 0;
	static bool	my_random_initialized = false;
	char	namebuf[128];
	int		fdesc;
	void   *addr;

	if (!my_random_initialized)
	{
		my_random_seed = (uint)MyProcPid ^ 0xcafebabeU;
		my_random_initialized = true;
	}

	do {
		snprintf(namebuf, sizeof(namebuf),
				 ".pgstrom_shmbuf_%u_%d",
				 PostPortNumber, rand_r(&my_random_seed));
		fdesc = shm_open(namebuf, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fdesc < 0 && errno != EEXIST)
			elog(ERROR, "failed on shm_open('%s'): %m", namebuf);
	} while (fdesc < 0);

	PG_TRY();
	{
		length = (length + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

		if (ftruncate(fdesc, length) != 0)
			elog(ERROR, "failed on ftruncate('%s', %zu): %m", namebuf, length);
		addr = __mmapFile(NULL, length,
						  PROT_READ | PROT_WRITE,
						  MAP_SHARED,
						  fdesc, length);
	}
	PG_CATCH();
	{
		if (shm_unlink(namebuf) != 0)
			elog(WARNING, "failed on shm_unlink('%s'): %m", namebuf);
		if (close(fdesc) != 0)
			elog(WARNING, "failed on close(%s): %m", namebuf);
		PG_RE_THROW();
	}
	PG_END_TRY();

	if (shm_unlink(namebuf) != 0)
		elog(WARNING, "failed on shm_unlink('%s'): %m", namebuf);

	return addr;
}
