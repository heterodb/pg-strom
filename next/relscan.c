/*
 * relscan.c
 *
 * Routines related to outer relation scan
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* Data structure for collecting qual clauses that match an index */
typedef struct
{
	bool		nonempty;	/* True if lists are not all empty */
	/* Lists of RestrictInfos, one per index column */
	List	   *indexclauses[INDEX_MAX_KEYS];
} IndexClauseSet;

/* static variables */
static bool		pgstrom_enable_brin;
static HTAB	   *tablespace_optimal_xpu_htable = NULL;

/*
 * simple_match_clause_to_indexcol
 *
 * It is a simplified version of match_clause_to_indexcol.
 * Also see optimizer/path/indxpath.c
 */
static bool
simple_match_clause_to_indexcol(IndexOptInfo *index,
								int indexcol,
								RestrictInfo *rinfo)
{
	Expr	   *clause = rinfo->clause;
	Index		index_relid = index->rel->relid;
	Oid			opfamily = index->opfamily[indexcol];
	Oid			idxcollation = index->indexcollations[indexcol];
	Node	   *leftop;
	Node	   *rightop;
	Relids		left_relids;
	Relids		right_relids;
	Oid			expr_op;
	Oid			expr_coll;

	/* Clause must be a binary opclause */
	if (!is_opclause(clause))
		return false;

	leftop = get_leftop(clause);
	rightop = get_rightop(clause);
	if (!leftop || !rightop)
		return false;
	left_relids = rinfo->left_relids;
	right_relids = rinfo->right_relids;
	expr_op = ((OpExpr *) clause)->opno;
	expr_coll = ((OpExpr *) clause)->inputcollid;

	if (OidIsValid(idxcollation) && idxcollation != expr_coll)
		return false;

	/*
	 * Check for clauses of the form:
	 *    (indexkey operator constant) OR
	 *    (constant operator indexkey)
	 */
	if (match_index_to_operand(leftop, indexcol, index) &&
		!bms_is_member(index_relid, right_relids) &&
		!contain_volatile_functions(rightop) &&
		op_in_opfamily(expr_op, opfamily))
		return true;

	if (match_index_to_operand(rightop, indexcol, index) &&
		!bms_is_member(index_relid, left_relids) &&
		!contain_volatile_functions(leftop) &&
		op_in_opfamily(get_commutator(expr_op), opfamily))
		return true;

	return false;
}

/*
 * simple_match_clause_to_index
 *
 * It is a simplified version of match_clause_to_index.
 * Also see optimizer/path/indxpath.c
 */
static void
simple_match_clause_to_index(IndexOptInfo *index,
							 RestrictInfo *rinfo,
							 IndexClauseSet *clauseset)
{
	int		indexcol;

	/*
	 * Never match pseudoconstants to indexes.  (Normally a match could not
	 * happen anyway, since a pseudoconstant clause couldn't contain a Var,
	 * but what if someone builds an expression index on a constant? It's not
	 * totally unreasonable to do so with a partial index, either.)
	 */
	if (rinfo->pseudoconstant)
		return;

	/*
	 * If clause can't be used as an indexqual because it must wait till after
	 * some lower-security-level restriction clause, reject it.
	 */
	if (!restriction_is_securely_promotable(rinfo, index->rel))
		return;

	/* OK, check each index column for a match */
	for (indexcol = 0; indexcol < index->ncolumns; indexcol++)
	{
		if (simple_match_clause_to_indexcol(index,
											indexcol,
											rinfo))
		{
			clauseset->indexclauses[indexcol] =
				list_append_unique_ptr(clauseset->indexclauses[indexcol],
									   rinfo);
			clauseset->nonempty = true;
			break;
		}
	}
}

/*
 * estimate_brinindex_scan_nblocks
 *
 * see brincostestimate at utils/adt/selfuncs.c
 */
static int64_t
estimate_brinindex_scan_nblocks(PlannerInfo *root,
                                RelOptInfo *baserel,
                                IndexOptInfo *index,
                                IndexClauseSet *clauseset,
                                List **p_indexQuals)
{
	Relation		indexRel;
	BrinStatsData	statsData;
	List		   *indexQuals = NIL;
	ListCell	   *lc;
	int				icol;
	Selectivity		qualSelectivity;
	Selectivity		indexSelectivity;
	double			indexCorrelation = 0.0;
	double			indexRanges;
	double			minimalRanges;
	double			estimatedRanges;

	/* Obtain some data from the index itself. */
	indexRel = index_open(index->indexoid, AccessShareLock);
	brinGetStats(indexRel, &statsData);
	index_close(indexRel, AccessShareLock);

	/* Get selectivity of the index qualifiers */
	icol = 1;
	foreach (lc, index->indextlist)
	{
		TargetEntry *tle = lfirst(lc);
		ListCell   *cell;
		VariableStatData vardata;

		foreach (cell, clauseset->indexclauses[icol-1])
		{
			RestrictInfo *rinfo = lfirst(cell);

			indexQuals = lappend(indexQuals, rinfo);
		}

		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;
			RangeTblEntry *rte;

			/* in case of BRIN index on simple column */
			rte = root->simple_rte_array[var->varno];
			if (get_relation_stats_hook &&
				(*get_relation_stats_hook)(root, rte, var->varattno,
										   &vardata))
			{
				if (HeapTupleIsValid(vardata.statsTuple) && !vardata.freefunc)
					elog(ERROR, "no callback to release stats variable");
			}
			else
			{
				vardata.statsTuple =
					SearchSysCache3(STATRELATTINH,
									ObjectIdGetDatum(rte->relid),
									Int16GetDatum(var->varattno),
									BoolGetDatum(false));
				vardata.freefunc = ReleaseSysCache;
			}
		}
		else
		{
			if (get_index_stats_hook &&
				(*get_index_stats_hook)(root, index->indexoid, icol,
										&vardata))
			{
				if (HeapTupleIsValid(vardata.statsTuple) && !vardata.freefunc)
					elog(ERROR, "no callback to release stats variable");
			}
			else
			{
				vardata.statsTuple
					= SearchSysCache3(STATRELATTINH,
									  ObjectIdGetDatum(index->indexoid),
									  Int16GetDatum(icol),
									  BoolGetDatum(false));
                vardata.freefunc = ReleaseSysCache;
			}
		}

		if (HeapTupleIsValid(vardata.statsTuple))
		{
			AttStatsSlot	sslot;

			if (get_attstatsslot(&sslot, vardata.statsTuple,
								 STATISTIC_KIND_CORRELATION,
								 InvalidOid,
								 ATTSTATSSLOT_NUMBERS))
			{
				double		varCorrelation = 0.0;

				if (sslot.nnumbers > 0)
					varCorrelation = Abs(sslot.numbers[0]);

				if (varCorrelation > indexCorrelation)
					indexCorrelation = varCorrelation;

				free_attstatsslot(&sslot);
			}
		}
		ReleaseVariableStats(vardata);

		icol++;
	}
	qualSelectivity = clauselist_selectivity(root,
											 indexQuals,
											 baserel->relid,
											 JOIN_INNER,
											 NULL);

	/* estimate number of blocks to read */
	indexRanges = ceil((double) baserel->pages / statsData.pagesPerRange);
	if (indexRanges < 1.0)
		indexRanges = 1.0;
	minimalRanges = ceil(indexRanges * qualSelectivity);

	//elog(INFO, "strom: qualSelectivity=%.6f indexRanges=%.6f minimalRanges=%.6f indexCorrelation=%.6f", qualSelectivity, indexRanges, minimalRanges, indexCorrelation);

	if (indexCorrelation < 1.0e-10)
		estimatedRanges = indexRanges;
	else
		estimatedRanges = Min(minimalRanges / indexCorrelation, indexRanges);

	indexSelectivity = estimatedRanges / indexRanges;
	if (indexSelectivity < 0.0)
		indexSelectivity = 0.0;
	if (indexSelectivity > 1.0)
		indexSelectivity = 1.0;

	/* index quals, if any */
	if (p_indexQuals)
		*p_indexQuals = indexQuals;
	/* estimated number of blocks to read */
	return (int64_t)(indexSelectivity * (double) baserel->pages);
}

/*
 * extract_index_conditions
 */
static Node *
__fixup_indexqual_operand(Node *node, IndexOptInfo *indexOpt)
{
	ListCell   *lc;

	if (!node)
		return NULL;

	if (IsA(node, RelabelType))
	{
		RelabelType *relabel = (RelabelType *) node;

		return __fixup_indexqual_operand((Node *)relabel->arg, indexOpt);
	}

	foreach (lc, indexOpt->indextlist)
	{
		TargetEntry *tle = lfirst(lc);

		if (equal(node, tle->expr))
		{
			return (Node *)makeVar(INDEX_VAR,
								   tle->resno,
								   exprType((Node *)tle->expr),
								   exprTypmod((Node *) tle->expr),
								   exprCollation((Node *) tle->expr),
								   0);
		}
	}
	if (IsA(node, Var))
		elog(ERROR, "Bug? variable is not found at index tlist");
	return expression_tree_mutator(node, __fixup_indexqual_operand, indexOpt);
}

static List *
extract_index_conditions(List *index_quals, IndexOptInfo *indexOpt)
{
	List	   *result = NIL;
	ListCell   *lc;

	foreach (lc, index_quals)
	{
		RestrictInfo *rinfo = lfirst(lc);
		OpExpr	   *op = (OpExpr *) rinfo->clause;

		if (!IsA(rinfo->clause, OpExpr))
			elog(ERROR, "Bug? unexpected index clause: %s",
				 nodeToString(rinfo->clause));
		if (list_length(((OpExpr *)rinfo->clause)->args) != 2)
			elog(ERROR, "indexqual clause must be binary opclause");
		op = (OpExpr *)copyObject(rinfo->clause);
		if (!bms_equal(rinfo->left_relids, indexOpt->rel->relids))
			CommuteOpExpr(op);
		/* replace the indexkey expression with an index Var */
		linitial(op->args) = __fixup_indexqual_operand(linitial(op->args),
													   indexOpt);
		result = lappend(result, op);
	}
	return result;
}

/*
 * pgstrom_tryfind_brinindex
 */
bool
pgstrom_tryfind_brinindex(PlannerInfo *root,
						  RelOptInfo *baserel,
						  IndexOptInfo **p_indexOpt,
						  List **p_indexConds,
						  List **p_indexQuals,
						  int64_t *p_indexNBlocks)
{
	int64_t			indexNBlocks = INT64_MAX;
	IndexOptInfo   *indexOpt = NULL;
	List		   *indexQuals = NIL;
	ListCell	   *cell;

	if (!pgstrom_enable_brin || baserel->indexlist == NIL)
		return NULL;

	foreach (cell, baserel->indexlist)
	{
		IndexOptInfo   *index = (IndexOptInfo *) lfirst(cell);
        List		   *temp = NIL;
        ListCell	   *lc;
        uint64_t		nblocks;
        IndexClauseSet	clauseset;

        /* Protect limited-size array in IndexClauseSets */
        Assert(index->ncolumns <= INDEX_MAX_KEYS);

        /* Ignore partial indexes that do not match the query. */
        if (index->indpred != NIL && !index->predOK)
            continue;

        /* Only BRIN-indexes are now supported */
        if (index->relam != BRIN_AM_OID)
            continue;

        /* see match_clauses_to_index */
        memset(&clauseset, 0, sizeof(IndexClauseSet));
        foreach (lc, index->indrestrictinfo)
        {
            RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

            simple_match_clause_to_index(index, rinfo, &clauseset);
        }
        if (!clauseset.nonempty)
            continue;

        /*
         * In case when multiple BRIN-indexes are configured,
         * the one with minimal selectivity is the best choice.
         */
        nblocks = estimate_brinindex_scan_nblocks(root, baserel,
												  index,
												  &clauseset,
												  &temp);
		if (indexNBlocks > nblocks)
		{
			indexOpt = index;
			indexQuals = temp;
			indexNBlocks = nblocks;
		}
	}

	if (indexOpt)
	{
		*p_indexOpt = indexOpt;
		*p_indexConds = extract_index_conditions(indexQuals, indexOpt);
		*p_indexQuals = indexQuals;
		*p_indexNBlocks = indexNBlocks;
		return true;
	}
	return false;
}

/* ----------------------------------------------------------------
 *
 * GPUDirectSQL related routines
 *
 * ----------------------------------------------------------------
 */
static HTAB	   *tablespace_optimal_gpu_htable = NULL;
typedef struct
{
	Oid			tablespace_oid;
	bool		is_valid;
	Bitmapset	optimal_gpus;
} tablespace_optimal_gpu_hentry;

static void
tablespace_optimal_gpu_cache_callback(Datum arg, int cacheid, uint32 hashvalue)
{
	/* invalidate all the cached status */
	if (tablespace_optimal_gpu_htable)
	{
		hash_destroy(tablespace_optimal_gpu_htable);
		tablespace_optimal_gpu_htable = NULL;
	}
}

/*
 * GetOptimalGpusForTablespace
 */
static const Bitmapset *
GetOptimalGpusForTablespace(Oid tablespace_oid)
{
	tablespace_optimal_gpu_hentry *hentry;
	bool		found;

	if (!pgstrom_gpudirect_enabled())
		return NULL;

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	if (!tablespace_optimal_gpu_htable)
	{
		HASHCTL		hctl;
		int			nwords = (numGpuDevAttrs / BITS_PER_BITMAPWORD) + 1;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = MAXALIGN(offsetof(tablespace_optimal_gpu_hentry,
										   optimal_gpus.words[nwords]));
		tablespace_optimal_gpu_htable
			= hash_create("TablespaceOptimalGpu", 128,
						  &hctl, HASH_ELEM | HASH_BLOBS);
	}

	hentry = (tablespace_optimal_gpu_hentry *)
		hash_search(tablespace_optimal_gpu_htable,
					&tablespace_oid,
					HASH_ENTER,
					&found);
	if (!found || !hentry->is_valid)
	{
		char	   *pathname;
		File		filp;
		Bitmapset  *optimal_gpus;

		Assert(hentry->tablespace_oid == tablespace_oid);

		pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);
		filp = PathNameOpenFile(pathname, O_RDONLY);
		if (filp < 0)
		{
			elog(WARNING, "failed on open('%s') of tablespace %u: %m",
				 pathname, tablespace_oid);
			return NULL;
		}
		optimal_gpus = extraSysfsLookupOptimalGpus(filp);
		if (!optimal_gpus)
			hentry->optimal_gpus.nwords = 0;
		else
		{
			Assert(optimal_gpus->nwords <= (numGpuDevAttrs/BITS_PER_BITMAPWORD)+1);
			memcpy(&hentry->optimal_gpus, optimal_gpus,
				   offsetof(Bitmapset, words[optimal_gpus->nwords]));
			bms_free(optimal_gpus);
		}
		FileClose(filp);
		hentry->is_valid = true;
	}
	Assert(hentry->is_valid);
	return (hentry->optimal_gpus.nwords > 0 ? &hentry->optimal_gpus : NULL);
}

const Bitmapset *
GetOptimalGpusForRelation(PlannerInfo *root, RelOptInfo *rel)
{
	RangeTblEntry *rte;
	HeapTuple	tup;
	char		relpersistence;
	const Bitmapset *optimal_gpus;
#if 0
	if (baseRelIsArrowFdw(rel))
	{
		if (pgstrom_gpudirect_enabled())
			return GetOptimalGpusForArrowFdw(root, rel);
		return NULL;
	}
#endif
	optimal_gpus = GetOptimalGpusForTablespace(rel->reltablespace);
	if (!bms_is_empty(optimal_gpus))
	{
		/* only permanent / unlogged table can use NVMe-Strom */
		rte = root->simple_rte_array[rel->relid];
		tup = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "cache lookup failed for relation %u", rte->relid);
		relpersistence = ((Form_pg_class) GETSTRUCT(tup))->relpersistence;
		ReleaseSysCache(tup);

		if (relpersistence == RELPERSISTENCE_PERMANENT ||
			relpersistence == RELPERSISTENCE_UNLOGGED)
			return optimal_gpus;
	}
	return NULL;
}

/*
 * baseRelCanUseGpuDirect - checks wthere the relation can use GPU-Direct SQL.
 * If possible, it returns a bitmap of optimal GPUs.
 */
static size_t
__total_partitioned_relpages(PlannerInfo *root, Index relid)
{
	RelOptInfo *rel = root->simple_rel_array[relid];
	size_t		total_pages = 0;
	ListCell   *lc;

	if (rel && rel->reloptkind == RELOPT_BASEREL)
		return rel->pages;
	foreach (lc, root->append_rel_list)
	{
		AppendRelInfo *appinfo = (AppendRelInfo *) lfirst(lc);

		if (appinfo->parent_relid == relid)
			total_pages += __total_partitioned_relpages(root, appinfo->child_relid);
	}
	return total_pages;
}

const Bitmapset *
baseRelCanUseGpuDirect(PlannerInfo *root, RelOptInfo *baserel)
{
	const Bitmapset *optimal_gpus;
	size_t		total_scan_pages;

	if (!pgstrom_gpudirect_enabled())
		return NULL;

	optimal_gpus = GetOptimalGpusForTablespace(baserel->reltablespace);
	if (!optimal_gpus)
		return NULL;

	/*
	 * Check expected amount of the scan i/o.
	 * If 'baserel' is children of partition table, threshold shall be
	 * checked towards the entire partition size, because the range of
	 * child tables fully depend on scan qualifiers thus variable time
	 * by time. Once user focus on a particular range, but he wants to
	 * focus on other area. It leads potential thrashing on i/o.
	 */
	if (baserel->reloptkind == RELOPT_BASEREL)
	{
		total_scan_pages = baserel->pages;
	}
	else if (baserel->reloptkind == RELOPT_OTHER_MEMBER_REL)
	{
		Index		curr_relid = baserel->relid;
		ListCell   *lc;

		do {
			foreach (lc, root->append_rel_list)
			{
				AppendRelInfo *appinfo = (AppendRelInfo *) lfirst(lc);

				if (curr_relid == appinfo->child_relid)
				{
					curr_relid = appinfo->parent_relid;
					break;
				}
			}
		} while (lc != NULL);
		total_scan_pages = __total_partitioned_relpages(root, curr_relid);
	}
	else
	{
		/* elsewhere, not possible to use GPU-Direct SQL */
		return NULL;
	}

	if (total_scan_pages >= pgstrom_gpudirect_threshold() / BLCKSZ)
		return optimal_gpus;
	return NULL;
}





void
pgstrom_init_relscan(void)
{
	static char *nvme_manual_distance_map = NULL;
	char	buffer[1280];
	int		index = 0;

	/* pg_strom.enable_brin */
	DefineCustomBoolVariable("pg_strom.enable_brin",
							 "Enables to use BRIN-index",
							 NULL,
							 &pgstrom_enable_brin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/*
	 * pg_strom.nvme_distance_map
	 *
	 * config := <token>[,<token>...]
	 * token  := nvmeXX:gpuXX
	 *
	 * eg) nvme0:gpu0,nvme1:gpu1
	 */
	DefineCustomStringVariable("pg_strom.nvme_distance_map",
							   "Manual configuration of optimal GPU for each NVME",
							   NULL,
							   &nvme_manual_distance_map,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	extraSysfsSetupDistanceMap(nvme_manual_distance_map);
	while (extraSysfsPrintNvmeInfo(index, buffer, sizeof(buffer)) >= 0)
	{
		elog(LOG, "- %s", buffer);
		index++;
	}
	/* hash table for tablespace <-> optimal GPU */
	tablespace_optimal_xpu_htable = NULL;
	CacheRegisterSyscacheCallback(TABLESPACEOID,
								  tablespace_optimal_gpu_cache_callback,
								  (Datum) 0);
}
