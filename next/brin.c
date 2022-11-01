/*
 * brin.c
 *
 * Routines to support BRIN index
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "access/brin_revmap.h"
#include "executor/nodeIndexscan.h"

/* Data structure for collecting qual clauses that match an index */
typedef struct
{
	bool		nonempty;	/* True if lists are not all empty */
	/* Lists of RestrictInfos, one per index column */
	List	   *indexclauses[INDEX_MAX_KEYS];
} IndexClauseSet;

static bool		pgstrom_enable_brin;

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
estimateBrinIndexScanNBlocks(PlannerInfo *root,
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
IndexOptInfo *
pgstromTryFindBrinIndex(PlannerInfo *root,
						RelOptInfo *baserel,
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
        nblocks = estimateBrinIndexScanNBlocks(root, baserel,
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
		*p_indexConds = extract_index_conditions(indexQuals, indexOpt);
		*p_indexQuals = extract_actual_clauses(indexQuals, false);
		*p_indexNBlocks = indexNBlocks;
	}
	return indexOpt;
}

/*
 * cost_brin_bitmap_build
 */
Cost
cost_brin_bitmap_build(PlannerInfo *root,
					   RelOptInfo *baserel,
					   IndexOptInfo *indexOpt,
					   List *indexQuals)
{
	BrinStatsData	statsData;
	Relation		indexRel;
	Cost			index_build_cost;
	double			index_nitems;
	double			spc_rand_page_cost;
	double			spc_seq_page_cost;
	ListCell	   *lc;

	indexRel = index_open(indexOpt->indexoid, AccessShareLock);
	brinGetStats(indexRel, &statsData);
	index_close(indexRel, AccessShareLock);

	get_tablespace_page_costs(indexOpt->reltablespace,
							  &spc_rand_page_cost,
							  &spc_seq_page_cost);
	index_build_cost = spc_rand_page_cost * statsData.revmapNumPages;
	index_nitems = ceil(baserel->pages / (double)statsData.pagesPerRange);
	foreach (lc, indexQuals)
	{
		Node	   *qual = lfirst(lc);
		QualCost	qcost;

		cost_qual_eval_node(&qcost, qual, root);
		index_build_cost += (qcost.startup +
							 qcost.per_tuple * index_nitems);
	}
	return index_build_cost;
}


typedef struct
{
	volatile int	build_status;
	slock_t			lock;	/* once 'build_status' is set, no need to take
							 * this lock again. */
	pg_atomic_uint32 index;
	uint32_t		nitems;
	BlockNumber		chunks[FLEXIBLE_ARRAY_MEMBER];
} BrinIndexResults;

struct BrinIndexState
{
	Relation		index_rel;
	BlockNumber		nblocks;
	BlockNumber		nchunks;
	BlockNumber		pagesPerRange;
	BrinRevmap	   *brinRevmap;
	BrinDesc	   *brinDesc;
	ScanKeyData	   *ScanKeys;
	int				NumScanKeys;
	IndexRuntimeKeyInfo *RuntimeKeys;
	int				NumRuntimeKeys;
	bool			RuntimeKeysIsReady;
	ExprContext	   *RuntimeExprContext;
	BrinIndexResults *brinResults;
	uint32_t		curr_chunk_id;
	uint32_t		curr_block_id;
	TBMIterateResult tbmres;	/* must be tail */
};

/*
 * BrinIndexExecBegin
 */
void
pgstromBrinIndexExecBegin(pgstromTaskState *pts,
						  Oid index_oid,
						  List *index_conds,
						  List *index_quals)
{
	/* see ExecInitBitmapIndexScan */
	EState		   *estate = pts->css.ss.ps.state;
	Relation		relation = pts->css.ss.ss_currentRelation;
	Index			scanrelid = ((Scan *)pts->css.ss.ps.plan)->scanrelid;
	LOCKMODE		lockmode = NoLock;
	BrinIndexState *br_state;

	if (!OidIsValid(index_oid))
	{
		Assert(index_conds == NIL && index_quals == NIL);
		return;
	}
	br_state = palloc0(offsetof(BrinIndexState, tbmres.offsets) +
					   sizeof(BlockNumber) * MaxHeapTuplesPerPage);
	/*
	 * open the index relation
	 */
	lockmode = exec_rt_fetch(scanrelid, estate)->rellockmode;
	br_state->index_rel = index_open(index_oid, lockmode);
	br_state->brinRevmap = brinRevmapInitialize(br_state->index_rel,
												&br_state->pagesPerRange,
												estate->es_snapshot);
	br_state->brinDesc = brin_build_desc(br_state->index_rel); 
	br_state->nblocks = RelationGetNumberOfBlocks(relation);
	br_state->nchunks = (br_state->nblocks +
						 br_state->pagesPerRange - 1) / br_state->pagesPerRange;
	br_state->curr_chunk_id = 0;
	br_state->curr_block_id = UINT_MAX;

	/*
	 * build the index scan keys from the index conditions
	 */
	ExecIndexBuildScanKeys(&pts->css.ss.ps,
						   br_state->index_rel,
						   index_conds,
						   false,
						   &br_state->ScanKeys,
						   &br_state->NumScanKeys,
						   &br_state->RuntimeKeys,
						   &br_state->NumRuntimeKeys,
						   NULL, NULL);
	Assert(br_state->NumScanKeys >= br_state->NumRuntimeKeys);
	if (br_state->NumRuntimeKeys != 0)
	{
		ExprContext	   *econtext_saved = pts->css.ss.ps.ps_ExprContext;

		ExecAssignExprContext(estate, &pts->css.ss.ps);
		br_state->RuntimeExprContext = pts->css.ss.ps.ps_ExprContext;
		pts->css.ss.ps.ps_ExprContext = econtext_saved;
	}
	else
	{
		br_state->RuntimeExprContext = NULL;
	}
	pts->br_state = br_state;
}

/*
 * BrinIndexExecReset
 */
void
pgstromBrinIndexExecReset(pgstromTaskState *pts)
{
	/* See, ExecReScanBitmapIndexScan */
	BrinIndexState *br_state = pts->br_state;

	if (br_state->NumRuntimeKeys != 0)
	{
		ExprContext	*econtext = br_state->RuntimeExprContext;

		ResetExprContext(econtext);
		ExecIndexEvalRuntimeKeys(econtext,
								 br_state->RuntimeKeys,
								 br_state->NumRuntimeKeys);
	}
	br_state->RuntimeKeysIsReady = false;

	br_state->curr_chunk_id = 0;
	br_state->curr_block_id = UINT_MAX;
}

/*
 * check_null_keys from access/brin/brin.c
 */
static bool
check_null_keys(BrinValues *bval, ScanKey *nullkeys, int nnullkeys)
{
	int		keyno;

	/*
	 * First check if there are any IS [NOT] NULL scan keys, and if we're
	 * violating them.
	 */
	for (keyno = 0; keyno < nnullkeys; keyno++)
	{
		ScanKey		key = nullkeys[keyno];

		Assert(key->sk_attno == bval->bv_attno);

		/* Handle only IS NULL/IS NOT NULL tests */
		if (!(key->sk_flags & SK_ISNULL))
			continue;

		if (key->sk_flags & SK_SEARCHNULL)
		{
			/* IS NULL scan key, but range has no NULLs */
			if (!bval->bv_allnulls && !bval->bv_hasnulls)
				return false;
		}
		else if (key->sk_flags & SK_SEARCHNOTNULL)
		{
			/*
			 * For IS NOT NULL, we can only skip ranges that are known to have
			 * only nulls.
			 */
			if (bval->bv_allnulls)
				return false;
		}
		else
		{
			/*
			 * Neither IS NULL nor IS NOT NULL was used; assume all indexable
			 * operators are strict and thus return false with NULL value in
			 * the scan key.
			 */
			return false;
		}
	}
	return true;
}

/*
 * __BrinIndexExecBuildResults
 */
static void
__BrinIndexExecBuildResults(pgstromTaskState *pts)
{
	/* see bringetbitmap() */
	EState		   *estate = pts->css.ss.ps.state;
	BrinIndexState *br_state = pts->br_state;
	BrinIndexResults *br_results = br_state->brinResults;
	BrinDesc	   *bdesc = br_state->brinDesc;
	TupleDesc		bd_tupdesc = bdesc->bd_tupdesc;
	Buffer			buffer = InvalidBuffer;
	FmgrInfo	   *consistentFn;
	MemoryContext	oldcxt;
	MemoryContext	per_range_cxt;
	BrinMemTuple   *dtup;
	ScanKey		  **keys;
	ScanKey		  **nullkeys;
	int			   *nkeys;
	int			   *nnullkeys;
	int				j, keyno;
	uint32_t		chunk_id;

	/*
	 * Make room for the consistent support procedures of indexed columns.  We
	 * don't look them up here; we do that lazily the first time we see a scan
	 * key reference each of them.  We rely on zeroing fn_oid to InvalidOid.
	 */
	consistentFn = alloca(sizeof(FmgrInfo) * bd_tupdesc->natts);
	memset(consistentFn, 0, sizeof(FmgrInfo) * bd_tupdesc->natts);

	/*
	 * Make room for per-attribute lists of scan keys that we'll pass to the
	 * consistent support procedure. We don't know which attributes have scan
	 * keys, so we allocate space for all attributes. That may use more memory
	 * but it's probably cheaper than determining which attributes are used.
	 */
	keys = alloca(sizeof(ScanKey *) * bd_tupdesc->natts);
	nullkeys = alloca(sizeof(ScanKey *) * bd_tupdesc->natts);
	nkeys = alloca(sizeof(int) * bd_tupdesc->natts);
	nnullkeys = alloca(sizeof(int) * bd_tupdesc->natts);
	for (j=0; j < bd_tupdesc->natts; j++)
	{
		keys[j] = alloca(sizeof(ScanKey) * br_state->NumScanKeys);
		nullkeys[j] = alloca(sizeof(ScanKey) * br_state->NumScanKeys);
	}
	memset(nkeys, 0, sizeof(int) * bd_tupdesc->natts);
	memset(nnullkeys, 0, sizeof(int) * bd_tupdesc->natts);

	/* Preprocess the scan keys - split them into per-attribute arrays. */
	for (keyno=0; keyno < br_state->NumScanKeys; keyno++)
	{
		ScanKey		key = &br_state->ScanKeys[keyno];
		AttrNumber	keyattno = key->sk_attno;

		/* the collation must mutually match */
		Assert((key->sk_flags & SK_ISNULL) ||
			   (key->sk_collation == TupleDescAttr(bd_tupdesc,
												   keyattno - 1)->attcollation));

		/* First time we see this index attribute, so init as needed. */
		if (consistentFn[keyattno-1].fn_oid == InvalidOid)
		{
			FmgrInfo   *tmp;

			Assert(nkeys[keyattno-1] == 0 &&
				   nnullkeys[keyattno-1] == 0);
			tmp = index_getprocinfo(br_state->index_rel, keyattno,
									BRIN_PROCNUM_CONSISTENT);
			fmgr_info_copy(&consistentFn[keyattno-1], tmp,
						   CurrentMemoryContext);
		}

		/* Add key to the proper per-attribute array. */
		if (key->sk_flags & SK_ISNULL)
		{
			int		idx = nnullkeys[keyattno-1]++;
			
			nullkeys[keyattno-1][idx] = key;
		}
		else
		{
			int		idx = nkeys[keyattno-1]++;

			keys[keyattno-1][idx] = key;
		}
	}
	/* allocate an initial in-memory tuple, out of the per-range memcxt */
	dtup = brin_new_memtuple(bdesc);

	/* setup and switch a per-range memory context */
	per_range_cxt = AllocSetContextCreate(CurrentMemoryContext,
										  "__ExecBuildBrinIndexResults working",
										  ALLOCSET_DEFAULT_SIZES);
	oldcxt = MemoryContextSwitchTo(per_range_cxt);

	/*
	 * Now scan the revmap.  We start by querying for heap page 0,
	 * incrementing by the number of pages per range; this gives us a full
	 * view of the table.
	 */
	br_results->nitems = 0;
	for (chunk_id = 0; chunk_id < br_state->nchunks; chunk_id++)
	{
		BrinTuple  *__btup;
		BrinTuple  *btup = NULL;
		Size		btupsz = 0;
		OffsetNumber off;
		Size		size;
		bool		addrange = true;

		CHECK_FOR_INTERRUPTS();

		MemoryContextResetAndDeleteChildren(per_range_cxt);

		__btup = brinGetTupleForHeapBlock(br_state->brinRevmap,
										  chunk_id * br_state->pagesPerRange,
										  &buffer,
										  &off,
										  &size,
										  BUFFER_LOCK_SHARE,
										  estate->es_snapshot);
		if (!__btup)
			goto skip;
		btup = brin_copy_tuple(__btup, size, btup, &btupsz);
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);

		dtup = brin_deform_tuple(bdesc, btup, dtup);
		if (dtup->bt_placeholder)
			goto skip;
		/*
		 * Compare scan keys with summary values stored for the range.
		 * If scan keys are matched, the page range must be added to
		 * the bitmap.  We initially assume the range needs to be
		 * added; in particular this serves the case where there are
		 * no keys.
		 */
		for (j=0; j < bd_tupdesc->natts; j++)
		{
			BrinValues *bval;
			Datum		add;
			Oid			collation;

			 /*
			  * skip attributes without any scan keys (both regular and
			  * IS [NOT] NULL)
			  */
			if (nkeys[j] == 0 && nnullkeys[j] == 0)
				continue;

			bval = &dtup->bt_columns[j];

			/*
			  * First check if there are any IS [NOT] NULL scan keys,
			  * and if we're violating them. In that case we can
			  * terminate early, without invoking the support function.
			  */
			if (bdesc->bd_info[j]->oi_regular_nulls &&
				!check_null_keys(bval, nullkeys[j], nnullkeys[j]))
			{
				addrange = false;
				break;
			}

			/*
			 * So either there are no IS [NOT] NULL keys, or all
			 * passed. If there are no regular scan keys, we're done -
			 * the page range matches. If there are regular keys, but
			 * the page range is marked as 'all nulls' it can't
			 * possibly pass (we're assuming the operators are
			 * strict).
			 */
			if (!nkeys[j])
				continue;
			Assert(nkeys[j] > 0 && nkeys[j] <= br_state->NumScanKeys);

			/* If it is all nulls, it cannot possibly be consistent. */
			if (bval->bv_allnulls)
			{
				addrange = false;
				break;
			}

			
			/*
			 * Collation from the first key (has to be the same for
			 * all keys for the same attribute).
			 */
			collation = keys[j][0]->sk_collation;

			/*
			 * Check whether the scan key is consistent with the page
			 * range values; if so, have the pages in the range added
			 * to the output bitmap.
			 */
			if (consistentFn[j].fn_nargs >= 4)
			{
				/* Check all keys at once */
				add = FunctionCall4Coll(&consistentFn[j],
										collation,
										PointerGetDatum(bdesc),
										PointerGetDatum(bval),
										PointerGetDatum(keys[j]),
										Int32GetDatum(nkeys[j]));
				addrange = DatumGetBool(add);
			}
			else
			{
				/*
				 * Check keys one by one
				 *
				 * When there are multiple scan keys, failure to meet
				 * the criteria for a single one of them is enough to
				 * discard the range as a whole, so break out of the
				 * loop as soon as a false return value is obtained.
				 */
				for (keyno = 0; keyno < nkeys[j]; keyno++)
				{
					add = FunctionCall3Coll(&consistentFn[j],
											keys[j][keyno]->sk_collation,
											PointerGetDatum(bdesc),
											PointerGetDatum(bval),
											PointerGetDatum(keys[j][keyno]));
					addrange = DatumGetBool(add);
					if (!addrange)
						break;
				}
			}
		}
	skip:
		if (addrange)
		{
			int		idx = br_results->nitems++;

			br_results->chunks[idx] = chunk_id;
		}
	}
	MemoryContextSwitchTo(oldcxt);
	MemoryContextDelete(per_range_cxt);

	if (buffer != InvalidBuffer)
		ReleaseBuffer(buffer);
	pg_memory_barrier();
	br_results->build_status = 1;
}

static inline BrinIndexResults *
__BrinIndexGetResults(pgstromTaskState *pts)
{
	BrinIndexState *br_state = pts->br_state;
	BrinIndexResults *br_results;

	/*
	 * At the first call of pgstromBrinIndexNextXXXX() at the single process
	 * execution, 'brinResults' shall not be allocated yet. So, allocate it
	 * on the local memory.
	 */
	if (!br_state->brinResults)
		pgstromBrinIndexInitDSM(pts, NULL);

	if (br_state->NumRuntimeKeys != 0 &&
		!br_state->RuntimeKeysIsReady)
		pgstromBrinIndexExecReset(pts);

	br_results = br_state->brinResults;
	if (br_results->build_status <= 0)
	{
		SpinLockAcquire(&br_results->lock);
		PG_TRY();
		{
			if (br_results->build_status == 0)
				__BrinIndexExecBuildResults(pts);
			else if (br_results->build_status < 0)
				elog(ERROR, "failed on __BrinIndexExecBuildResults by other workers");
		}
		PG_CATCH();
		{
			br_results->build_status = -1;
			SpinLockRelease(&br_results->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		Assert(br_results->build_status > 0);
		SpinLockRelease(&br_results->lock);
	}
	return br_results;
}

TBMIterateResult *
pgstromBrinIndexNextBlock(pgstromTaskState *pts)
{
	BrinIndexState *br_state = pts->br_state;
	BrinIndexResults *br_results = __BrinIndexGetResults(pts);
	uint32_t		index;
	BlockNumber		blockno;

	if (br_state->curr_block_id >= br_state->pagesPerRange)
	{
		index = pg_atomic_fetch_add_u32(&br_results->index, 1);
		if (index >= br_results->nitems)
			return NULL;
		br_state->curr_chunk_id = br_results->chunks[index];
		br_state->curr_block_id = 0;
	}
	blockno = (br_state->curr_chunk_id * br_state->pagesPerRange +
			   br_state->curr_block_id++);
	if (blockno >= br_state->nblocks)
		return NULL;

	br_state->tbmres.blockno = blockno;
	br_state->tbmres.ntuples = -1;
	br_state->tbmres.recheck = true;
	return &br_state->tbmres;
}

bool
pgstromBrinIndexNextChunk(pgstromTaskState *pts)
{
	BrinIndexState *br_state = pts->br_state;
	BrinIndexResults *br_results = __BrinIndexGetResults(pts);
	uint32_t		index;

	index = pg_atomic_fetch_add_u32(&br_results->index, 1);
	if (index < br_results->nitems)
	{
		BlockNumber	pagesPerRange = br_state->pagesPerRange;

		pts->curr_block_num  = br_results->chunks[index] * pagesPerRange;
		pts->curr_block_tail = pts->curr_block_num + pagesPerRange;
		if (pts->curr_block_num >= br_state->nblocks)
			return false;
		if (pts->curr_block_tail > br_state->nblocks)
			pts->curr_block_tail = br_state->nblocks;
		return true;
	}
	return false;
}

void
pgstromBrinIndexExecEnd(pgstromTaskState *pts)
{
	BrinIndexState *br_state = pts->br_state;

	if (br_state->brinRevmap)
		brinRevmapTerminate(br_state->brinRevmap);
	if (br_state->brinDesc)
		brin_free_desc(br_state->brinDesc);
	if (br_state->index_rel)
		index_close(br_state->index_rel, NoLock);
}

Size
pgstromBrinIndexEstimateDSM(pgstromTaskState *pts)
{
	BrinIndexState *br_state = pts->br_state;

	return MAXALIGN(offsetof(BrinIndexResults, chunks[br_state->nchunks]));
}

Size
pgstromBrinIndexInitDSM(pgstromTaskState *pts, char *dsm_addr)
{
	BrinIndexState *br_state = pts->br_state;
	BrinIndexResults *br_results;
	Size		dsm_len = 0;

	dsm_len = MAXALIGN(offsetof(BrinIndexResults,
								chunks[br_state->nchunks]));
	if (dsm_addr)
		br_results = (BrinIndexResults *)dsm_addr;
	else
	{
		EState	   *estate = pts->css.ss.ps.state;

		br_results = MemoryContextAlloc(estate->es_query_cxt, dsm_len);
	}
	memset(br_results, 0, offsetof(BrinIndexResults, chunks));
	SpinLockInit(&br_results->lock);

	br_state->brinResults = br_results;

	return (dsm_addr ? dsm_len : 0);
}

void
pgstromBrinIndexReInitDSM(pgstromTaskState *pts)
{
	BrinIndexState *br_state = pts->br_state;
	BrinIndexResults *br_results = br_state->brinResults;

	br_results->build_status = 0;
	br_results->nitems   = 0;
	pg_atomic_init_u32(&br_results->index, 0);
}

Size
pgstromBrinIndexAttachDSM(pgstromTaskState *pts, char *dsm_addr)
{
	BrinIndexState *br_state = pts->br_state;

	br_state->brinResults = (BrinIndexResults *)dsm_addr;
	return MAXALIGN(offsetof(BrinIndexResults,
							 chunks[br_state->nchunks]));
}

void
pgstromBrinIndexShutdownDSM(pgstromTaskState *pts)
{
	/* nothing to do */
}

void
pgstrom_init_brin(void)
{
	/* pg_strom.enable_brin */
	DefineCustomBoolVariable("pg_strom.enable_brin",
							 "Enables to use BRIN-index",
							 NULL,
							 &pgstrom_enable_brin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}
