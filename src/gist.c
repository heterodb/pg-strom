/*
 * gist.c
 *
 * Routines to support BRIN index
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "nodes/supportnodes.h"

/*
 * get_index_clause_from_support
 *
 * (from optimizer/path/indxpath.c)
 */
static IndexClause *
get_index_clause_from_support(PlannerInfo *root,
                              RestrictInfo *rinfo,
                              Oid funcid,
                              int indexarg,
                              int indexcol,
                              IndexOptInfo *index)
{
	Oid			prosupport = get_func_support(funcid);
	SupportRequestIndexCondition req;
	List	   *sresult;

	if (!OidIsValid(prosupport))
		return NULL;

	req.type = T_SupportRequestIndexCondition;
	req.root = root;
	req.funcid = funcid;
	req.node = (Node *) rinfo->clause;
	req.indexarg = indexarg;
	req.index = index;
	req.indexcol = indexcol;
	req.opfamily = index->opfamily[indexcol];
	req.indexcollation = index->indexcollations[indexcol];
	req.lossy = true;		/* default assumption */

	sresult = (List *)
		DatumGetPointer(OidFunctionCall1(prosupport,
										 PointerGetDatum(&req)));
	if (sresult != NIL)
	{
		IndexClause *iclause = makeNode(IndexClause);
		List	   *indexquals = NIL;
		ListCell   *lc;

		/*
		 * The support function API says it should just give back bare
		 * clauses, so here we must wrap each one in a RestrictInfo.
		 */
		foreach(lc, sresult)
		{
			Expr	   *clause = (Expr *) lfirst(lc);

			indexquals = lappend(indexquals,
								 make_simple_restrictinfo(root, clause));
		}
		iclause->rinfo = rinfo;
		iclause->indexquals = indexquals;
		iclause->lossy = req.lossy;
		iclause->indexcol = indexcol;
		iclause->indexcols = NIL;

		return iclause;
	}
	return NULL;
}

/*
 * match_opclause_to_indexcol
 */
static IndexClause *
match_opclause_to_indexcol(PlannerInfo *root,
                           RestrictInfo *rinfo,
                           IndexOptInfo *index,
                           int indexcol)
{
	IndexClause *iclause;
	OpExpr	   *op = (OpExpr *) rinfo->clause;
	Node	   *leftop;
	Node	   *rightop;
	Index		index_relid = index->rel->relid;
	Oid			index_collid;
	Oid			opfamily;

	/* only binary operators */
	if (list_length(op->args) != 2)
		return NULL;

	leftop = (Node *)linitial(op->args);
	rightop = (Node *)lsecond(op->args);
	opfamily = index->opfamily[indexcol];
	index_collid = index->indexcollations[indexcol];

	if (match_index_to_operand(leftop, indexcol, index) &&
		!bms_is_member(index_relid, rinfo->right_relids) &&
		!contain_volatile_functions(rightop))
	{
		if ((!OidIsValid(index_collid) || index_collid == op->inputcollid) &&
			op_in_opfamily(op->opno, opfamily))
		{
			iclause = makeNode(IndexClause);
			iclause->rinfo = rinfo;
			iclause->indexquals = list_make1(rinfo);
			iclause->lossy = false;
			iclause->indexcol = indexcol;
			iclause->indexcols = NIL;
			return iclause;
		}
		set_opfuncid(op);
		return get_index_clause_from_support(root,
											 rinfo,
											 op->opfuncid,
											 0,		/* indexarg on left */
											 indexcol,
											 index);
	}

	if (match_index_to_operand(rightop, indexcol, index) &&
		!bms_is_member(index_relid, rinfo->left_relids) &&
		!contain_volatile_functions(leftop))
	{
		Oid		comm_op = get_commutator(op->opno);

		if ((!OidIsValid(index_collid) || index_collid == op->inputcollid) &&
			op_in_opfamily(comm_op, opfamily))
		{
			RestrictInfo *commrinfo;

			commrinfo = commute_restrictinfo(rinfo, comm_op);

			iclause = makeNode(IndexClause);
			iclause->rinfo = rinfo;
			iclause->indexquals = list_make1(commrinfo);
			iclause->lossy = false;
			iclause->indexcol = indexcol;
			iclause->indexcols = NIL;
			return iclause;
		}
		set_opfuncid(op);
		return get_index_clause_from_support(root,
											 rinfo,
											 op->opfuncid,
											 1,		/* indexarg on right */
											 indexcol,
											 index);
	}
	return NULL;
}

/*
 * match_funcclause_to_indexcol
 */
static IndexClause *
match_funcclause_to_indexcol(PlannerInfo *root,
							 RestrictInfo *rinfo,
							 IndexOptInfo *index,
							 int indexcol)
{
	FuncExpr   *func = (FuncExpr *) rinfo->clause;
	int			indexarg = 0;
	ListCell   *lc;

	foreach (lc, func->args)
	{
		Node   *node = lfirst(lc);

		if (match_index_to_operand(node, indexcol, index))
		{
			return get_index_clause_from_support(root,
												 rinfo,
												 func->funcid,
												 indexarg,
												 indexcol,
												 index);
		}
		indexarg++;
	}
	return NULL;
}

/*
 * device index special access catalog
 */
#define __POSTGIS		"@postgis"
#define __GEOM			"geometry" __POSTGIS
#define __BOX2D			"box2df" __POSTGIS

static struct {
	/* geometry overlap operator */
	const char *orig_fname;
	const char *orig_left;
	const char *orig_right;
	const char *gist_fname;		/* index evaluation operator */
	const char *gist_left;
	const char *gist_right;
} devindex_catalog[] = {
	/* '&&' overlap operators */
	{
		"geometry_overlaps" __POSTGIS, __GEOM,  __GEOM,
		"overlaps_2d"       __POSTGIS, __BOX2D, __GEOM,
	},
	{
		"geometry_overlaps" __POSTGIS, __GEOM,  __GEOM,
		"overlaps_2d"       __POSTGIS, __GEOM, __BOX2D,
	},
	{
		"overlaps_2d"       __POSTGIS, __GEOM,  __BOX2D,
		"overlaps_2d"       __POSTGIS, __BOX2D, __BOX2D,
	},
	{
		"overlaps_2d"       __POSTGIS, __BOX2D,  __GEOM,
		"overlaps_2d"       __POSTGIS, __BOX2D, __BOX2D,
	},
	/* '~' contains operators */
	{
		"geometry_contains" __POSTGIS, __GEOM,  __GEOM,
		"contains_2d"       __POSTGIS, __BOX2D, __GEOM,
	},
	{
		"geometry_contains" __POSTGIS, __GEOM,  __GEOM,
		"contains_2d"       __POSTGIS, __GEOM, __BOX2D,
	},
	{
		"contains_2d"       __POSTGIS, __GEOM,  __BOX2D,
		"contains_2d"       __POSTGIS, __BOX2D, __BOX2D,
	},
	{
		"contains_2d"       __POSTGIS, __BOX2D,  __GEOM,
		"contains_2d"       __POSTGIS, __BOX2D, __BOX2D,
	},
	/* '@' within operators */
	{
		"geometry_within"   __POSTGIS, __GEOM,  __GEOM,
		"is_contained_2d"   __POSTGIS, __BOX2D, __GEOM,
	},
	{
		"geometry_within"   __POSTGIS, __GEOM,  __GEOM,
		"is_contained_2d"   __POSTGIS, __GEOM, __BOX2D,
	},
	{
		"is_contained_2d"   __POSTGIS, __GEOM,  __BOX2D,
		"is_contained_2d"   __POSTGIS, __BOX2D, __BOX2D,
	},
	{
		"is_contained_2d"   __POSTGIS, __BOX2D,  __GEOM,
		"is_contained_2d"   __POSTGIS, __BOX2D, __BOX2D,
	},
	{ NULL, NULL, NULL, NULL, NULL, NULL },
};

static char *
__get_type_signature(Oid type_oid)
{
	char   *type_name = get_type_name(type_oid, false);
	char   *ext_name = get_type_extension_name(type_oid);

	if (ext_name)
		type_name = psprintf("%s@%s", type_name, ext_name);
	return type_name;
}

static char *
__get_func_signature(Oid func_oid)
{
	char   *func_name = get_func_name(func_oid);
	char   *ext_name = get_func_extension_name(func_oid);

	if (ext_name)
		func_name = psprintf("%s@%s", func_name, ext_name);
	return func_name;
}

/*
 * fixup_gist_clause_for_device
 */
static Oid
__lookup_gist_device_function(Oid opno, Oid left_oid, Oid right_oid, Oid gist_oid)
{
	char	   *orig_fname;
	char	   *orig_left;
	char	   *orig_right;
	char	   *gist_left;
	char	   *gist_right;
	oidvector  *gist_argtypes;

	orig_fname = __get_func_signature(get_opcode(opno));
	orig_left  = __get_type_signature(left_oid);
	orig_right = __get_type_signature(right_oid);
	gist_left  = __get_type_signature(gist_oid);
	gist_right = orig_right;	/* unchanged */
	gist_argtypes = __buildoidvector2(left_oid, right_oid);

	/* lookup the catalog */
	for (int i=0; devindex_catalog[i].orig_fname != NULL; i++)
	{
		if (strcmp(orig_fname, devindex_catalog[i].orig_fname) == 0 &&
			strcmp(orig_left,  devindex_catalog[i].orig_left) == 0 &&
			strcmp(orig_right, devindex_catalog[i].orig_right) == 0 &&
			strcmp(gist_left,  devindex_catalog[i].gist_left) == 0 &&
			strcmp(gist_right, devindex_catalog[i].gist_right) == 0)
		{
			char   *func_name = pstrdup(devindex_catalog[i].gist_fname);
			char   *extname;
			Oid		func_oid = InvalidOid;

			extname = strchr(func_name, '@');
			if (extname)
				*extname++ = '\0';

			if (!extname)
			{
				func_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
										   Anum_pg_proc_oid,
										   CStringGetDatum(func_name),
										   PointerGetDatum(gist_argtypes),
										   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
			}
			else
			{
				CatCList   *catlist;

				catlist = SearchSysCacheList1(PROCNAMEARGSNSP,
											  CStringGetDatum(func_name));
				for (int k=0; k < catlist->n_members; k++)
				{
					HeapTuple	htup = &catlist->members[k]->tuple;
					Form_pg_proc proc = (Form_pg_proc) GETSTRUCT(htup);
					char	   *__extname = get_func_extension_name(proc->oid);

					if (__extname && strcmp(extname, __extname) == 0)
					{
						func_oid = proc->oid;
						break;
					}
				}
				ReleaseSysCacheList(catlist);
			}
			if (OidIsValid(func_oid))
				return func_oid;
		}
	}
	return InvalidOid;
}

static bool
fixup_gist_clause_for_device(PlannerInfo *root,
							 IndexOptInfo *index,
							 AttrNumber indexcol,
							 IndexClause *iclause,
							 Expr **p_clause,
							 Oid *p_func_oid)
{
	RestrictInfo *rinfo;
	OpExpr	   *op;
	Expr	   *left;
	Expr	   *right;
	Oid			left_oid;
	Oid			right_oid;
	Oid			gist_oid;
	Oid			func_oid;

	/* Not a supported index types? */
	if (iclause->indexcols != NIL)
		return false;
	if (list_length(iclause->indexquals) != 1)
		return false;

	/* fixup indexquals to reference the index-var */
	rinfo = linitial(iclause->indexquals);
	Assert(IsA(rinfo, RestrictInfo));
	op = (OpExpr *)rinfo->clause;
	if (!IsA(op, OpExpr) || list_length(op->args) != 2)
		return false;
	left = (Expr *)linitial(op->args);
	right = (Expr *)lsecond(op->args);

	gist_oid = get_atttype(index->indexoid, indexcol+1);
	if (match_index_to_operand((Node *)left, indexcol, index))
	{
		left_oid = exprType((Node *)left);
		right_oid = exprType((Node *)right);
	}
	else if (match_index_to_operand((Node *)right, indexcol, index))
	{
		/* swap left and right */
		op = (OpExpr *)make_opclause(get_commutator(op->opno),
									 op->opresulttype,
									 op->opretset,
									 right,
									 left,
									 op->opcollid,
									 op->inputcollid);
		left_oid = exprType((Node *)right);
		right_oid = exprType((Node *)left);
	}
	else
		return false;

	func_oid = __lookup_gist_device_function(op->opno,
											 left_oid,
											 right_oid,
											 gist_oid);
	if (OidIsValid(func_oid))
	{
		*p_clause = (Expr *)op;
		*p_func_oid = func_oid;
		return true;
	}
	return false;
}

/*
 * match_clause_to_gist_index
 *
 * MEMO: its logic is almost equivalent to match_join_clauses_to_index()
 */
static bool
match_clause_to_gist_index(PlannerInfo *root,
						   IndexOptInfo *index,
						   AttrNumber indexcol,
						   List *restrict_clauses,
						   uint32_t xpu_task_flags,
						   List *input_rels_tlist,
						   Expr **p_clause,
						   Oid *p_func_oid,
						   Selectivity *p_selectivity)
{
	RelOptInfo *heap_rel = index->rel;
	ListCell   *lc;
	Expr	   *clause = NULL;
	Oid			func_oid = InvalidOid;
	Selectivity	selectivity = 1.0;

	/* identify the restriction clauses that can match the index. */
	/* see, match_join_clauses_to_index */
	foreach (lc, restrict_clauses)
	{
		RestrictInfo *rinfo = lfirst(lc);
		IndexClause *iclause;
		Expr	   *__clause;
		Oid			__func_oid;
		Selectivity	__selectivity;

		if (rinfo->pseudoconstant)
			continue;
		if (!join_clause_is_movable_to(rinfo, heap_rel))
			continue;
		if (!rinfo->clause || restriction_is_or_clause(rinfo))
			continue;
		
		if (IsA(rinfo->clause, OpExpr))
			iclause = match_opclause_to_indexcol(root, rinfo, index, indexcol);
		else if (IsA(rinfo->clause, FuncExpr))
			iclause = match_funcclause_to_indexcol(root, rinfo, index, indexcol);
		else
			iclause = NULL;

		if (iclause &&
			fixup_gist_clause_for_device(root,
										 index,
										 indexcol,
										 iclause,
										 &__clause,
										 &__func_oid)
#if 1
			&&
			pgstrom_xpu_expression(__clause,
								   xpu_task_flags,
								   input_rels_tlist, NULL)
#endif
			)
		{
			__selectivity = clauselist_selectivity(root,
												   iclause->indexquals,
												   heap_rel->relid,
												   JOIN_INNER,
												   NULL);
			if (!clause || selectivity > __selectivity)
			{
				clause = __clause;
				func_oid = __func_oid;
				selectivity = __selectivity;
			}
		}
	}
	if (clause)
	{
		*p_clause = clause;
		*p_func_oid = func_oid;
		*p_selectivity = selectivity;
		return true;
	}
	return false;
}

/*
 * pgstromTryFindGistIndex
 *
 * its logic is almost equivalent to match_join_clauses_to_index()
 */
void
pgstromTryFindGistIndex(PlannerInfo *root,
						Path *inner_path,
						List *restrict_clauses,
						uint32_t xpu_task_flags,
						List *input_rels_tlist,
						pgstromPlanInnerInfo *pp_inner)
{

	RelOptInfo	   *inner_rel = inner_path->parent;
	IndexOptInfo   *gist_index = NULL;
	AttrNumber		gist_index_col = InvalidAttrNumber;
	Oid				gist_func_oid = InvalidOid;
	Expr		   *gist_clause = NULL;
	Expr		   *gist_clause_fallback = NULL;
	Selectivity		gist_selectivity = 1.0;
	ListCell	   *lc;

	/*
	 * Not only GiST, index should be built on normal relations.
	 * And, IndexOnlyScan may not contain CTID, so not supported.
	 */
	Assert(pp_inner->hash_outer_keys == NIL &&
		   pp_inner->hash_inner_keys == NIL);
	if (!IS_SIMPLE_REL(inner_rel) || inner_path->pathtype == T_IndexOnlyScan)
		return;
	/* see the logic in create_index_paths */
	foreach (lc, inner_rel->indexlist)
	{
		IndexOptInfo *curr_index = (IndexOptInfo *) lfirst(lc);
		int		nkeycolumns = curr_index->nkeycolumns;

		Assert(curr_index->rel == inner_rel);
		/* only GiST index is supported  */
		if (curr_index->relam != GIST_AM_OID)
			continue;
		/* ignore partial indexes */
		if (curr_index->indpred != NIL)
			continue;

		for (int indexcol=0; indexcol < nkeycolumns; indexcol++)
		{
			Selectivity curr_selectivity = 1.0;
			Expr	   *curr_clause = NULL;
			Oid			curr_func_oid = InvalidOid;

			if (match_clause_to_gist_index(root,
										   curr_index,
										   indexcol,
										   restrict_clauses,
										   xpu_task_flags,
										   input_rels_tlist,
										   &curr_clause,
										   &curr_func_oid,
										   &curr_selectivity) &&
				(!gist_index || gist_selectivity > curr_selectivity))
			{
				gist_index           = curr_index;
				gist_index_col       = indexcol;
				gist_func_oid        = curr_func_oid;
				gist_clause          = curr_clause;
				gist_selectivity     = curr_selectivity;
			}
		}
	}
	if (!gist_index)
		return;
	
	elog(DEBUG2, "GiST Index (oid: %u, col: %d, sel: %.4f), CLAUSE => %s, FALLBACK => %s",
		 gist_index->indexoid, gist_index_col, gist_selectivity,
		 nodeToString(gist_clause),
		 nodeToString(gist_clause_fallback));
	
	/* store the result if any */
	pp_inner->gist_index_oid       = gist_index->indexoid;
	pp_inner->gist_index_col       = gist_index_col;
	pp_inner->gist_func_oid        = gist_func_oid;
	pp_inner->gist_slot_id         = -1;	/* to be set later */
	pp_inner->gist_clause          = gist_clause;
	pp_inner->gist_selectivity     = gist_selectivity;
	pp_inner->gist_npages          = gist_index->pages;
	pp_inner->gist_height          = gist_index->tree_height;
}
