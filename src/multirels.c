/*
 * multirels.c
 *
 * Inner relations loader for GpuJoin
 * ----
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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
#include "pg_strom.h"
#include "cuda_gpujoin.h"

/* static variables */
static CustomScanMethods	multirels_plan_methods;
static PGStromExecMethods	multirels_exec_methods;

/*
 * MultiRelsInfo - state object of CustomScan(MultiRels)
 */
typedef struct
{
	int			depth;		/* depth of this inner relation */
	int			nbatches;	/* expected number of batches */
	Size		total_size;
	double		proportion;

	/* width of hash-slot if hash-join case */
	cl_uint		nslots;
    /*
     * NOTE: setrefs.c adjusts varnode reference on hash_keys because
     * of custom-scan interface contract. It shall be redirected to
     * INDEX_VAR reference, to reference pseudo-scan tlist.
     * MultiHash has idential pseudo-scan tlist with its outer scan-
     * path, it always reference correct attribute as long as tuple-
     * slot is stored on ecxt_scantuple of ExprContext.
     */
	List	   *hash_keys;
} MultiRelsInfo;

/*
 * MultiRelsState - execution state object of CustomScan(MultiRels)
 */
typedef struct
{
	CustomScanState	css;
	GpuContext	   *gcontext;
	int				depth;
	int				nbatches_plan;
	int				nbatches_exec;
	Size			total_size;		/* entire kern_multirels buffer */
	double			proportion;		/* 0.00 - 1.00 towards total size */

	TupleTableSlot *outer_overflow;
	bool			outer_done;

	void		   *curr_chunk;

	/* for hash-join, below */
	cl_uint			nslots;
	List		   *hash_keys;
	List		   *hash_keylen;
	List		   *hash_keybyval;
	List		   *hash_keytype;
} MultiRelsState;

/*
 * form_multirels_info
 */
static void
form_multirels_info(CustomScan *cscan, MultiRelsInfo *mr_info)
{
	List	   *exprs = NIL;
	List	   *privs = NIL;

	privs = lappend(privs, makeInteger(mr_info->depth));
	privs = lappend(privs, makeInteger(mr_info->logic));
	privs = lappend(privs, makeInteger(mr_info->nbatches));
	privs = lappend(privs, makeInteger(mr_info->total_size));
	privs = lappend(privs, makeInteger((long)(mr_info->proportion * 1000000.0)));
	privs = lappend(privs, makeInteger(mr_info->nslots));
	exprs = lappend(exprs, mr_info->hash_keys);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

/*
 * deform_multirels_info
 */
static MultiRelsInfo *
deform_multirels_info(CustomScan *cscan)
{
	MultiRelsInfo  *mr_info = palloc0(sizeof(MultiRelsInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;

	mr_info->depth = intVal(list_nth(privs, pindex++));
	mr_info->logic = intVal(list_nth(privs, pindex++));
	mr_info->nbatches = intVal(list_nth(privs, pindex++));
	mr_info->total_size = intVal(list_nth(privs, pindex++));
	mr_info->proportion = (double)intVal(list_nth(privs, pindex++)) / 1000000.0;
	mr_info->nslots = intVal(list_nth(privs, pindex++));
	mr_info->hash_keys = list_nth(exprs, eindex++);

	return mr_info;
}

/*
 * pgstrom_plan_is_multirels
 */
bool
pgstrom_plan_is_multirels(const Plan *plan)
{
	CustomScan *cscan = (CustomScan *) plan;

	if (IsA(cscan, CustomScan) &&
		cscan->methods == &multirels_plan_methods)
		return true;
	return false;
}

/*
 * pgstrom_create_multirels_plan
 *
 *
 */
CustomScan *
pgstrom_create_multirels_plan(PlannerInfo *root, int depth, Path *outer_path,
							  int nbatches, Size buffer_size, double threshold,
							  int nslots, List *hash_keys)
{
	CustomScan	   *cscan;
	MultiRelsInfo	mr_info;
	Path		   *outer_plan = create_plan_recurse(root, outer_path);

	cscan = makeNode(CustomScan);
	cscan->scan.plan.startup_cost = outer_plan->startup_cost;
	cscan->scan.plan.total_cost = (outer_plan->total_cost +
								   outer_plan->plan_rows * cpu_tuple_cost);
	if (hash_keys != NIL)
		cscan->scan.plan.total_cost += (outer_plan->plan_rows *
										cpu_operator_cost *
										list_length(hash_keys));
	cscan->scan.plan.plan_rows = outer_plan->plan_rows;
	cscan->scan.plan.plan_width = outer_plan->plan_width;
	cscan->scan.plan.targetlist = outer_plan->targetlist;
	cscan->scan.plan.qual = NIL;
	cscan->scan.scanrelid = 0;
	cscan->flags = 0;
	cscan->custom_ps_tlist = copyObject(outer_plan->targetlist);
	cscan->custom_relids = NULL;
	cscan->methods = &multirels_plan_methods;
	outerPlan(cscan) = outer_plan;

	memset(&mr_info, 0, sizeof(MultiRelsInfo));
	mr_info.depth = depth;
	mr_info.logic = (hash_keys != NIL
					 ? GPUJOIN_LOGIC_HASHJOIN
					 : GPUJOIN_LOGIC_NESTLOOP);
	mr_info.nbatches = nbatches;
	mr_info.buffer_size = buffer_size;
	mr_info.threshold = threshold;
	mr_info.nslots = nslots;
	mr_info.hash_keys = hash_keys;
	form_multirels_info(mrels, &mr_info);

	return cscan;
}


static void
multirels_create_scan_state(CustomScan *cscan)
{
	MultiRelsState *mrs = palloc0(sizeof(MultiRelsState));

	NodeSetTag(mhs, T_CustomScanState);
	mrs->css.flags = cscan->flags;
	mrs->css.methods = &multirels_exec_methods.c;
	mrs->gcontext = pgstrom_get_gpucontext();

	return (Node *) mrs;
}


static void
multirels_begin(CustomScanState *node, EState *estate, int eflags)
{
	MultiRelsState *mrs = (MultiRelsState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	MultiRelsInfo  *mr_info = deform_multirels_info(cscan);

	/* check for unsupported flags */
	Assert(!(eflags & (EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)));
	/* ensure the plan is MultiHash */
	Assert(pgstrom_plan_is_multirels((Plan *) cscan));

	mrs->depth = mr_info->depth;
	mrs->nbatches_plan = mr_info->nbatches;
	mrs->nbatches_exec = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0 ? -1 : 0);
	mrs->total_size = mr_info->total_size;
	mrs->proportion = mr_info->proportion;
	mrs->nslots = mr_info->nslots;

	mrs->outer_overflow = NULL;
	mrs->outer_done = NULL;
	mrs->curr_chunk = NULL;

	/*
	 * initialize the expression state of hash-keys, if any
	 */
	foreach (lc, mr_info->hash_keys)
	{
		Oid		type_oid = exprType(lfirst(lc));
		int16	type_len;
		bool	type_byval;
		Expr   *key_expr;

		get_typlenbyval(type_oid, &typlen, &typbyval);

		key_expr = ExecInitExpr(lfirst(lc), &mrs->css.ss.ps);
		hash_keys = lappend(hash_keys, key_expr);
		hash_keylen = append_int(hash_keylen, typlen);
		hash_keybyval = lappend_int(hash_keybyval, typbyval);
		hash_keytype = lappend_oid(hash_keytype, type_oid);
	}
	mrs->hash_keys = hash_keys;
	mrs->hash_keylen = hash_keylen;
	mrs->hash_keybyval = hash_keybyval;
	mrs->hash_keytype = hash_keytype;

	/*
     * initialize child nodes
     */
	outerPlanState(mrs) = ExecInitNode(outerPlan(cscan), estate, eflags);
	innerPlanState(mrs) = ExecInitNode(innerPlan(cscan), estate, eflags);
}

static TupleTableSlot *
multirels_exec(CustomScanState *node)
{
	elog(ERROR, "MultiRels does not support ExecProcNode call convention");
	return NULL;
}

static Size
multirels_expand_total_size(MultiRelsState *mrs)
{
	MultiRelsState *temp;
	Size		total_size_new;

	for (temp = mrs;
		 innerPlanState(temp) != NULL;
		 temp = innerPlanState(temp))
	{
		Assert(IsA(temp, CustomScanState) &&
			   mrs->css.methods == &multirels_exec_methods.c);
	}
	total_size_new = 2 * temp->total_size;

	/*
	 * If no more physical space to expand, we give up.
	 */
	if (total_size_new > gpuMemMaxAllocSize())
		return 0;

	/*
	 * Update initial length of pgstrom_multirels
	 */
	temp = mrs;
	do {
		temp->total_size = total_size_new;
		temp = (MultiRelsState *) innerPlanState(mrs);
	} while (temp);

	return total_size_new;
}

static kern_hashtable *
create_kern_multihash(MultiRelsState *mrs, pgstrom_multirels *pmrels)
{
	kern_hashtable *khtable;
	TupleTableSlot *scan_slot = mrs->css.ss.ss_ScanTupleSlot;
    TupleDesc		scan_desc = scan_slot->tts_tupleDescriptor;
	Size			chunk_size;
	Size			required;
	cl_uint		   *hash_slot;
	int				attcacheoff;
	int				attalign;
	int				i;

	required = (LONGALIGN(offsetof(kern_hashtable,
								   colmeta[scan_desc->natts])) +
				LONGALIGN(sizeof(cl_uint) * mrs->nslots));
	chunk_size = (Size)(mrs->proportion *
						(double)(pmrels->length - pmrels->length_head));
	while (chunk_size < required)
	{
		new_length = multirels_expand_total_size(mrs);
		if (new_length == 0)
			elog(ERROR, "failed to allocate minimum required memory");
		chunk_size = (Size)(mrs->proportion *
							(double)(new_length - pmrels->length_head));
	}

	khtable = MemoryContextAlloc(mrs->gcontext, chunk_size);
	khtable->length = chunk_size;
	khtable->ncols = tupdesc->natts;
	khtable->nitems = 0;
    khtable->nslots = mhs->nslots;
	khtable->hash_min = 0;
	khtable->hash_max = UINT_MAX;

	attcacheoff = offsetof(HeapTupleHeaderData, t_bits);
	if (tupdesc->tdhasoid)
		attcacheoff += sizeof(Oid);
	attcacheoff = MAXALIGN(attcacheoff);

	for (i=0; i < scan_desc->natts; i++)
	{
		Form_pg_attribute attr = scan_desc->attrs[i];

		attalign = typealign_get_width(attr->attalign);
		if (attcacheoff > 0)
		{
			if (attr->attlen > 0)
				attcacheoff = TYPEALIGN(attalign, attcacheoff);
			else
				attcacheoff = -1;	/* no more shortcut any more */
		}
		khtable->colmeta[i].attbyval = attr->attbyval;
		khtable->colmeta[i].attalign = attalign;
		khtable->colmeta[i].attlen = attr->attlen;
		khtable->colmeta[i].attnum = attr->attnum;
		khtable->colmeta[i].attcacheoff = attcacheoff;
		if (attcacheoff >= 0)
			attcacheoff += attr->attlen;
	}
	hash_slots = KERN_HASHTABLE_SLOT(khtable);
	memset(hash_slots, 0, sizeof(cl_uint) * khtable->nslots);

	return khtable;
}

static kern_multihash *
expand_kern_multihash()
{}

static bool
insert_kern_multihash(MultiRelsState *mrs, TupleTableSlot *slot)
{
	HeapTuple	tuple = ExecFetchSlotTuple(slot);
	pg_crc32	hash;
	ListCell   *lc1;
	ListCell   *lc2;
	ListCell   *lc3;
	ListCell   *lc4;
	Size		required;

	/* do we have enough space to store? */
	required = khtable->usage + MAXALIGN(tuple->t_len) +
		STROMALIGN(sizeof(cl_uint) * (khtable->nitems + 1));
	if (required > khtable->length)
		return false;

	/* calculation of a hash value of this entry */
	INIT_CRC32C(hash);
	econtext->ecxt_scantuple = slot;
	forfour (lc1, mrs->hash_keys,
			 lc2, mrs->hash_keylen,
			 lc3, mrs->hash_keybyval,
			 lc4, mrs->hash_keytype)
	{
		ExprState  *clause = lfirst(lc1);
		int			keylen = lfirst_int(lc2);
		bool		keybyval = lfirst_int(lc3);
		Oid			keytype = lfirst_oid(lc4);
		int			errcode;
		Datum		value;
		bool		isnull;

		value = ExecEvalExpr(clause, econtext, &isnull, NULL);
		if (isnull)
			continue;

		/* fixup host representation to special internal format. */
		if (keytype == NUMERICOID)
		{
			pg_numeric_t	temp
				= pg_numeric_from_varlena(&errcode, (struct varlena *)
										  DatumGetPointer(value));
			keylen = sizeof(temp.value);
			keybyval = true;
			value = temp.value;
		}

		if (keylen > 0)
		{
			if (keybyval)
				COMP_CRC32C(hash, &value, keylen);
			else
				COMP_CRC32C(hash, DatumGetPointer(value), keylen);
		}
		else
		{
			COMP_CRC32C(hash,
						VARDATA_ANY(value),
						VARSIZE_ANY_EXHDR(value));
		}
	}
	FIN_CRC32C(hash);

	/* put this tuple on the hash table */
	hentry = (kern_hashentry *)((char *)khtable + khtable->usage);
	hentry->hash = hash;
	hentry->rowid = 0;	/* to be set */
	hentry->t_len = tuple->t_len;
	memcpy(&hentry->htup, tuple->t_data, tuple->t_len);

	i = hash % khtable->nslots;
	hentry->next = hash_slot[i];
	hash_slot[i] = (uintptr_t)hentry - (uintptr_t)khtable;

	/* increment usage */
	khtable->usage += MAXALIGN(offsetof(kern_hashentry, htup) + tuple->t_len);

	return true;
}








static kern_multirels *
multirels_preload_hash(MultiRelsState *mrs, double *ntuples)
{
	elog(ERROR, "Not implemented yet");	
}

static pgstrom_data_store *
multirels_preload_heap(MultiRelsState *mrs,
					   pgstrom_multirels *pmrels,
					   double *p_ntuples)
{
	TupleTableSlot *scan_slot = mrs->css.ss.ss_ScanTupleSlot;
	TupleDesc		scan_dest = scan_slot->tts_tupleDescriptor;
	HeapTuple		scan_tuple;
	Size			chunk_size;
	long			ntuples = 0;
	pgstrom_data_store *pds;

	/*
	 * Make a pgstrom_data_store for materialization
	 */
	chunk_size = (Size)(mrs->proportion * (double)(pmrels->length -
												   pmrels->length_head));
	pds = pgstrom_create_data_store_row(mrs->gcontext,
										scan_desc, chunk_size, false);
	while (true)
	{
		if (!mrs->outer_overflow)
			scan_slot = ExecProcNode(outerPlanState(mrs));
		else
		{
			scan_slot = mrs->outer_overflow;
			mrs->outer_overflow = NULL;
		}

		if (TupIsNull(scan_slot))
		{
			mrs->outer_done = true;
			break;
		}

		if (pgstrom_data_store_insert_tuple(pds, scan_slot))
			ntuples++;
		else
		{
			MultiRelsState *temp;
			Size	new_length;

			/* to be inserted on the next try */
			Assert(mrs->outer_overflow == NULL);
			mrs->outer_overflow = scan_slot;

			/*
			 * We try to expand total length of pgstrom_multirels buffer,
			 * as long as it can be acquired on the device memory.
			 * If no more physical space is expected, we give up to preload
			 * entire relation on this store.
			 */
			new_length = multirels_expand_total_size(mrs);
			if (new_length == 0)
				break;

			/*
			 * Once total length of the buffer got expanded, current store
			 * also can have wider space.
			 */
			chunk_size = (Size)(mrs->proportion * (double)(new_length -
														   pmrels->length_head));
			pgstrom_expand_data_store(mrs->gcontext, pds, chunk_size);
		}
	}
	*p_ntuples = (double)ntuples;
	return pds;
}


static void *
multirels_exec_bulk(CustomScanState *node)
{
	MultiRelsState *mrs = (MultiRelsState *) node;
	GpuContext	   *gcontext = mrs->gcontext;
	PlanState	   *mrs_child;	/* underlying MultiRels, if any */
	double			ntuples = 0.0;


	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStartNode(node->ss.ps.instrument);

	mrs_child = innerPlanState(mrs);
	if (mrs_child)
	{
		pmrels = BulkExecMultiRels(mrs_child);
		if (!pmrels)
		{
			if (mrs->outer_done)
				goto out;
			ExecReScan(mrs);
			pmrels = BulkExecMultiRels(mrs_child);
			if (!pmrels)
				goto out;
			scan_forward = true;
		}
		else if (!mrs->curr_chunk)
			scan_forward = true;
	}
	else
	{
		/* No deeper relations, so create a new pgstrom_multi_relations */
		kern_multirels *kmrels;
		int		nrels = depth;
		Size	length;

		if (mrs->outer_done)
			goto out;
		scan_forward = true;

		/* allocation of pgstrom_multi_relations */
		length = (STROMALIGN(offsetof(pgstrom_multirels, prels[nrels])) +
				  STROMALIGN(offsetof(kern_multirels, krels[nrels])));
		pmrels = MemoryContextAllocZero(gcontext->memcxt, length);
		pmrels->length = mrs->inner_size;
		pmrels->usage = STROMALIGN(offsetof(kern_multirels, krels[nrels]));
		pmrels->is_divided = false;

		kmrels = (kern_multirels *)(pmrels->prels + nrels);
		memcpy(kmrels->pg_crc32_table,
			   pg_crc32c_table,
			   sizeof(uint32) * 256);
		kmrels->nrels = nrels;
		pmrels->kmrels = kmrels;
	}
	Assert(pmrels != NULL);

	if (scan_forward)
	{
		if (mrs->curr_chunk != NULL)
		{
			pfree(mrs->curr_chunk);
			mrs->curr_chunk = NULL;
		}

		if (mrs->hash_keys != NIL)
			multirels_preload_hash(mrs, &ntuples);
		else
			multirels_preload_heap(mrs, &ntuples);
	}
	Assert(mrs->curr_chunk != NULL);
	pmrels->prels[depth - 1] = mrs->curr_chunk;
	pmrels->usage += mrs->curr_length;
out:
	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStopNode(node->ss.ps.instrument, ntuples);

	if (pmrels)
		mrs->nbatches_exec++;

	return pmrels;
}

static void
multirels_end(CustomScanState *node)
{
	MultiRelsState *mrs = (MultiRelsState *) node;

	/*
	 * Shutdown the subplans
	 */
	ExecEndNode(outerPlanState(node));
    ExecEndNode(innerPlanState(node));

	/* Release GpuContext */
	pgstrom_put_gpucontext(mrs->gcontext);
}

static void
multirels_rescan(CustomScanState *node)
{
	MultiRelsState *mrs = (MultiRelsState *) node;

	if (innerPlanState(node))
		ExecReScan(innerPlanState(node));
	ExecReScan(outerPlanState(node));
	mrs->outer_done = false;
	mrs->outer_overflow = NULL;
}

static void
multirels_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	MultiRelsState *mrs = (MultiRelsState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	MultiRelsInfo  *mr_info = deform_multirels_info(cscan);
	StringInfoData	str;
	List		   *context;
	ListCell	   *lc;

	/* set up deparsing context */
	context = set_deparse_context_planstate(es->deparse_cxt,
											(Node *) node,
											ancestors);
	/* shows hash keys, if any */
	initStringInfo(&str);
	if (mr_info->hash_keys != NIL)
	{
		foreach (lc, mr_info->hash_keys)
		{
			char   *exprstr;

			if (lc != list_head(mr_info->hash_keys))
				appendStringInfo(&str, ", ");

			exprstr = deparse_expression(lfirst(lc),
										 context,
										 es->verbose,
										 false);
			appendStringInfo(&str, "%s", exprstr);
			pfree(exprstr);
		}
		ExplainPropertyText("hash keys", str.data, es);
	}

	/* shows other properties */
	if (es->format != EXPLAIN_FORMAT_TEXT)
	{
		resetStringInfo(&str);
		if (mrs->nbatches_exec >= 0)
			ExplainPropertyInteger("nBatches", mrs->nbatches_exec, es);
		else
			ExplainPropertyInteger("nBatches", mrs->nbatches_plan, es);
		ExplainPropertyInteger("Buckets", mr_info->nslots, es);
		appendStringInfo(&str, "%.2f%%", 100.0 * mr_info->threshold);
		ExplainPropertyText("Memory Usage", str.data, es);
	}
	else
	{
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str,
						 "nBatches: %u  Buckets: %u  Memory Usage: %.2f%%\n",
						 mrs->nbatches_exec >= 0
						 ? mrs->nbatches_exec
						 : mrs->nbatches_plan,
						 mr_info->nslots,
						 100.0 * mr_info->threshold);
	}
}

/*
 * pgstrom_init_multirels
 *
 * entrypoint of this custom-scan provider
 */
void
pgstrom_init_multirels(void)
{
	/* setup plan methods */
	multirels_plan_methods.CustomName			= "MultiRels";
	multirels_plan_methods.CreateCustomScanState
		= multirels_create_scan_state;
	multirels_plan_methods.TextOutCustomScan	= NULL;

	/* setup exec methods */
	multirels_exec_methods.c.CustomName			= "MultiRels";
	multirels_exec_methods.c.BeginCustomScan	= multirels_begin;
	multirels_exec_methods.c.ExecCustomScan		= multirels_exec;
	multirels_exec_methods.c.EndCustomScan		= multirels_end;
	multirels_exec_methods.c.ReScanCustomScan	= multirels_rescan;
	multirels_exec_methods.c.MarkPosCustomScan	= NULL;
	multirels_exec_methods.c.RestrPosCustomScan	= NULL;
	multirels_exec_methods.c.ExplainCustomScan	= multirels_explain;
	multirels_exec_methods.ExecCustomBulk		= multirels_exec_bulk;
}
