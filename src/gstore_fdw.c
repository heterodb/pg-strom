/*
 * gstore_fdw.c
 *
 * On GPU column based data store as FDW provider.
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "access/reloptions.h"
#include "access/xact.h"
#include "catalog/namespace.h"
#include "catalog/pg_attribute.h"
#include "catalog/pg_class.h"
#include "catalog/pg_foreign_data_wrapper.h"
#include "catalog/pg_foreign_server.h"
#include "catalog/pg_foreign_table.h"
#include "catalog/pg_language.h"
#include "catalog/pg_proc.h"
#include "commands/defrem.h"
#include "foreign/fdwapi.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/planmain.h"
#include "storage/ipc.h"
#include "storage/lmgr.h"
#include "storage/procarray.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"
#include "utils/pg_crc.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
#include "pg_strom.h"

#define GPUSTORE_CHUNK_SIZE		(1UL << 30)

/*
 * GpuStoreChunk
 */
struct GpuStoreChunk
{
	struct GpuStoreMap *gs_map;	/* reference to relevant GpuStoreMap */
	dlist_node	chain;
	pg_crc32	hash;			/* hash value by (database_oid + table_oid) */
	Oid			database_oid;
	Oid			table_oid;
	HeapTupleHeaderData htup;	/* for MVCC checks */
	dsm_handle	handle;
};
typedef struct GpuStoreChunk	GpuStoreChunk;

/*
 * GpuStoreMap - status of local mapping
 */
struct GpuStoreMap
{
	struct GpuStoreChunk *gs_chunk;	/* reference to relevant GpuStoreChunk */
	dsm_segment	   *dsm_seg;
};
typedef struct GpuStoreMap		GpuStoreMap;

/*
 * GpuStoreHead
 */
#define GSTORE_CHUNK_HASH_NSLOTS	97
typedef struct
{
	pg_atomic_uint32 has_warm_chunks;
	slock_t			lock;
	dlist_head		free_chunks;
	dlist_head		active_chunks[GSTORE_CHUNK_HASH_NSLOTS];
	GpuStoreChunk	gs_chunks[FLEXIBLE_ARRAY_MEMBER];
} GpuStoreHead;

/* ---- static functions ---- */
static Oid	gstore_fdw_read_options(Oid table_oid,
									char **p_synonym,
									bool *p_pinning);




/* ---- static variables ---- */
static int				gstore_max_nchunks;		/* GUC */
static shmem_startup_hook_type shmem_startup_next;
static GpuStoreHead	   *gstore_head = NULL;
static GpuStoreMap	   *gstore_maps = NULL;

/*
 * gstore_fdw_mapped_chunk
 */
static inline kern_data_store *
gstore_fdw_mapped_chunk(GpuStoreChunk *gs_chunk)
{
	GpuStoreMap	   *gs_map = gs_chunk->gs_map;

	if (!gs_map->dsm_seg)
	{
		gs_map->dsm_seg = dsm_attach(gs_chunk->handle);
	}
	else if (dsm_segment_handle(gs_map->dsm_seg) != gs_chunk->handle)
	{
		dsm_detach(gs_map->dsm_seg);

		gs_map->dsm_seg = dsm_attach(gs_chunk->handle);
		dsm_pin_mapping(gs_map->dsm_seg);
	}
	return (kern_data_store *)dsm_segment_address(gs_map->dsm_seg);
}

/*
 * gstore_fdw_(first|next)_chunk
 */
static GpuStoreChunk *
gstore_fdw_first_chunk(Relation frel, Snapshot snapshot)
{
	Oid				database_oid = MyDatabaseId;
	Oid				table_oid = RelationGetRelid(frel);
	GpuStoreChunk  *gs_chunk;
	dlist_iter		iter;
	pg_crc32		hash;
	int				index;

	INIT_TRADITIONAL_CRC32(hash);
	COMP_TRADITIONAL_CRC32(hash, &database_oid, sizeof(Oid));
	COMP_TRADITIONAL_CRC32(hash, &table_oid, sizeof(Oid));
	FIN_TRADITIONAL_CRC32(hash);
	index = hash % GSTORE_CHUNK_HASH_NSLOTS;

	dlist_foreach(iter, &gstore_head->active_chunks[index])
	{
		gs_chunk = dlist_container(GpuStoreChunk, chain, iter.cur);

		if (gs_chunk->hash == hash &&
			gs_chunk->database_oid == MyDatabaseId &&
			gs_chunk->table_oid == table_oid)
		{
			HeapTupleData	tuple;

			if (!snapshot)
				return gs_chunk;
			memset(&tuple, 0, sizeof(HeapTupleData));
			tuple.t_data = &gs_chunk->htup;
			if (HeapTupleSatisfiesVisibility(&tuple, snapshot, -1))
				return gs_chunk;
		}
	}
	return NULL;
}

static GpuStoreChunk *
gstore_fdw_next_chunk(GpuStoreChunk *gs_chunk, Snapshot snapshot)
{
	Oid			database_oid = gs_chunk->database_oid;
	Oid			table_oid = gs_chunk->table_oid;
	pg_crc32	hash = gs_chunk->hash;
	int			index = hash % GSTORE_CHUNK_HASH_NSLOTS;
	dlist_head *active_chunks = &gstore_head->active_chunks[index];
	dlist_node *dnode;

	while (dlist_has_next(active_chunks, &gs_chunk->chain))
	{
		dnode = dlist_next_node(active_chunks, &gs_chunk->chain);
		gs_chunk = dlist_container(GpuStoreChunk, chain, dnode);

		if (gs_chunk->hash == hash &&
			gs_chunk->database_oid == database_oid &&
			gs_chunk->table_oid == table_oid)
		{
			HeapTupleData	tuple;

			if (!snapshot)
				return gs_chunk;
			memset(&tuple, 0, sizeof(HeapTupleData));
			tuple.t_data = &gs_chunk->htup;
			if (HeapTupleSatisfiesVisibility(&tuple, snapshot, -1))
				return gs_chunk;
		}
	}
	return NULL;
}

/*
 * gstoreGetForeignRelSize
 */
static void
gstoreGetForeignRelSize(PlannerInfo *root,
						RelOptInfo *baserel,
						Oid ftable_oid)
{
	GpuStoreChunk  *gs_chunk;
	size_t			nitems = 0;
	size_t			length = 0;
	char		   *synonym;
	Relation		frel = NULL;

	gstore_fdw_read_options(ftable_oid, &synonym, NULL);
	if (!synonym)
		frel = heap_open(ftable_oid, AccessShareLock);
	else
	{
		List   *names = stringToQualifiedNameList(synonym);

		frel = heap_openrv(makeRangeVarFromNameList(names),
						   AccessShareLock);
	}

	SpinLockAcquire(&gstore_head->lock);
	PG_TRY();
	{
		for (gs_chunk = gstore_fdw_first_chunk(frel, NULL);
			 gs_chunk != NULL;
			 gs_chunk = gstore_fdw_next_chunk(gs_chunk, NULL))
		{
			kern_data_store *kds = gstore_fdw_mapped_chunk(gs_chunk);

			nitems += kds->nitems;
			length += TYPEALIGN(BLCKSZ, kds->length);
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&gstore_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&gstore_head->lock);
	baserel->rows	= nitems;
	baserel->pages	= length / BLCKSZ;
	heap_close(frel, NoLock);
}

/*
 * gstoreGetForeignPaths
 */
static void
gstoreGetForeignPaths(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Oid foreigntableid)
{
	ParamPathInfo *param_info;
	ForeignPath *fpath;
	Cost		startup_cost = baserel->baserestrictcost.startup;
	Cost		per_tuple = baserel->baserestrictcost.per_tuple;
	Cost		run_cost;
	QualCost	qcost;

	param_info = get_baserel_parampathinfo(root, baserel, NULL);
	if (param_info)
	{
		cost_qual_eval(&qcost, param_info->ppi_clauses, root);
		startup_cost += qcost.startup;
		per_tuple += qcost.per_tuple;
	}
	run_cost = per_tuple * baserel->rows;

	fpath = create_foreignscan_path(root,
									baserel,
									NULL,	/* default pathtarget */
									baserel->rows,
									startup_cost,
									startup_cost + run_cost,
									NIL,	/* no pathkeys */
									NULL,	/* no outer rel either */
									NULL,	/* no extra plan */
									NIL);	/* no fdw_private */
	add_path(baserel, (Path *) fpath);
}

/*
 * gstoreGetForeignPlan
 */
static ForeignScan *
gstoreGetForeignPlan(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid foreigntableid,
					 ForeignPath *best_path,
					 List *tlist,
					 List *scan_clauses,
					 Plan *outer_plan)
{
	List	   *scan_quals = NIL;
	ListCell   *lc;

	foreach (lc, scan_clauses)
	{
		RestrictInfo *rinfo = (RestrictInfo *) lfirst(lc);

		Assert(IsA(rinfo, RestrictInfo));
		if (rinfo->pseudoconstant)
			continue;
		scan_quals = lappend(scan_quals, rinfo->clause);
	}

	return make_foreignscan(tlist,
							scan_quals,
							baserel->relid,
							NIL,		/* fdw_exprs */
							NIL,		/* fdw_private */
							NIL,		/* fdw_scan_tlist */
							NIL,		/* fdw_recheck_quals */
							NULL);		/* outer_plan */
}

/*
 * gstoreScanState - state object for scan
 */
typedef struct
{
	GpuStoreChunk  *gs_chunk;
	cl_ulong		gs_index;
	Relation		gs_rel;
	bool			pinning;
	cl_uint			nattrs;
	AttrNumber		attnos[FLEXIBLE_ARRAY_MEMBER];
} gstoreScanState;

/*
 * gstoreBeginForeignScan
 */
static void
gstoreBeginForeignScan(ForeignScanState *node, int eflags)
{
	//ForeignScan	   *fscan = (ForeignScan *) node->ss.ps.plan;
	TupleDesc		tupdesc = RelationGetDescr(node->ss.ss_currentRelation);
	Relation		gs_rel = NULL;
	gstoreScanState *gss_state;
	char		   *synonym;
	int				i, j;

	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;

	gstore_fdw_read_options(RelationGetRelid(node->ss.ss_currentRelation),
							&synonym, NULL);
	if (synonym)
	{
		List   *names = stringToQualifiedNameList(synonym);

		gs_rel = heap_openrv(makeRangeVarFromNameList(names),
							 AccessShareLock);
	}
	gss_state = palloc(offsetof(gstoreScanState,
								attnos[tupdesc->natts]));
	gss_state->gs_chunk = NULL;
	gss_state->gs_index = 0;
	gss_state->gs_rel = gs_rel;
	gss_state->nattrs = tupdesc->natts;
	if (gs_rel != NULL)
	{
		TupleDesc	phydesc = RelationGetDescr(gs_rel);

		for (i=0; i < tupdesc->natts; i++)
		{
			const char *attname = NameStr(tupdesc->attrs[i]->attname);

			for (j=0; j < phydesc->natts; j++)
			{
				Form_pg_attribute	pattr = phydesc->attrs[j];

				if (strcmp(attname, NameStr(pattr->attname)) == 0)
				{
					gss_state->attnos[i] = pattr->attnum;
					break;
				}
			}
			if (j >= phydesc->natts)
				elog(ERROR, "attribute \"%s\" was not found at \"%s\"",
					 attname, RelationGetRelationName(gs_rel));
		}
	}
	else
	{
		for (i=0; i < tupdesc->natts; i++)
			gss_state->attnos[i] = i+1;
	}
	node->fdw_state = (void *)gss_state;
}

/*
 * gstoreIterateForeignScan
 */
static TupleTableSlot *
gstoreIterateForeignScan(ForeignScanState *node)
{
	gstoreScanState	*gss_state = (gstoreScanState *) node->fdw_state;
	Relation		frel = node->ss.ss_currentRelation;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	EState		   *estate = node->ss.ps.state;
	Snapshot		snapshot = estate->es_snapshot;
	GpuStoreChunk  *gs_chunk;
	kern_data_store *kds;
	cl_ulong		index;
	cl_int			i, colidx;
	cl_uint			offset;
	kern_colmeta   *kcmeta;
	char		   *att_values;

	ExecClearTuple(slot);
	if (!gss_state->gs_chunk)
	{
		SpinLockAcquire(&gstore_head->lock);
		gs_chunk = gstore_fdw_first_chunk(frel, snapshot);
		SpinLockRelease(&gstore_head->lock);
		if (!gs_chunk)
			return NULL;
		gss_state->gs_chunk = gs_chunk;
	}
next:
	kds = gstore_fdw_mapped_chunk(gss_state->gs_chunk);
	if (gss_state->gs_index >= kds->nitems)
	{
		gss_state->gs_chunk = gstore_fdw_next_chunk(gss_state->gs_chunk,
													snapshot);
		gss_state->gs_index = 0;
		if (!gss_state->gs_chunk)
			return NULL;
		goto next;
	}
	index = gss_state->gs_index++;
	ExecStoreAllNullTuple(slot);

	for (i=0; i < gss_state->nattrs; i++)
	{
		colidx = gss_state->attnos[i] - 1;
		Assert(colidx >= 0 && colidx < kds->ncols);
		kcmeta = &kds->colmeta[colidx];
		att_values = (char *)kds + kcmeta->values_offset;
		if (kcmeta->attlen > 0)
		{
			/* null-check */
			if (kcmeta->extra_sz > 0 &&
				index < BITS_PER_BYTE * kcmeta->extra_sz)
			{
				bits8  *nullmap = (bits8 *)
					(att_values + MAXALIGN(kcmeta->attlen * kds->nitems));

				if (att_isnull(index, nullmap))
					continue;
			}
			att_values += kcmeta->attlen * index;
			slot->tts_values[i] = fetch_att(att_values,
											kcmeta->attbyval,
											kcmeta->attlen);
		}
		else
		{
			/* varlena field */
			Assert(kcmeta->attlen == -1);

			offset = ((cl_uint *)att_values)[index];
			/* null-check */
			if (offset == 0)
				continue;
			Assert(offset >= sizeof(cl_uint) * kds->nitems &&
				   offset <  sizeof(cl_uint) * kds->nitems + kcmeta->extra_sz);
			slot->tts_values[i] = PointerGetDatum(att_values + offset);
		}
	}
	return slot;
}

/*
 * gstoreReScanForeignScan
 */
static void
gstoreReScanForeignScan(ForeignScanState *node)
{
	gstoreScanState *gss_state = (gstoreScanState *) node->fdw_state;

	gss_state->gs_chunk = NULL;
	gss_state->gs_index = 0;
}

/*
 * gstoreEndForeignScan
 */
static void
gstoreEndForeignScan(ForeignScanState *node)
{
}

/*
 * gstoreIsForeignRelUpdatable
 */
static int
gstoreIsForeignRelUpdatable(Relation rel)
{
	char   *synonym;

	gstore_fdw_read_options(RelationGetRelid(rel), &synonym, NULL);
	/* 'synonym' table is not updatable */
	if (synonym != NULL)
		return 0;

	return CMD_INSERT | CMD_DELETE;
}

/*
 * gstorePlanDirectModify - allows only DELETE with no WHERE-clause
 */
static bool
gstorePlanDirectModify(PlannerInfo *root,
					   ModifyTable *plan,
					   Index resultRelation,
					   int subplan_index)
{
	CmdType		operation = plan->operation;
	Plan	   *subplan = (Plan *) list_nth(plan->plans, subplan_index);

	/* only DELETE command */
	if (operation != CMD_DELETE)
		return false;
	/* no WHERE-clause */
	if (subplan->qual != NIL)
		return false;
	/* no RETURNING-clause */
	if (plan->returningLists != NIL)
		return false;
	/* subplan should be GpuStore FDW */
	if (!IsA(subplan, ForeignScan))
		return false;

	/*
	 * XXX - can we expect data type compatibility of target-list
	 * by the core? likely, no need to check again.
	 */
	return true;
}

/*
 * gstorePlanForeignModify
 */
static List *
gstorePlanForeignModify(PlannerInfo *root,
						ModifyTable *plan,
						Index resultRelation,
						int subplan_index)
{
	CmdType		operation = plan->operation;

	if (operation != CMD_INSERT)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("gstore_fdw: not a supported operation"),
				 errdetail("gstore_fdw supports either INSERT into an empty GpuStore or DELETE without WHERE-clause only")));

	return NIL;
}

/*
 * gstoreLoadState - state object for INSERT
 */
typedef struct
{
	size_t		length;		/* available size except for KDS header */
	size_t		nrooms;		/* available max number of items */
	size_t		nitems;		/* current number of items */
	MemoryContext memcxt;	/* memcxt for construction per chunk */
	struct {
		HTAB   *vl_dict;	/* dictionary of varlena datum, if any */
		size_t	extra_sz;	/* usage by varlena datum */
		bits8  *nullmap;	/* NULL-bitmap */
		void   *values;		/* array of values */
		int		align;		/* numeric form of attalign */
	} a[FLEXIBLE_ARRAY_MEMBER];
} gstoreLoadState;

typedef struct
{
	cl_uint		offset;		/* to be used later */
	struct varlena *vl_datum;
} vl_dict_key;

static uint32
vl_dict_hash_value(const void *__key, Size keysize)
{
	const vl_dict_key *key = __key;
	pg_crc32		crc;

	Assert(keysize == sizeof(vl_dict_key));
	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, key->vl_datum, VARSIZE_ANY(key->vl_datum));
	FIN_LEGACY_CRC32(crc);

	return (uint32) crc;
}

static int
vl_dict_compare(const void *__key1, const void *__key2, Size keysize)
{
	const vl_dict_key *key1 = __key1;
	const vl_dict_key *key2 = __key2;

	if (VARSIZE_ANY(key1->vl_datum) == VARSIZE_ANY(key2->vl_datum))
		return memcmp(key1->vl_datum, key2->vl_datum,
					  VARSIZE_ANY(key1->vl_datum));
	return 1;
}

static HTAB *
vl_dict_create(MemoryContext memcxt, size_t nrooms)
{
	HASHCTL		hctl;

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.hash = vl_dict_hash_value;
	hctl.match = vl_dict_compare;
	hctl.keysize = sizeof(vl_dict_key);
	hctl.hcxt = memcxt;

	return hash_create("varlena dictionary hash-table",
					   Max(nrooms / 10, 4096),
					   &hctl,
					   HASH_FUNCTION |
					   HASH_COMPARE |
					   HASH_CONTEXT);
}

/*
 * gstore_fdw_writeout_chunk
 */
static void
gstore_fdw_writeout_chunk(Relation relation, gstoreLoadState *gs_lstate)
{
	TupleDesc		tupdesc = RelationGetDescr(relation);
	MemoryContext	memcxt = gs_lstate->memcxt;
	size_t			nrooms = gs_lstate->nrooms;
	size_t			nitems = gs_lstate->nitems;
	size_t			length;
	size_t			offset;
	Oid				table_oid = RelationGetRelid(relation);
	pg_crc32		hash;
	cl_int			i, j, unitsz;
	dsm_segment	   *dsm_seg;
	kern_data_store *kds;
	dlist_node	   *dnode;
	GpuStoreChunk  *gs_chunk = NULL;
	GpuStoreMap	   *gs_map = NULL;
	HeapTupleHeaderData htup;

	length = offset = MAXALIGN(offsetof(kern_data_store,
										colmeta[tupdesc->natts]));
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		if (attr->attlen < 0)
		{
			length += (MAXALIGN(sizeof(cl_uint) * nitems) +
					   gs_lstate->a[i].extra_sz);
		}
		else
		{
			if (gs_lstate->a[i].nullmap)
				length += MAXALIGN(BITMAPLEN(nitems));
			length += MAXALIGN(TYPEALIGN(gs_lstate->a[i].align,
										 attr->attlen) * nitems);
		}
	}

	dsm_seg = dsm_create(length, 0);
	kds = dsm_segment_address(dsm_seg);

	init_kernel_data_store(kds,
						   tupdesc,
						   length,
						   KDS_FORMAT_COLUMN,
						   nitems);
	for (i=0; i < tupdesc->natts; i++)
	{
		kern_colmeta   *cmeta = &kds->colmeta[i];

		cmeta->values_offset = offset;
		if (cmeta->attlen < 0)
		{
			HASH_SEQ_STATUS	hseq;
			vl_dict_key	  **entries_array;
			vl_dict_key	   *entry;
			cl_uint		   *base;
			char		   *extra;

			/* put varlena datum on the extra area */
			base = (cl_uint *)((char *)kds + offset);
			extra = (char *)base + MAXALIGN(sizeof(cl_uint) * nitems);
			hash_seq_init(&hseq, gs_lstate->a[i].vl_dict);
			while ((entry = hash_seq_search(&hseq)) != NULL)
			{
				entry->offset = extra - (char *)base;
				unitsz = VARSIZE_ANY(entry->vl_datum);
				memcpy(extra, entry->vl_datum, unitsz);
				cmeta->extra_sz += MAXALIGN(unitsz);
				extra += MAXALIGN(unitsz);
			}
			hash_seq_term(&hseq);

			/* put offset of varlena datum */
			entries_array = (vl_dict_key **)gs_lstate->a[i].values;
			for (j=0; j < nitems; j++)
			{
				entry = entries_array[j];
				base[j] = (!entry ? 0 : entry->offset);
			}
		}
		else
		{
			unitsz = TYPEALIGN(gs_lstate->a[i].align, cmeta->attlen);

			memcpy((char *)kds + offset,
				   gs_lstate->a[i].values,
				   unitsz * nitems);
			offset += MAXALIGN(unitsz * nitems);
			if (gs_lstate->a[i].nullmap)
			{
				memcpy((char *)kds + offset,
					   gs_lstate->a[i].nullmap,
					   BITMAPLEN(nitems));
				cmeta->extra_sz = BITMAPLEN(nitems);
				offset += MAXALIGN(BITMAPLEN(nitems));
			}
		}
	}
	/* hash value */
	INIT_TRADITIONAL_CRC32(hash);
    COMP_TRADITIONAL_CRC32(hash, &MyDatabaseId, sizeof(Oid));
	COMP_TRADITIONAL_CRC32(hash, &table_oid, sizeof(Oid));
	FIN_TRADITIONAL_CRC32(hash);
	/* see heap_prepare_insert() */
	memset(&htup, 0, sizeof(HeapTupleHeaderData));
	htup.t_infomask = HEAP_XMAX_INVALID;
	HeapTupleHeaderSetXmin(&htup, GetCurrentTransactionId());
	HeapTupleHeaderSetCmin(&htup, GetCurrentCommandId(true));
	HeapTupleHeaderSetXmax(&htup, 0);

	SpinLockAcquire(&gstore_head->lock);
	if (dlist_is_empty(&gstore_head->free_chunks))
	{
		SpinLockRelease(&gstore_head->lock);
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("too many gstore_fdw chunks required")));
	}
	dnode = dlist_pop_head_node(&gstore_head->free_chunks);
	gs_chunk = dlist_container(GpuStoreChunk, chain, dnode);
	gs_map = gs_chunk->gs_map;

	gs_chunk->hash = hash;
	gs_chunk->database_oid = MyDatabaseId;
	gs_chunk->table_oid = table_oid;
	memcpy(&gs_chunk->htup, &htup, sizeof(HeapTupleHeaderData));
	gs_chunk->handle = dsm_segment_handle(dsm_seg);
	gs_map->dsm_seg = dsm_seg;

	i = hash % GSTORE_CHUNK_HASH_NSLOTS;
	dlist_push_tail(&gstore_head->active_chunks[i], &gs_chunk->chain);
	SpinLockRelease(&gstore_head->lock);

	/* reset temporary buffer */
	MemoryContextReset(memcxt);
	for (i=0; i < tupdesc->natts; i++)
	{
		if (gs_lstate->a[i].vl_dict)
			gs_lstate->a[i].vl_dict = vl_dict_create(memcxt, nrooms);
		if (gs_lstate->a[i].nullmap)
		{
			pfree(gs_lstate->a[i].nullmap);
			gs_lstate->a[i].nullmap = NULL;
		}
	}
	gs_lstate->nitems = 0;
}

/*
 * gstoreBeginForeignModify
 */
static void
gstoreBeginForeignModify(ModifyTableState *mtstate,
						 ResultRelInfo *rrinfo,
						 List *fdw_private,
						 int subplan_index,
						 int eflags)
{
	EState		   *estate = mtstate->ps.state;
	Relation		relation = rrinfo->ri_RelationDesc;
	TupleDesc		tupdesc = RelationGetDescr(relation);
	GpuStoreChunk  *gs_chunk;
	gstoreLoadState *gs_lstate;
	size_t			nrooms;
	cl_int			i, unitsz;

	LockRelationOid(RelationGetRelid(relation), AccessExclusiveLock);
	SpinLockAcquire(&gstore_head->lock);
	gs_chunk = gstore_fdw_first_chunk(relation, estate->es_snapshot);
	SpinLockRelease(&gstore_head->lock);

	//XXX - resowner hook may be able to merge them
	if (gs_chunk)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("gstore_fdw: foreign table \"%s\" is not empty",
						RelationGetRelationName(relation))));
	/* state object */
	gs_lstate = palloc0(offsetof(gstoreLoadState, a[tupdesc->natts]));
	gs_lstate->memcxt = AllocSetContextCreate(estate->es_query_cxt,
											  "gstore_fdw temporary context",
											  ALLOCSET_DEFAULT_SIZES);
	gs_lstate->length = GPUSTORE_CHUNK_SIZE -
		offsetof(kern_data_store, colmeta[tupdesc->natts]);

	/*
	 * calculation of the maximum possible nrooms, in case when all the
	 * column has no NULLs (thus no null bitmap), and extra_sz by varlena
	 * values are ignored.
	 */
	unitsz = 0;
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];
		int		align;

		if (attr->attlen < 0)
			unitsz += sizeof(cl_uint);	/* varlena offset */
		else
		{
			if (attr->attalign == 'c')
				align = sizeof(cl_char);
			else if (attr->attalign == 's')
				align = sizeof(cl_short);
			else if (attr->attalign == 'i')
				align = sizeof(cl_int);
			else if (attr->attalign == 'd')
				align = sizeof(cl_long);
			else
				elog(ERROR, "Bug? unexpected alignment: %c", attr->attalign);
			unitsz += TYPEALIGN(align, attr->attlen);
			gs_lstate->a[i].align = align;
		}
	}
	nrooms = (gs_lstate->length -	/* consider the margin for alignment */
			  MAXIMUM_ALIGNOF * tupdesc->natts) / unitsz;
	gs_lstate->nrooms = nrooms;
	gs_lstate->nitems = 0;

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		if (attr->attlen < 0)
		{
			MemoryContext	memcxt = gs_lstate->memcxt;
			gs_lstate->a[i].vl_dict = vl_dict_create(memcxt, nrooms);
			gs_lstate->a[i].values = palloc(sizeof(void *) * nrooms);
		}
		else
		{
			gs_lstate->a[i].values = palloc(TYPEALIGN(gs_lstate->a[i].align,
													  attr->attlen) * nrooms);
		}
	}
	rrinfo->ri_FdwState = gs_lstate;
}

/*
 * gstoreExecForeignInsert
 */
static TupleTableSlot *
gstoreExecForeignInsert(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	TupleDesc	tupdesc = slot->tts_tupleDescriptor;
	gstoreLoadState *gs_lstate = rrinfo->ri_FdwState;
	size_t		nrooms = gs_lstate->nrooms;
	size_t		nitems = gs_lstate->nitems;
	size_t		usage = 0;
	size_t		index;
	cl_int		i;

	slot_getallattrs(slot);

	/*
	 * calculation of extra consumption by this new line
	 */
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		if (attr->attlen < 0)
		{
			if (!slot->tts_isnull[i])
			{
				vl_dict_key	key;

				key.offset = 0;
				key.vl_datum = (struct varlena *)slot->tts_values[i];

				if (!hash_search(gs_lstate->a[i].vl_dict,
								 &key, HASH_FIND, NULL))
					usage += MAXALIGN(VARSIZE_ANY(key.vl_datum));
			}
			usage += sizeof(cl_uint);
		}
		else
		{
			if (gs_lstate->a[i].nullmap || slot->tts_isnull[i])
				usage += MAXALIGN(BITMAPLEN(nitems + 1));
			usage += TYPEALIGN(gs_lstate->a[i].align,
							   attr->attlen) * (nitems + 1);
		}
	}

	if (usage > gs_lstate->length)
		gstore_fdw_writeout_chunk(rrinfo->ri_RelationDesc, gs_lstate);

	index = gs_lstate->nitems++;
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];
		bits8	   *nullmap = gs_lstate->a[i].nullmap;
		char	   *values = gs_lstate->a[i].values;
		Datum		datum = slot->tts_values[i];

		if (slot->tts_isnull[i])
		{
			if (attr->attnotnull)
				elog(ERROR,
					 "attribute \"%s\" of relation \"%s\" must be NOT NULL",
					 NameStr(attr->attname),
					 RelationGetRelationName(rrinfo->ri_RelationDesc));
			if (!nullmap)
			{
				nullmap = MemoryContextAlloc(gs_lstate->memcxt,
											 MAXALIGN(BITMAPLEN(nrooms)));
				memset(nullmap, -1, BITMAPLEN(index));
				gs_lstate->a[i].nullmap = nullmap;
			}
			nullmap[index >> 3] &= ~(1 << (index & 7));
		}
		else
		{
			if (nullmap)
				nullmap[index >> 3] |=  (1 << (index & 7));
			if (attr->attlen < 0)
			{
				vl_dict_key	key, *entry;
				bool		found;

				key.offset = 0;
				key.vl_datum = (struct varlena *)DatumGetPointer(datum);
				entry = hash_search(gs_lstate->a[i].vl_dict,
									&key,
									HASH_ENTER,
									&found);
				if (!found)
				{
					MemoryContext oldcxt
						= MemoryContextSwitchTo(gs_lstate->memcxt);
					entry->offset = 0;
					entry->vl_datum = PG_DETOAST_DATUM_COPY(datum);
					MemoryContextSwitchTo(oldcxt);

					gs_lstate->a[i].extra_sz += MAXALIGN(VARSIZE_ANY(datum));
				}
				((vl_dict_key **)values)[index] = entry;
			}
			else if (!attr->attbyval)
			{
				values += TYPEALIGN(gs_lstate->a[i].align,
									attr->attlen) * index;
				memcpy(values, DatumGetPointer(datum), attr->attlen);
			}
			else
			{
				values += TYPEALIGN(gs_lstate->a[i].align,
									attr->attlen) * index;
				switch (attr->attlen)
				{
					case sizeof(cl_char):
						*((cl_char *)values) = DatumGetChar(datum);
						break;
					case sizeof(cl_short):
						*((cl_short *)values) = DatumGetInt16(datum);
						break;
					case sizeof(cl_int):
						*((cl_int *)values) = DatumGetInt32(datum);
						break;
					case sizeof(cl_long):
						*((cl_long *)values) = DatumGetInt64(datum);
						break;
					default:
						elog(ERROR, "Unexpected attribute length: %d",
							 attr->attlen);
				}
			}
		}
	}
	return slot;
}

/*
 * gstoreEndForeignModify
 */
static void
gstoreEndForeignModify(EState *estate,
					   ResultRelInfo *rrinfo)
{
	gstoreLoadState *gs_lstate = rrinfo->ri_FdwState;

	if (gs_lstate->nitems > 0)
		gstore_fdw_writeout_chunk(rrinfo->ri_RelationDesc, gs_lstate);
}

/*
 * gstoreBeginDirectModify
 */
static void
gstoreBeginDirectModify(ForeignScanState *node, int eflags)
{}

/*
 * gstoreIterateDirectModify
 */
static TupleTableSlot *
gstoreIterateDirectModify(ForeignScanState *node)
{
	return NULL;
}

/*
 * gstoreEndDirectModify
 */
static void
gstoreEndDirectModify(ForeignScanState *node)
{}

/*
 * gstoreXactCallbackPerChunk
 */
static bool
gstoreXactCallbackPerChunk(XactEvent event, GpuStoreChunk *gs_chunk,
						   TransactionId oldestXmin)
{
	HeapTupleHeader	htup = &gs_chunk->htup;
	bool		retval = false;

	if ((htup->t_infomask & HEAP_XMAX_INVALID) == 0)
	{
		/*
		 * Remove chunk if not visible to any concurrent transactions.
		 * Logic follows how vacuum prunes dead tuples
		 */
		HeapTupleData	tupleData;
		HTSV_Result		state;

		memset(&tupleData, 0, sizeof(HeapTupleData));
		tupleData.t_self.ip_posid = 1;
		tupleData.t_tableOid = gs_chunk->table_oid;
		tupleData.t_data = htup;

		state = HeapTupleSatisfiesVacuum(&tupleData, oldestXmin, -1);
		if (state == HEAPTUPLE_DEAD)
		{
			GpuStoreMap	   *gs_map = gs_chunk->gs_map;

			dlist_delete(&gs_chunk->chain);
			if (gs_map->dsm_seg != NULL)
			{
				dsm_detach(gs_map->dsm_seg);
				gs_map->dsm_seg = NULL;
			}
			/* NOTE: dsm_unpin_segment() is supported at PGv10 */
			//dsm_unpin_segment(gs_chunk->handle);

			/* reset */
			gs_chunk->hash = 0;
			gs_chunk->table_oid = 0;
			gs_chunk->database_oid = 0;
			memset(&gs_chunk->htup, 0, sizeof(HeapTupleHeaderData));
			gs_chunk->handle = UINT_MAX;

			dlist_push_tail(&gstore_head->free_chunks, &gs_chunk->chain);
		}
		retval = true;
	}
	else if (!HeapTupleHeaderXminFrozen(htup))
	{
		/* freeze chunks if visible to any transactions */
		/* see logic in tuple_all_visible() */
		HeapTupleData	tupleData;
		HTSV_Result		state;

		memset(&tupleData, 0, sizeof(HeapTupleData));
		tupleData.t_self.ip_posid = 1;
		tupleData.t_tableOid = gs_chunk->table_oid;
		tupleData.t_data = htup;

		state = HeapTupleSatisfiesVacuum(&tupleData, oldestXmin, -1);
		if (state == HEAPTUPLE_LIVE)
		{
			TransactionId	xmin = HeapTupleHeaderGetXmin(htup);

			if (TransactionIdPrecedes(xmin, oldestXmin))
			{
				/* make it frozen */
				HeapTupleHeaderSetXmin(htup, FrozenTransactionId);
				HeapTupleHeaderSetXminFrozen(htup);
			}
		}
		retval = true;
	}
	return retval;
}

/*
 * gstoreXactCallback
 */
static void
gstoreXactCallback(XactEvent event, void *arg)
{
	TransactionId oldestXmin;
	bool		meet_warm_chunks = false;
	cl_int		i;

	if (pg_atomic_read_u32(&gstore_head->has_warm_chunks) == 0)
		return;

	oldestXmin = GetOldestXmin(NULL, true);
	SpinLockAcquire(&gstore_head->lock);
	for (i=0; i < GSTORE_CHUNK_HASH_NSLOTS; i++)
	{
		dlist_mutable_iter	iter;

		dlist_foreach_modify(iter, &gstore_head->active_chunks[i])
		{
			GpuStoreChunk  *gs_chunk
				= dlist_container(GpuStoreChunk, chain, iter.cur);

			if (gstoreXactCallbackPerChunk(event, gs_chunk, oldestXmin))
				meet_warm_chunks = true;
		}
	}

	if (!meet_warm_chunks)
		pg_atomic_write_u32(&gstore_head->has_warm_chunks, 0);
	SpinLockRelease(&gstore_head->lock);
}

/*
 * relation_is_gstore_fdw
 */
static bool
relation_is_gstore_fdw(Oid table_oid, bool allow_synonym)
{
	HeapTuple	tup;
	Oid			fserv_oid;
	Oid			fdw_oid;
	Oid			handler_oid;
	PGFunction	handler_fn;
	Datum		datum;
	char	   *synonym;
	char	   *prosrc;
	char	   *probin;
	bool		isnull;
	/* it should be foreign table, of course */
	if (get_rel_relkind(table_oid) != RELKIND_FOREIGN_TABLE)
		return false;
	/* pull OID of foreign-server */
	fserv_oid = gstore_fdw_read_options(table_oid,
										&synonym,
										NULL);
	if (!allow_synonym && synonym != NULL)
		return false;

	/* pull OID of foreign-data-wrapper */
	tup = SearchSysCache1(FOREIGNSERVEROID, ObjectIdGetDatum(fserv_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "foreign server with OID %u does not exist", fserv_oid);
	fdw_oid = ((Form_pg_foreign_server) GETSTRUCT(tup))->srvfdw;
	ReleaseSysCache(tup);

	/* pull OID of FDW handler function */
	tup = SearchSysCache1(FOREIGNDATAWRAPPEROID, ObjectIdGetDatum(fdw_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign-data wrapper %u",fdw_oid);
	handler_oid = ((Form_pg_foreign_data_wrapper) GETSTRUCT(tup))->fdwhandler;
	ReleaseSysCache(tup);
	/* pull library path & function name */
	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(handler_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", handler_oid);
	if (((Form_pg_proc) GETSTRUCT(tup))->prolang != ClanguageId)
		elog(ERROR, "FDW handler function is not written with C-language");

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "null prosrc for C function %u", handler_oid);
	prosrc = TextDatumGetCString(datum);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_probin, &isnull);
	if (isnull)
		elog(ERROR, "null probin for C function %u", handler_oid);
	probin = TextDatumGetCString(datum);
	ReleaseSysCache(tup);
	/* check whether function pointer is identical */
	handler_fn = load_external_function(probin, prosrc, true, NULL);
	if (handler_fn != pgstrom_gstore_fdw_handler)
		return false;
	/* OK, it is GpuStore foreign table */
	return true;
}

/*
 * gstore_fdw_read_options
 */
static Oid
gstore_fdw_read_options(Oid table_oid,
						char **p_synonym,
						bool *p_pinning)
{
	HeapTuple	tup;
	Datum		datum;
	bool		isnull;
	Oid			fserv_oid;
	char	   *synonym = NULL;
	bool		pinning = false;

	tup = SearchSysCache1(FOREIGNTABLEREL, ObjectIdGetDatum(table_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign table %u", table_oid);
	fserv_oid = ((Form_pg_foreign_table) GETSTRUCT(tup))->ftserver;
	datum = SysCacheGetAttr(FOREIGNTABLEREL, tup,
							Anum_pg_foreign_table_ftoptions,
							&isnull);
	if (!isnull)
	{
		List	   *options = untransformRelOptions(datum);
		ListCell   *lc;

		foreach (lc, options)
		{
			DefElem	   *defel = lfirst(lc);

			if (strcmp(defel->defname, "synonym") == 0)
			{
				synonym = defGetString(defel);
			}
			else if (strcmp(defel->defname, "pinning") == 0)
			{
				pinning = defGetBoolean(defel);
			}
			else
				elog(ERROR, "Unknown FDW option: '%s'='%s'",
					 defel->defname, defGetString(defel));
		}
	}
	ReleaseSysCache(tup);
	if (p_synonym)
		*p_synonym = synonym;
	if (p_pinning)
		*p_pinning = pinning;
	return fserv_oid;
}

/*
 * pgstrom_gstore_fdw_validator
 */
Datum
pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS)
{
	List	   *options = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid			catalog = PG_GETARG_OID(1);
	ListCell   *lc;
	bool		meet_synonym = false;
	bool		meet_pinning = false;
	bool		config_pinning = false;

	foreach(lc, options)
	{
		DefElem	   *defel = lfirst(lc);

		if (strcmp(defel->defname, "synonym") == 0 &&
			catalog == ForeignTableRelationId)
		{
			char   *relname = defGetString(defel);
			List   *names;
			Oid		reloid;

			if (meet_synonym)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
                         errmsg("\"synonym\" option appears twice")));

			names = stringToQualifiedNameList(relname);
			reloid = RangeVarGetRelid(makeRangeVarFromNameList(names),
									  NoLock, false);
			if (!relation_is_gstore_fdw(reloid, false))
				elog(ERROR, "%s: not a gstore_fdw foreign table, or synonym",
					 relname);
			meet_synonym = true;
		}
		else if (strcmp(defel->defname, "pinning") == 0 &&
				 catalog == ForeignTableRelationId)
		{
			if (meet_pinning)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"pinning\" option appears twice")));

			/* Don't care what the value is, as long as it's a legal boolean */
			config_pinning = defGetBoolean(defel);

			elog(ERROR, "Not supported yet");			

			meet_pinning = true;
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_SYNTAX_ERROR),
					 errmsg("FDW option \"%s\" = \"%s\" is not supported",
							defel->defname, defGetString(defel))));
		}
	}
	if (config_pinning && meet_synonym)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("cannot use 'synonym' and 'pinning' together")));
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_validator);

/*
 * pgstrom_gstore_fdw_handler
 */
Datum
pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS)
{
	FdwRoutine *routine = makeNode(FdwRoutine);

	/* functions for scanning foreign tables */
	routine->GetForeignRelSize	= gstoreGetForeignRelSize;
	routine->GetForeignPaths	= gstoreGetForeignPaths;
	routine->GetForeignPlan		= gstoreGetForeignPlan;
	routine->BeginForeignScan	= gstoreBeginForeignScan;
	routine->IterateForeignScan	= gstoreIterateForeignScan;
	routine->ReScanForeignScan	= gstoreReScanForeignScan;
	routine->EndForeignScan		= gstoreEndForeignScan;

	/* functions for INSERT/DELETE foreign tables */
	routine->IsForeignRelUpdatable = gstoreIsForeignRelUpdatable;

	routine->PlanForeignModify	= gstorePlanForeignModify;
	routine->BeginForeignModify	= gstoreBeginForeignModify;
	routine->ExecForeignInsert	= gstoreExecForeignInsert;
	routine->EndForeignModify	= gstoreEndForeignModify;

	routine->PlanDirectModify	= gstorePlanDirectModify;
    routine->BeginDirectModify	= gstoreBeginDirectModify;
    routine->IterateDirectModify = gstoreIterateDirectModify;
    routine->EndDirectModify	= gstoreEndDirectModify;

	PG_RETURN_POINTER(routine);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_handler);

/*
 * pgstrom_startup_gstore_fdw
 */
static void
pgstrom_startup_gstore_fdw(void)
{
	bool		found;
	int			i;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	gstore_head = ShmemInitStruct("GPU Store Control Structure",
								  offsetof(GpuStoreHead,
										   gs_chunks[gstore_max_nchunks]),
								  &found);
	if (found)
		elog(ERROR, "Bug? shared memory for gstore_fdw already built");
	gstore_maps = MemoryContextAlloc(TopMemoryContext,
									 sizeof(GpuStoreMap) * gstore_max_nchunks);
	SpinLockInit(&gstore_head->lock);
	dlist_init(&gstore_head->free_chunks);
	for (i=0; i < GSTORE_CHUNK_HASH_NSLOTS; i++)
		dlist_init(&gstore_head->active_chunks[i]);
	for (i=0; i < gstore_max_nchunks; i++)
	{
		GpuStoreChunk  *gs_chunk = &gstore_head->gs_chunks[i];
		GpuStoreMap	   *gs_map = &gstore_maps[i];

		memset(gs_chunk, 0, sizeof(GpuStoreChunk));
		gs_chunk->gs_map = gs_map;
		gs_chunk->handle = UINT_MAX;

		gs_map->gs_chunk = gs_chunk;
		gs_map->dsm_seg = NULL;

		dlist_push_tail(&gstore_head->free_chunks, &gs_chunk->chain);
	}
}

/*
 * pgstrom_init_gstore_fdw
 */
void
pgstrom_init_gstore_fdw(void)
{
	DefineCustomIntVariable("pg_strom.gstore_max_nchunks",
							"maximum number of gstore_fdw relations",
							NULL,
							&gstore_max_nchunks,
							2048,
							1024,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	RequestAddinShmemSpace(MAXALIGN(offsetof(GpuStoreHead,
											 gs_chunks[gstore_max_nchunks])));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gstore_fdw;

	RegisterXactCallback(gstoreXactCallback, NULL);
}
