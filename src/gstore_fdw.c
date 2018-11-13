/*
 * gstore_fdw.c
 *
 * On GPU column based data store as FDW provider.
 * ----
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#include "cuda_plcuda.h"

/*
 * GpuStoreChunk - shared structure
 */
typedef struct
{
	dlist_node		chain;
	cl_uint			revision;
	pg_crc32		hash;
	Oid				database_oid;
	Oid				table_oid;
	TransactionId	xmax;
	TransactionId	xmin;
	bool			xmax_committed;
	bool			xmin_committed;
	cl_int			pinning;	/* CUDA device index */
	cl_int			format;		/* one of GSTORE_FDW_FORMAT__* */
	size_t			rawsize;	/* rawsize regardless of the internal format */
	size_t			nitems;		/* nitems regardless of the internal format */
	CUipcMemHandle	ipc_mhandle;
} GpuStoreChunk;

/*
 * GpuStoreHead - shared structure
 */
#define GSTORE_CHUNK_HASH_NSLOTS	97
typedef struct
{
	pg_atomic_uint32 revision_seed;
	pg_atomic_uint32 has_warm_chunks;
	slock_t			lock;
	dlist_head		free_chunks;
	dlist_head		active_chunks[GSTORE_CHUNK_HASH_NSLOTS];
	GpuStoreChunk	gs_chunks[FLEXIBLE_ARRAY_MEMBER];
} GpuStoreHead;

/*
 * GpuStoreBuffer - local buffer of the GPU device memory
 */
typedef struct
{
	TransactionId	xmin;
	TransactionId	xmax;
	CommandId		cid;
	bool			xmin_committed;
	bool			xmax_committed;
} MVCCAttrs;

typedef struct
{
	Oid				table_oid;	/* oid of the gstore_fdw */
	cl_int			pinning;	/* CUDA device index */
	cl_int			format;		/* one of GSTORE_FDW_FORMAT__* */
	cl_uint			revision;	/* revision number of the buffer */
	bool			read_only;	/* true, if read-write buffer is not ready */
	bool			is_dirty;	/* true, if any updates happen on the read-
								 * write buffer, thus read-only buffer is
								 * not uptodata any more. */
	MemoryContext	memcxt;		/* context for the buffers below */
	/* read-only buffer */
	union {
		kern_data_store *kds;	/* copy of GPU device memory, if any */
		void	   *buffer;
	} h;
	size_t			rawsize;
	/* read/write buffer */
	MVCCAttrs	   *cs_mvcc;	/* t_xmin/t_xmax/t_cid and flags */
	ccacheBuffer	cc_buf;		/* buffer for regular attributes */
} GpuStoreBuffer;

/*
 * gstoreScanState - state object for scan/insert/update/delete
 */
typedef struct
{
	GpuStoreBuffer *gs_buffer;
	cl_ulong		gs_index;
	AttrNumber		ctid_anum;	/* only UPDATE or DELETE */
} GpuStoreExecState;

/* ---- static functions ---- */
static void	gstore_fdw_table_options(Oid gstore_oid,
									int *p_pinning, int *p_format);
static void gstore_fdw_column_options(Oid gstore_oid, AttrNumber attnum,
									  int *p_compression);

/* ---- static variables ---- */
static int				gstore_max_relations;		/* GUC */
static shmem_startup_hook_type shmem_startup_next;
static object_access_hook_type object_access_next;
static GpuStoreHead	   *gstore_head = NULL;
static HTAB			   *gstore_buffer_htab = NULL;
static Oid				reggstore_type_oid = InvalidOid;

Datum pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_chunk_info(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_format(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_nitems(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_nattrs(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_rawsize(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_in(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_out(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_recv(PG_FUNCTION_ARGS);
Datum pgstrom_reggstore_send(PG_FUNCTION_ARGS);

/*
 * gstore_fdw_chunk_visibility - equivalent to HeapTupleSatisfiesMVCC,
 * but simplified for GpuStoreChunk because only commited chunks are written
 * to the shared memory object.
 */
static bool
gstore_fdw_chunk_visibility(GpuStoreChunk *gs_chunk, Snapshot snapshot)
{
	/* xmin is committed, but maybe not according to our snapshot */
	if (gs_chunk->xmin != FrozenTransactionId &&
		XidInMVCCSnapshot(gs_chunk->xmin, snapshot))
		return false;		/* treat as still in progress */
	/* by here, the inserting transaction has committed */
	if (!TransactionIdIsValid(gs_chunk->xmax))
		return true;	/* nobody deleted yet */
	/* xmax is committed, but maybe not according to our snapshot */
	if (XidInMVCCSnapshot(gs_chunk->xmax, snapshot))
		return true;
	/* xmax transaction committed */
	return false;
}

/*
 * gstore_fdw_tuple_visibility
 */
static bool
gstore_fdw_tuple_visibility(MVCCAttrs *mvcc, Snapshot snapshot)
{
	if (!mvcc->xmin_committed)
	{
		if (mvcc->xmin == InvalidTransactionId)
			return false;
		if (TransactionIdIsCurrentTransactionId(mvcc->xmin))
		{
			if (mvcc->cid >= snapshot->curcid)
				return false;	/* inserted after scan started */
			if (mvcc->xmax == InvalidTransactionId)
				return true;
			if (!TransactionIdIsCurrentTransactionId(mvcc->xmax))
			{
				/* deleting subtransaction must have aborted */
				mvcc->xmax = InvalidTransactionId;
				mvcc->xmax_committed = false;
				return true;
			}
			if (mvcc->cid >= snapshot->curcid)
				return true;	/* deleted after scan started */
			else
				return false;	/* deleted before scan started */
		}
		else if (XidInMVCCSnapshot(mvcc->xmin, snapshot))
			return false;
        else if (TransactionIdDidCommit(mvcc->xmin))
			mvcc->xmin_committed = true;
        else
        {
            /* it must have aborted or crashed */
			mvcc->xmin = InvalidTransactionId;
			return false;
		}
	}
	else
	{
		/* xmin is committed, but maybe not according to our snapshot */
		if (mvcc->xmin != FrozenTransactionId &&
			XidInMVCCSnapshot(mvcc->xmin, snapshot))
			return false;	/* treat as still in progress */
	}

	/* by here, the inserting transaction has committed */
	if (mvcc->xmax == InvalidTransactionId)
	{
		Assert(!mvcc->xmax_committed);
		return true;
	}
	if (!mvcc->xmax_committed)
	{
		if (TransactionIdIsCurrentTransactionId(mvcc->xmax))
		{
			if (mvcc->cid >= snapshot->curcid)
				return true;	/* deleted after scan started */
			else
				return false;	/* deleted before scan started */
		}
		if (XidInMVCCSnapshot(mvcc->xmax, snapshot))
			return true;
		if (!TransactionIdDidCommit(mvcc->xmax))
		{
			mvcc->xmax = InvalidTransactionId;
			mvcc->xmax_committed = false;
			return true;
		}
		/* xmax transaction committed*/
		mvcc->xmax_committed = true;
	}
	else
	{
		/* xmax is committed, but maybe not according to our snapshot */
		if (XidInMVCCSnapshot(mvcc->xmax, snapshot))
			return true;
	}
	/* xmax transaction committed */
	return false;
}

/*
 * gstore_fdw_visibility_bitmap
 */
static bits8 *
gstore_fdw_visibility_bitmap(GpuStoreBuffer *gs_buffer, size_t *p_nrooms)
{
	size_t		i, j, nitems = gs_buffer->cc_buf.nitems;
	size_t		nrooms = 0;
	bits8	   *rowmap;

	if (nitems == 0)
	{
		*p_nrooms = 0;
		return NULL;
	}

	rowmap = palloc0(BITMAPLEN(nitems));
	for (i=0, j=-1; i < nitems; i++)
	{
		MVCCAttrs  *mvcc = &gs_buffer->cs_mvcc[i];

		if ((i & (BITS_PER_BYTE-1)) == 0)
			j++;
		if (!TransactionIdIsCurrentTransactionId(mvcc->xmax))
		{
			/*
			 * Row is exist on the initial load (it means somebody others
			 * inserted or updated, then committed. gstore_fdw always takes
			 * exclusive lock towards concurrent writer operations (INSERT/
			 * UPDATE/DELETE), so no need to pay attention for the updates
			 * by the concurrent transactions.
			 */
			rowmap[j] |= (1 << (i & 7));
			nrooms++;
		}
	}
	*p_nrooms = nrooms;
	return rowmap;
}

/*
 * gstore_fdw_chunk_hashvalue
 */
static inline pg_crc32
gstore_fdw_chunk_hashvalue(Oid gstore_oid)
{
	pg_crc32	hash;

	INIT_LEGACY_CRC32(hash);
	COMP_LEGACY_CRC32(hash, &MyDatabaseId, sizeof(Oid));
	COMP_LEGACY_CRC32(hash, &gstore_oid, sizeof(Oid));
	FIN_LEGACY_CRC32(hash);

	return hash;
}

/*
 * gstore_fdw_lookup_chunk
 */
static GpuStoreChunk *
gstore_fdw_lookup_chunk_nolock(Oid gstore_oid, Snapshot snapshot)
{
	GpuStoreChunk *gs_chunk = NULL;
	pg_crc32	hash = gstore_fdw_chunk_hashvalue(gstore_oid);
	int			index = hash % GSTORE_CHUNK_HASH_NSLOTS;
	dlist_iter	iter;

	dlist_foreach(iter, &gstore_head->active_chunks[index])
	{
		GpuStoreChunk  *gs_temp = dlist_container(GpuStoreChunk,
												  chain, iter.cur);
		if (gs_temp->hash == hash &&
			gs_temp->database_oid == MyDatabaseId &&
			gs_temp->table_oid == gstore_oid &&
			gstore_fdw_chunk_visibility(gs_temp, snapshot))
		{
			if (!gs_chunk)
				gs_chunk = gs_temp;
			else
				elog(ERROR, "Bug? multiple GpuStoreChunks are visible");
		}
	}
	return gs_chunk;
}

static GpuStoreChunk *
gstore_fdw_lookup_chunk(Oid gstore_oid, Snapshot snapshot)
{
	GpuStoreChunk  *gs_chunk = NULL;

	SpinLockAcquire(&gstore_head->lock);
	PG_TRY();
	{
		gs_chunk = gstore_fdw_lookup_chunk_nolock(gstore_oid, snapshot);
	}
	PG_CATCH();
	{
		SpinLockRelease(&gstore_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&gstore_head->lock);

	return gs_chunk;
}

/*
 * gstore_fdw_insert_chunk - host-to-device DMA
 */
static void
gstore_fdw_insert_chunk(GpuStoreBuffer *gs_buffer, size_t nrooms)
{
	CUresult		rc;
	dlist_node	   *dnode;
	GpuStoreChunk  *gs_chunk;
	int				index;
	dlist_iter		iter;

	Assert(gs_buffer->pinning < numDevAttrs);

	/* setup GpuStoreChunk */
	SpinLockAcquire(&gstore_head->lock);
	if (dlist_is_empty(&gstore_head->free_chunks))
	{
		SpinLockRelease(&gstore_head->lock);
		elog(ERROR, "gstore_fdw: out of GpuStoreChunk strucure");
	}
	dnode = dlist_pop_head_node(&gstore_head->free_chunks);
	gs_chunk = dlist_container(GpuStoreChunk, chain, dnode);
	SpinLockRelease(&gstore_head->lock);

	gs_chunk->revision
		= pg_atomic_add_fetch_u32(&gstore_head->revision_seed, 1);
	gs_chunk->hash = gstore_fdw_chunk_hashvalue(gs_buffer->table_oid);
	gs_chunk->database_oid = MyDatabaseId;
	gs_chunk->table_oid = gs_buffer->table_oid;
	gs_chunk->xmax = InvalidTransactionId;
	gs_chunk->xmin = GetCurrentTransactionId();
	gs_chunk->pinning = gs_buffer->pinning;
	gs_chunk->format = gs_buffer->format;
	gs_chunk->rawsize = gs_buffer->rawsize;
	gs_chunk->nitems = nrooms;

	/* DMA to device */
	rc = gpuMemAllocPreserved(gs_buffer->pinning,
							  &gs_chunk->ipc_mhandle,
							  gs_buffer->rawsize);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAllocPreserved: %s", errorText(rc));

	PG_TRY();
	{
		gpuIpcMemCopyFromHost(gs_chunk->pinning,
							  gs_chunk->ipc_mhandle,
							  0,
							  gs_buffer->h.buffer,
							  gs_buffer->rawsize);
	}
	PG_CATCH();
	{
		gpuMemFreePreserved(gs_buffer->pinning,
							gs_chunk->ipc_mhandle);
		memset(gs_chunk, 0, sizeof(GpuStoreChunk));
		SpinLockAcquire(&gstore_head->lock);
		dlist_push_head(&gstore_head->free_chunks, &gs_chunk->chain);
		SpinLockRelease(&gstore_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	gs_buffer->revision = gs_chunk->revision;

	/* add GpuStoreChunk to the shared hash table */
	index = gs_chunk->hash % GSTORE_CHUNK_HASH_NSLOTS;
	SpinLockAcquire(&gstore_head->lock);
	dlist_foreach(iter, &gstore_head->active_chunks[index])
	{
		GpuStoreChunk  *gs_temp = dlist_container(GpuStoreChunk,
												  chain, iter.cur);
		if (gs_temp->hash == gs_chunk->hash &&
			gs_temp->database_oid == gs_chunk->database_oid &&
			gs_temp->table_oid == gs_chunk->table_oid &&
			gs_temp->xmax == InvalidTransactionId)
		{
			gs_temp->xmax = gs_chunk->xmin;
		}
	}
	dlist_push_head(&gstore_head->active_chunks[index],
					&gs_chunk->chain);
	pg_atomic_add_fetch_u32(&gstore_head->has_warm_chunks, 1);
	SpinLockRelease(&gstore_head->lock);
}

/*
 * gstore_fdw_release_chunk
 *
 * memo: must be called under the 'gstore_head->lock'
 */
static void
gstore_fdw_release_chunk(GpuStoreChunk *gs_chunk)
{
	dlist_delete(&gs_chunk->chain);
	gpuMemFreePreserved(gs_chunk->pinning,
						gs_chunk->ipc_mhandle);
	memset(gs_chunk, 0, sizeof(GpuStoreChunk));
	dlist_push_head(&gstore_head->free_chunks,
					&gs_chunk->chain);
}

/*
 * gstore_fdw_make_buffer_writable
 */
static void
gstore_fdw_make_buffer_writable(Relation frel, GpuStoreBuffer *gs_buffer)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	ccacheBuffer   *cc_buf = &gs_buffer->cc_buf;
	size_t			nrooms;
	size_t			nitems;
	size_t			i;

	/* already done? */
	if (!gs_buffer->read_only)
		return;
	/* calculation of nrooms */
	if (!gs_buffer->h.buffer)
	{
		nitems = 0;
		nrooms = 10000;
	}
	else if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
	{
		nitems = gs_buffer->h.kds->nitems;
		nrooms = gs_buffer->h.kds->nitems + 10000;
	}
	else
		elog(ERROR, "gstore_fdw: Bug? unknown buffer format: %d",
			 gs_buffer->format);
	gs_buffer->cs_mvcc = MemoryContextAllocHuge(gs_buffer->memcxt,
												sizeof(MVCCAttrs) * nrooms);
	ccache_setup_buffer(tupdesc,
						&gs_buffer->cc_buf,
						false,	/* no system columns */
						nrooms,
						gs_buffer->memcxt);
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, i);
		int		vl_compress;

		gstore_fdw_column_options(attr->attrelid, attr->attnum,
								  &vl_compress);
		gs_buffer->cc_buf.vl_compress[i] = vl_compress;
	}
	gs_buffer->cc_buf.nitems = nitems;
	if (nitems > 0)
	{
		if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
		{
			ccache_copy_buffer_from_kds(tupdesc, cc_buf,
										gs_buffer->h.kds,
										gs_buffer->memcxt);
		}
		else
			elog(ERROR, "gstore_fdw: Bug? unknown buffer format: %d",
				 gs_buffer->format);
		/* initial tuples are all visible */
		for (i=0; i < nitems; i++)
		{
			MVCCAttrs   *mvcc = &gs_buffer->cs_mvcc[i];

			mvcc->xmin = FrozenTransactionId;
			mvcc->xmax = InvalidTransactionId;
			mvcc->cid = 0;
			mvcc->xmin_committed = true;
			mvcc->xmax_committed = false;
		}
	}
	gs_buffer->read_only = false;
}

/*
 * gstore_fdw_create_buffer - make a local buffer of GpuStore
 */
static GpuStoreBuffer *
gstore_fdw_create_buffer(Relation frel, Snapshot snapshot)
{
	GpuStoreBuffer *gs_buffer = NULL;
	GpuStoreChunk  *gs_chunk = NULL;
	MemoryContext	memcxt = NULL;
	bool			found;

	if (!gstore_buffer_htab)
	{
		HASHCTL	hctl;
		int		flags;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(GpuStoreBuffer);
		hctl.hcxt = CacheMemoryContext;
		flags = HASH_ELEM | HASH_BLOBS | HASH_CONTEXT;

		gstore_buffer_htab = hash_create("GpuStoreBuffer HTAB",
										 100, &hctl, flags);
	}

	gs_buffer = hash_search(gstore_buffer_htab,
							&RelationGetRelid(frel),
							HASH_ENTER,
							&found);
	if (found)
	{
		Assert(gs_buffer->table_oid == RelationGetRelid(frel));
		gs_chunk = gstore_fdw_lookup_chunk(RelationGetRelid(frel), snapshot);
		if (!gs_chunk)
		{
			if (gs_buffer->revision == 0)
				return gs_buffer;	/* no gs_chunk right now */
		}
		else if (gs_buffer->revision == gs_chunk->revision)
			return gs_buffer;		/* ok local buffer is up to date */
		/* oops, local cache is older than in-GPU image... */
		MemoryContextDelete(gs_buffer->memcxt);
	}
	else
	{
		gs_chunk = gstore_fdw_lookup_chunk(RelationGetRelid(frel), snapshot);
	}

	/*
	 * Local buffer is not found, or invalid. So, re-initialize it again.
	 */
	PG_TRY();
	{
		cl_int		pinning;
		cl_int		format;
		cl_uint		revision;

		memcxt = AllocSetContextCreate(CacheMemoryContext,
									   "GpuStoreBuffer",
									   ALLOCSET_DEFAULT_SIZES);
		if (!gs_chunk)
		{
			gstore_fdw_table_options(RelationGetRelid(frel),
									 &pinning, &format);
			gs_buffer->pinning   = pinning;
			gs_buffer->format    = format;
			gs_buffer->revision  = 0;
			gs_buffer->read_only = true;
			gs_buffer->is_dirty  = false;
			gs_buffer->memcxt    = memcxt;
			gs_buffer->h.buffer  = NULL;
			gs_buffer->cs_mvcc   = NULL;
			memset(&gs_buffer->cc_buf, 0, sizeof(ccacheBuffer));
			gstore_fdw_make_buffer_writable(frel, gs_buffer);
		}
		else
		{
			size_t		rawsize;
			void	   *hbuf;

			pinning  = gs_chunk->pinning;
			format   = gs_chunk->format;
			revision = gs_chunk->revision;
			rawsize  = gs_chunk->rawsize;
			hbuf = MemoryContextAllocHuge(memcxt, rawsize);

			gpuIpcMemCopyToHost(hbuf,
								gs_chunk->pinning,
								gs_chunk->ipc_mhandle,
								0,
								rawsize);

			Assert(gs_buffer->table_oid == RelationGetRelid(frel));
			gs_buffer->pinning   = pinning;
			gs_buffer->format    = format;
			gs_buffer->revision  = revision;
			gs_buffer->read_only = true;
			gs_buffer->is_dirty  = false;
			gs_buffer->memcxt    = memcxt;
			gs_buffer->h.buffer  = hbuf;
			gs_buffer->rawsize   = rawsize;
			gs_buffer->cs_mvcc   = NULL;
			memset(&gs_buffer->cc_buf, 0, sizeof(ccacheBuffer));
		}
	}
	PG_CATCH();
	{
		if (gs_buffer)
		{
			hash_search(gstore_buffer_htab,
						&RelationGetRelid(frel),
						HASH_REMOVE,
						&found);
			Assert(found);
		}
		if (memcxt)
			MemoryContextDelete(memcxt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return gs_buffer;
}

/*
 * gstoreGetForeignRelSize
 */
static void
gstoreGetForeignRelSize(PlannerInfo *root,
						RelOptInfo *baserel,
						Oid ftable_oid)
{
	Snapshot		snapshot;
	GpuStoreChunk  *gs_chunk;

	snapshot = RegisterSnapshot(GetTransactionSnapshot());
	gs_chunk = gstore_fdw_lookup_chunk(ftable_oid, snapshot);
	UnregisterSnapshot(snapshot);

	baserel->rows	= (gs_chunk ? gs_chunk->nitems : 0);
	baserel->pages	= (gs_chunk ? gs_chunk->rawsize / BLCKSZ : 0);
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
 * gstoreAddForeignUpdateTargets
 */
static void
gstoreAddForeignUpdateTargets(Query *parsetree,
							  RangeTblEntry *target_rte,
							  Relation target_relation)
{
	Var			*var;
	TargetEntry *tle;

	/*
	 * We carry row_index as ctid system column
	 */

	/* Make a Var representing the desired value */
	var = makeVar(parsetree->resultRelation,
				  SelfItemPointerAttributeNumber,
				  TIDOID,
				  -1,
				  InvalidOid,
				  0);

	/* Wrap it in a resjunk TLE with the right name ... */
	tle = makeTargetEntry((Expr *) var,
						  list_length(parsetree->targetList) + 1,
						  "ctid",
						  true);

	/* ... and add it to the query's targetlist */
	parsetree->targetList = lappend(parsetree->targetList, tle);
}

/*
 * gstoreBeginForeignScan
 */
static void
gstoreBeginForeignScan(ForeignScanState *node, int eflags)
{
	EState	   *estate = node->ss.ps.state;
	GpuStoreExecState *gstate;

	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;

	if (!IsMVCCSnapshot(estate->es_snapshot))
		elog(ERROR, "cannot scan gstore_fdw table without MVCC snapshot");

	gstate = palloc0(sizeof(GpuStoreExecState));
	node->fdw_state = (void *) gstate;
}

/*
 * gstoreIterateForeignScan
 */
static TupleTableSlot *
gstoreIterateForeignScan(ForeignScanState *node)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) node->fdw_state;
	Relation		frel = node->ss.ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	EState		   *estate = node->ss.ps.state;
	Snapshot		snapshot = estate->es_snapshot;
	ForeignScan	   *fscan = (ForeignScan *)node->ss.ps.plan;
	GpuStoreBuffer *gs_buffer;
	size_t			row_index;

	ExecClearTuple(slot);
	if (!gstate->gs_buffer)
		gstate->gs_buffer = gstore_fdw_create_buffer(frel, snapshot);
	gs_buffer = gstate->gs_buffer;
lnext:
	row_index = gstate->gs_index++;
	if (gs_buffer->h.buffer)
	{
		/* read from read-only buffer */
		switch (gs_buffer->format)
		{
			case GSTORE_FDW_FORMAT__PGSTROM:
				if (!KDS_fetch_tuple_column(slot,
											gs_buffer->h.kds,
											row_index))
					ExecClearTuple(slot);
				break;

			default:
				elog(ERROR, "gstore_fdw: unexpected format: %d",
					 gs_buffer->format);
				break;
		}
	}
	else if (row_index < gs_buffer->cc_buf.nitems)
	{
		ccacheBuffer   *cc_buf = &gs_buffer->cc_buf;
		cl_int			j;

		if (!gstore_fdw_tuple_visibility(&gs_buffer->cs_mvcc[row_index],
										 snapshot))
			goto lnext;

		/* OK, tuple is visible */
		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
			vl_dict_key	*vkey;
			int			unitsz;
			void	   *addr;

			if (att_isnull(j, cc_buf->nullmap[j]))
			{
				slot->tts_isnull[j] = true;
				continue;
			}
			slot->tts_isnull[j] = false;
			if (attr->attlen < 0)
			{
				vkey = ((vl_dict_key **)cc_buf->values[j])[row_index];
				slot->tts_values[j] = PointerGetDatum(vkey->vl_datum);
			}
			else
			{
				unitsz = att_align_nominal(attr->attlen,
										   attr->attalign);
				addr = (char *)cc_buf->values[j] + unitsz * row_index;
				if (!attr->attbyval)
					slot->tts_values[j] = PointerGetDatum(addr);
				else if (attr->attlen == sizeof(cl_char))
					slot->tts_values[j] = CharGetDatum(*((cl_char *)addr));
				else if (attr->attlen == sizeof(cl_short))
					slot->tts_values[j] = Int16GetDatum(*((cl_short *)addr));
				else if (attr->attlen == sizeof(cl_int))
					slot->tts_values[j] = Int32GetDatum(*((cl_int *)addr));
				else if (attr->attlen == sizeof(cl_long))
					slot->tts_values[j] = Int64GetDatum(*((cl_long *)addr));
				else
					elog(ERROR, "gstore_fdw: unexpected attlen: %d",
						 attr->attlen);
			}
		}
		ExecStoreVirtualTuple(slot);
	}
	else
	{
		ExecClearTuple(slot);
	}

	/*
	 * Add system column information if required
	 */
	if (fscan->fsSystemCol && !TupIsNull(slot))
	{
		HeapTuple   tup = ExecMaterializeSlot(slot);

		tup->t_self.ip_blkid.bi_hi = (row_index >> 32) & 0x0000ffff;
		tup->t_self.ip_blkid.bi_lo = (row_index >> 16) & 0x0000ffff;
		tup->t_self.ip_posid       = (row_index & 0x0000ffff);
		tup->t_tableOid = RelationGetRelid(frel);
		if (gs_buffer->read_only)
		{
			tup->t_data->t_choice.t_heap.t_xmin = FrozenTransactionId;
			tup->t_data->t_choice.t_heap.t_xmax = InvalidTransactionId;
			tup->t_data->t_choice.t_heap.t_field3.t_cid = 0;
		}
		else
		{
			MVCCAttrs  *mvcc = &gs_buffer->cs_mvcc[row_index];
			tup->t_data->t_choice.t_heap.t_xmin = mvcc->xmin;
			tup->t_data->t_choice.t_heap.t_xmax = mvcc->xmax;
			tup->t_data->t_choice.t_heap.t_field3.t_cid = mvcc->cid;
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
	GpuStoreExecState *gstate = (GpuStoreExecState *) node->fdw_state;

	gstate->gs_index = 0;
}

/*
 * gstoreEndForeignScan
 */
static void
gstoreEndForeignScan(ForeignScanState *node)
{
	GpuStoreExecState  *gstate = (GpuStoreExecState *) node->fdw_state;

	if (gstate)
	{
		GpuStoreBuffer *gs_buffer = gstate->gs_buffer;
		/*
		 * Once buffer gets dirty, read-only buffer shall not
		 * contain valid image no longer.
		 */
		if (gs_buffer->is_dirty && gs_buffer->h.buffer)
		{
			pfree(gs_buffer->h.buffer);
			gs_buffer->h.buffer = NULL;
		}
	}
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

	if (operation != CMD_INSERT &&
		operation != CMD_UPDATE &&
		operation != CMD_DELETE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("gstore_fdw: not a supported operation")));
	return NIL;
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
	GpuStoreExecState *gstate = palloc0(sizeof(GpuStoreExecState));
	Relation	frel = rrinfo->ri_RelationDesc;
	CmdType		operation = mtstate->operation;

	/*
	 * NOTE: gstore_fdw does not support update operations by multiple
	 * concurrent transactions. So, we require stronger lock than usual
	 * INSERT/UPDATE/DELETE operations. It may lead unexpected deadlock,
	 * in spite of the per-tuple update capability.
	 */
	LockRelationOid(RelationGetRelid(frel), ShareUpdateExclusiveLock);

	/* Find the ctid resjunk column in the subplan's result */
	if (operation == CMD_UPDATE || operation == CMD_DELETE)
	{
		Plan	   *subplan = mtstate->mt_plans[subplan_index]->plan;
		AttrNumber	ctid_anum;

		ctid_anum = ExecFindJunkAttributeInTlist(subplan->targetlist, "ctid");
		if (!AttributeNumberIsValid(ctid_anum))
			elog(ERROR, "could not find junk ctid column");
		gstate->ctid_anum = ctid_anum;
	}
	rrinfo->ri_FdwState = gstate;
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
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Relation		frel = rrinfo->ri_RelationDesc;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	Snapshot		snapshot = estate->es_snapshot;
	GpuStoreBuffer *gs_buffer;
	ccacheBuffer   *cc_buf;
	MVCCAttrs	   *mvcc;
	size_t			index;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = gstore_fdw_create_buffer(frel, snapshot);
	gs_buffer = gstate->gs_buffer;
	if (gs_buffer->read_only)
		gstore_fdw_make_buffer_writable(frel, gs_buffer);
	cc_buf = &gs_buffer->cc_buf;

	/* expand buffer on demand */
	while (cc_buf->nitems >= cc_buf->nrooms)
	{
		ccache_expand_buffer(tupdesc, cc_buf, gs_buffer->memcxt);
		gs_buffer->cs_mvcc = repalloc_huge(gs_buffer->cs_mvcc,
										   sizeof(MVCCAttrs) * cc_buf->nrooms);
	}
	/* write out a tuple */
	slot_getallattrs(slot);
	ccache_buffer_append_row(RelationGetDescr(frel),
							 cc_buf,
							 NULL,	/* no system columns */
							 slot->tts_isnull,
							 slot->tts_values,
							 gs_buffer->memcxt);
	index = cc_buf->nitems++;
	mvcc = &gs_buffer->cs_mvcc[index];
	memset(mvcc, 0, sizeof(MVCCAttrs));
	mvcc->xmin = GetCurrentTransactionId();
	mvcc->xmax = InvalidTransactionId;
	mvcc->cid  = snapshot->curcid;
	gs_buffer->is_dirty = true;

	return slot;
}

/*
 * gstoreExecForeignUpdate
 */
static TupleTableSlot *
gstoreExecForeignUpdate(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Relation		frel = rrinfo->ri_RelationDesc;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	Snapshot		snapshot = estate->es_snapshot;
	GpuStoreBuffer *gs_buffer;
	ccacheBuffer   *cc_buf;
	Datum			datum;
	bool			isnull;
	ItemPointer		t_self;
	MVCCAttrs	   *mvcc;
	size_t			index;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = gstore_fdw_create_buffer(frel, snapshot);
	gs_buffer = gstate->gs_buffer;
	if (gs_buffer->read_only)
		gstore_fdw_make_buffer_writable(frel, gs_buffer);
	cc_buf = &gs_buffer->cc_buf;

	/* extract ctid from a resjunk column */
	datum = ExecGetJunkAttribute(planSlot,
								 gstate->ctid_anum,
								 &isnull);
	if (isnull)
		elog(ERROR, "gstore_fdw: ctid is null");
	t_self = (ItemPointer)DatumGetPointer(datum);
	index = ((cl_ulong)t_self->ip_blkid.bi_hi << 32 |
			 (cl_ulong)t_self->ip_blkid.bi_lo << 16 |
			 (cl_ulong)t_self->ip_posid);
	if (index >= gs_buffer->cc_buf.nitems)
		elog(ERROR, "gstore_fdw: UPDATE row out of range (%lu of %zu)",
			 index, gs_buffer->cc_buf.nitems);
	mvcc = &gs_buffer->cs_mvcc[index];
	mvcc->xmax = GetCurrentTransactionId();
	mvcc->cid  = snapshot->curcid;

	/* insert a new version */
	while (cc_buf->nitems >= cc_buf->nrooms)
	{
		ccache_expand_buffer(tupdesc, cc_buf, gs_buffer->memcxt);
		gs_buffer->cs_mvcc = repalloc_huge(gs_buffer->cs_mvcc,
										   sizeof(HeapTupleFields) *
										   cc_buf->nrooms);
	}
	slot_getallattrs(slot);
	ccache_buffer_append_row(RelationGetDescr(frel),
							 cc_buf,
							 NULL,	/* no system columns */
							 slot->tts_isnull,
							 slot->tts_values,
							 gs_buffer->memcxt);
	index = cc_buf->nitems++;
	mvcc = &gs_buffer->cs_mvcc[index];
	memset(mvcc, 0, sizeof(MVCCAttrs));
	mvcc->xmin = GetCurrentTransactionId();
	mvcc->xmax = InvalidTransactionId;
	mvcc->cid  = snapshot->curcid;

	/* make buffer dirty */
	gs_buffer->is_dirty = true;

	return slot;
}

/*
 * gstoreExecForeignDelete
 */
static TupleTableSlot *
gstoreExecForeignDelete(EState *estate,
						ResultRelInfo *rrinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	Relation		frel = rrinfo->ri_RelationDesc;
	Snapshot		snapshot = estate->es_snapshot;
	GpuStoreBuffer *gs_buffer;
	Datum			datum;
	bool			isnull;
	ItemPointer		t_self;
	MVCCAttrs	   *mvcc;
	cl_ulong		index;

	if (snapshot->curcid > INT_MAX)
		elog(ERROR, "gstore_fdw: too much sub-transactions");

	if (!gstate->gs_buffer)
		gstate->gs_buffer = gstore_fdw_create_buffer(frel, snapshot);
	gs_buffer = gstate->gs_buffer;
	if (gs_buffer->read_only)
		gstore_fdw_make_buffer_writable(frel, gs_buffer);

	/* extract ctid from a resjunk column */
	datum = ExecGetJunkAttribute(planSlot,
								 gstate->ctid_anum,
								 &isnull);
	if (isnull)
		elog(ERROR, "gstore_fdw: ctid is null");
	t_self = (ItemPointer)DatumGetPointer(datum);
	index = ((cl_ulong)t_self->ip_blkid.bi_hi << 32 |
			 (cl_ulong)t_self->ip_blkid.bi_lo << 16 |
			 (cl_ulong)t_self->ip_posid);
	if (index >= gs_buffer->cc_buf.nitems)
		elog(ERROR, "gstore_fdw: DELETE row out of range (%lu of %zu)",
			 index, gs_buffer->cc_buf.nitems);
	/* negative command id means the tuple was removed */
	mvcc = &gs_buffer->cs_mvcc[index];
	mvcc->xmax = GetCurrentTransactionId();
	mvcc->cid  = snapshot->curcid;

	/* make buffer dirty */
	gs_buffer->is_dirty = true;

	return slot;
}

/*
 * gstoreEndForeignModify
 */
static void
gstoreEndForeignModify(EState *estate,
					   ResultRelInfo *rrinfo)
{
	GpuStoreExecState *gstate = (GpuStoreExecState *) rrinfo->ri_FdwState;
	GpuStoreBuffer *gs_buffer = gstate->gs_buffer;

	/* release read-only buffer, if it is not up-to-date */
	if (gs_buffer &&
		gs_buffer->h.buffer &&
		gs_buffer->is_dirty)
	{
		pfree(gs_buffer->h.buffer);
		gs_buffer->h.buffer = NULL;
	}
}

/*
 * gstore_fdw_alloc_pgstrom_buffer
 */
static void
gstore_fdw_alloc_pgstrom_buffer(Relation frel,
								GpuStoreBuffer *gs_buffer,
								bits8 *rowmap, size_t nrooms)
{
	TupleDesc	tupdesc = RelationGetDescr(frel);
	ccacheBuffer *cc_buf = &gs_buffer->cc_buf;
	kern_data_store *kds;
	size_t		length;
	int			j, ncols;

	Assert(tupdesc->natts == gs_buffer->cc_buf.nattrs);
	/* 1. size estimation */
	ncols = tupdesc->natts + NumOfSystemAttrs;
	length = KDS_CALCULATE_HEAD_LENGTH(ncols, true);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		if (attr->attisdropped)
			continue;
		if (attr->attlen < 0)
		{
			length += (MAXALIGN(sizeof(cl_uint) * nrooms) +
					   MAXALIGN(cc_buf->extra_sz[j]));
		}
		else
		{
			int		unitsz = att_align_nominal(attr->attlen,
											   attr->attalign);
			length += MAXALIGN(unitsz * nrooms);
			if (cc_buf->hasnull[j])
				length += MAXALIGN(BITMAPLEN(nrooms));
		}
	}
	/* 2. allocation */
	kds = MemoryContextAllocHuge(gs_buffer->memcxt, length);

	/* 3. write out kern_data_store */
	ccache_copy_buffer_to_kds(kds, tupdesc, cc_buf, rowmap, nrooms);
	Assert(kds->length <= length);

	gs_buffer->rawsize = kds->length;
	gs_buffer->h.kds = kds;
}

/*
 * gstoreXactCallbackOnPreCommit
 */
static void
gstoreXactCallbackOnPreCommit(void)
{
	HASH_SEQ_STATUS	status;
	GpuStoreBuffer *gs_buffer;

	if (!gstore_buffer_htab)
		return;

	hash_seq_init(&status, gstore_buffer_htab);
	while ((gs_buffer = hash_seq_search(&status)) != NULL)
	{
		Relation	frel;
		bits8	   *rowmap;
		size_t		nrooms = gs_buffer->cc_buf.nitems;

		/* any writes happen? */
		if (!gs_buffer->is_dirty)
			continue;
		/* release read-only buffer if any */
		if (gs_buffer->h.buffer)
		{
			pfree(gs_buffer->h.buffer);
			gs_buffer->h.buffer = NULL;
		}
		/* check visibility for each rows (if any) */
		rowmap = gstore_fdw_visibility_bitmap(gs_buffer, &nrooms);

		/*
		 * once all the rows are removed from the gstore_fdw, we don't
		 * add new version of GpuStoreChunk/GpuStoreBuffer.
		 * Older version will be removed when it becomes invisible from
		 * all the transactions.
		 */
		if (nrooms == 0)
		{
			Oid			gstore_oid = gs_buffer->table_oid;
			pg_crc32	hash = gstore_fdw_chunk_hashvalue(gstore_oid);
			int			index = hash % GSTORE_CHUNK_HASH_NSLOTS;
			dlist_iter	iter;
			bool		found;

			SpinLockAcquire(&gstore_head->lock);
			dlist_foreach(iter, &gstore_head->active_chunks[index])
			{
				GpuStoreChunk  *gs_temp = dlist_container(GpuStoreChunk,
														  chain, iter.cur);
				if (gs_temp->hash == hash &&
					gs_temp->database_oid == MyDatabaseId &&
					gs_temp->table_oid == gstore_oid &&
					gs_temp->xmax == InvalidTransactionId)
				{
					gs_temp->xmax = GetCurrentTransactionId();
				}
			}
			pg_atomic_add_fetch_u32(&gstore_head->has_warm_chunks, 1);
			SpinLockRelease(&gstore_head->lock);
			/* also remove the buffer */
			pfree(gs_buffer->cs_mvcc);
			ccache_release_buffer(&gs_buffer->cc_buf);
			hash_search(gstore_buffer_htab,
						&gstore_oid,
						HASH_REMOVE,
						&found);
			Assert(found);
			continue;
		}

		/* construction of new version of GPU device memory image */
		frel = heap_open(gs_buffer->table_oid, NoLock);
		if (gs_buffer->format == GSTORE_FDW_FORMAT__PGSTROM)
			gstore_fdw_alloc_pgstrom_buffer(frel, gs_buffer, rowmap, nrooms);
		else
			elog(ERROR, "gstore_fdw: unknown format %d", gs_buffer->format);
		heap_close(frel, NoLock);
		pfree(rowmap);

		/* host-to-device DMA */
		gstore_fdw_insert_chunk(gs_buffer, nrooms);

		/* release read-write buffer */
		pfree(gs_buffer->cs_mvcc);
		gs_buffer->cs_mvcc = NULL;
		ccache_release_buffer(&gs_buffer->cc_buf);

		gs_buffer->read_only = true;
		gs_buffer->is_dirty = false;
	}
}

/*
 * gstoreXactCallbackOnAbort - clear all the local buffering
 */
static void
gstoreXactCallbackOnAbort(void)
{
	HASH_SEQ_STATUS	status;
	GpuStoreBuffer *gs_buffer;

	if (gstore_buffer_htab)
	{
		hash_seq_init(&status, gstore_buffer_htab);
		while ((gs_buffer = hash_seq_search(&status)) != NULL)
			MemoryContextDelete(gs_buffer->memcxt);
		hash_destroy(gstore_buffer_htab);
		gstore_buffer_htab = NULL;
	}
}

/*
 * gstoreXactCallbackPerChunk
 */
static bool
gstoreOnXactCallbackPerChunk(bool is_commit,
							 GpuStoreChunk *gs_chunk,
							 TransactionId oldestXmin)
{
	if (TransactionIdIsCurrentTransactionId(gs_chunk->xmax))
	{
		if (is_commit)
			gs_chunk->xmax_committed = true;
		else
			gs_chunk->xmax = InvalidTransactionId;
	}
	if (TransactionIdIsCurrentTransactionId(gs_chunk->xmin))
	{
		if (is_commit)
			gs_chunk->xmin_committed = true;
		else
		{
			gstore_fdw_release_chunk(gs_chunk);
			return false;
		}
	}

	if (TransactionIdIsValid(gs_chunk->xmax))
	{
		/* someone tried to delete chunk, but not commited yet */
		if (!gs_chunk->xmax_committed)
			return true;
		/*
		 * chunk deletion is commited, but some open transactions may
		 * still reference the chunk
		 */
		if (!TransactionIdPrecedes(gs_chunk->xmax, oldestXmin))
			return true;

		/* Otherwise, GpuStoreChunk can be released immediately */
		gstore_fdw_release_chunk(gs_chunk);
	}
	else if (TransactionIdIsNormal(gs_chunk->xmin))
	{
		/* someone tried to insert chunk, but not commited yet */
		if (!gs_chunk->xmin_committed)
			return true;
		/*
		 * chunk insertion is commited, but some open transaction may
		 * need MVCC style visibility control
		 */
		if (!TransactionIdPrecedes(gs_chunk->xmin, oldestXmin))
			return true;

		/* Otherwise, GpuStoreChunk can be visible to everybody */
		gs_chunk->xmin = FrozenTransactionId;
	}
	else if (!TransactionIdIsValid(gs_chunk->xmin))
	{
		/* GpuChunk insertion aborted */
		gstore_fdw_release_chunk(gs_chunk);
	}
	return false;
}

/*
 * gstoreXactCallback
 */
static void
gstoreXactCallback(XactEvent event, void *arg)
{
	TransactionId oldestXmin;
	bool		is_commit;
	bool		meet_warm_chunks = false;
	cl_int		i;

	switch (event)
	{
		case XACT_EVENT_PRE_COMMIT:
			gstoreXactCallbackOnPreCommit();
			return;
		case XACT_EVENT_COMMIT:
			is_commit = true;
			break;
		case XACT_EVENT_ABORT:
			gstoreXactCallbackOnAbort();
			is_commit = false;
			break;
		default:
			/* do nothing */
			return;
	}
#if 0
	elog(INFO, "gstoreXactCallback xid=%u (oldestXmin=%u)",
		 GetCurrentTransactionIdIfAny(), oldestXmin);
#endif
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

			if (gstoreOnXactCallbackPerChunk(is_commit, gs_chunk, oldestXmin))
				meet_warm_chunks = true;
		}
	}
	if (!meet_warm_chunks)
		pg_atomic_write_u32(&gstore_head->has_warm_chunks, 0);
	SpinLockRelease(&gstore_head->lock);
}

#if 0
/*
 * gstoreSubXactCallback - just for debug
 */
static void
gstoreSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
					  SubTransactionId parentSubid, void *arg)
{
	elog(INFO, "gstoreSubXactCallback event=%s my_xid=%u pr_xid=%u",
		 (event == SUBXACT_EVENT_START_SUB ? "StartSub" :
		  event == SUBXACT_EVENT_COMMIT_SUB ? "CommitSub" :
		  event == SUBXACT_EVENT_ABORT_SUB ? "AbortSub" :
		  event == SUBXACT_EVENT_PRE_COMMIT_SUB ? "PreCommitSub" : "???"),
		 mySubid, parentSubid);
}
#endif

/*
 * relation_is_gstore_fdw
 */
static bool
relation_is_gstore_fdw(Oid table_oid)
{
	HeapTuple	tup;
	Oid			fserv_oid;
	Oid			fdw_oid;
	Oid			handler_oid;
	PGFunction	handler_fn;
	Datum		datum;
	char	   *prosrc;
	char	   *probin;
	bool		isnull;
	/* it should be foreign table, of course */
	if (get_rel_relkind(table_oid) != RELKIND_FOREIGN_TABLE)
		return false;
	/* pull OID of foreign-server */
	tup = SearchSysCache1(FOREIGNTABLEREL, ObjectIdGetDatum(table_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign table %u", table_oid);
	fserv_oid = ((Form_pg_foreign_table) GETSTRUCT(tup))->ftserver;
	ReleaseSysCache(tup);

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
 * gstore_fdw_table_options
 */
static void
__gstore_fdw_table_options(List *options,
						  int *p_pinning,
						  int *p_format)
{
	ListCell   *lc;
	int			pinning = -1;
	int			format = -1;

	foreach (lc, options)
	{
		DefElem	   *defel = lfirst(lc);

		if (strcmp(defel->defname, "pinning") == 0)
		{
			if (pinning >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"pinning\" option appears twice")));
			pinning = atoi(defGetString(defel));
			if (pinning < 0 || pinning >= numDevAttrs)
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("\"pinning\" on unavailable GPU device")));
		}
		else if (strcmp(defel->defname, "format") == 0)
		{
			char   *format_name;

			if (format >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"format\" option appears twice")));
			format_name = defGetString(defel);
			if (strcmp(format_name, "pgstrom") == 0 ||
				strcmp(format_name, "default") == 0)
				format = GSTORE_FDW_FORMAT__PGSTROM;
			else
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("gstore_fdw: format \"%s\" is unknown",
								format_name)));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_SYNTAX_ERROR),
					 errmsg("gstore_fdw: unknown option \"%s\"",
							defel->defname)));
		}
	}
	if (pinning < 0)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("gstore_fdw: No pinning GPU device"),
				 errhint("use 'pinning' option to specify GPU device")));

	/* put default if not specified */
	if (format < 0)
		format = GSTORE_FDW_FORMAT__PGSTROM;

	/* result the results */
	if (p_pinning)
		*p_pinning = pinning;
	if (p_format)
		*p_format = format;
}

static void
gstore_fdw_table_options(Oid gstore_oid, int *p_pinning, int *p_format)
{
	HeapTuple	tup;
	Datum		datum;
	bool		isnull;
	List	   *options = NIL;

	tup = SearchSysCache1(FOREIGNTABLEREL, ObjectIdGetDatum(gstore_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for foreign table %u", gstore_oid);
	datum = SysCacheGetAttr(FOREIGNTABLEREL, tup,
							Anum_pg_foreign_table_ftoptions,
							&isnull);
	if (!isnull)
		options = untransformRelOptions(datum);
	__gstore_fdw_table_options(options, p_pinning, p_format);
	ReleaseSysCache(tup);
}

/*
 * gstore_fdw_column_options
 */
static void
__gstore_fdw_column_options(List *options, int *p_compression)
{
	ListCell   *lc;
	char	   *temp;
	int			compression = -1;

	foreach (lc, options)
	{
		DefElem	   *defel = lfirst(lc);

		if (strcmp(defel->defname, "compression") == 0)
		{
			if (compression >= 0)
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("\"compression\" option appears twice")));
			temp = defGetString(defel);
			if (pg_strcasecmp(temp, "none") == 0)
				compression = GSTORE_COMPRESSION__NONE;
			else if (pg_strcasecmp(temp, "pglz") == 0)
				compression = GSTORE_COMPRESSION__PGLZ;
			else
				ereport(ERROR,
						(errcode(ERRCODE_SYNTAX_ERROR),
						 errmsg("unknown compression logic: %s", temp)));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_SYNTAX_ERROR),
					 errmsg("gstore_fdw: unknown option \"%s\"",
							defel->defname)));
		}
	}
	/* set default, if no valid options were supplied */
	if (compression < 0)
		compression = GSTORE_COMPRESSION__NONE;

	/* set results */
	if (p_compression)
		*p_compression = compression;
}

static void
gstore_fdw_column_options(Oid gstore_oid, AttrNumber attnum,
						  int *p_compression)
{
	List	   *options = GetForeignColumnOptions(gstore_oid, attnum);

	__gstore_fdw_column_options(options, p_compression);
}

/*
 * gstore_fdw_post_alter
 */
static void
gstore_fdw_post_alter(Oid relid, AttrNumber attnum)
{
	GpuStoreBuffer *gs_buffer;
	GpuStoreChunk  *gs_chunk;
	bool			found;

	/* not a gstore_fdw foreign-table */
	if (!relation_is_gstore_fdw(relid))
		return;

	/* we don't allow ALTER FOREIGN TABLE onto non-empty gstore_fdw */
	if (gstore_buffer_htab)
	{
		gs_buffer = hash_search(gstore_buffer_htab,
								&relid,
								HASH_FIND,
								&found);
		if (gs_buffer)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("gstore_fdw: unable to run ALTER FOREIGN TABLE for non-empty gstore_fdw table")));
	}

	gs_chunk = gstore_fdw_lookup_chunk(relid, GetActiveSnapshot());
	if (gs_chunk)
	{
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("gstore_fdw: unable to run ALTER FOREIGN TABLE for non-empty gstore_fdw table")));
	}
}

/*
 * gstore_fdw_post_drop
 */
static void
gstore_fdw_post_drop(Oid relid, AttrNumber attnum)
{
	GpuStoreChunk *gs_chunk;
	pg_crc32	hash = gstore_fdw_chunk_hashvalue(relid);
	int			index = hash % GSTORE_CHUNK_HASH_NSLOTS;
	dlist_iter	iter;

	SpinLockAcquire(&gstore_head->lock);
	dlist_foreach(iter, &gstore_head->active_chunks[index])
	{
		gs_chunk = dlist_container(GpuStoreChunk, chain, iter.cur);

		if (gs_chunk->hash == hash &&
			gs_chunk->database_oid == MyDatabaseId &&
			gs_chunk->table_oid == relid &&
			gs_chunk->xmax == InvalidTransactionId)
		{
			gs_chunk->xmax = GetCurrentTransactionId();
		}
	}
	pg_atomic_add_fetch_u32(&gstore_head->has_warm_chunks, 1);
	SpinLockRelease(&gstore_head->lock);
}

/*
 * gstore_fdw_object_access
 */
static void
gstore_fdw_object_access(ObjectAccessType access,
						 Oid classId,
						 Oid objectId,
						 int subId,
						 void *__arg)
{
	if (object_access_next)
		(*object_access_next)(access, classId, objectId, subId, __arg);

	switch (access)
	{
		case OAT_POST_CREATE:
			if (classId == RelationRelationId)
			{
				ObjectAccessPostCreate *arg = __arg;

				if (arg->is_internal)
					break;
				/* A new gstore_fdw table is obviously empty */
				if (subId != 0)
					gstore_fdw_post_alter(objectId, subId);
			}
			break;

		case OAT_POST_ALTER:
			if (classId == RelationRelationId)
			{
				ObjectAccessPostAlter  *arg = __arg;

				if (arg->is_internal)
					break;
				gstore_fdw_post_alter(objectId, subId);
			}
			break;

		case OAT_DROP:
			if (classId == RelationRelationId)
			{
				ObjectAccessDrop	   *arg = __arg;

				if ((arg->dropflags & PERFORM_DELETION_INTERNAL) != 0)
					break;

				if (subId == 0)
				{
					if (relation_is_gstore_fdw(objectId))
						gstore_fdw_post_drop(objectId, subId);
				}
				else
					gstore_fdw_post_alter(objectId, subId);
			}
			break;

		default:
			/* do nothing */
			break;
	}
}

/*
 * pgstrom_gstore_fdw_validator
 */
Datum
pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS)
{
	List	   *options = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid			catalog = PG_GETARG_OID(1);

	switch (catalog)
	{
		case ForeignTableRelationId:
			__gstore_fdw_table_options(options, NULL, NULL);
			break;

		case AttributeRelationId:
			__gstore_fdw_column_options(options, NULL);
			break;

		case ForeignServerRelationId:
			if (options)
				elog(ERROR, "gstore_fdw: no options are supported on SERVER");
			break;

		case ForeignDataWrapperRelationId:
			if (options)
				elog(ERROR, "gstore_fdw: no options are supported on FOREIGN DATA WRAPPER");
			break;

		default:
			elog(ERROR, "gstore_fdw: no options are supported on catalog %s",
				 get_rel_name(catalog));
			break;
	}
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
	routine->AddForeignUpdateTargets = gstoreAddForeignUpdateTargets;
	routine->BeginForeignScan	= gstoreBeginForeignScan;
	routine->IterateForeignScan	= gstoreIterateForeignScan;
	routine->ReScanForeignScan	= gstoreReScanForeignScan;
	routine->EndForeignScan		= gstoreEndForeignScan;

	/* functions for INSERT/UPDATE/DELETE foreign tables */

	routine->PlanForeignModify	= gstorePlanForeignModify;
	routine->BeginForeignModify	= gstoreBeginForeignModify;
	routine->ExecForeignInsert	= gstoreExecForeignInsert;
	routine->ExecForeignUpdate  = gstoreExecForeignUpdate;
	routine->ExecForeignDelete	= gstoreExecForeignDelete;
	routine->EndForeignModify	= gstoreEndForeignModify;

	PG_RETURN_POINTER(routine);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_handler);

/*
 * pgstrom_reggstore_in
 */
Datum
pgstrom_reggstore_in(PG_FUNCTION_ARGS)
{
	Datum	datum = regclassin(fcinfo);

	if (!relation_is_gstore_fdw(DatumGetObjectId(datum)))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						DatumGetObjectId(datum))));
	PG_RETURN_DATUM(datum);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_in);

/*
 * pgstrom_reggstore_out
 */
Datum
pgstrom_reggstore_out(PG_FUNCTION_ARGS)
{
	Oid		relid = PG_GETARG_OID(0);

	if (!relation_is_gstore_fdw(relid))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						relid)));
	return regclassout(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_out);

/*
 * pgstrom_reggstore_recv
 */
Datum
pgstrom_reggstore_recv(PG_FUNCTION_ARGS)
{
	/* exactly the same as oidrecv, so share code */
	Datum	datum = oidrecv(fcinfo);

	if (!relation_is_gstore_fdw(DatumGetObjectId(datum)))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						DatumGetObjectId(datum))));
	PG_RETURN_DATUM(datum);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_recv);

/*
 * pgstrom_reggstore_send
 */
Datum
pgstrom_reggstore_send(PG_FUNCTION_ARGS)
{
	Oid		relid = PG_GETARG_OID(0);

	if (!relation_is_gstore_fdw(relid))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("Relation %u is not a foreign table of gstore_fdw",
						relid)));
	/* Exactly the same as oidsend, so share code */
	return oidsend(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_reggstore_send);

/*
 * get_reggstore_type_oid
 */
Oid
get_reggstore_type_oid(void)
{
	if (!OidIsValid(reggstore_type_oid))
	{
		Oid		temp_oid;

		temp_oid = GetSysCacheOid2(TYPENAMENSP,
								   CStringGetDatum("reggstore"),
								   ObjectIdGetDatum(PG_PUBLIC_NAMESPACE));
		if (!OidIsValid(temp_oid) ||
			!type_is_reggstore(temp_oid))
			elog(ERROR, "type \"reggstore\" is not defined");
		reggstore_type_oid = temp_oid;
	}
	return reggstore_type_oid;
}

/*
 * reset_reggstore_type_oid
 */
static void
reset_reggstore_type_oid(Datum arg, int cacheid, uint32 hashvalue)
{
	reggstore_type_oid = InvalidOid;
}

/*
 * pgstrom_gstore_export_ipchandle
 */
Datum
pgstrom_gstore_export_ipchandle(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	cl_int			pinning;
	GpuStoreChunk  *gs_chunk;
	char		   *result;

	if (!relation_is_gstore_fdw(gstore_oid))
		elog(ERROR, "relation %u is not gstore_fdw foreign table",
			 gstore_oid);
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	gstore_fdw_table_options(gstore_oid, &pinning, NULL);
	if (pinning < 0)
		elog(ERROR, "gstore_fdw: \"%s\" is not pinned on GPU devices",
			 get_rel_name(gstore_oid));
	if (pinning >= numDevAttrs)
		elog(ERROR, "gstore_fdw: \"%s\" is not pinned on valid GPU device",
			 get_rel_name(gstore_oid));

	gs_chunk = gstore_fdw_lookup_chunk(gstore_oid, GetActiveSnapshot());
	if (!gs_chunk)
		PG_RETURN_NULL();

	result = palloc(VARHDRSZ + sizeof(CUipcMemHandle));
	memcpy(result + VARHDRSZ, &gs_chunk->ipc_mhandle, sizeof(CUipcMemHandle));
	SET_VARSIZE(result, VARHDRSZ + sizeof(CUipcMemHandle));

	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_export_ipchandle);

/*
 * type_is_reggstore
 */
bool
type_is_reggstore(Oid type_oid)
{
	Oid			typinput;
	HeapTuple	tup;
	char	   *prosrc;
	char	   *probin;
	Datum		datum;
	bool		isnull;
	PGFunction	handler_fn;

	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typinput = ((Form_pg_type) GETSTRUCT(tup))->typinput;
	ReleaseSysCache(tup);

	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(typinput));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", typinput);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "null prosrc for C function %u", typinput);
	prosrc = TextDatumGetCString(datum);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_probin, &isnull);
	if (isnull)
		elog(ERROR, "null probin for C function %u", typinput);
	probin = TextDatumGetCString(datum);
	ReleaseSysCache(tup);

	/* check whether function pointer is identical */
	handler_fn = load_external_function(probin, prosrc, true, NULL);
	if (handler_fn != pgstrom_reggstore_in)
		return false;
	/* ok, it is reggstore type */
	return true;
}

static CUdeviceptr
gstore_open_device_memory(GpuContext *gcontext, Relation frel)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	GpuStoreBuffer *gs_buffer;
	GpuStoreChunk  *gs_chunk;
	CUdeviceptr		m_deviceptr;
	CUresult		rc;
	bool			found;
	size_t			nrooms;
	size_t			length;
	cl_int			j, ncols;
	bits8		   *rowmap;

	gs_chunk = gstore_fdw_lookup_chunk(RelationGetRelid(frel),
									   GetActiveSnapshot());
	gs_buffer = (!gstore_buffer_htab
				 ? NULL
				 : hash_search(gstore_buffer_htab,
							   &RelationGetRelid(frel),
							   HASH_FIND,
							   &found));
	if (gs_chunk)
	{
		/*
		 * If device memory is valid and up-to-date, open IpcHandle
		 * and returns this device address.
		 */
		if (!gs_buffer || (gs_buffer->revision == gs_chunk->revision &&
						   !gs_buffer->is_dirty))
		{
			if (gcontext->cuda_dindex != gs_chunk->pinning)
				elog(ERROR, "Bug? gstore_fdw: \"%s\" has wrong pinning",
					 RelationGetRelationName(frel));

			rc = cuCtxPushCurrent(gcontext->cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));
			rc = gpuIpcOpenMemHandle(gcontext,
									 &m_deviceptr,
									 gs_chunk->ipc_mhandle,
									 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on gpuIpcOpenMemHandle: %s",
					 errorText(rc));

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPopCurrent: %s", errorText(rc));

			return m_deviceptr;
		}
		/* Hmm... on device image is not up to date... */
	}
	/*
	 * corner case: we have neither device memory nor local buffer.
	 * in this case, we make an empty store.
	 */
	if (!gs_buffer)
	{
		ncols = tupdesc->natts + NumOfSystemAttrs;
		length = KDS_CALCULATE_HEAD_LENGTH(ncols, true);
		rc = gpuMemAllocManaged(gcontext,
								&m_deviceptr,
								Max(length, KDS_LEAST_LENGTH),
								CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));

		init_kernel_data_store((kern_data_store *)m_deviceptr,
							   tupdesc,
							   length,
							   KDS_FORMAT_COLUMN,
							   0,
							   true);
		return m_deviceptr;
	}

	/*
	 * local read-write buffer is up-to-date.
	 * So, we need to construct in-kernel image.
	 *
	 * Logic is almost same to gstore_fdw_alloc_pgstrom_buffer()
	 */
	rowmap = gstore_fdw_visibility_bitmap(gs_buffer, &nrooms);
	ncols = tupdesc->natts + NumOfSystemAttrs;
	length = KDS_CALCULATE_HEAD_LENGTH(ncols, true);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		if (attr->attisdropped)
			continue;
		if (attr->attlen < 0)
		{
			length += (MAXALIGN(sizeof(cl_uint) * nrooms) +
					   MAXALIGN(gs_buffer->cc_buf.extra_sz[j]));
		}
		else
		{
			int		unitsz = att_align_nominal(attr->attlen,
											   attr->attalign);
			length += MAXALIGN(unitsz * nrooms);
			if (gs_buffer->cc_buf.hasnull[j])
				length += MAXALIGN(BITMAPLEN(nrooms));
		}
	}
	rc = gpuMemAllocManaged(gcontext,
							&m_deviceptr,
							Max(length, KDS_LEAST_LENGTH),
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));

	ccache_copy_buffer_to_kds((kern_data_store *)m_deviceptr,
							  tupdesc, &gs_buffer->cc_buf,
							  rowmap, nrooms);
	return m_deviceptr;
}

/*
 * gstore_fdw_preferable_device
 */
int
gstore_fdw_preferable_device(FunctionCallInfo fcinfo)
{
	FmgrInfo   *flinfo = fcinfo->flinfo;
	HeapTuple	protup;
	oidvector  *proargtypes;
	cl_int		i, cuda_dindex = -1;

	protup = SearchSysCache1(PROCOID, ObjectIdGetDatum(flinfo->fn_oid));
	if (!HeapTupleIsValid(protup))
		elog(ERROR, "cache lookup failed function %u", flinfo->fn_oid);
	proargtypes = &((Form_pg_proc)GETSTRUCT(protup))->proargtypes;
	for (i=0; i < proargtypes->dim1; i++)
	{
		Oid		gstore_oid;
		int		pinning;

		if (proargtypes->values[i] != REGGSTOREOID)
			continue;
		gstore_oid = DatumGetObjectId(fcinfo->arg[i]);
		if (!relation_is_gstore_fdw(gstore_oid))
			elog(ERROR, "relation %u is not gstore_fdw foreign table",
				 gstore_oid);
		gstore_fdw_table_options(gstore_oid, &pinning, NULL);
		if (pinning < 0 || pinning >= numDevAttrs)
			elog(ERROR, "gstore_fdw: \"%s\" is pinned on unknown device %d",
				 get_rel_name(gstore_oid), pinning);
		if (cuda_dindex < 0)
			cuda_dindex = pinning;
		else if (cuda_dindex != pinning)
			elog(ERROR, "function %s: called with gstore_fdw foreign tables which are pinned on difference devices",
				 format_procedure(flinfo->fn_oid));
	}
	ReleaseSysCache(protup);

	return cuda_dindex;
}

/*
 * gstore_fdw_load_function_args
 */
void
gstore_fdw_load_function_args(GpuContext *gcontext,
							  FunctionCallInfo fcinfo,
							  List **p_gstore_oid_list,
							  List **p_gstore_devptr_list,
							  List **p_gstore_dindex_list)
{
	FmgrInfo   *flinfo = fcinfo->flinfo;
	HeapTuple	protup;
	oidvector  *proargtypes;
	List	   *gstore_oid_list = NIL;
	List	   *gstore_devptr_list = NIL;
	List	   *gstore_dindex_list = NIL;
	ListCell   *lc;
	int			i;

	protup = SearchSysCache1(PROCOID, ObjectIdGetDatum(flinfo->fn_oid));
	if (!HeapTupleIsValid(protup))
		elog(ERROR, "cache lookup failed function %u", flinfo->fn_oid);
	proargtypes = &((Form_pg_proc)GETSTRUCT(protup))->proargtypes;
	for (i=0; i < proargtypes->dim1; i++)
	{
		Relation	frel;
		Oid			gstore_oid;
		int			pinning;
		CUdeviceptr	m_deviceptr;

		if (proargtypes->values[i] != REGGSTOREOID)
			continue;
		gstore_oid = DatumGetObjectId(fcinfo->arg[i]);
		/* already loaded? */
		foreach (lc, gstore_oid_list)
		{
			if (gstore_oid == lfirst_oid(lc))
				break;
		}
		if (lc != NULL)
			continue;

		if (!relation_is_gstore_fdw(gstore_oid))
			elog(ERROR, "relation %u is not gstore_fdw foreign table",
				 gstore_oid);

		gstore_fdw_table_options(gstore_oid, &pinning, NULL);
		if (pinning >= 0 && gcontext->cuda_dindex != pinning)
			elog(ERROR, "unable to load gstore_fdw foreign table \"%s\" on the GPU device %d; GpuContext is assigned on the device %d",
				 get_rel_name(gstore_oid), pinning, gcontext->cuda_dindex);

		frel = heap_open(gstore_oid, AccessShareLock);
		m_deviceptr = gstore_open_device_memory(gcontext, frel);
		heap_close(frel, NoLock);

		gstore_oid_list = lappend_oid(gstore_oid_list, gstore_oid);
		gstore_devptr_list = lappend(gstore_devptr_list,
									 (void *)m_deviceptr);
		gstore_dindex_list = lappend_int(gstore_dindex_list, pinning);
	}
	ReleaseSysCache(protup);
	*p_gstore_oid_list = gstore_oid_list;
	*p_gstore_devptr_list = gstore_devptr_list;
	*p_gstore_dindex_list = gstore_dindex_list;
}

/*
 * pgstrom_gstore_fdw_format
 */
Datum
pgstrom_gstore_fdw_format(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	GpuStoreChunk  *gs_chunk;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	gs_chunk = gstore_fdw_lookup_chunk(gstore_oid, GetActiveSnapshot());
	if (!gs_chunk)
		PG_RETURN_NULL();

	/* currently, only 'pgstrom' is the supported format */
	PG_RETURN_TEXT_P(cstring_to_text("pgstrom"));
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_format);

/*
 * pgstrom_gstore_fdw_nitems
 */
Datum
pgstrom_gstore_fdw_nitems(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	GpuStoreChunk  *gs_chunk;
	int64			retval = 0;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	gs_chunk = gstore_fdw_lookup_chunk(gstore_oid, GetActiveSnapshot());
	if (gs_chunk)
		retval = gs_chunk->nitems;

	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_nitems);

/*
 * pgstrom_gstore_fdw_nattrs
 */
Datum
pgstrom_gstore_fdw_nattrs(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	Relation		frel;
	int64			retval = 0;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	frel = heap_open(gstore_oid, AccessShareLock);
	retval = RelationGetNumberOfAttributes(frel);
	heap_close(frel, NoLock);

	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_nattrs);

/*
 * pgstrom_gstore_fdw_rawsize
 */
Datum
pgstrom_gstore_fdw_rawsize(PG_FUNCTION_ARGS)
{
	Oid				gstore_oid = PG_GETARG_OID(0);
	GpuStoreChunk  *gs_chunk;
	int64			retval = 0;

	if (!relation_is_gstore_fdw(gstore_oid))
		PG_RETURN_NULL();
	strom_foreign_table_aclcheck(gstore_oid, GetUserId(), ACL_SELECT);

	gs_chunk = gstore_fdw_lookup_chunk(gstore_oid, GetActiveSnapshot());
	if (gs_chunk)
		retval = gs_chunk->rawsize;

	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_rawsize);

/*
 * pgstrom_gstore_fdw_chunk_info
 */
Datum
pgstrom_gstore_fdw_chunk_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	GpuStoreChunk  *gs_chunk;
	GpuStoreChunk  *gs_temp;
	List	   *chunks_list;
	Datum		values[9];
	bool		isnull[9];
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(9, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "database_oid",
						   OIDOID, -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber) 2, "table_oid",
						   REGCLASSOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "revision",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "xmin",
						   XIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "xmax",
						   XIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "pinning",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 7, "format",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 8, "rawsize",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 9, "nitems",
						   INT8OID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		chunks_list = NIL;
		SpinLockAcquire(&gstore_head->lock);
		PG_TRY();
		{
			dlist_iter	iter;
			int			i;

			for (i=0; i < GSTORE_CHUNK_HASH_NSLOTS; i++)
			{
				dlist_foreach(iter, &gstore_head->active_chunks[i])
				{
					gs_chunk = dlist_container(GpuStoreChunk, chain, iter.cur);
					if (!superuser())
					{
						if (gs_chunk->database_oid != MyDatabaseId)
							continue;
						if (pg_class_aclcheck(gs_chunk->table_oid,
											  GetUserId(),
											  ACL_SELECT) != ACLCHECK_OK)
							continue;
					}
					gs_temp = palloc(sizeof(GpuStoreChunk));
					memcpy(gs_temp, gs_chunk, sizeof(GpuStoreChunk));

					chunks_list = lappend(chunks_list, gs_temp);
				}
			}
		}
		PG_CATCH();
		{
			SpinLockRelease(&gstore_head->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(&gstore_head->lock);

		fncxt->user_fctx = chunks_list;
		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	chunks_list = fncxt->user_fctx;
	if (chunks_list == NIL)
		SRF_RETURN_DONE(fncxt);
	gs_chunk = linitial(chunks_list);
	Assert(gs_chunk != NULL);
	fncxt->user_fctx = list_delete_first(chunks_list);

	memset(isnull, 0, sizeof(isnull));
	values[0] = ObjectIdGetDatum(gs_chunk->database_oid);
	values[1] = ObjectIdGetDatum(gs_chunk->table_oid);
	values[2] = Int32GetDatum(gs_chunk->revision);
	values[3] = TransactionIdGetDatum(gs_chunk->xmin);
	values[4] = TransactionIdGetDatum(gs_chunk->xmax);
	values[5] = Int32GetDatum(gs_chunk->pinning);
	if (gs_chunk->format == GSTORE_FDW_FORMAT__PGSTROM)
		values[6] = CStringGetTextDatum("pgstrom");
	else
		values[6] = CStringGetTextDatum(psprintf("unknown - %u",
												 gs_chunk->format));
	values[7] = Int64GetDatum(gs_chunk->rawsize);
	values[8] = Int64GetDatum(gs_chunk->nitems);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_chunk_info);

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
										   gs_chunks[gstore_max_relations]),
								  &found);
	if (found)
		elog(ERROR, "Bug? shared memory for gstore_fdw already built");

	pg_atomic_init_u32(&gstore_head->revision_seed, 1);
	SpinLockInit(&gstore_head->lock);
	dlist_init(&gstore_head->free_chunks);
	for (i=0; i < GSTORE_CHUNK_HASH_NSLOTS; i++)
		dlist_init(&gstore_head->active_chunks[i]);
	for (i=0; i < gstore_max_relations; i++)
	{
		GpuStoreChunk  *gs_chunk = &gstore_head->gs_chunks[i];

		memset(gs_chunk, 0, sizeof(GpuStoreChunk));
		dlist_push_tail(&gstore_head->free_chunks, &gs_chunk->chain);
	}
}

/*
 * pgstrom_init_gstore_fdw
 */
void
pgstrom_init_gstore_fdw(void)
{
	DefineCustomIntVariable("pg_strom.gstore_max_relations",
							"maximum number of gstore_fdw relations",
							NULL,
							&gstore_max_relations,
							100,
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	RequestAddinShmemSpace(MAXALIGN(offsetof(GpuStoreHead,
											gs_chunks[gstore_max_relations])));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gstore_fdw;

	object_access_next = object_access_hook;
	object_access_hook = gstore_fdw_object_access;

	RegisterXactCallback(gstoreXactCallback, NULL);
	//RegisterSubXactCallback(gstoreSubXactCallback, NULL);

	/* invalidation of reggstore_oid variable */
	CacheRegisterSyscacheCallback(TYPEOID, reset_reggstore_type_oid, 0);
}
