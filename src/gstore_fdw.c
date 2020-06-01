/*
 * gstore_fdw.c
 *
 * GPU data store based on Apache Arrow and fast updatabled REDO logs.
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#include "cuda_gstore.h"
#include <libpmem.h>

#define GPUSTORE_SHARED_DESC_NSLOTS		107
typedef struct
{
	/* Database name to connect */
	char			kicker_next_database[NAMEDATALEN];
	
	/* IPC to GPU Memory Synchronizer */
	Latch		   *gpumem_sync_latch;
	dlist_head		gpumem_sync_list;
	slock_t			gpumem_sync_lock;
	ConditionVariable gpumem_sync_cond;

	/* Hash slot for GpuStoreSharedState */
	slock_t			gstore_sstate_lock[GPUSTORE_SHARED_DESC_NSLOTS];
	dlist_head		gstore_sstate_slot[GPUSTORE_SHARED_DESC_NSLOTS];
} GpuStoreSharedHead;

/*
 * GstoreFdw Base file structure
 *
 * +------------------------+
 * | GpuStoreBaseFileHead   |
 * |  * rowid_map_offset   --------+
 * |  * hash_index_offset  ------+ |
 * | +----------------------+    | |
 * | | schema definition    |    | |
 * | | (kern_data_store)    |    | |
 * | | * extra_hoffset   ------------+
 * | =                      =    | | |
 * | | fixed length portion |    | | |
 * | | (nullmap + values)   |    | | |
 * +-+----------------------+ <--+ | |
 * | GpuStoreRowIdMap       |      | |
 * |                        |      | |
 * +------------------------+ <----+ |
 * | GpuStoreHashHead       |        |
 * | (optional, if PK)      |        |
 * +------------------------+ <------+
 * | Extra Buffer           |
 * | (optional, if varlena) |
 * =                        =
 * | buffer size shall be   |
 * | expanded on demand     |
 * +------------------------+
 *
 * The base file contains four sections internally.
 * The 1st section contains schema definition in device side (thus, it
 * additionally has xmin,xmax,cid columns after the user defined columns).
 * The 2nd section contains row-id map for fast row-id allocation.
 * The 3rd section is hash-based PK index.
 * The above three sections are all fixed-length, thus, its size shall not
 * be changed unless 'max_num_rows' is not reconfigured.
 * The 4th section (extra buffer) is used to store the variable length
 * values, and can be expanded on the demand.
 */

#define GPUSTORE_BASEFILE_SIGNATURE		"@BASE-1@"
#define GPUSTORE_BASEFILE_MAPPED_SIGNATURE "%Base-1%"
#define GPUSTORE_ROWIDMAP_SIGNATURE		"@ROW-ID@"
#define GPUSTORE_HASHINDEX_SIGNATURE	"@HINDEX@"
#define GPUSTORE_REDOLOG_SIGNATURE		"@REDO-1@"

typedef struct
{
	char		signature[8];
	uint64		rowid_map_offset;
	uint64		hash_index_offset;	/* optionsl (if primary key exist)  */
	char		ftable_name[NAMEDATALEN];
	kern_data_store schema;
} GpuStoreBaseFileHead;

/*
 * RowID map - it shall locate next to the base area (schema + fixed-length
 * array of base file), and prior to the extra buffer.
 */
typedef struct
{
	char		signature[8];
	size_t		length;
	cl_uint		nrooms;
	cl_ulong	base[FLEXIBLE_ARRAY_MEMBER];
} GpuStoreRowIdMapHead;

/*
 * GpuStoreHashIndexHead - section for optional hash-based PK index
 */
typedef struct
{
	char		signature[8];
	uint64		nrooms;
	uint64		nslots;
	uint32		slots[FLEXIBLE_ARRAY_MEMBER];
} GpuStoreHashIndexHead;

/*
 * GpuStoreRedoLogHead - file header of REDO-Log
 */
typedef struct
{
	char		signature[8];
	uint64		timestamp;	/* timestamp when REDO-Log is switched */
	uint64		checkpoint_offset;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} GpuStoreRedoLogHead;

#define MMAP_SYNC_METHOD__NONE		0
#define MMAP_SYNC_METHOD__MSYNC		1
#define MMAP_SYNC_METHOD__PMEM		2

typedef struct
{
	dlist_node		hash_chain;
	dlist_node		sync_chain;		/* link to gpumem_sync_list */
	Oid				database_oid;
	Oid				ftable_oid;

	/* FDW options */
	cl_int			cuda_dindex;
	ssize_t			max_num_rows;
	ssize_t			num_hash_slots;
	AttrNumber		primary_key;
	const char	   *base_file;
	const char	   *redo_log_file;
	size_t			redo_log_limit;
	int				redo_log_sync_method;	/* one of REDO_LOG_SYNC__* */
	const char	   *redo_log_backup_dir;
	cl_int			gpu_update_interval;
	cl_int			gpu_update_threshold;

	/* Runtime state */
	pthread_rwlock_t mmap_lock;
	uint32			mmap_revision;
	slock_t			rowid_map_lock;		/* lock for RowID-map */
	slock_t		   *hash_slots_lock;	/* locks for PK hash-slots */

	pg_atomic_uint64 redo_log_pos;

	/* Device data store */
	pthread_rwlock_t gpu_bufer_lock;
	CUresult		gpu_sync_status;
	CUipcMemHandle	gpu_main_mhandle;		/* mhandle to main portion */
	CUipcMemHandle	gpu_extra_mhandle;		/* mhandle to extra portion */
	size_t			gpu_main_size;
	size_t			gpu_extra_size;
	Timestamp		gpu_update_timestamp;	/* time when last update */
} GpuStoreSharedState;

typedef struct
{
	Oid				database_oid;
	Oid				ftable_oid;
	GpuStoreSharedState *gs_sstate;
	/* base file mapping */
	uint32			mmap_revision;
	GpuStoreBaseFileHead *base_mmap;
	size_t			base_mmap_sz;
	bool			base_mmap_is_pmem;
	GpuStoreRowIdMapHead *rowid_map;	/* RowID map section */
	GpuStoreHashIndexHead *hash_index;	/* Hash-index section (optional) */
	/* redo log mapping */
	GpuStoreRedoLogHead *redo_mmap;
	size_t			redo_mmap_sz;
	bool			redo_mmap_is_pmem;
	/* fields below are valid only GpuUpdator */
	CUdeviceptr		gpu_main_devptr;
	CUdeviceptr		gpu_extra_devptr;
} GpuStoreDesc;

/*
 * GpuStoreFdwState - runtime state object
 */
typedef struct
{
	GpuStoreDesc   *gs_desc;

	pg_atomic_uint64  __read_pos;
	pg_atomic_uint64 *read_pos;

	Bitmapset	   *attr_refs;
	cl_uint			nitems;			/* kds->nitems on BeginScan */
	cl_bool			is_first;
	cl_uint			last_rowid;		/* last rowid returned */
	ExprState	   *indexExprState;
	Datum			indexExprHash;
} GpuStoreFdwState;

/*
 * NOTE: Gstore_Fdw adds a 64-bit system column to save the timestamp on
 * INSERT/UPDATE/DELETE. Its upper 62bit hold nano-second scale timestamp
 * from the PostgreSQL epoch. Other bits are used to lock and removed flag.
 */
#define GSTORE_SYSATTR__LOCKED		0x0001UL
#define GSTORE_SYSATTR__REMOVED		0x0002UL
#define GSTORE_SYSATTR__MASK		0x0003UL

#define REDO_LOG_TYPE__INSERT		1
#define REDO_LOG_TYPE__UPDATE		2
#define REDO_LOG_TYPE__DELETE		3

STATIC_INLINE(cl_uint)
gstore_get_tupitem_logkind(kern_tupitem *tupitem)
{
	return (cl_uint)tupitem->t_self.ip_posid;
}

STATIC_INLINE(cl_uint)
gstore_get_tupitem_rowid(kern_tupitem *tupitem)
{
	return (((cl_uint)tupitem->t_self.ip_blkid.bi_hi << 16) |
			((cl_uint)tupitem->t_self.ip_blkid.bi_lo));
}

STATIC_INLINE(size_t)
gstore_get_tupitem_vlpos(kern_tupitem *tupitem)
{
	cl_uint	vlpos = (((cl_uint)tupitem->htup.t_ctid.ip_blkid.bi_hi << 16) |
					 ((cl_uint)tupitem->htup.t_ctid.ip_blkid.bi_lo));
	return __kds_unpack(vlpos);
}




/* ---- static variables ---- */
static FdwRoutine	pgstrom_gstore_fdw_routine;
static GpuStoreSharedHead  *gstore_shared_head = NULL;
static HTAB *gstore_desc_htab = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static bool			pgstrom_gstore_fdw_auto_preload;	/* GUC */

/* ---- Forward declarations ---- */
Datum pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_synchronize(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_compaction(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_post_creation(PG_FUNCTION_ARGS);
void  GstoreFdwStartupKicker(Datum arg);

static GpuStoreDesc *gstoreFdwLookupGpuStoreDesc(Relation frel);
static size_t	gstoreFdwRowIdMapSize(cl_uint nrooms);
static cl_uint	gstoreFdwAllocateRowId(GpuStoreDesc *gs_desc,
									   cl_uint min_rowid);
static void		gstoreFdwReleaseRowId(GpuStoreDesc *gs_desc, cl_uint rowid);

/* ---- Atomic operations ---- */
static inline cl_uint atomicRead32(volatile cl_uint *ptr)
{
	return *ptr;
}








static void
GstoreGetForeignRelSize(PlannerInfo *root,
						RelOptInfo *baserel,
						Oid foreigntableid)
{
	Relation	frel;
	GpuStoreDesc *gs_desc;

	frel = heap_open(foreigntableid, AccessShareLock);
	gs_desc = gstoreFdwLookupGpuStoreDesc(frel);
	heap_close(frel, AccessShareLock);

	baserel->tuples = (double) gs_desc->base_mmap->schema.nitems;
	baserel->rows = baserel->tuples *
		clauselist_selectivity(root,
							   baserel->baserestrictinfo,
							   0,
							   JOIN_INNER,
							   NULL);
	baserel->fdw_private = gs_desc;
}

static Node *
match_clause_to_primary_key(PlannerInfo *root,
							RelOptInfo *baserel,
							RestrictInfo *rinfo,
							int primary_key)
{
	OpExpr	   *op = (OpExpr *)rinfo->clause;
	Node	   *left;
	Node	   *right;

	if (!IsA(op, OpExpr) || list_length(op->args) != 2)
		return false;
	left = (Node *) linitial(op->args);
	if (IsA(left, RelabelType))
		left = (Node *)((RelabelType *)left)->arg;
	right = (Node *) lsecond(op->args);
	if (IsA(right, RelabelType))
		right = (Node *)((RelabelType *)right)->arg;

	if (IsA(left, Var))
	{
		Var	   *var = (Var *)left;

		if (var->varno == baserel->relid &&
			var->varattno == primary_key &&
			!bms_is_member(baserel->relid, rinfo->right_relids) &&
			!contain_volatile_functions(right))
		{
			/* Ok, Left-VAR = Right-Expression */
			return right;
		}
	}

	if (IsA(right, Var))
	{
		Var	   *var = (Var *)right;

		if (var->varno == baserel->relid &&
			var->varattno == primary_key &&
			!bms_is_member(baserel->relid, rinfo->left_relids) &&
			!contain_volatile_functions(left))
		{
			/* Ok, Right-Var = Left-Expression */
			return left;
		}
	}
	return NULL;
}

static void
GstoreGetForeignPaths(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Oid foreigntableid)
{
	GpuStoreDesc   *gs_desc = baserel->fdw_private;
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	AttrNumber		primary_key = gs_sstate->primary_key;
	Relids			required_outer = baserel->lateral_relids;
	ParamPathInfo  *param_info;
	ForeignPath	   *fpath;
	ListCell	   *lc;
	Node		   *indexExpr = NULL;
	QualCost		qual_cost;
	Cost			startup_cost = 0.0;
	Cost			total_cost = 0.0;
	double			ntuples = baserel->tuples;

	if (primary_key > 0)
	{
		foreach (lc, baserel->baserestrictinfo)
		{
			RestrictInfo   *rinfo = lfirst(lc);

			indexExpr = match_clause_to_primary_key(root,
													baserel,
													rinfo,
													primary_key);
			if (indexExpr)
				break;
		}
	}
	/* simple cost estimation */
	param_info = get_baserel_parampathinfo(root, baserel, required_outer);
	if (param_info)
		cost_qual_eval(&qual_cost, param_info->ppi_clauses, root);
	else
		qual_cost = baserel->baserestrictcost;
	if (indexExpr)
		ntuples = 1.0;
	startup_cost += qual_cost.startup;
	startup_cost += baserel->reltarget->cost.startup;
	total_cost += (cpu_tuple_cost + qual_cost.per_tuple) * ntuples;
	total_cost += baserel->reltarget->cost.per_tuple * baserel->rows;

	fpath = create_foreignscan_path(root,
									baserel,
									NULL,	/* default pathtarget */
									baserel->rows,
									startup_cost,
									total_cost,
									NIL,	/* no pathkeys */
									required_outer,
									NULL,	/* no extra plan */
									list_make1(indexExpr));
	add_path(baserel, (Path *)fpath);
	
	//TODO: parameterized paths
}

static ForeignScan *
GstoreGetForeignPlan(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid foreigntableid,
					 ForeignPath *best_path,
					 List *tlist,
					 List *scan_clauses,
					 Plan *outer_plan)
{
	List   *attr_refs = NIL;
	int		i, j;

	for (i=baserel->min_attr, j=0; i <= baserel->max_attr; i++, j++)
	{
		if (!bms_is_empty(baserel->attr_needed[j]))
			attr_refs = lappend_int(attr_refs, i);
	}
	scan_clauses = extract_actual_clauses(scan_clauses, false);

	return make_foreignscan(tlist,
							scan_clauses,
							baserel->relid,
							best_path->fdw_private,	/* indexExpr */
							attr_refs,	/* referenced attnums */
							NIL,		/* no custom tlist */
							NIL,		/* no remote quals */
							outer_plan);
}

static void
GstoreBeginForeignScan(ForeignScanState *node, int eflags)
{
	Relation			frel = node->ss.ss_currentRelation;
	ForeignScan		   *fscan = (ForeignScan *)node->ss.ps.plan;
	GpuStoreFdwState   *fdw_state = palloc0(sizeof(GpuStoreFdwState));
	GpuStoreDesc	   *gs_desc = gstoreFdwLookupGpuStoreDesc(frel);
	Bitmapset		   *attr_refs = NULL;
	ListCell		   *lc;

	/* setup GpuStoreExecState */
	fdw_state->gs_desc = gs_desc;
	pg_atomic_init_u64(&fdw_state->__read_pos, 0);
	fdw_state->read_pos = &fdw_state->__read_pos;
	fdw_state->nitems = gs_desc->base_mmap->schema.nitems;
	fdw_state->last_rowid = UINT_MAX;
	fdw_state->is_first = true;

	foreach (lc, fscan->fdw_private)
	{
		int		k = lfirst_int(lc) - FirstLowInvalidHeapAttributeNumber;

		attr_refs = bms_add_member(attr_refs, k);
	}
	fdw_state->attr_refs = attr_refs;

	if (fscan->fdw_exprs != NIL)
	{
		Expr   *indexExpr = linitial(fscan->fdw_exprs);

		fdw_state->indexExprState = ExecInitExpr(indexExpr, &node->ss.ps);
	}
}

static uint64
lookupGpuStorePrimaryKey(GpuStoreDesc *gs_desc,
						 Datum key_value, cl_uint last_rowid)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	GpuStoreHashIndexHead *hash_index = gs_desc->hash_index;
	AttrNumber		primary_key = gs_sstate->primary_key;
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	kern_colmeta	*cmeta = &kds->colmeta[primary_key - 1];
	TypeCacheEntry *tcache;
	Datum			hash;
	size_t			hindex;
	uint32			rowid;
	uint32		   *rowmap;

	tcache = lookup_type_cache(cmeta->atttypid,
							   TYPECACHE_HASH_PROC_FINFO |
							   TYPECACHE_EQ_OPR_FINFO);
	hash = FunctionCall1(&tcache->hash_proc_finfo, key_value);
	hindex = hash % hash_index->nslots;
	rowmap = (uint32 *)&hash_index->slots[hash_index->nslots];

	SpinLockAcquire(&gs_sstate->hash_slots_lock[hindex]);
	if (last_rowid == UINT_MAX)
		rowid = hash_index->slots[hindex];
	else if (last_rowid < hash_index->nrooms)
		rowid = rowmap[last_rowid];
	else
		rowid = UINT_MAX;

	while (rowid != UINT_MAX)
	{
		Datum	comp;
		Datum	datum;
		bool	isnull;

		if (rowid >= hash_index->nrooms)
		{
			/* index corruption? */
			rowid = UINT_MAX;
			break;
		}
		datum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
		if (!isnull)
		{
			comp = FunctionCall2(&tcache->eq_opr_finfo, key_value, datum);
			if (DatumGetBool(comp))
				break;
		}
		rowid = rowmap[rowid];
	} while (rowid != UINT_MAX);
	SpinLockRelease(&gs_sstate->hash_slots_lock[hindex]);

	return rowid;
}

static bool
KDS_column_check_visibility(kern_data_store *kds, size_t row_index,
							Snapshot snapshot)
{
	/*
	 * NOTE: Right now, Gstore_Fdw only supports Read-Committed
	 *       isolation level.
	 */
	kern_colmeta   *cmeta = &kds->colmeta[kds->ncols-3];
	TransactionId	xmin;
	TransactionId	xmax;
	CommandId		cid;
	bool			isnull;

	Assert(kds->format == KDS_FORMAT_COLUMN &&
		   kds->ncols >= 3);
	xmin = KDS_fetch_datum_column(kds, cmeta++, row_index, &isnull);
	Assert(!isnull);
	xmax = KDS_fetch_datum_column(kds, cmeta++, row_index, &isnull);
	Assert(!isnull);
	cid  = KDS_fetch_datum_column(kds, cmeta++, row_index, &isnull);
	Assert(!isnull);

	if (xmin == InvalidTransactionId)
		return false;
	if (xmin != FrozenTransactionId)
	{
		if (TransactionIdIsCurrentTransactionId(xmin))
		{
			if (cid >= snapshot->curcid)
				return false;	/* inserted after scan started */
			if (xmax == InvalidTransactionId)
				return true;	/* not deleted yet */
			if (xmax == FrozenTransactionId)
				return false;	/* deleted, and committed */
			if (!TransactionIdIsCurrentTransactionId(xmax))
				return true;	/* deleted by others, but not committed */
		}
		else
			return false;		/* inserted by others, but not committed */
	}
	/* by here, the inserting transaction has committed */
	if (xmax == InvalidTransactionId)
		return true;			/* not deleted yet */
	if (xmax != FrozenTransactionId)
	{
		if (TransactionIdIsCurrentTransactionId(xmax))
		{
			if (cid >= snapshot->curcid)
				return true;	/* deleted after scan started */
			else
				return false;	/* deleted before scan started */
		}
		return true;	/* deleted by others, but not committed */
	}
	/* deleted, and transaction committed */
	return false;
}

static TupleTableSlot *
GstoreIterateForeignScan(ForeignScanState *node)
{
	GpuStoreFdwState *fdw_state = node->fdw_state;
	EState		   *estate = node->ss.ps.state;
	GpuStoreDesc   *gs_desc = fdw_state->gs_desc;
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	Datum			keyval;
	bool			isnull;
	uint64			rowid;

	if (fdw_state->indexExprState)
	{
		/* PK Index Scan */
		rowid = pg_atomic_fetch_add_u64(fdw_state->read_pos, 1);
		if (rowid > 0)
			return NULL;	/* PK already fetched */

		keyval = ExecEvalExpr(fdw_state->indexExprState,
							  node->ss.ps.ps_ExprContext,
							  &isnull);
		if (isnull)
			return NULL;

		rowid = UINT_MAX;
		for (;;)
		{
			rowid = lookupGpuStorePrimaryKey(gs_desc, keyval, rowid);
			if (rowid == UINT_MAX)
				return NULL;	/* not found */
			if (KDS_column_check_visibility(kds, rowid, estate->es_snapshot))
				break;
		}
	}
	else
	{
		/* SeqScan */
		for (;;)
		{
			rowid = pg_atomic_fetch_add_u64(fdw_state->read_pos, 1);
			if (rowid >= fdw_state->nitems)
				return NULL;
			if (KDS_column_check_visibility(kds, rowid, estate->es_snapshot))
				break;
		}
	}
	if (KDS_fetch_tuple_column(slot, kds, rowid))
		return slot;
	return NULL;
}

static void
GstoreReScanForeignScan(ForeignScanState *node)
{
	GpuStoreFdwState  *fdw_state = node->fdw_state;

	pg_atomic_write_u64(fdw_state->read_pos, 0);
}

static void
GstoreEndForeignScan(ForeignScanState *node)
{
	/* nothing to do */
}

static bool
GstoreIsForeignScanParallelSafe(PlannerInfo *root,
								RelOptInfo *rel,
								RangeTblEntry *rte)
{
	return false;
}

static Size
GstoreEstimateDSMForeignScan(ForeignScanState *node,
							 ParallelContext *pcxt)
{
	return 0;
}

static void
GstoreInitializeDSMForeignScan(ForeignScanState *node,
							   ParallelContext *pcxt,
							   void *coordinate)
{}

static void
GstoreReInitializeDSMForeignScan(ForeignScanState *node,
								 ParallelContext *pcxt,
								 void *coordinate)
{}

static void
GstoreInitializeWorkerForeignScan(ForeignScanState *node,
								  shm_toc *toc,
								  void *coordinate)
{}

static void
GstoreShutdownForeignScan(ForeignScanState *node)
{}

static void
GstoreAddForeignUpdateTargets(Query *parsetree,
							  RangeTblEntry *target_rte,
							  Relation target_relation)
{}

static List *
GstorePlanForeignModify(PlannerInfo *root,
						ModifyTable *plan,
						Index resultRelation,
						int subplan_index)
{
	return NIL;
}

static void
GstoreBeginForeignModify(ModifyTableState *mtstate,
						 ResultRelInfo *rinfo,
						 List *fdw_private,
						 int subplan_index,
						 int eflags)
{}

static TupleTableSlot *
GstoreExecForeignInsert(EState *estate,
						ResultRelInfo *rinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	return NULL;
}

static TupleTableSlot *
GstoreExecForeignUpdate(EState *estate,
						ResultRelInfo *rinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	return NULL;
}

static TupleTableSlot *
GstoreExecForeignDelete(EState *estate,
						ResultRelInfo *rinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	return NULL;
}

static void
GstoreEndForeignModify(EState *estate, ResultRelInfo *rinfo)
{}

static void
GstoreExplainForeignScan(ForeignScanState *node, ExplainState *es)
{}

static void
GstoreExplainForeignModify(ModifyTableState *mtstate,
						   ResultRelInfo *rinfo,
						   List *fdw_private,
						   int subplan_index,
						   ExplainState *es)
{}

static bool
GstoreAnalyzeForeignTable(Relation relation,
						  AcquireSampleRowsFunc *func,
						  BlockNumber *totalpages)
{
	return false;
}

Datum
pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS)
{
	List	   *options = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid			catalog = PG_GETARG_OID(1);
	ListCell   *lc;

	if (catalog != ForeignTableRelationId)
		elog(ERROR, "unknown FDW options");
	/*
	 * Only syntax shall be checked here, then OAT_POST_CREATE hook
	 * actually validates the configuration.
	 */
	foreach (lc, options)
	{
		DefElem	   *def = (DefElem *) lfirst(lc);
		char	   *endp;

		if (catalog != ForeignTableRelationId)
		{
			const char *suffix;
			
			if (catalog == AttributeRelationId)
				suffix = " at columns";
			if (catalog == ForeignServerRelationId)
				suffix = " at foreign-servers";
			if (catalog == ForeignDataWrapperRelationId)
				suffix = " at foreign-data-wrappers";
			elog(ERROR, "No FDW options are supported%s.", suffix);
		}
		else if (strcmp(def->defname, "gpu_device_id") == 0)
		{
			char   *token = defGetString(def);

			if (strtol(token, &endp, 10) < 0 || *endp != '\0')
				elog(ERROR, "gpu_device_id = '%s' is not valid", token);
		}
		else if (strcmp(def->defname, "max_num_rows") == 0)
		{
			char   *token = defGetString(def);

			if (strtol(token, &endp, 10) < 0 || *endp != '\0')
				elog(ERROR, "max_num_rows = '%s' is not valid", token);
		}
		else if (strcmp(def->defname, "base_file") == 0 ||
				 strcmp(def->defname, "redo_log_file") == 0 ||
				 strcmp(def->defname, "redo_log_backup_dir") == 0)
		{
			/* file pathname shall be checked at post-creation steps */
		}
		else if (strcmp(def->defname, "redo_log_limit") == 0)
		{
			char   *token = defGetString(def);

			if (strtol(token, &endp, 10) <= 0 ||
				(strcasecmp(endp, "k")  != 0 &&
				 strcasecmp(endp, "kb") != 0 &&
				 strcasecmp(endp, "m")  != 0 &&
				 strcasecmp(endp, "mb") != 0 &&
				 strcasecmp(endp, "g")  != 0 &&
				 strcasecmp(endp, "gb") != 0))
				elog(ERROR, "'%s' is not a valid configuration for '%s'",
					 token, def->defname);
		}
		else if (strcmp(def->defname, "gpu_update_interval") == 0)
		{
			char   *token = defGetString(def);
			long	interval = strtol(token, &endp, 10);

			if (interval <= 0 || interval > INT_MAX || *endp != '\0')
				elog(ERROR, "'%s' is not a valid configuration for '%s'",
					 token, def->defname);
		}
		else if (strcmp(def->defname, "gpu_update_threshold") == 0)
		{
			char   *token = defGetString(def);
			long	threshold = strtol(token, &endp, 10);

			if (threshold <= 0 || threshold > INT_MAX || *endp != '\0')
				elog(ERROR, "'%s' is not a valid configuration for '%s'",
					 token, def->defname);
		}
		else if (strcmp(def->defname, "primary_key") == 0)
		{
			/* column name shall be validated later */
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
					 errmsg("invalid option \"%s\"", def->defname)));
		}
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_validator);

static void
gstoreFdwExtractOptions(Relation frel,
						cl_int *p_cuda_dindex,
						ssize_t *p_max_num_rows,
						const char **p_base_file,
						const char **p_redo_log_file,
						const char **p_redo_log_backup_dir,
						size_t *p_redo_log_limit,
						cl_int *p_gpu_update_interval,
						cl_int *p_gpu_update_threshold,
						AttrNumber *p_primary_key)
{
	ForeignTable *ft = GetForeignTable(RelationGetRelid(frel));
	TupleDesc	tupdesc = RelationGetDescr(frel);
	ListCell   *lc;
	cl_int		cuda_dindex = 0;
	ssize_t		max_num_rows = -1;
	const char *base_file = NULL;
	const char *redo_log_file = NULL;
	const char *redo_log_backup_dir = NULL;
	size_t		redo_log_limit = (512U << 20);	/* default: 512MB */
	cl_int		gpu_update_interval = 15;		/* default: 15s */
	cl_int		gpu_update_threshold = 40000;	/* default: 40k rows */
	AttrNumber	primary_key = -1;
	
	/*
	 * check foreign relation's option
	 */
	foreach (lc, ft->options)
	{
		DefElem	   *def = lfirst(lc);
		char	   *endp;

		if (strcmp(def->defname, "gpu_device_id") == 0)
		{
			long	device_id = strtol(defGetString(def), &endp, 10);
			int		i, cuda_dindex = -1;

			if (device_id < 0 || device_id > INT_MAX || *endp != '\0')
				elog(ERROR, "unexpected input for gpu_device_id: %s",
					 defGetString(def));
			for (i=0; i < numDevAttrs; i++)
			{
				if (devAttrs[i].DEV_ID == device_id)
				{
					cuda_dindex = i;
					break;
				}
			}
			if (cuda_dindex < 0)
				elog(ERROR, "gpu_device_id = %ld was not found", device_id);
		}
		else if (strcmp(def->defname, "max_num_rows") == 0)
		{
			max_num_rows = strtol(defGetString(def), &endp, 10);
			if (max_num_rows < 0 || max_num_rows >= UINT_MAX || *endp != '\0')
				elog(ERROR, "unexpected input for max_num_rows: %s",
					 defGetString(def));
		}
		else if (strcmp(def->defname, "base_file") == 0)
		{
			char   *path = defGetString(def);
			char   *d_name;

			if (access(path, R_OK | W_OK) != 0)
			{
				if (errno != ENOENT)
					elog(ERROR, "base_file '%s' is not accesible: %m", path);

				d_name = dirname(pstrdup(path));
				if (access(d_name, R_OK | W_OK | X_OK) != 0)
					elog(ERROR, "unable to create base_file '%s': %m", path);
			}
			base_file = path;
		}
		else if (strcmp(def->defname, "redo_log_file") == 0)
		{
			char   *path = defGetString(def);
			char   *d_name;

			if (access(path, R_OK | W_OK) != 0)
			{
				if (errno != ENOENT)
					elog(ERROR, "redo_log_file '%s' is not accesible: %m",
						 path);
				d_name = dirname(pstrdup(path));
				if (access(d_name, R_OK | W_OK | X_OK) != 0)
					elog(ERROR, "unable tp create redo_log_file '%s': %m",
						 path);
			}
			redo_log_file = path;
		}
		else if (strcmp(def->defname, "redo_log_backup_dir") == 0)
		{
			redo_log_backup_dir = defGetString(def);

			if (access(redo_log_backup_dir, R_OK | W_OK | X_OK) != 0)
			{
				elog(ERROR, "redo_log_backup_dir '%s' is not accessible: %m",
					 redo_log_backup_dir);
			}
		}
		else if (strcmp(def->defname, "redo_log_limit") == 0)
		{
			char   *value = defGetString(def);
			ssize_t	limit = strtol(value, &endp, 10);

			if (limit <= 0)
				elog(ERROR, "invalid redo_log_limit: %s", value);
			if (strcmp(endp, "k") == 0 || strcmp(endp, "kb") == 0)
				limit *= (1UL << 10);
			else if (strcmp(endp, "m") == 0 || strcmp(endp, "mb") == 0)
				limit *= (1UL << 20);
			else if (strcmp(endp, "g") == 0 || strcmp(endp, "gb") == 0)
				limit *= (1UL << 30);
			else if (*endp != '\0')
				elog(ERROR, "invalid redo_log_limit: %s", value);
			if (limit % PAGE_SIZE != 0)
				elog(ERROR, "redo_log_limit must be multiple of PAGE_SIZE");
			if (limit < (128UL << 20))
				elog(ERROR, "redo_log_limit must be larger than 128MB");
			redo_log_limit = limit;
		}
		else if (strcmp(def->defname, "gpu_update_interval") == 0)
		{
			char   *value = defGetString(def);
			long	interval = strtol(value, &endp, 10);

			if (interval <= 0 || interval > INT_MAX || *endp != '\0')
				elog(ERROR, "invalid gpu_update_interval: %s", value);
			gpu_update_interval = interval;
		}
		else if (strcmp(def->defname, "gpu_update_threshold") == 0)
		{
			char   *value = defGetString(def);
			long	threshold = strtol(value, &endp, 10);

			if (threshold <= 0 || threshold > INT_MAX || *endp != '\0')
				elog(ERROR, "invalid gpu_update_threshold: %s", value);
			gpu_update_threshold = threshold;
		}
		else if (strcmp(def->defname, "primary_key") == 0)
		{
			char   *pk_name = defGetString(def);
			int		j;

			for (j=0; j < tupdesc->natts; j++)
			{
				Form_pg_attribute attr = tupleDescAttr(tupdesc,j);

				if (strcmp(pk_name, NameStr(attr->attname)) == 0)
				{
					primary_key = attr->attnum;
					break;
				}
			}
			if (primary_key < 0)
				elog(ERROR, "'%s' specified by 'primary_key' option not found",
					 pk_name);
		}
		else
			elog(ERROR, "gstore_fdw: unknown option: %s", def->defname);
	}

	/*
	 * Check Mandatory Options
	 */
	if (max_num_rows < 0)
		elog(ERROR, "max_num_rows must be specified");
	if (!redo_log_file)
		elog(ERROR, "redo_log_file must be specified");

	/*
	 * Write-back Results
	 */
	*p_cuda_dindex          = cuda_dindex;
	*p_max_num_rows         = max_num_rows;
	*p_base_file            = base_file;
	*p_redo_log_file        = redo_log_file;
	*p_redo_log_backup_dir  = redo_log_backup_dir;
	*p_redo_log_limit       = redo_log_limit;
	*p_gpu_update_interval  = gpu_update_interval;
	*p_gpu_update_threshold = gpu_update_threshold;
	*p_primary_key          = primary_key;
}

/*
 * gstoreFdwDeviceTupleDesc
 *
 * GstoreFdw appends a virtual system column at the tail of user-defined
 * columns to control visibility of rows. This function generates a pseudo
 * TupleDesc that contains the system column. It shall be released by pfree().
 */
static TupleDesc
gstoreFdwDeviceTupleDesc(Relation frel)
{
	TupleDesc	tupdesc = RelationGetDescr(frel);
	TupleDesc	__tupdesc = CreateTemplateTupleDesc(tupdesc->natts + 3);
	Form_pg_attribute attr;

	for (int j=0; j < tupdesc->natts; j++)
	{
		memcpy(tupleDescAttr(__tupdesc, j),
			   tupleDescAttr(tupdesc, j),
			   ATTRIBUTE_FIXED_PART_SIZE);
	}
	TupleDescInitEntry(__tupdesc,
					   tupdesc->natts + 1,
					   "..sysattr.xmin..",
					   XIDOID, -1, 0);
	attr = tupleDescAttr(__tupdesc, tupdesc->natts);
	attr->attnotnull = true;

	TupleDescInitEntry(__tupdesc,
					   tupdesc->natts + 2,
					   "..sysattr.xmax..",
					   XIDOID, -1, 0);
	attr = tupleDescAttr(__tupdesc, tupdesc->natts + 1);
	attr->attnotnull = true;

	TupleDescInitEntry(__tupdesc,
					   tupdesc->natts + 3,
					   "..sysattr.cid..",
					   CIDOID, -1, 0);
	attr = tupleDescAttr(__tupdesc, tupdesc->natts + 2);
	attr->attnotnull = true;

	return __tupdesc;
}

/*
 * gstoreFdwAllocSharedState
 */
static GpuStoreSharedState *
gstoreFdwAllocSharedState(Relation frel)
{
	GpuStoreSharedState *gs_sstate;
	cl_int		cuda_dindex;
	ssize_t		max_num_rows;
	ssize_t		num_hash_slots = -1;
	const char *base_file;
	const char *redo_log_file;
	const char *redo_log_backup_dir;
	size_t		redo_log_limit;
	cl_int		gpu_update_interval;
	cl_int		gpu_update_threshold;
	AttrNumber	primary_key;
	size_t		len;
	char	   *pos;

	/* extract table/column options */
	gstoreFdwExtractOptions(frel,
							&cuda_dindex,
							&max_num_rows,
							&base_file,
							&redo_log_file,
							&redo_log_backup_dir,
							&redo_log_limit,
							&gpu_update_interval,
							&gpu_update_threshold,
							&primary_key);
	/* allocation of GpuStoreSharedState */
	len = MAXALIGN(sizeof(GpuStoreSharedState));
	if (base_file)
		len += MAXALIGN(strlen(base_file) + 1);
	if (redo_log_file)
		len += MAXALIGN(strlen(redo_log_file) + 1);
	if (redo_log_backup_dir)
		len += MAXALIGN(strlen(redo_log_backup_dir) + 1);
	if (primary_key >= 0)
	{
		num_hash_slots = 1.2 * (double)max_num_rows + 1000.0;
		len += MAXALIGN(sizeof(slock_t) * num_hash_slots);
	}

	gs_sstate = MemoryContextAllocZero(TopSharedMemoryContext, len);
	pos = (char *)gs_sstate + MAXALIGN(sizeof(GpuStoreSharedState));
	if (base_file)
	{
		strcpy(pos, base_file);
		gs_sstate->base_file = pos;
		pos += MAXALIGN(strlen(base_file) + 1);
	}
	if (redo_log_file)
	{
		strcpy(pos, redo_log_file);
		gs_sstate->redo_log_file = pos;
		pos += MAXALIGN(strlen(redo_log_file) + 1);
	}
	if (redo_log_backup_dir)
	{
		strcpy(pos, redo_log_backup_dir);
		gs_sstate->redo_log_backup_dir = pos;
		pos += MAXALIGN(strlen(redo_log_backup_dir) + 1);
	}
	if (num_hash_slots > 0)
	{
		gs_sstate->hash_slots_lock = (slock_t *)pos;
		pos += MAXALIGN(sizeof(slock_t) * num_hash_slots);
	}
	Assert(pos - (char *)gs_sstate == len);

	gs_sstate->cuda_dindex = cuda_dindex;
	gs_sstate->max_num_rows = max_num_rows;
	gs_sstate->num_hash_slots = num_hash_slots;
	gs_sstate->primary_key = primary_key;
	gs_sstate->redo_log_limit = redo_log_limit;
	gs_sstate->gpu_update_interval = gpu_update_interval;
	gs_sstate->gpu_update_threshold = gpu_update_threshold;

	pthreadRWLockInit(&gs_sstate->mmap_lock);
	gs_sstate->mmap_revision = random();

	pthreadRWLockInit(&gs_sstate->gpu_bufer_lock);
	gs_sstate->gpu_sync_status = CUDA_ERROR_NOT_INITIALIZED;

	return gs_sstate;
}



/*
 * Routines to Hash-based Primary-Key Index
 */







/*
 * Routines to allocate/release RowIDs
 */
static size_t
gstoreFdwRowIdMapSize(cl_uint nrooms)
{
	size_t		n;

	if (nrooms <= (1U << 8))		/* single level */
		n = 4;
	else if (nrooms <= (1U << 16))	/* double level */
		n = 4 + TYPEALIGN(256, nrooms) / 64;
	else if (nrooms <= (1U << 24))	/* triple level */
		n = 4 + (4 << 8) + TYPEALIGN(256, nrooms) / 64;
	else
		n = 4 + (4 << 8) + (4 << 16) + TYPEALIGN(256, nrooms) / 64;

	return offsetof(GpuStoreRowIdMapHead, base[n]);
}

static cl_uint
__gstoreFdwAllocateRowId(cl_ulong *base, cl_uint nrooms,
						 cl_uint min_id, int depth, cl_uint offset,
						 cl_bool *p_has_unused_rowids)
{
	cl_ulong   *next = NULL;
	cl_ulong	mask = (1UL << ((min_id >> 24) & 0x3fU)) - 1;
	int			k, start = (min_id >> 30);
	cl_uint		rowid = UINT_MAX;

	if ((offset << 8) >= nrooms)
	{
		*p_has_unused_rowids = false;
		return UINT_MAX;	/* obviously, out of range */
	}

	switch (depth)
	{
		case 0:
			if (nrooms > 256)
				next = base + 4;
			break;
		case 1:
			if (nrooms > 65536)
				next = base + 4 + (4 << 8);
			break;
		case 2:
			if (nrooms > 16777216)
				next = base + 4 + (4 << 8) + (4 << 16);
			break;
		case 3:
			next = NULL;
			break;
		default:
			return UINT_MAX;	/* Bug? */
	}
	base += (4 * offset);
	for (k=start; k < 4; k++, mask=0)
	{
		cl_ulong	map = base[k] | mask;
		cl_ulong	bit;
	retry:
		if (map != ~0UL)
		{
			/* lookup the first zero position */
			rowid = (__builtin_ffsl(~map) - 1);
			bit = (1UL << rowid);
			rowid |= (offset << 8) | (k << 6);	/* add offset */

			if (!next)
			{
				if (rowid < nrooms)
					base[k] |= bit;
				else
					rowid = UINT_MAX;	/* not a valid RowID */
			}
			else
			{
				cl_bool		has_unused_rowids;
			
				rowid = __gstoreFdwAllocateRowId(next, nrooms,
												 min_id << 8,
												 depth+1, rowid,
												 &has_unused_rowids);
				if (!has_unused_rowids)
					base[k] |= bit;
				if (rowid == UINT_MAX)
				{
					map |= bit;
					min_id = 0;
					goto retry;
				}
			}
			break;
		}
	}

	if ((base[0] & base[1] & base[2] & base[3]) == ~0UL)
		*p_has_unused_rowids = false;
	else
		*p_has_unused_rowids = true;
	
	return rowid;
}

static cl_uint
gstoreFdwAllocateRowId(GpuStoreDesc *gs_desc, cl_uint min_rowid)
{
	GpuStoreRowIdMapHead *rowid_map = gs_desc->rowid_map;
	GpuStoreSharedState *gs_sstate  = gs_desc->gs_sstate;
	cl_uint		rowid;
	cl_bool		has_unused_rowids;

	if (min_rowid < rowid_map->nrooms)
	{
		if (rowid_map->nrooms <= (1U << 8))
			min_rowid = (min_rowid << 24);		/* single level */
		else if (rowid_map->nrooms <= (1U << 16))
			min_rowid = (min_rowid << 16);		/* dual level */
		else if (rowid_map->nrooms <= (1U << 24))
			min_rowid = (min_rowid << 8);		/* triple level */
		SpinLockAcquire(&gs_sstate->rowid_map_lock);
		rowid = __gstoreFdwAllocateRowId(rowid_map->base,
										 rowid_map->nrooms,
										 min_rowid, 0, 0,
										 &has_unused_rowids);
		SpinLockRelease(&gs_sstate->rowid_map_lock);
		return rowid;
	}
	return UINT_MAX;
}

static bool
__gstoreFdwReleaseRowId(cl_ulong *base, cl_uint nrooms, cl_uint rowid)
{
	cl_ulong   *bottom;
	
	if (rowid < nrooms)
	{
		if (nrooms <= (1U << 8))
		{
			/* single level */
			Assert((rowid & 0xffffff00) == 0);
			bottom = base;
			if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f))) == 0)
				return false;	/* RowID is not allocated yet */
			base[rowid >> 6] &= ~(1UL << (rowid & 0x3f));
		}
		else if (nrooms <= (1U << 16))
		{
			/* double level */
			Assert((rowid & 0xffff0000) == 0);
			bottom = base + 4;
			if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f))) == 0)
				return false;	/* RowID is not allocated yet */
			base[(rowid >> 14)] &= ~(1UL << ((rowid >> 8) & 0x3f));
			base += 4;
			base[(rowid >>  6)] &= ~(1UL << (rowid & 0x3f));
		}
		else if (nrooms <= (1U << 24))
		{
			/* triple level */
			Assert((rowid & 0xff000000) == 0);
			bottom = base + 4 + 1024;
			if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f))) == 0)
				return false;	/* RowID is not allocated yet */
			base[(rowid >> 22)] &= ~(1UL << ((rowid >> 16) & 0x3f));
			base += 4;
			base[(rowid >> 14)] &= ~(1UL << ((rowid >>  8) & 0x3f));
			base += 1024;
			base[(rowid >>  6)] &= ~(1UL << (rowid & 0x3f));
		}
		else
		{
			/* full level */
			bottom = base + 4 + 1024 + 262144;
			if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f))) == 0)
				return false;	/* RowID is not allocated yet */
			base[(rowid >> 30)] &= ~(1UL << ((rowid >> 24) & 0x3f));
			base += 4;
			base[(rowid >> 22)] &= ~(1UL << ((rowid >> 16) & 0x3f));
			base += 1024;
			base[(rowid >> 14)] &= ~(1UL << ((rowid >>  8) & 0x3f));
			base += 262144;
			base[(rowid >>  6)] &= ~(1UL << (rowid & 0x3f));
		}
		return true;
	}
	return false;
}

static void
gstoreFdwReleaseRowId(GpuStoreDesc *gs_desc, cl_uint rowid)
{
    GpuStoreRowIdMapHead *rowid_map = gs_desc->rowid_map;
    GpuStoreSharedState  *gs_sstate = gs_desc->gs_sstate;

	SpinLockAcquire(&gs_sstate->rowid_map_lock);
	__gstoreFdwReleaseRowId(rowid_map->base,
							rowid_map->nrooms, rowid);
	SpinLockRelease(&gs_sstate->rowid_map_lock);
}

/*
 * gstoreFdwCreateBaseFile
 */
static void
gstoreFdwCreateBaseFile(Relation frel, GpuStoreSharedState *gs_sstate,
						File fdesc)
{
	TupleDesc	__tupdesc = gstoreFdwDeviceTupleDesc(frel);
	GpuStoreBaseFileHead *hbuf;
	kern_data_store *schema;
	int			rawfd = FileGetRawDesc(fdesc);
	const char *base_file = FilePathName(fdesc);
	size_t		hbuf_sz, main_sz, sz;
	size_t		rowmap_sz = 0;
	size_t		hash_sz = 0;
	size_t		extra_sz = 0;
	size_t		file_sz;
	size_t		nrooms = gs_sstate->max_num_rows;
	size_t		nslots = gs_sstate->num_hash_slots;
	int			j, unitsz;

	/*
	 * Setup GpuStoreBaseFileHead
	 */
	main_sz = KDS_calculateHeadSize(__tupdesc);
	hbuf_sz = offsetof(GpuStoreBaseFileHead, schema) + main_sz;
	hbuf = alloca(hbuf_sz);
	memset(hbuf, 0, hbuf_sz);
	memcpy(hbuf->signature, GPUSTORE_BASEFILE_SIGNATURE, 8);
	strcpy(hbuf->ftable_name, RelationGetRelationName(frel));

	init_kernel_data_store(&hbuf->schema,
						   __tupdesc,
						   0,	/* to be set later */
						   KDS_FORMAT_COLUMN,
						   nrooms);
	hbuf->schema.table_oid = RelationGetRelid(frel);
	schema = &hbuf->schema;
	for (j=0; j < __tupdesc->natts; j++)
	{
		Form_pg_attribute	attr = tupleDescAttr(__tupdesc, j);
		kern_colmeta	   *cmeta = &schema->colmeta[j];

		if (!attr->attnotnull)
		{
			sz = MAXALIGN(BITMAPLEN(nrooms));
			cmeta->nullmap_offset = __kds_packed(main_sz);
			cmeta->nullmap_length = __kds_packed(sz);
			main_sz += sz;
		}
		
		if (attr->attlen > 0)
		{
			unitsz = att_align_nominal(attr->attlen,
									   attr->attalign);
			sz = MAXALIGN(unitsz * nrooms);
			cmeta->values_offset = __kds_packed(main_sz);
			cmeta->values_length = __kds_packed(sz);
			main_sz += sz;
		}
		else if (attr->attlen == -1)
		{
			/* offset == 0 means NULL-value for varlena */
			sz = MAXALIGN(sizeof(cl_uint) * nrooms);
			cmeta->values_offset = __kds_packed(main_sz);
			cmeta->values_length = __kds_packed(sz);
			main_sz += sz;
			unitsz = get_typavgwidth(attr->atttypid,
									 attr->atttypmod);
			extra_sz += MAXALIGN(unitsz) * nrooms;
		}
		else
		{
			elog(ERROR, "unexpected type length (%d) at %s.%s",
				 attr->attlen,
				 RelationGetRelationName(frel),
				 NameStr(attr->attname));
		}
	}
	schema->length = main_sz;

	file_sz = PAGE_ALIGN(offsetof(GpuStoreBaseFileHead, schema) + main_sz);
	hbuf->rowid_map_offset = file_sz;
	rowmap_sz = gstoreFdwRowIdMapSize(nrooms);
	file_sz += PAGE_ALIGN(rowmap_sz);
	if (gs_sstate->primary_key >= 0)
	{
		hbuf->hash_index_offset = file_sz;
		hash_sz = offsetof(GpuStoreHashIndexHead, slots[nslots + nrooms]);
		file_sz += PAGE_ALIGN(hash_sz);
	}
	if (schema->has_varlena)
	{
		hbuf->schema.extra_hoffset
			= file_sz - offsetof(GpuStoreBaseFileHead, schema);
		hbuf->schema.extra_hlength = extra_sz;
		file_sz += PAGE_ALIGN(extra_sz);
	}
	/* Write out GpuStoreBaseFileHead section */
	if (__writeFile(rawfd, hbuf, hbuf_sz) != hbuf_sz)
		elog(ERROR, "failed on __writeFile('%s'): %m", base_file);
	/* Expand the file size */
	if (posix_fallocate(rawfd, 0, file_sz) != 0)
		elog(ERROR, "failed on posix_fallocate('%s',%zu): %m",
			 base_file, file_sz);
	/* Write out GpuStoreRowIdMapHead section */
	{
		GpuStoreRowIdMapHead rowmap_buf;

		memset(&rowmap_buf, 0, sizeof(GpuStoreRowIdMapHead));
		memcpy(rowmap_buf.signature, GPUSTORE_ROWIDMAP_SIGNATURE, 8);
		rowmap_buf.length = rowmap_sz;
		rowmap_buf.nrooms = gs_sstate->max_num_rows;

		if (lseek(rawfd, hbuf->rowid_map_offset, SEEK_SET) < 0)
			elog(ERROR, "failed on lseek('%s',%zu): %m",
				 base_file, hbuf->rowid_map_offset);
		sz = offsetof(GpuStoreRowIdMapHead, base);
		if (__writeFile(rawfd, &rowmap_buf, sz) != sz)
			elog(ERROR, "failed on __writeFile('%s'): %m", base_file);
	}
	/* Write out GpuStoreHashHead section (if PK is configured) */
	if (gs_sstate->primary_key >= 0)
	{
		GpuStoreHashIndexHead hindex_buf;
		uint32		rowid_buf[4096];
		size_t		nbytes;

		memset(&hindex_buf, 0, sizeof(GpuStoreHashIndexHead));
		memcpy(hindex_buf.signature, GPUSTORE_HASHINDEX_SIGNATURE, 8);
		hindex_buf.nrooms = gs_sstate->max_num_rows;
		hindex_buf.nslots = gs_sstate->num_hash_slots;

		if (lseek(rawfd, hbuf->hash_index_offset, SEEK_SET) < 0)
			elog(ERROR, "failed on lseek('%s',%zu): %m",
				 base_file, hbuf->hash_index_offset);
		sz = offsetof(GpuStoreHashIndexHead, slots);
		if (__writeFile(rawfd, &hindex_buf, sz) != sz)
			elog(ERROR, "failed on __writeFile('%s'): %m", base_file);

		/* '-1' initialization for all the hash-slots/rowid-array */
		memset(rowid_buf, -1, sizeof(rowid_buf));
		nbytes = sizeof(uint32) * (hindex_buf.nrooms + hindex_buf.nslots);
		while (nbytes > 0)
		{
			sz = Min(nbytes, sizeof(rowid_buf));
			if (__writeFile(rawfd, rowid_buf, sz) != sz)
				elog(ERROR, "failed on __writeFile('%s'): %m", base_file);
			nbytes -= sz;
		}
	}
	pfree(__tupdesc);
}

/*
 * gstoreFdwValidateBaseFile
 */
static void
gstoreFdwValidateBaseFile(Relation frel,
						  GpuStoreSharedState *gs_sstate,
						  bool *p_basefile_is_sanity)
{
	TupleDesc	__tupdesc = gstoreFdwDeviceTupleDesc(frel);
	size_t		nrooms = gs_sstate->max_num_rows;
	size_t		nslots = gs_sstate->num_hash_slots;
	size_t		mmap_sz;
	int			mmap_is_pmem;
	size_t		main_sz, sz;
	int			j, unitsz;
	GpuStoreBaseFileHead *base_mmap;
	GpuStoreRowIdMapHead *rowid_map = NULL;
	GpuStoreHashIndexHead *hash_index = NULL;
	kern_data_store *schema;

	base_mmap = pmem_map_file(gs_sstate->base_file, 0,
							  0, 0600,
							  &mmap_sz,
							  &mmap_is_pmem);
	if (!base_mmap)
		elog(ERROR, "failed on pmem_map_file('%s'): %m",
			 gs_sstate->base_file);
	PG_TRY();
	{
		/* GpuStoreBaseFileHead validation  */
		if (memcmp(base_mmap->signature,
				   GPUSTORE_BASEFILE_SIGNATURE, 8) != 0)
		{
			if (memcmp(base_mmap->signature,
					   GPUSTORE_BASEFILE_MAPPED_SIGNATURE, 8) == 0)
				*p_basefile_is_sanity = false;
			else
				elog(ERROR, "file '%s' has wrong signature",
					 gs_sstate->base_file);
		}
		main_sz = KDS_calculateHeadSize(__tupdesc);
		if (offsetof(GpuStoreBaseFileHead, schema) + main_sz > mmap_sz)
			elog(ERROR, "file '%s' is too small than expects",
				 gs_sstate->base_file);

		schema = &base_mmap->schema;
		if (schema->ncols != __tupdesc->natts ||
			schema->nrooms != nrooms ||
			schema->format != KDS_FORMAT_COLUMN ||
			schema->tdtypeid != __tupdesc->tdtypeid ||
			schema->tdtypmod != __tupdesc->tdtypmod ||
			schema->table_oid != RelationGetRelid(frel))
		{
			elog(ERROR, "Base file '%s' has incompatible schema definition",
				 gs_sstate->base_file);
		}

		for (j=0; j < __tupdesc->natts; j++)
		{
			Form_pg_attribute	attr = tupleDescAttr(__tupdesc, j);
			kern_colmeta	   *cmeta = &schema->colmeta[j];

			if ((cmeta->attbyval && !attr->attbyval) ||
				(!cmeta->attbyval && attr->attbyval) ||
				cmeta->attalign != typealign_get_width(attr->attalign) ||
				cmeta->attnum != attr->attnum ||
				cmeta->atttypid != attr->atttypid ||
				cmeta->atttypmod != attr->atttypmod ||
				(cmeta->nullmap_offset != 0 && attr->attnotnull) ||
				(cmeta->nullmap_offset == 0 && !attr->attnotnull))
				elog(ERROR, "Base file '%s' column '%s' is incompatible",
					 gs_sstate->base_file, NameStr(attr->attname));
			if (attr->attnotnull)
			{
				if (cmeta->nullmap_offset != 0 ||
					cmeta->nullmap_length != 0)
					elog(ERROR, "Base file '%s' column '%s' is incompatible",
						 gs_sstate->base_file, NameStr(attr->attname));
			}
			else
			{
				sz = MAXALIGN(BITMAPLEN(nrooms));
				if (cmeta->nullmap_offset != __kds_packed(main_sz) ||
					cmeta->nullmap_length != __kds_packed(sz))
					elog(ERROR, "Base file '%s' column '%s' is incompatible",
						 gs_sstate->base_file, NameStr(attr->attname));
				main_sz += sz;
			}
			
			if (attr->attlen > 0)
			{
				unitsz = att_align_nominal(attr->attlen,
										   attr->attalign);
				sz = MAXALIGN(unitsz * nrooms);
				if (cmeta->values_offset != __kds_packed(main_sz) ||
					cmeta->values_length != __kds_packed(sz))
					elog(ERROR, "Base file '%s' column '%s' is incompatible",
						 gs_sstate->base_file, NameStr(attr->attname));
				main_sz += sz;
			}
			else if (attr->attlen == -1)
			{
				sz = MAXALIGN(sizeof(cl_uint) * nrooms);
				if (cmeta->values_offset != __kds_packed(main_sz) ||
					cmeta->values_length != __kds_packed(sz))
					elog(ERROR, "Base file '%s' column '%s' is incompatible",
						 gs_sstate->base_file, NameStr(attr->attname));
				main_sz += sz;
			}
			else
			{
				elog(ERROR, "unexpected type length (%d) at %s.%s",
					 attr->attlen,
					 RelationGetRelationName(frel),
					 NameStr(attr->attname));
			}
		}
		if (main_sz != schema->length ||
			main_sz >  mmap_sz)
			elog(ERROR, "Base file '%s' is too small then required",
				 gs_sstate->base_file);

		/*
		 * validate GpuStoreRowIdMapHead section
		 */
		sz = gstoreFdwRowIdMapSize(nrooms);
		if (base_mmap->rowid_map_offset + sz > mmap_sz)
			elog(ERROR, "Base file '%s' is too small then necessity",
				 gs_sstate->base_file);
		rowid_map = (GpuStoreRowIdMapHead *)
			((char *)base_mmap + base_mmap->rowid_map_offset);
		if (memcmp(rowid_map->signature,
				   GPUSTORE_ROWIDMAP_SIGNATURE, 8) != 0 ||
			rowid_map->length != sz ||
			rowid_map->nrooms != nrooms)
			elog(ERROR, "Base file '%s' has corrupted RowID-map",
				 gs_sstate->base_file);

		/*
		 * validate GpuStoreHashIndexHead section, if any
		 */
		if (gs_sstate->primary_key < 0)
		{
			if (base_mmap->hash_index_offset != 0)
				elog(ERROR, "Base file '%s' has PK Index, but foreign-table '%s' has no primary key definition",
					 gs_sstate->base_file, RelationGetRelationName(frel));
		}
		else
		{
			if (base_mmap->hash_index_offset == 0)
				elog(ERROR, "Base file '%s' has no PK Index, but foreign-table '%s' has primary key definition",
					 gs_sstate->base_file,
					 RelationGetRelationName(frel));
		sz = offsetof(GpuStoreRowIdMapHead, base[nslots + nrooms]);
		if (base_mmap->hash_index_offset + sz > mmap_sz)
			elog(ERROR, "Base file '%s' is smaller then the estimation",
				 gs_sstate->base_file);
		hash_index = (GpuStoreHashIndexHead *)
			((char *)base_mmap + base_mmap->hash_index_offset);
		if (memcmp(hash_index->signature,
				   GPUSTORE_HASHINDEX_SIGNATURE, 8) != 0 ||
			hash_index->nrooms != nrooms ||
			hash_index->nslots != nslots)
			elog(ERROR, "Base file '%s' has corrupted Hash-index",
				 gs_sstate->base_file);
		}

		/*
		 * validate Extra Buffer of varlena, if any
		 */
		if (!schema->has_varlena)
		{
			if (base_mmap->schema.extra_hoffset != 0 ||
				base_mmap->schema.extra_hlength != 0)
				elog(ERROR, "Base file '%s' has extra buffer, but foreign-table '%s' has no variable-length fields",
					 gs_sstate->base_file, RelationGetRelationName(frel));
		}
		else
		{
			if (base_mmap->schema.extra_hoffset == 0 ||
				base_mmap->schema.extra_hlength == 0)
				elog(ERROR, "Base file '%s' has no extra buffer, but foreign-table '%s' has variable-length fields",
					 gs_sstate->base_file, RelationGetRelationName(frel));
			if (offsetof(GpuStoreBaseFileHead, schema) +
				base_mmap->schema.extra_hoffset +
				base_mmap->schema.extra_hlength < mmap_sz)
				elog(ERROR, "Base file '%s' is smaller then the required",
					 gs_sstate->base_file);
		}
	}
	PG_CATCH();
	{
		if (pmem_unmap(base_mmap, mmap_sz) != 0)
			elog(WARNING, "failed on pmem_unmap: %m");
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* Ok, validated */
	if (pmem_unmap(base_mmap, mmap_sz) != 0)
		elog(WARNING, "failed on pmem_unmap: %m");
	pfree(__tupdesc);
}

/*
 * gstoreFdwCreateOrValidateBaseFile
 */
static void
gstoreFdwCreateOrValidateBaseFile(Relation frel,
								  GpuStoreSharedState *gs_sstate,
								  bool *p_basefile_is_sanity)
{
	File		fdesc;

	fdesc = PathNameOpenFile(gs_sstate->base_file, O_RDWR | O_CREAT | O_EXCL);
	if (fdesc < 0)
	{
		if (errno != EEXIST)
			elog(ERROR, "failed on open('%s'): %m", gs_sstate->base_file);
		/* Base file already exists */
		gstoreFdwValidateBaseFile(frel, gs_sstate, p_basefile_is_sanity);
	}
	else
	{
		PG_TRY();
		{
			gstoreFdwCreateBaseFile(frel, gs_sstate, fdesc);
		}
		PG_CATCH();
		{
			if (unlink(gs_sstate->base_file) != 0)
				elog(WARNING, "failed on unlink('%s'): %m",
					 gs_sstate->base_file);
			FileClose(fdesc);
			PG_RE_THROW();
		}
		PG_END_TRY();
		FileClose(fdesc);
	}
}

/*
 * gstoreFdwOpenRedoLog
 */
static void
gstoreFdwCreateRedoLog(Relation frel,
					   GpuStoreSharedState *gs_sstate)
{
	const char *redo_log_file = gs_sstate->redo_log_file;
	File		redo_fdesc;

	redo_fdesc = PathNameOpenFile(redo_log_file, O_RDWR);
	if (redo_fdesc < 0)
	{
		/* Create a new REDO log file; which is empty of course. */
		if (errno != ENOENT)
			elog(ERROR, "failed on open('%s'): %m", redo_log_file);
		redo_fdesc = PathNameOpenFile(redo_log_file,
									  O_RDWR | O_CREAT | O_EXCL);
		if (redo_fdesc < 0)
			elog(ERROR, "failed on open('%s'): %m", redo_log_file);
		PG_TRY();
		{
			GpuStoreRedoLogHead temp;
			int			rawfd = FileGetRawDesc(redo_fdesc);
			size_t		sz;

			sz = PAGE_ALIGN(gs_sstate->redo_log_limit);
			if (posix_fallocate(rawfd, 0, sz) != 0)
				elog(ERROR, "failed on posix_fallocate('%s',%zu): %m",
					 redo_log_file, sz);

			memset(&temp, 0, sizeof(temp));
			memcpy(temp.signature, GPUSTORE_REDOLOG_SIGNATURE, 8);
			temp.timestamp = GetCurrentTimestamp();
			sz = MAXALIGN(offsetof(GpuStoreRedoLogHead, data));
			temp.checkpoint_offset = sz;
			if (__writeFile(rawfd, &temp, sz) != sz)
				elog(ERROR, "failed on __writeFile('%s'): %m", redo_log_file);
		}
		PG_CATCH();
		{
			unlink(redo_log_file);
			PG_RE_THROW();
		}
		PG_END_TRY();
		FileClose(redo_fdesc);
	}
	else
	{
		struct stat	stat_buf;
		int			rawfd = FileGetRawDesc(redo_fdesc);
		size_t		sz;

		if (fstat(rawfd, &stat_buf) != 0)
			elog(ERROR, "failed on fstat('%s'): %m", redo_log_file);
		/* expand the Redo-log file on demand */
		sz = PAGE_ALIGN(gs_sstate->redo_log_limit);
		if (stat_buf.st_size < sz)
		{
			if (posix_fallocate(rawfd, 0, sz) != 0)
				elog(ERROR, "failed on posix_fallocate('%s',%zu): %m",
					 redo_log_file, sz);
		}
		FileClose(redo_fdesc);
	}
}

/*
 * GSTORE_TX_LOG__INSERT
 */
static void
__ApplyRedoLogInsert(Relation frel,
					 GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz,
					 GstoreTxLogRow *r_log)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	Datum		   *values = alloca(sizeof(Datum) * tupdesc->natts);
	bool		   *isnull = alloca(sizeof(bool) * tupdesc->natts);
	kern_data_store *kds = &base_mmap->schema;
	HeapTupleData	tuple;
	TransactionId	t_xmin = r_log->htup.t_choice.t_heap.t_xmin;
	CommandId		t_cid = r_log->htup.t_choice.t_heap.t_field3.t_cid;
	int				j;

	memset(&tuple, 0, sizeof(HeapTupleData));
	tuple.t_data = &r_log->htup;
	heap_deform_tuple(&tuple, tupdesc,
					  values, isnull);
	Assert(kds->ncols == tupdesc->natts + 3);	/* xmin,xmax,cid */
	for (j=0; j < tupdesc->natts; j++)
	{
		KDS_store_datum_column(kds, &kds->colmeta[j],
							   r_log->rowid,
							   values[j], isnull[j]);
	}
	/* xmin */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts],
						   r_log->rowid, t_xmin, false);
	/* xmax */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts+1],
						   r_log->rowid, InvalidTransactionId, false);
	/* cid */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts+2],
						   r_log->rowid, t_cid, false);
}

/*
 * GSTORE_TX_LOG__UPDATE
 */
static void
__ApplyRedoLogUpdate(Relation frel,
					 GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz,
					 GstoreTxLogRow *r_log)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	Datum		   *values = alloca(sizeof(Datum) * tupdesc->natts);
	bool		   *isnull = alloca(sizeof(bool)  * tupdesc->natts);
	kern_data_store *kds = &base_mmap->schema;
	HeapTupleData	tuple;
	bits8		   *update_mask = NULL;
	TransactionId	t_xmin = r_log->htup.t_choice.t_heap.t_xmin;
	CommandId		t_cid = r_log->htup.t_choice.t_heap.t_field3.t_cid;
	cl_uint			oldid;
	int				j;

	oldid = (((cl_uint)r_log->htup.t_ctid.ip_blkid.bi_hi << 16) |
			 ((cl_uint)r_log->htup.t_ctid.ip_blkid.bi_lo));
	if (r_log->update_mask != 0)
		update_mask = (bits8 *)((char *)r_log + r_log->update_mask);

	memset(&tuple, 0, sizeof(HeapTupleData));
	tuple.t_data = &r_log->htup;
	heap_deform_tuple(&tuple, tupdesc,
					  values, isnull);
	for (j=0; j < tupdesc->natts; j++)
	{
		Datum		__datum = values[j];
		bool		__isnull = isnull[j];

		if (update_mask && (update_mask[j>>3] & (1<<(j&7))) != 0)
		{
			/* column is not updated */
			__datum = KDS_fetch_datum_column(kds, &kds->colmeta[j],
											 oldid,
											 &__isnull);
		}
		KDS_store_datum_column(kds, &kds->colmeta[j],
							   r_log->rowid,
							   __datum, __isnull);		
	}
	/* xmin (new) */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts],
                           r_log->rowid, t_xmin, false);
	/* cid (new) */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts+2],
						   r_log->rowid, t_cid, false);
	/* xmax (old) */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts+1],
						   oldid, t_xmin, false);
	/* cid (old) */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts+2],
						   oldid, t_cid, false);
}

/*
 * GSTORE_TX_LOG__DELETE
 */
static void
__ApplyRedoLogDelete(Relation frel,
					 GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz,
					 GstoreTxLogRow *r_log)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	kern_data_store *kds = &base_mmap->schema;
	TransactionId	t_xmax = r_log->htup.t_choice.t_heap.t_xmax;
	CommandId		t_cid = r_log->htup.t_choice.t_heap.t_field3.t_cid;

	/* xmax */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts+1],
						   r_log->rowid, t_xmax, false);
	/* cid */
	KDS_store_datum_column(kds, &kds->colmeta[tupdesc->natts+2],
						   r_log->rowid, t_cid, false);
}

static void
__rebuildRowIdMap(Relation frel, GpuStoreSharedState *gs_sstate,
				  GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz)
{
	/* TODO: rebuild RowIdMap */
}

static void
__rebuildHashIndex(Relation frel, GpuStoreSharedState *gs_sstate,
				   GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz)
{
	/* TODO: rebuild HashIndex */
}

/*
 * __applyRedoLogMain
 */
static size_t
__ApplyRedoLogMain(Relation frel, GpuStoreSharedState *gs_sstate,
				   GpuStoreRedoLogHead *redo_mmap, size_t redo_mmap_sz)
{
	GpuStoreBaseFileHead *base_mmap = NULL;
	size_t		base_mmap_sz;
	int			base_is_pmem;
	size_t		redo_pos = 0;

	/* Map the base file */
	base_mmap = pmem_map_file(gs_sstate->base_file, 0,
							  0, 0600,
							  &base_mmap_sz,
							  &base_is_pmem);
	if (!base_mmap)
		elog(ERROR, "failed on pmem_map_file('%s'): %m",
			 gs_sstate->base_file);
	PG_TRY();
	{
		redo_pos = redo_mmap->checkpoint_offset;
		while (redo_pos + offsetof(GstoreTxLogCommon, data) < redo_mmap_sz)
		{
			GstoreTxLogCommon *tx_log = (GstoreTxLogCommon *)
				((char *)redo_mmap + redo_pos);

			if (tx_log->type == GSTORE_TX_LOG__INSERT ||
				tx_log->type == GSTORE_TX_LOG__UPDATE ||
				tx_log->type == GSTORE_TX_LOG__DELETE)
			{
				GstoreTxLogRow *r_log = (GstoreTxLogRow *)tx_log;
				size_t		sz;

				/* expand the extra buffer on demand */
				if (base_mmap->schema.has_varlena &&
					(r_log->type == GSTORE_TX_LOG__INSERT ||
					 r_log->type == GSTORE_TX_LOG__UPDATE))
				{
					/*
					 * This length calculation is not correct strictly,
					 * however, can allocate sufficient amount of extra
					 * buffer preliminary. We expect it shall be consumed
					 * later. ;-)
					 */
					sz = r_log->length - offsetof(GstoreTxLogRow, htup);
					sz += __kds_unpack(base_mmap->schema.usage);
					if (sz > base_mmap->schema.extra_hlength)
					{
						int		rawfd;
						
						if (pmem_unmap(base_mmap, base_mmap_sz) != 0)
							elog(ERROR, "failed on pmem_unmap: %m");
						base_mmap = NULL;

						sz = PAGE_ALIGN(sz + (64UL << 20));
						rawfd = open(gs_sstate->base_file, O_RDWR);
						if (rawfd < 0)
							elog(ERROR, "failed on open('%s'): %m",
								 gs_sstate->base_file);

						if (posix_fallocate(rawfd, 0, sz) != 0)
						{
							close(rawfd);
							elog(ERROR, "failed on posix_fallocate('%s'): %m",
								 gs_sstate->base_file);
						}
						close(rawfd);

						/* Map the base file again */
						base_mmap = pmem_map_file(gs_sstate->base_file, 0,
												  0, 0600,
												  &base_mmap_sz,
												  &base_is_pmem);
						if (!base_mmap)
							elog(ERROR, "failed on pmem_map_file('%s'): %m",
								 gs_sstate->base_file);
						/* adjust extra buffer size */
						base_mmap->schema.extra_hlength = base_mmap_sz -
							(offsetof(GpuStoreBaseFileHead, schema) +
							 base_mmap->schema.extra_hoffset);
						if (base_is_pmem)
							pmem_persist(base_mmap, PAGE_SIZE);
						else
							pmem_msync(base_mmap, PAGE_SIZE);
					}
				}
				if (r_log->type == GSTORE_TX_LOG__INSERT)
					__ApplyRedoLogInsert(frel, base_mmap, base_mmap_sz, r_log);
                else if (r_log->type == GSTORE_TX_LOG__UPDATE)
					__ApplyRedoLogUpdate(frel, base_mmap, base_mmap_sz, r_log);
				else if (r_log->type == GSTORE_TX_LOG__DELETE)
					__ApplyRedoLogDelete(frel, base_mmap, base_mmap_sz, r_log);
				else
					elog(ERROR, "Bug? unexpected Redo-Log type");
				redo_pos += r_log->length;
			}
			else if (tx_log->type == GSTORE_TX_LOG__COMMIT)
			{
				/* commit log can be ignored on CPU side */
				redo_pos += tx_log->length;
			}
			else
			{
				/* end of the REDO log */
				break;
			}
		}
		/* Rebuild indexes */
		 __rebuildRowIdMap(frel, gs_sstate, base_mmap, base_mmap_sz);
		 __rebuildHashIndex(frel, gs_sstate, base_mmap, base_mmap_sz);
		 if (base_is_pmem)
			 pmem_persist(base_mmap, base_mmap_sz);
		 else
			 pmem_msync(base_mmap, base_mmap_sz);
	}
	PG_CATCH();
	{
		if (base_mmap)
			pmem_unmap(base_mmap, base_mmap_sz);
		PG_RE_THROW();
	}
	PG_END_TRY();
	if (pmem_unmap(base_mmap, base_mmap_sz) != 0)
		elog(WARNING, "failed on pmem_unmap('%s',%zu): %m",
			 gs_sstate->base_file, base_mmap_sz);
	return redo_pos;
}

static uint64
gstoreFdwApplyRedoLog(Relation frel, GpuStoreSharedState *gs_sstate,
					  bool basefile_is_sanity)
{
	GpuStoreRedoLogHead *redo_mmap;
	size_t		redo_mmap_sz;
	int			redo_is_pmem;
	uint64		redo_pos;

	/* Map REDO-Log */
	redo_mmap = pmem_map_file(gs_sstate->redo_log_file, 0,
							  0, 0600,
							  &redo_mmap_sz,
							  &redo_is_pmem);
	if (!redo_mmap)
		elog(ERROR, "failed on pmem_map_file('%s'): %m",
			 gs_sstate->redo_log_file);
	PG_TRY();
	{
		if (!basefile_is_sanity)
		{
			/* Apply REDO-log and rebuild row/hash index */
			redo_pos = __ApplyRedoLogMain(frel, gs_sstate,
										  redo_mmap, redo_mmap_sz);
		}
		else
		{
			/* No need to apply REDO-log, just seek to the position */
			redo_pos = redo_mmap->checkpoint_offset;
			while (redo_pos + offsetof(GstoreTxLogCommon,
									   data) < redo_mmap_sz)
			{
				GstoreTxLogCommon  *tx_log = (GstoreTxLogCommon *)
					((char *)redo_mmap + redo_pos);

				if (tx_log->type == GSTORE_TX_LOG__INSERT ||
					tx_log->type == GSTORE_TX_LOG__UPDATE ||
					tx_log->type == GSTORE_TX_LOG__DELETE ||
					tx_log->type == GSTORE_TX_LOG__COMMIT)
				{
					/* Ok, likely it is a valid REDO log, move to the next */
					redo_pos += tx_log->length;
				}
				else
				{
					if (tx_log->type != 0)
						elog(LOG, "Redo Log file '%s' meets uncertain magic code %08x at %lu. Base file '%s' is sanity, so restart logs from here.\n",
							 gs_sstate->redo_log_file,
							 tx_log->type,
							 redo_pos,
							 gs_sstate->base_file);
					break;
				}
			}
		}
	}
	PG_CATCH();
	{
		pmem_unmap(redo_mmap, redo_mmap_sz);
		PG_RE_THROW();
	}
	PG_END_TRY();
	if (pmem_unmap(redo_mmap, redo_mmap_sz) != 0)
		elog(WARNING, "failed on pmem_unmap('%s',%zu): %m",
			 gs_sstate->redo_log_file, redo_mmap_sz);

	return redo_pos;
}

#if 0
static void
__gstoreFdwBuildHashIndexMain(kern_data_store *kds,
							  AttrNumber primary_key,
							  char *extra_buf, size_t extra_sz,
							  GpuStoreHashHead *hash_head)
{
	kern_colmeta   *cmeta = &kds->colmeta[primary_key - 1];	/* PK column */
	kern_colmeta   *smeta = &kds->colmeta[kds->ncols - 1];	/* Sys column */
	TypeCacheEntry *tcache;
	char		   *nullmap;
	char		   *cbase;
	char		   *sbase;
	uint32		   *hash_rowmap;
	cl_uint			i, unitsz;

	tcache = lookup_type_cache(cmeta->atttypid,
							   TYPECACHE_HASH_PROC |
							   TYPECACHE_HASH_PROC_FINFO);
	if (!OidIsValid(tcache->hash_proc))
		elog(ERROR, "type '%s' has no hash function",
			 format_type_be(cmeta->atttypid));

	Assert(smeta->attbyval && smeta->attlen == sizeof(cl_ulong));
	nullmap = (char *)kds + cmeta->nullmap_offset;
	cbase = (char *)kds + cmeta->values_offset;
	sbase = (char *)kds + smeta->values_offset;

	hash_rowmap = (uint32 *)&hash_head->slots[hash_head->nslots];
	for (i=0; i < kds->nrooms; i++)
	{
		Timestamp	ts = ((uint64 *)sbase)[i];
		Datum		datum;
		Datum		hash;
		uint32		rowid;

		/* Is this row already removed? */
		if ((ts & GSTORE_SYSATTR__REMOVED) != 0)
			continue;
		/* PK should never be NULL */
		if (att_isnull(i, nullmap))
			elog(ERROR, "data corruption? Primary Key is NULL (rowid=%u)", i);
		/* Fetch value */
		if (cmeta->attbyval)
		{
			if (cmeta->attlen == sizeof(cl_ulong))
				datum = ((cl_ulong *)cbase)[i];
			else if (cmeta->attlen == sizeof(cl_uint))
				datum = ((cl_uint *)cbase)[i];
			else if (cmeta->attlen == sizeof(cl_ushort))
				datum = ((cl_ushort *)cbase)[i];
			else if (cmeta->attlen == sizeof(cl_uchar))
				datum = ((cl_uchar *)cbase)[i];
			else
				elog(ERROR, "unexpected type definition");
		}
		else if (cmeta->attlen > 0)
		{
			unitsz = TYPEALIGN(cmeta->attalign, cmeta->attlen);
			datum = PointerGetDatum(cbase + unitsz * i);
		}
		else if (cmeta->attlen == -1)
		{
			size_t		offset = __kds_unpack(((cl_uint *)cbase)[i]);
			if (offset >= extra_sz)
				elog(ERROR, "varlena points out of extra buffer");
			datum = PointerGetDatum(extra_buf + offset);
		}
		else
			elog(ERROR, "unexpected type definition");
		/* call the type hash function */
		hash = FunctionCall1(&tcache->hash_proc_finfo, datum);
		hash = hash % hash_head->nslots;
		rowid = hash_head->slots[hash].rowid;

		hash_head->slots[hash].rowid = i;
		if (rowid != UINT_MAX)
			hash_rowmap[i] = rowid;
	}
}

static uint32
gstoreFdwBuildHashIndex(Relation frel, File base_fdesc,
						AttrNumber primary_key)
{
	GpuStoreFileHead *base_head;
	kern_data_store *kds;
	char		   *extra_buf;
	size_t			extra_sz;
	size_t			main_sz;
	char			namebuf[200];
	uint32			suffix;
	int				rawfd = -1;
	uint64			nslots;
	size_t			hash_sz;
	struct stat 	stat_buf;
	GpuStoreHashHead *hash_head;

	/* mmap base file */
	if (fstat(FileGetRawDesc(base_fdesc), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", FilePathName(base_fdesc));
	base_head = __mmapFile(NULL, TYPEALIGN(PAGE_SIZE, stat_buf.st_size),
						   PROT_READ | PROT_WRITE,
						   MAP_SHARED | MAP_POPULATE,
						   FileGetRawDesc(base_fdesc), 0);
	kds = &base_head->schema;
	main_sz = offsetof(GpuStoreFileHead, schema) + kds->length;
	if (stat_buf.st_size < main_sz)
		elog(ERROR, "base file ('%s') header corruption",
			 FilePathName(base_fdesc));
	extra_buf = (char *)base_head + main_sz;
	extra_sz = stat_buf.st_size - main_sz;

	/* open a new shared memory segment */
	do {
		suffix = random();
		if (suffix == 0)
			continue;
		snprintf(namebuf, sizeof(namebuf), "/gstore_fdw.index.%u", suffix);
		rawfd = shm_open(namebuf, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (rawfd < 0 && errno != EEXIST)
			elog(ERROR, "failed on shm_open('%s'): %m", namebuf);
	} while (rawfd < 0);

	PG_TRY();
	{
		nslots = 1.2 * (double)kds->nrooms + 1000;
		hash_sz = offsetof(GpuStoreHashHead,
						   slots[nslots]) + sizeof(uint32) * kds->nrooms;
		if (fallocate(rawfd, 0, 0, hash_sz) != 0)
			elog(ERROR, "failed on fallocate('%s',%zu): %m", namebuf, hash_sz);
		hash_head = __mmapFile(NULL, TYPEALIGN(PAGE_SIZE, hash_sz),
							   PROT_READ | PROT_WRITE,
							   MAP_SHARED | MAP_POPULATE,
							   rawfd, 0);
		if (hash_head == MAP_FAILED)
			elog(ERROR, "failed on mmap('%s',%zu): %m", namebuf, hash_sz);

		memset(hash_head, -1, hash_sz);
		memcpy(hash_head->signature, GPUSTORE_HASHINDEX_SIGNATURE, 8);
		hash_head->primary_key = primary_key;
		hash_head->nrooms = kds->nrooms;
		hash_head->nslots = nslots;

		close(rawfd);
	}
	PG_CATCH();
	{
		close(rawfd);
		shm_unlink(namebuf);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* Ok, both of base/hash files are mapped, go to the index build */
	__gstoreFdwBuildHashIndexMain(kds, primary_key,
								  extra_buf, extra_sz, hash_head);
	__munmapFile(hash_head);
	__munmapFile(base_head);

	return suffix;
}
#endif

static GpuStoreSharedState *
gstoreFdwCreateSharedState(Relation frel)
{
	GpuStoreSharedState *gs_sstate = gstoreFdwAllocSharedState(frel);
	uint64		redo_pos;

	PG_TRY();
	{
		bool	basefile_is_sanity = true;

		/* Create or validate base file */
		gstoreFdwCreateOrValidateBaseFile(frel, gs_sstate,
										  &basefile_is_sanity);
		/* Create redo-log file on demand */
		gstoreFdwCreateRedoLog(frel, gs_sstate);
		/* Apply REDO-log, if needed */
		redo_pos = gstoreFdwApplyRedoLog(frel, gs_sstate,
										 basefile_is_sanity);
		pg_atomic_init_u64(&gs_sstate->redo_log_pos, redo_pos);
	}
	PG_CATCH();
	{
		pfree(gs_sstate);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Kick GPU memory synchronizer for initial allocation */
	SpinLockAcquire(&gstore_shared_head->gpumem_sync_lock);
	dlist_push_tail(&gstore_shared_head->gpumem_sync_list,
					&gs_sstate->sync_chain);
	if (gstore_shared_head->gpumem_sync_latch)
		SetLatch(gstore_shared_head->gpumem_sync_latch);
	SpinLockRelease(&gstore_shared_head->gpumem_sync_lock);

	return gs_sstate;
}

static GpuStoreDesc *
__gstoreFdwSetupGpuStoreDesc(GpuStoreDesc *gs_desc)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	GpuStoreBaseFileHead *base_mmap;
	GpuStoreRedoLogHead *redo_mmap;
	size_t		mmap_sz;
	int			is_pmem;

	if (gs_desc->mmap_revision == gs_sstate->mmap_revision)
		return gs_desc;		/* nothing to do */
	PG_TRY();
	{
		/* mmap base-file */
		base_mmap = pmem_map_file(gs_sstate->base_file, 0,
								  0, 0600,
								  &mmap_sz,
								  &is_pmem);
		if (!base_mmap)
			elog(ERROR, "failed on pmem_map_file('%s'): %m",
				 gs_sstate->base_file);
		gs_desc->base_mmap = base_mmap;
		gs_desc->base_mmap_sz = mmap_sz;
		gs_desc->base_mmap_is_pmem = is_pmem;
		gs_desc->rowid_map = (GpuStoreRowIdMapHead *)
			((char *)base_mmap + base_mmap->rowid_map_offset);
		if (base_mmap->hash_index_offset != 0)
			gs_desc->hash_index = (GpuStoreHashIndexHead *)
				((char *)base_mmap + base_mmap->hash_index_offset);

		/* mmap redo-log */
		redo_mmap = pmem_map_file(gs_sstate->redo_log_file, 0,
								  0, 0600,
								  &mmap_sz,
								  &is_pmem);
		if (!redo_mmap)
			elog(ERROR, "failed on pmem_map_file('%s'): %m",
				 gs_sstate->redo_log_file);
		gs_desc->redo_mmap = redo_mmap;
		gs_desc->redo_mmap_sz = mmap_sz;
		gs_desc->redo_mmap_is_pmem = is_pmem;

		gs_desc->mmap_revision = gs_sstate->mmap_revision;
	}
	PG_CATCH();
	{
		if (gs_desc->base_mmap)
		{
			pmem_unmap(gs_desc->base_mmap,
					   gs_desc->base_mmap_sz);
			gs_desc->base_mmap = NULL;
			gs_desc->base_mmap_sz = 0;
			gs_desc->base_mmap_is_pmem = 0;
		}
		if (gs_desc->redo_mmap)
		{
			pmem_unmap(gs_desc->redo_mmap,
					   gs_desc->redo_mmap_sz);
			gs_desc->redo_mmap = NULL;
			gs_desc->redo_mmap_sz = 0;
			gs_desc->redo_mmap_is_pmem = 0;
		}
		PG_RE_THROW();
	}
	PG_END_TRY();

	return gs_desc;
}

static GpuStoreDesc *
gstoreFdwSetupGpuStoreDesc(GpuStoreDesc *gs_desc)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;

	pthreadRWLockReadLock(&gs_sstate->mmap_lock);
	PG_TRY();
	{
		__gstoreFdwSetupGpuStoreDesc(gs_desc);
	}
	PG_CATCH();
	{
		pthreadRWLockUnlock(&gs_sstate->mmap_lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	pthreadRWLockUnlock(&gs_sstate->mmap_lock);

	return gs_desc;
}

static GpuStoreDesc *
gstoreFdwLookupGpuStoreDesc(Relation frel)
{
	GpuStoreDesc *gs_desc;
	Oid			hkey[2];
	bool		found;

	hkey[0] = MyDatabaseId;
	hkey[1] = RelationGetRelid(frel);
	gs_desc = (GpuStoreDesc *) hash_search(gstore_desc_htab,
										   hkey,
										   HASH_ENTER,
										   &found);
	if (!found)
	{
		GpuStoreSharedState *gs_sstate = NULL;
		int			hindex;
		slock_t	   *hlock;
		dlist_head *hslot;
		dlist_iter	iter;

		hindex = hash_any((const unsigned char *)hkey,
						  sizeof(hkey)) % GPUSTORE_SHARED_DESC_NSLOTS;
		hlock = &gstore_shared_head->gstore_sstate_lock[hindex];
		hslot = &gstore_shared_head->gstore_sstate_slot[hindex];
		SpinLockAcquire(hlock);

		PG_TRY();
		{
			dlist_foreach(iter, hslot)
			{
				gs_sstate = dlist_container(GpuStoreSharedState,
											hash_chain, iter.cur);
				if (gs_sstate->database_oid == MyDatabaseId &&
					gs_sstate->ftable_oid == RelationGetRelid(frel))
					goto found;
			}
			gs_sstate = gstoreFdwCreateSharedState(frel);
			dlist_push_tail(hslot, &gs_sstate->hash_chain);
			gs_desc->gs_sstate = gs_sstate;
		found:
			;
		}
		PG_CATCH();
		{
			SpinLockRelease(hlock);
			hash_search(gstore_desc_htab, hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(hlock);
	}
	return gstoreFdwSetupGpuStoreDesc(gs_desc);
}









Datum
pgstrom_gstore_fdw_post_creation(PG_FUNCTION_ARGS)
{
	EventTriggerData   *trigdata;

	if (!CALLED_AS_EVENT_TRIGGER(fcinfo))
		elog(ERROR, "%s must be called as a event trigger", __FUNCTION__);

	trigdata = (EventTriggerData *) fcinfo->context;
	if (strcmp(trigdata->event, "ddl_command_end") != 0)
		elog(ERROR, "%s must be called at ddl_command_end", __FUNCTION__);
	elog(INFO, "tag [%s]", trigdata->tag);
	if (strcmp(trigdata->tag, "CREATE FOREIGN TABLE") == 0)
	{
		CreateStmt *stmt = (CreateStmt *)trigdata->parsetree;
		Relation	frel;

		frel = relation_openrv_extended(stmt->relation, AccessShareLock, true);
		if (!frel)
			PG_RETURN_NULL();
		elog(INFO, "table [%s]", RelationGetRelationName(frel));
		if (frel->rd_rel->relkind == RELKIND_FOREIGN_TABLE)
		{
			FdwRoutine	   *routine = GetFdwRoutineForRelation(frel, false);

			if (memcmp(routine, &pgstrom_gstore_fdw_routine,
					   sizeof(FdwRoutine)) == 0)
			{
				/*
				 * Ensure the supplied FDW options are reasonable,
				 * and create base/redo-log files preliminary.
				 */
				gstoreFdwLookupGpuStoreDesc(frel);
			}
		}
		relation_close(frel, AccessShareLock);
	}
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_post_creation);

Datum
pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&pgstrom_gstore_fdw_routine);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_handler);

Datum
pgstrom_gstore_fdw_synchronize(PG_FUNCTION_ARGS)
{
	/* not implement yet */
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_synchronize);

Datum
pgstrom_gstore_fdw_compaction(PG_FUNCTION_ARGS)
{
	/* not implement yet */
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_compaction);

/*
 * GpuBufferUpdaterInitialLoad
 *
 * must be called under the exclusive lock of gs_sstate->gpu_bufer_lock
 */
static bool
GpuBufferUpdaterInitialLoad(GpuStoreDesc *gs_desc)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	GpuStoreBaseFileHead *base_mmap = gs_desc->base_mmap;
	kern_data_store *schema = &base_mmap->schema;
	const char *ftable_name = base_mmap->ftable_name;
	CUresult	rc;

	/* main portion of the device buffer */
	rc = cuMemAlloc(&gs_desc->gpu_main_devptr, schema->length);
	if (rc != CUDA_SUCCESS)
		goto error_0;
	rc = cuMemcpyHtoD(gs_desc->gpu_main_devptr, schema, schema->length);
	if (rc != CUDA_SUCCESS)
		goto error_1;
	rc = cuIpcGetMemHandle(&gs_sstate->gpu_main_mhandle,
						   gs_desc->gpu_main_devptr);
	if (rc != CUDA_SUCCESS)
		goto error_1;
	elog(LOG, "GstoreFdw: %s main buffer (sz: %lu) allocated",
		 ftable_name, schema->length);

	/* extra portion of the device buffer (if needed) */
	if (schema->has_varlena)
	{
		rc = cuMemAlloc(&gs_desc->gpu_extra_devptr,
						schema->extra_hlength);
		if (rc != CUDA_SUCCESS)
			goto error_1;
		rc = cuMemcpyHtoD(gs_desc->gpu_extra_devptr,
						  (char *)schema + schema->extra_hoffset,
						  schema->extra_hlength);
		if (rc != CUDA_SUCCESS)
			goto error_2;
		rc = cuIpcGetMemHandle(&gs_sstate->gpu_extra_mhandle,
							   gs_desc->gpu_extra_devptr);
		if (rc != CUDA_SUCCESS)
			goto error_2;

		elog(LOG, "GstoreFdw: %s extra buffer (sz: %lu) allocated",
			 ftable_name, schema->extra_hlength);
	}
	return true;

error_2:
	cuMemFree(gs_desc->gpu_extra_devptr);
error_1:
	cuMemFree(gs_desc->gpu_main_devptr);
error_0:
	gs_desc->gpu_extra_devptr = 0UL;
	gs_desc->gpu_main_devptr = 0UL;
	memset(&gs_sstate->gpu_main_mhandle, 0, sizeof(CUipcMemHandle));
	memset(&gs_sstate->gpu_extra_mhandle, 0, sizeof(CUipcMemHandle));

	return false;
}

static void
GpuBufferUpdaterApplyLogs(GpuStoreDesc *gs_desc)
{
	//GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;

	//TODO: kick log apply kernel
	
}

/*
 * BeginGstoreFdwGpuBufferUpdate
 *
 * It is called once when GPU memory keeper bgworker process launched.
 */
void
BeginGstoreFdwGpuBufferUpdate(void)
{
	gstore_shared_head->gpumem_sync_latch = MyLatch;
}

/*
 * DispatchGstoreFdwGpuUpdator
 *
 * It is called every time when GPU memory keeper bgworker process wake up.
 */
bool
DispatchGstoreFdwGpuUpdator(CUcontext *cuda_context_array)
{
	GpuStoreSharedState *gs_sstate = NULL;
	GpuStoreDesc   *gs_desc = NULL;
	dlist_node	   *dnode;
	CUcontext		cuda_context;
	CUresult		rc;
	bool			found;
	bool			retval = false;

	SpinLockAcquire(&gstore_shared_head->gpumem_sync_lock);
	if (dlist_is_empty(&gstore_shared_head->gpumem_sync_list))
	{
		SpinLockRelease(&gstore_shared_head->gpumem_sync_lock);
		return false;
	}
	dnode = dlist_pop_head_node(&gstore_shared_head->gpumem_sync_list);
	gs_sstate = dlist_container(GpuStoreSharedState, sync_chain, dnode);
	memset(&gs_sstate->sync_chain, 0, sizeof(dlist_node));
	if (!dlist_is_empty(&gstore_shared_head->gpumem_sync_list))
		retval = true;
	SpinLockRelease(&gstore_shared_head->gpumem_sync_lock);

	/* switch CUDA context */
	cuda_context = cuda_context_array[gs_sstate->cuda_dindex];
	rc = cuCtxPushCurrent(cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuCtxPushCurrent: %s", errorText(rc));
		goto out;
	}

	/*
	 * Do GPU Buffer Synchronization
	 */
	pthreadRWLockReadLock(&gs_sstate->mmap_lock);
	pthreadRWLockWriteLock(&gs_sstate->gpu_bufer_lock);
	PG_TRY();
	{
		/* Lookup local mapping */
		gs_desc = (GpuStoreDesc *) hash_search(gstore_desc_htab,
											   gs_sstate,
											   HASH_ENTER,
											   &found);
		if (!found)
			gs_desc->gs_sstate = gs_sstate;
		else
			Assert(gs_desc->gs_sstate == gs_sstate);
		__gstoreFdwSetupGpuStoreDesc(gs_desc);
		if (gs_desc->gpu_main_devptr == 0UL)
		{
			GpuBufferUpdaterInitialLoad(gs_desc);
		}
		else
		{
			GpuBufferUpdaterApplyLogs(gs_desc);
		}
	}
	PG_CATCH();
	{
		pthreadRWLockUnlock(&gs_sstate->gpu_bufer_lock);
		pthreadRWLockUnlock(&gs_sstate->mmap_lock);
		FlushErrorState();
		gs_sstate->gpu_sync_status = CUDA_ERROR_MAP_FAILED;
	}
	PG_END_TRY();
	pthreadRWLockUnlock(&gs_sstate->gpu_bufer_lock);
	pthreadRWLockUnlock(&gs_sstate->mmap_lock);

	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
out:
	ConditionVariableBroadcast(&gstore_shared_head->gpumem_sync_cond);

	return retval;
}

/*
 * GstoreFdwStartupKicker
 */
void
GstoreFdwStartupKicker(Datum arg)
{
	const char *database_name;
	bool		first_launch = false;
	Relation	srel;
	SysScanDesc	sscan;
	ScanKeyData	skey;
	HeapTuple	tuple;
	int			exit_code = 1;

	BackgroundWorkerUnblockSignals();
	database_name = pstrdup(gstore_shared_head->kicker_next_database);
	if (*database_name == '\0')
	{
		database_name = "template1";
		first_launch = true;
	}
	BackgroundWorkerInitializeConnection(database_name, NULL, 0);

	StartTransactionCommand();
	PushActiveSnapshot(GetTransactionSnapshot());
	srel = table_open(DatabaseRelationId, AccessShareLock);
	ScanKeyInit(&skey,
				Anum_pg_database_datname,
				BTGreaterStrategyNumber, F_NAMEGT,
				CStringGetDatum(database_name));
	sscan = systable_beginscan(srel, DatabaseNameIndexId,
							   true,
							   NULL,
							   first_launch ? 0 : 1, &skey);
next_database:
	tuple = systable_getnext(sscan);
	if (HeapTupleIsValid(tuple))
	{
		Form_pg_database dat = (Form_pg_database) GETSTRUCT(tuple);
		if (!dat->datallowconn)
			goto next_database;
		strncpy(gstore_shared_head->kicker_next_database,
				NameStr(dat->datname),
				NAMEDATALEN);
	}
	else
	{
		/* current database is the last one */
		exit_code = 0;
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	if (!first_launch)
	{
		Form_pg_foreign_table ftable;
		Relation	frel;
		FdwRoutine *routine;

		srel = table_open(ForeignTableRelationId, AccessShareLock);
		sscan = systable_beginscan(srel, InvalidOid, false, NULL, 0, NULL);
		while ((tuple = systable_getnext(sscan)) != NULL)
		{
			ftable = (Form_pg_foreign_table)GETSTRUCT(tuple);
			frel = heap_open(ftable->ftrelid, AccessShareLock);
			routine = GetFdwRoutineForRelation(frel, false);

			if (memcmp(routine, &pgstrom_gstore_fdw_routine,
					   sizeof(FdwRoutine)) == 0)
			{
				gstoreFdwLookupGpuStoreDesc(frel);
				elog(LOG, "GstoreFdw preload '%s' in database '%s'",
					 RelationGetRelationName(frel), database_name);
			}
			heap_close(frel, AccessShareLock);
		}
		systable_endscan(sscan);
		table_close(srel, AccessShareLock);
	}
	
	PopActiveSnapshot();
	CommitTransactionCommand();
	proc_exit(exit_code);
}

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

	gstore_shared_head = ShmemInitStruct("GpuStore Shared Head",
										 sizeof(GpuStoreSharedHead),
										 &found);
	if (found)
		elog(ERROR, "Bug? GpuStoreSharedHead already exists");
	memset(gstore_shared_head, 0, sizeof(GpuStoreSharedHead));
	dlist_init(&gstore_shared_head->gpumem_sync_list);
	SpinLockInit(&gstore_shared_head->gpumem_sync_lock);
	ConditionVariableInit(&gstore_shared_head->gpumem_sync_cond);
	for (i=0; i < GPUSTORE_SHARED_DESC_NSLOTS; i++)
	{
		SpinLockInit(&gstore_shared_head->gstore_sstate_lock[i]);
		dlist_init(&gstore_shared_head->gstore_sstate_slot[i]);
	}
}

/*
 * pgstrom_init_gstore_fdw
 */
void
pgstrom_init_gstore_fdw(void)
{
	FdwRoutine	   *r = &pgstrom_gstore_fdw_routine;
	HASHCTL			hctl;
	BackgroundWorker worker;

	memset(r, 0, sizeof(FdwRoutine));
	NodeSetTag(r, T_FdwRoutine);
	/* SCAN support */
    r->GetForeignRelSize			= GstoreGetForeignRelSize;
    r->GetForeignPaths				= GstoreGetForeignPaths;
    r->GetForeignPlan				= GstoreGetForeignPlan;
	r->BeginForeignScan				= GstoreBeginForeignScan;
    r->IterateForeignScan			= GstoreIterateForeignScan;
    r->ReScanForeignScan			= GstoreReScanForeignScan;
    r->EndForeignScan				= GstoreEndForeignScan;

	/* Parallel support */
	r->IsForeignScanParallelSafe	= GstoreIsForeignScanParallelSafe;
    r->EstimateDSMForeignScan		= GstoreEstimateDSMForeignScan;
    r->InitializeDSMForeignScan		= GstoreInitializeDSMForeignScan;
    r->ReInitializeDSMForeignScan	= GstoreReInitializeDSMForeignScan;
    r->InitializeWorkerForeignScan	= GstoreInitializeWorkerForeignScan;
    r->ShutdownForeignScan			= GstoreShutdownForeignScan;

	/* UPDATE/INSERT/DELETE */
    r->AddForeignUpdateTargets		= GstoreAddForeignUpdateTargets;
    r->PlanForeignModify			= GstorePlanForeignModify;
    r->BeginForeignModify			= GstoreBeginForeignModify;
    r->ExecForeignInsert			= GstoreExecForeignInsert;
    r->ExecForeignUpdate			= GstoreExecForeignUpdate;
    r->ExecForeignDelete			= GstoreExecForeignDelete;
    r->EndForeignModify				= GstoreEndForeignModify;

	/* EXPLAIN/ANALYZE */
	r->ExplainForeignScan			= GstoreExplainForeignScan;
    r->ExplainForeignModify			= GstoreExplainForeignModify;
	r->AnalyzeForeignTable			= GstoreAnalyzeForeignTable;

	/*
	 * Local hash table for GpuStoreDesc
	 */
	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize	= 2 * sizeof(Oid);		/* database_oid + ftable_oid */
	hctl.entrysize	= sizeof(GpuStoreDesc);
	hctl.hcxt		= CacheMemoryContext;
	gstore_desc_htab = hash_create("GpuStoreDesc Hash-table", 256,
								   &hctl,
								   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	/* GUC: pg_strom.gstore_fdw_auto_preload  */
	DefineCustomBoolVariable("pg_strom.gstore_fdw_auto_preload",
							 "Enables auto preload of GstoreFdw GPU buffers",
							 NULL,
							 &pgstrom_gstore_fdw_auto_preload,
							 true,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/*
	 * Background worker to load GPU store on startup
	 */
	if (pgstrom_gstore_fdw_auto_preload)
	{
		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "GstoreFdw Starup Kicker");
		worker.bgw_flags = (BGWORKER_SHMEM_ACCESS |
							BGWORKER_BACKEND_DATABASE_CONNECTION);
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = 1;
		snprintf(worker.bgw_library_name, BGW_MAXLEN,
				 "$libdir/pg_strom");
		snprintf(worker.bgw_function_name, BGW_MAXLEN,
				 "GstoreFdwStartupKicker");
		worker.bgw_main_arg = 0;
		RegisterBackgroundWorker(&worker);
	}
	/* Request for the static shared memory */
	RequestAddinShmemSpace(STROMALIGN(sizeof(GpuStoreSharedHead)));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gstore_fdw;
}
