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


/*
 * GpuStoreBackgroundCommand
 */
#define GSTORE_BACKGROUND_CMD__INITIAL_LOAD		'I'
#define GSTORE_BACKGROUND_CMD__APPLY_REDO		'A'
#define GSTORE_BACKGROUND_CMD__COMPACTION		'C'
typedef struct
{
	dlist_node	chain;
	Oid			database_oid;
	Oid			ftable_oid;
	Latch	   *backend;		/* MyLatch of the backend, if any */
	int			command;		/* one of GSTORE_MAINTAIN_CMD__* */
	CUresult	retval;
	/* for APPLY_REDO */
	uint64		end_pos;
	uint32		nitems;
} GpuStoreBackgroundCommand;

/*
 * GpuStoreSharedHead
 */
#define GPUSTORE_SHARED_DESC_NSLOTS		107
typedef struct
{
	/* Database name to connect next */
	char		kicker_next_database[NAMEDATALEN];

	/* IPC to GpuStore Background Worker */
	Latch	   *background_latch;
	slock_t		background_cmd_lock;
	dlist_head	background_cmd_queue;
	dlist_head	background_free_cmds;
	GpuStoreBackgroundCommand __background_cmds[200];

	/* Sync object for Row-level Locks */
	ConditionVariable row_lock_cond;
	
	/* Hash slot for GpuStoreSharedState */
	slock_t		gstore_sstate_lock[GPUSTORE_SHARED_DESC_NSLOTS];
	dlist_head	gstore_sstate_slot[GPUSTORE_SHARED_DESC_NSLOTS];
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
#define GPUSTORE_EXTRABUF_SIGNATURE		"@EXTRA1@"

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

typedef struct
{
	dlist_node		hash_chain;
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
	const char	   *redo_log_backup_dir;
	size_t			redo_log_backup_limit;
	cl_long			gpu_update_interval;
	size_t			gpu_update_threshold;

#define GSTORE_NUM_BASE_ROW_LOCKS		4000
#define GSTORE_NUM_HASH_SLOT_LOCKS		5000
	/* Runtime state */
	LWLock			base_mmap_lock;
	uint32			base_mmap_revision;
	slock_t			rowid_map_lock;		/* lock for RowID-map */
	/* NOTE: hash_slot_lock must be acquired outside of the base_row_lock */
	slock_t			base_row_lock[GSTORE_NUM_BASE_ROW_LOCKS];
	slock_t			hash_slot_lock[GSTORE_NUM_HASH_SLOT_LOCKS];

	slock_t			redo_pos_lock;
	uint64			redo_write_nitems;
	uint64			redo_write_pos;
	uint64			redo_read_nitems;
	uint64			redo_read_pos;
	uint64			redo_sync_pos;
	uint64			redo_last_timestamp;	/* time when last command sent.
											 * not a timestamp redo-logs are
											 * applied on the GPU buffer. */
	/* Device data store */
	pthread_rwlock_t gpu_bufer_lock;
	CUipcMemHandle	gpu_main_mhandle;		/* mhandle to main portion */
	CUipcMemHandle	gpu_extra_mhandle;		/* mhandle to extra portion */
	size_t			gpu_main_size;
	size_t			gpu_extra_size;
} GpuStoreSharedState;

typedef struct
{
	Oid				database_oid;
	Oid				ftable_oid;
	GpuStoreSharedState *gs_sstate;
	/* base file mapping */
	GpuStoreBaseFileHead *base_mmap;
	uint32			base_mmap_revision;
	size_t			base_mmap_sz;
	int				base_mmap_is_pmem;
	GpuStoreRowIdMapHead *rowid_map;	/* RowID map section */
	GpuStoreHashIndexHead *hash_index;	/* Hash-index section (optional) */
	/* redo log mapping */
	char		   *redo_mmap;
	size_t			redo_mmap_sz;
	int				redo_mmap_is_pmem;
	/* fields below are valid only GpuStore Maintainer */
	int				redo_backup_fdesc;
	CUdeviceptr		gpu_main_devptr;
	CUdeviceptr		gpu_extra_devptr;
} GpuStoreDesc;

/*
 * GpuStoreFdwState - runtime state object for READ
 */
struct GpuStoreFdwState
{
	GpuStoreDesc   *gs_desc;

	pg_atomic_uint64  __read_pos;
	pg_atomic_uint64 *read_pos;

	Bitmapset	   *referenced;
	cl_bool			sysattr_refs;
	cl_uint			nitems;			/* kds->nitems on BeginScan */
	cl_bool			is_first;
	cl_uint			last_rowid;		/* last rowid returned */
	ExprState	   *indexExprState;
};

/*
 * GpuStoreUndoLogs 
 */
typedef struct
{
	dlist_node		chain;
	Oid				ftable_oid;
	TransactionId	xid_top;
	TransactionId	xid_sub;
	GpuStoreDesc   *gs_desc;
	cl_uint			nitems;
	StringInfoData	buf;
} GpuStoreUndoLogs;

/*
 * GpuStoreFdwModify - runtime state object for WRITE
 */
typedef struct
{
	GpuStoreDesc   *gs_desc;
	Bitmapset	   *updatedCols;
	cl_uint			next_rowid;
	TransactionId	oldestXmin;
	AttrNumber		ctid_attno;
	GpuStoreUndoLogs *gs_undo;
} GpuStoreFdwModify;

/* ---- static variables ---- */
static FdwRoutine	pgstrom_gstore_fdw_routine;
static GpuStoreSharedHead  *gstore_shared_head = NULL;
static HTAB		   *gstore_desc_htab = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static bool			gstore_fdw_enabled;		/* GUC */
#define GSTORE_UNDO_LOGS_NSLOTS			200
static dlist_head	gstore_undo_logs_slots[GSTORE_UNDO_LOGS_NSLOTS];

/* ---- Forward declarations ---- */
Datum pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_apply_redo(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_compaction(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_post_creation(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_sysattr_in(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_sysattr_out(PG_FUNCTION_ARGS);
void  GstoreFdwStartupKicker(Datum arg);
void  GstoreFdwMaintainerMain(Datum arg);

static bool		gstoreFdwRemapBaseFile(GpuStoreDesc *gs_desc,
									   bool abort_on_error);
static GpuStoreDesc *gstoreFdwLookupGpuStoreDesc(Relation frel);
static CUresult	gstoreFdwInvokeInitialLoad(Relation frel, bool is_async);
static CUresult gstoreFdwInvokeApplyRedo(Oid ftable_oid, bool is_async,
										 uint64 end_pos, uint32 nitems);
static CUresult gstoreFdwInvokeCompaction(Relation frel, bool is_async);
static size_t	gstoreFdwRowIdMapSize(cl_uint nrooms);
static cl_uint	gstoreFdwAllocateRowId(GpuStoreDesc *gs_desc,
									   cl_uint min_rowid);
static void		gstoreFdwReleaseRowId(GpuStoreDesc *gs_desc, cl_uint rowid);

static void		gstoreFdwInsertIntoPrimaryKey(GpuStoreUndoLogs *gs_undo,
											  cl_uint rowid);
static void		gstoreFdwRemoveFromPrimaryKey(GpuStoreUndoLogs *gs_undo,
											  cl_uint rowid);

/* ---- Atomic operations ---- */
static inline cl_ulong atomicRead64(volatile cl_ulong *ptr)
{
	return *ptr;
}

static inline cl_bool atomicCAS64(volatile cl_ulong *ptr,
								  cl_ulong *oldval, cl_ulong newval)
{
	return __atomic_compare_exchange_n(ptr, oldval, newval,
									   false,
									   __ATOMIC_SEQ_CST,
									   __ATOMIC_SEQ_CST);
}

static inline cl_ulong atomicMax64(volatile cl_ulong *ptr, cl_ulong newval)
{
	cl_ulong	oldval;

	do {
		oldval = atomicRead64(ptr);
		if (oldval >= newval)
			return oldval;
	} while (!atomicCAS64(ptr, &oldval, newval));

	return newval;
}

static inline cl_ulong atomicRead32(volatile cl_uint *ptr)
{
	return *ptr;
}

static inline cl_bool atomicCAS32(volatile cl_uint *ptr,
                                  cl_uint *oldval, cl_uint newval)
{
	return __atomic_compare_exchange_n(ptr, oldval, newval,
									   false,
									   __ATOMIC_SEQ_CST,
									   __ATOMIC_SEQ_CST);
}

static inline cl_uint atomicMax32(volatile cl_uint *ptr, cl_uint newval)
{
	cl_uint		oldval;

	do {
		oldval = atomicRead32(ptr);
		if (oldval >= newval)
			return oldval;
	} while (!atomicCAS32(ptr, &oldval, newval));

	return newval;
}

/* ---- Lock/unlock rows/hash-slot ---- */
static inline bool
gstoreFdwSpinLockBaseRow(GpuStoreDesc *gs_desc, cl_uint rowid)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	cl_uint		lindex = rowid % GSTORE_NUM_BASE_ROW_LOCKS;

	SpinLockAcquire(&gs_sstate->base_row_lock[lindex]);

	return true;
}

static inline bool
gstoreFdwSpinUnlockBaseRow(GpuStoreDesc *gs_desc, cl_uint rowid)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	cl_uint		lindex = rowid % GSTORE_NUM_BASE_ROW_LOCKS;

	SpinLockRelease(&gs_sstate->base_row_lock[lindex]);

	return false;
}

static inline bool
gstoreFdwSpinLockHashSlot(GpuStoreDesc *gs_desc, Datum hash)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	cl_uint		lindex = hash % GSTORE_NUM_HASH_SLOT_LOCKS;

	SpinLockAcquire(&gs_sstate->hash_slot_lock[lindex]);

	return true;
}

static inline bool
gstoreFdwSpinUnlockHashSlot(GpuStoreDesc *gs_desc, Datum hash)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	cl_uint		lindex = hash % GSTORE_NUM_HASH_SLOT_LOCKS;

	SpinLockRelease(&gs_sstate->hash_slot_lock[lindex]);

	return false;
}

/*
 * baseRelIsGstoreFdw
 */
bool
baseRelIsGstoreFdw(RelOptInfo *baserel)
{
	if ((baserel->reloptkind == RELOPT_BASEREL ||
		 baserel->reloptkind == RELOPT_OTHER_MEMBER_REL) &&
		baserel->rtekind == RTE_RELATION &&
		OidIsValid(baserel->serverid) &&
		baserel->fdwroutine &&
		memcmp(baserel->fdwroutine,
			   &pgstrom_gstore_fdw_routine,
			   sizeof(FdwRoutine)) == 0)
		return true;

	return false;
}

/*
 * RelationIsGstoreFdw
 */
bool
RelationIsGstoreFdw(Relation frel)
{
	if (RelationGetForm(frel)->relkind == RELKIND_FOREIGN_TABLE)
	{
		FdwRoutine *routine = GetFdwRoutineForRelation(frel, false);

		if (memcmp(routine, &pgstrom_gstore_fdw_routine,
				   sizeof(FdwRoutine)) == 0)
			return true;
	}
	return false;
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
	TypeCacheEntry *tcache;

	if (!IsA(op, OpExpr) || list_length(op->args) != 2)
		return false;	/* binary operator */

	left = (Node *) linitial(op->args);
	right = (Node *) lsecond(op->args);
	if (exprType(left) != exprType(right))
		return false;	/* type not compatible */
	tcache = lookup_type_cache(exprType(left),
							   TYPECACHE_EQ_OPR);
	if (tcache->eq_opr != op->opno)
		return false;	/* opno is not equal operator */

	if (IsA(left, RelabelType))
		left = (Node *)((RelabelType *)left)->arg;
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

/*
 * GetOptimalGpuForGstoreFdw
 */
int
GetOptimalGpuForGstoreFdw(PlannerInfo *root,
						  RelOptInfo *baserel)
{
	GpuStoreDesc   *gs_desc;
	
	if (!baserel->fdw_private)
	{
		RangeTblEntry *rte = root->simple_rte_array[baserel->relid];

		GstoreGetForeignRelSize(root, baserel, rte->relid);
	}
	gs_desc = baserel->fdw_private;
	return gs_desc->gs_sstate->cuda_dindex;
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
	Cost			run_cost = 0.0;
	double			ntuples = baserel->tuples;

	/* gstore_fdw.enabled */
	if (!gstore_fdw_enabled)
		startup_cost += disable_cost;

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
	run_cost += (cpu_tuple_cost + qual_cost.per_tuple) * ntuples;
	run_cost += baserel->reltarget->cost.per_tuple * baserel->rows;

	fpath = create_foreignscan_path(root,
									baserel,
									NULL,	/* default pathtarget */
									baserel->rows,
									startup_cost,
									startup_cost + run_cost,
									NIL,	/* no pathkeys */
									required_outer,
									NULL,	/* no extra plan */
									list_make1(indexExpr));
	add_path(baserel, (Path *)fpath);
	
	//TODO: parameterized paths
}

/*
 * gstoreCheckVisibilityForRead
 *
 * It checks visibility of row when reader tries to fetch it.
 * Caller must have the spinlock of the row.
 */
static bool
gstoreCheckVisibilityForRead(GpuStoreDesc *gs_desc, cl_uint rowid,
							 Snapshot snapshot,
							 GstoreFdwSysattr *p_sysattr)
{
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	kern_colmeta   *cmeta = &kds->colmeta[kds->ncols - 1];
	GstoreFdwSysattr *sysattr;
	Datum			datum;
	bool			isnull;

	Assert(kds->format == KDS_FORMAT_COLUMN &&
		   kds->ncols >= 1 &&
		   rowid < kds->nrooms);
	datum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
	Assert(!isnull);
	sysattr = (GstoreFdwSysattr *)DatumGetPointer(datum);
	if (p_sysattr)
		memcpy(p_sysattr, sysattr, sizeof(GstoreFdwSysattr));

	if (sysattr->xmin != FrozenTransactionId)
	{
		if (TransactionIdIsCurrentTransactionId(sysattr->xmin))
		{
			if (sysattr->cid >= snapshot->curcid)
				return false;	/* inserted after scan started */
			if (sysattr->xmax == InvalidTransactionId)
				return true;	/* not deleted yet */
			if (sysattr->xmax == FrozenTransactionId)
				return false;	/* deleted, and committed */
			if (!TransactionIdIsCurrentTransactionId(sysattr->xmax))
				return true;	/* deleted by others, but not committed */
		}
		else if (TransactionIdDidCommit(sysattr->xmin))
		{
			/* inserted, and already committed */
			sysattr->xmin = FrozenTransactionId;
		}
		else if (TransactionIdIsInProgress(sysattr->xmin))
		{
			/* inserted by other session, and not committed */
			return false;
		}
		else
		{
			/* it must be aborted, or crashed */
			sysattr->xmin = InvalidTransactionId;
			return false;
		}
	}
	/* by here, the inserting transaction has committed */
	if (sysattr->xmax == InvalidTransactionId)
		return true;			/* not deleted yet */
	if (sysattr->xmax != FrozenTransactionId)
	{
		if (TransactionIdIsCurrentTransactionId(sysattr->xmax))
		{
			if (sysattr->cid >= snapshot->curcid)
				return true;	/* deleted after scan started */
			else
				return false;	/* deleted before scan started */
		}
		else if (TransactionIdIsInProgress(sysattr->xmax))
		{
			/* removed by other transaction in-progress */
			return true;
		}
		else if (TransactionIdDidCommit(sysattr->xmax))
		{
			/* transaction that removed this row has been committed */
			sysattr->xmax = FrozenTransactionId;
		}
		else
		{
			/* deleter is aborted or crashed */
			sysattr->xmax = InvalidTransactionId;
		}
	}
	/* deleted, and transaction committed */
	return false;
}

/*
 * gstoreCheckVisibilityForInsert
 */
static bool
gstoreCheckVisibilityForInsert(GpuStoreDesc *gs_desc, cl_uint rowid,
							   TransactionId oldestXmin)
{
	/* see logic in HeapTupleSatisfiesVacuum */
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	kern_colmeta   *cmeta = &kds->colmeta[kds->ncols - 1];
	GstoreFdwSysattr *sysattr;
	Datum			datum;
	bool			isnull;	

	datum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
	Assert(!isnull);
	sysattr = (GstoreFdwSysattr *)DatumGetPointer(datum);

	if (sysattr->xmin != FrozenTransactionId)
	{
		if (sysattr->xmin == InvalidTransactionId)
			return true;	/* not written yet */
		else if (TransactionIdIsCurrentTransactionId(sysattr->xmin))
			return false;	/* already inserted by itself */
		else if (TransactionIdIsInProgress(sysattr->xmin))
			return false;	/* inserted by other, but not committed yet */
		else if (TransactionIdDidCommit(sysattr->xmin))
			sysattr->xmin = FrozenTransactionId;	/* already committed */
		else
		{
			/* not in progress, not committed, so either aborted or crashed */
			sysattr->xmin = InvalidTransactionId;
			return true;
		}
	}
	/* Ok, INSERT is commited */
	if (sysattr->xmax == InvalidTransactionId)
		return false;		/* row still lives */
	if (sysattr->xmax != FrozenTransactionId)
	{
		if (TransactionIdIsInProgress(sysattr->xmax))
			return false;	/* DELETE is in progress, so find another one */
		else if (TransactionIdDidCommit(sysattr->xmax))
		{
			/*
			 * Deleter committed, but perhaps it was recent enough that some
			 * open transactions could still see the tuple.
			 */
			if (!TransactionIdPrecedes(sysattr->xmax, oldestXmin))
				return false;
			sysattr->xmax = FrozenTransactionId;
		}
		else
		{
			/* not in progress, not committed, so either aborted or crashed */
			sysattr->xmax = InvalidTransactionId;
			return false;
		}
	}
	return true;	/* completely dead, and removable */
}

/*
 * gstoreCheckVisibilityTryDelete
 *
 * it checks visibility and mark 'deleted' if possible
 */
static int
__gstoreCheckVisibilityTryDelete(GpuStoreDesc *gs_desc, cl_uint rowid,
								 TransactionId curr_xid,
								 GstoreFdwSysattr *sysattr)
{
	if (sysattr->xmin != FrozenTransactionId)
	{
		if (sysattr->xmin == InvalidTransactionId)
		{
			elog(WARNING, "tried to update/delete xmin=Invalid at rowid=%u", rowid);
			return 0;		/* not written */
		}
		else if (TransactionIdIsCurrentTransactionId(sysattr->xmin))
		{
			if (sysattr->xmax == InvalidTransactionId)
			{
				sysattr->xmax = curr_xid;
				sysattr->cid  = GetCurrentCommandId(true);
				return 1;
			}
			else if (TransactionIdIsCurrentTransactionId(sysattr->xmax))
			{
				elog(WARNING, "tried to update/remove a row already removed by itself at rowid=%u", rowid);
				return 0;
			}
			else
			{
				/* not in progress, not committed, so either aborted or crashed */
				Assert(sysattr->xmax != FrozenTransactionId &&
					   !TransactionIdDidCommit(sysattr->xmax));
				sysattr->xmax = curr_xid;
				sysattr->cid = GetCurrentCommandId(true);
				return 1;
			}
		}
		else if (TransactionIdIsInProgress(sysattr->xmin))
		{
			return -1;		/* inserted by others, but not committed yet */
		}
		else if (TransactionIdDidCommit(sysattr->xmin))
		{
			sysattr->xmin = FrozenTransactionId;
		}
		else
		{
			/* not in progress, not committed, so either aborted or crashed */
			sysattr->xmin = InvalidTransactionId;
			return 0;
		}
	}
	/* Ok, the row is committed */
	if (sysattr->xmax == InvalidTransactionId)
	{
		sysattr->xmax = curr_xid;
		sysattr->cid = GetCurrentCommandId(true);
		return 1;
	}
	if (sysattr->xmax != FrozenTransactionId)
	{
		if (TransactionIdIsCurrentTransactionId(sysattr->xmax))
		{
			elog(WARNING, "tried to update/delete a row already removed by itself, at rowid=%u", rowid);
			return 0;
		}
		else if (TransactionIdIsInProgress(sysattr->xmax))
		{
			/* concurrent transaction removed, but not committed yet */
			return -1;
		}
		else if (!TransactionIdDidCommit(sysattr->xmax))
		{
			/* not in progress, not committed, so either aborted or crashed */
			sysattr->xmax = curr_xid;
			sysattr->cid = GetCurrentCommandId(true);
			return 1;
		}
	}
	return 0;		/* already removed */
}

static int
gstoreCheckVisibilityTryDelete(GpuStoreDesc *gs_desc, cl_uint rowid,
                               TransactionId curr_xid,
                               GstoreFdwSysattr *p_sysattr)
{
    /* see logic in HeapTupleSatisfiesVacuum */
    kern_data_store *kds = &gs_desc->base_mmap->schema;
    kern_colmeta   *cmeta = &kds->colmeta[kds->ncols - 1];
    GstoreFdwSysattr *sysattr;
    Datum           datum;
    bool            isnull;
	int				rv;

	datum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
    Assert(!isnull);
    sysattr = (GstoreFdwSysattr *)DatumGetPointer(datum);
	rv = __gstoreCheckVisibilityTryDelete(gs_desc, rowid, curr_xid, sysattr);
	if (p_sysattr)
		memcpy(p_sysattr, sysattr, sizeof(GstoreFdwSysattr));
	return rv;
}

/*
 * gstoreCheckVisibilityForIndex
 *
 * It checks row's visibility on the primary-key duplication checks.
 * Caller must not hold any row-spinlock.
 * It returns positive (1) for visible, zero for invisible, or -1 for pending to
 * update/delete. If -1 is returned, caller must wait for commit/abort of the
 * blocker transaction, then retry violation checks.
 */
static int
__gstoreCheckVisibilityForIndexNoLock(GpuStoreDesc *gs_desc, cl_uint rowid)
{
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	kern_colmeta   *cmeta = &kds->colmeta[kds->ncols - 1];
	GstoreFdwSysattr *sysattr;
	Datum			datum;
	bool			isnull;	

	datum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
	Assert(!isnull);
	sysattr = (GstoreFdwSysattr *)DatumGetPointer(datum);

	if (sysattr->xmin != FrozenTransactionId)
	{
		if (sysattr->xmin == InvalidTransactionId)
		{
			/* row is not written yet, but on the index. corruption? */
			elog(WARNING, "index corruption? rowid=%u has xmin=Invalid", rowid);
			return 0;
		}
		else if (TransactionIdIsCurrentTransactionId(sysattr->xmin))
		{
			if (sysattr->xmax == InvalidTransactionId)
				return 1;	/* not deleted yet */
			if (sysattr->xmax == FrozenTransactionId)
				return 0;	/* deleted, and committed */
			if (!TransactionIdIsCurrentTransactionId(sysattr->xmax))
				return -1;	/* deleted in other transaction, but not commited.
							 * so, caller shall be blocked by row-lock. */
		}
		else if (TransactionIdIsInProgress(sysattr->xmin))
		{
			/* inserted by the concurrent transaction; must be blocked */
			return -1;
		}
		else if (!TransactionIdDidCommit(sysattr->xmin))
		{
			/* not in progress, not committed, so either aborted or crashed */
			return 0;
		}
	}
	/* Ok, row is inserted and committed */
	if (sysattr->xmax == InvalidTransactionId)
		return 1;		/* row still lives */
	if (sysattr->xmax != FrozenTransactionId)
	{
		if (TransactionIdIsCurrentTransactionId(sysattr->xmax))
		{
			/*
			 * deleted by the current transaction. it should be invisible
			 * for primary key checks, regardless of the command-id.
			 */
			return 0;
		}
		else if (TransactionIdIsInProgress(sysattr->xmax))
		{
			/* removed by other transaction, but not committed yet */
			return -1;
		}
		else if (TransactionIdDidCommit(sysattr->xmax))
		{
			/* deleter is already committed */
			return 0;
		}
		else
		{
			/* deleter is aborted or crashed */
			return 1;
		}
	}
	return 0;	/* completely dead, and removable */
}

static int
gstoreCheckVisibilityForIndex(GpuStoreDesc *gs_desc, cl_uint rowid)
{
	int		retval;

	gstoreFdwSpinLockBaseRow(gs_desc, rowid);
	PG_TRY();
	{
		retval = __gstoreCheckVisibilityForIndexNoLock(gs_desc, rowid);
	}
	PG_CATCH();
	{
		gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);
		PG_RE_THROW();
	}
	PG_END_TRY();
	gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);

	return retval;
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
	Bitmapset  *referenced = NULL;
	List	   *outer_refs = NIL;
	int			i, j, k;

	scan_clauses = extract_actual_clauses(scan_clauses, false);
	pull_varattnos((Node *)scan_clauses, baserel->relid, &referenced);
	for (i=baserel->min_attr, j=0; i <= baserel->max_attr; i++, j++)
	{
		k = i - FirstLowInvalidHeapAttributeNumber;
		if (!bms_is_empty(baserel->attr_needed[j]))
			referenced = bms_add_member(referenced, k);
	}

	for (j = bms_next_member(referenced, -1);
		 j >= 0;
		 j = bms_next_member(referenced, j))
	{
		k = j + FirstLowInvalidHeapAttributeNumber;
		outer_refs = lappend_int(outer_refs, k);
	}

	return make_foreignscan(tlist,
							scan_clauses,
							baserel->relid,
							best_path->fdw_private,	/* indexExpr */
							outer_refs,	/* referenced attnums */
							NIL,		/* no custom tlist */
							NIL,		/* no remote quals */
							outer_plan);
}

static CUresult
gstoreFdwApplyRedoDeviceBuffer(Relation frel, GpuStoreSharedState *gs_sstate)
{
	cl_uint		nitems;
	size_t		curr_pos;

	/* see  __gstoreFdwAppendRedoLog */
	SpinLockAcquire(&gs_sstate->redo_pos_lock);
	curr_pos = gs_sstate->redo_write_pos;
	nitems = (gs_sstate->redo_write_nitems -
			  gs_sstate->redo_read_nitems);
	SpinLockRelease(&gs_sstate->redo_pos_lock);

	return gstoreFdwInvokeApplyRedo(RelationGetRelid(frel), false,
									curr_pos, nitems);
}

static GpuStoreFdwState *
__ExecInitGstoreFdw(ScanState *ss, Bitmapset *outer_refs, Expr *indexExpr,
					bool apply_redo_log)
{
	Relation		frel = ss->ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	GpuStoreFdwState *fdw_state = palloc0(sizeof(GpuStoreFdwState));
	GpuStoreDesc   *gs_desc = gstoreFdwLookupGpuStoreDesc(frel);
	Bitmapset	   *referenced = NULL;
	int				j, k;

	for (j = bms_next_member(outer_refs, -1);
		 j >= 0;
		 j = bms_next_member(outer_refs, j))
	{
		int		anum = j + FirstLowInvalidHeapAttributeNumber;

		if (anum < 0)
		{
#if PG_VERSION_NUM >= 120000
			/*
			 * PG12 adds tts_tid to carry ItemPointer without forming
			 * a HeapTuple; an optimization of TupleTableSlot.
			 */
			if (anum != SelfItemPointerAttributeNumber)
				fdw_state->sysattr_refs = true;
#else
			fdw_state->sysattr_refs = true;
#endif
		}
		else if (anum == 0)
		{
			for (k=0; k < tupdesc->natts; k++)
			{
				Form_pg_attribute attr = tupleDescAttr(tupdesc,k);

				if (attr->attisdropped)
					continue;
				referenced = bms_add_member(referenced, attr->attnum -
											FirstLowInvalidHeapAttributeNumber);
			}
		}
		else
		{
			referenced = bms_add_member(referenced, anum -
										FirstLowInvalidHeapAttributeNumber);
        }
	}
	
	/* setup GpuStoreExecState */
	fdw_state->gs_desc = gs_desc;
	pg_atomic_init_u64(&fdw_state->__read_pos, 0);
	fdw_state->read_pos = &fdw_state->__read_pos;
	fdw_state->nitems = gs_desc->base_mmap->schema.nitems;
	fdw_state->last_rowid = UINT_MAX;
	fdw_state->is_first = true;
	fdw_state->referenced = referenced;
	if (indexExpr != NULL)
		fdw_state->indexExprState = ExecInitExpr(indexExpr, &ss->ps);

	/* synchronize device buffer prior to the kernel call */
	if (apply_redo_log)
		gstoreFdwApplyRedoDeviceBuffer(frel, gs_desc->gs_sstate);

	return fdw_state;
}

GpuStoreFdwState *
ExecInitGstoreFdw(ScanState *ss, int eflags,
				  Bitmapset *outer_refs)
{
	bool	apply_redo_log = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0);

	return __ExecInitGstoreFdw(ss, outer_refs, NULL, apply_redo_log);
}

static void
GstoreBeginForeignScan(ForeignScanState *node, int eflags)
{
	ForeignScan	   *fscan = (ForeignScan *)node->ss.ps.plan;
	Bitmapset	   *outer_refs = NULL;
	Expr		   *indexExpr = NULL;
	ListCell	   *lc;

	foreach (lc, fscan->fdw_private)
	{
		int		anum = lfirst_int(lc);

		outer_refs = bms_add_member(outer_refs, anum -
									FirstLowInvalidHeapAttributeNumber);
	}

	if (fscan->fdw_exprs != NIL)
		indexExpr = linitial(fscan->fdw_exprs);

	node->fdw_state = __ExecInitGstoreFdw(&node->ss, outer_refs, indexExpr, false);
}

/*
 * ExecScanChunkGstoreFdw
 */
pgstrom_data_store *
ExecScanChunkGstoreFdw(GpuTaskState *gts)
{
	Relation		frel = gts->css.ss.ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	GpuStoreFdwState *fdw_state = gts->gs_state;
	pgstrom_data_store *pds = NULL;

	if (pg_atomic_fetch_add_u64(fdw_state->read_pos, 1) == 0)
	{
		EState		   *estate = gts->css.ss.ps.state;
		GpuStoreDesc   *gs_desc = fdw_state->gs_desc;
		GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
		size_t			sz = KDS_calculateHeadSize(tupdesc);

		pds = MemoryContextAllocZero(estate->es_query_cxt,
									 offsetof(pgstrom_data_store, kds) + sz);
		pg_atomic_init_u32(&pds->refcnt, 1);
		pds->gs_sstate = gs_sstate;
		init_kernel_data_store(&pds->kds, tupdesc, sz, KDS_FORMAT_COLUMN, 0);
	}
	return pds;
}

/*
 * gstoreFdwMapDeviceMemory
 */
CUresult
gstoreFdwMapDeviceMemory(GpuContext *gcontext, pgstrom_data_store *pds)
{
	GpuStoreSharedState *gs_sstate = pds->gs_sstate;
	CUdeviceptr	m_kds_base = 0UL;
	CUdeviceptr	m_kds_extra = 0UL;
	CUresult	rc;

	Assert(pds->kds.format == KDS_FORMAT_COLUMN);
	pthreadRWLockReadLock(&gs_sstate->gpu_bufer_lock);
	if (gs_sstate->gpu_main_size > 0)
	{
		rc = gpuIpcOpenMemHandle(gcontext,
								 &m_kds_base,
								 gs_sstate->gpu_main_mhandle,
								 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
			goto error_1;
	}

	if (gs_sstate->gpu_extra_size > 0)
	{
		rc = gpuIpcOpenMemHandle(gcontext,
								 &m_kds_extra,
								 gs_sstate->gpu_extra_mhandle,
								 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
			goto error_2;
	}
	pds->m_kds_base = m_kds_base;
	pds->m_kds_extra = m_kds_extra;

	return CUDA_SUCCESS;

error_2:
	gpuIpcCloseMemHandle(gcontext, m_kds_base);
error_1:
	pthreadRWLockUnlock(&gs_sstate->gpu_bufer_lock);
	return rc;
}

void
gstoreFdwUnmapDeviceMemory(GpuContext *gcontext, pgstrom_data_store *pds)
{
	GpuStoreSharedState *gs_sstate = pds->gs_sstate;

	Assert(pds->kds.format == KDS_FORMAT_COLUMN);
	if (pds->m_kds_base != 0UL)
	{
		gpuIpcCloseMemHandle(gcontext, pds->m_kds_base);
		pds->m_kds_base = 0UL;
	}
	if (pds->m_kds_extra != 0UL)
	{
		gpuIpcCloseMemHandle(gcontext, pds->m_kds_extra);
		pds->m_kds_extra = 0UL;
	}
	pthreadRWLockUnlock(&gs_sstate->gpu_bufer_lock);
}

/*
 * GstoreIterateForeignScan
 */
static TupleTableSlot *
__gstoreFillupTupleTableSlot(TupleTableSlot *slot,
							 GpuStoreDesc *gs_desc,
							 cl_uint rowid,
							 GstoreFdwSysattr *sysattr,
							 GpuStoreFdwState *fdw_state)
{
	TupleDesc	tupdesc = slot->tts_tupleDescriptor;
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	Datum		datum;
	bool		isnull;
	int			j;

	/* user defined attributes are immutable out of the row-spinlock */
	Assert(tupdesc->natts == kds->ncols - 1);
	for (j=0; j < tupdesc->natts; j++)
	{
		int		k = j + 1 - FirstLowInvalidHeapAttributeNumber;

		if (!bms_is_member(k, fdw_state->referenced))
		{
			slot->tts_values[j] = 0;
			slot->tts_isnull[j] = true;
		}
		else
		{
			kern_colmeta *cmeta = &kds->colmeta[j];
			
			datum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
			slot->tts_values[j] = datum;
			slot->tts_isnull[j] = isnull;
		}
	}
	ExecStoreVirtualTuple(slot);
	/*
	 * PG12 modified the definition of TupleTableSlot; it allows to carry
	 * ctid (32bit row-id in Gstore_Fdw) in the virtual-tuple form.
	 * 'sysattr_refs' is only set if any system-columns are referenced,
	 * except for the SelfItemPointer in PG12.
	 */
#if PG_VERSION_NUM >= 120000
	slot->tts_tid.ip_blkid.bi_hi = (rowid >> 16);
	slot->tts_tid.ip_blkid.bi_lo = (rowid & 0x0000ffffU);
	slot->tts_tid.ip_posid       = 0;
#endif
	if (fdw_state->sysattr_refs)
	{
		HeapTuple	tuple = heap_form_tuple(tupdesc,
											slot->tts_values,
											slot->tts_isnull);
		HeapTupleHeaderSetXmin(tuple->t_data, sysattr->xmin);
		HeapTupleHeaderSetXmax(tuple->t_data, sysattr->xmax);
		HeapTupleHeaderSetCmin(tuple->t_data, sysattr->cid);
		tuple->t_self.ip_blkid.bi_hi = (rowid >> 16);
		tuple->t_self.ip_blkid.bi_lo = (rowid & 0x0000ffff);
		tuple->t_self.ip_posid       = 0;

		ExecForceStoreHeapTuple(tuple, slot, false);
	}
	return slot;
}

static TupleTableSlot *
__gstoreIterateForeignIndexScan(ForeignScanState *node)
{
	Relation		frel = node->ss.ss_currentRelation;
	EState		   *estate = node->ss.ps.state;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	GpuStoreFdwState *fdw_state = node->fdw_state;
	GpuStoreDesc   *gs_desc = fdw_state->gs_desc;
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	GpuStoreHashIndexHead *hash_index = gs_desc->hash_index;
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	kern_colmeta *cmeta = &kds->colmeta[gs_sstate->primary_key - 1];
	cl_uint		   *rowmap;
	TypeCacheEntry *tcache;
	cl_uint			rowid = UINT_MAX;
	Datum			kdatum;
	Datum			vdatum;
	Datum			hash;
	bool			isnull;
	GstoreFdwSysattr sysattr;

	Assert(hash_index != NULL &&
		   gs_sstate->primary_key > 0 &&
		   gs_sstate->primary_key <= RelationGetNumberOfAttributes(frel));
	if (pg_atomic_fetch_add_u64(fdw_state->read_pos, 1) > 0)
		return NULL;
	rowmap = &hash_index->slots[hash_index->nslots];
		
	/* extract the key value */
	kdatum = ExecEvalExpr(fdw_state->indexExprState,
						  node->ss.ps.ps_ExprContext,
						  &isnull);
	if (isnull)
		return NULL;
	/* calculation of hash-value */
	tcache = lookup_type_cache(cmeta->atttypid,
							   TYPECACHE_HASH_PROC_FINFO |
							   TYPECACHE_EQ_OPR_FINFO);
	hash = FunctionCall1(&tcache->hash_proc_finfo, kdatum);

	/* walk on the hash-table */
	gstoreFdwSpinLockHashSlot(gs_desc, hash);
	PG_TRY();
	{
		GstoreFdwSysattr __sysattr;
		cl_uint		curr_id;

		curr_id = hash_index->slots[hash % hash_index->nslots];
		while (curr_id < hash_index->nrooms)
		{
			gstoreFdwSpinLockBaseRow(gs_desc, curr_id);
			PG_TRY();
			{
				if (gstoreCheckVisibilityForRead(gs_desc, curr_id,
												 estate->es_snapshot,
												 &__sysattr))
				{
					vdatum = KDS_fetch_datum_column(kds, cmeta,
													curr_id,
													&isnull);
					if (!isnull && FunctionCall2(&tcache->eq_opr_finfo,
												 kdatum, vdatum))
					{
						if (rowid != UINT_MAX)
							elog(ERROR, "index corruption? duplicate primary key");
						rowid = curr_id;
						sysattr = __sysattr;
					}
				}
			}
			PG_CATCH();
			{
				gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);
				PG_RE_THROW();
			}
			PG_END_TRY();
			gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);

			curr_id = rowmap[curr_id];
		}
	}
	PG_CATCH();
	{
		gstoreFdwSpinUnlockHashSlot(gs_desc, hash);
		PG_RE_THROW();
	}
	PG_END_TRY();
	gstoreFdwSpinUnlockHashSlot(gs_desc, hash);

	if (rowid == UINT_MAX)
		return NULL;	/* not found */

	return __gstoreFillupTupleTableSlot(slot, gs_desc,
										rowid, &sysattr,
										fdw_state);
}

static TupleTableSlot *
__gstoreIterateForeignSeqScan(ForeignScanState *node)
{
	Relation		frel = node->ss.ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	EState		   *estate = node->ss.ps.state;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	GpuStoreFdwState *fdw_state = node->fdw_state;
	GpuStoreDesc   *gs_desc = fdw_state->gs_desc;
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	cl_uint			rowid;
	bool			visible;
	GstoreFdwSysattr sysattr;

	Assert(tupdesc->natts + 1 == kds->ncols);
	do {
		rowid = pg_atomic_fetch_add_u64(fdw_state->read_pos, 1);
		if (rowid >= fdw_state->nitems)
			return NULL;
		gstoreFdwSpinLockBaseRow(gs_desc, rowid);
		visible = gstoreCheckVisibilityForRead(gs_desc, rowid,
											   estate->es_snapshot,
											   &sysattr);
		gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);
	} while (!visible);

	return __gstoreFillupTupleTableSlot(slot, gs_desc,
										rowid, &sysattr,
										fdw_state);
}

static TupleTableSlot *
GstoreIterateForeignScan(ForeignScanState *node)
{
	GpuStoreFdwState *fdw_state = node->fdw_state;

	ExecClearTuple(node->ss.ss_ScanTupleSlot);
	if (fdw_state->indexExprState)
		return __gstoreIterateForeignIndexScan(node);
	else
		return __gstoreIterateForeignSeqScan(node);
}

void
ExecReScanGstoreFdw(GpuStoreFdwState *fdw_state)
{
	pg_atomic_write_u64(fdw_state->read_pos, 0);
}

static void
GstoreReScanForeignScan(ForeignScanState *node)
{
	ExecReScanGstoreFdw((GpuStoreFdwState *)node->fdw_state);
}

void
ExecEndGstoreFdw(GpuStoreFdwState *fdw_state)
{
	/* nothing to do */
}

static void
GstoreEndForeignScan(ForeignScanState *node)
{
	ExecEndGstoreFdw((GpuStoreFdwState *)node->fdw_state);
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
	return MAXALIGN(sizeof(pg_atomic_uint64));
}

void
ExecInitDSMGstoreFdw(GpuStoreFdwState *fdw_state,
					 pg_atomic_uint64 *gstore_read_pos)
{
	pg_atomic_write_u64(gstore_read_pos, 0);
	fdw_state->read_pos = gstore_read_pos;
}

static void
GstoreInitializeDSMForeignScan(ForeignScanState *node,
							   ParallelContext *pcxt,
							   void *coordinate)
{
	GpuStoreFdwState *fdw_state = node->fdw_state;
	
	ExecInitDSMGstoreFdw(fdw_state, (pg_atomic_uint64 *)coordinate);
}

void
ExecReInitDSMGstoreFdw(GpuStoreFdwState *fdw_state)
{
	pg_atomic_write_u64(fdw_state->read_pos, 0);
}

static void
GstoreReInitializeDSMForeignScan(ForeignScanState *node,
								 ParallelContext *pcxt,
								 void *coordinate)
{
	GpuStoreFdwState   *fdw_state = node->fdw_state;

	Assert(fdw_state->read_pos == coordinate);
	ExecReInitDSMGstoreFdw(fdw_state);
}

void
ExecInitWorkerGstoreFdw(GpuStoreFdwState *fdw_state,
						pg_atomic_uint64 *gstore_read_pos)
{
	fdw_state->read_pos = gstore_read_pos;
}

static void
GstoreInitializeWorkerForeignScan(ForeignScanState *node,
								  shm_toc *toc,
								  void *coordinate)
{
	GpuStoreFdwState *fdw_state = node->fdw_state;
	ExecInitWorkerGstoreFdw(fdw_state, (pg_atomic_uint64 *)coordinate);
}

void
ExecShutdownGstoreFdw(GpuStoreFdwState *fdw_state)
{
	/* nothing to do */
}

static void
GstoreShutdownForeignScan(ForeignScanState *node)
{
	ExecShutdownGstoreFdw((GpuStoreFdwState *)node->fdw_state);
}

static void
__gstoreFdwAppendRedoLog(GpuStoreDesc *gs_desc,
						 GstoreTxLogCommon *tx_log)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	size_t		required = tx_log->length + sizeof(uint32);
	uint32		terminator = GSTORE_TX_LOG__TERMINATOR;
	bool		has_base_mmap_lock = false;

	Assert(tx_log->length == MAXALIGN(tx_log->length));
	if (required > gs_sstate->redo_log_limit)
		elog(ERROR, "gstore_fdw: length of REDO must be larger than 'redo_log_limit'");
	for (;;)
	{
		char	   *dest_ptr;
		size_t		dest_pos;
		size_t		len;

		SpinLockAcquire(&gs_sstate->redo_pos_lock);
		Assert(gs_sstate->redo_write_pos >= gs_sstate->redo_read_pos);
		/*
		 * If this redo-log makes the write position rewound, we must ensure
		 * the base_mmap is up-to-date prior to the overwrites of redo-log
		 * buffer.
		 */
		dest_pos = gs_sstate->redo_write_pos % gs_sstate->redo_log_limit;
		if (dest_pos + required > gs_sstate->redo_log_limit)
		{
			if (!has_base_mmap_lock)
			{
				SpinLockRelease(&gs_sstate->redo_pos_lock);

				LWLockAcquire(&gs_sstate->base_mmap_lock, LW_SHARED);
				if (gs_desc->base_mmap_revision != gs_sstate->base_mmap_revision)
					gstoreFdwRemapBaseFile(gs_desc, true);
				has_base_mmap_lock = true;
				continue;
			}
			len = gs_sstate->redo_log_limit - dest_pos;
			memset(gs_desc->redo_mmap + dest_pos, 0, len);
			gs_sstate->redo_write_pos += len;

			/* checkpoint of the base file */
			if (gs_desc->base_mmap_is_pmem)
				pmem_persist(gs_desc->base_mmap,
							 gs_desc->base_mmap_sz);
			else if (pmem_msync(gs_desc->base_mmap,
								gs_desc->base_mmap_sz) != 0)
			{
				elog(WARNING, "failed on pmem_msync('%s'): %m", gs_sstate->base_file);
			}
			//CheckPointWarning
			elog(LOG, "gstore_fdw: checkpoint applied on '%s'", gs_sstate->base_file);
			dest_pos = 0;
		}

		len = gs_sstate->redo_write_pos - gs_sstate->redo_read_pos;
		if (len + required > gs_sstate->redo_log_limit)
		{
			/*
			 * We have no space to write out redo-log any more, so we must
			 * kick the background worker to apply redo-log, and wait for
			 * completion of the task.
			 * If redo_last_timestamp is very recently updated, concurrent
			 * session might kick APPLY_REDO command. In this case, we just
			 * sleep for a moment, and recheck it.
			 */
			uint64	curr_timestamp = GetCurrentTimestamp();
			uint64	last_timestamp = gs_sstate->redo_last_timestamp;

			if (curr_timestamp < last_timestamp + 20000)	/* 20ms margin */
			{
				SpinLockRelease(&gs_sstate->redo_pos_lock);
				pg_usleep(4000L);	/* wait for 4ms */
			}
			else
			{
				size_t		curr_pos = gs_sstate->redo_read_pos;
				cl_uint		nitems = (gs_sstate->redo_write_nitems -
									  gs_sstate->redo_read_nitems);
				gs_sstate->redo_last_timestamp = curr_timestamp;
				SpinLockRelease(&gs_sstate->redo_pos_lock);
				gstoreFdwInvokeApplyRedo(gs_desc->ftable_oid, false, curr_pos, nitems);
			}
			continue;
		}
		/* Ok, we have enough space to write out redo-log buffer */
		if (len > gs_sstate->gpu_update_threshold)
		{
			uint64	curr_timestamp = GetCurrentTimestamp();
			uint64	last_timestamp = gs_sstate->redo_last_timestamp;

			if (curr_timestamp >= last_timestamp + 20000)	/* 20ms margin */
			{
				/*
				 * Even though redo-log buffer has space to write out, amount of
				 * logs not applied yet exceeds the threshold to kick the baclground
				 * worker, but to be asynchronous.
				 */
				size_t		curr_pos = gs_sstate->redo_read_pos;
				cl_uint		nitems = (gs_sstate->redo_write_nitems -
									  gs_sstate->redo_read_nitems);
				gs_sstate->redo_last_timestamp = curr_timestamp;
				gstoreFdwInvokeApplyRedo(gs_desc->ftable_oid, true, curr_pos, nitems);
			}
		}
		dest_ptr = gs_desc->redo_mmap + dest_pos;

		memcpy(dest_ptr, tx_log, tx_log->length);
		memcpy(dest_ptr + tx_log->length, &terminator, sizeof(uint32));
		gs_sstate->redo_write_pos += tx_log->length;
		gs_sstate->redo_write_nitems++;
		SpinLockRelease(&gs_sstate->redo_pos_lock);

		break;
	}
	if (has_base_mmap_lock)
		LWLockRelease(&gs_sstate->base_mmap_lock);
}

static void
gstoreFdwAppendInsertLog(Relation frel,
						 GpuStoreDesc *gs_desc,
						 cl_uint rowid,
						 cl_uint oldid,		/* UINT_MAX, if INSERT */
						 TransactionId xmin,
						 HeapTuple tuple)
{
	GstoreTxLogInsert *i_log;
	size_t		sz;

	sz = MAXALIGN(offsetof(GstoreTxLogInsert, htup) + tuple->t_len);
	i_log = alloca(sz + sizeof(cl_uint));
	memset(i_log, 0, sz + sizeof(cl_uint));
	i_log->type = GSTORE_TX_LOG__INSERT;
	i_log->length = sz;
	i_log->timestamp = GetCurrentTimestamp();
	i_log->rowid = rowid;
	memcpy(&i_log->htup, tuple->t_data, tuple->t_len);
	HeapTupleHeaderSetXmin(&i_log->htup, xmin);
	HeapTupleHeaderSetXmax(&i_log->htup, InvalidTransactionId);
	HeapTupleHeaderSetCmin(&i_log->htup, InvalidCommandId);
	i_log->htup.t_ctid.ip_blkid.bi_hi = (oldid >> 16);
	i_log->htup.t_ctid.ip_blkid.bi_lo = (oldid & 0x0000ffffU);
	i_log->htup.t_ctid.ip_posid = 0;

	__gstoreFdwAppendRedoLog(gs_desc, (GstoreTxLogCommon *)i_log);
}

static void
gstoreFdwAppendDeleteLog(Relation frel,
						 GpuStoreDesc *gs_desc,
						 cl_uint rowid,
						 TransactionId xmin,
						 TransactionId xmax)
{
	GstoreTxLogDelete d_log;

	memset(&d_log, 0, sizeof(GstoreTxLogDelete));
	d_log.type = GSTORE_TX_LOG__DELETE;
	d_log.length = MAXALIGN(sizeof(GstoreTxLogDelete));
	d_log.timestamp = GetCurrentTimestamp();
	d_log.rowid = rowid;
	d_log.xmin = xmin;
	d_log.xmax = xmax;

	__gstoreFdwAppendRedoLog(gs_desc, (GstoreTxLogCommon *)&d_log);
}

static GpuStoreUndoLogs *
gstoreFdwLookupUndoLogs(GpuStoreDesc *gs_desc)
{
	TransactionId	xid_top = GetTopTransactionId();
	TransactionId	xid_sub = GetCurrentTransactionId();
	int				hindex = xid_top % GSTORE_UNDO_LOGS_NSLOTS;
	dlist_iter		iter;
	GpuStoreUndoLogs *gs_undo;
	MemoryContext	oldcxt;

	dlist_foreach(iter, &gstore_undo_logs_slots[hindex])
	{
		gs_undo = dlist_container(GpuStoreUndoLogs, chain, iter.cur);

		if (gs_undo->xid_top == xid_top &&
			gs_undo->xid_sub == xid_sub &&
			gs_undo->ftable_oid == gs_desc->ftable_oid)
			return gs_undo;
	}
	/* construct a new one */
	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);
	gs_undo = palloc0(sizeof(GpuStoreUndoLogs));
	gs_undo->ftable_oid = gs_desc->ftable_oid;
	gs_undo->xid_top = xid_top;
	gs_undo->xid_sub = xid_sub;
	gs_undo->gs_desc = gs_desc;
	gs_undo->nitems = 0;
	initStringInfo(&gs_undo->buf);
	MemoryContextSwitchTo(oldcxt);

	dlist_push_tail(&gstore_undo_logs_slots[hindex],
					&gs_undo->chain);
	return gs_undo;
}

static void
GstoreAddForeignUpdateTargets(Query *parsetree,
							  RangeTblEntry *target_rte,
							  Relation target_relation)
{
	TargetEntry *tle;
	Var	   *var;

	/*
	 * Gstore_Fdw carries rowid, using ctid system column
	 */
	var = makeVar(parsetree->resultRelation,
				  SelfItemPointerAttributeNumber,
				  TIDOID,
				  -1,
				  InvalidOid,
				  0);
	tle = makeTargetEntry((Expr *)var,
						  list_length(parsetree->targetList) + 1,
						  pstrdup("ctid"),
						  true);
	parsetree->targetList = lappend(parsetree->targetList, tle);
}

static List *
GstorePlanForeignModify(PlannerInfo *root,
						ModifyTable *plan,
						Index resultRelation,
						int subplan_index)
{
	RangeTblEntry *rte = planner_rt_fetch(resultRelation, root);
	List	   *updatedColsList = NIL;
	int			j = -1;

	while ((j = bms_next_member(rte->updatedCols, j)) >= 0)
	{
		updatedColsList = lappend_int(updatedColsList, j +
									  FirstLowInvalidHeapAttributeNumber);
	}
	return updatedColsList;		/* anything others? */
}

static void
GstoreBeginForeignModify(ModifyTableState *mtstate,
						 ResultRelInfo *rinfo,
						 List *fdw_private,
						 int subplan_index,
						 int eflags)
{
	GpuStoreFdwModify *gs_mstate = palloc0(sizeof(GpuStoreFdwModify));
	Relation	frel = rinfo->ri_RelationDesc;
	List	   *updatedColsList = fdw_private;
	Bitmapset  *updatedCols = NULL;
	AttrNumber	ctid_attno = InvalidAttrNumber;
	Plan	   *subplan = mtstate->mt_plans[subplan_index]->plan;
	ListCell   *lc;

	if (mtstate->operation == CMD_UPDATE)
	{
		foreach (lc, updatedColsList)
		{
			int		attnum = lfirst_int(lc);

			if (attnum > 0)
				updatedCols = bms_add_member(updatedCols, attnum);
		}
	}
	if (mtstate->operation == CMD_UPDATE ||
		mtstate->operation == CMD_DELETE)
	{
		ctid_attno = ExecFindJunkAttributeInTlist(subplan->targetlist,
												  "ctid");
		if (!AttributeNumberIsValid(ctid_attno))
			elog(ERROR, "could not find junk ctid column");
	}
	gs_mstate->gs_desc = gstoreFdwLookupGpuStoreDesc(frel);
	gs_mstate->updatedCols = updatedCols;
	gs_mstate->next_rowid = 0;
	gs_mstate->oldestXmin = GetOldestXmin(frel, PROCARRAY_FLAGS_VACUUM);
	gs_mstate->ctid_attno = ctid_attno;
	gs_mstate->gs_undo = gstoreFdwLookupUndoLogs(gs_mstate->gs_desc);
	rinfo->ri_FdwState = gs_mstate;
}

static char *
__gstoreAllocExtraBuffer(GpuStoreDesc *gs_desc, size_t extra_sz)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	kern_data_store *kds;
	kern_data_extra *extra;
	bool		has_exclusive_lock = false;

	/* must be aligned */
	Assert(extra_sz == MAXALIGN(extra_sz));
	elog(INFO, "%s: extra_sz=%zu", __FUNCTION__, extra_sz);
	LWLockAcquire(&gs_sstate->base_mmap_lock, LW_SHARED);
retry:
	if (gs_desc->base_mmap_revision != gs_sstate->base_mmap_revision)
		gstoreFdwRemapBaseFile(gs_desc, true);
	kds = &gs_desc->base_mmap->schema;
	extra = (kern_data_extra *)((char *)kds + kds->extra_hoffset);
	for (;;)
	{
		cl_ulong	old_usage = atomicRead64(&extra->usage);
		cl_ulong	new_usage = old_usage + extra_sz;

		elog(INFO, "old_usage=%lu new_usage=%lu length=%lu", old_usage, new_usage, extra->length);
		Assert(old_usage >= offsetof(kern_data_extra, data));
		if (new_usage <= extra->length)
		{
			if (atomicCAS64(&extra->usage, &old_usage, new_usage))
			{
				elog(INFO, "--> usage=%lu length=%lu", extra->usage, extra->length);
				LWLockRelease(&gs_sstate->base_mmap_lock);

				return (char *)extra + old_usage;
			}
		}
		else if (!has_exclusive_lock)
		{
			/* oops, we have to expand the base file, but no exclusive lock */
			LWLockRelease(&gs_sstate->base_mmap_lock);
			LWLockAcquire(&gs_sstate->base_mmap_lock, LW_EXCLUSIVE);
			has_exclusive_lock = true;
			goto retry;
		}
		else
		{
			/* expand the base file */
			size_t		sz = extra->length;
			size_t		off;
			int			rawfd;
			uint32		revision;

			sz = extra->length;
			if (kds->nitems > 0)
				sz = ((double)kds->nrooms / (double)kds->nitems) * (double)sz;
			sz = Max(sz, extra->length + (64UL << 20));
			sz = PAGE_ALIGN(sz);
			off = offsetof(GpuStoreBaseFileHead, schema) + kds->extra_hoffset;

			elog(INFO, "sz = %zu off = %zu", sz, off);
			
			rawfd = open(gs_sstate->base_file, O_RDWR);
			if (rawfd < 0)
				elog(ERROR, "failed on open('%s'): %m", gs_sstate->base_file);
			if (posix_fallocate(rawfd, off, sz) != 0)
			{
				close(rawfd);
				elog(ERROR, "failed on posix_fallocate('%s',%zu): %m",
					 gs_sstate->base_file, sz);
			}
			close(rawfd);
			extra->length = sz;

			do {
				revision = random();
			} while (revision == UINT_MAX ||
					 revision == gs_sstate->base_mmap_revision);
			gs_sstate->base_mmap_revision = revision;
			/* Ok, remap base and retry */
			goto retry;
		}
	}
}

static TupleTableSlot *
__gstoreExecForeignModify(EState *estate,
						  ResultRelInfo *rinfo,
						  CmdType operation,
						  TupleTableSlot *slot,		/* INSERT or UPDATE */
						  TupleTableSlot *planSlot)	/* UPDATE or DELETE */
{
	GpuStoreFdwModify *gs_mstate = rinfo->ri_FdwState;
	GpuStoreDesc   *gs_desc = gs_mstate->gs_desc;
	GpuStoreUndoLogs *gs_undo = gs_mstate->gs_undo;
	kern_data_store *kds = &gs_desc->base_mmap->schema;
	Relation		frel = rinfo->ri_RelationDesc;
	TransactionId	curr_xid = GetCurrentTransactionId();
	cl_uint			rowid = UINT_MAX;
	cl_uint			oldid = UINT_MAX;
	int				j, natts = RelationGetNumberOfAttributes(frel);
	char			temp[10];

	enlargeStringInfo(&gs_undo->buf, 2 * (sizeof(char) + sizeof(cl_uint)));
	/*
	 * UPDATE or DELETE marks 'removed' on the old row; it shall work as
	 * low-level lock for other concurrent transactions.
	 */
	if (operation == CMD_UPDATE || operation == CMD_DELETE)
	{
		GstoreFdwSysattr sysattr;
		ItemPointer	ctid;
		Datum		datum;
		bool		isnull;
		bool		locked = false;

		datum = ExecGetJunkAttribute(planSlot, gs_mstate->ctid_attno, &isnull);
		if (isnull)
			elog(ERROR, "ctid is NULL");
		ctid = (ItemPointer)DatumGetPointer(datum);
		oldid = (((cl_uint)ctid->ip_blkid.bi_hi << 16) |
				 ((cl_uint)ctid->ip_blkid.bi_lo));

		ConditionVariablePrepareToSleep(&gstore_shared_head->row_lock_cond);
		PG_TRY();
		{
			int		visible;
		wakeup_retry:
			locked = gstoreFdwSpinLockBaseRow(gs_desc, oldid);
			visible = gstoreCheckVisibilityTryDelete(gs_desc, oldid, curr_xid,
													 &sysattr);
			locked = gstoreFdwSpinUnlockBaseRow(gs_desc, oldid);
			if (visible < 0)
			{
				ConditionVariableSleep(&gstore_shared_head->row_lock_cond,
									   PG_WAIT_LOCK);
				goto wakeup_retry;
			}
			if (visible == 0)
				elog(ERROR, "concurrent update/delete of foreign table '%s' at rowid=%u",
					 RelationGetRelationName(frel), oldid);
		}
		PG_CATCH();
		{
			if (locked)
				gstoreFdwSpinUnlockBaseRow(gs_desc, oldid);
			ConditionVariableCancelSleep();
			PG_RE_THROW();
		}
		PG_END_TRY();
		ConditionVariableCancelSleep();

		/* Put DELETE log */
		gstoreFdwAppendDeleteLog(frel, gs_desc, oldid,
								 sysattr.xmin,
								 sysattr.xmax);
		/* UNDO Log also */
		temp[0] = 'D';
		*((uint32 *)(temp + 1)) = oldid;
        appendBinaryStringInfo(&gs_undo->buf, temp, 5);
        gs_undo->nitems++;
	}

	/*
	 * INSERT or UPDATE adds a new row
	 */
	if (operation == CMD_INSERT || operation == CMD_UPDATE)
	{
		Bitmapset  *updatedCols = gs_mstate->updatedCols;
		size_t		extra_sz = 0;
		char	   *extra_buf = NULL;

		slot_getallattrs(slot);

		/* calculation of required extra buffer size */
		if (kds->has_varlena)
		{
			for (j=0; j < natts; j++)
			{
				kern_colmeta   *cmeta = &kds->colmeta[j];
				Datum			datum = slot->tts_values[j];
				bool			isnull = slot->tts_isnull[j];

			if (cmeta->attlen == -1 && !isnull &&
				(!updatedCols || bms_is_member(cmeta->attnum, updatedCols)))
				extra_sz += MAXALIGN(VARSIZE_ANY(datum));
			}
		}

		/* new RowId allocation */
		for (;;)
		{
			rowid = gstoreFdwAllocateRowId(gs_desc, gs_mstate->next_rowid);
			if (rowid >= kds->nrooms)
			elog(ERROR, "gstore_fdw: '%s' has no room to INSERT any rows",
				 RelationGetRelationName(frel));
			gs_mstate->next_rowid = rowid + 1;
			gstoreFdwSpinLockBaseRow(gs_desc, rowid);
			if (gstoreCheckVisibilityForInsert(gs_desc, rowid,
											   gs_mstate->oldestXmin))
				break;
			gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);
			gstoreFdwReleaseRowId(gs_desc, rowid);
		}
		/* set up user defined attributes under the row-lock */
		PG_TRY();
		{
			GstoreFdwSysattr sysattr;
			HeapTuple		tuple;

			/* Put INSERT Log */
			tuple = ExecFetchSlotHeapTuple(slot, false, false);
			gstoreFdwAppendInsertLog(frel, gs_desc, rowid, oldid, curr_xid, tuple);
			/* UNDO Log also */
			temp[0] = 'I';
			*((uint32 *)(temp + 1)) = rowid;
			appendBinaryStringInfo(&gs_undo->buf, temp, 5);
			gs_undo->nitems++;

			if (extra_sz > 0)
				extra_buf = __gstoreAllocExtraBuffer(gs_desc, extra_sz);
			kds = &gs_desc->base_mmap->schema;
			for (j=0; j < natts; j++)
			{
				kern_colmeta   *cmeta = &kds->colmeta[j];
				Datum			datum = slot->tts_values[j];
				bool			isnull = slot->tts_isnull[j];

				if (cmeta->attlen == -1 && !isnull)
				{
					size_t		sz = VARSIZE_ANY(datum);

					if (!updatedCols || bms_is_member(cmeta->attnum, updatedCols))
					{
						memcpy(extra_buf, DatumGetPointer(datum), sz);
						datum = PointerGetDatum(extra_buf);
						extra_buf += MAXALIGN(sz);
					}
					else
					{
						/*
						 * If UPDATE does not change this varlena column,
						 * we don't need to consume redundant extra buffer for
						 * identicalsame value. So, we reuse the older version
						 * of the unchanged value. It also saves consumption of
						 * GPU device memory.
						 */
						kern_data_extra *extra = (kern_data_extra *)
							((char *)kds + kds->extra_hoffset);
						datum = KDS_fetch_datum_column(kds, cmeta, oldid, &isnull);
						if (isnull)
							elog(ERROR, "Bug? unchanged column has different value");
						Assert((char *)datum >= (char *)extra &&
							   (char *)datum <  (char *)extra + extra->length);
					}
				}
				__KDS_store_datum_column(kds, cmeta, rowid, datum, isnull);
			}
			sysattr.xmin = curr_xid;
			sysattr.xmax = InvalidTransactionId;
			sysattr.cid  = GetCurrentCommandId(true);
			KDS_store_datum_column(kds, &kds->colmeta[natts], rowid,
								   PointerGetDatum(&sysattr),
								   false);
			atomicMax32(&kds->nitems, rowid + 1);
		}
		PG_CATCH();
		{
			gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);
			gstoreFdwReleaseRowId(gs_desc, rowid);
			PG_RE_THROW();
		}
		PG_END_TRY();
		gstoreFdwSpinUnlockBaseRow(gs_desc, rowid);
	}
	/* UPDATE or DELETE */
	if (oldid != UINT_MAX)
	{
		Assert(operation == CMD_UPDATE || operation == CMD_DELETE);
		gstoreFdwRemoveFromPrimaryKey(gs_undo, oldid);
	}
	/* INSERT or UPDATE */
	if (rowid != UINT_MAX)
	{
		Assert(operation == CMD_INSERT || operation == CMD_UPDATE);
		gstoreFdwInsertIntoPrimaryKey(gs_undo, rowid);
	}
	return slot;
}

static TupleTableSlot *
GstoreExecForeignInsert(EState *estate,
						ResultRelInfo *rinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	return __gstoreExecForeignModify(estate, rinfo, CMD_INSERT, slot, planSlot);
}

static TupleTableSlot *
GstoreExecForeignUpdate(EState *estate,
						ResultRelInfo *rinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	return __gstoreExecForeignModify(estate, rinfo, CMD_UPDATE, slot, planSlot);
}

static TupleTableSlot *
GstoreExecForeignDelete(EState *estate,
						ResultRelInfo *rinfo,
						TupleTableSlot *slot,
						TupleTableSlot *planSlot)
{
	return __gstoreExecForeignModify(estate, rinfo, CMD_DELETE, slot, planSlot);
}

static void
GstoreEndForeignModify(EState *estate, ResultRelInfo *rinfo)
{}

void
ExplainGstoreFdw(GpuStoreFdwState *fdw_state,
				 Relation frel, ExplainState *es)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	GpuStoreDesc   *gs_desc = fdw_state->gs_desc;
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	int				j = -1;
	StringInfoData	buf;

	/* shows referenced columns */
	initStringInfo(&buf);
	if (es->verbose)
	{
		while ((j = bms_next_member(fdw_state->referenced, j)) >= 0)
		{
			int		anum = j + FirstLowInvalidHeapAttributeNumber;
			Form_pg_attribute attr;

			if (anum <= 0)
				continue;
			attr = tupleDescAttr(tupdesc, anum-1);
			if (buf.len > 0)
				appendStringInfoString(&buf, ", ");
			appendStringInfo(&buf, "%s", NameStr(attr->attname));
		}
		ExplainPropertyText("Referenced", buf.data, es);
	}

	/* shows index condition */
	resetStringInfo(&buf);
	if (fdw_state->indexExprState)
	{
		Node	   *indexExpr = (Node *)fdw_state->indexExprState->expr;
		Form_pg_attribute attr;

		Assert(gs_sstate->primary_key > 0 &&
			   gs_sstate->primary_key <= tupdesc->natts);
		attr = tupleDescAttr(tupdesc, gs_sstate->primary_key - 1);
		appendStringInfo(&buf, "%s = %s",
						 quote_identifier(NameStr(attr->attname)),
						 deparse_expression(indexExpr, NIL, false, false));
		ExplainPropertyText("Index Cond", buf.data, es);
	}

	/* shows base&redo filename */
	if (es->verbose)
	{
		ExplainPropertyText("Base file", gs_sstate->base_file, es);
		ExplainPropertyText("Redo log", gs_sstate->redo_log_file, es);
		if (gs_sstate->redo_log_backup_dir)
			ExplainPropertyText("Backup directory",
								gs_sstate->redo_log_backup_dir, es);
	}
	pfree(buf.data);
}

static void
GstoreExplainForeignScan(ForeignScanState *node, ExplainState *es)
{
	ExplainGstoreFdw((GpuStoreFdwState *)node->fdw_state,
					 node->ss.ss_currentRelation, es);
}

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
	{
		if (options != NIL)
			elog(ERROR, "unknown FDW options");
		PG_RETURN_VOID();
	}
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
		else if (strcmp(def->defname, "redo_log_limit") == 0 ||
				 strcmp(def->defname, "gpu_update_threshold") == 0)
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
						cl_long *p_gpu_update_interval,
						size_t *p_gpu_update_threshold,
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
	ssize_t		redo_log_limit = (512U << 20);	/* default: 512MB */
	cl_long		gpu_update_interval = 15;		/* default: 15s */
	ssize_t		gpu_update_threshold = -1;		/* default: 20% of redo_log_limit */
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
			ssize_t	threshold = strtol(value, &endp, 10);

			if (threshold <= 0)
				elog(ERROR, "invalid gpu_update_threshold: %s", value);
			if (strcmp(endp, "k") == 0 || strcmp(endp, "kb") == 0)
				threshold *= (1UL << 10);
			else if (strcmp(endp, "m") == 0 || strcmp(endp, "mb") == 0)
				threshold *= (1UL << 20);
			else if (strcmp(endp, "g") == 0 || strcmp(endp, "gb") == 0)
				threshold *= (1UL << 30);
			else if (*endp != '\0')
				elog(ERROR, "invalid redo_log_sz: %s", value);
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
	if (gpu_update_threshold < 0)
		gpu_update_threshold = redo_log_limit / 5;
	else if (gpu_update_threshold < redo_log_limit / 50 ||
			 gpu_update_threshold > redo_log_limit / 2)
		elog(ERROR, "gpu_update_threshold is out of range: must be [%zu..%zu]",
			 redo_log_limit / 50, redo_log_limit / 2);
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
	TupleDesc	__tupdesc = CreateTemplateTupleDesc(tupdesc->natts + 1);

	for (int j=0; j < tupdesc->natts; j++)
	{
		memcpy(tupleDescAttr(__tupdesc, j),
			   tupleDescAttr(tupdesc, j),
			   ATTRIBUTE_FIXED_PART_SIZE);
	}
	TupleDescInitEntry(__tupdesc,
					   __tupdesc->natts,
					   "..gstore_fdw.sysattr..",
					   GSTORE_FDW_SYSATTR_OID, -1, 0);
	tupleDescAttr(__tupdesc, __tupdesc->natts - 1)->attnotnull = true;

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
	cl_long		gpu_update_interval;
	size_t		gpu_update_threshold;
	AttrNumber	primary_key;
	size_t		len;
	char	   *pos;
	cl_int		i;

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
		/* num_hash_slots >= (2^32) does not make sense because hash_any
		 * shall return hash-value in uint32. */
		num_hash_slots = 1.2 * (double)max_num_rows + 1000.0;
		num_hash_slots = Min(num_hash_slots, UINT_MAX);
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
	Assert(pos - (char *)gs_sstate == len);

	gs_sstate->database_oid = MyDatabaseId;
	gs_sstate->ftable_oid = RelationGetRelid(frel);
	gs_sstate->cuda_dindex = cuda_dindex;
	gs_sstate->max_num_rows = max_num_rows;
	gs_sstate->num_hash_slots = num_hash_slots;
	gs_sstate->primary_key = primary_key;
	gs_sstate->redo_log_limit = redo_log_limit;
	gs_sstate->redo_log_backup_limit = 5 * redo_log_limit;
	gs_sstate->gpu_update_interval = gpu_update_interval;
	gs_sstate->gpu_update_threshold = gpu_update_threshold;

	LWLockInitialize(&gs_sstate->base_mmap_lock, -1);
	gs_sstate->base_mmap_revision = UINT_MAX;

	SpinLockInit(&gs_sstate->rowid_map_lock);
	for (i=0; i < GSTORE_NUM_BASE_ROW_LOCKS; i++)
		SpinLockInit(&gs_sstate->base_row_lock[i]);
	for (i=0; i < GSTORE_NUM_HASH_SLOT_LOCKS; i++)
		SpinLockInit(&gs_sstate->hash_slot_lock[i]);
	SpinLockInit(&gs_sstate->redo_pos_lock);
	gs_sstate->redo_write_pos = 0;
	gs_sstate->redo_read_pos = 0;
	
	pthreadRWLockInit(&gs_sstate->gpu_bufer_lock);

	return gs_sstate;
}










/* ----------------------------------------------------------------
 *
 * Routines to allocate/release RowIDs
 *
 * ----------------------------------------------------------------
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

/* ----------------------------------------------------------------
 *
 * Routines for hash-base primary key index
 *
 * ----------------------------------------------------------------
 */
static void
gstoreFdwInsertIntoPrimaryKey(GpuStoreUndoLogs *gs_undo, cl_uint rowid)
{
	GpuStoreDesc   *gs_desc = gs_undo->gs_desc;
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	GpuStoreHashIndexHead *hash_index = gs_desc->hash_index;
	kern_data_store	*kds = &gs_desc->base_mmap->schema;
	kern_colmeta   *cmeta;
	TypeCacheEntry *tcache;
	Datum			kdatum;
	Datum			hash;
	bool			isnull;
	bool			is_locked;
	char			temp[10];

	if (!hash_index)
		return;		/* no primary key index */
	Assert(gs_sstate->primary_key > 0 &&
		   gs_sstate->primary_key <= kds->ncols);
	Assert(hash_index->nrooms == kds->nrooms);
	if (rowid >= hash_index->nrooms)
		elog(ERROR, "Bug? rowid=%u is larger than index's nrooms=%lu",
			 rowid, hash_index->nrooms);
	enlargeStringInfo(&gs_undo->buf, sizeof(char) + 2 * sizeof(cl_uint));

	/* calculation of hash-value for the PK */
	cmeta = &kds->colmeta[gs_sstate->primary_key - 1];
	tcache = lookup_type_cache(cmeta->atttypid,
							   TYPECACHE_EQ_OPR_FINFO |
							   TYPECACHE_HASH_PROC_FINFO);
	kdatum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
	if (isnull)
		elog(ERROR, "primary key '%s' of foreign table '%s' is NULL",
			 NameStr(cmeta->attname), get_rel_name(kds->table_oid));
	hash = FunctionCall1(&tcache->hash_proc_finfo, kdatum);
	is_locked = gstoreFdwSpinLockHashSlot(gs_desc, hash);
	PG_TRY();
	{
		cl_uint	   *rowmap = &hash_index->slots[hash_index->nslots];
		cl_uint		curr_id;
		cl_uint		k = hash % hash_index->nslots;

		/* duplication check of the primary key */
		ConditionVariablePrepareToSleep(&gstore_shared_head->row_lock_cond);
	wakeup_retry:
		for (curr_id = hash_index->slots[k];
			 curr_id < hash_index->nrooms;
			 curr_id = rowmap[curr_id])
		{
			int		visible;
			Datum	cdatum;

			visible = gstoreCheckVisibilityForIndex(gs_desc, curr_id);
			if (visible == 0)
				continue;
			cdatum = KDS_fetch_datum_column(kds, cmeta, curr_id, &isnull);
			if (isnull)
			{
				elog(WARNING, "Bug? primary key '%s' has NULL at rowid=%u",
					 NameStr(cmeta->attname), curr_id);
				continue;
			}
			if (DatumGetBool(FunctionCall2(&tcache->eq_opr_finfo,
										   kdatum, cdatum)))
			{
				if (visible > 0)
				{
					Oid		typoutput;
					bool	typisvarlena;

					getTypeOutputInfo(cmeta->atttypid,
									  &typoutput,
									  &typisvarlena);
					elog(ERROR,"duplicate primary key violation; %s = %s already exists",
						 NameStr(cmeta->attname),
						 OidOutputFunctionCall(typoutput, kdatum));
				}
				/*
				 * NOTE: This row has duplicated value, however, not committed
				 * yet, thus, this session must be blocked by the row-level lock.
				 */
				is_locked = gstoreFdwSpinUnlockHashSlot(gs_desc, hash);
				ConditionVariableSleep(&gstore_shared_head->row_lock_cond,
									   PG_WAIT_LOCK);
				is_locked = gstoreFdwSpinLockHashSlot(gs_desc, hash);
				goto wakeup_retry;
			}
		}
		/* Undo Log - Add PK with hash(u32) + rowid(u32) */
		temp[0] = 'A';
		*((cl_uint *)(temp + 1)) = hash;
		*((cl_uint *)(temp + 5)) = rowid;
		appendBinaryStringInfo(&gs_undo->buf, temp,
							   sizeof(char) + 2 * sizeof(cl_uint));
		gs_undo->nitems++;
		
		/* Ok, no primary key violation */
		rowmap[rowid] = hash_index->slots[k];
		hash_index->slots[k] = rowid;
	}
	PG_CATCH();
	{
		if (is_locked)
			gstoreFdwSpinUnlockHashSlot(gs_desc, hash);
		ConditionVariableCancelSleep();
		PG_RE_THROW();
	}
	PG_END_TRY();
	Assert(is_locked);
	gstoreFdwSpinUnlockHashSlot(gs_desc, hash);
	ConditionVariableCancelSleep();

	elog(INFO, "Insert PK of rowid=%u, hash=%08lx", rowid, hash);
}

static void
gstoreFdwRemoveFromPrimaryKey(GpuStoreUndoLogs *gs_undo, cl_uint rowid)
{
	GpuStoreDesc	*gs_desc = gs_undo->gs_desc;
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	GpuStoreHashIndexHead *hash_index = gs_desc->hash_index;
	kern_data_store	*kds = &gs_desc->base_mmap->schema;
	kern_colmeta   *cmeta = &kds->colmeta[gs_sstate->primary_key - 1];
	TypeCacheEntry *tcache;
	Datum			datum;
	Datum			hash;
	bool			isnull;

	if (gs_sstate->primary_key < 1 ||
		gs_sstate->primary_key >= kds->ncols)
		return;		/* no primary key is configured */
	Assert(hash_index != NULL);
	if (rowid >= hash_index->nrooms)
		elog(ERROR, "index corruption? rowid=%u is larger than index's nrooms=%lu",
			 rowid, hash_index->nrooms);

	tcache = lookup_type_cache(cmeta->atttypid,
							   TYPECACHE_HASH_PROC |
							   TYPECACHE_HASH_PROC_FINFO);
	datum = KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
	if (isnull)
		elog(ERROR, "primary key '%s' of foreign table '%s' is NULL",
			 NameStr(cmeta->attname), get_rel_name(kds->table_oid));
	hash = FunctionCall1(&tcache->hash_proc_finfo, datum);

	enlargeStringInfo(&gs_undo->buf, sizeof(char) + 2*sizeof(cl_uint));
	gstoreFdwSpinLockHashSlot(gs_desc, hash);
	/*
	 * Ensure rowid exists on the hash-index.
	 */
	{
		cl_uint	   *rowmap = &hash_index->slots[hash_index->nslots];
		cl_uint		curr_id;
		char		temp[10];
		bool		found = false;

		for (curr_id = hash_index->slots[hash % hash_index->nslots];
			 curr_id < hash_index->nrooms;
			 curr_id = rowmap[curr_id])
		{
			if (curr_id == rowid)
			{
				found = true;
				break;
			}
		}

		if (found)
		{
			/* Remove from PK with hash(u32) + rowid(u32) */
			temp[0] = 'R';
			*((cl_uint *)(temp + 1)) = hash;
			*((cl_uint *)(temp + 5)) = rowid;
			appendBinaryStringInfo(&gs_undo->buf, temp,
								   sizeof(char) + 2 * sizeof(cl_uint));
			gs_undo->nitems++;
		}
		else
		{
			elog(WARNING, "primary key '%s' for rowid='%u' not found",
				 NameStr(cmeta->attname), rowid);
		}
	}
	gstoreFdwSpinUnlockHashSlot(gs_desc, hash);

	elog(INFO, "Remove PK of rowid=%u, hash=%08lx", rowid, hash);
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

	/* write out kern_data_extra section (if any varlena) */
	if (schema->has_varlena)
	{
		kern_data_extra		extra;

		memset(&extra, 0, offsetof(kern_data_extra, data));
		memcpy(extra.signature, GPUSTORE_EXTRABUF_SIGNATURE, 8);
		extra.length = extra_sz;
		extra.usage = offsetof(kern_data_extra, data);

		sz = offsetof(GpuStoreBaseFileHead,
					  schema) + hbuf->schema.extra_hoffset;
		if (lseek(rawfd, sz, SEEK_SET) < 0)
			elog(ERROR, "failed on lseek('%s',%zu): %m",
				 base_file, hbuf->schema.extra_hoffset);
		sz = offsetof(kern_data_extra, data);
		if (__writeFile(rawfd, &extra, sz) != sz)
			elog(ERROR, "failed on __writeFile('%s'): %m", base_file);
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
			if (base_mmap->schema.extra_hoffset != 0)
				elog(ERROR, "Base file '%s' has extra buffer, but foreign-table '%s' has no variable-length fields",
					 gs_sstate->base_file, RelationGetRelationName(frel));
		}
		else
		{
			kern_data_store *schema = &base_mmap->schema;
			kern_data_extra *extra = (kern_data_extra *)
				((char *)schema + schema->extra_hoffset);

			if (schema->extra_hoffset == 0)
				elog(ERROR, "Base file '%s' has no extra buffer, but foreign-table '%s' has variable-length fields",
					 gs_sstate->base_file, RelationGetRelationName(frel));
			if (offsetof(GpuStoreBaseFileHead, schema) +
				schema->extra_hoffset + extra->length > mmap_sz)
				elog(ERROR, "Base file '%s' is smaller then the required",
					 gs_sstate->base_file);
			if (extra->usage > extra->length)
				elog(ERROR, "Extra buffer of base file '%s' has larger usage (%lu) than length (%lu)", gs_sstate->base_file, extra->usage, extra->length);
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
	gs_sstate->base_mmap_revision = random();
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
	size_t		sz;
	
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
	}
	sz = PAGE_ALIGN(gs_sstate->redo_log_limit);
	if (posix_fallocate(FileGetRawDesc(redo_fdesc), 0, sz) != 0)
		elog(ERROR, "failed on posix_fallocate('%s',%zu): %m",
			 redo_log_file, sz);
	FileClose(redo_fdesc);
}

/*
 * GSTORE_TX_LOG__INSERT
 */
static void
__ApplyRedoLogInsert(Relation frel,
					 GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz,
					 GstoreTxLogInsert *i_log)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	Datum		   *values = alloca(sizeof(Datum) * tupdesc->natts);
	bool		   *isnull = alloca(sizeof(bool) * tupdesc->natts);
	kern_data_store *kds = &base_mmap->schema;
	HeapTupleData	tuple;
	GstoreFdwSysattr sysattr;
	int				j;

	memset(&tuple, 0, sizeof(HeapTupleData));
	tuple.t_data = &i_log->htup;
	heap_deform_tuple(&tuple, tupdesc, values, isnull);
	Assert(kds->ncols == tupdesc->natts + 1);	/* + sysattr */
	for (j=0; j < tupdesc->natts; j++)
	{
		KDS_store_datum_column(kds, &kds->colmeta[j],
							   i_log->rowid,
							   values[j], isnull[j]);
	}
	memset(&sysattr, 0, sizeof(GstoreFdwSysattr));
	sysattr.xmin = HeapTupleHeaderGetRawXmin(&i_log->htup);
	sysattr.xmax = HeapTupleHeaderGetRawXmax(&i_log->htup);
	sysattr.cid  = HeapTupleHeaderGetRawCommandId(&i_log->htup);
	KDS_store_datum_column(kds, &kds->colmeta[kds->ncols-1],
						   i_log->rowid,
						   PointerGetDatum(&sysattr), false);
}

/*
 * GSTORE_TX_LOG__DELETE
 */
static void
__ApplyRedoLogDelete(Relation frel,
					 GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz,
					 GstoreTxLogDelete *d_log)
{
	kern_data_store *kds = &base_mmap->schema;
	GstoreFdwSysattr *sysattr;
	bool			isnull;

	sysattr = (GstoreFdwSysattr *)
		KDS_fetch_datum_column(kds, &kds->colmeta[kds->ncols-1],
							   d_log->rowid, &isnull);
	Assert(!isnull);
	/* xmax & cid */
	sysattr->xmin = d_log->xmin;
	sysattr->xmax = d_log->xmax;
	sysattr->cid  = InvalidCommandId;
}

static void
__rebuildRowIdMap(Relation frel, GpuStoreSharedState *gs_sstate,
				  GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz)
{
#if 0
	kern_data_store *kds = &base_mmap->schema;
	kern_colmeta *cmeta = &kds->colmeta[kds->ncols - 1];
	GpuStoreRowIdMapHead *rowmap;
	size_t		rowmap_sz;
	cl_uint		rowid;

	/* clear the rowid-map */
	rowmap_sz = gstoreFdwRowIdMapSize(kds->nrooms);
	rowmap = (GpuStoreRowIdMapHead *)
		((char *)base_mmap + base_mmap->rowid_map_offset);
	memset(rowmap, 0, rowmap_sz);
	memcpy(rowmap->signature, GPUSTORE_ROWIDMAP_SIGNATURE, 8);
	rowmap->length = rowmap_sz;
	rowmap->nrooms = kds->nrooms;

	for (rowid=0; rowid < kds->nitems; rowid++)
	{
		GstoreFdwSysattr *sysattr;
		bool		isnull;

		sysattr = (GstoreFdwSysattr *)
			KDS_fetch_datum_column(kds, cmeta, rowid, &isnull);
		if (isnull)
		{
			elog(WARNING, "foreign table '%s', rowid=%u has NULL on system column",
				 RelationGetRelationName(frel), rowid);
			continue;
		}

		if ((sysattr->xmin == FrozenTransactionId ||
			 TransactionIdDidCommit(sysattr->xmin)) &&
			(sysattr->xmax != FrozenTransactionId &&
			 !TransactionIdDidCommit(sysattr->xmax)))
		{
			gstoreFdwAllocateRowId();
		}
	}
#endif
}

static void
__rebuildHashIndex(Relation frel, GpuStoreSharedState *gs_sstate,
				   GpuStoreBaseFileHead *base_mmap, size_t base_mmap_sz)
{
	/* TODO: rebuild HashIndex */
}

static int
__GstoreTxLogCommonCompare(const void *pos1, const void *pos2)
{
	const GstoreTxLogCommon *tx_log1 = pos1;
	const GstoreTxLogCommon *tx_log2 = pos2;

	if (tx_log1->timestamp < tx_log2->timestamp)
		return -1;
	if (tx_log1->timestamp > tx_log2->timestamp)
		return  1;
	return 0;
}

static void
gstoreFdwApplyRedoHostBuffer(Relation frel, GpuStoreSharedState *gs_sstate)
{
	GpuStoreBaseFileHead *base_mmap = NULL;
	size_t		base_mmap_sz;
	int			base_is_pmem;
	char	   *redo_mmap = NULL;
	size_t		redo_mmap_sz;
	int			redo_is_pmem;

	PG_TRY();
	{
		GstoreTxLogCommon *tail_log = NULL;
		cl_uint		i, nitems = 0;
		char	   *pos, *end;
		StringInfoData buf;

		base_mmap = pmem_map_file(gs_sstate->base_file, 0,
								  0, 0600,
								  &base_mmap_sz,
								  &base_is_pmem);
		if (!base_mmap)
			elog(ERROR, "failed on pmem_map_file('%s'): %m",
				 gs_sstate->base_file);
		redo_mmap = pmem_map_file(gs_sstate->redo_log_file, 0,
								  0, 0600,
								  &redo_mmap_sz,
								  &redo_is_pmem);
		if (!redo_mmap)
			elog(ERROR, "failed on pmem_map_file('%s'): %m",
				 gs_sstate->redo_log_file);

		/*
		 * Seek to the first and last log position
		 */
		initStringInfo(&buf);
		pos = redo_mmap;
		end = redo_mmap + gs_sstate->redo_log_limit;
		while (pos + offsetof(GstoreTxLogCommon, data) <= end)
		{
			GstoreTxLogCommon *curr = (GstoreTxLogCommon *)pos;

			/* Quick validation of the Log */
			if (((curr->type != GSTORE_TX_LOG__INSERT) &&
				 (curr->type != GSTORE_TX_LOG__DELETE) &&
				 (curr->type != GSTORE_TX_LOG__COMMIT)) ||
				(char *)curr + curr->length + sizeof(uint64) > end ||
				curr->length != MAXALIGN(curr->length) ||
				((cl_uint *)curr + curr->length)[-1] != GSTORE_TX_LOG__TERMINATOR)
			{
				pos += sizeof(cl_long);
				continue;
			}

			if (!tail_log || curr->timestamp > tail_log->timestamp)
				tail_log = curr;

			if (curr->type != GSTORE_TX_LOG__COMMIT)
			{
				enlargeStringInfo(&buf, sizeof(GstoreTxLogCommon *));
				((GstoreTxLogCommon **)buf.data)[nitems++] = curr;
				buf.len += sizeof(GstoreTxLogCommon *);
			}
			pos += curr->length;
		}

		if (nitems > 0)
		{
			uint64		start_pos;

			/* apply GstoreTxLogRow records */
			qsort(buf.data, nitems, sizeof(GstoreTxLogCommon *),
				  __GstoreTxLogCommonCompare);
			for (i=0; i < nitems; i++)
			{
				GstoreTxLogCommon *tx_log = ((GstoreTxLogCommon **)buf.data)[i];

				if (tx_log->type == GSTORE_TX_LOG__INSERT)
				{
					__ApplyRedoLogInsert(frel, base_mmap, base_mmap_sz,
										 (GstoreTxLogInsert *)tx_log);
				}
				else if (tx_log->type == GSTORE_TX_LOG__DELETE)
				{
					__ApplyRedoLogDelete(frel, base_mmap, base_mmap_sz,
										 (GstoreTxLogDelete *)tx_log);
				}
				else
				{
					elog(ERROR, "unexpected log type on REDO File");
				}
			}
			/* rebbuild row-id map, and PK hash-index */
			__rebuildRowIdMap(frel, gs_sstate, base_mmap, base_mmap_sz);
			__rebuildHashIndex(frel, gs_sstate, base_mmap, base_mmap_sz);
			if (base_is_pmem)
				pmem_persist(base_mmap, base_mmap_sz);
			else
				pmem_msync(base_mmap, base_mmap_sz);

			/* Seek the redo_xxx_pos next to the tail */
			start_pos = (char *)tail_log + tail_log->length - (char *)redo_mmap;
			gs_sstate->redo_write_pos = start_pos;
			gs_sstate->redo_read_pos  = start_pos;
			gs_sstate->redo_sync_pos  = start_pos;
		}
		pfree(buf.data);
	}
	PG_CATCH();
	{
		if (base_mmap)
			pmem_unmap(base_mmap, base_mmap_sz);
		if (redo_mmap)
			pmem_unmap(redo_mmap, redo_mmap_sz);
		PG_RE_THROW();
	}
	PG_END_TRY();
	if (pmem_unmap(base_mmap, base_mmap_sz) != 0)
		elog(WARNING, "failed on pmem_unmap('%s',%zu): %m",
			 gs_sstate->base_file, base_mmap_sz);
	if (pmem_unmap(redo_mmap, redo_mmap_sz) != 0)
		elog(WARNING, "failed on pmem_unmap('%s',%zu): %m",
			 gs_sstate->redo_log_file, redo_mmap_sz);
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

	PG_TRY();
	{
		bool	basefile_is_sanity = true;

		/* Create or validate base file */
		gstoreFdwCreateOrValidateBaseFile(frel, gs_sstate,
										  &basefile_is_sanity);
		/* Create redo-log file on demand */
		gstoreFdwCreateRedoLog(frel, gs_sstate);
		/* Apply Redo-Log, if needed */
		if (!basefile_is_sanity)
			gstoreFdwApplyRedoHostBuffer(frel, gs_sstate);
	}
	PG_CATCH();
	{
		pfree(gs_sstate);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return gs_sstate;
}

static bool
gstoreFdwRemapBaseFile(GpuStoreDesc *gs_desc, bool abort_on_error)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	GpuStoreBaseFileHead   *base_mmap;

	if (gs_desc->base_mmap != NULL)
	{
		if (pmem_unmap(gs_desc->base_mmap,
					   gs_desc->base_mmap_sz) != 0)
			elog(WARNING, "failed on pmem_unmap('%s'): %m",
				 gs_sstate->base_file);
	}
	elog(INFO, "gstoreFdwRemapBaseFile calls pmem_map_file");
	gs_desc->base_mmap = pmem_map_file(gs_sstate->base_file, 0,
									   0, 0600,
									   &gs_desc->base_mmap_sz,
									   &gs_desc->base_mmap_is_pmem);
	if (!gs_desc->base_mmap)
	{
		gs_desc->rowid_map = NULL;
		gs_desc->hash_index = NULL;
		gs_desc->base_mmap_revision = UINT_MAX;
		elog(abort_on_error ? ERROR : LOG,
			 "failed on pmem_map_file('%s'): %m",
			 gs_sstate->base_file);
		return false;
	}
	base_mmap = gs_desc->base_mmap;
	gs_desc->rowid_map = (GpuStoreRowIdMapHead *)
		((char *)base_mmap + base_mmap->rowid_map_offset);
	if (base_mmap->hash_index_offset == 0)
		gs_desc->hash_index = NULL;
	else
		gs_desc->hash_index = (GpuStoreHashIndexHead *)
			((char *)base_mmap + base_mmap->hash_index_offset);
	gs_desc->base_mmap_revision = gs_sstate->base_mmap_revision;

	return true;
}

static bool
gstoreFdwSetupGpuStoreDesc(GpuStoreDesc *gs_desc, bool abort_on_error)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;

	/* memory map the base-file */
	LWLockAcquire(&gs_sstate->base_mmap_lock, LW_SHARED);
	if (gs_desc->base_mmap == NULL ||
		gs_desc->base_mmap_revision != gs_sstate->base_mmap_revision)
	{
		if (!gstoreFdwRemapBaseFile(gs_desc, abort_on_error))
		{
			LWLockRelease(&gs_sstate->base_mmap_lock);
			return false;
		}
	}
	LWLockRelease(&gs_sstate->base_mmap_lock);

	/* memory map redo-log file */
	if (gs_desc->redo_mmap == NULL)
	{
		gs_desc->redo_mmap = pmem_map_file(gs_sstate->redo_log_file, 0,
										   0, 0600,
										   &gs_desc->redo_mmap_sz,
										   &gs_desc->redo_mmap_is_pmem);
		if (!gs_desc->redo_mmap)
		{
			elog(abort_on_error ? ERROR : LOG,
				 "failed on pmem_map_file('%s'): %m",
				 gs_sstate->redo_log_file);
			LWLockRelease(&gs_sstate->base_mmap_lock);
			return false;
		}
	}
	return true;
}

static GpuStoreDesc *
gstoreFdwLookupGpuStoreDesc(Relation frel)
{
	GpuStoreDesc *gs_desc;
	Oid			hkey[2];
	bool		found;
	bool		invoke_initial_load = false;

	if (!RelationIsGstoreFdw(frel))
		elog(ERROR, "relation '%s' is not a foreign table managed by gstore_fdw",
			 RelationGetRelationName(frel));
	
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
		dlist_iter	iter;

		memset((char *)gs_desc + sizeof(hkey), 0, sizeof(GpuStoreDesc) - sizeof(hkey));
		
		hindex = hash_any((const unsigned char *)hkey,
						  sizeof(hkey)) % GPUSTORE_SHARED_DESC_NSLOTS;
		SpinLockAcquire(&gstore_shared_head->gstore_sstate_lock[hindex]);
		PG_TRY();
		{
			dlist_foreach(iter, &gstore_shared_head->gstore_sstate_slot[hindex])
			{
				GpuStoreSharedState *temp;

				temp = dlist_container(GpuStoreSharedState,
									   hash_chain, iter.cur);
				if (temp->database_oid == MyDatabaseId &&
					temp->ftable_oid == RelationGetRelid(frel))
				{
					gs_sstate = temp;
					break;
				}
			}

			if (!gs_sstate)
			{
				gs_sstate = gstoreFdwCreateSharedState(frel);
				dlist_push_tail(&gstore_shared_head->gstore_sstate_slot[hindex],
								&gs_sstate->hash_chain);
				gs_desc->gs_sstate = gs_sstate;
				invoke_initial_load = true;
			}
			gs_desc->gs_sstate = gs_sstate;
			gs_desc->base_mmap_revision = UINT_MAX;
		}
		PG_CATCH();
		{
			SpinLockRelease(&gstore_shared_head->gstore_sstate_lock[hindex]);
			hash_search(gstore_desc_htab, hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(&gstore_shared_head->gstore_sstate_lock[hindex]);
		/* Preload GPU Memory Store, if new segment */
		if (invoke_initial_load)
			gstoreFdwInvokeInitialLoad(frel, true);
	}
	/* ensure the file mapping is latest */
	gstoreFdwSetupGpuStoreDesc(gs_desc, true);
	return gs_desc;
}

/*
 * __gstoreFdwXactOnPreCommit
 */
static void
__gstoreFdwXactOnPreCommit(GpuStoreUndoLogs *gs_undo)
{
	GstoreTxLogCommit *c_log = alloca(GSTORE_TX_LOG_COMMIT_ALLOCSZ);
	char		   *pos = gs_undo->buf.data;
	cl_uint			count = 1;

	/* setup commit log buffer */
	c_log->type = GSTORE_TX_LOG__COMMIT;
	c_log->length = offsetof(GstoreTxLogCommit, data);
	c_log->xid = gs_undo->xid_sub;
	c_log->timestamp = GetCurrentTimestamp();
	c_log->nitems = 0;

	while (count <= gs_undo->nitems)
	{
		bool	flush_commit_log = (count == gs_undo->nitems);
		
		switch (*pos)
		{
			case 'I':	/* INSERT with rowid(u32) */
			case 'D':	/* DELETE with rowid(u32) */
				if (c_log->length + 5 > GSTORE_TX_LOG_COMMIT_ALLOCSZ)
				{
					flush_commit_log = true;
					break;
				}
				memcpy((char *)c_log + c_log->length, pos, 5);
				c_log->length += 5;
				c_log->nitems++;
				pos += 5;
				count++;
				break;

			case 'A':	/* Add PK Index with hash(u32) + rowid(u32) */
			case 'R':	/* Remove PK Index + hash(u32) + rowid(u32) */
				/* skip in the pre-commit phase */
				pos += sizeof(char) + 2 * sizeof(cl_uint);
				count++;
				break;
			default:
				elog(FATAL, "Broken internal Undo log: tag='%c'", *pos);
				break;
		}

		if (flush_commit_log)
		{
			int			diff = MAXALIGN(c_log->length) - c_log->length;

			if (diff > 0)
			{
				memset((char *)c_log + c_log->length, 0, diff);
				c_log->length += diff;
			}
			__gstoreFdwAppendRedoLog(gs_undo->gs_desc,
									 (GstoreTxLogCommon *)c_log);
			/* rewind */
			c_log->length = offsetof(GstoreTxLogCommit, data);
			c_log->nitems = 0;
		}
	}
	Assert(pos <= gs_undo->buf.data + gs_undo->buf.len);
}

/*
 * __gstoreFdwXactFinalize
 */
static void
__gstoreFdwXactFinalize(GpuStoreUndoLogs *gs_undo, bool is_commit)
{
	GpuStoreDesc *gs_desc = gs_undo->gs_desc;
	GpuStoreHashIndexHead *hash_index = gs_desc->hash_index;
	char	   *pos = gs_undo->buf.data;
    cl_uint		count;
	
	for (count=0; count < gs_undo->nitems; count++)
	{
		uint32		rowid;

		switch (*pos)
		{
			case 'I':	/* INSERT */
				if (!is_commit)
				{
					rowid = *((uint32 *)(pos + 1));
					gstoreFdwReleaseRowId(gs_desc, rowid);
				}
				pos += sizeof(char) + sizeof(uint32);
				break;
			case 'D':	/* DELETE */
				if (is_commit)
				{
					rowid = *((uint32 *)(pos + 1));
					gstoreFdwReleaseRowId(gs_desc, rowid);
				}
				pos += sizeof(char) + sizeof(uint32);
				break;
			case 'A':	/* Add PK */
				if (!is_commit)
				{
					/* remove rowid from the index on abort  */
					uint32	   *rowmap = &hash_index->slots[hash_index->nslots];
					uint32		hash  = *((uint32 *)(pos + 1));
					uint32		rowid = *((uint32 *)(pos + 5));
					uint32	   *next;

					gstoreFdwSpinLockHashSlot(gs_desc, hash);
                    next = &hash_index->slots[hash % hash_index->nslots];
					while (*next < hash_index->nrooms)
					{
						if (*next == rowid)
						{
							*next = rowmap[rowid];
							rowmap[rowid] = UINT_MAX;
							break;
						}
						next = rowmap + *next;
					}
					gstoreFdwSpinUnlockHashSlot(gs_desc, hash);
				}
				pos += sizeof(char) + 2 * sizeof(uint32);
				break;
			case 'R':	/* Remove PK */
				Assert(hash_index != NULL);
				if (is_commit)
				{
					/* remove rowid from the index on commit */
					uint32	   *rowmap = &hash_index->slots[hash_index->nslots];
					uint32		hash  = *((uint32 *)(pos + 1));
					uint32		rowid = *((uint32 *)(pos + 5));
					uint32	   *next;

					gstoreFdwSpinLockHashSlot(gs_desc, hash);
					next = &hash_index->slots[hash % hash_index->nslots];
					while (*next < hash_index->nrooms)
					{
						if (*next == rowid)
						{
							*next = rowmap[rowid];
							rowmap[rowid] = UINT_MAX;
							break;
						}
						next = rowmap + *next;
					}
					gstoreFdwSpinUnlockHashSlot(gs_desc, hash);
				}
				pos += sizeof(char) + 2 * sizeof(uint32);
				break;
			default:
				elog(FATAL, "wrong undo log entry tag '%c'", *pos);
		}
	}
	Assert(pos <= gs_undo->buf.data + gs_undo->buf.len);
}

/*
 * gstoreFdwXactCallback
 */
static void
gstoreFdwXactCallback(XactEvent event, void *arg)
{
#if 0
	elog(INFO, "XactCallback: ev=%s xid=%u top-xid=%u",
		 event == XACT_EVENT_COMMIT  ? "XACT_EVENT_COMMIT" :
		 event == XACT_EVENT_ABORT   ? "XACT_EVENT_ABORT"  :
		 event == XACT_EVENT_PREPARE ? "XACT_EVENT_PREPARE" :
		 event == XACT_EVENT_PRE_COMMIT ? "XACT_EVENT_PRE_COMMIT" :
		 event == XACT_EVENT_PRE_PREPARE ? "XACT_EVENT_PRE_PREPARE" : "????",
		 GetCurrentTransactionIdIfAny(),
		 GetTopTransactionIdIfAny());
#endif
	if (event == XACT_EVENT_PRE_COMMIT)
	{
		dlist_iter		iter;
		TransactionId	xid = GetCurrentTransactionIdIfAny();
		int				hindex = xid % GSTORE_UNDO_LOGS_NSLOTS;

		dlist_foreach(iter, &gstore_undo_logs_slots[hindex])
		{
			GpuStoreUndoLogs *gs_undo
				= dlist_container(GpuStoreUndoLogs, chain, iter.cur);
			if (gs_undo->xid_top == xid)
				__gstoreFdwXactOnPreCommit(gs_undo);
		}
	}
	else if (event == XACT_EVENT_COMMIT ||
			 event == XACT_EVENT_ABORT)
	{
		dlist_mutable_iter iter;
		TransactionId	xid = GetCurrentTransactionIdIfAny();
		int				hindex = xid % GSTORE_UNDO_LOGS_NSLOTS;

		dlist_foreach_modify(iter, &gstore_undo_logs_slots[hindex])
		{
			GpuStoreUndoLogs *gs_undo
				= dlist_container(GpuStoreUndoLogs, chain, iter.cur);
			if (gs_undo->xid_top == xid)
			{
				__gstoreFdwXactFinalize(gs_undo, event == XACT_EVENT_COMMIT);
				dlist_delete(&gs_undo->chain);
				pfree(gs_undo->buf.data);
				pfree(gs_undo);
			}
		}
		/* Wake up other backends blocked by row-level lock */
		ConditionVariableBroadcast(&gstore_shared_head->row_lock_cond);
	}
}

/* ----------------------------------------------------------------
 *
 * SQL functions...
 *
 * ---------------------------------------------------------------- */
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
		if (RelationIsGstoreFdw(frel))
		{
			/*
			 * Ensure the supplied FDW options are reasonable,
			 * and create base/redo files preliminary.
			 */
			gstoreFdwLookupGpuStoreDesc(frel);
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
pgstrom_gstore_fdw_apply_redo(PG_FUNCTION_ARGS)
{
	Oid				ftable_oid = PG_GETARG_OID(0);
	Relation		frel = table_open(ftable_oid, AccessShareLock);
	GpuStoreDesc   *gs_desc = gstoreFdwLookupGpuStoreDesc(frel);
	CUresult		rc;
	
	rc = gstoreFdwApplyRedoDeviceBuffer(frel, gs_desc->gs_sstate);

	table_close(frel, NoLock);

	PG_RETURN_INT32((int)rc);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_apply_redo);

Datum
pgstrom_gstore_fdw_compaction(PG_FUNCTION_ARGS)
{
	Oid			ftable_oid = PG_GETARG_OID(0);
	bool		with_host_buffer = PG_GETARG_BOOL(1);
	Relation	frel;
	CUresult	rc;

	frel = table_open(ftable_oid, (with_host_buffer 
								   ? AccessExclusiveLock
								   : AccessShareLock));
#if 0
	if (with_host_buffer)
		do host buffer compaction...;
#endif
	rc = gstoreFdwInvokeCompaction(frel, false);

	table_close(frel, NoLock);

	PG_RETURN_INT32((int)rc);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_compaction);

Datum
pgstrom_gstore_fdw_sysattr_in(PG_FUNCTION_ARGS)
{
	GstoreFdwSysattr *sysatt;
	char	   *str = PG_GETARG_CSTRING(0);
	char	   *buf = pstrdup(str);
	char	   *end = buf + strlen(buf) - 1;
	char	   *saveptr = NULL;
	char	   *tok;
	int			count;
	unsigned long val;

	if (buf[0] != '(' || *end != ')')
		elog(ERROR, "invalid input [%s]", str);
	*end = '\0';

	sysatt = palloc0(sizeof(GstoreFdwSysattr));
	for (tok = strtok_r(buf+1, ",", &saveptr), count=0;
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &saveptr), count++)
	{
		val = strtoul(tok, &end, 10);
		if (val == ULONG_MAX || *end != '\0')
			elog(ERROR, "invalid input [%s]", str);
		switch (count)
		{
			case 0:
				sysatt->xmin = val;
				break;
			case 1:
				sysatt->xmax = val;
				break;
			case 2:
				sysatt->cid = val;
				break;
			default:
				elog(ERROR, "invalid input [%s]", str);
				break;
		}
	}
	if (count != 3)
		elog(ERROR, "invalid input [%s]", str);
	PG_RETURN_POINTER(sysatt);
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_sysattr_in);

Datum
pgstrom_gstore_fdw_sysattr_out(PG_FUNCTION_ARGS)
{
	GstoreFdwSysattr *sysatt = (GstoreFdwSysattr *)PG_GETARG_POINTER(0);

	PG_RETURN_CSTRING(psprintf("(%u,%u,%u)",
							   sysatt->xmin,
							   sysatt->xmax,
							   sysatt->cid));
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_sysattr_out);

/*
 * __gstoreFdwInvokeBackgroundCommand
 */
static CUresult
__gstoreFdwInvokeBackgroundCommand(GpuStoreBackgroundCommand *__lcmd, bool is_async)
{
	GpuStoreBackgroundCommand *cmd;
	dlist_node	   *dnode;
	CUresult		retval = CUDA_SUCCESS;

	SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
	for (;;)
	{
		if (!gstore_shared_head->background_latch)
		{
			SpinLockRelease(&gstore_shared_head->background_cmd_lock);
			return CUDA_ERROR_NOT_READY;
		}
		if (!dlist_is_empty(&gstore_shared_head->background_free_cmds))
			break;
		SpinLockRelease(&gstore_shared_head->background_cmd_lock);
		CHECK_FOR_INTERRUPTS();
		pg_usleep(20000L);		/* 20msec */
		SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
	}
	dnode = dlist_pop_head_node(&gstore_shared_head->background_free_cmds);
	cmd = dlist_container(GpuStoreBackgroundCommand, chain, dnode);
	memcpy(cmd, __lcmd, sizeof(GpuStoreBackgroundCommand));
	cmd->backend = (is_async ? NULL : MyLatch);
	cmd->retval = (CUresult) UINT_MAX;
	dlist_push_tail(&gstore_shared_head->background_cmd_queue,
					&cmd->chain);
	SpinLockRelease(&gstore_shared_head->background_cmd_lock);
	SetLatch(gstore_shared_head->background_latch);
	if (!is_async)
	{
		SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
		while (cmd->retval == (CUresult) UINT_MAX)
		{
			SpinLockRelease(&gstore_shared_head->background_cmd_lock);
			PG_TRY();
			{
				int		ev = WaitLatch(MyLatch,
									   WL_LATCH_SET |
									   WL_TIMEOUT |
									   WL_POSTMASTER_DEATH,
									   1000L,
									   PG_WAIT_EXTENSION);
				ResetLatch(MyLatch);
				if (ev & WL_POSTMASTER_DEATH)
					elog(FATAL, "unexpected postmaster dead");
				CHECK_FOR_INTERRUPTS();
			}
			PG_CATCH();
			{
				/*
				 * switch the request to async mode - because nobody can
				 * return the GpuStoreBackgroundCommand to free-list.
				 */
				SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
				if (cmd->retval == (CUresult) UINT_MAX)
					cmd->backend = NULL;
				else
					dlist_push_tail(&gstore_shared_head->background_free_cmds,
									&cmd->chain);
				SpinLockRelease(&gstore_shared_head->background_cmd_lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
		}
		retval = cmd->retval;
		SpinLockRelease(&gstore_shared_head->background_cmd_lock);
	}
	return retval;
}

static CUresult
gstoreFdwInvokeInitialLoad(Relation frel, bool is_async)
{
	GpuStoreBackgroundCommand lcmd;

	memset(&lcmd, 0, sizeof(GpuStoreBackgroundCommand));
	lcmd.database_oid = MyDatabaseId;
	lcmd.ftable_oid = RelationGetRelid(frel);
	lcmd.backend = (is_async ? NULL : MyLatch);
	lcmd.command = GSTORE_BACKGROUND_CMD__INITIAL_LOAD;

	return __gstoreFdwInvokeBackgroundCommand(&lcmd, is_async);
}

static CUresult
gstoreFdwInvokeApplyRedo(Oid ftable_oid, bool is_async,
						 uint64 end_pos, uint32 nitems)
{
	GpuStoreBackgroundCommand lcmd;

	memset(&lcmd, 0, sizeof(GpuStoreBackgroundCommand));
	lcmd.database_oid = MyDatabaseId;
	lcmd.ftable_oid = ftable_oid;
	lcmd.backend = (is_async ? NULL : MyLatch);
	lcmd.command = GSTORE_BACKGROUND_CMD__APPLY_REDO;
	lcmd.end_pos = end_pos;
	lcmd.nitems  = nitems;

	return __gstoreFdwInvokeBackgroundCommand(&lcmd, is_async);
}

static CUresult
gstoreFdwInvokeCompaction(Relation frel, bool is_async)
{
	GpuStoreBackgroundCommand lcmd;

	memset(&lcmd, 0, sizeof(GpuStoreBackgroundCommand));
	lcmd.database_oid = MyDatabaseId;
	lcmd.ftable_oid = RelationGetRelid(frel);
	lcmd.backend = (is_async ? NULL : MyLatch);
	lcmd.command = GSTORE_BACKGROUND_CMD__COMPACTION;

	return __gstoreFdwInvokeBackgroundCommand(&lcmd, is_async);
}

/*
 * GstoreFdwOpenLogBackupFile
 */
static int
GstoreFdwOpenLogBackupFile(GpuStoreSharedState *gs_sstate)
{
	time_t		t = time(NULL);
	char	   *temp;
	char	   *path;
	struct tm	tm;
	int			rawfd;

	if (!gs_sstate->redo_log_backup_dir)
		return -1;
	
	temp = alloca(strlen(gs_sstate->redo_log_file) + 1);
	path = alloca(strlen(gs_sstate->redo_log_backup_dir) +
				  strlen(gs_sstate->redo_log_file) + 100);
	localtime_r(&t, &tm);
	strcpy(temp, gs_sstate->redo_log_file);
	sprintf(path, "%s/%s_%04d-%02d-%02d_%02d:%02d:%02d",
			gs_sstate->redo_log_backup_dir,
			basename(temp),
			tm.tm_year + 1900,
			tm.tm_mon,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec);
	rawfd = open(path, O_RDWR | O_CREAT | O_APPEND);
	if (rawfd < 0)
		elog(LOG, "failed to open('%s') for Redo-Log backup: %m", path);
	return rawfd;
}

/*
 * __gstoreFdwGetCudaModule
 */
static CUresult
__gstoreFdwGetCudaModule(CUmodule *p_cuda_module, cl_int cuda_dindex)
{
	static void		*cuda_fatbin_image = NULL;
	static CUmodule *cuda_modules_array = NULL;

	Assert(cuda_dindex >= 0 && cuda_dindex < numDevAttrs);
	if (!cuda_fatbin_image)
	{
		const char *path = PGSHAREDIR "/pg_strom/cuda_gstore.fatbin";
		int			rawfd;
		struct stat	stat_buf;
		ssize_t		nbytes;

		rawfd = open(path, O_RDONLY);
		if (rawfd < 0)
			elog(ERROR, "failed on open('%s'): %m", path);
		PG_TRY();
		{
			if (fstat(rawfd, &stat_buf) != 0)
				elog(ERROR, "failed on fstat('%s'): %m", path);
			cuda_fatbin_image = malloc(stat_buf.st_size);
			nbytes = __readFileSignal(rawfd, cuda_fatbin_image,
									  stat_buf.st_size, false);
			if (nbytes != stat_buf.st_size)
				elog(ERROR, "failed on __readFile('%s'): %m", path);
		}
		PG_CATCH();
		{
			if (cuda_fatbin_image)
			{
				free(cuda_fatbin_image);
				cuda_fatbin_image = NULL;
			}
			close(rawfd);
			PG_RE_THROW();
		}
		PG_END_TRY();
		close(rawfd);
	}
	if (!cuda_modules_array)
	{
		cuda_modules_array = calloc(numDevAttrs, sizeof(CUmodule));
		if (!cuda_modules_array)
			elog(ERROR, "out of memory");
	}
	if (cuda_modules_array[cuda_dindex] == NULL)
	{
		CUmodule	cuda_module;
		CUresult	rc;

		rc = cuModuleLoadFatBinary(&cuda_module, cuda_fatbin_image);
		if (rc != CUDA_SUCCESS)
		{
			elog(LOG, "failed on cuModuleLoadFatBinary: %s", errorText(rc));
			return rc;
		}
		cuda_modules_array[cuda_dindex] = cuda_module;
	}
	*p_cuda_module = cuda_modules_array[cuda_dindex];
	return CUDA_SUCCESS;
}

/*
 * __gstoreFdwCallKernelApplyRedo
 */
static CUresult
__gstoreFdwCallKernelApplyRedo(GpuStoreDesc *gs_desc, CUdeviceptr m_redo)
{
	size_t		nitems = ((kern_gpustore_redolog *)m_redo)->nitems;
	int			cuda_dindex = gs_desc->gs_sstate->cuda_dindex;
	CUmodule	cuda_module;
	CUfunction	kfunc_setup_owner;
	CUfunction	kfunc_apply_redo;
	CUresult	rc;
	int			phase;
	int			grid_sz, __grid_sz;
	int			block_sz, __block_sz;
	void	   *kern_args[4];

	rc = __gstoreFdwGetCudaModule(&cuda_module, cuda_dindex);
	if (rc != CUDA_SUCCESS)
		return rc;
	rc = cuModuleGetFunction(&kfunc_setup_owner, cuda_module,
							 "kern_gpustore_setup_owner");
	if (rc != CUDA_SUCCESS)
		return rc;
	rc = cuModuleGetFunction(&kfunc_apply_redo, cuda_module,
							 "kern_gpustore_apply_redo");
	if (rc != CUDA_SUCCESS)
		return rc;

	rc = __gpuOptimalBlockSize(&grid_sz,
							   &block_sz,
							   kfunc_setup_owner,
							   cuda_dindex, 0, 0);
	if (rc != CUDA_SUCCESS)
		return rc;
	rc = __gpuOptimalBlockSize(&__grid_sz,
                               &__block_sz,
							   kfunc_apply_redo,
							   cuda_dindex, 0, 0);
	if (rc != CUDA_SUCCESS)
		return rc;
	block_sz = Min(block_sz, __block_sz);
	grid_sz = Min3(grid_sz, __grid_sz, (nitems + block_sz - 1) / block_sz);
	
	/*
	 * setup-owner (phase-0) - clear the owner-id field
	 */
	phase = 0;
	kern_args[0] = &m_redo;
	kern_args[1] = &gs_desc->gpu_main_devptr;
	kern_args[2] = &gs_desc->gpu_extra_devptr;
	kern_args[3] = &phase;
	rc = cuLaunchKernel(kfunc_setup_owner,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		goto out_error;
	}

	/*
	 * setup-owner (phase-1) - assign largest owner-id for each rows modified
	 */
	phase = 1;
	rc = cuLaunchKernel(kfunc_setup_owner,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		goto out_error;
	}

	/*
	 * apply redo logs (phase-2) - apply INSERT/DELETE logs
	 */
	phase = 2;
	rc = cuLaunchKernel(kfunc_apply_redo,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		goto out_error;
	}

	/*
	 * setup-owner (phase-3) - assign largest owner-id of commit-logs
	 */
	phase = 3;
	rc = cuLaunchKernel(kfunc_setup_owner,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		goto out_error;
	}

	/*
	 * apply redo logs (phase-4) - apply COMMIT logs
	 */
	phase = 4;
	rc = cuLaunchKernel(kfunc_apply_redo,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		goto out_error;
	}
	/* check status of the kernel execution status */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		return rc;

	//if out of memory, do compaction and retry.
	return CUDA_SUCCESS;

out_error:
	cuStreamSynchronize(CU_STREAM_PER_THREAD);
	return rc;
}

/*
 * GstoreFdwBackgrondInitialLoad
 */
static CUresult
GstoreFdwBackgroundInitialLoad(GpuStoreDesc *gs_desc)
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
	gs_sstate->gpu_main_size = schema->length;
	elog(LOG, "GstoreFdw: %s main buffer (sz: %lu) allocated",
		 ftable_name, schema->length);

	/* extra portion of the device buffer (if needed) */
	if (schema->has_varlena)
	{
		kern_data_extra *extra = (kern_data_extra *)
			((char *)schema + schema->extra_hoffset);

		rc = cuMemAlloc(&gs_desc->gpu_extra_devptr, extra->length);
		if (rc != CUDA_SUCCESS)
			goto error_1;
		rc = cuMemcpyHtoD(gs_desc->gpu_extra_devptr, extra, extra->length);
		if (rc != CUDA_SUCCESS)
			goto error_2;
		rc = cuIpcGetMemHandle(&gs_sstate->gpu_extra_mhandle,
							   gs_desc->gpu_extra_devptr);
		if (rc != CUDA_SUCCESS)
			goto error_2;
		gs_sstate->gpu_extra_size = extra->length;

		elog(LOG, "GstoreFdw: %s extra buffer (sz: %lu) allocated",
			 ftable_name, extra->length);
	}
	return CUDA_SUCCESS;

error_2:
	cuMemFree(gs_desc->gpu_extra_devptr);
error_1:
	cuMemFree(gs_desc->gpu_main_devptr);
error_0:
	gs_desc->gpu_extra_devptr = 0UL;
	gs_desc->gpu_main_devptr = 0UL;
	memset(&gs_sstate->gpu_main_mhandle, 0, sizeof(CUipcMemHandle));
	memset(&gs_sstate->gpu_extra_mhandle, 0, sizeof(CUipcMemHandle));

	return rc;
}

/*
 * GSTORE_BACKGROUND_CMD__APPLY_REDO command
 */
static CUresult
GstoreFdwBackgroundApplyRedoLog(GpuStoreDesc *gs_desc,
								uint64 end_pos, uint32 nitems)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	size_t		length;
	size_t		offset;
	int			index;
	uint64		head_pos;
	uint64		curr_pos;
	uint64		tail_pos = ULONG_MAX;
	kern_gpustore_redolog *h_redo;
	CUdeviceptr	m_redo = 0UL;
	CUresult	rc;

	/* device memory must be allocated */
	if (gs_desc->gpu_main_devptr == 0UL)
	{
		rc = GstoreFdwBackgroundInitialLoad(gs_desc);
		if (rc != CUDA_SUCCESS)
			return rc;
	}

	SpinLockAcquire(&gs_sstate->redo_pos_lock);
	if (end_pos <= gs_sstate->redo_read_pos)
	{
		/* nothing to do */
		SpinLockRelease(&gs_sstate->redo_pos_lock);
		return CUDA_SUCCESS;
	}
	head_pos = curr_pos = gs_sstate->redo_read_pos;
	Assert(end_pos <= gs_sstate->redo_write_pos);
	SpinLockRelease(&gs_sstate->redo_pos_lock);

	/*
	 * allocation of managed memory for kern_gpustore_redolog
	 * (index to log and redo-log itself)
	 */
	length = (MAXALIGN(offsetof(kern_gpustore_redolog,
								log_index[nitems])) +
			  MAXALIGN(end_pos - head_pos));
	rc = cuMemAllocManaged(&m_redo, length, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "failed on cuMemAllocManaged(%zu): %s",
			 length, errorText(rc));
		return rc;
	}
	h_redo = (kern_gpustore_redolog *)m_redo;
	memset(h_redo, 0, offsetof(kern_gpustore_redolog, log_index));
	h_redo->nrooms = nitems;
	h_redo->length = length;
	offset = MAXALIGN(offsetof(kern_gpustore_redolog,
							   log_index[nitems]));
	index = 0;
	while (curr_pos < end_pos && index < nitems)
	{
		uint64		file_pos = (curr_pos % gs_sstate->redo_log_limit);
		GstoreTxLogCommon *tx_log
			= (GstoreTxLogCommon *)(gs_desc->redo_mmap + file_pos);

		if (file_pos + offsetof(GstoreTxLogCommon, data) > gs_sstate->redo_log_limit)
		{
			tail_pos = curr_pos;
			curr_pos += (gs_sstate->redo_log_limit - file_pos);
			continue;
		}
		if ((tx_log->type & 0xffffff00U) != GSTORE_TX_LOG__MAGIC)
		{
			tail_pos = curr_pos;
			curr_pos += (gs_sstate->redo_log_limit - file_pos);
			continue;
		}
		Assert(tx_log->length == MAXALIGN(tx_log->length));
		memcpy((char *)h_redo + offset, tx_log, tx_log->length);
		h_redo->log_index[index++] = __kds_packed(offset);
		offset += tx_log->length;
		curr_pos += tx_log->length;
	}
	h_redo->nitems = index;
	h_redo->length = offset;
	
	/*
	 * Kick the kernel to apply REDO log
	 */
	pthreadRWLockWriteLock(&gs_sstate->gpu_bufer_lock);
	if (__gstoreFdwCallKernelApplyRedo(gs_desc, m_redo) == CUDA_SUCCESS)
	{
		SpinLockAcquire(&gs_sstate->redo_pos_lock);
		gs_sstate->redo_read_pos = curr_pos;
		SpinLockRelease(&gs_sstate->redo_pos_lock);
	}
	elog(LOG, "GPU Apply Redo (nitems=%u, length=%zu) pos %zu => %zu",
		 h_redo->nitems, h_redo->length,
		 head_pos, curr_pos);
	pthreadRWLockUnlock(&gs_sstate->gpu_bufer_lock);

	rc = cuMemFree(m_redo);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuMemFree: %s", errorText(rc));

	/*
	 * Write out REDO-Log to the backup file, if any
	 */
	if (gs_desc->redo_backup_fdesc >= 0)
	{
		uint64		file_pos = (head_pos % gs_sstate->redo_log_limit);
		ssize_t		rv, nbytes;

		if (tail_pos == ULONG_MAX)
		{
			nbytes = curr_pos - head_pos;
			rv = __writeFileSignal(gs_desc->redo_backup_fdesc,
								   gs_desc->redo_mmap + file_pos,
								   nbytes, false);
			if (rv != nbytes)
			{
				elog(LOG, "failed on writes of Redo-Log backup");
				close(gs_desc->redo_backup_fdesc);
				gs_desc->redo_backup_fdesc = -1;
			}
		}
		else
		{
			nbytes = tail_pos - head_pos;
			rv = __writeFileSignal(gs_desc->redo_backup_fdesc,
								   gs_desc->redo_mmap + file_pos,
								   nbytes, false);
			if (rv != nbytes)
			{
				elog(LOG, "failed on writes of Redo-Log backup");
				close(gs_desc->redo_backup_fdesc);
				gs_desc->redo_backup_fdesc = -1;
			}
			else
			{
				nbytes = curr_pos % gs_sstate->redo_log_limit;
				rv = __writeFileSignal(gs_desc->redo_backup_fdesc,
									   gs_desc->redo_mmap,
									   nbytes, false);
				if (rv != nbytes)
				{
					elog(LOG, "failed on writes of Redo-Log backup");
					close(gs_desc->redo_backup_fdesc);
					gs_desc->redo_backup_fdesc = -1;
				}
			}
		}
	}
	/*
	 * Switch Log Backup File if size exceeds the limit
	 */
	if (gs_desc->redo_backup_fdesc >= 0)
	{
		struct stat	stat_buf;

		if (fstat(gs_desc->redo_backup_fdesc, &stat_buf) != 0)
		{
			elog(WARNING, "failed on fstat(2): %m");
		}
		else if (stat_buf.st_size > gs_sstate->redo_log_backup_limit)
		{
			close(gs_desc->redo_backup_fdesc);
			gs_desc->redo_backup_fdesc = GstoreFdwOpenLogBackupFile(gs_sstate);
		}
	}
	return CUDA_SUCCESS;
}

/*
 * GSTORE_BACKGROUND_CMD__COMPACTION command
 */
static CUresult
GstoreFdwBackgroundCompation(GpuStoreDesc *gs_desc)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	kern_data_extra	h_extra;
	CUdeviceptr		m_new_extra;
	CUdeviceptr		m_temp;
	CUipcMemHandle	new_extra_mhandle;
	CUmodule		cuda_module;
	CUfunction		kfunc_compaction;
	CUresult		rc;
	int				grid_sz, block_sz;
	void		   *kern_args[4];

	pthreadRWLockWriteLock(&gs_sstate->gpu_bufer_lock);

	memset(&h_extra, 0, offsetof(kern_data_extra, data));
	memcpy(h_extra.signature, GPUSTORE_EXTRABUF_SIGNATURE, 8);
	h_extra.length = gs_sstate->gpu_extra_size;
	h_extra.usage  = offsetof(kern_data_extra, data);
	
	/* device memory must be allocated */
	if (gs_desc->gpu_main_devptr == 0UL)
	{
		rc = GstoreFdwBackgroundInitialLoad(gs_desc);
		if (rc != CUDA_SUCCESS)
			goto error_0;
	}

	/* main portion of the compaction */
	rc = cuMemAlloc(&m_new_extra, h_extra.length);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuMemAlloc(%zu): %s",
			 h_extra.length, errorText(rc));
		goto error_0;
	}
	rc = cuIpcGetMemHandle(&new_extra_mhandle, m_new_extra);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuIpcGetMemHandle: %s", errorText(rc));
		goto error_1;
	}
	rc = cuMemcpyHtoD(m_new_extra, &h_extra, offsetof(kern_data_extra, data));
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuMemcpyHtoD: %s", errorText(rc));
		goto error_1;
	}
	/* kick the compaction kernel */
	rc = __gstoreFdwGetCudaModule(&cuda_module, gs_sstate->cuda_dindex);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on __gstoreFdwGetCudaModule: %s", errorText(rc));
		goto error_1;
	}
	rc = cuModuleGetFunction(&kfunc_compaction, cuda_module,
							 "kern_gpustore_compaction");
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "GPU kernel function 'kern_gpustore_compaction' not found: %s",
			 errorText(rc));
		goto error_1;
	}
	rc = __gpuOptimalBlockSize(&grid_sz,
							   &block_sz,
							   kfunc_compaction,
							   gs_sstate->cuda_dindex, 0, 0);
	if (rc != CUDA_SUCCESS)
		goto error_1;
	kern_args[0] = &gs_desc->gpu_main_devptr;
	kern_args[1] = &gs_desc->gpu_extra_devptr;
	kern_args[2] = &m_new_extra;
	rc = cuLaunchKernel(kfunc_compaction,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		goto error_1;
	}
	/* check status of the kernel execution status */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		goto error_1;
	rc = cuMemcpyDtoH(&h_extra, m_new_extra, offsetof(kern_data_extra, data));
	if (rc != CUDA_SUCCESS)
		goto error_1;
	/* All Ok, swap old and new */
	m_temp = gs_desc->gpu_extra_devptr;
	gs_desc->gpu_extra_devptr = m_new_extra;
	m_new_extra = m_temp;
	memcpy(&gs_sstate->gpu_extra_mhandle,
		   &new_extra_mhandle, sizeof(CUipcMemHandle));

	elog(LOG, "Gstore_Fdw compaction: extra {length=%zu, usage=%zu}", h_extra.length, h_extra.usage);
	
error_1:
	cuMemFree(m_new_extra);
error_0:
	pthreadRWLockUnlock(&gs_sstate->gpu_bufer_lock);	
	return rc;
}

/*
 * gstoreFdwBgWorkerXXXX
 */
void
gstoreFdwBgWorkerBegin(void)
{
	SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
	gstore_shared_head->background_latch = MyLatch;
	SpinLockRelease(&gstore_shared_head->background_cmd_lock);
}

static CUresult
__gstoreFdwBgWorkerDispatchCommand(GpuStoreBackgroundCommand *cmd,
								   CUcontext *cuda_context_array)
{
	GpuStoreDesc *gs_desc;
	Oid			hkey[2];
	bool		found;
	int			cuda_dindex;
	CUresult	rc;
	
	/*
	 * Lookup GpuStoreDesc, but never construct GpuStoreSharedState
	 * unlike gstoreFdwLookupGpuStoreDesc. It is not our role.
	 */
	hkey[0] = cmd->database_oid;
	hkey[1] = cmd->ftable_oid;
	gs_desc = (GpuStoreDesc *) hash_search(gstore_desc_htab,
										   hkey,
										   HASH_ENTER,
										   &found);
	if (!found)
	{
		GpuStoreSharedState *gs_sstate = NULL;
		dlist_iter	iter;
		int			hindex;

		hindex = hash_any((const unsigned char *)hkey,
						  sizeof(hkey)) % GPUSTORE_SHARED_DESC_NSLOTS;
		SpinLockAcquire(&gstore_shared_head->gstore_sstate_lock[hindex]);
		dlist_foreach(iter, &gstore_shared_head->gstore_sstate_slot[hindex])
		{
			GpuStoreSharedState *temp = dlist_container(GpuStoreSharedState,
														hash_chain, iter.cur);
			if (temp->database_oid == cmd->database_oid &&
				temp->ftable_oid   == cmd->ftable_oid)
			{
				gs_sstate = temp;
				break;
			}
		}
		SpinLockRelease(&gstore_shared_head->gstore_sstate_lock[hindex]);

		if (!gs_sstate)
		{
			elog(LOG, "No GpuStoreSharedState found for database=%u, ftable=%u",
				 cmd->database_oid, cmd->ftable_oid);
			hash_search(gstore_desc_htab, hkey, HASH_REMOVE, NULL);
			return CUDA_ERROR_INVALID_VALUE;
		}
		gs_desc->gs_sstate = gs_sstate;
		gs_desc->base_mmap = NULL;
		gs_desc->base_mmap_revision = UINT_MAX;
		gs_desc->redo_mmap = NULL;
		gs_desc->gpu_main_devptr = 0UL;
		gs_desc->gpu_extra_devptr = 0UL;
		gs_desc->redo_backup_fdesc = GstoreFdwOpenLogBackupFile(gs_sstate);
	}
	if (!gstoreFdwSetupGpuStoreDesc(gs_desc, false))
		return CUDA_ERROR_MAP_FAILED;

	/* Switch CUDA Context to the target device */
	cuda_dindex = gs_desc->gs_sstate->cuda_dindex;
	Assert(cuda_dindex >= 0 && cuda_dindex < numDevAttrs);
	rc = cuCtxSetCurrent(cuda_context_array[cuda_dindex]);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "failed on cuCtxSetCurrent: %s", errorText(rc));
		return rc;
	}

	/* handle the command for each */
	switch (cmd->command)
	{
		case GSTORE_BACKGROUND_CMD__INITIAL_LOAD:
			rc = GstoreFdwBackgroundInitialLoad(gs_desc);
			break;
		case GSTORE_BACKGROUND_CMD__APPLY_REDO:
			rc = GstoreFdwBackgroundApplyRedoLog(gs_desc, cmd->end_pos, cmd->nitems);
			break;
		case GSTORE_BACKGROUND_CMD__COMPACTION:
			rc = GstoreFdwBackgroundCompation(gs_desc);
			break;
		default:
			elog(LOG, "Unsupported Gstore maintainer command: %d", cmd->command);
			rc = CUDA_ERROR_INVALID_VALUE;
			break;
	}
	cuCtxSetCurrent(NULL);

	return rc;
}

bool
gstoreFdwBgWorkerDispatch(CUcontext *cuda_context_array)
{
	GpuStoreBackgroundCommand *cmd;
	dlist_node	   *dnode;

	SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
	if (dlist_is_empty(&gstore_shared_head->background_cmd_queue))
	{
		SpinLockRelease(&gstore_shared_head->background_cmd_lock);
		return true;	/* gstoreFdw allows bgworker to sleep */
	}
	dnode = dlist_pop_head_node(&gstore_shared_head->background_cmd_queue);
	cmd = dlist_container(GpuStoreBackgroundCommand, chain, dnode);
	memset(&cmd->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&gstore_shared_head->background_cmd_lock);

	cmd->retval = __gstoreFdwBgWorkerDispatchCommand(cmd, cuda_context_array);

	SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
	if (cmd->backend)
	{
		/*
		 * A backend process who kicked GpuStore maintainer is waiting
		 * for the response. It shall check the retval, and return the
		 * GpuStoreBackgroundCommand to free list again.
		 */
		SetLatch(cmd->backend);
	}
	else
	{
		/*
		 * GpuStore maintainer was kicked asynchronously, so nobody is
		 * waiting for the response, thus, GpuStoreBackgroundCommand
		 * must be backed to the free list again.
		 */
		dlist_push_head(&gstore_shared_head->background_free_cmds,
						&cmd->chain);
	}
	SpinLockRelease(&gstore_shared_head->background_cmd_lock);
	return false;
}

bool
gstoreFdwBgWorkerIdleTask(CUcontext *cuda_context_array)
{
	int			hindex;
	bool		retval = true;

	for (hindex=0; hindex < GPUSTORE_SHARED_DESC_NSLOTS; hindex++)
	{
		slock_t	   *lock = &gstore_shared_head->gstore_sstate_lock[hindex];
		dlist_head *slot = &gstore_shared_head->gstore_sstate_slot[hindex];
		dlist_iter	iter;

		SpinLockAcquire(lock);
		dlist_foreach(iter, slot)
		{
			GpuStoreSharedState *gs_sstate;
			uint64		threshold;

			gs_sstate = dlist_container(GpuStoreSharedState,
										hash_chain, iter.cur);
			SpinLockAcquire(&gs_sstate->redo_pos_lock);
			threshold = (gs_sstate->gpu_update_interval * 1000000L +
						 gs_sstate->redo_last_timestamp);
			if (GetCurrentTimestamp () > threshold &&
				gs_sstate->redo_write_nitems > gs_sstate->redo_read_nitems)
			{
				dlist_head *cmd_flist = &gstore_shared_head->background_free_cmds;
				slock_t	   *cmd_lock  = &gstore_shared_head->background_cmd_lock;

				SpinLockAcquire(cmd_lock);
				if (!dlist_is_empty(cmd_flist))
				{
					GpuStoreBackgroundCommand *cmd;

					cmd = dlist_container(GpuStoreBackgroundCommand, chain,
										  dlist_pop_head_node(cmd_flist));
					memset(cmd, 0, sizeof(GpuStoreBackgroundCommand));
					cmd->database_oid = gs_sstate->database_oid;
					cmd->ftable_oid   = gs_sstate->ftable_oid;
					cmd->command      = GSTORE_BACKGROUND_CMD__APPLY_REDO;
					cmd->end_pos      = gs_sstate->redo_write_pos;
					cmd->nitems       = (gs_sstate->redo_write_nitems -
										 gs_sstate->redo_read_nitems);
					cmd->retval = (CUresult) UINT_MAX;
					dlist_push_tail(&gstore_shared_head->background_cmd_queue,
									&cmd->chain);
					retval = true;
				}
				SpinLockRelease(cmd_lock);
				gs_sstate->redo_last_timestamp = GetCurrentTimestamp();
			}
			SpinLockRelease(&gs_sstate->redo_pos_lock);
		}
		SpinLockRelease(lock);
	}
	return retval;
}

void
gstoreFdwBgWorkerEnd(void)
{
	SpinLockAcquire(&gstore_shared_head->background_cmd_lock);
	gstore_shared_head->background_latch = NULL;
	SpinLockRelease(&gstore_shared_head->background_cmd_lock);
}

/*
 * GstoreFdwStartupKicker
 */
void
GstoreFdwStartupKicker(Datum arg)
{
	char	   *database_name;
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
	SpinLockInit(&gstore_shared_head->background_cmd_lock);
	dlist_init(&gstore_shared_head->background_cmd_queue);
	dlist_init(&gstore_shared_head->background_free_cmds);
	for (i=0; i < lengthof(gstore_shared_head->__background_cmds); i++)
	{
		GpuStoreBackgroundCommand *cmd;

		cmd = &gstore_shared_head->__background_cmds[i];
		dlist_push_tail(&gstore_shared_head->background_free_cmds,
						&cmd->chain);
	}

	for (i=0; i < GPUSTORE_SHARED_DESC_NSLOTS; i++)
	{
		SpinLockInit(&gstore_shared_head->gstore_sstate_lock[i]);
		dlist_init(&gstore_shared_head->gstore_sstate_slot[i]);
	}
	ConditionVariableInit(&gstore_shared_head->row_lock_cond);
}

/*
 * pgstrom_init_gstore_fdw
 */
void
pgstrom_init_gstore_fdw(void)
{
	static bool		gstore_fdw_auto_preload;	/* GUC */
	FdwRoutine	   *r = &pgstrom_gstore_fdw_routine;
	HASHCTL			hctl;
	BackgroundWorker worker;
	int				i;

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
	/*
	 * Local hash slot for GpuStoreUndoLogs
	 */
	for (i=0; i < GSTORE_UNDO_LOGS_NSLOTS; i++)
		dlist_init(&gstore_undo_logs_slots[i]);

	/* GUC: gstore_fdw.enabled */
	DefineCustomBoolVariable("gstore_fdw.enabled",
							 "Enables the  planner's use of Gstore_Fdw",
							 NULL,
							 &gstore_fdw_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	
	/* GUC: gstore_fdw.auto_preload  */
	DefineCustomBoolVariable("gstore_fdw.auto_preload",
							 "Enables auto preload of GstoreFdw GPU buffers",
							 NULL,
							 &gstore_fdw_auto_preload,
							 true,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/*
	 * Background worker to load GPU store on startup
	 */
	if (gstore_fdw_auto_preload)
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

	/* transaction callbacks */
	RegisterXactCallback(gstoreFdwXactCallback, NULL);
}
