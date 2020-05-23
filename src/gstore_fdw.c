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
#include <libpmem.h>

#define GPUSTORE_SHARED_DESC_NSLOTS		107
typedef struct
{
	/* IPC to GPU Memory Synchronizer */
	Latch		   *gpumem_sync_latch;
	dlist_head		gpumem_sync_list;
	slock_t			gpumem_sync_lock;
	ConditionVariable gpumem_sync_cond;

	/* Hash slot for GpuStoreSharedState */
	slock_t			gstore_sstate_lock[GPUSTORE_SHARED_DESC_NSLOTS];
	dlist_head		gstore_sstate_slot[GPUSTORE_SHARED_DESC_NSLOTS];
} GpuStoreSharedHead;

#define GPUSTORE_BASEFILE_SIGNATURE		"@BASE-1@"
#define GPUSTORE_REDOLOG_SIGNATURE		"@REDO-1@"
#define GPUSTORE_HASHINDEX_SIGNATURE	"@HASH-1@"


typedef struct
{
	char		signature[8];
	uint64		checkpoint_offset;	/* valid only REDO log */
	char		ftable_name[NAMEDATALEN];
	kern_data_store schema;
} GpuStoreFileHead;

#define REDO_LOG_SYNC__NONE			1
#define REDO_LOG_SYNC__MSYNC		2
#define REDO_LOG_SYNC__PMEM			3

typedef struct
{
	dlist_node		hash_chain;
	dlist_node		sync_chain;		/* link to gpumem_sync_list */
	Oid				database_oid;
	Oid				ftable_oid;

	/* misc fields */
	cl_int			cuda_dindex;
	cl_long			max_num_rows;
	AttrNumber		primary_key;

	pthread_rwlock_t mmap_lock;
	uint32			mmap_revision;

	/* Base file buffer */
	pthread_rwlock_t base_file_lock;
	const char	   *base_file;
	uint32			hash_index_suffix;

	/* REDO Log state */
	const char	   *redo_log_file;
	int				redo_log_sync_method;
	size_t			redo_log_limit;
	pg_atomic_uint64 redo_log_pos;
	const char	   *redo_log_backup_dir;

	/* Device data store */
	pthread_rwlock_t gpu_bufer_lock;
	CUresult		gpu_sync_status;
	CUipcMemHandle	gpu_main_mhandle;		/* mhandle to main portion */
	CUipcMemHandle	gpu_extra_mhandle;		/* mhandle to extra portion */
	size_t			gpu_main_size;
	size_t			gpu_extra_size;
	Timestamp		gpu_update_timestamp;	/* time when last update */
	cl_int			gpu_update_interval;	/* fdw-option */
	cl_int			gpu_update_threshold;	/* fdw-option */
} GpuStoreSharedState;

typedef struct
{
	Oid				database_oid;
	Oid				ftable_oid;
	GpuStoreSharedState *gs_sstate;
	/* host mapped buffers */
	uint32			mmap_revision;
	GpuStoreFileHead *base_file_mmap;
	size_t			base_file_sz;
	GpuStoreFileHead *redo_log_mmap;
	size_t			redo_log_sz;
	int				redo_log_sync_method;
	/* fields below are valid only GpuUpdator */
	CUdeviceptr		gpu_main_devptr;
	CUdeviceptr		gpu_extra_devptr;
} GpuStoreDesc;

#define REDO_LOG_TYPE__INSERT		1
#define REDO_LOG_TYPE__UPDATE		2
#define REDO_LOG_TYPE__DELETE		3

/*
 * NOTE: Gstore_Fdw adds a 64-bit system column to save the timestamp on
 * INSERT/UPDATE/DELETE. Its upper 62bit hold nano-second scale timestamp
 * from the PostgreSQL epoch. Other bits are used to lock and removed flag.
 */
#define GSTORE_SYSATTR__LOCKED		0x0001UL
#define GSTORE_SYSATTR__REMOVED		0x0002UL
#define GSTORE_SYSATTR__MASK		0x0003UL

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

/*
 * A simple Hash-Index for PK
 */
typedef struct
{
	uint32		rowid;
	uint32		next;	/* next rowid, or UINT_MAX */
} GpuStoreHashEntry;

typedef struct
{
	char		signature[8];
	uint32		nslots;
	struct {
		pg_atomic_uint32 seqlock;
		uint32	next;
	} hslots[FLEXIBLE_ARRAY_MEMBER];
} GpuStoreHashIndex;






/* ---- static variables ---- */
static FdwRoutine	pgstrom_gstore_fdw_routine;
static GpuStoreSharedHead  *gstore_shared_head = NULL;
static HTAB *gstore_desc_htab = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;

/* ---- Forward declarations ---- */
Datum pgstrom_gstore_fdw_handler(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_validator(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_synchronize(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_compaction(PG_FUNCTION_ARGS);
Datum pgstrom_gstore_fdw_precheck_schema(PG_FUNCTION_ARGS);
void  GstoreFdwGpuBufferUpdater(Datum arg);

static GpuStoreDesc *gstoreFdwLookupGpuStoreDesc(Relation frel);










static void
GstoreGetForeignRelSize(PlannerInfo *root,
						RelOptInfo *baserel,
						Oid foreigntableid)
{}

static void
GstoreGetForeignPaths(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Oid foreigntableid)
{}

static ForeignScan *
GstoreGetForeignPlan(PlannerInfo *root,
					 RelOptInfo *baserel,
					 Oid foreigntableid,
					 ForeignPath *best_path,
					 List *tlist,
					 List *scan_clauses,
					 Plan *outer_plan)
{
	return NULL;
}

static void
GstoreBeginForeignScan(ForeignScanState *node, int eflags)
{}

static TupleTableSlot *
GstoreIterateForeignScan(ForeignScanState *node)
{
	return NULL;
}

static void
GstoreReScanForeignScan(ForeignScanState *node)
{}

static void
GstoreEndForeignScan(ForeignScanState *node)
{}

static bool
GstoreIsForeignScanParallelSafe(PlannerInfo *root,
								RelOptInfo *rel,
								RangeTblEntry *rte)
{
	return true;
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

	/*
	 * NOTE: we check option's syntax only at the validator.
	 */
	foreach (lc, options)
	{
		DefElem	   *def = (DefElem *) lfirst(lc);
		char	   *endp;

		if (catalog == AttributeRelationId)
		{
			if (strcmp(def->defname, "primary_key") == 0)
			{
				char   *token = defGetString(def);
				if (strcmp(token, "hash") != 0)
					elog(ERROR, "unknown index type \"%s\" for 'primary_key' option of Gstore_Fdw", token);
				continue;
			}
		}
		else if (catalog == RelationRelationId)
		{
			if (strcmp(def->defname, "gpu_device_id") == 0)
			{
				char   *token = defGetString(def);
				if (strtol(token, &endp, 10) < 0 || *endp != '\0')
					elog(ERROR, "gpu_device_id = '%s' is not valid", token);
				continue;
			}
			if (strcmp(def->defname, "max_num_rows") == 0)
			{
				char   *token = defGetString(def);

				if (strtol(token, &endp, 10) < 0 || *endp != '\0')
					elog(ERROR, "max_num_rows = '%s' is not valid", token);
				continue;
			}
			if (strcmp(def->defname, "base_file") == 0 ||
				strcmp(def->defname, "redo_log_file") == 0 ||
				strcmp(def->defname, "redo_log_backup_dir") == 0)
			{
				/* file pathnames are checked on runtime */
				continue;
			}
			if (strcmp(def->defname, "redo_log_sync_method") == 0)
			{
				char   *token = defGetString(def);

				if (strcmp(token, "pmem") == 0 &&
					strcmp(token, "msync") == 0 &&
					strcmp(token, "none") == 0)
					continue;
				elog(ERROR, "redo_log_sync_method = '%s' is not valid", token);
			}
			if (strcmp(def->defname, "redo_log_limit") == 0)
			{
				char   *token = defGetString(def);

				if (strtol(token, &endp, 10) > 0 &&
					(strcasecmp(endp, "k") == 0 ||
					 strcasecmp(endp, "kb") == 0 ||
					 strcasecmp(endp, "m") == 0 ||
					 strcasecmp(endp, "mb") == 0 ||
					 strcasecmp(endp, "g") == 0 ||
					 strcasecmp(endp, "gb") == 0))
					continue;	/* Ok */
				elog(ERROR, "redo_log_limit = '%s' is not valid", token);
			}
			if (strcmp(def->defname, "gpu_update_interval") == 0)
			{
				long	interval = strtol(defGetString(def), &endp, 10);

				if (interval <= 0 || interval > INT_MAX || *endp != '\0')
					elog(ERROR, "gpu_update_interval = '%s' is not valid",
						 defGetString(def));
				continue;
			}
			if (strcmp(def->defname, "gpu_update_threshold") == 0)
			{
				long	threshold = strtol(defGetString(def), &endp, 10);

				if (threshold <= 0 || threshold > INT_MAX || *endp != '\0')
					elog(ERROR, "gpu_update_threshold = '%s' is not valid",
						 defGetString(def));
			}
		}
		ereport(ERROR,
				(errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
				 errmsg("invalid option \"%s\"", def->defname)));
	}
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_validator);

static void
gstoreFdwExtractOptions(Relation frel,
						cl_int *p_cuda_dindex,
						cl_long *p_max_num_rows,
						const char **p_base_file,
						const char **p_redo_log_file,
						const char **p_redo_log_backup_dir,
						cl_int *p_redo_log_sync_method,
						size_t *p_redo_log_limit,
						cl_int *p_gpu_update_interval,
						cl_int *p_gpu_update_threshold,
						AttrNumber *p_primary_key)
{
	ForeignTable *ft = GetForeignTable(RelationGetRelid(frel));
	TupleDesc	tupdesc = RelationGetDescr(frel);
	ListCell   *lc;
	cl_int		cuda_dindex = -1;
	cl_long		max_num_rows = -1;
	const char *base_file = NULL;
	const char *redo_log_file = NULL;
	const char *redo_log_backup_dir = NULL;
	int			redo_log_sync_method = REDO_LOG_SYNC__PMEM; /* default: pmem */
	size_t		redo_log_limit = (512U << 20);	/* default: 512MB */
	cl_int		gpu_update_interval = 15;		/* default: 15s */
	cl_int		gpu_update_threshold = 40000;	/* default: 40k rows */
	AttrNumber	primary_key = -1;
	int			i, j;
	
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
			int		cuda_dindex = -1;

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
			if (max_num_rows < 0 || max_num_rows > UINT_MAX || *endp != '\0')
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
		else if (strcmp(def->defname, "redo_log_sync_method") == 0)
		{
			char   *value = defGetString(def);

			if (strcmp(value, "pmem") == 0)
				redo_log_sync_method = REDO_LOG_SYNC__PMEM;
			else if (strcmp(value, "msync") == 0)
				redo_log_sync_method = REDO_LOG_SYNC__MSYNC;
			else if (strcmp(value, "none") == 0)
				redo_log_sync_method = REDO_LOG_SYNC__NONE;
			else
				elog(ERROR, "redo_log_sync_method = '%s' is unknown", value);
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
		else
			elog(ERROR, "gstore_fdw: unknown option: %s", def->defname);
	}

	/*
	 * check attribute options
	 */
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc,j);
		List   *options = GetForeignColumnOptions(attr->attrelid,
												  attr->attnum);
		foreach (lc, options)
		{
			DefElem	   *def = lfirst(lc);

			if (strcmp(def->defname, "primary_key") == 0)
			{
				char   *token = defGetString(def);

				if (strcmp(token, "hash") != 0)
					elog(ERROR, "gstore_fdw: unknown index type: %s", token);
				if (primary_key < 0)
					primary_key = attr->attnum;
				else
					elog(ERROR, "gstore_fdw: primary_key specified twice");
			}
			else
				elog(ERROR, "gstore_fdw: unknown option '%s'", def->defname);
		}
	}

	/*
	 * Check Mandatory Options
	 */
	if (cuda_dindex < 0)
		elog(ERROR, "gpu_device_id must be specified");
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
	*p_redo_log_sync_method = redo_log_sync_method;
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
					   INT8OID,
					   -1,
					   0);
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
	cl_long		max_num_rows;
	const char *base_file;
	const char *redo_log_file;
	const char *redo_log_backup_dir;
	cl_int		redo_log_sync_method;
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
							&redo_log_sync_method,
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

	gs_sstate->cuda_dindex = cuda_dindex;
	gs_sstate->max_num_rows = max_num_rows;
	gs_sstate->primary_key = primary_key;

	pthreadRWLockInit(&gs_sstate->mmap_lock);
	do {
		gs_sstate->mmap_revision = random();
	} while (gs_sstate->mmap_revision == 0);
	gs_sstate->hash_index_suffix = 0;	/* set later, if any */
	gs_sstate->redo_log_sync_method = redo_log_sync_method;
	gs_sstate->redo_log_limit = redo_log_limit;

	pthreadRWLockInit(&gs_sstate->gpu_bufer_lock);
	gs_sstate->gpu_sync_status = CUDA_ERROR_NOT_INITIALIZED;
	gs_sstate->gpu_update_interval = gpu_update_interval;
	gs_sstate->gpu_update_threshold = gpu_update_threshold;

	return gs_sstate;
}








/*
 * gstoreFdwOpenBaseFile
 */
static File
gstoreFdwOpenBaseFile(Relation frel, const char *base_file, cl_uint nrooms)
{
	TupleDesc	__tupdesc = gstoreFdwDeviceTupleDesc(frel);
	File		fdesc;
	size_t		kds_sz = KDS_calculateHeadSize(__tupdesc);
	size_t		main_sz, extra_sz, sz;
	kern_data_store *schema;
	int			j, unitsz;

	/* setup schema definition according to the __tupdesc */
	schema = alloca(kds_sz);
	memset(schema, 0, kds_sz);
	init_kernel_data_store(schema,
						   __tupdesc,
						   0,	/* to be set later */
						   KDS_FORMAT_COLUMN,
						   nrooms);
	main_sz = kds_sz;
	extra_sz = 0;
	for (j=0; j < __tupdesc->natts; j++)
	{
		Form_pg_attribute	attr = tupleDescAttr(__tupdesc, j);
		kern_colmeta	   *cmeta = &schema->colmeta[j];

		sz = MAXALIGN(BITMAPLEN(nrooms));
		cmeta->nullmap_offset = __kds_packed(main_sz);
		cmeta->nullmap_length = __kds_packed(sz);
		main_sz += sz;
		if (attr->attlen > 0)
		{
			unitsz = att_align_nominal(attr->attlen, attr->attalign);
			sz = MAXALIGN(unitsz * nrooms);
			cmeta->values_offset = __kds_packed(main_sz);
			cmeta->values_length = __kds_packed(sz);
			main_sz += sz;
		}
		else if (attr->attlen == -1)
		{
			sz = MAXALIGN(sizeof(cl_uint) * nrooms);
			cmeta->values_offset = __kds_packed(main_sz);
			cmeta->values_length = __kds_packed(sz);
			main_sz += sz;

			unitsz = get_typavgwidth(attr->atttypid, attr->atttypmod);
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

	fdesc = PathNameOpenFile(base_file, O_RDWR);
	if (fdesc < 0)
	{
		GpuStoreFileHead *hbuf;
		size_t		hbuf_sz = offsetof(GpuStoreFileHead, schema) + kds_sz;

		if (errno != ENOENT)
			elog(ERROR, "failed on open('%s'): %m", base_file);
		fdesc = PathNameOpenFile(base_file, O_RDWR | O_CREAT | O_EXCL);
		if (fdesc < 0)
			elog(ERROR, "failed on creat('%s'): %m", base_file);

		/* setup header */
		hbuf = alloca(hbuf_sz);
		memset(hbuf, 0, hbuf_sz);
		memcpy(hbuf->signature, GPUSTORE_BASEFILE_SIGNATURE, sizeof(cl_ulong));
		strcpy(hbuf->ftable_name, RelationGetRelationName(frel));
		memcpy(&hbuf->schema, schema, kds_sz);
		if (__writeFile(FileGetRawDesc(fdesc), hbuf, hbuf_sz) != hbuf_sz)
			elog(ERROR, "failed on __writeFile('%s'): %m", base_file);

		/* expand file size */
		sz = offsetof(GpuStoreFileHead, schema) + main_sz + extra_sz;
		if (FileTruncate(fdesc, sz, WAIT_EVENT_DATA_FILE_TRUNCATE) < 0)
			elog(ERROR, "failed on FileTruncate('%s',%zu): %m",
				 base_file, sz);
	}
	else
	{
		GpuStoreFileHead *hbuf;
		size_t		hbuf_sz;
		
		if (__readFile(FileGetRawDesc(fdesc), hbuf, hbuf_sz) != hbuf_sz)
			elog(ERROR, "failed on __readFile('%s'): %m", base_file);
		/* check signature */
		if (memcmp(hbuf->signature, GPUSTORE_BASEFILE_SIGNATURE, 8) != 0)
			elog(ERROR, "fail '%s' has incompatible signature", base_file);
		/* check schema compatibility */
		if (hbuf->schema.length != schema->length ||
			hbuf->schema.nrooms != schema->nrooms ||
			hbuf->schema.ncols  != schema->ncols  ||
			hbuf->schema.format != schema->format ||
			hbuf->schema.tdtypeid != schema->tdtypeid ||
			hbuf->schema.tdtypmod != schema->tdtypmod ||
			hbuf->schema.table_oid != schema->table_oid ||
			hbuf->schema.nr_colmeta != schema->nr_colmeta ||
			memcmp(hbuf->schema.colmeta, schema->colmeta,
				   sizeof(kern_colmeta) * schema->nr_colmeta) != 0)
			elog(ERROR, "file '%s' is not compatible to foreign table %s",
				 base_file, RelationGetRelationName(frel));
	}
	pfree(__tupdesc);
	return fdesc;
}

static size_t
__ApplyRedoLogMain(GpuStoreFileHead *redo_head, char *redo_tail,
				   TupleDesc __tupdesc, File base_fdesc)
{
	GpuStoreFileHead *base_head;
	char	   *extra_buf;
	size_t		extra_sz;
	size_t		main_sz;
	Datum	   *values = alloca(sizeof(Datum) * __tupdesc->natts);
	bool	   *isnull = alloca(sizeof(bool)  * __tupdesc->natts);
	char	   *redo_pos;
	struct stat	stat_buf;

	/* mmap base file */
	if (fstat(FileGetRawDesc(base_fdesc), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", FilePathName(base_fdesc));
	base_head = __mmapFile(NULL, TYPEALIGN(PAGE_SIZE, stat_buf.st_size),
						   PROT_READ | PROT_WRITE,
						   MAP_SHARED | MAP_POPULATE,
						   FileGetRawDesc(base_fdesc), 0);
	if (base_head == MAP_FAILED)
		elog(ERROR, "failed on __mmapFile('%s'): %m",
			 FilePathName(base_fdesc));
	main_sz = offsetof(GpuStoreFileHead, schema) + base_head->schema.length;
	if (stat_buf.st_size < main_sz)
		elog(ERROR, "base file ('%s') header corruption",
			 FilePathName(base_fdesc));
	extra_buf = (char *)base_head + main_sz;
	extra_sz  = stat_buf.st_size - main_sz;

	/* read the redo log file */
	redo_pos = (char *)redo_head + redo_head->checkpoint_offset;
	while (redo_pos + sizeof(cl_ushort) < redo_tail)
	{
		kern_tupitem   *tupitem = (kern_tupitem *)redo_pos;
		kern_data_store *kds;
		HeapTupleData	tupData;
		cl_int			logkind;
		cl_uint			rowid;
		size_t			vlpos;
		size_t			sz;
		int				j;

		if (tupitem->t_len == 0)
			break;		/* no more items */
		sz = offsetof(kern_tupitem, htup) + tupitem->t_len;
		if (redo_pos + sz > redo_tail)
			break;		/* data corruption? */
		redo_pos += MAXALIGN(sz);

		logkind = gstore_get_tupitem_logkind(tupitem);
		rowid = gstore_get_tupitem_rowid(tupitem);
		vlpos = gstore_get_tupitem_vlpos(tupitem);
		if (rowid >= base_head->schema.nrooms)
			elog(NOTICE, "REDO rowid=%u, larger than max_num_rows=%u found in the REDO log file, skipped",
				 rowid, base_head->schema.nrooms);
		if (logkind != REDO_LOG_TYPE__INSERT &&
			logkind != REDO_LOG_TYPE__UPDATE &&
			logkind != REDO_LOG_TYPE__DELETE)
			elog(NOTICE, "REDO rowid=%u is neither INSERT, UPDATE nor DELETE",
				 rowid);

		/* expand the extra buffer on demand */
		if (vlpos > 0 && vlpos + tupitem->t_len > extra_sz)
		{
			sz = main_sz + vlpos + tupitem->t_len + (64UL << 20);
			sz = TYPEALIGN(PAGE_SIZE, sz);
			if (fallocate(FileGetRawDesc(base_fdesc), 0, 0, sz) != 0)
				elog(ERROR, "failed on fallocate('%s',%zu): %m",
					 FilePathName(base_fdesc), sz);
			base_head = __mremapFile(base_head, sz);
			if (base_head == MAP_FAILED)
				elog(ERROR, "failed on __mremapFile('%s',%zu): %m",
                     FilePathName(base_fdesc), sz);
			extra_buf = (char *)base_head + main_sz;
			extra_sz  = sz - main_sz;
		}
		kds = &base_head->schema;

		tupData.t_data = &tupitem->htup;
		heap_deform_tuple(&tupData, __tupdesc, values, isnull);
		for (j=0; j < __tupdesc->natts; j++)
		{
			kern_colmeta *cmeta = &base_head->schema.colmeta[j];
			char	   *nullmap, *vaddr;

			Assert(cmeta->nullmap_offset != 0 &&
				   cmeta->values_offset != 0);
			nullmap = (char *)kds + __kds_unpack(cmeta->nullmap_offset);
			vaddr   = (char *)kds + __kds_unpack(cmeta->values_offset);

			if (j == __tupdesc->natts - 1)
			{
				Timestamp	ts = DatumGetTimestamp(values[j]);
				/* internal system attribute (timestamp + delete flag) */
				Assert(cmeta->attbyval && cmeta->attlen == sizeof(Timestamp));
				if (isnull[j])
					elog(ERROR, "REDO log corruption? sysattr is NULL");
				nullmap[rowid >> 3] |= (1 << (rowid & 7));
				ts &= ~GSTORE_SYSATTR__MASK;
				if (logkind == REDO_LOG_TYPE__DELETE)
					ts |= GSTORE_SYSATTR__REMOVED;
				((Timestamp *)vaddr)[j] = ts;
			}
			else if (isnull[j])
			{
				if (j == __tupdesc->natts - 1)
					elog(ERROR, "REDO log corruption? timestamp is NULL");
				nullmap[rowid >> 3] |= (1 << (rowid & 7));
			}
			else
			{
				nullmap[rowid >> 3] |= (1 << (rowid & 7));

				if (cmeta->attbyval)
				{
					Assert(cmeta->attlen <= sizeof(cl_ulong));
					memcpy(vaddr + rowid * cmeta->attlen,
						   &values[j], cmeta->attlen);
				}
				else if (cmeta->attlen > 0)
				{
					memcpy(vaddr + rowid * cmeta->attlen,
						   DatumGetPointer(values[j]), cmeta->attlen);
				}
				else if (cmeta->attlen == -1)
				{
					char	   *vlbuf = extra_buf + vlpos;
					cl_uint		vl_sz = VARSIZE_ANY(values[j]);

					Assert(vlpos > 0);
					memcpy(vlbuf, DatumGetPointer(values[j]), vl_sz);
					((cl_uint *)vaddr)[rowid] = __kds_packed(vlpos);
					vlpos += MAXALIGN(vl_sz);
				}
				else
					elog(ERROR, "unexpected type definition");
			}
		}
	}
	__munmapFile(base_head);
	return redo_pos - (char *)redo_head;
}

static uint64
gstoreFdwApplyRedoLog(Relation frel,
					  File base_fdesc,
					  const char *redo_log_file,
					  size_t redo_log_file_limit)
{
	TupleDesc   __tupdesc = gstoreFdwDeviceTupleDesc(frel);
	File		redo_fdesc;
	GpuStoreFileHead *redo_head;
	char	   *redo_tail;
	size_t		redo_pos;
	struct stat	stat_buf;

	redo_fdesc = PathNameOpenFile(redo_log_file, O_RDWR);
	if (redo_fdesc < 0)
	{
		size_t	sz;
		/*
		 * Create a new REDO log file. Of course, new file shall not have
		 * any written-logs to be applied.
		 */
		if (errno != ENOENT)
			elog(ERROR, "failed on open('%s'): %m", redo_log_file);
		redo_fdesc = PathNameOpenFile(redo_log_file,
									  O_RDWR | O_CREAT | O_EXCL);
		if (redo_fdesc < 0)
			elog(ERROR, "failed on open('%s'): %m", redo_log_file);
		if (FileTruncate(redo_fdesc, redo_log_file_limit,
						 WAIT_EVENT_DATA_FILE_TRUNCATE) < 0)
			elog(ERROR, "failed on FileTruncate('%s'): %m", redo_log_file);

		sz = (offsetof(GpuStoreFileHead, schema) +
			  KDS_calculateHeadSize(__tupdesc));
		redo_head = alloca(sz);
		memset(redo_head, 0, sz);
		memcpy(redo_head->signature, GPUSTORE_REDOLOG_SIGNATURE, 8);
		redo_head->checkpoint_offset = sz;
		strcpy(redo_head->ftable_name, RelationGetRelationName(frel));
		init_kernel_data_store(&redo_head->schema,
							   __tupdesc,
							   0,
							   KDS_FORMAT_ROW,
							   UINT_MAX);
		if (__writeFile(FileGetRawDesc(redo_fdesc), redo_head, sz) != sz)
			elog(ERROR, "failed on __writeFile('%s'): %m", redo_log_file);
		FileClose(redo_fdesc);

		return sz;
	}
	/* mmap redo file */
	if (fstat(FileGetRawDesc(redo_fdesc), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", redo_log_file);
	if (stat_buf.st_size < redo_log_file_limit)
	{
		if (FileTruncate(redo_fdesc, redo_log_file_limit,
						 WAIT_EVENT_DATA_FILE_TRUNCATE) < 0)
			elog(ERROR, "failed on FileTruncate('%s'): %m", redo_log_file);
	}
	redo_head = __mmapFile(NULL, TYPEALIGN(PAGE_SIZE, stat_buf.st_size),
						   PROT_READ | PROT_WRITE,
						   MAP_SHARED | MAP_POPULATE,
						   FileGetRawDesc(redo_fdesc), 0);
	if (redo_head == MAP_FAILED)
		elog(ERROR, "failed on __mmapFile('%s'): %m", redo_log_file);
	redo_tail = (char *)redo_head + stat_buf.st_size;

	redo_pos = __ApplyRedoLogMain(redo_head, redo_tail,
								  __tupdesc, base_fdesc);
	/* ensure the changes at page-cache */
	if (fdatasync(FileGetRawDesc(base_fdesc)) != 0)
	{
		elog(WARNING, "failed on fdatasync('%s'): %m",
			 FilePathName(base_fdesc));
	}
	else
	{
		redo_head->checkpoint_offset = redo_pos;
		if (fdatasync(FileGetRawDesc(redo_fdesc)) != 0)
			elog(WARNING, "failed on fdatasync('%s'): %m",
				 FilePathName(redo_fdesc));
	}
	__munmapFile(redo_head);
	FileClose(redo_fdesc);

	return redo_pos;
}

static GpuStoreSharedState *
gstoreFdwCreateSharedState(Relation frel)
{
	GpuStoreSharedState *gs_sstate = gstoreFdwAllocSharedState(frel);
	File		base_fdesc;
	uint64		redo_pos;

	PG_TRY();
	{
		/* open or create base file */
		base_fdesc = gstoreFdwOpenBaseFile(frel,
										   gs_sstate->base_file,
										   gs_sstate->max_num_rows);
		/* apply REDO-log file */
		redo_pos = gstoreFdwApplyRedoLog(frel, base_fdesc,
										 gs_sstate->redo_log_file,
										 gs_sstate->redo_log_limit);
		pg_atomic_init_u64(&gs_sstate->redo_log_pos, redo_pos);

		//TODO: make hash-index 


	}
	PG_CATCH();
	{
		pfree(gs_sstate);
		PG_RE_THROW();
	}
	PG_END_TRY();
	FileClose(base_fdesc);
	/* Kick GPU memory synchronizer */
	//TODO:

	return gs_sstate;
}

static GpuStoreDesc *
__gstoreFdwSetupGpuStoreDesc(GpuStoreDesc *gs_desc)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;

	if (gs_desc->mmap_revision != gs_sstate->mmap_revision)
	{
		void	   *mmap_ptr;
		size_t		mmap_sz;
		File		fdesc;
		struct stat	stat_buf;

		PG_TRY();
		{
			int		redo_log_sync_method = gs_sstate->redo_log_sync_method;
			int		is_pmem;

			/* mmap Base-file */
			fdesc = PathNameOpenFile(gs_sstate->base_file, O_RDWR);
			if (fdesc < 0)
				elog(ERROR, "failed on open('%s'): %m",
					 gs_sstate->base_file);
			if (fstat(FileGetRawDesc(fdesc), &stat_buf) != 0)
				elog(ERROR, "failed on fstat('%s'): %m",
					 gs_sstate->base_file);
			mmap_sz = TYPEALIGN(PAGE_SIZE, stat_buf.st_size);
			mmap_ptr = mmap(NULL, mmap_sz,
							PROT_READ | PROT_WRITE,
							MAP_SHARED | MAP_POPULATE,
							FileGetRawDesc(fdesc), 0);
			if (mmap_ptr == MAP_FAILED)
				elog(ERROR, "failed on mmap('%s',%zu): %m",
					 gs_sstate->base_file, mmap_sz);
			gs_desc->base_file_mmap = mmap_ptr;
			gs_desc->base_file_sz   = mmap_sz;
			FileClose(fdesc);

			/* pmem_map REDO Log file */
			mmap_sz = TYPEALIGN(PAGE_SIZE, gs_sstate->redo_log_limit);
			mmap_ptr = pmem_map_file(gs_sstate->redo_log_file, mmap_sz,
									 0, 0600,
									 &gs_desc->redo_log_sz,
									 &is_pmem);
			if (!mmap_ptr)
				elog(ERROR, "failed on pmem_map_file('%s',%zu): %m",
					 gs_sstate->redo_log_file, mmap_sz);
			gs_desc->redo_log_mmap = mmap_ptr;
			/* degrade sync-method, if not persistent memory */
			if (!is_pmem && redo_log_sync_method == REDO_LOG_SYNC__PMEM)
				redo_log_sync_method = REDO_LOG_SYNC__MSYNC;
			gs_desc->redo_log_sync_method = redo_log_sync_method;
		}
		PG_CATCH();
		{
			if (gs_desc->base_file_mmap)
			{
				munmap(gs_desc->base_file_mmap,
					   gs_desc->base_file_sz);
				gs_desc->base_file_mmap = NULL;
			}
			if (gs_desc->redo_log_mmap)
			{
				pmem_unmap(gs_desc->redo_log_mmap,
						   gs_desc->redo_log_sz);
				gs_desc->redo_log_mmap = NULL;
			}
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
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
pgstrom_gstore_fdw_precheck_schema(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(pgstrom_gstore_fdw_precheck_schema);

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
 * GstoreFdwGpuBufferUpdater
 */
static bool gpumem_updater_got_signal = false;

static void
GpuBufferUpdaterSigTerm(SIGNAL_ARGS)
{
	int		saved_errno = errno;

	gpumem_updater_got_signal = true;

	pg_memory_barrier();

	SetLatch(MyLatch);

	errno = saved_errno;
}

static void
GpuBufferUpdaterSwitchCudaContext(cl_int cuda_dindex)
{
	static CUcontext   *cuda_contexts_array = NULL;
	static CUmodule	   *cuda_modules_array = NULL;
	CUdevice			cuda_device;
	CUcontext			cuda_context;
//	CUmodule			cuda_module;
	CUresult			rc;

	if (!cuda_contexts_array)
	{
		cuda_contexts_array = calloc(numDevAttrs, sizeof(CUcontext));
		if (!cuda_contexts_array)
			elog(ERROR, "out of memory");
	}
	if (!cuda_modules_array)
	{
		cuda_modules_array = calloc(numDevAttrs, sizeof(CUmodule));
		if (!cuda_modules_array)
			elog(ERROR, "out of memory");
	}
	Assert(cuda_dindex < numDevAttrs);

	if (!cuda_contexts_array[cuda_dindex])
	{
		rc = cuDeviceGet(&cuda_device, devAttrs[cuda_dindex].DEV_ID);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", errorText(rc));
		rc = cuCtxCreate(&cuda_context,
						 CU_CTX_SCHED_AUTO,
						 cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));
		//load module
	}
	else
	{
		rc = cuCtxPushCurrent(cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));
	}
}

/*
 * GpuBufferUpdaterInitialLoad
 *
 * must be called under the exclusive lock of gs_sstate->gpu_bufer_lock
 */
static bool
GpuBufferUpdaterInitialLoad(GpuStoreDesc *gs_desc)
{
	GpuStoreSharedState *gs_sstate = gs_desc->gs_sstate;
	kern_data_store *kds_head = &gs_desc->base_file_mmap->schema;
	size_t		base_sz = kds_head->length;
	size_t		extra_sz;
	CUresult	rc;

	/* main portion of the device buffer */
	rc = cuMemAlloc(&gs_desc->gpu_main_devptr, base_sz);
	if (rc != CUDA_SUCCESS)
		goto error_0;
	rc = cuMemcpyHtoD(gs_desc->gpu_main_devptr, kds_head, base_sz);
	if (rc != CUDA_SUCCESS)
		goto error_1;
	rc = cuIpcGetMemHandle(&gs_sstate->gpu_main_mhandle,
						   gs_desc->gpu_main_devptr);
	if (rc != CUDA_SUCCESS)
		goto error_1;

	/* extra portion of the device buffer */
	extra_sz = __kds_unpack(kds_head->usage); //right?
	rc = cuMemAlloc(&gs_desc->gpu_extra_devptr, extra_sz);
	if (rc != CUDA_SUCCESS)
		goto error_1;
	rc = cuMemcpyHtoD(gs_desc->gpu_extra_devptr,
					  (char *)kds_head + base_sz,
					  extra_sz);
	if (rc != CUDA_SUCCESS)
		goto error_2;
	rc = cuIpcGetMemHandle(&gs_sstate->gpu_extra_mhandle,
						   gs_desc->gpu_extra_devptr);
	if (rc != CUDA_SUCCESS)
		goto error_2;
	
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

void
GstoreFdwGpuBufferUpdater(Datum arg)
{
	CUcontext  *cuda_context_array;
	CUresult	rc;
	
	pqsignal(SIGTERM, GpuBufferUpdaterSigTerm);
	BackgroundWorkerUnblockSignals();

	/* force disable to use MPS */
	if (setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1) != 0)
		elog(ERROR, "failed on setenv: %m");
	/* init CUDA driver  */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", errorText(rc));
	/* CUDA context shall be built on demand */
	cuda_context_array = alloca(sizeof(CUcontext) * numDevAttrs);
	memset(cuda_context_array, 0, sizeof(CUcontext) * numDevAttrs);

	/*
	 * Event Loop
	 */
	gstore_shared_head->gpumem_sync_latch = MyLatch;
	pg_memory_barrier();
	while (!gpumem_updater_got_signal)
	{
		GpuStoreSharedState *gs_sstate;
		GpuStoreDesc   *gs_desc;
		dlist_node	   *dnode;

		SpinLockAcquire(&gstore_shared_head->gpumem_sync_lock);
		if (dlist_is_empty(&gstore_shared_head->gpumem_sync_list))
		{
			int		ev;

			SpinLockRelease(&gstore_shared_head->gpumem_sync_lock);
			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   5000L,
						   PG_WAIT_EXTENSION);
			ResetLatch(MyLatch);
			if (ev & WL_POSTMASTER_DEATH)
			{
				ConditionVariableBroadcast(&gstore_shared_head->gpumem_sync_cond);
				elog(FATAL, "unexpected Postmaster dead");
			}
			CHECK_FOR_INTERRUPTS();
			continue;
		}
		dnode = dlist_pop_head_node(&gstore_shared_head->gpumem_sync_list);
		gs_sstate = dlist_container(GpuStoreSharedState, sync_chain, dnode);
		memset(&gs_sstate->sync_chain, 0, sizeof(dlist_node));
		SpinLockRelease(&gstore_shared_head->gpumem_sync_lock);

		PG_TRY();
		{
			bool		found;
			
			GpuBufferUpdaterSwitchCudaContext(gs_sstate->cuda_dindex);

			pthreadRWLockReadLock(&gs_sstate->mmap_lock);
			pthreadRWLockWriteLock(&gs_sstate->gpu_bufer_lock);
			/* Lookup local mapping */
			gs_desc = (GpuStoreDesc *) hash_search(gstore_desc_htab,
												   gs_sstate,
												   HASH_ENTER,
												   &found);
			if (!found)
				gs_desc->gs_sstate = gs_sstate;
			__gstoreFdwSetupGpuStoreDesc(gs_desc);

			PG_TRY();
			{
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
				PG_RE_THROW();
			}
			PG_END_TRY();
			pthreadRWLockUnlock(&gs_sstate->gpu_bufer_lock);
			pthreadRWLockUnlock(&gs_sstate->mmap_lock);

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(LOG, "failed on cuCtxPopCurrent: %s", errorText(rc));
		}
		PG_CATCH();
		{
			ConditionVariableBroadcast(&gstore_shared_head->gpumem_sync_cond);
			PG_RE_THROW();
		}
		PG_END_TRY();

		ConditionVariableBroadcast(&gstore_shared_head->gpumem_sync_cond);
	}
	
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
	/*
	 * Background worker to synchronize GPU device memory
	 */
	memset(&worker, 0, sizeof(BackgroundWorker));
    snprintf(worker.bgw_name, sizeof(worker.bgw_name),
			 "GstoreFdw GPU Buffer Updater");
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = 1;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_strom");
	snprintf(worker.bgw_function_name, BGW_MAXLEN,
			 "GstoreFdwGpuBufferUpdater");
	worker.bgw_main_arg = 0;
	RegisterBackgroundWorker(&worker);

	/*
	 * Request for the static shared memory
	 */
	RequestAddinShmemSpace(STROMALIGN(sizeof(GpuStoreSharedHead)));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gstore_fdw;
}
