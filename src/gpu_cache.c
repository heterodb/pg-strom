/*
 * gpu_cache.c
 *
 * GPU data cache that syncronizes a PostgreSQL table
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
#include "cuda_gcache.h"

/*
 * GpuCacheBackgroundCommand
 */
#define GCACHE_BGWORKER_CMD__APPLY_REDO		'A'
#define GCACHE_BGWORKER_CMD__COMPACTION		'C'
#define GCACHE_BGWORKER_CMD__DROP_UNLOAD	'D'
typedef struct
{
	dlist_node  chain;
	Oid         database_oid;
	Oid         table_oid;
	Datum		signature;
	Latch      *backend;        /* MyLatch of the backend, if any */
	int         command;        /* one of GCACHE_BGWORKER_CMD__* */
	CUresult    retval;
	uint64      end_pos;        /* for APPLY_REDO */
} GpuCacheBackgroundCommand;

/*
 * GpuCacheSharedHead (shared structure; static)
 */
#define GPUCACHE_SHARED_DESC_NSLOTS		37
typedef struct
{
	/* hash slot for GpuCacheSharedState */
	slock_t		gcache_sstate_lock;
	dlist_head	gcache_sstate_slot[GPUCACHE_SHARED_DESC_NSLOTS];
	/* database name for preloading */
	int			preload_database_status;
	char		preload_database_name[NAMEDATALEN];
	/* IPC to GpuCache background workers */
	slock_t		bgworker_cmd_lock;
	dlist_head	bgworker_free_cmds;
	GpuCacheBackgroundCommand __bgworker_cmds[300];
	struct {
		Latch	   *latch;
		dlist_head	cmd_queue;
	} bgworkers[FLEXIBLE_ARRAY_MEMBER];
} GpuCacheSharedHead;

/*
 * GpuCacheRowIdHash
 */
#define GPUCACHE_DSM_MAGIC		0xfee1dead
typedef struct
{
	uint32		magic;
	slock_t		lock;
	cl_uint		nslots;
	cl_uint		nrooms;
	cl_uint		free_list;
	cl_uint		hash_slot[FLEXIBLE_ARRAY_MEMBER];
} GpuCacheRowIdHash;

/*
 * GpuCacheRowIdMap
 */
typedef struct
{
	cl_uint		next;
	ItemPointerData	ctid;
} GpuCacheRowId;

typedef struct
{
	uint32		magic;
	uint32		nrooms;
	GpuCacheRowId r_items[FLEXIBLE_ARRAY_MEMBER];
} GpuCacheRowIdMap;

/*
 * GpuCacheSharedState (shared structure; dynamic portable)
 */
typedef struct
{
	dlist_node		chain;
	Oid				database_oid;
	Oid				table_oid;
	Datum			signature;
	char			table_name[NAMEDATALEN];	/* for debugging */
	pg_atomic_uint32 refcnt;
	/* GPU memory store resources */
	int				initial_loading;
	dsm_handle		dsm_handle;
	/* GPU memory store parameters */
	int64			max_num_rows;
	int32			cuda_dindex;
	size_t			redo_buffer_size;
	size_t			gpu_sync_threshold;
	int32			gpu_sync_interval;

	/* Device resources */
	pthread_rwlock_t gpu_buffer_lock;
	CUipcMemHandle	gpu_main_mhandle;
	CUipcMemHandle	gpu_extra_mhandle;
	ssize_t			gpu_main_size;
	ssize_t			gpu_extra_size;
	CUdeviceptr		gpu_main_devptr;	/* valid only bgworker */
	CUdeviceptr		gpu_extra_devptr;	/* valid only bgworker */

	/* REDO buffer properties */
	slock_t			redo_lock;
	uint64			redo_write_timestamp;
	uint64			redo_write_nitems;
	uint64			redo_write_pos;
	uint64			redo_read_nitems;
	uint64			redo_read_pos;
	uint64			redo_sync_pos;

	/* schema definitions (KDS_FORMAT_COLUMN) */
	kern_data_extra	kds_extra;
	kern_data_store	kds_head;
} GpuCacheSharedState;

/*
 * GpuCacheDesc (GpuCache Descriptor per backend)
 */
typedef struct
{
	Oid					database_oid;
	Oid					table_oid;
	Datum				signature;
	GpuCacheSharedState *gc_sstate;	/* NULL, if no GpuCache is available */
	dlist_head			gc_handles;	/* list of GpuCacheHandle */
	dsm_segment		   *dsm_seg;	/* DSM */
	GpuCacheRowIdHash  *rowhash;	/* DSM */
	GpuCacheRowIdMap   *rowmap;		/* DSM */
} GpuCacheDesc;

/*
 * GpuCacheHandle (per-transaction state)
 */
typedef struct
{
	char			tag;
	ItemPointerData ctid;
	cl_uint			rowid;
} PendingRowIdItem;

typedef struct
{
	Oid				table_oid;
	Datum			signature;
	TransactionId	xid;
	GpuCacheDesc   *gc_desc;
	dlist_node		chain;				/* GpuCacheDesc->gc_handles */
	bool			drop_on_rollback;	/* newly configured */
	bool			drop_on_commit;		/* DROP or TRUNCATE */
	/* array of PendingRowIdItem */
	uint32			nitems;
	StringInfoData	buf;
} GpuCacheHandle;

/*
 * GpuCacheState - executor state object
 */
struct GpuCacheState
{
	pg_atomic_uint32	__gc_fetch_count;
	pg_atomic_uint32   *gc_fetch_count;
	GpuCacheDesc	   *gc_desc;
};

/* --- static variables --- */
static GpuCacheSharedHead *gcache_shared_head = NULL;
static HTAB		   *gcache_descriptors_htab = NULL;
static HTAB		   *gcache_handle_htab = NULL;
static HTAB		   *gcache_signatures_htab = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static object_access_hook_type object_access_next = NULL;
static void		  (*gpucache_xact_redo_next)(XLogReaderState *record) = NULL;
static void		  (*gpucache_heap_redo_next)(XLogReaderState *record) = NULL;
static CUmodule		gcache_cuda_module = NULL;
static CUfunction	gcache_kfunc_setup_owner = NULL;
static CUfunction	gcache_kfunc_apply_redo = NULL;
static CUfunction	gcache_kfunc_compaction = NULL;

/* --- function declarations --- */
static uint32	__gpuCacheAllocateRowId(GpuCacheDesc *gc_desc,
										ItemPointer ctid);
static void		__gpuCacheAppendLog(GpuCacheDesc *gc_desc,
									GCacheTxLogCommon *tx_log);

static CUresult gpuCacheInvokeApplyRedo(GpuCacheSharedState *gc_sstate,
										uint64 end_pos,
										bool is_async);
static CUresult gpuCacheInvokeCompaction(GpuCacheSharedState *gc_sstate,
										 bool is_async);
static CUresult gpuCacheInvokeDropUnload(GpuCacheSharedState *gc_sstate,
										 bool is_async);
void	GpuCacheStartupPreloader(Datum arg);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_sync_trigger);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_apply_redo);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_compaction);

/*
 * gpucache_sync_trigger_function_oid
 */
static Oid	__gpucache_sync_trigger_function_oid = InvalidOid;

static Oid
gpucache_sync_trigger_function_oid(void)
{
	if (!OidIsValid(__gpucache_sync_trigger_function_oid))
	{
		Oid		namespace_oid;
		oidvector argtypes;

		namespace_oid = get_namespace_oid("pgstrom", true);
		if (!OidIsValid(namespace_oid))
			return InvalidOid;

		memset(&argtypes, 0, sizeof(oidvector));
		SET_VARSIZE(&argtypes, offsetof(oidvector, values[0]));
		argtypes.ndim = 1;
		argtypes.dataoffset = 0;
		argtypes.elemtype = OIDOID;
		argtypes.dim1 = 0;
		argtypes.lbound1 = 0;

		__gpucache_sync_trigger_function_oid
			= GetSysCacheOid3(PROCNAMEARGSNSP,
							  Anum_pg_proc_oid,
							  CStringGetDatum("gpucache_sync_trigger"),
							  PointerGetDatum(&argtypes),
							  ObjectIdGetDatum(namespace_oid));
	}
	return __gpucache_sync_trigger_function_oid;
}

/*
 * parseSyncTriggerOptions
 */
typedef struct
{
	int			cuda_dindex;
	int32		gpu_sync_interval;
	size_t		gpu_sync_threshold;
	int64		max_num_rows;
	size_t		redo_buffer_size;
} GpuCacheOptions;

static void
__parseSyncTriggerOptions(char *config, GpuCacheOptions *gc_options)
{
	int			cuda_dindex = 0;				/* default: GPU0 */
	int			gpu_sync_interval = 5000000L;	/* default: 5sec = 5000000us */
	ssize_t		gpu_sync_threshold = -1;		/* default: auto */
	int64		max_num_rows = (10UL << 20);	/* default: 10M rows */
	ssize_t		redo_buffer_size = (160UL << 20);	/* default: 160MB */
	char	   *key, *value;
	char	   *saved;

	if (!config)
		goto out;

	for (key = strtok_r(config, ",", &saved);
		 key != NULL;
		 key = strtok_r(NULL,   ",", &saved))
	{
		value = strchr(key, '=');
		if (!value)
			elog(ERROR, "gpucache: options syntax error [%s]", key);
		*value++ = '\0';

		key = trim_cstring(key);
		value = trim_cstring(value);

		if (strcmp(key, "gpu_device_id") == 0)
		{
			int		i, gpu_device_id;
			char   *host;

			gpu_device_id = strtol(value, &host, 10);
			if (*host == '@')
			{
				char	name[512];

				host++;
				if (gethostname(name, sizeof(name)) != 0)
					elog(ERROR, "failed on gethostname: %m");
				if (strcmp(host, name) != 0)
					continue;
			}
			else if (*host != '\0')
			{
				elog(ERROR, "gpucache: invalid option [%s]=[%s]", key, value);
			}

			cuda_dindex = -1;
			for (i=0; i < numDevAttrs; i++)
			{
				if (devAttrs[i].DEV_ID == gpu_device_id)
				{
					cuda_dindex = i;
					break;
				}
			}

			if (cuda_dindex < 0)
				elog(ERROR, "gpucache: gpu_device_id (%d) not found", gpu_device_id);
		}
		else if (strcmp(key, "max_num_rows") == 0)
		{
			char   *end;

			max_num_rows = strtol(value, &end, 10);
			if (*end != '\0')
				elog(ERROR, "gpucache: invalid option [%s]=[%s]", key, value);
		}
		else if (strcmp(key, "gpu_sync_interval") == 0)
		{
			char   *end;

			gpu_sync_interval = strtol(value, &end, 10);
			if (*end != '\0')
				elog(ERROR, "gpucache: invalid option [%s]=[%s]", key, value);
			gpu_sync_interval *= 1000000L;	/* [sec -> us] */
		}
		else if (strcmp(key, "gpu_sync_threshold") == 0)
		{
			char   *end;

			gpu_sync_threshold = strtol(value, &end, 10);
			if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0)
				gpu_sync_threshold = (gpu_sync_threshold << 30);
			else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0)
				gpu_sync_threshold = (gpu_sync_threshold << 20);
			else if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0)
				gpu_sync_threshold = (gpu_sync_threshold << 10);
			else if (*end != '\0')
				elog(ERROR, "gpucache: invalid option [%s]=[%s]", key, value);
		}
		else if (strcmp(key, "redo_buffer_size") == 0)
		{
			char   *end;

			redo_buffer_size = strtol(value, &end, 10);
			if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0)
				redo_buffer_size = (redo_buffer_size << 30);
			else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0)
				redo_buffer_size = (redo_buffer_size << 20);
			else if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0)
				redo_buffer_size = (redo_buffer_size << 10);
			else if (*end != '\0')
				elog(ERROR, "gpucache: invalid option [%s]=[%s]", key, value);
			if (redo_buffer_size < (16UL << 20))
				elog(ERROR, "gpucache: 'redo_buffer_size' too small (%zu)",
					 redo_buffer_size);
		}
		else
		{
			elog(ERROR, "gpucache: unknown option [%s]=[%s]", key, value);
		}
	}
out:
	if (gc_options)
	{
		if (gpu_sync_threshold < 0)
			gpu_sync_threshold = redo_buffer_size / 4;
		if (gpu_sync_threshold > redo_buffer_size / 2)
			elog(ERROR, "gpucache: gpu_sync_threshold is too small");

		memset(gc_options, 0, sizeof(GpuCacheOptions));
		gc_options->cuda_dindex       = cuda_dindex;
		gc_options->gpu_sync_interval = gpu_sync_interval;
		gc_options->gpu_sync_threshold = gpu_sync_threshold;
		gc_options->max_num_rows      = max_num_rows;
		gc_options->redo_buffer_size  = redo_buffer_size;
	}
}

static inline void
parseSyncTriggerOptions(Relation rel, GpuCacheOptions *gc_options)
{
	char	   *config = NULL;
	TriggerDesc *trigdesc = rel->trigdesc;

	if (trigdesc && trigdesc->numtriggers > 0)
	{
		int		i;

		for (i=0; i < trigdesc->numtriggers; i++)
		{
			Trigger *trig = &trigdesc->triggers[i];

			if (trig->tgenabled &&
				trig->tgtype == (TRIGGER_TYPE_ROW |
								 TRIGGER_TYPE_AFTER |
								 TRIGGER_TYPE_INSERT |
								 TRIGGER_TYPE_DELETE |
								 TRIGGER_TYPE_UPDATE) &&
				trig->tgfoid == gpucache_sync_trigger_function_oid())
			{
				if (trig->tgnargs == 1)
				{
					if (trig->tgargs[0] != NULL)
						config = pstrdup(trig->tgargs[0]);
				}
				else if (trig->tgnargs > 1)
				{
					elog(ERROR, "gpucache: too much trigger arguments");
				}
			}
			else if (trig->tgenabled &&
					 trig->tgtype == (TRIGGER_TYPE_TRUNCATE) &&
					 trig->tgfoid == gpucache_sync_trigger_function_oid())
			{
				if (trig->tgnargs != 0)
					elog(ERROR, "gpucache: Too much trigger arguments");
			}
		}
	}
	__parseSyncTriggerOptions(config, gc_options);

	if (config)
		pfree(config);
}

/*
 * GpuCacheTableSignature - A base structure to calculate table signature
 */
typedef struct
{
	Oid		reltablespace;
	Oid		relfilenode;	/* if 0, cannot have gpucache */
	int16	relnatts;

	Oid		tg_sync_row;	/* if InvalidOid, cannot have gpucache */
	Oid		tg_sync_stmt;	/* if InvalidOid, cannot have gpucache */

	struct {
		Oid		atttypid;
		int32	atttypmod;
		bool	attnotnull;
		bool	attisdropped;
	} attrs[FLEXIBLE_ARRAY_MEMBER];
} GpuCacheTableSignatureBuffer;

typedef struct
{
	Oid		table_oid;
	Datum	signature;
} GpuCacheTableSignatureCache;

static Datum
__gpuCacheTableSignature(Relation rel, Oid trigger_oid_dropping)
{
	GpuCacheTableSignatureBuffer *sig;
	TupleDesc	tupdesc = RelationGetDescr(rel);
	int			j, natts = RelationGetNumberOfAttributes(rel);
	size_t		len = offsetof(GpuCacheTableSignatureBuffer, attrs[natts]);
	Form_pg_class rd_rel = RelationGetForm(rel);
	TriggerDesc *trigdesc = rel->trigdesc;
	bool		has_row_trigger = false;
	bool		has_stmt_trigger = false;

	sig = alloca(len);
	memset(sig, 0, len);

	/* pg_class related */
	if (rd_rel->relkind != RELKIND_RELATION &&
		rd_rel->relkind != RELKIND_PARTITIONED_TABLE)
		return 0UL;
	if (rd_rel->relfilenode == 0)
		return 0UL;
	sig->reltablespace	= rd_rel->reltablespace;
	sig->relfilenode	= rd_rel->relfilenode;
	sig->relnatts		= rd_rel->relnatts;

	/* sync trigger */
	if (!trigdesc)
		return 0UL;
	for (j=0; j < trigdesc->numtriggers; j++)
	{
		Trigger *trig = &trigdesc->triggers[j];

		if (OidIsValid(trigger_oid_dropping) &&
			trig->tgoid == trigger_oid_dropping)
		{
			/*
			 * On DROP TRIGGER, we check whether it affects to GpuCache
			 * by the signature as if it does not exist.
			 */
			continue;
		}
		else if (trig->tgenabled &&
				 trig->tgtype == (TRIGGER_TYPE_ROW |
								  TRIGGER_TYPE_AFTER |
								  TRIGGER_TYPE_INSERT |
								  TRIGGER_TYPE_DELETE |
								  TRIGGER_TYPE_UPDATE) &&
			trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (has_row_trigger)
				return 0UL;	/* misconfiguration */
			sig->tg_sync_row = trig->tgoid;
			has_row_trigger = true;
		}
		else if (trig->tgenabled &&
				 trig->tgtype == (TRIGGER_TYPE_TRUNCATE) &&
				 trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (has_stmt_trigger)
				return 0UL;	/* misconfiguration */
			sig->tg_sync_stmt = trig->tgoid;
			has_stmt_trigger = true;
		}
	}
	if (!has_row_trigger || !has_stmt_trigger)
		return 0UL;		/* no sync triggers */

	/* pg_attribute related */
	for (j=0; j < natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		sig->attrs[j].atttypid	= attr->atttypid;
		sig->attrs[j].atttypmod	= attr->atttypmod;
		sig->attrs[j].attnotnull = attr->attnotnull;
		sig->attrs[j].attisdropped = attr->attisdropped;
	}
	return hash_any((unsigned char *)sig, len) | 0x100000000UL;
}

static inline Datum
gpuCacheTableSignature(Relation rel)
{
	GpuCacheTableSignatureCache *entry;
	Oid			table_oid = RelationGetRelid(rel);
	bool		found;

	entry = hash_search(gcache_signatures_htab,
						&table_oid, HASH_ENTER, &found);
	if (!found)
	{
		Assert(entry->table_oid == table_oid);
		PG_TRY();
		{
			entry->signature = __gpuCacheTableSignature(rel, InvalidOid);
		}
		PG_CATCH();
		{
			hash_search(gcache_signatures_htab,
						&table_oid, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return entry->signature;
}

static Datum
gpuCacheTableSignatureSnapshot(Oid table_oid, Snapshot snapshot)
{
	GpuCacheTableSignatureBuffer *sig;
	Relation	srel;
	ScanKeyData	skey[2];
	SysScanDesc	sscan;
	HeapTuple	tuple;
	bool		has_row_trigger = false;
	bool		has_stmt_trigger = false;
	int			j, len;
	Form_pg_class rd_rel;

	/* pg_class */
	srel = table_open(RelationRelationId, AccessShareLock);
	ScanKeyInit(&skey[0],
				Anum_pg_class_oid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(table_oid));
	sscan = systable_beginscan(srel, ClassOidIndexId,
							   true, snapshot, 1, skey);
	tuple = systable_getnext(sscan);
	if (!HeapTupleIsValid(tuple))
		goto no_gpu_cache;
	rd_rel = (Form_pg_class) GETSTRUCT(tuple);
	if (rd_rel->relkind != RELKIND_RELATION &&
		rd_rel->relkind != RELKIND_PARTITIONED_TABLE)
		goto no_gpu_cache;
	len = offsetof(GpuCacheTableSignatureBuffer, attrs[rd_rel->relnatts]);
	sig = alloca(len);
	memset(sig, 0, len);

	if (rd_rel->relfilenode == 0)
		goto no_gpu_cache;
	sig->reltablespace  = rd_rel->reltablespace;
	sig->relfilenode    = rd_rel->relfilenode;
	sig->relnatts       = rd_rel->relnatts;

	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	/* pg_trigger */
	srel = table_open(TriggerRelationId, AccessShareLock);
	ScanKeyInit(&skey[0],
				Anum_pg_trigger_tgrelid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(table_oid));
	sscan = systable_beginscan(srel, TriggerRelidNameIndexId,
							   true, snapshot, 1, skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Form_pg_trigger pg_trig = (Form_pg_trigger) GETSTRUCT(tuple);

		if (pg_trig->tgenabled &&
			pg_trig->tgtype == (TRIGGER_TYPE_ROW |
								TRIGGER_TYPE_INSERT |
								TRIGGER_TYPE_DELETE |
								TRIGGER_TYPE_UPDATE) &&
			pg_trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (has_row_trigger)
				goto no_gpu_cache;
			sig->tg_sync_row = PgTriggerTupleGetOid(tuple);
			has_row_trigger = true;
		}
		else if (pg_trig->tgenabled &&
				 pg_trig->tgtype == TRIGGER_TYPE_TRUNCATE &&
				 pg_trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (has_stmt_trigger)
				goto no_gpu_cache;
			sig->tg_sync_stmt = PgTriggerTupleGetOid(tuple);
			has_stmt_trigger = true;
		}
	}
	if (!has_row_trigger || !has_stmt_trigger)
		goto no_gpu_cache;
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	/* pg_attribute */
	srel = table_open(AttributeRelationId, AccessShareLock);
	ScanKeyInit(&skey[0],
				Anum_pg_attribute_attrelid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(table_oid));
	ScanKeyInit(&skey[1],
				Anum_pg_attribute_attnum,
				BTGreaterStrategyNumber, F_INT2GT,
				Int16GetDatum(0));
	sscan = systable_beginscan(srel, AttributeRelidNumIndexId,
							   true, snapshot, 2, skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Form_pg_attribute attr = (Form_pg_attribute) GETSTRUCT(tuple);

		Assert(attr->attnum > 0 && attr->attnum <= sig->relnatts);
		j = attr->attnum - 1;
		sig->attrs[j].atttypid  = attr->atttypid;
		sig->attrs[j].atttypmod = attr->atttypmod;
		sig->attrs[j].attnotnull = attr->attnotnull;
		sig->attrs[j].attisdropped = attr->attisdropped;
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	return hash_any((unsigned char *)sig, len) | 0x100000000UL;

no_gpu_cache:
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
	return 0UL;
}

/*
 * baseRelHasGpuCache
 */
bool
baseRelHasGpuCache(PlannerInfo *root, RelOptInfo *baserel)
{
	RangeTblEntry *rte = root->simple_rte_array[baserel->relid];
	bool		retval = false;

	if (rte->rtekind == RTE_RELATION &&
		(baserel->reloptkind == RELOPT_BASEREL ||
		 baserel->reloptkind == RELOPT_OTHER_MEMBER_REL))
	{
		GpuCacheTableSignatureCache *entry;
		Relation	rel;
		bool		found;

		entry = hash_search(gcache_signatures_htab,
							&rte->relid, HASH_ENTER, &found);
		if (!found)
		{
			PG_TRY();
			{
				rel = table_open(rte->relid, NoLock);
				entry->signature = __gpuCacheTableSignature(rel, InvalidOid);
				table_close(rel, NoLock);
			}
			PG_CATCH();
			{
				hash_search(gcache_signatures_htab,
							&rte->relid, HASH_REMOVE, NULL);
				PG_RE_THROW();
			}
			PG_END_TRY();
		}
		Assert(entry->table_oid == rte->relid);

		retval = (entry->signature != 0UL);
	}
	return retval;
}

/*
 * RelationHasGpuCache
 */
bool
RelationHasGpuCache(Relation rel)
{
	return (gpuCacheTableSignature(rel) != 0UL);
}

/*
 * __execGpuCacheInitLoad
 */
static void
__execGpuCacheInitLoad(GpuCacheDesc *gc_desc, Relation rel)
{
	TableScanDesc	scandesc;
	Snapshot		snapshot;
	HeapTuple		tuple;
	size_t			item_sz = 2048;
	GCacheTxLogInsert *item = palloc(item_sz);

	snapshot = RegisterSnapshot(GetLatestSnapshot());
	scandesc = table_beginscan(rel, snapshot, 0, NULL);
	while ((tuple = heap_getnext(scandesc, ForwardScanDirection)) != NULL)
	{
		size_t		sz = MAXALIGN(offsetof(GCacheTxLogInsert,
										   htup) + tuple->t_len);
		if (sz > item_sz)
		{
			item_sz = 2 * sz;
			item = repalloc(item, item_sz);
		}
		item->type = GCACHE_TX_LOG__INSERT;
		item->length = sz;
		item->timestamp = GetCurrentTimestamp();
		item->rowid = __gpuCacheAllocateRowId(gc_desc, &tuple->t_self);
		memcpy(&item->htup, tuple->t_data, tuple->t_len);
		HeapTupleHeaderSetXmin(&item->htup, GetCurrentTransactionId());
		HeapTupleHeaderSetXmax(&item->htup, InvalidTransactionId);
		HeapTupleHeaderSetCmin(&item->htup, InvalidCommandId);

		__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)item);

		CHECK_FOR_INTERRUPTS();
	}
	table_endscan(scandesc);
	UnregisterSnapshot(snapshot);

	pfree(item);
}

/*
 * __OnDetachGpuCacheDSMSegment
 */
static void
__OnDetachGpuCacheDSMSegment(dsm_segment *dsm_seg, Datum __gc_sstate)
{
	GpuCacheSharedState *gc_sstate = (GpuCacheSharedState *)__gc_sstate;

	SpinLockAcquire(&gcache_shared_head->gcache_sstate_lock);
	Assert(pg_atomic_read_u32(&gc_sstate->refcnt) >= 2);
	if (pg_atomic_sub_fetch_u32(&gc_sstate->refcnt, 2) == 0)
	{
		elog(LOG, "pid=%u: Drop GpuCacheSharedState (%s:%lx)",
			 getpid(),
			 gc_sstate->table_name,
			 gc_sstate->signature);
		
		Assert(gc_sstate->gpu_main_devptr == 0UL &&
			   gc_sstate->gpu_extra_devptr == 0UL);
		Assert(gc_sstate->chain.prev != NULL &&
			   gc_sstate->chain.next != NULL);
		dlist_delete(&gc_sstate->chain);
		pfree(gc_sstate);
	}
	SpinLockRelease(&gcache_shared_head->gcache_sstate_lock);
}

/*
 * __createGpuCacheDSMSegment
 */
static void
__createGpuCacheDSMSegment(GpuCacheDesc *gc_desc)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	dsm_segment *dsm_seg;
	GpuCacheRowIdHash *rowhash;
    GpuCacheRowIdMap *rowmap;
	int64		i, nrooms = gc_sstate->max_num_rows;
	int64		nslots = (1.5 * (double)nrooms);
	size_t		sz;
	char	   *addr;

	nslots = Max(Min(nslots, UINT_MAX), 40000);
	sz = (PAGE_ALIGN(gc_sstate->redo_buffer_size) +
		  PAGE_ALIGN(offsetof(GpuCacheRowIdMap, r_items[nrooms])) +
		  PAGE_ALIGN(offsetof(GpuCacheRowIdHash, hash_slot[nslots])));
	dsm_seg = dsm_create(sz, 0);
	if (!dsm_seg)
		elog(ERROR, "failed on dsm_create(%zu,0)", sz);
	addr = dsm_segment_address(dsm_seg);
	/* REDO buffer */
	addr += PAGE_ALIGN(gc_sstate->redo_buffer_size);
	/* GpuCacheRowIdMap */
	rowmap = (GpuCacheRowIdMap *)addr;
	rowmap->magic = GPUCACHE_DSM_MAGIC;
	rowmap->nrooms = nrooms;
	for (i=1; i <= nrooms; i++)
	{
		rowmap->r_items[i-1].next = (i < nrooms ? i : UINT_MAX);
	}
	addr += PAGE_ALIGN(offsetof(GpuCacheRowIdMap, r_items[nrooms]));
	/* GpuCacheRowIdHash */
	rowhash = (GpuCacheRowIdHash *)addr;
	rowhash->magic = GPUCACHE_DSM_MAGIC;
	SpinLockInit(&rowhash->lock);
	rowhash->nslots = nslots;
	rowhash->nrooms = nrooms;
	rowhash->free_list = 0;
	memset(rowhash->hash_slot, ~0U, sizeof(cl_uint) * nslots);

	gc_desc->dsm_seg = dsm_seg;
	gc_desc->rowhash = rowhash;
	gc_desc->rowmap  = rowmap;

	on_dsm_detach(dsm_seg, __OnDetachGpuCacheDSMSegment,
				  PointerGetDatum(gc_sstate));
	gc_sstate->dsm_handle = dsm_segment_handle(dsm_seg);
	dsm_pin_mapping(dsm_seg);
	dsm_pin_segment(dsm_seg);
}

/*
 * __createGpuCacheSharedState
 */
static void
__createGpuCacheSharedState(GpuCacheDesc *gc_desc,
							Relation rel,
							GpuCacheOptions *gc_options)
{
	GpuCacheSharedState *gc_sstate;
	const char	   *table_name = RelationGetRelationName(rel);
	TupleDesc		tupdesc = RelationGetDescr(rel);
	kern_data_store *kds_head;
	kern_colmeta   *cmeta;
	uint32			nrooms;
	size_t			sz, off;
	size_t			extra_sz = 0;
	int				j, unitsz;

	/* allocation of GpuCacheSharedState */
	off = KDS_calculateHeadSize(tupdesc);
	sz = offsetof(GpuCacheSharedState, kds_head) + off;
	gc_sstate = MemoryContextAllocZero(TopSharedMemoryContext, sz);
	gc_sstate->database_oid = gc_desc->database_oid;
	gc_sstate->table_oid = gc_desc->table_oid;
	gc_sstate->signature = gc_desc->signature;
	strncpy(gc_sstate->table_name, table_name, NAMEDATALEN);
	pg_atomic_init_u32(&gc_sstate->refcnt, 2);
	gc_sstate->initial_loading = -1;	/* not yet */
	gc_sstate->dsm_handle = DSM_HANDLE_INVALID;	/* to be set later */

	Assert(gc_options->max_num_rows < UINT_MAX);
	gc_sstate->max_num_rows       = gc_options->max_num_rows;
	gc_sstate->cuda_dindex        = gc_options->cuda_dindex;
	gc_sstate->redo_buffer_size   = gc_options->redo_buffer_size;
	gc_sstate->gpu_sync_threshold = gc_options->gpu_sync_threshold;
	gc_sstate->gpu_sync_interval  = gc_options->gpu_sync_interval;

	pthreadRWLockInit(&gc_sstate->gpu_buffer_lock);
	SpinLockInit(&gc_sstate->redo_lock);

	/* init schema definition in KDS_FORMAT_COLUMN */
	kds_head = &gc_sstate->kds_head;
	nrooms = gc_options->max_num_rows;
	init_kernel_data_store(kds_head,
						   tupdesc,
						   0,	/* to be set later */
						   KDS_FORMAT_COLUMN,
						   nrooms);
	kds_head->table_oid = RelationGetRelid(rel);
	Assert(kds_head->nr_colmeta > tupdesc->natts);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		cmeta = &kds_head->colmeta[j];
		if (!attr->attnotnull)
		{
			sz = MAXALIGN(BITMAPLEN(nrooms));
			cmeta->nullmap_offset = __kds_packed(off);
			cmeta->nullmap_length = __kds_packed(sz);
			off += sz;
		}

		if (attr->attlen > 0)
		{
			unitsz = att_align_nominal(attr->attlen,
									   attr->attalign);
			sz = MAXALIGN(unitsz * nrooms);
			cmeta->values_offset = __kds_packed(off);
			cmeta->values_length = __kds_packed(sz);
			off += sz;
		}
		else if (attr->attlen == -1)
		{
			sz = MAXALIGN(sizeof(uint32) * nrooms);
			cmeta->values_offset = __kds_packed(off);
			cmeta->values_length = __kds_packed(sz);
			off += sz;
			unitsz = get_typavgwidth(attr->atttypid,
									 attr->atttypmod);
			extra_sz += MAXALIGN(unitsz) * nrooms;
		}
		else
		{
			pfree(gc_sstate);
			elog(ERROR, "unexpected type length (%d) at %s.%s",
				 attr->attlen,
				 RelationGetRelationName(rel),
				 NameStr(attr->attname));
		}
	}
	/* system column */
	cmeta = &kds_head->colmeta[kds_head->nr_colmeta - 1];
	sz = MAXALIGN(cmeta->attlen * nrooms);
	cmeta->values_offset = __kds_packed(off);
	cmeta->values_length = __kds_packed(sz);
	off += sz;

	kds_head->length = off;

	if (extra_sz > 0)
	{
		/* 25% margin */
		extra_sz += extra_sz / 4;
		extra_sz += offsetof(kern_data_extra, data);
	}
	gc_sstate->kds_extra.length = extra_sz;
	gc_sstate->kds_extra.usage  = 0;

	gc_desc->gc_sstate = gc_sstate;
}

/*
 * __attachGpuCacheSharedState
 */
static void
__attachGpuCacheSharedState(GpuCacheDesc *gc_desc,
							GpuCacheSharedState *gc_sstate,
							bool abort_on_error)
{
	GpuCacheRowIdHash *rowhash;
	GpuCacheRowIdMap *rowmap;
	dsm_segment	*dsm_seg;
	char		*base;
	char		*addr;

	dsm_seg = dsm_attach(gc_sstate->dsm_handle);
	if (!dsm_seg)
	{
		elog(abort_on_error ? ERROR : WARNING,
			 "gpucache: failed on dsm_attach");
		return;
	}
	addr = base = dsm_segment_address(dsm_seg);
	/* redo log buffer */
	addr += PAGE_ALIGN(gc_sstate->redo_buffer_size);
	/* GpuCacheRowIdMap */
	rowmap = (GpuCacheRowIdMap *)addr;
	Assert(rowmap->magic == GPUCACHE_DSM_MAGIC &&
		   rowmap->nrooms == gc_sstate->max_num_rows);
	addr += PAGE_ALIGN(offsetof(GpuCacheRowIdMap,
								r_items[rowmap->nrooms]));
	/* GpuCacheRowIdHash */
	rowhash = (GpuCacheRowIdHash *)addr;
	Assert(rowhash->magic == GPUCACHE_DSM_MAGIC);
	addr += PAGE_ALIGN(offsetof(GpuCacheRowIdHash,
								hash_slot[rowhash->nslots]));
	/* sanity checks */
	if ((size_t)(addr - base) > dsm_segment_map_length(dsm_seg))
	{
		dsm_detach(dsm_seg);
		elog(abort_on_error ? ERROR : WARNING,
			 "gpucache: DSM segment is smaller than the required");
		return;
	}
	on_dsm_detach(dsm_seg, __OnDetachGpuCacheDSMSegment,
				  PointerGetDatum(gc_sstate));
	dsm_pin_mapping(dsm_seg);
	/* ok, all green */
	pg_atomic_fetch_add_u32(&gc_sstate->refcnt, 2);
	gc_desc->gc_sstate = gc_sstate;
	gc_desc->dsm_seg = dsm_seg;
	gc_desc->rowhash = rowhash;
	gc_desc->rowmap  = rowmap;
}

/*
 * __setupGpuCacheDesc
 */
static void
__setupGpuCacheDesc(GpuCacheDesc *gc_desc,
					Relation rel,		/* maybe NULL, if bgworker */
					GpuCacheOptions *gc_options)
{
	GpuCacheSharedState *gc_sstate = NULL;
	Datum			hvalue;
	int				hindex;
	dlist_head	   *slot;
	dlist_iter		iter;
	slock_t		   *lock;

	/* init fields */
	gc_desc->gc_sstate = NULL;
	dlist_init(&gc_desc->gc_handles);
	gc_desc->dsm_seg = NULL;
	gc_desc->rowhash = NULL;
	gc_desc->rowmap  = NULL;

	/* lookup relevant GpuCacheSharedState */
	hvalue = hash_any((unsigned char *)gc_desc,
					  offsetof(GpuCacheDesc, signature) + sizeof(Datum));
	hindex = hvalue % GPUCACHE_SHARED_DESC_NSLOTS;
	slot = &gcache_shared_head->gcache_sstate_slot[hindex];
	lock = &gcache_shared_head->gcache_sstate_lock;
retry:
	SpinLockAcquire(lock);
	dlist_foreach(iter, slot)
	{
		gc_sstate = dlist_container(GpuCacheSharedState, chain, iter.cur);

		if (gc_sstate->database_oid == gc_desc->database_oid &&
			gc_sstate->table_oid    == gc_desc->table_oid &&
			gc_sstate->signature    == gc_desc->signature)
		{
			if ((pg_atomic_read_u32(&gc_sstate->refcnt) & 1) == 0)
			{
				/*
				 * If refcnt is even number, GpuCacheSharedState is already
				 * dropped, thus, no longer available.
				 */
				SpinLockRelease(lock);
				return;
			}
			if (rel && gc_sstate->initial_loading > 0)
			{
				/*
				 * positive initial_loading means someone is under
				 * initial-loading, but still in-progress.
				 */
				SpinLockRelease(lock);
				pg_usleep(5000L);	/* 5ms */
				CHECK_FOR_INTERRUPTS();
				goto retry;
			}
			if (rel && gc_sstate->initial_loading < 0)
			{
				/*
				 * negative initial_loading means someone has never
				 * tried initial-loading, or failed once.
				 */
				pg_atomic_fetch_add_u32(&gc_sstate->refcnt, 2);
				goto found_uninitialized;
			}
			/* ok, all green */
			SpinLockRelease(lock);
			__attachGpuCacheSharedState(gc_desc, gc_sstate, rel != NULL);
			return;
		}
	}
	/* quick bailout if caller don't want to create a new one */
	if (!rel)
	{
		gc_desc->gc_sstate = NULL;
		SpinLockRelease(lock);
		return;
	}

	/* Allocation of a new GpuCacheSharedState. */
	PG_TRY();
	{
		Assert(gc_options != NULL);
		__createGpuCacheSharedState(gc_desc, rel, gc_options);
		__createGpuCacheDSMSegment(gc_desc);
	}
	PG_CATCH();
	{
		if (gc_desc->gc_sstate)
			pfree(gc_desc->gc_sstate);
		SpinLockRelease(lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* move to the initial loading */
	dlist_push_tail(slot, &gc_desc->gc_sstate->chain);
found_uninitialized:
	gc_sstate = gc_desc->gc_sstate;
	gc_sstate->initial_loading = 1;
	SpinLockRelease(lock);

	/*
	 * Note that BgWorker may grab the GpuCacheSharedState that is
	 * still running the initial-loading process.
	 * So, we have to put the gc_sstate carefully.
	 */
	PG_TRY();
	{
		__execGpuCacheInitLoad(gc_desc, rel);
	}
	PG_CATCH();
	{
		/* rewind the buffer */
		SpinLockAcquire(&gc_sstate->redo_lock);
		gc_sstate->redo_write_nitems = 0;
		gc_sstate->redo_write_pos = 0;
		gc_sstate->redo_read_nitems = 0;
		gc_sstate->redo_read_pos = 0;
		gc_sstate->redo_sync_pos = 0;
		SpinLockRelease(&gc_sstate->redo_lock);
		/* release device memory, if any */
		gpuCacheInvokeDropUnload(gc_sstate, true);
		/*
		 * NOTE: dsm_detach() internally calls __OnDetachGpuCacheDSMSegment()
		 * that eventually drops the GpuCacheSharedState.
		 */
		dsm_detach(gc_desc->dsm_seg);
		gc_desc->gc_sstate = NULL;
		gc_desc->dsm_seg = NULL;
		gc_desc->rowhash = NULL;
		gc_desc->rowmap = NULL;
		PG_RE_THROW();

	}
	PG_END_TRY();
	/* ok, all done */
	SpinLockAcquire(lock);
	gc_sstate->initial_loading = 0;		/* ready now */
	pg_atomic_fetch_or_u32(&gc_sstate->refcnt, 1);	/* odd number - a valid entry */
	SpinLockRelease(lock);
}

/*
 * GetGpuCacheDesc
 */
static GpuCacheDesc *
GetGpuCacheDesc(Relation rel, Datum signature)
{
	GpuCacheDesc	hkey;
	GpuCacheDesc   *gc_desc;
	bool			found;

	Assert(signature != 0);
	hkey.database_oid = MyDatabaseId;
	hkey.table_oid = RelationGetRelid(rel);
	hkey.signature = signature;
	gc_desc = hash_search(gcache_descriptors_htab,
						  &hkey, HASH_ENTER, &found);
	if (!found)
	{
		GpuCacheOptions gc_options;

		PG_TRY();
		{
			parseSyncTriggerOptions(rel, &gc_options);
			__setupGpuCacheDesc(gc_desc, rel, &gc_options);
		}
		PG_CATCH();
		{
			hash_search(gcache_descriptors_htab,
						&hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	elog(LOG, "GetGpuCacheDesc: %s", RelationGetRelationName(rel));
	return (gc_desc->gc_sstate ? gc_desc : NULL);
}

/*
 * GetGpuCacheDescBgWorker - never create GpuCacheSharedState
 */
static GpuCacheDesc *
GetGpuCacheDescBgWorker(Oid database_oid,
						Oid table_oid,
						Datum signature)
{
	GpuCacheDesc	hkey;
	GpuCacheDesc   *gc_desc;
	bool			found;

	hkey.database_oid = database_oid;
	hkey.table_oid = table_oid;
	hkey.signature = signature;
	gc_desc = hash_search(gcache_descriptors_htab,
						  &hkey, HASH_ENTER, &found);
	if (!found)
	{
		__setupGpuCacheDesc(gc_desc, NULL, NULL);
		if (!gc_desc->gc_sstate)
		{
			hash_search(gcache_descriptors_htab,
						&hkey, HASH_REMOVE, NULL);
			return NULL;
		}
	}
	return gc_desc;
}

/*
 * ReleaseGpuCacheDescIfUnused
 */
static void
ReleaseGpuCacheDescIfUnused(GpuCacheDesc *gc_desc)
{
	GpuCacheSharedState *gc_sstate;

	if (!dlist_is_empty(&gc_desc->gc_handles))
		return;

	gc_sstate = gc_desc->gc_sstate;
	if (gc_sstate)
	{
		if ((pg_atomic_read_u32(&gc_sstate->refcnt) & 1) == 0)
		{
			dsm_detach(gc_desc->dsm_seg);
			hash_search(gcache_descriptors_htab, gc_desc, HASH_REMOVE, NULL);
		}
	}
	else
	{
		Assert(!gc_desc->dsm_seg);
		hash_search(gcache_descriptors_htab, gc_desc, HASH_REMOVE, NULL);
	}
}

/*
 * __gpuCacheAllocateRowId
 */
static uint32
__gpuCacheAllocateRowId(GpuCacheDesc *gc_desc, ItemPointer ctid)
{
	GpuCacheRowIdHash *rowhash = gc_desc->rowhash;
	GpuCacheRowIdMap *rowmap = gc_desc->rowmap;
	GpuCacheRowId *r_item;
	cl_uint		hash;
	cl_uint		index;
	cl_uint		rowid;

	hash = hash_any((unsigned char *)ctid, sizeof(ItemPointerData));
	index = hash % rowhash->nslots;

	SpinLockAcquire(&rowhash->lock);
	if (rowhash->free_list >= rowhash->nrooms)
	{
		SpinLockRelease(&rowhash->lock);
		elog(ERROR, "No more rooms in the GPU Store");
	}
	rowid = rowhash->free_list;
	r_item = &rowmap->r_items[rowid];
	
	Assert(!ItemPointerIsValid(&r_item->ctid));
	rowhash->free_list = r_item->next;

	ItemPointerCopy(ctid, &r_item->ctid);
	r_item->next = rowhash->hash_slot[index];
	rowhash->hash_slot[index] = rowid;

	SpinLockRelease(&rowhash->lock);

	return rowid;
}

/*
 * __gpuCacheLookupRowId / __gpuCacheReleaseRowId
 */
static uint32
__gpuCacheLookupOrReleaseRowId(GpuCacheDesc *gc_desc,
							   ItemPointer ctid,
							   bool release_rowid)
{
	GpuCacheRowIdHash *rowhash = gc_desc->rowhash;
	GpuCacheRowIdMap *rowmap = gc_desc->rowmap;
	GpuCacheRowId *r_item;
	GpuCacheRowId *r_prev;
	cl_uint		hash;
	cl_uint		index;
	cl_uint		rowid;

	hash = hash_any((unsigned char *)ctid, sizeof(ItemPointerData));
	index = hash % rowhash->nslots;

	SpinLockAcquire(&rowhash->lock);
	for (rowid = rowhash->hash_slot[index], r_prev = NULL;
		 rowid < rowhash->nrooms;
		 rowid = r_item->next, r_prev = r_item)
	{
		r_item = &rowmap->r_items[rowid];

		if (ItemPointerEquals(&r_item->ctid, ctid))
		{
			if (release_rowid)
			{
				if (!r_prev)
					rowhash->hash_slot[index] = r_item->next;
				else
					r_prev->next = r_item->next;
				ItemPointerSetInvalid(&r_item->ctid);
				r_item->next = rowhash->free_list;
				rowhash->free_list = rowid;
			}
			SpinLockRelease(&rowhash->lock);
			return rowid;
		}
	}
	SpinLockRelease(&rowhash->lock);

	return UINT_MAX;
}

static uint32
__gpuCacheLookupRowId(GpuCacheDesc *gc_desc, ItemPointer ctid)
{
	return __gpuCacheLookupOrReleaseRowId(gc_desc, ctid, false);
}

static inline void
__gpuCacheReleaseRowId(GpuCacheDesc *gc_desc, PendingRowIdItem *pitem)
{
	uint32	__rowid;

	__rowid = __gpuCacheLookupOrReleaseRowId(gc_desc, &pitem->ctid, true);
	Assert(__rowid == pitem->rowid);
}

static inline void
__initGpuCacheHandle(GpuCacheHandle *gc_handle)
{
	gc_handle->gc_desc = NULL;
	gc_handle->drop_on_rollback = false;
	gc_handle->drop_on_commit = false;
	gc_handle->nitems = 0;
	memset(&gc_handle->buf, 0, sizeof(StringInfoData));
}

/*
 * GetGpuCacheHandle
 */
static GpuCacheHandle *
GetGpuCacheHandle(Relation rel)
{
	GpuCacheHandle	hkey;
	GpuCacheHandle *gc_handle;
	bool			found;

	hkey.table_oid = RelationGetRelid(rel);
	hkey.signature = gpuCacheTableSignature(rel);
	hkey.xid       = GetCurrentTransactionIdIfAny();
	if (hkey.signature == 0UL || hkey.xid == InvalidTransactionId)
		return NULL;

	gc_handle = hash_search(gcache_handle_htab,
							&hkey, HASH_ENTER, &found);
	if (!found)
	{
		__initGpuCacheHandle(gc_handle);
		PG_TRY();
		{
			GpuCacheDesc   *gc_desc = GetGpuCacheDesc(rel, hkey.signature);

			if (gc_desc)
			{
				gc_handle->gc_desc = gc_desc;
				dlist_push_tail(&gc_desc->gc_handles, &gc_handle->chain);
			}
		}
		PG_CATCH();
		{
			hash_search(gcache_handle_htab,
						&hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return (gc_handle->gc_desc ? gc_handle : NULL);
}

/*
 * ReleaseGpuCacheHandle
 */
static void
ReleaseGpuCacheHandle(GpuCacheHandle *gc_handle, bool normal_commit)
{
	GpuCacheDesc *gc_desc = gc_handle->gc_desc;

	if ((gc_handle->drop_on_rollback && !normal_commit) ||
		(gc_handle->drop_on_commit   &&  normal_commit))
	{
		GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;

		dsm_unpin_segment(gc_sstate->dsm_handle);
		pg_atomic_fetch_and_u32(&gc_sstate->refcnt, 0xfffffffeU);
		gpuCacheInvokeDropUnload(gc_sstate, true);
	}
	else
	{
		char   *pos = gc_handle->buf.data;
		uint32	count;

		for (count=0; count < gc_handle->nitems; count++)
		{
			PendingRowIdItem *pitem = (PendingRowIdItem *)pos;

			switch (pitem->tag)
			{
				case 'I':	/* INSERT */
					if (!normal_commit)
						__gpuCacheReleaseRowId(gc_desc, pitem);
					pos += sizeof(PendingRowIdItem);
					break;

				case 'D':	/* DELETE */
					if (normal_commit)
						__gpuCacheReleaseRowId(gc_desc, pitem);
					pos += sizeof(PendingRowIdItem);
					break;

				default:
					elog(FATAL, "Bug? GpuCacheHandle has corruption");
			}
		}
		Assert(pos <= gc_handle->buf.data + gc_handle->buf.len);
	}
	elog(LOG, "ReleaseGpuCacheHandle: %s:%lx xid=%u",
		 gc_desc->gc_sstate->table_name,
		 gc_desc->gc_sstate->signature,
		 gc_handle->xid);

	/* detach GpuCacheDesc */
	dlist_delete(&gc_handle->chain);
	ReleaseGpuCacheDescIfUnused(gc_desc);
	/* cleanup */
	if (gc_handle->buf.data)
		pfree(gc_handle->buf.data);
	hash_search(gcache_handle_htab, &gc_handle, HASH_REMOVE, NULL);
}

/*
 * __gpuCacheAppendLog
 */
static void
__gpuCacheAppendLog(GpuCacheDesc *gc_desc, GCacheTxLogCommon *tx_log)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	char	   *redo_buffer = dsm_segment_address(gc_desc->dsm_seg);
	size_t		buffer_sz = gc_sstate->redo_buffer_size;
	uint64		offset;
	uint64		sync_pos;
	bool		append_done = false;

	Assert(tx_log->length == MAXALIGN(tx_log->length));
	for (;;)
	{
		SpinLockAcquire(&gc_sstate->redo_lock);
		Assert(gc_sstate->redo_write_pos >= gc_sstate->redo_read_pos &&
			   gc_sstate->redo_write_pos <= gc_sstate->redo_read_pos + buffer_sz &&
			   gc_sstate->redo_sync_pos >= gc_sstate->redo_read_pos &&
			   gc_sstate->redo_sync_pos <= gc_sstate->redo_write_pos);
		offset = gc_sstate->redo_write_pos % buffer_sz;
		/* rewind to the head */
		if (offset + tx_log->length > buffer_sz)
		{
			size_t	sz = buffer_sz - offset;

			/* oops, it looks overwrites... */
			if (gc_sstate->redo_write_pos + sz > gc_sstate->redo_read_pos + buffer_sz)
				goto skip;
			/* fill-up by zero */
			memset(redo_buffer + offset, 0, sz);
			gc_sstate->redo_write_pos += sz;
			offset = 0;
		}
		/* check overwrites */
		if ((gc_sstate->redo_write_pos +
			 tx_log->length) > gc_sstate->redo_read_pos + buffer_sz)
			goto skip;

		/* Ok, append the log item */
		memcpy(redo_buffer + offset, tx_log, tx_log->length);
		gc_sstate->redo_write_pos += tx_log->length;
		gc_sstate->redo_write_nitems++;
		gc_sstate->redo_write_timestamp = GetCurrentTimestamp();
		append_done = true;
	skip:
		/* 25% of REDO buffer is in-use. Async kick of GPU kernel */
		if (gc_sstate->redo_write_pos > (gc_sstate->redo_read_pos +
										 gc_sstate->gpu_sync_threshold))
		{
			sync_pos = gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
			SpinLockRelease(&gc_sstate->redo_lock);
			gpuCacheInvokeApplyRedo(gc_sstate, sync_pos, true);
		}
		else
		{
			SpinLockRelease(&gc_sstate->redo_lock);
		}
		if (append_done)
			break;
		pg_usleep(1000L);	/* 1ms wait */
	}
}

/*
 * __gpuCacheInsertLog
 */
static void
__gpuCacheInsertLog(HeapTuple tuple, GpuCacheHandle *gc_handle)
{
	GpuCacheDesc *gc_desc = gc_handle->gc_desc;
	GCacheTxLogInsert *item;
	PendingRowIdItem rlog;
	cl_uint		rowid;
	size_t		sz;

	/* Track RowId allocation */
	rowid = __gpuCacheAllocateRowId(gc_desc, &tuple->t_self);
	rlog.tag = 'I';
	rlog.ctid = tuple->t_self;
	rlog.rowid = rowid;
	appendBinaryStringInfo(&gc_handle->buf,
						   (char *)&rlog,
						   sizeof(PendingRowIdItem));
	gc_handle->nitems++;

	/* INSERT Log */
	sz = MAXALIGN(offsetof(GCacheTxLogInsert, htup) + tuple->t_len);
	item = alloca(sz);
	item->type = GCACHE_TX_LOG__INSERT;
	item->length = sz;
	item->timestamp = GetCurrentTimestamp();
	item->rowid = rowid;
	memcpy(&item->htup, tuple->t_data, tuple->t_len);
	HeapTupleHeaderSetXmin(&item->htup, GetCurrentTransactionId());
	HeapTupleHeaderSetXmax(&item->htup, InvalidTransactionId);
	HeapTupleHeaderSetCmin(&item->htup, InvalidCommandId);

	__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)item);
}

/*
 * __gpuCacheDeleteLog
 */
static void
__gpuCacheDeleteLog(HeapTuple tuple, GpuCacheHandle *gc_handle)
{
	GpuCacheDesc   *gc_desc = gc_handle->gc_desc;
	GCacheTxLogDelete item;
	PendingRowIdItem rlog;
	cl_uint		rowid;

	/* Track RowId release */
	rowid = __gpuCacheLookupRowId(gc_desc, &tuple->t_self);
	rlog.tag = 'D';
	rlog.ctid = tuple->t_self;
	rlog.rowid = rowid;
	appendBinaryStringInfo(&gc_handle->buf,
						   (char *)&rlog,
						   sizeof(PendingRowIdItem));
	gc_handle->nitems++;

	/* DELETE Log */
	item.type = GCACHE_TX_LOG__DELETE;
	item.length = MAXALIGN(sizeof(GCacheTxLogDelete));
	item.timestamp = GetCurrentTimestamp();
	item.rowid = rowid;
	item.xid = GetCurrentTransactionId();

	__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)&item);
}

/*
 * __gpuCacheTruncateLog
 */
static void
__gpuCacheTruncateLog(GpuCacheHandle *gc_handle)
{
	Assert(!gc_handle->drop_on_commit);

	gc_handle->drop_on_commit = true;
}

/*
 * pgstrom_gpucache_sync_trigger
 */
Datum
pgstrom_gpucache_sync_trigger(PG_FUNCTION_ARGS)
{
	TriggerData	   *trigdata = (TriggerData *) fcinfo->context;
	TriggerEvent	tg_event;
	GpuCacheHandle *gc_handle;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as trigger",
			 get_func_name(fcinfo->flinfo->fn_oid));

	tg_event = trigdata->tg_event;
	if (!TRIGGER_FIRED_AFTER(tg_event))
		elog(ERROR, "%s: must be called as AFTER ROW/STATEMENT",
			 get_func_name(fcinfo->flinfo->fn_oid));

	gc_handle = GetGpuCacheHandle(trigdata->tg_relation);
	if (!gc_handle)
		elog(ERROR, "gpucache is not configured for %s",
			 RelationGetRelationName(trigdata->tg_relation));
	if (gc_handle->buf.data == NULL)
		initStringInfoContext(&gc_handle->buf, CacheMemoryContext);

	if (TRIGGER_FIRED_FOR_ROW(tg_event))
	{
		/* FOR EACH ROW */
		if (TRIGGER_FIRED_BY_INSERT(tg_event))
		{
			__gpuCacheInsertLog(trigdata->tg_trigtuple, gc_handle);
		}
		else if (TRIGGER_FIRED_BY_UPDATE(trigdata->tg_event))
		{
			__gpuCacheDeleteLog(trigdata->tg_trigtuple, gc_handle);
			__gpuCacheInsertLog(trigdata->tg_newtuple, gc_handle);
		}
		else if (TRIGGER_FIRED_BY_DELETE(trigdata->tg_event))
		{
			__gpuCacheDeleteLog(trigdata->tg_trigtuple, gc_handle);
		}
		else
		{
			elog(ERROR, "gpucache: unexpected trigger event type (%u)", tg_event);
		}
	}
	else
	{
		/* FOR EACH STATEMENT */
		if (TRIGGER_FIRED_BY_TRUNCATE(tg_event))
		{
			__gpuCacheTruncateLog(gc_handle);
		}
		else
		{
			elog(ERROR, "gpucache: unexpected trigger event type (%u)", tg_event);
		}
	}
	PG_RETURN_POINTER(trigdata->tg_trigtuple);
}

/*
 * pgstrom_gpucache_apply_redo
 */
Datum
pgstrom_gpucache_apply_redo(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	Relation	rel;
	Datum		signature;
	CUresult	rc = CUDA_ERROR_INVALID_VALUE;

	rel = table_open(table_oid, RowExclusiveLock);
	signature = gpuCacheTableSignature(rel);
	if (signature != 0UL)
	{
		GpuCacheDesc *gc_desc = GetGpuCacheDesc(rel, signature);
		GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
		uint64		sync_pos;

		SpinLockAcquire(&gc_sstate->redo_lock);
		sync_pos = gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
		SpinLockRelease(&gc_sstate->redo_lock);

		rc = gpuCacheInvokeApplyRedo(gc_sstate, sync_pos, false);
	}
	table_close(rel, RowExclusiveLock);

	PG_RETURN_INT32(rc);
}

/*
 * pgstrom_gpucache_compaction
 */
Datum
pgstrom_gpucache_compaction(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	Relation	rel;
	Datum		signature;
	CUresult	rc = CUDA_ERROR_INVALID_VALUE;

	rel = table_open(table_oid, RowExclusiveLock);
	signature = gpuCacheTableSignature(rel);
	if (signature != 0UL)
	{
		GpuCacheDesc *gc_desc = GetGpuCacheDesc(rel, signature);

		rc = gpuCacheInvokeCompaction(gc_desc->gc_sstate, false);
	}
	table_close(rel, AccessShareLock);

	PG_RETURN_INT32(rc);
}

/* ---------------------------------------------------------------- *
 *
 * Executor callbacks
 *
 * ---------------------------------------------------------------- */
GpuCacheState *
ExecInitGpuCache(ScanState *ss, int eflags, Bitmapset *outer_refs)
{
	Relation		relation = ss->ss_currentRelation;
	Datum			signature;
	GpuCacheDesc   *gc_desc;
	GpuCacheState  *gcache_state;
	uint64			sync_pos;

	if (!relation)
		return NULL;

	signature = gpuCacheTableSignature(relation);
	if (signature == 0UL)
		return NULL;

	gc_desc = GetGpuCacheDesc(relation, signature);
	if (!gc_desc)
		return NULL;

	gcache_state = palloc0(sizeof(GpuCacheState));
	gcache_state->gc_fetch_count = &gcache_state->__gc_fetch_count;
	gcache_state->gc_desc = gc_desc;
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
	{
		GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;

		SpinLockAcquire(&gc_sstate->redo_lock);
		sync_pos = gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
		SpinLockRelease(&gc_sstate->redo_lock);

		gpuCacheInvokeApplyRedo(gc_sstate, sync_pos, false);
	}
	return gcache_state;
}

static inline pgstrom_data_store *
__ExecScanChunkGpuCache(GpuTaskState *gts, GpuCacheDesc *gc_desc)
{
	EState		   *estate = gts->css.ss.ps.state;
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	pgstrom_data_store *pds;
	size_t			head_sz;

	head_sz = KERN_DATA_STORE_HEAD_LENGTH(&gc_sstate->kds_head);
	pds = MemoryContextAllocZero(estate->es_query_cxt,
								 offsetof(pgstrom_data_store, kds) + head_sz);
	pg_atomic_init_u32(&pds->refcnt, 1);
	pds->gc_sstate = gc_sstate;
	memcpy(&pds->kds, &gc_sstate->kds_head, head_sz);

	return pds;
}

pgstrom_data_store *
ExecScanChunkGpuCache(GpuTaskState *gts)
{
	GpuCacheState  *gcache_state = gts->gc_state;

	if (pg_atomic_fetch_add_u32(gcache_state->gc_fetch_count, 1) != 0)
		return NULL;
	return __ExecScanChunkGpuCache(gts, gcache_state->gc_desc);
}

void
ExecReScanGpuCache(GpuCacheState *gcache_state)
{
	pg_atomic_write_u32(gcache_state->gc_fetch_count, 0);
}

void
ExecEndGpuCache(GpuCacheState *gcache_state)
{
	ReleaseGpuCacheDescIfUnused(gcache_state->gc_desc);
}

void
ExecInitDSMGpuCache(GpuCacheState *gcache_state,
					GpuTaskSharedState *gtss)
{
	pg_atomic_init_u32(&gtss->gc_fetch_count, 0);
	gcache_state->gc_fetch_count = &gtss->gc_fetch_count;
}

void
ExecReInitDSMGpuCache(GpuCacheState *gcache_state)
{
	pg_atomic_write_u32(gcache_state->gc_fetch_count, 0);
}

void
ExecInitWorkerGpuCache(GpuCacheState *gcache_state,
					    GpuTaskSharedState *gtss)
{
	gcache_state->gc_fetch_count = &gtss->gc_fetch_count;
}

void
ExecShutdownGpuCache(GpuCacheState *gcache_state)
{
	/* do nothing */
}

void
ExplainGpuCache(GpuCacheState *gcache_state,
				Relation frel, ExplainState *es)
{
	GpuCacheDesc   *gc_desc = gcache_state->gc_desc;
	GpuCacheSharedState *gc_sstate = gc_desc;
	int				cuda_dindex = gc_sstate->cuda_dindex;
	size_t			gpu_main_size = 0UL;
	size_t			gpu_extra_size = 0UL;

	if (cuda_dindex >= 0 && cuda_dindex < numDevAttrs)
		ExplainPropertyText("GPU Cache", devAttrs[cuda_dindex].DEV_NAME, es);

	pthreadRWLockReadLock(&gc_sstate->gpu_buffer_lock);
	if (gc_sstate->gpu_main_devptr != 0UL)
		gpu_main_size = gc_sstate->gpu_main_size;
	if (gc_sstate->gpu_extra_devptr != 0UL)
		gpu_extra_size = gc_sstate->gpu_extra_size;
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);

	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		char	temp[300];

		snprintf(temp, sizeof(temp),
				 "main: %s, extra: %s",
				 format_numeric(gpu_main_size),
				 format_numeric(gpu_extra_size));
		ExplainPropertyText("GPU Cache Size", temp, es);
	}
	else
	{
		ExplainPropertyInteger("GPU Cache Main Size", NULL, gpu_main_size, es);
		ExplainPropertyInteger("GPU Cache Extra Size", NULL, gpu_extra_size, es);
	}
}

CUresult
gpuCacheMapDeviceMemory(GpuContext *gcontext,
						pgstrom_data_store *pds)
{
	return CUDA_ERROR_OUT_OF_MEMORY;
}

void
gpuCacheUnmapDeviceMemory(GpuContext *gcontext,
						  pgstrom_data_store *pds)
{

}











static void
__gpuCacheCallbackOnAlterTable(Oid table_oid)
{
	Datum		signature_old;
	Datum		signature_new;
	HASH_SEQ_STATUS	hseq;
	GpuCacheDesc *gc_desc;
	GpuCacheHandle *gc_handle;
	bool		found;

	if (hash_get_num_entries(gcache_descriptors_htab) == 0)
		return;

	signature_old = gpuCacheTableSignatureSnapshot(table_oid, NULL);
	signature_new = gpuCacheTableSignatureSnapshot(table_oid, SnapshotSelf);
	if (signature_old == signature_new)
		return;

	hash_seq_init(&hseq, gcache_descriptors_htab);
	while ((gc_desc = hash_seq_search(&hseq)) != NULL)
	{
		Assert(gc_desc->database_oid == MyDatabaseId);
		if (gc_desc->table_oid == table_oid &&
			gc_desc->signature == signature_old)
		{
			gc_handle = hash_search(gcache_handle_htab,
									gc_desc, HASH_ENTER, &found);
			if (!found)
			{
				__initGpuCacheHandle(gc_handle);
				gc_handle->gc_desc = gc_desc;
				dlist_push_tail(&gc_desc->gc_handles, &gc_handle->chain);
			}
			gc_handle->drop_on_commit = true;
		}
	}
}

static void
__gpuCacheCallbackOnAlterTrigger(Oid trigger_oid)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;

	srel = table_open(TriggerRelationId, AccessShareLock);
	ScanKeyInit(&skey,
				Anum_pg_trigger_oid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(trigger_oid));
	sscan = systable_beginscan(srel, TriggerOidIndexId, true,
							   SnapshotSelf, 1, &skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Oid		table_oid = ((Form_pg_trigger)GETSTRUCT(tuple))->tgrelid;

		__gpuCacheCallbackOnAlterTable(table_oid);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
}

static void
__gpuCacheOnDropRelation(Oid table_oid)
{
	Datum		signature;
	HASH_SEQ_STATUS hseq;
	GpuCacheDesc *gc_desc;
	GpuCacheHandle *gc_handle;
	bool		found;

	if (hash_get_num_entries(gcache_descriptors_htab) == 0)
		return;
	signature = gpuCacheTableSignatureSnapshot(table_oid, NULL);
	if (signature == 0)
		return;

	hash_seq_init(&hseq, gcache_descriptors_htab);
	while ((gc_desc = hash_seq_search(&hseq)) != NULL)
	{
		Assert(gc_desc->database_oid == MyDatabaseId);
		if (gc_desc->table_oid == table_oid &&
			gc_desc->signature == signature)
		{
			gc_handle = hash_search(gcache_handle_htab,
									gc_desc, HASH_ENTER, &found);
			if (!found)
			{
				__initGpuCacheHandle(gc_handle);
				gc_handle->gc_desc = gc_desc;
				dlist_push_tail(&gc_desc->gc_handles, &gc_handle->chain);
			}
			gc_handle->drop_on_commit = true;
		}
	}
}

static void
__gpuCacheOnDropTrigger(Oid trigger_oid)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;

	srel = table_open(TriggerRelationId, AccessShareLock);
	ScanKeyInit(&skey, Anum_pg_trigger_oid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(trigger_oid));
	sscan = systable_beginscan(srel, TriggerOidIndexId,
							   true, NULL, 1, &skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Oid		table_oid = ((Form_pg_trigger) GETSTRUCT(tuple))->tgrelid;

		__gpuCacheCallbackOnAlterTable(table_oid);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
}

static void
gpuCacheObjectAccess(ObjectAccessType access,
					 Oid classId,
					 Oid objectId,
					 int subId,
					 void *arg)
{
	if (object_access_next)
		object_access_next(access, classId, objectId, subId, arg);

	if (access == OAT_POST_CREATE)
	{
		if (classId == RelationRelationId && subId > 0)
		{
			/* ALTER TABLE ... ADD COLUMN */
			//elog(LOG, "pid=%u OAT_POST_CREATE (pg_class, objectId=%u, subId=%d)", getpid(), objectId, subId);
			__gpuCacheCallbackOnAlterTable(objectId);
		}
		else if (classId == TriggerRelationId)
		{
			/* CREATE OR REPLACE TRIGGER */
			//elog(LOG, "pid=%u OAT_POST_CREATE (pg_trigger, objectId=%u)", getpid(), objectId);
			__gpuCacheCallbackOnAlterTrigger(objectId);
		}
	}
	else if (access == OAT_POST_ALTER)
	{
		if (classId == RelationRelationId)
		{
			//elog(LOG, "pid=%u OAT_POST_ALTER (pg_class, objectId=%u, subId=%d)", getpid(), objectId, subId);
			__gpuCacheCallbackOnAlterTable(objectId);
		}
		else if (classId == TriggerRelationId)
		{
			//elog(LOG, "pid=%u OAT_POST_ALTER (pg_trigger, objectId=%u)", getpid(), objectId);
			__gpuCacheCallbackOnAlterTrigger(objectId);
		}
	}
	else if (access == OAT_DROP)
	{
		if (classId == RelationRelationId)
		{
			//elog(LOG, "pid=%u OAT_DROP (pg_class, objectId=%u, subId=%d)", getpid(), objectId, subId);
			__gpuCacheOnDropRelation(objectId);
		}
		else if (classId == TriggerRelationId)
		{
			//elog(LOG, "pid=%u OAT_DROP (pg_trigger, objectId=%u)", getpid(), objectId);
			__gpuCacheOnDropTrigger(objectId);
		}
	}
}

static void
gpuCacheRelcacheCallback(Datum arg, Oid relid)
{
	HASH_SEQ_STATUS	hseq;
	GpuCacheDesc   *gc_desc;

	//elog(LOG, "pid=%u: gpuCacheRelcacheCallback (relid=%u)", getpid(), relid);
	if (hash_get_num_entries(gcache_descriptors_htab) == 0)
		return;

	hash_seq_init(&hseq, gcache_descriptors_htab);
	while ((gc_desc = hash_seq_search(&hseq)) != NULL)
	{
		Assert(gc_desc->database_oid == MyDatabaseId);
		if (gc_desc->table_oid == relid)
			ReleaseGpuCacheDescIfUnused(gc_desc);
	}
}

static void
gpuCacheSyscacheCallback(Datum arg, int cacheid, uint32 hashvalue)
{
	//elog(LOG, "pid=%u: gpuCacheSyscacheCallback (cacheid=%u)", getpid(), cacheid);
	__gpucache_sync_trigger_function_oid = InvalidOid;
}

/*
 * gpuCacheAddCommitLog
 */
static void
gpuCacheAddCommitLog(GpuCacheHandle *gc_handle)
{
	GCacheTxLogCommit *c_log = alloca(GCACHE_TX_LOG_COMMIT_ALLOCSZ);
	PendingRowIdItem *pitem;
	char	   *pos = gc_handle->buf.data;
	uint32		count = 1;

	/* commit log buffer */
	c_log->type = GCACHE_TX_LOG__COMMIT;
	c_log->length = offsetof(GCacheTxLogCommit, data);
	c_log->xid = gc_handle->xid;
	c_log->timestamp = GetCurrentTimestamp();
	c_log->nitems = 0;

	while (count <= gc_handle->nitems)
	{
		bool	flush_commit_log = (count == gc_handle->nitems);
		char   *temp;

		pitem = (PendingRowIdItem *)pos;
		switch (pitem->tag)
		{
			case 'I':	/* INSERT with rowid(u32) */
			case 'D':	/* DELETE with rowid(u32) */
				if (c_log->length + 5 > GCACHE_TX_LOG_COMMIT_ALLOCSZ)
				{
					flush_commit_log = true;
					break;
				}
				temp = (char *)c_log + c_log->length;
				*temp++ = pitem->tag;
				*((uint32 *)temp) = pitem->rowid;
				c_log->nitems++;
				c_log->length += (sizeof(char) + sizeof(uint32));
				count++;
				pos += sizeof(PendingRowIdItem);
				break;

			default:
				elog(FATAL, "broken internal PendingRowIdItem");
		}

		if (flush_commit_log)
		{
			int		diff = MAXALIGN(c_log->length) - c_log->length;

			if (diff > 0)
			{
				memset((char *)c_log + c_log->length, 0, diff);
				c_log->length += diff;
			}
			__gpuCacheAppendLog(gc_handle->gc_desc,
								(GCacheTxLogCommon *)c_log);
			/* rewind */
			c_log->length = offsetof(GCacheTxLogCommit, data);
			c_log->nitems = 0;
		}
	}
	elog(LOG, "AddCommitLog: %s:%lx xid=%u nitems=%u",
		 gc_handle->gc_desc->gc_sstate->table_name,
		 gc_handle->gc_desc->gc_sstate->signature,
		 gc_handle->xid,
		 gc_handle->nitems);
}

/*
 * gpuCacheXactCallback
 */
static void
gpuCacheXactCallback(XactEvent event, void *arg)
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
	if (hash_get_num_entries(gcache_handle_htab) > 0)
	{
		TransactionId	curr_xid = GetCurrentTransactionIdIfAny();
		HASH_SEQ_STATUS	hseq;
		GpuCacheHandle *gc_handle;

		if (event == XACT_EVENT_PRE_COMMIT)
		{
			hash_seq_init(&hseq, gcache_handle_htab);
			while ((gc_handle = hash_seq_search(&hseq)) != NULL)
			{
				if (gc_handle->xid == curr_xid)
					gpuCacheAddCommitLog(gc_handle);
			}
		}
		else if (event == XACT_EVENT_COMMIT ||
				 event == XACT_EVENT_ABORT)
		{
			bool	normal_commit = (event == XACT_EVENT_COMMIT);

			hash_seq_init(&hseq, gcache_handle_htab);
			while ((gc_handle = hash_seq_search(&hseq)) != NULL)
			{
				if (gc_handle->xid == curr_xid)
					ReleaseGpuCacheHandle(gc_handle, normal_commit);
			}
		}
	}
}

/*
 * gpuCacheSubXactCallback 
 */
static void
gpuCacheSubXactCallback(SubXactEvent event,
						SubTransactionId mySubid,
						SubTransactionId parentSubid, void *arg)
{
#if 0
	elog(INFO, "SubXactCallback: ev=%s xid=%u top-xid=%u",
		 event == SUBXACT_EVENT_START_SUB ? "SUBXACT_EVENT_START_SUB" :
		 event == SUBXACT_EVENT_COMMIT_SUB ? "SUBXACT_EVENT_COMMIT_SUB" :
		 event == SUBXACT_EVENT_ABORT_SUB ? "SUBXACT_EVENT_ABORT_SUB" :
		 event == SUBXACT_EVENT_PRE_COMMIT_SUB ? "SUBXACT_EVENT_PRE_COMMIT_SUB" : "???",
		 GetCurrentTransactionIdIfAny(),
		 GetTopTransactionIdIfAny());
#endif
	if (hash_get_num_entries(gcache_handle_htab) == 0)
		return;

	if (event == SUBXACT_EVENT_ABORT_SUB)
	{
		TransactionId	curr_xid = GetCurrentTransactionIdIfAny();
		HASH_SEQ_STATUS	hseq;
		GpuCacheHandle *gc_handle;

		hash_seq_init(&hseq, gcache_handle_htab);
		while ((gc_handle = hash_seq_search(&hseq)) != NULL)
		{
			if (gc_handle->xid == curr_xid)
				ReleaseGpuCacheHandle(gc_handle, false);
		}
	}
}

/*
 * __gpuCacheInvokeBackgroundCommand
 */
static CUresult
__gpuCacheInvokeBackgroundCommand(Oid database_oid,
								  Oid table_oid,
								  Datum signature,
								  int cuda_dindex,
								  bool is_async,
								  int command,
								  uint64 end_pos)
{
	GpuCacheBackgroundCommand *cmd = NULL;
	dlist_node	   *dnode;
	Latch		   *latch;
	CUresult		retval = CUDA_SUCCESS;

	Assert(cuda_dindex >= 0 && cuda_dindex < numDevAttrs);
	SpinLockAcquire(&gcache_shared_head->bgworker_cmd_lock);
	for (;;)
	{
		if (gcache_shared_head->bgworkers[cuda_dindex].latch &&
			!dlist_is_empty(&gcache_shared_head->bgworker_free_cmds))
		{
			/*
			 * Ok, GPU memory keeper is alive, and GpuCacheBackgroundCommand
			 * is available now.
			 */
			break;
		}
		SpinLockRelease(&gcache_shared_head->bgworker_cmd_lock);
		CHECK_FOR_INTERRUPTS();
		pg_usleep(2000L);	/* 2ms */
		SpinLockAcquire(&gcache_shared_head->bgworker_cmd_lock);
	}
	latch = gcache_shared_head->bgworkers[cuda_dindex].latch;
	dnode = dlist_pop_head_node(&gcache_shared_head->bgworker_free_cmds);
	cmd = dlist_container(GpuCacheBackgroundCommand, chain, dnode);

	memset(cmd, 0, sizeof(GpuCacheBackgroundCommand));
    cmd->database_oid = database_oid;
    cmd->table_oid = table_oid;
	cmd->signature = signature;
    cmd->backend = (is_async ? NULL : MyLatch);
    cmd->command = command;
    cmd->retval  = (CUresult) UINT_MAX;
    cmd->end_pos = end_pos;
	dlist_push_tail(&gcache_shared_head->bgworkers[cuda_dindex].cmd_queue,
					&cmd->chain);
	SpinLockRelease(&gcache_shared_head->bgworker_cmd_lock);
	SetLatch(latch);

	if (!is_async)
	{
		SpinLockAcquire(&gcache_shared_head->bgworker_cmd_lock);
		while (cmd->retval == (CUresult) UINT_MAX)
		{
			SpinLockRelease(&gcache_shared_head->bgworker_cmd_lock);
			PG_TRY();
			{
				int		ev;

				ev = WaitLatch(MyLatch,
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
				SpinLockAcquire(&gcache_shared_head->bgworker_cmd_lock);
				if (cmd->retval == (CUresult) UINT_MAX)
				{
					/*
					 * If not completed yet, the command is switched to
					 * asynchronous mode - because nobody can return the
					 * GpuCacheBackgroundCommand to free-list no longer.
					 */
					cmd->backend = NULL;
				}
				else
				{
					/* completed, so back to the free-list by itself */
					dlist_push_tail(&gcache_shared_head->bgworker_free_cmds,
									&cmd->chain);
				}
                SpinLockRelease(&gcache_shared_head->bgworker_cmd_lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockAcquire(&gcache_shared_head->bgworker_cmd_lock);
		}
		retval = cmd->retval;
		dlist_push_tail(&gcache_shared_head->bgworker_free_cmds,
						&cmd->chain);
		SpinLockRelease(&gcache_shared_head->bgworker_cmd_lock);
	}
	return retval;
}

/*
 * GCACHE_BGWORKER_CMD__APPLY_REDO
 */
static CUresult
gpuCacheInvokeApplyRedo(GpuCacheSharedState *gc_sstate,
						uint64 sync_pos,
						bool is_async)
{
	return __gpuCacheInvokeBackgroundCommand(gc_sstate->database_oid,
											 gc_sstate->table_oid,
											 gc_sstate->signature,
											 gc_sstate->cuda_dindex,
											 is_async,
											 GCACHE_BGWORKER_CMD__APPLY_REDO,
											 sync_pos);
}

/*
 * GCACHE_BGWORKER_CMD__COMPACTION
 */
static CUresult
gpuCacheInvokeCompaction(GpuCacheSharedState *gc_sstate, bool is_async)
{
	return __gpuCacheInvokeBackgroundCommand(gc_sstate->database_oid,
											 gc_sstate->table_oid,
											 gc_sstate->signature,
											 gc_sstate->cuda_dindex,
											 is_async,
											 GCACHE_BGWORKER_CMD__COMPACTION,
											 0);
}

/*
 * GCACHE_BGWORKER_CMD__DROP_UNLOAD
 */
static CUresult
gpuCacheInvokeDropUnload(GpuCacheSharedState *gc_sstate, bool is_async)
{
	return __gpuCacheInvokeBackgroundCommand(gc_sstate->database_oid,
											 gc_sstate->table_oid,
											 gc_sstate->signature,
											 gc_sstate->cuda_dindex,
											 is_async,
											 GCACHE_BGWORKER_CMD__DROP_UNLOAD,
											 0);
}

/*
 * gcache_xact_redo_hook
 */
static void
gcache_xact_redo_hook(XLogReaderState *record)
{
	gpucache_xact_redo_next(record);
	if (InRecovery)
	{
		//add transaction logs

	}
}

/*
 * gcache_heap_redo_hook
 */
static void
gcache_heap_redo_hook(XLogReaderState *record)
{
	gpucache_heap_redo_next(record);
	if (InRecovery)
	{
		//add redo logs
		
	}
}

/*
 * __gpuCacheLoadCudaModule
 */
static CUresult
__gpuCacheLoadCudaModule(void)
{
	const char	   *path = PGSHAREDIR "/pg_strom/cuda_gcache.fatbin";
	int				rawfd = -1;
	struct stat		stat_buf;
	ssize_t			nbytes;
	char		   *image;
	CUresult		rc = CUDA_ERROR_SYSTEM_NOT_READY;
	CUmodule		cuda_module = NULL;

	rawfd = open(path, O_RDONLY);
	if (rawfd < 0)
	{
		elog(LOG, "gpucache: failed on open('%s'): %m", path);
		goto bailout;
	}
	if (fstat(rawfd, &stat_buf) != 0)
	{
		elog(LOG, "gpucache: failed on fstat('%s'): %m", path);
		goto bailout;
	}
	image = alloca(stat_buf.st_size + 1);
	nbytes = __readFile(rawfd, image, stat_buf.st_size);
	if (nbytes != stat_buf.st_size)
	{
		elog(LOG, "gpucache: failed on __readFile('%s'): %m", path);
		goto bailout;
	}
	image[nbytes] = '\0';

	rc = cuModuleLoadFatBinary(&cuda_module, image);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuModuleLoadFatBinary: %s", errorText(rc));
		goto bailout;
	}

	rc = cuModuleGetFunction(&gcache_kfunc_setup_owner,
							 cuda_module,
							 "kern_gpucache_setup_owner");
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuModuleGetFunction: %s", errorText(rc));
		goto bailout;
	}

	rc = cuModuleGetFunction(&gcache_kfunc_apply_redo,
							 cuda_module,
							 "kern_gpucache_apply_redo");
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuModuleGetFunction: %s", errorText(rc));
		goto bailout;
	}

	rc = cuModuleGetFunction(&gcache_kfunc_compaction,
							 cuda_module,
							 "kern_gpucache_compaction");
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuModuleGetFunction: %s", errorText(rc));
		goto bailout;
	}

	/* ok, all green */
	gcache_cuda_module = cuda_module;

	return CUDA_SUCCESS;

bailout:
	if (cuda_module)
		cuModuleUnload(cuda_module);
	if (rawfd >= 0)
		close(rawfd);
	return rc;
}

static inline CUresult
gpuCacheLoadCudaModule(void)
{
	if (gcache_cuda_module)
		return CUDA_SUCCESS;
	return __gpuCacheLoadCudaModule();
}

/*
 * gpuCacheAllocDeviceMemory
 */
static CUresult
gpuCacheAllocDeviceMemory(GpuCacheDesc *gc_desc)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	CUdeviceptr		m_main = 0;
	CUdeviceptr		m_extra = 0;
	CUresult		rc;
	size_t			sz;

	if (gc_sstate->gpu_main_devptr != 0UL)
		return CUDA_SUCCESS;

	/* main portion of the device buffer */
	rc = cuMemAlloc(&m_main, gc_sstate->kds_head.length);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuMemAlloc: %s", errorText(rc));
		goto error_0;
	}

	sz = KERN_DATA_STORE_HEAD_LENGTH(&gc_sstate->kds_head);
	rc = cuMemcpyHtoD(m_main, &gc_sstate->kds_head, sz);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuMemcpyHtoD: %s", errorText(rc));
		goto error_1;
	}

	rc = cuIpcGetMemHandle(&gc_sstate->gpu_main_mhandle, m_main);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuIpcGetMemHandle: %s", errorText(rc));
		goto error_1;
	}

	/* extra buffer, if any */
	if (gc_sstate->kds_extra.length > 0)
	{
		rc = cuMemAlloc(&m_extra, gc_sstate->kds_extra.length);
		if (rc != CUDA_SUCCESS)
		{
			elog(WARNING, "failed on cuMemAlloc: %s", errorText(rc));
			goto error_1;
		}

		rc = cuMemcpyHtoD(m_extra, &gc_sstate->kds_extra,
						  offsetof(kern_data_extra, data));
		if (rc != CUDA_SUCCESS)
		{
			elog(WARNING, "failed on cuMemcpyHtoD: %s", errorText(rc));
			goto error_2;
		}

		rc = cuIpcGetMemHandle(&gc_sstate->gpu_extra_mhandle, m_extra);
		if (rc != CUDA_SUCCESS)
		{
			elog(WARNING, "failed on cuIpcGetMemHandle: %s", errorText(rc));
			goto error_2;
		}
	}
	elog(LOG, "gpucache: AllocMemory %s:%lx (main_sz=%zu, extra_sz=%zu)",
		 gc_sstate->table_name,
		 gc_sstate->signature,
		 gc_sstate->kds_head.length,
		 gc_sstate->kds_extra.length);

	gc_sstate->gpu_main_size = gc_sstate->kds_head.length;
	gc_sstate->gpu_extra_size = gc_sstate->kds_extra.length;
	gc_sstate->gpu_main_devptr = m_main;
	gc_sstate->gpu_extra_devptr = m_extra;
	return CUDA_SUCCESS;

error_2:
	cuMemFree(m_extra);
error_1:
	cuMemFree(m_main);
error_0:
	return rc;
}

/*
 * GCACHE_BGWORKER_CMD__COMPACTION command
 *
 * caller must hold exclusive lock on gc_sstate->gpu_buffer_lock
 */
static CUresult
gpuCacheBgWorkerExecCompactionNoLock(GpuCacheDesc *gc_desc)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	kern_data_extra	h_extra;
	int				grid_sz, block_sz;
	size_t			curr_usage;
	CUdeviceptr		m_try_extra = 0UL;
	CUdeviceptr		m_new_extra = 0UL;
	CUdeviceptr		m_temp;
	CUipcMemHandle	new_mhandle;
	CUresult		rc;
	void		   *kern_args[3];

	if (gc_sstate->gpu_extra_devptr == 0UL)
		return CUDA_SUCCESS;	/* nothing to do, if no extra buffer */

	rc = __gpuOptimalBlockSize(&grid_sz,
							   &block_sz,
							   gcache_kfunc_compaction,
							   gc_sstate->cuda_dindex,
							   0, 0);
	if (rc != CUDA_SUCCESS)
		return rc;

	/*
	 * phase-1: Estimation of the required device memory. This dummy
	 * extra buffer is initialized to usage > length, so compaction
	 * kernel never copy the varlena values.
	 */
	rc = cuMemAllocManaged(&m_try_extra, sizeof(kern_data_extra),
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "gpucache: failed on cuMemAllocManaged: %s", errorText(rc));
		return rc;
	}
	memset(&h_extra, 0, offsetof(kern_data_extra, data));
	h_extra.usage  = offsetof(kern_data_extra, data);
	memcpy((void *)m_try_extra, &h_extra, offsetof(kern_data_extra, data));

	kern_args[0] = &gc_sstate->gpu_main_devptr;
	kern_args[1] = &gc_sstate->gpu_extra_devptr;
	kern_args[2] = &m_try_extra;
	rc = cuLaunchKernel(gcache_kfunc_compaction,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "gpucache: failed on cuLaunchKernel: %s", errorText(rc));
		goto bailout;
	}
	/* check status of the kernel execution status */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "gpucache: failed on cuStreamSynchronize: %s", errorText(rc));
		goto bailout;
	}
	curr_usage = ((kern_data_extra *)m_try_extra)->usage;

	/*
	 * phase-2: Main portion of the compaction. 
	 * allocation of the new buffer, then, runs the compaction kernel.
	 */
	h_extra.length = Max(curr_usage + (64UL << 20),		/* 64MB margin */
						 (double)curr_usage * 1.15);	/* 15% margin */
	h_extra.usage = offsetof(kern_data_extra, data);
	rc = cuMemAlloc(&m_new_extra, h_extra.length);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "gpucache: failed on cuMemAlloc(%zu): %s",
			 h_extra.length, errorText(rc));
		goto bailout;
	}
	rc = cuIpcGetMemHandle(&new_mhandle, m_new_extra);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "gpucache: failed on cuIpcGetMemHandle: %s",
			 errorText(rc));
		goto bailout;
	}
	rc = cuMemcpyHtoD(m_new_extra, &h_extra, offsetof(kern_data_extra, data));
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuMemcpyHtoD: %s", errorText(rc));
		goto bailout;
	}

	/* kick the compaction kernel */
	kern_args[0] = &gc_sstate->gpu_main_devptr;
	kern_args[1] = &gc_sstate->gpu_extra_devptr;
	kern_args[2] = &m_new_extra;
	rc = cuLaunchKernel(gcache_kfunc_compaction,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "gpucache: failed on cuLaunchKernel: %s", errorText(rc));
		goto bailout;
	}
	/* check status of the kernel execution status */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		goto bailout;
	rc = cuMemcpyDtoH(&h_extra, m_new_extra, offsetof(kern_data_extra, data));
	if (rc != CUDA_SUCCESS)
		goto bailout;

	elog(LOG, "gpucache: extra compaction (%s:%lx) {length=%zu->%zu, usage=%zu}",
		 gc_sstate->table_name,
		 gc_sstate->signature,
		 gc_sstate->gpu_extra_size, h_extra.length, h_extra.usage);

	/* swap buffer */
	m_temp = gc_sstate->gpu_extra_devptr;
	gc_sstate->gpu_extra_devptr = m_new_extra;
	memcpy(&gc_sstate->gpu_extra_mhandle, &new_mhandle, sizeof(CUipcMemHandle));
	gc_sstate->gpu_extra_size = h_extra.length;
	m_new_extra = m_temp;

	/* ok, release old/temporary buffers */
bailout:
	if (m_new_extra != 0UL)
		cuMemFree(m_new_extra);
	if (m_try_extra != 0UL)
		cuMemFree(m_try_extra);
	return rc;
}

static inline CUresult
gpuCacheBgWorkerExecCompaction(GpuCacheDesc *gc_desc)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	CUresult	rc;

	pthreadRWLockWriteLock(&gc_sstate->gpu_buffer_lock);
	rc = gpuCacheBgWorkerExecCompactionNoLock(gc_desc);
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);

	return rc;
}

/*
 * GCACHE_BGWORKER_CMD__APPLY_REDO command
 */
static CUresult
__gpuCacheSetupRedoLogBuffer(GpuCacheDesc *gc_desc, uint64 end_pos,
							 CUdeviceptr *p_m_redo)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	kern_gpucache_redolog *h_redo = NULL;
	char		   *base = dsm_segment_address(gc_desc->dsm_seg);
	size_t			length;
	size_t			offset;
	uint64			index, nitems;
	uint64			head_pos, tail_pos, curr_pos;
	CUdeviceptr		m_redo = 0UL;
	CUresult		rc;

	SpinLockAcquire(&gc_sstate->redo_lock);
	if (end_pos <= gc_sstate->redo_read_pos)
	{
		SpinLockRelease(&gc_sstate->redo_lock);
		*p_m_redo = 0UL;		/* nothing to do */
		return CUDA_SUCCESS;
	}
	nitems = (gc_sstate->redo_write_nitems -
			  gc_sstate->redo_read_nitems);
	head_pos = gc_sstate->redo_read_pos;
	tail_pos = gc_sstate->redo_write_pos;
	Assert(head_pos <= tail_pos);
	Assert(end_pos <= gc_sstate->redo_write_pos);
	SpinLockRelease(&gc_sstate->redo_lock);

	/*
	 * allocation of managed memory for kern_gpucache_redolog
	 * (index to log and redo-log itself)
	 */
	length = (MAXALIGN(offsetof(kern_gpucache_redolog,
								log_index[nitems])) +
			  MAXALIGN(tail_pos - head_pos));
	rc = cuMemAllocManaged(&m_redo, length, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuMemAllocManaged(%zu): %s",
			 length, errorText(rc));
		*p_m_redo = 0UL;
		return rc;
	}
	h_redo = (kern_gpucache_redolog *)m_redo;
	memset(h_redo, 0, offsetof(kern_gpucache_redolog, log_index));
	h_redo->nrooms = nitems;
	h_redo->length = length;

	offset = MAXALIGN(offsetof(kern_gpucache_redolog,
							   log_index[nitems]));
	index = 0;
	curr_pos = head_pos;
	while (curr_pos < tail_pos && index < nitems)
	{
		GCacheTxLogCommon *tx_log;
		uint64		__curr_pos = (curr_pos % gc_sstate->redo_buffer_size);

		tx_log = (GCacheTxLogCommon *)(base + __curr_pos);
		if (__curr_pos + offsetof(GCacheTxLogCommon,
								  data) > gc_sstate->redo_buffer_size ||
			(tx_log->type & 0xffffff00U) != GCACHE_TX_LOG__MAGIC)
		{
			curr_pos += (gc_sstate->redo_buffer_size - __curr_pos);
			continue;
		}
		Assert(__curr_pos + tx_log->length <= gc_sstate->redo_buffer_size);
		Assert(tx_log->length == MAXALIGN(tx_log->length));
		memcpy((char *)h_redo + offset, tx_log, tx_log->length);
		h_redo->log_index[index++] = __kds_packed(offset);
		offset += tx_log->length;
		curr_pos += tx_log->length;
	}
	/* update redo_read_xxxx */
	SpinLockAcquire(&gc_sstate->redo_lock);
	gc_sstate->redo_read_nitems += nitems;
	gc_sstate->redo_read_pos = tail_pos;
	SpinLockRelease(&gc_sstate->redo_lock);

	if (index == 0)
	{
		cuMemFree(m_redo);
		m_redo = 0UL;	/* nothing to do */
	}
	else
	{
		h_redo->nitems = index;
		h_redo->length = offset;
	}
	*p_m_redo = m_redo;
	return CUDA_SUCCESS;
}

static CUresult
__gpuCacheLaunchApplyRedoKernel(GpuCacheDesc *gc_desc, CUdeviceptr m_redo)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	kern_gpucache_redolog *h_redo = (kern_gpucache_redolog *)m_redo;
	int			cuda_dindex = gc_sstate->cuda_dindex;
	int			o_grid_sz, o_block_sz;
	int			r_grid_sz, r_block_sz;
	int			phase;
	void	   *kern_args[4];
	CUresult	rc;

	rc = __gpuOptimalBlockSize(&o_grid_sz,
							   &o_block_sz,
							   gcache_kfunc_setup_owner,
							   cuda_dindex, 0, 0);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on __gpuOptimalBlockSize: %s", errorText(rc));
		return rc;
	}
	o_grid_sz = Min(o_grid_sz, (h_redo->nitems + o_block_sz - 1) / o_block_sz);

	rc = __gpuOptimalBlockSize(&r_grid_sz,
							   &r_block_sz,
							   gcache_kfunc_apply_redo,
							   cuda_dindex, 0, 0);
	r_grid_sz = Min(r_grid_sz, (h_redo->nitems + r_block_sz - 1) / r_block_sz);
	
retry:
	phase = 0;
	kern_args[0] = &m_redo;
	kern_args[1] = &gc_sstate->gpu_main_devptr;
	kern_args[2] = &gc_sstate->gpu_extra_devptr;
	kern_args[3] = &phase;

	/*
	 * setup-owner (phase-0) - clear the owner-id field
	 */
	rc = cuLaunchKernel(gcache_kfunc_setup_owner,
						o_grid_sz, 1, 1,
						o_block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		return rc;
	}

	/*
     * setup-owner (phase-1) - assign largest owner-id for each rows modified
     */
	phase = 1;
	rc = cuLaunchKernel(gcache_kfunc_setup_owner,
						o_grid_sz, 1, 1,
						o_block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		return rc;
	}

	/*
	 * apply redo logs (phase-2) - apply INSERT/DELETE logs
	 */
	phase = 2;
	rc = cuLaunchKernel(gcache_kfunc_apply_redo,
						r_grid_sz, 1, 1,
						r_block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		return rc;
	}

	/*
	 * setup-owner (phase-3) - assign largest owner-id of commit-logs
	 */
	phase = 3;
	rc = cuLaunchKernel(gcache_kfunc_setup_owner,
						o_grid_sz, 1, 1,
						o_block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		return rc;
	}

	/*
	 * apply redo logs (phase-4) - apply COMMIT logs
	 */
	phase = 4;
	rc = cuLaunchKernel(gcache_kfunc_apply_redo,
						r_grid_sz, 1, 1,
						r_block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(WARNING, "failed on cuLaunchKernel: %s", errorText(rc));
		return rc;
	}

	/* check status of the above kernel execution */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		return rc;

	if (h_redo->kerror.errcode == ERRCODE_OUT_OF_MEMORY)
	{
		rc = gpuCacheBgWorkerExecCompactionNoLock(gc_desc);
		if (rc == CUDA_SUCCESS)
		{
			memset(&h_redo->kerror, 0, sizeof(kern_errorbuf));
			goto retry;
		}
	}
	return CUDA_SUCCESS;
}

static CUresult
gpuCacheBgWorkerApplyRedoLog(GpuCacheDesc *gc_desc, uint64 end_pos)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	CUdeviceptr	m_redo = 0UL;
	CUresult	rc, __rc;

	rc = gpuCacheLoadCudaModule();
	if (rc != CUDA_SUCCESS)
		return rc;

	pthreadRWLockWriteLock(&gc_sstate->gpu_buffer_lock);
	
	rc = gpuCacheAllocDeviceMemory(gc_desc);
	if (rc != CUDA_SUCCESS)
		goto out_unlock;

	rc = __gpuCacheSetupRedoLogBuffer(gc_desc, end_pos, &m_redo);
	if (rc != CUDA_SUCCESS)
		goto out_unlock;
	if (m_redo != 0UL)
	{
		rc = __gpuCacheLaunchApplyRedoKernel(gc_desc, m_redo);

		__rc = cuMemFree(m_redo);
		if (__rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemFree: %s", errorText(__rc));
	}
out_unlock:
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);
	return rc;
}

static CUresult
gpuCacheBgWorkerDropUnload(GpuCacheDesc *gc_desc)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	CUresult	rc;

	pthreadRWLockWriteLock(&gc_sstate->gpu_buffer_lock);
	elog(LOG, "gpucache: DropUnload at %s:%lx main_sz=%zu extra_sz=%zu",
		 gc_sstate->table_name,
		 gc_sstate->signature,
		 gc_sstate->gpu_main_size,
		 gc_sstate->gpu_extra_size);
	if (gc_sstate->gpu_main_devptr != 0UL)
	{
		rc = cuMemFree(gc_sstate->gpu_main_devptr);
		if (rc != CUDA_SUCCESS)
			elog(LOG, "gpucache: failed on cuMemFree: %s", errorText(rc));
		gc_sstate->gpu_main_devptr = 0;
		gc_sstate->gpu_main_size = 0;
		memset(&gc_sstate->gpu_main_mhandle, 0, sizeof(CUipcMemHandle));
	}
	if (gc_sstate->gpu_extra_devptr != 0UL)
	{
		rc = cuMemFree(gc_sstate->gpu_extra_devptr);
		if (rc != CUDA_SUCCESS)
			elog(LOG, "gpucache: failed on cuMemFree: %s", errorText(rc));
		gc_sstate->gpu_extra_devptr = 0;
		gc_sstate->gpu_extra_size = 0;
		memset(&gc_sstate->gpu_extra_mhandle, 0, sizeof(CUipcMemHandle));
	}
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);
	
	return CUDA_SUCCESS;
}

/*
 * gpuCacheBgWorkerBegin
 */
void
gpuCacheBgWorkerBegin(int cuda_dindex)
{
	Assert(cuda_dindex >= 0 && cuda_dindex < numDevAttrs);
	SpinLockAcquire(&gcache_shared_head->bgworker_cmd_lock);
	gcache_shared_head->bgworkers[cuda_dindex].latch = MyLatch;
	SpinLockRelease(&gcache_shared_head->bgworker_cmd_lock);
}

/*
 * gpuCacheBgWorkerDispatch
 */
bool
gpuCacheBgWorkerDispatch(int cuda_dindex)
{
	GpuCacheBackgroundCommand *cmd;
	slock_t		   *cmd_lock = &gcache_shared_head->bgworker_cmd_lock;
	dlist_head	   *free_cmds = &gcache_shared_head->bgworker_free_cmds;
	dlist_head	   *cmd_queue = &gcache_shared_head->bgworkers[cuda_dindex].cmd_queue;
	dlist_node	   *dnode;
	GpuCacheDesc   *gc_desc;
	GpuCacheSharedState *gc_sstate;
	CUresult		rc;
	bool			retval;

	SpinLockAcquire(cmd_lock);
	if (dlist_is_empty(cmd_queue))
	{
		SpinLockRelease(cmd_lock);
		return true;	/* GpuCache allows bgworker to sleep */
	}
	dnode = dlist_pop_head_node(cmd_queue);
	cmd = dlist_container(GpuCacheBackgroundCommand, chain, dnode);
	memset(&cmd->chain, 0, sizeof(dlist_node));
	SpinLockRelease(cmd_lock);

	gc_desc = GetGpuCacheDescBgWorker(cmd->database_oid,
									  cmd->table_oid,
									  cmd->signature);
	if (!gc_desc)
	{
		elog(LOG, "gpucache: (cmd=%c, key=%u:%u:%lx) was not found",
			 cmd->command,
			 cmd->database_oid,
			 cmd->table_oid,
			 cmd->signature);
		rc = CUDA_ERROR_NOT_FOUND;
		goto out;
	}
	gc_sstate = gc_desc->gc_sstate;
	
	if (gc_sstate->cuda_dindex != cuda_dindex)
	{
		elog(LOG, "gpucache: (cmd=%c, rel=%s:%lx) was not on GPU-%d",
			 cmd->command,
			 gc_sstate->table_name,
			 gc_sstate->signature,
			 gc_sstate->cuda_dindex);
		rc = CUDA_ERROR_INVALID_VALUE;
		ReleaseGpuCacheDescIfUnused(gc_desc);
		goto out;
	}

	switch (cmd->command)
	{
		case GCACHE_BGWORKER_CMD__APPLY_REDO:
			rc = gpuCacheBgWorkerApplyRedoLog(gc_desc, cmd->end_pos);
			break;
		case GCACHE_BGWORKER_CMD__COMPACTION:
			rc = gpuCacheBgWorkerExecCompaction(gc_desc);
			break;
		case GCACHE_BGWORKER_CMD__DROP_UNLOAD:
			rc = gpuCacheBgWorkerDropUnload(gc_desc);
			break;
		default:
			rc = CUDA_ERROR_INVALID_VALUE;
			elog(LOG, "Unexpected GpuCache background command: %d",
				 cmd->command);
			break;
	}
	elog(LOG, "gpucache: (cmd=%c, key=%s:%lx) rc=%d",
		 cmd->command,
		 gc_sstate->table_name,
		 gc_sstate->signature,
		 (int)rc);
	ReleaseGpuCacheDescIfUnused(gc_desc);
out:
	cmd->retval = rc;

	SpinLockAcquire(cmd_lock);
	if (cmd->backend)
	{
		/*
		 * A backend process who kicked GpuCache maintainer is waiting
		 * for the response. It shall check the retval, and return the
		 * GpuCacheBackgroundCommand to free list again.
		 */
		SetLatch(cmd->backend);
	}
	else
	{
		/*
		 * GpuCache maintainer was kicked asynchronously, so nobody is
		 * waiting for the response, thus, GpuCacheBackgroundCommand
		 * must be backed to the free list again.
		 */
		dlist_push_head(free_cmds, &cmd->chain);
	}
	retval = dlist_is_empty(cmd_queue);
	SpinLockRelease(cmd_lock);
	return retval;
}

/*
 * gpuCacheBgWorkerIdleTask
 */
bool
gpuCacheBgWorkerIdleTask(int cuda_dindex)
{
	slock_t    *cmd_lock = &gcache_shared_head->bgworker_cmd_lock;
	dlist_head *free_cmds = &gcache_shared_head->bgworker_free_cmds;
	dlist_head *cmd_queue = &gcache_shared_head->bgworkers[cuda_dindex].cmd_queue;
	int			hindex;
	bool		retval = true;

	for (hindex = 0; hindex < GPUCACHE_SHARED_DESC_NSLOTS; hindex++)
	{
		dlist_head *slot = &gcache_shared_head->gcache_sstate_slot[hindex];
		dlist_iter	iter;

		SpinLockAcquire(&gcache_shared_head->gcache_sstate_lock);
		dlist_foreach(iter, slot)
		{
			GpuCacheSharedState *gc_sstate;
			uint64		timestamp;

			gc_sstate = dlist_container(GpuCacheSharedState,
										chain, iter.cur);
			if (gc_sstate->cuda_dindex != cuda_dindex)
				continue;

			SpinLockAcquire(&gc_sstate->redo_lock);
			timestamp = GetCurrentTimestamp();
			if ((gc_sstate->redo_write_pos > gc_sstate->redo_sync_pos &&
				 timestamp > (gc_sstate->redo_write_timestamp +
							  gc_sstate->gpu_sync_interval)) ||
				(gc_sstate->redo_write_pos > (gc_sstate->redo_sync_pos +
											  gc_sstate->gpu_sync_threshold)))
			{
				SpinLockAcquire(cmd_lock);
				if (!dlist_is_empty(free_cmds))
				{
					GpuCacheBackgroundCommand *cmd
						= dlist_container(GpuCacheBackgroundCommand, chain,
										  dlist_pop_head_node(free_cmds));

					memset(cmd, 0, sizeof(GpuCacheBackgroundCommand));
					cmd->database_oid = gc_sstate->database_oid;
					cmd->table_oid    = gc_sstate->table_oid;
					cmd->signature    = gc_sstate->signature;
					cmd->backend      = NULL;
					cmd->command      = GCACHE_BGWORKER_CMD__APPLY_REDO;
					cmd->end_pos      = gc_sstate->redo_write_pos;
					cmd->retval       = (CUresult) UINT_MAX;

					dlist_push_tail(cmd_queue, &cmd->chain);

					gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
				}
				SpinLockRelease(cmd_lock);
				retval = false;
			}
			SpinLockRelease(&gc_sstate->redo_lock);
		}
		SpinLockRelease(&gcache_shared_head->gcache_sstate_lock);
	}
	return retval;
}

/*
 * gpuCacheBgWorkerEnd
 */
void
gpuCacheBgWorkerEnd(int cuda_dindex)
{
	Assert(cuda_dindex >= 0 && cuda_dindex < numDevAttrs);
	SpinLockAcquire(&gcache_shared_head->bgworker_cmd_lock);
	gcache_shared_head->bgworkers[cuda_dindex].latch = NULL;
	SpinLockRelease(&gcache_shared_head->bgworker_cmd_lock);
}

/*
 * pgstrom_startup_gpu_cache
 */
static void
pgstrom_startup_gpu_cache(void)
{
	size_t	sz;
	bool	found;
	int		i;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	sz = offsetof(GpuCacheSharedHead, bgworkers[numDevAttrs]);
	gcache_shared_head = ShmemInitStruct("GpuCache Shared Head", sz, &found);
	if (found)
		elog(ERROR, "Bug? GpuCacheSharedHead already exists");
	memset(gcache_shared_head, 0, sz);
	SpinLockInit(&gcache_shared_head->gcache_sstate_lock);
	for (i=0; i < GPUCACHE_SHARED_DESC_NSLOTS; i++)
	{
		dlist_init(&gcache_shared_head->gcache_sstate_slot[i]);
	}
	/* IPC to GPU memory keeper background worker */
	SpinLockInit(&gcache_shared_head->bgworker_cmd_lock);
	dlist_init(&gcache_shared_head->bgworker_free_cmds);
	for (i=0; i < lengthof(gcache_shared_head->__bgworker_cmds); i++)
	{
		GpuCacheBackgroundCommand *cmd;

		cmd = &gcache_shared_head->__bgworker_cmds[i];
		dlist_push_tail(&gcache_shared_head->bgworker_free_cmds,
						&cmd->chain);
	}
	for (i=0; i < numDevAttrs; i++)
	{
		dlist_init(&gcache_shared_head->bgworkers[i].cmd_queue);
	}
}

/*
 * pgstrom_init_gpu_store
 */
void
pgstrom_init_gpu_cache(void)
{
	static bool gpucache_auto_preload;
	static bool gpucache_with_replication;
	BackgroundWorker worker;
	HASHCTL		hctl;
	int			i;

	/* GUC: pg_strom.gpucache_auto_preload */
	DefineCustomBoolVariable("pg_strom.gpucache_auto_preload",
							 "Enables auto preload of GPU memory store",
							 NULL,
							 &gpucache_auto_preload,
							 false,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* GPU: pg_strom.gpucache_with_replication */
	DefineCustomBoolVariable("pg_strom.gpucache_with_replication",
							 "Enables to synchronize GPU Store on replication slave",
							 NULL,
							 &gpucache_with_replication,
							 true,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup local hash tables */
	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = offsetof(GpuCacheDesc, signature) + sizeof(Datum);
	hctl.entrysize = sizeof(GpuCacheDesc);
	hctl.hcxt = CacheMemoryContext;
	gcache_descriptors_htab = hash_create("GpuCache Descriptors", 48, &hctl,
										  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = offsetof(GpuCacheHandle, xid) + sizeof(TransactionId);
	hctl.entrysize = sizeof(GpuCacheHandle);
	hctl.hcxt = CacheMemoryContext;
	gcache_handle_htab = hash_create("GpuCache Handles", 48, &hctl,
									 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	memset(&hctl, 0, sizeof(HASHCTL));
    hctl.keysize = sizeof(Oid);
	hctl.entrysize = sizeof(GpuCacheTableSignatureCache);
	hctl.hcxt = CacheMemoryContext;
	gcache_signatures_htab = hash_create("GpuCache Table Signature", 256, &hctl,
										 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	/*
	 * Background worke to load GPU Store on startup
	 */
	if (gpucache_auto_preload)
	{
		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "GPUCache Startup Preloader");
		worker.bgw_flags = (BGWORKER_SHMEM_ACCESS |
							BGWORKER_BACKEND_DATABASE_CONNECTION);
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = 1;
		snprintf(worker.bgw_library_name, BGW_MAXLEN,
				 "$libdir/pg_strom");
		snprintf(worker.bgw_function_name, BGW_MAXLEN,
				 "GpuCacheStartupPreloader");
		worker.bgw_main_arg = 0;
		RegisterBackgroundWorker(&worker);
	}

	/*
	 * Add hook for WAL replaying
	 */	
	if (gpucache_with_replication)
	{
		uintptr_t	start = TYPEALIGN_DOWN(PAGE_SIZE, &RmgrTable[0]);
		uintptr_t	end = TYPEALIGN(PAGE_SIZE, &RmgrTable[RM_MAX_ID+1]);

		if (mprotect((void *)start, end - start,
					 PROT_READ | PROT_WRITE | PROT_EXEC) != 0)
		{
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not enable GPU store on replication slave: %m"),
					 errhint("try to turn off pg_strom.gpucache_with_replication")));
		}

		for (i=0; i <= RM_MAX_ID; i++)
		{
			if (strcmp(RmgrTable[i].rm_name, "Transaction") == 0)
			{
				gpucache_xact_redo_next = RmgrTable[i].rm_redo;
				*((void **)&RmgrTable[i].rm_redo) = gcache_xact_redo_hook;
			}
			else if (strcmp(RmgrTable[i].rm_name, "Heap") == 0)
			{
				gpucache_heap_redo_next = RmgrTable[i].rm_redo;
				*((void **)&RmgrTable[i].rm_redo) = gcache_heap_redo_hook;
			}
		}
		Assert(gpucache_xact_redo_next != NULL &&
			   gpucache_heap_redo_next != NULL);
	}
	/* request for the static shared memory */
	RequestAddinShmemSpace(STROMALIGN(offsetof(GpuCacheSharedHead,
											   bgworkers[numDevAttrs])));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_cache;

	/* callback when trigger is changed */
	object_access_next = object_access_hook;
	object_access_hook = gpuCacheObjectAccess;
	/* callbacks for invalidation messages */
	CacheRegisterRelcacheCallback(gpuCacheRelcacheCallback, 0);
	CacheRegisterSyscacheCallback(PROCOID, gpuCacheSyscacheCallback, 0);
	/* transaction callbacks */
	RegisterXactCallback(gpuCacheXactCallback, NULL);
	RegisterSubXactCallback(gpuCacheSubXactCallback, NULL);
}
