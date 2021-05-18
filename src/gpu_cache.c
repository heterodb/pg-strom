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
	/* pg_strom.gpucache_auto_preload related */
	int32		gcache_auto_preload_count;
	NameData	gcache_auto_preload_dbname;
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
 * GpuCacheSharedState (shared structure; dynamic portable)
 */
typedef struct
{
	dlist_node		chain;
	Oid				database_oid;
	Oid				table_oid;
	Datum			signature;
	char			table_name[NAMEDATALEN];	/* for debugging */
	int32			refcnt;
	int				initial_loading;
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
	char		   *redo_buffer;

	/* schema definitions (KDS_FORMAT_COLUMN) */
	size_t			kds_extra_sz;
	kern_data_store	kds_head;
} GpuCacheSharedState;

/*
 * GpuCacheDesc (GpuCache Descriptor per backend)
 */
typedef struct
{
	Oid				database_oid;
	Oid				table_oid;
	Datum			signature;
	TransactionId	xid;
	GpuCacheSharedState *gc_sstate;
	bool			drop_on_rollback;
	bool			drop_on_commit;
	uint32			nitems;
	StringInfoData	buf;		/* array of PendingCtidItem */
} GpuCacheDesc;

typedef struct
{
	char			tag;
	ItemPointerData	ctid;
} PendingCtidItem;

/* --- static variables --- */
static char		   *pgstrom_gpucache_auto_preload;		/* GUC */
static bool			enable_gpucache;					/* GUC */
static GpuCacheSharedHead *gcache_shared_head = NULL;
static HTAB		   *gcache_descriptors_htab = NULL;
static HTAB		   *gcache_signatures_htab = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static object_access_hook_type object_access_next = NULL;
static CUmodule		gcache_cuda_module = NULL;
static CUfunction	gcache_kfunc_init_empty = NULL;
static CUfunction	gcache_kfunc_apply_redo = NULL;
static CUfunction	gcache_kfunc_compaction = NULL;

/* --- function declarations --- */
static void		__gpuCacheAppendLog(GpuCacheDesc *gc_desc,
									GCacheTxLogCommon *tx_log);

static CUresult gpuCacheInvokeApplyRedo(GpuCacheSharedState *gc_sstate,
										uint64 end_pos,
										bool is_async);
static CUresult gpuCacheInvokeCompaction(GpuCacheSharedState *gc_sstate,
										 bool is_async);
static CUresult gpuCacheInvokeDropUnload(GpuCacheSharedState *gc_sstate,
										 bool is_async);
void	gpuCacheStartupPreloader(Datum arg);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_sync_trigger);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_apply_redo);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_compaction);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_info);

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

static bool
__parseSyncTriggerOptions(const char *__config, GpuCacheOptions *gc_options)
{
	int			cuda_dindex = 0;				/* default: GPU0 */
	int			gpu_sync_interval = 5000000L;	/* default: 5sec = 5000000us */
	ssize_t		gpu_sync_threshold = -1;		/* default: auto */
	int64		max_num_rows = (10UL << 20);	/* default: 10M rows */
	ssize_t		redo_buffer_size = (160UL << 20);	/* default: 160MB */
	char	   *config;
	char	   *key, *value;
	char	   *saved;

	if (!__config)
		goto out;
	config = alloca(strlen(__config) + 1);
	strcpy(config, __config);

	for (key = strtok_r(config, ",", &saved);
		 key != NULL;
		 key = strtok_r(NULL,   ",", &saved))
	{
		value = strchr(key, '=');
		if (!value)
		{
			elog(WARNING, "gpucache: options syntax error [%s]", key);
			return false;
		}
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
				{
					elog(WARNING, "gpucache: failed on gethostname: %m");
					return false;
				}
				if (strcmp(host, name) != 0)
					continue;
			}
			else if (*host != '\0')
			{
				elog(WARNING, "gpucache: invalid option [%s]=[%s]",
					 key, value);
				return false;
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
			{
				elog(WARNING, "gpucache: gpu_device_id (%d) not found",
					 gpu_device_id);
				return false;
			}
		}
		else if (strcmp(key, "max_num_rows") == 0)
		{
			char   *end;

			max_num_rows = strtol(value, &end, 10);
			if (*end != '\0')
			{
				elog(WARNING, "gpucache: invalid option [%s]=[%s]",
					 key, value);
				return false;
			}
			if (max_num_rows >= UINT_MAX)
			{
				elog(WARNING, "gpucache: max_num_rows too large (%lu)",
					 max_num_rows);
				return false;
			}
		}
		else if (strcmp(key, "gpu_sync_interval") == 0)
		{
			char   *end;

			gpu_sync_interval = strtol(value, &end, 10);
			if (*end != '\0')
			{
				elog(WARNING, "gpucache: invalid option [%s]=[%s]",
					 key, value);
				return false;
			}
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
			{
				elog(WARNING, "gpucache: invalid option [%s]=[%s]",
					 key, value);
				return false;
			}
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
			{
				elog(WARNING, "gpucache: invalid option [%s]=[%s]",
					 key, value);
				return false;
			}
			if (redo_buffer_size < (16UL << 20))
			{
				elog(WARNING, "gpucache: 'redo_buffer_size' too small (%zu)",
					 redo_buffer_size);
				return false;
			}
		}
		else
		{
			elog(WARNING, "gpucache: unknown option [%s]=[%s]", key, value);
		}
	}
out:
	if (gc_options)
	{
		if (gpu_sync_threshold < 0)
			gpu_sync_threshold = redo_buffer_size / 4;
		if (gpu_sync_threshold > redo_buffer_size / 2)
		{
			elog(WARNING, "gpucache: gpu_sync_threshold is too small");
			return false;
		}

		memset(gc_options, 0, sizeof(GpuCacheOptions));
		gc_options->cuda_dindex       = cuda_dindex;
		gc_options->gpu_sync_interval = gpu_sync_interval;
		gc_options->gpu_sync_threshold = gpu_sync_threshold;
		gc_options->max_num_rows      = max_num_rows;
		gc_options->redo_buffer_size  = redo_buffer_size;
	}
	return true;
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
	GpuCacheOptions gc_options;

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
	GpuCacheOptions gc_options;
} GpuCacheTableSignatureCache;

static void
__gpuCacheTableSignature(Relation rel, GpuCacheTableSignatureCache *entry)
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
		goto no_gpu_cache;
	if (rd_rel->relfilenode == 0)
		goto no_gpu_cache;
	sig->reltablespace	= rd_rel->reltablespace;
	sig->relfilenode	= rd_rel->relfilenode;
	sig->relnatts		= rd_rel->relnatts;

	/* sync trigger */
	if (!trigdesc)
		goto no_gpu_cache;
	for (j=0; j < trigdesc->numtriggers; j++)
	{
		Trigger *trig = &trigdesc->triggers[j];

		if (trig->tgenabled != TRIGGER_FIRES_ON_ORIGIN &&
			trig->tgenabled != TRIGGER_FIRES_ALWAYS)
			continue;

		if (trig->tgtype == (TRIGGER_TYPE_ROW |
							 TRIGGER_TYPE_AFTER |
							 TRIGGER_TYPE_INSERT |
							 TRIGGER_TYPE_DELETE |
							 TRIGGER_TYPE_UPDATE) &&
			trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (has_row_trigger)
				goto no_gpu_cache;		/* should not call trigger twice per row */
			if ((trig->tgnargs == 0 &&
				 __parseSyncTriggerOptions(NULL,
										   &sig->gc_options)) ||
				(trig->tgnargs == 1 &&
				 __parseSyncTriggerOptions(trig->tgargs[0],
										   &sig->gc_options)))
			{
				sig->tg_sync_row = trig->tgoid;
				has_row_trigger = true;
			}
			else
			{
				goto no_gpu_cache;
			}
		}
		else if (trig->tgtype == (TRIGGER_TYPE_TRUNCATE) &&
				 trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (has_stmt_trigger)
				goto no_gpu_cache;	/* misconfiguration */
			sig->tg_sync_stmt = trig->tgoid;
			has_stmt_trigger = true;
		}
	}
	if (!has_row_trigger || !has_stmt_trigger)
		goto no_gpu_cache;			/* no sync triggers */

	/* pg_attribute related */
	for (j=0; j < natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		sig->attrs[j].atttypid	= attr->atttypid;
		sig->attrs[j].atttypmod	= attr->atttypmod;
		sig->attrs[j].attnotnull = attr->attnotnull;
		sig->attrs[j].attisdropped = attr->attisdropped;
	}
	memcpy(&entry->gc_options, &sig->gc_options,
		   sizeof(GpuCacheOptions));
	entry->signature = hash_any((unsigned char *)sig, len) | 0x100000000UL;
	return;

no_gpu_cache:
	memset(&entry->gc_options, 0, sizeof(GpuCacheOptions));
	entry->signature = 0;
}

static inline Datum
gpuCacheTableSignature(Relation rel, GpuCacheOptions *gc_options)
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
			__gpuCacheTableSignature(rel, entry);
		}
		PG_CATCH();
		{
			hash_search(gcache_signatures_htab,
						&table_oid, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	if (gc_options)
		memcpy(gc_options, &entry->gc_options, sizeof(GpuCacheOptions));
	return entry->signature;
}

static Datum
__gpuCacheTableSignatureSnapshot(HeapTuple pg_class_tuple,
								 Snapshot snapshot,
								 GpuCacheOptions *gc_options)
{
	GpuCacheTableSignatureBuffer *sig;
	Form_pg_class pg_class = (Form_pg_class) GETSTRUCT(pg_class_tuple);
	Oid			table_oid = PgClassTupleGetOid(pg_class_tuple);
	Relation	srel;
	ScanKeyData	skey[2];
	SysScanDesc	sscan;
	HeapTuple	tuple;
	bool		has_row_trigger = false;
	bool		has_stmt_trigger = false;
	int			j, len;

	/* pg_class validation */
	if (pg_class->relkind != RELKIND_RELATION &&
		pg_class->relkind != RELKIND_PARTITIONED_TABLE)
		return 0UL;
	if (pg_class->relfilenode == 0)
		return 0UL;
	if (!pg_class->relhastriggers)
		return 0UL;
	len = offsetof(GpuCacheTableSignatureBuffer,
				   attrs[pg_class->relnatts]);
	sig = alloca(len);
	memset(sig, 0, len);

	sig->reltablespace  = pg_class->reltablespace;
	sig->relfilenode    = pg_class->relfilenode;
	sig->relnatts       = pg_class->relnatts;

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

		if (pg_trig->tgenabled != TRIGGER_FIRES_ON_ORIGIN &&
			pg_trig->tgenabled != TRIGGER_FIRES_ALWAYS)
			continue;

		if (pg_trig->tgtype == (TRIGGER_TYPE_ROW |
								TRIGGER_TYPE_INSERT |
								TRIGGER_TYPE_DELETE |
								TRIGGER_TYPE_UPDATE) &&
			pg_trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (has_row_trigger)
				goto no_gpu_cache;
			if (pg_trig->tgnargs == 0)
			{
				if (!__parseSyncTriggerOptions(NULL, &sig->gc_options))
					goto no_gpu_cache;
			}
			else if (pg_trig->tgnargs == 1)
			{
				Datum	datum;
				bool	isnull;

				datum = fastgetattr(tuple, Anum_pg_trigger_tgargs,
									RelationGetDescr(srel), &isnull);
				if (isnull)
					goto no_gpu_cache;
				if (!__parseSyncTriggerOptions(VARDATA_ANY(datum),
											   &sig->gc_options))
					goto no_gpu_cache;
			}
			else
			{
				goto no_gpu_cache;
			}
			sig->tg_sync_row = PgTriggerTupleGetOid(tuple);
			has_row_trigger = true;
		}
		else if (pg_trig->tgtype == TRIGGER_TYPE_TRUNCATE &&
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

	if (gc_options)
		memcpy(gc_options, &sig->gc_options, sizeof(GpuCacheOptions));
	return hash_any((unsigned char *)sig, len) | 0x100000000UL;

no_gpu_cache:
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
	return 0UL;
}

static Datum
gpuCacheTableSignatureSnapshot(Oid table_oid,
							   Snapshot snapshot,
							   GpuCacheOptions *gc_options)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;
	Datum		signature = 0UL;

	/* pg_class */
	srel = table_open(RelationRelationId, AccessShareLock);
	ScanKeyInit(&skey,
				Anum_pg_class_oid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(table_oid));
	sscan = systable_beginscan(srel, ClassOidIndexId,
							   true, snapshot, 1, &skey);
	tuple = systable_getnext(sscan);
	if (HeapTupleIsValid(tuple))
	{
		signature = __gpuCacheTableSignatureSnapshot(tuple,
													 snapshot,
													 gc_options);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	return signature;
}

/*
 * gpuCacheTableSignatureInvalidation
 */
static void
gpuCacheTableSignatureInvalidation(Oid table_oid)
{
	hash_search(gcache_signatures_htab,
				&table_oid, HASH_REMOVE, NULL);
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
				__gpuCacheTableSignature(rel, entry);
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
	return (enable_gpucache ? retval : false);
}

/*
 * RelationHasGpuCache
 */
bool
RelationHasGpuCache(Relation rel)
{
	if (enable_gpucache)
		return (gpuCacheTableSignature(rel,NULL) != 0UL);
	return false;
}

/*
 * __hashSlotGpuCacheSharedState
 */
static inline dlist_head *
__hashSlotGpuCacheSharedState(Oid database_oid,
							  Oid table_oid,
							  Datum signature)
{
	struct {
		Oid		database_oid;
		Oid		table_oid;
		Datum	signature;
	} hkey;
	uint32		hvalue;
	uint32		hindex;

	hkey.database_oid = database_oid;
	hkey.table_oid    = table_oid;
	hkey.signature    = signature;
	hvalue = hash_any((unsigned char *)&hkey, sizeof(hkey));
	hindex = hvalue % GPUCACHE_SHARED_DESC_NSLOTS;

	return &gcache_shared_head->gcache_sstate_slot[hindex];
}

/*
 * lookupGpuCacheSharedState
 *
 * Note that caller must hold gcache_sstate_lock
 */
static GpuCacheSharedState *
lookupGpuCacheSharedState(Oid database_oid,
						  Oid table_oid,
						  Datum signature)
{
	GpuCacheSharedState *gc_sstate;
	dlist_head	   *slot;
	dlist_iter		iter;

	slot = __hashSlotGpuCacheSharedState(database_oid,
										 table_oid,
										 signature);
	dlist_foreach(iter, slot)
	{
		gc_sstate = dlist_container(GpuCacheSharedState,
									chain, iter.cur);
		if (gc_sstate->database_oid == database_oid &&
			gc_sstate->table_oid    == table_oid &&
			gc_sstate->signature    == signature)
		{
			return gc_sstate;
		}
	}
	return NULL;
}

/*
 * putGpuCacheSharedState
 */
static void
putGpuCacheSharedState(GpuCacheSharedState *gc_sstate, bool drop_shared_state)
{
	slock_t	   *lock = &gcache_shared_head->gcache_sstate_lock;

	SpinLockAcquire(lock);
	if (drop_shared_state)
		gc_sstate->refcnt &= 0xfffffffeU;
	Assert(gc_sstate->refcnt >= 2);
	gc_sstate->refcnt -= 2;
	if (gc_sstate->refcnt == 0)
	{
		dlist_delete(&gc_sstate->chain);
		if (gc_sstate->gpu_main_devptr != 0UL ||
			gc_sstate->gpu_extra_devptr != 0UL)
		{
			elog(WARNING, "gpucache: Bug? device memory for %s:%lx still remain (main: %zu, extra: %zu)",
				 gc_sstate->table_name,
				 gc_sstate->signature,
				 gc_sstate->gpu_main_size,
				 gc_sstate->gpu_extra_size);
		}
		elog(LOG, "gpucache: table %s:%lx is dropped", gc_sstate->table_name, gc_sstate->signature);
		pfree(gc_sstate);
	}
	SpinLockRelease(lock);
}

/*
 * __gpuCacheInitLoadVisibilityCheck
 */
static bool
__gpuCacheInitLoadVisibilityCheck(GpuCacheDesc *gc_desc,
								  HeapTuple tuple,
								  TransactionId *gcache_xmin,
								  TransactionId *gcache_xmax)
{
	HeapTupleHeader		htup = tuple->t_data;
	TransactionId		xmin;
	TransactionId		xmax;

	if (!HeapTupleHeaderXminCommitted(htup))
	{
		if (HeapTupleHeaderXminInvalid(htup))
			return false;
		xmin = HeapTupleHeaderGetRawXmin(htup);
		if (TransactionIdIsCurrentTransactionId(xmin))
		{
			*gcache_xmin = xmin;
			if (htup->t_infomask & HEAP_XMAX_INVALID)
			{
				/* xmax invalid */
				*gcache_xmax = InvalidTransactionId;
				return true;
			}

			if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
			{
				/* not deleter */
				*gcache_xmax = InvalidTransactionId;
				return true;
			}

			if (htup->t_infomask & HEAP_XMAX_IS_MULTI)
			{
				xmax = HeapTupleGetUpdateXid(htup);
				/* not LOCKED_ONLY, so it has to have an xmax */
				Assert(TransactionIdIsValid(xmax));

				/* updating subtransaction must have aborted */
				if (TransactionIdIsCurrentTransactionId(xmax))
				{
					*gcache_xmax = xmax;
					return true;
				}
				elog(WARNING, "gpucache: initial load on '%s' met a tuple inserted (not committed yet), but deleted by other concurrent transaction. Why? ctid=(%u,%u)",
					 get_rel_name(gc_desc->table_oid),
					 BlockIdGetBlockNumber(&tuple->t_self.ip_blkid),
					 tuple->t_self.ip_posid);
				return false;
			}

			xmax = HeapTupleHeaderGetRawXmax(htup);
			if (TransactionIdIsCurrentTransactionId(xmax))
			{
				/* tuple is already deleted by the current transaction */
				*gcache_xmax = xmax;
			}
			else
			{
				/* elsewhere, deleting subtransaction should have aborted */
				*gcache_xmax = InvalidTransactionId;
			}
			return true;
		}
		else if (TransactionIdIsInProgress(xmin))
		{
			/*
			 * Because GpuCache is built on after row / statement triggers,
			 * we may meet a tuple on the shared buffer inserted by the other
			 * concurrent transactions, during the initial-loading process.
			 * In this case, the inserter should be waiting for the completion
			 * of the current initial-loading process, then it adds REDO log
			 * entry of the new tuple.
			 * So, initial-loading can ignore the tuples not responsible.
			 */
			return false;
		}
		else if (!TransactionIdDidCommit(xmin))
		{
			/* aborted or crashed */
			return false;
		}
	}
	/* by here, the inserting transaction has committed */
	*gcache_xmin = FrozenTransactionId;

	if (htup->t_infomask & HEAP_XMAX_INVALID)
	{
		/* xid invalid or aborted */
		*gcache_xmax = InvalidTransactionId;
		return true;
	}
	if (htup->t_infomask & HEAP_XMAX_COMMITTED)
	{
		if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
		{
			*gcache_xmax = InvalidTransactionId;
			return true;
		}
		return false;	/* updated by other, and committed  */
	}

	xmax = HeapTupleHeaderGetRawXmax(htup);
	if (TransactionIdIsCurrentTransactionId(xmax))
	{
		if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
			*gcache_xmax = InvalidTransactionId;
		else
			*gcache_xmax = xmax;
		return true;
	}

	if (TransactionIdIsInProgress(xmax))
	{
		/*
		 * Because GpuCache is built on after row / statement triggers,
		 * we may meet a tuple on the shared buffer deleted by the other
		 * concurrent transactions, during the initial-loading process. 
		 * In this case, the deleter should be waiting for the completion
		 * of the current initial-loading process, then it adds REDO log
		 * entry for deletion (regardless of COMMIT or ABORT).
		 * So, initial-loading must load the body of tuple to be deleted
		 * once. If deletion is actually committed, its XACT log will
		 * invalidate the entry.
		 */
		*gcache_xmax = InvalidTransactionId;
		return true;
	}

	if (!TransactionIdDidCommit(xmax))
	{
		/* it must have aborted or crashed */
		*gcache_xmax = InvalidTransactionId;
		return true;
	}

	/* xmax transaction committed */
	if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
	{
		*gcache_xmax = InvalidTransactionId;
		return true;
	}
	return false;
}

/*
 * __gpuCacheInitLoadTrackCtid
 */
static void
__gpuCacheInitLoadTrackCtid(GpuCacheDesc *gc_desc,
							TransactionId xid,
							char tag, ItemPointer ctid)
{
	PendingCtidItem	pitem;

	if (gc_desc->xid != xid)
	{
		GpuCacheDesc   *__gc_temp;
		GpuCacheDesc	hkey;
		bool			found;

		hkey.database_oid = gc_desc->database_oid;
		hkey.table_oid    = gc_desc->table_oid;
		hkey.signature    = gc_desc->signature;
		hkey.xid          = xid;
		__gc_temp = hash_search(gcache_descriptors_htab,
								&hkey, HASH_ENTER, &found);
		if (!found)
		{
			GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;

			SpinLockAcquire(&gcache_shared_head->gcache_sstate_lock);
			gc_sstate->refcnt += 2;
			SpinLockRelease(&gcache_shared_head->gcache_sstate_lock);

			__gc_temp->gc_sstate = gc_sstate;
			__gc_temp->drop_on_rollback = true;
			__gc_temp->drop_on_commit = false;
			__gc_temp->nitems = 0;
			memset(&__gc_temp->buf, 0, sizeof(StringInfoData));
		}
		gc_desc = __gc_temp;
	}
	if (!gc_desc->buf.data)
		initStringInfoContext(&gc_desc->buf, CacheMemoryContext);
	Assert(tag == 'I' || tag == 'D');
	pitem.tag = tag;
	pitem.ctid = *ctid;
	appendBinaryStringInfo(&gc_desc->buf, (char *)&pitem,
						   sizeof(PendingCtidItem));
	gc_desc->nitems++;
}

/*
 * __execGpuCacheInitLoad
 */
static void
__execGpuCacheInitLoad(GpuCacheDesc *gc_desc, Relation rel)
{
	TableScanDesc	scandesc;
	HeapTuple		tuple;
	size_t			item_sz = 2048;
	GCacheTxLogInsert *item = palloc(item_sz);

	scandesc = table_beginscan(rel, SnapshotAny, 0, NULL);
	while ((tuple = heap_getnext(scandesc, ForwardScanDirection)) != NULL)
	{
		TransactionId	gcache_xmin;
		TransactionId	gcache_xmax;
		size_t			sz;

		if (!__gpuCacheInitLoadVisibilityCheck(gc_desc, tuple,
											   &gcache_xmin,
											   &gcache_xmax))
			continue;

		sz = MAXALIGN(offsetof(GCacheTxLogInsert, htup) + tuple->t_len);
		if (sz > item_sz)
		{
			item_sz = 2 * sz;
			item = repalloc(item, item_sz);
		}
		item->type = GCACHE_TX_LOG__INSERT;
		item->length = sz;
		item->rowid = UINT_MAX;
		item->rowid_found = false;
		memcpy(&item->htup, tuple->t_data, tuple->t_len);
		HeapTupleHeaderSetXmin(&item->htup, gcache_xmin);
		HeapTupleHeaderSetXmax(&item->htup, gcache_xmax);
		HeapTupleHeaderSetCmin(&item->htup, InvalidCommandId);
		__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)item);

		if (TransactionIdIsNormal(gcache_xmin))
			__gpuCacheInitLoadTrackCtid(gc_desc, gcache_xmin, 'I', &tuple->t_self);
		if (TransactionIdIsNormal(gcache_xmax))
			__gpuCacheInitLoadTrackCtid(gc_desc, gcache_xmax, 'D', &tuple->t_self);
		CHECK_FOR_INTERRUPTS();
	}
	table_endscan(scandesc);

	pfree(item);
}

/*
 * __createGpuCacheSharedState
 */
static GpuCacheSharedState *
__createGpuCacheSharedState(Relation rel,
							Datum signature,
							GpuCacheOptions *gc_options)
{
	GpuCacheSharedState *gc_sstate;
	TupleDesc		tupdesc = RelationGetDescr(rel);
	kern_data_store *kds_head;
	kern_colmeta   *cmeta;
	uint32			nrooms;
	size_t			sz, off;
	size_t			extra_sz = 0;
	int				j, unitsz;

	/* allocation of GpuCacheSharedState */
	off = (KDS_calculateHeadSize(tupdesc) +
		   STROMALIGN(sizeof(kern_colmeta)));
	sz = offsetof(GpuCacheSharedState, kds_head) + off;
	gc_sstate = MemoryContextAlloc(TopSharedMemoryContext,
								   MAXALIGN(sz) + gc_options->redo_buffer_size);
	memset(gc_sstate, 0, sz);
	gc_sstate->database_oid = MyDatabaseId;
	gc_sstate->table_oid = RelationGetRelid(rel);
	gc_sstate->signature = signature;
	strncpy(gc_sstate->table_name, RelationGetRelationName(rel), NAMEDATALEN);
	gc_sstate->refcnt = 3;
	gc_sstate->initial_loading = -1;	/* not yet */

	Assert(gc_options->max_num_rows < UINT_MAX);
	gc_sstate->max_num_rows       = gc_options->max_num_rows;
	gc_sstate->cuda_dindex        = gc_options->cuda_dindex;
	gc_sstate->redo_buffer_size   = gc_options->redo_buffer_size;
	gc_sstate->gpu_sync_threshold = gc_options->gpu_sync_threshold;
	gc_sstate->gpu_sync_interval  = gc_options->gpu_sync_interval;

	pthreadRWLockInit(&gc_sstate->gpu_buffer_lock);
	SpinLockInit(&gc_sstate->redo_lock);
	gc_sstate->redo_buffer = (char *)gc_sstate + MAXALIGN(sz);

	/* init schema definition in KDS_FORMAT_COLUMN */
	kds_head = &gc_sstate->kds_head;
	nrooms = gc_options->max_num_rows;
	init_kernel_data_store(kds_head,
						   tupdesc,
						   0,	/* to be set later */
						   KDS_FORMAT_COLUMN,
						   nrooms);
	kds_head->nslots = Max(Min(1.5 * (double)nrooms, UINT_MAX), 40000);
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
	gc_sstate->kds_extra_sz = extra_sz;

	return gc_sstate;
}

/*
 * __setupGpuCacheDesc
 */
static void
__setupGpuCacheDesc(GpuCacheDesc *gc_desc,
					Relation rel,		/* can be NULL, if no initial-loading */
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
	gc_desc->drop_on_rollback = false;
	gc_desc->drop_on_commit = false;
	gc_desc->nitems = 0;
	memset(&gc_desc->buf, 0, sizeof(StringInfoData));

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
			if ((gc_sstate->refcnt & 1) == 0)
			{
				/*
				 * If refcnt is even number, GpuCacheSharedState is already
				 * dropped, thus, should not be available longer.
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
				gc_sstate->refcnt += 2;
				goto found_uninitialized;
			}
			/* ok, all green */
			gc_sstate->refcnt += 2;
			SpinLockRelease(lock);

			gc_desc->gc_sstate = gc_sstate;
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
		gc_sstate = __createGpuCacheSharedState(rel,
												gc_desc->signature,
												gc_options);
		elog(LOG, "create GpuCacheSharedState %s:%lx",
			 gc_sstate->table_name, gc_sstate->signature);
	}
	PG_CATCH();
	{
		SpinLockRelease(lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* move to the initial loading */
	gc_desc->gc_sstate = gc_sstate;
	dlist_push_tail(slot, &gc_sstate->chain);
found_uninitialized:
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
		/* revert the status to empty & unloaded */
		gpuCacheInvokeDropUnload(gc_sstate, true);
		SpinLockAcquire(lock);
		gc_sstate->initial_loading = -1;	/* not yet loaded */
		SpinLockRelease(lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* ok, all done */
	SpinLockAcquire(lock);
	gc_sstate->initial_loading = 0;		/* ready now */
	SpinLockRelease(lock);

	gc_desc->gc_sstate = gc_sstate;
}

/*
 * lookupGpuCacheDesc
 *
 * This function tries to lookup (or create on the demand) GpuCacheDesc that
 * is per transaction state of GpuCache. Then, kicks its initial-loading
 * process if not completed yet.
 * Even if GpuCacheDesc is acquired multiple times, it shall be released at
 * the end of transaction once, so we don't need to release everytime.
 */
static GpuCacheDesc *
lookupGpuCacheDesc(Relation rel)
{
	GpuCacheOptions	gc_options;
	GpuCacheDesc	hkey;
	GpuCacheDesc   *gc_desc;
	bool			found;

	hkey.database_oid = MyDatabaseId;
	hkey.table_oid = RelationGetRelid(rel);
	hkey.signature = gpuCacheTableSignature(rel, &gc_options);
	if (hkey.signature == 0UL)
		return NULL;
	hkey.xid = GetCurrentTransactionId();
	Assert(TransactionIdIsValid(hkey.xid));

	gc_desc = hash_search(gcache_descriptors_htab,
						  &hkey, HASH_ENTER, &found);
	if (!found)
	{
		PG_TRY();
		{
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
	return (gc_desc->gc_sstate ? gc_desc : NULL);
}

/*
 * lookupGpuCacheDescNoLoad
 *
 * lookup (or create a new entry on demand) of a GpuCacheDesc entry,
 * but does not kick initial loading even if not initialized yet.
 */
static GpuCacheDesc *
lookupGpuCacheDescNoLoad(Oid database_oid,
						 Oid table_oid,
						 Datum signature,
						 GpuCacheOptions *gc_options)
{
	GpuCacheDesc	hkey;
	GpuCacheDesc   *gc_desc;
	bool			found;

	hkey.database_oid = database_oid;
	hkey.table_oid = table_oid;
	hkey.signature = signature;
	if (hkey.signature == 0UL)
		return NULL;
	hkey.xid = GetCurrentTransactionIdIfAny();
	Assert(TransactionIdIsValid(hkey.xid));

	gc_desc = hash_search(gcache_descriptors_htab,
						  &hkey, HASH_ENTER, &found);
	if (!found)
	{
		PG_TRY();
		{
			__setupGpuCacheDesc(gc_desc, NULL, gc_options);
		}
		PG_CATCH();
		{
			hash_search(gcache_descriptors_htab,
						&hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return (gc_desc->gc_sstate ? gc_desc : NULL);
}

/*
 * releaseGpuCacheDesc
 */
static void
releaseGpuCacheDesc(GpuCacheDesc *gc_desc, bool is_normal_commit)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;

	if (gc_sstate)
	{
		bool		drop_shared_state = (is_normal_commit
										 ? gc_desc->drop_on_commit
										 : gc_desc->drop_on_rollback);
		if (drop_shared_state)
			gpuCacheInvokeDropUnload(gc_sstate, true);
		else
		{
			char   *pos = gc_desc->buf.data;
			uint32	count;
			char	tag;

			for (count=0; count < gc_desc->nitems; count++)
			{
				PendingCtidItem	   *pitem = (PendingCtidItem *)pos;
				GCacheTxLogXact		x_log;

				tag = (is_normal_commit ? pitem->tag : tolower(pitem->tag));
				x_log.type   = GCACHE_TX_LOG__XACT;
				x_log.length = sizeof(GCacheTxLogXact);
				x_log.rowid  = UINT_MAX;
				x_log.rowid_found = false;
				x_log.tag    = tag;
				memcpy(&x_log.ctid, &pitem->ctid,
					   sizeof(ItemPointerData));

				__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)&x_log);
				pos += sizeof(PendingCtidItem);
			}
			elog(DEBUG2, "AddXactLog: %s:%lx xid=%u nitems=%u",
				 gc_sstate->table_name,
				 gc_sstate->signature,
				 gc_desc->xid,
				 gc_desc->nitems);
		}
		putGpuCacheSharedState(gc_sstate, drop_shared_state);
	}
	if (gc_desc->buf.data)
		pfree(gc_desc->buf.data);
	hash_search(gcache_descriptors_htab, gc_desc, HASH_REMOVE, NULL);
}

/*
 * __gpuCacheAppendLog
 */
static void
__gpuCacheAppendLog(GpuCacheDesc *gc_desc, GCacheTxLogCommon *tx_log)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	char	   *redo_buffer = gc_sstate->redo_buffer;
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
		if (gc_sstate->redo_write_pos > (gc_sstate->redo_sync_pos +
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
__gpuCacheInsertLog(HeapTuple tuple, GpuCacheDesc *gc_desc)
{
	GCacheTxLogInsert *item;
	PendingCtidItem pitem;
	size_t		sz;

	/* track ctid not committed yet */
	pitem.tag = 'I';
	pitem.ctid = tuple->t_self;
	appendBinaryStringInfo(&gc_desc->buf,
						   (char *)&pitem,
						   sizeof(PendingCtidItem));
	gc_desc->nitems++;

	/* INSERT Log */
	sz = MAXALIGN(offsetof(GCacheTxLogInsert, htup) + tuple->t_len);
	item = alloca(sz);
	item->type = GCACHE_TX_LOG__INSERT;
	item->length = sz;
	item->rowid = UINT_MAX;		/* to be set by kernel */
	item->rowid_found = false;	/* to be set by kernel */
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
__gpuCacheDeleteLog(HeapTuple tuple, GpuCacheDesc *gc_desc)
{
	GCacheTxLogDelete item;
	PendingCtidItem	pitem;

	/* track ctid to be released */
	pitem.tag = 'D';
	pitem.ctid = tuple->t_self;
	appendBinaryStringInfo(&gc_desc->buf,
						   (char *)&pitem,
						   sizeof(PendingCtidItem));
	gc_desc->nitems++;

	/* DELETE Log */
	item.type = GCACHE_TX_LOG__DELETE;
	item.length = MAXALIGN(sizeof(GCacheTxLogDelete));
	item.xid = GetCurrentTransactionId();
	item.rowid = UINT_MAX;		/* to be set by kernel */
	item.rowid_found = false;	/* to be set by kernel */
	memcpy(&item.ctid, &tuple->t_self, sizeof(ItemPointerData));

	__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)&item);
}

/*
 * __gpuCacheTruncateLog
 */
static void
__gpuCacheTruncateLog(GpuCacheDesc *gc_desc)
{
	Assert(!gc_desc->drop_on_commit);

	gc_desc->drop_on_commit = true;
}

/*
 * pgstrom_gpucache_sync_trigger
 */
Datum
pgstrom_gpucache_sync_trigger(PG_FUNCTION_ARGS)
{
	TriggerData	   *trigdata = (TriggerData *) fcinfo->context;
	TriggerEvent	tg_event;
	GpuCacheDesc   *gc_desc;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as trigger",
			 get_func_name(fcinfo->flinfo->fn_oid));

	tg_event = trigdata->tg_event;
	if (!TRIGGER_FIRED_AFTER(tg_event))
		elog(ERROR, "%s: must be called as AFTER ROW/STATEMENT",
			 get_func_name(fcinfo->flinfo->fn_oid));

	gc_desc = lookupGpuCacheDesc(trigdata->tg_relation);
	if (!gc_desc)
		elog(ERROR, "gpucache is not configured for %s",
			 RelationGetRelationName(trigdata->tg_relation));
	if (gc_desc->buf.data == NULL)
		initStringInfoContext(&gc_desc->buf, CacheMemoryContext);

	if (TRIGGER_FIRED_FOR_ROW(tg_event))
	{
		/* FOR EACH ROW */
		if (TRIGGER_FIRED_BY_INSERT(tg_event))
		{
			__gpuCacheInsertLog(trigdata->tg_trigtuple, gc_desc);
		}
		else if (TRIGGER_FIRED_BY_UPDATE(trigdata->tg_event))
		{
			__gpuCacheDeleteLog(trigdata->tg_trigtuple, gc_desc);
			__gpuCacheInsertLog(trigdata->tg_newtuple, gc_desc);
		}
		else if (TRIGGER_FIRED_BY_DELETE(trigdata->tg_event))
		{
			__gpuCacheDeleteLog(trigdata->tg_trigtuple, gc_desc);
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
			__gpuCacheTruncateLog(gc_desc);
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
	GpuCacheDesc *gc_desc;
	CUresult	rc = CUDA_ERROR_INVALID_VALUE;

	rel = table_open(table_oid, RowExclusiveLock);
	gc_desc = lookupGpuCacheDesc(rel);
	if (gc_desc)
	{
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
	GpuCacheDesc *gc_desc;
	CUresult	rc = CUDA_ERROR_INVALID_VALUE;

	rel = table_open(table_oid, RowExclusiveLock);
	gc_desc = lookupGpuCacheDesc(rel);
	if (gc_desc)
	{
		rc = gpuCacheInvokeCompaction(gc_desc->gc_sstate, false);
	}
	table_close(rel, AccessShareLock);

	PG_RETURN_INT32(rc);
}

/*
 * pgstrom_gpucache_info
 */
static List *
__pgstrom_gpucache_info(void)
{
	slock_t	   *lock = &gcache_shared_head->gcache_sstate_lock;
	dlist_head *slot;
	dlist_iter	iter;
	int			hindex;
	GpuCacheSharedState *gc_sstate;
	GpuCacheSharedState *gc_temp;
	List	   *results = NIL;

	SpinLockAcquire(lock);
	PG_TRY();
	{
		for (hindex = 0; hindex < GPUCACHE_SHARED_DESC_NSLOTS; hindex++)
		{
			slot = &gcache_shared_head->gcache_sstate_slot[hindex];

			dlist_foreach (iter, slot)
			{
				gc_sstate = dlist_container(GpuCacheSharedState,
											chain, iter.cur);
				gc_temp = pmemdup(gc_sstate, sizeof(GpuCacheSharedState));
				results = lappend(results, gc_temp);
			}
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(lock);

	return results;
}

Datum
pgstrom_gpucache_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	GpuCacheSharedState *gc_sstate;
	List	   *info_list;
	Datum		values[14];
	bool		isnull[14];
	HeapTuple	tuple;
	char	   *database_name;
	char	   *options;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc	tupdesc;
		MemoryContext oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);
		tupdesc = CreateTemplateTupleDesc(14);
		TupleDescInitEntry(tupdesc,  1, "database_oid",
						   OIDOID, -1, 0);
		TupleDescInitEntry(tupdesc,  2, "database_name",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc,  3, "table_oid",
						   OIDOID, -1, 0);
		TupleDescInitEntry(tupdesc,  4, "table_name",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc,  5, "signature",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc,  6, "gpu_main_sz",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc,  7, "gpu_extra_sz",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc,  8, "redo_write_ts",
						   TIMESTAMPTZOID, -1, 0);
		TupleDescInitEntry(tupdesc,  9, "redo_write_nitems",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 10, "redo_write_pos",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 11, "redo_read_nitems",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 12, "redo_read_pos",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 13, "redo_sync_pos",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 14, "config_options",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);
		fncxt->user_fctx = __pgstrom_gpucache_info();

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();
	info_list = (List *)fncxt->user_fctx;
	if (info_list == NIL)
		SRF_RETURN_DONE(fncxt);
	gc_sstate = linitial(info_list);
	fncxt->user_fctx = list_delete_first(info_list);

	memset(isnull, 0, sizeof(isnull));
	database_name = get_database_name(gc_sstate->database_oid);

	values[0] = ObjectIdGetDatum(gc_sstate->database_oid);
	values[1] = CStringGetTextDatum(database_name);
	values[2] = ObjectIdGetDatum(gc_sstate->table_oid);
	values[3] = CStringGetTextDatum(gc_sstate->table_name);
	values[4] = Int8GetDatum(gc_sstate->signature);
	values[5] = Int8GetDatum(gc_sstate->gpu_main_size);
	values[6] = Int8GetDatum(gc_sstate->gpu_extra_size);
	values[7] = TimestampGetDatum(gc_sstate->redo_write_timestamp);
	values[8] = Int8GetDatum(gc_sstate->redo_write_nitems);
	values[9] = Int8GetDatum(gc_sstate->redo_write_pos);
	values[10] = Int8GetDatum(gc_sstate->redo_read_nitems);
	values[11] = Int8GetDatum(gc_sstate->redo_read_pos);
	values[12] = Int8GetDatum(gc_sstate->redo_sync_pos);

	if (gc_sstate->cuda_dindex >= 0 &&
		gc_sstate->cuda_dindex < numDevAttrs)
	{
		options = psprintf("gpu_device_id=%d,"
						   "max_num_rows=%ld,"
						   "redo_buffer_size=%zu,"
						   "gpu_sync_interval=%d,"
						   "gpu_sync_threshold=%zu",
						   devAttrs[gc_sstate->cuda_dindex].DEV_ID,
						   gc_sstate->max_num_rows,
						   gc_sstate->redo_buffer_size,
						   gc_sstate->gpu_sync_interval,
						   gc_sstate->gpu_sync_threshold);
		values[13] = CStringGetTextDatum(options);
	}
	else
	{
		isnull[13] = true;
	}
	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);
	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}

/* ---------------------------------------------------------------- *
 *
 * Executor callbacks
 *
 * ---------------------------------------------------------------- */

/* GpuCacheState - executor state object */
struct GpuCacheState
{
	pg_atomic_uint32	__gc_fetch_count;
	pg_atomic_uint32   *gc_fetch_count;
	GpuCacheDesc	   *gc_desc;
	Datum				signature;
	GpuCacheOptions		gc_options;
};

GpuCacheState *
ExecInitGpuCache(ScanState *ss, int eflags, Bitmapset *outer_refs)
{
	Relation		relation = ss->ss_currentRelation;
	Datum			signature;
	GpuCacheOptions	gc_options;
	GpuCacheState  *gcache_state;

	if (!relation)
		return NULL;
	/* GpuCache is not workable on hot-standby server */
	if (RecoveryInProgress())
	{
		elog(DEBUG2, "gpucache: not valid in hot-standby slave server");
		return NULL;
	}
	/* only READ COMMITTED transaction can use GpuCache */
	if (XactIsoLevel > XACT_READ_COMMITTED)
	{
		elog(DEBUG2, "gpucache: not valid in serializable/repeatable-read transaction");
		return NULL;
	}
	/* table must be configured for GpuCache */
	signature = gpuCacheTableSignature(relation, &gc_options);
	if (signature == 0UL)
	{
		elog(DEBUG2, "gpucache: table '%s' is not configured - check row/statement triggers with pgstrom.gpucache_sync_trigger()",
			 RelationGetRelationName(relation));
		return NULL;
	}
	gcache_state = palloc0(sizeof(GpuCacheState));
	gcache_state->gc_fetch_count = &gcache_state->__gc_fetch_count;
	gcache_state->gc_desc = NULL;
	gcache_state->signature = signature;
	memcpy(&gcache_state->gc_options, &gc_options, sizeof(GpuCacheOptions));

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
	GpuCacheState	   *gcache_state = gts->gc_state;
	Relation			relation = gts->css.ss.ss_currentRelation;
	pgstrom_data_store *pds = NULL;

	if (pg_atomic_fetch_add_u32(gcache_state->gc_fetch_count, 1) == 0)
	{
		GpuCacheDesc   *gc_desc = gcache_state->gc_desc;
		GpuCacheSharedState *gc_sstate;
		uint64			write_pos;
		uint64			sync_pos = ULONG_MAX;

		if (!gc_desc)
		{
			gc_desc = lookupGpuCacheDesc(relation);
			if (!gc_desc)
				elog(ERROR, "GpuCache on relation '%s' is not available",
					 RelationGetRelationName(relation));
			gcache_state->gc_desc = gc_desc;
		}
		gc_sstate = gc_desc->gc_sstate;

		SpinLockAcquire(&gc_sstate->redo_lock);
		write_pos = gc_sstate->redo_write_pos;
		if (gc_sstate->redo_sync_pos < gc_sstate->redo_write_pos)
			sync_pos = gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
		SpinLockRelease(&gc_sstate->redo_lock);

		/* redo_write_pos == 0 means that the table is empty */
		if (write_pos != 0)
		{
			if (sync_pos != ULONG_MAX)
				gpuCacheInvokeApplyRedo(gc_sstate, sync_pos, false);
			pds = __ExecScanChunkGpuCache(gts, gc_desc);
		}
	}
	return pds;
}

void
ExecReScanGpuCache(GpuCacheState *gcache_state)
{
	pg_atomic_write_u32(gcache_state->gc_fetch_count, 0);
}

void
ExecEndGpuCache(GpuCacheState *gcache_state)
{
	/* nothing to do */
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
				Relation rel, ExplainState *es)
{
	GpuCacheOptions *gc_options = &gcache_state->gc_options;
	char		temp[1024];
	size_t		gpu_main_size = 0UL;
	size_t		gpu_extra_size = 0UL;

	/* GPU memory usage */
	if (gcache_state->gc_desc)
	{
		GpuCacheDesc   *gc_desc = gcache_state->gc_desc;
		GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;

		gpu_main_size = gc_sstate->gpu_main_size;
		gpu_extra_size = gc_sstate->gpu_extra_size;
	}
	else
	{
		GpuCacheSharedState *gc_sstate;
		slock_t	   *lock = &gcache_shared_head->gcache_sstate_lock;

		SpinLockAcquire(lock);
		gc_sstate = lookupGpuCacheSharedState(MyDatabaseId,
											  RelationGetRelid(rel),
											  gcache_state->signature);
		if (gc_sstate)
		{
			gpu_main_size = gc_sstate->gpu_main_size;
			gpu_extra_size = gc_sstate->gpu_extra_size;
		}
		SpinLockRelease(lock);
	}

	/* config options */
	if (gc_options->cuda_dindex >= 0 &&
		gc_options->cuda_dindex < numDevAttrs)
	{
		if (!pgstrom_regression_test_mode)
		{
			sprintf(temp, "%s [max_num_rows: %ld, main: %s, extra: %s]",
					devAttrs[gc_options->cuda_dindex].DEV_NAME,
					gc_options->max_num_rows,
					format_numeric(gpu_main_size),
					format_numeric(gpu_extra_size));
		}
		else
		{
			sprintf(temp, "GPU%d [max_num_rows: %ld, main: %s, extra: %s]",
					gc_options->cuda_dindex,
					gc_options->max_num_rows,
					format_numeric(gpu_main_size),
					format_numeric(gpu_extra_size));
		}
		ExplainPropertyText("GPU Cache", temp, es);
	}
	else
	{
		ExplainPropertyText("GPU Cache", "invalid device", es);
	}

	if (es->verbose)
	{
		int		gpu_device_id = -1;

		if (gc_options->cuda_dindex >= 0 &&
			gc_options->cuda_dindex < numDevAttrs)
			gpu_device_id = devAttrs[gc_options->cuda_dindex].DEV_ID;

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			snprintf(temp, sizeof(temp),
					 "gpu_device_id=%d,"
					 "max_num_rows=%ld,"
					 "redo_buffer_size=%zu,"
					 "gpu_sync_interval=%d,"
					 "gpu_sync_threshold=%zu",
					 gpu_device_id,
					 gc_options->max_num_rows,
					 gc_options->redo_buffer_size,
					 gc_options->gpu_sync_interval,
					 gc_options->gpu_sync_threshold);
			ExplainPropertyText("GPU Cache Options", temp, es);
		}
		else
		{
			ExplainPropertyInteger("GPU Cache Options:gpu_device_id", NULL,
								   gpu_device_id, es);
			ExplainPropertyInteger("GPU Cache Options:max_num_rows", NULL,
								   gc_options->max_num_rows, es);
			ExplainPropertyInteger("GPU Cache Options:redo_buffer_size", NULL,
								   gc_options->redo_buffer_size, es);
			ExplainPropertyInteger("GPU Cache Options:gpu_sync_threshold", NULL,
								   gc_options->gpu_sync_threshold, es);
			ExplainPropertyInteger("GPU Cache Options:gpu_sync_interval", "s",
								   gc_options->gpu_sync_interval, es);
		}
	}
}

CUresult
gpuCacheMapDeviceMemory(GpuContext *gcontext,
						pgstrom_data_store *pds)
{
	GpuCacheSharedState *gc_sstate = pds->gc_sstate;
	CUdeviceptr	m_kds_main = 0UL;
	CUdeviceptr	m_kds_extra = 0UL;
	CUresult	rc = CUDA_ERROR_NOT_MAPPED;

	Assert(pds->kds.format == KDS_FORMAT_COLUMN);
	pthreadRWLockReadLock(&gc_sstate->gpu_buffer_lock);
	if (gc_sstate->gpu_main_devptr != 0UL)
	{
		rc = gpuIpcOpenMemHandle(gcontext,
								 &m_kds_main,
								 gc_sstate->gpu_main_mhandle,
								 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
			goto out_unlock;

		if (gc_sstate->gpu_extra_devptr != 0UL)
		{
			rc = gpuIpcOpenMemHandle(gcontext,
									 &m_kds_extra,
									 gc_sstate->gpu_extra_mhandle,
									 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
			if (rc != CUDA_SUCCESS)
			{
				gpuIpcCloseMemHandle(gcontext, m_kds_main);
				goto out_unlock;
			}
		}
		pds->m_kds_main  = m_kds_main;
		pds->m_kds_extra = m_kds_extra;

		return CUDA_SUCCESS;
	}
out_unlock:
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);
	return rc;
}

void
gpuCacheUnmapDeviceMemory(GpuContext *gcontext,
						  pgstrom_data_store *pds)
{
	GpuCacheSharedState *gc_sstate = pds->gc_sstate;

	Assert(pds->kds.format == KDS_FORMAT_COLUMN);
	if (pds->m_kds_main != 0UL)
	{
		gpuIpcCloseMemHandle(gcontext, pds->m_kds_main);
		pds->m_kds_main = 0UL;
	}
	if (pds->m_kds_extra != 0UL)
	{
		gpuIpcCloseMemHandle(gcontext, pds->m_kds_extra);
		pds->m_kds_extra = 0UL;
	}
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);	
}

/*
 * gpuCacheObjectAccess
 *
 * This callback marks drop_on_commit / drop_on_rollback for the pending
 * GpuCache entries on DDL commands.
 */
static void
__gpuCacheCallbackOnAlterTable(Oid table_oid)
{
	Datum		signature_old;
	Datum		signature_new;
	GpuCacheOptions options_old;
	GpuCacheOptions options_new;
	GpuCacheDesc *gc_desc;

	signature_old = gpuCacheTableSignatureSnapshot(table_oid, NULL,
												   &options_old);
	signature_new = gpuCacheTableSignatureSnapshot(table_oid, SnapshotSelf,
												   &options_new);
	elog(LOG, "__gpuCacheCallbackOnAlterTable: signature %lx -> %lx",
		 signature_old, signature_new);

	if (signature_old != 0UL &&
		signature_old != signature_new)
	{
		gc_desc = lookupGpuCacheDescNoLoad(MyDatabaseId,
										   table_oid,
										   signature_old,
										   &options_old);
		if (gc_desc)
			gc_desc->drop_on_commit = true;
	}

	if (signature_new != 0UL &&
		signature_new != signature_old)
	{
		gc_desc = lookupGpuCacheDescNoLoad(MyDatabaseId,
										   table_oid,
										   signature_new,
										   &options_new);
		if (gc_desc)
			gc_desc->drop_on_rollback = true;
	}
}

/*
 * gpuCacheObjectAccess, and related...
 */
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
	GpuCacheOptions gc_options;
	GpuCacheDesc *gc_desc;

	signature = gpuCacheTableSignatureSnapshot(table_oid, NULL,
											   &gc_options);
	if (signature != 0UL)
	{
		gc_desc = lookupGpuCacheDescNoLoad(MyDatabaseId,
										   table_oid,
										   signature,
										   &gc_options);
		if (gc_desc)
			gc_desc->drop_on_commit = true;
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
			elog(LOG, "pid=%u OAT_POST_ALTER (pg_class, objectId=%u, subId=%d)", getpid(), objectId, subId);
			__gpuCacheCallbackOnAlterTable(objectId);
		}
		else if (classId == TriggerRelationId)
		{
			elog(LOG, "pid=%u OAT_POST_ALTER (pg_trigger, objectId=%u)", getpid(), objectId);
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
	//elog(LOG, "pid=%u: gpuCacheRelcacheCallback (table_oid=%u)", getpid(), relid);
	gpuCacheTableSignatureInvalidation(relid);
}

static void
gpuCacheSyscacheCallback(Datum arg, int cacheid, uint32 hashvalue)
{
	//elog(LOG, "pid=%u: gpuCacheSyscacheCallback (cacheid=%u)", getpid(), cacheid);
	__gpucache_sync_trigger_function_oid = InvalidOid;
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
	if (hash_get_num_entries(gcache_descriptors_htab) > 0 &&
		(event == XACT_EVENT_COMMIT || event == XACT_EVENT_ABORT))
	{
		TransactionId	curr_xid = GetCurrentTransactionIdIfAny();
		HASH_SEQ_STATUS	hseq;
		GpuCacheDesc   *gc_desc;
		bool			normal_commit = (event == XACT_EVENT_COMMIT);

		hash_seq_init(&hseq, gcache_descriptors_htab);
		while ((gc_desc = hash_seq_search(&hseq)) != NULL)
		{
			if (gc_desc->xid == curr_xid)
				releaseGpuCacheDesc(gc_desc, normal_commit);
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
	if (hash_get_num_entries(gcache_descriptors_htab) > 0 &&
		event == SUBXACT_EVENT_ABORT_SUB)
	{
		TransactionId	curr_xid = GetCurrentTransactionIdIfAny();
		HASH_SEQ_STATUS	hseq;
		GpuCacheDesc   *gc_desc;

		hash_seq_init(&hseq, gcache_descriptors_htab);
		while ((gc_desc = hash_seq_search(&hseq)) != NULL)
		{
			if (gc_desc->xid == curr_xid)
				releaseGpuCacheDesc(gc_desc, false);
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
 * __gpuCacheLoadCudaModule
 */
static CUresult
__gpuCacheLoadCudaModule(void)
{
#ifndef DPGSTROM_DEBUG_BUILD
	const char	   *path = PGSHAREDIR "/pg_strom/cuda_gcache.fatbin";
#else
	const char	   *path = PGSHAREDIR "/pg_strom/cuda_gcache.gfatbin";
#endif
	int				rawfd = -1;
	struct stat		stat_buf;
	ssize_t			nbytes;
	char		   *image;
	CUresult		rc = CUDA_ERROR_SYSTEM_NOT_READY;
	CUmodule		cuda_module = NULL;

	rawfd = open(path, O_RDONLY);
	if (rawfd < 0)
		elog(ERROR, "failed on open('%s'): %m", path);

	if (fstat(rawfd, &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", path);

	image = alloca(stat_buf.st_size + 1);
	nbytes = __readFile(rawfd, image, stat_buf.st_size);
	if (nbytes != stat_buf.st_size)
		elog(ERROR, "failed on __readFile('%s'): %m", path);
	image[nbytes] = '\0';

	rc = cuModuleLoadFatBinary(&cuda_module, image);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleLoadFatBinary: %s", errorText(rc));

	rc = cuModuleGetFunction(&gcache_kfunc_init_empty,
							 cuda_module,
							 "kern_gpucache_init_empty");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&gcache_kfunc_apply_redo,
							 cuda_module,
							 "kern_gpucache_apply_redo");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&gcache_kfunc_compaction,
							 cuda_module,
							 "kern_gpucache_compaction");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/* ok, all green */
	gcache_cuda_module = cuda_module;

	return CUDA_SUCCESS;
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
gpuCacheAllocDeviceMemory(GpuCacheSharedState *gc_sstate)
{
	CUdeviceptr		m_main = 0;
	CUdeviceptr		m_extra = 0;
	CUresult		rc;
	int				grid_sz, block_sz;
	void		   *kern_args[2];
	cl_uint			nrooms = gc_sstate->kds_head.nrooms;
	cl_uint			nslots = gc_sstate->kds_head.nslots;
	size_t			main_sz = 0;
	size_t			head_sz;

	if (gc_sstate->gpu_main_devptr != 0UL)
		return CUDA_SUCCESS;

	/* main portion of the device buffer */
	main_sz = (PAGE_ALIGN(gc_sstate->kds_head.length) +
			   PAGE_ALIGN(offsetof(kern_gpucache_rowhash, slots[nslots])) +
			   PAGE_ALIGN(sizeof(uint32) * nrooms));
	
	rc = cuMemAlloc(&m_main, main_sz);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuMemAlloc(%zu): %s",
			 main_sz, errorText(rc));
		goto error_0;
	}

	head_sz = KERN_DATA_STORE_HEAD_LENGTH(&gc_sstate->kds_head);
	rc = cuMemcpyHtoD(m_main, &gc_sstate->kds_head, head_sz);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuMemcpyHtoD: %s", errorText(rc));
		goto error_1;
	}

	rc = cuIpcGetMemHandle(&gc_sstate->gpu_main_mhandle, m_main);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuIpcGetMemHandle: %s", errorText(rc));
		goto error_1;
	}

	/* extra buffer, if any */
	if (gc_sstate->kds_extra_sz > 0)
	{
		kern_data_extra	kds_extra;

		memset(&kds_extra, 0, offsetof(kern_data_extra, data));
		kds_extra.length = gc_sstate->kds_extra_sz;
		kds_extra.usage = offsetof(kern_data_extra, data);

		rc = cuMemAlloc(&m_extra, kds_extra.length);
		if (rc != CUDA_SUCCESS)
		{
			elog(LOG, "gpucache: failed on cuMemAlloc(%zu): %s",
				 kds_extra.length, errorText(rc));
			goto error_1;
		}

		rc = cuMemcpyHtoD(m_extra, &kds_extra,
						  offsetof(kern_data_extra, data));
		if (rc != CUDA_SUCCESS)
		{
			elog(LOG, "gpucache: failed on cuMemcpyHtoD: %s", errorText(rc));
			goto error_2;
		}

		rc = cuIpcGetMemHandle(&gc_sstate->gpu_extra_mhandle, m_extra);
		if (rc != CUDA_SUCCESS)
		{
			elog(LOG, "gpucache: failed on cuIpcGetMemHandle: %s", errorText(rc));
			goto error_2;
		}
	}

	/* init kds/extra */
	rc = __gpuOptimalBlockSize(&grid_sz,
							   &block_sz,
							   gcache_kfunc_init_empty,
							   gc_sstate->cuda_dindex,
							   0, 0);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on __gpuOptimalBlockSize: %s", errorText(rc));
		goto error_2;
	}

	kern_args[0] = &m_main;
	kern_args[1] = &m_extra;
	rc = cuLaunchKernel(gcache_kfunc_init_empty,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuLaunchKernel: %s", errorText(rc));
		goto error_2;
	}

	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "gpucache: failed on cuStreamSynchronize: %s", errorText(rc));
		goto error_2;
	}
	
	elog(LOG, "gpucache: AllocMemory %s:%lx (main_sz=%zu, extra_sz=%zu)",
		 gc_sstate->table_name,
		 gc_sstate->signature,
		 main_sz,
		 gc_sstate->kds_extra_sz);

	gc_sstate->gpu_main_size = main_sz;
	gc_sstate->gpu_extra_size = gc_sstate->kds_extra_sz;
	gc_sstate->gpu_main_devptr = m_main;
	gc_sstate->gpu_extra_devptr = m_extra;

	SpinLockAcquire(&gcache_shared_head->gcache_sstate_lock);
	gc_sstate->refcnt += 2;
	SpinLockRelease(&gcache_shared_head->gcache_sstate_lock);

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
gpuCacheBgWorkerExecCompactionNoLock(GpuCacheSharedState *gc_sstate)
{
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
		elog(ERROR, "failed on cuMemAllocManaged: %s", errorText(rc));

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
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));

	/* check status of the kernel execution status */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamSynchronize: %s", errorText(rc));

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
		elog(ERROR, "failed on cuMemAlloc(%zu): %s",
			 h_extra.length, errorText(rc));

	rc = cuIpcGetMemHandle(&new_mhandle, m_new_extra);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuIpcGetMemHandle: %s", errorText(rc));

	rc = cuMemcpyHtoD(m_new_extra, &h_extra, offsetof(kern_data_extra, data));
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoD: %s", errorText(rc));

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
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));

	/* check status of the kernel execution status */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamSynchronize: %s", errorText(rc));

	rc = cuMemcpyDtoH(&h_extra, m_new_extra, offsetof(kern_data_extra, data));
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", errorText(rc));

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
	if (m_new_extra != 0UL)
		cuMemFree(m_new_extra);
	if (m_try_extra != 0UL)
		cuMemFree(m_try_extra);
	return rc;
}

static inline CUresult
gpuCacheBgWorkerExecCompaction(GpuCacheSharedState *gc_sstate)
{
	CUresult	rc;

	rc = gpuCacheLoadCudaModule();
	if (rc != CUDA_SUCCESS)
		return rc;

	pthreadRWLockWriteLock(&gc_sstate->gpu_buffer_lock);
	rc = gpuCacheBgWorkerExecCompactionNoLock(gc_sstate);
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);

	return rc;
}

/*
 * GCACHE_BGWORKER_CMD__APPLY_REDO command
 */
static CUresult
__gpuCacheSetupRedoLogBuffer(GpuCacheSharedState *gc_sstate, uint64 end_pos,
							 CUdeviceptr *p_m_redo)
{
	kern_gpucache_redolog *h_redo = NULL;
	char		   *base = gc_sstate->redo_buffer;
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
		elog(ERROR, "failed on cuMemAllocManaged(%zu): %s",
			 length, errorText(rc));

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
	if (gc_sstate->redo_sync_pos < tail_pos)
		gc_sstate->redo_sync_pos = tail_pos;
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
__gpuCacheLaunchApplyRedoKernel(GpuCacheSharedState *gc_sstate, CUdeviceptr m_redo)
{
	kern_gpucache_redolog *h_redo = (kern_gpucache_redolog *)m_redo;
	int			cuda_dindex = gc_sstate->cuda_dindex;
	int			grid_sz, block_sz;
	int			phase;
	void	   *kern_args[4];
	CUresult	rc;

	rc = __gpuOptimalBlockSize(&grid_sz,
							   &block_sz,
							   gcache_kfunc_apply_redo,
							   cuda_dindex, 0, 0);
	grid_sz = Min(grid_sz, (h_redo->nitems + block_sz - 1) / block_sz);
retry:
	for (phase = 0; phase <= 6; phase++)
	{
		kern_args[0] = &m_redo;
		kern_args[1] = &gc_sstate->gpu_main_devptr;
		kern_args[2] = &gc_sstate->gpu_extra_devptr;
		kern_args[3] = &phase;

		rc = cuLaunchKernel(gcache_kfunc_apply_redo,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							0,
							CU_STREAM_PER_THREAD,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	}

	/* check status of the above kernel execution */
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamSynchronize: %s", errorText(rc));

	if (h_redo->kerror.errcode == ERRCODE_OUT_OF_MEMORY)
	{
		rc = gpuCacheBgWorkerExecCompactionNoLock(gc_sstate);
		if (rc == CUDA_SUCCESS)
		{
			memset(&h_redo->kerror, 0, sizeof(kern_errorbuf));
			goto retry;
		}
	}
	return CUDA_SUCCESS;
}

static CUresult
gpuCacheBgWorkerApplyRedoLog(GpuCacheSharedState *gc_sstate, uint64 end_pos)
{
	CUdeviceptr	m_redo = 0UL;
	CUresult	rc, __rc;

	rc = gpuCacheLoadCudaModule();
	if (rc != CUDA_SUCCESS)
		return rc;

	pthreadRWLockWriteLock(&gc_sstate->gpu_buffer_lock);
	
	rc = gpuCacheAllocDeviceMemory(gc_sstate);
	if (rc != CUDA_SUCCESS)
		goto out_unlock;

	rc = __gpuCacheSetupRedoLogBuffer(gc_sstate, end_pos, &m_redo);
	if (rc != CUDA_SUCCESS)
		goto out_unlock;
	if (m_redo != 0UL)
	{
		rc = __gpuCacheLaunchApplyRedoKernel(gc_sstate, m_redo);

		__rc = cuMemFree(m_redo);
		if (__rc != CUDA_SUCCESS)
			elog(LOG, "failed on cuMemFree: %s", errorText(__rc));
	}
out_unlock:
	pthreadRWLockUnlock(&gc_sstate->gpu_buffer_lock);
	return rc;
}

static CUresult
gpuCacheBgWorkerDropUnload(GpuCacheSharedState *gc_sstate)
{
	CUresult	rc;

	pthreadRWLockWriteLock(&gc_sstate->gpu_buffer_lock);
	elog(LOG, "gpucache: DropUnload at %s:%lx main_sz=%zu extra_sz=%zu",
		 gc_sstate->table_name,
		 gc_sstate->signature,
		 gc_sstate->gpu_main_size,
		 gc_sstate->gpu_extra_size);
	if (gc_sstate->gpu_main_devptr != 0UL || gc_sstate->gpu_extra_devptr != 0UL)
	{
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
		/* decrement, but shall not become zero here */
		SpinLockAcquire(&gcache_shared_head->gcache_sstate_lock);
		Assert(gc_sstate->refcnt >= 4);
		gc_sstate->refcnt -= 2;
		SpinLockRelease(&gcache_shared_head->gcache_sstate_lock);
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
	slock_t		   *sstate_lock = &gcache_shared_head->gcache_sstate_lock;
	dlist_head	   *free_cmds = &gcache_shared_head->bgworker_free_cmds;
	dlist_head	   *cmd_queue = &gcache_shared_head->bgworkers[cuda_dindex].cmd_queue;
	dlist_node	   *dnode;
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

	SpinLockAcquire(sstate_lock);
	gc_sstate = lookupGpuCacheSharedState(cmd->database_oid,
										  cmd->table_oid,
										  cmd->signature);
	if (!gc_sstate)
	{
		SpinLockRelease(sstate_lock);
		elog(LOG, "gpucache: (cmd=%c, key=%u:%u:%lx) was not found",
			 cmd->command,
			 cmd->database_oid,
			 cmd->table_oid,
			 cmd->signature);
		rc = CUDA_ERROR_NOT_FOUND;
	}
	else if (gc_sstate->cuda_dindex != cuda_dindex)
	{
		SpinLockRelease(sstate_lock);
		elog(LOG, "gpucache: (cmd=%c, rel=%s:%lx) was not on GPU-%d",
			 cmd->command,
			 gc_sstate->table_name,
			 gc_sstate->signature,
			 gc_sstate->cuda_dindex);
        rc = CUDA_ERROR_INVALID_VALUE;
	}
	else
	{
		gc_sstate->refcnt += 2;
		SpinLockRelease(sstate_lock);

		PG_TRY();
		{
			switch (cmd->command)
			{
				case GCACHE_BGWORKER_CMD__APPLY_REDO:
					rc = gpuCacheBgWorkerApplyRedoLog(gc_sstate, cmd->end_pos);
					break;
				case GCACHE_BGWORKER_CMD__COMPACTION:
					rc = gpuCacheBgWorkerExecCompaction(gc_sstate);
					break;
				case GCACHE_BGWORKER_CMD__DROP_UNLOAD:
					rc = gpuCacheBgWorkerDropUnload(gc_sstate);
					break;
				default:
					rc = CUDA_ERROR_INVALID_VALUE;
					elog(LOG, "Unexpected GpuCache background command: %d",
						 cmd->command);
					break;
			}
		}
		PG_CATCH();
		{
			/* emergency return the result */
			cmd->retval = CUDA_ERROR_UNKNOWN;
			SpinLockAcquire(cmd_lock);
			if (cmd->backend)
				SetLatch(cmd->backend);
			else
				dlist_push_head(free_cmds, &cmd->chain);
			SpinLockRelease(cmd_lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
#ifdef PGSTROM_DEBUG_BUILD
		elog(LOG, "gpucache: (cmd=%c, key=%s:%lx) rc=%d",
			 cmd->command,
			 gc_sstate->table_name,
			 gc_sstate->signature,
			 (int)rc);
#endif
		putGpuCacheSharedState(gc_sstate, false);
	}
	/* return the result */
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
 * pg_strom.gpucache_auto_preload configuration
 */
typedef struct
{
	char   *database_name;
	char   *schema_name;
	char   *table_name;
} GpuCacheAutoPreloadEntry;

static GpuCacheAutoPreloadEntry *gpucache_auto_preload_entries = NULL;
static int		gpucache_auto_preload_num_entries = 0;

/*
 * __gpuCacheAutoPreloadEntryComp
 */
static int
__gpuCacheAutoPreloadEntryComp(const void *__x, const void *__y)
{
	const GpuCacheAutoPreloadEntry *x = __x;
	const GpuCacheAutoPreloadEntry *y = __y;
	int		comp;

	comp = strcmp(x->database_name, y->database_name);
	if (comp == 0)
	{
		comp = strcmp(x->schema_name, y->schema_name);
		if (comp == 0)
			comp = strcmp(x->table_name, y->table_name);
	}
	return comp;
}

/*
 * __parseGpuCacheAutoPreload
 */
static void
__parseGpuCacheAutoPreload(void)
{
	char	   *config;
	char	   *token;
	int			nitems = 0;
	int			nrooms = 0;

	config = alloca(strlen(pgstrom_gpucache_auto_preload) + 1);
	strcpy(config, pgstrom_gpucache_auto_preload);
	config = trim_cstring(config);

	/* special case - auto preloading */
	if (strcmp(config, "*") == 0)
		return;

	for (token = strtok(config, ",");
		 token != NULL;
		 token = strtok(NULL,   ","))
	{
		GpuCacheAutoPreloadEntry *entry;
		char   *database_name = trim_cstring(token);
		char   *schema_name;
		char   *table_name;
		char   *pos;

		pos = strchr(database_name, '.');
		if (!pos)
			elog(ERROR, "pgstrom.gpucache_auto_preload syntax error [%s]",
				 pgstrom_gpucache_auto_preload);
		*pos++ = '\0';
		schema_name = trim_cstring(pos);

		pos = strchr(schema_name, '.');
		if (!pos)
			elog(ERROR, "pgstrom.gpucache_auto_preload syntax error [%s]",
				 pgstrom_gpucache_auto_preload);
		*pos++ = '\0';
		table_name = trim_cstring(pos);

		if (nitems >= nrooms)
		{
			nrooms = 2 * nrooms + 20;
			gpucache_auto_preload_entries
				= realloc(gpucache_auto_preload_entries,
						  sizeof(GpuCacheAutoPreloadEntry) * nrooms);
			if (!gpucache_auto_preload_entries)
				elog(ERROR, "out of memory");
		}
		entry = &gpucache_auto_preload_entries[nitems++];
		entry->database_name = strdup(database_name);
		entry->schema_name = strdup(schema_name);
		entry->table_name = strdup(table_name);
		if (!entry->database_name ||
			!entry->schema_name ||
			!entry->table_name)
			elog(ERROR, "out of memory");
	}
	gpucache_auto_preload_num_entries = nitems;

	/* sort by database_name */
	if (gpucache_auto_preload_num_entries > 0)
	{
		qsort(gpucache_auto_preload_entries,
			  gpucache_auto_preload_num_entries,
			  sizeof(GpuCacheAutoPreloadEntry),
			  __gpuCacheAutoPreloadEntryComp);
	}
}

/*
 * __gpuCacheAutoPreloadConnectDatabaseAny
 */
static int
__gpuCacheAutoPreloadConnectDatabaseAny(int32 *p_start, int32 *p_end)
{
	char	   *database_name;
	Relation	srel;
	SysScanDesc	sscan;
    ScanKeyData	skey;
	int			nkeys;
	HeapTuple	tuple;
	int			nitems = 0;
	int			nrooms = 0;
	int			exit_code = 1;	/* restart again */

	if (gcache_shared_head->gcache_auto_preload_count++ == 0)
	{
		database_name = "template1";
		nkeys = 0;
	}
	else
	{
		database_name = NameStr(gcache_shared_head->gcache_auto_preload_dbname);

		ScanKeyInit(&skey,
					Anum_pg_database_datname,
					BTGreaterStrategyNumber, F_NAMEGT,
					CStringGetDatum(database_name));
		nkeys = 1;
	}

	PG_TRY();
	{
		BackgroundWorkerInitializeConnection(database_name, NULL, 0);
	}
	PG_CATCH();
	{
		ErrorData	   *edata;

		MemoryContextSwitchTo(TopMemoryContext);
		edata = CopyErrorData();

		elog(LOG, "failed to connect database [%s], stop preloading - %s (%s:%d)",
			 database_name,
			 edata->message,
			 edata->filename,
			 edata->lineno);
		proc_exit(0);	/* stop to restart bgworker any more */
	}
	PG_END_TRY();
	StartTransactionCommand();
	PushActiveSnapshot(GetTransactionSnapshot());
	srel = table_open(DatabaseRelationId, AccessShareLock);
	sscan = systable_beginscan(srel,
							   DatabaseNameIndexId,
							   true,
							   NULL,
							   nkeys, &skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Form_pg_database dat = (Form_pg_database) GETSTRUCT(tuple);

		if (!dat->datistemplate && dat->datallowconn)
		{
			/* save the next database */
			memcpy(&gcache_shared_head->gcache_auto_preload_dbname,
				   &dat->datname,
				   sizeof(NameData));
			break;
		}
	}
	if (!HeapTupleIsValid(tuple))
		exit_code = 0;	/* stop preloading */
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	/* build gpucache_auto_preload_entries on this database */
	database_name = get_database_name(MyDatabaseId);
	srel = table_open(RelationRelationId, AccessShareLock);
	sscan = systable_beginscan(srel, InvalidOid, false, NULL, 0, NULL);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		GpuCacheAutoPreloadEntry *entry;
		Form_pg_class	pg_class = (Form_pg_class) GETSTRUCT(tuple);
		Oid				namespace_oid = pg_class->relnamespace;

		if (namespace_oid == PG_CATALOG_NAMESPACE)
			continue;
		if (__gpuCacheTableSignatureSnapshot(tuple, NULL, NULL) == 0)
			continue;

		while (nitems >= nrooms)
		{
			nrooms = 2 * nrooms + 20;
			gpucache_auto_preload_entries
				= realloc(gpucache_auto_preload_entries,
						  sizeof(GpuCacheAutoPreloadEntry) * nrooms);
			if (!gpucache_auto_preload_entries)
				elog(ERROR, "out of memory");
		}
		entry = &gpucache_auto_preload_entries[nitems++];
        entry->database_name = strdup(database_name);
		entry->schema_name = strdup(get_namespace_name(namespace_oid));
		entry->table_name = strdup(NameStr(pg_class->relname));
	}
	gpucache_auto_preload_num_entries = nitems;

	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	PopActiveSnapshot();
	CommitTransactionCommand();
	
	*p_start = 0;
	*p_end   = nitems;

	return exit_code;
}

/*
 * gpuCacheAutoPreloadDatabaseName
 */
static int
gpuCacheAutoPreloadConnectDatabase(int32 *p_start, int32 *p_end)
{
	GpuCacheAutoPreloadEntry *entry;
	const char *database_name;
	int32		start;
	int32		curr;

	if (!gpucache_auto_preload_entries)
		return __gpuCacheAutoPreloadConnectDatabaseAny(p_start, p_end);

	start = curr = gcache_shared_head->gcache_auto_preload_count;
	if (start >= gpucache_auto_preload_num_entries)
		proc_exit(0);	/* no more preloading */

	database_name = gpucache_auto_preload_entries[start].database_name;
	while (curr < gpucache_auto_preload_num_entries)
	{
		entry = &gpucache_auto_preload_entries[curr];

		if (strcmp(database_name, entry->database_name) != 0)
			break;
	}
	gcache_shared_head->gcache_auto_preload_count = curr;

	BackgroundWorkerInitializeConnection(database_name, NULL, 0);

	*p_start = start;
	*p_end   = curr;

	return (curr >= gpucache_auto_preload_num_entries ? 0 : 1);
}

/*
 * gpuCacheStartupPreloader
 */
void
gpuCacheStartupPreloader(Datum arg)
{
	int32		start;
	int32		end;
	int32		index;
	int			exit_code;

	BackgroundWorkerUnblockSignals();

	exit_code = gpuCacheAutoPreloadConnectDatabase(&start, &end);
	StartTransactionCommand();
	GetCurrentTransactionId();
	for (index=start; index < end; index++)
	{
		GpuCacheAutoPreloadEntry *entry = &gpucache_auto_preload_entries[index];
		RangeVar		rvar;
		Relation		rel;

		memset(&rvar, 0, sizeof(RangeVar));
		rvar.type = T_RangeVar;
		rvar.schemaname = entry->schema_name;
		rvar.relname = entry->table_name;

		rel = table_openrv(&rvar, AccessShareLock);
		(void)lookupGpuCacheDesc(rel);
		table_close(rel, NoLock);

		elog(LOG, "gpucache: auto preload '%s.%s' (DB: %s)",
			 entry->schema_name,
			 entry->table_name,
			 entry->database_name);
	}
	CommitTransactionCommand();

	proc_exit(exit_code);
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
	BackgroundWorker worker;
	HASHCTL		hctl;

	/* GUC: pg_strom.gpucache_auto_preload */
	DefineCustomStringVariable("pg_strom.gpucache_auto_preload",
							   "list of tables or '*' for GpuCache preloading",
							   NULL,
							   &pgstrom_gpucache_auto_preload,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	/* GUC: pg_strom.enable_gpucache */
	DefineCustomBoolVariable("pg_strom.enable_gpucache",
							 "Enables GpuCache as a data source for scan",
							 NULL,
							 &enable_gpucache,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup local hash tables */
	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = offsetof(GpuCacheDesc, xid) + sizeof(TransactionId);
	hctl.entrysize = sizeof(GpuCacheDesc);
	hctl.hcxt = CacheMemoryContext;
	gcache_descriptors_htab = hash_create("GpuCache Descriptors", 48, &hctl,
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
	if (pgstrom_gpucache_auto_preload)
	{
		__parseGpuCacheAutoPreload();

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
				 "gpuCacheStartupPreloader");
		worker.bgw_main_arg = 0;
		RegisterBackgroundWorker(&worker);
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
