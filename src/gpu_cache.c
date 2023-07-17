/*
 * gpu_cache.c
 *
 * GPU data cache that syncronizes a PostgreSQL table
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"

/*
 * GpuCacheControlCommand
 */
#define GCACHE_CONTROL_CMD__APPLY_REDO		'A'
#define GCACHE_CONTROL_CMD__COMPACTION		'C'
#define GCACHE_CONTROL_CMD__DROP_UNLOAD		'D'
#define GCACHE_CONTROL_CMD__ERRORBUF_SIZE	120

typedef struct
{
	dlist_node	chain;
	GpuCacheIdent ident;
	Latch	   *backend;
	int			command;	/* one of GCACHE_CONTROL_CMD__* */
	uint64_t	end_pos;	/* for APPLY_REDO */
	int			errcode;
	char		errbuf[GCACHE_CONTROL_CMD__ERRORBUF_SIZE];
} GpuCacheControlCommand;

/*
 * GpuCacheSharedHead (shared structure; static)
 */
typedef struct
{
	/* pg_strom.gpucache_auto_preload related */
	int32			gcache_auto_preload_count;
	NameData		gcache_auto_preload_dbname;
	/* Mutex for creation of GpuCacheSharedState shared-memory segment */
	pthread_mutex_t	gcache_sstate_mutex;
	/* IPC to GpuService background workers */
	pthread_mutex_t	gcache_cmd_mutex;
	dlist_head		gcache_free_cmds;
	GpuCacheControlCommand __gcache_control_cmds[100];
	struct {
		pthread_cond_t	cond;
		dlist_head		queue;
	} gpus[FLEXIBLE_ARRAY_MEMBER];		/* per GPU device */
} GpuCacheSharedHead;

/*
 * GpuCacheOptions
 */
typedef struct
{
	Oid			tg_sync_row;
	int			cuda_dindex;
	int32		gpu_sync_interval;
	size_t		gpu_sync_threshold;
	int64		max_num_rows;
	int64		rowid_hash_nslots;
	size_t		redo_buffer_size;
} GpuCacheOptions;

INLINE_FUNCTION(bool)
GpuCacheOptionsEqual(const GpuCacheOptions *a, const GpuCacheOptions *b)
{
	return (a->tg_sync_row        == b->tg_sync_row &&
			a->cuda_dindex        == b->cuda_dindex &&
			a->gpu_sync_interval  == b->gpu_sync_interval &&
			a->gpu_sync_threshold == b->gpu_sync_threshold &&
			a->max_num_rows       == b->max_num_rows &&
			a->rowid_hash_nslots  == b->rowid_hash_nslots &&
			a->redo_buffer_size   == b->redo_buffer_size);
}

/*
 * GpuCacheSharedState (shared structure; dynamic memory mapped)
 */
typedef struct
{
	char			magic[8];	/* = "GpuCache" */
	GpuCacheIdent	ident;
	char			table_name[NAMEDATALEN];	/* for debug */
	uint64_t		rowid_map_offset;
	uint64_t		redo_buffer_offset;

	/* GpuCache configuration parameters */
	GpuCacheOptions	gc_options;

	/* current status */
#define GCACHE_PHASE__NOT_BUILT			0	/* not built yet */
#define GCACHE_PHASE__IS_EMPTY			1	/* not initial loaded */
#define GCACHE_PHASE__IS_READY			2	/* now ready */
#define GCACHE_PHASE__IS_CORRUPTED		3	/* corrupted */
	pg_atomic_uint32 phase;

	/* device memory allocation (just for information) */
	pg_atomic_uint64 gcache_main_size;
	pg_atomic_uint64 gcache_main_nitems;
    pg_atomic_uint64 gcache_extra_size;
	pg_atomic_uint64 gcache_extra_usage;	/* used in extra buffer (incl dead space) */
	pg_atomic_uint64 gcache_extra_dead;		/* dead space in extra buffer */

	/* rowid-map propertoes */
	pthread_mutex_t	rowid_mutex;
	uint32_t		rowid_next_free;
	uint32_t		rowid_num_free;

	/* redo buffer properties */
	pthread_mutex_t	redo_mutex;
	uint64_t		redo_write_timestamp;
	uint64_t		redo_write_nitems;
	uint64_t		redo_write_pos;
	uint64_t		redo_read_nitems;
	uint64_t		redo_read_pos;
	uint64_t		redo_sync_pos;

	/* schema definitions (KDS_FORMAT_COLUMN) */
	size_t			kds_extra_sz;
	kern_data_store	kds_head;
} GpuCacheSharedState;

#define GpuCacheSharedStateName(nameBuf,nameLen,datOid,relOid,signature) \
	snprintf((nameBuf), (nameLen),										\
			 ".gpucache_p%u_d%u_r%u.%09lx.buf",							\
			 PostPortNumber, (datOid), (relOid), (signature))

/*
 * GpuCacheRowIdItem
 */
typedef struct
{
	uint32_t		next;
	ItemPointerData	ctid;
	uint16_t		__padding__;
} GpuCacheRowIdItem;

INLINE_FUNCTION(char *)
gpuCacheRedoLogBuffer(GpuCacheSharedState *gc_sstate)
{
	return (char *)gc_sstate + gc_sstate->redo_buffer_offset;
}

INLINE_FUNCTION(uint32_t *)
gpuCacheRowIdHashSlot(GpuCacheSharedState *gc_sstate)
{
	return (uint32_t *)((char *)gc_sstate + gc_sstate->rowid_map_offset);
}

INLINE_FUNCTION(GpuCacheRowIdItem *)
gpuCacheRowIdItemArray(GpuCacheSharedState *gc_sstate)
{
	return (GpuCacheRowIdItem *)
		(gpuCacheRowIdHashSlot(gc_sstate) + gc_sstate->gc_options.rowid_hash_nslots);
}

/*
 * GpuCacheLocalMapping
 */
typedef struct
{
	dlist_node		chain;
	GpuCacheIdent	ident;
	int				refcnt;
	GpuCacheSharedState *gc_sstate;
	size_t			mmap_sz;
	/* fields below are valid only GpuService context */
	pthread_rwlock_t gcache_rwlock;
	CUdeviceptr		gcache_main_devptr;
	CUdeviceptr		gcache_extra_devptr;
	ssize_t			gcache_main_size;
	ssize_t			gcache_extra_size;
} GpuCacheLocalMapping;

/*
 * GpuCacheDesc (GpuCache Descriptor per backend)
 */
struct GpuCacheDesc
{
	GpuCacheIdent	ident;
	TransactionId 	xid;
	GpuCacheOptions	gc_options;
	GpuCacheLocalMapping *gc_lmap;
	bool			drop_on_rollback;
	bool			drop_on_commit;
	uint32_t		nitems;
	StringInfoData	buf;	/* array of PendingCtidItem */
};

typedef struct
{
	uint32_t		rowid;
	char			tag;
	ItemPointerData	ctid;
} PendingCtidItem;

/* --- static variables --- */
static char	   *pgstrom_gpucache_auto_preload;		/* GUC */
static bool		pgstrom_enable_gpucache;			/* GUC */
static GpuCacheSharedHead *gcache_shared_head = NULL;
static HTAB	   *gcache_descriptors_htab = NULL;
static HTAB	   *gcache_signatures_htab = NULL;
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static object_access_hook_type object_access_next = NULL;

/* --- function declarations --- */
static bool		__gpuCacheAppendLog(GpuCacheDesc *gc_desc,
									GCacheTxLogCommon *tx_log);
static void		gpuCacheInvokeDropUnload(const GpuCacheDesc *gc_desc,
										 bool is_async);
void	gpuCacheStartupPreloader(Datum arg);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_sync_trigger);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_apply_redo);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_compaction);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_recovery);
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
		Oid			namespace_oid;
		oidvector	argtypes;

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
static bool
__parseSyncTriggerOptions(const char *__config, GpuCacheOptions *gc_options)
{
	int			cuda_dindex = 0;				/* default: GPU0 */
	int			gpu_sync_interval = 5000000L;	/* default: 5sec = 5000000us */
	ssize_t		gpu_sync_threshold = -1;		/* default: auto */
	int64		max_num_rows = (10UL << 20);	/* default: 10M rows */
	int64		rowid_hash_nslots = -1;			/* default: auto */
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

		key = __trim(key);
		value = __trim(value);

		if (strcmp(key, "gpu_device_id") == 0)
		{
			int		i, gpu_device_id;
			char   *end;

			gpu_device_id = strtol(value, &end, 10);
			if (*end != '\0')
			{
				elog(WARNING, "gpucache: invalid option [%s]=[%s]",
					 key, value);
				return false;
			}

			cuda_dindex = -1;
			for (i=0; i < numGpuDevAttrs; i++)
			{
				if (gpuDevAttrs[i].DEV_ID == gpu_device_id)
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
		if (rowid_hash_nslots < 0)
			rowid_hash_nslots = (max_num_rows + max_num_rows / 5);
		gc_options->cuda_dindex       = cuda_dindex;
		gc_options->gpu_sync_interval = gpu_sync_interval;
		gc_options->gpu_sync_threshold = gpu_sync_threshold;
		gc_options->max_num_rows      = max_num_rows;
		gc_options->rowid_hash_nslots = rowid_hash_nslots;
		gc_options->redo_buffer_size  = redo_buffer_size;
	}
	return true;
}

/* ------------------------------------------------------------
 *
 * Routines to manage the table signature
 *
 * In some cases we have to rebuild GpuCache even if the cached-table is still
 * valid. For example, add/remove columns by ALTER TABLE, enables/disables
 * the trigger function to synchronize GpuCache and so on.
 * The table signature is a simple and lightweight way to detect these cases.
 * ------------------------------------------------------------
 */
typedef struct
{
	Oid		reltablespace;
	Oid		relfilenode;	/* if 0, cannot have gpucache */
	int16	relnatts;
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
	Oid			table_oid;
	uint64_t	signature;
	GpuCacheOptions gc_options;
} GpuCacheTableSignatureCache;

static void
__gpuCacheTableSignature(Relation rel, GpuCacheTableSignatureCache *entry)
{
	GpuCacheTableSignatureBuffer *sig;
	TupleDesc	tupdesc = RelationGetDescr(rel);
	int			j, natts = RelationGetNumberOfAttributes(rel);
	size_t		len;
	Form_pg_class rd_rel = RelationGetForm(rel);
	TriggerDesc *trigdesc = rel->trigdesc;

	/* pg_class related */
	if (rd_rel->relkind != RELKIND_RELATION &&
		rd_rel->relkind != RELKIND_PARTITIONED_TABLE)
		goto no_gpu_cache;
	if (rd_rel->relfilenode == 0)
		goto no_gpu_cache;

	/* alloc GpuCacheTableSignatureBuffer */
	len = offsetof(GpuCacheTableSignatureBuffer, attrs[natts]);
	sig = alloca(len);
	memset(sig, 0, len);
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
			if (OidIsValid(sig->gc_options.tg_sync_row))
				goto no_gpu_cache;		/* should not call trigger twice per row */
			if ((trig->tgnargs == 0 &&
				 __parseSyncTriggerOptions(NULL,
										   &sig->gc_options)) ||
				(trig->tgnargs == 1 &&
				 __parseSyncTriggerOptions(trig->tgargs[0],
										   &sig->gc_options)))
			{
				sig->gc_options.tg_sync_row = trig->tgoid;
			}
			else
			{
				goto no_gpu_cache;
			}
		}
	}
	if (!OidIsValid(sig->gc_options.tg_sync_row))
		goto no_gpu_cache;		/* no row sync trigger */

	/* pg_attribute related */
	for (j=0; j < natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

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

static inline uint64_t
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

static uint64_t
__gpuCacheTableSignatureSnapshot(HeapTuple pg_class_tuple,
								 Snapshot snapshot,
								 GpuCacheOptions *gc_options)
{
	GpuCacheTableSignatureBuffer *sig;
	Form_pg_class pg_class = (Form_pg_class) GETSTRUCT(pg_class_tuple);
	Oid			table_oid = pg_class->oid;
	Relation	srel;
	ScanKeyData	skey[2];
	SysScanDesc	sscan;
	HeapTuple	tuple;
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
								TRIGGER_TYPE_AFTER |
								TRIGGER_TYPE_INSERT |
								TRIGGER_TYPE_DELETE |
								TRIGGER_TYPE_UPDATE) &&
			pg_trig->tgfoid == gpucache_sync_trigger_function_oid())
		{
			if (OidIsValid(sig->gc_options.tg_sync_row))
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
			sig->gc_options.tg_sync_row = pg_trig->oid;
		}
	}
	if (!OidIsValid(sig->gc_options.tg_sync_row))
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

static uint64_t
gpuCacheTableSignatureSnapshot(Oid table_oid,
							   Snapshot snapshot,
							   GpuCacheOptions *gc_options)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;
	uint64_t	signature = 0UL;

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
int
baseRelHasGpuCache(PlannerInfo *root, RelOptInfo *baserel)
{
	RangeTblEntry *rte = root->simple_rte_array[baserel->relid];
	int		cuda_dindex = -1;

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

		cuda_dindex = entry->gc_options.cuda_dindex;
	}
	return (pgstrom_enable_gpucache ? cuda_dindex : -1);
}

/*
 * RelationHasGpuCache
 */
bool
RelationHasGpuCache(Relation rel)
{
	if (pgstrom_enable_gpucache)
		return (gpuCacheTableSignature(rel,NULL) != 0UL);
	return false;
}

/* ------------------------------------------------------------
 *
 * Routines to manage GpuCacheSharedState
 *
 * ------------------------------------------------------------
 */
#define GCACHE_SHARED_MAPPING_NSLOTS		79
static pthread_mutex_t	gcache_shared_mapping_lock;
static dlist_head		gcache_shared_mapping_slot[GCACHE_SHARED_MAPPING_NSLOTS];

static inline dlist_head *
__gpuCacheSharedMappingHashSlot(Oid database_oid,
								Oid table_oid,
								uint64_t signature)
{
	GpuCacheIdent	hkey;
	uint32_t		hash;

	hkey.database_oid = database_oid;
	hkey.table_oid    = table_oid;
	hkey.signature    = signature;
	hash = hash_any((unsigned char *)&hkey, sizeof(hkey));
	return &gcache_shared_mapping_slot[hash % GCACHE_SHARED_MAPPING_NSLOTS];
}

/*
 * __setup_kern_data_store_column
 */
static void
__setup_kern_data_store_column(kern_data_store *kds_head,
							   size_t *p_extra_sz,
							   Relation rel,
							   uint32_t nrooms)
{
	TupleDesc	tupdesc = RelationGetDescr(rel);
	kern_colmeta *cmeta;
	size_t		sz, off;
	size_t		unitsz;
	size_t		extra_sz = 0;

	setup_kern_data_store(kds_head, tupdesc, 0, KDS_FORMAT_COLUMN);
	kds_head->table_oid = RelationGetRelid(rel);
	kds_head->column_nrooms = nrooms;
	Assert(kds_head->nr_colmeta > tupdesc->natts);

	off = KDS_HEAD_LENGTH(kds_head);
	for (int j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);
		kern_colmeta   *cmeta = &kds_head->colmeta[j];

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

	/* varlena buffer size */
	if (extra_sz > 0)
	{
		/* 25% margin */
		extra_sz += extra_sz / 4;
		extra_sz += offsetof(kern_data_extra, data);
	}
	*p_extra_sz = extra_sz;
}

/*
 * __openGpuCacheSharedState
 *
 * NOTE: caller must hold gcache_shared_mapping_lock
 */
static GpuCacheLocalMapping *
__openGpuCacheSharedState(Oid database_oid,
						  Oid table_oid,
						  uint64_t signature,
						  char *errbuf, size_t errbuf_sz)
{
	int			fdesc = -1;
	struct stat	stat_buf;
	char		namebuf[MAXPGPATH];
	size_t		off;
	dlist_head *hslot;
	GpuCacheSharedState *gc_sstate = MAP_FAILED;
	GpuCacheLocalMapping *gc_lmap;
	
	GpuCacheSharedStateName(namebuf, MAXPGPATH,
							database_oid,
							table_oid,
							signature);
	fdesc = shm_open(namebuf, O_RDWR, 0600);
	if (fdesc < 0)
	{
		if (errno == ENOENT)
			return MAP_FAILED;
		snprintf(errbuf, errbuf_sz,
				 "failed on shm_open('%s'): %m", namebuf);
		goto bailout;
	}

	if (fstat(fdesc, &stat_buf) != 0)
	{
		snprintf(errbuf, errbuf_sz,
				 "failed on fstat('%s'): %m", namebuf);
		goto bailout;
	}
	if ((stat_buf.st_size & PAGE_MASK) != 0)
	{
		snprintf(errbuf, errbuf_sz,
				 "file-size of '%s' is not aligned: %zu",
				 namebuf, stat_buf.st_size);
		goto bailout;
	}
	gc_sstate = mmap(NULL, stat_buf.st_size,
					 PROT_READ | PROT_WRITE,
					 MAP_SHARED,
					 fdesc, 0);
	if (gc_sstate == MAP_FAILED)
	{
		snprintf(errbuf, errbuf_sz,
				 "failed on mmap('%s', %zu): %m",
				 namebuf, stat_buf.st_size);
		goto bailout;
	}
	/*
	 * wait for completion of the initial setup
	 */
	for (;;)
	{
		uint32_t	phase = pg_atomic_read_u32(&gc_sstate->phase);

		if (phase == GCACHE_PHASE__IS_EMPTY ||
			phase == GCACHE_PHASE__IS_READY)
			break;
		else if (phase == GCACHE_PHASE__IS_CORRUPTED)
		{
			snprintf(errbuf, errbuf_sz,
					 "GpuCacheSharedState is already corrupted");
			goto bailout;
		}
		else
		{
			snprintf(errbuf, errbuf_sz,
					 "GpuCacheSharedState has unknown phase (%u)", phase);
			goto bailout;
		}
		usleep(1000L);		/* 1ms */
	}

	/* simple validation checks */
	if (memcmp(gc_sstate->magic, "GpuCache", 8) != 0)
	{
		snprintf(errbuf, errbuf_sz,
				 "GpuCacheSharedState validation error");
		goto bailout;
	}
	off = PAGE_ALIGN(offsetof(GpuCacheSharedState, kds_head) +
					 KDS_HEAD_LENGTH(&gc_sstate->kds_head));
	if (off != gc_sstate->rowid_map_offset)
	{
		snprintf(errbuf, errbuf_sz,
				 "GpuCacheSharedState validation error");
		goto bailout;
	}
	off += PAGE_ALIGN(sizeof(uint32_t) * gc_sstate->gc_options.rowid_hash_nslots +
					  sizeof(GpuCacheRowIdItem) * gc_sstate->gc_options.max_num_rows);
	if (off != gc_sstate->redo_buffer_offset)
	{
		snprintf(errbuf, errbuf_sz, "GpuCacheSharedState validation error");
		goto bailout;
	}
	off += gc_sstate->gc_options.redo_buffer_size;
	if (off != stat_buf.st_size)
	{
		snprintf(errbuf, errbuf_sz,
				 "GpuCacheSharedState validation error");
		goto bailout;
	}

	/*
	 * Append to the hash-slot
	 * (note: caller already holds gcache_shared_mapping_lock)
	 */
	gc_lmap = calloc(1, sizeof(GpuCacheLocalMapping));
	if (!gc_lmap)
	{
		snprintf(errbuf, errbuf_sz, "out of memory");
		goto bailout;
	}
	gc_lmap->ident.database_oid = database_oid;
	gc_lmap->ident.table_oid = table_oid;
	gc_lmap->ident.signature = signature;
	Assert(GpuCacheIdentEqual(&gc_lmap->ident, &gc_sstate->ident));
	gc_lmap->refcnt = 3;
	gc_lmap->gc_sstate = gc_sstate;
	gc_lmap->mmap_sz = stat_buf.st_size;
	pthreadRWLockInit(&gc_lmap->gcache_rwlock);
	hslot = __gpuCacheSharedMappingHashSlot(database_oid,
											table_oid,
											signature);
	dlist_push_tail(hslot, &gc_lmap->chain);

	close(fdesc);

	return gc_lmap;

bailout:
	if (gc_sstate != MAP_FAILED)
		munmap(gc_sstate, stat_buf.st_size);
	if (fdesc >= 0)
		close(fdesc);
	return MAP_FAILED;
}

/*
 * __resetGpuCacheSharedState
 */
static void
__resetGpuCacheSharedState(GpuCacheSharedState *gc_sstate)
{
	uint32_t   *rowid_hslot = gpuCacheRowIdHashSlot(gc_sstate);
	GpuCacheRowIdItem *rowid_items = gpuCacheRowIdItemArray(gc_sstate);
	uint32_t	rowid_nslots = gc_sstate->gc_options.rowid_hash_nslots;
	uint32_t	rowid_nrooms = gc_sstate->gc_options.max_num_rows;

	/* reset rowid-map */
	pthreadMutexLock(&gc_sstate->rowid_mutex);
	for (uint32_t i=0; i < rowid_nslots; i++)
		rowid_hslot[i] = UINT_MAX;
	for (uint32_t i=1; i < rowid_nrooms; i++)
		rowid_items[i-1].next = i;
	rowid_items[rowid_nrooms-1].next = UINT_MAX;	/* terminator */
	gc_sstate->rowid_next_free = 0;
	gc_sstate->rowid_num_free = rowid_nrooms;
	pthreadMutexUnlock(&gc_sstate->rowid_mutex);

	/* reset redo-log-buffer */
	pthreadMutexLock(&gc_sstate->redo_mutex);
	gc_sstate->redo_write_timestamp = GetCurrentTimestamp();
	gc_sstate->redo_write_nitems = 0;
    gc_sstate->redo_write_pos = 0;
    gc_sstate->redo_read_nitems = 0;
    gc_sstate->redo_read_pos = 0;
    gc_sstate->redo_sync_pos = 0;
	pthreadMutexUnlock(&gc_sstate->redo_mutex);

	/* make this GpuCache available again */
	pg_atomic_init_u32(&gc_sstate->phase, GCACHE_PHASE__IS_EMPTY);
}

/*
 * __createGpuCacheSharedState
 */
static GpuCacheLocalMapping *
__createGpuCacheSharedState(Relation rel,
							uint64_t signature,
							const GpuCacheOptions *gc_options)
{
	TupleDesc	tupdesc = RelationGetDescr(rel);
	int			fdesc = -1;
	size_t		rowid_map_offset;
	size_t		redo_buffer_offset;
	size_t		mmap_sz;
	char		namebuf[MAXPGPATH];
	dlist_head *hslot;
	GpuCacheSharedState *gc_sstate = MAP_FAILED;
	GpuCacheLocalMapping *gc_lmap;

	Assert(signature != 0UL);
	GpuCacheSharedStateName(namebuf,MAXPGPATH,
							MyDatabaseId,
							RelationGetRelid(rel),
							signature);
	mmap_sz = PAGE_ALIGN(offsetof(GpuCacheSharedState, kds_head) +
						 estimate_kern_data_store(tupdesc));
	rowid_map_offset = mmap_sz;
	mmap_sz += PAGE_ALIGN(sizeof(uint32_t) * gc_options->rowid_hash_nslots +
						  sizeof(GpuCacheRowIdItem) * gc_options->max_num_rows);
	redo_buffer_offset = mmap_sz;
	mmap_sz += PAGE_ALIGN(gc_options->redo_buffer_size);

	fdesc = shm_open(namebuf, O_RDWR | O_CREAT | O_EXCL | O_TRUNC, 0600);
	if (fdesc < 0)
	{
		char	errbuf[512];

		if (errno != EEXIST)
			elog(ERROR, "failed on shm_open('%s'): %m\n", namebuf);
		gc_lmap = __openGpuCacheSharedState(MyDatabaseId,
											RelationGetRelid(rel),
											signature,
											errbuf, sizeof(errbuf));
		if (!gc_lmap || gc_lmap == MAP_FAILED)
			elog(ERROR, "%s: %s", __FUNCTION__, errbuf);
		return gc_lmap;
	}

	PG_TRY();
	{
		if (ftruncate(fdesc, mmap_sz) != 0)
			elog(ERROR, "failed on ftruncate('%s', %zu): %m\n", namebuf, mmap_sz);
		gc_sstate = mmap(NULL, mmap_sz,
						 PROT_READ | PROT_WRITE,
						 MAP_SHARED,
						 fdesc, 0);
		if (gc_sstate == MAP_FAILED)
			elog(ERROR, "failed on mmap('%s',%zu): %m", namebuf, mmap_sz);
		memset(gc_sstate, 0, offsetof(GpuCacheSharedState, kds_head));
		memcpy(gc_sstate->magic, "GpuCache", 8);
		gc_sstate->ident.database_oid = MyDatabaseId;
		gc_sstate->ident.table_oid    = RelationGetRelid(rel);
		gc_sstate->ident.signature    = signature;
		strncpy(gc_sstate->table_name, RelationGetRelationName(rel), NAMEDATALEN);
		gc_sstate->rowid_map_offset = rowid_map_offset;
		gc_sstate->redo_buffer_offset = redo_buffer_offset;
		memcpy(&gc_sstate->gc_options, gc_options, sizeof(GpuCacheOptions));
		pthreadMutexInitShared(&gc_sstate->rowid_mutex);
		pthreadMutexInitShared(&gc_sstate->redo_mutex);
		__setup_kern_data_store_column(&gc_sstate->kds_head,
									   &gc_sstate->kds_extra_sz,
									   rel,
									   gc_options->max_num_rows);
		__resetGpuCacheSharedState(gc_sstate);

		/* build GpuCacheLocalMapping */
		gc_lmap = calloc(1, sizeof(GpuCacheLocalMapping));
		if (!gc_lmap)
			elog(ERROR, "out of memory: %m");

		memcpy(&gc_lmap->ident, &gc_sstate->ident, sizeof(GpuCacheOptions));
		gc_lmap->refcnt       = 3;
		gc_lmap->gc_sstate    = gc_sstate;
		gc_lmap->mmap_sz      = mmap_sz;
		pthreadRWLockInit(&gc_lmap->gcache_rwlock);

		hslot = __gpuCacheSharedMappingHashSlot(MyDatabaseId,
												RelationGetRelid(rel),
												signature);
		pthreadMutexLock(&gcache_shared_mapping_lock);
		dlist_push_tail(hslot, &gc_lmap->chain);
		pthreadMutexUnlock(&gcache_shared_mapping_lock);
	}
	PG_CATCH();
	{
		if (gc_sstate != MAP_FAILED)
			munmap(gc_sstate, mmap_sz);
		if (fdesc >= 0)
			close(fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();

	close(fdesc);

	return gc_lmap;
}

/*
 * getGpuCacheLocalMappingIfExist
 */
static GpuCacheLocalMapping *
getGpuCacheLocalMappingIfExist(Oid database_oid,
							   Oid table_oid,
							   uint64_t signature)
{
	GpuCacheLocalMapping *gc_lmap;
	dlist_head	   *hslot;
	dlist_iter		iter;
	char			errbuf[512];

	if (signature == 0)
		return NULL;
	hslot = __gpuCacheSharedMappingHashSlot(database_oid,
											table_oid,
											signature);
	pthreadMutexLock(&gcache_shared_mapping_lock);
	dlist_foreach (iter, hslot)
	{
		gc_lmap = dlist_container(GpuCacheLocalMapping,
								  chain, iter.cur);
		if (gc_lmap->ident.database_oid == database_oid &&
			gc_lmap->ident.table_oid    == table_oid    &&
			gc_lmap->ident.signature    == signature)
		{
			gc_lmap->refcnt += 2;
			pthreadMutexUnlock(&gcache_shared_mapping_lock);
			return gc_lmap;
		}
	}
	gc_lmap = __openGpuCacheSharedState(database_oid,
										table_oid,
										signature,
										errbuf, sizeof(errbuf));

	pthreadMutexUnlock(&gcache_shared_mapping_lock);
	if (!gc_lmap)
		elog(ERROR, "%s: %s", __FUNCTION__, errbuf);
	return (gc_lmap != MAP_FAILED ? gc_lmap : NULL);
}

/*
 * getGpuCacheLocalMapping
 */
static GpuCacheLocalMapping *
getGpuCacheLocalMapping(Relation rel,
						uint64_t signature,
						const GpuCacheOptions *gc_options)
{
	GpuCacheLocalMapping *gc_lmap;

	if (signature == 0)
		return NULL;
	gc_lmap = getGpuCacheLocalMappingIfExist(MyDatabaseId,
											 RelationGetRelid(rel),
											 signature);
	if (!gc_lmap)
	{
		pthreadMutexLock(&gcache_shared_head->gcache_sstate_mutex);
		PG_TRY();
		{
			gc_lmap = __createGpuCacheSharedState(rel, signature, gc_options);
		}
		PG_CATCH();
		{
			pthreadMutexUnlock(&gcache_shared_head->gcache_sstate_mutex);
			PG_RE_THROW();
		}
		PG_END_TRY();
		pthreadMutexUnlock(&gcache_shared_head->gcache_sstate_mutex);
	}
	return gc_lmap;
}

/*
 * __removeGpuCacheLocalMapping
 */
static void
__removeGpuCacheLocalMapping(GpuCacheLocalMapping *gc_lmap)
{
	CUresult	rc;

	Assert(gc_lmap->refcnt == 0);
	dlist_delete(&gc_lmap->chain);
	munmap(gc_lmap->gc_sstate, gc_lmap->mmap_sz);
	if (gc_lmap->gcache_main_devptr != 0UL)
	{
		/* only GpuService context */
		rc = cuMemFree(gc_lmap->gcache_main_devptr);
		if (rc != CUDA_SUCCESS)
			fprintf(stderr, "failed on cuMemFree: %s\n", cuStrError(rc));
	}
	if (gc_lmap->gcache_extra_devptr != 0UL)
	{
		/* only GpuService context */
		rc = cuMemFree(gc_lmap->gcache_extra_devptr);
		if (rc != CUDA_SUCCESS)
			fprintf(stderr, "failed on cuMemFree: %s\n", cuStrError(rc));
	}

	if (gc_lmap->gcache_main_devptr != 0UL)
		fprintf(stderr, "%s: main %llu, extra %llu\n",
				__FUNCTION__,
				gc_lmap->gcache_main_devptr,
				gc_lmap->gcache_extra_devptr);
	free(gc_lmap);
}

/*
 * putGpuCacheLocalMapping
 */
static void
__putGpuCacheLocalMappingNoLock(GpuCacheLocalMapping *gc_lmap)
{
	Assert(gc_lmap->refcnt >= 2);
	gc_lmap->refcnt -= 2;
	if (gc_lmap->refcnt == 0)
		__removeGpuCacheLocalMapping(gc_lmap);
}

static void
putGpuCacheLocalMapping(GpuCacheLocalMapping *gc_lmap)
{
	pthreadMutexLock(&gcache_shared_mapping_lock);
	__putGpuCacheLocalMappingNoLock(gc_lmap);
	pthreadMutexUnlock(&gcache_shared_mapping_lock);
}

/*
 * unmapGpuCacheLocalMapping
 */
static void
unmapGpuCacheLocalMapping(Oid table_oid)
{
	pthreadMutexLock(&gcache_shared_mapping_lock);
	for (int i=0; i < GCACHE_SHARED_MAPPING_NSLOTS; i++)
	{
		dlist_head *hslot = &gcache_shared_mapping_slot[i];
		dlist_mutable_iter iter;

		dlist_foreach_modify(iter, hslot)
		{
			GpuCacheLocalMapping *gc_lmap = dlist_container(GpuCacheLocalMapping,
															 chain, iter.cur);
			if (gc_lmap->ident.database_oid == MyDatabaseId &&
				gc_lmap->ident.table_oid    == table_oid)
			{
				gc_lmap->refcnt &= 0xfffffffeU;
				if (gc_lmap->refcnt == 0)
					__removeGpuCacheLocalMapping(gc_lmap);
			}
		}
	}
	pthreadMutexUnlock(&gcache_shared_mapping_lock);
}

/*
 * getGpuCacheDescIdent
 */
const GpuCacheIdent *
getGpuCacheDescIdent(const GpuCacheDesc *gc_desc)
{
	return &gc_desc->ident;
}

/* ------------------------------------------------------------
 *
 * Routines to manage RowId
 *
 * ------------------------------------------------------------
 */
static uint32_t
__allocGpuCacheRowId(GpuCacheLocalMapping *gc_lmap, const ItemPointer ctid)
{
	GpuCacheSharedState *gc_sstate = gc_lmap->gc_sstate;
	uint32_t   *hslot = gpuCacheRowIdHashSlot(gc_sstate);
	GpuCacheRowIdItem *rowitems = gpuCacheRowIdItemArray(gc_sstate);
	uint32_t	hash, hindex;
	uint32_t	rowid;

	hash = hash_bytes((unsigned char *)ctid, sizeof(ItemPointerData));
	hindex = (hash % gc_sstate->gc_options.rowid_hash_nslots);

	pthreadMutexLock(&gc_sstate->rowid_mutex);
	rowid = gc_sstate->rowid_next_free;
	if (rowid < gc_sstate->gc_options.max_num_rows)
	{
		GpuCacheRowIdItem *ritem = &rowitems[rowid];

		ItemPointerCopy(ctid, &ritem->ctid);
		gc_sstate->rowid_next_free = ritem->next;
		ritem->next = hslot[hindex];
		hslot[hindex] = rowid;
		Assert(gc_sstate->rowid_num_free > 0);
		gc_sstate->rowid_num_free--;
	}
	else
	{
		Assert(rowid == UINT_MAX);
	}
	pthreadMutexUnlock(&gc_sstate->rowid_mutex);

	return rowid;
}

static uint32_t
__lookupGpuCacheRowId(GpuCacheLocalMapping *gc_lmap, const ItemPointer ctid)
{
	GpuCacheSharedState *gc_sstate = gc_lmap->gc_sstate;
	uint32_t   *hslot = gpuCacheRowIdHashSlot(gc_sstate);
	GpuCacheRowIdItem *rowitems = gpuCacheRowIdItemArray(gc_sstate);
	uint32_t	hash, rowid;

	hash = hash_bytes((unsigned char *)ctid, sizeof(ItemPointerData));

	pthreadMutexLock(&gc_sstate->rowid_mutex);
	rowid = hslot[hash % gc_sstate->gc_options.rowid_hash_nslots];
	while (rowid < gc_sstate->gc_options.max_num_rows)
	{
		GpuCacheRowIdItem *ritem = &rowitems[rowid];

		if (ItemPointerEquals(&ritem->ctid, ctid))
			break;
		rowid = ritem->next;
	}
	pthreadMutexUnlock(&gc_sstate->rowid_mutex);

	return rowid;
}

static bool
__removeGpuCacheRowId(GpuCacheLocalMapping *gc_lmap, const ItemPointer ctid)
{
	GpuCacheSharedState *gc_sstate = gc_lmap->gc_sstate;
	uint32_t   *hslot = gpuCacheRowIdHashSlot(gc_sstate);
	GpuCacheRowIdItem *rowitems = gpuCacheRowIdItemArray(gc_sstate);
	uint32_t	hash, rowid;
	uint32_t   *prev;
	bool		removal = false;

	hash = hash_bytes((unsigned char *)ctid, sizeof(ItemPointerData));

	pthreadMutexLock(&gc_sstate->rowid_mutex);
	prev = &hslot[hash % gc_sstate->gc_options.rowid_hash_nslots];
	for (rowid = *prev;
		 rowid < gc_sstate->gc_options.max_num_rows;
		 rowid = *prev)
	{
		GpuCacheRowIdItem *ritem = &rowitems[rowid];

		if (ItemPointerEquals(&ritem->ctid, ctid))
		{
			*prev = ritem->next;

			ritem->next = gc_sstate->rowid_next_free;
			gc_sstate->rowid_next_free = rowid;
			ItemPointerSetInvalid(&ritem->ctid);
			gc_sstate->rowid_num_free++;
			removal = true;
			break;
		}
		prev = &ritem->next;
	}
	pthreadMutexUnlock(&gc_sstate->rowid_mutex);

	return removal;
}

/* ------------------------------------------------------------
 *
 * Routines to manage GpuCacheDesc
 *
 * ------------------------------------------------------------
 */
static bool		initialLoadGpuCache(GpuCacheDesc *gc_desc, Relation rel);

/*
 * lookupGpuCacheDescNoLoad
 */
static GpuCacheDesc *
lookupGpuCacheDescNoLoad(Oid table_oid,
						 uint64_t signature,
						 TransactionId xid,
						 GpuCacheOptions *gc_options)
{
	GpuCacheDesc	hkey;
	GpuCacheDesc   *gc_desc;
	bool			found;

	if (signature == 0)
		return NULL;
	memset(&hkey, 0, sizeof(GpuCacheDesc));
    hkey.ident.database_oid = MyDatabaseId;
	hkey.ident.table_oid = table_oid;
	hkey.ident.signature = signature;
	hkey.xid = (TransactionIdIsValid(xid) ? xid : GetCurrentTransactionIdIfAny());
	Assert(TransactionIdIsValid(hkey.xid));

	gc_desc = hash_search(gcache_descriptors_htab,
                          &hkey, HASH_ENTER, &found);
	if (!found)
	{
		PG_TRY();
		{
			GpuCacheLocalMapping *gc_lmap
				= getGpuCacheLocalMappingIfExist(MyDatabaseId,
												 table_oid,
												 signature);
			Assert(!gc_lmap || GpuCacheIdentEqual(&gc_desc->ident,
												  &gc_lmap->ident));
			memcpy(&gc_desc->gc_options, gc_options, sizeof(GpuCacheOptions));
			Assert(!gc_lmap || GpuCacheOptionsEqual(&gc_desc->gc_options,
													&gc_lmap->gc_sstate->gc_options));
			gc_desc->gc_lmap = gc_lmap;		//may be NULL
			gc_desc->drop_on_rollback = false;
			gc_desc->drop_on_commit = false;
			gc_desc->nitems = 0;
			memset(&gc_desc->buf, 0, sizeof(StringInfoData));
		}
		PG_CATCH();
		{
			hash_search(gcache_descriptors_htab,
						&hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return gc_desc;
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
	GpuCacheDesc	hkey;
	GpuCacheDesc   *gc_desc;
	GpuCacheOptions	gc_options;
	uint64_t		signature;
	bool			found;

	signature = gpuCacheTableSignature(rel, &gc_options);
	if (signature == 0)
		return NULL;
	memset(&hkey, 0, sizeof(GpuCacheDesc));
    hkey.ident.database_oid = MyDatabaseId;
	hkey.ident.table_oid = RelationGetRelid(rel);
	hkey.ident.signature = signature;
	hkey.xid = GetCurrentTransactionId();
	Assert(TransactionIdIsValid(hkey.xid));

	gc_desc = hash_search(gcache_descriptors_htab,
                          &hkey, HASH_ENTER, &found);
	if (!found)
	{
		PG_TRY();
		{
			GpuCacheLocalMapping *gc_lmap
				= getGpuCacheLocalMapping(rel, signature, &gc_options);
			Assert(gc_lmap != NULL);
			Assert(GpuCacheIdentEqual(&gc_desc->ident,
									  &gc_lmap->ident));
			memcpy(&gc_desc->gc_options, &gc_options, sizeof(GpuCacheOptions));
			Assert(GpuCacheOptionsEqual(&gc_desc->gc_options,
										&gc_lmap->gc_sstate->gc_options));
			gc_desc->gc_lmap = gc_lmap;
			gc_desc->drop_on_rollback = false;
			gc_desc->drop_on_commit = false;
			gc_desc->nitems = 0;
			memset(&gc_desc->buf, 0, sizeof(StringInfoData));
		}
		PG_CATCH();
		{
			hash_search(gcache_descriptors_htab,
						&hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return gc_desc;
}

static void
releaseGpuCacheDesc(GpuCacheDesc *gc_desc, bool normal_commit)
{
	if (normal_commit
		? gc_desc->drop_on_commit
		: gc_desc->drop_on_rollback)
	{
		char	namebuf[MAXPGPATH];

		/* unload from the server */
		gpuCacheInvokeDropUnload(gc_desc, true);
		/* unlink the shared memory segment */
		GpuCacheSharedStateName(namebuf, MAXPGPATH,
								gc_desc->ident.database_oid,
								gc_desc->ident.table_oid,
								gc_desc->ident.signature);
		shm_unlink(namebuf);

		if (gc_desc->gc_lmap)
			putGpuCacheLocalMapping(gc_desc->gc_lmap);
	}
	else if (gc_desc->gc_lmap)
	{
		const char *pos = gc_desc->buf.data;

		for (uint32_t i=0; i < gc_desc->nitems; i++)
		{
			PendingCtidItem	   *pitem = (PendingCtidItem *)pos;
			GCacheTxLogXact		tx_log;

			if (pitem->tag == 'I')
			{
				if (normal_commit)
					tx_log.type = GCACHE_TX_LOG__COMMIT_INS;
				else
					tx_log.type = GCACHE_TX_LOG__ABORT_INS;
				tx_log.length = sizeof(GCacheTxLogXact);
				tx_log.rowid = pitem->rowid;
			}
			else if (pitem->tag == 'D')
			{
				if (normal_commit)
					tx_log.type = GCACHE_TX_LOG__COMMIT_DEL;
				else
					tx_log.type = GCACHE_TX_LOG__ABORT_DEL;
				tx_log.length = sizeof(GCacheTxLogXact);
				tx_log.rowid = pitem->rowid;
			}
			else
			{
				elog(WARNING, "Bug? unexpected PendingCtidItem tag '%c'",
					 pitem->tag);
				continue;
			}
			if (!__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)&tx_log))
				elog(WARNING, "Bug? unable to write out GpuCache Log");
			pos += sizeof(PendingCtidItem);
		}
		putGpuCacheLocalMapping(gc_desc->gc_lmap);
	}
	/* cleanup itself */
	if (gc_desc->buf.data)
		pfree(gc_desc->buf.data);
	hash_search(gcache_descriptors_htab,
				gc_desc, HASH_REMOVE, NULL);
}

/*
 * __initialLoadGpuCacheVisibilityCheck
 */
static bool
__initialLoadGpuCacheVisibilityCheck(HeapTuple tuple,
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
					 get_rel_name(tuple->t_tableOid),
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
 * __makeFlattenHeapTuple
 */
static HeapTuple
__makeFlattenHeapTuple(Relation rel, HeapTuple tuple)
{
	TupleDesc		tupdesc = RelationGetDescr(rel);
	HeapTupleHeader htup = tuple->t_data;
	StringInfoData	buf;
	bits8		   *nullmap = NULL;
	int				j, natts;
	uint32			hoff, diff;
	int				alignval;
	Datum			zero = 0;

	/* shortcut path, if no varlena attributes */
	if (!HeapTupleHasVarWidth(tuple))
		return tuple;

	/* HeapTupleData */
	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *)tuple, sizeof(HeapTupleData));
	diff = MAXALIGN(sizeof(HeapTupleData)) - sizeof(HeapTupleData);
	if (diff > 0)
		appendBinaryStringInfo(&buf, (char *)&zero, diff);

	/* HeapTupleHeaderData */
	appendBinaryStringInfo(&buf, (char *)htup, htup->t_hoff);
	natts = Min(HeapTupleHeaderGetNatts(htup), tupdesc->natts);
	if (HeapTupleHasNulls(tuple))
		nullmap = htup->t_bits;
	hoff = htup->t_hoff;
	Assert(buf.len == MAXALIGN(buf.len));
	for (j=0; j < natts; j++)
	{
		Form_pg_attribute	attr = TupleDescAttr(tupdesc, j);

		if (nullmap && att_isnull(j, nullmap))
			continue;

		if (attr->attbyval || attr->attlen > 0)
		{
			alignval = typealign_get_width(attr->attalign);
			hoff = TYPEALIGN(alignval, hoff);
			diff = TYPEALIGN(alignval, buf.len) - buf.len;
			if (diff > 0)
				appendBinaryStringInfo(&buf, (char *)&zero, diff);
			appendBinaryStringInfo(&buf, (char *)htup + hoff, attr->attlen);
			hoff += attr->attlen;
		}
		else if (attr->attlen == -1)
		{
			struct varlena *datum;
			void	   *addr;

			alignval = typealign_get_width(attr->attalign);
			if (!VARATT_NOT_PAD_BYTE((char *)htup + hoff))
				hoff = TYPEALIGN(alignval, hoff);
			addr = (char *)htup + hoff;
			hoff += VARSIZE_ANY(addr);

			datum = pg_detoast_datum_packed(addr);
			if (VARATT_IS_4B(datum))
			{
				diff = TYPEALIGN(alignval, buf.len) - buf.len;
				if (diff > 0)
					appendBinaryStringInfo(&buf, (char *)&zero, diff);
			}
			appendBinaryStringInfo(&buf, (char *)datum, VARSIZE_ANY(datum));
			if (datum != addr)
				pfree(datum);
		}
		else
		{
			elog(ERROR, "unexpected type length of '%s'",
                 format_type_be(attr->atttypid));
		}
	}
	/* final setup of HeapTupleData */
	tuple = (HeapTuple)buf.data;
	tuple->t_len = buf.len - MAXALIGN(sizeof(HeapTupleData));
	tuple->t_data = (HeapTupleHeader)(buf.data + MAXALIGN(sizeof(HeapTupleData)));
	tuple->t_data->t_infomask &= ~HEAP_HASEXTERNAL;

	return tuple;
}

/*
 * __gpuCacheInitLoadTrackCtid
 */
static void
__gpuCacheInitLoadTrackCtid(GpuCacheDesc *gc_desc,
							TransactionId xid,
							char tag,
							uint32_t rowid,
							ItemPointer ctid)
{
	PendingCtidItem	pitem;

	if (gc_desc->xid != xid)
	{
		gc_desc = lookupGpuCacheDescNoLoad(gc_desc->ident.table_oid,
										   gc_desc->ident.signature,
										   xid,
										   &gc_desc->gc_options);
	}
	if (!gc_desc->buf.data)
		initStringInfoCxt(CacheMemoryContext, &gc_desc->buf);
	Assert(tag == 'I' || tag == 'D');
	pitem.rowid = rowid;
	pitem.tag = tag;
	ItemPointerCopy(ctid, &pitem.ctid);
	appendBinaryStringInfo(&gc_desc->buf, (char *)&pitem,
						   sizeof(PendingCtidItem));
	gc_desc->nitems++;
}

/*
 * __initialLoadGpuCache - entrypoint of the initial loading
 */
static void
__initialLoadGpuCache(GpuCacheDesc *gc_desc, Relation rel)
{
	TableScanDesc	hscan;
	HeapTuple		scantup;
	HeapTuple		tuple;
	size_t			item_sz = 2048;
	GCacheTxLogInsert *item = alloca(item_sz);

	Assert(gc_desc->gc_lmap != NULL);

	hscan = table_beginscan(rel, SnapshotAny, 0, NULL);
	while ((scantup = heap_getnext(hscan, ForwardScanDirection)) != NULL)
	{
		TransactionId	gcache_xmin;
		TransactionId	gcache_xmax;
		uint32_t		rowid;
		size_t			sz;

		CHECK_FOR_INTERRUPTS();

		if (!__initialLoadGpuCacheVisibilityCheck(scantup,
												  &gcache_xmin,
												  &gcache_xmax))
			continue;

		tuple = __makeFlattenHeapTuple(rel, scantup);
		sz = MAXALIGN(offsetof(GCacheTxLogInsert, htup) + tuple->t_len);
		if (sz > item_sz)
		{
			item_sz = 2 * sz + 1024;
			item = alloca(item_sz);
		}

		rowid = __allocGpuCacheRowId(gc_desc->gc_lmap, &tuple->t_self);
		PG_TRY();
		{
			if (TransactionIdIsNormal(gcache_xmin))
				__gpuCacheInitLoadTrackCtid(gc_desc, gcache_xmin,
											'I', rowid, &tuple->t_self);
			if (TransactionIdIsNormal(gcache_xmax))
				__gpuCacheInitLoadTrackCtid(gc_desc, gcache_xmax,
											'D', rowid, &tuple->t_self);

			item->type = GCACHE_TX_LOG__INSERT;
			item->length = sz;
			item->rowid = rowid;
			memcpy(&item->htup, tuple->t_data, tuple->t_len);
			HeapTupleHeaderSetXmin(&item->htup, gcache_xmin);
			HeapTupleHeaderSetXmax(&item->htup, gcache_xmax);
			HeapTupleHeaderSetCmin(&item->htup, InvalidCommandId);
			if (!__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)item))
				elog(WARNING, "unable to write out GpuCache TxLogInsert");
		}
		PG_CATCH();
		{
			__removeGpuCacheRowId(gc_desc->gc_lmap, &tuple->t_self);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	table_endscan(hscan);
}

static bool
initialLoadGpuCache(GpuCacheDesc *gc_desc, Relation rel)
{
	GpuCacheSharedState *gc_sstate;

	if (!gc_desc->gc_lmap)
	{
		GpuCacheOptions gc_options;
		uint64_t	signature;

		signature = gpuCacheTableSignature(rel, &gc_options);
		Assert(gc_desc->ident.signature == signature);
		gc_desc->gc_lmap = getGpuCacheLocalMapping(rel, signature, &gc_options);
	}
	gc_sstate = gc_desc->gc_lmap->gc_sstate;
	for (;;)
	{
		uint32_t	phase = GCACHE_PHASE__IS_EMPTY;

		if (pg_atomic_compare_exchange_u32(&gc_sstate->phase,
										   &phase, UINT_MAX))
		{
			PG_TRY();
			{
				__initialLoadGpuCache(gc_desc, rel);
			}
			PG_CATCH();
			{
				phase = pg_atomic_exchange_u32(&gc_sstate->phase,
											   GCACHE_PHASE__IS_CORRUPTED);
				PG_RE_THROW();
			}
			PG_END_TRY();
			phase = pg_atomic_exchange_u32(&gc_sstate->phase,
										   GCACHE_PHASE__IS_READY);
			Assert(phase == UINT_MAX);
		}
		else if (phase == GCACHE_PHASE__IS_READY)
		{
			return true;
		}
		else if (phase == GCACHE_PHASE__IS_CORRUPTED)
		{
			return false;
		}
		else if (phase != UINT_MAX)
		{
			elog(ERROR, "Bug? GpuCache has uncertain state (phase=%u)", phase);
		}
		/* wait for completion of the initial loading by another backend */
		CHECK_FOR_INTERRUPTS();
		pg_usleep(4000L);
	}
}

/* ------------------------------------------------------------
 *
 * Routines to write out REDO Logs
 *
 * ------------------------------------------------------------
 */

/*
 * __gpuCacheInvokeBackgroundCommand
 */
static void
__gpuCacheInvokeBackgroundCommand(const GpuCacheIdent *ident,
								  int cuda_dindex,
								  bool is_async,
								  int command,
								  uint64 end_pos)
{
	GpuCacheControlCommand *cmd = NULL;
	dlist_node	   *dnode;

	Assert(cuda_dindex >= 0 && cuda_dindex < numGpuDevAttrs);
	pthreadMutexLock(&gcache_shared_head->gcache_cmd_mutex);
	while (dlist_is_empty(&gcache_shared_head->gcache_free_cmds))
	{
		pthreadMutexUnlock(&gcache_shared_head->gcache_cmd_mutex);
		CHECK_FOR_INTERRUPTS();
		pg_usleep(2000L);	/* 2ms */
		pthreadMutexLock(&gcache_shared_head->gcache_cmd_mutex);
	}
	dnode = dlist_pop_head_node(&gcache_shared_head->gcache_free_cmds);
	cmd = dlist_container(GpuCacheControlCommand, chain, dnode);
	pthreadMutexUnlock(&gcache_shared_head->gcache_cmd_mutex);

	/* enqueue command */
	memset(cmd, 0, sizeof(GpuCacheControlCommand));
	memcpy(&cmd->ident, ident, sizeof(GpuCacheIdent));
	cmd->backend = (is_async ? NULL : MyLatch);
	cmd->command = command;
	cmd->end_pos = end_pos;
	cmd->errcode = -1;

	pthreadMutexLock(&gcache_shared_head->gcache_cmd_mutex);
	dlist_push_tail(&gcache_shared_head->gpus[cuda_dindex].queue,
					&cmd->chain);
	pthreadCondSignal(&gcache_shared_head->gpus[cuda_dindex].cond);
	pthreadMutexUnlock(&gcache_shared_head->gcache_cmd_mutex);

	if (!is_async)
	{
		int		errcode;
		char	errbuf[GCACHE_CONTROL_CMD__ERRORBUF_SIZE];

		pthreadMutexLock(&gcache_shared_head->gcache_cmd_mutex);
		while (cmd->errcode < 0)
		{
			pthreadMutexUnlock(&gcache_shared_head->gcache_cmd_mutex);
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
				pthreadMutexLock(&gcache_shared_head->gcache_cmd_mutex);
				if (cmd->errcode < 0)
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
					dlist_push_tail(&gcache_shared_head->gcache_free_cmds,
									&cmd->chain);
				}
				pthreadMutexUnlock(&gcache_shared_head->gcache_cmd_mutex);
				PG_RE_THROW();
			}
			PG_END_TRY();
			pthreadMutexLock(&gcache_shared_head->gcache_cmd_mutex);
		}
		errcode = cmd->errcode;
		if (errcode)
			strcpy(errbuf, cmd->errbuf);
		dlist_push_tail(&gcache_shared_head->gcache_free_cmds,
						&cmd->chain);
		pthreadMutexUnlock(&gcache_shared_head->gcache_cmd_mutex);
		if (errcode)
			elog(ERROR, "gpucache: %s (cmd='%c')", errbuf, command);
	}
}

/*
 * GCACHE_BGWORKER_CMD__APPLY_REDO
 */
static void
gpuCacheInvokeApplyRedo(const GpuCacheDesc *gc_desc,
						uint64 sync_pos,
						bool is_async)
{
	__gpuCacheInvokeBackgroundCommand(&gc_desc->ident,
									  gc_desc->gc_options.cuda_dindex,
									  is_async,
									  GCACHE_CONTROL_CMD__APPLY_REDO,
									  sync_pos);
}

/*
 * GCACHE_BGWORKER_CMD__COMPACTION
 */
static void
gpuCacheInvokeCompaction(const GpuCacheDesc *gc_desc, bool is_async)
{
	__gpuCacheInvokeBackgroundCommand(&gc_desc->ident,
									  gc_desc->gc_options.cuda_dindex,
									  is_async,
									  GCACHE_CONTROL_CMD__COMPACTION,
									  0);
}

/*
 * GCACHE_BGWORKER_CMD__DROP_UNLOAD
 */
static void
gpuCacheInvokeDropUnload(const GpuCacheDesc *gc_desc, bool is_async)
{
	__gpuCacheInvokeBackgroundCommand(&gc_desc->ident,
									  gc_desc->gc_options.cuda_dindex,
									  is_async,
									  GCACHE_CONTROL_CMD__DROP_UNLOAD,
									  0);
}

/*
 * __gpuCacheAppendLog
 */
static bool
__gpuCacheAppendLog(GpuCacheDesc *gc_desc, GCacheTxLogCommon *tx_log)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_lmap->gc_sstate;
	char	   *redo_buffer = gpuCacheRedoLogBuffer(gc_sstate);
	size_t		buffer_sz = gc_sstate->gc_options.redo_buffer_size;
	bool		append_done = false;

	Assert(tx_log->length == MAXALIGN(tx_log->length));
	while (!append_done)
	{
		size_t		usage;
		uint32_t	phase;
		uint64_t	sync_pos;

		/*
		 * Once GPU buffer is marked to 'corrupted', any following REDO-logs
		 * make no sense, until pgstrom.gpucache_recovery() is called.
		 */
		phase = pg_atomic_read_u32(&gc_sstate->phase);
		if (phase == GCACHE_PHASE__IS_CORRUPTED)
			return false;
		Assert(phase == UINT_MAX ||		/* during initial-loading */
			   phase == GCACHE_PHASE__IS_READY);

		pthreadMutexLock(&gc_sstate->redo_mutex);
		Assert(gc_sstate->redo_write_pos >= gc_sstate->redo_read_pos &&
			   gc_sstate->redo_write_pos <= gc_sstate->redo_read_pos + buffer_sz &&
			   gc_sstate->redo_sync_pos >= gc_sstate->redo_read_pos &&
			   gc_sstate->redo_sync_pos <= gc_sstate->redo_write_pos);
		usage = gc_sstate->redo_write_pos - gc_sstate->redo_read_pos;

		/* buffer has enough space? */
		if (usage + tx_log->length <= buffer_sz)
		{
			const char *pos = (const char *)tx_log;
			size_t		remain = tx_log->length;
			size_t		offset;
			size_t		nbytes;

			while (remain > 0)
			{
				offset = gc_sstate->redo_write_pos % buffer_sz;
				if (offset + remain > buffer_sz)
					nbytes = buffer_sz - offset;
				else
					nbytes = remain;
				memcpy(redo_buffer + offset, pos, nbytes);
				gc_sstate->redo_write_pos += nbytes;
				pos += nbytes;
				remain -= nbytes;
			}
			gc_sstate->redo_write_nitems++;
			gc_sstate->redo_write_timestamp = GetCurrentTimestamp();
			append_done = true;
		}
		/*
		 * check whether the REDO log buffer usage exceeds the threshold of
		 * the synchronization.
		 */
		if (gc_sstate->redo_write_pos >= (gc_sstate->redo_sync_pos +
										  gc_sstate->gc_options.gpu_sync_threshold))
		{
			sync_pos = gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
			pthreadMutexUnlock(&gc_sstate->redo_mutex);
			gpuCacheInvokeApplyRedo(gc_desc, sync_pos, append_done);
		}
		else
		{
			pthreadMutexUnlock(&gc_sstate->redo_mutex);
		}
	}
	return append_done;
}

/*
 * __gpuCacheInsertLog
 */
static void
__gpuCacheInsertLog(HeapTuple tuple, GpuCacheDesc *gc_desc)
{
	uint32_t	rowid;
	size_t		sz;

	if (!gc_desc->gc_lmap)
		elog(ERROR, "Bug? unable to add GpuCacheLog");
	rowid = __allocGpuCacheRowId(gc_desc->gc_lmap, &tuple->t_self);
	if (rowid == UINT_MAX)
		elog(ERROR, "No rowid of GpuCache is available right now");
	PG_TRY();
	{
		GCacheTxLogInsert  *item;
		PendingCtidItem		pitem;

		/* track rowid not committed yet */
		pitem.rowid = rowid;
		pitem.tag = 'I';
		ItemPointerCopy(&tuple->t_self, &pitem.ctid);
		appendBinaryStringInfo(&gc_desc->buf, (char *)&pitem,
							   sizeof(PendingCtidItem));
		gc_desc->nitems++;

		/* INSERT Log */
		sz = MAXALIGN(offsetof(GCacheTxLogInsert, htup) + tuple->t_len);
		item = alloca(sz);
		item->type = GCACHE_TX_LOG__INSERT;
		item->length = sz;
		item->rowid = rowid;
		memcpy(&item->htup, tuple->t_data, tuple->t_len);
		HeapTupleHeaderSetXmin(&item->htup, GetCurrentTransactionId());
		HeapTupleHeaderSetXmax(&item->htup, InvalidTransactionId);
		HeapTupleHeaderSetCmin(&item->htup, InvalidCommandId);

		__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)item);
	}
	PG_CATCH();
	{
		__removeGpuCacheRowId(gc_desc->gc_lmap, &tuple->t_self);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * __gpuCacheDeleteLog
 */
static void
__gpuCacheDeleteLog(HeapTuple tuple, GpuCacheDesc *gc_desc)
{
	GCacheTxLogDelete item;
	PendingCtidItem	pitem;
	uint32_t	rowid;

	if (!gc_desc->gc_lmap)
		elog(ERROR, "Bug? unable to add GpuCacheLog");

	rowid = __lookupGpuCacheRowId(gc_desc->gc_lmap, &tuple->t_self);
	if (rowid == UINT_MAX)
		elog(ERROR, "No rowid of GpuCache is assigned to ctid(%u,%u)",
			 (uint32_t)tuple->t_self.ip_blkid.bi_hi << 16 |
			 (uint32_t)tuple->t_self.ip_blkid.bi_lo,
			 (uint32_t)tuple->t_self.ip_posid);

	/* track rowid to be released */
	pitem.rowid = rowid;
	pitem.tag = 'D';
	ItemPointerCopy(&tuple->t_self, &pitem.ctid);
	appendBinaryStringInfo(&gc_desc->buf, (char *)&pitem,
						   sizeof(PendingCtidItem));
	gc_desc->nitems++;

	/* DELETE Log */
	item.type = GCACHE_TX_LOG__DELETE;
	item.length = MAXALIGN(sizeof(GCacheTxLogDelete));
	item.xid = GetCurrentTransactionId();
	item.rowid = rowid;
	memcpy(&item.ctid, &tuple->t_self, sizeof(ItemPointerData));

	__gpuCacheAppendLog(gc_desc, (GCacheTxLogCommon *)&item);
}

/*
 * __gpuCacheTruncateLog
 */
static void
__gpuCacheTruncateLog(Oid table_oid)
{
	GpuCacheOptions	gc_options;
	GpuCacheDesc   *gc_desc;
	uint64_t		signature;

	signature = gpuCacheTableSignatureSnapshot(table_oid, NULL,
											   &gc_options);
	if (!signature)
		return;
	/* force to assign a valid transaction-id */
	(void)GetCurrentTransactionId();

	gc_desc = lookupGpuCacheDescNoLoad(table_oid, signature,
									   InvalidTransactionId,
									   &gc_options);
	if (gc_desc)
	{
		Assert(!gc_desc->drop_on_commit);
		gc_desc->drop_on_commit = true;
	}
}

/*
 * pgstrom_gpucache_sync_trigger
 */
Datum
pgstrom_gpucache_sync_trigger(PG_FUNCTION_ARGS)
{
	TriggerData	   *trigdata = (TriggerData *) fcinfo->context;
	HeapTuple		tuple;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as trigger",
			 get_func_name(fcinfo->flinfo->fn_oid));

	if (TRIGGER_FIRED_FOR_ROW(trigdata->tg_event))
	{
		GpuCacheDesc   *gc_desc;

		if (!TRIGGER_FIRED_AFTER(trigdata->tg_event))
			elog(ERROR, "%s: must be declared as AFTER ROW trigger",
				 trigdata->tg_trigger->tgname);

		gc_desc = lookupGpuCacheDesc(trigdata->tg_relation);
		if (!gc_desc)
			elog(ERROR, "gpucache is not configured for %s",
				 RelationGetRelationName(trigdata->tg_relation));
		initialLoadGpuCache(gc_desc, trigdata->tg_relation);
		if (gc_desc->buf.data == NULL)
			initStringInfoCxt(CacheMemoryContext, &gc_desc->buf);

		if (TRIGGER_FIRED_BY_INSERT(trigdata->tg_event))
		{
			tuple = __makeFlattenHeapTuple(trigdata->tg_relation,
										   trigdata->tg_trigtuple);
			__gpuCacheInsertLog(tuple, gc_desc);
			if (tuple != trigdata->tg_trigtuple)
				pfree(tuple);
		}
		else if (TRIGGER_FIRED_BY_UPDATE(trigdata->tg_event))
		{
			tuple = __makeFlattenHeapTuple(trigdata->tg_relation,
										   trigdata->tg_newtuple);
			__gpuCacheDeleteLog(trigdata->tg_trigtuple, gc_desc);
			__gpuCacheInsertLog(tuple, gc_desc);
			if (tuple != trigdata->tg_newtuple)
				pfree(tuple);
		}
		else if (TRIGGER_FIRED_BY_DELETE(trigdata->tg_event))
		{
			__gpuCacheDeleteLog(trigdata->tg_trigtuple, gc_desc);
		}
		else
		{
			elog(ERROR, "gpucache: unexpected trigger event type (%u)",
				 trigdata->tg_event);
		}
	}
	else
	{
		elog(ERROR, "gpucache: unexpected trigger event type (%u)",
			 trigdata->tg_event);
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

	rel = table_open(table_oid, RowExclusiveLock);
	gc_desc = lookupGpuCacheDesc(rel);
	if (gc_desc)
	{
		GpuCacheSharedState *gc_sstate = gc_desc->gc_lmap->gc_sstate;
		uint64_t	sync_pos;

		initialLoadGpuCache(gc_desc, rel);
		pthreadMutexLock(&gc_sstate->redo_mutex);
		sync_pos = gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
		pthreadMutexUnlock(&gc_sstate->redo_mutex);

		gpuCacheInvokeApplyRedo(gc_desc, sync_pos, false);
	}
	table_close(rel, RowExclusiveLock);

	PG_RETURN_VOID();
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

	rel = table_open(table_oid, RowExclusiveLock);
	gc_desc = lookupGpuCacheDesc(rel);
	if (gc_desc)
	{
		initialLoadGpuCache(gc_desc, rel);
		gpuCacheInvokeCompaction(gc_desc, false);
	}
	table_close(rel, RowExclusiveLock);

	PG_RETURN_VOID();
}

/*
 * pgstrom_gpucache_recovery
 */
Datum
pgstrom_gpucache_recovery(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	Relation	rel;
	GpuCacheDesc *gc_desc;

	rel = table_open(table_oid, ShareRowExclusiveLock);
    gc_desc = lookupGpuCacheDesc(rel);
	if (gc_desc)
	{
		GpuCacheSharedState *gc_sstate = gc_desc->gc_lmap->gc_sstate;
		uint32_t	phase = GCACHE_PHASE__IS_CORRUPTED;

		gpuCacheInvokeDropUnload(gc_desc, false);
		for (;;)
		{
			if (pg_atomic_compare_exchange_u32(&gc_sstate->phase,
											   &phase, UINT_MAX))
			{
				__resetGpuCacheSharedState(gc_sstate);
				break;
			}
			else if (phase == GCACHE_PHASE__IS_EMPTY ||
					 phase == GCACHE_PHASE__IS_READY)
			{
				/* nothing to do */
				break;
			}
			else if (phase == UINT_MAX)
			{
				/* someone is working now */
				pg_usleep(4000L);
				phase = GCACHE_PHASE__IS_CORRUPTED;
			}
		}
	}
	table_close(rel, ShareRowExclusiveLock);

	PG_RETURN_VOID();
}

/* ------------------------------------------------------------
 *
 * Routines to support executor
 *
 * ------------------------------------------------------------
 */
GpuCacheDesc *
pgstromGpuCacheExecInit(pgstromTaskState *pts)
{
	Relation	rel = pts->css.ss.ss_currentRelation;
	uint64_t	signature;
	GpuCacheOptions gc_options;

	if (!rel)
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
	/* check pg_strom.enable_gpucache */
	if (!pgstrom_enable_gpucache)
		return NULL;
	/* table must be configured for GpuCache */
	signature = gpuCacheTableSignature(rel, &gc_options);
	if (signature == 0UL)
	{
		elog(DEBUG2, "gpucache: table '%s' is not configured - check row/statement triggers with pgstrom.gpucache_sync_trigger()",
			 RelationGetRelationName(rel));
		return NULL;
	}
	return lookupGpuCacheDesc(rel);
}

XpuCommand *
pgstromScanChunkGpuCache(pgstromTaskState *pts,
						 struct iovec *xcmd_iov,
						 int *xcmd_iovcnt)
{
	Relation	rel = pts->css.ss.ss_currentRelation;
	GpuCacheDesc *gc_desc = pts->gcache_desc;
	XpuCommand *xcmd = NULL;

	if (!gc_desc)
		elog(ERROR, "Bug? no GpuCacheDesc is assigned");
	if (!initialLoadGpuCache(gc_desc, rel))
		elog(ERROR, "GpuCache is now corrupted (fallback should be used)");
	if (pg_atomic_fetch_add_u32(pts->gcache_fetch_count, 1) == 0)
	{
		GpuCacheSharedState *gc_sstate = gc_desc->gc_lmap->gc_sstate;
		uint64_t	write_pos;
		uint64_t	sync_pos = ULONG_MAX;

		pthreadMutexLock(&gc_sstate->redo_mutex);
		write_pos = gc_sstate->redo_write_pos;
		if (gc_sstate->redo_sync_pos < gc_sstate->redo_write_pos)
			sync_pos = gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
		pthreadMutexUnlock(&gc_sstate->redo_mutex);

		/* is the target table empty? */
		if (write_pos == 0)
		{
			pts->scan_done = true;
			return NULL;
		}
		/* force to apply pending REDO logs, if any */
		if (sync_pos != ULONG_MAX)
		{
			/*
			 * If REDO logs could not be applied correctly, GpuCache shall be
			 * moved to 'corrupted' state during execution. This execution will
			 * fail because no data-store is kept in the GPU devices.
			 * However, next execution (by retry) will use heap storage like
			 * as a fallback.
			 */
			gpuCacheInvokeApplyRedo(gc_desc, sync_pos, true);
		}
		xcmd = (XpuCommand *)pts->xcmd_buf.data;
		Assert(xcmd->length == pts->xcmd_buf.len);
		xcmd_iov->iov_base = pts->xcmd_buf.data;
		xcmd_iov->iov_len  = pts->xcmd_buf.len;
		*xcmd_iovcnt = 1;
	}
	else
	{
		pts->scan_done = true;
	}
	return xcmd;
}

void
pgstromGpuCacheExecEnd(pgstromTaskState *pts)
{
	/* nothing to do */
}

void
pgstromGpuCacheExecReset(pgstromTaskState *pts)
{
	pg_atomic_write_u32(pts->gcache_fetch_count, 0);
}

void
pgstromGpuCacheInitDSM(pgstromTaskState *pts, pgstromSharedState *ps_state)
{
	pts->gcache_fetch_count = &ps_state->__gcache_fetch_count_data;
}

void
pgstromGpuCacheAttachDSM(pgstromTaskState *pts, pgstromSharedState *ps_state)
{
	pts->gcache_fetch_count = &ps_state->__gcache_fetch_count_data;
}

void
pgstromGpuCacheShutdown(pgstromTaskState *pts)
{
	EState	   *estate = pts->css.ss.ps.state;
	uint32_t	count = pg_atomic_read_u32(pts->gcache_fetch_count);

	pts->gcache_fetch_count = MemoryContextAlloc(estate->es_query_cxt,
												 sizeof(pg_atomic_uint32));
	pg_atomic_write_u32(pts->gcache_fetch_count, count);
}

void
pgstromGpuCacheExplain(pgstromTaskState *pts,
					   ExplainState *es,
					   List *dcontext)
{
	GpuCacheDesc *gc_desc = pts->gcache_desc;
	GpuCacheLocalMapping *gc_lmap;
	GpuCacheSharedState *gc_sstate;
	GpuCacheOptions *gc_options;
	char		temp[2048];

	if (!gc_desc)
		return;
	gc_lmap = gc_desc->gc_lmap;
	gc_sstate = gc_lmap->gc_sstate;
	gc_options = &gc_desc->gc_options;

	/* config options */
	if (gc_options->cuda_dindex >= 0 &&
		gc_options->cuda_dindex < numGpuDevAttrs)
	{
		size_t	gpu_main_size;
		size_t	gpu_extra_size;
		const char *phase;

		switch (pg_atomic_read_u32(&gc_sstate->phase))
		{
			case GCACHE_PHASE__NOT_BUILT:
				phase = "not built";
				break;
			case GCACHE_PHASE__IS_EMPTY:
				phase = "empty";
				break;
			case GCACHE_PHASE__IS_READY:
				phase = "ready";
				break;
			case GCACHE_PHASE__IS_CORRUPTED:
				phase = "corrupted";
				break;
			default:
				phase = "unknown-phase";
				break;
		}

		if (!pgstrom_regression_test_mode)
		{
			gpu_main_size = pg_atomic_read_u64(&gc_sstate->gcache_main_size);
			gpu_extra_size = pg_atomic_read_u64(&gc_sstate->gcache_extra_size);

			snprintf(temp, sizeof(temp),
					 "%s [phase: %s, max_num_rows: %ld, main: %s, extra: %s]",
					 gpuDevAttrs[gc_options->cuda_dindex].DEV_NAME,
					 phase,
					 gc_options->max_num_rows,
					 format_numeric(gpu_main_size),
					 format_numeric(gpu_extra_size));
		}
		else
		{
			snprintf(temp, sizeof(temp),
                     "GPU%d [phase: %s, max_num_rows: %ld]",
					 gc_options->cuda_dindex,
					 phase,
					 gc_options->max_num_rows);
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
			gc_options->cuda_dindex < numGpuDevAttrs)
			gpu_device_id = gpuDevAttrs[gc_options->cuda_dindex].DEV_ID;

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
}

/* ------------------------------------------------------------
 *
 * Routines to support DDL callbacks
 *
 * ------------------------------------------------------------
 */

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
	elog(LOG, "__gpuCacheCallbackOnAlterTable(table_oid=%u): signature %lx -> %lx",
		 table_oid, signature_old, signature_new);

	if (signature_old != 0UL &&
		signature_old != signature_new)
	{
		gc_desc = lookupGpuCacheDescNoLoad(table_oid,
										   signature_old,
										   InvalidTransactionId,
										   &options_old);
		if (gc_desc)
			gc_desc->drop_on_commit = true;
	}

	if (signature_new != 0UL &&
		signature_new != signature_old)
	{
		gc_desc = lookupGpuCacheDescNoLoad(table_oid,
										   signature_new,
										   InvalidTransactionId,
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
		gc_desc = lookupGpuCacheDescNoLoad(table_oid,
										   signature,
										   InvalidTransactionId,
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
		Datum	signature;
		GpuCacheOptions gc_options;
		GpuCacheDesc *gc_desc;

		signature = gpuCacheTableSignatureSnapshot(table_oid, NULL,
												   &gc_options);
		if (signature == 0UL)
			continue;

		/* force to assign a valid transaction-id */
		(void)GetCurrentTransactionId();

		gc_desc = lookupGpuCacheDescNoLoad(table_oid,
										   signature,
										   InvalidTransactionId,
										   &gc_options);
		if (gc_desc)
		{
			if (gc_options.tg_sync_row == trigger_oid)
				gc_desc->drop_on_commit = true;
		}
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
	else if (access == OAT_TRUNCATE)
	{
		//elog(LOG, "pid=%u OAT_TRUNCATE (objectId=%u)", getpid(), objectId);
		__gpuCacheTruncateLog(objectId);
	}
}

static void
gpuCacheRelcacheCallback(Datum arg, Oid relid)
{
	elog(LOG, "pid=%u: gpuCacheRelcacheCallback (table_oid=%u)", getpid(), relid);
	gpuCacheTableSignatureInvalidation(relid);
	unmapGpuCacheLocalMapping(relid);
}

static void
gpuCacheSyscacheCallback(Datum arg, int cacheid, uint32 hashvalue)
{
	elog(LOG, "pid=%u: gpuCacheSyscacheCallback (cacheid=%u)", getpid(), cacheid);
	__gpucache_sync_trigger_function_oid = InvalidOid;
}

/* ------------------------------------------------------------
 *
 * Callback routines to handle transaction / resource events
 *
 * ------------------------------------------------------------
 */

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

/* ------------------------------------------------------------
 *
 * GpuCache Manager Routines
 *
 * ------------------------------------------------------------
 */

/*
 * __gpucacheAllocDeviceMemory
 *
 * NOTE: must be called under the exclusive lock of gcache_rwlock
 */
static int
__gpucacheAllocDeviceMemory(GpuCacheLocalMapping *gc_lmap,
							char *errbuf, int errbuf_sz)
{

	GpuCacheSharedState *gc_sstate = gc_lmap->gc_sstate;
	size_t		gcache_main_size = gc_sstate->kds_head.length;
	size_t		gcache_extra_size = gc_sstate->kds_extra_sz;
	CUdeviceptr	gcache_main_devptr = 0UL;
	CUdeviceptr	gcache_extra_devptr = 0UL;
	CUresult	rc;

	rc = cuMemAllocManaged(&gcache_main_devptr,
						   gcache_main_size,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(errbuf, errbuf_sz,
				 "failed on cuMemAllocManaged: %s", cuStrError(rc));
		return ENOMEM;
	}
	memcpy((void *)gcache_main_devptr,
		   &gc_sstate->kds_head,
		   KDS_HEAD_LENGTH(&gc_sstate->kds_head));

	if (gcache_extra_size > 0)
	{
		kern_data_extra	   *kds_extra;

		rc = cuMemAllocManaged(&gcache_extra_devptr,
							   gcache_extra_size,
							   CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
		{
			cuMemFree(gcache_main_devptr);
			snprintf(errbuf, errbuf_sz,
					 "failed on cuMemAllocManaged: %s", cuStrError(rc));
			return ENOMEM;
		}
		kds_extra = (kern_data_extra *)gcache_extra_devptr;
		kds_extra->length = gcache_extra_size;
		kds_extra->usage = offsetof(kern_data_extra, data);
	}
	gc_lmap->gcache_main_devptr = gcache_main_devptr;
	gc_lmap->gcache_extra_devptr = gcache_extra_devptr;
	gc_lmap->gcache_main_size = gcache_main_size;
	gc_lmap->gcache_extra_size = gcache_extra_size;
	pg_atomic_write_u64(&gc_sstate->gcache_main_size, gcache_main_size);
	pg_atomic_write_u64(&gc_sstate->gcache_extra_size, gcache_extra_size);
#if 1
	fprintf(stderr, "%s: main %p (%lu), extra %p (%lu)\n",
			__FUNCTION__,
			(void *)gc_lmap->gcache_main_devptr,
			gc_lmap->gcache_main_size,
			(void *)gc_lmap->gcache_extra_devptr,
			gc_lmap->gcache_extra_size);
#endif
	return 0;
}

/*
 * __gpucacheMarkAsCorrupted
 */
static void
__gpucacheMarkAsCorrupted(GpuCacheLocalMapping *gc_lmap)
{
	GpuCacheSharedState *gc_sstate = gc_lmap->gc_sstate;

	pg_atomic_write_u32(&gc_sstate->phase, GCACHE_PHASE__IS_CORRUPTED);
	if (gc_lmap->gcache_main_devptr != 0)
	{
		cuMemFree(gc_lmap->gcache_main_devptr);
		gc_lmap->gcache_main_devptr = 0;
		gc_lmap->gcache_main_size = 0;
	}
	if (gc_lmap->gcache_extra_devptr != 0UL)
	{
		cuMemFree(gc_lmap->gcache_extra_devptr);
		gc_lmap->gcache_extra_devptr = 0;
		gc_lmap->gcache_extra_size = 0;
	}
	/* clear the preserving flag */
	pthreadMutexLock(&gcache_shared_mapping_lock);
	gc_lmap->refcnt &= 0xfffffffeU;
	Assert(gc_lmap->refcnt >= 2);
	pthreadMutexUnlock(&gcache_shared_mapping_lock);

	pg_atomic_write_u64(&gc_sstate->gcache_main_size, 0);
	pg_atomic_write_u64(&gc_sstate->gcache_main_nitems, 0);
	pg_atomic_write_u64(&gc_sstate->gcache_extra_size, 0);
	pg_atomic_write_u64(&gc_sstate->gcache_extra_usage, 0);
	pg_atomic_write_u64(&gc_sstate->gcache_extra_dead, 0);
}

/*
 * GCACHE_CONTROL_CMD__COMPACTION
 */
static int
__gpucacheExecCompactionKernel(GpuCacheControlCommand *cmd,
							   GpuCacheLocalMapping *gc_lmap,
							   CUfunction f_gcache_compaction,
							   size_t gcache_extra_size)
{
	GpuCacheSharedState *gc_sstate = gc_lmap->gc_sstate;
	kern_data_extra *kds_extra;
	CUdeviceptr	m_kds_extra = 0UL;
	int			grid_sz, block_sz;
	unsigned int shmem_sz;
	void	   *kern_args[4];
	CUresult	rc;

	if (gcache_extra_size < gc_lmap->gcache_extra_size)
		gcache_extra_size = gc_lmap->gcache_extra_size;
retry:
	rc = cuMemAllocManaged(&m_kds_extra,
						   gcache_extra_size,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "failed on cuMemAllocManaged: %s", cuStrError(rc));
		return ENOMEM;
	}
	kds_extra = (kern_data_extra *)m_kds_extra;
	kds_extra->length = gcache_extra_size;
	kds_extra->usage = offsetof(kern_data_extra, data);

	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 &shmem_sz,
							 f_gcache_compaction,
							 0, 0);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "failed on gpuOptimalBlockSize: %s", cuStrError(rc));
		goto bailout;
	}
	kern_args[0] = &gc_lmap->gcache_main_devptr;
	kern_args[1] = &gc_lmap->gcache_extra_devptr;	/* OLD extra */
	kern_args[2] = &m_kds_extra;					/* NEW extra */
	rc = cuLaunchKernel(f_gcache_compaction,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						shmem_sz,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "failed on cuLaunchKernel: %s", cuStrError(rc));
		goto bailout;
	}
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "failed on cuStreamSynchronize: %s", cuStrError(rc));
		goto bailout;
	}
	if (kds_extra->usage > kds_extra->length)
	{
		gcache_extra_size = PAGE_ALIGN(kds_extra->usage * 5 / 4);	/* 25% margin */
		cuMemFree(m_kds_extra);
		m_kds_extra = 0UL;
		goto retry;
	}
#if 1
	fprintf(stderr, "%s: extra %p (%lu) --> %p (%lu)\n",
			__FUNCTION__,
			(void *)gc_lmap->gcache_extra_devptr,
			gc_lmap->gcache_extra_size,
			(void *)m_kds_extra,
			gcache_extra_size);
#endif
	/* swap extra buffers */
	cuMemFree(gc_lmap->gcache_extra_devptr);
	gc_lmap->gcache_extra_devptr = m_kds_extra;
	gc_lmap->gcache_extra_size   = gcache_extra_size;
	pg_atomic_write_u64(&gc_sstate->gcache_extra_size, gcache_extra_size);
	return 0;

bailout:
	if (m_kds_extra != 0)
		cuMemFree(m_kds_extra);
	return EIO;
}

static int
__gpucacheExecCompaction(GpuCacheControlCommand *cmd,
						 CUfunction f_gcache_compaction)
{
	GpuCacheLocalMapping *gc_lmap;
	int		status = 0;

	gc_lmap = getGpuCacheLocalMappingIfExist(cmd->ident.database_oid,
											 cmd->ident.table_oid,
											 cmd->ident.signature);
	if (!gc_lmap)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "shared memory segment (dat=%u,rel=%u,sig=%09lx) not found",
				 cmd->ident.database_oid,
				 cmd->ident.table_oid,
				 cmd->ident.signature);
		return EEXIST;
	}
	pthreadRWLockWriteLock(&gc_lmap->gcache_rwlock);
	if (gc_lmap->gcache_main_devptr != 0UL)
	{
		status = __gpucacheExecCompactionKernel(cmd,
												gc_lmap,
												f_gcache_compaction,
												gc_lmap->gcache_extra_size);
	}
	pthreadRWLockUnlock(&gc_lmap->gcache_rwlock);
	return status;
}

/*
 * GCACHE_CONTROL_CMD__APPLY_REDO
 */
static int
__gpucacheExecApplyRedoKernel(GpuCacheControlCommand *cmd,
							  GpuCacheLocalMapping *gc_lmap,
							  CUfunction f_gcache_apply_redo,
							  CUfunction f_gcache_compaction)
{
	GpuCacheSharedState *gc_sstate = gc_lmap->gc_sstate;
	char	   *redo_buf = gpuCacheRedoLogBuffer(gc_sstate);
	size_t		redo_bufsz = gc_sstate->gc_options.redo_buffer_size;
	uint64_t	head_pos, tail_pos;
	uint64_t	nitems;
	size_t		length;
	size_t		offset;
	char	   *pos, *end;
	int			grid_sz, block_sz;
	unsigned int shmem_sz;
	void	   *kern_args[4];
	kern_gpucache_redolog *gcache_redo;
	CUdeviceptr	m_gcache_redo = 0UL;
	CUresult	rc;
	int			status = EIO;

	pthreadMutexLock(&gc_sstate->redo_mutex);
	Assert(gc_sstate->redo_write_pos    >= gc_sstate->redo_read_pos &&
		   gc_sstate->redo_write_nitems >= gc_sstate->redo_read_nitems);
	head_pos = gc_sstate->redo_read_pos;
	tail_pos = gc_sstate->redo_write_pos;
	nitems  = (gc_sstate->redo_write_nitems - gc_sstate->redo_read_nitems);
	pthreadMutexUnlock(&gc_sstate->redo_mutex);

	/* alloc kern_gpucache_redolog */
	length = (MAXALIGN(offsetof(kern_gpucache_redolog,
								redo_items[nitems])) +
			  MAXALIGN(tail_pos - head_pos));
	rc = cuMemAllocManaged(&m_gcache_redo,
						   length,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "failed on cuMemAllocManaged: %s\n", cuStrError(rc));
		return EIO;
	}
	gcache_redo = (kern_gpucache_redolog *)m_gcache_redo;
	memset(gcache_redo, 0, offsetof(kern_gpucache_redolog, redo_items));
	gcache_redo->length = length;
	gcache_redo->nrooms = nitems;
	gcache_redo->nitems = 0;

	/* copy redo-log into device memory */
	pos = (char *)gcache_redo + MAXALIGN(offsetof(kern_gpucache_redolog,
												  redo_items[nitems]));
	length = (tail_pos - head_pos);
	offset = (head_pos % redo_bufsz);
	if (offset + length <= redo_bufsz)
	{
		memcpy(pos, redo_buf + offset, length);
	}
	else
	{
		size_t	sz = redo_bufsz - offset;

		memcpy(pos, redo_buf + offset, sz);
		memcpy(pos + sz, redo_buf, length - sz);
	}
	end = pos + length;

	/* make advance the read position */
	pthreadMutexLock(&gc_sstate->redo_mutex);
	gc_sstate->redo_read_pos    += length;
	gc_sstate->redo_read_nitems += nitems;
	Assert(gc_sstate->redo_write_pos    >= gc_sstate->redo_read_pos &&
		   gc_sstate->redo_write_nitems >= gc_sstate->redo_read_nitems);
	pthreadMutexUnlock(&gc_sstate->redo_mutex);

	/* setup kern_gpucache_redolog index */
	while (pos < end)
	{
		GCacheTxLogCommon *tx_log = (GCacheTxLogCommon *)pos;

		gcache_redo->redo_items[gcache_redo->nitems++]
			= __kds_packed((char *)pos - (char *)gcache_redo);
		pos += tx_log->length;
	}

	/* GPU kernel invocation */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 &shmem_sz,
							 f_gcache_apply_redo,
							 0, 0);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "failed on __gpuOptimalBlockSize: %s\n", cuStrError(rc));
		goto bailout;
	}
retry:
	for (uint32_t phase = 1; phase <= 6; phase++)
	{
		kern_args[0] = &m_gcache_redo;
		kern_args[1] = &gc_lmap->gcache_main_devptr;
		kern_args[2] = &gc_lmap->gcache_extra_devptr;
		kern_args[3] = &phase;

		rc = cuLaunchKernel(f_gcache_apply_redo,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							shmem_sz,
							CU_STREAM_PER_THREAD,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
		{
			snprintf(cmd->errbuf, sizeof(cmd->errbuf),
					 "failed on cuLaunchKernel: %s\n", cuStrError(rc));
			goto bailout;
		}
	}
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
    if (rc != CUDA_SUCCESS)
    {
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "failed on cuStreamSynchronize: %s\n", cuStrError(rc));
		goto bailout;
	}

	if (gcache_redo->kerror.errcode == ERRCODE_BUFFER_NO_SPACE)
	{
		kern_data_extra *extra = (kern_data_extra *)gc_lmap->gcache_extra_devptr;
		size_t		gcache_extra_size;
		int			__status;

		/* expand the extra buffer with 25% larger virtual space */
		gcache_extra_size = PAGE_ALIGN((extra->length * 5) / 4);
		__status = __gpucacheExecCompactionKernel(cmd,
												  gc_lmap,
												  f_gcache_compaction,
												  gcache_extra_size);
		if (__status == 0)
			goto retry;
		/* abort */
		status = __status;
	}
	else if (gcache_redo->kerror.errcode != ERRCODE_STROM_SUCCESS)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "gpucache: %s (%s:%d) %s (errcode=%d)\n",
				 gcache_redo->kerror.funcname,
				 gcache_redo->kerror.filename,
				 gcache_redo->kerror.lineno,
				 gcache_redo->kerror.message,
				 gcache_redo->kerror.errcode);
	}
	else
	{
		kern_data_store	   *kds = (kern_data_store *)gc_lmap->gcache_main_devptr;
		kern_data_extra	   *extra = (kern_data_extra *)gc_lmap->gcache_extra_devptr;

		pg_atomic_write_u64(&gc_sstate->gcache_main_nitems, kds->nitems);
		if (extra)
		{
			pg_atomic_write_u64(&gc_sstate->gcache_extra_usage, extra->usage);
			pg_atomic_write_u64(&gc_sstate->gcache_extra_dead, extra->deadspace);
		}
#if 1
		fprintf(stderr, "gpucache: redo log applied (nitems=%lu, %lu bytes)\n",
				nitems, length);
		fprintf(stderr, "stats: gcache_main_size=%lu, gcache_main_nitems=%lu, gcache_extra_size=%lu, gcache_extra_usage=%lu, gcache_extra_dead=%lu\n",
				pg_atomic_read_u64(&gc_sstate->gcache_main_size),
				pg_atomic_read_u64(&gc_sstate->gcache_main_nitems),
				pg_atomic_read_u64(&gc_sstate->gcache_extra_size),
				pg_atomic_read_u64(&gc_sstate->gcache_extra_usage),
				pg_atomic_read_u64(&gc_sstate->gcache_extra_dead));
#endif
		status = 0;		/* success */
	}
bailout:
	if (m_gcache_redo != 0UL)
		cuMemFree(m_gcache_redo);
	if (status)
		__gpucacheMarkAsCorrupted(gc_lmap);
	return status;
}

static int
__gpucacheExecApplyRedo(GpuCacheControlCommand *cmd,
						CUfunction f_gcache_apply_redo,
						CUfunction f_gcache_compaction)
{
	GpuCacheLocalMapping *gc_lmap;
	int		status = 0;

	gc_lmap = getGpuCacheLocalMappingIfExist(cmd->ident.database_oid,
											 cmd->ident.table_oid,
											 cmd->ident.signature);
	if (!gc_lmap)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "shared memory segment (dat=%u,rel=%u,sig=%09lx) not found",
				 cmd->ident.database_oid,
				 cmd->ident.table_oid,
				 cmd->ident.signature);
		return EEXIST;
	}

	pthreadRWLockWriteLock(&gc_lmap->gcache_rwlock);
	if (gc_lmap->gcache_main_devptr == 0UL)
	{
		status = __gpucacheAllocDeviceMemory(gc_lmap,
											 cmd->errbuf,
											 sizeof(cmd->errbuf));
		if (status)
			goto bailout;
	}
	status = __gpucacheExecApplyRedoKernel(cmd, gc_lmap,
										   f_gcache_apply_redo,
										   f_gcache_compaction);
bailout:
	if (status)
		__gpucacheMarkAsCorrupted(gc_lmap);
	pthreadRWLockUnlock(&gc_lmap->gcache_rwlock);
	putGpuCacheLocalMapping(gc_lmap);
	return status;
}

/*
 * GCACHE_CONTROL_CMD__DROP_UNLOAD
 */
static int
__gpucacheExecDropUnload(GpuCacheControlCommand *cmd)
{
	GpuCacheLocalMapping *gc_lmap;

	gc_lmap = getGpuCacheLocalMappingIfExist(cmd->ident.database_oid,
											 cmd->ident.table_oid,
											 cmd->ident.signature);
	if (!gc_lmap)
	{
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "shared memory segment (dat=%u,rel=%u,sig=%09lx) not found",
				 cmd->ident.database_oid,
				 cmd->ident.table_oid,
				 cmd->ident.signature);
		return EEXIST;
	}
	pthreadMutexLock(&gcache_shared_mapping_lock);
	gc_lmap->refcnt &= 0xfffffffeU;
	__putGpuCacheLocalMappingNoLock(gc_lmap);
	pthreadMutexUnlock(&gcache_shared_mapping_lock);
	return 0;
}

/*
 * gpucacheManagerEventLoop
 */
void
gpucacheManagerEventLoop(int cuda_dindex,
						 CUcontext cuda_context,
						 CUmodule cuda_module)
{
	pthread_mutex_t *cmd_mutex = &gcache_shared_head->gcache_cmd_mutex;
	pthread_cond_t *cmd_cond = &gcache_shared_head->gpus[cuda_dindex].cond;
	dlist_head	   *cmd_queue = &gcache_shared_head->gpus[cuda_dindex].queue;
	CUfunction		f_gcache_apply_redo;
	CUfunction		f_gcache_compaction;
	CUresult		rc;
	GpuCacheControlCommand *cmd;

	rc = cuModuleGetFunction(&f_gcache_apply_redo,
							 cuda_module,
							 "kern_gpucache_apply_redo");
	if (rc != CUDA_SUCCESS)
	{
		fprintf(stderr, "gpucache: unable to lookup gpucache_apply_redo\n");
		return;
	}
	rc = cuModuleGetFunction(&f_gcache_compaction,
							 cuda_module,
							 "kern_gpucache_compaction");
	if (rc != CUDA_SUCCESS)
	{
		fprintf(stderr, "gpucache: unable to lookup gpucache_compaction\n");
		return;
	}
	
	pthreadMutexLock(cmd_mutex);
	while (!gpuServiceGoingTerminate())
	{
		int		status;

		if (dlist_is_empty(cmd_queue))
		{
			if (!pthreadCondWaitTimeout(cmd_cond, cmd_mutex, 5000L))
			{
				/* timeout -> add some maintenance work here*/
			}
			continue;
		}
		/* fetch a command */
		cmd = dlist_container(GpuCacheControlCommand, chain,
							  dlist_pop_head_node(cmd_queue));
		dlist_delete(&cmd->chain);
		memset(&cmd->chain, 0, sizeof(dlist_node));
		pthreadMutexUnlock(cmd_mutex);

		switch (cmd->command)
		{
			case GCACHE_CONTROL_CMD__APPLY_REDO:
				status = __gpucacheExecApplyRedo(cmd,
												 f_gcache_apply_redo,
												 f_gcache_compaction);
				break;
			case GCACHE_CONTROL_CMD__COMPACTION:
				status = __gpucacheExecCompaction(cmd, f_gcache_compaction);
				break;
			case GCACHE_CONTROL_CMD__DROP_UNLOAD:
				status = __gpucacheExecDropUnload(cmd);
				break;
			default:
				status = EINVAL;
				snprintf(cmd->errbuf, sizeof(cmd->errbuf),
						 "gpucache: unknown command ('%c')",
						 cmd->command);
				break;
		}
		pthreadMutexLock(cmd_mutex);
		cmd->errcode = status;
		if (cmd->backend)
			SetLatch(cmd->backend);
		else
			dlist_push_head(&gcache_shared_head->gcache_free_cmds, &cmd->chain);
	}
	/* returns the pending commands immediately */
	while (!dlist_is_empty(cmd_queue))
	{
		cmd = dlist_container(GpuCacheControlCommand, chain,
							  dlist_pop_head_node(cmd_queue));
		dlist_delete(&cmd->chain);
		memset(&cmd->chain, 0, sizeof(dlist_node));
		snprintf(cmd->errbuf, sizeof(cmd->errbuf),
				 "GpuService is now shutting down...");
		cmd->errcode = EBUSY;
		if (cmd->backend)
			SetLatch(cmd->backend);
		else
			dlist_push_head(&gcache_shared_head->gcache_free_cmds, &cmd->chain);
	}
	pthreadMutexUnlock(cmd_mutex);
}

/*
 * gpucacheManagerWakeUp
 */
void
gpucacheManagerWakeUp(int cuda_dindex)
{
	if (cuda_dindex >= 0 && cuda_dindex < numGpuDevAttrs)
	{
		pthreadCondBroadcast(&gcache_shared_head->gpus[cuda_dindex].cond);
	}
	else
	{
		/* wakeup all */
		for (int i=0; i < cuda_dindex; i++)
			pthreadCondBroadcast(&gcache_shared_head->gpus[i].cond);
	}
}

/*
 * gpuCacheGetDeviceBuffer
 */
void *
gpuCacheGetDeviceBuffer(const GpuCacheIdent *ident,
						CUdeviceptr *p_gcache_main_devptr,
						CUdeviceptr *p_gcache_extra_devptr,
						char *errbuf, size_t errbuf_sz)
{
	GpuCacheLocalMapping *gc_lmap;
	bool		has_exclusive = false;

	gc_lmap = getGpuCacheLocalMappingIfExist(ident->database_oid,
											 ident->table_oid,
											 ident->signature);
	if (!gc_lmap)
		return NULL;
	pthreadRWLockReadLock(&gc_lmap->gcache_rwlock);
retry:
	if (gc_lmap->gcache_main_devptr == 0UL)
	{
		if (!has_exclusive)
		{
			pthreadRWLockUnlock(&gc_lmap->gcache_rwlock);
			has_exclusive = true;
			pthreadRWLockWriteLock(&gc_lmap->gcache_rwlock);
			goto retry;
		}
		if (__gpucacheAllocDeviceMemory(gc_lmap,
										errbuf, errbuf_sz) != 0)
		{
			pthreadRWLockUnlock(&gc_lmap->gcache_rwlock);
			putGpuCacheLocalMapping(gc_lmap);
			return NULL;
		}
	}
	/* ok, valid result */
	*p_gcache_main_devptr  = gc_lmap->gcache_main_devptr;
	*p_gcache_extra_devptr = gc_lmap->gcache_extra_devptr;
	return gc_lmap;
}

/*
 * gpuCachePutDeviceBuffer
 */
void
gpuCachePutDeviceBuffer(void *gc_lmap)
{
	putGpuCacheLocalMapping((GpuCacheLocalMapping *)gc_lmap);
}

/* ------------------------------------------------------------
 *
 * pg_strom.gpucache_auto_preload configuration
 *
 * ------------------------------------------------------------
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
__parseGpuCacheAutoPreload(const char *__config)
{
	char	   *config;
	char	   *token;
	int			nitems = 0;
	int			nrooms = 0;

	config = alloca(strlen(__config) + 1);
	strcpy(config, __config);
	config = __trim(config);

	/* special case - auto preloading */
	if (strcmp(config, "*") == 0)
		return;

	for (token = strtok(config, ",");
		 token != NULL;
		 token = strtok(NULL,   ","))
	{
		GpuCacheAutoPreloadEntry *entry;
		char   *database_name = __trim(token);
		char   *schema_name;
		char   *table_name;
		char   *pos;

		pos = strchr(database_name, '.');
		if (!pos)
			elog(ERROR, "pgstrom.gpucache_auto_preload syntax error [%s]",
				 pgstrom_gpucache_auto_preload);
		*pos++ = '\0';
		schema_name = __trim(pos);

		pos = strchr(schema_name, '.');
		if (!pos)
			elog(ERROR, "pgstrom.gpucache_auto_preload syntax error [%s]",
				 pgstrom_gpucache_auto_preload);
		*pos++ = '\0';
		table_name = __trim(pos);

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
		GpuCacheDesc   *gc_desc;
		RangeVar		rvar;
		Relation		rel;

		memset(&rvar, 0, sizeof(RangeVar));
		rvar.type = T_RangeVar;
		rvar.schemaname = entry->schema_name;
		rvar.relname = entry->table_name;

		rel = table_openrv(&rvar, AccessShareLock);
		gc_desc = lookupGpuCacheDesc(rel);
		initialLoadGpuCache(gc_desc, rel);
		table_close(rel, NoLock);

		elog(LOG, "gpucache: auto preload '%s.%s' (DB: %s)",
			 entry->schema_name,
			 entry->table_name,
			 entry->database_name);
	}
	CommitTransactionCommand();

	proc_exit(exit_code);
}

/* ------------------------------------------------------------
 *
 * pgstrom_gpucache_info
 *
 * ------------------------------------------------------------
 */
static List *
__pgstrom_gpucache_info(void)
{
	const char *dirname = "/dev/shm";
	DIR		   *dir;
	struct dirent *dentry;
	char		buffer[sizeof(GpuCacheSharedState)];
	List	   *results = NIL;

	dir = AllocateDir(dirname);
	while ((dentry = ReadDir(dir, dirname)) != NULL)
	{
		uint32_t	__port;
		uint32_t	__database_oid;
		uint32_t	__table_oid;
		uint64_t	__signature;
		ssize_t		nbytes = 0;
		ssize_t		off = 0;
		int			fdesc;

		if (strncmp(dentry->d_name, ".gpucache_", 10) != 0 ||
			sscanf(dentry->d_name,
				   ".gpucache_p%u_d%u_r%u.%09lx.buf",
				   &__port,
				   &__database_oid,
				   &__table_oid,
				   &__signature) != 4)
			continue;

		fdesc = openat(dirfd(dir), dentry->d_name, O_RDONLY);
		if (fdesc < 0)
		{
			elog(WARNING, "failed on open '%s/%s': %m",
				 dirname, dentry->d_name);
			continue;
		}
		while (off < sizeof(GpuCacheSharedState))
		{
			nbytes = read(fdesc, buffer+off, sizeof(GpuCacheSharedState)-off);
			if (nbytes > 0)
				off += nbytes;
			else if (errno != EINTR)
			{
				elog(WARNING, "unable to read '%s/%s': %m",
					 dirname, dentry->d_name);
				break;
			}
		}
		close(fdesc);

		if (off >= sizeof(GpuCacheSharedState))
		{
			GpuCacheSharedState *gc_sstate
				= pmemdup(buffer, sizeof(GpuCacheSharedState));
			results = lappend(results, gc_sstate);
		}
	}
	FreeDir(dir);

	return results;
}

Datum
pgstrom_gpucache_info(PG_FUNCTION_ARGS)
{
	GpuCacheSharedState *gc_sstate;
	FuncCallContext *fncxt;
	List	   *info_list;
	Datum		values[20];
	bool		isnull[20];
	HeapTuple	tuple;
	uint32_t	phase;
	char	   *str;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc	tupdesc;
		MemoryContext oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);
		tupdesc = CreateTemplateTupleDesc(20);
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
		TupleDescInitEntry(tupdesc,  6, "phase",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc,  7, "rowid_num_used",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc,  8, "rowid_num_free",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc,  9, "gpu_main_sz",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 10, "gpu_main_nitems",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 11, "gpu_extra_sz",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 12, "gpu_extra_usage",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 13, "gpu_extra_dead",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 14, "redo_write_ts",
						   TIMESTAMPTZOID, -1, 0);
		TupleDescInitEntry(tupdesc, 15, "redo_write_nitems",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 16, "redo_write_pos",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 17, "redo_read_nitems",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 18, "redo_read_pos",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 19, "redo_sync_pos",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 20, "config_options",
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
	values[0] = ObjectIdGetDatum(gc_sstate->ident.database_oid);
	str = get_database_name(gc_sstate->ident.database_oid);
	values[1] = CStringGetTextDatum(str);
	values[2] = ObjectIdGetDatum(gc_sstate->ident.table_oid);
	values[3] = CStringGetTextDatum(gc_sstate->table_name);
	values[4] = Int64GetDatum(gc_sstate->ident.signature);
	phase = pg_atomic_read_u32(&gc_sstate->phase);
	if (phase == GCACHE_PHASE__NOT_BUILT)
		str = "not_built";
	else if (phase == GCACHE_PHASE__IS_EMPTY)
		str = "is_empty";
	else if (phase == GCACHE_PHASE__IS_READY)
		str = "is_ready";
	else if (phase == GCACHE_PHASE__IS_CORRUPTED)
		str = "corrupted";
	else
		str = psprintf("unknown-%u", phase);
	values[5] = CStringGetTextDatum(str);
	values[6] = Int64GetDatum(gc_sstate->gc_options.max_num_rows -
							  gc_sstate->rowid_num_free);
	values[7] = Int64GetDatum(gc_sstate->rowid_num_free);
	values[8] = Int64GetDatum(pg_atomic_read_u64(&gc_sstate->gcache_main_size));
	values[9] = Int64GetDatum(pg_atomic_read_u64(&gc_sstate->gcache_main_nitems));
	values[10] = Int64GetDatum(pg_atomic_read_u64(&gc_sstate->gcache_extra_size));
	values[11] = Int64GetDatum(pg_atomic_read_u64(&gc_sstate->gcache_extra_usage));
	values[12] = Int64GetDatum(pg_atomic_read_u64(&gc_sstate->gcache_extra_dead));
	values[13] = TimestampGetDatum(gc_sstate->redo_write_timestamp);
	values[14] = Int64GetDatum(gc_sstate->redo_write_nitems);
	values[15] = Int64GetDatum(gc_sstate->redo_write_pos);
	values[16] = Int64GetDatum(gc_sstate->redo_read_nitems);
	values[17] = Int64GetDatum(gc_sstate->redo_read_pos);
	values[18] = Int64GetDatum(gc_sstate->redo_sync_pos);
	if (gc_sstate->gc_options.cuda_dindex >= 0 &&
		gc_sstate->gc_options.cuda_dindex < numGpuDevAttrs)
	{
		str = psprintf("gpu_device_id=%d,"
					   "max_num_rows=%ld,"
					   "redo_buffer_size=%zu,"
					   "gpu_sync_interval=%d,"
					   "gpu_sync_threshold=%zu",
					   gpuDevAttrs[gc_sstate->gc_options.cuda_dindex].DEV_ID,
					   gc_sstate->gc_options.max_num_rows,
					   gc_sstate->gc_options.redo_buffer_size,
					   gc_sstate->gc_options.gpu_sync_interval,
					   gc_sstate->gc_options.gpu_sync_threshold);
		values[19] = CStringGetTextDatum(str);
	}
	else
	{
		isnull[19] = true;
	}
	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);
	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}

/*
 * gpuCacheCleanupShmSegments
 */
static void
gpuCacheCleanupShmSegments(int code, Datum arg)
{
	const char *dirname = "/dev/shm";
	DIR		   *dir;
	struct dirent *dentry;

	dir = opendir(dirname);
	if (!dir)
		return;
	while ((dentry = readdir(dir)) != NULL)
	{
		uint32_t	__port;
		uint32_t	__database_oid;
		uint32_t	__table_oid;
		uint64_t	__signature;

		if (strncmp(dentry->d_name, ".gpucache_", 10) == 0 &&
			sscanf(dentry->d_name,
				   ".gpucache_p%u_d%u_r%u.%09lx.buf",
				   &__port,
				   &__database_oid,
				   &__table_oid,
				   &__signature) == 4)
		{
			fprintf(stderr, "Orphan shared memory segment [%s]\n", dentry->d_name );
			shm_unlink(dentry->d_name);
		}
	}
	closedir(dir);
}

/*
 * pgstrom_request_gpu_cache
 */
static void
pgstrom_request_gpu_cache(void)
{
	Size	sz;

	if (shmem_request_next)
		shmem_request_next();
	sz = offsetof(GpuCacheSharedHead, gpus[numGpuDevAttrs]);
	RequestAddinShmemSpace(MAXALIGN(sz));
}

/*
 * pgstrom_startup_gpu_cache
 */
static void
pgstrom_startup_gpu_cache(void)
{
	size_t	sz;
	bool	found;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	sz = offsetof(GpuCacheSharedHead, gpus[numGpuDevAttrs]);
	gcache_shared_head = ShmemInitStruct("GpuCache Shared Head", sz, &found);
	if (found)
		elog(ERROR, "Bug? GpuCacheSharedHead already exists");
	memset(gcache_shared_head, 0, sz);
	pthreadMutexInitShared(&gcache_shared_head->gcache_sstate_mutex);
	pthreadMutexInitShared(&gcache_shared_head->gcache_cmd_mutex);
	dlist_init(&gcache_shared_head->gcache_free_cmds);
	for (int i=0; i < lengthof(gcache_shared_head->__gcache_control_cmds); i++)
	{
		GpuCacheControlCommand *cmd = &gcache_shared_head->__gcache_control_cmds[i];

		dlist_push_tail(&gcache_shared_head->gcache_free_cmds,
						&cmd->chain);
	}
	for (int i=0; i < numGpuDevAttrs; i++)
	{
		pthreadCondInitShared(&gcache_shared_head->gpus[i].cond);
		dlist_init(&gcache_shared_head->gpus[i].queue);
	}
	gpuCacheCleanupShmSegments(0, 0);
}

/*
 * pgstrom_init_gpu_store
 */
void
pgstrom_init_gpu_cache(void)
{
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
							 &pgstrom_enable_gpucache,
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

	pthreadMutexInit(&gcache_shared_mapping_lock);
	for (int i=0; i < GCACHE_SHARED_MAPPING_NSLOTS; i++)
		dlist_init(&gcache_shared_mapping_slot[i]);

	/*
	 * Background worke to load GPU Store on startup
	 */
	if (pgstrom_gpucache_auto_preload)
	{
		BackgroundWorker worker;

		__parseGpuCacheAutoPreload(pgstrom_gpucache_auto_preload);

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
	shmem_request_next = shmem_request_hook;
	shmem_request_hook = pgstrom_request_gpu_cache;
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
	/* cleanup shmem segment on startup/exit */
	on_shmem_exit(gpuCacheCleanupShmSegments, 0);
}
