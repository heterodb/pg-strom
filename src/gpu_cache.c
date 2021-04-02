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
#include "cuda_gstore.h"

/*
 * GpuCacheBackgroundCommand
 */
#define GCACHE_BACKGROUND_CMD__INITIAL_LOAD     'I'
#define GCACHE_BACKGROUND_CMD__APPLY_REDO       'A'
#define GCACHE_BACKGROUND_CMD__COMPACTION       'C'
#define GCACHE_BACKGROUND_CMD__DROP_UNLOAD      'D'
typedef struct
{
	dlist_node  chain;
	Oid         database_oid;
	Oid         table_oid;
	Latch      *backend;        /* MyLatch of the backend, if any */
	int         command;        /* one of GSTORE_MAINTAIN_CMD__* */
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
#define GPUCACHE_DSM_ROWIDHASH_MAGIC		0xfee1dead
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
#define GPUCACHE_DSM_ROWIDMAP_MAGIC			0xdeadbeaf
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
	Oid				trigger_oid;
	int				refcnt;
	/* GPU memory store resources */
	bool			initial_load_in_progress;
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
	size_t			gpu_main_size;
	size_t			gpu_extra_size;

	/* REDO buffer properties */
	slock_t			redo_lock;
	uint64			redo_timestamp;
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
 * GpuCacheDSMMap (Per-backend DSM mapping)
 */
typedef struct
{
	Oid				database_oid;
	Oid				table_oid;
} GpuCacheDSMMapHashKey;

typedef struct
{
	Oid					database_oid;
	Oid					table_oid;
	Oid					trigger_oid;
	int					refcnt;
	GpuCacheSharedState *gc_sstate;	/* NULL, if no GpuCache is available */
	dsm_segment		   *dsm_seg;		/* dsm_segment_address() == Redo buffer */
	GpuCacheRowIdHash  *rowhash;	/* DSM */
	GpuCacheRowIdMap   *rowmap;		/* DSM */
} GpuCacheDSMMap;

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
	Oid				trigger_oid;
	TransactionId	xid;
	GpuCacheDSMMap *gc_dmap;
	/* array of PendingRowIdItem */
	uint32			nitems;
	StringInfoData	buf;
} GpuCacheHandle;

/*
 * GpuCacheSyncTrigger
 */
typedef struct
{
	Oid			table_oid;
	Oid			trigger_oid;
} GpuCacheSyncTrigger;

/* --- static variables --- */
static GpuCacheSharedHead *gcache_shared_head = NULL;
static HTAB		   *gcache_dsmmap_htab = NULL;
static HTAB		   *gcache_handle_htab = NULL;
static HTAB		   *gcache_sync_trigger_htab = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static object_access_hook_type object_access_next = NULL;
static void		  (*gpucache_xact_redo_next)(XLogReaderState *record) = NULL;
static void		  (*gpucache_heap_redo_next)(XLogReaderState *record) = NULL;

/* --- function declarations --- */
static uint32	__gpuCacheAllocateRowId(GpuCacheDSMMap *gc_dmap,
										ItemPointer ctid);
static void		__gpuCacheAppendLog(GpuCacheDSMMap *gc_dmap,
									GstoreTxLogCommon *tx_log);

static CUresult gpuCacheInvokeApplyRedo(GpuCacheSharedState *gc_sstate,
										uint64 end_pos,
										bool is_async);
static CUresult gpuCacheInvokeCompaction(GpuCacheSharedState *gc_sstate,
										 bool is_async);
void	GpuCacheStartupPreloader(Datum arg);
PG_FUNCTION_INFO_V1(pgstrom_gpucache_sync_trigger);
//PG_FUNCTION_INFO_V1(pgstrom_gpucache_precheck_event_trigger);

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
parseSyncTriggerOptions(Trigger *trig, GpuCacheOptions *gc_options)
{
	int			cuda_dindex = 0;				/* default: GPU0 */
	int			gpu_sync_interval = 8;			/* default: 8sec */
	ssize_t		gpu_sync_threshold = -1;		/* default: auto */
	int64		max_num_rows = (10UL << 20);	/* default: 10M rows */
	ssize_t		redo_buffer_size = (160UL << 20);	/* default: 160MB */
	char	   *config;
	char	   *key, *value;
	char	   *saved;

	/* extract trigger argument option string, if any */
	if (trig->tgnargs == 0)
		goto out;	/* all default */
	else if (trig->tgnargs == 1)
	{
		Const  *con = (Const *)stringToNode(trig->tgargs[0]);

		if (!IsA(con, Const))
			elog(ERROR, "trigger argument must be const value");
		if (con->constisnull)
			goto out;	/* all default */
		if (con->consttype != TEXTOID)
			elog(ERROR, "trigger argument must be text value");
		config = TextDatumGetCString(con->constvalue);
	}
	else
	{
		elog(ERROR, "too much trigger arguments");
	}

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
	pfree(config);
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

/*
 * lookup_gpucache_sync_trigger_oid
 */
static Oid
lookup_gpucache_sync_trigger_oid(void)
{
	static Oid	__gpucache_sync_trigger_oid = InvalidOid;

	if (!OidIsValid(__gpucache_sync_trigger_oid))
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

		__gpucache_sync_trigger_oid
			= GetSysCacheOid3(PROCNAMEARGSNSP,
							  Anum_pg_proc_oid,
							  CStringGetDatum("gpucache_sync_trigger"),
							  PointerGetDatum(&argtypes),
							  ObjectIdGetDatum(namespace_oid));
	}
	return __gpucache_sync_trigger_oid;
}

/*
 * relationLookupSyncTrigger
 */
static Oid
relationLookupSyncTrigger(Relation rel, GpuCacheOptions *gc_options)
{
	TriggerDesc *trigdesc = rel->trigdesc;
	Oid		function_oid;
	int		i;

	if (!trigdesc ||
		!trigdesc->trig_insert_after_row ||
		!trigdesc->trig_update_after_row ||
		!trigdesc->trig_delete_after_row)
		return InvalidOid;	/* quick bailout */

	function_oid = lookup_gpucache_sync_trigger_oid();
	for (i=0; i < trigdesc->numtriggers; i++)
	{
		Trigger	   *trig = &trigdesc->triggers[i];
		
		if (trig->tgfoid == function_oid &&
			trig->tgenabled &&
			trig->tgtype == (TRIGGER_TYPE_ROW |
							 TRIGGER_TYPE_INSERT |
							 TRIGGER_TYPE_DELETE |
							 TRIGGER_TYPE_UPDATE))
		{
			parseSyncTriggerOptions(trig, gc_options);
			return trig->tgoid;
		}
	}
	return InvalidOid;
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
		GpuCacheSyncTrigger *gc_trig;
		Relation	rel;
		bool		found;

		gc_trig = hash_search(gcache_sync_trigger_htab,
							  &rte->relid, HASH_ENTER, &found);
		if (!found)
		{
			PG_TRY();
			{
				rel = relation_open(rte->relid, NoLock);
				gc_trig->trigger_oid = relationLookupSyncTrigger(rel, NULL);
				relation_close(rel, NoLock);
			}
			PG_CATCH();
			{
				hash_search(gcache_sync_trigger_htab,
							&rte->relid, HASH_REMOVE, NULL);
				PG_RE_THROW();
			}
			PG_END_TRY();
		}
		retval = OidIsValid(gc_trig->trigger_oid);
	}
	return retval;
}

/*
 * RelationHasGpuCache
 */
bool
RelationHasGpuCache(Relation rel)
{
	bool	trigger_oid = relationLookupSyncTrigger(rel, NULL);

	return OidIsValid(trigger_oid);
}

/*
 * __gpuCacheLoadCudaModule
 */
static CUresult
__gpuCacheLoadCudaModule(CUmodule *p_cuda_module)
{
	const char *path = PGSHAREDIR "/pg_strom/cuda_gstore.fatbin";
	int			rawfd;
	struct stat	stat_buf;
	ssize_t		nbytes;
	void	   *fatbin_image;
	CUmodule	cuda_module;
	CUresult	rc = CUDA_ERROR_FILE_NOT_FOUND;

	rawfd = open(path, O_RDONLY);
	if (rawfd < 0)
		goto error_0;

	if (fstat(rawfd, &stat_buf) != 0)
		goto error_1;

	fatbin_image = malloc(stat_buf.st_size + 1);
	if (!fatbin_image)
		goto error_1;

	nbytes = __readFileSignal(rawfd, fatbin_image,
							  stat_buf.st_size, false);
	if (nbytes != stat_buf.st_size)
		goto error_2;

	rc = cuModuleLoadFatBinary(&cuda_module, fatbin_image);
	if (rc == CUDA_SUCCESS)
		*p_cuda_module = cuda_module;
error_2:
	free(fatbin_image);
error_1:
	close(rawfd);
error_0:
	return rc;
}

/*
 * __gpuCacheAllocateDSM
 */
static void
__gpuCacheAllocateDSM(GpuCacheDSMMap *gc_dmap)
{
	GpuCacheSharedState *gc_sstate = gc_dmap->gc_sstate;
	GpuCacheRowIdHash *rowhash;
	GpuCacheRowIdMap  *rowmap;
	dsm_segment *dsm_seg;
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
		elog(ERROR, "failed on dsm_create(%zu, 0)", sz);

	addr = dsm_segment_address(dsm_seg);
	/* REDO buffer */
	addr += PAGE_ALIGN(gc_sstate->redo_buffer_size);
	/* GpuCacheRowIdMap */
	rowmap = (GpuCacheRowIdMap *)addr;
	rowmap->magic = GPUCACHE_DSM_ROWIDMAP_MAGIC;
	rowmap->nrooms = nrooms;
	for (i=1; i <= nrooms; i++)
	{
		rowmap->r_items[i-1].next = (i < nrooms ? i : UINT_MAX);
	}
	addr += PAGE_ALIGN(offsetof(GpuCacheRowIdMap, r_items[nrooms]));
	/* GpuCacheRowIdHash */
	rowhash = (GpuCacheRowIdHash *)addr;
	rowhash->magic = GPUCACHE_DSM_ROWIDHASH_MAGIC;
	SpinLockInit(&rowhash->lock);
	rowhash->nslots = nslots;
	rowhash->nrooms = nrooms;
	rowhash->free_list = 0;
	memset(rowhash->hash_slot, -1, sizeof(cl_uint) * nslots);

	gc_dmap->dsm_seg = dsm_seg;
	gc_dmap->rowhash = rowhash;
	gc_dmap->rowmap  = rowmap;

	gc_sstate->dsm_handle = dsm_segment_handle(dsm_seg);
}

/*
 * __gpuCacheCreateKernelBuffer
 */
static kern_data_store *
__gpuCacheCreateKernelBuffer(Relation rel, int64 nrooms,
							 kern_data_extra *kds_extra)
{
	TupleDesc	tupdesc = RelationGetDescr(rel);
	kern_data_store *kds_head;
	kern_colmeta *cmeta;
	size_t		main_sz;
	size_t		unitsz, sz;
	int			j;

	main_sz = (KDS_calculateHeadSize(tupdesc) +
			   STROMALIGN(sizeof(kern_colmeta)));
	kds_head = palloc0(main_sz);
	init_kernel_data_store(kds_head,
						   tupdesc,
						   0,	/* to be set later */
						   KDS_FORMAT_COLUMN,
						   nrooms);
	kds_head->table_oid = RelationGetRelid(rel);
	Assert(main_sz >= offsetof(kern_data_store,
							   colmeta[kds_head->nr_colmeta]));
	memset(kds_extra, 0, offsetof(kern_data_extra, data));
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = tupleDescAttr(tupdesc, j);

		cmeta = &kds_head->colmeta[j];
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
			kds_extra->length += MAXALIGN(unitsz) * nrooms;
		}
		else
		{
			elog(ERROR, "unexpected type length (%d) at %s.%s",
				 attr->attlen,
				 RelationGetRelationName(rel),
				 NameStr(attr->attname));
		}
	}
	/* system attributes */
	cmeta = &kds_head->colmeta[kds_head->nr_colmeta - 1];
	sz = cmeta->attlen * nrooms;
	cmeta->values_offset = __kds_packed(main_sz);
	cmeta->values_length = __kds_packed(sz);
	main_sz += sz;
	/* total length */
	kds_head->length = main_sz;

	return kds_head;
}

/*
 * __gpuCacheExecInitialLoad
 */
static void
__gpuCacheExecInitialLoad(GpuCacheDSMMap *gc_dmap, Relation rel)
{
	TableScanDesc	scandesc;
	Snapshot		snapshot;
	HeapTuple		tuple;
	size_t			item_sz = 2048;
	GstoreTxLogInsert *item = palloc(item_sz);

	snapshot = RegisterSnapshot(GetLatestSnapshot());
	scandesc = table_beginscan(rel, snapshot, 0, NULL);
	while ((tuple = heap_getnext(scandesc, ForwardScanDirection)) != NULL)
	{
		size_t		sz = MAXALIGN(offsetof(GstoreTxLogInsert,
										   htup) + tuple->t_len);
		if (sz > item_sz)
		{
			item_sz = 2 * sz;
			item = repalloc(item, item_sz);
		}
		item = alloca(sz);
		item->type = GSTORE_TX_LOG__INSERT;
		item->length = sz;
		item->timestamp = GetCurrentTimestamp();
		item->rowid = __gpuCacheAllocateRowId(gc_dmap, &tuple->t_self);
		memcpy(&item->htup, tuple->t_data, tuple->t_len);
		HeapTupleHeaderSetXmin(&item->htup, GetCurrentTransactionId());
		HeapTupleHeaderSetXmax(&item->htup, InvalidTransactionId);
		HeapTupleHeaderSetCmin(&item->htup, InvalidCommandId);

		__gpuCacheAppendLog(gc_dmap, (GstoreTxLogCommon *)item);

		CHECK_FOR_INTERRUPTS();
	}
	pfree(item);
}

/*
 * init_kernel_data_store_column
 */
static void
init_kernel_data_store_column(GpuCacheSharedState *gc_sstate,
							  Relation rel, uint32 nrooms)
{
	kern_data_extra *kds_extra = &gc_sstate->kds_extra;
	kern_data_store *kds_head  = &gc_sstate->kds_head;
	TupleDesc	tupdesc = RelationGetDescr(rel);
	size_t		extra_sz = 0;
	size_t		sz, off;
	int			j, unitsz;

	init_kernel_data_store(kds_head,
						   tupdesc,
						   0,	/* to be set later */
						   KDS_FORMAT_COLUMN,
						   nrooms);
	kds_head->table_oid = RelationGetRelid(rel);
	Assert(kds_head->ncols == tupdesc->natts);
	off = KDS_calculateHeadSize(tupdesc);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute	attr = tupleDescAttr(tupdesc, j);
		kern_colmeta	   *cmeta = &kds_head->colmeta[j];

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
	kds_head->length = off;

	if (extra_sz > 0)
	{
		/* 20% margin */
		extra_sz += extra_sz / 5;
		extra_sz += offsetof(kern_data_extra, data);
	}
	kds_extra->length = extra_sz;
	kds_extra->usage = 0;
}

/*
 * PutGpuCacheSharedState
 */
static void
PutGpuCacheSharedState(GpuCacheSharedState *gc_sstate)
{
	slock_t	   *lock = &gcache_shared_head->gcache_sstate_lock;

	SpinLockAcquire(lock);
	Assert(gc_sstate->refcnt > 0);
	if (--gc_sstate->refcnt == 0)
	{
		//device memory should be already released

		

		Assert(gc_sstate->chain.prev != NULL &&
			   gc_sstate->chain.next != NULL);
		dlist_delete(&gc_sstate->chain);
		pfree(gc_sstate);
	}
	SpinLockRelease(lock);
}

/*
 * GetGpuCacheSharedState
 */
static void
GetGpuCacheSharedState(Relation rel,
					   GpuCacheDSMMap *gc_dmap,
					   GpuCacheOptions *gc_options)
{
	GpuCacheSharedState *gc_sstate = NULL;
	dsm_segment	   *dsm_seg;
	Oid				hkey[3];
	Datum			hvalue;
	int				hindex;
	dlist_head	   *slot;
	dlist_iter		iter;
	char		   *addr;
	slock_t		   *lock;

	hkey[0] = MyDatabaseId;
	hkey[1] = gc_dmap->table_oid;
	hkey[2] = gc_dmap->trigger_oid;
	hvalue = hash_any((const unsigned char *)hkey, sizeof(hkey));
	hindex = hvalue % GPUCACHE_SHARED_DESC_NSLOTS;
	slot = &gcache_shared_head->gcache_sstate_slot[hindex];
	lock = &gcache_shared_head->gcache_sstate_lock;
retry:
	CHECK_FOR_INTERRUPTS();
	SpinLockAcquire(lock);
	dlist_foreach(iter, slot)
	{
		gc_sstate = dlist_container(GpuCacheSharedState, chain, iter.cur);
		if (gc_sstate->database_oid == MyDatabaseId &&
			gc_sstate->table_oid    == gc_dmap->table_oid &&
			gc_sstate->trigger_oid  == gc_dmap->trigger_oid)
		{
			Assert(gc_sstate->refcnt > 0);
			if (gc_sstate->initial_load_in_progress)
			{
				/*
				 * It means someone already allocated GpuCacheSharedState,
				 * then initial-load is still in progress.
				 * So, we will wait for completion of the initial task.
				 */
				SpinLockRelease(lock);

				pg_usleep(10000L);	/* 10ms */
				goto retry;
			}
			/*
			 * Ok, GpuCacheSharedState is now available. Map its DSM
			 * segment on the process's virtual address space.
			 */
			dsm_seg = dsm_attach(gc_sstate->dsm_handle);
			if (!dsm_seg)
				elog(ERROR, "gpustore: could not attach DSM segment");

			addr = dsm_segment_address(dsm_seg);
			gc_dmap->dsm_seg = dsm_seg;
			addr += PAGE_ALIGN(gc_sstate->redo_buffer_size);
			gc_dmap->rowmap = (GpuCacheRowIdMap *)addr;
			Assert(gc_dmap->rowmap->magic == GPUCACHE_DSM_ROWIDMAP_MAGIC);
			addr += PAGE_ALIGN(offsetof(GpuCacheRowIdMap,
										r_items[gc_sstate->max_num_rows]));
			gc_dmap->rowhash = (GpuCacheRowIdHash *)addr;
			Assert(gc_dmap->rowhash->magic == GPUCACHE_DSM_ROWIDHASH_MAGIC);
			/* All green, GpuCacheSharedState was got */
			gc_sstate->refcnt++;
			gc_dmap->gc_sstate = gc_sstate;

			SpinLockRelease(lock);
			return;
		}
	}

	/* allocation of GpuCacheSharedState */
	gc_sstate = NULL;
	PG_TRY();
	{
		size_t	sz;

		sz = (offsetof(GpuCacheSharedState, kds_head) +
			  KDS_calculateHeadSize(RelationGetDescr(rel)));
		gc_sstate = MemoryContextAllocZero(TopSharedMemoryContext, sz);
		gc_sstate->database_oid = MyDatabaseId;
		gc_sstate->table_oid = gc_dmap->table_oid;
		gc_sstate->trigger_oid = gc_dmap->trigger_oid;
		gc_sstate->refcnt = 2;
		gc_sstate->initial_load_in_progress = true;      /* !!! blocker !!! */
		gc_sstate->max_num_rows = gc_options->max_num_rows;
		gc_sstate->cuda_dindex = gc_options->cuda_dindex;
		gc_sstate->redo_buffer_size = gc_options->redo_buffer_size;
		gc_sstate->gpu_sync_threshold = gc_options->gpu_sync_threshold;
		gc_sstate->gpu_sync_interval = gc_options->gpu_sync_interval;
		pthreadRWLockInit(&gc_sstate->gpu_buffer_lock);
		SpinLockInit(&gc_sstate->redo_lock);
		init_kernel_data_store_column(gc_sstate, rel, gc_options->max_num_rows);
	}
	PG_CATCH();
	{
		if (gc_sstate)
			pfree(gc_sstate);
		SpinLockRelease(lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	dlist_push_tail(slot, &gc_sstate->chain);
	SpinLockRelease(lock);

	/* run, initial load */
	gc_dmap->gc_sstate = gc_sstate;
	PG_TRY();
	{
		__gpuCacheAllocateDSM(gc_dmap);
		__gpuCacheExecInitialLoad(gc_dmap, rel);
		dsm_pin_mapping(gc_dmap->dsm_seg);
		dsm_pin_segment(gc_dmap->dsm_seg);
	}
	PG_CATCH();
	{
		SpinLockAcquire(lock);
		dlist_delete(&gc_sstate->chain);
		SpinLockRelease(lock);
		pfree(gc_sstate);

		PG_RE_THROW();
	}
	PG_END_TRY();
	/* allows concurrent jobs to use this GpuStore */
	SpinLockAcquire(lock);
	gc_sstate->initial_load_in_progress = false;	/* !!! unblock !!! */
	SpinLockRelease(lock);
}

/*
 * GpuStoreGetDesc
 */
static GpuCacheDSMMap *
GetGpuCacheDSMMap(Relation rel, Trigger *trig)
{
	GpuCacheDSMMap *gc_dmap;
	Oid			hkey[3];
	bool		found;

	hkey[0] = MyDatabaseId;
	hkey[1] = RelationGetRelid(rel);
	hkey[2] = trig->tgoid;
	gc_dmap = hash_search(gcache_dsmmap_htab,
						  hkey, HASH_ENTER, &found);
	if (!found)
	{
		GpuCacheOptions gc_options;

		Assert(gc_dmap->database_oid == MyDatabaseId &&
			   gc_dmap->table_oid == RelationGetRelid(rel) &&
			   gc_dmap->trigger_oid == trig->tgoid);
		gc_dmap->refcnt = 1;
		PG_TRY();
		{
			parseSyncTriggerOptions(trig, &gc_options);
			GetGpuCacheSharedState(rel, gc_dmap, &gc_options);
		}
		PG_CATCH();
		{
			hash_search(gcache_dsmmap_htab, hkey, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return (gc_dmap->gc_sstate ? gc_dmap : NULL);
}

/*
 * PutGpuCacheDSMMap
 */
static void
PutGpuCacheDSMMap(GpuCacheDSMMap *gc_dmap)
{
	Assert(gc_dmap->refcnt > 0);
	if (--gc_dmap->refcnt == 0)
	{
		if (gc_dmap->gc_sstate)
		{
			dsm_detach(gc_dmap->dsm_seg);
			PutGpuCacheSharedState(gc_dmap->gc_sstate);
		}
		else
		{
			Assert(!gc_dmap->dsm_seg &&
				   !gc_dmap->rowhash &&
				   !gc_dmap->rowmap);
		}
		/* cleanup */
		hash_search(gcache_dsmmap_htab, gc_dmap, HASH_REMOVE, NULL);
	}
}

/*
 * __gpuCacheAllocateRowId
 */
static uint32
__gpuCacheAllocateRowId(GpuCacheDSMMap *gc_dmap, ItemPointer ctid)
{
	GpuCacheRowIdHash *rowhash = gc_dmap->rowhash;
	GpuCacheRowIdMap *rowmap = gc_dmap->rowmap;
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

	ItemPointerCopy(&r_item->ctid, ctid);
	r_item->next = rowhash->hash_slot[index];
	rowhash->hash_slot[index] = rowid;

	SpinLockRelease(&rowhash->lock);

	return rowid;
}

/*
 * __gpuStoreLookupRowId / __gpuStoreReleaseRowId
 */
static uint32
__gpuStoreLookupOrReleaseRowId(GpuCacheDSMMap *gc_dmap,
							   ItemPointer ctid,
							   bool release_rowid)
{
	GpuCacheRowIdHash *rowhash = gc_dmap->rowhash;
	GpuCacheRowIdMap *rowmap = gc_dmap->rowmap;
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
__gpuStoreLookupRowId(GpuCacheDSMMap *gc_dmap, ItemPointer ctid)
{
	return __gpuStoreLookupOrReleaseRowId(gc_dmap, ctid, false);
}

static inline void
__gpuStoreReleaseRowId(GpuCacheDSMMap *gc_dmap, PendingRowIdItem *pitem)
{
	uint32	__rowid;

	__rowid = __gpuStoreLookupOrReleaseRowId(gc_dmap, &pitem->ctid, true);
	Assert(__rowid == pitem->rowid);
}

/*
 * AllocGpuCacheHandle
 */
static GpuCacheHandle *
AllocGpuCacheHandle(Relation rel, Trigger *trig)
{
	GpuCacheHandle	hkey;
	GpuCacheHandle *gc_handle;
	bool			found;

	hkey.table_oid = RelationGetRelid(rel);
	hkey.trigger_oid = trig->tgoid;
	hkey.xid = GetCurrentTransactionIdIfAny();
	if (hkey.xid == InvalidTransactionId)
		return NULL;
	gc_handle = hash_search(gcache_handle_htab,
							&hkey, HASH_ENTER, &found);
	if (!found)
	{
		PG_TRY();
		{
			gc_handle->gc_dmap = GetGpuCacheDSMMap(rel, trig);
			gc_handle->nitems = 0;
			if (gc_handle->gc_dmap)
			{
				MemoryContext	oldcxt;

				oldcxt = MemoryContextSwitchTo(CacheMemoryContext);
				initStringInfo(&gc_handle->buf);
				MemoryContextSwitchTo(oldcxt);
			}
			else
			{
				memset(&gc_handle->buf, 0, sizeof(StringInfoData));
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
	return (gc_handle->gc_dmap ? gc_handle : NULL);
}

/*
 * ReleaseGpuCacheHandle
 */
static void
ReleaseGpuCacheHandle(GpuCacheHandle *gc_handle, bool normal_commit)
{
	GpuCacheDSMMap *gc_dmap = gc_handle->gc_dmap;
	char	   *pos = gc_handle->buf.data;
	uint32		count;

	for (count=0; count < gc_handle->nitems; count++)
	{
		PendingRowIdItem *pitem = (PendingRowIdItem *)pos;

		switch (pitem->tag)
		{
			case 'I':	/* INSERT */
				if (!normal_commit)
					__gpuStoreReleaseRowId(gc_dmap, pitem);
				pos += sizeof(PendingRowIdItem);
				break;

			case 'D':	/* DELETE */
				if (normal_commit)
					__gpuStoreReleaseRowId(gc_dmap, pitem);
				pos += sizeof(PendingRowIdItem);
				break;

			default:
				elog(FATAL, "Bug? GpuCacheHandle has corruption");
		}
	}
	Assert(pos <= gc_handle->buf.data + gc_handle->buf.len);
	if (gc_handle->gc_dmap)
		PutGpuCacheDSMMap(gc_handle->gc_dmap);
	/* cleanup */
	if (gc_handle->buf.data)
		pfree(gc_handle->buf.data);
	hash_search(gcache_handle_htab, &gc_handle, HASH_REMOVE, NULL);
}

/*
 * __gpuCacheAppendLog
 */
static void
__gpuCacheAppendLog(GpuCacheDSMMap *gc_dmap, GstoreTxLogCommon *tx_log)
{
	GpuCacheSharedState *gc_sstate = gc_dmap->gc_sstate;
	char	   *redo_buffer = dsm_segment_address(gc_dmap->dsm_seg);
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
		gc_sstate->redo_timestamp = GetCurrentTimestamp();
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
 * __gpuStoreInsertLog
 */
static void
__gpuStoreInsertLog(HeapTuple tuple, GpuCacheHandle *gc_handle)
{
	GpuCacheDSMMap   *gc_dmap = gc_handle->gc_dmap;
	GstoreTxLogInsert *item;
	PendingRowIdItem rlog;
	cl_uint		rowid;
	size_t		sz;

	/* Track RowId allocation */
	enlargeStringInfo(&gc_handle->buf, sizeof(PendingRowIdItem));
	rowid = __gpuCacheAllocateRowId(gc_dmap, &tuple->t_self);
	rlog.tag = 'I';
	rlog.ctid = tuple->t_self;
	rlog.rowid = rowid;
	appendBinaryStringInfo(&gc_handle->buf,
						   (char *)&rlog,
						   sizeof(PendingRowIdItem));
	/* INSERT Log */
	sz = MAXALIGN(offsetof(GstoreTxLogInsert, htup) + tuple->t_len);
	item = alloca(sz);
	item->type = GSTORE_TX_LOG__INSERT;
	item->length = sz;
	item->timestamp = GetCurrentTimestamp();
	item->rowid = rowid;
	memcpy(&item->htup, tuple->t_data, tuple->t_len);
	HeapTupleHeaderSetXmin(&item->htup, GetCurrentTransactionId());
	HeapTupleHeaderSetXmax(&item->htup, InvalidTransactionId);
	HeapTupleHeaderSetCmin(&item->htup, InvalidCommandId);

	__gpuCacheAppendLog(gc_dmap, (GstoreTxLogCommon *)item);
}

/*
 * __gpuStoreDeleteLog
 */
static void
__gpuStoreDeleteLog(HeapTuple tuple, GpuCacheHandle *gc_handle)
{
	GpuCacheDSMMap   *gc_dmap = gc_handle->gc_dmap;
	GstoreTxLogDelete item;
	PendingRowIdItem rlog;
	cl_uint		rowid;

	/* Track RowId release */
	enlargeStringInfo(&gc_handle->buf, sizeof(PendingRowIdItem));
	rowid = __gpuStoreLookupRowId(gc_dmap, &tuple->t_self);
	rlog.tag = 'D';
	rlog.ctid = tuple->t_self;
	rlog.rowid = rowid;
	appendBinaryStringInfo(&gc_handle->buf,
						   (char *)&rlog,
						   sizeof(PendingRowIdItem));
	/* DELETE Log */
	item.type = GSTORE_TX_LOG__DELETE;
	item.length = MAXALIGN(sizeof(GstoreTxLogDelete));
	item.timestamp = GetCurrentTimestamp();
	item.rowid = rowid;
	item.xid = GetCurrentTransactionId();

	__gpuCacheAppendLog(gc_dmap, (GstoreTxLogCommon *)&item);
}

/*
 * pgstrom_gpucache_sync_trigger
 */
Datum
pgstrom_gpucache_sync_trigger(PG_FUNCTION_ARGS)
{
	TriggerData	   *trigdata = (TriggerData *) fcinfo->context;
	TriggerEvent	tg_event = trigdata->tg_event;
	Relation		rel = trigdata->tg_relation;
	GpuCacheHandle *gc_handle;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as trigger",
			 __FUNCTION__);
	if (!TRIGGER_FIRED_FOR_ROW(tg_event) ||
		!TRIGGER_FIRED_AFTER(tg_event))
		elog(ERROR, "%s: must be called as ROW-AFTER trigger",
			 __FUNCTION__);

	gc_handle = AllocGpuCacheHandle(rel, trigdata->tg_trigger);
	if (!gc_handle)
		elog(ERROR, "%s: GPU Store is not configured for %s",
			 __FUNCTION__, RelationGetRelationName(rel));

	if (TRIGGER_FIRED_BY_INSERT(tg_event))
	{
		__gpuStoreInsertLog(trigdata->tg_trigtuple, gc_handle);
	}
	else if (TRIGGER_FIRED_BY_UPDATE(trigdata->tg_event))
	{
		__gpuStoreDeleteLog(trigdata->tg_trigtuple, gc_handle);
		__gpuStoreInsertLog(trigdata->tg_newtuple, gc_handle);
	}
	else if (TRIGGER_FIRED_BY_DELETE(trigdata->tg_event))
	{
		__gpuStoreDeleteLog(trigdata->tg_trigtuple, gc_handle);
	}
	else
	{
		elog(ERROR, "%s: must be called for INSERT, DELETE or UPDATE",
			 __FUNCTION__);
	}
	PG_RETURN_POINTER(trigdata->tg_trigtuple);
}

#if 0
/*
 * pgstrom_gpucache_precheck_event_trigger
 */
Datum
pgstrom_gpucache_precheck_event_trigger(PG_FUNCTION_ARGS)
{
	EventTriggerData *trigdata;
	const char	   *command_tag;

	if (!CALLED_AS_EVENT_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as EventTrigger", __FUNCTION__);
	trigdata = (EventTriggerData *) fcinfo->context;
	if (strcmp(trigdata->event, "ddl_command_end") != 0)
		elog(ERROR, "%s: must be called on ddl_command_end event", __FUNCTION__);

	command_tag = GetCommandTagName(trigdata->tag);
	if (strcmp(command_tag, "CREATE TRIGGER") == 0)
	{
		CreateTrigStmt *stmt = (CreateTrigStmt *)trigdata->parsetree;
		Relation		rel;
		GpuCacheDSMMap *gc_dmap;

		rel = relation_openrv_extended(stmt->relation,
									   AccessShareLock, true);
		if (rel)
		{
			gc_dmap = GetGpuCacheDSMMap(rel);
			if (gc_dmap)
				PutGpuCacheDSMMap(gc_dmap);
		}
		relation_close(rel, AccessShareLock);
	}
	PG_RETURN_NULL();
}
#endif

/* ---------------------------------------------------------------- *
 *
 * Executor callbacks
 *
 * ---------------------------------------------------------------- */
GpuStoreState *
ExecInitGpuStore(ScanState *ss, int eflags, Bitmapset *outer_refs)
{
	return NULL;
}

pgstrom_data_store *
ExecScanChunkGpuStore(GpuTaskState *gts)
{
	return NULL;
}

void
ExecReScanGpuStore(GpuStoreState *gstore_state)
{}

void
ExecEndGpuStore(GpuStoreState *gstore_state)
{}

Size
ExecEstimateDSMGpuStore(GpuStoreState *gstore_state)
{
	return 0;
}

void
ExecInitDSMGpuStore(GpuStoreState *gstore_state,
					pg_atomic_uint64 *gstore_read_pos)
{}

void
ExecReInitDSMGpuStore(GpuStoreState *gstore_state)
{}

void
ExecInitWorkerGpuStore(GpuStoreState *gstore_state,
					   pg_atomic_uint64 *gstore_read_pos)
{}

void
ExecShutdownGpuStore(GpuStoreState *gstore_state)
{}

void
ExplainGpuStore(GpuStoreState *gstore_state,
				Relation frel, ExplainState *es)
{}

CUresult
gpuStoreMapDeviceMemory(GpuContext *gcontext,
						pgstrom_data_store *pds)
{
	return CUDA_ERROR_OUT_OF_MEMORY;
}

void
gpuStoreUnmapDeviceMemory(GpuContext *gcontext,
						  pgstrom_data_store *pds)
{

}





















static void
gpuCachePostDeletion(ObjectAccessType access,
					 Oid classId,
					 Oid objectId,
					 int subId,
					 void *arg)
{
	if (object_access_next)
		object_access_next(access, classId, objectId, subId, arg);



	
}

static void
gpuCacheInvalidateBuffers(Datum arg, Oid relid)
{
	/* unmap local buffers */
}


/*
 * gpuStoreAddCommitLog
 */
static void
gpuStoreAddCommitLog(GpuCacheHandle *gc_handle)
{
	GstoreTxLogCommit *c_log = alloca(GSTORE_TX_LOG_COMMIT_ALLOCSZ);
	PendingRowIdItem *pitem;
	char	   *pos = gc_handle->buf.data;
	uint32		count = 1;

	/* commit log buffer */
	c_log->type = GSTORE_TX_LOG__COMMIT;
	c_log->length = offsetof(GstoreTxLogCommit, data);
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
				if (c_log->length + 5 > GSTORE_TX_LOG_COMMIT_ALLOCSZ)
				{
					flush_commit_log = true;
					break;
				}
				temp = c_log->data + c_log->length;
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
			__gpuCacheAppendLog(gc_handle->gc_dmap,
								(GstoreTxLogCommon *)c_log);
			/* rewind */
			c_log->length = offsetof(GstoreTxLogCommit, data);
			c_log->nitems = 0;
		}
	}
}

/*
 * gpuCacheXactCallback
 */
static void
gpuCacheXactCallback(XactEvent event, void *arg)
{
#if 1
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
					gpuStoreAddCommitLog(gc_handle);
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
#if 1
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
 * GCACHE_BACKGROUND_CMD__APPLY_REDO
 */
static CUresult
gpuCacheInvokeApplyRedo(GpuCacheSharedState *gc_sstate,
						uint64 end_pos,
						bool is_async)
{
	return __gpuCacheInvokeBackgroundCommand(gc_sstate->database_oid,
											 gc_sstate->table_oid,
											 gc_sstate->cuda_dindex,
											 is_async,
											 GCACHE_BACKGROUND_CMD__APPLY_REDO,
											 end_pos);
}

/*
 * GCACHE_BACKGROUND_CMD__COMPACTION
 */
static CUresult
gpuCacheInvokeCompaction(GpuCacheSharedState *gc_sstate, bool is_async)
{
	return __gpuCacheInvokeBackgroundCommand(gc_sstate->database_oid,
											 gc_sstate->table_oid,
											 gc_sstate->cuda_dindex,
											 is_async,
											 GCACHE_BACKGROUND_CMD__COMPACTION,
											 0);
}


/*
 * gstore_xact_redo_hook
 */
static void
gstore_xact_redo_hook(XLogReaderState *record)
{
	gpucache_xact_redo_next(record);
	if (InRecovery)
	{
		//add transaction logs

	}
}

/*
 * gstore_heap_redo_hook
 */
static void
gstore_heap_redo_hook(XLogReaderState *record)
{
	gpucache_heap_redo_next(record);
	if (InRecovery)
	{
		//add redo logs
		
	}
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
	slock_t	   *cmd_lock = &gcache_shared_head->bgworker_cmd_lock;
	dlist_head *free_cmds = &gcache_shared_head->bgworker_free_cmds;
	dlist_head *cmd_queue = &gcache_shared_head->bgworkers[cuda_dindex].cmd_queue;
	dlist_node *dnode;

	SpinLockAcquire(cmd_lock);
	if (dlist_is_empty(cmd_queue))
	{
		SpinLockRelease(cmd_lock);
		return true;	/* GpuStore allows bgworker to sleep */
	}
	dnode = dlist_pop_head_node(cmd_queue);
	cmd = dlist_container(GpuCacheBackgroundCommand, chain, dnode);
    memset(&cmd->chain, 0, sizeof(dlist_node));
    SpinLockRelease(cmd_lock);

	cmd->retval = EINVAL;

	SpinLockAcquire(cmd_lock);
	if (cmd->backend)
	{
		/*
		 * A backend process who kicked GpuStore maintainer is waiting
		 * for the response. It shall check the retval, and return the
		 * GpuCacheBackgroundCommand to free list again.
		 */
		SetLatch(cmd->backend);
	}
	else
	{
		/*
		 * GpuStore maintainer was kicked asynchronously, so nobody is
		 * waiting for the response, thus, GpuCacheBackgroundCommand
		 * must be backed to the free list again.
		 */
		dlist_push_head(free_cmds, &cmd->chain);
	}
	SpinLockRelease(cmd_lock);
   	return false;
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
	bool		retval = false;

	//1. fetch command
	//2. map DSM of the GpuStore (if not yet)
	//3. run command
	//3-1. Apply Redo
	//3-2. Unmap DSM
	
#if 0
	for (hindex = 0; hindex < GPUCACHE_SHARED_DESC_NSLOTS; hindex++)
	{
		slock_t    *lock = &gcache_shared_head->gcache_sstate_lock[hindex];
		dlist_head *slot = &gcache_shared_head->gcache_sstate_slot[hindex];
		dlist_iter	iter;

		SpinLockAcquire(lock);
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
			if (gc_sstate->redo_write_nitems > gc_sstate->redo_read_nitems &&
				timestamp > (gc_sstate->gpu_sync_interval * 1000000L +
							 gc_sstate->redo_timestamp))
			{
				SpinLockAcquire(cmd_lock);
				if (!dlist_is_empty(free_cmds))
				{
					GpuCacheBackgroundCommand *cmd;

					cmd = dlist_container(GpuCacheBackgroundCommand, chain,
                                          dlist_pop_head_node(free_cmds));
					memset(cmd, 0, sizeof(GpuCacheBackgroundCommand));
					cmd->database_oid = gc_sstate->database_oid;
                    cmd->table_oid    = gc_sstate->table_oid;
                    cmd->backend      = NULL;
                    cmd->command      = GCACHE_BACKGROUND_CMD__APPLY_REDO;
                    cmd->end_pos      = gc_sstate->redo_write_pos;
                    cmd->retval       = (CUresult) UINT_MAX;

					dlist_push_tail(cmd_queue, &cmd->chain);

					gc_sstate->redo_sync_pos = gc_sstate->redo_write_pos;
					gc_sstate->redo_timestamp = timestamp;
				}
				SpinLockRelease(cmd_lock);

				retval = true;
			}
			SpinLockRelease(&gc_sstate->redo_lock);
		}
		SpinLockRelease(lock);
	}
#endif
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
	gcache_shared_head = ShmemInitStruct("GpuStore Shared Head", sz, &found);
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
	hctl.keysize = 3 * sizeof(Oid);
	hctl.entrysize = sizeof(GpuCacheDSMMap);
	hctl.hcxt = CacheMemoryContext;
	gcache_dsmmap_htab = hash_create("GpuCache DSM Mappings", 48, &hctl,
									 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = offsetof(GpuCacheHandle, xid) + sizeof(TransactionId);
	hctl.entrysize = sizeof(GpuCacheHandle);
	hctl.hcxt = CacheMemoryContext;
	gcache_handle_htab = hash_create("GpuCache Handles", 48, &hctl,
									 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(Oid);
	hctl.entrysize = sizeof(GpuCacheSyncTrigger);
	hctl.hcxt = CacheMemoryContext;
	gcache_sync_trigger_htab = hash_create("GpuCache Sync Triggers", 256, &hctl,
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
				*((void **)&RmgrTable[i].rm_redo) = gstore_xact_redo_hook;
			}
			else if (strcmp(RmgrTable[i].rm_name, "Heap") == 0)
			{
				gpucache_heap_redo_next = RmgrTable[i].rm_redo;
				*((void **)&RmgrTable[i].rm_redo) = gstore_heap_redo_hook;
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

	/* callback when trigger is dropped */
	object_access_next = object_access_hook;
	object_access_hook = gpuCachePostDeletion;
	/* callbacks to unmap shared buffers */
	CacheRegisterRelcacheCallback(gpuCacheInvalidateBuffers, 0);
	/* transaction callbacks */
	RegisterXactCallback(gpuCacheXactCallback, NULL);
	RegisterSubXactCallback(gpuCacheSubXactCallback, NULL);
}
