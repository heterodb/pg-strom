/*
 * gpu_cache.c
 *
 * GPU data cache that syncronizes a PostgreSQL table
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/*
 * GpuCacheSharedState
 */
typedef struct
{
	dlist_node	chain;
	Oid			database_oid;
	Oid			table_oid;
	uint32_t	table_sig;		/* signature */
	gpumask_t	gpumask;
	size_t		gpucache_sz;
	/* request flags */
	gpumask_t	req_compaction;
	gpumask_t	req_recovery;
	/* current status */
#define GPUCACHE_PHASE__NOT_BUILT		0	/* not built yet */
#define GPUCACHE_PHASE__NOW_LOADING		1	/* now initial loading */
#define GPUCACHE_PHASE__IS_READY		2	/* now ready */
#define GPUCACHE_PHASE__IS_CORRUPTED	3	/* corrupted */
    pg_atomic_uint32 phase;
	/* statistics */
	struct {
		pg_atomic_uint32 nitems;
		pg_atomic_uint32 dead_nitems;
		pg_atomic_uint64 usage;
		pg_atomic_uint64 dead_space;
		pg_atomic_uint64 virtual_mem_sz;
		pg_atomic_uint64 physical_mem_sz;
	} gpus[1];
} GpuCacheSharedState;

/*
 * GpuCacheSharedHead
 */
#define GPUCACHE_STATE_HASH_NSLOTS		509
typedef struct
{
	pg_atomic_uint32 maintenance;
	pthread_mutex_t hash_mutex;
	gpumask_t	req_apply_redo;
	dlist_head	free_list;
	dlist_head	hash_slots[GPUCACHE_STATE_HASH_NSLOTS];
	struct {
		pthread_mutex_t	pipe_mutex;
	} gpus[1];
} GpuCacheSharedHead;

/*
 * GpuCacheRelSignature
 */
typedef struct
{
	Oid			table_oid;
	uint32_t	table_sig;
	gpumask_t	gpumask;		/* parameter */
	size_t		gpucache_sz;	/* parameter */
} GpuCacheRelSignature;

/*
 * GpuCacheRelDesc
 */
struct GpuCacheDesc
{
	Oid			table_oid;
	uint32_t	table_sig;
	TransactionId xid;
	gpumask_t	gpumask;
	size_t		gpucache_sz;
	GpuCacheSharedState *gc_sstate;
	bool		drop_on_rollback;
	bool		drop_on_commit;
	uint32_t	nitems;
	StringInfoData buf;	/* array of PendingCtidItem */
	int			nr_gpus;
	int			pindex[1];
};

typedef struct
{
	char		tag;
	uint8_t		pindex;
	ItemPointerData ctid;
} PendingCtidItem;

/* static variables */
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static object_access_hook_type object_access_next = NULL;
static GpuCacheSharedHead *gpucache_shared_head = NULL;
static HTAB	   *gpucache_table_sig_htab = NULL;
static HTAB	   *gpucache_table_desc_htab = NULL;
static bool		pgstrom_enable_gpucache;			/* GUC */
static int		pgstrom_gpucache_max_relation_entries;	/* GUC */
static size_t	pgstrom_gpucache_log_buffer_sz;		/* GUC */
static int		pgstrom_gpucache_sync_interval;		/* GUC */
static int		pgstrom_gpucache_sync_threshold;	/* GUC */
static int	   *gpucache_pipe_read_fdesc = NULL;	/* per-GPU */
static int	   *gpucache_pipe_write_fdesc = NULL;	/* per-GPU */

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

static inline bool
is_gpucache_sync_trigger(int16 trig_type,
						 char trig_enabled,
						 Oid trig_func_oid,
						 int16 trig_nargs)
{
	if (trig_type == (TRIGGER_TYPE_ROW |
					  TRIGGER_TYPE_AFTER |
					  TRIGGER_TYPE_INSERT |
					  TRIGGER_TYPE_DELETE |
					  TRIGGER_TYPE_UPDATE) &&
		(trig_enabled == TRIGGER_FIRES_ON_ORIGIN ||
		 trig_enabled == TRIGGER_FIRES_ALWAYS) &&
		trig_func_oid == gpucache_sync_trigger_function_oid() &&
		(trig_nargs == 0 || trig_nargs == 1))
	{
		return true;
	}
	return false;
}

/*
 * parseSyncTriggerOptions
 */
static bool
__parseSyncTriggerOptions(int elevel,
						  const char *trigger_name,
						  const char *trigger_config,
						  gpumask_t *p_gpumask,
						  size_t *p_gpucache_sz)
{
	gpumask_t	gpumask = GetSystemAvailableGpus();	/* default: any GPUs */
	size_t		gpucache_sz = (512UL << 20);		/* default: 512MB */

	if (trigger_config)
	{
		char   *config = strdupa(trigger_config);
		char   *key, *val, *saved;

		for (key = strtok_r(config, ",", &saved);
			 key != NULL;
			 key = strtok_r(NULL, ",", &saved))
		{
			val = strchr(key, '=');
			if (!val)
			{
				elog(WARNING, "gpucache: options syntax error [%s]", key);
				return false;
			}
			*val++ = '\0';
			key = __trim(key);
			val = __trim(val);

			if (strcmp(key, "gpumask") == 0)
			{
				char   *end;

				gpumask = strtol(val, &end, 10);
				if (*end != '\0')
				{
					elog(elevel, "gpucache: invalid gpumask [%s]", val);
					return false;
				}
				if ((gpumask & ~GetSystemAvailableGpus()) != 0)
				{
					elog(elevel, "gpucache: gpumask=%08lx out of range", gpumask);
					return false;
				}
				if (gpumask == 0)
				{
					elog(elevel, "gpucache: gpumask=0 no GPUs are configured");
					return false;
				}
			}
			else if (strcmp(key, "cache_size") == 0)
			{
				char   *end;

				gpucache_sz = strtol(val, &end, 10);
				if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0)
					gpucache_sz <<= 30;
				else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0)
					gpucache_sz <<= 20;
				else if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0)
					gpucache_sz <<= 10;
				else if (*end != '\0')
				{
					elog(elevel, "gpucache: invalid cache_size [%s]", val);
					return false;
				}
				if (gpucache_sz < (64UL<<20))
				{
					elog(elevel, "gpucache: too small cache_size [%s], at least 64MB", val);
					return false;
				}
			}
			else
			{
				elog(elevel, "gpucache: unknown option [%s]=[%s]", key, val);
				return false;
			}
		}
	}
	if (p_gpumask)
		*p_gpumask = gpumask;
	if (p_gpucache_sz)
		*p_gpucache_sz = gpucache_sz;
	return true;
}

/*
 * gpuCacheSendTxLog
 */
static void
__gpuCacheSendTxLogOne(GpuCacheDesc *gc_desc, int pindex,
					   const GpuCacheLogCommon *gc_log)
{
	pthread_mutex_t *mutex = &gpucache_shared_head->gpus[pindex].pipe_mutex;
	int			fdesc = gpucache_pipe_write_fdesc[pindex];
	const char *buf = (const char *)gc_log;
	size_t		len = gc_log->length;
	size_t		off = 0;

	pthreadMutexLock(mutex);
	while (off < len)
	{
		ssize_t	nbytes;

		nbytes = write(fdesc, buf+off, len-off);
		if (nbytes > 0)
			off += nbytes;
		else if (nbytes == 0 || errno != EINTR)
		{
			pthreadMutexUnlock(mutex);
			elog(NOTICE, "GPU-Cache for '%s' is marked as corrupted",
				 get_rel_name(gc_desc->table_oid));
			return;
		}
	}
	pthreadMutexUnlock(mutex);
}

static void
gpuCacheSendTxLog(GpuCacheDesc *gc_desc, int pindex,
				  const GpuCacheLogCommon *gc_log)
{
	if (pindex < 0)
	{
		/* broadcast */
		for (int i=0; i < gc_desc->nr_gpus; i++)
			__gpuCacheSendTxLogOne(gc_desc, gc_desc->pindex[i], gc_log);
	}
	else
	{
		assert(pindex < gc_desc->nr_gpus);
		__gpuCacheSendTxLogOne(gc_desc, pindex, gc_log);
	}
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
	Oid			reltablespace;
	Oid			relfilenode;	/* if 0, cannot have GPU-Cache */
	gpumask_t	gpumask;        /* parameter */
	size_t		gpucache_sz;    /* parameter */
	int			relnattrs;
	struct {
		Oid		atttypid;
		int32	atttypmod;
		int16	attlen;
		bool	attbyval;
		char	attalign;
		bool	attnotnull;
		bool	attisdropped;
	} attrs[1];
} GpuCacheTableSignatureBuffer;

static uint32_t
gpuCacheTableSignatureCommon(int elevel,
							 Form_pg_class pg_class,
							 TupleDesc pg_tupdesc,
							 Oid trigger_oid,
							 const char *trigger_name,
							 const char *trigger_config,
							 gpumask_t *p_gpumask,
							 size_t *p_gpucache_sz)
{
	GpuCacheTableSignatureBuffer *sig;
	size_t		len = offsetof(GpuCacheTableSignatureBuffer,
							   attrs[pg_class->relnatts]);

	sig = (GpuCacheTableSignatureBuffer *)alloca(len);
	memset(sig, 0, len);
	sig->reltablespace	= pg_class->reltablespace;
	sig->relfilenode	= pg_class->relfilenode;
	sig->relnattrs		= pg_class->relnatts;
	for (int j=0; j < pg_class->relnatts; j++)
	{
		Form_pg_attribute att = TupleDescAttr(pg_tupdesc, j);

		sig->attrs[j].atttypid   = att->atttypid;
		sig->attrs[j].atttypmod  = att->atttypmod;
		sig->attrs[j].attlen     = att->attlen;
		sig->attrs[j].attbyval   = att->attbyval;
		sig->attrs[j].attalign   = att->attalign;
		sig->attrs[j].attnotnull = att->attnotnull;
		sig->attrs[j].attisdropped = att->attisdropped;
	}
	if (__parseSyncTriggerOptions(elevel,
								  trigger_name,
								  trigger_config,
								  &sig->gpumask,
								  &sig->gpucache_sz))
	{
		if (p_gpumask)
			*p_gpumask = sig->gpumask;
		if (p_gpucache_sz)
			*p_gpucache_sz = sig->gpucache_sz;
		return (hash_bytes((unsigned char *)sig, len) | 0x80000000U);
	}
	return 0U;
}

static uint32_t
gpuCacheTableSignature(Relation relation,
					   gpumask_t *p_gpumask,
					   size_t *p_gpucache_sz)
{
	GpuCacheRelSignature *entry;
	Oid			table_oid = RelationGetRelid(relation);
	bool		found;

	if (!gpucache_table_sig_htab)
	{
		HASHCTL	hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(GpuCacheRelSignature);
		hctl.hcxt = CacheMemoryContext;
		gpucache_table_sig_htab = hash_create("GPU-Cache Signature",
											  128,
											  &hctl,
											  HASH_ELEM |
											  HASH_BLOBS |
											  HASH_CONTEXT);
	}
	entry = hash_search(gpucache_table_sig_htab,
						&table_oid,
						HASH_ENTER,
						&found);
	if (!found)
	{
		Form_pg_class rel_form = RelationGetForm(relation);
		TupleDesc	tupdesc = RelationGetDescr(relation);

		memset((char *)entry + sizeof(Oid), 0, sizeof(GpuCacheRelSignature) - sizeof(Oid));

		if (rel_form->relkind == RELKIND_RELATION &&
			rel_form->relfilenode != 0 &&
			relation->trigdesc)
		{
			TriggerDesc *trigdesc = relation->trigdesc;
			Oid			trigger_oid = InvalidOid;
			const char *trigger_name = NULL;
			const char *trigger_config = NULL;

			for (int i=0; i < trigdesc->numtriggers; i++)
			{
				Trigger *trig = &trigdesc->triggers[i];

				if (is_gpucache_sync_trigger(trig->tgtype,
											 trig->tgenabled,
											 trig->tgfoid,
											 trig->tgnargs))
				{
					if (OidIsValid(trigger_oid))
					{
						elog(WARNING, "relation %s has multiple row-sync trigger for GPU-Cache",
							 RelationGetRelationName(relation));
						goto out;
					}
					trigger_oid = trig->tgoid;
					trigger_name = trig->tgname;
					if (trig->tgnargs == 1)
						trigger_config = trig->tgargs[0];
					break;
				}
			}

			if (OidIsValid(trigger_oid))
			{
				entry->table_sig =
					gpuCacheTableSignatureCommon(WARNING,
												 rel_form,
												 tupdesc,
												 trigger_oid,
												 trigger_name,
												 trigger_config,
												 &entry->gpumask,
												 &entry->gpucache_sz);
			}
		}
	}
out:
	if (p_gpumask)
		*p_gpumask = entry->gpumask;
	if (p_gpucache_sz)
		*p_gpucache_sz = entry->gpucache_sz;
	return entry->table_sig;
}

/*
 * gpuCacheTableSignatureSnapshot
 */
static uint32_t
__gpuCacheTableSignatureSnapshot(Form_pg_class pg_class,
								 Snapshot snapshot,
								 gpumask_t *p_gpumask,
								 size_t *p_gpucache_sz)
{
	Oid			table_oid = pg_class->oid;
	Oid			trigger_oid = InvalidOid;
	const char *trigger_name = NULL;
	const char *trigger_config = NULL;
	Relation	srel;
	ScanKeyData	skey[2];
	SysScanDesc	sscan;
    HeapTuple	tuple;
    TupleDesc	tupdesc;
	uint32_t	table_sig = 0U;

	/* check pg_class */
	if (pg_class->relkind != RELKIND_RELATION ||
		pg_class->relfilenode == 0 ||
		!pg_class->relhastriggers)
		goto out;
	/* check pg_trigger */
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

		if (is_gpucache_sync_trigger(pg_trig->tgtype,
									 pg_trig->tgenabled,
									 pg_trig->tgfoid,
									 pg_trig->tgnargs))
		{
			if (pg_trig->tgnargs == 1)
			{
				Datum	datum;
				bool	isnull;

				datum = heap_getattr(tuple,
									 Anum_pg_trigger_tgargs,
									 RelationGetDescr(srel),
									 &isnull);
				if (!isnull)
					trigger_config = TextDatumGetCString(datum);
			}
			trigger_oid = pg_trig->oid;
			trigger_name = pstrdup(NameStr(pg_trig->tgname));
			break;
		}
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
	if (!OidIsValid(trigger_oid))
		goto out;
	/* check pg_attribute */
	tupdesc = CreateTemplateTupleDesc(pg_class->relnatts);
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

		Assert(attr->attnum > 0 && attr->attnum <= pg_class->relnatts);
		memcpy(TupleDescAttr(tupdesc, attr->attnum-1),
			   attr, ATTRIBUTE_FIXED_PART_SIZE);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	/* parse & validate options */
	table_sig = gpuCacheTableSignatureCommon(WARNING,
											 pg_class,
											 tupdesc,
											 trigger_oid,
											 trigger_name,
											 trigger_config,
											 p_gpumask,
											 p_gpucache_sz);
	FreeTupleDesc(tupdesc);
out:
	return table_sig;
}

static uint32_t
gpuCacheTableSignatureSnapshot(Oid table_oid,
							   Snapshot snapshot,
							   gpumask_t *p_gpumask,
							   size_t *p_gpucache_sz)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;
	uint32_t	table_sig = 0U;

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
		Form_pg_class	pg_class = (Form_pg_class)GETSTRUCT(tuple);

		table_sig = __gpuCacheTableSignatureSnapshot(pg_class,
													 snapshot,
													 p_gpumask,
													 p_gpucache_sz);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);

	return table_sig;
}

/*
 * gpuCacheTableSignatureInvalidation
 */
static void
gpuCacheTableSignatureInvalidation(Oid table_oid)
{
	if (gpucache_table_sig_htab)
		hash_search(gpucache_table_sig_htab,
					&table_oid,
					HASH_REMOVE,
					NULL);
}

/*
 * gpuCacheSharedStateHashIndex
 */
static uint32_t
gpuCacheSharedStateHashIndex(Oid database_oid,
							 Oid table_oid,
							 uint32_t table_sig)
{
	struct {
		Oid			database_oid;
		Oid			table_oid;
		uint32_t	table_sig;
	} hkey;
	memset(&hkey, 0, sizeof(hkey));
	hkey.database_oid = database_oid;
	hkey.table_oid = table_oid;
	hkey.table_sig = table_sig;
	return hash_bytes((unsigned char *)&hkey, sizeof(hkey)) % GPUCACHE_KDS_HASH_NSLOTS;
}

/*
 * getGpuCacheSharedState
 */
static GpuCacheSharedState *
getGpuCacheSharedState(Oid database_oid,
					   Oid table_oid,
					   uint32_t table_sig,
					   gpumask_t gpumask,
					   size_t gpucache_sz)
{
	GpuCacheSharedState *gc_sstate;
	dlist_iter	iter;
	uint32_t	hindex = gpuCacheSharedStateHashIndex(database_oid,
													  table_oid,
													  table_sig);
	pthreadMutexLock(&gpucache_shared_head->hash_mutex);
	dlist_foreach(iter, &gpucache_shared_head->hash_slots[hindex])
	{
		gc_sstate = dlist_container(GpuCacheSharedState,
									chain, iter.cur);
		if (gc_sstate->database_oid == database_oid &&
			gc_sstate->table_oid == table_oid &&
			gc_sstate->table_sig == table_sig)
		{
			Assert(gc_sstate->gpumask == gpumask &&
				   gc_sstate->gpucache_sz == gpucache_sz);
			goto found;
		}
	}
	/* not found */
	if (dlist_is_empty(&gpucache_shared_head->free_list))
		gc_sstate = NULL;
	else
	{
		dlist_node *dnode = dlist_pop_head_node(&gpucache_shared_head->free_list);

		gc_sstate = dlist_container(GpuCacheSharedState, chain, dnode);
		memset(gc_sstate, 0, sizeof(GpuCacheSharedState));
		gc_sstate->database_oid = database_oid;
		gc_sstate->table_oid = table_oid;
		gc_sstate->table_sig = table_sig;
		gc_sstate->gpumask = gpumask;
		gc_sstate->gpucache_sz = gpucache_sz;
		pg_atomic_init_u32(&gc_sstate->phase,
						   GPUCACHE_PHASE__NOT_BUILT);
		dlist_push_tail(&gpucache_shared_head->hash_slots[hindex],
						&gc_sstate->chain);
	}
found:
	pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
	return gc_sstate;
}

/*
 * lookupGpuCacheDesc
 *
 * This function tries to lookup (or create) GpuCacheDesc that is
 * per transaction state of GpuCache.
 * Even if GpuCacheDesc is acquired multiple times, it shall be released
 * at the end of transaction once, so we don't need to release everytime.
 */
static GpuCacheDesc *
lookupGpuCacheDescRaw(Oid table_oid,
					  uint32_t table_sig,
					  TransactionId xid,
					  gpumask_t gpumask,
					  size_t gpucache_sz)
{
	GpuCacheDesc	hkey;
	GpuCacheDesc   *entry;
	bool			found;

	if (!gpucache_table_desc_htab)
	{
		HASHCTL	hctl;

		memset(&hctl, 0, sizeof(hctl));
		hctl.keysize = offsetof(GpuCacheDesc, xid) + sizeof(TransactionId);
		hctl.entrysize = offsetof(GpuCacheDesc, pindex[numGpuDevAttrs]);
		hctl.hcxt = CacheMemoryContext;
		gpucache_table_desc_htab = hash_create("GPU-Cache Descriptor",
											   128,
											   &hctl,
											   HASH_ELEM |
											   HASH_BLOBS |
											   HASH_CONTEXT);
	}
	memset(&hkey, 0, sizeof(hkey));
	hkey.table_oid = table_oid;
	hkey.table_sig = table_sig;
	hkey.xid = xid;
	entry = hash_search(gpucache_table_desc_htab,
						&hkey,
						HASH_ENTER,
						&found);
	if (!found)
	{
		entry->gpumask = gpumask;
		entry->gpucache_sz = gpucache_sz;
		entry->gc_sstate = NULL;
		entry->drop_on_rollback = false;
		entry->drop_on_commit = false;
		entry->nitems = 0;
		initStringInfoCxt(CacheMemoryContext, &entry->buf);
		entry->nr_gpus = get_bitcount(gpumask);
		for (int i=0, k=0; i < numGpuDevAttrs; i++)
		{
			if ((gpumask & (1UL<<i)) != 0)
				entry->pindex[k++] = i;
			assert(k <= entry->nr_gpus);
		}
		entry->gc_sstate = getGpuCacheSharedState(MyDatabaseId,
												  hkey.table_oid,
												  hkey.table_sig,
												  gpumask,
												  gpucache_sz);
		if (!entry->gc_sstate)
			elog(NOTICE, "too much GpuCacheSharedState entries are consumed. Please expand 'pg_strom.gpucache_max_relation_entries' then restart.");
	}
	return (entry->gc_sstate ? entry : NULL);
}

static GpuCacheDesc *
lookupGpuCacheDesc(Relation relation)
{
	GpuCacheDesc   *gc_desc = NULL;
	uint32_t		table_sig;
	gpumask_t		gpumask;
	size_t			gpucache_sz;

	table_sig = gpuCacheTableSignature(relation,
									   &gpumask,
									   &gpucache_sz);
	if (table_sig != 0U)
		gc_desc = lookupGpuCacheDescRaw(RelationGetRelid(relation),
										table_sig,
										GetCurrentTransactionId(),
										gpumask,
										gpucache_sz);
	return gc_desc;
}

/*
 * __gpuCacheSetupCacheLog
 */
static void
__gpuCacheSetupCacheLog(GpuCacheDesc *gc_desc, Relation rel)
{
	GpuCacheLogSetupCache *gc_log;
	TupleDesc	tupdesc = RelationGetDescr(rel);
	size_t		head_sz = estimate_kern_data_store(tupdesc);
	size_t		length = offsetof(GpuCacheLogSetupCache, kds_head) + head_sz;
	size_t		gpucache_sz = gc_desc->gpucache_sz;
	int32_t		data_width = get_rel_data_width(rel, NULL);
	int32_t		avg_width;
	size_t		hash_nslots;

	/* estimation of nitems and nslots */
	avg_width = MAXALIGN(offsetof(kern_hashitem, t.t_bits) +
						 BITMAPLEN(tupdesc->natts) +
						 sizeof(kern_tupitem_xact_attrs)) /* xact attributes */
		+ MAXALIGN(data_width + sizeof(uint32_t))	/* payload */
		+ sizeof(uint64_t)							/* row-id array consumption */
		+ sizeof(uint64_t);							/* hash slot consumption */
	hash_nslots = gpucache_sz / avg_width;
	if (hash_nslots > INT_MAX)
		hash_nslots = INT_MAX;
	else if (hash_nslots < 10000)
		hash_nslots = hash_nslots;

	gc_log = (GpuCacheLogSetupCache *)alloca(length);
	memset(gc_log, 0, offsetof(GpuCacheLogSetupCache, kds_head));
	gc_log->c.type = GCACHE_TX_LOG__SETUP_CACHE;
	gc_log->c.length = length;
	gc_log->database_oid = MyDatabaseId;
	gc_log->table_oid = gc_desc->table_oid;
	gc_log->table_sig = gc_desc->table_sig;
	setup_kern_data_store(&gc_log->kds_head,
						  tupdesc,
						  head_sz,
						  KDS_FORMAT_HASH);
	gc_log->kds_head.table_oid = gc_desc->table_oid;
	gc_log->kds_head.hash_nslots = hash_nslots;
	gc_log->kds_head.length = gpucache_sz;

	gpuCacheSendTxLog(gc_desc, -1, &gc_log->c);
}

/*
 * __gpuCacheEraseCacheLog
 */
static void
__gpuCacheEraseCacheLog(GpuCacheDesc *gc_desc)
{
	GpuCacheLogEraseCache gc_log;

	memset(&gc_log, 0, sizeof(GpuCacheLogEraseCache));
	gc_log.c.type = GCACHE_TX_LOG__ERASE_CACHE;
	gc_log.c.length = sizeof(GpuCacheLogEraseCache);
	gc_log.database_oid = MyDatabaseId;
	gc_log.table_oid = gc_desc->table_oid;
	gc_log.table_sig = gc_desc->table_sig;
	/* mark this GPU-Cache is corrupted */
	if (gc_desc->gc_sstate)
	{
		pg_atomic_write_u32(&gc_desc->gc_sstate->phase,
							GPUCACHE_PHASE__IS_CORRUPTED);
	}
	gpuCacheSendTxLog(gc_desc, -1, &gc_log.c);
}

/*
 * getCtidHash
 */
static inline uint32_t
getCtidHash(ItemPointer p_ctid)
{
	return hash_bytes((unsigned char *)p_ctid, sizeof(ItemPointerData));
}

/*
 * __setupGpuCacheTupleItem
 */
static void
__setupGpuCacheTupleItem(StringInfo buf,
						 Relation rel,
						 HeapTuple tuple,
						 TransactionId xmin,
						 TransactionId xmax)
{
	TupleDesc	tupdesc = RelationGetDescr(rel);
	HeapTupleHeader htup = tuple->t_data;
	uint8_t	   *nullmap = NULL;
	int			nattrs = Min(HeapTupleHeaderGetNatts(tuple->t_data), tupdesc->natts);
	int			base = buf->len;
	uint32_t	head_sz;
	uint32_t	hoff, diff;
	kern_tupitem *tupitem;
	kern_tupitem_xact_attrs *xattrs;
	static uint64_t zero = 0;

	assert(base == MAXALIGN(base));
	/* allocation of the header portion first */
	if (HeapTupleHasNulls(tuple))
	{
		nullmap = htup->t_bits;
		head_sz = MAXALIGN(offsetof(kern_tupitem, t_bits)
						   + BITMAPLEN(nattrs)
						   + sizeof(kern_tupitem_xact_attrs));
	}
	else
	{
		head_sz = MAXALIGN(offsetof(kern_tupitem, t_bits)
						   + sizeof(kern_tupitem_xact_attrs));
	}
	enlargeStringInfo(buf, head_sz);
	memset(buf->data + buf->len, 0, head_sz);
	buf->len += head_sz;
	/* deploy the payload first */
	hoff = htup->t_hoff;
	for (int j=0; j < nattrs; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);
		int		alignval = typealign_get_width(attr->attalign);

		if (nullmap && att_isnull(j, nullmap))
			continue;
		if (attr->attisdropped)
			continue;
		if (attr->attbyval || attr->attlen > 0)
		{
			hoff = TYPEALIGN(alignval, hoff);
			diff = TYPEALIGN(alignval, buf->len) - buf->len;
			if (diff > 0)
				appendBinaryStringInfo(buf, (char *)&zero, diff);
			appendBinaryStringInfo(buf, (char *)htup + hoff, attr->attlen);
			hoff += attr->attlen;
		}
		else if (attr->attlen == -1)
		{
			struct varlena *datum;
			void	   *addr;

			if (!VARATT_NOT_PAD_BYTE((char *)htup + hoff))
				hoff = TYPEALIGN(alignval, hoff);
			addr = (char *)htup + hoff;
			hoff += VARSIZE_ANY(addr);

			datum = pg_detoast_datum_packed(addr);
			if (VARATT_IS_4B(datum))
			{
				diff = TYPEALIGN(alignval, buf->len) - buf->len;
				if (diff > 0)
					appendBinaryStringInfo(buf, (char *)&zero, diff);
			}
			appendBinaryStringInfo(buf, (char *)datum, VARSIZE_ANY(datum));
			if (datum != addr)
				pfree(datum);
		}
		else
		{
			elog(ERROR, "unexpected type length for '%s'",
				 format_type_be(attr->atttypid));
		}
	}
	/* 32bit rowid portion (to be set on GPU-service) */
	diff = INTALIGN(buf->len) - buf->len;
	appendBinaryStringInfo(buf, &zero, diff + sizeof(uint32_t));

	/* header portion */
	tupitem = (kern_tupitem *)(buf->data + base);
	tupitem->t_len = (buf->len - base);
	tupitem->hash  = getCtidHash(&tuple->t_self);
	tupitem->has_xact_attrs = true;
	tupitem->t_infomask2 = nattrs;
	tupitem->t_infomask  = htup->t_infomask;
	tupitem->t_hoff = head_sz + MINIMAL_TUPLE_OFFSET;
	if (nullmap)
		memcpy(tupitem->t_bits, nullmap, BITMAPLEN(nattrs));
	xattrs = KERN_TUPITEM_GET_XACT_ATTRS(tupitem);
	xattrs->xmin = xmin;
	xattrs->xmax = xmax;
	memcpy(&xattrs->ctid, &tuple->t_self, sizeof(ItemPointerData));
}

/*
 * __gpuCacheInsertLog
 */
static void
__gpuCacheInsertLog(GpuCacheDesc *gc_desc,
					Relation rel,
					HeapTuple tuple,
					TransactionId xmin,
					TransactionId xmax)
{
	GpuCacheLogInsert *ins;
	PendingCtidItem pci;
	StringInfoData buf;
	uint32_t	pindex;
	/* setup INSERT log */
	initStringInfo(&buf);
	buf.len = offsetof(GpuCacheLogInsert, tupitem);
	__setupGpuCacheTupleItem(&buf, rel, tuple, xmin, xmax);
	ins = (GpuCacheLogInsert *)buf.data;
	ins->c.type = GCACHE_TX_LOG__INSERT;
	ins->c.length = buf.len;
	ins->database_oid = MyDatabaseId;
	ins->table_oid = gc_desc->table_oid;
	ins->table_sig = gc_desc->table_sig;
	pindex = gc_desc->pindex[ins->tupitem.hash % gc_desc->nr_gpus];
	/* also write out pending ctid */
	pci.tag = 'I';
	pci.pindex = pindex;
	memcpy(&pci.ctid, &tuple->t_self, sizeof(ItemPointerData));
	appendBinaryStringInfo(&gc_desc->buf, &pci, sizeof(pci));
	gc_desc->nitems++;
	/* write to the pipe */
	gpuCacheSendTxLog(gc_desc, pindex, &ins->c);
	pfree(buf.data);
}

/*
 * __gpuCacheDeleteLog
 */
static void
__gpuCacheDeleteLog(GpuCacheDesc *gc_desc, const HeapTuple tuple)
{
	GpuCacheLogDelete del;
	PendingCtidItem pci;
	uint32_t	hash = getCtidHash(&tuple->t_self);
	int			pindex = gc_desc->pindex[hash % gc_desc->nr_gpus];
	/* setup DELETE log */
	del.c.type = GCACHE_TX_LOG__DELETE;
	del.c.length = sizeof(del);
	del.database_oid = MyDatabaseId;
	del.table_oid = gc_desc->table_oid;
	del.table_sig = gc_desc->table_sig;
	del.xid = GetCurrentTransactionId();
	memcpy(&del.ctid, &tuple->t_self, sizeof(ItemPointerData));
	/* also write out commit log */
	pci.tag = 'D';
	pci.pindex = pindex;
	memcpy(&pci.ctid, &tuple->t_self, sizeof(ItemPointerData));
	appendBinaryStringInfo(&gc_desc->buf, &pci, sizeof(pci));
	gc_desc->nitems++;
	/* write to the pipe */
	gpuCacheSendTxLog(gc_desc, pindex, &del.c);
}

/*
 * __initialLoadVisibilityCheck
 */
static bool
__initialLoadVisibilityCheck(HeapTuple tuple,
							 TransactionId *gcache_xmin,
							 TransactionId *gcache_xmax)
{
	HeapTupleHeader	htup = tuple->t_data;
	TransactionId	xmin;
	TransactionId	xmax;

	if (!HeapTupleHeaderXminCommitted(htup))
	{
		if (HeapTupleHeaderXminInvalid(htup))
			return false;
		xmin = HeapTupleHeaderGetRawXmin(htup);
		if (TransactionIdIsCurrentTransactionId(xmin))
		{
			if (HeapTupleHeaderGetCmin(htup) >= GetCurrentCommandId(false))
			{
				/*
				 * This tuple is written in this command (INSERT/UPDATE),
				 * and it should be tracked by the AFTER ROW trigger.
				 * Thus, rowid allocation and insertion shall be done
				 * in the trigger function to be called later.
				 */
				return false;
			}
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
		return false;   /* updated by other, and committed  */
	}
	xmax = HeapTupleHeaderGetRawXmax(htup);
	if (TransactionIdIsCurrentTransactionId(xmax))
	{
		if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
			*gcache_xmax = InvalidTransactionId;
		else if (HeapTupleHeaderGetCmax(htup) >= GetCurrentCommandId(false))
		{
			/*
			 * This tuple is removed by the current command (UPDATE/DELETE),
			 * and its relevant AFTER-ROW trigger function should release its
			 * rowid after the initial loading.
			 * So, at this point, we perform like as this tuple is not removed
			 * yet.
			 */
			*gcache_xmax = InvalidTransactionId;
		}
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
 * __gpuCacheInitialLoadMain
 */
static void
__gpuCacheInitialLoadMain(GpuCacheDesc *gc_desc, Relation rel)
{
	TableScanDesc	hscan;
	HeapTuple		tuple;

	hscan = table_beginscan(rel, SnapshotAny, 0, NULL, SO_NONE);
	while ((tuple = heap_getnext(hscan, ForwardScanDirection)) != NULL)
	{
		TransactionId	xmin, xmax;

		CHECK_FOR_INTERRUPTS();
		if (!__initialLoadVisibilityCheck(tuple, &xmin, &xmax))
			continue;

		__gpuCacheInsertLog(gc_desc, rel, tuple, xmin, xmax);
	}
	table_endscan(hscan);
}

/*
 * __gpuCacheInitialLoad
 */
static bool
__gpuCacheInitialLoad(GpuCacheDesc *gc_desc, Relation rel)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	uint32_t	phase;

	while ((phase = pg_atomic_read_u32(&gc_sstate->phase)) < GPUCACHE_PHASE__IS_READY)
	{
		if (phase == GPUCACHE_PHASE__NOT_BUILT &&
			pg_atomic_compare_exchange_u32(&gc_sstate->phase,
										   &phase,
										   GPUCACHE_PHASE__NOW_LOADING))
		{
			PG_TRY();
			{
				__gpuCacheSetupCacheLog(gc_desc, rel);
				
				__gpuCacheInitialLoadMain(gc_desc, rel);
				pg_atomic_exchange_u32(&gc_sstate->phase,
									   GPUCACHE_PHASE__IS_READY);
			}
			PG_CATCH();
			{
				pg_atomic_exchange_u32(&gc_sstate->phase,
									   GPUCACHE_PHASE__IS_CORRUPTED);
				PG_RE_THROW();
			}
			PG_END_TRY();
		}
		else if (phase == GPUCACHE_PHASE__NOW_LOADING)
		{
			CHECK_FOR_INTERRUPTS();
			pg_usleep(1000L);	/* 1ms */
		}
	}
	return (phase == GPUCACHE_PHASE__IS_READY);
}

/*
 * pgstrom_gpucache_sync_trigger
 */
PG_FUNCTION_INFO_V1(pgstrom_gpucache_sync_trigger);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_sync_trigger(PG_FUNCTION_ARGS)
{
	TriggerData	   *trigdata = (TriggerData *) fcinfo->context;
	GpuCacheDesc   *gc_desc;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as trigger",
			 get_func_name(fcinfo->flinfo->fn_oid));
	if (TRIGGER_FIRED_FOR_ROW(trigdata->tg_event))
	{
		if (!TRIGGER_FIRED_AFTER(trigdata->tg_event))
			elog(ERROR, "%s: must be declared as AFTER ROW trigger",
				 trigdata->tg_trigger->tgname);
		gc_desc = lookupGpuCacheDesc(trigdata->tg_relation);
		if (!gc_desc)
			goto bailout;
		__gpuCacheInitialLoad(gc_desc, trigdata->tg_relation);
		if (TRIGGER_FIRED_BY_INSERT(trigdata->tg_event))
		{
			__gpuCacheInsertLog(gc_desc,
								trigdata->tg_relation,
								trigdata->tg_trigtuple,
								GetCurrentTransactionId(),
								InvalidTransactionId);
		}
		else if (TRIGGER_FIRED_BY_UPDATE(trigdata->tg_event))
		{
			__gpuCacheDeleteLog(gc_desc, trigdata->tg_trigtuple);
			__gpuCacheInsertLog(gc_desc,
								trigdata->tg_relation,
								trigdata->tg_newtuple,
								GetCurrentTransactionId(),
								InvalidTransactionId);
		}
		else if (TRIGGER_FIRED_BY_DELETE(trigdata->tg_event))
		{
			__gpuCacheDeleteLog(gc_desc, trigdata->tg_trigtuple);
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
bailout:
	PG_RETURN_POINTER(NULL);
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_apply_redo);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_apply_redo(PG_FUNCTION_ARGS)
{
	bool	apply_redo_done = false;

	pthreadMutexLock(&gpucache_shared_head->hash_mutex);
	gpucache_shared_head->req_apply_redo |= GetSystemAvailableGpus();
	pg_atomic_fetch_and_u32(&gpucache_shared_head->maintenance, 1);
	pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
	/* wait for completion */
	while (!apply_redo_done)
	{
		pthreadMutexLock(&gpucache_shared_head->hash_mutex);
		if (gpucache_shared_head->req_apply_redo == 0)
			apply_redo_done = true;
		pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
		CHECK_FOR_INTERRUPTS();
		pg_usleep(10000L);
	}
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_compaction);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_compaction(PG_FUNCTION_ARGS)
{
	GpuCacheDesc *gc_desc;
	Oid			table_oid = PG_GETARG_OID(0);
	Relation	rel;

	rel = relation_open(table_oid, AccessShareLock);
	gc_desc = lookupGpuCacheDesc(rel);
	if (!gc_desc)
	{
		 elog(NOTICE, "GPU-Cache for '%s' is not built, so compaction is not necessary",
			  RelationGetRelationName(rel));
	}
	else
	{
		bool	compaction_done = false;

		pthreadMutexLock(&gpucache_shared_head->hash_mutex);
		gc_desc->gc_sstate->req_compaction |= gc_desc->gpumask;
		pg_atomic_fetch_and_u32(&gpucache_shared_head->maintenance, 1);
		pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
		/* wait for completion */
		while (!compaction_done)
		{
			pthreadMutexLock(&gpucache_shared_head->hash_mutex);
			if (gc_desc->gc_sstate->req_compaction == 0)
				compaction_done = true;
			pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
			CHECK_FOR_INTERRUPTS();
			pg_usleep(10000L);
		}
	}
	relation_close(rel, AccessShareLock);
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_recovery);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_recovery(PG_FUNCTION_ARGS)
{
	GpuCacheDesc *gc_desc;
	Oid			table_oid = PG_GETARG_OID(0);
	Relation	rel;

	rel = relation_open(table_oid, AccessShareLock);
	gc_desc = lookupGpuCacheDesc(rel);
	if (!gc_desc)
	{
		elog(NOTICE, "GPU-Cache for '%s' is not built, so recovery is not necessary",
			 RelationGetRelationName(rel));
	}
	else
	{
		uint32_t	phase;
		bool		recovery_done = false;

		pthreadMutexLock(&gpucache_shared_head->hash_mutex);
		phase = pg_atomic_read_u32(&gc_desc->gc_sstate->phase);
		if (phase == GPUCACHE_PHASE__NOT_BUILT ||
			phase == GPUCACHE_PHASE__NOW_LOADING ||
			phase == GPUCACHE_PHASE__IS_READY)
		{
			elog(NOTICE, "GPU-Cache for '%s' is not corrupted, so recovery is not necessary",
				 RelationGetRelationName(rel));
			recovery_done = true;
		}
		else
		{
			gc_desc->gc_sstate->req_recovery |= gc_desc->gpumask;
			pg_atomic_fetch_and_u32(&gpucache_shared_head->maintenance, 1);
		}
		pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
		/* wait for completion */
		while (!recovery_done)
		{
			pthreadMutexLock(&gpucache_shared_head->hash_mutex);
			phase = pg_atomic_read_u32(&gc_desc->gc_sstate->phase);
			if (phase == GPUCACHE_PHASE__NOT_BUILT ||
				phase == GPUCACHE_PHASE__NOW_LOADING ||
				phase == GPUCACHE_PHASE__IS_READY)
				recovery_done = true;
			pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
			CHECK_FOR_INTERRUPTS();
			pg_usleep(10000L);
		}
	}
	relation_close(rel, AccessShareLock);
	PG_RETURN_VOID();
}

static void
append_gpucache_info(StringInfo buf,
					 GpuCacheSharedState *gc_sstate,
					 size_t *p_total_virtual_sz,
					 size_t *p_total_physical_sz)
{
	uint64_t	total_virtual_sz = 0;
	uint64_t	total_physical_sz = 0;
	bool		is_first = true;
	const char *phase;

	appendStringInfo(buf,
					 "{ \"database_oid\" : %u"
					 ", \"table_oid\" : %u"
					 ", \"table_sig\" : %u",
					 gc_sstate->database_oid,
					 gc_sstate->table_oid,
					 gc_sstate->table_sig);
	if (gc_sstate->database_oid == MyDatabaseId)
	{
		const char *table_name = get_rel_name(gc_sstate->table_oid);
		appendStringInfo(buf, ", \"table_name\" : ");
		escape_json(buf, table_name);
	}
	switch (pg_atomic_read_u32(&gc_sstate->phase))
	{
		case GPUCACHE_PHASE__NOT_BUILT:
			phase = "not_built";
			break;
		case GPUCACHE_PHASE__NOW_LOADING:
			phase = "loading";
			break;
		case GPUCACHE_PHASE__IS_READY:
			phase = "ready";
			break;
		case GPUCACHE_PHASE__IS_CORRUPTED:
			phase = "corrupted";
			break;
		default:
			phase = "unknown";
			break;
	}
	appendStringInfo(buf,
					 ", \"gpu_mask\" : \"%08lx\""
					 ", \"cache_sz\" : %lu"
					 ", \"phase\" : \"%s\"",
					 gc_sstate->gpumask,
					 gc_sstate->gpucache_sz,
					 phase);
	appendStringInfo(buf, ", \"gpus\" : [");
	is_first = true;
	for (int k=0; k < numGpuDevAttrs; k++)
	{
		if ((gc_sstate->gpumask & (1UL<<k)) != 0)
		{
			uint32_t	__nitems = pg_atomic_read_u32(&gc_sstate->gpus[k].nitems);
			uint64_t	__usage = pg_atomic_read_u64(&gc_sstate->gpus[k].usage);
			uint32_t	__dead_nitems = pg_atomic_read_u32(&gc_sstate->gpus[k].dead_nitems);
			uint64_t	__dead_space = pg_atomic_read_u64(&gc_sstate->gpus[k].dead_space);
			uint64_t	__virtual_sz = pg_atomic_read_u64(&gc_sstate->gpus[k].virtual_mem_sz);
			uint64_t	__physical_sz = pg_atomic_read_u64(&gc_sstate->gpus[k].physical_mem_sz);

			if (!is_first)
				appendStringInfo(buf, ",");
			appendStringInfo(buf, " {"
							 ", \"nitems\" : %u"
							 ", \"usage\" : %lu"
							 ", \"dead_nitems\" : %u"
							 ", \"dead_space\" : %lu"
							 ", \"virtual_mem_sz : \"%lu\""
							 ", \"physical_mem_sz : \"%lu\""
							 "}",
							 __nitems,
							 __usage,
							 __dead_nitems,
							 __dead_space,
							 __virtual_sz,
							 __physical_sz);
			total_virtual_sz += __virtual_sz;
			total_physical_sz += __physical_sz;
			is_first = false;
		}
	}
	appendStringInfo(buf, "]}");
	if (p_total_virtual_sz)
		*p_total_virtual_sz += total_virtual_sz;
	if (p_total_physical_sz)
		*p_total_physical_sz += total_physical_sz;
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_info_one);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_info_one(PG_FUNCTION_ARGS)
{
	StringInfoData buf;
    Oid         table_oid = PG_GETARG_OID(0);
	Relation	rel;
	GpuCacheDesc *gc_desc;

	initStringInfo(&buf);
	buf.len = VARHDRSZ;
	rel = relation_open(table_oid, AccessShareLock);
    gc_desc = lookupGpuCacheDesc(rel);
	if (!gc_desc)
	{
		relation_close(rel, AccessShareLock);
		PG_RETURN_NULL();
	}
	append_gpucache_info(&buf, gc_desc->gc_sstate, NULL, NULL);
	relation_close(rel, AccessShareLock);
	SET_VARSIZE(buf.data, buf.len);
	PG_RETURN_POINTER(buf.data);
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_info);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_info(PG_FUNCTION_ARGS)
{
	StringInfoData	buf;
	dlist_iter		iter;

	initStringInfo(&buf);
	buf.len = VARHDRSZ;
	pthreadMutexLock(&gpucache_shared_head->hash_mutex);
	PG_TRY();
	{
		int		num_gpucache = 0;
		size_t	total_virtual_sz = 0;
		size_t	total_physical_sz = 0;

		appendStringInfo(&buf, "{ \"caches\" : [");
		for (int hindex=0; hindex < GPUCACHE_STATE_HASH_NSLOTS; hindex++)
		{
			dlist_foreach(iter, &gpucache_shared_head->hash_slots[hindex])
			{
				 GpuCacheSharedState *gc_sstate
					 = dlist_container(GpuCacheSharedState,
									   chain, iter.cur);
				 if (num_gpucache > 0)
					 appendStringInfo(&buf, ", ");
				 append_gpucache_info(&buf, gc_sstate,
									  &total_virtual_sz,
									  &total_physical_sz);
				 num_gpucache++;
			}
		}
		appendStringInfo(&buf, "], "
						 "\"num_caches\" : %d, "
						 "\"total_virtual_sz\" : %lu, "
						 "\"total_physical_sz\" : %lu }",
						 num_gpucache,
						 total_virtual_sz,
						 total_physical_sz);
	}
	PG_CATCH();
	{
		pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
		PG_RE_THROW();
	}
	PG_END_TRY();
	pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
	SET_VARSIZE(buf.data, buf.len);
	PG_RETURN_POINTER(buf.data);
}

/* ------------------------------------------------------------
 *
 * Routines for query optimization
 *
 * ------------------------------------------------------------
 */
bool
baseRelHasGpuCache(PlannerInfo *root, RelOptInfo *baserel)
{
	RangeTblEntry *rte = root->simple_rte_array[baserel->relid];
	bool	retval = false;

	if (pgstrom_enable_gpucache &&
		rte->rtekind == RTE_RELATION &&
		(baserel->reloptkind == RELOPT_BASEREL ||
		 baserel->reloptkind == RELOPT_OTHER_MEMBER_REL))
    {
		Relation	rel = table_open(rte->relid, NoLock);
		GpuCacheDesc *gc_desc = lookupGpuCacheDesc(rel);
		uint32_t	phase;

		if (gc_desc)
		{
			phase = pg_atomic_read_u32(&gc_desc->gc_sstate->phase);
			if (phase == GPUCACHE_PHASE__NOT_BUILT ||
				phase == GPUCACHE_PHASE__NOW_LOADING ||
				phase == GPUCACHE_PHASE__IS_READY)
				retval = true;
		}
		table_close(rel, NoLock);
	}
	return retval;
}

/* ------------------------------------------------------------
 *
 * Routines for query execution
 *
 * ------------------------------------------------------------
 */
bool
pgstromGpuCacheExecInit(pgstromTaskScanState *ptss)
{
	ptss->gpucache_scan_count = NULL;
	if (pgstrom_enable_gpucache)
	{
		GpuCacheDesc *gc_desc = lookupGpuCacheDesc(ptss->scan_rel);

		if (gc_desc && __gpuCacheInitialLoad(gc_desc, ptss->scan_rel))
			ptss->gpucache_desc = gc_desc;
	}
	return (ptss->gpucache_desc != NULL);
}

XpuCommand *
pgstromScanChunkGpuCache(pgstromTaskState *pts,
						 pgstromTaskScanState *ptss,
						 struct iovec *xcmd_iov,
						 int *xcmd_iovcnt)
{
	GpuCacheDesc *gc_desc = ptss->gpucache_desc;
	uint32_t	gpucache_count;

	if (!ptss->gpucache_scan_count)
		ptss->gpucache_scan_count = &ptss->__gpucache_count_data;
	gpucache_count = pg_atomic_fetch_add_u32(ptss->gpucache_scan_count, 1);
	if (gpucache_count < gc_desc->nr_gpus)
	{
		XpuCommand	__xcmd;

		if (!__gpuCacheInitialLoad(gc_desc, ptss->scan_rel))
			elog(ERROR, "gpucache: corrupted GPU-Cache for '%s'",
				 RelationGetRelationName(ptss->scan_rel));
		resetStringInfo(&pts->xcmd_buf);
		memset(&__xcmd, 0, sizeof(__xcmd));
		__xcmd.magic  = XpuCommandMagicNumber;
		__xcmd.tag    = XpuCommandTag__XpuTaskExecGpuCache;
		__xcmd.length = offsetof(XpuCommand, u.gc_task) + sizeof(kern_exec_task_gpucache);
		__xcmd.u.gc_task.t.scan_relidx = ptss->scan_relidx;
		__xcmd.u.gc_task.database_oid = MyDatabaseId;
		__xcmd.u.gc_task.table_oid    = gc_desc->table_oid;
		__xcmd.u.gc_task.table_sig    = gc_desc->table_sig;
		__xcmd.u.gc_task.cuda_dindex  = gc_desc->pindex[gpucache_count];
		appendBinaryStringInfo(&pts->xcmd_buf, &__xcmd, __xcmd.length);

		xcmd_iov->iov_base = pts->xcmd_buf.data;
		xcmd_iov->iov_len  = pts->xcmd_buf.len;
		*xcmd_iovcnt = 1;

		return (XpuCommand *)pts->xcmd_buf.data;
	}
	else
	{
		/* move to the next relation */
		pts->curr_scan_rel++;
	}
	return NULL;
}

void
pgstromGpuCacheExecReset(pgstromTaskScanState *ptss)
{
	ptss->gpucache_scan_count = NULL;
}

void
pgstromGpuCacheInitDSM(pgstromTaskScanState *ptss,
					   pgstromSharedScanState *psss)
{
	ptss->gpucache_scan_count = &psss->gpucache_count_data;
}

void
pgstromGpuCacheAttachDSM(pgstromTaskScanState *ptss,
						 pgstromSharedScanState *psss)
{
	ptss->gpucache_scan_count = &psss->gpucache_count_data;
}

void
pgstromGpuCacheShutdownDSM(pgstromTaskScanState *ptss,
						   pgstromSharedScanState *psss)
{
	if (ptss->gpucache_scan_count)
	{
		uint32_t	ival = pg_atomic_read_u32(ptss->gpucache_scan_count);

		pg_atomic_write_u32(&ptss->__gpucache_count_data, ival);
		ptss->gpucache_scan_count = &ptss->__gpucache_count_data;
	}
}

void
pgstromGpuCacheExplain(pgstromTaskScanState *ptss,
					   ExplainState *es,
					   const char *prefix)
{
	GpuCacheDesc *gc_desc = ptss->gpucache_desc;
	char	label[100];
	char	buf[1024];

	if (gc_desc)
	{
		if (!prefix)
			snprintf(label, sizeof(label), "GPU-Cache");
		else
			snprintf(label, sizeof(label), "GPU-Cache [%s]", prefix);

		snprintf(buf, sizeof(buf),
				 "cache-sz: %s, gpumask: %08lx",
				 format_bytesz(gc_desc->gpucache_sz),
				 gc_desc->gpumask);
		ExplainPropertyText(label, buf, es);
	}
}

/* ------------------------------------------------------------
 *
 * Routines to support DDL callbacks
 *
 * ------------------------------------------------------------
 */
static void
__gpuCacheCallbackOnAlterTable(Oid table_oid)
{
	uint32_t	new_signature, old_signature;
	gpumask_t	new_gpumask, old_gpumask;
	size_t		new_gpucache_sz, old_gpucache_sz;
	GpuCacheDesc *gc_desc;

	new_signature = gpuCacheTableSignatureSnapshot(table_oid,
												   SnapshotSelf,
												   &new_gpumask,
												   &new_gpucache_sz);
	old_signature = gpuCacheTableSignatureSnapshot(table_oid,
												   NULL,
												   &old_gpumask,
												   &old_gpucache_sz);
	if (old_signature != 0U &&
		old_signature != new_signature)
	{
	    gc_desc = lookupGpuCacheDescRaw(table_oid,
										old_signature,
										GetCurrentTransactionId(),
										old_gpumask,
										old_gpucache_sz);
		if (gc_desc)
			gc_desc->drop_on_commit = true;
    }

	if (new_signature != 0U &&
		new_signature != old_signature)
	{
		gc_desc = lookupGpuCacheDescRaw(table_oid,
										new_signature,
										GetCurrentTransactionId(),
										new_gpumask,
										new_gpucache_sz);
		if (gc_desc)
			gc_desc->drop_on_rollback = true;
	}
}

static void
__gpuCacheCallbackOnAlterTrigger(Oid trigger_oid, int elevel)
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
		Form_pg_trigger pg_trig = (Form_pg_trigger)GETSTRUCT(tuple);
		Relation	__rel = table_open(pg_trig->tgrelid, NoLock);
		const char *trigger_name = NameStr(pg_trig->tgname);
		const char *trigger_config = NULL;
		Datum		datum;
		bool		isnull;

		datum = heap_getattr(tuple,
							 Anum_pg_trigger_tgargs,
							 RelationGetDescr(srel),
							 &isnull);
		if (!isnull)
			trigger_config = TextDatumGetCString(datum);

		gpuCacheTableSignatureCommon(ERROR,
									 RelationGetForm(__rel),
									 RelationGetDescr(__rel),
									 trigger_oid,
									 trigger_name,
									 trigger_config,
									 NULL, NULL);
		/* duplication checks */
		if (__rel->trigdesc)
		{
			TriggerDesc *trigdesc = __rel->trigdesc;

			for (int i=0; i < trigdesc->numtriggers; i++)
			{
				Trigger *trigger = &trigdesc->triggers[i];

				if (trigger->tgoid != trigger_oid &&
					is_gpucache_sync_trigger(trigger->tgtype,
											 trigger->tgenabled,
											 trigger->tgfoid,
											 trigger->tgnargs))
				{
					elog(elevel, "gpucache: relation %s has multiple row-sync triggers",
						 RelationGetRelationName(__rel));
				}
			}
		}
		table_close(__rel, NoLock);

		__gpuCacheCallbackOnAlterTable(pg_trig->tgrelid);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
}

static void
__gpuCacheOnDropRelation(Oid table_oid)
{
	uint32_t	table_sig;
	gpumask_t	gpumask;
	size_t		gpucache_sz;
	GpuCacheDesc *gc_desc;

	table_sig = gpuCacheTableSignatureSnapshot(table_oid,
											   NULL,
											   &gpumask,
											   &gpucache_sz);
	if (table_sig != 0U)
	{
		gc_desc = lookupGpuCacheDescRaw(table_oid,
										table_sig,
										GetCurrentTransactionId(),
										gpumask,
										gpucache_sz);
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
		Oid			table_oid = ((Form_pg_trigger) GETSTRUCT(tuple))->tgrelid;
		uint32_t	table_sig;
		gpumask_t	gpumask;
		size_t		gpucache_sz;
		GpuCacheDesc *gc_desc;

		table_sig = gpuCacheTableSignatureSnapshot(table_oid,
												   NULL,
												   &gpumask,
												   &gpucache_sz);
		if (table_sig != 0U)
		{
			gc_desc = lookupGpuCacheDescRaw(table_oid,
											table_sig,
											GetCurrentTransactionId(),
											gpumask,
											gpucache_sz);
			if (gc_desc)
				gc_desc->drop_on_commit = true;
		}
	}
    systable_endscan(sscan);
    table_close(srel, AccessShareLock);
}

static void
__gpuCacheTruncateLog(Oid table_oid)
{
	uint32_t	table_sig;
	gpumask_t	gpumask;
	size_t		gpucache_sz;
	GpuCacheDesc *gc_desc;

	table_sig = gpuCacheTableSignatureSnapshot(table_oid,
											   NULL,
											   &gpumask,
											   &gpucache_sz);
	if (table_sig != 0U)
	{
		gc_desc = lookupGpuCacheDescRaw(table_oid,
										table_sig,
										GetCurrentTransactionId(),
										gpumask,
										gpucache_sz);
		if (gc_desc)
			gc_desc->drop_on_commit = true;
	}
}

/*
 * gpuCacheObjectAccess
 */
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
			__gpuCacheCallbackOnAlterTable(objectId);
		}
		else if (classId == TriggerRelationId)
		{
			/* CREATE OR REPLACE TRIGGER */
			__gpuCacheCallbackOnAlterTrigger(objectId, ERROR);
		}
	}
	else if (access == OAT_POST_ALTER)
	{
		if (classId == RelationRelationId)
		{
			/* ALTER TABLE */
			__gpuCacheCallbackOnAlterTable(objectId);
		}
		else if (classId == TriggerRelationId)
		{
			/* ALTER TRIGGER */
			__gpuCacheCallbackOnAlterTrigger(objectId, WARNING);
		}
	}
	else if (access == OAT_DROP)
	{
		if (classId == RelationRelationId)
		{
			/* DROP TABLE */
			/* ALTER TABLE ... DROP COLUMN */
			__gpuCacheOnDropRelation(objectId);
		}
		else if (classId == TriggerRelationId)
		{
			/* DROP TRIGGER */
			__gpuCacheOnDropTrigger(objectId);
		}
	}
	else if (access == OAT_TRUNCATE)
	{
		__gpuCacheTruncateLog(objectId);
	}
}

/*
 * releaseGpuCacheDesc
 */
static void
releaseGpuCacheDesc(GpuCacheDesc *gc_desc, bool normal_commit)
{
	if (normal_commit
		? gc_desc->drop_on_commit
		: gc_desc->drop_on_rollback)
	{
		__gpuCacheEraseCacheLog(gc_desc);
	}
	else
	{
		const char *pos = gc_desc->buf.data;

		for (int i=0; i < gc_desc->nitems; i++, pos += sizeof(PendingCtidItem))
		{
			const PendingCtidItem *pci = (const PendingCtidItem *)pos;
			GpuCacheLogXact xact;

			if (pci->tag == 'I')
			{
				if (normal_commit)
					xact.c.type = GCACHE_TX_LOG__COMMIT_INS;
				else
					xact.c.type = GCACHE_TX_LOG__ABORT_INS;
			}
			else if (pci->tag == 'D')
			{
				if (normal_commit)
					xact.c.type = GCACHE_TX_LOG__COMMIT_DEL;
				else
					xact.c.type = GCACHE_TX_LOG__ABORT_DEL;
			}
			else
			{
				elog(WARNING, "Bug? unexpected PendingCtidItem tag '%c'",
					 pci->tag);
				continue;
			}
			xact.c.length = sizeof(GpuCacheLogXact);
			xact.database_oid = MyDatabaseId;
			xact.table_oid = gc_desc->table_oid;
			xact.table_sig = gc_desc->table_sig;
			memcpy(&xact.ctid, &pci->ctid, sizeof(ItemPointerData));
			gpuCacheSendTxLog(gc_desc, pci->pindex, &xact.c);
		}
	}
	/* cleanup itself */
	if (gc_desc->buf.data)
		pfree(gc_desc->buf.data);
	hash_search(gpucache_table_desc_htab,
				gc_desc, HASH_REMOVE, NULL);
}

/*
 * gpuCacheXactCallback
 */
static void
gpuCacheXactCallback(XactEvent event, void *arg)
{
#ifdef GPUCACHE_DEBUG_MESSAGE
	elog(INFO, "XactCallback: ev=%s xid=%u top-xid=%u",
		 event == XACT_EVENT_COMMIT  ? "XACT_EVENT_COMMIT" :
		 event == XACT_EVENT_ABORT   ? "XACT_EVENT_ABORT"  :
		 event == XACT_EVENT_PREPARE ? "XACT_EVENT_PREPARE" :
		 event == XACT_EVENT_PRE_COMMIT ? "XACT_EVENT_PRE_COMMIT" :
		 event == XACT_EVENT_PRE_PREPARE ? "XACT_EVENT_PRE_PREPARE" : "????",
		 GetCurrentTransactionIdIfAny(),
		 GetTopTransactionIdIfAny());
#endif
	if (gpucache_table_desc_htab &&
		hash_get_num_entries(gpucache_table_desc_htab) > 0 &&
		(event == XACT_EVENT_COMMIT || event == XACT_EVENT_ABORT))
	{
		TransactionId   curr_xid = GetCurrentTransactionIdIfAny();
		HASH_SEQ_STATUS hseq;
		GpuCacheDesc   *gc_desc;
		bool            normal_commit = (event == XACT_EVENT_COMMIT);

		hash_seq_init(&hseq, gpucache_table_desc_htab);
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
#ifdef GPUCACHE_DEBUG_MESSAGE
	elog(INFO, "SubXactCallback: ev=%s xid=%u top-xid=%u",
		 event == SUBXACT_EVENT_START_SUB ? "SUBXACT_EVENT_START_SUB" :
		 event == SUBXACT_EVENT_COMMIT_SUB ? "SUBXACT_EVENT_COMMIT_SUB" :
		 event == SUBXACT_EVENT_ABORT_SUB ? "SUBXACT_EVENT_ABORT_SUB" :
		 event == SUBXACT_EVENT_PRE_COMMIT_SUB ? "SUBXACT_EVENT_PRE_COMMIT_SUB" : "???",
		 GetCurrentTransactionIdIfAny(),
		 GetTopTransactionIdIfAny());
#endif
	if (gpucache_table_desc_htab &&
		hash_get_num_entries(gpucache_table_desc_htab) > 0 &&
		event == SUBXACT_EVENT_ABORT_SUB)
	{
		TransactionId   curr_xid = GetCurrentTransactionIdIfAny();
		HASH_SEQ_STATUS hseq;
		GpuCacheDesc   *gc_desc;

		hash_seq_init(&hseq, gpucache_table_desc_htab);
		while ((gc_desc = hash_seq_search(&hseq)) != NULL)
		{
			if (gc_desc->xid == curr_xid)
				releaseGpuCacheDesc(gc_desc, false);
		}
	}
}

static void
gpuCacheRelcacheCallback(Datum arg, Oid relid)
{
#ifdef GPUCACHE_DEBUG_MESSAGE
	elog(LOG, "pid=%u: gpuCacheRelcacheCallback (table_oid=%u)", getpid(), relid);
#endif
	gpuCacheTableSignatureInvalidation(relid);
}

static void
gpuCacheSyscacheCallback(Datum arg, int cacheid, uint32 hashvalue)
{
#ifdef GPUCACHE_DEBUG_MESSAGE
	elog(LOG, "pid=%u: gpuCacheSyscacheCallback (cacheid=%u)", getpid(), cacheid);
#endif
	__gpucache_sync_trigger_function_oid = InvalidOid;
}

/*
 * pgstrom_request_gpu_cache
 */
static void
pgstrom_request_gpu_cache(void)
{
	size_t		len;

	if (shmem_request_next)
		shmem_request_next();
	len = MAXALIGN(offsetof(GpuCacheSharedHead, gpus[numGpuDevAttrs]))
		+ MAXALIGN(offsetof(GpuCacheSharedState, gpus[numGpuDevAttrs]))
		* pgstrom_gpucache_max_relation_entries;
	RequestAddinShmemSpace(len);
}

/*
 * pgstrom_startup_gpu_cache
 */
static void
pgstrom_startup_gpu_cache(void)
{
	size_t		len;
	size_t		unitsz;
	char	   *pos;
	bool		found;

	if (shmem_startup_next)
		shmem_startup_next();
	unitsz = MAXALIGN(offsetof(GpuCacheSharedState, gpus[numGpuDevAttrs]));
	len = MAXALIGN(offsetof(GpuCacheSharedHead, gpus[numGpuDevAttrs]))
		+ unitsz * pgstrom_gpucache_max_relation_entries;
	pos = ShmemInitStruct("GPU-Cache Shared Head", len, &found);
	if (found)
		elog(ERROR, "Bug? GpuCacheSharedHead already exist");
	memset(pos, 0, len);

	gpucache_shared_head = (GpuCacheSharedHead *)pos;
	dlist_init(&gpucache_shared_head->free_list);
	for (int i=0; i < GPUCACHE_STATE_HASH_NSLOTS; i++)
		dlist_init(&gpucache_shared_head->hash_slots[i]);
	pthreadMutexInitShared(&gpucache_shared_head->hash_mutex);
	for (int k=0; k < numGpuDevAttrs; k++)
		pthreadMutexInitShared(&gpucache_shared_head->gpus[k].pipe_mutex);
	pos += MAXALIGN(offsetof(GpuCacheSharedHead, gpus[numGpuDevAttrs]));

	for (int k=0; k < pgstrom_gpucache_max_relation_entries; k++)
	{
		GpuCacheSharedState *gc_sstate = (GpuCacheSharedState *)pos;

		dlist_push_tail(&gpucache_shared_head->free_list, &gc_sstate->chain);
		pos += unitsz;
	}
	/* setup pipe */
	gpucache_pipe_read_fdesc = calloc(2*numGpuDevAttrs, sizeof(int));
	gpucache_pipe_write_fdesc = gpucache_pipe_read_fdesc + numGpuDevAttrs;
	for (int k=0; k < numGpuDevAttrs; k++)
	{
		int		__pipefd[2];

		if (pipe(__pipefd) != 0)
			elog(ERROR, "failed on pipe(2): %m");
		gpucache_pipe_read_fdesc[k] = __pipefd[0];
		gpucache_pipe_write_fdesc[k] = __pipefd[1];
	}
}

/*
 * pgstrom_init_gpu_store
 */
void
pgstrom_init_gpu_cache(void)
{
	static int	__pgstrom_gpucache_log_buffer_sz_mb;

	/* pg_strom.enable_gpucache */
	DefineCustomBoolVariable("pg_strom.enable_gpucache",
							 "Enables GPU-Cache as data source for Scan",
							 NULL,
							 &pgstrom_enable_gpucache,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.gpucache_max_relation_entries */
	DefineCustomIntVariable("pg_strom.gpucache_max_relation_entries",
							"max number of relations GPU-Cache can be configured. Usually, no need to change from the default.",
							NULL,
							&pgstrom_gpucache_max_relation_entries,
							1024,
							1024,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* pg_strom.gpucache_sync_threshold */
	DefineCustomIntVariable("pg_strom.gpucache_log_buffer_size",
							"gpucache: log buffer size per GPU",
							NULL,
							&__pgstrom_gpucache_log_buffer_sz_mb,
							512,	/* 512MB */
							64,		/* 64MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MB,
							NULL, NULL, NULL);
	pgstrom_gpucache_log_buffer_sz = (size_t)__pgstrom_gpucache_log_buffer_sz_mb << 20;
	/*  pg_strom.gpucache_sync_interval */
	DefineCustomIntVariable("pg_strom.gpucache_sync_interval",
							"gpucache: time to kick update log synchronization in ms",
							NULL,
							&pgstrom_gpucache_sync_interval,
							3000,	/* 3.0sec */
							500,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* pg_strom.gpucache_sync_threshold */
	DefineCustomIntVariable("pg_strom.gpucache_sync_threshold",
							"gpucache: size to kick update log synchronization in MB",
							NULL,
							&pgstrom_gpucache_sync_threshold,
							256,	/* 256MB */
							16,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
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
}

/* ----------------------------------------------------------------
 *
 * GPU-Cache Service Code
 *
 * Note that the code below shall be executed under the GPU-Service context,
 * so it is not capable to touch PostgreSQL objects.
 *
 * ----------------------------------------------------------------
 */
#define __GC_LOG(fmt,...)		__gsLogLabel("GPU-cache worker",fmt,##__VA_ARGS__)

/*
 * gpuCacheAllocMasterState
 */
static bool
gpuCacheAllocMasterState(gpuContext *gcontext)
{
	if (!gcontext->gpucache_master_state)
	{
		size_t		bufsz = (256UL << 20);
		CUdeviceptr	m_bufptr;
		CUresult	rc;

		rc = cuMemAllocManaged(&m_bufptr, bufsz,
							   CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
		{
			__GC_LOG("out of managed memory");
			return false;
		}
		memset((void *)m_bufptr, 0, offsetof(kern_gpucache_master_state, log_items));
		gcontext->gpucache_master_state = (kern_gpucache_master_state *)m_bufptr;
		gcontext->gpucache_master_state->length = bufsz;
	}
	return true;
}

/*
 * gpuCacheReleaseMasterState
 */
static void
gpuCacheReleaseMasterState(gpuContext *gcontext)
{
	pthreadRWLockWriteLock(&gcontext->gpucache_rwlock);
	if (gcontext->gpucache_master_state)
	{
		cuMemFree((CUdeviceptr)gcontext->gpucache_master_state);
		gcontext->gpucache_master_state = NULL;
	}
	pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
}

/*
 * gpuCacheMarkAsCorrupted
 */
static void
gpuCacheMarkAsCorrupted(kern_gpucache_data_store *curr)
{
	dlist_iter	iter;
	uint32_t	hindex = gpuCacheSharedStateHashIndex(curr->database_oid,
													  curr->table_oid,
													  curr->table_sig);

	pthreadMutexLock(&gpucache_shared_head->hash_mutex);
	dlist_foreach(iter, &gpucache_shared_head->hash_slots[hindex])
	{
		GpuCacheSharedState *gc_sstate = dlist_container(GpuCacheSharedState,
														 chain, iter.cur);
		if (gc_sstate->database_oid == curr->database_oid &&
			gc_sstate->table_oid    == curr->table_oid &&
			gc_sstate->table_sig    == curr->table_sig)
		{
			pg_atomic_write_u32(&gc_sstate->phase,
								GPUCACHE_PHASE__IS_CORRUPTED);
		}
	}
	/* notify other GPU-Cache services to release corrupted caches */
	pg_atomic_fetch_and_u32(&gpucache_shared_head->maintenance, 1);
	pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
}

/*
 * gpuCacheExecCompactionOne
 */
static kern_gpucache_data_store *
gpuCacheExecCompactionOne(gpuContext *gcontext, kern_gpucache_data_store *orig)
{
	kern_gpucache_data_store *comp = NULL;
	size_t		length = (offsetof(kern_gpucache_data_store, kds) + orig->kds.length);
	int			grid_sz;
	int			block_sz;
	void	   *kern_args[2];
	CUresult	rc;

	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 gcontext->cufn_gpucache_compaction, 0);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("failed on gpuOptimalBlockSize: %s", cuStrError(rc));
		return NULL;
	}

	rc = cuMemAllocManaged((CUdeviceptr *)&comp, length,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("out of managed memory");
		return NULL;
	}
	/* make an empty KDS */
	comp->next         = orig->next;
	comp->database_oid = orig->database_oid;
	comp->table_oid    = orig->table_oid;
	comp->table_sig    = orig->table_sig;
	comp->dead_items_nums = 0;
	comp->dead_items_sz = 0;
	memcpy(&comp->kds,
		   &orig->kds,
		   KDS_HEAD_LENGTH(&orig->kds));
	comp->kds.nitems = 0;	/* reset */
	comp->kds.usage  = 0;	/* reset */

	/* launch compaction kernel */
	kern_args[0] = &comp;
	kern_args[1] = &orig;
	rc = cuLaunchKernel(gcontext->cufn_gpucache_compaction,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						0);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("failed on cuLaunchKernel(kern_gpucache_compaction): %s",
				 cuStrError(rc));
		(void)cuStreamSynchronize(CU_STREAM_PER_THREAD);
		(void)cuMemFree((CUdeviceptr)comp);
		return NULL;
	}
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("failed on cuStreamSynchronize: %s",
				 cuStrError(rc));
		(void)cuMemFree((CUdeviceptr)comp);
		return NULL;
	}
	return comp;
}

/*
 * gpuCacheExecCompaction
 */
static bool
gpuCacheExecCompactionNoLock(gpuContext *gcontext)
{
	kern_gpucache_master_state *gc_mstate = gcontext->gpucache_master_state;
	double		compaction_ratio = 0.80;
	CUresult	rc;

	for (int i=0; i < GPUCACHE_KDS_HASH_NSLOTS; i++)
	{
		kern_gpucache_data_store *curr = gc_mstate->hslots[i];
		kern_gpucache_data_store *prev = NULL;

		while (curr)
		{
			size_t	threshold = (double)curr->kds.length * compaction_ratio;
			size_t	consumed  = (offsetof(kern_gpucache_data_store, kds) +
								 sizeof(uint64_t) * (curr->kds.hash_nslots +
													curr->kds.nitems) + curr->kds.usage);
			size_t	deadspace = (curr->dead_items_nums * sizeof(uint64_t) +
								 curr->dead_items_sz);

			if (consumed <= threshold)
			{
				/* no need to kick compaction */
				prev = curr;
				curr = curr->next;
			}
			else if (deadspace > 0 &&
					 consumed - deadspace <= curr->kds.length)
			{
				/* compaction may make sense */
				kern_gpucache_data_store *comp
					= gpuCacheExecCompactionOne(gcontext, curr);
				if (comp)
				{
					/* compaction done, replaced */
					assert(comp->next == curr->next);
					rc = cuMemFree((CUdeviceptr)curr);
					if (rc != CUDA_SUCCESS)
						__GC_LOG("failed on cuMemFree: %s", cuStrError(rc));

					if (prev)
						prev->next = comp;
					else
						gc_mstate->hslots[i] = comp;
					curr = comp;
				}
				else
				{
					/* failed on compaction */
					kern_gpucache_data_store *next = curr->next;

					gpuCacheMarkAsCorrupted(curr);
					rc = cuMemFree((CUdeviceptr)curr);
					if (rc != CUDA_SUCCESS)
						__GC_LOG("failed on cuMemFree: %s", cuStrError(rc));
					/* detach the corrupted one */
					if (prev)
						prev->next = next;
					else
						gc_mstate->hslots[i] = next;
					curr = next;
				}
			}
			else if (consumed > curr->kds.length)
			{
				/* compaction does not make sense, corrupted */
				kern_gpucache_data_store *next = curr->next;

				gpuCacheMarkAsCorrupted(curr);
				rc = cuMemFree((CUdeviceptr)curr);
				if (rc != CUDA_SUCCESS)
					__GC_LOG("failed on cuMemFree: %s", cuStrError(rc));
				/* detach corrupted one */
				if (prev)
					prev->next = next;
				else
					gc_mstate->hslots[i] = next;
				curr = next;
			}
		}
	}
	return true;
}

/*
 * gpuCacheMaintenanceCompaction
 *
 * NOTE: caller must hold exclusive lock
 */
static void
gpuCacheMaintenanceCompaction(gpuContext *gcontext,
							  GpuCacheSharedState *gc_sstate)
{
	if (gcontext->gpucache_master_state)
	{
		kern_gpucache_master_state *gc_mstate = gcontext->gpucache_master_state;
		kern_gpucache_data_store *curr;
		kern_gpucache_data_store *prev = NULL;
		uint32_t	hindex
			= gpuCacheSharedStateHashIndex(gc_sstate->database_oid,
										   gc_sstate->table_oid,
										   gc_sstate->table_sig);
		curr = gc_mstate->hslots[hindex];
		while (curr)
		{
			kern_gpucache_data_store *next = curr->next;
			CUresult	rc;

			if (curr->database_oid == gc_sstate->database_oid &&
				curr->table_oid    == gc_sstate->table_oid &&
				curr->table_sig    == gc_sstate->table_sig)
			{
				kern_gpucache_data_store *comp
					= gpuCacheExecCompactionOne(gcontext, curr);
				if (comp)
				{
					/* compaction done, replaced */
					assert(comp->next == next);
					rc = cuMemFree((CUdeviceptr)curr);
					if (rc != CUDA_SUCCESS)
						__GC_LOG("failed on cuMemFree: %s", cuStrError(rc));
					if (prev)
						prev->next = comp;
					else
						gc_mstate->hslots[hindex] = comp;
					prev = comp;
				}
				else
				{
					/* failed on compaction */
					pg_atomic_write_u32(&gc_sstate->phase,
										GPUCACHE_PHASE__IS_CORRUPTED);
					rc = cuMemFree((CUdeviceptr)curr);
					if (rc != CUDA_SUCCESS)
						__GC_LOG("failed on cuMemFree: %s", cuStrError(rc));
					if (prev)
						prev->next = next;
					else
						gc_mstate->hslots[hindex] = next;
				}
			}
			curr = next;
		}
	}
}

/*
 * gpuCacheMaintenanceRecovery
 *
 * NOTE: caller must hold exclusive lock
 */
static void
gpuCacheMaintenanceRecovery(gpuContext *gcontext,
							GpuCacheSharedState *gc_sstate)
{
	kern_gpucache_master_state *gc_mstate = gcontext->gpucache_master_state;

	if (gc_mstate)
	{
		kern_gpucache_data_store *curr;
		kern_gpucache_data_store *prev = NULL;
		uint32_t	hindex
			= gpuCacheSharedStateHashIndex(gc_sstate->database_oid,
										   gc_sstate->table_oid,
										   gc_sstate->table_sig);
		curr = gc_mstate->hslots[hindex];
		while (curr)
		{
			kern_gpucache_data_store *next = curr->next;
			CUresult	rc;

			if (curr->database_oid == gc_sstate->database_oid &&
				curr->table_oid    == gc_sstate->table_oid &&
				curr->table_sig    == gc_sstate->table_sig)
			{
				/* release the buffer */
				rc = cuMemFree((CUdeviceptr)curr);
				if (rc != CUDA_SUCCESS)
					 __GC_LOG("failed on cuMemFree: %s", cuStrError(rc));
				if (prev)
					prev->next = next;
				else
					gc_mstate->hslots[hindex] = next;
			}
			curr = next;
		}
	}
}

/*
 * gpuCacheFlushPendingLogs
 */
static bool
gpuCacheFlushPendingLogs(gpuContext *gcontext)
{
	kern_gpucache_master_state *gc_mstate;
	CUresult	rc;
	int			grid_sz;
	int			block_sz;
	void	   *kern_args[2];
	bool		compaction_done = false;
	bool		retval = false;

	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 gcontext->cufn_gpucache_apply_logs, 0);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("failed on gpuOptimalBlockSize for kern_gpucache_apply_logs: %s",
				 cuStrError(rc));
		return false;
	}
	pthreadRWLockWriteLock(&gcontext->gpucache_rwlock);
	gc_mstate = gcontext->gpucache_master_state;
again:
	for (int phase=1; phase <= 3; phase++)
	{
		kern_args[0] = &gc_mstate;
		kern_args[1] = &phase;
		rc = cuLaunchKernel(gcontext->cufn_gpucache_apply_logs,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							0,
							CU_STREAM_PER_THREAD,
							kern_args,
							0);
		if (rc != CUDA_SUCCESS)
		{
			__GC_LOG("failed on cuLaunchKernel(kern_gpucache_apply_logs): %s",
					 cuStrError(rc));
			(void)cuStreamSynchronize(CU_STREAM_PER_THREAD);
			goto out;
		}
	}
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("failed on cuStreamSynchronize: %s",
				 cuStrError(rc));
		goto out;
	}
	/*
	 * If ERRCODE_SUSPEND_NO_SPACE was returned, it means at least one GPU-Cache
	 * has memory pressure. So, we try compaction once.
	 * If compaction cannot help the memory pressure, we mark this GPU-Cache as
	 * corrupted, and ignore the further updates, until it is recovered by hand.
	 */
	if (gc_mstate->kerror.errcode == ERRCODE_SUSPEND_NO_SPACE && !compaction_done)
	{
		if (gpuCacheExecCompactionNoLock(gcontext))
		{
			compaction_done = true;
			goto again;
		}
		__GC_LOG("failed on GPU-Cache compaction");
	}
	else if (gc_mstate->kerror.errcode != ERRCODE_STROM_SUCCESS)
	{
		__GC_LOG("%s (%s:%d, %s)",
				 gc_mstate->kerror.message,
				 gc_mstate->kerror.filename,
				 gc_mstate->kerror.lineno,
				 gc_mstate->kerror.funcname);
	}
	else
	{
		gc_mstate->nitems = 0;
		gc_mstate->usage  = 0;
		retval = true;
	}
out:
	pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
	return retval;
}

/*
 * gpuCacheProcessSetupCacheLog
 */
static bool
gpuCacheProcessSetupCacheLog(gpuContext *gcontext, GpuCacheLogSetupCache *log)
{
	kern_gpucache_data_store *kds_gc;
	kern_gpucache_master_state *gc_mstate;
	CUdeviceptr	m_kds_gc;
	CUresult	rc;
	size_t		head_sz;
	size_t		required;
	uint32_t	hindex;

	if (!gpuCacheFlushPendingLogs(gcontext))
		return false;
	/* sanity checks */
	head_sz = KDS_HEAD_LENGTH(&log->kds_head);
	if (offsetof(GpuCacheLogSetupCache, kds_head) + head_sz > log->c.length)
	{
		__GC_LOG("corrupted GCACHE_TX_LOG__SETUP_CACHE log (len=%lu %u)",
				 offsetof(GpuCacheLogSetupCache, kds_head) + head_sz,
				 log->c.length);
		return false;
	}
	required = offsetof(kern_gpucache_data_store,
						kds) + Max(head_sz, log->kds_head.length);
	rc = cuMemAllocManaged(&m_kds_gc, required,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("out of managed memory");
		return false;
	}
	kds_gc = (kern_gpucache_data_store *)m_kds_gc;
	kds_gc->database_oid = log->database_oid;
	kds_gc->table_oid    = log->table_oid;
	kds_gc->table_sig    = log->table_sig;
	memcpy(&kds_gc->kds, &log->kds_head, head_sz);

	/* add to the gpucache master state */
	hindex = gpuCacheSharedStateHashIndex(log->database_oid,
										  log->table_oid,
										  log->table_sig);
	pthreadRWLockWriteLock(&gcontext->gpucache_rwlock);
	gc_mstate = gcontext->gpucache_master_state;
	kds_gc->next = gc_mstate->hslots[hindex];
	gc_mstate->hslots[hindex] = kds_gc;
	pthreadRWLockUnlock(&gcontext->gpucache_rwlock);

	return true;
}

/*
 * gpuCacheProcessEraseCacheLog
 */
static bool
gpuCacheProcessEraseCacheLog(gpuContext *gcontext, GpuCacheLogEraseCache *log)
{
	kern_gpucache_data_store *kds_curr, *kds_prev;
	kern_gpucache_master_state *gc_mstate;
	uint32_t	hindex;
	CUresult	rc;

	if (!gpuCacheFlushPendingLogs(gcontext))
		return false;
	hindex = gpuCacheSharedStateHashIndex(log->database_oid,
										  log->table_oid,
										  log->table_sig);
	pthreadRWLockWriteLock(&gcontext->gpucache_rwlock);
	gc_mstate = (kern_gpucache_master_state *)gcontext->gpucache_master_state;
	for (kds_curr = gc_mstate->hslots[hindex], kds_prev = NULL;
		 kds_curr != NULL;
		 kds_prev = kds_curr, kds_curr = kds_curr->next)
	{
		if (kds_curr->database_oid == log->database_oid &&
			kds_curr->table_oid    == log->table_oid &&
			kds_curr->table_sig    == log->table_sig)
		{
			if (!kds_prev)
				gc_mstate->hslots[hindex] = kds_curr->next;
			else
				kds_prev->next = kds_curr->next;

			rc = cuMemFree((CUdeviceptr)kds_curr);
			if (rc != CUDA_SUCCESS)
				__GC_LOG("failed on cuMemFree: %s", cuStrError(rc));
			goto found;
		}
	}
	__GC_LOG("ERASE-CACHE no GPU-Cache buffer found for database=%u table=%u/%08x",
			 log->database_oid, log->table_oid, log->table_sig);
found:
	pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
	return true;
}

/*
 * gpuCacheProcessRedoLog
 */
static bool
gpuCacheProcessRedoLog(gpuContext *gcontext, GpuCacheLogCommon *log)
{
	kern_gpucache_master_state *gc_mstate = gcontext->gpucache_master_state;
	union {
		uint64_t	u64;
		struct {
			uint32_t nitems;
			uint32_t usage;
		} s;
	} oldval, curval, newval;
restart:
	pthreadRWLockReadLock(&gcontext->gpucache_rwlock);
	oldval.u64 = *((volatile uint64_t *)&gc_mstate->nitems);
	for (;;)
	{
		newval.s.nitems = oldval.s.nitems + 1;
		newval.s.usage  = oldval.s.usage + MAXALIGN(log->length);
		if (offsetof(kern_gpucache_master_state, log_items) +
			sizeof(uint32_t) * newval.s.nitems +
			newval.s.usage > gc_mstate->length)
		{
			/* buffer is full, so flush it once */
			pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
			if (!gpuCacheFlushPendingLogs(gcontext))
				return false;
			goto restart;
		}
		curval.u64 = __atomic_cas_uint64((uint64_t *)&gc_mstate->nitems,
										 oldval.u64, newval.u64);
		if (curval.u64 != oldval.u64)
		{
			/* try again */
			oldval.u64 = curval.u64;
		}
		else
		{
			uint32_t	offset = (gc_mstate->length - newval.s.usage);

			gc_mstate->log_items[newval.s.nitems - 1] = offset;
			memcpy((char *)gc_mstate + offset, log, log->length);
			break;
		}
	}
	pthreadRWLockUnlock(&gcontext->gpucache_rwlock);

	return true;
}

/*
 * gpuCacheProcessOneRedoLog
 */
static bool
gpuCacheProcessOneRedoLog(gpuContext *gcontext, GpuCacheLogCommon *log)
{
	if (!gpuCacheAllocMasterState(gcontext))
		return false;
	switch (log->type)
	{
		case GCACHE_TX_LOG__SETUP_CACHE:
			return gpuCacheProcessSetupCacheLog(gcontext,
												(GpuCacheLogSetupCache *)log);
		case GCACHE_TX_LOG__ERASE_CACHE:
			return gpuCacheProcessEraseCacheLog(gcontext,
												(GpuCacheLogEraseCache *)log);
		case GCACHE_TX_LOG__INSERT:
		case GCACHE_TX_LOG__DELETE:
		case GCACHE_TX_LOG__COMMIT_INS:
		case GCACHE_TX_LOG__COMMIT_DEL:
		case GCACHE_TX_LOG__ABORT_INS:
		case GCACHE_TX_LOG__ABORT_DEL:
			return gpuCacheProcessRedoLog(gcontext, log);
		default:
			__GC_LOG("unknown GPU-Cache log type: %08x", log->type);
			break;
	}
	return false;
}

/*
 * gpuCacheMaintenanceHandler
 */
static void
gpuCacheMaintenanceHandler(gpuContext *gcontext)
{
	gpumask_t	cuda_dmask = gcontext->cuda_dmask;
	dlist_iter	iter;

	pthreadRWLockWriteLock(&gcontext->gpucache_rwlock);
	pthreadMutexLock(&gpucache_shared_head->hash_mutex);
	if ((gpucache_shared_head->req_apply_redo & cuda_dmask) != 0)
	{
		gpuCacheFlushPendingLogs(gcontext);
		gpucache_shared_head->req_apply_redo &= ~cuda_dmask;
	}
	for (int k=0; k < GPUCACHE_STATE_HASH_NSLOTS; k++)
	{
		dlist_foreach(iter, &gpucache_shared_head->hash_slots[k])
		{
			GpuCacheSharedState *gc_sstate = dlist_container(GpuCacheSharedState,
															 chain, iter.cur);
			if ((gc_sstate->req_recovery & cuda_dmask) != 0)
			{
				gpuCacheMaintenanceRecovery(gcontext, gc_sstate);
				gc_sstate->req_recovery &= ~cuda_dmask;
				if (gc_sstate->req_recovery == 0)
					pg_atomic_write_u32(&gc_sstate->phase, GPUCACHE_PHASE__NOT_BUILT);
			}
			if ((gc_sstate->req_compaction & cuda_dmask) != 0)
			{
				gpuCacheMaintenanceCompaction(gcontext, gc_sstate);
				gc_sstate->req_compaction &= ~cuda_dmask;
			}
		}
	}
	pthreadMutexUnlock(&gpucache_shared_head->hash_mutex);
	pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
}

/*
 * gpuCacheWorkerMain
 */
void *
gpuCacheWorkerMain(void *__gcontext)
{
	gpuContext *gcontext = (gpuContext *)__gcontext;
	int			fdesc = gpucache_pipe_read_fdesc[gcontext->cuda_dindex];
	char	   *pipe_buffer = NULL;
	size_t		pipe_bufsz = (4UL<<20);	/* 4MB */
	char	   *wip_buffer = NULL;
	size_t		wip_length = (4UL<<20);	/* start from 4MB */
	size_t		wip_usage = 0;
	uint32_t	maintenance_curr = 0;
	uint32_t	maintenance_last = 0;
	CUresult	rc;

	__GC_LOG("start for GPU-%d", gcontext->cuda_dindex);
	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		__GC_LOG("failed on cuCtxSetCurrent: %s", cuStrError(rc));
		return NULL;
	}
	pipe_buffer = malloc(pipe_bufsz);
	if (!pipe_buffer)
	{
		__GC_LOG("out of memory");
		goto bailout;
	}
	wip_buffer = malloc(wip_length);
	if (!wip_buffer)
	{
		__GC_LOG("out of memory");
		goto bailout;
	}
	/* main loop */
	while (!gpuServiceGoingTerminate())
	{
		GpuCacheLogCommon *log = NULL;
		struct pollfd pfd;
		size_t		read_off = 0;
		ssize_t		nbytes;

		/* maintenance works */
		maintenance_curr = pg_atomic_read_u32(&gpucache_shared_head->maintenance);
		if (maintenance_curr != maintenance_last)
		{
			gpuCacheMaintenanceHandler(gcontext);
			maintenance_last = maintenance_curr;
		}
		/* wait for the next message */
		pfd.fd = fdesc;
		pfd.events = POLLIN | POLLERR | POLLHUP;
		pfd.revents = 0;
		if (poll(&pfd, 1, 1000) <= 0)
		{
			/* check gpuServiceGoingTerminate() */
			continue;
		}
		nbytes = read(fdesc, pipe_buffer, pipe_bufsz);
		if (nbytes <= 0)
		{
			/* check gpuServiceGoingTerminate() */
			continue;
		}

		if (wip_usage > 0)
		{
			size_t	sz;

			if (wip_usage + nbytes < sizeof(GpuCacheLogCommon))
			{
				memcpy(wip_buffer + wip_usage, pipe_buffer, nbytes);
				wip_usage += nbytes;
				continue;
			}
			log = (GpuCacheLogCommon *)wip_buffer;
			if (wip_length < log->length)
			{
				size_t	new_length = (log->length + wip_length);
				char   *new_buffer = realloc(wip_buffer, new_length);

				if (!new_buffer)
				{
					__GC_LOG("out of memory");
					goto bailout;
				}
				wip_buffer = new_buffer;
				wip_length = new_length;
				log = (GpuCacheLogCommon *)wip_buffer;
			}
			sz = Min(log->length - wip_usage, nbytes);
			memcpy(wip_buffer + wip_usage, pipe_buffer, sz);
			wip_usage += sz;
			if (wip_usage < log->length)
				continue;
			gpuCacheProcessOneRedoLog(gcontext, log);
			read_off = sz;
		}
		while (read_off < nbytes)
		{
			size_t	remained = nbytes - read_off;

			if (remained < sizeof(GpuCacheLogCommon))
			{
				memcpy(wip_buffer, pipe_buffer + read_off, remained);
				wip_usage = remained;
				break;
			}
			log = (GpuCacheLogCommon *)(pipe_buffer + read_off);
			if (remained < log->length)
			{
				if (wip_length < log->length)
				{
					size_t	new_length = (log->length + wip_length);
					char   *new_buffer = realloc(wip_buffer, new_length);
					if (!new_buffer)
					{
						__GC_LOG("out of memory");
						goto bailout;
					}
					wip_buffer = new_buffer;
					wip_length = new_length;
				}
				memcpy(wip_buffer, log, remained);
				wip_usage = remained;
				break;
			}
			gpuCacheProcessOneRedoLog(gcontext, log);
			read_off += log->length;
		}
	}
bailout:
	__GC_LOG("exit for GPU-%d", gcontext->cuda_dindex);
	/* cleanup */
	gpuCacheReleaseMasterState(gcontext);
	if (wip_buffer)
		free(wip_buffer);
	if (pipe_buffer)
		free(pipe_buffer);
	return NULL;
}

/*
 * gpuCacheGetKdsBuffer
 */
CUdeviceptr
gpuCacheGetKdsBuffer(gpuContext *gcontext,
					 uint32_t database_oid,
					 uint32_t table_oid,
					 uint32_t table_sig)
{
	kern_gpucache_master_state *gc_mstate;
	CUdeviceptr	m_kds = 0UL;
again:
	pthreadRWLockReadLock(&gcontext->gpucache_rwlock);
	gc_mstate = gcontext->gpucache_master_state;
	if (gc_mstate)
	{
		kern_gpucache_data_store *kds_gc;
		uint32_t	hindex;

		/* flush pending logs if any */
		if (gc_mstate->nitems > 0)
		{
			pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
			gpuCacheFlushPendingLogs(gcontext);
			goto again;
		}
		hindex = gpuCacheSharedStateHashIndex(database_oid,
											  table_oid,
											  table_sig);
		for (kds_gc = gc_mstate->hslots[hindex];
			 kds_gc != NULL;
			 kds_gc = kds_gc->next)
		{
			if (kds_gc->database_oid == database_oid &&
				kds_gc->table_oid    == table_oid &&
				kds_gc->table_sig    == table_sig)
			{
				m_kds = (CUdeviceptr)&kds_gc->kds;
				break;
			}
		}
	}
	if (m_kds == 0UL)
		pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
	return m_kds;
}

/*
 * gpuCachePutKdsBuffer
 */
void
gpuCachePutKdsBuffer(gpuContext *gcontext)
{
	pthreadRWLockUnlock(&gcontext->gpucache_rwlock);
}
