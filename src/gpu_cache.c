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
	/* current status */
#define GPUCACHE_PHASE__NOT_BUILT		0	/* not built yet */
#define GPUCACHE_PHASE__NOW_LOADING		1	/* now initial loading */
#define GPUCACHE_PHASE__IS_READY		2	/* now ready */
#define GPUCACHE_PHASE__IS_CORRUPTED	3	/* corrupted */
    pg_atomic_uint32 phase;
	/* statistics */
	//TODO
} GpuCacheSharedState;

/*
 * GpuCacheSharedHead
 */
#define GPUCACHE_STATE_HASH_NSLOTS		509
typedef struct
{
	pthread_mutex_t hash_mutex;
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
typedef struct
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
} GpuCacheDesc;

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
#if 1
	switch (gc_log->type)
	{
		case GCACHE_TX_LOG__SETUP_CACHE:
			elog(NOTICE, "gpucache: SETUP CACHE table_oid=%u table_sig=%08x len=%u",
				 ((const GpuCacheLogSetupCache *)gc_log)->table_oid,
				 ((const GpuCacheLogSetupCache *)gc_log)->table_sig,
				 gc_log->length);
			break;
		case GCACHE_TX_LOG__ERASE_CACHE:
			elog(NOTICE, "gpucache: DROP TABLE table_oid=%u table_sig=%08x len=%u",
				 ((const GpuCacheLogEraseCache *)gc_log)->table_oid,
                 ((const GpuCacheLogEraseCache *)gc_log)->table_sig,
				 gc_log->length);
			break;
		case GCACHE_TX_LOG__INSERT:
			elog(NOTICE, "gpucache: INSERT table_oid=%u table_sig=%08x len=%u",
				 ((const GpuCacheLogInsert *)gc_log)->table_oid,
				 ((const GpuCacheLogInsert *)gc_log)->table_sig,
				 gc_log->length);
			break;
		case GCACHE_TX_LOG__DELETE:
			elog(NOTICE, "gpucache: DELETE table_oid=%u table_sig=%08x len=%u",
				 ((const GpuCacheLogDelete *)gc_log)->table_oid,
				 ((const GpuCacheLogDelete *)gc_log)->table_sig,
				 gc_log->length);
			break;
		case GCACHE_TX_LOG__COMMIT_INS:
			elog(NOTICE, "gpucache: COMMIT_INS");
			break;
		case GCACHE_TX_LOG__COMMIT_DEL:
			elog(NOTICE, "gpucache: COMMIT_DEL");
			break;
		case GCACHE_TX_LOG__ABORT_INS:
			elog(NOTICE, "gpucache: ABORT_INS");
			break;
		case GCACHE_TX_LOG__ABORT_DEL:
			elog(NOTICE, "gpucache: ABORT_DEL");
			break;
		default:
			elog(NOTICE, "gpucache: UNKNOWN");
			break;
	}
	return;
#endif
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
 * gpuCacheSharedStateHash
 */
static uint32_t
gpuCacheSharedStateHash(Oid database_oid,
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
	return hash_bytes((unsigned char *)&hkey, sizeof(hkey));
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
	uint32_t		hash = gpuCacheSharedStateHash(database_oid, table_oid, table_sig);
	uint32_t		hindex = (hash % GPUCACHE_STATE_HASH_NSLOTS);
	dlist_iter		iter;
	GpuCacheSharedState *gc_sstate;

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
						 sizeof(GpuCacheSysAttr))	/* in front of the payload */
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
	uint8_t	   *nullmap = NULL;
	int			nattrs = Min(HeapTupleHeaderGetNatts(tuple->t_data), tupdesc->natts);
	int			base = buf->len;
	uint32_t	head_sz;
	uint32_t	hoff, diff;
	kern_tupitem *tupitem;
	GpuCacheSysAttr *sysatt;
	static uint64_t zero = 0;

	assert(base == MAXALIGN(base));
	/* allocation of the header portion first */
	if (HeapTupleHasNulls(tuple))
	{
		nullmap = tuple->t_data->t_bits;
		head_sz = MAXALIGN(offsetof(kern_tupitem, t_bits)
						   + BITMAPLEN(nattrs)
						   + sizeof(GpuCacheSysAttr));
	}
	else
	{
		head_sz = MAXALIGN(offsetof(kern_tupitem, t_bits)
						   + sizeof(GpuCacheSysAttr));
	}
	enlargeStringInfo(buf, head_sz);
	memset(buf->data + buf->len, 0, head_sz);
	buf->len += head_sz;
	/* deploy the payload first */
	hoff = tuple->t_data->t_hoff;
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
			appendBinaryStringInfo(buf, (char *)tuple->t_data + hoff, attr->attlen);
			hoff += attr->attlen;
		}
		else if (attr->attlen == -1)
		{
			struct varlena *datum;
			void	   *addr;

			if (!VARATT_NOT_PAD_BYTE((char *)tuple + hoff))
				hoff = TYPEALIGN(alignval, hoff);
			addr = (char *)tuple + hoff;
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
	tupitem->t_infomask2 = nattrs;
	tupitem->t_infomask  = tuple->t_data->t_infomask;
	tupitem->t_hoff = head_sz;
	if (nullmap)
		memcpy(tupitem->t_bits, nullmap, BITMAPLEN(nattrs));
	sysatt = (GpuCacheSysAttr *)((char *)tupitem + head_sz - sizeof(GpuCacheSysAttr));
	sysatt->xmin = xmin;
	sysatt->xmax = xmax;
	memcpy(&sysatt->ctid, &tuple->t_self, sizeof(ItemPointerData));
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


	hscan = table_beginscan(rel, SnapshotAny, 0, NULL);
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
	PG_RETURN_POINTER(trigdata->tg_trigtuple);
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_apply_redo);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_apply_redo(PG_FUNCTION_ARGS)
{
	elog(ERROR, "pgstrom.gpucache_apply_redo is no longer supported");
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_compaction);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_compaction(PG_FUNCTION_ARGS)
{
	elog(ERROR, "pgstrom.gpucache_compaction is no longer supported");
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_recovery);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_recovery(PG_FUNCTION_ARGS)
{
	elog(ERROR, "pgstrom.gpucache_recovery is no longer supported");
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_info);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_info(PG_FUNCTION_ARGS)
{
	elog(ERROR, "pgstrom.gpucache_info is no longer supported");
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
			xact.pindex = pci->pindex;
			memcpy(&xact.ctid, &pci->ctid, sizeof(ItemPointerData));
			gpuCacheSendTxLog(gc_desc, xact.pindex, &xact.c);
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
		+ MAXALIGN(sizeof(GpuCacheSharedState) * pgstrom_gpucache_max_relation_entries);
	RequestAddinShmemSpace(len);
}

/*
 * pgstrom_startup_gpu_cache
 */
static void
pgstrom_startup_gpu_cache(void)
{
	GpuCacheSharedState *gc_sstate;
	size_t		len;
	char	   *pos;
	bool		found;

	if (shmem_startup_next)
		shmem_startup_next();
	len = MAXALIGN(offsetof(GpuCacheSharedHead, gpus[numGpuDevAttrs]))
		+ MAXALIGN(sizeof(GpuCacheSharedState) * pgstrom_gpucache_max_relation_entries);
	pos = ShmemInitStruct("GPU-Cache Shared Head", len, &found);
	if (found)
		elog(ERROR, "Bug? GpuCacheSharedHead already exist");
	memset(pos, 0, len);

	gpucache_shared_head = (GpuCacheSharedHead *)pos;
	pos += MAXALIGN(offsetof(GpuCacheSharedHead, gpus[numGpuDevAttrs]));
	dlist_init(&gpucache_shared_head->free_list);
	for (int i=0; i < GPUCACHE_STATE_HASH_NSLOTS; i++)
		dlist_init(&gpucache_shared_head->hash_slots[i]);
	pthreadMutexInitShared(&gpucache_shared_head->hash_mutex);
	for (int k=0; k < numGpuDevAttrs; k++)
		pthreadMutexInitShared(&gpucache_shared_head->gpus[k].pipe_mutex);

	gc_sstate = (GpuCacheSharedState *)pos;
	pos += MAXALIGN(sizeof(GpuCacheSharedState) * pgstrom_gpucache_max_relation_entries);
	for (int k=0; k < pgstrom_gpucache_max_relation_entries; k++)
	{
		dlist_push_tail(&gpucache_shared_head->free_list, &gc_sstate[k].chain);
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
