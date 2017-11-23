/*
 * ccache.c
 *
 * Columnar cache implementation of PG-Strom
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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

#define CCACHE_MAX_NUM_DATABASES	100
#define CCBUILDER_STATE__SHUTDOWN	0
#define CCBUILDER_STATE__STARTUP	1
#define CCBUILDER_STATE__PRELOAD	2
#define CCBUILDER_STATE__TRYLOAD	3
#define CCBUILDER_STATE__SLEEP		4
typedef struct
{
	char		dbname[NAMEDATALEN];
	bool		invalid_database;
	pg_atomic_uint64 curr_scan_pos;
} ccacheDatabase;

typedef struct
{
	Oid			database_oid;
	int			state;		/* one of CCBUILDER_STATE__* */
	Latch	   *latch;
} ccacheBuilder;

typedef struct
{
	/* hash slot of ccache chunks */
	slock_t		free_chunks_lock;
	dlist_head	free_chunks_list;
	slock_t	   *active_locks;
	dlist_head *active_slots;
	/* management of ccache builder workers */
	pg_atomic_uint32 generation;
	slock_t			lock;
	int				rr_count;
	int				num_databases;
	ccacheDatabase	databases[CCACHE_MAX_NUM_DATABASES];
	ccacheBuilder	builders[FLEXIBLE_ARRAY_MEMBER];
} ccacheState;

/*
 * ccacheChunk
 */
#define CCACHE_CHUNK_SIZE		(128L << 20)	/* 128MB */

typedef struct
{
	dlist_node	chain;
	pg_crc32	hash;			/* hash value */
	Oid			database_oid;	/* OID of the cached database */
	Oid			table_oid;		/* OID of the cached table */
	BlockNumber	block_nr;		/* block number where is head of the chunk */
	cl_int		refcnt;			/* reference counter */
	struct timeval ctime;		/* time of the cache creation */
	struct timeval atime;		/* time of the last access */
	struct {
		off_t	offset;
		size_t	length;
	} attrs[FLEXIBLE_ARRAY_MEMBER];
} ccacheChunk;

/* static variables */
static shmem_startup_hook_type shmem_startup_next = NULL;
static char		   *ccache_startup_databases;	/* GUC */
static int			ccache_num_builders;		/* GUC */
static size_t		ccache_max_size;			/* GUC */
static char		   *ccache_base_dir_name;		/* GUC */
static DIR		   *ccache_base_dir = NULL;
static ccacheState *ccache_state = NULL;		/* shmem */
static cl_int		ccache_num_chunks;
static cl_int		ccache_num_slots;
static cl_int 		ccache_max_nattrs;			/* GUC (hidden) */
static ccacheDatabase *ccache_database = NULL;	/* only builder */
static ccacheBuilder  *ccache_builder = NULL;	/* only builder */
static bool			ccache_builder_got_sigterm = false;
static oidvector   *ccache_relations_oid = NULL;
static Oid			ccache_invalidator_func_oid = InvalidOid;

extern void ccache_builder_main(Datum arg);

/*
 * ccache_compute_hashvalue
 */
static inline pg_crc32
ccache_compute_hashvalue(Oid database_oid, Oid table_oid, BlockNumber block_nr)
{
	pg_crc32	hash;

	Assert((block_nr & ((CCACHE_CHUNK_SIZE / BLCKSZ) - 1)) == 0);
	INIT_LEGACY_CRC32(hash);
	COMP_LEGACY_CRC32(hash, &database_oid, sizeof(Oid));
	COMP_LEGACY_CRC32(hash, &table_oid, sizeof(Oid));
	COMP_LEGACY_CRC32(hash, &block_nr, sizeof(BlockNumber));
	FIN_LEGACY_CRC32(hash);

	return hash;
}

/*
 * ccache_chunk_filename
 */
static inline void
ccache_chunk_filename(char *fname,
					  Oid database_oid, Oid table_oid, BlockNumber block_nr)
{
	Assert((block_nr & ((CCACHE_CHUNK_SIZE / BLCKSZ)-1)) == 0);

	snprintf(fname, MAXPGPATH, "CC%u-%u:%ld.dat",
			 database_oid, table_oid,
			 block_nr / (CCACHE_CHUNK_SIZE / BLCKSZ));
}

/*
 * ccache_get_chunk
 */
static ccacheChunk *
ccache_get_chunk(Oid table_oid, BlockNumber block_nr)
{
	pg_crc32	hash;
	cl_int		index;
	dlist_iter	iter;
	ccacheChunk *cc_chunk = NULL;

	hash = ccache_compute_hashvalue(MyDatabaseId, table_oid, block_nr);
	index = hash % ccache_num_slots;

	SpinLockAcquire(&ccache_state->active_locks[index]);
	dlist_foreach (iter, &ccache_state->active_slots[index])
	{
		ccacheChunk *temp = dlist_container(ccacheChunk, chain, iter.cur);

		if (temp->hash == hash &&
			temp->database_oid == MyDatabaseId &&
			temp->table_oid == table_oid &&
			temp->block_nr == block_nr)
		{
			cc_chunk = temp;
			cc_chunk->refcnt++;
			break;
		}
	}
	SpinLockRelease(&ccache_state->active_locks[index]);

	return cc_chunk;
}

/*
 * ccache_put_chunk
 */
static void
ccache_put_chunk_nolock(ccacheChunk *cc_chunk)
{
	Assert(cc_chunk->refcnt > 0);
	if (--cc_chunk->refcnt == 0)
	{
		char		fname[MAXPGPATH];

		Assert(cc_chunk->chain.prev == NULL &&
			   cc_chunk->chain.next == NULL);
		ccache_chunk_filename(fname,
							  cc_chunk->database_oid,
							  cc_chunk->table_oid,
							  cc_chunk->block_nr);
		if (unlinkat(dirfd(ccache_base_dir), fname, 0) != 0)
			elog(WARNING, "failed on unlinkat \"%s\": %m", fname);
		/* back to the free list */
		SpinLockAcquire(&ccache_state->free_chunks_lock);
		dlist_push_head(&ccache_state->free_chunks_list,
						&cc_chunk->chain);
		SpinLockRelease(&ccache_state->free_chunks_lock);
	}
}

static void
ccache_put_chunk(ccacheChunk *cc_chunk)
{
	cl_int		index = cc_chunk->hash % ccache_num_slots;

	SpinLockAcquire(&ccache_state->active_locks[index]);
	ccache_put_chunk_nolock(cc_chunk);
	SpinLockRelease(&ccache_state->active_locks[index]);
}











/*
 * ccache_invalidator_oid - returns OID of invalidator trigger function
 */
static Oid
ccache_invalidator_oid(bool missing_ok)
{
	Oid			pgstrom_namespace_oid;
	oidvector	proc_args;
	Form_pg_proc proc_form;
	HeapTuple	tup;
	PGFunction	invalidator_fn;
	Oid			invalidator_oid;
	Datum		datum;
	bool		isnull;
	char	   *probin;
	char	   *prosrc;

	if (OidIsValid(ccache_invalidator_func_oid))
		return ccache_invalidator_func_oid;

	pgstrom_namespace_oid = get_namespace_oid("pgstrom", missing_ok);
	if (!OidIsValid(pgstrom_namespace_oid))
		return InvalidOid;

	SET_VARSIZE(&proc_args, offsetof(oidvector, values));
	proc_args.ndim = 1;
	proc_args.dataoffset = 0;
	proc_args.elemtype = OIDOID;
	proc_args.dim1 = 0;
	proc_args.lbound1 = 1;

	tup = SearchSysCache3(PROCNAMEARGSNSP,
						  CStringGetDatum("ccache_invalidator"),
						  PointerGetDatum(&proc_args),
						  ObjectIdGetDatum(pgstrom_namespace_oid));
	if (!HeapTupleIsValid(tup))
	{
		if (!missing_ok)
			elog(ERROR, "cache lookup failed for function pgstrom.ccache_invalidator");
		return InvalidOid;
	}
	invalidator_oid = HeapTupleGetOid(tup);
	proc_form = (Form_pg_proc) GETSTRUCT(tup);

	if (proc_form->prolang != ClanguageId)
		elog(ERROR, "pgstrom.ccache_invalidator is not C function");

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_prosrc, &isnull);
	if (isnull)
		elog(ERROR, "null prosrc for pgstrom.ccache_invalidator function");
	prosrc = TextDatumGetCString(datum);

	datum = SysCacheGetAttr(PROCOID, tup, Anum_pg_proc_probin, &isnull);
	if (isnull)
		elog(ERROR, "null probin for pgstrom.ccache_invalidator function");
	probin = TextDatumGetCString(datum);
	ReleaseSysCache(tup);

	invalidator_fn = load_external_function(probin, prosrc,
											!missing_ok, NULL);
	if (invalidator_fn != pgstrom_ccache_invalidator)
		return InvalidOid;

	elog(LOG, "ccache invalidator found: oid=%u %s",
		 invalidator_oid, format_procedure(invalidator_oid));

	ccache_invalidator_func_oid = invalidator_oid;
	return ccache_invalidator_func_oid;
}

/*
 * refresh_ccache_source_relations
 */
static void
refresh_ccache_source_relations(void)
{
	Oid			invalidator_oid;
	Relation	hrel;
	Relation	irel;
	SysScanDesc	sscan;
	HeapTuple	tup;
	List	   *relations_oid = NIL;
	ListCell   *lc;
	cl_int		i, len;
	Oid			curr_relid = InvalidOid;
	bool		has_row_insert = false;
	bool		has_row_update = false;
	bool		has_row_delete = false;
	bool		has_stmt_truncate = false;

	/* already built? */
	if (ccache_relations_oid)
		return;
	/* invalidator function installed? */
	invalidator_oid = ccache_invalidator_oid(true);
	if (!OidIsValid(invalidator_oid))
		return;
	/* walk on the pg_trigger catalog */
	hrel = heap_open(TriggerRelationId, AccessShareLock);
	irel = index_open(TriggerRelidNameIndexId, AccessShareLock);
	sscan = systable_beginscan_ordered(hrel, irel, NULL, 0, NULL);

	for (;;)
	{
		Oid		trig_tgrelid;
		int		trig_tgtype;

		tup = systable_getnext_ordered(sscan, ForwardScanDirection);
		if (HeapTupleIsValid(tup))
		{
			Form_pg_trigger trig_form = (Form_pg_trigger) GETSTRUCT(tup);

			if (trig_form->tgfoid != invalidator_oid)
				continue;
			if (trig_form->tgenabled)
				continue;

			trig_tgrelid = trig_form->tgrelid;
			trig_tgtype  = trig_form->tgtype;
		}
		else
		{
			trig_tgrelid = InvalidOid;
			trig_tgtype = 0;
		}

		/* switch current focus if any */
		if (OidIsValid(curr_relid) && curr_relid != trig_tgrelid)
		{
			if (has_row_insert &&
				has_row_update &&
				has_row_delete &&
				has_stmt_truncate)
			{
				Assert(curr_relid != InvalidOid);
				relations_oid = list_append_unique_oid(relations_oid,
													   curr_relid);
			}
			curr_relid = trig_tgrelid;
			has_row_insert = false;
			has_row_update = false;
			has_row_delete = false;
			has_stmt_truncate = false;
		}
		if (!HeapTupleIsValid(tup))
			break;

		/* is invalidator configured correctly? */
		if (TRIGGER_FOR_AFTER(trig_tgtype))
		{
			if (TRIGGER_FOR_ROW(trig_tgtype))
			{
				if (TRIGGER_FOR_INSERT(trig_tgtype))
					has_row_insert = true;
				if (TRIGGER_FOR_UPDATE(trig_tgtype))
					has_row_update = true;
				if (TRIGGER_FOR_DELETE(trig_tgtype))
					has_row_delete = true;
			}
			else
			{
				if (TRIGGER_FOR_TRUNCATE(trig_tgtype))
					has_stmt_truncate = true;
			}
		}
	}
	systable_endscan_ordered(sscan);
	index_close(irel, AccessShareLock);
	heap_close(hrel, AccessShareLock);

	len = offsetof(oidvector, values[list_length(relations_oid)]);
	ccache_relations_oid = MemoryContextAlloc(CacheMemoryContext, len);
	SET_VARSIZE(ccache_relations_oid, len);
	ccache_relations_oid->ndim = 1;
	ccache_relations_oid->dataoffset = 0;
	ccache_relations_oid->elemtype = OIDOID;
	ccache_relations_oid->dim1 = list_length(relations_oid);
	ccache_relations_oid->lbound1 = 0;
	i = 0;
	foreach (lc, relations_oid)
		ccache_relations_oid->values[i++] = lfirst_oid(lc);
	list_free(relations_oid);
}

/*
 * ccache_callback_on_reloid - catcache callback on RELOID
 */
static void
ccache_callback_on_reloid(Datum arg, int cacheid, uint32 hashvalue)
{
	Assert(cacheid == RELOID);

	if (ccache_relations_oid)
		pfree(ccache_relations_oid);
	ccache_relations_oid = NULL;
}

/*
 * ccache_callback_on_procoid - catcache callback on PROCOID
 */
static void
ccache_callback_on_procoid(Datum arg, int cacheid, uint32 hashvalue)
{
	Assert(cacheid == PROCOID);

	ccache_invalidator_func_oid = InvalidOid;
}









/*
 * pgstrom_ccache_invalidator
 */
Datum
pgstrom_ccache_invalidator(PG_FUNCTION_ARGS)
{
	TriggerData	   *trigdata = (TriggerData *) fcinfo->context;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as trigger", __FUNCTION__);
	if (!TRIGGER_FIRED_AFTER(trigdata->tg_event))
		elog(ERROR, "%s: must be configured as AFTER trigger", __FUNCTION__);
	if (TRIGGER_FIRED_FOR_ROW(trigdata->tg_event))
	{
		Relation	rel = trigdata->tg_relation;
		HeapTuple	tuple = trigdata->tg_trigtuple;
		BlockNumber	block_nr;
		pg_crc32	hash;
		int			index;
		dlist_iter	iter;

		if (!TRIGGER_FIRED_BY_INSERT(trigdata->tg_event) &&
			!TRIGGER_FIRED_BY_DELETE(trigdata->tg_event) &&
			!TRIGGER_FIRED_BY_UPDATE(trigdata->tg_event))
			elog(ERROR, "%s: triggered by unknown event", __FUNCTION__);

		block_nr = BlockIdGetBlockNumber(&tuple->t_self.ip_blkid);
		block_nr &= ~((CCACHE_CHUNK_SIZE / BLCKSZ) - 1);

		hash = ccache_compute_hashvalue(MyDatabaseId,
										RelationGetRelid(rel),
										block_nr);
		index = hash % ccache_num_slots;
		SpinLockAcquire(&ccache_state->active_locks[index]);
		dlist_foreach(iter, &ccache_state->active_slots[index])
		{
			ccacheChunk *temp = dlist_container(ccacheChunk,
												chain, iter.cur);
			if (temp->hash == hash &&
				temp->database_oid == MyDatabaseId &&
				temp->table_oid == RelationGetRelid(rel) &&
				temp->block_nr == block_nr)
			{
				dlist_delete(&temp->chain);
				memset(&temp->chain, 0, sizeof(dlist_node));
				ccache_put_chunk_nolock(temp);
				break;
			}
		}
		SpinLockRelease(&ccache_state->active_locks[index]);
	}
	else
	{
		Relation	rel = trigdata->tg_relation;
		int			index;
		dlist_mutable_iter iter;

		if (!TRIGGER_FIRED_BY_TRUNCATE(trigdata->tg_event))
			elog(ERROR, "%s: triggered by unknown event", __FUNCTION__);

		for (index=0; index < ccache_num_slots; index++)
		{
			SpinLockAcquire(&ccache_state->active_locks[index]);
			dlist_foreach_modify(iter, &ccache_state->active_slots[index])
			{
				ccacheChunk *temp = dlist_container(ccacheChunk,
													chain, iter.cur);
				if (temp->database_oid == MyDatabaseId &&
					temp->table_oid == RelationGetRelid(rel))
				{
					dlist_delete(&temp->chain);
					memset(&temp->chain, 0, sizeof(dlist_node));
					ccache_put_chunk_nolock(temp);
				}
			}
			SpinLockRelease(&ccache_state->active_locks[index]);
		}
	}
	return PointerGetDatum(NULL);
}
PG_FUNCTION_INFO_V1(pgstrom_ccache_invalidator);

/*
 * ccache_builder_sigterm
 */
static void
ccache_builder_sigterm(SIGNAL_ARGS)
{
	int		saved_errno = errno;

	ccache_builder_got_sigterm = true;

	pg_memory_barrier();

    SetLatch(MyLatch);

	errno = saved_errno;
}

/*
 * ccache_builder_sighup
 */
static void
ccache_builder_sighup(SIGNAL_ARGS)
{
	SetLatch(MyLatch);
}

/*
 * ccache_builder_connectdb
 */
static uint32
ccache_builder_connectdb(cl_int builder_id)
{
	int		i, j, ev;
	uint32	generation;
	char	dbname[NAMEDATALEN];
	bool	startup_log = false;

	/*
	 * Pick up a database to connect
	 */
	for (;;)
	{
		ResetLatch(MyLatch);

		if (ccache_builder_got_sigterm)
			elog(ERROR, "terminating ccache builder%d", builder_id);

		SpinLockAcquire(&ccache_state->lock);
		for (i=0; i < ccache_state->num_databases; i++)
		{
			if (!ccache_state->databases[i].invalid_database)
				break;
		}

		if (i < ccache_state->num_databases)
		{
			/* any valid databases are configured */
			j = (ccache_state->rr_count++ %
				 ccache_state->num_databases);
			if (ccache_state->databases[j].invalid_database)
			{
				SpinLockRelease(&ccache_state->lock);
				continue;
			}
			ccache_database = &ccache_state->databases[j];
			strncpy(dbname, ccache_database->dbname, NAMEDATALEN);
			generation = pg_atomic_read_u32(&ccache_state->generation);
			SpinLockRelease(&ccache_state->lock);
			break;
		}
		else
		{
			/* no valid databases are configured right now */
			SpinLockRelease(&ccache_state->lock);
			if (!startup_log)
			{
				elog(LOG, "ccache builder%d is now started but not assigned to a particular database", builder_id);
				startup_log = true;
			}

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   60000L);
			if (ev & WL_POSTMASTER_DEATH)
				elog(FATAL, "Unexpected postmaster dead");
		}
	}

	/*
	 * Try to connect database
	 */
	PG_TRY();
	{
		//XXX - How to catch FATAL error and invalidate database?
		BackgroundWorkerInitializeConnection(dbname, NULL);
	}
	PG_CATCH();
	{
		/* remove database entry from pg_strom.ccache_databases */
		SpinLockAcquire(&ccache_state->lock);
		ccache_database->invalid_database = true;
		SpinLockRelease(&ccache_state->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	elog(LOG, "ccache builder%d (gen=%u) now ready on database \"%s\"",
		 builder_id, generation, dbname);
	return generation;
}

/*
 * ccache_builder_startup
 *
 * Filesystem may still keep valid ccache-chunks. Reload it on startup time.
 */
static int
ccache_builder_startup(cl_long *timeout)
{


	return CCBUILDER_STATE__PRELOAD;
}

/*
 * ccache_builder_preload
 *
 * Load target relations to empty chunks.
 */
static int
ccache_builder_preload(cl_long *timeout)
{
	return CCBUILDER_STATE__TRYLOAD;
}

/*
 * ccache_builder_tryload
 */
static int
ccache_builder_tryload(cl_long *timeout)
{

	*timeout = 5000L;

	return CCBUILDER_STATE__TRYLOAD;
}

/*
 * ccache_builder_main 
 */
void
ccache_builder_main(Datum arg)
{
	cl_int		builder_id = DatumGetInt32(arg);
	uint32		generation;
	int			curr_state;
	int			next_state;
	int			ev;

	pqsignal(SIGTERM, ccache_builder_sigterm);
	pqsignal(SIGHUP, ccache_builder_sighup);
	BackgroundWorkerUnblockSignals();

	CurrentResourceOwner = ResourceOwnerCreate(NULL, "CCache Builder");
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "CCache Builder Context",
												 ALLOCSET_DEFAULT_SIZES);
	ccache_builder = &ccache_state->builders[builder_id];
	SpinLockAcquire(&ccache_state->lock);
	ccache_builder->database_oid = InvalidOid;
	ccache_builder->state = curr_state = CCBUILDER_STATE__STARTUP;
	ccache_builder->latch = MyLatch;
	SpinLockRelease(&ccache_state->lock);

	PG_TRY();
	{
		/* connect to one of the databases */
		generation = ccache_builder_connectdb(builder_id);

		for (;;)
		{
			long		timeout = 0L;

			if (ccache_builder_got_sigterm)
				elog(ERROR, "terminating ccache builder%d", builder_id);
			ResetLatch(MyLatch);

			/* pg_strom.ccache_databases updated? */
			if (generation != pg_atomic_read_u32(&ccache_state->generation))
				elog(ERROR,"restarting ccache builder%d", builder_id);

			/*
			 * ---------------------
			 *   BEGIN Transaction
			 * ---------------------
			 */
			SetCurrentStatementStartTimestamp();
			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());

			/* setup oidvector of relations to be cached */
			refresh_ccache_source_relations();

			switch (curr_state)
			{
				case CCBUILDER_STATE__STARTUP:
					next_state = ccache_builder_startup(&timeout);
					break;

				case CCBUILDER_STATE__PRELOAD:
					next_state = ccache_builder_preload(&timeout);
					break;

				case CCBUILDER_STATE__TRYLOAD:
					next_state = ccache_builder_tryload(&timeout);
					break;

				default:
					break;
			}

			/*
			 * -------------------
			 *   END Transaction
			 * -------------------
			 */
			PopActiveSnapshot();
			CommitTransactionCommand();

			SpinLockAcquire(&ccache_state->lock);
			ccache_builder->state = CCBUILDER_STATE__SLEEP;
			SpinLockRelease(&ccache_state->lock);

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   timeout);
			if (ev & WL_POSTMASTER_DEATH)
				elog(FATAL, "Unexpected postmaster dead");

			SpinLockAcquire(&ccache_state->lock);
			ccache_builder->state = curr_state = next_state;
			SpinLockRelease(&ccache_state->lock);
		}
	}
	PG_CATCH();
	{
		SpinLockAcquire(&ccache_state->lock);
		ccache_builder->database_oid = InvalidOid;
		ccache_builder->state = CCBUILDER_STATE__SHUTDOWN;
		ccache_builder->latch = NULL;
		SpinLockRelease(&ccache_state->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	elog(FATAL, "Bug? ccache builder%d should not exit normaly", builder_id);
}

/*
 * GUC callbacks for pg_strom.ccache_databases
 */
static bool
guc_check_ccache_databases(char **newval, void **extra, GucSource source)
{
	char	   *rawnames = pstrdup(*newval);
	List	   *options;
	ListCell   *lc1, *lc2;
	ccacheDatabase *my_extra;
	int			i;

	/* Parse string into list of identifiers */
	if (!SplitIdentifierString(rawnames, ',', &options))
	{
		/* syntax error in name list */
		GUC_check_errdetail("List syntax is invalid.");
		pfree(rawnames);
		list_free(options);
		return false;
	}

	foreach (lc1, options)
	{
		const char   *dbname = lfirst(lc1);

		if (strlen(dbname) >= NAMEDATALEN)
			elog(ERROR, "too long database name: \"%s\"", dbname);
		/* check existence if under transaction */
		if (IsTransactionState())
			get_database_oid(dbname, false);
		/* duplication check */
		foreach (lc2, options)
		{
			if (lc1 == lc2)
				break;
			if (strcmp(dbname, lfirst(lc2)) == 0)
				elog(ERROR, "database \"%s\" appeared twice", dbname);
		}
	}

	if (list_length(options) > CCACHE_MAX_NUM_DATABASES)
		elog(ERROR, "pg_strom.ccache_databases configured too much databases");
	if (list_length(options) > ccache_num_builders)
		elog(ERROR, "number of the configured databases by pg_strom.ccache_databases is larger than number of the builder processes by pg_strom.ccache_num_builders, so columnar cache will be never built on some databases");

	my_extra = calloc(list_length(options) + 1, sizeof(ccacheDatabase));
	if (!my_extra)
		elog(ERROR, "out of memory");
	i = 0;
	foreach (lc1, options)
	{
		strncpy(my_extra[i].dbname, lfirst(lc1), NAMEDATALEN);
		i++;
	}
	my_extra[i].invalid_database = true;

	*extra = my_extra;

	return true;
}

static void
guc_assign_ccache_databases(const char *newval, void *extra)
{
	ccacheDatabase *my_extra = extra;

	if (ccache_state)
	{
		int		i = 0;

		SpinLockAcquire(&ccache_state->lock);
		while (!my_extra[i].invalid_database)
		{
			Assert(i < CCACHE_MAX_NUM_DATABASES);
			strncpy(ccache_state->databases[i].dbname,
					my_extra[i].dbname,
					NAMEDATALEN);
			ccache_state->databases[i].invalid_database = false;
			i++;
		}
		ccache_state->num_databases = i;

		/* force to restart ccache builder */
		pg_atomic_fetch_add_u32(&ccache_state->generation, 1);
		for (i=0; i < ccache_num_builders; i++)
		{
			if (ccache_state->builders[i].latch)
				SetLatch(ccache_state->builders[i].latch);
		}
		SpinLockRelease(&ccache_state->lock);
	}
}

static const char *
guc_show_ccache_databases(void)
{
	StringInfoData str;
	int		i;

	initStringInfo(&str);
	SpinLockAcquire(&ccache_state->lock);
	PG_TRY();
	{
		for (i=0; i < ccache_state->num_databases; i++)
		{
			const char *dbname = ccache_state->databases[i].dbname;

			if (!ccache_state->databases[i].invalid_database)
				appendStringInfo(&str, "%s%s",
								 str.len > 0 ? "," : "",
								 quote_identifier(dbname));
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&ccache_state->lock);
	}
	PG_END_TRY();
	SpinLockRelease(&ccache_state->lock);

	return str.data;
}

/*
 * pgstrom_startup_ccache
 */
static void
pgstrom_startup_ccache(void)
{
	ccacheChunk *cc_chunk;
	size_t		required;
	bool		found;
	int			i;
	void	   *extra = NULL;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	required = MAXALIGN(offsetof(ccacheState,
								 builders[ccache_num_builders])) +
		MAXALIGN(sizeof(slock_t) * ccache_num_slots) +
		MAXALIGN(sizeof(dlist_head) * ccache_num_slots) +
		MAXALIGN(offsetof(ccacheChunk,
						  attrs[ccache_max_nattrs+1])) * ccache_num_chunks;
	ccache_state = ShmemInitStruct("Columnar Cache Shared Segment",
								   required, &found);
	if (found)
		elog(ERROR, "Bug? Columnar Cache Shared Segment is already built");
	memset(ccache_state, 0, required);

	ccache_state->active_locks = (slock_t *)
		((char *)ccache_state +
		 MAXALIGN(offsetof(ccacheState, builders[ccache_num_builders])));
	ccache_state->active_slots = (dlist_head *)
		((char *)ccache_state->active_locks +
		 MAXALIGN(sizeof(slock_t) * ccache_num_slots));
	/* hash slot of ccache chunks */
	SpinLockInit(&ccache_state->free_chunks_lock);
	dlist_init(&ccache_state->free_chunks_list);
	for (i=0; i < ccache_num_slots; i++)
	{
		SpinLockInit(&ccache_state->active_locks[i]);
		dlist_init(&ccache_state->active_slots[i]);
	}
	/* ccache-chunks */
	cc_chunk = (ccacheChunk *)
		((char *)ccache_state->active_slots +
		 MAXALIGN(sizeof(dlist_head) * ccache_num_slots));
	for (i=0; i < ccache_num_chunks; i++)
	{
		dlist_push_tail(&ccache_state->free_chunks_list,
						&cc_chunk->chain);
		cc_chunk = (ccacheChunk *)
			((char *)cc_chunk +
			 MAXALIGN(offsetof(ccacheChunk, attrs[ccache_max_nattrs+1])));
	}
	/* fields for management of builder processes */
	SpinLockInit(&ccache_state->lock);

	/* setup GUC again */
	if (!guc_check_ccache_databases(&ccache_startup_databases,
									&extra, PGC_S_DEFAULT))
		elog(ERROR, "Bug? failed on parse pg_strom.ccache_databases");
	guc_assign_ccache_databases(ccache_startup_databases, extra);
}

/*
 * pgstrom_init_ccache
 */
void
pgstrom_init_ccache(void)
{
	static int	ccache_max_size_kb;
	long		sc_pagesize = sysconf(_SC_PAGESIZE);
	long		sc_phys_pages = sysconf(_SC_PHYS_PAGES);
	size_t		required = 0;
	BackgroundWorker worker;
	char		pathname[MAXPGPATH];
	int			i;

	DefineCustomStringVariable("pg_strom.ccache_databases",
							   "databases where ccache builder works on",
							   NULL,
							   &ccache_startup_databases,
							   "",
							   PGC_SUSET,
							   GUC_NOT_IN_SAMPLE,
							   guc_check_ccache_databases,
							   guc_assign_ccache_databases,
							   guc_show_ccache_databases);
	DefineCustomIntVariable("pg_strom.ccache_num_builders",
							"number of ccache builder worker processes",
							NULL,
							&ccache_num_builders,
							2,
							0,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.ccache_max_size",
							"possible maximum allocation of ccache",
							NULL,
							&ccache_max_size_kb,
							sc_phys_pages * (sc_pagesize >> 10),
							0,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_UNIT_KB,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.ccache_max_nattrs",
							"maximum number of attributes ccache can keep",
							NULL,
							&ccache_max_nattrs,
							40,
							20,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);

	ccache_max_size = (size_t)ccache_max_size_kb << 10;
	ccache_num_slots = Max(ccache_max_size / CCACHE_CHUNK_SIZE, 300);
	ccache_num_chunks = 3 * ccache_num_slots;

	DefineCustomStringVariable("pg_strom.ccache_base_dir",
							   "directory name used by ccache",
							   NULL,
							   &ccache_base_dir_name,
							   "/dev/shm",
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	snprintf(pathname, sizeof(pathname), "%s/.pg_strom.ccache.%u",
			 ccache_base_dir_name, PostPortNumber);
	ccache_base_dir = opendir(pathname);
	if (!ccache_base_dir)
	{
		if (errno != ENOENT)
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not open ccache directory \"%s\": %m",
							pathname)));
		else if (ccache_num_builders > 0)
		{
			/*
			 * Even if ccache directory is not found, we try to make
			 * an empty directory if ccache builder process will run.
			 */
			if (mkdir(pathname, 0700) != 0)
				ereport(ERROR,
						(errcode_for_file_access(),
						 errmsg("could not make a ccache directory \"%s\": %m",
								pathname)));
			ccache_base_dir = opendir(pathname);
			if (!ccache_base_dir)
				ereport(ERROR,
						(errcode_for_file_access(),
						 errmsg("could not open ccache directory \"%s\": %m",
								pathname)));
		}
	}

	/* bgworker registration */
	for (i=0; i < ccache_num_builders; i++)
	{
		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "PG-Strom CCache Builder[%d]", i+1);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS
			| BGWORKER_BACKEND_DATABASE_CONNECTION;
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = 2;
		snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_strom");
		snprintf(worker.bgw_function_name, BGW_MAXLEN, "ccache_builder_main");
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* request for static shared memory */
	required = MAXALIGN(offsetof(ccacheState,
								 builders[ccache_num_builders])) +
		MAXALIGN(sizeof(slock_t) * ccache_num_slots) +
		MAXALIGN(sizeof(dlist_head) * ccache_num_slots) +
		MAXALIGN(offsetof(ccacheChunk,
						  attrs[ccache_max_nattrs+1])) * ccache_num_chunks;
	RequestAddinShmemSpace(required);

	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_ccache;

	CacheRegisterSyscacheCallback(RELOID, ccache_callback_on_reloid, 0);
	CacheRegisterSyscacheCallback(PROCOID, ccache_callback_on_procoid, 0);
}
