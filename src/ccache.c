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

typedef struct
{
	Latch	   *latch;
} ccacheBuilder;

#define CCACHE_MAX_NUM_DATABASES	100
typedef struct
{
	pg_atomic_uint32 generation;
	slock_t		lock;
	int			rr_count;
	int			num_databases;
	char		dbnames[CCACHE_MAX_NUM_DATABASES][NAMEDATALEN];
	ccacheBuilder cc_builders[FLEXIBLE_ARRAY_MEMBER];
} ccacheBuilderControl;

static ccacheBuilderControl	   *cc_builder_control = NULL;

/* static variables */
static shmem_startup_hook_type shmem_startup_next = NULL;
static char		   *ccache_startup_databases;	/* GUC */
static int			ccache_num_builders;		/* GUC */
static size_t		ccache_max_size;			/* GUC */
static bool			ccache_builder_got_sigterm = false;
static int			ccache_builder_id = -1;
static HTAB		   *ccache_relations_htab = NULL;
static MemoryContext ccache_relations_mcxt;

static Oid			ccache_invalidator_func_oid = InvalidOid;

extern void ccache_builder_main(Datum arg);

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

	ccache_invalidator_func_oid = invalidator_oid;
	return ccache_invalidator_func_oid;
}

/*
 * refresh_ccache_relations_htab
 */
static void
refresh_ccache_relations_htab(void)
{
	Oid			invalidator_oid;
	Relation	hrel;
	Relation	irel;
	SysScanDesc	sscan;
	HeapTuple	tup;
	Form_pg_trigger trig_form;
	HTAB	   *htab;
	HASHCTL		hctl;
	bool		found;
	Oid			curr_relid = InvalidOid;
	int			curr_config = 0;
#define MASK_CONFIG		(TRIGGER_TYPE_TRUNCATE | TRIGGER_TYPE_INSERT | \
						 TRIGGER_TYPE_UPDATE | TRIGGER_TYPE_DELETE)
	/* already built? */
	if (ccache_relations_htab)
		return;
	/* invalidator function installed? */
	invalidator_oid = ccache_invalidator_oid(true);
	if (!OidIsValid(invalidator_oid))
		return;
	/* construct a hash table */
	hctl.keysize = sizeof(Oid);
	hctl.entrysize = sizeof(Oid);	//other information?
	hctl.hcxt = ccache_relations_mcxt;
	htab = hash_create("CCache Relations HTAB",
					   256,
					   &hctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	/* walk on the pg_trigger catalog */
	hrel = heap_open(TriggerRelationId, AccessShareLock);
	irel = index_open(TriggerRelidNameIndexId, AccessShareLock);
	sscan = systable_beginscan_ordered(hrel, irel, NULL, 0, NULL);

	for (;;)
	{
		tup = systable_getnext_ordered(sscan, ForwardScanDirection);
		if (!HeapTupleIsValid(tup))
			break;
		trig_form = (Form_pg_trigger) GETSTRUCT(tup);
		/* related trigger? */
		if (trig_form->tgfoid != invalidator_oid)
			continue;

		if (!trig_form->tgenabled ||
			!TRIGGER_FOR_AFTER(trig_form->tgtype))
			continue;

		if (curr_relid != trig_form->tgrelid)
		{
			if ((curr_config & MASK_CONFIG) == MASK_CONFIG)
			{
				hash_search(htab,
							&curr_relid,
							HASH_ENTER,
							&found);
				if (found)
					elog(ERROR, "Bug? relation %u appears twice on ccache",
						 curr_relid);
				elog(LOG, "ccache builder%d: added relation \"%s\"",
					 ccache_builder_id, get_rel_name(curr_relid));
			}
			curr_relid = trig_form->tgrelid;
			curr_config = 0;
		}

		if (TRIGGER_FOR_ROW(trig_form->tgtype))
		{
			if (TRIGGER_FOR_INSERT(trig_form->tgtype))
				curr_config |= TRIGGER_TYPE_INSERT;
			if (TRIGGER_FOR_UPDATE(trig_form->tgtype))
				curr_config |= TRIGGER_TYPE_UPDATE;
			if (TRIGGER_FOR_DELETE(trig_form->tgtype))
				curr_config |= TRIGGER_TYPE_DELETE;
		}
		else
		{
			if (TRIGGER_FOR_TRUNCATE(trig_form->tgtype))
				curr_config |= TRIGGER_TYPE_TRUNCATE;
		}
	}
	/* last one entry if any */
	if ((curr_config & MASK_CONFIG) == MASK_CONFIG)
	{
		hash_search(htab,
					&curr_relid,
					HASH_ENTER,
					&found);
		if (found)
			elog(ERROR, "Bug? relation %u appears twice on ccache",
				 curr_relid);
		elog(LOG, "ccache builder%d: added relation \"%s\"",
			 ccache_builder_id, get_rel_name(curr_relid));
	}
	systable_endscan_ordered(sscan);
	index_close(irel, AccessShareLock);
	heap_close(hrel, AccessShareLock);

	ccache_relations_htab = htab;
}

/*
 * ccache_callback_on_reloid - catcache callback on RELOID
 */
static void
ccache_callback_on_reloid(Datum arg, int cacheid, uint32 hashvalue)
{
	Assert(cacheid == RELOID);

	MemoryContextReset(ccache_relations_mcxt);
	ccache_relations_htab = NULL;
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
	TriggerData *trigdata = (TriggerData *) fcinfo->context;

	//memo: see suppress_redundant_updates_trigger

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
 * ccache_builder_main 
 */
void
ccache_builder_main(Datum arg)
{
	char	dbname[NAMEDATALEN];
	uint32	generation;
	int		i, j, ev;
	bool	startup_log = false;

	ccache_builder_id = DatumGetInt32(arg);
	pqsignal(SIGTERM, ccache_builder_sigterm);
	pqsignal(SIGHUP, ccache_builder_sighup);
	BackgroundWorkerUnblockSignals();

	CurrentResourceOwner = ResourceOwnerCreate(NULL, "CCache Builder");
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "CCache Builder Context",
												 ALLOCSET_DEFAULT_SIZES);
	SpinLockAcquire(&cc_builder_control->lock);
	cc_builder_control->cc_builders[ccache_builder_id].latch = MyLatch;
	SpinLockRelease(&cc_builder_control->lock);

	PG_TRY();
	{
		/*
		 * Pick up a database to connect
		 */
		for (;;)
		{
			if (ccache_builder_got_sigterm)
				elog(ERROR, "terminating ccache builder%d", ccache_builder_id);

			SpinLockAcquire(&cc_builder_control->lock);
			if (cc_builder_control->num_databases == 0)
			{
				SpinLockRelease(&cc_builder_control->lock);
				if (!startup_log)
				{
					elog(LOG, "ccache builder%d is now started but not assigned to a particular database", ccache_builder_id);
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
			else
			{
				i = (cc_builder_control->rr_count++ %
					 cc_builder_control->num_databases);
				strcpy(dbname, cc_builder_control->dbnames[i]);
				generation = pg_atomic_read_u32(&cc_builder_control->generation);
				SpinLockRelease(&cc_builder_control->lock);

				break;
			}
		}

		/*
		 * Try to connect database
		 */
		PG_TRY();
		{
			BackgroundWorkerInitializeConnection(dbname, NULL);
		}
		PG_CATCH();
		{
			/* remove database entry from pg_strom.ccache_databases */
			SpinLockAcquire(&cc_builder_control->lock);
			for (i=0; i < cc_builder_control->num_databases; i++)
			{
				if (strcmp(cc_builder_control->dbnames[i], dbname) == 0)
				{
					for (j=i+1; j < cc_builder_control->num_databases; j++)
					{
						strcpy(cc_builder_control->dbnames[j-1],
							   cc_builder_control->dbnames[j]);
					}
					cc_builder_control->num_databases--;
					break;
				}
			}
			SpinLockRelease(&cc_builder_control->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();

		elog(LOG, "ccache builder%d now ready on database \"%s\"",
			 ccache_builder_id, dbname);

		while (!ccache_builder_got_sigterm)
		{
			ResetLatch(MyLatch);

			/* pg_strom.ccache_databases updated? */
			if (generation !=
				pg_atomic_read_u32(&cc_builder_control->generation))
				elog(ERROR,"restarting ccache builder%d", ccache_builder_id);

			/*
			 * ---------------------
			 *   BEGIN Transaction
			 * ---------------------
			 */
			SetCurrentStatementStartTimestamp();
			StartTransactionCommand();
			PushActiveSnapshot(GetTransactionSnapshot());

			/* setup list of relations to be cached */
			refresh_ccache_relations_htab();






			/*
			 * -------------------
			 *   END Transaction
			 * -------------------
			 */
			PopActiveSnapshot();
			CommitTransactionCommand();

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   5000L);
			if (ev & WL_POSTMASTER_DEATH)
				elog(FATAL, "Unexpected postmaster dead");
		}
	}
	PG_CATCH();
	{
		SpinLockAcquire(&cc_builder_control->lock);
		cc_builder_control->cc_builders[ccache_builder_id].latch = MyLatch;
		SpinLockRelease(&cc_builder_control->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* normal termination */
	SpinLockAcquire(&cc_builder_control->lock);
	cc_builder_control->cc_builders[ccache_builder_id].latch = MyLatch;
	SpinLockRelease(&cc_builder_control->lock);
	elog(LOG, "terminating ccache builder%d", ccache_builder_id);
}

/*
 * GUC callbacks for pg_strom.ccache_databases
 */
static bool
guc_check_ccache_databases(char **newval, void **extra, GucSource source)
{
	char	   *rawnames = pstrdup(*newval);
	List	   *options;
	ListCell   *lc;
	ccacheBuilderControl *my_extra;

	my_extra = malloc(offsetof(ccacheBuilderControl, cc_builders));
	if (!my_extra)
		elog(ERROR, "out of memory");
	memset(my_extra, 0, offsetof(ccacheBuilderControl, cc_builders));

	/* Parse string into list of identifiers */
	if (!SplitIdentifierString(rawnames, ',', &options))
	{
		/* syntax error in name list */
		GUC_check_errdetail("List syntax is invalid.");
		pfree(rawnames);
		list_free(options);
		return false;
	}

	PG_TRY();
	{
		foreach (lc, options)
		{
			char   *dbname = lfirst(lc);
			int		i;

			if (strlen(dbname) >= NAMEDATALEN)
				elog(ERROR, "too long database name: \"%s\"", dbname);
			/* check existence if under transaction */
			if (IsTransactionState())
				get_database_oid(dbname, false);
			/* duplication check */
			for (i=0; i < my_extra->num_databases; i++)
			{
				if (strcmp(dbname, my_extra->dbnames[i]) == 0)
					elog(ERROR, "database \"%s\" appeared in pg_strom.ccache_databases twice", dbname);
			}
			strcpy(my_extra->dbnames[my_extra->num_databases], dbname);
			my_extra->num_databases++;

			if (my_extra->num_databases > CCACHE_MAX_NUM_DATABASES)
				elog(ERROR, "pg_strom.ccache_databases specified too much databases");
		}

		if (my_extra->num_databases > ccache_num_builders)
			elog(WARNING, "number of specified databases in pg_strom.ccache_databases are larger than pg_strom.ccache_num_builders, so columnar cache will never build on some databases");
	}
	PG_CATCH();
	{
		free(my_extra);
		PG_RE_THROW();
	}
	PG_END_TRY();
	*extra = my_extra;

	return true;
}

static void
guc_assign_ccache_databases(const char *newval, void *extra)
{
	ccacheBuilderControl *my_extra = extra;
	int			i;

	Assert(my_extra->num_databases <= CCACHE_MAX_NUM_DATABASES);
	if (cc_builder_control)
	{
		/* update cc_builder_control */
		SpinLockAcquire(&cc_builder_control->lock);
		for (i=0; i < my_extra->num_databases; i++)
		{
			strncpy(cc_builder_control->dbnames[i],
					my_extra->dbnames[i],
					NAMEDATALEN);
		}
		cc_builder_control->num_databases = i;
		/* force to restart ccache builder */
		pg_atomic_fetch_add_u32(&cc_builder_control->generation, 1);
		for (i=0; i < ccache_num_builders; i++)
		{
			ccacheBuilder *cc_builder = &cc_builder_control->cc_builders[i];

			if (cc_builder->latch)
				SetLatch(cc_builder->latch);
		}
		SpinLockRelease(&cc_builder_control->lock);
	}
}

static const char *
guc_show_ccache_databases(void)
{
	StringInfoData str;
	int		i;

	initStringInfo(&str);
	SpinLockAcquire(&cc_builder_control->lock);
	PG_TRY();
	{
		for (i=0; i < cc_builder_control->num_databases; i++)
		{
			const char *dbname = cc_builder_control->dbnames[i];
			appendStringInfo(&str, "%s%s",
							 str.len > 0 ? "," : "",
							 quote_identifier(dbname));
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&cc_builder_control->lock);
	}
	PG_END_TRY();
	SpinLockRelease(&cc_builder_control->lock);

	return str.data;
}

/*
 * pgstrom_startup_ccache
 */
static void
pgstrom_startup_ccache(void)
{
	size_t		required;
	bool		found;
	void	   *extra = NULL;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	required = offsetof(ccacheBuilderControl,
						cc_builders[ccache_num_builders]);
	cc_builder_control = ShmemInitStruct("CCache Builder Control Segment",
										 required, &found);
	if (found)
		elog(ERROR, "Bug? shared memory for ccache builders already built");

	memset(cc_builder_control, 0, required);
	SpinLockInit(&cc_builder_control->lock);

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
	size_t		required;
	BackgroundWorker worker;
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
	ccache_max_size = (size_t)ccache_max_size_kb << 10;

	ccache_relations_mcxt = AllocSetContextCreate(CacheMemoryContext,
												  "HTAB of ccache relations",
												  ALLOCSET_DEFAULT_SIZES);
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
	required = MAXALIGN(offsetof(ccacheBuilderControl,
								 cc_builders[ccache_num_builders]));
	RequestAddinShmemSpace(required);

	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_ccache;

	CacheRegisterSyscacheCallback(RELOID, ccache_callback_on_reloid, 0);
	CacheRegisterSyscacheCallback(PROCOID, ccache_callback_on_procoid, 0);
}
