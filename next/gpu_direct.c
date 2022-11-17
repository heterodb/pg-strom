/*
 * gpu_direct.c
 *
 * Routines to support GPU Direct SQL feature
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

static bool		gpudirect_driver_initialized = false;
static bool		pgstrom_gpudirect_enabled;			/* GUC */
static int		__pgstrom_gpudirect_threshold_kb;		/* GUC */
#define pgstrom_gpudirect_threshold		((size_t)__pgstrom_gpudirect_threshold_kb << 10)
static HTAB	   *tablespace_optimal_gpu_htable = NULL;

typedef struct
{
	Oid			tablespace_oid;
	bool		is_valid;
	Bitmapset	optimal_gpus;
} tablespace_optimal_gpu_hentry;

static void
tablespace_optimal_gpu_cache_callback(Datum arg, int cacheid, uint32 hashvalue)
{
	/* invalidate all the cached status */
	if (tablespace_optimal_gpu_htable)
	{
		hash_destroy(tablespace_optimal_gpu_htable);
		tablespace_optimal_gpu_htable = NULL;
	}
}

/*
 * GetOptimalGpusForTablespace
 */
static const Bitmapset *
GetOptimalGpusForTablespace(Oid tablespace_oid)
{
	tablespace_optimal_gpu_hentry *hentry;
	bool		found;

	if (!pgstrom_gpudirect_enabled)
		return NULL;

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	if (!tablespace_optimal_gpu_htable)
	{
		HASHCTL		hctl;
		int			nwords = (numGpuDevAttrs / BITS_PER_BITMAPWORD) + 1;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = MAXALIGN(offsetof(tablespace_optimal_gpu_hentry,
										   optimal_gpus.words[nwords]));
		tablespace_optimal_gpu_htable
			= hash_create("TablespaceOptimalGpu", 128,
						  &hctl, HASH_ELEM | HASH_BLOBS);
	}

	hentry = (tablespace_optimal_gpu_hentry *)
		hash_search(tablespace_optimal_gpu_htable,
					&tablespace_oid,
					HASH_ENTER,
					&found);
	if (!found || !hentry->is_valid)
	{
		char	   *pathname;
		File		filp;
		Bitmapset  *optimal_gpus;

		Assert(hentry->tablespace_oid == tablespace_oid);

		pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);
		filp = PathNameOpenFile(pathname, O_RDONLY);
		if (filp < 0)
		{
			elog(WARNING, "failed on open('%s') of tablespace %u: %m",
				 pathname, tablespace_oid);
			return NULL;
		}
		optimal_gpus = extraSysfsLookupOptimalGpus(filp);
		if (!optimal_gpus)
			hentry->optimal_gpus.nwords = 0;
		else
		{
			Assert(optimal_gpus->nwords <= (numGpuDevAttrs/BITS_PER_BITMAPWORD)+1);
			memcpy(&hentry->optimal_gpus, optimal_gpus,
				   offsetof(Bitmapset, words[optimal_gpus->nwords]));
			bms_free(optimal_gpus);
		}
		FileClose(filp);
		hentry->is_valid = true;
	}
	Assert(hentry->is_valid);
	return (hentry->optimal_gpus.nwords > 0 ? &hentry->optimal_gpus : NULL);
}

/*
 * baseRelCanUseGpuDirect - checks wthere the relation can use GPU-Direct SQL.
 * If possible, it returns a bitmap of optimal GPUs.
 */
const Bitmapset *
baseRelCanUseGpuDirect(PlannerInfo *root, RelOptInfo *baserel)
{
	const Bitmapset *optimal_gpus;
	double		total_sz;

	if (!pgstrom_gpudirect_enabled)
		return NULL;
#if 0
	if (baseRelIsArrowFdw(baserel))
	{
		if (pgstrom_gpudirect_enabled)
			return GetOptimalGpusForArrowFdw(root, baserel);
		return NULL;
	}
#endif
	total_sz = (size_t)baserel->pages * (size_t)BLCKSZ;
	if (total_sz < pgstrom_gpudirect_threshold)
		return NULL;	/* table is too small */

	optimal_gpus = GetOptimalGpusForTablespace(baserel->reltablespace);
	if (optimal_gpus)
	{
		RangeTblEntry *rte = root->simple_rte_array[baserel->relid];
		HeapTuple	tup;
		char		relpersistence;

		tup = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "cache lookup failed for relation %u", rte->relid);
		relpersistence = ((Form_pg_class) GETSTRUCT(tup))->relpersistence;
		ReleaseSysCache(tup);

		/* temporary table is not supported by GPU-Direct SQL */
		if (relpersistence != RELPERSISTENCE_PERMANENT &&
			relpersistence != RELPERSISTENCE_UNLOGGED)
			optimal_gpus = NULL;
	}
	return optimal_gpus;
}

/* ----------------------------------------------------------------
 *
 * GPU Direct SQL - executor callbacks
 *
 * ----------------------------------------------------------------
 */
struct GpuDirectState
{
	const Bitmapset *gpuset;
};

/*
 * pgstromGpuDirectExecBegin
 */
void
pgstromGpuDirectExecBegin(pgstromTaskState *pts, const Bitmapset *gpuset)
{
	if (!bms_is_empty(gpuset))
	{
		GpuDirectState *gd_state = palloc(sizeof(GpuDirectState));

		gd_state->gpuset = bms_copy(gpuset);

		pts->gd_state = gd_state;
	}
}

const Bitmapset *
pgstromGpuDirectDevices(pgstromTaskState *pts)
{
	if (pts->gd_state)
	{
		GpuDirectState *gd_state = pts->gd_state;

		return gd_state->gpuset;
	}
	return NULL;
}

/*
 * pgstromGpuDirectExecEnd
 */
void
pgstromGpuDirectExecEnd(pgstromTaskState *pts)
{
	/* do nothing */
}

/*
 * pgstrom_gpudirect_enabled_checker
 */
static bool
pgstrom_gpudirect_enabled_checker(bool *p_newval, void **extra, GucSource source)
{
	bool		newval = *p_newval;

	if (newval && !gpudirect_driver_initialized)
		elog(ERROR, "cannot enables GPU-Direct SQL without driver module loaded");
	return true;
}

/*
 * pgstrom_init_gpu_direct
 */
void
pgstrom_init_gpu_direct(void)
{
	static char *nvme_manual_distance_map = NULL;
	bool		default_gpudirect_enabled = true;
	char		buffer[1280];
	int			i;

	/*
	 * pgstrom.gpudirect_enabled
	 */
	if (gpuDirectInitDriver() == 0)
	{
		for (i=0; i < numGpuDevAttrs; i++)
		{
			if (gpuDevAttrs[i].DEV_SUPPORT_GPUDIRECTSQL)
				default_gpudirect_enabled = true;
		}
		gpudirect_driver_initialized = true;
	}
	DefineCustomBoolVariable("pg_strom.gpudirect_enabled",
							 "enables GPUDirect SQL",
							 NULL,
							 &pgstrom_gpudirect_enabled,
							 default_gpudirect_enabled,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 pgstrom_gpudirect_enabled_checker, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpudirect_threshold",
							"table-size threshold to use GPU-Direct SQL",
							NULL,
							&__pgstrom_gpudirect_threshold_kb,
							2097152,	/* 2GB */
							0,
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	/*
	 * pg_strom.nvme_distance_map
	 *
	 * config := <token>[,<token>...]
	 * token  := nvmeXX:gpuXX
	 *
	 * eg) nvme0:gpu0,nvme1:gpu1
	 */
	DefineCustomStringVariable("pg_strom.nvme_distance_map",
							   "Manual configuration of optimal GPU for each NVME",
							   NULL,
							   &nvme_manual_distance_map,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	extraSysfsSetupDistanceMap(nvme_manual_distance_map);
	for (i=0; extraSysfsPrintNvmeInfo(i, buffer, sizeof(buffer)) >= 0; i++)
		elog(LOG, "- %s", buffer);

	/* hash table for tablespace <-> optimal GPU */
	tablespace_optimal_gpu_htable = NULL;
	CacheRegisterSyscacheCallback(TABLESPACEOID,
								  tablespace_optimal_gpu_cache_callback,
								  (Datum) 0);

}
