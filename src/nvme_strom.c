/*
 * nvme_strom.c
 *
 * Routines related to NVME-SSD devices
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

/* static variables */
static char	   *nvme_manual_distance_map;	/* GUC */

/*
 * TablespaceCanUseNvmeStrom
 */
typedef struct
{
	Oid		tablespace_oid;
	int		optimal_gpu;
} tablespace_optimal_gpu_hentry;

typedef struct
{
	dev_t	st_dev;		/* may be a partition device */
	int		optimal_gpu;
} filesystem_optimal_gpu_hentry;

static HTAB	   *tablespace_optimal_gpu_htable = NULL;

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
 * GetOptimalGpuForFile
 */
int
GetOptimalGpuForFile(File fdesc)
{
	struct stat	stat_buf;

	if (fstat(FileGetRawDesc(fdesc), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", FilePathName(fdesc));

	return extraSysfsLookupOptimalGpu(stat_buf.st_dev);
}

static cl_int
GetOptimalGpuForTablespace(Oid tablespace_oid)
{
	tablespace_optimal_gpu_hentry *hentry;
	bool		found;

	if (!pgstrom_gpudirect_enabled())
		return -1;

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	if (!tablespace_optimal_gpu_htable)
	{
		HASHCTL		ctl;

		memset(&ctl, 0, sizeof(HASHCTL));
		ctl.keysize = sizeof(Oid);
		ctl.entrysize = sizeof(tablespace_optimal_gpu_hentry);
		tablespace_optimal_gpu_htable
			= hash_create("TablespaceOptimalGpu", 128,
						  &ctl, HASH_ELEM | HASH_BLOBS);
		CacheRegisterSyscacheCallback(TABLESPACEOID,
									  tablespace_optimal_gpu_cache_callback,
									  (Datum) 0);
	}
	hentry = (tablespace_optimal_gpu_hentry *)
		hash_search(tablespace_optimal_gpu_htable,
					&tablespace_oid,
					HASH_ENTER,
					&found);
	if (!found)
	{
		PG_TRY();
		{
			char	   *pathname;
			struct stat	stat_buf;

			Assert(hentry->tablespace_oid == tablespace_oid);
			hentry->optimal_gpu = -1;

			pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);
			if (stat(pathname, &stat_buf) != 0)
			{
				elog(WARNING, "failed on stat('%s') of tablespace %u: %m",
					 pathname, tablespace_oid);
			}
			else
			{
				hentry->optimal_gpu = extraSysfsLookupOptimalGpu(stat_buf.st_dev);
			}
		}
		PG_CATCH();
		{
			hash_search(tablespace_optimal_gpu_htable,
						&tablespace_oid,
						HASH_REMOVE,
						NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return hentry->optimal_gpu;
}

cl_int
GetOptimalGpuForRelation(PlannerInfo *root, RelOptInfo *rel)
{
	RangeTblEntry *rte;
	HeapTuple	tup;
	char		relpersistence;
	cl_int		cuda_dindex;

	if (baseRelIsArrowFdw(rel))
	{
		if (pgstrom_gpudirect_enabled())
			return GetOptimalGpuForArrowFdw(root, rel);
		return -1;
	}

	cuda_dindex = GetOptimalGpuForTablespace(rel->reltablespace);
	if (cuda_dindex < 0 || cuda_dindex >= numDevAttrs)
		return -1;

	/* only permanent / unlogged table can use NVMe-Strom */
	rte = root->simple_rte_array[rel->relid];
	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for relation %u", rte->relid);
	relpersistence = ((Form_pg_class) GETSTRUCT(tup))->relpersistence;
	ReleaseSysCache(tup);

	if (relpersistence == RELPERSISTENCE_PERMANENT ||
		relpersistence == RELPERSISTENCE_UNLOGGED)
		return cuda_dindex;

	return -1;
}

bool
RelationCanUseNvmeStrom(Relation relation)
{
	Oid		tablespace_oid = RelationGetForm(relation)->reltablespace;
	cl_int	cuda_dindex;
	/* SSD2GPU on temp relation is not supported */
	if (RelationUsesLocalBuffers(relation))
		return false;
	cuda_dindex = GetOptimalGpuForTablespace(tablespace_oid);
	return (cuda_dindex >= 0 &&
			cuda_dindex <  numDevAttrs);
}

/*
 * ScanPathWillUseNvmeStrom - Optimizer Hint
 */
bool
ScanPathWillUseNvmeStrom(PlannerInfo *root, RelOptInfo *baserel)
{
	size_t		num_scan_pages = 0;

	if (!pgstrom_gpudirect_enabled())
		return false;

	/*
	 * Check expected amount of the scan i/o.
	 * If 'baserel' is children of partition table, threshold shall be
	 * checked towards the entire partition size, because the range of
	 * child tables fully depend on scan qualifiers thus variable time
	 * by time. Once user focus on a particular range, but he wants to
	 * focus on other area. It leads potential thrashing on i/o.
	 */
	if (baserel->reloptkind == RELOPT_BASEREL)
	{
		if (GetOptimalGpuForRelation(root, baserel) >= 0)
			num_scan_pages = baserel->pages;
	}
	else if (baserel->reloptkind == RELOPT_OTHER_MEMBER_REL)
	{
		ListCell   *lc;
		Index		parent_relid = 0;

		foreach (lc, root->append_rel_list)
		{
			AppendRelInfo  *appinfo = (AppendRelInfo *) lfirst(lc);

			if (appinfo->child_relid == baserel->relid)
			{
				parent_relid = appinfo->parent_relid;
				break;
			}
		}
		if (!lc)
		{
			elog(NOTICE, "Bug? child table (%d) not found in append_rel_list",
				 baserel->relid);
			return false;
		}

		foreach (lc, root->append_rel_list)
		{
			AppendRelInfo  *appinfo = (AppendRelInfo *) lfirst(lc);
			RelOptInfo	   *rel;

			if (appinfo->parent_relid != parent_relid)
				continue;
			rel = root->simple_rel_array[appinfo->child_relid];
			if (GetOptimalGpuForRelation(root, rel) >= 0)
				num_scan_pages += rel->pages;
		}
	}
	else
		elog(ERROR, "Bug? unexpected reloptkind of base relation: %d",
			 (int)baserel->reloptkind);

	if (num_scan_pages < pgstrom_gpudirect_threshold() / BLCKSZ)
		return false;
	/* ok, this table scan can use nvme-strom */
	return true;
}

/*
 * pgstrom_init_gpu_direct
 */
void
pgstrom_init_gpu_direct(void)
{
	char	buffer[1024];
	int		index = 0;

	/*
	 * pg_strom.nvme_distance_map
	 *
	 * config := <token>[,<token>...]
	 * token  := nvmeXX:gpuXX
	 *
	 * eg) nvme0:gpu0,nvme1:gpu1
	 */
	DefineCustomStringVariable("pg_strom.nvme_distance_map",
							   "Manual configuration of GPU<->NVME distances",
							   NULL,
							   &nvme_manual_distance_map,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	extraSysfsSetupDistanceMap(nvme_manual_distance_map);

	while (extraSysfsPrintNvmeInfo(index, buffer, sizeof(buffer)) >= 0)
	{
		elog(LOG, "- %s", buffer);
		index++;
	}
}
