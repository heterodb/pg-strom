/*
 * relscan.c
 *
 * Routines related to outer relation scan
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* ----------------------------------------------------------------
 *
 * GPUDirectSQL related routines
 *
 * ----------------------------------------------------------------
 */
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

	if (!pgstrom_gpudirect_enabled())
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

	if (!pgstrom_gpudirect_enabled())
		return NULL;
#if 0
	if (baseRelIsArrowFdw(baserel))
	{
		if (pgstrom_gpudirect_enabled())
			return GetOptimalGpusForArrowFdw(root, baserel);
		return NULL;
	}
#endif
	total_sz = (size_t)baserel->pages * (size_t)BLCKSZ;
	if (total_sz < pgstrom_gpudirect_threshold())
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
 * Routines to setup kern_data_store
 *
 * ----------------------------------------------------------------
 */
static int
count_num_of_subfields(Oid type_oid)
{
	TypeCacheEntry *tcache;
	int		j, count = 0;

	tcache = lookup_type_cache(type_oid, TYPECACHE_TUPDESC);
	if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		/* array type */
		count = 1 + count_num_of_subfields(tcache->typelem);
	}
	else if (tcache->tupDesc)
	{
		/* composite type */
		TupleDesc	tupdesc = tcache->tupDesc;

		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

			count += count_num_of_subfields(attr->atttypid);
		}
	}
	return count;
}

static void
__setup_kern_colmeta(kern_data_store *kds,
					 int column_index,
					 const char *attname,
					 int attnum,
					 bool attbyval,
					 char attalign,
					 int16 attlen,
					 Oid atttypid,
					 int atttypmod,
					 int *p_attcacheoff)
{
	kern_colmeta   *cmeta = &kds->colmeta[column_index];
	TypeCacheEntry *tcache;

	cmeta->attbyval	= attbyval;
	cmeta->attalign	= typealign_get_width(attalign);
	cmeta->attlen	= attlen;
	if (attlen == 0 || attlen < -1)
		elog(ERROR, "attribute %s has unexpected length (%d)", attname, attlen);
	else if (attlen == -1)
		kds->has_varlena = true;
	cmeta->attnum	= attnum;

	if (!p_attcacheoff || *p_attcacheoff < 0)
		cmeta->attcacheoff = -1;
	else if (attlen > 0)
	{
		cmeta->attcacheoff = att_align_nominal(*p_attcacheoff, attalign);
		*p_attcacheoff = cmeta->attcacheoff + attlen;
	}
	else if (attlen == -1)
	{
		/*
		 * Note that attcacheoff is also available on varlena datum
		 * only if it appeared at the first, and its offset is aligned.
		 * Elsewhere, we cannot utilize the attcacheoff for varlena
		 */
		uint32_t	__off = att_align_nominal(*p_attcacheoff, attalign);

		if (*p_attcacheoff == __off)
			cmeta->attcacheoff = __off;
		else
			cmeta->attcacheoff = -1;
		*p_attcacheoff = -1;
	}
	else
	{
		cmeta->attcacheoff = *p_attcacheoff = -1;
	}
	cmeta->atttypid = atttypid;
	cmeta->atttypmod = atttypmod;
	strncpy(cmeta->attname, attname, NAMEDATALEN);

	/* array? composite type? */
	tcache = lookup_type_cache(atttypid, TYPECACHE_TUPDESC);
	if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		char		elem_name[NAMEDATALEN+10];
		int16		elem_len;
		bool		elem_byval;
		char		elem_align;

		cmeta->atttypkind = TYPE_KIND__ARRAY;
		cmeta->idx_subattrs = kds->nr_colmeta++;
		cmeta->num_subattrs = 1;

		snprintf(elem_name, sizeof(elem_name), "__%s", attname);
		get_typlenbyvalalign(tcache->typelem,
							 &elem_len,
							 &elem_byval,
							 &elem_align);
		__setup_kern_colmeta(kds,
							 cmeta->idx_subattrs,
							 elem_name,			/* attname */
							 1,					/* attnum */
							 elem_byval,		/* attbyval */
							 elem_align,		/* attalign */
							 elem_len,			/* attlen */
							 tcache->typelem,	/* atttypid */
							 -1,				/* atttypmod */
							 NULL);				/* attcacheoff */
	}
	else if (tcache->tupDesc)
	{
		TupleDesc	tupdesc = tcache->tupDesc;
		int			j, attcacheoff = -1;

		cmeta->atttypkind = TYPE_KIND__COMPOSITE;
		cmeta->idx_subattrs = kds->nr_colmeta;
		cmeta->num_subattrs = tupdesc->natts;
		kds->nr_colmeta += tupdesc->natts;

		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

			__setup_kern_colmeta(kds,
								 cmeta->idx_subattrs + j,
								 NameStr(attr->attname),
								 attr->attnum,
								 attr->attbyval,
								 attr->attalign,
								 attr->attlen,
								 attr->atttypid,
								 attr->atttypmod,
								 &attcacheoff);
		}
	}
	else
	{
		switch (tcache->typtype)
		{
			case TYPTYPE_BASE:
				cmeta->atttypkind = TYPE_KIND__BASE;
				break;
			case TYPTYPE_DOMAIN:
				cmeta->atttypkind = TYPE_KIND__DOMAIN;
				break;
			case TYPTYPE_ENUM:
				cmeta->atttypkind = TYPE_KIND__ENUM;
				break;
			case TYPTYPE_PSEUDO:
				cmeta->atttypkind = TYPE_KIND__PSEUDO;
				break;
			case TYPTYPE_RANGE:
				cmeta->atttypkind = TYPE_KIND__RANGE;
				break;
			default:
				elog(ERROR, "Unexpected typtype ('%c')", tcache->typtype);
				break;
		}
	}
	/*
	 * for the reverse references to KDS
	 */
	cmeta->kds_format = kds->format;
	cmeta->kds_offset = (char *)cmeta - (char *)kds;
}

size_t
setup_kern_data_store(kern_data_store *kds,
					  TupleDesc tupdesc,
					  size_t length,
					  char format)
{
	int		j, attcacheoff = -1;

	memset(kds, 0, offsetof(kern_data_store, colmeta));
	kds->length		= length;
	kds->nitems		= 0;
	kds->usage		= 0;
	kds->ncols		= tupdesc->natts;
	kds->format		= format;
	kds->tdhasoid	= false;	/* PG12 removed 'oid' system column */
	kds->tdtypeid	= tupdesc->tdtypeid;
	kds->tdtypmod	= tupdesc->tdtypmod;
	kds->table_oid	= InvalidOid;	/* to be set by the caller */
	kds->nslots		= 0;			/* to be set by the caller, if any */
	kds->nr_colmeta	= tupdesc->natts;

	if (format == KDS_FORMAT_ROW  ||
		format == KDS_FORMAT_HASH ||
		format == KDS_FORMAT_BLOCK)
		attcacheoff = 0;

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		__setup_kern_colmeta(kds, j,
							 NameStr(attr->attname),
							 attr->attnum,
							 attr->attbyval,
							 attr->attalign,
							 attr->attlen,
							 attr->atttypid,
							 attr->atttypmod,
							 &attcacheoff);
	}
	/* internal system attribute */
	if (format == KDS_FORMAT_COLUMN)
	{
		kern_colmeta *cmeta = &kds->colmeta[kds->nr_colmeta++];

		memset(cmeta, 0, sizeof(kern_colmeta));
		cmeta->attbyval = true;
		cmeta->attalign = sizeof(int32_t);
		cmeta->attlen = sizeof(GpuCacheSysattr);
		cmeta->attnum = -1;
		cmeta->attcacheoff = -1;
		cmeta->atttypid = InvalidOid;
		cmeta->atttypmod = -1;
		cmeta->atttypkind = TYPE_KIND__BASE;
		strcpy(cmeta->attname, "__gcache_sysattr__");
	}
	return MAXALIGN(offsetof(kern_data_store, colmeta[kds->nr_colmeta]));
}

size_t
estimate_kern_data_store(TupleDesc tupdesc)
{
	int		j, nr_colmeta = tupdesc->natts;

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		nr_colmeta += count_num_of_subfields(attr->atttypid);
	}
	return MAXALIGN(offsetof(kern_data_store, colmeta[nr_colmeta]));
}

/* ----------------------------------------------------------------
 *
 * Routines to load chunks from storage
 *
 * ----------------------------------------------------------------
 */
static bool
__kds_row_insert_tuple(TupleTableSlot *slot,
					   kern_data_store *kds, size_t kds_length)
{
	uint32_t   *rowindex = KDS_GET_ROWINDEX(kds);
	HeapTuple	tuple;
	size_t		sz, __usage;
	bool		should_free;
	kern_tupitem *titem;

	Assert(kds->format == KDS_FORMAT_ROW && kds->nslots == 0);
	tuple = ExecFetchSlotHeapTuple(slot, false, &should_free);

	__usage = (__kds_unpack(kds->usage) +
			   MAXALIGN(offsetof(kern_tupitem, htup) + tuple->t_len));
	sz = KDS_HEAD_LENGTH(kds) + sizeof(uint32_t) * (kds->nitems + 1) + __usage;
	if (sz > kds_length)
		return false;	/* no more items! */
	titem = (kern_tupitem *)((char *)kds + kds_length - __usage);
	titem->t_len = tuple->t_len;
	titem->rowid = kds->nitems;
	memcpy(&titem->htup, tuple->t_data, tuple->t_len);
	kds->usage = rowindex[kds->nitems++] = __kds_packed(__usage);

	if (should_free)
		heap_freetuple(tuple);
	ExecClearTuple(slot);

	return true;
}

bool
pgstromRelScanChunkNormal(pgstromTaskState *pts,
						  kern_data_store *kds, size_t kds_length)
{
	EState		   *estate = pts->css.ss.ps.state;
	TableScanDesc	scan = pts->css.ss.ss_currentScanDesc;
	TupleTableSlot *slot = pts->base_slot;

	if (pts->br_state)
	{
		/* scan by BRIN index */
		for (;;)
		{
			if (!pts->curr_tbm)
			{
				TBMIterateResult *next_tbm = pgstromBrinIndexNextBlock(pts);

				if (!next_tbm)
					break;
				if (!table_scan_bitmap_next_block(scan, next_tbm))
					elog(ERROR, "failed on table_scan_bitmap_next_block");
				pts->curr_tbm = next_tbm;
			}
			if (!TTS_EMPTY(slot) &&
				!__kds_row_insert_tuple(slot, kds, kds_length))
				break;
			if (!table_scan_bitmap_next_tuple(scan, pts->curr_tbm, slot))
				pts->curr_tbm = NULL;
			else if (!__kds_row_insert_tuple(slot, kds, kds_length))
				break;
		}
	}
	else
	{
		/* full table scan */
		for (;;)
		{
			if (!TTS_EMPTY(slot) &&
				!__kds_row_insert_tuple(slot, kds, kds_length))
				break;
			if (!table_scan_getnextslot(scan, estate->es_direction, slot))
				break;
			if (!__kds_row_insert_tuple(slot, kds, kds_length))
				break;
		}
	}
	return (kds->nitems > 0);
}

/*
 * pgstromSharedStateEstimateDSM
 */
Size
pgstromSharedStateEstimateDSM(pgstromTaskState *pts)
{
	EState	   *estate = pts->css.ss.ps.state;
	Snapshot	snapshot = estate->es_snapshot;
	Relation	relation = pts->css.ss.ss_currentRelation;
	Size		len = 0;

	if (pts->br_state)
		len += pgstromBrinIndexEstimateDSM(pts);
	len += MAXALIGN(sizeof(pgstromSharedState) +
					table_parallelscan_estimate(relation, snapshot));
	return len;
}

/*
 * pgstromSharedStateInitDSM
 */
void
pgstromSharedStateInitDSM(pgstromTaskState *pts, char *dsm_addr)
{
	pgstromSharedState *ps_state;
	Relation		relation = pts->css.ss.ss_currentRelation;
	TableScanDesc	scan;

	if (pts->br_state)
		dsm_addr += pgstromBrinIndexInitDSM(pts, dsm_addr);

	Assert(!pts->css.ss.ss_currentScanDesc);
	if (dsm_addr)
	{
		ps_state = (pgstromSharedState *)dsm_addr;
		memset(ps_state, 0, offsetof(pgstromSharedState, bpscan));
		scan = table_beginscan_parallel(relation, &ps_state->bpscan.base);
	}
	else
	{
		EState	   *estate = pts->css.ss.ps.state;

		ps_state = MemoryContextAllocZero(estate->es_query_cxt,
										  sizeof(pgstromSharedState));
		scan = table_beginscan(relation, estate->es_snapshot, 0, NULL);
	}
	pts->ps_state = ps_state;
	pts->css.ss.ss_currentScanDesc = scan;
}

/*
 * pgstromSharedStateReInitDSM
 */
void
pgstromSharedStateReInitDSM(pgstromTaskState *pts)
{
	//pgstromSharedState *ps_state = pts->ps_state;
	if (pts->br_state)
		pgstromBrinIndexReInitDSM(pts);
}

/*
 * pgstromSharedStateAttachDSM
 */
void
pgstromSharedStateAttachDSM(pgstromTaskState *pts, char *dsm_addr)
{
	if (pts->br_state)
		dsm_addr += pgstromBrinIndexAttachDSM(pts, dsm_addr);
	pts->ps_state = (pgstromSharedState *)dsm_addr;
}

/*
 * pgstromSharedStateShutdownDSM
 */
void
pgstromSharedStateShutdownDSM(pgstromTaskState *pts)
{
	pgstromSharedState *src_state = pts->ps_state;
	pgstromSharedState *dst_state;
	EState	   *estate = pts->css.ss.ps.state;

	if (pts->br_state)
		pgstromBrinIndexShutdownDSM(pts);
	if (src_state)
	{
		dst_state = MemoryContextAllocZero(estate->es_query_cxt,
										   sizeof(pgstromSharedState));
		memcpy(dst_state, src_state, sizeof(pgstromSharedState));
		pts->ps_state = dst_state;
	}
}

void
pgstrom_init_relscan(void)
{
	static char *nvme_manual_distance_map = NULL;
	char	buffer[1280];
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
							   "Manual configuration of optimal GPU for each NVME",
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
	/* hash table for tablespace <-> optimal GPU */
	tablespace_optimal_gpu_htable = NULL;
	CacheRegisterSyscacheCallback(TABLESPACEOID,
								  tablespace_optimal_gpu_cache_callback,
								  (Datum) 0);
}
