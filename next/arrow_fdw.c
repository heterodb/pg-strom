/*
 * arrow_fdw.c
 *
 * Routines to map Apache Arrow files as PG's Foreign-Table.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "arrow_defs.h"
#include "arrow_ipc.h"
#include "xpu_numeric.h"

/*
 * min/max statistics datum
 */
typedef struct
{
	bool		isnull;
	union {
		Datum	datum;
		NumericData	numeric;	/* if NUMERICOID */
	} min;
	union {
		Datum	datum;
		NumericData numeric;	/* if NUMERICOID */
	} max;
} MinMaxStatDatum;

/*
 * RecordBatchState
 */
typedef struct RecordBatchFieldState
{
	/* common fields with cache */
	Oid			atttypid;
	int			atttypmod;
	ArrowTypeOptions attopts;
	int64		nitems;				/* usually, same with rb_nitems */
	int64		null_count;
	off_t		nullmap_offset;
	size_t		nullmap_length;
	off_t		values_offset;
	size_t		values_length;
	off_t		extra_offset;
	size_t		extra_length;
	MinMaxStatDatum stat_datum;
	/* sub-fields if any */
	int			num_children;
	struct RecordBatchFieldState *children;
} RecordBatchFieldState;

typedef struct RecordBatchState
{
	struct ArrowFileState *af_state;	/* reference to ArrowFileState */
	int			rb_index;	/* index number in a file */
	off_t		rb_offset;	/* offset from the head */
	size_t		rb_length;	/* length of the entire RecordBatch */
	int64		rb_nitems;	/* number of items */
	/* per column information */
	int			nfields;
	RecordBatchFieldState fields[FLEXIBLE_ARRAY_MEMBER];
} RecordBatchState;

typedef struct ArrowFileState
{
	const char *filename;
	const char *dpu_path;	/* relative pathname, if DPU */
	struct stat	stat_buf;
	List	   *rb_list;	/* list of RecordBatchState */
} ArrowFileState;

/*
 * ArrowFdwState - executor state to run apache arrow
 */
typedef struct
{
	Bitmapset	   *stat_attrs;
	Bitmapset	   *load_attrs;
	List		   *orig_quals;		/* for EXPLAIN */
	List		   *eval_quals;
	ExprState	   *eval_state;
	ExprContext	   *econtext;
} arrowStatsHint;

struct ArrowFdwState
{
	Bitmapset		   *referenced;		/* referenced columns */
	arrowStatsHint	   *stats_hint;		/* min/max statistics, if any */
	pg_atomic_uint32   *rbatch_index;
	pg_atomic_uint32	__rbatch_index_local;	/* if single process */
	pg_atomic_uint32   *rbatch_nload;
	pg_atomic_uint32	__rbatch_nload_local;	/* if single process */
	pg_atomic_uint32   *rbatch_nskip;
	pg_atomic_uint32	__rbatch_nskip_local;	/* if single process */
	StringInfoData		chunk_buffer;	/* buffer to load record-batch */
	File				curr_filp;		/* current arrow file to read */
	kern_data_store	   *curr_kds;		/* current chunk to read */
	uint32_t			curr_index;		/* current index on the chunk */
	List			   *af_states_list;	/* list of ArrowFileState */
	uint32_t			rb_nitems;		/* number of record-batches */
	RecordBatchState   *rb_states[FLEXIBLE_ARRAY_MEMBER]; /* flatten RecordBatchState */
};

/*
 * Metadata Cache (on shared memory)
 */
#define ARROW_METADATA_BLOCKSZ		(128 * 1024)	/* 128kB */
typedef struct
{
	dlist_node	chain;		/* link to free_blocks; NULL if active */
	int32_t		unitsz;		/* unit size of slab items  */
	int32_t		n_actives;	/* number of active items */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} arrowMetadataCacheBlock;
#define ARROW_METADATA_CACHE_FREE_MAGIC		(0xdeadbeafU)
#define ARROW_METADATA_CACHE_ACTIVE_MAGIC	(0xcafebabeU)

typedef struct arrowMetadataFieldCache	arrowMetadataFieldCache;
typedef struct arrowMetadataCache		arrowMetadataCache;

struct arrowMetadataFieldCache
{
	arrowMetadataCacheBlock *owner;
	dlist_node	chain;				/* link to free/fields[children] list */
	/* common fields with cache */
	Oid			atttypid;
	int			atttypmod;
	ArrowTypeOptions attopts;
	int64		nitems;				/* usually, same with rb_nitems */
	int64		null_count;
	off_t		nullmap_offset;
	size_t		nullmap_length;
	off_t		values_offset;
	size_t		values_length;
	off_t		extra_offset;
	size_t		extra_length;
	MinMaxStatDatum stat_datum;
	/* sub-fields if any */
	int			num_children;
	dlist_head	children;
	uint32_t	magic;
};

struct arrowMetadataCache
{
	arrowMetadataCacheBlock *owner;
	dlist_node	chain;		/* link to free/hash list */
	dlist_node	lru_chain;	/* link to lru_list */
	struct timeval lru_tv;	/* last access time */
	arrowMetadataCache *next; /* next record-batch if any */
	struct stat stat_buf;	/* result of stat(2) */
	int			rb_index;	/* index number in a file */
	off_t		rb_offset;	/* offset from the head */
	size_t		rb_length;	/* length of the entire RecordBatch */
	int64		rb_nitems;	/* number of items */
	/* per column information */
	int			nfields;
	dlist_head	fields;		/* list of arrowMetadataFieldCache */
	uint32_t	magic;
};

/*
 * Metadata cache management
 */
#define ARROW_METADATA_HASH_NSLOTS		2000
typedef struct
{
	LWLock		mutex;
	slock_t		lru_lock;		/* protect lru related stuff */
	dlist_head	lru_list;
	dlist_head	free_blocks;	/* list of arrowMetadataCacheBlock */
	dlist_head	free_mcaches;	/* list of arrowMetadataCache */
	dlist_head	free_fcaches;	/* list of arrowMetadataFieldCache */
	dlist_head	hash_slots[ARROW_METADATA_HASH_NSLOTS];
} arrowMetadataCacheHead;

/*
 * Static variables
 */
static FdwRoutine			pgstrom_arrow_fdw_routine;
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static arrowMetadataCacheHead *arrow_metadata_cache = NULL;
static bool					arrow_fdw_enabled;	/* GUC */
static bool					arrow_fdw_stats_hint_enabled;	/* GUC */
static int					arrow_metadata_cache_size_kb;	/* GUC */

PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_handler);
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_validator);
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_import_file);
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_precheck_schema);

/* ----------------------------------------------------------------
 *
 * Apache Arrow <--> PG Types Mapping Routines
 *
 * ----------------------------------------------------------------
 */

/*
 * arrowFieldGetPGTypeHint
 */
static Oid
arrowFieldGetPGTypeHint(const ArrowField *field)
{
	for (int i=0; i < field->_num_custom_metadata; i++)
	{
		ArrowKeyValue *kv = &field->custom_metadata[i];
		char	   *namebuf, *pos;
		Oid			namespace_oid = PG_CATALOG_NAMESPACE;
		HeapTuple	tup;

		if (strcmp(kv->key, "pg_type") != 0)
			continue;
		namebuf = alloca(kv->_value_len + 10);
		strcpy(namebuf, kv->value);
		pos = strchr(namebuf, '.');
		if (pos)
		{
			*pos++ = '\0';
			namespace_oid = get_namespace_oid(namebuf, true);
			if (!OidIsValid(namespace_oid))
				continue;
			namebuf = pos;
		}
		tup = SearchSysCache2(TYPENAMENSP,
							  PointerGetDatum(namebuf),
							  ObjectIdGetDatum(namespace_oid));
		if (HeapTupleIsValid(tup))
		{
			Oid		hint = ((Form_pg_type) GETSTRUCT(tup))->oid;

			ReleaseSysCache(tup);

			return hint;
		}
	}
	return InvalidOid;
}

/* ------------------------------------------------
 * Metadata Cache Management Routines
 *
 * MEMO: all of them requires the caller must have exclusive lock
 *       on the arrowMetadataCache::mutex
 * ------------------------------------------------
 */
static void
__releaseMetadataFieldCache(arrowMetadataFieldCache *fcache)
{
	arrowMetadataCacheBlock *mc_block = fcache->owner;

	Assert(fcache->magic == ARROW_METADATA_CACHE_ACTIVE_MAGIC);
	/* also release sub-fields if any */
	while (!dlist_is_empty(&fcache->children))
	{
		arrowMetadataFieldCache	*__fcache
			= dlist_container(arrowMetadataFieldCache, chain,
							  dlist_pop_head_node(&fcache->children));
		__releaseMetadataFieldCache(__fcache);
	}
	fcache->magic = ARROW_METADATA_CACHE_FREE_MAGIC;
	dlist_push_tail(&arrow_metadata_cache->free_fcaches,
					&fcache->chain);

	/* also back the owner block if all slabs become free */
	Assert(mc_block->n_actives > 0);
	if (--mc_block->n_actives == 0)
	{
		char   *pos = mc_block->data;
		char   *end = (char *)mc_block + ARROW_METADATA_BLOCKSZ;

		Assert(mc_block->unitsz == MAXALIGN(sizeof(arrowMetadataFieldCache)));
		while (pos + mc_block->unitsz <= end)
		{
			arrowMetadataFieldCache *__fcache = (arrowMetadataFieldCache *)pos;
			Assert(__fcache->owner == mc_block &&
				   __fcache->magic == ARROW_METADATA_CACHE_FREE_MAGIC);
			dlist_delete(&__fcache->chain);
			pos += mc_block->unitsz;
		}
		Assert(!mc_block->chain.prev &&
			   !mc_block->chain.next);	/* must be active block */
		dlist_push_tail(&arrow_metadata_cache->free_blocks,
						&mc_block->chain);
	}
}

static void
__releaseMetadataCache(arrowMetadataCache *mcache)
{
	while (mcache)
	{
		arrowMetadataCacheBlock *mc_block = mcache->owner;
		arrowMetadataCache   *__mcache_next = mcache->next;

		Assert(mcache->magic == ARROW_METADATA_CACHE_ACTIVE_MAGIC);
		/*
		 * MEMO: Caller already detach the leader mcache from the hash-
		 * slot and the LRU-list. The follower mcaches should never be
		 * linked to hash-slot and LRU-list.
		 * So, we just put Assert() here.
		 */
		Assert(!mcache->chain.prev && !mcache->chain.next &&
			   !mcache->lru_chain.prev && !mcache->lru_chain.next);

		/* also release arrowMetadataFieldCache */
		while (!dlist_is_empty(&mcache->fields))
		{
			arrowMetadataFieldCache *fcache
				= dlist_container(arrowMetadataFieldCache, chain,
								  dlist_pop_head_node(&mcache->fields));
			__releaseMetadataFieldCache(fcache);
		}
		mcache->magic = ARROW_METADATA_CACHE_FREE_MAGIC;
		dlist_push_tail(&arrow_metadata_cache->free_mcaches,
						&mcache->chain);
		/* also back the owner block if all slabs become free */
		Assert(mc_block->n_actives > 0);
		if (--mc_block->n_actives == 0)
		{
			char   *pos = mc_block->data;
			char   *end = (char *)mc_block + ARROW_METADATA_BLOCKSZ;

			Assert(mc_block->unitsz == MAXALIGN(sizeof(arrowMetadataCache)));
			while (pos + mc_block->unitsz <= end)
			{
				arrowMetadataCache *__mcache = (arrowMetadataCache *)pos;

				Assert(__mcache->owner == mc_block &&
					   __mcache->magic == ARROW_METADATA_CACHE_FREE_MAGIC);
				dlist_delete(&__mcache->chain);
				pos += mc_block->unitsz;
			}
			Assert(!mc_block->chain.prev &&
				   !mc_block->chain.next);	/* must be active block */
			dlist_push_tail(&arrow_metadata_cache->free_blocks,
							&mc_block->chain);
		}
		mcache = __mcache_next;
	}
}

static bool
__reclaimMetadataCache(void)
{
	SpinLockAcquire(&arrow_metadata_cache->lru_lock);
	if (!dlist_is_empty(&arrow_metadata_cache->lru_list))
	{
		arrowMetadataCache *mcache;
		dlist_node	   *dnode;
		struct timeval	curr_tv;
		int64_t			elapsed;

		gettimeofday(&curr_tv, NULL);
		dnode = dlist_tail_node(&arrow_metadata_cache->lru_list);
		mcache = dlist_container(arrowMetadataCache, lru_chain, dnode);
		elapsed = ((curr_tv.tv_sec - mcache->lru_tv.tv_sec) * 1000000 +
				   (curr_tv.tv_usec - mcache->lru_tv.tv_usec));
		if (elapsed > 30000000UL)	/* > 30s */
		{
			dlist_delete(&mcache->lru_chain);
			memset(&mcache->lru_chain, 0, sizeof(dlist_node));
			SpinLockRelease(&arrow_metadata_cache->lru_lock);
			dlist_delete(&mcache->chain);
			memset(&mcache->chain, 0, sizeof(dlist_node));

			__releaseMetadataCache(mcache);
			return true;
		}
	}
	SpinLockRelease(&arrow_metadata_cache->lru_lock);
	return false;
}

static arrowMetadataFieldCache *
__allocMetadataFieldCache(void)
{
	arrowMetadataFieldCache *fcache;
	dlist_node *dnode;

	while (dlist_is_empty(&arrow_metadata_cache->free_fcaches))
	{
		arrowMetadataCacheBlock *mc_block;
		char   *pos, *end;

		while (dlist_is_empty(&arrow_metadata_cache->free_blocks))
		{
			if (!__reclaimMetadataCache())
				return NULL;
		}
		dnode = dlist_pop_head_node(&arrow_metadata_cache->free_blocks);
		mc_block = dlist_container(arrowMetadataCacheBlock, chain, dnode);
		memset(mc_block, 0, offsetof(arrowMetadataCacheBlock, data));
		mc_block->unitsz = MAXALIGN(sizeof(arrowMetadataFieldCache));
		for (pos = mc_block->data, end = (char *)mc_block + ARROW_METADATA_BLOCKSZ;
			 pos + mc_block->unitsz <= end;
			 pos += mc_block->unitsz)
		{
			fcache = (arrowMetadataFieldCache *)pos;
			fcache->owner = mc_block;
			fcache->magic = ARROW_METADATA_CACHE_FREE_MAGIC;
			dlist_push_tail(&arrow_metadata_cache->free_fcaches,
							&fcache->chain);
		}
	}
	dnode = dlist_pop_head_node(&arrow_metadata_cache->free_fcaches);
	fcache = dlist_container(arrowMetadataFieldCache, chain, dnode);
	fcache->owner->n_actives++;
	Assert(fcache->magic == ARROW_METADATA_CACHE_FREE_MAGIC);
	memset(&fcache->chain, 0, (offsetof(arrowMetadataFieldCache, magic) -
							   offsetof(arrowMetadataFieldCache, chain)));
	fcache->magic = ARROW_METADATA_CACHE_ACTIVE_MAGIC;
	return fcache;
}

static arrowMetadataCache *
__allocMetadataCache(void)
{
	arrowMetadataCache *mcache;
	dlist_node *dnode;

	if (dlist_is_empty(&arrow_metadata_cache->free_mcaches))
	{
		arrowMetadataCacheBlock *mc_block;
		char   *pos, *end;

		while (dlist_is_empty(&arrow_metadata_cache->free_blocks))
		{
			if (!__reclaimMetadataCache())
				return NULL;
		}
		dnode = dlist_pop_head_node(&arrow_metadata_cache->free_blocks);
		mc_block = dlist_container(arrowMetadataCacheBlock, chain, dnode);
		memset(mc_block, 0, offsetof(arrowMetadataCacheBlock, data));
		mc_block->unitsz = MAXALIGN(sizeof(arrowMetadataCache));
		for (pos = mc_block->data, end = (char *)mc_block + ARROW_METADATA_BLOCKSZ;
			 pos + mc_block->unitsz <= end;
			 pos += mc_block->unitsz)
		{
			mcache = (arrowMetadataCache *)pos;
			mcache->owner = mc_block;
			mcache->magic = ARROW_METADATA_CACHE_FREE_MAGIC;
			dlist_push_tail(&arrow_metadata_cache->free_mcaches,
							&mcache->chain);
		}
	}
	dnode = dlist_pop_head_node(&arrow_metadata_cache->free_mcaches);
	mcache = dlist_container(arrowMetadataCache, chain, dnode);
	mcache->owner->n_actives++;
	Assert(mcache->magic == ARROW_METADATA_CACHE_FREE_MAGIC);
	memset(&mcache->chain, 0, (offsetof(arrowMetadataCache, magic) -
							   offsetof(arrowMetadataCache, chain)));
	mcache->magic = ARROW_METADATA_CACHE_ACTIVE_MAGIC;
	return mcache;
}

/*
 * lookupArrowMetadataCache
 *
 * caller must hold "at least" shared lock on the arrow_metadata_cache->mutex.
 * if exclusive lock is held, it may invalidate legacy cache if any.
 */
static inline uint32_t
arrowMetadataHashIndex(struct stat *stat_buf)
{
	struct {
		dev_t	st_dev;
		ino_t	st_ino;
	} hkey;
	uint32_t	hash;

	hkey.st_dev = stat_buf->st_dev;
	hkey.st_ino = stat_buf->st_ino;
	hash = hash_bytes((unsigned char *)&hkey, sizeof(hkey));
	return hash % ARROW_METADATA_HASH_NSLOTS;
}

static arrowMetadataCache *
lookupArrowMetadataCache(struct stat *stat_buf, bool has_exclusive)
{
	arrowMetadataCache *mcache;
	uint32_t	hindex;
	dlist_iter	iter;

	hindex = arrowMetadataHashIndex(stat_buf);
	dlist_foreach(iter, &arrow_metadata_cache->hash_slots[hindex])
	{
		mcache = dlist_container(arrowMetadataCache, chain, iter.cur);

		if (stat_buf->st_dev == mcache->stat_buf.st_dev &&
			stat_buf->st_ino == mcache->stat_buf.st_ino)
		{
			/*
			 * Is the metadata cache still valid?
			 */
			if (stat_buf->st_mtim.tv_sec < mcache->stat_buf.st_mtim.tv_sec ||
				(stat_buf->st_mtim.tv_sec == mcache->stat_buf.st_mtim.tv_sec &&
				 stat_buf->st_mtim.tv_nsec <= mcache->stat_buf.st_mtim.tv_nsec))
			{
				/* ok, found */
				SpinLockAcquire(&arrow_metadata_cache->lru_lock);
				gettimeofday(&mcache->lru_tv, NULL);
				dlist_move_head(&arrow_metadata_cache->lru_list,
								&mcache->lru_chain);
				SpinLockRelease(&arrow_metadata_cache->lru_lock);
				return mcache;
			}
			else if (has_exclusive)
			{
				/*
				 * Unfortunatelly, metadata cache is already invalid.
				 * If caller has exclusive lock, we release it.
				 */
				SpinLockAcquire(&arrow_metadata_cache->lru_lock);
				dlist_delete(&mcache->lru_chain);
				memset(&mcache->lru_chain, 0, sizeof(dlist_node));
				SpinLockRelease(&arrow_metadata_cache->lru_lock);
				dlist_delete(&mcache->chain);
				memset(&mcache->chain, 0, sizeof(dlist_node));

				__releaseMetadataCache(mcache);
			}
		}
	}
	return NULL;
}

/* ----------------------------------------------------------------
 *
 * buildArrowStatsBinary
 *
 * ...and, routines related to Arrow Min/Max statistics
 *
 * ----------------------------------------------------------------
 */
typedef struct arrowFieldStatsBinary
{
	uint32	nrooms;		/* number of record-batches */
	MinMaxStatDatum *stat_values;
	int		nfields;	/* if List/Struct data type */
	struct arrowFieldStatsBinary *subfields;
} arrowFieldStatsBinary;

typedef struct
{
	int		nitems;		/* number of record-batches */
	int		nfields;	/* number of columns */
	arrowFieldStatsBinary fields[FLEXIBLE_ARRAY_MEMBER];
} arrowStatsBinary;

static void
__releaseArrowFieldStatsBinary(arrowFieldStatsBinary *bstats)
{
	if (bstats->subfields)
	{
		for (int j=0; j < bstats->nfields; j++)
			__releaseArrowFieldStatsBinary(&bstats->subfields[j]);
		pfree(bstats->subfields);
	}
	if (bstats->stat_values)
		pfree(bstats->stat_values);
}

static void
releaseArrowStatsBinary(arrowStatsBinary *arrow_bstats)
{
	if (arrow_bstats)
	{
		for (int j=0; j < arrow_bstats->nfields; j++)
			__releaseArrowFieldStatsBinary(&arrow_bstats->fields[j]);
		pfree(arrow_bstats);
	}
}

static int128_t
__atoi128(const char *tok, bool *p_isnull)
{
	int128_t	ival = 0;
	bool		is_minus = false;

	if (*tok == '-')
	{
		is_minus = true;
		tok++;
	}
	while (isdigit(*tok))
	{
		ival = 10 * ival + (*tok - '0');
		tok++;
	}

	if (*tok != '\0')
		*p_isnull = true;
	if (is_minus)
	{
		if (ival == 0)
			*p_isnull = true;
		ival = -ival;
	}
	return ival;
}

static bool
__parseArrowFieldStatsBinary(arrowFieldStatsBinary *bstats,
							 ArrowField *field,
							 const char *min_tokens,
							 const char *max_tokens)
{
	MinMaxStatDatum *stat_values;
	char	   *min_buffer;
	char	   *max_buffer;
	char	   *tok1, *pos1;
	char	   *tok2, *pos2;
	uint32_t	index;

	/* parse the min_tokens/max_tokens */
	min_buffer = alloca(strlen(min_tokens) + 1);
	max_buffer = alloca(strlen(max_tokens) + 1);
	strcpy(min_buffer, min_tokens);
	strcpy(max_buffer, max_tokens);

	stat_values = palloc0(sizeof(MinMaxStatDatum) * bstats->nrooms);
	for (tok1 = strtok_r(min_buffer, ",", &pos1),
		 tok2 = strtok_r(max_buffer, ",", &pos2), index = 0;
		 tok1 != NULL && tok2 != NULL && index < bstats->nrooms;
		 tok1 = strtok_r(NULL, ",", &pos1),
		 tok2 = strtok_r(NULL, ",", &pos2), index++)
	{
		bool		__isnull = false;
		int128_t	__min = __atoi128(__trim(tok1), &__isnull);
		int128_t	__max = __atoi128(__trim(tok2), &__isnull);

		if (__isnull)
		{
			stat_values[index].isnull = true;
			continue;
		}

		switch (field->type.node.tag)
		{
			case ArrowNodeTag__Int:
			case ArrowNodeTag__FloatingPoint:
				stat_values[index].min.datum = (Datum)__min;
				stat_values[index].max.datum = (Datum)__min;
				break;

			case ArrowNodeTag__Decimal:
				__xpu_numeric_to_varlena((char *)&stat_values[index].min.numeric,
										 field->type.Decimal.scale,
										 __min);
				__xpu_numeric_to_varlena((char *)&stat_values[index].max.numeric,
                                         field->type.Decimal.scale,
                                         __max);
				break;

			case ArrowNodeTag__Date:
				switch (field->type.Date.unit)
				{
					case ArrowDateUnit__Day:
						stat_values[index].min.datum = __min
							- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
						stat_values[index].max.datum = __max
							- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
						break;
					case ArrowDateUnit__MilliSecond:
						stat_values[index].min.datum = __min / (SECS_PER_DAY * 1000)
							- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
						stat_values[index].max.datum = __max / (SECS_PER_DAY * 1000)
							- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
						break;
					default:
						goto bailout;
				}
				break;

			case ArrowNodeTag__Time:
				switch (field->type.Time.unit)
				{
					case ArrowTimeUnit__Second:
						stat_values[index].min.datum = __min * 1000000L;
						stat_values[index].max.datum = __max * 1000000L;
						break;
					case ArrowTimeUnit__MilliSecond:
						stat_values[index].min.datum = __min * 1000L;
						stat_values[index].max.datum = __max * 1000L;
						break;
					case ArrowTimeUnit__MicroSecond:
						stat_values[index].min.datum = __min;
						stat_values[index].max.datum = __max;
						break;
					case ArrowTimeUnit__NanoSecond:
						stat_values[index].min.datum = __min / 1000;
						stat_values[index].max.datum = __max / 1000;
						break;
					default:
						goto bailout;
				}
				break;

			case ArrowNodeTag__Timestamp:
				switch (field->type.Timestamp.unit)
				{
					case ArrowTimeUnit__Second:
						stat_values[index].min.datum = __min * 1000000L;
						stat_values[index].max.datum = __max * 1000000L;
						break;
					case ArrowTimeUnit__MilliSecond:
						stat_values[index].min.datum = __min * 1000L;
						stat_values[index].max.datum = __max * 1000L;
						break;
					case ArrowTimeUnit__MicroSecond:
						stat_values[index].min.datum = __min;
						stat_values[index].max.datum = __max;
						break;
					case ArrowTimeUnit__NanoSecond:
						stat_values[index].min.datum = __min / 1000;
						stat_values[index].max.datum = __max / 1000;
						break;
					default:
						goto bailout;
				}
				break;
			default:
				goto bailout;
		}
	}
	/* sanity checks */
	if (!tok1 && !tok2 && index == bstats->nrooms)
	{
		bstats->stat_values = stat_values;
		return true;
	}
bailout:
	pfree(stat_values);
	return false;
}

static bool
__buildArrowFieldStatsBinary(arrowFieldStatsBinary *bstats,
							 ArrowField *field,
							 uint32 numRecordBatches)
{
	const char *min_tokens = NULL;
	const char *max_tokens = NULL;
	int			j, k;
	bool		retval = false;

	for (k=0; k < field->_num_custom_metadata; k++)
	{
		ArrowKeyValue *kv = &field->custom_metadata[k];

		if (strcmp(kv->key, "min_values") == 0)
			min_tokens = kv->value;
		else if (strcmp(kv->key, "max_values") == 0)
			max_tokens = kv->value;
	}

	bstats->nrooms = numRecordBatches;
	if (min_tokens && max_tokens)
	{
		if (__parseArrowFieldStatsBinary(bstats, field,
										 min_tokens,
										 max_tokens))
		{
			retval = true;
		}
	}

	if (field->_num_children > 0)
	{
		bstats->nfields = field->_num_children;
		bstats->subfields = palloc0(sizeof(arrowFieldStatsBinary) * bstats->nfields);
		for (j=0; j < bstats->nfields; j++)
		{
			if (__buildArrowFieldStatsBinary(&bstats->subfields[j],
											 &field->children[j],
											 numRecordBatches))
				retval = true;
		}
	}
	return retval;
}

static arrowStatsBinary *
buildArrowStatsBinary(const ArrowFooter *footer, Bitmapset **p_stat_attrs)
{
	arrowStatsBinary *arrow_bstats;
	int		nfields = footer->schema._num_fields;
	bool	found = false;

	arrow_bstats = palloc0(offsetof(arrowStatsBinary,
									fields[nfields]));
	arrow_bstats->nitems = footer->_num_recordBatches;
	arrow_bstats->nfields = nfields;
	for (int j=0; j < nfields; j++)
	{
		if (__buildArrowFieldStatsBinary(&arrow_bstats->fields[j],
										 &footer->schema.fields[j],
										 footer->_num_recordBatches))
		{
			if (p_stat_attrs)
				*p_stat_attrs = bms_add_member(*p_stat_attrs, j+1);
			found = true;
		}
	}
	if (!found)
	{
		releaseArrowStatsBinary(arrow_bstats);
		return NULL;
	}
	return arrow_bstats;
}

/*
 * applyArrowStatsBinary
 */
static void
__applyArrowFieldStatsBinary(RecordBatchFieldState *rb_field,
							 arrowFieldStatsBinary *bstats,
							 int rb_index)
{
	int		j;

	if (bstats->stat_values)
	{
		memcpy(&rb_field->stat_datum,
			   &bstats->stat_values[rb_index], sizeof(MinMaxStatDatum));
	}
	else
	{
		rb_field->stat_datum.isnull = true;
	}
	Assert(rb_field->num_children == bstats->nfields);
	for (j=0; j < rb_field->num_children; j++)
	{
		RecordBatchFieldState  *__rb_field = &rb_field->children[j];
		arrowFieldStatsBinary  *__bstats = &bstats->subfields[j];

		__applyArrowFieldStatsBinary(__rb_field, __bstats, rb_index);
	}
}

static void
applyArrowStatsBinary(RecordBatchState *rb_state, arrowStatsBinary *arrow_bstats)
{
	Assert(rb_state->nfields == arrow_bstats->nfields &&
		   rb_state->rb_index < arrow_bstats->nitems);
	for (int j=0; j < rb_state->nfields; j++)
	{
		__applyArrowFieldStatsBinary(&rb_state->fields[j],
									 &arrow_bstats->fields[j],
									 rb_state->rb_index);
	}
}

/*
 * execInitArrowStatsHint / execCheckArrowStatsHint / execEndArrowStatsHint
 *
 * ... are executor routines for min/max statistics.
 */
static bool
__buildArrowStatsOper(arrowStatsHint *as_hint,
					  ScanState *ss,
					  OpExpr *op,
					  bool reverse)
{
	Index		scanrelid = ((Scan *)ss->ps.plan)->scanrelid;
	Oid			opcode;
	Var		   *var;
	Node	   *arg;
	Expr	   *expr;
	Oid			opfamily = InvalidOid;
	StrategyNumber strategy = InvalidStrategy;
	CatCList   *catlist;
	int			i;

	if (!reverse)
	{
		opcode = op->opno;
		var = linitial(op->args);
		arg = lsecond(op->args);
	}
	else
	{
		opcode = get_commutator(op->opno);
		var = lsecond(op->args);
		arg = linitial(op->args);
	}
	/* Is it VAR <OPER> ARG form? */
	if (!IsA(var, Var) || var->varno != scanrelid || !OidIsValid(opcode))
		return false;
	if (!bms_is_member(var->varattno, as_hint->stat_attrs))
		return false;
	if (contain_var_clause(arg) ||
		contain_volatile_functions(arg))
		return false;

	catlist = SearchSysCacheList1(AMOPOPID, ObjectIdGetDatum(opcode));
	for (i=0; i < catlist->n_members; i++)
	{
		HeapTuple	tuple = &catlist->members[i]->tuple;
		Form_pg_amop amop = (Form_pg_amop) GETSTRUCT(tuple);

		if (amop->amopmethod == BRIN_AM_OID)
		{
			opfamily = amop->amopfamily;
			strategy = amop->amopstrategy;
			break;
		}
	}
	ReleaseSysCacheList(catlist);

	if (strategy == BTLessStrategyNumber ||
		strategy == BTLessEqualStrategyNumber)
	{
		/* if (VAR < ARG) --> (Min >= ARG), can be skipped */
		/* if (VAR <= ARG) --> (Min > ARG), can be skipped */
		opcode = get_negator(opcode);
		if (!OidIsValid(opcode))
			return false;
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(INNER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		as_hint->eval_quals = lappend(as_hint->eval_quals, expr);
	}
	else if (strategy == BTGreaterEqualStrategyNumber ||
			 strategy == BTGreaterStrategyNumber)
	{
		/* if (VAR > ARG) --> (Max <= ARG), can be skipped */
		/* if (VAR >= ARG) --> (Max < ARG), can be skipped */
		opcode = get_negator(opcode);
		if (!OidIsValid(opcode))
			return false;
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(OUTER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		as_hint->eval_quals = lappend(as_hint->eval_quals, expr);
	}
	else if (strategy == BTEqualStrategyNumber)
	{
		/* (VAR = ARG) --> (Min > ARG) || (Max < ARG), can be skipped */
		opcode = get_opfamily_member(opfamily, var->vartype,
									 exprType((Node *)arg),
									 BTGreaterStrategyNumber);
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(INNER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		as_hint->eval_quals = lappend(as_hint->eval_quals, expr);

		opcode = get_opfamily_member(opfamily, var->vartype,
									 exprType((Node *)arg),
									 BTLessEqualStrategyNumber);
		expr = make_opclause(opcode,
							 op->opresulttype,
							 op->opretset,
							 (Expr *)makeVar(OUTER_VAR,
											 var->varattno,
											 var->vartype,
											 var->vartypmod,
											 var->varcollid,
											 0),
							 (Expr *)copyObject(arg),
							 op->opcollid,
							 op->inputcollid);
		set_opfuncid((OpExpr *)expr);
		as_hint->eval_quals = lappend(as_hint->eval_quals, expr);
	}
	else
	{
		return false;
	}
	as_hint->load_attrs = bms_add_member(as_hint->load_attrs, var->varattno);

	return true;
}

static arrowStatsHint *
execInitArrowStatsHint(ScanState *ss, List *outer_quals, Bitmapset *stat_attrs)
{
	Relation		relation = ss->ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(relation);
	arrowStatsHint *as_hint;
	ExprContext	   *econtext;
	Expr		   *eval_expr;
	ListCell	   *lc;

	as_hint = palloc0(sizeof(arrowStatsHint));
	as_hint->stat_attrs = stat_attrs;
	foreach (lc, outer_quals)
	{
		OpExpr *op = lfirst(lc);

		if (IsA(op, OpExpr) && list_length(op->args) == 2 &&
			(__buildArrowStatsOper(as_hint, ss, op, false) ||
			 __buildArrowStatsOper(as_hint, ss, op, true)))
		{
			as_hint->orig_quals = lappend(as_hint->orig_quals, op);
		}
	}
	if (as_hint->eval_quals == NIL)
		return NULL;
	if (list_length(as_hint->eval_quals) == 1)
		eval_expr = linitial(as_hint->eval_quals);
	else
		eval_expr = make_orclause(as_hint->eval_quals);

	econtext = CreateExprContext(ss->ps.state);
	econtext->ecxt_innertuple = MakeSingleTupleTableSlot(tupdesc, &TTSOpsVirtual);
	econtext->ecxt_outertuple = MakeSingleTupleTableSlot(tupdesc, &TTSOpsVirtual);

	as_hint->eval_state = ExecInitExpr(eval_expr, &ss->ps);
	as_hint->econtext = econtext;

	return as_hint;
}

static bool
execCheckArrowStatsHint(arrowStatsHint *stats_hint,
						RecordBatchState *rb_state)
{
	ExprContext	   *econtext = stats_hint->econtext;
	TupleTableSlot *min_values = econtext->ecxt_innertuple;
	TupleTableSlot *max_values = econtext->ecxt_outertuple;
	int				anum;
	Datum			datum;
	bool			isnull;

	/* load the min/max statistics */
	ExecStoreAllNullTuple(min_values);
	ExecStoreAllNullTuple(max_values);
	for (anum = bms_next_member(stats_hint->load_attrs, -1);
		 anum >= 0;
		 anum = bms_next_member(stats_hint->load_attrs, anum))
	{
		RecordBatchFieldState *rb_field = &rb_state->fields[anum-1];

		Assert(anum > 0 && anum <= rb_state->nfields);
		if (!rb_field->stat_datum.isnull)
		{
			min_values->tts_isnull[anum-1] = false;
			max_values->tts_isnull[anum-1] = false;
			if (rb_field->atttypid == NUMERICOID)
			{
				min_values->tts_values[anum-1]
					= PointerGetDatum(&rb_field->stat_datum.min.numeric);
				max_values->tts_values[anum-1]
					= PointerGetDatum(&rb_field->stat_datum.max.numeric);
			}
			else
			{
				min_values->tts_values[anum-1] = rb_field->stat_datum.min.datum;
				max_values->tts_values[anum-1] = rb_field->stat_datum.max.datum;
			}
		}
	}
	datum = ExecEvalExprSwitchContext(stats_hint->eval_state, econtext, &isnull);
//	elog(INFO, "file [%s] rb_index=%u datum=%lu isnull=%d",
//		 FilePathName(rb_state->fdesc), rb_state->rb_index, datum, (int)isnull);
	if (!isnull && DatumGetBool(datum))
		return true;	/* ok, skip this record-batch */
	return false;
}

static void
execEndArrowStatsHint(arrowStatsHint *stats_hint)
{
	ExprContext	   *econtext = stats_hint->econtext;

	ExecDropSingleTupleTableSlot(econtext->ecxt_innertuple);
	ExecDropSingleTupleTableSlot(econtext->ecxt_outertuple);
	econtext->ecxt_innertuple = NULL;
	econtext->ecxt_outertuple = NULL;

	FreeExprContext(econtext, true);
}


/* ----------------------------------------------------------------
 *
 * BuildArrowFileState
 *
 * It build RecordBatchState based on the metadata-cache, or raw Arrow files.
 * ----------------------------------------------------------------
 */
static void
__buildRecordBatchFieldStateByCache(RecordBatchFieldState *rb_field,
									arrowMetadataFieldCache *fcache)
{
	rb_field->atttypid       = fcache->atttypid;
	rb_field->atttypmod      = fcache->atttypmod;
	rb_field->attopts        = fcache->attopts;
	rb_field->nitems         = fcache->nitems;
	rb_field->null_count     = fcache->null_count;
	rb_field->nullmap_offset = fcache->nullmap_offset;
	rb_field->nullmap_length = fcache->nullmap_length;
	rb_field->values_offset  = fcache->values_offset;
	rb_field->values_length  = fcache->values_length;
	rb_field->extra_offset   = fcache->extra_offset;
	rb_field->extra_length   = fcache->extra_length;
	memcpy(&rb_field->stat_datum,
		   &fcache->stat_datum, sizeof(MinMaxStatDatum));
	if (fcache->num_children > 0)
	{
		dlist_iter	iter;
		int			j = 0;

		rb_field->num_children = fcache->num_children;
		rb_field->children = palloc0(sizeof(RecordBatchFieldState) *
									 fcache->num_children);
		dlist_foreach(iter, &fcache->children)
		{
			arrowMetadataFieldCache *__fcache
				= dlist_container(arrowMetadataFieldCache, chain, iter.cur);
			__buildRecordBatchFieldStateByCache(&rb_field->children[j++], __fcache);
		}
		Assert(j == rb_field->num_children);
	}
	else
	{
		Assert(dlist_is_empty(&fcache->children));
	}
}

static ArrowFileState *
__buildArrowFileStateByCache(const char *filename,
							 arrowMetadataCache *mcache,
							 Bitmapset **p_stat_attrs)
{
	ArrowFileState	   *af_state;

	af_state = palloc0(sizeof(ArrowFileState));
	af_state->filename = pstrdup(filename);
	memcpy(&af_state->stat_buf, &mcache->stat_buf, sizeof(struct stat));

	while (mcache)
	{
		RecordBatchState *rb_state;
		dlist_iter	iter;
		int			j = 0;

		rb_state = palloc0(offsetof(RecordBatchState,
									fields[mcache->nfields]));
		rb_state->af_state  = af_state;
		rb_state->rb_index  = mcache->rb_index;
		rb_state->rb_offset = mcache->rb_offset;
		rb_state->rb_length = mcache->rb_length;
		rb_state->rb_nitems = mcache->rb_nitems;
		rb_state->nfields   = mcache->nfields;
		dlist_foreach(iter, &mcache->fields)
		{
			arrowMetadataFieldCache *fcache;

			fcache = dlist_container(arrowMetadataFieldCache, chain, iter.cur);
			if (p_stat_attrs && fcache->stat_datum.isnull)
				*p_stat_attrs = bms_add_member(*p_stat_attrs, j+1);
			__buildRecordBatchFieldStateByCache(&rb_state->fields[j++], fcache);
		}
		Assert(j == rb_state->nfields);
		af_state->rb_list = lappend(af_state->rb_list, rb_state);

		mcache = mcache->next;
	}
	return af_state;
}

/*
 * Routines to setup RecordBatchState by raw-file
 */
typedef struct
{
	ArrowBuffer	   *buffer_curr;
	ArrowBuffer	   *buffer_tail;
	ArrowFieldNode *fnode_curr;
	ArrowFieldNode *fnode_tail;
} setupRecordBatchContext;

static Oid
__lookupCompositePGType(int nattrs, Oid *type_oids, Oid hint_oid)
{
	Relation	rel;
	ScanKeyData	skeys[3];
	SysScanDesc	sscan;
	Oid			comp_oid = InvalidOid;

	rel = table_open(RelationRelationId, AccessShareLock);
	ScanKeyInit(&skeys[0],
				Anum_pg_class_relkind,
				BTEqualStrategyNumber, F_CHAREQ,
				CharGetDatum(RELKIND_COMPOSITE_TYPE));
	ScanKeyInit(&skeys[1],
				Anum_pg_class_relnatts,
				BTEqualStrategyNumber, F_INT2EQ,
				Int16GetDatum(nattrs));
	ScanKeyInit(&skeys[2],
				Anum_pg_class_oid,
				BTEqualStrategyNumber, F_OIDNE,
				ObjectIdGetDatum(hint_oid));
	sscan = systable_beginscan(rel, InvalidOid, false, NULL,
							   OidIsValid(hint_oid) ? 3 : 2, skeys);
	for (;;)
	{
		HeapTuple	htup;
		TupleDesc	tupdesc;
		int			j;

		if (OidIsValid(hint_oid))
		{
			comp_oid = hint_oid;
			hint_oid = InvalidOid;
		}
		else
		{
			htup = systable_getnext(sscan);
			if (!HeapTupleIsValid(htup))
				break;
			comp_oid = ((Form_pg_type) GETSTRUCT(htup))->oid;
		}

		if (pg_type_aclcheck(comp_oid,
							 GetUserId(),
							 ACL_USAGE) != ACLCHECK_OK)
			continue;

		tupdesc = lookup_rowtype_tupdesc_noerror(comp_oid, -1, true);
		if (!tupdesc)
			continue;
		if (tupdesc->natts == nattrs)
		{
			for (j=0; j < tupdesc->natts; j++)
			{
				Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

				if (attr->atttypid != type_oids[j])
					break;
			}
			if (j == tupdesc->natts)
			{
				ReleaseTupleDesc(tupdesc);
				goto found;
			}
		}
		ReleaseTupleDesc(tupdesc);
	}
	comp_oid = InvalidOid;	/* not found */
found:
	systable_endscan(sscan);
	table_close(rel, AccessShareLock);

	return comp_oid;
}

static void
__arrowFieldTypeToPGType(const ArrowField *field,
						 Oid *p_type_oid,
						 int32_t *p_type_mod,
						 ArrowTypeOptions *p_attopts)
{
	const ArrowType *t = &field->type;
	Oid			type_oid = InvalidOid;
	int32_t		type_mod = -1;
	Oid			hint_oid = arrowFieldGetPGTypeHint(field);
	ArrowTypeOptions attopts;

	memset(&attopts, 0, sizeof(ArrowTypeOptions));
	switch (t->node.tag)
	{
		case ArrowNodeTag__Int:
			attopts.tag = ArrowType__Int;
			switch (t->Int.bitWidth)
			{
				case 8:
					attopts.unitsz = sizeof(int8_t);
					type_oid =
						GetSysCacheOid2(TYPENAMENSP,
										Anum_pg_type_oid,
										CStringGetDatum("int1"),
										ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
					break;
				case 16:
					attopts.unitsz = sizeof(int16_t);
					type_oid = INT2OID;
					break;
				case 32:
					attopts.unitsz = sizeof(int32_t);
					type_oid = INT4OID;
					break;
				case 64:
					attopts.unitsz = sizeof(int64_t);
					type_oid = INT8OID;
					break;
				default:
					elog(ERROR, "Arrow::Int bitWidth=%d is not supported",
						 t->Int.bitWidth);
			}
			attopts.integer.bitWidth  = t->Int.bitWidth;
			attopts.integer.is_signed = t->Int.is_signed;
			break;

		case ArrowNodeTag__FloatingPoint:
			attopts.tag = ArrowType__FloatingPoint;
			switch (t->FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					attopts.unitsz = sizeof(float2_t);
					type_oid =
						GetSysCacheOid2(TYPENAMENSP,
										Anum_pg_type_oid,
										CStringGetDatum("float2"),
										ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
					break;
				case ArrowPrecision__Single:
					attopts.unitsz = sizeof(float4_t);
					type_oid = FLOAT4OID;
					break;
				case ArrowPrecision__Double:
					attopts.unitsz = sizeof(float8_t);
					type_oid = FLOAT8OID;
					break;
				default:
					elog(ERROR, "Arrow::FloatingPoint unknown precision (%d)",
						 (int)t->FloatingPoint.precision);
			}
			attopts.floating_point.precision = t->FloatingPoint.precision;
			break;

		case ArrowNodeTag__Bool:
			attopts.tag = ArrowType__Bool;
			attopts.unitsz = -1;		/* values is bitmap */
			type_oid = BOOLOID;
			break;

		case ArrowNodeTag__Decimal:
			if (t->Decimal.bitWidth != 128)
				elog(ERROR, "Arrow::Decimal%u is not supported", t->Decimal.bitWidth);
			attopts.tag               = ArrowType__Decimal;
			attopts.unitsz            = sizeof(int128_t);
			attopts.decimal.precision = t->Decimal.precision;
			attopts.decimal.scale     = t->Decimal.scale;
			attopts.decimal.bitWidth  = t->Decimal.bitWidth;
			type_oid = NUMERICOID;
			break;

		case ArrowNodeTag__Date:
			attopts.tag = ArrowType__Date;
			switch (t->Date.unit)
			{
				case ArrowDateUnit__Day:
					attopts.unitsz = sizeof(int32_t);
					break;
				case ArrowDateUnit__MilliSecond:
					attopts.unitsz = sizeof(int32_t);
					break;
				default:
					elog(ERROR, "Arrow::Date unknown unit (%d)",
						 (int)t->Date.unit);
			}
			attopts.date.unit = t->Date.unit;
			type_oid = DATEOID;
			break;

		case ArrowNodeTag__Time:
			attopts.tag = ArrowType__Time;
			switch (t->Time.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
					attopts.unitsz = sizeof(int32_t);
					break;
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					attopts.unitsz = sizeof(int64_t);
					break;
				default:
					elog(ERROR, "unknown Time::unit (%d)",
						 (int)t->Time.unit);
			}
			attopts.time.unit = t->Time.unit;
			type_oid = TIMEOID;
			break;

		case ArrowNodeTag__Timestamp:
			attopts.tag = ArrowType__Timestamp;
			switch (t->Timestamp.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					attopts.unitsz = sizeof(int64_t);
					break;
				default:
					elog(ERROR, "unknown Timestamp::unit (%d)",
						 (int)t->Timestamp.unit);
			}
			attopts.timestamp.unit = t->Timestamp.unit;
			type_oid = (t->Timestamp.timezone
						? TIMESTAMPTZOID
						: TIMESTAMPOID);
			break;

		case ArrowNodeTag__Interval:
			attopts.tag = ArrowType__Interval;
			switch (t->Interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					attopts.unitsz = sizeof(int32_t);
					break;
				case ArrowIntervalUnit__Day_Time:
					attopts.unitsz = sizeof(int64_t);
					break;
				default:
					elog(ERROR, "unknown Interval::unit (%d)",
                         (int)t->Interval.unit);
			}
			attopts.interval.unit = t->Interval.unit;
			type_oid = INTERVALOID;
			break;

		case ArrowNodeTag__FixedSizeBinary:
			attopts.tag = ArrowType__FixedSizeBinary;
			attopts.unitsz = t->FixedSizeBinary.byteWidth;
			attopts.fixed_size_binary.byteWidth = t->FixedSizeBinary.byteWidth;
			if (t->FixedSizeBinary.byteWidth <= 0 ||
				t->FixedSizeBinary.byteWidth > BLCKSZ)
				elog(ERROR, "arrow_fdw: %s with byteWidth=%d is not supported", 
					 t->node.tagName,
					 t->FixedSizeBinary.byteWidth);
			if (hint_oid == MACADDROID &&
				t->FixedSizeBinary.byteWidth == 6)
			{
				type_oid = MACADDROID;
			}
			else if (hint_oid == INETOID &&
					 (t->FixedSizeBinary.byteWidth == 4 ||
                      t->FixedSizeBinary.byteWidth == 16))
			{
				type_oid = INETOID;
			}
			else
			{
				type_oid = BPCHAROID;
				type_mod = VARHDRSZ + t->FixedSizeBinary.byteWidth;
			}
			break;

		case ArrowNodeTag__Utf8:
			attopts.tag = ArrowType__Utf8;
			attopts.unitsz = sizeof(uint32_t);
			type_oid = TEXTOID;
			break;

		case ArrowNodeTag__LargeUtf8:
			attopts.tag = ArrowType__LargeUtf8;
			attopts.unitsz = sizeof(uint64_t);
			type_oid = TEXTOID;
			break;

		case ArrowNodeTag__Binary:
			attopts.tag = ArrowType__Binary;
			attopts.unitsz = sizeof(uint32_t);
			type_oid = BYTEAOID;
			break;

		case ArrowNodeTag__LargeBinary:
			attopts.tag = ArrowType__LargeBinary;
			attopts.unitsz = sizeof(uint64_t);
			type_oid = BYTEAOID;
			break;

		case ArrowNodeTag__List:
		case ArrowNodeTag__LargeList:
			if (field->_num_children != 1)
				elog(ERROR, "Bug? List of arrow type is corrupted");
			else
			{
				Oid			__type_oid = InvalidOid;

				attopts.tag = ArrowType__List;
				attopts.unitsz = (t->node.tag == ArrowNodeTag__List
								  ? sizeof(uint32_t)
								  : sizeof(uint64_t));
				__arrowFieldTypeToPGType(&field->children[0],
										 &__type_oid,
										 NULL,
										 NULL);
				type_oid = get_array_type(__type_oid);
				if (!OidIsValid(type_oid))
					elog(ERROR, "arrow_fdw: no array type for '%s'",
						 format_type_be(__type_oid));
			}
			break;

		case ArrowNodeTag__Struct:
			{
				Oid	   *__type_oids;

				attopts.tag = ArrowType__Struct;
				attopts.unitsz = 0;		/* only nullmap */
				__type_oids = alloca(sizeof(Oid) * (field->_num_children + 1));
				for (int j=0; j < field->_num_children; j++)
				{
					__arrowFieldTypeToPGType(&field->children[j],
											 &__type_oids[j],
											 NULL,
											 NULL);
				}
				type_oid = __lookupCompositePGType(field->_num_children,
												   __type_oids,
												   hint_oid);
				if (!OidIsValid(type_oid))
					elog(ERROR, "arrow_fdw: no suitable composite type");
			}
			break;

		default:
			elog(ERROR, "Bug? ArrowSchema contains unsupported types");
	}

	if (p_type_oid)
		*p_type_oid = type_oid;
	if (p_type_mod)
		*p_type_mod = type_mod;
	if (p_attopts)
		memcpy(p_attopts, &attopts, sizeof(ArrowTypeOptions));
}

static void
__buildRecordBatchFieldState(setupRecordBatchContext *con,
							 RecordBatchFieldState *rb_field,
							 ArrowField *field, int depth)
{
	ArrowFieldNode *fnode;
	ArrowBuffer	   *buffer_curr;
	size_t			least_values_length = 0;
	bool			has_extra_buffer = false;

	if (con->fnode_curr >= con->fnode_tail)
		elog(ERROR, "RecordBatch has less ArrowFieldNode than expected");
	fnode = con->fnode_curr++;
	rb_field->atttypid    = InvalidOid;
	rb_field->atttypmod   = -1;
	rb_field->nitems      = fnode->length;
	rb_field->null_count  = fnode->null_count;
	rb_field->stat_datum.isnull = true;
	__arrowFieldTypeToPGType(field,
							 &rb_field->atttypid,
							 &rb_field->atttypmod,
							 &rb_field->attopts);
	/* assign buffers */
	switch (field->type.node.tag)
	{
		case ArrowNodeTag__Bool:
			least_values_length = BITMAPLEN(rb_field->nitems);
			break;
		case ArrowNodeTag__Int:
		case ArrowNodeTag__FloatingPoint:
		case ArrowNodeTag__Decimal:
		case ArrowNodeTag__Date:
		case ArrowNodeTag__Time:
		case ArrowNodeTag__Timestamp:
		case ArrowNodeTag__Interval:
		case ArrowNodeTag__FixedSizeBinary:
			least_values_length = rb_field->attopts.unitsz * rb_field->nitems;
			break;

		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__LargeUtf8:
		case ArrowNodeTag__Binary:
		case ArrowNodeTag__LargeBinary:
			least_values_length = rb_field->attopts.unitsz * (rb_field->nitems + 1);
			has_extra_buffer = true;
			break;

		case ArrowNodeTag__List:
        case ArrowNodeTag__LargeList:
			if (depth > 0)
				elog(ERROR, "nested array type is not supported");
			least_values_length = rb_field->attopts.unitsz * (rb_field->nitems + 1);
			break;

		case ArrowNodeTag__Struct:
			if (depth > 0)
				elog(ERROR, "nested composite type is not supported");
			/* no values and extra buffer, only nullmap */
			break;
		default:
			elog(ERROR, "Bug? ArrowSchema contains unsupported types");
	}

	/* setup nullmap buffer */
	buffer_curr = con->buffer_curr++;
	if (buffer_curr >= con->buffer_tail)
		elog(ERROR, "RecordBatch has less buffers than expected");
	if (rb_field->null_count > 0)
	{
		rb_field->nullmap_offset = buffer_curr->offset;
		rb_field->nullmap_length = buffer_curr->length;
		if (rb_field->nullmap_length < BITMAPLEN(rb_field->nitems))
			elog(ERROR, "nullmap length is smaller than expected");
		if (rb_field->nullmap_offset != MAXALIGN(rb_field->nullmap_offset))
			elog(ERROR, "nullmap is not aligned well");
	}

	/* setup values buffer */
	if (least_values_length > 0)
	{
		buffer_curr = con->buffer_curr++;
		if (buffer_curr >= con->buffer_tail)
			elog(ERROR, "RecordBatch has less buffers than expected");
		rb_field->values_offset = buffer_curr->offset;
		rb_field->values_length = buffer_curr->length;
		if (rb_field->values_length < least_values_length)
			elog(ERROR, "values array is smaller than expected");
		if (rb_field->values_offset != MAXALIGN(rb_field->values_offset))
			elog(ERROR, "values array is not aligned well");
	}

	/* setup extra buffer */
	if (has_extra_buffer)
	{
		Assert(least_values_length > 0);
		buffer_curr = con->buffer_curr++;
		if (buffer_curr >= con->buffer_tail)
			elog(ERROR, "RecordBatch has less buffers than expected");
		rb_field->extra_offset = buffer_curr->offset;
		rb_field->extra_length = buffer_curr->length;
		if (rb_field->extra_offset != MAXALIGN(rb_field->extra_offset))
			elog(ERROR, "extra buffer is not aligned well");
	}

	/* child fields, if any */
	if (field->_num_children > 0)
	{
		rb_field->children = palloc0(sizeof(RecordBatchFieldState) *
									 field->_num_children);
		for (int j=0; j < field->_num_children; j++)
		{
			__buildRecordBatchFieldState(con,
										 &rb_field->children[j],
										 &field->children[j],
										 depth+1);
		}
	}
	rb_field->num_children = field->_num_children;
}

static RecordBatchState *
__buildRecordBatchStateOne(ArrowSchema *schema,
						   ArrowFileState *af_state,
						   int rb_index,
						   ArrowBlock *block,
						   ArrowRecordBatch *rbatch)
{
	setupRecordBatchContext con;
	RecordBatchState *rb_state;
	int			nfields = schema->_num_fields;

	if (rbatch->compression)
		elog(ERROR, "arrow_fdw: right now, compressed record-batche is not supported");

	rb_state = palloc0(offsetof(RecordBatchState, fields[nfields]));
	rb_state->af_state = af_state;
	rb_state->rb_index = rb_index;
	rb_state->rb_offset = block->offset + block->metaDataLength;
	rb_state->rb_length = block->bodyLength;
	rb_state->rb_nitems = rbatch->length;
	rb_state->nfields   = nfields;

	memset(&con, 0, sizeof(setupRecordBatchContext));
	con.buffer_curr = rbatch->buffers;
	con.buffer_tail = rbatch->buffers + rbatch->_num_buffers;
	con.fnode_curr  = rbatch->nodes;
	con.fnode_tail  = rbatch->nodes + rbatch->_num_nodes;
	for (int j=0; j < nfields; j++)
	{
		RecordBatchFieldState *rb_field = &rb_state->fields[j];
		ArrowField	   *field = &schema->fields[j];

		__buildRecordBatchFieldState(&con, rb_field, field, 0);
	}
	if (con.buffer_curr != con.buffer_tail ||
		con.fnode_curr  != con.fnode_tail)
		elog(ERROR, "arrow_fdw: RecordBatch may be corrupted");
	return rb_state;
}

/*
 * readArrowFile
 */
static bool
readArrowFile(const char *filename, ArrowFileInfo *af_info, bool missing_ok)
{
    File	filp = PathNameOpenFile(filename, O_RDONLY | PG_BINARY);

	if (filp < 0)
	{
		if (missing_ok && errno == ENOENT)
			return false;
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open file \"%s\": %m", filename)));
	}
	readArrowFileDesc(FileGetRawDesc(filp), af_info);
	FileClose(filp);
	if (af_info->dictionaries != NULL)
		elog(ERROR, "DictionaryBatch is not supported at '%s'", filename);
	Assert(af_info->footer._num_dictionaries == 0);
	return true;
}

static ArrowFileState *
__buildArrowFileStateByFile(const char *filename, Bitmapset **p_stat_attrs)
{
	ArrowFileInfo af_info;
	ArrowFileState *af_state;
	arrowStatsBinary *arrow_bstats;

	if (!readArrowFile(filename, &af_info, true))
	{
		elog(DEBUG2, "file '%s' is missing: %m", filename);
		return NULL;
	}
	if (af_info.recordBatches == NULL)
	{
		elog(DEBUG2, "arrow file '%s' contains no RecordBatch", filename);
		return NULL;
	}
	/* allocate ArrowFileState */
	af_state = palloc0(sizeof(ArrowFileInfo));
	af_state->filename = pstrdup(filename);
	memcpy(&af_state->stat_buf, &af_info.stat_buf, sizeof(struct stat));

	arrow_bstats = buildArrowStatsBinary(&af_info.footer, p_stat_attrs);
	for (int i=0; i < af_info.footer._num_recordBatches; i++)
	{
		ArrowBlock	     *block  = &af_info.footer.recordBatches[i];
		ArrowRecordBatch *rbatch = &af_info.recordBatches[i].body.recordBatch;
		RecordBatchState *rb_state;

		rb_state = __buildRecordBatchStateOne(&af_info.footer.schema,
											  af_state, i, block, rbatch);
		if (arrow_bstats)
			applyArrowStatsBinary(rb_state, arrow_bstats);
		af_state->rb_list = lappend(af_state->rb_list, rb_state);
	}
	releaseArrowStatsBinary(arrow_bstats);

	return af_state;
}


static arrowMetadataFieldCache *
__buildArrowMetadataFieldCache(RecordBatchFieldState *rb_field)
{
	arrowMetadataFieldCache *fcache;

	fcache = __allocMetadataFieldCache();
	if (!fcache)
		return NULL;
	fcache->atttypid = rb_field->atttypid;
	fcache->atttypmod = rb_field->atttypmod;
	memcpy(&fcache->attopts, &rb_field->attopts, sizeof(ArrowTypeOptions));
	fcache->nitems = rb_field->nitems;
	fcache->null_count = rb_field->null_count;
	fcache->nullmap_offset = rb_field->nullmap_offset;
	fcache->nullmap_length = rb_field->nullmap_length;
	fcache->values_offset = rb_field->values_offset;
	fcache->values_length = rb_field->values_length;
	fcache->extra_offset = rb_field->extra_offset;
	fcache->extra_length = rb_field->extra_length;
	memcpy(&fcache->stat_datum,
		   &rb_field->stat_datum, sizeof(MinMaxStatDatum));
	fcache->num_children = rb_field->num_children;
	dlist_init(&fcache->children);
	for (int j=0; j < rb_field->num_children; j++)
	{
		arrowMetadataFieldCache *__fcache;

		__fcache = __buildArrowMetadataFieldCache(&rb_field->children[j]);
		if (!__fcache)
		{
			__releaseMetadataFieldCache(fcache);
			return NULL;
		}
		dlist_push_tail(&fcache->children, &__fcache->chain);
	}
	return fcache;
}

/*
 * __buildArrowMetadataCacheNoLock
 *
 * it builds arrowMetadataCache entries according to the supplied
 * ArrowFileState
 */
static void
__buildArrowMetadataCacheNoLock(ArrowFileState *af_state)
{
	arrowMetadataCache *mcache_head = NULL;
	arrowMetadataCache *mcache_prev = NULL;
	arrowMetadataCache *mcache;
	uint32_t	hindex;
	ListCell   *lc;

	foreach (lc, af_state->rb_list)
	{
		RecordBatchState *rb_state = lfirst(lc);

		mcache = __allocMetadataCache();
		if (!mcache)
		{
			__releaseMetadataCache(mcache_head);
			return;
		}
		memcpy(&mcache->stat_buf,
			   &af_state->stat_buf, sizeof(struct stat));
		mcache->rb_index  = rb_state->rb_index;
		mcache->rb_offset = rb_state->rb_offset;
		mcache->rb_length = rb_state->rb_length;
		mcache->rb_nitems = rb_state->rb_nitems;
		mcache->nfields   = rb_state->nfields;
		dlist_init(&mcache->fields);
		if (!mcache_head)
			mcache_head = mcache;
		else
			mcache_prev->next = mcache;

		for (int j=0; j < rb_state->nfields; j++)
		{
			arrowMetadataFieldCache *fcache;

			fcache = __buildArrowMetadataFieldCache(&rb_state->fields[j]);
			if (!fcache)
			{
				__releaseMetadataCache(mcache_head);
				return;
			}
			dlist_push_tail(&mcache->fields, &fcache->chain);
		}
		mcache_prev = mcache;
	}
	/* chain to the list */
	hindex = arrowMetadataHashIndex(&af_state->stat_buf);
	dlist_push_tail(&arrow_metadata_cache->hash_slots[hindex],
					&mcache_head->chain );
	SpinLockAcquire(&arrow_metadata_cache->lru_lock);
	gettimeofday(&mcache_head->lru_tv, NULL);
	dlist_push_head(&arrow_metadata_cache->lru_list, &mcache_head->lru_chain);
	SpinLockRelease(&arrow_metadata_cache->lru_lock);
}

static ArrowFileState *
BuildArrowFileState(Relation frel, const char *filename, Bitmapset **p_stat_attrs)
{
	arrowMetadataCache *mcache;
	ArrowFileState *af_state;
	RecordBatchState *rb_state;
	struct stat		stat_buf;
	TupleDesc		tupdesc;

	if (stat(filename, &stat_buf) != 0)
		elog(ERROR, "failed on stat('%s'): %m", filename);
	LWLockAcquire(&arrow_metadata_cache->mutex, LW_SHARED);
	mcache = lookupArrowMetadataCache(&stat_buf, false);
	if (mcache)
	{
		/* found a valid metadata-cache */
		af_state = __buildArrowFileStateByCache(filename, mcache,
												p_stat_attrs);
	}
	else
	{
		LWLockRelease(&arrow_metadata_cache->mutex);

		/* here is no valid metadata-cache, so build it from the raw file */
		af_state = __buildArrowFileStateByFile(filename, p_stat_attrs);
		if (!af_state)
			return NULL;	/* file not found? */

		LWLockAcquire(&arrow_metadata_cache->mutex, LW_EXCLUSIVE);
		mcache = lookupArrowMetadataCache(&af_state->stat_buf, true);
		if (!mcache)
			__buildArrowMetadataCacheNoLock(af_state);
	}
	LWLockRelease(&arrow_metadata_cache->mutex);

	/* compatibility checks */
	rb_state = linitial(af_state->rb_list);
	tupdesc = RelationGetDescr(frel);
	if (tupdesc->natts != rb_state->nfields)
		elog(ERROR, "arrow_fdw: foreign table '%s' is not compatible to '%s'",
			 RelationGetRelationName(frel), filename);
	for (int j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute	attr = TupleDescAttr(tupdesc, j);
		RecordBatchFieldState *rb_field = &rb_state->fields[j];

		if (attr->atttypid != rb_field->atttypid)
			elog(ERROR, "arrow_fdw: foreign table '%s' column '%s' (%s) is not compatible to the arrow field (%s) in the '%s'",
				 RelationGetRelationName(frel),
				 NameStr(attr->attname),
				 format_type_be(attr->atttypid),
				 format_type_be(rb_field->atttypid),
				 filename);
	}
	return af_state;
}

/*
 * baseRelIsArrowFdw
 */
bool
baseRelIsArrowFdw(RelOptInfo *baserel)
{
	if ((baserel->reloptkind == RELOPT_BASEREL ||
		 baserel->reloptkind == RELOPT_OTHER_MEMBER_REL) &&
		baserel->rtekind == RTE_RELATION &&
		OidIsValid(baserel->serverid) &&
		baserel->fdwroutine &&
		memcmp(baserel->fdwroutine,
			   &pgstrom_arrow_fdw_routine,
			   sizeof(FdwRoutine)) == 0)
		return true;

	return false;
}

/*
 * RelationIsArrowFdw
 */
bool
RelationIsArrowFdw(Relation frel)
{
	if (RelationGetForm(frel)->relkind == RELKIND_FOREIGN_TABLE)
	{
		FdwRoutine *routine = GetFdwRoutineForRelation(frel, false);

		if (memcmp(routine, &pgstrom_arrow_fdw_routine, sizeof(FdwRoutine)) == 0)
			return true;
	}
	return false;
}

/*
 * GetOptimalGpusForArrowFdw
 */
const Bitmapset *
GetOptimalGpusForArrowFdw(PlannerInfo *root, RelOptInfo *baserel)
{
	List	   *priv_list = (List *)baserel->fdw_private;
	Bitmapset  *optimal_gpus = NULL;

	if (baseRelIsArrowFdw(baserel) &&
		IsA(priv_list, List) && list_length(priv_list) == 2)
	{
		List	   *af_list = lsecond(priv_list);
		ListCell   *lc;

		foreach (lc, af_list)
		{
			ArrowFileState *af_state = lfirst(lc);
			const Bitmapset *__optimal_gpus;

			__optimal_gpus = GetOptimalGpuForFile(af_state->filename);
			if (lc == list_head(af_list))
				optimal_gpus = bms_copy(__optimal_gpus);
			else
				optimal_gpus = bms_intersect(optimal_gpus, __optimal_gpus);
		}
	}
	return optimal_gpus;
}

/*
 * GetOptimalDpuForArrowFdw
 */
const DpuStorageEntry *
GetOptimalDpuForArrowFdw(PlannerInfo *root, RelOptInfo *baserel)
{
	List	   *priv_list = (List *)baserel->fdw_private;
	const DpuStorageEntry *ds_entry = NULL;

	if (baseRelIsArrowFdw(baserel) &&
		IsA(priv_list, List) && list_length(priv_list) == 2)
	{
		List	   *af_list = lsecond(priv_list);
		ListCell   *lc;

		foreach (lc, af_list)
		{
			ArrowFileState *af_state = lfirst(lc);
			const DpuStorageEntry *__ds_entry;

			__ds_entry = GetOptimalDpuForFile(af_state->filename, NULL);
			if (lc == list_head(af_list))
				ds_entry = __ds_entry;
			else if (ds_entry && ds_entry != __ds_entry)
				ds_entry = NULL;
		}
	}
	return ds_entry;
}

/*
 * arrowFdwExtractFilesList
 */
static List *
arrowFdwExtractFilesList(List *options_list,
						 int *p_parallel_nworkers)
{

	ListCell   *lc;
	List	   *filesList = NIL;
	char	   *dir_path = NULL;
	char	   *dir_suffix = NULL;
	int			parallel_nworkers = -1;

	foreach (lc, options_list)
	{
		DefElem	   *defel = lfirst(lc);

		Assert(IsA(defel->arg, String));
		if (strcmp(defel->defname, "file") == 0)
		{
			char   *temp = strVal(defel->arg);

			if (access(temp, R_OK) != 0)
				elog(ERROR, "arrow_fdw: unable to access '%s': %m", temp);
			filesList = lappend(filesList, makeString(pstrdup(temp)));
		}
		else if (strcmp(defel->defname, "files") == 0)
		{
			char   *temp = pstrdup(strVal(defel->arg));
			char   *saveptr;
			char   *tok;

			while ((tok = strtok_r(temp, ",", &saveptr)) != NULL)
			{
				tok = __trim(tok);

				if (*tok != '/')
					elog(ERROR, "arrow_fdw: file '%s' must be absolute path", tok);
				if (access(tok, R_OK) != 0)
					elog(ERROR, "arrow_fdw: unable to access '%s': %m", tok);
				filesList = lappend(filesList, makeString(pstrdup(tok)));
			}
			pfree(temp);
		}
		else if (strcmp(defel->defname, "dir") == 0)
		{
			dir_path = strVal(defel->arg);
			if (*dir_path != '/')
				elog(ERROR, "arrow_fdw: dir '%s' must be absolute path", dir_path);
		}
		else if (strcmp(defel->defname, "suffix") == 0)
		{
			dir_suffix = strVal(defel->arg);
		}
		else if (strcmp(defel->defname, "parallel_workers") == 0)
		{
			if (parallel_nworkers >= 0)
				elog(ERROR, "'parallel_workers' appeared twice");
			parallel_nworkers = atoi(strVal(defel->arg));
		}
		else
			elog(ERROR, "arrow: unknown option (%s)", defel->defname);
	}
	if (dir_suffix && !dir_path)
		elog(ERROR, "arrow: cannot use 'suffix' option without 'dir'");

	if (dir_path)
	{
		struct dirent *dentry;
		DIR	   *dir;
		char   *temp;

		dir = AllocateDir(dir_path);
		while ((dentry = ReadDir(dir, dir_path)) != NULL)
		{
			if (strcmp(dentry->d_name, ".") == 0 ||
				strcmp(dentry->d_name, "..") == 0)
				continue;
			if (dir_suffix)
			{
				char   *pos = strrchr(dentry->d_name, '.');

				if (!pos || strcmp(pos+1, dir_suffix) != 0)
					continue;
			}
			temp = psprintf("%s/%s", dir_path, dentry->d_name);
			if (access(temp, R_OK) != 0)
			{
				elog(DEBUG1, "arrow_fdw: unable to read '%s', so skipped", temp);
				continue;
			}
			filesList = lappend(filesList, makeString(temp));
		}
		FreeDir(dir);
	}

	if (p_parallel_nworkers)
		*p_parallel_nworkers = parallel_nworkers;
	return filesList;
}

/* ----------------------------------------------------------------
 *
 * arrowFdwLoadRecordBatch() and related routines
 *
 * it setup KDS (ARROW format) with IOvec according to RecordBatchState
 *
 * ----------------------------------------------------------------
 */

/*
 * arrowFdwSetupIOvector
 */
typedef struct
{
	off_t		rb_offset;
	off_t		f_offset;
	off_t		m_offset;
	size_t		kds_head_sz;
	int32_t		depth;
	int32_t		io_index;
	strom_io_chunk ioc[FLEXIBLE_ARRAY_MEMBER];
} arrowFdwSetupIOContext;

static void
__setupIOvectorField(arrowFdwSetupIOContext *con,
					 off_t chunk_offset,
					 size_t chunk_length,
					 uint32_t *p_cmeta_offset,
					 uint32_t *p_cmeta_length)
{
	off_t		f_pos = con->rb_offset + chunk_offset;
	size_t		__length = MAXALIGN(chunk_length);

	Assert(con->m_offset == MAXALIGN(con->m_offset));

	if (f_pos == con->f_offset)
	{
		/* good, buffer is fully continuous */
		*p_cmeta_offset = __kds_packed(con->kds_head_sz +
									   con->m_offset);
		*p_cmeta_length = __kds_packed(__length);

		con->m_offset += __length;
		con->f_offset += __length;
	}
	else if (f_pos > con->f_offset &&
			 (f_pos & ~PAGE_MASK) == (con->f_offset & ~PAGE_MASK) &&
			 (f_pos - con->f_offset) == MAXALIGN(f_pos - con->f_offset))
	{
		/*
		 * we can also consolidate the i/o of two chunks, if file position
		 * of the next chunk (f_pos) and the current file tail position
		 * (con->f_offset) locate within the same file page, and if gap bytes
		 * on the file does not break alignment.
		 */
		size_t	__gap = (f_pos - con->f_offset);

		/* put gap bytes */
		Assert(__gap < PAGE_SIZE);
		con->m_offset += __gap;
		con->f_offset += __gap;

		*p_cmeta_offset = __kds_packed(con->kds_head_sz +
									   con->m_offset);
		*p_cmeta_length = __kds_packed(__length);

		con->m_offset += __length;
		con->f_offset += __length;
	}
	else
	{
		/*
		 * Elsewhere, we have no chance to consolidate this chunk to
		 * the previous i/o-chunk. So, make a new i/o-chunk.
		 */
		off_t		f_base = TYPEALIGN_DOWN(PAGE_SIZE, f_pos);
		off_t		gap = f_pos - f_base;
		strom_io_chunk *ioc;

		if (con->io_index < 0)
			con->io_index = 0;		/* no previous i/o chunks */
		else
		{
			off_t	f_tail = PAGE_ALIGN(con->f_offset);

			ioc = &con->ioc[con->io_index++];
			ioc->nr_pages = f_tail / PAGE_SIZE - ioc->fchunk_id;
			con->m_offset += (f_tail - con->f_offset);	/* margin for alignment */
		}
		Assert(con->m_offset == PAGE_ALIGN(con->m_offset));
		ioc = &con->ioc[con->io_index];
		ioc->m_offset   = con->m_offset;
		ioc->fchunk_id  = f_base / PAGE_SIZE;

		con->m_offset += gap;
		*p_cmeta_offset = __kds_packed(con->kds_head_sz +
									   con->m_offset);
		*p_cmeta_length = __kds_packed(__length);
		con->m_offset += __length;
		con->f_offset  = f_pos + __length;
	}
}

static void
arrowFdwSetupIOvectorField(arrowFdwSetupIOContext *con,
						   RecordBatchFieldState *rb_field,
						   kern_data_store *kds,
						   kern_colmeta *cmeta)
{
	//int		index = cmeta - kds->colmeta;

	if (rb_field->nullmap_length > 0)
	{
		Assert(rb_field->null_count > 0);
		__setupIOvectorField(con,
							 rb_field->nullmap_offset,
							 rb_field->nullmap_length,
							 &cmeta->nullmap_offset,
							 &cmeta->nullmap_length);
		//elog(INFO, "D%d att[%d] nullmap=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, rb_field->nullmap_offset, rb_field->nullmap_length, con->m_offset, con->f_offset);
	}
	if (rb_field->values_length > 0)
	{
		__setupIOvectorField(con,
							 rb_field->values_offset,
							 rb_field->values_length,
							 &cmeta->values_offset,
							 &cmeta->values_length);
		//elog(INFO, "D%d att[%d] values=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, rb_field->values_offset, rb_field->values_length, con->m_offset, con->f_offset);
	}
	if (rb_field->extra_length > 0)
	{
		__setupIOvectorField(con,
							 rb_field->extra_offset,
							 rb_field->extra_length,
							 &cmeta->extra_offset,
							 &cmeta->extra_length);
		//elog(INFO, "D%d att[%d] extra=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, rb_field->extra_offset, rb_field->extra_length, con->m_offset, con->f_offset);
	}

	/* nested sub-fields if composite types */
	if (cmeta->atttypkind == TYPE_KIND__ARRAY ||
		cmeta->atttypkind == TYPE_KIND__COMPOSITE)
	{
		kern_colmeta *subattr;
		int		j;

		Assert(rb_field->num_children == cmeta->num_subattrs);
		con->depth++;
		for (j=0, subattr = &kds->colmeta[cmeta->idx_subattrs];
			 j < cmeta->num_subattrs;
			 j++, subattr++)
		{
			RecordBatchFieldState *child = &rb_field->children[j];

			arrowFdwSetupIOvectorField(con, child, kds, subattr);
		}
		con->depth--;
	}
}

static strom_io_vector *
arrowFdwSetupIOvector(RecordBatchState *rb_state,
					  Bitmapset *referenced,
					  kern_data_store *kds)
{
	arrowFdwSetupIOContext *con;
	strom_io_vector *iovec;

	Assert(kds->ncols <= kds->nr_colmeta &&
		   kds->ncols == rb_state->nfields);
	con = alloca(offsetof(arrowFdwSetupIOContext,
						  ioc[3 * kds->nr_colmeta]));
	con->rb_offset = rb_state->rb_offset;
	con->f_offset  = ~0UL;	/* invalid offset */
	con->m_offset  = 0;
	con->kds_head_sz = KDS_HEAD_LENGTH(kds);
	con->depth = 0;
	con->io_index = -1;		/* invalid index */
	for (int j=0; j < kds->ncols; j++)
	{
		RecordBatchFieldState *rb_field = &rb_state->fields[j];
		kern_colmeta *cmeta = &kds->colmeta[j];
		int			attidx = j + 1 - FirstLowInvalidHeapAttributeNumber;

		if (bms_is_member(attidx, referenced) ||
			bms_is_member(-FirstLowInvalidHeapAttributeNumber, referenced))
			arrowFdwSetupIOvectorField(con, rb_field, kds, cmeta);
		else
			cmeta->atttypkind = TYPE_KIND__NULL;	/* unreferenced */
	}
	if (con->io_index >= 0)
	{
		/* close the last I/O chunks */
		strom_io_chunk *ioc = &con->ioc[con->io_index++];

		ioc->nr_pages = (TYPEALIGN(PAGE_SIZE, con->f_offset) / PAGE_SIZE -
						 ioc->fchunk_id);
		con->m_offset = ioc->m_offset + PAGE_SIZE * ioc->nr_pages;
	}
	kds->length = con->m_offset;

	iovec = palloc0(offsetof(strom_io_vector, ioc[con->io_index]));
	iovec->nr_chunks = con->io_index;
	if (iovec->nr_chunks > 0)
		memcpy(iovec->ioc, con->ioc, sizeof(strom_io_chunk) * con->io_index);
#if 0
	/* for debug - dump the i/o vector */
	{
		elog(INFO, "nchunks = %d", iovec->nr_chunks);
		for (int j=0; j < iovec->nr_chunks; j++)
		{
			strom_io_chunk *ioc = &iovec->ioc[j];

			elog(INFO, "io[%d] [ m_offset=%lu, f_read=%lu...%lu, nr_pages=%u}",
				 j,
				 ioc->m_offset,
				 ioc->fchunk_id * PAGE_SIZE,
				 (ioc->fchunk_id + ioc->nr_pages) * PAGE_SIZE,
				 ioc->nr_pages);
		}

		elog(INFO, "kds {length=%zu nitems=%u typeid=%u typmod=%u table_oid=%u}",
			 kds->length, kds->nitems,
			 kds->tdtypeid, kds->tdtypmod, kds->table_oid);
		for (int j=0; j < kds->nr_colmeta; j++)
		{
			kern_colmeta *cmeta = &kds->colmeta[j];

			elog(INFO, "%ccol[%d] nullmap=%lu,%lu values=%lu,%lu extra=%lu,%lu",
				 j < kds->ncols ? ' ' : '*', j,
				 __kds_unpack(cmeta->nullmap_offset),
				 __kds_unpack(cmeta->nullmap_length),
				 __kds_unpack(cmeta->values_offset),
				 __kds_unpack(cmeta->values_length),
				 __kds_unpack(cmeta->extra_offset),
				 __kds_unpack(cmeta->extra_length));
		}
	}
#endif
	return iovec;
}

/*
 * arrowFdwLoadRecordBatch
 */
static void
__arrowKdsAssignAttrOptions(kern_data_store *kds,
							kern_colmeta *cmeta,
							RecordBatchFieldState *rb_field)
{
	memcpy(&cmeta->attopts,
		   &rb_field->attopts, sizeof(ArrowTypeOptions));
	if (cmeta->atttypkind == TYPE_KIND__ARRAY)
	{
		Assert(cmeta->idx_subattrs >= kds->ncols &&
			   cmeta->num_subattrs == 1 &&
			   cmeta->idx_subattrs + cmeta->num_subattrs <= kds->nr_colmeta &&
			   rb_field->num_children == 1);
		__arrowKdsAssignAttrOptions(kds,
									&kds->colmeta[cmeta->idx_subattrs],
									&rb_field->children[0]);
	}
	else if (cmeta->atttypkind == TYPE_KIND__COMPOSITE)
	{
		Assert(cmeta->idx_subattrs >= kds->ncols &&
			   cmeta->num_subattrs == rb_field->num_children &&
			   cmeta->idx_subattrs + cmeta->num_subattrs <= kds->nr_colmeta);
		for (int j=0; j < cmeta->num_subattrs; j++)
		{
			__arrowKdsAssignAttrOptions(kds,
										&kds->colmeta[cmeta->idx_subattrs + j],
										&rb_field->children[j]);
		}
	}
}

static strom_io_vector *
arrowFdwLoadRecordBatch(Relation relation,
						Bitmapset *referenced,
						RecordBatchState *rb_state,
						StringInfo chunk_buffer)
{
	TupleDesc	tupdesc = RelationGetDescr(relation);
	size_t		head_sz = estimate_kern_data_store(tupdesc);
	kern_data_store *kds;

	/* setup KDS and I/O-vector */
	enlargeStringInfo(chunk_buffer, head_sz);
	kds = (kern_data_store *)(chunk_buffer->data +
							  chunk_buffer->len);
	setup_kern_data_store(kds, tupdesc, 0, KDS_FORMAT_ARROW);
	kds->nitems = rb_state->rb_nitems;
	kds->table_oid = RelationGetRelid(relation);
	Assert(head_sz == KDS_HEAD_LENGTH(kds));
	Assert(kds->ncols == rb_state->nfields);
	for (int j=0; j < kds->ncols; j++)
		__arrowKdsAssignAttrOptions(kds,
									&kds->colmeta[j],
									&rb_state->fields[j]);
	chunk_buffer->len += head_sz;

	return arrowFdwSetupIOvector(rb_state, referenced, kds);
}

static kern_data_store *
arrowFdwFillupRecordBatch(Relation relation,
						  Bitmapset *referenced,
						  RecordBatchState *rb_state,
						  StringInfo chunk_buffer)
{
	ArrowFileState	*af_state = rb_state->af_state;
	kern_data_store	*kds;
	strom_io_vector	*iovec;
	char	   *base;
	File		filp;

	resetStringInfo(chunk_buffer);
	iovec = arrowFdwLoadRecordBatch(relation,
									referenced,
									rb_state,
									chunk_buffer);
	kds = (kern_data_store *)chunk_buffer->data;
	enlargeStringInfo(chunk_buffer, kds->length);
	kds = (kern_data_store *)chunk_buffer->data;
	filp = PathNameOpenFile(af_state->filename, O_RDONLY | PG_BINARY);
	base = (char *)kds + KDS_HEAD_LENGTH(kds);
	for (int i=0; i < iovec->nr_chunks; i++)
	{
		strom_io_chunk *ioc = &iovec->ioc[i];
		char	   *dest = base + ioc->m_offset;
		off_t		f_pos = (size_t)ioc->fchunk_id * PAGE_SIZE;
		size_t		len = (size_t)ioc->nr_pages * PAGE_SIZE;
		ssize_t		sz;

		while (len > 0)
		{
			CHECK_FOR_INTERRUPTS();

			sz = FileRead(filp, dest, len, f_pos,
						  WAIT_EVENT_REORDER_BUFFER_READ);
			if (sz > 0)
			{
				Assert(sz <= len);
				dest  += sz;
				f_pos += sz;
				len   -= sz;
			}
			else if (sz == 0)
			{
				/*
				 * Due to the page_sz alignment, we may try to read the file
				 * over its tail. So, pread(2) may tell us unable to read
				 * any more. The expected scenario happend only when remained
				 * length is less than PAGE_SIZE.
				 */
				memset(dest, 0, len);
				break;
			}
			else if (errno != EINTR)
			{
				assert(false);
				elog(ERROR, "failed on FileRead('%s', pos=%lu, len=%lu): %m",
					 af_state->filename, f_pos, len);
			}
		}
	}
	chunk_buffer->len += kds->length;
	FileClose(filp);

	pfree(iovec);

	return kds;
}

/*
 * ArrowGetForeignRelSize
 */
static size_t
__recordBatchFieldLength(RecordBatchFieldState *rb_field)
{
	size_t		len = 0;

	if (rb_field->null_count > 0)
		len += rb_field->nullmap_length;
	len += (rb_field->values_length +
			rb_field->extra_length);
	for (int j=0; j < rb_field->num_children; j++)
		len += __recordBatchFieldLength(&rb_field->children[j]);
	return len;
}

static void
ArrowGetForeignRelSize(PlannerInfo *root,
					   RelOptInfo *baserel,
					   Oid foreigntableid)
{
	ForeignTable   *ft = GetForeignTable(foreigntableid);
	Relation		frel = table_open(foreigntableid, NoLock);
	List		   *filesList;
	List		   *results = NIL;
	Bitmapset	   *referenced = NULL;
	ListCell	   *lc1, *lc2;
	size_t			totalLen = 0;
	double			ntuples = 0.0;
	int				parallel_nworkers;

	/* columns to be referenced */
	foreach (lc1, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(lc1);

		pull_varattnos((Node *)rinfo->clause, baserel->relid, &referenced);
	}
	referenced = pickup_outer_referenced(root, baserel, referenced);

	/* read arrow-file metadta */
	filesList = arrowFdwExtractFilesList(ft->options, &parallel_nworkers);
	foreach (lc1, filesList)
	{
		ArrowFileState *af_state;
		char	   *fname = strVal(lfirst(lc1));

		af_state = BuildArrowFileState(frel, fname, NULL);
		if (!af_state)
			continue;

		/*
		 * Size calculation based the record-batch metadata
		 */
		foreach (lc2, af_state->rb_list)
		{
			RecordBatchState *rb_state = lfirst(lc2);

			/* whole-row reference? */
			if (bms_is_member(-FirstLowInvalidHeapAttributeNumber, referenced))
			{
				totalLen += rb_state->rb_length;
			}
			else
			{
				int		j, k;

				for (k = bms_next_member(referenced, -1);
					 k >= 0;
					 k = bms_next_member(referenced, k))
				{
					j = k + FirstLowInvalidHeapAttributeNumber;
					if (j <= 0 || j > rb_state->nfields)
						continue;
					totalLen += __recordBatchFieldLength(&rb_state->fields[j-1]);
				}
			}
			ntuples += rb_state->rb_nitems;
		}
		results = lappend(results, af_state);
	}
	table_close(frel, NoLock);

	/* setup baserel */
	baserel->rel_parallel_workers = parallel_nworkers;
	baserel->fdw_private = list_make2(results, referenced);
	baserel->pages = totalLen / BLCKSZ;
	baserel->tuples = ntuples;
	baserel->rows = ntuples *
		clauselist_selectivity(root,
							   baserel->baserestrictinfo,
							   0,
							   JOIN_INNER,
							   NULL);
}

/*
 * cost_arrow_fdw_seqscan
 */
static void
cost_arrow_fdw_seqscan(Path *path,
					   PlannerInfo *root,
					   RelOptInfo *baserel,
					   ParamPathInfo *param_info,
					   int num_workers)
{
	Cost		startup_cost = 0.0;
	Cost		disk_run_cost = 0.0;
	Cost		cpu_run_cost = 0.0;
	QualCost	qcost;
	double		nrows;
	double		spc_seq_page_cost;

	if (param_info)
		nrows = param_info->ppi_rows;
	else
		nrows = baserel->rows;

	/* arrow_fdw.enabled */
	if (!arrow_fdw_enabled)
		startup_cost += disable_cost;

	/*
	 * Storage costs
	 *
	 * XXX - smaller number of columns to read shall have less disk cost
	 * because of columnar format. Right now, we don't discount cost for
	 * the pages not to be read.
	 */
	get_tablespace_page_costs(baserel->reltablespace,
							  NULL,
							  &spc_seq_page_cost);
	disk_run_cost = spc_seq_page_cost * baserel->pages;

	/* CPU costs */
	if (param_info)
	{
		cost_qual_eval(&qcost, param_info->ppi_clauses, root);
		qcost.startup += baserel->baserestrictcost.startup;
        qcost.per_tuple += baserel->baserestrictcost.per_tuple;
	}
	else
		qcost = baserel->baserestrictcost;
	startup_cost += qcost.startup;
	cpu_run_cost = (cpu_tuple_cost + qcost.per_tuple) * baserel->tuples;

	/* tlist evaluation costs */
	startup_cost += path->pathtarget->cost.startup;
	cpu_run_cost += path->pathtarget->cost.per_tuple * path->rows;

	/* adjust cost for CPU parallelism */
	if (num_workers > 0)
	{
		double		leader_contribution;
		double		parallel_divisor = (double) num_workers;

		/* see get_parallel_divisor() */
		leader_contribution = 1.0 - (0.3 * (double)num_workers);
		parallel_divisor += Max(leader_contribution, 0.0);

		/* The CPU cost is divided among all the workers. */
		cpu_run_cost /= parallel_divisor;

		/* Estimated row count per background worker process */
		nrows = clamp_row_est(nrows / parallel_divisor);
	}
	path->rows = nrows;
	path->startup_cost = startup_cost;
	path->total_cost = startup_cost + cpu_run_cost + disk_run_cost;
	path->parallel_workers = num_workers;
}

/*
 * ArrowGetForeignPaths
 */
static void
ArrowGetForeignPaths(PlannerInfo *root,
                     RelOptInfo *baserel,
                     Oid foreigntableid)
{
	ForeignPath	   *fpath;
	ParamPathInfo  *param_info;
	Relids			required_outer = baserel->lateral_relids;

	param_info = get_baserel_parampathinfo(root, baserel, required_outer);
	fpath = create_foreignscan_path(root,
									baserel,
									NULL,	/* default pathtarget */
									-1.0,	/* dummy */
									-1.0,	/* dummy */
									-1.0,	/* dummy */
									NIL,	/* no pathkeys */
									required_outer,
									NULL,	/* no extra plan */
									NIL);	/* no particular private */
	cost_arrow_fdw_seqscan(&fpath->path,
						   root,
						   baserel,
						   param_info, 0);
	add_path(baserel, &fpath->path);

	if (baserel->consider_parallel)
	{
		int		num_workers =
			compute_parallel_worker(baserel,
									baserel->pages, -1.0,
									max_parallel_workers_per_gather);
		if (num_workers == 0)
			return;

		fpath = create_foreignscan_path(root,
										baserel,
										NULL,	/* default pathtarget */
										-1.0,	/* dummy */
										-1.0,	/* dummy */
										-1.0,	/* dummy */
										NIL,	/* no pathkeys */
										required_outer,
										NULL,	/* no extra plan */
										NIL);	/* no particular private */
		fpath->path.parallel_aware = true;
		cost_arrow_fdw_seqscan(&fpath->path,
							   root,
							   baserel,
							   param_info,
							   num_workers);
		add_partial_path(baserel, (Path *)fpath);
	}
}

/*
 * ArrowGetForeignPlan
 */
static ForeignScan *
ArrowGetForeignPlan(PlannerInfo *root,
					RelOptInfo *baserel,
					Oid foreigntableid,
					ForeignPath *best_path,
					List *tlist,
					List *scan_clauses,
					Plan *outer_plan)
{
	Bitmapset  *referenced = lsecond(baserel->fdw_private);
	List	   *ref_list = NIL;
	int			k;

	for (k = bms_next_member(referenced, -1);
		 k >= 0;
		 k = bms_next_member(referenced, k))
	{
		ref_list = lappend_int(ref_list, k);
	}
	return make_foreignscan(tlist,
							extract_actual_clauses(scan_clauses, false),
							baserel->relid,
							NIL,	/* no expressions to evaluate */
							ref_list, /* list of referenced attnums */
							NIL,	/* no custom tlist */
							NIL,	/* no remote quals */
							outer_plan);
}

/* ----------------------------------------------------------------
 *
 * Routines related to Arrow datum fetch
 *
 * ----------------------------------------------------------------
 */
static void		pg_datum_arrow_ref(kern_data_store *kds,
								   kern_colmeta *cmeta,
								   size_t index,
								   Datum *p_datum,
								   bool *p_isnull);

static Datum
pg_varlena32_arrow_ref(kern_data_store *kds,
					   kern_colmeta *cmeta, size_t index)
{
	uint32_t   *offset = (uint32_t *)((char *)kds +
									  __kds_unpack(cmeta->values_offset));
	char	   *extra = (char *)kds + __kds_unpack(cmeta->extra_offset);
	uint32_t	len;
	struct varlena *res;

	if (sizeof(uint32_t) * (index+2) > __kds_unpack(cmeta->values_length))
		elog(ERROR, "corruption? varlena index out of range");
	len = offset[index+1] - offset[index];
	if (offset[index] > offset[index+1] ||
		offset[index+1] > __kds_unpack(cmeta->extra_length))
		elog(ERROR, "corruption? varlena points out of extra buffer");
	if (len >= (1UL<<VARLENA_EXTSIZE_BITS) - VARHDRSZ)
		elog(ERROR, "variable size too large");
	res = palloc(VARHDRSZ + len);
	SET_VARSIZE(res, VARHDRSZ + len);
	memcpy(VARDATA(res), extra + offset[index], len);

	return PointerGetDatum(res);
}

static Datum
pg_varlena64_arrow_ref(kern_data_store *kds,
					   kern_colmeta *cmeta, size_t index)
{
	uint64_t   *offset = (uint64_t *)((char *)kds +
									  __kds_unpack(cmeta->values_offset));
	char	   *extra = (char *)kds + __kds_unpack(cmeta->extra_offset);
	uint64_t	len;
	struct varlena *res;

	if (sizeof(uint64_t) * (index+2) > __kds_unpack(cmeta->values_length))
		elog(ERROR, "corruption? varlena index out of range");
	len = offset[index+1] - offset[index];
	if (offset[index] > offset[index+1] ||
		offset[index+1] > __kds_unpack(cmeta->extra_length))
		elog(ERROR, "corruption? varlena points out of extra buffer");
	if (len >= (1UL<<VARLENA_EXTSIZE_BITS) - VARHDRSZ)
		elog(ERROR, "variable size too large");
	res = palloc(VARHDRSZ + len);
	SET_VARSIZE(res, VARHDRSZ + len);
	memcpy(VARDATA(res), extra + offset[index], len);

	return PointerGetDatum(res);
}

static Datum
pg_bpchar_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	char	   *values = ((char *)kds + __kds_unpack(cmeta->values_offset));
	size_t		length = __kds_unpack(cmeta->values_length);
	int32_t		unitsz = cmeta->attopts.fixed_size_binary.byteWidth;
	struct varlena *res;

	if (unitsz <= 0)
		elog(ERROR, "CHAR(%d) is not expected", unitsz);
	if (unitsz * index >= length)
		elog(ERROR, "corruption? bpchar points out of range");
	res = palloc(VARHDRSZ + unitsz);
	memcpy((char *)res + VARHDRSZ, values + unitsz * index, unitsz);
	SET_VARSIZE(res, VARHDRSZ + unitsz);

	return PointerGetDatum(res);
}

static Datum
pg_bool_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	uint8_t	   *bitmap = (uint8_t *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	uint8_t		mask = (1 << (index & 7));

	if (sizeof(uint8_t) * index >= length)
		elog(ERROR, "corruption? bool points out of range");
	return BoolGetDatum((bitmap[index>>3] & mask) != 0 ? true : false);
}

static Datum
pg_simple_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	int32_t		unitsz = cmeta->attopts.unitsz;
	char	   *values = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	Datum		retval = 0;

	Assert(unitsz > 0 && unitsz <= sizeof(Datum));
	if (unitsz * index >= length)
		elog(ERROR, "corruption? simple int8 points out of range");
	memcpy(&retval, values + unitsz * index, unitsz);
	return retval;
}

static Datum
pg_numeric_arrow_ref(kern_data_store *kds,
					 kern_colmeta *cmeta, size_t index)
{
	char	   *result = palloc0(sizeof(struct NumericData));
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	int			dscale = cmeta->attopts.decimal.scale;
	int128_t	ival;

	if (sizeof(int128_t) * index >= length)
		elog(ERROR, "corruption? numeric points out of range");
	ival = ((int128_t *)base)[index];
	__xpu_numeric_to_varlena(result, dscale, ival);

	return PointerGetDatum(result);
}

static Datum
pg_date_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	DateADT		dt;

	switch (cmeta->attopts.date.unit)
	{
		case ArrowDateUnit__Day:
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Date[day] points out of range");
			dt = ((uint32 *)base)[index];
			break;
		case ArrowDateUnit__MilliSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Date[ms] points out of range");
			dt = ((uint64 *)base)[index] / 1000;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Date type");
	}
	/* convert UNIX epoch to PostgreSQL epoch */
	dt -= (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
	return DateADTGetDatum(dt);
}

static Datum
pg_time_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	TimeADT		tm;

	switch (cmeta->attopts.time.unit)
	{
		case ArrowTimeUnit__Second:
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Time[sec] points out of range");
			tm = ((uint32 *)base)[index] * 1000000L;
			break;
		case ArrowTimeUnit__MilliSecond:
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Time[ms] points out of range");
			tm = ((uint32 *)base)[index] * 1000L;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Time[us] points out of range");
			tm = ((uint64 *)base)[index];
			break;
		case ArrowTimeUnit__NanoSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Time[ns] points out of range");
			tm = ((uint64 *)base)[index] / 1000L;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Time type");
			break;
	}
	return TimeADTGetDatum(tm);
}

static Datum
pg_timestamp_arrow_ref(kern_data_store *kds,
					   kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	Timestamp	ts;

	switch (cmeta->attopts.timestamp.unit)
	{
		case ArrowTimeUnit__Second:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[sec] points out of range");
			ts = ((uint64 *)base)[index] * 1000000UL;
			break;
		case ArrowTimeUnit__MilliSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[ms] points out of range");
			ts = ((uint64 *)base)[index] * 1000UL;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[us] points out of range");
			ts = ((uint64 *)base)[index];
			break;
		case ArrowTimeUnit__NanoSecond:
			if (sizeof(uint64) * index >= length)
				elog(ERROR, "corruption? Timestamp[ns] points out of range");
			ts = ((uint64 *)base)[index] / 1000UL;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Timestamp type");
			break;
	}
	/* convert UNIX epoch to PostgreSQL epoch */
	ts -= (POSTGRES_EPOCH_JDATE -
		   UNIX_EPOCH_JDATE) * USECS_PER_DAY;
	return TimestampGetDatum(ts);
}

static Datum
pg_interval_arrow_ref(kern_data_store *kds,
					  kern_colmeta *cmeta, size_t index)
{
	char	   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t		length = __kds_unpack(cmeta->values_length);
	Interval   *iv = palloc0(sizeof(Interval));

	switch (cmeta->attopts.interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			/* 32bit: number of months */
			if (sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Interval[Year/Month] points out of range");
			iv->month = ((uint32 *)base)[index];
			break;
		case ArrowIntervalUnit__Day_Time:
			/* 32bit+32bit: number of days and milliseconds */
			if (2 * sizeof(uint32) * index >= length)
				elog(ERROR, "corruption? Interval[Day/Time] points out of range");
			iv->day  = ((int32 *)base)[2 * index];
			iv->time = ((int32 *)base)[2 * index + 1] * 1000;
			break;
		default:
			elog(ERROR, "Bug? unexpected unit of Interval type");
	}
	return PointerGetDatum(iv);
}

static Datum
pg_macaddr_arrow_ref(kern_data_store *kds,
					 kern_colmeta *cmeta, size_t index)
{
	char   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t	length = __kds_unpack(cmeta->values_length);

	if (cmeta->attopts.fixed_size_binary.byteWidth != sizeof(macaddr))
		elog(ERROR, "Bug? wrong FixedSizeBinary::byteWidth(%d) for macaddr",
			 cmeta->attopts.fixed_size_binary.byteWidth);
	if (sizeof(macaddr) * index >= length)
		elog(ERROR, "corruption? Binary[macaddr] points out of range");

	return PointerGetDatum(base + sizeof(macaddr) * index);
}

static Datum
pg_inet_arrow_ref(kern_data_store *kds,
				  kern_colmeta *cmeta, size_t index)
{
	char   *base = (char *)kds + __kds_unpack(cmeta->values_offset);
	size_t	length = __kds_unpack(cmeta->values_length);
	inet   *ip = palloc(sizeof(inet));

	if (cmeta->attopts.fixed_size_binary.byteWidth == 4)
	{
		if (4 * index >= length)
			elog(ERROR, "corruption? Binary[inet4] points out of range");
		ip->inet_data.family = PGSQL_AF_INET;
		ip->inet_data.bits = 32;
		memcpy(ip->inet_data.ipaddr, base + 4 * index, 4);
	}
	else if (cmeta->attopts.fixed_size_binary.byteWidth == 16)
	{
		if (16 * index >= length)
			elog(ERROR, "corruption? Binary[inet6] points out of range");
		ip->inet_data.family = PGSQL_AF_INET6;
		ip->inet_data.bits = 128;
		memcpy(ip->inet_data.ipaddr, base + 16 * index, 16);
	}
	else
		elog(ERROR, "Bug? wrong FixedSizeBinary::byteWidth(%d) for inet",
			 cmeta->attopts.fixed_size_binary.byteWidth);

	SET_INET_VARSIZE(ip);
	return PointerGetDatum(ip);
}

static Datum
pg_array_arrow_ref(kern_data_store *kds,
				   kern_colmeta *smeta,
				   uint32_t start, uint32_t end)
{
	ArrayType  *res;
	size_t		sz;
	uint32_t	i, nitems = end - start;
	bits8	   *nullmap = NULL;
	size_t		usage, __usage;

	/* sanity checks */
	if (start > end)
		elog(ERROR, "Bug? array index has reversed order [%u..%u]", start, end);

	/* allocation of the result buffer */
	if (smeta->nullmap_offset != 0)
		sz = ARR_OVERHEAD_WITHNULLS(1, nitems);
	else
		sz = ARR_OVERHEAD_NONULLS(1);

	if (smeta->attlen > 0)
	{
		sz += TYPEALIGN(smeta->attalign,
						smeta->attlen) * nitems;
	}
	else if (smeta->attlen == -1)
	{
		sz += 400;		/* tentative allocation */
	}
	else
		elog(ERROR, "Bug? corrupted kernel column metadata");

	res = palloc0(sz);
	res->ndim = 1;
	if (smeta->nullmap_offset != 0)
	{
		res->dataoffset = ARR_OVERHEAD_WITHNULLS(1, nitems);
		nullmap = ARR_NULLBITMAP(res);
	}
	res->elemtype = smeta->atttypid;
	ARR_DIMS(res)[0] = nitems;
	ARR_LBOUND(res)[0] = 1;
	usage = ARR_DATA_OFFSET(res);
	for (i=0; i < nitems; i++)
	{
		Datum	datum;
		bool	isnull;

		pg_datum_arrow_ref(kds, smeta, start+i, &datum, &isnull);
		if (isnull)
		{
			if (!nullmap)
				elog(ERROR, "Bug? element item should not be NULL");
		}
		else if (smeta->attlen > 0)
		{
			if (nullmap)
				nullmap[i>>3] |= (1<<(i&7));
			__usage = TYPEALIGN(smeta->attalign, usage);
			while (__usage + smeta->attlen > sz)
			{
				sz += sz;
				res = repalloc(res, sz);
			}
			if (__usage > usage)
				memset((char *)res + usage, 0, __usage - usage);
			memcpy((char *)res + __usage, &datum, smeta->attlen);
			usage = __usage + smeta->attlen;
		}
		else if (smeta->attlen == -1)
		{
			int32_t		vl_len = VARSIZE(datum);

			if (nullmap)
				nullmap[i>>3] |= (1<<(i&7));
			__usage = TYPEALIGN(smeta->attalign, usage);
			while (__usage + vl_len > sz)
			{
				sz += sz;
				res = repalloc(res, sz);
			}
			if (__usage > usage)
				memset((char *)res + usage, 0, __usage - usage);
			memcpy((char *)res + __usage, DatumGetPointer(datum), vl_len);
			usage = __usage + vl_len;

			pfree(DatumGetPointer(datum));
		}
		else
			elog(ERROR, "Bug? corrupted kernel column metadata");
	}
	SET_VARSIZE(res, usage);

	return PointerGetDatum(res);
}

/*
 * pg_datum_arrow_ref
 */
static void
pg_datum_arrow_ref(kern_data_store *kds,
				   kern_colmeta *cmeta,
				   size_t index,
				   Datum *p_datum,
				   bool *p_isnull)
{
	Datum		datum = 0;
	bool		isnull = false;

	if (cmeta->nullmap_offset != 0)
	{
		size_t	nullmap_offset = __kds_unpack(cmeta->nullmap_offset);
		uint8  *nullmap = (uint8 *)kds + nullmap_offset;

		if (att_isnull(index, nullmap))
		{
			isnull = true;
			goto out;
		}
	}
	
	switch (cmeta->attopts.tag)
	{
		case ArrowType__Int:
		case ArrowType__FloatingPoint:
			datum = pg_simple_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__Bool:
			datum = pg_bool_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__Decimal:
			datum = pg_numeric_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__Date:
			datum = pg_date_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__Time:
			datum = pg_time_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__Timestamp:
			datum = pg_timestamp_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__Interval:
			datum = pg_interval_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__Utf8:
		case ArrowType__Binary:
			datum = pg_varlena32_arrow_ref(kds, cmeta, index);
			break;
		case ArrowType__LargeUtf8:
		case ArrowType__LargeBinary:
			datum = pg_varlena64_arrow_ref(kds, cmeta, index);
			break;

		case ArrowType__FixedSizeBinary:
			switch (cmeta->atttypid)
			{
				case MACADDROID:
					datum = pg_macaddr_arrow_ref(kds, cmeta, index);
					break;
				case INETOID:
					datum = pg_inet_arrow_ref(kds, cmeta, index);
					break;
				case BPCHAROID:
					datum = pg_bpchar_arrow_ref(kds, cmeta, index);
					break;
				default:
					elog(ERROR, "unknown FixedSizeBinary mapping");
					break;
			}
			break;
			
		case ArrowType__List:
			{
				kern_colmeta   *smeta;
				uint32_t	   *offset;

				if (cmeta->num_subattrs != 1 ||
					cmeta->idx_subattrs < kds->ncols ||
					cmeta->idx_subattrs >= kds->nr_colmeta)
					elog(ERROR, "Bug? corrupted kernel column metadata");
				if (sizeof(uint32_t) * (index+2) > __kds_unpack(cmeta->values_length))
					elog(ERROR, "Bug? array index is out of range");
				smeta = &kds->colmeta[cmeta->idx_subattrs];
				offset = (uint32_t *)((char *)kds + __kds_unpack(cmeta->values_offset));
				datum = pg_array_arrow_ref(kds, smeta,
										   offset[index],
										   offset[index+1]);
				isnull = false;
			}
			break;

		case ArrowType__LargeList:
			{
				kern_colmeta   *smeta;
				uint64_t	   *offset;

				if (cmeta->num_subattrs != 1 ||
					cmeta->idx_subattrs < kds->ncols ||
					cmeta->idx_subattrs >= kds->nr_colmeta)
					elog(ERROR, "Bug? corrupted kernel column metadata");
				if (sizeof(uint64_t) * (index+2) > __kds_unpack(cmeta->values_length))
					elog(ERROR, "Bug? array index is out of range");
				smeta = &kds->colmeta[cmeta->idx_subattrs];
				offset = (uint64_t *)((char *)kds + __kds_unpack(cmeta->values_offset));
				datum = pg_array_arrow_ref(kds, smeta,
										   offset[index],
										   offset[index+1]);
				isnull = false;
			}
			break;

		case ArrowType__Struct:
			{
				TupleDesc	tupdesc = lookup_rowtype_tupdesc(cmeta->atttypid, -1);
				Datum	   *sub_values = alloca(sizeof(Datum) * tupdesc->natts);
				bool	   *sub_isnull = alloca(sizeof(bool)  * tupdesc->natts);
				HeapTuple	htup;

				if (tupdesc->natts != cmeta->num_subattrs)
					elog(ERROR, "Struct definition is conrrupted?");
				if (cmeta->idx_subattrs < kds->ncols ||
					cmeta->idx_subattrs + cmeta->num_subattrs > kds->nr_colmeta)
					elog(ERROR, "Bug? strange kernel column metadata");
				for (int j=0; j < tupdesc->natts; j++)
				{
					kern_colmeta *sub_meta = &kds->colmeta[cmeta->idx_subattrs + j];

					pg_datum_arrow_ref(kds, sub_meta, index,
									   sub_values + j,
									   sub_isnull + j);
				}
				htup = heap_form_tuple(tupdesc, sub_values, sub_isnull);

				ReleaseTupleDesc(tupdesc);

				datum = PointerGetDatum(htup->t_data);
				isnull = false;
			}
			break;
		default:
			/* TODO: custom data type support here */
			elog(ERROR, "arrow_fdw: unknown or unsupported type");
	}
out:
	*p_datum  = datum;
	*p_isnull = isnull;
}

/*
 * KDS_fetch_tuple_arrow
 */
bool
kds_arrow_fetch_tuple(TupleTableSlot *slot,
					  kern_data_store *kds,
					  size_t index,
					  const Bitmapset *referenced)
{
	int		j, k;

	if (index >= kds->nitems)
		return false;
	ExecStoreAllNullTuple(slot);
	for (k = bms_next_member(referenced, -1);
		 k >= 0;
		 k = bms_next_member(referenced, k))
	{
		j = k + FirstLowInvalidHeapAttributeNumber - 1;
		if (j < 0)
			continue;
		pg_datum_arrow_ref(kds,
						   &kds->colmeta[j],
						   index,
						   slot->tts_values + j,
						   slot->tts_isnull + j);
	}
	return true;
}

/* ----------------------------------------------------------------
 *
 * Executor callbacks
 *
 * ----------------------------------------------------------------
 */

/*
 * __arrowFdwExecInit
 */
static ArrowFdwState *
__arrowFdwExecInit(ScanState *ss,
				   List *outer_quals,
				   const Bitmapset *outer_refs,
				   const Bitmapset **p_optimal_gpus,
				   const DpuStorageEntry **p_ds_entry)
{
	Relation		frel = ss->ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(frel));
	Bitmapset	   *referenced = NULL;
	Bitmapset	   *stat_attrs = NULL;
	Bitmapset	   *optimal_gpus = NULL;
	DpuStorageEntry *ds_entry = NULL;
	bool			whole_row_ref = false;
	List		   *filesList;
	List		   *af_states_list = NIL;
	uint32_t		rb_nrooms = 0;
	uint32_t		rb_nitems = 0;
	ArrowFdwState *arrow_state;
	ListCell	   *lc1, *lc2;

	Assert(RelationIsArrowFdw(frel));
	/* expand 'referenced' if it has whole-row reference */
	if (bms_is_member(-FirstLowInvalidHeapAttributeNumber, outer_refs))
		whole_row_ref = true;
	for (int j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);
		int		k = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		if (attr->attisdropped)
			continue;
		if (whole_row_ref || bms_is_member(k, outer_refs))
			referenced = bms_add_member(referenced, k);
	}

	/* setup ArrowFileState */
	filesList = arrowFdwExtractFilesList(ft->options, NULL);
	foreach (lc1, filesList)
	{
		char	   *fname = strVal(lfirst(lc1));
		ArrowFileState *af_state;

		af_state = BuildArrowFileState(frel, fname, &stat_attrs);
		if (af_state)
		{
			rb_nrooms += list_length(af_state->rb_list);
			if (p_optimal_gpus)
			{
				const Bitmapset  *__optimal_gpus = GetOptimalGpuForFile(fname);

				if (af_states_list == NIL)
					optimal_gpus = bms_copy(__optimal_gpus);
				else
					optimal_gpus = bms_intersect(optimal_gpus, __optimal_gpus);
			}
			if (p_ds_entry)
			{
				DpuStorageEntry *ds_temp;

				if (af_states_list == NIL)
					ds_entry = GetOptimalDpuForFile(fname, &af_state->dpu_path);
				else if (ds_entry)
				{
					ds_temp = GetOptimalDpuForFile(fname, &af_state->dpu_path);
					if (!DpuStorageEntryIsEqual(ds_entry, ds_temp))
						ds_entry = NULL;
				}
			}
			af_states_list = lappend(af_states_list, af_state);
		}
	}

	/* setup ArrowFdwState */
	arrow_state = palloc0(offsetof(ArrowFdwState, rb_states[rb_nrooms]));
	arrow_state->referenced = referenced;
	if (arrow_fdw_stats_hint_enabled)
		arrow_state->stats_hint = execInitArrowStatsHint(ss, outer_quals, stat_attrs);
	arrow_state->rbatch_index = &arrow_state->__rbatch_index_local;
	arrow_state->rbatch_nload = &arrow_state->__rbatch_nload_local;
	arrow_state->rbatch_nskip = &arrow_state->__rbatch_nskip_local;
	initStringInfo(&arrow_state->chunk_buffer);
	arrow_state->curr_filp  = -1;
	arrow_state->curr_kds   = NULL;
	arrow_state->curr_index = 0;
	arrow_state->af_states_list = af_states_list;
	foreach (lc1, af_states_list)
	{
		ArrowFileState *af_state = lfirst(lc1);

		foreach (lc2, af_state->rb_list)
		{
			RecordBatchState *rb_state = lfirst(lc2);

			arrow_state->rb_states[rb_nitems++] = rb_state;
		}
	}
	Assert(rb_nrooms == rb_nitems);
	arrow_state->rb_nitems = rb_nitems;

	if (p_optimal_gpus)
		*p_optimal_gpus = optimal_gpus;
	if (p_ds_entry)
		*p_ds_entry = ds_entry;

	return arrow_state;
}

/*
 * pgstromArrowFdwExecInit
 */
bool
pgstromArrowFdwExecInit(pgstromTaskState *pts,
						uint64_t devkind_mask,
						List *outer_quals,
						const Bitmapset *outer_refs)
{
	Relation		frel = pts->css.ss.ss_currentRelation;
	ArrowFdwState  *arrow_state = NULL;

	if (RelationIsArrowFdw(frel))
	{
		arrow_state = __arrowFdwExecInit(&pts->css.ss,
										 outer_quals,
										 outer_refs,
										 (devkind_mask & DEVKIND__NVIDIA_GPU) != 0
											? &pts->optimal_gpus : NULL,
										 (devkind_mask & DEVKIND__NVIDIA_DPU) != 0
											? &pts->ds_entry : NULL);
	}
	pts->arrow_state = arrow_state;
	return (pts->arrow_state != NULL);
}

/*
 * ArrowBeginForeignScan
 */
static void
ArrowBeginForeignScan(ForeignScanState *node, int eflags)
{
	ForeignScan	   *fscan = (ForeignScan *)node->ss.ps.plan;
	Bitmapset	   *referenced = NULL;
	ListCell	   *lc;

	foreach (lc, fscan->fdw_private)
	{
		int		k = lfirst_int(lc);

		referenced = bms_add_member(referenced, k);
	}
	node->fdw_state = __arrowFdwExecInit(&node->ss,
										 fscan->scan.plan.qual,
										 referenced,
										 NULL,	/* no GPU */
										 NULL);	/* no DPU */
}

/*
 * ExecArrowScanChunk
 */
static inline RecordBatchState *
__arrowFdwNextRecordBatch(ArrowFdwState *arrow_state)
{
	RecordBatchState *rb_state;
	uint32_t	rb_index;

retry:
	rb_index = pg_atomic_fetch_add_u32(arrow_state->rbatch_index, 1);
	if (rb_index >= arrow_state->rb_nitems)
		return NULL;	/* no more chunks to load */
	rb_state = arrow_state->rb_states[rb_index];
	if (arrow_state->stats_hint)
	{
		if (execCheckArrowStatsHint(arrow_state->stats_hint, rb_state))
		{
			pg_atomic_fetch_add_u32(arrow_state->rbatch_nskip, 1);
			goto retry;
		}
		pg_atomic_fetch_add_u32(arrow_state->rbatch_nload, 1);
	}
	return rb_state;
}

/*
 * pgstromScanChunkArrowFdw
 */
XpuCommand *
pgstromScanChunkArrowFdw(pgstromTaskState *pts,
						 struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	ArrowFdwState  *arrow_state = pts->arrow_state;
	StringInfo		chunk_buffer = &arrow_state->chunk_buffer;
	RecordBatchState *rb_state;
	ArrowFileState *af_state;
	strom_io_vector *iovec;
	XpuCommand	   *xcmd;
	uint32_t		kds_src_offset;
	uint32_t		kds_src_iovec;
	uint32_t		kds_src_pathname;

	rb_state = __arrowFdwNextRecordBatch(arrow_state);
	if (!rb_state)
		return NULL;
	af_state = rb_state->af_state;

	/* XpuCommand header */
	resetStringInfo(chunk_buffer);
	appendBinaryStringInfo(chunk_buffer,
						   pts->xcmd_buf.data,
						   pts->xcmd_buf.len);
	/* kds_src + iovec */
	kds_src_offset = chunk_buffer->len;
	iovec = arrowFdwLoadRecordBatch(pts->css.ss.ss_currentRelation,
									arrow_state->referenced,
									rb_state,
									chunk_buffer);
	kds_src_iovec = __appendBinaryStringInfo(chunk_buffer,
											 iovec,
											 offsetof(strom_io_vector,
													  ioc[iovec->nr_chunks]));
	/* arrow filename */
	kds_src_pathname = chunk_buffer->len;
	if (!pts->ds_entry)
		appendStringInfoString(chunk_buffer, af_state->filename);
	else
		appendStringInfoString(chunk_buffer, af_state->dpu_path);
	appendStringInfoChar(chunk_buffer, '\0');

	/* assign offset of XpuCommand */
	xcmd = (XpuCommand *)chunk_buffer->data;
	xcmd->length = chunk_buffer->len;
	xcmd->u.scan.kds_src_fullpath = kds_src_pathname;
	xcmd->u.scan.kds_src_pathname = kds_src_pathname;
	xcmd->u.scan.kds_src_iovec    = kds_src_iovec;
	xcmd->u.scan.kds_src_offset   = kds_src_offset;

	xcmd_iov->iov_base = xcmd;
	xcmd_iov->iov_len  = xcmd->length;
	*xcmd_iovcnt = 1;

	return xcmd;
}

/*
 * ArrowIterateForeignScan
 */
static TupleTableSlot *
ArrowIterateForeignScan(ForeignScanState *node)
{
	ArrowFdwState *arrow_state = node->fdw_state;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	kern_data_store *kds;

	while ((kds = arrow_state->curr_kds) == NULL ||
		   arrow_state->curr_index >= kds->nitems)
	{
		RecordBatchState *rb_state;

		arrow_state->curr_index = 0;
		arrow_state->curr_kds = NULL;
		rb_state = __arrowFdwNextRecordBatch(arrow_state);
		if (!rb_state)
			return NULL;
		arrow_state->curr_kds
			= arrowFdwFillupRecordBatch(node->ss.ss_currentRelation,
										arrow_state->referenced,
										rb_state,
										&arrow_state->chunk_buffer);
	}
	Assert(kds && arrow_state->curr_index < kds->nitems);
	if (kds_arrow_fetch_tuple(slot, kds,
							  arrow_state->curr_index++,
							  arrow_state->referenced))
		return slot;
	return NULL;
}

/*
 * ArrowReScanForeignScan
 */
void
pgstromArrowFdwExecReset(ArrowFdwState *arrow_state)
{
	pg_atomic_write_u32(arrow_state->rbatch_index, 0);
	if (arrow_state->curr_kds)
		pfree(arrow_state->curr_kds);
	arrow_state->curr_kds = NULL;
	arrow_state->curr_index = 0;
}

static void
ArrowReScanForeignScan(ForeignScanState *node)
{
	pgstromArrowFdwExecReset(node->fdw_state);
}

/*
 * ExecEndArrowScan
 */
void
pgstromArrowFdwExecEnd(ArrowFdwState *arrow_state)
{
	if (arrow_state->curr_filp >= 0)
		FileClose(arrow_state->curr_filp);
	if (arrow_state->stats_hint)
		execEndArrowStatsHint(arrow_state->stats_hint);
}

static void
ArrowEndForeignScan(ForeignScanState *node)
{
	pgstromArrowFdwExecEnd(node->fdw_state);
}

/*
 * ArrowIsForeignScanParallelSafe
 */
static bool
ArrowIsForeignScanParallelSafe(PlannerInfo *root,
							   RelOptInfo *rel,
							   RangeTblEntry *rte)
{
	return true;
}

/*
 * ArrowEstimateDSMForeignScan
 */
static Size
ArrowEstimateDSMForeignScan(ForeignScanState *node,
							ParallelContext *pcxt)
{
	return offsetof(pgstromSharedState, bpscan);
}

/*
 * ArrowInitializeDSMForeignScan
 */
void
pgstromArrowFdwInitDSM(ArrowFdwState *arrow_state,
					   pgstromSharedState *ps_state)
{
	arrow_state->rbatch_index = &ps_state->arrow_rbatch_index;
	arrow_state->rbatch_nload = &ps_state->arrow_rbatch_nload;
	arrow_state->rbatch_nskip = &ps_state->arrow_rbatch_nskip;
}

static void
ArrowInitializeDSMForeignScan(ForeignScanState *node,
                              ParallelContext *pcxt,
                              void *coordinate)
{
	pgstromSharedState *ps_state = (pgstromSharedState *)coordinate;

	memset(ps_state, 0, offsetof(pgstromSharedState, bpscan));
	pgstromArrowFdwInitDSM(node->fdw_state, ps_state);
}

/*
 * ArrowInitializeWorkerForeignScan
 */
void
pgstromArrowFdwAttachDSM(ArrowFdwState *arrow_state,
						 pgstromSharedState *ps_state)
{
	arrow_state->rbatch_index = &ps_state->arrow_rbatch_index;
	arrow_state->rbatch_nload = &ps_state->arrow_rbatch_nload;
	arrow_state->rbatch_nskip = &ps_state->arrow_rbatch_nskip;
}

static void
ArrowInitializeWorkerForeignScan(ForeignScanState *node,
								 shm_toc *toc,
								 void *coordinate)
{
	pgstromSharedState *ps_state = (pgstromSharedState *)coordinate;

	pgstromArrowFdwAttachDSM(node->fdw_state, ps_state);
}

/*
 * ArrowShutdownForeignScan
 */
void
pgstromArrowFdwShutdown(ArrowFdwState *arrow_state)
{
	uint32		temp;

	temp = pg_atomic_read_u32(arrow_state->rbatch_index);
	pg_atomic_write_u32(&arrow_state->__rbatch_index_local, temp);
	arrow_state->rbatch_index = &arrow_state->__rbatch_index_local;

	temp = pg_atomic_read_u32(arrow_state->rbatch_nload);
	pg_atomic_write_u32(&arrow_state->__rbatch_nload_local, temp);
	arrow_state->rbatch_nload = &arrow_state->__rbatch_nload_local;

	temp = pg_atomic_read_u32(arrow_state->rbatch_nskip);
	pg_atomic_write_u32(&arrow_state->__rbatch_nskip_local, temp);
	arrow_state->rbatch_nskip = &arrow_state->__rbatch_nskip_local;

}

static void
ArrowShutdownForeignScan(ForeignScanState *node)
{
	pgstromArrowFdwShutdown(node->fdw_state);
}

/*
 * ArrowExplainForeignScan
 */
void
pgstromArrowFdwExplain(ArrowFdwState *arrow_state,
					   Relation frel,
					   ExplainState *es,
					   List *dcontext)
{
	TupleDesc	tupdesc = RelationGetDescr(frel);
	size_t	   *chunk_sz;
	ListCell   *lc1, *lc2;
	int			fcount = 0;
	int			j, k;
	char		label[100];
	StringInfoData	buf;

	initStringInfo(&buf);
	/* shows referenced columns */
	for (k = bms_next_member(arrow_state->referenced, -1);
		 k >= 0;
		 k = bms_next_member(arrow_state->referenced, k))
	{
		j = k + FirstLowInvalidHeapAttributeNumber;

		if (j > 0)
		{
			Form_pg_attribute attr = TupleDescAttr(tupdesc, j-1);
			const char	   *attname = NameStr(attr->attname);

			if (buf.len > 0)
				appendStringInfoString(&buf, ", ");
			appendStringInfoString(&buf, quote_identifier(attname));
		}
	}
	ExplainPropertyText("referenced", buf.data, es);

	/* shows stats hint if any */
	if (arrow_state->stats_hint)
	{
		arrowStatsHint *stats_hint = arrow_state->stats_hint;

		resetStringInfo(&buf);
		foreach (lc1, stats_hint->orig_quals)
		{
			Node   *qual = lfirst(lc1);
			char   *temp;

			temp = deparse_expression(qual, dcontext, es->verbose, false);
			if (buf.len > 0)
				appendStringInfoString(&buf, ", ");
			appendStringInfoString(&buf, temp);
			pfree(temp);
		}
		if (es->analyze)
			appendStringInfo(&buf, "  [loaded: %u, skipped: %u]",
							 pg_atomic_read_u32(arrow_state->rbatch_nload),
							 pg_atomic_read_u32(arrow_state->rbatch_nskip));
		ExplainPropertyText("Stats-Hint", buf.data, es);
	}

	/* shows files on behalf of the foreign table */
	chunk_sz = alloca(sizeof(size_t) * tupdesc->natts);
	memset(chunk_sz, 0, sizeof(size_t) * tupdesc->natts);
	foreach (lc1, arrow_state->af_states_list)
	{
		ArrowFileState *af_state = lfirst(lc1);
		size_t		total_sz = af_state->stat_buf.st_size;
		size_t		read_sz = 0;
		size_t		sz;

		foreach (lc2, af_state->rb_list)
		{
			RecordBatchState *rb_state = lfirst(lc2);

			if (bms_is_member(-FirstLowInvalidHeapAttributeNumber,
							  arrow_state->referenced))
			{
				/* whole-row reference */
				read_sz += rb_state->rb_length;
			}
			else
			{
				for (k = bms_next_member(arrow_state->referenced, -1);
					 k >= 0;
					 k = bms_next_member(arrow_state->referenced, k))
				{
					j = k + FirstLowInvalidHeapAttributeNumber - 1;
					if (j < 0 || j >=  tupdesc->natts)
						continue;
					sz = __recordBatchFieldLength(&rb_state->fields[j]);
					read_sz += sz;
					chunk_sz[j] += sz;
				}
			}
		}

		/* file size and read size */
		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			resetStringInfo(&buf);
			appendStringInfo(&buf, "%s (read: %s, size: %s)",
							 af_state->filename,
							 format_bytesz(read_sz),
							 format_bytesz(total_sz));
			snprintf(label, sizeof(label), "file%d", fcount);
			ExplainPropertyText(label, buf.data, es);
		}
		else
		{
			snprintf(label, sizeof(label), "file%d", fcount);
			ExplainPropertyText(label, af_state->filename, es);

			snprintf(label, sizeof(label), "file%d-read", fcount);
			ExplainPropertyText(label, format_bytesz(read_sz), es);

			snprintf(label, sizeof(label), "file%d-size", fcount);
			ExplainPropertyText(label, format_bytesz(total_sz), es);
		}
		fcount++;
	}

	/* read-size per column (only verbose mode) */
	if (es->verbose && arrow_state->rb_nitems > 0 &&
		!bms_is_member(-FirstLowInvalidHeapAttributeNumber,
					   arrow_state->referenced))
	{
		resetStringInfo(&buf);
		for (k = bms_next_member(arrow_state->referenced, -1);
			 k >= 0;
			 k = bms_next_member(arrow_state->referenced, k))
		{
			Form_pg_attribute attr;

			j = k + FirstLowInvalidHeapAttributeNumber - 1;
			if (j < 0 || j >= tupdesc->natts)
				continue;
			attr = TupleDescAttr(tupdesc, j);
			snprintf(label, sizeof(label), "  %s", NameStr(attr->attname));
			ExplainPropertyText(label, format_bytesz(chunk_sz[j]), es);
		}
	}
	pfree(buf.data);
}

static void
ArrowExplainForeignScan(ForeignScanState *node, ExplainState *es)
{
	Relation		frel = node->ss.ss_currentRelation;
	List		   *dcontext;

	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										NULL);
	pgstromArrowFdwExplain(node->fdw_state, frel, es, dcontext);
}

/*
 * ArrowAnalyzeForeignTable
 */
static int
RecordBatchAcquireSampleRows(Relation relation,
							 RecordBatchState *rb_state,
							 HeapTuple *rows,
							 int nsamples)
{
	TupleDesc		tupdesc = RelationGetDescr(relation);
	kern_data_store *kds;
	Bitmapset	   *referenced = NULL;
	StringInfoData	buffer;
	Datum		   *values;
	bool		   *isnull;
	int				count;
	uint32_t		index;

	/* ANALYZE needs to fetch all the attributes */
	referenced = bms_make_singleton(-FirstLowInvalidHeapAttributeNumber);
	initStringInfo(&buffer);
	kds = arrowFdwFillupRecordBatch(relation,
									referenced,
									rb_state,
									&buffer);
	values = alloca(sizeof(Datum) * tupdesc->natts);
	isnull = alloca(sizeof(bool)  * tupdesc->natts);
	for (count = 0; count < nsamples; count++)
	{
		/* fetch a row randomly */
		index = (double)kds->nitems * drand48();
		Assert(index < kds->nitems);

		for (int j=0; j < kds->ncols; j++)
		{
			kern_colmeta   *cmeta = &kds->colmeta[j];

			pg_datum_arrow_ref(kds,
							   cmeta,
							   index,
							   values + j,
							   isnull + j);
		}
		rows[count] = heap_form_tuple(tupdesc, values, isnull);
	}
	pfree(buffer.data);

	return count;
}

static int
ArrowAcquireSampleRows(Relation relation,
					   int elevel,
					   HeapTuple *rows,
					   int nrooms,
					   double *p_totalrows,
					   double *p_totaldeadrows)
{
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(relation));
	List		   *filesList = arrowFdwExtractFilesList(ft->options, NULL);
	List		   *rb_state_list = NIL;
	ListCell	   *lc1, *lc2;
	int64			total_nrows = 0;
	int64			count_nrows = 0;
	int				nsamples_min = nrooms / 100;
	int				nitems = 0;

	foreach (lc1, filesList)
	{
		ArrowFileState *af_state;
		char	   *fname = strVal(lfirst(lc1));

		af_state = BuildArrowFileState(relation, fname, NULL);
		if (!af_state)
			continue;
		foreach (lc2, af_state->rb_list)
		{
			RecordBatchState *rb_state = lfirst(lc2);

			if (rb_state->rb_nitems == 0)
				continue;	/* not reasonable to sample, skipped */
			total_nrows += rb_state->rb_nitems;
			rb_state_list = lappend(rb_state_list, rb_state);
		}
	}
	nrooms = Min(nrooms, total_nrows);

	/* fetch samples for each record-batch */
	foreach (lc1, rb_state_list)
	{
		RecordBatchState *rb_state = lfirst(lc1);
		int			nsamples;

		count_nrows += rb_state->rb_nitems;
		nsamples = (double)nrooms * ((double)count_nrows /
									 (double)total_nrows) - nitems;
		if (nitems + nsamples > nrooms)
			nsamples = nrooms - nitems;
		if (nsamples > nsamples_min)
			nitems += RecordBatchAcquireSampleRows(relation,
												   rb_state,
												   rows + nitems,
												   nsamples);
	}
	*p_totalrows = total_nrows;
	*p_totaldeadrows = 0.0;

	return nitems;
}

/*
 * ArrowAnalyzeForeignTable
 */
static bool
ArrowAnalyzeForeignTable(Relation frel,
						 AcquireSampleRowsFunc *p_sample_rows_func,
						 BlockNumber *p_totalpages)
{
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(frel));
	List		   *filesList = arrowFdwExtractFilesList(ft->options, NULL);
	ListCell	   *lc;
	size_t			totalpages = 0;

	foreach (lc, filesList)
	{
		const char	   *fname = strVal(lfirst(lc));
		struct stat		stat_buf;

		if (stat(fname, &stat_buf) != 0)
		{
			elog(NOTICE, "failed on stat('%s') on behalf of '%s', skipped",
				 fname, get_rel_name(ft->relid));
			continue;
		}
		totalpages += (stat_buf.st_size + BLCKSZ - 1) / BLCKSZ;
	}
	if (totalpages > MaxBlockNumber)
		totalpages = MaxBlockNumber;

	*p_sample_rows_func = ArrowAcquireSampleRows;
	*p_totalpages = totalpages;

	return true;
}

/*
 * ArrowImportForeignSchema
 */
static List *
ArrowImportForeignSchema(ImportForeignSchemaStmt *stmt, Oid serverOid)
{
	ArrowSchema	schema;
	List	   *filesList;
	ListCell   *lc;
	int			j;
	StringInfoData	cmd;

	/* sanity checks */
	switch (stmt->list_type)
	{
		case FDW_IMPORT_SCHEMA_ALL:
			break;
		case FDW_IMPORT_SCHEMA_LIMIT_TO:
			elog(ERROR, "arrow_fdw does not support LIMIT TO clause");
			break;
		case FDW_IMPORT_SCHEMA_EXCEPT:
			elog(ERROR, "arrow_fdw does not support EXCEPT clause");
			break;
		default:
			elog(ERROR, "arrow_fdw: Bug? unknown list-type");
			break;
	}
	filesList = arrowFdwExtractFilesList(stmt->options, NULL);
	if (filesList == NIL)
		ereport(ERROR,
				(errmsg("No valid apache arrow files are specified"),
				 errhint("Use 'file' or 'dir' option to specify apache arrow files on behalf of the foreign table")));

	/* read the schema */
	memset(&schema, 0, sizeof(ArrowSchema));
	foreach (lc, filesList)
	{
		ArrowFileInfo af_info;
		const char *fname = strVal(lfirst(lc));

		readArrowFile(fname, &af_info, false);
		if (lc == list_head(filesList))
		{
			copyArrowNode(&schema.node, &af_info.footer.schema.node);
		}
		else
		{
			/* compatibility checks */
			ArrowSchema	   *stemp = &af_info.footer.schema;

			if (schema.endianness != stemp->endianness ||
				schema._num_fields != stemp->_num_fields)
				elog(ERROR, "file '%s' has incompatible schema definition", fname);
			for (j=0; j < schema._num_fields; j++)
			{
				if (arrowFieldTypeIsEqual(&schema.fields[j],
										  &stemp->fields[j]))
					elog(ERROR, "file '%s' has incompatible schema definition", fname);
			}
		}
	}

	/* makes a command to define foreign table */
	initStringInfo(&cmd);
	appendStringInfo(&cmd, "CREATE FOREIGN TABLE %s (\n",
					 quote_identifier(stmt->remote_schema));
	for (j=0; j < schema._num_fields; j++)
	{
		ArrowField *field = &schema.fields[j];
		Oid				type_oid;
		int32			type_mod;
		char		   *schema;
		HeapTuple		htup;
		Form_pg_type	__type;

		__arrowFieldTypeToPGType(field, &type_oid, &type_mod, NULL);
		if (!OidIsValid(type_oid))
			elog(ERROR, "unable to map Arrow type on any PG type");
		htup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
		if (!HeapTupleIsValid(htup))
			elog(ERROR, "cache lookup failed for type %u", type_oid);
		__type = (Form_pg_type) GETSTRUCT(htup);
		schema = get_namespace_name(__type->typnamespace);
		if (!schema)
			elog(ERROR, "cache lookup failed for schema %u", __type->typnamespace);
		if (j > 0)
			appendStringInfo(&cmd, ",\n");
		if (type_mod < 0)
		{
			appendStringInfo(&cmd, "  %s %s.%s",
							 quote_identifier(field->name),
							 quote_identifier(schema),
							 NameStr(__type->typname));
		}
		else
		{
			Assert(type_mod >= VARHDRSZ);
			appendStringInfo(&cmd, "  %s %s.%s(%d)",
							 quote_identifier(field->name),
							 quote_identifier(schema),
							 NameStr(__type->typname),
							 type_mod - VARHDRSZ);
		}
		ReleaseSysCache(htup);
	}
	appendStringInfo(&cmd,
					 "\n"
					 ") SERVER %s\n"
					 "  OPTIONS (", stmt->server_name);
	foreach (lc, stmt->options)
	{
		DefElem	   *defel = lfirst(lc);

		if (lc != list_head(stmt->options))
			appendStringInfo(&cmd, ",\n           ");
		appendStringInfo(&cmd, "%s '%s'",
						 defel->defname,
						 strVal(defel->arg));
	}
	appendStringInfo(&cmd, ")");

	return list_make1(cmd.data);
}

/*
 * pgstrom_arrow_fdw_import_file
 *
 * NOTE: Due to historical reason, PostgreSQL does not allow to define
 * columns more than MaxHeapAttributeNumber (1600) for foreign-tables also,
 * not only heap-tables. This restriction comes from NULL-bitmap length
 * in HeapTupleHeaderData and width of t_hoff.
 * However, it is not a reasonable restriction for foreign-table, because
 * it does not use heap-format internally.
 */
static void
__insertPgAttributeTuple(Relation pg_attr_rel,
						 CatalogIndexState pg_attr_index,
						 Oid ftable_oid,
						 AttrNumber attnum,
						 ArrowField *field)
{
	Oid			type_oid;
	int32		type_mod;
	int16		type_len;
	bool		type_byval;
	char		type_align;
	int32		type_ndims;
	char		type_storage;
	Datum		values[Natts_pg_attribute];
	bool		isnull[Natts_pg_attribute];
	HeapTuple	tup;
	ObjectAddress myself, referenced;

	__arrowFieldTypeToPGType(field, &type_oid, &type_mod, NULL);
	get_typlenbyvalalign(type_oid,
						 &type_len,
						 &type_byval,
						 &type_align);
	type_ndims = (type_is_array(type_oid) ? 1 : 0);
	type_storage = get_typstorage(type_oid);

	memset(values, 0, sizeof(values));
	memset(isnull, 0, sizeof(isnull));

	values[Anum_pg_attribute_attrelid - 1] = ObjectIdGetDatum(ftable_oid);
	values[Anum_pg_attribute_attname - 1] = CStringGetDatum(field->name);
	values[Anum_pg_attribute_atttypid - 1] = ObjectIdGetDatum(type_oid);
	values[Anum_pg_attribute_attstattarget - 1] = Int32GetDatum(-1);
	values[Anum_pg_attribute_attlen - 1] = Int16GetDatum(type_len);
	values[Anum_pg_attribute_attnum - 1] = Int16GetDatum(attnum);
	values[Anum_pg_attribute_attndims - 1] = Int32GetDatum(type_ndims);
	values[Anum_pg_attribute_attcacheoff - 1] = Int32GetDatum(-1);
	values[Anum_pg_attribute_atttypmod - 1] = Int32GetDatum(type_mod);
	values[Anum_pg_attribute_attbyval - 1] = BoolGetDatum(type_byval);
	values[Anum_pg_attribute_attstorage - 1] = CharGetDatum(type_storage);
	values[Anum_pg_attribute_attalign - 1] = CharGetDatum(type_align);
	values[Anum_pg_attribute_attnotnull - 1] = BoolGetDatum(!field->nullable);
	values[Anum_pg_attribute_attislocal - 1] = BoolGetDatum(true);
	isnull[Anum_pg_attribute_attacl - 1] = true;
	isnull[Anum_pg_attribute_attoptions - 1] = true;
	isnull[Anum_pg_attribute_attfdwoptions - 1] = true;
	isnull[Anum_pg_attribute_attmissingval - 1] = true;

	tup = heap_form_tuple(RelationGetDescr(pg_attr_rel), values, isnull);
	CatalogTupleInsertWithInfo(pg_attr_rel, tup, pg_attr_index);

	/* add dependency */
	myself.classId = RelationRelationId;
	myself.objectId = ftable_oid;
	myself.objectSubId = attnum;
	referenced.classId = TypeRelationId;
	referenced.objectId = type_oid;
	referenced.objectSubId = 0;
	recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);

	heap_freetuple(tup);
}

Datum
pgstrom_arrow_fdw_import_file(PG_FUNCTION_ARGS)
{
	CreateForeignTableStmt stmt;
	ArrowSchema	schema;
	List	   *tableElts = NIL;
	char	   *ftable_name;
	char	   *file_name;
	char	   *namespace_name;
	DefElem	   *defel;
	int			j, nfields;
	Oid			ftable_oid;
	ObjectAddress myself;
	ArrowFileInfo af_info;

	/* read schema of the file */
	if (PG_ARGISNULL(0))
		elog(ERROR, "foreign table name is not supplied");
	ftable_name = text_to_cstring(PG_GETARG_TEXT_PP(0));

	if (PG_ARGISNULL(1))
		elog(ERROR, "arrow filename is not supplied");
	file_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
	defel = makeDefElem("file", (Node *)makeString(file_name), -1);

	if (PG_ARGISNULL(2))
		namespace_name = NULL;
	else
		namespace_name = text_to_cstring(PG_GETARG_TEXT_PP(2));

	readArrowFile(file_name, &af_info, false);
	copyArrowNode(&schema.node, &af_info.footer.schema.node);
	if (schema._num_fields > SHRT_MAX)
		Elog("Arrow file '%s' has too much fields: %d",
			 file_name, schema._num_fields);

	/* setup CreateForeignTableStmt */
	memset(&stmt, 0, sizeof(CreateForeignTableStmt));
	NodeSetTag(&stmt, T_CreateForeignTableStmt);
	stmt.base.relation = makeRangeVar(namespace_name, ftable_name, -1);

	nfields = Min(schema._num_fields, 100);
	for (j=0; j < nfields; j++)
	{
		ColumnDef  *cdef;
		Oid			type_oid;
		int32_t		type_mod;

		__arrowFieldTypeToPGType(&schema.fields[j],
								 &type_oid,
								 &type_mod,
								 NULL);
		cdef = makeColumnDef(schema.fields[j].name,
							 type_oid,
							 type_mod,
							 InvalidOid);
		tableElts = lappend(tableElts, cdef);
	}
	stmt.base.tableElts = tableElts;
	stmt.base.oncommit = ONCOMMIT_NOOP;
	stmt.servername = "arrow_fdw";
	stmt.options = list_make1(defel);

	myself = DefineRelation(&stmt.base,
							RELKIND_FOREIGN_TABLE,
							InvalidOid,
							NULL,
							__FUNCTION__);
	ftable_oid = myself.objectId;
	CreateForeignTable(&stmt, ftable_oid);

	if (nfields < schema._num_fields)
	{
		Relation	c_rel = table_open(RelationRelationId, RowExclusiveLock);
		Relation	a_rel = table_open(AttributeRelationId, RowExclusiveLock);
		CatalogIndexState c_index = CatalogOpenIndexes(c_rel);
		CatalogIndexState a_index = CatalogOpenIndexes(a_rel);
		HeapTuple	tup;

		tup = SearchSysCacheCopy1(RELOID, ObjectIdGetDatum(ftable_oid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "cache lookup failed for relation %u", ftable_oid);

		for (j=nfields; j < schema._num_fields; j++)
		{
			__insertPgAttributeTuple(a_rel,
									 a_index,
									 ftable_oid,
									 j+1,
                                     &schema.fields[j]);
		}
		/* update relnatts also */
		((Form_pg_class) GETSTRUCT(tup))->relnatts = schema._num_fields;
		CatalogTupleUpdate(c_rel, &tup->t_self, tup);

		CatalogCloseIndexes(a_index);
		CatalogCloseIndexes(c_index);
		table_close(a_rel, RowExclusiveLock);
		table_close(c_rel, RowExclusiveLock);

		CommandCounterIncrement();
	}
	PG_RETURN_VOID();
}

/*
 * handler of Arrow_Fdw
 */
Datum
pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&pgstrom_arrow_fdw_routine);
}

/*
 * validator of Arrow_Fdw
 */
Datum
pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS)
{
	List   *options = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid		catalog = PG_GETARG_OID(1);

	if (catalog == ForeignTableRelationId)
	{
		List	   *filesList = arrowFdwExtractFilesList(options, NULL);
		ListCell   *lc;

		foreach (lc, filesList)
		{
			const char *fname = strVal(lfirst(lc));
			ArrowFileInfo af_info;

			readArrowFile(fname, &af_info, true);
		}
	}
	else if (options != NIL)
	{
		const char *label;

		switch (catalog)
		{
			case ForeignDataWrapperRelationId:
				label = "FOREIGN DATA WRAPPER";
				break;
			case ForeignServerRelationId:
				label = "SERVER";
				break;
			case UserMappingRelationId:
				label = "USER MAPPING";
				break;
			case AttributeRelationId:
				label = "attribute of FOREIGN TABLE";
				break;
			default:
				label = "????";
				break;
		}
		elog(ERROR, "Arrow_Fdw does not support any options for %s", label);
	}
	PG_RETURN_VOID();
}

/*
 * pgstrom_arrow_fdw_precheck_schema
 */
Datum
pgstrom_arrow_fdw_precheck_schema(PG_FUNCTION_ARGS)
{
	EventTriggerData *trigdata;
	Relation	frel = NULL;
	ListCell   *lc;
	bool		check_schema_compatibility = false;

	if (!CALLED_AS_EVENT_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as EventTrigger", __FUNCTION__);
	trigdata = (EventTriggerData *) fcinfo->context;
	if (strcmp(trigdata->event, "ddl_command_end") != 0)
		elog(ERROR, "%s: must be called on ddl_command_end event", __FUNCTION__);

	if (strcmp(GetCommandTagName(trigdata->tag),
			   "CREATE FOREIGN TABLE") == 0)
	{
		CreateStmt *stmt = (CreateStmt *)trigdata->parsetree;

		frel = relation_openrv_extended(stmt->relation, NoLock, true);
		if (frel && RelationIsArrowFdw(frel))
			check_schema_compatibility = true;
	}
	else if (strcmp(GetCommandTagName(trigdata->tag),
					"ALTER FOREIGN TABLE") == 0 &&
			 IsA(trigdata->parsetree, AlterTableStmt))
	{
		AlterTableStmt *stmt = (AlterTableStmt *)trigdata->parsetree;

		frel = relation_openrv_extended(stmt->relation, NoLock, true);
		if (frel && RelationIsArrowFdw(frel))
		{
			foreach (lc, stmt->cmds)
			{
				AlterTableCmd  *cmd = lfirst(lc);

				if (cmd->subtype == AT_AddColumn ||
					cmd->subtype == AT_DropColumn ||
					cmd->subtype == AT_AlterColumnType)
				{
					check_schema_compatibility = true;
					break;
				}
			}
		}
	}

	if (check_schema_compatibility)
	{
		ForeignTable *ft = GetForeignTable(RelationGetRelid(frel));
		List	   *filesList = arrowFdwExtractFilesList(ft->options, NULL);

		foreach (lc, filesList)
		{
			const char *fname = strVal(lfirst(lc));

			(void)BuildArrowFileState(frel, fname, NULL);
		}
	}
	if (frel)
		relation_close(frel, NoLock);
	PG_RETURN_NULL();
}

/*
 * pgstrom_request_arrow_fdw
 */
static void
pgstrom_request_arrow_fdw(void)
{
	size_t	sz;

	if (shmem_request_next)
		shmem_request_next();
	sz = TYPEALIGN(ARROW_METADATA_BLOCKSZ,
				   (size_t)arrow_metadata_cache_size_kb << 10);
	RequestAddinShmemSpace(MAXALIGN(sizeof(arrowMetadataCacheHead)) + sz);
}

/*
 * pgstrom_startup_arrow_fdw
 */
static void
pgstrom_startup_arrow_fdw(void)
{
	bool	found;
	size_t	sz;
	char   *buffer;
	int		i, n;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	arrow_metadata_cache = ShmemInitStruct("arrowMetadataCache(head)",
										   MAXALIGN(sizeof(arrowMetadataCacheHead)),
										   &found);
	Assert(!found);
	
	LWLockInitialize(&arrow_metadata_cache->mutex, LWLockNewTrancheId());
	SpinLockInit(&arrow_metadata_cache->lru_lock);
	dlist_init(&arrow_metadata_cache->lru_list);
	dlist_init(&arrow_metadata_cache->free_blocks);
	dlist_init(&arrow_metadata_cache->free_mcaches);
	dlist_init(&arrow_metadata_cache->free_fcaches);
	for (i=0; i < ARROW_METADATA_HASH_NSLOTS; i++)
		dlist_init(&arrow_metadata_cache->hash_slots[i]);

	/* slab allocator */
	sz = TYPEALIGN(ARROW_METADATA_BLOCKSZ,
				   (size_t)arrow_metadata_cache_size_kb << 10);
	n = sz / ARROW_METADATA_BLOCKSZ;
	buffer = ShmemInitStruct("arrowMetadataCache(body)", sz, &found);
	Assert(!found);
	for (i=0; i < n; i++)
	{
		arrowMetadataCacheBlock *mc_block = (arrowMetadataCacheBlock *)buffer;

		memset(mc_block, 0, offsetof(arrowMetadataCacheBlock, data));
		dlist_push_tail(&arrow_metadata_cache->free_blocks, &mc_block->chain);

		buffer += ARROW_METADATA_BLOCKSZ;
	}
}

/*
 * pgstrom_init_arrow_fdw
 */
void
pgstrom_init_arrow_fdw(void)
{
	FdwRoutine *r = &pgstrom_arrow_fdw_routine;

	memset(r, 0, sizeof(FdwRoutine));
	NodeSetTag(r, T_FdwRoutine);
	/* SCAN support */
	r->GetForeignRelSize			= ArrowGetForeignRelSize;
	r->GetForeignPaths				= ArrowGetForeignPaths;
	r->GetForeignPlan				= ArrowGetForeignPlan;
	r->BeginForeignScan				= ArrowBeginForeignScan;
	r->IterateForeignScan			= ArrowIterateForeignScan;
	r->ReScanForeignScan			= ArrowReScanForeignScan;
	r->EndForeignScan				= ArrowEndForeignScan;
	/* EXPLAIN support */
	r->ExplainForeignScan			= ArrowExplainForeignScan;
	/* ANALYZE support */
	r->AnalyzeForeignTable			= ArrowAnalyzeForeignTable;
	/* CPU Parallel support */
	r->IsForeignScanParallelSafe	= ArrowIsForeignScanParallelSafe;
	r->EstimateDSMForeignScan		= ArrowEstimateDSMForeignScan;
	r->InitializeDSMForeignScan		= ArrowInitializeDSMForeignScan;
	//r->ReInitializeDSMForeignScan	= ArrowReInitializeDSMForeignScan;
	r->InitializeWorkerForeignScan	= ArrowInitializeWorkerForeignScan;
	r->ShutdownForeignScan			= ArrowShutdownForeignScan;
	/* IMPORT FOREIGN SCHEMA support */
	r->ImportForeignSchema			= ArrowImportForeignSchema;

	/*
	 * Turn on/off arrow_fdw
	 */
	DefineCustomBoolVariable("arrow_fdw.enabled",
							 "Enables the planner's use of Arrow_Fdw",
							 NULL,
							 &arrow_fdw_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/*
	 * Turn on/off min/max statistics hint
	 */
	DefineCustomBoolVariable("arrow_fdw.stats_hint_enabled",
							 "Enables min/max statistics hint, if any",
							 NULL,
							 &arrow_fdw_stats_hint_enabled,
							 true,
							 PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);
	/*
	 * Configurations for arrow_fdw metadata cache
	 */
	DefineCustomIntVariable("arrow_fdw.metadata_cache_size",
							"size of shared metadata cache for arrow files",
							NULL,
							&arrow_metadata_cache_size_kb,
							512 * 1024,		/* 512MB */
							32 * 1024,		/* 32MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	/* shared memory size */
	shmem_request_next = shmem_request_hook;
	shmem_request_hook = pgstrom_request_arrow_fdw;
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_arrow_fdw;
}








