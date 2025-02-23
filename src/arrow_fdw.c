/*
 * arrow_fdw.c
 *
 * Routines to map Apache Arrow files as PG's Foreign-Table.
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
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
	char		attname[NAMEDATALEN];
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
	/* virtual column per record-batch */
	Datum		virtual_datum;
	bool		virtual_isnull;
	/* per column information */
	int			nfields;
	RecordBatchFieldState fields[FLEXIBLE_ARRAY_MEMBER];
} RecordBatchState;

typedef struct virtualColumnDef
{
	char		kind;
	char	   *key;
	char	   *value;
	char		buf[FLEXIBLE_ARRAY_MEMBER];
} virtualColumnDef;

#define __FIELD_INDEX_SPECIAL__VIRTUAL_PER_FILE				(-1)
#define __FIELD_INDEX_SPECIAL__VIRTUAL_PER_RECORD_BATCH		(-2)
typedef struct ArrowFileState
{
	const char *filename;
	const char *dpu_path;	/* relative pathname, if DPU */
	struct stat	stat_buf;
	List	   *rb_list;	/* list of RecordBatchState */
	uint32_t	ncols;
	struct {
		int		field_index;		/* PG-column <-> Arrow-field mapping */
		bool	virtual_isnull;		/* virtual isnull, if any */
		Datum	virtual_datum;		/* virtual datum, if any */
	} attrs[FLEXIBLE_ARRAY_MEMBER];
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
typedef struct arrowMetadataCacheBlock	arrowMetadataCacheBlock;
typedef struct arrowMetadataCache		arrowMetadataCache;
typedef struct arrowMetadataFieldCache	arrowMetadataFieldCache;
typedef struct arrowMetadataKeyValueCache arrowMetadataKeyValueCache;

/*
 * arrowMetadataKeyValueCache
 */
struct arrowMetadataKeyValueCache
{
	arrowMetadataKeyValueCache *next;
	const char *key;
	const char *value;
	int			_key_len;
	int			_value_len;
	char		data[FLEXIBLE_ARRAY_MEMBER];
};

/*
 * arrowMetadataFieldCache - metadata for a particular field in record-batch
 */
struct arrowMetadataFieldCache
{
	dlist_node	chain;				/* link to free/fields[children] list */
	/* common fields with cache */
	char		attname[NAMEDATALEN];
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
	arrowMetadataKeyValueCache *custom_metadata;	/* valid only in RecordBatch-0 */
	/* sub-fields if any */
	int			num_children;
	dlist_head	children;
};

/*
 * arrowMetadataCache - metadata for a particular record-batch
 */
struct arrowMetadataCache
{
	arrowMetadataCache *next; /* next record-batch if any */
	int			rb_index;	/* index number in a file */
	off_t		rb_offset;	/* offset from the head */
	size_t		rb_length;	/* length of the entire RecordBatch */
	int64		rb_nitems;	/* number of items */
	/* per column information */
	int			nfields;
	dlist_head	fields;		/* list of arrowMetadataFieldCache */
};

/*
 * arrowMetadataCacheBlock - metadata for a particular arrow file, and
 *                           allocation unit
 */
#define ARROW_METADATA_BLOCKSZ		(128 * 1024)	/* 128kB */
typedef struct arrowMetadataCacheBlock
{
	arrowMetadataCacheBlock *next;
	uint32_t	usage;
	/* <---- the fields below are valid only the first block ----> */
	dlist_node	chain;		/* link to free/hash list */
	dlist_node	lru_chain;	/* link to lru_list */
	struct timeval lru_tv;	/* last access time */
	struct stat	stat_buf;	/* result of stat(2) */
	arrowMetadataKeyValueCache *custom_metadata;
	arrowMetadataCache mcache_head;	/* the first arrowMetadataCache (record batch)
									 * in this file */
} arrowMetadataCacheBlock;

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
		Oid			extension_oid = InvalidOid;
		Oid			namespace_oid = InvalidOid;
		Oid			hint_oid;
		char	   *namebuf, *pos;

		/* pg_type = NAMESPACE.TYPENAME@EXTENSION */
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
		pos = strchr(namebuf, '@');
		if (pos)
		{
			*pos++ = '\0';
			extension_oid = get_extension_oid(pos, true);
			if (!OidIsValid(extension_oid))
				continue;
		}

		if (OidIsValid(namespace_oid))
		{
			hint_oid = GetSysCacheOid2(TYPENAMENSP,
									   Anum_pg_type_oid,
									   CStringGetDatum(namebuf),
									   ObjectIdGetDatum(namespace_oid));
			if (OidIsValid(hint_oid))
			{
				if (!OidIsValid(extension_oid) ||
					getExtensionOfObject(TypeRelationId,
										 hint_oid) == extension_oid)
					return hint_oid;
			}
		}
		else
		{
			CatCList   *typelist;
			HeapTuple	htup;

			/* 1st try: 'pg_catalog' + typname */
			hint_oid = GetSysCacheOid2(TYPENAMENSP,
									   Anum_pg_type_oid,
									   CStringGetDatum(namebuf),
									   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
			if (OidIsValid(hint_oid))
			{
				if (!OidIsValid(extension_oid) ||
					getExtensionOfObject(TypeRelationId,
										 hint_oid) == extension_oid)
					return hint_oid;
			}
			/* 2nd try: any other namespaces */
			typelist = SearchSysCacheList1(TYPENAMENSP,
										   CStringGetDatum(namebuf));
			for (int k=0; k < typelist->n_members; k++)
			{
				htup = &typelist->members[k]->tuple;
				hint_oid = ((Form_pg_type) GETSTRUCT(htup))->oid;

				if (!OidIsValid(extension_oid) ||
					getExtensionOfObject(TypeRelationId,
										 hint_oid) == extension_oid)
				{
					ReleaseCatCacheList(typelist);
					return hint_oid;
				}
			}
			ReleaseCatCacheList(typelist);
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
__releaseMetadataCacheBlock(arrowMetadataCacheBlock *mc_block_curr)
{
	while (mc_block_curr)
	{
		arrowMetadataCacheBlock *mc_block_next = mc_block_curr->next;

		/* must be already detached */
		Assert(!mc_block_curr->chain.prev &&
			   !mc_block_curr->chain.prev &&
			   !mc_block_curr->lru_chain.prev  &&
			   !mc_block_curr->lru_chain.next);
		dlist_push_head(&arrow_metadata_cache->free_blocks,
						&mc_block_curr->chain);
		mc_block_curr = mc_block_next;
	}
}

static bool
__reclaimMetadataCacheBlock(void)
{
	SpinLockAcquire(&arrow_metadata_cache->lru_lock);
	if (!dlist_is_empty(&arrow_metadata_cache->lru_list))
	{
		arrowMetadataCacheBlock *mc_block;
		dlist_node	   *dnode;
		struct timeval	curr_tv;
		int64_t			elapsed;

		gettimeofday(&curr_tv, NULL);
		dnode = dlist_tail_node(&arrow_metadata_cache->lru_list);
		mc_block = dlist_container(arrowMetadataCacheBlock, lru_chain, dnode);
		elapsed = ((curr_tv.tv_sec - mc_block->lru_tv.tv_sec) * 1000000 +
				   (curr_tv.tv_usec - mc_block->lru_tv.tv_usec));
		if (elapsed > 30000000UL)	/* > 30s */
		{
			dlist_delete(&mc_block->lru_chain);
			memset(&mc_block->lru_chain, 0, sizeof(dlist_node));
			SpinLockRelease(&arrow_metadata_cache->lru_lock);
			dlist_delete(&mc_block->chain);
			memset(&mc_block->chain, 0, sizeof(dlist_node));

			__releaseMetadataCacheBlock(mc_block);
			return true;
		}
	}
	SpinLockRelease(&arrow_metadata_cache->lru_lock);
	return false;
}

static arrowMetadataCacheBlock *
__allocMetadataCacheBlock(void)
{
	arrowMetadataCacheBlock *mc_block;
	dlist_node	   *dnode;

	if (dlist_is_empty(&arrow_metadata_cache->free_blocks))
	{
		if (!__reclaimMetadataCacheBlock())
			return NULL;
		Assert(!dlist_is_empty(&arrow_metadata_cache->free_blocks));
	}
	dnode = dlist_pop_head_node(&arrow_metadata_cache->free_blocks);
	mc_block = dlist_container(arrowMetadataCacheBlock, chain, dnode);
	memset(mc_block, 0, offsetof(arrowMetadataCacheBlock,
								 usage) + sizeof(uint32_t));
	mc_block->usage = MAXALIGN(offsetof(arrowMetadataCacheBlock,
										usage) + sizeof(uint32_t));
	return mc_block;
}

static void *
__allocMetadataCacheCommon(arrowMetadataCacheBlock **p_mc_block, size_t sz)
{
	arrowMetadataCacheBlock *mc_block = *p_mc_block;
	char	   *pos;

	if (mc_block->usage + MAXALIGN(sz) > ARROW_METADATA_BLOCKSZ)
	{
		if (offsetof(arrowMetadataCacheBlock,
					 chain) + MAXALIGN(sz) > ARROW_METADATA_BLOCKSZ)
			return NULL;	/* too large */
		mc_block = __allocMetadataCacheBlock();
		if (!mc_block)
			return NULL;
		mc_block->next = NULL;
		mc_block->usage = offsetof(arrowMetadataCacheBlock, chain);
		Assert(!(*p_mc_block)->next);
		(*p_mc_block)->next = mc_block;
		(*p_mc_block) = mc_block;
	}
	pos = ((char *)mc_block + mc_block->usage);
	mc_block->usage += MAXALIGN(sz);

	return pos;
}

static arrowMetadataKeyValueCache *
__allocMetadataKeyValueCache(arrowMetadataCacheBlock **p_mc_block,
							 ArrowKeyValue *kv)
{
	arrowMetadataKeyValueCache *mc_kv;
	size_t		sz = (offsetof(arrowMetadataKeyValueCache,data)
					  + kv->_key_len + 1
					  + kv->_value_len + 1);
	mc_kv = __allocMetadataCacheCommon(p_mc_block, sz);
	if (mc_kv)
	{
		char   *pos = mc_kv->data;

		mc_kv->next = NULL;
		mc_kv->key = pos;
		mc_kv->_key_len = kv->_key_len;
		strncpy(pos, kv->key, kv->_key_len);
		pos[kv->_key_len] = '\0';
		pos += kv->_key_len + 1;

		mc_kv->value = pos;
		mc_kv->_value_len = kv->_value_len;
		strncpy(pos, kv->value, kv->_value_len);
		pos[kv->_value_len] = '\0';
   }
   return mc_kv;
}

static arrowMetadataFieldCache *
__allocMetadataFieldCache(arrowMetadataCacheBlock **p_mc_block)
{
	return (arrowMetadataFieldCache *)
		__allocMetadataCacheCommon(p_mc_block, sizeof(arrowMetadataFieldCache));
}

static arrowMetadataCache *
__allocMetadataCache(arrowMetadataCacheBlock **p_mc_block)
{
	return (arrowMetadataCache *)
		__allocMetadataCacheCommon(p_mc_block, sizeof(arrowMetadataCache));
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

static arrowMetadataCacheBlock *
lookupArrowMetadataCache(struct stat *stat_buf, bool has_exclusive)
{
	uint32_t	hindex = arrowMetadataHashIndex(stat_buf);
	dlist_mutable_iter iter;

	dlist_foreach_modify(iter, &arrow_metadata_cache->hash_slots[hindex])
	{
		arrowMetadataCacheBlock *mc_block
			= dlist_container(arrowMetadataCacheBlock, chain, iter.cur);

		if (stat_buf->st_dev == mc_block->stat_buf.st_dev &&
			stat_buf->st_ino == mc_block->stat_buf.st_ino)
		{
			/*
			 * Is the metadata cache still valid?
			 */
			if (stat_buf->st_mtim.tv_sec < mc_block->stat_buf.st_mtim.tv_sec ||
				(stat_buf->st_mtim.tv_sec == mc_block->stat_buf.st_mtim.tv_sec &&
				 stat_buf->st_mtim.tv_nsec <= mc_block->stat_buf.st_mtim.tv_nsec))
			{
				/* ok, found */
				SpinLockAcquire(&arrow_metadata_cache->lru_lock);
				gettimeofday(&mc_block->lru_tv, NULL);
				dlist_move_head(&arrow_metadata_cache->lru_list,
								&mc_block->lru_chain);
				SpinLockRelease(&arrow_metadata_cache->lru_lock);
				return mc_block;
			}
			else if (has_exclusive)
			{
				/*
				 * Unfortunatelly, metadata cache is already invalid.
				 * If caller has exclusive lock, we release it.
				 */
				SpinLockAcquire(&arrow_metadata_cache->lru_lock);
				dlist_delete(&mc_block->lru_chain);
				memset(&mc_block->lru_chain, 0, sizeof(dlist_node));
				SpinLockRelease(&arrow_metadata_cache->lru_lock);
				dlist_delete(&mc_block->chain);
				memset(&mc_block->chain, 0, sizeof(dlist_node));

				__releaseMetadataCacheBlock(mc_block);
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
		int64_t		__drift;

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
				stat_values[index].max.datum = (Datum)__max;
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
				__drift = (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * USECS_PER_DAY;
				switch (field->type.Timestamp.unit)
				{
					case ArrowTimeUnit__Second:
						stat_values[index].min.datum = __min * 1000000L - __drift;
						stat_values[index].max.datum = __max * 1000000L - __drift;
						break;
					case ArrowTimeUnit__MilliSecond:
						stat_values[index].min.datum = __min * 1000L - __drift;
						stat_values[index].max.datum = __max * 1000L - __drift;
						break;
					case ArrowTimeUnit__MicroSecond:
						stat_values[index].min.datum = __min - __drift;
						stat_values[index].max.datum = __max - __drift;
						break;
					case ArrowTimeUnit__NanoSecond:
						stat_values[index].min.datum = __min / 1000 - __drift;
						stat_values[index].max.datum = __max / 1000 - __drift;
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
				*p_stat_attrs = bms_add_member(*p_stat_attrs, j);
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
	Scan	   *scan = (Scan *)ss->ps.plan;
	Oid			opcode;
	Var		   *var;
	Node	   *arg;
	Expr	   *expr;
	Oid			opfamily = InvalidOid;
	StrategyNumber strategy = InvalidStrategy;
	CatCList   *catlist;
	int			i, anum;

	/* quick bailout if not binary operator */
	if (list_length(op->args) != 2)
		return false;

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
	/*
	 * Is it VAR <OPER> ARG form?
	 *
	 * MEMO: expression nodes (like Var) might be rewritten to INDEX_VAR +
	 * resno on the custom_scan_tlist by setrefs.c, so we should reference
	 * Var::varnosyn and ::varattnosyn, instead of ::varno and ::varattno.
	 */
	if (!IsA(var, Var) || !OidIsValid(opcode))
		return false;
	if (var->varnosyn != scan->scanrelid)
		return false;
	anum = var->varattnosyn - FirstLowInvalidHeapAttributeNumber;
	if (!bms_is_member(anum, as_hint->stat_attrs))
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
									 BTLessStrategyNumber);
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

static bool
__buildArrowStatsBoolOp(arrowStatsHint *as_hint,
						ScanState *ss, Expr *expr)
{
	Scan   *scan = (Scan *)ss->ps.plan;

	if (IsA(expr, BoolExpr))
	{
		BoolExpr   *b = (BoolExpr *)expr;

		if (b->boolop == NOT_EXPR && list_length(b->args) == 1)
		{
			Var	   *var = (Var *)linitial(b->args);

			if (IsA(var, Var) &&
				var->vartype == BOOLOID &&
				var->varnosyn == scan->scanrelid)
			{
				/*
				 * WHERE NOT <BOOL_VAL>
				 *  --> If MIN_VALUE == true, no chance to match
				 */
				int		anum = var->varattnosyn - FirstLowInvalidHeapAttributeNumber;

				if (!bms_is_member(anum, as_hint->stat_attrs))
					return false;
				expr = (Expr *)makeVar(INNER_VAR,
									   var->varattno,
									   var->vartype,
									   var->vartypmod,
									   var->varcollid, 0);
				as_hint->eval_quals = lappend(as_hint->eval_quals, expr);
				as_hint->load_attrs = bms_add_member(as_hint->load_attrs, var->varattno);
				return true;
			}
		}
	}
	else if (IsA(expr, Var))
	{
		Var	   *var = (Var *)expr;

		if (IsA(var, Var) &&
			var->vartype == BOOLOID &&
			var->varnosyn == scan->scanrelid)
		{
			/*
			 * WHERE <BOOL_VAL>
			 *  --> If MAX_VALUE == false, no change to match
			 */
			int		anum = var->varattnosyn - FirstLowInvalidHeapAttributeNumber;

			if (!bms_is_member(anum, as_hint->stat_attrs))
				return false;
			expr = make_notclause((Expr *)makeVar(OUTER_VAR,
												  var->varattno,
												  var->vartype,
												  var->vartypmod,
												  var->varcollid, 0));
			as_hint->eval_quals = lappend(as_hint->eval_quals, expr);
			as_hint->load_attrs = bms_add_member(as_hint->load_attrs, var->varattno);
			return true;
		}
	}
	return false;
}

static bool
__buildArrowStatsScalarArrayOp(arrowStatsHint *as_hint,
							   ScanState *ss,
							   ScalarArrayOpExpr *sa_op)
{
	Oid			opcode = sa_op->opno;
	Var		   *var;
	Const	   *con;
	Expr	   *expr;
	Oid			elem_oid;
	int16		elem_typlen;
	bool		elem_typbyval;
	Datum		elem_datum;
	bool		elem_isnull;
	Oid			opfamily = InvalidOid;
	StrategyNumber strategy = InvalidStrategy;
	ArrayType  *arr;
	ArrayIterator iter;
	CatCList   *catlist;
	List	   *result_args = NIL;
	bool		retval = false;
	int			anum;

	if (list_length(sa_op->args) != 2)
		return false;
	var = linitial(sa_op->args);
	con = lsecond(sa_op->args);

	/*
	 * Is it VAR <OPER> ARRAY form?
	 *
	 * MEMO: expression nodes (like Var) might be rewritten to INDEX_VAR +
	 * resno on the custom_scan_tlist by setrefs.c, so we should reference
	 * Var::varnosyn and ::varattnosyn, instead of ::varno and ::varattno.
	 */
	if (!OidIsValid(opcode) ||
		!IsA(var, Var) ||
		!IsA(con, Const) ||
		con->constisnull)
		return false;
	elem_oid = get_element_type(con->consttype);
	if (!OidIsValid(elem_oid))
		return false;
	get_typlenbyval(elem_oid, &elem_typlen, &elem_typbyval);

	anum = var->varattnosyn - FirstLowInvalidHeapAttributeNumber;
	if (!bms_is_member(anum, as_hint->stat_attrs))
		return false;

	/*
	 * Identify the comparison strategy
	 */
	catlist = SearchSysCacheList1(AMOPOPID, ObjectIdGetDatum(sa_op->opno));
	for (int i=0; i < catlist->n_members; i++)
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

	/*
	 * Iterate for each array element
	 */
	arr = DatumGetArrayTypeP(con->constvalue);
	iter = array_create_iterator(arr, 0, NULL);
	while (array_iterate(iter, &elem_datum, &elem_isnull))
	{
		if (strategy == BTLessStrategyNumber ||
			strategy == BTLessEqualStrategyNumber)
		{
			/*
			 * if (VAR < ARG) --> (Min >= ARG), can be skipped
			 * if (VAR <= ARG) --> (Min > ARG), can be skipped
			 */
			Oid		negator = get_negator(opcode);

			if (!OidIsValid(negator))
				goto bailout;
			expr = make_opclause(negator,
								 get_op_rettype(negator),
								 false,
								 (Expr *)makeVar(INNER_VAR,
												 var->varattno,
												 var->vartype,
												 var->vartypmod,
												 var->varcollid,
												 0),
								 (Expr *)makeConst(elem_oid,
												   con->consttypmod,
												   con->constcollid,
												   elem_typlen,
												   elem_datum,
												   elem_isnull,
												   elem_typbyval),
								 InvalidOid,
								 sa_op->inputcollid);
			set_opfuncid((OpExpr *)expr);
			result_args = lappend(result_args, expr);
		}
		else if (strategy == BTGreaterEqualStrategyNumber ||
				 strategy == BTGreaterStrategyNumber)
		{
			/* if (VAR > ARG) --> (Max <= ARG), can be skipped */
			/* if (VAR >= ARG) --> (Max < ARG), can be skipped */
			Oid		negator = get_negator(opcode);

			if (!OidIsValid(negator))
				goto bailout;
			expr = make_opclause(negator,
								 get_op_rettype(negator),
								 false,
								 (Expr *)makeVar(OUTER_VAR,
												 var->varattno,
												 var->vartype,
												 var->vartypmod,
												 var->varcollid,
												 0),
								 (Expr *)makeConst(elem_oid,
												   con->consttypmod,
												   con->constcollid,
												   elem_typlen,
												   elem_datum,
												   elem_isnull,
												   elem_typbyval),
								 InvalidOid,
								 sa_op->inputcollid);
			set_opfuncid((OpExpr *)expr);
			result_args = lappend(result_args, expr);
		}
		else if (strategy == BTEqualStrategyNumber)
		{
			/* (VAR = ARG) --> (Min > ARG) || (Max < ARG), can be skipped */
			Oid		gt_opcode;
			Oid		lt_opcode;
			Expr   *gt_expr;
			Expr   *lt_expr;

			gt_opcode = get_opfamily_member(opfamily, var->vartype,
											elem_oid,
											BTGreaterStrategyNumber);
			gt_expr = make_opclause(gt_opcode,
									get_op_rettype(gt_opcode),
									false,
									(Expr *)makeVar(INNER_VAR,
													var->varattno,
													var->vartype,
													var->vartypmod,
													var->varcollid,
													0),
									(Expr *)makeConst(elem_oid,
													  con->consttypmod,
													  con->constcollid,
													  elem_typlen,
													  elem_datum,
													  elem_isnull,
													  elem_typbyval),
									InvalidOid,
									sa_op->inputcollid);
			set_opfuncid((OpExpr *)gt_expr);

			lt_opcode = get_opfamily_member(opfamily, var->vartype,
											elem_oid,
											BTLessStrategyNumber);
			lt_expr = make_opclause(lt_opcode,
									get_op_rettype(lt_opcode),
									false,
									(Expr *)makeVar(OUTER_VAR,
													var->varattno,
													var->vartype,
													var->vartypmod,
													var->varcollid,
													0),
									(Expr *)makeConst(elem_oid,
													  con->consttypmod,
													  con->constcollid,
													  elem_typlen,
													  elem_datum,
													  elem_isnull,
													  elem_typbyval),
									InvalidOid,
									sa_op->inputcollid);
			set_opfuncid((OpExpr *)lt_expr);

			expr = makeBoolExpr(OR_EXPR,
								list_make2(gt_expr, lt_expr),
								-1);
			result_args = lappend(result_args, expr);
		}
		else
		{
			goto bailout;
		}
	}
	as_hint->eval_quals = lappend(as_hint->eval_quals,
								  makeBoolExpr(sa_op->useOr ? AND_EXPR : OR_EXPR,
											   result_args,
											   -1));
	as_hint->load_attrs = bms_add_member(as_hint->load_attrs, var->varattno);
	retval = true;
bailout:
	array_free_iterator(iter);

	return retval;
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

	outer_quals = fixup_scanstate_quals(ss, outer_quals);
	as_hint = palloc0(sizeof(arrowStatsHint));
	as_hint->stat_attrs = stat_attrs;
	foreach (lc, outer_quals)
	{
		Expr   *expr = lfirst(lc);

		if (IsA(expr, OpExpr) &&
			(__buildArrowStatsOper(as_hint, ss, (OpExpr *)expr, false) ||
			 __buildArrowStatsOper(as_hint, ss, (OpExpr *)expr, true)))
		{
			as_hint->orig_quals = lappend(as_hint->orig_quals, expr);
		}
		else if (IsA(expr, ScalarArrayOpExpr) &&
				 __buildArrowStatsScalarArrayOp(as_hint, ss,
												(ScalarArrayOpExpr *)expr))
		{
			as_hint->orig_quals = lappend(as_hint->orig_quals, expr);
		}
		else if (__buildArrowStatsBoolOp(as_hint, ss, expr))
		{
			as_hint->orig_quals = lappend(as_hint->orig_quals, expr);
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
	ArrowFileState *af_state = rb_state->af_state;
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
		int		field_index;

		Assert(anum > 0 && anum <= af_state->ncols);
		field_index = af_state->attrs[anum-1].field_index;
		if (field_index >= 0)
		{
			RecordBatchFieldState *rb_field = &rb_state->fields[field_index];

			Assert(field_index < rb_state->nfields);
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
		else if (field_index == __FIELD_INDEX_SPECIAL__VIRTUAL_PER_FILE)
		{
			bool	virtual_isnull = af_state->attrs[anum-1].virtual_isnull;
			Datum	virtual_datum  = af_state->attrs[anum-1].virtual_datum;

			min_values->tts_isnull[anum-1] = virtual_isnull;
			max_values->tts_isnull[anum-1] = virtual_isnull;
			if (!virtual_isnull)
			{
				min_values->tts_values[anum-1] = virtual_datum;
				max_values->tts_values[anum-1] = virtual_datum;
			}
		}
		else if (field_index == __FIELD_INDEX_SPECIAL__VIRTUAL_PER_RECORD_BATCH)
		{
			bool	virtual_isnull = rb_state->virtual_isnull;
			Datum	virtual_datum  = rb_state->virtual_datum;

			min_values->tts_isnull[anum-1] = virtual_isnull;
			max_values->tts_isnull[anum-1] = virtual_isnull;
			if (!virtual_isnull)
			{
				min_values->tts_values[anum-1] = virtual_datum;
				max_values->tts_values[anum-1] = virtual_datum;
			}
		}
		else
		{
			elog(ERROR, "Bug? unexpected field-index (%d)", field_index);
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
	strcpy(rb_field->attname, fcache->attname);
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

static bool
__setupArrowFileStateByCache(ArrowFileState *af_state,
							 const char *filename,
							 arrowMetadataCacheBlock *mc_block,
							 Bitmapset **p_stat_attrs)
{
	arrowMetadataCache *mcache = &mc_block->mcache_head;

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
			if (p_stat_attrs && !fcache->stat_datum.isnull)
				*p_stat_attrs = bms_add_member(*p_stat_attrs, j);
			__buildRecordBatchFieldStateByCache(&rb_state->fields[j++], fcache);
		}
		Assert(j == rb_state->nfields);
		af_state->rb_list = lappend(af_state->rb_list, rb_state);

		mcache = mcache->next;
	}
	return true;
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
	attopts.align = ALIGNOF_LONG;	/* some data types expand the alignment */
	switch (t->node.tag)
	{
		case ArrowNodeTag__Int:
			attopts.tag = ArrowType__Int;
			switch (t->Int.bitWidth)
			{
				case 8:
					attopts.unitsz = sizeof(int8_t);
					type_oid = get_int1_type_oid(false);
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
					type_oid = get_float2_type_oid(false);
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
			attopts.align             = sizeof(int128_t);
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
			if (OidIsValid(hint_oid) &&
				hint_oid == get_cube_type_oid(true))
				type_oid = hint_oid;
			else
				type_oid = BYTEAOID;
			break;

		case ArrowNodeTag__LargeBinary:
			attopts.tag = ArrowType__LargeBinary;
			attopts.unitsz = sizeof(uint64_t);
			if (OidIsValid(hint_oid) &&
				hint_oid == get_cube_type_oid(true))
				type_oid = hint_oid;
			else
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
	strncpy(rb_field->attname, field->name, NAMEDATALEN);
	rb_field->attname[NAMEDATALEN-1] = '\0';
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

static bool
__setupArrowFileStateByFile(ArrowFileState *af_state,
							const char *filename,
							ArrowFileInfo *af_info,
							Bitmapset **p_stat_attrs)
{
	arrowStatsBinary *arrow_bstats;

	if (!readArrowFile(filename, af_info, true))
	{
		elog(DEBUG2, "file '%s' is missing: %m", filename);
		return false;
	}
	if (af_info->recordBatches == NULL)
	{
		elog(DEBUG2, "arrow file '%s' contains no RecordBatch", filename);
		return false;
	}
	/* set up ArrowFileState */
	arrow_bstats = buildArrowStatsBinary(&af_info->footer, p_stat_attrs);
	for (int i=0; i < af_info->footer._num_recordBatches; i++)
	{
		ArrowBlock	     *block  = &af_info->footer.recordBatches[i];
		ArrowRecordBatch *rbatch = &af_info->recordBatches[i].body.recordBatch;
		RecordBatchState *rb_state;

		rb_state = __buildRecordBatchStateOne(&af_info->footer.schema,
											  af_state, i, block, rbatch);
		if (arrow_bstats)
			applyArrowStatsBinary(rb_state, arrow_bstats);
		af_state->rb_list = lappend(af_state->rb_list, rb_state);
	}
	releaseArrowStatsBinary(arrow_bstats);

	return true;
}

static arrowMetadataFieldCache *
__buildArrowMetadataFieldCache(RecordBatchFieldState *rb_field,
							   ArrowField *arrow_field,
							   arrowMetadataFieldCache *fcache_prev,
							   arrowMetadataCacheBlock **p_mc_block)
{
	arrowMetadataFieldCache *fcache;
	dlist_node	   *__dnode_prev = NULL;

	fcache = __allocMetadataFieldCache(p_mc_block);
	if (!fcache)
		return NULL;
	strcpy(fcache->attname, rb_field->attname);
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
	/* custom-metadata can be reused for the record-batch > 0 */
	if (fcache_prev)
		fcache->custom_metadata = fcache_prev->custom_metadata;
	else
	{
		arrowMetadataKeyValueCache *mc_kv_prev = NULL;
		arrowMetadataKeyValueCache *mc_kv;

		fcache->custom_metadata = NULL;
		for (int k=0; k < arrow_field->_num_custom_metadata; k++)
		{
			mc_kv = __allocMetadataKeyValueCache(p_mc_block,
												 &arrow_field->custom_metadata[k]);
			if (!mc_kv)
				return NULL;
			if (mc_kv_prev)
				mc_kv_prev->next = mc_kv;
			else
				fcache->custom_metadata = mc_kv;
			mc_kv_prev = mc_kv;
		}
	}

	/* walk down the child fields if any */
	fcache->num_children = rb_field->num_children;
		dlist_init(&fcache->children);
	for (int j=0; j < rb_field->num_children; j++)
	{
		arrowMetadataFieldCache *__fcache_prev = NULL;
		arrowMetadataFieldCache *__fcache;

		if (fcache_prev)
		{
			if (!__dnode_prev)
				__dnode_prev = dlist_head_node(&fcache_prev->children);
			else
				__dnode_prev = dlist_next_node(&fcache_prev->children,
											   __dnode_prev);
			__fcache_prev = dlist_container(arrowMetadataFieldCache,
											chain, __dnode_prev);
		}
		__fcache = __buildArrowMetadataFieldCache(&rb_field->children[j],
												  &arrow_field->children[j],
												  __fcache_prev,
												  p_mc_block);
		if (!__fcache)
			return NULL;
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
static arrowMetadataCacheBlock *
__buildArrowMetadataCacheNoLock(ArrowFileState *af_state,
								ArrowFileInfo *af_info)
{
	ArrowSchema	   *schema = &af_info->footer.schema;
	arrowMetadataCacheBlock *mc_block_head = NULL;
	arrowMetadataCacheBlock *mc_block_curr;
	arrowMetadataCache *mcache_prev = NULL;
	arrowMetadataCache *mcache;
	uint32_t	hindex;
	ListCell   *lc;

	Assert(list_length(af_state->rb_list) > 0);
	foreach (lc, af_state->rb_list)
	{
		RecordBatchState *rb_state = lfirst(lc);
		dlist_node	   *dnode_prev = NULL;

		if (!mc_block_head)
		{
			arrowMetadataKeyValueCache *mc_kv_prev = NULL;
			arrowMetadataKeyValueCache *mc_kv;

			mc_block_head = __allocMetadataCacheBlock();
			if (!mc_block_head)
				goto bailout;
			memcpy(&mc_block_head->stat_buf,
				   &af_state->stat_buf, sizeof(struct stat));
			mc_block_head->usage = MAXALIGN(offsetof(arrowMetadataCacheBlock,
													 mcache_head) +
											sizeof(arrowMetadataCache));
			mc_block_curr = mc_block_head;
			/* custom-metadata; must be setup after usage assignment */
			for (int k=0; k < schema->_num_custom_metadata; k++)
			{
				mc_kv = __allocMetadataKeyValueCache(&mc_block_curr,
													 &schema->custom_metadata[k]);
				if (!mc_kv)
					goto bailout;
				if (mc_kv_prev)
					mc_kv_prev->next = mc_kv;
				else
					mc_block_head->custom_metadata = mc_kv;
				mc_kv_prev = mc_kv;
			}
			/* metadata-cache for the first record-batch */
			mcache = &mc_block_head->mcache_head;
		}
		else
		{
			mcache = __allocMetadataCache(&mc_block_curr);
			if (!mcache)
				goto bailout;
		}
		memset(mcache, 0, sizeof(arrowMetadataCache));
		mcache->rb_index  = rb_state->rb_index;
		mcache->rb_offset = rb_state->rb_offset;
		mcache->rb_length = rb_state->rb_length;
		mcache->rb_nitems = rb_state->rb_nitems;
		mcache->nfields   = rb_state->nfields;
		dlist_init(&mcache->fields);
		if (mcache_prev)
			mcache_prev->next = mcache;
		for (int j=0; j < rb_state->nfields; j++)
		{
			arrowMetadataFieldCache *fcache_prev = NULL;
			arrowMetadataFieldCache *fcache;

			if (mcache_prev)
			{
				if (!dnode_prev)
					dnode_prev = dlist_head_node(&mcache_prev->fields);
				else
					dnode_prev = dlist_next_node(&mcache_prev->fields,
												 dnode_prev);
				fcache_prev = dlist_container(arrowMetadataFieldCache,
											  chain, dnode_prev);
			}
			Assert(j < schema->_num_fields);
			fcache = __buildArrowMetadataFieldCache(&rb_state->fields[j],
													&schema->fields[j],
													fcache_prev,
													&mc_block_curr);
			if (!fcache)
				goto bailout;
			dlist_push_tail(&mcache->fields, &fcache->chain);
		}
		mcache_prev = mcache;
	}
	/* chain to the list */
	hindex = arrowMetadataHashIndex(&af_state->stat_buf);
	dlist_push_tail(&arrow_metadata_cache->hash_slots[hindex],
					&mc_block_head->chain );
	SpinLockAcquire(&arrow_metadata_cache->lru_lock);
	gettimeofday(&mc_block_head->lru_tv, NULL);
	dlist_push_head(&arrow_metadata_cache->lru_list, &mc_block_head->lru_chain);
	SpinLockRelease(&arrow_metadata_cache->lru_lock);
	return mc_block_head;

bailout:
	if (mc_block_head)
		__releaseMetadataCacheBlock(mc_block_head);
	return NULL;
}

/*
 * __processVirtualColumn - process one token of the virtual token cstring
 */
static Datum
__processVirtualColumn(Form_pg_attribute attr,
					   char *vc_key,
					   char *vc_value,
					   Relation frel,
					   const char *filename)
{
	MemoryContext oldcxt = CurrentMemoryContext;
	Oid		type_input;
	Oid		type_ioparam;
	Datum	datum;

	getTypeInputInfo(attr->atttypid,
					 &type_input,
					 &type_ioparam);
	PG_TRY();
	{
		datum = OidInputFunctionCall(type_input,
									 vc_value,
									 type_ioparam,
									 attr->atttypmod);
	}
	PG_CATCH();
	{
		MemoryContext errcxt = MemoryContextSwitchTo(oldcxt);
		ErrorData  *edata = CopyErrorData();

		ereport(Max(ERROR, edata->elevel),
				errmsg("(%s:%d) %s",
					   edata->filename,
					   edata->lineno,
					   edata->message),
				errdetail("arrow_fdw: processing virtual column '%s' of the file '%s' at the attribute '%s' of foreign table '%s'",
						  vc_key, filename,
						  NameStr(attr->attname),
						  RelationGetRelationName(frel)));
		MemoryContextSwitchTo(errcxt);
	}
	PG_END_TRY();

	return datum;
}

static inline const char *
__fetchVirtualSourceSpecial(ArrowFileState *af_state, const char *key)
{
	if (*key == '@')
	{
		if (strcmp(key, "@pathname") == 0)
			return af_state->filename;
		else if (strcmp(key, "@filename") == 0)
		{
			const char *pos = strrchr(af_state->filename, '/');

			if (pos)
				return pos+1;
			return af_state->filename;
		}
	}
	return NULL;
}

static List *
__fetchVirtualSourceByCache(ArrowFileState *af_state,
							arrowMetadataCacheBlock *mc_block,
							List *virtual_columns,
							List *source_fields)
{
	List	   *results = NIL;
	ListCell   *lc1, *lc2;

	foreach (lc1, source_fields)
	{
		const char *src = lfirst(lc1);
		const char *value = NULL;

		if (strncmp(src, "virtual:", 8) == 0)
		{
			const char *key = src + 8;

			value = __fetchVirtualSourceSpecial(af_state, key);
			if (value)
				goto found;
			foreach (lc2, virtual_columns)
			{
				virtualColumnDef *vcdef = lfirst(lc2);

				if (strcmp(key, vcdef->key) == 0)
				{
					value = vcdef->value;
					goto found;
				}
			}
		}
		else if (strncmp(src, "metadata:", 9) == 0 ||
				 strncmp(src, "metadata-split:", 15) == 0)
		{
			char   *key = alloca(strlen(src));
			char   *pos;

			strcpy(key, strchr(src, ':') + 1);
			pos = strchr(key, '.');
			if (pos)
				*pos++ = '\0';
			if (!pos)
			{
				/* fetch custom-metadata from Schema */
				arrowMetadataKeyValueCache *mc_kv = mc_block->custom_metadata;

				while (mc_kv)
				{
					if (strcmp(mc_kv->key, key) == 0)
					{
						value = mc_kv->value;
						goto found;
					}
					mc_kv = mc_kv->next;
				}
			}
			else
			{
				/* fetch custom-metadata from Fields */
				dlist_iter	iter;

				dlist_foreach (iter, &mc_block->mcache_head.fields)
				{
					arrowMetadataFieldCache *fcache = dlist_container(arrowMetadataFieldCache,
																	  chain, iter.cur);
					if (strcmp(fcache->attname, key) == 0)
					{
						arrowMetadataKeyValueCache *mc_kv = fcache->custom_metadata;

						while (mc_kv)
						{
							if (strcmp(mc_kv->key, pos) == 0)
							{
								value = mc_kv->value;
								goto found;
							}
							mc_kv = mc_kv->next;
						}
					}
				}
			}
		}
	found:
		results = lappend(results, value ? pstrdup(value) : NULL);
	}
	return results;
}

static List *
__fetchVirtualSourceByFile(ArrowFileState *af_state,
						   ArrowFileInfo *af_info,
						   List *virtual_columns,
						   List *source_fields)
{
	List	   *results = NIL;
	ListCell   *lc1, *lc2;

	foreach (lc1, source_fields)
	{
		const char *src = lfirst(lc1);
		const char *value = NULL;

		if (strncmp(src, "virtual:", 8) == 0)
		{
			const char *key = src + 8;

			value = __fetchVirtualSourceSpecial(af_state, key);
			if (value)
				goto found;
			foreach (lc2, virtual_columns)
			{
				virtualColumnDef *vcdef = lfirst(lc2);

				if (strcmp(key, vcdef->key) == 0)
				{
					value = vcdef->value;
					goto found;
				}
			}
		}
		else if (strncmp(src, "metadata:", 9) == 0 ||
				 strncmp(src, "metadata-split:", 15) == 0)
		{
			ArrowSchema	*schema = &af_info->footer.schema;
			char   *key = alloca(strlen(src));
			char   *pos;

			strcpy(key, strchr(src, ':') + 1);
			pos = strchr(key, '.');
			if (pos)
				*pos++ = '\0';

			if (!pos)
			{
				/* fetch custom-metadata from Schema */
				for (int i=0; i < schema->_num_custom_metadata; i++)
				{
					ArrowKeyValue  *kv = &schema->custom_metadata[i];

					if (strcmp(kv->key, key) == 0)
					{
						value = kv->value;
						goto found;
					}
				}
			}
			else
			{
				/* fetch custom-metadata from Fields */
				for (int i=0; i < schema->_num_fields; i++)
				{
					ArrowField *field = &schema->fields[i];

					if (strcmp(field->name, key) == 0)
					{
						for (int k=0; k < field->_num_custom_metadata; k++)
						{
							ArrowKeyValue  *kv = &field->custom_metadata[k];

							if (strcmp(kv->key, pos) == 0)
							{
								value = kv->value;
								goto found;
							}
						}
					}
				}
			}
		}
	found:
		results = lappend(results, value ? pstrdup(value) : NULL);
	}
	return results;
}

static ArrowFileState *
BuildArrowFileState(Relation frel,
					const char *filename,
					List *source_fields,
					List *virtual_columns,
					Bitmapset **p_stat_pg_attrs)
{
	TupleDesc		tupdesc = RelationGetDescr(frel);
	arrowMetadataCacheBlock *mc_block;
	ArrowFileState *af_state;
	RecordBatchState *rb_state;
	Bitmapset	   *stat_arrow_attrs = NULL;
	Bitmapset	   *stat_pg_attrs = NULL;
	struct stat		stat_buf;
	List		   *virtual_sources = NIL;
	ListCell	   *lc1, *lc2;
	int				j;

	if (stat(filename, &stat_buf) != 0)
		elog(ERROR, "failed on stat('%s'): %m", filename);
	af_state = palloc0(offsetof(ArrowFileState, attrs[tupdesc->natts]));
	af_state->filename = pstrdup(filename);
	memcpy(&af_state->stat_buf, &stat_buf, sizeof(struct stat));
	af_state->ncols = tupdesc->natts;
	
	LWLockAcquire(&arrow_metadata_cache->mutex, LW_SHARED);
	mc_block = lookupArrowMetadataCache(&stat_buf, false);
	if (mc_block)
	{
		/* found a valid metadata-cache */
		__setupArrowFileStateByCache(af_state,
									 filename,
									 mc_block,
									 &stat_arrow_attrs);
		/* extract virtual column source info */
		virtual_sources = __fetchVirtualSourceByCache(af_state,
													  mc_block,
													  virtual_columns,
													  source_fields);
	}
	else
	{
		ArrowFileInfo af_info;

		LWLockRelease(&arrow_metadata_cache->mutex);
		/* here is no valid metadata-cache, so build it from the raw file */
		if (!__setupArrowFileStateByFile(af_state,
										 filename,
										 &af_info,
										 &stat_arrow_attrs))
			return NULL;	/* file not found? */
		/* extract virtual column source info */
		virtual_sources = __fetchVirtualSourceByFile(af_state,
													 &af_info,
													 virtual_columns,
													 source_fields);
		LWLockAcquire(&arrow_metadata_cache->mutex, LW_EXCLUSIVE);
		mc_block = lookupArrowMetadataCache(&af_state->stat_buf, true);
		if (!mc_block)
			__buildArrowMetadataCacheNoLock(af_state, &af_info);
	}
	LWLockRelease(&arrow_metadata_cache->mutex);

	/*
	 * Maps PG-attribute on a particular Arrow-field, or virtual-column
	 * according to the column option
	 */
	rb_state = linitial(af_state->rb_list);
	Assert(tupdesc->natts == af_state->ncols &&
		   tupdesc->natts == list_length(source_fields) &&
		   tupdesc->natts == list_length(virtual_sources));
	j = 0;
	forboth (lc1, source_fields,
			 lc2, virtual_sources)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);
		char	   *sfield = lfirst(lc1);
		char	   *vsource = lfirst(lc2);

		if (attr->attisdropped)
		{
			af_state->attrs[j].field_index = __FIELD_INDEX_SPECIAL__VIRTUAL_PER_FILE;
			af_state->attrs[j].virtual_isnull = true;
			af_state->attrs[j].virtual_datum = 0;
		}
		else if (strncmp(sfield, "field:", 6) == 0)
		{
			const char *field_name = sfield + 6;
			int			field_index = -1;

			for (int k=0; k < rb_state->nfields; k++)
			{
				RecordBatchFieldState *field = &rb_state->fields[k];

				if (strcmp(field->attname, field_name) == 0)
				{
					/* also checks data type compatibility */
					if (IsBinaryCoercible(field->atttypid,
										  attr->atttypid))
					{
						field_index = k;
						break;
					}
					elog(ERROR, "arrow_fdw: foreign table '%s' of '%s' is not compatible to '%s' of '%s'",
						 NameStr(attr->attname),
						 RelationGetRelationName(frel),
						 field->attname,
						 filename);
				}
			}
			if (field_index < 0)
				elog(ERROR, "arrow_fdw: foreign table '%s' of '%s' could not find out the field '%s' on the arrow file '%s'",
					 NameStr(attr->attname),
					 RelationGetRelationName(frel),
					 field_name,
					 filename);
			af_state->attrs[j].field_index = field_index;
			af_state->attrs[j].virtual_isnull = true;
			af_state->attrs[j].virtual_datum = 0;

			if (bms_is_member(field_index, stat_arrow_attrs))
				stat_pg_attrs = bms_add_member(stat_pg_attrs, attr->attnum -
											   FirstLowInvalidHeapAttributeNumber);
		}
		else if (vsource && (strncmp(sfield, "virtual:", 8) == 0 ||
							 strncmp(sfield, "metadata:", 9) == 0))
		{
			Datum	datum = __processVirtualColumn(attr,
												   sfield,
												   vsource,
												   frel, filename);
			af_state->attrs[j].field_index = __FIELD_INDEX_SPECIAL__VIRTUAL_PER_FILE;
			af_state->attrs[j].virtual_isnull = false;
			af_state->attrs[j].virtual_datum = datum;
			/*
			 * The virtual value is immutable to a particular record-batch,
			 * so we can consider it also works as min/max statistics to skip
			 * obviously unmatched record-batches.
			 */
			stat_pg_attrs = bms_add_member(stat_pg_attrs, attr->attnum -
										   FirstLowInvalidHeapAttributeNumber);
		}
		else if (vsource && strncmp(sfield, "metadata-split:", 15) == 0)
		{
			ListCell   *cell;
			char	   *buffer = pstrdup(vsource);
			char	   *tok, *pos;

			tok = strtok_r(buffer, ",", &pos);
			foreach (cell, af_state->rb_list)
			{
				RecordBatchState *__rb_state = lfirst(cell);

				if (!tok)
				{
					__rb_state->virtual_isnull = true;
					__rb_state->virtual_datum = 0;
				}
				else
				{
					Datum	datum = __processVirtualColumn(attr,
														   sfield,
														   tok,
														   frel,
														   filename);
					__rb_state->virtual_datum = datum;
					__rb_state->virtual_isnull = false;
					tok = strtok_r(NULL, ",", &pos);
				}
			}
			af_state->attrs[j].field_index = __FIELD_INDEX_SPECIAL__VIRTUAL_PER_RECORD_BATCH;
			af_state->attrs[j].virtual_isnull = true;
			af_state->attrs[j].virtual_datum = 0;
			/* see the comment above */
			stat_pg_attrs = bms_add_member(stat_pg_attrs, attr->attnum -
										   FirstLowInvalidHeapAttributeNumber);
			/* cleanup */
			pfree(buffer);
		}
		else
		{
			/* just put a virtual NULL */
			af_state->attrs[j].field_index = __FIELD_INDEX_SPECIAL__VIRTUAL_PER_FILE;
			af_state->attrs[j].virtual_isnull = true;
			af_state->attrs[j].virtual_datum = 0;
		}
		j++;
	}
	bms_free(stat_arrow_attrs);
	if (p_stat_pg_attrs)
		*p_stat_pg_attrs = stat_pg_attrs;
	else
		bms_free(stat_pg_attrs);
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
gpumask_t
GetOptimalGpusForArrowFdw(PlannerInfo *root, RelOptInfo *baserel)
{
	List	   *priv_list = (List *)baserel->fdw_private;
	gpumask_t	optimal_gpus = 0;

	if (baseRelIsArrowFdw(baserel) &&
		IsA(priv_list, List) && list_length(priv_list) == 2)
	{
		const char *relname = getRelOptInfoName(root, baserel);
		List	   *arrow_files_list = linitial(priv_list);
		ListCell   *lc;

		foreach (lc, arrow_files_list)
		{
			ArrowFileState *af_state = lfirst(lc);
			gpumask_t		__optimal_gpus;

			__optimal_gpus = GetOptimalGpuForFile(af_state->filename);
			if (__optimal_gpus == INVALID_GPUMASK)
				__optimal_gpus = 0;
			if (lc == list_head(arrow_files_list))
			{
				optimal_gpus = __optimal_gpus;
				if (optimal_gpus == 0)
					__Debug("foreign-table='%s' arrow-file='%s' has no schedulable GPUs", relname, af_state->filename);
			}
			else
			{
				__optimal_gpus &= optimal_gpus;
				if (optimal_gpus != __optimal_gpus)
					__Debug("foreign-table='%s' arrow-file='%s' reduced GPUs-set %08lx => %08lx", relname, af_state->filename, optimal_gpus, __optimal_gpus);
				optimal_gpus = __optimal_gpus;
			}
			if (optimal_gpus == 0)
				break;
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
	const DpuStorageEntry *ds_entry = NULL;
	List	   *priv_list = (List *)baserel->fdw_private;

	if (baseRelIsArrowFdw(baserel) &&
		IsA(priv_list, List) && list_length(priv_list) == 2)
	{
		List	   *af_list = linitial(priv_list);
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
 * arrowFdwExcludeFileNamesByPattern
 */
static List *
arrowFdwExcludeFileNamesByPattern(List *filesList,
								  const char *pattern,
								  List **p_virtAttrsList)
{
	List	   *results = NIL;	/* only valid files */
	List	   *attrsList = NIL;
	ListCell   *lc;

	foreach (lc, filesList)
	{
		String *path = lfirst(lc);
		List   *attrKinds = NIL;
		List   *attrKeys = NIL;
		List   *attrValues = NIL;

		if (pathNameMatchByPattern(strVal(path),
								   pattern,
								   &attrKinds,
								   &attrKeys,
								   &attrValues))
		{
			if (p_virtAttrsList)
			{
				List	   *vcdef_list = NIL;
				ListCell   *lc1, *lc2, *lc3;

				forthree (lc1, attrKinds,
						  lc2, attrKeys,
						  lc3, attrValues)
				{
					int			kind  = lfirst_int(lc1);
					const char *key   = lfirst(lc2);
					const char *value = lfirst(lc3);
					char	   *pos;
					virtualColumnDef *vcdef;

					vcdef = palloc(offsetof(virtualColumnDef, buf) +
								   strlen(key) + strlen(value) + 2);
					vcdef->kind = kind;
					pos = vcdef->buf;
					strcpy(pos, key);
					vcdef->key = pos;

					pos += strlen(key) + 1;
					strcpy(pos, value);
					vcdef->value = pos;

					vcdef_list = lappend(vcdef_list, vcdef);
				}
				attrsList = lappend(attrsList, vcdef_list);
			}
			results = lappend(results, path);

			list_free_deep(attrKeys);
			list_free_deep(attrValues);
		}
	}
	if (p_virtAttrsList)
	{
		Assert(list_length(results) == list_length(attrsList));
		*p_virtAttrsList = attrsList;
	}
	return results;
}

/*
 * arrowFdwExtractFilesList
 */
static List *
arrowFdwExtractFilesList(List *options_list,
						 List **p_virtAttrsList,
						 int *p_parallel_nworkers)
{
	List	   *filesList = NIL;
	char	   *dir_path = NULL;
	char	   *dir_suffix = NULL;
	char	   *pattern = NULL;
	int			parallel_nworkers = -1;
	ListCell   *lc;

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
		else if (strcmp(defel->defname, "pattern") == 0)
		{
			if (pattern)
				elog(ERROR, "'pattern' appeared twice");
			pattern = strVal(defel->arg);
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
	/* exclude the file names by pattern */
	if (pattern)
	{
		filesList = arrowFdwExcludeFileNamesByPattern(filesList, pattern,
													  p_virtAttrsList);
	}
	else if (p_virtAttrsList)
	{
		/* add empty file attributes list for forboth() macro */
		List   *virtAttrsList = NIL;

		foreach (lc, filesList)
			virtAttrsList = lappend(virtAttrsList, NULL);
		*p_virtAttrsList = virtAttrsList;
	}
	if (p_parallel_nworkers)
		*p_parallel_nworkers = parallel_nworkers;
	return filesList;
}

/*
 * arrowFdwExtractSourceFields
 */
static List *
arrowFdwExtractSourceFields(Relation frel)
{
	Oid			frelid = RelationGetRelid(frel);
	TupleDesc	tupdesc = RelationGetDescr(frel);
	List	   *results = NIL;

	for (int j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);
		List	   *options = GetForeignColumnOptions(frelid, attr->attnum);
		ListCell   *lc;
		const char *field_name = NULL;
		const char *virtual_key = NULL;
		const char *virtual_metadata = NULL;
		const char *virtual_metadata_split = NULL;

		if (attr->attisdropped)
		{
			/* always NULL */
			results = lappend(results, "none");
			continue;
		}
		/* check column options to identify the source */
		foreach (lc, options)
		{
			DefElem *defel = lfirst(lc);

			Assert(IsA(defel->arg, String));
			if (strcmp(defel->defname, "field") == 0)
				field_name = strVal(defel->arg);
			else if (strcmp(defel->defname, "virtual") == 0)
				virtual_key = strVal(defel->arg);
			else if (strcmp(defel->defname, "virtual_metadata") == 0)
				virtual_metadata = strVal(defel->arg);
			else if (strcmp(defel->defname, "virtual_metadata_split") == 0)
				virtual_metadata_split = strVal(defel->arg);
			else
			{
				elog(ERROR, "unknown foreign table options in '%s' of '%s'",
					 NameStr(attr->attname),
					 RelationGetRelationName(frel));
			}
		}
		if ((field_name != NULL ? 1 : 0) +
			(virtual_key != NULL ? 1 : 0) +
			(virtual_metadata != NULL ? 1 : 0) +
			(virtual_metadata_split != NULL ? 1 : 0) > 1)
			elog(ERROR, "arrow_fdw: column option 'field', 'virtual', 'virtual_metadata' and 'virtual_metadata_split' must be mutually exclusive");

		if (virtual_key)
			results = lappend(results, psprintf("virtual:%s", virtual_key));
		else if (virtual_metadata)
			results = lappend(results, psprintf("metadata:%s", virtual_metadata));
		else if (virtual_metadata_split)
			results = lappend(results, psprintf("metadata-split:%s", virtual_metadata_split));
		else
		{
			if (!field_name)
				field_name = NameStr(attr->attname);
			results = lappend(results, psprintf("field:%s", field_name));
		}
	}
	return results;
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
	off_t		kds_head_sz;
	int32_t		depth;
	int32_t		io_index;
	strom_io_chunk ioc[FLEXIBLE_ARRAY_MEMBER];
} arrowFdwSetupIOContext;

static void
__setupIOvectorField(arrowFdwSetupIOContext *con,
					 uint32_t chunk_align,
					 off_t    chunk_offset,
					 size_t   chunk_length,
					 uint64_t *p_cmeta_offset,
					 uint64_t *p_cmeta_length)
{
	off_t		f_pos = con->rb_offset + chunk_offset;
	off_t		f_gap;
	off_t		f_base;
	off_t		m_offset;
	strom_io_chunk *ioc;

	/*
	 * Round up chunk_length to 64bit boundary.
	 * Apache Arrow does not require the chunk_length is aligned to 64bit boundary,
	 * chunk_offset must be aligned to 64bit (512bit recommended).
	 * So, some extra bytes (up to -7bytes) may improve the potency of i/o chunk
	 * consolication.
	 */

	if (f_pos >= con->f_offset &&
		(f_pos & ~PAGE_MASK) == (con->f_offset & ~PAGE_MASK))
	{
		/*
		 * we can consolidate the two i/o chunks, if file position of the next
		 * chunk (f_pos) and the current file tail position (con->f_offset) locate
		 * within the same file page, and gap bytes does not break alignment.
		 */
		f_gap = f_pos - con->f_offset;
		m_offset = con->m_offset + f_gap;

		if (m_offset == TYPEALIGN(chunk_align, con->m_offset))
		{
			/* put the gap bytes, if any */
			if (f_gap > 0)
			{
				con->m_offset += f_gap;
				con->f_offset += f_gap;
			}
			*p_cmeta_offset = con->kds_head_sz + con->m_offset;
			*p_cmeta_length = MAXALIGN(chunk_length);
			con->m_offset += chunk_length;
			con->f_offset += chunk_length;
			return;
		}
	}
	/*
	 * Elsewhere, we have to close the current i/o chunk once, then
	 * restart a new i/o chunk to load the disjoin chunks.
	 */
	if (con->io_index < 0)
		con->io_index = 0;		/* no current active i/o chunk */
	else
	{
		off_t	f_tail = PAGE_ALIGN(con->f_offset);

		ioc = &con->ioc[con->io_index++];
		ioc->nr_pages = (f_tail / PAGE_SIZE) - ioc->fchunk_id;
		con->m_offset += (f_tail - con->f_offset);	/* padding bytes */
	}

	f_base = PAGE_ALIGN_DOWN(f_pos);
	f_gap = f_pos - f_base;
	m_offset = TYPEALIGN(chunk_align, con->m_offset + f_gap);

	ioc = &con->ioc[con->io_index];
	ioc->m_offset = m_offset - f_gap;
	ioc->fchunk_id = f_base / PAGE_SIZE;

	*p_cmeta_offset = con->kds_head_sz + m_offset;
	*p_cmeta_length = MAXALIGN(chunk_length);
	con->m_offset = m_offset + chunk_length;
	con->f_offset = f_pos + chunk_length;
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
							 sizeof(int64_t),	/* 64bit alignment */
							 rb_field->nullmap_offset,
							 rb_field->nullmap_length,
							 &cmeta->nullmap_offset,
							 &cmeta->nullmap_length);
		//elog(INFO, "D%d att[%d] nullmap=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, rb_field->nullmap_offset, rb_field->nullmap_length, con->m_offset, con->f_offset);
	}
	if (rb_field->values_length > 0)
	{
		__setupIOvectorField(con,
							 rb_field->attopts.align,
							 rb_field->values_offset,
							 rb_field->values_length,
							 &cmeta->values_offset,
							 &cmeta->values_length);
		//elog(INFO, "D%d att[%d] values=%lu,%lu m_offset=%lu f_offset=%lu", con->depth, index, rb_field->values_offset, rb_field->values_length, con->m_offset, con->f_offset);
	}
	if (rb_field->extra_length > 0)
	{
		__setupIOvectorField(con,
							 sizeof(int64_t),	/* 64bit alignment */
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
	ArrowFileState *af_state = rb_state->af_state;
	arrowFdwSetupIOContext *con;
	strom_io_vector *iovec;
	unsigned int	nr_chunks = 0;

	Assert(kds->format == KDS_FORMAT_ARROW &&
		   kds->ncols <= kds->nr_colmeta);
	con = alloca(offsetof(arrowFdwSetupIOContext,
						  ioc[3 * kds->nr_colmeta]));
	con->rb_offset = rb_state->rb_offset;
	con->f_offset  = ~0UL;	/* invalid offset */
	con->m_offset  = 0;
	con->kds_head_sz = KDS_HEAD_LENGTH(kds) + kds->arrow_virtual_usage;
	con->depth = 0;
	con->io_index = -1;		/* invalid index */
	for (int j=0; j < kds->ncols; j++)
	{
		kern_colmeta *cmeta = &kds->colmeta[j];
		int			attidx = j + 1 - FirstLowInvalidHeapAttributeNumber;

		if (bms_is_member(attidx, referenced) ||
			bms_is_member(-FirstLowInvalidHeapAttributeNumber, referenced))
		{
			int		field_index = af_state->attrs[j].field_index;

			if (field_index < 0)
			{
				/* !!!virtual column!!! */
				Assert(cmeta->virtual_offset != 0);
			}
			else
			{
				RecordBatchFieldState *rb_field = &rb_state->fields[field_index];

				arrowFdwSetupIOvectorField(con, rb_field, kds, cmeta);
			}
		}
		else
			cmeta->atttypkind = TYPE_KIND__NULL;	/* unreferenced */
	}
	if (con->io_index >= 0)
	{
		/* close the last I/O chunks */
		strom_io_chunk *ioc = &con->ioc[con->io_index++];

		ioc->nr_pages = (PAGE_ALIGN(con->f_offset) / PAGE_SIZE - ioc->fchunk_id);
		con->m_offset += PAGE_SIZE * ioc->nr_pages;
		nr_chunks = con->io_index;
	}
	kds->length = con->kds_head_sz + con->m_offset;

	iovec = palloc0(offsetof(strom_io_vector, ioc[nr_chunks]));
	iovec->nr_chunks = nr_chunks;
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
				 cmeta->nullmap_offset,
				 cmeta->nullmap_length,
				 cmeta->values_offset,
				 cmeta->values_length,
				 cmeta->extra_offset,
				 cmeta->extra_length);
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

static void
__arrowKdsAssignVirtualColumns(kern_data_store *kds,
							   kern_colmeta *cmeta,
							   bool virtual_isnull,
							   Datum virtual_datum,
							   StringInfo chunk_buffer)
{
	if (virtual_isnull)
	{
		cmeta->virtual_offset = -1;
	}
	else
	{
		Assert(chunk_buffer->len == MAXALIGN(chunk_buffer->len));
		cmeta->virtual_offset = (chunk_buffer->data +
								 chunk_buffer->len - (char *)kds);
		if (cmeta->attbyval)
		{
			appendBinaryStringInfo(chunk_buffer,
								   (char *)&virtual_datum,
								   cmeta->attlen);
		}
		else if (cmeta->attlen > 0)
		{
			appendBinaryStringInfo(chunk_buffer,
								   DatumGetPointer(virtual_datum),
								   cmeta->attlen);
		}
		else if (cmeta->attlen == -1)
		{
			appendBinaryStringInfo(chunk_buffer,
								   DatumGetPointer(virtual_datum),
								   VARSIZE_ANY(virtual_datum));
		}
		else
		{
			elog(ERROR, "unknown type length: %d", cmeta->attlen);
		}
		__appendZeroStringInfo(chunk_buffer, 0);
	}
}

static strom_io_vector *
arrowFdwLoadRecordBatch(Relation relation,
						Bitmapset *referenced,
						RecordBatchState *rb_state,
						StringInfo chunk_buffer)
{
	ArrowFileState *af_state = rb_state->af_state;
	TupleDesc	tupdesc = RelationGetDescr(relation);
	size_t		head_off = chunk_buffer->len;
	kern_data_store *kds;

	/* setup KDS and I/O-vector */
	enlargeStringInfo(chunk_buffer, estimate_kern_data_store(tupdesc));
	kds = (kern_data_store *)(chunk_buffer->data + head_off);
	setup_kern_data_store(kds, tupdesc, 0, KDS_FORMAT_ARROW);
	kds->nitems = rb_state->rb_nitems;
	kds->table_oid = RelationGetRelid(relation);
	chunk_buffer->len += KDS_HEAD_LENGTH(kds);

	Assert(kds->ncols == af_state->ncols);
	for (int j=0; j < kds->ncols; j++)
	{
		int		field_index = af_state->attrs[j].field_index;

		if (field_index >= 0)
		{
			Assert(field_index < rb_state->nfields);
			__arrowKdsAssignAttrOptions(kds,
										&kds->colmeta[j],
										&rb_state->fields[field_index]);
		}
		else if (field_index == __FIELD_INDEX_SPECIAL__VIRTUAL_PER_FILE)
		{
			__arrowKdsAssignVirtualColumns(kds,
										   &kds->colmeta[j],
										   af_state->attrs[j].virtual_isnull,
										   af_state->attrs[j].virtual_datum,
										   chunk_buffer);
			/*
			 * 'chunk_buffer' may be expanded during assignment of virtual
			 * columns, because repalloc() may change the base address,
			 * so kds must be refreshed.
			 */
			kds = (kern_data_store *)(chunk_buffer->data + head_off);
		}
		else if (field_index == __FIELD_INDEX_SPECIAL__VIRTUAL_PER_RECORD_BATCH)
		{
			__arrowKdsAssignVirtualColumns(kds,
										   &kds->colmeta[j],
										   rb_state->virtual_isnull,
										   rb_state->virtual_datum,
										   chunk_buffer);
			/* see the comment above */
			kds = (kern_data_store *)(chunk_buffer->data + head_off);
		}
		else
		{
			elog(ERROR, "Bug? unexpected field-index (%d)", field_index);
		}
	}
	kds->arrow_virtual_usage = (chunk_buffer->len
								- (head_off + KDS_HEAD_LENGTH(kds)));
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
	base = (char *)kds + KDS_HEAD_LENGTH(kds) + kds->arrow_virtual_usage;
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
	List		   *sourceFields;
	List		   *virtualColumnsList;
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
	filesList = arrowFdwExtractFilesList(ft->options,
										 &virtualColumnsList,
										 &parallel_nworkers);
	sourceFields = arrowFdwExtractSourceFields(frel);
	forboth (lc1, filesList,
			 lc2, virtualColumnsList)
	{
		ArrowFileState *af_state;
		const char *fname = strVal(lfirst(lc1));
		List	   *virtual_columns = lfirst(lc2);
		ListCell   *cell;

		af_state = BuildArrowFileState(frel, fname,
									   sourceFields,
									   virtual_columns, NULL);
		if (!af_state)
			continue;

		/*
		 * Size calculation based the record-batch metadata
		 */
		foreach (cell, af_state->rb_list)
		{
			RecordBatchState *rb_state = lfirst(cell);

			//XXX - fix to support column-field mapping

			/* whole-row reference? */
			if (bms_is_member(-FirstLowInvalidHeapAttributeNumber, referenced))
			{
				totalLen += rb_state->rb_length;
			}
			else
			{
				int		i, j, k;

				for (k = bms_next_member(referenced, -1);
					 k >= 0;
					 k = bms_next_member(referenced, k))
				{
					j = k + FirstLowInvalidHeapAttributeNumber;
					if (j <= 0 || j > af_state->ncols)
						continue;
					i = af_state->attrs[j-1].field_index;
					if (i >= 0 && i < rb_state->nfields)
						totalLen += __recordBatchFieldLength(&rb_state->fields[i]);
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
#if PG_VERSION_NUM >= 170000
									NIL,	/* no restrict-info of Join push-down */
#endif
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
//FIXME: Just a workaround to add inner_path of GpuJoin in parallel mode.
//       We should add non-parallel inner_path
//		if (num_workers == 0)
//			return;

		fpath = create_foreignscan_path(root,
										baserel,
										NULL,	/* default pathtarget */
										-1.0,	/* dummy */
										-1.0,	/* dummy */
										-1.0,	/* dummy */
										NIL,	/* no pathkeys */
										required_outer,
										NULL,	/* no extra plan */
#if PG_VERSION_NUM >= 170000
										NIL,	/* no restrict-info of Join push-down */
#endif
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
					   kern_colmeta *cmeta,
					   size_t index, bool *p_isnull)
{
	struct varlena *res = NULL;
	const void *addr;
	int			length;

	addr = KDS_ARROW_REF_VARLENA32_DATUM(kds, cmeta, index, &length);
	if (!addr)
		*p_isnull = true;
	else
	{
		*p_isnull = false;
		res = palloc(VARHDRSZ + length);
		memcpy(res->vl_dat, addr, length);
		SET_VARSIZE(res, VARHDRSZ + length);
	}
	return PointerGetDatum(res);
}

static Datum
pg_varlena64_arrow_ref(kern_data_store *kds,
					   kern_colmeta *cmeta,
					   size_t index, bool *p_isnull)
{
	struct varlena *res = NULL;
	const void *addr;
	int			length;

	addr = KDS_ARROW_REF_VARLENA64_DATUM(kds, cmeta, index, &length);
	if (!addr)
		*p_isnull = true;
	else
	{
		*p_isnull = false;
		res = palloc(VARHDRSZ + length);
		memcpy(res->vl_dat, addr, length);
		SET_VARSIZE(res, VARHDRSZ + length);
	}
	return PointerGetDatum(res);
}

static Datum
pg_bpchar_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	char	   *values = ((char *)kds + cmeta->values_offset);
	size_t		length = cmeta->values_length;
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
	uint8_t	   *bitmap = (uint8_t *)kds + cmeta->values_offset;
	size_t		length = cmeta->values_length;
	bool		rv;

	if (sizeof(uint8_t) * (index>>3) >= length)
		elog(ERROR, "corruption? bool points out of range");
	rv = ((bitmap[index>>3] & (1<<(index&7))) != 0);
	return BoolGetDatum(rv);
}

static Datum
pg_simple_arrow_ref(kern_data_store *kds,
					kern_colmeta *cmeta, size_t index)
{
	int32_t		unitsz = cmeta->attopts.unitsz;
	char	   *values = (char *)kds + cmeta->values_offset;
	size_t		length = cmeta->values_length;
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
	char	   *base = (char *)kds + cmeta->values_offset;
	size_t		length = cmeta->values_length;
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
	char	   *base = (char *)kds + cmeta->values_offset;
	size_t		length = cmeta->values_length;
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
	char	   *base = (char *)kds + cmeta->values_offset;
	size_t		length = cmeta->values_length;
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
	char	   *base = (char *)kds + cmeta->values_offset;
	size_t		length = cmeta->values_length;
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
	char	   *base = (char *)kds + cmeta->values_offset;
	size_t		length = cmeta->values_length;
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
	char   *base = (char *)kds + cmeta->values_offset;
	size_t	length = cmeta->values_length;

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
	char   *base = (char *)kds + cmeta->values_offset;
	size_t	length = cmeta->values_length;
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

	if (cmeta->virtual_offset != 0)
	{
		if (cmeta->virtual_offset < 0)
			isnull = true;
		else if (cmeta->attbyval)
		{
			void   *addr = ((char *)kds + cmeta->virtual_offset);

			switch (cmeta->attlen)
			{
				case 1:	datum = *((uint8_t  *)addr); break;
				case 2: datum = *((uint16_t *)addr); break;
				case 4: datum = *((uint32_t *)addr); break;
				case 8: datum = *((uint64_t *)addr); break;
				default:
					elog(ERROR, "unexpected inline type length: %d", cmeta->attlen);
			}
		}
		else
			datum = PointerGetDatum((char *)kds + cmeta->virtual_offset);
		goto out;
	}

	if (KDS_ARROW_CHECK_ISNULL(kds, cmeta, index))
	{
		isnull = true;
		goto out;
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
			datum = pg_varlena32_arrow_ref(kds, cmeta, index, &isnull);
			break;
		case ArrowType__LargeUtf8:
		case ArrowType__LargeBinary:
			datum = pg_varlena64_arrow_ref(kds, cmeta, index, &isnull);
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
				if (sizeof(uint32_t) * (index+2) > cmeta->values_length)
					elog(ERROR, "Bug? array index is out of range");
				smeta = &kds->colmeta[cmeta->idx_subattrs];
				offset = (uint32_t *)((char *)kds + cmeta->values_offset);
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
				if (sizeof(uint64_t) * (index+2) > cmeta->values_length)
					elog(ERROR, "Bug? array index is out of range");
				smeta = &kds->colmeta[cmeta->idx_subattrs];
				offset = (uint64_t *)((char *)kds + cmeta->values_offset);
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
				   pgstromTaskState *pts)
{
	Relation		frel = ss->ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(frel);
	ForeignTable   *ft = GetForeignTable(RelationGetRelid(frel));
	Bitmapset	   *referenced = NULL;
	Bitmapset	   *stat_attrs = NULL;
	gpumask_t		optimal_gpus = 0UL;
	const DpuStorageEntry *ds_entry = NULL;
	bool			whole_row_ref = false;
	List		   *filesList;
	List		   *sourceFields;
	List		   *virtualColumnsList;
	List		   *af_states_list = NIL;
	uint32_t		rb_nrooms = 0;
	uint32_t		rb_nitems = 0;
	ArrowFdwState  *arrow_state;
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
	filesList = arrowFdwExtractFilesList(ft->options,
										 &virtualColumnsList, NULL);
	sourceFields = arrowFdwExtractSourceFields(frel);
	forboth (lc1, filesList,
			 lc2, virtualColumnsList)
	{
		char	   *fname = strVal(lfirst(lc1));
		List	   *virtual_columns = lfirst(lc2);
		ArrowFileState *af_state;

		af_state = BuildArrowFileState(frel, fname,
									   sourceFields,
									   virtual_columns,
									   &stat_attrs);
		if (af_state)
		{
			rb_nrooms += list_length(af_state->rb_list);
			if (pts)
			{
				if ((pts->xpu_task_flags & DEVKIND__NVIDIA_GPU) != 0)
				{
					gpumask_t	__optimal_gpus = GetOptimalGpuForFile(fname);

					if (__optimal_gpus == INVALID_GPUMASK)
						optimal_gpus = 0;
					if (af_states_list == NIL)
					{
						optimal_gpus = __optimal_gpus;
						if (optimal_gpus == 0)
							__Debug("foreign-table='%s' arrow-file='%s' has no schedulable GPUs", RelationGetRelationName(frel), fname);
					}
					else
					{
						__optimal_gpus &= optimal_gpus;
						if (optimal_gpus != __optimal_gpus)
							__Debug("foreign-table='%s' arrow-file='%s' reduced GPUs-Set %08lx -> %08lx", RelationGetRelationName(frel), fname, optimal_gpus, __optimal_gpus);
						optimal_gpus = __optimal_gpus;
					}
				}
				else if ((pts->xpu_task_flags & DEVKIND__NVIDIA_DPU) != 0)
				{
					const DpuStorageEntry *ds_temp;

					if (af_states_list == NIL)
						ds_entry = GetOptimalDpuForFile(fname, &af_state->dpu_path);
					else if (ds_entry)
					{
						ds_temp = GetOptimalDpuForFile(fname, &af_state->dpu_path);
						if (!DpuStorageEntryIsEqual(ds_entry, ds_temp))
							ds_entry = NULL;
					}
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

	if (pts)
	{
		if ((pts->xpu_task_flags & DEVKIND__NVIDIA_GPU) != 0)
		{
			if (optimal_gpus != 0)
			{
				pts->xpu_task_flags |= DEVTASK__USED_GPUDIRECT;
				pts->optimal_gpus = optimal_gpus;
			}
			else
			{
				pts->optimal_gpus = GetSystemAvailableGpus();
			}
		}
		else if ((pts->xpu_task_flags & DEVKIND__NVIDIA_DPU) != 0)
		{
			pts->ds_entry = ds_entry;
		}
		else
		{
			elog(ERROR, "ExecPlan is neither GPU nor DPU");
		}
	}
	return arrow_state;
}

/*
 * pgstromArrowFdwExecInit
 */
bool
pgstromArrowFdwExecInit(pgstromTaskState *pts,
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
										 pts);
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
										 NULL);
}

/*
 * ExecArrowScanChunk
 */
static inline RecordBatchState *
__arrowFdwNextRecordBatch(ArrowFdwState *arrow_state,
						  int32_t num_scan_repeats,
						  int32_t *p_scan_repeat_id)
{
	RecordBatchState *rb_state;
	uint32_t	raw_index;
	uint32_t	rb_index;

	Assert(num_scan_repeats > 0);
retry:
	raw_index = pg_atomic_fetch_add_u32(arrow_state->rbatch_index, 1);
	if (raw_index >= arrow_state->rb_nitems * num_scan_repeats)
		return NULL;	/* no more chunks to load */
	rb_index = (raw_index % arrow_state->rb_nitems);
	if (p_scan_repeat_id)
		*p_scan_repeat_id = (raw_index / arrow_state->rb_nitems);
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
	XpuCommand *xcmd;
	uint32_t	kds_src_offset;
	uint32_t	kds_src_iovec;
	uint32_t	kds_src_pathname;
	int32_t		scan_repeat_id;

	rb_state = __arrowFdwNextRecordBatch(arrow_state,
										 pts->num_scan_repeats,
										 &scan_repeat_id);
	if (!rb_state)
	{
		pts->scan_done = true;
		return NULL;
	}
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
	xcmd->u.task.kds_src_pathname = kds_src_pathname;
	xcmd->u.task.kds_src_iovec    = kds_src_iovec;
	xcmd->u.task.kds_src_offset   = kds_src_offset;
	xcmd->u.task.scan_repeat_id   = scan_repeat_id;

	xcmd_iov->iov_base = xcmd;
	xcmd_iov->iov_len  = xcmd->length;
	*xcmd_iovcnt = 1;

	/* XXX - debug message */
    if (scan_repeat_id > 0 && scan_repeat_id != pts->last_repeat_id)
        elog(NOTICE, "arrow scan on '%s' moved into %dth loop for inner-buffer partitions (pid: %u)",
             RelationGetRelationName(pts->css.ss.ss_currentRelation),
			 scan_repeat_id+1,
			 MyProcPid);
    pts->last_repeat_id = scan_repeat_id;

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
		rb_state = __arrowFdwNextRecordBatch(arrow_state, 1, NULL);
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
	return offsetof(pgstromSharedState, inners);
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

	memset(ps_state, 0, offsetof(pgstromSharedState, inners));
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
	int			nfiles = 0;
	int			fcount = 0;
	int			i, j, k;
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
	nfiles = list_length(arrow_state->af_states_list);
	foreach (lc1, arrow_state->af_states_list)
	{
		ArrowFileState *af_state = lfirst(lc1);
		const char *filename = af_state->filename;
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
				continue;
			}

			for (k = bms_next_member(arrow_state->referenced, -1);
				 k >= 0;
				 k = bms_next_member(arrow_state->referenced, k))
			{
				j = k + FirstLowInvalidHeapAttributeNumber;
				if (j <= 0 || j > af_state->ncols)
					continue;
				i = af_state->attrs[j-1].field_index;
				if (i >= 0 && i < rb_state->nfields)
				{
					sz = __recordBatchFieldLength(&rb_state->fields[i]);
					read_sz += sz;
					chunk_sz[j] += sz;
				}
			}
		}

		/* displays only basename if regression test mode */
		if (pgstrom_regression_test_mode)
			filename = basename(pstrdup(filename));

		/* file size and read size */
		if (!pgstrom_explain_developer_mode &&
			nfiles >= 6 && fcount >= 2 && fcount < nfiles-2)
		{
			if (es->format == EXPLAIN_FORMAT_TEXT && fcount == 2)
				ExplainPropertyText("    :\t\t\t", "\t\t\t:", es);
		}
		else if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			resetStringInfo(&buf);
			appendStringInfo(&buf, "%s (read: %s, size: %s)",
							 filename,
							 format_bytesz(read_sz),
							 format_bytesz(total_sz));
			snprintf(label, sizeof(label), "file%d", fcount);
			ExplainPropertyText(label, buf.data, es);
		}
		else
		{
			snprintf(label, sizeof(label), "file%d", fcount);
			ExplainPropertyText(label, filename, es);

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

			j = k + FirstLowInvalidHeapAttributeNumber;
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
	List		   *filesList;
	List		   *virtualColumnsList;
	List		   *sourceFields;
	List		   *rb_state_list = NIL;
	ListCell	   *lc1, *lc2;
	int64			total_nrows = 0;
	int64			count_nrows = 0;
	int				nsamples_min = nrooms / 100;
	int				nitems = 0;

	filesList = arrowFdwExtractFilesList(ft->options,
										 &virtualColumnsList, NULL);
	sourceFields = arrowFdwExtractSourceFields(relation);
	forboth (lc1, filesList,
			 lc2, virtualColumnsList)
	{
		ArrowFileState *af_state;
		char	   *fname = strVal(lfirst(lc1));
		List	   *virtual_columns = lfirst(lc2);
		ListCell   *cell;

		af_state = BuildArrowFileState(relation, fname,
									   sourceFields,
									   virtual_columns, NULL);
		if (!af_state)
			continue;
		foreach (cell, af_state->rb_list)
		{
			RecordBatchState *rb_state = lfirst(cell);

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
	List		   *filesList;
	List		   *virtualColumnsList;
	ListCell	   *lc;
	size_t			totalpages = 0;

	filesList = arrowFdwExtractFilesList(ft->options,
										 &virtualColumnsList, NULL);
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
 * ensureUniqueFieldNames
 */
static const char **
ensureUniqueFieldNames(ArrowSchema *schema, List *virtual_columns)
{
	const char **column_names;
	int			k, count = 2;
	ListCell   *lc;

	column_names = palloc0(sizeof(char *) * (schema->_num_fields + 1 +
											 list_length(virtual_columns)));
	for (k=0; k < schema->_num_fields; k++)
	{
		const char *cname = schema->fields[k].name;
	retry:
		for (int j=0; j < k; j++)
		{
			if (strcasecmp(cname, column_names[j]) == 0)
			{
				cname = psprintf("__%s_%d", schema->fields[k].name, count++);
				goto retry;
			}
		}
		if (schema->fields[k].name != cname)
			elog(NOTICE, "Arrow::field[%d] '%s' meets a duplicated field name, so renamed to '%s'",
				 k, schema->fields[k].name, cname);
		column_names[k] = cname;
	}

	foreach (lc, virtual_columns)
	{
		virtualColumnDef *vcdef = lfirst(lc);
		const char *cname = vcdef->key;
	again:
		for (int j=0; j < k; j++)
		{
			if (strcasecmp(cname, column_names[j]) == 0)
			{
				cname = psprintf("__%s_%d", vcdef->key, count++);
				goto again;
			}
		}
		if (vcdef->key != cname)
			elog(NOTICE, "Arrow virtual column '%s' meets a duplicated field name, so renamed to '%s'",
				 vcdef->key, cname);
		column_names[k++] = cname;
	}
	Assert(column_names[k] == NULL);
	return column_names;
}

/*
 * ArrowImportForeignSchema
 */
static List *
ArrowImportForeignSchema(ImportForeignSchemaStmt *stmt, Oid serverOid)
{
	ArrowSchema	schema;
	List	   *virtual_columns_prime = NIL;
	List	   *filesList;
	List	   *virtualColumnsList;
	ListCell   *lc1, *lc2;
	const char **column_names;
	int			i;
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
	filesList = arrowFdwExtractFilesList(stmt->options,
										 &virtualColumnsList, NULL);
	if (filesList == NIL)
		ereport(ERROR,
				(errmsg("No valid apache arrow files are specified"),
				 errhint("Use 'file' or 'dir' option to specify apache arrow files on behalf of the foreign table")));

	/* read the schema */
	memset(&schema, 0, sizeof(ArrowSchema));
	forboth (lc1, filesList,
			 lc2, virtualColumnsList)
	{
		ArrowFileInfo af_info;
		const char *fname = strVal(lfirst(lc1));

		readArrowFile(fname, &af_info, false);
		if (lc1 == list_head(filesList))
		{
			copyArrowNode(&schema.node, &af_info.footer.schema.node);
			virtual_columns_prime = lfirst(lc2);
		}
		else
		{
			/* compatibility checks */
			ArrowSchema	   *stemp = &af_info.footer.schema;

			if (schema.endianness != stemp->endianness ||
				schema._num_fields != stemp->_num_fields)
				elog(ERROR, "file '%s' has incompatible schema definition", fname);
			for (int j=0; j < schema._num_fields; j++)
			{
				bool	found = false;

				for (int k=0; k < stemp->_num_fields; k++)
				{
					if (strcmp(schema.fields[j].name,
							   stemp->fields[k].name) == 0)
					{
						if (arrowFieldTypeIsEqual(&schema.fields[j],
												  &stemp->fields[k]))
						{
							found = true;
							break;
						}
						elog(ERROR, "field '%s' of '%s' has incompatible data type",
							 schema.fields[j].name, fname);
					}
				}

				if (!found)
					elog(ERROR, "field '%s' was not found in the file '%s'",
						 schema.fields[j].name, fname);
			}
		}
	}
	/* ensure the field-names are unique */
	column_names = ensureUniqueFieldNames(&schema, virtual_columns_prime);

	/* makes a command to define foreign table */
	initStringInfo(&cmd);
	appendStringInfo(&cmd, "CREATE FOREIGN TABLE %s (\n",
					 quote_identifier(stmt->remote_schema));
	for (i=0; i < schema._num_fields; i++)
	{
		ArrowField *field = &schema.fields[i];
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
		if (i > 0)
			appendStringInfo(&cmd, ",\n");
		if (type_mod < 0)
		{
			appendStringInfo(&cmd, "  %s %s.%s",
							 quote_identifier(column_names[i]),
							 quote_identifier(schema),
							 NameStr(__type->typname));
		}
		else
		{
			Assert(type_mod >= VARHDRSZ);
			appendStringInfo(&cmd, "  %s %s.%s(%d)",
							 quote_identifier(column_names[i]),
							 quote_identifier(schema),
							 NameStr(__type->typname),
							 type_mod - VARHDRSZ);
		}
		if (field->name != column_names[i])
			appendStringInfo(&cmd, " options (field '%s')", field->name);		
		ReleaseSysCache(htup);
	}

	foreach (lc1,  virtual_columns_prime)
	{
		virtualColumnDef *vcdef = lfirst(lc1);
		const char	   *label;

		Assert(column_names[i] != NULL);
		if (i > 0)
			appendStringInfo(&cmd, ",\n");
		if (vcdef->kind == '@')
			label = "pg_catalog.int8";
		else if (vcdef->kind == '$')
			label = "pg_catalog.text";
		else
			 elog(ERROR, "arrow_fdw: Bug? unknown virtual column type '%c'", vcdef->kind);

		appendStringInfo(&cmd, "  %s %s options(virtual '%s')",
						 quote_identifier(column_names[i]),
						 label,
						 vcdef->key);
		i++;
	}
	Assert(column_names[i] == NULL);
	appendStringInfo(&cmd,
					 "\n"
					 ") SERVER %s\n"
					 "  OPTIONS (", stmt->server_name);
	foreach (lc1, stmt->options)
	{
		DefElem	   *defel = lfirst(lc1);

		if (lc1 != list_head(stmt->options))
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
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_import_file);
static void
__insertPgAttributeTuple(Relation pg_attr_rel,
						 CatalogIndexState pg_attr_index,
						 Oid ftable_oid,
						 AttrNumber attnum,
						 const char *attname,
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
	values[Anum_pg_attribute_attname - 1] = CStringGetDatum(attname);
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

PUBLIC_FUNCTION(Datum)
pgstrom_arrow_fdw_import_file(PG_FUNCTION_ARGS)
{
	CreateForeignTableStmt stmt;
	ArrowSchema	schema;
	List	   *tableElts = NIL;
	char	   *ftable_name;
	char	   *file_name;
	char	   *namespace_name;
	const char **column_names;
	DefElem	   *defel;
	int			nfields;
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
	column_names = ensureUniqueFieldNames(&schema, NIL);

	/* setup CreateForeignTableStmt */
	memset(&stmt, 0, sizeof(CreateForeignTableStmt));
	NodeSetTag(&stmt, T_CreateForeignTableStmt);
	stmt.base.relation = makeRangeVar(namespace_name, ftable_name, -1);

	nfields = Min(schema._num_fields, 100);
	for (int j=0; j < nfields; j++)
	{
		ColumnDef  *cdef;
		Oid			type_oid;
		int32_t		type_mod;

		__arrowFieldTypeToPGType(&schema.fields[j],
								 &type_oid,
								 &type_mod,
								 NULL);
		cdef = makeColumnDef(column_names[j],
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

		for (int j=nfields; j < schema._num_fields; j++)
		{
			__insertPgAttributeTuple(a_rel,
									 a_index,
									 ftable_oid,
									 j+1,
									 column_names[j],
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
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_handler);
PUBLIC_FUNCTION(Datum)
pgstrom_arrow_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&pgstrom_arrow_fdw_routine);
}

/*
 * validator of Arrow_Fdw
 */
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_validator);
PUBLIC_FUNCTION(Datum)
pgstrom_arrow_fdw_validator(PG_FUNCTION_ARGS)
{
	List   *options = untransformRelOptions(PG_GETARG_DATUM(0));
	Oid		catalog = PG_GETARG_OID(1);

	if (catalog == ForeignTableRelationId)
	{
		List	   *filesList;
		List	   *virtualColumnsList;
		ListCell   *lc;

		filesList = arrowFdwExtractFilesList(options,
											 &virtualColumnsList, NULL);
		foreach (lc, filesList)
		{
			const char *fname = strVal(lfirst(lc));
			ArrowFileInfo af_info;

			readArrowFile(fname, &af_info, true);
		}
	}
	else if (catalog == AttributeRelationId)
	{
		bool		meet_field = false;
		bool		meet_virtual = false;
		bool		meet_virtual_metadata = false;
		bool		meet_virtual_metadata_split = false;
		ListCell   *lc;

		foreach (lc, options)
		{
			DefElem	   *defel = lfirst(lc);

			if (strcmp(defel->defname, "field") == 0)
			{
				if (strlen(strVal(defel->arg)) >= NAMEDATALEN-1)
					elog(ERROR, "arrow_fdw: column option '%s' is too long [%s]",
						 defel->defname, strVal(defel->arg));
				meet_field = true;
			}
			else if (strcmp(defel->defname, "virtual") == 0)
				meet_virtual = true;
			else if (strcmp(defel->defname, "virtual_metadata") == 0)
				meet_virtual_metadata = true;
			else if (strcmp(defel->defname, "virtual_metadata_split") == 0)
				meet_virtual_metadata_split = true;
			else
			{
				elog(ERROR, "arrow_fdw: column option '%s' is unknown",
					 defel->defname);
			}
		}
		if ((meet_field ? 1 : 0) +
			(meet_virtual ? 1 : 0) +
			(meet_virtual_metadata ? 1 : 0) +
			(meet_virtual_metadata_split ? 1 : 0) > 1)
			elog(ERROR, "arrow_fdw: column option 'field', 'virtual', 'virtual_metadata' and 'virtual_metadata_split' are mutually exclusive");
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
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_precheck_schema);
PUBLIC_FUNCTION(Datum)
pgstrom_arrow_fdw_precheck_schema(PG_FUNCTION_ARGS)
{
	EventTriggerData *trigdata;
	Relation	frel = NULL;
	ListCell   *lc1, *lc2;
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
			foreach (lc1, stmt->cmds)
			{
				AlterTableCmd  *cmd = lfirst(lc1);

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
		List	   *filesList;
		List	   *sourceFields;
		List	   *virtualColumnsList;

		filesList = arrowFdwExtractFilesList(ft->options,
											 &virtualColumnsList, NULL);
		sourceFields = arrowFdwExtractSourceFields(frel);
		forboth (lc1, filesList,
				 lc2, virtualColumnsList)
		{
			const char *fname = strVal(lfirst(lc1));
			List	   *virtual_columns = lfirst(lc2);

			(void)BuildArrowFileState(frel, fname,
									  sourceFields,
									  virtual_columns, NULL);
		}
	}
	if (frel)
		relation_close(frel, NoLock);
	PG_RETURN_NULL();
}

/*
 * pgstrom_check_pattern
 */
PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_check_pattern);
PUBLIC_FUNCTION(Datum)
pgstrom_arrow_fdw_check_pattern(PG_FUNCTION_ARGS)
{
	text	   *t = PG_GETARG_TEXT_P(0);
	text	   *p = PG_GETARG_TEXT_P(1);
	List	   *attrKinds = NIL;
	List	   *attrKeys = NIL;
	List	   *attrValues = NIL;
	ListCell   *lc1, *lc2, *lc3;
	bool		retval;
	StringInfoData buf;

	retval = pathNameMatchByPattern(text_to_cstring(t),
									text_to_cstring(p),
									&attrKinds,
									&attrKeys,
									&attrValues);
	initStringInfo(&buf);
	if (retval)
	{
		bool	need_comma = false;

		appendStringInfo(&buf, "true");
		forthree (lc1, attrKinds,
				  lc2, attrKeys,
				  lc3, attrValues)
		{
			if (!need_comma)
				appendStringInfo(&buf, " {");
			else
				appendStringInfo(&buf, ", ");
			appendStringInfo(&buf, "%c[%s]=[%s]",
							 (int)lfirst_int(lc1),
							 (char *)lfirst(lc2),
							 (char *)lfirst(lc3));
			need_comma = true;
		}
		if (need_comma)
			appendStringInfo(&buf, "}");
	}
	else
	{
		appendStringInfo(&buf, "false");
	}
	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

/*
 * pgstrom_arrow_fdw_metadata_info
 */
typedef struct arrowFdwMetadataInfo
{
	dlist_node	chain;
	Oid			frelid;
	text	   *filename;
	text	   *field;
	text	   *key;
	text	   *value;
} arrowFdwMetadataInfo;

static void
__build_arrow_fdw_metadata_info(dlist_head *md_info_dlist,
								Oid frelid, const char *filename,
								arrowMetadataFieldCache *fcache,
								const char *prefix,
								arrowMetadataKeyValueCache *custom_metadata)
{
	arrowFdwMetadataInfo *md_info;
	arrowMetadataKeyValueCache *mc_kv;

	for (mc_kv = custom_metadata; mc_kv != NULL; mc_kv = mc_kv->next)
	{
		md_info = palloc0(sizeof(arrowFdwMetadataInfo));
		md_info->frelid = frelid;
		md_info->filename = cstring_to_text(filename);
		if (fcache)
		{
			char   *s = psprintf("XXXX%s%s", prefix, fcache->attname);
			md_info->field = (text *)s;
			SET_VARSIZE(md_info->field, strlen(s));
		}
		md_info->key = cstring_to_text(mc_kv->key);
		md_info->value = cstring_to_text(mc_kv->value);
		dlist_push_tail(md_info_dlist, &md_info->chain);
	}

	if (fcache)
	{
		dlist_iter	iter;
		char	   *__prefix = NULL;

		dlist_foreach (iter, &fcache->children)
		{
			arrowMetadataFieldCache *__fcache =
				dlist_container(arrowMetadataFieldCache, chain, iter.cur);
			if (!__prefix)
			{
				__prefix = alloca(strlen(prefix) +
								  strlen(fcache->attname) + 10);
				sprintf(__prefix, "%s%s.", prefix, fcache->attname);
			}
			__build_arrow_fdw_metadata_info(md_info_dlist,
											frelid, filename,
											__fcache,
											__prefix,
											fcache->custom_metadata);
		}
	}
}

static dlist_head *
__setup_arrow_fdw_metadata_info(Oid frelid)
{
	ForeignTable   *ft = GetForeignTable(frelid);
	List		   *filesList = arrowFdwExtractFilesList(ft->options, NULL, NULL);
	ListCell	   *lc;
	dlist_head	   *md_info_dlist = palloc(sizeof(dlist_head));

	dlist_init(md_info_dlist);
	foreach (lc, filesList)
	{
		const char *filename = strVal(lfirst(lc));
		struct stat	stat_buf;
		dlist_iter	iter;
		arrowMetadataCacheBlock *mc_block;

		if (stat(filename, &stat_buf) != 0)
		{
			if (errno == ENOENT)
				continue;	/* file might be removed concurrently */
			elog(ERROR, "failed on stat('%s'): %m", filename);
		}
		LWLockAcquire(&arrow_metadata_cache->mutex, LW_SHARED);
		mc_block = lookupArrowMetadataCache(&stat_buf, false);
		/* if not built yet, construct a metadata cache entry */
		if (!mc_block)
		{
			ArrowFileInfo	af_info;
			ArrowFileState	af_state;

			LWLockRelease(&arrow_metadata_cache->mutex);

			memset(&af_state, 0, sizeof(af_state));
			af_state.filename = filename;
			memcpy(&af_state.stat_buf, &stat_buf, sizeof(struct stat));

			if (!__setupArrowFileStateByFile(&af_state,
											 filename,
											 &af_info,
											 NULL))
				elog(ERROR, "unable to read the arrow file '%s'", filename);
			LWLockAcquire(&arrow_metadata_cache->mutex, LW_EXCLUSIVE);
			mc_block = lookupArrowMetadataCache(&af_state.stat_buf, true);
			if (!mc_block)
			{
				mc_block = __buildArrowMetadataCacheNoLock(&af_state, &af_info);
				if (!mc_block)
					elog(ERROR, "unable to build arrow metadata cache, consider to expand 'arrow_fdw.metadata_cache_size'");
			}
		}
		/* copy the metadata info */
		__build_arrow_fdw_metadata_info(md_info_dlist,
										frelid,
										filename,
										NULL, "",
										mc_block->custom_metadata);
		dlist_foreach (iter, &mc_block->mcache_head.fields)
		{
			arrowMetadataFieldCache *fcache
				= dlist_container(arrowMetadataFieldCache,
								  chain, iter.cur);
			__build_arrow_fdw_metadata_info(md_info_dlist,
											frelid,
											filename,
											fcache, "",
											fcache->custom_metadata);
		}
		LWLockRelease(&arrow_metadata_cache->mutex);
	}
	return md_info_dlist;
}

PG_FUNCTION_INFO_V1(pgstrom_arrow_fdw_metadata_info);
PUBLIC_FUNCTION(Datum)
pgstrom_arrow_fdw_metadata_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	arrowFdwMetadataInfo *md_info;
	Datum		values[5];
	bool		isnull[5];
	HeapTuple	tuple;
	dlist_head *md_info_dlist;

	if (SRF_IS_FIRSTCALL())
	{
		Oid				frelid = PG_GETARG_OID(0);
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(5);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "relid",
						   OIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "filename",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "field",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "key",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = __setup_arrow_fdw_metadata_info(frelid);

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();
	md_info_dlist = fncxt->user_fctx;
	if (dlist_is_empty(md_info_dlist))
		SRF_RETURN_DONE(fncxt);
	md_info = dlist_container(arrowFdwMetadataInfo, chain,
							  dlist_pop_head_node(md_info_dlist));

	memset(isnull, 0, sizeof(isnull));
	values[0] = ObjectIdGetDatum(md_info->frelid);
	values[1] = PointerGetDatum(md_info->filename);
	if (md_info->field)
		values[2] = PointerGetDatum(md_info->field);
	else
		isnull[2] = true;
	values[3] = PointerGetDatum(md_info->key);
	values[4] = PointerGetDatum(md_info->value);
	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);
	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
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
	char	   *buffer;
	bool		found;
	size_t		n, sz;

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
	for (int i=0; i < ARROW_METADATA_HASH_NSLOTS; i++)
		dlist_init(&arrow_metadata_cache->hash_slots[i]);
	/* arrowMetadataCacheBlock allocation  */
	sz = TYPEALIGN(ARROW_METADATA_BLOCKSZ,
				   (size_t)arrow_metadata_cache_size_kb << 10);
	n = sz / ARROW_METADATA_BLOCKSZ;
	buffer = ShmemInitStruct("arrowMetadataCache(body)", sz, &found);
	Assert(!found);
	for (int i=0; i < n; i++)
	{
		arrowMetadataCacheBlock *mc_block = (arrowMetadataCacheBlock *)buffer;

		memset(mc_block, 0, offsetof(arrowMetadataCacheBlock, mcache_head));
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








