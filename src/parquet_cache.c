/*
 * parquet_cache.c
 *
 * Routines of Parquet flatten cache in C-portion
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

typedef struct {
	dlist_node		chain;		/* hash, siblings or free list */
	dlist_head		siblings;	/* continuous blocks */
	/* hash and keys */
	dev_t			st_dev;		/* file identifier from stat(2) */
	ino_t			st_ino;		/* file identifier from stat(2) */
	uint64_t		st_utime;	/* file identifier from stat(2) */
	int32_t			rg_index;	/* row-group index */
	int32_t			field_id;	/* field-index */
	uint32_t		hash;		/* hash value; also note that hash-value must be
								 * valid until it is in LRU list */
	/* state control */
	int32_t			refcnt;		/* -1: not ready (WIP), > 0: someone in use */
	uint32_t		nr_pages;	/* usage of this block, by # of pages */
	int32_t			lru_bonus;	/* increased by 1, per cache hit */
	/* arrow buffer definitions */
	uint64_t		nullmap_length;
	uint64_t		values_length;
	uint64_t		extra_length;
	/* LRU */
	dlist_node		lru_chain;		/* LRU list */
	uint64_t		lru_timestamp;	/* last referenced timestamp */
} parquetCacheEntry;

typedef struct {
	/* Hash slots */
	pthread_mutex_t	   *hash_locks;
	dlist_head		   *hash_slots;
	uint64_t			hash_nslots;
	uint64_t			num_blocks;
	/* Free list */
	pthread_mutex_t		free_lock;
	dlist_head			free_list;
	/* LRU list */
	pthread_mutex_t		lru_lock;
	dlist_head			lru_list;
	/* Statistics */
	pg_atomic_uint64	num_active_blocks;
	pg_atomic_uint64	num_cached_chunks;
	pg_atomic_uint64	total_actual_usage;
} parquetCacheHead;

/*
 * Static variables
 */
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static char	   *pgstrom_parquet_cache_path;				/* GUC */
static int		pgstrom_parquet_cache_size_mb;			/* GUC */
static int		pgstrom_parquet_cache_unitsz_mb;		/* GUC */
static double	pgstrom_parquet_cache_eviction_min_seconds; /* GUC */
static double	pgstrom_parquet_cache_eviction_max_seconds; /* GUC */
static double	pgstrom_parquet_cache_eviction_bonus;	/* GUC */
static double	pgstrom_parquet_cache_target_usage;		/* GUC */
static int		pgstrom_parquet_cache_max_async_write;	/* GUC */
static uint64_t	pgstrom_parquet_cache_nblocks;		/* immutable */
static uint64_t	pgstrom_parquet_cache_nslots;		/* immutable */
static int		pgstrom_parquet_cache_fdesc = -1;
static parquetCacheHead  *parquet_cache_head = NULL;
static parquetCacheEntry *parquet_cache_array = NULL;

/* parquetCacheEntry to the base file offset */
static inline off_t
__parquetCacheFileOffset(const parquetCacheEntry *entry)
{
	size_t	block_sz = (size_t)pgstrom_parquet_cache_unitsz_mb << 20;
	return (size_t)(entry - parquet_cache_array) * block_sz;
}

/* __timespec_to_utime */
static inline uint64_t
__timespec_to_uts(const struct timespec *t_spec)
{
	return ((uint64_t)t_spec->tv_sec  * 1000000UL +
			(uint64_t)t_spec->tv_nsec / 1000UL);
}

/* __timespec_to_utime */
static inline uint64_t
__timeval_to_uts(const struct timeval *t_val)
{
	return ((uint64_t)t_val->tv_sec * 1000000UL +
			(uint64_t)t_val->tv_usec);
}

/*
 * __parquetCacheHash
 */
static uint32_t
__parquetCacheHash(dev_t st_dev,
				   dev_t st_ino,
				   uint64_t st_utime,
				   int32_t rg_index,
				   int32_t field_id)
{
	struct {
		dev_t		st_dev;
		ino_t		st_ino;
		uint64_t	st_utime;
		int32_t		rg_index;
		int32_t		field_id;
	} key;

	memset(&key, 0, sizeof(key));
	key.st_dev		= st_dev;
	key.st_ino		= st_ino;
	key.st_utime	= st_utime;
	key.rg_index	= rg_index;
	key.field_id	= field_id;

	return hash_any((unsigned char *)&key, sizeof(key));
}

/*
 * parquetCacheLookup
 */
void *
parquetCacheLookup(const struct stat *pq_fstat,
				   int32_t rg_index,
				   int32_t field_id)
{
	parquetCacheEntry *result = NULL;
	uint32_t	hash, slot_id;
	uint64_t	pq_utime;
	dlist_iter	iter;

	if (!parquet_cache_head)
		return NULL;

	pq_utime = __timespec_to_uts(&pq_fstat->st_mtim);
	hash = __parquetCacheHash(pq_fstat->st_dev,
							  pq_fstat->st_ino,
							  pq_utime,
							  rg_index,
							  field_id);
	slot_id = hash % parquet_cache_head->hash_nslots;
	pthreadMutexLock(&parquet_cache_head->hash_locks[slot_id]);
	dlist_foreach(iter, &parquet_cache_head->hash_slots[slot_id])
	{
		parquetCacheEntry *entry = dlist_container(parquetCacheEntry,
													 chain, iter.cur);
		if (entry->hash     == hash &&
			entry->st_dev   == pq_fstat->st_dev &&
			entry->st_ino   == pq_fstat->st_ino &&
			entry->st_utime == pq_utime &&
			entry->rg_index == rg_index &&
			entry->field_id == field_id)
		{
			/*
			 * NOTE: we don't need to update LRU list at the lookup phase,
			 * because any cache entry with non-zero refcnt shall not be
			 * evicted, and entry shall be moved to the tail of LRU list
			 * on the release time.
			 */
			if (entry->refcnt >= 0)
			{
				// Note that negative refcnt means someone is still writing out
				// the Arrow buffers, but works in progress, and not ready.
				entry->refcnt++;
				entry->lru_bonus++;
				result = entry;
			}
			break;
		}
	}
	pthreadMutexUnlock(&parquet_cache_head->hash_locks[slot_id]);
	return result;
}

/*
 * parquetCacheRelease
 */
void
parquetCacheRelease(void *__entry)
{
	parquetCacheEntry *entry = (parquetCacheEntry *)__entry;
	uint32_t	slot_id = entry->hash % parquet_cache_head->hash_nslots;
	struct timespec tv;

	pthreadMutexLock(&parquet_cache_head->hash_locks[slot_id]);
	assert(entry->refcnt > 0);
	pthreadMutexLock(&parquet_cache_head->lru_lock);
	if (clock_gettime(CLOCK_REALTIME, &tv) == 0)
		entry->lru_timestamp = __timespec_to_uts(&tv);
	dlist_move_tail(&parquet_cache_head->lru_list, &entry->lru_chain);
	pthreadMutexUnlock(&parquet_cache_head->lru_lock);
	entry->refcnt--;
	pthreadMutexUnlock(&parquet_cache_head->hash_locks[slot_id]);
}

/*
 * parquetCacheLength
 */
size_t
parquetCacheLength(const void *__entry)
{
	const parquetCacheEntry *entry = (const parquetCacheEntry *)__entry;

	return (PAGE_ALIGN(entry->nullmap_length) +
			PAGE_ALIGN(entry->values_length) +
			PAGE_ALIGN(entry->extra_length));
}

/*
 * parquetCacheReadChunks
 */
ssize_t
parquetCacheReadChunks(void *__entry,
					   kern_colmeta *cmeta,
					   size_t kds_offset,
					   CUdeviceptr m_segment,
					   off_t m_offset,
					   uint32_t *p_npages_direct_read,
					   uint32_t *p_npages_vfs_read)
{
	parquetCacheEntry *entry = (parquetCacheEntry *)__entry;
	size_t		block_sz = (size_t)pgstrom_parquet_cache_unitsz_mb << 20;
	dlist_iter	iter;
	int			nr_chunks = 1;
	int			chunk_id = 0;
	size_t		total_usage = 0;
	size_t		__usage;
	ssize_t		retval;
	strom_io_vector *iovec;
	strom_io_chunk *ioc;

	assert(m_segment  == MAXALIGN(m_segment) &&
		   m_offset   == MAXALIGN(m_offset) &&
		   kds_offset == MAXALIGN(kds_offset));
	/* setup cmeta */
	if (entry->nullmap_length == 0)
		cmeta->nullmap_offset = cmeta->nullmap_length = 0;
	else
	{
		cmeta->nullmap_offset = kds_offset + total_usage;
		cmeta->nullmap_length = entry->nullmap_length;
		total_usage += PAGE_ALIGN(entry->nullmap_length);
	}
	if (entry->values_length == 0)
		cmeta->values_offset = cmeta->values_length = 0;
	else
	{
		cmeta->values_offset = kds_offset + total_usage;
		cmeta->values_length = entry->values_length;
		total_usage += PAGE_ALIGN(entry->values_length);
	}
	if (entry->extra_length == 0)
		cmeta->extra_offset = cmeta->extra_length = 0;
	else
	{
		cmeta->extra_offset = kds_offset + total_usage;
		cmeta->extra_length = entry->extra_length;
		total_usage += PAGE_ALIGN(entry->extra_length);
	}
	retval = total_usage;
	/* setup iovec */
	nr_chunks = 1 + __dlist_length(&entry->siblings);
	iovec = (strom_io_vector *)alloca(offsetof(strom_io_vector,
											   ioc[nr_chunks]));
	/* setup the primary block */
	__usage = Min(block_sz, total_usage);
	ioc = &iovec->ioc[chunk_id++];
	ioc->m_offset  = kds_offset;
	ioc->fchunk_id = __parquetCacheFileOffset(entry) / PAGE_SIZE;
	ioc->nr_pages  = __usage / PAGE_SIZE;
	kds_offset    += __usage;
	total_usage   -= __usage;
	/* setup other sibling blocks, if any */
	dlist_foreach (iter, &entry->siblings)
	{
		parquetCacheEntry *buddy = dlist_container(parquetCacheEntry,
												   chain, iter.cur);
		assert(total_usage > 0);
		__usage = Min(block_sz, total_usage);
		ioc = &iovec->ioc[chunk_id++];
		ioc->m_offset  = kds_offset;
		ioc->fchunk_id = __parquetCacheFileOffset(buddy) / PAGE_SIZE;
		ioc->nr_pages  = __usage / PAGE_SIZE;
		kds_offset    += __usage;
		total_usage   -= __usage;
	}
	iovec->nr_chunks = chunk_id;
	/* read the chunk array from the file */
	if (!gpuDirectFileReadIOV(pgstrom_parquet_cache_path,
							  m_segment,
							  m_offset,
							  -1L,
							  iovec,
							  true,
							  p_npages_direct_read,
							  p_npages_vfs_read))
	{
		return -1;
	}
	return retval;
}

/*
 * __parquetCacheReclaimBlocks
 */
static void
__parquetCacheReclaimBlocks(int target)
{
	int			reclaimed = 0;
	uint64_t	lru_threshold;
	uint64_t	lru_latest = 0;
	dlist_head	pending;
	struct timespec tv;
	dlist_mutable_iter iter;

	if (clock_gettime(CLOCK_REALTIME, &tv) != 0)
	{
		__Notice("failed on clock_gettime(CLOCK_REALTIME): %m");
		return;		/* should not happen */
	}
	lru_threshold = __timespec_to_uts(&tv)
		- (pgstrom_parquet_cache_eviction_min_seconds
		   + (pgstrom_parquet_cache_eviction_max_seconds -
			  pgstrom_parquet_cache_eviction_min_seconds) * drand48()) * 1000000.0;
	dlist_init(&pending);
	pthreadMutexLock(&parquet_cache_head->lru_lock);
	/*
	 * When a worker thread is idle, it performs maintenance work to release
	 * active cache blocks so that the cache usage converges to the value
	 * specified by pg_strom.parquet_cache_target_usage.
	 *
	 * At that time, the actual number of blocks to be released has to be
	 * calculated while holding lru_lock. Otherwise, if many worker threads
	 * become idle at once and start reclamation simultaneously, it is not
	 * possible to determine the correct number of blocks to reclaim in advance
	 * outside the lock.
	 */
	if (target < 0)
	{
		uint64_t	nactives = pg_atomic_read_u64(&parquet_cache_head->num_active_blocks);
		uint64_t	nblocks = ((double)pgstrom_parquet_cache_nblocks * 
							   (double)pgstrom_parquet_cache_target_usage / 100.0);
		if (nactives > nblocks)
			target = Min(nactives - nblocks, 50);
	}
	if (target <= 0)
		goto skip;
	dlist_foreach_modify (iter, &parquet_cache_head->lru_list)
	{
		parquetCacheEntry *entry = dlist_container(parquetCacheEntry,
												   lru_chain, iter.cur);
		uint32_t	slot_id = entry->hash % parquet_cache_head->hash_nslots;

		if (pthreadMutexTryLock(&parquet_cache_head->hash_locks[slot_id]))
		{
			if (entry->refcnt == 0)
			{
				double	bonus = 0.0;

				if (entry->lru_bonus > 0)
					bonus = (double)(entry->lru_bonus--) * pgstrom_parquet_cache_eviction_bonus;

				if (entry->lru_timestamp >= lru_threshold)
					target = 0;		/* no entry should be found, any more */
				else if (entry->lru_timestamp + (uint64_t)(bonus * 1000000.0) < lru_threshold)
				{
					dlist_delete(&entry->lru_chain);
					dlist_delete(&entry->chain);
					dlist_push_tail(&pending, &entry->chain);
					reclaimed += 1 + __dlist_length(&entry->siblings);
				}
			}
			pthreadMutexUnlock(&parquet_cache_head->hash_locks[slot_id]);
			if (reclaimed >= target)
				break;
		}
	}
skip:
	pthreadMutexUnlock(&parquet_cache_head->lru_lock);
	/* back to the free list, if any */
	if (!dlist_is_empty(&pending))
	{
		pthreadMutexLock(&parquet_cache_head->free_lock);
		while (!dlist_is_empty(&pending))
		{
			parquetCacheEntry *entry
				= dlist_container(parquetCacheEntry, chain,
								  dlist_pop_head_node(&pending));
			size_t	usage = parquetCacheLength(entry);
			int		nr_blocks = 1;

			while (!dlist_is_empty(&entry->siblings))
			{
				parquetCacheEntry *__entry
					= dlist_container(parquetCacheEntry, chain,
									  dlist_pop_head_node(&entry->siblings));
				dlist_delete(&__entry->chain);
				dlist_push_head(&parquet_cache_head->free_list, &__entry->chain);
				nr_blocks++;
			}
			dlist_delete(&entry->chain);
			dlist_push_head(&parquet_cache_head->free_list, &entry->chain);
			/* update statistics */
			pg_atomic_fetch_sub_u64(&parquet_cache_head->num_active_blocks, nr_blocks);
			pg_atomic_fetch_sub_u64(&parquet_cache_head->num_cached_chunks, 1);
			pg_atomic_fetch_sub_u64(&parquet_cache_head->total_actual_usage, usage);
		}
		pthreadMutexUnlock(&parquet_cache_head->free_lock);
	}
	if (reclaimed > 0)
		__Info("parquet-cache evicted %d blocks, latest access at %s",
			   reclaimed,
			   format_utime_utc(lru_latest));
}

/* ----------------------------------------------------------------
 *
 * parquetCacheWriteAsync
 *
 * ----------------------------------------------------------------
 */
static pthread_mutex_t	parquet_cache_async_write_mutex;
static pthread_cond_t	parquet_cache_async_write_cond;
static dlist_head		parquet_cache_async_write_list;
static pg_atomic_uint32	parquet_cache_async_write_num_workers;
static pg_atomic_uint32	parquet_cache_async_write_num_available_workers;
static pg_atomic_uint32	parquet_cache_async_write_num_active_tasks;

static bool
__parquetCacheAsyncWriteBlock(off_t *p_f_offset,
							  char *block_buf,
							  const off_t *fchunk_base,
							  const uint8_t *arrow_ptr,
							  size_t arrow_len,
							  bool is_last_buffer)
{
	size_t		block_sz = (size_t)pgstrom_parquet_cache_unitsz_mb << 20;
	size_t		f_offset = *p_f_offset;
	size_t		arrow_off = 0;

	while (arrow_off < arrow_len)
	{
		uint32_t	block_idx = (f_offset / block_sz);
		uint32_t	block_off = (f_offset % block_sz);
		size_t		sz = Min(block_sz - block_off, arrow_len - arrow_off);

		memcpy(block_buf + block_off, arrow_ptr + arrow_off, sz);
		arrow_off += sz;
		f_offset  += sz;
		block_off += sz;
		if (block_off >= block_sz || (is_last_buffer && arrow_off >= arrow_len))
		{
			off_t	off = 0;
			ssize_t	nbytes;

			block_off = PAGE_ALIGN(block_off);
			assert(block_off <= block_sz && arrow_off <= arrow_len);
			while (off < block_off)
			{
				nbytes = pwrite(pgstrom_parquet_cache_fdesc,
								block_buf + off,
								block_off - off,
								fchunk_base[block_idx] + off);
				if (nbytes > 0)
					off += nbytes;
				else if (nbytes == 0 || errno != EINTR)
				{
					__Notice("parquet-cache: failed on pwrite at %08lx, sz=%ld: %m",
							 fchunk_base[block_idx] + off,
							 block_off - off);
					return false;
				}
			}
		}
	}
	*p_f_offset = PAGE_ALIGN(f_offset);
	return true;
}

static void
__parquetCacheAsyncWriteOne(parquetCacheWriteAsyncData *data, char *block_buf)
{
	parquetCacheEntry *entry = NULL;
	uint64_t	st_utime = __timespec_to_uts(&data->pq_fstat.st_mtim);
	size_t		block_sz = (size_t)pgstrom_parquet_cache_unitsz_mb << 20;
	size_t		required = (PAGE_ALIGN(data->nullmap_len) +
							PAGE_ALIGN(data->values_len) +
							PAGE_ALIGN(data->extra_len));
	int			nr_blocks = (required + block_sz - 1) / block_sz;
	int			block_cnt = 0;
	off_t	   *fchunk_base = (off_t *)alloca(sizeof(off_t) * nr_blocks);
	off_t		f_offset = 0;
	bool		retry_done = false;
	uint32_t	slot_id;
	dlist_iter	iter;
	struct timeval tv;

	assert((uintptr_t)block_buf == PAGE_ALIGN((uintptr_t)block_buf));
	if (nr_blocks == 0)
		return;		/* should not happen */
	/* 1. allocation of cache entry */
again:
	pthreadMutexLock(&parquet_cache_head->free_lock);
	while (block_cnt < nr_blocks)
	{
		parquetCacheEntry *curr;
		dlist_node	   *dnode;

		if (dlist_is_empty(&parquet_cache_head->free_list))
		{
			pthreadMutexUnlock(&parquet_cache_head->free_lock);
			if (!retry_done)
			{
				__parquetCacheReclaimBlocks(nr_blocks);
				retry_done = true;
				goto again;
			}
			goto bailout;	/* give up */
		}
		dnode = dlist_pop_head_node(&parquet_cache_head->free_list);
		curr = dlist_container(parquetCacheEntry, chain, dnode);
		memset(curr, 0, sizeof(parquetCacheEntry));
		if (entry)
			dlist_push_tail(&entry->siblings, &curr->chain);
		else
		{
			dlist_init(&curr->siblings);
			entry = curr;
		}
		fchunk_base[block_cnt++] = __parquetCacheFileOffset(curr);
		//stats update
		pg_atomic_fetch_add_u64(&parquet_cache_head->num_active_blocks, 1);
	}
	pthreadMutexUnlock(&parquet_cache_head->free_lock);

	/* 2. setup new cache entry */
	entry->st_dev   = data->pq_fstat.st_dev;
	entry->st_ino   = data->pq_fstat.st_ino;
	entry->st_utime = st_utime;
	entry->rg_index = data->rg_index;
	entry->field_id = data->field_id;
	entry->hash     = __parquetCacheHash(entry->st_dev,
										 entry->st_ino,
										 entry->st_utime,
										 entry->rg_index,
										 entry->field_id);
	entry->refcnt = -1;		/* flag for WIP */

	/* 3. check cache duplication */
	slot_id = entry->hash % pgstrom_parquet_cache_nslots;
	pthreadMutexLock(&parquet_cache_head->hash_locks[slot_id]);
	dlist_foreach(iter, &parquet_cache_head->hash_slots[slot_id])
	{
		parquetCacheEntry *curr = dlist_container(parquetCacheEntry,
												  chain, iter.cur);
		if (entry->hash     == curr->hash &&
			entry->st_dev   == curr->st_dev &&
			entry->st_ino   == curr->st_ino &&
			entry->st_utime == curr->st_utime &&
			entry->rg_index == curr->rg_index &&
			entry->field_id == curr->field_id)
		{
			/* found duplicated cache entry */
			pthreadMutexUnlock(&parquet_cache_head->hash_locks[slot_id]);
			goto bailout;
		}
	}
	/* ok, I am responsible to write out this cache entry */
	gettimeofday(&tv, NULL);
	pthreadMutexLock(&parquet_cache_head->lru_lock);
	dlist_push_tail(&parquet_cache_head->lru_list, &entry->lru_chain);
	entry->lru_timestamp = __timeval_to_uts(&tv);
	pthreadMutexUnlock(&parquet_cache_head->lru_lock);
	dlist_push_tail(&parquet_cache_head->hash_slots[slot_id], &entry->chain);
	pthreadMutexUnlock(&parquet_cache_head->hash_locks[slot_id]);

	/* 4. direct write to cache file */
	if (!data->nullmap_ptr)
		entry->nullmap_length = 0;
	else if (__parquetCacheAsyncWriteBlock(&f_offset,
										   block_buf,
										   fchunk_base,
										   data->nullmap_ptr,
										   data->nullmap_len,
										   !data->values_ptr &&
										   !data->extra_ptr))
		entry->nullmap_length = data->nullmap_len;
	else
		goto bailout_unlink;

	if (!data->values_ptr)
		entry->values_length = 0;
	else if (__parquetCacheAsyncWriteBlock(&f_offset,
										   block_buf,
										   fchunk_base,
										   data->values_ptr,
										   data->values_len,
										   !data->extra_ptr))
		entry->values_length = data->values_len;
	else
		goto bailout_unlink;

	if (!data->extra_ptr)
		entry->extra_length = 0;
	else if (__parquetCacheAsyncWriteBlock(&f_offset,
										   block_buf,
										   fchunk_base,
										   data->extra_ptr,
										   data->extra_len,
										   true))
		entry->extra_length = data->extra_len;
	else
		goto bailout_unlink;
	assert(f_offset == required);
	/* 5. activate the cache entry */
	pthreadMutexLock(&parquet_cache_head->hash_locks[slot_id]);
	assert(entry->refcnt == -1);
	entry->refcnt = 0;
	pthreadMutexUnlock(&parquet_cache_head->hash_locks[slot_id]);
	/* 6. update statistics and debug message */
	pg_atomic_fetch_add_u64(&parquet_cache_head->num_cached_chunks, 1);
	pg_atomic_fetch_add_u64(&parquet_cache_head->total_actual_usage, required);
	__Debug("parquet cache written: st_dev=%ld, st_ino=%ld, st_mtime=%s, row-group=%u, field-id=%u, nr-blocks=%u, length=%s",
			data->pq_fstat.st_dev,
			data->pq_fstat.st_ino,
			format_utime_utc(st_utime),
			data->rg_index,
			data->field_id,
			nr_blocks,
			format_bytesz(required));
	return;		/* ok */

bailout_unlink:
	/* unlink the entry from hash list */
	pthreadMutexLock(&parquet_cache_head->hash_locks[slot_id]);
	dlist_delete(&entry->chain);
	pthreadMutexUnlock(&parquet_cache_head->hash_locks[slot_id]);
bailout:
	/* back the entry to the free list */
	if (entry)
	{
		int		nblocks = 1;

		pthreadMutexLock(&parquet_cache_head->free_lock);
		while (!dlist_is_empty(&entry->siblings))
		{
			dlist_node	   *dnode = dlist_pop_head_node(&entry->siblings);
			dlist_push_head(&parquet_cache_head->free_list, dnode);
			nblocks++;
		}
		memset(entry, 0, sizeof(parquetCacheEntry));
		dlist_push_head(&parquet_cache_head->free_list, &entry->chain);
		pthreadMutexUnlock(&parquet_cache_head->free_lock);
		pg_atomic_fetch_sub_u64(&parquet_cache_head->num_active_blocks, nblocks);
	}
	__Debug("parquet cache skipped: st_dev=%ld, st_ino=%ld, st_mtime=%s, row-group=%u, field-id=%u, nr-blocks=%u, length=%s",
		   data->pq_fstat.st_dev,
		   data->pq_fstat.st_ino,
		   format_utime_utc(st_utime),
		   data->rg_index,
		   data->field_id,
		   nr_blocks,
		   format_bytesz(required));
}

static void *
__parquetCacheAsyncWriteWorker(void *__priv)
{
	size_t	block_sz = (size_t)pgstrom_parquet_cache_unitsz_mb << 20;
	char   *block_buf, *malloc_ptr;
	int		noop_counter = 0;

	__Info("parquet-cache: async write worker start (tid: %u)", gettid());
	/* alloc i/o buffer; aligned for O_DIRECT */
	malloc_ptr = (char *)malloc(block_sz + PAGE_SIZE);
	if (!malloc_ptr)
	{
		__Notice("parquet-cache: out of memory");
		return NULL;	/* out of memory */
	}
	block_buf = (char *)PAGE_ALIGN(malloc_ptr);

	pthreadMutexLock(&parquet_cache_async_write_mutex);
	pg_atomic_fetch_add_u32(&parquet_cache_async_write_num_workers, 1);
	for (;;)
	{
		if (!dlist_is_empty(&parquet_cache_async_write_list))
		{
			dlist_node *dnode = dlist_pop_head_node(&parquet_cache_async_write_list);
			parquetCacheWriteAsyncData *data
				= dlist_container(parquetCacheWriteAsyncData, chain, dnode);
			pthreadMutexUnlock(&parquet_cache_async_write_mutex);
			memset(&data->chain, 0, sizeof(dlist_node));
			__parquetCacheAsyncWriteOne(data, block_buf);
			pg_atomic_fetch_sub_u32(&parquet_cache_async_write_num_active_tasks, 1);
			/* cleanup */
			releaseParquetCacheWriteAsyncData(data);
			/* reset */
			noop_counter = 0;
			pthreadMutexLock(&parquet_cache_async_write_mutex);
		}
		else
		{
			pg_atomic_fetch_add_u32(&parquet_cache_async_write_num_available_workers, 1);
			if (pthreadCondWaitTimeout(&parquet_cache_async_write_cond,
									   &parquet_cache_async_write_mutex,
									   10000L))		/* 10sec */
			{
				/* someone wake up this worker */
				pg_atomic_fetch_sub_u32(&parquet_cache_async_write_num_available_workers, 1);
			}
			else
			{
				/* nobody wake up this worker */
				pg_atomic_fetch_sub_u32(&parquet_cache_async_write_num_available_workers, 1);
				/* maintenance work */
				if (noop_counter++ < 30)
					__parquetCacheReclaimBlocks(-1);
				else
				{
					/* after 5min of no-operations, exit worker-thread */
					break;
				}
			}
		}
	}
	pg_atomic_fetch_sub_u32(&parquet_cache_async_write_num_workers, 1);
	pthreadMutexUnlock(&parquet_cache_async_write_mutex);
	free(malloc_ptr);
	__Info("parquet-cache: async write worker exit (tid: %u)", gettid());
	return NULL;
}

void
parquetCacheWriteAsync(parquetCacheWriteAsyncData *data)
{
	uint32_t	ntasks;

	if (!parquet_cache_head)
		goto out;
	ntasks = pg_atomic_read_u32(&parquet_cache_async_write_num_active_tasks);
	while (ntasks < pgstrom_parquet_cache_max_async_write)
	{
		if (pg_atomic_compare_exchange_u32(&parquet_cache_async_write_num_active_tasks,
										   &ntasks, ntasks+1))
		{
			pthreadMutexLock(&parquet_cache_async_write_mutex);
			dlist_push_tail(&parquet_cache_async_write_list, &data->chain);
			pthreadCondSignal(&parquet_cache_async_write_cond);
			pthreadMutexUnlock(&parquet_cache_async_write_mutex);
			/* launch worker thread if it looks lacking */
			if (pg_atomic_read_u32(&parquet_cache_async_write_num_available_workers) == 0)
			{
				pthread_t	thread;

				if ((errno = pthread_create(&thread,
											NULL,
											__parquetCacheAsyncWriteWorker,
											NULL)) != 0)
					__FATAL("failed on pthread_create: %m\n");
				if ((errno = pthread_detach(thread)) != 0)
					__FATAL("failed on pthread_detach: %m\n");
			}
			return;
		}
	}
out:
	/* release async task immediately */
	releaseParquetCacheWriteAsyncData(data);
}

/*
 * pgstrom_parquet_cache_info
 */
PG_FUNCTION_INFO_V1(pgstrom_parquet_cache_info);
PUBLIC_FUNCTION(Datum)
pgstrom_parquet_cache_info(PG_FUNCTION_ARGS)
{
	StringInfoData	buf;
	uint64_t	num_active_blocks;
	uint64_t	num_cached_chunks;
	uint64_t	total_actual_usage;
	size_t		cache_size = (size_t)pgstrom_parquet_cache_size_mb << 20;
	size_t		block_size = (size_t)pgstrom_parquet_cache_unitsz_mb << 20;

	if (!pgstrom_parquet_cache_path)
		PG_RETURN_POINTER(cstring_to_text("{ \"cache_path\" : null }"));
	assert(parquet_cache_head != NULL);
	num_active_blocks = pg_atomic_read_u64(&parquet_cache_head->num_active_blocks);
	num_cached_chunks = pg_atomic_read_u64(&parquet_cache_head->num_cached_chunks);
	total_actual_usage = pg_atomic_read_u64(&parquet_cache_head->total_actual_usage);

	initStringInfo(&buf);
	appendStringInfoSpaces(&buf, VARHDRSZ);
	appendStringInfo(&buf,
					 "{ \"cache_path\" : \"%s\""
					 ", \"cache_size\" : \"%s\""
					 ", \"block_size\" : \"%s\""
					 ", \"hash_nslots\" : %lu"
					 ", \"active_blocks\" : %ld"
					 ", \"free_blocks\" : %ld"
					 ", \"active_ratio\" : \"%.2f%%\""
					 ", \"num_cached_chunks\" : %ld"
					 ", \"total_actual_usage\" : \"%s\"",
					 pgstrom_parquet_cache_path,
					 format_bytesz(cache_size),
					 format_bytesz(block_size),
					 parquet_cache_head->hash_nslots,
					 num_active_blocks,
					 parquet_cache_head->num_blocks - num_active_blocks,
					 100.0 * (double)num_active_blocks / (double)parquet_cache_head->num_blocks,
					 num_cached_chunks,
					 format_bytesz(total_actual_usage));
	if (num_active_blocks > 0)
	{
		double	ratio = 100.0 * (double)total_actual_usage /
								(double)(num_active_blocks * block_size);
		appendStringInfo(&buf,", \"cache_usage_ratio\" : %.2f%%", ratio);
	}
	else
	{
		appendStringInfo(&buf,", \"cache_usage_ratio\" : null");
	}
	appendStringInfo(&buf, " }");

	SET_VARSIZE(buf.data, buf.len);
	PG_RETURN_POINTER(buf.data);
}

/*
 * __request_parquet_cache_size
 */
static size_t
__request_parquet_cache_size(void)
{
	return (MAXALIGN(sizeof(parquetCacheHead)) +
			MAXALIGN(sizeof(pthread_mutex_t)   * pgstrom_parquet_cache_nslots) +
			MAXALIGN(sizeof(dlist_head)        * pgstrom_parquet_cache_nslots) +
			MAXALIGN(sizeof(parquetCacheEntry) * pgstrom_parquet_cache_nblocks));
}

/*
 * pgstrom_request_parquet_cache
 */
static void
pgstrom_request_parquet_cache(void)
{
	if (shmem_request_next)
		shmem_request_next();
	RequestAddinShmemSpace(__request_parquet_cache_size());
}

/*
 * pgstrom_startup_parquet_cache
 */
static void
pgstrom_startup_parquet_cache(void)
{
	pthread_mutex_t *hash_locks;
	dlist_head	   *hash_slots;
	char		   *pos;
	bool			found;
	int				fdesc;
	size_t			length;

	if (shmem_startup_next)
		shmem_startup_next();

	/*
	 * setup shared memory structure
	 */
	pos = ShmemInitStruct("parquetCacheHead",
						  __request_parquet_cache_size(),
						  &found);
	Assert(!found);
	parquet_cache_head = (parquetCacheHead *)pos;
	pos += MAXALIGN(sizeof(parquetCacheHead));
	hash_locks = (pthread_mutex_t *)pos;
	pos += MAXALIGN(sizeof(pthread_mutex_t) * pgstrom_parquet_cache_nslots);
	hash_slots = (dlist_head *)pos;
	pos += MAXALIGN(sizeof(dlist_head)      * pgstrom_parquet_cache_nslots);
	parquet_cache_array = (parquetCacheEntry *)pos;
	for (uint32_t i=0; i < pgstrom_parquet_cache_nslots; i++)
	{
		pthreadMutexInit(&hash_locks[i]);
		dlist_init(&hash_slots[i]);
	}
	pos += MAXALIGN(sizeof(parquetCacheEntry) * pgstrom_parquet_cache_nblocks);

	parquet_cache_head->hash_locks = hash_locks;
	parquet_cache_head->hash_slots = hash_slots;
	parquet_cache_head->hash_nslots = pgstrom_parquet_cache_nslots;
	parquet_cache_head->num_blocks = pgstrom_parquet_cache_nblocks;
	pthreadMutexInit(&parquet_cache_head->free_lock);
	dlist_init(&parquet_cache_head->free_list);
	pthreadMutexInit(&parquet_cache_head->lru_lock);
	dlist_init(&parquet_cache_head->lru_list);
	pg_atomic_init_u64(&parquet_cache_head->num_active_blocks, 0);
	pg_atomic_init_u64(&parquet_cache_head->num_cached_chunks, 0);
	pg_atomic_init_u64(&parquet_cache_head->total_actual_usage, 0);

	for (uint32_t i=0; i < pgstrom_parquet_cache_nblocks; i++)
	{
		parquetCacheEntry *entry = &parquet_cache_array[i];

		memset(entry, 0, sizeof(parquetCacheEntry));
		dlist_push_tail(&parquet_cache_head->free_list, &entry->chain);
	}

	/*
	 * cache file creation
	 */
	fdesc = open(pgstrom_parquet_cache_path,
				 O_RDWR | O_CREAT | O_DIRECT,
				 0600);
	if (fdesc < 0)
		elog(ERROR, "failed on open('%s'): %m [parquet cache file]",
			 pgstrom_parquet_cache_path);
	length = ((size_t)pgstrom_parquet_cache_size_mb << 20);
	if (ftruncate(fdesc, length) < 0)
		elog(ERROR, "failed on ftruncate('%s', %ld): %m",
			 pgstrom_parquet_cache_path, length);
	pgstrom_parquet_cache_fdesc = fdesc;
}

/*
 * pgstrom_init_parquet_cache
 */
void
pgstrom_init_parquet_cache(void)
{
	DefineCustomStringVariable("pg_strom.parquet_cache_path",
							   "Path of Parquet cache file",
							   NULL,
							   &pgstrom_parquet_cache_path,
							   NULL,	/* disabled by default */
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.parquet_cache_size",
							"Size of Parquet cache file",
							NULL,
							&pgstrom_parquet_cache_size_mb,
							160 * 1024,		/* 160GB */
							32 * 1024,		/* 32GB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MB,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.parquet_cache_unitsz",
							"Size of Parquet cache allocation unit",
							NULL,
							&pgstrom_parquet_cache_unitsz_mb,
							2,		/* 2MB */
							1,		/* 1MB */
							256,	/* 256MB */
							PGC_POSTMASTER | GUC_NO_SHOW_ALL,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MB,
							NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.parquet_cache_eviction_min_seconds",
							 "[experimental] minimum seconds before Parquet cache eviction",
							 NULL,
							 &pgstrom_parquet_cache_eviction_min_seconds,
							 600.0,	/* 10min */
							 0.0,
							 FLT_MAX,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.parquet_cache_eviction_max_seconds",
							 "[experimental] maxmum seconds before Parquet cache eviction",
							 NULL,
							 &pgstrom_parquet_cache_eviction_max_seconds,
							 7200.0,	/* 2 hour */
							 pgstrom_parquet_cache_eviction_min_seconds,
							 FLT_MAX,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.parquet_cache_eviction_bonus",
							 "[experimental] bonus seconds per access for Parquet cache eviction",
							 NULL,
							 &pgstrom_parquet_cache_eviction_bonus,
							 2.0,		/* 2.0sec per access */
							 0.0,
							 FLT_MAX,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.parquet_cache_target_usage",
							 "kick Parquet cache eviction when active block ratio exceed this value",
							 NULL,
							 &pgstrom_parquet_cache_target_usage,
							 98.0,
							 80.0,
							 100.0,
							 PGC_POSTMASTER | GUC_NO_SHOW_ALL,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.parquet_cache_max_async_write",
							"max number of asynchronous cache writeback tasks",
							NULL,
							&pgstrom_parquet_cache_max_async_write,
							80,
							1,
							INT_MAX,
							PGC_POSTMASTER | GUC_NO_SHOW_ALL,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	if ((pgstrom_parquet_cache_unitsz_mb & (pgstrom_parquet_cache_unitsz_mb-1)) != 0)
		elog(ERROR, "pg_strom.parquet_cache_unitsz must be 2^N MB");
	if (1000 * pgstrom_parquet_cache_unitsz_mb >= pgstrom_parquet_cache_size_mb)
		elog(ERROR, "pg_strom.parquet_cache_size (%d) is too small. it must be 1000 * pg_strom.parquet_cache_unitsz (%d) for valuable disk caching",
			 pgstrom_parquet_cache_unitsz_mb,
			 pgstrom_parquet_cache_size_mb);
	/* number of cache blocks */
	pgstrom_parquet_cache_nblocks
		= (pgstrom_parquet_cache_size_mb +
		   pgstrom_parquet_cache_unitsz_mb - 1) / pgstrom_parquet_cache_unitsz_mb;
	/* number of hash slots */
	pgstrom_parquet_cache_nslots = ((double)pgstrom_parquet_cache_nblocks /
									log((double)pgstrom_parquet_cache_nblocks)) + 2000.0;
	/* async write stuff */
	pthreadMutexInit(&parquet_cache_async_write_mutex);
	pthreadCondInit(&parquet_cache_async_write_cond);
	dlist_init(&parquet_cache_async_write_list);
	pg_atomic_init_u32(&parquet_cache_async_write_num_workers, 0);
	pg_atomic_init_u32(&parquet_cache_async_write_num_available_workers, 0);
	pg_atomic_init_u32(&parquet_cache_async_write_num_active_tasks, 0);

	/* shared memory allocation callback */
	if (pgstrom_parquet_cache_path)
	{
		elog(LOG, "Parquet cache file [%s] (size: %s, block: %uMB, nslots: %lu)",
			 pgstrom_parquet_cache_path,
			 format_bytesz((size_t)pgstrom_parquet_cache_size_mb << 20),
			 pgstrom_parquet_cache_unitsz_mb,
			 pgstrom_parquet_cache_nslots);
		shmem_request_next = shmem_request_hook;
		shmem_request_hook = pgstrom_request_parquet_cache;
		shmem_startup_next = shmem_startup_hook;
		shmem_startup_hook = pgstrom_startup_parquet_cache;
	}
}
