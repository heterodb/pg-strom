/*
 * bufmgr.c - buffer management of shared memory segment
 *
 * Copyright (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "pg_boost.h"
#include <alloca.h>

/*
 * shmbuf_head_t
 */
#define SHMBUF_NUM_ACTIVE_SLOT	1024

typedef struct {
	shmlist_t		active_list[SHMBUF_NUM_ACTIVE_SLOT];
	pthread_mutex_t	active_lock[SHMBUF_NUM_ACTIVE_SLOT];
	int				reclaim_hint;
	int				cache_hint;
	size_t			storage_size_total;
	size_t			cache_size_total;
	size_t			cache_size_limit;
} shmbuf_head_t;

/*
 * shmbuf_create
 *
 *
 *
 */
bool
shmbuf_create(shmbuf_t *shmbuf, int tag, size_t size)
{
	shmbuf_head_t  *sbh = (shmbuf_head_t *)shmmgr_get_bufmgr_head();
	void   *cached;
	int		index;

	cached = shmmgr_alloc(size);
	if (!cached)
		return false;

	index = __sync_fetch_and_and(&sbh->cache_hint, 1)
		% SHMBUF_NUM_ACTIVE_SLOT;

	shmbuf->storage = addr_to_offset(NULL);
	shmbuf->cached = addr_to_offset(cached);
	shmbuf->size = size;
	shmbuf->refcnt = 0;
	shmbuf->index = index;
	shmbuf->tag = tag;
	shmbuf->flags = SHMBUF_FLAGS_DIRTY_CACHE | SHMBUF_FLAGS_HOT_CACHE;
	shmmgr_init_mutex(&shmbuf->lock);

	__sync_add_and_fetch(&sbh->cache_size_total, shmmgr_get_size(shmbuf));

	pthread_mutex_lock(&sbh->active_lock[index]);

	shmlist_add(&sbh->active_list[index], &shmbuf->list);

	pthread_mutex_unlock(&sbh->active_lock[index]);

	return true;
}

/*
 * shmbuf_delete
 *
 * Note that caller side must ensure the shmbuf being deleted is never
 * referenced to others.
 */
void
shmbuf_delete(shmbuf_t *shmbuf)
{
	shmbuf_head_t  *sbh = (shmbuf_head_t *)shmmgr_get_bufmgr_head();
	void   *storage = offset_to_addr(shmbuf->storage);
	void   *cached = offset_to_addr(shmbuf->cached);

	if (storage)
	{
		__sync_sub_and_fetch(&sbh->storage_size_total,
							 shmmgr_get_size(shmbuf));
		shmmgr_free(storage);
	}

	if (cached)
	{
		int	index = shmbuf->index;

		pthread_mutex_lock(&sbh->active_lock[index]);
		shmlist_del(&shmbuf->list);
		pthread_mutex_unlock(&sbh->active_lock[index]);

		__sync_sub_and_fetch(&sbh->cache_size_total,
							 shmmgr_get_size(shmbuf));
		shmmgr_free(cached);
	}
}

/*
 * shmbuf_reclaim
 *
 *
 *
 */
static size_t
shmbuf_try_reclaim(shmbuf_head_t *sbh, shmbuf_t *shmbuf)
{
	char   *cached = offset_to_addr(shmbuf->cached);
	size_t	result;

	Assert(cached != NULL);

	if (shmbuf->refcnt > 0 || (shmbuf->flags & SHMBUF_FLAGS_HOT_CACHE) != 0)
	{
		/*
		 * If and when this shmbuf is pinned or recently referenced,
		 * it is not available to reclaim this cache.
		 * We just return with clearing HOT_CACHE flag.
		 */
		shmbuf->flags &= ~SHMBUF_FLAGS_HOT_CACHE;
		result = 0;
	}
	else if ((shmbuf->flags & SHMBUF_FLAGS_DIRTY_CACHE) == 0)
	{
		/*
		 * Simple case; if cache is not dirty, all we need to do is
		 * just release this cache.
		 */
		result = shmmgr_get_size(cached);
		shmbuf->cached = 0;
		shmlist_del(&shmbuf->list);
		shmbuf->flags &= ~(SHMBUF_FLAGS_DIRTY_CACHE | SHMBUF_FLAGS_HOT_CACHE);

		__sync_sub_and_fetch(&sbh->cache_size_total, result);
		shmmgr_free(cached);
	}
	else
	{
		/*
		 * Complex case; if cache is dirty, we need to write back this
		 * change into storage area. 
		 * In high-memory-presure situation, it may not be available to
		 * compress the dirty cache. In this case, we 
		 *
		 */
		struct PGLZ_Header *temp = alloca(PGLZ_MAX_OUTPUT(shmbuf->size));
		char   *storage_old = offset_to_addr(shmbuf->storage);
		char   *storage_new;

		result = shmmgr_get_size(cached);

		if (pglz_compress(cached, shmbuf->size, temp, PGLZ_strategy_default) &&
			(storage_new = shmmgr_alloc(VARSIZE(temp))) != NULL)
		{
			memcpy(storage_new, VARDATA(temp), VARSIZE(temp));
			shmbuf->storage = addr_to_offset(storage_new);
			shmbuf->cached = 0;
			shmlist_del(&shmbuf->list);
			shmbuf->flags = SHMBUF_FLAGS_COMPRESSED;
			shmmgr_free(cached);

			__sync_add_and_fetch(&sbh->storage_size_total,
								 shmmgr_get_size(storage_new));
			__sync_sub_and_fetch(&sbh->cache_size_total, result);
		}
		else
		{
			shmbuf->storage = shmbuf->cached;
			shmbuf->cached = 0;
			shmlist_del(&shmbuf->list);
			shmbuf->flags = 0;

			__sync_add_and_fetch(&sbh->storage_size_total, result);
			__sync_sub_and_fetch(&sbh->cache_size_total, result);
		}

		/* release older storage, if exist */
		if (storage_old)
		{
			__sync_fetch_and_sub(&sbh->storage_size_total,
								 shmmgr_get_size(storage_old));
			shmmgr_free(storage_old);
		}
	}
	return result;
}

void
shmbuf_reclaim(size_t size)
{
	shmbuf_head_t *sbh = (shmbuf_head_t *)shmmgr_get_bufmgr_head();
	shmbuf_t   *shmbuf;
	size_t		reclaimed = 0;
	int			index;

	while (reclaimed < size)
	{
		index = __sync_fetch_and_add(sbh->reclaim_hint, 1)
			% SHMBUF_NUM_ACTIVE_SLOT;

		pthread_mutex_lock(&sbh->active_lock[index]);

		shmlist_foreach_entry(shmbuf, &sbh->active_list[index], list)
		{
			pthread_mutex_lock(&shmbuf->lock);

			PG_TRY();
			{
				reclaimed += shmbuf_try_reclaim(sbh, shmbuf);
			}
			PG_CATCH();
			{
				pthread_mutex_unlock(&shmbuf->lock);
				pthread_mutex_unlock(&sbh->active_lock[index]);
				PG_RE_THROW();
			}
			PG_END_TRY();

			pthread_mutex_unlock(&shmbuf->lock);
		}
		pthread_mutex_unlock(&sbh->active_lock[index]);
	}
}

/*
 * shmbuf_get_buffer
 *
 *
 */
void *
shmbuf_get_buffer(shmbuf_t *shmbuf)
{
	shmbuf_head_t *sbh = (shmbuf_head_t *)shmmgr_get_bufmgr_head();
	void   *cached;

	pthread_mutex_lock(&shmbuf->lock);

	cached = offset_to_addr(shmbuf->cached);
	if (!cached)
	{
		void   *storage = offset_to_addr(shmbuf->storage);
		int		reclaim;
		int		index;

		/*
		 * XXX - uncached shmbuf never chained to active list,
		 * so we can call shmbuf_reclaim with holding the lock.
		 */
		while ((cached = shmmgr_alloc(shmbuf->size)) == NULL)
			shmbuf_reclaim(shmbuf->size);
		__sync_fetch_and_add(&sbh->cache_size_total,
							 shmmgr_get_size(cached));
		/*
		 * Reclaim other cache, if overusage
		 */
		reclaim = sbh->cache_size_total - sbh->cache_size_limit;
		if (reclaim > 0)
			shmbuf_reclaim(reclaim);

		if (shmbuf->flags & SHMBUF_FLAGS_COMPRESSED)
		{
			PG_TRY()
			{
				pglz_decompress(storage, cached);
			}
			PG_CATCH();
			{
				pthread_mutex_unlock(&shmbuf->lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
		}
		else
		{
			memcpy(cached, storage, shmbuf->size);
		}
		shmbuf->cached = cached;
		shmbuf->flags &= ~SHMBUF_FLAGS_DIRTY_CACHE;

		index = __sync_fetch_and_and(&sbh->cache_hint, 1)
			% SHMBUF_NUM_ACTIVE_SLOT;

		pthread_mutex_lock(&sbh->active_lock[index]);

		shmlist_add(&sbh->active_list[index], &shmbuf->list);

		pthread_mutex_unlock(&sbh->active_lock[index]);
	}
	shmbuf->refcnt++;
	shmbuf->flags |= SHMBUF_FLAGS_HOT_CACHE;

	pthread_mutex_unlock(&shmbuf->lock);

	return cached;
}

/*
 * shmbuf_put_buffer
 *
 *
 */
void
shmbuf_put_buffer(shmbuf_t *shmbuf)
{
	pthread_mutex_lock(&shmbuf->lock);
	shmbuf->refcnt--;
	pthread_mutex_unlock(&shmbuf->lock);
}

/*
 * shmbuf_set_dirty
 *
 */
void
shmbuf_set_dirty(shmbuf_t *shmbuf)
{
	pthread_mutex_lock(&shmbuf->lock);
	shmbuf->flags |= SHMBUF_FLAGS_DIRTY_CACHE;
	pthread_mutex_unlock(&shmbuf->lock);
}

/*
 * shmbuf_init
 *
 *
 */
offset_t
shmbuf_init(size_t size)
{
	shmbuf_head_t *sbh;
	int		i;

	sbh = shmmgr_alloc(sizeof(shmbuf_head_t));
	if (!sbh)
		return false;

	for (i=0; i < SHMBUF_NUM_ACTIVE_SLOT; i++)
	{
		shmlist_init(&sbh->active_list[i]);
		shmmgr_init_mutex(&sbh->active_lock[i]);
	}
	sbh->reclaim_hint = 0;
	sbh->cache_hint = 0;
	sbh->cache_size_total = 0;
	sbh->cache_size_limit = size * 33 / 100; // To be fixed

	return addr_to_offset(sbh);
}
