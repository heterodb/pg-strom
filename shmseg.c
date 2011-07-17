/*
 * shmseg.c
 *
 * Routines to manage shared memory segment
 *
 * Copyright (c) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 */
#include "pg_boost.h"
#include "utils/guc.h"
#include "utils/pg_lzcompress.h"
#include <sys/types.h>
#include <sys/shm.h>


#define SHMSEG_CLASS_MIN_BITS	6		/* 64bytes */
#define SHMSEG_CLASS_MAX_BITS	30		/* 1GBytes */
#define SHMSEG_CLASS_MIN_SIZE	(1 << SHMSEG_CLASS_MIN_BITS)
#define SHMSEG_CLASS_MAX_SIZE	(1 << SHMSEG_CLASS_MAX_BITS)
#define SHMSEG_CACHE_NUM_SLOTS	64

typedef struct {
	int		shmid;
	size_t	segment_size;
	size_t	active_size;
	size_t	free_size;
	mlist_t	free_list[SHMSEG_CLASS_MAX_BITS + 1];
	int		num_active[SHMSEG_CLASS_MAX_BITS + 1];
	int		num_free[SHMSEG_CLASS_MAX_BITS + 1];
	pthread_mutex_t	lock;
	/*
	 * memory cache mechanism
	 */
	mlist_t			mcache_list[SHMSEG_CACHE_NUM_SLOTS];
	pthread_mutex_t	mcache_lock[SHMSEG_CACHE_NUM_SLOTS];
	int				mcache_index;
	int				mcache_reclaim;
	size_t			mcache_size;
	size_t			mcache_limit;
} msegment_t;

typedef struct {
	uint16		mclass;
	bool		is_active;
	bool		is_cached;
} mchunk_common_t;

typedef struct {
	mchunk_common_t	c;
	mlist_t			list;
} mchunk_free_t;

typedef struct {
	mchunk_common_t	c;
	uintptr_t	data[0];
} mchunk_item_t;

#define SHMSEG_FLAG_DIRTY_CACHE		0x01
#define SHMSEG_FLAG_HOT_CACHE		0x02
#define SHMSEG_FLAG_COMPRESSED		0x04

typedef struct mchunk_cache_s {
	mchunk_common_t	c;
	uint32		refcnt;
	void	   *storage;
	void	   *cached;
	mlist_t		list;
	uint32		length;
	uint16		index;
	uint8		tag;
	uint8		flags;
	pthread_mutex_t lock;
} mchunk_cache_t;

/*
 * Static variables
 */
static msegment_t  *msegment = NULL;
static pthread_mutexattr_t  msegment_mutex_attr;
static pthread_rwlockattr_t msegment_rwlock_attr;

#define addr_to_offset(addr)	(((uintptr_t)(addr)) - ((uintptr_t)(msegment)))
#define offset_to_addr(offset)	(void *)(((uintptr_t)(msegment)) + (offset))

/*
 * ffs - returns first (smallest) bit of the value
 */
static inline int fast_ffs(uintptr_t value)
{
	return __builtin_ffsl((unsigned long)value);
}

/*
 * fls - returns last (biggest) bit of the value
 */
static inline int fast_fls(uintptr_t value)
{
	if (!value)
		return 0;
	return sizeof(value) * 8 - __builtin_clz(value);
}

static bool shmseg_split_chunk(int mclass)
{
	return true;
}

void *shmseg_alloc(size_t size)
{
	void   *result = shmseg_try_alloc(size);

	if (!result)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("pg_boost: out of shared memory")));
	return result;
}

void *shmseg_try_alloc(size_t size)
{
	mchunk_item_t  *mitem;
	mlist_t		   *list;
	int				mclass;

	mclass = fast_fls(size + offsetof(mchunk_item_t, data) - 1);
	if (mclass > SHMSEG_CLASS_MAX_BITS)
		return NULL;
	if (mclass < SHMSEG_CLASS_MIN_BITS)
		mclass = SHMSEG_CLASS_MIN_BITS;

	pthread_mutex_lock(&msegment->lock);

	/*
	 * If and when free_list of the mclass is not available,
	 * it tries to split a larger free chunk into two.
	 * If unavailable anymore, we cannot allocate a new free
	 * chunk.
	 */
	if (mlist_empty(&msegment->free_list[mclass]))
	{
		if (!shmseg_split_chunk(mclass + 1))
		{
			pthread_mutex_unlock(&msegment->lock);
			return NULL;
		}
	}
	Assert(!mlist_empty(&msegment->free_list[mclass]));

	list = msegment->free_list[mclass].next;
	mitem = (mchunk_item_t *)container_of(list, mchunk_free_t, list);
	mlist_del(list);

	Assert(mitem->c.mclass == mclass);
	mitem->c.is_active = true;
	mitem->c.is_cached = false;

	msegment->num_free[mclass]--;
	msegment->num_active[mclass]++;
	msegment->active_size += (1 << mclass);
	msegment->free_size -= (1 << mclass);

	pthread_mutex_unlock(&msegment->lock);

	return (void *)mitem->data;
}

void shmseg_free(void *addr)
{
	mchunk_free_t  *chunk = (mchunk_free_t *)container_of(addr, mchunk_item_t, data);
	mchunk_free_t  *buddy;
	uintptr_t		offset_chunk;
	uintptr_t		offset_buddy;
	int				mclass = chunk->c.mclass;

	pthread_mutex_lock(&msegment->lock);

	chunk->c.is_active = false;
	chunk->c.is_cached = false;
	msegment->num_active[mclass]--;
	msegment->active_size -= (1 << mclass);

	/*
	 * If its buddy is also free, we consolidate them to one
	 */
	offset_chunk = addr_to_offset(chunk);

	while (mclass < SHMSEG_CLASS_MAX_BITS)
	{
		if (offset_chunk & (1 << mclass))
			offset_buddy = offset_chunk & ~(1 << mclass);
		else
			offset_buddy = offset_chunk | (1 << mclass);

		/* buddy should not be exist within msegment_t struct */
		if (offset_buddy < sizeof(msegment_t))
			break;

		/*
		 * If buddy is also free and same mclass, consolidate them
		 */
		buddy = offset_to_addr(offset_buddy);
		if (buddy->c.is_active || buddy->c.mclass != mclass)
			break;

		mlist_del(&buddy->list);
		msegment->num_free[mclass]--;
		msegment->free_size -= (1 << mclass);

		mclass++;
		offset_chunk &= ~((1 << mclass) - 1);
		chunk = offset_to_addr(offset_chunk);

		chunk->c.mclass = mclass;
		chunk->c.is_active = false;
		chunk->c.is_cached = false;
	}
	/*
	 * Attach this chunk to free_list[mclass]
	 */
	mlist_add(&msegment->free_list[mclass], &chunk->list);
	msegment->num_free[mclass]++;
	msegment->free_size += (1 << mclass);

	pthread_mutex_unlock(&msegment->lock);
}

size_t shmseg_size(void *addr)
{
	mchunk_item_t  *chunk = container_of(addr, mchunk_item_t, data);

	return (1 << chunk->c.mclass);
}

void *shmseg_try_resize(void *addr, size_t new_size)
{
	mchunk_item_t  *mitem = container_of(addr, mchunk_item_t, data);
	void		   *data_new;
	int				mclass_new;

	mclass_new = fast_fls(new_size + offsetof(mchunk_item_t, data) - 1);
	if (mclass_new > SHMSEG_CLASS_MAX_BITS)
		return NULL;
	if (mclass_new < SHMSEG_CLASS_MIN_BITS)
		mclass_new = SHMSEG_CLASS_MIN_BITS;

	/* unchanged case */
	if (mitem->c.mclass == mclass_new)
		return mitem->data;

	/* reduction case */
	if (mitem->c.mclass > mclass_new)
	{
		uintptr_t	offset_s = addr_to_offset(mitem) + (1 << mclass_new);
		uintptr_t	offset_e = addr_to_offset(mitem) + (1 << mitem->c.mclass);

		pthread_mutex_lock(&msegment->lock);

		msegment->num_active[mitem->c.mclass]--;
		mitem->c.mclass = mclass_new;
		msegment->num_active[mitem->c.mclass]++;

		while (offset_s < offset_e)
		{
			mchunk_free_t  *mfree = offset_to_addr(offset_s);
			int				mclass = fast_ffs(offset_s);

			if (mclass > SHMSEG_CLASS_MAX_BITS)
				mclass = SHMSEG_CLASS_MAX_BITS;
			Assert(mclass >= SHMSEG_CLASS_MAX_BITS);

			/* if (offset + chunk_size) over the tail, truncate it */
			while (offset_s + (1 << mclass) > offset_e)
				mclass--;
			Assert(mclass < SHMSEG_CLASS_MIN_BITS);

			mfree->c.mclass = mclass;
			mfree->c.is_active = false;
			mfree->c.is_cached = false;
			mlist_add(&msegment->free_list[mclass], &mfree->list);

			offset_s += (1 << mclass);
			msegment->num_free[mclass]++;
		}
		pthread_mutex_unlock(&msegment->lock);

		return mitem->data;
	}

	/* expand case */
	data_new = shmseg_try_alloc(new_size);
	memcpy(data_new, mitem->data,
		   (1 << mitem->c.mclass) - offsetof(mchunk_item_t, data));
	shmseg_free(mitem->data);

	return data_new;
}

void *shmseg_resize(void *addr, size_t new_size)
{
	void   *result = shmseg_try_resize(addr, new_size);

	if (!result)
		ereport(ERROR,
                (errcode(ERRCODE_OUT_OF_MEMORY),
                 errmsg("pg_boost: out of shared memory")));
	return result;
}

static void shmseg_try_cache_reclaim(mchunk_cache_t *mcache)
{
	Assert(mcache->cached != NULL);

	/*
	 * Not available to reclaim cache being pinned.
	 */
	if (mcache->refcnt > 0)
		return;

	/*
	 * Postpone cache buffer being recently referenced.
	 */
	if (mcache->flags & SHMSEG_FLAG_HOT_CACHE)
	{
		mcache->flags &= ~SHMSEG_FLAG_HOT_CACHE;
		return;
	}

	/*
	 * Complex case: the cache buffer is dirty, or does
	 * not have storage area. In this case, we need to
	 * allocate a new storage area, and write to compressed
	 * data.
	 */
	if (mcache->flags & SHMSEG_FLAG_DIRTY_CACHE || !mcache->storage)
	{
		struct PGLZ_Header *temp = alloca(PGLZ_MAX_OUTPUT(mcache->length));
		size_t		cached_size = shmseg_size(mcache->cached);

		if (pglz_compress(mcache->cached, mcache->length, temp,
						  PGLZ_strategy_default))
		{
			void   *storage_new
				= shmseg_resize(mcache->cached, VARSIZE(temp));
			memcpy(storage_new, VARDATA(temp), VARSIZE(temp));

			mlist_del(&mcache->list);
			if (mcache->storage)
				shmseg_free(mcache->storage);
			mcache->storage = storage_new;
			mcache->cached = NULL;
			mcache->flags = SHMSEG_FLAG_COMPRESSED;

			__sync_fetch_and_sub(&msegment->mcache_size, cached_size);
		}
		else
		{
			mlist_del(&mcache->list);
			if (mcache->storage)
				shmseg_free(mcache->storage);
			mcache->storage = mcache->cached;
			mcache->cached = NULL;
			mcache->flags = 0;

			__sync_fetch_and_sub(&msegment->mcache_size, cached_size);
		}
		return;
	}

	/*
	 * Simple case: the cache buffer is not dirty, and
	 * it already have a storage area. So, all we need
	 * to do is just release cache buffer.
	 */
	mlist_del(&mcache->list);
	__sync_fetch_and_sub(&msegment->mcache_size,
						 shmseg_size(mcache->cached));
	shmseg_free(mcache->cached);
	mcache->cached = NULL;

	return;
}

static void shmseg_cache_reclaim(size_t size)
{
	mchunk_cache_t *mcache;
	mchunk_cache_t *temp;

	while (msegment->mcache_limit >= msegment->mcache_size + size)
	{
		int	index = __sync_fetch_and_add(&msegment->mcache_reclaim, 1)
			% SHMSEG_CACHE_NUM_SLOTS;

		pthread_mutex_lock(&msegment->mcache_lock[index]);

		mlist_foreach_entry_safe(mcache, temp, &msegment->mcache_list[index], list)
		{
			pthread_mutex_lock(&mcache->lock);
			PG_TRY();
			{
				shmseg_try_cache_reclaim(mcache);
			}
			PG_CATCH();
			{
				pthread_mutex_unlock(&mcache->lock);
				pthread_mutex_unlock(&msegment->mcache_lock[index]);
				PG_RE_THROW();
			}
			PG_END_TRY();
			pthread_mutex_unlock(&mcache->lock);
		}
		pthread_mutex_unlock(&msegment->mcache_lock[index]);
	}
}

mchunk_cache_t *shmseg_cache_alloc(size_t size)
{
	mchunk_cache_t *mcache;
	void		   *cached;
	int				index;
	int				retry = 4;

	mcache = (mchunk_cache_t *)container_of(shmseg_alloc(sizeof(mchunk_cache_t)),
											mchunk_item_t, data);
	while ((cached = shmseg_try_alloc(size)) == NULL && retry-- > 0)
		shmseg_cache_reclaim(size);

	if (!cached)
	{
		shmseg_free(mcache);
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("pg_boost: out of shared memory")));
	}
	index = __sync_fetch_and_add(&msegment->mcache_index, 1);

	mcache->c.is_cached = true;
	mcache->refcnt = 0;
	mcache->storage = NULL;
	mcache->cached = cached;
	mcache->length = size;
	mcache->index = index;
	mcache->flags = SHMSEG_FLAG_DIRTY_CACHE | SHMSEG_FLAG_HOT_CACHE;

	pthread_mutex_lock(&msegment->mcache_lock[index]);
	mlist_add(&msegment->mcache_list[index], &mcache->list);
	pthread_mutex_unlock(&msegment->mcache_lock[index]);

	return mcache;
}

void shmseg_cache_free(mchunk_cache_t *mcache)
{
	if (mcache->storage)
		shmseg_free(mcache->storage);

	if (mcache->cached)
	{
		int		index = mcache->index;

		pthread_mutex_lock(&msegment->mcache_lock[index]);
		mlist_del(&mcache->list);
		pthread_mutex_unlock(&msegment->mcache_lock[index]);

		shmseg_free(mcache->cached);
	}
	shmseg_free(((mchunk_item_t *)(mcache))->data);
}

void *shmseg_cache_get(mchunk_cache_t *mcache)
{
	pthread_mutex_lock(&mcache->lock);
	if (!mcache->cached)
	{
		int		index;
		void   *cached = NULL;

		Assert(mcache->storage != NULL);

		if ((mcache->flags & SHMSEG_FLAG_COMPRESSED) == 0)
		{
			/*
			 * In the case when storage is not compressed, we don't
			 * need to have two same buffers. So, just swap them.
			 */
			mcache->cached = mcache->storage;
			mcache->storage = NULL;

			mcache->flags |= SHMSEG_FLAG_DIRTY_CACHE;
		}
		else
		{
			PG_TRY();
			{
				int		retry = 4;

				while ((cached = shmseg_try_alloc(mcache->length)) == NULL
					   && retry-- > 0)
					shmseg_cache_reclaim(mcache->length);

				pglz_decompress(mcache->storage, cached);
			}
			PG_CATCH();
			{
				if (cached)
					shmseg_free(cached);
				pthread_mutex_unlock(&mcache->lock);
				PG_RE_THROW();
			}
			PG_END_TRY();

			mcache->cached = cached;
			mcache->flags &= ~SHMSEG_FLAG_DIRTY_CACHE;
		}
		__sync_fetch_and_add(&msegment->mcache_size, shmseg_size(mcache->cached));
		shmseg_cache_reclaim(0);

		index = __sync_fetch_and_add(&msegment->mcache_index, 1) % SHMSEG_CACHE_NUM_SLOTS;
		pthread_mutex_lock(&msegment->mcache_lock[index]);
		mlist_add(&msegment->mcache_list[index], &mcache->list);
		pthread_mutex_unlock(&msegment->mcache_lock[index]);
	}
	mcache->refcnt++;
	mcache->flags |= SHMSEG_FLAG_HOT_CACHE;
	pthread_mutex_unlock(&mcache->lock);

	return mcache->cached;
}

void shmseg_cache_put(mchunk_cache_t *mcache, bool set_dirty)
{
	pthread_mutex_lock(&mcache->lock);
	mcache->refcnt--;
	if (set_dirty)
		mcache->flags |= SHMSEG_FLAG_DIRTY_CACHE;
	pthread_mutex_unlock(&mcache->lock);
}

void shmseg_init(void)
{
	static int	segment_size;
	static int	cache_ratio;
	static bool	with_hugetlb;
	int			shmid;
	int			shmflags;
	uintptr_t	offset;
	int			mclass;
	int			index;

	/*
	 * Init mutex objects
	 */
	if (pthread_mutexattr_init(&msegment_mutex_attr) != 0 ||
		pthread_mutexattr_setpshared(&msegment_mutex_attr,
									 PTHREAD_PROCESS_SHARED))
		elog(ERROR, "failed on init mutex attribute");

	if (pthread_rwlockattr_init(&msegment_rwlock_attr) != 0 ||
		pthread_rwlockattr_setpshared(&msegment_rwlock_attr,
									  PTHREAD_PROCESS_SHARED))
		elog(ERROR, "failed on init rwlock attribute");

	DefineCustomIntVariable("pg_boost.segment_size",
							"Size of shared memory segment in MB",
							NULL,
							&segment_size,
							128,		/* 128MB */
							32,			/*  32MB */
							1024 * 1024,/*   1TB */
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	DefineCustomIntVariable("pg_boost.cache_ratio",
							"Ratio of cache size on shared memory segment",
							NULL,
							&segment_size,
							60,
							5,
							95,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	DefineCustomBoolVariable("pg_boost.with_hugetlb",
							 "True, if shared memory segment uses HugeTlb",
							 NULL,
							 &with_hugetlb,
							 true,
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/*
	 * Create and map shared memory segment according to the configured
	 * parameters. The shmctl(IPC_RMID) shall be invoked to ensure
	 * deletion of the shared memory segment after process crashes, but
	 * it is not destroyed as long as a process maps it at least.
	 */
	shmflags = 0600 | IPC_CREAT | IPC_EXCL;
	if (with_hugetlb)
		shmflags |= SHM_HUGETLB;

	shmid = shmget(IPC_PRIVATE, segment_size, shmflags);
	if (shmid < 0)
		elog(ERROR, "could not create a shared memory segment: %m");

	msegment = shmat(shmid, NULL, 0);

	shmctl(shmid, IPC_RMID, NULL);

	if (msegment == (void *)(-1))
		elog(ERROR, "could not attach a shared memory segment: %m");

	/*
	 * Set up msegment_t members
	 */
	msegment->shmid = shmid;
	msegment->segment_size = segment_size * 1024 * 1024;
	msegment->active_size = 0;
	msegment->free_size = 0;

	for (mclass = 0; mclass <= SHMSEG_CLASS_MAX_BITS; mclass++)
	{
		mlist_init(&msegment->free_list[mclass]);
		msegment->num_active[mclass] = 0;
		msegment->num_free[mclass] = 0;
	}

	offset = 1 << (fast_fls(sizeof(msegment_t)) + 1);
	if (offset < SHMSEG_CLASS_MIN_SIZE)
		offset = SHMSEG_CLASS_MIN_SIZE;

	while (msegment->segment_size - offset >= SHMSEG_CLASS_MIN_SIZE)
	{
		mchunk_free_t  *mfree;

		/* choose an appropriate chunk class */
		mclass = fast_ffs(offset);
		if (mclass > SHMSEG_CLASS_MAX_BITS)
			mclass = SHMSEG_CLASS_MAX_BITS;
		Assert(mclass >= SHMSEG_CLASS_MIN_BITS);

		/* if (offset + chunk_size) over the tail, truncate it */
		while (msegment->segment_size < offset + (1 << mclass))
			mclass--;

		if (mclass < SHMSEG_CLASS_MIN_BITS)
			break;

		mfree = (mchunk_free_t *)(((uintptr_t)msegment) + offset);
		mfree->c.mclass = mclass;
		mfree->c.is_active = false;
		mfree->c.is_cached = false;
		mlist_add(&msegment->free_list[mclass], &mfree->list);

		msegment->free_size += (1 << mclass);
		msegment->num_free[mclass]++;

		offset += (1 << mclass);
	}

	for (index = 0; index < SHMSEG_CACHE_NUM_SLOTS; index++)
	{
		mlist_init(&msegment->mcache_list[index]);
		shmseg_init_mutex(&msegment->mcache_lock[index]);
	}
	msegment->mcache_index = 0;
	msegment->mcache_reclaim = 0;
	msegment->mcache_size = 0;
	msegment->mcache_limit = msegment->segment_size * cache_ratio / 100;
}

void shmseg_exit(void)
{

}
