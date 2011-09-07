/*
 * shmmgr.c
 *
 * Routines to manage shared memory segment.
 *
 * Copyright (C) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "postgres.h"
#include "pg_boost.h"
#include "shmlist.h"



#define MSEGMENT_CLASS_MIN_BITS		15		/* 32KBytes */
#define MSEGMENT_CLASS_MAX_BITS		30		/* 1GBytes */
#define MSEGMENT_CLASS_MIN_SIZE		(1 << MSEGMENT_CLASS_MIN_BITS)
#define MSEGMENT_CLASS_MAX_SIZE		(1 << MSEGMENT_CLASS_MAX_BITS)
#define MSEGMENT_NUM_CLASSES		\
	(MSEGMENT_CLASS_MAX_BITS - MSEGMENT_CLASS_MIN_BITS + 1)

struct msegment_s {
	int			shmid;
	uintptr_t	segment_size;
	uintptr_t	segment_usage;
	mlist_t		free_list[MSEGMENT_NUM_CLASSES];
	int			num_active[MSEGMENT_NUM_CLASSES];
	int			num_free[MSEGMENT_NUM_CLASSES];
	pthread_mutex_t	lock;
};

#define MCHUNK_MAGIC_NUMBER			0xbeaf
struct mchunk_s {
	uint16			magic;
	uint8			mclass;
	bool			is_active;
	union {
		mlist_t		list;		/* if free chunk */
		uintptr_t	data[0];	/* if active chunk */
	} v;
};

struct mbuffer_s {
	uintptr_t			buffer;		/* offset to the buffer chunk */
	mlist_t				list;
	uint32				length;
	pthread_rwlock_t	lock;
};

/*
 * Global/Local Variables
 */
msegment_t *msegment = NULL;

static pthread_mutexattr_t  msegment_mutex_attr;
static pthread_rwlockattr_t msegment_rwlock_attr;

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
    return sizeof(value) * 8 - __builtin_clzl(value);
}

static bool
shmmgr_split_chunk(int mclass)
{
	mchunk_t   *mchunk1;
	mchunk_t   *mchunk2;
	mlist_t	   *mlist;
	uintptr_t	offset;
	int			index;

    Assert(mclass > MSEGMENT_CLASS_MIN_BITS &&
           mclass <= MSEGMENT_CLASS_MAX_BITS);

    if (mlist_empty(&msegment->free_list[mclass - MSEGMENT_CLASS_MIN_BITS]))
    {
        if (mclass == MSEGMENT_CLASS_MAX_BITS)
            return false;
		else if (!shmmgr_split_chunk(mclass + 1))
			return false;
	}
    mlist = offset_to_addr(msegment->free_list[index].next);
    mchunk1 = container_of(mlist, mchunk_t, v.list);
	Assert(mchunk1->magic == MCHUNK_MAGIC_NUMBER);
    Assert(mchunk1->mclass == mclass);

	mlist_del(&mchunk1->list);
	msegment->num_free[mclass]--;

	offset = addr_to_offset(mchunk1);
	mclass--;
	mchunk2 = offset_to_addr(offset + (1 << mclass));

	mchunk1->magic = mchunk2->magic = MCHUNK_MAGIC_NUMBER;
	mchunk1->mclass = mchunk2->mclass = mclass;
	mchunk1->is_active = mchunk2->is_active = false;

	index = mclass - MSEGMENT_CLASS_MIN_BITS;
	mlist_add(&msegment->free_list[index], &mchunk1->list);
	mlist_add(&msegment->free_list[index], &mchunk2->list);
	msegment->num_free[index] += 2;

	return true;
}

mchunk_t *
shmmgr_alloc_chunk(size_t size)
{
	void   *result = shmseg_try_alloc(size);

	if (!result)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("pg_boost: out of shared memory")));
	return result;
}

mchunk_t *
shmmgr_try_alloc_chunk(size_t size)
{
	mchunk_t   *mchunk;
	mlist_t	   *mlist;
	int			mclass;
	int			index;
	int			num_retry = 4;

	mclass = fast_fls(size + offsetof(mchunk_t, v.data) - 1);
	if (mclass > MSEGMENT_CLASS_MAX_BITS)
		return NULL;
	if (mclass < MSEGMENT_CLASS_MIN_BITS)
		mclass = MSEGMENT_CLASS_MIN_BITS;
	index = mclass - MSEGMENT_CLASS_MIN_BITS;

retry:
	pthread_mutex_lock(&msegment->lock);

	if (mlist_empty(&msegment->free_list[index]))
	{
		if (!shmmgr_split_chunk(mclass + 1))
		{
			pthread_mutex_unlock(&msegment->lock);
			if (num_retry-- > 0)
			{
				//shmmgr_reclaim_buffer(mclass);
				goto retry;
			}
			return NULL;
		}
		Assert(!mlist_empty(&msegment->free_list[index]));
	}

	mlist = offset_to_addr(msegment->free_list[index].next);
	mchunk = (mchunk_t *)container_of(mlist, mchunk_t, v.list);
	mlist_del(mlist);

	Assert(mchunk->magic == MCHUNK_MAGIC_NUMBER);
	Assert(mchunk->mclass == mclass);

	mchunk->is_active = true;

	msegment->num_free[index]--;
	msegment->num_active[index]++;
	msegment->segment_usage += (1 << mclass);

	pthread_mutex_unlock(&msegment->lock);

	return (void *)mchunk->v.data;
}

void
shmmgr_free_chunk(mchunk_t *mchunk)
{
	mchunk_t   *buddy;
	uintptr_t	offset_mchunk;
	uintptr_t	offset_buddy;
	int			mclass;

	mclass = mchunk->mclass;
	Assert(mchunk->is_active);

	pthread_mutex_lock(&msegment->lock);

	mchunk->is_active = false;
	msegment->num_active[mclass - MSEGMENT_CLASS_MIN_BITS]--;
	msegment->segment_usage -= (1 << mclass);

	/*
	 * If its buddy is also free and same class, it shall be
	 * consolidated into one larger free chunk.
	 */
	offset_mchunk = addr_to_offset(mchunk);

	while (mclass < MSEGMENT_CLASS_MAX_BITS)
	{
		if (offset_mchunk & (1 << mclass))
			offset_buddy = (offset_mchunk & ~(1 << mclass));
		else
			offset_buddy = (offset_mchunk | (1 << mclass));
		Assert((offset_buddy & ((1 << mclass) - 1)) == 0);

		/*
		 * buddy must exist within the shared memory segment
		 */
		if (offset_buddy < MSEGMENT_CLASS_MIN_SIZE ||
			offset_buddy + (1 << mclass) > msegment->segment_size)
			break;

		/*
		 * Also free? and same class?
		 */
		buddy = offset_to_addr(offset_buddy);
		Assert(buddy->magic == MCHUNK_MAGIC_NUMBER);
		if (buddy->mclass != mclass || buddy->is_active)
			break;

		/*
		 * Consolidate them
		 */
		mlist_del(&buddy->v.list);
		msegment->num_free[mclass - MSEGMENT_CLASS_MIN_BITS]--;

		mclass++;
		offset_mchunk &= ~((1 << mclass) - 1);
		mchunk = offset_to_addr(offset_mchunk);
		mchunk->magic = MCHUNK_MAGIC_NUMBER;
		mchunk->mclass = mclass;
		mchunk->is_active = false;
	}

	/*
	 * Attach this chunk to msegment->free_list[]
	 */
	mlist_add(&msegment->free_list[mclass - MSEGMENT_CLASS_MIN_BITS],
			  &mchunk->list);
	msegment->num_free[mclass - MSEGMENT_CLASS_MIN_BITS]++;

	pthread_mutex_unlock(&msegment->lock);
}

void *
shmmgr_get_chunk_data(mchunk_t *mchunk)
{
	Assert(mchunk->magic == MCHUNK_MAGIC_NUMBER);
	Assert(mchunk->is_active);
	return (void *)mchunk->v.data;
}

int
shmmgr_get_chunk_class(mchunk_t *mchunk)
{
	Assert(mchunk->magic == MCHUNK_MAGIC_NUMBER);
	return mchunk->mclass;
}

mbuffer_t *
shmmgr_alloc_buffer(size_t size)
{}

mbuffer_t *
shmmgr_try_alloc_buffer(sizez_t size)
{}

void
shmmgr_free_buffer(mbuffer_t *mbuffer)
{}

void *
shmmgr_get_read_buffer(mbuffer_t *mbuffer)
{}

void *
shmmgr_get_write_buffer(mbuffer_t *mbuffer)
{}

void
shmmgr_put_buffer(mbuffer_t *mbuffer, bool is_dirty)
{}

mslab_t *
shmmgr_create_slab(size_t unit_size)
{}

void
shmmgr_destroy_slab(mslab_t *mslab)
{}

void *
shmmgr_alloc_slab(mslab_t *mslab)
{}

void
shmmgr_free_slab(mslab_t *mslab, void *ptr)
{}

void
shmmgr_init_mutex(pthread_mutex_t *lock)
{
	if (pthread_mutex_init(lock, &msegment_mutex_attr) != 0)
		elog(ERROR, "Failed to initialize mutex object : %m");
}

void
shmmgr_init_rwlock(pthread_rwlock_t *lock)
{
	if (pthread_rwlock_init(lock, &msegment_rwlock_attr) != 0)
		elog(ERROR, "Failed to initialize rwlock object : %m");
}

static void
shmmgr_msegment_init(void)
{
	size_t		segment_size = (guc_segment_size << 20);
	uintptr_t	offset;
	int			shmid;
	int			shmflags;
	int			mclass;
	int			index;

	/*
	 * Create and map shared memory segment according to the configured
	 * parameters. The shmctl(IPC_RMID) shall be invoked to ensure
	 * deletion of the shared memory segment after process crashes, but
	 * it is not destroyed as long as a process maps it at least.
	 */
	shmflags = 0600 | IPC_CREAT | IPC_EXCL;
	if (guc_with_hugetlb)
		shmflags |= SHM_HUGETLB;

	shmid = shmget(IPC_PRIVATE, segment_size, shmflags);
	if (shmid < 0)
		elog(ERROR, "could not create a shared memory segment: %m");

	msegment = shmat(shmid, NULL, 0);

	shmctl(shmid, IPC_RMID, NULL);

	if (msegment == (void *)(-1))
		elog(ERROR, "could not attach a shared memory segment: %m");

	msegment->shmid = shmid;
	msegment->segment_size = segment_size;
	msegment->segment_usage = 0;
	for (index = 0; index < MSEGMENT_NUM_CLASSES; index++)
	{
		mlist_init(&msegment->free_list[index]);
		msegment->num_active[index] = 0;
		msegment->num_free[index] = 0;
	}
	shmmgr_init_mutex(&msegment->lock);

	offset = MSEGMENT_CLASS_MIN_SIZE;
	Assert(sizeof(msegment_t) < MSEGMENT_CLASS_MIN_SIZE);

	while (segment_size >= offset + MSEGMENT_CLASS_MIN_SIZE)
	{
		mchunk_t   *mchunk;

		/* choose an appropriate mclass of the chunk */
		mclass = fast_ffs(offset) - 1;
		if (mclass >= MSEGMENT_CLASS_MAX_BITS)
			mclass = MSEGMENT_CLASS_MAX_BITS;
		Assert(mclass >= MSEGMENT_CLASS_MIN_BITS);

		/* if (offset + chunk size) over the tail, truncate it */
		while (segment_size < offset + (1 << mclass))
			mclass--;
		if (mclass < MSEGMENT_CLASS_MIN_BITS)
			break;

		/* chain this chunk to msegment->free_list */
		index = mclass - MSEGMENT_CLASS_MIN_BITS;

		mchunk = (mchunk_t *)offset_to_addr(offset);
		mchunk->magic = MCHUNK_MAGIC_NUMBER;
		mchunk->mclass = mclass;
		mchunk->is_active = false;

		mlist_add(&msegment->free_list[index], &mfree->list);
		msegment->num_free[index]++;

		offset += (1 << mclass);
	}






}

void
shmmgr_init(void)
{
	/*
	 * Init mutex attributes
	 */
	if (pthread_mutexattr_init(&msegment_mutex_attr) != 0 ||
		pthread_mutexattr_setpshared(&msegment_mutex_attr,
									 PTHREAD_PROCESS_SHARED))
		elog(ERROR, "failed on init mutex attribute");

	if (pthread_rwlockattr_init(&msegment_rwlock_attr) != 0 ||
		pthread_rwlockattr_setpshared(&msegment_rwlock_attr,
									  PTHREAD_PROCESS_SHARED))
		elog(ERROR, "failed on init rwlock attribute");

	/*
	 * Init shared memory segment
	 */
	shmmgr_msegment_init();




}
