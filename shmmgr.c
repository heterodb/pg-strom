/*
 * shmmgr.c
 *
 * Routines to manage shared memory segment.
 *
 * Copyright (C) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "postgres.h"
#include "utils/pg_lzcompress.h"
#include "pg_boost.h"
#include <alloca.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>


#define MSEGMENT_CLASS_MIN_BITS		15		/* 32KBytes */
#define MSEGMENT_CLASS_MAX_BITS		30		/* 1GBytes */
#define MSEGMENT_CLASS_MIN_SIZE		(1 << MSEGMENT_CLASS_MIN_BITS)
#define MSEGMENT_CLASS_MAX_SIZE		(1 << MSEGMENT_CLASS_MAX_BITS)
#define MSEGMENT_NUM_CLASSES		\
	(MSEGMENT_CLASS_MAX_BITS - MSEGMENT_CLASS_MIN_BITS + 1)
#define MSEGMENT_NUM_BUFFER_SLOTS	32

struct msegment_s {
	int			shmid;
	uintptr_t	segment_size;
	uintptr_t	segment_usage;
	mlist_t		free_list[MSEGMENT_NUM_CLASSES];
	int			num_active[MSEGMENT_NUM_CLASSES];
	int			num_free[MSEGMENT_NUM_CLASSES];
	pthread_mutex_t	lock;

	mlist_t		mbuffer_list[MSEGMENT_NUM_BUFFER_SLOTS];
	pthread_mutex_t	mbuffer_lock[MSEGMENT_NUM_BUFFER_SLOTS];
	int			mbuffer_index;
	int			mbuffer_reclaim;
	uintptr_t	mbuffer_usage;
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

struct mslab_head_s {
	int				unit_size;
	int				unit_nums;
	int				num_active;
	int				num_free;
	mlist_t			full_list;
	mlist_t			free_list;
	pthread_mutex_t	lock;
	char			slabname[NAMEDATALEN];
};

#define MSLAB_BODY_MAX_UNITS		(1 << 9)
#define MSLAB_BODY_USEMAP_SIZE		\
	(MSLAB_BODY_MAX_UNITS / (sizeof(unsigned long) * 8))
struct mslab_body_s {
	mlist_t			list;
	unsigned long	usemap[MSLAB_BODY_USEMAP_SIZE];
	int				count;
	uintptr_t		body[0];
};

struct mbuffer_s {
	uintptr_t			buffer;		/* offset to the buffer chunk */
	mlist_t				list;
	uint32				length;
	uint16				index;
	bool				is_dirty;
	bool				is_compressed;
	time_t				reftimestamp;
	pthread_rwlock_t	lock;
};

/*
 * Global/Local Variables
 */
msegment_t			   *msegment = NULL;
static mslab_head_t	   *mslab_head_slab_head = NULL;
static mslab_head_t	   *mbuffer_slab_head = NULL;

static pthread_mutexattr_t  msegment_mutex_attr;
static pthread_rwlockattr_t msegment_rwlock_attr;

/*
 * Static functions
 */
static size_t	shmmgr_try_reclaim_buffer(mbuffer_t *mbuffer);
static void		shmmgr_reclaim_buffer(size_t required);
static void		shmmgr_load_buffer(mbuffer_t *mbuffer);
static void		shmmgr_init_slab(void);
static void		shmmgr_msegment_init(void);

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

/*
 * memory chunk - a buddy based memory allocation.
 *
 * A shared memory segment managed by msegment_t structure is divided into
 * (2^N) bytes region being called chunk. Several bytes of head of each
 * chunks saves its size and status. Rest of regions are used to store data.
 * At the initial state, all the chunks are free and chained to free_list[]
 * of msegment structure. When shmmgr_alloc_chunk() is called, it picks up
 * a chunk from suitable free_list. If no chunks were chained, it tries to
 * divide a larger chunk into two chunks. Furthermore, if no chunks are
 * available to divide any mode, it tries to reclaim buffered chunks
 * managed by mbuffer_t structure.
 * Conversely, when we release a chunk, shmmgr_free_chunk() tries to find
 * a buddy chunk to consolidate them into one larger chunk.
 * 
 * The minimum size of memory chunk is 32KBytes, so, it is not suitable to
 * allocate a small object using memory chunk. Use slab, instead of chunk,
 * in this case.
 */

/*
 * shmmgr_split_chunk
 *
 * It splits a chunk within the supplied mclass. If no chunk is available
 * to split, it tries to split larger chunk recursively.
 */
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

	index = mclass - MSEGMENT_CLASS_MIN_BITS;
    if (mlist_empty(&msegment->free_list[index]))
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

	mlist_del(&mchunk1->v.list);
	msegment->num_free[mclass]--;

	offset = addr_to_offset(mchunk1);
	mclass--;
	mchunk2 = offset_to_addr(offset + (1 << mclass));

	mchunk1->magic = mchunk2->magic = MCHUNK_MAGIC_NUMBER;
	mchunk1->mclass = mchunk2->mclass = mclass;
	mchunk1->is_active = mchunk2->is_active = false;

	index = mclass - MSEGMENT_CLASS_MIN_BITS;
	mlist_add(&msegment->free_list[index], &mchunk1->v.list);
	mlist_add(&msegment->free_list[index], &mchunk2->v.list);
	msegment->num_free[index] += 2;

	return true;
}

/*
 * shmmgr_(try_)alloc_chunk
 *
 * It allocate a memory chunk with enough size for the requirement.
 */
mchunk_t *
shmmgr_alloc_chunk(size_t size)
{
	void   *result = shmmgr_try_alloc_chunk(size);

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

	/*
	 * determine an appropriate mclass for the supplied size.
	 */
	mclass = fast_fls(size + offsetof(mchunk_t, v.data[0]) - 1);
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
				shmmgr_reclaim_buffer(1 << mclass);
				goto retry;
			}
			return NULL;
		}
		Assert(!mlist_empty(&msegment->free_list[index]));
	}
	/*
	 * Pick up a memory chunk from free_list, and switch its attribute
	 * from free to active chunk.
	 */
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

/*
 * shmmgr_free_chunk
 *
 * It free the given chunk.
 */
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
			  &mchunk->v.list);
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

/*
 * memory buffer 
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */



mbuffer_t *
shmmgr_alloc_buffer(size_t size)
{
	mbuffer_t  *mbuffer = shmmgr_try_alloc_buffer(size);

	if (!mbuffer)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("pg_boost: out of shared memory buffer")));
	return mbuffer;
}

mbuffer_t *
shmmgr_try_alloc_buffer(size_t size)
{
	mbuffer_t  *mbuffer;
	mchunk_t   *mchunk;
	int			index;

	mbuffer = shmmgr_try_alloc_slab(mbuffer_slab_head);

	mchunk = shmmgr_try_alloc_chunk(size);
	if (!mchunk)
	{
		shmmgr_free_slab(mbuffer_slab_head, mbuffer);
		return NULL;
	}

	mbuffer->buffer = addr_to_offset(mchunk);
	mbuffer->length = size;
	mbuffer->is_dirty = true;
	mbuffer->is_compressed = false;
	shmmgr_init_rwlock(&mbuffer->lock);

	index = __sync_fetch_and_add(&msegment->mbuffer_index, 1)
		& (MSEGMENT_NUM_BUFFER_SLOTS - 1);

	pthread_mutex_lock(&msegment->mbuffer_lock[index]);
	mlist_add(&msegment->mbuffer_list[index], &mbuffer->list);
	pthread_mutex_unlock(&msegment->mbuffer_lock[index]);

	__sync_add_and_fetch(&msegment->mbuffer_usage, (1 << mchunk->mclass));

	return mbuffer;
}

/*
 * shmmgr_free_buffer
 *
 * it releases the supplied buffer. Also note that this routine assumes
 * the caller ensures nobody references this buffer yet.
 */
void
shmmgr_free_buffer(mbuffer_t *mbuffer)
{
	pthread_rwlock_wrlock(&mbuffer->lock);

	if (mbuffer->buffer)
	{
		mchunk_t   *mchunk = offset_to_addr(mbuffer->buffer);

		pthread_mutex_lock(&msegment->mbuffer_lock[mbuffer->index]);
		mlist_del(&mbuffer->list);
		pthread_mutex_unlock(&msegment->mbuffer_lock[mbuffer->index]);

		__sync_sub_and_fetch(&msegment->mbuffer_usage,
							 (1 << mchunk->mclass));
		shmmgr_free_chunk(mchunk);
	}
	pthread_rwlock_unlock(&mbuffer->lock);

	shmmgr_free_slab(mbuffer_slab_head, mbuffer);
}

static size_t
shmmgr_try_reclaim_buffer(mbuffer_t *mbuffer)
{
	mchunk_t   *mchunk = offset_to_addr(mbuffer->buffer);
	int			mclass;

	Assert(mchunk != NULL);
	mclass = mchunk->mclass;

	if (mbuffer->is_dirty)
	{
		struct PGLZ_Header *temp;
		char	filename[1024];
		int		fdesc;

		temp = alloca(PGLZ_MAX_OUTPUT(mbuffer->length));
		if (!temp)
			elog(ERROR, "pg_boost: out of memory by alloca(3) : %m");

		snprintf(filename, sizeof(filename), "%s/d%06lx-%06lx",
				 guc_unbuffered_dir,
				 (addr_to_offset(mbuffer) >> 24) & 0x00ffffff,
				 (addr_to_offset(mbuffer)        & 0x00ffffff));

		fdesc = open(filename, O_WRONLY | O_CREAT| O_TRUNC | 0600);
		if (fdesc < 0)
			elog(ERROR, "pg_boost: failed to open(2) \"%s\" : %m", filename);

		PG_TRY();
		{
			char	   *wbuf;
			ssize_t		wlen, sz = 0;
			bool		compress;

			if (pglz_compress((char *)mchunk->v.data, mbuffer->length, temp,
							  PGLZ_strategy_default))
			{
				wlen = VARSIZE(temp);
				wbuf = (char *)temp;
				compress = true;
			}
			else
			{
				wlen = mbuffer->length;
				wbuf = (char *)mchunk->v.data;
				compress = false;
			}

			do {
				sz += write(fdesc, wbuf + sz, wlen - sz);
			} while (sz < wlen && errno != EINTR);

			if (sz < wlen)
				elog(ERROR, "pg_boost: failed to write(2) buffer");

			mbuffer->is_dirty = false;
			mbuffer->is_compressed = compress;
		}
		PG_CATCH();
		{
			close(fdesc);
			PG_RE_THROW();
		}
		PG_END_TRY();

		close(fdesc);
	}
	mlist_del(&mbuffer->list);
	__sync_sub_and_fetch(&msegment->mbuffer_usage,
						 (1 << mchunk->mclass));
	shmmgr_free_chunk(mchunk);
	mbuffer->buffer = 0;
	mbuffer->index = -1;

	return (1 << mclass);
}

static void
shmmgr_reclaim_buffer(size_t required)
{
	ssize_t		reclaimed = 0;
	time_t		threshold = 64;
	int			index;

	while (reclaimed < required)
	{
		struct timeval	tv;
		mbuffer_t	   *mbuffer, *temp;

		gettimeofday(&tv, NULL);

		index = __sync_fetch_and_add(&msegment->mbuffer_reclaim, 1)
			& (MSEGMENT_NUM_BUFFER_SLOTS - 1);

		pthread_mutex_lock(&msegment->mbuffer_lock[index]);

		mlist_foreach_entry_safe(mbuffer, temp,
								 &msegment->mbuffer_list[index], list)
		{
			if (pthread_rwlock_trywrlock(&mbuffer->lock) == 0 &&
				tv.tv_sec - mbuffer->reftimestamp >= threshold)
			{
				PG_TRY();
				{
					reclaimed += shmmgr_try_reclaim_buffer(mbuffer);
				}
				PG_CATCH();
				{
					pthread_rwlock_unlock(&mbuffer->lock);
					pthread_mutex_unlock(&msegment->mbuffer_lock[index]);
					PG_RE_THROW();
				}
				PG_END_TRY();
				pthread_rwlock_unlock(&mbuffer->lock);
			}
		}
		pthread_mutex_unlock(&msegment->mbuffer_lock[index]);

		threshold >>= 1;
	}
}

/*
 * shmmgr_load_buffer
 *
 * It loads contents of buffer from the storage.
 * Exclusive-lock must be held on the caller site.
 */
static void
shmmgr_load_buffer(mbuffer_t *mbuffer)
{
	mchunk_t   *mchunk = NULL;
	char		filename[1024];
	int			fdesc = -1;
	ssize_t		sz = 0;
	struct stat	stbuf;

	if (mbuffer->buffer)
		return;		/* nothing to do */

	PG_TRY();
	{
		mchunk = shmmgr_alloc_chunk(mbuffer->length);

		snprintf(filename, sizeof(filename), "%s/d%06lx-%06lx",
				 guc_unbuffered_dir,
				 (addr_to_offset(mbuffer) >> 24) & 0x00ffffff,
				 (addr_to_offset(mbuffer)        & 0x00ffffff));

		if ((fdesc = open(filename, O_RDONLY)) < 0)
			elog(ERROR, "pg_boost: failed to open(2) \"%s\" : %m", filename);
		if (fstat(fdesc, &stbuf) != 0)
			elog(ERROR, "pg_boost: failed to fstat(2) \"%s\" : %m", filename);

		if (mbuffer->is_compressed)
		{
			char   *filebuf = alloca(stbuf.st_size);

			if (!filebuf)
				elog(ERROR, "pg_boost: failed to alloca(%lu) : %m", stbuf.st_size);
			do {
				sz += read(fdesc, filebuf + sz, stbuf.st_size - sz);
			} while (sz < stbuf.st_size && errno != EINTR);

			if (sz < stbuf.st_size)
				elog(ERROR, "pg_boost: failed to read \"%s\" : %m", filename);

			pglz_decompress((PGLZ_Header *)filebuf, (char *)mchunk->v.data);
		}
		else
		{
			do {
				sz += read(fdesc,
						   ((char *)mchunk->v.data) + sz,
						   stbuf.st_size - sz);
			} while (sz < stbuf.st_size && errno != EINTR);

			if (sz < stbuf.st_size)
				elog(ERROR, "pg_boost: failed to read \"%s\" : %m", filename);
		}
		close(fdesc);

		mbuffer->buffer = addr_to_offset(mchunk);
		mbuffer->is_dirty = false;

		__sync_add_and_fetch(&msegment->mbuffer_usage,
							 (1 << mchunk->mclass));
	}
	PG_CATCH();
	{
		if (mchunk)
			shmmgr_free_chunk(mchunk);
		if (fdesc >= 0)
			close(fdesc);


		PG_RE_THROW();
	}
	PG_END_TRY();
}

void *
shmmgr_get_read_buffer(mbuffer_t *mbuffer)
{
	mchunk_t	   *mchunk;
	struct timeval	tv;

	pthread_rwlock_rdlock(&mbuffer->lock);
	while (!mbuffer->buffer)
	{
		/* Lock upgrade */
		pthread_rwlock_unlock(&mbuffer->lock);
		pthread_rwlock_wrlock(&mbuffer->lock);

		PG_TRY();
		{
			shmmgr_load_buffer(mbuffer);
		}
		PG_CATCH();
		{
			pthread_rwlock_unlock(&mbuffer->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();

		/* Lock downgrade */
		pthread_rwlock_unlock(&mbuffer->lock);
		pthread_rwlock_rdlock(&mbuffer->lock);
	}
	// XXX - Is it correct way to set a value in atomic?
	gettimeofday(&tv, NULL);
	__sync_lock_test_and_set(&mbuffer->reftimestamp, tv.tv_sec);

	mchunk = offset_to_addr(mbuffer->buffer);
	return (void *)mchunk->v.data;
}

void *
shmmgr_get_write_buffer(mbuffer_t *mbuffer)
{
	mchunk_t	   *mchunk;
	struct timeval	tv;

	pthread_rwlock_wrlock(&mbuffer->lock);
	PG_TRY();
	{
		shmmgr_load_buffer(mbuffer);
	}
	PG_CATCH();
	{
		pthread_rwlock_unlock(&mbuffer->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();

	gettimeofday(&tv, NULL);
	mbuffer->reftimestamp = tv.tv_sec;

	mchunk = offset_to_addr(mbuffer->buffer);
	return (void *)mchunk->v.data;
}

void
shmmgr_put_buffer(mbuffer_t *mbuffer, bool is_dirty)
{
	if (is_dirty)
		mbuffer->is_dirty = true;
	pthread_rwlock_unlock(&mbuffer->lock);
}

/*
 * memory slab - suitable for memory allocation of small object
 *
 * A slab is a small and fixed-length frag within a memory chunk.
 * We divide a chunk to some slabs and its management info.
 * 
 * mslab_body_t.usemap[] is a bitmap to inform what slabs are in use,
 * and not in use.
 *
 * +--------------------------+
 * | mchunk_t                 |
 * |                          |
 * | +------------------------+ <-- mchunk_t.v.data[0]
 * | | mslab_body_t           |
 * | |------------------------+
 * | | mlist_t       list     |
 * | | unsigned long usemap[] |
 * | | int           count    |
 * | |------------------------+ <-- mslab_body_t.body[0]
 * | | Slab[0]                |
 * | |------------------------+
 * | | Slab[1]                |
 * = =  :           ----------+
 * | | Slab[N]                |
 * +-+------------------------+
 */

/*
 * shmmgr_init_slab
 *
 *
 */
static void
shmmgr_init_slab(void)
{
	mchunk_t	   *mchunk;
	mslab_head_t   *mslabh;
	mslab_body_t   *mslabb;

	mchunk = shmmgr_alloc_chunk(MSEGMENT_CLASS_MIN_SIZE -
								offset_of(mchunk_t, v.data[0]));

	/* init first mslab_body_t */
	mslabb = (mslab_body_t *)mchunk->v.data;
	memset(mslabb->usemap, 0, sizeof(mslabb->usemap));
	mslabb->usemap[0] |= 1;
	mslabb->count = 1;

	/* init first mslab_head_t */
	mslabh = (mslab_head_t *)mslabb->body;

	mslabh->unit_size = MAXALIGN(sizeof(mslab_head_t));
	mslabh->unit_nums = (MSEGMENT_CLASS_MIN_SIZE -
						 offset_of(mchunk_t, v.data[0]) -
						 offset_of(mslab_body_t, body[0])) / mslabh->unit_size;
	mslabh->num_active = 1;
	mslabh->num_free = mslabh->unit_nums - 1;
	mlist_init(&mslabh->full_list);
	mlist_init(&mslabh->free_list);
	shmmgr_init_mutex(&mslabh->lock);
	snprintf(mslabh->slabname, sizeof(NAMEDATALEN), "mslab_head");

	mlist_add(&mslabh->free_list, &mslabb->list);

	mslab_head_slab_head = mslabh;
}

mslab_head_t *
shmmgr_create_slab(const char *slabname, int unit_size)
{
	mslab_head_t   *mslabh;
	int				unit_nums;

	unit_size = MAXALIGN(unit_size);
	unit_nums = (MSEGMENT_CLASS_MIN_SIZE -
				 offset_of(mchunk_t, v.data[0]) -
				 offset_of(mslab_body_t, body[0]));
	if (unit_nums == 0)
		elog(ERROR, "slab %s is too large (unit_size=%d)",
			 slabname, unit_size);
	if (unit_nums > MSLAB_BODY_MAX_UNITS)
		unit_nums = MSLAB_BODY_MAX_UNITS;

	if (strlen(slabname) >= NAMEDATALEN)
		elog(ERROR, "slab %s: name too long", slabname);

	mslabh = shmmgr_alloc_slab(mslab_head_slab_head);

	mslabh->unit_size = unit_size;
	mslabh->unit_nums = unit_nums;
	mslabh->num_active = 0;
	mslabh->num_free = 0;
	mlist_init(&mslabh->full_list);
	mlist_init(&mslabh->free_list);
	shmmgr_init_mutex(&mslabh->lock);

	return mslabh;
}

void *
shmmgr_alloc_slab(mslab_head_t *mslabh)
{
	void   *result = shmmgr_try_alloc_slab(mslabh);

	if (!result)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("pg_boost: failed to allocate a slab (%s)",
						mslabh->slabname)));
	return result;
}

void *
shmmgr_try_alloc_slab(mslab_head_t *mslabh)
{
	mchunk_t	   *mchunk;
	mslab_body_t   *mslabb;
	int				i, j, k;
	void		   *result = NULL;

	pthread_mutex_lock(&mslabh->lock);

	/*
	 * If we have no slab-body chunks with free slabs, we try to allocate
	 * a new chunk and construct a slab-body structure on this chunk.
	 * The number of free slabs shall be increased by slabh->unit_nums,
	 * then one of the free slabs shall be allocated to this invocation.
	 */
	if (mlist_empty(&mslabh->free_list))
	{
		size_t	length;

		length = offset_of(mslab_body_t, body[0]) +
			mslabh->unit_size * mslabh->unit_nums;
		Assert(length <= MSEGMENT_CLASS_MIN_SIZE -
						 offset_of(mchunk_t, v.data[0]));
		mchunk = shmmgr_try_alloc_chunk(length);
		if (!mchunk)
		{
			pthread_mutex_unlock(&mslabh->lock);
			return NULL;
		}

		mslabb = (mslab_body_t *)mchunk->v.data;

		mlist_init(&mslabb->list);
		memset(mslabb->usemap, 0, sizeof(mslabb->usemap));
		mslabb->count = 0;

		mslabh->num_free += mslabh->unit_nums;
		mlist_add(&mslabh->free_list, &mslabb->list);
	}
	Assert(!&mslabh->free_list);

	mslabb = list_entry(mslabh->free_list.next, mslab_body_t, list);
	Assert(mslabb->count < mslabh->unit_nums);

	/*
	 * Find a free slab within the slab-body picked up.
	 * A zero bit of usemap[] means corresponding slab is free now.
	 */
	for (i=0; i < MSLAB_BODY_USEMAP_SIZE; i++)
	{
		if (~mslabb->usemap[i])
		{
			j = fast_ffs(~mslabb->usemap[i]) - 1;
			k = i * sizeof(unsigned long) * 8 + j;
			Assert(k < mslabh->unit_nums);
			result = (void *)(((uintptr_t)mslabb->body) +
							  mslabh->unit_size * k);
			mslabb->usemap[i] |= (1 << j);
			mslabb->count++;

			/*
			 * In the case when it was the last free slab and allocated
			 * by this invocation, we relocate this slab-body from
			 * free_list into full_list.
			 */
			if (mslabb->count == mslabh->unit_nums)
			{
				mlist_del(&mslabb->list);
				mlist_add(&mslabh->full_list, &mslabb->list);
			}
			mslabh->num_active++;
			mslabh->num_free--;
			break;
		}
	}
	pthread_mutex_unlock(&mslabh->lock);

	/* a slab-body in full_list should have a free slab at least */
	Assert(result != NULL);

	return result;
}

void
shmmgr_free_slab(mslab_head_t *mslabh, void *ptr)
{
	mchunk_t	   *mchunk;
	mslab_body_t   *mslabb;
	int				i, j, k;

	pthread_mutex_lock(&mslabh->lock);

	mchunk = offset_to_addr(addr_to_offset(ptr) &
							~(MSEGMENT_CLASS_MIN_SIZE - 1));
	mslabb = (mslab_body_t *)mchunk->v.data;
	k = ((uintptr_t)(ptr) - (uintptr_t)(mslabb->body)) / mslabh->unit_size;
	j = k % (sizeof(unsigned long) * 8);
	i = k / (sizeof(unsigned long) * 8);

	Assert(i < MSLAB_BODY_USEMAP_SIZE);
	Assert((mslabb->usemap[i] & (1 << j)) != 0);

	/*
	 * If the supplied ptr points to the last element of the mslab_body_t,
	 * the mchunk_t owning this mslab_body_t shall be released instead of
	 * reusing; by chaining with mslab_head_t.free_list.
	 */
	if (mslabb->count == 1)
	{
		mslabh->num_active--;
		mslabh->num_free -= mslabh->unit_nums - 1;
		mlist_del(&mslabb->list);
		shmmgr_free_chunk(mchunk);

		goto out_unlock;
	}

	/*
	 * In the case when mslab_body_t didn't have any free slab, it should
	 * have been chained to mslab_head_t.full_list. This release of slab
	 * enables to provides a free slot to the next invocation, so we
	 * re-chain this mslab_body_t to the free_list.
	 */
	if (mslabb->count == mslabh->unit_nums)
	{
		mlist_del(&mslabb->list);
		mlist_add(&mslabh->free_list, &mslabb->list);
	}
	mslabb->usemap[i] &= ~(1 << j);
	mslabb->count--;

	mslabh->num_active--;
	mslabh->num_free++;

out_unlock:	
	pthread_mutex_unlock(&mslabh->lock);
}

/*
 * shmmgr_init_mutex
 *
 * It initializes an exclusive lock on shared memory segment
 */
void
shmmgr_init_mutex(pthread_mutex_t *lock)
{
	if (pthread_mutex_init(lock, &msegment_mutex_attr) != 0)
		elog(ERROR, "Failed to initialize mutex object : %m");
}

/*
 * shmmgr_init_rwlock
 *
 * It initializes a read-write lock on shared memory segment
 */
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

		mlist_add(&msegment->free_list[index], &mchunk->v.list);
		msegment->num_free[index]++;

		offset += (1 << mclass);
	}

	/*
	 * init memory buffer stuff
	 */
	for (index=0; index < MSEGMENT_NUM_BUFFER_SLOTS; index++)
	{
		mlist_init(&msegment->mbuffer_list[index]);
		shmmgr_init_mutex(&msegment->mbuffer_lock[index]);
	}
	msegment->mbuffer_index = 0;
	msegment->mbuffer_reclaim = 0;
	msegment->mbuffer_usage = 0;
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
	 * Init shared memory segment, slab and buffer
	 */
	shmmgr_msegment_init();
	shmmgr_init_slab();
	mbuffer_slab_head = shmmgr_create_slab("mbuffer", sizeof(mbuffer_t));
}
