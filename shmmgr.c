/*
 * shmmgr.c
 *
 * Routines to manage shared memory segment
 *
 * Copyright (c) 2011, KaiGai Kohei
 */
#include "pg_boost.h"



/*
 * shmchunk_t - memory chunk of shared memory segment
 */
#define SHMBUFFER_FLAG_DIRTY_CACHE	0x01
#define SHMBUFFER_HOT_CACHE			0x02
#define SHMBUFFER_COMPRESSED		0x04
#define SHMBUFFER_NUM_ACTIVE_SLOTS	64
typedef struct {
	uint16				mclass;
	bool				is_free;
	bool				is_buffer;
	union {
		struct {
			mlist_t		list;
		} free;		/* is_free == true */
		struct {
			uint8		data[0];
		} item;
		struct {	/* is_free == false && is_buffer == false */
			uintptr_t	storage;
			uintptr_t	cached;
			uint32		length;
			uint32		refcnt;
			mlist_t		list;
			uint16		index;
			uint8		tag;
			uint8		flags;
		} buffer;	/* is_free == false && is_buffer == true */
	};
} shmchunk_t;

/*
 * Shared memory segment header
 */
#define SHMCLASS_MIN_BITS		5	/* 32bytes */
#define SHMCLASS_MAX_BITS		30	/* 1Gbytes */
#define SHMCLASS_MIN_SIZE		(1 << SHMCLASS_MIN_BITS)
#define SHMCLASS_MAX_SIZE		(1 << SHMCLASS_MAX_BITS)

typedef struct {
	int			shmid;
	int			index;
	size_t		segment_size;
	mlist_t		free_list[SHMCLASS_MAX_BITS + 1];
	int			num_active[SHMCLASS_MAX_BITS + 1];
	int			num_free[SHMCLASS_MAX_BITS + 1];
	pthread_mutex_t	lock;

	/* buffering support */
	struct {
		mlist_t		active_list[SHMBUFFER_NUM_ACTIVE_SLOTS];
		pthread_mutex_t	active_lock[SHMBUFFER_NUM_ACTIVE_SLOTS];
		int			buffer_hint;
		int			reclaim_hint;
		size_t		total_size;
		size_t		limit_size;
	} buffer;
} shmsegment_t;

#ifdef	SIZEOF_VOID_P == 8
#define SHMSEGMENT_NUM_MAX_BITS		12		/* 4096 */
#define SHMSEGMENT_SIZE_MAX_BITS	36		/* 64GB */
#elseif SIZEOF_VOID_P == 4
#define SHMSEGMENT_NUM_MAX_BITS		0		/* No multisegment support */
#define SHMSEGMENT_SIZE_MAX_BITS	32		/* 4 GB */
#endif
#define SHMSEGMENT_NUM_MAX			(1 << SHMSEGMENT_NUM_MAX_BITS)
#define SHMSEGMENT_SIZE_MAX			(1 << SHMSEGMENT_SIZE_MAX)
#define shmptr_to_segment(p)		(((p) >> SHMSEGMENT_SIZE_MAX_BITS) & (SHMSEGMENT_SIZE_MAX - 1)
#define shmptr_to_offset(p)			((p) & (SHMSEGMENT_SIZE_MAX - 1))

static msegment_t **msegments;
static int			msegments_num;




extern uintptr_t	shmmgr_alloc(size_t size, bool is_buffer);
extern uintptr_t	shmmgr_try_alloc(size_t size, bool is_buffer);
extern void			shmmgr_free(uintptr_t ptr);
extern void			shmmgr_init_mutex(pthread_mutex_t *lock);
extern void			shmmgr_init_rwlock(pthread_rwlock_t *lock);
extern void		   *shmmgr_get_addr(uintptr_t ptr);
extern void			shmmgr_put_addr(uintptr_t ptr);
extern void			shmmgr_get_size(uintptr_t ptr);
extern void			shmmgr_set_dirty(uintptr_t ptr);
extern void			shmmgr_init(int nsegments, size_t segment_size);
extern void			shmmgr_exit(void);

/*
 * shmmgr_init
 *
 *
 */
static void
shmmgr_init_one(msegment_t *msegment)
{
	int			mclass;
	shmptr_t	offset;

	for (mclass = 0; mclass <= SHMCLASS_MAX_BITS; mclass++)
	{
		shmlist_init(&msegment->free_list[mclass]);
		msegment->num_free[mclass] = 0;
		msegment->num_active[mclass] = 0;
	}

	offset = 1 << (fls64(sizeof(shmhead_t)) + 1);
	if (offset < SHMCLASS_MIN_SIZE)
		offset = SHMCLASS_MIN_SIZE;

	while (msegment->segment_size - offset >= SHMCLASS_MIN_SIZE)
	{
		shmchunk_t *chunk;

		/* choose an appropriate chunk class */
		mclass = ffs(offset) - 1;
		if (mclass > SHMCLASS_MAX_BITS)
			mclass = SHMCLASS_MAX_BITS;
		Assert(mclass >= SHMCLASS_MIN_BITS);

		/* if (offset + chunk_size) over the tail, truncate it */
		while (msegment->segment_size < offset + (1 << mclass))
			mclass--;

		if (mclass < SHMCLASS_MIN_BITS)
			break;

		/* chain this chunk to free_list */
		chunk = (shmchunk_t *)((uintptr_t)(msegment) + offset);
		chunk->mclass = mclass;
		chunk->is_free = true;

		shmlist_add(&msegment->free_list[mclass], &chunk->list);
		msegment->num_free[mclass]++;

		offset += (1 << mclass);
	}








	}
	










}

void
shmmgr_init(int nsegments, size_t segment_size)
{
	int		shmid;
	int		shmflags = 0600 | IPC_CREAT | IPC_EXCL;
	int		index;

	msegments = malloc(sizeof(msegment_t *) * nsegments);
	if (!msegments)
		elog(ERROR, "out of memory");

	for (index = 0; index < nsegments; index++)
	{
		shmid = shmget(IPC_PRIVATE, size, shmflag);
		if (shmid < 0)
			goto error;

		msegments[index] = shmat(shmid, NULL, 0);

		/*
		 * To make sure the shared memory segment being released
		 * when this process exit. If shmat(2) was failed, this
		 * segment shall be removed soon, because nobody maps it.
		 */
		shmctl(shmid, IPC_RMID, NULL);

		if (shmhead == (void *)(-1))
			goto error;

		msegments[index].shmid = shmid;
		msegments[index].index = index;
		msegments[index].segment_size = segment_size;
		shmmgr_init_one(msegments[index]);
	}
	return;

error:
	while (--index >= 0)
		shmdt(msegments[index]);

	elog(ERROR,
		 (errcode(ERRCODE_OUT_OF_MEMORY),
		  errmsg("could not map shared memory segment")));
}

/*
 * shmmgr_exit
 *
 */
void
shmmgr_exit(void)
{


}




#if 0
/*
 * shmmgr.c - shared memory management
 *
 * Copyright (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 *
 */
#include "postgres.h"
#include "pg_boost.h"
#include <assert.h>
#include <inttypes.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

/*
 * Shared memory chunk
 */
typedef struct {
	uint8		mclass;		/* power of chunk size */
	bool		active;		/* true, if active chunk */
	shmlist_t	list;
} shmchunk_t;

/*
 * Shared memory segment header
 */
#define SHMCLASS_MIN_BITS	6		/* 64bytes */
#define SHMCLASS_MAX_BITS	31		/* 2Gbytes */
#define SHMCLASS_MIN_SIZE	(1 << SHMCLASS_MIN_BITS)
#define SHMCLASS_MAX_SIZE	(1 << SHMCLASS_MAX_BITS)

typedef struct {
	int			shmid;
	offset_t	segment_size;
	shmlist_t	free_list[SHMCLASS_MAX_BITS + 1];
	int			num_active[SHMCLASS_MAX_BITS + 1];
	int			num_free[SHMCLASS_MAX_BITS + 1];
	pthread_mutex_t	lock;

	offset_t	shmbuf_head;	/* superblock of shared buffer */
} shmhead_t;

static shmhead_t   *shmhead = NULL;

/*
 * ffs64 - returns first (smallest) bit of the value
 */
static inline int ffs64(uint64 value)
{
	int		ret = 1;

	if (!value)
		return 0;
	if (!(value & 0xffffffff))
	{
		value >>= 32;
		ret += 32;
	}
	if (!(value & 0x0000ffff))
	{
		value >>= 16;
		ret += 16;
	}
	if (!(value & 0x000000ff))
	{
		value >>= 8;
		ret += 8;
	}
	if (!(value & 0x0000000f))
	{
		value >>= 4;
		ret += 4;
	}
	if (!(value & 0x00000003))
	{
		value >>= 2;
		ret += 2;
	}
	if (!(value & 0x00000001))
	{
		value >>= 1;
		ret += 1;
	}
	return ret;
}


/*
 * fls64 - returns last (biggest) bit of the value
 */
static inline int fls64(uint64 value)
{
	int		ret = 1;

	if (!value)
		return 0;
	if (value & 0xffffffff00000000)
	{
		value >>= 32;
		ret += 32;
	}
	if (value & 0xffff0000)
	{
		value >>= 16;
		ret += 16;
	}
	if (value & 0xff00)
	{
		value >>= 8;
		ret += 8;
	}
	if (value & 0xf0)
	{
		value >>= 4;
		ret += 4;
	}
	if (value & 0xc)
	{
		value >>= 2;
		ret += 2;
	}
	if (value & 0x2)
	{
		value >>= 1;
		ret += 1;
	}
	return ret;
}

/*
 * addr_to_offset - Translation from an address to offset
 */
offset_t
addr_to_offset(void *addr)
{
	assert(shmhead != NULL);

	if (!addr)
		return 0;

	return (offset_t)((unsigned long)(addr) - (unsigned long)(shmhead));
}

/*
 * offset_to_addr - Translation from an offset to address
 */
void *
offset_to_addr(offset_t offset)
{
	assert(shmhead != NULL);

	if (offset == 0)
		return NULL;
	return (void *)((uint64_t)shmhead + offset);
}

/*
 * shmlist_empty - check whether the list is empty, or not
 */
bool
shmlist_empty(shmlist_t *list)
{
	return offset_to_addr(list->next) == list;
}

/*
 * shmlist_init - initialize the list as an empty list
 */
void
shmlist_init(shmlist_t *list)
{
	list->next = list->prev = addr_to_offset(list);
}

/*
 * shmlist_add - add an element to the base list
 */
void
shmlist_add(shmlist_t *base, shmlist_t *list)
{
	shmlist_t  *nlist = offset_to_addr(base->next);

	base->next = addr_to_offset(list);
	list->prev = addr_to_offset(base);
	list->next = addr_to_offset(nlist);
	nlist->prev = addr_to_offset(list);
}

/*
 * shmlist_del - delete an element from the list
 */
void
shmlist_del(shmlist_t *list)
{
	shmlist_t  *plist = offset_to_addr(list->prev);
	shmlist_t  *nlist = offset_to_addr(list->next);

	plist->next = addr_to_offset(nlist);
	nlist->prev = addr_to_offset(plist);

	shmlist_init(list);
}

/*
 * shmmgr_init_mutex - initialize mutex on shared memory segment
 */
bool
shmmgr_init_mutex(pthread_mutex_t *lock)
{
	pthread_mutexattr_t	attr;

	if (pthread_mutexattr_init(&attr) != 0)
		return false;
	if (pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) != 0)
		goto error;
	if (pthread_mutex_init(lock, &attr) != 0)
		goto error;
	pthread_mutexattr_destroy(&attr);
	return true;

 error:
	pthread_mutexattr_destroy(&attr);
	return false;
}

/*
 * shmmgr_init_rwlock - initialize rwlock on shared memory segment
 */
bool
shmmgr_init_rwlock(pthread_rwlock_t *lock)
{
	pthread_rwlockattr_t	attr;

	if (pthread_rwlockattr_init(&attr) != 0)
		return false;
	if (pthread_rwlockattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) != 0)
		goto error;
	if (pthread_rwlock_init(lock, &attr) != 0)
		goto error;
	pthread_rwlockattr_destroy(&attr);
	return true;

 error:
	pthread_rwlockattr_destroy(&attr);
	return false;
}

/*
 * shmmgr_split_chunk - split a large chunk into two half size
 */
static bool
shmmgr_split_chunk(int mclass)
{
	shmchunk_t *chunk1;
	shmchunk_t *chunk2;
	shmlist_t  *list;
	offset_t	offset;

	assert(mclass > SHMCLASS_MIN_BITS && mclass <= SHMCLASS_MAX_BITS);

	if (shmlist_empty(&shmhead->free_list[mclass]))
	{
		if (mclass == SHMCLASS_MAX_BITS)
			return false;
		else if (!shmmgr_split_chunk(mclass + 1))
			return false;
	}
	list = offset_to_addr(shmhead->free_list[mclass].next);
	chunk1 = container_of(list, shmchunk_t, list);
	assert(chunk1->mclass == mclass);

	shmlist_del(&chunk1->list);
	shmhead->num_free[mclass]--;

	offset = addr_to_offset(chunk1);
	mclass--;
	chunk2 = offset_to_addr(offset + (1 << mclass));

	chunk1->mclass = chunk2->mclass = mclass;
	chunk1->active = chunk2->active = false;

	shmlist_add(&shmhead->free_list[mclass], &chunk1->list);
	shmlist_add(&shmhead->free_list[mclass], &chunk2->list);
	shmhead->num_free[mclass] += 2;

	return true;
}

/*
 * shmmgr_alloc - allocate a memory chunk on the shared memory segment
 */
void *
shmmgr_alloc(size_t size)
{
	shmchunk_t *chunk;
	shmlist_t  *list;
	int			mclass;

	mclass = fls64(size + offset_of(shmchunk_t, list) - 1);
	if (mclass > SHMCLASS_MAX_BITS)
		return NULL;
	if (mclass < SHMCLASS_MIN_BITS)
		mclass = SHMCLASS_MIN_BITS;

	pthread_mutex_lock(&shmhead->lock);

	/*
	 * when free_list of the mclass is not available, it tries to split
	 * a larger free chunk into two. If unavailable anymore, we cannot
	 * allocate a new free chunk.
	 */
	if (shmlist_empty(&shmhead->free_list[mclass]))
	{
		if (!shmmgr_split_chunk(mclass + 1))
		{
			pthread_mutex_unlock(&shmhead->lock);
			return NULL;
		}
	}
	assert(!shmlist_empty(&shmhead->free_list[mclass]));

	list = offset_to_addr(shmhead->free_list[mclass].next);
	chunk = container_of(list, shmchunk_t, list);
	assert(chunk->mclass == mclass);

	shmlist_del(&chunk->list);
	shmhead->num_free[mclass]--;
	shmhead->num_active[mclass]++;

	pthread_mutex_unlock(&shmhead->lock);

	return (void *)&chunk->list;
}

/*
 * shmmgr_free - free a memory chunk on the shared memory segment
 */
void
shmmgr_free(void *ptr)
{
	shmchunk_t *chunk = container_of(ptr, shmchunk_t, list);
	shmchunk_t *buddy;
	offset_t	offset;
	offset_t	offset_buddy;
	int			mclass = chunk->mclass;

	pthread_mutex_lock(&shmhead->lock);

	chunk->active = false;
	shmhead->num_active[mclass]--;

	/*
	 * If its buddy is also free, we consolidate them into one.
	 */
	offset = addr_to_offset(chunk);

	while (mclass < SHMCLASS_MAX_BITS)
	{
		if (offset & (1 << mclass))
			offset_buddy = offset & ~(1 << mclass);
		else
			offset_buddy = offset | (1 << mclass);

		/* offset should not be within the shmhead structure */
		if (offset_buddy < sizeof(shmhead_t))
			break;
		buddy = offset_to_addr(offset_buddy);

		/*
		 * If buddy is also free and same size, we consolidate them
		 */
		if (buddy->active || buddy->mclass != mclass)
			break;

		shmlist_del(&buddy->list);
		shmhead->num_free[mclass]--;

		mclass++;
		offset &= ~((1 << mclass) - 1);
		chunk = offset_to_addr(offset);

		chunk->mclass = mclass;
		chunk->active = false;
	}
	/*
	 * Attach this mchunk on the freelist[mclass]
	 */
	shmlist_add(&shmhead->free_list[mclass], &chunk->list);
	shmhead->num_free[mclass]++;

	pthread_mutex_unlock(&shmhead->lock);
}

void *
shmmgr_get_bufmgr_head(void)
{
	return offset_to_addr(shmhead->shmbuf_head);
}

size_t
shmmgr_get_size(void *ptr)
{
	shmchunk_t *chunk = container_of(ptr, shmchunk_t, list);

	return (1 << chunk->mclass);
}

bool
shmmgr_init(size_t size, bool hugetlb)
{
	int			shmid;
	int			shmflag = 0600 | IPC_CREAT | IPC_EXCL;
	int			mclass;
	offset_t	offset;

	if (hugetlb)
		shmflag |= SHM_HUGETLB;

	shmid = shmget(IPC_PRIVATE, size, shmflag);
	if (shmid < 0)
		return false;

	shmhead = shmat(shmid, NULL, 0);

	/*
	 * To make sure the shared memory segment being released
	 * when the process exit.
	 * If shmat(2) failed, this segment shall be removed soon,
	 * because nobody maps it.
	 */
	shmctl(shmid, IPC_RMID, NULL);

	if (shmhead == (void *)(-1))
		return false;

	shmhead->shmid = shmid;
	shmhead->segment_size = size;

	for (mclass = 0; mclass <= SHMCLASS_MAX_BITS; mclass++)
	{
		shmlist_init(&shmhead->free_list[mclass]);
		shmhead->num_free[mclass] = 0;
		shmhead->num_active[mclass] = 0;
	}

	offset = 1 << (fls64(sizeof(shmhead_t)) + 1);
	if (offset < SHMCLASS_MIN_SIZE)
		offset = SHMCLASS_MIN_SIZE;

	while (shmhead->segment_size - offset >= SHMCLASS_MIN_SIZE)
	{
		shmchunk_t *chunk;

		/* choose an appropriate chunk class */
		mclass = ffs64(offset) - 1;
		if (mclass > SHMCLASS_MAX_BITS)
			mclass = SHMCLASS_MAX_BITS;
		assert(mclass >= SHMCLASS_MIN_BITS);

		/* if (offset + chunk_size) over the tail, truncate it */
		while (shmhead->segment_size < offset + (1 << mclass))
			mclass--;

		if (mclass < SHMCLASS_MIN_BITS)
			break;

		/* chain this free-chunk to the free_list */
		chunk = offset_to_addr(offset);
		chunk->mclass = mclass;
		chunk->active = false;

		shmlist_add(&shmhead->free_list[mclass], &chunk->list);
		shmhead->num_free[mclass]++;

		offset += (1 << mclass);
	}
	shmmgr_init_mutex(&shmhead->lock);

	/* initialization of shared buffer management */
	shmhead->shmbuf_head = shmbuf_init(size);

	return true;
}
#endif
#if 0
/*
 * Routines to module testing
 */
static void shmmgr_dump(void)
{
	uint64_t	total_active = 0;
	uint64_t	total_free = 0;
	int			mclass;

	if (shmhead == NULL)
		return;

	pthread_mutex_lock(&shmhead->lock);

	printf("segment size: %" PRIu64 "\n", shmhead->segment_size);
	for (mclass = SHMCLASS_MIN_BITS; mclass <= SHMCLASS_MAX_BITS; mclass++)
	{
		if (mclass < 10)
			printf("% 5uB: % 6u of used, % 6u of free\n", (1<<mclass),
				   shmhead->num_active[mclass], shmhead->num_free[mclass]);
		else if (mclass < 20)
			printf("% 4uKB: % 6u of used, % 6u of free\n", (1<<(mclass - 10)),
				   shmhead->num_active[mclass], shmhead->num_free[mclass]);
		else if (mclass < 30)
			printf("% 4uMB: % 6u of used, % 6u of free\n", (1<<(mclass - 20)),
				   shmhead->num_active[mclass], shmhead->num_free[mclass]);
		else
			printf("% 4uGB: % 6u of used, % 6u of free\n", (1<<(mclass - 30)),
				   shmhead->num_active[mclass], shmhead->num_free[mclass]);
	}
	printf("total active: %" PRIu64 "\n", total_active);
	printf("total free:   %" PRIu64 "\n", total_free);
	printf("total size:   %" PRIu64 "\n", total_active + total_free);

	pthread_mutex_unlock(&shmhead->lock);
}


int main(int argc, char *argv[])
{
	ssize_t		size;
	char	   *cmd;
	int			i, j, k;
	void	   *ptr[1024];

	if (argc < 2)
		return 1;

	size = atol(argv[1]);

	if (shmmgr_init(size, false) < 0)
	{
		perror("failed to init shared memory segment");
		return;
	}
	shmmgr_dump();

	for (i = 2, j = 0; i < argc; i++)
	{
		cmd = argv[i];

		if (cmd[0] == 'a' && cmd[1] == ':')
		{
			size = atol(cmd + 2);
			ptr[j++] = shmmgr_alloc(size);
		}
		else if (cmd[0] == 'f' && cmd[1] == ':')
		{
			k = atoi(cmd + 2);
			shmmgr_free(ptr[k]);
		}
		else
		{
			printf("command unknown: %s\n", cmd);
		}
		printf("------\n");
		shmmgr_dump();
	}
	return 0;
}
#endif
