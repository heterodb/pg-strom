/*
 * shmmgr.c - shared memory management
 *
 * Copyright (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 *
 */
#include "pg_boost.h"

/*
 * Shared memory chunk
 */
typedef struct {
	uint8_t		mclass;		/* power of chunk size */
	bool		active;		/* true, if active chunk */
	unsigned long data[];	/* to be aligned by compiler */
} shmchunk_t;

/*
 * Shared memory header
 */
typedef struct {
	int			shmid;
	offset_t	total_size;
	offset_t	super_size;
	mlist_t		free_list[SHMCLASS_MAX_BITS + 1];
	uint64_t	num_active[SHMCLASS_MAX_BITS + 1];
	uint64_t	num_free[SHMCLASS_MAX_BITS + 1];
	pthread_mutex_t	lock;
} shmhead_t;

static shmhead_t   *shmhead = NULL;

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


static bool
shmmgr_split_chunk(int mclass)
{}




void *
shmmgr_alloc(size_t size)
{}

void
shmmgr_free(void *ptr)
{

}

int
shmmgr_init(key_t key, size_t size, bool hugetlb)
{

}

void
shmmgr_exit(void)
{
	
}
