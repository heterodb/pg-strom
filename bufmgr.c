/*
 * bufmgr.c - buffer management of shared memory segment
 *
 * Copyright (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "pg_boost.h"

#define SHMBUF_TAG_BTREE			0x01
#define SHMBUF_TAG_HEAP				0x02

#define SHMBUF_FLAGS_DIRTY_CACHE	0x01
#define SHMBUF_FLAGS_HOT_CACHE		0x02

struct shmbuf_t
{
	offset_t	storage;
	offset_t	cached;
	size_t		size;
	shmlist_t	list;
	int			refcnt;
	uint16_t	index;
	uint8_t		tag;
	uint8_t		flags;
	pthread_mutex_t	lock;
};

#define SHMBUF_NUM_ACTIVE_SLOT		160
typedef struct
{
	shmlist_t		active_list[SHMBUF_NUM_ACTIVE_SLOT];
	pthread_mutex_t	active_lock[SHMBUF_NUM_ACTIVE_SLOT];
	int				reclaim_hint;
	int				cache_hint;
} shmbuf_slot_t;

static shmbuf_slot_t   *shmbuf_slot = NULL;




bool
shmbuf_create(shmbuf_t *shmbuf, int tag, size_t size)
{
	void   *cached;
	int		index;

	cached = shmmgr_alloc(size);
	if (!cached)
		return false;

	index = __sync_fetch_and_and(&shmbuf->cache_hint, 1)
		% SHMBUF_NUM_ACTIVE_SLOT;

	shmbuf->storage = addr_to_offset(NULL);
	shmbuf->cached = addr_to_offset(cached);
	shmbuf->size = size;
	shmbuf->refcnt = 0;
	shmbuf->index = index;
	shmbuf->tag = tag;
	shmbuf->flags = SHMBUF_FLAGS_DIRTY_CACHE | SHMBUF_FLAGS_HOT_CACHE;

	pthread_mutex_lock(&shmbuf_slot->active_lock[index]);

	shmlist_add(&shmbuf_slot->active_list[index], &shmbuf->list);

	pthread_mutex_unlock(&shmbuf_slot->active_lock[index]);

	return shmbuf_t;
}



extern shmbuf_t	   *shmbuf_alloc(int tag, size_t size);
extern void			shmbuf_free(shmbuf_t *shmbuf);
extern void		   *shmbuf_get(shmbuf_t *shmbuf);
extern void			shmbuf_put(shmbuf_t *shmbuf);
extern void			shmbuf_set_dirty(shmbuf_t *shmbuf);

bool
shmbuf_init(size_t size)
{
	int		i;

	shmbuf_slot = shmmgr_alloc(sizeof(shmbuf_slot_t));
	if (!shmbuf_slot)
		return false;

	for (i=0; i < SHMBUF_NUM_ACTIVE_SLOT; i++)
	{
		shmlist_init(&shmbuf_slot->active_list[i]);
		shmmgr_init_mutex(&shmbuf_slot->active_lock[i]);
	}
	shmbuf_slot->reclaim_hint = 0;
	shmbuf_slot->cache_hint = 0;

	return true;
}
