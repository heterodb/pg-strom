/*
 * pg_boost.h - Header file of pg_boost module
 *
 * Copyright (c) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * 
 */
#ifndef PG_BOOST_H
#define PG_BOOST_H
#include "postgres.h"
#include <pthread.h>

/*
 * Dual linked list
 */
typedef struct mlist_s {
	struct mlist_s *prev;
	struct mlist_s *next;
} mlist_t;

#define container_of(addr, type, member)				\
	(type *)(((uintptr_t)(addr)) - offsetof(type, member))

/*
 * mlist_foreach_entry(_safe)
 *
 * Iterator of mlist. "_safe" version is safe against remove items.
 * @entry  : pointer of type * owning the list as a loop cursor.
 * @temp   : another type * to use as temporary storage
 * @head   : head of the list to be iterated
 * @member : name of the shmlist_t within the struct
 */
#define mlist_foreach_entry(entry, head, member)						\
	for (entry = container_of((head)->next, typeof(*entry), member);	\
		 &entry->member != (head);										\
		 entry = container_of(entry->member.next, typeof(*entry), member))

#define mlist_foreach_entry_safe(entry, temp, head, member)				\
	for (entry = container_of((head)->next, typeof(*entry), member),	\
		 temp = container_of(entry->member.next, typeof(*entry), member); \
		 &entry->member != (head);										\
		 entry = temp,													\
		 temp = container_of(entry->member.next, typeof(*entry), member))

static inline bool mlist_empty(mlist_t *mlist)
{
	return mlist->next == mlist ? true : false;
}

static inline void mlist_init(mlist_t *mlist)
{
	mlist->next = mlist->prev = mlist;
}

static inline void mlist_add(mlist_t *base, mlist_t *list)
{
	mlist_t	   *next = base->next;

	base->next = list;
	list->prev = base;
	list->next = next;
	next->prev = list;
}

static inline void mlist_del(mlist_t *mlist)
{
	mlist_t	   *plist = mlist->prev;
	mlist_t	   *nlist = mlist->next;

	plist->next = nlist;
	nlist->prev = plist;

	mlist_init(mlist);
}

/*
 * Management of shared memory segment
 */
typedef struct mchunk_cache_s *mcache_t;

extern void *shmseg_alloc(size_t size);
extern void	*shmseg_try_alloc(size_t size);
extern void shmseg_free(void *addr);
extern size_t shmseg_size(void *addr);
extern void *shmseg_resize(void *addr, size_t new_size);
extern void *shmseg_try_resize(void *addr, size_t new_size);
extern mcache_t shmseg_cache_alloc(size_t size);
extern void shmseg_cache_resize(mcache_t mcache, size_t new_size);
extern void shmseg_cache_free(mcache_t mcache);
extern void *shmseg_cache_get(mcache_t mcache);
extern void shmseg_cache_put(mcache_t mcache, bool set_dirty);
extern void	shmseg_init_mutex(pthread_mutex_t *lock);
extern void	shmseg_init_rwlock(pthread_rwlock_t *lock);
extern void shmseg_init(void);
extern void shmseg_exit(void);

#endif	/* PG_BOOST_H */
