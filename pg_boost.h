/*
 * pg_boost.h - Header file of pg_boost module
 *
 * Copyright (c) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#ifndef PG_BOOST_H
#define PG_BOOST_H
#include <pthread.h>

/*
 * msegment.c - management of shared memory segment
 */
extern struct msegment_t *msegment;
typedef struct mchunk_buffer_t *mbuffer_t;

#define addr_to_offset(p)		\
	((p) == NULL ? 0 : (((uintptr_t)(p)) - (uintptr_t)(msegment)))
#define offset_to_addr(p)		\
	((p) == 0 ? NULL : (void *)((p) + (uintptr_t)(msegment)))

extern void		   *shmseg_alloc(size_t size);
extern void		   *shmseg_try_alloc(size_t size);
extern void		   *shmseg_resize(void *addr, size_t new_size);
extern void		   *shmseg_try_resize(void *addr, size_t new_size);
extern void			shmseg_free(void *addr);
extern size_t		shmseg_get_size(void *addr);
extern mbuffer_t	shmseg_alloc_buffer(size_t size);
extern void			shmseg_free_buffer(mbuffer_t mbuffer);
extern void		   *shmseg_get_read_buffer(mbuffer_t mbuffer);
extern void		   *shmseg_get_write_buffer(mbuffer_t mbuffer);
extern void			shmseg_put_buffer(mbuffer_t mbuffer, bool is_dirty);
extern void			shmseg_init_mutex(pthread_mutex_t *lock);
extern void			shmseg_init_rwlock(pthread_rwlock_t *lock);
extern void			shmseg_init(void);
extern void			shmseg_exit(void);

/*
 * Dual linked list
 */
typedef struct mlist_t
{
	uintptr_t	prev;
	uintptr_t	next;
} mlist_t;

#define container_of(addr, type, member)		\
	(type *)(((uintptr_t)(addr)) - offsetof(type, member))
#define list_entry(offset, type, member)		\
	container_of(offset_to_addr(offset), type, member)

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
	for (entry = list_entry((head)->next, typeof(*entry), member);		\
		 &entry->member != (head);										\
		 entry = list_entry(entry->member.next, typeof(*entry), member))

#define mlist_foreach_entry_safe(entry, temp, head, member)				\
	for (entry = list_entry((head)->next, typeof(*entry), member),		\
		 temp = list_entry(entry->member.next, typeof(*entry), member); \
		 &entry->member != (head);										\
		 entry = temp,													\
		 temp = list_entry(entry->member.next, typeof(*entry), member))

static inline bool mlist_empty(mlist_t *mlist)
{
	return mlist->next == addr_to_offset(mlist) ? true : false;
}

static inline void mlist_init(mlist_t *mlist)
{
	mlist->next = mlist->prev = addr_to_offset(mlist);
}

static inline void mlist_add(mlist_t *base, mlist_t *list)
{
	mlist_t	   *next = offset_to_addr(base->next);

	base->next = addr_to_offset(list);
	list->prev = addr_to_offset(base);
	list->next = addr_to_offset(next);
	next->prev = addr_to_offset(list);
}

static inline void mlist_del(mlist_t *mlist)
{
	mlist_t	   *plist = offset_to_addr(mlist->prev);
	mlist_t	   *nlist = offset_to_addr(mlist->next);

	plist->next = mlist->next;
	nlist->prev = mlist->prev;

	mlist_init(mlist);
}

#endif	/* PG_BOOST_H */
