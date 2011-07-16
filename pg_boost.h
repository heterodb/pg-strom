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
 * Pointer on the shared memory segment
 */
typedef uintptr_t	shmptr_t;

/*
 * Dual linked list on shared memory segment
 */
typedef struct {
	offset_t		prev;
	offset_t		next;
} shmlist_t;

#define offset_of(type, member)					\
	((unsigned long) &((type *)0)->member)
#define container_of(ptr, type, member)			\
	(type *)(((char *)ptr) - offset_of(type, member))
#define list_entry(offset, type, member)		\
	container_of(offset_to_addr(offset), type, member)

/*
 * shmlist_foreach_entry(_safe)
 *
 * iterator of the shmlist. _safe version is safe against remove items.
 * @entry  : pointer of type * owning the list as a loop cursor.
 * @temp   : another type * to use as temporary storage
 * @head   : head of the list to be iterated
 * @member : name of the shmlist_t within the struct
 */
#define shmlist_foreach_entry(entry, head, member)						\
	for (entry = list_entry((head)->next, typeof(*entry), member);		\
		 &entry->member != (head);										\
		 entry = list_entry(entry->member.next, typeof(*entry), member))

#define shmlist_foreach_entry_safe(entry, temp, head, member)			\
	for (entry = list_entry((head)->next, typeof(*entry), member),		\
		 temp = list_entry(entry->member.next, typeof(*entry), member); \
		 &entry->member != (head);										\
		 entry = temp,													\
		 temp = list_entry(entry->member.next, typeof(*entry), member))

extern bool		shmlist_empty(shmlist_t *list);
extern void		shmlist_init(shmlist_t *list);
extern void		shmlist_add(shmlist_t *base, shmlist_t *list);
extern void		shmlist_del(shmlist_t *list);

/*
 * Management of shared memory segment
 */
extern shmptr_t	shmmgr_alloc(size_t size, bool is_buffer);
extern shmptr_t	shmmgr_try_alloc(size_t size, bool is_buffer);
extern void		shmmgr_free(shmptr_t ptr);
extern void		shmmgr_init_mutex(pthread_mutex_t *lock);
extern void		shmmgr_init_rwlock(pthread_rwlock_t *lock);
extern void	   *shmmgr_get_addr(shmptr_t ptr);
extern void		shmmgr_put_addr(shmptr_t ptr);
extern void		shmmgr_get_size(shmptr_t ptr);
extern void		shmmgr_set_dirty(shmptr_t ptr);
extern void		shmmgr_init(int nsegments, size_t segment_size);
extern void		shmmgr_exit(void);

#endif	/* PG_BOOST_H */
