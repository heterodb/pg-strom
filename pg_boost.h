/*
 * pg_boost.h - Header file of pg_boost module
 *
 * Copyright (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * 
 */
#ifndef PG_BOOST_H
#define PG_BOOST_H
#include "postgres.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

/*
 * offset_t - offset from the head of shared memory segment
 */
typedef uint64	offset_t;

/*
 * Dual linked list on shared memory segment
 */
typedef struct {
	offset_t	prev;
	offset_t	next;
} shmlist_t;

#define offset_of(type, member)					\
	((unsigned long) &((type *)0)->member)
#define container_of(ptr, type, member)			\
	(type *)(((char *)ptr) - offset_of(type, member))

/*
 * iterator of shmlist
 * @entry  : the typed pointer to use as a loop cursor
 * @head   : the head of shmlist
 * @member : the name of the shmlist_t member within the struct
 */
#define shmlist_foreach_entry(entry, head, member)						\
	for (entry = container_of(offset_to_addr((head)->next), typeof(*entry), member); \
		 &entry->member != (head);										\
		 entry = container_of(offset_to_addr(entry->member.next), typeof(*entry), member))

extern bool			shmlist_empty(shmlist_t *list);
extern void			shmlist_init(shmlist_t *list);
extern void			shmlist_add(shmlist_t *base, shmlist_t *list);
extern void			shmlist_del(shmlist_t *list);

/*
 * Management of shared memory segment
 */
extern bool			shmmgr_init(size_t size, bool hugetlb);
extern void			shmmgr_exit(void);
extern void		   *shmmgr_alloc(size_t size);
extern void			shmmgr_free(void *ptr);
extern bool			shmmgr_init_mutex(pthread_mutex_t *lock);
extern bool			shmmgr_init_rwlock(pthread_rwlock_t *lock);
extern offset_t		addr_to_offset(void *addr);
extern void		   *offset_to_addr(offset_t offset);

extern void		   *shmmgr_get_bufmgr_head(void);
extern size_t		shmmgr_get_size(void *ptr);

/*
 * Shared buffer management
 */
#define SHMBUF_FLAGS_DIRTY_CACHE	0x01
#define SHMBUF_FLAGS_HOT_CACHE		0x02
#define SHMBUF_FLAGS_COMPRESSED		0x04

typedef struct {
	offset_t	storage;
	offset_t	cached;
	shmlist_t	list;
	size_t		size;
	int			refcnt;
	uint16		index;
	uint8		tag;
	uint8		flags;
	pthread_mutex_t lock;
} shmbuf_t;

extern bool		shmbuf_create(shmbuf_t *shmbuf, int tag, size_t size);
extern void		shmbuf_delete(shmbuf_t *shmbuf);
extern void	   *shmbuf_get_buffer(shmbuf_t *shmbuf);
extern void	   *shmbuf_put_buffer(shmbuf_t *shmbuf);
extern void		shmbuf_set_dirty(shmbuf_t);
extern offset_t	shmbuf_init(void);

#endif	/* PG_BOOST_H */
