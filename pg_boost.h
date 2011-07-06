/*
 * pg_boost.h - Header file of pg_boost module
 *
 * Copyright (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * 
 */
#ifndef PG_BOOST_H
#define PG_BOOST_H
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

/*
 * XXX - to be defined in postgres.h
 */
#ifndef bool
typedef char bool;
#endif

#ifndef true
#define true	((bool) 1)
#endif

#ifndef false
#define false	((bool) 0)
#endif

/*
 * shmmgr.c - Management of shared memory segment
 */

/* offset_t - Representation on the shared memory segment */
typedef uint64_t	offset_t;

/*
 * alloc/free memory chunk on shared memory segment
 */
extern int			shmmgr_init(size_t size, bool hugetlb);
extern void			shmmgr_exit(void);
extern void		   *shmmgr_alloc(size_t size);
extern void			shmmgr_free(void *ptr);
extern bool			shmmgr_init_mutex(pthread_mutex_t *lock);
extern bool			shmmgr_init_rwlock(pthread_rwlock_t *lock);
extern offset_t		addr_to_offset(void *addr);
extern void		   *offset_to_addr(offset_t offset);

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

extern bool			shmlist_empty(shmlist_t *list);
extern void			shmlist_init(shmlist_t *list);
extern void			shmlist_add(shmlist_t *base, shmlist_t *list);
extern void			shmlist_del(shmlist_t *list);

/*
 * Shared buffer management
 */
#define SHMBUF_TAG_BTREE	1
#define SHMBUF_TAG_HEAP		2

typedef struct shmbuf_t shmbuf_t;

extern shmbuf_t	   *shmbuf_alloc(int tag, size_t size);
extern void			shmbuf_free(shmbuf_t *shmbuf);
extern void		   *shmbuf_get(shmbuf_t *shmbuf);
extern void			shmbuf_put(shmbuf_t *shmbuf);
extern void			shmbuf_set_dirty(shmbuf_t *shmbuf);
extern bool			shmbuf_init(size_t size);

#endif	/* PG_BOOST_H */
