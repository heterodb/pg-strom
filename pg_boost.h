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
typedef char bool
#endif

#ifndef true
#define true	((bool) 1)
#endif

#ifndef false
#define false	((bool) 0)
#endif

/*
 * shmmgr.c
 */
typedef uint64_t	offset_t;

/* dual linked list */
typedef struct {
  offset_t	prev;
  offset_t	next;
} shmlist_t;

#define offset_of(type, member)	\
	((unsigned long) &((type *)0)->member)
#define container_of(ptr, type, member) \
	(type *)(((char *)ptr) - offset_of(type, member))

extern bool	shmlist_empty(shmlist_t *list);
extern void	shmlist_init(shmlist_t *list);
extern void	shmlist_add(shmlist_t *base, shmlist_t *list);
extern void	shmlist_del(shmlist_t *list);

extern int	shmmgr_init(key_t key, size_t size, bool hugetlb);
extern void	shmmgr_exit(void);
extern void    *shmmgr_alloc(size_t size);
extern void	shmmgr_free(void *ptr);
extern bool	shmmgr_init_mutex(pthread_mutex_t *lock);
extern bool	shmmgr_init_rwlock(pthread_rwlock_t *lock);
extern offset_t	addr_to_offset(void *addr);
extern void    *offset_to_addr(offset_t offset);







#endif	/* PG_BOOST_H */





