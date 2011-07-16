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
 * Management of shared memory segment
 */
extern shmptr_t	shmmgr_alloc(size_t size);
extern shmptr_t shmmgr_try_alloc(size_t size);
extern shmptr_t	shmmgr_buffer_alloc(size_t size);
extern void		shmmgr_free(shmptr_t ptr);
extern void		shmmgr_init_mutex(pthread_mutex_t *lock);
extern void		shmmgr_init_rwlock(pthread_rwlock_t *lock);
extern void	   *shmptr_to_addr(shmptr_t ptr);
extern shmptr_t	shmptr_from_addr(void *addr);
extern void	   *shmmgr_get_addr(shmptr_t ptr);
extern void		shmmgr_put_addr(shmptr_t ptr, bool is_dirty);
extern void		shmmgr_init(size_t size, bool hugetlb);
extern void		shmmgr_exit(void);

#endif	/* PG_BOOST_H */
