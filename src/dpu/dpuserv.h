/*
 * dpuserv.h
 *
 * Common definitions for DPU Service
 * --------
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef DPUSERV_H
#define DPUSERV_H
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <poll.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <netdb.h>
#include <arpa/inet.h>
#include "xpu_common.h"
#include "float2.h"
#include "heterodb_extra.h"

#define __Elog(fmt,...)									\
	do {												\
		fprintf(stderr, "error (%s:%d): " fmt "\n",		\
				__FILE__, __LINE__, ##__VA_ARGS__);		\
		exit(1);										\
	} while(0)
#define __Notice(fmt,...)								\
	fprintf(stderr, "notice (%s:%d): " fmt "\n",		\
			__FILE__, __LINE__, ##__VA_ARGS__)
#define __NoticeKexp(kcxt)								\
	fprintf(stderr, "notice (%s:%d): %s [%s]\n",		\
			(kcxt)->error_filename,						\
			(kcxt)->error_lineno,						\
			(kcxt)->error_message,						\
			(kcxt)->error_funcname)
#define __Info(fmt,...)									\
	do {												\
		if (verbose > 0)								\
			fprintf(stderr, "info (%s:%d): " fmt "\n",	\
					__FILE__, __LINE__, ##__VA_ARGS__);	\
	} while(0)
#define __Debug(fmt,...)								\
	do {												\
		if (verbose > 1)								\
			fprintf(stderr, "debug (%s:%d): " fmt "\n",	\
					__FILE__, __LINE__, ##__VA_ARGS__);	\
	} while(0)

#define Max(a,b)		((a) > (b) ? (a) : (b))
#define Min(a,b)		((a) < (b) ? (a) : (b))
#define Assert(cond)	assert(cond)

#define PAGE_SIZE				4096	/* assumes x86_64 host */
#define PAGE_ALIGN(LEN)			TYPEALIGN(PAGE_SIZE,LEN)
#define PAGE_ALIGN_DOWN(LEN)	TYPEALIGN_DOWN(PAGE_SIZE,LEN)
#define PGSTROM_CHUNK_SIZE		(65534UL << 10)

/*
 * dlist inline functions
 */
#ifndef offsetof
#define offsetof(type, field)	((long) &((type *)0)->field)
#endif
#define dlist_container(type, field, ptr)				\
	((type *)((char *)(ptr) - offsetof(type, field)))

typedef struct dlist_iter
{
	dlist_node *cur;	/* current element */
	dlist_node *end;	/* last node we'll iterate to */
} dlist_iter;

#define dlist_foreach(iter, lhead)										\
	for ((iter).end = &(lhead)->head,									\
		 (iter).cur = (iter).end->next ? (iter).end->next : (iter).end;	\
		 (iter).cur != (iter).end;										\
		 (iter).cur = (iter).cur->next)

static inline void
dlist_init(dlist_head *head)
{
	head->head.next = head->head.prev = &head->head;
}

static inline bool
dlist_is_empty(dlist_head *head)
{
	return (head->head.next == &(head->head) &&
			head->head.prev == &(head->head));
}

static inline void
dlist_push_tail(dlist_head *head, dlist_node *node)
{
	node->next = &head->head;
	node->prev = head->head.prev;
	node->prev->next = node;
	head->head.prev = node;
}

static inline void
dlist_delete(dlist_node *node)
{
	node->prev->next = node->next;
	node->next->prev = node->prev;
	memset(node, 0, sizeof(dlist_node));
}

static inline dlist_node *
dlist_pop_head_node(dlist_head *head)
{
	dlist_node *node;

	Assert(!dlist_is_empty(head));
	node = head->head.next;
	dlist_delete(node);
	return node;
}

/*
 * thin wrapper of mutex functions
 */
static inline void
pthreadMutexInit(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_init(mutex, NULL)) != 0)
		__Elog("failed on pthread_mutex_init: %m");
}

static inline void
pthreadMutexLock(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_lock(mutex)) != 0)
		__Elog("failed on pthread_mutex_lock: %m");
}

static inline void
pthreadMutexUnlock(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_unlock(mutex)) != 0)
		__Elog("failed on pthread_mutex_unlock: %m");
}

static inline void
pthreadRWLockInit(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_init(rwlock, NULL)) != 0)
		__Elog("failed on pthread_rwlock_init: %m");
}

static inline void
pthreadRWLockReadLock(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_rdlock(rwlock)) != 0)
		__Elog("failed on pthread_rwlock_rdlock: %m");
}

static inline void
pthreadRWLockWriteLock(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_wrlock(rwlock)) != 0)
		__Elog("failed on pthread_rwlock_wrlock: %m");
}

static inline void
pthreadRWLockUnlock(pthread_rwlock_t *rwlock)
{
	if ((errno = pthread_rwlock_unlock(rwlock)) != 0)
		__Elog("failed on pthread_rwlock_unlock: %m");
}

static inline void
pthreadCondInit(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_init(cond, NULL)) != 0)
		__Elog("failed on pthread_cond_init: %m");
}

static inline void
pthreadCondWait(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
	if ((errno = pthread_cond_wait(cond, mutex)) != 0)
		__Elog("failed on pthread_cond_wait: %m");
}

static inline void
pthreadCondBroadcast(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_broadcast(cond)) != 0)
		__Elog("failed on pthread_cond_broadcast: %m");
}

static inline void
pthreadCondSignal(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_signal(cond)) != 0)
		__Elog("failed on pthread_cond_signal: %m");
}

#endif	/* DPUSERV_H */
