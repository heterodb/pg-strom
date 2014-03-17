/*
 * ipc.c
 *
 * Routines for inter-process communication stuff.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"

#include "pg_strom.h"

typedef struct {
	pthread_mutex_t	lock;
	pthread_cond_t	cond;
	dlist_head		qhead;
	bool			closed;
} pgstrom_queue;

typedef struct {
	int				type;
	dlist_node		chain;
	pgstrom_queue  *recvq;
} pgstrom_queue_item;

static pthread_mutexattr_t	mutex_attr;
static pthread_rwlockattr_t	rwlock_attr;
static pthread_condattr_t	cond_attr;


bool
pgstrom_queue_init(pgstrom_queue *queue)
{
	int		rc;

	rc = pthread_mutex_init(&queue->lock, &mutex_attr);
	if (rc != 0)
		return false;

	rc = pthread_cond_init(&queue->cond, &cond_attr);
	if (rc != 0)
	{
		pthread_mutex_destroy(&queue->lock);
		return false;
	}
	dlist_init(&queue->qhead);
	queue->closed = false;

	return true;
}

bool
pgstrom_enqueue_item(pgstrom_queue *queue, pgstrom_queue_item *qitem)
{
	bool	result = false;
	int		rc;

	pthread_mutex_lock(&queue->lock);
	if (!queue->closed)
	{
		dlist_push_tail(&queue->qhead, &qitem->chain);
		result = true;
		rc = pthread_cond_signal(&queue->cond);
		Assert(rc == 0);
	}
	pthread_mutex_unlock(&queue->unlock);

	return result;
}

pgstrom_queue_item *
pgstrom_dequeue_item(pgstrom_queue *queue)
{
	pgstrom_queue_item *qitem;
	dlist_node	   *dnode;

	pthread_mutex_lock(&queue->lock);
	if (dlist_is_empty(&queue->qhead))
	{
		rc = pthread_cond_wait(&queue->cond, &queue->lock);
		Assert(rc == 0);

		if (dlist_is_empty(&queue->qhead))
		{
			pthread_mutex_unlock(&queue->unlock);

			return NULL;
		}
	}
	dnode = dlist_pop_head_node(&queue->qhead);
	qitem = dlist_container(pgstrom_queue_item, chain, dnode);

	pthread_mutex_unlock(&queue->unlock);

	return qitem;
}

pgstrom_queue_item *
pgstrom_try_dequeue(pgstrom_queue *queue)
{}

pgstrom_queue_item *
pgstrom_dequeue_timeout(pgstrom_queue *queue, long wait_usec)
{}

void
pgstrom_close_queue(pgstrom_queue *queue)
{
	pthread_mutex_lock(&queue->lock);
	if (!queue->closed)
		queue->closed = true;
	pthread_mutex_unlock(&queue->unlock);
}


void
pgstrom_ipc_init(void)
{
	int		rc;

	/* initialization of mutex_attr */
	rc = pthread_mutexattr_init(&mutex_attr);
	if (rc != 0)
		elog(ERROR, "failed on pthread_mutexattr_init: %s",
			 strerror(rc));
	rc = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
	if (rc != 0)
		elog(ERROR, "failed on pthread_mutexattr_setpshared: %s",
			 strerror(rc));

	/* initialization of rwlock_attr */
	rc = pthread_rwlockattr_init(&rwlock_attr);
	if (rc != 0)
		elog(ERROR, "failed on pthread_rwlockattr_init: %s",
			 strerror(rc));
	rc = pthread_rwlockattr_setpshared(&rwlock_attr, PTHREAD_PROCESS_SHARED);
	if (rc != 0)
		elog(ERROR, "failed on pthread_rwlockattr_setpshared: %s",
			 strerror(rc));

	/* initialization of cond_attr */
	rc = pthread_condattr_init(&cond_attr);
	if (rc != 0)
		elog(ERROR, "failed on pthread_condattr_init: %s",
			 strerror(rc));
	rc = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
	if (rc != 0)
		elog(ERROR, "failed on pthread_condattr_setpshared: %s",
			 strerror(rc));
}
