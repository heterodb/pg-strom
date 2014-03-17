/*
 * mqueue.c
 *
 * Routines for inter-process communication via message queues
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

static pthread_mutexattr_t	mutex_attr;
static pthread_rwlockattr_t	rwlock_attr;
static pthread_condattr_t	cond_attr;

typedef struct {
	pthread_mutex_t	lock;
	pthread_cond_t	cond;
	dlist_head		qhead;
	int				refcnt;
	bool			closed;
} pgstrom_queue;

typedef struct {
	int				type;
	dlist_node		chain;
	pgstrom_queue  *respq;	/* queue for response message */
} pgstrom_message;


pgstrom_queue *
pgstrom_create_queue(bool persistent)
{
	shmem_context  *context = pgstrom_get_mqueue_context();
	pgstrom_queue  *queue;
	int		rc;

	Assert(context != NULL);
	queue = pgstrom_shmem_alloc(context, sizeof(pgstrom_queue));
	if (!queue)
		return NULL;

	if (pthread_mutex_init(&queue->lock, &mutex_attr) != 0)
		goto error;
	if (pthread_cond_init(&queue->cond, &cond_attr) != 0)
		goto error;
	dlist_init(&queue->qhead);
	if (persistent)
		queue->refcnt = -1;
	else
		queue->refcnt = 1;

	queue->closed = false;

	return queue;

error:
	pgstrom_shmem_free(queue);
	return NULL;
}

bool
pgstrom_enqueue_message(pgstrom_queue *queue,
						pgstrom_message *message)
{
	bool	result;
	int		rc;

	pthread_mutex_lock(&queue->lock);
	if (queue->closed)
		result = false;
	else
	{
		if (message->respq)
		{
			pgstrom_queue  *respq = message->respq;

			pthread_mutex_lock(&respq->lock);
			if (respq->refcnt >= 0)
				respq->refcnt++;
			pthread_mutex_unlock(&respq->lock);
		}
		dlist_push_tail(&queue->qhead, &qitem->chain);
		rc = pthread_cond_signal(&queue->cond);
		Assert(rc == 0);
		result = true;
	}
	pthread_mutex_unlock(&queue->lock);

	return result;
}

pgstrom_message *
pgstrom_dequeue_message(pgstrom_queue *queue)
{}

pgstrom_message *
pgstrom_try_dequeue_message(pgstrom_queue *queue)
{}

pgstrom_message *
pgstrom_dequeue_message_timeout(pgstrom_queue *queue, long wait_usec)
{}

void
pgstrom_close_queue(pgstrom_queue *queue)
{}


bool
pgstrom_enqueue_item(pgstrom_queue *queue, pgstrom_queue_item *qitem)
{
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

/*
 * pgstrom_mqueue_setup
 *
 * initialization post shared memory setting up
 */
void
pgstrom_mqueue_setup(void)
{
	shmem_context  *context = pgstrom_shmem_context_create("message queue");

	if (!context)
		elog(ERROR, "failed to create shared memory context");

	pgstrom_register_mqueue_context(context);
}

/*
 * pgstrom_mqueue_init
 *
 * initialization at library loading
 */
void
pgstrom_mqueue_init(void)
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
