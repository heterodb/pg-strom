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
#include "miscadmin.h"
#include "utils/guc.h"
#include <limits.h>
#include <sys/time.h>
#include "pg_strom.h"

static pthread_mutexattr_t	mutex_attr;
static pthread_rwlockattr_t	rwlock_attr;
static pthread_condattr_t	cond_attr;
static int		pgstrom_mqueue_timeout;

/*
 * pgstrom_create_queue
 *
 * It creates a message queue of PG-Strom. Here is two types of message
 * queues; one is for OpenCL background server, the other is for backend
 * process to be used to receive response messages.
 * A message queue for the background server is controled with reference-
 * counter. It shall be incremented when backend enqueues a message to
 * OpenCL server, then decremented when backend dequeued a message, the
 * server tried to enqueue a message to "closed" queue, or the message
 * queue is closed.
 * It is needed for error handling in the backend side, because PostgreSQL
 * adopts exception-catch style error handling, it may lose the messages
 * being executed asynchronously.
 */
pgstrom_queue *
pgstrom_create_queue(bool is_server)
{
	shmem_context  *context = pgstrom_get_mqueue_context();
	pgstrom_queue  *queue;

	Assert(context != NULL);
	queue = pgstrom_shmem_alloc(context, sizeof(pgstrom_queue));
	if (!queue)
		return NULL;

	if (pthread_mutex_init(&queue->lock, &mutex_attr) != 0)
		goto error;
	if (pthread_cond_init(&queue->cond, &cond_attr) != 0)
		goto error;
	dlist_init(&queue->qhead);
	if (is_server)
		queue->refcnt = -1;
	else
		queue->refcnt = 1;

	queue->closed = false;

	return queue;

error:
	pgstrom_shmem_free(queue);
	return NULL;
}

/*
 * pgstrom_enqueue
 *
 * It allows the backend to enqueue a message towards OpenCL background
 * server.
 */
bool
pgstrom_enqueue_message(pgstrom_queue *queue, pgstrom_message *message)
{
	pgstrom_queue *respq;
	int		rc;

	pthread_mutex_lock(&queue->lock);
	if (queue->closed)
	{
		pthread_mutex_unlock(&queue->lock);
		return false;
	}
	/*
	 * NOTE: we assume the sender gives a response queue that has refcnt
	 * more than 0 at this function call, and the core backend never
	 * performs multi-threading. So, it is safe to acquire the response
	 * queue after the unlocking of this queue.
	 */
	respq = message->respq;

	/* enqueue this message */
	Assert(queue->refcnt < 0);
	dlist_push_tail(&queue->qhead, &message->chain);
	rc = pthread_cond_signal(&queue->cond);
	Assert(rc == 0);
	pthread_mutex_unlock(&queue->lock);

	/*
	 * Increment expected number of response messages, because we cannot
	 * drop the response message queue until all the messages in execution
	 * got returned.
	 * A message queue is constructed with refcnt=1, and pgstrom_close_queue
	 * decrements the reference count. Usually, it drops the message queue
	 * itself, however, asynchronouse execution messages will still need
	 * the response queue to back execution status or drop message itself.
	 */
	Assert(respq != NULL);

	pthread_mutex_lock(&respq->lock);
	Assert(respq->refcnt > 0);
	respq->refcnt++;
	pthread_mutex_unlock(&respq->lock);

	return true;
}

/*
 * pgstrom_reply_message
 *
 * It enqueues a response message by the OpenCL background server.
 * It shouldn't be called by backend processes.
 */
void
pgstrom_reply_message(pgstrom_message *message)
{
	pgstrom_queue  *respq = message->respq;
	bool	queue_free = false;
	bool	message_free = false;
	int		rc;

	pthread_mutex_lock(&respq->lock);
	if (respq->closed)
		message_free = true;
	else
	{
		/* reply this message */
		Assert(respq->refcnt > 0);
		dlist_push_tail(&respq->qhead, &message->chain);
		rc = pthread_cond_signal(&respq->cond);
		Assert(rc == 0);
	}
	if (--respq->refcnt == 0)
	{
		Assert(respq->closed);
		queue_free = true;
	}
	pthread_mutex_unlock(&respq->lock);

	/*
	 * Release message and response queue, if the backend already aborted
	 * and nobody can receive the response messages.
	 */
	if (message_free)
	{
		if (message->cb_release)
			message->cb_release(message);
		pgstrom_shmem_free(message);
	}

	if (queue_free)
	{
		pthread_cond_destroy(&respq->cond);
		pthread_mutex_destroy(&respq->lock);
		pgstrom_shmem_free(respq);
	}
}

/*
 * pgstrom_dequeue_message
 *
 * It fetches a message from the message queue. If empty, it waits for new
 * messages will come, or returns NULL if it exceeds timeout or it got
 * a signal being pending.
 */
pgstrom_message *
pgstrom_dequeue_message(pgstrom_queue *queue)
{
#define POOLING_INTERVAL	200000000	/* 0.2msec */
	pgstrom_message *result = NULL;
	struct timeval	basetv;
	struct timespec	timeout;
	ulong	timeleft = ((ulong)pgstrom_mqueue_timeout) * 1000000UL;
	bool	queue_release = false;
	int		rc;

	rc = gettimeofday(&basetv, NULL);
	Assert(rc == 0);
	timeout.tv_sec = basetv.tv_sec;
	timeout.tv_nsec = basetv.tv_usec * 1000;

	for (;;)
	{
		pthread_mutex_lock(&queue->lock);
		/* dequeue a message from the message queue */
		if (!dlist_is_empty(&queue->qhead))
		{
			dlist_node *dnode
				= dlist_pop_head_node(&queue->qhead);

			result = dlist_container(pgstrom_message, chain, dnode);
			if (queue->refcnt > 0)
			{
				if (--queue->refcnt == 0)
					queue_release = true;
			}
			pthread_mutex_unlock(&queue->lock);
			break;
		}
		else if (timeleft == 0)
		{
			pthread_mutex_unlock(&queue->lock);
			break;
		}
		else
		{
			/* setting up the next timeout */
			if (timeleft > POOLING_INTERVAL)
			{
				timeout.tv_nsec += POOLING_INTERVAL;
				timeleft -= POOLING_INTERVAL;
			}
			else
			{
				timeout.tv_nsec += timeleft;
				timeleft = 0;
			}
			rc = pthread_cond_timedwait(&queue->cond, &queue->lock, &timeout);
			Assert(rc == 0 || rc == ETIMEDOUT);

			/* signal will break waiting loop */
			if (InterruptPending)
				timeleft = 0;
		}
	}
	/*
	 * If this queue is already closed and no messages will come,
	 * the queue shall be dropped.
	 */
	if (queue_release)
	{
		Assert(queue->closed);
		pthread_cond_destroy(&queue->cond);
		pthread_mutex_destroy(&queue->lock);
		pgstrom_shmem_free(queue);
	}
	return result;
}

/*
 * pgstrom_try_dequeue_message
 *
 * It is almost equivalent to pgstrom_dequeue_message(), however, it never
 * wait for new messages, will return immediately.
 */
pgstrom_message *
pgstrom_try_dequeue_message(pgstrom_queue *queue)
{
	pgstrom_message *result = NULL;
	bool	queue_release = false;

	pthread_mutex_lock(&queue->lock);
	if (!dlist_is_empty(&queue->qhead))
	{
		dlist_node *dnode
			= dlist_pop_head_node(&queue->qhead);

		result = dlist_container(pgstrom_message, chain, dnode);
		if (queue->refcnt > 0)
		{
			if (--queue->refcnt == 0)
				queue_release = true;
		}
	}
	pthread_mutex_unlock(&queue->lock);

	/*
	 * If this queue is already closed and no messages will come,
	 * the queue shall be dropped.
	 */
	if (queue_release)
	{
		pthread_cond_destroy(&queue->cond);
		pthread_mutex_destroy(&queue->lock);
		pgstrom_shmem_free(queue);
	}
	return result;
}

/*
 * pgstrom_close_queue
 *
 * It closes this message queue. Once a message queue got closed, it does not
 * accept any new messages and the queue will be dropped when last message
 * is dequeued or last expected message is tried to enqueue.
 */
void
pgstrom_close_queue(pgstrom_queue *queue)
{
	bool	queue_release = false;

	pthread_mutex_lock(&queue->lock);
	Assert(!queue->closed);
	queue->closed = true;

	if (queue->refcnt > 0)
	{
		if (--queue->refcnt == 0)
			queue_release = true;
	}
	pthread_mutex_unlock(&queue->lock);

	if (queue_release)
	{
		pthread_cond_destroy(&queue->cond);
		pthread_mutex_destroy(&queue->lock);
		pgstrom_shmem_free(queue);
	}
}

/*
 * pgstrom_get_queue
 *
 * increment reference counter of the queue
 */
void
pgstrom_get_queue(pgstrom_queue *queue)
{
	pthread_mutex_lock(&queue->lock);
	Assert(queue->refcnt > 0);
	queue->refcnt++;
	pthread_mutex_unlock(&queue->lock);
}

/*
 * pgstrom_put_queue
 *
 * decrement reference counter of the queue
 */
void
pgstrom_put_queue(pgstrom_queue *queue)
{
	bool	queue_release = false;

	pthread_mutex_lock(&queue->lock);
	if (queue->refcnt > 0)
	{
		if (--queue->refcnt == 0)
		{
			/* only closed queue should become refcnt==0 */
			Assert(queue->closed);
			queue_release = true;
		}
	}
	pthread_mutex_unlock(&queue->lock);

	if (queue_release)
	{
		pthread_cond_destroy(&queue->cond);
		pthread_mutex_destroy(&queue->lock);
		pgstrom_shmem_free(queue);
	}
}

/*
 * pgstrom_init_mqueue
 *
 * initialization at library loading
 */
void
pgstrom_init_mqueue(void)
{
	int		rc;

	/* timeout configuration of the message queue feature */
	DefineCustomIntVariable("pgstrom.mqueue_timeout",
							"timeout of PG-Strom's message queue in msec",
							NULL,
							&pgstrom_mqueue_timeout,
							60 * 1000,	/* 60 sec */
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

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
