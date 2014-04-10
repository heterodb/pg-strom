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
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "utils/guc.h"
#include <limits.h>
#include <sys/time.h>
#include "pg_strom.h"

static pthread_mutexattr_t	mutex_attr;
static pthread_rwlockattr_t	rwlock_attr;
static pthread_condattr_t	cond_attr;
static int		pgstrom_mqueue_timeout;

/* variables related to shared memory segment */
static shmem_startup_hook_type shmem_startup_hook_next;
static struct {
	slock_t			lock;
	dlist_head		free_queue_list;
	uint32			num_free;
	uint32			num_active;
	pgstrom_queue	serv_mqueue;	/* queue to OpenCL server */
} *mqueue_shm_values;

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
pgstrom_create_queue(void)
{
	pgstrom_queue  *mqueue;
	dlist_node	   *dnode;

	SpinLockAcquire(&mqueue_shm_values->lock);
	if (dlist_is_empty(&mqueue_shm_values->free_queue_list))
	{
		pgstrom_queue  *queues;
		int		i, numq;

		numq = (SHMEM_BLOCKSZ - sizeof(cl_uint)) / sizeof(pgstrom_queue);
		queues = pgstrom_shmem_alloc(sizeof(pgstrom_queue) * numq);
		if (!queues)
		{
			SpinLockRelease(&mqueue_shm_values->lock);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("out of shared memory")));
		}

		for (i=0; i < numq; i++)
		{
			dlist_push_tail(&mqueue_shm_values->free_queue_list,
							&queues[i].chain);
		}
		mqueue_shm_values->num_free += numq;
	}
	Assert(mqueue_shm_values->num_free > 0);
	mqueue_shm_values->num_free--;
	mqueue_shm_values->num_active++;
	dnode = dlist_pop_head_node(&mqueue_shm_values->free_queue_list);
	mqueue = dlist_container(pgstrom_queue, chain, dnode);
	SpinLockRelease(&mqueue_shm_values->lock);

	/* initialize the queue as new one */
	mqueue->stag = StromTag_MsgQueue;
	mqueue->chain.next = mqueue->chain.prev = NULL;
	mqueue->refcnt = 1;
	if (pthread_mutex_init(&mqueue->lock, &mutex_attr) != 0)
		goto error;
	if (pthread_cond_init(&mqueue->cond, &cond_attr) != 0)
		goto error;
	dlist_init(&mqueue->qhead);
	mqueue->closed = false;

	return mqueue;

error:
	SpinLockAcquire(&mqueue_shm_values->lock);
	dlist_push_tail(&mqueue_shm_values->free_queue_list, &mqueue->chain);
	mqueue_shm_values->num_free++;
    mqueue_shm_values->num_active--;	
	SpinLockRelease(&mqueue_shm_values->lock);
	elog(ERROR, "failed on initialization of message queue");
	return NULL;	/* be compiler quiet */
}

/*
 * pgstrom_enqueue_message
 *
 * It enqueues a message towardss OpenCL intermediation server.
 */
bool
pgstrom_enqueue_message(pgstrom_message *message)
{
	pgstrom_queue *mqueue = &mqueue_shm_values->serv_mqueue;
	int		rc;

	pthread_mutex_lock(&mqueue->lock);
	if (mqueue->closed)
	{
		pthread_mutex_unlock(&mqueue->lock);
		return false;
	}

	/*
	 * We assume the message being enqueued in the server message-queue is
	 * already acquired by the server process, not only backend process.
	 * So, we ensure the messages shall not be released during server jobs.
	 * Increment of reference counter prevent unexpected resource free by
	 * elog(ERROR, ...).
	 * 
	 * Please note that the server process may enqueue messages again.
	 * In this case, we don't need to increment reference counter of the
	 * message again (because server process already acquires this message!).
	 * So, it shall be increment only when backend process tries to enqueue
	 * a message.
	 */
	SpinLockAcquire(&message->lock);
	Assert(message->refcnt > 0);
	if (!pgstrom_i_am_clserv)
		message->refcnt++;
	dlist_push_tail(&mqueue->qhead, &message->chain);
	SpinLockRelease(&message->lock);

	/* notification to waiter */
	rc = pthread_cond_signal(&mqueue->cond);
	Assert(rc == 0);
	pthread_mutex_unlock(&mqueue->lock);

	return true;
}

/*
 * pgstrom_reply_message
 *
 * It allows OpenCL intermediation server to enqueue a response message
 * towards the backend process, shouldn't be called by backend itself.
 */
void
pgstrom_reply_message(pgstrom_message *message)
{
	pgstrom_queue  *respq = message->respq;
	int		rc;

	Assert(pgstrom_i_am_clserv);
	pthread_mutex_lock(&respq->lock);
	if (respq->closed)
	{
		pthread_mutex_unlock(&respq->lock);

		/*
		 * In case when response queue is closed, it means nobody waits for
		 * response message, and reference counter of message might be
		 * already decremented by error handler. If current context is the
		 * last one who put this message, we have to release messages.
		 */
		pgstrom_put_message(message);
	}
	else
	{
		SpinLockAcquire(&message->lock);
		/*
		 * Error handler always close the response queue first, then
		 * decrements reference counter of the messages in progress.
		 * So, we can assume the message is acquired by both of backend
		 * and server at the timing when OpenCL server tried to enqueue
		 * response message into an open queue.
		 */
		Assert(message->refcnt > 1);
		message->refcnt--;	/* So, we never call on_release handler */
		dlist_push_tail(&respq->qhead, &message->chain);
		SpinLockRelease(&message->lock);

		/* notification towards waiter */
		rc = pthread_cond_signal(&respq->cond);
		Assert(rc == 0);
		pthread_mutex_unlock(&respq->lock);
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
pgstrom_dequeue_message(pgstrom_queue *mqueue)
{
#define POOLING_INTERVAL	200000000	/* 200msec */
	pgstrom_message *result = NULL;
	struct timeval	basetv;
	struct timespec	timeout;
	ulong	timeleft = ((ulong)pgstrom_mqueue_timeout) * 1000000UL;
	int		rc;

	rc = gettimeofday(&basetv, NULL);
	Assert(rc == 0);
	timeout.tv_sec = basetv.tv_sec;
	timeout.tv_nsec = basetv.tv_usec * 1000UL;

	pthread_mutex_lock(&mqueue->lock);
	for (;;)
	{
		/* dequeue a message from the message queue */
		if (!dlist_is_empty(&mqueue->qhead))
		{
			dlist_node *dnode
				= dlist_pop_head_node(&mqueue->qhead);

			result = dlist_container(pgstrom_message, chain, dnode);
			pthread_mutex_unlock(&mqueue->lock);
			break;
		}
		else if (timeleft == 0)
		{
			pthread_mutex_unlock(&mqueue->lock);
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
			if (timeout.tv_nsec >= 1000000000)
			{
				timeout.tv_sec += timeout.tv_nsec / 1000000000;
				timeout.tv_nsec = timeout.tv_nsec % 1000000000;
			}
			rc = pthread_cond_timedwait(&mqueue->cond,
										&mqueue->lock,
										&timeout);
			Assert(rc == 0 || rc == ETIMEDOUT);

			/*
			 * XXX - we need to have detailed investigation here,
			 * whether this implementation is best design or not.
			 * It assumes backend side blocks until all the messages
			 * are backed.
			 */
			if (pgstrom_i_am_clserv && pgstrom_clserv_exit_pending)
				timeleft = 0;
		}
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
pgstrom_try_dequeue_message(pgstrom_queue *mqueue)
{
	pgstrom_message *result = NULL;

	pthread_mutex_lock(&mqueue->lock);
	if (!dlist_is_empty(&mqueue->qhead))
	{
		dlist_node *dnode
			= dlist_pop_head_node(&mqueue->qhead);

		result = dlist_container(pgstrom_message, chain, dnode);
	}
	pthread_mutex_unlock(&mqueue->lock);

	return result;
}

/*
 * pgstrom_dequeue_server_message
 *
 * dequeue a message from the server message queue
 */
pgstrom_message *
pgstrom_dequeue_server_message(void)
{
	return pgstrom_dequeue_message(&mqueue_shm_values->serv_mqueue);
}

/*
 * pgstrom_cancel_server_loop
 *
 * It shall be called by signal handler of OpenCL server to cancel
 * server loop to wait messages.
 */
void
pgstrom_cancel_server_loop(void)
{
	pgstrom_queue  *mqueue = &mqueue_shm_values->serv_mqueue;

	pthread_cond_broadcast(&mqueue->cond);
}

/*
 * pgstrom_close_server_queue
 *
 * close the server message queue not to receive message any more,
 * and clean-up queued messages.
 */
void
pgstrom_close_server_queue(void)
{
	pgstrom_queue	*svqueue = &mqueue_shm_values->serv_mqueue;
	pgstrom_message	*msg;

	Assert(pgstrom_i_am_clserv);

	pgstrom_close_queue(svqueue);
	/*
	 * Once server message queue is closed, messages being already queued
	 * are immediately replied to the backend with error code.
	 */
	while ((msg = pgstrom_try_dequeue_message(svqueue)) != NULL)
	{
		msg->errcode = StromError_ServerNotReady;
		pgstrom_reply_message(msg);
	}
}

/*
 * pgstrom_close_queue
 *
 * It closes this message queue. Once a message queue got closed, it does not
 * accept any new messages and the queue will be dropped when last message
 * is dequeued or last expected message is tried to enqueue.
 */
void
pgstrom_close_queue(pgstrom_queue *mqueue)
{
	bool	queue_release = false;

	pthread_mutex_lock(&mqueue->lock);
	Assert(!mqueue->closed);
	mqueue->closed = true;

	if (--mqueue->refcnt == 0)
		queue_release = true;
	pthread_mutex_unlock(&mqueue->lock);

	if (queue_release && mqueue != &mqueue_shm_values->serv_mqueue)
	{
		pthread_cond_destroy(&mqueue->cond);
		pthread_mutex_destroy(&mqueue->lock);
		pgstrom_shmem_free(mqueue);
	}
}

/*
 * pgstrom_get_queue
 *
 * It increases reference counter of the supplied message queue.
 */
pgstrom_queue *
pgstrom_get_queue(pgstrom_queue *mqueue)
{
	pthread_mutex_lock(&mqueue->lock);
	Assert(mqueue->refcnt > 0);
	mqueue->refcnt++;
	pthread_mutex_unlock(&mqueue->lock);

	return mqueue;
}

/*
 * pgstrom_put_queue
 *
 * It decrements reference counter of the supplied message queue,
 * and releases the message queue if it is already closed and
 * nobody will reference this queue any more.
 */
void
pgstrom_put_queue(pgstrom_queue *mqueue)
{
	bool	queue_release = false;

	pthread_mutex_lock(&mqueue->lock);
	if (--mqueue->refcnt == 0)
	{
		/*
		 * Someone should close the message queue prior to close
		 * the reference counter reaches to zero.
		 */
		Assert(mqueue->closed);
		queue_release = true;
	}
	pthread_mutex_unlock(&mqueue->lock);

	/*
	 * Once a message queue got refcnt == 0, it is free to reuse, so we
	 * link this queue to the common free_queue_list for later reusing.
	 */
	if (queue_release)
	{
		pthread_cond_destroy(&mqueue->cond);
		pthread_mutex_destroy(&mqueue->lock);
		SpinLockAcquire(&mqueue_shm_values->lock);
		dlist_push_tail(&mqueue_shm_values->free_queue_list,
						&mqueue->chain);
		SpinLockRelease(&mqueue_shm_values->lock);
	}
}

/*
 * pgstrom_put_message
 *
 * It decrements reference counter of message, and may also decrements
 * reference counter of response queue being attached, if this message
 * got released.
 */
void
pgstrom_put_message(pgstrom_message *message)
{
	bool	release_message = false;

	SpinLockAcquire(&message->lock);
	Assert(message->refcnt > 0);
	if (--message->refcnt == 0)
		release_message = true;
	SpinLockRelease(&message->lock);

	/*
	 * Any classes that delivered from pgstrom_message type is responsible
	 * to implement 'cb_release' method to release itself. This handler
	 * also has to unlink message queue being associated with the message
	 * object.
	 */
	if (release_message)
	{
		Assert(message->cb_release != NULL);
		(*message->cb_release)(message);
	}
}

/*
 * pgstrom_startup_mqueue
 *
 * allocation of shared memory for message queue
 */
static void
pgstrom_startup_mqueue(void)
{
	pgstrom_queue  *mqueue;
	bool	found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	mqueue_shm_values = ShmemInitStruct("mqueue_shm_values",
										MAXALIGN(sizeof(*mqueue_shm_values)),
										&found);
	Assert(!found);
	memset(mqueue_shm_values, 0, sizeof(*mqueue_shm_values));
	SpinLockInit(&mqueue_shm_values->lock);
	dlist_init(&mqueue_shm_values->free_queue_list);

	mqueue = &mqueue_shm_values->serv_mqueue;
	mqueue->stag = StromTag_MsgQueue;
	mqueue->refcnt = 0;
	if (pthread_mutex_init(&mqueue->lock, &mutex_attr) != 0)
		elog(ERROR, "failed on pthread_mutex_init for server mqueue");
    if (pthread_cond_init(&mqueue->cond, &cond_attr) != 0)
        elog(ERROR, "failed on pthread_cond_init for server mqueue");
    dlist_init(&mqueue->qhead);
    mqueue->closed = false;
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

	/* aquires shared memory region */
	RequestAddinShmemSpace(MAXALIGN(sizeof(*mqueue_shm_values)));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_mqueue;
}
