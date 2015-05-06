/*
 * mqueue.c
 *
 * Routines for inter-process communication via message queues
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "postgres.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
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
	dlist_head		blocks_list;
	dlist_head		free_queue_list;
	uint32			num_free;
	uint32			num_active;
	pgstrom_queue	serv_mqueue;	/* queue to OpenCL server */
} *mqueue_shm_values;

/* number of message queues per block */
#define MQUEUES_PER_BLOCK								\
	((SHMEM_BLOCKSZ - SHMEM_ALLOC_COST					\
	  - sizeof(dlist_node))	/ sizeof(pgstrom_queue))

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

	/*
	 * server should not create a message queue. it is constructed on
	 * starting-up time
	 */
	Assert(!pgstrom_i_am_clserv);

	SpinLockAcquire(&mqueue_shm_values->lock);
	if (dlist_is_empty(&mqueue_shm_values->free_queue_list))
	{
		dlist_node	   *block;
		pgstrom_queue  *mqueues;
		int		i;

		block = pgstrom_shmem_alloc(sizeof(dlist_node) +
									sizeof(pgstrom_queue) *
									MQUEUES_PER_BLOCK);
		if (!block)
		{
			SpinLockRelease(&mqueue_shm_values->lock);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("out of shared memory")));
		}
		dlist_push_tail(&mqueue_shm_values->blocks_list, block);

		mqueues = (pgstrom_queue *)(block + 1);
		for (i=0; i < MQUEUES_PER_BLOCK; i++)
		{
			if (pthread_mutex_init(&mqueues[i].lock, &mutex_attr) != 0 ||
				pthread_cond_init(&mqueues[i].cond, &cond_attr) != 0)
			{
				/* recovery when initialization got failed on the way */
				while (--i >= 0)
					dlist_delete(&mqueues[i].chain);
				dlist_delete(block);
				SpinLockRelease(&mqueue_shm_values->lock);
				pgstrom_shmem_free(block);
				elog(ERROR, "failed on initialization of message queue");
			}
			mqueues[i].sobj.stag = StromTag_MsgQueue;
			dlist_push_tail(&mqueue_shm_values->free_queue_list,
							&mqueues[i].chain);
		}
		mqueue_shm_values->num_free += MQUEUES_PER_BLOCK;
	}
	Assert(mqueue_shm_values->num_free > 0);
	mqueue_shm_values->num_free--;
	mqueue_shm_values->num_active++;
	dnode = dlist_pop_head_node(&mqueue_shm_values->free_queue_list);
	mqueue = dlist_container(pgstrom_queue, chain, dnode);

	/* mark it as active one */
	Assert(StromTagIs(mqueue, MsgQueue));
	memset(&mqueue->chain, 0, sizeof(dlist_node));
	mqueue->owner = MyProc;
	mqueue->refcnt = 1;
	dlist_init(&mqueue->qhead);
	mqueue->closed = false;
	SpinLockRelease(&mqueue_shm_values->lock);

	return mqueue;
}

/*
 * pgstrom_enqueue_message
 *
 * It enqueues a message towardss OpenCL intermediation server.
 */
bool
pgstrom_enqueue_message(pgstrom_message *message)
{
	pgstrom_queue  *mqueue = &mqueue_shm_values->serv_mqueue;
	int		rc;

	pthread_mutex_lock(&mqueue->lock);
	if (mqueue->closed)
	{
		pthread_mutex_unlock(&mqueue->lock);
		return false;
	}

	/* performance monitoring */
	if (message->pfm.enabled)
		gettimeofday(&message->pfm.tv, NULL);

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
	Assert(respq != &mqueue_shm_values->serv_mqueue);
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
		/* performance monitoring */
		if (message->pfm.enabled)
			gettimeofday(&message->pfm.tv, NULL);

		SpinLockAcquire(&message->lock);
		if (message->refcnt > 1)
		{
			message->refcnt--;	/* we never call on_release handler here */
			dlist_push_tail(&respq->qhead, &message->chain);
			SpinLockRelease(&message->lock);

			/* notification towards the waiter process */
			rc = pthread_cond_signal(&respq->cond);
			Assert(rc == 0);
			pthread_mutex_unlock(&respq->lock);
		}
		else
		{
			message->refcnt--;
			Assert(message->refcnt == 0);
			SpinLockRelease(&message->lock);

			pthread_mutex_unlock(&respq->lock);
			/*
			 * Usually, release handler of message object will detach
			 * a response message queue also. It needs to acquire a lock
			 * on the message queue to touch reference counter, so we
			 * have to release the lock prior to invocation of release
			 * handler.
			 */
			Assert(message->cb_release != NULL);
			(*message->cb_release)(message);
		}
		SetLatch(&respq->owner->procLatch);
	}
}

/*
 * pgstrom_sync_dequeue_message
 *
 * It fetches a message from the message queue. If empty, it waits for new
 * messages will come, or returns NULL if it exceeds timeout or it got
 * a signal being pending.
 */
static pgstrom_message *
pgstrom_sync_dequeue_message(pgstrom_queue *mqueue)
{
#define POOLING_INTERVAL	200000000	/* 200msec */
	pgstrom_message *result = NULL;
	struct timeval	basetv;
	struct timespec	timeout;
	cl_ulong		timeleft = ((cl_ulong)pgstrom_mqueue_timeout) * 1000000UL;
	int				rc;

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
 * pgstrom_dequeue_message
 *
 * dequeue a message from backend responding queue
 */
pgstrom_message *
pgstrom_dequeue_message(pgstrom_queue *mqueue)
{
	pgstrom_message	   *msg;
	struct timeval		tv;

	Assert(!pgstrom_i_am_clserv);
	msg = pgstrom_sync_dequeue_message(mqueue);
	if (msg && msg->pfm.enabled)
	{
		gettimeofday(&tv, NULL);
		msg->pfm.time_in_recvq += timeval_diff(&msg->pfm.tv, &tv);
	}
	return msg;
}

/*
 * pgstrom_dequeue_server_message
 *
 * dequeue a message from the server message queue
 */
pgstrom_message *
pgstrom_dequeue_server_message(void)
{
	pgstrom_message	   *msg;
	struct timeval		tv;

	Assert(pgstrom_i_am_clserv);
	msg = pgstrom_sync_dequeue_message(&mqueue_shm_values->serv_mqueue);
	if (msg && msg->pfm.enabled)
	{
		gettimeofday(&tv, NULL);
		msg->pfm.time_in_sendq += timeval_diff(&msg->pfm.tv, &tv);
	}
	return msg;
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
	{
		queue_release = true;
		mqueue->owner = NULL;
	}
	pthread_mutex_unlock(&mqueue->lock);

	if (queue_release && mqueue != &mqueue_shm_values->serv_mqueue)
	{
		SpinLockAcquire(&mqueue_shm_values->lock);
		mqueue_shm_values->num_active--;
		mqueue_shm_values->num_free++;
        dlist_push_tail(&mqueue_shm_values->free_queue_list,
                        &mqueue->chain);
        SpinLockRelease(&mqueue_shm_values->lock);
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
		mqueue->owner = NULL;
	}
	pthread_mutex_unlock(&mqueue->lock);

	/*
	 * Once a message queue got refcnt == 0, it is free to reuse, so we
	 * link this queue to the common free_queue_list for later reusing.
	 */
	if (queue_release)
	{
		SpinLockAcquire(&mqueue_shm_values->lock);
		mqueue_shm_values->num_active--;
		mqueue_shm_values->num_free++;
		dlist_push_tail(&mqueue_shm_values->free_queue_list,
						&mqueue->chain);
		SpinLockRelease(&mqueue_shm_values->lock);
	}
}

/*
 * pgstrom_put_message
 *
 * It decrements reference counter of the message, and calls its release
 * handle if this reference counter got reached to zero.
 * We assume the release handler also decrements reference counter of
 * the response queue being attached; that eventually releases the queue
 * also.
 * It returns true, if message was actually released.
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
 * pgstrom_init_message
 *
 * utility routine to initialize common message header
 */
void
pgstrom_init_message(pgstrom_message *message,
					 StromTag stag,
					 pgstrom_queue *respq,
					 void (*cb_process)(pgstrom_message *message),
					 void (*cb_release)(pgstrom_message *message),
					 bool perfmon_enabled)
{
	memset(message, 0, sizeof(pgstrom_message));
	message->sobj.stag = stag;
	SpinLockInit(&message->lock);
	message->refcnt = 1;
	if (respq != NULL)
		message->respq = pgstrom_get_queue(respq);
	message->cb_process = cb_process;
	message->cb_release = cb_release;
	message->pfm.enabled = perfmon_enabled;
}

/*
 * pgstrom_mqueue_info
 *
 * shows all the message queues being already acquired as SQL funcion
 */
typedef struct {
	void	   *mqueue;
	pid_t		owner_pid;
	char		state;	/* 'a' = active, 'c' = closed, 'f' = free*/
	int			refcnt;
} mqueue_info;

Datum
pgstrom_mqueue_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	mqueue_info	   *mq_info;
	HeapTuple		tuple;
	Datum			values[4];
	bool			isnull[4];
	char			buf[256];
	int				i;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		dlist_iter		iter;
		List		   *mq_list = NIL;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(4, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "mqueue",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "owner",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "state",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "refcnt",
						   INT4OID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		SpinLockAcquire(&mqueue_shm_values->lock);
		PG_TRY();
		{
			/* server mqueue */
			mq_info = palloc(sizeof(mqueue_info));
			mq_info->mqueue = &mqueue_shm_values->serv_mqueue;
			mq_info->owner_pid = 0;		/* usuall NULL */
			mq_info->state = mqueue_shm_values->serv_mqueue.closed ? 'c' : 'a';
			mq_info->refcnt = mqueue_shm_values->serv_mqueue.refcnt;
			mq_list = lappend(mq_list, mq_info);

			/* backend mqueues */
			dlist_foreach(iter, &mqueue_shm_values->blocks_list)
			{
				pgstrom_queue  *mqueues = (pgstrom_queue *)(iter.cur + 1);

				for (i=0; i < MQUEUES_PER_BLOCK; i++)
				{
					mq_info = palloc(sizeof(mqueue_info));
					mq_info->mqueue = &mqueues[i];
					mq_info->owner_pid = mqueues[i].owner->pid;
					if (!mqueues[i].chain.prev || !mqueues[i].chain.next)
						mq_info->state = (mqueues[i].closed ? 'c' : 'a');
					else
						mq_info->state = 'f';

					pthread_mutex_lock(&mqueues[i].lock);
					mq_info->refcnt = mqueues[i].refcnt;
					pthread_mutex_unlock(&mqueues[i].lock);

					mq_list = lappend(mq_list, mq_info);
				}
			}
		}
		PG_CATCH();
		{
			SpinLockRelease(&mqueue_shm_values->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(&mqueue_shm_values->lock);

		fncxt->user_fctx = mq_list;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
		SRF_RETURN_DONE(fncxt);

	mq_info = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	snprintf(buf, sizeof(buf), "%p", mq_info->mqueue);
	values[0] = CStringGetTextDatum(buf);
	values[1] = Int32GetDatum(mq_info->owner_pid);
	snprintf(buf, sizeof(buf), "%s",
			 (mq_info->state == 'a' ? "active" :
			  (mq_info->state == 'c' ? "closed" :
			   (mq_info->state == 'f' ? "free" : "unknown"))));
	values[2] = CStringGetTextDatum(buf);
	values[3] = Int32GetDatum(mq_info->refcnt);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_mqueue_info);

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
	memset(mqueue, 0, sizeof(pgstrom_queue));
	mqueue->sobj.stag = StromTag_MsgQueue;
	mqueue->owner = NULL;
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
	DefineCustomIntVariable("pg_strom.mqueue_timeout",
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
