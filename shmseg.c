/*
 * shmseg.c
 *
 * Routines to manage shared memory segment
 * --
 * Copyright 2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include <limits.h>
#include <semaphore.h>
#include <unistd.h>

#define SHMSEG_BLOCK_MAGIC		0xdeadbeaf

typedef struct ShmsegBlock {
	uint32		magic;
	struct ShmsegBlock  *prev_block;	/* start block, if null */
	struct ShmsegBlock  *next_block;	/* end block, if null */
	SHM_QUEUE	list;		/* list entry of free_list or used_list */
	Size		size;		/* size of this block includes header */
	int			refcnt;		/* free block, if refcnt == 0 */
	Datum		data[0];
} ShmsegBlock;

typedef struct {
	Size			total_size;
	Size			used_size;
	Size			free_size;
	SHM_QUEUE		used_list;
	SHM_QUEUE		free_list;
	pthread_mutex_t	lock;
	ShmsegBlock		block[0];
} ShmsegHead;

typedef struct {
	int				refcnt;
	sem_t			qsem;
	pthread_mutex_t	qlock;
	SHM_QUEUE		qhead;
} ShmsegQueue;

#define container_of(ptr, type, field)			\
	((void *)((uintptr_t)(ptr) - offsetof(type, field)))

/*
 * Local declaration
 */
static int						pgstrom_chunk_buffer_size;
static ShmsegHead			   *shmseg_head;
static pthread_mutexattr_t		shmseg_mutex_attr;
static shmem_startup_hook_type	shmem_startup_hook_next;

SHM_QUEUE *
pgstrom_shmqueue_create(void)
{
	ShmsegQueue	*shmq;

	shmq = pgstrom_shmseg_create(sizeof(ShmsegQueue));
	if (!shmq)
		return NULL;

	shmq->refcnt = 1;
	if (sem_init(&shmq->qsem, 1, 0) != 0)
		goto error_1;
	if (pthread_mutex_init(&shmq->qlock, &shmseg_mutex_attr) != 0)
		goto error_2;
	SHMQueueInit(&shmq->qhead);

	return &shmq->qhead;

error_2:
	sem_destroy(&shmq->qsem);
error_1:
	pgstrom_shmseg_release(shmq);
	return NULL;
}

void
pgstrom_shmqueue_retain(SHM_QUEUE *shmqhead)
{
	ShmsegQueue *shmq = container_of(shmqhead, ShmsegQueue, qhead);

	Assert(shmqhead == &shmq->qhead);
	pthread_mutex_lock(&shmq->qlock);
	shmq->refcnt++;
	pthread_mutex_unlock(&shmq->qlock);
}

void
pgstrom_shmqueue_release(SHM_QUEUE *shmqhead)
{
	ShmsegQueue *shmq = container_of(shmqhead, ShmsegQueue, qhead);

	Assert(shmqhead == &shmq->qhead);
	pthread_mutex_lock(&shmq->qlock);
	if (--shmq->refcnt == 0)
	{
		
		
		
		
	}
	pthread_mutex_unlock(&shmq->qlock);
}

bool
pgstrom_shmqueue_enqueue(SHM_QUEUE *shmqhead, SHM_QUEUE *item)
{
	ShmsegQueue *shmq = container_of(shmqhead, ShmsegQueue, qhead);

	Assert(shmqhead == &shmq->qhead);
	pthread_mutex_lock(&shmq->qlock);
	SHMQueueInsertBefore(&shmq->qhead, item);
	sem_post(&shmq->qsem);
	pthread_mutex_unlock(&shmq->qlock);

	return true;
}

SHM_QUEUE *
pgstrom_shmqueue_dequeue(SHM_QUEUE *shmqhead, bool wait)
{
	ShmsegQueue	*shmq = container_of(shmqhead, ShmsegQueue, qhead);
	SHM_QUEUE	*item;

	Assert(shmqhead == &shmq->qhead);
retry:
	if ((wait ?
		 sem_wait(&shmq->qsem) :
		 sem_trywait(&shmq->qsem)) == 0)
	{
		pthread_mutex_lock(&shmq->qlock);
		Assert(!SHMQueueEmpty(&shmq->qhead));
		item = shmq->qhead.next;
		SHMQueueDelete(item);
		pthread_mutex_unlock(&shmq->qlock);

		return item;
	}
	else if (errno == EINTR)
		goto retry;

	return NULL;
}

void *
pgstrom_shmseg_create(Size size)
{
	ShmsegBlock	   *block;
	ShmsegBlock	   *block_new;
	Size			required = MAXALIGN(offsetof(ShmsegBlock, data) + size);

	pthread_mutex_lock(&shmseg_head->lock);

	block = (ShmsegBlock *)SHMQueueNext(&shmseg_head->free_list,
										&shmseg_head->free_list,
										offsetof(ShmsegBlock, list));
	while (block)
	{
		Assert(block->refcnt == 0);

		/*
		 * Size of the current free block is not enough to assign a shared
		 * memory block with required size, so we try to check next free
		 * block.
		 */
		if (block->size < required)
		{
			block = (ShmsegBlock *)SHMQueueNext(&shmseg_head->free_list,
												&block->list,
												offsetof(ShmsegBlock, list));
			continue;
		}

		/*
		 * In case of the current free block size is similar to the required
		 * size, we assign whole of the block to the requirement to avoid
		 * management overhead on such a small flaction.
		 *
		 * Otherwise, we split off the free block into two.
		 */
		if (block->size <= required + getpagesize() / 2)
		{
			SHMQueueDelete(&block->list);
			block->refcnt = 1;
			SHMQueueInsertAfter(&shmseg_head->used_list, &block->list);

			shmseg_head->used_size += block->size;
			shmseg_head->free_size -= block->size;
		}
		else
		{
			SHMQueueDelete(&block->list);

			block_new = (ShmsegBlock *)(((uintptr_t) block) + required);
			block_new->magic = SHMSEG_BLOCK_MAGIC;
			block_new->prev_block = block;
			block_new->next_block = block->next_block;
			if (block_new->next_block)
				block_new->next_block->prev_block = block_new;
			block_new->size = block->size - required;
			block_new->refcnt = 0;
			SHMQueueInsertAfter(&shmseg_head->free_list, &block_new->list);

			block->next_block = block_new;
			block->size = required;
			block->refcnt = 1;
			SHMQueueInsertAfter(&shmseg_head->used_list, &block->list);

			shmseg_head->used_size += required;
			shmseg_head->free_size -= required;
		}
		break;
	}
	pthread_mutex_unlock(&shmseg_head->lock);

	return (!block ? NULL : (void *)block->data);
}

void
pgstrom_shmseg_retain(void *ptr)
{
	ShmsegBlock	*block = container_of(ptr, ShmsegBlock, data);

	Assert(block->magic == SHMSEG_BLOCK_MAGIC);
	pthread_mutex_lock(&shmseg_head->lock);
	block->refcnt++;
	pthread_mutex_unlock(&shmseg_head->lock);
}

void
pgstrom_shmseg_release(void *ptr)
{
	ShmsegBlock *temp;
	ShmsegBlock	*block = container_of(ptr, ShmsegBlock, data);

	Assert(block->magic == SHMSEG_BLOCK_MAGIC);
	pthread_mutex_lock(&shmseg_head->lock);
	if (--block->refcnt == 0)
	{
		SHMQueueDelete(&block->list);
		shmseg_head->used_size -= block->size;
		shmseg_head->free_size += block->size;

		/*
		 * Does the neighboring blocks available to merge?
		 */
		temp = block->next_block;
		if (temp && temp->refcnt == 0)
		{
			SHMQueueDelete(&temp->list);

			Assert(temp->prev_block == block);
			block->next_block = temp->next_block;
			if (temp->next_block)
				temp->next_block->prev_block = block;
			block->size += temp->size;
		}

		temp = block->prev_block;
		if (temp && temp->refcnt == 0)
		{
			SHMQueueDelete(&temp->list);

			Assert(temp->next_block == block);
			temp->next_block = block->next_block;
			if (block->next_block)
				block->next_block->prev_block = temp;
			temp->size += block->size;

			block = temp;
		}

		SHMQueueInsertAfter(&shmseg_head->free_list, &block->list);
	}
	pthread_mutex_unlock(&shmseg_head->lock);
}

static void
pgstrom_shmseg_startup(void)
{
	ShmsegBlock	*block;
	Size	chunk_bufsz = (pgstrom_chunk_buffer_size << 20);
	bool	found;

	shmseg_head = ShmemInitStruct("Shared chunk buffer of PG-Strom",
								  sizeof(ShmsegHead) + chunk_bufsz, &found);
	Assert(!found);

	/* Init PgStromShmsegHead field */
	shmseg_head->total_size = chunk_bufsz;
	shmseg_head->used_size = 0;
	shmseg_head->free_size = chunk_bufsz;
	SHMQueueInit(&shmseg_head->used_list);
	SHMQueueInit(&shmseg_head->free_list);
	if (pthread_mutex_init(&shmseg_head->lock, &shmseg_mutex_attr) != 0)
		elog(ERROR, "failed to init mutex lock");

	/* Init ShmsegBlock as an empty big block */
	block = &shmseg_head->block[0];
	block->prev_block = NULL;
	block->next_block = NULL;
	SHMQueueInsertAfter(&shmseg_head->free_list, &block->list);
	block->size = chunk_bufsz - offsetof(ShmsegHead, block);
	block->refcnt = 0;

	/* launch OpenCL computing server */
	pgstrom_opencl_startup(block, chunk_bufsz);
}

void
pgstrom_shmseg_init(void)
{
	/* Prepare mutex-attribute on shared memory segment */
	if (pthread_mutexattr_init(&shmseg_mutex_attr) != 0 ||
		pthread_mutexattr_setpshared(&shmseg_mutex_attr,
									 PTHREAD_PROCESS_SHARED) != 0)
		elog(ERROR, "failed to init mutex attribute");

	/* GUC */
	DefineCustomIntVariable("pg_strom.chunk_buffer_size",
							"size of chunk buffer acquired on shmem [MB]",
							NULL,
							&pgstrom_chunk_buffer_size,
							64,		/*   64MB */
							64,		/*   64MB */
							INT_MAX,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);

	/* Acquire share memory segment for chunk buffer */
	RequestAddinShmemSpace((pgstrom_chunk_buffer_size << 20) + getpagesize());
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_shmseg_startup;
}
