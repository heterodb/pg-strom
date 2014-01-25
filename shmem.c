/*
 * shmem.c
 *
 * The entrypoint of PG-Strom extension.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "fmgr.h"
#include "lib/ilist.h"
#include "libpq/md5.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include <limits.h>

typedef struct
{
	slock_t			lock;
	dlist_head		free_list;
	Size			curr_usage;
	Size			max_usage;
	void		   *base_address;
	dlist_node		blocks[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_shmem_head;

typedef struct
{
	slock_t			lock;
	dlist_head		vacancy_blocks;
	dlist_head		full_blocks;
	char			name[NAMEDATALEN];
	char			magic[16];	/* MD5 digest */
	Size			unit_size;
	long			num_allocated;	/* objects already allocated */
	long			num_active;		/* active objects in this block */
} pgstrom_shmem_slab;

typedef struct
{
	char			magic[16];	/* MD5 digest */
	pgstrom_shmem_slab *slab;
	long			num_active;	/* active objects in this block */
	dlist_node		chain;
	dlist_head		free_list;
	dlist_node		objects[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_shmem_slab_block;

static shmem_startup_hook_type shmem_startup_hook_next;
static int	pgstrom_shmem_numblocks;
static Size	pgstrom_shmem_blocksize;
static Size pgstrom_shmem_totalsize;
static pgstrom_shmem_head  *shmem_head;


#define SLAB_NUM_OBJECTS_PER_BLOCK(slab)					\
	((pgstrom_shmem_blocksize -								\
	  offsetof(pgstrom_shmem_slab_block, objects[0])) /		\
	 (sizeof(dlist_node) + MAXALIGN((slab)->unit_size)))

static void
pgstrom_slab_init(pgstrom_shmem_slab *slab, const char *name, Size size)
{
	if (strlen(name) >= NAMEDATALEN)
		elog(ERROR, "%s: slab name (\"%s\") too long", __FUNCTION__, name);

	SpinLockInit(&slab->lock);
	dlist_init(&slab->vacancy_blocks);
	dlist_init(&slab->full_blocks);
	strncpy(slab->name, name, NAMEDATALEN);
	pg_md5_binary(slab->name, strlen(slab->name), slab->magic);
	slab->unit_size = size;
	slab->num_allocated = 0;
	slab->num_active = 0;
}

static void *
pgstrom_slab_alloc(pgstrom_shmem_slab *slab)
{
	pgstrom_shmem_slab_block *sblock;
	dlist_node *dnode;
	int			i, nobjects = SLAB_NUM_OBJECTS_PER_BLOCK(slab);
	Size		baseaddr;
	void	   *result = NULL;

	SpinLockAcquire(&slab->lock);
	if (dlist_is_empty(&slab->vacancy_blocks))
	{
		Assert(slab->num_active == slab->num_allocated);
		sblock = pgstrom_shmem_block_alloc();
		if (!sblock)
			goto out_unlock;

		memcpy(sblock->magic, slab->magic, sizeof(sblock->magic));
		sblock->slab = slab;
		sblock->num_active = 0;
		for (i=0; i < nobjects; i++)
			dlist_push_tail(&sblock->free_list, &sblock->objects[i]);

		dlist_push_tail(&slab->vacancy_blocks, &sblock->chain);
		slab->num_allocated += nobjects;
	}

	/* pick up a block from vacancy_blocks */
	sblock = dlist_container(pgstrom_shmem_slab_block, chain,
							 dlist_head_node(&slab->vacancy_blocks));
	Assert(memcmp(slab->magic, sblock->magic, sizeof(slab->magic)) == 0);
	Assert(!dlist_is_empty(&sblock->free_list));

	dnode = dlist_pop_head_node(&sblock->free_list);
	i = dnode - &sblock->objects[0];
	baseaddr = (Size)&sblock->objects[nobjects];

	result = (void *)(baseaddr + MAXALIGN(slab->unit_size) * i);
	slab->num_active++;
	sblock->num_active++;

	/*
	 * Move the slab_block to the list of blocks that has no room to
	 * allocate objects towards upcoming request.
	 */
	if (dlist_is_empty(&sblock->free_list))
	{
		dlist_delete(&sblock->chain);
		dlist_push_tail(&slab->full_blocks, &sblock->chain);
	}

	Assert(slab->num_active <= slab->num_allocated);
out_unlock:
	SpinLockRelease(&slab->lock);

	return result;
}

static void
pgstrom_slab_free(void *object)
{
	pgstrom_shmem_slab		   *slab;
	pgstrom_shmem_slab_block   *sblock;
	int			i, nobjects;
	bool		full_block;
	Size		baseaddr;

	/*
	 * All the shared memory block is aligned to pgstrom_shmem_blocksize.
	 * So, clearing lower bits allows to get pgstrom_shmem_slab_block that
	 * owns the slab object being supplied.
	 */
	sblock = (pgstrom_shmem_slab_block *)
		((Size)object & ~(pgstrom_shmem_blocksize - 1));
	slab = sblock->slab;
	Assert(memcmp(slab->magic, sblock->magic, sizeof(slab->magic)) == 0);

	nobjects = SLAB_NUM_OBJECTS_PER_BLOCK(slab);
	baseaddr = (Size)&sblock->objects[nobjects];
	i = ((Size)object - baseaddr) / MAXALIGN(slab->unit_size);

	SpinLockAcquire(&slab->lock);
	full_block = dlist_is_empty(&sblock->free_list);

	dlist_push_head(&sblock->free_list, &sblock->objects[i]);
	sblock->num_active--;
	slab->num_active--;

	/*
	 * If this release makes this block from full to vacancy, it also needs
	 * to be re-chained to find out free slab object on the next allocation
	 * request.
	 */
	if (full_block)
	{
		dlist_delete(&sblock->chain);
		dlist_push_head(&slab->vacancy_blocks, &sblock->chain);
	}

	/*
	 * If this release makes all the objects in this block free status, we
	 * release this block itself.
	 */
	if (sblock->num_active == 0)
	{
		dlist_delete(&sblock->chain);
		pgstrom_shmem_block_free(sblock);
		slab->num_allocated -= nobjects;
	}
	SpinLockRelease(&slab->lock);
}

Datum
pgstrom_system_slabinfo(PG_FUNCTION_ARGS)
{

	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(pgstrom_system_slabinfo);

Size
pgstrom_shmem_get_blocksize(void)
{
	return pgstrom_shmem_blocksize;
}

void *
pgstrom_shmem_block_alloc(void)
{
	dlist_node *dnode = NULL;

	SpinLockAcquire(&shmem_head->lock);
	if (!dlist_is_empty(&shmem_head->free_list))
	{
		dnode = dlist_pop_head_node(&shmem_head->free_list);
		shmem_head->curr_usage += pgstrom_shmem_blocksize;
		shmem_head->max_usage = Max(shmem_head->max_usage,
									shmem_head->curr_usage);
	}
	SpinLockRelease(&shmem_head->lock);

	if (dnode)
	{
		int		index = dnode - shmem_head->blocks;

		dnode->prev = dnode->next = NULL;
		return (void *)((Size)shmem_head->base_address +
						index * pgstrom_shmem_blocksize);
	}
	return NULL;
}

void
pgstrom_shmem_block_free(void *address)
{
	dlist_node *dnode;
	int			index;

	Assert((Size)address % pgstrom_shmem_blocksize == 0);
	index = (((Size)address - (Size)shmem_head->base_address)
			 / pgstrom_shmem_blocksize);
	Assert(index >= 0 && index < pgstrom_shmem_numblocks);

	dnode = &shmem_head->blocks[index];
	Assert(!dnode->prev && !dnode->next);

	SpinLockAcquire(&shmem_head->lock);
	dlist_push_head(&shmem_head->free_list, dnode);
	shmem_head->curr_usage -= pgstrom_shmem_blocksize;
	SpinLockRelease(&shmem_head->lock);
}

static void
pgstrom_setup_shmem(void)
{
	int		i;
	bool	found;

	shmem_head = ShmemInitStruct("PG-Strom: shared memory segment",
								 offsetof(pgstrom_shmem_head,
										  blocks[pgstrom_shmem_numblocks]) +
								 pgstrom_shmem_totalsize +
								 pgstrom_shmem_blocksize,
								 &found);
	Assert(!found);

	/* init pgstrom_shmem_head fields */
	SpinLockInit(&shmem_head->lock);
	dlist_init(&shmem_head->free_list);
	shmem_head->curr_usage = 0;
	shmem_head->max_usage = 0;
	shmem_head->base_address
		= (void *)TYPEALIGN(pgstrom_shmem_blocksize,
							&shmem_head->blocks[pgstrom_shmem_numblocks]);
	for (i=0; i < pgstrom_shmem_numblocks; i++)
		dlist_push_tail(&shmem_head->free_list, &shmem_head->blocks[i]);

	/*
	 *
	 * TODO: initialize slab allocator
	 *
	 */





}

void
pgstrom_init_shmem(void)
{
	static int	shmem_blocksize;
	static int	shmem_totalsize;

	/*
	 * Definition of GUC variables for shared memory management
	 */
	DefineCustomIntVariable("pgstrom.shmem_blocksize",
							"size per shared memory allocation block [MB]",
							NULL,
							&shmem_blocksize,
							4,	/* 4MB */
							1,	/* 1MB */
							INT_MAX,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	if ((shmem_blocksize & ~(shmem_blocksize - 1)) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("\"pgstrom.shmem_blocksize\" must be power of 2")));

	DefineCustomIntVariable("pgstrom.shmem_totalsize",
							"total size of shared memory segment block [MB]",
							NULL,
							&shmem_totalsize,
							2048,		/* 2GB */
							64,			/* 64MB */
							INT_MAX,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	if (shmem_totalsize < shmem_blocksize * 128)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("\"pgstrom.shmem_totalsize\" too small")));
	if (shmem_totalsize % shmem_blocksize != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("\"pgstrom.shmem_totalsize\" must be multiple of \"pgstrom.shmem_blocksize\"")));

	pgstrom_shmem_numblocks = shmem_totalsize / shmem_blocksize;
	pgstrom_shmem_blocksize = shmem_blocksize << 20;
	pgstrom_shmem_totalsize = shmem_totalsize << 20;

	/* Acquire shared memory segment */
	RequestAddinShmemSpace(offsetof(pgstrom_shmem_head,
									blocks[pgstrom_shmem_numblocks]) +
						   pgstrom_shmem_totalsize +
						   pgstrom_shmem_blocksize);
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_setup_shmem;
}
