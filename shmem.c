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
#include "lib/ilist.h"
#include "libpq/md5.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"
#include <limits.h>

#define SHMEM_BLOCKSZ_BITS_MAX	34			/* 16GB */
#define SHMEM_BLOCKSZ_BITS		22			/*  4MB */
#define SHMEM_BLOCKSZ			(1UL << SHMEM_BLOCKSZ_BITS_MIN)
#define SHMEM_BLOCK_MAGIC		0xDEADBEAF

typedef union
{
	dlist_node		chain;		/* to be chained free_list of shmem_zone */
	struct {
		void	   *nullmark;	/* NULL if active block */
		Size		size;		/* size of allocation block */
	} active;
} shmem_block;

typedef struct
{
	slock_t		lock;
	long		num_blocks;		/* number of total blocks */
	long		num_active;		/* number of active blocks */
	dlist_head	free_list[SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS + 1];
	void	   *base_address;
	void	   *private;
	shmem_block	blocks[FLEXIBLE_ARRAY_MEMBER];
} shmem_zone;

typedef struct
{
	bool		is_ready;
	int			num_zones;
	shmem_zone	zones[FLEXIBLE_ARRAY_MEMBER];
} shmem_head;


#define SHMCHUNK_MAGIC_FREE		0xF9EEA9EA
#define SHMCHUNK_MAGIC_ACTIVE	0xAC715AEA

typedef struct
{
	pid_t		owner;
	slock_t		lock;
	dlist_head	block_list;
	Size		total_active;
	Size		total_alloc;
	long		num_chunks;
	char		md5[16];
	char		name[FLEXIBLE_ARRAY_MEMBER];
} shmem_context;

typedef struct
{
	shmem_context  *context;
	char			md5[16];
	dlist_node		chain;
	dlist_head		chunk_list;
	dlist_head		free_list;

	dlist_node		first_chunk;
} shmem_bunch;

typedef struct
{
	dlist_node		chunk_chain;	/* linked to chunk_list */
	union {
		dlist_node	free_chain;		/* linked to free_list */
		struct {
			void   *nullmark;
			Size	size;
		} active;
	};
	char			userdata[FLEXIBLE_ARRAY_MEMBER];
} shmem_chunk;

#define SHMEM_CHUNK_MAGIC(chunk)										\
	*((uint32 *)(!(chunk)->active.nullmark								\
				 ? (char *)(chunk) + (chunk)->active.size				\
				 : (char *)(chunk)->chunk_chain.next - sizeof(uint32))

/* static variables */
static shmem_startup_hook_type shmem_startup_hook_next;
static Size		pgstrom_shmem_totalsize;
static int		pgstrom_shmem_maxzones;
static pgstrom_shmem_head  *shmem_head;

pgstrom_shmem_context  *TopShmemContext = NULL;

pgstrom_shmem_context *
pgstrom_shmem_context_create(const char *name)
{
	pgstrom_shmem_context  *context;
	int		namelen = strlen(name);

	context = pgstrom_shmem_alloc(TopShmemContext, 
								  sizeof(pgstrom_shmem_context) +
								  Max(namelen+1, NAMEDATALEN));
	if (!context)
		return NULL;

	context->owner = getpid();
	SpinLockInit(&context->lock);
	dlist_init(&context->block_list);
	context->num_chunks = 0;
	context->num_blocks = 0;
	pg_md5_binary(name, namelen, context->md5);
	strcpy(context->name, name);

	return context;
}

void *
pgstrom_shmem_alloc(pgstrom_shmem_context *context, size)
{
	pgstrom_shmem_bunch	*sbunch;
	pgstrom_shmem_chunk *schunk;
	dlist_node *dnode;
	dlist_iter	iter1;
	dlist_iter	iter2;
	Size		length;
	Size		required;
	void	   *address = NULL;

	required = MAXALIGN(offsetof(pgstrom_shmem_chunk,
								 userdata[0]) + size + sizeof(uint32));
	SpinLockAcquire(&context->lock);
retry:
	dlist_foreach(iter1, &context->block_list)
	{
		sbunch = dlist_container(pgstrom_shmem_bunch, chain, iter1.cur);

		Assert((Size)sbunch % SHMEM_BLOCKSZ == 0);
		Assert(memcmp(context->md5, sbunch->md5, 16) == 0);

		dlist_foreach(iter2, &sbunch->free_list)
		{
			int		padlen;

			schunk = dlist_container(pgstrom_shmem_chunk, free_chain,
									 iter2.cur);
			dnode = dlist_next_node(&sbunch->chunk_list,
									&schunk->chunk_chain);
			length = (char *)dnode - (char *)schunk->userdata;
			Assert(*((uint32 *)dnode - 1) == SHMCHUNK_MAGIC_FREE);

			/* Is this free chunk has enough space to store it? */
			if (length < required)
				continue;
			/* Split this chunk, if it has enough space to store */
			padlen = MAXALIGN(offsetof(pgstrom_shmem_chunk,
									   userdata[0]) + sizeof(uint32));
			if (length >= required + padlen)
			{
				pgstrom_shmem_chunk	*nchunk
					= (pgstrom_shmem_chunk *)((char *)schunk + required);
				dlist_insert_after(&schunk->chunk_chain,
								   &nchunk->chunk_chain);
			}
			dlist_delete(&schunk->free_chain);
			schunk->nullmark = NULL;
			schunk->size = size;
			*((uint32 *)(&schunk->userdata[size])) = SHMEM_MAGIC_ACTIVE_CHUNK;

			address = schunk->userdata;
			goto out_unlock;
		}
	}
	/* no room to allocate a chunk, assign a new block */
	cbunch = pgstrom_shmem_block_alloc(offsetof(pgstrom_shmem_bunch,
												first_chunk) +
									   required +
									   sizeof(dlist_node));
	if (!cbunch)
		goto out_unlock;
	/* XXX - need to get block size */
	/* make a large free block */


	memcpy(chead->md5, context->md5, 16);
	dlist_init(&chead->chunk_list);
	dlist_init(&chead->free_list);

	/* first free chunk for whole of the block */
	schunk = (pgstrom_shmem_chunk *) &chead->first_chunk;
	dlist_push_head(&chead->chunk_list, &schunk->chunk_chain);
	dlist_push_head(&chead->free_list, &schunk->free_chain);

	/* terminate marker */
	dnode = (dlist_node *)((char *)chead + SHMEM_BLOCKSZ - sizeof(dlist_node));
	*((uint32 *)((char *)dnode - sizeof(uint32))) = SHMEM_MAGIC_FREE_CHUNK;
	dlist_insert_after(&schunk->chunk_chain, dnode);

	dlist_push_tail(&context->block_list, &chead->chain);

	goto retry;

out_unlock:
	SpinLockRelease(&context->lock);

	return address;
}

void
pgstrom_shmem_free(void *address)
{
	pgstrom_shmem_context  *context;
	pgstrom_shmem_chunk_head *chead
		= (pgstrom_shmem_chunk_head *)((Size)address & ~(SHMEM_BLOCKSZ - 1));

	


}

void
pgstrom_shmem_context_reset(pgstrom_shmem_context *context)
{
	dlist_mutable_iter iter;

	SpinLockAcquire(&context->lock);
	dlist_foreach_modify(iter, &context->block_list)
	{
		pgstrom_shmem_chunk_head   *chead
			= dlist_container(pgstrom_shmem_chunk_head, chain, iter.cur);

		Assert(memcmp(context->md5, chead->md5, 16) == 0);
		dlist_delete(&chead->chain);
		pgstrom_shmem_block_free(chead);
	}
	Assert(dlist_is_empty(&context->block_list));
	SpinLockRelease(&context->lock);
}

void
pgstrom_shmem_context_delete(pgstrom_shmem_context *context)
{
	pgstrom_shmem_context_reset(context);
	pgstrom_shmem_free(context);
}


/*
 * find_least_pot
 *
 * It looks up the least power of two that is equal or larger than
 * the provided size.
 */
static __inline__ int
find_least_pot(Size size)
{
	int		bit = 0;

#ifdef __builtin_clzl
	bit = sizeof(unsigned long) * BITS_PER_BYTE * __builtin_clzl(size - 1);
#else
	size--;
#if sizeof(Size) == 8
	if ((size & 0xffffffff00000000UL) != 0)
	{
		size >>= 32;
		bit += 32;
	}
#endif
	if ((size & 0xffff0000UL) != 0)
	{
		size >>= 16;
		bit += 16;
	}
	if ((size & 0x0000ff00UL) != 0)
	{
		size >>= 8;
		bit += 8;
	}
	if ((size & 0x000000f0UL) != 0)
	{
		size >>= 4;
		bit += 4;
	}
	if ((size & 0x0000000cUL) != 0)
	{
		size >> 2;
		bit += 2;
	}
	if ((size & 0x00000002UL) != 0)
	{
		size >> 1;
		bit += 1;
	}
#endif
	return Max(bit, SHMEM_BLOCKSZ_BITS) - SHMEM_BLOCKSZ_BITS;
}

static bool
pgstrom_shmem_block_divide(int shift)
{
	pgstrom_shmem_block	*sblock;
	dlist_node	   *dnode;

	if (dlist_is_empty(&shmem_head->free_list[shift]))
	{
		if (shift == SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS ||
			!pgstrom_shmem_block_divide(shift + 1))
			return false;
	}
	Assert(!dlist_is_empty(&shmem_head->free_list[shift]));

	dnode = dlist_pop_head_node(&shmem_head->free_list[shift]);
	sblock = dlist_container(pgstrom_shmem_block, chain, dnode);

	/* sblock must be aligned to (1 << shift)'s block */
	Assert(((sblock - &shmem_head->blocks[0]) & ((1 << shift) - 1)) == 0);
	dlist_push_tail(&shmem_head->free_list[shift - 1], &sblock->chain);

	sblock = sblock + (1 << (shift - 1));
	dlist_push_tail(&shmem_head->free_list[shift - 1], &sblock->chain);

	return true;
}

static void *
pgstrom_shmem_block_alloc(Size size)
{
	pgstrom_shmem_block	*sblock;
	dlist_node	   *dnode;
	int				shift;
	int				index;
	void		   *address = NULL;

	if (size == 0)
		elog(ERROR, "unable to allocate 0 byte");
	if (size > (1UL << SHMEM_BLOCKSZ_BITS_MAX))
		elog(ERROR, "too large memory requirement (%lu)", size);

	SpinLockAcquire(&shmem_head->lock);
	shift = find_least_pot(size + sizeof(uint32));
	if (dlist_is_empty(&shmem_head->free_list[shift]))
	{
		if (!pgstrom_shmem_block_divide(shift + 1))
			goto out_unlock;
	}
	Assert(!dlist_is_empty(&shmem_head->free_list[shift]));

	dnode = dlist_pop_head_node(&shmem_head->free_list[shift]);
	sblock = dlist_container(pgstrom_shmem_block, chain, dnode);
	sblock->nullmark = NULL;
	sblock->size = size;

	index = sblock - &shmem_head->blocks[0];
	address = (void *)((Size)shmem_head->base_address + index * SHMEM_BLOCKSZ);
	/* put a sentry to detect overrun */
	*((uint32 *)((char *)address + size)) = SHMEM_BLOCK_MAGIC;
	shmem_head->curr_usage += (1 << shift);
	shmem_head->max_usage = Max(shmem_head->max_usage,
								shmem_head->curr_usage);
out_unlock:
	SpinLockRelease(&shmem_head->lock);

	return address;	
}

void
pgstrom_shmem_block_free(void *address)
{
	pgstrom_shmem_block	*sblock;
	int			index;
	int			shift;

	Assert((Size)address % SHMEM_BLOCKSZ == 0);
	index = (((Size)address - (Size)shmem_head->base_address) / SHMEM_BLOCKSZ);
	Assert(index >= 0 && index < pgstrom_shmem_numblocks);

	sblock = &shmem_head->blocks[index];
	Assert(sblock->active.nullmark == NULL);
	Assert(*((uint32 *)((char *)shmem_head->base_address +
						index * SHMEM_BLOCKSZ +
						sblock->active.size)) == SHMEM_BLOCK_MAGIC);

	shift = find_least_pot(sblock->size + sizeof(uint32));

	SpinLockAcquire(&shmem_head->lock);

	shmem_head->curr_usage -= (1 << shift);

	/* try to merge if buddy block is also free */
	while (shift < SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS)
	{
		pgstrom_shmem_block	*buddy;
		int		buddy_index = index ~ (1 << shift);

		buddy = &shmem_head->blocks[buddy_index];
		if (buddy->active.nullmark == NULL)
			break;

		dlist_delete(&buddy->dnode);
		if (buddy_index < index)
		{
			sblock = buddy;
			index = buddy_index;
		}
		shift++;
	}
	dlist_push_head(&shmem_head->free_list[shift], &sblock->dnode);
	SpinLockRelease(&shmem_head->lock);
}

void
pgstrom_construct_shmem(Size zone_length,
						void *(*callback)(void *address, Size length))
{
	shmem_zone *zone;
	Size		curr_size = 0;
	Size		next_size;

	zone = ShmemInitStruct("PG-Strom: shmem_zone",
						   pgstrom_shmem_totalsize, &found);
	Assert(!found);

	zone_length = TYPEALIGN(SHMEM_BLOCKSZ, zone_length);
	while (curr_size < pgstrom_shmem_totalsize)
	{
		if (curr_size + zone_length < pgstrom_shmem_totalsize)
			next_size = zone_length;
		else
			next_size = pgstrom_shmem_totalsize - curr_size;

		Assert(next_size % SHMEM_BLOCKSZ == 0);



		zone = ShmemInitStruct





}

static void
pgstrom_setup_shmem(void)
{
	Size	length;
	bool	found;

	length = offsetof(shmem_head, zones[pgstrom_shmem_maxzones]);
	shmem_head = ShmemInitStruct("PG-Strom: shmem_head",
								 length, &found);
	Assert(!found);

	memset(shmem_head, 0, length);
}
#if 0
	/* init pgstrom_shmem_head fields */
	SpinLockInit(&shmem_head->lock);
	for (i=0; i <= SHMEM_BLOCKSZ_BITS_MAX; i++)
		dlist_init(&shmem_head->free_list[i - SHMEM_BLOCKSZ_BITS]);

	shmem_head->curr_usage = 0;
	shmem_head->max_usage = 0;
	shmem_head->base_address
		= (void *)TYPEALIGN(SHMEM_BLOCKSZ,
							&shmem_head->blocks[pgstrom_shmem_numblocks]);

	memset(shmem_head->blocks, 0,
		   sizeof(pgstrom_shmem_block) * pgstrom_shmem_numblocks);
	i = 0;
	j = SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS;
	while (i < pgstrom_shmem_numblocks)
	{
		int		nblocks = (1 << j);

		if (i + nblocks < pgstrom_shmem_numblocks)
		{
			dlist_push_tail(&shmem_head->free_list[j],
							&shmem_head->blocks[i].chain);
			i += nblocks;
		}
		else if (j > 0)
			j--;
	}
	/*
	 * TODO: setup memory context on shmem segment
	 */
}
#endif

void
pgstrom_init_shmem(void)
{
	static int	shmem_totalsize;
	Size		length;

	/*
	 * Definition of GUC variables for shared memory management
	 */
	DefineCustomIntVariable("pgstrom.shmem_totalsize",
							"total size of shared memory segment for PG-Strom",
							NULL,
							&shmem_totalsize,
							SHMEM_BLOCKSZ * 1024,	/* 2GB */
							SHMEM_BLOCKSZ * 128,
							INT_MAX,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	if ((shmem_totalsize % (SHMEM_BLOCKSZ >> 10)) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
		errmsg("\"pgstrom.shmem_totalsize\" must be multiple of block size")));

	DefineCustomIntVariable("pgstrom.shmem_maxzones",
							"max number of shared memory zones for PG-Strom",
							NULL,
							&pgstrom_shmem_maxzones,
							256,
							1,
							INT_MAX,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	pgstrom_shmem_totalsize = ((Size)shmem_totalsize) << 10;

	/* Acquire shared memory segment */
	length = offsetof(shmem_head, zones[pgstrom_shmem_maxzones]);
	RequestAddinShmemSpace(MAXALIGN(length));
	/*
	 * XXX - to be replaced with dynamic shared memory or system
	 *       mmap(2) on hugetlbfs in the future.
	 */
	RequestAddinShmemSpace(pgstrom_shmem_totalsize + SHMEM_BLOCKSZ);

	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_setup_shmem;
}
