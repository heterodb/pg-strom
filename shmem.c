/*
 * shmem.c
 *
 * Management of shared memory segment
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
	void	   *private;
	void	   *block_baseaddr;
	shmem_block	blocks[FLEXIBLE_ARRAY_MEMBER];
} shmem_zone;

typedef struct
{
	bool		is_ready;
	int			num_zones;
	void	   *zone_baseaddr;
	Size		zone_length;
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
	long		num_blocks;
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

/* top-level shared memory context */
shmem_context  *TopShmemContext = NULL;

shmem_context *
pgstrom_shmem_context_create(const char *name)
{
	shmem_context  *context;
	int		namelen = strlen(name);

	context = pgstrom_shmem_alloc(TopShmemContext,
								  offsetof(shmem_context, name[namelen+1]));
	if (!context)
		return NULL;

	context->owner = getpid();
	SpinLockInit(&context->lock);
	context->total_active;
	context->num_blocks = 0;
	context->num_chunks = 0;
	pg_md5_binary(name, namelen, context->md5);
	strcpy(context->name, name);

	return context;
}

void
pgstrom_shmem_context_reset(shmem_context *context)
{
	dlist_mutable_iter	iter;

	SpinLockAcquire(&context->lock);
	dlist_foreach_modify(iter, &context->block_list)
	{
		shmem_bunch *bunch = dlist_container(shmem_bunch, chain, iter.cur);

		Assert(memcmp(context->md5, bunch->md5, 16) == 0);
		dlist_delete(&bunch->chain);
		pgstrom_shmem_block_free(bunch);
	}
	Assert(dlist_is_empty(&context->block_list));
	SpinLockRelease(&context->lock);
}

void
pgstrom_shmem_context_delete(shmem_context *context)
{
	pgstrom_shmem_context_reset(context);
	pgstrom_shmem_free(context);
}

void *
pgstrom_shmem_alloc(shmem_context *contetx, Size size)
{
	dlist_iter	iter1;
	dlist_iter	iter2;
	Size		required;
	int			shift;
	Size		bunch_len;
	void	   *address = NULL;

	required = MAXALIGN(offsetof(shmem_chunk, userdata[0]) +
						size + sizeof(uint32));
	SpinLockAcquire(&context->lock);
retry:
	dlist_foreach(iter1, &context->block_list)
	{
		shmem_bunch *bunch
			= dlist_container(shmem_bunch, chain, iter1.cur);

		Assert((Size)bunch % SHMEM_BLOCKSZ == 0);
		Assert(memcmp(bunch->md5, context->md5, 16) == 0);

		dlist_foreach(iter2, &bunch->free_list)
		{
			shmem_chunk	   *chunk;
			dlist_node	   *dnode;
			Size			chunk_len;
			Size			padding;

			chunk = dlist_container(shmem_chunk, chain, iter2.cur);
			dnode = dlist_next_node(&bunch->chunk_list,
									&chunk->chunk_chain);
			chunk_len = (char *)dnode - (char *)chunk;

			Assert(SHMEM_CHUNK_MAGIC(chunk) == SHMCHUNK_MAGIC_FREE);

			/* is this free chunk has enough space to store it? */
			if (required > length)
				continue;
			/*
			 * split this chunk, if it has enough space to store.
			 * "NAMEDATALEN" is just a threshold to avoid too small free
			 * chunk that shall not be able to allocate any more
			 */
			padding = MAXALIGN(offsetof(shmem_chunk, userdata[0]) +
							   NAMEDATALEN + sizeof(uint32));
			if (required + passing < length)
			{
				shmem_chunk	   *nchunk
					= (shmem_chunk *)((char *)chunk + required);
				dlist_insert_after(&chunk->chunk_chain,
								   &nchunk->chunk_chain);
				dlist_insert_after(&chunk->free_chain,
								   &nchunk->free_chain);
			}
			dlist_delete(&chunk->free_chain);
			chunk->active.nullmark = NULL;
			chunk->active.size = size;
			SHMEM_CHUNK_MAGIC(chunk) = SHMCHUNK_MAGIC_ACTIVE;

			address = chunk->userdata;
			goto out_unlock;
		}
	}

	bunch_len = (offsetof(shmem_bunch, first_chunk) +
				 required + sizeof(dlist_node));
	shift = find_least_pot(bunch_len + sizeof(uint32));
	bunch = pgstrom_shmem_block_alloc((1 << shift) - sizeof(uint32));
	if (!bunch)
		goto out_unlock;

	bunch->context = context;
	memcpy(bunch->md5, context->md5, 16);
	dlist_init(&bunch->chunk_list);
	dlist_init(&bunch->free_list);

	chunk = (shmem_chunk *) &bunch->first_chunk;
	dlist_push_tail(&bunch->chunk_list, &chunk->chunk_chain);
	dlist_push_tail(&bunch->free_list, &chunk->free_chain);

	dnode = (dlist_node *)((char *)bunch + bunch_len) - 1;
	dlist_push_tail(&bunch->chunk_list, dnode);
	SHMEM_CHUNK_MAGIC(chunk) = SHMCHUNK_MAGIC_FREE;

	dlist_push_tail(&context->block_list, &bunch->chain);
	goto retry;

out_unlock:
	SpinLockRelease(&context->lock);

	return address;
}

void
pgstrom_shmem_free(void *address)
{
	shmem_context  *context = ...;

	/* pull a zone that contains the address */
	Assert((Size)address >= (Size)shmem_head->zone_baseaddr &&
		   (Size)address < ((Size)shmem_head->zone_baseaddr +
							SHMEM_BLOCKSZ * shmem_head->num_zones));
	zone_index = ((Size)address -
				  (Size)shmem_head->zone_baseaddr) / shmem_head->zone_length;
	zone = shmem_head->zones[zone_index];

	/* pull blocksize of the block that contains the address */
	SpinLockAcquire(&zone->lock);
	Assert((Size)address >= (Size)zone->block_baseaddr &&
		   (Size)address < ((Size)zone->block_baseaddr +
							SHMEM_BLOCKSZ * zone->num_blocks));
	block_index = ((Size)address -
				   (Size)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = zone->blocks[block_index];
	Assert(!block->active.nullmark);
	block_size = block->active.size;
	SpinLockRelease(&zone->lock);

	/**/
	bunch = (Size)address & 

	// addr to context



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

/*
 *
 * XXX - caller must have lock of supplied zone
 */
static bool
pgstrom_shmem_zone_block_split(shmem_zone *zone, int shift)
{
	shmem_block	   *block;

	Assert(shift > 0 && shift <= SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS);
	if (dlist_is_empty(&zone->free_list[shift]))
	{
		if (shift == SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS ||
			!pgstrom_shmem_zone_block_split(zone, shift+1))
			return false;
	}
	Assert(!dlist_is_empty(&zone->free_list[shift]));

	block = dlist_container(shmem_block, chain,
							dlist_pop_head_node(&zone->free_list[shift]));
	index = block - &zone->blocks[0];
	Assert((index & ((1 << shift) - 1)) == 0);
	dlist_push_tail(&zone->free_list[shift-1], &block->chain);

	block += (1 << (shift - 1));
	dlist_push_tail(&zone->free_list[shift-1], &block->chain);

	return true;
}

static void *
pgstrom_shmem_zone_block_alloc(shmem_zone *zone, Size size)
{
	shmem_block	*sblock;
	int		shift = find_least_pot(MAXALIGN(size) + sizeof(uint32));
	int		index;
	void   *address;

	if (dlist_is_empty(&zone->free_list[shift]))
	{
		if (!pgstrom_shmem_zone_block_split(zone, shift+1))
			return NULL;
	}
	Assert(!dlist_is_empty(&zone->free_list[shift]));

	sblock = dlist_container(shmem_block, chain,
							 dlist_pop_head_node(&zone->free_list[shift]));
	sblock->nullmark = NULL;
	sblock->size = MAXALIGN(size);

	index = sblock - &zone->blocks[0];
	address = (void *)((Size)zone->base_address + index * SHMEM_BLOCKSZ);
	*((uint32 *)((char *)address + MAXALIGN(size))) = SHMEM_BLOCK_MAGIC;
	zone->num_active += (1 << shift);

	return address;
}

static void
pgstrom_shmem_zone_block_free(shmem_zone *zone, void *address)
{
	shmem_block	   *block;
	int				index;
	int				shift;

	Assert((Size)address >= (Size)zone->block_baseaddr &&
		   (Size)address < (Size)zone + SHMEM_BLOCKSZ * zone->num_blocks);

	index = (shmem_block *)address - &zone->blocks[0];
	block = &zone->blocks[index];
	Assert(block->active.nullmark == NULL);
	Assert(*((uint32 *)((char *)address +
						MAXALIGN(block->active.size))) == SHMEM_BLOCK_MAGIC);

	shift = find_least_pot(MAXALIGN(block->size) + sizeof(uint32));
	zone->num_active -= (1 << shift);

	/* try to merge buddy blocks if it is also free */
	while (shift < SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS)
	{
		shmem_block	   *buddy;
		int				buddy_index = index ~ (1 << shift);

		buddy = &zone->blocks[buddy_index];
		if (buddy->active.nullmark == NULL)
			break;

		dlist_delete(&buddy->chain);
		if (buddy_index < index)
		{
			block = buddy;
			index = buddy_index;
		}
		shift++;
	}
	dlist_push_head(&zone->free_list[shift], &block->chain);
}

static void *
pgstrom_shmem_block_alloc(Size size)
{
	/* unable to allocate 0-byte or too large */
	if (size == 0 || size > (1UL << SHMEM_BLOCKSZ_BITS_MAX))
		return NULL;

	/*
	 * find a zone we can allocate
	 *
	 *
	 *
	 */
	SpinLockAcquire(&zone->lock);


	SpinLockRelease(&zone->lock);

	return address;	
}

static void
pgstrom_shmem_block_free(void *address)
{
	shmem_zone *zone;
	void	   *zone_baseaddr = shmem_head->zone_baseaddr;
	Size		zone_length = shmem_head->zone_length;
	int			zone_index;

	Assert((Size)address % SHMEM_BLOCKSZ == 0);

	zone_index = ((Size)address - (Size)zone_baseaddr) / zone_length;
	Assert(zone_index >= 0 && zone_index < shmem_head->num_zones);

	zone = shmem_head->zones[zone_index];
	SpinLockAcquire(&zone->lock);
	pgstrom_shmem_zone_block_free(zone, address);
	SpinLockRelease(&zone->lock);
}

void
pgstrom_setup_shmem(Size zone_length,
					void *(*callback)(void *address, Size length))
{
	shmem_zone *zone;
	int			zone_index;
	int			num_zones;
	int			num_blocks;
	int			num_zones;
	Size		offset;

	pg_memory_barrier();
	if (shmem_head->is_ready)
		elog(ERROR, "tried to setup shared memory segment twice");

	zone_length = TYPEALIGN_DOWN(SHMEM_BLOCKSZ, zone_length);
	num_zones = (pgstrom_shmem_totalsize + zone_length - 1) / zone_length;

	zone = (void *)TYPEALIGN(SHMEM_BLOCKSZ, &shmem_head->zones[num_zones]);
	shmem_head->zone_baseaddr = zone;
	shmem_head->zone_length = zone_length;

	zone_index = 0;
	offset = 0;
	while (offset < pgstrom_shmem_totalsize)
	{
		Size	length;
		long	blkno;
		int		shift;

		Assert((Size)zone % SHMEM_BLOCKSZ);
		if (offset + zone_length < pgstrom_shmem_totalsize)
			length = zone_length;
		else if (offset + SHMEM_BLOCKSZ < pgstrom_shmem_totalsize)
			length = pgstrom_shmem_totalsize - curr_size;
		else
			break;
		Assert(length % SHMEM_BLOCKSZ == 0);
		num_blocks = ((length - offsetof(shmem_zone, blocks[0])) /
					  sizeof(shmem_block) + SHMEM_BLOCKSZ);

		/* per zone initialization */
		SpinLockInit(&zone->lock);
		zone->num_blocks = num_blocks;
		zone->num_active = 0;
		zone->block_baseaddr
			= (void *)TYPEALIGN(SHMEM_BLOCKSZ, &zone->blocks[num_blocks]);

		blkno = 0;
		shift = SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS;
		while (blkno < pgstrom_shmem_numblocks)
		{
			int		nblocks = (1 << shift);

			if (blkno + nblocks < zone->num_blocks)
			{
				dlist_push_tail(&zone->free_list[shift],
								&zone->blocks[blkno].chain);
				blkno += nblocks;
			}
			else if (shift > 0)
				shift--;
		}
		/* put zone on the shmem_head */
		shmem->zones[zone_index++] = zone;
		zone = (shmem_zone *)((char *)zone + zone_length);
	}
	Assert(index == num_zones);

	shmem_head->num_zones = num_zones;
	shmem_head->is_ready = true;
	pg_memory_barrier();
}

static void
pgstrom_startup_shmem(void)
{
	Size	length;
	bool	found;

	length = MAXALIGN(offsetof(shmem_head, zones[pgstrom_shmem_maxzones])) +
		pgstrom_shmem_totalsize + SHMEM_BLOCKSZ;

	shmem_head = ShmemInitStruct("PG-Strom: shmem_head", length, &found);
	Assert(!found);

	memset(shmem_head, 0, length);
}

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
	shmem_startup_hook = pgstrom_startup_shmem;
}
