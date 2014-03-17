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
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "lib/ilist.h"
#include "libpq/md5.h"
#include "storage/barrier.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"
#include <limits.h>
#include <unistd.h>

/*
 * management of shared memory segment in PG-Strom
 *
 * Once a shared memory segment is allocated, PG-Strom split it into
 * multiple zones. A zone usually has more than 500MB, according to
 * the capability of OpenCL driver to map a parciular area as page-locked
 * memory. Also, it shall be associated with a particular NUMA node for
 * better memory access latency, in the future version.
 * 
 * A zone contains a certain number of fixed-length (= SHMEM_BLOCKSZ) blocks. 
 * Block allocation system allocates 2^n blocks for the request.
 * On the other hand, context based allocation also allows to assign smaller
 * chunks on blocks being allocated by block allocation system.
 */

#define SHMEM_BLOCKSZ_BITS_MAX	34			/* 16GB */
#define SHMEM_BLOCKSZ_BITS		22			/*  4MB */
#define SHMEM_BLOCKSZ_BITS_RANGE	\
	(SHMEM_BLOCKSZ_BITS_MAX - SHMEM_BLOCKSZ_BITS)
#define SHMEM_BLOCKSZ			(1UL << SHMEM_BLOCKSZ_BITS)

typedef union
{
	dlist_node		chain;		/* to be chained free_list of shmem_zone */
	struct {
		void	   *nullmark;	/* NULL if active block */
		uint32		nshift;		/* 2^nshift blocks were allocated */
	} active;
} shmem_block;

typedef struct
{
	slock_t		lock;
	long		num_blocks;		/* number of total blocks */
	long		num_active[SHMEM_BLOCKSZ_BITS_RANGE + 1];
	long		num_free[SHMEM_BLOCKSZ_BITS_RANGE + 1];
	dlist_head	free_list[SHMEM_BLOCKSZ_BITS_RANGE + 1];
	void	   *private;		/* cl_mem being mapped (Only OpenCL server) */
	void	   *block_baseaddr;
	shmem_block	blocks[FLEXIBLE_ARRAY_MEMBER];
} shmem_zone;

typedef struct
{
	/* for context management */
	slock_t		context_lock;
	dlist_head	context_list;

	/* for device management */
	int			device_num;
	pgstrom_device_info	**device_info;

	/* for zone management */
	bool		is_ready;
	int			num_zones;
	void	   *zone_baseaddr;
	Size		zone_length;
	shmem_zone *zones[FLEXIBLE_ARRAY_MEMBER];
} shmem_head;

/*
 * shmem_context / shmem_bunch / shmem_chunk
 *
 * It is a mechanism to allocate smaller fraction of shared memory than
 * SHMEM_BLOCKSZ. A shmem_context is a set of shmem_bunch that can contain
 * multiple shmem_chunk objects.
 * The shmem_bunch is assigned on the head of blocks, and assigned a certain
 * shmem_context being linked to context->block_list. It shall be allocated
 * as the following diagram.
 * 
 * shmem_bunch
 * +-----------------------+
 * |                       |
 * |  [shmem_bunch]        |
 * |                       |
 * +-----------------------+ <-- &bunch->first_chain
 * |  [shmem_chunk]        |
 * |   chunk_chain      O------+
 * |   free_fain / active  |   | dual linked list towards the neighbor's chunk
 * |   userdata[]          |   |
 * |       :               |   |
 * |       :               |   |
 * |   magic(uint32)       |   |
 * +-----------------------+   |
 * |  [shmem_chunk]        |   |
 * |   chunk_chain      <------+
 * |   free_fain / active  |
 * |   userdata[]          |
 * |       :               |
 * |       :               |
 * |   magic(uint32)       |
 * +-----------------------+
 * |                       |
 * /                       /   :
 * /                       /   :
 * |                       |   |
 * |   magic(uint32)       |   |
 * +-----------------------+   |
 * |   dlist_node       <------+ terminator of the dual linked list.
 * +-----------------------+
 *
 * The chunk_chain of shmem_chunk is a dual linked list, so it allows to walk on
 * the chunks being located on the neighbor. A dlist_node is put on the end of
 * shmem_bunch block, so you can check termination of shmem_chunk chain using
 * dlist_has_next().
 * If allocation request is enough small towards the size of free chunk,
 * allocator will split a free chunk into two chunks; one for active, and the
 * remaining one for still free.
 */
typedef struct shmem_context
{
	dlist_node	chain;		/* to be linked shmem_head->context_list */

	pid_t		owner;
	slock_t		lock;
	dlist_head	block_list;
	Size		total_active;
	long		num_blocks;
	long		num_chunks;
	char		md5[16];
	char		name[NAMEDATALEN];
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

#define SHMCHUNK_MAGIC_FREE			0xF9EEA9EA
#define SHMCHUNK_MAGIC_ACTIVE		0xAC715AEA
#define SHMEM_CHUNK_MAGIC(chunk)										\
	*((uint32 *)(!(chunk)->active.nullmark ?							\
				 (char *)(chunk) + (chunk)->active.size :				\
				 (char *)(chunk)->chunk_chain.next - sizeof(uint32)))

#define ADDRESS_IN_SHMEM(address)										\
	((Size)(address) >= (Size)pgstrom_shmem_head->zone_baseaddr &&		\
	 (Size)(address) < ((Size)pgstrom_shmem_head->zone_baseaddr +		\
						pgstrom_shmem_totalsize))
#define ADDRESS_IN_SHMEM_ZONE(zone,address)					\
	((Size)(address) >= (Size)(zone)->block_baseaddr &&		\
	 (Size)(address) < ((Size)(zone)->block_baseaddr +		\
						(zone)->num_blocks) * SHMEM_BLOCKSZ)
/* static functions */
static void *pgstrom_shmem_block_alloc(Size size);
static void  pgstrom_shmem_block_free(void *address);

/* static variables */
static shmem_startup_hook_type shmem_startup_hook_next;
static Size			pgstrom_shmem_totalsize;
static int			pgstrom_shmem_maxzones;
static shmem_head  *pgstrom_shmem_head;

/* top-level shared memory context */
shmem_context  *TopShmemContext = NULL;

/*
 * find_least_pot
 *
 * It looks up the least power of two that is equal or larger than
 * the provided size.
 */
static IF_INLINE int
find_least_pot(Size size)
{
	int		shift = 0;

#ifdef __builtin_clzl
	shift = sizeof(unsigned long) * BITS_PER_BYTE * __builtin_clzl(size - 1);
#else
	size--;
#if SIZEOF_VOID_P == 8
	if ((size & 0xffffffff00000000UL) != 0)
	{
		size >>= 32;
		shift += 32;
	}
#endif
	if ((size & 0xffff0000UL) != 0)
	{
		size >>= 16;
		shift += 16;
	}
	if ((size & 0x0000ff00UL) != 0)
	{
		size >>= 8;
		shift += 8;
	}
	if ((size & 0x000000f0UL) != 0)
	{
		size >>= 4;
		shift += 4;
	}
	if ((size & 0x0000000cUL) != 0)
	{
		size >>= 2;
		shift += 2;
	}
	if ((size & 0x00000002UL) != 0)
	{
		size >>= 1;
		shift += 1;
	}
#endif
	return Max(shift, SHMEM_BLOCKSZ_BITS) - SHMEM_BLOCKSZ_BITS;
}

shmem_context *
pgstrom_shmem_context_create(const char *name)
{
	shmem_context  *context;
	Size			namelen = strlen(name);

	if (namelen >= NAMEDATALEN - 1)
		return NULL;	/* name too long */

	context = pgstrom_shmem_alloc(TopShmemContext, sizeof(shmem_context));
	if (!context)
		return NULL;

	context->owner = getpid();
	SpinLockInit(&context->lock);
	dlist_init(&context->block_list);
	context->total_active = 0;
	context->num_blocks = 0;
	context->num_chunks = 0;
	pg_md5_binary(name, namelen, context->md5);
	strcpy(context->name, name);

	SpinLockAcquire(&pgstrom_shmem_head->context_lock);
	dlist_push_tail(&pgstrom_shmem_head->context_list,
					&context->chain);
	SpinLockRelease(&pgstrom_shmem_head->context_lock);

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
	Assert(context != TopShmemContext);

	/* remove from the global list */
	SpinLockAcquire(&context->lock);
	dlist_delete(&context->chain);
	SpinLockRelease(&context->lock);

	/* then. release all */
	pgstrom_shmem_context_reset(context);
	pgstrom_shmem_free(context);
}

void *
pgstrom_shmem_alloc(shmem_context *context, Size size)
{
	dlist_iter	iter1;
	dlist_iter	iter2;
	Size		required;
	int			shift;
	void	   *address = NULL;

	required = MAXALIGN(offsetof(shmem_chunk, userdata[size + sizeof(uint32)]));

	SpinLockAcquire(&context->lock);
	while (true)
	{
		shmem_bunch	   *bunch;
		Size			bunch_len;
		shmem_chunk	   *chunk;
		Size			chunk_len;
		dlist_node	   *dnode;

		dlist_foreach(iter1, &context->block_list)
		{
			bunch = dlist_container(shmem_bunch, chain, iter1.cur);

			Assert((Size)bunch % SHMEM_BLOCKSZ == 0);
			Assert(memcmp(bunch->md5, context->md5, 16) == 0);

			dlist_foreach(iter2, &bunch->free_list)
			{
				Size			padding;

				chunk = dlist_container(shmem_chunk, free_chain, iter2.cur);
				dnode = dlist_next_node(&bunch->chunk_list,
										&chunk->chunk_chain);
				chunk_len = (char *)dnode - (char *)chunk;

				Assert(SHMEM_CHUNK_MAGIC(chunk) == SHMCHUNK_MAGIC_FREE);

				/* is this free chunk has enough space to store it? */
				if (required > chunk_len)
					continue;

				/*
				 * Split this chunk, if the later portion has enough space to
				 * perform as a free chunk being longer than "NAMEDATALEN" just
				 * for threshold to avoid too small fraction.
				 */
				padding = MAXALIGN(offsetof(shmem_chunk,
											userdata[NAMEDATALEN + sizeof(uint32)]));
				if (required + padding < chunk_len)
				{
					shmem_chunk	   *nchunk
						= (shmem_chunk *)((char *)chunk + required);
					dlist_insert_after(&chunk->chunk_chain,
									   &nchunk->chunk_chain);
					dlist_insert_after(&chunk->free_chain,
									   &nchunk->free_chain);
					SHMEM_CHUNK_MAGIC(nchunk) = SHMCHUNK_MAGIC_FREE;
					context->num_chunks++;
				}
				dlist_delete(&chunk->free_chain);
				chunk->active.nullmark = NULL;
				chunk->active.size = size;
				SHMEM_CHUNK_MAGIC(chunk) = SHMCHUNK_MAGIC_ACTIVE;

				address = chunk->userdata;

				/* adjust statistic */
				dnode = dlist_next_node(&bunch->chunk_list,
										&chunk->chunk_chain);
				context->total_active += (char *)dnode - (char *)chunk;
				goto out_unlock;
			}
		}
		/*
		 * No free space in the existing blocks, so allocate a new one.
		 */
		bunch_len = (offsetof(shmem_bunch, first_chunk) +			/* shmem_bunch */
					 offsetof(shmem_chunk, userdata[required]) +	/* shmem_chunk */
					 sizeof(uint32) +		/* SHMEM_CHUNK_MAGIC */
					 sizeof(dlist_node));	/* terminator */
		bunch = pgstrom_shmem_block_alloc(bunch_len);
		if (!bunch)
			goto out_unlock;

		bunch->context = context;
		memcpy(bunch->md5, context->md5, 16);
		dlist_init(&bunch->chunk_list);
		dlist_init(&bunch->free_list);

		chunk = (shmem_chunk *) &bunch->first_chunk;
		dlist_push_tail(&bunch->chunk_list, &chunk->chunk_chain);
		dlist_push_tail(&bunch->free_list, &chunk->free_chain);

		shift = find_least_pot(bunch_len);
		dnode = (dlist_node *)((char *)bunch + (1 << shift)) - 1;
		dlist_push_tail(&bunch->chunk_list, dnode);
		SHMEM_CHUNK_MAGIC(chunk) = SHMCHUNK_MAGIC_FREE;

		dlist_push_tail(&context->block_list, &bunch->chain);
		context->num_blocks += (1 << shift);
	}
out_unlock:
	SpinLockRelease(&context->lock);

	return address;
}

void
pgstrom_shmem_free(void *address)
{
	shmem_zone	   *zone;
	shmem_block	   *block;
	shmem_bunch	   *bunch;
	shmem_context  *context;
	shmem_chunk	   *chunk;
	shmem_chunk	   *pchunk;
	shmem_chunk	   *nchunk;
	int				index;
	int				nshift;
	dlist_node	   *dnode;

	/* pull a zone that contains the address */
	Assert(ADDRESS_IN_SHMEM(address));
	index = ((Size)address - (Size)pgstrom_shmem_head->zone_baseaddr)
		/ pgstrom_shmem_head->zone_length;
	zone = pgstrom_shmem_head->zones[index];

	/* pull blocksize of the block that contains the address */
	Assert(ADDRESS_IN_SHMEM_ZONE(zone, address));
	index = ((Size)address - (Size)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = &zone->blocks[index];
	SpinLockAcquire(&zone->lock);
	Assert(!block->active.nullmark);
	nshift = block->active.nshift + SHMEM_BLOCKSZ_BITS;
	SpinLockRelease(&zone->lock);

	/* ok, we could pull a bunch that contains the supplied address  */
	bunch = (shmem_bunch *)((Size)address & ~((1UL << nshift) - 1));
	context = bunch->context;

	Assert(ADDRESS_IN_SHMEM(context));
	Assert(memcmp(bunch->md5, context->md5, 16) == 0);

	SpinLockAcquire(&context->lock);
#ifdef PGSTROM_DEBUG
	do {
		dlist_iter	iter;
		bool		found = false;

		dlist_foreach(iter, &bunch->chunk_list)
		{
			chunk = dlist_container(shmem_chunk, chunk_chain, iter.cur);

			if (chunk->userdata == address)
				found = true;
			Assert(chunk->active.nullmark ?
				   SHMEM_CHUNK_MAGIC(chunk) == SHMCHUNK_MAGIC_FREE :
				   SHMEM_CHUNK_MAGIC(chunk) == SHMCHUNK_MAGIC_ACTIVE);
		}
		Assert(found);
	} while(0);
#endif
	chunk = (shmem_chunk *)((Size)address - offsetof(shmem_chunk,
													 userdata[0]));
	/* adjust statistics */
	dnode = dlist_next_node(&bunch->chunk_list,
							&chunk->chunk_chain);
	context->total_active -= (char *)dnode - (char *)chunk;

	/*
	 * merge with previous chunk, if it is also free.
	 */
	if (dlist_has_prev(&bunch->chunk_list, &chunk->chunk_chain))
	{
		dnode = dlist_prev_node(&bunch->chunk_list, &chunk->chunk_chain);
		pchunk = dlist_container(shmem_chunk, chunk_chain, dnode);

		if (pchunk->active.nullmark)
		{
			context->num_chunks--;
			dlist_delete(&chunk->chunk_chain);
			dlist_delete(&pchunk->free_chain);
			chunk = pchunk;
		}
	}

	/*
	 * merge with next chunk, if it is also free.
	 * note that we need to confirm the dnode is not terminator of bunch
	 */
	dnode = dlist_next_node(&bunch->chunk_list, &chunk->chunk_chain);
	if (dlist_has_next(&bunch->chunk_list, dnode))
	{
		nchunk = dlist_container(shmem_chunk, chunk_chain, dnode);

		if (nchunk->active.nullmark)
		{
			context->num_chunks--;
			dlist_delete(&nchunk->chunk_chain);
			dlist_delete(&nchunk->free_chain);
		}
	}
	dlist_push_head(&bunch->free_list, &chunk->free_chain);

	SpinLockRelease(&context->lock);
}

/*
 *
 * XXX - caller must have lock of supplied zone
 */
static bool
pgstrom_shmem_zone_block_split(shmem_zone *zone, int shift)
{
	shmem_block	   *block;
	dlist_node	   *dnode;
	int				index;

	Assert(shift > 0 && shift <= SHMEM_BLOCKSZ_BITS_RANGE);
	if (dlist_is_empty(&zone->free_list[shift]))
	{
		if (shift == SHMEM_BLOCKSZ_BITS_RANGE ||
			!pgstrom_shmem_zone_block_split(zone, shift+1))
			return false;
	}
	Assert(!dlist_is_empty(&zone->free_list[shift]));

	dnode = dlist_pop_head_node(&zone->free_list[shift]);
	zone->num_free[shift]--;

	block = dlist_container(shmem_block, chain, dnode);
	index = block - &zone->blocks[0];
	Assert((index & ((1 << shift) - 1)) == 0);
	dlist_push_tail(&zone->free_list[shift-1], &block->chain);

	block += (1 << (shift - 1));
	dlist_push_tail(&zone->free_list[shift-1], &block->chain);

	zone->num_free[shift-1] += 2;

	return true;
}

static void *
pgstrom_shmem_zone_block_alloc(shmem_zone *zone, Size size)
{
	shmem_block	*sblock;
	dlist_node	*dnode;
	int		shift = find_least_pot(size);
	int		index;
	void   *address;

	if (dlist_is_empty(&zone->free_list[shift]))
	{
		if (!pgstrom_shmem_zone_block_split(zone, shift+1))
			return NULL;
	}
	Assert(!dlist_is_empty(&zone->free_list[shift]));

	dnode = dlist_pop_head_node(&zone->free_list[shift]);
	sblock = dlist_container(shmem_block, chain, dnode);
	sblock->active.nullmark = NULL;
	sblock->active.nshift = shift;

	index = sblock - &zone->blocks[0];
	address = (void *)((Size)zone->block_baseaddr + index * SHMEM_BLOCKSZ);
	zone->num_free[shift]--;
	zone->num_active[shift]++;

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

	index = ((Size)address - (Size)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = &zone->blocks[index];
	Assert(block->active.nullmark == NULL);
	Assert(block->active.nshift <= SHMEM_BLOCKSZ_BITS_RANGE);
	shift = block->active.nshift;

	zone->num_active[shift]--;

	/* try to merge buddy blocks if it is also free */
	while (shift < SHMEM_BLOCKSZ_BITS_RANGE)
	{
		shmem_block	   *buddy;
		int				buddy_index = index ^ (1 << shift);

		buddy = &zone->blocks[buddy_index];
		if (buddy->active.nullmark == NULL)
			break;

		dlist_delete(&buddy->chain);
		if (buddy_index < index)
		{
			block = buddy;
			index = buddy_index;
		}
		zone->num_free[shift]--;
		shift++;
	}
	dlist_push_head(&zone->free_list[shift], &block->chain);
}

/*
 * pgstrom_shmem_block_alloc
 *
 * It is an internal API; that allocates a continuous 2^N blocks from a particular
 * shared memory zone. It tries to split a larger memory blocks if suitable memory
 * blocks are not free. If no memory blocks are available, it goes into another
 * zone to allocate memory.
 */
static void *
pgstrom_shmem_block_alloc(Size size)
{
	static int	zone_index = 0;
	int			start;
	shmem_zone *zone;
	void	   *address;

	/* unable to allocate 0-byte or too large */
	if (size == 0 || size > (1UL << SHMEM_BLOCKSZ_BITS_MAX))
		return NULL;

	/*
	 * find a zone we should allocate.
	 *
	 * XXX - To be put more wise zone selection
	 *  - NUMA aware
	 *  - Memory reclaim when no blocks are available
	 */
	start = zone_index;
	do {
		zone = pgstrom_shmem_head->zones[zone_index];
		SpinLockAcquire(&zone->lock);

		address = pgstrom_shmem_zone_block_alloc(zone, size);

		SpinLockRelease(&zone->lock);

		if (address)
			break;

		zone_index = (zone_index + 1) % pgstrom_shmem_head->num_zones;
	} while (zone_index != start);

	return address;	
}

static void
pgstrom_shmem_block_free(void *address)
{
	shmem_zone *zone;
	void	   *zone_baseaddr = pgstrom_shmem_head->zone_baseaddr;
	Size		zone_length = pgstrom_shmem_head->zone_length;
	int			zone_index;

	Assert((Size)address % SHMEM_BLOCKSZ == 0);

	zone_index = ((Size)address - (Size)zone_baseaddr) / zone_length;
	Assert(zone_index >= 0 && zone_index < pgstrom_shmem_head->num_zones);

	zone = pgstrom_shmem_head->zones[zone_index];
	SpinLockAcquire(&zone->lock);
	pgstrom_shmem_zone_block_free(zone, address);
	SpinLockRelease(&zone->lock);
}

/*
 * collect_shmem_block_info
 *
 * It collects statistical information of shared memory zone.
 * Note that it does not trust statistical values if debug build, thus it may
 * take longer time because of walking of shared memory zone.
 */
typedef struct
{
	int		zone;
	int		shift;
	int		num_active;
	int		num_free;
} shmem_block_info;

static List *
collect_shmem_block_info(shmem_zone *zone, int zone_index)
{
	List	   *results = NIL;
	long		num_active[SHMEM_BLOCKSZ_BITS_RANGE + 1];
	long		num_free[SHMEM_BLOCKSZ_BITS_RANGE + 1];
	long		i;

	/*
	 * For debugging, we don't trust statistical information. Even though
	 * it takes block counting under the execlusive lock, we try to pick
	 * up raw data.
	 * Elsewhere, we just pick up statistical data.
	 */
#ifdef PGSTROM_DEBUG
	memset(num_active, 0, sizeof(num_active));
	memset(num_free, 0, sizeof(num_free));

	i = 0;
	while (i < zone->num_blocks)
	{
		shmem_block	*block = &zone->blocks[i];
		dlist_node	*dnode;

		if (!block->active.nullmark)
		{
			Assert(block->active.nshift < SHMEM_BLOCKSZ_BITS_RANGE + 1);
			num_active[block->active.nshift]++;
			i += (1 << block->active.nshift);
		}
		else
		{
			Size addr_min = (Size)(zone->free_list);
			Size addr_max = (Size)(zone->free_list +
								   SHMEM_BLOCKSZ_BITS_RANGE + 1);
			dnode = &block->chain;
			while (true)
			{
				if ((Size)dnode >= addr_min && (Size)dnode < addr_max)
				{
					int		j;

					for (j=0; j <= SHMEM_BLOCKSZ_BITS_RANGE; j++)
					{
						if (dnode == &zone->free_list[j].head)
						{
							num_free[j]++;
							i += (1 << j);
							break;
						}
					}
					Assert(j <= SHMEM_BLOCKSZ_BITS_RANGE);
					break;
				}
				dnode = dnode->next;
				Assert(dnode != &block->chain);
			}
		}
	}
	Assert(memcmp(num_active, zone->num_active, sizeof(num_active)) == 0);
	Assert(memcmp(num_free, zone->num_free, sizeof(num_free)) == 0);
#else
	memcpy(num_active, zone->num_active, sizeof(num_active));
	memcpy(num_free, zone->num_free, sizeof(num_free));
#endif
	for (i=0; i <= SHMEM_BLOCKSZ_BITS_RANGE; i++)
	{
		shmem_block_info *block_info
			= palloc(sizeof(shmem_block_info));
		block_info->zone = zone_index;
		block_info->shift = i;
		block_info->num_active = zone->num_active[i];
		block_info->num_free = zone->num_free[i];

		results = lappend(results, block_info);
	}
	return results;
}

Datum
pgstrom_shmem_block_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	   *fncxt;
	shmem_block_info   *block_info;
	HeapTuple	tuple;
	Datum		values[4];
	bool		isnull[4];
	int			shift;
	char		buf[32];

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		shmem_zone	   *zone;
		List		   *block_info_list = NIL;
		int				i;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(4, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "zone",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "size",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "active",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "free",
						   INT8OID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		for (i=0; i < pgstrom_shmem_head->num_zones; i++)
		{
			zone = pgstrom_shmem_head->zones[i];

			SpinLockAcquire(&zone->lock);
			PG_TRY();
			{
				List   *temp = collect_shmem_block_info(zone, i);

				block_info_list = list_concat(block_info_list, temp);
			}
			PG_CATCH();
			{
				SpinLockRelease(&zone->lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockRelease(&zone->lock);
		}
			fncxt->user_fctx = block_info_list;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
		SRF_RETURN_DONE(fncxt);

	block_info = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(block_info->zone);
	shift = block_info->shift + SHMEM_BLOCKSZ_BITS;
	if (shift < 20)
		snprintf(buf, sizeof(buf), "%zuK", 1UL << (shift - 10));
	else if (shift < 30)
		snprintf(buf, sizeof(buf), "%zuM", 1UL << (shift - 20));
	else if (shift < 40)
		snprintf(buf, sizeof(buf), "%zuG", 1UL << (shift - 30));
	else
		snprintf(buf, sizeof(buf), "%zuT", 1UL << (shift - 40));
	values[1] = CStringGetTextDatum(buf);
	values[2] = Int64GetDatum(block_info->num_active);
	values[3] = Int64GetDatum(block_info->num_free);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_block_info);

typedef struct
{
	text	   *name;
	pid_t		owner;
	int64		usage;
	int64		alloc;
	int64		num_chunks;
	text	   *md5sum;
} shmem_context_info;

static text *
md5_to_text(const char md5[16])
{
	static const char *hex = "0123456789abcdef";
	char	buf[33];
	int		q, w;

	for (q = 0, w = 0; q < 16; q++)
	{
		buf[w++] = hex[(md5[q] >> 4) & 0x0F];
		buf[w++] = hex[md5[q] & 0x0F];
	}
	buf[w] = '\0';

	return cstring_to_text(buf);
}

Datum
pgstrom_shmem_context_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	shmem_context_info *cxt_info;
	Datum		values[6];
	bool		isnull[6];
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		List		   *cxt_list = NIL;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(6, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "name",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "owner",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "usage",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "alloc",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "num_chunks",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "md5sum",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		SpinLockAcquire(&pgstrom_shmem_head->context_lock);
		PG_TRY();
		{
			dlist_iter	iter;

			dlist_foreach(iter, &pgstrom_shmem_head->context_list)
			{
				shmem_context  *context
					= dlist_container(shmem_context, chain, iter.cur);

				cxt_info = palloc(sizeof(shmem_context_info));
				cxt_info->name = cstring_to_text(context->name);
				cxt_info->owner = context->owner;
				cxt_info->usage = context->total_active;
				cxt_info->alloc = context->num_blocks * SHMEM_BLOCKSZ;
				cxt_info->num_chunks = context->num_chunks;
				cxt_info->md5sum = md5_to_text(context->md5);

				cxt_list = lappend(cxt_list, cxt_info);
			}
		}
		PG_CATCH();
		{
			SpinLockRelease(&pgstrom_shmem_head->context_lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(&pgstrom_shmem_head->context_lock);

		fncxt->user_fctx = cxt_list;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
		SRF_RETURN_DONE(fncxt);

	cxt_info = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	values[0] = PointerGetDatum(cxt_info->name);
	values[1] = Int32GetDatum(cxt_info->owner);
	values[2] = Int64GetDatum(cxt_info->usage);
	values[3] = Int64GetDatum(cxt_info->alloc);
	values[4] = Int64GetDatum(cxt_info->num_chunks);
	values[5] = PointerGetDatum(cxt_info->md5sum);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_context_info);

static void
construct_shmem_top_context(const char *name, shmem_zone *zone)
{
	shmem_bunch	   *bunch;
	shmem_chunk	   *chunk1;
	shmem_chunk	   *chunk2;
	dlist_node	   *dnode;
	shmem_context  *context;
	Size			name_len = strlen(name);

	/* allocate a block */
	bunch = pgstrom_shmem_zone_block_alloc(zone, SHMEM_BLOCKSZ);
	if (!bunch)
		elog(ERROR, "out of shared memory");

	dlist_init(&bunch->chunk_list);
	dlist_init(&bunch->free_list);

	/* 1st active chunk */
	chunk1 = (shmem_chunk *)&bunch->first_chunk;
	chunk1->active.nullmark = NULL;
	chunk1->active.size =
		offsetof(shmem_chunk, userdata[0]) +
		offsetof(shmem_context, name[name_len + 1]);
	context = (shmem_context *)chunk1->userdata;
	dlist_push_tail(&bunch->chunk_list, &chunk1->chunk_chain);

	/* 2nd free chunk */
	chunk2 = (shmem_chunk *)
		((char *)chunk1 + MAXALIGN(chunk1->active.size + sizeof(uint32)));
	dlist_push_tail(&bunch->free_list, &chunk2->free_chain);
	dlist_push_tail(&bunch->chunk_list, &chunk2->chunk_chain);

	/* End of bunch marker */
	dnode = (dlist_node *)
		 ((char *)bunch + SHMEM_BLOCKSZ - sizeof(dlist_node));
	dlist_push_tail(&bunch->chunk_list, dnode);

	SHMEM_CHUNK_MAGIC(chunk1) = SHMCHUNK_MAGIC_ACTIVE;
	SHMEM_CHUNK_MAGIC(chunk2) = SHMCHUNK_MAGIC_FREE;

	/* initialize the context */
	context->owner = getpid();
	SpinLockInit(&context->lock);
	dlist_init(&context->block_list);
	context->total_active = (char *)chunk2 - (char *)chunk1;
	context->num_blocks = 1;
	context->num_chunks = 2;
	pg_md5_binary(name, name_len, context->md5);
	strcpy(context->name, name);

	bunch->context = context;
	memcpy(bunch->md5, context->md5, 16);
	dlist_push_head(&context->block_list, &bunch->chain);

	SpinLockAcquire(&pgstrom_shmem_head->context_lock);
	dlist_push_tail(&pgstrom_shmem_head->context_list, &context->chain);
	SpinLockRelease(&pgstrom_shmem_head->context_lock);

	TopShmemContext = context;
}

void
pgstrom_setup_shmem(Size zone_length,
					void *(*callback)(void *address,
									  Size length,
									  void *callback_private),
					void *callback_private)
{
	shmem_zone	   *zone;
	long			zone_index;
	long			num_zones;
	long			num_blocks;
	Size			offset;

	pg_memory_barrier();
	if (pgstrom_shmem_head->is_ready)
	{
		elog(LOG, "shared memory segment is already set up");
		return;
	}

	zone_length = TYPEALIGN_DOWN(SHMEM_BLOCKSZ, zone_length);
	num_zones = (pgstrom_shmem_totalsize + zone_length - 1) / zone_length;
	pgstrom_shmem_head->zone_baseaddr
		= (void *)TYPEALIGN(SHMEM_BLOCKSZ,
							&pgstrom_shmem_head->zones[num_zones]);
	pgstrom_shmem_head->zone_length = zone_length;

	offset = 0;
	for (zone_index = 0; zone_index < num_zones; zone_index++)
	{
		Size	length;
		long	blkno;
		int		shift;
		int		i;

		if (offset + zone_length < pgstrom_shmem_totalsize)
			length = zone_length;
		else
			length = pgstrom_shmem_totalsize - offset;
		Assert(length > 0 && length % SHMEM_BLOCKSZ == 0);

		num_blocks = ((length - offsetof(shmem_zone, blocks[0])) /
					  (sizeof(shmem_block) + SHMEM_BLOCKSZ));

		/* per zone initialization */
		zone = (shmem_zone *)
			((char *)pgstrom_shmem_head->zone_baseaddr + offset);
		Assert((Size)zone % SHMEM_BLOCKSZ == 0);

		SpinLockInit(&zone->lock);
		zone->num_blocks = num_blocks;
		for (i=0; i < SHMEM_BLOCKSZ_BITS_RANGE+1; i++)
		{
			dlist_init(&zone->free_list[i]);
			zone->num_active[i] = 0;
			zone->num_free[i] = 0;
		}
		zone->block_baseaddr
			= (void *)TYPEALIGN(SHMEM_BLOCKSZ, &zone->blocks[num_blocks]);
		blkno = 0;
		shift = SHMEM_BLOCKSZ_BITS_RANGE;
		while (blkno < num_blocks)
		{
			int		nblocks = (1 << shift);

			if (blkno + nblocks <= zone->num_blocks)
			{
				dlist_push_tail(&zone->free_list[shift],
								&zone->blocks[blkno].chain);
				zone->num_free[shift]++;
				blkno += nblocks;
			}
			else if (shift > 0)
				shift--;
		}
		/* per zone initialization */
		(*callback)(zone->block_baseaddr,
					zone->num_blocks * SHMEM_BLOCKSZ, 
					callback_private);
		/* put zone on the pgstrom_shmem_head */
		pgstrom_shmem_head->zones[zone_index] = zone;
		offset += length;
	}

	Assert(zone_index == num_zones);
	pgstrom_shmem_head->num_zones = num_zones;

	/* construct TopShmemContext */
	construct_shmem_top_context("Top Shmem Context",
								pgstrom_shmem_head->zones[0]);

	/* OK, now ready to use shared memory segment */
	pgstrom_shmem_head->is_ready = true;
}

static void
pgstrom_startup_shmem(void)
{
	Size	length;
	bool	found;

	length = MAXALIGN(offsetof(shmem_head, zones[pgstrom_shmem_maxzones])) +
		pgstrom_shmem_totalsize + SHMEM_BLOCKSZ;

	pgstrom_shmem_head = ShmemInitStruct("pgstrom_shmem_head",
										 length, &found);
	Assert(!found);

	memset(pgstrom_shmem_head, 0, length);
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
							"total size of shared memory segment in MB",
							NULL,
							&shmem_totalsize,
							2048,	/* 2GB */
							128,	/* 128MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	if ((shmem_totalsize % (SHMEM_BLOCKSZ >> 20)) != 0)
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
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	pgstrom_shmem_totalsize = ((Size)shmem_totalsize) << 20;

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

/*
 * Routines to get device properties.
 */
int
pgstrom_get_device_nums(void)
{
	pg_memory_barrier();
	return pgstrom_shmem_head->device_num;
}

pgstrom_device_info *
pgstrom_get_device_info(int index)
{
	pg_memory_barrier();
	if (index < 0 || index >= pgstrom_shmem_head->device_num)
		return NULL;
	return pgstrom_shmem_head->device_info[index];
}

#define PLDEVINFO_SHIFT(dest,src,field)						\
	(dest)->field = (char *)(dest) + ((src)->field - (char *)(src))

void
pgstrom_register_device_info(List *dev_list)
{
	pgstrom_platform_info  *pl_info;
	pgstrom_platform_info  *pl_info_sh;
	pgstrom_device_info **dev_array;
	ListCell   *cell;
	Size		length;
	int			index;

	Assert(pgstrom_shmem_head->device_num == 0);
	Assert(TopShmemContext != NULL);

	/* copy platform info into shared memory segment */
	pl_info = ((pgstrom_device_info *) linitial(dev_list))->pl_info;
	length = offsetof(pgstrom_platform_info, buffer[pl_info->buflen]);
	pl_info_sh = pgstrom_shmem_alloc(TopShmemContext, length);
	if (!pl_info_sh)
		elog(ERROR, "out of shared memory");
	memcpy(pl_info_sh, pl_info, length);
	PLDEVINFO_SHIFT(pl_info_sh, pl_info, pl_profile);
	PLDEVINFO_SHIFT(pl_info_sh, pl_info, pl_version);
	PLDEVINFO_SHIFT(pl_info_sh, pl_info, pl_name);
	PLDEVINFO_SHIFT(pl_info_sh, pl_info, pl_vendor);
	PLDEVINFO_SHIFT(pl_info_sh, pl_info, pl_extensions);

	/* copy device info into shared memory segment */
	length = sizeof(pgstrom_device_info *) * list_length(dev_list);
	dev_array = pgstrom_shmem_alloc(TopShmemContext, length);
	if (!dev_array)
		elog(ERROR, "out of shared memory");

	index = 0;
	foreach (cell, dev_list)
	{
		pgstrom_device_info	*dev_info = lfirst(cell);
		pgstrom_device_info *dest;

		length = offsetof(pgstrom_device_info, buffer[dev_info->buflen]);
		dest = pgstrom_shmem_alloc(TopShmemContext, length);
		if (!dest)
			elog(ERROR, "out of shared memory");
		memcpy(dest, dev_info, length);

		/* pointer adjustment */
		dest->pl_info = pl_info_sh;
		PLDEVINFO_SHIFT(dest, dev_info, dev_device_extensions);
		PLDEVINFO_SHIFT(dest, dev_info, dev_name);
		PLDEVINFO_SHIFT(dest, dev_info, dev_opencl_c_version);
		PLDEVINFO_SHIFT(dest, dev_info, dev_profile);
		PLDEVINFO_SHIFT(dest, dev_info, dev_vendor);
		PLDEVINFO_SHIFT(dest, dev_info, dev_version);
		PLDEVINFO_SHIFT(dest, dev_info, driver_version);

		dev_array[index++] = dest;
	}
	Assert(index == list_length(dev_list));

	pgstrom_shmem_head->device_info = dev_array;
	pgstrom_shmem_head->device_num = index;
	pg_memory_barrier();
}
