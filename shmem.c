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
#include "storage/barrier.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
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

	/* for message queues */
	shmem_context  *mqueue_context;
	pgstrom_queue  *mqueue_server;

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
	pg_crc32	magic_active;
	pg_crc32	magic_free;
	char		name[NAMEDATALEN];
} shmem_context;

typedef struct
{
	shmem_context  *context;
	Size			nblocks;		/* number of blocks allocated */
	dlist_node		chain;			/* lined of block_list of shmem_context */
	dlist_head		chunk_list;		/* list of chunks in this bunch */
	dlist_head		free_list;		/* list of free chunks in this bunch */
	dlist_node		first_chunk;
} shmem_bunch;

typedef struct
{
	dlist_node		chunk_chain;	/* linked to chunk_list of bunch */
	shmem_bunch	   *bunch;			/* back pointer to shmem_bunch */
	Size			chunk_sz;		/* length of this chunk */
	union {
		dlist_node	free_chain;
		char		userdata[1];
	};
} shmem_chunk;

#define SHMEM_CHUNK_MAGIC(chunk)				\
	*((pg_crc32 *)(((char *)(chunk)) + (chunk)->chunk_sz - sizeof(pg_crc32)))

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
	pg_crc32		crc;

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
	INIT_CRC32(crc);
	COMP_CRC32(crc, name, strlen(name));
	FIN_CRC32(crc);
	context->magic_active = crc;
	context->magic_free = crc ^ 0xaaaaaaaa;
	strcpy(context->name, name);

	SpinLockAcquire(&pgstrom_shmem_head->context_lock);
	dlist_push_tail(&pgstrom_shmem_head->context_list,
					&context->chain);
	SpinLockRelease(&pgstrom_shmem_head->context_lock);

	return context;
}

/*
 * NOTE: caller must hold a lock on this context
 */
static void
pgstrom_shmem_context_reset_nolock(shmem_context *context)
{
	dlist_mutable_iter	iter;

	dlist_foreach_modify(iter, &context->block_list)
	{
		shmem_bunch *bunch = dlist_container(shmem_bunch, chain, iter.cur);

		Assert(bunch->context == context);
		dlist_delete(&bunch->chain);
		pgstrom_shmem_block_free(bunch);
	}
	Assert(dlist_is_empty(&context->block_list));
	context->total_active = 0;
	context->num_blocks = 0;
	context->num_chunks = 0;
}

void
pgstrom_shmem_context_reset(shmem_context *context)
{
	SpinLockAcquire(&context->lock);
	pgstrom_shmem_context_reset_nolock(context);
	SpinLockRelease(&context->lock);
}

void
pgstrom_shmem_context_delete(shmem_context *context)
{
	Assert(context != TopShmemContext);

	SpinLockAcquire(&context->lock);
	/* remove from the global list */
	SpinLockAcquire(&pgstrom_shmem_head->context_lock);
	dlist_delete(&context->chain);
	SpinLockRelease(&pgstrom_shmem_head->context_lock);

	/* then, release all */
	pgstrom_shmem_context_reset_nolock(context);
	SpinLockRelease(&context->lock);

	pgstrom_shmem_free(context);
}

void *
pgstrom_shmem_alloc(shmem_context *context, Size size)
{
	dlist_iter	iter1;
	dlist_iter	iter2;
	Size		required;
	void	   *address = NULL;

	required = MAXALIGN(offsetof(shmem_chunk,
								 userdata[size + sizeof(pg_crc32)]));

	SpinLockAcquire(&context->lock);
	for (;;)
	{
		shmem_bunch	   *bunch;
		shmem_chunk	   *cchunk;
		Size			bunch_len;
		Size			threshold;

		dlist_foreach(iter1, &context->block_list)
		{
			bunch = dlist_container(shmem_bunch, chain, iter1.cur);

			Assert((Size)bunch % SHMEM_BLOCKSZ == 0);
			Assert(bunch->context == context);

			dlist_foreach(iter2, &bunch->free_list)
			{
				cchunk = dlist_container(shmem_chunk, free_chain, iter2.cur);

				Assert(cchunk->bunch == bunch);
				Assert(SHMEM_CHUNK_MAGIC(cchunk) == context->magic_free);

				/* is this free chunk has enough space to allocate it? */
				if (required > cchunk->chunk_sz)
					continue;

				/*
				 * Split this chunk, if the later portion has enough space to
				 * perform as a free chunk being longer than "NAMEDATALEN" just
				 * for threshold to avoid too small fraction.
				 */
				threshold = MAXALIGN(offsetof(shmem_chunk,
											  userdata[NAMEDATALEN +
													   sizeof(pg_crc32)]));
				if (required + threshold <= cchunk->chunk_sz)
				{
					shmem_chunk	   *nchunk
						= (shmem_chunk *)((char *)cchunk + required);

					nchunk->bunch = cchunk->bunch;
					nchunk->chunk_sz = cchunk->chunk_sz - required;
					cchunk->chunk_sz = required;
					SHMEM_CHUNK_MAGIC(nchunk) = context->magic_free;
					SHMEM_CHUNK_MAGIC(cchunk) = context->magic_active;
					dlist_insert_after(&cchunk->chunk_chain,
									   &nchunk->chunk_chain);
					dlist_insert_after(&cchunk->free_chain,
									   &nchunk->free_chain);
					dlist_delete(&cchunk->free_chain);

					context->num_chunks++;
				}
				else
				{
					dlist_delete(&cchunk->free_chain);
					SHMEM_CHUNK_MAGIC(cchunk) = context->magic_active;
				}
				address = cchunk->userdata;

				/* adjust statistic */
				context->total_active += cchunk->chunk_sz;
				goto out_unlock;
			}
		}
		/*
		 * No free space in the existing blocks, so allocate a new one.
		 */
		bunch_len = (offsetof(shmem_bunch, first_chunk) +
					 offsetof(shmem_chunk, userdata[required +
													sizeof(pg_crc32)]));
		bunch = pgstrom_shmem_block_alloc(bunch_len);
		if (!bunch)
			goto out_unlock;

		bunch->context = context;
		bunch->nblocks = (1 << find_least_pot(bunch_len));
		dlist_init(&bunch->chunk_list);
		dlist_init(&bunch->free_list);
		dlist_push_tail(&context->block_list, &bunch->chain);

		cchunk = (shmem_chunk *) &bunch->first_chunk;
		cchunk->bunch = bunch;
		cchunk->chunk_sz = (SHMEM_BLOCKSZ * bunch->nblocks -
							offsetof(shmem_bunch, first_chunk));
		SHMEM_CHUNK_MAGIC(cchunk) = context->magic_free;
		dlist_push_tail(&bunch->chunk_list, &cchunk->chunk_chain);
		dlist_push_tail(&bunch->free_list, &cchunk->free_chain);

		context->num_blocks += bunch->nblocks;
	}
out_unlock:
	SpinLockRelease(&context->lock);

	return address;
}

void
pgstrom_shmem_free(void *address)
{
	shmem_context  *context;
	shmem_bunch	   *bunch;
	shmem_chunk	   *cchunk;
	shmem_chunk	   *pchunk;
	shmem_chunk	   *nchunk;
	dlist_node	   *dnode;

	Assert(ADDRESS_IN_SHMEM(address));
	cchunk = (shmem_chunk *)((Size)address -
							 offsetof(shmem_chunk, userdata[0]));
	bunch = cchunk->bunch;
	Assert((Size)bunch == ((Size)address & ~((1UL << bunch->nblocks) - 1)));

	context = bunch->context;
	Assert(SHMEM_CHUNK_MAGIC(cchunk) == context->magic_free);

	SpinLockAcquire(&context->lock);
#ifdef PGSTROM_DEBUG
	do {
		dlist_iter	iter;
		bool		found = false;

		dlist_foreach(iter, &bunch->chunk_list)
		{
			cchunk = dlist_container(shmem_chunk, chunk_chain, iter.cur);

			if (cchunk->userdata == address)
				found = true;
			Assert(SHMEM_CHUNK_MAGIC(cchunk) == context->magic_free ||
				   SHMEM_CHUNK_MAGIC(cchunk) == context->magic_active);
		}
		Assert(found);
	} while(0);
#endif
	/* adjust statistics */
	context->total_active -= cchunk->chunk_sz;

	/*
	 * merge with previous chunk, if it is also free.
	 */
	if (dlist_has_prev(&bunch->chunk_list, &cchunk->chunk_chain))
	{
		dnode = dlist_prev_node(&bunch->chunk_list, &cchunk->chunk_chain);
		pchunk = dlist_container(shmem_chunk, chunk_chain, dnode);

		if (SHMEM_CHUNK_MAGIC(pchunk) == context->magic_free)
		{
			context->num_chunks--;
			dlist_delete(&cchunk->chunk_chain);
			pchunk->chunk_sz += cchunk->chunk_sz;
			dlist_delete(&pchunk->free_chain);
			cchunk = pchunk;
		}
	}

	/*
	 * merge with next chunk, if it is also free.
	 */
	if (dlist_has_next(&bunch->chunk_list, &cchunk->chunk_chain))
	{
		dnode = dlist_next_node(&bunch->chunk_list, &cchunk->chunk_chain);
		nchunk = dlist_container(shmem_chunk, chunk_chain, dnode);

		if (SHMEM_CHUNK_MAGIC(nchunk) == context->magic_free)
		{
			context->num_chunks--;
			cchunk->chunk_sz += nchunk->chunk_sz;
			dlist_delete(&nchunk->chunk_chain);
			dlist_delete(&nchunk->free_chain);
		}
	}

	/* mark it as free chunk */
	SHMEM_CHUNK_MAGIC(cchunk) = context->magic_free;
	dlist_push_head(&bunch->free_list, &cchunk->free_chain);

	/*
	 * In case when this chunk is the last one in this bunch and
	 * became free, it it time to release this bunch also.
	 */
	if (dlist_head_node(&bunch->chunk_list) == &cchunk->chunk_chain &&
		!dlist_has_next(&bunch->chunk_list, &cchunk->chunk_chain))
	{
		Assert((void *)&bunch->first_chunk == (void *)&cchunk->chunk_chain);
		dlist_delete(&bunch->chain);
		context->num_blocks -= bunch->nblocks;
		context->num_chunks--;
		pgstrom_shmem_block_free(bunch);
	}
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
	text	   *crc32;
} shmem_context_info;

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
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "n_chunks",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "crc32",
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
				cxt_info->crc32
					= cstring_to_text(psprintf("%08x,%08x",
											   context->magic_active,
											   context->magic_free));
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
	values[5] = PointerGetDatum(cxt_info->crc32);

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
	shmem_context  *context;
	Size			length;

	/* allocate a block */
	bunch = pgstrom_shmem_zone_block_alloc(zone, SHMEM_BLOCKSZ);
	if (!bunch)
		elog(ERROR, "out of shared memory");

	/* setup a bunch */
	bunch->nblocks = 1;
	dlist_init(&bunch->chunk_list);
	dlist_init(&bunch->free_list);

	/* 1st active chunk for top shared memory context */
	length = MAXALIGN(offsetof(shmem_chunk, userdata[sizeof(shmem_context) +
													 sizeof(pg_crc32)]));
	chunk1 = (shmem_chunk *)&bunch->first_chunk;
	chunk1->bunch = bunch;
	chunk1->chunk_sz = length;

	/* contents of chunk1 is memory context */
	context = (shmem_context *)chunk1->userdata;
	context->owner = getpid();
	SpinLockInit(&context->lock);
	dlist_init(&context->block_list);
	context->total_active = chunk1->chunk_sz;
	context->num_blocks = bunch->nblocks;
	context->num_chunks = 2;
	strcpy(context->name, "Top Shared Memory Context");
	INIT_CRC32(context->magic_active);
	COMP_CRC32(context->magic_active, context->name, strlen(context->name));
	FIN_CRC32(context->magic_active);
	context->magic_free = context->magic_active ^ 0xaaaaaaaa;

	dlist_push_head(&context->block_list, &bunch->chain);

	SHMEM_CHUNK_MAGIC(chunk1) = context->magic_active;
	dlist_push_tail(&bunch->chunk_list, &chunk1->chunk_chain);

	/* 2nd free chunk */
	chunk2 = (shmem_chunk *)((char *)chunk1 + chunk1->chunk_sz);
	chunk2->bunch = bunch;
	chunk2->chunk_sz = (SHMEM_BLOCKSZ * bunch->nblocks -
						offsetof(shmem_bunch, first_chunk) -
						chunk1->chunk_sz);
	SHMEM_CHUNK_MAGIC(chunk2) = context->magic_free;
	dlist_push_tail(&bunch->free_list, &chunk2->free_chain);
    dlist_push_tail(&bunch->chunk_list, &chunk2->chunk_chain);

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

/*
 * shared facility for message queue
 */
shmem_context *
pgstrom_get_mqueue_context(void)
{
	Assert(pgstrom_shmem_head->mqueue_context != NULL);
	return pgstrom_shmem_head->mqueue_context;
}

pgstrom_queue *
pgstrom_get_server_mqueue(void)
{
	Assert(pgstrom_shmem_head->mqueue_server != NULL);
	return pgstrom_shmem_head->mqueue_server;
}

void
pgstrom_shmem_mqueue_setup(void)
{
	shmem_context  *context;

	context = pgstrom_shmem_context_create("PG-Strom Message Queue");
	if (!context)
		elog(ERROR, "failed to create shared memory context");
	Assert(pgstrom_shmem_head->mqueue_context == NULL);
	pgstrom_shmem_head->mqueue_context = context;

	/* create a message queue for OpenCL background server */
	pgstrom_shmem_head->mqueue_server = pgstrom_create_queue(true);
}
