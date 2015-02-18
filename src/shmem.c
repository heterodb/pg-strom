/*
 * shmem.c
 *
 * Management of shared memory segment
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
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "common/pg_crc.h"
#include "funcapi.h"
#include "lib/ilist.h"
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
typedef struct
{
	dlist_node		chain;		/* to be chained free_list of shmem_zone */
	Size			blocksz;	/* block size in bytes if head */
} shmem_block;

#define BLOCK_IS_ACTIVE(block)			\
	(((block)->chain.next == NULL  &&	\
	  (block)->chain.prev == NULL) && 	\
	 (block)->blocksz > 0)
#define BLOCK_IS_FREE(block)			\
	(((block)->chain.next != NULL  ||	\
	  (block)->chain.prev != NULL) &&	\
	 (block)->blocksz > 0)

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

typedef struct {
	dlist_node	chain;		/* link to the free list */
	const char *filename;
	uint32		lineno;
	pid_t		owner;
	Datum		data[FLEXIBLE_ARRAY_MEMBER];
} shmem_slab;

typedef struct {
	dlist_node	chain;
	int			slab_index;	/* index of slab_sizes */
	int			slab_nums;	/* number of slabs per block */
	shmem_slab	entry;	/* first entry in this block */
} shmem_slab_head;

#define SHMEM_SLAB_SIZE(div)									\
	MAXALIGN_DOWN((SHMEM_BLOCKSZ - SHMEM_ALLOC_COST) / (div) -	\
				  (offsetof(shmem_slab, data) + sizeof(cl_uint)))
static size_t slab_sizes[] = {
	SHMEM_SLAB_SIZE(56),	/* about 100B */
	SHMEM_SLAB_SIZE(30),	/* about 240B */
	SHMEM_SLAB_SIZE(15),	/* about 512B */
	SHMEM_SLAB_SIZE(6),		/* about 1.2KB */
	SHMEM_SLAB_SIZE(3),		/* about 2.5KB */
};
#undef SHMEM_SLAB_SIZE

typedef struct
{
	slock_t		slab_locks[lengthof(slab_sizes)];
	dlist_head	slab_freelist[lengthof(slab_sizes)];
	dlist_head	slab_blocklist[lengthof(slab_sizes)];

	/* for zone management */
	bool		is_ready;
	int			num_zones;
	void	   *zone_baseaddr;
	Size		zone_length;
	shmem_zone *zones[FLEXIBLE_ARRAY_MEMBER];
} shmem_head;

typedef struct {
	uint32		magic;			/* = SHMEM_BODY_MAGIC */
	pid_t		owner;
	const char *filename;
	int			lineno;
	Datum		data[FLEXIBLE_ARRAY_MEMBER];
} shmem_body;

/* XXX - we need to ensure SHMEM_ALLOC_COST is enough large */

#define SHMEM_BODY_MAGIC		0xabadcafe
#define SHMEM_BLOCK_MAGIC		0xdeadbeaf
#define SHMEM_SLAB_MAGIC		0xabadf11e

#define ADDRESS_IN_SHMEM(address)										\
	((Size)(address) >= (Size)pgstrom_shmem_head->zone_baseaddr &&		\
	 (Size)(address) < ((Size)pgstrom_shmem_head->zone_baseaddr +		\
						pgstrom_shmem_totalsize))
#define ADDRESS_IN_SHMEM_ZONE(zone,address)					\
	((Size)(address) >= (Size)(zone)->block_baseaddr &&		\
	 (Size)(address) < ((Size)(zone)->block_baseaddr +		\
						(zone)->num_blocks) * SHMEM_BLOCKSZ)

/* static variables */
static shmem_startup_hook_type shmem_startup_hook_next;
static Size			pgstrom_shmem_totalsize;
static int			pgstrom_shmem_maxzones;
static shmem_head  *pgstrom_shmem_head;

/*
 * find_least_pot
 *
 * It looks up the least power of two that is equal or larger than
 * the provided size.
 */
static inline int
find_least_pot(Size size)
{
	int		shift = 0;

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
	if ((size & 0x00000001UL) != 0)
		shift += 1;
	return Max(shift, SHMEM_BLOCKSZ_BITS) - SHMEM_BLOCKSZ_BITS;
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
	int				i;

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
	block->blocksz = (1 << (shift-1)) * SHMEM_BLOCKSZ;
	index = block - &zone->blocks[0];
	Assert((index & ((1UL << shift) - 1)) == 0);
	dlist_push_tail(&zone->free_list[shift-1], &block->chain);

	/* is it exactly block head? */
	for (i=1; i < (1 << shift); i++)
		Assert(!BLOCK_IS_ACTIVE(block+i) && !BLOCK_IS_FREE(block+i));

	block += (1 << (shift - 1));
	block->blocksz = (1 << (shift-1)) * SHMEM_BLOCKSZ;
	dlist_push_tail(&zone->free_list[shift-1], &block->chain);

	zone->num_free[shift-1] += 2;

	return true;
}

static void *
pgstrom_shmem_zone_block_alloc(shmem_zone *zone,
							   const char *filename, int lineno, Size size)
{
	shmem_block	*block;
	shmem_body	*body;
	dlist_node	*dnode;
	Size	total_size;
	int		shift;
	int		index;
	void   *address;
	int		i;

	total_size = offsetof(shmem_body, data[0]) + size + sizeof(cl_uint);
	if (total_size > (1UL << SHMEM_BLOCKSZ_BITS_MAX))
		return NULL;	/* too large size required */
	shift = find_least_pot(total_size);
	if (dlist_is_empty(&zone->free_list[shift]))
	{
		if (!pgstrom_shmem_zone_block_split(zone, shift+1))
			return NULL;
	}
	Assert(!dlist_is_empty(&zone->free_list[shift]));

	dnode = dlist_pop_head_node(&zone->free_list[shift]);
	block = dlist_container(shmem_block, chain, dnode);
	Assert(block->blocksz == (1UL << shift) * SHMEM_BLOCKSZ);

	memset(block, 0, sizeof(shmem_block));
	block->blocksz = size;

	/* non-head block are zero cleared? */
	for (i=1; i < (1 << shift); i++)
		Assert(!BLOCK_IS_ACTIVE(block+i) && !BLOCK_IS_FREE(block+i));

	index = block - &zone->blocks[0];
	body = (shmem_body *)((char *)zone->block_baseaddr +
						  index * SHMEM_BLOCKSZ);
	zone->num_free[shift]--;
	zone->num_active[shift]++;

	/* tracking info */
	body->magic = SHMEM_BODY_MAGIC;
	body->owner = getpid();
	body->filename = filename;	/* must be static cstring! */
	body->lineno = lineno;
	address = (void *)body->data;

	/* to detect overrun */
	*((cl_uint *)((uintptr_t)address + size)) = SHMEM_BLOCK_MAGIC;

	return address;
}

static void
pgstrom_shmem_zone_block_free(shmem_zone *zone, shmem_body *body)
{
	shmem_block	   *block;
	long			index;
	long			shift;

	Assert(ADDRESS_IN_SHMEM_ZONE(zone, body));

	index = ((uintptr_t)body -
			 (uintptr_t)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = &zone->blocks[index];
	Assert(BLOCK_IS_ACTIVE(block));
	/* detect overrun */
	Assert(*((cl_uint *)((uintptr_t)body->data +
						 block->blocksz)) == SHMEM_BLOCK_MAGIC);
	shift = find_least_pot(block->blocksz +
						   offsetof(shmem_body, data[0]) +
						   sizeof(cl_uint));
	Assert(shift <= SHMEM_BLOCKSZ_BITS_RANGE);
	Assert((index & ~((1UL << shift) - 1)) == index);

	zone->num_active[shift]--;

	/* try to merge buddy blocks if it is also free */
	while (shift < SHMEM_BLOCKSZ_BITS_RANGE)
	{
		shmem_block	   *buddy;
		long			buddy_index = index ^ (1UL << shift);

		if (buddy_index + (1UL << shift) >= zone->num_blocks)
			break;

		buddy = &zone->blocks[buddy_index];
		/*
		 * The buddy block can be merged if it is also free and same size.
		 */
		if (BLOCK_IS_ACTIVE(buddy) ||
			buddy->blocksz != (1UL << shift) * SHMEM_BLOCKSZ)
			break;
		/* ensure buddy is block head */
		Assert(BLOCK_IS_FREE(buddy));

		dlist_delete(&buddy->chain);
		if (buddy_index < index)
		{
			/* mark this block is not a head */
			memset(block, 0, sizeof(shmem_block));
			block = buddy;
			index = buddy_index;
		}
		else
		{
			/* mark this block is not a head */
			memset(buddy, 0, sizeof(shmem_block));
		}
		zone->num_free[shift]--;
		shift++;
	}
	zone->num_free[shift]++;
	block->blocksz = (1UL << shift) * SHMEM_BLOCKSZ;
	dlist_push_head(&zone->free_list[shift], &block->chain);
}

/*
 * pgstrom_alloc_slab
 */
static void *
pgstrom_alloc_slab(const char *filename, int lineno, int index)
{
	shmem_slab_head *sblock;
	shmem_slab	   *entry;
	dlist_node	   *dnode;
	Size			slab_sz = slab_sizes[index];
	Size			unitsz = MAXALIGN(offsetof(shmem_slab, data[0]) +
									  INTALIGN(slab_sz) + sizeof(cl_uint));
	SpinLockAcquire(&pgstrom_shmem_head->slab_locks[index]);
	if (dlist_is_empty(&pgstrom_shmem_head->slab_freelist[index]))
	{
		Size		length;
		Size		offset;
		int			count;

		/* allocate a block */
		sblock = __pgstrom_shmem_alloc_alap(filename, lineno,
											sizeof(shmem_slab_head), &length);
		if (!sblock)
		{
			SpinLockRelease(&pgstrom_shmem_head->slab_locks[index]);
			return NULL;
		}
		dlist_push_tail(&pgstrom_shmem_head->slab_blocklist[index],
						&sblock->chain);

		for (offset = offsetof(shmem_slab_head, entry), count=0;
			 offset + unitsz <= length;
			 offset += unitsz, count++)
		{
			entry = (shmem_slab *)((char *)sblock + offset);
			/* set magic number */
			*((uint32 *)((char *)entry->data +
						 INTALIGN(slab_sz))) = SHMEM_SLAB_MAGIC;
			dlist_push_head(&pgstrom_shmem_head->slab_freelist[index],
							&entry->chain);
		}
		sblock->slab_index = index;
		sblock->slab_nums = count;
	}
	Assert(!dlist_is_empty(&pgstrom_shmem_head->slab_freelist[index]));
	dnode = dlist_pop_head_node(&pgstrom_shmem_head->slab_freelist[index]);
	entry = dlist_container(shmem_slab, chain, dnode);
	memset(&entry->chain, 0, sizeof(dlist_node));
	entry->owner = getpid();
	entry->filename = filename;
	entry->lineno = lineno;
	SpinLockRelease(&pgstrom_shmem_head->slab_locks[index]);

	return (void *)entry->data;
}

/*
 * pgstrom_shmem_block_alloc
 *
 * It is an internal API; that allocates a continuous 2^N blocks from
 * a particular shared memory zone. It tries to split a larger memory blocks
 * if suitable memory blocks are not free. If no memory blocks are available,
 * it goes into another zone to allocate memory.
 */
void *
__pgstrom_shmem_alloc(const char *filename, int lineno, Size size)
{
	static int	zone_index = 0;
	int			start;
	shmem_zone *zone;
	void	   *address;
	int			i;

	/* does shared memory segment already set up? */
	if (!pgstrom_shmem_head->is_ready)
	{
		elog(LOG, "PG-Strom's shared memory segment has not been ready");
		return NULL;
	}

	/*
	 * Size check whether we should allocate bare-blocks, or a piece of
	 * slabs. If required size is unable to allocate, return NULL soon.
	 */
	if (size == 0 || size > (1UL << SHMEM_BLOCKSZ_BITS_MAX))
		return NULL;
	for (i=0; i < lengthof(slab_sizes); i++)
	{
		if (size <= slab_sizes[i])
			return pgstrom_alloc_slab(filename, lineno, i);
	}

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

		address = pgstrom_shmem_zone_block_alloc(zone, filename, lineno, size);

		SpinLockRelease(&zone->lock);

		if (address)
			break;

		zone_index = (zone_index + 1) % pgstrom_shmem_head->num_zones;
	} while (zone_index != start);
#ifdef PGSTROM_DEBUG
	/* For debugging, we dump current status of shared memory segment
	 * if we have to return "out of shared memory" error */
	if (!address)
	{
		pgstrom_shmem_dump();
		clserv_log("%s:%d required %zu bytes of shared memory",
				   filename, lineno, size);
	}
#endif
	return address;	
}

/*
 * pgstrom_shmem_alloc_alap
 *
 * pgstrom_shmem_alloc "as large as possible"
 * It is unavoidable to make unused memory area in buddy memory allocation
 * algorithm. In case when we want to acquire a memory block larget than
 * a particular size, likely toast buffer, best storategy is to allocate
 * least 2^N block larger than required size.
 * This function round up the required size into the best-fit one.
 *
 * Also note that, it never falls to slab.
 */
void *
__pgstrom_shmem_alloc_alap(const char *filename, int lineno,
						   Size required, Size *allocated)
{
	int		shift = find_least_pot(required + sizeof(cl_uint));
	void   *result;

	required = (1UL << (shift + SHMEM_BLOCKSZ_BITS))
		- offsetof(shmem_body, data[0])
		- sizeof(cl_uint);
	result = __pgstrom_shmem_alloc(filename, lineno, required);
	if (result && allocated)
		*allocated = required;
	return result;
}

/*
 * pgstrom_free_slab
 */
static void
pgstrom_free_slab(shmem_slab_head *sblock, shmem_slab *entry)
{
	int		index = sblock->slab_index;

	Assert(!entry->chain.next && !entry->chain.prev);
	Assert(*((uint32 *)((char *)entry->data +
						INTALIGN(slab_sizes[index]))) == SHMEM_SLAB_MAGIC);
	SpinLockAcquire(&pgstrom_shmem_head->slab_locks[index]);
	dlist_push_head(&pgstrom_shmem_head->slab_freelist[index],
					&entry->chain);
	SpinLockRelease(&pgstrom_shmem_head->slab_locks[index]);
}

void
pgstrom_shmem_free(void *address)
{
	shmem_zone *zone;
	shmem_body *body;
	void	   *zone_baseaddr;
	Size		zone_length ;
	int			zone_index;
	Size		offset;

	offset = ((uintptr_t)address & (SHMEM_BLOCKSZ - 1));
	if (offset != offsetof(shmem_body, data[0]))
	{
		shmem_slab_head *sblock = (shmem_slab_head *)
			((char *)address - offset + offsetof(shmem_body, data));
		shmem_slab		*entry = (shmem_slab *)
			((char *)address - offsetof(shmem_slab, data[0]));
		pgstrom_free_slab(sblock, entry);
		return;
	}

	/* elsewhere, the supplied address is bare blocks */
	zone_baseaddr = pgstrom_shmem_head->zone_baseaddr;
	zone_length = pgstrom_shmem_head->zone_length;

	Assert(pgstrom_shmem_sanitycheck(address));
	body = (shmem_body *)((char *)address -
						  offsetof(shmem_body, data[0]));

	zone_index = ((uintptr_t)body - (uintptr_t)zone_baseaddr) / zone_length;
	Assert(zone_index >= 0 && zone_index < pgstrom_shmem_head->num_zones);

	zone = pgstrom_shmem_head->zones[zone_index];
	SpinLockAcquire(&zone->lock);
	pgstrom_shmem_zone_block_free(zone, body);
	SpinLockRelease(&zone->lock);
}

/*
 * pgstrom_shmem_realloc
 *
 * It allocate a shared memory block, and copy the contents in the supplied
 * oldaddr to the new one, then release shared memory block.
 */
void *
__pgstrom_shmem_realloc(const char *filename, int lineno,
						void *oldaddr, Size newsize)
{
	void   *newaddr;

	newaddr = __pgstrom_shmem_alloc(filename, lineno, newsize);
	if (!newaddr)
		return NULL;
	if (oldaddr)
	{
		Size	oldsize = pgstrom_shmem_getsize(oldaddr);

		memcpy(newaddr, oldaddr, Min(newsize, oldsize));
		pgstrom_shmem_free(oldaddr);
	}
	return newaddr;
}

/*
 * pgstrom_shmem_getsize
 *
 * It returns size of the supplied active block
 */
Size
pgstrom_shmem_getsize(void *address)
{
	shmem_zone *zone;
	shmem_body *body;
	shmem_block *block;
	void	   *zone_baseaddr = pgstrom_shmem_head->zone_baseaddr;
	Size		zone_length = pgstrom_shmem_head->zone_length;
	Size		blocksz;
	long		index;

	/* find a zone on which address belongs to */
	Assert(pgstrom_shmem_sanitycheck(address));
	body = (shmem_body *)((char *)address -
						  offsetof(shmem_body, data[0]));
	index = ((uintptr_t)body -
			 (uintptr_t)zone_baseaddr) / zone_length;
	Assert(index >= 0 && index < pgstrom_shmem_head->num_zones);
	zone = pgstrom_shmem_head->zones[index];

	/* find shmem_block and get its status */
	SpinLockAcquire(&zone->lock);
	index = ((uintptr_t)body -
			 (uintptr_t)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = &zone->blocks[index];
	Assert(BLOCK_IS_ACTIVE(block));
	blocksz = block->blocksz;
    pgstrom_shmem_zone_block_free(zone, body);
    SpinLockRelease(&zone->lock);

	return blocksz;
}

/*
 * pgstrom_shmem_zone_length
 *
 * it returns the configured zone length
 */
Size
pgstrom_shmem_zone_length(void)
{
	return pgstrom_shmem_head->zone_length;
}

/*
 * pgstrom_shmem_maxalloc
 *
 * it returns the length of maximum allocatable length
 */
Size
pgstrom_shmem_maxalloc(void)
{
	static Size		maxalloc_length = 0;

	if (!maxalloc_length)
	{
		Size	zone_length = pgstrom_shmem_head->zone_length;
		int		nbits;

		zone_length = Min(zone_length, (1UL << SHMEM_BLOCKSZ_BITS_MAX));
		nbits = get_next_log2(zone_length + 1);

		maxalloc_length = ((1UL << (nbits - 1)) -	/* half of zone */
						   offsetof(shmem_body, data[0]) -
						   sizeof(cl_uint));
	}
	return maxalloc_length;
}

/*
 * pgstrom_init_slab
 *
 * init slab management structure
 */
static void
pgstrom_init_slab(void)
{
	int		i;

	for (i=0; i < lengthof(slab_sizes); i++)
	{
		SpinLockInit(&pgstrom_shmem_head->slab_locks[i]);
		dlist_init(&pgstrom_shmem_head->slab_freelist[i]);
		dlist_init(&pgstrom_shmem_head->slab_blocklist[i]);
	}
}

/*
 * pgstrom_shmem_slab_info
 *
 * shows list of slabs being allocated
 */
typedef struct
{
	void	   *address;
	const char *filename;
	int			lineno;
	int			index;
	uint32		owner;
	bool		active;
	bool		broken;
} shmem_slab_info;

static void
collect_shmem_slab_info(List **p_results,
						slock_t *slab_lock,
						dlist_head *slab_blocklist,
						size_t slab_size)
{
	SpinLockAcquire(slab_lock);
	PG_TRY();
	{
		dlist_iter	iter;
		Size		unitsz = MAXALIGN(offsetof(shmem_slab, data[0]) +
									  INTALIGN(slab_size) +
									  sizeof(uint32));
		dlist_foreach (iter, slab_blocklist)
		{
			shmem_slab_head *sblock;
			shmem_slab *entry;
			uint32	   *magic;
			int			count;

			sblock = dlist_container(shmem_slab_head, chain, iter.cur);
			for (count=0; count < sblock->slab_nums; count++)
			{
				shmem_slab_info	*slinfo = palloc0(sizeof(shmem_slab_info));

				entry = (shmem_slab *)((char *)&sblock->entry +
									   unitsz * count);
				magic = (uint32 *)((char *)entry->data + INTALIGN(slab_size));
				slinfo->address   = entry->data;
				slinfo->filename  = entry->filename;
				slinfo->lineno    = entry->lineno;
				slinfo->index     = sblock->slab_index;
				slinfo->owner     = entry->owner;
				slinfo->active    = (!entry->chain.prev && !entry->chain.next);
				if (*magic != SHMEM_SLAB_MAGIC ||
					(!entry->chain.prev && entry->chain.next) ||
					(entry->chain.prev && !entry->chain.next))
					slinfo->broken = true;
				else
					slinfo->broken = false;

				*p_results = lappend(*p_results, slinfo);
			}
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(slab_lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(slab_lock);
}

Datum
pgstrom_shmem_slab_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	   *fncxt;
	shmem_slab_info	   *slinfo;
	HeapTuple			tuple;
	Datum				values[6];
	bool				isnull[6];
	char				buf[256];

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		int				i;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(6, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "address",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "slabname",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "owner",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "location",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "active",
						   BOOLOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "broken",
						   BOOLOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		for (i=0; i < lengthof(slab_sizes); i++)
		{

			collect_shmem_slab_info((List **)&fncxt->user_fctx,
									&pgstrom_shmem_head->slab_locks[i],
									&pgstrom_shmem_head->slab_blocklist[i],
									slab_sizes[i]);
		}
		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
		SRF_RETURN_DONE(fncxt);

	slinfo = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	values[0] = Int64GetDatum((uint64)slinfo->address);
	snprintf(buf, sizeof(buf), "slab-%zu", slab_sizes[slinfo->index]);
	values[1] = CStringGetTextDatum(buf);
	if (slinfo->active)
	{
		values[2] = Int32GetDatum((uint32)slinfo->owner);
		snprintf(buf, sizeof(buf), "%s:%d", slinfo->filename, slinfo->lineno);
		values[3] = CStringGetTextDatum(buf);
	}
	else
	{
		isnull[2] = true;
		isnull[3] = true;
	}
	values[4] = BoolGetDatum(slinfo->active);
	values[5] = BoolGetDatum(slinfo->broken);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

    SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_slab_info);

/*
 * pgstrom_shmem_sanitycheck
 *
 * it checks whether magic number of the supplied shared-memory block is
 * still valid, or not. If someone overuses the block, magic number should
 * be broken and we can detect it.
 */
bool
pgstrom_shmem_sanitycheck(const void *address)
{
	shmem_zone	   *zone;
	shmem_block	   *block;
	shmem_body	   *body;
	void		   *zone_baseaddr = pgstrom_shmem_head->zone_baseaddr;
	Size			zone_length = pgstrom_shmem_head->zone_length;
	int				zone_index;
	int				block_index;
	cl_uint		   *p_magic;

	body = (shmem_body *)((char *)address -
						  offsetof(shmem_body, data[0]));
	Assert((uintptr_t)body % SHMEM_BLOCKSZ == 0);
	Assert(body->magic == SHMEM_BODY_MAGIC);

	zone_index = ((uintptr_t)body - (uintptr_t)zone_baseaddr) / zone_length;
	Assert(zone_index >= 0 && zone_index < pgstrom_shmem_head->num_zones);

	zone = pgstrom_shmem_head->zones[zone_index];
	Assert(ADDRESS_IN_SHMEM_ZONE(zone, body));

	block_index = ((uintptr_t)body -
				   (uintptr_t)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = &zone->blocks[block_index];
	Assert(BLOCK_IS_ACTIVE(block));

	p_magic = (cl_uint *)((char *)address + block->blocksz);

	return (*p_magic == SHMEM_BLOCK_MAGIC ? true : false);
}

/*
 * pgstrom_shmem_dump
 *
 * it logs current layout of shared memory segment
 */
#define DUMP(fmt,...)						\
	do {									\
		if (pgstrom_i_am_clserv)			\
			clserv_log(fmt,__VA_ARGS__);	\
		else								\
			elog(INFO,fmt,__VA_ARGS__);		\
	} while(0)

static void
pgstrom_shmem_dump_zone(shmem_zone *zone, int zone_index)
{
	long	i = 0;

	while (i < zone->num_blocks)
	{
		shmem_block	   *block = &zone->blocks[i];

		if (BLOCK_IS_ACTIVE(block))
		{
			shmem_body	   *body;
			cl_uint		   *p_magic;
			int				nshift;

			nshift = find_least_pot(offsetof(shmem_body, data[0]) +
									block->blocksz + sizeof(cl_uint));
			body = (shmem_body *)((char *)zone->block_baseaddr +
								  i * SHMEM_BLOCKSZ);
			p_magic = (cl_uint *)((char *)body->data + block->blocksz);

			DUMP("[%d:% 6ld] %p (size=%zu, owner=%u, %s:%d%s%s)",
				 zone_index, i, body,
				 block->blocksz,
				 body->owner,
				 body->filename,
				 body->lineno,
				 body->magic != SHMEM_BODY_MAGIC ? ", broken" : "",
				 *p_magic != SHMEM_BLOCK_MAGIC ? ", overrun" : "");
			i += (1 << nshift);
		}
		else if (BLOCK_IS_FREE(block))
		{
			int		nshift = find_least_pot(block->blocksz);
			Assert(nshift <= SHMEM_BLOCKSZ_BITS_RANGE);
            i += (1 << nshift);
		}
		else
		{
			DUMP("[%d:% 6ld] %p corrupted; neither active nor free",
				 zone_index, i,
				 (char *)zone->block_baseaddr + i * SHMEM_BLOCKSZ);
			break;
		}
	}
}
#undef DUMP

void
pgstrom_shmem_dump(void)
{
	int		i;

	for (i=0; i < pgstrom_shmem_head->num_zones; i++)
	{
		shmem_zone *zone = pgstrom_shmem_head->zones[i];

		SpinLockAcquire(&zone->lock);
		pgstrom_shmem_dump_zone(zone, i);
		SpinLockRelease(&zone->lock);
	}
}

/*
 * collect_shmem_info
 *
 * It collects statistical information of shared memory zone.
 * Note that it does not trust statistical values if debug build, thus
 * it may take longer time because of walking of shared memory zone.
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

		if (BLOCK_IS_ACTIVE(block))
		{
			int		nshift = find_least_pot(offsetof(shmem_body, data[0]) +
											block->blocksz +
											sizeof(cl_uint));
			Assert(nshift <= SHMEM_BLOCKSZ_BITS_RANGE);
			num_active[nshift]++;
			i += (1 << nshift);
		}
		else if (BLOCK_IS_FREE(block))
		{
			int		nshift = find_least_pot(block->blocksz);

			Assert(nshift <= SHMEM_BLOCKSZ_BITS_RANGE);
			num_free[nshift]++;
			i += (1 << nshift);
		}
		else
			elog(ERROR, "block %ld is neither active nor free", i);
	}
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
		block_info->num_active = num_active[i];
		block_info->num_free = num_free[i];

		results = lappend(results, block_info);
	}
	return results;
}

Datum
pgstrom_shmem_info(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_shmem_info);

typedef struct
{
	int			zone;
	void	   *address;
	cl_uint		size;
	pid_t		owner;
	const char *filename;
	int			lineno;
	bool		broken;
	bool		overrun;
} shmem_active_info;

static List *
collect_shmem_active_info(shmem_zone *zone, int zone_index)
{
	List	   *results = NIL;
	shmem_body *body;
	long		i;

	i = 0;
	while (i < zone->num_blocks)
	{
		shmem_block	*block = &zone->blocks[i];

		if (BLOCK_IS_ACTIVE(block))
		{
			shmem_active_info *sainfo;
			cl_uint	   *p_magic;
			int		nshift = find_least_pot(offsetof(shmem_body, data[0]) +
											block->blocksz +
											sizeof(cl_uint));

			body = (shmem_body *)((char *)zone->block_baseaddr +
								  i * SHMEM_BLOCKSZ);
			sainfo = palloc0(sizeof(shmem_active_info));
			sainfo->zone = zone_index;
			sainfo->address = body->data;
			sainfo->size = block->blocksz;
			sainfo->owner = body->owner;
			sainfo->filename = body->filename;
			sainfo->lineno = body->lineno;
			if (body->magic != SHMEM_BODY_MAGIC)
				sainfo->broken = true;
			p_magic = (cl_uint *)((char *)body->data + block->blocksz);
			if (*p_magic != SHMEM_BLOCK_MAGIC)
				sainfo->overrun = true;
			results = lappend(results, sainfo);

			Assert(nshift <= SHMEM_BLOCKSZ_BITS_RANGE);
			i += (1 << nshift);
		}
		else if (BLOCK_IS_FREE(block))
		{
			int		nshift = find_least_pot(block->blocksz);

			Assert(nshift <= SHMEM_BLOCKSZ_BITS_RANGE);
			i += (1 << nshift);
		}
		else
			elog(ERROR, "block %ld is neither active nor free", i);
	}
	return results;
}

Datum
pgstrom_shmem_active_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	   *fncxt;
	shmem_active_info  *sainfo;
	HeapTuple	tuple;
	Datum		values[8];
	bool		isnull[8];
	char		buf[256];

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		shmem_zone	   *zone;
		List		   *active_info_list = NIL;
		int				i;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(7, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "zone",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "address",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "size",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "owner",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "location",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "broken",
						   BOOLOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 7, "overrun",
						   BOOLOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		for (i=0; i < pgstrom_shmem_head->num_zones; i++)
		{
			zone = pgstrom_shmem_head->zones[i];

			SpinLockAcquire(&zone->lock);
			PG_TRY();
			{
				List   *temp = collect_shmem_active_info(zone, i);

				active_info_list = list_concat(active_info_list, temp);
			}
			PG_CATCH();
			{
				SpinLockRelease(&zone->lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockRelease(&zone->lock);
		}
		fncxt->user_fctx = active_info_list;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
		SRF_RETURN_DONE(fncxt);

	sainfo = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(sainfo->zone);
	values[1] = Int64GetDatum(sainfo->address);
	values[2] = Int32GetDatum(sainfo->size);
	values[3] = Int32GetDatum(sainfo->owner);
	snprintf(buf, sizeof(buf), "%s:%d", sainfo->filename, sainfo->lineno);
	values[4] = CStringGetTextDatum(buf);
	values[5] = BoolGetDatum(sainfo->broken);
	values[6] = BoolGetDatum(sainfo->overrun);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_active_info);

/*
 * Debugging facilities for shmem.c
 */
Datum
pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	Size	size = PG_GETARG_INT64(0);
	void   *address;

	address = pgstrom_shmem_alloc(size);

	PG_RETURN_INT64((Size) address);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_alloc_func);

Datum
pgstrom_shmem_free_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	void		   *address = (void *) PG_GETARG_INT64(0);

	pgstrom_shmem_free(address);

	PG_RETURN_BOOL(true);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_free_func);

void
pgstrom_setup_shmem(Size zone_length,
					bool (*callback)(void *address, Size length,
									 const char *label,
									 bool abort_on_error))
{
	shmem_zone	   *zone;
	long			zone_index;
	long			num_zones;
	long			num_blocks;
	Size			offset;
	bool			hostmem_mapped;

	pg_memory_barrier();
	if (pgstrom_shmem_head->is_ready)
	{
		elog(LOG, "shared memory segment is already set up");
		return;
	}

	/* NOTE: Host unified memory device tends to have much larger zone-
	 * length than discrete devices. It may mislead the query planner,
	 * and cause unexpected large memory requirement.
	 */
	zone_length = Min(pgstrom_shmem_totalsize, zone_length);

	zone_length = TYPEALIGN_DOWN(SHMEM_BLOCKSZ, zone_length);
	num_zones = (pgstrom_shmem_totalsize + zone_length - 1) / zone_length;
	pgstrom_shmem_head->zone_baseaddr
		= (void *)TYPEALIGN(SHMEM_BLOCKSZ,
							&pgstrom_shmem_head->zones[num_zones]);
	pgstrom_shmem_head->zone_length = zone_length;

	/* NOTE: If run-time support host mapped memory which is larger than
	 * zone-length, we map the host memory at once.
	 * If unavailable (rc == CL_INVALID_BUFFER_SIZE), it tries to map
	 * host memory per zone basis.
	 */
	hostmem_mapped = callback(pgstrom_shmem_head->zone_baseaddr,
							  pgstrom_shmem_totalsize, "shmem", false);
	offset = 0;
	for (zone_index = 0; zone_index < num_zones; zone_index++)
	{
		Size	length;
		long	blkno;
		int		shift;
		int		i;

		length = Min(zone_length, pgstrom_shmem_totalsize - offset);
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
		/*
		 * Zero clear. Non-head block has all zero field, unless it becomes
		 * a head of continuous blocks
		 */
		memset(zone->blocks, 0, sizeof(shmem_block) * num_blocks);
		zone->block_baseaddr
			= (void *)TYPEALIGN(SHMEM_BLOCKSZ, &zone->blocks[num_blocks]);
		Assert((uintptr_t)zone + length ==
			   (uintptr_t)zone->block_baseaddr + SHMEM_BLOCKSZ * num_blocks);

		blkno = 0;
		shift = SHMEM_BLOCKSZ_BITS_RANGE;
		while (blkno < zone->num_blocks)
		{
			int		nblocks = (1 << shift);

			if (blkno + nblocks <= zone->num_blocks)
			{
				zone->blocks[blkno].blocksz = SHMEM_BLOCKSZ * nblocks;
				dlist_push_tail(&zone->free_list[shift],
								&zone->blocks[blkno].chain);
				zone->num_free[shift]++;
				blkno += nblocks;
			}
			else if (shift > 0)
				shift--;
		}
		/* host memory mapping per zone basis, if needed */
		if (!hostmem_mapped)
			callback(zone->block_baseaddr,
					 zone->num_blocks * SHMEM_BLOCKSZ,
					 "shmem", true);
		/* put zone on the pgstrom_shmem_head */
		pgstrom_shmem_head->zones[zone_index] = zone;
		offset += length;
	}
	Assert(zone_index <= num_zones);
	pgstrom_shmem_head->num_zones = zone_index;

	/* OK, now ready to use shared memory segment */
	pgstrom_shmem_head->is_ready = true;
}

static void
pgstrom_startup_shmem(void)
{
	Size	length;
	bool	found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	length = MAXALIGN(offsetof(shmem_head, zones[pgstrom_shmem_maxzones])) +
		pgstrom_shmem_totalsize + SHMEM_BLOCKSZ;

	pgstrom_shmem_head = ShmemInitStruct("pgstrom_shmem_head",
										 length, &found);
	Assert(!found);

	memset(pgstrom_shmem_head, 0, length);
	/* initialize fields for slabs */
	pgstrom_init_slab();
}

void
pgstrom_init_shmem(void)
{
	static int	shmem_totalsize;
	Size		length;

	/*
	 * Definition of GUC variables for shared memory management
	 */
	DefineCustomIntVariable("pg_strom.shmem_totalsize",
							"total size of shared memory segment in MB",
							NULL,
							&shmem_totalsize,
							2048,	/* 2GB */
							128,	/* 128MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	DefineCustomIntVariable("pg_strom.shmem_maxzones",
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
