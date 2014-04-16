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

typedef struct
{
	/* for context management */
	slock_t		context_lock;
	dlist_head	context_list;

	/* for zone management */
	bool		is_ready;
	int			num_zones;
	void	   *zone_baseaddr;
	Size		zone_length;
	shmem_zone *zones[FLEXIBLE_ARRAY_MEMBER];
} shmem_head;

#define SHMEM_BLOCK_MAGIC		0xdeadbeaf

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
pgstrom_shmem_zone_block_alloc(shmem_zone *zone, Size size)
{
	shmem_block	*block;
	dlist_node	*dnode;
	int		shift = find_least_pot(size + sizeof(cl_uint));
	int		index;
	void   *address;
	int		i;

	if (dlist_is_empty(&zone->free_list[shift]))
	{
		if (!pgstrom_shmem_zone_block_split(zone, shift+1))
			return NULL;
	}
	Assert(!dlist_is_empty(&zone->free_list[shift]));

	dnode = dlist_pop_head_node(&zone->free_list[shift]);
	block = dlist_container(shmem_block, chain, dnode);
	Assert(block->blocksz == (1UL << shift) * SHMEM_BLOCKSZ);

	memset(&block->chain, 0, sizeof(dlist_node));
	block->blocksz = size;

	/* non-head block are zero cleared? */
	for (i=1; i < (1 << shift); i++)
		Assert(!BLOCK_IS_ACTIVE(block+i) && !BLOCK_IS_FREE(block+i));

	index = block - &zone->blocks[0];
	address = (void *)((Size)zone->block_baseaddr + index * SHMEM_BLOCKSZ);
	zone->num_free[shift]--;
	zone->num_active[shift]++;

	/* to detect overrun */
	*((cl_uint *)((uintptr_t)address + size)) = SHMEM_BLOCK_MAGIC;

	return address;
}

static void
pgstrom_shmem_zone_block_free(shmem_zone *zone, void *address)
{
	shmem_block	   *block;
	long			index;
	long			shift;

	Assert(ADDRESS_IN_SHMEM_ZONE(zone, address));

	index = ((Size)address - (Size)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = &zone->blocks[index];
	Assert(BLOCK_IS_ACTIVE(block));
	/* detect overrun */
	Assert(*((cl_uint *)((uintptr_t)address +
						 block->blocksz)) == SHMEM_BLOCK_MAGIC);
	shift = find_least_pot(block->blocksz + sizeof(cl_uint));
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
 * pgstrom_shmem_block_alloc
 *
 * It is an internal API; that allocates a continuous 2^N blocks from
 * a particular shared memory zone. It tries to split a larger memory blocks
 * if suitable memory blocks are not free. If no memory blocks are available,
 * it goes into another zone to allocate memory.
 */
void *
pgstrom_shmem_alloc(Size size)
{
	static int	zone_index = 0;
	int			start;
	shmem_zone *zone;
	void	   *address;

	/* does shared memory segment already set up? */
	if (!pgstrom_shmem_head->is_ready)
	{
		elog(LOG, "PG-Strom's shared memory segment has not been ready");
		return NULL;
	}


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

void
pgstrom_shmem_free(void *address)
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
	void		   *zone_baseaddr = pgstrom_shmem_head->zone_baseaddr;
	Size			zone_length = pgstrom_shmem_head->zone_length;
	int				zone_index;
	int				block_index;
	cl_uint		   *p_magic;

	Assert((Size)address % SHMEM_BLOCKSZ == 0);

	zone_index = ((Size)address - (Size)zone_baseaddr) / zone_length;
	Assert(zone_index >= 0 && zone_index < pgstrom_shmem_head->num_zones);

	zone = pgstrom_shmem_head->zones[zone_index];
	Assert(ADDRESS_IN_SHMEM_ZONE(zone, address));

	block_index = ((Size)address -
				   (Size)zone->block_baseaddr) / SHMEM_BLOCKSZ;
	block = &zone->blocks[block_index];
	Assert(BLOCK_IS_ACTIVE(block));

	p_magic = (cl_uint *)((uintptr_t)address + block->blocksz);

	return (*p_magic == SHMEM_BLOCK_MAGIC ? true : false);
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
			int		nshift = find_least_pot(block->blocksz + sizeof(cl_uint));

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
	//Assert(memcmp(num_active, zone->num_active, sizeof(num_active)) == 0);
	//Assert(memcmp(num_free, zone->num_free, sizeof(num_free)) == 0);
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

void
pgstrom_setup_shmem(Size zone_length,
					void *(*callback)(void *address, Size length))
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

		/*
		 * If remaining area is too small than expected length, we skip to
		 * set up to avoid unexpected troubles.
		 */
		if (length <= zone_length / 8)
			break;

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
		/* per zone initialization */
		(*callback)(zone->block_baseaddr,
					zone->num_blocks * SHMEM_BLOCKSZ);
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
