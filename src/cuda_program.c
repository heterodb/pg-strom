/*
 * cuda_program.c
 *
 * Routines for just-in-time comple cuda code
 * ----
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "pg_strom.h"

typedef struct
{
	dlist_node	chain;
	int			shift;	/* memory block class of this entry */
	pg_crc32	crc;	/* extra_flags + cuda_source */
	int			extra_flags;
	int			refcnt;
	char	   *cuda_source;
	Size		cuda_source_len;
	char	   *cuda_binary;
	Size		cuda_binary;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} program_cache_entry;

#define PGCACHE_ACTIVE_ENTRY(entry)		((entry)->refcnt == 0)
#define PGCACHE_FREE_ENTRY(entry)		((entry)->refcnt > 0)
#define PGCACHE_MAGIC					0xabadcafe

#define PGCACHE_MIN_BITS		10		/* 1KB */
#define PGCACHE_MAX_BITS		24		/* 16MB */	
#define PGCACHE_HASH_SIZE		1024
typedef struct
{
	LWLock		mutex;
	long		num_actives[PGCACHE_MAX_BITS];
	long		num_free[PGCACHE_MAX_BITS];
	dlist_head	free_list[PGCACHE_MAX_BITS];
	dlist_head	active_list[PGCACHE_HASH_SIZE];
	program_cache_entry *entry_begin;	/* start address of entries */
	program_cache_entry *entry_end;		/* end address of entries */
	Bitmapset	waiting_backends;	/* flexible length */
} program_cache_head;

static shmem_startup_hook_type shmem_startup_next;
static Size		program_cache_size;
static volatile program_cache_head *pgcache_head;

/*
 *
 * XXX - caller must have lock of supplied zone
 */
static bool
pgstrom_program_cache_split(int shift)
{
	shmem_block	   *block;
	dlist_node	   *dnode;
	int				index;
	int				i;

	Assert(shift > PGCACHE_MIN_BITS && shift <= PGCACHE_MAX_BITS);
	if (dlist_is_empty(&zone->free_list[shift]))
	{
		if (shift == SHMEM_BLOCKSZ_BITS_RANGE ||
			!pgstrom_program_cache_split(shift + 1))
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

static pgcache_entry *
pgstrom_program_cache_alloc(Size required)
{
	program_cache_entry *entry;
	dlist_node	*dnode;
	Size	total_size;
	int		shift;
	int		index;
	void   *address;
	int		i;

	total_size = offsetof(program_cache_entry,
						  data[required + sizeof(cl_uint)]);
	if (total_size > (1UL << PGCACHE_MAX_BITS))
		return NULL;	/* too large size required */

	shift = get_next_log2(total_size);
	if (dlist_is_empty(&pgcache_head->free_list[shift]))
	{
		if (!pgstrom_shmem_zone_block_split(shift + 1))
			return NULL;
	}
	Assert(!dlist_is_empty(&pgcache_head->free_list[shift]));

	dnode = dlist_pop_head_node(&pgcache_head-->free_list[shift]);
	entry = dlist_container(program_cache_entry, chain, dnode);
	Assert(entry->shift == shift);

	memset(entry, 0, sizeof(program_cache_entry));
	entry->shift = shift;
	entry->refcnt = 1;
	*((cl_uint *)(entry->data + required)) = PGCACHE_MAGIC;

	/* update statictics */
	pgcache_head->num_free[shift]--;
	pgcache_head->num_actives[shift]++;

	return entry;
}

static void
pgstrom_program_cache_free(program_cache_entry *entry)
{
	program_cache_entry *buddy;
	int			shift = entry->shift;
	Size		offset;

	Assert(entry->refcnt == 0);

	offset = (uintptr_t)entry - (uintptr_t)pgcache_head->entry_begin;
	pgcache_head->num_active[shift]--;

	/* try to merge buddy entry, if it is also free */
	while (shift < PGCACHE_MAX_BITS)
	{
		program_cache_entry *buddy;

		offset = (uintptr_t) entry - (uintptr_t)pgcache_head->entry_begin;
		if ((offset & (1UL << shift)) == 0)
			buddy = (program_cache_entry *)((char *)entry + (1UL << shift));
		else
			buddy = (program_cache_entry *)((char *)entry - (1UL << shift));

		if (buddy >= pgcache_head->entry_end ||	/* out of range? */
			buddy->shift != shift ||			/* same size? */
			PGCACHE_ACTIVE_ENTRY(buddy))		/* and free entry? */
			break;
		/* OK, chunk and buddy can be merged */
		zone->num_free[shift]--;

		dlist_delete(&buddy->chain);
		if (buddy < chunk)
			chunk = buddy;
		chunk->shift = ++shift;
	}
	pgcache_head->num_free[shift]++;
	dlist_push_head(&pgcache_head->free_list[shift], &chunk->chain);
}












static void
pgstrom_startup_cuda_program(void)
{
	int			total_procs;
	int			i, nwords;
	int			shift;
	char	   *curr_addr;
	char	   *end_addr;
	Bitmapset  *tempset;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	pgcache_head = ShmemInitStruct("PG-Strom program cache",
								   program_cache_size, &found);
	if (found)
		elog(ERROR, "Bug? shared memory for program cache already exists");

	/* initialize program cache header */
	memset(pgcache_head, 0, sizeof(program_cache_head));
	LWLockInitialize(&pgcache_head->mutex);
	for (i=0; i < PGCACHE_MAX_BITS; i++)
		dlist_init(&pgcache_head->free_list[i]);
	for (i=0; i < PGCACHE_HASH_SIZE; i++)
		dlist_init(&pgcache_head->active_list[i]);

	total_procs = MaxBackends + NUM_AUXILIARY_PROCS + max_prepared_xacts;
	nwords = (total_procs + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	tempset = &pgcache_head->waiting_backends;
	tempset->nwords = nwords;
	memset(tempset->words, 0, sizeof(bitmapword) * nwords);
	pgcache_head->entry_begin =
		(pgcache_entry *) BUFFERALIGN(tempset->words + nwords);

	/* makes free entries */
	curr_addr = (char *) pgcache_head->entry_begin;
	end_addr = ((char *) pgcache_head) + program_cache_size;
	shift = PGCACHE_MAX_BITS;
	while (shift >= PGCACHE_MIN_BITS)
	{
		if (curr_addr + (1UL << shift) > end_addr)
		{
			shift--;
			continue;
		}
		entry = (pgcache_entry *) curr_addr;
		memset(entry, 0, sizeof(program_cache_entry));
		dlist_push_tail(&pgcache_head->free_list[shift], &entry->chain);

		curr_addr += (1UL << shift);
	}
	pgcache_head->entry_end = curr_addr;
}

void
pgstrom_init_cuda_program(void)
{
	static int	__program_cache_size;

	DefineCustomIntVariable("pg_strom.program_cache_size",
							"size of shared program cache",
							NULL,
							&__program_cache_size,
							48 * 1024,		/* 48MB */
							16 * 1024,		/* 16MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	program_cache_size = (Size)__program_cache_size * 1024L;

	/* allocation of static shared memory */
	RequestAddinShmemSpace(program_cache_size);
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_program;
}
