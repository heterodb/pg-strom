/*
 * cuda_control.c
 *
 * Overall logic to control cuda context and devices.
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
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "lib/ilist.h"
#include "port/atomics.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "storage/procsignal.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include "utils/resowner.h"
#include <math.h>
#include "pg_strom.h"

/* available devices set by postmaster startup */
static List		   *cuda_device_ordinals = NIL;
static List		   *cuda_device_capabilities = NIL;
static List		   *cuda_device_mem_sizes = NIL;	/* in MB */
static size_t		cuda_max_malloc_size = INT_MAX;
static size_t		cuda_max_threads_per_block = INT_MAX;
static size_t		cuda_local_mem_size = INT_MAX;
static int			cuda_compute_capability = INT_MAX;

/* stuffs related to GpuContext */
static dlist_head	gcontext_list;

/* CUDA runtime stuff per backend process */
static int			cuda_num_devices = -1;
static CUdevice	   *cuda_devices = NULL;
static CUcontext   *cuda_last_contexts = NULL;	/* last used sanity context */

/* misc static variables */
static shmem_startup_hook_type shmem_startup_next;

/* ----------------------------------------------------------------
 *
 * Routines to share the status of device resource consumption
 *
 * ----------------------------------------------------------------
 */
typedef struct {
	cl_uint				num_devices;	/* never updated */
	pg_atomic_uint32	num_gcontext;	/* total number of GpuContext */
	struct {
		cl_ulong			gmem_size;	/* never updated */
		pg_atomic_uint64	gmem_used;	/* total amount of DRAM usage */
	} gpu[FLEXIBLE_ARRAY_MEMBER];
} GpuScoreBoard;

static GpuScoreBoard	   *gpuScoreBoard;

#define GpuScoreCurrNumContext()				\
	pg_atomic_read_u32(&gpuScoreBoard->num_gcontext)
#define GpuScoreCurrMemUsage(cuda_index)		\
	pg_atomic_read_u64(&gpuScoreBoard->gpu[(cuda_index)].gmem_used)
#define GpuScoreInclMemUsage(gcontext,cuda_index,size)		\
	do {													\
		pg_atomic_fetch_add_u64(&gpuScoreBoard->gpu[(cuda_index)].gmem_used, \
								(size));					\
		gcontext->gpu[(cuda_index)].gmem_used += (size);	\
	} while(0)
#define GpuScoreDeclMemUsage(gcontext,cuda_index,size)		\
	do {													\
		pg_atomic_fetch_sub_u64(&gpuScoreBoard->gpu[(cuda_index)].gmem_used, \
								(size));					\
		(gcontext)->gpu[(cuda_index)].gmem_used -= (size);	\
	} while(0)

/* ----------------------------------------------------------------
 *
 * Routines to support lightwight userspace device memory allocator
 *
 * ----------------------------------------------------------------
 */
typedef struct GpuMemBlock
{
	dlist_node		chain;			/* link to active/unused_blocks */
	CUdeviceptr		block_addr;		/* head of device address */
	size_t			block_size;		/* length of the block */
	size_t			max_free_size;	/* max available space in this block */
	dlist_head		addr_chunks;	/* chunks in order of address */
	dlist_head		free_chunks;	/* free chunks */
} GpuMemBlock;

typedef struct GpuMemChunk
{
	GpuMemBlock	   *gm_block;	/* memory block this chunk belong to */
	dlist_node		addr_chain;	/* link to addr_chunks */
	dlist_node		free_chain;	/* link to free_chunks, or zero if active */
	dlist_node		hash_chain;	/* link to hash_table, or zero if free  */
	CUdeviceptr		chunk_addr;
	size_t			chunk_size;
} GpuMemChunk;

static inline void
gpuMemHeadInit(GpuMemHead *gm_head)
{
	int		i;

	gm_head->empty_block = NULL;
	dlist_init(&gm_head->active_blocks);
	dlist_init(&gm_head->unused_chunks);
	dlist_init(&gm_head->unused_blocks);
	for (i=0; i < lengthof(gm_head->hash_slots); i++)
		dlist_init(&gm_head->hash_slots[i]);
}

static inline int
gpuMemHashIndex(GpuMemHead *gm_head, CUdeviceptr chunk_addr)
{
	pg_crc32    crc;

	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &chunk_addr, sizeof(CUdeviceptr));
	FIN_LEGACY_CRC32(crc);

	return crc % lengthof(gm_head->hash_slots);
}

/*
 * gpuMemMaxAllocSize
 *
 * it return the max size of device memory allocation for all the
 * installed deviced.
 */
Size
gpuMemMaxAllocSize(void)
{
	return cuda_max_malloc_size;
}

/*
 * gpuMemDump
 *
 * For debug, it dumps all the device memory chunks
 */
static void
__gpuMemDump(GpuMemBlock *gm_block, bool is_active)
{
	GpuMemChunk	   *gm_chunk;
	dlist_iter		iter;

	elog(INFO, "GpuMemBlock: %p - %p (size: %zu, free: %zu, %s)",
		 (char *)(gm_block->block_addr),
		 (char *)(gm_block->block_addr + gm_block->block_size),
		 gm_block->block_size,
		 gm_block->max_free_size,
		 is_active ? "active" : "unused");

	dlist_foreach (iter, &gm_block->addr_chunks)
	{
		gm_chunk = dlist_container(GpuMemChunk, addr_chain, iter.cur);

		elog(INFO, "GpuMemChunk: %p - %p (offset: %08zx size: %08zx, %s)",
			 (char *)(gm_chunk->chunk_addr),
			 (char *)(gm_chunk->chunk_addr + gm_chunk->chunk_size),
			 (char *)gm_chunk->chunk_addr - (char *)gm_block->block_addr,
			 gm_chunk->chunk_size,
			 (!gm_chunk->free_chain.prev &&
			  !gm_chunk->free_chain.next) ? "active" : "free");
	}
}

static void
gpuMemDump(GpuContext *gcontext, int cuda_index)
{
	GpuMemHead	   *gm_head = &gcontext->gpu[cuda_index].cuda_memory;
	GpuMemBlock	   *gm_block;
	dlist_iter		iter;

	dlist_foreach (iter, &gm_head->active_blocks)
	{
		gm_block = dlist_container(GpuMemBlock, chain, iter.cur);
		__gpuMemDump(gm_block, true);
	}
	if (gm_head->empty_block)
		__gpuMemDump(gm_head->empty_block, false);
}

#ifdef NOT_USED
static void
__gpuMemSanityCheck(GpuMemHead *gm_head, GpuMemBlock *gm_block)
{
	GpuMemChunk	   *gm_chunk;
	GpuMemChunk	   *gm_temp;
	dlist_iter		iter;
	dlist_iter		hiter;
	CUdeviceptr		curr_addr = gm_block->block_addr;

	dlist_foreach (iter, &gm_block->addr_chunks)
	{
		gm_chunk = dlist_container(GpuMemChunk, addr_chain, iter.cur);
		Assert(gm_chunk->chunk_addr == curr_addr);
		curr_addr += gm_chunk->chunk_size;
		Assert(curr_addr <= gm_block->block_addr + gm_block->block_size);

		if (gm_chunk->hash_chain.prev && gm_chunk->hash_chain.next)
		{
			int		index = gpuMemHashIndex(gm_head, gm_chunk->chunk_addr);
			bool	found = false;

			Assert(!gm_chunk->free_chain.prev && !gm_chunk->free_chain.next);
			dlist_foreach (hiter, &gm_head->hash_slots[index])
			{
				gm_temp = dlist_container(GpuMemChunk, hash_chain, hiter.cur);
				if (gm_temp->chunk_addr == gm_chunk->chunk_addr)
				{
					found = true;
					break;
				}
			}
			Assert(found);
			elog(INFO, "sanity %zx - %zx (active)",
				 (Size)(gm_chunk->chunk_addr),
				 (Size)(gm_chunk->chunk_addr + gm_chunk->chunk_size));
		}
		else if (gm_chunk->free_chain.prev && gm_chunk->free_chain.next)
		{
			bool	found = false;

			Assert(!gm_chunk->hash_chain.prev && !gm_chunk->hash_chain.next);
			dlist_foreach (hiter, &gm_block->free_chunks)
			{
				gm_temp = dlist_container(GpuMemChunk, free_chain, hiter.cur);
				if (gm_temp->chunk_addr == gm_chunk->chunk_addr)
				{
					found = true;
					break;
				}
			}
			Assert(found);
			elog(INFO, "sanity %zx - %zx (free)",
				 (Size)(gm_chunk->chunk_addr),
				 (Size)(gm_chunk->chunk_addr + gm_chunk->chunk_size));
		}
		else
			Assert(false);	/* neither active nor free */
	}
	Assert(curr_addr == gm_block->block_addr + gm_block->block_size);
}

static void
gpuMemSanityCheck(GpuContext *gcontext, int cuda_index)
{
	GpuMemHead	   *gm_head = &gcontext->gpu[cuda_index].cuda_memory;
	GpuMemBlock	   *gm_block;
	dlist_iter		iter;

	dlist_foreach (iter, &gm_head->active_blocks)
	{
		gm_block = dlist_container(GpuMemBlock, chain, iter.cur);
		__gpuMemSanityCheck(gm_head, gm_block);
	}
	if (gm_head->empty_block)
		__gpuMemSanityCheck(gm_head, gm_head->empty_block);
	elog(INFO, "looks to sanity...");
}
#endif

CUdeviceptr
__gpuMemAlloc(GpuContext *gcontext, int cuda_index, size_t bytesize)
{
	GpuMemHead	   *gm_head;
	GpuMemBlock	   *gm_block;
	GpuMemChunk	   *gm_chunk;
	GpuMemChunk	   *new_chunk;
	dlist_node	   *dnode;
	dlist_iter		iter;
	CUdeviceptr		block_addr;
	CUresult		rc;
	int				index;
	uint32			curr_numcxt;
	size_t			curr_limit;
	size_t			required;

	/* round up to 1KB align */
	bytesize = TYPEALIGN(1024, bytesize);

	/* is it reasonable size to allocate? */
	if (bytesize > cuda_max_malloc_size)
		elog(ERROR, "too large device memory request %zu bytes, max %zu",
			 bytesize, cuda_max_malloc_size);

	/* try to find out preliminary allocated block */
	Assert(cuda_index < gcontext->num_context);
	gm_head = &gcontext->gpu[cuda_index].cuda_memory;

	dlist_foreach(iter, &gm_head->active_blocks)
	{
		gm_block = dlist_container(GpuMemBlock, chain, iter.cur);
		if (gm_block->max_free_size > bytesize)
			goto found;
	}

	if (gm_head->empty_block)
	{
		gm_block = gm_head->empty_block;
		if (gm_block->max_free_size > bytesize)
		{
			Assert(!gm_block->chain.prev && !gm_block->chain.next);
			gm_head->empty_block = NULL;
			dlist_push_head(&gm_head->active_blocks, &gm_block->chain);
			goto found;
		}
	}
	/*
	 * no space available on the preliminary allocated block,
	 * so we try to allocate device memory in advance.
	 *
	 * NOTE: we should give more practical estimation for
	 * device memory requirement. smaller number of kernel
	 * driver call makes better performance!
	 */
	required = Max(pgstrom_chunk_size() * 11, bytesize);
	required = TYPEALIGN(1024 * 1024, required);	/* round up to 1MB */
#ifdef USE_ASSERT_CHECKING
	{
		/*
		 * We expect caller already set appropriate cuda_context
		 * on the current thread. Ensure the context here.
		 */
		CUcontext	curr_context;

		rc = cuCtxGetCurrent(&curr_context);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxGetCurrent: %s", errorText(rc));
		Assert(curr_context == gcontext->gpu[cuda_index].cuda_context);
	}
#endif
	/*
	 * NOTE: GPU device memory allocation limit.
	 * We assume GPU has relatively small amount of RAM compared to the
	 * host system, so large concurrent job easily makes memory starvation.
	 * So, we put a resource limitation here to avoid overconsumption.
	 * Individual backends cannot allocate device memory over the
	 *   (total size of device memory) / sqrt(1 + (num of GpuContext))
	 * constraint.
	 */
	curr_numcxt = GpuScoreCurrNumContext();
	curr_limit = ((double)gpuScoreBoard->gpu[cuda_index].gmem_size /
				  sqrt((double)(curr_numcxt + 1)));
	if (gcontext->gpu[cuda_index].gmem_used >= curr_limit)
	{
		/*
		 * Even if GPU memory usage is larger than threshold, we may be
		 * able to allocate amount of actually allocated is smaller than
		 * expectation, we allow GPU memory allocation.
		 */

		/* TODO: rule to allow device memory */
		/* GpuScoreCurrMemUsage(cuda_index) */
		elog(DEBUG1, "gpuMemAlloc failed due to resource limitation");

		return 0UL;		/* need to wait... */
	}

	/*
	 * TODO: too frequent device memory allocation request will lock
	 * down the system. We may need to have cooling-down time here.
	 */
	rc = cuMemAlloc(&block_addr, required);
	if (rc != CUDA_SUCCESS)
	{
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		{
			elog(DEBUG1, "cuMemAlloc failed due to CUDA_ERROR_OUT_OF_MEMORY");
			return 0UL;		/* need to wait... */
		}
		elog(ERROR, "failed on cuMemAlloc: %s", errorText(rc));
	}
	/* update scoreboard for resource control */
	GpuScoreInclMemUsage(gcontext, cuda_index, required);

	elog(DEBUG1, "cuMemAlloc(%08zx - %08zx, size=%zuMB)",
		 (size_t)(block_addr),
		 (size_t)(block_addr + required),
		 (size_t)required >> 20);

	if (dlist_is_empty(&gm_head->unused_blocks))
		gm_block = MemoryContextAlloc(gcontext->memcxt, sizeof(GpuMemBlock));
	else
	{
		dnode = dlist_pop_head_node(&gm_head->unused_blocks);
		gm_block = dlist_container(GpuMemBlock, chain, dnode);
	}
	memset(gm_block, 0, sizeof(GpuMemBlock));
	gm_block->block_addr = block_addr;
	gm_block->block_size = required;
	gm_block->max_free_size = required;
	dlist_init(&gm_block->addr_chunks);
	dlist_init(&gm_block->free_chunks);

	if (dlist_is_empty(&gm_head->unused_chunks))
		gm_chunk = MemoryContextAlloc(gcontext->memcxt, sizeof(GpuMemChunk));
	else
	{
		dnode = dlist_pop_head_node(&gm_head->unused_chunks);
		gm_chunk = dlist_container(GpuMemChunk, addr_chain, dnode);
	}
	memset(gm_chunk, 0, sizeof(GpuMemChunk));
	gm_chunk->gm_block = gm_block;
	dlist_push_head(&gm_block->addr_chunks, &gm_chunk->addr_chain);
	dlist_push_head(&gm_block->free_chunks, &gm_chunk->free_chain);
	memset(&gm_chunk->hash_chain, 0, sizeof(dlist_node));
	gm_chunk->chunk_addr = block_addr;
	gm_chunk->chunk_size = required;

	dlist_push_head(&gm_head->active_blocks, &gm_block->chain);

found:
	dlist_foreach(iter, &gm_block->free_chunks)
	{
		gm_chunk = dlist_container(GpuMemChunk, free_chain, iter.cur);

		Assert(!gm_chunk->hash_chain.prev && !gm_chunk->hash_chain.next);
		if (gm_chunk->chunk_size < bytesize)
			continue;

		/* no need to split, just replace free chunk */
		if (gm_chunk->chunk_size == bytesize)
		{
			dlist_delete(&gm_chunk->free_chain);
			memset(&gm_chunk->free_chain, 0, sizeof(dlist_node));

			index = gpuMemHashIndex(gm_head, gm_chunk->chunk_addr);
			dlist_push_tail(&gm_head->hash_slots[index],
							&gm_chunk->hash_chain);
			if (gm_block->max_free_size == gm_chunk->chunk_size)
			{
				gm_block->max_free_size = 0;
				dlist_foreach(iter, &gm_block->free_chunks)
				{
					GpuMemChunk	   *temp = dlist_container(GpuMemChunk,
														   free_chain,
														   iter.cur);
					gm_block->max_free_size = Max(gm_block->max_free_size,
												  temp->chunk_size);
				}
			}
			return gm_chunk->chunk_addr;
		}

		/* larger free chunk found, so let's split it */
		if (dlist_is_empty(&gm_head->unused_chunks))
			new_chunk = MemoryContextAlloc(gcontext->memcxt,
										   sizeof(GpuMemChunk));
		else
		{
			dnode = dlist_pop_head_node(&gm_head->unused_chunks);
			new_chunk = dlist_container(GpuMemChunk, addr_chain, dnode);
		}
		memset(new_chunk, 0, sizeof(GpuMemChunk));
		new_chunk->gm_block = gm_block;
		new_chunk->chunk_addr = gm_chunk->chunk_addr + bytesize;
		new_chunk->chunk_size = gm_chunk->chunk_size - bytesize;
		gm_chunk->chunk_size = bytesize;

		/* add new one just after the old one */
		dlist_insert_after(&gm_chunk->addr_chain, &new_chunk->addr_chain);
		/* remove old one from the free list */
		dlist_delete(&gm_chunk->free_chain);
		memset(&gm_chunk->free_chain, 0, sizeof(dlist_node));
		/* add new one to the free list */
		dlist_push_tail(&gm_block->free_chunks, &new_chunk->free_chain);

		/* add active portion to the hash table */
		index = gpuMemHashIndex(gm_head, gm_chunk->chunk_addr);
		dlist_push_tail(&gm_head->hash_slots[index],
						&gm_chunk->hash_chain);
		if (gm_block->max_free_size == new_chunk->chunk_size + bytesize)
		{
			gm_block->max_free_size = 0;
			dlist_foreach(iter, &gm_block->free_chunks)
			{
				GpuMemChunk    *temp = dlist_container(GpuMemChunk,
													   free_chain,
													   iter.cur);
				gm_block->max_free_size = Max(gm_block->max_free_size,
											  temp->chunk_size);
			}
		}
		return gm_chunk->chunk_addr;
	}
	gpuMemDump(gcontext, cuda_index);
	elog(ERROR, "Bug? we could not find a free chunk in GpuMemBlock (%zu)", bytesize);
}

CUdeviceptr
gpuMemAlloc(GpuTask *gtask, size_t bytesize)
{
	return __gpuMemAlloc(gtask->gts->gcontext, gtask->cuda_index, bytesize);
}

void
__gpuMemFree(GpuContext *gcontext, int cuda_index, CUdeviceptr chunk_addr)
{
	GpuMemHead	   *gm_head;
	GpuMemBlock	   *gm_block;
	GpuMemChunk	   *gm_chunk;
	GpuMemChunk	   *gm_prev;
	GpuMemChunk	   *gm_next;
	dlist_node	   *dnode;
	dlist_iter		iter;
	CUresult		rc;
	int				index;

	/* find out the cuda-context */
	Assert(cuda_index < gcontext->num_context);
	gm_head = &gcontext->gpu[cuda_index].cuda_memory;

	index = gpuMemHashIndex(gm_head, chunk_addr);
	dlist_foreach(iter, &gm_head->hash_slots[index])
	{
		gm_chunk = dlist_container(GpuMemChunk, hash_chain, iter.cur);
		if (gm_chunk->chunk_addr == chunk_addr)
			goto found;
	}
	elog(WARNING, "Bug? device address %p was not tracked",
		 (void *)chunk_addr);
	return;

found:
	/* unlink from the hash */
	dlist_delete(&gm_chunk->hash_chain);
	memset(&gm_chunk->hash_chain, 0, sizeof(dlist_node));

	/* sanity check; chunks should be within block */
	gm_block = gm_chunk->gm_block;
	Assert(gm_chunk->chunk_addr >= gm_block->block_addr &&
		   (gm_chunk->chunk_addr + gm_chunk->chunk_size) <=
		   (gm_block->block_addr + gm_block->block_size));
	gm_block->max_free_size = Max(gm_block->max_free_size,
								  gm_chunk->chunk_size);

	/* back to the free_list */
	Assert(!gm_chunk->free_chain.prev && !gm_chunk->free_chain.next);
	dlist_push_head(&gm_block->free_chunks, &gm_chunk->free_chain);

	if (dlist_has_prev(&gm_block->addr_chunks,
					   &gm_chunk->addr_chain))
	{
		dnode = dlist_prev_node(&gm_block->addr_chunks,
								&gm_chunk->addr_chain);
		gm_prev = dlist_container(GpuMemChunk, addr_chain, dnode);
		Assert(gm_prev->chunk_addr +
			   gm_prev->chunk_size == gm_chunk->chunk_addr);
		if (gm_prev->free_chain.prev && gm_prev->free_chain.next)
		{
			Assert(!gm_prev->hash_chain.prev &&
				   !gm_prev->hash_chain.next);
			/* OK, it can be merged */
			dlist_delete(&gm_chunk->addr_chain);
			dlist_delete(&gm_chunk->free_chain);
			gm_prev->chunk_size += gm_chunk->chunk_size;

			/* update max_free_size if needed */
			gm_block->max_free_size = Max(gm_block->max_free_size,
										  gm_prev->chunk_size);

			/* GpuMemChunk entry may be reused soon */
			memset(gm_chunk, 0, sizeof(GpuMemChunk));
			dlist_push_head(&gm_head->unused_chunks, &gm_chunk->addr_chain);

			gm_chunk = gm_prev;
		}
		else
			Assert(!gm_prev->free_chain.prev && !gm_prev->free_chain.next);
	}

	if (dlist_has_next(&gm_block->addr_chunks,
					   &gm_chunk->addr_chain))
	{
		dnode = dlist_next_node(&gm_block->addr_chunks,
								&gm_chunk->addr_chain);
		gm_next = dlist_container(GpuMemChunk, addr_chain, dnode);
		Assert(gm_chunk->chunk_addr +
			   gm_chunk->chunk_size == gm_next->chunk_addr);
		if (gm_next->free_chain.prev && gm_next->free_chain.next)
		{
			Assert(!gm_next->hash_chain.prev &&
				   !gm_next->hash_chain.next);
			/* OK, it can be merged */
			dlist_delete(&gm_next->addr_chain);
			dlist_delete(&gm_next->free_chain);
			gm_chunk->chunk_size += gm_next->chunk_size;

			/* update max_free_size if needed */
			gm_block->max_free_size = Max(gm_block->max_free_size,
										  gm_chunk->chunk_size);

			/* GpuMemChunk entry may be reused soon */
			memset(gm_next, 0, sizeof(GpuMemChunk));
			dlist_push_head(&gm_head->unused_chunks, &gm_next->addr_chain);
		}
		else
			Assert(!gm_next->free_chain.prev && !gm_next->free_chain.next);
	}

	/*
	 * Try to check GpuMemBlock is still active or not
	 */
	if (!dlist_has_prev(&gm_block->addr_chunks, &gm_chunk->addr_chain) &&
		!dlist_has_next(&gm_block->addr_chunks, &gm_chunk->addr_chain))
	{
		Assert(gm_chunk->free_chain.prev && gm_chunk->free_chain.next);
		Assert(!gm_chunk->hash_chain.prev && !gm_chunk->hash_chain.next);
		Assert(gm_block->block_addr == gm_chunk->chunk_addr &&
			   gm_block->block_size == gm_chunk->chunk_size);
		Assert(gm_block->max_free_size == gm_block->block_size);
		/* OK, it looks to up an empty block */
		dlist_delete(&gm_block->chain);
		memset(&gm_block->chain, 0, sizeof(dlist_node));

		/* One empty block shall be kept, but no more */
		if (!gm_head->empty_block)
			gm_head->empty_block = gm_block;
		else
		{
			rc = cuCtxPushCurrent(gcontext->gpu[cuda_index].cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

			rc = cuMemFree(gm_block->block_addr);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemFree: %s", errorText(rc));

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

			/* update scoreboard for resource control */
			GpuScoreDeclMemUsage(gcontext, cuda_index, gm_block->block_size);

			elog(DEBUG1, "cuMemFree(%08zx - %08zx, size=%zuMB)",
				 (size_t)gm_block->block_addr,
				 ((size_t)gm_block->block_addr + gm_block->block_size),
				 ((size_t)gm_block->block_size) >> 20);

			memset(gm_chunk, 0, sizeof(GpuMemChunk));
			dlist_push_head(&gm_head->unused_chunks, &gm_chunk->addr_chain);
			memset(gm_block, 0, sizeof(GpuMemBlock));
			dlist_push_head(&gm_head->unused_blocks, &gm_block->chain);
		}
	}
}

void
gpuMemFree(GpuTask *gtask, CUdeviceptr chunk_addr)
{
	__gpuMemFree(gtask->gts->gcontext, gtask->cuda_index, chunk_addr);
}

/*
 * gpuMemFreeAll
 *
 * It releases all the device memory associated with the supplied
 * GpuContext.
 */
static void
gpuMemFreeAll(GpuContext *gcontext)
{
	GpuMemHead	   *gm_head;
	GpuMemBlock	   *gm_block;
	dlist_node	   *dnode;
	int				index;
	CUresult		rc;

	for (index=0; index < cuda_num_devices; index++)
	{
		rc = cuCtxPushCurrent(gcontext->gpu[index].cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPushCurrent: %s", errorText(rc));

		gm_head = &gcontext->gpu[index].cuda_memory;

		if (gm_head->empty_block)
		{
			gm_block = gm_head->empty_block;
			rc = cuMemFree(gm_block->block_addr);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemFree: %s", errorText(rc));
			GpuScoreDeclMemUsage(gcontext, index, gm_block->block_size);
			gm_head->empty_block = NULL;
		}

		while (!dlist_is_empty(&gm_head->active_blocks))
		{
			dnode = dlist_pop_head_node(&gm_head->active_blocks);
			gm_block = dlist_container(GpuMemBlock, chain, dnode);

			rc = cuMemFree(gm_block->block_addr);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemFree: %s", errorText(rc));
			GpuScoreDeclMemUsage(gcontext, index, gm_block->block_size);
		}

		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
	}
}

/*
 * pgstrom_cuda_init
 *
 * initialize CUDA runtime per backend process.
 */
static void
pgstrom_init_cuda(void)
{
	CUdevice	device;
	CUresult	rc;
	ListCell   *cell;
	int			i = 0;

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", errorText(rc));

	cuda_num_devices = list_length(cuda_device_ordinals);
	cuda_devices = MemoryContextAllocZero(TopMemoryContext,
										  sizeof(CUdevice) * cuda_num_devices);
	foreach (cell, cuda_device_ordinals)
	{
		int		ordinal = lfirst_int(cell);

		rc = cuDeviceGet(&device, ordinal);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", errorText(rc));
		cuda_devices[i++] = device;
	}
	cuda_last_contexts = MemoryContextAllocZero(TopMemoryContext,
												sizeof(CUcontext) *
												cuda_num_devices);
}

/*
 * pgstrom_cleanup_cuda
 *
 * It cleans up any active CUDA resources (Does it really needed? It looks
 * to me needed, at least. Hmm... proprietary module is mysterious), and
 * fixes-up contents of the score-board.
 */
static void
pgstrom_cleanup_cuda(int code, Datum arg)
{
	int		count = 0;
	bool	context_cached = false;

	/*
	 * Decrement usage of GPU memory usage by every active GpuContext
	 */
	while (!dlist_is_empty(&gcontext_list))
	{
		dlist_node	   *dnode = dlist_pop_head_node(&gcontext_list);
		GpuContext	   *gcontext = dlist_container(GpuContext, chain, dnode);

		gpuMemFreeAll(gcontext);
		count++;
	}

	/*
	 * Drop cached CUDA context
	 */
	if (cuda_last_contexts && cuda_last_contexts[0] != NULL)
	{
		int		i;

		for (i=0; i < cuda_num_devices; i++)
		{
			CUcontext	context = cuda_last_contexts[i];
			CUresult	rc;

			Assert(context != NULL);
			cuda_last_contexts[i] = NULL;

			rc = cuCtxDestroy(context);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
		}
		context_cached = true;
	}

	if (count > 0 || context_cached)
		elog(DEBUG1, "%s: %u active GpuContext%s were released",
			 __FUNCTION__, count,
			 context_cached ? " and cached CUDA context" : "");
}

static GpuContext *
pgstrom_create_gpucontext(ResourceOwner resowner, bool *context_reused)
{
	static bool		on_shmem_callback_registered = false;
	GpuContext	   *gcontext = NULL;
	MemoryContext	memcxt = NULL;
	CUcontext	   *cuda_context_temp;
	Size			length_gcxt;
	Size			length_init;
	Size			length_max;
	char			namebuf[200];
	int				index;
	CUresult		rc;

	if (cuda_num_devices < 0)
		pgstrom_init_cuda();
	if (cuda_num_devices < 1)
		elog(ERROR, "No cuda device were detected");
	if (!on_shmem_callback_registered)
	{
		on_shmem_exit(pgstrom_cleanup_cuda, 0);
		on_shmem_callback_registered = true;
	}

	/*
	 * Tries to find a cached CUDA context or create new one if not cached,
	 * then construct a GPU context based on the CUDA context.
	 */
	cuda_context_temp = palloc0(sizeof(CUcontext) * cuda_num_devices);
	PG_TRY();
	{
		if (cuda_last_contexts[0] != NULL)
		{
			memcpy(cuda_context_temp, cuda_last_contexts,
				   sizeof(CUcontext) * cuda_num_devices);
			memset(cuda_last_contexts, 0,
				   sizeof(CUcontext) * cuda_num_devices);
			*context_reused = true;
		}
		else
		{
			for (index=0; index < cuda_num_devices; index++)
			{
				rc = cuCtxCreate(&cuda_context_temp[index],
								 CU_CTX_SCHED_AUTO,
								 cuda_devices[index]);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuCtxCreate: %s",
						 errorText(rc));

				rc = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuCtxSetCacheConfig: %s",
						 errorText(rc));
			}
			*context_reused = false;
		}
		/* cuda_context_temp[] contains cuContext references here */
		/* make a new memory context on the primary cuda_context */
		snprintf(namebuf, sizeof(namebuf), "GPU DMA Buffer (%p)", resowner);
		length_init = 4 * (1UL << get_next_log2(pgstrom_chunk_size()));
		length_max = 1024 * length_init;

		memcxt = HostPinMemContextCreate(NULL,
										 namebuf,
										 cuda_context_temp[0],
										 length_init,
										 length_max);
		length_gcxt = offsetof(GpuContext, gpu[cuda_num_devices]);
		gcontext = MemoryContextAllocZero(memcxt, length_gcxt);
		gcontext->refcnt = 1;
		gcontext->resowner = resowner;
		gcontext->memcxt = memcxt;
		dlist_init(&gcontext->pds_list);
		for (index=0; index < cuda_num_devices; index++)
		{
			gcontext->gpu[index].cuda_context = cuda_context_temp[index];
			gcontext->gpu[index].cuda_device = cuda_devices[index];
			gpuMemHeadInit(&gcontext->gpu[index].cuda_memory);
		}
		gcontext->num_context = cuda_num_devices;
        gcontext->next_context = (MyProc->pgprocno % cuda_num_devices);

		/* Update the scoreboard of GPU usage */
		pg_atomic_fetch_add_u32(&gpuScoreBoard->num_gcontext, 1);
	}
	PG_CATCH();
	{
		if (memcxt != NULL)
			MemoryContextDelete(memcxt);

		for (index=0; index < cuda_num_devices; index++)
		{
			if (cuda_context_temp[index])
			{
				rc = cuCtxDestroy(cuda_context_temp[index]);
				if (rc != CUDA_SUCCESS)
                    elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
			}
		}
		PG_RE_THROW();
	}
	PG_END_TRY();

	return gcontext;
}

GpuContext *
pgstrom_get_gpucontext(void)
{
	GpuContext *gcontext;
	dlist_iter	iter;
	bool		context_reused;

	/* Does the current resource owner already have a GpuContext? */
	dlist_foreach (iter, &gcontext_list)
	{
		gcontext = dlist_container(GpuContext, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner)
		{
			dlist_move_head(&gcontext_list, &gcontext->chain);
			gcontext->refcnt++;
			elog(DEBUG2, "Get existing GpuContext (refcnt=%d, resowner=%p)",
				 gcontext->refcnt, gcontext->resowner);
			return gcontext;
		}
	}

	/*
	 * Hmm... no gpu context is not attached this resource owner,
	 * so create a new one.
	 */
	gcontext = pgstrom_create_gpucontext(CurrentResourceOwner,
										 &context_reused);
	dlist_push_head(&gcontext_list, &gcontext->chain);
	elog(DEBUG2, "Create new GpuContext%s (refcnt=%d, resowner=%p)",
		 context_reused ? " with CUDA context reused" : "",
		 gcontext->refcnt, gcontext->resowner);
	return gcontext;
}

static void
pgstrom_release_gpucontext(GpuContext *gcontext, bool sanity_release)
{
	CUcontext	cuda_context;
	CUresult	rc;
	int			i;

	/* detach this GpuContext from the global list */
	dlist_delete(&gcontext->chain);
	memset(&gcontext->chain, 0, sizeof(dlist_node));

	/*
	 * Synchronization of all the asynchronouse events, if any
	 */
	for (i=0; i < gcontext->num_context; i++)
	{
		rc = cuCtxSetCurrent(gcontext->gpu[i].cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxSetCurrent: %s", errorText(rc));

		rc = cuCtxSynchronize();
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxSynchronize: %s", errorText(rc));
	}
	/* Ensure CUDA context is empty */
	rc = cuCtxSetCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxSetCurrent(NULL): %s", errorText(rc));

	/*
	 * Release pgstrom_data_store; because KDS_FORMAT_ROW may have mmap(2)
	 * state in case of file-mapped data-store, so we have to ensure
	 * these temporary files are removed and unmapped.
	 */
	while (!dlist_is_empty(&gcontext->pds_list))
	{
		pgstrom_data_store	   *pds =
			dlist_container(pgstrom_data_store, pds_chain,
							dlist_head_node(&gcontext->pds_list));
		pgstrom_release_data_store(pds);

		/* No PDS should be orphan if sanity release */
		if (sanity_release)
			elog(DEBUG1, "Orphan PDS found, no cuContext shall be cached %p", pds);
		sanity_release = false;
	}

	/*
	 * Release all the GPU device memory
	 */
	gpuMemFreeAll(gcontext);

	for (i=0; i < gcontext->num_context; i++)
	{
		/* No device memory should be acquired if sanity release */
		if (sanity_release && gcontext->gpu[i].gmem_used > 0)
		{
			elog(DEBUG1, "Orphan GPU memory %zuKB on device %u, no cuContext shall be cached",
				 gcontext->gpu[i].gmem_used / 1024, i);
			sanity_release = false;
		}
		GpuScoreDeclMemUsage(gcontext, i, gcontext->gpu[i].gmem_used);
	}

	/*
	 * Also, decrement number of GpuContext
	 */
	pg_atomic_fetch_sub_u32(&gpuScoreBoard->num_gcontext, 1);

	/*
	 * If a series of queries are successfully executed, we keep cuContext
	 * being cached for the next execution. In this case, only memory context
	 * shall be released.
	 */
	if (sanity_release && cuda_last_contexts[0] == NULL)
	{
		for (i=0; i < cuda_num_devices; i++)
		{
			Assert(cuda_last_contexts[i] == NULL);
			cuda_last_contexts[i] = gcontext->gpu[i].cuda_context;
		}
		/* release the host pinned memory context, but retain cuContext */
		MemoryContextDelete(gcontext->memcxt);

		elog(DEBUG2, "CUDA context is cached to reuse");

		return;
	}

	/*
	 * NOTE: Be careful to drop the primary CUDA context because it also
	 * removes/unmaps all the memory region allocated by cuMemHostAlloc()
	 * that includes the GpuContext object and MemoryContext.
	 * So, we have to drop non-primary CUDA context, memory context, then
	 * the primary CUDA context.
	 */
	cuda_context = gcontext->gpu[0].cuda_context;
	for (i = gcontext->num_context - 1; i > 0; i--)
	{
		rc = cuCtxDestroy(gcontext->gpu[i].cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
	}
	/* release host pinned memory context */
	MemoryContextDelete(gcontext->memcxt);

	/* Drop the primary CUDA context */
	rc = cuCtxDestroy(cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
}

void
pgstrom_put_gpucontext(GpuContext *gcontext)
{
	Assert(gcontext->refcnt > 0);
	if (--gcontext->refcnt == 0)
		pgstrom_release_gpucontext(gcontext, true);
}

/*
 * pgstrom_cleanup_gputaskstate
 *
 * cleanup any active tasks and release cuda related resource, to end or
 * rescan executor.
 */
void
pgstrom_cleanup_gputaskstate(GpuTaskState *gts)
{
	GpuContext	   *gcontext = gts->gcontext;
	GpuTask		   *gtask;
	dlist_node	   *dnode;
	CUresult		rc;
	int				i;

	/*
	 * Synchronize all the concurrent task, if any
	 */
	if (gcontext)
	{
		for (i=0; i < gcontext->num_context; i++)
		{
			rc = cuCtxPushCurrent(gcontext->gpu[i].cuda_context);
			Assert(rc == CUDA_SUCCESS);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxPushCurrent: %s", errorText(rc));

			rc = cuCtxSynchronize();
			Assert(rc == CUDA_SUCCESS);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxSynchronize: %s", errorText(rc));

			rc = cuCtxPopCurrent(NULL);
			Assert(rc == CUDA_SUCCESS);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		}
	}

	/*
	 * Release GpuTasks tracked by this GpuTaskState
	 */
	SpinLockAcquire(&gts->lock);
	while (!dlist_is_empty(&gts->tracked_tasks))
	{
		dnode = dlist_pop_head_node(&gts->tracked_tasks);
		gtask = dlist_container(GpuTask, tracker, dnode);
		SpinLockRelease(&gts->lock);

		gts->cb_task_release(gtask);

		SpinLockAcquire(&gts->lock);
	}
	dlist_init(&gts->running_tasks);
	dlist_init(&gts->pending_tasks);
	dlist_init(&gts->completed_tasks);
	dlist_init(&gts->ready_tasks);
	gts->num_running_tasks = 0;
	gts->num_pending_tasks = 0;
	gts->num_ready_tasks = 0;
	SpinLockRelease(&gts->lock);

	gts->curr_task = NULL;
	gts->curr_index = 0;
}

void
pgstrom_release_gputaskstate(GpuTaskState *gts)
{
	GpuContext *gcontext = gts->gcontext;
	CUresult	rc;
	int			i;

	/* clean-up and release any concurrent tasks */
	pgstrom_cleanup_gputaskstate(gts);

	/* put any CUDA resources, if any */
	if (gcontext)
	{
		/* release cuda module, if any */
		if (gts->cuda_modules)
		{
			for (i=0; i < gcontext->num_context; i++)
			{
				rc = cuModuleUnload(gts->cuda_modules[i]);
				if (rc != CUDA_SUCCESS)
					elog(WARNING, "failed on cuModuleUnload: %s",
						 errorText(rc));
			}
			gts->cuda_modules = NULL;
		}
		/* put reference to the GpuContext */
		pgstrom_put_gpucontext(gts->gcontext);
		gts->gcontext = NULL;
	}
}

void
pgstrom_init_gputaskstate(GpuContext *gcontext, GpuTaskState *gts)
{
	gts->gcontext = gcontext;
	gts->kern_source = NULL;	/* to be set later */
	gts->extra_flags = 0;		/* to be set later */
	gts->cuda_modules = NULL;
	gts->scan_done = false;
	gts->scan_bulk = false;
	gts->scan_overflow = NULL;
	SpinLockInit(&gts->lock);
	dlist_init(&gts->tracked_tasks);
	dlist_init(&gts->running_tasks);
	dlist_init(&gts->pending_tasks);
	dlist_init(&gts->completed_tasks);
	dlist_init(&gts->ready_tasks);
	gts->num_running_tasks = 0;
	gts->num_pending_tasks = 0;
	gts->num_ready_tasks = 0;
	/* NOTE: caller has to set callbacks */
	gts->cb_task_process = NULL;
	gts->cb_task_complete = NULL;
	gts->cb_task_release = NULL;
	gts->cb_task_polling = NULL;
	gts->cb_next_chunk = NULL;
	gts->cb_next_tuple = NULL;
	memset(&gts->pfm_accum, 0, sizeof(pgstrom_perfmon));
	gts->pfm_accum.enabled = pgstrom_perfmon_enabled;
}

/*
 * check_completed_tasks
 *
 * It tries to move tasks in completed_tasks to ready_tasks after
 * device resource release. CUDA runtime hold device resources
 * until these are explicitly released, we should release these
 * resources as soon as possible we can.
 *
 * NOTE: spinlock has to be acquired before call
 */
static inline void
check_completed_tasks(GpuTaskState *gts)
{
	GpuTask		   *gtask;
	dlist_node	   *dnode;

	/*
	 * Allows GpuTaskState to check additional concurrent tasks,
	 * like dynamic background workers based on CPUs, if it has
	 * hybrid approach implementation.
	 * If and when concurrent task got completed, typically,
	 * callback detach task from the running_tasks and reconnect
	 * to completed_tasks
	 */
	if (gts->cb_task_polling)
		gts->cb_task_polling(gts);

	while (!dlist_is_empty(&gts->completed_tasks))
	{
		dnode = dlist_pop_head_node(&gts->completed_tasks);
		gtask = dlist_container(GpuTask, chain, dnode);
		gts->num_completed_tasks--;
		Assert(gtask->gts == gts);
		SpinLockRelease(&gts->lock);

		/*
		 * NOTE: cb_task_complete() allows task object to clean-up
		 * cuda resources in the earliest path, and some other task
		 * specific operations that have to be done in the backend
		 * context.
		 * Usually, it returns 'true' then task object is chained
		 * to the queue of ready_tasks. Elsewhere, callback side
		 * shall do all the necessary stuff - like retrying with
		 * larger buffer.
		 */
		if (gts->cb_task_complete(gtask))
		{
			/* release common cuda fields and its stream */
			pgstrom_cleanup_gputask_cuda_resources(gtask);

			SpinLockAcquire(&gts->lock);
			if (gtask->kerror.errcode != StromError_Success)
				dlist_push_head(&gts->ready_tasks, &gtask->chain);
			else
				dlist_push_tail(&gts->ready_tasks, &gtask->chain);
			gts->num_ready_tasks++;
		}
		else
		{
			/* all the exception handling was done on the callback */
			SpinLockAcquire(&gts->lock);
		}
	}
}

/*
 *
 *
 *
 */
static void
launch_pending_tasks(GpuTaskState *gts)
{
	GpuContext	   *gcontext = gts->gcontext;
	GpuTask		   *gtask;
	dlist_node	   *dnode;
	bool			launch;
	struct timeval	tv1, tv2;

	/*
	 * Unless kernel build is completed, we cannot launch it.
	 */
	if (!gts->cuda_modules)
	{
		if (!pgstrom_load_cuda_program(gts))
			return;
	}

	PERFMON_BEGIN(&gts->pfm_accum, &tv1);
	while (!dlist_is_empty(&gts->pending_tasks))
	{
		dnode = dlist_pop_head_node(&gts->pending_tasks);
		gtask = dlist_container(GpuTask, chain, dnode);
		gts->num_pending_tasks--;
		memset(&gtask->chain, 0, sizeof(dlist_node));
		SpinLockRelease(&gts->lock);

		/*
		 * Assign CUDA resources, if not yet
		 */
		if (!gtask->cuda_stream && !gtask->no_cuda_setup)
		{
			CUcontext	cuda_context;
			CUdevice	cuda_device;
			CUmodule	cuda_module;
			CUstream	cuda_stream;
			CUresult	rc;
			int			index;

			index = (gcontext->next_context++ % gcontext->num_context);
			cuda_device = gcontext->gpu[index].cuda_device;
			cuda_context = gcontext->gpu[index].cuda_context;
			cuda_module = gts->cuda_modules[index];
			rc = cuCtxPushCurrent(cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

			rc = cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuStreamCreate: %s", errorText(rc));

			gtask->cuda_index = index;
			gtask->cuda_context = cuda_context;
			gtask->cuda_device = cuda_device;
			gtask->cuda_module = cuda_module;
			gtask->cuda_stream = cuda_stream;

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		}

		/*
		 * Then, tries to launch this task.
		 */
		launch = gts->cb_task_process(gtask);

		/*
		 * NOTE: cb_process may complete task immediately, prior to get
		 * the spinlock again. So, we need to ensure gtask is not
		 * linked to the completed list at this moment.
		 */
		SpinLockAcquire(&gts->lock);
		if (!gtask->chain.prev && !gtask->chain.next)
		{
			if (launch)
			{
				dlist_push_tail(&gts->running_tasks, &gtask->chain);
				gts->num_running_tasks++;
			}
			else
			{
				/*
				 * NOTE: In case when callback required to keep the
				 * GpuTask in the pending queue, it implies device
				 * resources are in starvation. It does not make
				 * sense to enqueue tasks any more, at this moment.
				 */
				dlist_push_head(&gts->pending_tasks, &gtask->chain);
				gts->num_pending_tasks++;
				break;
			}
		}
	}
	PERFMON_END(&gts->pfm_accum, time_launch_cuda, &tv1, &tv2);
}

/*
 *
 *
 *
 *
 *
 *
 */
static bool
__waitfor_ready_tasks(GpuTaskState *gts)
{
	bool	retry_next = true;
	bool	wait_latch = true;
	bool	short_timeout = false;
	int		rc;

	/*
	 * Unless CUDA module is not loaded, we cannot launch process
	 * of GpuTask callback. So, we go on the long-waut path.
	 */
	if (gts->cuda_modules || pgstrom_load_cuda_program(gts))
	{
		SpinLockAcquire(&gts->lock);
		if (!dlist_is_empty(&gts->ready_tasks))
		{
			/*
			 * If we already has a ready chunk, no need to wait for
			 * the concurrent tasks. So, break loop immediately.
			 */
			wait_latch = false;
			retry_next = false;
		}
		else if (!dlist_is_empty(&gts->completed_tasks))
		{
			/*
			 * Even if we have no ready chunk yet, completed tasks
			 * will produce ready chunks immediately, without job
			 * synchronization.
			 */
			wait_latch = false;
			retry_next = true;
		}
		else if (!dlist_is_empty(&gts->pending_tasks))
		{
			/*
			 * Existence of pending tasks implies lack of device
			 * resources (like memory). Shorter blocking may allow
			 * to device resource polling. :-)
			 */
			short_timeout = true;
		}
		else if (!dlist_is_empty(&gts->running_tasks))
		{
			/*
			 * Even if no pending task we have, running task will
			 * wake-up our thread once it got completed. So, long-
			 * wait is sufficient to run.
			 */
		}
		else if (!gts->scan_done)
		{
			/*
			 * No ready, completed, running and pending tasks, even
			 * if scan is not completed, implies relation scan ratio
			 * is slower than GPU computing performance. So, we move
			 * to load the next chunk without blocking.
			 */
			wait_latch = false;
			retry_next = true;
		}
		else
		{
			wait_latch = false;
			retry_next = false;
		}
		SpinLockRelease(&gts->lock);
	}

	if (wait_latch)
	{
		struct timeval	tv1, tv2;

		PERFMON_BEGIN(&gts->pfm_accum, &tv1);

		rc = WaitLatch(&MyProc->procLatch,
					   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
					   !short_timeout ? 5000 : 200);
		ResetLatch(&MyProc->procLatch);
		if (rc & WL_POSTMASTER_DEATH)
			elog(ERROR, "Emergency bail out because of Postmaster crash");

		PERFMON_END(&gts->pfm_accum, time_sync_tasks, &tv1, &tv2);
	}
	return retry_next;
}

static bool
waitfor_ready_tasks(GpuTaskState *gts)
{
	bool	save_set_latch_on_sigusr1 = set_latch_on_sigusr1;
	bool	status;

	set_latch_on_sigusr1 = true;
	PG_TRY();
	{
		status = __waitfor_ready_tasks(gts);
	}
	PG_CATCH();
	{
		set_latch_on_sigusr1 = save_set_latch_on_sigusr1;
		PG_RE_THROW();
	}
	PG_END_TRY();
	set_latch_on_sigusr1 = save_set_latch_on_sigusr1;

	return status;
}

/*
 * gpucontext_health_check
 *
 * It checks current status of CUDA context. CUDA API may return an error
 * generated by previous asynchronouse call by the relevant CUDA context,
 * if GPU kernel got a fatal error.
 * The complited_tasks list is attached by callback of the CUDA runtime,
 * we may have infinite loop (probably, this callback does not work if
 * CUDA context got failed) with no health check capability.
 *
 * NOTE: We doubt cuCtxGetApiVersion() really returns an error. If we have
 * another better prober, we like to switch.
 */
static inline void
gpucontext_health_check(GpuContext *gcontext)
{
	cl_uint			i;

	for (i=0; i < gcontext->num_context; i++)
	{
		size_t		pvalue;
		CUresult	rc;

		rc = cuCtxPushCurrent(gcontext->gpu[i].cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

		rc = cuCtxGetLimit(&pvalue, CU_LIMIT_STACK_SIZE);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxGetLimit: %s", errorText(rc));

		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
	}
}

/*
 * pgstrom_fetch_gputask
 *
 * It loads a chunk from the target relation, then enqueue the GpuScan
 * chunk to be processed by OpenCL devices if valid device kernel was
 * constructed. Elsewhere, it works as a wrapper of pgstrom_load_gpuscan,
 */
GpuTask *
pgstrom_fetch_gputask(GpuTaskState *gts)
{
	GpuTask		   *gtask;
	dlist_node	   *dnode;
	bool			is_first_loop = false;

	/*
	 * In case when no device code will be executed, we do not need to have
	 * asynchronous execution. So, just return a chunk with synchronous
	 * manner.
	 */
	if (!gts->kern_source)
	{
		gtask = gts->cb_next_chunk(gts);
		Assert(!gtask || gtask->gts == gts);
		return gtask;
	}

	/*
	 * We try to keep multiple GpuTask requests being enqueued, unless
	 * it does not reach to pgstrom_max_async_tasks.
	 *
	 * TODO: number of requests should be controled by GpuContext, not
	 * GpuTaskState granuality. Needs more investigation.
	 */
	do {
		CHECK_FOR_INTERRUPTS();

		/* Health check of CUDA context */
		if (!is_first_loop)
			gpucontext_health_check(gts->gcontext);

		SpinLockAcquire(&gts->lock);
		check_completed_tasks(gts);
		launch_pending_tasks(gts);

		if (!gts->scan_done)
		{
			while (pgstrom_max_async_tasks > (gts->num_running_tasks +
											   gts->num_pending_tasks +
											   gts->num_ready_tasks))
			{
				/* no urgent reason why to make the scan progress */
				if (!dlist_is_empty(&gts->ready_tasks) &&
					pgstrom_max_async_tasks < (gts->num_running_tasks +
											   gts->num_pending_tasks))
					break;
				SpinLockRelease(&gts->lock);

				gtask = gts->cb_next_chunk(gts);
				Assert(!gtask || gtask->gts == gts);

				SpinLockAcquire(&gts->lock);
				if (!gtask)
				{
					gts->scan_done = true;
					elog(DEBUG1, "scan done (%s)",
						 gts->css.methods->CustomName);
					break;
				}
				dlist_push_tail(&gts->pending_tasks, &gtask->chain);
				gts->num_pending_tasks++;

				/* kick pending tasks */
				check_completed_tasks(gts);
				launch_pending_tasks(gts);
			}
		}
		else
		{
			if (gts->num_pending_tasks > 0)
				launch_pending_tasks(gts);
			else if (gts->num_running_tasks == 0)
			{
				SpinLockRelease(&gts->lock);
				break;
			}
		}
		SpinLockRelease(&gts->lock);
		is_first_loop = false;
	} while (waitfor_ready_tasks(gts));

	/*
	 * Picks up next available chunk if any
	 */
	SpinLockAcquire(&gts->lock);
	if (gts->num_ready_tasks == 0)
	{
		Assert(dlist_is_empty(&gts->ready_tasks));
		SpinLockRelease(&gts->lock);
		return NULL;
	}
	gts->num_ready_tasks--;
	dnode = dlist_pop_head_node(&gts->ready_tasks);
	gtask = dlist_container(GpuTask, chain, dnode);
    memset(&gtask->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&gts->lock);

	/*
	 * Error handling
	 */
	if (gtask->kerror.errcode != StromError_Success)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("PG-Strom: CUDA execution error (%s)",
						errorTextKernel(&gtask->kerror))));
	}
	return gtask;
}

TupleTableSlot *
pgstrom_exec_gputask(GpuTaskState *gts)
{
	TupleTableSlot *slot = gts->css.ss.ss_ScanTupleSlot;

	ExecClearTuple(slot);

	while (!gts->curr_task || !(slot = gts->cb_next_tuple(gts)))
	{
		GpuTask	   *gtask = gts->curr_task;

		/* release the current GpuTask object that was already scanned */
		if (gtask)
		{
			SpinLockAcquire(&gts->lock);
			dlist_delete(&gtask->tracker);
			SpinLockRelease(&gts->lock);
			gts->cb_task_release(gtask);
			gts->curr_task = NULL;
			gts->curr_index = 0;
		}
		/* reload next chunk to be scanned */
		gtask = pgstrom_fetch_gputask(gts);
		if (!gtask)
			break;
		gts->curr_task = gtask;
		gts->curr_index = 0;
	}
	return slot;
}

bool
pgstrom_recheck_gputask(GpuTaskState *gts, TupleTableSlot *slot)
{
	/* no GpuTaskState class needs recheck */
	return true;
}

void
pgstrom_init_gputask(GpuTaskState *gts, GpuTask *gtask)
{
	memset(gtask, 0, sizeof(GpuTask));
	gtask->gts = gts;
	/* to be tracked by GpuTaskState */
	SpinLockAcquire(&gts->lock);
	dlist_push_tail(&gts->tracked_tasks, &gtask->tracker);
	SpinLockRelease(&gts->lock);
	gtask->pfm.enabled = gts->pfm_accum.enabled;
}

void
pgstrom_release_gputask(GpuTask *gtask)
{
	GpuTaskState   *gts = gtask->gts;

	/* untrack this task by GpuTaskState */
	SpinLockAcquire(&gts->lock);
	dlist_delete(&gtask->tracker);
	SpinLockRelease(&gts->lock);

	/* per task cleanup */
	gts->cb_task_release(gtask);
}

/*
 * pgstrom_cleanup_gputask_cuda_resources
 *
 * it clears a common cuda resources; assigned on cb_task_process
 */
void
pgstrom_cleanup_gputask_cuda_resources(GpuTask *gtask)
{
	CUresult	rc;

	if (gtask->cuda_stream)
	{
		rc = cuStreamDestroy(gtask->cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuStreamDestroy: %s", errorText(rc));
	}
	gtask->cuda_index = 0;
	gtask->cuda_context = NULL;
	gtask->cuda_device = 0UL;
	gtask->cuda_stream = NULL;
	gtask->cuda_module = NULL;
}

/*
 *
 */
static void
gpucontext_cleanup_callback(ResourceReleasePhase phase,
							bool is_commit,
							bool is_toplevel,
							void *arg)
{
	dlist_iter		iter;

	if (phase != RESOURCE_RELEASE_AFTER_LOCKS)
		return;

	dlist_foreach(iter, &gcontext_list)
	{
		GpuContext *gcontext = dlist_container(GpuContext, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner)
		{
			if (is_commit)
				elog(WARNING, "Probably, someone forgot to put GpuContext");
			pgstrom_release_gpucontext(gcontext, false);
			break;
		}
	}
}

/*
 * gpuLocalMemSize
 *
 * It returns the size of local memory block on GPU device
 */
size_t
gpuLocalMemSize(void)
{
	return cuda_local_mem_size;
}

/*
 * gpuMaxThreadsPerBlock
 *
 * it returns the max size of local threads per block on GPU device
 */
cl_uint
gpuMaxThreadsPerBlock(void)
{
	return cuda_max_threads_per_block;
}

/*
 *
 *
 */
static __thread int __dynamic_shmem_per_thread;
static size_t
dynamic_shmem_size_per_block(int blockSize)
{
	return __dynamic_shmem_per_thread * (size_t)blockSize;
}

void
pgstrom_compute_workgroup_size(size_t *p_grid_size,
							   size_t *p_block_size,
							   CUfunction function,
							   CUdevice device,
							   bool maximum_blocksize,
							   size_t nitems,
							   size_t dynamic_shmem_per_thread)
{
	int			grid_size;
	int			block_size;
	int			max_shmem_size;
	int			static_shmem_size;
	int			warp_size;
	CUresult	rc;

	/* get statically allocated shared memory */
	rc = cuFuncGetAttribute(&static_shmem_size,
							CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
							function);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

	/* get warp size of the target device */
	rc = cuDeviceGetAttribute(&warp_size,
							  CU_DEVICE_ATTRIBUTE_WARP_SIZE,
							  device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s",
			 errorText(rc));

	if (maximum_blocksize)
	{
		rc = cuFuncGetAttribute(&block_size,
								CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
								function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s",
				 errorText(rc));

		rc = cuDeviceGetAttribute(&max_shmem_size,
							CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
								  device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGetAttribute: %s",
				 errorText(rc));

		if (static_shmem_size +
			dynamic_shmem_per_thread * block_size > max_shmem_size)
		{
			block_size = ((max_shmem_size - static_shmem_size)
						  / dynamic_shmem_per_thread);
			block_size &= ~(warp_size - 1);

			if (static_shmem_size > max_shmem_size || block_size < warp_size)
				elog(ERROR, "Too much GPU shared memory required");
		}
	}
	else
	{
		__dynamic_shmem_per_thread = dynamic_shmem_per_thread;
		rc = cuOccupancyMaxPotentialBlockSize(&grid_size,
											  &block_size,
											  function,
											  dynamic_shmem_size_per_block,
											  static_shmem_size,
											  cuda_max_threads_per_block);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuOccupancyMaxPotentialBlockSize: %s",
				 errorText(rc));
	}

	/*
	 * Adjustment of the block size, if it is larger than nitems including
	 * the case when nitems == 0. We expects block_size it at least equal
	 * to the WARP_SIZE.
	 */
	if (block_size > nitems)
		block_size = (nitems + warp_size - 1) & ~(warp_size - 1);
	block_size = Max(block_size, warp_size);

	*p_block_size = block_size;
	*p_grid_size = (nitems + block_size - 1) / block_size;
}

/*
 * pgstrom_compute_workgroup_size_2d
 *
 * special routine to compute an optimal kernel launch size if caller
 * wants to have 2-dimensional threading model.
 * It intends to maximize the number of concurrent threads as long as
 * resource consumption allows.
 */
void
pgstrom_compute_workgroup_size_2d(size_t *p_grid_xsize,
								  size_t *p_block_xsize,
								  size_t *p_grid_ysize,
								  size_t *p_block_ysize,
								  CUfunction function,
								  CUdevice device,
								  size_t x_nitems,
								  size_t y_nitems,
								  size_t dynamic_shmem_per_xitems,
								  size_t dynamic_shmem_per_yitems,
								  size_t dynamic_shmem_per_thread)
{
	int			block_xsize;
	int			block_ysize;
	int			root_block_size;
	int			max_block_size;
	int			max_shmem_size;
	int			static_shmem_size;
	int			warp_size;
	CUresult	rc;

	/* get capability of the device */
	rc = cuDeviceGetAttribute(&max_shmem_size,
							  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
							  device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	rc = cuDeviceGetAttribute(&warp_size,
							  CU_DEVICE_ATTRIBUTE_WARP_SIZE,
							  device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	/* get resource consumption by the kernel function */
	rc = cuFuncGetAttribute(&static_shmem_size,
							CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
							function);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

   	rc = cuFuncGetAttribute(&max_block_size,
							CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
							function);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

	/* reduce max block size, if unavailable to allocate shared memory */
	if (static_shmem_size +
		dynamic_shmem_per_thread * max_block_size > max_shmem_size)
	{
		max_block_size = ((max_shmem_size - static_shmem_size)
						  / dynamic_shmem_per_thread);
		max_block_size &= ~(warp_size - 1);

		if (static_shmem_size > max_shmem_size || max_block_size < warp_size)
			elog(ERROR, "Too much GPU shared memory required");
	}
	/* 1024 = 32 * 32 */
	root_block_size = (int) sqrt(max_block_size);

	block_ysize = Min(root_block_size, y_nitems);
	block_ysize = Max(block_ysize, 1);
	block_xsize = ((max_block_size / block_ysize) & ~(warp_size - 1));
	block_xsize = Max(block_xsize, 1);

	/*
	 * adjust block_xsize and _ysize according to the expected shared memory
	 * consumption.
	 */
	while (static_shmem_size +
		   block_xsize * dynamic_shmem_per_xitems +
		   block_ysize * dynamic_shmem_per_yitems > max_shmem_size)
	{
		if (block_xsize * dynamic_shmem_per_xitems >=
			block_ysize * dynamic_shmem_per_yitems)
		{
			if (block_xsize <= warp_size)
				elog(ERROR, "We cannot reduce block_xsize any more");
			block_xsize -= warp_size;
			block_ysize = max_block_size / block_xsize;
		}
		else
		{
			if (block_ysize <= 1)
				elog(ERROR, "We cannot reduce block_ysize any more");
			block_ysize--;
			block_xsize = (max_block_size / block_ysize) & ~(warp_size - 1);
		}
	}
	/* put results */
	*p_block_xsize = block_xsize;
	*p_block_ysize = block_ysize;
	*p_grid_xsize = Max((x_nitems + block_xsize - 1) / block_xsize, 1);
	*p_grid_ysize = Max((y_nitems + block_ysize - 1) / block_ysize, 1);
}

/*
 * Device properties referenced to log messages on starting-up time,
 * and validate devices to be used.
 */
struct target_cuda_device
{
	int		dev_id;
	char   *dev_name;
	size_t	dev_mem_sz;
	int		dev_mem_clk;
	int		dev_mem_width;
	int		dev_l2_sz;
	int		dev_cap_major;
	int		dev_cap_minor;
	int		dev_mpu_nums;
	int		dev_mpu_clk;
	int		dev_max_threads_per_block;
	int		dev_local_mem_sz;
};

static inline void
check_target_cuda_device(struct target_cuda_device *dattr)
{
	MemoryContext oldcxt;
	int		cores_per_mpu = -1;
	int		dev_cap;

	/*
	 * CUDA device older than Kepler is not supported
	 */
	if (dattr->dev_cap_major < 3)
		goto out;

	/*
	 * Number of CUDA cores (just for log messages)
	 */
	if (dattr->dev_cap_major == 1)
		cores_per_mpu = 8;
	else if (dattr->dev_cap_major == 2)
	{
		if (dattr->dev_cap_minor == 0)
			cores_per_mpu = 32;
		else if (dattr->dev_cap_minor == 1)
			cores_per_mpu = 48;
		else
			cores_per_mpu = -1;
	}
	else if (dattr->dev_cap_major == 3)
		cores_per_mpu = 192;
	else if (dattr->dev_cap_major == 5)
		cores_per_mpu = 128;

	/*
	 * Referenced computing capability
	 */
	dev_cap = 10 * dattr->dev_cap_major + dattr->dev_cap_minor;

	/*
	 * Track referenced device property
	 */
	oldcxt = MemoryContextSwitchTo(TopMemoryContext);

	cuda_max_malloc_size = Min(cuda_max_malloc_size,
							   (dattr->dev_mem_sz / 3) & ~((1UL << 20) - 1));
	cuda_max_threads_per_block = Min(cuda_max_threads_per_block,
									 dattr->dev_max_threads_per_block);
	cuda_local_mem_size = Min(cuda_local_mem_size,
							  dattr->dev_local_mem_sz);
	cuda_compute_capability = Min(cuda_compute_capability, dev_cap);

	cuda_device_ordinals = lappend_int(cuda_device_ordinals, dattr->dev_id);
	cuda_device_capabilities =
		list_append_unique_int(cuda_device_capabilities, dev_cap);
	cuda_device_mem_sizes = lappend_int(cuda_device_mem_sizes,
										(int)(dattr->dev_mem_sz >> 20));
	MemoryContextSwitchTo(oldcxt);
out:
	/* Log the brief CUDA device properties */
	elog(LOG, "GPU%d %s (%d %s, %dMHz), L2 %dKB, RAM %zuMB (%dbits, %dMHz), capability %d.%d%s",
		 dattr->dev_id,
		 dattr->dev_name,
		 (cores_per_mpu > 0 ? cores_per_mpu : 1) * dattr->dev_mpu_nums,
		 (cores_per_mpu > 0 ? "CUDA cores" : "SMs"),
		 dattr->dev_mpu_clk / 1000,
		 dattr->dev_l2_sz >> 10,
		 dattr->dev_mem_sz >> 20,
		 dattr->dev_mem_width,
		 dattr->dev_mem_clk / 1000,
		 dattr->dev_cap_major,
		 dattr->dev_cap_minor,
		 dattr->dev_cap_major < 3 ? ", NOT SUPPORTED" : "");

	/* clear the target_cuda_device structure */
	if (dattr->dev_name)
		pfree(dattr->dev_name);
	memset(dattr, 0, sizeof(struct target_cuda_device));
	dattr->dev_id = -1;
}

static void
pickup_target_cuda_devices(void)
{
	FILE	   *filp;
	char	   *cmdline;
	char		linebuf[2048];
	char	   *attkey;
	char	   *attval;
	char	   *pos;
	struct target_cuda_device dattr;

	/* device properties to be set */
	memset(&dattr, 0, sizeof(dattr));
	dattr.dev_id = -1;

	cmdline = psprintf("%s -m", CMD_GPUINFO_PATH);
	filp = OpenPipeStream(cmdline, PG_BINARY_R);

	while (fgets(linebuf, sizeof(linebuf), filp) != NULL)
	{
		attval = strrchr(linebuf, '=');
		if (!attval)
		{
			if (linebuf[0] == '\n' || linebuf[0] == '\0')
				continue;
			elog(ERROR, "Unexpected gpuinfo -m input:\n%s\n", linebuf);
		}
		*attval++ = '\0';
		pos = attval + strlen(attval) - 1;
		while (isspace(*pos))
			*pos-- = '\0';

		if (strncmp(linebuf, "CU_PLATFORM_ATTRIBUTE_", 22) == 0)
		{
			attkey = linebuf + 22;

			if (strcmp(attkey, "CUDA_RUNTIME_VERSION") == 0)
			{
				elog(LOG, "CUDA Runtime version: %s", attval);
			}
			else if (strcmp(attkey, "NVIDIA_DRIVER_VERSION") == 0)
			{
				elog(LOG, "NVIDIA driver version: %s", attval);
			}
		}
		else if (strncmp(linebuf, "CU_DEVICE_ATTRIBUTE_", 20) == 0)
		{
			attkey = linebuf + 20;

			if (strcmp(attkey, "DEVICE_ID") == 0)
			{
				if (dattr.dev_id >= 0)
					check_target_cuda_device(&dattr);
				dattr.dev_id = atoi(attval);
			}
			else if (strcmp(attkey, "DEVICE_NAME") == 0)
				dattr.dev_name = pstrdup(attval);
			else if (strcmp(attkey, "GLOBAL_MEMORY_SIZE") == 0)
				dattr.dev_mem_sz = (size_t) atol(attval);
			else if (strcmp(attkey, "MAX_THREADS_PER_BLOCK") == 0)
				dattr.dev_max_threads_per_block = atoi(attval);
			else if (strcmp(attkey, "MAX_SHARED_MEMORY_PER_BLOCK") == 0)
				dattr.dev_local_mem_sz = atoi(attval);
			else if (strcmp(attkey, "MEMORY_CLOCK_RATE") == 0)
				dattr.dev_mem_clk = atoi(attval);
			else if (strcmp(attkey, "GLOBAL_MEMORY_BUS_WIDTH") == 0)
				dattr.dev_mem_width = atoi(attval);
			else if (strcmp(attkey, "L2_CACHE_SIZE") == 0)
				dattr.dev_l2_sz = atoi(attval);
			else if (strcmp(attkey, "COMPUTE_CAPABILITY_MAJOR") == 0)
				dattr.dev_cap_major = atoi(attval);
			else if (strcmp(attkey, "COMPUTE_CAPABILITY_MINOR") == 0)
				dattr.dev_cap_minor = atoi(attval);
			else if (strcmp(attkey, "MULTIPROCESSOR_COUNT") == 0)
				dattr.dev_mpu_nums = atoi(attval);
			else if (strcmp(attkey, "CLOCK_RATE") == 0)
				dattr.dev_mpu_clk = atoi(attval);
		}
		else if (strcmp(linebuf, "\n") != 0)
			elog(LOG, "Unknown attribute: %s", linebuf);
	}
	/* dump the last device */
	if (dattr.dev_id >= 0)
		check_target_cuda_device(&dattr);

	ClosePipeStream(filp);
}

static void
pgstrom_startup_cuda_control(void)
{
	cl_int		i, num_devices;
	ListCell   *lc;
	bool		found;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	Assert(list_length(cuda_device_ordinals) ==
		   list_length(cuda_device_mem_sizes));
	num_devices = list_length(cuda_device_mem_sizes);
	gpuScoreBoard = ShmemInitStruct("PG-Strom GPU Score Board",
									offsetof(GpuScoreBoard,
											 gpu[num_devices]), &found);
	if (found)
		elog(ERROR, "Bug? shared memory for GPU score board already exists");

	/* initialize fields */
	memset(gpuScoreBoard, 0, offsetof(GpuScoreBoard, gpu[num_devices]));
	i = 0;
	gpuScoreBoard->num_devices = num_devices;
	foreach (lc, cuda_device_mem_sizes)
	{
		gpuScoreBoard->gpu[i].gmem_size = ((size_t)lfirst_int(lc) << 20);
		i++;
	}
}

void
pgstrom_init_cuda_control(void)
{
	static char	   *cuda_visible_devices;
	int				count;

	/*
	 * GPU variables related to CUDA environment
	 */
	DefineCustomStringVariable("pg_strom.cuda_visible_devices",
							   "CUDA_VISIBLE_DEVICES of CUDA runtime",
							   NULL,
							   &cuda_visible_devices,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	if (cuda_visible_devices)
	{
		if (setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices, 1) != 0)
			elog(ERROR, "failed to set CUDA_VISIBLE_DEVICES");
	}

	/*
	 * Picks up target CUDA devices
	 */
	pickup_target_cuda_devices();
	if (cuda_device_ordinals == NIL)
		elog(ERROR, "No CUDA devices, PG-Strom was disabled");
	if (list_length(cuda_device_capabilities) > 1)
		elog(WARNING, "Mixture of multiple GPU device capabilities");

	/*
	 * initialization of GpuContext related stuff
	 */
	dlist_init(&gcontext_list);
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);

	/*
	 * allocation of static shared memory for resource scoreboard
	 */
	count = list_length(cuda_device_ordinals);
	RequestAddinShmemSpace(MAXALIGN(offsetof(GpuScoreBoard, gpu[count])));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_control;
}

/*
 * pgstrom_baseline_cuda_capability
 *
 * it returns the baseline cuda capability to be compiled.
 */
int
pgstrom_baseline_cuda_capability(void)
{
	return cuda_compute_capability;
}

/*
 * errorText
 *
 * translation from cuda error code to text representation
 */
const char *
errorText(int errcode)
{
	static __thread char buffer[512];
	const char *error_val;
	const char *error_str;

	switch (errcode)
	{
		case StromError_CpuReCheck:
			snprintf(buffer, sizeof(buffer), "CPU ReCheck");
			break;
		case StromError_CudaInternal:
			snprintf(buffer, sizeof(buffer), "CUDA Internal Error");
			break;
		case StromError_OutOfMemory:
			snprintf(buffer, sizeof(buffer), "Out of memory");
			break;
		case StromError_OutOfSharedMemory:
			snprintf(buffer, sizeof(buffer), "Out of shared memory");
			break;
		case StromError_DataStoreCorruption:
			snprintf(buffer, sizeof(buffer), "Data store corruption");
			break;
		case StromError_DataStoreNoSpace:
			snprintf(buffer, sizeof(buffer), "Data store no space");
			break;
		case StromError_DataStoreOutOfRange:
			snprintf(buffer, sizeof(buffer), "Data store out of range");
			break;
		case StromError_SanityCheckViolation:
			snprintf(buffer, sizeof(buffer), "Sanity check violation");
			break;
		default:
			if (cuGetErrorName(errcode, &error_val) == CUDA_SUCCESS &&
				cuGetErrorString(errcode, &error_str) == CUDA_SUCCESS)
				snprintf(buffer, sizeof(buffer), "%s - %s",
						 error_val, error_str);
			else
				snprintf(buffer, sizeof(buffer), "%d - unknown", errcode);
	}
	return buffer;
}

/*
 * errorTextKernel
 *
 * translation from kern_errorbuf to human readable representation
 */
const char *
errorTextKernel(kern_errorbuf *kerror)
{
	static __thread char buffer[1024];
	const char *kernel_name;

#define KERN_ENTRY(KERNEL)						\
	case StromKernel_##KERNEL: kernel_name = #KERNEL; break

	switch (kerror->kernel)
	{
		KERN_ENTRY(CudaRuntime);
		KERN_ENTRY(gpuscan_qual);
		KERN_ENTRY(gpujoin_preparation);
		KERN_ENTRY(gpujoin_exec_nestloop);
		KERN_ENTRY(gpujoin_exec_hashjoin);
		KERN_ENTRY(gpujoin_outer_nestloop);
		KERN_ENTRY(gpujoin_outer_hashjoin);
		KERN_ENTRY(gpujoin_projection_row);
		KERN_ENTRY(gpujoin_projection_slot);
		KERN_ENTRY(gpupreagg_preparation);
		KERN_ENTRY(gpupreagg_local_reduction);
		KERN_ENTRY(gpupreagg_global_reduction);
		KERN_ENTRY(gpupreagg_nogroup_reduction);
		KERN_ENTRY(gpupreagg_final_preparation);
		KERN_ENTRY(gpupreagg_final_reduction);
		KERN_ENTRY(gpupreagg_fixup_varlena);
		KERN_ENTRY(gpusort_preparation);
		KERN_ENTRY(gpusort_bitonic_local);
		KERN_ENTRY(gpusort_bitonic_step);
		KERN_ENTRY(gpusort_bitonic_merge);
		KERN_ENTRY(gpusort_fixup_datastore);
		default:
			kernel_name = "unknown kernel";
			break;
	}
#undef KERN_ENTRY
	snprintf(buffer, sizeof(buffer), "%s:%d %s",
			 kernel_name, kerror->lineno,
			 errorText(kerror->errcode));
	return buffer;
}

/*
 * pgstrom_scoreboard_info
 *
 *
 */
Datum
pgstrom_scoreboard_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	const char	   *att_name;
	char		   *att_value;
	Datum			values[2];
	bool			isnull[2];
	HeapTuple		tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
        oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(2, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "attribute",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = 0;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->call_cntr == 0)
	{
		att_name = "num devices";
		att_value = psprintf("%u", gpuScoreBoard->num_devices);
	}
	else if (fncxt->call_cntr == 1)
	{
		att_name = "num contexts";
		att_value = psprintf("%u", GpuScoreCurrNumContext());
	}
	else if (fncxt->call_cntr < 2 + 2 * gpuScoreBoard->num_devices)
	{
		int		cuda_index = (fncxt->call_cntr - 2) / 2;
		int		attr_index = (fncxt->call_cntr - 2) % 2;
		size_t	length;

		switch (attr_index)
		{
			case 0:
				length = gpuScoreBoard->gpu[cuda_index].gmem_size;
				att_name = psprintf("GPU RAM size %u", cuda_index);
				att_value = psprintf("%zu MB", length >> 20);
				break;
			case 1:
				length = GpuScoreCurrMemUsage(cuda_index);
				att_name = psprintf("GPU RAM usage %u", cuda_index);
				att_value = psprintf("%zu MB", length >> 20);
				break;
			default:
				elog(ERROR, "unexpected attribute of pgstrom_scoreboard_info");
				break;
		}
	}
	else
		SRF_RETURN_DONE(fncxt);

	memset(isnull, 0, sizeof(isnull));
	values[0] = CStringGetTextDatum(att_name);
	values[1] = CStringGetTextDatum(att_value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_scoreboard_info);


/*
 * pgstrom_device_info
 *
 *
 *
 */
#define DEVATTR_BOOL		1
#define DEVATTR_INT			2
#define DEVATTR_KB			3
#define DEVATTR_MHZ			4
#define DEVATTR_COMP_MODE	5
#define DEVATTR_BITS		6
Datum
pgstrom_device_info(PG_FUNCTION_ARGS)
{
	static struct {
		CUdevice_attribute	attrib;
		const char		   *attname;
		int					attkind;
	} catalog[] = {
		{CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
		 "max threads per block", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
		 "Maximum block dimension X", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
		 "Maximum block dimension Y", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
		 "Maximum block dimension Z", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
		 "Maximum grid dimension X", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
		 "Maximum grid dimension Y", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
		 "Maximum grid dimension Z", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
		 "Maximum shared memory available per block", DEVATTR_KB },
		{CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
		 "Memory available on device for __constant__", DEVATTR_KB},
		{CU_DEVICE_ATTRIBUTE_WARP_SIZE,
		 "Warp size in threads", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_PITCH,
		 "Maximum pitch in bytes allowed by memory copies", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
		 "Maximum number of 32bit registers available per block", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
		 "Typical clock frequency in kilohertz", DEVATTR_MHZ},
		{CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
		 "Alignment requirement for textures", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
		 "Number of multiprocessors on device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
		 "Has kernel execution timeout", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_INTEGRATED,
		 "Integrated with host memory", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
		 "Host memory can be mapped to CUDA address space", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
		 "Compute mode", DEVATTR_COMP_MODE},
		{CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
		 "Alignment requirement for surfaces", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
		 "Multiple concurrent kernel support", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
		 "Device has ECC support enabled", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
		 "PCI bus ID of the device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
		 "PCI device ID of the device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
		 "Device is using TCC driver model", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
		 "Peak memory clock frequency", DEVATTR_MHZ},
		{CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
		 "Global memory bus width", DEVATTR_BITS},
		{CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
		 "Size of L2 cache in bytes", DEVATTR_KB},
		{CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
		 "Maximum threads per multiprocessor", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
		 "Number of asynchronous engines", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
		 "Device shares unified address space", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
		 "PCI domain ID of the device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		 "Major compute capability version number", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		 "Minor compute capability version number", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
		 "Device supports stream priorities", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
		 "Device supports caching globals in L1", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
		 "Device supports caching locals in L1", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
		 "Maximum shared memory per multiprocessor", DEVATTR_KB},
		{CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
		 "Maximum number of 32bit registers per multiprocessor", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
		 "Device can allocate managed memory on this system", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
		 "Device is on a multi-GPU board", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
		 "Unique id if device is on a multi-GPU board", DEVATTR_INT},
	};
	FuncCallContext *fncxt;
	CUresult	rc;
	int			dindex;
	int			aindex;
	const char *att_name;
	char	   *att_value;
	Datum		values[3];
	bool		isnull[3];
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
        oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(3, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "attribute",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = 0;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	dindex = fncxt->call_cntr / (lengthof(catalog) + 2);
	aindex = fncxt->call_cntr % (lengthof(catalog) + 2);

	if (cuda_num_devices < 0)
		pgstrom_init_cuda();

	if (dindex >= cuda_num_devices)
		SRF_RETURN_DONE(fncxt);

	if (aindex == 0)
	{
		char	dev_name[256];

		rc = cuDeviceGetName(dev_name, sizeof(dev_name), cuda_devices[dindex]);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGetName: %s", errorText(rc));
		att_name = "Device name";
		att_value = pstrdup(dev_name);
	}
	else if (aindex == 1)
	{
		size_t	dev_memsz;

		rc = cuDeviceTotalMem(&dev_memsz, cuda_devices[dindex]);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceTotalMem: %s", errorText(rc));
		att_name = "Total global memory size";
		att_value = psprintf("%zu MBytes", dev_memsz >> 20);
	}
	else
	{
		int		pindex = aindex - 2;
		int		property;

		rc = cuDeviceGetAttribute(&property,
								  catalog[pindex].attrib,
								  cuda_devices[dindex]);
		Assert(rc == CUDA_SUCCESS);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGetAttribute: %s",
				 errorText(rc));

		att_name = catalog[pindex].attname;
		switch (catalog[pindex].attkind)
		{
			case DEVATTR_BOOL:
				att_value = psprintf("%s", property != 0 ? "True" : "False");
				break;
			case DEVATTR_INT:
				att_value = psprintf("%d", property);
				break;
			case DEVATTR_KB:
				att_value = psprintf("%d KBytes", property / 1024);
				break;
			case DEVATTR_MHZ:
				att_value = psprintf("%d MHz", property / 1000);
				break;
			case DEVATTR_COMP_MODE:
				switch (property)
				{
					case CU_COMPUTEMODE_DEFAULT:
						att_value = "Default";
						break;
					case CU_COMPUTEMODE_EXCLUSIVE:
						att_value = "Exclusive";
						break;
					case CU_COMPUTEMODE_PROHIBITED:
						att_value = "Prohibited";
						break;
					case CU_COMPUTEMODE_EXCLUSIVE_PROCESS:
						att_value = "Exclusive Process";
						break;
					default:
						att_value = "Unknown";
						break;
				}
				break;
			case DEVATTR_BITS:
				att_value = psprintf("%d bits", property);
				break;
			default:
				elog(ERROR, "Bug? unexpected device attribute type");
		}
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dindex);
	values[1] = CStringGetTextDatum(att_name);
	values[2] = CStringGetTextDatum(att_value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_device_info);
