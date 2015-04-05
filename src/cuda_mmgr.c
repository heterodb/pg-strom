/*
 * cuda_mmgr.c
 *
 * Routines of memory context for host pinned memory.
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

#include "utils/memdebug.h"
#include "utils/memutils.h"

#include "pg_strom.h"

#define HOSTMEM_CHUNKSZ_MAX_BIT		36
#define HOSTMEM_CHUNKSZ_MIN_BIT		8
#define HOSTMEM_CHUNKSZ_MAX			(1UL << HOSTMEM_CHUNKSZ_MAX_BIT)
#define HOSTMEM_CHUNKSZ_MIN			(1UL << HOSTMEM_CHUNKSZ_MIN_BIT)
#define HOSTMEM_CHUNK_DATA(chunk)	((chunk)->chunk_data)
#define HOSTMEM_CHUNK_MAGIC_CODE		0xdeadbeaf
#define HOSTMEM_CHUNK_MAGIC(chunk)				\
	*((cl_uint *)((char *)(chunk) +				\
				  (chunk)->chunk_head.size -	\
				  sizeof(cl_uint)))

struct cudaHostMemBlock;

typedef struct
{
	struct cudaHostMemBlock *chm_block;	/* block that owns this chunk */
	dlist_node		addr_chain;	/* link to addr_chunks */
	dlist_node		free_chain;	/* link to free_chunks, or zero if active */
	StandardChunkHeader chunk_head;
	/*
	 * chunk_head.context : MemoryContext that owns this block
	 * chunk_head.size    : size of this chunk; to be 2^N
	 * chunk_head.requested_size : size actually requested
	 */
	char			chunk_data[FLEXIBLE_ARRAY_MEMBER];
} cudaHostMemChunk;

typedef struct cudaHostMemBlock
{
	dlist_node			chain;			/* link to active_blocks */
	dlist_head			addr_chunks;	/* list of chunks in address order, or
										 * zero if external block. */
	cudaHostMemChunk	first_chunk;	/* first chunk of this block */
} cudaHostMemBlock;

typedef struct
{
	MemoryContextData	header;
	CUcontext			cuda_context;
	dlist_head			blocks;
	dlist_head			free_chunks[HOSTMEM_CHUNKSZ_MAX_BIT + 1];
	/* allocation parameters for this context */
	Size				block_size_init;	/* init block size */
	Size				block_size_next;
	Size				block_size_max;		/* max block size */
} cudaHostMemHead;




static bool
cudaHostMemSplit(cudaHostMemHead *chm_head, int chm_class)
{
	cudaHostMemChunk   *chunk1;
	cudaHostMemChunk   *chunk2;
	dlist_node		   *dnode;

	Assert(chm_class > HOSTMEM_CHUNKSZ_MIN_BIT);

	if (dlist_is_empty(&chm_head->free_chunks[chm_class]))
	{
		if (chm_class >= HOSTMEM_CHUNKSZ_MAX_BIT)
			return false;	/* nothing to split any more */
		if (!cudaHostMemSplit(chm_head, chm_class + 1))
			return false;	/* no larger free chunk any more */
	}
	Assert(!dlist_is_empty(&chm_head->free_chunks[chm_class]));
	dnode = dlist_pop_head_node(&chm_head->free_chunks[chm_class]);
	chunk1 = dlist_container(cudaHostMemChunk, free_chain, dnode);
	Assert((chunk1->chunk_head.size & (chunk1->chunk_head.size - 1)) == 0);
	Assert(chm_class == get_next_log2(chunk1->chunk_head.size));
	chunk1->chunk_head.size /= 2;

	chunk2 = (cudaHostMemChunk *)((char *)chunk1 + chunk1->chunk_head.size);
	chunk2->chm_block = chunk1->chm_block;
	chunk2->chunk_head.context = chunk1->chunk_head.context;
	chunk2->chunk_head.size = chunk1->chunk_head.size;

	dlist_insert_after(&chunk1->addr_chain,
					   &chunk2->addr_chain);
	dlist_push_tail(&chm_head->free_chunks[chm_class - 1],
					&chunk1->free_chain);
	dlist_push_tail(&chm_head->free_chunks[chm_class - 1],
                    &chunk2->free_chain);
	return true;
}

static void
cudaHostMemAllocBlock(cudaHostMemHead *chm_head, int least_class)
{
	cudaHostMemBlock *chm_block;
	cudaHostMemChunk *chm_chunk;
	Size		block_size = chm_head->block_size_next;
	Size		least_size = (1UL << least_class);
	int			index;
	CUresult	rc;

	/* find out the best fit, and update next allocation standpoint */
	while (block_size < least_size)
		block_size *= 2;
	chm_head->block_size_next = Min(2 * chm_head->block_size_next,
									chm_head->block_size_max);
	Assert((block_size & (block_size - 1)) == 0);

	/* allocate host pinned memory */
	rc = cuCtxPushCurrent(chm_head->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

	rc = cuMemAllocHost((void **)&chm_block,
						offsetof(cudaHostMemBlock, first_chunk) + block_size);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemAllocHost: %s", errorText(rc));

	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPopCurrent: %s", errorText(rc));

	/* init block */
	dlist_init(&chm_block->addr_chunks);

	/* init first chunk */
	chm_chunk = &chm_block->first_chunk;
	memset(chm_chunk, 0, sizeof(cudaHostMemChunk));
	chm_chunk->chm_block = chm_block;
	chm_chunk->chunk_head.size = block_size;
	chm_chunk->chunk_head.context = &chm_head->header;
	/* add chunk to addr list */
	dlist_push_tail(&chm_block->addr_chunks, &chm_chunk->addr_chain);

	/* add chunk to free list */
	index = get_next_log2(chm_chunk->chunk_head.size);
	dlist_push_tail(&chm_head->free_chunks[index], &chm_chunk->free_chain);
}

static void *
cudaHostMemAlloc(MemoryContext context, Size required)
{
	cudaHostMemHead	   *chm_head = (cudaHostMemHead *) context;
	cudaHostMemChunk   *chm_chunk;
	dlist_node		   *dnode;
	Size				chunk_size;
	int					chunk_class;

	/* formalize the required size to 2^N of chunk_size */
	chunk_size = MAXALIGN(offsetof(cudaHostMemChunk, chunk_data) +
						  required +
						  sizeof(cl_uint));
	chunk_size = Max(chunk_size, HOSTMEM_CHUNKSZ_MIN);
	chunk_class = get_next_log2(chunk_size);
	chunk_size = (1UL << chunk_class);	/* to be 2^N bytes */
	if (chunk_size > chm_head->block_size_max)
		elog(ERROR, "pinned memory requiest %zu bytes too large", required);

	/* find a free chunk */
retry:
	if (dlist_is_empty(&chm_head->free_chunks[chunk_class]))
	{
		if (!cudaHostMemSplit(chm_head, chunk_class + 1))
		{
			cudaHostMemAllocBlock(chm_head, chunk_class);
			goto retry;
		}
	}
	Assert(!dlist_is_empty(&chm_head->free_chunks[chunk_class]));

	dnode = dlist_pop_head_node(&chm_head->free_chunks[chunk_class]);
	chm_chunk = dlist_container(cudaHostMemChunk, free_chain, dnode);
	memset(&chm_chunk->free_chain, 0, sizeof(dlist_node));
	Assert(chm_chunk->chunk_head.context = &chm_head->header);
	Assert(chm_chunk->chunk_head.size == chunk_size);
#ifdef MEMORY_CONTEXT_CHECKING
	chm_chunk->chunk_head.requested_size = required;
#endif
	HOSTMEM_CHUNK_MAGIC(chm_chunk) = HOSTMEM_CHUNK_MAGIC_CODE;

	return HOSTMEM_CHUNK_DATA(chm_chunk);
}

static void
cudaHostMemFree(MemoryContext context, void *pointer)
{
	cudaHostMemHead	   *chm_head = (cudaHostMemHead *) context;
	cudaHostMemBlock   *chm_block;
	cudaHostMemChunk   *chunk;
	cudaHostMemChunk   *buddy;
	dlist_node		   *dnode;
	uintptr_t			offset;
	int					index;

	chunk = (cudaHostMemChunk *)
		((char *)pointer - offsetof(cudaHostMemChunk, chunk_data));
	Assert(HOSTMEM_CHUNK_MAGIC(chunk) == HOSTMEM_CHUNK_MAGIC_CODE);
	chm_block = chunk->chm_block;

	while (true)
	{
		Assert((chunk->chunk_head.size & (chunk->chunk_head.size - 1)) == 0);
		Assert(!chunk->free_chain.prev && !chunk->free_chain.next);
		Assert(chunk->chm_block == chm_block);

		offset = (uintptr_t)chunk - (uintptr_t)&chm_block->first_chunk;
		Assert((offset & (chunk->chunk_head.size - 1)) == 0);
		if ((offset & chunk->chunk_head.size) == 0)
		{
			/* this chunk should be merged with next chunk */
			if (!dlist_has_next(&chm_block->addr_chunks,
								&chunk->addr_chain))
				break;

			/* cannot merge with active chunk, of course */
			dnode = dlist_next_node(&chm_block->addr_chunks,
									&chunk->addr_chain);
			buddy = dlist_container(cudaHostMemChunk, addr_chain, dnode);
			if (!buddy->free_chain.prev || !buddy->free_chain.next)
				break;
			/* buddy has to be same size */
			if (chunk->chunk_head.size != buddy->chunk_head.size)
				break;
			Assert((uintptr_t)chunk +
				   chunk->chunk_head.size == (uintptr_t)buddy);

			/* OK, merge */
			dlist_delete(&buddy->addr_chain);
			dlist_delete(&buddy->free_chain);
			chunk->chunk_head.size += chunk->chunk_head.size;
			HOSTMEM_CHUNK_MAGIC(chunk) = HOSTMEM_CHUNK_MAGIC_CODE;
		}
		else
		{
			/* this chunk should be merged with previous chunk */
			if (!dlist_has_prev(&chm_block->addr_chunks,
								&chunk->addr_chain))
				break;

			/* cannot merge with active chunk, of course */
			dnode = dlist_prev_node(&chm_block->addr_chunks,
									&chunk->addr_chain);
			buddy = dlist_container(cudaHostMemChunk, addr_chain, dnode);
			if (!buddy->free_chain.prev || !buddy->free_chain.next)
				break;
			/* buddy has to be same size */
			if (chunk->chunk_head.size != buddy->chunk_head.size)
				break;
			Assert((uintptr_t)chunk -
				   chunk->chunk_head.size == (uintptr_t)buddy);
			/* OK, merge */
			dlist_delete(&chunk->addr_chain);
			dlist_delete(&buddy->free_chain);
			memset(&buddy->free_chain, 0, sizeof(dlist_node));
			buddy->chunk_head.size += buddy->chunk_head.size;
			HOSTMEM_CHUNK_MAGIC(buddy) = HOSTMEM_CHUNK_MAGIC_CODE;
			chunk = buddy;
		}
	}
	/* OK, add chunk to free list */
	index = get_next_log2(chunk->chunk_head.size);
	Assert(index >= HOSTMEM_CHUNKSZ_MIN_BIT &&
		   index <= HOSTMEM_CHUNKSZ_MAX_BIT);
	dlist_push_head(&chm_head->free_chunks[index], &chunk->free_chain);
}

static void *
cudaHostMemRealloc(MemoryContext context, void *pointer, Size size)
{
	cudaHostMemChunk   *chm_chunk;
	Size				length;
	void			   *result;

	chm_chunk = (cudaHostMemChunk *)
		((char *)pointer - offsetof(cudaHostMemChunk, chunk_data));

	/* if newsize is still in margin, nothing to do */
	length = MAXALIGN(offsetof(cudaHostMemChunk, chunk_data) +
					  size +
					  sizeof(cl_uint));
	if (length <= chm_chunk->chunk_head.size)
	{
#ifdef MEMORY_CONTEXT_CHECKING
		chm_chunk->chunk_head.requested_size = size;
#endif
		HOSTMEM_CHUNK_MAGIC(chm_chunk) = HOSTMEM_CHUNK_MAGIC_CODE;
		return pointer;
	}
	/* elsewhere, alloc new chunk and copy old contents */
	result = cudaHostMemAlloc(context, size);
	length = (chm_chunk->chunk_head.size -
			  offsetof(cudaHostMemChunk, chunk_data));
	memcpy(result, pointer, length);
	/* release old one */
	cudaHostMemFree(context, pointer);

	return result;
}

static void
cudaHostMemInit(MemoryContext context)
{
	/* do nothing here */
}

static void
cudaHostMemReset(MemoryContext context)
{
	cudaHostMemHead	   *chm_head = (cudaHostMemHead *) context;
	cudaHostMemBlock   *chm_block;
	dlist_mutable_iter	miter;
	CUresult			rc;
	int					i;

	dlist_foreach_modify(miter, &chm_head->blocks)
	{
		chm_block = dlist_container(cudaHostMemBlock, chain, miter.cur);
		dlist_delete(&chm_block->chain);

		rc = cuMemFreeHost(chm_block);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemFreeHost: %s", errorText(rc));
	}
	Assert(dlist_is_empty(&chm_head->blocks));
	for (i=0; i <= HOSTMEM_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&chm_head->free_chunks[i]);
	chm_head->block_size_next = chm_head->block_size_init;
}

static void
cudaHostMemDelete(MemoryContext context)
{
	/*
	 * Unlike AllocSet, we don't have a keeper block. So, same as reset.
	 */
	cudaHostMemReset(context);
}

static Size
cudaHostMemGetChunkSpace(MemoryContext context, void *pointer)
{
	cudaHostMemChunk   *chm_chunk =
		(cudaHostMemChunk *)((char *)pointer -
							 offsetof(cudaHostMemChunk, chunk_head) -
							 STANDARDCHUNKHEADERSIZE);
	Assert(!chm_chunk->free_chain.prev &&
		   !chm_chunk->free_chain.next);
	return chm_chunk->chunk_head.size;
}

static bool
cudaHostMemIsEmpty(MemoryContext context)
{
	if (context->isReset)
		return true;
	return false;
}

static void
cudaHostMemStats(MemoryContext context, int level)
{
	cudaHostMemHead	   *chm_head = (cudaHostMemHead *) context;
	dlist_iter			iter1;
	dlist_iter			iter2;

	dlist_foreach(iter1, &chm_head->blocks)
	{
		cudaHostMemBlock   *chm_block;
		cudaHostMemChunk   *head_chunk;
		cudaHostMemChunk   *tail_chunk;

		chm_block = dlist_container(cudaHostMemBlock, chain, iter1.cur);
		Assert(!dlist_is_empty(&chm_block->addr_chunks));
		head_chunk = dlist_container(cudaHostMemChunk, addr_chain,
									 dlist_head_node(&chm_block->addr_chunks));
		tail_chunk = dlist_container(cudaHostMemChunk, addr_chain,
									 dlist_tail_node(&chm_block->addr_chunks));
		elog(INFO, "---- cuda host memory block [%p - %p] ----",
			 (char *)head_chunk,
			 (char *)tail_chunk + tail_chunk->chunk_head.size - 1);

		dlist_foreach(iter2, &chm_block->addr_chunks)
		{
			cudaHostMemChunk   *chm_chunk
				= dlist_container(cudaHostMemChunk, addr_chain, iter2.cur);
			elog(INFO, "%p - %p %s (size: %zu%s)",
				 (char *)chm_chunk,
				 (char *)chm_chunk + chm_chunk->chunk_head.size,
				 (!chm_chunk->free_chain.prev &&
				  !chm_chunk->free_chain.next) ? "active" : "free",
				 chm_chunk->chunk_head.size,
				 (!chm_chunk->free_chain.prev &&
				  !chm_chunk->free_chain.next &&
				  HOSTMEM_CHUNK_MAGIC(chm_chunk) != HOSTMEM_CHUNK_MAGIC_CODE
				  ? ", corrupted" : ""));
		}
	}
}

#ifdef MEMORY_CONTEXT_CHECKING
static void
cudaHostMemCheck(MemoryContext context)
{
	cudaHostMemHead	   *chm_head = (cudaHostMemHead *) context;
	dlist_iter			iter1;
	dlist_iter			iter2;

	dlist_foreach(iter1, &chm_head->blocks)
	{
		cudaHostMemBlock   *chm_block
			= dlist_container(cudaHostMemBlock, chain, iter1.cur);

		dlist_foreach(iter2, &chm_block->addr_chunks)
		{
			cudaHostMemChunk *chm_chunk
				= dlist_container(cudaHostMemChunk, addr_chain, iter2.cur);
			if (chm_chunk->free_chain.prev || chm_chunk->free_chain.next)
				continue;
			Assert(HOSTMEM_CHUNK_MAGIC(chm_chunk) == HOSTMEM_CHUNK_MAGIC_CODE);
		}
	}
}
#endif

/*
 * This is the virtual function table for AllocSet contexts.
 */
static MemoryContextMethods cudaHostMemMethods = {
	cudaHostMemAlloc,
	cudaHostMemFree,
	cudaHostMemRealloc,
	cudaHostMemInit,
	cudaHostMemReset,
	cudaHostMemDelete,
	cudaHostMemGetChunkSpace,
	cudaHostMemIsEmpty,
	cudaHostMemStats,
#ifdef MEMORY_CONTEXT_CHECKING
	cudaHostMemCheck,
#endif
};

/*
 * AllocSetContextCreate
 *		Create a new AllocSet context.
 *
 * parent: parent context, or NULL if top-level context
 * name: name of context (for debugging --- string will be copied)
 * minContextSize: minimum context size
 * initBlockSize: initial allocation block size
 * maxBlockSize: maximum allocation block size
 */
MemoryContext
HostPinMemContextCreate(MemoryContext parent,
						const char *name,
						CUcontext cuda_context,
						Size block_size_init,
						Size block_size_max)
{
	cudaHostMemHead	   *chm_head;

	/* Do the type-independent part of context creation */
	chm_head = (cudaHostMemHead *)
		MemoryContextCreate(T_AllocSetContext,
							sizeof(cudaHostMemHead),
							&cudaHostMemMethods,
							parent,
							name);
	/* save the reference to cuda_context */
	chm_head->cuda_context = cuda_context;

	/*
	 * Make sure alloc parameters are reasonable
	 */
	if (block_size_init < 10 * pgstrom_chunk_size())
		block_size_init = 10 * pgstrom_chunk_size();
	block_size_init = MAXALIGN(block_size_init);
	block_size_init = (1UL << get_next_log2(block_size_init));

	if (block_size_max < block_size_init)
		block_size_max = block_size_init;
	block_size_max = MAXALIGN(block_size_max);
	block_size_max = (1UL << get_next_log2(block_size_max));

	Assert(AllocHugeSizeIsValid(block_size_max)); /* must be safe to double */
	chm_head->block_size_init = block_size_init;
	chm_head->block_size_next = block_size_init;
	chm_head->block_size_max  = block_size_max;

	return &chm_head->header;
}
