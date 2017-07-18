/*
 * gpu_memory.c
 *
 * Routines to manage GPU device memory.
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "lib/ilist.h"
#include "pg_strom.h"
#include <pthread.h>

#define GPUMEM_CHUNKSZ_MAX_BIT		34		/* 16GB */
#define GPUMEM_CHUNKSZ_MIN_BIT		12		/* 4KB */
#define GPUMEM_CHUNKSZ_MAX			(1UL << GPUMEM_CHUNKSZ_MAX_BIT)
#define GPUMEM_CHUNKSZ_MIN			(1UL << GPUMEM_CHUNKSZ_MIN_BIT)

#define GPUMEM_SEGMENTSZ_BIT		30		/* 1GB */
#define GPUMEM_SEGMENTSZ			(1UL << GPUMEM_SEGMENTSZ_BIT)

typedef struct
{
	dlist_node		chain;		/* link to free_chunks[],
								 * or zero if active chunk */
	cl_int			mclass;		/* class of memory chunk */
} GpuMemChunk;

#define GPUMEMCHUNK_IS_FREE(chunk)								   \
	((chunk)->chain.prev != NULL && (chunk)->chain.next != NULL && \
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&				   \
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT)
#define GPUMEMCHUNK_IS_ACTIVE(chunk)							   \
	((chunk)->chain.prev == NULL && (chunk)->chain.next == NULL && \
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&				   \
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT)

typedef struct
{
	dlist_node		segment_chain;		/* link to active/inactive list */
	CUdeviceptr		m_segment;
	cl_int			num_total_chunks;	/* length of gpumem_chunks[] */
	/* extra attributes for i/o mapped memory */
	CUipcMemHandle	cuda_mhandle;
	unsigned long	iomap_handle;
	uint32_t		gpu_page_sz;
	uint32_t		gpu_npages;
	/* free_chunks[] is protected by the lock */
	slock_t			lock;
	cl_int			num_active_chunks;	/* # of active chunks */
	dlist_head		free_chunks[GPUMEM_CHUNKSZ_MAX_BIT + 1];
	GpuMemChunk		gpumem_chunks[FLEXIBLE_ARRAY_MEMBER];
} GpuMemSegment;

static pthread_rwlock_t gpumem_segment_rwlock;
static dlist_head	gpumem_active_segment_list;
static dlist_head	gpumem_inactive_segment_list;

/*
 * gpuMemMaxAllocSize
 */
Size
gpuMemMaxAllocSize(void)
{
	return (1UL << 31);		/* 2GB at this moment */
}

/*
 * GpuMemSegmentSplit - must be called under gm_seg->lock
 */
static bool
GpuMemSegmentSplit(GpuMemSegment *gm_seg, int mclass)
{
	GpuMemChunk	   *gm_chunk1;
	GpuMemChunk	   *gm_chunk2;
	dlist_node	   *dnode;
	int				offset;

	if (mclass > GPUMEM_CHUNKSZ_MAX_BIT)
		return false;
	Assert(mclass > GPUMEM_CHUNKSZ_MIN_BIT);
	if (dlist_is_empty(&gm_seg->free_chunks[mclass]))
	{
		if (!GpuMemSegmentSplit(gm_seg, mclass + 1))
			return false;
	}
	Assert(!dlist_is_empty(&gm_seg->free_chunks[mclass]));
	offset = 1UL << (mclass - 1 - GPUMEM_CHUNKSZ_MIN_BIT);
	dnode = dlist_pop_head_node(&gm_seg->free_chunks[mclass]);
	gm_chunk1 = dlist_container(GpuMemChunk, chain, dnode);
	gm_chunk2 = gm_chunk1 + offset;
	Assert(GPUMEMCHUNK_IS_FREE(gm_chunk1));
	Assert(gm_chunk2->mclass == 0);
	gm_chunk1->mclass = mclass - 1;
	gm_chunk2->mclass = mclass - 1;

	dlist_push_tail(&gm_seg->free_chunks[mclass - 1],
					&gm_chunk1->chain);
	dlist_push_tail(&gm_seg->free_chunks[mclass - 1],
					&gm_chunk2->chain);
	return true;
}

/*
 * GpuMemSegmentAlloc - caller must have a lock on gm_seg->lock
 */
static CUdeviceptr
GpuMemSegmentAlloc(GpuContext *gcontext,
				   GpuMemSegment *gm_seg, int mclass)
{
	GpuMemChunk	   *gm_chunk;
	CUdeviceptr		result;
	dlist_node	   *dnode;
	cl_int			index;

	if (dlist_is_empty(&gm_seg->free_chunks[mclass]))
	{
		/* split larger chunk into two */
		if (!GpuMemSegmentSplit(gm_seg, mclass + 1))
			return (CUdeviceptr)(0UL);
	}
	Assert(!dlist_is_empty(&gm_seg->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&gm_seg->free_chunks[mclass]);
	gm_chunk = dlist_container(GpuMemChunk, chain, dnode);
	Assert(GPUMEMCHUNK_IS_FREE(gm_chunk));
	index = gm_chunk - gm_seg->gpumem_chunks;
	Assert(index >= 0 && index < gm_seg->num_total_chunks);
	result = gm_seg->m_segment + GPUMEM_CHUNKSZ_MIN * index;

	/* track this device memory by GpuContext */
	if (!trackGpuMem(gcontext, result, gm_seg))
	{
		dlist_push_head(&gm_seg->free_chunks[mclass],
						&gm_chunk->chain);
		return (CUdeviceptr)(0UL);
	}
	memset(&gm_chunk->chain, 0, sizeof(dlist_node));
	Assert(gm_chunk->mclass == mclass);

	return result;
}

/*
 * gpuMemAlloc
 */
CUresult
gpuMemAlloc(GpuContext *gcontext, CUdeviceptr *p_devptr, size_t bytesize)
{
	bool			has_exclusive_lock = false;
	GpuMemSegment  *gm_seg;
	GpuMemChunk	   *gm_chunk;
	dlist_iter		iter;
	dlist_node	   *dnode;
	int				i, mclass;
	int				nchunks;
	CUdeviceptr		m_segment;
	CUdeviceptr		m_result;
	CUresult		rc;

	Assert(IsGpuServerProcess());
	mclass = get_next_log2(bytesize);
	if (mclass < GPUMEM_CHUNKSZ_MIN_BIT)
		mclass = GPUMEM_CHUNKSZ_MIN_BIT;
	else if (mclass > GPUMEM_SEGMENTSZ_BIT)
	{
		/*
		 * memory block larger than segment shall be acquired / released
		 * individually.
		 */
		rc = cuMemAllocManaged(&m_result, bytesize, CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
			return rc;
		if (!trackGpuMem(gcontext, m_result, NULL))
		{
			rc = cuMemFree(m_result);
			if (rc != CUDA_SUCCESS)
				wnotice("failed on cuMemFree: %s", errorText(rc));
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
		*p_devptr = m_result;
		return CUDA_SUCCESS;
	}

	if ((errno = pthread_rwlock_rdlock(&gpumem_segment_rwlock)) < 0)
		wfatal("failed on pthread_rwlock_rdlock: %m");
retry:
	/* lookup an active segment with free space */
	dlist_foreach(iter, &gpumem_active_segment_list)
	{
		gm_seg = dlist_container(GpuMemSegment, segment_chain, iter.cur);

		SpinLockAcquire(&gm_seg->lock);
		for (i=mclass; i <= GPUMEM_SEGMENTSZ_BIT; i++)
		{
			if (dlist_is_empty(&gm_seg->free_chunks[i]))
				continue;

			m_result = GpuMemSegmentAlloc(gcontext, gm_seg, mclass);
			if (m_result)
				gm_seg->num_active_chunks++;
			SpinLockRelease(&gm_seg->lock);
			if ((errno = pthread_rwlock_unlock(&gpumem_segment_rwlock)) < 0)
				wfatal("failed on pthread_rwlock_unlock: %m");
			if (m_result == 0UL)
				return CUDA_ERROR_OUT_OF_MEMORY;
			*p_devptr = m_result;
			return CUDA_SUCCESS;
		}
		SpinLockRelease(&gm_seg->lock);
	}

	/*
	 * No space left on the active segment. Try inactive segment or
	 * create a new segment on demand.
	 */
	if (!has_exclusive_lock)
	{
		if ((errno = pthread_rwlock_unlock(&gpumem_segment_rwlock)) < 0)
			wfatal("failed on pthread_rwlock_unlock: %m");
		if ((errno = pthread_rwlock_wrlock(&gpumem_segment_rwlock)) < 0)
			wfatal("failed on pthread_rwlock_wrlock: %m");
		has_exclusive_lock = true;
		goto retry;
	}

	/* try to pick up an inactive segment, if any */
	if (!dlist_is_empty(&gpumem_inactive_segment_list))
	{
		dnode = dlist_pop_head_node(&gpumem_inactive_segment_list);
		dlist_push_head(&gpumem_active_segment_list, dnode);
		goto retry;
	}

	/* here is no available segment, so create a new one */
	nchunks = GPUMEM_SEGMENTSZ / GPUMEM_CHUNKSZ_MIN;
	gm_seg = calloc(1, offsetof(GpuMemSegment, gpumem_chunks[nchunks]));
	if (!gm_seg)
	{
		if ((errno = pthread_rwlock_unlock(&gpumem_segment_rwlock)) < 0)
			wfatal("failed on pthread_rwlock_unlock: %m");
		return CUDA_ERROR_OUT_OF_MEMORY;
	}

	rc = cuMemAllocManaged(&m_segment,
						   GPUMEM_SEGMENTSZ,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		if ((errno = pthread_rwlock_unlock(&gpumem_segment_rwlock)) < 0)
			wfatal("failed on pthread_rwlock_unlock: %m");
		free(gm_seg);
		return rc;
	}
	gm_seg->m_segment			= m_segment;
	gm_seg->num_total_chunks	= nchunks;
	SpinLockInit(&gm_seg->lock);
	gm_seg->num_active_chunks = 0;
	for (i=0; i <= GPUMEM_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&gm_seg->free_chunks[i]);
	gm_chunk = &gm_seg->gpumem_chunks[0];
	gm_chunk->mclass = GPUMEM_SEGMENTSZ_BIT;
	dlist_push_head(&gm_seg->free_chunks[GPUMEM_SEGMENTSZ_BIT],
					&gm_chunk->chain);
	dlist_push_head(&gpumem_active_segment_list,
					&gm_seg->segment_chain);
	goto retry;
}

/*
 * gpuMemAllocManaged
 */
CUresult
gpuMemAllocManaged(GpuContext *gcontext,
				   CUdeviceptr *p_devptr, size_t bytesize, int flags)
{
	CUdeviceptr	m_result;
	CUresult	rc;

	rc = cuMemAllocManaged(&m_result, bytesize, flags);
	if (rc != CUDA_SUCCESS)
		return rc;
	if (!trackGpuMem(gcontext, m_result, NULL))
	{
		rc = cuMemFree(m_result);
		if (rc != CUDA_SUCCESS)
			wnotice("failed on cuMemFree: %s", errorText(rc));
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	*p_devptr = m_result;
	return CUDA_SUCCESS;
}

/*
 * gpuMemFree
 */
CUresult
gpuMemFree(GpuContext *gcontext, CUdeviceptr devptr)
{
	GpuMemSegment  *gm_seg;
	GpuMemChunk	   *gm_chunk;
	GpuMemChunk	   *gm_buddy;
	int				index, shift;

	/* pull GpuMemSegment from resource tracker */
	gm_seg = untrackGpuMem(gcontext, devptr);
	if (!gm_seg)
		return cuMemFree(devptr);

	Assert((devptr & (GPUMEM_CHUNKSZ_MIN - 1)) == 0);
	index = (devptr - gm_seg->m_segment) >> GPUMEM_CHUNKSZ_MIN_BIT;

	SpinLockAcquire(&gm_seg->lock);
	gm_chunk = &gm_seg->gpumem_chunks[index];
	Assert(GPUMEMCHUNK_IS_ACTIVE(gm_chunk));
	/*
	 * Try to merge with the neighbor chunks
	 */
	while (gm_chunk->mclass < GPUMEM_CHUNKSZ_MAX_BIT)
	{
		index = gm_chunk - gm_seg->gpumem_chunks;
		shift = 1UL << (gm_chunk->mclass - GPUMEM_CHUNKSZ_MIN_BIT);
		Assert((index & (shift - 1)) == 0);
		if ((index & shift) == 0)
		{
			/* try to merge with next */
			gm_buddy = &gm_seg->gpumem_chunks[index + shift];
			if (gm_buddy->chain.prev != NULL &&
				gm_buddy->chain.next != NULL &&
				gm_buddy->mclass == gm_chunk->mclass)
			{
				/* OK, let's merge */
				dlist_delete(&gm_buddy->chain);
				memset(gm_buddy, 0, sizeof(GpuMemChunk));
				gm_chunk->mclass++;
			}
			else
				break;	/* give up to merge chunks any more */
		}
		else
		{
			/* try to merge with prev */
			gm_buddy = &gm_seg->gpumem_chunks[index - shift];
			if (gm_buddy->chain.prev != NULL &&
				gm_buddy->chain.next != NULL &&
				gm_buddy->mclass == gm_chunk->mclass)
			{
				/* OK, let's merge */
				dlist_delete(&gm_buddy->chain);
				memset(gm_buddy, 0, sizeof(GpuMemChunk));
				gm_buddy->mclass = gm_chunk->mclass + 1;
				memset(gm_chunk, 0, sizeof(GpuMemChunk));
				gm_chunk = gm_buddy;
			}
			else
				break;	/* give up to merge chunks any more */
		}
	}
	/* back to the free list again */
	Assert(GPUMEMCHUNK_IS_ACTIVE(gm_chunk));
	dlist_push_head(&gm_seg->free_chunks[gm_chunk->mclass],
					&gm_chunk->chain);
	gm_seg->num_active_chunks--;
	Assert(gm_seg->num_active_chunks >= 0);
	SpinLockRelease(&gm_seg->lock);
	return CUDA_SUCCESS;
}

/*
 * gpuMemReclaim - reclaim inactive segment when GPU is in idle
 */
void
gpuMemReclaim(void)
{
	Assert(IsGpuServerProcess());

}

/*
 * pgstrom_startup_gpu_memory
 */
static void
pgstrom_startup_gpu_memory(void)
{

}

/*
 * pgstrom_init_gpu_memory
 */
void
pgstrom_init_gpu_memory(void)
{
	if ((errno = pthread_rwlock_init(&gpumem_segment_rwlock, NULL)) < 0)
		elog(ERROR, "failed on pthread_rwlock_init: %m");
	dlist_init(&gpumem_active_segment_list);
	dlist_init(&gpumem_inactive_segment_list);
}
