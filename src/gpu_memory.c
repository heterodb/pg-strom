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
#include "utils/pg_crc.h"
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
	cl_int			refcnt;		/* can be referenced by multiple-gcontext */
} GpuMemChunk;

#define GPUMEMCHUNK_IS_FREE(chunk)								   \
	((chunk)->chain.prev != NULL && (chunk)->chain.next != NULL && \
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&				   \
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT &&				   \
	 (chunk)->refcnt == 0)
#define GPUMEMCHUNK_IS_ACTIVE(chunk)							   \
	((chunk)->chain.prev == NULL && (chunk)->chain.next == NULL && \
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&				   \
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT &&				   \
	 (chunk)->refcnt > 0)

typedef struct
{
	dlist_node		segment_chain;		/* link to active/inactive list */
	CUdeviceptr		m_segment;
	cl_int			num_total_chunks;	/* length of gpumem_chunks[] */
	cl_bool			is_managed_memory;	/* true, if managed memory segment */
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
static dlist_head		gpumem_segment_list;
static cl_int			gpumem_segment_count = 0;

typedef struct
{
	dlist_node		chain;
	CUdeviceptr		m_deviceptr;
	cl_int			refcnt;
} GpuMemLargeChunk;

#define GPUMEM_LARGECHUNK_NSLOTS		73
static slock_t			gpumem_largechunk_lock;
static dlist_head		gpumem_largechunk_slot[GPUMEM_LARGECHUNK_NSLOTS];

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
	gm_chunk->refcnt = 1;

	return result;
}

/*
 * GpuMemSegmentFree
 */
static CUresult
GpuMemSegmentFree(GpuMemSegment *gm_seg, CUdeviceptr deviceptr)
{
	GpuMemChunk	   *gm_chunk;
	GpuMemChunk	   *gm_buddy;
	int				index, shift;

	if ((deviceptr & (GPUMEM_CHUNKSZ_MIN - 1)) != 0 ||
		(deviceptr < gm_seg->m_segment) ||
		(deviceptr >= (gm_seg->m_segment +
					   GPUMEM_CHUNKSZ_MIN * gm_seg->num_total_chunks)))
		return CUDA_ERROR_INVALID_VALUE;

	SpinLockAcquire(&gm_seg->lock);
	index = (deviceptr - gm_seg->m_segment) >> GPUMEM_CHUNKSZ_MIN_BIT;
	gm_chunk = &gm_seg->gpumem_chunks[index];
	Assert(GPUMEMCHUNK_IS_ACTIVE(gm_chunk));
	/*
	 * In case when GPU memory chunk is shared by multiple GpuContext,
	 * it is too early to release this chunk.
	 */
	if (--gm_chunk->refcnt > 0)
	{
		SpinLockRelease(&gm_seg->lock);
		return CUDA_SUCCESS;
	}

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
	dlist_push_head(&gm_seg->free_chunks[gm_chunk->mclass],
					&gm_chunk->chain);
	gm_seg->num_active_chunks--;
	Assert(gm_seg->num_active_chunks >= 0);
	SpinLockRelease(&gm_seg->lock);
	return CUDA_SUCCESS;
}


/*
 * gpuMemAllocCommonRaw
 */
static inline CUresult
gpuMemAllocCommonRaw(bool is_managed_memory,
					 GpuContext *gcontext,
					 CUdeviceptr *p_deviceptr,
					 size_t bytesize)
{
	GpuMemLargeChunk *lchunk;
	CUdeviceptr		m_deviceptr;
	CUresult		rc, __rc;
	char		   *extra;
	pg_crc32		crc;
	int				index;

	lchunk = calloc(1, sizeof(GpuMemLargeChunk));
	if (!lchunk)
		return CUDA_ERROR_OUT_OF_MEMORY;
	Assert(PointerIsAligned(lchunk, int));

	/* device memory allocation */
	if (!is_managed_memory)
	{
		rc = cuMemAlloc(&m_deviceptr, bytesize);
		if (rc != CUDA_SUCCESS)
			goto error_1;
	}
	else
	{
		rc = cuMemAllocManaged(&m_deviceptr, bytesize, CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
			goto error_1;

		rc = cuMemAdvise(m_deviceptr, bytesize,
						 CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
						 gpuserv_cuda_device);
		if (rc != CUDA_SUCCESS)
			goto error_2;
		rc = cuMemAdvise(m_deviceptr, bytesize,
						 CU_MEM_ADVISE_SET_ACCESSED_BY,
						 gpuserv_cuda_device);
		if (rc != CUDA_SUCCESS)
			goto error_2;
	}
	lchunk->m_deviceptr = m_deviceptr;
	lchunk->refcnt = 1;

	/* least 1bit is a flag to indicate GpuMemLargeChunk */
	extra = (char *)(((uintptr_t)lchunk) | 1UL);
	if (!trackGpuMem(gcontext, m_deviceptr, extra))
	{
		rc = CUDA_ERROR_OUT_OF_MEMORY;
		goto error_2;
	}
	/* track this large chunk */
	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &m_deviceptr, sizeof(CUdeviceptr));
	FIN_LEGACY_CRC32(crc);
	index = crc % GPUMEM_LARGECHUNK_NSLOTS;

	SpinLockAcquire(&gpumem_largechunk_lock);
	dlist_push_head(&gpumem_largechunk_slot[index], &lchunk->chain);
	SpinLockRelease(&gpumem_largechunk_lock);

	*p_deviceptr = m_deviceptr;

	return CUDA_SUCCESS;

error_2:
	__rc = cuMemFree(m_deviceptr);
	if (__rc != CUDA_SUCCESS)
		wnotice("failed on cuMemFree: %s", errorText(rc));
error_1:
	free(lchunk);
	return rc;
}

/*
 * gpuMemAllocRaw - simple wrapper for cuMemAlloc
 */
CUresult
gpuMemAllocRaw(GpuContext *gcontext,
			   CUdeviceptr *p_deviceptr,
			   size_t bytesize)
{
	return gpuMemAllocCommonRaw(false, gcontext, p_deviceptr, bytesize);
}

/*
 * gpuMemAllocManagedRaw - simple wrapper for cuMemAllocManaged
 */
CUresult
gpuMemAllocManagedRaw(GpuContext *gcontext,
					  CUdeviceptr *p_deviceptr,
					  size_t bytesize)
{
	return gpuMemAllocCommonRaw(true, gcontext, p_deviceptr, bytesize);
}

/*
 * gpuMemAllocCommon
 */
static inline CUresult
gpuMemAllocCommon(cl_bool is_managed_memory,
				  GpuContext *gcontext,
				  CUdeviceptr *p_deviceptr,
				  size_t bytesize)
{
	bool			has_exclusive_lock = false;
	GpuMemSegment  *gm_seg;
	GpuMemChunk	   *gm_chunk;
	dlist_iter		iter;
	int				i, mclass;
	int				nchunks;
	CUdeviceptr		m_segment;
	CUdeviceptr		m_deviceptr;
	CUresult		rc, __rc;

	Assert(IsGpuServerProcess());
	mclass = get_next_log2(bytesize);
	if (mclass < GPUMEM_CHUNKSZ_MIN_BIT)
		mclass = GPUMEM_CHUNKSZ_MIN_BIT;
	else if (mclass > GPUMEM_SEGMENTSZ_BIT)
		return gpuMemAllocCommonRaw(is_managed_memory,
									gcontext,
									p_deviceptr,
									bytesize);

	pthreadRWLockReadLock(&gpumem_segment_rwlock);
retry:
	/* lookup an active segment with free space */
	dlist_foreach(iter, &gpumem_segment_list)
	{
		gm_seg = dlist_container(GpuMemSegment, segment_chain, iter.cur);

		if (is_managed_memory != gm_seg->is_managed_memory)
			continue;

		SpinLockAcquire(&gm_seg->lock);
		for (i=mclass; i <= GPUMEM_SEGMENTSZ_BIT; i++)
		{
			if (dlist_is_empty(&gm_seg->free_chunks[i]))
				continue;

			m_deviceptr = GpuMemSegmentAlloc(gcontext, gm_seg, mclass);
			if (m_deviceptr)
				gm_seg->num_active_chunks++;
			SpinLockRelease(&gm_seg->lock);
			pthreadRWLockUnlock(&gpumem_segment_rwlock);
			if (m_deviceptr == 0UL)
				return CUDA_ERROR_OUT_OF_MEMORY;
			*p_deviceptr = m_deviceptr;
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
		pthreadRWLockUnlock(&gpumem_segment_rwlock);
		pthreadRWLockWriteLock(&gpumem_segment_rwlock);
		has_exclusive_lock = true;
		goto retry;
	}

	/* here is no available segment, so create a new one */
	nchunks = GPUMEM_SEGMENTSZ / GPUMEM_CHUNKSZ_MIN;
	gm_seg = calloc(1, offsetof(GpuMemSegment, gpumem_chunks[nchunks]));
	if (!gm_seg)
	{
		rc = CUDA_ERROR_OUT_OF_MEMORY;
		goto error0;
	}

	/* device memory allocation */
	if (!is_managed_memory)
	{
		rc = cuMemAlloc(&m_segment, GPUMEM_SEGMENTSZ);
		if (rc != CUDA_SUCCESS)
			goto error1;
	}
	else
	{
		rc = cuMemAllocManaged(&m_segment, GPUMEM_SEGMENTSZ,
							   CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
			goto error1;

		rc = cuMemAdvise(m_segment, GPUMEM_SEGMENTSZ,
						 CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
						 gpuserv_cuda_device);
		if (rc != CUDA_SUCCESS)
			goto error2;
		rc = cuMemAdvise(m_segment, GPUMEM_SEGMENTSZ,
						 CU_MEM_ADVISE_SET_ACCESSED_BY,
						 gpuserv_cuda_device);
		if (rc != CUDA_SUCCESS)
			goto error2;
	}
	/*
	 * OK, create a new device memory segment
	 */
	gm_seg->m_segment			= m_segment;
	gm_seg->num_total_chunks	= nchunks;
	gm_seg->is_managed_memory	= is_managed_memory;
	SpinLockInit(&gm_seg->lock);
	gm_seg->num_active_chunks = 0;
	for (i=0; i <= GPUMEM_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&gm_seg->free_chunks[i]);
	gm_chunk = &gm_seg->gpumem_chunks[0];
	gm_chunk->mclass = GPUMEM_SEGMENTSZ_BIT;
	dlist_push_head(&gm_seg->free_chunks[GPUMEM_SEGMENTSZ_BIT],
					&gm_chunk->chain);
	dlist_push_head(&gpumem_segment_list,
					&gm_seg->segment_chain);
	gpumem_segment_count++;
	goto retry;

error2:
	if ((__rc = cuMemFree(m_segment)) != CUDA_SUCCESS)
		wfatal("failed on cuMemFree: %s", errorText(__rc));
error1:
	free(gm_seg);
error0:
	pthreadRWLockUnlock(&gpumem_segment_rwlock);
	return rc;
}

/*
 * gpuMemAlloc
 */
CUresult
gpuMemAlloc(GpuContext *gcontext,
			CUdeviceptr *p_deviceptr,
			size_t bytesize)
{
	return gpuMemAllocCommon(false, gcontext, p_deviceptr, bytesize);
}

/*
 * gpuMemAllocManaged
 */
CUresult
gpuMemAllocManaged(GpuContext *gcontext,
				   CUdeviceptr *p_deviceptr,
				   size_t bytesize)
{
	return gpuMemAllocCommon(true, gcontext, p_deviceptr, bytesize);
}

/*
 * gpuMemFreeExtra - to be called by only resource cleanup handler
 */
CUresult
gpuMemFreeExtra(void *extra, CUdeviceptr deviceptr)
{
	GpuMemLargeChunk *lchunk;
	CUresult	rc = CUDA_SUCCESS;

	if ((((uintptr_t)extra) & 1UL) == 0)
		return GpuMemSegmentFree((GpuMemSegment *)extra, deviceptr);

	lchunk = (GpuMemLargeChunk *)(((uintptr_t)extra) & ~1UL);
	SpinLockAcquire(&gpumem_largechunk_lock);
	Assert(lchunk->m_deviceptr == deviceptr);
	Assert(lchunk->refcnt > 0);
	if (--lchunk->refcnt == 0)
	{
		dlist_delete(&lchunk->chain);
		SpinLockRelease(&gpumem_largechunk_lock);
		free(lchunk);

		rc = cuMemFree(deviceptr);
	}
	else
		SpinLockRelease(&gpumem_largechunk_lock);
	return rc;

}

/*
 * gpuMemFree
 */
CUresult
gpuMemFree(GpuContext *gcontext, CUdeviceptr deviceptr)
{
	/* If called on PostgreSQL backend, send a request to release */
	if (!IsGpuServerProcess())
	{
		gpuservSendGpuMemFree(gcontext, deviceptr);
		return CUDA_SUCCESS;
	}
	
	/*
	 * Pulls either GpuMemSegment or GpuMemLargeChunk using resource tracker.
	 */
	return gpuMemFreeExtra(untrackGpuMem(gcontext, deviceptr),
						   deviceptr);
}

/*
 * gpuMemRetain - get reference to device memory chunk acquired by different
 *                context
 */
CUresult
gpuMemRetain(GpuContext *gcontext, CUdeviceptr deviceptr)
{
	GpuMemSegment  *gm_seg;
	GpuMemChunk	   *gm_chunk;
	dlist_iter		iter;
	pg_crc32		crc;
	int				index;
	char		   *extra;

	/*
	 * Lookup small chunks in segment first
	 */
	pthreadRWLockReadLock(&gpumem_segment_rwlock);
	dlist_foreach(iter, &gpumem_segment_list)
	{
		gm_seg = dlist_container(GpuMemSegment, segment_chain, iter.cur);
		if (deviceptr < gm_seg->m_segment ||
			deviceptr >= (gm_seg->m_segment +
						  GPUMEM_CHUNKSZ_MIN * gm_seg->num_total_chunks))
			continue;

		/* OK, found */
		if (!trackGpuMem(gcontext, deviceptr, gm_seg))
		{
			pthreadRWLockUnlock(&gpumem_segment_rwlock);
			return CUDA_ERROR_OUT_OF_MEMORY;
		}

		SpinLockAcquire(&gm_seg->lock);
		index = (deviceptr - gm_seg->m_segment) >> GPUMEM_CHUNKSZ_MIN_BIT;
		gm_chunk = &gm_seg->gpumem_chunks[index];
		Assert(GPUMEMCHUNK_IS_ACTIVE(gm_chunk));
		gm_chunk->refcnt++;
		SpinLockRelease(&gm_seg->lock);
		pthreadRWLockUnlock(&gpumem_segment_rwlock);

		return CUDA_SUCCESS;
	}
	pthreadRWLockUnlock(&gpumem_segment_rwlock);

	/*
	 * Elsewhere, deviceptr might be a large chunk
	 */
	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &deviceptr, sizeof(CUdeviceptr));
	FIN_LEGACY_CRC32(crc);
	index = crc % GPUMEM_LARGECHUNK_NSLOTS;

	SpinLockAcquire(&gpumem_largechunk_lock);
	dlist_foreach(iter, &gpumem_largechunk_slot[index])
	{
		GpuMemLargeChunk *lchunk
			= dlist_container(GpuMemLargeChunk, chain, iter.cur);

		/* OK, found */
		if (lchunk->m_deviceptr == deviceptr)
		{
			extra = (char *)(((uintptr_t)(lchunk)) | 1UL);
			if (!trackGpuMem(gcontext, deviceptr, extra))
			{
				SpinLockRelease(&gpumem_largechunk_lock);
				return CUDA_ERROR_OUT_OF_MEMORY;
			}
			lchunk->refcnt++;
			SpinLockRelease(&gpumem_largechunk_lock);

			return CUDA_SUCCESS;
		}
	}
	SpinLockRelease(&gpumem_largechunk_lock);

	return CUDA_ERROR_INVALID_VALUE;
}

/*
 * gpuMemReclaim - reclaim inactive segment when GPU is in idle
 */
void
gpuMemReclaim(void)
{
	dlist_iter		iter;
	CUresult		rc;

	Assert(IsGpuServerProcess());

	if (!pthreadRWLockWriteTryLock(&gpumem_segment_rwlock))
		return;		/* someone actively works, no need to reclaim now */
	if (gpumem_segment_count > 1)
	{
		dlist_reverse_foreach(iter, &gpumem_segment_list)
		{
			GpuMemSegment  *gm_seg = dlist_container(GpuMemSegment,
													 segment_chain,
													 iter.cur);
			/*
			 * NOTE: No concurrent task shall appear under the exclusive
			 * lock on the gpumem_segment_rwlock. So, no need to acquire
			 * gm_seg->lock here.
			 */
			if (gm_seg->num_active_chunks == 0)
			{
				dlist_delete(&gm_seg->segment_chain);
				rc = cuMemFree(gm_seg->m_segment);
				if (rc != CUDA_SUCCESS)
					wfatal("failed on cuMemFree: %s", errorText(rc));
				free(gm_seg);
				gpumem_segment_count--;
				break;
			}
		}
	}
	pthreadRWLockUnlock(&gpumem_segment_rwlock);
}

#if 0
/*
 * pgstrom_startup_gpu_memory
 */
static void
pgstrom_startup_gpu_memory(void)
{

}
#endif

/*
 * pgstrom_init_gpu_memory
 */
void
pgstrom_init_gpu_memory(void)
{
	pthreadRWLockInit(&gpumem_segment_rwlock);
	dlist_init(&gpumem_segment_list);
}
