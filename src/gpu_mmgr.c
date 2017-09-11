/*
 * gpu_mmgr.c
 *
 * Routines to manage GPU device memory
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
#include "commands/tablespace.h"
#include "postmaster/bgworker.h"
#include "storage/bufmgr.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "utils/guc.h"
#include "utils/inval.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "pg_strom.h"
#include <sys/ioctl.h>

#define GPUMEM_CHUNKSZ_MAX_BIT		34		/* 16GB */
#define GPUMEM_CHUNKSZ_MIN_BIT		16		/* 64KB */
#define GPUMEM_CHUNKSZ_MAX			(1UL << GPUMEM_CHUNKSZ_MAX_BIT)
#define GPUMEM_CHUNKSZ_MIN			(1UL << GPUMEM_CHUNKSZ_MIN_BIT)

typedef enum
{
	GpuMemKind__NormalMemory	= (1 << 0),
	GpuMemKind__ManagedMemory	= (1 << 1),
	GpuMemKind__IOMapMemory		= (1 << 2),
	GpuMemKind__HostMemory		= (1 << 3),
} GpuMemKind;

typedef struct
{
	dlist_node		chain;
	cl_int			mclass;
	cl_int			refcnt;
} GpuMemChunk;

#define GPUMEMCHUNK_IS_FREE(chunk)					\
	((chunk)->chain.prev != NULL &&					\
	 (chunk)->chain.next != NULL &&					\
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&	\
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT &&	\
	 (chunk)->refcnt == 0)
#define GPUMEMCHUNK_IS_ACTIVE(chunk)				 \
	((chunk)->chain.prev == NULL &&					 \
	 (chunk)->chain.next == NULL &&					 \
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&	 \
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT &&	 \
	 (chunk)->refcnt > 0)

/* shared structure */
typedef struct
{
	cl_uint			segment_id;	/* index of the segment (never changed) */
	dlist_node		chain;		/* link to segment list */
	GpuMemKind		gm_kind;
	pg_atomic_uint32 mapcount;	/* # of GpuContext that maps this segment */
	CUdeviceptr		m_segment;	/* Device pointer for GPU Mmgr */
	CUipcMemHandle	m_handle;	/* IPC handler of the device memory */
	unsigned long	iomap_handle; /* only if GpuMemKind__IOMapMemory */
	slock_t			lock;		/* protection of free_chunks[] */
	dlist_head		free_chunks[GPUMEM_CHUNKSZ_MAX_BIT + 1];
	GpuMemChunk		gm_chunks[FLEXIBLE_ARRAY_MEMBER];
} GpuMemSegment;

/* shared structure (per device) */
typedef struct GpuMemDevice
{
	/* interaction backend <-> mmgr */
	pthread_mutex_t	mutex;
	pthread_cond_t	cond;
	cl_uint			alloc_request;			/* in: bitmap of GpuMemKind */
	cl_uint			revision;				/* out: revision of the retcode */
	CUresult		status_alloc_normal;	/* out: mmgr -> backend */
	CUresult		status_alloc_iomap;		/* out: mmgr -> backend */
	Latch		   *serverLatch;
	/* management of device memory segment */
	pthread_rwlock_t rwlock;
	dlist_head		normal_segment_list;
	dlist_head		iomap_segment_list;
} GpuMemDevice;

/* shared structure (system global) */
typedef struct GpuMemSystemHead
{
	slock_t			lock;		/* protection of free_segment_list */
	dlist_head		free_segment_list;
	GpuMemDevice	gm_dev_array[FLEXIBLE_ARRAY_MEMBER];
} GpuMemSystemHead;

/* per-context structure */
struct GpuMemSegMap
{
	GpuMemSegment  *gm_seg;		/* reference to the shared portion */
	dlist_node		chain;		/* link to local_xxx_segment_list */
	CUdeviceptr		m_segment;	/* device pointer */
};
typedef struct GpuMemSegMap		GpuMemSegMap;

/* static variables */
static shmem_startup_hook_type shmem_startup_next = NULL;
static GpuMemSystemHead *gm_shead = NULL;
static int			gpu_memory_segment_size_kb;	/* GUC */
static int			num_gpu_memory_segments;	/* GUC */
static int			num_gpu_iomap_segments;
static int			iomap_gpu_memory_size_kb;	/* GUC */
static bool			debug_force_nvme_strom;		/* GUC */
static bool			nvme_strom_enabled;			/* GUC */
static long			nvme_strom_threshold;

static bool			gpu_mmgr_got_sigterm = false;
static int			gpu_mmgr_cuda_dindex = -1;
static CUdevice		gpu_mmgr_cuda_device;
static CUcontext	gpu_mmgr_cuda_context;

#define GPUMEM_TRACKER_RAW_EXTRA		((void *)(~0L))

/*
 * nvme_strom_ioctl
 */
static int
nvme_strom_ioctl(int cmd, void *arg)
{
	static int		fdesc_nvme_strom = -1;

	if (fdesc_nvme_strom < 0)
	{
		fdesc_nvme_strom = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
		if (fdesc_nvme_strom < 0)
		{
			int		saved_errno = errno;

			fprintf(stderr, "failed on open('%s'): %m\n",
					NVME_STROM_IOCTL_PATHNAME);
			errno = saved_errno;
			return -1;
		}
	}
	return ioctl(fdesc_nvme_strom, cmd, arg);
}

/*
 * gpuMemFreeChunk
 */
static CUresult
gpuMemFreeChunk(GpuContext *gcontext,
				CUdeviceptr m_deviceptr,
				GpuMemSegment *gm_seg)
{
	size_t			segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	GpuMemSegMap   *gm_smap = &gcontext->gm_smap_array[gm_seg->segment_id];
	GpuMemChunk	   *gm_chunk;
	size_t			offset;
	cl_int			index;

	/* sanity checks */
	if ((gm_seg->gm_kind != GpuMemKind__NormalMemory &&
		 gm_seg->gm_kind != GpuMemKind__IOMapMemory) ||
		gm_seg->segment_id >= num_gpu_memory_segments ||
		m_deviceptr <  gm_smap->m_segment ||
		m_deviceptr >= gm_smap->m_segment + segment_sz)
		return CUDA_ERROR_INVALID_VALUE;

	SpinLockAcquire(&gm_seg->lock);
	offset = (m_deviceptr - gm_smap->m_segment);
	Assert(offset % pgstrom_chunk_size() == 0);
	index = offset / pgstrom_chunk_size();
	gm_chunk = &gm_seg->gm_chunks[index];
	Assert(!gm_chunk->chain.prev && !gm_chunk->chain.next);
	if (--gm_chunk->refcnt > 0)
	{
		SpinLockRelease(&gm_seg->lock);
		return CUDA_SUCCESS;
	}
	dlist_push_head(&gm_seg->free_chunks[0], &gm_chunk->chain);
	SpinLockRelease(&gm_seg->lock);
	return CUDA_SUCCESS;
}

/*
 * gpuMemFreeManagedChunk
 */
static CUresult
gpuMemFreeManagedChunkNoLock(GpuContext *gcontext,
							 CUdeviceptr m_deviceptr,
							 GpuMemSegment *gm_seg)
{
	GpuMemChunk	   *gm_chunk;
	GpuMemChunk	   *gm_buddy;
	cl_long			index, shift;

	index = (m_deviceptr - gm_seg->m_segment) >> GPUMEM_CHUNKSZ_MIN_BIT;
	gm_chunk = &gm_seg->gm_chunks[index];
	Assert(GPUMEMCHUNK_IS_ACTIVE(gm_chunk));
	if (--gm_chunk->refcnt > 0)
		return CUDA_SUCCESS;

	/* Try to merge with the neighbor chunks */
	while (gm_chunk->mclass < GPUMEM_CHUNKSZ_MAX_BIT)
	{
		index = (gm_chunk - gm_seg->gm_chunks);
		shift = 1 << (gm_chunk->mclass - GPUMEM_CHUNKSZ_MIN_BIT);
		Assert((index & (shift - 1)) == 0);
		if ((index & shift) == 0)
		{
			/* try to merge with next */
			gm_buddy = &gm_seg->gm_chunks[index + shift];
			if (gm_buddy->chain.prev != NULL &&
				gm_buddy->chain.next != NULL &&
				gm_buddy->mclass == gm_chunk->mclass)
			{
				/* ok, let's merge */
				dlist_delete(&gm_buddy->chain);
				memset(gm_buddy, 0, sizeof(GpuMemChunk));
				gm_chunk->mclass++;
			}
			else
				break;	/* give up */
		}
		else
		{
			/* try to merge with prev */
			gm_buddy = &gm_seg->gm_chunks[index - shift];
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
				break;	/* give up */
		}
	}
	/* back to the free list again */
	dlist_push_head(&gm_seg->free_chunks[gm_chunk->mclass],
					&gm_chunk->chain);
	return CUDA_SUCCESS;
}

static CUresult
gpuMemFreeManagedChunk(GpuContext *gcontext,
					   CUdeviceptr m_deviceptr,
					   GpuMemSegment *gm_seg)
{
	size_t		segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	CUresult	rc;

	/* sanity checks */
	if (gm_seg->gm_kind != GpuMemKind__ManagedMemory ||
		m_deviceptr < gm_seg->m_segment ||
		m_deviceptr >= gm_seg->m_segment + segment_sz)
		return CUDA_ERROR_INVALID_VALUE;

	SpinLockAcquire(&gm_seg->lock);
	rc = gpuMemFreeManagedChunkNoLock(gcontext, m_deviceptr, gm_seg);
	SpinLockRelease(&gm_seg->lock);

	return rc;
}

/*
 * gpuMemFreeExtra
 */
CUresult
gpuMemFreeExtra(GpuContext *gcontext,
				CUdeviceptr m_deviceptr,
				void *extra)
{
	GpuMemSegment  *gm_seg = extra;

	if (!extra)
		return CUDA_ERROR_INVALID_VALUE;
	if (extra == GPUMEM_TRACKER_RAW_EXTRA)
		return cuMemFree(m_deviceptr);
	if (gm_seg->gm_kind == GpuMemKind__ManagedMemory)
		return gpuMemFreeManagedChunk(gcontext, m_deviceptr, gm_seg);
	return gpuMemFreeChunk(gcontext, m_deviceptr, gm_seg);
}

/*
 * gpuMemFree
 */
CUresult
gpuMemFree(GpuContext *gcontext,
		   CUdeviceptr m_deviceptr)
{
	return gpuMemFreeExtra(gcontext,
						   m_deviceptr,
						   untrackGpuMem(gcontext, m_deviceptr));
}

/*
 * gpuMemAllocRaw
 */
CUresult
__gpuMemAllocRaw(GpuContext *gcontext,
				 CUdeviceptr *p_devptr,
				 size_t bytesize,
				 const char *filename, int lineno)
{
	CUdeviceptr	m_deviceptr;
	CUresult	rc;

	rc = cuCtxPushCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		wnotice("failed on cuCtxPushCurrent: %s", errorText(rc));
		return rc;
	}

	rc = cuMemAlloc(&m_deviceptr, bytesize);
	if (rc != CUDA_SUCCESS)
	{
		wnotice("failed on cuMemAlloc(%zu): %s", bytesize, errorText(rc));
		cuCtxPopCurrent(NULL);
		return rc;
	}
	if (!trackGpuMem(gcontext, m_deviceptr,
					 GPUMEM_TRACKER_RAW_EXTRA))
	{
		cuMemFree(m_deviceptr);
		cuCtxPopCurrent(NULL);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	cuCtxPopCurrent(NULL);

	return CUDA_SUCCESS;
}

/*
 * gpuMemAllocManagedRaw
 */
CUresult
__gpuMemAllocManagedRaw(GpuContext *gcontext,
						CUdeviceptr *p_deviceptr,
						size_t bytesize,
						int flags,
						const char *filename, int lineno)
{
	CUdeviceptr	m_deviceptr;
	CUresult	rc;

	rc = cuCtxPushCurrent(gcontext->cuda_context);
    if (rc != CUDA_SUCCESS)
        return rc;

	rc = cuMemAllocManaged(&m_deviceptr, bytesize, flags);
	if (rc != CUDA_SUCCESS)
	{
		cuCtxPopCurrent(NULL);
		return rc;
	}
	if (!trackGpuMem(gcontext, m_deviceptr,
					 GPUMEM_TRACKER_RAW_EXTRA))
	{
		cuMemFree(m_deviceptr);
		cuCtxPopCurrent(NULL);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	cuCtxPopCurrent(NULL);
	*p_deviceptr = m_deviceptr;

	return CUDA_SUCCESS;
}

/*
 * gpuMemAllocChunk
 */
static CUresult
gpuMemAllocChunk(GpuMemKind gm_kind,
				 GpuContext *gcontext,
				 CUdeviceptr *p_deviceptr,
				 bool may_expand_segment,
				 const char *filename, int lineno)
{
	size_t			segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	cl_int			dindex = gcontext->cuda_dindex;
	GpuMemDevice   *gm_dev = &gm_shead->gm_dev_array[dindex];
	GpuMemSegment  *gm_seg;
	GpuMemSegMap   *gm_smap;
	GpuMemChunk	   *gm_chunk;
	CUdeviceptr		m_deviceptr;
	dlist_iter		iter;
	dlist_node	   *dnode;
	dlist_head	   *local_segment_list;
	dlist_head	   *global_segment_list;
	CUresult		rc;
	cl_int			i, nchunks;
	cl_uint			revision;
	bool			has_exclusive_lock = false;

	/* only backend can call */
	Assert(gpu_mmgr_cuda_dindex < 0);
	Assert(gm_kind == GpuMemKind__NormalMemory ||
		   gm_kind == GpuMemKind__IOMapMemory);
	nchunks = segment_sz / pgstrom_chunk_size();
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			local_segment_list = &gcontext->gm_normal_list;
			global_segment_list = &gm_dev->normal_segment_list;
			break;
		case GpuMemKind__IOMapMemory:
			local_segment_list = &gcontext->gm_iomap_list;
			global_segment_list = &gm_dev->iomap_segment_list;
			break;
		default:
			elog(FATAL, "Bug? unknown GpuMemKind: %d", (int)gm_kind);
	}


	/*
	 * Try to lookup locally mapped segments
	 */
	pthreadRWLockReadLock(&gcontext->gm_rwlock);
retry:
	dlist_foreach(iter, local_segment_list)
	{
		gm_smap = dlist_container(GpuMemSegMap, chain, iter.cur);
		gm_seg = gm_smap->gm_seg;
		SpinLockAcquire(&gm_seg->lock);
		if (!dlist_is_empty(&gm_seg->free_chunks[0]))
		{
			dnode = dlist_pop_head_node(&gm_seg->free_chunks[0]);
			gm_chunk = dlist_container(GpuMemChunk, chain, dnode);
			memset(&gm_chunk->chain, 0, sizeof(dlist_node));
			gm_chunk->refcnt++;
			SpinLockRelease(&gm_seg->lock);

			Assert(gm_chunk >= gm_seg->gm_chunks &&
				   gm_chunk <  gm_seg->gm_chunks + nchunks);
			i = gm_chunk - gm_seg->gm_chunks;
			m_deviceptr = gm_smap->m_segment + i * pgstrom_chunk_size();

			/* track this device memory with GpuContext */
			if (!trackGpuMem(gcontext, m_deviceptr, gm_seg))
			{
				SpinLockAcquire(&gm_seg->lock);
				gm_chunk->refcnt--;
				dlist_push_head(&gm_seg->free_chunks[0],
								&gm_chunk->chain);
				SpinLockRelease(&gm_seg->lock);
				pthreadRWLockUnlock(&gcontext->gm_rwlock);
				return CUDA_ERROR_OUT_OF_MEMORY;
			}
			pthreadRWLockUnlock(&gcontext->gm_rwlock);
			*p_deviceptr = m_deviceptr;
			return CUDA_SUCCESS;
		}
		SpinLockRelease(&gm_seg->lock);
	}

	if (!has_exclusive_lock)
	{
		pthreadRWLockUnlock(&gcontext->gm_rwlock);
		has_exclusive_lock = true;
		pthreadRWLockWriteLock(&gcontext->gm_rwlock);
		goto retry;
	}

	/*
	 * Try to lookup a segment already allocated, but not mapped locally.
	 */
	pthreadRWLockReadLock(&gm_dev->rwlock);
	dlist_foreach(iter, global_segment_list)
	{
		gm_seg = dlist_container(GpuMemSegment, chain, iter.cur);
		Assert(gm_seg->segment_id < num_gpu_memory_segments);
		gm_smap = &gcontext->gm_smap_array[gm_seg->segment_id];
		if (gm_smap->m_segment != 0UL)
			continue;
		Assert(gm_seg->gm_kind == gm_kind);
		Assert(!gm_smap->gm_seg &&
			   !gm_smap->chain.prev &&
			   !gm_smap->chain.next);
		rc = cuCtxPushCurrent(gcontext->cuda_context);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gm_dev->rwlock);
			pthreadRWLockUnlock(&gcontext->gm_rwlock);
			return rc;
		}
		rc = cuIpcOpenMemHandle(&gm_smap->m_segment,
								gm_seg->m_handle,
								CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gm_dev->rwlock);
            pthreadRWLockUnlock(&gcontext->gm_rwlock);
            return rc;
		}
		cuCtxPopCurrent(NULL);
		pg_atomic_fetch_add_u32(&gm_seg->mapcount, 1);

		gm_smap->gm_seg = gm_seg;
		dlist_push_tail(local_segment_list, &gm_smap->chain);
		pthreadRWLockUnlock(&gm_dev->rwlock);
		goto retry;
	}
	pthreadRWLockUnlock(&gm_dev->rwlock);
	pthreadRWLockUnlock(&gcontext->gm_rwlock);
	has_exclusive_lock = false;

	/* Give up, if no additional segments */
	if (!may_expand_segment)
		return CUDA_ERROR_OUT_OF_MEMORY;

	/*
	 * Here is no space left on the existing device memory segment.
	 * So, raise a request to allocate a new one.
	 */
	pthreadMutexLock(&gm_dev->mutex);
	if (!gm_dev->serverLatch)
	{
		pthreadMutexUnlock(&gm_dev->mutex);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	revision = gm_dev->revision;
	gm_dev->alloc_request |= gm_kind;
	SetLatch(gm_dev->serverLatch);
	pthreadCondWait(&gm_dev->cond, &gm_dev->mutex);
	if (gm_dev->revision == revision)
		rc = CUDA_SUCCESS;	/* retry anyway */
	else
	{
		switch (gm_kind)
		{
			case GpuMemKind__NormalMemory:
				rc = gm_dev->status_alloc_normal;
				break;
			case GpuMemKind__IOMapMemory:
				/* XXX - should not happen at this moment */
				rc = gm_dev->status_alloc_iomap;
				break;
			default:
				elog(FATAL, "Bug? unexpected gm_kind: %d", (int)gm_kind);
				break;
		}
	}
	pthreadMutexUnlock(&gm_dev->mutex);
	if (rc != CUDA_SUCCESS)
		return rc;

	pthreadRWLockReadLock(&gcontext->gm_rwlock);
	goto retry;
}

/*
 * gpuMemAlloc
 */
CUresult
__gpuMemAlloc(GpuContext *gcontext,
			  CUdeviceptr *p_deviceptr,
			  size_t bytesize,
			  const char *filename, int lineno)
{
	if (bytesize != pgstrom_chunk_size())
		return __gpuMemAllocRaw(gcontext,
								p_deviceptr,
								bytesize,
								filename, lineno);
	return gpuMemAllocChunk(GpuMemKind__NormalMemory,
							gcontext, p_deviceptr, true,
							filename, lineno);
}

/*
 * gpuMemAllocIOMap
 */
CUresult
__gpuMemAllocIOMap(GpuContext *gcontext,
				   CUdeviceptr *p_deviceptr,
				   size_t bytesize,
				   const char *filename, int lineno)
{
	/*
	 * NOTE: At this moment, only source PDS for NVMe-Strom will need
	 * I/O mapped device memory. So, @bytesize shall be always chunk-size.
	 */
	if (bytesize != pgstrom_chunk_size())
		return CUDA_ERROR_INVALID_VALUE;

	return gpuMemAllocChunk(GpuMemKind__IOMapMemory,
							gcontext, p_deviceptr, false,
							filename, lineno);
}

/*
 * gpuMemSplitManagedChunk
 */
static bool
gpuMemSplitManagedChunk(GpuMemSegment *gm_seg, int mclass)
{
	GpuMemChunk	   *gm_chunk1;
	GpuMemChunk	   *gm_chunk2;
	dlist_node	   *dnode;
	long			offset;

	if (mclass > GPUMEM_CHUNKSZ_MAX_BIT)
		return false;
	Assert(mclass > GPUMEM_CHUNKSZ_MIN_BIT);
	if (dlist_is_empty(&gm_seg->free_chunks[mclass]))
	{
		if (!gpuMemSplitManagedChunk(gm_seg, mclass + 1))
			return false;
	}
	Assert(!dlist_is_empty(&gm_seg->free_chunks[mclass]));
	offset = 1UL << (mclass - 1 - GPUMEM_CHUNKSZ_MIN_BIT);
	dnode = dlist_pop_head_node(&gm_seg->free_chunks[mclass]);
	gm_chunk1 = dlist_container(GpuMemChunk, chain, dnode);
	gm_chunk2 = gm_chunk1 + offset;
	Assert(GPUMEMCHUNK_IS_FREE(gm_chunk1));
	Assert(gm_chunk2->mclass == 0);
	Assert(gm_chunk1->refcnt == 0 && gm_chunk2->refcnt == 0);
	gm_chunk1->mclass = mclass - 1;
	gm_chunk2->mclass = mclass - 1;

	dlist_push_tail(&gm_seg->free_chunks[mclass - 1],
					&gm_chunk1->chain);
	dlist_push_tail(&gm_seg->free_chunks[mclass - 1],
					&gm_chunk2->chain);
	return true;
}

/*
 * gpuMemTryAllocManagedChunk - pick up a chunk from the supplied segment
 *
 * NOTE: caller must hold gm_seg->lock
 */
static inline GpuMemChunk *
gpuMemTryAllocManagedChunk(GpuMemSegment *gm_seg, int mclass)
{
	GpuMemChunk	   *gm_chunk;
	dlist_node	   *dnode;

	if (dlist_is_empty(&gm_seg->free_chunks[mclass]))
	{
		/* split larger chunk */
		if (!gpuMemSplitManagedChunk(gm_seg, mclass + 1))
			return NULL;
	}
	Assert(!dlist_is_empty(&gm_seg->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&gm_seg->free_chunks[mclass]);
	gm_chunk = dlist_container(GpuMemChunk, chain, dnode);
	Assert(GPUMEMCHUNK_IS_FREE(gm_chunk) &&
		   gm_chunk->mclass == mclass);
	memset(&gm_chunk->chain, 0, sizeof(dlist_node));
	gm_chunk->refcnt++;

	return gm_chunk;
}

/*
 * gpuMemAllocManaged
 */
CUresult
__gpuMemAllocManaged(GpuContext *gcontext,
					 CUdeviceptr *p_deviceptr,
					 size_t bytesize,
					 int flags,
					 const char *filename, int lineno)
{
	size_t			segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	size_t			segment_usage;
	dlist_iter		iter;
	GpuMemSegment  *gm_seg;
	GpuMemChunk	   *gm_chunk;
	CUdeviceptr		m_deviceptr;
	CUresult		rc;
	cl_long			i, mclass;
	cl_long			nchunks = segment_sz >> GPUMEM_CHUNKSZ_MIN_BIT;
	bool			has_exclusive_lock = false;

	if (flags != CU_MEM_ATTACH_GLOBAL || bytesize > segment_sz / 2)
		__gpuMemAllocManagedRaw(gcontext,
								p_deviceptr,
								bytesize,
								flags,
								filename, lineno);

	mclass = get_next_log2(bytesize);
	if (mclass < GPUMEM_CHUNKSZ_MIN_BIT)
		mclass = GPUMEM_CHUNKSZ_MIN_BIT;
	else if (mclass > GPUMEM_CHUNKSZ_MAX_BIT)
		return CUDA_ERROR_OUT_OF_MEMORY;

	pthreadRWLockReadLock(&gcontext->gm_rwlock);
retry:
	dlist_foreach(iter, &gcontext->gm_managed_list)
	{
		gm_seg = dlist_container(GpuMemSegment, chain, iter.cur);
		SpinLockAcquire(&gm_seg->lock);
		gm_chunk = gpuMemTryAllocManagedChunk(gm_seg, mclass);
		if (gm_chunk)
		{
			i = gm_chunk - gm_seg->gm_chunks;
			m_deviceptr = gm_seg->m_segment + (i << GPUMEM_CHUNKSZ_MIN_BIT);
			if (trackGpuMem(gcontext, m_deviceptr, gm_seg))
			{
				SpinLockRelease(&gm_seg->lock);
				pthreadRWLockUnlock(&gcontext->gm_rwlock);
				*p_deviceptr = m_deviceptr;
				wnotice("get managed chunk %zu", bytesize);
				return CUDA_SUCCESS;
			}
			/* Oops, failed on tracker allocation */
			gpuMemFreeManagedChunkNoLock(gcontext, m_deviceptr, gm_seg);
		}
		SpinLockRelease(&gm_seg->lock);
	}

	if (!has_exclusive_lock)
	{
		pthreadRWLockUnlock(&gcontext->gm_rwlock);
		has_exclusive_lock = true;
		pthreadRWLockWriteLock(&gcontext->gm_rwlock);
		goto retry;
	}

	/*
	 * allocation of a new managed memory segment
	 */
	rc = CUDA_ERROR_OUT_OF_MEMORY;
	gm_seg = calloc(1, offsetof(GpuMemSegment, gm_chunks[nchunks]));
	if (!gm_seg)
		goto error_1;
	memset(gm_seg, 0, offsetof(GpuMemSegment, gm_chunks[nchunks]));

	wnotice("begin cuMemAllocManaged");
   	rc = cuCtxPushCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		goto error_2;

	rc = cuMemAllocManaged(&m_deviceptr,
						   segment_sz,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		goto error_3;
	cuCtxPopCurrent(NULL);
	wnotice("end cuMemAllocManaged");

	gm_seg->segment_id	= -1;
	gm_seg->gm_kind		= GpuMemKind__ManagedMemory;
	gm_seg->m_segment	= m_deviceptr;
	SpinLockInit(&gm_seg->lock);
	for (i=0; i <= GPUMEM_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&gm_seg->free_chunks[i]);
	mclass = GPUMEM_CHUNKSZ_MAX_BIT;
	segment_usage = 0;
	while (segment_usage < segment_sz &&
		   mclass >= GPUMEM_CHUNKSZ_MIN_BIT)
	{
		if (segment_usage + (1UL << mclass) > segment_sz)
			mclass--;
		else
		{
			i = (segment_usage >> GPUMEM_CHUNKSZ_MIN_BIT);
			gm_chunk = &gm_seg->gm_chunks[i];
			gm_chunk->mclass = mclass;
			dlist_push_tail(&gm_seg->free_chunks[mclass],
							&gm_chunk->chain);
			segment_usage += (1UL << mclass);
		}
	}
	Assert(segment_usage == segment_sz);
	dlist_push_head(&gcontext->gm_managed_list, &gm_seg->chain);
	goto retry;

error_3:
	cuCtxPopCurrent(NULL);
error_2:
	free(gm_seg);
error_1:
	pthreadRWLockUnlock(&gcontext->gm_rwlock);
	return rc;
}

/*
 * gpu_mmgr_alloc_segment - physical segment allocator
 */
static CUresult
__gpu_mmgr_alloc_segment(GpuMemDevice *gm_dev,
						 GpuMemSegment *gm_seg,
						 GpuMemKind gm_kind)
{
	size_t			segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	GpuMemChunk	   *gm_chunk;
	CUdeviceptr		m_segment = 0UL;
	CUresult		rc;
	cl_uint			i, nchunks;

	wlog("Enter __gpu_mmgr_alloc_segment");

	/* init fields */
	gm_seg->gm_kind = gm_kind;
	pg_atomic_init_u32(&gm_seg->mapcount, 0);
	SpinLockInit(&gm_seg->lock);
	memset(&gm_seg->m_handle, 0, sizeof(CUipcMemHandle));
	gm_seg->iomap_handle = 0;
	dlist_init(&gm_seg->free_chunks[0]);

	/* allocation device memory */
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
		case GpuMemKind__IOMapMemory:
			rc = cuMemAlloc(&m_segment, segment_sz);
			if (rc != CUDA_SUCCESS)
			{
				elog(LOG, "GPU Mmgr: failed on cuMemAlloc: %s",
					 errorText(rc));
				return rc;
			}
			if (gm_kind == GpuMemKind__IOMapMemory)
			{
				StromCmd__MapGpuMemory cmd;

				memset(&cmd, 0, sizeof(StromCmd__MapGpuMemory));
				cmd.vaddress = m_segment;
				cmd.length = segment_sz;
				if (nvme_strom_ioctl(STROM_IOCTL__MAP_GPU_MEMORY, &cmd) != 0)
				{
					elog(LOG, "STROM_IOCTL__MAP_GPU_MEMORY failed: %m");
					cuMemFree(m_segment);
					return CUDA_ERROR_MAP_FAILED;
				}
                gm_seg->iomap_handle = cmd.handle;
			}
			break;
		default:
			elog(FATAL, "Bug? unexpected GpuMemKind: %d", (int)gm_kind);
			break;
	}
	/* IPC handler */
	rc = cuIpcGetMemHandle(&gm_seg->m_handle, m_segment);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "GPU Mmgr: failed on cuIpcGetMemHandle: %s",
			 errorText(rc));
		cuMemFree(m_segment);
		return rc;
	}

	{
		elog(LOG, "handle %08x %08x %08x %08x  %08x %08x %08x %08x",
			 ((int *)gm_seg->m_handle.reserved)[0],
			 ((int *)gm_seg->m_handle.reserved)[1],
			 ((int *)gm_seg->m_handle.reserved)[2],
			 ((int *)gm_seg->m_handle.reserved)[3],
			 ((int *)gm_seg->m_handle.reserved)[4],
			 ((int *)gm_seg->m_handle.reserved)[5],
			 ((int *)gm_seg->m_handle.reserved)[6],
			 ((int *)gm_seg->m_handle.reserved)[7]);
		elog(LOG, "------ %08x %08x %08x %08x  %08x %08x %08x %08x",
			 ((int *)gm_seg->m_handle.reserved)[8],
			 ((int *)gm_seg->m_handle.reserved)[9],
			 ((int *)gm_seg->m_handle.reserved)[10],
			 ((int *)gm_seg->m_handle.reserved)[11],
			 ((int *)gm_seg->m_handle.reserved)[12],
			 ((int *)gm_seg->m_handle.reserved)[13],
			 ((int *)gm_seg->m_handle.reserved)[14],
			 ((int *)gm_seg->m_handle.reserved)[15]);
	}

	/* Setup free chunks */
	nchunks = segment_sz / pgstrom_chunk_size();
	for (i=0; i < nchunks; i++)
	{
		gm_chunk = &gm_seg->gm_chunks[i];
		gm_chunk->refcnt = 0;
		dlist_push_tail(&gm_seg->free_chunks[0],
						&gm_chunk->chain);
	}
	gm_seg->m_segment	= m_segment;

	/* segment becomes ready to use for backend processes */
	pthreadRWLockWriteLock(&gm_dev->rwlock);
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			wlog("push to normal segment list %p", gm_seg);
			dlist_push_tail(&gm_dev->normal_segment_list,
							&gm_seg->chain);
			break;
		case GpuMemKind__IOMapMemory:
			dlist_push_tail(&gm_dev->iomap_segment_list,
							&gm_seg->chain);
			break;
		default:
			elog(FATAL, "Bug? unexpected GpuMemKind: %d", (int)gm_kind);
	}
	pthreadRWLockUnlock(&gm_dev->rwlock);

	return CUDA_SUCCESS;
}

static CUresult
gpu_mmgr_alloc_segment(GpuMemDevice *gm_dev,
					   GpuMemKind gm_kind)
{
	GpuMemSegment  *gm_seg = NULL;
	CUresult	rc;

	SpinLockAcquire(&gm_shead->lock);
	if (!dlist_is_empty(&gm_shead->free_segment_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gm_shead->free_segment_list);
		gm_seg = dlist_container(GpuMemSegment, chain, dnode);
	}
	SpinLockRelease(&gm_shead->lock);

	if (!gm_seg)
		return CUDA_ERROR_OUT_OF_MEMORY;
	rc = __gpu_mmgr_alloc_segment(gm_dev, gm_seg, gm_kind);
	if (rc != CUDA_SUCCESS)
	{
		SpinLockAcquire(&gm_shead->lock);
		dlist_push_head(&gm_shead->free_segment_list,
						&gm_seg->chain);
		SpinLockRelease(&gm_shead->lock);
	}
	return rc;
}

/*
 * gpu_mmgr_reclaim_segment
 */
static void
gpu_mmgr_reclaim_segment(GpuMemDevice *gm_dev)
{
	GpuMemSegment  *gm_seg;
	dlist_iter		iter;
	CUresult		rc;

	return; //tentative

	pthreadRWLockWriteLock(&gm_dev->rwlock);
	dlist_foreach(iter, &gm_dev->normal_segment_list)
	{
		gm_seg = dlist_container(GpuMemSegment,
								 chain, iter.cur);
		Assert(gm_seg->gm_kind == GpuMemKind__NormalMemory);
		if (pg_atomic_read_u32(&gm_seg->mapcount) == 0)
		{
			dlist_delete(&gm_seg->chain);
			pthreadRWLockUnlock(&gm_dev->rwlock);

			/* release resources */
			rc = cuMemFree(gm_seg->m_segment);
			if (rc != CUDA_SUCCESS)
				elog(FATAL, "GPU Mmgr: failed on cuMemFree: %s",
					 errorText(rc));
			gm_seg->m_segment = 0UL;
			memset(&gm_seg->m_handle, 0, sizeof(CUipcMemHandle));
			gm_seg->iomap_handle = 0;

			/* segment can be reused */
			SpinLockAcquire(&gm_shead->lock);
			dlist_push_head(&gm_shead->free_segment_list,
							&gm_seg->chain);
			SpinLockRelease(&gm_shead->lock);
			return;
		}
	}
	pthreadRWLockUnlock(&gm_dev->rwlock);
}

/*
 * Buddy allocator of Unified Memory
 */










/*
 * gpu_mmgr_sigterm_handler - SIGTERM handler
 */
static void
gpu_mmgr_sigterm_handler(SIGNAL_ARGS)
{
	int		save_errno = errno;

	gpu_mmgr_got_sigterm = true;

	SetLatch(MyLatch);

	errno = save_errno;
}

/*
 * gpu_mmgr_event_loop
 */
static void
gpu_mmgr_event_loop(GpuMemDevice *gm_dev)
{
	int		ev = 0;

	while (!gpu_mmgr_got_sigterm)
	{
		cl_uint	alloc_request;
		int		nstatus = 0;
		int		istatus = 0;

		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();

		if (ev & WL_TIMEOUT)
			gpu_mmgr_reclaim_segment(gm_dev);

		pthreadMutexLock(&gm_dev->mutex);
		alloc_request = gm_dev->alloc_request;
		gm_dev->alloc_request = 0;
		gm_dev->revision++;
		pthreadMutexUnlock(&gm_dev->mutex);

		if ((alloc_request & GpuMemKind__NormalMemory) != 0)
			nstatus = gpu_mmgr_alloc_segment(gm_dev,
											 GpuMemKind__NormalMemory);
		if ((alloc_request & GpuMemKind__IOMapMemory) != 0)
			istatus = gpu_mmgr_alloc_segment(gm_dev,
											 GpuMemKind__IOMapMemory);
		pthreadMutexLock(&gm_dev->mutex);
		/* write back error status, if any */
		if ((alloc_request & GpuMemKind__NormalMemory) != 0)
			gm_dev->status_alloc_normal	= nstatus;
		if ((alloc_request & GpuMemKind__IOMapMemory) != 0)
			gm_dev->status_alloc_iomap	= istatus;
		/* clear the requests already allocated */
		gm_dev->alloc_request &= ~alloc_request;
		/* wake up any of the waiting backends  */
		pthreadCondBroadcast(&gm_dev->cond);
		pthreadMutexUnlock(&gm_dev->mutex);

		ev = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_TIMEOUT |
					   WL_POSTMASTER_DEATH,
					   5 * 1000);		/* wake up per 5sec */

		/* emergency bailout if postmaster gets dead */
		if (ev & WL_POSTMASTER_DEATH)
			elog(FATAL, "emergency bailout due to unexpected postmaster dead");
	}
}

/*
 * gpu_mmgr_bgworker_main
 */
static void
gpu_mmgr_bgworker_main(Datum bgworker_arg)
{
	size_t			segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	GpuMemDevice   *gm_dev;
	GpuMemSegment  *gm_seg;
	dlist_node	   *dnode;
	cl_long			i, nsegments;
	CUresult		rc;

	/* allows to accept signals */
	pqsignal(SIGTERM, gpu_mmgr_sigterm_handler);
	BackgroundWorkerUnblockSignals();

	/* init resource management stuff */
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "GPU Mmgr");
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "GPU Mmgr",
												 ALLOCSET_DEFAULT_MINSIZE,
												 ALLOCSET_DEFAULT_INITSIZE,
												 ALLOCSET_DEFAULT_MAXSIZE);
	/*
	 * init CUDA driver APIs stuff
	 */
	gpu_mmgr_cuda_dindex = DatumGetInt32(bgworker_arg);
	Assert(gpu_mmgr_cuda_dindex >= 0 &&
		   gpu_mmgr_cuda_dindex < numDevAttrs);
	gm_dev = &gm_shead->gm_dev_array[gpu_mmgr_cuda_dindex];

	/* ensure not to use MPS daemon */
	setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1);

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	rc = cuDeviceGet(&gpu_mmgr_cuda_device,
					 devAttrs[gpu_mmgr_cuda_dindex].DEV_ID);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&gpu_mmgr_cuda_context,
					 CU_CTX_SCHED_AUTO,
					 gpu_mmgr_cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxCreate: %s", errorText(rc));

	/* allocation of I/O mapped memory */
	nsegments = iomap_gpu_memory_size_kb / gpu_memory_segment_size_kb;
	for (i=0; i < nsegments; i++)
	{
		if (dlist_is_empty(&gm_shead->free_segment_list))
		{
			elog(LOG, "Bug? All the I/O map segments were not configured");
			break;
		}
		dnode = dlist_pop_head_node(&gm_shead->free_segment_list);
		gm_seg = dlist_container(GpuMemSegment, chain, dnode);
		rc = __gpu_mmgr_alloc_segment(gm_dev,
									  gm_seg,
									  GpuMemKind__IOMapMemory);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on allocation of I/O map memory: %s",
				 errorText(rc));
		elog(LOG, "I/O mapped memory %p-%p for GPU%u [%s]",
			 (void *)(gm_seg->m_segment),
			 (void *)(gm_seg->m_segment + segment_sz - 1),
             gpu_mmgr_cuda_dindex,
             devAttrs[gpu_mmgr_cuda_dindex].DEV_NAME);
	}

	/* enables to accept device memory allocation request */
	pthreadMutexLock(&gm_dev->mutex);
	gm_dev->serverLatch = MyLatch;
	pthreadMutexUnlock(&gm_dev->mutex);

	/* event loop of GPU Mmgr */
	PG_TRY();
	{
		gpu_mmgr_event_loop(gm_dev);
	}
	PG_CATCH();
	{
		pthreadMutexLock(&gm_dev->mutex);
		gm_dev->serverLatch = NULL;
		pthreadCondBroadcast(&gm_dev->cond);
		pthreadMutexUnlock(&gm_dev->mutex);
		PG_RE_THROW();
	}
	PG_END_TRY();

	pthreadMutexLock(&gm_dev->mutex);
	gm_dev->serverLatch = NULL;
	pthreadCondBroadcast(&gm_dev->cond);
	pthreadMutexUnlock(&gm_dev->mutex);

	elog(ERROR, "GPU Mmgr%d [%s] normally terminated",
		 gpu_mmgr_cuda_dindex,
		 devAttrs[gpu_mmgr_cuda_dindex].DEV_NAME);
}

/*
 * pgstrom_gpu_mmgr_init_gpucontext - Per GpuContext initialization
 */
bool
pgstrom_gpu_mmgr_init_gpucontext(GpuContext *gcontext)
{
	pthreadRWLockInit(&gcontext->gm_rwlock);
	dlist_init(&gcontext->gm_normal_list);
	dlist_init(&gcontext->gm_iomap_list);
	gcontext->gm_smap_array = calloc(num_gpu_memory_segments,
									 sizeof(GpuMemSegMap));
	return (gcontext->gm_smap_array != NULL);
}

/*
 * pgstrom_gpu_mmgr_cleanup_gpucontext - Per GpuContext cleanup
 */
void
pgstrom_gpu_mmgr_cleanup_gpucontext(GpuContext *gcontext)
{
	GpuMemSegMap   *gm_smap;
	dlist_iter		iter;
	CUresult		rc;

	dlist_foreach(iter, &gcontext->gm_normal_list)
	{
		gm_smap = dlist_container(GpuMemSegMap, chain, iter.cur);
		rc = cuIpcCloseMemHandle(gm_smap->m_segment);
		if (rc != CUDA_SUCCESS)
			elog(NOTICE, "failed on cuIpcCloseMemHandle: %s", errorText(rc));
		pg_atomic_sub_fetch_u32(&gm_smap->gm_seg->mapcount, 1);
	}

	dlist_foreach(iter, &gcontext->gm_iomap_list)
	{
		gm_smap = dlist_container(GpuMemSegMap, chain, iter.cur);
		rc = cuIpcCloseMemHandle(gm_smap->m_segment);
		if (rc != CUDA_SUCCESS)
			elog(NOTICE, "failed on cuIpcCloseMemHandle: %s", errorText(rc));
		pg_atomic_sub_fetch_u32(&gm_smap->gm_seg->mapcount, 1);
	}
}

/*
 * pgstrom_startup_gpu_mmgr
 */
static void
pgstrom_startup_gpu_mmgr(void)
{
	GpuMemSegment *gm_seg;
	size_t	segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	Size	nchunks;
	Size	unitsz;
	Size	head_sz;
	Size	required;
	bool	found;
	int		i;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	/*
	 * GpuMemSegment (shared structure)
	 */
	nchunks = segment_sz / pgstrom_chunk_size();
	head_sz = STROMALIGN(offsetof(GpuMemSystemHead,
								  gm_dev_array[numDevAttrs]));
	unitsz = STROMALIGN(offsetof(GpuMemSegment,
								 gm_chunks[nchunks]));
	required = (head_sz + unitsz * num_gpu_memory_segments);
	gm_shead = ShmemInitStruct("GPU Device Memory Management Structure",
							   required, &found);
	if (found)
		elog(ERROR, "Bug? GPU Device Memory Management Structure exists");
	memset(gm_shead, 0, required);

	SpinLockInit(&gm_shead->lock);
	dlist_init(&gm_shead->free_segment_list);
	for (i=0; i < numDevAttrs; i++)
	{
		GpuMemDevice *gm_dev = &gm_shead->gm_dev_array[i];

		pthreadMutexInit(&gm_dev->mutex);
		pthreadCondInit(&gm_dev->cond);
		pthreadRWLockInit(&gm_dev->rwlock);
		dlist_init(&gm_dev->normal_segment_list);
		dlist_init(&gm_dev->iomap_segment_list);
	}

	gm_seg = (GpuMemSegment *)((char *)gm_shead + head_sz);
	for (i=0; i < num_gpu_memory_segments; i++)
	{
		gm_seg->segment_id	= i;
		dlist_push_tail(&gm_shead->free_segment_list,
						&gm_seg->chain);
		gm_seg = (GpuMemSegment *)((char *)gm_seg + unitsz);
	}
}

/*
 * pgstrom_init_gpu_mmgr
 */
void
pgstrom_init_gpu_mmgr(void)
{
	BackgroundWorker worker;
	long		sysconf_pagesize;		/* _SC_PAGESIZE */
	long		sysconf_phys_pages;		/* _SC_PHYS_PAGES */
	Size		shared_buffer_size = (Size)NBuffers * (Size)BLCKSZ;
	Size		segment_sz;
	Size		required;
	size_t		nchunks;
	int			i;

	/*
	 * segment size of the device memory in kB
	 */
	DefineCustomIntVariable("pg_strom.gpu_memory_segment_size",
							"default size of the GPU device memory segment",
							NULL,
							&gpu_memory_segment_size_kb,
							(1UL << 19),	/* 512MB */
							pgstrom_chunk_size() >> 10,
							GPUMEM_CHUNKSZ_MAX >> 10,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	if (segment_sz % pgstrom_chunk_size() != 0)
		elog(ERROR, "pg_strom.gpu_memory_segment_size(%dkB) must be multiple number of pg_strom.chunk_size(%dkB)",
			 gpu_memory_segment_size_kb,
			 (int)(pgstrom_chunk_size() >> 10));

	/*
	 * count total number of device memory segments
	 */
	num_gpu_memory_segments = 0;
	for (i=0; i < numDevAttrs; i++)
	{
		num_gpu_memory_segments += (devAttrs[i].DEV_TOTAL_MEMSZ +
									segment_sz - 1) / segment_sz;
	}

	/*
	 * size of the i/o mapped device memory (for NVMe-Strom)
	 */
	DefineCustomIntVariable("pg_strom.gpu_memory_iomap_size",
							"size of I/O mapped GPU device memory",
							NULL,
							&iomap_gpu_memory_size_kb,
							0,
							0,
							gpu_memory_segment_size_kb *
							num_gpu_memory_segments,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	if (gpu_memory_segment_size_kb % gpu_memory_segment_size_kb != 0)
		elog(ERROR, "pg_strom.gpu_memory_iomap_size (%dkB) must be multiple number of pg_strom.gpu_memory_segment_size(%dkB)",
			 iomap_gpu_memory_size_kb,
			 gpu_memory_segment_size_kb);
	num_gpu_iomap_segments = (iomap_gpu_memory_size_kb /
							  gpu_memory_segment_size_kb) * numDevAttrs;
	if (num_gpu_iomap_segments >= num_gpu_memory_segments)
		elog(ERROR, "pg_strom.gpu_memory_iomap_size (%dkB) is too large",
			 iomap_gpu_memory_size_kb);

	/* pg_strom.nvme_strom_enabled */
	DefineCustomBoolVariable("pg_strom.nvme_strom_enabled",
							 "Turn on/off SSD-to-GPU P2P DMA",
							 NULL,
							 &nvme_strom_enabled,
							 true,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* pg_strom.debug_force_nvme_strom */
	DefineCustomBoolVariable("pg_strom.debug_force_nvme_strom",
							 "(DEBUG) force to use raw block scan mode",
							 NULL,
							 &debug_force_nvme_strom,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/*
	 * MEMO: Threshold of table's physical size to use NVMe-Strom:
	 *   ((System RAM size) -
	 *    (shared_buffer size)) * 0.67 + (shared_buffer size)
	 *
	 * If table size is enough large to issue real i/o, NVMe-Strom will
	 * make advantage by higher i/o performance.
	 */
	sysconf_pagesize = sysconf(_SC_PAGESIZE);
	if (sysconf_pagesize < 0)
		elog(ERROR, "failed on sysconf(_SC_PAGESIZE): %m");
	sysconf_phys_pages = sysconf(_SC_PHYS_PAGES);
	if (sysconf_phys_pages < 0)
		elog(ERROR, "failed on sysconf(_SC_PHYS_PAGES): %m");
	if (sysconf_pagesize * sysconf_phys_pages < shared_buffer_size)
		elog(ERROR, "Bug? shared_buffer is larger than system RAM");
	nvme_strom_threshold = ((sysconf_pagesize * sysconf_phys_pages -
							 shared_buffer_size) * 2 / 3 +
							shared_buffer_size) / BLCKSZ;

	/*
	 * request for the static shared memory
	 */
	nchunks = segment_sz / pgstrom_chunk_size();
	required = STROMALIGN(offsetof(GpuMemSystemHead,
								   gm_dev_array[numDevAttrs])) +
		STROMALIGN(offsetof(GpuMemSegment,
							gm_chunks[nchunks])) * num_gpu_memory_segments;

	RequestAddinShmemSpace(required);
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_mmgr;

	/*
	 * setup a background server process for memory management
	 */
	for (i=0; i < numDevAttrs; i++)
	{
		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "GPU Mmgr%d [%s]", i, devAttrs[i].DEV_NAME);

		worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = 1;
		worker.bgw_main = gpu_mmgr_bgworker_main;
		worker.bgw_main_arg = Int32GetDatum(i);
		RegisterBackgroundWorker(&worker);
	}
}

/* ----------------------------------------------------------------
 *
 * APIs for SSD-to-GPU Direct DMA
 *
 * ---------------------------------------------------------------- */

/*
 * gpuMemCopyFromSSDWaitRaw
 */
static void
gpuMemCopyFromSSDWaitRaw(unsigned long dma_task_id)
{
	StromCmd__MemCopyWait cmd;

	memset(&cmd, 0, sizeof(StromCmd__MemCopyWait));
	cmd.dma_task_id = dma_task_id;

	if (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_WAIT, &cmd) != 0)
		werror("failed on nvme_strom_ioctl(STROM_IOCTL__MEMCPY_WAIT): %m");
}

/*
 * gpuMemCopyFromSSD - kick SSD-to-GPU Direct DMA, then wait for completion
 */
void
gpuMemCopyFromSSD(GpuTask *gtask,
				  CUdeviceptr m_kds,
				  pgstrom_data_store *pds)
{
	GpuContext	   *gcontext = GpuWorkerCurrentContext;
	size_t			segment_sz = (size_t)gpu_memory_segment_size_kb << 10;
	StromCmd__MemCopySsdToGpu cmd;
	GpuMemSegment  *gm_seg;
	GpuMemSegMap   *gm_smap;
	BlockNumber	   *block_nums;
	void		   *block_data;
	size_t			offset;
	size_t			length;
	cl_uint			nr_loaded;
	CUresult		rc;

	if (iomap_gpu_memory_size_kb == 0)
		werror("NVMe-Strom is not configured");
	/* ensure the @m_kds is exactly i/o mapped buffer */
	Assert(gcontext != NULL);
	gm_seg = lookupGpuMem(gcontext, m_kds);
	if (!gm_seg || gm_seg->iomap_handle == 0UL)
		werror("nvme-strom: invalid device pointer");
	Assert(pds->kds.format == KDS_FORMAT_BLOCK);

	gm_smap = &gcontext->gm_smap_array[gm_seg->segment_id];
	if (m_kds < gm_smap->m_segment ||
		m_kds + pds->kds.length >= gm_smap->m_segment + segment_sz)
		werror("nvme-strom: P2P DMA destination out of range");
	offset = m_kds - gm_smap->m_segment;

	/* nothing special if all the blocks are already loaded */
	if (pds->nblocks_uncached == 0)
	{
		rc = cuMemcpyHtoDAsync(m_kds,
							   &pds->kds,
							   pds->kds.length,
							   CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		return;
	}
	Assert(pds->nblocks_uncached <= pds->kds.nitems);
	nr_loaded = pds->kds.nitems - pds->nblocks_uncached;
	length = ((char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds, nr_loaded) -
			  (char *)(&pds->kds));
	offset += length;

	/* userspace pointers */
	block_nums = (BlockNumber *)KERN_DATA_STORE_BODY(&pds->kds) + nr_loaded;
	block_data = KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds, nr_loaded);

	/* setup ioctl(2) command */
	memset(&cmd, 0, sizeof(StromCmd__MemCopySsdToGpu));
	cmd.handle		= gm_seg->iomap_handle;
	cmd.offset		= offset;
	cmd.file_desc	= gtask->file_desc;
	cmd.nr_chunks	= pds->nblocks_uncached;
	cmd.chunk_sz	= BLCKSZ;
	cmd.relseg_sz	= RELSEG_SIZE;
	cmd.chunk_ids	= block_nums;
	cmd.wb_buffer	= block_data;

	/* (1) kick SSD2GPU P2P DMA */
	if (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_SSD2GPU, &cmd) != 0)
		werror("failed on STROM_IOCTL__MEMCPY_SSD2GPU: %m");

	/* (2) kick RAM2GPU DMA (earlier half) */
	rc = cuMemcpyHtoDAsync(m_kds,
						   &pds->kds,
						   length,
						   CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuMemCopyFromSSDWaitRaw(cmd.dma_task_id);
		werror("failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	}

	/* (3) kick RAM2GPU DMA (later half; if any) */
	if (cmd.nr_ram2gpu > 0)
	{
		length = BLCKSZ * cmd.nr_ram2gpu;
		offset = ((char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds,
													   pds->kds.nitems) -
				  (char *)&pds->kds) - length;
		rc = cuMemcpyHtoDAsync(m_kds + offset,
							   (char *)&pds->kds + offset,
							   length,
							   CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuMemCopyFromSSDWaitRaw(cmd.dma_task_id);
			werror("failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		}
	}
	/* (4) wait for completion of SSD2GPU P2P DMA */
	gpuMemCopyFromSSDWaitRaw(cmd.dma_task_id);
}

/*
 * TablespaceCanUseNvmeStrom
 */
typedef struct
{
	Oid		tablespace_oid;
	bool	nvme_strom_supported;
} vfs_nvme_status;

static HTAB	   *vfs_nvme_htable = NULL;
static Oid		nvme_last_tablespace_oid = InvalidOid;
static bool		nvme_last_tablespace_supported;

static void
vfs_nvme_cache_callback(Datum arg, int cacheid, uint32 hashvalue)
{
	/* invalidate all the cached status */
	if (vfs_nvme_htable)
	{
		hash_destroy(vfs_nvme_htable);
		vfs_nvme_htable = NULL;
		nvme_last_tablespace_oid = InvalidOid;
	}
}

static bool
TablespaceCanUseNvmeStrom(Oid tablespace_oid)
{
	vfs_nvme_status *entry;
	const char *pathname;
	int			fdesc;
	bool		found;

	if (iomap_gpu_memory_size_kb == 0 || !nvme_strom_enabled)
		return false;	/* NVMe-Strom is not configured or enabled */

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	/* quick lookup but sufficient for more than 99.99% cases */
	if (OidIsValid(nvme_last_tablespace_oid) &&
		nvme_last_tablespace_oid == tablespace_oid)
		return nvme_last_tablespace_supported;

	if (!vfs_nvme_htable)
	{
		HASHCTL		ctl;

		memset(&ctl, 0, sizeof(HASHCTL));
		ctl.keysize = sizeof(Oid);
		ctl.entrysize = sizeof(vfs_nvme_status);
		vfs_nvme_htable = hash_create("VFS:NVMe-Strom status", 64,
									  &ctl, HASH_ELEM | HASH_BLOBS);
		CacheRegisterSyscacheCallback(TABLESPACEOID,
									  vfs_nvme_cache_callback, (Datum) 0);
	}
	entry = (vfs_nvme_status *) hash_search(vfs_nvme_htable,
											&tablespace_oid,
											HASH_ENTER,
											&found);
	if (found)
	{
		nvme_last_tablespace_oid = tablespace_oid;
		nvme_last_tablespace_supported = entry->nvme_strom_supported;
		return entry->nvme_strom_supported;
	}

	/* check whether the tablespace is supported */
	entry->tablespace_oid = tablespace_oid;
	entry->nvme_strom_supported = false;

	pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);
	fdesc = open(pathname, O_RDONLY | O_DIRECTORY);
	if (fdesc < 0)
	{
		elog(WARNING, "failed to open \"%s\" of tablespace \"%s\": %m",
			 pathname, get_tablespace_name(tablespace_oid));
	}
	else
	{
		StromCmd__CheckFile cmd;

		cmd.fdesc = fdesc;
		if (nvme_strom_ioctl(STROM_IOCTL__CHECK_FILE, &cmd) == 0)
			entry->nvme_strom_supported = true;
		else
		{
			ereport(NOTICE,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("nvme_strom does not support tablespace \"%s\"",
							get_tablespace_name(tablespace_oid))));
		}
	}
	nvme_last_tablespace_oid = tablespace_oid;
	nvme_last_tablespace_supported = entry->nvme_strom_supported;
	return entry->nvme_strom_supported;
}

bool
RelationCanUseNvmeStrom(Relation relation)
{
	Oid		tablespace_oid = RelationGetForm(relation)->reltablespace;
	/* SSD2GPU on temp relation is not supported */
	if (RelationUsesLocalBuffers(relation))
		return false;
	return TablespaceCanUseNvmeStrom(tablespace_oid);
}

/*
 * RelationWillUseNvmeStrom
 */
bool
RelationWillUseNvmeStrom(Relation relation, BlockNumber *p_nr_blocks)
{
	BlockNumber		nr_blocks;

	/* at least, storage must support NVMe-Strom */
	if (!RelationCanUseNvmeStrom(relation))
		return false;

	/*
	 * NOTE: RelationGetNumberOfBlocks() has a significant but helpful
	 * side-effect. It opens all the underlying files of MAIN_FORKNUM,
	 * then set @rd_smgr of the relation.
	 * It allows extension to touch file descriptors without invocation of
	 * ReadBuffer().
	 */
	nr_blocks = RelationGetNumberOfBlocks(relation);
	if (!debug_force_nvme_strom &&
		nr_blocks < nvme_strom_threshold)
		return false;

	/*
	 * ok, it looks to me NVMe-Strom is supported, and relation size is
	 * reasonably large to run with SSD-to-GPU Direct mode.
	 */
	if (p_nr_blocks)
		*p_nr_blocks = nr_blocks;
	return true;
}

/*
 * ScanPathWillUseNvmeStrom - Optimizer Hint
 */
bool
ScanPathWillUseNvmeStrom(PlannerInfo *root, RelOptInfo *baserel)
{
	RangeTblEntry *rte;
	HeapTuple	tuple;
	bool		relpersistence;

	if (!TablespaceCanUseNvmeStrom(baserel->reltablespace))
		return false;

	/* unable to apply NVMe-Strom on temporay tables */
	rte = root->simple_rte_array[baserel->relid];
	tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for relation %u", rte->relid);
	relpersistence = ((Form_pg_class) GETSTRUCT(tuple))->relpersistence;
	ReleaseSysCache(tuple);

	if (relpersistence != RELPERSISTENCE_PERMANENT &&
		relpersistence != RELPERSISTENCE_UNLOGGED)
		return false;

	/* Is number of blocks sufficient to NVMe-Strom? */
	if (!debug_force_nvme_strom && baserel->pages < nvme_strom_threshold)
		return false;

	/* ok, this table scan can use nvme-strom */
	return true;
}
