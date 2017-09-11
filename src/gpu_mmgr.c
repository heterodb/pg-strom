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

#define GPUMEM_CHUNKSZ_MAX_BIT		30		/* 1GB */
#define GPUMEM_CHUNKSZ_MIN_BIT		14		/* 16KB */
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

typedef struct
{
	dlist_node		chain;
	GpuMemKind		gm_kind;	/* one of GpuMemKind__* */
	CUdeviceptr		m_segment;	/* device pointer of the segment */
	unsigned long	iomap_handle; /* only if GpuMemKind__IOMapMemory */
	slock_t			lock;		/* protection of chunks */
	dlist_head		free_chunks[GPUMEM_CHUNKSZ_MAX_BIT + 1];
	GpuMemChunk		gm_chunks[FLEXIBLE_ARRAY_MEMBER];
} GpuMemSegment;

/* statistics of GPU memory usage (shared; per device) */
//to be used for memory release request mechanism
typedef struct
{
	size_t				total_size;
	pg_atomic_uint64	normal_usage;
	pg_atomic_uint64	managed_usage;
	pg_atomic_uint64	iomap_usage;
} GpuMemStatistics;

/* static variables */
static shmem_startup_hook_type shmem_startup_next = NULL;
static GpuMemStatistics *gm_stat_array = NULL;
static int			gpu_memory_segment_size_kb;	/* GUC */
static size_t		gm_segment_sz;	/* bytesize */

static bool			debug_force_nvme_strom;		/* GUC */
static bool			nvme_strom_enabled;			/* GUC */
static long			nvme_strom_threshold;

#define GPUMEM_DEVICE_RAW_EXTRA		((void *)(~0L))
#define GPUMEM_HOST_RAW_EXTRA		((void *)(~1L))

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
	GpuMemChunk	   *gm_chunk;
	GpuMemChunk	   *gm_buddy;
	cl_long			index;
	cl_long			shift;
	cl_long			nchunks;

	Assert(m_deviceptr >= gm_seg->m_segment &&
		   m_deviceptr <  gm_seg->m_segment + gm_segment_sz);
	if (gm_seg->gm_kind == GpuMemKind__ManagedMemory)
		index = (m_deviceptr - gm_seg->m_segment) >> GPUMEM_CHUNKSZ_MIN_BIT;
	else
		index = (m_deviceptr - gm_seg->m_segment) / pgstrom_chunk_size();
	gm_chunk = &gm_seg->gm_chunks[index];
	Assert(GPUMEMCHUNK_IS_ACTIVE(gm_chunk));
	SpinLockAcquire(&gm_seg->lock);
	if (--gm_chunk->refcnt > 0)
		return CUDA_SUCCESS;

	/* GpuMemKind__ManagedMemory tries to merge with prev/next chunks */
	if (gm_seg->gm_kind == GpuMemKind__ManagedMemory)
	{
		nchunks = gm_segment_sz >> GPUMEM_CHUNKSZ_MIN_BIT;

		while (gm_chunk->mclass < GPUMEM_CHUNKSZ_MAX_BIT)
		{
			index = (gm_chunk - gm_seg->gm_chunks);
			shift = 1 << (gm_chunk->mclass - GPUMEM_CHUNKSZ_MIN_BIT);
			Assert((index & (shift - 1)) == 0);
			if ((index & shift) == 0)
			{
				/* try to merge with next */
				if (index + shift >= nchunks)
					break;	/* out of range, give up */

				gm_buddy = &gm_seg->gm_chunks[index + shift];
				if (GPUMEMCHUNK_IS_FREE(gm_buddy) &&
					gm_chunk->mclass == gm_buddy->mclass)
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
				if (index < shift)
					break;	/* out of range, give up */
				gm_buddy = &gm_seg->gm_chunks[index - shift];
				if (GPUMEMCHUNK_IS_FREE(gm_buddy) &&
					gm_chunk->mclass == gm_buddy->mclass)
				{
					/* ok, let's merge */
					dlist_delete(&gm_buddy->chain);
					memset(gm_chunk, 0, sizeof(GpuMemChunk));
					gm_buddy->mclass++;
					gm_chunk = gm_buddy;
				}
				else
					break;	/* give up */
			}
		}
	}
	/* back to the free list again */
	dlist_push_head(&gm_seg->free_chunks[gm_chunk->mclass],
					&gm_chunk->chain);
	SpinLockRelease(&gm_seg->lock);

    return CUDA_SUCCESS;
}

/*
 * gpuMemFreeExtra
 */
CUresult
gpuMemFreeExtra(GpuContext *gcontext,
				CUdeviceptr m_deviceptr,
				void *extra)
{
	if (!extra)
		return CUDA_ERROR_INVALID_VALUE;
	else if (extra == GPUMEM_DEVICE_RAW_EXTRA)
		return cuMemFree(m_deviceptr);
	else if (extra == GPUMEM_HOST_RAW_EXTRA)
		return cuMemFreeHost((void *)m_deviceptr);
	return gpuMemFreeChunk(gcontext, m_deviceptr, (GpuMemSegment *)extra);
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
					 GPUMEM_DEVICE_RAW_EXTRA,
					 filename, lineno))
	{
		cuMemFree(m_deviceptr);
		cuCtxPopCurrent(NULL);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	cuCtxPopCurrent(NULL);

	return CUDA_SUCCESS;
}

/*
 * __gpuMemAllocManagedRaw
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
					 GPUMEM_DEVICE_RAW_EXTRA,
					 filename, lineno))
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
 * __gpuMemAllocHostRaw
 */
CUresult
__gpuMemAllocHostRaw(GpuContext *gcontext,
					 void **p_hostptr,
					 size_t bytesize,
					 const char *filename, int lineno)
{
	void	   *hostptr;
	CUresult	rc;

	rc = cuCtxPushCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		return rc;

	rc = cuMemAllocHost(&hostptr, bytesize);
	if (rc != CUDA_SUCCESS)
	{
		cuCtxPopCurrent(NULL);
		return rc;
	}

	if (!trackGpuMem(gcontext, (CUdeviceptr)hostptr,
					 GPUMEM_HOST_RAW_EXTRA,
					 filename, lineno))
	{
		cuMemFreeHost(hostptr);
		cuCtxPopCurrent(NULL);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	cuCtxPopCurrent(NULL);
	*p_hostptr = hostptr;

	return CUDA_SUCCESS;
}

/*
 * gpuMemSplitChunk
 */
static bool
gpuMemSplitChunk(GpuMemSegment *gm_seg, cl_int mclass)
{
	GpuMemChunk	   *gm_chunk1;
	GpuMemChunk	   *gm_chunk2;
	dlist_node	   *dnode;
	long			offset;

	Assert(gm_seg->gm_kind == GpuMemKind__ManagedMemory);
	if (mclass > GPUMEM_CHUNKSZ_MAX_BIT)
		return false;
	Assert(mclass > GPUMEM_CHUNKSZ_MIN_BIT);
	if (dlist_is_empty(&gm_seg->free_chunks[mclass]))
	{
		if (!gpuMemSplitChunk(gm_seg, mclass + 1))
			return false;
	}
	Assert(!dlist_is_empty(&gm_seg->free_chunks[mclass]));
	dnode = dlist_pop_head_node(&gm_seg->free_chunks[mclass]);
	offset = 1UL << (mclass - 1 - GPUMEM_CHUNKSZ_MIN_BIT);
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
 * gpuMemAllocChunk
 */
static CUresult
gpuMemAllocChunk(GpuMemKind gm_kind,
				 GpuContext *gcontext,
				 CUdeviceptr *p_deviceptr,
				 cl_int mclass,
				 const char *filename, int lineno)
{
	GpuMemStatistics *gm_stat;
	GpuMemSegment  *gm_seg;
	GpuMemChunk	   *gm_chunk;
	CUdeviceptr		m_segment;
	dlist_iter		iter;
	dlist_node	   *dnode;
	dlist_head	   *gm_segment_list;
	CUresult		rc;
	cl_int			i, nchunks;
	bool			has_exclusive_lock = false;

	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			gm_segment_list = &gcontext->gm_normal_list;
			nchunks = gm_segment_sz / pgstrom_chunk_size();
			break;
		case GpuMemKind__IOMapMemory:
			gm_segment_list = &gcontext->gm_iomap_list;
			nchunks = gm_segment_sz >> GPUMEM_CHUNKSZ_MIN_BIT;
			break;
		case GpuMemKind__ManagedMemory:
			gm_segment_list = &gcontext->gm_managed_list;
			nchunks = gm_segment_sz / pgstrom_chunk_size();
			break;
		case GpuMemKind__HostMemory:
			gm_segment_list = &gcontext->gm_hostmem_list;
			nchunks = gm_segment_sz / pgstrom_chunk_size();
			break;
		default:
			return CUDA_ERROR_INVALID_VALUE;
	}

	/*
	 * Try to lookup already allocated segment first
	 */
	pthreadRWLockReadLock(&gcontext->gm_rwlock);
retry:
	dlist_foreach(iter, gm_segment_list)
	{
		gm_seg = dlist_container(GpuMemSegment, chain, iter.cur);
		SpinLockAcquire(&gm_seg->lock);
		/* try to split larger chunks if managed-memory */
		if (gm_kind == GpuMemKind__ManagedMemory &&
			dlist_is_empty(&gm_seg->free_chunks[mclass]))
		{
			gpuMemSplitChunk(gm_seg, mclass + 1);
		}

		if (!dlist_is_empty(&gm_seg->free_chunks[mclass]))
		{
			dnode = dlist_pop_head_node(&gm_seg->free_chunks[mclass]);
			gm_chunk = dlist_container(GpuMemChunk, chain, dnode);
			Assert(GPUMEMCHUNK_IS_FREE(gm_chunk) &&
				   gm_chunk->mclass == mclass);
			memset(&gm_chunk->chain, 0, sizeof(dlist_node));
			gm_chunk->refcnt++;
			SpinLockRelease(&gm_seg->lock);
			pthreadRWLockUnlock(&gcontext->gm_rwlock);
			/* ok, found */
			
			
			
			return CUDA_SUCCESS;
		}
	}

	if (!has_exclusive_lock)
	{
		pthreadRWLockUnlock(&gcontext->gm_rwlock);
		has_exclusive_lock = true;
		pthreadRWLockWriteLock(&gcontext->gm_rwlock);
		goto retry;
	}

	/*
	 * allocation of a new segment
	 */
	gm_seg = calloc(1, offsetof(GpuMemSegment, gm_chunks[nchunks]));
	if (!gm_seg)
	{
		pthreadRWLockUnlock(&gcontext->gm_rwlock);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}

	rc = cuCtxPushCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		free(gm_seg);
		pthreadRWLockUnlock(&gcontext->gm_rwlock);
		return rc;
	}

	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			rc = cuMemAlloc(&m_segment, gm_segment_sz);
			break;

		case GpuMemKind__ManagedMemory:
			rc = cuMemAllocManaged(&m_segment, gm_segment_sz,
								   CU_MEM_ATTACH_GLOBAL);
			break;

		case GpuMemKind__IOMapMemory:
			rc = cuMemAlloc(&m_segment, gm_segment_sz);
			if (rc == CUDA_SUCCESS)
			{
				StromCmd__MapGpuMemory cmd;

				memset(&cmd, 0, sizeof(StromCmd__MapGpuMemory));
				cmd.vaddress = m_segment;
				cmd.length = gm_segment_sz;
				if (nvme_strom_ioctl(STROM_IOCTL__MAP_GPU_MEMORY, &cmd) == 0)
					gm_seg->iomap_handle = cmd.handle;
				else
				{
					wnotice("failed on STROM_IOCTL__MAP_GPU_MEMORY: %m");
					cuMemFree(m_segment);
					rc = CUDA_ERROR_MAP_FAILED;
				}
			}
			break;

		case GpuMemKind__HostMemory:
			rc = cuMemHostAlloc((void **)&m_segment, gm_segment_sz,
								CU_MEMHOSTALLOC_PORTABLE);
			break;

		default:
			rc = CUDA_ERROR_INVALID_VALUE;
			break;
	}
	cuCtxPopCurrent(NULL);

	if (rc != CUDA_SUCCESS)
	{
		free(gm_seg);
		pthreadRWLockUnlock(&gcontext->gm_rwlock);
		return rc;
	}
	/* setup of GpuMemSegment */
	gm_seg->gm_kind		= gm_kind;
	gm_seg->m_segment	= m_segment;
	SpinLockInit(&gm_seg->lock);
	for (i=0; i <= GPUMEM_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&gm_seg->free_chunks[i]);

	if (gm_kind == GpuMemKind__ManagedMemory)
	{
		cl_int	i, __mclass = GPUMEM_CHUNKSZ_MAX_BIT;
		size_t	segment_usage = 0;

		while (segment_usage < gm_segment_sz &&
			   __mclass >= GPUMEM_CHUNKSZ_MIN_BIT)
		{
			if (segment_usage + (1UL << __mclass) > gm_segment_sz)
				__mclass--;
			else
			{
				i = (segment_usage >> GPUMEM_CHUNKSZ_MIN_BIT);
				Assert(i < nchunks);
				gm_chunk = &gm_seg->gm_chunks[i];
				gm_chunk->mclass = __mclass;
				dlist_push_tail(&gm_seg->free_chunks[__mclass],
								&gm_chunk->chain);
				segment_usage += (1UL << __mclass);
			}
		}
		Assert(segment_usage == gm_segment_sz);
	}
	else
	{
		for (i=0; i < nchunks; i++)
		{
			gm_chunk = &gm_seg->gm_chunks[i];
			dlist_push_tail(&gm_seg->free_chunks[mclass],
							&gm_chunk->chain);
		}
	}

	/* track by GpuContext */
	if (!trackGpuMem(gcontext, m_segment, gm_seg,
					 filename, lineno))
	{
		if (gm_kind != GpuMemKind__HostMemory)
			cuMemFree(m_segment);
		else
			cuMemFreeHost((void *)m_segment);
		free(gm_seg);
		pthreadRWLockUnlock(&gcontext->gm_rwlock);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	dlist_push_head(gm_segment_list, &gm_seg->chain);

	/* update statistics */
	gm_stat = &gm_stat_array[gcontext->cuda_dindex];
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			pg_atomic_add_fetch_u64(&gm_stat->normal_usage, gm_segment_sz);
			break;
		case GpuMemKind__ManagedMemory:
			pg_atomic_add_fetch_u64(&gm_stat->managed_usage, gm_segment_sz);
			break;
		case GpuMemKind__IOMapMemory:
			pg_atomic_add_fetch_u64(&gm_stat->iomap_usage, gm_segment_sz);
			break;
		default:
			break;
	}
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
	cl_int		mclass = get_next_log2(pgstrom_chunk_size());

	if (bytesize != pgstrom_chunk_size())
		return __gpuMemAllocRaw(gcontext,
								p_deviceptr,
								bytesize,
								filename, lineno);
	return gpuMemAllocChunk(GpuMemKind__NormalMemory,
							gcontext, p_deviceptr, mclass,
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
	cl_int		mclass = get_next_log2(pgstrom_chunk_size());

	/* not supported at this moment */
	if (bytesize != pgstrom_chunk_size())
		return CUDA_ERROR_INVALID_VALUE;
	return gpuMemAllocChunk(GpuMemKind__IOMapMemory,
                            gcontext, p_deviceptr, mclass,
							filename, lineno);
}

/*
 * __gpuMemAllocManaged
 */
CUresult
__gpuMemAllocManaged(GpuContext *gcontext,
					 CUdeviceptr *p_deviceptr,
					 size_t bytesize,
					 int flags,
					 const char *filename, int lineno)
{
	cl_int		mclass = get_next_log2(bytesize);

	if (flags != CU_MEM_ATTACH_GLOBAL ||
		bytesize > gm_segment_sz / 2)
		return __gpuMemAllocManagedRaw(gcontext,
									   p_deviceptr,
									   bytesize,
									   flags,
									   filename, lineno);
	return gpuMemAllocChunk(GpuMemKind__NormalMemory,
							gcontext, p_deviceptr, mclass,
							filename, lineno);
}

/*
 * __gpuMemAllocHost
 */
CUresult
__gpuMemAllocHost(GpuContext *gcontext,
				  void **p_hostptr,
				  size_t bytesize,
				  const char *filename, int lineno)
{
	cl_int		mclass = get_next_log2(pgstrom_chunk_size());
	CUdeviceptr	tempptr;
	CUresult	rc;

	if (bytesize != pgstrom_chunk_size())
		return CUDA_ERROR_INVALID_VALUE;

	rc = gpuMemAllocChunk(GpuMemKind__HostMemory,
						  gcontext, &tempptr, mclass,
						  filename, lineno);
	if (rc == CUDA_SUCCESS)
		*p_hostptr = (void *)tempptr;
	return rc;
}

/*
 * pgstrom_gpu_mmgr_init_gpucontext - Per GpuContext initialization
 */
void
pgstrom_gpu_mmgr_init_gpucontext(GpuContext *gcontext)
{
	pthreadRWLockInit(&gcontext->gm_rwlock);
	dlist_init(&gcontext->gm_normal_list);
	dlist_init(&gcontext->gm_iomap_list);
	dlist_init(&gcontext->gm_managed_list);
	dlist_init(&gcontext->gm_hostmem_list);
}

/*
 * pgstrom_gpu_mmgr_cleanup_gpucontext - Per GpuContext cleanup
 */
void
pgstrom_gpu_mmgr_cleanup_gpucontext(GpuContext *gcontext)
{
	GpuMemStatistics *gm_stat = &gm_stat_array[gcontext->cuda_dindex];
	GpuMemSegment  *gm_seg;
	dlist_node	   *dnode;

	while (!dlist_is_empty(&gcontext->gm_normal_list))
	{
		dnode = dlist_pop_head_node(&gcontext->gm_normal_list);
		gm_seg = dlist_container(GpuMemSegment, chain, dnode);
		pg_atomic_sub_fetch_u64(&gm_stat->normal_usage, gm_segment_sz);
		free(gm_seg);
	}

	while (!dlist_is_empty(&gcontext->gm_managed_list))
	{
		dnode = dlist_pop_head_node(&gcontext->gm_managed_list);
		gm_seg = dlist_container(GpuMemSegment, chain, dnode);
		pg_atomic_sub_fetch_u64(&gm_stat->managed_usage, gm_segment_sz);
		free(gm_seg);
	}

	while (!dlist_is_empty(&gcontext->gm_iomap_list))
	{
		dnode = dlist_pop_head_node(&gcontext->gm_iomap_list);
		gm_seg = dlist_container(GpuMemSegment, chain, dnode);
		pg_atomic_sub_fetch_u64(&gm_stat->iomap_usage, gm_segment_sz);
		free(gm_seg);
	}

	while (!dlist_is_empty(&gcontext->gm_hostmem_list))
	{
		dnode = dlist_pop_head_node(&gcontext->gm_hostmem_list);
		gm_seg = dlist_container(GpuMemSegment, chain, dnode);
		free(gm_seg);
	}
}

/*
 * pgstrom_startup_gpu_mmgr
 */
static void
pgstrom_startup_gpu_mmgr(void)
{
	size_t		required;
	bool		found;
	int			i;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	/*
	 * GpuMemStatistics
	 */
	required = STROMALIGN(sizeof(GpuMemStatistics) * numDevAttrs);
	gm_stat_array = ShmemInitStruct("GPU Device Memory Statistics",
									required, &found);
	if (found)
		elog(ERROR, "Bug? GPU Device Memory Statistics already exist");
	memset(gm_stat_array, 0, required);
	for (i=0; i < numDevAttrs; i++)
		gm_stat_array[i].total_size = devAttrs[i].DEV_TOTAL_MEMSZ;
}

/*
 * pgstrom_init_gpu_mmgr
 */
void
pgstrom_init_gpu_mmgr(void)
{
	long		sysconf_pagesize;		/* _SC_PAGESIZE */
	long		sysconf_phys_pages;		/* _SC_PHYS_PAGES */
	Size		shared_buffer_size = (Size)NBuffers * (Size)BLCKSZ;
	Size		segment_sz;
	Size		required;

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
	required = STROMALIGN(sizeof(GpuMemStatistics) * numDevAttrs);
	RequestAddinShmemSpace(required);
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_mmgr;
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
	StromCmd__MemCopySsdToGpu cmd;
	GpuMemSegment  *gm_seg;
	BlockNumber	   *block_nums;
	void		   *block_data;
	size_t			offset;
	size_t			length;
	cl_uint			nr_loaded;
	CUresult		rc;

	/* ensure the @m_kds is exactly i/o mapped buffer */
	Assert(gcontext != NULL);
	gm_seg = lookupGpuMem(gcontext, m_kds);
	if (!gm_seg ||
		gm_seg->gm_kind != GpuMemKind__IOMapMemory ||
		gm_seg->iomap_handle == 0UL)
		werror("nvme-strom: invalid device pointer");
	Assert(pds->kds.format == KDS_FORMAT_BLOCK);
	Assert(m_kds >= gm_seg->m_segment &&
		   m_kds + pds->kds.length <= gm_seg->m_segment + gm_segment_sz);
	offset = m_kds - gm_seg->m_segment;

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

//	if (iomap_gpu_memory_size_kb == 0 || !nvme_strom_enabled)
	if (!nvme_strom_enabled)
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
