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
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "utils/guc.h"
#include "utils/memutils.h"
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
} GpuMemKind;

typedef struct
{
	dlist_node		chain;
	cl_int			mclass;
	cl_int			refcnt;
} GpuMemChunk;

#define GPUMEMCHUNK_IS_FREE(chunk)					\
	((chunk)->chain.prev != UINT_MAX &&				\
	 (chunk)->chain.next != UINT_MAX &&				\
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&	\
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT &&	\
	 (chunk)->refcnt == 0)
#define GPUMEMCHUNK_IS_ACTIVE(chunk)				 \
	((chunk)->chain.prev == UINT_MAX &&				 \
	 (chunk)->chain.next == UINT_MAX &&				 \
	 (chunk)->mclass >= GPUMEM_CHUNKSZ_MIN_BIT &&	 \
	 (chunk)->mclass <= GPUMEM_CHUNKSZ_MAX_BIT &&	 \
	 (chunk)->refcnt > 0)

struct GpuMemSegmentMap;

/* shared structure */
typedef struct
{
	struct GpuMemSegmentMap *gm_smap;/* reference to private (never changed) */
	size_t			segment_sz;	/* segment size (never changed) */
	dlist_node		chain;		/* link to segment list */
	GpuMemKind		gm_kind;
	pg_atomic_uint32 mapcount;	/* # of GpuContext that maps this segment */
	CUipcMemHandle	m_handle;	/* IPC handler of the device memory */
	unsigned long	iomap_handle; /* only if GpuMemKind__IOMapMemory */
	slock_t			lock;		/* protection of free_chunks[] */
	dlist_head		free_chunks[GPUMEM_CHUNKSZ_MAX_BIT + 1];
	GpuMemChunk		gm_chunks[FLEXIBLE_ARRAY_MEMBER];
} GpuMemSegmentDesc;

/* shared structure (per device) */
struct GpuMemSegmentContext
{
	/* interaction backend <-> mmgr */
	pthread_mutex_t	mutex;
	pthread_cond_t	cond;
	cl_uint			alloc_request;			/* in: bitmap of GpuMemKind */
	cl_uint			revision;				/* out: revision of the retcode */
	CUresult		status_alloc_normal;	/* out: mmgr -> backend */
	CUresult		status_alloc_managed;	/* out: mmgr -> backend */
	CUresult		status_alloc_iomap;		/* out: mmgr -> backend */
	Latch		   *serverLatch;
	/* management of device memory segment */
	pthread_rwlock_t rwlock;
	dlist_head		normal_segment_list;
	dlist_head		managed_segment_list;
	dlist_head		iomap_segment_list;
};
typedef struct GpuMemSegmentContext	GpuMemSegmentContext;

/* shared structure (system global) */
struct GpuMemSegmentHead
{
	slock_t			lock;		/* protection of free_segment_list */
	dlist_head		free_segment_list;
	GpuMemSegmentContext gm_scxt_array[FLEXIBLE_ARRAY_MEMBER];
};
typedef struct GpuMemSegmentHead	GpuMemSegmentHead;

/* per-process structure */
struct GpuMemSegmentMap
{
	GpuMemSegmentDesc *gm_sdesc;	/* reference to the shared portion */
	dlist_node		chain;			/* link to local_xxx_segment_list */
	CUdeviceptr		m_segment;		/* device pointer */
};
typedef struct GpuMemSegmentMap		GpuMemSegmentMap;

/* static variables */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static GpuMemSegmentHead *gm_shead = NULL;
static int			gpu_memory_segment_size_kb;	/* GUC */
static int			num_gpu_memory_segments;	/* GUC */
static int			iomap_gpu_memory_size_kb;	/* GUC */

/* process local structure */
static GpuMemSegmentMap *gm_smap_array = NULL;
static pthread_rwlock_t	local_segment_map_rwlock;
static dlist_head	   *local_normal_segment_lists;
static dlist_head	   *local_managed_segment_lists;
static dlist_head	   *local_iomap_segment_lists;




static bool			gpu_mmgr_got_sigterm = false;
static int			gpu_mmgr_cuda_dindex = -1;
static CUdevice		gpu_mmgr_cuda_device;
static CUcontext	gpu_mmgr_cuda_context;

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
 * gpuMemAllocRaw
 */
CUresult
__gpuMemAllocRaw(GpuContext *gcontext,
				 CUdeviceptr *p_devptr,
				 size_t bytesize,
				 const char *filename, int lineno)
{
	return CUDA_SUCCESS;
}

/*
 * gpuMemAllocManagedRaw
 */
CUresult
__gpuMemAllocManagedRaw(GpuContext *gcontext,
						CUdeviceptr *p_devptr,
						size_t bytesize,
						int flags,
						const char *filename, int lineno)
{
	return CUDA_SUCCESS;
}

/*
 * gpuMemSplitChunk
 */
static bool
gpuMemSplitChunk(GpuMemSegmentDesc *gm_sdesc, int mclass)
{
	GpuMemChunk	   *gm_chunk1;
	GpuMemChunk	   *gm_chunk2;
	dlist_node	   *dnode;
	long			offset;

	if (mclass > GPUMEM_CHUNKSZ_MAX_BIT)
		return false;
	Assert(mclass > GPUMEM_CHUNKSZ_MIN_BIT);
	if (dlist_is_empty(&gm_sdesc->free_chunks[mclass]))
	{
		if (!gpuMemSplitChunk(gm_sdesc, mclass + 1))
			return false;
	}
	Assert(!dlist_is_empty(&gm_sdesc->free_chunks[mclass]));
	offset = 1UL << (mclass - 1 - GPUMEM_CHUNKSZ_MIN_BIT);
	dnode = dlist_pop_head_node(&gm_sdesc->free_chunks[mclass]);
	gm_chunk1 = dlist_container(GpuMemChunk, chain, dnode);
	gm_chunk2 = gm_chunk1 + offset;
	Assert(GPUMEMCHUNK_IS_FREE(gm_chunk1));
	Assert(gm_chunk2->mclass == 0);
	Assert(gm_chunk1->refcnt == 0 && gm_chunk2->refcnt == 0);
	gm_chunk1->mclass = mclass - 1;
	gm_chunk2->mclass = mclass - 1;

	dlist_push_tail(&gm_sdesc->free_chunks[mclass - 1],
					&gm_chunk1->chain);
	dlist_push_tail(&gm_sdesc->free_chunks[mclass - 1],
					&gm_chunk2->chain);
	return true;
}

/*
 * gpuMemAllocChunk - pick up a chunk from the supplied segment
 *
 * NOTE: caller must hold gm_sdesc->lock
 */
static inline GpuMemChunk *
gpuMemTryAllocChunk(GpuMemSegmentDesc *gm_sdesc, int mclass)
{
	GpuMemChunk	   *gm_chunk;
	dlist_node	   *dnode;

	if (dlist_is_empty(&gm_sdesc->free_chunks[mclass]))
	{
		/* split larger chunk */
		if (!gpuMemSplitChunk(gm_sdesc, mclass+1))
			return NULL;
	}
	Assert(!dsm_list_is_empty(&gm_sdesc->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&gm_sdesc->free_chunks[mclass]);
	gm_chunk = dlist_container(GpuMemChunk, chain, dnode);
	Assert(GPUMEMCHUNK_IS_FREE(gm_chunk) &&
		   gm_chunk->mclass == mclass);
	gm_chunk->refcnt++;

	return gm_chunk;
}

/*
 * gpuMemAllocCommon
 */
static CUresult
gpuMemAllocCommon(GpuMemKind gm_kind,
				  GpuContext *gcontext,
				  CUdeviceptr *p_devptr,
				  cl_int mclass,
				  bool may_alloc_segment,
				  const char *filename, int lineno)
{
	cl_int			dindex = gcontext->gpuserv_id; //TOBEFIXED
	GpuMemSegmentContext *gm_scxt = &gm_shead->gm_scxt_array[dindex];
	GpuMemSegmentDesc *gm_sdesc;
	GpuMemSegmentMap *gm_smap;
	GpuMemChunk	   *gm_chunk;
	dlist_iter		iter;
	dlist_head	   *local_segment_list;
	dlist_head	   *global_segment_list;
	CUresult	   *p_alloc_status;
	CUresult		rc;
	cl_uint			revision;
	bool			has_exclusive_lock = false;

	/* only backend can call */
	Assert(gpu_mmgr_cuda_dindex < 0);

	/* try to lookup local segments; already mapped */
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			local_segment_list = &local_normal_segment_lists[dindex];
			global_segment_list = &gm_scxt->normal_segment_list;
			p_alloc_status = &gm_scxt->status_alloc_normal;
			break;
		case GpuMemKind__ManagedMemory:
			local_segment_list = &local_managed_segment_lists[dindex];
			global_segment_list = &gm_scxt->managed_segment_list;
			p_alloc_status = &gm_scxt->status_alloc_managed;
			break;
		case GpuMemKind__IOMapMemory:
			local_segment_list = &local_iomap_segment_lists[dindex];
			global_segment_list = &gm_scxt->iomap_segment_list;
			p_alloc_status = &gm_scxt->status_alloc_iomap;
			break;
		default:
			elog(PANIC, "Bug? unknown GpuMemKind: %d", (int)gm_kind);
	}

	/*
	 * Try to lookup locally mapped segments
	 */
	pthreadRWLockReadLock(&local_segment_map_rwlock);
retry:
	dlist_foreach(iter, local_segment_list)
	{
		gm_smap = dlist_container(GpuMemSegmentMap, chain, iter.cur);
		gm_sdesc = gm_smap->gm_sdesc;
		gm_chunk = gpuMemTryAllocChunk(gm_sdesc, mclass);
		if (gm_chunk)
		{
			cl_long		i, nchunks __attribute__((unused));

			pthreadRWLockUnlock(&local_segment_map_rwlock);
			nchunks = gm_sdesc->segment_sz >> GPUMEM_CHUNKSZ_MIN_BIT;
			Assert(gm_chunk >= gm_sdesc->gm_chunks &&
				   gm_chunk < (gm_sdesc->gm_chunks + nchunks));
			i = gm_chunk - gm_sdesc->gm_chunks;
			*p_devptr = gm_smap->m_segment + (i << GPUMEM_CHUNKSZ_MIN_BIT);

			//TODO: track gm_chunk with GpuContext

			return CUDA_SUCCESS;
		}
	}

	if (!has_exclusive_lock)
	{
		pthreadRWLockUnlock(&local_segment_map_rwlock);
		has_exclusive_lock = true;
		pthreadRWLockWriteLock(&local_segment_map_rwlock);
		goto retry;
	}

	/*
	 * try to lookup segments already allocated, but not mapped locally.
	 */
	pthreadRWLockReadLock(&gm_scxt->rwlock);
	dlist_foreach(iter, global_segment_list)
	{
		gm_sdesc = dlist_container(GpuMemSegmentDesc, chain, iter.cur);
		gm_smap = gm_sdesc->gm_smap;
		if (gm_smap->m_segment != 0UL)
			continue;
		Assert(gm_sdesc->gm_kind == gm_kind);
		Assert(!gm_smap->chain.prev && !gm_smap->chain.next);
		rc = cuIpcOpenMemHandle(&gm_smap->m_segment,
								gm_sdesc->m_handle,
								0);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gm_scxt->rwlock);
			pthreadRWLockUnlock(&local_segment_map_rwlock);
			return rc;
		}
		pg_atomic_fetch_add_u32(&gm_sdesc->mapcount, 1);
		pthreadRWLockUnlock(&gm_scxt->rwlock);

		dlist_push_tail(local_segment_list, &gm_smap->chain);
		goto retry;
	}
	pthreadRWLockUnlock(&gm_scxt->rwlock);
	pthreadRWLockUnlock(&local_segment_map_rwlock);
	has_exclusive_lock = false;

	/* Give up, if no additional segments */
	if (!may_alloc_segment)
		return CUDA_ERROR_OUT_OF_MEMORY;

	/*
	 * Here is no space left on the existing device memory segment.
	 * So, raise a request to allocate a new one.
	 */
	pthreadMutexLock(&gm_scxt->mutex);
	if (!gm_scxt->serverLatch)
	{
		pthreadMutexUnlock(&gm_scxt->mutex);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	revision = gm_scxt->revision;
	gm_scxt->alloc_request |= gm_kind;
	SetLatch(gm_scxt->serverLatch);
	pthreadCondWait(&gm_scxt->cond, &gm_scxt->mutex);
	rc = (revision == gm_scxt->revision
		  ? CUDA_SUCCESS
		  : *p_alloc_status);
	pthreadMutexUnlock(&gm_scxt->mutex);
	if (rc != CUDA_SUCCESS)
		return rc;

	pthreadRWLockReadLock(&local_segment_map_rwlock);
	goto retry;
}

/*
 * gpuMemAlloc
 */
CUresult
__gpuMemAlloc(GpuContext *gcontext,
			  CUdeviceptr *p_devptr,
			  size_t bytesize,
			  const char *filename, int lineno)
{
	cl_int		mclass;

	if (bytesize > ((size_t)gpu_memory_segment_size_kb << 9))
		return __gpuMemAllocRaw(gcontext,
								p_devptr,
								bytesize,
								filename, lineno);
	mclass = get_next_log2(bytesize);
	if (mclass < GPUMEM_CHUNKSZ_MIN_BIT)
		mclass = GPUMEM_CHUNKSZ_MIN_BIT;
	else if (mclass > GPUMEM_CHUNKSZ_MAX_BIT)
		return CUDA_ERROR_OUT_OF_MEMORY;

	return gpuMemAllocCommon(GpuMemKind__NormalMemory,
							 gcontext, p_devptr, mclass, true,
							 filename, lineno);
}

/*
 * gpuMemAllocManaged
 */
CUresult
__gpuMemAllocManaged(GpuContext *gcontext,
					 CUdeviceptr *p_devptr,
					 size_t bytesize,
					 int flags,
					 const char *filename, int lineno)
{
	cl_int		mclass;

	/* only CU_MEM_ATTACH_GLOBAL is supported */
	if (flags != CU_MEM_ATTACH_GLOBAL)
		return CUDA_ERROR_NOT_SUPPORTED;
	if (bytesize > ((size_t)gpu_memory_segment_size_kb << 9))
		__gpuMemAllocManagedRaw(gcontext,
								p_devptr,
								bytesize,
								flags,
								filename, lineno);

	mclass = get_next_log2(bytesize);
	if (mclass < GPUMEM_CHUNKSZ_MIN_BIT)
		mclass = GPUMEM_CHUNKSZ_MIN_BIT;
	else if (mclass > GPUMEM_CHUNKSZ_MAX_BIT)
		return CUDA_ERROR_OUT_OF_MEMORY;

	return gpuMemAllocCommon(GpuMemKind__ManagedMemory,
							 gcontext, p_devptr, mclass, true,
							 filename, lineno);
}

/*
 * gpuMemFree_v2
 */
CUresult
gpuMemFree_v2(GpuContext *gcontext, CUdeviceptr devptr)
{
	//lookup tracker
	//put gm_chunk
	//if refcnt==0, back to the free_chunks
	//merge chunks, if available

	return CUDA_SUCCESS;
}

/*
 * gpu_mmgr_alloc_normal_segment - segment allocator for NormalMemory
 */
static CUresult
__gpu_mmgr_alloc_segment(GpuMemSegmentContext *gm_scxt,
						 GpuMemSegmentDesc *gm_sdesc,
						 GpuMemKind gm_kind)
{
	GpuMemSegmentMap *gm_smap = gm_sdesc->gm_smap;
	GpuMemChunk	   *gm_chunk;
	CUdeviceptr		m_segment = 0UL;
	CUresult		rc;
	size_t			segment_usage;
	int				i, mclass;

	/* GpuMemSegmentMap must be unmapped */
	Assert(gm_smap->gm_sdesc == gm_sdesc &&
		   gm_smap->chain.prev == NULL &&
		   gm_smap->chain.next == NULL &&
		   gm_smap->dsm_seg == NULL &&
		   gm_smap->m_segment == 0UL);

	/* init fields */
	gm_sdesc->gm_kind = gm_kind;
	pg_atomic_init_u32(&gm_sdesc->mapcount, 0);
	SpinLockInit(&gm_sdesc->lock);
	memset(&gm_sdesc->m_handle, 0, sizeof(CUipcMemHandle));
	gm_sdesc->iomap_handle = 0;
	for (i=0; i <= GPUMEM_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&gm_sdesc->free_chunks[i]);

	/* allocation device memory */
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			rc = cuMemAlloc(&m_segment, gm_sdesc->segment_sz);
			if (rc != CUDA_SUCCESS)
			{
				elog(LOG, "GPU Mmgr: failed on cuMemAlloc: %s",
					 errorText(rc));
				return rc;
			}
			break;

		case GpuMemKind__ManagedMemory:
			rc = cuMemAllocManaged(&m_segment, gm_sdesc->segment_sz,
								   CU_MEM_ATTACH_GLOBAL);
			if (rc != CUDA_SUCCESS)
			{
				elog(LOG, "GPU Mmgr: failed on cuMemAllocManaged: %s",
					 errorText(rc));
				return rc;
			}
			break;

		case GpuMemKind__IOMapMemory:
			rc = cuMemAlloc(&m_segment, gm_sdesc->segment_sz);
			if (rc != CUDA_SUCCESS)
			{
				elog(LOG, "GPU Mmgr: failed on cuMemAlloc: %s",
					 errorText(rc));
				return rc;
			}
			else
			{
				StromCmd__MapGpuMemory cmd;

				memset(&cmd, 0, sizeof(StromCmd__MapGpuMemory));
				cmd.vaddress = m_segment;
				cmd.length = gm_sdesc->segment_sz;
				if (nvme_strom_ioctl(STROM_IOCTL__MAP_GPU_MEMORY, &cmd) != 0)
				{
					elog(LOG, "STROM_IOCTL__MAP_GPU_MEMORY failed: %m");
					cuMemFree(m_segment);
					return CUDA_ERROR_MAP_FAILED;
				}
				gm_sdesc->iomap_handle = cmd.handle;
			}
			break;

		default:
			elog(PANIC, "Bug? unknown GpuMemKind: %d", (int)gm_kind);
	}
	/* IPC handler */
	rc = cuIpcGetMemHandle(&gm_sdesc->m_handle, m_segment);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "GPU Mmgr: failed on cuIpcGetMemHandle: %s",
			 errorText(rc));
		cuMemFree(m_segment);
		return rc;
	}

	/* Setup free chunks */
	mclass = GPUMEM_CHUNKSZ_MAX_BIT;
	segment_usage = 0;
	while (segment_usage < gm_sdesc->segment_sz &&
		   mclass >= GPUMEM_CHUNKSZ_MIN_BIT)
	{
		if (segment_usage + (1UL << mclass) > gm_sdesc->segment_sz)
			mclass--;
		else
		{
			i = (segment_usage >> GPUMEM_CHUNKSZ_MIN_BIT);
			gm_chunk = &gm_sdesc->gm_chunks[i];
			dlist_push_tail(&gm_sdesc->free_chunks[mclass],
							&gm_chunk->chain);
			segment_usage += (1UL << mclass);
		}
	}
	Assert(segment_usage == gm_sdesc->segment_sz);

	/* setup GpuMemSegmentMap for the GPU Mmgr process */
	pthreadRWLockWriteLock(&local_segment_map_rwlock);
	gm_smap->m_segment	= m_segment;
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			dlist_push_tail(&local_normal_segment_lists[gpu_mmgr_cuda_dindex],
							&gm_smap->chain);
			break;
		case GpuMemKind__ManagedMemory:
			dlist_push_tail(&local_managed_segment_lists[gpu_mmgr_cuda_dindex],
							&gm_smap->chain);
			break;
		case GpuMemKind__IOMapMemory:
			dlist_push_tail(&local_iomap_segment_lists[gpu_mmgr_cuda_dindex],
							&gm_smap->chain);
			break;
		default:
			elog(FATAL, "Bug? unknown GpuMemKind: %d", (int)gm_kind);
	}
	pthreadRWLockUnlock(&local_segment_map_rwlock);

	/* segment becomes ready to use for backend processes */
	pthreadRWLockWriteLock(&gm_scxt->rwlock);
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			dlist_push_tail(&gm_scxt->normal_segment_list,
							&gm_sdesc->chain);
			break;
		case GpuMemKind__ManagedMemory:
			dlist_push_tail(&gm_scxt->managed_segment_list,
							&gm_sdesc->chain);
			break;
		case GpuMemKind__IOMapMemory:
			dlist_push_tail(&gm_scxt->iomap_segment_list,
							&gm_sdesc->chain);
			break;
		default:
			elog(PANIC, "Bug? unknown GpuMemKind: %d", (int)gm_kind);
	}
	pthreadRWLockUnlock(&gm_scxt->rwlock);

	return CUDA_SUCCESS;
}

static CUresult
gpu_mmgr_alloc_segment(GpuMemSegmentContext *gm_scxt,
					   GpuMemKind gm_kind)
{
	GpuMemSegmentDesc  *gm_sdesc = NULL;
	CUresult	rc;

	SpinLockAcquire(&gm_shead->lock);
	if (!dlist_is_empty(&gm_shead->free_segment_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gm_shead->free_segment_list);
		gm_sdesc = dlist_container(GpuMemSegmentDesc, chain, dnode);
	}
	SpinLockRelease(&gm_shead->lock);

	if (!gm_sdesc)
		return CUDA_ERROR_OUT_OF_MEMORY;
	rc = __gpu_mmgr_alloc_segment(gm_scxt, gm_sdesc, gm_kind);
	if (rc != CUDA_SUCCESS)
	{
		SpinLockAcquire(&gm_shead->lock);
		dlist_push_head(&gm_shead->free_segment_list,
						&gm_sdesc->chain);
		SpinLockRelease(&gm_shead->lock);
	}
	return rc;
}

/*
 * gpu_mmgr_reclaim_segment
 */
static void
gpu_mmgr_reclaim_segment(GpuMemSegmentContext *gm_scxt)
{
	static int			reclaim_rr = 0;
	GpuMemSegmentDesc  *gm_sdesc;
	GpuMemSegmentMap   *gm_smap;
	dlist_head		   *segment_list;
	dlist_iter			iter;
	CUresult			rc;
	int					loop;

	pthreadRWLockWriteLock(&gm_scxt->rwlock);
	for (loop=0; loop < 2; loop++)
	{
		segment_list = (reclaim_rr++ % 2 == 0
						? &gm_scxt->normal_segment_list
						: &gm_scxt->managed_segment_list);

		dlist_foreach(iter, segment_list)
		{
			gm_sdesc = dlist_container(GpuMemSegmentDesc,
									   chain, iter.cur);
			Assert(gm_sdesc->gm_kind == GpuMemKind__NormalMemory ||
				   gm_sdesc->gm_kind == GpuMemKind__ManagedMemory);
			if (pg_atomic_read_u32(&gm_sdesc->mapcount) == 0)
			{
				gm_smap = gm_sdesc->gm_smap;
				Assert(gm_smap->chain.prev != NULL &&
					   gm_smap->chain.next != NULL &&
					   gm_smap->gm_sdesc == gm_sdesc &&
					   gm_smap->dsm_seg != NULL &&
					   gm_smap->m_segment != 0UL);
				dlist_delete(&gm_sdesc->chain);
				dlist_delete(&gm_smap->chain);
				pthreadRWLockUnlock(&gm_scxt->rwlock);

				/* release resources */
				Assert(gm_smap->iomap_handle == 0UL);
				rc = cuMemFree(gm_smap->m_segment);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "GPU Mmgr: failed on cuMemFree: %s",
						 errorText(rc));

				memset(&gm_smap->chain, 0, sizeof(dlist_node));
				gm_smap->m_segment = 0UL;
				memset(&gm_sdesc->m_handle, 0, sizeof(CUipcMemHandle));
				gm_sdesc->iomap_handle = 0;

				/* segment can be reused */
				SpinLockAcquire(&gm_shead->lock);
				dlist_push_head(&gm_shead->free_segment_list,
								&gm_sdesc->chain);
				SpinLockRelease(&gm_shead->lock);
				return;
			}
		}
	}
	pthreadRWLockUnlock(&gm_scxt->rwlock);
}

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
gpu_mmgr_event_loop(GpuMemSegmentContext *gm_scxt)
{
	int		ev = 0;

	while (!gpu_mmgr_got_sigterm)
	{
		cl_uint	alloc_request;
		int		nstatus = 0;
		int		mstatus = 0;

		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();

		if (ev & WL_TIMEOUT)
			gpu_mmgr_reclaim_segment(gm_scxt);

		pthreadMutexLock(&gm_scxt->mutex);
		alloc_request = gm_scxt->alloc_request;
		gm_scxt->alloc_request = 0;
		gm_scxt->revision++;
		pthreadMutexUnlock(&gm_scxt->mutex);

		if ((alloc_request & GpuMemKind__NormalMemory) != 0)
			nstatus = gpu_mmgr_alloc_segment(gm_scxt,
											 GpuMemKind__NormalMemory);
		if ((alloc_request & GpuMemKind__ManagedMemory) != 0)
			mstatus = gpu_mmgr_alloc_segment(gm_scxt,
											 GpuMemKind__ManagedMemory);

		pthreadMutexLock(&gm_scxt->mutex);
		/* write back error status, if any */
		if ((alloc_request & GpuMemKind__NormalMemory) != 0)
			gm_scxt->status_alloc_normal	= nstatus;
		if ((alloc_request & GpuMemKind__ManagedMemory) != 0)
			gm_scxt->status_alloc_managed	= mstatus;
		/* clear the requests already allocated */
		gm_scxt->alloc_request &= ~alloc_request;
		/* wake up any of the waiting backends  */
		pthreadCondBroadcast(&gm_scxt->cond);
		pthreadMutexUnlock(&gm_scxt->mutex);

		ev = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_TIMEOUT |
					   WL_POSTMASTER_DEATH,
					   10 * 1000);		/* wake up per 10sec */

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
	GpuMemSegmentContext *gm_scxt;
	CUresult	rc;

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
	gm_scxt = &gm_shead->gm_scxt_array[gpu_mmgr_cuda_dindex];

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
	if (iomap_gpu_memory_size_kb == 0)
		Assert(dlist_is_empty(&gm_scxt->iomap_segment_list));
	else
	{
		dlist_node *dnode = dlist_pop_head_node(&gm_scxt->iomap_segment_list);
		GpuMemSegmentDesc *gm_sdesc = dlist_container(GpuMemSegmentDesc,
													  chain, dnode);
		Assert(dlist_is_empty(&gm_scxt->iomap_segment_list));
		rc = __gpu_mmgr_alloc_segment(gm_scxt,
									  gm_sdesc,
									  GpuMemKind__IOMapMemory);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on allocation of I/O map memory: %s",
				 errorText(rc));
	}

	/* enables to accept device memory allocation request */
	pthreadMutexLock(&gm_scxt->mutex);
	gm_scxt->serverLatch = MyLatch;
	pthreadMutexUnlock(&gm_scxt->mutex);

	/* event loop of GPU Mmgr */
	PG_TRY();
	{
		gpu_mmgr_event_loop(gm_scxt);
	}
	PG_CATCH();
	{
		pthreadMutexLock(&gm_scxt->mutex);
		gm_scxt->serverLatch = NULL;
		pthreadCondBroadcast(&gm_scxt->cond);
		pthreadMutexUnlock(&gm_scxt->mutex);
		PG_RE_THROW();
	}
	PG_END_TRY();

	pthreadMutexLock(&gm_scxt->mutex);
	gm_scxt->serverLatch = NULL;
	pthreadCondBroadcast(&gm_scxt->cond);
	pthreadMutexUnlock(&gm_scxt->mutex);

	elog(ERROR, "GPU Mmgr%d [%s] normally terminated",
		 gpu_mmgr_cuda_dindex,
		 devAttrs[gpu_mmgr_cuda_dindex].DEV_NAME);
}

/*
 * pgstrom_startup_gpu_mmgr
 */
static void
pgstrom_startup_gpu_mmgr(void)
{
	GpuMemSegmentHead *gm_shead;
	GpuMemSegmentDesc *gm_sdesc;
	GpuMemSegmentMap  *gm_smap;
	Size	nchunks1;
	Size	nchunks2;
	Size	head_sz;
	Size	unitsz1;
	Size	unitsz2;
	Size	required;
	bool	found;
	int		i, j = 0;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/*
	 * GpuMemSegmentMap (local structure)
	 */
	gm_smap_array = calloc(num_gpu_memory_segments + numDevAttrs,
						   sizeof(GpuMemSegmentMap));
	if (!gm_smap_array)
		elog(ERROR, "out of memory");

	/*
	 * GpuMemSegmentDesc (shared structure)
	 */
	nchunks1 = ((size_t)gpu_memory_segment_size_kb << 10) / GPUMEM_CHUNKSZ_MIN;
	nchunks2 = ((size_t)iomap_gpu_memory_size_kb << 10) / GPUMEM_CHUNKSZ_MIN;
	head_sz = STROMALIGN(offsetof(GpuMemSegmentHead,
								  gm_scxt_array[numDevAttrs]));
	unitsz1 = STROMALIGN(offsetof(GpuMemSegmentDesc,
								  gm_chunks[nchunks1]));
	unitsz2 = STROMALIGN(offsetof(GpuMemSegmentDesc,
                                  gm_chunks[nchunks2]));
	required = (head_sz + 
				unitsz1 * num_gpu_memory_segments +
				unitsz2 * numDevAttrs);
	gm_shead = ShmemInitStruct("GPU Device Memory Management Structure",
							   required, &found);
	if (found)
		elog(ERROR, "Bug? GPU Device Memory Management Structure exists");
	memset(gm_shead, 0, required);

	SpinLockInit(&gm_shead->lock);
	dlist_init(&gm_shead->free_segment_list);
	for (i=0; i < numDevAttrs; i++)
	{
		GpuMemSegmentContext *gm_scxt = &gm_shead->gm_scxt_array[i];

		pthreadMutexInit(&gm_scxt->mutex);
		pthreadCondInit(&gm_scxt->cond);
		pthreadRWLockInit(&gm_scxt->rwlock);
		dlist_init(&gm_scxt->normal_segment_list);
		dlist_init(&gm_scxt->managed_segment_list);
		dlist_init(&gm_scxt->iomap_segment_list);
	}

	gm_sdesc = (GpuMemSegmentDesc *)((char *)gm_shead + head_sz);
	for (i=0; i < num_gpu_memory_segments; i++)
	{
		gm_smap = &gm_smap_array[j++];
		gm_smap->gm_sdesc = gm_sdesc;
		gm_sdesc->gm_smap = gm_smap;
		gm_sdesc->segment_sz = (size_t)nchunks1 << GPUMEM_CHUNKSZ_MIN_BIT;
		/* initialization of other fields shall be done on allocation */
		dlist_push_tail(&gm_shead->free_segment_list,
						&gm_sdesc->chain);
		gm_sdesc = (GpuMemSegmentDesc *)((char *)gm_sdesc + unitsz1);
	}

	if (iomap_gpu_memory_size_kb > 0)
	{
		for (i=0; i < numDevAttrs; i++)
		{
			GpuMemSegmentContext *gm_scxt = &gm_shead->gm_scxt_array[i];

			gm_smap = &gm_smap_array[j++];
			gm_smap->gm_sdesc = gm_sdesc;
			gm_sdesc->gm_smap = gm_smap;
			gm_sdesc->segment_sz = (size_t)nchunks2 << GPUMEM_CHUNKSZ_MIN_BIT;
			/* initialization of other fields shall be done on allocation */
			dlist_push_tail(&gm_scxt->iomap_segment_list,
							&gm_sdesc->chain);
			gm_sdesc = (GpuMemSegmentDesc *)((char *)gm_sdesc + unitsz2);
		}
	}
}

/*
 * pgstrom_init_gpu_mmgr
 */
void
pgstrom_init_gpu_mmgr(void)
{
	BackgroundWorker worker;
	Size		length;
	Size		required;
	size_t		nchunks;
	size_t		nchunks_iomap;
	int			i;

	/*
	 * segment size of the device memory in kB
	 */
	DefineCustomIntVariable("pg_strom.gpu_memory_segment_size",
							"default size of the GPU device memory segment",
							NULL,
							&gpu_memory_segment_size_kb,
							(1UL << 20),	/* 1GB */
							(1UL << 17),	/* 128MB */
							GPUMEM_CHUNKSZ_MAX >> 10,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	length = (size_t)gpu_memory_segment_size_kb << 10;
	if ((length & (GPUMEM_CHUNKSZ_MIN - 1)) != 0)
		elog(ERROR, "pg_strom.gpu_memory_segment_size must be multiple number of %zukB", (size_t)(GPUMEM_CHUNKSZ_MIN >> 10));
	nchunks = length >> GPUMEM_CHUNKSZ_MIN_BIT;

	/*
	 * total number of device memory segments
	 */
	DefineCustomIntVariable("pg_strom.gpu_memory_num_segments",
							"number of the GPU device memory segments",
							NULL,
							&num_gpu_memory_segments,
							512 * Max(numDevAttrs, 1),
							32,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	/*
	 * size of the i/o mapped device memory (for NVMe-Strom)
	 */
	DefineCustomIntVariable("pg_strom.gpu_memory_iomap_size",
							"size of I/O mapped GPU device memory",
							NULL,
							&iomap_gpu_memory_size_kb,
							0,
							0,
							GPUMEM_CHUNKSZ_MAX >> 10,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	length = (size_t)iomap_gpu_memory_size_kb << 10;
	if ((length & (GPUMEM_CHUNKSZ_MIN - 1)) != 0)
		elog(ERROR, "pg_strom.gpu_memory_iomap_size must be multiple number of %zukB", (size_t)(GPUMEM_CHUNKSZ_MIN >> 10));
	nchunks_iomap = length >> GPUMEM_CHUNKSZ_MIN_BIT;

	/*
	 * request for the static shared memory
	 */
	required = STROMALIGN(offsetof(GpuMemSegmentHead,
								   gm_scxt_array[numDevAttrs])) +
		STROMALIGN(offsetof(GpuMemSegmentDesc,
							gm_chunks[nchunks])) * num_gpu_memory_segments +
		STROMALIGN(offsetof(GpuMemSegmentDesc,
							gm_chunks[nchunks_iomap]));

	RequestAddinShmemSpace(required);
	shmem_startup_hook_next = shmem_startup_hook;
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

	/*
	 * Misc initialization
	 */
	pthreadRWLockInit(&local_segment_map_rwlock);
	local_normal_segment_lists = malloc(sizeof(dlist_head) * numDevAttrs);
	if (!local_normal_segment_lists)
		elog(ERROR, "out of memory");
	local_managed_segment_lists = malloc(sizeof(dlist_head) * numDevAttrs);
	if (!local_managed_segment_lists)
		elog(ERROR, "out of memory");
	local_iomap_segment_lists = malloc(sizeof(dlist_head) * numDevAttrs);
	if (!local_iomap_segment_lists)
		elog(ERROR, "out of memory");
	for (i=0; i < numDevAttrs; i++)
	{
		dlist_init(&local_normal_segment_lists[i]);
		dlist_init(&local_managed_segment_lists[i]);
		dlist_init(&local_iomap_segment_lists[i]);
	}
}
