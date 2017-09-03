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
#include "dsm_list.h"


#define GPUMEM_CHUNKSZ_MAX_BIT		34		/* 16GB */
#define GPUMEM_CHUNKSZ_MIN_BIT		12		/* 4KB */
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
	dsm_list_node	chain;
	cl_int			mclass;
	cl_int			refcnt;
} GpuMemChunk;

struct GpuMemSegmentMap;

/* shared structure */
typedef struct
{
	struct GpuMemSegmentMap *gm_smap;/* reference to private (never changed) */
	dlist_node		chain;
	GpuMemKind		gm_kind;
	pg_atomic_uint32 mapcount;	/* # of GpuContext which maps this segment */
	dsm_handle		dsm_handle;	/* array to GpuMemChunk */
	CUipcMemHandle	m_handle;	/* IPC handler of the device memory */
	unsigned long	iomap_handle; /* only if GpuMemKind__IOMapMemory */
	size_t			segment_sz;	/* segment size in bytes */
	slock_t			lock;		/* protection for free_chunks[] */
	dsm_list_head	free_chunks[GPUMEM_CHUNKSZ_MAX_BIT + 1];
} GpuMemSegmentDesc;

/* shared structure */
struct GpuMemSegmentHead
{
	pthread_mutex_t	mutex;
	pthread_cond_t	cond;
	int				alloc_new_segment;		/* in: backend -> mmgr */
	int				status_normal_alloc;	/* out: mmgr -> backend */
	int				status_managed_alloc;	/* out: mmgr -> backend */
	int				status_iomap_alloc;		/* out: mmgr -> backend */
	Latch		   *gpuMmgrLatch;
	/* management of device memory */
	pthread_rwlock_t rwlock;
	int				num_normal_segments;
	int				num_managed_segments;
	int				num_iomap_segments;
	dlist_head		normal_segment_list;
	dlist_head		managed_segment_list;
	dlist_head		iomap_segment_list;
	dlist_head		free_segment_list;
	GpuMemSegmentDesc gm_sdesc_array[FLEXIBLE_ARRAY_MEMBER];
};
typedef struct GpuMemSegmentHead	GpuMemSegmentHead;

/* per-process structure */
struct GpuMemSegmentMap
{
	GpuMemSegmentDesc *gm_sdesc;	/* reference to shared resource */
	dlist_node		chain;
	dsm_segment	   *dsm_seg;		/* GpuMemChunk array */
	CUdeviceptr		m_segment;		/* device pointer */
};
typedef struct GpuMemSegmentMap		GpuMemSegmentMap;

/* static variables */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static GpuMemSegmentHead  **gm_shead_array = NULL;
static int			gpu_memory_segment_size_kb;	/* GUC */
static int			num_gpu_memory_segments;	/* GUC */
static int			iomap_gpu_memory_size_kb;	/* GUC */

/* process local structure */
static GpuMemSegmentMap *gm_smap_array = NULL;
static slock_t		local_segment_map_lock;
static dlist_head	local_normal_segment_list;
static dlist_head	local_managed_segment_list;
static dlist_head	local_iomap_segment_list;




static bool			gpu_mmgr_got_sigterm = false;
static CUdevice		gpu_mmgr_cuda_device;
static CUcontext	gpu_mmgr_cuda_context;










/*
 * gpu_mmgr_alloc_normal_segment - segment allocator for NormalMemory
 */
static int
gpu_mmgr_alloc_segment(GpuMemSegmentHead *gm_shead,
					   GpuMemKind gm_kind,
					   size_t segment_sz)
{
	GpuMemSegmentDesc *gm_sdesc = NULL;
	GpuMemSegmentMap *gm_smap;
	GpuMemChunk	   *gm_chunks;
	dlist_node	   *dnode;
	dsm_segment	   *dsm_seg;
	CUdeviceptr		m_segment = 0UL;
	unsigned long	iomap_handle = 0UL;
	size_t			segment_usage = 0;
	size_t			n_chunks;
	int				i, mclass;
	int				status = -ENOMEM;
	CUresult		rc;

	pthreadRWLockWriteLock(&gm_shead->rwlock);
	if (dlist_is_empty(&gm_shead->free_segment_list))
	{
		elog(LOG, "GPU Mmgr: no available free GpuMemSegmentDesc");
		goto error_1;
	}
	dnode = dlist_pop_head_node(&gm_shead->free_segment_list);
	gm_sdesc = dlist_container(GpuMemSegmentDesc, chain, dnode);
	memset(&gm_sdesc->chain, 0, sizeof(dlist_node));

	gm_smap = gm_sdesc->gm_smap;
	Assert(gm_smap->chain.prev == NULL &&
		   gm_smap->chain.next == NULL &&
		   gm_smap->dsm_seg == NULL &&
		   gm_smap->m_segment == 0UL);

	/* DSM allocation of GpuMemChunk array */
	n_chunks = (segment_sz >> GPUMEM_CHUNKSZ_MIN_BIT);
	dsm_seg = dsm_create(sizeof(GpuMemChunk) * n_chunks, 0);
	if (!dsm_seg)
	{
		elog(LOG, "GPU Mmgr: failed on dsm_create");
		goto error_2;
	}
	gm_chunks = dsm_segment_address(dsm_seg);
	memset(gm_chunks, 0, sizeof(GpuMemChunk) * n_chunks);

	/* allocation of device memory */
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			rc = cuMemAlloc(&m_segment, segment_sz);
			if (rc != CUDA_SUCCESS)
			{
				elog(LOG, "GPU Mmgr: failed on cuMemAlloc: %s",
					 errorText(rc));
				status = (int) rc;
				goto error_3;
			}
			break;
		case GpuMemKind__ManagedMemory:
			rc = cuMemAllocManaged(&m_segment, segment_sz,
								   CU_MEM_ATTACH_GLOBAL);
			if (rc != CUDA_SUCCESS)
			{
				elog(LOG, "GPU Mmgr: failed on cuMemAllocManaged: %s",
					 errorText(rc));
				status = (int) rc;
				goto error_3;
			}
			break;
		case GpuMemKind__IOMapMemory:
			rc = cuMemAlloc(&m_segment, segment_sz);
			if (rc == CUDA_SUCCESS)
			{
				//XXX todo map gpu device memory here
			}
			break;
		default:
			elog(FATAL, "Bug? unknown GpuMemKind: %d", (int)gm_kind);
	}
	/* setup GpuMemSegmentDesc */
	gm_sdesc->gm_kind = gm_kind;
	pg_atomic_init_u32(&gm_sdesc->mapcount, 0);
	gm_sdesc->dsm_handle = dsm_segment_handle(dsm_seg);
	rc = cuIpcGetMemHandle(&gm_sdesc->m_handle, m_segment);
	if (rc != CUDA_SUCCESS)
	{
		elog(LOG, "GPU Mmgr: failed on cuIpcGetMemHandle: %s",
			 errorText(rc));
		status = (int) rc;
		goto error_3;
	}
	gm_sdesc->iomap_handle = iomap_handle;
	SpinLockInit(&gm_sdesc->lock);
	for (i=0; i <= GPUMEM_CHUNKSZ_MAX_BIT; i++)
		dsm_list_init(&gm_sdesc->free_chunks[i]);

	mclass = GPUMEM_CHUNKSZ_MAX_BIT;
	while (segment_usage < segment_sz &&
		   mclass >= GPUMEM_CHUNKSZ_MIN_BIT)
	{
		if (segment_usage + (1UL << mclass) > segment_sz)
			mclass--;
		else
		{
			i = (segment_usage >> GPUMEM_CHUNKSZ_MIN_BIT);
			gm_chunks[i].mclass = mclass;
			dsm_list_push_head(dsm_seg,
							   &gm_sdesc->free_chunks[mclass],
							   &gm_chunks[i].chain);
			segment_usage += (1UL << mclass);
		}
	}
	gm_sdesc->segment_sz = segment_usage;

	/* setup GpuMemSegmentMap of the GPU Mmgr process */
	gm_smap->gm_sdesc	= gm_sdesc;
	gm_smap->dsm_seg	= dsm_seg;
	gm_smap->m_segment	= m_segment;

	SpinLockAcquire(&local_segment_map_lock);
	switch (gm_kind)
	{
		case GpuMemKind__NormalMemory:
			dlist_push_tail(&local_normal_segment_list, &gm_smap->chain);
			break;
		case GpuMemKind__ManagedMemory:
			dlist_push_tail(&local_managed_segment_list, &gm_smap->chain);
			break;
		case GpuMemKind__IOMapMemory:
			dlist_push_tail(&local_iomap_segment_list, &gm_smap->chain);
			break;
		default:
			elog(FATAL, "Bug? unknown GpuMemKind: %d", (int)gm_kind);
	}
	SpinLockRelease(&local_segment_map_lock);

	return 0;

error_3:
	memset(gm_smap, 0, sizeof(GpuMemSegmentMap));
	dsm_detach(dsm_seg);
error_2:
	dlist_push_head(&gm_shead->free_segment_list, &gm_sdesc->chain);
error_1:
	pthreadRWLockUnlock(&gm_shead->rwlock);

	return status;
}

/*
 * gpu_mmgr_reclaim_segment
 */
static void
gpu_mmgr_reclaim_segment(GpuMemSegmentHead *gm_shead)
{
	static int			reclaim_rr = 0;
	GpuMemSegmentDesc  *gm_sdesc;
	GpuMemSegmentMap   *gm_smap;
	dlist_head		   *segment_list;
	dlist_iter			iter;
	CUresult			rc;
	int					loop;

	pthreadRWLockWriteLock(&gm_shead->rwlock);
	for (loop=0; loop < 2; loop++)
	{
		segment_list = (reclaim_rr++ % 2 == 0
						? &gm_shead->normal_segment_list
						: &gm_shead->managed_segment_list);

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

				/* release resources */
				dsm_detach(gm_smap->dsm_seg);
				rc = cuMemFree(gm_smap->m_segment);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "GPU Mmgr: failed on cuMemFree: %s",
						 errorText(rc));

				/* reset */
				gm_sdesc->dsm_handle	= 0;
				memset(&gm_sdesc->m_handle, 0, sizeof(CUipcMemHandle));
				gm_sdesc->iomap_handle	= 0;

				memset(&gm_smap->chain, 0, sizeof(dlist_node));
				gm_smap->dsm_seg		= NULL;
				gm_smap->m_segment		= 0UL;

				dlist_push_head(&gm_shead->free_segment_list,
								&gm_sdesc->chain);
				goto out;
			}
		}
	}
out:
	pthreadRWLockUnlock(&gm_shead->rwlock);
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
 * gpu_mmgr_bgworker_main
 */
static void
gpu_mmgr_bgworker_main(Datum cuda_dindex)
{
	GpuMemSegmentHead *gm_shead = gm_shead_array[cuda_dindex];
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

	/* ensure not to use MPS daemon */
	setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1);

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	rc = cuDeviceGet(&gpu_mmgr_cuda_device,
					 devAttrs[cuda_dindex].DEV_ID);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&gpu_mmgr_cuda_context,
					 CU_CTX_SCHED_AUTO,
					 gpu_mmgr_cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxCreate: %s", errorText(rc));

	/* allows to accept memory allocation request */
	pthreadMutexLock(&gm_shead->mutex);
	gm_shead->gpuMmgrLatch = MyLatch;
	pthreadMutexUnlock(&gm_shead->mutex);

	/* event loop of GPU Mmgr */
	PG_TRY();
	{
		int		ev = 0;
		int		alloc_new_segment;
		size_t	segment_size = (size_t)gpu_memory_segment_size_kb << 10;

		while (!gpu_mmgr_got_sigterm)
		{
			int		nstatus = 0;
			int		mstatus = 0;
			int		istatus = 0;

			ResetLatch(MyLatch);

			CHECK_FOR_INTERRUPTS();

			if (ev & WL_TIMEOUT)
				gpu_mmgr_reclaim_segment(gm_shead);

			pthreadMutexLock(&gm_shead->mutex);
			alloc_new_segment = gm_shead->alloc_new_segment;
			gm_shead->alloc_new_segment = 0;
			pthreadMutexUnlock(&gm_shead->mutex);

			if ((alloc_new_segment & GpuMemKind__NormalMemory) != 0)
				nstatus = gpu_mmgr_alloc_segment(gm_shead,
												 GpuMemKind__NormalMemory,
												 segment_size);
			if ((alloc_new_segment & GpuMemKind__ManagedMemory) != 0)
				mstatus = gpu_mmgr_alloc_segment(gm_shead,
												 GpuMemKind__ManagedMemory,
												 segment_size);
			if ((alloc_new_segment & GpuMemKind__IOMapMemory) != 0)
				istatus = gpu_mmgr_alloc_segment(gm_shead,
												 GpuMemKind__IOMapMemory,
												 segment_size);

			pthreadMutexLock(&gm_shead->mutex);
			/* write back error status, if any */
			if ((alloc_new_segment & GpuMemKind__NormalMemory) != 0)
				gm_shead->status_normal_alloc	= nstatus;
			if ((alloc_new_segment & GpuMemKind__ManagedMemory) != 0)
				gm_shead->status_managed_alloc	= mstatus;
			if ((alloc_new_segment & GpuMemKind__IOMapMemory) != 0)
				gm_shead->status_iomap_alloc	= istatus;
			/* clear the requests already allocated */
			gm_shead->alloc_new_segment &= ~alloc_new_segment;
			/* wake up any of the waiting backends  */
			if ((errno = pthread_cond_broadcast(&gm_shead->cond)) != 0)
				elog(FATAL, "failed on pthread_cond_broadcast: %m");
			pthreadMutexUnlock(&gm_shead->mutex);

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   15 * 1000);		/* wake up per 15sec */

			/* emergency bailout if postmaster gets dead */
			if (ev & WL_POSTMASTER_DEATH)
				elog(FATAL, "emergency bailout due to unexpected postmaster dead");
		}
	}
	PG_CATCH();
	{
		pthreadMutexLock(&gm_shead->mutex);
		gm_shead->gpuMmgrLatch = NULL;
		if ((errno = pthread_cond_broadcast(&gm_shead->cond)) != 0)
			elog(FATAL, "failed on pthread_cond_broadcast: %m");
		pthreadMutexUnlock(&gm_shead->mutex);
		PG_RE_THROW();
	}
	PG_END_TRY();

	pthreadMutexLock(&gm_shead->mutex);
	gm_shead->gpuMmgrLatch = NULL;
	if ((errno = pthread_cond_broadcast(&gm_shead->cond)) != 0)
		elog(FATAL, "failed on pthread_cond_broadcast: %m");
	pthreadMutexUnlock(&gm_shead->mutex);

	elog(ERROR, "GPU Mmgr%d [%s] normally terminated",
		 (int)cuda_dindex, devAttrs[cuda_dindex].DEV_NAME);
}

/*
 * pgstrom_startup_gpu_mmgr
 */
static void
pgstrom_startup_gpu_mmgr(void)
{
	GpuMemSegmentHead *gm_shead;
	Size	required;
	bool	found;
	int		i, j, k;
	pthread_condattr_t condattr;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();


	/*
	 * GpuMemSegmentMap (local structure)
	 */
	gm_smap_array = calloc(num_gpu_memory_segments * numDevAttrs,
						   sizeof(GpuMemSegmentMap));
	if (!gm_smap_array)
		elog(ERROR, "out of memory");

	/*
	 * GpuMemSegmentDesc (shared structure)
	 */
	gm_shead_array = malloc(sizeof(GpuMemSegmentHead) * numDevAttrs);
	if (!gm_shead_array)
		elog(ERROR, "out of memory");

	required = STROMALIGN(offsetof(GpuMemSegmentHead,
								   gm_sdesc_array[num_gpu_memory_segments]));
	gm_shead = ShmemInitStruct("GPU Device Memory Segment Head",
							   required * numDevAttrs, &found);
	if (found)
		elog(FATAL, "Bug? GPU Device Memory Segment Head already exists");
	memset(gm_shead, 0, required * numDevAttrs);

	if ((errno = pthread_condattr_init(&condattr)) != 0)
		elog(FATAL, "failed on pthread_condattr_init: %m");
	if ((errno = pthread_condattr_setpshared(&condattr, 1)) != 0)
		elog(FATAL, "failed on pthread_condattr_setpshared: %m");

	for (i=0, k=0; i < numDevAttrs; i++)
	{
		pthreadMutexInit(&gm_shead->mutex);
		if ((errno = pthread_cond_init(&gm_shead->cond, &condattr)) != 0)
			elog(FATAL, "failed on pthread_cond_init: %m");
		gm_shead->gpuMmgrLatch = NULL;

		pthreadRWLockInit(&gm_shead->rwlock);
		dlist_init(&gm_shead->normal_segment_list);
		dlist_init(&gm_shead->managed_segment_list);
		dlist_init(&gm_shead->iomap_segment_list);
		dlist_init(&gm_shead->free_segment_list);

		for (j=0; j < num_gpu_memory_segments; j++)
		{
			GpuMemSegmentMap   *gm_smap = &gm_smap_array[k++];
			GpuMemSegmentDesc  *gm_sdesc = &gm_shead->gm_sdesc_array[j];

			gm_smap->gm_sdesc = gm_sdesc;
			gm_sdesc->gm_smap = gm_smap;
			dlist_push_tail(&gm_shead->free_segment_list, &gm_smap->chain);
		}
		gm_shead_array[i] = gm_shead;
		gm_shead = (GpuMemSegmentHead *)((char *)gm_shead + required);
	}
	if ((errno = pthread_condattr_destroy(&condattr)) != 0)
		elog(FATAL, "failed on pthread_condattr_destroy: %m");
}

/*
 * pgstrom_init_gpu_mmgr
 */
void
pgstrom_init_gpu_mmgr(void)
{
	BackgroundWorker worker;
	Size		required;
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
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);

	/*
	 * total number of device memory segments
	 */
	DefineCustomIntVariable("pg_strom.gpu_memory_num_segments",
							"number of the GPU device memory segments",
							NULL,
							&num_gpu_memory_segments,
							1024,
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
							0,		/* not used */
							0,		/* not used */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);

	/*
	 * request for the static shared memory
	 */
	required = STROMALIGN(offsetof(GpuMemSegmentHead,
								   gm_sdesc_array[num_gpu_memory_segments]));
	RequestAddinShmemSpace(numDevAttrs * required);
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
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/*
	 * Misc initialization
	 */
	SpinLockInit(&local_segment_map_lock);
	dlist_init(&local_normal_segment_list);
	dlist_init(&local_managed_segment_list);
	dlist_init(&local_iomap_segment_list);
}
