/*
 * gpu_service.c
 *
 * A background worker process that handles any interactions with GPU
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"
/*
 * gpuContext / gpuMemory
 */
typedef struct
{
	pthread_mutex_t	lock;
	size_t			total_sz;	/* total pool size */
	size_t			hard_limit;
	size_t			keep_limit;
	dlist_head		segment_list;
} gpuMemoryPool;

struct gpuContext
{
	dlist_node		chain;
	int				serv_fd;		/* for accept(2) */
	int				cuda_dindex;
	CUdevice		cuda_device;
	CUcontext		cuda_context;
	CUmodule		cuda_module;
	HTAB		   *cuda_type_htab;
	HTAB		   *cuda_func_htab;
	xpu_encode_info *cuda_encode_catalog;
	gpuMemoryPool	pool;
	/* GPU client */
	pthread_mutex_t	client_lock;
	dlist_head		client_list;
	/* XPU commands */
	pthread_cond_t	cond;
	pthread_mutex_t	lock;
	dlist_head		command_list;
	/* GPU task workers */
	int				n_workers;
	pthread_t		workers[1];
};
typedef struct gpuContext	gpuContext;
#define SizeOfGpuContext	\
	offsetof(gpuContext, workers[pgstrom_max_async_gpu_tasks])

/*
 * variables
 */
int		pgstrom_max_async_gpu_tasks;	/* GUC */
__thread int		CU_DINDEX_PER_THREAD = -1;
__thread CUdevice	CU_DEVICE_PER_THREAD = -1;
__thread CUcontext	CU_CONTEXT_PER_THREAD = NULL;
__thread CUevent	CU_EVENT_PER_THREAD = NULL;
static __thread gpuContext *GpuWorkerCurrentContext = NULL;
static volatile int		gpuserv_bgworker_got_signal = 0;
static dlist_head		gpuserv_gpucontext_list;
static int				gpuserv_epoll_fdesc = -1;
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static pg_atomic_uint32 *gpuserv_debug_output = NULL;
static bool				__gpuserv_debug_output_dummy;

#define __GpuServDebug(fmt,...)										\
	do {															\
		if (*((volatile bool *)gpuserv_debug_output))				\
		{															\
			const char *filename = __FILE__;						\
			for (const char *pos = filename; *pos != '\0'; pos++)	\
			{														\
				if (pos[0] == '/' && pos[1] != '\0')				\
					filename = pos + 1;								\
			}														\
			fprintf(stderr, "GpuServ: " fmt " (%s:%d)\n",			\
					##__VA_ARGS__, filename, __LINE__);				\
		}															\
	} while(0)

static void
gpuserv_debug_output_assign(bool newval, void *extra)
{
	if (gpuserv_debug_output)
		pg_atomic_write_u32(gpuserv_debug_output, newval ? 1 : 0);
	else
		__gpuserv_debug_output_dummy = (newval ? 1 : 0);
}

static const char *
gpuserv_debug_output_show(void)
{
	if (gpuserv_debug_output)
		return (pg_atomic_read_u32(gpuserv_debug_output) != 0 ? "on" : "off");

	return (__gpuserv_debug_output_dummy ? "on" : "off");
}

/*
 * cuStrError
 */
const char *
cuStrError(CUresult rc)
{
	static __thread char buffer[300];
	const char	   *err_name;

	/* is it cufile error? */
	if ((int)rc > CUFILEOP_BASE_ERR)
		return cufileop_status_error((CUfileOpError)rc);
	if (cuGetErrorName(rc, &err_name) == CUDA_SUCCESS)
		return err_name;
	snprintf(buffer, sizeof(buffer), "Unknown CUDA Error (%d)", (int)rc);
	return buffer;
}

/* ----------------------------------------------------------------
 *
 * GPU Memory Allocator
 *
 * ----------------------------------------------------------------
 */
static int		pgstrom_gpu_mempool_segment_sz_kb;	/* GUC */
static double	pgstrom_gpu_mempool_max_ratio;		/* GUC */
static double	pgstrom_gpu_mempool_min_ratio;		/* GUC */
static int		pgstrom_gpu_mempool_release_delay;	/* GUC */
typedef struct
{
	dlist_node		chain;
	size_t			segment_sz;
	size_t			active_sz;		/* == 0 can be released */
	CUdeviceptr		devptr;
	dlist_head		free_chunks;	/* list of free chunks */
	dlist_head		addr_chunks;	/* list of ordered chunks */
	struct timeval	tval;
} gpuMemorySegment;

typedef struct
{
	dlist_node		free_chain;
	dlist_node		addr_chain;
	gpuMemorySegment *mseg;
	gpuMemChunk		c;
} gpuMemoryChunkInternal;

static gpuMemoryChunkInternal *
__gpuMemAllocFromSegment(gpuMemoryPool *pool,
						 gpuMemorySegment *mseg,
						 size_t bytesize)
{
	gpuMemoryChunkInternal *chunk;
	gpuMemoryChunkInternal *buddy;
	dlist_iter		iter;

	dlist_foreach(iter, &mseg->free_chunks)
	{
		chunk = dlist_container(gpuMemoryChunkInternal,
								free_chain, iter.cur);
		if (bytesize <= chunk->c.__length__)
		{
			size_t	surplus = chunk->c.__length__ - bytesize;

			/* try to split, if free chunk is enough large (>4MB) */
			if (surplus > (4UL << 20))
			{
				buddy = calloc(1, sizeof(gpuMemoryChunkInternal));
				if (!buddy)
					return NULL;	/* out of memory */
				chunk->c.__length__ -= surplus;

				buddy->mseg   = mseg;
				buddy->c.__base__ = mseg->devptr;
				buddy->c.__offset__ = chunk->c.__offset__ + chunk->c.__length__;
				buddy->c.__length__ = surplus;
				buddy->c.m_devptr = (buddy->c.__base__ + buddy->c.__offset__);
				dlist_insert_after(&chunk->free_chain, &buddy->free_chain);
				dlist_insert_after(&chunk->addr_chain, &buddy->addr_chain);
			}
			/* mark it as an active chunk */
			dlist_delete(&chunk->free_chain);
			memset(&chunk->free_chain, 0, sizeof(dlist_node));
			mseg->active_sz += chunk->c.__length__;

			/* update the LRU ordered segment list and timestamp */
			gettimeofday(&mseg->tval, NULL);
			dlist_move_head(&pool->segment_list, &mseg->chain);

			return chunk;
		}
	}
	return NULL;
}

static gpuMemorySegment *
__gpuMemAllocNewSegment(gpuMemoryPool *pool, size_t segment_sz)
{
	gpuMemorySegment *mseg = calloc(1, sizeof(gpuMemorySegment));
	gpuMemoryChunkInternal *chunk = calloc(1, sizeof(gpuMemoryChunkInternal));
	CUresult	rc;

	if (!mseg || !chunk)
		goto error;
	mseg->segment_sz = segment_sz;
	mseg->active_sz = 0;
	dlist_init(&mseg->free_chunks);
	dlist_init(&mseg->addr_chunks);

	rc = cuMemAlloc(&mseg->devptr, mseg->segment_sz);
	if (rc != CUDA_SUCCESS)
		goto error;
	if (!gpuDirectMapGpuMemory(mseg->devptr,
							   mseg->segment_sz))
		goto error;

	chunk->mseg   = mseg;
	chunk->c.__base__ = mseg->devptr;
	chunk->c.__offset__ = 0;
	chunk->c.__length__ = segment_sz;
	chunk->c.m_devptr = (chunk->c.__base__ + chunk->c.__offset__);
	dlist_push_head(&mseg->free_chunks, &chunk->free_chain);
	dlist_push_head(&mseg->addr_chunks, &chunk->addr_chain);

	dlist_push_head(&pool->segment_list, &mseg->chain);
	pool->total_sz += segment_sz;

	return mseg;
error:
	if (mseg->devptr)
		cuMemFree(mseg->devptr);
	if (mseg)
		free(mseg);
	if (chunk)
		free(chunk);
	return NULL;
}

static gpuMemChunk *
__gpuMemAlloc(size_t bytesize)
{
	gpuMemoryPool  *pool = &GpuWorkerCurrentContext->pool;
	dlist_iter		iter;
	size_t			segment_sz;
	gpuMemoryChunkInternal *chunk = NULL;

	bytesize = PAGE_ALIGN(bytesize);
	pthreadMutexLock(&pool->lock);
	dlist_foreach(iter, &pool->segment_list)
	{
		gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
												 chain, iter.cur);
		if (mseg->active_sz + bytesize <= mseg->segment_sz)
		{
			chunk = __gpuMemAllocFromSegment(pool, mseg, bytesize);
			if (chunk)
				goto out_unlock;
		}
	}
	segment_sz = ((size_t)pgstrom_gpu_mempool_segment_sz_kb << 10);
	if (segment_sz < bytesize)
		segment_sz = bytesize;
	if (pool->total_sz + segment_sz <= pool->hard_limit)
	{
		gpuMemorySegment *mseg = __gpuMemAllocNewSegment(pool, segment_sz);

		if (mseg)
			chunk = __gpuMemAllocFromSegment(pool, mseg, bytesize);
	}
out_unlock:	
	pthreadMutexUnlock(&pool->lock);

	return (chunk ? &chunk->c : NULL);
}

const gpuMemChunk *
gpuMemAlloc(size_t bytesize)
{
	return __gpuMemAlloc(bytesize);
}

void
gpuMemFree(const gpuMemChunk *__chunk)
{
	gpuMemoryPool  *pool = &GpuWorkerCurrentContext->pool;
	gpuMemorySegment *mseg;
	gpuMemoryChunkInternal *chunk;
	gpuMemoryChunkInternal *buddy;
	dlist_node	   *dnode;

	chunk = (gpuMemoryChunkInternal *)
		((char *)__chunk - offsetof(gpuMemoryChunkInternal, c));
	Assert(!chunk->free_chain.prev && !chunk->free_chain.next);

	pthreadMutexLock(&pool->lock);
	/* revert this chunk state to 'free' */
	mseg = chunk->mseg;
	mseg->active_sz -= chunk->c.__length__;
	dlist_push_head(&mseg->free_chunks,
					&chunk->free_chain);

	/* try merge if next chunk is also free */
	if (dlist_has_next(&mseg->addr_chunks,
					   &chunk->addr_chain))
	{
		dnode = dlist_next_node(&mseg->addr_chunks,
								&chunk->addr_chain);
		buddy = dlist_container(gpuMemoryChunkInternal,
								addr_chain, dnode);
		if (buddy->free_chain.prev && buddy->addr_chain.next)
		{
			Assert(chunk->c.__offset__ +
				   chunk->c.__length__ == buddy->c.__offset__);
			dlist_delete(&buddy->free_chain);
			dlist_delete(&buddy->addr_chain);
			chunk->c.__length__ += buddy->c.__length__;
			free(buddy);
		}
	}
	/* try merge if prev chunk is also free */
	if (dlist_has_prev(&mseg->addr_chunks,
					   &chunk->addr_chain))
	{
		dnode = dlist_prev_node(&mseg->addr_chunks,
								&chunk->addr_chain);
		buddy = dlist_container(gpuMemoryChunkInternal,
								addr_chain, dnode);
		/* merge if prev chunk is also free */
		if (buddy->free_chain.prev && buddy->addr_chain.next)
		{
			Assert(buddy->c.__offset__ +
				   buddy->c.__length__ == chunk->c.__offset__);
			dlist_delete(&chunk->free_chain);
			dlist_delete(&chunk->addr_chain);
			buddy->c.__length__ += chunk->c.__length__;
			free(chunk);
		}
	}
	/* update the LRU ordered segment list and timestamp */
	gettimeofday(&mseg->tval, NULL);
	dlist_move_head(&pool->segment_list, &mseg->chain);
	pthreadMutexUnlock(&pool->lock);
}

static void
gpuMemoryPoolMaintenance(gpuContext *gcontext)
{
	gpuMemoryPool  *pool = &gcontext->pool;
	dlist_iter		iter;
	struct timeval	tval;
	int64			tdiff;
	CUresult		rc;

	if (!pthreadMutexTryLock(&pool->lock))
		return;
	if (pool->total_sz > pool->keep_limit)
	{
		gettimeofday(&tval, NULL);
		dlist_reverse_foreach(iter, &pool->segment_list)
		{
			gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
													 chain, iter.cur);
			/* still in active? */
			if (mseg->active_sz != 0)
				continue;

			/* enough time to release is elapsed? */
			tdiff = ((tval.tv_sec  - mseg->tval.tv_sec)  * 1000 +
					 (tval.tv_usec - mseg->tval.tv_usec) / 1000);
			if (tdiff < pgstrom_gpu_mempool_release_delay)
				continue;

			/* ok, this segment should be released */
			if (!gpuDirectUnmapGpuMemory(mseg->devptr))
				__FATAL("failed on gpuDirectUnmapGpuMemory");
			rc = cuMemFree(mseg->devptr);
			if (rc != CUDA_SUCCESS)
				__FATAL("failed on cuMemFree: %s", cuStrError(rc));
			/* detach segment */
			dlist_delete(&mseg->chain);
			while (!dlist_is_empty(&mseg->addr_chunks))
			{
				dlist_node	   *dnode = dlist_pop_head_node(&mseg->addr_chunks);
				gpuMemoryChunkInternal *chunk;

				chunk = dlist_container(gpuMemoryChunkInternal,
										addr_chain, dnode);
				Assert(chunk->free_chain.prev &&
					   chunk->free_chain.next);
				free(chunk);
			}
			__GpuServDebug("GPU-%d: i/o mapped device memory %lu bytes released",
						   gcontext->cuda_dindex, mseg->segment_sz);
			Assert(pool->total_sz >= mseg->segment_sz);
			pool->total_sz -= mseg->segment_sz;
			free(mseg);
			break;
		}
	}
	pthreadMutexUnlock(&pool->lock);
}

static void
gpuMemoryPoolInit(gpuMemoryPool *pool, size_t dev_total_memsz)
{
	pthreadMutexInit(&pool->lock);
	pool->total_sz = 0;
	pool->hard_limit = pgstrom_gpu_mempool_max_ratio * (double)dev_total_memsz;
	pool->keep_limit = pgstrom_gpu_mempool_min_ratio * (double)dev_total_memsz;
	dlist_init(&pool->segment_list);
}

/* ----------------------------------------------------------------
 *
 * Session buffer support routines
 *
 * This buffer is used by GpuJoin's inner buffer and GpuPreAgg.
 * It is kept until session end, and can be shared by multiple sessions.
 *
 * ----------------------------------------------------------------
 */
struct gpuQueryBuffer
{
	dlist_node		chain;
	int				refcnt;
	volatile int	phase;			/*  0: not initialized,
									 *  1: buffer is ready,
									 * -1: error, during buffer setup */
	uint64_t		buffer_id;		/* unique buffer id */
	int				cuda_dindex;	/* GPU device identifier */
	CUdeviceptr		m_kmrels;		/* GpuJoin inner buffer (device) */
	void		   *h_kmrels;		/* GpuJoin inner buffer (host) */
	size_t			kmrels_sz;		/* GpuJoin inner buffer size */
	CUdeviceptr		m_kds_final;	/* GpuPreAgg final buffer (device) */
	size_t			m_kds_final_length;	/* length of GpuPreAgg final buffer */
	pthread_rwlock_t m_kds_final_rwlock;  /* RWLock for the final buffer */
};
typedef struct gpuQueryBuffer		gpuQueryBuffer;

#define GPU_QUERY_BUFFER_NSLOTS		320
static dlist_head		gpu_query_buffer_hslot[GPU_QUERY_BUFFER_NSLOTS];
static pthread_mutex_t	gpu_query_buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	gpu_query_buffer_cond = PTHREAD_COND_INITIALIZER;

static void
__putGpuQueryBufferNoLock(gpuQueryBuffer *gq_buf)
{
	Assert(gq_buf->refcnt > 0);
	if (--gq_buf->refcnt == 0)
	{
		CUresult	rc;

		if (gq_buf->m_kmrels)
		{
			rc = cuMemFree(gq_buf->m_kmrels);
			if (rc != CUDA_SUCCESS)
				__GpuServDebug("failed on cuMemFree: %s", cuStrError(rc));
		}
		if (gq_buf->h_kmrels)
		{
			if (munmap(gq_buf->h_kmrels,
					   gq_buf->kmrels_sz) != 0)
				__GpuServDebug("failed on munmap: %m");
		}
		if (gq_buf->m_kds_final)
		{
			rc = cuMemFree(gq_buf->m_kds_final);
			if (rc != CUDA_SUCCESS)
				__GpuServDebug("failed on cuMemFree: %s", cuStrError(rc));
		}
		dlist_delete(&gq_buf->chain);
		free(gq_buf);
	}
}

static void
putGpuQueryBuffer(gpuQueryBuffer *gq_buf)
{
	pthreadMutexLock(&gpu_query_buffer_mutex);
	__putGpuQueryBufferNoLock(gq_buf);
	pthreadMutexUnlock(&gpu_query_buffer_mutex);
}

static bool
__setupGpuQueryJoinGiSTIndexBuffer(gpuContext *gcontext,
								   gpuQueryBuffer *gq_buf,
								   char *errmsg, size_t errmsg_sz)
{
	kern_multirels *h_kmrels = gq_buf->h_kmrels;
	CUfunction	f_prep_gist = NULL;
	CUresult	rc;
	int			grid_sz;
	int			block_sz;
	unsigned int shmem_sz;
	void	   *kern_args[10];
	bool		has_gist = false;

	for (int depth=1; depth <= h_kmrels->num_rels; depth++)
	{
		if (h_kmrels->chunks[depth-1].gist_offset == 0)
			continue;
		if (!f_prep_gist)
		{
			rc = cuModuleGetFunction(&f_prep_gist,
									 gcontext->cuda_module,
									 "gpujoin_prep_gistindex");
			if (rc != CUDA_SUCCESS)
			{
				snprintf(errmsg, errmsg_sz,
						 "failed on cuModuleGetFunction: %s", cuStrError(rc));
				return false;
			}
			rc = gpuOptimalBlockSize(&grid_sz,
									 &block_sz,
									 &shmem_sz,
									 f_prep_gist,
									 0, 0);
			if (rc != CUDA_SUCCESS)
			{
				snprintf(errmsg, errmsg_sz,
						 "failed on gpuOptimalBlockSize: %s", cuStrError(rc));
				return false;
			}
		}
		kern_args[0] = &gq_buf->m_kmrels;
		kern_args[1] = &depth;
		rc = cuLaunchKernel(f_prep_gist,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							shmem_sz,
							CU_STREAM_PER_THREAD,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
		{
			snprintf(errmsg, errmsg_sz,
					 "failed on cuLaunchKernel: %s", cuStrError(rc));
			return false;
		}
		has_gist = true;
	}

	if (has_gist)
	{
		rc = cuEventRecord(CU_EVENT_PER_THREAD, CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			snprintf(errmsg, errmsg_sz,
					 "failed on cuEventRecord: %s", cuStrError(rc));
			return false;
		}
		rc = cuEventSynchronize(CU_EVENT_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			snprintf(errmsg, errmsg_sz,
					 "failed on cuEventSynchronize: %s", cuStrError(rc));
			return false;
		}
	}
	return true;
}

static bool
__setupGpuQueryJoinInnerBuffer(gpuContext *gcontext,
							   gpuQueryBuffer *gq_buf,
							   uint32_t kmrels_handle,
							   char *errmsg, size_t errmsg_sz)
{
	kern_multirels *h_kmrels;
	CUdeviceptr	m_kmrels;
	CUresult	rc;
	int			fdesc;
	struct stat	stat_buf;
	char		namebuf[100];
	size_t		mmap_sz;

	if (kmrels_handle == 0)
		return true;

	snprintf(namebuf, sizeof(namebuf),
			 ".pgstrom_shmbuf_%u_%d",
			 PostPortNumber, kmrels_handle);
	fdesc = shm_open(namebuf, O_RDWR, 0600);
	if (fdesc < 0)
	{
		snprintf(errmsg, errmsg_sz,
				 "failed on shm_open('%s'): %m", namebuf);
		return false;
	}
	if (fstat(fdesc, &stat_buf) != 0)
	{
		snprintf(errmsg, errmsg_sz,
				 "failed on fstat('%s'): %m", namebuf);
		close(fdesc);
		return false;
	}
	mmap_sz = PAGE_ALIGN(stat_buf.st_size);

	h_kmrels = mmap(NULL, mmap_sz,
					PROT_READ | PROT_WRITE,
					MAP_SHARED,
					fdesc, 0);
	close(fdesc);
	if (h_kmrels == MAP_FAILED)
	{
		snprintf(errmsg, errmsg_sz,
				 "failed on mmap('%s', %zu): %m", namebuf, mmap_sz);
		return false;
	}

	rc = cuMemAllocManaged(&m_kmrels, mmap_sz,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(errmsg, errmsg_sz,
				 "failed on cuMemAllocManaged: %s", cuStrError(rc));
		munmap(h_kmrels, mmap_sz);
		return false;
	}
	memcpy((void *)m_kmrels, h_kmrels, mmap_sz);
	(void)cuMemPrefetchAsync(m_kmrels, mmap_sz,
							 CU_DEVICE_PER_THREAD,
							 CU_STREAM_PER_THREAD);
	gq_buf->m_kmrels = m_kmrels;
	gq_buf->h_kmrels = h_kmrels;
	gq_buf->kmrels_sz = mmap_sz;

	/* preparation of GiST-index buffer, if any */
	if (!__setupGpuQueryJoinGiSTIndexBuffer(gcontext, gq_buf,
											errmsg, errmsg_sz))
	{
		cuMemFree(m_kmrels);
		munmap(h_kmrels, mmap_sz);
		return false;
	}
	return true;
}

static bool
__setupGpuQueryGroupByBuffer(gpuContext *gcontext,
							 gpuQueryBuffer *gq_buf,
							 kern_data_store *kds_final_head,
							 char *errmsg, size_t errmsg_sz)
{
	CUdeviceptr	m_kds_final;
	CUresult	rc;

	if (!kds_final_head)
		return true;	/* nothing to do */

	Assert(KDS_HEAD_LENGTH(kds_final_head) <= kds_final_head->length);
	rc =  cuMemAllocManaged(&m_kds_final,
							kds_final_head->length,
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		snprintf(errmsg, errmsg_sz,
				 "failed on cuMemAllocManaged(%zu): %s",
				 kds_final_head->length, cuStrError(rc));
		return false;
	}
	memcpy((void *)m_kds_final,
		   kds_final_head,
		   KDS_HEAD_LENGTH(kds_final_head));
	(void)cuMemPrefetchAsync(m_kds_final,
							 KDS_HEAD_LENGTH(kds_final_head),
							 CU_DEVICE_PER_THREAD,
							 CU_STREAM_PER_THREAD);
	gq_buf->m_kds_final = m_kds_final;
	gq_buf->m_kds_final_length = kds_final_head->length;
	pthreadRWLockInit(&gq_buf->m_kds_final_rwlock);

	return true;
}

/*
 * __expandGpuQueryGroupByBuffer
 */
static bool
__expandGpuQueryGroupByBuffer(gpuQueryBuffer *gq_buf,
							  size_t kds_length_last)
{
	assert(kds_length_last != 0);	/* must be 2nd or later trial */
	pthreadRWLockWriteLock(&gq_buf->m_kds_final_rwlock);
	if (gq_buf->m_kds_final_length == kds_length_last)
	{
		kern_data_store *kds_old = (kern_data_store *)gq_buf->m_kds_final;
		kern_data_store *kds_new;
		CUdeviceptr		m_devptr;
		CUresult		rc;
		size_t			sz, length;

		assert(kds_old->length == gq_buf->m_kds_final_length);
		length = kds_old->length + Min(kds_old->length, 1UL<<30);
		rc = cuMemAllocManaged(&m_devptr, length,
							   CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gq_buf->m_kds_final_rwlock);
			return false;
		}
		kds_new = (kern_data_store *)m_devptr;

		/* early half */
		sz = (KDS_HEAD_LENGTH(kds_old) +
			  MAXALIGN(sizeof(uint32_t) * (kds_old->nitems +
										   kds_old->hash_nslots)));
		memcpy(kds_new, kds_old, sz);
		kds_new->length = length;

		/* later falf */
		sz = __kds_unpack(kds_old->usage);
		memcpy((char *)kds_new + kds_new->length - sz,
			   (char *)kds_old + kds_old->length - sz, sz);

		/* swap them */
		fprintf(stderr, "kds_final expand: %lu => %lu\n",
				kds_old->length, kds_new->length);
		cuMemFree(gq_buf->m_kds_final);
		gq_buf->m_kds_final = m_devptr;
		gq_buf->m_kds_final_length = length;
	}
	pthreadRWLockUnlock(&gq_buf->m_kds_final_rwlock);

	return true;
}

static gpuQueryBuffer *
getGpuQueryBuffer(gpuContext *gcontext,
				  uint64_t buffer_id,
				  uint32_t kmrels_handle,
				  kern_data_store *kds_final_head,
				  char *errmsg, size_t errmsg_sz)
{
	gpuQueryBuffer *gq_buf;
	dlist_iter		iter;
	int				hindex;
	struct {
		uint64_t	buffer_id;
		uint32_t	cuda_dindex;
	} hkey;

	/* lookup hash table first */
	memset(&hkey, 0, sizeof(hkey));
	hkey.buffer_id = buffer_id;
	hkey.cuda_dindex = CU_DINDEX_PER_THREAD;
	hindex = hash_bytes((unsigned char *)&hkey,
						sizeof(hkey)) % GPU_QUERY_BUFFER_NSLOTS;
	pthreadMutexLock(&gpu_query_buffer_mutex);
	dlist_foreach(iter, &gpu_query_buffer_hslot[hindex])
	{
		gq_buf = dlist_container(gpuQueryBuffer,
								 chain, iter.cur);
		if (gq_buf->buffer_id   == buffer_id &&
			gq_buf->cuda_dindex == CU_DINDEX_PER_THREAD)
		{
			gq_buf->refcnt++;

			/* wait for initial setup by other thread */
			while (gq_buf->phase == 0)
			{
				pthreadCondWait(&gpu_query_buffer_cond,
								&gpu_query_buffer_mutex);
			}

			if (gq_buf->phase < 0)
			{
				__putGpuQueryBufferNoLock(gq_buf);
				gq_buf = NULL;
			}
			pthreadMutexUnlock(&gpu_query_buffer_mutex);
			return gq_buf;
		}
	}
	/* not found, so create a new one */
	gq_buf = calloc(1, sizeof(gpuQueryBuffer));
	if (!gq_buf)
	{
		pthreadMutexUnlock(&gpu_query_buffer_mutex);
		return NULL;	/* out of memory */
	}
	gq_buf->refcnt = 1;
	gq_buf->phase  = 0;	/* not initialized yet */
	gq_buf->buffer_id = buffer_id;
	gq_buf->cuda_dindex = CU_DINDEX_PER_THREAD;
	dlist_push_tail(&gpu_query_buffer_hslot[hindex], &gq_buf->chain);
	pthreadMutexUnlock(&gpu_query_buffer_mutex);

	if ((kmrels_handle == 0 ||
		 __setupGpuQueryJoinInnerBuffer(gcontext,
										gq_buf, kmrels_handle,
										errmsg, errmsg_sz)) &&
		(kds_final_head == NULL ||
		 __setupGpuQueryGroupByBuffer(gcontext,
									  gq_buf, kds_final_head,
									  errmsg, errmsg_sz)))
	{
		/* ok, buffer is now ready */
		pthreadMutexLock(&gpu_query_buffer_mutex);
		gq_buf->phase = 1;		/* buffer is now ready */
		pthreadCondBroadcast(&gpu_query_buffer_cond);
		pthreadMutexUnlock(&gpu_query_buffer_mutex);		
		return gq_buf;
	}
	/* unable to setup the buffer */
	pthreadMutexLock(&gpu_query_buffer_mutex);
	gq_buf->phase = -1;			/* buffer unavailable */
	__putGpuQueryBufferNoLock(gq_buf);
	pthreadCondBroadcast(&gpu_query_buffer_cond);
	pthreadMutexUnlock(&gpu_query_buffer_mutex);
	return NULL;
}

/*
 * __gpuservLoadKdsCommon
 */
static const gpuMemChunk *
__gpuservLoadKdsCommon(gpuClient *gclient,
					   kern_data_store *kds,
					   size_t base_offset,
					   const char *pathname,
					   strom_io_vector *kds_iovec)
{
	gpuMemChunk *chunk;
	CUresult	rc;
	off_t		off = PAGE_ALIGN(base_offset);
	size_t		gap = off - base_offset;

	chunk = __gpuMemAlloc(gap + kds->length);
	if (!chunk)
	{
		gpuClientELog(gclient, "failed on gpuMemAlloc(%zu)", kds->length);
		return NULL;
	}
	chunk->m_devptr = chunk->__base__ + chunk->__offset__ + gap;

	rc = cuMemcpyHtoD(chunk->m_devptr, kds, base_offset);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on cuMemcpyHtoD: %s", cuStrError(rc));
		goto error;
	}
	if (!gpuDirectFileReadIOV(pathname,
							  chunk->__base__,
							  chunk->__offset__ + off,
							  kds_iovec))
	{
		gpuClientELog(gclient, "failed on gpuDirectFileReadIOV");
		goto error;
	}
	return chunk;

error:
	gpuMemFree(chunk);
	return NULL;
}

/*
 * gpuservLoadKdsBlock
 *
 * fill up KDS_FORMAT_BLOCK using GPU-Direct
 */
const gpuMemChunk *
gpuservLoadKdsBlock(gpuClient *gclient,
					kern_data_store *kds,
					const char *pathname,
					strom_io_vector *kds_iovec)
{
	size_t		base_offset;

	Assert(kds->format == KDS_FORMAT_BLOCK);
	base_offset = kds->block_offset + kds->block_nloaded * BLCKSZ;
	return __gpuservLoadKdsCommon(gclient, kds, base_offset, pathname, kds_iovec);
}

/*
 * gpuservLoadKdsArrow
 *
 * fill up KDS_FORMAT_ARROW using GPU-Direct
 */
const gpuMemChunk *
gpuservLoadKdsArrow(gpuClient *gclient,
					kern_data_store *kds,
					const char *pathname,
					strom_io_vector *kds_iovec)
{
	size_t		base_offset;

	Assert(kds->format == KDS_FORMAT_ARROW);
	base_offset = KDS_HEAD_LENGTH(kds);
	return __gpuservLoadKdsCommon(gclient, kds, base_offset, pathname, kds_iovec);
}

/*
 * gpuServiceGoingTerminate
 */
bool
gpuServiceGoingTerminate(void)
{
	return (gpuserv_bgworker_got_signal != 0);
}

/*
 * gpuClientPut
 */
static void
gpuClientPut(gpuClient *gclient, bool exit_monitor_thread)
{
	int		cnt = (exit_monitor_thread ? 1 : 2);
	int		val;

	if ((val = pg_atomic_sub_fetch_u32(&gclient->refcnt, cnt)) == 0)
	{
		gpuContext *gcontext = gclient->gcontext;
		CUresult	rc;

		pthreadMutexLock(&gcontext->client_lock);
		dlist_delete(&gclient->chain);
		pthreadMutexUnlock(&gcontext->client_lock);

		if (gclient->sockfd >= 0)
			close(gclient->sockfd);
		if (gclient->gq_buf)
			putGpuQueryBuffer(gclient->gq_buf);
		if (gclient->session)
		{
			CUdeviceptr		dptr = ((CUdeviceptr)gclient->session -
									offsetof(XpuCommand, u.session));
			rc = cuMemFree(dptr);
			if (rc != CUDA_SUCCESS)
				__GpuServDebug("failed on cuMemFree: %s\n", cuStrError(rc));
		}
		free(gclient);
	}
}

/*
 * gpuClientWriteBack
 */
static void
__gpuClientWriteBack(gpuClient *gclient, struct iovec *iov, int iovcnt)
{
	pthreadMutexLock(&gclient->mutex);
	if (gclient->sockfd >= 0)
	{
		ssize_t		nbytes;

		while (iovcnt > 0)
		{
			nbytes = writev(gclient->sockfd, iov, iovcnt);
			if (nbytes > 0)
			{
				do {
					if (iov->iov_len <= nbytes)
					{
						nbytes -= iov->iov_len;
						iov++;
						iovcnt--;
					}
					else
					{
						iov->iov_base = (char *)iov->iov_base + nbytes;
						iov->iov_len -= nbytes;
						break;
					}
				} while (iovcnt > 0 && nbytes > 0);
			}
			else if (errno != EINTR)
			{
				/*
				 * Peer socket is closed? Anyway, it looks we cannot continue
				 * to send back the message any more. So, clean up this gpuClient.
				 */
				pg_atomic_fetch_and_u32(&gclient->refcnt, ~1U);
				close(gclient->sockfd);
				gclient->sockfd = -1;
				break;
			}
		}
	}
	pthreadMutexUnlock(&gclient->mutex);
}

void
gpuClientWriteBack(gpuClient  *gclient,
				   XpuCommand *resp,
				   size_t      resp_sz,
				   int         kds_nitems,
				   kern_data_store **kds_array)
{
	struct iovec   *iov_array;
	struct iovec   *iov;
	int				i, iovcnt = 0;

	iov_array = alloca(sizeof(struct iovec) * (3 * kds_nitems + 1));
	iov = &iov_array[iovcnt++];
	iov->iov_base = resp;
	iov->iov_len  = resp_sz;
	for (i=0; i < kds_nitems; i++)
	{
		kern_data_store *kds = kds_array[i];
		size_t		sz1, sz2, sz3;

		if (kds->format == KDS_FORMAT_HASH)
		{
			assert(kds->hash_nslots > 0);
			sz1 = KDS_HEAD_LENGTH(kds);
			iov = &iov_array[iovcnt++];
			iov->iov_base = kds;
			iov->iov_len  = sz1;

			sz2 = MAXALIGN(sizeof(uint32_t) * kds->nitems);
			if (sz2 > 0)
			{
				iov = &iov_array[iovcnt++];
				iov->iov_base = KDS_GET_ROWINDEX(kds);
				iov->iov_len  = sz2;
			}

			sz3 = __kds_unpack(kds->usage);
			if (sz3 > 0)
			{
				iov = &iov_array[iovcnt++];
				iov->iov_base = (char *)kds + kds->length - sz3;
				iov->iov_len  = sz3;
			}
			/* fixup kds */
			kds->format = KDS_FORMAT_ROW;
			kds->hash_nslots = 0;
			kds->length = (sz1 + sz2 + sz3);
		}
		else if (kds->format == KDS_FORMAT_ROW)
		{
			assert(kds->hash_nslots == 0);
			sz1 = (KDS_HEAD_LENGTH(kds) +
				   MAXALIGN(sizeof(uint32_t) * kds->nitems));
			sz2 = __kds_unpack(kds->usage);
			if (sz1 + sz2 == kds->length)
			{
				iov = &iov_array[iovcnt++];
				iov->iov_base = kds;
				iov->iov_len  = kds->length;
			}
			else
			{
				assert(sz1 + sz2 < kds->length);
				iov = &iov_array[iovcnt++];
				iov->iov_base = kds;
				iov->iov_len  = sz1;

				if (sz2 > 0)
				{
					iov = &iov_array[iovcnt++];
					iov->iov_base = (char *)kds + kds->length - sz2;
					iov->iov_len  = sz2;
				}
				kds->length = (sz1 + sz2);
			}
		}
		else
		{
			/*
			 * KDS_FORMAT_BLOCK and KDS_FORMAT_ARROW may happen if CPU fallback
			 * tries to send back the source buffer.
			 */
			iov = &iov_array[iovcnt++];
			iov->iov_base = kds;
			iov->iov_len  = kds->length;
		}
		resp_sz += kds->length;
	}
	resp->length = resp_sz;
	__gpuClientWriteBack(gclient, iov_array, iovcnt);
}

/*
 * gpuClientELog
 */
void
__gpuClientELogRaw(gpuClient *gclient, kern_errorbuf *errorbuf)
{
	XpuCommand		resp;
	struct iovec	iov;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Error;
	resp.length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	memcpy(&resp.u.error, errorbuf, sizeof(kern_errorbuf));

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);
}

void
__gpuClientELog(gpuClient *gclient,
				int errcode,
				const char *filename, int lineno,
				const char *funcname,
				const char *fmt, ...)
{
	XpuCommand		resp;
	va_list			ap;
	struct iovec	iov;
	const char	   *pos;

	for (pos = filename; *pos != '\0'; pos++)
	{
		if (pos[0] == '/' && pos[1] != '\0')
			filename = pos + 1;
	}
	
	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Error;
	resp.length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	resp.u.error.errcode = errcode,
	resp.u.error.lineno = lineno;
	strncpy(resp.u.error.filename, filename, KERN_ERRORBUF_FILENAME_LEN);
	strncpy(resp.u.error.funcname, funcname, KERN_ERRORBUF_FUNCNAME_LEN);

	va_start(ap, fmt);
	vsnprintf(resp.u.error.message, KERN_ERRORBUF_MESSAGE_LEN, fmt, ap);
	va_end(ap);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);

	/* unable to continue GPU service, so try to restart */
	if (errcode == ERRCODE_DEVICE_FATAL)
	{
		__GpuServDebug("(%s:%d, %s) GPU fatal - %s\n",
					   resp.u.error.filename,
					   resp.u.error.lineno,
					   resp.u.error.funcname,
					   resp.u.error.message);
		gpuserv_bgworker_got_signal |= (1 << SIGHUP);
		pg_memory_barrier();
		SetLatch(MyLatch);
	}
}

/*
 * gpuservHandleOpenSession
 */
static bool
__resolveDevicePointersWalker(gpuContext *gcontext,
							  kern_expression *kexp,
							  char *emsg, size_t emsg_sz)
{
	xpu_function_catalog_entry *xpu_func;
	xpu_type_catalog_entry *xpu_type;
	kern_expression *karg;
	int		i;

	/* lookup device function */
	xpu_func = hash_search(gcontext->cuda_func_htab,
						   &kexp->opcode,
						   HASH_FIND, NULL);
	if (!xpu_func)
	{
		snprintf(emsg, emsg_sz,
				 "device function pointer for opcode:%u not found.",
				 (int)kexp->opcode);
		return false;
	}
	kexp->fn_dptr = xpu_func->func_dptr;

	/* lookup device type operator */
	xpu_type = hash_search(gcontext->cuda_type_htab,
						   &kexp->exptype,
						   HASH_FIND, NULL);
	if (!xpu_type)
	{
		snprintf(emsg, emsg_sz,
				 "device type pointer for opcode:%u not found.",
				 (int)kexp->exptype);
		return false;
	}
	kexp->expr_ops = xpu_type->type_ops;

	/* special case if CASE ... WHEN */
	if (kexp->opcode == FuncOpCode__CaseWhenExpr)
	{
		if (kexp->u.casewhen.case_comp)
		{
			karg = (kern_expression *)((char *)kexp + kexp->u.casewhen.case_comp);
			if (!__KEXP_IS_VALID(kexp,karg))
			{
				snprintf(emsg, emsg_sz,
						 "XPU code corruption at kexp (%d)", kexp->opcode);
				return false;
			}
			if (!__resolveDevicePointersWalker(gcontext, karg, emsg, emsg_sz))
				return false;

		}
		if (kexp->u.casewhen.case_else)
		{
			karg = (kern_expression *)((char *)kexp + kexp->u.casewhen.case_else);
			if (!__KEXP_IS_VALID(kexp,karg))
			{
				snprintf(emsg, emsg_sz,
						 "XPU code corruption at kexp (%d)", kexp->opcode);
				return false;
			}
			if (!__resolveDevicePointersWalker(gcontext, karg, emsg, emsg_sz))
				return false;
		}
	}

	for (i=0, karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		if (!__KEXP_IS_VALID(kexp,karg))
		{
			snprintf(emsg, emsg_sz, "XPU code corruption at kexp (%d)", kexp->opcode);
			return false;
		}
		if (!__resolveDevicePointersWalker(gcontext, karg, emsg, emsg_sz))
			return false;
	}
	return true;
}

static bool
__resolveDevicePointers(gpuContext *gcontext,
						kern_session_info *session,
						char *emsg, size_t emsg_sz)
{
	xpu_encode_info	*encode = SESSION_ENCODE(session);
	kern_expression *__kexp[20];
	int			i, nitems = 0;

	__kexp[nitems++] = SESSION_KEXP_SCAN_LOAD_VARS(session);
	__kexp[nitems++] = SESSION_KEXP_SCAN_QUALS(session);
	__kexp[nitems++] = SESSION_KEXP_JOIN_LOAD_VARS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_JOIN_QUALS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_HASH_VALUE(session, -1);
	__kexp[nitems++] = SESSION_KEXP_GIST_EVALS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_PROJECTION(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYHASH(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYLOAD(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYCOMP(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_ACTIONS(session);

	for (i=0; i < nitems; i++)
	{
		if (__kexp[i] && !__resolveDevicePointersWalker(gcontext,
														__kexp[i],
														emsg, emsg_sz))
			return false;
	}

	if (encode)
	{
		xpu_encode_info *catalog = gcontext->cuda_encode_catalog;
		int		i;

		for (i=0; ; i++)
		{
			if (!catalog[i].enc_mblen || catalog[i].enc_maxlen < 1)
			{
				snprintf(emsg, emsg_sz,
						 "encode [%s] was not found.", encode->encname);
				return false;
			}
			if (strcmp(encode->encname, catalog[i].encname) == 0)
			{
				encode->enc_maxlen = catalog[i].enc_maxlen;
				encode->enc_mblen  = catalog[i].enc_mblen;
				break;
			}
		}
	}
	return true;
}

static bool
gpuservHandleOpenSession(gpuClient *gclient, XpuCommand *xcmd)
{
	gpuContext	   *gcontext = gclient->gcontext;
	kern_session_info *session = &xcmd->u.session;
	XpuCommand		resp;
	char			emsg[512];
	struct iovec	iov;

	if (gclient->session)
	{
		gpuClientELog(gclient, "OpenSession is called twice");
		return false;
	}

	/* resolve device pointers */
	if (!__resolveDevicePointers(gcontext, session, emsg, sizeof(emsg)))
	{
		gpuClientELog(gclient, "%s", emsg);
		return false;
	}
	if (session->join_inner_handle != 0 ||
		session->groupby_kds_final != 0)
	{
		kern_data_store *kds_final_head = NULL;

		if (session->groupby_kds_final != 0)
		{
			kds_final_head = (kern_data_store *)
				((char *)session + session->groupby_kds_final);
		}
		gclient->gq_buf = getGpuQueryBuffer(gcontext,
											session->query_plan_id,
											session->join_inner_handle,
											kds_final_head,
											emsg, sizeof(emsg));
		if (!gclient->gq_buf)
		{
			gpuClientELog(gclient, "%s", emsg);
			return false;
		}
	}
	gclient->session = session;

	/* success status */
	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Success;
	resp.length = offsetof(XpuCommand, u);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);

	return true;
}

/* ----------------------------------------------------------------
 *
 * gpuservHandleGpuTaskExec
 *
 * ----------------------------------------------------------------
 */
static void
gpuservHandleGpuTaskExec(gpuClient *gclient, XpuCommand *xcmd)
{
	gpuContext		*gcontext = gclient->gcontext;
	kern_session_info *session = gclient->session;
	gpuQueryBuffer  *gq_buf = gclient->gq_buf;
	kern_gputask	*kgtask = NULL;
	const char		*kds_src_pathname = NULL;
	strom_io_vector *kds_src_iovec = NULL;
	kern_data_store *kds_src = NULL;
	kern_data_extra	*kds_extra = NULL;
	kern_data_store *kds_dst = NULL;
	kern_data_store *kds_dst_head = NULL;
	kern_data_store **kds_dst_array = NULL;
	int				kds_dst_nrooms = 0;
	int				kds_dst_nitems = 0;
//	uint32_t		kcxt_kvars_nslots = session->kcxt_kvars_nslots;
//	uint32_t		kcxt_kvars_nbytes = session->kcxt_kvars_nbytes;
	int				num_inner_rels = 0;
	CUfunction		f_kern_gpuscan;
	const gpuMemChunk *chunk = NULL;
	CUdeviceptr		m_kds_src = 0UL;
	CUdeviceptr		m_kmrels = 0UL;
	CUdeviceptr		dptr;
	CUresult		rc;
	int				grid_sz;
	int				block_sz;
	unsigned int	shmem_sz;
	size_t			kds_final_length = 0;
	bool			kds_final_locked = false;
	size_t			sz;
	void		   *kern_args[10];

	if (xcmd->u.task.kds_src_pathname)
		kds_src_pathname = (char *)xcmd + xcmd->u.task.kds_src_pathname;
	if (xcmd->u.task.kds_src_iovec)
		kds_src_iovec = (strom_io_vector *)((char *)xcmd + xcmd->u.task.kds_src_iovec);
	if (xcmd->u.task.kds_src_offset)
		kds_src = (kern_data_store *)((char *)xcmd + xcmd->u.task.kds_src_offset);
	if (xcmd->u.task.kds_dst_offset)
		kds_dst_head = (kern_data_store *)((char *)xcmd + xcmd->u.task.kds_dst_offset);

	if (!kds_src)
	{
		gpuClientELog(gclient, "KDS_FORMAT_COLUMN is not yet implemented");
		return;
	}
	else if (kds_src->format == KDS_FORMAT_ROW)
	{
		m_kds_src = (CUdeviceptr)kds_src;
	}
	else if (kds_src->format == KDS_FORMAT_BLOCK)
	{
		if (kds_src_pathname && kds_src_iovec)
		{
			chunk = gpuservLoadKdsBlock(gclient,
										kds_src,
										kds_src_pathname,
										kds_src_iovec);
			if (!chunk)
				return;
			m_kds_src = chunk->m_devptr;
		}
		else
		{
			Assert(kds_src->block_nloaded == kds_src->nitems);
			m_kds_src = (CUdeviceptr)kds_src;
		}
	}
	else if (kds_src->format == KDS_FORMAT_ARROW)
	{
		if (kds_src_iovec->nr_chunks == 0)
			m_kds_src = (CUdeviceptr)kds_src;
		else
		{
			if (!kds_src_pathname)
			{
				gpuClientELog(gclient, "GpuScan: arrow file is missing");
				return;
			}
			chunk = gpuservLoadKdsArrow(gclient,
										kds_src,
										kds_src_pathname,
										kds_src_iovec);
			if (!chunk)
				return;
			m_kds_src = chunk->m_devptr;
		}
	}
	else
	{
		gpuClientELog(gclient, "unknown GpuScan Source format (%c)",
					  kds_src->format);
		return;
	}
	/* inner buffer of GpuJoin */
	if (gq_buf && gq_buf->m_kmrels)
	{
		kern_multirels *h_kmrels = (kern_multirels *)gq_buf->h_kmrels;

		m_kmrels = gq_buf->m_kmrels;
		num_inner_rels = h_kmrels->num_rels;
	}

	rc = cuModuleGetFunction(&f_kern_gpuscan,
							 gcontext->cuda_module,
							 "kern_gpujoin_main");
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuModuleGetFunction: %s",
					   cuStrError(rc));
		goto bailout;
	}

	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 &shmem_sz,
							 f_kern_gpuscan,
							 0,
							 __KERN_WARP_CONTEXT_BASESZ(num_inner_rels));
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on gpuOptimalBlockSize: %s",
					   cuStrError(rc));
		goto bailout;
	}
	
	/*
	 * Allocation of the control structure
	 */
	grid_sz = Min(grid_sz, (kds_src->nitems + block_sz - 1) / block_sz);
//	block_sz = 128;
//	grid_sz = 1;

	sz = KERN_GPUTASK_LENGTH(num_inner_rels,
							 session->kcxt_kvars_ndims,
							 session->kcxt_kvars_nbytes,
							 grid_sz * block_sz);
	rc = cuMemAllocManaged(&dptr, sz, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuMemAllocManaged(%lu): %s",
					   sz, cuStrError(rc));
		goto bailout;
	}
	kgtask = (kern_gputask *)dptr;
	memset(kgtask, 0, offsetof(kern_gputask, stats[0]));
	kgtask->grid_sz  = grid_sz;
	kgtask->block_sz = block_sz;
	kgtask->kvars_nslots = session->kcxt_kvars_nslots;
	kgtask->kvars_nbytes = session->kcxt_kvars_nbytes;
	kgtask->kvars_ndims  = session->kcxt_kvars_ndims;
	kgtask->n_rels       = num_inner_rels;

	/* prefetch source KDS, if managed memory */
	if (!chunk)
	{
		rc = cuMemPrefetchAsync((CUdeviceptr)kds_src,
								kds_src->length,
								CU_DEVICE_PER_THREAD,
								CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientFatal(gclient, "failed on cuMemPrefetchAsync: %s",
						   cuStrError(rc));
			goto bailout;
		}
	}

	/*
	 * Allocation of the destination buffer
	 */
resume_kernel:
	if (gq_buf && gq_buf->m_kds_final)
	{
		/*
		 * Suspend of GpuPreAgg kernel means the kds_final buffer is
		 * almost full, thus GPU kernel wants to expand the buffer.
		 * It must be done under the exclusive lock.
		 */
		if (kgtask->resume_context)
		{
			if (!__expandGpuQueryGroupByBuffer(gq_buf, kds_final_length))
			{
				gpuClientFatal(gclient, "unable to expand GpuPreAgg final buffer");
				goto bailout;
			}
		}
		pthreadRWLockReadLock(&gq_buf->m_kds_final_rwlock);
		kds_dst = (kern_data_store *)gq_buf->m_kds_final;
		kds_final_length = gq_buf->m_kds_final_length;
		kds_final_locked = true;
	}
	else
	{
		sz = KDS_HEAD_LENGTH(kds_dst_head) + PGSTROM_CHUNK_SIZE;
		rc = cuMemAllocManaged(&dptr, sz, CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientFatal(gclient, "failed on cuMemAllocManaged(%lu): %s",
						   sz, cuStrError(rc));
			goto bailout;
		}
		kds_dst = (kern_data_store *)dptr;
		memcpy(kds_dst, kds_dst_head, KDS_HEAD_LENGTH(kds_dst_head));
		kds_dst->length = sz;
		if (kds_dst_nitems >= kds_dst_nrooms)
		{
			kern_data_store **kds_dst_temp;

			kds_dst_nrooms = 2 * kds_dst_nrooms + 10;
			kds_dst_temp = alloca(sizeof(kern_data_store *) * kds_dst_nrooms);
			if (kds_dst_nitems > 0)
				memcpy(kds_dst_temp,
					   kds_dst_array,
					   sizeof(kern_data_store *) * kds_dst_nitems);
			kds_dst_array = kds_dst_temp;
		}
		kds_dst_array[kds_dst_nitems++] = kds_dst;
	}

	/*
	 * Launch kernel
	 */
	kern_args[0] = &gclient->session;
	kern_args[1] = &kgtask;
	kern_args[2] = &m_kmrels;
	kern_args[3] = &m_kds_src;
	kern_args[4] = &kds_extra;
	kern_args[5] = &kds_dst;

	rc = cuLaunchKernel(f_kern_gpuscan,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						shmem_sz,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuLaunchKernel: %s", cuStrError(rc));
		goto bailout;
	}

	rc = cuEventRecord(CU_EVENT_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuEventRecord: %s", cuStrError(rc));
		goto bailout;
	}

	/* point of synchronization */
	rc = cuEventSynchronize(CU_EVENT_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuEventSynchronize: %s", cuStrError(rc));
		goto bailout;
	}
	/* unlock kds_final buffer */
	if (kds_final_locked)
	{
		pthreadRWLockUnlock(&gq_buf->m_kds_final_rwlock);
		kds_final_locked = false;
	}

	/* status check */
	if (kgtask->kerror.errcode == ERRCODE_STROM_SUCCESS)
	{
		XpuCommand *resp;
		size_t		resp_sz;

		if (kgtask->suspend_count > 0)
		{
			if (gpuServiceGoingTerminate())
			{
				gpuClientFatal(gclient, "GpuService is going to terminate during GpuScan kernel suspend/resume");
				goto bailout;
			}
			/* restore warp context from the previous state */
			kgtask->resume_context = true;
			kgtask->suspend_count = 0;
			__GpuServDebug("suspend / resume happen\n");
			if (kds_final_locked)
				pthreadRWLockUnlock(&gq_buf->m_kds_final_rwlock);
			goto resume_kernel;
		}
		/* send back status and kds_dst */
		resp_sz = MAXALIGN(offsetof(XpuCommand,
									u.results.stats[num_inner_rels]));
		resp = alloca(resp_sz);
		memset(resp, 0, resp_sz);
		resp->magic = XpuCommandMagicNumber;
		resp->tag   = XpuCommandTag__Success;
		resp->u.results.chunks_nitems = kds_dst_nitems;
		resp->u.results.chunks_offset = resp_sz;
		resp->u.results.nitems_raw = kgtask->nitems_raw;
		resp->u.results.nitems_in  = kgtask->nitems_in;
		resp->u.results.nitems_out = kgtask->nitems_out;
		resp->u.results.num_rels = num_inner_rels;
		for (int i=0; i < num_inner_rels; i++)
		{
			resp->u.results.stats[i].nitems_gist = kgtask->stats[i].nitems_gist;
			resp->u.results.stats[i].nitems_out  = kgtask->stats[i].nitems_out;
		}
		gpuClientWriteBack(gclient,
						   resp, resp_sz,
						   kds_dst_nitems, kds_dst_array);
	}
	else if (kgtask->kerror.errcode == ERRCODE_CPU_FALLBACK &&
			 (session->xpu_task_flags & DEVTASK__MASK) != DEVTASK__PREAGG)
	{
		XpuCommand	resp;
		
		/* send back kds_src with XpuCommandTag__CPUFallback */
		memset(&resp, 0, sizeof(resp));
		resp.magic = XpuCommandMagicNumber;
		resp.tag   = XpuCommandTag__CPUFallback;
		resp.u.results.chunks_nitems = 1;
		resp.u.results.chunks_offset = offsetof(XpuCommand, u.results.stats);
		gpuClientWriteBack(gclient,
						   &resp, resp.u.results.chunks_offset,
						   1, &kds_src);
	}
	else
	{
		/* send back error status */
		__gpuClientELogRaw(gclient, &kgtask->kerror);
	}
bailout:
	if (kds_final_locked)
		pthreadRWLockUnlock(&gq_buf->m_kds_final_rwlock);
	if (kgtask)
	{
		rc = cuMemFree((CUdeviceptr)kgtask);
		if (rc != CUDA_SUCCESS)
			__GpuServDebug("failed on cuMemFree: %s\n", cuStrError(rc));
	}
	if (chunk)
		gpuMemFree(chunk);
	while (kds_dst_nitems > 0)
	{
		kds_dst = kds_dst_array[--kds_dst_nitems];

		rc = cuMemFree((CUdeviceptr)kds_dst);
		if (rc != CUDA_SUCCESS)
			__GpuServDebug("failed on cuMemFree: %s\n", cuStrError(rc));
	}
}

/* ----------------------------------------------------------------
 *
 * gpuservHandleGpuTaskFinal
 *
 * ----------------------------------------------------------------
 */
static void
gpuservHandleGpuTaskFinal(gpuClient *gclient, XpuCommand *xcmd)
{
	kern_final_task *kfin = &xcmd->u.fin;
	gpuQueryBuffer *gq_buf = gclient->gq_buf;
	XpuCommand		resp;
	kern_data_store	*kds_final = NULL;

	memset(&resp, 0, sizeof(XpuCommand));
	resp.magic = XpuCommandMagicNumber;
	resp.tag   = XpuCommandTag__Success;
	resp.u.results.chunks_nitems = 0;
	resp.u.results.chunks_offset = MAXALIGN(offsetof(XpuCommand, u.results.stats));

	/*
	 * Is the outer-join-map written back to the host buffer?
	 */
	if (kfin->final_this_device)
	{
		if (gq_buf &&
			gq_buf->m_kmrels != 0UL &&
			gq_buf->h_kmrels != NULL)
		{
			kern_multirels *d_kmrels = (kern_multirels *)gq_buf->m_kmrels;
			kern_multirels *h_kmrels = (kern_multirels *)gq_buf->h_kmrels;

			for (int i=0; i < d_kmrels->num_rels; i++)
			{
				kern_data_store *kds = KERN_MULTIRELS_INNER_KDS(h_kmrels, i);
				bool   *d_ojmap = KERN_MULTIRELS_OUTER_JOIN_MAP(d_kmrels, i);
				bool   *h_ojmap = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, i);

				if (d_ojmap && h_ojmap)
				{
					for (uint32_t j=0; j < kds->nitems; j++)
						h_ojmap[j] |= d_ojmap[j];
					resp.u.results.final_this_device = true;
				}
			}
		}
	}
	/*
	 * Is the GpuPreAgg final buffer written back?
	 */
	if (kfin->final_plan_node)
	{
		if (gq_buf && gq_buf->m_kds_final != 0UL)
		{
			kds_final = (kern_data_store *)gq_buf->m_kds_final;
			resp.u.results.chunks_nitems = 1;
			resp.u.results.final_plan_node = true;
		}
	}
	//fprintf(stderr, "gpuservHandleGpuTaskFinal: kfin => {final_this_device=%d final_plan_node=%d} resp => {final_this_device=%d final_plan_node=%d}\n", kfin->final_this_device, kfin->final_plan_node, resp.u.results.final_this_device, resp.u.results.final_plan_node);

	gpuClientWriteBack(gclient, &resp,
					   resp.u.results.chunks_offset,
					   resp.u.results.chunks_nitems,
					   &kds_final);
}

/*
 * gpuservGpuWorkerMain -- actual worker
 */
static void *
gpuservGpuWorkerMain(void *__arg)
{
	gpuContext *gcontext = (gpuContext *)__arg;
	gpuClient  *gclient;
	CUevent		cuda_event;
	CUresult	rc;

	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuCtxSetCurrent: %s", cuStrError(rc));
	rc = cuEventCreate(&cuda_event, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuEventCreate: %s", cuStrError(rc));

	GpuWorkerCurrentContext = gcontext;
	CU_DINDEX_PER_THREAD  = gcontext->cuda_dindex;
	CU_DEVICE_PER_THREAD  = gcontext->cuda_device;
	CU_CONTEXT_PER_THREAD = gcontext->cuda_context;
	CU_EVENT_PER_THREAD = cuda_event;
	pg_memory_barrier();

	pthreadMutexLock(&gcontext->lock);
	while (!gpuServiceGoingTerminate())
	{
		XpuCommand *xcmd;
		dlist_node *dnode;

		if (!dlist_is_empty(&gcontext->command_list))
		{
			dnode = dlist_pop_head_node(&gcontext->command_list);
			xcmd = dlist_container(XpuCommand, chain, dnode);
			pthreadMutexUnlock(&gcontext->lock);

			gclient = xcmd->priv;
			/*
			 * MEMO: If the least bit of gclient->refcnt is not set,
			 * it means the gpu-client connection is no longer available.
			 * (already closed, or error detected.)
			 */
			if ((pg_atomic_read_u32(&gclient->refcnt) & 1) == 1)
			{
				switch (xcmd->tag)
				{
					case XpuCommandTag__OpenSession:
						if (gpuservHandleOpenSession(gclient, xcmd))
							xcmd = NULL;	/* session information shall be kept until
											 * end of the session. */
						break;
					case XpuCommandTag__XpuTaskExec:
						gpuservHandleGpuTaskExec(gclient, xcmd);
						break;
					case XpuCommandTag__XpuTaskFinal:
						gpuservHandleGpuTaskFinal(gclient, xcmd);
						break;
					default:
						gpuClientELog(gclient, "unknown XPU command (%d)",
									  (int)xcmd->tag);
						break;
				}
			}
			if (xcmd)
				cuMemFree((CUdeviceptr)xcmd);
			gpuClientPut(gclient, false);
			pthreadMutexLock(&gcontext->lock);
		}
		else if (!pthreadCondWaitTimeout(&gcontext->cond,
										 &gcontext->lock,
										 5000))
		{
			pthreadMutexUnlock(&gcontext->lock);
			/* maintenance works */
			gpuMemoryPoolMaintenance(gcontext);
			pthreadMutexLock(&gcontext->lock);
		}
	}
	pthreadMutexUnlock(&gcontext->lock);
	return NULL;
}

/*
 * gpuservMonitorClient
 */
static void *
__gpuServiceAllocCommand(void *__priv, size_t sz)
{
	CUdeviceptr	devptr;
	CUresult	rc;

	rc = cuMemAllocManaged(&devptr, sz, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		return NULL;
	return (void *)devptr;
}

static void
__gpuServiceAttachCommand(void *__priv, XpuCommand *xcmd)
{
	gpuClient  *gclient = (gpuClient *)__priv;
	gpuContext *gcontext = gclient->gcontext;

	pg_atomic_fetch_add_u32(&gclient->refcnt, 2);
	xcmd->priv = gclient;

	pthreadMutexLock(&gcontext->lock);
	dlist_push_tail(&gcontext->command_list, &xcmd->chain);
	pthreadMutexUnlock(&gcontext->lock);
	pthreadCondSignal(&gcontext->cond);
}
TEMPLATE_XPU_CONNECT_RECEIVE_COMMANDS(__gpuService)

static void *
gpuservMonitorClient(void *__priv)
{
	gpuClient  *gclient = __priv;
	gpuContext *gcontext = gclient->gcontext;
	pgsocket	sockfd = gclient->sockfd;
	char		elabel[32];
	CUresult	rc;

	snprintf(elabel, sizeof(elabel), "GPU-%d", gcontext->cuda_dindex);

	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		__GpuServDebug("[%s] failed on cuCtxSetCurrent: %s\n",
					   elabel, cuStrError(rc));
		goto out;
	}
	GpuWorkerCurrentContext = gcontext;
	pg_memory_barrier();

	for (;;)
	{
		struct pollfd  pfd;
		int		nevents;

		pfd.fd = sockfd;
		pfd.events = POLLIN;
		pfd.revents = 0;
		nevents = poll(&pfd, 1, -1);
		if (nevents < 0)
		{
			if (errno == EINTR)
				continue;
			__GpuServDebug("[%s] failed on poll(2): %m", elabel);
			break;
		}
		if (nevents == 0)
			continue;
		Assert(nevents == 1);
		if (pfd.revents == POLLIN)
		{
			if (__gpuServiceReceiveCommands(sockfd, gclient, elabel) < 0)
				break;
		}
		else if (pfd.revents & ~POLLIN)
		{
			__GpuServDebug("[%s] peer socket closed.", elabel);
			break;
		}
	}
out:
	gpuClientPut(gclient, true);
	return NULL;
}

/*
 * gpuservAcceptClient
 */
static void
gpuservAcceptClient(gpuContext *gcontext)
{
	gpuClient  *gclient;
	pgsocket	sockfd;
	int			errcode;

	sockfd = accept(gcontext->serv_fd, NULL, NULL);
	if (sockfd < 0)
	{
		elog(LOG, "GPU%d: could not accept new connection: %m",
			 gcontext->cuda_dindex);
		pg_usleep(10000L);		/* wait 10ms */
		return;
	}

	gclient = calloc(1, sizeof(gpuClient));
	if (!gclient)
	{
		elog(LOG, "GPU%d: out of memory: %m",
			 gcontext->cuda_dindex);
		close(sockfd);
		return;
	}
	gclient->gcontext = gcontext;
	pg_atomic_init_u32(&gclient->refcnt, 1);
	pthreadMutexInit(&gclient->mutex);
	gclient->sockfd = sockfd;

	if ((errcode = pthread_create(&gclient->worker, NULL,
								  gpuservMonitorClient,
								  gclient)) != 0)
	{
		elog(LOG, "failed on pthread_create: %s", strerror(errcode));
		close(sockfd);
		free(gclient);
		return;
	}
	pthreadMutexLock(&gcontext->client_lock);
	dlist_push_tail(&gcontext->client_list, &gclient->chain);
	pthreadMutexUnlock(&gcontext->client_lock);
}

/*
 * __setupDevTypeLinkageTable
 */
static HTAB *
__setupDevTypeLinkageTable(CUmodule cuda_module)
{
	xpu_type_catalog_entry *xpu_types_catalog;
	HASHCTL		hctl;
	HTAB	   *htab = NULL;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;
	int			i;
	
	/* build device type table */
	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(TypeOpCode);
	hctl.entrysize = sizeof(xpu_type_catalog_entry);
	hctl.hcxt = TopMemoryContext;
	htab = hash_create("CUDA device type hash table",
					   512,
					   &hctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	rc = cuModuleGetGlobal(&dptr, &nbytes, cuda_module,
						   "builtin_xpu_types_catalog");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal: %s", cuStrError(rc));

	xpu_types_catalog = alloca(nbytes);
	rc = cuMemcpyDtoH(xpu_types_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));
	for (i=0; xpu_types_catalog[i].type_opcode != TypeOpCode__Invalid; i++)
	{
		TypeOpCode	type_opcode = xpu_types_catalog[i].type_opcode;
		xpu_type_catalog_entry *entry;
		bool		found;

		entry = hash_search(htab, &type_opcode, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? duplicated TypeOpCode: %u", (uint32_t)type_opcode);
		Assert(entry->type_opcode == type_opcode);
		entry->type_ops = xpu_types_catalog[i].type_ops;
	}
	return htab;
}

/*
 * __setupDevFuncLinkageTable
 */
static HTAB *
__setupDevFuncLinkageTable(CUmodule cuda_module)
{
	xpu_function_catalog_entry *xpu_funcs_catalog;
	HASHCTL		hctl;
	HTAB	   *htab;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;
	int			i;

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(FuncOpCode);
	hctl.entrysize = sizeof(xpu_function_catalog_entry);
	hctl.hcxt = TopMemoryContext;
	htab = hash_create("CUDA device function hash table",
					   1024,
					   &hctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	rc = cuModuleGetGlobal(&dptr, &nbytes, cuda_module,
						   "builtin_xpu_functions_catalog");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal: %s", cuStrError(rc));
	xpu_funcs_catalog = alloca(nbytes);
	rc = cuMemcpyDtoH(xpu_funcs_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));
	for (i=0; xpu_funcs_catalog[i].func_opcode != FuncOpCode__Invalid; i++)
	{
		FuncOpCode	func_opcode = xpu_funcs_catalog[i].func_opcode;
		xpu_function_catalog_entry *entry;
		bool		found;

		entry = hash_search(htab, &func_opcode, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? duplicated FuncOpCode: %u", (uint32_t)func_opcode);
		Assert(entry->func_opcode == func_opcode);
		entry->func_dptr = xpu_funcs_catalog[i].func_dptr;
	}
	return htab;
}

/*
 * __setupDevEncodeLinkageCatalog
 */
static xpu_encode_info *
__setupDevEncodeLinkageCatalog(CUmodule cuda_module)
{
	xpu_encode_info *xpu_encode_catalog;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;

	rc = cuModuleGetGlobal(&dptr, &nbytes, cuda_module,
                           "xpu_encode_catalog");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal: %s", cuStrError(rc));
	xpu_encode_catalog = MemoryContextAlloc(TopMemoryContext, nbytes);
	rc = cuMemcpyDtoH(xpu_encode_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));

	return xpu_encode_catalog;
}

/*
 * gpuservSetupGpuModule
 */
static void
gpuservSetupGpuModule(gpuContext *gcontext)
{
	CUmodule	cuda_module;
	CUlinkState	lstate;
	CUjit_option jit_options[16];
	void	   *jit_option_values[16];
	int			jit_index = 0;
	void	   *bin_image;
	size_t		bin_length;
	char		log_buffer[16384];
	char	   *cuda_builtin_objs;
	char	   *tok, *saveptr;
	CUresult	rc;

	/* Limit max number of registers per threads for ABI compatibility */
	jit_options[jit_index] = CU_JIT_MAX_REGISTERS;
	jit_option_values[jit_index] = (void *)CUDA_MAXREGCOUNT;
	jit_index++;

	/* Get optimal binary to the current context */
	jit_options[jit_index] = CU_JIT_TARGET_FROM_CUCONTEXT;
	jit_option_values[jit_index] = NULL;
	jit_index++;

	/* Compile with L1 cache enabled */
	jit_options[jit_index] = CU_JIT_CACHE_MODE;
	jit_option_values[jit_index] = (void *)CU_JIT_CACHE_OPTION_CA;
	jit_index++;

	jit_options[jit_index] = CU_JIT_GENERATE_LINE_INFO;
	jit_option_values[jit_index] = (void *)1UL;
	jit_index++;

	/* Link log buffer */
	jit_options[jit_index] = CU_JIT_ERROR_LOG_BUFFER;
	jit_option_values[jit_index] = (void *)log_buffer;
	jit_index++;

	jit_options[jit_index] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
	jit_option_values[jit_index] = (void *)sizeof(log_buffer);
	jit_index++;

	rc = cuLinkCreate(jit_index, jit_options, jit_option_values, &lstate);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkCreate: %s", cuStrError(rc));

	/* Add builtin fatbin files */
	cuda_builtin_objs = alloca(sizeof(CUDA_BUILTIN_OBJS) + 1);
	strcpy(cuda_builtin_objs, CUDA_BUILTIN_OBJS);
	for (tok = strtok_r(cuda_builtin_objs, " ", &saveptr);
		 tok != NULL;
		 tok = strtok_r(NULL, " ", &saveptr))
	{
		char	pathname[MAXPGPATH];

		snprintf(pathname, MAXPGPATH,
				 PGSHAREDIR "/pg_strom/%s.fatbin",
				 __trim(tok));
		rc = cuLinkAddFile(lstate, CU_JIT_INPUT_FATBINARY,
						   pathname, 0, NULL, NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLinkAddFile('%s'): %s",
				 pathname, cuStrError(rc));
	}
	//TODO: Load the extra CUDA module

	/* do the linkage */
	rc = cuLinkComplete(lstate, &bin_image, &bin_length);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkComplete: %s\n%s",
			 cuStrError(rc), log_buffer);

	rc = cuModuleLoadData(&cuda_module, bin_image);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleLoadData: %s", cuStrError(rc));

	rc = cuLinkDestroy(lstate);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkDestroy: %s", cuStrError(rc));

	/* setup XPU linkage hash tables */
	gcontext->cuda_type_htab = __setupDevTypeLinkageTable(cuda_module);
	gcontext->cuda_func_htab = __setupDevFuncLinkageTable(cuda_module);
	gcontext->cuda_encode_catalog = __setupDevEncodeLinkageCatalog(cuda_module);
	gcontext->cuda_module = cuda_module;
}

/*
 * __gpuContextTerminateWorkers
 *
 * Wake up all the worker threads, and terminate them.
 * The event_fd is polled with edge-trigger mode for individual wake-up,
 * thus we have to clear them to wake up all.
 */
static void
__gpuContextTerminateWorkers(gpuContext *gcontext)
{
	int		i;

	gpuserv_bgworker_got_signal |= (1 << SIGHUP);
	pg_memory_barrier();

	pthreadCondBroadcast(&gcontext->cond);
	for (i=0; i < gcontext->n_workers; i++)
		pthread_join(gcontext->workers[i], NULL);
}

/*
 * gpuservSetupGpuContext
 */
static gpuContext *
gpuservSetupGpuContext(int cuda_dindex)
{
	GpuDevAttributes *dattrs = &gpuDevAttrs[cuda_dindex];
	gpuContext *gcontext = NULL;
	CUresult	rc;
	size_t		stack_sz;
	int			i, status;
	struct sockaddr_un addr;
	struct epoll_event ev;

	/* gpuContext allocation */
	gcontext = calloc(1, SizeOfGpuContext);
	if (!gcontext)
		elog(ERROR, "out of memory");
	gcontext->serv_fd = -1;
	gcontext->cuda_dindex = cuda_dindex;
	gpuMemoryPoolInit(&gcontext->pool, dattrs->DEV_TOTAL_MEMSZ);
	pthreadMutexInit(&gcontext->client_lock);
	dlist_init(&gcontext->client_list);

	pthreadCondInit(&gcontext->cond);
	pthreadMutexInit(&gcontext->lock);
	dlist_init(&gcontext->command_list);

	PG_TRY();
	{
		/* Open the listen socket for this GPU */
		gcontext->serv_fd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (gcontext->serv_fd < 0)
			elog(ERROR, "failed on socket(2): %m");
		snprintf(addr.sun_path, sizeof(addr.sun_path),
				 ".pg_strom.%u.gpu%u.sock",
				 PostmasterPid, gcontext->cuda_dindex);
		addr.sun_family = AF_UNIX;
		if (bind(gcontext->serv_fd, (struct sockaddr *) &addr, sizeof(addr)) != 0)
			elog(ERROR, "failed on bind('%s'): %m", addr.sun_path);
		if (listen(gcontext->serv_fd, 32) != 0)
			elog(ERROR, "failed on listen(2): %m");
		ev.events = EPOLLIN;
		ev.data.ptr = gcontext;
		if (epoll_ctl(gpuserv_epoll_fdesc,
					  EPOLL_CTL_ADD,
					  gcontext->serv_fd, &ev) != 0)
			elog(ERROR, "failed on epoll_ctl(2): %m");

		/* Setup raw CUDA context */
		rc = cuDeviceGet(&gcontext->cuda_device, dattrs->DEV_ID);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", cuStrError(rc));
		rc = cuCtxCreate(&gcontext->cuda_context,
						 CU_CTX_SCHED_AUTO,
						 gcontext->cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", cuStrError(rc));

		rc = cuCtxGetLimit(&stack_sz, CU_LIMIT_STACK_SIZE);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxGetLimit: %s", cuStrError(rc));
		stack_sz += 4096;
		rc = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, stack_sz);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxSetLimit: %s", cuStrError(rc));

		gpuservSetupGpuModule(gcontext);

		/* launch worker threads */
		for (i=0; i < pgstrom_max_async_gpu_tasks; i++)
		{
			if ((status = pthread_create(&gcontext->workers[i], NULL,
										 gpuservGpuWorkerMain, gcontext)) != 0)
			{
				__gpuContextTerminateWorkers(gcontext);
				elog(ERROR, "failed on pthread_create: %s", strerror(status));
			}
			gcontext->n_workers++;
		}
	}
	PG_CATCH();
	{
		if (gcontext->cuda_context)
			cuCtxDestroy(gcontext->cuda_context);
		if (gcontext->serv_fd >= 0)
			close(gcontext->serv_fd);
		free(gcontext);
		PG_RE_THROW();
	}
	PG_END_TRY();
	return gcontext;
}

/*
 * gpuservCleanupGpuContext
 */
static void
gpuservCleanupGpuContext(gpuContext *gcontext)
{
	CUresult	rc;

	__gpuContextTerminateWorkers(gcontext);

	while (!dlist_is_empty(&gcontext->client_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gcontext->client_list);
		gpuClient  *gclient = dlist_container(gpuClient, chain, dnode);

		if (close(gclient->sockfd) != 0)
			elog(LOG, "failed on close(sockfd): %m");
	}
	if (close(gcontext->serv_fd) != 0)
		elog(LOG, "failed on close(serv_fd): %m");
	rc = cuCtxDestroy(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(LOG, "failed on cuCtxDestroy: %s", cuStrError(rc));
}

/*
 * SIGTERM/SIGHUP handler
 */
static void
gpuservBgWorkerSignal(SIGNAL_ARGS)
{
	int		saved_errno = errno;

	gpuserv_bgworker_got_signal |= (1 << postgres_signal_arg);

	pg_memory_barrier();

	SetLatch(MyLatch);

	errno = saved_errno;
}

/*
 * gpuservClenupListenSocket
 */
static void
gpuservCleanupOnProcExit(int code, Datum arg)
{
	int		i;

	/* cleanup UNIX domain socket */
	for (i=0; i < numGpuDevAttrs; i++)
	{
		struct stat	stat_buf;
		char		path[MAXPGPATH];

		snprintf(path, sizeof(path),
				 ".pg_strom.%u.gpu%u.sock", PostmasterPid, i);
		if (stat(path, &stat_buf) == 0 &&
			(stat_buf.st_mode & S_IFMT) == S_IFSOCK)
		{
			if (unlink(path) < 0)
				elog(LOG, "failed on unlink('%s'): %m", path);
		}
	}
}

/*
 * gpuservBgWorkerMain
 */
void
gpuservBgWorkerMain(Datum arg)
{
	CUresult	rc;
	int			dindex;

	pqsignal(SIGTERM, gpuservBgWorkerSignal);
	pqsignal(SIGHUP,  gpuservBgWorkerSignal);
	BackgroundWorkerUnblockSignals();

	/* Disable MPS */
	if (setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1) != 0)
		elog(ERROR, "failed on setenv: %m");

	/* Registration of resource cleanup handler */
	dlist_init(&gpuserv_gpucontext_list);
	before_shmem_exit(gpuservCleanupOnProcExit, 0);

	/* Open epoll descriptor */
	gpuserv_epoll_fdesc = epoll_create(30);
	if (gpuserv_epoll_fdesc < 0)
		elog(ERROR, "failed on epoll_create: %m");

	/* Init GPU Context for each devices */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", cuStrError(rc));

	PG_TRY();
	{
		for (dindex=0; dindex < numGpuDevAttrs; dindex++)
		{
			gpuContext *gcontext = gpuservSetupGpuContext(dindex);
			dlist_push_tail(&gpuserv_gpucontext_list, &gcontext->chain);
		}
		gpuDirectOpenDriver();
		while (!gpuServiceGoingTerminate())
		{
			struct epoll_event	ep_ev;
			int		status;

			if (!PostmasterIsAlive())
				elog(FATAL, "unexpected postmaster dead");
			CHECK_FOR_INTERRUPTS();

			status = epoll_wait(gpuserv_epoll_fdesc, &ep_ev, 1, 4000);
			if (status < 0)
			{
				if (errno != EINTR)
				{
					elog(LOG, "failed on epoll_wait: %m");
					break;
				}
			}
			else if (status > 0)
			{
				/* errors on server socker? */
				if ((ep_ev.events & ~EPOLLIN) != 0)
					break;
				/* any connection pending? */
				if ((ep_ev.events & EPOLLIN) != 0)
					gpuservAcceptClient((gpuContext *)ep_ev.data.ptr);
			}
		}
	}
	PG_CATCH();
	{
		while (!dlist_is_empty(&gpuserv_gpucontext_list))
		{
			dlist_node *dnode = dlist_pop_head_node(&gpuserv_gpucontext_list);
			gpuContext *gcontext = dlist_container(gpuContext, chain, dnode);
			gpuservCleanupGpuContext(gcontext);
		}
		gpuDirectCloseDriver();
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* cleanup */
	while (!dlist_is_empty(&gpuserv_gpucontext_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gpuserv_gpucontext_list);
		gpuContext *gcontext = dlist_container(gpuContext, chain, dnode);
		gpuservCleanupGpuContext(gcontext);
	}
	gpuDirectCloseDriver();

	/*
	 * If it received only SIGHUP (no SIGTERM), try to restart rather than
	 * shutdown.
	 */
	if (gpuserv_bgworker_got_signal == (1 << SIGHUP))
		proc_exit(1);
}

/*
 * pgstrom_request_executor
 */
static void
pgstrom_request_executor(void)
{
	if (shmem_request_next)
		(*shmem_request_next)();
	RequestAddinShmemSpace(MAXALIGN(sizeof(bool)));
}

/*
 * pgstrom_startup_executor
 */
static void
pgstrom_startup_executor(void)
{
	bool    found;

	if (shmem_startup_next)
		(*shmem_startup_next)();
	gpuserv_debug_output = ShmemInitStruct("pg_strom.gpuserv_debug_output",
										   MAXALIGN(sizeof(pg_atomic_uint32)),
										   &found);
	pg_atomic_init_u32(gpuserv_debug_output, __gpuserv_debug_output_dummy);
}

/*
 * pgstrom_init_gpu_service
 */
void
pgstrom_init_gpu_service(void)
{
	BackgroundWorker worker;

	Assert(numGpuDevAttrs > 0);
	DefineCustomIntVariable("pg_strom.max_async_gpu_tasks",
							"Max number of asynchronous GPU tasks",
							NULL,
							&pgstrom_max_async_gpu_tasks,
							8,
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpu_mempool_segment_sz",
							"Segment size of GPU memory pool",
							NULL,
							&pgstrom_gpu_mempool_segment_sz_kb,
							1048576,	/* 1GB */
							262144,		/* 256MB */
							16777216,	/* 16GB */
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.gpu_mempool_max_ratio",
							 "GPU memory pool: maximum usable ratio for memory pool",
							 NULL,
							 &pgstrom_gpu_mempool_max_ratio,
							 0.50,		/* 50% */
							 0.20,		/* 20% */
							 0.80,		/* 80% */
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.gpu_mempool_min_ratio",
							 "GPU memory pool: minimum preserved ratio memory pool",
							 NULL,
							 &pgstrom_gpu_mempool_min_ratio,
							 0.05,		/*  8% */
							 0.0,		/*  0% */
							 pgstrom_gpu_mempool_max_ratio,
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpu_mempool_release_delay",
							"GPU memory pool: time to release device memory segment after the last chunk is released",
							NULL,
							&pgstrom_gpu_mempool_release_delay,
							5000,		/* 5sec */
							1,
							INT_MAX,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MS | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.gpuserv_debug_output",
							 "enables to generate debug message of GPU service",
							 NULL,
							 &__gpuserv_debug_output_dummy,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE | GUC_SUPERUSER_ONLY,
							 NULL,
							 gpuserv_debug_output_assign,
							 gpuserv_debug_output_show);
	for (int i=0; i < GPU_QUERY_BUFFER_NSLOTS; i++)
		dlist_init(&gpu_query_buffer_hslot[i]);

	memset(&worker, 0, sizeof(BackgroundWorker));
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = 5;
	snprintf(worker.bgw_name, BGW_MAXLEN, "PG-Strom GPU Service");
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_strom");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "gpuservBgWorkerMain");
	worker.bgw_main_arg = 0;
	RegisterBackgroundWorker(&worker);
	/* shared memory setup */
	shmem_request_next = shmem_request_hook;
	shmem_request_hook = pgstrom_request_executor;
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_executor;
}
