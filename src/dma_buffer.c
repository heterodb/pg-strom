/*
 * dma_buffer.c
 *
 * Routines to manage host-pinned DMA buffer and portable shared memory
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
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "lib/ilist.h"
#include "libpq/pqsignal.h"
#include "postmaster/autovacuum.h"
#include "postmaster/postmaster.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <unistd.h>

/*
 * dmaBufferChunk - chunk of DMA buffer.
 */
#define DMABUF_CHUNKSZ_MAX_BIT		34		/* 16GB */
#define DMABUF_CHUNKSZ_MIN_BIT		8		/* 256B */
#define DMABUF_CHUNKSZ_MAX			(1UL << DMABUF_CHUNKSZ_MAX_BIT)
#define DMABUF_CHUNKSZ_MIN			(1UL << DMABUF_CHUNKSZ_MIN_BIT)
#define DMABUF_CHUNK_DATA(chunk)	((chunk)->data)
#define DMABUF_CHUNK_MAGIC_CODE		0xDEADBEAF

typedef struct dmaBufferChunk
{
	dlist_node	free_chain;		/* link to free chunks, or zero if active */
	dlist_node	gcxt_chain;		/* link to GpuContext tracker */
	SharedGpuContext *shgcon;	/* GpuContext that owns this chunk */
	size_t		required;		/* required length */
	cl_uint		mclass;			/* class of the chunk size */
	cl_uint		magic_head;		/* = DMABUF_CHUNK_MAGIC_HEAD */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferChunk;

#define DMABUF_CHUNK_MAGIC_HEAD(chunk)			((chunk)->magic_head)
#define DMABUF_CHUNK_MAGIC_TAIL(chunk)			\
	*((cl_int *)((chunk)->data + INTALIGN((chunk)->required)))

/*
 * dmaBufferEntryHead / dmaBufferEntry
 *
 * It manages the current status of DMA buffers.
 */
#define SHMSEGMENT_NAME(namebuf, segment_id, revision)			\
	snprintf((namebuf),sizeof(namebuf),"/.pg_strom.%u.%u:%u",	\
			 PostPortNumber, (segment_id), (revision)>>1)

typedef struct dmaBufferSegment
{
	dlist_node	chain;		/* link to active/inactive list */
	cl_uint		segment_id;	/* (const) unique identifier of the segment */
	bool		persistent;	/* (const) this segment will never released */
	void	   *mmap_ptr;	/* (const) address to be attached */
	pg_atomic_uint32 revision; /* revision of the shared memory segment and
								* its status. Odd number, if segment exists.
								* Elsewhere, no segment exists. This field
								* is referenced in the signal handler, so
								* we don't use lock to update the field.
								*/
	slock_t		lock;		/* lock of the fields below */
	cl_int		num_chunks;	/* number of active chunks */
	dlist_head	free_chunks[DMABUF_CHUNKSZ_MAX_BIT + 1];
} dmaBufferSegment;

#define SHMSEG_EXISTS(revision)			(((revision) & 0x0001) != 0)

typedef struct dmaBufferSegmentHead
{
	pthread_rwlock_t rwlock;
	cl_uint		num_active_segments;
	dlist_head	active_segment_list;
	dlist_head	inactive_segment_list;
	dmaBufferSegment segments[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferSegmentHead;

/*
 * dmaBufferLocalMap - status of local mapping of dmaBuffer
 */
typedef struct dmaBufferLocalMap
{
	dmaBufferSegment *segment;	/* (const) reference to the segment */
	pthread_mutex_t lock;			/* lock to manage local memory map */
	uint32			revision;		/* revision number when mapped */
	bool			is_attached;	/* true, if segment is already attached */
} dmaBufferLocalMap;

/*
 * dmaBufferWaitList - tracker of backend processes that gets out of memory
 */
typedef struct dmaBufferWaitList
{
	slock_t			lock;
	dlist_head		wait_proc_list;
	/*
	 * NOTE: wait_proc_chain[MyProc->pgprocno] is associated for each backend
	 * or worker processes. If it is already in the wait_proc_list, its @prev
	 * and @next field should not be NULL.
	 */
	dlist_node		wait_proc_chain[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferWaitList;

/*
 * static variables
 */
static dmaBufferSegmentHead *dmaBufSegHead = NULL;	/* shared memory */
static dmaBufferLocalMap *dmaBufLocalMaps = NULL;
static dmaBufferWaitList *dmaBufWaitList = NULL;	/* shared memory */
static void	   *dma_segment_vaddr_head = NULL;
static void	   *dma_segment_vaddr_tail = NULL;
static size_t	dma_segment_size;
static int		dma_segment_size_kb;		/* GUC */
static int		num_logical_dma_segments;	/* GUC */
static int		num_segments_hardlimit;
static int		num_segments_softlimit;
static int		num_segments_guarantee;
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static void	  (*sighandler_sigsegv_orig)(int,siginfo_t *,void *) = NULL;
static void	  (*sighandler_sigbus_orig)(int,siginfo_t *,void *) = NULL;
static __thread	dmaBufferLocalMap *currentBufLocalMap = NULL;

/* for debug */
#ifdef PGSTROM_DEBUG
static __thread const char *last_caller_alloc_filename = NULL;
static __thread int			last_caller_alloc_lineno = -1;
static __thread const char *last_caller_free_filename = NULL;
static __thread int			last_caller_free_lineno = -1;
#endif

/*
 * dmaBufferCreateSegment - create a new DMA buffer segment
 *
 * NOTE: caller must have exclusive-lock on &dmaBufSegHead->rwlock
 */
static void
dmaBufferCreateSegment(dmaBufferSegment *seg)
{
	dmaBufferLocalMap  *l_map;
	dmaBufferChunk	   *chunk;
	char				namebuf[80];
	int					revision;
	int					fdesc;
	int					mclass;
	char			   *head_ptr;
	char			   *tail_ptr;

	Assert(seg->segment_id < num_logical_dma_segments);
	revision = pg_atomic_read_u32(&seg->revision);
	Assert(!SHMSEG_EXISTS(revision));	/* even number now */

	SHMSEGMENT_NAME(namebuf, seg->segment_id, revision);
	currentBufLocalMap = l_map = &dmaBufLocalMaps[seg->segment_id];

	/* Begin the critial section */
	if (pthread_mutex_lock(&l_map->lock) != 0)
	{
		currentBufLocalMap = NULL;
		wfatal("failed on pthread_mutex_lock(3)");
	}

	STROM_TRY();
	{
		/*
		 * NOTE: A ghost mapping may happen, if this process mapped
		 * the previous version on its private address space then some other
		 * process dropped the shared memory segment but this process had no
		 * chance to unmap.
		 * So, if we found a ghost mapping, unmap this area first.
		 */
		if (l_map->is_attached)
		{
			if (gpuserv_cuda_context)
			{
				CUresult	rc;

				Assert(IsGpuServerProcess());
				rc = cuMemHostUnregister(seg->mmap_ptr);
				if (rc != CUDA_SUCCESS)
					wfatal("failed on cuMemHostUnregister: %s", errorText(rc));
			}
			/* unmap the older/invalid segment first */
			if (munmap(seg->mmap_ptr, dma_segment_size) != 0)
				wfatal("failed on munmap('%s'): %m", namebuf);
			if (mmap(seg->mmap_ptr, dma_segment_size,
					 PROT_NONE,
					 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
					 -1, 0) != seg->mmap_ptr)
				wfatal("failed on mmap(PROT_NONE) for seg=%u at %p: %m",
					   seg->segment_id, seg->mmap_ptr);
			l_map->is_attached = false;
		}

		/*
		 * Open, expand and mmap the shared memory segment
		 */
		fdesc = shm_open(namebuf, O_RDWR | O_CREAT | O_TRUNC, 0600);
		if (fdesc < 0)
			werror("failed on shm_open('%s'): %m", namebuf);

		while (fallocate(fdesc, 0, 0, dma_segment_size) != 0)
		{
			if (errno == EINTR)
				continue;
			close(fdesc);
			shm_unlink(namebuf);
			werror("failed on fallocate(2): %m");
		}

		if (mmap(seg->mmap_ptr, dma_segment_size,
				 PROT_READ | PROT_WRITE,
				 MAP_SHARED | MAP_FIXED,
				 fdesc, 0) != seg->mmap_ptr)
		{
			close(fdesc);
			shm_unlink(namebuf);
			werror("failed on mmap: %m");
		}
		close(fdesc);

		if (gpuserv_cuda_context)
		{
			CUresult	rc;

			Assert(IsGpuServerProcess());
			rc = cuMemHostRegister(seg->mmap_ptr, dma_segment_size, 0);
			if (rc != CUDA_SUCCESS)
			{
				if (munmap(seg->mmap_ptr, dma_segment_size) != 0)
					wfatal("failed on munmap('%s'): %m", namebuf);
				if (mmap(seg->mmap_ptr, dma_segment_size,
						 PROT_NONE,
						 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
						 -1, 0) != seg->mmap_ptr)
					wfatal("failed on mmap(PROT_NONE) for seg=%u at %p: %m",
						   seg->segment_id, seg->mmap_ptr);
				werror("failed on cuMemHostRegister: %s", errorText(rc));
			}
		}

		/* successfully mapped, init this segment */
		for (mclass=0; mclass <= DMABUF_CHUNKSZ_MAX_BIT; mclass++)
			dlist_init(&seg->free_chunks[mclass]);
		head_ptr = (char *)seg->mmap_ptr;
		tail_ptr = (char *)seg->mmap_ptr + dma_segment_size;
		mclass = DMABUF_CHUNKSZ_MAX_BIT;
		while (mclass >= DMABUF_CHUNKSZ_MIN_BIT)
		{
			if (head_ptr + (1UL << mclass) > tail_ptr)
			{
				mclass--;
				continue;
			}
			chunk = (dmaBufferChunk *)head_ptr;
			memset(chunk, 0, offsetof(dmaBufferChunk, data));
			chunk->mclass = mclass;
			DMABUF_CHUNK_MAGIC_HEAD(chunk) = DMABUF_CHUNK_MAGIC_CODE;

			dlist_push_head(&seg->free_chunks[mclass], &chunk->free_chain);

			head_ptr += (1UL << mclass);
		}
		seg->num_chunks = 0;

		/* Also, update local mapping */
		l_map->is_attached = true;
		l_map->revision = pg_atomic_add_fetch_u32(&seg->revision, 1);
	}
	STROM_CATCH();
	{
		/* unlock */
		pthread_mutex_unlock(&l_map->lock);
		currentBufLocalMap = NULL;
		STROM_RE_THROW();
	}
	STROM_END_TRY();
	/* end of the critical section */
	pthread_mutex_unlock(&l_map->lock);
	currentBufLocalMap = NULL;

	wdebug("PID=%u dmaBufferCreateSegment seg_id=%u rev=%u\n"
#ifdef PGSTROM_DEBUG
		   " called by %s:%d"
#endif
		   ,getpid()
		   ,seg->segment_id
		   ,l_map->revision
#ifdef PGSTROM_DEBUG
		   ,last_caller_alloc_filename
		   ,last_caller_alloc_lineno
#endif
		);
}

/*
 * dmaBufferDetachSegment - detach a DMA buffer and delete shared memory
 * segment. If somebody still mapped this segment, further reference will
 * cause SIGBUS then signal handler will detach this segment.
 *
 * NOTE: caller must have exclusive-lock on &dmaBufSegHead->rwlock
 */
static void
dmaBufferDetachSegment(dmaBufferSegment *seg)
{
	dmaBufferLocalMap *l_map;
	char		namebuf[80];
	int			fdesc;
	uint32		revision = pg_atomic_fetch_add_u32(&seg->revision, 1);
	CUresult	rc;

	Assert(SHMSEG_EXISTS(revision));
	wdebug("PID=%u dmaBufferDetachSegment seg_id=%u rev=%u"
#ifdef PGSTROM_DEBUG
		   " called by %s:%d"
#endif
		   , getpid(), seg->segment_id, revision
#ifdef PGSTROM_DEBUG
		   ,last_caller_free_filename
		   ,last_caller_free_lineno
#endif
		);

	/* BEGIN Critical Section */
	currentBufLocalMap = l_map = &dmaBufLocalMaps[seg->segment_id];
	if (pthread_mutex_lock(&l_map->lock) != 0)
	{
		currentBufLocalMap = NULL;
		wfatal("failed on pthread_mutex_lock(3)");
	}

	STROM_TRY();
	{
		/*
		 * If caller process already attach this segment, we unmap
		 * this region altogether.
		 */
		if (l_map->is_attached)
		{
			/* unregister host pinned memory, if server process */
			if (gpuserv_cuda_context)
			{
				Assert(IsGpuServerProcess());
				rc = cuMemHostUnregister(seg->mmap_ptr);
				if (rc != CUDA_SUCCESS)
					wfatal("failed on cuMemHostUnregister: %s", errorText(rc));
			}
			/* unmap segment from private virtula address space */
			if (munmap(seg->mmap_ptr, dma_segment_size) != 0)
				wfatal("failed on munmap(seg=%u:%u at %p): %m",
					   seg->segment_id, l_map->revision/2, seg->mmap_ptr);
			/* and map invalid area instead */
			if (mmap(seg->mmap_ptr, dma_segment_size,
					 PROT_NONE,
					 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
					 -1, 0) != seg->mmap_ptr)
				wfatal("failed on mmap(PROT_NONE) for seg=%u at %p: %m",
					   seg->segment_id, seg->mmap_ptr);
			l_map->is_attached = false;
		}

		/*
		 * NOTE: dmaBufferDetachSegment() can never unmap this segment from
		 * the virtual address space of other processes, of course.
		 * On the other hands, this shared memory segment is already truncated
		 * to zero, thus, any access on the ghost mapping area will cause
		 * SIGBUS exception. It shall be processed by the signal handler, and
		 * then, this routine will unmap the old ghost segment.
		 */
		SHMSEGMENT_NAME(namebuf, seg->segment_id, revision);
		fdesc = shm_open(namebuf, O_RDWR | O_TRUNC, 0600);
		if (fdesc < 0)
			wfatal("failed on shm_open('%s', O_TRUNC): %m", namebuf);
		close(fdesc);

		if (shm_unlink(namebuf) < 0)
			wfatal("failed on shm_unlink('%s'): %m", namebuf);
	}
	STROM_CATCH();
	{
		/* unlock */
		pthread_mutex_unlock(&l_map->lock);
		currentBufLocalMap = NULL;
		STROM_RE_THROW();
	}
	STROM_END_TRY();

	/* END Critical Section */
	pthread_mutex_unlock(&l_map->lock);
	currentBufLocalMap = NULL;

	Assert(!SHMSEG_EXISTS(pg_atomic_read_u32(&seg->revision)));
}

/*
 * dmaBufferAttachSegmentOnDemand
 *
 * A signal handler to be called on SIGBUS/SIGSEGV. If memory address which
 * caused a fault is in a range of virtual DMA buffer mapping, it tries to
 * map the shared buffer page.
 * Note that this handler never create a new DMA buffer segment but maps
 * an existing range, because nobody (except for buggy code) will point
 * the location which not mapped yet.
 */
static void
dmaBufferAttachSegmentOnDemand(int signum, siginfo_t *siginfo, void *unused)
{
	static __thread bool internal_error = false;
	int			save_errno;

	if (!internal_error)
	{
		internal_error = true;	/* prevent infinite loop */
		save_errno = errno;
		PG_SETMASK(&BlockSig);

		if (dmaBufSegHead &&
			dma_segment_vaddr_head <= siginfo->si_addr &&
			dma_segment_vaddr_tail >  siginfo->si_addr)
		{
			dmaBufferSegment   *seg;
			dmaBufferLocalMap  *l_map;
			int			seg_id;
			uint32		revision;
			char		namebuf[80];
			int			fdesc;
			CUresult	rc;

			seg_id = ((uintptr_t)siginfo->si_addr -
					  (uintptr_t)dma_segment_vaddr_head) / dma_segment_size;
			Assert(seg_id < num_logical_dma_segments);
			seg = &dmaBufSegHead->segments[seg_id];

			revision = pg_atomic_read_u32(&seg->revision);
			if (!SHMSEG_EXISTS(revision))
			{
				fprintf(stderr, "%s: got %s on %p (segid=%u %p at rev=%u), "
						"but shared memory segment is not available\n",
						__FUNCTION__, strsignal(signum), siginfo->si_addr,
						seg->segment_id, seg->mmap_ptr, revision);
				goto normal_crash;
			}

			l_map = &dmaBufLocalMaps[seg_id];
			if (currentBufLocalMap == l_map)
			{
				fprintf(stderr,
						"%s: got %s on %p (segid=%u at %p, rev=%u), "
						"in the critical section of the same local mapping\n",
						__FUNCTION__, strsignal(signum), siginfo->si_addr,
						seg->segment_id, seg->mmap_ptr, revision);
				goto normal_crash;
			}
			/* ok, mutex_lock shall not lead self dead-lock */
			if (pthread_mutex_lock(&l_map->lock) != 0)
			{
				fprintf(stderr,
						"%s: got %s on %p (segid=%u at %p, rev=%u), "
						"failed on pthread_mutex_lock\n",
						__FUNCTION__, strsignal(signum), siginfo->si_addr,
						seg->segment_id, seg->mmap_ptr, revision);
				goto normal_crash;
			}
			/* BEGIN critical section */
			if (l_map->is_attached)
			{
				if (l_map->revision == revision)
				{
					pthread_mutex_unlock(&l_map->lock);
#ifdef NOT_USED
					fprintf(stderr,
							"%s: got %s on %p (segid=%u at %p, rev=%u), "
							"but latest revision is already mapped\n",
							__FUNCTION__, strsignal(signum), siginfo->si_addr,
							seg->segment_id, seg->mmap_ptr, revision);
#endif
					goto segment_already_mapped;
				}

				/*
				 * unregister host pinned memory, if any
				 *
				 * If gpuserv_cuda_context==NULL, it means this process is not
				 * GPU server process or GPU server process is going to die.
				 */
				if (gpuserv_cuda_context)
				{
					Assert(IsGpuServerProcess());
					rc = cuMemHostUnregister(seg->mmap_ptr);
					if (rc != CUDA_SUCCESS)
					{
						pthread_mutex_unlock(&l_map->lock);
						fprintf(stderr,
						"%s: failed on cuMemHostUnregister(id=%u at %p): %s\n",
								__FUNCTION__, seg->segment_id, seg->mmap_ptr,
								errorText(rc));
						goto normal_crash;
					}
				}
				/* unmap the old/invalid segment */
				if (munmap(seg->mmap_ptr, dma_segment_size) != 0)
				{
					pthread_mutex_unlock(&l_map->lock);
					fprintf(stderr, "%s: failed on munmap (id=%u at %p): %m\n",
							__FUNCTION__, seg->segment_id, seg->mmap_ptr);
					goto normal_crash;
				}
				l_map->is_attached = false;
			}
			/* open an "existing" shared memory segment */
			SHMSEGMENT_NAME(namebuf, seg->segment_id, revision);
			fdesc = shm_open(namebuf, O_RDWR, 0600);
			if (fdesc < 0)
			{
				pthread_mutex_unlock(&l_map->lock);
				fprintf(stderr, "%s: got %s on segment (id=%u at %p), "
						"but failed on shm_open('%s'): %m\n",
						__FUNCTION__, strsignal(signum),
						seg->segment_id, seg->mmap_ptr, namebuf);
				goto normal_crash;
			}

			/*
			 * NOTE: no need to call ftruncate(2) here because somebody
			 * who created the segment should already expand the segment
			 */

			/* map this shared memory segment */
			if (mmap(seg->mmap_ptr, dma_segment_size,
					 PROT_READ | PROT_WRITE,
					 MAP_SHARED | MAP_FIXED,
					 fdesc, 0) != seg->mmap_ptr)
			{
				close(fdesc);
				pthread_mutex_unlock(&l_map->lock);
				fprintf(stderr, "%s: got %s on segment (id=%u at %p), "
						"but unable to mmap(2) the segment '%s': %m\n",
						__FUNCTION__, strsignal(signum),
						seg->segment_id, seg->mmap_ptr, namebuf);
				goto normal_crash;
			}
			close(fdesc);

			/*
			 * Registers the segment as a host pinned memory, if GPU server
			 * process with healthy status. If CUDA context is not valid,
			 * it means GPU server is going to die.
			 */
			if (gpuserv_cuda_context)
			{
				Assert(IsGpuServerProcess());
				rc = cuMemHostRegister(seg->mmap_ptr, dma_segment_size, 0);
				if (rc != CUDA_SUCCESS)
				{
					pthread_mutex_unlock(&l_map->lock);
					fprintf(stderr,
						  "%s: failed on cuMemHostRegister(id=%u at %p): %s\n",
							__FUNCTION__, seg->segment_id, seg->mmap_ptr,
							errorText(rc));
					goto normal_crash;
				}
			}

			/* ok, this segment is successfully mapped */
			l_map->revision = revision;
			l_map->is_attached = true;
			pthread_mutex_unlock(&l_map->lock);
#if NOT_USED
			fprintf(stderr, "%s: pid=%u got %s, then attached shared memory "
					"segment (id=%u at %p, rev=%u)\n",
					__FUNCTION__, MyProcPid, strsignal(signum),
					seg->segment_id, seg->mmap_ptr, revision);
#endif
		segment_already_mapped:
			PG_SETMASK(&UnBlockSig);

			errno = save_errno;
			internal_error = false;
			return;		/* problem solved */
		}
	normal_crash:
		PG_SETMASK(&UnBlockSig);
		errno = save_errno;
	}

	if (signum == SIGSEGV)
	{
		if (sighandler_sigsegv_orig)
			(*sighandler_sigsegv_orig)(signum, siginfo, unused);
		else
			abort();
	}
	else if (signum == SIGBUS)
	{
		if (sighandler_sigbus_orig)
			(*sighandler_sigbus_orig)(signum, siginfo, unused);
		else
			abort();
	}
	else
	{
		fprintf(stderr, "%s received %s, panic\n",
				__FUNCTION__, strsignal(signum));
		abort();
	}
	internal_error = false;		/* reset */
}


/*
 * dmaBufferSplitChunk
 *
 * NOTE: caller must have &dmaBufferSegment->lock
 */
static bool
dmaBufferSplitChunk(dmaBufferSegment *segment, int mclass)
{
	dlist_node	   *dnode;
	dmaBufferChunk *chunk_1;
	dmaBufferChunk *chunk_2;

	if (mclass >= DMABUF_CHUNKSZ_MAX_BIT)
		return false;
	if (dlist_is_empty(&segment->free_chunks[mclass]))
	{
		if (!dmaBufferSplitChunk(segment, mclass + 1))
			return false;
	}
	Assert(!dlist_is_empty(&segment->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&segment->free_chunks[mclass]);
	chunk_1 = dlist_container(dmaBufferChunk, free_chain, dnode);
	Assert(chunk_1->mclass == mclass);
	Assert(chunk_1->magic_head == DMABUF_CHUNK_MAGIC_CODE);

	/* earlier half */
	memset(chunk_1, 0, offsetof(dmaBufferChunk, data));
	chunk_1->mclass = mclass - 1;
	chunk_1->magic_head = DMABUF_CHUNK_MAGIC_CODE;
	dlist_push_tail(&segment->free_chunks[mclass - 1], &chunk_1->free_chain);

	/* later half */
	chunk_2 = (dmaBufferChunk *)((char *)chunk_1 + (1UL << (mclass - 1)));
	memset(chunk_2, 0, offsetof(dmaBufferChunk, data));
	chunk_2->mclass = mclass - 1;
	chunk_2->magic_head = DMABUF_CHUNK_MAGIC_CODE;
	dlist_push_tail(&segment->free_chunks[mclass - 1], &chunk_2->free_chain);

	return true;
}

/*
 * dmaBufferAllocChunk
 *
 * NOTE: caller must have shared-lock on &dmaBufSegHead->rwlock
 */
static void *
dmaBufferAllocChunk(dmaBufferSegment *seg, int mclass, Size required)
{
	dmaBufferChunk *chunk = NULL;
	dlist_node	   *dnode;

	Assert(mclass <= DMABUF_CHUNKSZ_MAX_BIT);
	SpinLockAcquire(&seg->lock);
	if (dlist_is_empty(&seg->free_chunks[mclass]))
	{
		if (!dmaBufferSplitChunk(seg, mclass + 1))
			goto out;
	}
	Assert(!dlist_is_empty(&seg->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&seg->free_chunks[mclass]);
	chunk = dlist_container(dmaBufferChunk, free_chain, dnode);
	Assert(chunk->mclass == mclass);
	Assert(DMABUF_CHUNK_MAGIC_HEAD(chunk) == DMABUF_CHUNK_MAGIC_CODE);

	/* init dmaBufferChunk */
	memset(&chunk->free_chain, 0, sizeof(dlist_node));
	chunk->shgcon = NULL;	/* caller will set */
	chunk->required = required;
	chunk->mclass = mclass;
	DMABUF_CHUNK_MAGIC_HEAD(chunk) = DMABUF_CHUNK_MAGIC_CODE;
	DMABUF_CHUNK_MAGIC_TAIL(chunk) = DMABUF_CHUNK_MAGIC_CODE;

	/* update dmaBufferSegment status */
	seg->num_chunks++;
out:
	SpinLockRelease(&seg->lock);
	return chunk;
}

/*
 * dmaBufferAlloc
 */
static void *
dmaBufferAllocInternal(SharedGpuContext *shgcon, Size required)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	dlist_node		   *dnode;
	dlist_iter			iter;
	Size				chunk_size;
	int					mclass;
	bool				has_exclusive_lock = false;

	/* normalize the required size to 2^N of chunks size */
	chunk_size = MAXALIGN(offsetof(dmaBufferChunk, data) +
						  required +
						  sizeof(cl_uint));
	chunk_size = Max(chunk_size, DMABUF_CHUNKSZ_MIN);
	mclass = get_next_log2(chunk_size);
	if ((1UL << mclass) > dma_segment_size)
		elog(ERROR, "DMA buffer request %zu MB too large", required >> 20);

	/* find out an available segment */
	pthreadRWLockReadLock(&dmaBufSegHead->rwlock);
	STROM_TRY();
	{
	retry:
		dlist_foreach(iter, &dmaBufSegHead->active_segment_list)
		{
			seg = dlist_container(dmaBufferSegment, chain, iter.cur);
			Assert(SHMSEG_EXISTS(pg_atomic_read_u32(&seg->revision)));

			chunk = dmaBufferAllocChunk(seg, mclass, required);
			if (chunk)
				goto found;
		}

		/* Oops, no available free chunks in the active list */
		if (!has_exclusive_lock)
		{
			pthreadRWLockUnlock(&dmaBufSegHead->rwlock);
			pthreadRWLockWriteLock(&dmaBufSegHead->rwlock);
			has_exclusive_lock = true;
			goto retry;
		}

		/*
		 * check for resource limitation
		 *
		 * XXX - Does it make sense to limit creation of a new segment?
		 * Once a segment is constucted, it can be used by both of backend
		 * and GPU server process.
		 */
		if (dmaBufSegHead->num_active_segments >= (IsGpuServerProcess()
												   ? num_segments_hardlimit
												   : num_segments_softlimit))
		{
			chunk = NULL;
			goto found;
		}

		if (dlist_is_empty(&dmaBufSegHead->inactive_segment_list))
			werror("Out of DMA buffer segment");

		/*
		 * Create a new DMA buffer segment
		 */
		dnode = dlist_pop_head_node(&dmaBufSegHead->inactive_segment_list);
		seg = dlist_container(dmaBufferSegment, chain, dnode);
		Assert(!SHMSEG_EXISTS(pg_atomic_read_u32(&seg->revision)));
		STROM_TRY();
		{
			dmaBufferCreateSegment(seg);
		}
		STROM_CATCH();
		{
			dlist_push_head(&dmaBufSegHead->inactive_segment_list,
							&seg->chain);
			STROM_RE_THROW();
		}
		STROM_END_TRY();
		dlist_push_head(&dmaBufSegHead->active_segment_list, &seg->chain);
		dmaBufSegHead->num_active_segments++;

		/*
		 * allocation of a new chunk from the new chunk to ensure num_chunks
		 * is larger than zero.
		 */
		chunk = dmaBufferAllocChunk(seg, mclass, required);
		Assert(chunk != NULL);
	found:
		;
	}
	STROM_CATCH();
	{
		pthreadRWLockUnlock(&dmaBufSegHead->rwlock);
		STROM_RE_THROW();
	}
	STROM_END_TRY();
	pthreadRWLockUnlock(&dmaBufSegHead->rwlock);

	if (chunk != NULL)
	{
		/* track this chunk with GpuContext */
		SpinLockAcquire(&shgcon->lock);
		chunk->shgcon = shgcon;
		dlist_push_tail(&shgcon->dma_buffer_list,
						&chunk->gcxt_chain);
		SpinLockRelease(&shgcon->lock);
#ifdef NOT_USED
		/*
		 * NOTE: It may make sense for debugging, however, caller
		 * exactly knows which area needs to be cleared and initialized.
		 Not a job of allocator.
		*/
		memset(chunk->data, 0xAE, chunk->required);
#endif
		return chunk->data;
	}
	return NULL;
}

void *
__dmaBufferAlloc(GpuContext *gcontext, Size required,
				 const char *filename, int lineno)
{
#ifdef PGSTROM_DEBUG
	last_caller_alloc_filename = filename;
	last_caller_alloc_lineno   = lineno;
#endif
	return dmaBufferAllocInternal(gcontext->shgcon, required);
}

/*
 * pointer_validation - rough pointer validation for realloc/free
 */
static dmaBufferChunk *
pointer_validation(void *pointer, dmaBufferSegment **p_seg)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	int					seg_id;

	chunk = (dmaBufferChunk *)
		((char *)pointer - offsetof(dmaBufferChunk, data));
	if (!dmaBufSegHead ||
		(void *)chunk <  dma_segment_vaddr_head ||
		(void *)chunk >= dma_segment_vaddr_tail)
		werror("Bug? %p is out of DMA buffer", pointer);

	seg_id = ((uintptr_t)chunk -
			  (uintptr_t)dma_segment_vaddr_head) / dma_segment_size;
	Assert(seg_id < num_logical_dma_segments);
	seg = &dmaBufSegHead->segments[seg_id];
	Assert(SHMSEG_EXISTS(pg_atomic_read_u32(&seg->revision)));

	if (offsetof(dmaBufferChunk, data) +
		chunk->required + sizeof(cl_uint) > (1UL << chunk->mclass) ||
		DMABUF_CHUNK_MAGIC_HEAD(chunk) != DMABUF_CHUNK_MAGIC_CODE ||
		DMABUF_CHUNK_MAGIC_TAIL(chunk) != DMABUF_CHUNK_MAGIC_CODE)
		werror("Bug? DMA buffer %p is corrupted", pointer);

	if (chunk->free_chain.prev != NULL ||
		chunk->free_chain.next != NULL)
		werror("Bug? %p points a free DMA buffer", pointer);

	if (p_seg)
		*p_seg = seg;
	return chunk;
}

/*
 * dmaBufferRealloc
 */
void *
__dmaBufferRealloc(void *pointer, Size required,
				   const char *filename, int lineno)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	Size				chunk_size;
	int					mclass;
	void			   *result;

#ifdef PGSTROM_DEBUG
	last_caller_alloc_filename = filename;
	last_caller_alloc_lineno   = lineno;
#endif
	/* sanity checks */
	chunk = pointer_validation(pointer, &seg);

	/* normalize the new required size to 2^N of chunks size */
	chunk_size = MAXALIGN(offsetof(dmaBufferChunk, data) +
						  required +
						  sizeof(cl_uint));
	chunk_size = Max(chunk_size, DMABUF_CHUNKSZ_MIN);
	mclass = get_next_log2(chunk_size);

	if (mclass == chunk->mclass)
	{
		/* no need to expand/shrink */
		chunk->required = required;
		DMABUF_CHUNK_MAGIC_TAIL(chunk) = DMABUF_CHUNK_MAGIC_CODE;
		return chunk->data;
	}
	else if (mclass < chunk->mclass)
	{
		/* no need to expand, but release unused area */
		char   *head_ptr = (char *)chunk + (1UL << mclass);
		char   *tail_ptr = (char *)chunk + (1UL << chunk->mclass);
		int		shift = chunk->mclass;

		SpinLockAcquire(&seg->lock);
		/* shrink the original chunk */
		chunk->required = required;
		chunk->mclass = mclass;
		DMABUF_CHUNK_MAGIC_TAIL(chunk) = DMABUF_CHUNK_MAGIC_CODE;

		/*
		 * Unlike dmaBufferFree, we have no chance to merge with neighbor 
		 * chunks due to 2^N boundary, so we just add fractions to the
		 * free chunk list.
		 */
		while (shift >= mclass)
		{
			dmaBufferChunk *temp;

			if (head_ptr + (1UL << shift) > tail_ptr)
			{
				shift--;
				continue;
			}
			temp = (dmaBufferChunk *)(tail_ptr - (1UL << shift));
			memset(temp, 0, offsetof(dmaBufferChunk, data));
			temp->mclass = shift;
			DMABUF_CHUNK_MAGIC_HEAD(temp) = DMABUF_CHUNK_MAGIC_CODE;
			dlist_push_head(&seg->free_chunks[shift], &temp->free_chain);

			tail_ptr -= (1UL << shift);
		}
		SpinLockRelease(&seg->lock);

		Assert((char *)chunk + (1UL << mclass) == (char *)tail_ptr);

		return chunk->data;
	}
	/* allocate a larger new chunk, then copy the contents */
	result = dmaBufferAllocInternal(chunk->shgcon, required);
	memcpy(result, chunk->data, chunk->required);
	__dmaBufferFree(pointer, filename, lineno);

	return result;
}

/*
 * dmaBufferValidatePtr - validate the supplied pointer
 */
bool
dmaBufferValidatePtr(void *pointer)
{
	bool	result = true;

	STROM_TRY();
	{
		(void) pointer_validation(pointer, NULL);
	}
	STROM_CATCH();
	{
		if (IsGpuServerProcess() >= 0)
			FlushErrorState();	/* only if single-thread mode */
		result = false;
	}
	STROM_END_TRY();

	return result;
}

/*
 * dmaBufferSize - tells the length caller can use
 */
Size
dmaBufferSize(void *pointer)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;

	chunk = pointer_validation(pointer, &seg);

	return chunk->required;
}

/*
 * dmaBufferChunkSize - return the length physically allocated (always 2^N)
 */
Size
dmaBufferChunkSize(void *pointer)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;

	chunk = pointer_validation(pointer, &seg);

	return (1UL << chunk->mclass);
}

/*
 * dmaBufferFree
 */
void
__dmaBufferFree(void *pointer,
				const char *filename, int lineno)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	dmaBufferChunk	   *buddy;
	SharedGpuContext   *shgcon;
	bool				has_shared_lock = false;

#ifdef PGSTROM_DEBUG
	last_caller_free_filename = filename;
	last_caller_free_lineno = lineno;
#endif
	/* sanity checks */
	chunk = pointer_validation(pointer, &seg);
#ifdef NOT_USED
	/*
	 * NOTE: the memset() below usually works to detect incorrect memory
	 * usage, however, additional CPU cycles are not ignorable. So, it
	 * shall be commented out unless developer does not need it for
	 * investigation.
	 */
	memset(chunk->data, 0xf5, chunk->required);
#endif
	/* detach chunk from the GpuContext */
	shgcon = chunk->shgcon;
	SpinLockAcquire(&shgcon->lock);
	dlist_delete(&chunk->gcxt_chain);
	SpinLockRelease(&shgcon->lock);
	chunk->shgcon = NULL;
	memset(&chunk->gcxt_chain, 0, sizeof(dlist_node));

	/* try to merge the neighbor free chunk */
retry:
	SpinLockAcquire(&seg->lock);

	/*
	 * NOTE: If and when @num_chunks gets zero, segment may be detached and
	 * released by the background worker. It shall be protected with exclusive
	 * lock on the dmaBufferSegmentHead->rwlock, so we preliminary acquire
	 * the shared lock to avoid concurrent access to the segment.
	 */
	Assert(seg->num_chunks > 0);
	if (seg->num_chunks == 1)
	{
		if (!has_shared_lock)
		{
			SpinLockRelease(&seg->lock);
			pthreadRWLockReadLock(&dmaBufSegHead->rwlock);
			has_shared_lock = true;
			goto retry;
		}
	}

	/*
	 * Try to merge with the neighbor chunks
	 */
	while (chunk->mclass <= DMABUF_CHUNKSZ_MAX_BIT)
	{
		Size	offset = (uintptr_t)chunk - (uintptr_t)seg->mmap_ptr;

		if ((offset & (1UL << chunk->mclass)) == 0)
		{
			buddy = (dmaBufferChunk *)((char *)chunk + (1UL << chunk->mclass));

			if ((char *)buddy >= (char *)seg->mmap_ptr + dma_segment_size)
				break;		/* out of the segment */
			Assert(DMABUF_CHUNK_MAGIC_HEAD(buddy) == DMABUF_CHUNK_MAGIC_CODE);
			/* Is the buddy merginable? */
			if (buddy->mclass != chunk->mclass ||
				!buddy->free_chain.prev ||
				!buddy->free_chain.next)
				break;
			/* OK, let's merge them */
			Assert(buddy->shgcon == NULL &&
				   !buddy->gcxt_chain.prev &&
				   !buddy->gcxt_chain.next);
			dlist_delete(&buddy->free_chain);
			chunk->mclass++;
		}
		else
		{
			buddy = (dmaBufferChunk *)((char *)chunk - (1UL << chunk->mclass));

			if ((char *)buddy < (char *)seg->mmap_ptr)
				break;		/* out of the segment */
			Assert(DMABUF_CHUNK_MAGIC_HEAD(buddy) == DMABUF_CHUNK_MAGIC_CODE);
			/* Is the buddy merginable? */
			if (buddy->mclass != chunk->mclass ||
				!buddy->free_chain.prev ||
				!buddy->free_chain.next)
				break;
			/* OK, let's merge them */
			Assert(buddy->shgcon == NULL &&
				   !buddy->gcxt_chain.prev &&
				   !buddy->gcxt_chain.next);
			dlist_delete(&buddy->free_chain);
			buddy->mclass++;

			chunk = buddy;
		}
	}
	/* insert the chunk (might be merged) to the free list */
	dlist_push_head(&seg->free_chunks[chunk->mclass], &chunk->free_chain);
	seg->num_chunks--;

	/*
	 * NOTE: Segment shall not be moved to inactive segment list even if
	 * its number of active chunks gets zero, because background worker
	 * detach and release these segments after some delay.
	 * Creation of shared segment and registration to CUDA runtime are
	 * not lightweight operation. Even if no active chunks exist on
	 * a segment, it shall be reused near future.
	 */
	SpinLockRelease(&seg->lock);

	if (has_shared_lock)
		pthreadRWLockUnlock(&dmaBufSegHead->rwlock);
}

/*
 * dmaBufferFree - unlink all the DMA buffer chunks tracked by the supplied
 * shared gpu context
 */
void
__dmaBufferFreeAll(SharedGpuContext *shgcon,
				   const char *filename, int lineno)
{
	dmaBufferChunk *chunk;
	dlist_node	   *dnode;

#ifdef PGSTROM_DEBUG
	last_caller_free_filename = filename;
	last_caller_free_lineno = lineno;
#endif
	while (!dlist_is_empty(&shgcon->dma_buffer_list))
	{
		dnode = dlist_pop_head_node(&shgcon->dma_buffer_list);
		chunk = dlist_container(dmaBufferChunk, gcxt_chain, dnode);
		Assert(chunk->shgcon == shgcon);
		dmaBufferFree(chunk->data);
	}
}

/*
 * dmaBufferMaxAllocSize
 */
Size
dmaBufferMaxAllocSize(void)
{
	int		mclass = get_prev_log2(dma_segment_size);

	return (Size)(1UL << mclass)
		- (MAXALIGN(offsetof(dmaBufferChunk, data)) +
		   MAXALIGN(sizeof(cl_uint)));
}

/*
 * dmaBufferCleanupOnPostmasterExit - clean up all the active DMA buffers
 */
static void
dmaBufferCleanupOnPostmasterExit(int code, Datum arg)
{
	if (dmaBufSegHead && MyProcPid == PostmasterPid)
	{
		dlist_iter	iter;
		char		namebuf[80];
		int			fdesc;

		dlist_foreach(iter, &dmaBufSegHead->active_segment_list)
		{
			dmaBufferSegment *seg = dlist_container(dmaBufferSegment,
													chain, iter.cur);
			SHMSEGMENT_NAME(namebuf, seg->segment_id,
							pg_atomic_read_u32(&seg->revision));
			fdesc = shm_open(namebuf, O_RDWR | O_TRUNC, 0600);
			if (fdesc < 0)
				elog(WARNING, "failed to open active DMA buffer '%s': %m",
					 namebuf);
			else
			{
				close(fdesc);

				if (shm_unlink(namebuf) != 0)
					elog(WARNING,
						 "failed to unlink active DMA buffer '%s': %m",
						 namebuf);
			}
		}
	}
	dmaBufSegHead = NULL;	/* shared memory segment no longer valid */
}

/*
 * pgstrom_dma_buffer_alloc - wrapper to dmaBufferAlloc
 */
Datum
pgstrom_dma_buffer_alloc(PG_FUNCTION_ARGS)
{
	int64	required = PG_GETARG_INT64(0);
	void   *pointer = dmaBufferAlloc(MasterGpuContext(), required);

	PG_RETURN_INT64(pointer);
}
PG_FUNCTION_INFO_V1(pgstrom_dma_buffer_alloc);

/*
 * pgstrom_dma_buffer_free - wrapper to dmaBufferFree
 */
Datum
pgstrom_dma_buffer_free(PG_FUNCTION_ARGS)
{
	int64	pointer = PG_GETARG_INT64(0);

	dmaBufferFree((void *)pointer);
	PG_RETURN_BOOL(true);
}
PG_FUNCTION_INFO_V1(pgstrom_dma_buffer_free);

/*
 * pgstrom_dma_buffer_info dump the current status of DMA buffer
 */
Datum
pgstrom_dma_buffer_info(PG_FUNCTION_ARGS)
{
	struct {
		cl_int		seg_id;
		cl_int		rev;
		cl_int		mclass;
		cl_int		n_actives;
		cl_int		n_frees;
	} *dma_seg_info;
	FuncCallContext *fncxt;
	Datum			values[5];
	bool			isnull[5];
	HeapTuple		tuple;
	List		   *results = NIL;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		dlist_iter		iter;
		int				i;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(5, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "seg_id",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "revision",
                           INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "mclass",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "actives",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "frees",
						   INT4OID, -1, 0);

		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		pthreadRWLockReadLock(&dmaBufSegHead->rwlock);
		dlist_foreach(iter, &dmaBufSegHead->active_segment_list)
		{
			dmaBufferSegment   *seg = dlist_container(dmaBufferSegment,
													  chain, iter.cur);
			SpinLockAcquire(&seg->lock);
			PG_TRY();
			{
				for (i =  DMABUF_CHUNKSZ_MIN_BIT;
					 i <= DMABUF_CHUNKSZ_MAX_BIT;
					 i++)
				{
					char   *pos = seg->mmap_ptr;
					char   *tail = pos + dma_segment_size;

					dma_seg_info = palloc0(sizeof(*dma_seg_info));
					dma_seg_info->seg_id = seg->segment_id;
					dma_seg_info->rev = pg_atomic_read_u32(&seg->revision);
					dma_seg_info->mclass = i;

					while (pos < tail)
					{
						dmaBufferChunk *chunk = (dmaBufferChunk *) pos;

						if (chunk->mclass == i)
						{
							if (!chunk->free_chain.prev ||
								!chunk->free_chain.next)
								dma_seg_info->n_actives++;
							else
								dma_seg_info->n_frees++;
						}
						pos += (1UL << chunk->mclass);
					}
					results = lappend(results, dma_seg_info);
				}
			}
			PG_CATCH();
			{
				SpinLockRelease(&seg->lock);
				pthreadRWLockUnlock(&dmaBufSegHead->rwlock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockRelease(&seg->lock);
		}
		pthreadRWLockUnlock(&dmaBufSegHead->rwlock);

		fncxt->user_fctx = results;
		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();
	results = fncxt->user_fctx;

	if (fncxt->call_cntr >= list_length(results))
		SRF_RETURN_DONE(fncxt);
	dma_seg_info = list_nth(results, fncxt->call_cntr);

	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dma_seg_info->seg_id);
	values[1] = Int32GetDatum(dma_seg_info->rev);
	values[2] = Int32GetDatum(dma_seg_info->mclass);
	values[3] = Int32GetDatum(dma_seg_info->n_actives);
	values[4] = Int32GetDatum(dma_seg_info->n_frees);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_dma_buffer_info);

/*
 * bgworker_reclaim_dma_buffer
 */
static void
bgworker_reclaim_dma_buffer(Datum arg)
{
	dlist_iter iter;
	int		ev;

	/* no special handling is needed on SIGTERM/SIGQUIT; just die */
	BackgroundWorkerUnblockSignals();

	/*
	 * Loop forever
	 */
	for (;;)
	{
		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();

		if (pthreadRWLockWriteTryLock(&dmaBufSegHead->rwlock))
		{
			dlist_foreach(iter, &dmaBufSegHead->active_segment_list)
			{
				dmaBufferSegment *seg
					= dlist_container(dmaBufferSegment, chain, iter.cur);
				Assert(SHMSEG_EXISTS(pg_atomic_read_u32(&seg->revision)));

				SpinLockAcquire(&seg->lock);
				if (seg->num_chunks > 0 || seg->persistent)
					SpinLockRelease(&seg->lock);
				else
				{
					dmaBufferDetachSegment(seg);
					SpinLockRelease(&seg->lock);

					dlist_delete(&seg->chain);
					dlist_push_head(&dmaBufSegHead->inactive_segment_list,
									&seg->chain);
					dmaBufSegHead->num_active_segments--;
					break;
				}
			}
			pthreadRWLockUnlock(&dmaBufSegHead->rwlock);
		}
		ev = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_TIMEOUT |
					   WL_POSTMASTER_DEATH,
					   15 * 1000);	/* wake up per 20 sec */
		/* Emergency bailout if postmaster has died. */
		if (ev & WL_POSTMASTER_DEATH)
			exit(1);
	}
}

/*
 * maxBackends - alternative of MaxBackends on system initialization
 */
static int
maxBackends(void)
{
	return MaxConnections + autovacuum_max_workers + 1 + max_worker_processes;
}

/*
 * pgstrom_startup_dma_buffer
 */
static void
pgstrom_startup_dma_buffer(void)
{
	Size		length;
	bool		found;
	int			i, j;
	char	   *mmap_ptr;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* dmaBufferEntryHead */
	length = offsetof(dmaBufferSegmentHead,
					  segments[num_logical_dma_segments]);
	dmaBufSegHead = ShmemInitStruct("dmaBufferSegmentHead", length, &found);
	Assert(!found);
	memset(dmaBufSegHead, 0, length);

	length = sizeof(dmaBufferLocalMap) * num_logical_dma_segments;
	dmaBufLocalMaps = MemoryContextAllocZero(TopMemoryContext, length);

	pthreadRWLockInit(&dmaBufSegHead->rwlock);
	dlist_init(&dmaBufSegHead->active_segment_list);
	dlist_init(&dmaBufSegHead->inactive_segment_list);

	/* preserve private address space but no physical memory */
	length = (Size)num_logical_dma_segments * dma_segment_size;
	dma_segment_vaddr_head = mmap(NULL, length,
								  PROT_NONE,
								  MAP_PRIVATE | MAP_ANONYMOUS,
								  -1, 0);
	if (dma_segment_vaddr_head == (void *)(~0UL))
		elog(ERROR, "failed on mmap(PROT_NONE, len=%zu) : %m", length);
	dma_segment_vaddr_tail = (char *)dma_segment_vaddr_head + length;

	for (i=0, mmap_ptr = dma_segment_vaddr_head;
		 i < num_logical_dma_segments;
		 i++, mmap_ptr += dma_segment_size)
	{
		dmaBufferSegment   *segment = &dmaBufSegHead->segments[i];
		dmaBufferLocalMap  *l_map = &dmaBufLocalMaps[i];

		/* dmaBufferSegment */
		memset(segment, 0, sizeof(dmaBufferSegment));
		segment->segment_id = i;
		segment->persistent = (i < num_segments_guarantee);
		segment->mmap_ptr = mmap_ptr;
		pg_atomic_init_u32(&segment->revision, 0);
		SpinLockInit(&segment->lock);
		for (j=0; j <= DMABUF_CHUNKSZ_MAX_BIT; j++)
			dlist_init(&segment->free_chunks[j]);

		dlist_push_tail(&dmaBufSegHead->inactive_segment_list,
						&segment->chain);
		/* dmaBufferLocalMap */
		l_map->segment = segment;
		pthread_mutex_init(&l_map->lock, NULL);
		l_map->revision = pg_atomic_read_u32(&segment->revision);
		l_map->is_attached = false;
	}

	/*
	 * init dmaBufferWaitList
	 */
	length = offsetof(dmaBufferWaitList, wait_proc_chain[maxBackends()]);
	dmaBufWaitList = ShmemInitStruct("dmaBufferWaitList", length, &found);
	Assert(!found);
	memset(dmaBufWaitList, 0, length);
	SpinLockInit(&dmaBufWaitList->lock);
	dlist_init(&dmaBufWaitList->wait_proc_list);
}

/*
 * pgstrom_init_dma_buffer
 */
void
pgstrom_init_dma_buffer(void)
{
	struct sigaction sigact;
	struct sigaction oldact;
	const char	   *shm_path = "/dev/shm";
	struct statfs	stat;
	size_t			total_sz;
	size_t			avail_sz;
	BackgroundWorker worker;

	/*
	 * Unit size of DMA buffer segment
	 *
	 * NOTE: It restricts the upper limit of memory allocation
	 */
	DefineCustomIntVariable("pg_strom.dma_segment_size",
							"Unit length per DMA segment",
							NULL,
							&dma_segment_size_kb,
							2 << 20,		/* 2GB */
							256 << 10,		/* 256MB */
							1UL << (DMABUF_CHUNKSZ_MAX_BIT - 10),
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	dma_segment_size = ((Size)dma_segment_size_kb << 10);

	if ((dma_segment_size & ((Size)getpagesize() - 1)) != 0)
		elog(ERROR, "pg_strom.dma_segment_size must be aligned to page size");

	/*
	 * Number of logical DMA buffer segments to be reserved with PROT_NONE
	 * pages on server startup.
	 */
	DefineCustomIntVariable("pg_strom.num_logical_dma_segments",
							"Num of logical DMA segments to be reserved",
							NULL,
							&num_logical_dma_segments,
							16384,	/* 32TB, if 2GB segment-size */
							64,		/* 128GB */
							65536,	/* 128TB */
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/*
	 * Calculation of hard/soft limit number of DMA segments and minimum
	 * guarantee of physical segments.
	 * - Backend cannot allocate DMA buffer more than soft limit (66.7%)
	 * - GPU server cannot allocate DMA buffer more than hard limit (80%)
	 * - DMA segments less than minimum guarantee shall be pinned on the
	 *   physical memory (20%).
	 */
	if (statfs("/dev/shm", &stat) != 0)
		elog(ERROR, "failed on statfs('/dev/shm'): %m");
	total_sz = (size_t)stat.f_bsize * (size_t)stat.f_blocks;
	avail_sz = (size_t)stat.f_bsize * (size_t)stat.f_bavail;

	num_segments_hardlimit = (4 * total_sz) / (5 * dma_segment_size);
	num_segments_softlimit = (2 * total_sz) / (3 * dma_segment_size);
	num_segments_guarantee = (2 * total_sz) / (5 * dma_segment_size);

	if (avail_sz < (size_t)num_segments_hardlimit * dma_segment_size)
	{
		const char *label;
		const char *threshold;
		int			elevel = WARNING;

		if (avail_sz < (size_t)num_segments_guarantee * dma_segment_size)
		{
			elevel = ERROR;
			label = "minimum guarantee";
			threshold = format_bytesz((size_t)num_segments_guarantee *
									  dma_segment_size);
		}
		else if (avail_sz < (size_t)num_segments_softlimit * dma_segment_size)
		{
			label = "soft limit";
			threshold = format_bytesz((size_t)num_segments_softlimit *
									  dma_segment_size);
		}
		else
		{
			label = "hard limit";
			threshold = format_bytesz((size_t)num_segments_hardlimit *
									  dma_segment_size);
		}
		elog(elevel, "Available size of %s volume (%s) is less than %s of DMA buffer size (%s). It will lead unexpected crash on run-time. Please cleanup unnecessary files or expand the size of volume.",
			 shm_path,
			 format_bytesz(avail_sz),
			 label,
			 threshold);
	}

	/*
	 * registration of signal handles for DMA buffers
	 */
	memset(&sigact, 0, sizeof(struct sigaction));
	sigact.sa_sigaction = dmaBufferAttachSegmentOnDemand;
	sigemptyset(&sigact.sa_mask);
	sigact.sa_flags = SA_SIGINFO;

	if (sigaction(SIGSEGV, &sigact, &oldact) != 0)
		elog(ERROR, "failed on sigaction for SIGSEGV: %m");
	sighandler_sigsegv_orig = oldact.sa_sigaction;

	if (sigaction(SIGBUS, &sigact, &oldact) != 0)
		elog(ERROR, "failed on sigaction for SIGBUS: %m");
	sighandler_sigbus_orig = oldact.sa_sigaction;

	/* request for the static shared memory */
	RequestAddinShmemSpace(offsetof(dmaBufferSegmentHead,
									segments[num_logical_dma_segments]));
	RequestAddinShmemSpace(offsetof(dmaBufferWaitList,
									wait_proc_chain[maxBackends()]));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_dma_buffer;

	/* launch a background worker for delayed DMA buffer reclaim */
	memset(&worker, 0, sizeof(BackgroundWorker));
	snprintf(worker.bgw_name, sizeof(worker.bgw_name),
			 "Delayed DMA Buffer Reclaim Worker");
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = BGW_NEVER_RESTART;
	worker.bgw_main = bgworker_reclaim_dma_buffer;
	RegisterBackgroundWorker(&worker);

	/* discard remained segment on exit of postmaster */
	before_shmem_exit(dmaBufferCleanupOnPostmasterExit, 0);
}
