/*
 * dma_buffer.c
 *
 * Routines to manage host-pinned DMA buffer and portable shared memory
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
	LWLock		mutex;
	dlist_head	active_segment_list;
	dlist_head	inactive_segment_list;
	dmaBufferSegment segments[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferSegmentHead;

/*
 * dmaBufferLocalMap - status of local mapping of dmaBuffer
 */
typedef enum dmaBufferCudaStatus
{
	/* no cuHostMemRegister() yet */
	dmaBufCudaStatusUnregistered			= 0,
	/* cuHostMemRegister() already done */
	dmaBufCudaStatusRegistered				= 1,
	/* cuHostMemRegister() is done, but underlying pages get invalidated;
	 * It needs to unregister the segment once, prior to the next
	 * registration.
	 */
	dmaBufCudaStatusRegisteredButInvalid	= 2,
} dmaBufferCudaStatus;

typedef struct dmaBufferLocalMap
{
	dmaBufferSegment *segment;	/* (const) reference to the segment */
	uint32		revision;		/* revision number when mapped */
	bool		is_attached;	/* true, if segment is already attached */
	dmaBufferCudaStatus cuda_status;
} dmaBufferLocalMap;

/*
 * static variables
 */
static dmaBufferSegmentHead *dmaBufSegHead = NULL;	/* shared memory */
static dmaBufferLocalMap *dmaBufLocalMaps = NULL;
static void	   *dma_segment_vaddr_head = NULL;
static void	   *dma_segment_vaddr_tail = NULL;
static size_t	dma_segment_size;
static int		dma_segment_size_kb;	/* GUC */
static int		max_dma_segment_nums;	/* GUC */
static int		min_dma_segment_nums;	/* GUC */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static void	  (*sighandler_sigsegv_orig)(int,siginfo_t *,void *) = NULL;
static void	  (*sighandler_sigbus_orig)(int,siginfo_t *,void *) = NULL;





/*
 * dmaBufferCreateSegment - create a new DMA buffer segment
 *
 * NOTE: caller must have LW_EXCLUSIVE on &dmaBufSegHead->mutex
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

	Assert(seg->segment_id < max_dma_segment_nums);
	revision = pg_atomic_read_u32(&seg->revision);
	Assert(!SHMSEG_EXISTS(revision));	/* even number now */

	SHMSEGMENT_NAME(namebuf, seg->segment_id, revision);
	l_map = &dmaBufLocalMaps[seg->segment_id];

	/*
	 * NOTE: A ghost mapping may happen, if this process mapped the previous
	 * version on its private address space then some other process dropped
	 * the shared memory segment but this process had no chance to unmap.
	 * So, if we found a ghost mapping, unmap this area first.
	 */
	if (l_map->is_attached)
	{
		if (l_map->cuda_status != dmaBufCudaStatusUnregistered)
		{
			CUresult	rc;

			Assert(IsGpuServerProcess());
			rc = cuMemHostUnregister(seg->mmap_ptr);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuMemHostUnregister: %s",
					 errorText(rc));
			l_map->cuda_status = dmaBufCudaStatusUnregistered;
		}
		/* unmap the older/invalid segment first */
		if (munmap(seg->mmap_ptr, dma_segment_size) != 0)
			elog(WARNING, "failed on munmap('%s'): %m", namebuf);
		if (mmap(seg->mmap_ptr, dma_segment_size,
				 PROT_NONE,
				 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
				 -1, 0) != seg->mmap_ptr)
			elog(WARNING, "failed on mmap(PROT_NONE) for seg=%u at %p: %m",
				 seg->segment_id, seg->mmap_ptr);
		l_map->is_attached = false;
	}

	/*
	 * Open, expand and mmap the shared memory segment
	 */
	fdesc = shm_open(namebuf, O_RDWR | O_CREAT | O_TRUNC, 0600);
	if (fdesc < 0)
		elog(ERROR, "failed on shm_open('%s'): %m", namebuf);

	if (ftruncate(fdesc, dma_segment_size) != 0)
	{
		close(fdesc);
		shm_unlink(namebuf);
		elog(ERROR, "failed on ftruncate(2): %m");
	}

	if (mmap(seg->mmap_ptr, dma_segment_size,
			 PROT_READ | PROT_WRITE,
			 MAP_SHARED | MAP_FIXED,
			 fdesc, 0) != seg->mmap_ptr)
	{
		close(fdesc);
		shm_unlink(namebuf);
		elog(ERROR, "failed on mmap: %m");
	}
	close(fdesc);

	/* successfully mapped, init this segment */
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
	l_map->cuda_status = dmaBufCudaStatusUnregistered;
	l_map->revision = pg_atomic_add_fetch_u32(&seg->revision, 1);
}

/*
 * dmaBufferDetachSegment - detach a DMA buffer and delete shared memory
 * segment. If somebody still mapped this segment, further reference will
 * cause SIGBUS then signal handler will detach this segment.
 *
 * NOTE: caller must have &dmaBufSegHead->mutex with LW_EXCLUSIVE
 */
static void
dmaBufferDetachSegment(dmaBufferSegment *seg)
{
	dmaBufferLocalMap *l_map = &dmaBufLocalMaps[seg->segment_id];
	char		namebuf[80];
	int			fdesc;
	uint32		revision = pg_atomic_fetch_add_u32(&seg->revision, 1);
	CUresult	rc;

	Assert(SHMSEG_EXISTS(revision));

	/*
	 * If caller process already attach this segment, we unmap this region
	 * altogether.
	 */
	if (l_map->is_attached)
	{
		/* unregister host pinned memory, if any */
		if (l_map->cuda_status != dmaBufCudaStatusUnregistered)
		{
			Assert(IsGpuServerProcess());
			rc = cuMemHostUnregister(seg->mmap_ptr);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuMemHostUnregister: %s",
					 errorText(rc));
			l_map->cuda_status = dmaBufCudaStatusUnregistered;
		}

		/* unmap segment from private virtula address space */
		if (munmap(seg->mmap_ptr, dma_segment_size) != 0)
			elog(WARNING, "failed on munmap(seg=%u:%u at %p): %m",
				 seg->segment_id, l_map->revision/2, seg->mmap_ptr);
		/* and map invalid area instead */
		if (mmap(seg->mmap_ptr, dma_segment_size,
				 PROT_NONE,
				 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
				 -1, 0) != seg->mmap_ptr)
			elog(WARNING, "failed on mmap(PROT_NONE) for seg=%u at %p: %m",
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
		elog(WARNING, "failed on shm_open('%s', O_TRUNC): %m", namebuf);
	close(fdesc);

	if (shm_unlink(namebuf) < 0)
		elog(WARNING, "failed on shm_unlink('%s'): %m", namebuf);

	Assert(SHMSEG_EXISTS(pg_atomic_read_u32(&seg->revision)));
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
	static bool	internal_error = false;
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

			seg_id = ((uintptr_t)siginfo->si_addr -
					  (uintptr_t)dma_segment_vaddr_head) / dma_segment_size;
			Assert(seg_id < max_dma_segment_nums);
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
			if (l_map->is_attached)
			{
				if (l_map->revision == revision)
				{
					fprintf(stderr,
							"%s: got %s on %p (segid=%u at %p, rev=%u), "
							"but latest revision is already mapped\n",
							__FUNCTION__, strsignal(signum), siginfo->si_addr,
							seg->segment_id, seg->mmap_ptr, revision);
					goto normal_crash;
				}

				/*
				 * MEMO: According to the simple test program at:
				 *   https://devtalk.nvidia.com/default/topic/942684/
				 * The virtual address range registered by cuMemHostRegister()
				 * is valid even if host memory is unmapped or remapped.
				 * However, we are not certain whether it is expected and
				 * guaranteed behavior, or a potential bug; that kicks
				 * zero-copy DMA on the memory region where it can be
				 * swapped out.
				 * So, right now, we enforce to unregister this segment
				 * once, prior to the registration next time.
				 */
				if (l_map->cuda_status == dmaBufCudaStatusRegistered)
					l_map->cuda_status = dmaBufCudaStatusRegisteredButInvalid;

				/* unmap the old/invalid segment */
				if (munmap(seg->mmap_ptr, dma_segment_size) != 0)
				{
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
				fprintf(stderr, "%s: got %s on segment (id=%u at %p), "
						"but unable to mmap(2) the segment '%s': %m\n",
						__FUNCTION__, strsignal(signum),
						seg->segment_id, seg->mmap_ptr, namebuf);
				goto normal_crash;
			}
			close(fdesc);

			/* ok, this segment is successfully mapped */
			l_map->revision = revision;
			l_map->is_attached = true;

			fprintf(stderr, "%s: pid=%u got %s, then attached shared memory "
					"segment (id=%u at %p, rev=%u)\n",
					__FUNCTION__, MyProcPid, strsignal(signum),
					seg->segment_id, seg->mmap_ptr, revision);

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
		(*sighandler_sigsegv_orig)(signum, siginfo, unused);
	else if (signum == SIGBUS)
		(*sighandler_sigbus_orig)(signum, siginfo, unused);
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
 * NOTE: caller must have LW_SHARED on &dmaBufSegHead->mutex
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
__dmaBufferAlloc(SharedGpuContext *shgcon, Size required)
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
		elog(ERROR, "DMA buffer request %zu bytes too large", required);

	/* find out an available segment */
	LWLockAcquire(&dmaBufSegHead->mutex, LW_SHARED);
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
		LWLockRelease(&dmaBufSegHead->mutex);
		LWLockAcquire(&dmaBufSegHead->mutex, LW_EXCLUSIVE);
		has_exclusive_lock = true;
		goto retry;
	}
	if (dlist_is_empty(&dmaBufSegHead->inactive_segment_list))
		elog(ERROR, "Out of DMA buffer segment");

	/*
	 * Create a new DMA buffer segment
	 */
	dnode = dlist_pop_head_node(&dmaBufSegHead->inactive_segment_list);
	seg = dlist_container(dmaBufferSegment, chain, dnode);
	Assert(!SHMSEG_EXISTS(pg_atomic_read_u32(&seg->revision)));
	PG_TRY();
	{
		dmaBufferCreateSegment(seg);
	}
	PG_CATCH();
	{
		dlist_push_head(&dmaBufSegHead->inactive_segment_list, &seg->chain);
		PG_RE_THROW();
	}
	PG_END_TRY();
	dlist_push_head(&dmaBufSegHead->active_segment_list, &seg->chain);

	/*
	 * allocation of a new chunk from the new chunk to ensure num_chunks
	 * is larger than zero.
	 */
	chunk = dmaBufferAllocChunk(seg, mclass, required);
	Assert(chunk != NULL);
found:
	LWLockRelease(&dmaBufSegHead->mutex);

	/* track this chunk with GpuContext */
	SpinLockAcquire(&shgcon->lock);
	chunk->shgcon = shgcon;
	dlist_push_tail(&shgcon->dma_buffer_list, &chunk->gcxt_chain);
	SpinLockRelease(&shgcon->lock);

	return chunk->data;
}

void *
dmaBufferAlloc(GpuContext_v2 *gcontext, Size required)
{
	return __dmaBufferAlloc(gcontext->shgcon, required);
}

/*
 * pointer_validation - rough pointer validation for realloc/free
 */
static dmaBufferChunk *
pointer_validation(void *pointer, dmaBufferSegment **p_seg)
{
	dmaBufferLocalMap  *l_map;
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	int					seg_id;

	chunk = (dmaBufferChunk *)
		((char *)pointer - offsetof(dmaBufferChunk, data));
	if (!dmaBufSegHead ||
		(void *)chunk <  dma_segment_vaddr_head ||
		(void *)chunk >= dma_segment_vaddr_tail)
		elog(ERROR, "Bug? %p is out of DMA buffer", pointer);

	seg_id = ((uintptr_t)chunk -
			  (uintptr_t)dma_segment_vaddr_head) / dma_segment_size;
	Assert(seg_id < max_dma_segment_nums);
	seg = &dmaBufSegHead->segments[seg_id];
	l_map = &dmaBufLocalMaps[seg_id];
	Assert(l_map->is_attached);

	if (offsetof(dmaBufferChunk, data) +
		chunk->required + sizeof(cl_uint) > (1UL << chunk->mclass) ||
		DMABUF_CHUNK_MAGIC_HEAD(chunk) != DMABUF_CHUNK_MAGIC_CODE ||
		DMABUF_CHUNK_MAGIC_TAIL(chunk) != DMABUF_CHUNK_MAGIC_CODE)
		elog(ERROR, "Bug? DMA buffer %p is corrupted", pointer);

	if (chunk->free_chain.prev != NULL ||
		chunk->free_chain.next != NULL)
		elog(ERROR, "Bug? %p points a free DMA buffer", pointer);

	*p_seg = seg;
	return chunk;
}

/*
 * dmaBufferRealloc
 */
void *
dmaBufferRealloc(void *pointer, Size required)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	Size				chunk_size;
	int					mclass;
	void			   *result;

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
	result = __dmaBufferAlloc(chunk->shgcon, required);
	memcpy(result, chunk->data, chunk->required);
	dmaBufferFree(pointer);

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
dmaBufferFree(void *pointer)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	dmaBufferChunk	   *buddy;
	SharedGpuContext   *shgcon;
	bool				has_exclusive_mutex = false;

	/* sanity checks */
	chunk = pointer_validation(pointer, &seg);

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
	 * NOTE: If num_chunks == 1, this thread may need to detach shared memory
	 * segment. It also moves this segment from the active list to inactive
	 * list; to be operated under the dmaBufferSegmentHead->mutex.
	 * So, we preliminary acquires the mutext, prior to chunk release.
	 */
	Assert(seg->num_chunks > 0);
	if (seg->num_chunks == 1)
	{
		if (!has_exclusive_mutex)
		{
			SpinLockRelease(&seg->lock);

			LWLockAcquire(&dmaBufSegHead->mutex, LW_EXCLUSIVE);
			has_exclusive_mutex = true;
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

	/* move the segment to inactive list, and remove shm segment */
	if (seg->num_chunks > 0)
		SpinLockRelease(&seg->lock);
	else
	{
		Assert(has_exclusive_mutex);
		dmaBufferDetachSegment(seg);
		SpinLockRelease(&seg->lock);

		dlist_delete(&seg->chain);
		dlist_push_head(&dmaBufSegHead->inactive_segment_list, &seg->chain);
	}

	if (has_exclusive_mutex)
		LWLockRelease(&dmaBufSegHead->mutex);
}

/*
 * dmaBufferFree - unlink all the DMA buffer chunks tracked by the supplied
 * shared gpu context
 */
void
dmaBufferFreeAll(SharedGpuContext *shgcon)
{
	dlist_iter		iter;
	dmaBufferChunk *chunk;

	fprintf(stderr, "%s: %d\n", __FUNCTION__, __LINE__);
	dlist_foreach(iter, &shgcon->dma_buffer_list)
	{
		chunk = dlist_container(dmaBufferChunk, gcxt_chain, iter.cur);
		Assert(chunk->shgcon == shgcon);
		//dmaBufferFree(chunk->data);
	}
	fprintf(stderr, "%s: %d\n", __FUNCTION__, __LINE__);
}

/*
 * dmaBufferCleanupOnPostmasterExit - clean up all the active DMA buffers
 */
static void
dmaBufferCleanupOnPostmasterExit(int code, Datum arg)
{
	if (MyProcPid == PostmasterPid)
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

		LWLockAcquire(&dmaBufSegHead->mutex, LW_SHARED);
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
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockRelease(&seg->lock);
		}
		LWLockRelease(&dmaBufSegHead->mutex);

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
	length = offsetof(dmaBufferSegmentHead, segments[max_dma_segment_nums]);
	dmaBufSegHead = ShmemInitStruct("dmaBufferSegmentHead", length, &found);
	Assert(!found);
	memset(dmaBufSegHead, 0, length);

	length = sizeof(dmaBufferLocalMap) * max_dma_segment_nums;
	dmaBufLocalMaps = MemoryContextAllocZero(TopMemoryContext, length);

	LWLockInitialize(&dmaBufSegHead->mutex, 0);
	dlist_init(&dmaBufSegHead->active_segment_list);
	dlist_init(&dmaBufSegHead->inactive_segment_list);

	/* preserve private address space but no physical memory */
	length = (Size)max_dma_segment_nums * dma_segment_size;
	dma_segment_vaddr_head = mmap(NULL, length,
								  PROT_NONE,
								  MAP_PRIVATE | MAP_ANONYMOUS,
								  -1, 0);
	if (dma_segment_vaddr_head == (void *)(~0UL))
		elog(ERROR, "failed on mmap(PROT_NONE, len=%zu) : %m", length);
	dma_segment_vaddr_tail = (char *)dma_segment_vaddr_head + length;

	for (i=0, mmap_ptr = dma_segment_vaddr_head;
		 i < max_dma_segment_nums;
		 i++, mmap_ptr += dma_segment_size)
	{
		dmaBufferSegment   *segment = &dmaBufSegHead->segments[i];
		dmaBufferLocalMap  *l_map = &dmaBufLocalMaps[i];

		/* dmaBufferSegment */
		memset(segment, 0, sizeof(dmaBufferSegment));
		segment->segment_id = i;
		segment->mmap_ptr = mmap_ptr;
		pg_atomic_init_u32(&segment->revision, 0);
		SpinLockInit(&segment->lock);
		for (j=0; j <= DMABUF_CHUNKSZ_MAX_BIT; j++)
			dlist_init(&segment->free_chunks[j]);

		dlist_push_tail(&dmaBufSegHead->inactive_segment_list,
						&segment->chain);
		/* dmaBufferLocalMap */
		l_map->segment = segment;
		l_map->revision = pg_atomic_read_u32(&segment->revision);
		l_map->is_attached = false;
		l_map->cuda_status = dmaBufCudaStatusUnregistered;
	}
}

/*
 * pgstrom_init_dma_buffer
 */
void
pgstrom_init_dma_buffer(void)
{
	struct sigaction	sigact;
	struct sigaction	oldact;

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
	 * Number of DMA buffer segment
	 */
	DefineCustomIntVariable("pg_strom.max_dma_segment_nums",
							"Max number of DMA segments",
							NULL,
							&max_dma_segment_nums,
							1024,		/* 2TB, if default */
							32,			/* 64GB, if default */
							32768,		/* 64TB, if default */
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/*
	 * Amount of reserved DMA buffer segment
	 */
	DefineCustomIntVariable("pg_strom.min_dma_segment_nums",
							"number of reserved DMA buffer segment",
							NULL,
							&min_dma_segment_nums,
							2,
							0,
							max_dma_segment_nums,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

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
									segments[max_dma_segment_nums]));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_dma_buffer;

	/* discard remained segment on exit of postmaster */
	before_shmem_exit(dmaBufferCleanupOnPostmasterExit, 0);
}
