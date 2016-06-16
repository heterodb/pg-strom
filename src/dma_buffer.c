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
#include "pg_strom.h"

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
#define SHMSEGMENT_NAME(namebuf, segment_id)				\
	snprintf((namebuf),sizeof(namebuf),"/.pg_strom.%u.%u",	\
			 PostPortNumber, (segment_id))

typedef struct dmaBufferSegment
{
	dlist_node	chain;		/* link to active/inactive list */
	cl_uint		segment_id;	/* (const) unique identifier of the segment */
	void	   *mmap_ptr;	/* (const) address to be attached */
	slock_t		lock;		/* lock of the fields below */
	cl_bool		has_shmseg;	/* true, if shm segment is exists */
	cl_int		num_chunks;	/* number of active chunks */
	cl_int		revision;	/* revision number of physical segment */
	dlist_head	free_chunks[DMABUF_CHUNKSZ_MAX_BIT + 1];
} dmaBufferSegment;

typedef struct dmaBufferSegmentHead
{
	char	   *vaddr_head;
	char	   *vaddr_tail;
	LWLock		mutex;
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
//	int			fdesc;			/* valid FD, if segment is mapped */
	int			revision;		/* revision number when mapped */
	bool		is_attached;	/* true, if segment is already attached */
	bool		cuda_pinned;	/* true, if already pinned by CUDA */
} dmaBufferLocalMap;

/*
 * static variables
 */
static dmaBufferSegmentHead *dmaBufSegHead = NULL;	/* shared memory */
static dmaBufferLocalMap *dmaBufLocalMaps = NULL;
static size_t	dma_segment_size;
static int		dma_segment_size_kb;	/* GUC */
static int		max_dma_segment_nums;	/* GUC */
static int		min_dma_segment_nums;	/* GUC */
static void	  (*sighandler_sigsegv_orig)(int,siginfo_t *,void *) = NULL;
static void	  (*sighandler_sigbus_orig)(int,siginfo_t *,void *) = NULL;





/*
 * dmaBufferCreateSegment - create a new DMA buffer segment
 *
 * NOTE: caller must have lock on &dmaBufferEntry->lock
 */
static void
dmaBufferCreateSegment(dmaBufferSegment *seg)
{
	dmaBufferLocalMap  *l_map;
	dmaBufferChunk	   *chunk;
	char		namebuf[80];
	int			fdesc;
	int			i, mclass;
	char	   *mmap_ptr;
	char	   *head_ptr;
	char	   *tail_ptr;

	Assert(seg->segment_id < max_dma_segment_nums);
	Assert(!seg->has_shmseg);

	l_map = &dmaBufLocalMaps[seg->segment_id];
	Assert(l_map->fdesc < 0);

	SHMSEGMENT_NAME(namebuf, entry->segment_id);
	fdesc = shm_open(namebuf, O_RDWR | O_CREAT | O_EXCL, 0600);
	if (fdesc < 0)
		elog(ERROR, "failed on shm_open('%s'): %m", namebuf);

	if (ftruncate(fdesc, dma_segment_size) != 0)
	{
		close(fdesc);
		shm_unlink(namebuf);
		elog(ERROR, "failed on ftruncate(): %m");
	}

	mmap_ptr = mmap(seg->mmap_ptr, dma_segment_size,
					PROT_READ | PROT_WRITE,
					MAP_SHARED | MAP_FIXED,
					fdesc, 0);
	if (mmap_ptr == (void *)(~0UL))
	{
		close(fdesc);
		shm_unlink(namebuf);
		elog(ERROR, "failed on mmap: %m");
	}
	Assert(mmap_ptr == seg->mmap_ptr);

	/* initialize the segment */
	head_ptr = mmap_ptr;
	tail_ptr = mmap_ptr + dma_segment_size;
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

	/* ok, shared memory segment gets successfully created */
	seg->has_shmseg = true;
	seg->num_chunks = 0;

	/* Also, update local mapping */
	l_map->fdesc = fdesc;
	l_map->revision = seg->revision;
	l_map->cuda_pinned = false;
}

/*
 * dmaBufferDetachSegment - detach a DMA buffer and delete shared memory
 * segment. If somebody still mapped this segment, further reference will
 * cause SIGBUS then signal handler will detach this segment.
 *
 * NOTE: caller must have lock on &dmaBufferSegment->lock
 */
static void
dmaBufferDetachSegment(dmaBufferSegment *seg)
{
	dmaBufferLocalMap *l_map = &dmaBufLocalMaps[seg->segment_id];
	char		namebuf[80];
	int			fdesc;
	CUresult	rc;

	/*
	 * If caller process already attach this segment, we unmap this region
	 * altogether.
	 */
	SHMSEGMENT_NAME(namebuf, seg->segment_id);
	if (l_map->fdesc >= 0)
	{
		/* unregister host pinned memory, if any */
		if (l_map->cuda_pinned)
		{
			Assert(IsGpuServerProcess());
			rc = cuMemHostUnregister(seg->mmap_ptr);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuMemHostUnregister: %s",
					 errorText(rc));
		}

		/* truncate then unlink the shared memory segment */
		if (ftruncate(l_map->fdesc, 0) != 0)
			elog(WARNING, "failed on ftruncate('%s'): %m", namebuf);
		if (shm_unlink(namebuf) != 0)
			elog(WARNING, "failed on shm_unlink('%s'): %m", namebuf);

		/* map invalid area, instead of the valid shared memory segment */
		if (mmap(seg->mmap_ptr, dma_segment_size,
				 PROT_NODE,
				 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
				 -1, 0) == (void *)(~0UL))
			elog(WARNING, "failed on mmap(PROT_NONE) (seg=%u at %p): %m",
				 seg->segment_id, seg->mmap_ptr);

		/* close file handler */
		if (close(l_map->fdesc) != 0)
			elog(WARNING, "failed on close('%s'): %m", namebuf);

		l_map->fdesc = -1;
		l_map->cuda_pinned = false;
	}
	else
	{
		fdesc = shm_open(namebuf, O_RDWR | O_TRUNC, 0600);
		if (fdesc < 0)
			elog(WARNING, "failed on shm_open('%s', O_TRUNC): %m", namebuf);
		else
			close(fdesc);

		if (shm_unlink(namebuf) < 0)
			elog(WARNING, "failed on shm_unlink('%s'): %m", namebuf);
	}

	/*
	 * Note that dmaBufferDetachSegment() never unmap this segment from
	 * the virtual address space of other processes which map this segment.
	 * On the other hands, this shared memory segment is already truncated
	 * to zero length, thus, further access to this region will cause SIGBUS
	 * error. The relevant signal handler will correct its virtual address
	 * space properly.
	 * What we have to pay attention is, some other process may construct
	 * another shared memory segment, but same segment id. Although these
	 * segment has same ID, but physically different shared memory segment.
	 * So, shared memory segment shall be identified with revision number
	 * in addition to segment id.
	 */
	seg->revision++;
	seg->has_shmseg = false;
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

	if (internal_error)
	{
		internal_error = true;	/* prevent infinite loop */
		save_errno = errno;
		PG_SETMASK(&BlockSig);

		if (dmaBufEntryHead &&
			dmaBufEntryHead->vaddr_head <= siginfo->sa_addr &&
			dmaBufEntryHead->vaddr_tail >  siginfo->sa_addr)
		{
			dmaBufferSegment   *seg;
			dmaBufferLocalMap  *l_map;
			int		seg_id;
			char	namebuf[80];
			int		fdesc;
			char   *mmap_ptr;

			seg_id = ((uintptr_t)siginfo->sa_addr -
					  (uintptr_t)dmaBufEntryHead->vaddr_head)
				/ dma_segment_size;
			Assert(seg_id < max_dma_segment_nums);
			seg = &dmaBufSegHead->segments[seg_id];
			l_map = &dmaBufLocalMaps[seg_id];

			SpinLockAcquire(&seg->lock);
			if (l_map->fdesc >= 0)
			{
				if (l_map->revision == seg->revision)
				{
					fprintf(stderr, "%s got %s on the segment (id=%u at %p) "
							"but the latest revision is already mapped\n",
							__FUNCTION__, strsignal(signum),
							seg->segment_id, seg->mmap_ptr);
					SpinLockRelease(&seg->lock);
					goto default_handle;
				}
				if (close(l_map->fdesc) != 0)
				{
					fprintf(stderr, "%s: failed on close: %m\n",
							__FUNCTION__, l_map->fdesc);
					SpinLockRelease(&seg->lock);
					goto default_handle;
				}
				l_map->fdesc = -1;

				/* unmap old */
				if (mmap(seg->mmap_ptr, dma_segment_size,
						 PROT_NONE,
						 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
						 -1, 0) == (void *)(~0UL))
				{
					fprintf(stderr, "%s: failed on mmap(PROT_NONE) at %p: %m",
							__FUNCTION__, seg->mmap_ptr);
					SpinLockRelease(&seg->lock);
					goto default_handle;
				}
			}

			if (!seg->has_shmseg)
			{
				fprintf(strerr, "%s: got %s on segment (id=%u at %p), "
						"but no physical shared memory segment\n",
						__FUNCTION__, strsignal(signum),
						seg->segment_id, seg->mmap_ptr);
				SpinLockRelease(&seg->lock);
				goto default_handle;
			}
			/* open an "existing" shared memory segment */
			SHMSEGMENT_NAME(namebuf, seg->segment_id);
			fdesc = shm_open(namebuf, O_RDWR, 0600);
			if (fdesc < 0)
			{
				fprintf(stderr, "%s: got %s on segment (id=%u at %p), "
						"but unable to open shared memory segment: %m\n",
						__FUNCTION__, strsignal(signum),
						seg->segment_id, seg->mmap_ptr);
				SpinLockRelease(&seg->lock);
				goto default_handle;
			}

			/*
			 * NOTE: no need to call ftruncate(2) here because somebody
			 * who created the segment should already expand the segment
			 */

			/* map this shared memory segment */
			if (mmap(seg->mmap_ptr, dma_segment_size,
					 PROT_READ | PROT_WRITE,
					 MAP_SHARED | MAP_FIXED,
					 fdesc, 0) == (void *)(~0UL))
			{
				fprintf(stderr, "%s: got %s on segment (id=%u at %p), "
						"but unable to mmap(2) this segment: %m\n",
						__FUNCTION__, strsignal(signum),
						seg->segment_id, seg->mmap_ptr);
				SpinLockRelease(&seg->lock);
				goto default_handle;
			}

			/* OK, this segment is successfully mapped */
			l_map->fdesc = fdesc;
			l_map->revision = seg->revision;
			fprintf(stderr, "%s: pid=%u got %s, then attach shared memory "
					"segment (id=%u at %p)\n",
					__FUNCTION__, MyProcPid, strsignal(signum),
					seg->segment_id, seg->mmap_ptr);
			SpinLockRelease(&seg->lock);

			PG_SETMASK(&UnBlockSig);
			errno = save_errno;
			internal_error = false;
			return;		/* problem solved */
		}
	default_handle:
		PG_SETMASK(&UnBlockSig);
		errno = save_errno;
	}

	if (signum == SIGSEGV)
		(*sighandler_sigsegv_orig)(signum, siginfo, unused);
	else if (sugnum == SIGBUS)
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
	Assert(chunk->mclass == mclass);
	Assert(chunk->magic_head == DMABUF_CHUNK_MAGIC_CODE);

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
 * NOTE: caller must have &dmaBufferSegment->lock
 */
static void *
dmaBufferAllocChunk(dmaBufferSegment *seg, int mclass, Size required)
{
	dmaBufferChunk *chunk;
	dlist_node	   *dnode;

	Assert(mclass <= DMABUF_CHUNKSZ_MAX_BIT);
	if (dlist_is_empty(&seg->free_chunks[mclass]))
	{
		if (!dmaBufferSplitChunk(seg, mclass + 1))
			return NULL;
	}
	Assert(!dlist_is_empty(&seg->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&seg->free_chunks[mclass]);
	chunk = dlist_container(dmaBufferChunk, free_chain, dnode);
	Assert(chunk->mclass == mclass);
	Assert(DMABUF_CHUNK_MAGIC_HEAD(chunk) == DMABUF_CHUNK_MAGIC_CODE);

	/* init dmaBufferChunk */
	memset(&chunk->free_chain, 0, sizeof(dlist_node));
	chunk->shgcon = shgcon;
	chunk->required = required;
	chunk->mclass = mclass;
	DMABUF_CHUNK_MAGIC_HEAD(chunk) = DMABUF_CHUNK_MAGIC_CODE;
	DMABUF_CHUNK_MAGIC_TAIL(chunk) = DMABUF_CHUNK_MAGIC_CODE;

	/* update dmaBufferSegment status */
	seg->num_chunks++;

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
	Size				chunk_size;
	int					mclass;
	void			   *result;
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
	LWLockAcquire(&dmaBufEntryHead->mutex, LW_SHARED);
retry:
	dlist_foreach(iter, &dmaBufSegHead->active_segment_list)
	{
		seg = dlist_container(dmaBufferSegment, chain, iter.cur);

		SpinLockAcquire(&seg->lock);
		Assert(seg->has_shmseg);
		chunk = dmaBufferAllocChunk(seg, mclass, required);
		if (chunk)
		{
			SpinLockRelease(&segment->lock);
			LWLockRelease(&dmaBufEntryHead->mutex);
			goto found;
		}
		SpinLockRelease(&seg->lock);
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
	SpinLockAcquire(&seg->lock);
	Assert(!seg->has_shmseg);
	PG_TRY();
	{
		dmaBufferCreateSegment(seg);
	}
	PG_CATCH();
	{
		SpinLockRelease(&seg->lock);
		dlist_push_head(&dmaBufSegHead->inactive_segment_list, dnode);
		PG_RE_THROW();
	}
	PG_END_TRY();
	dlist_push_head(&dmaBufSegHead->active_segment_list, &seg->chain);

	/* allocation of a new chunk from the new chunk to ensure num_chunks is
	 * larger than zero. */
	chunk = dmaBufferAllocChunk(seg, mclass, required);
	Assert(chunk != NULL);

	SpinLockRelease(&segment->lock);
	LWLockRelease(&dmaBufEntryHead->mutex);
found:
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
pointer_validation(void *pointer, dmaBufferSegment *p_seg)
{
	dmaBufferLocalMap  *l_map;
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	int					seg_id;

	chunk = (dmaBufferChunk *)
		((char *)pointer - offsetof(dmaBufferChunk, data));
	if (!dmaBufEntryHead ||
		(void *)chunk <  dmaBufEntryHead->vaddr_head ||
		(void *)chunk >= dmaBufEntryHead->vaddr_tail)
		elog(ERROR, "Bug? %p is out of DMA buffer", pointer);

	seg_id = ((uintptr_t)chunk -
			  (uintptr_t)dmaBufEntryHead->vaddr_head) / dma_segment_size;
	Assert(seg_id < max_dma_segment_nums);
	seg = &dmaBufSegHead->segments[seg_id];
	l_map = &dmaBufLocalMaps[seg_id];
	Assert(l_map->fdesc >= 0);

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
			memset(temp, 0, sizeof(dmaBufferChunk, data));
			temp->mclass = shift;
			DMABUF_CHUNK_MAGIC_HEAD(temp) = DMABUF_CHUNK_MAGIC_HEAD;
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

void
dmaBufferFree(void *pointer)
{
	dmaBufferSegment   *seg;
	dmaBufferChunk	   *chunk;
	dmaBufferChunk	   *buddy;
	SharedGpuContext   *shgcon;
	dlist_node		   *dnode;
	bool				has_exclusive_mutex = false;

	/* sanity checks */
	chunk = pointer_validation(pointer, &segment);

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
	Assert(seg->num_chunks > 0);

	/*
	 * NOTE: If num_chunks == 1, this thread may need to detach shared memory
	 * segment. It also moves this segment from the active list to inactive
	 * list; to be operated under the dmaBufferSegmentHead->mutex.
	 * So, we preliminary acquires the mutext, prior to chunk release.
	 */
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
		SpinLockRelease(&segment->lock);

		dlist_delete(&seg->chain);
		dlist_push_head(&dmaBufSegHead->inactive_segment_list, &seg->chain);
		LWLockRelease(&dmaBufSegHead->mutex);
	}
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
	length = offsetof(dmaBufferSegmentHead, segments[max_dma_segment_nums]);
	dmaBufSegHead = ShmemInitStruct("dmaBufferSegmentHead", length, &found);
	Assert(!found);
	memset(dmaBufSegHead, 0, length);

	length = sizeof(dmaBufferLocalMap) * max_dma_segment_nums;
	dmaBufLocalMaps = MemoryContextAllocZero(TopMemoryContext, length);

	LWLockInitialize(&dmaBufSegHead->mutex);
	dlist_init(&dmaBufSegHead->active_segment_list);
	dlist_init(&dmaBufSegHead->inactive_segment_list);

	/* preserve private address space but no physical memory */
	length = (Size)max_dma_segment_nums * dma_segment_size;
	dmaBufSegHead->vaddr_head = mmap(NULL, length,
									 PROT_NONE,
									 MAP_PRIVATE | MAP_ANONYMOUS,
									 -1, 0);
	if (dmaBufSegHead->vaddr_head == (void *)(~0UL))
		elog(ERROR, "failed on mmap(PROT_NONE, len=%zu) : %m", length);
	dmaBufSegHead->vaddr_tail = dmaBufSegHead->vaddr_head + length;

	for (i=0, mmap_ptr = dmaBufSegHead->vaddr_head;
		 i < max_dma_segment_nums;
		 i++, mmap_ptr += dma_segment_size)
	{
		dmaBufferSegment   *segment = &dmaBufSegHead->segments[i];
		dmaBufferLocalMap  *l_map = &dmaBufLocalMaps[i];

		/* dmaBufferSegment */
		memset(segment, 0, sizeof(dmaBufferSegment));
		segment->segment_id = i;
		segment->mmap_ptr = mmap_ptr;
		SpinLockInit(&segment->lock);
		for (j=0; j <= DMABUF_CHUNKSZ_MAX_BIT; j++)
			dlist_init(&segment->free_chunks[j]);

		dlist_push_tail(&dmaBufSegHead->inactive_segment_list,
						&segment->chain);
		/* dmaBufferLocalMap */
		l_map->entry = entry;
		l_map->fdesc = -1;
		l_map->cuda_pinned = false;
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
	char				namebuf[80];

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
	dma_segment_size = (dma_segment_size_kb << 10);

	if ((dma_segment_size & ((Size)getpagesize - 1)) != 0)
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
	 * Clean up shmem segment, if any
	 */
	for (i=0; i < max_dma_segment_nums; i++)
	{
		SHMSEGMENT_NAME(namebuf, i);
		if (shm_unlink(namebuf) && errno != ENOENT)
			elog(LOG, "shm_unlink('%s') : %m", namebuf);
	}

	/*
	 * registration of signal handles for DMA buffers
	 */
	memset(&sigact, 0, sizeof(struct aigaction));
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
}
