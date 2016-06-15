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
#include "storage/dsm.h"
#include "pg_strom.h"

/*
 * dmaBufferEntryHead / dmaBufferEntry
 *
 * It manages the current status of DMA buffers.
 */
typedef struct dmaBufferEntry
{
	dlist_node	chain;
	cl_uint		segment_id;
	slock_t		lock;
	cl_bool		has_shmseg;
	cl_int		map_count;
} dmaBufferEntry;

typedef struct dmaBufferEntryHead
{
	char	   *vaddr_head;
	char	   *vaddr_tail;
	LWLock		mutex;
	dlist_head	active_segment_list;
	dlist_head	inactive_segment_list;
	dmaBufferEntry entries[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferEntryHead;

/*
 * dmaBufferLocalMap - status of local mapping of dmaBuffer
 */
typedef struct dmaBufferLocalEntry
{
	dmaBufferEntry *entry;		/* reference to the global entry */
	void	   *mmap_ptr;		/* address for mmap(2) */
	int			fdesc;			/* valid FD, if segment is mapped */
	bool		cuda_pinned;	/* true, if already pinned by CUDA */
} dmaBufferLocalMap;

/*
 * static variables
 */
static dmaBufferEntryHead  *dmaBufEntryHead = NULL;		/* shared memory */
static dmaBufferLocalMap   *dmaBufLocalMaps = NULL;
static size_t	dma_segment_size;
static int		dma_segment_size_kb;	/* GUC */
static int		max_dma_segment_nums;	/* GUC */
static int		min_dma_segment_nums;	/* GUC */
static void	  (*sighandler_sigsegv_orig)(int,siginfo_t *,void *) = NULL;
static void	  (*sighandler_sigbus_orig)(int,siginfo_t *,void *) = NULL;

#define SHMSEGMENT_NAME(namebuf, segment_id)				\
	snprintf((namebuf),sizeof(namebuf),"/.pg_strom.%u.%u",	\
			 PostPortNumber, (segment_id))


#define DMABUF_CHUNKSZ_MAX_BIT		36
#define DMABUF_CHUNKSZ_MIN_BIT		8
#define DMABUF_CHUNKSZ_MAX			(1UL << DMABUF_CHUNKSZ_MAX_BIT)
#define DMABUF_CHUNKSZ_MIN			(1UL << DMABUF_CHUNKSZ_MIN_BIT)
#define DMABUF_CHUNK_DATA(chunk)	((chunk)->data)
#define DMABUF_CHUNK_MAGIC_CODE		0xDEADBEAF

typedef struct dmaBufferChunk
{
	dlist_node	addr_chain;		/* link by address order */
	dlist_node	free_chain;		/* link to free chunks, or zero if active */
	SharedGpuContext *shgcon;	/* GpuContext that owns this chunk */
	size_t		required;		/* required length */
	cl_uint		mclass;			/* class of the chunk size */
	cl_uint		magic_head;		/* = DMABUF_CHUNK_MAGIC_HEAD */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferChunk;

#define DMABUF_CHUNK_MAGIC_HEAD(chunk)			((chunk)->magic_head)
#define DMABUF_CHUNK_MAGIC_TAIL(chunk)			\
	*((cl_int *)((chunk)->data + INTALIGN((chunk)->required)))



typedef struct dmaBufferSegment
{
	slock_t		lock;
	cl_uint		num_actives;	/* number of active chunks */
	dlist_head	addr_chunks;	/* chunks in address order */
	dlist_head	free_chunks[DMABUF_CHUNKSZ_MAX_BIT + 1];
} dmaBufferSegment;

/*
 * create_dma_buffer_segment - create a new DMA buffer segment
 *
 * NOTE: caller must have lock on &dmaBufferEntry->lock
 */
static void
create_dma_buffer_segment(dmaBufferEntry *entry)
{
	dmaBufferLocalMap *l_map;
	char		namebuf[80];
	int			fdesc;

	Assert(entry->segment_id < max_dma_segment_nums);
	Assert(!entry->has_shmseg);

	l_map = &dmaBufLocalMaps[entry->segment_id];
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

	mmap_ptr = mmap(l_map->mmap_ptr, dma_segment_size,
					PROT_READ | PROT_WRITE,
					MAP_SHARED | MAP_FIXED | MAP_HUGETLB,
					fdesc, 0);
	if (mmap_ptr == (void *)(~0UL))
	{
		close(fdesc);
		shm_unlink(namebuf);
		elog(ERROR, "failed on mmap: %m");
	}
	Assert(mmap_ptr == l_map->mmap_ptr);

	/* OK, shared memory segment gets successfully created */
	entry->has_shmseg = true;
	entry->map_count++;
	Assert(entry->map_count == 1);

	l_map->fdesc = fdesc;
	l_map->cuda_pinned = false;
}

/*
 * attach_dma_buffer_segment - attach an existing DMA buffer segment
 *
 * NOTE: caller must have lock on &dmaBufferEntry->lock
 */
static bool
attach_dma_buffer_segment(dmaBufferEntry *entry)
{
	dmaBufferLocalMap  *l_map = &dmaBufLocalMaps[entry->segment_id];
	char		namebuf[80];
	int			fdesc;
	char	   *mmap_ptr;

	/* sanity checks */
	if (!entry->has_shmseg)
	{
		fprintf(stderr, "Bug? DMA buffer (seg=%u) not exists, but attached\n",
				entry->segment_id);
		return false;
	}
	Assert(entry->map_count > 0 ||
		   entry->segment_id < min_dma_segment_nums);

	if (l_map->fdesc >= 0)
	{
		fprintf(stderr, "Bug? DMA buffer (seg=%u at %p) is already attached\n",
				entry->segment_id, l_map->mmap_ptr);
		return false;
	}

	/* open an existing shared memory segment */
	SHMSEGMENT_NAME(namebuf, entry->segment_id);
	fdesc = shm_open(namebuf, O_RDWR, 0600);
	if (fdesc < 0)
	{
		fprintf(stderr, "failed on shm_open('%s') : %m\n", namebuf);
		return false;
	}

	/*
	 * NOTE: no need to call ftruncate(2) here because somebody who
	 * created the segment should already expand the segment
	 */

	/* map this shared memory segment */
	mmap_ptr = mmap(l_map->mmap_ptr, dma_segment_size,
					PROT_READ | PROT_WRITE,
					MAP_SHARED | MAP_FIXED | MAP_HUGETLB,
					fdesc, 0);
	if (mmap_ptr == (void *)(~0UL))
	{
		fprintf(stderr, "failed on mmap : %m\n", mmap_ptr);
		close(fdesc);
		return false;
	}
	Assert(mmap_ptr == l_map->mmap_ptr);

	/* OK, this segment is successfully mapped */
	l_map->fdesc = fdesc;
	entry->map_count++;

	return true;
}

/*
 * detach_dma_buffer_segment - detach a DMA buffer segment already attached
 *
 * NOTE: caller must have lock on &dmaBufferEntry->lock
 */
static void
detach_dma_buffer_segment(dmaBufferEntry *entry)
{
	dmaBufferLocalMap  *l_map = &dmaBufLocalMaps[entry->segment_id];
	char		namebuf[80];

	/* sanity checks */
	Assert(l_map->fdesc >= 0);
	Assert(entry->map_count > 0);

	/* map invalid area, instead of the shared memory segment */
	if (mmap(l_map->mmap_ptr, dma_segment_size,
			 PROT_NONE,
			 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
			 -1, 0) == (void *)(~0UL))
		fprintf(stderr, "failed to mmap(PROT_NONE) (seg=%u at %p): %m\n",
				entry->segment_id, l_map->mmap_ptr);

	/* close file; just warning for errors */
	if (close(l_map->fdesc) != 0)
		fprintf(stderr, "failed to close DMA buffer handler (seg=%u): %m\n",
				entry->segment_id);

	/* unmap the shared memory segment */
	if (--entry->map_count == 0 && entry->segment_id >= min_dma_segment_nums)
	{
		SHMSEGMENT_NAME(namebuf, entry->segment_id);
		if (shm_unlink(namebuf) != 0)
			fprintf(stderr, "failed to unlink DMA buffer (seg=%u): %m\n",
					entry->segment_id);
		entry->has_shmseg = false;
	}
}

/*
 * attach_dma_buffer_on_demand
 *
 * A signal handler to be called on SIGBUS/SIGSEGV. If memory address which
 * caused a fault is in a range of virtual DMA buffer mapping, it tries to
 * map the shared buffer page.
 * Note that this handler never create a new DMA buffer segment but maps
 * an existing range, because nobody (except for buggy code) will point
 * the location which not mapped yet.
 */
static void
attach_dma_buffer_on_demand(int signum, siginfo_t *siginfo, void *unused)
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
			dmaBufferLocalMap  *l_map;
			int		seg_id;

			seg_id = ((uintptr_t)siginfo->sa_addr -
					  (uintptr_t)dmaBufEntryHead->vaddr_head)
				/ dma_segment_size;
			Assert(seg_id < max_dma_segment_nums);
			l_map = &dmaBufLocalMaps[seg_id];
			if (l_map->fdesc >= 0)
			{
				fprintf(stderr, "%s on mapped DMA buffer (seg=%u at %p)\n",
						strsignal(signum), seg_id, siginfo->sa_addr);
				goto default_handle;
			}

			SpinLockAcquire(&entry->lock);
			if (!attach_dma_buffer_segment(entry))
			{
				SpinLockRelease(&entry->lock);
				goto default_handle;
			}
			SpinLockRelease(&entry->lock);
			return;
		}
	default_handle:
		PG_SETMASK(&UnBlockSig);
		errno = save_errno;
	}

	if (signum == SIGSEGV)
		(*sighandler_sigsegv_orig)(signum, siginfo, unused);
	else if (signum == SIGBUS)
		(*sighandler_sigbus_orig)(signum, siginfo, unused);
	else
	{
		fprintf(stderr, "%s was called for %s\n",
				__FUNCTION__, strsignal(signum));
		proc_exit(2);	/* panic */
	}
	internal_error = false;		/* reset */
}


/*
 * dmaBufferSplitChunk
 *
 */
static bool
dmaBufferSplitChunk(dmaBufferSegment *segment, int mclass)
{
	// split it

}

/*
 * dmaBufferAllocChunk
 *
 * NOTE: caller must have &dmaBufferSegment->lock
 */
static void *
dmaBufferAllocChunk(dmaBufferSegment *segment, int mclass, Size required)
{
	dmaBufferChunk *chunk;

	if (dlist_is_empty(&segment->free_chunks[mclass]))
	{
		if (!dmaBufferSplitChunk(segment, mclass + 1))
			return NULL;
	}
	Assert(!dlist_is_empty(&segment->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&segment->free_chunks[mclass]);
	chunk = dlist_container(dmaBufferChunk, free_chain, dnode);
	/* init dmaBufferChunk */
	memset(&chunk->free_chain, 0, sizeof(dlist_node));
	chunk->shgcon = CurrentSharedGpuContext;
	chunk->required = required;
	chunk->mclass = mclass;
	DMABUF_CHUNK_MAGIC_HEAD(chunk) = DMABUF_CHUNK_MAGIC_CODE;
	DMABUF_CHUNK_MAGIC_TAIL(chunk) = DMABUF_CHUNK_MAGIC_CODE;

	/* TODO: register this chunk to GpuContext */


	/* update dmaBufferSegment status */
	segment->num_actives++;

	return chunk->data;
}

/*
 * __dmaBufferAlloc
 *
 */
void *
__dmaBufferAlloc(SharedGpuContext *shgcon, Size required)
{
	dmaBufferEntry	   *entry;
	dmaBufferLocalMap  *l_map;
	dmaBufferSegment   *segment;
	dlist_node		   *dnode;
	Size				chunk_size;
	int					mclass;
	bool				has_exclusive_lock = false;

	/* normalize the required size to 2^N of chunks size */
	chunk_size = MAXALIGN(offsetof(dmaBufferChunk, data) +
						  required +
						  sizeof(cl_uint));
	chunk_size = Max(chunk_size, DMABUF_CHUNKSZ_MIN);
	mclass = get_next_log2(chunk_size);
	if ((1UL << mclass) > dma_segment_size / 2)
		elog(ERROR, "DMA buffer request %zu bytes too large", required);

	/* find out an available segment */
	LWLockAcquire(&dmaBufEntryHead->mutex, LW_SHARED);
retry:
	dlist_foreach(iter, &dmaBufEntryHead->active_segment_list)
	{
		entry = dlist_container(dmaBufferEntry, chain, iter.cur);
		Assert(entry->has_shmseg);
		l_map = &dmaBufLocalMaps[entry->segment_id];
		segment = (dmaBufferSegment *) l_map->mmap_ptr;

		SpinLockAcquire(&segment->lock);
		result = dmaBufferAllocChunk(segment, mclass, required);
		if (result)
		{
			SpinLockRelease(&segment->lock);
			LWLockRelease(&dmaBufEntryHead->mutex);
			return result;
		}
		SpinLockRelease(&segment->lock);
	}
	/* Oops, no available free chunks in the active list */

	if (!has_exclusive_lock)
	{
		LWLockRelease(&dmaBufEntryHead->mutex);
		LWLockAcquire(&dmaBufEntryHead->mutex, LW_EXCLUSIVE);
		has_exclusive_lock = true;
		goto retry;
	}

	if (dlist_is_empty(&dmaBufEntryHead->inactive_segment_list))
		elog(ERROR, "out of DMA buffer segment");

	/* create a new DMA buffer segment and attach it */
	dnode = dlist_pop_head_node(&dmaBufEntryHead->inactive_segment_list);
	entry = dlist_container(dmaBufferEntry, chain, dnode);
	SpinLockAcquire(&entry->lock);
	Assert(!entry->has_shmseg);
	Assert(entry->map_count == 0);
	create_dma_buffer_segment(entry);
	SpinLockRelease(&entry->lock);

	dlist_push_head(&dmaBufEntryHead->active_segment_list,
					&entry->&chain);
	goto retry;
}

void *
dmaBufferAlloc(GpuContext_v2 *gcontext, Size required)
{
	SharedGpuContext   *shgcon = gcontext->shgcon;

	return __dmaBufferAlloc(shgcon, required);
}




void *
dmaBufferRealloc(void *pointer, Size required)
{}

void
dmaBufferFree(void *l_ptr)
{

}





















/*
 * pgstrom_startup_dma_buffer
 */
static void
pgstrom_startup_dma_buffer(void)
{
	Size		length;
	bool		found;
	int			i;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* dmaBufferEntryHead */
	length = offsetof(dmaBufferEntryHead, entries[max_dma_segment_nums]);
	dmaBufEntryHead = ShmemInitStruct("dmaBufEntryHead", length, &found);
	Assert(!found);
	memset(dmaBufEntryHead, 0, length);

	length = sizeof(dmaBufferLocalMap) * max_dma_segment_nums;
	dmaBufLocalMaps = MemoryContextAllocZero(TopMemoryContext, length);

	LWLockInitialize(&dmaBufEntryHead->mutex);
	dlist_init(&dmaBufEntryHead->active_segment_list);
	dlist_init(&dmaBufEntryHead->inactive_segment_list);

	/* preserve private address space but no physical memory */
	length = (Size)max_dma_segment_nums * dma_segment_size;
	dmaBufEntryHead->vaddr_head = mmap(NULL, length,
									   PROT_NONE,
									   MAP_PRIVATE | MAP_ANONYMOUS,
									   -1, 0);
	if (dmaBufEntryHead->vaddr_head == (void *)(~0UL))
		elog(ERROR, "failed on mmap(PROT_NONE, len=%zu) : %m", length);
	dmaBufEntryHead->vaddr_tail = dmaBufEntryHead->vaddr_head + length;

	for (i=0; i < max_dma_segment_nums; i++)
	{
		dmaBufferEntry	   *entry = &dmaBufEntryHead->entries[i];
		dmaBufferLocalMap  *l_map = &dmaBufLocalMaps[i];

		memset(entry, 0, sizeof(dmaBufferEntry));
		entry->segment_id = i;
		SpinLockInit(&entry->lock);
		dlist_push_tail(&dmaBufEntryHead->inactive_segment_list,
						&entry->chain);

		l_map->entry = entry;
		l_map->mmap_ptr = (dmaBufEntryHead->vaddr_head +
						   (Size)i * dma_segment_size);
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
							1UL << (DMABUF_CHUNKSZ_MAX_BIT - 10),	/* 64GB */
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	dma_segment_size = (dma_segment_size_kb << 10);
	port_addr_shift = get_next_log2(dma_segment_size);
	port_addr_mask = (1UL << port_addr_shift) - 1;

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
	sigact.sa_sigaction = attach_dma_buffer_on_demand;
	sigemptyset(&sigact.sa_mask);
	sigact.sa_flags = SA_SIGINFO;

	if (sigaction(SIGSEGV, &sigact, &oldact) != 0)
		elog(ERROR, "failed on sigaction for SIGSEGV: %m");
	sighandler_sigsegv_orig = oldact.sa_sigaction;

	if (sigaction(SIGBUS, &sigact, &oldact) != 0)
		elog(ERROR, "failed on sigaction for SIGBUS: %m");
	sighandler_sigbus_orig = oldact.sa_sigaction;

	/* request for the static shared memory */
	RequestAddinShmemSpace(offsetof(dmaBufferEntryHead,
									entries[max_dma_segment_nums]));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_dma_buffer;
}
