/*
 * shmcxt.c
 *
 * Implementation of MemoryContext on dynamic shared-memory segments.
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#include "pg_strom.h"
#include "nodes/memnodes.h"

#define SHMBUF_CHUNK_MAGIC_CODE		0xdeadbeaf
#define SHMBUF_CHUNKSZ_MIN_BIT		7		/* 128B */
#define SHMBUF_CHUNKSZ_MAX_BIT		32		/* 4GB */
#define SHMBUF_CHUNKSZ_MIN			(1U << SHMBUF_CHUNKSZ_MIN_BIT)
#define SHMBUF_CHUNKSZ_MAX			(1U << SHMBUF_CHUNKSZ_MAX_BIT)

typedef struct
{
	dlist_node	chain;		/* link to free chunks, or zero if active */
	size_t		required;	/* required length */
	uint32		mclass;		/* class of the chunk size */
	uint32		magic_head;	/* = DMABUF_CHUNK_MAGIC */
	MemoryContext memcxt;	/* = shmBufferContext that owns this chunk */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} shmBufferChunk;

#define SHMBUF_CHUNK_MAGIC_HEAD(chunk)		((chunk)->magic_head)
#define SHMBUF_CHUNK_MAGIC_TAIL(chunk)		\
	*((uint32 *)((chunk)->data + (chunk)->required))
#define SHMBUF_CHUNK_CHECK_MAGIC(chunk)								\
	(SHMBUF_CHUNK_MAGIC_HEAD(chunk) == SHMBUF_CHUNK_MAGIC_CODE &&	\
	 SHMBUF_CHUNK_MAGIC_TAIL(chunk) == SHMBUF_CHUNK_MAGIC_CODE)
#define SHMBUF_POINTER_GET_CHUNK(pointer)							\
	((shmBufferChunk *)((char *)pointer - offsetof(shmBufferChunk, data)))

typedef struct
{
	dlist_node		chain;		/* link to free_segment_list if inactive,
								 * or active_segment_list if in-use */
	pg_atomic_uint32 revision;	/* revision of the shared memory segment and
								 * its status. Odd number, if segment exists.
								 * Elsewhere, no segment exists. This field
								 * is referenced in the signal handler, so
								 * we don't use lock to update the field.
								 */
	uint32			num_actives;/* number of active chunks */
	dlist_head		free_chunks[SHMBUF_CHUNKSZ_MAX_BIT -
								SHMBUF_CHUNKSZ_MIN_BIT + 1];
} shmBufferSegment;

#define SHMBUF_SEGMENT_EXISTS(revision)		(((revision) & 1) != 0)

typedef struct
{
	slock_t			lock;		/* protection of the list below */
	dlist_head		shmem_context_list;
	dlist_head		free_segment_list;
	shmBufferSegment segments[FLEXIBLE_ARRAY_MEMBER];
} shmBufferSegmentHead;

typedef struct
{
	shmBufferSegment *segment;	/* (const) reference to the segment */
	slock_t			mutex;		/* mutex to manage local memory map */
	uint32			revision;	/* revision number when local mapping */
	bool			is_attached;/* true, if segment is already attached */
} shmBufferLocalMap;

typedef struct
{
	MemoryContextData header;	/* Standard memory-context fields */
	dlist_node		chain;		/* link to shmem_context_list */
	slock_t			lock;		/* Lock for shared memory allocation */
	dlist_head		active_segment_list;
	char			namebuf[FLEXIBLE_ARRAY_MEMBER];
} shmBufferContext;

/* -------- static variables -------- */
static shmem_startup_hook_type shmem_startup_next = NULL;
static struct sigaction sigaction_orig_sigsegv;
static struct sigaction sigaction_orig_sigbus;
static size_t	shmbuf_segment_size;
static int		shmbuf_segment_size_kb;		/* GUC */
static int		shmbuf_num_logical_segment;	/* GUC */
static shmBufferSegmentHead *shmBufSegHead = NULL;	/* shared memory */
static shmBufferLocalMap *shmBufLocalMaps = NULL;
static char	   *shmbuf_segment_vaddr_head = NULL;
static char	   *shmbuf_segment_vaddr_tail = NULL;
static MemoryContextMethods sharedMemoryContextMethods;
MemoryContext	TopSharedMemoryContext = NULL;

/* -------- SQL functions -------- */
Datum pgstrom_shmbuf_info(PG_FUNCTION_ARGS);

/* -------- utility inline functions -------- */
static inline int
shmBufferSegmentId(shmBufferSegment *seg)
{
	Assert(seg >= shmBufSegHead->segments &&
		   seg <  shmBufSegHead->segments + shmbuf_num_logical_segment);
	return (seg - shmBufSegHead->segments);
}

static inline void *
shmBufferSegmentMmapPtr(shmBufferSegment *seg)
{
	uint32		segment_id = shmBufferSegmentId(seg);

	return shmbuf_segment_vaddr_head + segment_id * shmbuf_segment_size;
}

static inline shmBufferSegment *
shmBufferSegmentFromChunk(shmBufferChunk *chunk)
{
	uint32		segment_id;
	
	Assert((char *)chunk >= shmbuf_segment_vaddr_head &&
		   (char *)chunk <  shmbuf_segment_vaddr_tail);
	segment_id = ((uintptr_t)chunk -
				  (uintptr_t)shmbuf_segment_vaddr_head) / shmbuf_segment_size;
	Assert(segment_id < shmbuf_num_logical_segment);

	return &shmBufSegHead->segments[segment_id];
}


#define SHMBUF_SEGMENT_FILENAME(namebuf,segment_id,revision)	\
	snprintf((namebuf),NAMEDATALEN,"/.pg_shmbuf_%u.%u:%u",	\
			 PostPortNumber,(segment_id),(revision)>>1)

/*
 * shmBufferAttachSegmentOnDemand
 *
 * A signal handler to be called on SIGBUS/SIGSEGV. If memory address which
 * caused a fault is in a range of virtual shared memory mapping by mmap with
 * PROT_NONE, it tries to map the shared memory file on behalf of the segment.
 * Note that this handler never create a new shared memory segment, but only
 * maps existing one, because nobody (except for buggy code) should reference
 * the location which is not mapped yet.
 */
static void
shmBufferAttachSegmentOnDemand(int signum, siginfo_t *siginfo, void *unused)
{
	char   *fault_addr = siginfo->si_addr;
	
	assert(signum == SIGSEGV || signum == SIGBUS);
	if (shmBufSegHead &&
		fault_addr >= shmbuf_segment_vaddr_head &&
		fault_addr <  shmbuf_segment_vaddr_tail)
	{
		int			errno_saved = errno;
		shmBufferSegment *seg;
		shmBufferLocalMap *lmap;
		char	   *mmap_ptr;
		uint32		segment_id;
		uint32		revision;
		char		namebuf[NAMEDATALEN];
		int			fdesc;

		segment_id = (fault_addr -
					  shmbuf_segment_vaddr_head) / shmbuf_segment_size;
		Assert(segment_id < shmbuf_num_logical_segment);
		seg = &shmBufSegHead->segments[segment_id];
		lmap = &shmBufLocalMaps[segment_id];
		mmap_ptr = shmBufferSegmentMmapPtr(seg);

		revision = pg_atomic_read_u32(&seg->revision);
		if (!SHMBUF_SEGMENT_EXISTS(revision))
		{
			fprintf(stderr, "pid=%u: %s on %p (seg_id=%u,rev=%u) - "
					"not a valid shared memory segment.\n",
					MyProcPid, strsignal(signum), fault_addr,
					segment_id, revision);
			goto normal_crash;
		}

		/*
		 * If segment is already mapped, we need to check its revision
		 * number is the latest, or not. It often maps a shared memory
		 * file that is already removed by other concurrent processes.
		 * In this case, references to the shared memory area causes
		 * SIGBUS exception, then signal handler can fix up the state
		 * of mapping.
		 */
		SpinLockAcquire(&lmap->mutex);
		if (lmap->is_attached)
		{
			if (lmap->revision == revision)
			{
				SpinLockRelease(&lmap->mutex);
				fprintf(stderr, "pid=%u: %s on %p (seg_id=%u,rev=%u) - "
						"it should be a valid mapping but caught a signal.\n",
						MyProcPid, strsignal(signum), fault_addr,
						segment_id, revision);
				goto normal_crash;
			}
			/* unmap the older one first */
			if (munmap(mmap_ptr, shmbuf_segment_size) != 0)
			{
				SpinLockRelease(&lmap->mutex);
				fprintf(stderr, "pid=%u: %s on %p (seg_id=%u,rev=%u) - "
						"failed on munmap(%p, %zu): %m\n",
						MyProcPid, strsignal(signum), fault_addr,
						segment_id, revision,
						mmap_ptr, shmbuf_segment_size);
				goto normal_crash;
			}
			lmap->is_attached = false;
		}

		/*
		 * Open an "existing" shared memory segment
		 */
		SHMBUF_SEGMENT_FILENAME(namebuf, segment_id, revision);
		fdesc = shm_open(namebuf, O_RDWR, 0600);
		if (fdesc < 0)
		{
			SpinLockRelease(&lmap->mutex);
			fprintf(stderr, "pid=%u: %s on %p (seg_id=%u,rev=%u) - "
					"failed on shm_open('%s'): %m\n",
					MyProcPid, strsignal(signum), fault_addr,
					segment_id, revision,
					namebuf);
			goto normal_crash;
		}

		if (mmap(mmap_ptr, shmbuf_segment_size,
				 PROT_READ | PROT_WRITE,
				 MAP_SHARED | MAP_FIXED,
				 fdesc, 0) != mmap_ptr)
		{
			close(fdesc);
			shm_unlink(namebuf);
			SpinLockRelease(&lmap->mutex);
			fprintf(stderr, "pid=%u: %s on %p (seg_id=%u,rev=%u) - "
					"failed on mmap('%s'): %m",
					MyProcPid, strsignal(signum), fault_addr,
					segment_id, revision,
					namebuf);
			goto normal_crash;
		}
		close(fdesc);
		SpinLockRelease(&lmap->mutex);

		/* problem solved */
		fprintf(stderr, "pid=%u: %s on %p (seg_id=%u,rev=%u) - "
				"[%s] has been locally mapped on demand.\n",
				MyProcPid, strsignal(signum), fault_addr,
				segment_id, revision,
				namebuf);
		errno = errno_saved;
		return;

	normal_crash:
		errno = errno_saved;
	}

	if (signum == SIGSEGV)
	{
		if (sigaction_orig_sigsegv.sa_handler)
			sigaction_orig_sigsegv.sa_handler(signum);
		else if (sigaction_orig_sigsegv.sa_sigaction)
			sigaction_orig_sigsegv.sa_sigaction(signum, siginfo, unused);
	}
	else if (signum == SIGBUS)
	{
		if (sigaction_orig_sigbus.sa_handler)
			sigaction_orig_sigbus.sa_handler(signum);
		else if (sigaction_orig_sigbus.sa_sigaction)
			sigaction_orig_sigbus.sa_sigaction(signum, siginfo, unused);
	}
	abort();
}

/*
 * shmBufferCreateSegment - create a new shared memory segment
 */
static shmBufferSegment *
shmBufferCreateSegment(void)
{
	shmBufferSegment *seg;
	shmBufferLocalMap *lmap;
	dlist_node *dnode;
	uint32		segment_id;
	uint32		revision;
	int			mclass;
	char	   *mmap_ptr;
	char	   *head_ptr;
	char	   *tail_ptr;
	char		namebuf[NAMEDATALEN];
	int			i, fdesc;
	
	/* pick up a free shared memory segment */
	SpinLockAcquire(&shmBufSegHead->lock);
	if (dlist_is_empty(&shmBufSegHead->free_segment_list))
	{
		SpinLockRelease(&shmBufSegHead->lock);
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of shared memory"),
				 errhint("enlarge shmbuf.num_logical_segments")));
	}
	dnode = dlist_pop_head_node(&shmBufSegHead->free_segment_list);
	seg = dlist_container(shmBufferSegment, chain, dnode);
	memset(&seg->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&shmBufSegHead->lock);

	revision = pg_atomic_read_u32(&seg->revision);
	Assert(!SHMBUF_SEGMENT_EXISTS(revision));
	segment_id = shmBufferSegmentId(seg);
	lmap = &shmBufLocalMaps[segment_id];
	mmap_ptr = shmBufferSegmentMmapPtr(seg);
	SHMBUF_SEGMENT_FILENAME(namebuf, segment_id, revision);
	
	/*
	 * NOTE: A ghost mapping can happen. If this process already mapped
	 * the previous revision on its private address space, then other
	 * concurrent process dropped this shared memory segment but this
	 * process might not have any chance to unmap.
	 * So, unmap ghost mapping first.
	 */
	if (lmap->is_attached)
	{
		if (munmap(mmap_ptr, shmbuf_segment_size) != 0)
			elog(FATAL, "failed on munmap('%s'): %m", namebuf);
		if (mmap(mmap_ptr, shmbuf_segment_size,
				 PROT_NONE,
				 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
				 -1, 0) != mmap_ptr)
			elog(FATAL, "failed on mmap(PROT_NONE) for seg_id=%u at %p: %m",
				 segment_id, mmap_ptr);
		lmap->is_attached = false;
	}

	/*
	 * Create a new shared memory segment
	 */
	fdesc = shm_open(namebuf, O_RDWR | O_CREAT | O_TRUNC, 0600);
	if (fdesc < 0)
		elog(ERROR, "failed on shm_open('%s'): %m", namebuf);

	while (fallocate(fdesc, 0, 0, shmbuf_segment_size) != 0)
	{
		if (errno == EINTR)
			continue;
		close(fdesc);
		shm_unlink(namebuf);
		elog(ERROR, "failed on fallocate('%s'): %m", namebuf);
	}

	if (mmap(mmap_ptr, shmbuf_segment_size,
			 PROT_READ | PROT_WRITE,
			 MAP_SHARED | MAP_FIXED,
			 fdesc, 0) != mmap_ptr)
	{
		close(fdesc);
		shm_unlink(namebuf);
		elog(ERROR, "failed on mmap('%s'): %m", namebuf);
	}
	close(fdesc);

	/*
	 * Ok, successfully mapped.
	 */
	memset(&seg->chain, 0, sizeof(dlist_node));
	for (i=SHMBUF_CHUNKSZ_MIN_BIT; i <= SHMBUF_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&seg->free_chunks[i - SHMBUF_CHUNKSZ_MIN_BIT]);

	mclass = SHMBUF_CHUNKSZ_MAX_BIT;
	head_ptr = mmap_ptr;
	tail_ptr = mmap_ptr + shmbuf_segment_size;
	while (mclass >= SHMBUF_CHUNKSZ_MIN_BIT)
	{
		shmBufferChunk *chunk = (shmBufferChunk *)head_ptr;

		if (head_ptr + (1UL << mclass) > tail_ptr)
		{
			mclass--;
			continue;
		}
		memset(chunk, 0, offsetof(shmBufferChunk, data));
		chunk->mclass = mclass;
		SHMBUF_CHUNK_MAGIC_HEAD(chunk) = SHMBUF_CHUNK_MAGIC_CODE;
		dlist_push_tail(&seg->free_chunks[mclass - SHMBUF_CHUNKSZ_MIN_BIT],
						&chunk->chain);
		head_ptr += (1UL << mclass);
	}
	seg->num_actives = 0;

	/* also, update the local mapping */
	lmap->is_attached = true;
	lmap->revision = pg_atomic_add_fetch_u32(&seg->revision, 1);
	Assert(SHMBUF_SEGMENT_EXISTS(lmap->revision));

	return seg;
}

/*
 * shmBufferDropSegment - detach an empty shared memory segment. Somebody
 * may still map this segment, then further reference shall cause SIGBUS
 * that can be caught by the signal handler.
 *
 * NOTE: caller must have lock on shmBufSegHead->lock
 */
static void
shmBufferDropSegment(shmBufferSegment *seg)
{
	uint32		segment_id = shmBufferSegmentId(seg);
	char	   *mmap_ptr = shmBufferSegmentMmapPtr(seg);
	shmBufferLocalMap *lmap = &shmBufLocalMaps[segment_id];
	uint32		revision = pg_atomic_fetch_add_u32(&seg->revision, 1);
	char		namebuf[NAMEDATALEN];
	int			fdesc;

	if (lmap->is_attached)
	{
		/* unmap the segment from the private virtual address space */
		if (munmap(mmap_ptr, shmbuf_segment_size) != 0)
			elog(FATAL, "failed on munmap(seg_id=%u:%u at %p): %m",
				 segment_id, lmap->revision/2, mmap_ptr);
		/* and map invalid area instead */
		if (mmap(mmap_ptr, shmbuf_segment_size,
				 PROT_NONE,
				 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
				 -1, 0) != mmap_ptr)
			elog(FATAL, "failed on mmap(PROT_NONE) for seg_id=%u:%u at %p: %m",
				 segment_id, lmap->revision/2, mmap_ptr);

		lmap->is_attached = false;
	}
	/*
	 * shmBufferDropSegment() can never unmap this segment from the virtual
	 * address space of other process, of course.
	 * On the other hand, once the shared memory file on behalf of the segment
	 * is truncated, any further references to the segment will cause SIGBUS
	 * exception, and signal handler unmap the segment at other processes also.
	 */
	SHMBUF_SEGMENT_FILENAME(namebuf, segment_id, revision);
	fdesc = shm_open(namebuf, O_RDWR | O_TRUNC, 0600);
	if (fdesc < 0)
		elog(FATAL, "failed on shm_opem('%s') with O_TRUNC: %m", namebuf);
	close(fdesc);

	if (shm_unlink(namebuf) < 0)
		elog(FATAL, "failed on shm_unlink('%s'): %m", namebuf);
}

/*
 * shmBufferSplitChunk
 *
 * NOTE: It assumes shmBufferContext->lock is already held
 */
static bool
shmBufferSplitChunk(shmBufferSegment *seg, int mclass)
{
	dlist_node	   *dnode;
	shmBufferChunk *chunk_1;
	shmBufferChunk *chunk_2;
	int				mindex = mclass - SHMBUF_CHUNKSZ_MIN_BIT;

	Assert(mclass >  SHMBUF_CHUNKSZ_MIN_BIT &&
		   mclass <= SHMBUF_CHUNKSZ_MAX_BIT);
	if (dlist_is_empty(&seg->free_chunks[mindex]))
	{
		if (mclass >= SHMBUF_CHUNKSZ_MAX_BIT)
			return false;
		if (!shmBufferSplitChunk(seg, mclass+1))
			return false;
	}
	Assert(!dlist_is_empty(&seg->free_chunks[mindex]));

	dnode = dlist_pop_head_node(&seg->free_chunks[mindex]);
	chunk_1 = dlist_container(shmBufferChunk, chain, dnode);
	Assert(chunk_1->mclass == mclass &&
		   SHMBUF_CHUNK_MAGIC_HEAD(chunk_1) == SHMBUF_CHUNK_MAGIC_CODE);

	/* 1st half */
	memset(chunk_1, 0, offsetof(shmBufferChunk, data));
	chunk_1->mclass = mclass - 1;
	SHMBUF_CHUNK_MAGIC_HEAD(chunk_1) = SHMBUF_CHUNK_MAGIC_CODE;
	dlist_push_tail(&seg->free_chunks[mindex-1], &chunk_1->chain);

	/* 2nd half */
	chunk_2 = (shmBufferChunk *)((char *)chunk_1 + (1UL << (mclass - 1)));
	memset(chunk_2, 0, offsetof(shmBufferChunk, data));
	chunk_2->mclass = mclass - 1;
	SHMBUF_CHUNK_MAGIC_HEAD(chunk_2) = SHMBUF_CHUNK_MAGIC_CODE;
	dlist_push_tail(&seg->free_chunks[mindex-1], &chunk_2->chain);

	return true;
}

/*
 * shmBufferAllocChunk
 *
 * NOTE: caller must hold lock of the shmBufferContext that currently owns
 *       the segment.
 */
static shmBufferChunk *
__shmBufferAllocChunkFromSegment(shmBufferSegment *seg, Size required)
{
	shmBufferChunk *chunk;
	dlist_node	   *dnode;
	Size			chunk_sz;
	int				mclass, mindex;

	chunk_sz = (offsetof(shmBufferChunk, data) +	/* header */
				required +							/* payload */
				sizeof(uint32));					/* magic */
	mclass = get_next_log2(chunk_sz);
	if (mclass < SHMBUF_CHUNKSZ_MIN_BIT)
		mclass = SHMBUF_CHUNKSZ_MIN_BIT;
	else if (mclass > SHMBUF_CHUNKSZ_MAX_BIT)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("too large shared memory allocation required: %zu",
						required),
				 errhint("try to enlarge shmbuf.segment_size")));
	mindex = mclass - SHMBUF_CHUNKSZ_MIN_BIT;

	if (dlist_is_empty(&seg->free_chunks[mindex]))
	{
		if (!shmBufferSplitChunk(seg, mclass + 1))
			return NULL;
	}
	dnode = dlist_pop_head_node(&seg->free_chunks[mindex]);
	chunk = dlist_container(shmBufferChunk, chain, dnode);
	Assert(chunk->mclass == mclass);
	Assert(SHMBUF_CHUNK_MAGIC_HEAD(chunk) == SHMBUF_CHUNK_MAGIC_CODE);
	/* setup shmBufferChunk */
	memset(&chunk->chain, 0, sizeof(dlist_node));
	chunk->required = required;
	SHMBUF_CHUNK_MAGIC_TAIL(chunk) = SHMBUF_CHUNK_MAGIC_CODE;

	seg->num_actives++;

	return chunk;
}

static shmBufferChunk *
shmBufferAllocChunk(shmBufferContext *context, Size required)
{
	shmBufferSegment *seg;
	shmBufferChunk *chunk;
	dlist_iter		iter;

	dlist_foreach (iter, &context->active_segment_list)
	{
		shmBufferSegment   *seg = dlist_container(shmBufferSegment,
												  chain, iter.cur);
		chunk = __shmBufferAllocChunkFromSegment(seg, required);
		if (chunk)
		{
			chunk->memcxt = (MemoryContext) context;
			return chunk;
		}
	}
	/* create a new segment */
	seg = shmBufferCreateSegment();
	dlist_push_head(&context->active_segment_list, &seg->chain);
	chunk = __shmBufferAllocChunkFromSegment(seg, required);
	if (!chunk)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of shared memory"),
				 errhint("enlarge shmbuf.num_logical_segments")));
	
	chunk->memcxt = (MemoryContext) context;
	return chunk;
}

/*
 * shmBufferFreeChunk
 *
 * NOTE: caller must hold the shmBufferContext->lock of the memory context
 */
static bool
shmBufferFreeChunk(shmBufferSegment *seg, shmBufferChunk *chunk)
{
	char	   *mmap_ptr = shmBufferSegmentMmapPtr(seg);

	Assert(chunk->mclass >= SHMBUF_CHUNKSZ_MIN_BIT &&
		   chunk->mclass <= SHMBUF_CHUNKSZ_MAX_BIT &&
		   SHMBUF_CHUNK_CHECK_MAGIC(chunk));
	while (chunk->mclass <= SHMBUF_CHUNKSZ_MAX_BIT)
	{
		Size	offset = (uintptr_t)chunk - (uintptr_t)mmap_ptr;
		Size	shift = (1UL << chunk->mclass);

		if ((offset & shift) == 0)
		{
			shmBufferChunk *buddy
				= (shmBufferChunk *)((char *)chunk + shift);

			if ((char *)buddy >= mmap_ptr + shmbuf_segment_size)
				break;		/* out of range */
			Assert(SHMBUF_CHUNK_MAGIC_HEAD(buddy) == SHMBUF_CHUNK_MAGIC_CODE);
			/* is this buddy merginable? */
			if (buddy->mclass != chunk->mclass ||
				!buddy->chain.prev ||
				!buddy->chain.next)
				break;
			/* Ok, let's merge them */
			dlist_delete(&buddy->chain);
			memset(buddy, 0, offsetof(shmBufferChunk, data));
			chunk->mclass++;
		}
		else
		{
			shmBufferChunk *buddy
				= (shmBufferChunk *)((char *)chunk - shift);

			if ((char *)buddy < mmap_ptr)
				break;		/* out of range */
			Assert(SHMBUF_CHUNK_MAGIC_HEAD(buddy) == SHMBUF_CHUNK_MAGIC_CODE);
			/* is this buddy merginable? */
			if (buddy->mclass != chunk->mclass ||
				!buddy->chain.prev ||
				!buddy->chain.next)
				break;
			/* Ok, let's merge them */
			dlist_delete(&buddy->chain);
			memset(chunk, 0, offsetof(shmBufferChunk, data));
			chunk = buddy;
			chunk->mclass++;
		}
	}
	/* insert the chunk to the free list */
	dlist_push_head(&seg->free_chunks[chunk->mclass - SHMBUF_CHUNKSZ_MIN_BIT],
					&chunk->chain);
	Assert(seg->num_actives > 0);
	return (--seg->num_actives == 0);
}

/*
 * shmBufferCleanupOnPostmasterExit
 */
static void
shmBufferCleanupOnPostmasterExit(int code, Datum arg)
{
	if (MyProcPid == PostmasterPid)
	{
		DIR			   *dir = opendir("/dev/shm");
		struct dirent  *dentry;
		char			namebuf[NAMEDATALEN];
		size_t			namelen;

		namelen = snprintf(namebuf, sizeof(namebuf),
						   ".pg_shmbuf_%u.", PostPortNumber);
		if (!dir)
			return;
		while ((dentry = readdir(dir)) != NULL)
		{
			if (dentry->d_type != DT_REG)
				continue;
			if (strncmp(dentry->d_name, namebuf, namelen) == 0)
			{
				if (shm_unlink(dentry->d_name) != 0)
					elog(LOG, "failed on shm_unlink('%s'): %m",
						 dentry->d_name);
				else
					elog(LOG, "shared memory segment [%s] is removed.",
						 dentry->d_name);
			}
		}
		closedir(dir);
	}
}

/*
 * shmemContextAlloc
 */
static void *
shmemContextAlloc(MemoryContext __context, Size required)
{
	shmBufferContext *context = (shmBufferContext *) __context;
	shmBufferChunk *chunk;

	SpinLockAcquire(&context->lock);
	PG_TRY();
	{
		chunk = shmBufferAllocChunk(context, required);
	}
	PG_CATCH();
	{
		SpinLockRelease(&context->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&context->lock);

	if (!chunk)
		return NULL;
	Assert(chunk->memcxt == __context);
	return chunk->data;
}

/*
 * shmemPointerValidation
 */
#ifdef USE_ASSERT_CHECKING
static bool
shmemPointerValidation(shmBufferContext *context,
					   shmBufferSegment *seg,
					   shmBufferChunk *chunk)
{
	dlist_iter	iter;

	dlist_foreach (iter, &context->active_segment_list)
	{
		shmBufferSegment *__seg = dlist_container(shmBufferSegment,
												  chain, iter.cur);
		if (seg == __seg)
			return true;
	}
	return false;	/* not found */
}
#endif	/* USE_ASSERT_CHECKING */

/*
 * shmemContextFree
 */
static void
shmemContextFree(MemoryContext __context, void *pointer)
{
	shmBufferContext   *context = (shmBufferContext *) __context;
	shmBufferChunk	   *chunk = SHMBUF_POINTER_GET_CHUNK(pointer);
	shmBufferSegment   *seg = shmBufferSegmentFromChunk(chunk);

	SpinLockAcquire(&context->lock);
	Assert(shmemPointerValidation(context, seg, chunk));

	/* release chunk, and drop segment if it becomes empty */
	if (shmBufferFreeChunk(seg, chunk))
	{
		/*
		 * If this chunk is the last one in the segment, we detach it from
		 * the MemoryContext (so, nobody allocates a new chunk concurrently),
		 * then drop the shared memory file on behalf of the segment.
		 * It shall be backed to the free_segment_list for reuse, but it shall
		 * have different revision number when someone maps the segment again.
		 */
		dlist_delete(&seg->chain);
		shmBufferDropSegment(seg);

		SpinLockAcquire(&shmBufSegHead->lock);
		dlist_push_head(&shmBufSegHead->free_segment_list, &seg->chain);
		SpinLockRelease(&shmBufSegHead->lock);
	}
	SpinLockRelease(&context->lock);
}

/*
 * shmemContextRealloc
 */
static void *
shmemContextRealloc(MemoryContext __context, void *pointer, Size required)
{
	shmBufferContext   *context = (shmBufferContext *) __context;
	shmBufferChunk	   *chunk = SHMBUF_POINTER_GET_CHUNK(pointer);
	shmBufferSegment   *seg = shmBufferSegmentFromChunk(chunk);
	char			   *mmap_ptr = shmBufferSegmentMmapPtr(seg);
	Size				chunk_sz;
	Size				offset, shift;
	int					mclass;

	Assert(shmemPointerValidation(context, seg, chunk));
	chunk_sz = (offsetof(shmBufferChunk, data) +	/* header */
				required +							/* payload */
				sizeof(uint32));					/* magic */
	mclass = get_next_log2(chunk_sz);
	if (mclass < SHMBUF_CHUNKSZ_MIN_BIT)
		mclass = SHMBUF_CHUNKSZ_MIN_BIT;
	else if (mclass > SHMBUF_CHUNKSZ_MAX_BIT)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("too large shared memory allocation required: %zu",
						required),
				 errhint("try to enlarge shmbuf.segment_size")));

	SpinLockAcquire(&context->lock);
	PG_TRY();
	{
		offset = (uintptr_t)chunk - (uintptr_t)mmap_ptr;
		while (chunk->mclass < mclass)
		{
			shmBufferChunk *buddy;
		
			shift = (1UL << chunk->mclass);
			if ((offset & shift) != 0)
				break;		/* cannot merge with buddy chunk */
			buddy = (shmBufferChunk *)((char *)chunk + shift);
			if ((char *)buddy >= mmap_ptr + shmbuf_segment_size)
				break;		/* out of range */
			if (buddy->mclass != chunk->mclass)
				break;		/* next chunk has different size */
			if (!buddy->chain.prev || !buddy->chain.next)
				break;		/* next chunk is not free */

			/* merge */
			dlist_delete(&buddy->chain);
			memset(buddy, 0, offsetof(shmBufferChunk, data));
			chunk->mclass++;
		}

		if (chunk->mclass >= mclass)
		{
			/* fast realloc */
			chunk->required = required;
			SHMBUF_CHUNK_MAGIC_TAIL(chunk) = SHMBUF_CHUNK_MAGIC_CODE;
		}
		else
		{
			/* slow realloc */
			shmBufferSegment *seg = shmBufferSegmentFromChunk(chunk);
			shmBufferChunk *temp;

			Assert(required >= chunk->required);
			temp = shmBufferAllocChunk(context, required);
			memcpy(temp->data, chunk->data, chunk->required);

			/* release the original chunk */
			if (shmBufferFreeChunk(seg, chunk))
			{
				/* see comments in shmemContextFree() */
				dlist_delete(&seg->chain);
				shmBufferDropSegment(seg);

				SpinLockAcquire(&shmBufSegHead->lock);
				dlist_push_head(&shmBufSegHead->free_segment_list,
								&seg->chain);
				SpinLockRelease(&shmBufSegHead->lock);
			}
			/* replace the original chunk by the new one */
			chunk = temp;
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&context->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&context->lock);

	if (!chunk)
		return NULL;
	Assert(chunk->memcxt == __context);
	return chunk;
}

/*
 * shmemContextReset
 */
static void
shmemContextReset(MemoryContext __context)
{
	shmBufferContext   *context = (shmBufferContext *) __context;
	shmBufferSegment   *seg;
	dlist_node		   *dnode;

	SpinLockAcquire(&context->lock);
	while (!dlist_is_empty(&context->active_segment_list))
	{
		dnode = dlist_pop_head_node(&context->active_segment_list);
		seg = dlist_container(shmBufferSegment, chain, dnode);

		shmBufferDropSegment(seg);

		SpinLockAcquire(&shmBufSegHead->lock);
		dlist_push_head(&shmBufSegHead->free_segment_list, &seg->chain);
		SpinLockRelease(&shmBufSegHead->lock);
	}
	SpinLockRelease(&context->lock);
}

/*
 * shmemContextDelete
 */
static void
shmemContextDelete(MemoryContext __context)
{
	shmBufferContext   *context = (shmBufferContext *) __context;

	if (__context == TopSharedMemoryContext)
		elog(ERROR, "unable to delete TopSharedMemoryContext");
	shmemContextReset(__context);
	SpinLockAcquire(&shmBufSegHead->lock);
	dlist_delete(&context->chain);
	SpinLockRelease(&shmBufSegHead->lock);
	pfree(context);
}

/*
 * shmemContextGetChunkSpace
 */
static Size
shmemContextGetChunkSpace(MemoryContext context, void *pointer)
{
	shmBufferChunk *chunk = SHMBUF_POINTER_GET_CHUNK(pointer);

	return (1UL << chunk->mclass);
}

/*
 * shmemContextIsEmpty
 */
static bool
shmemContextIsEmpty(MemoryContext __context)
{
	shmBufferContext *context = (shmBufferContext *) __context;
	bool		result;

	SpinLockAcquire(&context->lock);
	result = dlist_is_empty(&context->active_segment_list);
	SpinLockRelease(&context->lock);

	return result;
}

/*
 * shmemContextStatsPrint
 *
 * NOTE: PG11 changed MemoryContextMethods->stat API
 */
#if PG_VERSION_NUM < 110000
typedef void (*MemoryStatsPrintFunc) (MemoryContext context,
									  void *passthru,
									  const char *stats_string);
static void
MemoryStatsPrintfStderr(MemoryContext context,
						void *passthru,
						const char *stats_string)
{
	int		i, level = *((int *)passthru);

	for (i=0; i < level; i++)
		fputc(' ', stderr);
	fputs(stats_string, stderr);
}
#endif

static void
__shmemContextStatsPrint(MemoryContext __context,
						 MemoryStatsPrintFunc printfunc, void *passthru,
						 MemoryContextCounters *totals)
{
	shmBufferContext *context = (shmBufferContext *) __context;
	dlist_iter	iter;
	int			active_chunks = 0;
	int			free_chunks = 0;
	Size		active_space = 0;
	Size		free_space = 0;

	SpinLockAcquire(&context->lock);
	dlist_foreach (iter, &context->active_segment_list)
	{
		shmBufferSegment *seg = dlist_container(shmBufferSegment,
												chain, iter.cur);
		int		seg_id = shmBufferSegmentId(seg);
		char   *curr, *head, *tail;

		curr = head = shmbuf_segment_vaddr_head + shmbuf_segment_size * seg_id;
		tail = head + shmbuf_segment_size;
		while (curr < tail)
		{
			shmBufferChunk *chunk = (shmBufferChunk *) curr;

			if (chunk->mclass < SHMBUF_CHUNKSZ_MIN_BIT ||
				chunk->mclass > SHMBUF_CHUNKSZ_MAX_BIT ||
				SHMBUF_CHUNK_MAGIC_HEAD(chunk) != SHMBUF_CHUNK_MAGIC_CODE ||
				curr + (1UL << chunk->mclass) > tail)
				elog(ERROR, "%s: segment[%d] chunk at %zu is corrupted (required=%zu, mclass=%d, magic_head=%08x)",
					 __context->name, seg_id, (curr - head),
					 chunk->required, chunk->mclass, chunk->magic_head);
			if (chunk->chain.prev && chunk->chain.next)
			{
				free_chunks++;
				free_space += (1UL << chunk->mclass);
			}
			else
			{
				active_chunks++;
				active_space += (1UL << chunk->mclass);
			}
			curr += (1UL << chunk->mclass);
		}
	}
	SpinLockRelease(&context->lock);

	if (printfunc)
	{
		char	temp[1024];

		snprintf(temp, sizeof(temp),
				 "active (%dblocks / %zu bytes), free (%dblocks / %zu bytes)",
				 active_chunks, active_space,
				 free_chunks, free_space);
		printfunc(__context, passthru, temp);
	}

	if (totals)
	{
		totals->nblocks += (active_chunks + free_chunks);
		totals->freechunks += free_chunks;
		totals->totalspace += (active_space + free_space);
		totals->freespace += free_space;
	}
}

#if PG_VERSION_NUM < 110000
static void
shmemContextStatsPrint(MemoryContext __context, int level, bool print,
					   MemoryContextCounters *totals)
{
	__shmemContextStatsPrint(__context,
							 print ? MemoryStatsPrintfStderr : NULL,
							 &level,
							 totals);
}
#else	/* PG10 or older */
static void
shmemContextStatsPrint(MemoryContext __context,
					   MemoryStatsPrintFunc printfunc, void *passthru,
					   MemoryContextCounters *totals)
{
	__shmemContextStatsPrint(__context, printfunc, passthru, totals);
}
#endif	/* PG11 or newer */

#ifdef MEMORY_CONTEXT_CHECKING
/*
 * shmemContextCheck
 */
static void
shmemContextCheck(MemoryContext __context)
{
	shmBufferContext *context = (shmBufferContext *) __context;
	dlist_iter		iter;

	SpinLockAcquire(&context->lock);
	dlist_foreach (iter, &context->active_segment_list)
	{
		shmBufferSegment *seg = dlist_container(shmBufferSegment,
												chain, iter.cur);
		uint32		seg_id = shmBufferSegmentId(seg);
		char	   *curr, *head, *tail;

		head = curr = shmbuf_segment_vaddr_head + shmbuf_segment_size * seg_id;
		tail = head + shmbuf_segment_size;
		while (curr < tail)
		{
			shmBufferChunk *chunk = (shmBufferChunk *)curr;

			if (chunk->mclass < SHMBUF_CHUNKSZ_MIN_BIT ||
				chunk->mclass > SHMBUF_CHUNKSZ_MAX_BIT ||
				chunk->magic_head != SHMBUF_CHUNK_MAGIC_CODE ||
				curr + (1UL << chunk->mclass) > tail)
			{
				elog(WARNING, "%s: segment[%d] contains corrupted chunk at %p (offset=%zu, required=%zu, mclass=%d, magic_head=%08x)",
					 __context->name, seg_id, chunk, (curr - head),
					 chunk->required, chunk->mclass, chunk->magic_head);
				return;
			}
			curr += (1UL << chunk->mclass);
		}
	}
	SpinLockRelease(&context->lock);
}
#endif

#if 1
Datum pgstrom_shmbuf_alloc(PG_FUNCTION_ARGS);
Datum pgstrom_shmbuf_free(PG_FUNCTION_ARGS);
Datum pgstrom_shmbuf_realloc(PG_FUNCTION_ARGS);

Datum
pgstrom_shmbuf_alloc(PG_FUNCTION_ARGS)
{
	int64	required = PG_GETARG_INT64(0);
	void   *pointer = NULL;

	if (required > 0)
		pointer = MemoryContextAlloc(TopSharedMemoryContext, required);

	PG_RETURN_INT64(pointer);
}
PG_FUNCTION_INFO_V1(pgstrom_shmbuf_alloc);

Datum
pgstrom_shmbuf_free(PG_FUNCTION_ARGS)
{
	int64	pointer = PG_GETARG_INT64(0);

	pfree((void *)pointer);
	PG_RETURN_VOID();
}
PG_FUNCTION_INFO_V1(pgstrom_shmbuf_free);

Datum
pgstrom_shmbuf_realloc(PG_FUNCTION_ARGS)
{
	int64	old_ptr = PG_GETARG_INT64(0);
	int64	required = PG_GETARG_INT64(1);
	void   *new_ptr = NULL;

	if (required > 0)
	{
		MemoryContext	oldcxt = MemoryContextSwitchTo(CurrentMemoryContext);

		new_ptr = repalloc((void *)old_ptr, required);

		MemoryContextSwitchTo(oldcxt);
	}
	PG_RETURN_INT64(new_ptr);
}
PG_FUNCTION_INFO_V1(pgstrom_shmbuf_realloc);
#endif

static void
__pgstrom_shmbuf_segment_info(StringInfo str, shmBufferSegment *seg)
{
	uint32		segment_id = shmBufferSegmentId(seg);
	uint32		revision = pg_atomic_read_u32(&seg->revision);
	char	   *curr, *head, *tail;
	int			active_chunks[SHMBUF_CHUNKSZ_MAX_BIT -
							  SHMBUF_CHUNKSZ_MIN_BIT + 1];
	int			free_chunks[SHMBUF_CHUNKSZ_MAX_BIT -
							SHMBUF_CHUNKSZ_MIN_BIT + 1];
	Size		required_space = 0;
	Size		alloc_space = 0;
	Size		free_space = 0;
	int			i, count = 0;

	appendStringInfo(str, "{ \"segment-id\" : %u, \"revision\" : %u",
					 segment_id, revision);
	
	curr = head = shmbuf_segment_vaddr_head + shmbuf_segment_size * segment_id;
	tail = head + shmbuf_segment_size;
	memset(active_chunks, 0, sizeof(active_chunks));
	memset(free_chunks, 0, sizeof(free_chunks));
	while (curr < tail)
	{
		shmBufferChunk *chunk = (shmBufferChunk *)curr;

		if (chunk->mclass < SHMBUF_CHUNKSZ_MIN_BIT ||
			chunk->mclass > SHMBUF_CHUNKSZ_MAX_BIT ||
			chunk->magic_head != SHMBUF_CHUNK_MAGIC_CODE ||
			curr + (1UL << chunk->mclass) > tail)
		{
			appendStringInfo(str, ", \"corrupted\" : true");
			goto out;
		}

		if (chunk->chain.prev && chunk->chain.next)
		{
			free_chunks[chunk->mclass - SHMBUF_CHUNKSZ_MIN_BIT]++;
			free_space += (1UL << chunk->mclass);
		}
		else
		{
			active_chunks[chunk->mclass - SHMBUF_CHUNKSZ_MIN_BIT]++;
			alloc_space += (1UL << chunk->mclass);
			required_space += chunk->required;
		}
		curr += (1UL << chunk->mclass);
	}

	appendStringInfo(str, ", \"chunks\" : [");
	for (i=SHMBUF_CHUNKSZ_MIN_BIT; i<=SHMBUF_CHUNKSZ_MAX_BIT; i++)
	{
		int			mindex = i - SHMBUF_CHUNKSZ_MIN_BIT;
		char		label[100];

		if (active_chunks[mindex] == 0 && free_chunks[mindex] == 0)
			continue;
		if (i < 10)
			snprintf(label, sizeof(label), "%lub", (1UL << i));
		else if (i < 20)
			snprintf(label, sizeof(label), "%lukB", (1UL << (i-10)));
		else if (i < 30)
			snprintf(label, sizeof(label), "%luMB", (1UL << (i-20)));
		else if (i < 40)
			snprintf(label, sizeof(label), "%luGB", (1UL << (i-30)));
		else
			snprintf(label, sizeof(label), "%luTB", (1UL << (i-40)));
		
		if (count++ > 0)
			appendStringInfo(str, ", ");
		appendStringInfo(str, "{\"chunk-sz\" : \"%s\", \"active\" : %d, \"free\" : %d }",
						 label, active_chunks[mindex], free_chunks[mindex]);
	}
	appendStringInfo(str, "]");
	appendStringInfo(str, ", \"required-space\" : %zu", required_space);
	appendStringInfo(str, ", \"alloc-space\" : %zu", alloc_space);
	appendStringInfo(str, ", \"free-space\" : %zu", free_space);
out:
	appendStringInfo(str, "}");
}

static void
__pgstrom_shmbuf_context_info(StringInfo str, shmBufferContext *context)
{
	dlist_iter		iter;
	int				count = 0;

	appendStringInfo(str,"{ \"name\" : \"%s\", \"segments\" : [",
					 context->header.name);
	dlist_foreach(iter, &context->active_segment_list)
	{
		shmBufferSegment *seg = dlist_container(shmBufferSegment,
												chain, iter.cur);
		if (count++ > 0)
			appendStringInfo(str, ", ");
		__pgstrom_shmbuf_segment_info(str, seg);
	}
	appendStringInfo(str, "]}");
}

Datum
pgstrom_shmbuf_info(PG_FUNCTION_ARGS)
{
	StringInfoData	str;

	initStringInfo(&str);
	str.len += VARHDRSZ;

	appendStringInfo(&str, "[");
	SpinLockAcquire(&shmBufSegHead->lock);
	PG_TRY();
	{
		dlist_iter	iter;
		int			count = 0;
		
		dlist_foreach(iter, &shmBufSegHead->shmem_context_list)
		{
			shmBufferContext *context = dlist_container(shmBufferContext,
														chain, iter.cur);
			SpinLockAcquire(&context->lock);
			PG_TRY();
			{
				if (count++ > 0)
					appendStringInfo(&str, ", ");
				__pgstrom_shmbuf_context_info(&str, context);
			}
			PG_CATCH();
			{
				SpinLockRelease(&context->lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockRelease(&context->lock);
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&shmBufSegHead->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&shmBufSegHead->lock);
	appendStringInfo(&str, "]");

	SET_VARSIZE(str.data, str.len);
	PG_RETURN_TEXT_P(str.data);
}
PG_FUNCTION_INFO_V1(pgstrom_shmbuf_info);

/*
 * SharedMemoryContextCreate
 */
MemoryContext
SharedMemoryContextCreate(const char *name)
{
	shmBufferContext   *scxt;
	MemoryContext		mcxt;

	scxt = MemoryContextAllocZero(TopSharedMemoryContext,
								  offsetof(shmBufferContext,
										   namebuf[strlen(name) + 1]));
	strcpy(scxt->namebuf, name);
	mcxt = &scxt->header;
	/* A fake node tag to avoid Assert() checks */
	NodeSetTag(mcxt, T_AllocSetContext);
	mcxt->methods = &sharedMemoryContextMethods;
	mcxt->name = scxt->namebuf;
	SpinLockInit(&scxt->lock);
	dlist_init(&scxt->active_segment_list);

	SpinLockAcquire(&shmBufSegHead->lock);
	dlist_push_tail(&shmBufSegHead->shmem_context_list, &scxt->chain);
	SpinLockRelease(&shmBufSegHead->lock);

	return mcxt;
}

/*
 * pgstrom_startup_shmbuf
 */
static void
pgstrom_startup_shmbuf(void)
{
	shmBufferSegment *seg;
	shmBufferLocalMap *lmap;
	shmBufferChunk *chunk;
	shmBufferContext *scxt;
	Size	length;
	bool	found;
	uint32	i, j;

	/* shmBufLocalMaps */
	length = sizeof(shmBufferLocalMap) * shmbuf_num_logical_segment;
	shmBufLocalMaps = MemoryContextAllocZero(TopMemoryContext, length);

	/* shmBufSegHead */
	length = offsetof(shmBufferSegmentHead,
					  segments[shmbuf_num_logical_segment]);
	shmBufSegHead = ShmemInitStruct("shmBufferSegmentHead", length, &found);
	if (!IsUnderPostmaster)
	{
		Assert(!found);
		memset(shmBufSegHead, 0, length);
	}
	else
		Assert(found);

	SpinLockInit(&shmBufSegHead->lock);
	dlist_init(&shmBufSegHead->shmem_context_list);
	dlist_init(&shmBufSegHead->free_segment_list);
	for (i=0; i < shmbuf_num_logical_segment; i++)
	{
		/* shmBufferSegment */
		seg = &shmBufSegHead->segments[i];
		for (j=SHMBUF_CHUNKSZ_MIN_BIT; j <= SHMBUF_CHUNKSZ_MAX_BIT; j++)
			dlist_init(&seg->free_chunks[j - SHMBUF_CHUNKSZ_MIN_BIT]);
		dlist_push_tail(&shmBufSegHead->free_segment_list,
						&seg->chain);

		/* shmBufferLocalMap */
		lmap = &shmBufLocalMaps[i];
		SpinLockInit(&lmap->mutex);
		lmap->revision = 0;
		lmap->is_attached = false;
	}

	/*
	 * Build TopSharedMemoryContext
	 */
	seg = shmBufferCreateSegment();
	chunk = __shmBufferAllocChunkFromSegment(seg, offsetof(shmBufferContext,
														   namebuf));
	if (!chunk)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of shared memory"),
				 errhint("enlarge shmbuf.num_logical_segments")));
	scxt = (shmBufferContext *)chunk->data;
	SpinLockInit(&scxt->lock);
	dlist_init(&scxt->active_segment_list);
	dlist_push_tail(&scxt->active_segment_list, &seg->chain);
	chunk->memcxt = (MemoryContext) scxt;
	
	TopSharedMemoryContext = &scxt->header;
	memset(TopSharedMemoryContext, 0, sizeof(MemoryContextData));
	/* It's a fake node tag to avoid assert checking */
	NodeSetTag(TopSharedMemoryContext, T_AllocSetContext);
	TopSharedMemoryContext->methods = &sharedMemoryContextMethods;
	TopSharedMemoryContext->name = "TopSharedMemoryContext";

	dlist_push_tail(&shmBufSegHead->shmem_context_list, &scxt->chain);

	/*
	 * Because TopSharedMemoryContext must exist prior to creation
	 * of other shared memory context, we call the shmem_startup_next
	 * at the tail of _startup handler.
	 */
	if (shmem_startup_next)
		(*shmem_startup_next)();
}

/*
 * pgstrom_init_shmbuf
 */
void
pgstrom_init_shmbuf(void)
{
	struct sigaction sigact;
	size_t		length;

	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("Shared Memory Context must be initialized at shared_preload_libraries")));

	DefineCustomIntVariable("shmbuf.segment_size",
							"Unit size of the shared memory segment",
							"must be factorial of 2",
							&shmbuf_segment_size_kb,
							 256 << 10,		/* default: 256MB */
							  32 << 10,		/* min: 32MB */
							4096 << 10,		/* max: 4GB */
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	if ((shmbuf_segment_size_kb & (shmbuf_segment_size_kb - 1)) != 0)
		elog(ERROR, "shmbuf.segment_size (%dkB) is not factorial of 2",
			 shmbuf_segment_size_kb);
	
	shmbuf_segment_size = ((size_t)shmbuf_segment_size_kb) << 10;

	DefineCustomIntVariable("shmbuf.num_logical_segments",
							"Number of the logical shared memory segments",
							NULL,
							&shmbuf_num_logical_segment,
							(2 * PHYS_PAGES * PAGE_SIZE) / shmbuf_segment_size,
							10,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* preserver private address space but no physical memory assignment */
	length = shmbuf_segment_size * shmbuf_num_logical_segment;
	shmbuf_segment_vaddr_head = mmap(NULL, length,
									 PROT_NONE,
									 MAP_PRIVATE | MAP_ANONYMOUS,
									 -1, 0);
	if (shmbuf_segment_vaddr_head == MAP_FAILED)
		elog(ERROR, "failed on mmap(2): %m");
	shmbuf_segment_vaddr_tail = shmbuf_segment_vaddr_head + length;

	/* allocation of static shared memory */
	RequestAddinShmemSpace(offsetof(shmBufferSegmentHead,
									segments[shmbuf_num_logical_segment]));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_shmbuf;

	/* registration of signal handles */
	memset(&sigact, 0, sizeof(struct sigaction));
	sigact.sa_sigaction = shmBufferAttachSegmentOnDemand;
	sigemptyset(&sigact.sa_mask);
	sigact.sa_flags = SA_SIGINFO;

	memset(&sigaction_orig_sigsegv, 0, sizeof(struct sigaction));
	if (sigaction(SIGSEGV, &sigact, &sigaction_orig_sigsegv) != 0)
		elog(ERROR, "failed on sigaction(2) for SIGSEGV: %m");

	memset(&sigaction_orig_sigbus, 0, sizeof(struct sigaction));
	if (sigaction(SIGBUS, &sigact, &sigaction_orig_sigbus) != 0)
		elog(ERROR, "failed on sigaction(2) for SIGBUS: %m");

	/* memory context methods */
	memset(&sharedMemoryContextMethods, 0, sizeof(MemoryContextMethods));
	sharedMemoryContextMethods.alloc    = shmemContextAlloc;
	sharedMemoryContextMethods.free_p   = shmemContextFree;
	sharedMemoryContextMethods.realloc  = shmemContextRealloc;
	sharedMemoryContextMethods.reset    = shmemContextReset;
	sharedMemoryContextMethods.delete_context = shmemContextDelete;
	sharedMemoryContextMethods.get_chunk_space = shmemContextGetChunkSpace;
	sharedMemoryContextMethods.is_empty = shmemContextIsEmpty;
	sharedMemoryContextMethods.stats    = shmemContextStatsPrint;
#ifdef MEMORY_CONTEXT_CHECKING
	sharedMemoryContextMethods.check    = shmemContextCheck;
#endif

	/* cleanup active segments on exit of postmaster */
	before_shmem_exit(shmBufferCleanupOnPostmasterExit, 0);
}
