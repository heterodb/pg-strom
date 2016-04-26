/*
 * shmem_aset.c
 *
 * Allocation set of host-pinned and portable shared memory segment
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

#include "nodes/memnodes.h"
#include "storage/dsm.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "pg_strom.h"

typedef struct shmemGlobalState
{
	LWLock			mutex;
	cl_uint			segid_global_max;
	pg_atomic_uint32	 num_alloced;
	dsm_handle		handles[FLEXIBLE_ARRAY_MEMBER];
} shmemGlobalState;

typedef struct shmemLocalState
{
	cl_uint			segid_local_max;
	dsm_segment	   *segments[FLEXIBLE_ARRAY_MEMBER];
} shmemLocalState;


/* shmemHead */
typedef struct shmemHead
{
	volatile slock_t lock;
	cl_uint			num_active_chunks;
	Size			effective_size;
	portable_list	free_list[sizeof(void *) * BITS_PER_BYE];
	char			data[FLEXIBLE_ARRAY_MEMBER];
} shmemHead;

/* shmemChunk */
#define SHMEM_CHUNK_HEAD_MAGIC		0xF00DCAFE
#define SHMEM_CHUNK_TAIL_MAGIC		0xDEADBEAF
typedef struct shmemChunk
{
	portable_list	free_chain;		/* link if free chunk */
	Size			required;		/* actually required size */
	cl_uint			class;			/* class of allocation size */
	cl_int			refcnt;			/* reference count */
	cl_uint			segid;			/* segment id this chunk belongs to */
	cl_uint			head_magic;		/* front overrun check */
	char			data[FLEXIBLE_ARRAY_MEMBER];
} shmemChunk;

#define SHMEM_CHUNK_IS_ACTIVE(chunk)			\
	(!(chunk)->free_chain.prev && !(chunk)->free_chain.next)
#define SHMEM_CHUNK_IS_FREE(chunk)				\
	(!SHMEM_CHUNK_IS_ACTIVE(chunk))
#define SHMEM_CHUNK_TAIL_MAGIC(chunk)			\
	*((cl_uint *)((chunk)->data + INTALIGN((chunk)->required)))
#define SHMEM_CHUNK_CHECK_MAGIC(chunk)									\
	do {																\
		Assert((chunk)->head_magic == SHMEM_CHUNK_HEAD_MAGIC);			\
		Assert(SHMEM_CHUNK_CHECK_MAGIC(chunk) == SHMEM_CHUNK_TAIL_MAGIC); \
	} while(0)

typedef struct shmemContext
{
	MemoryContextData header;	/* Standard memory-context fields */
	cl_uint		nrooms;		/* size of pointers[] array */
	cl_uint		nitems;		/* usage of pointers[] array */
	void	  **pointers;
} shmemContext;

/* static variables */
static shmem_startup_hook_type	shmem_startup_next = NULL;
static shmemGlobalState		   *shmem_global_state = NULL;
static shmemLocalState		   *shmem_local_state = NULL;
static Size						shmem_segment_size;
static int						shmem_num_segments;
static int						shmem_release_policy;
static int						portable_addr_shift;
static portable_addr			portable_addr_mask;
static bool						needs_shmem_register = false;

#define SHMEM_MIN_CHUNK_BITS	9		/* 512 bytes */
#define SHMEM_MAX_CHUNK_BITS	(portable_addr_shift - 1)
#define SHMEM_MIN_CHUNK_SIZE	(1UL << SHMEM_MIN_CHUNK_BITS)
#define SHMEM_MAX_CHUNK_SIZE	(1UL << SHMEM_MAX_CHUNK_BITS)

/* shmem release policy */
#define SHMEM_RELEASE_POLICY__ALWAYS		1
#define SHMEM_RELEASE_POLICY__ADAPTIVE		2
#define SHMEM_RELEASE_POLICY__NEVER			3

/*
 * map/unmap a particular DSM segment on local process
 */
static void
shmem_map_segment(int segid, dsm_segment *dsm)
{
	CUresult		rc;

	Assert(segid > 0 && segid <= shmem_num_segments);
	Assert(!shmem_local_state->segments[segid - 1]);

	if (needs_shmem_register)
	{
		rc = cuMemHostRegister(dsm_segment_address(dsm),
							   shmem_segment_size,
							   CU_MEMHOSTREGISTER_PORTABLE);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemHostRegister: %s",
				 errorText(rc));
	}
	dsm_pin_mapping(dsm);
	shmem_local_state->segments[segid - 1] = dsm;
	if (shmem_local_state->max_local_segid < segid)
		shmem_local_state->max_local_segid < segid;
}

static void
shmem_unmap_segment(int segid)
{
	dsm_segment	   *dsm;
	CUrecult		rc;

	Assert(segid > 0 && segid <= shmem_num_segments);
	Assert(shmem_local_state->segments[segid - 1]);

	dsm = shmem_local_state->segments[segid - 1];
	shmem_local_state->segments[segid - 1] = NULL;
	if (shmem_local_state->max_local_segid == segid)
	{
		while (segid > 0 && !shmem_local_state->segments[segid - 1])
			segid--;
		shmem_local_state->max_local_segid = segid;
	}

	if (shmem_needs_register)
	{
		rc = cuMemHostUnregister(dsm_segment_address(dsm));
		if (rc != CUDA_SUCCESS)
			elog(NOTICE, "failed on cuMemHostRegister: %s", errorText(rc));
	}
	dsm_detach(dsm);
}












/* ----------------------------------------------------------------
 *
 * Translation between 
 *
 *
 *
 */


#define PORTABLE_SEG_ID(paddr)		((paddr) >> portable_addr_shift)
#define PORTABLE_OFFSET(paddr)		((paddr) & portable_addr_mask)
#define MAKE_PORTABLE_ADDR(seg_id, offset)		\
	(((seg_id) << portable_addr_shift) | (offset))

/*
 * transform a local address into the portable form
 */
portable_addr
pg_addr_to_portable(void *local_addr)
{
	void	   *base_addr;
	cl_uint		i;

	for (i=0; i < shmem_local_state->num_segments; i++)
	{
		if (!shmem_local_state->segments[i])
			continue;

		base_addr = dsm_segment_address(shmem_local_state->segments[i]);
		if ((uintptr_t)local_addr >= (uintptr_t)base_addr &&
			(uintptr_t)local_addr < ((uintptr_t)base_addr +
									 shmem_local_state->segment_size))
		{
			Size	offset = (uintptr_t)local_addr - (uintptr_t)base_addr;

			return MAKE_PORTABLE_ADDR(i + 1, offset);
		}
	}
	elog(ERROR, "local_addr %p does not point portable shared memory segment",
		local_addr);

}

/*
 * transform a portable address into relevant local address
 * if segment that includes the portable address was not mapped yet,
 * it also mapps the segment.
 */
void *
pg_portable_to_addr(portable_addr paddr)
{
	dsm_segment	   *dsm;
	dsm_handle		handle;
	int				segid;
	Size			offset;

	/* quick bailout if NULL */
	if (paddr == 0)
		return NULL;

	segid = PORTABLE_SEG_ID(paddr);
	offset = PORTABLE_OFFSET(paddr);
	if (segid < 1 || segid > shmem_num_segments)
		elog(ERROR, "portable_address %lu out of range", paddr);
	Assert(offset < shmem_segment_size);

	dsm = shmem_local_state->segments[segid - 1];
	if (!dsm)
	{
		LWLockAcquire(&shmem_global_state->mutex, LW_SHARED);
		handle = shmem_global_state->handles[seg_id - 1];
		if (!handle)
			elog(ERROR, "portable_address %lu refers invalid segment", paddr);
		LWLockRelease(&shmem_global_state->mutex);
		shmem_map_segment(dsm_attach(handle), segid);
	}
	return (char *)dsm_segment_address(dsm) + offset;
}

/* caller has to hold shmem_global_state->mutex exclusively */
static void
shmem_init_segment(dsm_segment *dsm)
{
	shmemHead  *shead = dsm_segment_address(dsm);
	shmemChunk *chunk;
	char	   *curr_addr;
	char	   *end_addr;
	int			i, class;

	SpinLockInit(&shead->lock);
	shead->num_active_chunks = 0;
	for (i=0; i < lengthof(shead->free_list); i++)
	{
		plist_init(&shead->free_list[i]);
	}

	/* makes free entries */
	curr_addr = (char *) shead->first_chunk;
	end_addr = (char *) shead + shmem_segment_size;

	class = SHMEM_MAX_CHUNK_BITS;
	while (class >= SHMEM_MIN_CHUNK_BITS)
	{
		if (curr_addr + (1UL << class) > end_addr)
		{
			class--;
			continue;
		}
		chunk = (shmemChunk *) curr_addr;
		memset(chunk, 0, offsetof(shmemChunk, data));
		chunk->class = class;
		plist_push_node(&shead->free_list[class], &chunk->free_chain);

		curr_addr += (1UL << class);
	}
	shead->effective_size = curr_addr - shead->data;
}



/* caller must hold shead->lock */
static bool
shmem_chunk_split(shmemHead *shead, int class)
{
	shmemChunk	   *chunk;
	shmemChunk	   *buddy;
	portable_list  *pnode;

	if (plist_is_empty(&shead->free_list[class]))
	{
		if (shift == SHMEM_MAX_CHUNK_BITS ||
			!shmem_chunk_split(shead, class + 1))
			return false;
	}
	Assert(!plist_is_empty(&shead->free_list[class]));

	pnode = plist_pop_node(&shead->free_list[class]);
	chunk = plist_container(shmemChunk, free_chain, pnode);
	Assert(chunk->class == class);
	Assert((((uintptr_t)chunk -
			 (uintptr_t)shead->data) & ((1UL << class) - 1)) == 0);
	class--;

	/* earlier half */
	memset(chunk, 0, offsetof(shmemChunk, data[0]));
	chunk->class = class;
	plist_push_node(&shead->free_list[class], &chunk->free_chain);

	/* later half */
	chunk = (shmemChunk *)((char *)chunk + (1UL << class));
	memset(chunk, 0, offsetof(shmemChunk, data[0]));
	plist_push_node(&shead->free_list[class], &chunk->free_chain);

	return true;
}

static shmemChunk *
__shmem_chunk_alloc(dsm_segment *dsm, int segid, int class, Size required)
{
	shmemHead	   *shmem_head = dsm_segment_address(dsm);
	portable_list  *pnode;
	shmemChunk	   *chunk;

	SpinLockAcquire(&shmem_head->lock);
	if (plist_is_empty(&shmem_head->free_list[class]))
	{
		if (!shmem_chunk_split(shmem_head, class + 1))
		{
			SpinLockRelease(&shmem_head->lock);
			return NULL;
		}
	}
	Assert(!plist_is_empty(&shmem_head->free_list[class]));

	pnode = plist_pop_node(&shmem_head->free_list[class]);
	chunk = plist_container(shmemChunk, free_chain, pnode);
	memset(&chunk->free_chain, 0, sizeof(portable_list));	/* active */
	chunk->required = required;
	chunk->refcnt = 1;
	chunk->class = class;
	chunk->segid = segid;
	chunk->head_magic = SHMEM_CHUNK_HEAD_MAGIC;
	SHMEM_CHUNK_TAIL_MAGIC(chunk) = SHMEM_CHUNK_TAIL_MAGIC;

	SpinLockRelease(&shmem_head->lock);

	return chunk;
}

static void *
shmem_chunk_alloc(Size required)
{
	Size			total_sz;
	int				class;
	int				segid;
	dsm_handle		handle;
	dsm_segment	   *dsm;
	shmemChunk	   *chunk;

	/* length of chunk actually required */
	total_sz = (offsetof(shmemChunk, data) +
				INTALIGN(required) +
				sizeof(cl_uint));
	/* round up the total size if too small */
	if (total_sz < min_chunk_size)
		total_sz = min_chunk_size;
	/* give up if total size is too large */
	if (total_sz > max_chunk_size)
		return NULL;

	/* allocation shall be 2^class unit size */
	class = get_next_log2(total_sz);

	/*
	 * Step-1. Try to walk on the segment already mapped on
	 */
	Assert(shmem_local_state->segid_local_max <= shmem_num_segments);
	for (segid=1; segid < shmem_local_state->segid_local_max; segid++)
	{
		dsm = shmem_local_state->segments[segid - 1];
		if (dsm)
		{
			chunk = __shmem_chunk_alloc(dsm, segid, class, required);
			if (chunk)
				return chunk->data;
		}
	}

	/*
	 * Step-2. Try to walk on the segment already allocated on
	 */
	LWLockAcquire(&shmem_global_state->mutex, LW_SHARED);
	Assert(shmem_global_state->segid_global_max <= shmem_num_segments);
	for (segid = 1; shmem_global_state->segid_global_max; segid++)
	{
		if (shmem_local_state->segments[segid - 1])
			continue;

		handle = shmem_global_state->handles[segid - 1];
		if (handle)
		{
			dsm = dsm_attach(handle);
			shmem_map_segment(dsm, segid);
			chunk = __shmem_chunk_alloc(dsm, segid, class, required);
			if (chunk)
			{
				LWLockRelease(&shmem_global_state->mutex);
				return chunk->data;
			}
		}
	}
	LWLockRelease(&shmem_global_state->mutex);

	/*
	 * Step-3. Try to allocate a new DSM segment
	 */
	LWLockAcquire(&shmem_global_state->mutex, LW_EXCLUSIVE);
	for (segid=1; segid <= shmem_num_segments; segid++)
	{
		if (shmem_local_state->segments[segid - 1])
			continue;

		handle = shmem_global_state->handles[segid - 1];
		if (handle)
		{
			/* in case of concurrent allocation by other process */
			dsm = dsm_attach(handle);
			shmem_map_segment(dsm, segid);
		}
		else
		{
			/* create a new DSM segment */
			dsm = dsm_create(shmem_segment_size, 0);
			shmem_init_segment(dsm);
			shmem_map_segment(dsm, segid);

			/* DSM should not be removed unless explicit deletion */
			dsm_pin_segment(dsm);
			shmem_global_state->handles[segid - 1] = dsm_segment_handle(dsm);
			if (shmem_global_state->segid_global_max < segid)
				shmem_global_state->segid_global_max = segid;
		}

		chunk = __shmem_chunk_alloc(dsm, segid, class, required);
		if (chunk)
		{
			LWLockRelease(&shmem_global_state->mutex);
			return chunk->data;
		}
	}
	LWLockRelease(&shmem_global_state->mutex);

	return NULL;	/* no memory to allocate */
}

/* caller must acquire shead->lock */
static void
__shmem_chunk_free(shmemHead *shead, shmemChunk *chunk)
{
	cl_int		class = chunk->class;
	Size		offset = (uintptr_t)chunk - (uinttr_t)shead->data;
	shmemChunk *buddy;

	Assert(chunk->refcnt == 0);
	Assert(SHMEM_CHUNK_IS_ACTIVE(chunk));
	Assert((offset & ((1UL << class) - 1)) == 0);

	/* try to merge its buddy chunk, if it is also free */
	while (class < SHMEM_MAX_CHUNK_BITS)
	{
		offset = (uintptr_t)chunk - (uinttr_t)shead->data;
		if ((offset & (1UL << class)) == 0)
			buddy = (shmemChunk *)((char *)chunk + (1UL << class));
		else
			buddy = (shmemChunk *)((char *)chunk - (1UL << class));

		if ((char *)buddy < shead->data ||
			(char *)buddy + (1UL << class) > (shead->data +
											  shead->effective_size) ||
			buddy->class != class ||
			SHMEM_CHUNK_IS_ACTIVE(buddy))
			break;		/* cannot merge any more */

		/* OK, we can merge the chunk and buddy */
		Assert(SHMEM_CHUNK_IS_FREE(buddy));
		plist_delete(&buffer->free_list);
		if (buddy < chunk)
			chunk = buddy;
		chunk->class = ++class;
	}
	plist_push_node(&shead->free_list[class], &chunk->free_chain);
}

static void
shmem_chunk_free(void *pointer)
{
	shmemHead	   *shead;
	shmemChunk	   *chunk;
	dsm_segment	   *dsm;
	uintptr_t		head_addr;
	uintptr_t		tail_addr;
	int				segid;

	chunk = (shmemChunk *)((char *)pointer - offsetof(shmemChunk, data));
	segid = chunk->segid;

	if (segid < 1 || segid > shmem_local_state->segid_local_max)
		elog(ERROR, "Bogus shmem pointer %p was given", pointer);

	dsm = shmem_local_state->segment[segid - 1];
	if (!dsm)
		elog(ERROR, "segid=%d points unmapped shared memory segment", segid);

	if ((char *)chunk < shead->data ||
		(char *)chunk + (1UL << chunk->class) > (shead->data +
												 shead->effective_size))
		elog(ERROR, "pointer=%p points out of the segment", pointer);

	SpinLockAcquire(&shead->lock);
	Assert(shead->num_actives > 0);
	Assert(chunk->refcnt > 0);
	if (--chunk->refcnt == 0)
	{
		__shmem_chunk_free(shead, chunk);
		if (--shead->num_actives == 0)
		{
			/* TODO: release DSM? */
		}
	}
	SpinLockRelease(&shead->lock);
}





static void *
shmem_cxt_alloc(MemoryContext context, Size required)
{
	shmemContext   *shmem_cxt = (shmemContext *) context;
	void		   *addr;

	addr = shmem_chunk_alloc(required);
	if (!addr)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of pinned memory")));

	if (shmem_cxt->nitems >= shmem_cxt->nrooms)
	{
		cl_uint		nrooms_new = 2 * shmem_cxt->nrooms;

		shmem_cxt->pointers = repalloc(shmem_cxt->pointers,
									   sizeof(void *) * nrooms_new);
		shmem_cxt->nrooms = nrooms_new;
	}
	shmem_cxt->pointers[shmem_cxt->nitems++] = addr;

	return addr;
}

static void
shmem_cxt_free(MemoryContext context, void *pointer)
{}

static void *
shmem_cxt_realloc(MemoryContext context, void *pointer, Size size)
{}

static  void
shmem_cxt_init(MemoryContext context)
{
	shmemContext   *shmem_cxt = (shmemContext *) context;

	shmem_cxt->nrooms = 200;	/* initial value */
	shmem_cxt->nitems = 0;

	PG_TRY();
	{
		shmem_cxt->pointers = MemoryContextAllocZero(TopMemotyContext,
													 sizeof(void *) *
													 shmem_cxt->nrooms);
	}
	PG_CATCH();
	{
		pfree(shmem_cxt);
		PG_RE_THROW();
	}
	PG_END_TRY();

}

static void
shmem_cxt_reset(MemoryContext context)
{}

static void
shmem_cxt_delete_context(MemoryContext context)
{}

static Size
shmem_cxt_get_chunk_space(MemoryContext context, void *pointer)
{}

static bool
shmem_cxt_is_empty(MemoryContext context)
{}

static void
shmem_cxt_stats(MemoryContext context, int level, bool print,
				 MemoryContextCounters *totals)
{}

#ifdef MEMORY_CONTEXT_CHECKING
static void
shmem_cxt_check(MemoryContext context);
#endif




/*
 * table of callbacks
 */
static MemoryContextMethods shmem_context_methods = {
	shmem_cxt_alloc,
	shmem_cxt_free,
	shmem_cxt_realloc,
	shmem_cxt_init,
	shmem_cxt_reset,
	shmem_cxt_delete_context,
	shmem_cxt_get_chunk_space,
	shmem_cxt_is_empty,
	shmem_cxt_stats,
#ifdef MEMORY_CONTEXT_CHECKING
	shmem_cxt_check,
#endif
};

/*
 *
 */
MemoryContext
pgstrom_shmem_context_create(MemoryContext parent, const char *name)
{
	return MemoryContextCreate(T_AllocSetContext,
							   sizeof(shmemContext),
							   &shmem_context_methods,
							   parent,
							   name);
}

/*
 * allocation of static shared memory
 */
void
pgstrom_startup_shmem_context(void)
{
	bool	found;

	if (shmem_startup_next)
		shmem_startup_next();

	shmem_global_state = (shmemGlobalState *)
		ShmemInitStruct("PG-Strom Global Shmem State",
						MAXALIGN(offsetof(shmemGlobalState,
										  handles[shmem_num_segments])),
						&found);

	if (!IsUnderPostmaster)
	{
		Assert(!found);

		LWLockInitialize(&shmem_global_state->mutex);
		memset(shmem_global_state->handles,
			   0,
			   sizeof(dsm_handle) * shmem_num_segments);
	}
	else
		Assert(found);

	/* shmem_local_state */
	shmem_local_state = (shmemLocalState *)
		MemoryContextAllocZero(TopMemoryContext,
							   offsetof(shmemLocalState,
										segments[shmem_num_segments]));
}

/*
 * init handler of yet another shared memory management
 */
void
pgstrom_init_shmem_context(void)
{
	static int		guc_segment_size;
	static int		guc_segment_limit;
	static const struct config_enum_entry release_policy_options[] = {
		{"always",		SHMEM_RELEASE_POLICY__ALWAYS,	false},
		{"adaptive",	SHMEM_RELEASE_POLICY__ADAPTIVE,	false},
		{"never",		SHMEM_RELEASE_POLICY__NEVER,	false},
		{NULL, 0, false},
	};
	int		num_segments;

	DefineCustomIntVariable("pg_strom.shmem_segment_size",
							"size of shared memory segment",
							NULL,
							&guc_segment_size,
							1024 * 1024,	/* 1GB */
							 256 * 1024,	/* 256MB */
							INT_MAX,		/* 2TB */
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	if ((guc_segment_size & (guc_segment_size - 1)) != guc_segment_size)
		elog(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("\"pg_strom.shmem_segment_size\" must be power of 2")));

	DefineCustomIntVariable("pg_strom.shmem_segment_limit",
							"total limit of shared memory",
							NULL,
							&guc_segment_limit,
							256 * guc_segment_size,	/* 256GB */
							10 * seg_size,			/*  10GB */
							INT_MAX,				/*   2TB */
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	if (guc_segment_limit % guc_segment_size != 0)
		elog(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("\"pg_strom.shmem_segment_limit\" must be "
					 "multiple number of \"pg_strom.shmem_segment_size\"")));

	DefineCustomEnumVariable("pg_strom.shmem_release_policy",
							 "policy for release of unused shared memory",
							 NULL,
							 &shmem_release_policy,
							 SHMEM_RELEASE_POLICY__ADAPTIVE,
							 release_policy_options,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* setup of static variables */
	shmem_segment_size = (Size)guc_segment_size << 10;
	shmem_num_segments = guc_segment_limit / guc_segment_size;

	portable_addr_shift = get_next_log2(shmem_segment_size);
	portable_addr_mask = (1UL << portable_addr_shift) - 1;

	/* requirement of the static shared memory */
	RequestAddinShmemSpace(MAXALIGN(offsetof(shmemGlobalState,
											 handles[shmem_num_segments])));
	/* callback for static allocation */
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_shmem_context;
}
