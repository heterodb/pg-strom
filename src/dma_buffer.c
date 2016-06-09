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
 * It represents the status of DSM segments already exists, but may not be
 * mapped on the private address space of individual processes. Once a
 * portable address that points a DMA buffer not mapped but exists, it will
 * map the segment on demand.
 *
 * Note that dmaBufferEntryHead and dmaBufferEntry are allocated on the
 * static shared memory region, thus, we can use dlist instead of plist.
 */
typedef struct dmaBufferEntry
{
	dlist_node		chain;
	cl_uint			segment_id;
	slock_t			lock;		/* also used for alloc/free */
	cl_int			map_count;
	dsm_handle		handle;
} dmaBufferEntry;

typedef struct dmaBufferEntryHead
{
	LWLock			mutex;
	dlist_head		active_segment_list;
	dlist_head		inactive_segment_list;
	dmaBufferEntry	entries[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferEntryHead;

/*
 * dmaBufferLocalEntry - status of local mapping of dmaBuffer
 */
typedef struct dmaBufferLocalEntry
{
	dmaBufferEntry *entry;
	dsm_segment	   *segment;	/* system DSM segment */
	port_addr_t		p_addr;		/* shared portable address */
} dmaBufferLocalEntry;

/*
 * static variables
 */
static dmaBufferEntryHead	   *dmaBufEntryHead = NULL;		/* shared memory */
static dmaBufferLocalEntry	   *dmaBufLocalEntry = NULL;
static dmaBufferLocalEntry	  **dmaBufLocalIndex = NULL;	/* index by addr */
static int						dmaBufLocalCount = 0;
static size_t		dma_segment_size;
static int			dma_segment_size_kb;	/* GUC */
static int			max_dma_segment_nums;	/* GUC */
static int			min_dma_segment_nums;	/* GUC */
static int			port_addr_shift;
static port_addr_t	port_addr_mask;

#define PADDR_GET_SEG_ID(paddr)		((paddr) >> port_addr_shift)
#define PADDR_GET_SEG_OFFSET(paddr)	((paddr) & port_addr_mask)

#define DMABUF_CHUNKSZ_MAX_BIT		36
#define DMABUF_CHUNKSZ_MIN_BIT		8
#define DMABUF_CHUNKSZ_MAX			(1UL << DMABUF_CHUNKSZ_MAX_BIT)
#define DMABUF_CHUNKSZ_MIN			(1UL << DMABUF_CHUNKSZ_MIN_BIT)
#define DMABUF_CHUNK_DATA(chunk)	((chunk)->data)
#define DMABUF_CHUNK_MAGIC_HEAD		0xDEADBEAF
#define DMABUF_CHUNK_MAGIC_TAIL		0x
#define DMABUF_CHUNK_

typedef struct dmaBufferChunk
{
	// offset from the dmaBufferSegment
	plist_node		addr_chain;		/* link by addr order */
	plist_node		free_chain;		/* link to free chunks, or zero if active*/

	cl_uint			magic_head;		/* = DMABUF_CHUNK_MAGIC_HEAD */
	char			data[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferChunk;

typedef struct dmaBufferSegment
{
	
	plist_head		free_chunks[DMABUF_CHUNKSZ_MAX_BIT + 1];
} dmaBufferSegment;




static inline dmaBufferLocalEntry *
lookupDmaBufferLocalEntry(const void *addr)
{
	dmaBufferLocalEntry *l_ent;
	cl_uint		i_head = 0;
	cl_uint		i_tail = dmaBufLocalCount;
	cl_uint		i_curr;
	cl_uint		i;

	while (i_head < i_tail)
	{
		i_curr = (i_head + i_tail) / 2;

		Assert(i_curr < dmaBufLocalCount);
		l_ent = dmaBufLocalIndex[i_curr];
		if (addr < (const char *)dsm_segment_address(l_ent->segment))
			i_tail = i_curr - 1;
		else if (addr >= ((const char *)dsm_segment_address(l_ent->segment)
						  + dma_segment_size))
			i_head = i_curr + 1;
		else
			return l_ent;
	}
	elog(WARNING, "dmaBufferLocalEntry for addr=%p not found", addr);
	for (i=0; i < dmaBufLocalCount; i++)
	{
		elog(WARNING, "dmaBufLocalEntry[%d] (%p-%p) {DSM=%p, PADDR=%p}",
			 i,
			 (char *)dsm_segment_address(l_ent->segment),
			 (char *)dsm_segment_address(l_ent->segment) + dma_segment_size,
			 l_ent->segment,
			 l_ent->p_addr);
	}
	return NULL;
}

/*
 * callback on dsm_detach
 */
static void
on_detach_dma_buffer_segment(dsm_segment *segment, Datum arg)
{
	dmaBufferLocalEntry *l_ent;
	dmaBufferEntry *entry;
	cl_int			segment_id = DatumGetInt32(arg);
	bool			mutex_locked = false;
	cl_int			i;

	Assert(segment_id >= 0 && segment_id < max_dma_segment_nums);
	l_ent = &dmaBufLocalEntry[segment_id];
	Assert(!segment || l_ent->segment == segment);
	entry = l_ent->entry;
	Assert(entry->segment_id == segment_id);

retry:
	SpinLockAcquire(&entry->lock);
	Assert(entry->map_count > 0);	/* must be active segment */
	if (entry->map_count == 1)
	{
		if (!mutex_locked)
		{
			SpinLockRelease(&entry->lock);
			LWLockAcquire(&dmaBufEntryHead->mutex, LW_EXCLUSIVE);
			mutex_locked = true;
			goto retry;
		}
		dlist_delete(&entry->chain);
		entry->map_count--;
		entry->handle = 0x12345678;
		dlist_push_tail(&dmaBufEntryHead->inactive_segment_list,
						&entry->chain);
	}
	SpinLockRelease(&entry->lock);
	if (mutex_locked)
		LWLockRelease(&dmaBufEntryHead->mutex);

	/* clean up local entry */
	memset(l_ent, 0, sizeof(dmaBufferLocalEntry));

	/* remove from the index, if any */
	for (i=0; i < dmaBufLocalCount; i++)
	{
		if (dmaBufLocalIndex[i] == l_ent)
		{
			memmove(&dmaBufLocalIndex[i],
					&dmaBufLocalIndex[i+1],
					sizeof(dmaBufferLocalEntry *) *
					(dmaBufLocalCount - (i + 1)));
			dmaBufLocalCount--;
			break;
		}
	}
}

/*
 * common post process for both of create and attach
 */
static inline void
__post_dsm_attach(dmaBufferLocalEntry *l_ent)
{
	void   *map_addr;
	int		i;
	/*
	 * index of the dmaBufferLocalEntry, according to the local address
	 * of the shared segment attached.
	 */
	map_addr = dsm_segment_address(l_ent->segment);
	for (i=0; i < dmaBufLocalCount; i++)
	{
		if (dsm_segment_address(dmaBufLocalIndex[i]->segment) > map_addr)
		{
			memmove(&dmaBufLocalIndex[i+1],
					&dmaBufLocalIndex[i],
					sizeof(dmaBufferLocalEntry *) * (dmaBufLocalCount - i));
			dmaBufLocalIndex[i] = l_ent;
			dmaBufLocalCount++;
			return;
		}
	}
	dmaBufLocalIndex[dmaBufLocalCount++] = l_ent;

	/*
	 * Pin the DSM segment, if GpuServer.
	 */
	if (IsGpuServerProcess())
	{
		CUresult	rc;

		rc = cuMemHostRegister(map_addr, dma_segment_size, 0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemHostRegister: %s", errorText(rc));
	}

}

/*
 * create_dma_buffer_segment - create a new DMA segment and attach it
 */
static cl_uint
create_dma_buffer_segment(void)
{
	dlist_node	   *dnode;
	dsm_segment	   *segment;
	dmaBufferEntry *entry;
	dmaBufferLocalEntry *l_ent;
	int				i;

	LWLockAcquire(&dmaBufEntryHead->mutex, LW_EXCLUSIVE);
	if (dlist_is_empty(&dmaBufEntryHead->inactive_segment_list))
		elog(ERROR, "out of DMA buffer segment entries");
	dnode = dlist_pop_head_node(&dmaBufEntryHead->inactive_segment_list);
	entry = dlist_container(dmaBufferEntry, chain, dnode);
	Assert(entry->segment_id < max_dma_segment_nums);
	Assert(entry->map_count == 0);
	l_ent = dmaBufLocalEntry[entry->segment_id];
	Assert(l_ent->segment == NULL && l_ent->entry == NULL);

	PG_TRY();
	{
		segment = dsm_create(dma_segment_size, 0);
		/*
		 * NOTE: DSM segment shall be managed by resource owner indirectly.
		 * Individual chunks in the segment are managed by GpuContext, and
		 * GpuContext is owned by resource owner. So, once DSM segment gets
		 * unreferenced by all the GpuContext that had chunks on the DSM,
		 * it is the time for detach.
		 */
		dsm_pin_mapping(segment);

		/* unsure to decrement map_count on detach */
		on_dsm_detach(segment, on_detach_dma_buffer_segment,
					  Int32GetDatum(entry->segment_id));

		/* init shared entry */
		entry->map_count = 1;
		entry->handle = dsm_segment_handle(segment);

		/* init local entry */
		l_ent->entry = entry;
		l_ent->segment = segment;
		l_ent->p_addr = (port_addr_t)entry->segment_id << port_addr_shift;

		/* mark it as an active segment */
		dlist_push_tail(&dmaBufEntryHead->active_segment_list,
						&entry->chain);
	}
	PG_CATCH();
	{
		dlist_push_head(&dmaBufEntryHead->inactive_segment_list,
						&entry->chain);
		PG_RE_THROW();
	}
	PG_END_TRY();
	LWLockRelease(&dmaBufEntryHead->mutex);
	/* common post process */
	__post_dsm_attach(l_ent);
}

/*
 * attach_dma_buffer_segment - attach an exist DMA buffer segment on the
 * current process's local address space.
 */
static void
attach_dma_buffer_segment(cl_int segment_id)
{
	dsm_segment	   *segment = NULL;
	dsm_handle		handle;
	dmaBufferEntry *entry;
	dmaBufferLocalEntry *l_ent;

	Assert(segment_id >= 0 && segment_id < max_dma_segment_nums);
	l_ent = &dmaBufLocalEntry[segment_id];
	Assert(l_ent->segment == NULL && l_ent->entry == NULL);

	entry = &dmaBufEntryHead->entries[segment_id];
	SpinLockAcquire(&entry->lock);
	Assert(entry->map_count > 0);	/* must be an exist segment */
	entry->map_count++;
	l_ent->entry = entry;
	handle = entry->handle;
	SpinLockRelease(&entry->lock);

	PG_TRY();
	{
		segment = dsm_attach(handle);
		on_dsm_detach(segment, on_detach_dma_buffer_segment,
					  Int32GetDatum(segment_id));
	}
	PG_CATCH();
	{
		on_detach_dma_buffer_segment(segment, Int32GetDatum(segment_id));
		PG_RE_THROW();
	}
	PG_END_TRY();
	l_ent->segment = segment;
	l_ent->p_addr = (port_addr_t)segment_id << port_addr_shift;
	/* common post process */
	__post_dsm_attach(l_ent);
}

/*
 * detach_dma_buffer_segment - detach this segment
 */
static void
detach_dma_buffer_segment(cl_uint segment_id)
{
	l_ent = &dmaBufLocalEntry[segment_id];
	Assert(l_ent->segment != NULL && l_ent->entry != NULL);
	dsm_detach(l_ent->segment);
	Assert(l_ent->segment == NULL && l_ent->entry == NULL);
}

/*
 * paddr_to_local - transform a portable shared address to local pointer
 */
void *
paddr_to_local(port_addr_t paddr)
{}

/*
 * local_to_paddr - transform a local pointer to a portable shared address
 */
port_addr_t
local_to_paddr(void *l_ptr)
{}







void *
__dmaBufferAlloc(SharedGpuContext *shgcon, Size required)
{

}

void *
dmaBufferAlloc(GpuContext_v2 *gcontext, Size required)
{
	SharedGpuContext   *shgcon = (gcontext ? gcontext->shgcon : NULL);

	return __dmaBufferAlloc(shgcon);
}

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

	length = sizeof(dmaBufferLocalEntry) * max_dma_segment_nums;
	dmaBufLocalEntry = MemoryContextAllocZero(TopMemoryContext, length);

	length = sizeof(dmaBufferLocalEntry *) * max_dma_segment_nums;
	dmaBufLocalIndex = MemoryContextAllocZero(TopMemoryContext, length);

	LWLockInitialize(&dmaBufEntryHead->mutex);
	dlist_init(&dmaBufEntryHead->active_segment_list);
	dlist_init(&dmaBufEntryHead->inactive_segment_list);

	for (i=0; i < max_dma_segment_nums; i++)
	{
		dmaBufferEntry *entry = &dmaBufEntryHead->entries[i];

		dlist_push_tail(&dmaBufEntryHead->inactive_segment_list,
						&entry->chain);
		entry->segment_id = i;
		entry->map_count = 0;
	}
}

/*
 * pgstrom_init_dma_buffer
 */
void
pgstrom_init_dma_buffer(void)
{


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
							64 << 20,		/* 64GB */
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

	/* request for the static shared memory */
	RequestAddinShmemSpace(offsetof(dmaBufferEntryHead,
									entries[max_dma_segment_nums]));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_dma_buffer;
}






