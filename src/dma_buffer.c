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



typedef struct dmaBufferEntry
{
	dlist_node		chain;
	dsm_handle		handle;
	pg_atomic_uint32 refcnt;
} dmaBufferEntry;

typedef struct dmaBufferEntryHead
{
	slock_t			lock;
	dlist_head		active_segment_list[];
	dlist_head		empty_segment_list;
	bool			is_mapping;		/* true, if somebody mapping */
	Bitmapset	   *waitProcs;		/* bitmap of waiting processes */
	dmaBufferEntry	entries[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferEntryHead;

/*
 * dmaBufferEntryLocal - status of local mapping of dmaBuffer
 */
typedef struct dmaBufferEntryLocal
{
	port_addr_t			vaddr;	/* virtual portable addr */
	char			   *paddr;	/* private mapped addr */
} dmaBufferEntryLocal;

/*
 * static variables
 */
static dmaBufferEntryHead	   *dmaBufEntryHead = NULL;		/* shared memory */
static dmaBufferEntryLocal	   *dmsBufEntryLocal = NULL;	/* local memory */
static int		dma_segment_size_kb;	/* GUC */
static int		max_dma_segment_nums;	/* GUC */
static int		min_dma_segment_nums;	/* GUC */

#define DMABUF_CHUNKSZ_MAX_BIT		36
#define DMABUF_CHUNKSZ_MIN_BIT		9
#define DMABUF_CHUNKSZ_MAX			(1UL << DMABUF_CHUNKSZ_MAX_BIT)
#define DMABUF_CHUNKSZ_MIN			(1UL << DMABUF_CHUNKSZ_MIN_BIT)

typedef struct dmaBufferChunk
{
	// offset from the dmaBufferSegment
	plist_node		addr_chain;		/* link by addr order */
	plist_node		free_chain;		/* link to free chunks, or zero if active*/

	char		chunk_data[FLEXIBLE_ARRAY_MEMBER];
} dmaBufferChunk;

typedef struct dmaBufferSegment
{
	
	plist_head		free_chunks[DMABUF_CHUNKSZ_MAX_BIT + 1];
} dmaBufferSegment;








/*
 * pgstrom_startup_dma_buffer
 */
static void
pgstrom_startup_dma_buffer(void)
{
	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

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
	RequestAddinShmemSpace(...);

	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_dma_buffer;
}






