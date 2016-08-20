/*
 * nvme_strom.c
 *
 * Routines to support optional SSD-to-GPU Direct DMA Loading
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
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include "nvme_strom.h"
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define IOMAPBUF_CHUNKSZ_MAX_BIT		34		/* 16G */
#define IOMAPBUF_CHUNKSZ_MIN_BIT		12		/* 4KB */
#define IOMAPBUF_CHUNKSZ_MAX			(1UL << IOMAPBUF_CHUNKSZ_MAX_BIT)
#define IOMAPBUF_CHUNKSZ_MIN			(1UL << IOMAPBUF_CHUNKSZ_MIN_BIT)

typedef struct
{
	dlist_node			free_chain;	/* zero, if active chunk */
	cl_uint				mclass;		/* zero, if not chunk head */
} IOMapBufferChunk;

typedef struct
{
	slock_t				lock;
	dlist_head			addr_chunks;	/* chunks in order of address */
	dlist_head			free_chunks[IOMAPBUF_CHUNKSZ_MAX_BIT + 1];
	IOMapBufferChunk	iomap_chunks[FLEXIBLE_ARRAY_MEMBER];
} IOMapBufferSegment;

typedef struct
{
	IOMapBufferSegment *iomap_seg;		/* reference to shared memory */
	unsigned long		iomap_handle;
	uint32_t			gpu_page_sz;
	uint32_t			gpu_npages;
	CUdevice			cuda_device;
	CUcontext			cuda_context;
	CUdeviceptr			cuda_devptr;
	CUipcMemHandle		cuda_mhandle;
} IOMapBufferHead;

static shmem_startup_hook_type shmem_startup_next = NULL;
static const char	   *nvme_strom_ioctl_pathname = "/proc/nvme-strom";
static IOMapBufferHead *iomap_buffer_heads = NULL;	/* for each device */
static CUdeviceptr		iomap_buffer_base = 0UL;	/* per process vaddr */
static Size				iomap_buffer_size;			/* GUC */

/*
 * nvme_strom_ioctl
 */
static int
nvme_strom_ioctl(int cmd, const void *arg)
{
	static int		fdesc_nvme_strom = -1;

	if (fdesc_nvme_strom < 0)
	{
		fdesc_nvme_strom = open(nvme_strom_ioctl_pathname, O_RDONLY);
		if (fdesc_nvme_strom)
			elog(ERROR, "failed to open %s: %m", nvme_strom_ioctl_pathname);
	}
	return ioctl(fdesc_nvme_strom, cmd, arg);
}

/*
 * gpuDmaMemSplitIOMap
 */
static bool
gpuDmaMemSplitIOMap(IOMapBufferSegment *iomap_seg, int mclass)
{
	IOMapBufferChunk   *iomap_chunk_1;
	IOMapBufferChunk   *iomap_chunk_2;
	dlist_node		   *dnode;
	int					offset;

	if (mclass > IOMAPBUF_CHUNKSZ_MAX_BIT)
		return false;
	Assert(mclass > IOMAPBUF_CHUNKSZ_MIN_BIT);

	if (dlist_is_empty(&iomap_seg->free_chunks[mclass]))
	{
		if (!gpuDmaMemSplitIOMap(iomap_seg, mclass + 1))
			return false;
	}
	Assert(!dlist_is_empty(&iomap_seg->free_chunks[mclass]));

	offset = 1UL << (mclass - 1 - IOMAPBUF_CHUNKSZ_MIN_BIT);
	dnode = dlist_pop_head_node(&iomap_seg->free_chunks[mclass]);
	iomap_chunk_1 = dlist_container(IOMapBufferChunk, free_chain, dnode);
	iomap_chunk_2 = iomap_chunk_1 + offset;
	Assert(iomap_chunk_2->mclass == 0);

	iomap_chunk_1->mclass = mclass - 1;
	iomap_chunk_2->mclass = mclass - 1;

	dlist_push_tail(&iomap_seg->free_chunks[mclass - 1],
					&iomap_chunk_1->free_chain);
	dlist_push_tail(&iomap_seg->free_chunks[mclass - 1],
					&iomap_chunk_2->free_chain);
	return true;
}

/*
 * gpuDmaMemAllocIOMap
 *
 * Allocation of device memory which is mapped to I/O address space
 */
CUresult
gpuDmaMemAllocIOMap(GpuContext_v2 *gcontext,
					CUdeviceptr *p_devptr, size_t bytesize)
{
	IOMapBufferHead	   *iomap_head;
	IOMapBufferSegment *iomap_seg;
	IOMapBufferChunk   *iomap_chunk;
	int					mclass;
	dlist_node		   *dnode;
	CUresult			rc;

	Assert(IsGpuServerProcess());

	if (!iomap_buffer_heads)
		return CUDA_ERROR_OUT_OF_MEMORY;
	iomap_head = &iomap_buffer_heads[gpuserv_cuda_index];
	iomap_seg = iomap_head->iomap_seg;

	if (!iomap_buffer_base)
	{
		rc = cuIpcOpenMemHandle(&iomap_buffer_base,
								iomap_head->cuda_mhandle,
								CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuIpcOpenMemHandle: %s", errorText(rc));
	}

	/*
	 * Do allocation
	 */
	mclass = get_next_log2(bytesize);
	if (mclass < IOMAPBUF_CHUNKSZ_MIN_BIT)
		mclass = IOMAPBUF_CHUNKSZ_MIN_BIT;
	else if (mclass > IOMAPBUF_CHUNKSZ_MAX_BIT)
		return CUDA_ERROR_OUT_OF_MEMORY;

	SpinLockAcquire(&iomap_seg->lock);
	if (dlist_is_empty(&iomap_seg->free_chunks[mclass]))
	{
		/* split larger mclass */
		if (!gpuDmaMemSplitIOMap(iomap_seg, mclass + 1))
		{
			SpinLockRelease(&iomap_seg->lock);
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
	}
	Assert(!dlist_is_empty(&iomap_seg->free_chunks[mclass]));

	dnode = dlist_pop_head_node(&iomap_seg->free_chunks[mclass]);
	iomap_chunk = dlist_container(IOMapBufferChunk, free_chain, dnode);
	Assert(iomap_chunk->mclass == mclass);

	memset(&iomap_chunk->free_chain, 0, sizeof(dlist_node));
	SpinLockRelease(&iomap_seg->lock);

	*p_devptr = iomap_buffer_base + ((Size)index << IOMAPBUF_CHUNKSZ_MIN_BIT);

	return CUDA_SUCCESS;
}

/*
 * gpuDmaMemFreeIOMap
 *
 * Release of device memory which is mapped to I/O address space
 */
CUresult
gpuDmaMemFreeIOMap(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	IOMapBufferHead	   *iomap_head;
	IOMapBufferSegment *iomap_seg;
	IOMapBufferChunk   *iomap_chunk;
	IOMapBufferChunk   *iomap_buddy;
	int					index;
	int					shift;

	Assert(IsGpuServerProcess());
	if (!iomap_buffer_base)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (devptr < iomap_buffer_base ||
		devptr > iomap_buffer_base + iomap_buffer_size)
		return CUDA_ERROR_INVALID_VALUE;

	Assert((devptr & (IOMAPBUF_CHUNKSZ_MIN - 1)) == 0);
	iomap_head = &iomap_buffer_heads[gpuserv_cuda_index];
	iomap_seg = iomap_head->iomap_seg;

	SpinLockAcquire(&iomap_seg->lock);
	index = (devptr - iomap_buffer_base) >> IOMAPBUF_CHUNKSZ_MIN_BIT;
	iomap_chunk = &iomap_seg->iomap_chunks[index];
	Assert(!iomap_chunk->free_chain.prev &&
		   !iomap_chunk->free_chain.next);

	/*
	 * Try to merge with the neighbor chunks
	 */
	while (iomap_chunk->mclass < IOMAPBUF_CHUNKSZ_MAX_BIT)
	{
		index = iomap_chunk - iomap_seg->iomap_chunks;
		shift = 1UL << (iomap_chunk->mclass - IOMAPBUF_CHUNKSZ_MIN_BIT);
		Assert((index & (shift - 1)) == 0);
		if ((index & shift) == 0)
		{
			/* try to merge with next */
			iomap_buddy = &iomap_seg->iomap_chunks[index + shift];
			if (iomap_buddy->free_chain.prev &&
				iomap_buddy->free_chain.next &&
				iomap_buddy->mclass == iomap_chunk->mclass)
			{
				/* OK, let's merge */
				dlist_delete(&iomap_buddy->free_chain);
				memset(iomap_buddy, 0, sizeof(IOMapBufferChunk));
				iomap_chunk->mclass++;
			}
			else
				break;	/* give up to merge chunks any more */
		}
		else
		{
			/* try to merge with prev */
			iomap_buddy = &iomap_seg->iomap_chunks[index - shift];
			if (iomap_buddy->free_chain.prev &&
				iomap_buddy->free_chain.next &&
				iomap_buddy->mclass == iomap_chunk->mclass)
			{
				/* OK, let's merge */
				dlist_delete(&iomap_buddy->free_chain);
				memset(&iomap_buddy->free_chain, 0, sizeof(dlist_node));
				memset(iomap_chunk, 0, sizeof(IOMapBufferChunk));
				iomap_buddy->mclass++;
				iomap_chunk = iomap_buddy;
			}
			else
				break;	/* give up to merge chunks any more */
		}
	}
	/* back to the free list again */
	Assert(iomap_chunk->mclass >= IOMAPBUF_CHUNKSZ_MIN_BIT &&
		   iomap_chunk->mclass <= IOMAPBUF_CHUNKSZ_MAX_BIT);
	dlist_push_head(&iomap_seg->free_chunks[iomap_chunk->mclass],
					&iomap_chunk->free_chain);
	SpinLockRelease(&iomap_seg->lock);

	return CUDA_SUCCESS;
}

/*
 * pgstrom_iomap_buffer_info
 */
typedef struct
{
	cl_uint			nitems;
	struct {
		cl_int		gpuid;
		cl_bool		is_used;
		cl_ulong	offset;
		cl_ulong	length;
	} chunks[FLEXIBLE_ARRAY_MEMBER];
} iomap_buffer_info;

static void
setup_iomap_buffer_info(IOMapBufferSegment *iomap_seg,
						int gpuid,
						iomap_buffer_info *iomap_info)
{
	int		limit = iomap_buffer_size / IOMAPBUF_CHUNKSZ_MIN;
	int		index = 0;

	SpinLockAcquire(&iomap_seg->lock);
	while (index < limit)
	{
		IOMapBufferChunk *iomap_chunk = &iomap_seg->iomap_chunks[index];
		int		j = iomap_info->nitems++;

		iomap_info->chunks[j].gpuid = gpuid;
		iomap_info->chunks[j].is_used = (!iomap_chunk->free_chain.prev &&
										 !iomap_chunk->free_chain.next);
		iomap_info->chunks[j].offset = index * IOMAPBUF_CHUNKSZ_MIN;
		iomap_info->chunks[j].length = (1UL << iomap_chunk->mclass);

		index += 1UL << (iomap_chunk->mclass - IOMAPBUF_CHUNKSZ_MIN_BIT);
	}
	SpinLockRelease(&iomap_seg->lock);
}

Datum
pgstrom_iomap_buffer_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	*fncxt;
	iomap_buffer_info *iomap_info;
	Datum		values[4];
	bool		isnull[4];
	int			i;
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		int				max_nchunks;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(4, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "gpuid",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "offset",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "length",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "state",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		max_nchunks = iomap_buffer_size / IOMAPBUF_CHUNKSZ_MIN;
		iomap_info = palloc(offsetof(iomap_buffer_info,
									 chunks[max_nchunks * numDevAttrs]));
		iomap_info->nitems = 0;
		if (iomap_buffer_heads)
		{
			for (i=0; i < numDevAttrs; i++)
			{
				IOMapBufferSegment *iomap_seg
					= iomap_buffer_heads[i].iomap_seg;
				cl_int		gpuid = devAttrs[i].DEV_ID;

				setup_iomap_buffer_info(iomap_seg, gpuid, iomap_info);
			}
		}
		fncxt->user_fctx = iomap_info;
		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();
	iomap_info = fncxt->user_fctx;

	if (fncxt->call_cntr >= iomap_info->nitems)
		SRF_RETURN_DONE(fncxt);

	i = fncxt->call_cntr;
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(iomap_info->chunks[i].gpuid);
	values[1] = Int64GetDatum(iomap_info->chunks[i].offset);
	values[2] = Int64GetDatum(iomap_info->chunks[i].length);
	values[3] = PointerGetDatum(cstring_to_text(iomap_info->chunks[i].is_used
												? "used"
												: "free"));
	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_iomap_buffer_info);

/*
 * pgstrom_startup_nvme_strom
 */
static void
pgstrom_startup_nvme_strom(void)
{
	Size		max_nchunks;
	Size		required;
	char	   *pos;
	bool		found;
	int			i, j;
	struct stat	stbuf;
	CUresult	rc;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	/* is nvme-strom driver installed? */
	if (stat(nvme_strom_ioctl_pathname, &stbuf) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("failed on stat(2) for %s: %m",
						nvme_strom_ioctl_pathname),
				 errhint("nvme-strom.ko may not be installed on the system")));

	/* allocation of static shared memory */
	iomap_buffer_heads = malloc(sizeof(IOMapBufferHead) * numDevAttrs);
	if (!iomap_buffer_heads)
		elog(ERROR, "out of memory");

	max_nchunks = iomap_buffer_size / IOMAPBUF_CHUNKSZ_MIN;
	required = offsetof(IOMapBufferSegment, iomap_chunks[max_nchunks]);
	pos = ShmemInitStruct("iomap_buffer_heads",
						  MAXALIGN(required) * numDevAttrs,
						  &found);
	Assert(!found);
	memset(pos, 0, MAXALIGN(required) * numDevAttrs);

	/* setup i/o mapped device memory for each GPU device */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", errorText(rc));

	for (i=0; i < numDevAttrs; i++)
	{
		IOMapBufferHead	   *iomap_head = &iomap_buffer_heads[i];
		IOMapBufferSegment *iomap_seg = (IOMapBufferSegment *)pos;
		CUdevice			cuda_device;
		CUcontext			cuda_context;
		CUdeviceptr			cuda_devptr;
		CUipcMemHandle		cuda_mhandle;
		Size				remain;
		int					mclass;
		StromCmd__MapGpuMemory cmd;

		/* setup i/o mapped device memory for each GPU device */
		rc = cuDeviceGet(&cuda_device, devAttrs[i].DEV_ID);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", errorText(rc));

		rc = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));

		rc = cuMemAlloc(&cuda_devptr, iomap_buffer_size);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemAlloc: %s", errorText(rc));

		rc = cuIpcGetMemHandle(&cuda_mhandle, cuda_devptr);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuIpcGetMemHandle: %s", errorText(rc));

		memset(&cmd, 0, sizeof(StromCmd__MapGpuMemory));
		cmd.vaddress = cuda_devptr;
		cmd.length = iomap_buffer_size;
		if (nvme_strom_ioctl(STROM_IOCTL__MAP_GPU_MEMORY, &cmd) != 0)
			elog(ERROR, "STROM_IOCTL__MAP_GPU_MEMORY failed: %m");
		if (iomap_buffer_size % cmd.gpu_page_sz != 0)
			elog(WARNING, "i/o mapped GPU memory size (%zu) is not aligned to GPU page size(%u)",
				 iomap_buffer_size, cmd.gpu_page_sz);

		/* IOMapBufferSegment */
		SpinLockInit(&iomap_seg->lock);
		dlist_init(&iomap_seg->addr_chunks);
		for (j=0; j <= IOMAPBUF_CHUNKSZ_MAX_BIT; j++)
			dlist_init(&iomap_seg->free_chunks[j]);

		j = 0;
		mclass = IOMAPBUF_CHUNKSZ_MAX_BIT;
		remain = iomap_buffer_size;
		while (remain >= IOMAPBUF_CHUNKSZ_MIN &&
			   mclass >= IOMAPBUF_CHUNKSZ_MIN_BIT)
		{
			IOMapBufferChunk   *iomap_chunk = &iomap_seg->iomap_chunks[j];
			Size				chunk_sz = (1UL << mclass);

			if (remain < chunk_sz)
				mclass--;
			else
			{
				iomap_chunk->mclass = mclass;
				dlist_push_tail(&iomap_seg->free_chunks[mclass],
								&iomap_chunk->free_chain);

				remain -= chunk_sz;
				j += (chunk_sz >> IOMAPBUF_CHUNKSZ_MIN_BIT);
			}
		}

		/* IOMapBufferHead */
		iomap_head->iomap_seg		= iomap_seg;
		iomap_head->iomap_handle	= cmd.handle;
		iomap_head->gpu_page_sz		= cmd.gpu_page_sz;
		iomap_head->gpu_npages		= cmd.gpu_npages;
		iomap_head->cuda_device		= cuda_device;
		iomap_head->cuda_context	= cuda_context;
		iomap_head->cuda_devptr		= cuda_devptr;
		memcpy(&iomap_head->cuda_mhandle,
			   &cuda_mhandle,
			   sizeof(CUipcMemHandle));

		pos += MAXALIGN(required);
	}
}

/*
 * pgstrom_init_nvme_strom
 */
void
pgstrom_init_nvme_strom(void)
{
	static int	__iomap_buffer_size;
	Size		max_nchunks;
	Size		required;

	DefineCustomIntVariable("pg_strom.iomap_buffer_size",
							"I/O mapped buffer size for SSD-to-GPU Direct DMA",
							NULL,
							&__iomap_buffer_size,
							0,
							0,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	iomap_buffer_size = (Size)__iomap_buffer_size << 10;
	if (iomap_buffer_size % IOMAPBUF_CHUNKSZ_MIN != 0)
		elog(ERROR, "pg_strom.iomap_buffer_size is not aligned to 4KB");

	/*
	 * i/o mapped device memory shall be set up
	 * only when pg_strom.iomap_device_memory_size > 0.
	 */
	if (iomap_buffer_size > 0)
	{
		max_nchunks = iomap_buffer_size / IOMAPBUF_CHUNKSZ_MIN;
		required = offsetof(IOMapBufferSegment, iomap_chunks[max_nchunks]);
		RequestAddinShmemSpace(MAXALIGN(required) * numDevAttrs);
		shmem_startup_next = shmem_startup_hook;
		shmem_startup_hook = pgstrom_startup_nvme_strom;
	}
}
