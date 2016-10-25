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
#include "commands/tablespace.h"
#include "funcapi.h"
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/inval.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "pg_strom.h"
#include "nvme_strom.h"
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
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
	CUipcMemHandle		cuda_mhandle;
	unsigned long		iomap_handle;
	uint32_t			gpu_page_sz;
	uint32_t			gpu_npages;
	/* fields below are protected by the lock */
	slock_t				lock;
	dlist_head			free_chunks[IOMAPBUF_CHUNKSZ_MAX_BIT + 1];
	IOMapBufferChunk	iomap_chunks[FLEXIBLE_ARRAY_MEMBER];
} IOMapBufferSegment;

typedef struct
{
	int					num_devices;
	struct {
		CUdevice		cuda_device;
		CUcontext		cuda_context;
		CUdeviceptr		cuda_devptr;
		CUipcMemHandle	cuda_mhandle;		
	} dev[FLEXIBLE_ARRAY_MEMBER];
} IOMapCUDAResources;

static shmem_startup_hook_type shmem_startup_next = NULL;
static const char	   *nvme_strom_ioctl_pathname = "/proc/nvme-strom";
static void			   *iomap_buffer_segments = NULL;
static CUdeviceptr		iomap_buffer_base = 0UL;	/* per process vaddr */
static Size				iomap_buffer_size;			/* GUC */
static HTAB			   *vfs_nvme_htable = NULL;
static Oid				nvme_last_tablespace_oid = InvalidOid;
static bool				nvme_last_tablespace_supported;
static IOMapCUDAResources *iomap_cuda_resources = NULL;

#define SizeOfIOMapBufferSegment								\
	MAXALIGN(offsetof(IOMapBufferSegment,						\
					  iomap_chunks[iomap_buffer_size >>			\
								   IOMAPBUF_CHUNKSZ_MIN_BIT]))
#define GetIOMapBufferSegment(dindex)							\
	((IOMapBufferSegment *)((char *)iomap_buffer_segments +		\
							SizeOfIOMapBufferSegment * (dindex)))
/*
 * gpuMemSizeIOMap - returns configured size of the i/o mapped device memory;
 * never guaranteed it is actually allocated and mapped.
 */
Size
gpuMemSizeIOMap(void)
{
	return iomap_buffer_size;
}

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
		if (fdesc_nvme_strom < 0)
			elog(ERROR, "failed to open %s: %m", nvme_strom_ioctl_pathname);
	}
	return ioctl(fdesc_nvme_strom, cmd, arg);
}

/*
 * gpuMemSplitIOMap
 */
static bool
gpuMemSplitIOMap(IOMapBufferSegment *iomap_seg, int mclass)
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
		if (!gpuMemSplitIOMap(iomap_seg, mclass + 1))
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
 * gpuMemAllocIOMap
 *
 * Allocation of device memory which is mapped to I/O address space
 */
CUresult
gpuMemAllocIOMap(GpuContext_v2 *gcontext,
				 CUdeviceptr *p_devptr, size_t bytesize)
{
	IOMapBufferSegment *iomap_seg;
	IOMapBufferChunk   *iomap_chunk;
	int					mclass;
	int					index;
	dlist_node		   *dnode;
	CUdeviceptr			devptr;
	CUresult			rc;

	Assert(IsGpuServerProcess());

	if (!iomap_buffer_segments)
		return CUDA_ERROR_OUT_OF_MEMORY;
	iomap_seg = GetIOMapBufferSegment(gpuserv_cuda_dindex);

	if (!iomap_buffer_base)
	{
		rc = cuIpcOpenMemHandle(&iomap_buffer_base,
								iomap_seg->cuda_mhandle,
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
		if (!gpuMemSplitIOMap(iomap_seg, mclass + 1))
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

	Assert(iomap_chunk >= iomap_seg->iomap_chunks);
	index = iomap_chunk - iomap_seg->iomap_chunks;
	Assert(index < iomap_buffer_size >> IOMAPBUF_CHUNKSZ_MIN_BIT);

	devptr = iomap_buffer_base + ((Size)index << IOMAPBUF_CHUNKSZ_MIN_BIT);
	trackIOMapMem(gcontext, devptr);

	*p_devptr = devptr;

	return CUDA_SUCCESS;
}

/*
 * gpuMemFreeIOMap
 *
 * Release of device memory which is mapped to I/O address space
 */
CUresult
gpuMemFreeIOMap(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	IOMapBufferSegment *iomap_seg;
	IOMapBufferChunk   *iomap_chunk;
	IOMapBufferChunk   *iomap_buddy;
	int					index;
	int					shift;

	if (gcontext)
		untrackIOMapMem(gcontext, devptr);

	Assert(IsGpuServerProcess());
	if (!iomap_buffer_base)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (devptr < iomap_buffer_base ||
		devptr > iomap_buffer_base + iomap_buffer_size)
		return CUDA_ERROR_INVALID_VALUE;

	iomap_seg = GetIOMapBufferSegment(gpuserv_cuda_dindex);

	SpinLockAcquire(&iomap_seg->lock);
	Assert((devptr & (IOMAPBUF_CHUNKSZ_MIN - 1)) == 0);
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
 * gpuMemCopyFromSSDWait - callback to wait for SSD-to-GPU Direct DMA done
 */
static void
__MemCopyFromSSDWaitCallback(CUstream cuda_stream,
							 CUresult status, void *private)
{
	StromCmd__MemCpySsdToGpuWait cmd;
	GpuTask_v2	   *gtask = (GpuTask_v2 *) private;

	memset(&cmd, 0, sizeof(StromCmd__MemCpySsdToGpuWait));
	cmd.dma_task_id = gtask->dma_task_id;

	if (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_SSD2GPU_WAIT, &cmd) != 0)
	{
		fprintf(stderr, "failed on STROM_IOCTL__MEMCPY_SSD2GPU_WAIT: %m");
		if (gtask->kerror.errcode == 0)
		{
			gtask->kerror.errcode = StromError_Ssd2GpuDirectDma;
			gtask->kerror.kernel = StromKernel_NVMeStrom;
			gtask->kerror.lineno = 0;
		}
	}
	else if (cmd.status != 0)
	{
		fprintf(stderr, "NVMe-Strom: Direct DMA Status=0x%lx", cmd.status);
		if (gtask->kerror.errcode == 0)
		{
			gtask->kerror.errcode = StromError_Ssd2GpuDirectDma;
			gtask->kerror.kernel = StromKernel_NVMeStrom;
			gtask->kerror.lineno = 0;
		}
	}
	// TODO: we may need to handle signal interrupt
	//       however, it is worker thread context. How to do?

	/* no concurrent SSD2GPU P2P DMA */
	gtask->dma_task_id = 0;
}

void
gpuMemCopyFromSSDWait(GpuTask_v2 *gtask, CUstream cuda_stream)
{
	CUresult	rc;

	Assert(gtask->dma_task_id != 0UL);
	rc = cuStreamAddCallback(cuda_stream,
							 __MemCopyFromSSDWaitCallback,
							 gtask,
							 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamAddCallback: %s", errorText(rc));
}

/*
 * gpuMemCopyFromSSDAsync - kick SSD-to-GPU Direct DMA in asynchronous mode
 */
void
gpuMemCopyFromSSDAsync(GpuTask_v2 *gtask,
					   CUdeviceptr m_kds,
					   pgstrom_data_store *pds,
					   CUstream cuda_stream)
{
	StromCmd__MemCpySsdToGpuWriteBack *cmd;
	IOMapBufferSegment *iomap_seg;
	BlockNumber	   *block_nums;
	void		   *block_data;
	loff_t		   *file_pos;
	size_t			offset;
	size_t			length;
	cl_uint			nr_loaded;
	CUresult		rc;

	Assert(IsGpuServerProcess());
	if (!iomap_buffer_segments)
		elog(ERROR, "NVMe-Strom is not configured");
	iomap_seg = GetIOMapBufferSegment(gpuserv_cuda_dindex);

	/* Device memory should be already imported on allocation time */
	Assert(iomap_buffer_base != 0UL);
	/* Right now, multiple NVMe-Strom submit per task is not supported */
	Assert(gtask->dma_task_id == 0UL);
	/* PDS/KDS format check */
	Assert(pds->kds.format == KDS_FORMAT_BLOCK);

	if (m_kds < iomap_buffer_base ||
		m_kds + pds->kds.length > iomap_buffer_base + iomap_buffer_size)
		elog(ERROR, "NVMe-Strom: P2P DMA destination out of range");
	offset = m_kds - iomap_buffer_base;

	/* nothing special if all the blocks are already loaded */
	if (pds->nblocks_uncached == 0)
	{
		rc = cuMemcpyHtoDAsync(m_kds,
							   &pds->kds,
							   pds->kds.length,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		// TODO: update perfmon
		// gscan->bytes_dma_send += pds->kds.length;
		// gscan->num_dma_send++;

		return;
	}
	Assert(pds->nblocks_uncached <= pds->kds.nitems);
	nr_loaded = pds->kds.nitems - pds->nblocks_uncached;
	length = ((char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds, nr_loaded) -
			  (char *)(&pds->kds));
	offset += length;

	/* userspace pointers */
	block_nums = (BlockNumber *)KERN_DATA_STORE_BODY(&pds->kds) + nr_loaded;
	block_data = KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds, nr_loaded);
	file_pos = ((loff_t *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds,
													   pds->kds.nrooms) -
				pds->nblocks_uncached);

	/* setup ioctl(2) command */
	cmd = palloc(offsetof(StromCmd__MemCpySsdToGpuWriteBack,
						  file_pos[pds->nblocks_uncached]));
	memset(cmd, 0, offsetof(StromCmd__MemCpySsdToGpuWriteBack, handle));
	cmd->handle		= iomap_seg->iomap_handle;
	cmd->offset		= offset;
	cmd->block_size	= BLCKSZ;
	cmd->block_nums	= block_nums;	/* array of BlockNumber */
	cmd->block_data	= block_data;	/* buffer of uncached area */
	cmd->file_desc	= gtask->peer_fdesc;
	cmd->nchunks	= pds->nblocks_uncached;
	memcpy(cmd->file_pos, file_pos, sizeof(loff_t) * pds->nblocks_uncached);

	/* (1) kick SSD2GPU P2P DMA */
	if (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_SSD2GPU_WRITEBACK, cmd) != 0)
		elog(ERROR, "failed on STROM_IOCTL__MEMCPY_SSD2GPU_WRITEBACK: %m");
	gtask->dma_task_id  = cmd->dma_task_id;

	// TODO: update perfmon

	/* (2) kick RAM2GPU DMA (earlier half) */
	rc = cuMemcpyHtoDAsync(m_kds,
						   &pds->kds,
						   length,
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));

	/* (3) kick RAM2GPU DMA (later half; if any) */
	if (cmd->nr_ram2gpu > 0)
	{
		length = BLCKSZ * cmd->nr_ram2gpu;
		offset = ((char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds,
													   pds->kds.nitems) -
				  (char *)&pds->kds) - length;
		rc = cuMemcpyHtoDAsync(m_kds + offset,
							   (char *)&pds->kds + offset,
							   length,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	}
	pfree(cmd);
}

/*
 * RelationCanUseNvmeStrom
 */
typedef struct
{
	Oid		tablespace_oid;
	bool	nvme_strom_supported;
} vfs_nvme_status;

static void
vfs_nvme_cache_callback(Datum arg, int cacheid, uint32 hashvalue)
{
	/* invalidate all the cached status */
	if (vfs_nvme_htable)
	{
		hash_destroy(vfs_nvme_htable);
		vfs_nvme_htable = NULL;
		nvme_last_tablespace_oid = InvalidOid;
	}
}

static bool
__RelationCanUseNvmeStrom(Relation relation)
{
	vfs_nvme_status *entry;
	const char	   *pathname;
	Oid				tablespace_oid;
	int				fdesc;
	bool			found;

	if (iomap_buffer_size == 0)
		return false;	/* NVMe-Strom is not enabled */

	if (RelationUsesLocalBuffers(relation))
		return false;	/* SSD2GPU on temp relation is not supported */

	tablespace_oid = RelationGetForm(relation)->reltablespace;
	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	if (OidIsValid(nvme_last_tablespace_oid) &&
		nvme_last_tablespace_oid == tablespace_oid)
		return nvme_last_tablespace_supported;

	if (!vfs_nvme_htable)
	{
		HASHCTL		ctl;

		memset(&ctl, 0, sizeof(HASHCTL));
		ctl.keysize = sizeof(Oid);
		ctl.entrysize = sizeof(vfs_nvme_status);
		vfs_nvme_htable = hash_create("VFS:NVMe-Strom status", 64,
									  &ctl, HASH_ELEM | HASH_BLOBS);
		CacheRegisterSyscacheCallback(TABLESPACEOID,
									  vfs_nvme_cache_callback, (Datum) 0);
	}
	entry = (vfs_nvme_status *) hash_search(vfs_nvme_htable,
											&tablespace_oid,
											HASH_ENTER,
											&found);
	if (found)
	{
		nvme_last_tablespace_oid = tablespace_oid;
		nvme_last_tablespace_supported = entry->nvme_strom_supported;
		return entry->nvme_strom_supported;
	}

	/* check whether the tablespace is supported */
	entry->tablespace_oid = tablespace_oid;
	entry->nvme_strom_supported = false;

	pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);
	fdesc = open(pathname, O_RDONLY | O_DIRECTORY);
	elog(INFO, "pathname = %s fdesc = %d", pathname, fdesc);
	if (fdesc < 0)
	{
		elog(WARNING, "failed to open \"%s\" of tablespace \"%s\": %m",
			 pathname, get_tablespace_name(tablespace_oid));
	}
	else
	{
		StromCmd__CheckFile cmd;

		cmd.fdesc = fdesc;
		if (nvme_strom_ioctl(STROM_IOCTL__CHECK_FILE, &cmd) == 0)
			entry->nvme_strom_supported = true;
		else
		{
			ereport(NOTICE,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("nvme_strom does not support tablespace \"%s\"",
							get_tablespace_name(tablespace_oid))));
		}
	}
	nvme_last_tablespace_oid = tablespace_oid;
	nvme_last_tablespace_supported = entry->nvme_strom_supported;
	return entry->nvme_strom_supported;
}

bool
RelationCanUseNvmeStrom(Relation relation)
{
	bool	retval;

	PG_TRY();
	{
		retval = __RelationCanUseNvmeStrom(relation);
	}
	PG_CATCH();
	{
		/* clean up the cache if any error */
		vfs_nvme_cache_callback(0, 0, 0);
        PG_RE_THROW();
	}
	PG_END_TRY();

	return retval;
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

#if 1
void
dump_iomap_buffer_info(void)
{
	iomap_buffer_info *iomap_info;
	int			i, max_nchunks;

	max_nchunks = iomap_buffer_size / IOMAPBUF_CHUNKSZ_MIN;
	iomap_info = palloc(offsetof(iomap_buffer_info,
								 chunks[max_nchunks * numDevAttrs]));
	iomap_info->nitems = 0;
	if (iomap_buffer_segments)
	{
		for (i=0; i < numDevAttrs; i++)
		{
			IOMapBufferSegment *iomap_seg = GetIOMapBufferSegment(i);
			cl_int      gpuid = devAttrs[i].DEV_ID;

			setup_iomap_buffer_info(iomap_seg, gpuid, iomap_info);
		}

		if (iomap_info->nitems > 0)
			fputc('\n', stderr);
		for (i=0; i < iomap_info->nitems; i++)
		{
			fprintf(stderr, "GPU%d 0x%p - 0x%p len=%zu %s\n",
					iomap_info->chunks[i].gpuid,
					(char *)(iomap_info->chunks[i].offset),
					(char *)(iomap_info->chunks[i].offset +
							 iomap_info->chunks[i].length),
					(size_t)(iomap_info->chunks[i].length),
					iomap_info->chunks[i].is_used ? "used" : "free");
		}
	}
	pfree(iomap_info);
}
#endif

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
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "state",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		max_nchunks = iomap_buffer_size / IOMAPBUF_CHUNKSZ_MIN;
		iomap_info = palloc(offsetof(iomap_buffer_info,
									 chunks[max_nchunks * numDevAttrs]));
		iomap_info->nitems = 0;
		if (iomap_buffer_segments)
		{
			for (i=0; i < numDevAttrs; i++)
			{
				IOMapBufferSegment *iomap_seg = GetIOMapBufferSegment(i);
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

	/*
	 * setup of the stable CUDA resource
	 *
	 * NOTE: PostmasterStateMachine() may call the shared memory startup
	 * code again, but not exit of the postmaster process, if and when
	 * child processes are terminated abnormally.
	 * In this case, we have to ensure the previous CUDA context and
	 * relevant GPU resources are destroyed, to avoid memory leaks by
	 * the orphan CUDA context.
	 * (We may be able to reuse the CUDA context, however, child processes
	 * were crashed by some error condition, so we like to choose a safer
	 * option here.)
	 */
	if (!iomap_cuda_resources)
	{
		iomap_cuda_resources = calloc(1, offsetof(IOMapCUDAResources,
												  dev[numDevAttrs]));
		if (!iomap_cuda_resources)
			elog(ERROR, "out of memory");

		rc = cuInit(0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuInit: %s", errorText(rc));
	}
	else
	{
		for (i=0; i < iomap_cuda_resources->num_devices; i++)
		{
			if (!iomap_cuda_resources->dev[i].cuda_context)
				continue;
			rc = cuCtxDestroy(iomap_cuda_resources->dev[i].cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxDestroy: %s", errorText(rc));
		}
		iomap_cuda_resources = realloc(iomap_cuda_resources,
									   offsetof(IOMapCUDAResources,
												dev[numDevAttrs]));
		memset(iomap_cuda_resources, 0, offsetof(IOMapCUDAResources,
												 dev[numDevAttrs]));
	}
	iomap_cuda_resources->num_devices = numDevAttrs;

	/* allocation of static shared memory */
	required = SizeOfIOMapBufferSegment * numDevAttrs,
	iomap_buffer_segments = ShmemInitStruct("iomap_buffer_segments",
											required,
											&found);
	if (found)
		elog(FATAL, "Bug? \"iomap_buffer_segments\" is already initialized");
	memset(iomap_buffer_segments, 0, required);

	pos = iomap_buffer_segments;
	for (i=0; i < numDevAttrs; i++)
	{
		IOMapBufferSegment *iomap_seg = GetIOMapBufferSegment(i);
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

		/* save resources on iomap_cuda_resources */
		iomap_cuda_resources->dev[i].cuda_device = cuda_device;
		iomap_cuda_resources->dev[i].cuda_context = cuda_context;
		iomap_cuda_resources->dev[i].cuda_devptr = cuda_devptr;
		iomap_cuda_resources->dev[i].cuda_mhandle = cuda_mhandle;

		/* setup IOMapBufferSegment */
		elog(LOG, "NVMe-Strom: GPU Device Memory (%p-%p; %zuMB) is mapped",
			 (char *)cuda_devptr,
			 (char *)cuda_devptr + iomap_buffer_size - 1,
			 (size_t)(iomap_buffer_size >> 20));
		memcpy(&iomap_seg->cuda_mhandle, &cuda_mhandle,
			   sizeof(CUipcMemHandle));
		iomap_seg->iomap_handle = cmd.handle;
		iomap_seg->gpu_page_sz = cmd.gpu_page_sz;
		iomap_seg->gpu_npages = cmd.gpu_npages;
		SpinLockInit(&iomap_seg->lock);
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
		pos += SizeOfIOMapBufferSegment;
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

	/* pg_strom.iomap_buffer_size */
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
