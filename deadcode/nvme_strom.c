/*
 * nvme_strom.c
 *
 * Routines to support optional SSD-to-GPU Direct DMA Loading
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
#include "commands/tablespace.h"
#include "funcapi.h"
#include "storage/ipc.h"
#include "storage/bufmgr.h"
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

static shmem_startup_hook_type shmem_startup_next = NULL;
static const char	   *nvme_strom_ioctl_pathname = "/proc/nvme-strom";
static void			   *iomap_buffer_segments = NULL;
static CUdeviceptr		iomap_buffer_base = 0UL;	/* per process vaddr */
static Size				iomap_buffer_size;			/* GUC */
static HTAB			   *vfs_nvme_htable = NULL;
static Oid				nvme_last_tablespace_oid = InvalidOid;
static bool				nvme_last_tablespace_supported;
static bool				debug_force_nvme_strom = false;	/* GUC */
static bool				nvme_strom_enabled = true;		/* GUC */
static long				sysconf_pagesize;		/* _SC_PAGESIZE */
static long				sysconf_phys_pages;		/* _SC_PHYS_PAGES */
static long				nvme_strom_threshold;

#define SizeOfIOMapBufferSegment								\
	MAXALIGN(offsetof(IOMapBufferSegment,						\
					  iomap_chunks[iomap_buffer_size >>			\
								   IOMAPBUF_CHUNKSZ_MIN_BIT]))
#define GetIOMapBufferSegment(dindex)							\
	((IOMapBufferSegment *)((char *)iomap_buffer_segments +		\
							SizeOfIOMapBufferSegment * (dindex)))
#if 0
/*
 * gpuMemSizeIOMap - returns configured size of the i/o mapped device memory;
 * never guaranteed it is actually allocated and mapped.
 */
Size
gpuMemSizeIOMap(void)
{
	return iomap_buffer_size;
}
#endif

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

#if 0
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
gpuMemAllocIOMap(GpuContext *gcontext,
				 CUdeviceptr *p_devptr, size_t bytesize)
{
	IOMapBufferSegment *iomap_seg;
	IOMapBufferChunk   *iomap_chunk;
	int					mclass;
	int					index;
	dlist_node		   *dnode;
	CUdeviceptr			devptr;
	CUresult			rc;
	static pthread_mutex_t iomap_buffer_mutex = PTHREAD_MUTEX_INITIALIZER;

	Assert(IsGpuServerProcess());

	if (!iomap_buffer_segments)
		return CUDA_ERROR_OUT_OF_MEMORY;
	/* ensure the i/o mapped buffer is already available */
	iomap_seg = GetIOMapBufferSegment(gpuserv_cuda_dindex);
	pg_memory_barrier();
	if (!iomap_seg->iomap_handle)
		return CUDA_ERROR_OUT_OF_MEMORY;

	pthreadMutexLock(&iomap_buffer_mutex);
	if (!iomap_buffer_base)
	{
		rc = cuIpcOpenMemHandle(&iomap_buffer_base,
								iomap_seg->cuda_mhandle,
								CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
		{
			pthreadMutexUnlock(&iomap_buffer_mutex);
			werror("failed on cuIpcOpenMemHandle: %s", errorText(rc));
		}
	}
	pthreadMutexUnlock(&iomap_buffer_mutex);

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

	Assert(devptr >= iomap_buffer_base &&
		   devptr + bytesize <=  iomap_buffer_base + iomap_buffer_size);

	*p_devptr = devptr;

	return CUDA_SUCCESS;
}

/*
 * gpuMemFreeIOMap
 *
 * Release of device memory which is mapped to I/O address space
 */
CUresult
gpuMemFreeIOMap(GpuContext *gcontext, CUdeviceptr devptr)
{
	IOMapBufferSegment *iomap_seg;
	IOMapBufferChunk   *iomap_chunk;
	IOMapBufferChunk   *iomap_buddy;
	int					index;
	int					shift;

	/* If called on PostgreSQL backend, send a request to release */
	if (!IsGpuServerProcess())
	{
		gpuservSendIOMapMemFree(gcontext, devptr);
		return CUDA_SUCCESS;
	}

	if (gcontext)
		untrackIOMapMem(gcontext, devptr);

	if (!iomap_buffer_base)
		return CUDA_ERROR_NOT_INITIALIZED;
	/* ensure the i/o mapped buffer is already available */
	iomap_seg = GetIOMapBufferSegment(gpuserv_cuda_dindex);
	pg_memory_barrier();
	if (!iomap_seg->iomap_handle)
		return CUDA_ERROR_NOT_INITIALIZED;

	if (devptr < iomap_buffer_base ||
		devptr > iomap_buffer_base + iomap_buffer_size)
		return CUDA_ERROR_INVALID_VALUE;

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
#endif

/*
 * gpuMemCopyFromSSDWaitRaw
 */
static void
gpuMemCopyFromSSDWaitRaw(unsigned long dma_task_id)
{
	StromCmd__MemCopyWait cmd;

	memset(&cmd, 0, sizeof(StromCmd__MemCopyWait));
	cmd.dma_task_id = dma_task_id;

	if (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_WAIT, &cmd) != 0)
		werror("failed on nvme_strom_ioctl(STROM_IOCTL__MEMCPY_WAIT): %m");
}

/*
 * gpuMemCopyFromSSD - kick SSD-to-GPU Direct DMA, then wait for completion
 */
void
gpuMemCopyFromSSD(GpuTask *gtask,
				  CUdeviceptr m_kds,
				  pgstrom_data_store *pds)
{
	StromCmd__MemCopySsdToGpu cmd;
	IOMapBufferSegment *iomap_seg;
	BlockNumber	   *block_nums;
	void		   *block_data;
	size_t			offset;
	size_t			length;
	cl_uint			nr_loaded;
	CUresult		rc;

	Assert(IsGpuServerProcess());
	if (!iomap_buffer_segments)
		werror("NVMe-Strom is not configured");
	/* ensure the i/o mapped buffer is already available */
	iomap_seg = GetIOMapBufferSegment(gpuserv_cuda_dindex);
	pg_memory_barrier();
	if (!iomap_seg->iomap_handle)
		werror("NVMe-Strom is not initialized yet");

	/* Device memory should be already imported on allocation time */
	Assert(iomap_buffer_base != 0UL);
	/* PDS/KDS format check */
	Assert(pds->kds.format == KDS_FORMAT_BLOCK);

	if (m_kds < iomap_buffer_base ||
		m_kds + pds->kds.length > iomap_buffer_base + iomap_buffer_size)
		werror("NVMe-Strom: P2P DMA destination out of range");
	offset = m_kds - iomap_buffer_base;

	/* nothing special if all the blocks are already loaded */
	if (pds->nblocks_uncached == 0)
	{
		rc = cuMemcpyHtoDAsync(m_kds,
							   &pds->kds,
							   pds->kds.length,
							   CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuMemcpyHtoDAsync: %s", errorText(rc));
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

	/* setup ioctl(2) command */
	memset(&cmd, 0, sizeof(StromCmd__MemCopySsdToGpu));
	cmd.handle		= iomap_seg->iomap_handle;
	cmd.offset		= offset;
	cmd.file_desc	= gtask->file_desc;
	cmd.nr_chunks	= pds->nblocks_uncached;
	cmd.chunk_sz	= BLCKSZ;
	cmd.relseg_sz	= RELSEG_SIZE;
	cmd.chunk_ids	= block_nums;
	cmd.wb_buffer	= block_data;

	/* (1) kick SSD2GPU P2P DMA */
	if (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_SSD2GPU, &cmd) != 0)
		abort(); //werror("failed on STROM_IOCTL__MEMCPY_SSD2GPU: %m");

	/* (2) kick RAM2GPU DMA (earlier half) */
	rc = cuMemcpyHtoDAsync(m_kds,
						   &pds->kds,
						   length,
						   CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuMemCopyFromSSDWaitRaw(cmd.dma_task_id);
		werror("failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	}

	/* (3) kick RAM2GPU DMA (later half; if any) */
	if (cmd.nr_ram2gpu > 0)
	{
		length = BLCKSZ * cmd.nr_ram2gpu;
		offset = ((char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds,
													   pds->kds.nitems) -
				  (char *)&pds->kds) - length;
		rc = cuMemcpyHtoDAsync(m_kds + offset,
							   (char *)&pds->kds + offset,
							   length,
							   CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuMemCopyFromSSDWaitRaw(cmd.dma_task_id);
			werror("failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		}
	}
	/* (4) wait for completion of SSD2GPU P2P DMA */
	gpuMemCopyFromSSDWaitRaw(cmd.dma_task_id);
}

/*
 * TablespaceCanUseNvmeStrom
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
TablespaceCanUseNvmeStrom(Oid tablespace_oid)
{
	vfs_nvme_status *entry;
	const char *pathname;
	int			fdesc;
	bool		found;

	if (iomap_buffer_size == 0 || !nvme_strom_enabled)
		return false;	/* NVMe-Strom is not configured or enabled */

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	/* quick lookup but sufficient for more than 99.99% cases */
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
	Oid		tablespace_oid = RelationGetForm(relation)->reltablespace;
	/* SSD2GPU on temp relation is not supported */
	if (RelationUsesLocalBuffers(relation))
		return false;
	return TablespaceCanUseNvmeStrom(tablespace_oid);
}

/*
 * RelationWillUseNvmeStrom
 */
bool
RelationWillUseNvmeStrom(Relation relation, BlockNumber *p_nr_blocks)
{
	BlockNumber		nr_blocks;

	/* at least, storage must support NVMe-Strom */
	if (!RelationCanUseNvmeStrom(relation))
		return false;

	/*
	 * NOTE: RelationGetNumberOfBlocks() has a significant but helpful
	 * side-effect. It opens all the underlying files of MAIN_FORKNUM,
	 * then set @rd_smgr of the relation.
	 * It allows extension to touch file descriptors without invocation of
	 * ReadBuffer().
	 */
	nr_blocks = RelationGetNumberOfBlocks(relation);
	if (!debug_force_nvme_strom &&
		nr_blocks < nvme_strom_threshold)
		return false;

	/*
	 * ok, it looks to me NVMe-Strom is supported, and relation size is
	 * reasonably large to run with SSD-to-GPU Direct mode.
	 */
	if (p_nr_blocks)
		*p_nr_blocks = nr_blocks;
	return true;
}

/*
 * ScanPathWillUseNvmeStrom - Optimizer Hint
 */
bool
ScanPathWillUseNvmeStrom(PlannerInfo *root, RelOptInfo *baserel)
{
	RangeTblEntry *rte;
	HeapTuple	tuple;
	bool		relpersistence;

	if (!TablespaceCanUseNvmeStrom(baserel->reltablespace))
		return false;

	/* unable to apply NVMe-Strom on temporay tables */
	rte = root->simple_rte_array[baserel->relid];
	tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for relation %u", rte->relid);
	relpersistence = ((Form_pg_class) GETSTRUCT(tuple))->relpersistence;
	ReleaseSysCache(tuple);

	if (relpersistence != RELPERSISTENCE_PERMANENT &&
		relpersistence != RELPERSISTENCE_UNLOGGED)
		return false;

	/* Is number of blocks sufficient to NVMe-Strom? */
	if (!debug_force_nvme_strom && baserel->pages < nvme_strom_threshold)
		return false;

	/* ok, this table scan can use nvme-strom */
	return true;
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
 * iomap_buffer_owner_main
 *
 * MEMO: Since CUDA 8.0, once a process call cuInit(), then its forked child
 * processes will fail on cuInit() after that. It means, postmaster process
 * cannot touch CUDA APIs thus never be a holder of CUDA resources.
 * So, this background worker performs a lazy resource holder of i/o mapped
 * buffer for SSD2GPU P2P DMA.
 */
static void
iomap_buffer_owner_main(Datum __arg)
{
	char	   *pos;
	int			i, j;
	int			ev;
	CUresult	rc;

	/* no special handling is needed on SIGTERM/SIGQUIT; just die */
	BackgroundWorkerUnblockSignals();

	/* init CUDA runtime */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", errorText(rc));

	/* allocate device memory and map to the host physical memory space */
	pos = iomap_buffer_segments;
	for (i=0; i < numDevAttrs; i++)
	{
		IOMapBufferSegment *iomap_seg = GetIOMapBufferSegment(i);
		CUdevice		cuda_device;
		CUcontext		cuda_context;
		CUdeviceptr		cuda_devptr;
		CUipcMemHandle	cuda_mhandle;
		Size			remain;
		int				mclass;
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

		/* setup IOMapBufferSegment */
		elog(LOG, "NVMe-Strom: GPU Device Memory (%p-%p; %zuMB) is mapped",
			 (char *)cuda_devptr,
			 (char *)cuda_devptr + iomap_buffer_size - 1,
			 (size_t)(iomap_buffer_size >> 20));
		memcpy(&iomap_seg->cuda_mhandle, &cuda_mhandle,
			   sizeof(CUipcMemHandle));
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
		pg_memory_barrier();

		/*
		 * iomap_seg->iomap_handle != 0 indicates the i/o mapped device
		 * memory is now ready to use. So, we have to put the @handle
		 * last. Order shall be guaranteed with memory barrier.
		 */
		iomap_seg->iomap_handle = cmd.handle;

		pos += SizeOfIOMapBufferSegment;
	}

	/*
	 * Loop forever
	 */
	for (;;)
	{
		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();

		/*
		 * TODO: It may be a good idea to have a health check of i/o mapped
		 * device memory.
		 */

		ev = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_TIMEOUT |
					   WL_POSTMASTER_DEATH,
					   60 * 1000);		/* wake up per minutes */

		/* Emergency bailout if postmaster has died. */
		if (ev & WL_POSTMASTER_DEATH)
			exit(1);
	}
}

/*
 * pgstrom_startup_nvme_strom
 */
static void
pgstrom_startup_nvme_strom(void)
{
	Size		required;
	bool		found;
	struct stat	stbuf;

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
	required = SizeOfIOMapBufferSegment * numDevAttrs,
	iomap_buffer_segments = ShmemInitStruct("iomap_buffer_segments",
											required,
											&found);
	if (found)
		elog(FATAL, "Bug? \"iomap_buffer_segments\" is already initialized");
	memset(iomap_buffer_segments, 0, required);
}

/*
 * pgstrom_init_nvme_strom
 */
void
pgstrom_init_nvme_strom(void)
{
	static int	__iomap_buffer_size;
	Size		shared_buffer_size = (Size)NBuffers * (Size)BLCKSZ;
	Size		max_nchunks;
	Size		required;
	BackgroundWorker worker;

	/* pg_strom.iomap_buffer_size */
	DefineCustomIntVariable("pg_strom.iomap_buffer_size",
							"I/O mapped buffer size for SSD-to-GPU P2P DMA",
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

	/* pg_strom.nvme_strom_enabled */
	DefineCustomBoolVariable("pg_strom.nvme_strom_enabled",
							 "Turn on/off SSD-to-GPU P2P DMA",
							 NULL,
							 &nvme_strom_enabled,
							 true,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* pg_strom.debug_force_nvme_strom */
	DefineCustomBoolVariable("pg_strom.debug_force_nvme_strom",
							 "(DEBUG) force to use raw block scan mode",
							 NULL,
							 &debug_force_nvme_strom,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/*
	 * MEMO: Threshold of table's physical size to use NVMe-Strom:
	 *   ((System RAM size) -
	 *    (shared_buffer size)) * 0.67 + (shared_buffer size)
	 *
	 * If table size is enough large to issue real i/o, NVMe-Strom will
	 * make advantage by higher i/o performance.
	 */
	sysconf_pagesize = sysconf(_SC_PAGESIZE);
	if (sysconf_pagesize < 0)
		elog(ERROR, "failed on sysconf(_SC_PAGESIZE): %m");
	sysconf_phys_pages = sysconf(_SC_PHYS_PAGES);
	if (sysconf_phys_pages < 0)
		elog(ERROR, "failed on sysconf(_SC_PHYS_PAGES): %m");
	if (sysconf_pagesize * sysconf_phys_pages < shared_buffer_size)
		elog(ERROR, "Bug? shared_buffer is larger than system RAM");
	nvme_strom_threshold = ((sysconf_pagesize * sysconf_phys_pages -
							 shared_buffer_size) * 2 / 3 +
							shared_buffer_size) / BLCKSZ;

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

		/* also needs CUDA resource owner */
		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "NVMe-Strom I/O Mapped Buffer");
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
		worker.bgw_start_time = BgWorkerStart_PostmasterStart;
		worker.bgw_restart_time = BGW_NEVER_RESTART;
		worker.bgw_main = iomap_buffer_owner_main;
		RegisterBackgroundWorker(&worker);
	}
}
