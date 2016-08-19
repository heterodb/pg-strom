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
#include "storage/ipc.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include "nvme_strom.h"

#define IOMAPDMEM_CHUNKSZ_MAX_BIT		34		/* 16G */
#define IOMAPDMEM_CHUNKSZ_MIN_BIT		12		/* 4KB */
#define IOMAPDMEM_CHUNKSZ_MAX			(1UL << IOMAPDMEM_CHUNKSZ_MAX_BIT)
#define IOMAPDMEM_CHUNKSZ_MIN			(1UL << IOMAPDMEM_CHUNKSZ_MIN_BIT)

typedef struct
{
	dlist_node			addr_chain;
	dlist_node			free_chain;	/* zero, if active chunk */
	cl_uint				mclass;		/* size of the chunk */
} IOMapDevMemChunk;

typedef struct
{
	CUipcMemHandle		dmem_handle;	/* (const) IPC handle */
	slock_t				lock;
	dlist_head			unused_chunks;	/* unused IOMapDevMemChunk */
	dlist_head			addr_chunks;	/* chunks in order of address */
	dlist_head			free_chunks[IOMAPDMEM_CHUNKSZ_MAX_BIT + 1];
	IOMapDevMemChunk	dmem_chunks[FLEXIBLE_ARRAY_MEMBER];
} IOMapDevMemHead;

static shmem_startup_hook_type shmem_startup_next = NULL;
static const char	   *nvme_strom_ioctl_pathname = "/proc/nvme-strom";
static Size				iomap_device_memsz;
static IOMapDevMemHead **iomap_devmem_heads = NULL;	/* for each device */

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
 * gpuDmaMemAllocIOMap
 *
 * Allocation of device memory which is mapped to I/O address space
 */
CUresult
gpuDmaMemAllocIOMap()
{}

/*
 * gpuDmaMemFreeIOMap
 *
 * Release of device memory which is mapped to I/O address space
 */
CUresult
gpuDmaMemFreeIOMap()
{}

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
	CUresult	rc;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	/* is nvme-strom driver installed? */
	if (stat(nvme_strom_ioctl_pathname, &stbuf) != 0)
		elog(ERROR,
			 (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			  errmsg("failed on stat(2) for %s: %m",
					 nvme_strom_ioctl_pathname),
			  errhint("nvme-strom.ko may not be installed on the system")));

	/* allocation of static shared memory */
	iomap_devmem_heads = malloc(sizeof(IOMapDevMemHead *) * numDevAttrs);
	if (!iomap_devmem_heads)
		elog(ERROR, "out of memory");

	max_nchunks = (iomap_device_memsz +
				   IOMAPDMEM_CHUNKSZ_MIN - 1) / IOMAPDMEM_CHUNKSZ_MIN;
	required = offsetof(IOMapDevMemHead, dmem_chunks[max_nchunks]);
	pos = ShmemInitStruct("iomap_devmem_heads",
						  MAXALIGN(required) * numDevAttrs,
						  &found);
	Assert(!found);

	/* setup i/o mapped device memory for each GPU device */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", errorText(rc));

	for (i=0; i < numDevAttrs; i++)
	{
		IOMapDevMemHead	   *dmem_head = (IOMapDevMemHead *) pos;
		CUdevice			cuda_device;
		CUcontext			cuda_context;
		CUdeviceptr			cuda_devptr;

		/* setup i/o mapped device memory for each GPU device */
		rc = cuDeviceGet(&cuda_device, devAttrs[i].DEV_ID);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", errorText(rc));

		rc = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));

		rc = cuMemAlloc(&cuda_devptr, iomap_device_memsz);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemAlloc: %s", errorText(rc));

		rc = cuIpcGetMemHandle(&cuda_handle, cuda_devptr);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuIpcGetMemHandle: %s", errorText(rc));
		







		iomap_devmem_heads[i] = dmem_head;
		pos += MAXALIGN(required);
	}





}

/*
 * pgstrom_init_nvme_strom
 */
void
pgstrom_init_nvme_strom(void)
{
	static int	__iomap_device_memsz;
	Size		max_nchunks;
	Size		required;

	DefineCustomIntVariable("pg_strom.iomap_device_memory_size",
							"size of i/o mapped device memory",
							NULL,
							&__iomap_device_memsz,
							0,
							0,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	iomap_device_memsz = (Size)__iomap_device_memsz << 10;

	/*
	 * i/o mapped device memory shall be set up
	 * only when pg_strom.iomap_device_memory_size > 0.
	 */
	if (iomap_device_memsz > 0)
	{
		max_nchunks = (iomap_device_memsz +
					   IOMAPDMEM_CHUNKSZ_MIN - 1) / IOMAPDMEM_CHUNKSZ_MIN;
		required = offsetof(IOMapDevMemHead, dmem_chunks[max_nchunks]);

		RequestAddinShmemSpace(MAXALIGN(required) * numDevAttrs);
		shmem_startup_next = shmem_startup_hook;
		shmem_startup_hook = pgstrom_startup_nvme_strom;
	}
}
