/* ----------------------------------------------------------------
 *
 * nvme-strom.h
 *
 * Definition of SSD-to-GPU stuff
 *
 * ----------------------------------------------------------------
 */
#ifndef NVME_STROM_H
#define NVME_STROM_H
#ifndef __KERNEL__
#define __user
#endif
#include <asm/ioctl.h>

enum {
	STROM_IOCTL__CHECK_FILE					= _IO('S',0x80),
	STROM_IOCTL__MAP_GPU_MEMORY				= _IO('S',0x81),
	STROM_IOCTL__UNMAP_GPU_MEMORY			= _IO('S',0x82),
	STROM_IOCTL__LIST_GPU_MEMORY			= _IO('S',0x83),
	STROM_IOCTL__INFO_GPU_MEMORY			= _IO('S',0x84),
	STROM_IOCTL__MEMCPY_SSD2GPU				= _IO('S',0x85),
	STROM_IOCTL__MEMCPY_SSD2GPU_ASYNC		= _IO('S',0x86),
	STROM_IOCTL__MEMCPY_SSD2GPU_WAIT		= _IO('S',0x87),
};

/* path of ioctl(2) entrypoint */
#define NVME_STROM_IOCTL_PATHNAME		"/proc/nvme-strom"

/* STROM_IOCTL__CHECK_FILE */
typedef struct StromCmd__CheckFile
{
	int				fdesc;		/* in: file descriptor to be checked */
} StromCmd__CheckFile;

/* STROM_IOCTL__MAP_GPU_MEMORY */
typedef struct StromCmd__MapGpuMemory
{
	unsigned long	handle;		/* out: handler of the mapped region */
	uint32_t		gpu_page_sz;/* out: page size of GPU memory */
	uint32_t		gpu_npages;	/* out: number of page entries */
	uint64_t		vaddress;	/* in: virtual address of the device memory */
	size_t			length;		/* in: length of the device memory */
} StromCmd__MapGpuMemory;

/* STROM_IOCTL__UNMAP_GPU_MEMORY */
typedef struct StromCmd__UnmapGpuMemory
{
	unsigned long	handle;		/* in: handler of the mapped region */
} StromCmd__UnmapGpuMemory;

/* STROM_IOCTL__LIST_GPU_MEMORY */
typedef struct StromCmd__ListGpuMemory
{
	uint32_t		nrooms;		/* in: length of the @handles array */
	uint32_t		nitems;		/* out: number of mapped region */
	unsigned long	handles[1];	/* out: array of mapped region handles */
} StromCmd__ListGpuMemory;

/* STROM_IOCTL__INFO_GPU_MEMORY */
typedef struct StromCmd__InfoGpuMemory
{
	unsigned long	handle;		/* in: handler of the mapped region */
	uint32_t		nrooms;		/* in: length of the variable length array */
	uint32_t		nitems;		/* out: number of GPU pages */
	uint32_t		version;	/* out: 'version' of the page tables */
	uint32_t		gpu_page_sz;/* out: 'page_size' in bytes */
	uint32_t		owner;		/* out: UID of the owner */
	unsigned long	map_offset;	/* out: offset of valid area from the head */
	unsigned long	map_length;	/* out: length of valid area */
	uint64_t		paddrs[1];	/* out: array of physical addresses */
} StromCmd__InfoGpuMemory;

/* STROM_IOCTL__MEMCPY_SSD2GPU or STROM_IOCTL__MEMCPY_SSD2GPU_ASYNC */
typedef struct strom_dma_chunk
{
	loff_t			fpos;		/* in: position of the source file from 
								 *     the head of file */
	size_t			offset;		/* in: offset of the destination buffer from
								 *     the head of mapped GPU memory */
	size_t			length;		/* in: length of this chunk */
} strom_dma_chunk;

typedef struct StromCmd__MemCpySsdToGpu
{
	unsigned long	dma_task_id;/* out: ID of the DMA task (only async) */
	long			status;		/* out: status of the DMA task (only sync) */
	unsigned long	handle;		/* in: handler of the mapped GPU memory */
	int				fdesc;		/* in: descriptor of the source file */
	int				nchunks;	/* in: number of the source chunks */
	strom_dma_chunk	chunks[1];	/* in: ...variable length array... */
} StromCmd__MemCpySsdToGpu;

/* STROM_IOCTL__MEMCPY_SSD2GPU_WAIT */
typedef struct StromCmd__MemCpySsdToGpuWait
{
	unsigned long	dma_task_id;/* in: ID of the DMA task to wait */
	long			status;		/* out: status of the DMA task */
} StromCmd__MemCpySsdToGpuWait;

#endif /* NVME_STROM_H */
