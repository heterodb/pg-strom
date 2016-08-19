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
#include <asm/ioctl.h>

enum {
	STROM_IOCTL__CHECK_FILE				= _IO('S',0x80),
	STROM_IOCTL__MAP_GPU_MEMORY			= _IO('S',0x81),
	STROM_IOCTL__UNMAP_GPU_MEMORY		= _IO('S',0x82),
	STROM_IOCTL__INFO_GPU_MEMORY		= _IO('S',0x83),
	STROM_IOCTL__MEMCPY_SSD2GPU			= _IO('S',0x84),
	STROM_IOCTL__MEMCPY_SSD2GPU_ASYNC	= _IO('S',0x85),
	STROM_IOCTL__MEMCPY_SSD2GPU_WAIT	= _IO('S',0x86),
	STROM_IOCTL__DEBUG					= _IO('S',0x87),
};

/* STROM_IOCTL__CHECK_FILE */
struct StromCmd__CheckFile
{
	int				fdesc;		/* in: file descriptor to be checked */
};
typedef struct StromCmd__CheckFile		StromCmd__CheckFile;

/* STROM_IOCTL__MAP_GPU_MEMORY */
struct StromCmd__MapGpuMemory
{
	unsigned long	handle;		/* out: handler of the mapped region */
	uint32_t		gpu_page_sz;/* out: page size of GPU memory */
	uint32_t		gpu_npages;	/* out: number of page entries */
	uint64_t		vaddress;	/* in: virtual address of the device memory */
	size_t			length;		/* in: length of the device memory */
};
typedef struct StromCmd__MapGpuMemory	StromCmd__MapGpuMemory;

/* STROM_IOCTL__UNMAP_GPU_MEMORY */
struct StromCmd__UnmapGpuMemory
{
	unsigned long	handle;		/* in: handler of the mapped region */
};
typedef struct StromCmd__UnmapGpuMemory	StromCmd__UnmapGpuMemory;

/* STROM_IOCTL__INFO_GPU_MEMORY */
struct StromCmd__InfoGpuMemory
{
	unsigned long	handle;		/* in: handler of the mapped region */
	uint32_t		nrooms;		/* in: length of the variable length array */
	uint32_t		version;	/* out: 'version' of the page tables */
	uint32_t		gpu_page_sz;/* out: 'page_size' in bytes */
	uint32_t		nitems;		/* out: number of GPU pages */
	struct {
		void	   *vaddr;		/* out: io-mapped virtual address */
		uint64_t	paddr;		/* out: physical address */
	} pages[1];
};
typedef struct StromCmd__InfoGpuMemory	StromCmd__InfoGpuMemory;

/* STROM_IOCTL__MEMCPY_SSD2GPU */
struct strom_dma_chunk
{
	loff_t			fpos;		/* in: file position from the head */
	size_t			length;		/* in: length of this chunk */
};
typedef struct strom_dma_chunk	strom_dma_chunk;

struct StromCmd__MemCpySsdToGpu
{
	long			status;		/* out: status of this DMA operation */
	unsigned long	handle;		/* in: handler of the mapped GPU memory */
	size_t			offset;		/* in: destination offset from the head of
								 *     this mapped region */
	int				fdesc;		/* in: file descriptor, if any. Or, -1 */
	int				nchunks;	/* in: number of the source chunks */
	strom_dma_chunk	chunks[1];	/* in: ...variable length array... */
};
typedef struct StromCmd__MemCpySsdToGpu	StromCmd__MemCpySsdToGpu;

/* STROM_IOCTL__MEMCPY_SSD2GPU_ASYNC */
struct StromCmd__MemCpySsdToGpuAsync
{
	unsigned long	dma_task_id;/* out: identifier of the DMA task */
	long		   *p_status;	/* in: pointer to the async DMA status field */
	unsigned long	handle;		/* in: handler of the mapped GPU memory */
	size_t			offset;		/* in: destination offset from head of
								 *     the mapped region */
	int				fdesc;		/* in: file descriptor, if any. Or, -1 */
	int				nchunks;	/* in: number of the source chunks */
	strom_dma_chunk	chunks[1];	/* in: ...variable length array... */
};
typedef struct StromCmd__MemCpySsdToGpuAsync StromCmd__MemCpySsdToGpuAsync;

/* STROM_IOCTL__MEMCPY_SSD2GPU_WAIT */
typedef struct
{
	unsigned int	ntasks;		/* in: length of the dma_task_id[] array */
	unsigned int	nwaits;		/* in: Min number of DMA tasks to wait 
								 * out: num of DMA tasks actually completed */
	unsigned long	dma_task_id[1];	/* in: ID of the DMA tasks
									 * out: ID of the completed DMA tasks */
} StromCmd__MemCpySsdToGpuWait;

/* STROM_IOCTL_DEBUG */
typedef struct
{
	int				fdesc;		/* in: file descriptor */
	unsigned long	offset;
	unsigned long	length;
} StromCmd__Debug;


#endif /* NVME_STROM_H */
