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
#include <stdint.h>
#define __user
#endif
#include <asm/ioctl.h>

enum {
	STROM_IOCTL__LICENSE_VALIDATION	= _IO('S',0x60),
	STROM_IOCTL__CHECK_FILE			= _IO('S',0x80),
	STROM_IOCTL__MAP_GPU_MEMORY		= _IO('S',0x81),
	STROM_IOCTL__UNMAP_GPU_MEMORY	= _IO('S',0x82),
	STROM_IOCTL__LIST_GPU_MEMORY	= _IO('S',0x83),
	STROM_IOCTL__INFO_GPU_MEMORY	= _IO('S',0x84),
	STROM_IOCTL__ALLOC_DMA_BUFFER	= _IO('S',0x85),
	STROM_IOCTL__MEMCPY_SSD2GPU		= _IO('S',0x90),
	STROM_IOCTL__MEMCPY_SSD2RAM		= _IO('S',0x91),
	STROM_IOCTL__MEMCPY_WAIT		= _IO('S',0x92),
	STROM_IOCTL__STAT_INFO			= _IO('S',0x99),
};

/* path of ioctl(2) entrypoint */
#define NVME_STROM_IOCTL_PATHNAME		"/proc/nvme-strom"

/* STROM_IOCTL__LICENSE_VALIDATION */
typedef struct StromCmd__LicenseValidation
{
	uint32_t	version;		/* out: VERSION field */
	const char *serial_nr;		/* out: SERIAL_NR field */
	uint32_t	issued_at;		/* out: ISSUED_AT field; YYYYMMDD */
	uint32_t	expired_at;		/* out: EXPIRED_AT field; YYYYMMDD */
	const char *licensee_name;	/* out: LICENSEE_NAME field */
	const char *licensee_mail;	/* out: LICENSEE_MAIL field */
	const char *license_desc;	/* out: LICENSE_DESC field, if any */
	uint32_t	length;			/* in: length of the binary license image */
	unsigned char license[1];	/* in: binary license image
								 * out: buffer of variable length data */
} StromCmd__LicenseValidation;

/* STROM_IOCTL__CHECK_FILE */
typedef struct StromCmd__CheckFile
{
	int				fdesc;		/* in: file descriptor to be checked */
	/* out: NUMA node-id where the storage device is installed. It can
	 *      be -1, if md-raid0 stripes SSDs on multiple NUMA nodes. */
	int				numa_node_id;
	/* out: non-zero, if source SSD device supports 64bit DMA; which
	 *      means NUMA aware SSD2RAM DMA is also supported. */
	int				support_dma64;
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

/* STROM_IOCTL__MEMCPY_SSD2GPU */
typedef struct StromCmd__MemCopySsdToGpu
{
	unsigned long	dma_task_id;/* out: ID of the DMA task */
	unsigned int	nr_ram2gpu;	/* out: # of RAM2GPU chunks */
	unsigned int	nr_ssd2gpu;	/* out: # of SSD2GPU chunks */
	unsigned int	nr_dma_submit; /* out: # of SSD2GPU DMA submit */
	unsigned int	nr_dma_blocks; /* out: # of SSD2GPU DMA blocks */
	unsigned long	handle;		/* in: handle of the mapped GPU memory */
	size_t			offset;		/* in: offset from the head of GPU memory */
	int				file_desc;	/* in: file descriptor of the source file */
	unsigned int	nr_chunks;	/* in: number of chunks */
	unsigned int	chunk_sz;	/* in: chunk-size (BLCKSZ in PostgreSQL) */
	unsigned int	relseg_sz;	/* in: # of chunks per file. (RELSEG_SIZE
								 *     in PostgreSQL). 0 means no boundary. */
	uint32_t __user *chunk_ids;	/* in: array of BlockNumber in PostgreSQL */
	char __user	   *wb_buffer;	/* in: write-back buffer in user space;
								 * consumed from the tail, and must be at least
								 * chunk_sz * nr_chunks bytes. */
} StromCmd__MemCopySsdToGpu;

/* STROM_IOCTL__MEMCPY_WAIT */
typedef struct StromCmd__MemCopyWait
{
	unsigned long	dma_task_id;/* in: ID of the DMA task to wait */
	long			status;		/* out: status of the DMA task */
} StromCmd__MemCopyWait;

/* STROM_IOCTL__MEMCPY_SSD2RAM */
typedef struct StromCmd__MemCopySsdToRam
{
	unsigned long	dma_task_id;/* out: ID of the DMA task */
	unsigned int	nr_ram2ram; /* out: # of RAM2RAM chunks */
	unsigned int	nr_ssd2ram; /* out: # of SSD2RAM chunks */
	unsigned int	nr_dma_submit;	/* out: # of SSD2GPU DMA submit */
	unsigned int	nr_dma_blocks;	/* out: # of SSD2RAM DMA blocks */

	void __user	   *dest_uaddr;	/* in: virtual address of the destination
								 *     buffer; which must be mapped using
								 *     mmap(2) on /proc/nvme-strom */
	int				file_desc;	/* in: file descriptor of the source file */
	unsigned int	nr_chunks;	/* in: number of chunks */
	unsigned int    chunk_sz;	/* in: chunk-size (BLCKSZ in PostgreSQL) */
	unsigned int	relseg_sz;	/* in: # of chunks per file. (RELSEG_SIZE
								 *     in PostgreSQL). 0 means no boundary. */
	uint32_t __user *chunk_ids;	/* in: # of chunks per file (RELSEG_SIZE in
								 *     PostgreSQL). 0 means no boundary. */
} StromCmd__MemCopySsdToRam;

/* STROM_IOCTL__ALLOC_DMA_BUFFER */
typedef struct StromCmd__AllocDMABuffer
{
	size_t			length;		/* in: required length of DMA buffer */
	int				node_id;	/* in: numa-id to be located */
	int				dmabuf_fdesc; /* out: FD of anon file descriptor */
} StromCmd__AllocDMABuffer;

/* STROM_IOCTL__STAT_INFO */
#define NVME_STROM_STATFLAGS__DEBUG		0x0001
typedef struct StromCmd__StatInfo
{
	unsigned int	version;	/* in: = 1, always */
	unsigned int	flags;		/* in: one of NVME_STROM_STATFLAGS__* */
	uint64_t		tsc;		/* tsc counter */
	uint64_t		nr_ioctl_memcpy_submit;		/* MEMCPY_SSD2GPU or */
	uint64_t		clk_ioctl_memcpy_submit;	/* MEMCPY_SSD2RAM */
	uint64_t		nr_ioctl_memcpy_wait;		/* MEMCPY_WAIT */
	uint64_t		clk_ioctl_memcpy_wait;
	uint64_t		nr_ssd2gpu;
	uint64_t		clk_ssd2gpu;
	uint64_t		nr_setup_prps;
	uint64_t		clk_setup_prps;
	uint64_t		nr_submit_dma;
	uint64_t		clk_submit_dma;
	uint64_t		nr_wait_dtask;
	uint64_t		clk_wait_dtask;
	uint64_t		nr_wrong_wakeup;
	uint64_t		total_dma_length;
	uint64_t		cur_dma_count;
	uint64_t		max_dma_count;
	uint64_t		nr_debug1;
	uint64_t		clk_debug1;
	uint64_t		nr_debug2;
	uint64_t		clk_debug2;
	uint64_t		nr_debug3;
	uint64_t		clk_debug3;
	uint64_t		nr_debug4;
	uint64_t		clk_debug4;
} StromCmd__StatInfo;

#endif /* NVME_STROM_H */
