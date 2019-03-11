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
	STROM_IOCTL__LICENSE_ADMIN		= _IO('S',0x60),
	STROM_IOCTL__CHECK_FILE			= _IO('S',0x80),
	STROM_IOCTL__MAP_GPU_MEMORY		= _IO('S',0x81),
	STROM_IOCTL__UNMAP_GPU_MEMORY	= _IO('S',0x82),
	STROM_IOCTL__LIST_GPU_MEMORY	= _IO('S',0x83),
	STROM_IOCTL__INFO_GPU_MEMORY	= _IO('S',0x84),
	STROM_IOCTL__MEMCPY_SSD2GPU		= _IO('S',0x90),
	STROM_IOCTL__MEMCPY_SSD2GPU_RAW	= _IO('S',0x91),
	STROM_IOCTL__MEMCPY_WAIT		= _IO('S',0x92),
	STROM_IOCTL__STAT_INFO			= _IO('S',0x99),
};

/* path of ioctl(2) entrypoint */
#define NVME_STROM_IOCTL_PATHNAME		"/proc/nvme-strom"
/* default license location */
#define HETERODB_LICENSE_PATHNAME		"/etc/heterodb.license"
/* fixed length of the license key (2048bits) */
#define HETERODB_LICENSE_KEYLEN			256

/* STROM_IOCTL__LICENSE_ADMIN */
typedef struct StromCmd__LicenseInfo
{
	uint32_t	version;		/* out: VERSION field */
	const char *serial_nr;		/* out: SERIAL_NR field */
	uint32_t	issued_at;		/* out: ISSUED_AT field; YYYYMMDD */
	uint32_t	expired_at;		/* out: EXPIRED_AT field; YYYYMMDD */
	uint32_t	nr_gpus;		/* out: NR_GPUS field */
	const char *licensee_org;	/* out: LICENSEE_ORG field */
	const char *licensee_name;	/* out: LICENSEE_NAME field */
	const char *licensee_mail;	/* out: LICENSEE_MAIL field */
	const char *license_desc;	/* out: LICENSE_DESC field, if any */
	uint32_t	validation;		/* in: If 0, just query current licence info.
								 *     Elsewhere, validate a new license */
	unsigned char buffer[HETERODB_LICENSE_KEYLEN+2];
								/* in: binary license image, if @validation
								 *     is not zero.
								 * out: buffer of the variable length data. */
} StromCmd__LicenseInfo;

/* STROM_IOCTL__CHECK_FILE */
#define NVME_VOLUME_KIND__RAW_NVME		'R'
#define NVME_VOLUME_KIND__MD_RAID0		'M'

#define NVME_DEVICE_KIND__PCIE			'p'
#define NVME_DEVICE_KIND__RDMA			'r'

typedef struct StromCmd__CheckFile
{
	/* in: file descriptor to be checked */
	int				fdesc;
	/* in: length of the disks[] array */
	int				nrooms;
	/* out: type of the volume; one of the NVME_VOLUME_KIND__* */
	char			volume_kind;
	/* out: type of NVME device; one of the NVME_DEVICE_KIND__* */
	char			nvme_kind;
	/* out: number of the underlying raw-disks */
	int				ndisks;
	/*
	 * out: major/minor code of the raw-disk device where the file is
	 * located on. It is exactly device number of the disk, not partition,
	 * even if filesystem is constructed on a disk-partition.
	 */
	struct {
		int			major;
		int			minor;
	} rawdisks[1];
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

/* STROM_IOCTL__MEMCPY_SSD2GPU_RAW */
typedef struct
{
	unsigned long	m_offset;	/* destination offset from the base address
								 * base = mgmem + offset */
	unsigned int	fchunk_id;	/* source page index of the file. */
	unsigned int	nr_pages;	/* number of pages to be loaded */
} strom_io_chunk;

typedef struct StromCmd__MemCopySsdToGpuRaw
{
	unsigned long	dma_task_id;/* out: ID of the DMA task */
	unsigned int	nr_ram2gpu; /* out: # of RAM2GPU chunks */
	unsigned int	nr_ssd2gpu; /* out: # of SSD2GPU chunks */
	unsigned int	nr_dma_submit;	/* out: # of SSD2GPU DMA submit */
	unsigned int	nr_dma_blocks;	/* out: # of SSD2GPU DMA blocks */
	unsigned long	handle;		/* in: handle of the mapped GPU memory */
	size_t			offset;		/* in: offset from the head of GPU memory */
	int				file_desc;	/* in: file descriptor of the source file */
	unsigned int	nr_chunks;	/* in: number of chunks */
	unsigned int	page_sz;	/* in: page-size application assumes */
	strom_io_chunk __user *io_chunks; /* in: copy source, destination and
									   *     length per I/O chunk. */
} StromCmd__MemCopySsdToGpuRaw;

/* STROM_IOCTL__MEMCPY_WAIT */
typedef struct StromCmd__MemCopyWait
{
	unsigned long	dma_task_id;/* in: ID of the DMA task to wait */
	long			status;		/* out: status of the DMA task */
} StromCmd__MemCopyWait;

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
	uint64_t		nr_submit_wait;
	uint64_t		clk_submit_wait;
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

#ifndef	__KERNEL__
#include <stdio.h>
#include <errno.h>

/* support routine to parse heterodb license file */
static inline int
read_heterodb_license_file(StromCmd__LicenseInfo *cmd, FILE *outerr)
{
	FILE	   *filp = NULL;
	ssize_t		i;
	int			retval = -1;
	long		val = 0;
	int			bits = 0;

	filp = fopen(HETERODB_LICENSE_PATHNAME, "rb");
	if (!filp)
	{
		if (errno == ENOENT)
			return 1;
		else if (outerr)
			fprintf(outerr, "failed to open '%s': %m\n",
					HETERODB_LICENSE_PATHNAME);
		return -1;
	}
	/* Extract base64 */
	for (i=0; ;)
	{
		int		c = fgetc(filp);

		if (c == '=' || c == EOF)
			break;
		if (c >= 'A' && c <= 'Z')
			val |= ((c - 'A') << bits);
		else if (c >= 'a' && c <= 'z')
			val |= ((c - 'a' + 26) << bits);
		else if (c >= '0' && c <= '9')
			val |= ((c - '0' + 52) << bits);
		else if (c == '+')
			val |= (62 << bits);
		else if (c == '/')
			val |= (63 << bits);
		else
		{
			if (outerr)
				fprintf(outerr, "unexpected base64 character: %c\n", c);
			goto out;
		}
		bits += 6;
		while (bits >= 8)
		{
			if (i >= sizeof(cmd->buffer))
			{
				if (outerr)
					fprintf(outerr, "license file too large\n");
				goto out;
			}
			cmd->buffer[i++] = (val & 0xff);
			val >>= 8;
			bits -= 8;
		}
	}
	if (bits > 0)
	{
		if (i >= sizeof(cmd->buffer))
		{
			if (outerr)
				fprintf(outerr, "license file too large\n");
			goto out;
		}
		cmd->buffer[i++] = (val & 0xff);
	}
	if (i != sizeof(cmd->buffer))
	{
		if (outerr)
			fprintf(outerr, "license file length mismatch\n");
		goto out;
	}
	if (8 * HETERODB_LICENSE_KEYLEN != (((int)cmd->buffer[0] << 8) |
										((int)cmd->buffer[1])))
	{
		if (outerr)
			fprintf(outerr, "license file corruption?\n");
		goto out;
	}
	retval = 0;
out:
	fclose(filp);
	return retval;
}
#endif /* __KERNEL__ */
#endif /* NVME_STROM_H */
