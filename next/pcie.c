/*
 * pcie.c
 *
 * PCI-E device information collector routines
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include <sys/sysmacros.h>

typedef struct PciDevItem	PciDevItem;
typedef struct BlockDevItem	BlockDevItem;

#define PciDevItemKeySize		(offsetof(PciDevItem, pci_func_id) + sizeof(int))
#define PCIDEV_KIND__NVME		'n'
#define PCIDEV_KIND__GPU		'g'
#define PCIDEV_KIND__HCA		'h'
#define PCIDEV_KIND__UNKNOWN	'?'

struct PciDevItem
{
	int			pci_domain;			/* DDDD of DDDD:bb:dd.f */
	int			pci_bus_id;			/* bb of DDDD:bb:dd.f */
	int			pci_dev_id;			/* dd of DDDD:bb:dd.f */
	int			pci_func_id;		/* f of DDDD:bb:dd.f */
	/* above hash-key */
	int			depth;
	char	   *cpu_affinity;
	char		pci_kind;			/* one of PCIDEV_KIND__* */
	dlist_node	chain;
	PciDevItem *parent;
	dlist_head	children;
	int			distance;			/* distance to optimal GPUs */
	Bitmapset  *optimal_gpus;		/* optimal GPUs */
	union {
		struct {
			char		name[48];
			char		model[80];		/* optional */
			char		serial[64];		/* optional */
			char		firmware[64];	/* optional */
		} nvme;
		struct {
			int			cuda_dindex;
			const GpuDevAttributes *gpu_dev_attrs;
		} gpu;
		struct {
			char		name[48];
			char		hca_type[64];
			char		node_guid[64];
		} hca;
	} u;
};

#define BlockDevItemKeySize		(offsetof(BlockDevItem, minor) + sizeof(uint))
struct BlockDevItem
{
	uint		major;
	uint		minor;
	bool		is_valid;
	char		name[80];
	const Bitmapset *optimal_gpus;
};

#define VfsDevItemKeySize		(sizeof(char) * 240)
typedef struct
{
	char		dir[240];
	const Bitmapset *optimal_gpus;
} VfsDevItem;

static HTAB	   *block_dev_htable = NULL;
static HTAB	   *vfs_gpus_htable = NULL;
static List	   *pcie_root_list = NIL;
static List	   *nvme_devices_list = NIL;
static List	   *gpu_devices_list = NIL;

static const char *
__sysfs_read_line(const char *path, char *buffer, size_t buflen)
{
	int			fdesc;
	ssize_t		sz;
	char	   *pos;

	fdesc = open(path, O_RDONLY);
	if (fdesc < 0)
		return NULL;
	while ((sz = read(fdesc, buffer, buflen-1)) < 0)
	{
		if (errno != EINTR)
		{
			close(fdesc);
			return NULL;
		}
	}
	close(fdesc);
	buffer[sz] = '\0';
	pos = strchr(buffer, '\n');
	if (pos)
		*pos = '\0';
	return __trim(buffer);
}

static const char *
sysfs_read_line(const char *path)
{
	static char linebuf[2048];

	return __sysfs_read_line(path, linebuf, sizeof(linebuf));
}

static bool
__sysfs_read_pcie_gpu(PciDevItem *pcie, const char *dirname)
{
	char		path[MAXPGPATH];
	const char *line;

	/* check vendor */
	snprintf(path, sizeof(path), "%s/vendor", dirname);
	line = sysfs_read_line(path);
	if (!line || strcmp(line, "0x10de") != 0)
		return false;

	for (int i=0; i < numGpuDevAttrs; i++)
	{
		const GpuDevAttributes *gattrs = &gpuDevAttrs[i];

		if (pcie->pci_domain == gattrs->PCI_DOMAIN_ID &&
			pcie->pci_bus_id == gattrs->PCI_BUS_ID &&
			pcie->pci_dev_id == gattrs->PCI_DEVICE_ID &&
			(gattrs->MULTI_GPU_BOARD
			 ? pcie->pci_func_id == gattrs->MULTI_GPU_BOARD_GROUP_ID
			 : pcie->pci_func_id == 0))
		{
			pcie->u.gpu.cuda_dindex = i;
			pcie->u.gpu.gpu_dev_attrs = gattrs;

			gpu_devices_list = lappend(gpu_devices_list, pcie);
			return true;
		}
	}
	return false;
}

static bool
__sysfs_read_pcie_nvme(PciDevItem *pcie, const char *dirname)
{
	struct dirent *dent;
	DIR	   *dir;
	char	path[MAXPGPATH];

	/* check whether nvme or not */
	snprintf(path, sizeof(path), "%s/nvme", dirname);
	dir = AllocateDir(path);
	if (!dir)
		return false;	/* not nvme device */
	while ((dent = ReadDir(dir, path)) != NULL)
	{
		const char *s;
		char		__path[MAXPGPATH];

		if (strncmp(dent->d_name, "nvme", 4) != 0)
			continue;
		for (s = dent->d_name+4; isdigit(*s); s++);
		if (*s != '\0')
			continue;
		/* ok, it's nvmeXX */
		strncpy(pcie->u.nvme.name, dent->d_name, sizeof(pcie->u.nvme.name));

		snprintf(__path, sizeof(__path), "%s/nvme/%s/model",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pcie->u.nvme.model, s, sizeof(pcie->u.nvme.model));

		snprintf(__path, sizeof(__path), "%s/nvme/%s/serial",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pcie->u.nvme.serial, s, sizeof(pcie->u.nvme.serial));

		snprintf(__path, sizeof(__path), "%s/nvme/%s/firmware_rev",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pcie->u.nvme.firmware, s, sizeof(pcie->u.nvme.firmware));

		nvme_devices_list = lappend(nvme_devices_list, pcie);
		FreeDir(dir);
		return true;
	}
	FreeDir(dir);
	return false;
}

static bool
__sysfs_read_pcie_hca(PciDevItem *pcie, const char *dirname)
{
	struct dirent *dent;
	DIR	   *dir;
	char	path[MAXPGPATH];

	/* check whether nvme or not */
	snprintf(path, sizeof(path), "%s/infiniband", dirname);
	dir = AllocateDir(path);
	if (!dir)
		return false;	/* not nvme device */
	while ((dent = ReadDir(dir, path)) != NULL)
	{
		const char *s;
		char	__path[MAXPGPATH];

		if (strncmp(dent->d_name, "mlx", 3) != 0)
			continue;
		/* ok, it's mlxX_X */
		strncpy(pcie->u.hca.name, dent->d_name, sizeof(pcie->u.hca.name));

		snprintf(__path, sizeof(__path), "%s/infiniband/%s/hca_type",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pcie->u.hca.hca_type, s, sizeof(pcie->u.hca.hca_type));

		snprintf(__path, sizeof(__path), "%s/infiniband/%s/node_guid",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pcie->u.hca.node_guid, s, sizeof(pcie->u.hca.node_guid));

		nvme_devices_list = lappend(nvme_devices_list, pcie);
		FreeDir(dir);
		return true;
	}
	FreeDir(dir);
	return false;
}

static void
__sysfs_read_pcie_subtree(PciDevItem *parent,
						  const char *dirname)
{
	DIR		   *dir;
	struct dirent *dent;

	dir = AllocateDir(dirname);
	if (!dir)
		elog(ERROR, "failed on open '%s' directory: %m", dirname);
	while ((dent = ReadDir(dir, dirname)) != NULL)
	{
		const char *delim = "::.";
		char	   *pos;

		/* xxxx:xx:xx.x? */
		for (pos = dent->d_name; *pos != '\0'; pos++)
		{
			if (*pos == *delim)
				delim++;
			else if (*delim != '\0' ? !isxdigit(*pos) : !isdigit(*pos))
				break;
		}
		if (*pos == '\0' && *delim == '\0')
		{
			PciDevItem *pcie;
			int			pci_domain;
			int			pci_bus_id;
			int			pci_dev_id;
			int			pci_func_id;
			char		path[MAXPGPATH];

			if (sscanf(dent->d_name,
					   "%x:%02x:%02x.%d",
					   &pci_domain,
					   &pci_bus_id,
					   &pci_dev_id,
					   &pci_func_id) != 4)
				continue;

			pcie = palloc0(sizeof(PciDevItem));
			pcie->pci_domain = pci_domain;
			pcie->pci_bus_id = pci_bus_id;
			pcie->pci_dev_id = pci_dev_id;
			pcie->pci_func_id = pci_func_id;
			pcie->cpu_affinity = parent->cpu_affinity;
			pcie->depth = parent->depth + 1;
			pcie->distance = -1;
			pcie->optimal_gpus = NULL;
			snprintf(path, sizeof(path),
					 "%s/%s",
					 dirname, dent->d_name);
			if (__sysfs_read_pcie_gpu(pcie, path))
				pcie->pci_kind = PCIDEV_KIND__GPU;
			else if (__sysfs_read_pcie_nvme(pcie, path))
				pcie->pci_kind = PCIDEV_KIND__NVME;
			else if (__sysfs_read_pcie_hca(pcie, path))
				pcie->pci_kind = PCIDEV_KIND__HCA;
			else
				pcie->pci_kind = PCIDEV_KIND__UNKNOWN;
			/* child subtree */
			__sysfs_read_pcie_subtree(pcie, path);
			if (pcie->pci_kind != PCIDEV_KIND__UNKNOWN ||
				!dlist_is_empty(&pcie->children))
				dlist_push_tail(&parent->children, &pcie->chain);
		}
	}
	FreeDir(dir);
}

static void
sysfs_read_pcie_subtree(void)
{
	/* walks on the PCI-E bus tree for each root complex */
	const char	   *dirname = "/sys/devices";
	DIR			   *dir;
	struct dirent  *dent;

	dir = AllocateDir(dirname);
	if (!dir)
		elog(ERROR, "failed on open '%s' directory: %m", dirname);
	while ((dent = ReadDir(dir, dirname)) != NULL)
	{
		PciDevItem *pcie;
		int			pci_domain;
		int			pci_bus_id;
		const char *cpu_affinity;
		char		path[MAXPGPATH];

		/* only /sys/devices/pciXXXX:XX */
		if (sscanf(dent->d_name,
				   "pci%04x:%02x",
				   &pci_domain,
				   &pci_bus_id) != 2)
			continue;
		/* fetch CPU affinity if any */
		snprintf(path, sizeof(path),
				 "%s/%s/pci_bus/%04x:%02x/cpuaffinity",
				 dirname,
				 dent->d_name,
				 pci_domain,
				 pci_bus_id);
		if ((cpu_affinity = sysfs_read_line(path)) == NULL)
			cpu_affinity = "unknown";
		/* PCI root complex */
		pcie = palloc0(sizeof(PciDevItem));
		pcie->pci_domain = pci_domain;
		pcie->pci_bus_id = pci_bus_id;
		pcie->pci_kind = PCIDEV_KIND__UNKNOWN;
		pcie->cpu_affinity = pstrdup(cpu_affinity);
		pcie->distance = -1;
		dlist_init(&pcie->children);

		snprintf(path, sizeof(path),
				 "%s/%s", dirname, dent->d_name);
		__sysfs_read_pcie_subtree(pcie, path);
		if (pcie->pci_kind != PCIDEV_KIND__UNKNOWN || !dlist_is_empty(&pcie->children))
			pcie_root_list = lappend(pcie_root_list, pcie);
	}
	FreeDir(dir);
}

/*
 * sysfs_setup_optimal_gpus
 */
static int
__sysfs_calculate_distance(PciDevItem *curr,
						   PciDevItem *nvme, bool *p_nvme_found,
						   PciDevItem *gpu, bool *p_gpu_found)
{
	dlist_iter	iter;
	int		nvme_depth = -1;
	int		gpu_depth = -1;
	int		dist;

	if (curr == nvme)
		nvme_depth = 0;
	if (curr == gpu)
		gpu_depth = 0;
	Assert(gpu_depth < 0 || nvme_depth < 0);
	dlist_foreach (iter, &curr->children)
	{
		PciDevItem *child = dlist_container(PciDevItem, chain, iter.cur);
		bool		gpu_found = false;
		bool		nvme_found = false;

		dist = __sysfs_calculate_distance(child,
										  nvme, &nvme_found,
										  gpu, &gpu_found);
		if (gpu_found && nvme_found)
		{
			*p_nvme_found = true;
			*p_gpu_found = true;
			return dist;
		}
		else if (gpu_found)
		{
			Assert(gpu_depth < 0);
            gpu_depth = dist + 1;
		}
		else if (nvme_found)
		{
			Assert(nvme_depth < 0);
			nvme_depth = dist + 1;
		}
	}

	if (gpu_depth >= 0 && nvme_depth >= 0)
		dist = (gpu_depth + 1 + nvme_depth);
	else if (gpu_depth >= 0)
		dist = gpu_depth;
	else if (nvme_depth >= 0)
		dist = nvme_depth;
	else
		dist = -1;

	*p_gpu_found = (gpu_depth >= 0);
	*p_nvme_found = (nvme_depth >= 0);
	return dist;
}

static int
sysfs_calculate_distance_root(PciDevItem *nvme, PciDevItem *gpu)
{
	ListCell   *cell;
	int			gpu_depth = -1;
	int			nvme_depth = -1;
	int			root_gap = 5;
	int			dist;

	foreach (cell, pcie_root_list)
	{
		PciDevItem *root = lfirst(cell);
		bool	nvme_found = false;
		bool	gpu_found = false;

		dist = __sysfs_calculate_distance(root,
										  nvme, &nvme_found,
										  gpu, &gpu_found);
		if (gpu_found && nvme_found)
		{
			return dist;
		}
		else if (gpu_found)
		{
			Assert(gpu_depth < 0);
			gpu_depth = dist;
		}
		else if (nvme_found)
		{
			Assert(nvme_depth < 0);
			nvme_depth = dist;
		}
	}

	if (gpu_depth < 0 || nvme_depth < 0)
		return -1;	/* no optimal GPU/NVME */
	if (strcmp(gpu->cpu_affinity, nvme->cpu_affinity) != 0)
		root_gap = 99;
	return (gpu_depth + root_gap + nvme_depth);
}

static void
sysfs_setup_optimal_gpus(void)
{
	ListCell   *lc1, *lc2;
	int			dist;

	foreach (lc1, nvme_devices_list)
	{
		PciDevItem *nvme = lfirst(lc1);

		foreach (lc2, gpu_devices_list)
		{
			PciDevItem *gpu = lfirst(lc2);

			dist = sysfs_calculate_distance_root(nvme, gpu);
			if (nvme->distance < 0 ||
				dist < nvme->distance)
			{
				nvme->distance = dist;
				nvme->optimal_gpus = bms_make_singleton(gpu->u.gpu.cuda_dindex);
			}
			else if (dist == nvme->distance)
			{
				nvme->optimal_gpus = bms_add_member(nvme->optimal_gpus,
													gpu->u.gpu.cuda_dindex);
			}
		}
	}
}

/*
 * sysfs_print_pcie_subtree
 */
static void
__sysfs_print_pcie_subtree(PciDevItem *pcie, int depth)
{
	const GpuDevAttributes *gattrs;
	char		buffer[1024];
	size_t		off = 0;
	dlist_iter	iter;

	if (depth > 0)
	{
		for (int j=1; j < depth; j++)
			off += snprintf(buffer + off, sizeof(buffer) - off, "  ");
		off += snprintf(buffer + off, sizeof(buffer) - off, " - ");
	}
	off += snprintf(buffer + off, sizeof(buffer) - off,
					"[%04x:%02x:%02x.%d]",
					pcie->pci_domain,
					pcie->pci_bus_id,
					pcie->pci_dev_id,
					pcie->pci_func_id);
	switch (pcie->pci_kind)
	{
		case PCIDEV_KIND__NVME:
			off += snprintf(buffer + off, sizeof(buffer) - off,
							" ... %s (%s",
							pcie->u.nvme.name,
							pcie->u.nvme.model);
			break;
		case PCIDEV_KIND__GPU:
			gattrs = pcie->u.gpu.gpu_dev_attrs;
			off += snprintf(buffer + off, sizeof(buffer) - off,
							" ... GPU%d (%s",
							pcie->u.gpu.cuda_dindex,
							gattrs->DEV_NAME);
			break;
		case PCIDEV_KIND__HCA:
			off += snprintf(buffer + off, sizeof(buffer) - off,
							" ... %s (%s",
							pcie->u.hca.name,
							pcie->u.hca.hca_type);
			break;
		default:
			break;
	}

	if (pcie->distance >= 0)
	{
		int		k, count;

		for (k = bms_next_member(pcie->optimal_gpus, -1), count=0;
			 k >= 0;
			 k = bms_next_member(pcie->optimal_gpus, k), count++)
		{
			off += snprintf(buffer + off, sizeof(buffer) - off,
							"%sGPU%d",
							count == 0 ? " --> " : ", ", k);
		}
		off += snprintf(buffer + off, sizeof(buffer) - off,
						" [dist=%d]", pcie->distance);
    }
	off += snprintf(buffer + off, sizeof(buffer) - off, ")");

	elog(LOG, "%s", buffer);

	dlist_foreach (iter, &pcie->children)
	{
		PciDevItem *child = dlist_container(PciDevItem, chain, iter.cur);

		__sysfs_print_pcie_subtree(child, depth+1);
	}
}

static void
sysfs_print_pcie_subtree(void)
{
	ListCell   *lc;

	foreach (lc, pcie_root_list)
	{
		PciDevItem *pcie = lfirst(lc);

		__sysfs_print_pcie_subtree(pcie, 0);
	}
}



/*
 * pgstrom_lookup_optimal_gpus
 */
static const Bitmapset *
sysfs_lookup_optimal_gpus(uint major, uint minor);

static bool
__blkdev_setup_partition(BlockDevItem *bdev)
{
	const char *line;
	char	path[MAXPGPATH];
	int		__major;
	int		__minor;

	/* Is it a partition block? */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/partition",
			 bdev->major,
			 bdev->minor);
	if (access(path, R_OK) != 0)
	{
		if (errno != ENOENT)
			elog(ERROR, "failed on access('%s', R_OK): %m", path);
		return false;
	}

	/* Fetch its container block */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/../dev",
			 bdev->major,
			 bdev->minor);
	line = sysfs_read_line(path);
	if (!line || sscanf(line, "%u:%u",
						&__major,
						&__minor) != 2)
		elog(ERROR, "sysfs '%s' has unexpected value", path);
	bdev->optimal_gpus = sysfs_lookup_optimal_gpus(__major, __minor);
	return true;
}

static bool
__blkdev_setup_md_raid0(BlockDevItem *bdev)
{
	const char *line;
	const Bitmapset *optimal_gpus = NULL;
	char		path[MAXPGPATH];
	char	   *end;
	long		chunk_sz;
	int			count = 0;
	DIR		   *dir;
	struct dirent *dent;

	/* check whether md-raid drive */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/md",
			 bdev->major,
			 bdev->minor);
	if (access(path, R_OK|X_OK) != 0)
	{
		if (errno != ENOENT)
			elog(ERROR, "failed on access('%s', R_OK|X_OK): %m", path);
		return false;
	}

	/* check whether it is md-raid0, or nor */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/md/level",
			 bdev->major,
			 bdev->minor);
	line = sysfs_read_line(path);
	if (!line || strcmp(line, "raid0") != 0)
		goto out;

	/* check chunk size */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/md/chunk_size",
			 bdev->major,
			 bdev->minor);
	line = sysfs_read_line(path);
	if (!line)
		goto out;
	chunk_sz = strtol(line, &end, 10);
	if (*end != '\0' ||
		chunk_sz < PAGE_SIZE ||
		(chunk_sz & (PAGE_SIZE-1)) != 0)
		goto out;

	/* walks on the underlying devices */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/md",
			 bdev->major,
			 bdev->minor);
	dir = AllocateDir(path);
	if (!dir)
		goto out;

	count = 0;
	while ((dent = ReadDir(dir, path)) != NULL)
	{
		char	__path[MAXPGPATH];
		int		__major;
		int		__minor;
		const Bitmapset *__optimal_gpus;

		if (strncmp(dent->d_name, "rd", 2) != 0)
			continue;
		strtol(dent->d_name + 2, &end, 10);
		if (dent->d_name[2] == '\0' || *end != '\0')
			continue;

		snprintf(__path, sizeof(__path),
				 "/sys/dev/block/%u:%u/md/%s/block/dev",
				 bdev->major,
				 bdev->minor,
				 dent->d_name);
		line = sysfs_read_line(__path);
		if (!line || sscanf(line, "%u:%u",
							&__major,
							&__minor) != 2)
			continue;
		__optimal_gpus = sysfs_lookup_optimal_gpus(__major, __minor);
		if (count++ == 0)
			optimal_gpus = __optimal_gpus;
		else
			optimal_gpus = bms_intersect(optimal_gpus, __optimal_gpus);
	}
	FreeDir(dir);
out:
	bdev->optimal_gpus = optimal_gpus;
	return true;
}

static bool
__blkdev_setup_raw_nvme(BlockDevItem *bdev)
{
	const char *line;
	char		path[MAXPGPATH];
	char		temp[MAXPGPATH];
	char	   *name, *end;
	int			pci_domain;
	int			pci_bus_id;
	int			pci_dev_id;
	int			pci_func_id;
	ssize_t		sz;
	ListCell   *lc;

	/* check whether it is raw nvme or not */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/device",
			 bdev->major,
			 bdev->minor);
	if ((sz = readlink(path, temp, sizeof(temp))) < 0)
		return false;
	temp[sz] = '\0';
	name = basename(temp);
	if (strncmp(name, "nvme", 4) != 0)
		return false;
	strtol(name+4, &end, 10);
	if (name[4] == '\0' || *end != '\0')
		return false;

	/* check whether it is local NVME-SSD or not */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/device/transport",
			 bdev->major,
			 bdev->minor);
	line = sysfs_read_line(path);
	if (!line || strcmp(line, "pcie") != 0)
		return false;

	/* fetch PCI-E bus/dev id */
	snprintf(path, sizeof(path),
			 "/sys/dev/block/%u:%u/device/address",
			 bdev->major,
			 bdev->minor);
	line = sysfs_read_line(path);
	if (!line || sscanf(line, "%x:%x:%x.%d",
						&pci_domain,
						&pci_bus_id,
						&pci_dev_id,
						&pci_func_id) != 4)
		return false;

	foreach (lc, nvme_devices_list)
	{
		PciDevItem *pcie = lfirst(lc);

		if (pcie->pci_domain == pci_domain &&
			pcie->pci_bus_id == pci_bus_id &&
			pcie->pci_dev_id == pci_dev_id &&
			pcie->pci_func_id == pci_func_id &&
			pcie->pci_kind == PCIDEV_KIND__NVME)
		{
			MemoryContext	oldcxt = MemoryContextSwitchTo(TopMemoryContext);

			bdev->optimal_gpus = bms_copy(pcie->optimal_gpus);

			MemoryContextSwitchTo(oldcxt);
			return true;
		}
	}
	return false;
}

static const Bitmapset *
sysfs_lookup_optimal_gpus(uint major, uint minor)
{
	BlockDevItem hkey, *bdev;
	bool		found;

	memset(&hkey, 0, sizeof(BlockDevItem));
	hkey.major = major;
	hkey.minor = minor;
	bdev = hash_search(block_dev_htable, &hkey, HASH_ENTER, &found);
	if (!found)
	{
		char	path[MAXPGPATH];
		char	temp[MAXPGPATH];
		ssize_t	sz;

		PG_TRY();
		{
			/* identify block device name */
			snprintf(path, sizeof(path),
					 "/sys/dev/block/%u:%u",
					 major, minor);
			if ((sz = readlink(path, temp, sizeof(temp))) < 0)
				strncpy(bdev->name, "????", sizeof(bdev->name));
			else
			{
				temp[sz] = '\0';
				strncpy(bdev->name, basename(temp), sizeof(bdev->name));
			}

			if (!__blkdev_setup_partition(bdev) &&
				!__blkdev_setup_md_raid0(bdev) &&
				!__blkdev_setup_raw_nvme(bdev))
			{
				bdev->optimal_gpus = NULL;
				elog(DEBUG2, "block device (%u,%u) has no optimal GPUs", major, minor);
			}
			bdev->is_valid = true;
		}
		PG_FINALLY();
		{
			/* clean up hash entry */
			hash_search(block_dev_htable, &hkey, HASH_REMOVE, NULL);
		}
		PG_END_TRY();
	}
	else if (!bdev->is_valid)
	{
		elog(ERROR, "Bug? block device is looped");
	}
	return bdev->optimal_gpus;
}

/*
 * apply_manual_optimal_gpus
 */
static void
apply_manual_optimal_gpus(const char *__config)
{
	const char *guc_name = "pg_strom.manual_optimal_gpus";
	char	   *config;
	char	   *tok, *saveptr;

	config = alloca(strlen(__config) + 1);
	strcpy(config, __config);
	for (tok = strtok_r(config, ",", &saveptr);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &saveptr))
	{
		Bitmapset  *optimal_gpus = NULL;
		char	   *path, *gpus;
		char	   *pos, *end;
		char	   *__tok, *__saveptr;
		ListCell   *lc;

		pos = strchr(tok, '=');
		if (!pos)
			elog(ERROR, "syntax error at %s [%s]", guc_name, __config);
		*pos++ = '\0';

		path = __trim(tok);
		gpus = __trim(pos);

		/* setup optimal GPUs */
		if (strcmp(gpus, "none") != 0)
		{
			for (__tok = strtok_r(gpus, ":", &__saveptr);
				 __tok != NULL;
				 __tok = strtok_r(NULL, ":", &__saveptr))
			{
				char   *__gpu = __trim(__tok);
				long	dindex;

				if (strncmp(__gpu, "gpu", 3) != 0 || __gpu[3] == '\0')
					elog(ERROR, "%s: invalid GPU name [%s]", guc_name, __gpu);
				dindex = strtol(__gpu+3, &end, 10);
				if (*end != '\0')
					elog(ERROR, "%s: invalid GPU name [%s]", guc_name, __gpu);
				if (dindex >= 0 && dindex < numGpuDevAttrs)
					optimal_gpus = bms_add_member(optimal_gpus, dindex);
				else
					elog(ERROR, "%s: GPU [%s] is out of range", guc_name, __gpu);
			}
		}

		if (strncmp(path, "nvme", 4) == 0)
		{
			strtol(path+4, &end, 10);
			if (path[4] == '\0' || *end != '\0')
				elog(ERROR, "%s: invalid device name [%s]", guc_name, path);
			foreach (lc, nvme_devices_list)
			{
				PciDevItem *pcie = lfirst(lc);

				if (strcmp(pcie->u.nvme.name, path) == 0)
				{
					pcie->optimal_gpus = optimal_gpus;
					break;
				}
			}
			if (!lc)
				elog(ERROR, "%s: [%s] was not found", guc_name, path);
		}
		else if (path[0] == '/')
		{
			struct stat	stat_buf;
			VfsDevItem *vfs;
			bool		found;

			if (stat(path, &stat_buf) != 0)
			{
				if (errno == ENOENT)
					elog(ERROR, "%s: directory '%s' was not found", guc_name, path);
				else
					elog(ERROR, "%s: failed on stat('%s'): %m", guc_name, path);
			}
			else if (!S_ISDIR(stat_buf.st_mode))
				elog(ERROR, "%s: path '%s' is not a directory", guc_name, path);
			else if (strlen(path) >= sizeof(vfs->dir) - 1)
				elog(ERROR, "%s: directory '%s' is too long", guc_name, path);

			if (!vfs_gpus_htable)
			{
				HASHCTL		hctl;

				memset(&hctl, 0, sizeof(HASHCTL));
				hctl.keysize = VfsDevItemKeySize;
				hctl.entrysize = sizeof(VfsDevItem);
				hctl.hcxt = TopMemoryContext;
				vfs_gpus_htable = hash_create("VFS-GPUs Hash Table",
											  512,
											  &hctl,
											  HASH_ELEM | HASH_STRINGS | HASH_CONTEXT);
			}
			vfs = hash_search(vfs_gpus_htable, &path, HASH_ENTER, &found);
			if (!found)
				vfs->optimal_gpus = optimal_gpus;
		}
		else
		{
			elog(ERROR, "%s: does not support relative path [%s]", guc_name, path);
		}
	}
}

/*
 * GetOptimalGpuForFile
 */
const Bitmapset *
GetOptimalGpuForFile(const char *pathname)
{
	struct stat	stat_buf;

	if (stat(pathname, &stat_buf) != 0)
	{
		elog(WARNING, "failed on stat('%s'): %m", pathname);
		return NULL;
	}

	if (vfs_gpus_htable)
	{
		char   *namebuf;
		char   *dir;

		if (pathname[0] == '/')
		{
			namebuf = alloca(strlen(pathname) + 1);
			strcpy(namebuf, pathname);
		}
		else
		{
			namebuf = alloca(strlen(DataDir) + strlen(pathname) + 2);
			sprintf(namebuf, "%s/%s", DataDir, pathname);
		}
		Assert(namebuf[0] == '/');

		if (S_ISDIR(stat_buf.st_mode))
			dir = namebuf;
		else
			dir = dirname(namebuf);

		for (;;)
		{
			VfsDevItem *vfs = hash_search(vfs_gpus_htable,
										  dir, HASH_FIND, NULL);
			if (vfs)
				return vfs->optimal_gpus;
			if (strcmp(dir, "/") != 0)
				break;
			dir = dirname(dir);
		}
	}
	return sysfs_lookup_optimal_gpus(major(stat_buf.st_dev),
									 minor(stat_buf.st_dev));
}

/*
 * GetOptimalGpuForTablespace
 */
static HTAB	   *tablespace_optimal_gpu_htable = NULL;
static bool		pgstrom_gpudirect_enabled;			/* GUC */
static int		__pgstrom_gpudirect_threshold_kb;	/* GUC */
#define pgstrom_gpudirect_threshold		((size_t)__pgstrom_gpudirect_threshold_kb << 10)

typedef struct
{
	Oid			tablespace_oid;
	bool		is_valid;
	Bitmapset	optimal_gpus;
} tablespace_optimal_gpu_hentry;

static void
tablespace_optimal_gpu_cache_callback(Datum arg, int cacheid, uint32 hashvalue)
{
	/* invalidate all the cached status */
	if (tablespace_optimal_gpu_htable)
	{
		hash_destroy(tablespace_optimal_gpu_htable);
		tablespace_optimal_gpu_htable = NULL;
	}
}

/*
 * GetOptimalGpuForTablespace
 */
static const Bitmapset *
GetOptimalGpuForTablespace(Oid tablespace_oid)
{
	tablespace_optimal_gpu_hentry *hentry;
	bool		found;

	if (!pgstrom_gpudirect_enabled)
		return NULL;

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	if (!tablespace_optimal_gpu_htable)
	{
		HASHCTL		hctl;
		int			nwords = (numGpuDevAttrs / BITS_PER_BITMAPWORD) + 1;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = MAXALIGN(offsetof(tablespace_optimal_gpu_hentry,
										   optimal_gpus.words[nwords]));
		tablespace_optimal_gpu_htable
			= hash_create("TablespaceOptimalGpu", 128,
						  &hctl, HASH_ELEM | HASH_BLOBS);
	}

	hentry = (tablespace_optimal_gpu_hentry *)
		hash_search(tablespace_optimal_gpu_htable,
					&tablespace_oid,
					HASH_ENTER,
					&found);
	if (!found || !hentry->is_valid)
	{
		char	   *pathname;
		const Bitmapset *optimal_gpus;

		Assert(hentry->tablespace_oid == tablespace_oid);
		pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);
		optimal_gpus = GetOptimalGpuForFile(pathname);
		if (bms_is_empty(optimal_gpus))
			hentry->optimal_gpus.nwords = 0;
		else
		{
			Assert(optimal_gpus->nwords <= (numGpuDevAttrs/BITS_PER_BITMAPWORD)+1);
			memcpy(&hentry->optimal_gpus, optimal_gpus,
				   offsetof(Bitmapset, words[optimal_gpus->nwords]));
		}
		hentry->is_valid = true;
	}
	Assert(hentry->is_valid);
	return (hentry->optimal_gpus.nwords > 0 ? &hentry->optimal_gpus : NULL);
}

/*
 * GetOptimalGpuForRelation
 */
const Bitmapset *
GetOptimalGpuForRelation(Relation relation)
{
	Oid		tablespace_oid;

	/* only heap relation */
	Assert(RelationGetForm(relation)->relam == HEAP_TABLE_AM_OID);
	tablespace_oid = RelationGetForm(relation)->reltablespace;
	if (!OidIsValid(tablespace_oid))
		tablespace_oid = DEFAULTTABLESPACE_OID;

	return GetOptimalGpuForTablespace(tablespace_oid);
}

/*
 * GetOptimalGpuForBaseRel - checks wthere the relation can use GPU-Direct SQL.
 * If possible, it returns bitmap of the optimal GPUs.
 */
const Bitmapset *
GetOptimalGpuForBaseRel(PlannerInfo *root, RelOptInfo *baserel)
{
	const Bitmapset *optimal_gpus;
	double		total_sz;

	if (!pgstrom_gpudirect_enabled)
		return NULL;
	if (baseRelIsArrowFdw(baserel))
	{
		if (pgstrom_gpudirect_enabled)
			return GetOptimalGpusForArrowFdw(root, baserel);
		return NULL;
	}
	total_sz = (size_t)baserel->pages * (size_t)BLCKSZ;
	if (total_sz < pgstrom_gpudirect_threshold)
		return NULL;	/* table is too small */

	optimal_gpus = GetOptimalGpuForTablespace(baserel->reltablespace);
	if (!bms_is_empty(optimal_gpus))
	{
		RangeTblEntry *rte = root->simple_rte_array[baserel->relid];
		char	relpersistence = get_rel_persistence(rte->relid);

		/* temporary table is not supported by GPU-Direct SQL */
		if (relpersistence != RELPERSISTENCE_PERMANENT &&
			relpersistence != RELPERSISTENCE_UNLOGGED)
			optimal_gpus = NULL;
	}
	return optimal_gpus;
}










/*
 * sysfs_preload_block_devices
 */
static void
sysfs_preload_block_devices(void)
{
	const char *dirname = "/sys/dev/block";
	DIR		   *dir;
	struct dirent *dent;

	dir = AllocateDir(dirname);
	if (!dir)
		return;

	while ((dent = ReadDir(dir, dirname)) != NULL)
	{
		const Bitmapset *optimal_gpus	__attribute__((unused));
		int		major;
		int		minor;

		if (sscanf(dent->d_name, "%u:%u", &major, &minor) != 2)
			continue;
		optimal_gpus = sysfs_lookup_optimal_gpus(major, minor);
#if 0
		{
			char	path[MAXPGPATH];
			char	temp[MAXPGPATH];
			char	buffer[2048];
			int		k, count;
			ssize_t	sz, off = 0;

			snprintf(path, sizeof(path), "%s/%s", dirname, dent->d_name);
			if ((sz = readlink(path, temp, sizeof(temp))) < 0)
			{
				off += snprintf(buffer+off, sizeof(buffer)-off,
								"blkdev(%s)", dent->d_name);
			}
			else
			{
				temp[sz] = '\0';
				off += snprintf(buffer+off, sizeof(buffer)-off,
								"%s", basename(temp));
			}

			if (bms_is_empty(optimal_gpus))
			{
				off += snprintf(buffer+off, sizeof(buffer)-off,
								" --> no optimal GPUs");
			}
			else
			{
				for (k = bms_next_member(optimal_gpus, -1), count=0;
					 k >= 0;
					 k = bms_next_member(optimal_gpus, k), count++)
				{
					off += snprintf(buffer+off, sizeof(buffer)-off,
									"%sGPU%d",
									count == 0 ? " --> " : ", ", k);
				}
			}
			elog(LOG, "%s", buffer);
		}
#endif
	}
	FreeDir(dir);
}

/*
 * pgstrom_init_gpudirect
 */
static void
pgstrom_init_gpudirect(void)
{
	bool	has_gpudirectsql = gpuDirectIsAvailable();

	DefineCustomBoolVariable("pg_strom.gpudirect_enabled",
							 "enables GPUDirect SQL",
							 NULL,
							 &pgstrom_gpudirect_enabled,
							 (has_gpudirectsql ? true : false),
							 (has_gpudirectsql ? PGC_SUSET : PGC_POSTMASTER),
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpudirect_threshold",
							"table-size threshold to use GPU-Direct SQL",
							NULL,
							&__pgstrom_gpudirect_threshold_kb,
							2097152,	/* 2GB */
							0,
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	/* tablespace cache */
	tablespace_optimal_gpu_htable = NULL;
	CacheRegisterSyscacheCallback(TABLESPACEOID,
								  tablespace_optimal_gpu_cache_callback,
								  (Datum) 0);
}

/*
 * pgstrom_init_pcie
 */
void
pgstrom_init_pcie(void)
{
	static char	   *pgstrom_manual_optimal_gpus = NULL;
	MemoryContext	memcxt;
	HASHCTL			hctl;

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = BlockDevItemKeySize;
	hctl.entrysize = sizeof(BlockDevItem);
	hctl.hcxt = TopMemoryContext;
	block_dev_htable = hash_create("Block Device Hash Table",
								   1024,
								   &hctl,
								   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	/*
	 * pg_strom.manual_optimal_xpus
	 *
	 * config := <token>[,<token> ...]
	 * token  := <path>=<xpus>
	 * path   := (<absolute dir>|<nvmeX>)
	 * gpus   := <gpuX>[:<gpuX>...]
	 *
	 * e.g) /mnt/data_1=gpu0,/mnt/data_2=gpu1:gpu2,nvme3=gpu3,/mnt/data_2/extra=gpu0
	 */
	DefineCustomStringVariable("pg_strom.manual_optimal_gpus",
							   "manual configuration of optimal GPUs",
							   NULL,
							   &pgstrom_manual_optimal_gpus,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	memcxt = MemoryContextSwitchTo(TopMemoryContext);
	sysfs_read_pcie_subtree();
	sysfs_setup_optimal_gpus();
	if (pgstrom_manual_optimal_gpus)
		apply_manual_optimal_gpus(pgstrom_manual_optimal_gpus);
	sysfs_print_pcie_subtree();
	sysfs_preload_block_devices();
	MemoryContextSwitchTo(memcxt);

	/*
	 * Special initialization for GPU-Direct SQL
	 */
	pgstrom_init_gpudirect();
}

