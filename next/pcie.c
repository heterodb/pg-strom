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
	char		name[80];
	Bitmapset  *optimal_gpus;
};

static HTAB	   *block_dev_htable = NULL;
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
__sysfs_read_pcie_gpu(PciDevItem *pci, const char *dirname)
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

		if (pci->pci_domain == gattrs->PCI_DOMAIN_ID &&
			pci->pci_bus_id == gattrs->PCI_BUS_ID &&
			pci->pci_dev_id == gattrs->PCI_DEVICE_ID &&
			(gattrs->MULTI_GPU_BOARD
			 ? pci->pci_func_id == gattrs->MULTI_GPU_BOARD_GROUP_ID
			 : pci->pci_func_id == 0))
		{
			pci->u.gpu.cuda_dindex = i;
			pci->u.gpu.gpu_dev_attrs = gattrs;

			gpu_devices_list = lappend(gpu_devices_list, pci);
			return true;
		}
	}
	return false;
}

static bool
__sysfs_read_pcie_nvme(PciDevItem *pci, const char *dirname)
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
		strncpy(pci->u.nvme.name, dent->d_name, sizeof(pci->u.nvme.name));

		snprintf(__path, sizeof(__path), "%s/nvme/%s/model",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pci->u.nvme.model, s, sizeof(pci->u.nvme.model));

		snprintf(__path, sizeof(__path), "%s/nvme/%s/serial",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pci->u.nvme.serial, s, sizeof(pci->u.nvme.serial));

		snprintf(__path, sizeof(__path), "%s/nvme/%s/firmware_rev",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pci->u.nvme.firmware, s, sizeof(pci->u.nvme.firmware));

		nvme_devices_list = lappend(nvme_devices_list, pci);
		FreeDir(dir);
		return true;
	}
	FreeDir(dir);
	return false;
}

static bool
__sysfs_read_pcie_hca(PciDevItem *pci, const char *dirname)
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
		strncpy(pci->u.hca.name, dent->d_name, sizeof(pci->u.hca.name));

		snprintf(__path, sizeof(__path), "%s/infiniband/%s/hca_type",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pci->u.hca.hca_type, s, sizeof(pci->u.hca.hca_type));

		snprintf(__path, sizeof(__path), "%s/infiniband/%s/node_guid",
				 dirname, dent->d_name);
		s = sysfs_read_line(__path);
		if (s)
			strncpy(pci->u.hca.node_guid, s, sizeof(pci->u.hca.node_guid));

		nvme_devices_list = lappend(nvme_devices_list, pci);
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
			PciDevItem *pci;
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

			pci = palloc0(sizeof(PciDevItem));
			pci->pci_domain = pci_domain;
			pci->pci_bus_id = pci_bus_id;
			pci->pci_dev_id = pci_dev_id;
			pci->pci_func_id = pci_func_id;
			pci->cpu_affinity = parent->cpu_affinity;
			pci->depth = parent->depth + 1;
			pci->distance = -1;
			pci->optimal_gpus = NULL;
			snprintf(path, sizeof(path),
					 "%s/%s",
					 dirname, dent->d_name);
			if (__sysfs_read_pcie_gpu(pci, path))
				pci->pci_kind = PCIDEV_KIND__GPU;
			else if (__sysfs_read_pcie_nvme(pci, path))
				pci->pci_kind = PCIDEV_KIND__NVME;
			else if (__sysfs_read_pcie_hca(pci, path))
				pci->pci_kind = PCIDEV_KIND__HCA;
			else
				pci->pci_kind = PCIDEV_KIND__UNKNOWN;
			/* child subtree */
			__sysfs_read_pcie_subtree(pci, path);
			if (pci->pci_kind != PCIDEV_KIND__UNKNOWN ||
				!dlist_is_empty(&pci->children))
				dlist_push_tail(&parent->children, &pci->chain);
		}
	}
	FreeDir(dir);
}

static void
sysfs_read_pcie_subtree(void)
{
	/* walks on the PCI-E bus tree for each root complex */
	const char *dirname = "/sys/devices";
	DIR		   *dir;
	struct dirent *dent;

	dir = AllocateDir(dirname);
	if (!dir)
		elog(ERROR, "failed on open '%s' directory: %m", dirname);
	while ((dent = ReadDir(dir, dirname)) != NULL)
	{
		PciDevItem *pci;
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
		pci = palloc0(sizeof(PciDevItem));
		pci->pci_domain = pci_domain;
		pci->pci_bus_id = pci_bus_id;
		pci->pci_kind = PCIDEV_KIND__UNKNOWN;
		pci->cpu_affinity = pstrdup(cpu_affinity);
		dlist_init(&pci->children);

		snprintf(path, sizeof(path),
				 "%s/%s", dirname, dent->d_name);
		__sysfs_read_pcie_subtree(pci, path);
		if (pci->pci_kind != PCIDEV_KIND__UNKNOWN || !dlist_is_empty(&pci->children))
			pcie_root_list = lappend(pcie_root_list, pci);
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
__sysfs_print_pcie_subtree(PciDevItem *pci, int depth)
{
	const GpuDevAttributes *gattrs;
	char		buffer[1024];
	size_t		off = 0;
	int			k, count;
	dlist_iter	iter;

	if (depth > 0)
	{
		for (int j=1; j < depth; j++)
			off += snprintf(buffer + off, sizeof(buffer) - off, "  ");
		off += snprintf(buffer + off, sizeof(buffer) - off, " - ");
	}
	off += snprintf(buffer + off, sizeof(buffer) - off,
					"[%04x:%02x:%02x.%d]",
					pci->pci_domain,
					pci->pci_bus_id,
					pci->pci_dev_id,
					pci->pci_func_id);
	switch (pci->pci_kind)
	{
		case PCIDEV_KIND__NVME:
			off += snprintf(buffer + off, sizeof(buffer) - off,
							" ... %s (%s",
							pci->u.nvme.name,
							pci->u.nvme.model);
			if (pci->distance >= 0)
			{
				for (k = bms_next_member(pci->optimal_gpus, -1), count=0;
					 k >= 0;
					 k = bms_next_member(pci->optimal_gpus, k), count++)
				{
					off += snprintf(buffer + off, sizeof(buffer) - off,
									"%sGPU%d",
									count == 0 ? "; " : ", ", k);
				}
				off += snprintf(buffer + off, sizeof(buffer) - off,
								" [dist=%d]", pci->distance);
			}
			off += snprintf(buffer + off, sizeof(buffer) - off, ")");
			break;
		case PCIDEV_KIND__GPU:
			gattrs = pci->u.gpu.gpu_dev_attrs;
			off += snprintf(buffer + off, sizeof(buffer) - off,
							" ... GPU%d (%s)",
							pci->u.gpu.cuda_dindex,
							gattrs->DEV_NAME);
			break;
		case PCIDEV_KIND__HCA:
			off += snprintf(buffer + off, sizeof(buffer) - off,
							" ... %s (%s",
							pci->u.hca.name,
							pci->u.hca.hca_type);
			if (pci->distance >= 0)
			{
				for (k = bms_next_member(pci->optimal_gpus, -1), count=0;
					 k >= 0;
					 k = bms_next_member(pci->optimal_gpus, k), count++)
				{
					off += snprintf(buffer + off, sizeof(buffer) - off,
									"%sGPU%d",
									count == 0 ? "; " : ", ", k);
				}
				off += snprintf(buffer + off, sizeof(buffer) - off,
								" [dist=%d]", pci->distance);
			}
			off += snprintf(buffer + off, sizeof(buffer) - off, ")");
			break;
		default:
			break;
	}
	elog(LOG, "%s", buffer);

	dlist_foreach (iter, &pci->children)
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
		PciDevItem *pci = lfirst(lc);

		__sysfs_print_pcie_subtree(pci, 0);
	}
}

/*
 * pgstrom_init_pcie
 */
void
pgstrom_init_pcie(void)
{
	HASHCTL		hctl;

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = BlockDevItemKeySize;
	hctl.entrysize = sizeof(BlockDevItem);
	hctl.hcxt = CacheMemoryContext;
	block_dev_htable = hash_create("Block Device Hash Table",
								   1024,
								   &hctl,
								   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	sysfs_read_pcie_subtree();
	sysfs_setup_optimal_gpus();
	sysfs_print_pcie_subtree();

}





