/*
 * nvme_strom.c
 *
 * Routines related to NVME-SSD devices
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#include "pg_strom.h"

/*
 * NvmeAttributes - properties of NVMe disks
 */
typedef struct NvmeAttributes
{
	cl_int		nvme_major;			/* major device number */
	cl_int		nvme_minor;			/* minor device number */
	char		nvme_name[64];		/* nvme device name */
	char		nvme_serial[128];	/* serial number in sysfs */
	char		nvme_model[256];	/* model name in sysfs */
	cl_int		nvme_pcie_domain;	/* DDDD of DDDD:bb:dd.f */
	cl_int		nvme_pcie_bus_id;	/* bb of DDDD:bb:dd.f */
	cl_int		nvme_pcie_dev_id;	/* dd of DDDD:bb:dd.f */
	cl_int		nvme_pcie_func_id;	/* f of DDDD:bb:dd.f */
	cl_int		numa_node_id;		/* numa node id */
	cl_int		nvme_optimal_gpu;	/* optimal GPU index */
	cl_int		nvme_distances[FLEXIBLE_ARRAY_MEMBER];	/* distance map */
} NvmeAttributes;

/*
 * sysfs_read_pcie_attrs
 */
struct PCIDevEntry
{
	struct PCIDevEntry *parent;
	int		domain;
	int		bus_id;
	int		dev_id;
	int		func_id;
	int		depth;
	DevAttributes *gpu_attr;	/* if GPU device */
	NvmeAttributes *nvme_attr;	/* if NVMe device */
	List *children;
};
typedef struct PCIDevEntry	PCIDevEntry;

/* static variables/functions */
static HTAB	   *nvmeHash = NULL;
static char	   *pgstrom_gpudirect_driver;	/* GUC */
static bool		pgstrom_gpudirect_enabled;	/* GUC */
static int		pgstrom_gpudirect_threshold_kb;	/* GUC */


static char		   *nvme_manual_distance_map;	/* GUC */
static void			apply_nvme_manual_distance_map(void);
static bool			sysfs_read_pcie_root_complex(const char *dirname,
												 const char *my_name,
												 List **p_pcie_root);
/*
 * nvme_strom_threshold
 */
Size
nvme_strom_threshold(void)
{
	return (Size)pgstrom_gpudirect_threshold_kb << 10;
}

/*
 * nvme_strom_ioctl
 */
int
nvme_strom_ioctl(int cmd, void *arg)
{
	static int	fdesc_nvme_strom = -1;

	if (fdesc_nvme_strom < 0)
	{
		fdesc_nvme_strom = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
		if (fdesc_nvme_strom < 0)
			return -1;
	}
	return ioctl(fdesc_nvme_strom, cmd, arg);
}

/*
 * sysfs_read_line
 */
static const char *
sysfs_read_line(const char *path, bool abort_on_error)
{
	static char linebuf[2048];
	int		fdesc;
	char   *pos;
	ssize_t	sz;

	fdesc = open(path, O_RDONLY);
	if (fdesc < 0)
	{
		if (abort_on_error)
			elog(ERROR, "failed on open('%s'): %m", path);
		return NULL;
	}
retry:
	sz = read(fdesc, linebuf, sizeof(linebuf) - 1);
	if (sz < 0)
	{
		int		errno_saved = errno;

		if (errno == EINTR)
			goto retry;
		close(fdesc);
		errno = errno_saved;
		if (abort_on_error)
			elog(ERROR, "failed on read('%s'): %m", path);
		return NULL;
	}
	close(fdesc);
	linebuf[sz] = '\0';
	pos = linebuf + sz - 1;
	while (pos >= linebuf && isspace(*pos))
		*pos-- = '\0';
	return linebuf;
}

/*
 * sysfs_read_nvme_attrs
 */
static bool
sysfs_read_nvme_attrs(NvmeAttributes *nvme,
					  const char *sysfs_base,
					  const char *sysfs_nvme)
{
	char		namebuf[MAXPGPATH];
	const char *value;

	memset(nvme, 0, offsetof(NvmeAttributes,
							 nvme_distances));
	/* fetch major:minor */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/dev",
			 sysfs_base, sysfs_nvme);
	value = sysfs_read_line(namebuf, false);
	if (!value || sscanf(value, "%u:%u",
						 &nvme->nvme_major,
						 &nvme->nvme_minor) != 2)
		return false;

	/* fetch device (namespace) name */
	strncpy(nvme->nvme_name, sysfs_nvme, 64);

	/* fetch serial */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/device/serial",
			 sysfs_base, sysfs_nvme);
	value = sysfs_read_line(namebuf, false);
	strncpy(nvme->nvme_serial, value ? value : "Unknown", 128);

	/* fetch model */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/device/model",
			 sysfs_base, sysfs_nvme);
	value = sysfs_read_line(namebuf, false);
	strncpy(nvme->nvme_model, value ? value : "Unknown", 256);

	/* check whether it is PCIE or not */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/device/transport",
			 sysfs_base, sysfs_nvme);
	value = sysfs_read_line(namebuf, false);
	if (!value || strcmp(value, "pcie") != 0)
		goto bailout;

	/* fetch PCI-E Bus ID */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/device/address",
			 sysfs_base, sysfs_nvme);
	value = sysfs_read_line(namebuf, false);
	if (!value || sscanf(value, "%x:%02x:%02x.%d",
						 &nvme->nvme_pcie_domain,
						 &nvme->nvme_pcie_bus_id,
						 &nvme->nvme_pcie_dev_id,
						 &nvme->nvme_pcie_func_id) != 4)
		goto bailout;

	/* fetch numa_node_id */
	snprintf(namebuf, sizeof(namebuf),
			 "/sys/bus/pci/devices/%04x:%02x:%02x.%d/numa_node",
			 nvme->nvme_pcie_domain,
			 nvme->nvme_pcie_bus_id,
			 nvme->nvme_pcie_dev_id,
			 nvme->nvme_pcie_func_id);
	value = sysfs_read_line(namebuf, false);
	if (!value || (nvme->numa_node_id = atoi(value)) < 0)
		goto bailout;

	return true;

bailout:
	nvme->nvme_pcie_domain = -1;
	nvme->nvme_pcie_bus_id = -1;
	nvme->nvme_pcie_dev_id = -1;
	nvme->nvme_pcie_func_id = -1;
	nvme->numa_node_id = -1;

	return true;
}

#ifdef WITH_CUFILE
/*
 * sysfs_read_sfdv_attrs
 */
static bool
sysfs_read_sfdv_attrs(NvmeAttributes *nvme,
					  const char *sysfs_base,
					  const char *sysfs_sfdv)
{
	char		namebuf[MAXPGPATH];
	const char *value;

	memset(nvme, 0, offsetof(NvmeAttributes, nvme_distances));

	/* fetch major:minor */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/dev",
			 sysfs_base, sysfs_sfdv);
	value = sysfs_read_line(namebuf, false);
	if (!value || sscanf(value, "%u:%u",
						 &nvme->nvme_major,
						 &nvme->nvme_minor) != 2)
		return false;

	/* fetch device (namespace) name */
	strncpy(nvme->nvme_name, sysfs_sfdv, 64);

	/* fetch serial */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/serial",
			 sysfs_base, sysfs_sfdv);
	value = sysfs_read_line(namebuf, false);
	strncpy(nvme->nvme_serial, value ? value : "Unknown", 128);

	/* fetch model */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/model",
			 sysfs_base, sysfs_sfdv);
	value = sysfs_read_line(namebuf, false);
	strncpy(nvme->nvme_model, value ? value : "Unknown", 256);

	/* fetch PCI-E Bus ID */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/bus_info",
			 sysfs_base, sysfs_sfdv);
	value = sysfs_read_line(namebuf, false);
	if (!value || sscanf(value, "%x:%02x:%02x.%d",
						 &nvme->nvme_pcie_domain,
						 &nvme->nvme_pcie_bus_id,
						 &nvme->nvme_pcie_dev_id,
						 &nvme->nvme_pcie_func_id) != 4)
		goto bailout;

	/* fetch numa_node_id */
	snprintf(namebuf, sizeof(namebuf),
			 "/sys/bus/pci/devices/%04x:%02x:%02x.%d/numa_node",
			 nvme->nvme_pcie_domain,
			 nvme->nvme_pcie_bus_id,
			 nvme->nvme_pcie_dev_id,
			 nvme->nvme_pcie_func_id);
	value = sysfs_read_line(namebuf, false);
	if (!value || (nvme->numa_node_id = atoi(value)) < 0)
		goto bailout;

	return true;

bailout:
	nvme->nvme_pcie_domain = -1;
	nvme->nvme_pcie_bus_id = -1;
	nvme->nvme_pcie_dev_id = -1;
	nvme->nvme_pcie_func_id = -1;
	nvme->numa_node_id = -1;
	return true;
}
#endif	/* WITH_CUFILE */

/*
 * sysfs_read_block_attrs
 */
static void
sysfs_read_block_attrs(void)
{
	NvmeAttributes *nvme;
	NvmeAttributes	temp;
	const char *dirname = "/sys/class/block";
	DIR		   *dir;
	struct dirent *dent;
	
	dir = AllocateDir(dirname);
	if (!dir)
		elog(ERROR, "failed on AllocateDir('%s'): %m", dirname);

	while ((dent = ReadDir(dir, dirname)) != NULL)
	{
		const char *start, *pos;
		bool		found;
		int			i;

		if (strncmp("nvme", dent->d_name, 4) == 0)
		{
			/* only nvme[0-9]+n[0-9]+ devices */
			pos = start = dent->d_name + 4;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != 'n')
				continue;
			start = ++pos;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != '\0')
				continue;

			if (!sysfs_read_nvme_attrs(&temp, dirname, dent->d_name))
				continue;
		}
#ifdef WITH_CUFILE
		else if (strncmp("sfdv", dent->d_name, 4) == 0)
		{
			/* only sfdv[0-9]+n[0-9]+ devices */
			pos = start = dent->d_name + 4;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != 'n')
				continue;
			start = ++pos;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != '\0')
				continue;

			if (!sysfs_read_sfdv_attrs(&temp, dirname, dent->d_name))
				continue;
		}
#endif
		else
		{
			/* not a supported device type */
			continue;
		}

		if (!nvmeHash)
		{
			HASHCTL		hctl;

			memset(&hctl, 0, sizeof(HASHCTL));
			hctl.keysize = offsetof(NvmeAttributes, nvme_name);
			hctl.entrysize = offsetof(NvmeAttributes,
									  nvme_distances[numDevAttrs]);
			hctl.hcxt = TopMemoryContext;
			nvmeHash = hash_create("NVME Drives", 256, &hctl,
								   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
		}
		nvme = hash_search(nvmeHash, &temp, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? detects duplicate NVMe SSD (%d,%d)",
				 nvme->nvme_major, nvme->nvme_minor);
		Assert(nvme->nvme_major == temp.nvme_major &&
			   nvme->nvme_minor == temp.nvme_minor);
		memcpy(nvme, &temp, offsetof(NvmeAttributes,
									 nvme_distances));
		nvme->nvme_optimal_gpu = -1;
		for (i=0; i < numDevAttrs; i++)
			nvme->nvme_distances[i] = -1;
	}
	FreeDir(dir);
}

/*
 * sysfs_read_pcie_attrs
 */
static PCIDevEntry *
sysfs_read_pcie_attrs(const char *dirname, const char *my_name,
					  PCIDevEntry *parent, int depth,
					  List **p_pcie_root)
{
	PCIDevEntry *entry;
	DIR		   *dir;
	struct dirent *dent;
	char		path[MAXPGPATH];
	int			index;

	entry = palloc0(sizeof(PCIDevEntry));
	entry->parent = parent;
	if (!parent)
	{
		Assert(strncmp("pci", my_name, 3) == 0);
		if (sscanf(my_name+3, "%x:%02x",
				   &entry->domain,
				   &entry->bus_id) != 2)
			elog(ERROR, "unexpected sysfs entry: %s/%s", dirname, my_name);
		entry->dev_id = -1;		/* invalid */
		entry->func_id = -1;	/* invalid */
		entry->depth = depth;
	}
	else
	{
		if (sscanf(my_name, "%x:%02x:%02x.%d",
				   &entry->domain,
				   &entry->bus_id,
				   &entry->dev_id,
				   &entry->func_id) != 4)
			elog(ERROR, "unexpected sysfs entry: %s/%s", dirname, my_name);
		entry->depth = depth;

		/* Is it a GPU device? */
		for (index=0; index < numDevAttrs; index++)
		{
			DevAttributes *dattr = &devAttrs[index];

			if (entry->domain == dattr->PCI_DOMAIN_ID &&
				entry->bus_id == dattr->PCI_BUS_ID &&
				entry->dev_id == dattr->PCI_DEVICE_ID &&
				entry->func_id == (dattr->MULTI_GPU_BOARD ?
								   dattr->MULTI_GPU_BOARD_GROUP_ID : 0))
			{
				entry->gpu_attr = dattr;
				break;
			}
		}
		/* Elsewhere, is it a NVMe device? */
		if (!entry->gpu_attr)
		{
			NvmeAttributes *nvattr;
			HASH_SEQ_STATUS hseq;

			hash_seq_init(&hseq, nvmeHash);
			while ((nvattr = hash_seq_search(&hseq)) != NULL)
			{
				if (entry->domain == nvattr->nvme_pcie_domain &&
					entry->bus_id == nvattr->nvme_pcie_bus_id &&
					entry->dev_id == nvattr->nvme_pcie_dev_id &&
					entry->func_id == nvattr->nvme_pcie_func_id)
				{
					entry->nvme_attr = nvattr;
					hash_seq_term(&hseq);
					break;
				}
			}
		}
	}
	/* walk down the PCIe device tree */
	snprintf(path, sizeof(path), "%s/%s", dirname, my_name);
	dir = opendir(path);
	if (!dir)
		elog(ERROR, "failed on opendir('%s'): %m", dirname);
	while ((dent = readdir(dir)) != NULL)
	{
		PCIDevEntry *temp;
		const char *delim = "::.";
		char	   *pos;

		/* pcixxxx:xx sub-root? */
		if (sysfs_read_pcie_root_complex(path, dent->d_name,
										 p_pcie_root))
			continue;

		/* elsewhere, xxxx:xx:xx.x? */
		for (pos = dent->d_name; *pos != '\0'; pos++)
		{
			if (*pos == *delim)
				delim++;
			else if (*delim != '\0' ? !isxdigit(*pos) : !isdigit(*pos))
				break;
		}
		if (*pos == '\0' && *delim == '\0')
		{
			temp = sysfs_read_pcie_attrs(path, dent->d_name, entry, depth+1,
										 p_pcie_root);
			if (temp != NULL)
				entry->children = lappend(entry->children, temp);
		}
	}
	closedir(dir);

	if (entry->gpu_attr == NULL &&
		entry->nvme_attr == NULL &&
		entry->children == NIL)
	{
		pfree(entry);
		return NULL;
	}
	return entry;
}

/*
 * sysfs_read_pcie_root_complex
 */
static bool
sysfs_read_pcie_root_complex(const char *dirname,
							 const char *my_name,
							 List **p_pcie_root)
{
	const char	   *delim = ":";
	const char	   *pos = my_name;
	PCIDevEntry	   *entry;

	if (strncmp("pci", my_name, 3) == 0)
	{
		for (pos = my_name+3; *pos != '\0'; pos++)
		{
			if (*pos == *delim)
				delim++;
			else if (!isxdigit(*pos))
				break;
		}
		if (*pos == '\0' && *delim == '\0')
		{
			entry = sysfs_read_pcie_attrs(dirname, my_name, NULL, 0,
										  p_pcie_root);
			if (entry)
				*p_pcie_root = lappend(*p_pcie_root, entry);
			return true;
		}
	}
	return false;
}

/*
 * print_pcie_device_tree
 */
static void
print_pcie_device_tree(PCIDevEntry *entry, int indent)
{
	ListCell   *lc;

	if (!entry->parent)
		elog(LOG, "%*s PCIe[%04x:%02x]",
			 2 * indent, "- ",
			 entry->domain,
			 entry->bus_id);
	else if (entry->gpu_attr)
		elog(LOG, "%*s PCIe(%04x:%02x:%02x.%d) GPU%d (%s)",
			 2 * indent, "- ",
			 entry->domain,
			 entry->bus_id,
			 entry->dev_id,
			 entry->func_id,
			 entry->gpu_attr->DEV_ID,
			 entry->gpu_attr->DEV_NAME);
	else if (entry->nvme_attr)
		elog(LOG, "%*s PCIe(%04x:%02x:%02x.%d) %s (%s)",
			 2 * indent, "- ",
			 entry->domain,
			 entry->bus_id,
			 entry->dev_id,
			 entry->func_id,
			 entry->nvme_attr->nvme_name,
			 entry->nvme_attr->nvme_model);
	else
		elog(LOG, "%*s PCIe(%04x:%02x:%02x.%d)",
			 2 * indent, "- ",
			 entry->domain,
			 entry->bus_id,
			 entry->dev_id,
			 entry->func_id);

	foreach (lc, entry->children)
		print_pcie_device_tree(lfirst(lc), indent+2);
}

/*
 * calculate_nvme_distance_map
 */
static int
calculate_nvme_distance_map(List *pcie_siblings, int depth,
							DevAttributes *gpu, bool *p_gpu_found,
							NvmeAttributes *nvme, bool *p_nvme_found)
{
	int			gpu_depth = -1;
	int			nvme_depth = -1;
	int			dist;
	ListCell   *lc;

	foreach (lc, pcie_siblings)
	{
		PCIDevEntry *entry = lfirst(lc);

		if (entry->gpu_attr == gpu)
		{
			Assert(entry->nvme_attr == NULL &&
				   entry->children == NIL);
			Assert(gpu_depth < 0);
			gpu_depth = depth;
		}
		else if (entry->nvme_attr == nvme)
		{
			Assert(entry->gpu_attr == NULL &&
				   entry->children == NIL);
			Assert(nvme_depth < 0);
			nvme_depth = depth;
		}
		else if (entry->children != NIL)
		{
			bool	gpu_found = false;
			bool	nvme_found = false;
			int		dist;

			dist = calculate_nvme_distance_map(entry->children,
											   depth+1,
											   gpu, &gpu_found,
											   nvme, &nvme_found);
			if (gpu_found && nvme_found)
			{
				*p_gpu_found	= true;
				*p_nvme_found	= true;
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
	}

	if (gpu_depth >= 0 && nvme_depth >= 0)
	{
		dist = ((gpu_depth - depth) +
				(nvme_depth - depth) +
				(depth == 1 ? 2 : 1));
	}
	else if (gpu_depth >= 0)
	{
		dist = gpu_depth;
	}
	else if (nvme_depth >= 0)
	{
		dist = nvme_depth;
	}
	else
		dist = -1;

	*p_gpu_found = (gpu_depth >= 0);
	*p_nvme_found = (nvme_depth >= 0);

	return dist;
}

/*
 * setup_nvme_distance_map
 */
static void
setup_nvme_distance_map(void)
{
	NvmeAttributes *nvme;
	const char *dirname;
	DIR		   *dir;
	struct dirent *dent;
	List	   *pcie_root = NIL;
	ListCell   *lc;
	int			i, dist;
	HASH_SEQ_STATUS hseq;

	sysfs_read_block_attrs();
	if (!nvmeHash)
		return;		/* No NVME Drives found */

	/*
	 * Walk on the PCIe bus tree
	 */
	dirname = "/sys/devices";
	dir = AllocateDir(dirname);
	if (!dir)
		elog(ERROR, "failed on opendir('%s'): %m", dirname);
	while ((dent = ReadDir(dir, dirname)) != NULL)
	{
		sysfs_read_pcie_root_complex(dirname, dent->d_name, &pcie_root);
	}
	FreeDir(dir);

	/*
	 * calculation of SSD<->GPU distance map
	 */
	hash_seq_init(&hseq, nvmeHash);
	while ((nvme = hash_seq_search(&hseq)) != NULL)
	{
		int		optimal_gpu = -1;
		int		optimal_dist = INT_MAX;

		for (i=0; i < numDevAttrs; i++)
		{
			DevAttributes  *gpu = &devAttrs[i];
			bool	gpu_found = false;
			bool	nvme_found = false;

			if (gpu->NUMA_NODE_ID != nvme->numa_node_id)
			{
				nvme->nvme_distances[i] = -1;
				continue;
			}
			dist = calculate_nvme_distance_map(pcie_root, 1,
											   gpu, &gpu_found,
											   nvme, &nvme_found);
			if (gpu_found && nvme_found)
			{
				nvme->nvme_distances[i] = dist;
				if (dist < optimal_dist)
				{
					optimal_gpu = i;
					optimal_dist = dist;
				}
			}
			else
				nvme->nvme_distances[i] = -1;
		}
		nvme->nvme_optimal_gpu = optimal_gpu;
	}
	/* Print PCIe tree */
	foreach (lc, pcie_root)
		print_pcie_device_tree(lfirst(lc), 2);

	/* Overwrite the distance map by manual configuration */
	if (nvme_manual_distance_map)
		apply_nvme_manual_distance_map();

	/* Print GPU<->SSD Distance Matrix */
	if (numDevAttrs > 0 && nvmeHash != NULL)
	{
		HASH_SEQ_STATUS	hseq;
		StringInfoData	str;
		StringInfoData	dev;

		initStringInfo(&str);
		initStringInfo(&dev);

		for (i=0; i < numDevAttrs; i++)
			appendStringInfo(&str, "   GPU%d  ", i);
		elog(LOG, "GPU<->SSD Distance Matrix");
		elog(LOG, "        %s", str.data);

		hash_seq_init(&hseq, nvmeHash);
		while ((nvme = hash_seq_search(&hseq)) != NULL)
		{
			if (nvme->nvme_optimal_gpu < 0)
				appendStringInfo(&dev, "%s%s",
								 dev.len > 0 ? ", " : "",
								 nvme->nvme_name);
			resetStringInfo(&str);
			appendStringInfo(&str, "   %6s", nvme->nvme_name);
			for (i=0; i < numDevAttrs; i++)
			{
				int		dist = nvme->nvme_distances[i];

				if (nvme->nvme_optimal_gpu == i)
					appendStringInfo(&str, "  (%4d)", dist);
				else
					appendStringInfo(&str, "   %4d ", dist);
			}
			elog(LOG, "%s", str.data);
		}

		if (dev.len > 0)
		{
			ereport(LOG,
					(errmsg("Optimal GPUs are uncertain for NVME devices: %s",
							dev.data),
					 errhint("review your 'pg_strom.nvme_distance_map' configuration if these devices may be used in PostgreSQL")));
		}
		pfree(str.data);
		pfree(dev.data);
	}
}

/*
 * apply_nvme_manual_distance_map
 */
static void
apply_nvme_manual_distance_map(void)
{
	char	   *config = pstrdup(nvme_manual_distance_map);
	char	   *token = NULL;
	char	   *dev1, *dev2;
	char	   *pos1, *pos2;
	char	   *c;

	for (token = strtok_r(config, ",", &pos1);
		 token != NULL;
		 token = strtok_r(NULL, ",", &pos1))
	{
		int		cuda_dindex;
		bool	found = false;

		token = __trim(token);

		if ((dev1 = strtok_r(token, ":", &pos2)) == NULL ||
			(dev2 = strtok_r(NULL, ":", &pos2)) == NULL ||
			strtok_r(NULL, ":", &pos2) != NULL)
			goto syntax_error;
		dev1 = __trim(dev1);
		dev2 = __trim(dev2);

		/* nvme device index */
		if (strncasecmp(dev1, "nvme", 4) != 0)
			goto syntax_error;
		for (c = dev1+4; isdigit(*c); c++);
		if (c == dev1+4 || *c != '\0')
			goto syntax_error;

		/* GPU device index */
		if (strncasecmp(dev2, "gpu", 3) != 0)
			goto syntax_error;
		for (c = dev2+3; isdigit(*c); c++);
		if (c == dev2+3 || *c != '\0')
			goto syntax_error;
		cuda_dindex = atoi(dev2+3);
		if (cuda_dindex < 0 || cuda_dindex >= numDevAttrs)
			elog(ERROR, "GPU device '%s' was not found", dev2);

		if (nvmeHash)
		{
			NvmeAttributes *nvme;
			HASH_SEQ_STATUS	hseq;
			char		temp[128];
			size_t		sz;
			
			sz = snprintf(temp, sizeof(temp), "%sn", dev1);
			
			hash_seq_init(&hseq, nvmeHash);
			while ((nvme = hash_seq_search(&hseq)) != NULL)
			{
				if (strncasecmp(temp, nvme->nvme_name, sz) == 0)
				{
					nvme->nvme_optimal_gpu = cuda_dindex;
					found = true;
				}
			}
		}
		if (!found)
			elog(ERROR, "NVME device '%s' was not found", dev1);
	}
	return;

syntax_error:
	elog(ERROR, "wrong configuration at %ld character of '%s'",
		 token - config, nvme_manual_distance_map);
}

/*
 * TablespaceCanUseNvmeStrom
 */
typedef struct
{
	Oid		tablespace_oid;
	int		optimal_gpu;
} tablespace_optimal_gpu_hentry;

typedef struct
{
	dev_t	st_dev;		/* may be a partition device */
	int		optimal_gpu;
} filesystem_optimal_gpu_hentry;

static HTAB	   *tablespace_optimal_gpu_htable = NULL;
static HTAB	   *filesystem_optimal_gpu_htable = NULL;

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
 * __GetOptimalGpuForBlockDevice
 */
static int
__GetOptimalGpuForBlockDevice(int major, int minor)
{
	NvmeAttributes	key;
	NvmeAttributes *nvme;

	memset(&key, 0, sizeof(NvmeAttributes));
	key.nvme_major = major;
	key.nvme_minor = minor;

	if (!nvmeHash)
		return -1;
	nvme = hash_search(nvmeHash, &key, HASH_FIND, NULL);
	if (!nvme)
		return -1;
	return nvme->nvme_optimal_gpu;
}

/*
 * __GetOptimalGpuForRaidVolume
 */
static int
__GetOptimalGpuForRaidVolume(const char *sysfs_base)
{
	char		namebuf[MAXPGPATH];
	const char *value;
	int			optimal_gpu = -1;
	ssize_t		sz;
	DIR		   *dir;
	struct dirent *dent;

	snprintf(namebuf, sizeof(namebuf), "%s/level", sysfs_base);
	value = sysfs_read_line(namebuf, true);
	if (strcmp(value, "raid0") != 0)
		return -1;		/* unsupported md-raid level */

	snprintf(namebuf, sizeof(namebuf), "%s/chunk_size", sysfs_base);
	sz = atol(sysfs_read_line(namebuf, true));
	if (sz < PAGE_SIZE || (sz & (PAGE_SIZE - 1)) != 0)
		return -1;		/* unsupported md-raid chunk-size */

	dir = AllocateDir(sysfs_base);
	if (!dir)
		elog(ERROR, "failed on AllocateDir('%s'): %m", sysfs_base);
	
	while ((dent = ReadDir(dir, sysfs_base)) != NULL)
	{
		if (dent->d_name[0] == 'r' &&
			dent->d_name[1] == 'd' &&
			isdigit(dent->d_name[2]))
		{
			int		__major;
			int		__minor;
			int		__dindex;

			snprintf(namebuf, sizeof(namebuf),
					 "%s/%s/block/dev",
					 sysfs_base, dent->d_name);
			value = sysfs_read_line(namebuf, true);
			if (sscanf(value, "%d:%d", &__major, &__minor) != 2)
				elog(ERROR, "failed on parse '%s' [%s]",
					 namebuf, value);

			snprintf(namebuf, sizeof(namebuf),
					 "%s/%s/block/partition",
					 sysfs_base, dent->d_name);
			value = sysfs_read_line(namebuf, false);
			if (value && atoi(value) != 0)
			{
				/* a partitioned volume */
				snprintf(namebuf, sizeof(namebuf),
						 "/sys/dev/block/%d:%d/../dev",
						 __major, __minor);
				value = sysfs_read_line(namebuf, true);
				if (sscanf(value, "%d:%d", &__major, &__minor) != 2)
					elog(ERROR, "failed on parse '%s' [%s]",
						 namebuf, value);
			}
			__dindex = __GetOptimalGpuForBlockDevice(__major, __minor);
			if (__dindex < 0)
			{
				optimal_gpu = -1;
				break;		/* no optimal GPU */
			}
			else if (optimal_gpu < 0)
				optimal_gpu = __dindex;
			else if (optimal_gpu != __dindex)
			{
				optimal_gpu = -1;
				break;		/* no optimal GPU */
			}
		}
	}
	FreeDir(dir);

	return optimal_gpu;
}

/*
 * GetOptimalGpuForFile
 */
int
GetOptimalGpuForFile(File fdesc)
{
	filesystem_optimal_gpu_hentry *hentry;
	struct stat	stat_buf;
	bool		found;

	if (fstat(FileGetRawDesc(fdesc), &stat_buf) != 0)
		elog(ERROR, "failed on fstat('%s'): %m", FilePathName(fdesc));
	if (!filesystem_optimal_gpu_htable)
	{
		HASHCTL		ctl;

		memset(&ctl, 0, sizeof(HASHCTL));
		ctl.keysize = sizeof(dev_t);
		ctl.entrysize = sizeof(filesystem_optimal_gpu_hentry);
		filesystem_optimal_gpu_htable
			= hash_create("FileSystemOptimalGPU", 128,
						  &ctl, HASH_ELEM | HASH_BLOBS);
	}
	hentry = hash_search(filesystem_optimal_gpu_htable,
						 &stat_buf.st_dev,
						 HASH_ENTER,
						 &found);
	if (!found)
	{
		int		optimal_gpu = -1;
		int		major = major(stat_buf.st_dev);
		int		minor = minor(stat_buf.st_dev);
		char	namebuf[MAXPGPATH];
		const char *value;

		PG_TRY();
		{
			/* check whether the file is on partition volume */
			snprintf(namebuf, sizeof(namebuf),
					 "/sys/dev/block/%d:%d/partition",
					 major, minor);
			value = sysfs_read_line(namebuf, false);
			if (value && atoi(value) != 0)
			{
				/* lookup the mother block device */
				snprintf(namebuf, sizeof(namebuf),
						 "/sys/dev/block/%d:%d/../dev",
						 major, minor);
				value = sysfs_read_line(namebuf, true);
				if (sscanf(value, "%d:%d", &major, &minor) != 2)
					elog(ERROR, "failed on parse '%s' [%s]",
						 namebuf, value);
			}
			/* check whether the file is on md-raid volumn */
			snprintf(namebuf, sizeof(namebuf),
					 "/sys/dev/block/%d:%d/md", major, minor);
			if (stat(namebuf, &stat_buf) == 0)
			{
				if ((stat_buf.st_mode & S_IFMT) != S_IFDIR)
					elog(ERROR, "sysfs entry '%s' is not a directory", namebuf);
				optimal_gpu = __GetOptimalGpuForRaidVolume(namebuf);
			}
			else if (errno == ENOENT)
			{
				/* not a md-RAID device */
				optimal_gpu = __GetOptimalGpuForBlockDevice(major, minor);
			}
			else
			{
				elog(ERROR, "failed on stat('%s'): %m", namebuf);
			}
			hentry->optimal_gpu = optimal_gpu;
		}
		PG_CATCH();
		{
			hash_search(filesystem_optimal_gpu_htable,
						&stat_buf.st_dev,
						HASH_REMOVE,
						NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return hentry->optimal_gpu;
}

static cl_int
GetOptimalGpuForTablespace(Oid tablespace_oid)
{
	tablespace_optimal_gpu_hentry *hentry;
	char   *pathname;
	File	fdesc;
	bool	found;

	if (!pgstrom_gpudirect_enabled)
		return -1;

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	if (!tablespace_optimal_gpu_htable)
	{
		HASHCTL		ctl;

		memset(&ctl, 0, sizeof(HASHCTL));
		ctl.keysize = sizeof(Oid);
		ctl.entrysize = sizeof(tablespace_optimal_gpu_hentry);
		tablespace_optimal_gpu_htable
			= hash_create("TablespaceOptimalGpu", 128,
						  &ctl, HASH_ELEM | HASH_BLOBS);
		CacheRegisterSyscacheCallback(TABLESPACEOID,
									  tablespace_optimal_gpu_cache_callback,
									  (Datum) 0);
	}
	hentry = (tablespace_optimal_gpu_hentry *)
		hash_search(tablespace_optimal_gpu_htable,
					&tablespace_oid,
					HASH_ENTER,
					&found);
	if (!found)
	{
		PG_TRY();
		{
			Assert(hentry->tablespace_oid == tablespace_oid);
			hentry->optimal_gpu = -1;

			pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);
			fdesc = PathNameOpenFile(pathname, O_RDONLY | O_DIRECTORY);
			if (fdesc < 0)
			{
				elog(WARNING, "failed on open('%s') of tablespace %u: %m",
					 pathname, tablespace_oid);
			}
			else
			{
				hentry->optimal_gpu = GetOptimalGpuForFile(fdesc);
				FileClose(fdesc);
			}
		}
		PG_CATCH();
		{
			hash_search(tablespace_optimal_gpu_htable,
						&tablespace_oid,
						HASH_REMOVE,
						NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return hentry->optimal_gpu;
}

cl_int
GetOptimalGpuForRelation(PlannerInfo *root, RelOptInfo *rel)
{
	RangeTblEntry *rte;
	HeapTuple	tup;
	char		relpersistence;
	cl_int		cuda_dindex;

	if (baseRelIsArrowFdw(rel))
		return GetOptimalGpuForArrowFdw(root, rel);

	cuda_dindex = GetOptimalGpuForTablespace(rel->reltablespace);
	if (cuda_dindex < 0 || cuda_dindex >= numDevAttrs)
		return -1;

	/* only permanent / unlogged table can use NVMe-Strom */
	rte = root->simple_rte_array[rel->relid];
	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for relation %u", rte->relid);
	relpersistence = ((Form_pg_class) GETSTRUCT(tup))->relpersistence;
	ReleaseSysCache(tup);

	if (relpersistence == RELPERSISTENCE_PERMANENT ||
		relpersistence == RELPERSISTENCE_UNLOGGED)
		return cuda_dindex;

	return -1;
}

bool
RelationCanUseNvmeStrom(Relation relation)
{
	Oid		tablespace_oid = RelationGetForm(relation)->reltablespace;
	cl_int	cuda_dindex;
	/* SSD2GPU on temp relation is not supported */
	if (RelationUsesLocalBuffers(relation))
		return false;
	cuda_dindex = GetOptimalGpuForTablespace(tablespace_oid);
	return (cuda_dindex >= 0 &&
			cuda_dindex <  numDevAttrs);
}

/*
 * ScanPathWillUseNvmeStrom - Optimizer Hint
 */
bool
ScanPathWillUseNvmeStrom(PlannerInfo *root, RelOptInfo *baserel)
{
	size_t		num_scan_pages = 0;

	if (!pgstrom_gpudirect_enabled)
		return false;

	/*
	 * Check expected amount of the scan i/o.
	 * If 'baserel' is children of partition table, threshold shall be
	 * checked towards the entire partition size, because the range of
	 * child tables fully depend on scan qualifiers thus variable time
	 * by time. Once user focus on a particular range, but he wants to
	 * focus on other area. It leads potential thrashing on i/o.
	 */
	if (baserel->reloptkind == RELOPT_BASEREL)
	{
		if (GetOptimalGpuForRelation(root, baserel) >= 0)
			num_scan_pages = baserel->pages;
	}
	else if (baserel->reloptkind == RELOPT_OTHER_MEMBER_REL)
	{
		ListCell   *lc;
		Index		parent_relid = 0;

		foreach (lc, root->append_rel_list)
		{
			AppendRelInfo  *appinfo = (AppendRelInfo *) lfirst(lc);

			if (appinfo->child_relid == baserel->relid)
			{
				parent_relid = appinfo->parent_relid;
				break;
			}
		}
		if (!lc)
		{
			elog(NOTICE, "Bug? child table (%d) not found in append_rel_list",
				 baserel->relid);
			return false;
		}

		foreach (lc, root->append_rel_list)
		{
			AppendRelInfo  *appinfo = (AppendRelInfo *) lfirst(lc);
			RelOptInfo	   *rel;

			if (appinfo->parent_relid != parent_relid)
				continue;
			rel = root->simple_rel_array[appinfo->child_relid];
			if (GetOptimalGpuForRelation(root, rel) >= 0)
				num_scan_pages += rel->pages;
		}
	}
	else
		elog(ERROR, "Bug? unexpected reloptkind of base relation: %d",
			 (int)baserel->reloptkind);

	if (num_scan_pages < nvme_strom_threshold() / BLCKSZ)
		return false;
	/* ok, this table scan can use nvme-strom */
	return true;
}

/*
 * pgstrom_init_nvme_strom
 */
void
pgstrom_init_nvme_strom(void)
{
	long		default_threshold;
	Size		shared_buffer_size = (Size)NBuffers * (Size)BLCKSZ;
	bool		has_tesla_gpu = false;
	int			i;

	/*
	 * pg_strom.gpudirect_driver
	 */
	DefineCustomStringVariable("pg_strom.gpudirect_driver",
							   "Driver of SSD-to-GPU Direct SQL",
							   NULL,
							   &pgstrom_gpudirect_driver,
#ifdef WITH_CUFILE
							   "nvidia cufile",
#else
							   "heterodb nvme-strom",
#endif
							   PGC_INTERNAL,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	
	for (i=0; i < numDevAttrs; i++)
	{
		const char *dev_name = devAttrs[i].DEV_NAME;

		if (strncasecmp(dev_name, "Tesla P40",   9) == 0 ||
			strncasecmp(dev_name, "Tesla P100", 10) == 0 ||
			strncasecmp(dev_name, "Tesla V100", 10) == 0 ||
			strncasecmp(dev_name, "A100", 4) == 0)
		{
			has_tesla_gpu = true;
			break;
		}
	}
	DefineCustomBoolVariable("pg_strom.gpudirect_enabled",
							 "Enables SSD-to-GPU Direct SQL",
							 NULL,
							 &pgstrom_gpudirect_enabled,
							 has_tesla_gpu,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/*
	 * MEMO: Threshold of table's physical size to use NVMe-Strom:
	 *   ((System RAM size) -
	 *    (shared_buffer size)) * 0.5 + (shared_buffer size)
	 *
	 * If table size is enough large to issue real i/o, NVMe-Strom will
	 * make advantage by higher i/o performance.
	 */
	if (PAGE_SIZE * PHYS_PAGES < shared_buffer_size)
		elog(ERROR, "Bug? shared_buffer is larger than system RAM");
	default_threshold = ((PAGE_SIZE * PHYS_PAGES - shared_buffer_size) / 2
						 + shared_buffer_size);
	DefineCustomIntVariable("pg_strom.gpudirect_threshold",
							"Tablesize threshold to use SSD-to-GPU P2P DMA",
							NULL,
							&pgstrom_gpudirect_threshold_kb,
							default_threshold >> 10,
							262144,	/* 256MB */
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);

	/*
	 * pg_strom.nvme_distance_map
	 *
	 * config := <token>[,<token>...]
	 * token  := nvmeXX:gpuXX
	 *
	 * eg) nvme0:gpu0,nvme1:gpu1
	 */
	DefineCustomStringVariable("pg_strom.nvme_distance_map",
							   "Manual configuration of GPU<->NVME distances",
							   NULL,
							   &nvme_manual_distance_map,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	setup_nvme_distance_map();
}
