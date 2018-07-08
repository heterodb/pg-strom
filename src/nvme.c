/*
 * nvme.c
 *
 * Routines related to NVME-SSD devices
 * ----
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
} NVMEAttributes;

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
	NVMEAttributes *nvme_attr;	/* if NVMe device */
	List *children;
};
typedef struct PCIDevEntry	PCIDevEntry;

/* static variables */
static HTAB		   *nvmeHash = NULL;

/*
 * sysfs_read_line
 */
static const char *
sysfs_read_line(const char *path)
{
	static char linebuf[2048];
	FILE   *filp;
	char   *pos;

	filp = fopen(path, "r");
	if (!filp)
		elog(ERROR, "failed on fopen('%s'): %m", path);
	if (!fgets(linebuf, sizeof(linebuf), filp))
	{
		fclose(filp);
		elog(ERROR, "failed on fgets('%s'): %m", path);
	}
	pos = linebuf + strlen(linebuf) - 1;
	while (pos >= linebuf && isspace(*pos))
		*pos-- = '\0';
	return linebuf;
}

/*
 * sysfs_read_nvme_attrs
 */
static NVMEAttributes *
sysfs_read_nvme_attrs(const char *nvme_path)
{
	NVMEAttributes *nvmeAttr;
	char		path[MAXPGPATH];
	char		linebuf[2048];
	char	   *pos;
	const char *temp;
	int			i;

	nvmeAttr = palloc0(offsetof(NVMEAttributes,
								nvme_distances[numDevAttrs]));
	snprintf(path, sizeof(path), "%s/device/numa_node", nvme_path);
	temp = sysfs_read_line(path);
	nvmeAttr->numa_node_id = atoi(temp);

	snprintf(path, sizeof(path), "%s/dev", nvme_path);
	temp = sysfs_read_line(path);
	if (sscanf(temp, "%d:%d",
			   &nvmeAttr->nvme_major,
			   &nvmeAttr->nvme_minor) != 2)
		elog(ERROR, "'%s' has unexpected property: %s", path, temp);

	snprintf(path, sizeof(path), "%s/serial", nvme_path);
	temp = sysfs_read_line(path);
	strncpy(nvmeAttr->nvme_serial, temp, sizeof(nvmeAttr->nvme_serial));

	snprintf(path, sizeof(path), "%s/model", nvme_path);
	temp = sysfs_read_line(path);
	strncpy(nvmeAttr->nvme_model, temp, sizeof(nvmeAttr->nvme_model));

	snprintf(path, sizeof(path), "%s/device", nvme_path);
	if (readlink(path, linebuf, sizeof(linebuf)) < 0)
	{
		elog(ERROR, "failed on readlink('%s'): %m", path);
	}
	pos = strrchr(linebuf, '/');
	if (pos)
		pos++;
	else
		pos = linebuf;
	if (sscanf(pos, "%04x:%02x:%02x.%d",
			   &nvmeAttr->nvme_pcie_domain,
			   &nvmeAttr->nvme_pcie_bus_id,
			   &nvmeAttr->nvme_pcie_dev_id,
			   &nvmeAttr->nvme_pcie_func_id) != 4)
	{
		elog(ERROR, "'%s' has unexpected property: %s", path, linebuf);
	}

	for (i=0; i < numDevAttrs; i++)
		nvmeAttr->nvme_distances[i] = -1;

	return nvmeAttr;
}

/*
 * sysfs_read_pcie_attrs
 */
static PCIDevEntry *
sysfs_read_pcie_attrs(const char *dirname, const char *my_name,
					  PCIDevEntry *parent, int depth)
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
		if (my_name[0] != 'p' ||
			my_name[1] != 'c' ||
			my_name[2] != 'i')
			elog(ERROR, "unexpected sysfs entry: %s/%s", dirname, my_name);
		if (sscanf(my_name+3, "%04x:%02x",
				   &entry->domain,
				   &entry->bus_id) != 2)
			elog(ERROR, "unexpected sysfs entry: %s/%s", dirname, my_name);
		entry->dev_id = -1;		/* invalid */
		entry->func_id = -1;	/* invalid */
		entry->depth = depth;
	}
	else
	{
		if (sscanf(my_name, "%04x:%02x:%02x.%d",
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
			NVMEAttributes *nvattr;
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

		/* my_name should be xxxx:xx:xx.x */
		if (!isxdigit(dent->d_name[0]) ||
			!isxdigit(dent->d_name[1]) ||
			!isxdigit(dent->d_name[2]) ||
			!isxdigit(dent->d_name[3]) ||
			dent->d_name[4] != ':' ||
			!isxdigit(dent->d_name[5]) ||
            !isxdigit(dent->d_name[6]) ||
			dent->d_name[7] != ':' ||
			!isxdigit(dent->d_name[8]) ||
            !isxdigit(dent->d_name[9]) ||
			dent->d_name[10] != '.')
			continue;
		for (index=11; dent->d_name[index] != '\0'; index++)
		{
			if (!isxdigit(dent->d_name[index]))
				break;
		}
		if (dent->d_name[index] != '\0')
			continue;

		temp = sysfs_read_pcie_attrs(path, dent->d_name, entry, depth+1);
		if (temp != NULL)
			entry->children = lappend(entry->children, temp);
	}

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
		elog(LOG, "%*s PCIe(%04x:%02x:%02x.%d) %s",
			 2 * indent, "- ",
			 entry->domain,
			 entry->bus_id,
			 entry->dev_id,
			 entry->func_id,
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
							NVMEAttributes *nvme, bool *p_nvme_found)
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
	NVMEAttributes *nvme;
	NVMEAttributes *temp;
	List	   *nvmeAttrList = NIL;
	const char *dirname;
	DIR		   *dir;
	struct dirent *dent;
	char		path[MAXPGPATH];
	List	   *pcie_root = NIL;
	ListCell   *lc;
	int			i, dist;
	bool		found;

	/*
	 * collect individual nvme device's attributes
	 */
	dirname = "/sys/class/nvme";
	dir = opendir(dirname);
	if (!dir)
		elog(ERROR, "failed on opendir('%s'): %m", dirname);
	while ((dent = readdir(dir)) != NULL)
	{
		if (strncmp("nvme", dent->d_name, 4) != 0)
			continue;
		snprintf(path, sizeof(path), "%s/%s", dirname, dent->d_name);

		temp = sysfs_read_nvme_attrs(path);
		strncpy(temp->nvme_name, dent->d_name, sizeof(nvme->nvme_name));

		if (!nvmeHash)
		{
			HASHCTL		hctl;

			memset(&hctl, 0, sizeof(HASHCTL));
			hctl.keysize = offsetof(NVMEAttributes, nvme_name);
			hctl.entrysize = offsetof(NVMEAttributes,
									  nvme_distances[numDevAttrs]);
			hctl.hcxt = TopMemoryContext;
			nvmeHash = hash_create("NVMe SSDs", 256, &hctl,
								   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
		}
		nvme = hash_search(nvmeHash, temp, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? detects duplicate NVMe SSD (%d,%d)",
				 nvme->nvme_major, nvme->nvme_minor);
		Assert(nvme->nvme_major == temp->nvme_major &&
			   nvme->nvme_minor == temp->nvme_minor);
		memcpy(nvme, temp, offsetof(NVMEAttributes,
									nvme_distances[numDevAttrs]));
		nvmeAttrList = lappend(nvmeAttrList, nvme);

		pfree(temp);
	}
	closedir(dir);
	if (nvmeAttrList == NIL)
		return;

	/*
	 * Walk on the PCIe bus tree
	 */
	dirname = "/sys/devices";
	dir = opendir(dirname);
	if (!dir)
		elog(ERROR, "failed on opendir('%s'): %m", dirname);
	while ((dent = readdir(dir)) != NULL)
	{
		PCIDevEntry *pcie_item;

		if (strncmp("pci", dent->d_name, 3) != 0)
			continue;
		pcie_item = sysfs_read_pcie_attrs(dirname, dent->d_name, NULL, 0);
		if (pcie_item)
			pcie_root = lappend(pcie_root, pcie_item);
	}

	/*
	 * calculation of SSD<->GPU distance map
	 */
	for (i=0; i < numDevAttrs; i++)
	{
		DevAttributes  *gpu = &devAttrs[i];
		HASH_SEQ_STATUS	hseq;

		hash_seq_init(&hseq, nvmeHash);
		while ((nvme = hash_seq_search(&hseq)) != NULL)
		{
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
				nvme->nvme_distances[i] = dist;
			else
				nvme->nvme_distances[i] = -1;
		}
	}

	/* Print PCIe tree */
	foreach (lc, pcie_root)
		print_pcie_device_tree(lfirst(lc), 2);

	/* Print GPU<->SSD Distance Matrix */
	if (numDevAttrs > 0 && nvmeHash != NULL)
	{
		HASH_SEQ_STATUS	hseq;
		StringInfoData str;

		initStringInfo(&str);

		for (i=0; i < numDevAttrs; i++)
			appendStringInfo(&str, "    GPU%d", i);
		elog(LOG, "GPU<->SSD Distance Matrix");
		elog(LOG, "        %s", str.data);

		hash_seq_init(&hseq, nvmeHash);
		while ((nvme = hash_seq_search(&hseq)) != NULL)
		{
			resetStringInfo(&str);
			appendStringInfo(&str, " %6s ", nvme->nvme_name);
			for (i=0; i < numDevAttrs; i++)
			{
				appendStringInfo(&str, " % 6d ", nvme->nvme_distances[i]);
			}
			elog(LOG, "%s", str.data);
		}
		pfree(str.data);
	}
}

/*
 * pgstrom_init_nvme
 */
void
pgstrom_init_nvme(void)
{
	setup_nvme_distance_map();
}

