/*
 * extra.c
 *
 * Stuff related to invoke HeteroDB's commercial features.
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
#include <dlfcn.h>
#include "pg_strom.h"
#include <heterodb_extra.h>

/* pg_strom.gpudirect_driver */
#define GPUDIRECT_DRIVER_TYPE__NONE			1
#define GPUDIRECT_DRIVER_TYPE__CUFILE		2
#define GPUDIRECT_DRIVER_TYPE__NVME_STROM	3
#define GPUDIRECT_DRIVER_TYPE__DEFAULT		GPUDIRECT_DRIVER_TYPE__NVME_STROM

/* in case of no cufile APIs are not installed */
static struct config_enum_entry pgstrom_gpudirect_driver_options_1[] = {
	{"none",       GPUDIRECT_DRIVER_TYPE__NONE,       false },
	{"nvme_strom", GPUDIRECT_DRIVER_TYPE__NVME_STROM, false },
	{NULL, 0, false}
};

/* in case of cufile APIs are installed */
static struct config_enum_entry pgstrom_gpudirect_driver_options_2[] = {
	{"none",       GPUDIRECT_DRIVER_TYPE__NONE,       false },
	{"nvme_strom", GPUDIRECT_DRIVER_TYPE__NVME_STROM, false },
	{"cufile",     GPUDIRECT_DRIVER_TYPE__CUFILE,     false },
	{NULL, 0, false}
};



static int		__pgstrom_gpudirect_driver;			/* GUC */
static bool		__pgstrom_gpudirect_enabled;		/* GUC */
static int		__pgstrom_gpudirect_threshold;		/* GUC */
static bool		gpudirect_driver_is_initialized = false;

PG_FUNCTION_INFO_V1(pgstrom_license_query);

/*
 * pgstrom_gpudirect_enabled
 */
bool
pgstrom_gpudirect_enabled(void)
{
	return __pgstrom_gpudirect_enabled;
}

/*
 * pgstrom_gpudirect_enabled_checker
 */
static bool
pgstrom_gpudirect_enabled_checker(bool *p_newval, void **extra, GucSource source)
{
	bool	newval = *p_newval;

	if (newval && !gpudirect_driver_is_initialized)
		elog(ERROR, "cannot enable GPUDirectSQL without driver module loaded");
	return true;
}


/*
 * pgstrom_gpudirect_threshold
 */
Size
pgstrom_gpudirect_threshold(void)
{
	return (Size)__pgstrom_gpudirect_threshold << 10;
}

/*
 * heterodbExtraApiVersion
 */
static unsigned int (*p_heterodb_extra_api_version)(void) = NULL;

static unsigned int
heterodbExtraApiVersion(void)
{
	return p_heterodb_extra_api_version();
}

/*
 * heterodbExtraEreport
 */
static heterodb_extra_error_info   *p_heterodb_extra_error_data = NULL;

static void
heterodbExtraEreport(int elevel)
{
	pg_prevent_errno_in_scope();
	elog_start(p_heterodb_extra_error_data->filename,
			   p_heterodb_extra_error_data->lineno,
			   p_heterodb_extra_error_data->funcname);
	elog_finish(elevel, "%s", p_heterodb_extra_error_data->message);
}

/*
 * heterodbLicenseReload
 */
static const heterodb_license_info *(*p_heterodb_license_reload)(const char *pathname) = NULL;

static const heterodb_license_info *
heterodbLicenseReload(const char *pathname)
{
	if (!p_heterodb_license_reload)
		return NULL;
	return p_heterodb_license_reload(pathname);
}

/*
 * heterodb_license_query
 */
static void
heterodb_license_print_v2(StringInfo str, const heterodb_license_info_v2 *linfo)
{
	appendStringInfo(str, "{ \"version\" : %d",
					 linfo->version);
	if (linfo->serial_nr)
		appendStringInfo(str, ", \"serial_nr\" : \"%s\"",
						 linfo->serial_nr);
	appendStringInfo(str, ", \"issued_at\" : \"%04d-%02d-%02d\"",
					 (linfo->issued_at / 10000),
					 (linfo->issued_at / 100) % 100,
					 (linfo->issued_at % 100));
	appendStringInfo(str, ", \"expired_at\" : \"%04d-%02d-%02d\"",
					 (linfo->expired_at / 10000),
					 (linfo->expired_at / 100) % 100,
					 (linfo->expired_at % 100));
	if (linfo->licensee_org)
		appendStringInfo(str, ", \"licensee_org\" : \"%s\"",
						 linfo->licensee_org);
	if (linfo->licensee_name)
		appendStringInfo(str, ", \"licensee_name\" : \"%s\"",
						 linfo->licensee_name);
	if (linfo->licensee_mail)
		appendStringInfo(str, ", \"licensee_mail\" : \"%s\"",
						 linfo->licensee_mail);
	if (linfo->description)
		appendStringInfo(str, ", \"description\" : \"%s\"",
						 linfo->description);
	if (linfo->nr_gpus > 0)
	{
		int		i;

		appendStringInfo(str, ", \"gpus\" : [");
		for (i=0; i < linfo->nr_gpus; i++)
		{
			appendStringInfo(str, "%s{ \"uuid\" : \"%s\" }",
							 i > 0 ? ", " : " ",
							 linfo->gpu_uuid[i]);
		}
		appendStringInfo(str, " ]");
	}
	appendStringInfo(str, "}");
}

static char *
heterodb_license_query(void)
{
	const heterodb_license_info *linfo = heterodbLicenseReload(NULL);
	StringInfoData str;

	if (!linfo)
		return NULL;

	initStringInfo(&str);

	if (linfo->version == 2)
		heterodb_license_print_v2(&str, &linfo->v2);
	else
		elog(ERROR, "unknown license version: %d", linfo->version);
	return str.data;
}

Datum
pgstrom_license_query(PG_FUNCTION_ARGS)
{
	char   *license;

	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 (errmsg("only superuser can query commercial license"))));
	license = heterodb_license_query();
	if (!license)
		PG_RETURN_NULL();
	PG_RETURN_POINTER(DirectFunctionCall1(json_in, PointerGetDatum(license)));
}

/*
 * gpuDirectInitDriver
 */
static int	  (*p_gpudirect_init_driver)() = NULL;

static int
gpuDirectInitDriver(void)
{
	Assert(p_gpudirect_init_driver != NULL);
	if (p_gpudirect_init_driver())
	{
		heterodbExtraEreport(LOG);
		return 1;
	}
	return 0;
}

/*
 * gpuDirectFileDescOpen
 */
static int (*p_gpudirect_file_desc_open)(
	GPUDirectFileDesc *gds_fdesc,
	int rawfd, const char *pathname) = NULL;

void
gpuDirectFileDescOpen(GPUDirectFileDesc *gds_fdesc, File pg_fdesc)
{
	int		rawfd = FileGetRawDesc(pg_fdesc);
	char   *pathname = FilePathName(pg_fdesc);

	if (p_gpudirect_file_desc_open(gds_fdesc, rawfd, pathname))
		heterodbExtraEreport(ERROR);
}

/*
 * gpuDirectFileDescOpenByPath
 */
static int (*p_gpudirect_file_desc_open_by_path)(
	GPUDirectFileDesc *gds_fdesc,
	const char *pathname) = NULL;

void
gpuDirectFileDescOpenByPath(GPUDirectFileDesc *gds_fdesc,
							const char *pathname)
{
	if (p_gpudirect_file_desc_open_by_path(gds_fdesc, pathname))
		heterodbExtraEreport(ERROR);
}

/*
 * gpuDirectFileDescClose
 */
static void (*p_gpudirect_file_desc_close)(
	const GPUDirectFileDesc *gds_fdesc) = NULL;

void
gpuDirectFileDescClose(const GPUDirectFileDesc *gds_fdesc)
{
	Assert(p_gpudirect_file_desc_close != NULL);
	p_gpudirect_file_desc_close(gds_fdesc);
}

/*
 * gpuDirectMapGpuMemory
 */
static CUresult (*p_gpudirect_map_gpu_memory)(
	CUdeviceptr m_segment,
	size_t m_segment_sz,
	unsigned long *p_iomap_handle) = NULL;

CUresult
gpuDirectMapGpuMemory(CUdeviceptr m_segment,
					  size_t m_segment_sz,
					  unsigned long *p_iomap_handle)
{
	Assert(p_gpudirect_map_gpu_memory != NULL);
	return p_gpudirect_map_gpu_memory(m_segment, m_segment_sz, p_iomap_handle);
}

/*
 * gpuDirectUnmapGpuMemory
 */
static CUresult (*p_gpudirect_unmap_gpu_memory)(
	CUdeviceptr m_segment,
	unsigned long iomap_handle) = NULL;

CUresult
gpuDirectUnmapGpuMemory(CUdeviceptr m_segment,
						unsigned long iomap_handle)
{
	Assert(p_gpudirect_unmap_gpu_memory != NULL);
	return p_gpudirect_unmap_gpu_memory(m_segment, iomap_handle);
}

/*
 * gpuDirectFileReadIOV
 */
static int (*p_gpudirect_file_read_iov)(
	const GPUDirectFileDesc *gds_fdesc,
	CUdeviceptr m_segment,
	unsigned long iomap_handle,
	off_t m_offset,
	strom_io_vector *iovec) = NULL;

void
gpuDirectFileReadIOV(const GPUDirectFileDesc *gds_fdesc,
					 CUdeviceptr m_segment,
					 unsigned long iomap_handle,
					 off_t m_offset,
					 strom_io_vector *iovec)
{
	Assert(p_gpudirect_file_read_iov != NULL);
	if (p_gpudirect_file_read_iov(gds_fdesc,
								  m_segment,
								  iomap_handle,
								  m_offset,
								  iovec))
		werror("failed on gpuDirectFileReadIOV");
}

/*
 * extraSysfsSetupDistanceMap
 */
static int (*p_sysfs_setup_distance_map)(
	int gpu_count,
	GpuPciDevItem *gpu_array,
	const char *manual_config) = NULL;

void
extraSysfsSetupDistanceMap(const char *manual_config)
{
	GpuPciDevItem *gpu_array;
	int			i;

	if (!p_sysfs_setup_distance_map)
		return;		/* nothing to do */

	gpu_array = alloca(numDevAttrs * sizeof(GpuPciDevItem));
	memset(gpu_array, 0, numDevAttrs * sizeof(GpuPciDevItem));
	for (i=0; i < numDevAttrs; i++)
	{
		DevAttributes  *dattr = &devAttrs[i];
		GpuPciDevItem  *gpu = &gpu_array[i];

		gpu->device_id = dattr->DEV_ID;
		strncpy(gpu->device_name, dattr->DEV_NAME,
				sizeof(gpu->device_name));
		gpu->numa_node_id = dattr->NUMA_NODE_ID;
		gpu->pci_domain = dattr->PCI_DOMAIN_ID;
		gpu->pci_bus_id = dattr->PCI_BUS_ID;
		gpu->pci_dev_id = dattr->PCI_DEVICE_ID;
		if (dattr->MULTI_GPU_BOARD)
			gpu->pci_func_id = dattr->MULTI_GPU_BOARD_GROUP_ID;
	}
	if (p_sysfs_setup_distance_map(numDevAttrs,
								   gpu_array,
								   manual_config) < 0)
		heterodbExtraEreport(ERROR);
}

/*
 * extraSysfsLookupOptimalGpu
 */
static int (*p_sysfs_lookup_optimal_gpu)(dev_t st_dev) = NULL;

int
extraSysfsLookupOptimalGpu(dev_t st_dev)
{
	int		optimal_gpu;

	if (!p_sysfs_lookup_optimal_gpu)
		return -1;
	optimal_gpu = p_sysfs_lookup_optimal_gpu(st_dev);
	if (optimal_gpu < -1)
		heterodbExtraEreport(ERROR);
	return optimal_gpu;
}

/*
 * extraSysfsPrintNvmeInfo
 */
static ssize_t (*p_sysfs_print_nvme_info)(
	int index,
	char *buffer,
	ssize_t buffer_sz) = NULL;

ssize_t
extraSysfsPrintNvmeInfo(int index, char *buffer, ssize_t buffer_sz)
{
	if (!p_sysfs_print_nvme_info)
		return -1;
	return p_sysfs_print_nvme_info(index, buffer, buffer_sz);
}

/* lookup_heterodb_extra_function */
static void *
lookup_heterodb_extra_function(void *handle, const char *symbol)
{
	void   *fn_addr;

	fn_addr = dlsym(handle, symbol);
	if (!fn_addr)
		elog(ERROR, "could not find extra symbol \"%s\" - %s",
			 symbol, dlerror());
	return fn_addr;
}
#define LOOKUP_HETERODB_EXTRA_FUNCTION(symbol)	\
	p_##symbol = lookup_heterodb_extra_function(handle, #symbol)

/* lookup_gpudirect_function */
static void *
lookup_gpudirect_function(void *handle, const char *prefix, const char *func_name)
{
	char	symbol[128];

	snprintf(symbol, sizeof(symbol), "%s__%s", prefix, func_name);
	return lookup_heterodb_extra_function(handle, symbol);
}

#define LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix,func_name)	\
	p_gpudirect_##func_name = lookup_gpudirect_function(handle, prefix, #func_name)

/*
 * pgstrom_init_extra
 */
void
pgstrom_init_extra(void)
{
	const char *prefix = NULL;
	void	   *handle;
	char	   *license;
	uint32		api_version;
	bool		with_cufile = false;
	bool		__gpudirect_enabled = false;
	size_t		default_threshold = 0;
	size_t		shared_buffer_size = (size_t)NBuffers * (size_t)BLCKSZ;

	
	/* load the extra module */
	handle = dlopen(HETERODB_EXTRA_FILENAME,
					RTLD_NOW | RTLD_LOCAL);
	if (!handle)
	{
		handle = dlopen(HETERODB_EXTRA_PATHNAME, RTLD_NOW | RTLD_LOCAL);
		if (!handle)
			goto skip;
	}

	PG_TRY();
	{
		struct config_enum_entry *gpudirect_driver_options;

		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_error_data);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_api_version);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_reload);
		api_version = heterodbExtraApiVersion();
		if ((api_version & HETERODB_EXTRA_WITH_CUFILE) != 0)
			with_cufile = true;
		api_version /= 100;

		if (!with_cufile)
			gpudirect_driver_options = pgstrom_gpudirect_driver_options_1;
		else
			gpudirect_driver_options = pgstrom_gpudirect_driver_options_2;

		DefineCustomEnumVariable("pg_strom.gpudirect_driver",
								 "GPUDirectSQL Driver Selection",
								 "'nvme_strom', 'cufile' or 'none'",
								 &__pgstrom_gpudirect_driver,
								 GPUDIRECT_DRIVER_TYPE__DEFAULT,
								 gpudirect_driver_options,
								 PGC_POSTMASTER,
								 GUC_NOT_IN_SAMPLE,
								 NULL, NULL, NULL);
		if (__pgstrom_gpudirect_driver == GPUDIRECT_DRIVER_TYPE__CUFILE)
			prefix = "cufile";
		else if (__pgstrom_gpudirect_driver == GPUDIRECT_DRIVER_TYPE__NVME_STROM)
			prefix = "nvme_strom";

		if (prefix)
		{
			LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix, init_driver);
			LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix, file_desc_open);
			LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix, file_desc_open_by_path);
			LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix, file_desc_close);
			LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix, map_gpu_memory);
			LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix, unmap_gpu_memory);
			LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix, file_read_iov);

			if (gpuDirectInitDriver() == 0)
			{
				int		i;

				for (i=0; i < numDevAttrs; i++)
				{
					if (devAttrs[i].DEV_SUPPORT_GPUDIRECTSQL)
					{
						__gpudirect_enabled = true;
						break;
					}
				}
				gpudirect_driver_is_initialized = true;
			}
		}
		LOOKUP_HETERODB_EXTRA_FUNCTION(sysfs_setup_distance_map);
		LOOKUP_HETERODB_EXTRA_FUNCTION(sysfs_lookup_optimal_gpu);
		LOOKUP_HETERODB_EXTRA_FUNCTION(sysfs_print_nvme_info);
	}
	PG_CATCH();
    {
		p_heterodb_extra_error_data = NULL;
		p_heterodb_extra_api_version = NULL;
		p_heterodb_license_reload = NULL;
		p_gpudirect_init_driver = NULL;
		p_gpudirect_file_desc_open = NULL;
		p_gpudirect_file_desc_open_by_path = NULL;
		p_gpudirect_file_desc_close = NULL;
		p_gpudirect_map_gpu_memory = NULL;
		p_gpudirect_unmap_gpu_memory = NULL;
		p_gpudirect_file_read_iov = NULL;
		p_sysfs_setup_distance_map = NULL;
		p_sysfs_lookup_optimal_gpu = NULL;
		p_sysfs_print_nvme_info = NULL;
		dlclose(handle);
		PG_RE_THROW();
	}
	PG_END_TRY();

	elog(LOG, "HeteroDB Extra module loaded (API=%u%s)",
		 api_version,
		 with_cufile ? "; NVIDIA cuFile" : "");

	license = heterodb_license_query();
	if (license)
	{
		elog(LOG, "HeteroDB License: %s", license);
		pfree(license);
	}

skip:
	DefineCustomBoolVariable("pg_strom.gpudirect_enabled",
							 "Enables GPUDirectSQL",
							 NULL,
							 &__pgstrom_gpudirect_enabled,
							 __gpudirect_enabled,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 pgstrom_gpudirect_enabled_checker, NULL, NULL);
	/*
	 * MEMO: Threshold of table's physical size to use NVMe-Strom:
	 *   ((System RAM size) -
	 *    (shared_buffer size)) * 0.5 + (shared_buffer size)
	 *
	 * If table size is enough large to issue real i/o, NVMe-Strom will
	 * make advantage by higher i/o performance.
	 */
	if (PAGE_SIZE * PHYS_PAGES > shared_buffer_size / 2)
		default_threshold = (PAGE_SIZE * PHYS_PAGES - shared_buffer_size / 2);
	default_threshold += shared_buffer_size;

	DefineCustomIntVariable("pg_strom.gpudirect_threshold",
							"Tablesize threshold to use SSD-to-GPU P2P DMA",
							NULL,
							&__pgstrom_gpudirect_threshold,
							default_threshold >> 10,
							262144,	/* 256MB */
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
}
