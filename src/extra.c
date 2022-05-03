/*
 * extra.c
 *
 * Stuff related to invoke HeteroDB Extra Module
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <dlfcn.h>
#include "pg_strom.h"

/* pg_strom.gpudirect_driver */
#define GPUDIRECT_DRIVER_TYPE__NONE			1
#define GPUDIRECT_DRIVER_TYPE__CUFILE		2
#define GPUDIRECT_DRIVER_TYPE__NVME_STROM	3

static struct config_enum_entry pgstrom_gpudirect_driver_options[4];
static int		__pgstrom_gpudirect_driver;			/* GUC */

PG_FUNCTION_INFO_V1(pgstrom_license_query);

/*
 * heterodbExtraModuleInfo
 */
static char *(*p_heterodb_extra_module_init)(unsigned int pg_version_num) = NULL;

static char *
heterodbExtraModuleInit(void)
{
	char   *res;

	if (!p_heterodb_extra_module_init)
		elog(ERROR, "HeteroDB Extra module is not loaded yet");
	res = p_heterodb_extra_module_init(PG_VERSION_NUM);
	if (!res)
		elog(ERROR, "out of memory");
	return res;
}

/*
 * heterodbExtraEreport
 */
static heterodb_extra_error_info   *p_heterodb_extra_error_data = NULL;

static void
heterodbExtraEreport(int elevel)
{
	/* see ereport_domain definition */
#if PG_VERSION_NUM >= 130000
	pg_prevent_errno_in_scope();
	if (errstart(elevel, TEXTDOMAIN))
	{
		errcode(ERRCODE_INTERNAL_ERROR);
		errmsg("[extra] %s", p_heterodb_extra_error_data->message);
		errfinish(p_heterodb_extra_error_data->filename,
				  p_heterodb_extra_error_data->lineno,
				  p_heterodb_extra_error_data->funcname);
	}
#else
#if PG_VERSION_NUM >= 120000
	pg_prevent_errno_in_scope();
#endif
	if (errstart(elevel,
				 p_heterodb_extra_error_data->filename,
                 p_heterodb_extra_error_data->lineno,
                 p_heterodb_extra_error_data->funcname,
				 TEXTDOMAIN))
	{
		errcode(ERRCODE_INTERNAL_ERROR);
		errmsg("%s", p_heterodb_extra_error_data->message);
		errfinish(0);
	}
#endif
}

/*
 * heterodbLicenseReload
 */
static int	(*p_heterodb_license_reload)(void) = NULL;
static int
heterodbLicenseReload(void)
{
	if (!p_heterodb_license_reload)
		return -1;
	return p_heterodb_license_reload();
}

/*
 * heterodbLicenseQuery
 */
static ssize_t (*p_heterodb_license_query)(
	char *buf,
	size_t bufsz) = NULL;

static ssize_t
heterodbLicenseQuery(char *buf, size_t bufsz)
{
	if (!p_heterodb_license_query)
		return -1;
	return p_heterodb_license_query(buf, bufsz);
}

/*
 * pgstrom_license_query
 */
static char *
__heterodb_license_query(void)
{
	char	   *buf;
	size_t		bufsz;
	ssize_t		nbytes;

	if (heterodbLicenseReload() <= 0)
		return NULL;

	bufsz = 2048;
retry:
	buf = alloca(bufsz);
	nbytes = heterodbLicenseQuery(buf, bufsz);
	if (nbytes < 0)
		return NULL;
	if (nbytes < bufsz)
		return pstrdup(buf);
	bufsz += bufsz;
	goto retry;
}

Datum
pgstrom_license_query(PG_FUNCTION_ARGS)
{
	char	   *license;

	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 (errmsg("only superuser can query commercial license"))));
	license = __heterodb_license_query();

	PG_RETURN_POINTER(DirectFunctionCall1(json_in, PointerGetDatum(license)));
}

/*
 * gpuDirectInitDriver
 */
static int	  (*p_gpudirect_init_driver)() = NULL;

int
gpuDirectInitDriver(void)
{
	int		rv = -1;

	if (p_gpudirect_init_driver)
	{
		rv = p_gpudirect_init_driver();
		if (rv)
			heterodbExtraEreport(LOG);
	}
	return rv;
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
static int (*p_sysfs_lookup_optimal_gpus)(int fdesc,
										  int nrooms,
										  int *optimal_gpus) = NULL;
Bitmapset *
extraSysfsLookupOptimalGpus(File filp)
{
	Bitmapset  *optimal_gpus = NULL;
	int			fdesc = FileGetRawDesc(filp);
	int			i, nitems;
	int		   *__gpus;

	if (!p_sysfs_lookup_optimal_gpus || numDevAttrs == 0)
		return NULL;
	__gpus = alloca(sizeof(int) * numDevAttrs);
	nitems = p_sysfs_lookup_optimal_gpus(fdesc, numDevAttrs, __gpus);
	if (nitems < 0)
		heterodbExtraEreport(ERROR);
	for (i=0; i < nitems; i++)
	{
		Assert(__gpus[i] >= 0 && __gpus[i] < numDevAttrs);
		optimal_gpus = bms_add_member(optimal_gpus, __gpus[i]);
	}
	return optimal_gpus;
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
 * parse_heterodb_extra_module_info
 */
static void
parse_heterodb_extra_module_info(const char *extra_module_info,
								 uint32 *p_api_version,
								 bool *p_has_cufile,
								 bool *p_has_nvme_strom,
								 int *p_default_gpudirect_driver)
{
	char   *buffer;
	long	api_version = 0;
	bool	has_cufile = false;
	bool	has_nvme_strom = false;
	int		default_gpudirect_driver = GPUDIRECT_DRIVER_TYPE__NONE;
	char   *tok, *pos, *end;
	struct config_enum_entry *entry;

	buffer = alloca(strlen(extra_module_info) + 1);
	strcpy(buffer, extra_module_info);
	for (tok = strtok_r(buffer, ",", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &pos))
	{
		if (strncmp(tok, "api_version=", 12) == 0)
		{
			api_version = strtol(tok+12, &end, 10);
			if (api_version < 0 || *end != '\0')
				elog(ERROR, "invalid extra module token [%s]", tok);
		}
		else if (strncmp(tok, "cufile=", 7) == 0)
		{
			if (strcmp(tok+7, "on") == 0)
				has_cufile = true;
			else if (strcmp(tok+7, "off") == 0)
				has_cufile = false;
			else
				elog(ERROR, "invalid extra module token [%s]", tok);
		}
		else if (strncmp(tok, "nvme_strom=", 11) == 0)
		{
			if (strcmp(tok+11, "on") == 0)
				has_nvme_strom = true;
			else if (strcmp(tok+11, "off") == 0)
				has_nvme_strom = false;
			else
				elog(ERROR, "invalid extra module token [%s]", tok);
		}
	}

	if (api_version < HETERODB_EXTRA_API_VERSION)
		elog(ERROR, "HeteroDB Extra Module has Unsupported API version [%08lu]",
			 api_version);

	/* setup pgstrom.gpudirect_driver options */
	entry = pgstrom_gpudirect_driver_options;
	entry->name   = "none";
	entry->val    = GPUDIRECT_DRIVER_TYPE__NONE;
	entry->hidden = false;
	entry++;

	if (has_nvme_strom)
	{
		default_gpudirect_driver = GPUDIRECT_DRIVER_TYPE__NVME_STROM;
		entry->name    = "nvme_strom";
		entry->val     = GPUDIRECT_DRIVER_TYPE__NVME_STROM;
		entry->hidden  = false;
		entry++;
	}
	if (has_cufile)
	{
		default_gpudirect_driver = GPUDIRECT_DRIVER_TYPE__CUFILE;
		entry->name   = "cufile";
		entry->val    = GPUDIRECT_DRIVER_TYPE__CUFILE;
		entry->hidden = false;
		entry++;
	}
	memset(entry, 0, sizeof(struct config_enum_entry));

	*p_api_version		= api_version;
	*p_has_cufile		= has_cufile;
	*p_has_nvme_strom	= has_nvme_strom;
	*p_default_gpudirect_driver = default_gpudirect_driver;
}

/*
 * pgstrom_init_extra
 */
void
pgstrom_init_extra(void)
{
	const char *prefix = NULL;
	void	   *handle;
	char	   *license;
	char	   *extra_module_info;

	/* load the extra module */
	handle = dlopen(HETERODB_EXTRA_FILENAME,
					RTLD_NOW | RTLD_LOCAL);
	if (!handle)
	{
		handle = dlopen(HETERODB_EXTRA_PATHNAME, RTLD_NOW | RTLD_LOCAL);
		if (!handle)
		{
			elog(LOG, "HeteroDB Extra module is not available");
			return;
		}
	}

	PG_TRY();
	{
		uint32		api_version = 0;
		bool		has_cufile = false;
		bool		has_nvme_strom = false;
		int			default_gpudirect_driver;

		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_error_data);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_module_init);
		extra_module_info = heterodbExtraModuleInit();
		parse_heterodb_extra_module_info(extra_module_info,
										 &api_version,
										 &has_cufile,
										 &has_nvme_strom,
										 &default_gpudirect_driver);
		/* pg_strom.gpudirect_driver */
		DefineCustomEnumVariable("pg_strom.gpudirect_driver",
								 "Selection of the GPUDirectSQL Driver",
								 NULL,
								 &__pgstrom_gpudirect_driver,
								 default_gpudirect_driver,
								 pgstrom_gpudirect_driver_options,
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
		}
		LOOKUP_HETERODB_EXTRA_FUNCTION(sysfs_setup_distance_map);
		LOOKUP_HETERODB_EXTRA_FUNCTION(sysfs_lookup_optimal_gpus);
		LOOKUP_HETERODB_EXTRA_FUNCTION(sysfs_print_nvme_info);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_reload);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_query);
	}
	PG_CATCH();
    {
		p_heterodb_extra_error_data = NULL;
		p_heterodb_extra_module_init = NULL;
		p_gpudirect_init_driver = NULL;
		p_gpudirect_file_desc_open = NULL;
		p_gpudirect_file_desc_open_by_path = NULL;
		p_gpudirect_file_desc_close = NULL;
		p_gpudirect_map_gpu_memory = NULL;
		p_gpudirect_unmap_gpu_memory = NULL;
		p_gpudirect_file_read_iov = NULL;
		p_sysfs_setup_distance_map = NULL;
		p_sysfs_lookup_optimal_gpus = NULL;
		p_sysfs_print_nvme_info = NULL;
		p_heterodb_license_reload = NULL;
		p_heterodb_license_query = NULL;
		PG_RE_THROW();
	}
	PG_END_TRY();
	elog(LOG, "HeteroDB Extra module loaded [%s]", extra_module_info);

	license = __heterodb_license_query();
	if (license)
	{
		elog(LOG, "HeteroDB License: %s", license);
		pfree(license);
	}
}
