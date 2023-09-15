/*
 * extra.c
 *
 * Stuff related to invoke HeteroDB Extra Module
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <dlfcn.h>
#include "pg_strom.h"

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
static int (*p_heterodb_extra_get_error)(const char **p_filename,
                                         unsigned int *p_lineno,
                                         const char **p_funcname,
                                         char *buffer, size_t buffer_sz) = NULL;
int
heterodbExtraGetError(const char **p_filename,
					  unsigned int *p_lineno,
					  const char **p_funcname,
					  char *buffer, size_t buffer_sz)
{
	int		errcode = 0;

	if (p_heterodb_extra_get_error)
	{
		const char *filename;
		unsigned int lineno;
		const char *funcname;

		errcode = p_heterodb_extra_get_error(&filename,
											 &lineno,
											 &funcname,
											 buffer, buffer_sz);
		if (errcode != 0)
		{
			if (p_filename)
				*p_filename = filename;
			if (p_lineno)
				*p_lineno = lineno;
			if (p_funcname)
				*p_funcname = funcname;
		}
	}
	return errcode;
}

static void
heterodbExtraEreport(int elevel)
{
	int			errcode;
	const char *filename;
	unsigned int lineno;
	const char *funcname;
	char		buffer[2000];

	errcode = heterodbExtraGetError(&filename,
									&lineno,
									&funcname,
									buffer, sizeof(buffer));
	if (errcode)
	{
		elog(elevel, "(%s:%u) %s [%s]",
			 filename,
			 lineno,
			 buffer,
			 funcname);
	}
	else if (elevel >= ERROR)
		elog(ERROR, "something failed around heterodbExtraEreport");
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
 * heterodbValidateDevice
 */
static int (*p_heterodb_validate_device)(int gpu_device_id,
										 const char *gpu_device_name,
                                         const char *gpu_device_uuid) = NULL;
bool
heterodbValidateDevice(int gpu_device_id,
					   const char *gpu_device_name,
					   const char *gpu_device_uuid)
{
	if (!p_heterodb_validate_device)
		return false;
	return (p_heterodb_validate_device(gpu_device_id,
									   gpu_device_name,
									   gpu_device_uuid) > 0);
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
	if (!license)
		PG_RETURN_NULL();

	PG_RETURN_DATUM(DirectFunctionCall1(json_in, PointerGetDatum(license)));
}

/*
 * gpuDirectInitDriver
 */
static void	  (*p_cufile__driver_init_v2)() = NULL;

static void
gpuDirectInitDriver(void)
{
	if (!p_cufile__driver_init_v2)
		elog(ERROR, "heterodb_extra: cufile__driver_init_v2 is missing");
	p_cufile__driver_init_v2();
}

/*
 * gpuDirectOpenDriver
 */
static int	  (*p_cufile__driver_open_v2)() = NULL;

void
gpuDirectOpenDriver(void)
{
	if (p_cufile__driver_open_v2)
	{
		if (p_cufile__driver_open_v2() != 0)
			heterodbExtraEreport(ERROR);
	}
}

/*
 * gpuDirectCloseDriver
 */
static int	  (*p_cufile__driver_close_v2)() = NULL;

void
gpuDirectCloseDriver(void)
{
	if (p_cufile__driver_close_v2)
	{
		if (p_cufile__driver_close_v2() != 0)
			heterodbExtraEreport(LOG);
	}
}

/*
 * gpuDirectMapGpuMemory
 */
static int	(*p_cufile__map_gpu_memory_v2)(CUdeviceptr m_segment,
										   size_t segment_sz) = NULL;
bool
gpuDirectMapGpuMemory(CUdeviceptr m_segment,
					  size_t segment_sz)
{
	if (p_cufile__map_gpu_memory_v2)
	{
		if (p_cufile__map_gpu_memory_v2(m_segment, segment_sz) != 0)
			return false;
	}
	return true;
}

/*
 * gpuDirectUnmapGpuMemory
 */
static int	(*p_cufile__unmap_gpu_memory_v2)(CUdeviceptr m_segment) = NULL;

bool
gpuDirectUnmapGpuMemory(CUdeviceptr m_segment)
{
	if (p_cufile__unmap_gpu_memory_v2)
	{
		if (p_cufile__unmap_gpu_memory_v2(m_segment) != 0)
			return false;
	}
	return true;
}

/*
 * __fallbackFileReadIOV
 */
static bool
__fallbackFileReadIOV(const char *pathname,
					  CUdeviceptr m_segment,
					  off_t m_offset,
					  const strom_io_vector *iovec)
{
	size_t	io_unitsz = (16UL << 20);	/* 16MB */
	char   *buffer;
	int		fdesc;
	struct stat	stat_buf;

	fdesc = open(pathname, O_RDONLY);
	if (fdesc < 0)
	{
		fprintf(stderr, "failed on open('%s'): %m\n", pathname);
		goto error_0;
	}

	if (fstat(fdesc, &stat_buf) != 0)
	{
		fprintf(stderr, "failed on fstat('%s'): %m\n", pathname);
		goto error_1;
	}

	buffer = malloc(io_unitsz);
	if (!buffer)
	{
		fprintf(stderr, "out of memory: %m\n");
		goto error_1;
	}

	for (int i=0; i < iovec->nr_chunks; i++)
	{
		const strom_io_chunk *ioc = &iovec->ioc[i];
		size_t		remained = ioc->nr_pages * PAGE_SIZE;
		off_t		file_pos = ioc->fchunk_id * PAGE_SIZE;
		off_t		dest_pos = m_offset + ioc->m_offset;
		ssize_t		sz, nbytes;
		CUresult	rc;

		/* cut off the file tail */
		if (file_pos >= stat_buf.st_size)
			continue;
		if (file_pos + remained > stat_buf.st_size)
			remained = stat_buf.st_size - file_pos;

		while (remained > 0)
		{
			sz = Min(remained, io_unitsz);
			nbytes = pread(fdesc, buffer, sz, file_pos);
			if (nbytes <= 0)
			{
				fprintf(stderr, "failed on pread: %m\n");
				goto error_2;
			}
			rc = cuMemcpyHtoD(m_segment + dest_pos, buffer, nbytes);
			if (rc != CUDA_SUCCESS)
			{
				fprintf(stderr, "failed on cuMemcpyHtoD\n");
				goto error_2;
			}
			file_pos += nbytes;
			dest_pos += nbytes;
			remained -= nbytes;
		}
	}
	free(buffer);
	close(fdesc);

	return true;
error_2:
	free(buffer);
error_1:
	close(fdesc);
error_0:
	return false;
}

/*
 * gpuDirectFileReadIOV
 */
static int	(*p_cufile__read_file_iov_v2)(
	const char *pathname,
	CUdeviceptr m_segment,
	off_t m_offset,
	const strom_io_vector *iovec) = NULL;

bool
gpuDirectFileReadIOV(const char *pathname,
					 CUdeviceptr m_segment,
					 off_t m_offset,
					 const strom_io_vector *iovec)
{
	if (p_cufile__read_file_iov_v2)
		return (p_cufile__read_file_iov_v2(pathname,
										   m_segment,
										   m_offset,
										   iovec) == 0);
	/* fallback by the regular filesystem */
	return __fallbackFileReadIOV(pathname,
								 m_segment,
								 m_offset,
								 iovec);
}

/*
 * gpuDirectGetProperty
 */
static int	(*p_cufile__get_property_v2)(char *buffer,
										 size_t buffer_sz) = NULL;
char *
gpuDirectGetProperty(void)
{
	char	buffer[2000];

	if (!p_cufile__get_property_v2)
		elog(ERROR, "heterodb_extra: cufile__get_property_v2 is missing");
	if (p_cufile__get_property_v2(buffer, sizeof(buffer)) < 0)
		heterodbExtraEreport(ERROR);
	return pstrdup(buffer);
}

/*
 * gpuDirectSetProperty
 */
static int	(*p_cufile__set_property_v2)(const char *key,
										 const char *value) = NULL;
void
gpuDirectSetProperty(const char *key, const char *value)
{
	if (!p_cufile__set_property_v2)
		elog(ERROR, "heterodb_extra: cufile__set_property_v2 is missing");
	if (p_cufile__set_property_v2(key, value) != 0)
		heterodbExtraEreport(ERROR);
}

/*
 * gpuDirectIsSupported
 */
bool
gpuDirectIsAvailable(void)
{
	bool	has_gpudirectsql_supported = false;

	if (p_cufile__driver_init_v2 &&
		p_cufile__driver_open_v2 &&
		p_cufile__driver_close_v2 &&
		p_cufile__map_gpu_memory_v2 &&
		p_cufile__unmap_gpu_memory_v2 &&
		p_cufile__read_file_iov_v2 &&
		p_cufile__get_property_v2 &&
		p_cufile__set_property_v2)
	{
		for (int i=0; i < numGpuDevAttrs; i++)
		{
			if (gpuDevAttrs[i].DEV_SUPPORT_GPUDIRECTSQL)
			{
				has_gpudirectsql_supported = true;
				break;
			}
		}
	}
	return has_gpudirectsql_supported;
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

/*
 * parse_heterodb_extra_module_info
 */
static void
parse_heterodb_extra_module_info(const char *extra_module_info,
								 uint32 *p_api_version,
								 bool *p_has_cufile)
{
	char   *buffer;
	long	api_version = 0;
	bool	has_cufile = false;
	char   *tok, *pos, *end;

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
	}
	if (api_version < HETERODB_EXTRA_API_VERSION)
		elog(ERROR, "HeteroDB Extra Module has Unsupported API version [%08lu]",
			 api_version);
	*p_api_version		= api_version;
	*p_has_cufile		= has_cufile;
}

/*
 * pgstrom_init_extra
 */
void
pgstrom_init_extra(void)
{
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

		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_module_init);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_get_error);
		extra_module_info = heterodbExtraModuleInit();
		parse_heterodb_extra_module_info(extra_module_info,
										 &api_version,
										 &has_cufile);
		if (has_cufile)
		{
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__driver_init_v2);
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__driver_open_v2);
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__driver_close_v2);
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__map_gpu_memory_v2);
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__unmap_gpu_memory_v2);
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__read_file_iov_v2);
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__get_property_v2);
			LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__set_property_v2);

			gpuDirectInitDriver();
		}
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_reload);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_query);
		LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_validate_device);
	}
	PG_CATCH();
    {
		p_heterodb_extra_module_init = NULL;
		p_heterodb_extra_get_error = NULL;
		p_cufile__driver_init_v2 = NULL;
		p_cufile__driver_open_v2 = NULL;
		p_cufile__driver_close_v2 = NULL;
		p_cufile__map_gpu_memory_v2 = NULL;
		p_cufile__unmap_gpu_memory_v2 = NULL;
		p_cufile__read_file_iov_v2 = NULL;
		p_cufile__get_property_v2 = NULL;
		p_cufile__set_property_v2 = NULL;
		p_heterodb_license_reload = NULL;
		p_heterodb_license_query = NULL;
		p_heterodb_validate_device = NULL;
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
