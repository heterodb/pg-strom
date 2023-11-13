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


#define GPUDIRECT_DRIVER__CUFILE		'n'
#define GPUDIRECT_DRIVER__NVME_STROM	'h'
#define GPUDIRECT_DRIVER__VFS			'v'
static int		gpudirect_driver_kind;
static __thread void   *gpudirect_vfs_dma_buffer = NULL;
static __thread size_t	gpudirect_vfs_dma_buffer_sz = 0UL;
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
static void	  (*p_gpudirect__driver_init_v2)(void) = NULL;
static void
gpuDirectInitDriver(void)
{
	if (!p_gpudirect__driver_init_v2)
		elog(ERROR, "heterodb_extra: gpudirect__driver_init_v2 is missing");
	p_gpudirect__driver_init_v2();
}

/*
 * gpuDirectOpenDriver
 */
static int	  (*p_cufile__driver_open_v2)(void) = NULL;
static int	  (*p_nvme_strom__driver_open)(void) = NULL;

void
gpuDirectOpenDriver(void)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			if (!p_cufile__driver_open_v2)
				elog(ERROR, "cuFile is not available");
			if (p_cufile__driver_open_v2() != 0)
				heterodbExtraEreport(ERROR);
			break;

		case GPUDIRECT_DRIVER__NVME_STROM:
			if (!p_nvme_strom__driver_open)
				elog(ERROR, "nvme_strom is not available");
			if (p_nvme_strom__driver_open() != 0)
				heterodbExtraEreport(ERROR);
			break;
		default:
			break;
	}
}

/*
 * gpuDirectCloseDriver
 */
static int	  (*p_cufile__driver_close_v2)(void) = NULL;
static int	  (*p_nvme_strom__driver_close)(void) = NULL;

void
gpuDirectCloseDriver(void)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			if (!p_cufile__driver_close_v2)
				elog(ERROR, "cuFile is not available");
			if (p_cufile__driver_close_v2() != 0)
				heterodbExtraEreport(ERROR);
			break;

		case GPUDIRECT_DRIVER__NVME_STROM:
			if (!p_nvme_strom__driver_close)
				elog(ERROR, "nvme_strom is not available");
			if (p_nvme_strom__driver_close() != 0)
				heterodbExtraEreport(ERROR);
			break;
		default:
			break;
	}
}

/*
 * gpuDirectMapGpuMemory
 */
static int	(*p_cufile__map_gpu_memory_v2)(CUdeviceptr m_segment,
										   size_t segment_sz) = NULL;
static int	(*p_nvme_strom__map_gpu_memory)(CUdeviceptr m_segment,
											size_t m_segment_sz,
											unsigned long *p_iomap_handle) = NULL;
bool
gpuDirectMapGpuMemory(CUdeviceptr m_segment, size_t segment_sz,
					  unsigned long *p_iomap_handle)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			if (p_cufile__map_gpu_memory_v2 == NULL ||
				p_cufile__map_gpu_memory_v2(m_segment, segment_sz) != 0)
				return false;
			break;
		case GPUDIRECT_DRIVER__NVME_STROM:
			if (p_nvme_strom__map_gpu_memory == NULL ||
				p_nvme_strom__map_gpu_memory(m_segment, segment_sz, p_iomap_handle) != 0)
				return false;
			break;
		default:
			break;
	}
	return true;
}

/*
 * gpuDirectUnmapGpuMemory
 */
static int	(*p_cufile__unmap_gpu_memory_v2)(CUdeviceptr m_segment) = NULL;
static int	(*p_nvme_strom__unmap_gpu_memory)(unsigned long iomap_handle) = NULL;

bool
gpuDirectUnmapGpuMemory(CUdeviceptr m_segment,
						unsigned long iomap_handle)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			if (p_cufile__unmap_gpu_memory_v2 == NULL ||
				p_cufile__unmap_gpu_memory_v2(m_segment) != 0)
				return false;
			break;
		case GPUDIRECT_DRIVER__NVME_STROM:
			if (p_nvme_strom__unmap_gpu_memory == NULL ||
				p_nvme_strom__unmap_gpu_memory(iomap_handle) != 0)
				return false;
			break;
		default:
			break;
	}
	return true;
}

/*
 * gpuDirectRegisterStream
 */
static int	(*p_cufile__register_stream_v3)(CUstream cuda_stream,
											uint32_t flags) = NULL;
bool
gpuDirectRegisterStream(CUstream cuda_stream)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			/*
			 * NOTE: CU_FILE_STREAM_* labels are defined at CUDA12.2,
			 * so older CUDA version leads build errors. Right now,
			 * we don't use Async-Read APIs, so tentatively put the
			 * equivalent immidiate value.
			 */
			if (p_cufile__register_stream_v3 == NULL ||
				p_cufile__register_stream_v3(cuda_stream, 15) != 0)
				return false;
		default:
			break;
	}
	return true;
}

/*
 * gpuDirectDeregisterStream
 */
static int	(*p_cufile__deregister_stream_v3)(CUstream cuda_stream) = NULL;

bool
gpuDirectDeregisterStream(CUstream cuda_stream)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			if (p_cufile__deregister_stream_v3 == NULL ||
				p_cufile__deregister_stream_v3(cuda_stream) != 0)
				return false;
		default:
			break;
	}
	return true;
}

/*
 * __gpuDirectAllocDMABufferOnDemand
 */
static bool
__gpuDirectAllocDMABufferOnDemand(void)
{
	CUresult	rc;

	if (!gpudirect_vfs_dma_buffer)
	{
		size_t	bufsz = PGSTROM_CHUNK_SIZE + (8UL<<20);

		rc = cuMemAllocHost(&gpudirect_vfs_dma_buffer, bufsz);
		if (rc != CUDA_SUCCESS)
			return false;
		gpudirect_vfs_dma_buffer_sz = bufsz;
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
					  char *vfs_dma_buffer,
					  size_t vfs_dma_buffer_sz,
					  const strom_io_vector *iovec,
					  uint32_t *p_npages_direct_read,
					  uint32_t *p_npages_vfs_read)
{
	int			fdesc;
	uint32_t	nr_pages = 0;
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
			sz = Min(remained, vfs_dma_buffer_sz);
			nbytes = pread(fdesc, vfs_dma_buffer, sz, file_pos);
			if (nbytes <= 0)
			{
				if (errno == EINTR)
					continue;
				fprintf(stderr, "failed on pread: %m\n");
				goto error_1;
			}
			rc = cuMemcpyHtoD(m_segment + dest_pos, vfs_dma_buffer, nbytes);
			if (rc != CUDA_SUCCESS)
			{
				fprintf(stderr, "failed on cuMemcpyHtoD\n");
				goto error_1;
			}
			file_pos += nbytes;
			dest_pos += nbytes;
			remained -= nbytes;
		}
		nr_pages += ioc->nr_pages;
	}
	close(fdesc);
	/* update statistics */
	if (p_npages_direct_read)
		*p_npages_direct_read = 0;
	if (p_npages_vfs_read)
		*p_npages_vfs_read = nr_pages;
	return true;

error_1:
	close(fdesc);
error_0:
	return false;
}

/*
 * gpuDirectFileReadIOV
 */
static int	(*p_cufile__read_file_iov_v3)(
	const char *pathname,
	CUdeviceptr m_segment,
	off_t m_offset,
	const strom_io_vector *iovec,
	uint32_t *p_npages_direct_read,
	uint32_t *p_npages_vfs_read) = NULL;
static int	(*p_nvme_strom__read_file_iov)(
	const char *pathname,
	unsigned long iomap_handle,
	off_t m_offset,
	const strom_io_vector *iovec,
	uint32_t *p_npages_direct_read,
	uint32_t *p_npages_vfs_read) = NULL;
static int	(*p_vfs_fallback__read_file_iov)(
	const char *pathname,
	CUdeviceptr m_segment,
	off_t m_offset,
	void *dma_buffer,
	size_t dma_buffer_sz,
	CUstream cuda_stream,
	const strom_io_vector *iovec,
	uint32_t *p_npages_direct_read,
	uint32_t *p_npages_vfs_read) = NULL;

bool
gpuDirectFileReadIOV(const char *pathname,
					 CUdeviceptr m_segment,
					 off_t m_offset,
					 unsigned long iomap_handle,
					 const strom_io_vector *iovec,
					 uint32_t *p_npages_direct_read,
					 uint32_t *p_npages_vfs_read)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			if (p_cufile__read_file_iov_v3)
				return (p_cufile__read_file_iov_v3(pathname,
												   m_segment,
												   m_offset,
												   iovec,
												   p_npages_direct_read,
												   p_npages_vfs_read) == 0);
			break;
		case GPUDIRECT_DRIVER__NVME_STROM:
			if (p_nvme_strom__read_file_iov)
				return (p_nvme_strom__read_file_iov(pathname,
													iomap_handle,
													m_offset,
													iovec,
													p_npages_direct_read,
													p_npages_vfs_read) == 0);
			break;
		case GPUDIRECT_DRIVER__VFS:
			if (p_vfs_fallback__read_file_iov)
			{
				if (!__gpuDirectAllocDMABufferOnDemand())
					return false;
				return (p_vfs_fallback__read_file_iov(pathname,
													  m_segment,
													  m_offset,
													  gpudirect_vfs_dma_buffer,
													  gpudirect_vfs_dma_buffer_sz,
													  NULL,
													  iovec,
													  p_npages_direct_read,
													  p_npages_vfs_read) == 0);
			}
			break;
		default:
			break;
	}
	/* fallback using regular filesystem */
	if (!__gpuDirectAllocDMABufferOnDemand())
		return false;
	return __fallbackFileReadIOV(pathname,
								 m_segment,
								 m_offset,
								 gpudirect_vfs_dma_buffer,
								 gpudirect_vfs_dma_buffer_sz,
								 iovec,
								 p_npages_direct_read,
								 p_npages_vfs_read);
}

/*
 * gpuDirectFileReadAsyncIOV
 */
static int	(*p_cufile__read_file_async_iov_v3)(
	const char *pathname,
	CUdeviceptr m_segment,
	off_t m_offset,
	const strom_io_vector *iovec,
	CUstream cuda_stream,
	uint32_t *p_error_code_async,
	uint32_t *p_npages_direct_read,
	uint32_t *p_npages_vfs_read) = NULL;

bool
gpuDirectFileReadAsyncIOV(const char *pathname,
						  CUdeviceptr m_segment,
						  off_t m_offset,
						  unsigned long iomap_handle,
						  const strom_io_vector *iovec,
						  CUstream cuda_stream,
						  uint32_t *p_error_code_async,
						  uint32_t *p_npages_direct_read,
						  uint32_t *p_npages_vfs_read)
{
	switch (gpudirect_driver_kind)
	{
		case GPUDIRECT_DRIVER__CUFILE:
			if (p_cufile__read_file_iov_v3)
				return (p_cufile__read_file_async_iov_v3(pathname,
														 m_segment,
														 m_offset,
														 iovec,
														 cuda_stream,
														 p_error_code_async,
														 p_npages_direct_read,
														 p_npages_vfs_read) == 0);
			break;
		case GPUDIRECT_DRIVER__NVME_STROM:
			if (p_nvme_strom__read_file_iov)
				return (p_nvme_strom__read_file_iov(pathname,
													iomap_handle,
													m_offset,
													iovec,
													p_npages_direct_read,
													p_npages_vfs_read) == 0);
			break;
		case GPUDIRECT_DRIVER__VFS:
			if (p_vfs_fallback__read_file_iov)
			{
				if (!__gpuDirectAllocDMABufferOnDemand())
					return false;
				return (p_vfs_fallback__read_file_iov(pathname,
													  m_segment,
													  m_offset,
													  gpudirect_vfs_dma_buffer,
													  gpudirect_vfs_dma_buffer_sz,
                                                      cuda_stream,
                                                      iovec,
													  p_npages_direct_read,
                                                      p_npages_vfs_read) == 0);
			}
			break;
        default:
			break;
	}
	/* fallback using regular filesystem */
	if (!__gpuDirectAllocDMABufferOnDemand())
		return false;
	return __fallbackFileReadIOV(pathname,
								 m_segment,
								 m_offset,
								 gpudirect_vfs_dma_buffer,
								 gpudirect_vfs_dma_buffer_sz,
								 iovec,
								 p_npages_direct_read,
								 p_npages_vfs_read);
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
 * gpuDirectCleanUpOnThreadTerminate
 */
void
gpuDirectCleanUpOnThreadTerminate(void)
{
	CUresult	rc;

	/* release gpudirect_vfs_dma_buffer, if any */
	if (gpudirect_vfs_dma_buffer)
	{
		rc = cuMemFreeHost(gpudirect_vfs_dma_buffer);
		if (rc != CUDA_SUCCESS)
			fprintf(stderr, "failed on cuMemFreeHost(%p)\n", gpudirect_vfs_dma_buffer);
		gpudirect_vfs_dma_buffer = NULL;
		gpudirect_vfs_dma_buffer_sz = 0UL;
	}
}

/*
 * gpuDirectIsSupported
 */
bool
gpuDirectIsAvailable(void)
{
	bool	has_gpudirectsql_supported = false;

	if ((p_cufile__driver_open_v2 &&
		 p_cufile__driver_close_v2 &&
		 p_cufile__map_gpu_memory_v2 &&
		 p_cufile__unmap_gpu_memory_v2 &&
		 p_cufile__read_file_iov_v3) ||
		(p_nvme_strom__driver_open &&
		 p_nvme_strom__driver_close &&
		 p_nvme_strom__map_gpu_memory &&
		 p_nvme_strom__unmap_gpu_memory &&
		 p_nvme_strom__read_file_iov))
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
								 bool *p_has_cufile,
								 bool *p_has_nvme_strom)
{
	char   *buffer;
	long	api_version = 0;
	bool	has_cufile = false;
	bool	has_nvme_strom = false;
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
	*p_api_version		= api_version;
	*p_has_cufile		= has_cufile;
	*p_has_nvme_strom   = has_nvme_strom;
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
	uint32_t	api_version = 0;
	bool		has_cufile = false;
	bool		has_nvme_strom = false;
	int			enum_index = 0;
	static struct config_enum_entry enum_options[4];

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

	/* lookup extra symbols */
	LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_module_init);
	LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_get_error);
	extra_module_info = heterodbExtraModuleInit();
	parse_heterodb_extra_module_info(extra_module_info,
									 &api_version,
									 &has_cufile,
									 &has_nvme_strom);
	LOOKUP_HETERODB_EXTRA_FUNCTION(gpudirect__driver_init_v2);
	if (has_cufile)
	{
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__driver_open_v2);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__driver_close_v2);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__map_gpu_memory_v2);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__unmap_gpu_memory_v2);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__register_stream_v3);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__deregister_stream_v3);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__read_file_iov_v3);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__read_file_async_iov_v3);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__get_property_v2);
		LOOKUP_HETERODB_EXTRA_FUNCTION(cufile__set_property_v2);
	}
	if (has_nvme_strom)
	{
		LOOKUP_HETERODB_EXTRA_FUNCTION(nvme_strom__driver_open);
		LOOKUP_HETERODB_EXTRA_FUNCTION(nvme_strom__driver_close);
		LOOKUP_HETERODB_EXTRA_FUNCTION(nvme_strom__map_gpu_memory);
		LOOKUP_HETERODB_EXTRA_FUNCTION(nvme_strom__unmap_gpu_memory);
		LOOKUP_HETERODB_EXTRA_FUNCTION(nvme_strom__read_file_iov);
	}
	LOOKUP_HETERODB_EXTRA_FUNCTION(vfs_fallback__read_file_iov);
	if (has_cufile || has_nvme_strom)
		gpuDirectInitDriver();
	LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_reload);
	LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_query);
	LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_validate_device);
	elog(LOG, "HeteroDB Extra module loaded [%s]", extra_module_info);

	memset(enum_options, 0, sizeof(enum_options));
	if (has_cufile)
	{
		enum_options[enum_index].name = "cufile";
		enum_options[enum_index].val  = GPUDIRECT_DRIVER__CUFILE;
		enum_index++;
	}
	if (has_nvme_strom)
	{
		enum_options[enum_index].name = "nvme_strom";
		enum_options[enum_index].val  = GPUDIRECT_DRIVER__NVME_STROM;
		enum_index++;
	}
	enum_options[enum_index].name = "vfs";
	enum_options[enum_index].val  = GPUDIRECT_DRIVER__VFS;
	enum_index++;

	DefineCustomEnumVariable("pg_strom.gpudirect_driver",
							 "Choice of GPU-Direct SQL Driver",
							 NULL,
							 &gpudirect_driver_kind,
							 enum_options[0].val,
							 enum_options,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	license = __heterodb_license_query();
	if (license)
	{
		elog(LOG, "HeteroDB License: %s", license);
		pfree(license);
	}
}
