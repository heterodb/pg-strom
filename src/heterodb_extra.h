/*
 * heterodb_extra.h
 *
 * Definitions of HeteroDB Extra Package
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2017-2021 (C) HeteroDB,Inc
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef HETERODB_EXTRA_H
#define HETERODB_EXTRA_H
#include <stdbool.h>
#include <stdint.h>

#define HETERODB_EXTRA_FILENAME		"heterodb_extra.so"
#define HETERODB_EXTRA_PATHNAME		"/usr/lib64/" HETERODB_EXTRA_FILENAME
#define HETERODB_EXTRA_MAX_GPUS		63

#define HETERODB_LICENSE_PATHNAME	"/etc/heterodb.license"
/* fixed length of the license key (2048bits) */
#define HETERODB_LICENSE_KEYLEN		256
#define HETERODB_LICENSE_KEYBITS	(8 * HETERODB_LICENSE_KEYLEN)

#define HETERODB_EXTRA_CURRENT_API_VERSION	20240725
#define HETERODB_EXTRA_OLDEST_API_VERSION	20240418

/* cufile.c */
typedef struct
{
	unsigned long	m_offset;	/* destination offset from the base address
								 * base = mgmem + offset */
	unsigned int	fchunk_id;	/* source page index of the file. */
	unsigned int	nr_pages;	/* number of pages to be loaded */
} strom_io_chunk;

typedef struct
{
	unsigned int	nr_chunks;
	strom_io_chunk	ioc[1];
} strom_io_vector;

/*
 * extra.c
 */
extern const char  *heterodb_extra_init_module(const char *extra_pathname);
extern bool			heterodb_extra_parse_signature(const char *signature,
												   uint32_t *p_api_version,
												   bool *p_has_cufile,
												   bool *p_has_nvme_strom);
extern void			heterodbExtraSetError(int errcode,
										  const char *filename,
										  unsigned int lineno,
										  const char *funcname,
										  const char *fmt, ...)
#if defined(pg_attribute_printf)
	pg_attribute_printf(5,6)
#elif defined(__GNUC__)
	__attribute__((format(gnu_printf, 5,6)))
#endif
	;
extern int			heterodbExtraGetError(const char **p_filename,
										  unsigned int *p_lineno,
										  const char **p_funcname,
										  char *buffer, size_t buffer_sz);
extern int			heterodbLicenseReload(void);
extern int			heterodbLicenseReloadPath(const char *path);
extern ssize_t		heterodbLicenseQuery(char *buf, size_t bufsz);
extern const char  *heterodbLicenseDecrypt(const char *path);
extern int			heterodbValidateDevice(const char *gpu_device_name,
										   const char *gpu_device_uuid);
extern const char  *heterodbInitOptimalGpus(const char *manual_config);
extern int64_t		heterodbGetOptimalGpus(const char *path,
										   const char *policy);
/*
 * GPU related stuff must include <cuda.h> first, but DPU stuff skips them
 */
#ifdef __cuda_cuda_h__
extern bool			gpuDirectInitDriver(void);
extern bool			gpuDirectOpenDriver(void);
extern bool			gpuDirectCloseDriver(void);
extern bool			gpuDirectMapGpuMemory(CUdeviceptr m_segment,
										  size_t segment_sz,
										  unsigned long *p_iomap_handle);
extern bool			gpuDirectUnmapGpuMemory(CUdeviceptr m_segment,
											unsigned long iomap_handle);
extern bool			gpuDirectRegisterStream(CUstream cuda_stream);
extern bool			gpuDirectDeregisterStream(CUstream cuda_stream);
extern bool			gpuDirectFileReadIOV(const char *pathname,
										 CUdeviceptr m_segment,
										 off_t m_offset,
										 unsigned long iomap_handle,
										 const strom_io_vector *iovec,
										 uint32_t *p_npages_direct_read,
										 uint32_t *p_npages_vfs_read);
extern bool			gpuDirectFileReadAsyncIOV(const char *pathname,
											  CUdeviceptr m_segment,
											  off_t m_offset,
											  unsigned long iomap_handle,
											  const strom_io_vector *iovec,
											  CUstream cuda_stream,
											  uint32_t *p_error_code_async,
											  uint32_t *p_npages_direct_read,
											  uint32_t *p_npages_vfs_read);
#endif	/* __cuda_cuda_h__ */
extern const char  *gpuDirectGetProperty(void);
extern bool			gpuDirectSetProperty(const char *key,
										 const char *value);
extern void			gpuDirectCleanUpOnThreadTerminate(void);
extern bool			heterodbExtraCloudGetVMInfo(const char *cloud_name,
												const char **p_vm_type,
												const char **p_vm_image,
												const char **p_vm_ident);
#endif	/* HETERODB_EXTRA_H */
