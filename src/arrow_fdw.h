/*
 * arrow_fdw.h
 *
 * Routines related to Arrow/Parquet FDW support.
 * 
 * (*) Some C++ code cannot include PostgreSQL definitions, so we put separated
 *     definitions here.
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef _ARROW_FDW_H_
#define _ARROW_FDW_H_
#include "arrow_defs.h"
#include <cuda.h>

#ifdef __cplusplus
#define __EXTERN	extern "C"
#else
#define __EXTERN	extern
#endif

struct kern_data_store;
struct kern_colmeta;

/*
 * parquetCacheWriteAsyncData
 */
struct parquetCacheWriteAsyncData
{
	struct dlist_node chain;
	struct stat	pq_fstat;
	int32_t		rg_index;
	int32_t		field_id;
	const uint8_t *nullmap_ptr;
	size_t		nullmap_len;
	const uint8_t *values_ptr;
	size_t		values_len;
	const uint8_t *extra_ptr;
	size_t		extra_len;
#ifdef __cplusplus
	parquetCacheWriteAsyncData(const struct stat *__pq_fstat,
							   int32_t __rg_index,
							   int32_t __field_id)
	{
		memcpy(&pq_fstat, __pq_fstat, sizeof(struct stat));
		rg_index = __rg_index;
		field_id = __field_id;
		nullmap_ptr = NULL;
		nullmap_len = 0;
		values_ptr = NULL;
		values_len = 0;
		extra_ptr = NULL;
		extra_len = 0;
	}
#endif
};
typedef struct parquetCacheWriteAsyncData	parquetCacheWriteAsyncData;

/*
 * parquet_read.cc (C++ interface)
 */
__EXTERN struct kern_data_store *
parquetReadOneRowGroup(const char *filename,
					   const struct kern_data_store *kds_head,
					   bool  (*malloc_gpu_callback)(void *malloc_private,
													size_t malloc_size,
													CUdeviceptr *m_segment,
													off_t *m_offset),
					   void *(*malloc_host_callback)(void *malloc_private,
													 size_t malloc_size),
					   void *malloc_private,
					   uint32_t *p_npages_direct_read,
					   uint32_t *p_npages_vfs_read,
					   char *error_message,
					   size_t error_message_sz);
__EXTERN void
releaseParquetCacheWriteAsyncData(parquetCacheWriteAsyncData *data);

/*
 * parquet_cache.c
 */
__EXTERN void *
parquet_nvme_cache_lookup(const struct stat *pq_fstat,
						  int32_t rg_index,
						  int32_t field_id);
__EXTERN void
parquet_nvme_cache_release(void *entry);
__EXTERN size_t
parquet_nvme_cache_length(const void *entry);
__EXTERN ssize_t
parquet_nvme_cache_read_chunks(void *entry,
							   struct kern_colmeta *cmeta,
							   size_t kds_offset,
							   CUdeviceptr m_segment,
							   off_t m_offset,
							   uint32_t *p_npages_direct_read,
							   uint32_t *p_npages_vfs_read);

__EXTERN void
parquetCacheWriteAsync(parquetCacheWriteAsyncData *data);

__EXTERN void
parquet_nvme_cache_write_async(const struct stat *pq_fstat,
							   int32_t rg_index,
							   int32_t field_id,
							   const char *nullmap_ptr,
							   size_t nullmap_len,
							   const char *values_ptr,
							   size_t values_len,
							   const char *extra_ptr,
							   size_t extra_len,
							   void (*buffer_release_callback)(void *data),
							   void *buffer_release_data);
__EXTERN void		pgstrom_init_parquet_cache(void);
#undef __EXTERN
#endif	/* _ARROW_FDW_H_ */
