/*
 * parquet_read.cpp
 *
 * Routines to read Parquet files
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <iostream>
#include <memory>
#include <mutex>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/type.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/api/reader.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include "xpu_common.h"
#include "arrow_fdw.h"

/* version check: libarrow/libparquet v23 is minimum requirement */
#if PARQUET_VERSION_MAJOR < 23
#error libarrow/libparquet must be version 23 or later
#endif

/*
 * Error Reporting
 */
#define __Elog(fmt,...)										\
	do {													\
		char	__error_message_buffer[320];				\
		snprintf(__error_message_buffer,					\
				 sizeof(__error_message_buffer),			\
				 "[ERROR %s:%d] " fmt,						\
				 __basename(__FILE__),__LINE__,				\
				 ##__VA_ARGS__);							\
		throw std::runtime_error(__error_message_buffer);	\
	} while(0)

/*
 * Debug support
 */
extern "C" int		arrow_metadata_cache_size_kb;	/* GUC */
extern "C" void		pgstrom_inc_perf_counter(int num);
extern "C" void		pgstrom_add_perf_counter(int num, const struct timeval *tv_base);
extern "C" uint32_t hash_bytes(const unsigned char *k, int keylen);

/*
 * Parquet Metadata Cache
 */
using parquetFileMetadataCache  = struct ParquetFileMetadataCache;
using parquetFileMetadataChain  = struct __dlist_node<parquetFileMetadataCache>;
struct ParquetFileMetadataCache
{
	uint32_t	hash;
	parquetFileMetadataChain hash_chain;
	parquetFileMetadataChain lru_chain;
	std::string filename;
	struct stat stat_buf;
	int			metadata_status;
	std::shared_ptr<parquet::FileMetaData> metadata;
	ParquetFileMetadataCache()
	{
		hash_chain.owner = this;
		lru_chain.owner = this;
	}
};
static pthread_mutex_t		parquet_file_metadata_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t		parquet_file_metadata_cond = PTHREAD_COND_INITIALIZER;
static size_t				parquet_file_metadata_usage = 0;
#define PARQUET_FILE_METADATA_NSLOTS    6000
static parquetFileMetadataChain parquet_file_metadata_hash[PARQUET_FILE_METADATA_NSLOTS];
static parquetFileMetadataChain parquet_file_metadata_lru;

static inline void
__parquetFileMetadataMutexLock(void)
{
	if ((errno = pthread_mutex_lock(&parquet_file_metadata_mutex)) != 0)
	{
		fprintf(stderr, "[FATAL] failed on pthread_mutex_lock: %m\n");
		_exit(1);
	}
}
static inline void
__parquetFileMetadataMutexUnlock(void)
{
	if ((errno = pthread_mutex_unlock(&parquet_file_metadata_mutex)) != 0)
	{
		fprintf(stderr, "[FATAL] failed on pthread_mutex_unlock: %m\n");
		_exit(1);
	}
}
static inline void
__parquetFileMetadataMutexWait(void)
{
	if ((errno = pthread_cond_wait(&parquet_file_metadata_cond,
								   &parquet_file_metadata_mutex)) != 0)
	{
		fprintf(stderr, "[FATAL] failed on pthread_cond_wait: %m\n");
		_exit(1);
	}
}
static inline void
__parquetFileMetadataMutexNotify(void)
{
	if ((errno = pthread_cond_broadcast(&parquet_file_metadata_cond)) != 0)
	{
		fprintf(stderr, "[FATAL] failed on pthread_cond_broadcast: %m\n");
		_exit(1);
	}
}

static void
__parquetFileMetadataReclaim(size_t additional_usage)
{
	size_t	parquet_file_metadata_limit = ((size_t)arrow_metadata_cache_size_kb << 10);
	ParquetFileMetadataCache *entry;

	parquet_file_metadata_usage += additional_usage;
	__dlist_foreach(entry, &parquet_file_metadata_lru)
	{
		size_t		sz;

		if (parquet_file_metadata_usage <= parquet_file_metadata_limit)
			break;
		sz = sizeof(ParquetFileMetadataCache);
		if (entry->metadata)
			sz += entry->metadata->size();
		__dlist_delete(&entry->hash_chain);
		__dlist_delete(&entry->lru_chain);
		delete(entry);
		parquet_file_metadata_usage -= sz;
	}
}

static std::shared_ptr<parquet::FileMetaData>
__readParquetFileMetadata(std::shared_ptr<arrow::io::ReadableFile> filp, const char *fname)
{
	struct {
		uint32_t	metadata_len;
		char		signature[4];
	} foot;
	size_t			length;
	std::shared_ptr<arrow::Buffer> buffer;
	/* fetch metadata length */
	{
		auto	rv = filp->GetSize();
		if (!rv.ok())
			__Elog("failed on arrow::io::ReadableFile('%s')::GetSize: %s",
				   fname, rv.status().ToString().c_str());
		length = rv.ValueOrDie();
	}
	/* read footer 8bytes */
	{
		auto	rv = filp->ReadAt(length-sizeof(foot), sizeof(foot), &foot);
		if (!rv.ok())
			__Elog("failed on arrow::io::ReadableFile('%s')::ReadAt: %s",
				   fname, rv.status().ToString().c_str());
		if (rv.ValueOrDie() != sizeof(foot) ||
			foot.metadata_len + sizeof(foot) > length ||
			memcmp(foot.signature, "PAR1", 4) != 0)
			__Elog("signature check failed: file '%s' is not Parquet", fname);
	}
	/* read binary metadata */
	{
		auto	rv = filp->ReadAt(length -
								  sizeof(foot) -
								  foot.metadata_len,
								  foot.metadata_len);
		if (!rv.ok())
			__Elog("failed on arrow::io::ReadableFile('%s')::ReadAt: %s",
				   fname, rv.status().ToString().c_str());
		buffer = rv.ValueOrDie();
	}
	return parquet::FileMetaData::Make(buffer->data(), &foot.metadata_len);
}

static std::shared_ptr<parquet::FileMetaData>
lookupParquetFileMetadata(std::shared_ptr<arrow::io::ReadableFile> filp, const char *fname)
{
	int			fdesc = filp->file_descriptor();
	uint32_t	hash, hindex;
	struct stat	stat_buf;
	struct {
		dev_t   st_dev;
		ino_t   st_ino;
		struct timespec st_mtim;
	} hkey;
	parquetFileMetadataCache *pq_mcache;
	std::shared_ptr<parquet::FileMetaData> metadata = nullptr;

	/* hash value */
	if (fstat(fdesc, &stat_buf) != 0)
		__Elog("failed on fstat('%s'): %m", fname);
	memset(&hkey, 0, sizeof(hkey));
	hkey.st_dev  = stat_buf.st_dev;
	hkey.st_ino  = stat_buf.st_ino;
	hkey.st_mtim = stat_buf.st_mtim;
	hash = hash_bytes((unsigned char *)&hkey, sizeof(hkey));

	/* lookup hash table */
	hindex = hash % PARQUET_FILE_METADATA_NSLOTS;
	__parquetFileMetadataMutexLock();
again:
	__dlist_foreach(pq_mcache, &parquet_file_metadata_hash[hindex])
	{
		if (pq_mcache->hash                     == hash &&
			pq_mcache->stat_buf.st_dev          == stat_buf.st_dev &&
			pq_mcache->stat_buf.st_ino          == stat_buf.st_ino &&
			pq_mcache->stat_buf.st_mtim.tv_sec  == stat_buf.st_mtim.tv_sec &&
			pq_mcache->stat_buf.st_mtim.tv_nsec == stat_buf.st_mtim.tv_nsec)
		{
			/* found */
			__dlist_move_tail(&parquet_file_metadata_lru,
							  &pq_mcache->lru_chain);
			if (pq_mcache->metadata_status == 0)
			{
				__parquetFileMetadataMutexWait();
				goto again;
			}
			if (pq_mcache->metadata_status > 0)
				metadata = pq_mcache->metadata;
			__parquetFileMetadataMutexUnlock();
			return metadata;
		}
	}
	/* not found, so insert a WIP entry */
	try {
		assert(pq_mcache == NULL);
		pq_mcache = new(parquetFileMetadataCache);
		pq_mcache->hash = hash;
		pq_mcache->filename = std::string(fname);
		memcpy(&pq_mcache->stat_buf, &stat_buf, sizeof(struct stat));
		pq_mcache->metadata_status = 0;		/* still building */
		pq_mcache->metadata = nullptr;
		__dlist_push_tail(&parquet_file_metadata_hash[hindex], &pq_mcache->hash_chain);
		__dlist_push_tail(&parquet_file_metadata_lru,          &pq_mcache->lru_chain);
		__parquetFileMetadataReclaim(sizeof(parquetFileMetadataCache));
	}
	catch(const std::exception &e) {
		if (pq_mcache)
			delete(pq_mcache);
		__parquetFileMetadataMutexUnlock();
		throw;
	}
	__parquetFileMetadataMutexUnlock();
	/* read parquet file metadata */
	try {
		metadata = __readParquetFileMetadata(filp, fname);
	}
	catch(const std::exception &e) {
		__parquetFileMetadataMutexLock();
		pq_mcache->metadata_status = -1;
		__parquetFileMetadataMutexUnlock();
		__parquetFileMetadataMutexNotify();
		throw;
	}
	/* assign metadata */
	__parquetFileMetadataMutexLock();
	pq_mcache->metadata_status = 1;
	pq_mcache->metadata = metadata;
	__dlist_move_tail(&parquet_file_metadata_lru,
					  &pq_mcache->lru_chain);
	__parquetFileMetadataReclaim(metadata->size());
	__parquetFileMetadataMutexUnlock();
	__parquetFileMetadataMutexNotify();
	return metadata;
}

/*
 * setupParquetFileReferenced
 */
static void
__setupParquetFileReferenced(const parquet::arrow::SchemaField &schema_field,
							 std::vector<int> &referenced)
{
	if (schema_field.column_index >= 0)
		referenced.push_back(schema_field.column_index);
	for (int k=0; k < schema_field.children.size(); k++)
		__setupParquetFileReferenced(schema_field.children[k], referenced);
}

static void
setupParquetFileReferenced(const kern_data_store *kds_head,
						   const parquet::arrow::SchemaManifest &manifest,
						   bool consider_cached_chunks,
						   std::vector<void *> &parquet_cached_chunks,
						   std::vector<int> &referenced,
						   std::vector<int> &revmap)
{
	for (int j=0; j < kds_head->ncols; j++)
	{
		int		field_index = kds_head->colmeta[j].field_index;

		if (field_index < 0)
			continue;
		if (consider_cached_chunks && parquet_cached_chunks[field_index] != NULL)
			continue;
		if (field_index < manifest.schema_fields.size())
		{
			__setupParquetFileReferenced(manifest.schema_fields[j], referenced);
			revmap.push_back(j);
		}
		else
		{
			__Elog("not compatible Schema: field index %d out of range [%ld]",
				   field_index, manifest.schema_fields.size());
		}
	}
}

/*
 * lengthOfArrowColumnBuffer
 */
static size_t
lengthOfArrowColumnBuffer(std::shared_ptr<arrow::Array> chunk)
{
	auto		data = chunk->data();
	size_t		len = 0;

	for (const auto &buf : data->buffers)
	{
		if (buf)
			len += ARROW_ALIGN(buf->size());
	}
	/* dive into nested items */
	switch (chunk->type_id())
	{
		case arrow::Type::LIST: {
			auto list = std::static_pointer_cast<arrow::ListArray>(chunk);
			len += lengthOfArrowColumnBuffer(list->values());
			break;
		}
		case arrow::Type::LARGE_LIST: {
			auto list = std::static_pointer_cast<arrow::LargeListArray>(chunk);
			len += lengthOfArrowColumnBuffer(list->values());
			break;
		}
		case arrow::Type::STRUCT: {
			auto	comp = std::static_pointer_cast<arrow::StructArray>(chunk);
			for (const auto &child : comp->fields())
				len += lengthOfArrowColumnBuffer(child);
			break;
		}
		default:
			/* no nested entry */
			break;
	}
	return len;
}

/*
 * parquetCacheWriteAsyncDataArray
 */
struct parquetCacheWriteAsyncDataArray
	: parquetCacheWriteAsyncData
{
	std::shared_ptr<arrow::Array> array;
	parquetCacheWriteAsyncDataArray(const struct stat *__pq_fstat,
									int32_t __rg_index,
									int32_t __field_id)
		: parquetCacheWriteAsyncData(__pq_fstat, __rg_index, __field_id)
	{}
};
typedef struct parquetCacheWriteAsyncDataArray	parquetCacheWriteAsyncDataArray;

void
releaseParquetCacheWriteAsyncData(parquetCacheWriteAsyncData *__data)
{
	parquetCacheWriteAsyncDataArray *data = (parquetCacheWriteAsyncDataArray *)__data;
	/* unpin arrow::Array */
	delete(data);
}

/*
 * fillupArrowColumnBuffer
 */
static void
fillupArrowColumnBuffer(kern_data_store *kds,
						kern_colmeta *cmeta,
						CUdeviceptr m_kds_base,
						std::shared_ptr<arrow::Array> array,
						const struct stat *pq_fstat)
{
	parquetCacheWriteAsyncDataArray *data = nullptr;
	auto	dtype = array->type();
	auto	adata = array->data();
	bool	needs_nullmap_buffer = (array->null_count() > 0);
	bool	needs_main_buffer = true;
	bool	needs_extra_buffer = false;
	CUresult rc;

	switch (dtype->id())
	{
		case arrow::Type::type::NA:
			if (cmeta->attopts.tag == ArrowType__Null)
			{
				needs_nullmap_buffer = false;
				goto simple_buffer;
			}
			break;
		case arrow::Type::type::BOOL:
			if (cmeta->attopts.tag == ArrowType__Bool)
				goto simple_buffer;
			break;
		case arrow::Type::type::INT8:
		case arrow::Type::type::INT16:
		case arrow::Type::type::INT32:
		case arrow::Type::type::INT64:
			if (cmeta->attopts.tag == ArrowType__Int &&
				cmeta->attopts.integer.bitWidth == dtype->bit_width() &&
				cmeta->attopts.integer.is_signed)
				goto simple_buffer;
			break;
		case arrow::Type::type::UINT8:
		case arrow::Type::type::UINT16:
		case arrow::Type::type::UINT32:
		case arrow::Type::type::UINT64:
			if (cmeta->attopts.tag == ArrowType__Int &&
				cmeta->attopts.integer.bitWidth == dtype->bit_width() &&
				!cmeta->attopts.integer.is_signed)
				goto simple_buffer;
			break;
		case arrow::Type::type::HALF_FLOAT:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Half)
				goto simple_buffer;
			break;
		case arrow::Type::type::FLOAT:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Single)
				goto simple_buffer;
			break;
		case arrow::Type::type::DOUBLE:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Double)
				goto simple_buffer;
			break;
		case arrow::Type::type::DECIMAL128:
			if (cmeta->attopts.tag == ArrowType__Decimal &&
				cmeta->attopts.decimal.bitWidth == 128)
			{
				const auto	__dtype = std::static_pointer_cast<arrow::Decimal128Type>(dtype);
				if (cmeta->attopts.decimal.precision == __dtype->precision() &&
					cmeta->attopts.decimal.scale     == __dtype->scale())
					goto simple_buffer;
			}
			break;
		case arrow::Type::type::DATE32:
		case arrow::Type::type::DATE64:
			if (cmeta->attopts.tag == ArrowType__Date)
			{
				auto	__dtype = std::static_pointer_cast<arrow::DateType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::DateUnit::DAY:
						if (cmeta->attopts.date.unit == ArrowDateUnit__Day)
							goto simple_buffer;
						break;
					case arrow::DateUnit::MILLI:
						if (cmeta->attopts.date.unit == ArrowDateUnit__MilliSecond)
							goto simple_buffer;
						break;
					default:
						__Elog("Bug? unknown Date unit (%d)", (int)__dtype->unit());
				}
			}
			break;
		case arrow::Type::type::TIME32:
		case arrow::Type::type::TIME64:
			if (cmeta->attopts.tag == ArrowType__Time)
			{
				auto	__dtype = std::static_pointer_cast<arrow::TimeType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::TimeUnit::SECOND:
						if (cmeta->attopts.time.unit == ArrowTimeUnit__Second)
							goto simple_buffer;
						break;
					case arrow::TimeUnit::MILLI:
						if (cmeta->attopts.time.unit == ArrowTimeUnit__MilliSecond)
							goto simple_buffer;
						break;
					case arrow::TimeUnit::MICRO:
						if (cmeta->attopts.time.unit == ArrowTimeUnit__MicroSecond)
							goto simple_buffer;
						break;
					case arrow::TimeUnit::NANO:
						if (cmeta->attopts.time.unit == ArrowTimeUnit__NanoSecond)
							goto simple_buffer;
						break;
					default:
						__Elog("Bug? unknown Time unit (%d)", (int)__dtype->unit());
				}
			}
			break;
		case arrow::Type::type::TIMESTAMP:
			if (cmeta->attopts.tag == ArrowType__Timestamp)
			{
				auto	__dtype = std::static_pointer_cast<arrow::TimestampType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::TimeUnit::SECOND:
						if (cmeta->attopts.timestamp.unit == ArrowTimeUnit__Second)
							goto simple_buffer;
						break;
					case arrow::TimeUnit::MILLI:
						if (cmeta->attopts.timestamp.unit == ArrowTimeUnit__MilliSecond)
							goto simple_buffer;
						break;
					case arrow::TimeUnit::MICRO:
						if (cmeta->attopts.timestamp.unit == ArrowTimeUnit__MicroSecond)
							goto simple_buffer;
						break;
					case arrow::TimeUnit::NANO:
						if (cmeta->attopts.timestamp.unit == ArrowTimeUnit__NanoSecond)
							goto simple_buffer;
						break;
					default:
						__Elog("Bug? unknown Timestamp unit (%d)", (int)__dtype->unit());
				}
			}
			break;
		case arrow::Type::type::FIXED_SIZE_BINARY:
			if (cmeta->attopts.tag == ArrowType__FixedSizeBinary &&
				cmeta->attopts.fixed_size_binary.byteWidth == dtype->byte_width())
				goto simple_buffer;
			break;
		case arrow::Type::type::INTERVAL_MONTHS:
		case arrow::Type::type::INTERVAL_DAY_TIME:
			if (cmeta->attopts.tag == ArrowType__Interval &&
				cmeta->attopts.unitsz == dtype->byte_width())
				goto simple_buffer;
			break;
		case arrow::Type::type::STRING:
			if (cmeta->attopts.tag == ArrowType__Utf8)
				goto variable_buffer;
			break;
		case arrow::Type::type::BINARY:
			if (cmeta->attopts.tag == ArrowType__Binary)
				goto variable_buffer;
			break;
		case arrow::Type::type::LARGE_STRING:
			if (cmeta->attopts.tag == ArrowType__LargeUtf8)
				goto variable_buffer;
			break;
		case arrow::Type::type::LARGE_BINARY:
			if (cmeta->attopts.tag == ArrowType__LargeBinary)
				goto variable_buffer;
			break;
		case arrow::Type::type::LIST:
			if (cmeta->attopts.tag == ArrowType__List)
			{
				auto	list = std::static_pointer_cast<arrow::ListArray>(array);
				assert(cmeta->idx_subattrs >= kds->ncols &&
					   cmeta->idx_subattrs <  kds->nr_colmeta &&
					   cmeta->num_subattrs == 1);
				pq_fstat = NULL;	/* no parquet cache */
				fillupArrowColumnBuffer(kds,
										&kds->colmeta[cmeta->idx_subattrs],
										m_kds_base,
										list->values(),
										NULL);
				goto simple_buffer;
			}
			break;
		case arrow::Type::type::LARGE_LIST:
			if (cmeta->attopts.tag == ArrowType__LargeList)
			{
				auto	list = std::static_pointer_cast<arrow::LargeListArray>(array);
				assert(cmeta->idx_subattrs >= kds->ncols &&
					   cmeta->idx_subattrs <  kds->nr_colmeta &&
					   cmeta->num_subattrs == 1);
				pq_fstat = NULL;	/* no parquet cache */
				fillupArrowColumnBuffer(kds,
										&kds->colmeta[cmeta->idx_subattrs],
										m_kds_base,
										list->values(),
										NULL);
				goto simple_buffer;
			}
			break;
		case arrow::Type::type::STRUCT:
			if (cmeta->attopts.tag == ArrowType__Struct)
			{
				auto	comp = std::static_pointer_cast<arrow::StructArray>(array);
				auto	__dtype = std::static_pointer_cast<arrow::StructType>(dtype);
				assert(cmeta->num_subattrs == __dtype->num_fields() &&
					   cmeta->idx_subattrs >= kds->ncols &&
					   cmeta->idx_subattrs +
					   cmeta->num_subattrs <= kds->nr_colmeta);
				pq_fstat = NULL;	/* no parquet cache */
				for (int k=0; k < cmeta->num_subattrs; k++)
				{
					fillupArrowColumnBuffer(kds,
											&kds->colmeta[cmeta->idx_subattrs + k],
											m_kds_base,
											comp->field(k),
											NULL);
				}
				needs_main_buffer = false;
				goto simple_buffer;
			}
			break;
		default:
			break;
	}
	__Elog("Parquet data type mismatch at attname='%s', atttype='%s' arrow-type=%s",
		   cmeta->attname,
		   ArrowTypeTagAsCString(cmeta->attopts.tag),
		   dtype->ToString().c_str());
variable_buffer:
	needs_extra_buffer = true;
simple_buffer:
	if (pq_fstat)
	{
		data = new parquetCacheWriteAsyncDataArray(pq_fstat,
												   kds->parquet_row_group,
												   cmeta->field_index);
		data->array = array;
	}
	try {
		/* nullmap */
		if (needs_nullmap_buffer)
		{
			if (adata->buffers.size() < 1)
				__Elog("corruption? nullmap buffer is missing at '%s'", cmeta->attname);
			auto	buffer = adata->buffers[0];
			assert(buffer->data() != nullptr);
			cmeta->nullmap_offset = kds->usage;
			cmeta->nullmap_length = buffer->size();
			kds->usage += ARROW_ALIGN(cmeta->nullmap_length);
			if (m_kds_base == 0UL)
			{
				memcpy((char *)kds + cmeta->nullmap_offset,
					   buffer->data(),
					   cmeta->nullmap_length);
			}
			else
			{
				rc = cuMemcpyHtoD(m_kds_base + cmeta->nullmap_offset,
								  buffer->data(),
								  cmeta->nullmap_length);
				if (rc != CUDA_SUCCESS)
					__Elog("failed on cuMemcpyHtoD");
			}
			if (data)
			{
				data->nullmap_ptr = buffer->data();
				data->nullmap_len = buffer->size();
			}
		}
		else
		{
			cmeta->nullmap_offset = 0;
			cmeta->nullmap_length = 0;
		}
		/* values/offset */
		if (needs_main_buffer)
		{
			if (adata->buffers.size() < 2)
				__Elog("corruption? values/offset buffer is missing at '%s'", cmeta->attname);
			auto	buffer = adata->buffers[1];
			assert(buffer->data() != nullptr);
			cmeta->values_offset = kds->usage;
			cmeta->values_length = buffer->size();
			kds->usage += ARROW_ALIGN(cmeta->values_length);
			if (m_kds_base == 0UL)
			{
				memcpy((char *)kds + cmeta->values_offset,
					   buffer->data(),
					   cmeta->values_length);
			}
			else
			{
				rc = cuMemcpyHtoD(m_kds_base + cmeta->values_offset,
								  buffer->data(),
								  cmeta->values_length);
				if (rc != CUDA_SUCCESS)
					__Elog("failed on cuMemcpyHtoD");
			}
			if (data)
			{
				data->values_ptr = buffer->data();
				data->values_len = buffer->size();
			}
		}
		else
		{
			cmeta->values_offset = 0;
			cmeta->values_length = 0;
		}
		/* extra */
		if (needs_extra_buffer)
		{
			if (adata->buffers.size() < 3)
				__Elog("corruption? extra buffer is missing at '%s'", cmeta->attname);
			auto	buffer = adata->buffers[2];
			assert(buffer->data() != nullptr);
			cmeta->extra_offset = kds->usage;
			cmeta->extra_length = buffer->size();
			kds->usage += ARROW_ALIGN(cmeta->extra_length);
			if (m_kds_base == 0UL)
			{
				memcpy((char *)kds + cmeta->extra_offset,
					   buffer->data(),
					   cmeta->extra_length);
			}
			else
			{
				rc = cuMemcpyHtoD(m_kds_base + cmeta->extra_offset,
								  buffer->data(),
								  cmeta->extra_length);
				if (rc != CUDA_SUCCESS)
					__Elog("failed on cuMemcpyHtoD");
			}
			if (data)
			{
				data->extra_ptr = buffer->data();
				data->extra_len = buffer->size();
			}
		}
		else
		{
			cmeta->extra_offset = 0;
			cmeta->extra_length = 0;
		}
		/* write back to the parquet disk cache asynchronously */
		parquetCacheWriteAsync(data);
	}
	catch (const std::exception &e) {
		if (data)
			delete(data);
		throw;
	}
}

/*
 * __parquetReadOneRowGroupFullyCached
 *
 * Fully cached version of RowGroup reader. It returns KDS buffer of GPU raw memory.
 */
static kern_data_store *
__parquetReadOneRowGroupFullyCached(const kern_data_store *kds_head,
									std::vector<void *> &parquet_cached_chunks,
									bool (*malloc_gpu_callback)(void *malloc_private,
																size_t malloc_size,
																CUdeviceptr *m_segment,
																off_t *m_offset),
									void *malloc_private,
									uint32_t *p_npages_direct_read,
									uint32_t *p_npages_vfs_read)
{
	size_t		kds_headsz = KDS_HEAD_LENGTH(kds_head) + kds_head->arrow_virtual_usage;
	size_t		kds_length = PAGE_ALIGN(kds_headsz);
	size_t		kds_offset = kds_length;
	std::vector<char> __buf(kds_headsz);
	kern_data_store *kds_host = (kern_data_store *)__buf.data();
	CUdeviceptr	m_segment;
	off_t		m_offset;
	CUresult	rc;

	/* make a copy of kds_head */
	memcpy(kds_host, kds_head, kds_headsz);
	/* estimate KDS buffer length */
	for (int j=0; j < kds_host->ncols; j++)
	{
		kern_colmeta *cmeta = &kds_host->colmeta[j];
		void	   *pq_cache;
		if (cmeta->field_index < 0)
			continue;
		pq_cache = parquet_cached_chunks[cmeta->field_index];
		kds_length += parquetCacheLength(pq_cache);
	}
	/* allocation of GPU device buffer */
	if (!malloc_gpu_callback(malloc_private, kds_length,
							 &m_segment, &m_offset))
		__Elog("out of memory");
	/* copy to GPU raw memory (and update kds_host) */
	for (int j=0; j < kds_host->ncols; j++)
	{
		kern_colmeta *cmeta = &kds_host->colmeta[j];
		void	   *pq_cache;

		if (cmeta->field_index < 0)
			continue;
		pq_cache = parquet_cached_chunks[cmeta->field_index];
		kds_offset += parquetCacheReadChunks(pq_cache,
											 cmeta,
											 kds_offset,
											 m_segment,
											 m_offset + kds_offset,
											 p_npages_direct_read,
											 p_npages_vfs_read);
	}
	assert(kds_offset <= kds_length);
	kds_host->length = kds_offset;
	kds_host->format = KDS_FORMAT_ARROW;
	rc = cuMemcpyHtoD(m_segment + m_offset, kds_host, kds_headsz);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuMemcpyHtoD");
	/* ok */
	*p_npages_direct_read = (kds_offset - kds_headsz) / PAGE_SIZE;
	return (kern_data_store *)(m_segment + m_offset);
}

/*
 * parquetReadArrowTableSimple
 */
static kern_data_store *
parquetReadArrowTableSimple(std::shared_ptr<arrow::Table> table,
							std::vector<int> &revmap,
							const kern_data_store *kds_head,
							const struct stat *pq_fstat,
							void *(*malloc_cpu_callback)(void *malloc_private,
														 size_t malloc_size),
							void *malloc_private,
							uint32_t *p_npages_vfs_read,
							uint32_t *p_npages_direct_read)
{
	size_t		kds_headsz = ARROW_ALIGN(KDS_HEAD_LENGTH(kds_head) +
										 kds_head->arrow_virtual_usage);
	size_t		kds_length = kds_headsz;
	kern_data_store *kds;

	/* estimation of buffer length */
	assert(kds_head->format == KDS_FORMAT_PARQUET);
	for (const auto &column : table->columns())
	{
		//column = std::shared_ptr<arrow::ChunkedArray>
		assert(column->num_chunks() <= 1);
		for (const auto &chunk : column->chunks())
		{
			//chunk = std::shared_ptr<arrow::Array>
			kds_length += lengthOfArrowColumnBuffer(chunk);
		}
	}
	/* buffer allocation */
	kds = (kern_data_store *)malloc_cpu_callback(malloc_private, kds_length);
	if (!kds)
		__Elog("out of memory");
	memcpy(kds, kds_head, kds_headsz);
	kds->usage = ARROW_ALIGN(kds_headsz);
	for (int j=0; j < table->num_columns(); j++)
	{
		auto	carray = table->column(j);
		auto	cmeta = &kds->colmeta[revmap[j]];
		/* number of items must be identical */
		if (j == 0)
			kds->nitems = carray->length();
		else if (kds->nitems != carray->length())
			__Elog("number of elements mismatch");
		/* chunked-array must be single array */
		assert(carray->num_chunks() <= 1);
		if (carray->num_chunks() > 0)
			fillupArrowColumnBuffer(kds, cmeta, 0UL, carray->chunk(0), pq_fstat);
	}
	assert(kds->usage <= kds_length);
	kds->format = KDS_FORMAT_ARROW;
	kds->length = kds->usage;

	return kds;
}

static kern_data_store *
parquetReadArrowTableWithCache(std::shared_ptr<arrow::Table> table,
							   std::vector<int> &revmap,
							   const kern_data_store *kds_head,
							   const struct stat *pq_fstat,
							   std::vector<void *> &parquet_cached_chunks,
							   bool (*malloc_gpu_callback)(void *malloc_private,
														   size_t malloc_size,
														   CUdeviceptr *m_segment,
														   off_t *m_offset),
							   void *malloc_private,
							   uint32_t *p_npages_vfs_read,
							   uint32_t *p_npages_direct_read)
{
	size_t		kds_headsz = KDS_HEAD_LENGTH(kds_head) + kds_head->arrow_virtual_usage;
	size_t		kds_length = ARROW_ALIGN(kds_headsz);
	std::vector<char> __buf(kds_headsz);
	kern_data_store *kds_host = (kern_data_store *)__buf.data();
	CUdeviceptr	m_segment;
	off_t		m_offset;
	CUresult	rc;

	memcpy(kds_host, kds_head, kds_headsz);
	/* estimation of buffer length */
	assert(kds_host->format == KDS_FORMAT_PARQUET);
	for (const auto &column : table->columns())
	{
		//column = std::shared_ptr<arrow::ChunkedArray>
		assert(column->num_chunks() <= 1);
		for (const auto &chunk : column->chunks())
		{
			//chunk = std::shared_ptr<arrow::Array>
			kds_length += lengthOfArrowColumnBuffer(chunk);
		}
	}
	kds_length = PAGE_ALIGN(kds_length);
	for (int j=0; j < parquet_cached_chunks.size(); j++)
	{
		void   *pq_cache = parquet_cached_chunks[j];
		if (pq_cache)
			kds_length += PAGE_ALIGN(parquetCacheLength(pq_cache));
	}
	/* buffer allocation */
	if (!malloc_gpu_callback(malloc_private,
							 kds_length,
							 &m_segment,
							 &m_offset))
		__Elog("out of memory");
	/* buffer copy (arrow::Table --> GPU memory) */
	kds_host->usage = ARROW_ALIGN(kds_headsz);
	for (int j=0; j < table->num_columns(); j++)
	{
		auto	carray = table->column(j);
		auto	cmeta = &kds_host->colmeta[revmap[j]];
		/* number of items must be identical */
		if (j == 0)
			kds_host->nitems = carray->length();
		else if (kds_host->nitems != carray->length())
			__Elog("number of elements mismatch");
		/* chunked-array must be single array */
		assert(carray->num_chunks() <= 1);
		if (carray->num_chunks() > 0)
			fillupArrowColumnBuffer(kds_host, cmeta,
									m_segment + m_offset,
									carray->chunk(0),
									pq_fstat);
	}
	/* buffer copy (Parquet Cache --> GPU memory) */
	kds_host->usage = PAGE_ALIGN(kds_host->usage);
	for (int j=0; j < parquet_cached_chunks.size(); j++)
	{
		void   *pq_cache = parquet_cached_chunks[j];

		if (pq_cache)
		{
			auto	cmeta = &kds_host->colmeta[revmap[j]];
			
			kds_host->usage
				+= parquetCacheReadChunks(pq_cache,
										  cmeta,
										  kds_host->usage,
										  m_segment,
										  m_offset,
										  p_npages_direct_read,
										  p_npages_vfs_read);
		}
	}
	/* buffer copy (KDS header portion)  */
	assert(kds_host->usage <= kds_length);
	kds_host->format = KDS_FORMAT_ARROW;
	kds_host->length = kds_host->usage;
	rc = cuMemcpyHtoD(m_segment + m_offset,
					  kds_host,
					  kds_headsz);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuMemcpyHtoD");
	/* ok */
	return (kern_data_store *)(m_segment + m_offset);
}

/*
 * __parquetReadOneRowGroupNormal
 *
 * At least one columns must be loaded using libparquet file reader, even though
 * some columns can be fetched from the disk cache.
 */
static kern_data_store *
__parquetReadOneRowGroupNormal(std::shared_ptr<arrow::io::ReadableFile> parquet_filp,
							   const char *filename,
							   const struct stat *stat_buf,
							   const kern_data_store *kds_head,
							   std::vector<void *> &parquet_cached_chunks,
							   bool (*malloc_gpu_callback)(void *malloc_private,
														   size_t malloc_size,
														   CUdeviceptr *m_segment,
														   off_t *m_offset),
							   void *(*malloc_cpu_callback)(void *malloc_private,
															size_t malloc_size),
							   void *malloc_private,
							   uint32_t *p_npages_direct_read,
							   uint32_t *p_npages_vfs_read)
{
	std::unique_ptr<parquet::ParquetFileReader> raw_file_reader = nullptr;
	std::unique_ptr<parquet::arrow::FileReader> arrow_file_reader = nullptr;
	kern_data_store	   *kds = NULL;
	int					row_group_id = kds_head->parquet_row_group;

	/*
	 * Open a new Parquet File Reader dedicated for this thread, but
	 * utilize parquet::FileMetada once parsed by other one.
	 * As discussed in #937, libarrow/libparquet is not designed for
	 * thread-safe, so we use Parquet File Reader per thread.
	 * ----
	 * Open the Parquet File (with cached metadata)
	 */
	{
		auto		metadata = lookupParquetFileMetadata(parquet_filp, filename);
		parquet::ReaderProperties	reader_props;
		reader_props.set_buffer_size(8UL << 20);	/* default buffer size = 8MB */

		raw_file_reader = parquet::ParquetFileReader::Open(parquet_filp,
														   reader_props,
														   metadata);
	}
	/* Open the Arrow file reader */
	{
		auto	rv = parquet::arrow::FileReader::Make(arrow::default_memory_pool(),
													  std::move(raw_file_reader));
		if (!rv.ok())
			__Elog("failed on parquet::arrow::FileReader::Make('%s'): %s",
				   filename,
				   rv.status().ToString().c_str());
		arrow_file_reader = std::move(rv.ValueOrDie());
		/* enables thread-pool of the arrow file reader, if required */
		if (kds_head->parquet_parallel_load)
			arrow_file_reader->set_use_threads(true);
	}
	/* Read the row-group as Arrow Record-Batch */
	if (row_group_id < arrow_file_reader->num_row_groups())
	{
		std::shared_ptr<arrow::Table> table;
		std::vector<int>	referenced;
		std::vector<int>	revmap;
		arrow::Status		status;

		setupParquetFileReferenced(kds_head,
								   arrow_file_reader->manifest(),
								   (malloc_gpu_callback != NULL),
								   parquet_cached_chunks,
								   referenced,
								   revmap);
		status = arrow_file_reader->ReadRowGroup(row_group_id, referenced, &table);
		if (!status.ok())
			__Elog("failed on parquet::arrow::FileReader::ReadRowGroup('%s', %d): %s",
				   filename, row_group_id,
				   status.ToString().c_str());
		/*
		 * In case when the table contains multiple arrays in chunked-array, 
		 * it should be combined to single buffer for simplification, even if
		 * it unlikely happen.
		 */
		for (const auto &column : table->columns())
		{
			if (column->num_chunks() > 1)
			{
				auto	rv = table->CombineChunks();
				if (!rv.ok())
					__Elog("failed on arrow::Table::CombineChunks: %s",
						   rv.status().ToString().c_str());
				table = rv.ValueOrDie();
				break;
			}
		}
		/* load the arrow::Table to KDS */
		if (malloc_gpu_callback)
		{
			kds = parquetReadArrowTableWithCache(table,
												 revmap,
												 kds_head,
												 stat_buf,
												 parquet_cached_chunks,
												 malloc_gpu_callback,
												 malloc_private,
												 p_npages_direct_read,
												 p_npages_vfs_read);
		}
		else
		{
			kds = parquetReadArrowTableSimple(table,
											  revmap,
											  kds_head,
											  stat_buf,
											  malloc_cpu_callback,
											  malloc_private,
											  p_npages_direct_read,
											  p_npages_vfs_read);
		}
	}
	return kds;
}

/*
 * parquetReadOneRowGroup
 */
kern_data_store *
parquetReadOneRowGroup(const char *filename,
					   const kern_data_store *kds_head,
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
					   size_t error_message_sz)
{
	kern_data_store *kds = NULL;
	struct stat		stat_buf;
	std::vector<void *> parquet_cached_chunks;

	try {
		std::shared_ptr<arrow::io::ReadableFile> parquet_filp;
		/*
		 * Open the Parquet file
		 */
		{
			auto    rv = arrow::io::ReadableFile::Open(std::string(filename));
			if (!rv.ok())
				__Elog("failed on arrow::io::ReadableFile::Open('%s'): %s",
					   filename,
					   rv.status().ToString().c_str());
			parquet_filp = rv.ValueOrDie();
			/* also stat file */
			if (fstat(parquet_filp->file_descriptor(), &stat_buf) != 0)
				__Elog("failed on fstat('%s'): %m", filename);
		}
		/*
		 * Try Parquet disk cache if GPU-Direct SQL is available.
		 *
		 * Three cases are possible:
		 *
		 * - No cache hit at all
		 *   The libparquet reader loads the RowGroup and copies it into
		 *   the KDS on managed memory.
		 *   Then, the loaded RowGroup shall be written back to the disk cache
		 *   asynchronously.
		 *
		 * - Partial cache hit
		 *   The libparquet reader loads the RowGroups that were not found
		 *   in the disk cache, and copies them into the KDS on raw memory
		 *   using CUDA API, because cached portion must be loaded vid
		 *   GPU-Direct SQL semantics.
		 *   The columns found in the disk cache are transferred directly
		 *   from the disk cache to GPU memory using GDS.
		 *
		 * - Full cache hit
		 *   Without using libparquet, all referenced columns are transferred
		 *   directly from storage to the KDS on raw memory using GDS.
		 */
		if (malloc_gpu_callback)
		{
			int		num_cached_columns = 0;
			int		num_uncached_columns = 0;

			for (int j=0; j < kds_head->ncols; j++)
			{
				const kern_colmeta *cmeta = &kds_head->colmeta[j];
				int			row_group_id = kds_head->parquet_row_group;
				int			field_index = cmeta->field_index;
				void	   *pq_cache;

				if (field_index < 0 ||
					cmeta->idx_subattrs > 0 ||
					cmeta->num_subattrs > 0)
					continue;
				pq_cache = parquetCacheLookup(&stat_buf,
											  row_group_id,
											  field_index);
				if (pq_cache)
					num_cached_columns++;
				else
					num_uncached_columns++;
				if (field_index >= parquet_cached_chunks.size())
					parquet_cached_chunks.resize(field_index + 1);
				parquet_cached_chunks[field_index] = pq_cache;
			}
			/*
			 * Fast path if all the referenced columns are cached.
			 */
			if (num_cached_columns > 0 && num_uncached_columns == 0)
			{
				kds = __parquetReadOneRowGroupFullyCached(kds_head,
														  parquet_cached_chunks,
														  malloc_gpu_callback,
														  malloc_private,
														  p_npages_direct_read,
														  p_npages_vfs_read);
			}
		}
		/*
		 * Elsewhere, use arrow/parquet file reader
		 */
		if (!kds)
			kds = __parquetReadOneRowGroupNormal(parquet_filp,
												 filename,
												 &stat_buf,
												 kds_head,
												 parquet_cached_chunks,
												 malloc_gpu_callback,
												 malloc_host_callback,
												 malloc_private,
												 p_npages_direct_read,
												 p_npages_vfs_read);
	}
	catch (const std::exception &e) {
		if (error_message)
			snprintf(error_message, error_message_sz, "%s", e.what());
		kds = NULL;
	}
	/* release parquet cache if any */
	for (int j=0; j < parquet_cached_chunks.size(); j++)
	{
		void   *pq_cache = parquet_cached_chunks[j];
		if (pq_cache)
			parquetCacheRelease(pq_cache);
	}
	return kds;
}
