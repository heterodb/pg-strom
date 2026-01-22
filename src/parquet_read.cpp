/*
 * parquet_read.cc
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
#include <condition_variable>
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
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include "xpu_common.h"
#include "arrow_defs.h"

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
 * PostgreSQL / PG-Strom functions
 */
extern "C" void	pgstrom_inc_perf_counter(int num);
extern "C" void	pgstrom_add_perf_counter(int num, const struct timeval *tv_base);
extern "C" uint32_t hash_bytes(const unsigned char *k, int keylen);

/*
 * Parquet Metadata Cache
 */
using parquetFileMetadataCache	= struct ParquetFileMetadataCache;
using parquetFileMetadataChain	= struct __dlist_node<parquetFileMetadataCache>;
struct ParquetFileMetadataCache
{
	uint32_t	hash;
	int32_t		refcnt;
	parquetFileMetadataChain hash_chain;
	parquetFileMetadataChain lru_chain;
	std::string	filename;
	struct stat	stat_buf;
	std::shared_ptr<parquet::FileMetaData> metadata;
	parquet::arrow::SchemaManifest manifest;
	ParquetFileMetadataCache()
	{
		hash_chain.owner = this;
		lru_chain.owner = this;
	}
};
static std::mutex					parquet_file_metadata_mutex;
static std::condition_variable		parquet_file_metadata_cond;
#define	PARQUET_FILE_METADATA_NSLOTS	6000
static parquetFileMetadataChain parquet_file_metadata_hash[PARQUET_FILE_METADATA_NSLOTS];
static parquetFileMetadataChain parquet_file_metadata_lru;

static void
__putParquetFileMetadataNoLock(parquetFileMetadataCache *pq_mcache, bool do_deletion)
{
	assert(pq_mcache->refcnt >= 2);
	pq_mcache->refcnt -= 2;
	if (do_deletion)
		pq_mcache->refcnt &= 0xfffffffe;
	if (pq_mcache->refcnt == 0)
	{
		__dlist_delete(&pq_mcache->hash_chain);
		__dlist_delete(&pq_mcache->lru_chain);
		delete(pq_mcache);
	}
}

static void
putParquetFileMetadata(parquetFileMetadataCache *pq_mcache, bool do_deletion)
{
	parquet_file_metadata_mutex.lock();
	__putParquetFileMetadataNoLock(pq_mcache, do_deletion);
	parquet_file_metadata_mutex.unlock();
}

static std::shared_ptr<parquet::FileMetaData>
__readParquetFileMetadata(arrow::io::ReadableFile &filp, const char *fname)
{
	struct {
		uint32_t	metadata_len;
		char		signature[4];
	} foot;
	size_t			length;
	std::shared_ptr<arrow::Buffer> buffer;
	/* fetch metadata length */
	{
		auto	rv = filp.GetSize();
		if (!rv.ok())
			__Elog("failed on arrow::io::ReadableFile('%s')::GetSize: %s",
				   fname, rv.status().ToString().c_str());
		length = rv.ValueOrDie();
	}
	/* read footer 8bytes */
	{
		auto	rv = filp.ReadAt(length-sizeof(foot), sizeof(foot), &foot);
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
		auto	rv = filp.ReadAt(length -
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

static parquetFileMetadataCache *
lookupParquetFileMetadata(arrow::io::ReadableFile &filp, const char *fname)
{
	int			fdesc = filp.file_descriptor();
	uint32_t	hash, hindex;
	struct stat stat_buf;
	struct {
		dev_t	st_dev;
		ino_t	st_ino;
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
	parquet_file_metadata_mutex.lock();
	__dlist_foreach(pq_mcache, &parquet_file_metadata_hash[hindex])
	{
		if (pq_mcache->hash                     == hash &&
			pq_mcache->stat_buf.st_dev          == stat_buf.st_dev &&
			pq_mcache->stat_buf.st_ino          == stat_buf.st_ino &&
			pq_mcache->stat_buf.st_mtim.tv_sec  == stat_buf.st_mtim.tv_sec &&
			pq_mcache->stat_buf.st_mtim.tv_nsec == stat_buf.st_mtim.tv_nsec)
		{
			/* found */
			pq_mcache->refcnt += 2;
			for (;;)
			{
				if ((pq_mcache->refcnt & 1) == 0)
				{
					// already dropped, or failed on metadata read
					__putParquetFileMetadataNoLock(pq_mcache, false);
					parquet_file_metadata_mutex.unlock();
					return NULL;
				}
				else if (pq_mcache->metadata)
				{
					parquet_file_metadata_mutex.unlock();
					return pq_mcache;
				}
				else
				{
					/* wait for completion of the metadata setup */
					std::unique_lock<std::mutex> lk(parquet_file_metadata_mutex);
					parquet_file_metadata_cond.wait(lk);
				}
			}
		}
	}
	/* not found, so insert a WIP entry */
	try {
		assert(pq_mcache == NULL);
		pq_mcache = new(parquetFileMetadataCache);
		pq_mcache->hash = hash;
		pq_mcache->refcnt = 3;
		pq_mcache->filename = std::string(fname);
		memcpy(&pq_mcache->stat_buf, &stat_buf, sizeof(struct stat));
		pq_mcache->metadata = nullptr;		/* set up is still in-progress */
		__dlist_push_tail(&parquet_file_metadata_hash[hindex], &pq_mcache->hash_chain);
		__dlist_push_tail(&parquet_file_metadata_lru,          &pq_mcache->lru_chain);
	}
	catch(const std::exception &e) {
		if (pq_mcache)
			delete(pq_mcache);
		parquet_file_metadata_mutex.unlock();
		throw;
	}
	parquet_file_metadata_mutex.unlock();
	/* read parquet file metadata */
	try {
		parquet::ArrowReaderProperties arrow_props;
		arrow::Status	status;
		metadata = __readParquetFileMetadata(filp, fname);
		status = parquet::arrow::SchemaManifest::Make(metadata->schema(),
													  nullptr,		/* no key-value */
													  arrow_props,	/* default attributes */
													  &pq_mcache->manifest);
		if (!status.ok())
			__Elog("failed on parquet::arrow::SchemaManifest::Make: %s",
				   status.ToString().c_str());
	}
	catch(const std::exception &e) {
		putParquetFileMetadata(pq_mcache, true);
		parquet_file_metadata_cond.notify_all();
		throw;
	}
	/* assign metadata */
	parquet_file_metadata_mutex.lock();
	pq_mcache->metadata = metadata;
	parquet_file_metadata_cond.notify_all();
	parquet_file_metadata_mutex.unlock();
	return pq_mcache;
}

/*
 * length estimation
 */
static inline size_t
__booleanBufferLength(std::shared_ptr<arrow::Field> arrow_field, int64_t nrows)
{
	size_t	len = ARROW_ALIGN(BITMAPLEN(nrows));

	return (arrow_field->nullable() ? 2 * len : len);
}

static inline size_t
__simpleBufferLength(std::shared_ptr<arrow::Field> arrow_field, size_t unitsz, int64_t nrows)
{
	size_t	len = ARROW_ALIGN(unitsz * nrows);

	if (arrow_field->nullable())
		len += ARROW_ALIGN(BITMAPLEN(nrows));
	return len;
}

static inline size_t
__varlenaBufferLength(std::shared_ptr<arrow::Field> arrow_field,
					  bool large_offset, int64_t nrows, size_t extra_sz)
{
	size_t	len = extra_sz +
		ARROW_ALIGN((large_offset ? sizeof(int64_t) : sizeof(int32_t)) * (nrows + 1));
	if (arrow_field->nullable())
		len += ARROW_ALIGN(BITMAPLEN(nrows));
	return len;
}

/*
 * ColumnChunk Reader functions
 */
static inline void
SETUP_CMETA_OFFSET(kern_data_store *kds,
				   kern_colmeta *cmeta,
				   size_t nullmap_length,
				   size_t values_length,
				   void **p_nullmap,
				   void **p_values)
{
	assert(kds->usage == ARROW_ALIGN(kds->usage));
	if (nullmap_length > 0)
	{
		cmeta->nullmap_offset = kds->usage;
		cmeta->nullmap_length = nullmap_length;
		kds->usage += ARROW_ALIGN(nullmap_length);
		if (p_nullmap)
			*p_nullmap = (char *)kds + cmeta->nullmap_offset;
	}
	else
	{
		cmeta->nullmap_offset = cmeta->nullmap_length = 0;
		if (p_nullmap)
			*p_nullmap = NULL;
	}

	if (values_length > 0)
	{
		cmeta->values_offset = kds->usage;
		cmeta->values_length = values_length;
		kds->usage += ARROW_ALIGN(values_length);
		if (p_values)
			*p_values = (char *)kds + cmeta->values_offset;
	}
	else
	{
		cmeta->values_offset = cmeta->values_length = 0;
		if (p_values)
			*p_values = NULL;
	}
}

#define __LOAD_PARQUET_CHUNK_COMMON(NAME,READER_TYPE,					\
									ARRAY_TYPE,ARRAY_CAST,				\
									VALUES_TYPE,						\
									SET_NULL,SET_VALUE)					\
	static bool															\
	__loadParquet##NAME##Chunk(kern_data_store *kds,					\
							   kern_colmeta *cmeta,						\
							   parquet::ColumnReader *cc_reader,		\
							   bool nullable)							\
	{																	\
		auto		my_reader = static_cast<parquet::READER_TYPE *>(cc_reader);	\
		int64_t		nrooms = kds->nitems;								\
		int64_t		nitems = 0;											\
		uint8_t	   *nullmap;											\
		VALUES_TYPE *values;											\
		std::vector<int16_t> defs(nrooms);								\
		std::vector<ARRAY_TYPE> buffer(nrooms);							\
																		\
		SETUP_CMETA_OFFSET(kds,cmeta,									\
						   nullable ? BITMAPLEN(nrooms) : 0,			\
						   sizeof(VALUES_TYPE) * nrooms,				\
						   (void **)&nullmap,							\
						   (void **)&values);							\
		while (nitems < nrooms)											\
		{																\
			int64_t		ncount, nvalues;								\
			int64_t		i, j;											\
																		\
			ncount = my_reader->ReadBatch(nrooms - nitems,				\
										  nullmap ? defs.data() : nullptr, \
										  nullptr,						\
										  ARRAY_CAST buffer.data(),		\
										  &nvalues);					\
			if (ncount==0)												\
				break;													\
			for (i=0, j=0; i < ncount; i++)								\
			{															\
				int64_t		k = i + nitems;								\
																		\
				if (nullmap && defs[i] == 0)							\
				{														\
					nullmap[k>>3] &= ~(1U<<(k&7));						\
					SET_NULL(values, k);								\
				}														\
				else													\
				{														\
					assert(j < nvalues);								\
					nullmap[k>>3] |= (1U<<(k&7));						\
					SET_VALUE(values, k, buffer[j++]);					\
				}														\
			}															\
			nitems += ncount;											\
		}																\
		return (nitems == nrooms);										\
	}
#define __LOAD_PARQUET_BOOLEAN_SET_NULL(values,k)			\
	values[k>>3] &= ~(1U<<(k&7))
#define __LOAD_PARQUET_BOOLEAN_SET_VALUE(values,k,datum)	\
	do {													\
		if (datum)											\
			values[k>>3] |=  (1U<<(k&7));					\
		else												\
			values[k>>3] &= ~(1U<<(k&7));					\
	} while(0)
__LOAD_PARQUET_CHUNK_COMMON(Boolean, BoolReader,
							uint8_t, (bool *), uint8_t,
							__LOAD_PARQUET_BOOLEAN_SET_NULL,
							__LOAD_PARQUET_BOOLEAN_SET_VALUE)

#define __LOAD_PARQUET_SIMPLE_SET_NULL(values,k)			values[k] = 0
#define __LOAD_PARQUET_SIMPLE_SET_VALUE(values,k,datum)		values[k] = datum;
#define LOAD_PARQUET_SIMPLE_CHUNK(NAME,READER_TYPE,ARRAY_TYPE,VALUES_TYPE) \
	__LOAD_PARQUET_CHUNK_COMMON(NAME,READER_TYPE,ARRAY_TYPE,,VALUES_TYPE, \
								__LOAD_PARQUET_SIMPLE_SET_NULL,			\
								__LOAD_PARQUET_SIMPLE_SET_VALUE)
LOAD_PARQUET_SIMPLE_CHUNK(Int8, Int32Reader,int32_t,int8_t)
LOAD_PARQUET_SIMPLE_CHUNK(Int16,Int32Reader,int32_t,int16_t)
LOAD_PARQUET_SIMPLE_CHUNK(Int32,Int32Reader,int32_t,int32_t)
LOAD_PARQUET_SIMPLE_CHUNK(Int64,Int64Reader,int64_t,int64_t)
LOAD_PARQUET_SIMPLE_CHUNK(FP32,	FloatReader, float,float)
LOAD_PARQUET_SIMPLE_CHUNK(FP64,	DoubleReader,double,double)
#define __loadParquetDate32Chunk(a,b,c,d)		__loadParquetInt32Chunk(a,b,c,d)
#define __loadParquetDate64Chunk(a,b,c,d)		__loadParquetInt64Chunk(a,b,c,d)
#define __loadParquetTime32Chunk(a,b,c,d)		__loadParquetInt32Chunk(a,b,c,d)
#define __loadParquetTime64Chunk(a,b,c,d)		__loadParquetInt64Chunk(a,b,c,d)
#define __loadParquetTimestampChunk(a,b,c,d)	__loadParquetInt64Chunk(a,b,c,d)

#define __LOAD_PARQUET_FP16_SET_VALUE(values,k,datum)	\
	values[k] = *((uint16_t *)datum.ptr)
#define __LOAD_PARQUET_DECIMAL_SET_VALUE(values,k,datum)	\
	values[k] = *((__int128_t *)datum.ptr)
__LOAD_PARQUET_CHUNK_COMMON(FP16,FixedLenByteArrayReader,		\
							parquet::FixedLenByteArray,,		\
							int16_t,							\
							__LOAD_PARQUET_SIMPLE_SET_NULL,		\
							__LOAD_PARQUET_FP16_SET_VALUE)
__LOAD_PARQUET_CHUNK_COMMON(Decimal,FixedLenByteArrayReader,		\
							parquet::FixedLenByteArray,,			\
							__int128_t,								\
							__LOAD_PARQUET_SIMPLE_SET_NULL,			\
							__LOAD_PARQUET_DECIMAL_SET_VALUE)
typedef struct {
	uint32_t	month;
	uint32_t	days;
	uint32_t	msec;
}	parquet_physical_interval;
static inline void
__LOAD_PARQUET_INTERVAL32_SET_VALUE(uint32_t *values, int64_t k,
									parquet::FixedLenByteArray &datum)
{
	values[k] = ((parquet_physical_interval *)datum.ptr)->month;
}
static inline void
__LOAD_PARQUET_INTERVAL64_SET_VALUE(uint64_t *values, int64_t k,
									parquet::FixedLenByteArray &datum)
{
	auto	iv = (const parquet_physical_interval *)datum.ptr;
	values[k] = ((uint64_t)iv->days) | ((uint64_t)iv->msec << 32);
}
__LOAD_PARQUET_CHUNK_COMMON(Interval32,FixedLenByteArrayReader,
							parquet::FixedLenByteArray,,
							uint32_t,
							__LOAD_PARQUET_SIMPLE_SET_NULL,
							__LOAD_PARQUET_INTERVAL32_SET_VALUE)
__LOAD_PARQUET_CHUNK_COMMON(Interval64,FixedLenByteArrayReader,
							parquet::FixedLenByteArray,,
							uint64_t,
							__LOAD_PARQUET_SIMPLE_SET_NULL,
							__LOAD_PARQUET_INTERVAL64_SET_VALUE)
static bool
__loadParquetFLBAChunk(kern_data_store *kds,
					   kern_colmeta *cmeta,
					   parquet::ColumnReader *cc_reader,
					   bool nullable)
{
	auto		my_reader = static_cast<parquet::FixedLenByteArrayReader *>(cc_reader);
	size_t		unitsz = cmeta->attopts.fixed_size_binary.byteWidth;
	int64_t		nrooms = kds->nitems;
	int64_t		nitems = 0;
	uint8_t	   *nullmap;
	char	   *values;
	std::vector<int16_t> defs(nrooms);
	std::vector<parquet::FixedLenByteArray> buffer(nrooms);

	SETUP_CMETA_OFFSET(kds,cmeta,
					   nullable ? BITMAPLEN(nrooms) : 0,
					   unitsz * nrooms,
					   (void **)&nullmap,
					   (void **)&values);
	while (nitems < nrooms)
	{
		int64_t		ncount, nvalues;
		int64_t		i, j;

		ncount = my_reader->ReadBatch(nrooms - nitems,
									  nullmap ? defs.data() : nullptr,
									  nullptr,
									  buffer.data(),
									  &nvalues);
		if (ncount == 0)
			break;
		for (i=0, j=0; i < ncount; i++)
		{
			int64_t		k = i + nitems;

			if (nullmap && defs[i] == 0)
			{
				nullmap[k>>3] &= ~(1U<<(k&7));
				memset(values + unitsz * k, 0, unitsz);
			}
			else
			{
				assert(j < nvalues);
				nullmap[k>>3] |= (1U<<(k&7));
				memcpy(values + unitsz * k, buffer[j++].ptr, unitsz);
			}
		}
		nitems += ncount;
	}
	return (nitems == nrooms);
}

template <typename OFFSET_TYPE>
static bool
__loadParquetVarlenaChunk(kern_data_store *kds,
						  kern_colmeta *cmeta,
						  parquet::ColumnReader *cc_reader,
						  bool nullable)
{
	auto		my_reader = static_cast<parquet::ByteArrayReader *>(cc_reader);
	int64_t		nrooms = kds->nitems;
	int64_t		nitems = 0;
	uint8_t	   *nullmap;
	OFFSET_TYPE *values;
	std::vector<int16_t> defs(nrooms);
	std::vector<parquet::ByteArray> buffer(nrooms);

	SETUP_CMETA_OFFSET(kds,cmeta,
					   nullable ? BITMAPLEN(nrooms) : 0,
					   sizeof(OFFSET_TYPE) * (nrooms+1),
					   (void **)&nullmap,
					   (void **)&values);
	cmeta->extra_offset = kds->usage;
	cmeta->extra_length = 0;
	while (nitems < nrooms)
	{
		int64_t		ncount, nvalues;
		int64_t		i, j;

		ncount = my_reader->ReadBatch(nrooms - nitems,
									  nullmap ? defs.data() : nullptr,
									  nullptr,
									  buffer.data(),
									  &nvalues);
		if (ncount == 0)
			break;
		for (i=0, j=0; i < ncount; i++)
		{
			int64_t		k = i + nitems;

			if (k == 0)
				values[0] = 0;
			if (nullmap && defs[i] == 0)
			{
				nullmap[k>>3] &= ~(1U<<(k&7));
			}
			else
			{
				assert(j < nvalues);
				auto	datum = buffer[j++];
				nullmap[k>>3] |= (1U<<(k&7));
				memcpy((char *)kds
					   + cmeta->extra_offset
					   + cmeta->extra_length,
					   datum.ptr,
					   datum.len);
				cmeta->extra_length += datum.len;
			}
			values[k+1] = cmeta->extra_length;
		}
		nitems += ncount;
	}
	kds->usage = ARROW_ALIGN(cmeta->extra_offset +
							 cmeta->extra_length);
	return (nitems == nrooms);
}
#define __loadParquetBinaryChunk(a,b,c,d)					\
	__loadParquetVarlenaChunk<uint32_t>((a),(b),(c),(d))
#define __loadParquetLargeBinaryChunk(a,b,c,d)				\
	__loadParquetVarlenaChunk<uint64_t>((a),(b),(c),(d))

/*
 * __loadParquetFileColumn
 */
static void
__loadParquetFileColumnChunk(kern_data_store *kds, int column_index,
							 std::shared_ptr<arrow::io::ReadableFile> pq_filp,
							 const char *pq_filename,
							 const parquet::SchemaDescriptor *pq_schema,
							 const parquet::arrow::SchemaField &pq_field,
							 const parquet::RowGroupMetaData &rg_meta)
{
	kern_colmeta   *cmeta = &kds->colmeta[column_index];
	bool			nullable = pq_field.level_info.HasNullableValues();
	std::shared_ptr<parquet::ColumnReader> cc_reader;

	/*
	 * Load the ColumnChunk to on-memory buffer, if leaf-node
	 */
	if (pq_field.is_leaf())
	{
		assert(pq_field.column_index >=0 &&
			   pq_field.column_index < pq_schema->num_columns());
		auto	cc_meta = rg_meta.ColumnChunk(pq_field.column_index);
		auto	cc_desc = pq_schema->Column(pq_field.column_index);

		// identify the range to read
		int64_t		f_length = cc_meta->total_compressed_size();
		int64_t		f_offset = cc_meta->data_page_offset();
		if (cc_meta->has_dictionary_page() &&
			f_offset > cc_meta->dictionary_page_offset())
			f_offset = cc_meta->dictionary_page_offset();
		if (cc_meta->has_index_page() &&
			f_offset > cc_meta->index_page_offset())
			f_offset = cc_meta->index_page_offset();

		// read the above range of the parquet file
		auto	rv = pq_filp->ReadAt(f_offset, f_length);
		if (!rv.ok())
			__Elog("failed on arrow::io::ReadableFile('%s')->ReadAt(%ld,%ld): %s",
				   pq_filename, f_offset, f_length,
				   rv.status().ToString().c_str());

		// setup parquet::ColumnReader from the buffer
		auto	buf_stream = std::make_shared<arrow::io::BufferReader>(rv.ValueOrDie());
		auto	pg_reader = parquet::PageReader::Open(buf_stream,
													  rg_meta.num_rows(),
													  cc_meta->compression());
		cc_reader = parquet::ColumnReader::Make(cc_desc, std::move(pg_reader));
	}

	switch (cmeta->attopts.tag)
	{
		case ArrowType__Bool:
			if (cc_reader &&
				__loadParquetBooleanChunk(kds, cmeta, cc_reader.get(), nullable))
				return;
			break;
		case ArrowType__Int:
			switch (cmeta->attopts.integer.bitWidth)
			{
				case 8:
					if (cc_reader &&
						__loadParquetInt8Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case 16:
					if (cc_reader &&
						__loadParquetInt16Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case 32:
					if (cc_reader &&
						__loadParquetInt32Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case 64:
					if (cc_reader &&
						__loadParquetInt64Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
			}
			break;
		case ArrowType__FloatingPoint:
			switch (cmeta->attopts.floating_point.precision)
			{
				case ArrowPrecision__Half:
					if (cc_reader &&
						__loadParquetFP16Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case ArrowPrecision__Single:
					if (cc_reader &&
						__loadParquetFP32Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case ArrowPrecision__Double:
					if (cc_reader &&
						__loadParquetFP64Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				default:
					break;
			}
			break;
		case ArrowType__Decimal:
			if (cc_reader &&
				__loadParquetDecimalChunk(kds, cmeta, cc_reader.get(), nullable))
				return;
			break;
		case ArrowType__Binary:
		case ArrowType__Utf8:
			if (cc_reader &&
				__loadParquetBinaryChunk(kds, cmeta, cc_reader.get(), nullable))
				return;
			break;
		case ArrowType__LargeBinary:
		case ArrowType__LargeUtf8:
			if (cc_reader &&
				__loadParquetLargeBinaryChunk(kds, cmeta, cc_reader.get(), nullable))
				return;
			break;
		case ArrowType__Date:
			switch (cmeta->attopts.date.unit)
			{
				case ArrowDateUnit__Day:
					if (cc_reader &&
						__loadParquetDate32Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case ArrowDateUnit__MilliSecond:
					if (cc_reader &&
						__loadParquetDate64Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				default:
					break;
			}
			break;
		case ArrowType__Time:
			switch (cmeta->attopts.time.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
					if (cc_reader &&
						__loadParquetTime32Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					if (cc_reader &&
						__loadParquetTime64Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				default:
					break;
			}
			break;
		case ArrowType__Timestamp:
			switch (cmeta->attopts.timestamp.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					if (cc_reader &&
						__loadParquetTimestampChunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				default:
					break;
			}
			break;
		case ArrowType__Interval:
			switch (cmeta->attopts.interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					if (cc_reader &&
						__loadParquetInterval32Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				case ArrowIntervalUnit__Day_Time:
					if (cc_reader &&
						__loadParquetInterval64Chunk(kds, cmeta, cc_reader.get(), nullable))
						return;
					break;
				default:
					break;
			}
			break;
		case ArrowType__FixedSizeBinary:
			if (cc_reader &&
				__loadParquetFLBAChunk(kds, cmeta, cc_reader.get(), nullable))
				return;
			break;
		case ArrowType__List:
		case ArrowType__LargeList:
		case ArrowType__Struct:
		default:
			break;
	}
	__Elog("unable to load column chunk!!!");
}

/*
 * loadParquetFileRowGroup
 */
static void
loadParquetFileRowGroup(kern_data_store *kds,
						std::shared_ptr<arrow::io::ReadableFile> pq_filp,
						const char *pq_filename,
						const parquet::arrow::SchemaManifest &manifest,
						const parquet::RowGroupMetaData &rg_meta)
{
	//load the column-chunk for each referenced column
	for (int j=0; j < kds->ncols; j++)
	{
		int		field_index = kds->colmeta[j].field_index;

		if (field_index < 0)
			continue;
		if (field_index < manifest.schema_fields.size())
		{
			__loadParquetFileColumnChunk(kds, j, pq_filp, pq_filename,
										 manifest.descr,
										 manifest.schema_fields[field_index],
										 rg_meta);
		}
		else
		{
			__Elog("not compatible Schema: field index %d out of range [%ld]",
				   field_index, manifest.schema_fields.size());
		}
	}
	assert(kds->usage <= kds->length);
}

/*
 * __checkParquetFileColumn
 */
static size_t
__checkParquetFileColumn(const kern_data_store *kds_head, int column_index,
						 const parquet::SchemaDescriptor *pq_schema,
						 const parquet::arrow::SchemaField &pq_field,
						 const parquet::RowGroupMetaData &rg_meta)
{
	int64_t		nrooms = rg_meta.num_rows();
	auto		cmeta = &kds_head->colmeta[column_index];
	auto		field = pq_field.field;
	const parquet::ColumnDescriptor *pq_cdesc = nullptr;

	if (pq_field.is_leaf())
	{
		assert(pq_field.column_index >=0 &&
			   pq_field.column_index < pq_schema->num_columns());
		pq_cdesc = pq_schema->Column(pq_field.column_index);
	}

	switch (cmeta->attopts.tag)
	{
		case ArrowType__Bool:
			if (pq_cdesc &&
				pq_cdesc->physical_type() == parquet::Type::BOOLEAN)
				return __booleanBufferLength(field, nrooms);
			break;
		case ArrowType__Int:
			switch (cmeta->attopts.integer.bitWidth)
			{
				case 8:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT32)
						return __simpleBufferLength(field, sizeof(int8_t), nrooms);
					break;
				case 16:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT32)
						return __simpleBufferLength(field, sizeof(int16_t), nrooms);
                    break;
				case 32:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT32)
						return __simpleBufferLength(field, sizeof(int32_t), nrooms);
                    break;
				case 64:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT64)
						return __simpleBufferLength(field, sizeof(int64_t), nrooms);
                    break;
				default:
					/* not supported */
					break;
			}
			break;
		case ArrowType__FloatingPoint:
			switch (cmeta->attopts.floating_point.precision)
			{
				case ArrowPrecision__Half:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::FIXED_LEN_BYTE_ARRAY &&
						pq_cdesc->type_length() == sizeof(uint16_t))
						return __simpleBufferLength(field, sizeof(uint16_t), nrooms);
					break;
				case ArrowPrecision__Single:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::FLOAT)
						return __simpleBufferLength(field, sizeof(float), nrooms);
					break;
				case ArrowPrecision__Double:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::DOUBLE)
						return __simpleBufferLength(field, sizeof(double), nrooms);
					break;
			}
			break;
		case ArrowType__Decimal:
			if (pq_cdesc &&
				pq_cdesc->physical_type() == parquet::Type::FIXED_LEN_BYTE_ARRAY &&
				pq_cdesc->type_length() == sizeof(__int128_t) &&
				cmeta->attopts.decimal.precision == pq_cdesc->type_precision() &&
				cmeta->attopts.decimal.scale == pq_cdesc->type_scale())
				return __simpleBufferLength(field, sizeof(__int128_t), nrooms);
			break;
		case ArrowType__Binary:
		case ArrowType__Utf8:
			if (pq_cdesc &&
				pq_cdesc->physical_type() == parquet::Type::BYTE_ARRAY)
			{
				auto	cc_meta = rg_meta.ColumnChunk(pq_field.column_index);
				return __varlenaBufferLength(field, false, nrooms,
											 cc_meta->total_uncompressed_size());
			}
			break;
		case ArrowType__LargeBinary:
		case ArrowType__LargeUtf8:
			if (pq_cdesc &&
				pq_cdesc->physical_type() == parquet::Type::BYTE_ARRAY)
			{
				auto	cc_meta = rg_meta.ColumnChunk(pq_field.column_index);
				return __varlenaBufferLength(field, false, nrooms,
											 cc_meta->total_uncompressed_size());
			}
			break;
		case ArrowType__Date:
			switch (cmeta->attopts.date.unit)
			{
				case ArrowDateUnit__Day:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT32)
						return __simpleBufferLength(field, sizeof(int32_t), nrooms);
					break;
				case ArrowDateUnit__MilliSecond:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT64)
						return __simpleBufferLength(field, sizeof(int64_t), nrooms);
					break;
				default:
					break;
			}
			break;
		case ArrowType__Time:
			switch (cmeta->attopts.time.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT32)
						return __simpleBufferLength(field, sizeof(int32_t), nrooms);
					break;
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT64)
						return __simpleBufferLength(field, sizeof(int64_t), nrooms);
					break;
				default:
					break;
			}
			break;
		case ArrowType__Timestamp:
			switch (cmeta->attopts.timestamp.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::INT64)
						return __simpleBufferLength(field, sizeof(int64_t), nrooms);
					break;
				default:
					break;
			}
			break;
		case ArrowType__Interval:
			switch (cmeta->attopts.interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::FIXED_LEN_BYTE_ARRAY &&
						pq_cdesc->type_length() == sizeof(int32_t) * 3)
						return __simpleBufferLength(field, sizeof(int32_t), nrooms);
					break;
				case ArrowIntervalUnit__Day_Time:
					if (pq_cdesc &&
						pq_cdesc->physical_type() == parquet::Type::FIXED_LEN_BYTE_ARRAY &&
						pq_cdesc->type_length() == sizeof(int32_t) * 3)
						return __simpleBufferLength(field, sizeof(int32_t), nrooms);
					break;
				default:
					break;
			}
			break;
		case ArrowType__List:
		case ArrowType__LargeList:
			if (cmeta->num_subattrs == 1 &&
				pq_field.children.size() == 1)
			{
				size_t	len = (field->nullable() ? ARROW_ALIGN(BITMAPLEN(nrooms)) : 0);

				assert(cmeta->idx_subattrs >= kds_head->ncols &&
					   cmeta->idx_subattrs <  kds_head->nr_colmeta);
				len += __checkParquetFileColumn(kds_head,
												cmeta->idx_subattrs,
												pq_schema,
												pq_field.children[0],
												rg_meta);
				return len;
			}
			break;
		case ArrowType__Struct:
			if (cmeta->num_subattrs == pq_field.children.size())
			{
				size_t	len = (field->nullable() ? ARROW_ALIGN(BITMAPLEN(nrooms)) : 0);

				assert(cmeta->idx_subattrs >= kds_head->ncols &&
					   cmeta->idx_subattrs +
					   cmeta->num_subattrs < kds_head->nr_colmeta);
				for (int k=0; k < pq_field.children.size(); k++)
				{
					len += __checkParquetFileColumn(kds_head,
													cmeta->idx_subattrs + k,
													pq_schema,
													pq_field.children[k],
													rg_meta);
				}
				return len;
			}
			break;
		case ArrowType__FixedSizeBinary:
			if (pq_cdesc &&
				pq_cdesc->physical_type() == parquet::Type::FIXED_LEN_BYTE_ARRAY &&
				pq_cdesc->type_length() == cmeta->attopts.fixed_size_binary.byteWidth)
				return __simpleBufferLength(field, pq_cdesc->type_length(), nrooms);
			break;
		default:
			break;
	}
	if (!pq_cdesc)
		__Elog("not compatible Parquet Column('%s') expected type=%s",
			   cmeta->attname,
			   ArrowTypeTagAsCString(cmeta->attopts.tag));
	else
		__Elog("not compatible Parquet Column('%s') expected type=%s but parquet type=%s (physical=%s) def_level=%d rep_level=%d",
			   cmeta->attname,
			   ArrowTypeTagAsCString(cmeta->attopts.tag),
			   (pq_cdesc->logical_type()->type() > parquet::LogicalType::Type::UNDEFINED &&
				pq_cdesc->logical_type()->type() < parquet::LogicalType::Type::NONE)
			   ? pq_cdesc->logical_type()->ToString().c_str() 
			   : ((pq_cdesc->converted_type() > parquet::ConvertedType::NONE &&
				   pq_cdesc->converted_type() < parquet::ConvertedType::UNDEFINED)
				  ? ConvertedTypeToString(pq_cdesc->converted_type()).c_str() : "Unknown"),
			   parquet::TypeToString(pq_cdesc->physical_type()).c_str(),
			   pq_cdesc->max_definition_level(),
			   pq_cdesc->max_repetition_level());
}

/*
 * checkParquetFileSchema
 */
static size_t
checkParquetFileSchema(const kern_data_store *kds_head, size_t kds_head_sz,
					   const parquet::arrow::SchemaManifest &manifest,
					   const parquet::RowGroupMetaData &rg_meta)
{
	size_t	kds_length = ARROW_ALIGN(kds_head_sz);

	// Type compatibility checks (only referenced attributes)
	for (int j=0; j < kds_head->ncols; j++)
	{
		int		field_index = kds_head->colmeta[j].field_index;

		if (field_index < 0)
			continue;
		if (field_index < manifest.schema_fields.size())
		{
			kds_length += __checkParquetFileColumn(kds_head, j,
												   manifest.descr,
												   manifest.schema_fields[field_index],
												   rg_meta);
		}
		else
		{
			__Elog("not compatible Schema: field index %d out of range [%ld]",
				   field_index, manifest.schema_fields.size());
		}
	}
	return kds_length;
}

/*
 * parquetReadOneRowGroup
 *
 * It returns a KDS buffer with KDS_FORMAT_ARROW that loads the
 * specified row-group.
 */
static kern_data_store *
__parquetReadOneRowGroup(const char *filename,
						 kern_data_store *kds_head,
						 void *(*malloc_callback)(void *malloc_private,
												  size_t malloc_size),
						 void  (*mfree_callback)(void *malloc_private),
						 void *malloc_private)
{
	std::shared_ptr<arrow::io::ReadableFile> pq_filp;
	parquetFileMetadataCache *pq_mcache = nullptr;
	kern_data_store *kds = NULL;

	/* open the parquet file */
	{
		auto	rv = arrow::io::ReadableFile::Open(std::string(filename));
		if (!rv.ok())
			__Elog("failed on arrow::io::ReadableFile::Open('%s'): %s",
				   filename,
				   rv.status().ToString().c_str());
		pq_filp = rv.ValueOrDie();
	}
	/* fetch file metadata */
	pq_mcache = lookupParquetFileMetadata(*pq_filp, filename);
	try {
		auto	metadata = pq_mcache->metadata;
		int		rg_index = kds_head->parquet_row_group;
		auto	rg_meta = metadata->RowGroup(rg_index);
		size_t	head_sz = (KDS_HEAD_LENGTH(kds_head) + kds_head->arrow_virtual_usage);
		size_t	length;

		if (rg_index < 0 || rg_index >= metadata->num_row_groups())
			__Elog("row-group index %d is out of range for '%s'", rg_index, filename);

		//check schema compatibility and estimate buffer length
		length = checkParquetFileSchema(kds_head, head_sz,
										pq_mcache->manifest,
										*rg_meta);
		//allocation of the buffer (host accessible)
		kds = (kern_data_store *)malloc_callback(malloc_private, length);
		if (!kds)
			__Elog("out of memory (length=%lu)", length);
		assert(head_sz <= length);
		memcpy(kds, kds_head, head_sz);
		kds->length = length;
		kds->usage  = ARROW_ALIGN(head_sz);
		kds->nitems = rg_meta->num_rows();
		kds->format = KDS_FORMAT_ARROW;
		loadParquetFileRowGroup(kds, pq_filp, filename,
								pq_mcache->manifest,
								*rg_meta);
	}
	catch (const std::exception &e) {
		if (kds)
			mfree_callback(malloc_private);
		putParquetFileMetadata(pq_mcache, false);
		throw;
	}
	return kds;
}
/*
 * parquetReadOneRowGroup - interface to C-portion
 */
kern_data_store *
parquetReadOneRowGroup(const char *filename,
					   kern_data_store *kds_head,
					   void *(*malloc_callback)(void *malloc_private,
												size_t malloc_size),
					   void  (*mfree_callback)(void *malloc_private),
					   void *malloc_private,
					   char *error_message, size_t error_message_sz)
{
	kern_data_store *kds = NULL;
	try {
		kds = __parquetReadOneRowGroup(filename,
									   kds_head,
									   malloc_callback,
									   mfree_callback,
									   malloc_private);
	}
	catch (const std::exception &e) {
		if (error_message)
			snprintf(error_message, error_message_sz, "%s", e.what());
		return NULL;
	}
	return kds;
}
