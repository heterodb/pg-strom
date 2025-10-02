/*
 * parquet_read.cc
 *
 * Routines to read Parquet files
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
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
#include <parquet/api/reader.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include "xpu_common.h"
#include "arrow_defs.h"

/*
 * Error Reporting
 */
#ifdef PGSTROM_DEBUG_BUILD
#define __Elog(fmt,...)							\
	fprintf(stderr, "(%s:%d) " fmt "\n",		\
			__FILE__,__LINE__, ##__VA_ARGS__)
#else
#define __Elog(fmt,...)
#endif

/*
 * parquetMetaDataCache
 */
struct parquetMetaDataCache
{
	uint32_t		hash;
	int				refcnt;
	__dlist_node<parquetMetaDataCache> chain;
	struct stat		stat_buf;
	std::string		filename;
	std::shared_ptr<parquet::FileMetaData> metadata;
	/* constructor */
	parquetMetaDataCache(const char *__filename,
						 const struct stat *__stat_buf,
						 uint32_t __hash)
	{
		hash = __hash;
		refcnt = 1;
		chain.owner = this;
		memcpy(&stat_buf, __stat_buf, sizeof(struct stat));
		filename = std::string(__filename);
		metadata = nullptr;		/* to be set caller */
	}
};
using parquetMetaDataCache	= struct parquetMetaDataCache;

#define PQ_HASH_NSLOTS	797
static std::mutex		pq_hash_lock[PQ_HASH_NSLOTS];
static __dlist_node<parquetMetaDataCache> pq_hash_slot[PQ_HASH_NSLOTS];

/*
 * __parquetFileMetaDataHash
 */
static inline uint32_t
__parquetLocalFileHash(dev_t st_dev, ino_t st_ino)
{
	uint64_t	hkey = ((uint64_t)st_dev << 32 | (uint64_t)st_ino);

	hkey += 0x9e3779b97f4a7c15ULL;
	hkey = (hkey ^ (hkey >> 30)) * 0xbf58476d1ce4e5b9ULL;
	hkey = (hkey ^ (hkey >> 27)) * 0x94d049bb133111ebULL;
	return (int64_t)((hkey ^ (hkey >> 31)) & 0xffffffffU);
}

/*
 * parquetPutMetaDataCache
 */
static void
parquetPutMetaDataCache(parquetMetaDataCache *entry)
{
	uint32_t	hindex = entry->hash % PQ_HASH_NSLOTS;

	pq_hash_lock[hindex].lock();
	assert(entry->refcnt > 0);
	if (--entry->refcnt == 0)
	{
		__dlist_delete(&entry->chain);
		delete(entry);
	}
	pq_hash_lock[hindex].unlock();
}

/*
 * __checkParquetFileColumn
 */
static bool
__checkParquetFileColumn(const std::shared_ptr<arrow::Field> &field,
						 const kern_data_store *kds_head, int kds_col_index)
{
	auto	cmeta = &kds_head->colmeta[kds_col_index];
	auto	dtype = field->type();

	switch (dtype->id())
	{
		case arrow::Type::type::NA:
			if (cmeta->attopts.tag == ArrowType__Null)
				return true;
			__Elog("not compatible Null column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::BOOL:
			if (cmeta->attopts.tag == ArrowType__Bool)
				return true;
			__Elog("not compatible Bool column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::INT8:
		case arrow::Type::type::INT16:
		case arrow::Type::type::INT32:
		case arrow::Type::type::INT64:
			if (cmeta->attopts.tag == ArrowType__Int &&
				cmeta->attopts.integer.bitWidth == dtype->bit_width() &&
				cmeta->attopts.integer.is_signed)
				return true;
			__Elog("not compatible Int column[%d] TYPE=%s bitWidth=%d is_signed=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.integer.bitWidth,
				   cmeta->attopts.integer.is_signed ? "true" : "false");
			break;
		case arrow::Type::type::UINT8:
		case arrow::Type::type::UINT16:
		case arrow::Type::type::UINT32:
		case arrow::Type::type::UINT64:
			if (cmeta->attopts.tag == ArrowType__Int &&
				cmeta->attopts.integer.bitWidth == dtype->bit_width() &&
				!cmeta->attopts.integer.is_signed)
				return true;
			__Elog("not compatible Uint column[%d] TYPE=%s bitWidth=%d is_signed=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.integer.bitWidth,
				   cmeta->attopts.integer.is_signed ? "true" : "false");
			break;
		case arrow::Type::type::HALF_FLOAT:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Half)
				return true;
			__Elog("not compatible HalfFloat column[%d] TYPE=%s precision=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   ArrowPrecisionAsCString(cmeta->attopts.floating_point.precision));
			break;
		case arrow::Type::type::FLOAT:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Single)
				return true;
			__Elog("not compatible Float column[%d] TYPE=%s precision=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   ArrowPrecisionAsCString(cmeta->attopts.floating_point.precision));
			break;
		case arrow::Type::type::DOUBLE:
			if (cmeta->attopts.tag == ArrowType__FloatingPoint &&
				cmeta->attopts.floating_point.precision == ArrowPrecision__Double)
				return true;
			__Elog("not compatible Double column[%d] TYPE=%s precision=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   ArrowPrecisionAsCString(cmeta->attopts.floating_point.precision));
			break;
		case arrow::Type::type::DECIMAL128:
			if (cmeta->attopts.tag == ArrowType__Decimal &&
				cmeta->attopts.decimal.bitWidth == 128)
			{
				const auto __dtype = std::static_pointer_cast<arrow::Decimal128Type>(dtype);
				if (cmeta->attopts.decimal.precision == __dtype->precision() &&
					cmeta->attopts.decimal.scale     == __dtype->scale())
					return true;
				__Elog("not compatible Decimal column[%d] TYPE=%s precision=%d scale=%d",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   cmeta->attopts.decimal.precision,
					   cmeta->attopts.decimal.scale);
			}
			else
			{
				__Elog("not compatible Decimal column[%d] TYPE=%s bitWidth=%d",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   cmeta->attopts.decimal.bitWidth);
			}
			break;
		case arrow::Type::type::STRING:
			if (cmeta->attopts.tag == ArrowType__Utf8)
				return true;
			__Elog("not compatible Utf8 column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::LARGE_STRING:
			if (cmeta->attopts.tag == ArrowType__LargeUtf8)
				return true;
			__Elog("not compatible LargeUtf8 column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::BINARY:
			if (cmeta->attopts.tag == ArrowType__Binary)
				return true;
			__Elog("not compatible Binary column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::LARGE_BINARY:
			if (cmeta->attopts.tag == ArrowType__LargeBinary)
				return true;
			__Elog("not compatible LargeBinary column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
		case arrow::Type::type::FIXED_SIZE_BINARY:
			if (cmeta->attopts.tag == ArrowType__FixedSizeBinary &&
				cmeta->attopts.fixed_size_binary.byteWidth == dtype->byte_width())
				return true;
			__Elog("not compatible FixedSizeBinary column[%d] TYPE=%s byteWidth=%d",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.fixed_size_binary.byteWidth);
			break;
		case arrow::Type::type::DATE32:
		case arrow::Type::type::DATE64:
			if (cmeta->attopts.tag == ArrowType__Date)
			{
				auto __dtype = std::static_pointer_cast<arrow::DateType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::DateUnit::DAY:
						return (cmeta->attopts.date.unit == ArrowDateUnit__Day);
					case arrow::DateUnit::MILLI:
						return (cmeta->attopts.date.unit == ArrowDateUnit__MilliSecond);
					default:
						break;
				}
				__Elog("not compatible Time column[%d] TYPE=%s unit=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   ArrowDateUnitAsCString(cmeta->attopts.date.unit));
			}
			else
			{
				__Elog("not compatible Date column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::TIMESTAMP:
			if (cmeta->attopts.tag == ArrowType__Timestamp)
			{
				auto __dtype = std::static_pointer_cast<arrow::TimestampType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::TimeUnit::SECOND:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__Second);
					case arrow::TimeUnit::MILLI:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__MilliSecond);
					case arrow::TimeUnit::MICRO:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__MicroSecond);
					case arrow::TimeUnit::NANO:
						return (cmeta->attopts.timestamp.unit == ArrowTimeUnit__NanoSecond);
					default:
						break;
				}
				__Elog("not compatible Timestamp column[%d] TYPE=%s unit=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   ArrowTimeUnitAsCString(cmeta->attopts.timestamp.unit));
			}
			else
			{
				__Elog("not compatible Timestamp column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::TIME32:
		case arrow::Type::type::TIME64:
			if (cmeta->attopts.tag == ArrowType__Time)
			{
				auto __dtype = std::static_pointer_cast<arrow::TimeType>(dtype);
				switch (__dtype->unit())
				{
					case arrow::TimeUnit::SECOND:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__Second);
					case arrow::TimeUnit::MILLI:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__MilliSecond);
					case arrow::TimeUnit::MICRO:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__MicroSecond);
					case arrow::TimeUnit::NANO:
						return (cmeta->attopts.time.unit == ArrowTimeUnit__NanoSecond);
					default:
						break;
				}
				__Elog("not compatible Time column[%d] TYPE=%s unit=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag),
					   ArrowTimeUnitAsCString(cmeta->attopts.time.unit));
			}
			else
			{
				__Elog("not compatible Time column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::INTERVAL_MONTHS:
			if (cmeta->attopts.tag == ArrowType__Interval &&
				cmeta->attopts.unitsz == sizeof(int32_t))
				return true;
			__Elog("not compatible Interval column[%d] TYPE=%s unitsz=%d",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.unitsz);
			break;
		case arrow::Type::type::INTERVAL_DAY_TIME:
			if (cmeta->attopts.tag == ArrowType__Interval &&
				cmeta->attopts.unitsz == sizeof(int64_t))
				return true;
			__Elog("not compatible Interval column[%d] TYPE=%s unitsz=%d",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag),
				   cmeta->attopts.unitsz);
			break;
		case arrow::Type::type::LIST:
		case arrow::Type::type::LARGE_LIST:
			if (cmeta->attopts.tag == ArrowType__List)
			{
				auto __dtype = std::static_pointer_cast<arrow::BaseListType>(dtype);
				if (cmeta->idx_subattrs >= kds_head->ncols &&
					cmeta->idx_subattrs <  kds_head->nr_colmeta &&
					__checkParquetFileColumn(__dtype->value_field(),
											 kds_head, cmeta->idx_subattrs))
					return true;
				__Elog("List subfield is out of range");
			}
			else
			{
				__Elog("not compatible List column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		case arrow::Type::type::STRUCT:
			if (cmeta->attopts.tag == ArrowType__Struct)
			{
				auto __dtype = std::static_pointer_cast<arrow::StructType>(dtype);
				if (__dtype->num_fields() == cmeta->num_subattrs &&
					cmeta->idx_subattrs >= kds_head->ncols &&
					cmeta->idx_subattrs +
					cmeta->num_subattrs <= kds_head->nr_colmeta)
				{
					for (int k=0; k < __dtype->num_fields(); k++)
					{
						if (!__checkParquetFileColumn(__dtype->field(k),
													  kds_head, cmeta->idx_subattrs + k))
							return false;
					}
					return true;
				}
				__Elog("Struct subfield is out of range");
			}
			else
			{
				__Elog("not compatible Struct column[%d] TYPE=%s",
					   kds_col_index,
					   ArrowTypeTagAsCString(cmeta->attopts.tag));
			}
			break;
		default:
			__Elog("not compatible Unknown column[%d] TYPE=%s",
				   kds_col_index,
				   ArrowTypeTagAsCString(cmeta->attopts.tag));
			break;
	}
	return false;
}

/*
 * checkParquetFileSchema
 */
static bool
checkParquetFileSchema(std::shared_ptr<arrow::Schema> arrow_schema,
					   const kern_data_store *kds_head)
{
	// Type compatibility checks (only referenced attributes)
	for (int j=0; j < kds_head->ncols; j++)
	{
		int		field_index = kds_head->colmeta[j].field_index;

		if (field_index < 0)
			continue;
		if (field_index < arrow_schema->num_fields())
		{
			auto	field = arrow_schema->field(field_index);

			if (!__checkParquetFileColumn(field, kds_head, j))
			{
				__Elog("field_index = %d not compatible", field_index);
				return false;	/* not compatible */
			}
		}
		else
		{
			__Elog("not compatible Schema: field index %d out of range [%d]",
				   field_index, arrow_schema->num_fields());
			return false;	/* out of range*/
		}
	}
	return true;
}

/*
 * parquetReadArrowTable
 */
static kern_data_store *
parquetReadArrowTable(std::shared_ptr<arrow::Table> table,
					  std::vector<int> referenced,
					  const kern_data_store *kds_head,
					  void *(*malloc_callback)(void *malloc_private,
											   size_t malloc_size),
					  void *malloc_private)
{
	size_t		kds_length = KDS_HEAD_LENGTH(kds_head) + kds_head->arrow_virtual_usage;
	size_t		curr_pos = kds_length;
	kern_data_store *kds;

	/*
	 * estimate the buffer length
	 */
	assert(kds_head->format == KDS_FORMAT_PARQUET);
	for (int k=0; k < table->num_columns(); k++)
	{
		auto	column = table->column(k);
		for (const auto &chunk : column->chunks())
		{
			auto	data = chunk->data();
			int		count = 0;
			for (const auto &buf : data->buffers)
			{
				if (buf)
					kds_length += ARROW_ALIGN(buf->size());
				if (++count > 3)
				{
					__Elog("unknown buffer layout");
					return NULL;	/* unknown buffer layout */
				}
			}
		}
	}
	/*
	 * buffer allocation
	 */
	kds = (kern_data_store *)malloc_callback(malloc_private, kds_length);
	if (!kds)
	{
		__Elog("out of memory");
		return NULL;
	}
	/*
	 * fillup the buffer
	 */
	memcpy(kds, kds_head, curr_pos);
	for (int k=0; k < table->num_columns(); k++)
	{
		auto	column = table->column(k);
		auto	cmeta = &kds->colmeta[referenced[k]];

		for (const auto &chunk : column->chunks())
		{
			auto	data = chunk->data();
			int		phase = 0;
			for (const auto &buf : data->buffers)
			{
				uint64_t	__offset = 0;
				uint64_t	__length = 0;

				if (buf)
				{
					__offset = curr_pos;
					__length = buf->size();
					memcpy((char *)kds + __offset, buf->data(), buf->size());
					curr_pos += ARROW_ALIGN(__length);
				}
				switch (phase)
				{
					case 0:
						cmeta->nullmap_offset = __offset;
						cmeta->nullmap_length = __length;
						break;
					case 1:
						cmeta->values_offset = __offset;
						cmeta->values_length = __length;
						break;
					default:
						assert(phase == 2);
						cmeta->extra_offset = __offset;
						cmeta->extra_length = __length;
						break;
				}
				phase++;
			}
		}
	}
	kds->format = KDS_FORMAT_ARROW;

	return kds;
}

/*
 * parquetReadOneRowGroup
 *
 * It returns a KDS buffer with KDS_FORMAT_ARROW that loads the
 * specified row-group.
 */
kern_data_store *
parquetReadOneRowGroup(const char *filename,
					   const kern_data_store *kds_head,
					   void *(*malloc_callback)(void *malloc_private,
												size_t malloc_size),
					   void *malloc_private)
{
	uint32_t	row_group_index = kds_head->parquet_row_group;
	struct stat	stat_buf;
	uint32_t	hash, hindex;
	parquetMetaDataCache *entry;
	std::shared_ptr<parquet::FileMetaData> metadata = nullptr;
    std::unique_ptr<parquet::ParquetFileReader> raw_file_reader = nullptr;
    std::unique_ptr<parquet::arrow::FileReader> parquet_file_reader = nullptr;
	std::shared_ptr<arrow::Schema> arrow_schema;
	kern_data_store *kds = NULL;
	arrow::Status status;

	/*
	 * Open a new Parquet File Reader dedicated for this thread, but
	 * utilize parquet::FileMetada once parsed by other one.
	 * As discussed in #937, libarrow/libparquet is not designed for
	 * thread-safe, and caller must take care mutual controls.
	 */

	// Lookup the local file metadata cache first
	if (stat(filename, &stat_buf) != 0)
	{
		__Elog("failed on stat('%s'): %m", filename);
		return NULL;
	}
	hash = __parquetLocalFileHash(stat_buf.st_dev,
								  stat_buf.st_ino);
	hindex = hash % PQ_HASH_NSLOTS;

	pq_hash_lock[hindex].lock();
	__dlist_foreach(entry, &pq_hash_slot[hindex])
	{
		if (entry->stat_buf.st_dev == stat_buf.st_dev &&
			entry->stat_buf.st_ino == stat_buf.st_ino)
		{
			// confirm the stat_buf.st_mtim is identical to the hashed one.
			// if changed, it means the parquet file is modified on the storage.
			if (entry->stat_buf.st_mtim.tv_sec  == stat_buf.st_mtim.tv_sec &&
				entry->stat_buf.st_mtim.tv_nsec == stat_buf.st_mtim.tv_nsec)
			{
				entry->refcnt++;
				metadata = entry->metadata;
				pq_hash_lock[hindex].unlock();
				break;
			}
			// Oops, the parquet file was modified on the disk, so should be
			// invalid by the one who hold this entry.
			assert(entry->refcnt > 0);
		}
	}
	/*
	 * Open the Parquet File (with cached metadata)
	 */
	raw_file_reader = parquet::ParquetFileReader::OpenFile(std::string(filename),
														   false,	/* memory_map */
														   parquet::default_reader_properties(),
														   metadata);
	if (!raw_file_reader)
	{
		if (entry)
			parquetPutMetaDataCache(entry);
		else
			pq_hash_lock[hindex].unlock();
		__Elog("failed on parquet::ParquetFileReader::Open('%s'): %s", filename);
		return NULL;
	}

	/*
	 * Add metadata to the metadata cache (if not cached yet)
	 */
	if (!entry)
	{
		assert(!metadata);
		entry = new(std::nothrow) parquetMetaDataCache(filename, &stat_buf, hash);
		if (!entry)
		{
			pq_hash_lock[hindex].unlock();
			__Elog("out of memory");
			return NULL;
		}
		entry->metadata = raw_file_reader->metadata();
		pq_hash_lock[hindex].unlock();
	}
	/*
	 * Open the Arrow File Reader
	 */
	status = parquet::arrow::FileReader::Make(arrow::default_memory_pool(),
											  std::move(raw_file_reader),
											  &parquet_file_reader);
	if (!status.ok())
	{
		parquetPutMetaDataCache(entry);
		__Elog("failed on parquet::arrow::FileReader::Make('%s'): %s",
			   filename, status.ToString().c_str());
		return NULL;
	}
	// quick check of schema compatibility
	status = parquet_file_reader->GetSchema(&arrow_schema);
	if (!status.ok())
	{
		parquetPutMetaDataCache(entry);
		__Elog("failed on parquet::arrow::FileReader::GetSchema(): %s",
			   status.ToString().c_str());
		return NULL;
	}
	if (checkParquetFileSchema(arrow_schema, kds_head) &&
		row_group_index < parquet_file_reader->num_row_groups())
	{
		std::shared_ptr<arrow::Table> table;
		std::vector<int>	referenced;

		for (int j=0; j < kds_head->ncols; j++)
		{
			auto	cmeta = &kds_head->colmeta[j];

			if (cmeta->field_index >= 0)
				referenced.push_back(cmeta->field_index);
		}
		status = parquet_file_reader->ReadRowGroup(row_group_index,
												   referenced,
												   &table);
		if (status.ok())
		{
			kds = parquetReadArrowTable(table, referenced,
										kds_head,
										malloc_callback,
										malloc_private);
		}
		else
		{
			__Elog("failed on parquet::arrow::FileReader::ReadRowGroup: %s",
				   status.ToString.c_str());
		}
	}
	parquetPutMetaDataCache(entry);
	return kds;
}
