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
 * parquetFileEntry
 */
struct parquetFileEntry
{
	uint32_t	hash;
	int			refcnt;
	dlist_node	hash_chain;
	dlist_node	lru_chain;
	struct timeval lru_tv;
	struct stat	stat_buf;
	std::string	filename;
	std::shared_ptr<arrow::io::ReadableFile>	arrow_filp;
	std::unique_ptr<parquet::arrow::FileReader> file_reader;
};
using parquetFileEntry	= struct parquetFileEntry;

#define PQ_HASH_NSLOTS	797
static uint32_t			pq_hash_initialized = 0;
static std::mutex		pq_hash_lock[PQ_HASH_NSLOTS];
static dlist_head		pq_hash_slot[PQ_HASH_NSLOTS];
static std::mutex		pq_hash_lru_lock;
static dlist_head		pq_hash_lru_list;
static std::thread	   *pq_hash_worker = NULL;

/* copy from ilist.h */
#define dlist_foreach(iter, lhead)										\
	for ((iter).end = &(lhead)->head,									\
		 (iter).cur = (iter).end->next ? (iter).end->next : (iter).end;	\
		 (iter).cur != (iter).end;										\
		 (iter).cur = (iter).cur->next)
#define dlist_container(type, membername, ptr)							\
	((type *) ((char *) (ptr) - offsetof(type, membername)))

static inline void
dlist_init(dlist_head *head)
{
    head->head.next = head->head.prev = &head->head;
}
static inline void
dlist_delete(dlist_node *node)
{
	node->prev->next = node->next;
	node->next->prev = node->prev;
}
static inline void
dlist_push_tail(dlist_head *head, dlist_node *node)
{
	node->next = &head->head;
	node->prev = head->head.prev;
	node->prev->next = node;
	head->head.prev = node;
}
static inline void
dlist_move_tail(dlist_head *head, dlist_node *node)
{
	if (node->prev && node->next)
		dlist_delete(node);
	dlist_push_tail(head, node);
}

/*
 * Error Reporting
 */
#ifdef PGSTROM_DEBUG_BUILD
#define __Elog(fmt,...)							\
	fprintf(stderr, "(%s:%d)" fmt "\n",			\
			__FILE__,__LINE__, ##__VA_ARGS__)
#else
#define __Elog(fmt,...)
#endif

/*
 * ParquetFileHashTableWorker
 */
static void
ParquetFileHashTableWorker(void)
{
	for (;;)
	{
		struct timeval	curr;
		dlist_iter		iter;

		gettimeofday(&curr, NULL);
		pq_hash_lru_lock.lock();
		dlist_foreach (iter, &pq_hash_lru_list)
		{
			auto	entry = dlist_container(parquetFileEntry, lru_chain, iter.cur);

			if (entry->lru_tv.tv_sec > curr.tv_sec ||
				(entry->lru_tv.tv_sec  == curr.tv_sec &&
				 entry->lru_tv.tv_usec > curr.tv_usec) ||
				((curr.tv_sec  - entry->lru_tv.tv_sec)  * 1000000UL +
				 (curr.tv_usec - entry->lru_tv.tv_usec) < 8000000UL))
			{
				/* no more entry elapsed from the last access */
				break;
			}
			else
			{
				/* 8sec or more elapsed from the last access!! */
				uint32_t	hindex = entry->hash % PQ_HASH_NSLOTS;

				pq_hash_lock[hindex].lock();
				if (entry->refcnt == 0)
				{
					// OK, nobody references the entry, so we can drop it.
					// Note that 'hash_chain' may not be valid, if parquet
					// file was modified on the disk and detached from the
					// hash-table.
					if (entry->hash_chain.prev && entry->hash_chain.next)
						dlist_delete(&entry->hash_chain);
					dlist_delete(&entry->lru_chain);
					entry->arrow_filp->Close();
					__Elog("parquet::arrow::FileReader('%s') was closed",
						   entry->filename.c_str());
					delete(entry);
				}
				pq_hash_lock[hindex].unlock();
			}
		}
		pq_hash_lru_lock.unlock();

		sleep(1);
	}
}

/*
 * __tryInitializeParquetFileHashTable
 */
static void
__tryInitializeParquetFileHashTable(void)
{
	uint32_t	curr_val;
retry:
	curr_val = __atomic_cas_uint32(&pq_hash_initialized, 0, UINT_MAX);
	if (curr_val == 0)
	{
		for (int i=0; i < PQ_HASH_NSLOTS; i++)
			dlist_init(&pq_hash_slot[i]);
		dlist_init(&pq_hash_lru_list);
		pq_hash_worker = new std::thread(ParquetFileHashTableWorker);
		pq_hash_worker->detach();
		__atomic_exchange_uint32(&pq_hash_initialized, 1);	/* done */
	}
	else if (curr_val == UINT_MAX)
	{
		/* someone works in progress */
		usleep(100);		/* 100us */
		goto retry;
	}
	else
	{
		/* already initialized */
		assert(curr_val == 1);
	}
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
checkParquetFileSchema(parquetFileEntry *entry,
					   const kern_data_store *kds_head,
					   std::vector<int> &referenced)
{
	std::shared_ptr<arrow::Schema> arrow_schema;
	// Get Schema definition
	{
		auto status = entry->file_reader->GetSchema(&arrow_schema);
		if (!status.ok())
			return false;
	}
	// Type compatibility checks (only referenced attributes)
	for (auto cell = referenced.begin(); cell != referenced.end(); cell++)
	{
		int		j = (*cell);

		if (j >= 0 && j < arrow_schema->num_fields() && j < kds_head->ncols)
		{
			if (!__checkParquetFileColumn(arrow_schema->field(j), kds_head, j))
				return false;	/* not compatible */
		}
		else
		{
			__Elog("not compatible Schema: column index %d out of range (%d or %d)",
				   j, kds_head->ncols, arrow_schema->num_fields());
			return false;	/* out of range */
		}
	}
	return true;
}

/*
 * __parquetFileHash
 */
static inline uint32_t
__parquetFileHash(dev_t st_dev, ino_t st_ino)
{
	uint64_t	hkey = ((uint64_t)st_dev << 32 | (uint64_t)st_ino);

	hkey += 0x9e3779b97f4a7c15ULL;
	hkey = (hkey ^ (hkey >> 30)) * 0xbf58476d1ce4e5b9ULL;
	hkey = (hkey ^ (hkey >> 27)) * 0x94d049bb133111ebULL;
	return (hkey ^ (hkey >> 31)) & 0xffffffffU;
}

/*
 * lookupParquetFileEntry
 */
static parquetFileEntry *
lookupParquetFileEntry(const char *filename, struct stat *stat_buf, uint32_t hash)
{
	uint32_t	hindex = hash % PQ_HASH_NSLOTS;
	dlist_iter	iter;

	dlist_foreach (iter, &pq_hash_slot[hindex])
	{
		auto	entry = dlist_container(parquetFileEntry,
										hash_chain, iter.cur);
		if (entry->stat_buf.st_dev == stat_buf->st_dev &&
			entry->stat_buf.st_ino == stat_buf->st_ino)
		{
			// confirm the stat_buf.st_mtim is identical to the hashed one.
			// if changed, it means the parquet file is modified on the storage.
			if (entry->stat_buf.st_mtim.tv_sec  == stat_buf->st_mtim.tv_sec &&
				entry->stat_buf.st_mtim.tv_nsec == stat_buf->st_mtim.tv_nsec)
			{
				entry->refcnt++;
				return entry;
			}
			// oops, the parquet file was modified on the disk, so must be re-read.
			dlist_delete(&entry->hash_chain);
			memset(&entry->hash_chain, 0, sizeof(dlist_node));
			assert(entry->lru_chain.prev && entry->lru_chain.next);
			break;
		}
	}
	return NULL;
}

/*
 * buildParquetFileEntry
 */
static parquetFileEntry *
buildParquetFileEntry(const char *filename, struct stat *stat_buf, uint32_t hash)
{
	auto entry = new parquetFileEntry();
	entry->hash = hash;
    entry->refcnt = 1;
	entry->filename = std::string(filename);
	memcpy(&entry->stat_buf, stat_buf, sizeof(struct stat));

	// open the parquet file
	{
		auto rv = arrow::io::ReadableFile::Open(entry->filename);
		if (!rv.ok())
		{
			__Elog("failed on arrow::io::ReadableFile::Open('%s')", filename);
			goto error;
		}
		entry->arrow_filp = rv.ValueOrDie();
	}
#if PARQUET_VERSION_MAJOR >= 19
	{
		auto rv = parquet::arrow::OpenFile(entry->arrow_filp);
		if (!rv.ok())
		{
			__Elog("failed on parquet::arrow::OpenFile('%s')", filename);
			goto error;
		}
		entry->file_reader = std::move(rv.ValueOrDie());
	}
#else
	{
		/* OpenFile() API was changed at Arrow v19 (issue #44784) */
		auto rv = parquet::arrow::OpenFile(entry->arrow_filp,
										   ::arrow::default_memory_pool(),
										   &entry->file_reader);
		if (!rv.ok())
		{
			__Elog("failed on parquet::arrow::OpenFile('%s')", filename);
			goto error;
		}
	}
#endif
	return entry;
error:
	if (entry)
		delete(entry);
	return NULL;
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
					   int row_group_index,
					   const kern_data_store *kds_head,
					   int num_columns,
					   int *columns_index,
					   void *(*malloc_callback)(void *malloc_private,
												size_t malloc_size),
					   void *malloc_private)
{
	uint32_t			hash, hindex;
	struct stat			stat_buf;
	parquetFileEntry   *entry;
	kern_data_store	   *kds = NULL;
	std::vector<int>	referenced(columns_index, columns_index + num_columns);

	/* initialize the parquet file hash table, only once */
	__tryInitializeParquetFileHashTable();

	/* lookup the hash-table */
	if (stat(filename, &stat_buf) != 0)
	{
		__Elog("failed on stat('%s'): %m", filename);
		return NULL;
	}
	hash = __parquetFileHash(stat_buf.st_dev,
							 stat_buf.st_ino);
	hindex = hash % PQ_HASH_NSLOTS;

	pq_hash_lock[hindex].lock();
	entry = lookupParquetFileEntry(filename, &stat_buf, hash);
	if (!entry)
	{
		entry = buildParquetFileEntry(filename, &stat_buf, hash);
		if (entry)
			dlist_push_tail(&pq_hash_slot[hindex], &entry->hash_chain);
	}
	pq_hash_lock[hindex].unlock();
	if (!entry)
		return NULL;	/* not found and failed on build */
	// update LRU state (under the pq_hash_lru_lock)
	pq_hash_lru_lock.lock();
	gettimeofday(&entry->lru_tv, NULL);
	dlist_move_tail(&pq_hash_lru_list, &entry->lru_chain);	
	pq_hash_lru_lock.unlock();

	// quick check of schema compatibility
	if (checkParquetFileSchema(entry,
							   kds_head,
							   referenced) &&
		row_group_index < entry->file_reader->num_row_groups())
	{
		std::shared_ptr<arrow::Table> table;
		auto	status = entry->file_reader->ReadRowGroup(row_group_index,
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
			__Elog("failed on parquet::arrow::FileReader::ReadRowGroup");
		}
	}
	// put the hash table entry
	pq_hash_lock[hindex].lock();
	assert(entry->refcnt > 0);
	entry->refcnt--;
	pq_hash_lock[hindex].unlock();
	return kds;
}
