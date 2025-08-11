/*
 * arrow_meta.cc
 *
 * Routines to handle Apache Arrow/Parquet metadata
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "arrow-tools.h"
#include "flatbuffers/flatbuffers.h"			/* copied from arrow */
#include "flatbuffers/File_generated.h"			/* copied from arrow */
#include "flatbuffers/Message_generated.h"		/* copied from arrow */
#include "flatbuffers/Schema_generated.h"		/* copied from arrow */
#include "flatbuffers/SparseTensor_generated.h"	/* copied from arrow */
#include "flatbuffers/Tensor_generated.h"		/* copied from arrow */
#define ARROW_SIGNATURE		"ARROW1"

// ================================================================
//
// dump_arrow_metadata(arrow_file, filename)
//
// ================================================================
static void
__dump_arrow_custom_metadata(const flatbuffers::Vector<flatbuffers::Offset<org::apache::arrow::flatbuf::KeyValue>> *custom_metadata)
{
	if (custom_metadata)
	{
		flatbuffers::uoffset_t	i, n_keys = custom_metadata->size();

		for (i=0; i < n_keys; i++)
		{
			auto	kv = custom_metadata->Get(i);

			if (i > 0)
				printf(", ");
			printf("{KeyValue: key='%s', value='%s'}",
				   kv->key()->c_str(),
				   kv->value()->c_str());
		}
	}
}

static void
__dump_arrow_field(const org::apache::arrow::flatbuf::Field *field)
{
	auto	children = field->children();
	auto	dict = field->dictionary();

	printf("{Field: name=\"%s\", nullable=%s, type={",
		   field->name()->c_str(),
		   field->nullable() ? "true" : "false");
	switch (field->type_type())
	{
		case org::apache::arrow::flatbuf::Type::Null:
			printf("Null");
			break;
		case org::apache::arrow::flatbuf::Type::Int: {
			auto	__type = field->type_as_Int();
			printf("%sInt%d",
				   __type->is_signed() ? "" : "U",
				   __type->bitWidth());
			break;
		}
		case org::apache::arrow::flatbuf::Type::FloatingPoint: {
			auto	__type = field->type_as_FloatingPoint();
			printf("FloatingPoint: precision=%s",
				   __type->precision() == org::apache::arrow::flatbuf::Precision::HALF ? "Half" :
				   __type->precision() == org::apache::arrow::flatbuf::Precision::SINGLE ? "Single" :
				   __type->precision() == org::apache::arrow::flatbuf::Precision::DOUBLE ? "Double" : "???");
			break;
		}
		case org::apache::arrow::flatbuf::Type::Binary:
			printf("Binary");
			break;
		case org::apache::arrow::flatbuf::Type::Utf8:
			printf("Utf8");
			break;
		case org::apache::arrow::flatbuf::Type::Bool:
			printf("Bool");
			break;
		case org::apache::arrow::flatbuf::Type::Decimal: {
			auto	__type = field->type_as_Decimal();
			printf("Decimal%d: precision=%d, scale=%d",
				   __type->bitWidth(),
				   __type->precision(),
				   __type->scale());
			break;
		}
		case org::apache::arrow::flatbuf::Type::Date: {
			auto	__type = field->type_as_Date();
			switch (__type->unit())
			{
				case org::apache::arrow::flatbuf::DateUnit::DAY:
					printf("Date32: unit=day");
					break;
				case org::apache::arrow::flatbuf::DateUnit::MILLISECOND:
					printf("Date64: unit=ms");
					break;
				default:
					printf("Date??");
					break;
			}
			break;
		}
		case org::apache::arrow::flatbuf::Type::Time: {
			auto	__type = field->type_as_Time();
			switch (__type->unit())
			{
				case org::apache::arrow::flatbuf::TimeUnit::SECOND:
					printf("Time32: unit=sec");
					break;
				case org::apache::arrow::flatbuf::TimeUnit::MILLISECOND:
					printf("Time32: unit=ms");
					break;
				case org::apache::arrow::flatbuf::TimeUnit::MICROSECOND:
					printf("Time64: unit=us");
					break;
				case org::apache::arrow::flatbuf::TimeUnit::NANOSECOND:
					printf("Time64: unit=ns");
					break;
				default:
					printf("Time??: unit=???");
					break;
			}
			break;
		}
		case org::apache::arrow::flatbuf::Type::Timestamp: {
			auto	__type = field->type_as_Timestamp();
			auto	__tz = __type->timezone();
			switch (__type->unit())
			{
				case org::apache::arrow::flatbuf::TimeUnit::SECOND:
					printf("Timestamp: unit=sec");
					break;
                case org::apache::arrow::flatbuf::TimeUnit::MILLISECOND:
                    printf("Timestamp: unit=ms");
                    break;
                case org::apache::arrow::flatbuf::TimeUnit::MICROSECOND:
                    printf("Timestamp: unit=us");
                    break;
                case org::apache::arrow::flatbuf::TimeUnit::NANOSECOND:
                    printf("Timestamp: unit=ns");
                    break;
				default:
					printf("Timestamp: unit=???");
					break;
			}
			if (__tz)
				printf(", timezone=\"%s\"", __tz->c_str());
			break;
		}
		case org::apache::arrow::flatbuf::Type::Interval: {
			auto	__type = field->type_as_Interval();
			switch (__type->unit())
			{
				case org::apache::arrow::flatbuf::IntervalUnit::YEAR_MONTH:
					printf("Interval: unit=year:month");
					break;
				case org::apache::arrow::flatbuf::IntervalUnit::DAY_TIME:
					printf("Interval: unit=day:time");
					break;
				case org::apache::arrow::flatbuf::IntervalUnit::MONTH_DAY_NANO:
					printf("Interval: unit=month:day:nano");
					break;
				default:
					printf("Interval: unit=???");
					break;
			}
			break;
		}
		case org::apache::arrow::flatbuf::Type::List:
			printf("List");
			break;
		case org::apache::arrow::flatbuf::Type::Struct_:
			printf("Struct");
			break;
		case org::apache::arrow::flatbuf::Type::Union: {
			auto	__type = field->type_as_Union();
			auto	__mode = __type->mode();
			auto	__typeIds = __type->typeIds();
			printf("Union: mode=%s types=[", EnumNameUnionMode(__mode));
			for (auto cell = __typeIds->begin(); cell != __typeIds->end(); cell++)
			{
				if (cell != __typeIds->begin())
					printf(", ");
				printf("%d", (*cell));
			}
			printf("]");
			break;
		}
		case org::apache::arrow::flatbuf::Type::FixedSizeBinary: {
			auto	__type = field->type_as_FixedSizeBinary();
			printf("FixedSizeBinary: byteWidth=%d", __type->byteWidth());
			break;
		}
		case org::apache::arrow::flatbuf::Type::FixedSizeList: {
			auto	__type = field->type_as_FixedSizeList();
			printf("FixedSizeList: listSize=%d", __type->listSize());
			break;
		}
		case org::apache::arrow::flatbuf::Type::Map: {
			auto	__type = field->type_as_Map();
			printf("Map: keysSorted=%s", __type->keysSorted() ? "true" : "false");
			break;
		}
		case org::apache::arrow::flatbuf::Type::Duration: {
			auto	__type = field->type_as_Duration();
			switch (__type->unit())
			{
				case org::apache::arrow::flatbuf::TimeUnit::SECOND:
					printf("Duration: unit=sec");
					break;
                case org::apache::arrow::flatbuf::TimeUnit::MILLISECOND:
                    printf("Duration: unit=ms");
                    break;
                case org::apache::arrow::flatbuf::TimeUnit::MICROSECOND:
                    printf("Duration: unit=us");
                    break;
                case org::apache::arrow::flatbuf::TimeUnit::NANOSECOND:
                    printf("Duration: unit=ns");
                    break;
                default:
                    printf("Duration: unit=???");
					break;
			}
			break;
		}
		case org::apache::arrow::flatbuf::Type::LargeBinary:
			printf("LargeBinary");
			break;
		case org::apache::arrow::flatbuf::Type::LargeUtf8:
			printf("LargeUtf8");
			break;
		case org::apache::arrow::flatbuf::Type::LargeList:
			printf("LargeList");
			break;
		default:
			printf("Unknown");
			break;
	}
	printf("}");
	if (dict)
	{
		auto __type = dict->indexType();
		printf(", dict={DictionaryEncoding: id=%ld, indexType={%sInt%d}, isOrdered=%s}",
			   dict->id(),
			   (__type && !__type->is_signed()) ? "U" : "",
			   (__type ? __type->bitWidth() : 32),
			   dict->isOrdered() ? "true" : "false");
	}
	printf(", children=[");
	if (children)
	{
		for (auto cell=children->begin(); cell != children->end(); cell++)
		{
			auto	__field = (*cell);

			if (cell != children->begin())
				printf(", ");
			__dump_arrow_field(__field);
		}
	}
	printf("], custom_metadata=[");
	__dump_arrow_custom_metadata(field->custom_metadata());
	printf("]}");
}

static void
__dump_arrow_schema(const org::apache::arrow::flatbuf::Schema *schema)
{
	auto	__endian = schema->endianness();
	auto	__fields = schema->fields();
	auto	__custom_metadata = schema->custom_metadata();

	printf("{Schema: endianness=%s, fields=[",
		   EnumNameEndianness(__endian));
	for (auto cell = __fields->begin(); cell != __fields->end(); cell++)
	{
		auto	field = (*cell);

		if (cell != __fields->begin())
			printf(", ");
		__dump_arrow_field(field);
	}
	printf("], custom_metadata=[");
	__dump_arrow_custom_metadata(__custom_metadata);
	printf("]}");
}

static void
__dump_arrow_record_batch(const org::apache::arrow::flatbuf::RecordBatch *rbatch)
{
	auto	__length      = rbatch->length();
	auto	__nodes       = rbatch->nodes();
	auto	__buffers     = rbatch->buffers();
	auto	__compression = rbatch->compression();


	printf("{RecordBatch: length=%ld, nodes=[", __length);
	/* const flatbuffers::Vector<const org::apache::arrow::flatbuf::FieldNode *> */
	for (auto cell = __nodes->begin(); cell != __nodes->end(); cell++)
	{
		auto	fnode = (*cell);

		if (cell != __nodes->begin())
			putchar(',');
		printf("{FieldNode: length=%ld, nullcount=%ld}",
			   fnode->length(),
			   fnode->null_count());
	}
	printf("], buffers=[");
	/* const flatbuffers::Vector<const org::apache::arrow::flatbuf::Buffer *> */
	for (auto cell = __buffers->begin(); cell != __buffers->end(); cell++)
	{
		auto	fbuf = (*cell);

		if (cell != __buffers->begin())
			printf(", ");
		printf("{Buffer: offset=%ld, length=%ld}",
			   fbuf->offset(),
			   fbuf->length());
	}
	printf("]");
	if (__compression)
	{
		auto	codec = __compression->codec();
		auto	method = __compression->method();

		printf(", compression=<codec=%s, method=%s>",
			   EnumNameCompressionType(codec),
			   EnumNameBodyCompressionMethod(method));
	}
	printf("}");
}

static void
__dump_arrow_dictionary_batch(const org::apache::arrow::flatbuf::DictionaryBatch *dbatch)
{
	printf("{DictionaryBatch: id=%ld, record_batch=",
		   dbatch->id());
	__dump_arrow_record_batch(dbatch->data());
	printf(", isDelta=%s", dbatch->isDelta() ? "true" : "false");
}

static void
__dump_arrow_message(std::shared_ptr<arrow::io::ReadableFile> arrow_file,
					 const org::apache::arrow::flatbuf::Block *block)
{
	auto	rv = arrow_file->ReadAt(block->offset(),
									block->metaDataLength());
	if (!rv.ok())
		Elog("failed on arrow::io::ReadableFile::ReadAt: %s",
			 rv.status().ToString().c_str());
	auto	buffer = rv.ValueOrDie();
	auto	fb_base = buffer->data() + sizeof(uint32_t);	/* Continuation token (0xffffffff) */
	auto	message = org::apache::arrow::flatbuf::GetSizePrefixedMessage(fb_base);

	switch (message->header_type())
	{
		case org::apache::arrow::flatbuf::MessageHeader::Schema:
			__dump_arrow_schema(message->header_as_Schema());
			break;
		case org::apache::arrow::flatbuf::MessageHeader::DictionaryBatch:
			__dump_arrow_dictionary_batch(message->header_as_DictionaryBatch());
			break;
		case org::apache::arrow::flatbuf::MessageHeader::RecordBatch:
			__dump_arrow_record_batch(message->header_as_RecordBatch());
			break;
		case org::apache::arrow::flatbuf::MessageHeader::Tensor:
			Elog("Not implemented yet for MessageHeader::Tensor");
			//__dump_arrow_tensor(message->header_as_Tensor());
			break;
		case org::apache::arrow::flatbuf::MessageHeader::SparseTensor:
			//__dump_arrow_sparse_tensor(message->header_as_SparseTensor());
			Elog("Not implemented yet for MessageHeader::SparseTensor");
			break;
		default:
			Elog("Arrow file corruption? unknown message type %d", (int)message->header_type());
	}
}

void
dump_arrow_metadata(std::shared_ptr<arrow::io::ReadableFile> arrow_file, const char *filename)
{
	auto	file_sz = arrow_file->GetSize().ValueOrDie();
	auto	tail_sz = sizeof(uint32_t) + sizeof(ARROW_SIGNATURE)-1;
	int32_t	footer_sz;

	/* validate arrow file tail */
	{
		auto	rv = arrow_file->ReadAt(file_sz - tail_sz, tail_sz);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile::ReadAt: %s",
				 rv.status().ToString().c_str());
		auto	buffer = rv.ValueOrDie();

		if (memcmp((char *)buffer->data() + sizeof(uint32_t),
				   ARROW_SIGNATURE,
				   sizeof(ARROW_SIGNATURE)-1) != 0)
			Elog("signature check failed on '%s'", filename);
		footer_sz = *((int32_t *)buffer->data());
	}
	/* read the footer flat-buffer */
	{
		auto	rv = arrow_file->ReadAt(file_sz
										- tail_sz
										- footer_sz, footer_sz);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile::ReadAt: %s",
				 rv.status().ToString().c_str());
		auto	buffer = rv.ValueOrDie();
		auto	footer = flatbuffers::GetRoot<org::apache::arrow::flatbuf::Footer>(buffer->data());
		auto	__version = footer->version();
		auto	__schema = footer->schema();
		auto	__dictionaries = footer->dictionaries();
		auto	__record_batches = footer->recordBatches();
		auto	__custom_metadata = footer->custom_metadata();
		int		count;

		if (__version < org::apache::arrow::flatbuf::MetadataVersion::V4)
			Elog("No backward compatibility for Apache Arrow metadata version 3 or former");

		printf("[Footer]\n"
			   "{Footer: filename='%s', version=%s, schema=[",
			   filename,
			   org::apache::arrow::flatbuf::EnumNameMetadataVersion(__version));
		__dump_arrow_schema(__schema);
		printf("], dictionaries=[");
		for (auto cell = __dictionaries->begin(); cell != __dictionaries->end(); cell++)
		{
			auto	block = (*cell);

			if (cell != __dictionaries->begin())
				putchar(',');
			printf("{Block: offset=%ld, metaLen=%d, bodyLen=%ld}",
				   block->offset(),
				   block->metaDataLength(),
				   block->bodyLength());
		}
		printf("], recordBatches=[");
		for (auto cell = __record_batches->begin(); cell != __record_batches->end(); cell++)
		{
			auto	block = (*cell);

			if (cell != __record_batches->begin())
				putchar(',');
			printf("{Block: offset=%ld, metaLen=%d, bodyLen=%ld}",
				   block->offset(),
				   block->metaDataLength(),
				   block->bodyLength());
		}
		printf("], customMetadata=[");
		__dump_arrow_custom_metadata(__custom_metadata);
		printf("]}\n");
		/* Dump for each Dictionaries */
		count=0;
		for (auto cell = __dictionaries->begin(); cell != __dictionaries->end(); cell++)
		{
			auto	block = (*cell);
			printf("[Dictionary Batch-%d; offset: 0x%08lx, meta: %d, body: %ld]\n",
				   count++,
				   block->offset(),
				   block->metaDataLength(),
				   block->bodyLength());
			__dump_arrow_message(arrow_file, block);
			putchar('\n');
		}
		/* Dump for each Record-Batch */
		count = 0;
		for (auto cell = __record_batches->begin(); cell != __record_batches->end(); cell++)
		{
			auto	block = (*cell);
			printf("[Record Batch-%d; offset: 0x%08lx, meta: %d, body: %ld]\n",
				   count++,
				   block->offset(),
				   block->metaDataLength(),
				   block->bodyLength());
			__dump_arrow_message(arrow_file, block);
			putchar('\n');
		}
	}
}

#ifdef HAS_PARQUET
// ================================================================
//
// dump_arrow_metadata(parquet_file, filename)
//
// ================================================================
static void
__dump_parquet_column(const parquet::ColumnDescriptor *column)
{
	auto	ptype = column->physical_type();
	auto	ctype = column->converted_type();
//	auto	ltype = column->logical_type();

	std::cout << "name=\"" << column->name()
			  << "\", physical_type=";
	switch (ptype)
	{
		case parquet::Type::BOOLEAN:
			std::cout << "Boolean";
			break;
		case parquet::Type::INT32:
			std::cout << "Int32";
			break;
		case parquet::Type::INT64:
			std::cout << "Int64";
			break;
		case parquet::Type::INT96:
			std::cout << "Int96";
			break;
		case parquet::Type::FLOAT:
			std::cout << "Float";
			break;			
		case parquet::Type::DOUBLE:
			std::cout << "Double";
			break;
		case parquet::Type::BYTE_ARRAY:
			std::cout << "ByteArray";
			break;
		case parquet::Type::FIXED_LEN_BYTE_ARRAY:
			std::cout << "FixedLenByteArray[" << column->type_length() << "]";
			break;
		default:
			std::cout << "unknown";
			break;
	}
	std::cout << ", converted_type=";
	switch (ctype)
	{
		case parquet::ConvertedType::NONE:
			std::cout << "NONE";
			break;
		case parquet::ConvertedType::UTF8:
			std::cout << "Utf8";
			break;
		case parquet::ConvertedType::MAP:
			std::cout << "Map";
			break;
		case parquet::ConvertedType::MAP_KEY_VALUE:
			std::cout << "MapKeyValue";
			break;
		case parquet::ConvertedType::LIST:
			std::cout << "List";
			break;
		case parquet::ConvertedType::ENUM:
			std::cout << "Enum";
			break;
		case parquet::ConvertedType::DECIMAL:
			std::cout << "Decimal("
					  << column->type_precision() << ","
					  << column->type_scale() << ")";
			break;
		case parquet::ConvertedType::DATE:
			std::cout << "Date";
			break;
		case parquet::ConvertedType::TIME_MILLIS:
			std::cout << "Time[ms]";
			break;
		case parquet::ConvertedType::TIME_MICROS:
			std::cout << "Time[us]";
			break;
		case parquet::ConvertedType::TIMESTAMP_MILLIS:
			std::cout << "Timestamp[ms]";
			break;
		case parquet::ConvertedType::TIMESTAMP_MICROS:
			std::cout << "Timestamp[us]";
			break;
		case parquet::ConvertedType::UINT_8:
			std::cout << "UInt8";
			break;
		case parquet::ConvertedType::UINT_16:
			std::cout << "UInt16";
			break;
		case parquet::ConvertedType::UINT_32:
			std::cout << "UInt32";
			break;
		case parquet::ConvertedType::UINT_64:
			std::cout << "UInt64";
			break;
		case parquet::ConvertedType::INT_8:
			std::cout << "Int8";
			break;
		case parquet::ConvertedType::INT_16:
			std::cout << "Int16";
			break;
		case parquet::ConvertedType::INT_32:
			std::cout << "Int32";
			break;
		case parquet::ConvertedType::INT_64:
			std::cout << "Int64";
			break;
		case parquet::ConvertedType::JSON:
			std::cout << "Json";
			break;
		case parquet::ConvertedType::BSON:
			std::cout << "Bson";
			break;
		case parquet::ConvertedType::INTERVAL:
			std::cout << "Internal";
			break;
		default:
			std::cout << "unknown";
			break;
	}
}

void
dump_parquet_metadata(std::shared_ptr<arrow::io::ReadableFile> parquet_file,
					  const char *filename)
{
	auto	parquet_reader = parquet::ParquetFileReader::Open(parquet_file);
	auto	parquet_meta = parquet_reader->metadata();
	auto	parquet_schema = parquet_meta->schema();
	auto	parquet_kv = parquet_meta->key_value_metadata();

	// Print Global file-level metadata
	std::cout << "[Parquet File]" << std::endl;
	std::cout << "filename: " << filename << std::endl;
	std::cout << "version: " << ParquetVersionToString(parquet_meta->version()) << std::endl;
	std::cout << "created_by: " << parquet_meta->created_by() << std::endl;
	std::cout << "num_columns: " << parquet_meta->num_columns() << std::endl;
	std::cout << "num_total_rows: " << parquet_meta->num_rows() << std::endl;
	std::cout << "num_row_groups: " << parquet_meta->num_row_groups() << std::endl;
	// Schema Definition
	std::cout << "[Schema]" << std::endl;
	std::cout << "schema_name: " << parquet_schema->name() << std::endl;
	for (int j=0; j < parquet_schema->num_columns(); j++)
	{
		auto	column = parquet_schema->Column(j);

		std::cout << "column[" << (j+1) << "]: ";
		__dump_parquet_column(column);
		std::cout << std::endl;
	}
	// Key-Value Metadata
	if (parquet_kv)
	{
		std::cout << "[Key-Values]" << std::endl;
		for (int i=0; i < parquet_kv->size(); i++)
		{
			std::cout << parquet_kv->key(i) << ": "
					  << parquet_kv->value(i) << std::endl;
		}
	}
	// For each Row-Groups
	for (int i=0; i < parquet_meta->num_row_groups(); i++)
	{
		auto	rg_meta = parquet_meta->RowGroup(i);
		std::cout << "[Row Group-" << (i+1) << "]" << std::endl;
		std::cout << "file_offset: " << rg_meta->file_offset() << std::endl;
		std::cout << "num_rows: " << rg_meta->num_rows() << std::endl;
		if (rg_meta->total_byte_size() > 0)
			std::cout << "raw bytesize: " << rg_meta->total_byte_size() << std::endl;
		std::cout << "can_decompress: " << (rg_meta->can_decompress() ? "True" : "False") << std::endl;
		// Iterate over columns in the row group
		for (int j=0; j < rg_meta->num_columns(); j++)
		{
			auto	col_meta = rg_meta->ColumnChunk(j);
			auto	col_path = col_meta->path_in_schema();
			auto	col_stat = col_meta->statistics();

			std::cout << "column_chunk[" << (j+1) << "]: "
					  << "data_page_offset=" << col_meta->data_page_offset();
			if (col_meta->has_dictionary_page())
				std::cout << ", dictionary_page_offset=" << col_meta->dictionary_page_offset();
			if (col_meta->has_index_page())
				std::cout << ", index_page_offset=" << col_meta->index_page_offset();
			std::cout << ", total_compressed_size=" << col_meta->total_compressed_size()
					  << ", total_uncompressed_size=" << col_meta->total_uncompressed_size()
					  << ", can_decompress=" << (col_meta->can_decompress() ? "True" : "False")
					  << ", encodings=[";
			auto	col_encodings = col_meta->encodings();
			for (auto cell = col_encodings.begin(); cell != col_encodings.end(); cell++)
			{
				auto	enc = (*cell);

				if (cell != col_encodings.begin())
					std::cout << ", ";
				switch (enc)
				{
					case parquet::Encoding::PLAIN:
						std::cout << "PLAIN";
						break;
					case parquet::Encoding::PLAIN_DICTIONARY:
						std::cout << "PLAIN_DICTIONARY";
						break;
					case parquet::Encoding::RLE:
						std::cout << "RLE";
						break;
					case parquet::Encoding::BIT_PACKED:
						std::cout << "BIT_PACKED";
						break;
					case parquet::Encoding::DELTA_BINARY_PACKED:
						std::cout << "DELTA_BINARY_PACKED";
						break;
					case parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY:
						std::cout << "DELTA_LENGTH_BYTE_ARRAY";
						break;
					case parquet::Encoding::DELTA_BYTE_ARRAY:
						std::cout << "DELTA_BYTE_ARRAY";
						break;
					case parquet::Encoding::RLE_DICTIONARY:
						std::cout << "RLE_DICTIONARY";
						break;
					case parquet::Encoding::BYTE_STREAM_SPLIT:
						std::cout << "BYTE_STREAM_SPLIT";
						break;
					default:
						std::cout << "UNKNOWN_ENCODING";
						break;
				}
			}
			std::cout << "]";
			/* Compression::type */
			std::cout << ", compression=";
			switch (col_meta->compression())
			{
				case parquet::Compression::UNCOMPRESSED:
					std::cout << "uncompressed";
					break;
				case parquet::Compression::SNAPPY:
					std::cout << "snappy";
					break;
				case parquet::Compression::GZIP:
					std::cout << "gzip";
					break;
				case parquet::Compression::BROTLI:
					std::cout << "brotli";
					break;
				case parquet::Compression::ZSTD:
					std::cout << "zstd";
					break;
				case parquet::Compression::LZ4:
					std::cout << "lz4";
					break;
				case parquet::Compression::LZ4_FRAME:
					std::cout << "lz4_frame";
					break;
				case parquet::Compression::LZO:
					std::cout << "lzo";
					break;
				case parquet::Compression::BZ2:
					std::cout << "bz2";
					break;
				case parquet::Compression::LZ4_HADOOP:
					std::cout << "lz4_hadoop";
					break;
				default:
					std::cout << "unknown";
					break;
			}
			/* std::shared_ptr<schema::ColumnPath> */
			std::cout << ", column_path=\"" << col_path->ToDotString() << "\"";

			/* std::shared_ptr<Statistics> */
			if (col_stat->HasNullCount())
				std::cout << ", null_count=" << col_stat->null_count();
			if (col_stat->HasDistinctCount())
				std::cout << ", distinct_count=" << col_stat->distinct_count();
			std::cout << ", num_values=" << col_stat->num_values();
			if (col_stat->HasMinMax())
			{
				switch (col_stat->physical_type())
				{
					case parquet::Type::BOOLEAN: {
						auto	__stat = std::dynamic_pointer_cast<parquet::BoolStatistics>(col_stat);
						std::cout << ", min_value=" << __stat->min()
								  << ", max_value=" << __stat->max();
						break;
					}
					case parquet::Type::INT32: {
						auto	__stat = std::dynamic_pointer_cast<parquet::Int32Statistics>(col_stat);
						std::cout << ", min_value=" << __stat->min()
								  << ", max_value=" << __stat->max();
						break;
					}
					case parquet::Type::INT64: {
						auto	__stat = std::dynamic_pointer_cast<parquet::Int64Statistics>(col_stat);
						std::cout << ", min_value=" << __stat->min()
								  << ", max_value=" << __stat->max();
						break;
					}
					case parquet::Type::FLOAT: {
						auto	__stat = std::dynamic_pointer_cast<parquet::FloatStatistics>(col_stat);
						std::cout << ", min_value=" << __stat->min()
								  << ", max_value=" << __stat->max();
						break;
					}
					case parquet::Type::DOUBLE: {
						auto	__stat = std::dynamic_pointer_cast<parquet::DoubleStatistics>(col_stat);
						std::cout << ", min_value=" << __stat->min()
								  << ", max_value=" << __stat->max();
						break;
					}
#if 0
					// needs special encoding logic
					case parquet::Type::BYTE_ARRAY: {
						auto	__stat = std::dynamic_pointer_cast<parquet::ByteArrayStatistics>(col_stat);
						std::cout << ", min_value=" << __stat->min()
								  << ", max_value=" << __stat->max();
						break;
					}
					case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
						auto	__stat = std::dynamic_pointer_cast<parquet::FLBAStatistics>(col_stat);
						std::cout << ", min_value=" << __stat->min()
								  << ", max_value=" << __stat->max();
						break;
					}
#endif
					default: {
						std::cout << ", min_value=" << col_stat->EncodeMin()
								  << ", max_value=" << col_stat->EncodeMax();
						break;
					}
				}
			}
			std::cout << std::endl;
		}
	}
}
#endif	/* HAS_PARQUET */
