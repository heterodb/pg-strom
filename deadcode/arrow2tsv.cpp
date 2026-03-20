/*
 * arrow2tsv.cc
 *
 * A tool to dump Apache Arrow/Parquet file as CSV/TSV format
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <arrow/api.h>		/* dnf install arrow-devel, or apt install libarrow-dev */
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <getopt.h>
#include <parquet/column_reader.h>
#include <parquet/metadata.h>
#include <parquet/schema.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>



/*
 * static variables and command line argument
 */
static std::vector<const char *> input_filenames;
static const char  *output_filename = NULL;
static FILE		   *output_filp = NULL;
static bool			only_metadata = false;
static bool			shows_header = false;
static bool			csv_mode = false;
static long			skip_offset = 0;
static long			skip_limit = -1;
static int			verbose = 0;
static const char  *with_create_table = NULL;
static const char  *with_tablespace = NULL;
static const char  *with_partition_of = NULL;

// ------------------------------------------------
// Error Reporting
// ------------------------------------------------
#define Elog(fmt,...)									\
	do {												\
		fprintf(stderr, "[ERROR %s:%d] " fmt "\n",		\
				__FILE__,__LINE__, ##__VA_ARGS__);		\
		exit(1);										\
	} while(0)
#define Info(fmt,...)									\
	do {												\
		fprintf(stderr, "[INFO %s:%d] " fmt "\n",		\
				__FILE__,__LINE__, ##__VA_ARGS__);		\
	} while(0)
#define Debug(fmt,...)									\
	do {												\
		if (verbose)									\
			fprintf(stderr, "[DEBUG %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
	} while(0)

/*
 * checkInputFileFormat
 */
static bool
checkInputFileFormat(std::shared_ptr<arrow::io::ReadableFile> filp, const char *fname)
{
	char	buffer[8];
	size_t	nbytes;
	size_t	length;
	{
		auto	rv = filp->GetSize();
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')::GetSize: %s",
				 fname, rv.status().ToString().c_str());
		length = rv.ValueOrDie();
	}
	{
		auto	rv = filp->ReadAt(0, 6, buffer);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')::ReadAt: %s",
				 fname, rv.status().ToString().c_str());
		nbytes = rv.ValueOrDie();
	}
	if (nbytes >= 6 && memcmp(buffer, "ARROW1", 6) == 0)
	{
		auto	rv = filp->ReadAt(length-6, 6, buffer);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')::ReadAt: %s",
				 fname, rv.status().ToString().c_str());
		nbytes = rv.ValueOrDie();
		if (nbytes == 6 && memcmp(buffer, "ARROW1", 6) == 0)
			return false;	/* Arrow File */
	}
	if (nbytes >= 4 && memcmp(buffer, "PAR1", 4) == 0)
	{
		auto	rv = filp->ReadAt(length-4, 4, buffer);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')::ReadAt: %s",
				 fname, rv.status().ToString().c_str());
		nbytes = rv.ValueOrDie();
		if (nbytes == 4 && memcmp(buffer, "PAR1", 4) == 0)
			return true;	/* Parquet file */
	}
	Elog("file '%s' is neither Arrow nor Parquet", fname);
}

/*
 * fetchParquetMetadata
 */
static std::shared_ptr<parquet::FileMetaData>
fetchParquetMetadata(std::shared_ptr<arrow::io::ReadableFile> filp, const char *fname)
{
	struct {
		uint32_t  metadata_len;
		char	signature[4];
	} foot;
	std::shared_ptr<arrow::Buffer> buffer;
	size_t		length;
	/* fetch metadata length */
	{
		auto	rv = filp->GetSize();
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')::GetSize: %s",
				 fname, rv.status().ToString().c_str());
		length = rv.ValueOrDie();
	}
	{
		size_t	nbytes;
		auto	rv = filp->ReadAt(length-8, 8, &foot);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')::ReadAt: %s",
				 fname, rv.status().ToString().c_str());
		nbytes = rv.ValueOrDie();
		if (nbytes != 8 || memcmp(foot.signature, "PAR1", 4) != 0)
			Elog("signature check failed: file '%s' is not Parquet", fname);
	}
	/* read binary metadata */
	{
		auto	rv = filp->ReadAt(length -8 -foot.metadata_len, foot.metadata_len);

		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')::ReadAt: %s",
				 fname, rv.status().ToString().c_str());
		buffer = rv.ValueOrDie();
	}
	return parquet::FileMetaData::Make(buffer->data(), &foot.metadata_len);
}

/*
 * __loadParquetInt64ColumnChunk
 */
static int64_t
__loadParquetInt64ColumnChunk(const parquet::RowGroupMetaData &rg_meta,
							  const parquet::ColumnChunkMetaData &cc_meta,
							  std::shared_ptr<parquet::ColumnReader> __col_reader)
{
	auto		i64_reader = static_cast<parquet::Int64Reader *>(__col_reader.get());
	int64_t		nrooms = rg_meta.num_rows();
	int64_t		num_defs;
	int64_t		num_values;
	int64_t		i, j;
	std::vector<int16_t> def(nrooms);
	std::vector<int64_t> values(nrooms);

	num_defs = i64_reader->ReadBatch(nrooms, def.data(), nullptr, values.data(), &num_values);
	printf("num_defs=%ld num_values=%ld\n", num_defs, num_values);
	for (i=0, j=0; i < num_defs; i++)
	{
		if (i > 0)
			putchar(',');
		if (def[i] != 0)
			printf("%ld",values[j++]);
	}
	putchar('\n');
	return 0;
}

/*
 * __loadParquetFloat64ColumnChunk
 */
static int64_t
__loadParquetFloat64ColumnChunk(const parquet::RowGroupMetaData &rg_meta,
								const parquet::ColumnChunkMetaData &cc_meta,
								std::shared_ptr<parquet::ColumnReader> __col_reader)
{
	auto		f64_reader = static_cast<parquet::DoubleReader *>(__col_reader.get());
	int64_t		nrooms = rg_meta.num_rows();
	int64_t		nitems;
	int64_t		i, j;
	std::vector<int16_t> def(nrooms);
	std::vector<double> values(nrooms);

	nrooms = f64_reader->ReadBatch(nrooms, def.data(), nullptr, values.data(), &nitems);
	printf("num_defs=%ld num_values=%ld\n", nrooms, nitems);
	for (i=0, j=0; i < nrooms; i++)
	{
		if (i > 0)
			putchar(',');
		if (def[i] != 0)
			printf("%f",values[j++]);
	}
	putchar('\n');
	return 0;
}

/*
 * __loadParquetBinaryColumnChunk
 */
static int64_t
__loadParquetBinaryColumnChunk(const parquet::RowGroupMetaData &rg_meta,
							   const parquet::ColumnChunkMetaData &cc_meta,
							   std::shared_ptr<parquet::ColumnReader> __col_reader)
{
	auto		bin_reader = static_cast<parquet::ByteArrayReader *>(__col_reader.get());
	int64_t		nrooms = rg_meta.num_rows();
	int64_t		nitems;
	int64_t		i, j;
	std::vector<int16_t> def(nrooms);
	std::vector<parquet::ByteArray> values(nrooms);

	nrooms = bin_reader->ReadBatch(nrooms, def.data(), nullptr, values.data(), &nitems);
	printf("num_defs=%ld num_values=%ld\n", nrooms, nitems);
	for (i=0, j=0; i < nrooms; i++)
	{
		if (i > 0)
			putchar(',');
		if (def[i] != 0)
		{
			auto	datum = values[j++];
			printf("[%.*s]", datum.len, datum.ptr);
		}
	}
	putchar('\n');
	return 0;
}

/*
 * loadParquetOneColumnChunk
 */
static int64_t
loadParquetOneColumnChunk(std::shared_ptr<arrow::io::ReadableFile> input_filp,
						  const char *input_fname,
						  const parquet::RowGroupMetaData &rg_meta,
						  const parquet::ColumnChunkMetaData &cc_meta,
						  const parquet::ColumnDescriptor *c_descr)
{
	std::shared_ptr<arrow::io::BufferReader> input_stream;
	int64_t		f_length = cc_meta.total_compressed_size();
	int64_t		f_offset = cc_meta.data_page_offset();

	if (cc_meta.has_dictionary_page() &&
		f_offset > cc_meta.dictionary_page_offset())
		f_offset = cc_meta.dictionary_page_offset();
	if (cc_meta.has_index_page() &&
		f_offset > cc_meta.index_page_offset())
		f_offset = cc_meta.index_page_offset();
	{
		auto	rv = input_filp->ReadAt(f_offset, f_length);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile('%s')->ReadAt(%ld,%ld): %s",
				 input_fname, f_offset, f_length,
				 rv.status().ToString().c_str());
		//makes the buffer as input stream
		input_stream = std::make_shared<arrow::io::BufferReader>(rv.ValueOrDie());
	}
	auto	p_reader = parquet::PageReader::Open(input_stream,
												 rg_meta.num_rows(),
												 cc_meta.compression());
	auto	c_reader = parquet::ColumnReader::Make(c_descr, std::move(p_reader));
	auto	physical_type = c_descr->physical_type();

	switch (physical_type)
	{
		case parquet::Type::BOOLEAN: {
			auto	b_reader = static_cast<parquet::BoolReader *>(c_reader.get());
			break;
		}
		case parquet::Type::INT32: {
			auto	i_reader = static_cast<parquet::Int32Reader *>(c_reader.get());
			break;
		}
		case parquet::Type::INT64:
			return __loadParquetInt64ColumnChunk(rg_meta, cc_meta, c_reader);
		case parquet::Type::FLOAT: {
			auto	f_reader = static_cast<parquet::FloatReader *>(c_reader.get());
			break;
		}
		case parquet::Type::DOUBLE:
			return __loadParquetFloat64ColumnChunk(rg_meta, cc_meta, c_reader);
		case parquet::Type::BYTE_ARRAY:
			return __loadParquetBinaryColumnChunk(rg_meta, cc_meta, c_reader);
		case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
			auto	flb_reader = static_cast<parquet::FixedLenByteArrayReader *>(c_reader.get());
			break;
		}
		default:
			Info("Unknown physical type: %s", parquet::TypeToString(physical_type).c_str());
			break;
	}
	return -1;
}

/*
 * dumpParquetOneRowGroup
 */
static void
dumpParquetOneRowGroup(std::shared_ptr<arrow::io::ReadableFile> input_filp,
					   const char *input_fname,
					   const parquet::RowGroupMetaData &rg_meta)
{
	auto	schema = rg_meta.schema();

	std::cout << schema->ToString() << std::endl;
	for (int j=0; j < schema->num_columns(); j++)
		loadParquetOneColumnChunk(input_filp,
								  input_fname,
								  rg_meta,
								  *rg_meta.ColumnChunk(j),
								  schema->Column(j));
}



static void usage(const char *fmt, ...)
{
	if (fmt)
	{
		va_list	ap;

		va_start(ap, fmt);
		vfprintf(stderr, fmt, ap);
		va_end(ap);
		fputc('\n', stderr);
	}
	fputs("usage: arrow2tsv [OPTIONS] <file1> [<file2> ...]\n"
		  "\n"
		  "-o|--output=FILENAME specify the output filename (default: stdout)\n"
		  "   --csv             dump in CSV mode\n"
		  "   --meta            dump metadata of arrow/parquet\n"
		  "   --header          dump column names as CSV/TSV header\n"
		  "   --offset=NUM      skip first NUM rows\n"
		  "   --limit=NUM       dump only NUM rows\n"
		  "\n"
		  "   --create-table=TABLE_NAME   dump with CREATE TABLE statement\n"
		  "   --tablespace=TABLESPACE     specify tablespace of the table, if any\n"
		  "   --partition-of=PARENT_NAME  specify partition-parent of the table, if any\n"
		  "\n"
		  "-v|--verbose         verbose output\n"
		  "-h|--help            display this message\n"
		  "\n"
		  "arrow2tsv version " PGSTROM_VERSION " - reports bugs to <pgstrom@heterodb.com>.\n",
		  stderr);
	exit(1);
}

static void parse_options(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"output",       required_argument, NULL, 'o'},
		{"csv",          no_argument,       NULL, 1001},
		{"meta",         no_argument,       NULL, 1002},
		{"header",       no_argument,       NULL, 1003},
		{"offset",       required_argument, NULL, 1004},
		{"limit",        required_argument, NULL, 1005},
		{"create-table", required_argument, NULL, 1006},
		{"tablespace",   required_argument, NULL, 1007},
		{"partition-of", required_argument, NULL, 1008},
		{"verbose",      no_argument,       NULL, 'v'},
		{"help",         no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	int		c;
	char   *end;

	while ((c = getopt_long(argc, argv, "o:vh",
							long_options, NULL)) >= 0)
	{
		switch (c)
		{
			case 'o':	/* --output */
				if (output_filename)
					usage("-o|--output was given twice");
				output_filename = optarg;
				break;
			case 1001:	/* --csv */
				csv_mode = true;
				break;
			case 1002:	/* --meta */
				only_metadata = true;
				break;
			case 1003:	/* --header */
				shows_header = true;
				break;
			case 1004:	/* --offset */
				skip_offset = strtol(optarg, &end, 10);
				if (*end != '\0' || skip_offset < 0)
					usage("invalid --offset value '%s'", optarg);
				break;
			case 1005:	/* --limit */
				skip_limit = strtol(optarg, &end, 10);
				if (*end != '\0' || skip_offset < 0)
					usage("invalid --limit value '%s'", optarg);
				break;
			case 1006:	/* --create-table */
				if (with_create_table)
					usage("--create-table was given twice");
				with_create_table = optarg;
				break;
			case 1007:	/* --tablespace */
				if (with_tablespace)
					usage("--tablespace was given twice");
				with_tablespace = optarg;
				break;
			case 1008:	/* --partition-of */
				if (with_partition_of)
					usage("--partition-of was given twice");
				with_partition_of = optarg;
				break;
			case 'v':	/* --verbose */
				verbose++;
				break;
			default:	/* --help */
				usage(NULL);
		}
	}
	for (int k=optind; k < argc; k++)
		input_filenames.push_back(argv[k]);
	if (input_filenames.empty())
		usage("no input files given");
	if (shows_header && with_create_table)
		usage("--header and --create-table are mutually exclusive");
	if (with_tablespace && !with_create_table)
		usage("--tablespace must be used with --create-table");
	if (with_partition_of && !with_create_table)
		usage("--partition-of must be used with --create-table");
}

int main(int argc, char *argv[])
{
	parse_options(argc, argv);
	for (int k=0; k < input_filenames.size(); k++)
	{
		std::shared_ptr<arrow::io::ReadableFile> input_filp;
		const char *input_fname = input_filenames[k];
		bool		parquet_mode;
		
		/* open the input file */
		{
			auto	rv = arrow::io::ReadableFile::Open(std::string(input_fname));
			if (!rv.ok())
				Elog("failed on arrow::io::ReadableFile::Open('%s'): %s",
					 input_fname,
					 rv.status().ToString().c_str());
			input_filp = rv.ValueOrDie();
		}
		/* check input file format */
		parquet_mode = checkInputFileFormat(input_filp, input_fname);
		/* fetch the metadata */
		if (parquet_mode)
		{
			auto md = fetchParquetMetadata(input_filp, input_fname);
			for (int k=0; k < md->num_row_groups(); k++)
			{
				dumpParquetOneRowGroup(input_filp,
									   input_fname,
									   *md->RowGroup(k));
			}
		}
		else
		{
			Elog("Arrow mode is not implemented now");
		}
	}
	return 0;
}
