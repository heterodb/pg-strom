/*
 * vcf2arrow.cpp
 *
 * VCF format to Apache Arrow/Parquet converter
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <fcntl.h>
#include <iostream>
#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>
#include "unistd.h"
#include "arrow_write.h"

/* command line arguments */
static std::vector<const char *> input_filenames;
static const char  *output_filename = NULL;
static bool			force_write_file = false;
static bool			parquet_mode = false;
static bool			shows_progress = false;
static bool			verbose = false;
static size_t		batch_segment_sz = (1UL<<30);	/* default: 1GB */
static arrow::Compression::type default_compression = arrow::Compression::type::ZSTD;
static std::vector<std::string>	user_custom_metadata;
static const char  *schema_table_name = NULL;
static const char  *error_items_filename = NULL;
static FILE		   *error_items_filp = NULL;

/*
 * report_unconvertible_entry
 */
static void
report_unconvertible_entry(const char *line)
{
	if (!error_items_filp)
	{
		if (!error_items_filename)
			Elog("This line is not convertible, and -e OUTFILE is not given.\n%s", line);
		error_items_filp = fopen(error_items_filename, "ab");
		if (!error_items_filp)
			Elog("unable to open '%s' for error reporting", error_items_filename);
	}
	fputs(line, error_items_filp);
}

/*
 * openArrowFileStream
 */
static std::shared_ptr<arrow::io::FileOutputStream>
openArrowFileStream(void)
{
	int		output_fdesc;

	if (output_filename)
	{
		int		flags = O_RDWR | O_CREAT | O_TRUNC;

		if (!force_write_file)
			flags |= O_EXCL;
		output_fdesc = open(output_filename, flags, 0644);
		if (output_fdesc < 0)
			Elog("unable to open output file '%s': %m", output_filename);
	}
	else
	{
		const char *input_fname = input_filenames[0];
		size_t		len = strlen(input_fname);
		char	   *temp = (char *)alloca(len+50);
		char	   *dpos;
		const char *suffix = (!parquet_mode ? "arrow" : "parquet");

		strcpy(temp, input_fname);
		if (len > 4 && strcasecmp(temp+len-4, ".vcf") == 0)
			dpos = temp+len-4;
		else
			dpos = temp+len;
		for (int loop=0; loop < 100; loop++)
		{
			if (loop == 0)
				sprintf(dpos, ".%s", suffix);
			else
				sprintf(dpos, ".%d.%s", loop, suffix);
			output_fdesc = open(temp, O_RDWR | O_CREAT | O_EXCL, 0644);
			if (output_fdesc < 0)
			{
				if (errno != EEXIST)
					Elog("failed on open '%s': %m", temp);
			}
			else
			{
				output_filename = strdup(temp);
				if (!output_filename)
					Elog("out of memory");
				break;
			}
		}
		if (!output_filename)
			Elog("unable to generate new file name, please use -o FILENAME manually");
	}
	auto	rv = arrow::io::FileOutputStream::Open(output_fdesc);

	if (!rv.ok())
		Elog("failed on arrow::io::FileOutputStream::Open('%s'): %s",
			 output_filename,
			 rv.status().ToString().c_str());
	return rv.ValueOrDie();
}

/*
 * vcfSaveFilterMetaData
 */
static void
vcfSaveFilterMetaData(arrowMetadata custom_metadata, char *line)
{
	char   *tail = line + strlen(line) - 1;
	char   *tok, *pos;
	const char *ident = NULL;
	const char *descr = NULL;

	/* remove '<' and '>' */
	if (*line == '<' && *tail == '>')
	{
		line++;
		*tail = '\0';
	}
	/* fetch filter name and description */
	for (tok = __strtok_quotable(line, ',', &pos);
		 tok != NULL;
		 tok = __strtok_quotable(NULL, ',', &pos))
	{
		tok = __trim(tok);
		if (strncasecmp(tok, "ID=", 3) == 0)
			ident = __trim_quotable(tok+3);
		else if (strncasecmp(tok, "Description=", 12) == 0)
			descr = __trim_quotable(tok+12);
	}
	if (ident && descr)
	{
		custom_metadata->Append(std::string("filter.") +
								std::string(ident),
								std::string(descr));
	}
}

/*
 * addVcfFieldString
 */
static inline std::shared_ptr<class arrowFileWriterColumn>
makeVcfFieldString(const char *field_name)
{
	return std::make_shared<arrowFileWriterColumnUtf8>(field_name,
													   false,
													   default_compression);
}

/*
 * addVcfFieldInteger
 */
static inline std::shared_ptr<class arrowFileWriterColumn>
makeVcfFieldInteger(const char *field_name)
{
	return std::make_shared<arrowFileWriterColumnInt32>(field_name,
														false,
														default_compression);
}

/*
 * addVcfFieldFloat
 */
static inline std::shared_ptr<class arrowFileWriterColumn>
makeVcfFieldFloat(const char *field_name)
{
	return std::make_shared<arrowFileWriterColumnFloat32>(field_name,
														  false,
														  default_compression);
}

/*
 * addVcfFieldBoolean
 */
static inline std::shared_ptr<class arrowFileWriterColumn>
makeVcfFieldBoolean(const char *field_name)
{
	return std::make_shared<arrowFileWriterColumnBoolean>(field_name,
														  false,
														  default_compression);
}

/*
 * addVcfFieldByHeader
 */
static void
addVcfFieldByHeader(std::vector<std::shared_ptr<class arrowFileWriterColumn>> &arrow_columns,
					char *line)
{
	char   *tail = line + strlen(line) - 1;
	char   *tok, *pos;
	const char *ident = NULL;
	const char *number = NULL;
	const char *dtype = NULL;
	const char *descr = NULL;
	bool	has_allele_population = false;
	std::shared_ptr<arrowFileWriterColumn> column;

	if (*line == '<' && *tail == '>')
	{
		*tail = '\0';
		line++;
	}
	for (tok = __strtok_quotable(line, ',', &pos);
		 tok != NULL;
		 tok = __strtok_quotable(NULL, ',', &pos))
	{
		if (strncasecmp(tok, "ID=", 3) == 0)
			ident = __trim_quotable(tok+3);
		else if (strncasecmp(tok, "Number=", 7) == 0)
			number = __trim_quotable(tok+7);
		else if (strncasecmp(tok, "Type=", 5) == 0)
			dtype = __trim_quotable(tok+5);
		else if (strncasecmp(tok, "Description=", 12) == 0)
			descr = __trim_quotable(tok+12);
	}
	if (!ident || !number || !dtype)
		Elog("Unexpected field definition");
	if (strcmp(number, "A") == 0 || strcmp(number, "R") == 0)
		has_allele_population = true;
	if (strcmp(number, "0") == 0)
		column = makeVcfFieldBoolean(ident);
	else if ((strcmp(number, "1") == 0 ||
			  strcmp(number, "A") == 0) && strcmp(dtype, "Integer") == 0)
		column = makeVcfFieldInteger(ident);
	else if ((strcmp(number, "1") == 0 ||
			  strcmp(number, "A") == 0) && strcmp(dtype, "Float") == 0)
		column = makeVcfFieldFloat(ident);
	else
		column = makeVcfFieldString(ident);
	if (descr)
	{
		column->field_metadata->Append(std::string("description"),
									   std::string(descr));
	}
	arrow_columns.push_back(column);
}

/*
 * openVcfFileHeader
 */
static std::shared_ptr<arrowFileWriter>
openVcfFileHeader(FILE *filp)
{
	std::vector<std::shared_ptr<arrowFileWriterColumn>> format_columns;
	char	   *line;
	size_t		bufsz;
	ssize_t		nbytes;

	auto		af_builder = std::make_shared<arrowFileWriter>(parquet_mode,
															   default_compression);
	/* fixed fields */
	af_builder->arrow_columns.push_back(makeVcfFieldString("chrom"));
	af_builder->arrow_columns.push_back(makeVcfFieldString("pos"));
	af_builder->arrow_columns.push_back(makeVcfFieldString("id"));
	af_builder->arrow_columns.push_back(makeVcfFieldString("ref"));
	af_builder->arrow_columns.push_back(makeVcfFieldString("alt"));
	af_builder->arrow_columns.push_back(makeVcfFieldFloat("qual"));
	af_builder->arrow_columns.push_back(makeVcfFieldString("filter"));

	while ((nbytes = getline(&line, &bufsz, filp)) >= 0)
	{
		line = __trim(line);

		if (*line == '\0')
			continue;		/* empty line */
		else if (strncmp(line, "##fileformat=", 13) == 0)
			af_builder->table_metadata->Append(std::string("fileformat"),
											   std::string(line+13));
		else if (strncmp(line, "##reference=", 12) == 0)
			af_builder->table_metadata->Append(std::string("reference"),
											   std::string(line+12));
		else if (strncmp(line, "##FILTER=", 9) == 0)
			vcfSaveFilterMetaData(af_builder->table_metadata, __trim(line+9));
		else if (strncmp(line, "##INFO=", 7) == 0)
			addVcfFieldByHeader(af_builder->arrow_columns, __trim(line+7));
		else if (strncmp(line, "##FORMAT=", 9) == 0)
			addVcfFieldByHeader(format_columns, __trim(line+9));
		else if (strncmp(line, "#CHROM", 6) == 0 && isspace(line[6]))
			break;
	}
	return af_builder;
}

/*
 * usage
 */
static void usage(const char *format, ...)
{
	if (format)
	{
		va_list		va_args;

		va_start(va_args, format);
		vfprintf(stderr, format, va_args);
		va_end(va_args);
		fprintf(stderr, "\n\n");
	}
	fputs("vcf2arrow [OPTIONS] VCF_FILES ...\n"
		  "\n"
		  "OPTIONS:\n"
		  "  -f, --force            Force to write, if output file exists.\n"
		  "  -o, --output=OUTFILE   Output filename (default: auto)\n"
		  "  -q, --parquet          Enable Apache Parquer format\n"
		  "  -C, --compress=MODE    Specifies the default compression (default: snappy)\n"
		  "                         MODE := (snappy|gzip|brotli|zstd|lz4|lzo|bz2|none)\n"
		  "  -s, --segment-sz=SIZE  Size of RecordBatch/RowGroup (default: 1GB)\n"
		  "  -m, --user-metadata=KEY:VALUE Custom key-value pair to be embedded\n"
		  "  -e, --error-items=OUTFILE Filename to write out error items (default: stderr)\n"
		  "      --progress         Shows progress of VCF conversion.\n"
		  "      --schema           Print expected schema definition for the input files.\n"
		  "\n"
		  "SPECIALS:\n"
		  "\n"
		  "  -v, --verbose          Verbose output mode (for debugging)\n"
		  "  -h, --help             Print this message.\n", stderr);
	exit(1);

}

/*
 * parse_options
 */
static void
parse_options(int argc, char * const argv[])
{
	static struct option long_options[] = {
		{"force",         no_argument,       NULL, 'f'},
		{"output",        required_argument, NULL, 'o'},
		{"parquet",       no_argument,       NULL, 'q'},
		{"compress",      required_argument, NULL, 'C'},
		{"segment-sz",    required_argument, NULL, 's'},
		{"user-metadata", required_argument, NULL, 'm'},
		{"error-items",   optional_argument, NULL, 'e'},
		{"progress",      no_argument,       NULL, 1000},
		{"schema",        optional_argument, NULL, 1001},
		{"verbose",       no_argument,       NULL, 'v'},
		{"help",          no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	int		code;

	while ((code = getopt_long(argc, argv, "fo:qC:s:m:e::vh",
							   long_options, NULL)) >= 0)
	{
		switch (code)
		{
			case 'f':
				force_write_file = true;
				break;
			case 'o':
				output_filename = optarg;
				break;
			case 'q':
				parquet_mode = true;
				break;
			case 'C':
				if (strcmp(optarg, "snappy") == 0)
					default_compression = arrow::Compression::type::SNAPPY;
				else if (strcmp(optarg, "gzip") == 0)
					default_compression = arrow::Compression::type::GZIP;
				else if (strcmp(optarg, "brotli") == 0)
					default_compression = arrow::Compression::type::BROTLI;
				else if (strcmp(optarg, "zstd") == 0)
					default_compression = arrow::Compression::type::ZSTD;
				else if (strcmp(optarg, "lz4") == 0)
					default_compression = arrow::Compression::type::LZ4;
				else if (strcmp(optarg, "lzo") == 0)
					default_compression = arrow::Compression::type::LZO;
				else if (strcmp(optarg, "bz2") == 0)
					default_compression = arrow::Compression::type::BZ2;
				else if (strcmp(optarg, "none") == 0)
					default_compression = arrow::Compression::type::UNCOMPRESSED;
				else
					usage("unknown compression mode [%s]", optarg);
				break;
			case 's': {
				char   *end;
				long    nbytes = strtol(optarg, &end, 10);

				if (*end == '\0')
					batch_segment_sz = (size_t)nbytes;
				else if (strcasecmp(end, "k")  == 0 ||
						 strcasecmp(end, "kb") == 0)
					batch_segment_sz = (size_t)nbytes << 10;
				else if (strcasecmp(end, "m")  == 0 ||
						 strcasecmp(end, "mb") == 0)
					batch_segment_sz = (size_t)nbytes << 20;
				else if (strcasecmp(end, "g")  == 0 ||
						 strcasecmp(end, "gb") == 0)
					batch_segment_sz = (size_t)nbytes << 30;
				else if (strcasecmp(end, "t")  == 0 ||
						 strcasecmp(end, "tb") == 0)
					batch_segment_sz = (size_t)nbytes << 40;
				else
					usage("invalid -s, --segment-size [%s]", optarg);
				if (batch_segment_sz < (32UL<<20))
					usage("-s, --segment-size too small, at least 32MB");
				break;
			}
			case 'm': { /* user-metadata */
				if (strchr(optarg, '=') == NULL)
					usage("-m, --user-metadata must be KEY=VALUE form [%s]", optarg);
				user_custom_metadata.push_back(optarg);
				break;
			}
			case 'e':   /* --error-items */
				if (optarg)
					error_items_filename = optarg;
				else
				{
					error_items_filp = stderr;
					error_items_filename = "/dev/stderr";
				}
				break;
			case 1000:  /* --progress */
				shows_progress = true;
				break;
			case 1001:  /* --schema */
				if (optarg)
					schema_table_name = optarg;
				else
					schema_table_name = "VCF_FTABLE_NAME";
				break;
			case 'v':   /* --verbose */
				verbose = true;
				break;
			case 'h':   /* --help */
			default:
				usage(NULL);
				break;
		}
	}
	if (optind >= argc)
		usage("no input files given");
	for (int i=optind; i < argc; i++)
		input_filenames.push_back(argv[i]);
}

/*
 * main
 */
int main(int argc, char * const argv[])
{
	std::shared_ptr<arrow::io::FileOutputStream> arrow_out_stream;
	std::vector<FILE *> input_filp_array;
	std::shared_ptr<arrowFileWriter> arrow_builder = nullptr;

	try{
		parse_options(argc, argv);
		/* open VCF files and check schema definition */
		for (int i=0; i < input_filenames.size(); i++)
		{
			const char *fname = input_filenames[i];
			FILE	   *filp = fopen(fname, "rb");
			std::shared_ptr<arrowFileWriter> builder;

			if (!filp)
				Elog("unable to open input file '%s': %m", fname);
			builder = openVcfFileHeader(filp);
			if (!arrow_builder)
				arrow_builder = builder;
			else if (true)
				Elog("VCF file '%s' does not have compatible schema", fname);
			//open schema
			input_filp_array.push_back(filp);
		}
		arrow_out_stream = openArrowFileStream();

	//static const char  *output_filename = NULL;
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
