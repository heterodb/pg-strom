/*
 * tsv2arrow
 *
 * A tool to transform TSV (tab-separated values) stream to Apache Arrow
 * format.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <iostream>
#include <list>
#include <arrow/api.h>
#include <ctype.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef std::string									cppString;
typedef std::list<cppString>						cppStringList;
typedef std::vector<cppString>						cppStringVector;
typedef std::shared_ptr<arrow::Schema>				arrowSchema;
typedef std::shared_ptr<arrow::Field>				arrowField;
typedef std::shared_ptr<arrow::DataType>			arrowDataType;
typedef std::shared_ptr<arrow::KeyValueMetadata>	arrowKeyValueMetadata;

static arrowSchema		arrow_schema = NULL;
static const char	   *output_filename = NULL;
static size_t			segment_sz = (256UL << 20);		/* 256MB in default */
static bool				skip_header = false;
static long				skip_offset = 0;
static long				skip_limit = -1;
static bool				shows_progress = false;
static int				verbose = 0;
static cppStringList	tsv_input_files;

#define Elog(fmt,...)									\
	do {												\
		fprintf(stderr, "[ERROR %s:%d] " fmt "\n",		\
				__FILE__,__LINE__, ##__VA_ARGS__);		\
		exit(1);										\
	} while(0)
#define Info(fmt,...)									\
	do {												\
		if (verbose > 0)								\
			fprintf(stderr, "[INFO %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
	} while(0)

#define Debug(fmt,...)									\
	do {												\
		if (verbose > 1)								\
			fprintf(stderr, "[DEBUG %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);  \
	} while(0)
	





static void __usage(const char *fmt,...)
{
	va_list	ap;

	if (fmt)
	{
		va_start(ap, fmt);
		vfprintf(stderr, fmt, ap);
		va_end(ap);
		fputc('\n', stderr);
	}
	fputs("usage: tsv2arrow [OPTIONS] <file1> [<file2> ...]\n"
		  "\n"
		  "OPTIONS:\n"
		  "  -o|--output=FILENAME     output filename\n"
		  "  -S|--schema=SCHEMA       schema definition\n"
		  "  -s|--segment-sz=SIZE     unit size of record batch\n"
		  "     --skip-header         skip header line\n"
		  "     --offset=NUM          skip first NUM rows\n"
		  "     --limit=NUM           transform only NUM rows\n"
		  "  -m|--metadata=KEY:VALUE  embeds the key-value pair\n"
		  "     --progress            shows progress\n"
		  "  -v|--verbose             verbose output\n"
		  "  -h|--help                print this messagen\n"
		  "\n"
		  "The schema definition syntax is as follows:\n"
		  "  SCHEMA := <FIELD1>[,<FIELD2>,...]\n"
		  "  FIELD := <NAME>:<TYPE>[:<EXTRA1>[;<EXTRA2>...]]\n"
		  "  TYPE  := intX        signed X-bit integer [X=8,16,<32>,64]\n"
		  "           uintX       unsigned X-bit integer[X=8,16,<32>,64]\n"
		  "           floatX      floating point[X=2,4,<8>]\n"
		  "           decimal     decimal\n"
		  "           date        date\n"
		  "           time        time\n"
		  "           timestamp   timestamp\n"
		  "           utf8        variable length text\n"
		  "           binary      variable length binary\n"
		  "  EXTRA := not_null    field cannot contain NULLs\n"
		  "           nullable    field can contain NULLs (default)\n"
		  "           KEY=VALUE   field level custom metadata\n"
		  "\n"
		  "Report bugs to <pgstrom@heterodb.com>.\n",
		  stderr);
	exit(1);
}
#define usage()		__usage(NULL)



static arrowField
parse_field_definition(const char *f_name,
					   const char *t_name,
					   char *extra)
{
	cppString		field_name = f_name;
	arrowDataType	field_type;
	arrowKeyValueMetadata field_meta = NULLPTR;
	bool			nullable = true;

	/* parse extra attributes */
	if (extra)
	{
		cppStringVector	meta_keys;
		cppStringVector	meta_values;
		char	   *tok, *val, *pos;

		for (tok = strtok_r(extra, ";", &pos);
			 tok != NULL;
			 tok = strtok_r(NULL, ";", &pos))
		{
			if (strcasecmp(tok, "not_null") == 0)
				nullable = false;
			else if (strcasecmp(tok, "nullable") == 0)
				nullable = true;
			else if ((val = strchr(tok, '=')) != NULL)
			{
				std::string	_key;
				std::string	_value;

				*val++ = '\0';
				_key = tok;
				_value = val;
				meta_keys.push_back(_key);
				meta_values.push_back(_value);
			}
			else
			{
				Elog("unknown field extra option: %s", tok);
			}
		}

		if (!meta_keys.empty() && !meta_values.empty())
		{
			assert(meta_keys.size() == meta_values.size());
			field_meta = std::make_shared<arrow::KeyValueMetadata>(meta_keys,
																   meta_values);
		}
	}
	/* for each data types */
	if (strcasecmp(t_name, "int8") == 0)
		field_type = arrow::int8();
	else if (strcasecmp(t_name, "int16") == 0)
		field_type = arrow::int16();
	else if (strcasecmp(t_name, "int") == 0 ||
			 strcasecmp(t_name, "int32") == 0)
		field_type = arrow::int32();
	else if (strcasecmp(t_name, "int64") == 0)
		field_type = arrow::int64();
	else if (strcasecmp(t_name, "uint8") == 0)
		field_type = arrow::uint8();
	else if (strcasecmp(t_name, "uint16") == 0)
		field_type = arrow::uint16();
	else if (strcasecmp(t_name, "uint") == 0 ||
			 strcasecmp(t_name, "uint32") == 0)
		field_type = arrow::uint32();
	else if (strcasecmp(t_name, "uint64") == 0)
		field_type = arrow::uint64();
	else if (strcasecmp(t_name, "float2") == 0)
		field_type = arrow::float16();
	else if (strcasecmp(t_name, "float4") == 0)
		field_type = arrow::float32();
	else if (strcasecmp(t_name, "float") == 0 ||
			 strcasecmp(t_name, "float8") == 0)
		field_type = arrow::float64();
	else if (strcasecmp(t_name, "decimal") == 0 ||
			 strcasecmp(t_name, "decimal128") == 0)
		field_type = arrow::decimal128(27, 6);
	else if (strcasecmp(t_name, "decimal256") == 0)
		field_type = arrow::decimal256(36,9);
	else if (strcasecmp(t_name, "date") == 0 ||
			 strcasecmp(t_name, "date-sec") == 0)
		field_type = arrow::date32();
	else if (strcasecmp(t_name, "date-ms") == 0)
		field_type = arrow::date64();
	else if (strcasecmp(t_name, "time-sec") == 0)
		field_type = arrow::time32(arrow::TimeUnit::SECOND);
	else if (strcasecmp(t_name, "time") == 0 ||
			 strcasecmp(t_name, "time-ms") == 0)
		field_type = arrow::time32(arrow::TimeUnit::MILLI);
	else if (strcasecmp(t_name, "time-us") == 0)
		field_type = arrow::time64(arrow::TimeUnit::MICRO);
	else if (strcasecmp(t_name, "time-ns") == 0)
		field_type = arrow::time64(arrow::TimeUnit::NANO);
	else if (strcasecmp(t_name, "timestamp-sec") == 0)
		field_type = arrow::timestamp(arrow::TimeUnit::SECOND);
	else if (strcasecmp(t_name, "timestamp-ms") == 0)
		field_type = arrow::timestamp(arrow::TimeUnit::MILLI);
	else if (strcasecmp(t_name, "timestamp") == 0 ||
			 strcasecmp(t_name, "timestamp-us") == 0)
		field_type = arrow::timestamp(arrow::TimeUnit::MICRO);
	else if (strcasecmp(t_name, "timestamp-ns") == 0)
		field_type = arrow::timestamp(arrow::TimeUnit::NANO);
	else if (strcasecmp(t_name, "utf8") == 0)
		field_type = arrow::utf8();
	else if (strcasecmp(t_name, "utf8-large") == 0)
		field_type = arrow::large_utf8();
	else if (strcasecmp(t_name, "binary") == 0)
		field_type = arrow::binary();
	else if (strcasecmp(t_name, "binary-large") == 0)
		field_type = arrow::large_binary();
	else
		Elog("unknown field type: %s", t_name);

	return arrow::field(field_name,
						field_type,
						nullable,
						field_meta);
}

static void
parse_options(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"output",      required_argument, NULL, 'o'},
		{"schema",      required_argument, NULL, 'S'},
		{"segment-sz",  required_argument, NULL, 's'},
		{"skip-header", no_argument,       NULL, 1000},
		{"offset",      required_argument, NULL, 1001},
		{"limit",       required_argument, NULL, 1002},
		{"metadata",    required_argument, NULL, 'm'},
		{"progress",    no_argument,       NULL, 1003},
		{"verbose",     no_argument,       NULL, 'v'},
		{"help",        no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	arrowKeyValueMetadata metadata = NULLPTR;
	std::vector<std::string> meta_keys;
	std::vector<std::string> meta_values;
	char   *schema_definition = NULL;
	int		c;

	while ((c = getopt_long(argc, argv,
							"o:S:s:m:vh",
							long_options, NULL)) >= 0)
	{
		char	   *end, *sep;

		switch (c)
		{
			case 'o':	/* --output */
				if (output_filename)
					Elog("-o|--output was given twice");
				output_filename = optarg;
				break;
			case 'S':	/* --schema */
				if (schema_definition)
					Elog("-S|--schema was given twice");
				schema_definition = optarg;
				break;
			case 's':	/* --segment-sz */
				segment_sz = strtol(optarg, &end, 10);
				if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0)
					segment_sz <<= 10;
				else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0)
					segment_sz <<= 20;
				else if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0)
					segment_sz <<= 30;
				else if (strcasecmp(end, "t") == 0 || strcasecmp(end, "tb") == 0)
					segment_sz <<= 40;
				else if (*end != '\0')
					Elog("invalid segment size '%s'", optarg);
				if (segment_sz < (32UL << 20))
					Elog("too small segment size (at least 32MB)");
				break;
			case 1000:	/* --skip-header */
				skip_header = true;
				break;
			case 1001:	/* --offset */
				skip_offset = strtol(optarg, &end, 10);
				if (*end != '\0' || skip_offset < 0)
					Elog("invalid offset number '%s'", optarg);
				break;
			case 1002:	/* --limit */
				skip_limit = strtol(optarg, &end, 10);
				if (*end != '\0' || skip_limit < 0)
					Elog("invalid limit number '%s'", optarg);
				break;
			case 'm':	/* --metadata */
				sep = strchr(optarg, '=');
				if (!sep)
					Elog("invalid custom-metadata '%s' (must be KEY=VALUE)", optarg);
				else
				{
					std::string _key;
					std::string _value;

					*sep++ = '\0';
					_key = optarg;
					_value = sep;
					meta_keys.push_back(_key);
					meta_values.push_back(_value);
				}
				break;
			case 1003:	/* --progress */
				shows_progress = true;
				break;
			case 'v':	/* --verbose */
				verbose++;
				break;
			case 'h':	/* --help */
				usage();
			default:
				Elog("unknown option '%c'", c);
		}
	}
	if (!meta_keys.empty() && !meta_values.empty())
	{
		assert(meta_keys.size() == meta_values.size());
		metadata = std::make_shared<arrow::KeyValueMetadata>(meta_keys,
															 meta_values);
	}

	if (!schema_definition)
		__usage("-S|--schema must be given with schema definition");
	else
	{
		char   *temp = (char *)alloca(strlen(schema_definition) + 1);
		char   *f_name, *pos;
		arrow::FieldVector fields_vec;

		strcpy(temp, schema_definition);
		for (f_name = strtok_r(temp, ",", &pos);
			 f_name != NULL;
			 f_name = strtok_r(NULL, ",", &pos))
		{
			arrowField field;
			char   *t_name = strchr(f_name, ':');
			char   *extra;

			if (!t_name)
				Elog("schema definition has no type [%s]", f_name);
			*t_name++ = '\0';
			extra = strchr(t_name, ':');
			if (extra)
				*extra++ = '\0';
			field = parse_field_definition(f_name, t_name, extra);
			fields_vec.push_back(field);
		}
		arrow_schema = schema(fields_vec, metadata);
	}
}

int main(int argc, char *argv[])
{
	parse_options(argc, argv);

	std::cout << "Schema:\n" << arrow_schema->ToString(true) << "\n";

	return 0;
}
