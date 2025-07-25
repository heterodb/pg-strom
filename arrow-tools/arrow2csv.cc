/*
 * arrow2csv.cc
 *
 * A tool to dump Apache Arrow/Parquet file as CSV/TSV format
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
//#include <iostream>
#include <arrow/api.h>		/* dnf install libarrow-devel */
//#include <list>
//#include <typeinfo>

#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>

typedef std::string				cppString;
typedef std::vector<cppString>	cppStringVec;



static const char	   *output_filename = NULL;
static bool				shows_header = false;
static bool				csv_mode = true;
static long				skip_offset = 0;
static long				skip_limit = -1;
static const char	   *with_create_table = NULL;
static const char	   *with_tablespace = NULL;
static const char	   *with_partition_of = NULL;
static cppStringVec		input_filenames;
static int				verbose = 0;

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
					__FILE__,__LINE__, ##__VA_ARGS__);	\
	} while(0)

static void __usage(const char *fmt,...)
{
	if (fmt)
	{
		 va_list ap;

		va_start(ap, fmt);
		vfprintf(stderr, fmt, ap);
		va_end(ap);
		fputc('\n', stderr);
	}
	fputs("usage: arrow2csv [OPTIONS] <file1> [<file2> ...]\n"
		  "\n"
		  "-o|--output=FILENAME	specify the output filename\n"
		  "                     (default: stdout)\n"
		  "   --tsv             dump in TSV mode\n"
		  "   --csv             dump in CSV mode\n"
		  "   --header          dump column names as csv header\n"
		  "   --offset=NUM      skip the first NUM rows\n"
		  "   --limit=NUM       dump only NUM rows\n"
		  "\n"
		  "   --create-table=TABLE_NAME   dump with CREATE TABLE statement\n"
		  "   --tablespace=TABLESPACE     specify tablespace of the table, if any\n"
		  "   --partition-of=PARENT_NAME  specify partition-parent of the table, if any\n"
		  "\n"
		  "-v|--verbose          verbose output\n"
		  "-h|--help             print this message\n"
		  "\n"
		  "Report bugs to <pgstrom@heterodb.com>\n",
		  stderr);
}
#define usage()		__usage(NULL)




static void parse_options(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"output",  required_argument, NULL, 'o'},
		{"tsv",     no_argument,       NULL, 1001},
		{"csv",     no_argument,       NULL, 1002},
		{"header",  required_argument, NULL, 1003},
		{"offset",  required_argument, NULL, 1004},
		{"limit",   required_argument, NULL, 1005},
		{"create-table", required_argument, NULL, 1006},
		{"tablespace",   required_argument, NULL, 1007},
		{"partition-of", required_argument, NULL, 1008},
		{"verbose", no_argument,       NULL, 'v'},
		{"help",    no_argument,       NULL, 'h'},
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
					Elog("-o|--output was given twice");
				output_filename = optarg;
				break;
			case 1001:	/* --tsv */
				csv_mode = false;
				break;
			case 1002:	/* --csv */
				csv_mode = true;
				break;
			case 1003:	/* --header */
				shows_header = true;
				break;
			case 1004:	/* --offset */
				skip_offset = strtol(optarg, &end, 10);
				if (*end != '\0' || skip_offset < 0)
					Elog("invalid --offset value '%s'", optarg);
				break;
			case 1005:	/* --limit */
				skip_limit = strtol(optarg, &end, 10);
				if (*end != '\0' || skip_offset < 0)
					Elog("invalid --limit value '%s'", optarg);
				break;
			case 1006:	/* --create-table */
				if (with_create_table)
					Elog("--create-table was given twice");
				with_create_table = optarg;
				break;
			case 1007:	/* --tablespace */
				if (with_tablespace)
					Elog("--tablespace was given twice");
				with_tablespace = optarg;
				break;
			case 1008:	/* --partition-of */
				if (with_partition_of)
					Elog("--partition-of was given twice");
				with_partition_of = optarg;
				break;
			case 'v':	/* --verbose */
				verbose++;
				break;
			default:	/* --help */
				usage();
		}
	}
	for (int k=optind; k < argc; k++)
		input_filenames.push_back(argv[k]);
}

int main(int argc, char *argv[])
{
	parse_options(argc, argv);











	return 0;
}
