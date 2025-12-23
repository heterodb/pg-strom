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
#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>
#include "unistd.h"
#include "arrow_write.h"














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
		  "  -C, --compress=MODE    Specifies the compression mode [Parquet only]\n"
		  "                         MODE := (snappy|gzip|brotli|zstd|lz4|lzo|bz2)\n"
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

}

/*
 * main
 */
int main(int argc, char * const argv[])
{
	return 0;
}
