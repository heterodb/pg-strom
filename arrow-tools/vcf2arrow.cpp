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
 * __strdup
 */
#define STRDUP_BUFFER_SZ	(512UL<<10)		//512kB
static std::vector<char *>	strdup_buffers_array;
static char				   *strdup_buffers_curr = NULL;
static size_t				strdup_buffers_usage = 0;
static char *
__strdup(const char *str)
{
	char	   *copy = NULL;

	if (str)
	{
		size_t	sz = strlen(str) + 1;

		if (sz >= STRDUP_BUFFER_SZ / 4)
		{
			copy = strdup(str);
			if (!copy)
				Elog("out of memory");
			strdup_buffers_array.push_back(copy);
		}
		else if (!strdup_buffers_curr ||
				 strdup_buffers_usage + sz >= STRDUP_BUFFER_SZ)
		{
			if (strdup_buffers_curr)
				strdup_buffers_array.push_back(strdup_buffers_curr);
			strdup_buffers_curr = (char *)malloc(STRDUP_BUFFER_SZ);
			if (!strdup_buffers_curr)
				Elog("out of memory");
			copy = strdup_buffers_curr;
			strcpy(copy, str);
			strdup_buffers_usage = sz;
		}
		else
		{
			copy = strdup_buffers_curr + strdup_buffers_usage;
			strcpy(copy, str);
			strdup_buffers_usage += sz;
		}
	}
	return copy;
}

static void
__strdup_buffer_reset(void)
{
	for (int k=0; k < strdup_buffers_array.size(); k++)
		free(strdup_buffers_array[k]);
	strdup_buffers_array.clear();
	strdup_buffers_usage = 0;
}

/*
 * openArrowFileStream
 */
static void
openArrowFileStream(arrowFileBuilderTable af_builder)
{
	if (output_filename)
	{
		if (!af_builder->Open(output_filename, force_write_file))
			Elog("failed on arrowFileBuilderTable::Open('%s')", output_filename);
		return;
	}
	/* no -o FILENAME given, so generate a safe alternative */
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
		if (af_builder->Open(temp, false))
			return;
	}
	Elog("unable to generate output file name, please use -o FILENAME manually");
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
static inline arrowFileBuilderColumn
makeVcfFieldString(const char *field_name)
{
	return std::make_shared<ArrowFileBuilderColumnUtf8>(field_name,
														false,
														default_compression);
}

/*
 * addVcfFieldInteger
 */
static inline arrowFileBuilderColumn
makeVcfFieldInteger(const char *field_name)
{
	return std::make_shared<ArrowFileBuilderColumnInt32>(field_name,
														 false,
														 default_compression);
}

/*
 * addVcfFieldFloat
 */
static inline arrowFileBuilderColumn
makeVcfFieldFloat(const char *field_name)
{
	return std::make_shared<ArrowFileBuilderColumnFloat32>(field_name,
														  false,
														  default_compression);
}

/*
 * addVcfFieldBoolean
 */
static inline arrowFileBuilderColumn
makeVcfFieldBoolean(const char *field_name)
{
	return std::make_shared<ArrowFileBuilderColumnBoolean>(field_name,
														   false,
														   default_compression);
}

/*
 * addVcfFieldByHeader
 */
static void
addVcfInfoField(arrowFileBuilderTable &af_builder,
				const char *prefix, char *line)
{
	arrowFileBuilderColumn column;
	char	   *tail = line + strlen(line) - 1;
	char	   *tok, *pos;
	const char *ident = NULL;
	const char *number = NULL;
	const char *dtype = NULL;
	const char *descr = NULL;
	std::string	namebuf(prefix);

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
	namebuf += ident;
	if (strcmp(number, "0") == 0)
		column = makeVcfFieldBoolean(namebuf.c_str());
	else if ((strcmp(number, "1") == 0 ||
			  strcmp(number, "A") == 0) && strcmp(dtype, "Integer") == 0)
		column = makeVcfFieldInteger(namebuf.c_str());
	else if ((strcmp(number, "1") == 0 ||
			  strcmp(number, "A") == 0) && strcmp(dtype, "Float") == 0)
		column = makeVcfFieldFloat(namebuf.c_str());
	else
		column = makeVcfFieldString(namebuf.c_str());
	if (strcmp(number, "0") == 0 ||
		strcmp(number, "A") == 0 ||
		strcmp(number, "R") == 0)
        column->vcf_allele_policy = number[0];
	if (descr)
	{
		column->field_metadata->Append(std::string("description"),
									   std::string(descr));
	}
	af_builder->arrow_columns.push_back(column);
}

/*
 * parseLastVcfHeaderLine
 */
static void
parseLastVcfHeaderLine(arrowFileBuilderTable &af_builder,
					   std::vector<std::string> &format_lines,
					   const char *fname, char *line)
{
	const char *labels[] = {"#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"};
	char   *tok, *pos;
	int		loop;

	for (tok = strtok_r(line, " \t", &pos), loop=0;
		 tok != NULL;
		 tok = strtok_r(NULL, " \t", &pos), loop++)
	{
		/* skip fixed fields (even though FORMAT is optional) */
		if (loop > 8)
		{
			const char *sample_name = __trim(tok);
			char		temp[80];

			sprintf(temp, "sample%u__", loop-8);
			for (int i=0; i < format_lines.size(); i++)
			{
				std::string __line = format_lines[i];

				__line += '\0';		/* termination */
				addVcfInfoField(af_builder, temp, __line.data());
			}
			sprintf(temp, "sample%u_key", loop-8);
			af_builder->table_metadata->Append(std::string(temp),
											   std::string(sample_name));
		}
		else if (strcasecmp(tok, labels[loop]) != 0)
		{
			Elog("VCF file '%s' has unexpected header field [%s]", fname, tok);
		}
	}
}

/*
 * openVcfFileHeader
 */
static arrowFileBuilderTable
openVcfFileHeader(FILE *filp, const char *fname, long &line_no)
{
	std::vector<std::string> format_lines;
	char	   *line = NULL;
	size_t		bufsz = 0;
	ssize_t		nbytes;
	auto		af_builder = makeArrowFileBuilderTable(parquet_mode,
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
		line_no++;
		line = __trim(line);
		if (*line == '\0')
			continue;		/* empty line */
		else if (strncmp(line, "##fileformat=", 13) == 0)
		{
			std::string	label("fileformat");
			af_builder->table_metadata->Append(label, std::string(line+13));
		}
		else if (strncmp(line, "##reference=", 12) == 0)
		{
			std::string label("reference");
			af_builder->table_metadata->Append(label, std::string(line+12));
		}
		else if (strncmp(line, "##FILTER=", 9) == 0)
			vcfSaveFilterMetaData(af_builder->table_metadata, __trim(line+9));
		else if (strncmp(line, "##INFO=", 7) == 0)
			addVcfInfoField(af_builder, "info__", __trim(line+7));
		else if (strncmp(line, "##FORMAT=", 9) == 0)
			format_lines.push_back(std::string(__trim(line+9)));
		else if (strncmp(line, "#CHROM", 6) == 0 && isspace(line[6]))
		{
			parseLastVcfHeaderLine(af_builder, format_lines, fname, line);
			free(line);
			return af_builder;
		}
	}
	Elog("input file '%s' has no VCF header line", fname);
}

/*
 * __processVcfOneFieldItem
 */
static void
__processVcfOneFieldItem(arrowFileBuilderTable &af_builder,
						 std::vector<const char *> &vcf_tokens,
						 const char *field_prefix,
						 const char *field_name,
						 const char *field_tok,
						 int alt_loop)
{
	arrowFileBuilderColumn column;
	std::string key;

	if (field_prefix)
		key += field_prefix;
	key += field_name;
	column = af_builder->columns_htable.at(key);
	if (column->vcf_allele_policy == '0')
	{
		vcf_tokens[column->field_index] = "true";
		return;
	}
	else if (!field_tok)
	{
		Elog("Value of '%s' is missing for non-boolean field", key.c_str());
	}
	else if (column->vcf_allele_policy == 'A')
	{
		char   *temp = (char *)alloca(strlen(field_tok)+1);
		char   *tok, *pos;
		int		index;

		strcpy(temp, field_tok);
		for (tok = strtok_r(temp, ",", &pos), index=0;
			 tok != NULL;
			 tok = strtok_r(NULL, ",", &pos), index++)
		{
			if (index == alt_loop)
			{
				vcf_tokens[column->field_index] = __strdup(__trim(tok));
				return;
			}
		}
		Elog("corrupted 'A' item - %dth item is missing", alt_loop+1);
	}
	else if (column->vcf_allele_policy == 'R')
	{
		char   *temp = (char *)alloca(strlen(field_tok)+1);
		char   *tok, *pos;
		char   *first = NULL;
		int		index;

		strcpy(temp, field_tok);
		for (tok = strtok_r(temp, ",", &pos), index=-1;
			 tok != NULL;
			 tok = strtok_r(NULL, ",", &pos), index++)
		{
			if (index < 0)
				first = tok;
			else if (index == alt_loop)
			{
				char   *__temp = (char *)alloca(strlen(first) + strlen(tok) + 20);

				vcf_tokens[column->field_index] = __strdup(__temp);
				return;
			}
		}
		Elog("corrupted 'R' item - %dth item is missing", alt_loop+2);
	}
	else
	{
		vcf_tokens[column->field_index] = __strdup(field_tok);
	}
}

/*
 * __processVcfOneInfoItem
 */
static void
__processVcfOneInfoItem(arrowFileBuilderTable &af_builder,
						std::vector<const char *> &vcf_tokens,
						const char *__info,
						int alt_loop)
{
	char   *info = (char *)alloca(strlen(__info) + 1);
	char   *tok, *pos;

	strcpy(info, __info);
	for (tok = strtok_r(info, ";", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL, ";", &pos))
	{
		char   *val = strchr(tok, '=');

		if (val)
			*val++ = '\0';
		__processVcfOneFieldItem(af_builder,
								 vcf_tokens,
								 "info__",
								 __trim(tok),
								 __trim(val),
								 alt_loop);
	}
}

/*
 * __processVcfOneFormatItem
 */
static void
__processVcfOneFormatItem(arrowFileBuilderTable &af_builder,
						  std::vector<const char *> &vcf_tokens,
						  const char *prefix,
						  const char *__format,
						  const char *__sample,
						  int alt_loop)
{
	char   *format = (char *)alloca(strlen(__format) + 1);
	char   *sample = (char *)alloca(strlen(__sample) + 1);
	char   *tok1, *pos1;
	char   *tok2, *pos2;

	strcpy(format, __format);
	strcpy(sample, __sample);
	for (tok1 = strtok_r(format, ":", &pos1), tok2 = strtok_r(sample, ":", &pos2);
		 tok1 != NULL && tok2 != NULL;
		 tok1 = strtok_r(NULL,   ":", &pos1), tok2 = strtok_r(NULL,   ":", &pos2))
	{
		__processVcfOneFieldItem(af_builder,
								 vcf_tokens,
								 prefix,
								 __trim(tok1),
								 __trim(tok2),
								 alt_loop);
	}
	if (tok1 != NULL || tok2 != NULL)
		Elog("corrupted FORMAT - SAMPLE pairs");
}

/*
 * __processVcfFileOneVariant
 */
static void
__processVcfFileOneVariant(arrowFileBuilderTable &af_builder,
						   std::vector<const char *> &vcf_tokens,
						   char *vcf_chrom,
						   char *vcf_pos,
						   char *vcf_id,
						   char *vcf_ref,
						   char *vcf_alt,
						   int   alt_loop,
						   char *vcf_qual,
						   char *vcf_filter,
						   char *vcf_info,
						   char *vcf_format,
						   std::vector<char *> &vcf_samples)
{
	/* fixed fields */
	__processVcfOneFieldItem(af_builder, vcf_tokens, NULL, "chrom",  vcf_chrom,  alt_loop);
	__processVcfOneFieldItem(af_builder, vcf_tokens, NULL, "pos",    vcf_pos,    alt_loop);
	__processVcfOneFieldItem(af_builder, vcf_tokens, NULL, "id",     vcf_id,     alt_loop);
	__processVcfOneFieldItem(af_builder, vcf_tokens, NULL, "ref",    vcf_ref,    alt_loop);
	__processVcfOneFieldItem(af_builder, vcf_tokens, NULL, "alt",    vcf_alt,    alt_loop);
	__processVcfOneFieldItem(af_builder, vcf_tokens, NULL, "qual",   vcf_qual,   alt_loop);
	__processVcfOneFieldItem(af_builder, vcf_tokens, NULL, "filter", vcf_filter, alt_loop);
	/* INFO */
	__processVcfOneInfoItem(af_builder, vcf_tokens, vcf_info, alt_loop);
	/* FORMAT (optional) */
	if (vcf_format)
	{
		char	prefix[40];

		for (int k=0; k < vcf_samples.size(); k++)
		{
			sprintf(prefix, "sample%u__", k+1);

			__processVcfOneFormatItem(af_builder,
									  vcf_tokens,
									  prefix,
									  vcf_format,
									  vcf_samples[k],
									  alt_loop);
		}
	}
}

/*
 * __processVcfFileOneLine
 */
static void
__processVcfFileOneLine(arrowFileBuilderTable &af_builder,
						std::vector<std::vector<const char *>> &vcf_multi_tokens,
						char *line)
{
	int		ncols = af_builder->arrow_columns.size();
	char   *vcf_chrom = NULL;
	char   *vcf_pos = NULL;
	char   *vcf_id = NULL;
	char   *vcf_ref = NULL;
	char   *vcf_alt = NULL;
	char   *vcf_qual = NULL;
	char   *vcf_filter = NULL;
	char   *vcf_info = NULL;
	char   *vcf_format = NULL;
	std::vector<char *> vcf_samples;
	char   *tok, *pos;
	int		loop;

	for (tok = strtok_r(line, " \t", &pos), loop=0;
		 tok != NULL;
		 tok = strtok_r(NULL, " \t", &pos), loop++)
	{
		switch (loop)
		{
			case 0: vcf_chrom  = __trim(tok); break;
			case 1: vcf_pos    = __trim(tok); break;
			case 2: vcf_id     = __trim(tok); break;
			case 3: vcf_ref    = __trim(tok); break;
			case 4: vcf_alt    = __trim(tok); break;
			case 5: vcf_qual   = __trim(tok); break;
			case 6: vcf_filter = __trim(tok); break;
			case 7: vcf_info   = __trim(tok); break;
			case 8: vcf_format = __trim(tok); break;
			default:
				vcf_samples.push_back(__trim(tok));
				break;
		}
	}
	/* CHROME...INFO should be mandatory field */
	if (!vcf_info)
		Elog("mandatory fields are missing");
	/* Split VCF-ALT for each variant */
	for (tok = strtok_r(vcf_alt, ",", &pos), loop=0;
		 tok != NULL;
		 tok = strtok_r(NULL,    ",", &pos), loop++)
	{
		std::vector<const char *> vcf_tokens;

		vcf_tokens.resize(ncols);
		memset(vcf_tokens.data(), 0, sizeof(char *) * ncols);

		__processVcfFileOneVariant(af_builder,
								   vcf_tokens,
								   vcf_chrom,
								   vcf_pos,
								   vcf_id,
								   vcf_ref,
								   __trim(tok),	/* vcf_alt */
								   loop,
								   vcf_qual,
								   vcf_filter,
								   vcf_info,
								   vcf_format,
								   vcf_samples);
		vcf_multi_tokens.push_back(vcf_tokens);
	}
}

/*
 * processVcfFileBody
 */
static void
processVcfFileBody(arrowFileBuilderTable &af_builder,
				   FILE *filp,
				   const char *fname,
				   long line_no)
{
	size_t		line_sz = 0;
	char	   *line = NULL;
	size_t		work_sz = 0;
	char	   *work = NULL;
	ssize_t		nbytes;
	std::vector<std::vector<const char *>> vcf_multi_tokens;

	while ((nbytes = getline(&line, &line_sz, filp)) >= 0)
	{
		bool	skip_this_line = false;

		/*
		 * Once copy the VCF line to working buffer
		 * (if line is not convertible, the original should be reported)
		 */
		if (line_sz >= work_sz)
		{
			work_sz = line_sz * 2 + 8000;
			if (!work)
				work = (char *)malloc(work_sz+1);
			else
				work = (char *)realloc(work, work_sz+1);
			if (!work)
				Elog("out of memory");
		}
		strcpy(work, line);
		line_no++;

		/* Parse VCF line */
		try {
			__processVcfFileOneLine(af_builder, vcf_multi_tokens, work);
		}
		catch (std::exception &e)
		{
			if (!error_items_filp)
			{
				if (!error_items_filename)
				{
					fprintf(stderr, "Not convertible item at FILE=%s LINE=%ld: %s\n%s",
						 fname, line_no, e.what(), line);
					_exit(1);
				}
				error_items_filp = fopen(error_items_filename, "ab");
				if (!error_items_filp)
					Elog("unable to open '%s' for error reporting", error_items_filename);
			}
			fprintf(error_items_filp, "###UNCONVERTIBLE FILE=%s LINE=%ld MESSAGE=%s\n%s",
					fname, line_no, e.what(), line);
			skip_this_line = true;
		}
		/* Append One Line */
		if (!skip_this_line)
		{
			int		ncols = af_builder->arrow_columns.size();

			for (int k=0; k < vcf_multi_tokens.size(); k++)
			{
				auto	vcf_tokens = vcf_multi_tokens[k];
				size_t	length = 0;

				assert(vcf_tokens.size() == ncols);
				for (int j=0; j < ncols; j++)
				{
					auto	column = af_builder->arrow_columns[j];
					const char *token = vcf_tokens[j];

					if (token)
						length += column->putValue(token, strlen(token));
					else if (column->vcf_allele_policy == '0')
						length += column->putValue("false", 5);
					else
						length += column->putValue(NULL, 0);
				}
				if (length >= batch_segment_sz)
					af_builder->WriteChunk();
			}
		}
		/* Reset working buffer */
		vcf_multi_tokens.clear();
		__strdup_buffer_reset();
	}
}

/*
 * printSchemaDefinition
 */
static int
printSchemaDefinition(arrowFileBuilderTable af_builder)
{
	std::string	sql = af_builder->PrintSchema(schema_table_name,
											  output_filename);
	for (int j=0; j < af_builder->arrow_columns.size(); j++)
	{
		auto	column = af_builder->arrow_columns[j];

		for (int k=0; k < column->field_metadata->size(); k++)
		{
			auto	_key = column->field_metadata->key(k);
			auto	_value = column->field_metadata->value(k);

			if (_key == std::string("description"))
			{
				sql += (std::string("COMMENT ON ") +
						__quote_ident(schema_table_name) +
						std::string(".") +
						__quote_ident(column->field_name.c_str()) +
						std::string(" IS '") +
						_value +
						std::string("';\n"));
				break;
			}
		}
	}
	std::cout << sql << std::endl;
	return 0;
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
	std::vector<long>	input_line_no_array;
	arrowFileBuilderTable af_builder = nullptr;

	try{
		parse_options(argc, argv);
		/* open VCF files and check schema definition */
		for (int i=0; i < input_filenames.size(); i++)
		{
			const char *fname = input_filenames[i];
			FILE	   *filp = fopen(fname, "rb");
			long		line_no = 0;
			arrowFileBuilderTable __builder;

			if (!filp)
				Elog("unable to open input file '%s': %m", fname);
			__builder = openVcfFileHeader(filp, fname, line_no);
			if (!af_builder)
				af_builder = __builder;
			else if (!af_builder->checkCompatibility(__builder))
				Elog("VCF file '%s' does not have compatible schema", fname);
			input_filp_array.push_back(filp);
			input_line_no_array.push_back(line_no);
		}
		assert(af_builder != nullptr);
		/* assign arrow::Schema */
		af_builder->AssignSchema();
		/* print table definition if --schema is given */
		if (schema_table_name)
			return printSchemaDefinition(af_builder);
		openArrowFileStream(af_builder);
		/* process VCF files for each */
		for (int i=0; i < input_filenames.size(); i++)
		{
			const char *fname = input_filenames[i];
			FILE	   *filp = input_filp_array[i];
			long		line_no = input_line_no_array[i];

			processVcfFileBody(af_builder, filp, fname, line_no);
			fclose(filp);
		}
		/* finalize */
		af_builder->Close();
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
