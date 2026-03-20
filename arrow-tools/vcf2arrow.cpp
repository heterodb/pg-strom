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
#include <assert.h>
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
static bool			enable_numeric_conversion = true;
static bool			unconvertible_numeric_as_null = false;

/*
 * line-buffer - a lightweight string buffer
 */
static std::vector<char *>	line_buffers_array;
static char				   *line_buffer_curr = NULL;
static size_t				line_buffer_usage = 0;
static size_t				line_buffer_over_size = 0;
static size_t				line_buffer_alloc_size = (512UL<<10);	/* 512kB */
static char *
linebuf_strdup(const char *str)
{
	char	   *buf = NULL;
	if (str)
	{
		size_t	sz = strlen(str) + 1;

		if (line_buffer_curr &&
			line_buffer_usage + sz <= line_buffer_alloc_size)
		{
			buf = line_buffer_curr + line_buffer_usage;
			line_buffer_usage += sz;
		}
		else if (sz > line_buffer_alloc_size / 2)
		{
			buf = (char *)malloc(sz);
			if (!buf)
				Elog("out of memory");
			line_buffers_array.push_back(buf);
			line_buffer_over_size += sz;
		}
		else
		{
			buf = (char *)malloc(line_buffer_alloc_size);
			if (!buf)
				Elog("out of memory: %ld", line_buffer_alloc_size);
			if (line_buffer_curr)
			{
				line_buffers_array.push_back(line_buffer_curr);
				line_buffer_over_size += line_buffer_alloc_size;
			}
			line_buffer_curr = buf;
			line_buffer_usage = sz;
		}
		strcpy(buf, str);
	}
	return buf;
}
static void
linebuf_reset(void)
{
	for (int k=0; k < line_buffers_array.size(); k++)
		free(line_buffers_array[k]);
	line_buffers_array.clear();
	line_buffer_usage = 0;
	if (line_buffer_over_size > 0)
	{
		size_t	unitsz_512kb = ((512UL<<10) - 1);

		line_buffer_alloc_size += (line_buffer_over_size + unitsz_512kb) & ~unitsz_512kb;
		if (line_buffer_curr)
			free(line_buffer_curr);
		line_buffer_curr = NULL;
	}
	line_buffer_over_size = 0;
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
	const char *suffix = (!parquet_mode ? "arrow" : "parquet");

	if (len >= 4 && strcasecmp(input_fname+len-4, ".vcf") == 0)
		len -= 4;
	for (int loop=0; loop < 100; loop++)
	{
		std::string	fname(input_fname, len);

		if (loop > 0)
			fname += std::string(".") + std::to_string(loop);
		fname += std::string(suffix);
		if (af_builder->Open(fname.c_str(), false))
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
	else if (enable_numeric_conversion &&
			 ((strcmp(number, "1") == 0 ||
			   strcmp(number, "A") == 0) && strcmp(dtype, "Integer") == 0))
		column = makeVcfFieldInteger(namebuf.c_str());
	else if (enable_numeric_conversion &&
			 ((strcmp(number, "1") == 0 ||
			   strcmp(number, "A") == 0) && strcmp(dtype, "Float") == 0))
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
	/*
	 * check duplicated field name
	 */
	for (int j=0; j < af_builder->arrow_columns.size(); j++)
	{
		auto    __column = af_builder->arrow_columns[j];
		if (__column->field_name == column->field_name)
		{
			if (__column->arrow_type->Equals(column->arrow_type) &&
				__column->vcf_allele_policy == column->vcf_allele_policy)
			{
				Info("Duplicated field '%s' found, but compatible (type='%s' policy='%c'); second one is ignored.",
					 column->field_name.c_str(),
					 column->arrow_type->ToString().c_str(),
					 column->vcf_allele_policy);
				return;
			}
			Elog("Duplicated field '%s' found, and they are mutually inconsistent: the first has '%s' and policy='%c', but the second has '%s' and policy='%c'",
				 __column->field_name.c_str(),
				 __column->arrow_type->ToString().c_str(),
				 __column->vcf_allele_policy,
				 column->arrow_type->ToString().c_str(),
				 column->vcf_allele_policy);
		}
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
	af_builder->arrow_columns.push_back(makeVcfFieldInteger("pos"));
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
 * __validateFieldItem
 */
static const char *
__validateFieldItem(arrowFileBuilderColumn column, const char *tok)
{
	switch (column->arrow_type->id())
	{
		char   *end;

		case arrow::Type::INT32: {
			long    ival = strtol(tok, &end, 10);
			if (*end != '\0')
			{
				if (unconvertible_numeric_as_null)
					return NULL;
				Elog("Integer token [%s] is not convertible at field '%s'",
					 tok, column->field_name.c_str());
			}
			if (ival < INT_MIN || ival > INT_MAX)
			{
				if (unconvertible_numeric_as_null)
					return NULL;
				Elog("Integer token [%s] is out of range at field '%s'",
					 tok, column->field_name.c_str());
			}
			break;
		}
		case arrow::Type::FLOAT: {
			double  fval = strtod(tok, &end);
			if (*end != '\0')
			{
				if (unconvertible_numeric_as_null)
					return NULL;
				Elog("Float token [%s] is not convertible at field '%s'",
					 tok, column->field_name.c_str());
			}
			if (fval == 0.0 && end == tok)
			{
				if (unconvertible_numeric_as_null)
					return NULL;
				Elog("no conversion is performed on Float token['%s'] at field '%s'",
					 tok, column->field_name.c_str());
			}
			if (errno == ERANGE)
			{
				if (unconvertible_numeric_as_null)
					return NULL;
				Elog("Float token [%s] is out of range at field '%s'",
					 tok, column->field_name.c_str());
			}
			break;
		}
		default:
			assert(column->arrow_type->id() == arrow::Type::STRING);
			break;
	}
	return __trim(linebuf_strdup(tok));
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
		char   *temp = linebuf_strdup(field_tok);
		char   *tok, *pos;
		int		index;

		for (tok = strtok_r(temp, ",", &pos), index=0;
			 tok != NULL;
			 tok = strtok_r(NULL, ",", &pos), index++)
		{
			if (index == alt_loop)
			{
				vcf_tokens[column->field_index] = __validateFieldItem(column, tok);
				return;
			}
		}
		Elog("corrupted 'A' item - %dth item is missing", alt_loop+1);
	}
	else if (column->vcf_allele_policy == 'R')
	{
		char   *temp = linebuf_strdup(field_tok);
		char   *tok, *pos;
		char   *first = NULL;
		int		index;

		for (tok = strtok_r(temp, ",", &pos), index=-1;
			 tok != NULL;
			 tok = strtok_r(NULL, ",", &pos), index++)
		{
			if (index < 0)
				first = tok;
			else if (index == alt_loop)
			{
				char   *comb = (char *)alloca(strlen(first) + strlen(tok) + 2);
				sprintf(comb, "%s,%s", first, tok);
				vcf_tokens[column->field_index] = __validateFieldItem(column, comb);
				return;
			}
		}
		Elog("corrupted 'R' item - %dth item is missing", alt_loop+2);
	}
	else
	{
		vcf_tokens[column->field_index] = __validateFieldItem(column, field_tok);
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
	char   *info = linebuf_strdup(__info);
	char   *tok, *pos;

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
	char   *format = linebuf_strdup(__format);
	char   *sample = linebuf_strdup(__sample);
	char   *tok1, *pos1;
	char   *tok2, *pos2;

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
	ssize_t		nbytes;
	std::vector<std::vector<const char *>> vcf_multi_tokens;

	while ((nbytes = getline(&line, &line_sz, filp)) >= 0)
	{
		bool	skip_this_line = false;
		/* Parse VCF line */
		line_no++;
		try {
			char   *work = linebuf_strdup(line);
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
					af_builder->WriteChunk(shows_progress ? stdout : NULL);
			}
		}
		/* Reset working buffer */
		vcf_multi_tokens.clear();
		linebuf_reset();
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
		  "      --disable-numaric-conversion  Don't convert Number=Integer/Float items to\n"
		  "                                    numeric values in arrow/parquet.\n"
		  "      --unconvertible-numeric-as-null  Unconvertible numeric values are considered\n"
		  "                                       as NULL value, not a reportable error.\n"
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
		{"disable-numaric-conversion", no_argument, NULL, 2001},
		{"unconvertible-numeric-as-null", no_argument, NULL, 2002},
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
				if (!optarg)
					error_items_filp = stderr;
				else
				{
					error_items_filename = optarg;
					error_items_filp = NULL;
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
			case 2001:	/* --disable-numaric-conversion */
				enable_numeric_conversion = false;
				break;
			case 2002:	/* --unconvertible-numeric-as-null */
				unconvertible_numeric_as_null = true;
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
		af_builder->Close(shows_progress ? stdout : NULL);
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
