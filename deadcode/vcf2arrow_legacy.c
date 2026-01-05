/*
 * vcf2arrow.c
 *
 * VCF format to Apache Arrow converter
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include "arrow_ipc.h"

static char		   *output_filename = NULL;
static const char **input_filename = NULL;
static FILE		  **input_filedesc = NULL;
static long		   *input_fileline = NULL;
static int			input_file_nitems = 0;
static size_t		batch_segment_sz = (240UL << 20);
static int			embedded_headers_nitems = 0;
static const char **embedded_headers = NULL;
static int			user_metadata_nitems = 0;
static const char **user_metadata_keys = NULL;
static const char **user_metadata_values = NULL;
static SQLtable	   *vcf_table = NULL;
static bool			force_write_if_exists = false;
static bool			verbose_mode = false;
static bool			by_raw_format = false;
static bool			sort_by_pos = false;
static bool			shows_progress = false;

static ArrowKeyValue *customMetadata = NULL;
static int			customMetadata_nrooms = 0;
static int			customMetadata_nitems = 0;
static const char ***sampleNames = NULL;
static int			sampleNames_nitems = 0;
static int			sampleNames_nrooms = 0;
static struct StringInfoData *sampleNamesMetadata = NULL;

#define Max(a,b)	((a) > (b) ? (a) : (b))
#define Min(a,b)	((a) < (b) ? (a) : (b))
#define Info(fmt,...)													\
	fprintf(stderr, "%s:%d [info] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define VCF_ARROW_ANUM__CHROME		0
#define VCF_ARROW_ANUM__POS			1
#define VCF_ARROW_ANUM__ID			2
#define VCF_ARROW_ANUM__REF			3
#define VCF_ARROW_ANUM__ALT			4
#define VCF_ARROW_ANUM__QUAL		5
#define VCF_ARROW_ANUM__FILTER		6
#define VCF_ARROW_ANUM__INFO		7
#define VCF_ARROW_ANUM__FORMAT		8	/* only if --with-raw-format */
#define VCF_ARROW_NUM_BASE_FIELDS	8

static struct {
	ArrowTypeTag	type;
	const char	   *name;
	int				anum;
} vcf_arrow_column_defs[] = {
	{ArrowType__Utf8, "chrom",	VCF_ARROW_ANUM__CHROME},
	{ArrowType__Int,  "pos",	VCF_ARROW_ANUM__POS},
	{ArrowType__Utf8, "id",		VCF_ARROW_ANUM__ID},
	{ArrowType__Utf8, "ref",	VCF_ARROW_ANUM__REF},
	{ArrowType__Utf8, "alt",	VCF_ARROW_ANUM__ALT},
	{ArrowType__Utf8, "qual",	VCF_ARROW_ANUM__QUAL},
	{ArrowType__Utf8, "filter",	VCF_ARROW_ANUM__FILTER},
	{ArrowType__Utf8, "info",	VCF_ARROW_ANUM__INFO},
	{ArrowType__Utf8, "format",	VCF_ARROW_ANUM__FORMAT},
	{-1,NULL},
};

/*
 * __trim
 */
static inline char *__trim(char *token)
{
    char   *tail = token + strlen(token) - 1;

	while (isspace(*token))
		token++;
	while (tail >= token && isspace(*token))
		*tail-- = '\0';
	return token;
}

/*
 * StringInfo
 */
typedef struct StringInfoData
{
	char   *data;
	int		len;
	int		maxlen;
} StringInfoData;

typedef StringInfoData *StringInfo;

static void
resetStringInfo(StringInfo str)
{
	str->data[0] = '\0';
	str->len = 0;
}

static void
initStringInfo(StringInfo str)
{
	int		sz = 1024;

	str->data = (char *) palloc(sz);
	str->maxlen = sz;
	resetStringInfo(str);
}

static void
appendStringInfo(StringInfo str, const char *fmt,...)
{
	va_list		va_args;
	int			nbytes;

	va_start(va_args, fmt);
	for (;;)
	{
		nbytes = vsnprintf(str->data + str->len,
						   str->maxlen - str->len,
						   fmt, va_args);
		if (str->len + nbytes < str->maxlen)
		{
			str->len += nbytes;
			break;
		}
		str->maxlen += str->maxlen;
		str->data = repalloc(str->data, str->maxlen);
	}
	va_end(va_args);
}

/*
 * __strtok - quote and backslash aware strtok(3)
 */
static char *__strtok(char *str, const char *delim, char **saveptr)
{
	char   *r_pos = (str ? str : *saveptr);
	char   *w_pos = r_pos;
	char   *start = w_pos;
	bool	quote_in_token = false;
	int		phase = 0;
	int		c, d;

	assert(saveptr != NULL);
	if (!r_pos)
		return NULL;		/* already touched to the end */
	for (;;)
	{
		c = *r_pos++;
		switch (phase)
		{
			case 0:			/* before any valid token */
				if (c == '\0')
				{
					*saveptr = NULL;
					return NULL;
				}
				else if (c == '"')
				{
					phase = 2;	/* inside of the quoted string */
				}
				else if (c == '\\')
				{
					if (*r_pos == '\0')
					{
						*saveptr = NULL;
						return NULL;
					}
					*w_pos++ = *r_pos++;
					phase = 1;	/* inside of the valid token */
				}
				else
				{
					for (int i=0; (d=delim[i]) != '\0'; i++)
					{
						if (d == c)
							goto next;
					}
					quote_in_token = false;
					*w_pos++ = c;
					phase = 1;	/* inside of the valid token */
				}
			next:
				break;

			case 1:		/* inside of the valid token */
				if (c == '\0')
				{
					*w_pos++ = '\0';
					*saveptr = NULL;
					return start;
				}
				else if (c == '\\')
				{
					if (*r_pos == '\0')
					{
						*saveptr = NULL;
						*w_pos++ = '\0';
						return start;
					}
					*w_pos++ = *r_pos++;
				}
				else if (c == '"')
				{
					quote_in_token = true;
					*w_pos++ = c;
					phase = 2;
				}
				else
				{
					for (int i=0; (d=delim[i]) != '\0'; i++)
					{
						if (d == c)
						{
							*saveptr = r_pos;
							*w_pos++ = '\0';
							return start;
						}
					}
					*w_pos++ = c;
				}
				break;

			case 2:		/* inside of the double quotation */
				if (c == '\0')
				{
					*w_pos++ = '\0';
					*saveptr = NULL;
					return start;
				}
				else if (c == '"')
				{
					if (*r_pos == '"')
					{
						/* escape character */
						*w_pos++ = *r_pos++;
					}
					else
					{
						if (quote_in_token)
							*w_pos++ = c;
						phase = 1;	/* inside of the unquoted valid token */
					}
				}
				else if (c == '\\')
				{
					if (*r_pos == '\0')
					{
						*w_pos++ = '\0';
						*saveptr = NULL;
						return start;
					}
					/* escape character */
					*w_pos++ = *r_pos++;
				}
				else
				{
					*w_pos++ = c;
				}
				break;
			default:
				/* should not happen */
				return NULL;
		}
	}
}

/*
 * preprocess_vcf_file
 */
#define VCF_WHITESPACE		" \t\r\n"

static void
__preprocess_vcf_header(const char *fname, char *key, char *value,
						StringInfo buf, bool *headers_matched)
{
	bool	found = false;
	char   *end;

	/* ignore the header field, if not embedded */
	for (int k=0; k < embedded_headers_nitems; k++)
	{
		if (strcasecmp(embedded_headers[k], key) == 0)
		{
			found = true;
			break;
		}
	}
	if (!found)
		return;
	/*
	 * Try to convert to JSON string, if packed key-value pairs
	 */
	resetStringInfo(buf);
	end = value + strlen(value) - 1;
	if (*value != '<' || *end != '>')
	{
		appendStringInfo(buf, "%s", value);
	}
	else
	{
		char   *tok, *pos;
		bool	is_first;

		*end = '\0';
		appendStringInfo(buf, "{");
		for (tok = __strtok(value + 1, ",", &pos), is_first = true;
			 tok != NULL;
			 tok = __strtok(NULL, ",", &pos), is_first = false)
		{
			char   *__key, *__value, *__pos;

			__key = __strtok(tok, "=", &__pos);
			if (!__key)
				continue;	/* empty string */
			__value = __strtok(NULL, "=", &__pos);
			appendStringInfo(buf, (is_first ? "" : ","));
			if (__value)
				appendStringInfo(buf, "\"%s\" : \"%s\"", __key, __value);
			else
				appendStringInfo(buf, "\"%s\" : null", __key);
		}
		appendStringInfo(buf, "}");
	}

	/*
	 * first path -> build customMetadata[] array
	 * second or later -> check customMetadata[] array
	 */
	if (!headers_matched)
	{
		ArrowKeyValue  *kv;

		if (customMetadata_nitems == customMetadata_nrooms)
		{
			customMetadata_nrooms += (customMetadata_nrooms + 20);
			customMetadata = repalloc(customMetadata,
									  sizeof(ArrowKeyValue) * customMetadata_nrooms);
		}
		kv = &customMetadata[customMetadata_nitems++];
		initArrowNode(kv, KeyValue);
		kv->key = pstrdup(key);
		kv->value = pstrdup(buf->data);
		kv->_key_len = strlen(kv->key);
		kv->_value_len = strlen(kv->value);
	}
	else
	{
		for (int i=user_metadata_nitems; i < customMetadata_nitems; i++)
		{
			ArrowKeyValue *kv = &customMetadata[i];

			if (strcasecmp(key, kv->key) == 0 &&
				strcasecmp(buf->data, kv->value) == 0)
			{
				headers_matched[i] = true;
				return;
			}
		}
		Elog("VCF file '%s' has incompatible header line: KEY='%s' VALUE='%s'",
			 fname, key, buf->data);
	}
}

static void
__preprocess_vcf_fields(int file_index, const char *fname, char **saveptr)
{
	/* !!CHROM and POS are already checked by the caller!! */
	char   *id	   = __strtok(NULL, VCF_WHITESPACE, saveptr);
	char   *ref	   = __strtok(NULL, VCF_WHITESPACE, saveptr);
	char   *alt	   = __strtok(NULL, VCF_WHITESPACE, saveptr);
	char   *qual   = __strtok(NULL, VCF_WHITESPACE, saveptr);
	char   *filter = __strtok(NULL, VCF_WHITESPACE, saveptr);
	char   *info   = __strtok(NULL, VCF_WHITESPACE, saveptr);
	char   *format = __strtok(NULL, VCF_WHITESPACE, saveptr);
	char   *tok;
	int		j = 0;

	if (!id     || strcasecmp(id,     "ID") != 0 ||
		!ref    || strcasecmp(ref,    "REF") != 0 ||
		!alt    || strcasecmp(alt,    "ALT") != 0 ||
		!qual   || strcasecmp(qual,   "QUAL") != 0 ||
		!filter || strcasecmp(filter, "FILTER") != 0 ||
		!info   || strcasecmp(info,   "INFO") != 0)
	{
		Elog("VCF file '%s' corrupted? mandatory fields missing", fname);
	}
	if (format)
	{
		if (strcasecmp(format, "FORMAT") != 0)
			Elog("VCF file '%s' corrupted? not an expected field [%s]", fname, format);

		while ((tok = __strtok(NULL, VCF_WHITESPACE, saveptr)) != NULL)
		{
			if (j == sampleNames_nrooms)
			{
				int		new_nrooms = 2 * sampleNames_nrooms + 16;

				for (int k=0; k < input_file_nitems; k++)
				{
					sampleNames[k] = (const char **)
						repalloc(sampleNames[k],
								 sizeof(const char **) * new_nrooms);
					for (int i=sampleNames_nrooms; i < new_nrooms; i++)
						sampleNames[k][i] = NULL;
				}
				sampleNames_nrooms = new_nrooms;
			}
			sampleNames[file_index][j++] = pstrdup(tok);
		}
		sampleNames_nitems = Max(sampleNames_nitems, j);
	}
	else if (by_raw_format)
		Elog("VCF file '%s' has no FORMAT field in spite of --raw-format option", fname);
		
}

static void
preprocess_vcf_file(int file_index,
					StringInfo buf,
					bool *headers_matched)
{
	const char *fname = input_filename[file_index];
	FILE	   *filp = input_filedesc[file_index];
	long	   *p_lineno = &input_fileline[file_index];
	char	   *line = NULL;
	size_t		bufsz = 0;
	ssize_t		nbytes;

	while ((nbytes = getline(&line, &bufsz, filp)) > 0)
	{
		char   *tok, *second, *pos;

		(*p_lineno)++;

		tok = __strtok(line, VCF_WHITESPACE, &pos);
		if (!tok)
			continue;	/* empty line */
		second = __strtok(NULL, VCF_WHITESPACE, &pos);
		if (!second)
		{
			char   *sep = strchr(tok+2, '=');
			/* likely, VCF header line */
			if (tok[0] != '#' || tok[1] != '#' || !sep)
				Elog("VCF file '%s' has unexpected header line at %ld:\n%s\n",
					 fname, *p_lineno, tok);
			*sep++ = '\0';
			__preprocess_vcf_header(fname, tok+2, sep,
									buf, headers_matched);
		}
		else if (strcasecmp(tok, "#CHROM") == 0 &&
				 strcasecmp(second, "POS") == 0)
		{
			__preprocess_vcf_fields(file_index, fname, &pos);
			break;
		}
	}

	if (headers_matched)
	{
		for (int i=0; i < customMetadata_nitems; i++)
		{
			if (!headers_matched[i])
				Elog("VCF file '%s' misses a header line: KEY='%s' VALUE='%s'\n",
					 fname,
					 customMetadata[i].key,
					 customMetadata[i].value);
		}
	}
	if (line)
		free(line);
}

/*
 * SQLtable/SQLfield handlers
 */
static inline void
__put_inline_null_value(SQLfield *column, size_t row_index, int sz)
{
	column->nullcount++;
	sql_buffer_clrbit(&column->nullmap, row_index);
	sql_buffer_append_zero(&column->values, sz);
}

static inline void
__stats_update_int64(SQLfield *column, int64_t value)
{
	if (!column->stat_datum.is_valid)
	{
		column->stat_datum.min.i64 = value;
		column->stat_datum.max.i64 = value;
		column->stat_datum.is_valid = true;
	}
	else
	{
		if (column->stat_datum.min.i64 > value)
			column->stat_datum.min.i64 = value;
		if (column->stat_datum.max.i64 < value)
			column->stat_datum.max.i64 = value;
	}
}

static size_t
vcf_put_int64_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int64_t		value;
	char	   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		value = strtol(addr, &end, 10);
		if (*end != '\0')
			Elog("wrong integer literal [%s]", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));

		__stats_update_int64(column, value);
    }
    return __buffer_usage_inline_type(column);
}

static size_t
vcf_move_int64_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	size_t	dindex = dest->nitems++;

	if (!sql_buffer_getbit(&src->nullmap, sindex))
		__put_inline_null_value(dest, dindex, sizeof(int64_t));
	else
	{
		int64_t		value;

		value = ((int64_t *)src->values.data)[sindex];
		sql_buffer_setbit(&dest->nullmap, dindex);
		sql_buffer_append(&dest->values, &value, sizeof(int64_t));

		__stats_update_int64(dest, value);
	}
	return __buffer_usage_inline_type(dest);
}

static int
vcf_write_int64_stat(SQLfield *attr, char *buf, size_t len,
					 const SQLstat__datum *datum)
{
	return snprintf(buf, len, "%ld", datum->i64);
}

static size_t
vcf_put_variable_value(SQLfield *column,
					   const char *addr, int sz)
{
	size_t	row_index = column->nitems++;

	if (row_index == 0)
		sql_buffer_append_zero(&column->values, sizeof(uint32_t));
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->extra, addr, strlen(addr));
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	return __buffer_usage_varlena_type(column);
}

static size_t
vcf_move_variable_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	const char *addr = NULL;
	int		sz = 0;

	if (sql_buffer_getbit(&src->nullmap, sindex))
	{
		uint32_t	head = ((uint32_t *)src->values.data)[sindex];
		uint32_t	tail = ((uint32_t *)src->values.data)[sindex+1];

		assert(head <= tail && tail <= src->extra.usage);
		if (tail - head >= INT_MAX)
			Elog("too large variable data (len: %u)", tail - head);
		addr = src->extra.data + head;
		sz   = tail - head;
	}
	return vcf_put_variable_value(dest, addr, sz);
}

/*
 * __setup_vcf_column_int_buffer
 */
static int
__setup_vcf_column_int_buffer(SQLtable *table,
							  SQLfield *column,
							  const char *field_name)
{
	column->field_name = pstrdup(field_name);
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.is_signed = true;
	column->arrow_type.Int.bitWidth = 64;
	column->put_value = vcf_put_int64_value;
	column->move_value = vcf_move_int64_value;
	column->write_stat = vcf_write_int64_stat;
	column->stat_enabled = sort_by_pos;
	return 2;	/* nullmap + values */
}

/*
 * __setup_vcf_column_utf8_buffer
 */
static int
__setup_vcf_column_utf8_buffer(SQLtable *table,
							   SQLfield *column,
							   const char *field_name)
{
	column->field_name = pstrdup(field_name);
	initArrowNode(&column->arrow_type, Utf8);
	column->put_value = vcf_put_variable_value;
	column->move_value = vcf_move_variable_value;
	column->write_stat = NULL;
	column->stat_enabled = false;
	return 3;	/* nullmap + index + extra */
}

/*
 * __build_vcf_table_buffer
 */
static SQLtable *
__build_vcf_table_buffer(void)
{
	SQLtable   *table;
	int			nfields;

	/*
	 * For the multi-variable VCF files, we have secondary (or more) buffer
	 * after the vcf_table->columns[10], and switch them when we write out
	 * buffered values to Apache Arrow.
	 */
	nfields = VCF_ARROW_NUM_BASE_FIELDS;
	if (by_raw_format)
		nfields++;
	table = palloc0(offsetof(SQLtable, columns[nfields + sampleNames_nitems]));
	for (int j=0; j < nfields; j++)
	{
		const char	   *arrow_name = vcf_arrow_column_defs[j].name;
		ArrowTypeTag	arrow_type = vcf_arrow_column_defs[j].type;

		switch (arrow_type)
		{
			case ArrowType__Int:
				table->numBuffers +=
					__setup_vcf_column_int_buffer(table,
												  &table->columns[j],
												  arrow_name);
				break;
			case ArrowType__Utf8:
				table->numBuffers +=
					__setup_vcf_column_utf8_buffer(table,
												   &table->columns[j],
												   arrow_name);
				break;
			default:
				Elog("Bug? unexpected Arrow type tag for VCF conversion");
		}
	}
	/* extend the "sampleXX" fields */
	for (int j=0; j < sampleNames_nitems; j++)
	{
		SQLfield   *column = &table->columns[nfields+j];
		char		namebuf[80];

		snprintf(namebuf, sizeof(namebuf), "sample%u", j+1);
		table->numBuffers +=
			__setup_vcf_column_utf8_buffer(table, column, namebuf);
	}
	table->nfields = nfields + sampleNames_nitems;
	table->numFieldNodes = nfields + sampleNames_nitems;
	/* custom-metadata */
	table->customMetadata    = customMetadata;
	table->numCustomMetadata = customMetadata_nitems;
	/* setup metadata-buffer for sample-names */
	sampleNamesMetadata = palloc0(sizeof(StringInfoData) * sampleNames_nitems);
	for (int j=0; j < sampleNames_nitems; j++)
		initStringInfo(&sampleNamesMetadata[j]);

	return table;
}

/*
 * setup_vcf_table_buffer
 */
static void setup_vcf_table_buffer(void)
{
	StringInfoData buf;
	bool	   *headers_matched = NULL;

	initStringInfo(&buf);
	sampleNames = palloc0(sizeof(const char **) * input_file_nitems);
	for (int i=0; i < input_file_nitems; i++)
	{
		if (headers_matched)
			memset(headers_matched, 0, sizeof(bool) * customMetadata_nitems + 1);
		preprocess_vcf_file(i, &buf, headers_matched);
		/* metadata validation map */
		if (i == 0)
			headers_matched = alloca(sizeof(bool) * customMetadata_nitems + 1);
	}
	vcf_table = __build_vcf_table_buffer();

	pfree(buf.data);
	assert(vcf_table != NULL);

	/* open the output file */
	if (output_filename)
	{
		int		flags = O_RDWR | O_CREAT | O_TRUNC;
		int		fdesc;

		if (!force_write_if_exists)
			flags |= O_EXCL;
		fdesc = open(output_filename, flags, 0644);
		if (fdesc < 0)
			Elog("failed on open('%s'): %m", output_filename);
		vcf_table->fdesc = fdesc;
		vcf_table->filename = output_filename;
	}
	else
	{
		char    temp[200];
		int		fdesc;

		strcpy(temp, "/tmp/XXXXXX.arrow");
		fdesc = mkostemps(temp, 6, O_RDWR | O_CREAT | O_TRUNC);
		if (fdesc < 0)
			Elog("failed on mkostemps('%s'): %m", temp);
		vcf_table->fdesc = fdesc;
		vcf_table->filename = pstrdup(temp);
		fprintf(stderr,
				"NOTICE: -o, --output=FILENAME option was not given,\n"
				"        so a temporary file '%s' was built instead.\n", temp);
    }
	/* write out header stuff */
	arrowFileWrite(vcf_table, "ARROW1\0\0", 8);
	writeArrowSchema(vcf_table);
}

/*
 * __flush_vcf_arrow_file
 */
static void
__flush_vcf_arrow_file(SQLtable *table, int file_index)
{
	ArrowBlock	__block;
	int			__rb_index;

	if (table->nitems > 0)
	{
		__rb_index = writeArrowRecordBatch(table, &__block);
		for (int j=0; j < sampleNames_nitems; j++)
		{
			StringInfo	str = &sampleNamesMetadata[j];
			const char *name = sampleNames[file_index][j];

			if (__rb_index > 0)
				appendStringInfo(str, ",");
			appendStringInfo(str, "%s", name);
		}
		if (shows_progress)
		{
			time_t		tv = time(NULL);
			struct tm	tm;

			localtime_r(&tv, &tm);
			printf("%04d-%02d-%02d %02d:%02d:%02d "
				   "RecordBatch[%d]: "
				   "offset=%lu length=%lu (meta=%u, body=%lu) nitems=%zu\n",
				   tm.tm_year + 1900,
				   tm.tm_mon + 1,
				   tm.tm_mday,
				   tm.tm_hour,
				   tm.tm_min,
				   tm.tm_sec,
				   __rb_index,
				   __block.offset,
				   __block.metaDataLength + __block.bodyLength,
				   __block.metaDataLength,
				   __block.bodyLength,
				   table->nitems);
		}
	}
	/* clear the buffer */
	sql_table_clear(vcf_table);
}

/*
 * __convert_vcf_variant
 */
static size_t
__convert_vcf_variant(SQLfield *column, const char *__format, char *variant)
{
	char	   *format = NULL;
	char	   *buffer = NULL;
	char	   *tok1, *pos1;
	char	   *tok2, *pos2;
	size_t		off = 0;

	if (__format && variant)
	{
		buffer = alloca(strlen(__format) + strlen(variant) + 512);
		format = alloca(strlen(__format) + 100);
		strcpy(format, __format);

		for (tok1 = strtok_r(format, ":", &pos1),
			 tok2 = strtok_r(variant, ":", &pos2);
			 tok1 != NULL && tok2 != NULL;
			 tok1 = strtok_r(NULL, ":", &pos1),
			 tok2 = strtok_r(NULL, ":", &pos2))
		{
			if (off > 0)
				buffer[off++] = ':';
			off += sprintf(buffer+off, "%s=%s", tok1, tok2);
		}
		buffer[off++] = '\0';
	}
	return sql_field_put_value(column, buffer, -1);
}

/*
 * convert_vcf_file
 */
static void
convert_vcf_file(int file_index)
{
//	const char *fname = input_filename[file_index];
	FILE	   *filp  = input_filedesc[file_index];
	long	   *p_lineno = &input_fileline[file_index];
	char	   *line = NULL;
	size_t		bufsz = 0;
	ssize_t		nbytes;

	while ((nbytes = getline(&line, &bufsz, filp)) > 0)
	{
		char	   *tok, *pos;
		const char *format = NULL;;
		bool		meet_format = false;
		int			anum, j=0;
		size_t		usage = 0;

		(*p_lineno)++;
		for (tok = __strtok(line, VCF_WHITESPACE, &pos), anum=0;
			 j < vcf_table->nfields;
			 tok = __strtok(NULL, VCF_WHITESPACE, &pos), anum++)
		{
			if (anum < VCF_ARROW_NUM_BASE_FIELDS)
			{
				usage += sql_field_put_value(&vcf_table->columns[j++], tok, -1);
			}
			else if (by_raw_format)
			{
				usage += sql_field_put_value(&vcf_table->columns[j++], tok, -1);
			}
			else if (!meet_format)
			{
				format = tok;
				meet_format = true;
			}
			else
			{
				usage += __convert_vcf_variant(&vcf_table->columns[j++], format, tok);
			}
		}
		vcf_table->nitems++;
		if (usage >= batch_segment_sz)
			__flush_vcf_arrow_file(vcf_table, file_index);
	}
	__flush_vcf_arrow_file(vcf_table, file_index);
	if (line)
		free(line);
}

static void
attach_vcf_table_metadata(void)
{
	int		nfields = VCF_ARROW_NUM_BASE_FIELDS;
	int		nr_meta = vcf_table->numCustomMetadata;

	if (by_raw_format)
		nfields++;
	assert(vcf_table->nfields == nfields + sampleNames_nitems);
	vcf_table->customMetadata = (ArrowKeyValue *)
		repalloc(vcf_table->customMetadata,
				 sizeof(ArrowKeyValue) * (nr_meta + sampleNames_nitems));
	for (int j=0; j < sampleNames_nitems; j++)
	{
		StringInfo	buf = &sampleNamesMetadata[j];
		char		namebuf[80];
		ArrowKeyValue *kv;

		kv = &vcf_table->customMetadata[nr_meta + j];
		snprintf(namebuf, sizeof(namebuf), "sample%u_keys", j+1);
		initArrowNode(kv, KeyValue);
		kv->key = pstrdup(namebuf);
		kv->value = buf->data;
		kv->_key_len = strlen(kv->key);
		kv->_value_len = buf->len;
	}
	vcf_table->numCustomMetadata += sampleNames_nitems;
}

/*
 * usage
 */
static void usage(const char *format, ...)
{
	if (format)
	{
		va_list		va_args;

		fprintf(stderr, "[error] ");
		va_start(va_args, format);
		vfprintf(stderr, format, va_args);
		va_end(va_args);
		fprintf(stderr, "\n\n");
	}
	fputs("vcf2arrow [OPTIONS] VCF_FILES ...\n"
		  "\n"
		  "OPTIONS:\n"
		  "  -f|--force           : force to write if output file exists\n"
		  "  -o|--output ARROW_FILE : output filename (default: auto)\n"
		  "  -s|--segment-sz SIZE : size of record batch (default: 240MB)\n"
		  "  -E|--embedded-headers=HEADERS : comma separated header names list to be embedded.\n"
		  "                        (default: 'fileformat,reference,info,filter,format')\n"
		  "  -m|--user-metadata=KEY=VALUE : a custom key-value pair to be embedded\n"
		  "     --raw-format      : saves format and variant columns in raw string.\n"
		  "                        (*) in default, it transformed to KEY=VALUE form.\n"
//		  "     --sort-by-pos     : sort by the POS (optimization for min/max stats)\n"
//		  "                        (*) note that this option preload entire VCF file once.\n"
		  "     --progress        : shows progress of VCF conversion.\n"
		  "  -h|--help            : print this message.\n"
		  "  -v|--verbose         : verbose output mode (for software debug)\n",
		  stderr);
	exit(1);
}

/*
 * parse_options
 */
static void parse_options(int argc, char * const argv[])
{
	static struct option long_options[] = {
		{"force",            no_argument,       NULL, 'f'},
		{"output",           required_argument, NULL, 'o'},
		{"segment-sz",       required_argument, NULL, 's'},
		{"embedded-headers", required_argument, NULL, 'E'},
		{"user-metadata",    required_argument, NULL, 'm'},
		{"raw-format",       no_argument,       NULL, 1001},
//		{"sort-by-pos",      no_argument,       NULL, 1002},
		{"progress",         no_argument,       NULL, 1003},
		{"help",             no_argument,       NULL, 'h'},
		{"verbose",          no_argument,       NULL, 'v'},
		{NULL, 0, NULL, 0},
	};
	int		user_metadata_nrooms = 0;
	int		c;

	while ((c = getopt_long(argc, argv, "fo:s:E:m:hv",
							long_options, NULL)) >= 0)
	{
		switch (c)
		{
			case 'f':
				force_write_if_exists = true;
				break;
			case 'o':
				if (output_filename)
					usage("-o|--output option supplied twice.");
				output_filename = optarg;
				break;
			case 's':
				{
					char	   *end;
					long		ival;

					ival = strtol(optarg, &end, 10);
					if (ival < 0)
						usage("segment size [%s] is not a valid string", optarg);
					else if (*end != '\0')
					{
						if (strcasecmp("k", end) == 0 ||
							strcasecmp("kb", end) == 0)
							ival <<= 10;
						else if (strcasecmp("m", end) == 0 ||
								 strcasecmp("mb", end) == 0)
							ival <<= 20;
						else if (strcasecmp("g", end) == 0 ||
								 strcasecmp("gb", end) == 0)
							ival <<= 30;
						else if (strcasecmp("t", end) == 0 ||
								 strcasecmp("tb", end) == 0)
							ival <<= 40;
						else
							Elog("unknown segment size [%s]", optarg);
					}
					if (ival <= (32UL<<20))
						usage("segment size %.1fkB is too small", (double)ival / 1024.0);
					batch_segment_sz = ival;
				}
				break;
			case 'E':	/* --embedded-headers */
				if (embedded_headers)
					usage("-E|--embedded-headers option supplied twice");
				else
				{
					char   *temp = alloca(strlen(optarg) + 1);
					char   *tok, *pos;
					int		__nitems = 0;
					int		__nrooms = 0;

					strcpy(temp, optarg);
					for (tok = strtok_r(temp, ",", &pos);
						 tok != NULL;
						 tok = strtok_r(NULL, ",", &pos))
					{
						if (__nitems == __nrooms)
						{
							__nrooms = (__nrooms + 40);
							embedded_headers = repalloc(embedded_headers,
														sizeof(char *) * __nrooms);
						}
						embedded_headers[__nitems++] = pstrdup(__trim(tok));
					}
					if (__nitems == 0)
						usage("-E|--embedded-headers had empty string [%s]", optarg);
					embedded_headers_nitems = __nitems;
				}
				break;
			case 'm':	/* --user-metadata */
				{
					char   *temp = alloca(strlen(optarg) + 1);
					char   *pos;

					strcpy(temp, optarg);
					pos = strchr(temp, '=');
					if (!pos)
						usage("--user-metadata must take KEY=VALUE [%s]", optarg);
					*pos++ = '\0';

					if (user_metadata_nitems == user_metadata_nrooms)
					{
						user_metadata_nrooms += (user_metadata_nrooms + 40);
						user_metadata_keys = repalloc(user_metadata_keys,
												sizeof(char *) * user_metadata_nrooms);
						user_metadata_values = repalloc(user_metadata_values,
												sizeof(char *) * user_metadata_nrooms);
					}
					user_metadata_keys[user_metadata_nitems] = pstrdup(__trim(temp));
					user_metadata_values[user_metadata_nitems] = pstrdup(__trim(pos));
					user_metadata_nitems++;
				}
				break;
			case 1001:	/* --raw-format */
				by_raw_format = true;
				break;
			case 1002:	/* --sort-by-pos */
				sort_by_pos = true;
				break;
			case 1003:	/* --progress */
				shows_progress = true;
				break;
			case 'v':
				verbose_mode = true;
				break;
			default:
				usage(NULL);
				break;
		}
	}
	/* input filename */
	if (argc == optind)
		usage("No input VCF files supplied.");
	input_file_nitems = (argc - optind);
	input_filename = palloc0(sizeof(char *) * input_file_nitems);
	input_filedesc = palloc0(sizeof(FILE *) * input_file_nitems);
	input_fileline = palloc0(sizeof(long)   * input_file_nitems);
	for (int k=0; k < input_file_nitems; k++)
	{
		const char *filename = argv[optind + k];
		FILE	   *filp;

		filp = fopen(filename, "rb");
		if (!filp)
			Elog("unable to open source VCF file: '%s'", filename);
		input_filename[k] = filename;
		input_filedesc[k] = filp;
		input_fileline[k] = 0;
	}

	/* default embedded headers */
	if (!embedded_headers)
	{
		embedded_headers = palloc0(sizeof(char *) * 6);
		embedded_headers[0] = "fileformat";
		embedded_headers[1] = "reference";
		embedded_headers[2] = "info";
		embedded_headers[3] = "filter";
		embedded_headers[4] = "format";
		embedded_headers_nitems = 5;
	}

	/* build initial CustomMetadata */
	if (user_metadata_nitems > 0)
	{
		customMetadata_nrooms = user_metadata_nitems + 24;
		customMetadata = palloc0(sizeof(ArrowKeyValue) * customMetadata_nrooms);
		for (int i=0; i < user_metadata_nitems; i++)
		{
			ArrowKeyValue *kv = &customMetadata[i];
			initArrowNode(kv, KeyValue);
			kv->key = user_metadata_keys[i];
			kv->value = user_metadata_values[i];
			kv->_key_len = strlen(kv->key);
			kv->_value_len = strlen(kv->value);
		}
		customMetadata_nitems = user_metadata_nitems;
	}
}

/*
 * main
 */
int main(int argc, char * const argv[])
{
	parse_options(argc, argv);
	setup_vcf_table_buffer();
	for (int i=0; i < input_file_nitems; i++)
		convert_vcf_file(i);
	attach_vcf_table_metadata();
	writeArrowFooter(vcf_table);
	return 0;
}

/*
 * misc functions
 */
void *palloc(size_t sz)
{
	void   *ptr = malloc(sz);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void *palloc0(size_t sz)
{
	void   *ptr = malloc(sz);

	if (!ptr)
		Elog("out of memory");
	memset(ptr, 0, sz);
	return ptr;
}

char *pstrdup(const char *str)
{
	char   *ptr = strdup(str);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void *repalloc(void *old, size_t sz)
{
	char   *ptr = realloc(old, sz);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void pfree(void *ptr)
{
	free(ptr);
}
