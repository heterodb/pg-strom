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
#include "arrow-tools.h"
#include <iostream>
#include <list>
#include <typeinfo>
#include <ctype.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "float2.h"

using namespace arrow;

typedef std::shared_ptr<Schema>				arrowSchema;
typedef std::shared_ptr<Field>				arrowField;
typedef std::vector<arrowField>				arrowFieldVec;
typedef std::shared_ptr<DataType>			arrowDataType;
typedef std::shared_ptr<KeyValueMetadata>	arrowKeyValueMetadata;
typedef std::shared_ptr<ArrayBuilder>		arrowBufferBuilder;
typedef std::vector<arrowBufferBuilder>		arrowBufferBuilderVec;
typedef std::shared_ptr<Array>				arrowArray;
typedef std::vector<arrowArray>				arrowArrayVec;
typedef std::shared_ptr<RecordBatch>		arrowRecordBatch;
typedef std::shared_ptr<io::FileOutputStream>	arrowFileOutputStream;
typedef std::shared_ptr<ipc::RecordBatchWriter>	arrowRecordBatchWriter;

static arrowSchema		arrow_schema = NULLPTR;
static arrowBufferBuilderVec arrow_builders;
static arrowFileOutputStream arrow_out_stream;
static arrowRecordBatchWriter arrow_file_writer;
static arrowKeyValueMetadata arrow_schema_metadata = NULLPTR;
static const char	   *output_filename = NULL;
static size_t			segment_sz = (256UL << 20);		/* 256MB in default */
static bool				csv_mode = false;
static bool				skip_header = false;
static long				skip_offset = 0;
static long				skip_limit = -1;
static bool				shows_progress = false;
static int				verbose = 0;
static cppStringVec		tsv_input_files;
static bool				use_parquet = false;
#ifdef HAS_PARQUET
typedef struct parquetColumnBuffer
{
	parquetColumnBuffer() {
		arrow_type = NULLPTR;
		nitems = 0;
		nrooms = 0;
		nullcount = 0;
		extra_sz = 0;
		nullmap = NULL;
		values = NULL;
	}
	arrowDataType	arrow_type;
	size_t	  (*fetch_token)(struct parquetColumnBuffer *cbuf,
							 const char *token);
	void	  (*write_batch)(struct parquetColumnBuffer *cbuf,
							 parquet::ColumnWriter *__writer);
	int64_t		nitems;
	int64_t		nrooms;
	int64_t		nullcount;
	int64_t		extra_sz;
	int16_t	   *nullmap;
	void	   *values;
} parquetColumnBuffer;
static Compression::type parquet_compression = Compression::type::UNCOMPRESSED;
static std::shared_ptr<parquet::schema::GroupNode>	parquet_schema;
static std::shared_ptr<parquet::WriterProperties>	parquet_props;
static std::unique_ptr<parquet::ParquetFileWriter>	parquet_file_writer;
static std::vector<parquetColumnBuffer>				parquet_column_buffers;
#endif

#define ARROW_ALIGN(LEN)		(((uintptr_t)(LEN) + 63UL) & ~63UL)
#define BITMAPLEN(NITEMS)		(((NITEMS) + 7) / 8)
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

/*
 * __trim
 */
static inline char *
__trim(char *token)
{
	if (token)
	{
		char   *tail = token + strlen(token) - 1;

		while (isspace(*token))
			token++;
		while (tail >= token && isspace(*tail))
			*tail-- = '\0';
	}
	return token;
}

/*
 * __strtok_q - quotable strok
 */
static char *
__strtok_q(char *line, char delim, char **savepos)
{
	bool	in_quote = false;
	char   *r_pos;
	char   *w_pos;
	char   *head;

	if (line)
		r_pos = w_pos = line;
	else if (*savepos != NULL)
		r_pos = w_pos = *savepos;
	else
		return NULL;
	head = w_pos;

	while (*r_pos != '\0')
	{
		if (*r_pos == '"')		/* begin/end quotation */
		{
			in_quote = !in_quote;
			r_pos++;
		}
		else if (*r_pos == '\\')	/* escape */
		{
			switch (r_pos[1])
			{
				case '\0':
					*w_pos++ = *r_pos++;
					/* Now *r_pos == '\0', then it will terminate the loop */
					break;
				case '\\':
					*w_pos++ = '\\';
					r_pos += 2;
					break;
				case '"':
					*w_pos++ = '"';
					r_pos += 2;
					break;
				case 'n':
					*w_pos++ = '\n';
					r_pos += 2;
					break;
				case 'r':
					*w_pos++ = '\r';
					r_pos += 2;
					break;
				case 't':
					*w_pos++ = '\t';
					r_pos += 2;
					break;
				case 'f':
					*w_pos++ = '\f';
					r_pos += 2;
					break;
				case 'b':
					*w_pos++ = '\b';
					r_pos += 2;
					break;
				default:
					Elog("unknown escape '\\%c'", r_pos[1]);
			}
		}
		else if (!in_quote && *r_pos == delim)
		{
			/* delimiter */
			break;
		}
		else
		{
			/* payload */
			*w_pos++ = *r_pos++;
		}
	}
	*savepos = (*r_pos == '\0' ? NULL : r_pos+1);
	*w_pos++ = '\0';
	return __trim(head);
}

/*
 * Buffer builder
 */
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
		  "     --csv                 use comma as separator, instead of tab\n"
#if HAS_PARQUET
		  "  -p|--parquet[=COMPRESSION] write out in Parquet format\n"
#endif
		  "  -S|--schema=SCHEMA       schema definition\n"
		  "  -s|--segment-sz=SIZE     unit size of record batch (default: 256MB)\n"
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
		  "  TYPE  := bool          boolean\n"
		  "           intX          signed integer [X=8,16,<32>,64]\n"
		  "           uintX         unsigned integer [X=8,16,<32>,64]\n"
		  "           floatX        floating point [X=2,4,<8>]\n"
		  "           decimalX      fixed point numeric [X=<128>,256]\n"
		  "           dateX         date [X=<-sec>,-ms]\n"
		  "           timeX         time [X=-sec,<-ms>,-us,-ns]\n"
		  "           timestampX    timestamp [X-sec,-ms,<-us>,-ns]\n"
		  "           utf8,text     variable length text\n"
		  "           utf8-large    variable length text (64bit offset)\n"
		  "  EXTRA := not_null      field cannot contain NULLs\n"
		  "           nullable      field can contain NULLs (default)\n"
		  "           KEY=VALUE     field level custom metadata\n"
#if HAS_PARQUET
		  "           COMPRESSION   field compression algorithm for Parquet\n"
		  "  COMPRESSION := [uncompression|snappy|gzip|zstd|brotli|lz4]\n"
#endif
		  "\n"
		  "Example)\n"
		  "  -S user_id:uint:not_null,name:text,mail:text,birthday:date:my_key=comment\n"
		  "\n"
		  "Report bugs to <pgstrom@heterodb.com>.\n",
		  stderr);
	exit(1);
}
#define usage()		__usage(NULL)

static const char *
parse_options(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"output",      required_argument, NULL, 'o'},
		{"csv",         no_argument,       NULL, 1004},
#if HAS_PARQUET
		{"parquet",     optional_argument, NULL, 'p'},
#endif
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
	std::vector<std::string> meta_keys;
	std::vector<std::string> meta_values;
	const char *schema_definition = NULL;
	int			c;

	while ((c = getopt_long(argc, argv,
							"o:S:p::s:m:vh",
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
#ifdef HAS_PARQUET
			case 'p':	/* --parquet */
				if (use_parquet)
					Elog("-p|--parquet was given twice");
				use_parquet = true;
				if (!optarg)
					parquet_compression = Compression::type::ZSTD;	/* default */
				else if (strcasecmp(optarg, "uncompressed") == 0)
					parquet_compression = Compression::type::UNCOMPRESSED;
				else if (strcasecmp(optarg, "snappy") == 0)
					parquet_compression = Compression::type::SNAPPY;
				else if (strcasecmp(optarg, "gzip") == 0)
					parquet_compression = Compression::type::GZIP;
				else if (strcasecmp(optarg, "zstd") == 0)
					parquet_compression = Compression::type::ZSTD;
				else if (strcasecmp(optarg, "brotli") == 0)
					parquet_compression = Compression::type::BROTLI;
				else if (strcasecmp(optarg, "lz4") == 0)
					parquet_compression = Compression::type::LZ4;
				else
					Elog("unknown -p|--parquet compression option [%s]", optarg);
				break;
#endif
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
			case 1004:	/* --csv */
				csv_mode = true;
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
	for (int k=optind; k < argc; k++)
		tsv_input_files.push_back(argv[k]);

	if (!meta_keys.empty() && !meta_values.empty())
	{
		assert(meta_keys.size() == meta_values.size());
		arrow_schema_metadata = std::make_shared<KeyValueMetadata>(meta_keys,
																   meta_values);
	}
	if (!schema_definition)
		__usage("-S|--schema must be given with schema definition");
	return schema_definition;
}

#ifdef HAS_PARQUET
/*
 * Parquet Token Fetch Callbacks
 */
static inline void
__parquet_expand_buffer(parquetColumnBuffer *pqbuf, int unitsz)
{
	if (pqbuf->nitems >= pqbuf->nrooms)
	{
		int64_t		__nrooms = (10000 + 2 * pqbuf->nrooms);

		pqbuf->nullmap = (int16_t *)realloc(pqbuf->nullmap,
											sizeof(int16_t) * __nrooms);
		if (!pqbuf->nullmap)
			Elog("out of memory");
		pqbuf->values = (void *)realloc(pqbuf->values,
										unitsz * __nrooms);
		if (!pqbuf->values)
			Elog("out of memory");
		pqbuf->nrooms = __nrooms;
	}
}

static inline void
__parquet_fetch_null_token(parquetColumnBuffer *pqbuf)
{
	pqbuf->nullcount++;
	pqbuf->nullmap[pqbuf->nitems++] = 0;
}

template <typename B_TYPE, typename E_TYPE>
static size_t
__parquet_fetch_simple_token(parquetColumnBuffer *pqbuf,
							 const char *token,
							 const char *type_name)
{
	__parquet_expand_buffer(pqbuf, sizeof(E_TYPE));
	if (!token)
		__parquet_fetch_null_token(pqbuf);
	else
	{
		int64_t		index = pqbuf->nitems++;
		E_TYPE		value;

		if (!arrow::internal::ParseValue<B_TYPE>(token, std::strlen(token), &value))
			Elog("unable to fetch token [%s] for %s", token, type_name);
		pqbuf->nullmap[index] = 1;
		((E_TYPE *)pqbuf->values)[index] = value;
	}
	return (ARROW_ALIGN(pqbuf->nullcount > 0 ? BITMAPLEN(pqbuf->nitems) : 0) +
			ARROW_ALIGN(sizeof(E_TYPE) * pqbuf->nitems));
}
#define PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(NAME,B_TYPE,E_TYPE)			\
	static size_t														\
	parquet_fetch_token_##NAME(parquetColumnBuffer *pqbuf, const char *token) \
	{																	\
		return __parquet_fetch_simple_token<B_TYPE,E_TYPE>((pqbuf),(token),#B_TYPE); \
	}
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(boolean, BooleanType, bool)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(int8,   Int32Type, int32_t)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(int16,  Int32Type, int32_t)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(int32,  Int32Type, int32_t)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(int64,  Int64Type, int64_t)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(uint8,  Int32Type, int32_t)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(uint16, Int32Type, int32_t)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(uint32, Int32Type, int32_t)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(uint64, Int64Type, int64_t)

PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(float4, FloatType,   float)
PARQUET_FETCH_SIMPLE_TOKEN_TEMPLATE(float8, DoubleType,  double)
static size_t
parquet_fetch_token_float2(parquetColumnBuffer *pqbuf, const char *token)
{
	__parquet_expand_buffer(pqbuf, sizeof(uint16_t));
	if (!token)
		__parquet_fetch_null_token(pqbuf);
	else
	{
		int64_t		index = pqbuf->nitems++;
		float		fval;

		if (!arrow::internal::ParseValue<FloatType>(token, std::strlen(token), &fval))
			Elog("unable to fetch token [%s] for %s", token, "Float");
		pqbuf->nullmap[index] = 1;
		((uint16_t *)pqbuf->values)[index] = fp32_to_fp16(fval);
	}
	return (ARROW_ALIGN(pqbuf->nullcount > 0 ? BITMAPLEN(pqbuf->nitems) : 0) +
			ARROW_ALIGN(sizeof(uint16_t) * pqbuf->nitems));
}

template <typename B_TYPE, typename E_TYPE>
static size_t
__parquet_fetch_token_decimal_common(parquetColumnBuffer *pqbuf, const char *token)
{
	__parquet_expand_buffer(pqbuf,sizeof(E_TYPE));
	if (!token)
		__parquet_fetch_null_token(pqbuf);
	else
	{
		int64_t		index = pqbuf->nitems++;
		auto		b_type = std::dynamic_pointer_cast<B_TYPE>(pqbuf->arrow_type);
		int			precision = b_type->precision();
		int			scale = b_type->scale();
		E_TYPE		value;
		Status		rv;

		rv = E_TYPE::FromString(token, &value, &precision, &scale);
		if (!rv.ok())
		{
			cppString	type_name = b_type->ToString();
			Elog("unable to convert token [%s] for %s",
				 token, type_name.c_str());
		}
		pqbuf->nullmap[index] = 1;
		((E_TYPE *)pqbuf->values)[index] = value;
	}
	return (ARROW_ALIGN(pqbuf->nullcount > 0 ? BITMAPLEN(pqbuf->nitems) : 0) +
			ARROW_ALIGN(sizeof(E_TYPE) * pqbuf->nitems));
}
static size_t
parquet_fetch_token_decimal128(parquetColumnBuffer *pqbuf, const char *token)
{
	return __parquet_fetch_token_decimal_common<arrow::Decimal128Type,
												arrow::Decimal128>(pqbuf,token);
}
static size_t
parquet_fetch_token_decimal256(parquetColumnBuffer *pqbuf, const char *token)
{
	return __parquet_fetch_token_decimal_common<arrow::Decimal256Type,
												arrow::Decimal256>(pqbuf,token);
}

template <typename B_TYPE, typename E_TYPE>
static size_t
__parquet_fetch_datetime_token(parquetColumnBuffer *pqbuf,
							   const char *token,
							   const char *type_name)
{
	__parquet_expand_buffer(pqbuf, sizeof(E_TYPE));
	if (!token)
		__parquet_fetch_null_token(pqbuf);
	else
	{
		auto		ts_type = std::dynamic_pointer_cast<B_TYPE>(pqbuf->arrow_type);
		int64_t     index = pqbuf->nitems++;
		E_TYPE      value;

		if (!arrow::internal::ParseValue<B_TYPE>(*ts_type,
												 token,
												 std::strlen(token),
												 &value))
			Elog("unable to fetch token [%s] for %s", token, type_name);
		pqbuf->nullmap[index] = 1;
		((E_TYPE *)pqbuf->values)[index] = value;
	}
	return (ARROW_ALIGN(pqbuf->nullcount > 0 ? BITMAPLEN(pqbuf->nitems) : 0) +
			ARROW_ALIGN(sizeof(E_TYPE) * pqbuf->nitems));
}
#define PARQUET_FETCH_DATETIME_TOKEN_TEMPLATE(NAME,B_TYPE,E_TYPE)		\
	static size_t														\
	parquet_fetch_token_##NAME(parquetColumnBuffer *pqbuf, const char *token) \
	{																	\
		return __parquet_fetch_datetime_token<B_TYPE,E_TYPE>((pqbuf),(token),#B_TYPE); \
	}
PARQUET_FETCH_DATETIME_TOKEN_TEMPLATE(date32,Date32Type,int32_t)
PARQUET_FETCH_DATETIME_TOKEN_TEMPLATE(time32,Time32Type,int32_t)
PARQUET_FETCH_DATETIME_TOKEN_TEMPLATE(time64,Time64Type,int64_t)
PARQUET_FETCH_DATETIME_TOKEN_TEMPLATE(timestamp_ms,TimestampType,int64_t)
PARQUET_FETCH_DATETIME_TOKEN_TEMPLATE(timestamp_us,TimestampType,int64_t)

static size_t
parquet_fetch_token_utf8(parquetColumnBuffer *pqbuf, const char *token)
{
	__parquet_expand_buffer(pqbuf, sizeof(parquet::ByteArray));
	if (!token)
		__parquet_fetch_null_token(pqbuf);
	else
	{
		parquet::ByteArray value;
		int64_t		index = pqbuf->nitems++;

		value.len = std::strlen(token);
		value.ptr = (uint8_t *)strdup(token);
		if (!value.ptr)
			Elog("out of memory");
		pqbuf->nullmap[index] = 1;
		((parquet::ByteArray *)pqbuf->values)[index] = value;
		pqbuf->extra_sz += value.len;
	}
	return (ARROW_ALIGN(pqbuf->nullcount > 0 ? BITMAPLEN(pqbuf->nitems) : 0) +
			ARROW_ALIGN(sizeof(uint32_t) * pqbuf->nitems) +
			ARROW_ALIGN(pqbuf->extra_sz));
}

template <typename W_TYPE, typename E_TYPE>
static inline void
__parquet_write_batch_common(parquetColumnBuffer *pqbuf,
							 parquet::ColumnWriter *__writer)
{
	auto writer = static_cast<W_TYPE *>(__writer);

	writer->WriteBatch(pqbuf->nitems,
					   pqbuf->nullmap,
					   NULL,
					   (const E_TYPE *)pqbuf->values);
	/* buffer reset */
	pqbuf->nitems = 0;
	pqbuf->extra_sz = 0;
}
#define PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(NAME,W_TYPE,E_TYPE)			\
	static void															\
	parquet_write_batch_##NAME(parquetColumnBuffer *pqbuf,				\
							   parquet::ColumnWriter *__writer)			\
	{																	\
		__parquet_write_batch_common<parquet::W_TYPE,E_TYPE>(pqbuf,__writer); \
	}
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(boolean,BoolWriter,bool)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(int8,  Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(int16, Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(int32, Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(int64, Int64Writer,int64_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(uint8, Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(uint16,Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(uint32,Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(uint64,Int64Writer,int64_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(float2,FixedLenByteArrayWriter,parquet::FixedLenByteArray)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(float4,FloatWriter,float)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(float8,DoubleWriter,double)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(decimal128,FixedLenByteArrayWriter,parquet::FixedLenByteArray)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(decimal256,FixedLenByteArrayWriter,parquet::FixedLenByteArray)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(date32,Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(time32,Int32Writer,int32_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(time64,Int64Writer,int64_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(timestamp_ms,Int64Writer,int64_t)
PARQUET_WRITE_BATCH_SIMPLE_TEMPLATE(timestamp_us,Int64Writer,int64_t)
static void
parquet_write_batch_utf8(parquetColumnBuffer *pqbuf,
						 parquet::ColumnWriter *__writer)
{
	auto writer = static_cast<parquet::ByteArrayWriter *>(__writer);
	auto values = reinterpret_cast<const parquet::ByteArray*>(pqbuf->values);

	writer->WriteBatch(pqbuf->nitems,
					   pqbuf->nullmap,
					   NULL,
					   (const parquet::ByteArray *)pqbuf->values);
	/* buffer reset */
	for (int64_t i=0; i < pqbuf->nitems; i++)
	{
		if (pqbuf->nullmap && pqbuf->nullmap[i] != 0)
			free((void *)values[i].ptr);
	}
	pqbuf->nitems = 0;
	pqbuf->extra_sz = 0;
}
#endif	/* HAS_PARQUET */

static void
setup_field_definition(const char *f_name,
					   const char *t_name,
					   char *extra,
#ifdef HAS_PARQUET
					   parquet::schema::NodeVector &parquet_fields,
					   parquet::WriterProperties::Builder &parquet_config,
#endif
					   arrowFieldVec &arrow_fields)
{
	cppString		field_name = f_name;
	arrowKeyValueMetadata field_meta = NULLPTR;
	bool			nullable = true;
	MemoryPool	   *pool = default_memory_pool();
	arrowBufferBuilder build;
#ifdef HAS_PARQUET
	parquet::schema::NodePtr pqnode = NULLPTR;
	Compression::type compression = parquet_compression;
	parquetColumnBuffer cbuf;
#endif
	/* parse extra attributes */
	if (extra)
	{
		cppStringVec	meta_keys;
		cppStringVec	meta_values;
		char	   *tok, *val, *pos;

		for (tok = strtok_r(extra, ";", &pos);
			 tok != NULL;
			 tok = strtok_r(NULL, ";", &pos))
		{
			if (strcasecmp(tok, "not_null") == 0)
				nullable = false;
			else if (strcasecmp(tok, "nullable") == 0)
				nullable = true;
#ifdef HAS_PARQUET
			else if (strcasecmp(tok, "uncompressed") == 0)
				compression = Compression::type::UNCOMPRESSED;
			else if (strcasecmp(tok, "snappy") == 0)
				compression = Compression::type::SNAPPY;
			else if (strcasecmp(tok, "brotli") == 0)
				compression = Compression::type::BROTLI;
			else if (strcasecmp(tok, "zstd") == 0)
				compression = Compression::type::ZSTD;
			else if (strcasecmp(tok, "lz4") == 0)
				compression = Compression::type::LZ4;
#endif
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
			field_meta = std::make_shared<KeyValueMetadata>(meta_keys,
															meta_values);
		}
#ifdef HAS_PARQUET
		if (compression != parquet_compression)
			parquet_config.compression(f_name, compression);
#endif
	}

#ifdef HAS_PARQUET
#define __SETUP_PARQUET_NODE(PTYPE,CTYPE,CALLBACK,__length,__precision,__scale)	\
	if (use_parquet)													\
	{																	\
		cbuf.arrow_type = build->type();								\
		cbuf.fetch_token = parquet_fetch_token_##CALLBACK;				\
		cbuf.write_batch = parquet_write_batch_##CALLBACK;				\
		pqnode = parquet::schema::PrimitiveNode::Make(f_name,			\
													  parquet::Repetition::REQUIRED, \
													  parquet::Type::PTYPE, \
													  parquet::ConvertedType::CTYPE, \
													  (__length),		\
													  (__precision),	\
													  (__scale));		\
	}
#define SETUP_PARQUET_NODE(PTYPE,CTYPE,CALLBACK)		\
	__SETUP_PARQUET_NODE(PTYPE,CTYPE,CALLBACK,-1,-1,-1)
#else
#define __SETUP_PARQUET_NODE(PTYPE,CTYPE,CALLBACK,__length,__precision,__scale)
#define SETUP_PARQUET_NODE(PTYPE,CTYPE,CALLBACK)
#endif
	/* for each data types */
	if (strcasecmp(t_name, "bool") == 0)
	{
		build = std::make_shared<BooleanBuilder>(arrow::boolean(), pool);
		SETUP_PARQUET_NODE(BOOLEAN,NONE,boolean)
	}
	else if (strcasecmp(t_name, "int8") == 0)
	{
		build = std::make_shared<Int8Builder>(arrow::int8(), pool);
		SETUP_PARQUET_NODE(INT32,INT_8,int8)
	}
	else if (strcasecmp(t_name, "int16") == 0)
	{
		build = std::make_shared<Int16Builder>(arrow::int16(), pool);
		SETUP_PARQUET_NODE(INT32,INT_16,int16)
	}
	else if (strcasecmp(t_name, "int") == 0 ||
			 strcasecmp(t_name, "int32") == 0)
	{
		build = std::make_shared<Int32Builder>(arrow::int32(), pool);
		SETUP_PARQUET_NODE(INT32,NONE,int32);
	}
	else if (strcasecmp(t_name, "int64") == 0)
	{
		build = std::make_shared<Int64Builder>(arrow::int64(), pool);
		SETUP_PARQUET_NODE(INT64,NONE,int64);
	}
	else if (strcasecmp(t_name, "uint8") == 0)
	{
		build = std::make_shared<UInt8Builder>(arrow::uint8(), pool);
		SETUP_PARQUET_NODE(INT32,UINT_8,uint8)
	}
	else if (strcasecmp(t_name, "uint16") == 0)
	{
		build = std::make_shared<UInt16Builder>(arrow::uint16(), pool);
		SETUP_PARQUET_NODE(INT32,UINT_16,uint16)
	}
	else if (strcasecmp(t_name, "uint") == 0 ||
			 strcasecmp(t_name, "uint32") == 0)
	{
		build = std::make_shared<UInt32Builder>(arrow::uint32(), pool);
		SETUP_PARQUET_NODE(INT32,UINT_32,uint32);
	}
	else if (strcasecmp(t_name, "uint64") == 0)
	{
		build = std::make_shared<UInt64Builder>(arrow::uint64(), pool);
		SETUP_PARQUET_NODE(INT64,UINT_64,uint64);
	}
	else if (strcasecmp(t_name, "float2") == 0)
	{
		build = std::make_shared<HalfFloatBuilder>(arrow::float16(), pool);
		__SETUP_PARQUET_NODE(FIXED_LEN_BYTE_ARRAY,NONE,float2,2,-1,-1);
	}
	else if (strcasecmp(t_name, "float4") == 0)
	{
		build = std::make_shared<FloatBuilder>(arrow::float32(), pool);
		SETUP_PARQUET_NODE(FLOAT,NONE,float4);
	}
	else if (strcasecmp(t_name, "float") == 0 ||
			 strcasecmp(t_name, "float8") == 0)
	{
		build = std::make_shared<DoubleBuilder>(arrow::float64(), pool);
		SETUP_PARQUET_NODE(DOUBLE,NONE,float8);
	}
	else if (strcasecmp(t_name, "decimal") == 0 ||
			 strcasecmp(t_name, "decimal128") == 0)
	{
		build = std::make_shared<Decimal128Builder>(arrow::decimal128(27, 6), pool);
		__SETUP_PARQUET_NODE(FIXED_LEN_BYTE_ARRAY,DECIMAL,decimal128,16,27,6);
	}
	else if (strcasecmp(t_name, "decimal256") == 0)
	{
		build = std::make_shared<Decimal256Builder>(arrow::decimal256(36,9), pool);
		__SETUP_PARQUET_NODE(FIXED_LEN_BYTE_ARRAY,DECIMAL,decimal256,32,36,9);
	}
	else if (strcasecmp(t_name, "date") == 0 ||
			 strcasecmp(t_name, "date-sec") == 0)
	{
		build = std::make_shared<Date32Builder>(arrow::date32(), pool);
		SETUP_PARQUET_NODE(INT32,DATE,date32);
	}
	else if (strcasecmp(t_name, "date-ms") == 0)
	{
		if (use_parquet)
			Elog("date-ms is not available in Parquet output");
		build = std::make_shared<Date64Builder>(arrow::date64(), pool);
	}
	else if (strcasecmp(t_name, "time-sec") == 0)
	{
		if (use_parquet)
			Elog("time-sec is not available in Parquet output");
		build = std::make_shared<Time32Builder>(arrow::time32(TimeUnit::SECOND), pool);
	}
	else if (strcasecmp(t_name, "time") == 0 ||
			 strcasecmp(t_name, "time-ms") == 0)
	{
		build = std::make_shared<Time32Builder>(arrow::time32(TimeUnit::MILLI), pool);
		SETUP_PARQUET_NODE(INT32,TIME_MILLIS,time32);
	}
	else if (strcasecmp(t_name, "time-us") == 0)
	{
		build = std::make_shared<Time64Builder>(arrow::time64(TimeUnit::MICRO), pool);
		SETUP_PARQUET_NODE(INT64,TIME_MICROS,time64);
	}
	else if (strcasecmp(t_name, "time-ns") == 0)
	{
		if (use_parquet)
			Elog("time-ns is not available in Parquet output");
		build = std::make_shared<Time64Builder>(arrow::time64(TimeUnit::NANO), pool);
	}
	else if (strcasecmp(t_name, "timestamp-sec") == 0)
	{
		if (use_parquet)
			Elog("timestamp-sec is not available in Parquet output");
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::SECOND), pool);
	}
	else if (strcasecmp(t_name, "timestamp-ms") == 0)
	{
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::MILLI), pool);
		SETUP_PARQUET_NODE(INT64,TIMESTAMP_MILLIS,timestamp_ms);
	}
	else if (strcasecmp(t_name, "timestamp") == 0 ||
			 strcasecmp(t_name, "timestamp-us") == 0)
	{
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::MICRO), pool);
		SETUP_PARQUET_NODE(INT64,TIMESTAMP_MICROS,timestamp_us);
	}
	else if (strcasecmp(t_name, "timestamp-ns") == 0)
	{
		if (use_parquet)
			Elog("timestamp-ns is not available in Parquet output");
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::NANO), pool);
	}
	else if (strcasecmp(t_name, "utf8") == 0 ||
			 strcasecmp(t_name, "text") == 0)
	{
		build = std::make_shared<StringBuilder>(arrow::utf8(), pool);
		SETUP_PARQUET_NODE(BYTE_ARRAY,UTF8,utf8);
	}
	else if (strcasecmp(t_name, "utf8-large") == 0)
	{
		if (use_parquet)
			Elog("utf8-large is not available in Parquet output");
		build = std::make_shared<LargeStringBuilder>(arrow::large_utf8(), pool);
	}
	else
		Elog("unknown field type: %s", t_name);
	/* add one */
	arrow_builders.push_back(build);
#if HAS_PARQUET
	if (use_parquet)
	{
		parquet_fields.push_back(pqnode);
		parquet_column_buffers.push_back(cbuf);
	}
#endif
	arrow_fields.push_back(field(field_name,
								 build->type(),
								 nullable,
								 field_meta));
#undef MAKE_PARQUET_NODE
#undef __MAKE_PARQUET_NODE
#undef SETUP_PARQUET_FIELD
#undef __SETUP_PARQUET_FIELD
}

static void
setup_schema_buffer(const char *schema_definition)
{
	char   *temp = (char *)alloca(strlen(schema_definition) + 1);
	char   *f_name, *pos;
	FieldVector arrow_fields;
#ifdef HAS_PARQUET
	parquet::schema::NodeVector parquet_fields;
	parquet::WriterProperties::Builder parquet_config = parquet::WriterProperties::Builder();

	/* per-table Parquit parameters */
	if (use_parquet)
	{
		parquet_config.compression(parquet_compression);
		parquet_config.max_row_group_length(LONG_MAX);
	}
#endif
	strcpy(temp, schema_definition);
	for (f_name = strtok_r(temp, ",", &pos);
		 f_name != NULL;
		 f_name = strtok_r(NULL, ",", &pos))
	{
		char   *t_name = strchr(f_name, ':');
		char   *extra;

		if (!t_name)
			Elog("schema definition has no type [%s]", f_name);
		*t_name++ = '\0';
		extra = strchr(t_name, ':');
		if (extra)
			*extra++ = '\0';
		/* check field name duplication */
		for (auto elem = arrow_fields.begin(); elem != arrow_fields.end(); elem++)
		{
			if ((*elem)->name() == f_name)
				Elog("field name [%s] appeared twice", f_name);
		}
		setup_field_definition(f_name, t_name, extra,
#ifdef HAS_PARQUET
							   parquet_fields,
							   parquet_config,
#endif
							   arrow_fields);
	}
	assert(arrow_fields.size() == arrow_builders.size());
	arrow_schema = schema(arrow_fields, arrow_schema_metadata);
#ifdef HAS_PARQUET
	if (use_parquet)
	{
		parquet_schema = std::static_pointer_cast<parquet::schema::GroupNode>(
			parquet::schema::GroupNode::Make("schema",
											 parquet::Repetition::REQUIRED,
											 parquet_fields));
		parquet_props = parquet_config.build();
	}
#endif
}

#define __BUFFER_USAGE_INLINE_TYPE(build, unitsz)						\
	(ARROW_ALIGN((unitsz) * (build)->length()) +						\
	 ARROW_ALIGN((build)->null_count() > 0 ? BITMAPLEN((build)->length()) : 0))
#define __BUFFER_USAGE_VARLENA_TYPE(build)								\
	(ARROW_ALIGN(sizeof(int32_t) * ((build)->length() + 1)) +			\
	 ARROW_ALIGN((build)->value_data_length()) +						\
	 ARROW_ALIGN((build)->null_count() > 0 ? BITMAPLEN((build)->length()) : 0))
#define __BUFFER_USAGE_LARGE_VARLENA_TYPE(build)						\
	(ARROW_ALIGN(sizeof(int64_t) * ((build)->length() + 1)) +			\
	 ARROW_ALIGN((build)->value_data_length()) +						\
	 ARROW_ALIGN((build)->null_count() > 0 ? BITMAPLEN((build)->length()) : 0))

static inline size_t
append_bool_token(arrowBufferBuilder __build, const char *token)
{
	auto	build = std::dynamic_pointer_cast<arrow::BooleanBuilder>(__build);
	bool	value;
	Status	rv;

	if (!token)
		rv = build->AppendNull();
	else
	{
		if (!arrow::internal::ParseValue<BooleanType>(token, strlen(token), &value))
			Elog("unable to convert token to Boolean [%s]", token);
		rv = build->Append(value);
	}
	if (!rv.ok())
		Elog("unable to Append [%s] to buffer: %s",
			 token ? token : "NULL", rv.ToString().c_str());
	return (ARROW_ALIGN(BITMAPLEN(build->length())) +
			ARROW_ALIGN(build->null_count() > 0 ? BITMAPLEN(build->length()) : 0));
}

template <typename B_TYPE, typename D_TYPE, typename E_TYPE>
static inline size_t
__append_simple_numeric_token(arrowBufferBuilder __build, const char *token)
{
	auto	build = std::dynamic_pointer_cast<B_TYPE>(__build);
	auto	d_type = std::dynamic_pointer_cast<D_TYPE>(build->type());
	E_TYPE	value;
	Status	rv;

	if (!token)
		rv = build->AppendNull();
	else
	{
		if (!arrow::internal::ParseValue<D_TYPE>(token, std::strlen(token), &value))
		{
			cppString	type_name = d_type->ToString();
			Elog("unable to convert token to %s from [%s]",
				 type_name.c_str(), token);
		}
		rv = build->Append(value);

	}
	if (!rv.ok())
		Elog("unable to Append [%s] to buffer: %s",
			 token ? token : "NULL", rv.ToString().c_str());
	return __BUFFER_USAGE_INLINE_TYPE(build, sizeof(value));
}

#define append_int8_token(build,token)									\
	__append_simple_numeric_token<Int8Builder,Int8Type,int8_t>((build),(token))
#define append_int16_token(build,token)									\
	__append_simple_numeric_token<Int16Builder,Int16Type,int16_t>((build),(token))
#define append_int32_token(build,token)									\
	__append_simple_numeric_token<Int32Builder,Int32Type,int32_t>((build),(token))
#define append_int64_token(build,token)									\
	__append_simple_numeric_token<Int64Builder,Int64Type,int64_t>((build),(token))
#define append_uint8_token(build,token)									\
	__append_simple_numeric_token<UInt8Builder,UInt8Type,uint8_t>((build),(token))
#define append_uint16_token(build,token)								\
	__append_simple_numeric_token<UInt16Builder,UInt16Type,uint16_t>((build),(token))
#define append_uint32_token(build,token)								\
	__append_simple_numeric_token<UInt32Builder,UInt32Type,uint32_t>((build),(token))
#define append_uint64_token(build,token)								\
	__append_simple_numeric_token<UInt64Builder,UInt64Type,uint64_t>((build),(token))
#define append_float4_token(build,token)								\
	__append_simple_numeric_token<FloatBuilder,FloatType,float>((build),(token))
#define append_float8_token(build,token)								\
	__append_simple_numeric_token<DoubleBuilder,DoubleType,double>((build),(token))
static inline size_t
append_float2_token(arrowBufferBuilder __build, const char *token)
{
	auto    build = std::dynamic_pointer_cast<arrow::HalfFloatBuilder>(__build);
	float	fval;
	half_t	value;
	Status	rv;

	if (!token)
		rv = build->AppendNull();
	else
	{
		if (!arrow::internal::ParseValue<FloatType>(token, strlen(token), &fval))
			Elog("unable to convert token to HalfFloat [%s]", token);
		value = fp32_to_fp16(fval);
		rv = build->Append(value);
	}
	if (!rv.ok())
		Elog("unable to Append [%s] to buffer: %s",
			 token ? token : "NULL", rv.ToString().c_str());
	return __BUFFER_USAGE_INLINE_TYPE(build, sizeof(uint16_t));
}

template <typename B_TYPE, typename D_TYPE, typename E_TYPE>
static inline size_t
__append_common_decimal_token(arrowBufferBuilder __build, const char *token)
{
	auto	build = std::dynamic_pointer_cast<B_TYPE>(__build);
	auto	d_type = std::dynamic_pointer_cast<D_TYPE>(build->type());
	int		precision = d_type->precision();
	int		scale = d_type->scale();
	E_TYPE	value;
	Status	rv;

	if (!token)
		rv = build->AppendNull();
	else
	{
		rv = E_TYPE::FromString(token, &value, &precision, &scale);
		if (!rv.ok())
		{
			cppString	type_name = d_type->ToString();
			Elog("unable to convert token to %s from [%s]",
				 type_name.c_str(), token);
		}
		rv = build->Append(value);
	}
	if (!rv.ok())
		Elog("unable to Append [%s] to buffer: %s",
			 token ? token : "NULL", rv.ToString().c_str());
	return __BUFFER_USAGE_INLINE_TYPE(build, sizeof(E_TYPE));
}
#define append_decimal128_token(build,token)	\
	__append_common_decimal_token<Decimal256Builder,Decimal256Type,Decimal256>((build),(token))
#define append_decimal256_token(build,token)	\
	__append_common_decimal_token<Decimal128Builder,Decimal128Type,Decimal128>((build),(token))

template <typename B_TYPE, typename D_TYPE, typename E_TYPE>
static inline size_t
__append_common_datetime_token(arrowBufferBuilder __build, const char *token)
{
	auto	build = std::dynamic_pointer_cast<B_TYPE>(__build);
	auto	ts_type = std::dynamic_pointer_cast<D_TYPE>(build->type());
	E_TYPE	value;
	Status	rv;

	if (!token)
		rv = build->AppendNull();
	else
	{
		if (!arrow::internal::ParseValue<D_TYPE>(*ts_type,
												 token,
												 std::strlen(token),
												 &value))
		{
			cppString	type_name = ts_type->ToString();
			Elog("unable to convert token to %s from [%s]",
				 type_name.c_str(), token);
		}
		rv = build->Append(value);
	}
	if (!rv.ok())
	{
		cppString	type_name = ts_type->ToString();
		Elog("failed on append token [%s] on %s buffer: %s",
			 token ? token : "NULL", type_name.c_str(), rv.ToString().c_str());
	}
	return __BUFFER_USAGE_INLINE_TYPE(build, sizeof(value));
}
#define append_date32_token(build,token)		\
	__append_common_datetime_token<Date32Builder,Date32Type,int32_t>((build),(token))
#define append_date64_token(build,token)		\
	__append_common_datetime_token<Date64Builder,Date64Type,int64_t>((build),(token))
#define append_time32_token(build,token)		\
	__append_common_datetime_token<Time32Builder,Time32Type,int32_t>((build),(token))
#define append_time64_token(build,token)		\
	__append_common_datetime_token<Time64Builder,Time64Type,int64_t>((build),(token))
#define append_timestamp_token(build,token)				\
	__append_common_datetime_token<TimestampBuilder,TimestampType,int64_t>((build),(token))

static inline size_t
append_utf8_token(arrowBufferBuilder __build, const char *token)
{
	auto	build = std::dynamic_pointer_cast<arrow::StringBuilder>(__build);
	Status	rv;

	if (!token)
		rv = build->AppendNull();
	else
		rv = build->Append(token, strlen(token));
	if (!rv.ok())
		Elog("failed on append token [%s] on buffer: %s",
			 token ? token : "NULL", rv.ToString().c_str());
	return __BUFFER_USAGE_VARLENA_TYPE(build);
}

static inline size_t
append_large_utf8_token(arrowBufferBuilder __build, const char *token)
{
	auto	build = std::dynamic_pointer_cast<arrow::LargeStringBuilder>(__build);
	Status	rv;

	if (!token)
		rv = build->AppendNull();
	else
		rv = build->Append(token, strlen(token));
	if (!rv.ok())
		Elog("failed on append token [%s] on buffer: %s",
			 token ? token : "NULL", rv.ToString().c_str());
	return __BUFFER_USAGE_LARGE_VARLENA_TYPE(build);
}

static size_t
appendOneToken(arrowBufferBuilder build, const char *token)
{
	arrowDataType dtype = build->type();
	cppString	dtype_name;

	switch (dtype->id())
	{
		case Type::BOOL:
			return append_bool_token(build, token);
		case Type::INT8:
			return append_int8_token(build, token);
		case Type::INT16:
			return append_int16_token(build, token);
		case Type::INT32:
			return append_int32_token(build, token);
		case Type::INT64:
			return append_int64_token(build, token);
		case Type::UINT8:
			return append_uint8_token(build, token);
		case Type::UINT16:
			return append_uint16_token(build, token);
		case Type::UINT32:
			return append_uint32_token(build, token);
		case Type::UINT64:
			return append_uint64_token(build, token);
		case Type::HALF_FLOAT:
			return append_float2_token(build, token);
		case Type::FLOAT:
			return append_float4_token(build, token);
		case Type::DOUBLE:
			return append_float8_token(build, token);
		case Type::DECIMAL128:
			return append_decimal128_token(build, token);
		case Type::DECIMAL256:
			return append_decimal256_token(build, token);
		case Type::DATE32:
			return append_date32_token(build, token);
		case Type::DATE64:
			return append_date64_token(build, token);
		case Type::TIME32:
			return append_time32_token(build, token);
		case Type::TIME64:
			return append_time64_token(build, token);
		case Type::TIMESTAMP:
			return append_timestamp_token(build, token);
		case Type::STRING:
			return append_utf8_token(build, token);
		case Type::LARGE_STRING:
			return append_large_utf8_token(build, token);
		default:
			break;
	}
	dtype_name = dtype->ToString();
	Elog("not a supported data type: %s", dtype_name.c_str());
}

static void open_output_file(void)
{
	Result<arrowFileOutputStream>	rv1;
	Result<arrowRecordBatchWriter>	rv2;
	int			fdesc;
	const char *comment = "";

	if (output_filename)
	{
		fdesc = open(output_filename, O_RDWR | O_CREAT | O_TRUNC);
		if (fdesc < 0)
			Elog("failed on open('%s'): %m", output_filename);
	}
	else
	{
		int		suffixlen = (use_parquet ? 8 : 6);
		char   *namebuf = strdup(use_parquet
								 ? "/tmp/tsv2arrow_XXXXXX.parquet"
								 : "/tmp/tsv2arrow_XXXXXX.arrow");
		if (!namebuf)
			Elog("out of memory");
		fdesc = mkstemps(namebuf, suffixlen);
		if (fdesc < 0)
			Elog("failed on mkstemps('%s'): %m", namebuf);
		output_filename = namebuf;
		comment = ", automatically generated because of no -o FILENAME";
	}
	rv1 = io::FileOutputStream::Open(fdesc);
	if (!rv1.ok())
		Elog("failed on io::FileOutputStream::Open('%s'): %s",
			 output_filename, rv1.status().ToString().c_str());
	arrow_out_stream = rv1.ValueOrDie();
#ifdef HAS_PARQUET
	if (use_parquet)
	{
		parquet_file_writer = parquet::ParquetFileWriter::Open(arrow_out_stream,
															   parquet_schema,
															   parquet_props,
															   arrow_schema_metadata);
	}
	else
#endif
	{
		rv2 = ipc::MakeFileWriter(arrow_out_stream, arrow_schema);
		if (!rv2.ok())
			Elog("failed on ipc::MakeFileWriter for '%s': %s",
				 output_filename,
				 rv2.status().ToString().c_str());
		arrow_file_writer = rv2.ValueOrDie();
	}
	/* report */
	printf("tsv2arrow: opened the output file '%s'%s\n", output_filename, comment);
}

static void close_output_file(int rbatch_count, long total_nitems)
{
	if (arrow_file_writer)
		arrow_file_writer->Close();
#if HAS_PARQUET
	if (parquet_file_writer)
		parquet_file_writer->Close();
#endif
	arrow_out_stream->Close();
	/* report */
	printf("tsv2arrow: wrote on '%s' total %d record-batches, %ld nitems\n",
		   output_filename, rbatch_count, total_nitems);
}

static void write_out_record_batch(long nitems, int rbatch_count)
{
	arrowArrayVec	arrow_arrays;
	arrowArray		array;
	arrowRecordBatch rbatch;
	Status			rv;
	int64_t			nrows = -1;
	Result<int64_t>	foffset_before;
	Result<int64_t>	foffset_after;

	/* setup record batch */
	for (auto elem = arrow_builders.begin(); elem != arrow_builders.end(); elem++)
	{
		(*elem)->Finish(&array);
		arrow_arrays.push_back(array);
		if (nrows < 0)
			nrows = array->length();
		else
			assert(nrows == array->length());
	}
	foffset_before = arrow_out_stream->Tell();
#ifdef HAS_PARQUET
	if (use_parquet)
	{
		parquet::RowGroupWriter *pq_rgroup = parquet_file_writer->AppendRowGroup(nrows);
		for (auto elem = parquet_column_buffers.begin();
				  elem != parquet_column_buffers.end();
				  elem++)
		{
			(*elem).write_batch(&(*elem), pq_rgroup->NextColumn());
		}
	}
	else
#endif
	{
		/* setup a record batch */
		rbatch = RecordBatch::Make(arrow_schema, nitems, arrow_arrays);
		/* write out record batch */
		rv = arrow_file_writer->WriteRecordBatch(*rbatch);
		if (!rv.ok())
			Elog("failed on WriteRecordBatch: %s", rv.ToString().c_str());
	}
	foffset_after = arrow_out_stream->Tell();
	/* progress */
	if (shows_progress)
		printf("Record Batch[%d] nitems=%ld, length=%ld at file offset=%ld\n",
			   rbatch_count, nitems,
			   foffset_before.ok() && foffset_after.ok() ?
			   foffset_after.ValueOrDie() - foffset_before.ValueOrDie() : -1,
			   foffset_before.ok() ? foffset_before.ValueOrDie() : -1);
	/* reset buffers */
	for (auto elem = arrow_builders.begin(); elem != arrow_builders.end(); elem++)
		(*elem)->Reset();
}

static int		rbatch_count = 0;
static long		curr_nitems = 0;
static long		total_nitems = 0;

static bool
process_one_input_file(FILE *filp)
{
	arrowFieldVec arrow_fields = arrow_schema->fields();
	char	   *line = NULL;
	size_t		bufsz = 0;
	ssize_t		nbytes;
	char		delim = (csv_mode ? ',' : '\t');
	bool		is_first = true;

	assert(arrow_fields.size() == arrow_builders.size());
	while ((nbytes = getline(&line, &bufsz, stdin)) > 0)
	{
		size_t	curr_sz = 0;
		char   *tok, *saved;

		/* skip header line? */
		if (is_first)
		{
			is_first = false;
			if (skip_header)
				continue;
		}
		/* skip first N lines? */
		if (skip_offset > 0)
		{
			skip_offset--;
			continue;
		}
		/* dump only N lines? */
		if (skip_limit >= 0 && total_nitems >= skip_limit)
			return false;
		/* parse one line */
#ifdef HAS_PARQUET
		if (use_parquet)
		{
			tok = __strtok_q(line, delim, &saved);
			for (auto elem = parquet_column_buffers.begin();
					  elem != parquet_column_buffers.end();
					  elem++)
			{
				curr_sz += (*elem).fetch_token(&(*elem), tok);
				tok = __strtok_q(NULL, delim, &saved);
			}
		}
		else
#endif
		{
			tok = __strtok_q(line, delim, &saved);
			for (auto elem = arrow_builders.begin();
					  elem != arrow_builders.end();
					  elem++)
			{
				curr_sz += appendOneToken(*elem, tok);
				tok = __strtok_q(NULL, delim, &saved);
			}
		}
		curr_nitems++;
		total_nitems++;
		/* write out record batch */
		if (curr_sz >= segment_sz)
		{
			write_out_record_batch(curr_nitems, rbatch_count++);
			curr_nitems = 0;
		}
	}
	return true;
}

int main(int argc, char *argv[])
{
	const char *schema_definition;

	schema_definition = parse_options(argc, argv);
	setup_schema_buffer(schema_definition);
	std::cout << "Schema definition:\n" << arrow_schema->ToString(true) << "\n";
	/* open the output file */
	open_output_file();
	/* process for each input files */
	if (tsv_input_files.empty())
		process_one_input_file(stdin);
	else
	{
		for (auto elem=tsv_input_files.begin(); elem != tsv_input_files.end(); elem++)
		{
			const char *filename = (*elem).c_str();
			FILE	   *filp = fopen(filename, "rb");
			bool		status;

			if (!filp)
				Elog("failed to open '%s': %m", filename);
			status = process_one_input_file(filp);
			fclose(filp);
			if (!status)
				break;		/* reached --limit */
		}
	}
	/* write out remaining record batch */
	if (curr_nitems > 0)
		write_out_record_batch(curr_nitems, rbatch_count++);
	close_output_file(rbatch_count, total_nitems);
	return 0;
}
