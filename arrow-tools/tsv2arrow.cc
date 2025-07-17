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
#include <typeinfo>
#include <arrow/api.h>
#include <arrow/array/builder_adaptive.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/builder_time.h>
#include <arrow/array/builder_nested.h>
#include <arrow/util/value_parsing.h>
#include <ctype.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "float2.h"

using namespace arrow;

typedef std::string							cppString;
typedef std::list<cppString>				cppStringList;
typedef std::vector<cppString>				cppStringVector;
typedef std::shared_ptr<Schema>				arrowSchema;
typedef std::shared_ptr<Field>				arrowField;
typedef std::vector<arrowField>				arrowFieldVec;
typedef std::shared_ptr<DataType>			arrowDataType;
typedef std::shared_ptr<KeyValueMetadata>	arrowKeyValueMetadata;
typedef std::shared_ptr<ArrayBuilder>		arrowBufferBuilder;
typedef std::vector<arrowBufferBuilder>		arrowBufferBuilderVec;

static arrowSchema		arrow_schema = NULLPTR;
static arrowBufferBuilderVec arrow_builders;
static arrowKeyValueMetadata arrow_schema_metadata = NULLPTR;
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

/*
 * __trim
 */
static inline char *
__trim(char *token)
{
	char   *tail = token + strlen(token) - 1;

	while (isspace(*token))
		token++;
	while (tail >= token && isspace(*tail))
		*tail-- = '\0';
	return token;
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
		arrow_schema_metadata = std::make_shared<KeyValueMetadata>(meta_keys,
																   meta_values);
	}
	if (!schema_definition)
		__usage("-S|--schema must be given with schema definition");
	return schema_definition;
}

static arrowField
setup_field_definition(const char *f_name,
					   const char *t_name,
					   char *extra)
{
	cppString		field_name = f_name;
	arrowKeyValueMetadata field_meta = NULLPTR;
	bool			nullable = true;
	MemoryPool	   *pool = default_memory_pool();
	arrowBufferBuilder build;

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
			field_meta = std::make_shared<KeyValueMetadata>(meta_keys,
															meta_values);
		}
	}
	/* for each data types */
	if (strcasecmp(t_name, "bool") == 0)
		build = std::make_shared<BooleanBuilder>(arrow::boolean(), pool);
	else if (strcasecmp(t_name, "int8") == 0)
		build = std::make_shared<Int8Builder>(arrow::int8(), pool);
	else if (strcasecmp(t_name, "int16") == 0)
		build = std::make_shared<Int16Builder>(arrow::int16(), pool);
	else if (strcasecmp(t_name, "int") == 0 ||
			 strcasecmp(t_name, "int32") == 0)
		build = std::make_shared<Int32Builder>(arrow::int32(), pool);
	else if (strcasecmp(t_name, "int64") == 0)
		build = std::make_shared<Int64Builder>(arrow::int64(), pool);
	else if (strcasecmp(t_name, "uint8") == 0)
		build = std::make_shared<UInt8Builder>(arrow::uint8(), pool);
	else if (strcasecmp(t_name, "uint16") == 0)
		build = std::make_shared<UInt16Builder>(arrow::uint16(), pool);
	else if (strcasecmp(t_name, "uint") == 0 ||
			 strcasecmp(t_name, "uint32") == 0)
		build = std::make_shared<UInt32Builder>(arrow::uint32(), pool);
	else if (strcasecmp(t_name, "uint64") == 0)
		build = std::make_shared<UInt64Builder>(arrow::uint64(), pool);
	else if (strcasecmp(t_name, "float2") == 0)
		build = std::make_shared<HalfFloatBuilder>(arrow::float16(), pool);
	else if (strcasecmp(t_name, "float4") == 0)
		build = std::make_shared<FloatBuilder>(arrow::float32(), pool);
	else if (strcasecmp(t_name, "float") == 0 ||
			 strcasecmp(t_name, "float8") == 0)
		build = std::make_shared<DoubleBuilder>(arrow::float64(), pool);
	else if (strcasecmp(t_name, "decimal") == 0 ||
			 strcasecmp(t_name, "decimal128") == 0)
		build = std::make_shared<Decimal128Builder>(arrow::decimal128(27, 6), pool);
	else if (strcasecmp(t_name, "decimal256") == 0)
		build = std::make_shared<Decimal256Builder>(arrow::decimal256(36,9), pool);
	else if (strcasecmp(t_name, "date") == 0 ||
			 strcasecmp(t_name, "date-sec") == 0)
		build = std::make_shared<Date32Builder>(arrow::date32(), pool);
	else if (strcasecmp(t_name, "date-ms") == 0)
		build = std::make_shared<Date64Builder>(arrow::date64(), pool);
	else if (strcasecmp(t_name, "time-sec") == 0)
		std::make_shared<Time32Builder>(arrow::time32(TimeUnit::SECOND), pool);
	else if (strcasecmp(t_name, "time") == 0 ||
			 strcasecmp(t_name, "time-ms") == 0)
		build = std::make_shared<Time32Builder>(arrow::time32(TimeUnit::MILLI), pool);
	else if (strcasecmp(t_name, "time-us") == 0)
		build = std::make_shared<Time64Builder>(arrow::time64(TimeUnit::MICRO), pool);
	else if (strcasecmp(t_name, "time-ns") == 0)
		build = std::make_shared<Time64Builder>(arrow::time64(TimeUnit::NANO), pool);
	else if (strcasecmp(t_name, "timestamp-sec") == 0)
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::SECOND), pool);
	else if (strcasecmp(t_name, "timestamp-ms") == 0)
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::MILLI), pool);
	else if (strcasecmp(t_name, "timestamp") == 0 ||
			 strcasecmp(t_name, "timestamp-us") == 0)
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::MICRO), pool);
	else if (strcasecmp(t_name, "timestamp-ns") == 0)
		build = std::make_shared<TimestampBuilder>(arrow::timestamp(TimeUnit::NANO), pool);
	else if (strcasecmp(t_name, "utf8") == 0 ||
			 strcasecmp(t_name, "text") == 0)
		build = std::make_shared<StringBuilder>(arrow::utf8(), pool);
	else if (strcasecmp(t_name, "utf8-large") == 0)
		build = std::make_shared<LargeStringBuilder>(arrow::large_utf8(), pool);
	else
		Elog("unknown field type: %s", t_name);
	/* add one */
	arrow_builders.push_back(build);

	return field(field_name, build->type(), nullable, field_meta);
}

static void
setup_schema_buffer(const char *schema_definition)
{
	char   *temp = (char *)alloca(strlen(schema_definition) + 1);
	char   *f_name, *pos;
	FieldVector fields_vec;

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
		field = setup_field_definition(f_name, t_name, extra);
		fields_vec.push_back(field);
	}
	assert(fields_vec.size() == arrow_builders.size());
	arrow_schema = schema(fields_vec, arrow_schema_metadata);
}

#define ARROW_ALIGN(LEN)		(((uintptr_t)(LEN) + 63UL) & ~63UL)
#define BITMAPLEN(NITEMS)		(((NITEMS) + 7) / 8)
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

	if (!arrow::internal::ParseValue<BooleanType>(token, strlen(token), &value))
		Elog("unable to convert token to Boolean [%s]", token);
	build->Append(value);
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

	if (!arrow::internal::ParseValue<D_TYPE>(token, std::strlen(token), &value))
	{
		cppString	type_name = d_type->ToString();
		Elog("unable to convert token to %s from [%s]",
			 type_name.c_str(), token);
	}
	rv = build->Append(value);
	if (!rv.ok())
		Elog("unable to Append [%s] to buffer: %s",
			 token, rv.ToString().c_str());
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

	if (!arrow::internal::ParseValue<FloatType>(token, strlen(token), &fval))
		Elog("unable to convert token to HalfFloat [%s]", token);
	value = fp32_to_fp16(fval);
	build->Append(value);
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

	rv = E_TYPE::FromString(token, &value, &precision, &scale);
	if (!rv.ok())
	{
		cppString	type_name = d_type->ToString();
		Elog("unable to convert token to %s from [%s]",
			 type_name.c_str(), token);
	}
	build->Append(value);
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
	if (!rv.ok())
	{
		cppString	type_name = ts_type->ToString();
		Elog("failed on append token [%s] on %s buffer: %s",
			 token, type_name.c_str(), rv.ToString().c_str());
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

	build->Append(token, strlen(token));
	return __BUFFER_USAGE_VARLENA_TYPE(build);
}

static inline size_t
append_large_utf8_token(arrowBufferBuilder __build, const char *token)
{
	auto	build = std::dynamic_pointer_cast<arrow::LargeStringBuilder>(__build);

	build->Append(token, strlen(token));
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

static void write_out_record_batch(void)
{


	/* reset buffers */
	for (auto elem = arrow_builders.begin(); elem != arrow_builders.end(); elem++)
		elem->reset();
}

int main(int argc, char *argv[])
{
	const char *schema_definition;
	char	   *line = NULL;
	size_t		bufsz = 0;
	ssize_t		nbytes;
	long		total_lineno = 0;
	long		rbatch_lineno = 0;
	arrowFieldVec arrow_fields;

	schema_definition = parse_options(argc, argv);
	setup_schema_buffer(schema_definition);
	std::cout << "Schema definition:\n" << arrow_schema->ToString(true) << "\n";

	/* read and convert input TSV stream */
	arrow_fields = arrow_schema->fields();
	assert(arrow_fields.size() == arrow_builders.size());
	while ((nbytes = getline(&line, &bufsz, stdin)) > 0)
	{
		char   *tok, *pos;
		size_t	curr_sz = 0;

		/* skip header line? */
		if (total_lineno++ == 0 && skip_header)
			continue;
		tok = strtok_r(line, "\t", &pos);
		for (auto elem = arrow_builders.begin(); elem != arrow_builders.end(); elem++)
		{
			curr_sz += appendOneToken(*elem, __trim(tok));
			tok = strtok_r(NULL, "\t", &pos);
		}
		rbatch_lineno++;
		/* write out record batch */
		if (curr_sz >= segment_sz)
		{
			write_out_record_batch();
			rbatch_lineno = 0;
		}
	}
	/* write out remaining record batch */
	if (rbatch_lineno > 0)
		write_out_record_batch();
	return 0;
}
