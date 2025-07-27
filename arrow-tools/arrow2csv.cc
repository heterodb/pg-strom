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
#include <arrow/api.h>		/* dnf install libarrow-devel */
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <fcntl.h>
#include <getopt.h>
#include <iostream>
#ifdef HAS_PARQUET
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/schema.h>
#endif
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include "float2.h"

using namespace arrow;

typedef std::string				cppString;
typedef std::vector<cppString>	cppStringVec;
typedef std::shared_ptr<io::ReadableFile>	arrowReadableFile;

static const char	   *output_filename = NULL;
static FILE			   *output_filp = stdout;
static bool				shows_header = false;
static bool				csv_mode = true;
static bool				binary_output_hex = false;
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

static void
__print_string_token(const char *data, int length)
{
	auto   *pos = (const unsigned char *)data;
	auto   *end = pos + length;
	int		off = 0;

	/* needs quotation? */
	for (int i=0; i < length; i++)
	{
		int		c = data[i];

		if (!isprint(c) || c == '"' || (csv_mode && c == ','))
			goto quote;
	}
	/* ok, write string as is */
	while (off < length)
	{
		ssize_t	nbytes = fwrite(data + off, 1, length - off, stdout);

		if (nbytes <= 0)
		{
			if (errno == EINTR)
				continue;
			Elog("failed on fwrite(stdout): %m");
		}
		off += nbytes;
	}
	return;
quote:
	putchar('\"');
	while (pos < end)
	{
		int		c = *pos;

		if ((c & 0x80) == 0)
		{
			/* 1byte character */
			switch (c)
			{
				case '\n':
					putchar('\\');
					putchar('n');
					break;
				case '\r':
					putchar('\\');
					putchar('r');
					break;
				case '\t':
					putchar('\\');
					putchar('t');
					break;
				case '"':
					putchar('"');
					putchar('"');
					break;
				case '\\':
					putchar('\\');
					putchar('\\');
					break;
				default:
					if (isprint(c))
						putchar(c);
					else
						printf("\\u%04x", c);
					break;
			}
			pos++;
		}
		else if ((c & 0xe0) == 0xc0 && (pos+1 < end &&
										(pos[1] & 0xc0) == 0x80))
		{
			/* 2byte character */
			putchar(*pos++);
			putchar(*pos++);
		}
		else if ((c & 0xf0) == 0xe0 && (pos+2 < end &&
										(pos[1] & 0xc0) == 0x80 &&
										(pos[2] & 0xc0) == 0x80))
		{
			/* 3byte character */
			putchar(*pos++);
            putchar(*pos++);
			putchar(*pos++);
		}
		else if ((c & 0xf8) == 0xf0 && (pos+3 < end &&
										(pos[1] & 0xc0) == 0x80 &&
										(pos[2] & 0xc0) == 0x80 &&
										(pos[3] & 0xc0) == 0x80))
		{
			/* 4byte character */
			putchar(*pos++);
			putchar(*pos++);
            putchar(*pos++);
            putchar(*pos++);
		}
		else
		{
			/* broken UTF-8 */
			printf("\\u%04x", *pos++);
		}
	}
	putchar('\"');
}

static void
__print_binary_token(const char *data, int length)
{
	static const char hextable[] = "0123456789abcdef";

	if (binary_output_hex)
	{
		putchar('\\');
		putchar('x');
		for (int i=0; i < length; i++)
		{
			int		c = data[i];

			putchar(hextable[(c>>4) & 0x0f]);
			putchar(hextable[c & 0x0f]);
		}
	}
	else
	{
		for (int i=0; i < length; i++)
		{
			int		c = data[i];

			if (c == '\\')
			{
				putchar('\\');
				putchar('\\');
			}
			else if (isprint(c) && (!csv_mode || c != ','))
			{
				putchar(c);
			}
			else
			{
				putchar('\\');
				putchar(hextable[(c>>6) & 0x3]);
				putchar(hextable[(c>>3) & 0x7]);
				putchar(hextable[ c     & 0x7]);
			}
		}
	}
}

static void
print_field_names(std::shared_ptr<arrow::Schema> arrow_schema)
{
	auto	arrow_fields = arrow_schema->fields();

	for (auto cell = arrow_fields.begin(); cell != arrow_fields.end(); cell++)
	{
		auto	__field = (*cell);

		if (cell == arrow_fields.begin())
			std::cout << "#";
		else if (csv_mode)
			std::cout << ",";
		else
			std::cout << "\t";

		std::cout << "\""
				  << __field->name()
				  << "["
				  << __field->type()->ToString()
				  << "]\"";
	}
	std::cout << std::endl;
}

#define arrow_type_as_pgsql_name(field)						\
	__arrow_type_as_pgsql_name((field),(char *)alloca(80), 80)
static const char *
__arrow_type_as_pgsql_name(std::shared_ptr<arrow::Field> arrow_field,
						   char *buffer, size_t buffer_sz)
{
	auto	af_type = arrow_field->type();

	switch (af_type->id())
	{
		case Type::BOOL:
			return "bool";
		case Type::INT8:
		case Type::UINT8:
			return "int8";
		case Type::INT16:
		case Type::UINT16:
			return "smallint";
        case Type::INT32:
		case Type::UINT32:
			return "int";
		case Type::INT64:
		case Type::UINT64:
			return "bigint";
		case Type::HALF_FLOAT:
			return "float2";
        case Type::FLOAT:
			return "float4";
		case Type::DOUBLE:
			return "float8";
        case Type::DECIMAL128: {
			auto	d_type = std::static_pointer_cast<arrow::Decimal128Type>(af_type);
			snprintf(buffer, buffer_sz, "numeric(%d,%d)",
					 d_type->precision(),
					 d_type->scale());
			return buffer;
		}
		case Type::DECIMAL256: {
			auto	d_type = std::static_pointer_cast<arrow::Decimal256Type>(af_type);
			snprintf(buffer, buffer_sz, "numeric(%d,%d)",
					 d_type->precision(),
					 d_type->scale());
			return buffer;
		}
        case Type::STRING:
        case Type::LARGE_STRING:
			return "text";
        case Type::BINARY:
		case Type::LARGE_BINARY:
			return "bytea";
        case Type::FIXED_SIZE_BINARY: {
			auto	fb_type = std::static_pointer_cast<arrow::FixedSizeBinaryType>(af_type);
			snprintf(buffer, buffer_sz, "char(%u)", fb_type->byte_width());
			return buffer;
		}
		case Type::DATE32:
		case Type::DATE64:
			return "date";
		case Type::TIME32:
		case Type::TIME64:
			return "time";
        case Type::TIMESTAMP:
			return "timestamp";
		case Type::INTERVAL_MONTHS:
		case Type::INTERVAL_DAY_TIME:
		case Type::INTERVAL_MONTH_DAY_NANO:
		case Type::DURATION:
			return "interval";
		case Type::LIST: {
			auto	ls_type = std::static_pointer_cast<arrow::ListType>(af_type);
			auto	child = ls_type->value_field();
			snprintf(buffer, buffer_sz, "%s[]",
					 arrow_type_as_pgsql_name(child));
			return buffer;
		}
		case Type::LARGE_LIST: {
			auto	ls_type = std::static_pointer_cast<arrow::LargeListType>(af_type);
			auto	child = ls_type->value_field();
			snprintf(buffer, buffer_sz, "%s[]",
					 arrow_type_as_pgsql_name(child));
			return buffer;
		}
		case Type::STRUCT:
			snprintf(buffer, buffer_sz, "%s_comp", arrow_field->name().c_str());
			return buffer;
		default:
			Elog("not a supported data type");
	}
	return NULL;
}
	
static void
print_create_composite_type(std::shared_ptr<arrow::Field> arrow_field)
{
	auto	st_type = std::static_pointer_cast<arrow::StructType>(arrow_field->type());
	auto	children = st_type->fields();
	for (auto cell = children.begin(); cell != children.end(); cell++)
	{
		auto	__field = (*cell);

		if (__field->type()->id() == Type::STRUCT)
			print_create_composite_type(__field);
	}
	/* CREATE TYPE AS */
	std::cout << "CREATE TYPE \"" << arrow_field->name() << "_comp\" AS (";
	for (auto cell = children.begin(); cell != children.end(); cell++)
	{
		auto	__field = (*cell);

		if (cell != children.begin())
			std::cout << ",";
		std::cout << std::endl
				  << "    \"" << __field->name() << "\""
				  << arrow_type_as_pgsql_name(__field);
	}
	std::cout << std::endl << ");" << std::endl;
}

static void
print_create_table(std::shared_ptr<arrow::Schema> arrow_schema)
{
	auto	arrow_fields = arrow_schema->fields();

	/* add CREATE TYPE AS if Struct is given */
	for (auto cell = arrow_fields.begin(); cell != arrow_fields.end(); cell++)
	{
		auto	__field = (*cell);

		if (__field->type()->id() == Type::STRUCT)
			print_create_composite_type(__field);
	}
	
	std::cout << "CREATE TABLE \"" << with_create_table << "\"" << std::endl;
	std::cout << "(";
	for (auto cell = arrow_fields.begin(); cell != arrow_fields.end(); cell++)
	{
		auto	__field = (*cell);

		if (cell != arrow_fields.begin())
			std::cout << ",";
		std::cout << std::endl
				  << "    \""
				  << __field->name()
				  << "\" "
				  << arrow_type_as_pgsql_name(__field);
		if (!__field->nullable())
			std::cout << " not null";
	}
	std::cout << std::endl << ")";
	if (with_partition_of)
		std::cout << "  PARTITION OF \"" << with_partition_of << "\"";
	if (with_tablespace)
		std::cout << "  TABLESPACE (\"" << with_tablespace << "\")";
	std::cout << ";" << std::endl;

	/* COPY FROM stdin */
	std::cout << "COPY \"" << with_create_table << "\" FROM stdin WITH ("
			  << "DELIMITER " << (csv_mode ? "','" : "'\\t'")
			  << ", FORMAT " << (csv_mode ? "csv" : "text");
	if (shows_header)
		std::cout << ", HEADER true";
	std::cout << ");" << std::endl;
}

static void
process_one_token(std::shared_ptr<arrow::Array> carray, int64_t rowid, bool in_quote)
{
	if (carray->IsNull(rowid))
		return;
	switch (carray->type_id())
	{
		case Type::BOOL: {
			auto	arr = std::static_pointer_cast<arrow::BooleanArray>(carray);
			auto	value = arr->Value(rowid);
			std::cout << (value ? "true" : "false");
			break;
		}
		case Type::INT8: {
			auto	arr = std::static_pointer_cast<arrow::Int8Array>(carray);
			std::cout << (int)arr->Value(rowid);
			break;
		}
		case Type::INT16: {
			auto	arr = std::static_pointer_cast<arrow::Int16Array>(carray);
			std::cout << (int)arr->Value(rowid);
			break;
		}
		case Type::INT32: {
			auto	arr = std::static_pointer_cast<arrow::Int32Array>(carray);
			std::cout << arr->Value(rowid);
			break;
		}
		case Type::INT64: {
			auto	arr = std::static_pointer_cast<arrow::Int64Array>(carray);
			std::cout << arr->Value(rowid);
			break;
		}
		case Type::UINT8: {
			auto	arr = std::static_pointer_cast<arrow::UInt8Array>(carray);
			std::cout << (unsigned int)arr->Value(rowid);
			break;
		}
		case Type::UINT16: {
			auto	arr = std::static_pointer_cast<arrow::UInt16Array>(carray);
			std::cout << (unsigned int)arr->Value(rowid);
			break;
		}
		case Type::UINT32: {
			auto	arr = std::static_pointer_cast<arrow::UInt32Array>(carray);
			std::cout << arr->Value(rowid);
			break;
		}
		case Type::UINT64: {
			auto	arr = std::static_pointer_cast<arrow::UInt64Array>(carray);
			std::cout << arr->Value(rowid);
			break;
		}
		case Type::HALF_FLOAT: {
			auto	arr = std::static_pointer_cast<arrow::UInt16Array>(carray);
			std::cout << fp16_to_fp32(arr->Value(rowid));
			break;
		}
		case Type::FLOAT: {
			auto	arr = std::static_pointer_cast<arrow::FloatArray>(carray);
			std::cout << arr->Value(rowid);
			break;
		}
		case Type::DOUBLE: {
			auto	arr = std::static_pointer_cast<arrow::DoubleArray>(carray);
			std::cout << arr->Value(rowid);
			break;
		}
		case Type::DECIMAL128: {
			auto	arr = std::static_pointer_cast<arrow::Decimal128Array>(carray);
			auto	dtype = std::static_pointer_cast<arrow::Decimal128Type>(carray->type());
			Decimal128 dval;
			memcpy(&dval, arr->GetValue(rowid), sizeof(Decimal128));
			std::cout << dval.ToString(dtype->scale());
			break;
		}
		case Type::DECIMAL256: {
			auto	arr = std::static_pointer_cast<arrow::Decimal256Array>(carray);
			auto	dtype = std::static_pointer_cast<arrow::Decimal256Type>(carray->type());
			Decimal256 dval;
			memcpy(&dval, arr->GetValue(rowid), sizeof(Decimal256));
            std::cout << dval.ToString(dtype->scale());
			break;
		}
		case Type::STRING: {
			auto	arr = std::static_pointer_cast<arrow::StringArray>(carray);
			auto	sv = arr->GetView(rowid);
			__print_string_token(sv.data(), sv.length());
			break;
		}
		case Type::LARGE_STRING: {
			auto	arr = std::static_pointer_cast<arrow::LargeStringArray>(carray);
			auto	sv = arr->GetView(rowid);
			__print_string_token(sv.data(), sv.length());
			break;
		}
		case Type::BINARY: {
			auto	arr = std::static_pointer_cast<arrow::BinaryArray>(carray);
			auto	sv = arr->GetView(rowid);
			__print_binary_token(sv.data(), sv.size());
			break;
		}
		case Type::LARGE_BINARY: {
			auto	arr = std::static_pointer_cast<arrow::LargeBinaryArray>(carray);
			auto	sv = arr->GetView(rowid);
			__print_binary_token(sv.data(), sv.size());
			break;
		}
		case Type::FIXED_SIZE_BINARY: {
			auto	arr = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(carray);
			__print_binary_token((const char *)arr->GetValue(rowid),
								 arr->byte_width());
			break;
		}
		case Type::DATE32: {
			auto	arr = std::static_pointer_cast<arrow::Date32Array>(carray);
			int32_t	days = arr->Value(rowid);	/* days from UNIX epoch */
			time_t	tval = (time_t)days * 86400;
			struct tm __tm;

			gmtime_r(&tval, &__tm);
			printf("%04d-%02d-%02d",
				   __tm.tm_year + 1900,
				   __tm.tm_mon + 1,
				   __tm.tm_mday);
			break;
		}
		case Type::DATE64: {
			auto	arr = std::static_pointer_cast<arrow::Date64Array>(carray);
			int64_t	ms = arr->Value(rowid);		/* ms from UNIX epoch */
			time_t	tval = (time_t)(ms / 1000);
			int		msec = (ms % 1000);
			struct tm __tm;

			gmtime_r(&tval, &__tm);
			printf("%04d-%02d-%02d %02d:%02d:%02d.%03d",
				   __tm.tm_year + 1900,
				   __tm.tm_mon + 1,
				   __tm.tm_mday,
				   __tm.tm_hour,
				   __tm.tm_min,
				   __tm.tm_sec,
				   msec);
			break;
		}
		case Type::TIMESTAMP: {
			auto	arr = std::static_pointer_cast<arrow::TimestampArray>(carray);
			auto	ts_type = std::static_pointer_cast<arrow::TimestampType>(arr->type());
			time_t	tval = arr->Value(rowid);
			int		subsec;
			struct tm __tm;

			switch (ts_type->unit())
			{
				case TimeUnit::SECOND:
					gmtime_r(&tval, &__tm);
					printf("%04d-%02d-%02d %02d:%02d:%02d",
						   __tm.tm_year + 1900,
						   __tm.tm_mon + 1,
						   __tm.tm_mday,
						   __tm.tm_hour,
						   __tm.tm_min,
						   __tm.tm_sec);
					break;
				case TimeUnit::MILLI:
					subsec = tval % 1000;
					tval /= 1000;
					gmtime_r(&tval, &__tm);
					printf("%04d-%02d-%02d %02d:%02d:%02d.%03u",
						   __tm.tm_year + 1900,
						   __tm.tm_mon + 1,
						   __tm.tm_mday,
						   __tm.tm_hour,
						   __tm.tm_min,
						   __tm.tm_sec,
						   subsec);
					break;
				case TimeUnit::MICRO:
					subsec = tval % 1000000;
					tval /= 1000000;
					gmtime_r(&tval, &__tm);
					printf("%04d-%02d-%02d %02d:%02d:%02d.%06u",
						   __tm.tm_year + 1900,
						   __tm.tm_mon + 1,
						   __tm.tm_mday,
						   __tm.tm_hour,
						   __tm.tm_min,
						   __tm.tm_sec,
						   subsec);
					break;
				case TimeUnit::NANO:
					subsec = tval % 1000000000;
					tval /= 1000000000;
					gmtime_r(&tval, &__tm);
					printf("%04d-%02d-%02d %02d:%02d:%02d.%09u",
						   __tm.tm_year + 1900,
						   __tm.tm_mon + 1,
						   __tm.tm_mday,
						   __tm.tm_hour,
						   __tm.tm_min,
						   __tm.tm_sec,
						   subsec);
					break;
				default:
					Elog("unknown Arrow::Timestamp unit");
			}
			break;
		}
		case Type::TIME32: {
			auto	arr = std::static_pointer_cast<arrow::Time32Array>(carray);
			auto	tm_type = std::static_pointer_cast<arrow::Time32Type>(arr->type());
			int32_t	tval = arr->Value(rowid);
			int		hour, min, msec;

			switch (tm_type->unit())
			{
				case TimeUnit::SECOND:
					hour = tval / 3600; tval %= 3600;
					min  = tval / 60;   tval %= 60;
					printf("%02d:%02d:%02d", hour, min, tval);
					break;
				case TimeUnit::MILLI:
					msec = tval % 1000; tval /= 1000;
					hour = tval / 3600; tval %= 3600;
					min  = tval / 60;   tval %= 60;
					printf("%02d:%02d:%02d.%03d", hour, min, tval, msec);
					break;
				default:
					Elog("unknown arrow::TimeUnit (%d) for Time32", (int)tm_type->unit());
					break;
			}
			break;
		}
		case Type::TIME64: {
			auto	arr = std::static_pointer_cast<arrow::Time64Array>(carray);
			auto	tm_type = std::static_pointer_cast<arrow::Time32Type>(arr->type());
			int64_t	tval = arr->Value(rowid);
			int		hour, min, usec;

			switch (tm_type->unit())
			{
				case TimeUnit::MICRO:
					usec = tval % 1000000; tval /= 1000000;
					hour = tval / 3600;    tval %= 3600;
					min  = tval / 60;      tval %= 60;
					printf("%02d:%02d:%02d.%06d", hour, min, (int)tval, usec);
					break;
				case TimeUnit::NANO:
					usec = tval % 1000000000; tval /= 1000000000;
					hour = tval / 3600;       tval %= 3600;
					min  = tval / 60;         tval %= 60;
					printf("%02d:%02d:%02d.%09d", hour, min, (int)tval, usec);
					break;
				default:
					Elog("unknown arrow::TimeUnit (%d) for Time64", (int)tm_type->unit());
			}
			break;
		}
		case Type::INTERVAL_MONTHS: {
			auto	arr = std::static_pointer_cast<arrow::MonthIntervalArray>(carray);
			int32_t	ival = arr->Value(rowid);
			printf("%s%d days, %d ms%s",
				   csv_mode ? "\"" : "",
				   ival / 12, ival % 12,
				   csv_mode ? "\"" : "");
			break;
		}
		case Type::INTERVAL_DAY_TIME: {
			auto	arr = std::static_pointer_cast<arrow::DayTimeIntervalArray>(carray);
			auto	iv = arr->GetValue(rowid);
			/* DayTimeIntervalType {int32 dats, int32 milliseconds} */
			printf("%s%d days, %d ms%s",
				   csv_mode ? "\"" : "",
				   iv.days, iv.milliseconds,
				   csv_mode ? "\"" : "");
			break;
		}
		case Type::INTERVAL_MONTH_DAY_NANO: {
			auto	arr = std::static_pointer_cast<arrow::MonthDayNanoIntervalArray>(carray);
			auto	iv = arr->GetValue(rowid);
			/* MonthDayNanoIntervalType {int32 months, int32 days, int64 nanoseconds} */
			printf("%s%d months, %d days, %ld ns%s",
				   csv_mode ? "\"" : "",
				   iv.months,
				   iv.days,
				   iv.nanoseconds,
				   csv_mode ? "\"" : "");
			break;
		}
		case Type::DURATION: {
			auto	arr = std::static_pointer_cast<arrow::DurationArray>(carray);
			auto	du_type = std::static_pointer_cast<arrow::DurationType>(arr->type());
			int64_t	ival = arr->Value(rowid);

			switch (du_type->unit())
			{
				case TimeUnit::SECOND:
					std::cout << ival << "s";
					break;
				case TimeUnit::MILLI:
					std::cout << ival << "ms";
					break;
				case TimeUnit::MICRO:
					std::cout << ival << "us";
					break;
				case TimeUnit::NANO:
					std::cout << ival << "ns";
					break;
				default:
					Elog("unknown TimeUnit (%d) for Arrow::Duration", (int)du_type->unit());
					break;
			}
			break;
		}
		case Type::LIST: {
			auto	arr = std::static_pointer_cast<arrow::ListArray>(carray);
			auto	sub_array = arr->values();
			int32_t	start = arr->value_offset(rowid);
			int32_t	end   = arr->value_offset(rowid+1);

			std::cout << "{";
			for (int32_t k=start; k < end; k++)
			{
				if (k == start)
					std::cout << ",";
				process_one_token(sub_array, k, in_quote);
			}
			std::cout << "}";
			break;
		}
		case Type::LARGE_LIST: {
			auto	arr = std::static_pointer_cast<arrow::LargeListArray>(carray);
			auto	sub_array = arr->values();
			int64_t	start = arr->value_offset(rowid);
			int64_t	end   = arr->value_offset(rowid+1);

			std::cout << "{";
			for (int64_t k=start; k < end; k++)
			{
				if (k == start)
					std::cout << ",";
				process_one_token(sub_array, k, in_quote);
			}
			std::cout << "}";
			break;
		}
		case Type::FIXED_SIZE_LIST: {
			auto	arr = std::static_pointer_cast<arrow::FixedSizeListArray>(carray);
			auto	sub_array = arr->values();
			int32_t	unitsz = arr->value_length();
			int64_t	start = rowid * unitsz;
			int64_t	end = start + unitsz;

			std::cout << "{";
			for (int64_t k=start; k < end; k++)
			{
				if (k == start)
					std::cout << ",";
				process_one_token(sub_array, k, in_quote);
			}
			std::cout << "}";
			break;
		}
		case Type::STRUCT: {
			auto	arr = std::static_pointer_cast<arrow::StructArray>(carray);
			auto	st_type = std::static_pointer_cast<arrow::StructType>(arr->type());
			int		nfields = st_type->num_fields();

			std::cout << "\"(";
			for (int j=0; j < nfields; j++)
			{
				auto	sub_array = arr->field(j);
				if (j > 0)
					std::cout << ",";
				process_one_token(sub_array, rowid, true);
			}
			std::cout << ")\"";
			break;
		}
		case Type::DICTIONARY: {
			auto	arr = std::static_pointer_cast<arrow::DictionaryArray>(carray);
			auto	indices = arr->indices();
			auto	dict = arr->dictionary();
			int64_t	dindex;
			Result<std::shared_ptr<Scalar>> res;

			switch (indices->type_id())
			{
				case Type::INT8:
					dindex = std::static_pointer_cast<arrow::Int8Array>(indices)->Value(rowid);
					break;
				case Type::UINT8:
					dindex = std::static_pointer_cast<arrow::UInt8Array>(indices)->Value(rowid);
					break;
				case Type::INT16:
					dindex = std::static_pointer_cast<arrow::Int16Array>(indices)->Value(rowid);
					break;
				case Type::UINT16:
					dindex = std::static_pointer_cast<arrow::UInt16Array>(indices)->Value(rowid);
					break;
				case Type::INT32:
					dindex = std::static_pointer_cast<arrow::Int32Array>(indices)->Value(rowid);
					break;
				case Type::UINT32:
					dindex = std::static_pointer_cast<arrow::UInt32Array>(indices)->Value(rowid);
					break;
				case Type::INT64:
					dindex = std::static_pointer_cast<arrow::Int64Array>(indices)->Value(rowid);
					break;
				case Type::UINT64:
					dindex = std::static_pointer_cast<arrow::UInt64Array>(indices)->Value(rowid);
					break;
				default:
					Elog("Arrow::Dictionary has unsupported index type");
			}
			res = dict->GetScalar(dindex);
			if (!res.ok())
				Elog("unable to fetch values from dictionary (dindex=%ld)", dindex);
			std::cout << (*res)->ToString();
			break;
		}
		case Type::SPARSE_UNION:
			Elog("Arrow::SparseUnion type is not supported");
			break;
		case Type::DENSE_UNION:
			Elog("Arrow::DenseUnion type is not supported");
			break;
		case Type::MAP:
			Elog("Arrow::Map type is not supported");
			break;
		case Type::EXTENSION:
			Elog("Arrow::Extension type is not supported");
			break;
		default:
			Elog("unknown Apache Arrow type-id: %d", (int)carray->type_id());
			break;
	}
}

static bool
process_one_table(std::shared_ptr<arrow::Table> table)
{
	std::vector<std::shared_ptr<arrow::Array>> columns_array;
	auto	num_fields = table->num_columns();
	auto	num_rows   = table->num_rows();

	/* shortcut, if --offset wants to skip entire table */
	if (skip_offset >= num_rows)
	{
		skip_offset -= num_rows;
		return true;
	}

	for (int j=0; j < num_fields; j++)
	{
		auto	carray = table->column(j);
		/*
		 * The supplied table should be built for each RecordBatch
		 * or RowGroup, thus, ChunkedArray should have only one chunk.
		 */
		assert(carray->num_chunks() == 1);
		columns_array.push_back(carray->chunk(0));
	}

	for (int64_t i=skip_offset; i < num_rows; i++)
	{
		/* no more rows to dump */
		if (skip_limit == 0)
			return false;

		for (auto elem = columns_array.begin(); elem != columns_array.end(); elem++)
		{
			auto	carray = (*elem);

			/* add delimiter */
			if (elem != columns_array.begin())
				fputc(csv_mode ? ',' : '\t', output_filp);
			process_one_token(carray, i, false);
		}
		std::cout << std::endl;
		if (skip_limit > 0)
			skip_limit--;
	}
	return (skip_limit != 0);
}

static void
process_one_arrow_file(arrowReadableFile arrow_input_file, const char *filename)
{
	std::shared_ptr<ipc::RecordBatchFileReader> arrow_reader;
	std::shared_ptr<Schema> arrow_schema;
	static bool		is_first_call = true;

	{
		auto options = ipc::IpcReadOptions::Defaults();
		auto rv = ipc::RecordBatchFileReader::Open(arrow_input_file, options);

		if (!rv.ok())
			Elog("failed on arrow::ipc::RecordBatchFileReader::Open('%s'): %s",
				 filename, rv.status().ToString().c_str());
		arrow_reader = rv.ValueOrDie();
		arrow_schema = arrow_reader->schema();
	}
	/* shows CSV/TSV field names */
	if (shows_header)
		print_field_names(arrow_schema);
	else if (is_first_call && with_create_table)
		print_create_table(arrow_schema);
	is_first_call = false;

	/* read RecordBatch for each */
	auto num_record_batches = arrow_reader->num_record_batches();
	for (int k=0; k < num_record_batches; k++)
	{
		std::shared_ptr<RecordBatch>	rbatch;
		std::shared_ptr<Table>			table;
		{
			auto	rv = arrow_reader->ReadRecordBatch(k);
			if (!rv.ok())
				Elog("failed on arrow::ipc::RecordBatchFileReader::ReadRecordBatch: %s",
					 rv.status().ToString().c_str());
			rbatch = rv.ValueOrDie();
		}
		/*
		 * In the default, RecordBatch tries to load all the field unless
		 * we don't call RecordBatch::SelectColumns().
		 * So, here is no problems on our use cases.
		 */
		{
			auto	rv = arrow::Table::FromRecordBatches({rbatch});
			if (!rv.ok())
				Elog("failed on arrow::Table::FromRecordBatches(): %s",
					 rv.status().ToString().c_str());
			table = rv.ValueOrDie();
		}
		if (!process_one_table(table))
			break;		/* terminated by --limit */
	}
	printf("Schema:\n%s\n", arrow_schema->ToString().c_str());
}


#ifdef HAS_PARQUET
static void
process_one_parquet_file(arrowReadableFile arrow_input_file, const char *filename)
{
	std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
	std::shared_ptr<arrow::Schema> arrow_schema;
	Status		status;
	static bool	is_first_call = true;

	status = parquet::arrow::OpenFile(arrow_input_file,
									  default_memory_pool(),
									  &arrow_reader);
	if (!status.ok())
		Elog("failed on parquet::arrow::OpenFile('%s'): %s",
			 filename, status.ToString().c_str());

	status = arrow_reader->GetSchema(&arrow_schema);
	if (!status.ok())
		Elog("failed on parquet::arrow::FileReader::GetSchema('%s'): %s",
			 filename, status.ToString().c_str());
	//auto num_fields = arrow_schema->num_fields();
	
	
	if (shows_header)
		print_field_names(arrow_schema);
	else if (is_first_call && with_create_table)
		print_create_table(arrow_schema);
	is_first_call = false;

	/* read RowGroup for each */
	auto num_groups = arrow_reader->num_row_groups();
	for (int k=0; k < num_groups; k++)
	{
		std::shared_ptr<arrow::Table> table;

		status = arrow_reader->ReadRowGroup(k, &table);
		if (!status.ok())
			Elog("failed on parquet::arrow::FileReader::ReadRowGroup(%d): %s",
				 k, status.ToString().c_str());
		if (!process_one_table(table))
			break;		/* terminated by --limit */
	}
}
#endif

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
	exit(1);
}
#define usage()		__usage(NULL)

static void parse_options(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"output",  required_argument, NULL, 'o'},
		{"tsv",     no_argument,       NULL, 1001},
		{"csv",     no_argument,       NULL, 1002},
		{"header",  no_argument,       NULL, 1003},
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
	if (input_filenames.empty())
		Elog("no input files given");
	if (shows_header && with_create_table)
		Elog("--header and --create-table are mutually exclusive");
	if (with_tablespace && !with_create_table)
		Elog("--tablespace must be used with --create-table");
	if (with_partition_of && !with_create_table)
		Elog("--partition-of must be used with --create-table");
}

int main(int argc, char *argv[])
{
	parse_options(argc, argv);
	/* open the output stream (if not stdout) */
	if (output_filename)
	{
		if (std::freopen(output_filename, "wb", stdout) == NULL)
			Elog("failed on freopen('%s'): %m", output_filename);
	}
	/* process for each input files */
	for (auto cell = input_filenames.begin(); cell != input_filenames.end(); cell++)
	{
		Result<arrowReadableFile>rv;
		arrowReadableFile arrow_input_filp;
		const char *filename = (*cell).c_str();
		int			fdesc;
		bool		file_is_parquet	__attribute__((unused)) = false;
		char		magic[10];

		fdesc = open(filename, O_RDONLY);
		if (fdesc < 0)
			Elog("failed on open('%s'): %m", filename);
		/* determine the file type */
		if (pread(fdesc, magic, 6, 0) != 6)
			Elog("failed on pread('%s'): %m", filename);
		if (memcmp(magic, "ARROW1", 6) == 0)
			file_is_parquet = false;
#if HAS_PARQUET
		else if (memcmp(magic, "PAR1", 4) == 0)
			file_is_parquet = true;
#endif
		else
			Elog("input file '%s' is neither Arrow nor Parquet", filename);
		/* open the file stream */
		rv = io::ReadableFile::Open(fdesc);
		if (!rv.ok())
			Elog("failed on arrow::io::FileOutputStream::Open('%s'): %s",
				 filename, rv.status().ToString().c_str());
		arrow_input_filp = rv.ValueOrDie();
		/* process one Arrow or Parquet file */
#if HAS_PARQUET
		if (file_is_parquet)
		{
			process_one_parquet_file(arrow_input_filp, filename);
		}
		else
#endif
		{
			process_one_arrow_file(arrow_input_filp, filename);
		}
		/* close the file stream */
		arrow_input_filp->Close();
	}
	return 0;
}
