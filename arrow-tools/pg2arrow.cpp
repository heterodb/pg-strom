/*
 * pg2arrow
 *
 * A tool to dump PostgreSQL database for Apache Arrow/Parquet format.
 * ---
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <iostream>
#include <ctype.h>
#include <endian.h>
#include <fcntl.h>
#include <getopt.h>
#include <mutex>
#include <pthread.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <arrow/api.h>				/* dnf install arrow-devel, or apt install libarrow-dev */
#include <arrow/array.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <parquet/arrow/reader.h>	/* dnf install parquet-devel, or apt install libparquet-dev */
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/schema.h>
#include <parquet/statistics.h>
#include <parquet/stream_writer.h>
#include <libpq-fe.h>
#include "arrow_defs.h"

// ------------------------------------------------
// Error Reporting
// ------------------------------------------------
#define Elog(fmt,...)									\
	do {												\
		if (verbose)									\
			fprintf(stderr, "[ERROR %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
		else											\
			fprintf(stderr, "[ERROR] " fmt "\n",		\
					##__VA_ARGS__);						\
		exit(1);										\
	} while(0)
#define Info(fmt,...)									\
	do {												\
		if (verbose)									\
			fprintf(stderr, "[INFO %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
		else if (verbose > 0)							\
			fprintf(stderr, "[INFO] " fmt "\n",			\
					##__VA_ARGS__);						\
	} while(0)

#define Debug(fmt,...)									\
	do {												\
		if (verbose)									\
			fprintf(stderr, "[DEBUG %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
	} while(0)

// ------------------------------------------------
// Type definitions
// ------------------------------------------------
struct compressOption
{
	const char	   *colname;	/* may be NULL */
	arrow::Compression::type method;
};
using compressOption = struct compressOption;

struct configOption
{
	const char	   *name;
	const char	   *value;
};
using configOption	= struct configOption;

using	arrowField		= std::shared_ptr<arrow::Field>;
using	arrowBuilder	= std::shared_ptr<arrow::ArrayBuilder>;
using	arrowBuilderVector = std::vector<arrowBuilder>;
using	arrowArray		= std::shared_ptr<arrow::Array>;
using	arrowArrayVector = std::vector<arrowArray>;
using	pgsqlHandler		= std::shared_ptr<class pgsqlBinaryHandler>;
using	pgsqlHandlerVector	= std::vector<pgsqlHandler>;

// ------------------------------------------------
// Command line options
// ------------------------------------------------
static const char  *pgsql_hostname = NULL;			/* -h, --host */
static const char  *pgsql_port_num = NULL;			/* -p, --port */
static const char  *pgsql_username = NULL;			/* -u, --user */
static const char  *pgsql_password = NULL;
static const char  *pgsql_database = NULL;			/* -d, --database */
static int			pgsql_command_id = 0;
static std::vector<std::string> pgsql_command_list;
static const char  *raw_pgsql_command = NULL;		/* -c, --command */
static uint32_t		num_worker_threads = 0;			/* -n, --num-workers */
static const char  *ctid_target_table = NULL;	/*     --ctid-target */
static const char  *parallel_keys;					/* -k, --parallel-keys */
static bool			parquet_mode = false;			/* -q, --parquet */
static const char  *output_filename = NULL;			/* -o, --output */
static const char  *stat_embedded_columns = NULL;	/* -S, --stat */
static const char  *flatten_composite_columns = NULL; /* --flatten */
static size_t		batch_segment_sz = 0;			/* -s, --segment-size */
static std::vector<const compressOption *> compression_methods;	/* -C, --compress */
static const char  *dump_meta_filename = NULL;		/* --meta */
static const char  *dump_schema_filename = NULL;	/* --schema */
static const char  *dump_schema_tablename = NULL;	/* --schema-name */
static bool			shows_progress = false;			/* --progress */
static int			verbose;						/* --verbose */
static std::vector<configOption>	pgsql_config_options;	/* --set */

// ------------------------------------------------
// Other static variables
// ------------------------------------------------
static volatile bool	worker_setup_done  = false;
static pthread_mutex_t	worker_setup_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	worker_setup_cond  = PTHREAD_COND_INITIALIZER;
static std::vector<pthread_t>			worker_threads;
static std::vector<pgsqlHandlerVector>	worker_handlers;
static const char	   *snapshot_identifier = NULL;
static std::shared_ptr<arrow::Schema>					arrow_schema = NULL;
static std::shared_ptr<arrow::io::FileOutputStream>		arrow_out_stream;
static std::shared_ptr<arrow::ipc::RecordBatchWriter>	arrow_file_writer;
static std::unique_ptr<parquet::arrow::FileWriter>		parquet_file_writer;
static std::mutex		arrow_file_mutex;
static uint32_t			arrow_num_chunks = 0;
static uint64_t			arrow_num_items = 0;

// ------------------------------------------------
// Thread local variables
// ------------------------------------------------
static thread_local	PGconn	   *pgsql_conn = NULL;
static thread_local PGresult   *pgsql_res = NULL;
static thread_local uint32_t	pgsql_res_index = 0;
static thread_local uint32_t	worker_id = -1;
static thread_local const char *server_timezone = NULL;

// ------------------------------------------------
// pgsqlBinaryHandler - the base class
// ------------------------------------------------
class pgsqlBinaryHandler
{
public:
	std::string		attname;
	std::string		typname;
	Oid				type_oid;
	int32_t			type_mod;
	char			type_align;
	bool			type_byval;
	int				type_len;
	bool			stats_enabled;
	bool			composite_flatten;
	arrowBuilder	arrow_builder;
	arrowArray		arrow_array;
	virtual size_t	chunkSize(void) = 0;
	virtual size_t	putValue(const char *value, int sz) = 0;
	virtual size_t	moveValue(pgsqlHandler buddy, int64_t index) = 0;
	virtual void	enableStats(void)
	{
		Info("-S,--stat tried to enable min/max statistics on the field '%s' (%s), but not supported",
			 attname.c_str(), typname.c_str());
	}
	virtual bool	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		return false;
	}
	virtual void	appendArrays(arrowArrayVector &arrow_arrays_vector)
	{
		if (!arrow_array)
			Elog("Bug? appendArrays() must be called after Finish()");
		arrow_arrays_vector.push_back(arrow_array);
	}
	virtual int64_t	Finish(void)
	{
		auto	rv = arrow_builder->Finish(&arrow_array);
		if (!rv.ok())
			Elog("failed on arrow::ArrayBuilder::Finish: %s",
				 rv.ToString().c_str());
		return arrow_array->length();
	}
	virtual void	Reset(void)
	{
		arrow_builder->Reset();
		if (arrow_array)
			arrow_array = nullptr;
	}
};

// ------------------------------------------------
// Misc utility functions
// ------------------------------------------------
#define Max(a,b)		((a)>(b) ? (a) : (b))
#define Min(a,b)		((a)<(b) ? (a) : (b))
#define BITMAPLEN(NITEMS)		(((NITEMS) + 7) / 8)
#define ARROW_ALIGN(LEN)		(((uintptr_t)(LEN) + 63UL) & ~63UL)

static inline void *
palloc(size_t sz)
{
	void   *p = malloc(sz);

	if (!p)
		Elog("out of memory");
	return p;
}

static inline void *
palloc0(size_t sz)
{
	void   *p = malloc(sz);

	if (!p)
		Elog("out of memory");
	memset(p, 0, sz);
	return p;
}

static inline char *
pstrdup(const char *str)
{
	char   *p = strdup(str);

	if (!p)
		Elog("out of memory");
	return p;
}

static inline void *
repalloc(void *old, size_t sz)
{
	void   *p = realloc(old, sz);

	if (!p)
		Elog("out of memory");
	return p;
}

static inline void
pfree(void *ptr)
{
	free(ptr);
}

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

static inline const char *
__quote_ident(const char *ident, char *buffer)
{
	bool	safe = true;
	char   *wpos = buffer;

	*wpos++ = '"';
	for (const char *rpos = ident; *rpos; rpos++)
	{
		char	c = *rpos;

		if (!islower(c) && !isdigit(c) && c != '_')
		{
			if (c == '"')
				*wpos++ = '"';
			safe = false;
		}
		*wpos++ = c;
	}
	if (safe)
		return ident;
	*wpos++ = '"';
	*wpos++ = '\0';
	return buffer;
}
#define quote_ident(ident)						\
	__quote_ident((ident),(char *)alloca(2*strlen(ident)+20))

static char *
__read_file(const char *filename)
{
	struct stat	stat_buf;
	int		fdesc;
	loff_t	off = 0;
	char   *buffer;

	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
		Elog("failed on open('%s'): %m", filename);
	if (fstat(fdesc, &stat_buf) != 0)
		Elog("failed on fstat('%s'): %m", filename);
	buffer = (char *)palloc(stat_buf.st_size + 1);
	while (off < stat_buf.st_size)
	{
		ssize_t	nbytes = read(fdesc, buffer + off, stat_buf.st_size - off);
		if (nbytes < 0)
		{
			if (errno == EINTR)
				continue;
			Elog("failed on read('%s'): %m", filename);
		}
		else if (nbytes == 0)
			Elog("unexpected EOF at %lu of '%s'", off, filename);
		off += nbytes;
	}
	buffer[stat_buf.st_size] = '\0';
	close(fdesc);

	return buffer;
}

static void
__replace_string(std::string &str,
				 const std::string &from,
				 const std::string &to)
{
	size_t	pos = 0;

	while ((pos = str.find(from, pos)) != std::string::npos)
	{
		str.replace(pos, from.length(), to);
		pos += to.length();
	}
}

#include <arrow/array/array_decimal.h>
// ================================================================
//
// Type specific PostgreSQL handlers
//
// ================================================================
template <typename BUILDER_TYPE, typename ARRAY_TYPE, typename C_TYPE>
class pgsqlBinaryScalarHandler : public pgsqlBinaryHandler
{
public:
	virtual C_TYPE fetchBinary(const char *addr, int sz) = 0;
	virtual C_TYPE fetchArray(arrowArray __array, int64_t index) = 0;
	virtual C_TYPE updateStats(const C_TYPE &datum) = 0;
	size_t	putValue(const char *addr, int sz)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);
		arrow::Status rv;

		if (!addr)
			rv = builder->AppendNull();
		else
			rv = builder->Append(updateStats(fetchBinary(addr,sz)));
		if (!rv.ok())
			Elog("unable to put value to '%s' field (%s): %s",
				 attname.c_str(), typname.c_str(), rv.ToString().c_str());
		return chunkSize();
	}
	size_t	moveValue(pgsqlHandler buddy, int64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy->arrow_array);
		arrow::Status rv;

		assert(index < array->length());
		if (array->IsNull(index))
			rv = builder->AppendNull();
		else
			rv = builder->Append(updateStats(fetchArray(buddy->arrow_array, index)));
		if (!rv.ok())
			Elog("unable to put value to '%s' field (%s): %s",
				 attname.c_str(), typname.c_str(), rv.ToString().c_str());
		return chunkSize();
	}
};

template <typename BUILDER_TYPE, typename ARRAY_TYPE, typename C_TYPE>
class __pgsqlBinaryBitmapHandler : public pgsqlBinaryScalarHandler<BUILDER_TYPE,ARRAY_TYPE,C_TYPE>
{
public:
	size_t	chunkSize(void)
	{
		size_t	sz = ARROW_ALIGN(BITMAPLEN(this->arrow_builder->length()));
		if (this->arrow_builder->null_count() > 0)
			sz += ARROW_ALIGN(BITMAPLEN(this->arrow_builder->length()));
		return sz;
	}
};
#define pgsqlBinaryBitmapHandler(TYPE_PREFIX,C_TYPE)				\
	__pgsqlBinaryBitmapHandler<arrow::TYPE_PREFIX##Builder,arrow::TYPE_PREFIX##Array,C_TYPE>

template <typename BUILDER_TYPE, typename ARRAY_TYPE, typename C_TYPE>
class __pgsqlBinaryInlineHandler : public pgsqlBinaryScalarHandler<BUILDER_TYPE,ARRAY_TYPE,C_TYPE>
{
public:
	size_t	unitsz;
	size_t	chunkSize(void)
	{
		size_t	sz = ARROW_ALIGN(unitsz * this->arrow_builder->length());
		if (this->arrow_builder->null_count() > 0)
			sz += ARROW_ALIGN(BITMAPLEN(this->arrow_builder->length()));
		return sz;
	}
};
#define pgsqlBinaryInlineHandler(TYPE_PREFIX,C_TYPE)				\
	__pgsqlBinaryInlineHandler<arrow::TYPE_PREFIX##Builder,arrow::TYPE_PREFIX##Array,C_TYPE>

class pgsqlBinaryVarlenaHandler : public pgsqlBinaryHandler
{
public:
	size_t	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::BinaryBuilder>(arrow_builder);
		size_t	sz = (ARROW_ALIGN(sizeof(uint32_t) * builder->length()) +
					  ARROW_ALIGN(builder->value_data_length()));
		if (builder->null_count() > 0)
			sz += ARROW_ALIGN(BITMAPLEN(builder->length()));
		return sz;
	}
};

class pgsqlBinaryLargeVarlenaHandler : public pgsqlBinaryHandler
{
public:
	size_t	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::LargeBinaryBuilder>(arrow_builder);
		size_t	sz = (ARROW_ALIGN(sizeof(uint64_t) * builder->length()) +
					  ARROW_ALIGN(builder->value_data_length()));
		if (builder->null_count() > 0)
			sz += ARROW_ALIGN(BITMAPLEN(builder->length()));
		return sz;
	}
};

template <typename BUILDER_TYPE, typename ARRAY_TYPE, typename C_TYPE>
class __pgsqlBinarySimpleInlineHandler : public __pgsqlBinaryInlineHandler<BUILDER_TYPE,ARRAY_TYPE,C_TYPE>
{
public:
	bool	stats_is_valid;
	C_TYPE	stats_min_value;
	C_TYPE	stats_max_value;
	__pgsqlBinarySimpleInlineHandler()
	{
		stats_is_valid = false;
	}
	C_TYPE fetchArray(arrowArray __array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(__array);

		return array->Value(index);
	}
	C_TYPE updateStats(const C_TYPE &datum)
	{
		if (this->stats_enabled)
		{
			if (!stats_is_valid)
			{
				stats_min_value = datum;
				stats_max_value = datum;
				stats_is_valid  = true;
			}
			else
			{
				if (stats_min_value > datum)
					stats_min_value = datum;
				if (stats_max_value < datum)
					stats_max_value = datum;
			}
		}
		return datum;
	}
	void	enableStats(void)
	{
		this->stats_enabled = true;
	}
	bool	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		bool	retval = stats_is_valid;

		if (stats_is_valid)
		{
			custom_metadata->Append(std::string("min_max_stats.") + this->attname,
									std::to_string(stats_min_value) +
									std::string(",") +
									std::to_string(stats_max_value));
			stats_is_valid = false;
		}
		return retval;
	}
};
#define pgsqlBinarySimpleInlineHandler(TYPE_PREFIX,C_TYPE)				\
	__pgsqlBinarySimpleInlineHandler<arrow::TYPE_PREFIX##Builder,arrow::TYPE_PREFIX##Array,C_TYPE>

#define PGSQL_BINARY_PUT_VALUE_TEMPLATE(TYPE_PREFIX)					\
	size_t	putValue(const char *addr, int sz)							\
	{																	\
		auto	builder = std::dynamic_pointer_cast<arrow::TYPE_PREFIX##Builder>(arrow_builder); \
		arrow::Status rv;												\
		assert(builder != NULL);										\
		if (!addr)														\
			rv = builder->AppendNull();									\
		else															\
			rv = builder->Append(fetchBinary(addr,sz));					\
		if (!rv.ok())													\
			Elog("unable to put value to '%s' field (%s): %s",			\
				 attname.c_str(), #TYPE_PREFIX, rv.ToString().c_str());	\
		return chunkSize();												\
	}
#define PGSQL_BINARY_MOVE_VALUE_TEMPLATE(TYPE_PREFIX)					\
	size_t	moveValue(pgsqlHandler buddy, int64_t index)				\
	{																	\
		auto	builder = std::dynamic_pointer_cast<arrow::TYPE_PREFIX##Builder>(arrow_builder); \
		auto	array = std::dynamic_pointer_cast<arrow::TYPE_PREFIX##Array>(buddy->arrow_array); \
		arrow::Status rv;												\
		assert(builder != NULL && array != NULL);						\
		assert(index < array->length());								\
		if (array->IsNull(index))										\
			rv = builder->AppendNull();									\
		else															\
			rv = builder->Append(array->Value(index));					\
		if (!rv.ok())													\
			Elog("unable to move value in '%s' field (%s): %s",			\
				 attname.c_str(), #TYPE_PREFIX, rv.ToString().c_str());	\
		return chunkSize();												\
	}

class pgsqlBoolHandler final : public pgsqlBinaryBitmapHandler(Boolean,bool)
{
public:
	bool	updateStats(const bool &datum)
	{
		return datum;
	}
	bool	fetchBinary(const char *addr, int sz)
	{
		assert(sz == sizeof(char));
		return *addr ? true : false;
	}
	bool	fetchArray(arrowArray __array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<arrow::BooleanArray>(__array);

		return array->Value(index);
	}
};

class pgsqlInt8Handler final : public pgsqlBinarySimpleInlineHandler(Int8,int8_t)
{
public:
	pgsqlInt8Handler()
	{
		unitsz = sizeof(int8_t);
	}
	int8_t	fetchBinary(const char *addr, int sz)
	{
		assert(sz == sizeof(int8_t));
		return *((const int8_t *)addr);
	}
};

class pgsqlInt16Handler final : public pgsqlBinarySimpleInlineHandler(Int16,int16_t)
{
public:
	pgsqlInt16Handler()
	{
		unitsz = sizeof(int16_t);
	}
	int16_t	fetchBinary(const char *addr, int sz)
	{
		assert(sz == sizeof(int16_t));
		return be16toh(*((const int16_t *)addr));
	}
};

class pgsqlInt32Handler final : public pgsqlBinarySimpleInlineHandler(Int32,int32_t)
{
public:
	pgsqlInt32Handler()
	{
		unitsz = sizeof(int32_t);
	}
	int32_t	fetchBinary(const char *addr, int sz)
	{
		assert(sz == sizeof(int32_t));
		return be32toh(*((const int32_t *)addr));
	}
};

class pgsqlInt64Handler final : public pgsqlBinarySimpleInlineHandler(Int64,int64_t)
{
public:
	pgsqlInt64Handler()
	{
		unitsz = sizeof(int64_t);
	}
	int64_t	fetchBinary(const char *addr, int sz)
	{
		assert(sz == sizeof(int64_t));
		return be64toh(*((const int64_t *)addr));
	}
};

class pgsqlFloat16Handler final : public pgsqlBinarySimpleInlineHandler(HalfFloat,uint16_t)
{
public:
	pgsqlFloat16Handler()
	{
		unitsz = sizeof(uint16_t);
	}
	uint16_t	fetchBinary(const char *addr, int sz)
	{
		assert(sz == sizeof(uint16_t));
		return be16toh(*((const uint16_t *)addr));
	}
};

class pgsqlFloat32Handler final : public pgsqlBinarySimpleInlineHandler(Float,float)
{
public:
	pgsqlFloat32Handler()
	{
		unitsz = sizeof(float);
	}
	float	fetchBinary(const char *addr, int sz)
	{
		union {
			uint32_t	ival;
			float		fval;
		} u;
		u.ival = be32toh(*((const uint32_t *)addr));
		return u.fval;
	}
};

class pgsqlFloat64Handler final : public pgsqlBinarySimpleInlineHandler(Double,double)
{
public:
	pgsqlFloat64Handler()
	{
		unitsz = sizeof(double);
	}
	double fetchBinary(const char *addr, int sz)
	{
		union {
			uint64_t	ival;
			double		fval;
		} u;
		u.ival = be64toh(*((const uint64_t *)addr));
		return u.fval;
	}
};

/* parameters of Numeric type */
#define NUMERIC_DSCALE_MASK	0x3FFF
#define NUMERIC_SIGN_MASK	0xC000
#define NUMERIC_POS			0x0000
#define NUMERIC_NEG			0x4000
#define NUMERIC_NAN			0xC000
#define NUMERIC_PINF		0xD000
#define NUMERIC_NINF		0xF000

#define NBASE				10000
#define DEC_DIGITS			4

typedef struct
{
	uint16_t	ndigits;	/* number of digits */
	uint16_t	weight;		/* weight of first digit */
	uint16_t	sign;		/* NUMERIC_(POS|NEG|NAN) */
	uint16_t	dscale;		/* display scale */
	int16_t		digits[1];	/* numeric digists */
} __pgsql_numeric_binary;

class pgsqlNumericHandler final : public pgsqlBinaryInlineHandler(Decimal128,arrow::Decimal128)
{
public:
	bool	stats_is_valid;
	arrow::Decimal128	stats_min_value;
	arrow::Decimal128	stats_max_value;
	pgsqlNumericHandler()
	{
		stats_is_valid = false;
		unitsz = sizeof(__int128_t);
	}
	arrow::Decimal128 updateStats(const arrow::Decimal128 &datum)
	{
		if (stats_enabled)
		{
			if (!stats_is_valid)
			{
				stats_min_value = datum;
				stats_max_value = datum;
				stats_is_valid = true;
			}
			else
			{
				if (stats_min_value > datum)
					stats_min_value = datum;
				if (stats_max_value < datum)
					stats_max_value = datum;
			}
		}
		return datum;
	}
	arrow::Decimal128 fetchBinary(const char *addr, int sz)
	{
		auto	rawdata = (const __pgsql_numeric_binary *)addr;
		auto	builder = std::dynamic_pointer_cast<arrow::Decimal128Builder>(arrow_builder);
		auto	d_type = std::static_pointer_cast<arrow::Decimal128Type>(builder->type());
		int		d_scale = d_type->scale();
		int		ndigits = (int16_t)be16toh(rawdata->ndigits);
		int		weight = (int16_t)be16toh(rawdata->weight) + 1;
		int		sign = (int16_t)be16toh(rawdata->sign);
		int		diff = DEC_DIGITS * (ndigits - weight) - d_scale;
		arrow::Decimal128 value(0);
		static int __pow10[] = {
			1,				/* 10^0 */
			10,				/* 10^1 */
			100,			/* 10^2 */
			1000,			/* 10^3 */
			10000,			/* 10^4 */
			100000,			/* 10^5 */
			1000000,		/* 10^6 */
			10000000,		/* 10^7 */
			100000000,		/* 10^8 */
			1000000000,		/* 10^9 */
		};
		if ((sign & NUMERIC_SIGN_MASK) == NUMERIC_SIGN_MASK)
			Elog("Decimal128 cannot map NaN, +Inf or -Inf in PostgreSQL Numeric");
		/* can reduce some digits pgsql numeric scale is larger than arrow decimal's scale */
		if (diff >= DEC_DIGITS)
		{
			ndigits -= (diff / DEC_DIGITS);
			diff = (diff % DEC_DIGITS);
		}
		assert(diff < DEC_DIGITS);
		for (int i=0; i < ndigits; i++)
		{
			int		dig = be16toh(rawdata->digits[i]);
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is corrupted (out of range: %d)", dig);
			value = NBASE * value + dig;
		}
		/* adjust to the decimal128 */
		while (diff > 0)
		{
			int		k = Min(9, diff);
			value /= __pow10[k];
			diff -= k;
		}
		while (diff < 0)
		{
			int		k = Min(9, -diff);
			value *= __pow10[k];
			diff += k;
		}
		/* is it a negative value? */
		if ((sign & NUMERIC_NEG) != 0)
			value = -value;
		return value;
	}
	arrow::Decimal128 fetchArray(arrowArray array, int64_t index)
	{
		arrow::Decimal128 hoge(123);
		return hoge;
	}
	void enableStats(void)
    {
		this->stats_enabled = true;
    }
    bool appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
    {
		bool  retval = this->stats_is_valid;

		if (this->stats_enabled && this->stats_is_valid)
		{
			auto	builder = std::dynamic_pointer_cast<arrow::Decimal128Builder>(this->arrow_builder);
			auto	d_type = std::static_pointer_cast<arrow::Decimal128Type>(builder->type());
			int		scale = d_type->scale();
			custom_metadata->Append(std::string("min_max_stats.") + this->attname,
									this->stats_min_value.ToString(scale) +
									std::string(",") +
									this->stats_max_value.ToString(scale));
			this->stats_is_valid = false;
        }
        return retval;
    }
};

#define UNIX_EPOCH_JDATE		2440588UL	/* 1970-01-01 */
#define POSTGRES_EPOCH_JDATE	2451545UL	/* 2000-01-01 */
#define USECS_PER_DAY			86400000000UL

class pgsqlDateHandler final : public pgsqlBinarySimpleInlineHandler(Date32,int32_t)
{
public:
	pgsqlDateHandler()
	{
		unitsz = sizeof(int32_t);
	}
	int32_t	fetchBinary(const char *addr, int sz)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::Date32Builder>(arrow_builder);
		auto	d_type = std::static_pointer_cast<arrow::Date32Type>(builder->type());
		int32_t	value;

		assert(sz == sizeof(int32_t) && d_type->unit() == arrow::DateUnit::DAY);
		value = be32toh(*((const uint32_t *)addr));
		return value + (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
	}
};

class pgsqlTimeHandler final : public pgsqlBinarySimpleInlineHandler(Time64,int64_t)
{
public:
	pgsqlTimeHandler()
	{
		unitsz = sizeof(int64_t);
	}
	int64_t	fetchBinary(const char *addr, int sz)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::Time64Builder>(arrow_builder);
		auto	d_type = std::static_pointer_cast<arrow::Time64Type>(builder->type());
		int64_t	value;

		assert(sz == sizeof(int64_t) && d_type->unit() == arrow::TimeUnit::MICRO);
		value = be64toh(*((const uint64_t *)addr));
		return value;
	}
};

class pgsqlTimestampHandler final : public pgsqlBinarySimpleInlineHandler(Timestamp,int64_t)
{
public:
	pgsqlTimestampHandler()
	{
		unitsz = sizeof(int64_t);
	}
	int64_t	fetchBinary(const char *addr, int sz)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::TimestampBuilder>(arrow_builder);
		auto	d_type = std::static_pointer_cast<arrow::TimestampType>(builder->type());
		int64_t	value;

		assert(sz == sizeof(uint64_t) && d_type->unit() == arrow::TimeUnit::MICRO);
		value = be64toh(*((const uint64_t *)addr));
		return value + (POSTGRES_EPOCH_JDATE -
						UNIX_EPOCH_JDATE) * USECS_PER_DAY;
	}
};

typedef struct
{
	int64_t		time;
	int32_t		days;
	int32_t		months;
} __pgsql_interval_binary;

class pgsqlIntervalHandler final : public pgsqlBinaryInlineHandler(MonthDayNanoInterval,
																   arrow::MonthDayNanoIntervalType::MonthDayNanos)
{
public:
	pgsqlIntervalHandler()
	{
		unitsz = sizeof(arrow::MonthDayNanoIntervalType::MonthDayNanos);
	}
	arrow::MonthDayNanoIntervalType::MonthDayNanos updateStats(const arrow::MonthDayNanoIntervalType::MonthDayNanos &datum)
	{
		return datum;
	}
	arrow::MonthDayNanoIntervalType::MonthDayNanos fetchBinary(const char *addr, int sz)
	{
		auto	rawdata = (const __pgsql_interval_binary *)addr;
		arrow::MonthDayNanoIntervalType::MonthDayNanos iv;
		assert(sz == sizeof(__pgsql_interval_binary));
		iv.months = be32toh(rawdata->months);
		iv.days   = be32toh(rawdata->days);
		iv.nanoseconds = be64toh(rawdata->time) * 1000L;
		return iv;
	}
	arrow::MonthDayNanoIntervalType::MonthDayNanos fetchArray(arrowArray __array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<arrow::MonthDayNanoIntervalArray>(__array);

		return array->Value(index);
	}
};

class pgsqlTextHandler final : public pgsqlBinaryVarlenaHandler
{
public:
	std::string_view fetchBinary(const char *addr, int sz)
	{
		return std::string_view(addr, sz);
	}
	PGSQL_BINARY_PUT_VALUE_TEMPLATE(String)
	PGSQL_BINARY_MOVE_VALUE_TEMPLATE(String)
};

class pgsqlBpCharHandler final : public pgsqlBinaryInlineHandler(FixedSizeBinary,std::string)
{
public:
	pgsqlBpCharHandler(size_t width)
	{
		unitsz = width;
	}
	std::string updateStats(const std::string &datum)
	{
		return datum;
	}
	std::string fetchBinary(const char *addr, int sz)
	{
		auto	buf = std::string(addr, Min(sz, unitsz));

		while (sz < unitsz)
		{
			buf += ' ';
			sz++;
		}
		return buf;
	}
	std::string	fetchArray(arrowArray __array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(__array);
		const char *addr = (const char *)array->Value(index);

		return std::string(addr, unitsz);
	}
};

class pgsqlByteaHandler final : public pgsqlBinaryVarlenaHandler
{
public:
	std::string_view fetchBinary(const char *addr, int sz)
	{
		return std::string_view(addr, sz);
	}
	PGSQL_BINARY_PUT_VALUE_TEMPLATE(Binary)
	PGSQL_BINARY_MOVE_VALUE_TEMPLATE(Binary)
};

/* see array_send() in PostgreSQL */
typedef struct {
	int32_t		ndim;
	int32_t		hasnull;
	int32_t		element_oid;
	struct {
		int32_t	sz;
		int32_t	lb;
	} dim[1];
} __pgsql_array_binary;

class pgsqlListHandler final : public pgsqlBinaryHandler
{
	pgsqlHandler	element;
	Oid				element_oid;
public:
	pgsqlListHandler(pgsqlHandler __element, Oid __element_oid)
	{
		element = __element;
		element_oid = __element_oid;
	}
	size_t  chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::ListBuilder>(arrow_builder);
		size_t  sz = ARROW_ALIGN(sizeof(uint32_t) * (builder->length() + 1));

		if (builder->null_count() > 0)
			sz += ARROW_ALIGN(BITMAPLEN(builder->length()));
		return sz + element->chunkSize();
	}
	size_t	putValue(const char *addr, int sz)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::ListBuilder>(arrow_builder);
		arrow::Status rv;

		assert(builder != NULL);
		if (!addr)
		{
			rv = builder->AppendNull();
			if (!rv.ok())
				Elog("unable to put NULL to '%s' field (List): %s",
					 attname.c_str(), rv.ToString().c_str());
		}
		else
		{
			auto   *rawdata = (const __pgsql_array_binary *)addr;
			int		ndim = be32toh(rawdata->ndim);
			//bool	hasnull = (be32toh(rawdata->hasnull) != 0);
			int		nitems;
			int		elem_sz;
			const char *pos;

			if (be32toh(rawdata->element_oid) != element_oid)
				Elog("PostgreSQL array element type mismatch");
			if (ndim != 1)
				Elog("pg2arrow supports only 1-dimensional PostgreSQL array (ndim=%d)", ndim);
			nitems = be32toh(rawdata->dim[0].sz);
			rv = builder->Append();
			if (!rv.ok())
				Elog("unable to put value to '%s' field (List): %s",
					 attname.c_str(), rv.ToString().c_str());
			pos = (const char *)&rawdata->dim[ndim];
			for (int i=0; i < nitems; i++)
			{
				if (pos + sizeof(int32_t) > addr + sz)
					Elog("out of range - binary array has corrupted");
				elem_sz = be32toh(*((int32_t *)pos));
				pos += sizeof(int32_t);
				if (elem_sz < 0)
					element->putValue(NULL, 0);
				else
				{
					element->putValue(pos, elem_sz);
					pos += elem_sz;
				}
			}
		}
		return chunkSize();
	}
	size_t	moveValue(pgsqlHandler __buddy, int64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::ListBuilder>(arrow_builder);
		auto	buddy = std::dynamic_pointer_cast<pgsqlListHandler>(__buddy);
		auto	array = std::dynamic_pointer_cast<arrow::ListArray>(buddy->arrow_array);
		arrow::Status rv;

		assert(index < array->length());
		if (array->IsNull(index))
		{
			rv = builder->AppendNull();
			if (!rv.ok())
				Elog("unable to put NULL to '%s' field (List): %s",
					 attname.c_str(), rv.ToString().c_str());
		}
		else
		{
			int32_t		head = array->value_offset(index);
			int32_t		tail = array->value_offset(index+1);

			assert(head <= tail);
			rv = builder->Append();
			if (!rv.ok())
				Elog("unable to move value in '%s' field (List): %s",
					 attname.c_str(), rv.ToString().c_str());
			for (int32_t curr=head; curr < tail; curr++)
			{
				/* move and item from buddy's element to my element */
				element->moveValue(buddy->element, curr);
			}
		}
		return chunkSize();
	}
};

class pgsqlStructHandler final : public pgsqlBinaryVarlenaHandler
{
	pgsqlHandlerVector	children;
public:
	pgsqlStructHandler(pgsqlHandlerVector __children)
	{
		children = __children;
	}
	size_t	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StructBuilder>(arrow_builder);
		size_t	chunk_sz = 0;

		if (builder->null_count() > 0)
			chunk_sz += ARROW_ALIGN(BITMAPLEN(builder->length()));
		for (auto cell = children.begin(); cell != children.end(); cell++)
			chunk_sz += (*cell)->chunkSize();
		return chunk_sz;
	}
	size_t	putValue(const char *addr, int sz)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StructBuilder>(arrow_builder);
		const char *pos = addr;
		arrow::Status rv;
		int		nvalids;

		/* fillup by NULL */
		if (!addr)
		{
			rv = builder->AppendNull();
			if (!rv.ok())
				Elog("unable to put NULL to '%s' field (Struct): %s",
					 attname.c_str(), rv.ToString().c_str());
			for (int j=0; j < children.size(); j++)
				children[j]->putValue(NULL, 0);
		}
		else
		{
			rv = builder->Append();
			if (!rv.ok())
				Elog("unable to put values to '%s' field (Struct): %s",
					 attname.c_str(), rv.ToString().c_str());
			if (sz < sizeof(int32_t))
				Elog("out of range - binary composite has corrupted");
			nvalids = be32toh(*((int32_t *)pos));
			pos += sizeof(int32_t);
			for (int j=0; j < children.size(); j++)
			{
				auto	child = children[j];

				if (j >= nvalids)
					child->putValue(NULL, 0);
				else if (pos + 2*sizeof(int32_t) >= addr + sz)
					Elog("out of range - binary composite has corrupted");
				else
				{
					//Oid		item_oid = be32toh(((uint32_t *)pos)[0]);
					int		item_sz  = be32toh(((int32_t *)pos)[1]);

					pos += 2*sizeof(int32_t);
					if (item_sz < 0)
						child->putValue(NULL, 0);
					else // if (item_oid == child->type_oid)
					{
						child->putValue(pos, item_sz);
						pos += item_sz;
					}
					/*
					 * MEMO: record_send() may use different type to pack
					 * sub-field values, and often item_oid is not identical
					 * with the type defined in the system catalog.
					 * We need further investigation what is the best way
					 * to map sub-field type into arrow/parquet types.
					 */
				}
			}
		}
		return chunkSize();
	}
	size_t	moveValue(pgsqlHandler __buddy, int64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StructBuilder>(arrow_builder);
		auto	buddy = std::dynamic_pointer_cast<pgsqlStructHandler>(__buddy);
		auto	array = std::dynamic_pointer_cast<arrow::StructArray>(buddy->arrow_array);
		arrow::Status rv;

		assert(index < array->length());
		assert(children.size() == buddy->children.size());
		if (array->IsNull(index))
		{
			rv = builder->AppendNull();
			if (!rv.ok())
				Elog("unable to put NULL to '%s' field (Struct): %s",
					 attname.c_str(), rv.ToString().c_str());
			for (int j=0; j < children.size(); j++)
			{
				auto	child = children[j];

				child->putValue(NULL, 0);
			}
		}
		else
		{
			rv = builder->Append();
			if (!rv.ok())
				Elog("unable to put valid-bit to '%s' field (Struct): %s",
					 attname.c_str(), rv.ToString().c_str());
			for (int j=0; j < children.size(); j++)
			{
				auto	child = children[j];
				auto	buddy_child = buddy->children[j];

				child->moveValue(buddy_child, index);
			}
		}
		return chunkSize();
	}
	void	appendArrays(arrowArrayVector &arrow_arrays_vector)
	{
		if (!arrow_array)
			Elog("Bug? appendArrays() must be called after Finish()");
		if (!composite_flatten)
			arrow_arrays_vector.push_back(arrow_array);
		else
		{
			auto	comp_array = std::dynamic_pointer_cast<arrow::StructArray>(arrow_array);
			auto	sub_arrays = comp_array->fields();

			for (int k=0; k < sub_arrays.size(); k++)
				arrow_arrays_vector.push_back(sub_arrays[k]);
		}
	}
};

// ----------------------------------------------------------------
//
// Embedding the Statistics (Arrow)
//
// ----------------------------------------------------------------
static void
arrow_enables_statistics(pgsqlHandlerVector &pgsql_handlers)
{
	if (strcmp(stat_embedded_columns, "*") == 0)
	{
		for (int j=0; j < pgsql_handlers.size(); j++)
			pgsql_handlers[j]->enableStats();
	}
	else
	{
		char   *temp = (char *)alloca(strlen(stat_embedded_columns) + 1);
		char   *tok, *pos;

		strcpy(temp, stat_embedded_columns);
		for (tok = strtok_r(temp, ",", &pos);
			 tok != NULL;
			 tok = strtok_r(NULL, ",", &pos))
		{
			auto    cname = std::string(__trim(tok));
			bool    found = false;

			for (int j=0; j < pgsql_handlers.size(); j++)
			{
				auto    handler = pgsql_handlers[j];

				if (handler->attname == cname)
				{
					handler->enableStats();
					found = true;
					break;
				}
			}
			if (!found)
				Info("-S,--stat tried to enable min/max statistics on '%s' but not found, skipped",
					 cname.c_str());
		}
	}
}

// ----------------------------------------------------------------
//
// Construction of Schema Definition
//
// ----------------------------------------------------------------
#define WITH_RECURSIVE_PG_BASE_TYPE							\
	"WITH RECURSIVE pg_base_type AS ("						\
	"  SELECT 0 depth, oid type_id, oid base_id,"			\
	"         typname, typnamespace,"						\
	"         typlen, typbyval, typalign, typtype,"			\
	"         typrelid, typelem, NULL::int typtypmod"		\
	"    FROM pg_catalog.pg_type t"							\
	"   WHERE t.typbasetype = 0"							\
	" UNION ALL "											\
	"  SELECT b.depth+1, t.oid type_id, b.base_id,"			\
	"         b.typname, b.typnamespace,"					\
	"         b.typlen, b.typbyval, b.typalign, b.typtype,"	\
	"         b.typrelid, b.typelem,"						\
	"         CASE WHEN b.typtypmod IS NULL"				\
	"              THEN t.typtypmod"						\
	"              ELSE b.typtypmod"						\
	"         END typtypmod"								\
	"    FROM pg_catalog.pg_type t, pg_base_type b"			\
	"   WHERE t.typbasetype = b.type_id"					\
	")\n"

static pgsqlHandler
pgsql_define_arrow_list_field(const char *attname, Oid typelemid);
static pgsqlHandler
pgsql_define_arrow_composite_field(const char *attname, Oid typrelid);

/*
 * pgsql_define_arrow_field
 */
static void
pgsql_define_arrow_field(arrowField &arrow_field,
						 pgsqlHandler &pgsql_handler,
						 const char *attname,
						 Oid atttypid,
						 int atttypmod,
						 int attlen,
						 char attbyval,
						 char attalign,
						 char typtype,
						 Oid typrelid,			/* valid, if composite type */
						 Oid typelemid,			/* valid, if array type */
						 const char *nspname,
						 const char *typname,
						 const char *extname)	/* extension name, if any */
{
	auto	pool = arrow::default_memory_pool();
	arrowBuilder	builder = nullptr;
	pgsqlHandler	handler = nullptr;

	/* array type */
	if (typelemid != 0)
	{
		handler = pgsql_define_arrow_list_field(attname, typelemid);
		builder = handler->arrow_builder;
		goto out;
	}
	/* composite type */
	if (typrelid != 0)
	{
		handler = pgsql_define_arrow_composite_field(attname, typrelid);
		builder = handler->arrow_builder;
		goto out;
	}
	/* enum type */
	if (typtype == 'e')
	{
		Elog("Enum type is not supported yet");
	}
	/* several known type provided by extension */
	if (extname != NULL)
	{
		/* contrib/cube (relocatable) */
		if (strcmp(typname, "cube") == 0 &&
			strcmp(extname, "cube") == 0)
		{
			goto out;
		}
	}
	/* other built-in types */
	if (strcmp(typname, "bool") == 0 &&
		strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::BooleanBuilder>(arrow::boolean(), pool);
		handler = std::make_shared<pgsqlBoolHandler>();
	}
	else if (strcmp(typname, "int1") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::Int8Builder>(arrow::int8(), pool);
		handler = std::make_shared<pgsqlInt8Handler>();
	}
	else if (strcmp(typname, "int2") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::Int16Builder>(arrow::int16(), pool);
		handler = std::make_shared<pgsqlInt16Handler>();
	}
	else if (strcmp(typname, "int4") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::Int32Builder>(arrow::int32(), pool);
		handler = std::make_shared<pgsqlInt32Handler>();
	}
	else if (strcmp(typname, "int8") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::Int64Builder>(arrow::int64(), pool);
		handler = std::make_shared<pgsqlInt64Handler>();
	}
	else if (strcmp(typname, "float2") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::HalfFloatBuilder>(arrow::float16(), pool);
		handler = std::make_shared<pgsqlFloat16Handler>();
	}
	else if (strcmp(typname, "float4") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::FloatBuilder>(arrow::float32(), pool);
		handler = std::make_shared<pgsqlFloat32Handler>();
	}
	else if (strcmp(typname, "float8") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::DoubleBuilder>(arrow::float64(), pool);
		handler = std::make_shared<pgsqlFloat64Handler>();
	}
	else if (strcmp(typname, "numeric") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		int		precision = 36;
		int		scale = 9;

		if (atttypmod >= (int)sizeof(int32_t))
		{
			//See, numeric_typmod_precision and numeric_typmod_scale
			precision = ((atttypmod - sizeof(int32_t)) >> 16) & 0xffff;
			scale     = (((atttypmod - sizeof(int32_t)) & 0x7ff) ^ 1024) - 1024;
		}
		builder = std::make_shared<arrow::Decimal128Builder>
			(arrow::decimal128(precision, scale), pool);
		handler = std::make_shared<pgsqlNumericHandler>();
	}
	else if (strcmp(typname, "date") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::Date32Builder>(arrow::date32(), pool);
		handler = std::make_shared<pgsqlDateHandler>();
	}
	else if (strcmp(typname, "time") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::Time64Builder>
			(arrow::time64(arrow::TimeUnit::MICRO), pool);
		handler = std::make_shared<pgsqlTimeHandler>();
	}
	else if (strcmp(typname, "timestamp") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::TimestampBuilder>
			(arrow::timestamp(arrow::TimeUnit::MICRO), pool);
		handler = std::make_shared<pgsqlTimestampHandler>();
	}
	else if (strcmp(typname, "timestamptz") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		//set timezone
		builder = std::make_shared<arrow::TimestampBuilder>
			(arrow::timestamp(arrow::TimeUnit::MICRO), pool);
		handler = std::make_shared<pgsqlTimestampHandler>();
	}
	else if (strcmp(typname, "interval") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::DayTimeIntervalBuilder>
			(arrow::month_day_nano_interval(), pool);
		handler = std::make_shared<pgsqlIntervalHandler>();
	}
	else if ((strcmp(typname, "text") == 0 ||
			  strcmp(typname, "varchar") == 0) &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		builder = std::make_shared<arrow::StringBuilder>(arrow::utf8(), pool);
		handler = std::make_shared<pgsqlTextHandler>();
	}
	else if (strcmp(typname, "bpchar") == 0 &&
			 strcmp(nspname, "pg_catalog") == 0)
	{
		int		width = Max(atttypmod - sizeof(int32_t), 0);
		builder = std::make_shared<arrow::FixedSizeBinaryBuilder>
			(arrow::fixed_size_binary(width));
		handler = std::make_shared<pgsqlBpCharHandler>(width);
	}
	/* elsewhere, we save the values just bunch of binary data */
	else if (attlen == 1)
	{
		builder = std::make_shared<arrow::Int8Builder>(arrow::int8(), pool);
		handler = std::make_shared<pgsqlInt8Handler>();
	}
	else if (attlen == 2)
	{
		builder = std::make_shared<arrow::Int16Builder>(arrow::int16(), pool);
		handler = std::make_shared<pgsqlInt16Handler>();
	}
	else if (attlen == 4)
	{
		builder = std::make_shared<arrow::Int32Builder>(arrow::int32(), pool);
		handler = std::make_shared<pgsqlInt32Handler>();
	}
	else if (attlen == 8)
	{
		builder = std::make_shared<arrow::Int64Builder>(arrow::int64(), pool);
		handler = std::make_shared<pgsqlInt64Handler>();
	}
	else if (attlen == -1)
	{
		builder = std::make_shared<arrow::BinaryBuilder>(arrow::binary(), pool);
		handler = std::make_shared<pgsqlByteaHandler>();
	}
	else
	{
		/*
		 * MEMO: Unfortunately, we have no portable way to pack user defined
		 * fixed-length binary data types, because their 'send' handler often
		 * manipulate its internal data representation.
		 * Please check box_send() for example. It sends four float8 (which
		 * is reordered to bit-endien) values in 32bytes. We cannot understand
		 * its binary format without proper knowledge.
		 */
		Elog("PostgreSQL type: '%s' is not supported", typname);
	}
out:
	/*
	 * Common setup handler, builder, and field
	 */
	handler->attname = std::string(attname);
	handler->typname = std::string(typname);
	handler->type_oid = atttypid;
	handler->type_mod = atttypmod;
	handler->type_align = attalign;
	handler->type_byval = (attbyval == 't');
	handler->type_len = attlen;
	handler->stats_enabled = false;
	handler->arrow_builder = builder;
	handler->arrow_array = nullptr;
	pgsql_handler = handler;
	arrow_field = arrow::field(attname, builder->type(), true);
}

/*
 * pgsql_define_arrow_list_field
 */
static pgsqlHandler
pgsql_define_arrow_list_field(const char *attname, Oid typelemid)
{
	arrowField		element_field;
	arrowBuilder	element_builder;
	pgsqlHandler	element_handler;
	pgsqlHandler	handler;
	arrowBuilder	builder;
	PGresult	   *res;
	char		   *namebuf = (char *)alloca(strlen(attname) + 10);
	char			query[4096];

	sprintf(namebuf, "__%s", attname);
	snprintf(query, sizeof(query),
			 WITH_RECURSIVE_PG_BASE_TYPE
			 "SELECT n.nspname,"
			 "       b.typname,"
			 "       CASE WHEN b.typtypmod IS NULL"
			 "            THEN -1::int"
			 "            ELSE b.typtypmod"
			 "       END typtypmod,"
			 "       b.typlen,"
			 "       b.typbyval,"
			 "       b.typalign,"
			 "       b.typtype,"
			 "       b.typrelid,"
			 "       b.typelem,"
			 "       e.extname,"
			 "       CASE WHEN e.extrelocatable"
			 "            THEN e.extnamespace::regnamespace::text"
			 "            ELSE NULL::text"
			 "       END extnamespace"
			 "  FROM pg_catalog.pg_namespace n,"
			 "       pg_base_type b"
			 "  LEFT OUTER JOIN"
			 "      (pg_catalog.pg_depend d JOIN"
			 "       pg_catalog.pg_extension e ON"
			 "       d.classid = 'pg_catalog.pg_type'::regclass AND"
			 "       d.refclassid = 'pg_catalog.pg_extension'::regclass AND"
			 "       d.refobjid = e.oid AND"
			 "       d.deptype = 'e')"
			 "    ON b.base_id = d.objid"
			 " WHERE b.typnamespace = n.oid"
			 "   AND b.type_id = %u", typelemid);
	res = PQexec(pgsql_conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("failed on pg_type system catalog query: %s",
			 PQresultErrorMessage(res));
	if (PQntuples(res) != 1)
		Elog("unexpected number of result rows: %d", PQntuples(res));

	pgsql_define_arrow_field(element_field,
							 element_handler,
							 namebuf,
							 typelemid,						//type_oid
							 atol(PQgetvalue(res, 0, 2)),	//typtypmod
							 atol(PQgetvalue(res, 0, 3)),	//b.typlen
							 *PQgetvalue(res, 0, 4),		//b.typbyval
							 *PQgetvalue(res, 0, 5),		//b.typalign
							 *PQgetvalue(res, 0, 6),		//b.typtype
							 atol(PQgetvalue(res, 0, 7)),	//b.typrelid
							 atol(PQgetvalue(res, 0, 8)),	//b.typelem
							 PQgetvalue(res, 0, 0),			//n.nspname
							 PQgetvalue(res, 0, 1),			//b.typname
							 (PQgetisnull(res, 0, 9)
							  ? NULL
							  : PQgetvalue(res, 0, 9)));	//e.extname
	element_builder = element_handler->arrow_builder;
	/* setup arrow-builder and pgsql-handler */
	builder = std::make_shared<arrow::ListBuilder>(arrow::default_memory_pool(),
												   element_builder,
												   arrow::list(element_field->type()));
	handler = std::make_shared<pgsqlListHandler>(element_handler, typelemid);
	handler->arrow_builder = builder;
	return handler;
}

/*
 * pgsql_define_arrow_composite_field
 */
static pgsqlHandler
pgsql_define_arrow_composite_field(const char *attname, Oid comptype_relid)
{
	arrow::FieldVector children_fields;
	arrowBuilderVector children_builders;
	pgsqlHandlerVector children_handlers;
	pgsqlHandler	handler;
	arrowBuilder	builder;
	PGresult	   *res;
	char			query[4096];
	int				nfields;

	snprintf(query, sizeof(query),
			 WITH_RECURSIVE_PG_BASE_TYPE
			 "SELECT a.attname,"
			 "       a.attnum,"
			 "       b.base_id atttypid,"
			 "       CASE WHEN b.typtypmod IS NULL"
			 "            THEN a.atttypmod"
			 "            ELSE b.typtypmod"
			 "       END atttypmod,"
			 "       b.typlen,"
			 "       b.typbyval,"
			 "       b.typalign,"
			 "       b.typtype,"
			 "       b.typrelid,"
			 "       b.typelem,"
			 "       n.nspname,"
			 "       b.typname,"
			 "       e.extname,"
			 "       CASE WHEN e.extrelocatable"
			 "            THEN e.extnamespace::regnamespace::text"
			 "            ELSE NULL::text"
			 "       END extnamespace"
			 "  FROM pg_catalog.pg_attribute a,"
			 "       pg_catalog.pg_namespace n,"
			 "       pg_base_type b"
			 "  LEFT OUTER JOIN"
			 "      (pg_catalog.pg_depend d JOIN"
			 "       pg_catalog.pg_extension e ON"
			 "       d.classid = 'pg_catalog.pg_type'::regclass AND"
			 "       d.refclassid = 'pg_catalog.pg_extension'::regclass AND"
			 "       d.refobjid = e.oid AND"
			 "       d.deptype = 'e')"
			 "    ON b.base_id = d.objid"
			 " WHERE b.typnamespace = n.oid"
			 "   AND b.type_id = a.atttypid"
			 "   AND a.attnum > 0"
			 "   AND a.attrelid = %u", comptype_relid);
	res = PQexec(pgsql_conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("failed on pg_type system catalog query: %s",
			 PQresultErrorMessage(res));

	nfields = PQntuples(res);
	children_fields.resize(nfields);
	children_builders.resize(nfields);
	children_handlers.resize(nfields);
	for (int j=0; j < nfields; j++)
	{
		pgsql_define_arrow_field(children_fields[j],
								 children_handlers[j],
								 PQgetvalue(res, j, 0),			//attname
								 atol(PQgetvalue(res, j, 2)),	//atttypid
								 atol(PQgetvalue(res, j, 3)),	//atttypmod
								 atol(PQgetvalue(res, j, 4)),	//b.typlen
								 *PQgetvalue(res, j, 5),		//b.typbyval
								 *PQgetvalue(res, j, 6),		//b.typalign
								 *PQgetvalue(res, j, 7),		//b.typtype
								 atol(PQgetvalue(res, j, 8)),	//b.typrelid
								 atol(PQgetvalue(res, j, 9)),	//b.typelem
								 PQgetvalue(res, j, 10),		//n.nspname
								 PQgetvalue(res, j, 11),		//b.typname
								 (PQgetisnull(res, j, 12)		//e.extname
								  ? NULL
								  : PQgetvalue(res, j, 12)));
		children_builders[j] = children_handlers[j]->arrow_builder;
	}
	PQclear(res);
	/* setup arrow-builder and pgsql-handler */
	builder = std::make_shared<arrow::StructBuilder>(arrow::struct_(children_fields),
													 arrow::default_memory_pool(),
													 children_builders);
	handler = std::make_shared<pgsqlStructHandler>(children_handlers);
	handler->arrow_builder = builder;

	return handler;
}

/*
 * pullup_composite_subfields
 */
static void
pullup_composite_subfields(arrow::FieldVector &arrow_fields,
						   pgsqlHandlerVector &pgsql_handlers)
{
	arrow::FieldVector flatten_fields;

	assert(arrow_fields.size() == pgsql_handlers.size());
	for (int j=0; j < arrow_fields.size(); j++)
	{
		arrowField	field = arrow_fields[j];
		auto		d_type = field->type();

		if (d_type->id() != arrow::Type::STRUCT)
			flatten_fields.push_back(field);
		else
		{
			bool	be_flatten = false;

			if (strcmp(flatten_composite_columns, "*") == 0)
				be_flatten = true;
			else
			{
				char   *temp = (char *)alloca(strlen(flatten_composite_columns) + 1);
				char   *tok, *pos;

				strcpy(temp, flatten_composite_columns);
				for (tok = strtok_r(temp, ",", &pos);
					 tok != NULL;
					 tok = strtok_r(NULL, ",", &pos))
				{
					tok = __trim(tok);
					if (field->name() == std::string(tok))
					{
						be_flatten = true;
						break;
					}
				}
			}
			if (!be_flatten)
				flatten_fields.push_back(field);
			else
			{
				auto	comp_type = std::dynamic_pointer_cast<arrow::StructType>(d_type);

				for (int k=0; k < comp_type->num_fields(); k++)
				{
					auto subfield = comp_type->field(k);
					flatten_fields.push_back(subfield);
				}
				pgsql_handlers[j]->composite_flatten = true;
			}
		}
	}
	arrow_fields = flatten_fields;
}

/*
 * pgsql_define_arrow_schema
 */
static void
pgsql_define_arrow_schema(pgsqlHandlerVector &pgsql_handlers)
{
	arrow::FieldVector arrow_fields;
	int		nfields = PQnfields(pgsql_res);

	arrow_fields.resize(nfields);
	pgsql_handlers.resize(nfields);
	for (int j=0; j < nfields; j++)
	{
		const char *attname = PQfname(pgsql_res, j);
		Oid			atttypid = PQftype(pgsql_res, j);
		int32_t		atttypmod = PQfmod(pgsql_res, j);
		char		query[4096];
		PGresult   *res;

		snprintf(query, sizeof(query),
				 WITH_RECURSIVE_PG_BASE_TYPE
				 "SELECT n.nspname,"
				 "       b.typname,"
				 "       b.typlen,"
				 "       b.typbyval,"
				 "       b.typalign,"
				 "       b.typtype,"
				 "       b.typrelid,"
				 "       b.typelem,"
				 "       b.typtypmod,"
				 "       e.extname,"
				 "       CASE WHEN e.extrelocatable"
				 "            THEN e.extnamespace::regnamespace::text"
				 "            ELSE NULL::text"
				 "       END extnamespace"
				 "  FROM pg_catalog.pg_namespace n,"
				 "       pg_base_type b"
				 "  LEFT OUTER JOIN"
				 "      (pg_catalog.pg_depend d JOIN"
				 "       pg_catalog.pg_extension e ON"
				 "       d.classid = 'pg_catalog.pg_type'::regclass AND"
				 "       d.refclassid = 'pg_catalog.pg_extension'::regclass AND"
				 "       d.refobjid = e.oid AND"
				 "       d.deptype = 'e')"
				 "    ON b.base_id = d.objid"
				 " WHERE b.typnamespace = n.oid"
				 "   AND b.type_id = %u", atttypid);
		res = PQexec(pgsql_conn, query);
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("failed on pg_type system catalog query: %s",
				 PQresultErrorMessage(res));
		if (PQntuples(res) != 1)
			Elog("unexpected number of result rows: %d", PQntuples(res));
		/* setup arrow fields */
		pgsql_define_arrow_field(arrow_fields[j],
								 pgsql_handlers[j],
								 attname,						//attname
								 atttypid,						//atttypid
								 atttypmod,						//atttypmod
								 atol(PQgetvalue(res, 0, 2)),	//b.typbyval
								 *PQgetvalue(res, 0, 3),		//b.attbyval
								 *PQgetvalue(res, 0, 4),		//b.typalign
								 *PQgetvalue(res, 0, 5),		//b.typtype
								 atol(PQgetvalue(res, 0, 6)),	//b.typrelid
								 atol(PQgetvalue(res, 0, 7)),	//b.typelem
								 PQgetvalue(res, 0, 0),			//n.nspname
								 PQgetvalue(res, 0, 1),			//b.typname
								 PQgetisnull(res, 0, 9)			//e.extname
								 ? NULL
								 : PQgetvalue(res, 0, 9));
		PQclear(res);
		//TODO: flatten columns

	}
	/* pull up composite sub-fields to flatten fields */
	if (flatten_composite_columns)
		pullup_composite_subfields(arrow_fields, pgsql_handlers);
	/* enables min/max statistics in arrow */
	if (!parquet_mode && stat_embedded_columns)
		arrow_enables_statistics(pgsql_handlers);
	/*
	 * The primary thread builds and stores the 'arrow_schema', and it shall
	 * be referenced for compatibility checks by worker threads.
	 */
	if (!arrow_schema)
		arrow_schema = arrow::schema(arrow_fields);
	else
	{
		auto	prime_fields = arrow_schema->fields();

		if (prime_fields.size() != arrow_fields.size())
			Elog("Number of result fields mismatch between primary and worker threads. Please review the SQL command");
		for (uint32_t j=0; j < prime_fields.size(); j++)
		{
			auto	p_field = prime_fields[j];
			auto	a_field = arrow_fields[j];

			if (!p_field->IsCompatibleWith(a_field))
				Elog("field-%d is not compatible between primary and worker threads. Please review the SQL command", j);
		}
	}
}

/*
 * fetch_next_pgsql_command
 */
static const char *
fetch_next_pgsql_command(void)
{
	uint32_t	index = __atomic_fetch_add(&pgsql_command_id,
										   1, __ATOMIC_SEQ_CST);
	if (index < pgsql_command_list.size())
		return pgsql_command_list[index].c_str();
	return NULL;
}

/*
 * __pgsql_begin_next_query
 */
#define CURSOR_NAME		"my_cursor"
static bool
__pgsql_begin_next_query(uint32_t fetch_ntuples)
{
	PGresult   *res;

	/* clear resources in the last query */
	if (pgsql_res)
	{
		PQclear(pgsql_res);
		pgsql_res = NULL;
		pgsql_res_index = 0;
		/* CLOSE the cursor */
		res = PQexec(pgsql_conn, "CLOSE " CURSOR_NAME);
		if (PQresultStatus(res) != PGRES_COMMAND_OK)
			Elog("failed on CLOSE '%s': %s",
				 CURSOR_NAME, PQresultErrorMessage(res));
		PQclear(res);
	}

	assert(fetch_ntuples > 0);
	for (;;)
	{
		const char *command = fetch_next_pgsql_command();
		std::string	temp;
		char		query[1024];

		if (!command)
			return false;	/* no more SQL commands to run */
		/* print message */
		if (shows_progress)
			printf("worker-%u: QUERY=[%s]\n", worker_id, command);
		/* declare cursor */
		temp = std::string("DECLARE " CURSOR_NAME " BINARY CURSOR FOR ");
		temp += command;
		res = PQexec(pgsql_conn, temp.c_str());
		if (PQresultStatus(res) != PGRES_COMMAND_OK)
			Elog("unable to declare a SQL cursor[%s]: %s", temp.c_str(), PQresultErrorMessage(res));
		PQclear(res);

		/* fetch first results */
		sprintf(query, "FETCH FORWARD %u FROM " CURSOR_NAME, fetch_ntuples);
		res = PQexecParams(pgsql_conn,
						   query,
						   0, NULL, NULL, NULL, NULL,
						   1);  /* results in binary mode */
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("SQL execution failed: %s", PQresultErrorMessage(res));
		if (PQntuples(res) > 0)
		{
			pgsql_res = res;
			pgsql_res_index = 0;
			return true;
		}
		PQclear(res);
		/* Oops, the command returned an empty result. Try one more */
		res = PQexec(pgsql_conn, "CLOSE " CURSOR_NAME);
		if (PQresultStatus(res) != PGRES_COMMAND_OK)
			Elog("failed on CLOSE '%s': %s",
				 CURSOR_NAME, PQresultErrorMessage(res));
		PQclear(res);
	}
}

/*
 * pgsql_begin_primary_query
 */
static bool
pgsql_begin_primary_query(void)
{
	PGresult   *res;

	/* begin read-only transaction */
	res = PQexec(pgsql_conn, "BEGIN READ ONLY");
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("unable to begin transaction: %s",
			 PQresultErrorMessage(res));
	PQclear(res);

	res = PQexec(pgsql_conn, "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ");
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("unable to switch transaction isolation level: %s",
			 PQresultErrorMessage(res));
	PQclear(res);

	/* export snapshot */
	if (!snapshot_identifier)
	{
		res = PQexec(pgsql_conn, "SELECT pg_catalog.pg_export_snapshot()");
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("unable to export the current transaction snapshot: %s",
				 PQresultErrorMessage(res));
		if (PQntuples(res) != 1 || PQnfields(res) != 1)
			Elog("unexpected result for pg_export_snapshot()");
		snapshot_identifier = pstrdup(PQgetvalue(res, 0, 0));
		PQclear(res);
	}
	/* fetch first results */
	return __pgsql_begin_next_query(10);
}

/*
 * pgsql_begin_next_query
 */
static bool
pgsql_begin_next_query(void)
{
	/* fetch results by 100000 rows */
	return __pgsql_begin_next_query(100000);
}

/*
 * pgsql_flush_record_batch
 */
static void
pgsql_flush_record_batch(pgsqlHandlerVector &pgsql_handlers)
{
	arrowArrayVector arrow_arrays;
	arrow::Status	rv;
	int64_t			nrows = -1;
	uint32_t		chunk_id;
	arrow::Result<int64_t> foffset_before;
	arrow::Result<int64_t> foffset_after;
	auto			custom_metadata = std::make_shared<arrow::KeyValueMetadata>();

	/* setup record batch */
	for (auto cell = pgsql_handlers.begin(); cell != pgsql_handlers.end(); cell++)
	{
		auto		handler = (*cell);
		int64_t		__nrows = handler->Finish();

		if (nrows < 0)
			nrows = __nrows;
		else if (nrows != __nrows)
			Elog("Bug? number of rows mismatch across the buffers");
		handler->appendArrays(arrow_arrays);
		handler->appendStats(custom_metadata);
	}
	/* begin critical section */
	arrow_file_mutex.lock();
	foffset_before = arrow_out_stream->Tell();
	/* setup a record batch */
	auto	rbatch = arrow::RecordBatch::Make(arrow_schema,
											  nrows,
											  arrow_arrays);
	if (parquet_mode)
	{
		/* write out row-group */
		rv = parquet_file_writer->WriteRecordBatch(*rbatch);
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::WriteRecordBatch: %s",
				 rv.ToString().c_str());
		/* flush to the disk */
		rv = parquet_file_writer->NewBufferedRowGroup();
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::NewBufferedRowGroup: %s",
				 rv.ToString().c_str());
	}
	else
	{
		/* write out record batch */
		rv = arrow_file_writer->WriteRecordBatch(*rbatch, custom_metadata);
		if (!rv.ok())
			Elog("failed on arrow::ipc::RecordBatchWriter::WriteRecordBatch: %s",
				 rv.ToString().c_str());
	}
	chunk_id = arrow_num_chunks++;
	arrow_num_items += nrows;
	foffset_after = arrow_out_stream->Tell();
	arrow_file_mutex.unlock();
	/* end critical section */
	/* print progress */
	if (shows_progress)
	{
		time_t		t = time(NULL);
		struct tm	tm;

		localtime_r(&t, &tm);
		printf("%04d-%02d-%02d %02d:%02d:%02d %s[%u] nitems=%ld, length=%ld at file offset=%ld\n",
			   tm.tm_year + 1900,
			   tm.tm_mon + 1,
			   tm.tm_mday,
			   tm.tm_hour,
			   tm.tm_min,
			   tm.tm_sec,
			   !parquet_mode ? "Record Batch" : "Row Group",
			   chunk_id,
			   nrows,
			   foffset_before.ok() && foffset_after.ok() ?
			   foffset_after.ValueOrDie() - foffset_before.ValueOrDie() : -1,
			   foffset_before.ok() ? foffset_before.ValueOrDie() : -1);
	}
	/* reset buffers */
	for (auto cell = pgsql_handlers.begin(); cell != pgsql_handlers.end(); cell++)
	{
		(*cell)->Reset();
	}
}

/*
 * pgsql_process_one_tuple
 */
static size_t
pgsql_process_one_tuple(pgsqlHandlerVector &pgsql_handlers,
						PGresult *res, uint32_t index)
{
	uint32_t	nfields = PQnfields(res);
	size_t		chunk_sz = 0;

	for (uint32_t j=0; j < pgsql_handlers.size(); j++)
	{
		auto	handler = pgsql_handlers[j];
		const char *addr = NULL;
		int		sz = -1;

		if (j < nfields && PQgetisnull(res, index, j) == 0)
		{
			addr = PQgetvalue(res, index, j);
			sz = PQgetlength(res, index, j);
		}
		chunk_sz += handler->putValue(addr, sz);
	}
	return chunk_sz;
}

/*
 * pgsql_process_query_results
 */
static void
pgsql_process_query_results(pgsqlHandlerVector &handlers)
{
	for (;;)
	{
		uint32_t	ntuples = PQntuples(pgsql_res);
		PGresult   *res;

		while (pgsql_res_index < ntuples)
		{
			size_t	chunk_sz = pgsql_process_one_tuple(handlers,
													   pgsql_res,
													   pgsql_res_index++);
			if (chunk_sz >= batch_segment_sz)
				pgsql_flush_record_batch(handlers);
		}
		PQclear(pgsql_res);
		pgsql_res = NULL;

		/* fetch next result rest */
		res = PQexecParams(pgsql_conn,
						   "FETCH FORWARD 100000 FROM " CURSOR_NAME,
						   0, NULL, NULL, NULL, NULL,
						   1);	/* results in binary mode */
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("SQL execution failed: %s", PQresultErrorMessage(res));
		if (PQntuples(res) == 0)
		{
			PQclear(res);
			return;
		}
		pgsql_res = res;
		pgsql_res_index = 0;
	}
}

/*
 * build_pgsql_command_list
 */
static void
build_pgsql_command_list(void)
{
	assert(num_worker_threads > 0);
	if (num_worker_threads == 1)
	{
		/* simple non-parallel case */
		std::string	sql = std::string(raw_pgsql_command);
		pgsql_command_list.push_back(sql);
	}
	else if (ctid_target_table)
	{
		/* replace $(CTID_RANGE) by the special condition */
		char	   *buf = (char *)alloca(2 * strlen(ctid_target_table) + 1000);
		char	   *relkind;
		int64_t		unitsz;
		PGresult   *res;

		sprintf(buf,
				"SELECT c.relname, c.relkind,\n"
				"       GREATEST(pg_relation_size(c.oid), %lu)\n"
				"       / current_setting('block_size')::bigint"
				"  FROM pg_catalog.pg_class c\n"
				" WHERE oid = '%s'::regclass",
				batch_segment_sz * num_worker_threads,
				ctid_target_table);
		res = PQexec(pgsql_conn, buf);
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("failed on [%s]: %s", buf, PQresultErrorMessage(res));
		if (PQntuples(res) != 1)
			Elog("parallel target table [%s] is not exist", ctid_target_table);
		relkind = pstrdup(PQgetvalue(res, 0, 1));
		unitsz = atol(PQgetvalue(res, 0, 2)) / num_worker_threads;
		PQclear(res);

		if (strcmp(relkind, "r") != 0 &&
			strcmp(relkind, "m") != 0 &&
			strcmp(relkind, "t") != 0)
			Elog("--parallel-target [%s] must be either table, materialized view, or toast values",
				 ctid_target_table);
		for (uint32_t __worker=0; __worker < num_worker_threads; __worker++)
		{
			std::string	query = std::string(raw_pgsql_command);

			if (__worker == 0)
				sprintf(buf, "%s.ctid < '(%ld,0)'::tid",
						ctid_target_table,
						unitsz * (__worker+1));
			else if (__worker < num_worker_threads - 1)
				sprintf(buf, "%s.ctid >= '(%ld,0)' AND %s.ctid < '(%ld,0)'::tid",
						ctid_target_table,
						unitsz * __worker,
						ctid_target_table,
						unitsz * (__worker+1));
			else
				sprintf(buf, "%s.ctid >= '(%ld,0)'",
						ctid_target_table,
						unitsz * __worker);
			__replace_string(query,
							 std::string("$(CTID_RANGE)"),
							 std::string(buf));
			pgsql_command_list.push_back(query);
		}
	}
	else if (parallel_keys)
	{
		char   *copy = (char *)alloca(strlen(parallel_keys) + 1);
		char   *token, *pos;

		strcpy(copy, parallel_keys);
		for (token = strtok_r(copy, ",", &pos);
			 token != NULL;
			 token = strtok_r(NULL, ",", &pos))
		{
			std::string query = std::string(raw_pgsql_command);

			__replace_string(query,
							 std::string("$(PARALLEL_KEY)"),
							 std::string(token));
			pgsql_command_list.push_back(query);
		}
	}
	else if (strstr(raw_pgsql_command, "$(WORKER_ID)") &&
			 strstr(raw_pgsql_command, "$(N_WORKERS)"))
	{
		/* replace $(WORKER_ID) and $(N_WORKERS) */
		for (int __worker=0; worker_id < num_worker_threads; __worker++)
		{
			std::string	query = std::string(raw_pgsql_command);

			__replace_string(query,
							 std::string("$(WORKER_ID)"),
							 std::to_string(__worker));
			__replace_string(query,
							 std::string("$(N_WORKERS)"),
							 std::to_string(num_worker_threads));
			pgsql_command_list.push_back(query);
		}
	}
	else
	{
		Elog("Raw SQL command is not valid for parallel dump. It must contains $(WORKER_ID) and $(N_WORKERS) token if --ctid-target or --parallel-keys are not given");
	}
}

/*
 * pgsql_server_connect
 */
static PGconn *
pgsql_server_connect(void)
{
	PGconn	   *conn;
	PGresult   *res;
	const char *query;
	const char *keys[20];
	const char *values[20];
	int			index = 0;
	int			status;

	if (pgsql_hostname)
	{
		keys[index] = "host";
		values[index++] = pgsql_hostname;
	}
	if (pgsql_port_num)
	{
		keys[index] = "port";
		values[index++] = pgsql_port_num;
	}
	if (pgsql_username)
	{
		keys[index] = "user";
		values[index++] = pgsql_username;
    }
    if (pgsql_password)
    {
        keys[index] = "password";
        values[index++] = pgsql_password;
    }
	if (pgsql_database)
	{
		keys[index] = "dbname";
		values[index++] = pgsql_database;
    }
	keys[index] = "application_name";
	values[index++] = "pg2arrow";
	/* terminal */
	keys[index] = NULL;
	values[index] = NULL;

	/* open the connection */
	conn = PQconnectdbParams(keys, values, 0);
	if (!conn)
		Elog("out of memory");
	status = PQstatus(conn);
	if (status != CONNECTION_OK)
		Elog("failed on PostgreSQL connection: %s",
			 PQerrorMessage(conn));

	/* assign configuration parameters */
	for (auto conf = pgsql_config_options.begin(); conf != pgsql_config_options.end(); conf++)
	{
		std::ostringstream buf;

		buf << "SET " << conf->name << " = '" << conf->value << "'";
		query = buf.str().c_str();
		res = PQexec(conn, query);
		if (PQresultStatus(res) != PGRES_COMMAND_OK)
			Elog("failed on change parameter by [%s]: %s",
				 query, PQresultErrorMessage(res));
		PQclear(res);
	}
	/*
	 * ensure client encoding is UTF-8
	 *
	 * Even if user config tries to change client_encoding, pg2arrow
	 * must ensure text encoding is UTF-8.
	 */
	query = "set client_encoding = 'UTF8'";
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("failed on change client_encoding to UTF-8 by [%s]: %s",
			 query, PQresultErrorMessage(res));
	PQclear(res);

	/*
	 * collect server timezone info
	 */
	query = "show timezone";
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK ||
		PQntuples(res) != 1 ||
		PQgetisnull(res, 0, 0))
		Elog("failed on collecting server timezone by [%s]: %s",
			 query, PQresultErrorMessage(res));
	server_timezone = pstrdup(PQgetvalue(res, 0, 0));
	PQclear(res);

	return conn;
}

/*
 * dumpArrowMetadata (--meta=FILENAME)
 */
static int
dumpArrowMetadata(const char *filename)
{
	ArrowFileInfo	af_info;
	const char	   *json;

	if (readArrowFileInfo(filename, &af_info) != 0)
		Elog("unable to read '%s'", filename);
	json = dumpArrowFileInfo(&af_info);
	puts(json);
	return 0;
}

/*
 * dumpArrowSchema (--schema=FILENAME)
 */
static void
__dumpArrowSchemaFieldType(const ArrowField *field)
{
	char   *hint_typename = NULL;

	/* fetch pg_type hind from custom-metadata */
	for (int k=0; k < field->_num_custom_metadata; k++)
	{
		const ArrowKeyValue *kv = &field->custom_metadata[k];

		if (strcmp(kv->key, "pg_type") == 0)
		{
			char   *temp = (char *)alloca(kv->_value_len + 1);
			char   *pos;
			/* assume NAMESPACE.TYPENAME@EXTENSION */
			strcpy(temp, kv->value);

			pos = strchr(temp, '.');
			if (!pos)
				hint_typename = temp;
			else
				hint_typename = pos+1;
			pos = strrchr(hint_typename, '@');
			if (pos)
				*pos = '\0';
			break;
		}
	}

	switch (field->type.node.tag)
	{
		case ArrowNodeTag__Bool:
			std::cout << "bool";
			break;
		case ArrowNodeTag__Int:
			switch (field->type.Int.bitWidth)
			{
				case 8:
					std::cout << "int1";
					break;
				case 16:
					std::cout << "int2";
					break;
				case 32:
					std::cout << "int4";
					break;
				case 64:
					std::cout << "int8";
					break;
				default:
					Elog("unexpected Int bitWidth=%d in Field '%s'",
						 field->type.Int.bitWidth, field->name);
			}
			break;
		case ArrowNodeTag__FloatingPoint:
			switch (field->type.FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					std::cout << "float2";
					break;
				case ArrowPrecision__Single:
					std::cout << "float4";
					break;
				case ArrowPrecision__Double:
					std::cout << "float8";
					break;
				default:
					Elog("unexpected FloatingPoint precision (%d) in Field '%s'",
						 field->type.FloatingPoint.precision, field->name);
			}
			break;
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__LargeUtf8:
			std::cout << "text";
			break;

		case ArrowNodeTag__Binary:
		case ArrowNodeTag__LargeBinary:
			std::cout << "bytea";
			break;

		case ArrowNodeTag__Decimal:
			std::cout << "numeric("
					  << field->type.Decimal.scale
					  << ","
					  << field->type.Decimal.precision
					  << ")";
			break;
		case ArrowNodeTag__Date:
			std::cout << "date";
			break;
		case ArrowNodeTag__Time:
			std::cout << "time";
			break;
		case ArrowNodeTag__Timestamp:
			std::cout << "timestamp";
			break;
		case ArrowNodeTag__Interval:
			std::cout << "interval";
		case ArrowNodeTag__FixedSizeBinary:
			if (hint_typename)
			{
				if (strcmp(hint_typename, "macaddr") == 0 &&
					field->type.FixedSizeBinary.byteWidth == 6)
				{
					std::cout << "macaddr";
					break;
				}
				else if (strcmp(hint_typename, "inet") == 0 &&
						 (field->type.FixedSizeBinary.byteWidth == 4 ||
						  field->type.FixedSizeBinary.byteWidth == 16))
				{
					std::cout << "inet";
					break;
				}
			}
			std::cout << "bpchar("
					  << field->type.FixedSizeBinary.byteWidth
					  << ")";
			break;
		case ArrowNodeTag__List:
		case ArrowNodeTag__LargeList:
			assert(field->_num_children == 1);
			__dumpArrowSchemaFieldType(&field->children[0]);
			std::cout << "[]";
			break;
		case ArrowNodeTag__Struct: {
			char   *namebuf = (char *)alloca(strlen(field->name) + 10);

			sprintf(namebuf, "%s_comp", field->name);
			std::cout << quote_ident(namebuf);
			break;
		}
		default:
			Elog("unsupported type at Field '%s'", field->name);
	}
}

static void
__dumpArrowSchemaComposite(const ArrowField *field)
{
	char   *namebuf = (char *)alloca(strlen(field->name) + 10);

	/* check nested composite type */
	assert(field->type.node.tag == ArrowNodeTag__Struct);
	for (int j=0; j < field->_num_children; j++)
	{
		const ArrowField *__field = &field->children[j];

		if (__field->type.node.tag == ArrowNodeTag__Struct)
			__dumpArrowSchemaComposite(__field);
	}
	/* CREATE TYPE name AS */
	sprintf(namebuf, "%s_comp", field->name);
	std::cout << "CREATE TYPE " << quote_ident(namebuf) << " AS (\n";
	for (int j=0; j < field->_num_children; j++)
	{
		const ArrowField *__field = &field->children[j];
		if (j > 0)
			std::cout << ",\n";
		std::cout << "    " << quote_ident(field->name) << "  ";
		__dumpArrowSchemaFieldType(__field);
	}
	std::cout << "\n);\n";
}

static int
dumpArrowSchema(const char *filename)
{
	ArrowFileInfo	af_info;
	const ArrowSchema *schema;

	if (readArrowFileInfo(filename, &af_info) != 0)
		Elog("unable to read '%s'", filename);
	schema = &af_info.footer.schema;

	std::cout << "---\n"
			  << "--- DDL generated from [" << filename << "]\n"
			  << "---\n";
	/* predefine composite data type */
	for (int j=0; j < schema->_num_fields; j++)
	{
		const ArrowField *field = &schema->fields[j];

		if (field->type.node.tag == ArrowNodeTag__Struct)
			__dumpArrowSchemaComposite(field);
	}
	/* create table statement */
	if (!dump_schema_tablename)
	{
		char   *namebuf = (char *)alloca(strlen(filename) + 1);
		char   *pos;

		strcpy(namebuf, filename);
		namebuf = basename(namebuf);
		pos = strrchr(namebuf, '.');
		if (pos)
			*pos = '\0';
		dump_schema_tablename = namebuf;
	}
	std::cout << "CREATE TABLE " << quote_ident(dump_schema_tablename) << " (\n";
	for (int j=0; j < schema->_num_fields; j++)
	{
		const ArrowField *field = &schema->fields[j];

		if (j > 0)
			std::cout << ",\n";
		std::cout << "    " << quote_ident(field->name) << "  ";
		__dumpArrowSchemaFieldType(field);
        if (!field->nullable)
            std::cout << " not null";
	}
	std::cout << "\n);\n";

	return 0;
}

/*
 * parquet_my_writer_props
 * parquet_arrow_my_writer_props
 *  - properties for parquet::arrow::FileWriter
 */
static std::shared_ptr<parquet::WriterProperties>
parquet_my_writer_props(void)
{
	parquet::WriterProperties::Builder builder;
	auto	props = parquet::WriterProperties::Builder().build();

	/* created-by pg2arrow */
	builder.created_by(std::string("pg2arrow"));
	/* pg2arrow does not split row-groups by number of lines */
	builder.max_row_group_length(INT64_MAX);
	/* default compression method */
	builder.compression(arrow::Compression::type::ZSTD);
	for (auto cell = compression_methods.begin(); cell != compression_methods.end(); cell++)
	{
		auto	comp = *cell;

		if (comp->colname)
			builder.compression(std::string(comp->colname), comp->method);
		else
			builder.compression(comp->method);
	}
	// MEMO: Parquet enables min/max/null-count statistics in the default,
	// so we don't need to touch something special ...(like enable_statistics())
	return builder.build();
}

static std::shared_ptr<parquet::ArrowWriterProperties>
parquet_arrow_my_writer_props(void)
{
	return parquet::default_arrow_writer_properties();
}

/*
 * open_output_file
 */
static void open_output_file(void)
{
	int		fdesc;
	const char *comment = "";

	if (output_filename)
	{
		fdesc = open(output_filename, O_RDWR | O_CREAT | O_TRUNC, 0666);
		if (fdesc < 0)
			Elog("failed on open('%s'): %m", output_filename);
	}
	else
	{
		int		suffixlen = (parquet_mode ? 8 : 6);
		char   *namebuf = strdup(parquet_mode
								 ? "/tmp/pg2arrow_XXXXXX.parquet"
								 : "/tmp/pg2arrow_XXXXXX.arrow");
		if (!namebuf)
			Elog("out of memory");
		fdesc = mkostemps(namebuf, suffixlen,
						  O_RDWR | O_CREAT | O_TRUNC);
		if (fdesc < 0)
			Elog("failed on mkostemps('%s'): %m", namebuf);
		output_filename = namebuf;
		comment = "\nNOTE: -o, --output=FILENAME option was not given, so temporary file was built instead.";
	}
	/* open the output arrow stream */
	{
		auto	rv = arrow::io::FileOutputStream::Open(fdesc);

		if (!rv.ok())
			Elog("failed on arrow::io::FileOutputStream::Open('%s'): %s",
				 output_filename, rv.status().ToString().c_str());
		arrow_out_stream = rv.ValueOrDie();
	}
	/* attach file writer */
	if (parquet_mode)
	{
		auto	rv = parquet::arrow::FileWriter::Open(*arrow_schema,
													  arrow::default_memory_pool(),
													  arrow_out_stream,
													  parquet_my_writer_props(),
													  parquet_arrow_my_writer_props());
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::Open('%s'): %s",
				output_filename, rv.status().ToString().c_str());
		parquet_file_writer = std::move(rv).ValueOrDie();
	}
	else
	{
		auto	rv = arrow::ipc::MakeFileWriter(arrow_out_stream, arrow_schema);

		if (!rv.ok())
			Elog("failed on arrow::ipc::MakeFileWriter for '%s': %s",
				 output_filename,
				 rv.status().ToString().c_str());
		arrow_file_writer = rv.ValueOrDie();
	}
    printf("pg2arrow: opened the output file '%s'%s\n", output_filename, comment);
}

/*
 * close_output_file
 */
static void close_output_file(void)
{
	arrow::Status  rv;

    if (arrow_file_writer)
    {
        rv = arrow_file_writer->Close();
        if (!rv.ok())
            Elog("failed on arrow::ipc::RecordBatchWriter::Close: %s",
                 rv.ToString().c_str());
    }
    if (parquet_file_writer)
	{
        rv = parquet_file_writer->Close();
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::Close: %s",
				 rv.ToString().c_str());
	}
    rv = arrow_out_stream->Close();
    if (!rv.ok())
		Elog("failed on arrow::ipc::io::FileOutputStream::Close: %s",
			 rv.ToString().c_str());
	/* report */
	printf("pg2arrow: wrote on '%s' total %u %s, %ld items\n",
		   output_filename,
		   arrow_num_chunks,
		   parquet_mode ? "row-groups" : "record-batches",
		   arrow_num_items);
}

/*
 * usage
 */
static void usage(void)
{
	std::cerr
		<< "Usage:\n"
		<< "  pg2arrow [OPTION] [database] [username]\n"
		<< "\n"
		<< "General options:\n"
		<< "  -d, --dbname=DBNAME   Database name to connect to\n"
		<< "  -c, --command=COMMAND SQL command to run\n"
		<< "  -t, --table=TABLENAME Equivalent to '-c SELECT * FROM TABLENAME'\n"
		<< "     (-c and -t are exclusive, either of them can be given)\n"
		<< "  -n, --num-workers=N_WORKERS   Enables parallel dump mode.\n"
		<< "                        For parallel dump, the SQL command must contains\n"
		<< "                        - a pair of $(WORKER_ID) and $(N_WORKERS), or\n"
		<< "                        - $(CTID_RANGE) in the WHERE clause\n"
		<< "      --ctid-target=TABLENAME   Specifies the target table to assign the scan\n"
		<< "                                range using $(CTID_RANGE). Table must be a regular\n"
		<< "                                table; not view, foreign table or other relations.\n"
		<< "  -k, --parallel-keys=PARALLEL_KEYS Enables yet another parallel dump.\n"
		<< "                        It requires the SQL command contains $(PARALLEL_KEY)\n"
		<< "                        which is replaced by the comma separated token in\n"
		<< "                        the PARALLEL_KEYS.\n"
		<< "     (-k and -n are exclusive, either of them can be given)\n"
		<< "  -q, --parquet         Enables Parquet format.\n"
		<< "  -o, --output=FILENAME result file in Apache Arrow format\n"
		<< "                        If not given, pg2arrow creates a temporary file.\n"
		<< "  -S, --stats[=COLUMNS] embeds min/max statistics for each record batch\n"
		<< "                        COLUMNS is a comma-separated list of the target\n"
		<< "                        columns if partially enabled.\n"
		<< "      --flatten[=COLUMNS]    Enables to expand RECORD values into flatten\n"
		<< "                             element values.\n"
		<< "Format options:\n"
		<< "  -s, --segment-size=SIZE size of record batch for each [Arrow/Parquet]\n"
		<< "  -C, --compress=[COLUMN:]MODE   Specifies the compression mode [Parquet]\n"
		<< "        MODE := (snappy|gzip|brotli|zstd|lz4|lzo|bz2)\n"
		<< "\n"
		<< "Connection options:\n"
		<< "  -h, --host=HOSTNAME  database server host\n"
		<< "  -p, --port=PORT      database server port\n"
		<< "  -u, --user=USERNAME  database user name\n"
		<< "  -w, --no-password    never prompt for password\n"
		<< "  -W, --password       force password prompt\n"
		<< "\n"
		<< "Other options:\n"
		<< "      --meta=FILENAME  dump metadata of arrow/parquet file.\n"
		<< "      --schema=FILENAME dump schema definition as CREATE TABLE statement\n"
		<< "      --schema-name=NAME table name in the CREATE TABLE statement\n"
		<< "      --progress       shows progress of the job\n"
		<< "      --set=NAME:VALUE config option to set before SQL execution\n"
		<< "  -v, --verbose        shows verbose output\n"
		<< "      --help           shows this message\n"
		<< "\n"
		<< "pg2arrow version " PGSTROM_VERSION " - reports bugs to <pgstrom@heterodb.com>.\n",
	_exit(0);
}

/*
 * parse_options
 */
static void
parse_options(int argc, char * const argv[])
{
	static struct option long_options[] = {
		{"dbname",          required_argument, NULL, 'd'},
		{"command",         required_argument, NULL, 'c'},
		{"table",           required_argument, NULL, 't'},
		{"num-workers",     required_argument, NULL, 'n'},
		{"ctid-target",     required_argument, NULL, 1000},
		{"parallel-keys",   required_argument, NULL, 'k'},
		{"parquet",         no_argument,       NULL, 'q'},
		{"output",          required_argument, NULL, 'o'},
		{"stats",           optional_argument, NULL, 'S'},
		{"flatten",         optional_argument, NULL, 1002},
		{"segment-size",    required_argument, NULL, 's'},
		{"compress",        required_argument, NULL, 'C'},
		{"host",            required_argument, NULL, 'h'},
		{"port",            required_argument, NULL, 'p'},
		{"user",            required_argument, NULL, 'u'},
		{"no-password",     no_argument,       NULL, 'w'},
		{"password",        no_argument,       NULL, 'W'},
		{"meta",            required_argument, NULL, 1003},
		{"schema",          required_argument, NULL, 1004},
		{"schema-name",     required_argument, NULL, 1005},
		{"progress",        no_argument,       NULL, 1006},
		{"set",             required_argument, NULL, 1007},
		{"verbose",         no_argument,       NULL, 'v'},
		{"help",            no_argument,       NULL, 9999},
		{NULL, 0, NULL, 0},
	};
	char   *simple_table_name = NULL;	/* -t, --table */
	int		password_prompt = 0;		/* -w, -W */
	int		code;
	char   *end;

	while ((code = getopt_long(argc, argv,
							   "d:c:t:n:k:qo:S:s:C:h:p:u:wW",
							   long_options, NULL)) >= 0)
	{
		switch (code)
		{
			case 'd':		/* --dbname */
				if (pgsql_database)
					Elog("-d, --dbname was given twice.");
				pgsql_database = optarg;
				break;
			case 'c':		/* --command */
				if (raw_pgsql_command)
					Elog("-c, --command was given twice.");
				if (simple_table_name)
					Elog("-c and -t options are mutually exclusive.");
				if (strncmp(optarg, "file://", 7) == 0)
					raw_pgsql_command = __read_file(optarg + 7);
				else
					raw_pgsql_command = optarg;
				break;
			case 't':		/* --table */
				if (simple_table_name)
					Elog("-t, --table was given twice.");
				if (raw_pgsql_command)
					Elog("-c and -t options are mutually exclusive.");
				simple_table_name = optarg;
				break;
			case 'n':		/* --num-workers */
				if (num_worker_threads != 0)
					Elog("-n, --num-workers was given twice.");
				else
				{
					long	num = strtoul(optarg, &end, 10);

					if (*end != '\0' || num < 1 || num > 9999)
						Elog("not a valid -n|--num-workers option: %s", optarg);
					num_worker_threads = num + 1;	/* primary + threads */
				}
				break;
			case 1000:		/* --ctid-target */
				if (ctid_target_table)
					Elog("--parallel-target was given twice");
				if (parallel_keys)
					Elog("--ctid-target and -k, --parallel_keys are mutually exclusive.");
				ctid_target_table = optarg;
				break;
			case 'k':
				if (parallel_keys)
					Elog("-k, --parallel_keys was given twice.");
				if (num_worker_threads != 0)
					Elog("-k and -n are mutually exclusive.");
				if (ctid_target_table)
					Elog("--ctid-target and -k, --parallel_keys are mutually exclusive.");
				parallel_keys = optarg;
				break;
			case 'q':		/* --parquet */
				parquet_mode = true;
				break;
			case 'o':		/* --output */
				if (output_filename)
					Elog("-o, --output was supplied twice");
				output_filename = optarg;
				break;
			case 'S':		/* --stat */
				if (stat_embedded_columns)
					Elog("--stat option was supplied twice");
				if (optarg)
					stat_embedded_columns = optarg;
				else
					stat_embedded_columns = "*";
				break;
			case 1002:		/* --flatten */
				if (flatten_composite_columns)
					Elog("--flatten option was given twice");
				else if (optarg)
					flatten_composite_columns = optarg;
				else
					flatten_composite_columns = "*";	/* any RECORD values */
				break;
			case 's':		/* --segment-size */
				if (batch_segment_sz != 0)
					Elog("-s, --segment-size was given twice");
				else
				{
					long	sz = strtoul(optarg, &end, 10);

					if (sz == 0)
						Elog("not a valid segment size: %s", optarg);
					else if (*end == 0)
						batch_segment_sz = sz;
					else if (strcasecmp(end, "k") == 0 ||
							 strcasecmp(end, "kb") == 0)
						batch_segment_sz = (sz << 10);
					else if (strcasecmp(end, "m") == 0 ||
							 strcasecmp(end, "mb") == 0)
						batch_segment_sz = (sz << 20);
					else if (strcasecmp(end, "g") == 0 ||
							 strcasecmp(end, "gb") == 0)
						batch_segment_sz = (sz << 30);
					else
						Elog("not a valid segment size: %s", optarg);
				}
				break;
			case 'C':		/* --compress */
				{
					char   *pos = strchr(optarg, ':');
					auto	comp = (compressOption *)palloc0(sizeof(compressOption));

					if (!pos)
					{
						/* --compress=METHOD */
						comp->colname = NULL;
						pos = optarg;
					}
					else
					{
						/* --compress=COLUMN:METHOD */
						*pos++ = '\0';
						comp->colname = __trim(optarg);
					}
					if (strcasecmp(pos, "snappy") == 0)
						comp->method = arrow::Compression::type::SNAPPY;
					else if (strcasecmp(pos, "gzip") == 0)
						comp->method = arrow::Compression::type::GZIP;
					else if (strcasecmp(pos, "brotli") == 0)
						comp->method = arrow::Compression::type::BROTLI;
					else if (strcasecmp(pos, "zstd") == 0)
						comp->method = arrow::Compression::type::ZSTD;
					else if (strcasecmp(pos, "lz4") == 0)
						comp->method = arrow::Compression::type::LZ4;
					else if (strcasecmp(pos, "lzo") == 0)
						comp->method = arrow::Compression::type::LZO;
					else if (strcasecmp(pos, "bz2") == 0)
						comp->method = arrow::Compression::type::BZ2;
					else
						Elog("unknown --compress method [%s]", optarg);
					compression_methods.push_back(comp);
				}
				break;
			case 'h':		/* --host */
				if (pgsql_hostname)
					Elog("-h, --host was given twice");
				pgsql_hostname = optarg;
				break;
			case 'p':		/* --port */
				if (pgsql_port_num)
					Elog("-p, --port was given twice");
				pgsql_port_num = optarg;
				break;
			case 'u':		/* --user */
				if (pgsql_username)
					Elog("-u, --user was given twice");
				pgsql_username = optarg;
				break;
			case 'w':		/* --no-password */
				if (password_prompt > 0)
					Elog("-w and -W options are mutually exclusive");
				password_prompt = -1;
				break;
			case 'W':		/* --password */
				if (password_prompt < 0)
					Elog("-w and -W options are mutually exclusive");
				password_prompt = 1;
				break;
			case 1003:		/* --meta */
				if (dump_meta_filename)
					Elog("--meta was given twice");
				if (dump_schema_filename)
					Elog("--meta and --schema are mutually exclusive");
				dump_meta_filename = optarg;
				break;
			case 1004:		/* --schema */
				if (dump_schema_filename)
					Elog("--schema was given twice");
				if (dump_meta_filename)
					Elog("--meta and --schema are mutually exclusive");
				dump_schema_filename = optarg;
				break;
			case 1005:		/* --schema-name */
				if (dump_schema_tablename)
					Elog("--schema-name was given twice");
				dump_schema_tablename = optarg;
				break;
			case 1006:		/* --progress */
				shows_progress = true;
				break;
			case 1007:		/* --set */
				{
					char   *pos = strchr(optarg, ':');
					configOption config;

					if (!pos)
						Elog("config option must be --set=KEY:VALUE form");
					*pos++ = '\0';
					config.name  = __trim(optarg);
					config.value = __trim(pos);
					pgsql_config_options.push_back(config);
				}
				break;
			case 'v':		/* --verbose */
				verbose++;
				break;
			default:	/* --help */
				usage();
		}
	}

	if (optind + 1 == argc)
	{
		if (pgsql_database)
			Elog("database name was given twice");
		pgsql_database = argv[optind];
	}
	else if (optind + 2 == argc)
	{
		if (pgsql_database)
			Elog("database name was given twice");
		if (pgsql_username)
			Elog("database user was given twice");
		pgsql_database = argv[optind];
		pgsql_username = argv[optind+1];
	}
	else if (optind != argc)
		Elog("too much command line arguments");
	//
	// Set default parallel workers
	//
	if (num_worker_threads == 0)
	{
		if (ctid_target_table)
			num_worker_threads = 4;		/* default parallel */
		else if (parallel_keys)
		{
			const char *pos = parallel_keys;
			int			count = 0;

			while ((pos = strchr(pos, ',')) != NULL)
			{
				count++;
				pos++;
			}
			num_worker_threads = count + 1;
		}
		else
		{
			num_worker_threads = 1;		/* no parallel execution */
		}
	}
	//
	// Check command line option consistency
	//
	if (parquet_mode)
	{
		if (stat_embedded_columns)
			Elog("-S (--stat) is valid only Arrow mode. Parquet embeds statistics by the default");
	}
	else
	{
		if (compression_methods.size() > 0)
			Elog("-C (--compress) is valid only when Parquet mode (-q, --parquet) is enabled");
	}

	if (!raw_pgsql_command && !simple_table_name)
	{
		if (!dump_meta_filename && !dump_schema_filename)
			Elog("Either -c (--command) or -t (--table) command must be supplied");
	}
	else if (simple_table_name)
	{
		char   *buf = (char *)alloca(std::strlen(simple_table_name) + 100);

		assert(!raw_pgsql_command);
		if (ctid_target_table)
			Elog("-t (--table) and --ctid-target are mutually exclusive");
		if (parallel_keys)
			Elog("-t (--table) and --parallel-keys are mutually exclusive");
		if (num_worker_threads == 1)
			sprintf(buf, "SELECT * FROM %s", simple_table_name);
		else
		{
			ctid_target_table = simple_table_name;
			sprintf(buf, "SELECT * FROM %s WHERE $(CTID_RANGE)", simple_table_name);
		}
		raw_pgsql_command = pstrdup(buf);
	}
	else if (raw_pgsql_command)
	{
		assert(!simple_table_name);
		/* auto setting if --parallel_keys is given */
		if (parallel_keys && num_worker_threads == 1)
		{
			const char *pos;
			int		count = 0;

			for (pos = strchr(parallel_keys, ','); pos != NULL; pos = strchr(pos+1, ','))
				count++;
			num_worker_threads = count+1;
			Info("-n, --num-workers was not given, so %d was automatically assigned",
				 num_worker_threads-1);
		}
		if (ctid_target_table && num_worker_threads == 1)
			Elog("--ctid-target must be used with -n, --num-workers together");
		if (num_worker_threads > 1)
		{
			if (ctid_target_table)
			{
				if (!strstr(raw_pgsql_command, "$(CTID_RANGE)"))
					Elog("Raw SQL command must contain $(CTID_RANGE) token if --ctid-target is used together.");
				assert(!parallel_keys);
			}
			else if (parallel_keys)
			{
				if (!strstr(raw_pgsql_command, "$(PARALLEL_KEY)"))
					Elog("Raw SQL command must contain $(PARALLEL_KEY) token if --parallel-keys is used together.");
				assert(!ctid_target_table);
			}
			else if (!strstr(raw_pgsql_command, "$(WORKER_ID)") ||
					 !strstr(raw_pgsql_command, "$(N_WORKERS)"))
			{
				Elog("Raw SQL command must contain $(WORKER_ID) and $(N_WORKERS) if parallel workers are enabled without --ctid-target and --parallel-keys");
			}
		}
	}
	if (batch_segment_sz == 0)
		batch_segment_sz = (256UL << 20);
	if (password_prompt > 0)
		pgsql_password = pstrdup(getpass("Password: "));
}

static void sync_buddy_workers(void)
{
	auto	my_handlers = worker_handlers[worker_id];

	for (uint32_t shift=1; (worker_id & shift) == 0; shift <<= 1)
	{
		uint32_t	buddy_id = worker_id + shift;

		if (buddy_id >= num_worker_threads)
			break;
		if ((errno = pthread_join(worker_threads[buddy_id], NULL)) != 0)
			Elog("Failed on pthread_join: %m");
		/* merge the remained items */
		{
			pgsqlHandlerVector &buddy_handlers = worker_handlers[buddy_id];
			arrow::Status	rv;
			int64_t			nrows = -1;

			assert(my_handlers.size() == buddy_handlers.size());
			/* finalize the buddy buffer */
			for (auto cell = buddy_handlers.begin(); cell != buddy_handlers.end(); cell++)
			{
				int64_t		__nrows = (*cell)->Finish();

				if (nrows < 0)
					nrows = __nrows;
				else if (nrows != __nrows)
					Elog("Bug? number of rows mismatch across the buffers");
			}
			/* move the values from buddy buffer */
			for (int64_t i=0; i < nrows; i++)
			{
				size_t	chunk_sz = 0;

				for (uint32_t j=0; j < my_handlers.size(); j++)
					chunk_sz += my_handlers[j]->moveValue(buddy_handlers[j], i);
				if (chunk_sz >= batch_segment_sz)
					pgsql_flush_record_batch(my_handlers);
			}
		}
	}
}

static void *worker_main(void *__data)
{
	/* wait for setup by the primary thread */
	worker_id = (uintptr_t)__data;
	pgsqlHandlerVector &pgsql_handlers = worker_handlers[worker_id];

	pthread_mutex_lock(&worker_setup_mutex);
	while (!worker_setup_done)
	{
		pthread_cond_wait(&worker_setup_cond,
						  &worker_setup_mutex);
	}
	pthread_mutex_unlock(&worker_setup_mutex);

	/* open the worker connection */
	pgsql_conn = pgsql_server_connect();
	/* begin the primary query */
	if (pgsql_begin_primary_query())
	{
		/* define the schema from the first results */
		pgsql_define_arrow_schema(pgsql_handlers);
		do {
			/* fetch results and write to arrow/parqeut */
			pgsql_process_query_results(pgsql_handlers);
			/* try next SQL command, if any */
		} while (pgsql_begin_next_query());
	}
	else
	{
		Elog("worker-%d get empty results", worker_id);
		//TODO: pgsql_handlers must be initialized, based on the primary thread info
	}
	/* wait for termination of my buddy */
	sync_buddy_workers();
	/* exit thread. remained items shall be merged by buddy */
	return NULL;
}

int main(int argc, char * const argv[])
{
	parse_options(argc, argv);
	/* special case if --meta=FILENAME */
	if (dump_meta_filename)
		return dumpArrowMetadata(dump_meta_filename);
	/* special case if --schema=FILENAME */
	if (dump_schema_filename)
		return dumpArrowSchema(dump_schema_filename);
	/* allocate per-worker data */
	worker_threads.resize(num_worker_threads);
	worker_handlers.resize(num_worker_threads);
	worker_id = 0;		//primary thread
	/* start worker threads, if any */
	for (uintptr_t id=1; id < num_worker_threads; id++)
	{
		if ((errno = pthread_create(&worker_threads[id],
									NULL,
									worker_main, (void *)id)) != 0)
			Elog("failed on pthread_create: %m");
	}
	/* open the primary connection */
	pgsql_conn = pgsql_server_connect();
	/* build the SQL command to run */
	build_pgsql_command_list();
	/* begin the primary query */
	if (!pgsql_begin_primary_query())
		Elog("Query returned an empty results");
	/* define the schema from the first results */
	pgsql_define_arrow_schema(worker_handlers[0]);
	/* open the output file */
	open_output_file();
	/* start worker threads */
	worker_setup_done = true;
	if ((errno = pthread_cond_broadcast(&worker_setup_cond)) != 0)
		Elog("failed on pthread_cond_broadcast: %m");
	do {
		/* fetch results and write to arrow/parqeut */
		pgsql_process_query_results(worker_handlers[0]);
		/* try next SQL command, if any */
	} while (pgsql_begin_next_query());
	/* wait for termination of my buddy */
	sync_buddy_workers();
	/* flush final remaining items */
	pgsql_flush_record_batch(worker_handlers[0]);
	/* close the output file */
	close_output_file();
	
	return 0;
}
