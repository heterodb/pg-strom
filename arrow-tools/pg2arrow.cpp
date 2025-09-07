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
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <arrow/api.h>				/* dnf install arrow-devel, or apt install libarrow-dev */
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#ifdef HAS_PARQUET
#include <parquet/arrow/reader.h>	/* dnf install parquet-devel, or apt install libparquet-dev */
#include <parquet/file_reader.h>
#include <parquet/schema.h>
#include <parquet/statistics.h>
#include <parquet/stream_writer.h>
#endif
#include <libpq-fe.h>
#include "arrow_defs.h"

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
static int			num_worker_threads = 0;			/* -n, --num-workers */
static const char  *ctid_target_table = NULL;	/*     --ctid-target */
static const char  *parallel_keys;					/* -k, --parallel-keys */
static bool			parquet_mode = false;			/* -q, --parquet */
static const char  *output_filename = NULL;			/* -o, --output */
static const char  *append_filename = NULL;			/* --append */
static const char  *stat_embedded_columns = NULL;	/* -S, --stat */
static const char  *flatten_composite_columns = NULL; /* --flatten */
static size_t		batch_segment_sz = 0;			/* -s, --segment-size */
static std::vector<compressOption>	compression_methods;
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
static pthread_t	   *worker_threads = NULL;

// ------------------------------------------------
// Thread local variables
// ------------------------------------------------
static __thread PGconn	   *pgsql_conn = NULL;
static __thread const char *server_timezone = NULL;







// ------------------------------------------------
// Error Reporting
// ------------------------------------------------
#define Elog(fmt,...)									\
	do {												\
		if (verbose > 1)								\
			fprintf(stderr, "[ERROR %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
		else											\
			fprintf(stderr, "[ERROR] " fmt "\n",		\
					##__VA_ARGS__);						\
		exit(1);										\
	} while(0)
#define Info(fmt,...)									\
	do {												\
		if (verbose > 1)								\
			fprintf(stderr, "[INFO %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
		else if (verbose > 0)							\
			fprintf(stderr, "[INFO] " fmt "\n",			\
					##__VA_ARGS__);						\
	} while(0)

#define Debug(fmt,...)									\
	do {												\
		if (verbose > 1)								\
			fprintf(stderr, "[DEBUG %s:%d] " fmt "\n",	\
					__FILE__,__LINE__, ##__VA_ARGS__);	\
	} while(0)

// ------------------------------------------------
// Misc utility functions
// ------------------------------------------------
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
	return 0;
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












/*
 * build_pgsql_command_list
 */
static void
build_pgsql_command_list(void)
{
	if (num_worker_threads == 0)
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
				"       / current_setting('block_size')::int"
				"  FROM pg_catalog.pg_class\n"
				" WHERE oid = '%s'::regclass",
				batch_segment_sz * num_worker_threads,
				ctid_target_table);
		res = PQexec(pgsql_conn, buf);
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("failed on [%s]: %s", buf, PQresultErrorMessage(res));
		if (PQntuples(res) != 1)
			Elog("parallel target table [%s] is not exist", ctid_target_table);
		relkind = pstrdup(PQgetvalue(res, 0, 2));
		unitsz = atoi(PQgetvalue(res, 0, 3));
		PQclear(res);

		if (strcmp(relkind, "r") != 0 &&
			strcmp(relkind, "m") != 0 &&
			strcmp(relkind, "t") != 0)
			Elog("--parallel-target [%s] must be either table, materialized view, or toast values",
				 ctid_target_table);
		for (int worker_id=0; worker_id < num_worker_threads; worker_id++)
		{
			std::string	query = std::string(raw_pgsql_command);

			if (worker_id == 0)
				sprintf(buf, "%s.ctid < '(%ld,0)'::tid",
						ctid_target_table,
						unitsz * (worker_id+1));
			else if (worker_id < num_worker_threads - 1)
				sprintf(buf, "%s.ctid >= '(%ld,0)' AND %s.ctid < '(%ld,0)'::tid",
						ctid_target_table,
						unitsz * worker_id,
						ctid_target_table,
						unitsz * (worker_id+1));
			else
				sprintf(buf, "%s.ctid >= '(%ld,0)'",
						ctid_target_table,
						unitsz * worker_id);
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
		for (int worker_id=0; worker_id < num_worker_threads; worker_id++)
		{
			std::string	query = std::string(raw_pgsql_command);

			__replace_string(query,
							 std::string("$(WORKER_ID)"),
							 std::to_string(worker_id));
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
#ifdef HAS_PARQUET
		<< "  -q, --parquet         Enables Parquet format.\n"
#endif
		<< "  -o, --output=FILENAME result file in Apache Arrow format\n"
		<< "      --append=FILENAME result Apache Arrow file to be appended\n"
		<< "     (--output and --append are exclusive. If neither of them\n"
		<< "      are given, it creates a temporary file.)\n"
		<< "  -S, --stat[=COLUMNS] embeds min/max statistics for each record batch\n"
		<< "                        COLUMNS is a comma-separated list of the target\n"
		<< "                        columns if partially enabled.\n"
		<< "      --flatten[=COLUMNS]    Enables to expand RECORD values into flatten\n"
		<< "                             element values.\n"
		<< "Format options:\n"
		<< "  -s, --segment-size=SIZE size of record batch for each [Arrow/Parquet]\n"
#ifdef HAS_PARQUET
		<< "  -C, --compress=[COLUMN:]MODE   Specifies the compression mode [Parquet]\n"
		<< "        MODE := (snappy|gzip|brotli|zstd|lz4|lzo|bz2)\n"
#endif
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
		<< "\n";
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
#ifdef HAS_PARQUET
		{"parquet",         no_argument,       NULL, 'q'},
#endif
		{"output",          required_argument, NULL, 'o'},
		{"append",          required_argument, NULL, 1001},
		{"stat",            required_argument, NULL, 'S'},
		{"flatten",         optional_argument, NULL, 1002},
		{"segment-size",    required_argument, NULL, 's'},
#ifdef HAS_PARQUET
		{"compress",        required_argument, NULL, 'C'},
#endif
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
					num_worker_threads = num;
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
				if (ctid_target_table)
					Elog("--ctid-target and -k, --parallel_keys are mutually exclusive.");
				parallel_keys = optarg;
				break;
#ifdef HAS_PARQUET
			case 'q':		/* --parquet */
				parquet_mode = true;
				break;
#endif
			case 'o':		/* --output */
				if (output_filename)
					Elog("-o, --output was supplied twice");
				if (append_filename)
					Elog("-o and --append are mutually exclusive");
				output_filename = optarg;
				break;
			case 1001:		/* --append */
				if (append_filename)
					Elog("--append was supplied twice");
				if (output_filename)
					Elog("-o and --append are exclusive");
				append_filename = optarg;
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
#ifdef HAS_PARQUET
			case 'C':		/* --compress */
				{
					char   *pos = strchr(optarg, ':');
					compressOption comp;

					if (!pos)
					{
						/* --compress=METHOD */
						comp.colname = NULL;
						pos = optarg;
					}
					else
					{
						/* --compress=COLUMN:METHOD */
						*pos++ = '\0';
						comp.colname = __trim(optarg);
					}
					if (strcasecmp(pos, "snappy") == 0)
						comp.method = arrow::Compression::type::SNAPPY;
					else if (strcasecmp(pos, "gzip") == 0)
						comp.method = arrow::Compression::type::GZIP;
					else if (strcasecmp(pos, "brotli") == 0)
						comp.method = arrow::Compression::type::BROTLI;
					else if (strcasecmp(pos, "zstd") == 0)
						comp.method = arrow::Compression::type::ZSTD;
					else if (strcasecmp(pos, "lz4") == 0)
						comp.method = arrow::Compression::type::LZ4;
					else if (strcasecmp(pos, "lzo") == 0)
						comp.method = arrow::Compression::type::LZO;
					else if (strcasecmp(pos, "bz2") == 0)
						comp.method = arrow::Compression::type::BZ2;
					else
						Elog("unknown --compress method [%s]", optarg);
					compression_methods.push_back(comp);
				}
#endif
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
	// Check command line option consistency
	//
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
		if (num_worker_threads == 0)
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
		if (parallel_keys && num_worker_threads == 0)
		{
			const char *pos;
			int		count = 0;

			for (pos = strchr(parallel_keys, ','); pos != NULL; pos = strchr(pos+1, ','))
				count++;
			num_worker_threads = count+1;
			Info("-n, --num-workers was not given, so %d was automatically assigned",
				 num_worker_threads);
		}
		if (ctid_target_table && num_worker_threads == 0)
			Elog("--ctid-target must be used with -n, --num-workers together");
		if (num_worker_threads > 0)
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
	if (!parquet_mode && compression_methods.size() > 0)
		Elog("-C (--compress) is valid only when Parquet mode (-q, --parquet) is enabled");
	if (password_prompt > 0)
		pgsql_password = pstrdup(getpass("Password: "));
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
	/* open the primary connection */
	pgsql_conn = pgsql_server_connect();
	/* build the SQL command to run */
	build_pgsql_command_list();
	/* read the original arrow file, if --append mode */
	if (append_filename)
	{}
	// begin the query
	// check compatibility
	// define schema
	
	
	
	return 0;
}
