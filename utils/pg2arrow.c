/*
 * pg2arrow.c - main logic of the command
 *
 * Copyright 2018-2020 (C) KaiGai Kohei <kaigai@heterodb.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License. See the LICENSE file.
 */
#include "postgres.h"
#include <assert.h>
#include <endian.h>
#include <getopt.h>
#include <libpq-fe.h>
#include "arrow_ipc.h"

/* static functions */
#define CURSOR_NAME		"curr_pg2arrow"
static PGresult *pgsql_begin_query(PGconn *conn, const char *query);
static PGresult *pgsql_next_result(PGconn *conn);
static void      pgsql_end_query(PGconn *conn);
static void      pgsql_setup_composite_type(PGconn *conn,
											SQLtable *root,
											SQLfield *attr,
											Oid comptype_relid,
											int *p_numFieldNodes,
											int *p_numBuffers);
static void      pgsql_setup_array_element(PGconn *conn,
										   SQLtable *root,
										   SQLfield *attr,
										   Oid array_elemid,
										   int *p_numFieldNode,
										   int *p_numBuffers);
/* for --set option */
typedef struct userConfigOption		userConfigOption;
struct userConfigOption
{
	userConfigOption *next;
	char		query[1];		/* SET xxx='xxx' command */
};

/* command options */
static char	   *sql_command = NULL;
static char	   *output_filename = NULL;
static char	   *append_filename = NULL;
static size_t	batch_segment_sz = 0;
static char	   *pgsql_hostname = NULL;
static char	   *pgsql_portno = NULL;
static char	   *pgsql_username = NULL;
static int		pgsql_password_prompt = 0;
static char	   *pgsql_database = NULL;
static char	   *dump_arrow_filename = NULL;
static int		shows_progress = 0;
static userConfigOption *session_preset_commands = NULL;
/* server settings */
static char	   *server_timezone = NULL;
/* dictionary batches */
SQLdictionary  *pgsql_dictionary_list = NULL;

/* ntohll() */
#if __BYTE_ORDER == __BIG_ENDIAN
#define ntohll(x)		(x)
#else
#define ntohll(x)		__bswap_64(x)
#endif

static void
usage(void)
{
	fputs("Usage:\n"
		  "  pg2arrow [OPTION]... [DBNAME [USERNAME]]\n"
		  "\n"
		  "General options:\n"
		  "  -d, --dbname=DBNAME     database name to connect to\n"
		  "  -c, --command=COMMAND   SQL command to run\n"
		  "  -f, --file=FILENAME     SQL command from file\n"
		  "      (-c and -f are exclusive, either of them must be specified)\n"
		  "  -o, --output=FILENAME   result file in Apache Arrow format\n"
		  "      --append=FILENAME   result file to be appended\n"
		  "\n"
		  "      --output and --append are exclusive to use at the same time.\n"
		  "      If neither of them are specified, it creates a temporary file.)\n"
		  "\n"
		  "Arrow format options:\n"
		  "  -s, --segment-size=SIZE size of record batch for each\n"
		  "      (default: 256MB)\n"
		  "\n"
		  "Connection options:\n"
		  "  -h, --host=HOSTNAME     database server host\n"
		  "  -p, --port=PORT         database server port\n"
		  "  -U, --username=USERNAME database user name\n"
		  "  -w, --no-password       never prompt for password\n"
		  "  -W, --password          force password prompt\n"
		  "\n"
		  "Other options:\n"
		  "      --dump=FILENAME     dump information of arrow file\n"
		  "      --progress          shows progress of the job\n"
		  "      --set=NAME:VALUE    GUC option to set before SQL execution\n"
		  "\n"
		  "Report bugs to <pgstrom@heterodb.com>.\n",
		  stderr);
	exit(1);
}

static int
dumpArrowFile(const char *filename)
{
	ArrowFileInfo	af_info;
	int				i, fdesc;

	memset(&af_info, 0, sizeof(ArrowFileInfo));
	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
		Elog("unable to open '%s': %m", filename);
	readArrowFileDesc(fdesc, &af_info);

	printf("[Footer]\n%s\n",
		   dumpArrowNode((ArrowNode *)&af_info.footer));

	for (i=0; i < af_info.footer._num_dictionaries; i++)
	{
		printf("[Dictionary Batch %d]\n%s\n%s\n", i,
			   dumpArrowNode((ArrowNode *)&af_info.footer.dictionaries[i]),
			   dumpArrowNode((ArrowNode *)&af_info.dictionaries[i]));
	}

	for (i=0; i < af_info.footer._num_recordBatches; i++)
	{
		printf("[Record Batch %d]\n%s\n%s\n", i,
			   dumpArrowNode((ArrowNode *)&af_info.footer.recordBatches[i]),
			   dumpArrowNode((ArrowNode *)&af_info.recordBatches[i]));
	}
	return 0;
}

static void
parse_options(int argc, char * const argv[])
{
	static struct option long_options[] = {
		{"dbname",       required_argument,  NULL,  'd' },
		{"command",      required_argument,  NULL,  'c' },
		{"file",         required_argument,  NULL,  'f' },
		{"output",       required_argument,  NULL,  'o' },
		{"segment-size", required_argument,  NULL,  's' },
		{"host",         required_argument,  NULL,  'h' },
		{"port",         required_argument,  NULL,  'p' },
		{"username",     required_argument,  NULL,  'U' },
		{"no-password",  no_argument,        NULL,  'w' },
		{"password",     no_argument,        NULL,  'W' },
		{"dump",         required_argument,  NULL, 1000 },
		{"progress",     no_argument,        NULL, 1001 },
		{"append",       required_argument,  NULL, 1002 },
		{"set",          required_argument,  NULL, 1003 },
		{"help",         no_argument,        NULL, 9999 },
		{NULL, 0, NULL, 0},
	};
	int			c;
	char	   *pos;
	char	   *sql_file = NULL;
	userConfigOption *last_user_config = NULL;

	while ((c = getopt_long(argc, argv, "d:c:f:o:s:n:dh:p:U:wW",
							long_options, NULL)) >= 0)
	{
		switch (c)
		{
			case 'd':
				if (pgsql_database)
					Elog("-d option specified twice");
				pgsql_database = optarg;
				break;
			case 'c':
				if (sql_command)
					Elog("-c option specified twice");
				if (sql_file)
					Elog("-c and -f options are exclusive");
				sql_command = optarg;
				break;
			case 'f':
				if (sql_file)
					Elog("-f option specified twice");
				if (sql_command)
					Elog("-c and -f options are exclusive");
				sql_file = optarg;
				break;
			case 'o':
				if (output_filename)
					Elog("-o option specified twice");
				if (append_filename)
					Elog("-o and --append are exclusive");
				output_filename = optarg;
				break;
			case 's':
				if (batch_segment_sz != 0)
					Elog("-s option specified twice");
				pos = optarg;
				while (isdigit(*pos))
					pos++;
				if (*pos == '\0')
					batch_segment_sz = atol(optarg);
				else if (strcasecmp(pos, "k") == 0 ||
						 strcasecmp(pos, "kb") == 0)
					batch_segment_sz = atol(optarg) * (1UL << 10);
				else if (strcasecmp(pos, "m") == 0 ||
						 strcasecmp(pos, "mb") == 0)
					batch_segment_sz = atol(optarg) * (1UL << 20);
				else if (strcasecmp(pos, "g") == 0 ||
						 strcasecmp(pos, "gb") == 0)
					batch_segment_sz = atol(optarg) * (1UL << 30);
				else
					Elog("segment size is not valid: %s", optarg);
				break;
			case 'h':
				if (pgsql_hostname)
					Elog("-h option specified twice");
				pgsql_hostname = optarg;
				break;
			case 'p':
				if (pgsql_portno)
					Elog("-p option specified twice");
				pgsql_portno = optarg;
				break;
			case 'U':
				if (pgsql_username)
					Elog("-U option specified twice");
				pgsql_username = optarg;
				break;
			case 'w':
				if (pgsql_password_prompt > 0)
					Elog("-w and -W options are exclusive");
				pgsql_password_prompt = -1;
				break;
			case 'W':
				if (pgsql_password_prompt < 0)
					Elog("-w and -W options are exclusive");
				pgsql_password_prompt = 1;
				break;
			case 1000:		/* --dump */
				if (dump_arrow_filename)
					Elog("--dump option specified twice");
				dump_arrow_filename = optarg;
				break;
			case 1001:		/* --progress */
				if (shows_progress)
					Elog("--progress option specified twice");
				shows_progress = 1;
				break;
			case 1002:		/* --append */
				if (append_filename)
					Elog("--append option specified twice");
				if (output_filename)
					Elog("-o and --append are exclusive");
				append_filename = optarg;
				break;
			case 1003:		/* --set */
				{
					userConfigOption *conf;
					char   *pos, *tail;

					pos = strchr(optarg, ':');
					if (!pos)
						Elog("--set option should take NAME:VALUE");
					*pos++ = '\0';
					while (isspace(*pos))
						pos++;
					tail = pos + strlen(pos) - 1;
					while (tail > pos && isspace(*tail))
						tail--;
					conf = calloc(1, sizeof(userConfigOption) +
								  strlen(optarg) + strlen(pos) + 40);
					if (!conf)
						Elog("out of memory");
					sprintf(conf->query, "SET %s = '%s'", optarg, pos);
					if (last_user_config)
						last_user_config->next = conf;
					else
						session_preset_commands = conf;
					last_user_config = conf;
				}
				break;
			case 9999:		/* --help */
			default:
				usage();
				break;
		}
	}
	if (optind + 1 == argc)
	{
		if (pgsql_database)
			Elog("database name was specified twice");
		pgsql_database = argv[optind];
	}
	else if (optind + 2 == argc)
	{
		if (pgsql_database)
			Elog("database name was specified twice");
		if (pgsql_username)
			Elog("database user was specified twice");
		pgsql_database = argv[optind];
		pgsql_username = argv[optind + 1];
	}
	else if (optind != argc)
		Elog("Too much command line arguments");

	/* '--dump' option is exclusive other options */
	if (dump_arrow_filename)
	{
		if (sql_command || sql_file || output_filename)
			Elog("--dump FILENAME is exclusive with -c, -f, or -o");
		return;
	}

	if (batch_segment_sz == 0)
		batch_segment_sz = (1UL << 28);		/* 256MB in default */
	if (sql_file)
	{
		int			fdesc;
		char	   *buffer;
		struct stat	st_buf;
		ssize_t		nbytes, offset = 0;

		assert(!sql_command);
		fdesc = open(sql_file, O_RDONLY);
		if (fdesc < 0)
			Elog("failed on open '%s': %m", sql_file);
		if (fstat(fdesc, &st_buf) != 0)
			Elog("failed on fstat(2) on '%s': %m", sql_file);
		buffer = palloc(st_buf.st_size + 1);
		while (offset < st_buf.st_size)
		{
			nbytes = read(fdesc, buffer + offset, st_buf.st_size - offset);
			if (nbytes < 0)
			{
				if (errno != EINTR)
					Elog("failed on read('%s'): %m", sql_file);
			}
			else if (nbytes == 0)
				break;
		}
		buffer[offset] = '\0';

		sql_command = buffer;
	}
	else if (!sql_command)
		Elog("Neither -c nor -f options are specified");
}

static PGconn *
pgsql_server_connect(void)
{
	PGconn	   *conn;
	const char *keys[20];
	const char *values[20];
	int			index = 0;
	int			status;

	if (pgsql_hostname)
	{
		keys[index] = "host";
		values[index] = pgsql_hostname;
		index++;
	}
	if (pgsql_portno)
	{
		keys[index] = "port";
		values[index] = pgsql_portno;
		index++;
	}
	if (pgsql_database)
	{
		keys[index] = "dbname";
		values[index] = pgsql_database;
		index++;
	}
	if (pgsql_username)
	{
		keys[index] = "user";
		values[index] = pgsql_username;
		index++;
	}
	if (pgsql_password_prompt > 0)
	{
		keys[index] = "password";
		values[index] = getpass("Password: ");
		index++;
	}
	keys[index] = "application_name";
	values[index] = "pg2arrow";
	index++;
	/* terminal */
	keys[index] = NULL;
	values[index] = NULL;

	conn = PQconnectdbParams(keys, values, 0);
	if (!conn)
		Elog("out of memory");
	status = PQstatus(conn);
	if (status != CONNECTION_OK)
		Elog("failed on PostgreSQL connection: %s",
			 PQerrorMessage(conn));
	return conn;
}

/*
 * pgsql_init_session
 */
static void
pgsql_init_session(PGconn *conn)
{
	PGresult   *res;
	const char *query;
	userConfigOption *conf;

	/*
	 * Preset user's config option
	 */
	for (conf = session_preset_commands; conf != NULL; conf = conf->next)
	{
		res = PQexec(conn, conf->query);
		if (PQresultStatus(res) != PGRES_COMMAND_OK)
			Elog("failed on setting user's config option: %s", conf->query);
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
		Elog("failed to change client_encoding to UTF-8: %s", query);
	PQclear(res);

	/*
	 * collect server timezone info
	 */
#if 0
	res = PQexec(conn, "SET timezone='ASIA/TOKYO'");
	PQclear(res);
#endif
	query = "show timezone";
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK ||
		PQntuples(res) != 1 ||
		PQgetisnull(res, 0, 0))
		Elog("failed on getting server timezone configuration: %s", query);
	server_timezone = strdup(PQgetvalue(res, 0, 0));
	if (!server_timezone)
		Elog("out of memory");
	PQclear(res);
}

/*
 * pgsql_begin_query
 */
static PGresult *
pgsql_begin_query(PGconn *conn, const char *query)
{
	PGresult   *res;
	char	   *buffer;

	/* set transaction read-only */
	res = PQexec(conn, "BEGIN READ ONLY");
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("unable to begin transaction: %s", PQresultErrorMessage(res));
	PQclear(res);

	/* declare cursor */
	buffer = palloc(strlen(query) + 2048);
	sprintf(buffer, "DECLARE %s BINARY CURSOR FOR %s", CURSOR_NAME, query);
	res = PQexec(conn, buffer);
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("unable to declare a SQL cursor: %s", PQresultErrorMessage(res));
	PQclear(res);

	return pgsql_next_result(conn);
}

/*
 * pgsql_next_result
 */
static PGresult *
pgsql_next_result(PGconn *conn)
{
	PGresult   *res;
	/* fetch results per half million rows */
	res = PQexecParams(conn,
					   "FETCH FORWARD 500000 FROM " CURSOR_NAME,
					   0, NULL, NULL, NULL, NULL,
					   1);	/* results in binary mode */
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("SQL execution failed: %s", PQresultErrorMessage(res));
	if (PQntuples(res) == 0)
	{
		PQclear(res);
		return NULL;
	}
	return res;
}

/*
 * pgsql_end_query
 */
static void
pgsql_end_query(PGconn *conn)
{
	PGresult   *res;
	/* close the cursor */
	res = PQexec(conn, "CLOSE " CURSOR_NAME);
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("failed on close cursor '%s': %s", CURSOR_NAME,
			 PQresultErrorMessage(res));
	PQclear(res);

	/* close the connection */
	PQfinish(conn);
}

#define atooid(x)		((Oid) strtoul((x), NULL, 10))
#define InvalidOid		((Oid) 0)

static inline bool
pg_strtobool(const char *v)
{
	if (strcasecmp(v, "t") == 0 ||
		strcasecmp(v, "true") == 0 ||
		strcmp(v, "1") == 0)
		return true;
	else if (strcasecmp(v, "f") == 0 ||
			 strcasecmp(v, "false") == 0 ||
			 strcmp(v, "0") == 0)
		return false;
	Elog("unexpected boolean type literal: %s", v);
}

static inline char
pg_strtochar(const char *v)
{
	if (strlen(v) == 0)
		Elog("unexpected empty string");
	if (strlen(v) > 1)
		Elog("unexpected character string");
	return *v;
}

static SQLdictionary *
pgsql_create_dictionary(PGconn *conn, SQLtable *root, Oid enum_typeid)
{
	SQLdictionary *dict;
	PGresult   *res;
	char		query[4096];
	int			i, j, nitems;
	int			nslots;
	int			num_dicts = 0;

	for (dict = root->sql_dict_list; dict != NULL; dict = dict->next)
	{
		if (dict->enum_typeid == enum_typeid)
			return dict;
		num_dicts++;
	}

	snprintf(query, sizeof(query),
			 "SELECT enumlabel"
			 "  FROM pg_catalog.pg_enum"
			 " WHERE enumtypid = %u", enum_typeid);
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("failed on pg_enum system catalog query: %s",
			 PQresultErrorMessage(res));

	nitems = PQntuples(res);
	nslots = Min(Max(nitems, 1<<10), 1<<18);
	dict = palloc0(offsetof(SQLdictionary, hslots[nslots]));
	dict->enum_typeid = enum_typeid;
	dict->dict_id = num_dicts;
	sql_buffer_init(&dict->values);
	sql_buffer_init(&dict->extra);
	dict->nitems = nitems;
	dict->nslots = nslots;
	sql_buffer_append_zero(&dict->values, sizeof(int32));
	for (i=0; i < nitems; i++)
	{
		const char *enumlabel = PQgetvalue(res, i, 0);
		hashItem   *hitem;
		uint32		hash;
		size_t		len;

		if (PQgetisnull(res, i, 0) != 0)
			Elog("Unexpected result from pg_enum system catalog");

		len = strlen(enumlabel);
		hash = hash_any((const unsigned char *)enumlabel, len);
		j = hash % nslots;
		hitem = palloc0(offsetof(hashItem, label[len + 1]));
		strcpy(hitem->label, enumlabel);
		hitem->label_len = len;
		hitem->index = i;
		hitem->hash = hash;
		hitem->next = dict->hslots[j];
		dict->hslots[j] = hitem;

		sql_buffer_append(&dict->extra, enumlabel, len);
		sql_buffer_append(&dict->values, &dict->extra.usage, sizeof(int32));
	}
	dict->nitems = nitems;
	dict->next = root->sql_dict_list;
	root->sql_dict_list = dict;
	PQclear(res);

	return dict;
}

static SQLdictionary *
pgsql_duplicate_dictionary(SQLtable *root,
						   SQLdictionary *orig, int dict_id)
{
	SQLdictionary  *dict;
	int		i;

	dict = palloc0(offsetof(SQLdictionary, hslots[orig->nslots]));
	dict->enum_typeid = orig->enum_typeid;
	dict->dict_id = dict_id;
	dict->is_delta = true;
	sql_buffer_copy(&dict->values, &orig->values);
	sql_buffer_copy(&dict->extra,  &orig->extra);
	dict->nitems = orig->nitems;
	dict->nslots = orig->nslots;
	for (i=0; i < orig->nslots; i++)
	{
		hashItem   *hitem;
		hashItem   *hcopy;
		
		for (hitem = orig->hslots[i]; hitem != NULL; hitem = hitem->next)
		{
			hcopy = palloc0(offsetof(hashItem, label[hitem->label_len+1]));
			hcopy->hash = hitem->hash;
			hcopy->index = hitem->index;
			hcopy->label_len = hitem->label_len;
			strcpy(hcopy->label, hitem->label);

			hcopy->next = dict->hslots[i];
			dict->hslots[i] = hcopy;
		}
	}
	dict->next = root->sql_dict_list;
	root->sql_dict_list = dict;

	return dict;
}

/*
 * pgsql_setup_attribute
 */
static void
pgsql_setup_attribute(PGconn *conn,
					  SQLtable *root,
					  SQLfield *column,
					  const char *attname,
					  Oid atttypid,
					  int atttypmod,
					  int attlen,
					  char attbyval,
					  char attalign,
					  char typtype,
					  Oid typrelid,		/* valid, if composite type */
					  Oid typelemid,	/* valid, if array type */
					  const char *nspname,
					  const char *typname,
					  int *p_numFieldNodes,
					  int *p_numBuffers)
{
	*p_numFieldNodes += 1;
	*p_numBuffers += assignArrowTypePgSQL(column,
										  attname,
										  atttypid,
										  atttypmod,
										  typname,
										  nspname,
										  attlen,
										  attbyval,
										  typtype,
										  attalign,
										  typrelid,
										  typelemid,
										  server_timezone);
	if (typrelid != InvalidOid)
	{
		assert(typtype == 'c');
		pgsql_setup_composite_type(conn, root, column,
								   typrelid,
								   p_numFieldNodes,
								   p_numBuffers);
	}
	else if (typelemid != InvalidOid)
	{
		pgsql_setup_array_element(conn, root, column,
								  typelemid,
								  p_numFieldNodes,
								  p_numBuffers);
	}
	else if (typtype == 'e')
	{
		column->enumdict = pgsql_create_dictionary(conn, root, atttypid);
	}
}

/*
 * pgsql_setup_composite_type
 */
static void
pgsql_setup_composite_type(PGconn *conn,
						   SQLtable *root,
						   SQLfield *attr,
						   Oid comptype_relid,
						   int *p_numFieldNodes,
						   int *p_numBuffers)						   
{
	PGresult   *res;
	SQLfield *subfields;
	char		query[4096];
	int			j, nfields;

	snprintf(query, sizeof(query),
			 "SELECT attname, attnum, atttypid, atttypmod, attlen,"
			 "       attbyval, attalign, typtype, typrelid, typelem,"
			 "       nspname, typname"
			 "  FROM pg_catalog.pg_attribute a,"
			 "       pg_catalog.pg_type t,"
			 "       pg_catalog.pg_namespace n"
			 " WHERE t.typnamespace = n.oid"
			 "   AND a.atttypid = t.oid"
			 "   AND a.attrelid = %u", comptype_relid);
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("failed on pg_type system catalog query: %s",
			 PQresultErrorMessage(res));

	nfields = PQntuples(res);
	subfields = palloc0(sizeof(SQLfield) * nfields);
	for (j=0; j < nfields; j++)
	{
		const char *attname   = PQgetvalue(res, j, 0);
		const char *attnum    = PQgetvalue(res, j, 1);
		const char *atttypid  = PQgetvalue(res, j, 2);
		const char *atttypmod = PQgetvalue(res, j, 3);
		const char *attlen    = PQgetvalue(res, j, 4);
		const char *attbyval  = PQgetvalue(res, j, 5);
		const char *attalign  = PQgetvalue(res, j, 6);
		const char *typtype   = PQgetvalue(res, j, 7);
		const char *typrelid  = PQgetvalue(res, j, 8);
		const char *typelem   = PQgetvalue(res, j, 9);
		const char *nspname   = PQgetvalue(res, j, 10);
		const char *typname   = PQgetvalue(res, j, 11);
		int			index     = atoi(attnum);

		if (index < 1 || index > nfields)
			Elog("attribute number is out of range");
		pgsql_setup_attribute(conn,
							  root,
							  &subfields[index-1],
							  attname,
							  atooid(atttypid),
							  atoi(atttypmod),
							  atoi(attlen),
							  pg_strtobool(attbyval),
							  pg_strtochar(attalign),
							  pg_strtochar(typtype),
							  atooid(typrelid),
							  atooid(typelem),
							  nspname,
							  typname,
							  p_numFieldNodes,
							  p_numBuffers);
	}
	attr->nfields = nfields;
	attr->subfields = subfields;
}

static void
pgsql_setup_array_element(PGconn *conn,
						  SQLtable *root,
						  SQLfield *attr,
						  Oid typelemid,
						  int *p_numFieldNode,
						  int *p_numBuffers)
{
	SQLfield	   *element = palloc0(sizeof(SQLfield));
	PGresult	   *res;
	char			query[4096];
	const char     *nspname;
	const char	   *typname;
	const char	   *typlen;
	const char	   *typbyval;
	const char	   *typalign;
	const char	   *typtype;
	const char	   *typrelid;
	const char	   *typelem;

	snprintf(query, sizeof(query),
			 "SELECT nspname, typname,"
			 "       typlen, typbyval, typalign, typtype,"
			 "       typrelid, typelem"
			 "  FROM pg_catalog.pg_type t,"
			 "       pg_catalog.pg_namespace n"
			 " WHERE t.typnamespace = n.oid"
			 "   AND t.oid = %u", typelemid);
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("failed on pg_type system catalog query: %s",
			 PQresultErrorMessage(res));
	if (PQntuples(res) != 1)
		Elog("unexpected number of result rows: %d", PQntuples(res));
	nspname  = PQgetvalue(res, 0, 0);
	typname  = PQgetvalue(res, 0, 1);
	typlen   = PQgetvalue(res, 0, 2);
	typbyval = PQgetvalue(res, 0, 3);
	typalign = PQgetvalue(res, 0, 4);
	typtype  = PQgetvalue(res, 0, 5);
	typrelid = PQgetvalue(res, 0, 6);
	typelem  = PQgetvalue(res, 0, 7);

	pgsql_setup_attribute(conn,
						  root,
						  element,
						  typname,
						  typelemid,
						  -1,
						  atoi(typlen),
						  pg_strtobool(typbyval),
						  pg_strtochar(typalign),
						  pg_strtochar(typtype),
						  atooid(typrelid),
						  atooid(typelem),
						  nspname,
						  typname,
						  p_numFieldNode,
						  p_numBuffers);
	attr->element = element;
}

/*
 * pgsql_create_buffer
 */
SQLtable *
pgsql_create_buffer(PGconn *conn, PGresult *res,
					size_t segment_sz,
					const char *sql_command)
{
	int			j, nfields = PQnfields(res);
	SQLtable   *table;
	ArrowKeyValue *kv;

	table = palloc0(offsetof(SQLtable, columns[nfields]));
	table->segment_sz = segment_sz;
	table->nitems = 0;
	table->nfields = nfields;
	for (j=0; j < nfields; j++)
	{
		const char *attname = PQfname(res, j);
		Oid			atttypid = PQftype(res, j);
		int			atttypmod = PQfmod(res, j);
		PGresult   *__res;
		char		query[4096];
		const char *typlen;
		const char *typbyval;
		const char *typalign;
		const char *typtype;
		const char *typrelid;
		const char *typelem;
		const char *nspname;
		const char *typname;

		snprintf(query, sizeof(query),
				 "SELECT typlen, typbyval, typalign, typtype,"
				 "       typrelid, typelem, nspname, typname"
				 "  FROM pg_catalog.pg_type t,"
				 "       pg_catalog.pg_namespace n"
				 " WHERE t.typnamespace = n.oid"
				 "   AND t.oid = %u", atttypid);
		__res = PQexec(conn, query);
		if (PQresultStatus(__res) != PGRES_TUPLES_OK)
			Elog("failed on pg_type system catalog query: %s",
				 PQresultErrorMessage(res));
		if (PQntuples(__res) != 1)
			Elog("unexpected number of result rows: %d", PQntuples(__res));
		typlen   = PQgetvalue(__res, 0, 0);
		typbyval = PQgetvalue(__res, 0, 1);
		typalign = PQgetvalue(__res, 0, 2);
		typtype  = PQgetvalue(__res, 0, 3);
		typrelid = PQgetvalue(__res, 0, 4);
		typelem  = PQgetvalue(__res, 0, 5);
		nspname  = PQgetvalue(__res, 0, 6);
		typname  = PQgetvalue(__res, 0, 7);
		pgsql_setup_attribute(conn,
							  table,
							  &table->columns[j],
                              attname,
							  atttypid,
							  atttypmod,
							  atoi(typlen),
							  *typbyval,
							  *typalign,
							  *typtype,
							  atoi(typrelid),
							  atoi(typelem),
							  nspname,
							  typname,
							  &table->numFieldNodes,
							  &table->numBuffers);
		PQclear(__res);
	}

	/* save the SQL command as custom metadata */
	kv = palloc0(sizeof(ArrowKeyValue));
	initArrowNode(kv, KeyValue);
	kv->key = "sql_command";
	kv->_key_len = 11;
	kv->value = sql_command;
	kv->_value_len = strlen(sql_command);

	table->customMetadata = kv;
	table->numCustomMetadata = 1;

	return table;
}

static int
__arrow_type_is_compatible(SQLtable *root,
						   SQLfield *attr,
						   ArrowField *field)
{
	ArrowType  *sql_type = &attr->arrow_type;
	int			j;

	if (sql_type->node.tag != field->type.node.tag)
		return 0;

	switch (sql_type->node.tag)
	{
		case ArrowNodeTag__Int:
			if (sql_type->Int.bitWidth != field->type.Int.bitWidth)
				return 0;	/* not compatible */
			sql_type->Int.is_signed = field->type.Int.is_signed;
			break;
		case ArrowNodeTag__FloatingPoint:
			if (sql_type->FloatingPoint.precision
				!= field->type.FloatingPoint.precision)
				return 0;
			break;
		case ArrowNodeTag__Decimal:
			/* adjust precision and scale to the previous one */
			sql_type->Decimal.precision = field->type.Decimal.precision;
			sql_type->Decimal.scale = field->type.Decimal.scale;
			break;
		case ArrowNodeTag__Date:
			/* adjust unit */
			sql_type->Date.unit = field->type.Date.unit;
			break;
		case ArrowNodeTag__Time:
			/* adjust unit */
			sql_type->Time.unit = field->type.Time.unit;
			sql_type->Time.bitWidth = field->type.Time.bitWidth;
			break;
		case ArrowNodeTag__Timestamp:
			/* adjust unit */
			sql_type->Timestamp.unit = field->type.Timestamp.unit;
			break;
		case ArrowNodeTag__Interval:
			/* adjust unit */
			sql_type->Interval.unit = field->type.Interval.unit;
			break;
		case ArrowNodeTag__FixedSizeBinary:
			if (sql_type->FixedSizeBinary.byteWidth
				!= field->type.FixedSizeBinary.byteWidth)
				return 0;
			break;
		default:
			break;
	}

	if (attr->element)
	{
		/* array type */
		assert(sql_type->node.tag == ArrowNodeTag__List);
		if (field->_num_children != 1 ||
			!__arrow_type_is_compatible(root, attr->element, field->children))
			return 0;
	}
	else if (attr->subfields)
	{
		/* composite type */
		assert(sql_type->node.tag == ArrowNodeTag__Struct);
		if (attr->nfields != field->_num_children)
			return 0;
		for (j=0; j < attr->nfields; j++)
		{
			if (!__arrow_type_is_compatible(root,
											&attr->subfields[j],
											&field->children[j]))
				return 0;
		}
	}
	else if (attr->enumdict)
	{
		/* enum type; encoded UTF-8 values */
		SQLdictionary  *enumdict = attr->enumdict;

		assert(sql_type->node.tag == ArrowNodeTag__Utf8);
		if (field->dictionary.indexType.bitWidth != 32 ||
			!field->dictionary.indexType.is_signed)
			Elog("Index of DictionaryBatch must be unsigned 32bit");
		if (field->dictionary.isOrdered)
			Elog("Ordered DictionaryBatch is not supported right now");
		if (enumdict->is_delta)
		{
			if (enumdict->dict_id == field->dictionary.id)
				return 1;		/* nothing to do */
			enumdict = pgsql_duplicate_dictionary(root,
												  enumdict,
												  field->dictionary.id);
		}
		else
		{
			enumdict->is_delta = true;
			enumdict->dict_id  = field->dictionary.id;
		}
		/*
		 * Right now, we don't implement "true" delta DictionaryBatch.
		 * The appended RecordBatch will reference the new DictionaryBatch
		 * that replaced entire dictionary we previously written on.
		 */
	}
	return 1;
}

static void
initial_setup_append_file(SQLtable *table)
{
	ArrowFileInfo	af_info;
	ArrowSchema	   *af_schema;
	int				i, nitems;
	size_t			nbytes;
	off_t			offset;
	char			buffer[100];

	/*
	 * check schema compatibility
	 */
	readArrowFileDesc(table->fdesc, &af_info);
	af_schema = &af_info.footer.schema;
	if (af_schema->_num_fields != table->nfields)
		Elog("--append is given, but number of columns are different.");
	for (i=0; i < table->nfields; i++)
	{
		if (!__arrow_type_is_compatible(table,
										&table->columns[i],
										af_schema->fields + i))
			Elog("--append is given, but attribute %d is not compatible", i+1);
	}

	/* restore DictionaryBatches already in the file */
	nitems = af_info.footer._num_dictionaries;
	table->numDictionaries = nitems;
	table->dictionaries = palloc0(sizeof(ArrowBlock) * nitems);
	memcpy(table->dictionaries,
		   af_info.footer.dictionaries,
		   sizeof(ArrowBlock) * nitems);

	/* restore RecordBatches already in the file */
	nitems = af_info.footer._num_recordBatches;
	table->numRecordBatches = nitems;
	table->recordBatches = palloc0(sizeof(ArrowBlock) * nitems);
	memcpy(table->recordBatches,
		   af_info.footer.recordBatches,
		   sizeof(ArrowBlock) * nitems);

	/* move the file offset in front of the Footer portion */
	nbytes = sizeof(int32) + 6;	/* strlen("ARROW1") */
	offset = lseek(table->fdesc, -nbytes, SEEK_END);
	if (offset < 0)
		Elog("failed on lseek(%d, %zu, SEEK_END): %m",
			 table->fdesc, sizeof(int32) + 6);
	if (read(table->fdesc, buffer, nbytes) != nbytes)
		Elog("failed on read(2): %m");
	offset -= *((int32 *)buffer);
	if (lseek(table->fdesc, offset, SEEK_SET) < 0)
		Elog("failed on lseek(%d, %zu, SEEK_SET): %m",
			 table->fdesc, offset);
}

/*
 * pgsql_writeout_buffer
 */
static void
pgsql_writeout_buffer(SQLtable *table)
{
	size_t		nitems = table->nitems;

	if (table->nitems == 0)
		return;

	writeArrowRecordBatch(table);
	if (shows_progress)
	{
		ArrowBlock *block;
		int		index = table->numRecordBatches - 1;

		assert(index >= 0);
		block = &table->recordBatches[index];
		printf("RecordBatch[%d]: "
			   "offset=%lu length=%lu (meta=%u, body=%lu) nitems=%zu\n",
			   index,
			   block->offset,
			   block->metaDataLength + block->bodyLength,
			   block->metaDataLength,
			   block->bodyLength,
			   nitems);
	}
}

/*
 * pgsql_append_results
 */
void
pgsql_append_results(SQLtable *table, PGresult *res)
{
	SQLfield  *columns = table->columns;
	int		i, ntuples = PQntuples(res);
	int		j, nfields = PQnfields(res);

	assert(table->nfields == nfields);
	for (i=0; i < ntuples; i++)
	{
		size_t	usage = 0;
		size_t	nitems = columns[0].nitems;

		for (j=0; j < nfields; j++)
		{
			SQLfield	   *column = &columns[j];
			const char	   *addr;
			size_t			sz;
			/* data must be binary format */
			assert(PQfformat(res, j) == 1);
			if (PQgetisnull(res, i, j))
			{
				addr = NULL;
				sz = 0;
			}
			else
			{
				addr = PQgetvalue(res, i, j);
				sz = PQgetlength(res, i, j);
			}
			assert(column->nitems == nitems);
			usage += sql_field_put_value(column, addr, sz);
		}
		table->nitems++;
		/* exceeds the threshold to write? */
		if (usage > table->segment_sz)
		{
			if (nitems == 0)
				Elog("A result row is larger than size of record batch!!");
			pgsql_writeout_buffer(table);
		}
	}
}

/*
 * pgsql_dump_attribute
 */
static void
pgsql_dump_attribute(SQLfield *column, const char *label, int indent)
{
	int		j;

	for (j=0; j < indent; j++)
		putchar(' ');
	printf("%s {attname='%s', atttypid=%u, atttypmod=%d, attlen=%d,"
		   " attbyval=%s, attalign=%d, typtype=%c, arrow_type=%s}\n",
		   label,
		   column->field_name,
		   column->sql_type.pgsql.typeid,
		   column->sql_type.pgsql.typmod,
		   column->sql_type.pgsql.typlen,
		   column->sql_type.pgsql.typbyval ? "true" : "false",
		   column->sql_type.pgsql.typalign,
		   column->sql_type.pgsql.typtype,
		   column->arrow_typename);

	if (column->sql_type.pgsql.typtype == 'b')
	{
		if (column->element)
			pgsql_dump_attribute(column->element, "element", indent+2);
	}
	else if (column->sql_type.pgsql.typtype == 'c')
	{
		char		label[64];

		for (j=0; j < column->nfields; j++)
		{
			snprintf(label, sizeof(label), "subfields[%d]", j);
			pgsql_dump_attribute(&column->subfields[j], label, indent+2);
		}
	}
}

/*
 * pgsql_dump_buffer
 */
void
pgsql_dump_buffer(SQLtable *table)
{
	char	label[64];
	int		j;

	printf("Dump of SQL buffer:\n"
		   "nfields: %d\n"
		   "nitems: %zu\n",
		   table->nfields,
		   table->nitems);
	for (j=0; j < table->nfields; j++)
	{
		snprintf(label, sizeof(label), "attr[%d]", j);
		pgsql_dump_attribute(&table->columns[j], label, 0);
	}
}

/*
 * Entrypoint of pg2arrow
 */
int main(int argc, char * const argv[])
{
	PGconn	   *conn;
	PGresult   *res;
	SQLtable   *table = NULL;
	ssize_t		nbytes;

	parse_options(argc, argv);
	/* special case if '--dump <filename>' is given */
	if (dump_arrow_filename)
		return dumpArrowFile(dump_arrow_filename);

	/* open PostgreSQL connection */
	conn = pgsql_server_connect();
	/* initialize the session */
	pgsql_init_session(conn);
	/* run SQL command */
	res = pgsql_begin_query(conn, sql_command);
	if (!res)
		Elog("SQL command returned an empty result");
	table = pgsql_create_buffer(conn, res, batch_segment_sz, sql_command);
	if (append_filename)
	{
		table->fdesc = open(append_filename, O_RDWR, 0644);
		if (table->fdesc < 0)
			Elog("failed to open '%s'", append_filename);
		table->filename = append_filename;

		initial_setup_append_file(table);
	}
	else
	{
		if (output_filename)
		{
			table->fdesc = open(output_filename,
								O_RDWR | O_CREAT | O_TRUNC, 0644);
			if (table->fdesc < 0)
				Elog("failed to open '%s'", output_filename);
			table->filename = output_filename;
		}
		else
		{
			char	temp_fname[128];

			strcpy(temp_fname, "/tmp/XXXXXX.arrow");
			table->fdesc = mkostemps(temp_fname, 6,
									 O_RDWR | O_CREAT | O_TRUNC);
			if (table->fdesc < 0)
				Elog("failed to open '%s' : %m", temp_fname);
			table->filename = pstrdup(temp_fname);
			fprintf(stderr,
					"NOTICE: -o, --output=FILENAME option was not given,\n"
					"        so a temporary file '%s' was built instead.\n",
					temp_fname);
		}
		/* write out header stuff from the file head */
		nbytes = write(table->fdesc, "ARROW1\0\0", 8);
		if (nbytes != 8)
			Elog("failed on write(2): %m");
		nbytes = writeArrowSchema(table);
	}
	writeArrowDictionaryBatches(table);
	do {
		pgsql_append_results(table, res);
		PQclear(res);
		res = pgsql_next_result(conn);
	} while (res != NULL);
	pgsql_end_query(conn);
	pgsql_writeout_buffer(table);
	nbytes = writeArrowFooter(table);

	return 0;
}

/*
 * This hash function was written by Bob Jenkins
 * (bob_jenkins@burtleburtle.net), and superficially adapted
 * for PostgreSQL by Neil Conway. For more information on this
 * hash function, see http://burtleburtle.net/bob/hash/doobs.html,
 * or Bob's article in Dr. Dobb's Journal, Sept. 1997.
 *
 * In the current code, we have adopted Bob's 2006 update of his hash
 * function to fetch the data a word at a time when it is suitably aligned.
 * This makes for a useful speedup, at the cost of having to maintain
 * four code paths (aligned vs unaligned, and little-endian vs big-endian).
 * It also uses two separate mixing functions mix() and final(), instead
 * of a slower multi-purpose function.
 */

/* Get a bit mask of the bits set in non-uint32 aligned addresses */
#define UINT32_ALIGN_MASK (sizeof(uint32) - 1)

/* Rotate a uint32 value left by k bits - note multiple evaluation! */
#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

/*----------
 * mix -- mix 3 32-bit values reversibly.
 *
 * This is reversible, so any information in (a,b,c) before mix() is
 * still in (a,b,c) after mix().
 *
 * If four pairs of (a,b,c) inputs are run through mix(), or through
 * mix() in reverse, there are at least 32 bits of the output that
 * are sometimes the same for one pair and different for another pair.
 * This was tested for:
 * * pairs that differed by one bit, by two bits, in any combination
 *	 of top bits of (a,b,c), or in any combination of bottom bits of
 *	 (a,b,c).
 * * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 *	 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 *	 is commonly produced by subtraction) look like a single 1-bit
 *	 difference.
 * * the base values were pseudorandom, all zero but one bit set, or
 *	 all zero plus a counter that starts at zero.
 *
 * This does not achieve avalanche.  There are input bits of (a,b,c)
 * that fail to affect some output bits of (a,b,c), especially of a.  The
 * most thoroughly mixed value is c, but it doesn't really even achieve
 * avalanche in c.
 *
 * This allows some parallelism.  Read-after-writes are good at doubling
 * the number of bits affected, so the goal of mixing pulls in the opposite
 * direction from the goal of parallelism.  I did what I could.  Rotates
 * seem to cost as much as shifts on every machine I could lay my hands on,
 * and rotates are much kinder to the top and bottom bits, so I used rotates.
 *----------
 */
#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);	c += b; \
  b -= a;  b ^= rot(a, 6);	a += c; \
  c -= b;  c ^= rot(b, 8);	b += a; \
  a -= c;  a ^= rot(c,16);	c += b; \
  b -= a;  b ^= rot(a,19);	a += c; \
  c -= b;  c ^= rot(b, 4);	b += a; \
}

/*----------
 * final -- final mixing of 3 32-bit values (a,b,c) into c
 *
 * Pairs of (a,b,c) values differing in only a few bits will usually
 * produce values of c that look totally different.  This was tested for
 * * pairs that differed by one bit, by two bits, in any combination
 *	 of top bits of (a,b,c), or in any combination of bottom bits of
 *	 (a,b,c).
 * * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 *	 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 *	 is commonly produced by subtraction) look like a single 1-bit
 *	 difference.
 * * the base values were pseudorandom, all zero but one bit set, or
 *	 all zero plus a counter that starts at zero.
 *
 * The use of separate functions for mix() and final() allow for a
 * substantial performance increase since final() does not need to
 * do well in reverse, but is does need to affect all output bits.
 * mix(), on the other hand, does not need to affect all output
 * bits (affecting 32 bits is enough).  The original hash function had
 * a single mixing operation that had to satisfy both sets of requirements
 * and was slower as a result.
 *----------
 */
#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c, 4); \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

/*
 * hash_any() -- hash a variable-length key into a 32-bit value
 *		k		: the key (the unaligned variable-length array of bytes)
 *		len		: the length of the key, counting by bytes
 *
 * Returns a uint32 value.  Every bit of the key affects every bit of
 * the return value.  Every 1-bit and 2-bit delta achieves avalanche.
 * About 6*len+35 instructions. The best hash table sizes are powers
 * of 2.  There is no need to do mod a prime (mod is sooo slow!).
 * If you need less than 32 bits, use a bitmask.
 *
 * This procedure must never throw elog(ERROR); the ResourceOwner code
 * relies on this not to fail.
 *
 * Note: we could easily change this function to return a 64-bit hash value
 * by using the final values of both b and c.  b is perhaps a little less
 * well mixed than c, however.
 */
Datum
hash_any(const unsigned char *k, int keylen)
{
	register uint32 a,
				b,
				c,
				len;

	/* Set up the internal state */
	len = keylen;
	a = b = c = 0x9e3779b9 + len + 3923095;

	/* If the source pointer is word-aligned, we use word-wide fetches */
	if (((uintptr_t) k & UINT32_ALIGN_MASK) == 0)
	{
		/* Code path for aligned source data */
		register const uint32 *ka = (const uint32 *) k;

		/* handle most of the key */
		while (len >= 12)
		{
			a += ka[0];
			b += ka[1];
			c += ka[2];
			mix(a, b, c);
			ka += 3;
			len -= 12;
		}

		/* handle the last 11 bytes */
		k = (const unsigned char *) ka;
#ifdef WORDS_BIGENDIAN
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 8);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 24);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((uint32) k[6] << 8);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 16);
				/* fall through */
			case 5:
				b += ((uint32) k[4] << 24);
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((uint32) k[2] << 8);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 16);
				/* fall through */
			case 1:
				a += ((uint32) k[0] << 24);
				/* case 0: nothing left to add */
		}
#else							/* !WORDS_BIGENDIAN */
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 24);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 8);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((uint32) k[6] << 16);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((uint32) k[2] << 16);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
#endif							/* WORDS_BIGENDIAN */
	}
	else
	{
		/* Code path for non-aligned source data */

		/* handle most of the key */
		while (len >= 12)
		{
#ifdef WORDS_BIGENDIAN
			a += (k[3] + ((uint32) k[2] << 8) + ((uint32) k[1] << 16) + ((uint32) k[0] << 24));
			b += (k[7] + ((uint32) k[6] << 8) + ((uint32) k[5] << 16) + ((uint32) k[4] << 24));
			c += (k[11] + ((uint32) k[10] << 8) + ((uint32) k[9] << 16) + ((uint32) k[8] << 24));
#else							/* !WORDS_BIGENDIAN */
			a += (k[0] + ((uint32) k[1] << 8) + ((uint32) k[2] << 16) + ((uint32) k[3] << 24));
			b += (k[4] + ((uint32) k[5] << 8) + ((uint32) k[6] << 16) + ((uint32) k[7] << 24));
			c += (k[8] + ((uint32) k[9] << 8) + ((uint32) k[10] << 16) + ((uint32) k[11] << 24));
#endif							/* WORDS_BIGENDIAN */
			mix(a, b, c);
			k += 12;
			len -= 12;
		}

		/* handle the last 11 bytes */
#ifdef WORDS_BIGENDIAN
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 8);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 24);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += k[7];
				/* fall through */
			case 7:
				b += ((uint32) k[6] << 8);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 16);
				/* fall through */
			case 5:
				b += ((uint32) k[4] << 24);
				/* fall through */
			case 4:
				a += k[3];
				/* fall through */
			case 3:
				a += ((uint32) k[2] << 8);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 16);
				/* fall through */
			case 1:
				a += ((uint32) k[0] << 24);
				/* case 0: nothing left to add */
		}
#else							/* !WORDS_BIGENDIAN */
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 24);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 8);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ((uint32) k[7] << 24);
				/* fall through */
			case 7:
				b += ((uint32) k[6] << 16);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ((uint32) k[3] << 24);
				/* fall through */
			case 3:
				a += ((uint32) k[2] << 16);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
#endif							/* WORDS_BIGENDIAN */
	}

	final(a, b, c);

	/* report the result */
	return c;
}
