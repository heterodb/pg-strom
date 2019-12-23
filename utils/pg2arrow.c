/*
 * pg2arrow.c - main logic of the command
 *
 * Copyright 2018-2019 (C) KaiGai Kohei <kaigai@heterodb.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License. See the LICENSE file.
 */
#include "pg2arrow.h"
#include "arrow_read.c"
#include "arrow_types.c"

/* static functions */
#define CURSOR_NAME		"curr_pg2arrow"
static PGresult *pgsql_begin_query(PGconn *conn, const char *query);
static PGresult *pgsql_next_result(PGconn *conn);
static void      pgsql_end_query(PGconn *conn);
static SQLtable *pgsql_create_composite_type(PGconn *conn,
											 Oid comptype_relid);
static SQLattribute *pgsql_create_array_element(PGconn *conn,
												Oid array_elemid,
												int *p_numFieldNode,
												int *p_numBuffers);
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
int				shows_progress = 0;
/* dictionary batches */
SQLdictionary  *pgsql_dictionary_list = NULL;

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
		  "Debug options:\n"
		  "      --dump=FILENAME     dump information of arrow file\n"
		  "      --progress          shows progress of the job.\n"
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
	StringInfoData	buf1;
	StringInfoData	buf2;

	memset(&af_info, 0, sizeof(ArrowFileInfo));
	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
		Elog("unable to open '%s': %m", filename);
	readArrowFileDesc(fdesc, &af_info);

	initStringInfo(&buf1);
	initStringInfo(&buf2);
	__dumpArrowNode(&buf1, &af_info.footer.node);
	printf("[Footer]\n%s\n", buf1.data);

	for (i=0; i < af_info.footer._num_dictionaries; i++)
	{
		resetStringInfo(&buf1);
		resetStringInfo(&buf2);
		__dumpArrowNode(&buf1, &af_info.footer.dictionaries[i].node);
		__dumpArrowNode(&buf2, &af_info.dictionaries[i].node);
		printf("[Dictionary Batch %d]\n%s\n%s\n", i, buf1.data, buf2.data);
	}
	for (i=0; i < af_info.footer._num_recordBatches; i++)
	{
		resetStringInfo(&buf1);
		resetStringInfo(&buf2);
		__dumpArrowNode(&buf1, &af_info.footer.recordBatches[i].node);
		__dumpArrowNode(&buf2, &af_info.recordBatches[i].node);
		printf("[Record Batch %d]\n%s\n%s\n", i, buf1.data, buf2.data);
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
		{"help",         no_argument,        NULL, 9999 },
		{NULL, 0, NULL, 0},
	};
	int			c;
	char	   *pos;
	char	   *sql_file = NULL;

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
			case 1002:
				if (append_filename)
					Elog("--append option specified twice");
				if (output_filename)
					Elog("-o and --append are exclusive");
				append_filename = optarg;
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
pgsql_create_dictionary(PGconn *conn, Oid enum_typeid)
{
	SQLdictionary *dict;
	PGresult   *res;
	char		query[4096];
	int			i, j, nitems;
	int			nslots;
	int			num_dicts = 0;

	for (dict = pgsql_dictionary_list; dict != NULL; dict = dict->next)
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
	dict->next = pgsql_dictionary_list;
	pgsql_dictionary_list = dict;
	PQclear(res);

	return dict;
}

static SQLdictionary *
pgsql_duplicate_dictionary(SQLdictionary *orig, int dict_id)
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
	dict->next = pgsql_dictionary_list;
	pgsql_dictionary_list = dict;

	return dict;
}

/*
 * pgsql_setup_attribute
 */
static void
pgsql_setup_attribute(PGconn *conn,
					  SQLattribute *attr,
					  const char *attname,
					  Oid atttypid,
					  int atttypmod,
					  int attlen,
					  char attbyval,
					  char attalign,
					  char typtype,
					  Oid comp_typrelid,
					  Oid array_elemid,
					  const char *nspname,
					  const char *typname,
					  int *p_numFieldNodes,
					  int *p_numBuffers)
{
	attr->attname   = pstrdup(attname);
	attr->atttypid  = atttypid;
	attr->atttypmod = atttypmod;
	attr->attlen    = attlen;
	attr->attbyval  = attbyval;

	if (attalign == 'c')
		attr->attalign = sizeof(char);
	else if (attalign == 's')
		attr->attalign = sizeof(short);
	else if (attalign == 'i')
		attr->attalign = sizeof(int);
	else if (attalign == 'd')
		attr->attalign = sizeof(double);
	else
		Elog("unknown state of attalign: %c", attalign);

	attr->typnamespace = pstrdup(nspname);
	attr->typname = pstrdup(typname);
	attr->typtype = typtype;
	if (typtype == 'b')
	{
		if (array_elemid != InvalidOid)
			attr->element = pgsql_create_array_element(conn, array_elemid,
													   p_numFieldNodes,
													   p_numBuffers);
	}
	else if (typtype == 'c')
	{
		/* composite data type */
		SQLtable   *subtypes;

		assert(comp_typrelid != 0);
		subtypes = pgsql_create_composite_type(conn, comp_typrelid);
		*p_numFieldNodes += subtypes->numFieldNodes;
		*p_numBuffers += subtypes->numBuffers;

		attr->subtypes = subtypes;
	}
	else if (typtype == 'e')
	{
		attr->enumdict = pgsql_create_dictionary(conn, atttypid);
	}
	else
		Elog("unknown state pf typtype: %c", typtype);

	/* assign properties of Apache Arrow Type */
	assignArrowType(attr, p_numBuffers);
	*p_numFieldNodes += 1;
}

/*
 * pgsql_create_composite_type
 */
static SQLtable *
pgsql_create_composite_type(PGconn *conn, Oid comptype_relid)
{
	PGresult   *res;
	SQLtable   *table;
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
	table = palloc0(offsetof(SQLtable, attrs[nfields]));
	table->nfields = nfields;
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
							  &table->attrs[index-1],
							  attname,
							  atooid(atttypid),
							  atoi(atttypmod),
							  atoi(attlen),
							  pg_strtobool(attbyval),
							  pg_strtochar(attalign),
							  pg_strtochar(typtype),
							  atooid(typrelid),
							  atooid(typelem),
							  nspname, typname,
							  &table->numFieldNodes,
							  &table->numBuffers);
	}
	return table;
}

static SQLattribute *
pgsql_create_array_element(PGconn *conn, Oid array_elemid,
						   int *p_numFieldNode,
						   int *p_numBuffers)
{
	SQLattribute   *attr = palloc0(sizeof(SQLattribute));
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
			 "   AND t.oid = %u", array_elemid);
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
						  attr,
						  typname,
						  array_elemid,
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
	return attr;
}

/*
 * pgsql_create_buffer
 */
SQLtable *
pgsql_create_buffer(PGconn *conn, PGresult *res, size_t segment_sz)
{
	int			j, nfields = PQnfields(res);
	SQLtable   *table;

	table = palloc0(offsetof(SQLtable, attrs[nfields]));
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
							  &table->attrs[j],
                              attname,
							  atttypid,
							  atttypmod,
							  atoi(typlen),
							  *typbyval,
							  *typalign,
							  *typtype,
							  atoi(typrelid),
							  atoi(typelem),
							  nspname, typname,
							  &table->numFieldNodes,
							  &table->numBuffers);
		PQclear(__res);
	}
	return table;
}

static int
__arrow_type_is_compatible(SQLtable *root,
						   SQLattribute *attr,
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
	else if (attr->subtypes)
	{
		/* composite type */
		SQLtable   *subtypes = attr->subtypes;

		assert(sql_type->node.tag == ArrowNodeTag__Struct);
		if (subtypes->nfields != field->_num_children)
			return 0;
		for (j=0; j < subtypes->nfields; j++)
		{
			if (!__arrow_type_is_compatible(root,
											subtypes->attrs + j,
											field->children + j))
				return 0;
		}
	}
	else if (attr->enumdict)
	{
		/* enum type; encoded UTF-8 values */
		SQLdictionary  *enumdict = attr->enumdict;

		assert(sql_type->node.tag == ArrowNodeTag__Utf8);
		if (field->dictionary.indexType.bitWidth != 32 ||
			field->dictionary.indexType.is_signed)
			Elog("Index of DictionaryBatch must be unsigned 32bit");
		if (field->dictionary.isOrdered)
			Elog("Ordered DictionaryBatch is not supported right now");
		if (enumdict->is_delta)
		{
			if (enumdict->dict_id == field->dictionary.id)
				return 1;		/* nothing to do */
			enumdict = pgsql_duplicate_dictionary(enumdict,
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
										table->attrs + i,
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
 * pgsql_clear_attribute
 */
static void
pgsql_clear_attribute(SQLattribute *attr)
{
	attr->nitems = 0;
	attr->nullcount = 0;
	sql_buffer_clear(&attr->nullmap);
	sql_buffer_clear(&attr->values);
	sql_buffer_clear(&attr->extra);

	if (attr->subtypes)
	{
		SQLtable   *subtypes = attr->subtypes;
		int			j;

		for (j=0; j < subtypes->nfields; j++)
			pgsql_clear_attribute(&subtypes->attrs[j]);
	}
	if (attr->element)
		pgsql_clear_attribute(attr->element);
}

/*
 * pgsql_writeout_buffer
 */
void
pgsql_writeout_buffer(SQLtable *table)
{
	off_t		currPos;
	size_t		metaSize;
	size_t		bodySize;
	int			j, index;
	ArrowBlock *b;

	/* write a new record batch */
	currPos = lseek(table->fdesc, 0, SEEK_CUR);
	if (currPos < 0)
		Elog("unable to get current position of the file");
	if (currPos != LONGALIGN(currPos))
	{
		uint64	zero = 0;
		size_t	gap = LONGALIGN(currPos) - currPos;

		if (write(table->fdesc, &zero, gap) != gap)
			Elog("unable to fill up alignment gap: %m");
	}
	writeArrowRecordBatch(table, &metaSize, &bodySize);

	index = table->numRecordBatches++;
	if (index == 0)
		table->recordBatches = palloc(sizeof(ArrowBlock));
	else
		table->recordBatches = repalloc(table->recordBatches,
										sizeof(ArrowBlock) * (index+1));
	b = &table->recordBatches[index];
	INIT_ARROW_NODE(b, Block);
	b->offset = currPos;
	b->metaDataLength = metaSize;
	b->bodyLength = bodySize;

	/* shows progress (optional) */
	if (shows_progress)
	{
		printf("RecordBatch %d: offset=%lu length=%lu (meta=%zu, body=%zu)\n",
			   index, currPos, metaSize + bodySize, metaSize, bodySize);
	}

	/* makes table/attributes empty again */
	table->nitems = 0;
	for (j=0; j < table->nfields; j++)
		pgsql_clear_attribute(&table->attrs[j]);
}

/*
 * pgsql_append_results
 */
void
pgsql_append_results(SQLtable *table, PGresult *res)
{
	int		i, ntuples = PQntuples(res);
	int		j, nfields = PQnfields(res);
	size_t	usage;

	assert(nfields == table->nfields);
	for (i=0; i < ntuples; i++)
	{
		usage = 0;

		for (j=0; j < nfields; j++)
		{
			SQLattribute   *attr = &table->attrs[j];
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
			assert(attr->nitems == table->nitems);
			attr->put_value(attr, addr, sz);
			usage += attr->buffer_usage(attr);
		}
		table->nitems++;
		/* exceeds the threshold to write? */
		if (usage > table->segment_sz)
		{
			if (table->nitems == 0)
				Elog("A result row is larger than size of record batch!!");
			pgsql_writeout_buffer(table);
		}
	}
}

/*
 * pgsql_dump_attribute
 */
static void
pgsql_dump_attribute(SQLattribute *attr, const char *label, int indent)
{
	int		j;

	for (j=0; j < indent; j++)
		putchar(' ');
	printf("%s {attname='%s', atttypid=%u, atttypmod=%d, attlen=%d,"
		   " attbyval=%s, attalign=%d, typtype=%c, arrow_type=%s}\n",
		   label,
		   attr->attname,
		   attr->atttypid,
		   attr->atttypmod,
		   attr->attlen,
		   attr->attbyval ? "true" : "false",
		   attr->attalign,
		   attr->typtype,
		   attr->arrow_typename);

	if (attr->typtype == 'b')
	{
		if (attr->element)
			pgsql_dump_attribute(attr->element, "element", indent+2);
	}
	else if (attr->typtype == 'c')
	{
		SQLtable   *subtypes = attr->subtypes;
		char		label[64];

		for (j=0; j < subtypes->nfields; j++)
		{
			snprintf(label, sizeof(label), "subtype[%d]", j);
			pgsql_dump_attribute(&subtypes->attrs[j], label, indent+2);
		}
	}
}

/*
 * pgsql_dump_buffer
 */
void
pgsql_dump_buffer(SQLtable *table)
{
	int		j;
	char	label[64];

	printf("Dump of SQL buffer:\n"
		   "nfields: %d\n"
		   "nitems: %zu\n",
		   table->nfields,
		   table->nitems);
	for (j=0; j < table->nfields; j++)
	{
		snprintf(label, sizeof(label), "attr[%d]", j);
		pgsql_dump_attribute(&table->attrs[j], label, 0);
	}
}

/*
 * setupArrowFieldNode
 */
static int
setupArrowFieldNode(ArrowFieldNode *node, SQLattribute *attr)
{
	SQLtable   *subtypes = attr->subtypes;
	SQLattribute *element = attr->element;
	int			i, count = 1;

	memset(node, 0, sizeof(ArrowFieldNode));
	INIT_ARROW_NODE(node, FieldNode);
	node->length = attr->nitems;
	node->null_count = attr->nullcount;
	/* array types */
	if (element)
		count += setupArrowFieldNode(node + count, element);
	/* composite types */
	if (subtypes)
	{
		for (i=0; i < subtypes->nfields; i++)
			count += setupArrowFieldNode(node + count, &subtypes->attrs[i]);
	}
	return count;
}

void
writeArrowRecordBatch(SQLtable *table,
					  size_t *p_metaLength,
					  size_t *p_bodyLength)
{
	ArrowMessage	message;
	ArrowRecordBatch *rbatch;
	ArrowFieldNode *nodes;
	ArrowBuffer	   *buffers;
	int32			i, j;
	size_t			metaLength;
	size_t			bodyLength = 0;

	/* fill up [nodes] vector */
	nodes = alloca(sizeof(ArrowFieldNode) * table->numFieldNodes);
	for (i=0, j=0; i < table->nfields; i++)
	{
		SQLattribute   *attr = &table->attrs[i];
		assert(table->nitems == attr->nitems);
		j += setupArrowFieldNode(nodes + j, attr);
	}
	assert(j == table->numFieldNodes);

	/* fill up [buffers] vector */
	buffers = alloca(sizeof(ArrowBuffer) * table->numBuffers);
	for (i=0, j=0; i < table->nfields; i++)
	{
		SQLattribute   *attr = &table->attrs[i];
		j += attr->setup_buffer(attr, buffers+j, &bodyLength);
	}
	assert(j == table->numBuffers);

	/* setup Message of Schema */
	memset(&message, 0, sizeof(ArrowMessage));
	INIT_ARROW_NODE(&message, Message);
	message.version = ArrowMetadataVersion__V4;
	message.bodyLength = bodyLength;

	rbatch = &message.body.recordBatch;
	INIT_ARROW_NODE(rbatch, RecordBatch);
	rbatch->length = table->nitems;
	rbatch->nodes = nodes;
	rbatch->_num_nodes = table->numFieldNodes;
	rbatch->buffers = buffers;
	rbatch->_num_buffers = table->numBuffers;
	/* serialization */
	metaLength = writeFlatBufferMessage(table->fdesc, &message);
	for (i=0; i < table->nfields; i++)
	{
		SQLattribute   *attr = &table->attrs[i];
		attr->write_buffer(attr, table->fdesc);
	}
	*p_metaLength = metaLength;
	*p_bodyLength = bodyLength;
}

static void
setupArrowDictionaryEncoding(ArrowDictionaryEncoding *dict,
							 SQLattribute *attr)
{
	INIT_ARROW_NODE(dict, DictionaryEncoding);
	if (attr->enumdict)
	{
		ArrowTypeInt   *indexType = &dict->indexType;

		dict->id = attr->enumdict->dict_id;
		INIT_ARROW_TYPE_NODE(indexType, Int);
		indexType->bitWidth  = 32;		/* OID in PostgreSQL */
		indexType->is_signed = false;
		dict->isOrdered = false;
	}
}

static void
setupArrowField(ArrowField *field, SQLattribute *attr)
{
	memset(field, 0, sizeof(ArrowField));
	INIT_ARROW_NODE(field, Field);
	field->name = attr->attname;
	field->_name_len = strlen(attr->attname);
	field->nullable = true;
	field->type = attr->arrow_type;
	setupArrowDictionaryEncoding(&field->dictionary, attr);
	/* array type */
	if (attr->element)
	{
		field->children = palloc0(sizeof(ArrowField));
		field->_num_children = 1;
		setupArrowField(field->children, attr->element);
	}
	/* composite type */
	if (attr->subtypes)
	{
		SQLtable   *sub = attr->subtypes;
		int			i;

		field->children = palloc0(sizeof(ArrowField) * sub->nfields);
		field->_num_children = sub->nfields;
		for (i=0; i < sub->nfields; i++)
			setupArrowField(&field->children[i], &sub->attrs[i]);
	}
}

static ssize_t
writeArrowSchema(SQLtable *table)
{
	ArrowMessage	message;
	ArrowSchema	   *schema;
	int32			i;

	/* setup Message of Schema */
	memset(&message, 0, sizeof(ArrowMessage));
	INIT_ARROW_NODE(&message, Message);
	message.version = ArrowMetadataVersion__V4;
	schema = &message.body.schema;
	INIT_ARROW_NODE(schema, Schema);
	schema->endianness = ArrowEndianness__Little;
	schema->fields = alloca(sizeof(ArrowField) * table->nfields);
	schema->_num_fields = table->nfields;
	for (i=0; i < table->nfields; i++)
		setupArrowField(&schema->fields[i], &table->attrs[i]);
	/* serialization */
	return writeFlatBufferMessage(table->fdesc, &message);
}

static void
__writeArrowDictionaryBatch(int fdesc, ArrowBlock *block, SQLdictionary *dict)
{
	ArrowMessage	message;
	ArrowDictionaryBatch *dbatch;
	ArrowRecordBatch *rbatch;
	ArrowFieldNode	nodes[1];
	ArrowBuffer		buffers[3];
	loff_t			currPos;
	size_t			metaLength = 0;
	size_t			bodyLength = 0;

	memset(&message, 0, sizeof(ArrowMessage));

	/* DictionaryBatch portion */
	dbatch = &message.body.dictionaryBatch;
	INIT_ARROW_NODE(dbatch, DictionaryBatch);
	dbatch->id = dict->dict_id;
	dbatch->isDelta = false;

	/* ArrowFieldNode of RecordBatch */
	INIT_ARROW_NODE(&nodes[0], FieldNode);
	nodes[0].length = dict->nitems;
	nodes[0].null_count = 0;

	/* ArrowBuffer[0] of RecordBatch -- nullmap */
	INIT_ARROW_NODE(&buffers[0], Buffer);
	buffers[0].offset = bodyLength;
	buffers[0].length = 0;

	/* ArrowBuffer[1] of RecordBatch -- offset to extra buffer */
	INIT_ARROW_NODE(&buffers[1], Buffer);
	buffers[1].offset = bodyLength;
	buffers[1].length = ARROWALIGN(dict->values.usage);
	bodyLength += buffers[1].length;

	/* ArrowBuffer[2] of RecordBatch -- extra buffer */
	INIT_ARROW_NODE(&buffers[2], Buffer);
	buffers[2].offset = bodyLength;
	buffers[2].length = ARROWALIGN(dict->extra.usage);
	bodyLength += buffers[2].length;

	/* RecordBatch portion */
	rbatch = &dbatch->data;
	INIT_ARROW_NODE(rbatch, RecordBatch);
	rbatch->length = dict->nitems;
	rbatch->_num_nodes = 1;
    rbatch->nodes = nodes;
	rbatch->_num_buffers = 3;	/* empty nullmap + offset + extra buffer */
	rbatch->buffers = buffers;

	/* final wrap-up message */
	INIT_ARROW_NODE(&message, Message);
    message.version = ArrowMetadataVersion__V4;
	message.bodyLength = bodyLength;

	currPos = lseek(fdesc, 0, SEEK_CUR);
	if (currPos < 0)
		Elog("unable to get current position of the file");
	metaLength = writeFlatBufferMessage(fdesc, &message);
	__write_buffer_common(fdesc, dict->values.data, dict->values.usage);
	__write_buffer_common(fdesc, dict->extra.data,  dict->extra.usage);

	/* setup Block of Footer */
	INIT_ARROW_NODE(block, Block);
	block->offset = currPos;
	block->metaDataLength = metaLength;
	block->bodyLength = bodyLength;
}

static void
writeArrowDictionaryBatches(SQLtable *table)
{
	SQLdictionary  *dict;
	int				index, count;

	if (!pgsql_dictionary_list)
		return;

	for (dict = pgsql_dictionary_list, count=0;
		 dict != NULL;
		 dict = dict->next, count++);
	table->numDictionaries = count;
	table->dictionaries = palloc0(sizeof(ArrowBlock) * count);

	for (dict = pgsql_dictionary_list, index=0;
		 dict != NULL;
		 dict = dict->next, index++)
	{
		__writeArrowDictionaryBatch(table->fdesc,
									table->dictionaries + index,
									dict);
	}
}

static ssize_t
writeArrowFooter(SQLtable *table)
{
	ArrowFooter		footer;
	ArrowSchema	   *schema;
	int				i;

	/* setup Footer */
	memset(&footer, 0, sizeof(ArrowFooter));
	INIT_ARROW_NODE(&footer, Footer);
	footer.version = ArrowMetadataVersion__V4;
	/* setup Schema of Footer */
	schema = &footer.schema;
	INIT_ARROW_NODE(schema, Schema);
	schema->endianness = ArrowEndianness__Little;
	schema->fields = alloca(sizeof(ArrowField) * table->nfields);
	schema->_num_fields = table->nfields;
	for (i=0; i < table->nfields; i++)
		setupArrowField(&schema->fields[i], &table->attrs[i]);
	/* [dictionaries] */
	footer.dictionaries = table->dictionaries;
	footer._num_dictionaries = table->numDictionaries;

	/* [recordBatches] */
	footer.recordBatches = table->recordBatches;
	footer._num_recordBatches = table->numRecordBatches;

	/* serialization */
	return writeFlatBufferFooter(table->fdesc, &footer);
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
	/* run SQL command */
	res = pgsql_begin_query(conn, sql_command);
	if (!res)
		Elog("SQL command returned an empty result");
	table = pgsql_create_buffer(conn, res, batch_segment_sz);
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
	//pgsql_dump_buffer(table);
	writeArrowDictionaryBatches(table);
	do {
		pgsql_append_results(table, res);
		PQclear(res);
		res = pgsql_next_result(conn);
	} while (res != NULL);
	pgsql_end_query(conn);
	if (table->nitems > 0)
		pgsql_writeout_buffer(table);
	nbytes = writeArrowFooter(table);

	return 0;
}
