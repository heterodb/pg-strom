/*
 * pgsql_client.c - PostgreSQL specific portion for pg2arrow command
 *
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "sql2arrow.h"
#include <limits.h>
#include <libpq-fe.h>

#define CURSOR_NAME		"curr_pg2arrow"
static char	   *server_timezone = NULL;

static void		pgsql_setup_composite_type(PGconn *conn,
										   SQLtable *root,
										   SQLfield *attr,
										   Oid comptype_relid,
										   ArrowField *arrow_field,
										   int *p_numFieldNodes,
										   int *p_numBuffers);
static void		pgsql_setup_array_element(PGconn *conn,
										  SQLtable *root,
										  SQLfield *attr,
										  Oid typelemid,
										  ArrowField *arrow_field,
										  int *p_numFieldNode,
										  int *p_numBuffers);
typedef struct
{
	PGconn	   *conn;
	PGresult   *res;
	uint32_t	nitems;
	uint32_t	index;
} PGSTATE;

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

static PGconn *
pgsql_server_connect(const char *sqldb_hostname,
                     const char *sqldb_port_num,
                     const char *sqldb_username,
                     const char *sqldb_password,
                     const char *sqldb_database)
{
	PGconn	   *conn;
	const char *keys[20];
	const char *values[20];
	int			index = 0;
	int			status;

	if (sqldb_hostname)
	{
		keys[index] = "host";
		values[index] = sqldb_hostname;
		index++;
	}
	if (sqldb_port_num)
	{
		keys[index] = "port";
		values[index] = sqldb_port_num;
		index++;
	}
	if (sqldb_username)
	{
		keys[index] = "user";
		values[index] = sqldb_username;
		index++;
	}
	if (sqldb_password)
	{
		keys[index] = "password";
		values[index] = sqldb_password;
		index++;
	}
	if (sqldb_database)
	{
		keys[index] = "dbname";
		values[index] = sqldb_database;
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
 * pgsql_move_next
 */
static bool
pgsql_move_next(PGSTATE *pgstate)
{
	PGconn	   *conn = pgstate->conn;
	PGresult   *res;

	if (pgstate->index >= pgstate->nitems)
	{
		const char *query = "FETCH FORWARD 500000 FROM " CURSOR_NAME;

		if (pgstate->res)
			PQclear(pgstate->res);

		res = PQexecParams(conn, query,
						   0, NULL, NULL, NULL, NULL,
						   1);	/* results in binary mode */
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("SQL execution failed: %s",
				 PQresultErrorMessage(res));
		pgstate->res = res;
		pgstate->nitems = PQntuples(res);
		pgstate->index  = 0;

		if (pgstate->nitems == 0)
		{
			PQclear(pgstate->res);
			pgstate->res = NULL;
			return false;	/* no more tuples */
		}
	}
	return true;
}

/*
 * pgsql_create_dictionary
 */
static SQLdictionary *
pgsql_create_dictionary(PGconn *conn, SQLtable *root,
						Oid enum_typeid,
						ArrowField *arrow_field)
{
	SQLdictionary *dict = NULL;
	PGresult   *res;
	char		query[4096];
	int			i, nitems;
	int64_t		dict_id = enum_typeid;

	if (arrow_field)
	{
		assert(arrow_field->dictionary);
		dict_id = arrow_field->dictionary->id;
	}

	for (dict = root->sql_dict_list; dict != NULL; dict = dict->next)
	{
		if (dict->dict_id == dict_id)
			break;
	}
	if (!dict)
	{
		dict = palloc0(offsetof(SQLdictionary, hslots[1024]));
		dict->dict_id = dict_id;
		sql_buffer_init(&dict->values);
		sql_buffer_init(&dict->extra);
		dict->nslots = 1024;

		dict->next = root->sql_dict_list;
		root->sql_dict_list = dict;
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
	for (i=0; i < nitems; i++)
	{
		const char *enumlabel = PQgetvalue(res, i, 0);
		hashItem   *hitem;
		uint32_t	hash, hindex;
		size_t		sz;

		if (PQgetisnull(res, i, 0) != 0)
			Elog("Unexpected result from pg_enum system catalog");

		sz = strlen(enumlabel);
		hash = hash_any((const unsigned char *)enumlabel, sz);
		hindex = hash % dict->nslots;
		for (hitem = dict->hslots[hindex]; hitem != NULL; hitem = hitem->next)
		{
			if (hitem->hash == hash &&
				hitem->label_sz == sz &&
				strcmp(hitem->label, enumlabel) == 0)
				break;
		}

		if (!hitem)
		{
			hitem = palloc0(offsetof(hashItem, label[sz+1]));
			hitem->hash = hash;
			hitem->index = dict->nitems++;
			hitem->label_sz = sz;
			strcpy(hitem->label, enumlabel);

			hitem->next = dict->hslots[hindex];
			dict->hslots[hindex] = hitem;

			sql_buffer_append(&dict->extra, enumlabel, sz);
			if (dict->values.usage == 0)
				sql_buffer_append_zero(&dict->values, sizeof(uint32_t));
			sql_buffer_append(&dict->values,
							  &dict->extra.usage, sizeof(uint32_t));
		}
	}
	PQclear(res);

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
					  Oid typrelid,     /* valid, if composite type */
					  Oid typelemid,    /* valid, if array type */
					  const char *nspname,
					  const char *typname,
					  const char *extname,	/* extension name, if any */
					  ArrowField *arrow_field,
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
										  server_timezone,
										  extname,
										  arrow_field);
	if (typrelid != InvalidOid)
	{
		assert(typtype == 'c');
		pgsql_setup_composite_type(conn, root, column,
								   typrelid,
								   arrow_field,
								   p_numFieldNodes,
								   p_numBuffers);
	}
	else if (typelemid != InvalidOid)
	{
		pgsql_setup_array_element(conn, root, column,
								  typelemid,
								  arrow_field,
								  p_numFieldNodes,
								  p_numBuffers);
	}
	else if (typtype == 'e')
	{
		column->enumdict = pgsql_create_dictionary(conn, root, atttypid,
												   arrow_field);
	}
}

#define WITH_RECURSIVE_PG_BASE_TYPE								\
	"WITH RECURSIVE pg_base_type AS ("							\
	"  SELECT 0 depth, oid type_id, oid base_id,"				\
	"         typname, typnamespace,"							\
	"         typlen, typbyval, typalign, typtype,"				\
	"         typrelid, typelem, NULL::int typtypmod"			\
	"    FROM pg_catalog.pg_type t"								\
	"   WHERE t.typbasetype = 0"								\
	" UNION ALL "												\
	"  SELECT b.depth+1, t.oid type_id, b.base_id,"				\
	"         b.typname, b.typnamespace,"						\
	"         b.typlen, b.typbyval, b.typalign, b.typtype,"		\
	"         b.typrelid, b.typelem,"							\
	"         CASE WHEN b.typtypmod IS NULL"					\
	"              THEN t.typtypmod"							\
	"              ELSE b.typtypmod"							\
	"         END typtypmod"									\
	"    FROM pg_catalog.pg_type t, pg_base_type b"				\
	"   WHERE t.typbasetype = b.type_id"						\
	")\n"

/*
 * pgsql_setup_composite_type
 */
static void
pgsql_setup_composite_type(PGconn *conn,
						   SQLtable *root,
						   SQLfield *attr,
						   Oid comptype_relid,
						   ArrowField *arrow_field,
						   int *p_numFieldNodes,
						   int *p_numBuffers)						   
{
	PGresult   *res;
	SQLfield *subfields;
	char		query[4096];
	int			j, nfields;

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
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("failed on pg_type system catalog query: %s",
			 PQresultErrorMessage(res));

	nfields = PQntuples(res);
	if (arrow_field &&
		arrow_field->_num_children != nfields)
		Elog("unexpected number of child fields in ArrowField");
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
		const char *extname   = (PQgetisnull(res, j, 12) ? NULL : PQgetvalue(res, j, 12));
//		const char *extschema = (PQgetisnull(res, j, 13) ? NULL : PQgetvalue(res, j, 13));
		ArrowField *sub_field = NULL;
		int			index     = atoi(attnum);

		if (index < 1 || index > nfields)
			Elog("attribute number is out of range");
		if (arrow_field)
		{
			sub_field = &arrow_field->children[index-1];
			attname = arrow_field->name;	/* keep the original field name */
		}
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
							  extname,
							  sub_field,
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
						  ArrowField *arrow_field,
						  int *p_numFieldNode,
						  int *p_numBuffers)
{
	SQLfield	   *element = palloc0(sizeof(SQLfield));
	PGresult	   *res;
	char			query[4096];
	const char	   *attname;
	const char     *nspname;
	const char	   *typname;
	const char	   *typemod;
	const char	   *typlen;
	const char	   *typbyval;
	const char	   *typalign;
	const char	   *typtype;
	const char	   *typrelid;
	const char	   *typelem;
	const char	   *extname;
	ArrowField	   *elem_field = NULL;

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
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("failed on pg_type system catalog query: %s",
			 PQresultErrorMessage(res));
	if (PQntuples(res) != 1)
		Elog("unexpected number of result rows: %d", PQntuples(res));
	nspname  = PQgetvalue(res, 0, 0);
	typname  = PQgetvalue(res, 0, 1);
	typemod  = PQgetvalue(res, 0, 2);
	typlen   = PQgetvalue(res, 0, 3);
	typbyval = PQgetvalue(res, 0, 4);
	typalign = PQgetvalue(res, 0, 5);
	typtype  = PQgetvalue(res, 0, 6);
	typrelid = PQgetvalue(res, 0, 7);
	typelem  = PQgetvalue(res, 0, 8);
	extname  = (PQgetisnull(res, 0, 9) ? NULL : PQgetvalue(res, 0, 9));

	if (!arrow_field)
	{
		char   *namebuf = alloca(strlen(attr->field_name) + 2);
		sprintf(namebuf, "_%s", attr->field_name);
		attname = namebuf;
	}
	else
	{
		if (arrow_field->_num_children != 1)
			Elog("unexpected number of child fields in ArrowField");
		elem_field = &arrow_field->children[0];
		attname = elem_field->name;		/* keep the original field name */
	}
	pgsql_setup_attribute(conn,
						  root,
						  element,
						  attname,
						  typelemid,
						  atoi(typemod),
						  atoi(typlen),
						  pg_strtobool(typbyval),
						  pg_strtochar(typalign),
						  pg_strtochar(typtype),
						  atooid(typrelid),
						  atooid(typelem),
						  nspname,
						  typname,
						  extname,
						  elem_field,
						  p_numFieldNode,
						  p_numBuffers);
	attr->element = element;
}

/*
 * pgsql_create_buffer
 */
static SQLtable *
pgsql_create_buffer(PGSTATE *pgstate,
					ArrowFileInfo *af_info,
					SQLdictionary *sql_dict_list)
{
	PGconn	   *conn = pgstate->conn;
	PGresult   *res = pgstate->res;
	SQLtable   *table;
	int			nfields = PQnfields(res);

	//TODO: expand nfields if --
	table = palloc0(offsetof(SQLtable, columns[nfields]));
	table->nitems = 0;
	table->nfields = nfields;
	table->sql_dict_list = sql_dict_list;

	if (af_info &&
		af_info->footer.schema._num_fields != nfields)
		Elog("number of the fields mismatch");

	for (int j=0; j < nfields; j++)
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
		const char *typtypmod;
		const char *extname;
		ArrowField *arrow_field = NULL;

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
		__res = PQexec(conn, query);
		if (PQresultStatus(__res) != PGRES_TUPLES_OK)
			Elog("failed on pg_type system catalog query: %s",
				 PQresultErrorMessage(res));
		if (PQntuples(__res) != 1)
			Elog("unexpected number of result rows: %d", PQntuples(__res));
		nspname  = PQgetvalue(__res, 0, 0);
		typname  = PQgetvalue(__res, 0, 1);
		typlen   = PQgetvalue(__res, 0, 2);
		typbyval = PQgetvalue(__res, 0, 3);
		typalign = PQgetvalue(__res, 0, 4);
		typtype  = PQgetvalue(__res, 0, 5);
		typrelid = PQgetvalue(__res, 0, 6);
		typelem  = PQgetvalue(__res, 0, 7);
		typtypmod = PQgetvalue(__res, 0, 8);
		if (typtypmod)
		{
			int32_t		__typtypmod = atoi(typtypmod);

			if (atttypmod < 0 && __typtypmod > 0)
				atttypmod = __typtypmod;
		}
		extname  = (PQgetisnull(__res, 0, 9) ? NULL : PQgetvalue(__res, 0, 9));

		if (af_info)
		{
			arrow_field = &af_info->footer.schema.fields[j];
			if (strcmp(arrow_field->name, attname) != 0)
			{
				fprintf(stderr, "Query results field '%s' has incompatible name with Arrow field '%s', keep the original one.\n",
						attname, arrow_field->name);
				attname = arrow_field->name;
			}
		}
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
							  extname,
							  arrow_field,
							  &table->numFieldNodes,
							  &table->numBuffers);
		PQclear(__res);
	}
	return table;
}

/*
 * sqldb_server_connect - open and init session
 */
void *
sqldb_server_connect(const char *sqldb_hostname,
                     const char *sqldb_port_num,
                     const char *sqldb_username,
                     const char *sqldb_password,
                     const char *sqldb_database,
                     userConfigOption *session_config_list)
{
	PGSTATE	   *pgstate;
	PGconn	   *conn;
	PGresult   *res;
	const char *query;
	userConfigOption *conf;

	conn = pgsql_server_connect(sqldb_hostname,
								sqldb_port_num,
								sqldb_username,
								sqldb_password,
								sqldb_database);
	/*
	 * preset user's config option
	 */
	for (conf = session_config_list; conf; conf = conf->next)
	{
		res = PQexec(conn, conf->query);
		if (PQresultStatus(res) != PGRES_COMMAND_OK)
			Elog("failed on change parameter: %s", conf->query);
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
	query = "show timezone";
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_TUPLES_OK ||
		PQntuples(res) != 1 ||
		PQgetisnull(res, 0, 0))
		Elog("failed on getting server timezone configuration: %s", query);
	server_timezone = strdup(PQgetvalue(res, 0, 0));
	if (!server_timezone)
		Elog("out of memory");

	/* allocate PGSTATE */
	pgstate = palloc0(sizeof(PGSTATE));
	pgstate->conn = conn;
	pgstate->res  = NULL;
	return pgstate;
}

/*
 * sqldb_begin_query
 */
SQLtable *
sqldb_begin_query(void *sqldb_state,
                  const char *sqldb_command,
                  ArrowFileInfo *af_info,
                  SQLdictionary *dictionary_list)
{
	static char *snapshot_identifier = NULL;
	PGSTATE	   *pgstate = sqldb_state;
	PGconn	   *conn = pgstate->conn;
	PGresult   *res;
	char	   *query;

	/* begin read-only transaction */
	res = PQexec(conn, "BEGIN READ ONLY");
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("unable to begin transaction: %s", PQresultErrorMessage(res));
	PQclear(res);

	res = PQexec(conn, "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ");
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("unable to switch transaction isolation level: %s",
			 PQresultErrorMessage(res));
	PQclear(res);

	/* export snaphot / import snapshot */
	if (!snapshot_identifier)
	{
		res = PQexec(conn, "SELECT pg_catalog.pg_export_snapshot()");
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("unable to export the current transaction snapshot: %s",
				 PQresultErrorMessage(res));
		if (PQntuples(res) != 1 || PQnfields(res) != 1)
			Elog("unexpected result for pg_export_snapshot()");
		snapshot_identifier = pstrdup(PQgetvalue(res, 0, 0));
		PQclear(res);
	}
	else
	{
		char	temp[200];

		snprintf(temp, sizeof(temp),
				 "SET TRANSACTION SNAPSHOT '%s'",
				 snapshot_identifier);
		res = PQexec(conn, temp);
		if (PQresultStatus(res) != PGRES_COMMAND_OK)
			Elog("unable to import transaction shapshot: %s",
				 PQresultErrorMessage(res));
	}

	/* declare cursor */
	query = palloc(strlen(sqldb_command) + 1024);
	sprintf(query, "DECLARE " CURSOR_NAME " BINARY CURSOR FOR %s",
			sqldb_command);
	res = PQexec(conn, query);
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("unable to declare a SQL cursor: %s", PQresultErrorMessage(res));
	PQclear(res);

	/* fetch schema definition */
	res = PQexecParams(conn,
					   "FETCH FORWARD 100 FROM " CURSOR_NAME,
					   0, NULL, NULL, NULL, NULL,
					   1);	/* results in binary mode */
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("SQL execution failed: %s", PQresultErrorMessage(res));
	pgstate->res = res;
	pgstate->nitems = PQntuples(res);
	pgstate->index  = 0;

	return pgsql_create_buffer(pgstate,
							   af_info,
							   dictionary_list);
}

/*
 * sqldb_fetch_results
 */
bool
sqldb_fetch_results(void *sqldb_state, SQLtable *table)
{
	PGSTATE	   *pgstate = sqldb_state;
	PGresult   *res;
	uint32_t	index;
	size_t		usage = 0;

	if (!pgsql_move_next(pgstate))
		return false;		/* end of the scan */

	res = pgstate->res;
	index = pgstate->index++;
	for (int j=0; j < table->nfields; j++)
	{
		SQLfield   *column = &table->columns[j];
		const char *addr;
		size_t		sz;

		/* data must be binary format */
		if (j >= PQnfields(res) || PQgetisnull(res, index, j))
		{
			addr = NULL;
			sz = 0;
		}
		else
		{
			assert(PQfformat(res, j) == 1);
			addr = PQgetvalue(res, index, j);
			sz = PQgetlength(res, index, j);
		}
		usage += sql_field_put_value(column, addr, sz);
	}
	table->usage = usage;
	table->nitems++;

	return true;
}

void
sqldb_close_connection(void *sqldb_state)
{
	PGSTATE	   *pgstate = sqldb_state;
	PGconn	   *conn = pgstate->conn;
	PGresult   *res;

	if (pgstate->res)
		PQclear(pgstate->res);
	/* close the cursor */
	res = PQexec(conn, "CLOSE " CURSOR_NAME);
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("failed on close cursor '%s': %s", CURSOR_NAME,
			 PQresultErrorMessage(res));
	PQclear(res);
	/* close the connection */
	PQfinish(conn);
}

/*
 * sqldb_build_simple_command
 */
extern int	parseParallelDistKeys(const char *parallel_dist_keys,
								  const char *delim);
#define __RELKIND_LABEL(relkind)								\
	(strcmp((relkind), "r") == 0 ? "table" :					\
	 strcmp((relkind), "i") == 0 ? "index" :					\
	 strcmp((relkind), "S") == 0 ? "sequence" :					\
	 strcmp((relkind), "t") == 0 ? "toast values" :				\
	 strcmp((relkind), "v") == 0 ? "view" :						\
	 strcmp((relkind), "m") == 0 ? "materialized view" :		\
	 strcmp((relkind), "c") == 0 ? "composite type" :			\
	 strcmp((relkind), "f") == 0 ? "foreign table" :			\
	 strcmp((relkind), "p") == 0 ? "partitioned table" :		\
	 strcmp((relkind), "I") == 0 ? "partitioned index" : "???")

static char *
__sqldb_build_simple_command(PGSTATE *pgstate,
							 const char *simple_table_name,
							 int num_worker_threads,
							 size_t batch_segment_sz)
{
	PGconn	   *conn = pgstate->conn;
	PGresult   *res;
	char	   *sql = alloca(strlen(simple_table_name) + 1000);
	char	   *relkind;
	int			inhcnt;

	sprintf(sql,
			"SELECT c.relname, c.relkind, "
			"       (SELECT count(*)"
			"          FROM pg_catalog.pg_inherits"
			"         WHERE inhparent = c.oid) inhcnt"
			"  FROM pg_catalog.pg_class c"
			" WHERE oid = '%s'::regclass",
			simple_table_name);
	res = PQexec(conn, sql);
	if (PQresultStatus(res) != PGRES_TUPLES_OK)
		Elog("unable to check '%s' property: %s",
			 simple_table_name, PQresultErrorMessage(res));
	relkind = pstrdup(PQgetvalue(res, 0, 1));
	inhcnt  = atoi(PQgetvalue(res, 0, 2));
	PQclear(res);

	if (strcmp(relkind, "p") == 0 ||	/* RELKIND_PARTITIONED_TABLE */
		inhcnt != 0)					/* has inherited children */
	{
		/* check whether the supported relation of not */
		sprintf(sql,
				"WITH RECURSIVE r AS (\n"
				" SELECT NULL::regclass parent, '%s'::regclass table\n"
				"  UNION ALL\n"
				" SELECT inhparent, inhrelid\n"
				"   FROM pg_catalog.pg_inherits, r\n"
				"  WHERE inhparent = r.table\n"
				")\n"
				"SELECT r.*, c.relkind\n"
				"  FROM r, pg_catalog.pg_class c\n"
				" WHERE r.table = c.oid",
				simple_table_name);
		res = PQexec(conn, sql);
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("failed on [%s]: %s", sql, PQresultErrorMessage(res));
		for (int i=0; i < PQntuples(res); i++)
		{
			relkind = PQgetvalue(res, i, 2);
			if (strcmp(relkind, "r") != 0 &&
				strcmp(relkind, "m") != 0 &&
				strcmp(relkind, "t") != 0 &&
				strcmp(relkind, "p") != 0)
				Elog("unable to attach hashtid(ctid) on '%s' [%s]",
					 PQgetvalue(res, i, 1), __RELKIND_LABEL(relkind));
		}
		sprintf(sql,
				"SELECT * FROM %s WHERE hashtid(ctid) %% $(N_WORKERS) = $(WORKER_ID)",
				simple_table_name);
	}
	else if (strcmp(relkind, "r") == 0 ||	/* RELKIND_RELATION */
			 strcmp(relkind, "m") == 0 ||	/* RELKIND_MATVIEW */
			 strcmp(relkind, "t") == 0)		/* RELKIND_TOASTVALUE */
	{
		/* determine the block size per client */
		uint64_t	num_blocks;
		uint64_t	per_client;
		char	   *conds;
		char	   *pos;

		sprintf(sql,
				"SELECT GREATEST(pg_relation_size('%s'::regclass), %lu)"
				"     / current_setting('block_size')::int",
				simple_table_name, batch_segment_sz * num_worker_threads);
		res = PQexec(conn, sql);
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("failed on [%s]: %s", sql, PQresultErrorMessage(res));
		num_blocks = atol(PQgetvalue(res, 0, 0));
		per_client = num_blocks / num_worker_threads;
		PQclear(res);
		assert(per_client * (num_worker_threads - 1) < UINT_MAX);

		pos = conds = alloca(100 * num_worker_threads);
		for (int i=1; i <= num_worker_threads; i++)
		{
			if (i == 1)
				pos += sprintf(pos, "ctid < '(%lu,0)'::tid",
							   per_client);
			else if (i == num_worker_threads)
				pos += sprintf(pos, "\tctid >= '(%lu,0)'::tid",
							   per_client * (i-1));
			else
				pos += sprintf(pos, "\tctid >= '(%lu,0)'::tid AND ctid < '(%lu,0)'::tid",
							   per_client * (i-1),
							   per_client * i);
		}
		parseParallelDistKeys(conds, "\t");
		sprintf(sql, "SELECT * FROM %s WHERE $(PARALLEL_KEY)",
				simple_table_name);
	}
	else
	{
		Elog("unable to dump '%s' [%s] using -t, use -c instead",
			 simple_table_name, __RELKIND_LABEL(relkind));
	}
	return pstrdup(sql);
}

char *
sqldb_build_simple_command(void *sqldb_state,
						   const char *simple_table_name,
						   int num_worker_threads,
						   size_t batch_segment_sz)
{
	assert(num_worker_threads > 0);
	if (num_worker_threads == 1)
	{
		char   *buf = alloca(strlen(simple_table_name) + 80);

		sprintf(buf, "SELECT * FROM %s", simple_table_name);
		return pstrdup(buf);
	}
	return __sqldb_build_simple_command((PGSTATE *)sqldb_state,
										simple_table_name,
										num_worker_threads,
										batch_segment_sz);
}
