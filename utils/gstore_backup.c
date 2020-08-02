/*
 * gstore_backup.c
 *
 * A utility to make real-time backup of Gstore_Fdw.
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "postgres.h"
#include "catalog/pg_type_d.h"
#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <libpq-fe.h>
#include "gstore_fdw.h"

/* ---- static variables ---- */
static char	   *pgsql_hostname = NULL;
static char	   *pgsql_port_num = NULL;
static char	   *pgsql_username = NULL;
static char	   *pgsql_password = NULL;
static char	   *pgsql_database = NULL;
static char	   *pgsql_tablename = NULL;
static char	   *base_filename = NULL;
static char	   *redo_filename = NULL;
static int		base_fdesc = -1;
static long		PAGE_SIZE;

#define Elog(fmt, ...)                              \
	do {                                            \
		fprintf(stderr,"%s:%d  " fmt "\n",          \
				basename(__FILE__),__LINE__, ##__VA_ARGS__);	\
		exit(1);                                    \
	} while(0)

static ssize_t
__Write(int fdesc, const void *buf, size_t nbytes)
{
	ssize_t		rv, offset = 0;

	do {
		rv = write(fdesc, (char *)buf + offset, nbytes - offset);
		if (rv > 0)
			offset += rv;
		else if (rv == 0)
			break;
		else if (errno != EINTR)
			return -1;
	} while (offset < nbytes);

	return offset;
}

#if 0
static ssize_t
__Read(int fdesc, void *buf, size_t nbytes)
{
	ssize_t		rv, offset = 0;

	do {
		rv = read(fdesc, (char *)buf + offset, nbytes - offset);
		if (rv > 0)
			offset += rv;
		else if (rv == 0)
			break;
		else if (errno != EINTR)
			return -1;
	} while (offset < nbytes);

	return offset;
}
#endif

static PGconn *
pgsql_server_connect(const char *pgsql_hostname,
					 const char *pgsql_port_num,
					 const char *pgsql_username,
					 const char *pgsql_password,
					 const char *pgsql_database)
{
	PGconn	   *conn;
	const char *keys[20];
	const char *values[20];
	int			index = 0;
	int			status;

	memset(keys, 0, sizeof(keys));
	memset(values, 0, sizeof(values));
	if (pgsql_hostname)
	{
		keys[index] = "host";
		values[index] = pgsql_hostname;
		index++;
	}
	if (pgsql_port_num)
	{
		keys[index] = "port";
		values[index] = pgsql_port_num;
		index++;
	}
	if (pgsql_username)
	{
		keys[index] = "user";
		values[index] = pgsql_username;
		index++;
	}
	if (pgsql_password)
	{
		keys[index] = "password";
		values[index] = pgsql_password;
		index++;
	}
	if (pgsql_database)
	{
		keys[index] = "dbname";
		values[index] = pgsql_database;
		index++;
	}
	keys[index] = "application_name";
	values[index] = "gstore_backup";
	index++;

	conn = PQconnectdbParams(keys, values, 0);
	if (!conn)
		Elog("out of memory");
	status = PQstatus(conn);
	if (status != CONNECTION_OK)
		Elog("failed on PostgreSQL connection: %s",
			 PQerrorMessage(conn));
	return conn;
}

static uint64
build_base_backup(PGconn *conn, int fdesc)
{
	PGresult   *res;
	const char *command;
	GpuStoreBaseFileHead baseHead;
	kern_data_extra extraHead;
	size_t		base_sz = 0;
	size_t		extra_sz = 0;
	char		countBuf[20];
	Oid			paramTypes[2];
	const char *paramValues[2];
	int32		count = 0;
	char		phase = 'b';
	uint64		next_lpos = 0;

	/* begin transaction */
	res = PQexec(conn, "BEGIN");
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("SQL execution failed: %s", PQresultErrorMessage(res));
	PQclear(res);

	/* fetch base & extra portion */
	command = "SELECT pgstrom.gstore_fdw_replication_base($1,$2)";
	paramTypes[0] = TEXTOID;
	paramTypes[1] = INT4OID;
	paramValues[0] = pgsql_tablename;
	paramValues[1] = countBuf;

	for (count=0; ; count++)
	{
		GpuStoreReplicationChunk *chunk = NULL;
		size_t		chunk_sz = 0;
		size_t		len;

		sprintf(countBuf, "%d", count);
		res = PQexecParams(conn, command, 2,
						   paramTypes,
						   paramValues,
						   NULL,
						   NULL,
						   1);		/* result should be binary format */
		if (PQresultStatus(res) != PGRES_TUPLES_OK)
			Elog("SQL execution failed [%s] $1='%s', $2='%d'",
				 command, pgsql_tablename, count);
		if (PQnfields(res) != 1 || PQntuples(res) != 1)
			Elog("unexpected number of columns/rows returned for [%s]",
				 command);
	retry:
		if (phase == 'b')
		{
			if (!PQgetisnull(res, 0, 0))
			{
				chunk = (GpuStoreReplicationChunk *)PQgetvalue(res, 0, 0);
				chunk_sz = PQgetlength(res, 0, 0);
				assert(chunk_sz > offsetof(GpuStoreReplicationChunk, data));
				/* save the header portion */
				if (base_sz < sizeof(GpuStoreBaseFileHead))
				{
					len = Min(chunk_sz - offsetof(GpuStoreReplicationChunk, data),
							  sizeof(GpuStoreBaseFileHead) - base_sz);
					memcpy(((char *)&baseHead) + base_sz, chunk->data, len);
				}
				next_lpos = chunk->rep_lpos;
			}

			if (chunk && chunk->rep_kind == 'b')
			{
				assert(chunk_sz > offsetof(GpuStoreReplicationChunk, data));
				len = chunk_sz - offsetof(GpuStoreReplicationChunk, data);
				if (__Write(fdesc, chunk->data, len) != len)
					Elog("failed on __Write('%s'): %m", base_filename);
				base_sz += len;
			}
			else
			{
				cl_uint		nrooms = baseHead.schema.nrooms;
				cl_uint		nslots;
				
				/* end of the base chunk, switch to the extra chunk */
				if (base_sz < sizeof(GpuStoreBaseFileHead))
					Elog("Bug? base chunk of '%s' is incomplete", pgsql_tablename);

				assert(baseHead.rowid_map_offset >= base_sz);
				len = (baseHead.rowid_map_offset +
					   MAXALIGN(offsetof(GpuStoreRowIdMapHead,
										 rowid_chain[nrooms])));
				len = TYPEALIGN(PAGE_SIZE, len);

				if (baseHead.hash_index_offset > 0)
				{
					/* with primary key, thus hash-index exists */
					if (baseHead.hash_index_offset != len)
						Elog("Bug? location of hash-index is corrupted");
					nslots = Min(1.2 * (double)nrooms + 1000.0, UINT_MAX);
					len += offsetof(GpuStoreHashIndexHead,
									slots[nslots + nrooms]);
					len = TYPEALIGN(PAGE_SIZE, len);
				}

				if (ftruncate(fdesc, len) < 0)
					Elog("failed on ftruncate('%s',%zu): %m", base_filename, len);
				if (lseek(fdesc, 0, SEEK_END) < 0)
					Elog("failed on lseek('%s', 0, SEEK_END): %m", base_filename);
				base_sz = len;

				phase = chunk->rep_kind;
				if (chunk)
					goto retry;
			}
		}
		else if (phase == 'e')
		{
			if (!PQgetisnull(res, 0, 0))
			{
				chunk = (GpuStoreReplicationChunk *)PQgetvalue(res, 0, 0);
				chunk_sz = PQgetlength(res, 0, 0);
				assert(chunk_sz > offsetof(GpuStoreReplicationChunk, data));
				/* save the header portion */
				if (extra_sz < sizeof(kern_data_extra))
				{
					len = Min(chunk_sz - offsetof(GpuStoreReplicationChunk, data),
							  sizeof(kern_data_extra) - extra_sz);
					memcpy((char *)&extraHead + extra_sz, chunk->data, len);
				}
				next_lpos = chunk->rep_lpos;
			}

			if (chunk && chunk->rep_kind == 'e')
			{
				/* sanity checks */
				if (base_sz != (offsetof(GpuStoreBaseFileHead,
										 schema) + baseHead.schema.extra_hoffset))
					Elog("Bug? Location of extra buffer is corrupted");
				assert(chunk_sz > offsetof(GpuStoreReplicationChunk, data));
				len = chunk_sz - offsetof(GpuStoreReplicationChunk, data);
				if (__Write(fdesc, chunk->data, len) != len)
					Elog("failed on __Write('%s'): %m", base_filename);
				extra_sz += len;
			}
			else
			{
				/* end of the extra chunk, enlarge buffer to fit extra->length */
				if (extra_sz < sizeof(kern_data_extra))
					Elog("Bug? extra chunk of '%s' is incomplete", pgsql_tablename);
				assert(extraHead.length >= extra_sz);
				len = (base_sz + extraHead.length);
				if (ftruncate(fdesc, len) < 0)
					Elog("failed on ftruncate('%s',%zu): %m", base_filename, len);
				if (chunk)
					goto retry;		/* should not happen */
			}
		}
		else
			Elog("Bug? wrong internal phase '%c'", phase);
		PQclear(res);
		if (!chunk)
			break;
	}
	/* end transaction - release exclusive lock */
	res = PQexec(conn, "COMMIT");
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
		Elog("SQL execution failed: %s", PQresultErrorMessage(res));
	PQclear(res);
	
	return next_lpos;
}





static void
usage(int exitcode)
{
	fputs("gstore_backup [OPTIONS] BASE_FILE\n"
		  "\n"
		  "General options:\n"
		  "  -d, --dbname=DBNAME    database name to connect\n"
		  "  -t, --table=TABLENAME  table name for backup\n"
		  "  -r, --redo-log=FILENAME filename to store redo-log (optional)\n"
		  "\n"
		  "Connection options:\n"
		  "  -h, --host=HOSTNAME    database server host\n"
		  "  -p, --port=PORT        database server port\n"
		  "  -u, --user=USERNAME    database user name\n"
		  "  -w, --no-password      never prompt for password\n"
		  "  -W, --password         force password prompt\n",
		  stderr);
	exit(exitcode);
}

static void
parse_options(int argc, char * const argv[])
{
	static struct option long_options[] = {
		{"dbname",      required_argument, NULL,  'd' },
		{"table",       required_argument, NULL,  't' },
		{"redo-log",    required_argument, NULL,  'r' },
		{"host",        required_argument, NULL,  'h' },
		{"port",        required_argument, NULL,  'p' },
		{"user",        required_argument, NULL,  'u' },
		{"no-password", no_argument,       NULL,  'w' },
		{"password",    no_argument,       NULL,  'W' },
		{"help",        no_argument,       NULL, 9999 },
	};
	int		password_prompt = 0;
	int		c;

	while ((c = getopt_long(argc, argv, "d:t:r:h:p:u:wW",
							long_options, NULL)) >= 0)
	{
		switch (c)
		{
			case 'd':
				if (pgsql_database)
					Elog("-d option was supplied twice");
				pgsql_database = optarg;
				break;

			case 't':
				if (pgsql_tablename)
					Elog("-t option was supplied twice");
				pgsql_tablename = optarg;
				break;

			case 'r':
				if (redo_filename)
					Elog("-r option was supplied twice");
				redo_filename = optarg;
				break;

			case 'h':
				if (pgsql_hostname)
					Elog("-h option was supplied twice");
				pgsql_hostname = optarg;
				break;

			case 'p':
				if (pgsql_port_num)
					Elog("-p option was supplied twice");
				pgsql_port_num = optarg;
				break;

			case 'u':
				if (pgsql_username)
					Elog("-u option was supplied twice");
				pgsql_username = optarg;
				break;

			case 'w':
				if (password_prompt > 0)
					Elog("-w and -W option are exclusive");
				password_prompt = -1;
				break;

			case 'W':
				if (password_prompt < 0)
					Elog("-w and -W option are exclusive");
				password_prompt = 1;
				break;

			default:
				usage(1);
		}
	}

	if (optind + 1 != argc)
		Elog("BASEFILE was not specified");
	base_filename = argv[optind];

	if (redo_filename)
		Elog("-r option is not implemented yet");

	if (password_prompt > 0)
	{
		pgsql_password = strdup(getpass("Password: "));
		if (!pgsql_password)
			Elog("out of memory");
	}
}

static void
on_exit_cleanup(int status, void *arg)
{
	if (status == 0)
		return;		/* exit successfully */
	if (unlink(base_filename) != 0)
		fprintf(stderr, "failed on unlink('%s'): %m", base_filename);
}

int
main(int argc, char * const argv[])
{
	PGconn	   *conn;
	const char *dest_filename = NULL;
	uint64		next_lpos;

	/* system parameters */
	PAGE_SIZE = sysconf(_SC_PAGESIZE);

	parse_options(argc, argv);
	/* open the base file */
	base_fdesc = open(base_filename, O_RDWR | O_CREAT | O_EXCL, 0600);
	if (base_fdesc < 0)
	{
		if (errno == EEXIST)
		{
			dest_filename = base_filename;
			base_filename = malloc(strlen(dest_filename) + 10);
			sprintf(base_filename, "%s.XXXXXX", dest_filename);
			base_fdesc = mkstemp(base_filename);
		}
		if (base_fdesc < 0)
			Elog("failed on open('%s'): %m", base_filename);
	}
	on_exit(on_exit_cleanup, NULL);

	conn = pgsql_server_connect(pgsql_hostname,
								pgsql_port_num,
								pgsql_username,
								pgsql_password,
								pgsql_database);
	next_lpos = build_base_backup(conn, base_fdesc);

	printf("next_lpos = %lu\n", next_lpos);



	PQfinish(conn);
	/* switch file, if needed */
	if (dest_filename)
	{
		if (rename(base_filename, dest_filename) != 0)
		{
			unlink(base_filename);
			Elog("failed on rename('%s','%s'): %m",
				 dest_filename,
				 base_filename);
		}
	}
	return 0;
}
