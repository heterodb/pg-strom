/*
 * dpu_device.c
 *
 * Misc routines to support DPU (Smart-NIC/Smart-SSD) devices
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include <netdb.h>

static char	   *pgstrom_dpu_endpoint_list;	/* GUC */
static int		pgstrom_dpu_endpoint_default_port;	/* GUC */
#define PGSTROM_DPU_ENDPOINT_DEFAULT_PORT	6543
#define DEFAULT_DPU_SETUP_COST		(100 * DEFAULT_SEQ_PAGE_COST)
#define DEFAULT_DPU_OPERATOR_COST	(1.2 * DEFAULT_CPU_OPERATOR_COST)
#define DEFAULT_DPU_SEQ_PAGE_COST	(DEFAULT_SEQ_PAGE_COST / 4)
#define DEFAULT_DPU_TUPLE_COST		(DEFAULT_CPU_TUPLE_COST)

double		pgstrom_dpu_setup_cost    = DEFAULT_DPU_SETUP_COST;		/* GUC */
double		pgstrom_dpu_operator_cost = DEFAULT_DPU_OPERATOR_COST;	/* GUC */
double		pgstrom_dpu_seq_page_cost = DEFAULT_DPU_SEQ_PAGE_COST;	/* GUC */
double		pgstrom_dpu_tuple_cost    = DEFAULT_DPU_TUPLE_COST;		/* GUC */
bool		pgstrom_dpu_handle_cached_pages = false;	/* GUC */

struct DpuStorageEntry
{
	uint32_t	endpoint_id;
	const char *endpoint_dir;
	const char *config_host;
	const char *config_port;
	int			endpoint_domain;	/* AF_UNIX/AF_INET/AF_INET6 */
	const struct sockaddr *endpoint_addr;
	socklen_t	endpoint_addr_len;
	pg_atomic_uint32 *validated;	/* shared memory */
};

typedef struct
{
	uint32_t	nitems;
	DpuStorageEntry entries[1];
} DpuStorageArray;

static DpuStorageArray		   *dpu_tablespace_master = NULL;
static shmem_request_hook_type	shmem_request_next = NULL;
static shmem_startup_hook_type	shmem_startup_next = NULL;

/*
 * GetOptimalDpuForFile
 */
static DpuStorageEntry *
__getOptimalDpuForFile(const char *pathname)
{
	DpuStorageEntry *ds_entry = NULL;
	struct stat		stat_buf;

	if (strcmp(pathname, "/") != 0)
	{
		char   *namebuf = alloca(strlen(pathname) + 1);

		strcpy(namebuf, pathname);
		ds_entry = __getOptimalDpuForFile(dirname(namebuf));
	}

	if (stat(pathname, &stat_buf) == 0 &&
		S_ISDIR(stat_buf.st_mode))
	{
		for (int i=0; i < dpu_tablespace_master->nitems; i++)
		{
			DpuStorageEntry *curr = &dpu_tablespace_master->entries[i];
			struct stat		curr_buf;

			if (stat(curr->endpoint_dir, &curr_buf) == 0 &&
				S_ISDIR(curr_buf.st_mode) &&
				stat_buf.st_dev == curr_buf.st_dev &&
				stat_buf.st_ino == curr_buf.st_ino)
			{
				ds_entry = curr;
				break;
			}
		}
	}
	return ds_entry;
}

DpuStorageEntry *
GetOptimalDpuForFile(const char *filename)
{
	char   *path;
	size_t	len;

	/* quick bailout */
	if (!dpu_tablespace_master)
		return NULL;
	/* absolute path? */
	if (*filename == '/')
		return __getOptimalDpuForFile(filename);
	len = strlen(DataDir) + strlen(filename) + 10;
	path = alloca(len);
	snprintf(path, len, "%s/%s", DataDir, filename);
	return __getOptimalDpuForFile(path);
}

/*
 * Tablespace-DPU hash table
 */
typedef struct
{
	Oid		tablespace_oid;
	DpuStorageEntry	*ds_entry;
} DpuTablespaceCache;

static HTAB	   *dpu_tablespace_htable = NULL;

static void
dpu_tablespace_htable_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	hash_destroy(dpu_tablespace_htable);
	dpu_tablespace_htable = NULL;
}

/*
 * GetOptimalDpuForRelation
 */
DpuStorageEntry *
GetOptimalDpuForTablespace(Oid tablespace_oid)
{
	DpuTablespaceCache *dt_cache;
	bool		found;

	/* quick bailout */
	if (!dpu_tablespace_master)
		return NULL;	/* quick bailout */
	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;
	if (!dpu_tablespace_htable)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(DpuTablespaceCache);
		hctl.hcxt = CacheMemoryContext;
		dpu_tablespace_htable
			= hash_create("DPU-Tablespace hashtable",
						  256,
						  &hctl,
						  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}
	dt_cache = hash_search(dpu_tablespace_htable,
						   &tablespace_oid,
						   HASH_ENTER,
						   &found);
	if (!found)
	{
		char   *pathname = GetDatabasePath(MyDatabaseId, tablespace_oid);

		dt_cache->ds_entry =  GetOptimalDpuForFile(pathname);
	}
	return dt_cache->ds_entry;
}

/*
 * GetOptimalDpuForRelation
 */
DpuStorageEntry *
GetOptimalDpuForRelation(Relation relation)
{
	Oid		tablespace_oid = RelationGetForm(relation)->reltablespace;

	return GetOptimalDpuForTablespace(tablespace_oid);	
}

/*
 * DpuStorageEntryIsEqual
 */
bool
DpuStorageEntryIsEqual(const DpuStorageEntry *ds_entry1,
					   const DpuStorageEntry *ds_entry2)
{
	if (ds_entry1 && ds_entry2)
		return (ds_entry1->endpoint_id == ds_entry2->endpoint_id);
	else if (!ds_entry1 && !ds_entry2)
		return true;
	else
		return false;
}

/*
 * DpuClientOpenSession 
 */
void
DpuClientOpenSession(pgstromTaskState *pts,
					 const XpuCommand *session)
{
	const DpuStorageEntry *ds_entry = pts->ds_entry;
    pgsocket    sockfd;
    char        namebuf[32];

	if (!ds_entry)
		elog(ERROR, "Bug? no DPU device is configured");

	sockfd = socket(ds_entry->endpoint_domain, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2) dom=%d: %m", ds_entry->endpoint_domain);
	if (connect(sockfd,
				ds_entry->endpoint_addr,
				ds_entry->endpoint_addr_len) != 0)
	{
		close(sockfd);
		elog(ERROR, "failed on connect('%s'): %m", ds_entry->config_host);
	}
	snprintf(namebuf, sizeof(namebuf), "DPU-%u", ds_entry->endpoint_id);

	__xpuClientOpenSession(pts, session, sockfd, namebuf);
}

/*
 * parse_dpu_endpoint_list
 */
static bool
parse_dpu_endpoint_list(void)
{
	char	   *tok, *saveptr;
	char	   *buf;
	uint32_t	endpoint_id = 0;
	uint32_t	nrooms = 48;
	uint32_t	nitems = 0;
	char		__default_port[32];
	
	if (!pgstrom_dpu_endpoint_list)
		return 0;
	dpu_tablespace_master = malloc(offsetof(DpuStorageArray,
											entries[nrooms]));
	if (!dpu_tablespace_master)
		elog(ERROR, "out of memory");
	sprintf(__default_port, "%u", pgstrom_dpu_endpoint_default_port);

	buf = alloca(strlen(pgstrom_dpu_endpoint_list) + 1);
	strcpy(buf, pgstrom_dpu_endpoint_list);
	for (tok = strtok_r(buf, ",", &saveptr);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &saveptr))
	{
		DpuStorageEntry *curr;
		char	   *name;
		char	   *host;
		char	   *port;
		struct addrinfo hints, *addr;

		name = strchr(tok, '=');
		if (!name)
			elog(ERROR, "pg_strom.dpu_endpoint_list - invalid token [%s]", tok);
		*name++ = '\0';
		host = __trim(tok);
		name = __trim(name);
		if (*name != '/')
			elog(ERROR, "endpoint directory must be absolute path");

		if (nitems >= nrooms)
		{
			nrooms *= 2;
			dpu_tablespace_master = realloc(dpu_tablespace_master,
											offsetof(DpuStorageArray,
													 entries[nrooms]));
		}
		curr = &dpu_tablespace_master->entries[nitems++];
		memset(curr, 0, sizeof(DpuStorageEntry));

		curr->endpoint_id = endpoint_id++;
		curr->endpoint_dir = strdup(name);
		if (!curr->endpoint_dir)
			elog(ERROR, "out of memory");

		memset(&hints, 0, sizeof(struct addrinfo));
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_STREAM;
		/* TODO: IPv6 support */
		port = strrchr(host, ':');
		if (port)
			*port++ = '\0';
		else
			port = __default_port;
		if (getaddrinfo(host, port, &hints, &addr) != 0)
			elog(ERROR, "failed on getaddrinfo('%s','%s')", host, port);

		curr->config_host = strdup(host);
		curr->config_port = strdup(port);
		if (!curr->config_host || !curr->config_port)
			elog(ERROR, "out of memory");
		curr->endpoint_domain = addr->ai_family;
		curr->endpoint_addr = addr->ai_addr;
		curr->endpoint_addr_len = addr->ai_addrlen;
	}
	dpu_tablespace_master->nitems   = nitems;

	return (nitems > 0);
}

/*
 * pgstrom_request_dpu_device
 */
static void
pgstrom_request_dpu_device(void)
{
	if (shmem_request_next)
		shmem_request_next();
	RequestAddinShmemSpace(MAXALIGN(sizeof(pg_atomic_uint32) *
									dpu_tablespace_master->nitems));
}

/*
 * pgstrom_startup_dpu_device
 */
static void
pgstrom_startup_dpu_device(void)
{
	pg_atomic_uint32 *validated;
	uint32_t	nitems = dpu_tablespace_master->nitems;
	bool		found;

	if (shmem_startup_next)
		shmem_startup_next();
	Assert(nitems > 0);
	validated = ShmemInitStruct("DPU-Tablespace Info",
								sizeof(pg_atomic_uint32) * nitems,
								&found);
	for (int i=0; i < dpu_tablespace_master->nitems; i++)
	{
		dpu_tablespace_master->entries[i].validated = &validated[i];
	}
}

/*
 * pgstrom_init_dpu_options
 */
static void
pgstrom_init_dpu_options(void)
{
	/* cost factor for DPU setup */
	DefineCustomRealVariable("pg_strom.dpu_setup_cost",
							 "Cost to setup DPU device to run",
							 NULL,
							 &pgstrom_dpu_setup_cost,
							 DEFAULT_DPU_SETUP_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for DPU operator */
	DefineCustomRealVariable("pg_strom.dpu_operator_cost",
							 "Cost of processing each operators by DPU",
							 NULL,
							 &pgstrom_dpu_operator_cost,
							 DEFAULT_DPU_OPERATOR_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for DPU disk scan */
	DefineCustomRealVariable("pg_strom.dpu_seq_page_cost",
							 "Default cost to scan page on DPU device",
							 NULL,
							 &pgstrom_dpu_seq_page_cost,
							 DEFAULT_DPU_SEQ_PAGE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for DPU<-->Host data transfer per tuple */
	DefineCustomRealVariable("pg_strom.dpu_tuple_cost",
							 "Default cost to transfer DPU<->Host per tuple",
							 NULL,
							 &pgstrom_dpu_tuple_cost,
							 DEFAULT_DPU_TUPLE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* control whether DPU handles cached pages */
	DefineCustomBoolVariable("pg_strom.dpu_handle_cached_pages",
							 "Control whether DPUs handles cached clean pages",
							 NULL,
							 &pgstrom_dpu_handle_cached_pages,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}

/*
 * pgstrom_init_dpu_device
 */
bool
pgstrom_init_dpu_device(void)
{
	/*
	 * format:
	 * <host/ipaddr>[;<port>]=<pathname>[, ...]
	 */
	DefineCustomStringVariable("pg_strom.dpu_endpoint_list",
							   "List of DPU endpoint definitions for each tablespace",
							   NULL,
							   &pgstrom_dpu_endpoint_list,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.dpu_endpoint_default_port",
							"Default port number of DPU endpoint",
							NULL,
							&pgstrom_dpu_endpoint_default_port,
							PGSTROM_DPU_ENDPOINT_DEFAULT_PORT,
							1,
							65535,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	if (parse_dpu_endpoint_list())
	{
		pgstrom_init_dpu_options();

		shmem_request_next = shmem_request_hook;
		shmem_request_hook = pgstrom_request_dpu_device;
		shmem_startup_next = shmem_startup_hook;
		shmem_startup_hook = pgstrom_startup_dpu_device;

		CacheRegisterSyscacheCallback(TABLESPACEOID,
									  dpu_tablespace_htable_invalidator,
									  (Datum)0);
		/* output logs */
		for (int i=0; i < dpu_tablespace_master->nitems; i++)
		{
			DpuStorageEntry *ds_entry = &dpu_tablespace_master->entries[i];

			elog(LOG, "PG-Strom: DPU%d (dir: '%s', host: '%s', port: '%s')",
				 ds_entry->endpoint_id,
				 ds_entry->endpoint_dir,
				 ds_entry->config_host,
				 ds_entry->config_port);
		}
		return true;
	}
	return false;
}
