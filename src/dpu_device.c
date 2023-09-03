/*
 * dpu_device.c
 *
 * Misc routines to support DPU (Smart-NIC/Smart-SSD) devices
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
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
	int32_t		endpoint_id;
	const char *endpoint_dir;
	const char *config_host;
	const char *config_port;
	int			endpoint_domain;	/* AF_UNIX/AF_INET/AF_INET6 */
	const struct sockaddr *endpoint_addr;
	socklen_t	endpoint_addr_len;
	struct stat	endpoint_stat_buf;
	pg_atomic_uint32 *validated;	/* shared memory */
};

typedef struct
{
	uint32_t	nitems;
	DpuStorageEntry entries[1];
} DpuStorageArray;

static DpuStorageArray		   *dpu_storage_master_array = NULL;
static shmem_request_hook_type	shmem_request_next = NULL;
static shmem_startup_hook_type	shmem_startup_next = NULL;

/*
 * GetOptimalDpuForFile
 */
static DpuStorageEntry *
__getOptimalDpuForFile(const char *pathname, StringInfo dpu_path)
{
	DpuStorageEntry *ds_entry = NULL;
	char		namebuf[MAXPGPATH];
	ssize_t		nbytes;
	struct stat	stat_buf;

	if (lstat(pathname, &stat_buf) == 0)
	{
		for (int i=0; i < dpu_storage_master_array->nitems; i++)
		{
			DpuStorageEntry *curr = &dpu_storage_master_array->entries[i];

			if (curr->endpoint_stat_buf.st_mode == 0)
			{
				if (stat(curr->endpoint_dir, &curr->endpoint_stat_buf) != 0)
					continue;
			}
			if (stat_buf.st_dev == curr->endpoint_stat_buf.st_dev &&
				stat_buf.st_ino == curr->endpoint_stat_buf.st_ino)
			{
				if (dpu_path)
					resetStringInfo(dpu_path);
				return curr;
			}
		}

		if (S_ISLNK(stat_buf.st_mode) &&
			(nbytes = readlink(pathname, namebuf, MAXPGPATH)) > 0)
		{
			namebuf[nbytes] = '\0';
			ds_entry = __getOptimalDpuForFile(namebuf, dpu_path);
			if (ds_entry)
				return ds_entry;
		}

		if (strcmp(pathname, "/") != 0)
		{
			strncpy(namebuf, pathname, MAXPGPATH);
			ds_entry = __getOptimalDpuForFile(dirname(namebuf), dpu_path);
			if (ds_entry)
			{
				strncpy(namebuf, pathname, MAXPGPATH);
				appendStringInfo(dpu_path, "%s%s",
								 dpu_path->len > 0 ? "/" : "",
								 basename(namebuf));
				return ds_entry;
			}
		}
	}
	return NULL;
}

const DpuStorageEntry *
GetOptimalDpuForFile(const char *filename,
					 const char **p_dpu_pathname)
{
	DpuStorageEntry *ds_entry;
	StringInfoData buf;
	char	   *namebuf;
	size_t		len;

	/* quick bailout */
	if (!dpu_storage_master_array)
		return NULL;
	initStringInfo(&buf);
	/* absolute path? */
	if (*filename == '/')
		ds_entry = __getOptimalDpuForFile(filename, &buf);
	else
	{
		len = strlen(DataDir) + strlen(filename) + 10;
		namebuf = alloca(len);
		snprintf(namebuf, len, "%s/%s", DataDir, filename);
		ds_entry = __getOptimalDpuForFile(namebuf, &buf);
	}
	if (p_dpu_pathname)
		*p_dpu_pathname = pstrdup(buf.data);
	pfree(buf.data);

	return ds_entry;
}

/*
 * Relation Cached-DPU hash table invalidator
 */
typedef struct
{
	Oid		relation_oid;
	char   *dpu_pathname;
	const DpuStorageEntry *ds_entry;
} DpuRelCacheItem;

static HTAB	   *dpu_relcache_htable = NULL;

static void
dpu_relcache_htable_invalidator(Datum arg, Oid relation_oid)
{
	if (dpu_relcache_htable)
	{
		hash_search(dpu_relcache_htable,
					&relation_oid,
					HASH_REMOVE,
					NULL);
	}
}

/*
 * GetOptimalDpuForRelation
 */
const DpuStorageEntry *
GetOptimalDpuForRelation(Relation relation, const char **p_dpu_pathname)
{
	DpuRelCacheItem *drc_item;
	Oid		relation_oid;
	bool	found;

	if (!dpu_storage_master_array)
		return NULL;	/* quick bailout */

	relation_oid = RelationGetRelid(relation);
	if (!dpu_relcache_htable)
	{
		HASHCTL	hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(DpuRelCacheItem);
		hctl.hcxt = CacheMemoryContext;
		dpu_relcache_htable
			= hash_create("DPU-Relcache hashtable",
						  1024,
						  &hctl,
						  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}

	drc_item = hash_search(dpu_relcache_htable,
						   &relation_oid,
						   HASH_ENTER,
						   &found);
	if (!found)
	{
		const DpuStorageEntry *ds_entry;
		SMgrRelation smgr = RelationGetSmgr(relation);
		char	   *rel_pathname = smgr_relpath(smgr, MAIN_FORKNUM);
		const char *dpu_pathname;

		ds_entry = GetOptimalDpuForFile(rel_pathname, &dpu_pathname);
		if (ds_entry)
			drc_item->dpu_pathname = MemoryContextStrdup(CacheMemoryContext,
														 dpu_pathname);
		drc_item->ds_entry = ds_entry;
	}
	if (p_dpu_pathname)
		*p_dpu_pathname = drc_item->dpu_pathname;
	return drc_item->ds_entry;
}

/*
 * GetOptimalDpuForBaseRel
 */
const DpuStorageEntry *
GetOptimalDpuForBaseRel(PlannerInfo *root, RelOptInfo *baserel)
{
	RangeTblEntry  *rte = root->simple_rte_array[baserel->relid];
	const DpuStorageEntry *ds_entry = NULL;

	if (!dpu_storage_master_array)
		return NULL;	/* quick bailout */
	if (rte->rtekind == RTE_RELATION)
	{
		DpuRelCacheItem *drc_item;
		Relation	relation;

		/* fast path */
		if (dpu_relcache_htable &&
			(drc_item = hash_search(dpu_relcache_htable,
									&rte->relid,
									HASH_FIND,
									NULL)) != NULL)
			return drc_item->ds_entry;

		relation = table_open(rte->relid, AccessShareLock);
		ds_entry = GetOptimalDpuForRelation(relation, NULL);
		table_close(relation, NoLock);
	}
	return ds_entry;
}

/*
 * DpuStorageEntryBaseDir
 */
const char *
DpuStorageEntryBaseDir(const DpuStorageEntry *ds_entry)
{
	if (ds_entry)
		return ds_entry->endpoint_dir;
	return NULL;
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
 * DpuStorageEntryGetEndpointId
 */
int
DpuStorageEntryGetEndpointId(const DpuStorageEntry *ds_entry)
{
	return ds_entry ? ds_entry->endpoint_id : -1;
}

/*
 * DpuStorageEntryByEndpointId
 */
const DpuStorageEntry *
DpuStorageEntryByEndpointId(int endpoint_id)
{
	if (dpu_storage_master_array &&
		endpoint_id >= 0 &&
		endpoint_id < dpu_storage_master_array->nitems)
	{
		return &dpu_storage_master_array->entries[endpoint_id];
	}
	return NULL;
}

/*
 * DpuStorageEntryCount
 */
int
DpuStorageEntryCount(void)
{
	return (dpu_storage_master_array ? dpu_storage_master_array->nitems : 0);
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

	__xpuClientOpenSession(pts, session, sockfd, namebuf, ds_entry->endpoint_id);
}

/*
 * explainDpuStorageEntry
 */
void
explainDpuStorageEntry(const DpuStorageEntry *ds_entry, ExplainState *es)
{
	char	label[80];
	StringInfoData buf;

	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		snprintf(label, sizeof(label), "DPU%u", ds_entry->endpoint_id);
		initStringInfo(&buf);
		appendStringInfo(&buf, "dir='%s', host='%s', port='%s'",
						 ds_entry->endpoint_dir,
						 ds_entry->config_host,
						 ds_entry->config_port);
		ExplainPropertyText(label, buf.data, es);
		pfree(buf.data);
	}
	else
	{
		snprintf(label, sizeof(label), "DPU%u-dir", ds_entry->endpoint_id);
		ExplainPropertyText(label, ds_entry->endpoint_dir, es);

		snprintf(label, sizeof(label), "DPU%u-host", ds_entry->endpoint_id);
		ExplainPropertyText(label, ds_entry->config_host, es);

		snprintf(label, sizeof(label), "DPU%u-port", ds_entry->endpoint_id);
		ExplainPropertyText(label, ds_entry->config_port, es);
	}
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
	dpu_storage_master_array = malloc(offsetof(DpuStorageArray,
											   entries[nrooms]));
	if (!dpu_storage_master_array)
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
			dpu_storage_master_array = realloc(dpu_storage_master_array,
											   offsetof(DpuStorageArray,
														entries[nrooms]));
			if (!dpu_storage_master_array)
				elog(ERROR, "out of memory");
		}
		curr = &dpu_storage_master_array->entries[nitems++];
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
	dpu_storage_master_array->nitems   = nitems;

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
									dpu_storage_master_array->nitems));
}

/*
 * pgstrom_startup_dpu_device
 */
static void
pgstrom_startup_dpu_device(void)
{
	pg_atomic_uint32 *validated;
	uint32_t	nitems = dpu_storage_master_array->nitems;
	bool		found;

	if (shmem_startup_next)
		shmem_startup_next();
	Assert(nitems > 0);
	validated = ShmemInitStruct("DPU-Tablespace Info",
								sizeof(pg_atomic_uint32) * nitems,
								&found);
	for (int i=0; i < dpu_storage_master_array->nitems; i++)
	{
		dpu_storage_master_array->entries[i].validated = &validated[i];
	}
}

/*
 * pgstrom_dpu_operator_ratio
 */
double
pgstrom_dpu_operator_ratio(void)
{
	if (cpu_operator_cost > 0.0)
	{
		return pgstrom_dpu_operator_cost / cpu_operator_cost;
	}
	return (pgstrom_dpu_operator_cost == 1.0 ? 0.0 : disable_cost);
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
		CacheRegisterRelcacheCallback(dpu_relcache_htable_invalidator, 0);

		/* output logs */
		for (int i=0; i < dpu_storage_master_array->nitems; i++)
		{
			DpuStorageEntry *ds_entry = &dpu_storage_master_array->entries[i];

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
