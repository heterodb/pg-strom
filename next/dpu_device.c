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
	const char *tablespace_name;
	const char *config_host;
	const char *config_port;
	int			endpoint_domain;	/* AF_UNIX/AF_INET/AF_INET6 */
	const struct sockaddr *endpoint_addr;
	socklen_t	endpoint_addr_len;
	int			validated;
};

struct DpuTablespaceInfo
{
	uint32_t	extra_sz;
	uint32_t	nitems;
	DpuStorageEntry entries[1];
};
typedef struct DpuTablespaceInfo	DpuTablespaceInfo;

struct DpuTablespaceHash
{
	struct DpuTablespaceHash *next;
	Oid			tablespace_oid;
	DpuStorageEntry *entry;
};
typedef struct DpuTablespaceHash	DpuTablespaceHash;

static DpuTablespaceInfo	   *dpu_tablespace_master = NULL;	/* shared memory */
static shmem_request_hook_type	shmem_request_next = NULL;
static shmem_startup_hook_type	shmem_startup_next = NULL;
#define DPU_TABLESPACE_HASH_NSLOTS		640
static DpuTablespaceHash	   *dpu_tablespace_hash_slots[DPU_TABLESPACE_HASH_NSLOTS];
static MemoryContext			dpu_tablespace_hash_memcxt = NULL;

/*
 * GetOptimalDpuForRelation
 */
DpuStorageEntry *
GetOptimalDpuForTablespace(Oid tablespace_oid)
{
	DpuTablespaceHash *hitem;
	char	   *tablespace_name;
	uint32_t	i, hindex;

	if (!dpu_tablespace_master)
		return NULL;	/* quick bailout */
	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;
	hindex = hash_uint32((uint32)tablespace_oid) % DPU_TABLESPACE_HASH_NSLOTS;

	for (hitem = dpu_tablespace_hash_slots[hindex];
		 hitem != NULL;
		 hitem = hitem->next)
	{
		if (hitem->tablespace_oid == tablespace_oid)
			return hitem->entry;
	}
	/* not found, so create a new entry */
	tablespace_name = get_tablespace_name(tablespace_oid);
	hitem = MemoryContextAllocZero(dpu_tablespace_hash_memcxt,
								   sizeof(DpuTablespaceHash));
	hitem->tablespace_oid = tablespace_oid;
	hitem->next = dpu_tablespace_hash_slots[hindex];
	dpu_tablespace_hash_slots[hindex] = hitem;
	for (i=0; i < dpu_tablespace_master->nitems; i++)
	{
		DpuStorageEntry *curr = &dpu_tablespace_master->entries[i];

		if (strcmp(curr->tablespace_name, tablespace_name) == 0)
		{
			hitem->entry = curr;
			break;
		}
	}
	return hitem->entry;
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
 * DpuClientOpenSession 
 */
void
DpuClientOpenSession(pgstromTaskState *pts,
					 const XpuCommand *session)
{
	DpuStorageEntry *ds_entry = pts->ds_entry;
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
 * tablespace_optimal_dpu_htable_invalidate
 */
static void
tablespace_optimal_dpu_htable_invalidate(Datum arg, int cacheid, uint32 hashvalue)
{
    /* invalidate all the cached status */
	MemoryContextReset(dpu_tablespace_hash_memcxt);
	memset(dpu_tablespace_hash_slots, 0, sizeof(dpu_tablespace_hash_slots));
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
	size_t		extra_sz = 0;
	char		__default_port[32];
	
	if (!pgstrom_dpu_endpoint_list)
		return 0;
	dpu_tablespace_master = malloc(offsetof(DpuTablespaceInfo,
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

		host = strchr(tok, '=');
		if (!host)
			elog(ERROR, "pg_strom.dpu_endpoint_list - invalid token [%s]", name);
		*host++ = '\0';
		name = __trim(tok);
		host = __trim(host);
		extra_sz += MAXALIGN(strlen(name) + 1);

		if (nitems >= nrooms)
		{
			nrooms *= 2;
			dpu_tablespace_master = realloc(dpu_tablespace_master,
											offsetof(DpuTablespaceInfo,
													 entries[nrooms]));
		}
		curr = &dpu_tablespace_master->entries[nitems++];
		memset(curr, 0, sizeof(DpuStorageEntry));

		curr->endpoint_id = endpoint_id++;
		curr->tablespace_name = strdup(name);
		if (!curr->tablespace_name)
			elog(ERROR, "out of memory");

		if (strncmp(host, "unix://", 7) == 0)
		{
			struct sockaddr_un *addr = calloc(1, sizeof(struct sockaddr_un));

			addr->sun_family = AF_UNIX;
			strncpy(addr->sun_path, host+7, 107);

			curr->config_host = strdup(host);
			if (!curr->config_host)
				elog(ERROR, "out of memory");
			curr->config_port = NULL;
			extra_sz += MAXALIGN(strlen(host) + 1);
			curr->endpoint_domain = AF_UNIX;
			curr->endpoint_addr = (struct sockaddr *)addr;
			curr->endpoint_addr_len = sizeof(struct sockaddr_un);
			curr->validated = -1;
		}
		else
		{
			struct addrinfo hints;
			struct addrinfo *addr;

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
			extra_sz += MAXALIGN(strlen(host) + 1) + MAXALIGN(strlen(port) + 1);
			curr->endpoint_domain = addr->ai_family;
			curr->endpoint_addr = addr->ai_addr;
			curr->endpoint_addr_len = addr->ai_addrlen;
			curr->validated = -1;
		}
		extra_sz += MAXALIGN(curr->endpoint_addr_len);
	}
	extra_sz += MAXALIGN(offsetof(DpuTablespaceInfo, entries[nitems]));
	dpu_tablespace_master->extra_sz = extra_sz;
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
	RequestAddinShmemSpace(dpu_tablespace_master->extra_sz);
}

/*
 * pgstrom_startup_dpu_device
 */
static void
pgstrom_startup_dpu_device(void)
{
	DpuTablespaceInfo *new_tablespace_master;
	bool		found;
	char	   *extra;
	uint32_t	i, nitems = dpu_tablespace_master->nitems;

	if (shmem_startup_next)
		shmem_startup_next();
	
	Assert(nitems > 0);
	new_tablespace_master = ShmemInitStruct("DPU-Tablespace Info",
											dpu_tablespace_master->extra_sz,
											&found);
	extra = (char *)new_tablespace_master
		+ MAXALIGN(offsetof(DpuTablespaceInfo, entries[nitems]));
	for (i=0; i < dpu_tablespace_master->nitems; i++)
	{
		DpuStorageEntry *old_item = &dpu_tablespace_master->entries[i];
		DpuStorageEntry *new_item = &new_tablespace_master->entries[i];

		new_item->endpoint_id = old_item->endpoint_id;
		new_item->tablespace_name = extra;
		strcpy(extra, old_item->tablespace_name);
		extra += MAXALIGN(strlen(extra) + 1);
		if (!old_item->config_host)
			new_item->config_host = NULL;
		else
		{
			new_item->config_host = extra;
			strcpy(extra, old_item->config_host);
			extra += MAXALIGN(strlen(extra) + 1);
		}
		if (!old_item->config_port)
			new_item->config_port = NULL;
		else
		{
			new_item->config_port = extra;
			strcpy(extra, old_item->config_port);
			extra += MAXALIGN(strlen(extra) + 1);
		}
		new_item->endpoint_domain = old_item->endpoint_domain;
		new_item->endpoint_addr = (struct sockaddr *)extra;
		memcpy(extra, old_item->endpoint_addr, old_item->endpoint_addr_len);
		extra += MAXALIGN(old_item->endpoint_addr_len);
		new_item->endpoint_addr_len = old_item->endpoint_addr_len;
		new_item->validated = -1;	/* validation on demand */
	}
	new_tablespace_master->extra_sz = dpu_tablespace_master->extra_sz;
	new_tablespace_master->nitems   = dpu_tablespace_master->nitems;
	Assert(extra - (char *)new_tablespace_master == dpu_tablespace_master->extra_sz);
	/* switch to shared memory structure */
	dpu_tablespace_master = new_tablespace_master;
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
	 * <tablespace-name>=<host/ipaddr>[;<port>][, ...]
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

	dpu_tablespace_hash_memcxt = AllocSetContextCreate(CacheMemoryContext,
													   "DPU-Tablespace Hash",
													   ALLOCSET_DEFAULT_SIZES);
	memset(dpu_tablespace_hash_slots, 0, sizeof(dpu_tablespace_hash_slots));

	if (parse_dpu_endpoint_list())
	{
		pgstrom_init_dpu_options();

		shmem_request_next = shmem_request_hook;
		shmem_request_hook = pgstrom_request_dpu_device;
		shmem_startup_next = shmem_startup_hook;
		shmem_startup_hook = pgstrom_startup_dpu_device;

		CacheRegisterSyscacheCallback(TABLESPACEOID,
									  tablespace_optimal_dpu_htable_invalidate,
									  (Datum)0);
		return true;
	}
	return false;
}
