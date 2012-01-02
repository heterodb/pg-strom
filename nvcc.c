/*
 * nvcc.c
 *
 * routines to build the supplied GPU code and cache it on the shared
 * memory segment.
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "access/hash.h"
#include "utils/guc.h"
#include "miscadmin.h"
#include "storage/fd.h"
#include "storage/ipc.h"
#include "pg_strom.h"
#include <sys/param.h>

typedef struct {
	Datum		hash;	/* value of hash */
	uint32		next;	/* offset to the next entry */
	uint32		chain;	/* offset to the next block of this entry */
	bool		hot_cache;
	char		data[512];
} PgStromNvccCache;

typedef struct {
	uint32		length_src;
	uint32		length_bin;
	uint32		offset_bin;
	char		data[0];
} PgStromNvccEntry;

typedef struct {
	LWLockId	lock;		/* lock */
	uint32		hash_sz;	/* size of hash slots */
	uint32		i_reclaim;	/* index to be reclaimed */
	uint32		num_frees;	/* num of free blocks */
	uint32		free_list;	/* offset to free blocks */
	uint32		slots[0];	/* offset to cached blocks */
} PgStromNvccHash;
static PgStromNvccHash *pgstrom_nvcc_hash;

static char	   *pgstrom_nvcc_command;
static int		pgstrom_nvcc_cache_size;
static shmem_startup_hook_type	shmem_startup_hook_next;

#define offset_to_addr(offset)	\
	((offset) == 0 ? NULL		\
	 : (void *)(((uintptr_t)pgstrom_nvcc_hash) + (offset)))
#define addr_to_offset(addr)	\
	((addr) == NULL ? 0			\
	 : ((uintptr_t)(addr) - (uintptr_t)pgstrom_nvcc_hash))
#define offset_align(offset)	\
	(((offset) + sizeof(void *) + 1) & ~(sizeof(void *) - 1))


static void
pgstrom_nvcc_reclaim_cache(int num_reclaim)
{
	while (num_reclaim > 0)
	{
		PgStromNvccCache   *cache;
		PgStromNvccCache   *prev = NULL;
		PgStromNvccCache   *next;
		int		index = pgstrom_nvcc_hash->i_reclaim++;

		for (cache = offset_to_addr(pgstrom_nvcc_hash->slots[index]);
			 cache != NULL;
			 cache = next)
		{
			PgStromNvccCache   *temp1;
			PgStromNvccCache   *temp2;

			next = offset_to_addr(cache->next);

			if (cache->hot_cache-- > 0)
			{
				prev = cache;
				continue;
			}
			if (!prev)
				pgstrom_nvcc_hash->slots[index] = cache->next;
			else
				prev->next = cache->next;

			for (temp1 = cache; temp1 != NULL; temp1 = temp2)
			{
				temp2 = offset_to_addr(temp1->chain);

				temp1->next = pgstrom_nvcc_hash->free_list;
				pgstrom_nvcc_hash->free_list = addr_to_offset(temp1);
				pgstrom_nvcc_hash->num_frees++;

				num_reclaim--;
			}
		}
	}
}

static void
pgstrom_nvcc_insert_cache(const char *kernel_src, uint32 length_src,
						  const char *kernel_bin, uint32 length_bin)
{
	PgStromNvccCache   *prev = NULL;
	PgStromNvccCache   *cache;
	PgStromNvccEntry   *entry;
	uint32		total_length;
	uint32		offset;
	int			num_blocks;

	total_length = sizeof(PgStromNvccEntry) + length_src + length_bin + 2;
	entry = alloca(total_length);

	entry->length_src = length_src;
	entry->length_bin = length_bin;
	entry->offset_bin = length_src + 1;
	memcpy(entry->data, kernel_src, length_src + 1);
	memcpy(entry->data + entry->offset_bin, kernel_bin, length_bin + 1);

	LWLockAcquire(pgstrom_nvcc_hash->lock, LW_EXCLUSIVE);

	num_blocks = (total_length+sizeof(cache->data)-1) / sizeof(cache->data);
	if (pgstrom_nvcc_hash->num_frees < num_blocks)
		pgstrom_nvcc_reclaim_cache(2 * num_blocks);

	for (offset = 0; offset < total_length; offset += sizeof(cache->data))
	{
		cache = offset_to_addr(pgstrom_nvcc_hash->free_list);
		pgstrom_nvcc_hash->free_list = cache->next;

		memcpy(cache->data, ((char *)entry) + offset,
			   MIN(total_length - offset, sizeof(cache->data)));

		if (!prev)
		{
			Datum	hash = hash_any((unsigned char *)kernel_src, length_src);
			int		index = hash % pgstrom_nvcc_hash->hash_sz;

			cache->hash = hash;
			cache->next = pgstrom_nvcc_hash->slots[index];
			cache->chain = 0;
			cache->hot_cache = 2;	/* very hot */
			pgstrom_nvcc_hash->slots[index] = addr_to_offset(cache);
		}
		else
		{
			cache->hash = prev->hash;
			cache->next = 0;
			cache->chain = 0;
			cache->hot_cache = 2;	/* very hot */
			prev->chain = addr_to_offset(cache);
		}
		pgstrom_nvcc_hash->num_frees--;
		prev = cache;
	}
	LWLockRelease(pgstrom_nvcc_hash->lock);
}

static char *
pgstrom_nvcc_lookup_cache(const char *kernel_source)
{
	PgStromNvccCache   *cache1;
	PgStromNvccCache   *cache2;
	PgStromNvccEntry   *entry;
	StringInfoData		buf;
	uint32	length_src;
	uint32	index;
	Datum	hash;
	char   *result = NULL;

	length_src = strlen(kernel_source);
	hash = hash_any((unsigned char *)kernel_source, length_src);
	index = hash % pgstrom_nvcc_hash->hash_sz;

	initStringInfo(&buf);

	LWLockAcquire(pgstrom_nvcc_hash->lock, LW_SHARED);

	for (cache1 = offset_to_addr(pgstrom_nvcc_hash->slots[index]);
		 cache1 != NULL;
		 cache1 = offset_to_addr(cache1->next))
	{
		if (cache1->hash != hash)
			continue;

		resetStringInfo(&buf);
		for (cache2 = cache1;
			 cache2 != NULL;
			 cache2 = offset_to_addr(cache2->chain))
			appendBinaryStringInfo(&buf, cache2->data, sizeof(cache2->data));

		entry = (PgStromNvccEntry *)buf.data;

		if (entry->length_src == length_src &&
			memcmp(entry->data, kernel_source, length_src) == 0)
		{
			result = palloc(entry->length_bin);
			memcpy(result,
				   &entry->data[entry->offset_bin],
				   entry->length_bin);
			cache1->hot_cache = true;
			break;
		}
	}
	LWLockRelease(pgstrom_nvcc_hash->lock);
	pfree(buf.data);

	return result;
}

#define TEMPFILE_FMT	"/tmp/.pgstrom-%u-qual%s"
void *
pgstrom_nvcc_kernel_build(const char *kernel_source)
{
	StringInfoData	str;
	FILE	   *filp;
	int			code;
	size_t		nbytes;
	char		buffer[1024];
	char	   *result;

	/*
	 * Lookup pre-compiled binary cache to reduce duplicate compile
	 */
	result = pgstrom_nvcc_lookup_cache(kernel_source);
	if (result)
		return result;

	/*
	 * If not cached, we launch nvcc command to compile the supplied
	 * kernel source, then put it on the cache.
	 */
	initStringInfo(&str);

	/*
	 * Write source to temporary file
	 */
	appendStringInfo(&str, TEMPFILE_FMT, MyProcPid, ".gpu");
	filp = AllocateFile(str.data, PG_BINARY_W);
	if (!filp)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open temporary file \"%s\" : %m",
						str.data)));
	if (fputs(kernel_source, filp) < 0)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not write out source code : %m")));
	FreeFile(filp);

	/*
	 * Execute nvcc compiler
	 */
	resetStringInfo(&str);
	appendStringInfo(&str,
					 "'%s' -fatbin '" TEMPFILE_FMT "' "
					 "-o '" TEMPFILE_FMT "' 2>'" TEMPFILE_FMT "'",
					 pgstrom_nvcc_command,
					 MyProcPid, ".gpu",
					 MyProcPid, ".fatbin",
					 MyProcPid, ".log");
	if (system(str.data) != 0)
	{
		resetStringInfo(&str);
		appendStringInfo(&str, TEMPFILE_FMT, MyProcPid, ".log");
		filp = AllocateFile(str.data, PG_BINARY_R);
		if (filp != NULL)
		{
			resetStringInfo(&str);
			while ((code = fgetc(filp)) != EOF)
			{
				if (code == '\n')
				{
					elog(LOG, "%s", str.data);
					resetStringInfo(&str);
				}
				else
					appendStringInfoChar(&str, code);
			}
			FreeFile(filp);
		}
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("could not compile the supplied source")));
	}

	/*
	 * Read the built module
	 */
	resetStringInfo(&str);
	appendStringInfo(&str, TEMPFILE_FMT, MyProcPid, ".fatbin");

	filp = AllocateFile(str.data, PG_BINARY_R);
	if (!filp)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open temporary file \"%s\" : %m",
						str.data)));
	resetStringInfo(&str);

	while ((nbytes = fread(buffer, 1, sizeof(buffer), filp)) > 0)
		appendBinaryStringInfo(&str, buffer, nbytes);

	/*
	 * Put the pair of source and binary on the cache
	 */
	pgstrom_nvcc_insert_cache(kernel_source, strlen(kernel_source),
							  str.data, str.len);
	return (void *)str.data;
}

static void
pgstrom_nvcc_shmem_init(void)
{
	PgStromNvccCache   *cache;
	uint32		limit = (pgstrom_nvcc_cache_size << 20);
	uint32		offset;
	bool		found;

	pgstrom_nvcc_hash
		= ShmemInitStruct("JIT Cache of PgStrom", limit, &found);
	if (!IsUnderPostmaster)
	{
		Assert(!found);

		pgstrom_nvcc_hash->lock = LWLockAssign();
		pgstrom_nvcc_hash->hash_sz = pgstrom_nvcc_cache_size * 64;
		pgstrom_nvcc_hash->free_list = 0;
		memset(pgstrom_nvcc_hash->slots, 0,
			   sizeof(uint32) * pgstrom_nvcc_hash->hash_sz);

		offset = offset_align(sizeof(PgStromNvccHash) +
							  sizeof(uint32) * pgstrom_nvcc_hash->hash_sz);

		while (offset + sizeof(PgStromNvccCache) < limit)
		{
			cache = (PgStromNvccCache *)((char *)pgstrom_nvcc_hash + offset);
			memset(cache, 0, sizeof(PgStromNvccCache));

			cache->next = pgstrom_nvcc_hash->free_list;
			pgstrom_nvcc_hash->free_list = addr_to_offset(cache);
			pgstrom_nvcc_hash->num_frees++;

			offset = offset_align(offset + sizeof(PgStromNvccCache));
		}
	}
	else
		Assert(found);

	if (shmem_startup_hook_next)
		shmem_startup_hook_next();
}

void
pgstrom_nvcc_init(void)
{
	DefineCustomIntVariable("pg_strom.nvcc_cache_size",
							"size of shmem to cache compiled queries",
							NULL,
							&pgstrom_nvcc_cache_size,
							32,
							2,
							2048,
							PGC_SIGHUP,
							0,
							NULL, NULL, NULL);
	DefineCustomStringVariable("pg_strom.nvcc_command",
							   "full path of the nvcc command",
							   NULL,
							   &pgstrom_nvcc_command,
							   NVCC_CMD_DEFAULT,
							   PGC_SIGHUP,
							   0,
							   NULL, NULL, NULL);

	/* acquire shared memory segment for query cache */
	RequestAddinShmemSpace(pgstrom_nvcc_cache_size << 20);
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_nvcc_shmem_init;
}

#if 0
static void
pgstrom_build_kernel_source(PgStromExecState *sestate)
{
}
#endif
