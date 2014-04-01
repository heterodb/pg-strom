32;100;2c/*
 * opencl_devprog.c
 *
 * Routines to manage device programs/kernels
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "utils/guc.h"
#include "utils/pg_crc.h"
#include <limits.h>
#include "pg_strom.h"

static shmem_startup_hook_type shmem_startup_hook_next;
static int reclaim_threshold;

#define DEVPROG_HASH_SIZE	2048

static struct {
	shmem_context *shm_context;	/* memory context for device programs */
	slock_t		lock;
	int			refcnt;
	Size		usage;
	dlist_head	lru_list;
	dlist_head	slot[DEVPROG_HASH_SIZE];
} *opencl_devprog_shm_values;

typedef struct {
	dlist_node	hash_chain;
	dlist_node	lru_chain;
	/*
	 * NOTE: above members are protected by opencl_devprog_shm_values->lock.
	 */
	slock_t		lock;		/* protection of the fields below */
	int			refcnt;		/* reference counter of this device program */
	dlist_head	waitq;		/* wait queue of program build */
	cl_program	program;	/* valid only OpenCL intermediator */
	const char *errmsg;		/* error message if build error */

	/* The fields below are read-only once constructed */
	pg_crc32	crc;
	int32		extra_libs;	/* set of DEVFUNC_NEEDS_* */
	Size		source_len;
	char		source[FLEXIBLE_ARRAY_MEMBER];
} devprog_entry;

/*
 * clserv_lookup_device_program
 *
 * It assumes OpenCL intermediation server calls this routine.
 * According to the given program key, it returns a cl_program object if it
 * is already built and ready. The source code kept in this entry was not
 * built yet, it kicks the built-in compiler asynchronously. In this case,
 * the supplied MessageTag is linked to the waitq until this asynchronous
 * built getting finished, then enqueued 
 *
 *
 */
cl_program
clserv_lookup_device_program(Datum dprog_key, MessageTag *mtag)
{



	return NULL;
}

/*
 * pgstrom_reclaim_devprog
 *
 * It reclaims device program entries being no longer used according to LRU
 * algorism, but never tries to keep the total usage less than
 * reclaim_threshold, just release the least-recently-used one if total
 * usage is larger than the threshold.
 *
 * NOTE: This routine is assumed that opencl_devprog_shm_values->lock is
 * already held on the caller side, and OpenCL intermediation server calls.
 */
static void
pgstrom_reclaim_devprog(void)
{
	dlist_iter	iter;

	/*
	 * this logic may involves clReleaseProgram(), so only OpenCL
	 * intermediation server can handle reclaiming
	 */
	Assert(pgstrom_is_opencl_server(void));

	if (opencl_devprog_shm_values->usage < reclaim_threshold)
		return;

	dlist_reverse_foreach(iter, &opencl_devprog_shm_values->lru_list)
	{
		devprog_entry  *dprog
			= dlist_container(devprog_entry, lru_chain, iter.cur);

		if (dprog->refcnt > 0)
			continue;

		Assert(dprog->refcnt == 0);
		dlist_delete(&dprog->hash_chain);
		dlist_delete(&dprog->lru_chain);

		length = offsetof(devprog_entry, source[dprog->source_len]);
		if (dprog->errmsg)
		{
			length += strlen(dprog->errmsg);
			pgstrom_shmem_free(dprog->errmsg);
		}
		clReleaseProgram(dprog->program);
		opencl_devprog_shm_values->usage -= length;
		break;
	}
}

/*
 * pgstrom_get_devprog_key
 *
 * It returns a devprog-key that holds the given source and extra_libs.
 * If not found on the device program table, it also create a new one
 * and insert it, then returns its key.
 *
 * TODO: add a resource tracking here to ensure device program is released.
 */
Datum
pgstrom_get_devprog_key(const char *source, int32 extra_libs)
{
	devprog_entry *dprog = NULL;
    shmem_context *context;
	Size		source_len = strlen(source);
	pg_crc32	crc;
	ListCell   *cell;

	/* calculate a hash value */
	INIT_CRC32(crc);
	COMP_CRC32(crc, &extra_libs, sizeof(int32));
	COMP_CRC32(crc, kernel_source, kernel_length);
	FIN_CRC32(crc);

retry:
	index = crc % DEVPROG_HASH_SIZE;
	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	dlist_foreach (iter, &opencl_devprog_shm_values->slot[index])
	{
		devprog_entry *entry
			= dlist_container(devprog_entry, chain, iter.cur);

		if (entry->crc == crc &&
			entry->extra_libs == extra_libs &&
			entry->source_len == source_len &&
			strcmp(entry->source, source) == 0)
		{
			dlist_move_head(opencl_devprog_shm_values->lru_list,
							&entry->lru_chain);
			SpinLockAcquire(&entry->lock);
			entry->refcnt++;
			SpinLockRelease(&entry->lock);
			SpinLockRelease(&opencl_devprog_shm_values->lock);
			if (dprog)
				pgstrom_shmem_free(dprog);
			return PointerGetDatum(entry);
		}
	}
	/* !Not found on the existing cache! */

	/*
	 * If it is second trial, we could ensure no identical kernel source
	 * was inserted concurrently.
	 */
	if (dprog)
	{
		dlist_push_tail(&opencl_devprog_shm_values->slot[index],
						&dprog->hash_chain);
		dlist_push_head(&opencl_devprog_shm_values->lru_list,
						&dprog->lru_chain);
		SpinLockRelease(&opencl_devprog_shm_values->lock);

		return PointerGetDatum(dprog);
	}
	SpinLockRelease(&opencl_devprog_shm_values->lock);

	/* OK, create a new device program entry */
	context = opencl_devprog_shm_values->shm_context;

	length = offsetof(devprog_entry, source[source_len]);
	dprog = pgstrom_shmem_alloc(context, length);
	if (!dprog)
		elog(ERROR, "out of shared memory");

	dprog->refcnt = 1;
	SpinLockInit(&dprog->lock);
	dprog->program = NULL;
    dprog->errmsg = NULL;
	dprog->crc = crc;
	dprog->extra_libs = extra_libs;
	dprog->source_len = source_len;
	strcpy(dprog->source, source);
	opencl_devprog_shm_values->usage += length;

	goto retry;
}

/*
 * pgstrom_put_devprog_key
 *
 * It decrements reference counter of the given device program.
 * Note that refcnt==0 does not mean immediate object release, for further
 * reusing. If it actually overuses shared memory segment, OpenCL server
 * will reclaim it.
 */
void
pgstrom_put_devprog_key(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);

	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	dprog->refcnt--;
	Assert(dprog->refcnt >= 0);
	SpinLockRelease(&opencl_devprog_shm_values->lock);
}

/*
 * pgstrom_retain_devprog_key
 *
 * It increments reference counter of the given device program, to avoid
 * unexpected program destruction.
 */
void
pgstrom_retain_devprog_key(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);

	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	Assert(dprog->refcnt >= 0);
	dprog->refcnt++;
	SpinLockRelease(&opencl_devprog_shm_values->lock);
}

/*
 * pgstrom_setup_opencl_devprog
 *
 * callback post shared memory context getting ready
 */
void
pgstrom_setup_opencl_devprog(void)
{
	shmem_context  *context;

	context = pgstrom_shmem_context_create("PG-Strom device programs");
	if (!context)
		elog(ERROR, "failed to create shared memory context");
	opencl_devprog_shm_values->shm_context = context;
}

/*
 * pgstrom_startup_opencl_devprog
 *
 * callback for shared memory allocation
 */
static void
pgstrom_startup_opencl_devprog(void)
{
	bool	found;
	int		i;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	opencl_devprog_shm_values
		= ShmemInitStruct("opencl_devprog_shm_values",
						  MAXALIGN(sizeof(*opencl_devprog_shm_values)),
						  &found);
	Assert(!found);

	opencl_devprog_shm_values->shm_context = NULL;	/* to be set later */
	SpinLockInit(&opencl_devprog_shm_values->lock);
	opencl_devprog_shm_values->usage = 0;
	dlist_init(&opencl_devprog_shm_values->lru_list);
	for (i=0; i < DEVPROG_HASH_SIZE; i++)
		dlist_init(&opencl_devprog_shm_values->slot[i]);
}

/*
 * pgstrom_init_opencl_devprog
 *
 * entrypoint of this module
 */
void
pgstrom_init_opencl_devprog(void)
{
	/* threshold to reclaim the cached opencl programs */
	DefineCustomIntVariable("pgstrom.devprog_reclaim_threshold",
							"threahold to reclaim device program objects",
							NULL,
							&reclaim_threshold,
							16 << 20,	/* 16MB */
							0,	/* 0 means no threshold, not recommended */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* aquires shared memory region */
	RequestAddinShmemSpace(MAXALIGN(sizeof(*opencl_devprog_shm_values)));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_opencl_devprog;
}
