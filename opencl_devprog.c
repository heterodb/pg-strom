/*
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

#define DEVPROG_STATUS_NOT_BUILT			1
#define DEVPROG_STATUS_BUILD_IN_PROGRESS	2
#define DEVPROG_STATUS_BUILT_READY			3
#define DEVPROG_STATUS_BUILT_ERROR			4

typedef struct {
	dlist_node	chain;	/* NOTE: chain and refcnt is protected by */
	int			refcnt;	/* opencl_devprog_shm_values->lock */
	slock_t		lock;
	pg_crc32	crc;
	int			status;		/* one of the DEVPROG_STATUS_* */
	cl_program	program;	/* valid only OpenCL background server */
	char	   *errmsg;		/* error message if build error */
	int32		extra_libs;	/* set of DEVFUNC_NEEDS_* */
	Size		source_len;
	char	   *source;
	int			num_param_attrs;
	int			num_value_attrs;
	simple_pg_attribute	*param_attrs;
	simple_pg_attribute	*value_attrs;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} devprog_entry;

cl_program
pgstrom_lookup_opencl_devprog(Datum dprog_key)
{



	return NULL;
}

/*
 * pgstrom_create_opencl_devprog
 *
 * It creates a device program entry on the shared memory region, if identical
 * one is not here. Elsewhere, it just increments reference counter of the
 * identical entry.
 * It is intended to be called by backend processes, not OpenCL server.
 */
Datum
pgstrom_create_opencl_devprog(const char *source, int32 extra_libs)
{
	devprog_entry *dprog;
	shmem_context *context;
	Size		source_len = strlen(source);
	pg_crc32	crc;
	int			index;
	Size		length;
	dlist_iter	iter;

	INIT_CRC32(crc);
	COMP_CRC32(crc, &extra_libs, sizeof(int32));
	COMP_CRC32(crc, source, source_len);
	FIN_CRC32(crc);

	index = crc % DEVPROG_HASH_SIZE;
	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	dlist_foreach (iter, &opencl_devprog_shm_values->slot[index])
	{
		dprog = dlist_container(devprog_entry, chain, iter.cur);

		if (dprog->crc == crc &&
			dprog->extra_libs == extra_libs &&
			strcmp(dprog->source, source) == 0)
		{
			dprog->refcnt++;
			SpinLockRelease(&opencl_devprog_shm_values->lock);
			return PointerGetDatum(dprog);
		}
	}
	SpinLockRelease(&opencl_devprog_shm_values->lock);

	/*
	 * Not found! so, create a new one
	 */
	context = opencl_devprog_shm_values->shm_context;
	/*
	 * XXX FIXME - need to allocate area for kernel source, param attrs,
	 * var attrs
	 */
	length = sizeof(devprog_entry);

	dprog = pgstrom_shmem_alloc(context, length);
	if (!dprog)
		elog(ERROR, "out of shared memory");

	dprog->refcnt = 2;
	SpinLockInit(&dprog->lock);
	dprog->crc = crc;
	dprog->status = DEVPROG_STATUS_NOT_BUILT;
	dprog->program = NULL;
	dprog->errmsg = NULL;
	dprog->extra_libs = extra_libs;
	dprog->source_len = source_len;
	strcpy(dprog->source, source);

	/* ensure concurrent job does not add same device program */
	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	dlist_foreach (iter, &opencl_devprog_shm_values->slot[index])
	{
		devprog_entry *temp = dlist_container(devprog_entry, chain, iter.cur);

		if (temp->crc == dprog->crc &&
			temp->extra_libs == dprog->extra_libs &&
			strcmp(temp->source, dprog->source) == 0)
		{
			pgstrom_shmem_free(dprog);
			temp->refcnt++;
			SpinLockRelease(&opencl_devprog_shm_values->lock);
			return PointerGetDatum(temp);
		}
	}
	if (opencl_devprog_shm_values->usage + length >= reclaim_threshold)
		/* do cache reclaiming here */ ;

	dlist_push_tail(&opencl_devprog_shm_values->slot[index], &dprog->chain);
	opencl_devprog_shm_values->usage += length;

	SpinLockRelease(&opencl_devprog_shm_values->lock);
	return PointerGetDatum(dprog);
}

void
pgstrom_get_opencl_devprog(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);

	SpinLockAcquire(&dprog->lock);
	Assert(dprog->refcnt > 0);
	dprog->refcnt++;
	SpinLockRelease(&dprog->lock);
}

void
pgstrom_put_opencl_devprog(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);
	cl_int		rc;

	SpinLockAcquire(&dprog->lock);
	Assert(dprog->refcnt > 0);
	if (--dprog->refcnt == 0)
	{

		SpinLockAcquire(&opencl_devprog_shm_values->lock);
		dlist_delete(&dprog->chain);
		SpinLockRelease(&opencl_devprog_shm_values->lock);

		rc = clReleaseProgram(dprog->program);
		Assert(rc == CL_SUCCESS);

		if (dprog->errmsg)
			pgstrom_shmem_free(dprog->errmsg);
		pgstrom_shmem_free(dprog);
	}
	SpinLockRelease(&dprog->lock);
}

void
pgstrom_setup_opencl_devprog(void)
{
	shmem_context  *context;

	context = pgstrom_shmem_context_create("PG-Strom device programs");
	if (!context)
		elog(ERROR, "failed to create shared memory context");
	opencl_devprog_shm_values->shm_context = context;
}

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
