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
#include "storage/barrier.h"
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
	slock_t		lock;
	int			refcnt;
	Size		usage;
	dlist_head	lru_list;
	dlist_head	slot[DEVPROG_HASH_SIZE];
} *opencl_devprog_shm_values;

typedef struct {
	StromTag	stag;		/* = StromTag_DevProgram */
	dlist_node	hash_chain;
	dlist_node	lru_chain;
	/*
	 * NOTE: above members are protected by opencl_devprog_shm_values->lock.
	 */
	slock_t		lock;		/* protection of the fields below */
	int			refcnt;		/* reference counter of this device program */
	dlist_head	waitq;		/* wait queue of program build */
	cl_program	program;	/* valid only OpenCL intermediator */
	bool		build_running;	/* true, if async build is running */
	char	   *errmsg;		/* error message if build error */

	/* The fields below are read-only once constructed */
	pg_crc32	crc;
	int32		extra_libs;	/* set of DEVFUNC_NEEDS_* */
	Size		source_len;
	char		source[FLEXIBLE_ARRAY_MEMBER];
} devprog_entry;

/*
 * pgstrom_reclaim_devprog
 *
 * It reclaims device program entries being no longer used according to LRU
 * algorism, but never tries to keep the total usage less than
 * reclaim_threshold, just release the least-recently-used one if total
 * usage is larger than the threshold.
 */
static void
pgstrom_reclaim_devprog(void)
{
	dlist_iter	iter;
	Size		length;

	/*
	 * this logic may involves clReleaseProgram(), so only OpenCL
	 * intermediation server can handle reclaiming
	 */
	Assert(pgstrom_i_am_clserv);

	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	/* concurrent task already reclaimed it? */
	if (opencl_devprog_shm_values->usage < reclaim_threshold)
	{
		SpinLockRelease(&opencl_devprog_shm_values->lock);
		return;
	}

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
	SpinLockRelease(&opencl_devprog_shm_values->lock);
}

/*
 * clserv_devprog_build_callback
 *
 * callback function for clBuildProgram; that enqueues the waiting messages
 * into server's message queue again, and set status of devprog_entry
 * correctly.
 */
static void
clserv_devprog_build_callback(cl_program program, void *cb_private)
{
	devprog_entry *dprog = (devprog_entry *) cb_private;
	cl_build_status	status;
	dlist_mutable_iter iter;
	char	   *errmsg = NULL;
	cl_int		i, rc;

	/* check program build status */
	for (i=0; i < opencl_num_devices; i++)
	{
		rc = clGetProgramBuildInfo(program,
								   opencl_devices[i],
								   CL_PROGRAM_BUILD_STATUS,
								   sizeof(cl_build_status),
								   &status,
								   NULL);
		if (rc != CL_SUCCESS)
		{
			elog(LOG, "clGetProgramBuildInfo failed: %s",
				 opencl_strerror(rc));
			goto out_error;
		}

		/*
		 * An expected result is either CL_BUILD_SUCCESS or
		 * CL_BUILD_ERROR
		 */
		if (status == CL_BUILD_ERROR)
		{
			char	buffer[128 * 1024];	/* 128KB */
			size_t	buflen;

			rc = clGetProgramBuildInfo(program,
									   opencl_devices[i],
									   CL_PROGRAM_BUILD_LOG,
									   sizeof(buffer),
									   buffer,
									   &buflen);
			if (rc != CL_SUCCESS)
			{
				snprintf(buffer, sizeof(buffer),
						 "clGetProgramBuildInfo failed: %s",
						 opencl_strerror(rc));
				buflen = strlen(buffer);
			}
			errmsg = pgstrom_shmem_alloc(buflen + 1);
			if (errmsg)
			{
				opencl_devprog_shm_values->usage += buflen;
				strcpy(errmsg, buffer);
			}
			goto out_error;
		}
		else if (status != CL_BUILD_SUCCESS)
		{
			elog(LOG, "unexpected build status (%d) on device %d",
				 (int) status, i);
			goto out_error;
		}
	}
	/*
	 * OK, source build was successfully done for all the devices
	 */
	SpinLockAcquire(&dprog->lock);
	Assert(dprog->program == program);
	dlist_foreach_modify(iter, &dprog->waitq)
	{
		pgstrom_message	*msg
			= dlist_container(pgstrom_message, chain, iter.cur);

		dlist_delete(&msg->chain);
		pgstrom_enqueue_message(msg);
	}
	dprog->build_running = false;
	SpinLockRelease(&dprog->lock);
	return;


out_error:
	SpinLockAcquire(&dprog->lock);
	Assert(dprog->program == program);
	dprog->errmsg = errmsg;
	dlist_foreach_modify(iter, &dprog->waitq)
	{
		pgstrom_message *msg
			= dlist_container(pgstrom_message, chain, iter.cur);

		dlist_delete(&msg->chain);
        pgstrom_enqueue_message(msg);
    }
	dprog->build_running = false;
	rc = clReleaseProgram(program);
	Assert(rc == CL_SUCCESS);
	dprog->program = BAD_OPENCL_PROGRAM;
	SpinLockRelease(&dprog->lock);
}

/*
 * clserv_lookup_device_program
 *
 * It assumes OpenCL intermediation server calls this routine.
 * According to the given program key, it returns a cl_program object
 * if it is already built and ready to run.
 * Otherwise, we have three cases; 1. no cl_program object is not
 * constructed, 2. cl_program object is under compile & link, or
 * 3. cl_program object was built but has compile errors.
 * In case of (1), it construct a built-in compiler asynchronously,
 * and returns NULL; that means the program object is not ready now
 * but not an error. In case of (1) and (2), this routine links the
 * supplied message on the waiting list, then they shall be enqueued
 * into server mqueue again. It shall be handled by the server process
 * again.
 * In case of (3), it returns BAD_OPENCL_PROGRAM to inform caller the
 * supplied program has compile errors, or something broken.
 */
cl_program
clserv_lookup_device_program(Datum dprog_key, pgstrom_message *message)
{
	devprog_entry  *dprog = (devprog_entry *)DatumGetPointer(dprog_key);
	cl_int		rc;

	/*
	 * In case when shared memory usage of device program table exceeds
	 * threshold, we try to reclaim it. Any programs being required to
	 * run are already refcnt > 0, so no need to worry about unexpected
	 * destruction.
	 */
	pg_memory_barrier();
	if (opencl_devprog_shm_values->usage >= reclaim_threshold)
		pgstrom_reclaim_devprog();

	SpinLockAcquire(&dprog->lock);
	if (!dprog->program)
	{
		cl_program	program;
		const char *build_opts;
		const char *sources[32];
		size_t		lengths[32];
		cl_uint		count = 0;

		/* common opencl header */
		sources[count] = pgstrom_opencl_common_code;
		lengths[count] = strlen(pgstrom_opencl_common_code);
		count++;
#if 0
		/* opencl timelib */
		if (dprog->extra_libs & DEVFUNC_NEEDS_TIMELIB)
		{
			sources[count] = pgstrom_opencl_timelib_code;
			lengths[count] = strlen(pgstrom_opencl_timelib_code);
			count++;
		}
		/* opencl textlib */
		if (dprog->extra_libs & DEVFUNC_NEEDS_TEXTLIB)
		{
			sources[count] = pgstrom_opencl_textlib_code;
			lengths[count] = strlen(pgstrom_opencl_textlib_code);
			count++;
		}
		/* opencl numericlib */
		if (dprog->extra_libs & DEVFUNC_NEEDS_NUMERICLIB)
		{
			sources[count] = pgstrom_opencl_numericlib_code;
			lengths[count] = strlen(pgstrom_opencl_numericlib_code);
			count++;
		}
#endif
		/* gpuscan device implementation */
		if (dprog->extra_libs & DEVKERNEL_NEEDS_GPUSCAN)
		{
			sources[count] = pgstrom_opencl_gpuscan_code;
			lengths[count] = strlen(pgstrom_opencl_gpuscan_code);
			count++;
		}
#if 0
		/* gpusort device implementation */
		if (dprog->extra_libs & DEVKERNEL_NEEDS_GPUSORT)
		{
			sources[count] = pgstrom_opencl_gpusort_code;
			lengths[count] = strlen(pgstrom_opencl_gpusort_code);
			count++;
		}
		/* hashjoin device implementation */
		if (dprog->extra_libs & DEVKERNEL_NEEDS_HASHJOIN)
		{
			sources[count] = pgstrom_opencl_hashjoin_code;
			lengths[count] = strlen(pgstrom_opencl_hashjoin_code);
			count++;
		}
#endif
		/* source code of this program */
		sources[count] = dprog->source;
		lengths[count] = dprog->source_len;
		count++;

		/* OK, construct a program object */
		program = clCreateProgramWithSource(opencl_context,
											count,
											sources,
											lengths,
											&rc);
		if (rc != CL_SUCCESS)
		{
			elog(LOG, "clCreateProgramWithSource failed: %s",
				 opencl_strerror(rc));
			dprog->program = BAD_OPENCL_PROGRAM;
			goto out_unlock;
		}
		/* Next, launch an asynchronous program build process */
		build_opts = "-DOPENCL_DEVICE_CODE"
#if SIZEOF_VOID_P == 8
			" -DHOSTPTRLEN=8"
#else
			" -DHOSTPTRLEN=4"
#endif
#ifdef PGSTROM_DEBUG
			" -Werror"
#endif
			;
		rc = clBuildProgram(program,
							opencl_num_devices,
							opencl_devices,
							build_opts,
							clserv_devprog_build_callback,
							dprog);
		if (rc != CL_SUCCESS)
		{
			elog(LOG, "clBuildProgram failed: %s", opencl_strerror(rc));
			rc = clReleaseProgram(program);
			Assert(rc != CL_SUCCESS);
			dprog->program = BAD_OPENCL_PROGRAM;
			goto out_unlock;
		}
		dprog->program = program;
		dprog->build_running = true;
		if (message)
			dlist_push_tail(&dprog->waitq, &message->chain);
		SpinLockRelease(&dprog->lock);
		return NULL;
	}
	else if (dprog->program != BAD_OPENCL_PROGRAM)
	{
		cl_program	program;

		/*
		 * If valid program build process is still running, we chain the
		 * message object onto waiting queue of this device program.
		 */
		if (dprog->build_running)
		{
			if (message)
				dlist_push_tail(&dprog->waitq, &message->chain);
			SpinLockRelease(&dprog->lock);
			return NULL;
		}

		/*
		 * Elsewhere, everything is OK to run required device kernel
		 */
		rc = clRetainProgram(dprog->program);
		if (rc != CL_SUCCESS)
		{
			elog(LOG, "clRetainProgram failed: %s", opencl_strerror(rc));
			goto out_unlock;
		}
		program = dprog->program;
		SpinLockRelease(&dprog->lock);
		return program;
	}
out_unlock:
	SpinLockRelease(&dprog->lock);

	return BAD_OPENCL_PROGRAM;
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
	Size		source_len = strlen(source);
	Size		alloc_len;
	int			index;
	dlist_iter	iter;
	pg_crc32	crc;

	/* calculate a hash value */
	INIT_CRC32(crc);
	COMP_CRC32(crc, &extra_libs, sizeof(int32));
	COMP_CRC32(crc, source, source_len);
	FIN_CRC32(crc);

retry:
	index = crc % DEVPROG_HASH_SIZE;
	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	dlist_foreach (iter, &opencl_devprog_shm_values->slot[index])
	{
		devprog_entry *entry
			= dlist_container(devprog_entry, hash_chain, iter.cur);

		if (entry->crc == crc &&
			entry->extra_libs == extra_libs &&
			entry->source_len == source_len &&
			strcmp(entry->source, source) == 0)
		{
			dlist_move_head(&opencl_devprog_shm_values->lru_list,
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
	alloc_len = offsetof(devprog_entry, source[source_len]);
	dprog = pgstrom_shmem_alloc(alloc_len);
	if (!dprog)
		elog(ERROR, "out of shared memory");

	dprog->stag = StromTag_DevProgram;
	SpinLockInit(&dprog->lock);
	dprog->refcnt = 1;
	dlist_init(&dprog->waitq);
	dprog->program = NULL;
	dprog->build_running = false;
    dprog->errmsg = NULL;
	dprog->crc = crc;
	dprog->extra_libs = extra_libs;
	dprog->source_len = source_len;
	strcpy(dprog->source, source);
	opencl_devprog_shm_values->usage += alloc_len;

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
Datum
pgstrom_retain_devprog_key(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);

	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	Assert(dprog->refcnt >= 0);
	dprog->refcnt++;
	SpinLockRelease(&opencl_devprog_shm_values->lock);

	return dprog_key;
}

/*
 * pgstrom_get_devprog_errmsg
 *
 * It returns saved error message if OpenCL built-in compiler raised
 * compile error during clBuildProgram().
 * Note that we assume this devprog_entry is already acquired by the
 * caller, thus returned pointer is safe to reference unless it is not
 * unreferenced by pgstrom_put_devprog_key().
 */
const char *
pgstrom_get_devprog_errmsg(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);
	const char	   *errmsg;

	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	Assert(dprog->refcnt >= 0);
	errmsg = dprog->errmsg;
	SpinLockRelease(&opencl_devprog_shm_values->lock);

	return errmsg;
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
