/*
 * opencl_devprog.c
 *
 * Routines to manage device programs/kernels
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "catalog/pg_type.h"
#include "common/pg_crc.h"
#include "funcapi.h"
#include "storage/barrier.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include <limits.h>
#include "pg_strom.h"

static shmem_startup_hook_type shmem_startup_hook_next;
static int	reclaim_threshold;
static int	itemid_offset_shift;
static int	itemid_flags_shift;
static int	itemid_length_shift;
bool		devprog_enable_optimize;

#define DEVPROG_HASH_SIZE	2048

static struct {
	slock_t		lock;
	Size		usage;
	dlist_head	lru_list;
	dlist_head	slot[DEVPROG_HASH_SIZE];
} *opencl_devprog_shm_values;

typedef struct {
	StromObject	sobj;		/* = StromTag_DevProgram */
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
	int32		extra_flags;	/* set of DEVFUNC_NEEDS_* */
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
	struct timeval	tv;
	char		   *errmsg = NULL;
	cl_int			i, rc;

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
#ifdef PGSTROM_DEBUG
		else
		{
			char	buffer[128 * 1024];	/* 128KB */
			size_t	buflen;

			if (clGetProgramBuildInfo(program,
									  opencl_devices[i],
									  CL_PROGRAM_BUILD_LOG,
									  sizeof(buffer),
									  buffer,
									  &buflen) == CL_SUCCESS)
			{
				buffer[buflen] = '\0';
				clserv_log("opencl build log:\n%s", buffer);
			}
		}
#endif
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

		if (msg->pfm.enabled)
		{
			gettimeofday(&tv, NULL);
			msg->pfm.time_kern_build += timeval_diff(&msg->pfm.tv, &tv);
		}
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
	struct timeval	tv;
	cl_int		rc;

	/* performance monitor */
	if (message->pfm.enabled)
		gettimeofday(&message->pfm.tv, NULL);

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
		cl_program		program;
		const char	   *sources[32];
		size_t			lengths[32];
		char			build_opts[1024];
		cl_uint			ofs;
		cl_uint			count = 0;
		static size_t	common_code_length = 0;
		static const char *platform_specific = NULL;

		/* check platform specific configuration */
		if (!platform_specific)
		{
			const pgstrom_device_info   *dev_info = pgstrom_get_device_info(0);
			const pgstrom_platform_info	*pl_info = dev_info->pl_info;

			/*
			 * AMD's runtime makes compiler crash!, if FLEXIBLE_ARRAY_MEMBER
			 * is empty or zero.
			 */
			if (strncmp(pl_info->pl_name, "AMD ", 4) == 0)
			{
				platform_specific = " -DFLEXIBLE_ARRAY_MEMBER=1";
			}
			else
			{
				platform_specific = " -DFLEXIBLE_ARRAY_MEMBER=0";
			}
		}

		/* common opencl header */
		if (!common_code_length)
			common_code_length = strlen(pgstrom_opencl_common_code);
		sources[count] = pgstrom_opencl_common_code;
		lengths[count] = common_code_length;
		count++;

		/*
		 * Supplemental OpenCL Libraries
		 */

		/* opencl mathlib */
		if (dprog->extra_flags & DEVFUNC_NEEDS_MATHLIB)
		{
			static size_t	mathlib_code_length = 0;

			if (!mathlib_code_length)
				mathlib_code_length = strlen(pgstrom_opencl_mathlib_code);
			sources[count] = pgstrom_opencl_mathlib_code;
            lengths[count] = mathlib_code_length;
            count++;
		}

		/* opencl timelib */
		if (dprog->extra_flags & DEVFUNC_NEEDS_TIMELIB)
		{
			static size_t	timelib_code_length = 0;

			if (!timelib_code_length)
				timelib_code_length = strlen(pgstrom_opencl_timelib_code);
			sources[count] = pgstrom_opencl_timelib_code;
			lengths[count] = timelib_code_length;
			count++;
		}

		/* opencl textlib */
		if (dprog->extra_flags & DEVFUNC_NEEDS_TEXTLIB)
		{
			static size_t	textlib_code_length = 0;

			if (!textlib_code_length)
				textlib_code_length = strlen(pgstrom_opencl_textlib_code);
			sources[count] = pgstrom_opencl_textlib_code;
			lengths[count] = textlib_code_length;
			count++;
		}

		/* opencl numeric */
		if (dprog->extra_flags & DEVFUNC_NEEDS_NUMERIC)
		{
			static size_t  numeric_code_length = 0;

			if (!numeric_code_length)
				numeric_code_length = strlen(pgstrom_opencl_numeric_code);
			sources[count] = pgstrom_opencl_numeric_code;
			lengths[count] = numeric_code_length;
			count++;
		}

		/*
		 * main logic for each GPU task (scan, sort, join)
		 */

		/* gpuscan device implementation */
		if (dprog->extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
		{
			static size_t	gpuscan_code_length = 0;

			if (!gpuscan_code_length)
				gpuscan_code_length = strlen(pgstrom_opencl_gpuscan_code);
			sources[count] = pgstrom_opencl_gpuscan_code;
			lengths[count] = gpuscan_code_length;
			count++;
		}
		/* hashjoin device implementation */
		if (dprog->extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
		{
			static size_t	hashjoin_code_length = 0;

			if (!hashjoin_code_length)
				hashjoin_code_length = strlen(pgstrom_opencl_hashjoin_code);
			sources[count] = pgstrom_opencl_hashjoin_code;
			lengths[count] = hashjoin_code_length;
			count++;
		}
		/* gpupreagg device implementation */
		if (dprog->extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
		{
			static size_t	gpupreagg_code_length = 0;

			if (!gpupreagg_code_length)
				gpupreagg_code_length = strlen(pgstrom_opencl_gpupreagg_code);
			sources[count] = pgstrom_opencl_gpupreagg_code;
			lengths[count] = gpupreagg_code_length;
			count++;
		}
		/* gpusort device implementation */
		if (dprog->extra_flags & DEVKERNEL_NEEDS_GPUSORT)
		{
			static size_t	gpusort_code_length = 0;

			if (!gpusort_code_length)
				gpusort_code_length = strlen(pgstrom_opencl_gpusort_code);
			sources[count] = pgstrom_opencl_gpusort_code;
			lengths[count] = gpusort_code_length;
			count++;
		}

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
		dprog->program = program;
		dprog->build_running = true;
		if (message)
			dlist_push_tail(&dprog->waitq, &message->chain);

		/*
		 * NOTE: clBuildProgram() kicks kernel build asynchronously or
		 * synchronously depending on the OpenCL driver. In our trial,
		 * intel's driver performs asynchronously, however, nvidia's
		 * driver has synchronous manner.
		 * Its callback function on build completion acquires the lock
		 * of device-program, we have to release it prior to the call of
		 * clBuildProgram().
		 */
		SpinLockRelease(&dprog->lock);

		Assert(SIZEOF_VOID_P == 8 || SIZEOF_VOID_P == 4);
		ofs = snprintf(build_opts, sizeof(build_opts),
#ifdef PGSTROM_DEBUG
					   " -Werror"
#endif
					   " -DOPENCL_DEVICE_CODE -DHOSTPTRLEN=%u -DBLCKSZ=%u %s"
					   " -DITEMID_OFFSET_SHIFT=%u"
					   " -DITEMID_FLAGS_SHIFT=%u"
					   " -DITEMID_LENGTH_SHIFT=%u"
					   " -DMAXIMUM_ALIGNOF=%u",
					   SIZEOF_VOID_P, BLCKSZ, platform_specific,
					   itemid_offset_shift,
					   itemid_flags_shift,
					   itemid_length_shift,
					   MAXIMUM_ALIGNOF);
		if (dprog->extra_flags & DEVKERNEL_DISABLE_OPTIMIZE)
			ofs += snprintf(build_opts + ofs, sizeof(build_opts) - ofs,
							" -cl-opt-disable");
		if (dprog->extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
			ofs += snprintf(build_opts + ofs, sizeof(build_opts) - ofs,
							" -DKERNEL_IS_GPUSCAN=1");
		if (dprog->extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
			ofs += snprintf(build_opts + ofs, sizeof(build_opts) - ofs,
                            " -DKERNEL_IS_HASHJOIN=1");
		if (dprog->extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
			ofs += snprintf(build_opts + ofs, sizeof(build_opts) - ofs,
							" -DKERNEL_IS_GPUPREAGG=1");
		if (dprog->extra_flags & DEVKERNEL_NEEDS_GPUSORT)
			ofs += snprintf(build_opts + ofs, sizeof(build_opts) - ofs,
							" -DKERNEL_IS_GPUSORT=1");

		rc = clBuildProgram(program,
							opencl_num_devices,
							opencl_devices,
							build_opts,
							clserv_devprog_build_callback,
							dprog);
		if (rc != CL_SUCCESS)
		{
			dlist_mutable_iter iter;

			clserv_log("clBuildProgram failed: %s", opencl_strerror(rc));

			SpinLockAcquire(&dprog->lock);
			/*
			 * We need to pay attention both cases when synchronous build-
			 * failure or asynchronous build job input failure.
			 * In the first case, cl_program object is already released on
			 * the callback handler, so we have nothing to do anymore.
			 * Elsewhere, it is asynchronous job input failure, so we have
			 * to clean up cl_program object and mark this entry as a bad-
			 * program.
			 */

			/*
			 * NOTE: In case of synchronous build failure, program-build
			 * callback is already called; that makes response message
			 * with error code, so this message should be no longer handled
			 * by OpenCL server. (This callback clears 'build_running').
			 * In this case, we returns the caller NULL, to break its
			 * cb_process handler immediately, without duplicated message
			 * queuing.
			 */
			if (!dprog->build_running)
			{
				Assert(dlist_is_empty(&dprog->waitq));
				SpinLockRelease(&dprog->lock);
				return NULL;
			}

			/*
			 * otherwise, all the waiting messages shall be enqueued again
			 * to generate error response messages.
			 */
			dprog->build_running = false;
			dprog->program = BAD_OPENCL_PROGRAM;
			rc = clReleaseProgram(program);
			Assert(rc == CL_SUCCESS);

			dlist_foreach_modify(iter, &dprog->waitq)
			{
				pgstrom_message *msg
					= dlist_container(pgstrom_message, chain, iter.cur);

				dlist_delete(&msg->chain);
				if (msg->pfm.enabled)
				{
					gettimeofday(&tv, NULL);
					msg->pfm.time_kern_build
						+= timeval_diff(&msg->pfm.tv, &tv);
					gettimeofday(&msg->pfm.tv, NULL);
				}
				pgstrom_enqueue_message(msg);
			}
			goto out_unlock;
		}
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

		/* nealy zero seconds for kernel build, even if perfmon is enabled */
		if (message->pfm.enabled)
		{
			gettimeofday(&tv, NULL);
			message->pfm.time_kern_build
				+= timeval_diff(&message->pfm.tv, &tv);
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
 * It returns a devprog-key that holds the given source and extra_flags.
 * If not found on the device program table, it also create a new one
 * and insert it, then returns its key.
 *
 * TODO: add a resource tracking here to ensure device program is released.
 */
Datum
pgstrom_get_devprog_key(const char *source, int32 extra_flags)
{
	devprog_entry *dprog = NULL;
	Size		source_len = strlen(source);
	Size		alloc_len;
	int			index;
	dlist_iter	iter;
	pg_crc32	crc;

	/* calculate a hash value */
	INIT_CRC32C(crc);
	COMP_CRC32C(crc, &extra_flags, sizeof(int32));
	COMP_CRC32C(crc, source, source_len);
	FIN_CRC32C(crc);

retry:
	index = crc % DEVPROG_HASH_SIZE;
	SpinLockAcquire(&opencl_devprog_shm_values->lock);
	dlist_foreach (iter, &opencl_devprog_shm_values->slot[index])
	{
		devprog_entry *entry
			= dlist_container(devprog_entry, hash_chain, iter.cur);

		if (entry->crc == crc &&
			entry->extra_flags == extra_flags &&
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
	alloc_len = offsetof(devprog_entry, source[source_len + 1]);
	dprog = pgstrom_shmem_alloc(alloc_len);
	if (!dprog)
		elog(ERROR, "out of shared memory");

	dprog->sobj.stag = StromTag_DevProgram;
	SpinLockInit(&dprog->lock);
	dprog->refcnt = 1;
	dlist_init(&dprog->waitq);
	dprog->program = NULL;
	dprog->build_running = false;
    dprog->errmsg = NULL;
	dprog->crc = crc;
	dprog->extra_flags = extra_flags;
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
 * pgstrom_get_devprog_extra_flags
 *
 * it returns extra flags of device program
 */
int32
pgstrom_get_devprog_extra_flags(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);

	if (!dprog)
		return 0;
	/* no need to acquire lock because of read-only field once constructed */
	return dprog->extra_flags;
}

/*
 * pgstrom_get_devprog_kernel_source
 *
 * it returns kernel source of device program
 */
const char *
pgstrom_get_devprog_kernel_source(Datum dprog_key)
{
	devprog_entry  *dprog = (devprog_entry *) DatumGetPointer(dprog_key);

	if (!dprog)
		return 0;
	/* no need to acquire lock because of read-only field once constructed */
	return dprog->source;
}

/*
 * pgstrom_opencl_program_info
 *
 * shows all the device programs being on the program cache
 */
typedef struct {
	Datum		key;
	int			refcnt;
	int			state;	/* 'b' = build running, 'e' = error, 'r' = ready */
	pg_crc32	crc;
	int32		flags;
	Size		length;
	text	   *source;
	text	   *errmsg;
} devprog_info;

Datum
pgstrom_opencl_program_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	devprog_info	*dp_info;
	HeapTuple		tuple;
	Datum			values[8];
	bool			isnull[8];
	char			buf[256];

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		List		   *dp_list = NIL;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(8, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "key",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "refcnt",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "state",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "crc",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "flags",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "length",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 7, "source",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 8, "errmsg",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		SpinLockAcquire(&opencl_devprog_shm_values->lock);
		PG_TRY();
		{
			dlist_iter	iter;
			int			i;

			for (i=0; i < DEVPROG_HASH_SIZE; i++)
			{
				dlist_foreach (iter, &opencl_devprog_shm_values->slot[i])
				{
					devprog_entry *entry
						= dlist_container(devprog_entry, hash_chain, iter.cur);

					dp_info = palloc0(sizeof(devprog_info));
					dp_info->key = PointerGetDatum(entry);
					SpinLockAcquire(&entry->lock);
					dp_info->refcnt = entry->refcnt;
					if (!entry->program)
						dp_info->state = 'n';	/* not built */
					else if (entry->program == BAD_OPENCL_PROGRAM)
						dp_info->state = 'e';	/* build error */
					else if (entry->build_running)
						dp_info->state = 'b';	/* build running */
					else
						dp_info->state = 'r';	/* program is ready */
					SpinLockRelease(&entry->lock);
					dp_info->crc = entry->crc;
					dp_info->flags = entry->extra_flags;
					dp_info->length = entry->source_len;
					dp_info->source = cstring_to_text(entry->source);
					if (entry->errmsg)
						dp_info->errmsg = cstring_to_text(entry->errmsg);

					dp_list = lappend(dp_list, dp_info);
				}
			}
		}
		PG_CATCH();
		{
			SpinLockRelease(&opencl_devprog_shm_values->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(&opencl_devprog_shm_values->lock);

		fncxt->user_fctx = dp_list;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
        SRF_RETURN_DONE(fncxt);

	dp_info = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	snprintf(buf, sizeof(buf), "%p", (void *)dp_info->key);
	values[0] = CStringGetTextDatum(buf);
	values[1] = Int32GetDatum(dp_info->refcnt);
	snprintf(buf, sizeof(buf), "%s",
			 (dp_info->state == 'n' ? "not built" :
			  (dp_info->state == 'e' ? "build error" :
			   (dp_info->state == 'b' ? "build running" : "program ready"))));
	values[2] = CStringGetTextDatum(buf);
	snprintf(buf, sizeof(buf), "0x%08x", dp_info->crc);
	values[3] = CStringGetTextDatum(buf);
	values[4] = Int32GetDatum(dp_info->flags);
	values[5] = Int32GetDatum(dp_info->length);
	values[6] = PointerGetDatum(dp_info->source);
	if (dp_info->errmsg)
		values[7] = PointerGetDatum(dp_info->errmsg);
	else
		isnull[7] = true;

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_opencl_program_info);

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
	ItemIdData	item_id;
	cl_uint		code;

	/* turn on/off device program optimization */
	DefineCustomBoolVariable("pg_strom.devprog_enable_optimization",
							 "enables optimization on device program build",
							 NULL,
							 &devprog_enable_optimize,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* threshold to reclaim the cached opencl programs */
	DefineCustomIntVariable("pg_strom.devprog_reclaim_threshold",
							"threahold to reclaim device program objects",
							NULL,
							&reclaim_threshold,
							16 << 20,	/* 16MB */
							0,	/* 0 means no threshold, not recommended */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/*
	 * NOTE: Here is no C standard for bitfield layout (thus, OpenCL does not
	 * support bitfields), so we need to tell run-time compiler exact layout
	 * of the ItemIdData structure.
	 */
	Assert(sizeof(item_id) == sizeof(code));
	memset(&item_id, 0, sizeof(ItemIdData));
	item_id.lp_off = 1;
	memcpy(&code, &item_id, sizeof(ItemIdData));
	for (itemid_offset_shift = 0;
		 ((code >> itemid_offset_shift) & 0x0001) == 0;
		 itemid_offset_shift++);

	memset(&item_id, 0, sizeof(ItemIdData));
	item_id.lp_flags = 1;
	memcpy(&code, &item_id, sizeof(ItemIdData));
	for (itemid_flags_shift = 0;
		 ((code >> itemid_flags_shift) & 0x0001) == 0;
		 itemid_flags_shift++);

	memset(&item_id, 0, sizeof(ItemIdData));
	item_id.lp_len = 1;
	memcpy(&code, &item_id, sizeof(ItemIdData));
	for (itemid_length_shift = 0;
		 ((code >> itemid_length_shift) & 0x0001) == 0;
		 itemid_length_shift++);

	/* aquires shared memory region */
	RequestAddinShmemSpace(MAXALIGN(sizeof(*opencl_devprog_shm_values)));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_opencl_devprog;
}
