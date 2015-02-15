/*
 * opencl_serv.c
 *
 * Backend server process to manage OpenCL devices
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
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "nodes/pg_list.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "postmaster/syslogger.h"
#include "storage/bufmgr.h"
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "tcop/tcopprot.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <unistd.h>

/* static variables */
static int		opencl_num_threads;
static shmem_startup_hook_type shmem_startup_hook_next;
static struct {
	slock_t		serial_lock;
} *opencl_serv_shm_values;

/* signal flag */
volatile bool		pgstrom_clserv_exit_pending = false;
/* true, if OpenCL intermidiation server */
volatile bool		pgstrom_i_am_clserv = false;

static void
pgstrom_opencl_sigterm(SIGNAL_ARGS)
{
	pgstrom_clserv_exit_pending = true;
	pgstrom_cancel_server_loop();
	clserv_log("Got SIGTERM");
}

#if 0
static void
pgstrom_opencl_sighup(SIGNAL_ARGS)
{
	pgstrom_clserv_exit_pending = true;
	pgstrom_cancel_server_loop();
	clserv_log("Got SIGHUP");
}
#endif

/*
 * pgstrom_opencl_event_loop
 *
 * main loop of OpenCL intermediation server. each message class has its own
 * processing logic, so all we do here is just call the callback routine.
 */
static void *
pgstrom_opencl_event_loop(void *arg)
{
	pgstrom_message	   *msg;

	while (!pgstrom_clserv_exit_pending)
	{
		CHECK_FOR_INTERRUPTS();
		msg = pgstrom_dequeue_server_message();
		if (!msg)
			continue;
		msg->cb_process(msg);
	}
	return NULL;
}

/*
 * pgstrom_opencl_device_schedule
 *
 * It suggests which opencl device shall be the target of kernel execution.
 * We plan to select an optimal device according to NUMA characteristics
 * and current waiting queue length, however, it is simple round robin
 * right now.
 */
int
pgstrom_opencl_device_schedule(pgstrom_message *message)
{
	static int index = 0;

	return index++ % opencl_num_devices;
}

/*
 * on_shmem_zone_callback
 *
 * It is a callback function for each zone on shared memory segment
 * initialization. It assigns a buffer object of OpenCL for each zone
 * for asynchronous memory transfer later.
 */
static bool
on_shmem_zone_callback(void *address, Size length,
					   const char *label, bool abort_on_error)
{
	cl_int		rc;

	(void)clCreateBuffer(opencl_context,
						 CL_MEM_READ_WRITE |
						 CL_MEM_USE_HOST_PTR,
						 length,
						 address,
						 &rc);
	if (rc != CL_SUCCESS)
	{
		if (abort_on_error)
			elog(ERROR, "clCreateBuffer failed on host memory (%p-%p): %s",
				 address, (char *)address + length - 1, opencl_strerror(rc));
		return false;
	}
	elog(LOG, "PG-Strom: %s %p-%p was mapped (len: %luMB)",
		 label, address, (char *)address + length - 1, length >> 20);
	return true;
}

/*
 * init_opencl_context_and_shmem
 *
 * We can have performance gain using asynchronous DMA transfer when data
 * chunk it moved to OpenCL device from host machine, however, it requires
 * preparations to ensure the memory region to be copied to/from is pinned
 * on RAM; not swapped out. OpenCL provides an interface to map a certain
 * host address area as pinned buffer object, even though its size is
 * restricted to CL_DEVICE_MAX_MEM_ALLOC_SIZE parameter. Usually, it is
 * much less than size of shared memory to be assigned to PG-Strom, around
 * 500MB - 2GB in typical GPU/MIC device. So, we need to split a flat
 * continuous memory into several 'zones' to pin it using OpenCL interface.
 * Because it is a job of OpenCL intermediation server to collect properties
 * of devices, and this server shall be launched post initialization stage,
 * we also have to acquire and pin the shared memory region in the context
 * of OpenCL intermediation server, not postmaster itself.
 */
static void
init_opencl_context_and_shmem(void)
{
	Size	zone_length = LONG_MAX;
	cl_int	i, rc;

	/*
	 * Create an OpenCL context
	 */
	opencl_context = clCreateContext(NULL,
									 opencl_num_devices,
									 opencl_devices,
									 NULL,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clCreateContext failed: %s", opencl_strerror(rc));

	/*
	 * Create an OpenCL command queue for each device
	 */
	for (i=0; i < opencl_num_devices; i++)
	{
		const pgstrom_device_info *dev_info = pgstrom_get_device_info(i);

		opencl_cmdq[i] =
			clCreateCommandQueue(opencl_context,
								 opencl_devices[i],
								 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
								 CL_QUEUE_PROFILING_ENABLE,
								 &rc);
		if (rc != CL_SUCCESS)
			elog(ERROR, "clCreateCommandQueue failed: %s",
				 opencl_strerror(rc));

		if (zone_length > dev_info->dev_max_mem_alloc_size)
			zone_length = (dev_info->dev_max_mem_alloc_size &
						   ~((1UL << 20) - 1));
	}
	/* Lock shared memory of PG-Strom's private area */
	pgstrom_setup_shmem(zone_length, on_shmem_zone_callback);

	/* Lock shared memory of shared buffer area */
	if (!on_shmem_zone_callback(BufferBlocks,
								NBuffers * (Size) BLCKSZ,
								"buffer", false))
	{
		Size	total_size = NBuffers * (Size) BLCKSZ;
		Size	offset;

		Assert((zone_length & (BLCKSZ - 1)) == 0);

		for (offset = 0; offset < total_size; offset += zone_length)
		{
			on_shmem_zone_callback(BufferBlocks + offset,
								   Min(zone_length, total_size - offset),
								   "buffer", true);
		}
	}
}

/*
 * pgstrom_opencl_main
 *
 * Main routine of opencl intermediation server.
 *
 * TODO: enhancement to use multi-threaded message handler.
 */
static void
pgstrom_opencl_main(Datum main_arg)
{
	pthread_t  *threads;
	int			i;

	/* mark this process is OpenCL intermediator */
	pgstrom_i_am_clserv = true;

	/*
	 * Set up signal handlers. Currently, OpenCL Server does not pay
	 * attention on reloading of postgresql.conf, so we can ignore SIGHUP.
	 */
    pqsignal(SIGHUP, SIG_IGN);
    pqsignal(SIGTERM, pgstrom_opencl_sigterm);

    /* We're now ready to receive signals */
    BackgroundWorkerUnblockSignals();

	/* collect opencl platform/device info */
	construct_opencl_device_info();

	/* initialize opencl context and shared memory segment */
	init_opencl_context_and_shmem();
	elog(LOG, "Starting PG-Strom OpenCL Server");

	/*
	 * OK, ready to launch server thread. In the default, it creates
	 * same number with online CPUs, but user can give an explicit
	 * number using "pg_strom.opencl_num_threads" parameter.
	 *
	 * NOTE: sysconf(_SC_NPROCESSORS_ONLN) may not be portable.
	 */
	if (opencl_num_threads == 0)
		opencl_num_threads = sysconf(_SC_NPROCESSORS_ONLN);
	Assert(opencl_num_threads > 0);

	threads = malloc(sizeof(pthread_t) * opencl_num_threads);
	if (!threads)
	{
		elog(LOG, "out of memory");
		return;
	}

	for (i=0; i < opencl_num_threads; i++)
	{
		if (pthread_create(&threads[i],
						   NULL,
						   pgstrom_opencl_event_loop,
						   NULL) != 0)
			break;
	}

	/*
	 * In case of any failure during pthread_create(), worker threads
	 * will be terminated soon, then we can wait for thread joining.
	 */
	if (i < opencl_num_threads)
	{
		elog(LOG, "failed to create server threads");
		pgstrom_clserv_exit_pending = true;
		pgstrom_cancel_server_loop();
	}
	else
		elog(LOG, "PG-Strom: %d of server threads are up", opencl_num_threads);

	while (--i >= 0)
		pthread_join(threads[i], NULL);

	/* got a signal to stop background worker process */
	elog(LOG, "Stopping PG-Strom OpenCL Server");

	/*
	 * close the server queue and returns unprocessed message with error.
	 *
	 * XXX - here is possible bug if server got signals during program
	 *       building; that holds some messages and callback enqueues
	 *       the messages again.
	 */
	pgstrom_close_server_queue();
}

/*
 * __clserv_log
 *
 * Thread-safed error reporting.
 */
extern bool redirection_done;

void
__clserv_log(const char *funcname,
			 const char *filename, int lineno,
			 const char *fmt, ...)
{
	va_list	ap;
	size_t	ofs = 0;
	char	buf[8192];	/* usually enough size */

	/* setting up log message */
	if (Log_error_verbosity == PGERROR_VERBOSE)
		ofs += snprintf(buf+ofs, sizeof(buf)-ofs,
						"LOG: (%s:%d, %s) ", filename, lineno, funcname);
	else
		ofs += snprintf(buf+ofs, sizeof(buf)-ofs,
						"LOG: (%s:%d) ", filename, lineno);
	va_start(ap, fmt);
	ofs += vsnprintf(buf+ofs, sizeof(buf)-ofs, fmt, ap);
	va_end(ap);
	ofs += snprintf(buf+ofs, sizeof(buf)-ofs, "\n");

#ifdef HAVE_SYSLOG
	/* to be implemented later */
#endif

	/*
	 * write to the console (logic copied from write_pipe_chunks)
	 */
	if ((Log_destination & LOG_DESTINATION_STDERR) ||
		whereToSendOutput == DestDebug)
	{
		size_t	len = strlen(buf);
		int		fd = fileno(stderr);
		int		rc;

		if (redirection_done && !am_syslogger)
		{
			PipeProtoChunk	p;
			char   *data = buf;

			memset(&p, 0, sizeof(p.proto));
			p.proto.pid = MyProcPid;

			/* write all but the last chunk */
			while (len > PIPE_MAX_PAYLOAD)
			{
				p.proto.is_last = 'f';
				p.proto.len = PIPE_MAX_PAYLOAD;
				memcpy(p.proto.data, data, PIPE_MAX_PAYLOAD);
				rc = write(fd, &p, PIPE_HEADER_SIZE + PIPE_MAX_PAYLOAD);
				Assert(rc == PIPE_HEADER_SIZE + PIPE_MAX_PAYLOAD);
				data += PIPE_MAX_PAYLOAD;
				len -= PIPE_MAX_PAYLOAD;
			}

			/* write the last chunk */
			p.proto.is_last = 'f';
			p.proto.len = len;
			memcpy(&p.proto.is_last + 1, data, len);
			rc = write(fd, &p, PIPE_HEADER_SIZE + len);
			Assert(rc == PIPE_HEADER_SIZE + len);
		}
		else
		{
			rc = write(fd, buf, len);
		}
	}
}


static void
pgstrom_startup_opencl_server(void)
{
	bool		found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	opencl_serv_shm_values
		= ShmemInitStruct("opencl_serv_shm_values",
						  MAXALIGN(sizeof(*opencl_serv_shm_values)),
						  &found);
	Assert(!found);

	memset(opencl_serv_shm_values, 0, sizeof(*opencl_serv_shm_values));
	SpinLockInit(&opencl_serv_shm_values->serial_lock);
}

void
pgstrom_init_opencl_server(void)
{
	BackgroundWorker	worker;

	/* number of opencl server threads */
	DefineCustomIntVariable("pg_strom.opencl_num_threads",
							"number of opencl server threads",
							NULL,
							&opencl_num_threads,
							0,		/* auto selection */
							0,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* launch a background worker process */	
	memset(&worker, 0, sizeof(BackgroundWorker));
	strcpy(worker.bgw_name, "PG-Strom OpenCL Server");
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = BGW_NEVER_RESTART;

	worker.bgw_main = pgstrom_opencl_main;
	worker.bgw_main_arg = 0;
	RegisterBackgroundWorker(&worker);

	/* acquire shared memory */
	RequestAddinShmemSpace(MAXALIGN(sizeof(*opencl_serv_shm_values)));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_opencl_server;
}
