/*
 * gpu_server.c
 *
 * Routines of GPU/CUDA intermediation server
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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

#include "pg_strom.h"

typedef struct GpuContextHead
{
	pg_atomic_uint32	num_free_context;
	GpuContext_v2		gpucxt[FLEXIBLE_ARRAY_MEMBER];
} GpuContextHead;

typedef struct GpuWorkerState
{
	dlist_node		chain;
	pthread_t		thread;

	GpuContext_v2  *gcontext;
	GpuServComm		comm;
	// track of device memory?
} GpuWorkerState;

/*
 * static variables
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static GpuContextHead  *gpuCtxHead = NULL;
static int				numGpuCtx = 0;
static struct sockaddr_un gpuserv_addr;
static CUdevice			cuda_master_device;
static CUcontext		cuda_master_context;

static slock_t			gpuserv_workers_list_lock;
static dlist_head		gpuserv_active_workers_list;
static dlist_head		gpuserv_dead_workers_list;

static bool				gpuserv_loop;
static bool				is_gpuserv_proc = false;

/*
 * is_gpuserver_process - returns true, if caller is GPU server process
 */
bool
is_gpuserver_process(void)
{
	return is_gpuserv_proc;
}

/*
 * alloc_gpu_context - look up a free GpuContext and assign CUDA context
 * on the caller's thread
 */
static GpuContext_v2 *
gpuserv_alloc_gpucontext(cl_int pgprocno)
{
	int		i;

	if (pg_atomic_read_u32(&gpuCtxHead->num_free_context) > 0)
	{
		for (i=0; i < numGpuCtx; i++)
		{
			GpuContext_v2  *gcontext = &gpuCtxHead->gpucxt[i];

			SpinLockAcquire(&gcontext->lock);
			if (gcontext->refcnt == 0)
			{
				Assert(gcontext->pgprocno < 0);
				gcontext->pgprocno = pgprocno;
				gcontext->refcnt = 1;

				pg_atomic_fetch_sub_u32(&gpuCtxHead->num_free_context, 1);
				SpinLockRelease(&gcontext->lock);

				return gcontext;
			}
			SpinLockRelease(&gcontext->lock);
		}
	}
	return NULL;
}

GpuContext_v2 *
gpuserv_get_gpucontext(cl_int context_id)
{
	GpuContext_v2  *gcontext;

	if (context_id < 0 || context_id >= numGpuCtx)
		return NULL;	/* out of range */

	gcontext = &gpuCtxHead->gpucxt[context_id];
	SpinLockAcquire(&gcontext->lock);
	if (gcontext->refcnt == 0 ||
		gcontext->backend_id != InvalidBackendId)
	{
		SpinLockRelease(&gcontext->lock);
		return NULL;	/* gcontext status is a bit strange */
	}
	gcontext->backend_id = MyProc->backendId;
	gcontext->refcnt++;

	SpinLockRelease(&gcontext->lock);
	return gcontext;
}

void
gpuserv_put_gpucontext(GpuContext_v2 *gcontext)
{
	SpinLockAcquire(&gcontext->lock);
	if (--gcontext->refcnt == 0)
	{
		gcontext->pgprocno = -1;

		pg_atomic_fetch_add_u32(&gpuCtxHead->num_free_context, 1);
		// TODO: Release Shared Memory
		// TODO: Release CUDA Resources if remained
	}
	SpinLockRelease(&gcontext->lock);
}

/*
 * gpuserv_send_command
 *
 * send a command like to GPU Server/Backend
 */
bool
gpuserv_send_command(GpuServComm *comm, int cmd, Datum arg)
{
	struct pollfd pollbuf;
	char		sendbuf[128];
	time_t		timeout_end = time(NULL) + 20;	/* 20sec */
	int			timeout;
	ssize_t		cmd_len;
	ssize_t		sent_len;
	int			rv;

	switch (cmd)
	{
		case GPUSERV_CMD_OPEN:
		case GPUSERV_CMD_CONTEXT:
		case GPUSERV_CMD_CONFIEM:
		case GPUSERV_CMD_GPUTASK:
		case GPUSERV_CMD_RESPOND:
			cmd_len = snprintf(sendbuf, sizeof(sendbuf), "%c %lu\n", cmd, arg);
			break;
		default:
			fprintf(stderr, "%s: unknown command: %c %lu\n",
					__FUNCTION__, cmd, arg);
			return false;
	}

retry:
	pollbuf.fd = comm->sockfd;
	pollbuf.events = POLLOUT | POLLERR;
	pollbuf.revents = 0;

	timeout = (time(NULL) >= timeout_end ? 0 : 500);
	rv = poll(&pollbuf, 1, timeout);
	if (rv < 0)
	{
		fprintf(stderr, "%s: failed on poll(2): %m\n", __FUNCTION__);
		return false;
	}
	else if (rv == 0)
	{
		if (!is_gpuserver_process())
			CHECK_FOR_INTERRUPTS();
		else if (!gpuserv_loop)
			return false;
		goto retry;
	}
	else if ((pollbuf.revents & POLLOUT) == 0)
	{
		fprintf(stderr, "%s: unable to write socket: revents = %d\n",
				__FUNCTION__, pollbuf.revents);
		return false;
	}

	/* write out */
	sent_len = 0;
	do {
		rv = write(comm->sockfd, sendbuf + sent_len, cmd_len - sent_len);
		if (rv > 0)
			sent_len += rv;
		else
		{
			if (rv == 0)
				fprintf(stderr, "%s: no bytes were sent\n", __FUNCTION__);
			else
				fprintf(stderr, "failed on write(2): %m\n");
			return false;
		}
	} while (sent_len < cmd_len);

	return true;
}

/*
 * gpuserv_recv_command
 *
 * read a command like from GPU Server/Backend
 */
bool
gpuserv_recv_command(GpuServComm *comm, int *p_cmd, Datum *p_arg)
{
	struct pollfd pollbuf;
	char	   *pos, *head, *end;
	int			rv;

	for (;;)
	{
		/* try to parse a command line on recvbuf */
		for (head = pos = comm->recvbuf + comm->bufpos,
			 end = comm->recvbuf + comm->bufend; pos < end; pos++)
		{
			cl_int		cmd;
			cl_ulong	arg;

			if (*pos == '\n')
			{
				comm->bufpos = pos + 1 - comm->recvbuf;
				*pos = '\0';
				if (sscanf(head, "%c %lu", &cmd, &arg) != 2)
				{
					fprintf(stderr, "%s: invalid command format [%s]\n",
							__FUNCTION__, head);
					return false;
				}

				if (cmd == GPUSERV_CMD_OPEN ||
					cmd == GPUSERV_CMD_CONTEXT ||
					cmd == GPUSERV_CMD_CONFIRM ||
					cmd == GPUSERV_CMD_GPUTASK ||
					cmd == GPUSERV_CMD_RESPOND)
				{
					*p_cmd = cmd;
					*p_arg = arg;
					return true;
				}
				fprintf(stderr, "%s: unknown command [%s]\n",
						__FUNCTION__, head);
				return false;
			}
		}

		/* shift to the head */
		if (comm->bufpos > 0)
		{
			memmove(comm->recvbuf,
					comm->recvbuf + comm->bufpos,
					comm->bufend - comm->bufpos);
			comm->bufend -= comm->bufpos;
			comm->bufpos = 0;
		}

	retry:
		pollbuf.fd = comm->sockfd;
		pollbuf.events = POLLIN | POLLERR;
		pollbuf.revents = 0;

		timeout = (time(NULL) >= timeout_end ? 0 : 500);
		rv = poll(&pollbuf, 1, timeout);
		if (rv < 0)
		{
			fprintf(stderr, "%s: failed on poll(2): %m\n", __FUNCTION__);
			return false;
		}
		else if (rv == 0)
		{
			if (!is_gpuserver_process())
				CHECK_FOR_INTERRUPTS();
			else if (!gpuserv_loop)
				return false;
			goto retry;
		}
		else if ((pollbuf.revents & POLLIN) == 0)
		{
			fprintf(stderr, "%s: unable to read from the socket: %08x\n",
					__FUNCTION__, pollbuf.revents);
			return false;
		}

		/* OK, read a line from the socket */
		rv = read(comm->sockfd,
				  comm->recvbuf + comm->bufend,
				  sizeof(comm->recvbuf) - comm->bufend);
		if (rv <= 0)
		{
			if (rv == 0)
				fprintf(stderr, "%s: no bytes were read\n", __FUNCTION__);
			else
				fprintf(stderr, "failed on read(2): %m\n");
			return false;
		}
		comm->bufend += rv;
	}
}

/*
 *
 */
static void *
gpuserv_worker_main(void *thread_arg)
{
	GpuWorkerState	   *wstate = (GpuWorkerState *) thread_arg;
	GpuContext_v2	   *gcontext = NULL;
	int					cmd;
	Datum				cmdarg;

	/* Get backendId from the peer */
	if (!gpuserv_recv_command(&wstate->comm, &cmd, &cmdarg) ||
		cmd != GPUSERV_CMD_OPEN)
		goto out_exit;

	/* Lookup a free GpuContext */
	gcontext = gpuserv_alloc_gpucontext(DatumGetInt32(cmdarg));
	if (!gcontext)
		goto out_exit;

	/* Send back context_id and wait for acknowledge */
	if (!gpuserv_send_command(&wstate->comm,
							  GPUSERV_CMD_CONTEXT,
							  Int32GetDatum(gcontext->context_id)))
		goto out_exit;

	if (!gpuserv_recv_command(&wstate->comm, &cmd, &cmdarg) ||
		cmd != GPUSERV_CMD_CONFIRM || !DatumGetBool(cmdarg))
		goto out_exit;

	/* OK, assign CUDA context on this thread */
	rc = cuCtxPushCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		goto ...;

	if (sigsetjmp(...) == 0)
	{
		/* wait for user's request */
		while (gpuserv_recv_command(&wstate->comm, &cmd, &cmdarg) &&
			   cmd == GPUSERV_CMD_GPUTASK)
		{
			// transform the portable address to local address
			// enqueue the task to pending list.


			// enqueue the task to pending list.

			// TODO: unblocked recv_command

			
		}
		// close the socket

	}
	else
	{
		// close the socket

		// destroy the CUDA context

		// reconstruct the CUDA context



	}
	// put GpuContext -- release host pinned memory if any




	// enqueue a pending list
	// fetch a task from pending list
	// call the cb_process callback in this context

	// error handling
	// once error happen, put error status on device context
	// then destroy the current cuda context.
	// it implies all the device memory an relevant resource
	// shall be released

	// host pinned memory shall be released when DevContext
	// become unreferenced


	return NULL;
}

static void
gpuserv_launch_worker(pgsocket sockfd)
{
	GpuWorkerState *wstate;
	dlist_node	   *dnode;

	wstate = malloc(sizeof(GpuWorkerState));
	if (!wstate)
	{
		GELOG("out of memory: %m");
		close(sockfd);
		return;
	}
	memset(wstate, 0, sizeof(GpuWorkerState));
	wstate->comm.sockfd = sockfd;

	/* tracked as an active worker */
	SpinLockAcquire(&gpuserv_workers_list_lock);
	dlist_push_tail(&gpuserv_active_workers_list, &wstate->chain);
	SpinLockRelease(&gpuserv_workers_list_lock);

	/* launch a new worker thread */
	if (pthread_create(&wstate->thread, NULL,
					   gpuserv_worker_main, wstate) != 0)
	{
		CELOG("failed on pthread_create: %m");
		SpinLockAcquire(&gpuserv_workers_list_lock);
		dlist_delete(&wstate->chain);
		SpinLockRelease(&gpuserv_workers_list_lock);
		close(wstate->comm.sockfd);
		free(wstate);
	}
}

static void
gpuserv_reclaim_worker(bool sync_actives)
{
	GpuWorkerState *wstate;
	dlist_node	   *dnode;

	if (sync_actives)
	{
		/*
		 * NOTE: GpuWorkerState object shall be moved to the dead_list
		 * at the tail of worker's main, so we don't need to move wstate
		 * into the dead_list by ourself.
		 */
		SpinLockAcquire(&gpuserv_workers_list_lock);
		while (!dlist_is_empty(&gpuserv_active_workers_list))
		{
			wstate = dlist_head_element(GpuWorkerState, chain,
										&gpuserv_active_workers_list);
			SpinLockRelease(&gpuserv_workers_list_lock);
			if (pthread_join(&wstate->thread, NULL) != 0)
				GELOG("failed on pthread_join: %m");
			SpinLockAcquire(&gpuserv_workers_list_lock);
		}
		SpinLockRelease(&gpuserv_workers_list_lock);
	}

	SpinLockAcquire(&gpuserv_workers_list_lock);
	while (!dlist_is_empty(&gpuserv_dead_workers_list))
	{
		dnode = dlist_pop_head_node(&gpuserv_dead_workers_list);
		wstate = dlist_container(GpuWorkerState, chain, dnode);
		SpinLockRelease(&gpuserv_workers_list_lock);

		if (pthread_join(&wstate->thread, NULL) != 0)
			GELOG("failed on pthread_join: %m");
		SpinLockAcquire(&gpuserv_workers_list_lock);
	}
	SpinLockRelease(&gpuserv_workers_list_lock);
}

/*
 * gpuserv_init_cuda_context - construction of the master CUDA context;
 * that is never used to run GPU kernel, but used to pin shared DMA
 * buffer region, for reusing.
 */
static void
gpuserv_init_cuda_context(void)
{
	CUresult	rc;

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", errorText(rc));

	rc = cuDeviceGet(&cuda_master_device, devAttrs[0].DEV_ID);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&cuda_master_context, 0, cuda_master_device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));

	/* construction of GpuContext_v2 */
	for (i=0; i < numGpuCtx; i++)
	{
		GpuContext_v2  *gcontext = &gpuCtxHead->gpucxt[i];
		DevAttributes  *devattrs = &devAttrs[i % numDevAttrs];

		memset(gcontext, 0, sizeof(*gcontext));
		gcontext->context_id = i;
		SpinLockInit(&gcontext->lock);
		gcontext->refcnt = 0;
		gcontext->backend_id = InvalidBackendId;
		gcontext->device_id = devattrs->DEV_ID;

		rc = cuDeviceGet(&gcontext->cuda_device, gcontext->device_id);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", errorText(rc));

		rc = cuCtxCreate(&gcontext->cuda_context, 0, gcontext->cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));
	}
}

/*
 * gpuserv_open_server_socket - open a unix domain socket for accept(2)
 */
static pgsocket
gpuserv_open_server_socket(void)
{
	pgsocket	sockfd;

	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog();

	if (bind(sockfd, (struct sockaddr *)uaddr, sizeof(uaddr)) != 0)
		elog();

	if (listen(sockfd, 10) != 0)
		elog();

	return sockfd;
}

/*
 * gpuserv_accept_server_socket - accept a client connection
 */
static pgsocket
gpuserv_accept_server_socket(pgsocket serv_sock)
{
	struct pollfd	pollbuf;
	pgsocket		sockfd;
	int				retval;

	pollbuf.fd = serv_sock;
	pollbuf.events = POLLERR | POLLIN;
	pollbuf.revents = 0;

	retval = poll(&pollbuf, 1, 800);	/* wake up per 0.8sec */
	if (retval < 0)
	{
		fprintf(stderr, "failed on poll(2): %m\n");
		gpuserv_loop = false;
		return PGINVALID_SOCKET;
	}
	else if (retval == 0)
		return PGINVALID_SOCKET;

	sockfd = accept(serv_sock, NULL, NULL);
	if (sockfd < 0)
	{
		fprintf(stderr, "failed on accept(2): %m\n");
		return PGINVALID_SOCKET;
	}
	return sockfd;
}

/*
 * gpuserv_close_server_socket
 */
static void
gpuserv_close_server_socket(pgsocket serv_sock)
{
	if (close(serv_sock) != 0)
		fprintf(stderr, "failed on close to CUDA server socket: %m\n");

	if (unlink(uaddr.sun_path) != 0)
		fprintf(stderr, "failed on unlink(\"%s\"): %m\n", uaddr.sun_path);
}

/*
 * SIGTERM handler
 */
static void
gpuserv_got_sigterm(SIGNAL_ARGS)
{
	gpuserv_loop = false;
}

/*
 * pgstrom_cuda_serv_main - entrypoint of the CUDA server process
 */
static void
pgstrom_cuda_serv_main(Datum main_arg)
{
	pqsignal(SIGTERM, gpuserv_got_sigterm);
	BackgroundWorkerUnblockSignals();

	gpuserv_init_cuda_context();
	serv_sock = gpuserv_open_server_socket();

	SpinLockInit(&gpuserv_workers_list_lock);
	dlist_init(&gpuserv_active_workers_list);
	dlist_init(&gpuserv_dead_workers_list);

	gpuserv_loop = true;
	while (gpuserv_loop)
	{
		pgsocket		sockfd;
		dlist_node	   *dnode;
		GpuWorkerState *wstate;

		sockfd = gpuserv_accept_server_socket(serv_sock);
		if (sockfd != PGINVALID_SOCKET)
			gpuserv_launch_worker(sockfd);
		gpuserv_reclaim_worker(false);
	}
	gpuserv_close_server_socket(serv_sock);
	gpuserv_reclaim_worker(true);

	/*
	 * No need to release individual CUDA resources because these will be
	 * released on process exit time.
	 */


}

/*
 * pgstrom_startup_cuda_serv
 */
static void
pgstrom_startup_cuda_serv(void)
{
	Size	length;
	int		i;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	length = MAXALIGN(offsetof(DevContextHead, devcontext[numDevContexts]));
	devCtxHead = ShmemInitStruct("DevContextHead", length, &found);
	Assert(!found);

	/* init device context */
	memset(devCtxHead, 0, length);
	SpinLockInit(&devCxtHead->lock);
	devCxtHead->num_free_context = numDevContexts;

	for (i=0; i < numDevContexts; i++)
	{
		DevContext		devcxt = &devCtxHead->devcxt[i];
		DevAttributes	dattrs = &devAttrs[i % numDevAttrs];

		devcxt->dev_id = dattrs->DEV_ID; 
		devcxt->refcnt = 0;
		devcxt->backend_id = InvalidBackendId;
		devcxt->errcode = 0;
	}
}

/*
 * pgstrom_init_gpu_server
 */
void
pgstrom_init_gpu_server(void)
{
	static char	   *cuda_visible_devices;
	BackgroundWorker worker;
	CUresult		rc;
	long			suffix;
	int				count;

	/*
	 * Setup path of the UNIX domain socket
	 */
	suffix = (long) getpid();
	for (;;)
	{
		gpuserv_addr.sun_family = AF_UNIX;
		snprintf(gpuserv_addr.sun_path,
				 sizeof(gpuserv_addr.sun_path),
				 "base/pgsql_tmp/.pg_strom.sock.%u", suffix);
		if (stat(gpuserv_addr.sun_path, &stbuf) == 0)
			suffix ^= PostmasterRandom();
		else if (errno == ENOENT)
			break;	/* OK, We can create a UNIX domain socket */
		else
			elog(ERROR, "failed on stat(2) for \"%s\"", gpuserv_addr.sun_path);
	}

	/*
	 * Maximum number of CUDA context we can use concurrently.
	 */
	DefineCustomIntVariable("pg_strom.num_gpu_contexts",
							"Number of GPU device context",
							NULL,
							&numDevContexts,
							4 * numDevAttrs,
							2,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* Launch a background worker process (CUDA Server) */
	memset(&worker, 0, sizeof(BackgroundWorker));
	strcpy(worker.bgw_name, "PG-Strom CUDA Server");
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	worker.bgw_restart_time = 5;
	worker.bgw_main = pgstrom_cuda_serv_main;
	worker.bgw_main_arg = 0;

	RegisterBackgroundWorker(&worker);

	/* acquire shared memory */
	RequestAddinShmemSpace(MAXALIGN(offsetof(DevContextHead,
											 devcontext[numDevContexts])));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_serv;
}

/*
 * gpuserv_elog - a utility routine to dump log message from the multi-
 * threading environment
 */
void
gpuserv_elog(int elevel,
			 const char *filename,
			 int lineno,
			 const char *funcname,
			 const char *fmt,...)
{
	va_list		ap;
	size_t		ofs = 0;
	char		buf[2048];	/* usually enough size */

	/* setting up a log message */
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

	fprintf(stderr, "PG-Strom: %s\n", buf);

	/*
	 * TODO: write back error message to the backend
	 */



}
