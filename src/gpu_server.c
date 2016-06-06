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

typedef struct GpuServerState
{
	slock_t			lock;
	cl_uint			num_free_servers;
} GpuServerState;

typedef struct GpuContextHead
{
	slock_t			lock;	/* lock of the list below */
	dlist_head		active_gpu_context_list;
	dlist_head		free_gpu_context_list;
	GpuContext_v2	gpu_contexts[FLEXIBLE_ARRAY_MEMBER];
} GpuContextHead;

/*
 * static variables
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static pgsocket			gpu_server_sock = -1;
static int				gpu_server_got_signal = 0;
static GpuServerState  *gpuServState = NULL;
static GpuContextHead  *gpuContextHead;
static int				numGpuServers;	/* GUC */
static int				numGpuContexts;	/* GUC */
static CUdevice			cuda_device = NULL;
static CUcontext		cuda_context = NULL;


#define GPUSERV_CMD_OPEN
#define GPUSERV_CMD_GPUSCAN
#define GPUSERV_CMD_GPUJOIN
#define GPUSERV_CMD_GPUPREAGG
#define GPUSERV_CMD_GPUSORT
#define GPUSERV_CMD_PLCUDA
#define GPUSERV_CMD_CLOSE

typedef struct GpuServCommand
{
	cl_int		command;	/* one of the GPUSERV_CMD_* */
	cl_int		peer_fd;	/* if SCM_RIGHTS, elsewhere -1 */
	union {
		struct {
			cl_uint		context_id;
			BackendId	backend_id;
		} open;
		struct {
//			portable_addr_t gask;
			uintptr_t gtask;
		} task;
	} u;
} GpuServCommand;





static int
gpuserv_recv_command(pgsocket sockfd, GpuServCommand *cmd, long timeout_ms)
{
	GpuServCommand *__cmd;
	struct msghdr	msg;
	struct iovec	iov;
	struct cmsghdr *cmsg;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	int				peer_fd = -1;
	ssize_t			retval;

	retval = WaitLatchOrSocket(MyProc->procLatch,
							   WL_LATCH_SET |
							   WL_POSTMASTER_DEATH |
							   WL_SOCKET_READABLE,
							   sockfd,
							   timeout_ms);
	if ((retval & WL_SOCKET_READABLE) == 0)
		return (int)retval;

	memset(&msg, 0, sizeof(msg));
	memset(&iov, 0, sizeof(iov));
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	msg.msg_control = cmsgbuf;
	msg.msg_controllen = sizeof(cmsgbuf);

	retval = recvmsg(sockfd, &msg, MSG_CMSG_CLOEXEC | MSG_DONTWAIT);
	if (retval < 0)
		elog(ERROR, "failed on recvmsg: %m");
	if ((cmsg = CMSG_FIRSTHDR(&msg)) != NULL)
	{
		if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS)
			elog(ERROR, "unexpected cmsghdr {cmsg_level=%d cmsg_type=%d}",
				 cmsg->cmsg_level, cmsg->cmsg_type);
		/* needs to exit once then restart server */
		if ((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int) > 1)
			elog(FATAL, "we cannot handle two or more FDs at once");
		if (CMSG_NXTHDR(&msg, cmsg) != NULL)
			elog(FATAL. "we cannot handle two or more cmsghdr at once");

		peer_fd = ((int *)CMSG_DATA(cmsg))[0];
		// TODO: register the peer FD to the current resource owner or
		//       something other stuff
	}

	if (retval == 0 || msg.msg_iovlen == 0)
	{
		cmd->command = GPUSERV_CMD_CLOSE;
		cmd->peer_fd = peer_fd;
		elog(LOG, "no bytes received, likely connection closed");
		return 0;
	}
	else if (msg.msg_iovlen > 1 ||
			 iov.iov_len != sizeof(GpuServCommand))
	{
		elog(ERROR, "recvmsg(2) returned unexpected bytes format");
	}

	__cmd = (GpuServCommand *) iov.iov_base;
	cmd->command = __cmd->command;
	cmd->peer_fd = peer_fd;
	switch (cmd->command)
	{
		case GPUSERV_CMD_OPEN:
			cmd->context_id = __cmd->context_id;
			cmd->backend_id = __cmd->backend_id;
			if (cmd->peer_fd)
				elog(ERROR, "OPEN command never takes Peer-FD");
			break;
		case GPUSERV_CMD_GPUSCAN:
		case GPUSERV_CMD_GPUJOIN:
		case GPUSERV_CMD_GPUPREAGG:
		case GPUSERV_CMD_GPUSORT:
		case GPUSERV_CMD_PLCUDA:
		case GPUSERV_CMD_CLOSE:
			cmd->gputask = __cmd->gputask;
			break;
		default:
			elog(ERROR, "unexpected GpuServCommand: %d", __cmd->command);
			break;
	}

	return 0;
}

static bool
gpuserv_send_command(pgsocket sockfd, GpuServCommand *cmd, long timeout_ms)
{
	struct msghdr	msg;
	struct iovec	iov;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	int				retval;

	retval = WaitLatchOrSocket(MyProc->procLatch,
							   WL_LATCH_SET |
							   WL_POSTMASTER_DEATH |
							   WL_SOCKET_WRITABLE,
							   sockfd,
							   timeout_ms);
	if ((retval & WL_SOCKET_WRITABLE) == 0)
		return retval;

	memset(&msg, 0, sizeof(struct msghdr));
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	iov.iov_base = cmd;
	iov.iov_len = sizeof(GpuServCommand);

	if (cmd->peer_fd >= 0)
	{
		struct cmsghdr *cmsg;

		msg.msg_control = cmsgbuf;
		msg.msg_controllen = sizeof(cmsgbuf);
		cmsg = CMSG_FIRSTHDR(&msg);
		cmsg->cmsg_level = SOL_SOCKET;
		cmsg->cmsg_type = SCM_RIGHTS;
		cmsg->cmsg_len = CMSG_LEN(sizeof(int));
		((int *)CMSG_DATA(cmsg))[0] = cmd->peer_fd;
	}

	retval = sendmsg(sockfd, &msg, 0);
	if (retval < 0)
		elog(ERROR, "failed on sendmsg, peer_fd=%d: %m", peer_fd);
	else if (retval == 0)
		elog(ERROR, "no bytes send using sendmsg, peer_fd=%d: %m", peer_fd);

	return 0;
}




/*
 * gpuserv_accept_connection - accept a client connection
 */
static pgsocket
gpuserv_accept_connection(void)
{
	struct pollfd	pollbuf;
	pgsocket		sockfd;
	GpuServCommand	cmd;
	int				retval;

	pollbuf.fd = gpu_server_sock;
	pollbuf.events = POLLERR | POLLIN;
	pollbuf.revents = 0;
	retval = poll(&pollbuf, 1, 600);	/* wake up per 0.6sec */
	if (retval < 0)
	{
		if (gpu_server_got_signal == 0)
			gpu_server_got_signal = (errno == 0 ? -1 : errno);
		return PGINVALID_SOCKET;
	}
	else if (retval == 0)
		return PGINVALID_SOCKET;		/* no connection request arrived */

	sockfd = accept(gpu_server_sock, NULL, NULL);
	if (sockfd < 0)
	{
		if (errno == EAGAIN)
			return PGINVALID_SOCKET;
		elog(ERROR, "failed on accept(2): %m");
	}
	// TODO: needs to register the sockfd to close on ERROR

	/*
	 * Each backend shall send OPEN command at first
	 */
	if (gpuserv_recv_command(sockfd, &cmd, 1000) != 0 ||
		cmd.command != GPUSERV_CMD_OPEN)
		elog(ERROR, "could not receive OPEN command from the backend");

	if (cmd->context_id >= numGpuContexts)
		elog(ERROR, "supplied context_id is out of range");
	gcontext = &gpuContextHead->gpu_contexts[cmd->context_id];
	SpinLockAcquire(&gcontext->lock);
	if (gcontext->refcnt == 0 ||		/* nobody own the GpuContext */
		gcontext->backend == NULL ||	/* no backend assigned yet */
		gcontext->server != NULL ||		/* server already assigned */
		gcontext->backend->backendId != cmd->open.backend_id)
	{
		SpinLockRelease(&gcontext->lock);
		elog(ERROR, "supplied context_id is incorrect");
	}
	gcontext->refcnt++;
	gcontext->server = MyProc;
	SpinLockRelease(&gcontext->lock);

	// TODO: attach GpuContext



	return sockfd;
}

/*
 * gpuserv_init_connection - ensure both backend/server are connected to
 * a particular GpuContext
 */
static bool
gpuserv_init_connection(....)
{
	// get a open connection message


	// set latch of the backend


}

/*
 * gpuserv_init_cuda_context
 */
static void
gpuserv_init_cuda_context(cl_uint server_id)
{
	static bool	cuda_runtime_initialized = false;
	cl_int		device_id;
	CUresult	rc;

	if (!cuda_runtime_initialized)
	{
		rc = cuInit(0);
		if (rc != CUDA_SUCCESS)
			elog(FATAL, "failed on cuInit: %s", errorText(rc));

		device_id = devAttrs[server_id % numDevAttrs].DEV_ID;
		rc = cuDeviceGet(&cuda_device, device_id);
		if (rc != CUDA_SUCCESS)
			elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

		cuda_runtime_initialized = true;
	}
	rc = cuCtxCreate(&cuda_context, 0, cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));
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
pgstrom_gpu_server_main(Datum server_id)
{
	pgsocket	pgsocket;

	pqsignal(SIGTERM, gpuserv_got_sigterm);
	BackgroundWorkerUnblockSignals();

	gpuserv_init_cuda_context(DatumGetUint32(server_id));

	PG_TRY();
	{
		while (gpu_server_got_signal == 0)
		{
			sockfd = gpuserv_accept_connection();
			if (sockfd == PGINVALID_SOCKET)
				continue;

			// initial link up
			if (!...)
			{
				close(sockfd);
				continue;
			}






		}
	}
	PG_CATCH();
	{
		// delete any cuda context
		// put gpu context
		// -> may release portable shared memory
		// -> unpin the segment & shmem buffer

		PG_RE_THROW();
	}
	PG_END_TRY();

	// put gpu context



}

/*
 * pgstrom_startup_gpu_server
 */
static void
pgstrom_startup_gpu_server(void)
{
	Size	length;
	int		i;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* gpuServState */
	gpuServState = ShmemInitStruct("gpuServState",
								   sizeof(GpuServerState),
								   &found);
	Assert(!found);
	memset(gpuServState, 0, sizeof(GpuServerState));

	/* gpuContextHead */
	length = offsetof(GpuContextHead, gpu_context[numGpuContexts]);
	gpuContextHead = ShmemInitStruct("gpuContextHead", length, &found);
	Assert(!found);
	memset(gpuContextHead, 0, length);

	SpinLockInit(&gpuContextHead->lock);
	dlist_init(&gpuContextHead->active_gpu_context_list);
	dlist_init(&gpuContextHead->free_gpu_context_list);

	for (i=0; i < numGpuContexts; i++)
	{
		GpuContext_v2  *gcontext = &gpuContextHead->gpu_context[i];

		gcontext->refcnt = 0;
		gcontext->backend = NULL;
		gcontext->server = NULL;
		dlist_push_tail(&gpuContextHead->free_gpu_context_list,
						&gcontext->chain);
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
	uint32			TotalProcs;	/* see InitProcGlobal() */

	/*
	 * Maximum number of GPU servers we can use concurrently.
	 * (it is equivalent to the number of CUDA contexts)
	 */
	DefineCustomIntVariable("pg_strom.num_gpu_servers",
							"number of GPU/CUDA intermediation servers",
							NULL,
							&numGpuServers,
							2,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/*
	 * Maximum number of GPU context.
	 */
	TotalProcs = MaxBackends + NUM_AUXILIARY_PROCS + max_prepared_xacts;
	DefineCustomIntVariable("pg_strom.num_gpu_contexts",
							"maximum number of GpuContext",
							NULL,
							&numGpuContexts,
							2 * TotalProcs,
							MaxBackends,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/*
	 * Setup a UNIX domain socket for listen/accept
	 */
	gpu_server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
	if (gpu_server_sock < 0)
		elog(ERROR, "failed to open a UNIX domain socket: %m");

	suffix = (long) getpid();
	for (;;)
	{
		gpuserv_addr.sun_family = AF_UNIX;
		snprintf(gpuserv_addr.sun_path,
				 sizeof(gpuserv_addr.sun_path),
				 "base/pgsql_tmp/.pg_strom.sock.%u", suffix);
		if (bind(gpu_server_sock,
				 (struct sockaddr *) &gpuserv_addr,
				 sizeof(gpuserv_addr)) == 0)
			break;
		else if (errno == EADDRINUSE)
		{
			elog(LOG, "UNIX domain socket \"%s\" is already in use: %m",
				 gpuserv_addr.sun_path);
			suffix ^= PostmasterRandom();
		}
		else
			elog(ERROR, "failed on bind(2): %m");
	}

	if (listen(gpu_server_sock, numGpuServers) != 0)
		elog(ERROR, "failed on listen(2): %m");

	/* 0.6sec timeout; also interval to check signals */
	timeout.tv_sec = 0;
	timeout.tv_usec = 600000;
	if (setsockopt(gpu_server_sock, SOL_SOCKET, SO_RCVTIMEO,
				   &timeout, sizeof(timeout)) != 0)
		elog(ERROR, "failed on setsockopt(2): %m");

	/*
	 * Launch background worker processes for GPU/CUDA servers
	 */
	for (i=0; i < numGpuServers; i++)
	{
		BackgroundWorker	worker;

		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "PG-Strom GPU/CUDA Server [%d]", i);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = 5;
		worker.bgw_main = pgstrom_cuda_serv_main;
		worker.bgw_main_arg = i;
	}
	RegisterBackgroundWorker(&worker);

	/*
	 * Acquire the shared memory segment
	 */
	RequestAddinShmemSpace(MAXALIGN(sizeof(GpuServerState)));
	RequestAddinShmemSpace(MAXALIGN(sizeof(GpuContext) * numGpuContexts));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_server;
}
