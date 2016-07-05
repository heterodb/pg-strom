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
#include "access/hash.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"

#include <sys/socket.h>
#include <sys/un.h>

#define GPUSERV_CMD_OPEN		0x101
#define GPUSERV_CMD_TASK		0x102
#define GPUSERV_CMD_CLOSE		0x103

typedef struct GpuServConnState
{
	slock_t			lock;
	cl_int			num_wait_servs;
	cl_int			num_pending_conn;
	volatile Latch *waiters_latch[FLEXIBLE_ARRAY_MEMBER];
} GpuServConnState;

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
			void	   *gtask;	/* to be located on the shared segment */
		} task;
	} u;
} GpuServCommand;

/*
 * public variables
 */
SharedGpuContext	   *currentSharedGpuContext = NULL;

/*
 * static variables
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static GpuServConnState	*gpuServConnState = NULL;
static struct sockaddr_un gpuserv_addr;
static pgsocket			gpu_server_sock = PGINVALID_SOCKET;
static int				gpu_server_id = -1;
static bool				gpu_server_got_sigterm = false;
static int				numGpuServers;			/* GUC */
static int				GpuServerCommTimeout;	/* GUC */
static CUdevice			cuda_device = NULL;
static CUcontext		cuda_context = NULL;

/*
 * IsGpuServerProcess - returns true, if current process is gpu server
 */
bool
IsGpuServerProcess(void)
{
	return (bool)(gpu_server_id >= 0);
}

/*
 * gpuservWakeUpProcesses - wakes up sleeping GPU server processes
 */
void
gpuservWakeUpProcesses(cl_uint max_procs)
{
	Bitmapset  *waiting_servers = &gpuServConnState->waiting_servers;
	PGPROC	   *proc;
	cl_int		idx = -1;

	Assert(max_procs > 0);
	SpinLockAcquire(&gpuServConnState->lock);
	while ((idx = bms_next_member(waiting_servers, idx)) >= 0)
	{
		Assert(idx < ProcGlobal->allProcCount);

		proc = &ProcGlobal->allProcs[idx];
		SetLatch(&proc->procLatch);
		bms_del_member(waiting_servers, idx);

		if (--max_procs == 0)
			break;
	}
	SpinLockRelease(&gpuServConnState->lock);
}

/*
 * gpuservWaitForEvents - wait for the upcoming event; it is a simple wrapper
 * for WaitLatchOrSocket, except for registration to waiting_servers
 */
static int
gpuservWaitForEvents(volatile Latch *latch,
					 int wakeEvents, pgsocket sock, long timeout)
{
	cl_int		procno = MyProc->pgprocno;

	SpinLockAcquire(&gpuServConnState->lock);
	bms_add_member(&gpuServConnState->waiting_servers, procno);
	SpinLockRelease(&gpuServConnState->lock);

	return WaitLatchOrSocket(latch, wakeEvents, sock, timeout);
}








/* SIGTERM handler */
static void
gpuserv_got_sigterm(SIGNAL_ARGS)
{
	int		save_errno = errno;

	gpu_server_got_sigterm = true;

	SetLatch(MyLatch);

	errno = save_errno;
}

#define GPUSERV_CHECK_FOR_INTERRUPTS()									\
	do {																\
		CHECK_FOR_INTERRUPTS();											\
		if (gpu_server_got_sigterm)										\
			ereport(FATAL,												\
					(errcode(ERRCODE_ADMIN_SHUTDOWN),					\
					 errmsg("Terminating PG-Strom GPU/CUDA Server[%d]",	\
							gpu_server_id)));							\
	} while(0)







/*
 * gpuservSendCommand - an internal low-level interface
 */
static bool
gpuservSendCommand(GpuContext_v2 *gcontext, GpuServCommand *cmd, long timeout)
{
	struct msghdr	msg;
	struct iovec	iov;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	int				retval;
	int				ev;

	ev = gpuservWaitForEvents(MyLatch,
							  WL_LATCH_SET |
							  WL_POSTMASTER_DEATH |
							  WL_SOCKET_WRITEABLE |
							  (timeout < 0 ? 0 : WL_TIMEOUT),
							  gcontext->sockfd,
							  timeout);
	/* Is the socket writable? */
	if ((ev & WL_SOCKET_WRITEABLE) == 0)
		return false;

	memset(&msg, 0, sizeof(struct msghdr));
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	iov.iov_base = cmd;
	iov.iov_len = sizeof(GpuServCommand);

	/* do we attach a file-descriptor with this message? */
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

		if (IsGpuServerProcess())
			elog(FATAL, "Bug? GpuServer never send peer FD");
	}
	retval = sendmsg(gcontext->sockfd, &msg, 0);
	if (retval < 0)
		elog(ERROR, "failed on sendmsg(2), PeerFD=%d: %m",
			 cmd->peer_fd);
	else if (retval == 0)
		elog(ERROR, "no bytes sent using sendmsg(2), PeerFD=%d, %m",
			 cmd->peer_fd);

	return true;
}

/*
 * gpuservSendGpuTask - enqueue a GpuTask to the socket
 */
bool
gpuservSendGpuTask(GpuContext_v2 *gcontext, GpuTask *gtask, int peer_fd)
{
	GpuServCommand	cmd;

	cmd.command = GPUSERV_CMD_TASK;
	cmd.peer_fd = peer_fd;
	cmd.u.task.gtask = gtask;

	while (!gpuservSendCommand(gcontext, &cmd,
							   GpuServerCommTimeout))
	{
		GPUSERV_CHECK_FOR_INTERRUPTS();
	}

	/*
	 * TODO: A proper handling when client closed the connection
	 * prior to send a response
	 */
	return true;
}

/*
 * gpuservRecvCommand - internal low level interface
 */
static bool
gpuservRecvCommand(pgsocket sockfd, GpuServCommand *cmd, long timeout)
{
	GpuServCommand	__cmd;
	struct msghdr	msg;
	struct iovec	iov;
	struct cmsghdr *cmsg;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	int				peer_fd = -1;
	int				ev;
	ssize_t			retval;

	ev = gpuservWaitForEvents(MyLatch,
							  WL_LATCH_SET |
							  WL_POSTMASTER_DEATH |
							  WL_SOCKET_READABLE |
							  (timeout < 0 ? 0 : WL_TIMEOUT),
							  sockfd,
							  timeout);
	if ((ev & WL_SOCKET_READABLE) == 0)
		return false;

	/* fetch a message from the socket */
	memset(&msg, 0, sizeof(msg));
	memset(&iov, 0, sizeof(iov));
	iov.iov_base = &__cmd;;
	iov.iov_len = sizeof(GpuServCommand);
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	msg.msg_control = cmsgbuf;
	msg.msg_controllen = sizeof(cmsgbuf);

	retval = recvmsg(sockfd, &msg, 0);
	if (retval < 0)
		elog(ERROR, "failed on recvmsg: %m");

	PG_TRY();
	{
		/* pick up peer FD, if any */
		if ((cmsg = CMSG_FIRSTHDR(&msg)) != NULL)
		{
			if (cmsg->cmsg_level != SOL_SOCKET ||
				cmsg->cmsg_type != SCM_RIGHTS)
				elog(ERROR, "unexpected cmsghdr {cmsg_level=%d cmsg_type=%d}",
					 cmsg->cmsg_level, cmsg->cmsg_type);
			/* needs to exit once then restart server */
			if ((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int) > 1)
				elog(FATAL, "we cannot handle two or more FDs at once");
			if (CMSG_NXTHDR(&msg, cmsg) != NULL)
				elog(FATAL, "we cannot handle two or more cmsghdr at once");

			peer_fd = ((int *)CMSG_DATA(cmsg))[0];
			if (!IsGpuServerProcess())
				elog(FATAL, "Bug? backend never receive peer FD");
		}

		if (retval == 0 || msg.msg_iovlen == 0)
		{
			/* a dummy command if connection refused */
			if (peer_fd >= 0)
				elog(FATAL, "Bug? peer-FD was moved with connection closed");
			cmd->command = GPUSERV_CMD_CLOSE;
			cmd->peer_fd = -1;
			elog(LOG, "no bytes received, likely connection closed");
		}
		else if (msg.msg_iovlen == 1 && iov.iov_len == sizeof(GpuServCommand))
		{
			memcpy(cmd, &__cmd, sizeof(GpuServCommand));
			cmd->peer_fd = peer_fd;
		}
		else
		{
			elog(ERROR, "recvmsg(2) received unexpected bytes format");
		}
	}
	PG_CATCH();
	{
		if (peer_fd >= 0)
			close(peer_fd);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return true;
}

/*
 * gpuservRecvGpuTask - fetch a GpuTask from the socket
 */
GpuTask *
gpuservRecvGpuTask(GpuContext_v2 *gcontext, int *peer_fd)
{
	GpuServCommand	cmd;
	GpuTask		   *gtask = NULL;

	while (!gpuservRecvCommand(gcontext->sockfd, &cmd,
							   GpuServerCommTimeout))
	{
		GPUSERV_CHECK_FOR_INTERRUPTS();

		// TODO: flush out if completed task is not empty
	}

	if (cmd.command == GPUSERV_CMD_TASK)
	{
		/*
		 * Once peer_fd is tracked by the GpuContext, we can raise error
		 * with no explicit file-descriptor handling because GpuContext's
		 * resource tracker will clean up.
		 */
		if (cmd.peer_fd < 0)
			*peer_fd = PGINVALID_SOCKET;
		else
		{
			PG_TRY();
			{
				trackFileDesc(gcontext, cmd.peer_fd);
			}
			PG_CATCH();
			{
				close(cmd.peer_fd);
				PG_RE_THROW();
			}
			PG_END_TRY();
			*peer_fd = cmd.peer_fd;
		}
		gtask = cmd.u.task.gtask;
		Assert(dmaBufferValidatePtr(gtask));
	}
	else if (cmd.command == GPUSERV_CMD_CLOSE)
	{
		Assert(cmd.peer_fd < 0);
		*peer_fd = PGINVALID_SOCKET;
	}
	else
		elog(ERROR, "Bug? unexpected GpuServCommand %d", cmd.command);

	return gtask;
}

/*
 * gpuserv_open_connection - open a unix domain socket from the backend
 * (it may fail if no available GPU server)
 */
bool
gpuservOpenConnection(GpuContext_v2 *gcontext)
{
	pgsocket	sockfd = PGINVALID_SOCKET;

	Assert(!IsGpuServerProcess());
	Assert(gcontext->sockfd == PGINVALID_SOCKET);

	/*
	 * Confirm the number of waiting server process and backend processes
	 * which is in-progress for connections. If we have no chance to make
	 * a connection with servers, give up making a connection.
	 */
	SpinLockAcquire(&gpuServConnState->lock);
	if (gpuServConnState->num_pending_conn >= gpuServConnState->num_wait_servs)
	{
		SpinLockRelease(&gpuServConnState->lock);
		return false;
	}
	gpuServConnState->num_pending_conn++;
	SpinLockRelease(&gpuServConnState->lock);

	PG_TRY();
	{
		sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (sockfd < 0)
			elog(ERROR, "failed on socket(2): %m");

		while (connect(sockfd, (struct sockaddr *)&gpuserv_addr,
					   sizeof(struct sockaddr_un)) != 0)
		{
			/* obviously out of GPU server resources */
			if (errno == EAGAIN)
			{
				close(sockfd);
				sockfd = PGINVALID_SOCKET;
				break;
			}
			else if (errno != EINTR)
				elog(ERROR, "failed on connect(2): %m");
			GPUSERV_CHECK_FOR_INTERRUPTS();
		}

		/*
		 * NOTE: Exchange message with very short timeout to handle a corner
		 * case when no available GPU server, even though num_pending_conn is
		 * less than num_wait_servs on above checks.
		 * Once a GPU server process got a signal during accept(2), it will
		 * break accept(2) a new connection then decrease the num_wait_servs
		 * soon, but we cannot ensure no backend processes already began to
		 * open a connection according to the wrond information.
		 * The listen(2) will setup some margin for connection, thus, we
		 * cannot detect on connect(2), but the backend shall not get any
		 * response from the GPU server with reasonable time.
		 */
		if (sockfd != PGINVALID_SOCKET)
		{
			GpuServCommand	cmd;
			int				ev;

			/* assign sockfd on the GpuContext */
			gcontext->sockfd = sockfd;

			/* send OPEN command, with 100ms timeout */
			cmd.command = GPUSERV_CMD_OPEN;
			cmd.peer_fd = -1;
			cmd.u.open.context_id = gcontext->shgcon->context_id;
			cmd.u.open.backend_id = MyProc->backendId;
			if (!gpuservSendCommand(gcontext, &cmd, 100))
			{
				/* ...revert it... */
				gcontext->sockfd = PGINVALID_SOCKET;
				close(sockfd);
			}
			else
			{
				/*
				 * If GPU server successfully processes the OPEN command, it
				 * shall set backend's latch with a reasonable delay.
				 */
				ResetLatch(MyLatch);

				ev = WaitLatch(MyLatch,
							   WL_LATCH_SET |
							   WL_TIMEOUT |
							   WL_POSTMASTER_DEATH,
							   100);	/* 100ms */
				if ((ev & WL_LATCH_SET) == 0)
				{
					/* ...revert it... */
					gcontext->sockfd = PGINVALID_SOCKET;
					close(sockfd);
				}
			}
		}
	}
	PG_CATCH();
	{
		/* close the socket if opened */
		if (sockfd >= 0)
			close(sockfd);
		/* revert the number of pending clients */
		SpinLockAcquire(&gpuServConnState->lock);
		gpuServConnState->num_pending_conn--;
		SpinLockRelease(&gpuServConnState->lock);
	}
	PG_END_TRY();

	/* revert the number of pending clients */
	SpinLockAcquire(&gpuServConnState->lock);
	gpuServConnState->num_pending_conn--;
	SpinLockRelease(&gpuServConnState->lock);

	return (gcontext->sockfd != PGINVALID_SOCKET ? true : false);
}

/*
 * gpuservAcceptConnection - accept a client connection
 */
static GpuContext_v2 *
gpuservAcceptConnection(void)
{
	GpuServCommand	cmd;
	pgsocket		sockfd = PGINVALID_SOCKET;

	Assert(IsGpuServerProcess());

	/* server is now waiting */
	SpinLockAcquire(&gpuServConnState->lock);
	gpuServConnState->num_wait_servs++;
	SpinLockRelease(&gpuServConnState->lock);

	PG_TRY();
	{
		while ((sockfd = accept(gpu_server_sock, NULL, NULL)) < 0)
		{
			GPUSERV_CHECK_FOR_INTERRUPTS();
			if (errno != EINTR && errno != EAGAIN)
				elog(ERROR, "failed on accept(2): %m");
		}
	}
	PG_CATCH();
	{
		/* server now stopped waiting */
		SpinLockAcquire(&gpuServConnState->lock);
		gpuServConnState->num_wait_servs--;
		SpinLockRelease(&gpuServConnState->lock);

		PG_RE_THROW();
	}
	PG_END_TRY();

	/* server now stopped waiting */
	SpinLockAcquire(&gpuServConnState->lock);
	gpuServConnState->num_wait_servs--;
	SpinLockRelease(&gpuServConnState->lock);

	if (sockfd < 0)
		return NULL;

	/* receive OPEN command (timeout=100ms) */
	if (!gpuservRecvCommand(sockfd, &cmd, 100) ||
		cmd.command != GPUSERV_CMD_OPEN)
	{
		close(sockfd);
		return NULL;
	}
	return AttachGpuContext(sockfd,
							cmd.u.open.context_id,
							cmd.u.open.backend_id);
}

/*
 * gpuserv_session_main - it processes a session once established
 */
static void
gpuservSessionMain(GpuContext_v2 *gcontext)
{
	GpuTask		   *gtask;
	int				peer_fd;

	while (true	/* untila connection closed? */)
	{
		// TODO: Flush completed tasks to the backend
		GPUSERV_CHECK_FOR_INTERRUPTS();

		elog(LOG, "wait for recv packet");
		gtask = gpuservRecvGpuTask(gcontext, &peer_fd);
		if (!gtask)
			break;		/* EOF */

		elog(LOG, "%u wake up", getpid());
		/* do something for each gtask */
	}
}

/*
 * pgstrom_cuda_serv_main - entrypoint of the CUDA server process
 */
static void
gpuserv_main(Datum __server_id)
{
	cl_int		device_id;
	cl_int		dindex;
	CUresult	rc;

	/* I am a GPU server process */
	gpu_server_id = DatumGetInt32(__server_id);
	Assert(gpu_server_id >= 0);
	pqsignal(SIGTERM, gpuserv_got_sigterm);
	BackgroundWorkerUnblockSignals();

	/* Init CUDA runtime */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	dindex = gpu_server_id % numDevAttrs;
	device_id = devAttrs[dindex].DEV_ID;
	rc = cuDeviceGet(&cuda_device, device_id);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&cuda_context,
					 CU_CTX_SCHED_AUTO,
					 cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxCreate: %s", errorText(rc));

	/* memory context per session duration */
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "GPU Server");
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "GPU Server per session",
												 ALLOCSET_DEFAULT_MINSIZE,
												 ALLOCSET_DEFAULT_INITSIZE,
												 ALLOCSET_DEFAULT_MAXSIZE);
	elog(LOG, "PG-Strom GPU/CUDA Server [%d] is now ready on GPU-%d %s",
		 gpu_server_id, devAttrs[dindex].DEV_ID, devAttrs[dindex].DEV_NAME);

	PG_TRY();
	{
		GpuContext_v2  *gcontext;

		while (true)
		{
			gcontext = gpuservAcceptConnection();
			if (gcontext)
			{
				gpuservSessionMain(gcontext);
				PutGpuContext(gcontext);
				MemoryContextReset(CurrentMemoryContext);
			}
		}
	}
	PG_CATCH();
	{
		/*
		 * NOTE: ereport() eventually kills the background worker process.
		 * It also releases any CUDA resources privately held by this worker,
		 * so, we don't need to reclaim these objects here.
		 * SharedGpuContext is exception. Unless putting the SharedGpuContext
		 * we hold, nobody will release its shared resources.
		 */
		if (currentSharedGpuContext)
			PutSharedGpuContext(currentSharedGpuContext);
	}
	PG_END_TRY();

	/* detach shared resources if any */
	if (currentSharedGpuContext)
		PutSharedGpuContext(currentSharedGpuContext);
}

/*
 * gpuserv_on_postmaster_exit - remove UNIX domain socket on shutdown of
 * the postmaster process.
 */
static void
gpuserv_on_postmaster_exit(int code, Datum arg)
{
	if (MyProcPid == PostmasterPid)
	{
		if (unlink(gpuserv_addr.sun_path) != 0)
			elog(WARNING, "failed on unlink('%s'): %m", gpuserv_addr.sun_path);
	}
}

/*
 * pgstrom_startup_gpu_server
 */
static void
pgstrom_startup_gpu_server(void)
{
	Size		required;
	bool		found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	RequestAddinShmemSpace(offsetof(GpuServConnState,
                                    waiters_latch[numGpuServers]));
    shmem_startup_hook_next = shmem_startup_hook;
    shmem_startup_hook = pgstrom_startup_gpu_server;

	required = offsetof(GpuServConnState,
						waiters_latch[numGpuServers])):
	gpuServConnState = ShmemInitStruct("gpuServConnState",
									   REQUIRED<
									   &found);
	Assert(!found);
	memset(gpuServConnState, 0, sizeof(GpuServConnState));
	SpinLockInit(&gpuServConnState->lock);
	memset(GpuServConnState->waiters_latch,0,sizeof(Latch *) * numGpuServers);


    nwords = (ProcGlobal->allProcCount +
			  BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	gpuServConnState->waiting_servers.nwords = nwords;
	memset(gpuServConnState->waiting_servers.words,
		   0, sizeof(bitmapword) * nwords);
}

/*
 * pgstrom_init_gpu_server
 */
void
pgstrom_init_gpu_server(void)
{
	cl_uint			suffix;
	cl_uint			i;
	struct timeval	timeout;

	/*
	 * Maximum number of GPU servers we can use concurrently.
	 * (it is equivalent to the number of CUDA contexts)
	 */
	DefineCustomIntVariable("pg_strom.num_gpu_servers",
							"number of GPU/CUDA intermediation servers",
							NULL,
							&numGpuServers,
							2,
							2,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/*
	 * Connection timeout on unix domain socket between the backend and
	 * the GPU server.
	 */
	DefineCustomIntVariable("pg_strom.gpu_server_comm_timeout",
							"timeous[ms] for communication with GPU server",
							NULL,
							&GpuServerCommTimeout,
							-1,		/* no timeout */
							-1,
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
				 ".pg_strom.server.sock.%u", suffix);
		if (bind(gpu_server_sock,
				 (struct sockaddr *) &gpuserv_addr,
				 sizeof(gpuserv_addr)) == 0)
			break;
		else if (errno == EADDRINUSE)
		{
			elog(LOG, "UNIX domain socket \"%s\" is already in use: %m",
				 gpuserv_addr.sun_path);
			suffix++;
		}
		else
			elog(ERROR, "failed on bind(2): %m");
	}

	if (listen(gpu_server_sock, numGpuServers) != 0)
		elog(ERROR, "failed on listen(2): %m");

	/* check signal for each 500ms during accept(2) */
	timeout.tv_sec = 0;
	timeout.tv_usec = 500 * 1000;	/* 500ms */
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
		worker.bgw_main = gpuserv_main;
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* request for the static shared memory */
	RequestAddinShmemSpace(offsetof(GpuServConnState,
									waiters_latch[numGpuServers]));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_server;
	/* cleanup of UNIX domain socket */
	on_proc_exit(gpuserv_on_postmaster_exit, 0);
}
