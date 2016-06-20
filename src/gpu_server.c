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
#include "utils/memutils.h"
#include "pg_strom.h"


#define GPUSERV_CMD_OPEN
#define GPUSERV_CMD_GPUSCAN
#define GPUSERV_CMD_GPUJOIN
#define GPUSERV_CMD_GPUPREAGG
#define GPUSERV_CMD_GPUSORT
#define GPUSERV_CMD_PLCUDA
#define GPUSERV_CMD_CLOSE

typedef struct GpuServConnState
{
	slock_t		lock;
	cl_int		num_wait_servs;
	cl_int		num_pending_conn;
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
static GpuContext_v2   *CurrentGpuContext = NULL;

/*
 * static variables
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static GpuServConnState	*gpuServConnState = NULL;
static struct sockaddr_un gpuserv_addr;
static pgsocket			gpu_server_sock = -1;
static bool				gpu_server_process = false;
static int				gpuServGotSignal = 0;
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
	return gpu_server_process;
}

/*
 * Resource tracker - it is used to detect unreleased resources on the
 * GPU/CUDA server context. If orphan resources are found, after the
 * execution of a particular session, we can raise a warning and take
 * additional clean-up.
 */
#define RESOURCE_TRACKER_CLASS__FILEDESC	1
#define RESOURCE_TRACKER_CLASS__GPUMEMORY	2

typedef struct ResourceTracker
{
	cl_int			resclass;
	union {
		int			fdesc;		/* RESOURCE_TRACKER_CLASS__FILEDESC */
		CUdeviceptr	devptr;		/* RESOURCE_TRACKER_CLASS__GPUMEMORY */
	} u;
	dlist_node		chain;
} ResourceTracker;
#define ResourceTrackerNumSlots		175
static dlist_head	resource_tracker_slots[ResourceTrackerNumSlots];
static dlist_head	resource_tracker_free;

static inline ResourceTracker *
alloc_resource_tracker(void)
{
	ResourceTracker	   *tracker;

	if (dlist_is_empty(&resource_tracker_free))
		return (ResourceTracker *)
			MemoryContextAllocZero(CacheMemoryContext,
								   sizeof(ResourceTracker));

	tracker = dlist_container(ResourceTracker, chain,
							  dlist_pop_head_node(&resource_tracker_free));
	memset(tracker, 0, sizeof(ResourceTracker));
	return tracker;
}

/*
 * tracker of file descriptor
 */
static void
trackFileDesc(int fdesc)
{
	ResourceTracker *tracker = alloc_resource_tracker();
	cl_uint		index;

	tracker->resclass = RESOURCE_TRACKER_CLASS__FILEDESC;
	tracker->u.fdesc = fdesc;
	index = (hash_any(tracker, offsetof(ResourceTracker, chain))
			 % ResourceTrackerNumSlots);
	dlist_push_tail(&resource_tracker_slots[index], &tracker->chain);
}

static void
closeFileDesc(int fdesc)
{
	ResourceTracker	temp;
	cl_uint		index;
	dlist_iter	iter;

	memset(temp, 0, sizeof(ResourceTracker));
	temp.resclass = RESOURCE_TRACKER_CLASS__FILEDESC;
	temp.u.fdesc = fdesc;
	index = (hash_any(&temp, offsetof(ResourceTracker, chain))
			 % ResourceTrackerNumSlots);

	dlist_foreach(iter, &resource_tracker_slots[index])
	{
		ResourceTracker *tracker = (ResourceTracker *)
			dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->resclass == RESOURCE_TRACKER_CLASS__FILEDESC &&
			tracker->u.fdesc == fdesc)
		{
			dlist_delete(&tracker->chain);
			dlist_push_tail(&resource_tracker_free, &tracker->chain);
			if (close(fdesc) != 0)
				elog(ERROR, "failed on close(2): %m");
			return;
		}
	}
	elog(ERROR, "fdesc %d was not registered", fdesc);
}

/*
 * track of device memory allocation/free
 */
CUdeviceptr
gpuMemAlloc_v2(size_t bytesize)
{
	ResourceTracker *tracker = alloc_resource_tracker();
	CUdeviceptr	devptr;
	CUresult	rc;
	cl_uint		index;

	rc = cuMemAlloc(&devptr, bytesize);
	if (rc != CUDA_SUCCESS)
	{
		dlist_push_tail(&resource_tracker_free, &tracker->chain);
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
			return (CUdeviceptr)0UL;
		elog(ERROR, "failed on cuMemAlloc: %s", errorText(rc));
	}
	tracker->resclass = RESOURCE_TRACKER_CLASS__GPUMEMORY;
	tracker->u.devptr = devptr;
	index = (hash_any(tracker, offsetof(ResourceTracker, chain))
			 % ResourceTrackerNumSlots);
	dlist_push_tail(&resource_tracker_slots[index], &tracker->chain);

	return devptr;
}

void
GpuMemFree_v2(CUdeviceptr devptr)
{
	ResourceTracker	temp;
	dlist_iter		iter;

	memset(&temp, 0, sizeof(ResourceTracker));
	temp.resclass = RESOURCE_TRACKER_CLASS__GPUMEMORY;
	temp.resource = (Datum) devptr;
	index = (hash_any(&temp, offsetof(ResourceTracker, chain))
			 % ResourceTrackerNumSlots);

	dlist_foreach(iter, &resource_tracker_slots[index])
	{
		ResourceTracker *tracker = (ResourceTracker *)
			dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->resclass == RESOURCE_TRACKER_CLASS__GPUMEMORY &&
			tracker->u.devptr == devptr)
		{
			dlist_delete(&tracker->chain);
			dlist_push_tail(&resource_tracker_free, &tracker->chain);
			rc = cuMemFree(devptr);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemFree: %s", errorText(rc));
			return;
		}
	}
	elog(ERROR, "device addr %p was not tracked", devptr);
}











/*
 * gpuservRecvCommand
 *
 *
 */
static bool
gpuservRecvCommandTimeout(pgsocket sockfd, GpuServCommand *cmd, long timeout)
{

static int
gpuserv_recv_command(pgsocket sockfd, GpuServCommand *cmd)
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
							   (long)GpuServerCommTimeout);
	if ((retval & WL_POSTMASTER_DEATH) != 0)
		elog(ERROR, "Urgent bailout by crash of postmaster");
	if ((retval & WL_SOCKET_READABLE) == 0)
		return false;	/* something happen, caller must check */

	/* fetch a message from the socket */
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
		if (IsGpuServerProcess())
			trackFileDesc(peer_fd);
		else
			elog(FATAL, "Bug? backend never receive peer FD");
	}

	if (retval == 0 || msg.msg_iovlen == 0)
	{
		cmd->command = GPUSERV_CMD_CLOSE;
		cmd->peer_fd = peer_fd;
		elog(LOG, "no bytes received, likely connection closed");
		return false;
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
	return true;
}



}

bool
gpuservRecvCommand(pgsocket sockfd, GpuServCommand *cmd)
{
	return gpuservRecvCommandTimeout(sockfd, cmd, GpuServerCommTimeout);
}

/*
 * gpuservSendCommand
 *
 * send a command to the peer side
 */
static bool
gpuservSendCommandTimeout(pgsocket sockfd, GpuServCommand *cmd, long timeout)
{
	struct msghdr	msg;
	struct iovec	iov;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	int				retval;

	retval = WaitLatchOrSocket(MyLatch,
							   WL_LATCH_SET |
							   WL_POSTMASTER_DEATH |
							   WL_SOCKET_WRITABLE |
							   (timeout < 0 ? 0 : WL_TIMEOUT),
							   sockfd,
							   timeout);
	/* Is the socket writable? */
	if ((retval & WL_SOCKET_WRITABLE) == 0)
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
	retval = sendmsg(sockfd, &msg, 0);
	if (retval < 0)
		elog(ERROR, "failed on sendmsg(2), PeerFD=%d: %m",
			 cmd->peer_fd);
	else if (retval == 0)
		elog(ERROR, "no bytes sent using sendmsg(2), PeerFD=%d, %m",
			 cmd->peer_fd);

	return true;
}

bool
gpuservSendCommand(pgsocket sockfd, GpuServCommand *cmd)
{
	return gpuservSendCommandTimeout(sockfd, cmd, GpuServerCommTimeout);
}

/*
 * gpuserv_open_connection - open a unix domain socket from the backend
 * (it may fail if no available GPU server)
 */
pgsocket
gpuservOpenConnection(void)
{
	pgsocket	sockfd;
	int			num_conn_in_progress;
	int			num_waiting_servers;
	int			flags;

	Assert(!IsGpuServerProcess());



	/*
	 * Confirm the number of waiting server process and backend processes
	 * which is in-progress for connections. If we have no chance to make
	 * a connection with servers, give up making a connection.
	 */
	SpinLockAcquire(&gpuServConnState->lock);
	if (gpuServConnState->num_pending_conn >= gpuServConnState->num_wait_servs)
	{
		SpinLockRelease(&gpuServConnState->lock);
		return PGINVALID_SOCKET;
	}
	gpuServConnState->num_pending_conn++;
	SpinLockRelease(&gpuServConnState->lock);

	PG_TRY();
	{
		struct timeval	timeout;

		sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (sockfd < 0)
			elog(ERROR, "failed on socket(2): %m");

		while (connect(sockfd, (struct sockaddr *)&gpuserv_addr,
					   sizeof(gpuserv_addr)) != 0)
		{
			if (errno != EINTR)
				elog(ERROR, "failed on connect(2): %m");
			CHECK_FOR_INTERRUPTS();
		}

		/*
		 * NOTE: A very short timeout to detect 
		 *
		 *
		 *
		 *
		 *
		 *
		 */
		gpuservSendCommantTimeout(sockfd, cmd, 200);


		/*
		 * Revert timeout: note that 
		 *
		 *
		 */


		// setup timeout for message passing

		if (GpuServerCommTimeout < 0)

		timeout.tv_sec = 0;



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

	return sockfd;
	



	// setup timeout very short (0.2sec)



	// wait for acknowledge

	// revert the timeout





retry:

		if (errno == EAGAIN || errno == ECONNREFUSED)
		{
			close(sockfd);
			return PGINVALID_SOCKET;
		}
		close(sockfd);
		elog(ERROR, "failed on connect(\"%s\") : %m", gpuserv_addr.sun_path);
	}
	return sockfd;
}

/*
 * gpuserv_accept_connection - accept a client connection
 */
static pgsocket
__gpuservAcceptConnection(WaitEventSet *wset)
{
	long			timeout = 400;	/* 400msec sufficient for timeout */
	WaitEvent		event;
	pgsocket		sockfd;
	int				retval;

	ResetLatch(MyLatch);
	retval = WaitEventSetWait(wset, timeout, &event, 1);
	if (retval == 0)
		return PGINVALID_SOCKET;	/* timeout */
	else if ((event.events & (WL_LATCH_SET |
							  WL_POSTMASTER_DEATH)) != 0)
		return PGINVALID_SOCKET;	/* something urgent we have to care */







	/* accept a connection */
	sockfd = accept(gpu_server_sock, NULL, NULL);
	if (sockfd < 0)
		return PGINVALID_SOCKET;	/* something happen */







	struct pollfd	pollbuf;
	pgsocket		sockfd;
	GpuServCommand	cmd;
	SharedGpuContext *shgcon;
	int				retval;

	ResetLatch(MyLatch);
	while ((sockfd = accept(gpu_server_sock, NULL, NULL)) < 0)
	{
		if (MyLatch->is_set || gpuServGotSignal != 0)
			return PGINVALID_SOCKET; /* something happen, caller must check */

		CHECK_FOR_INTERRUPTS();
		if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
			elog(ERROR, "accept(2) failed: %m");
	}
	trackFileDesc(sockfd);

	/*
	 * Each backend shall send OPEN command at first
	 */
	while (!gpuserv_recv_command(sockfd, &cmd))
	{
		if (MyLatch->is_set || gpuServGotSignal != 0)
		{
			closeFileDesc(sockfd);
			return PGINVALID_SOCKET; /* something happen, caller must check */
		}
		CHECK_FOR_INTERRUPTS();
	}

	if (cmd.command != GPUSERV_CMD_OPEN)
		elog(ERROR, "GpuServ: 1st command was not GPUSERV_CMD_OPEN");
	shgcon = AttachGpuContext(cmd.context_id, cmd.backend_id);

	return sockfd;
}

static pgsocket
gpuservAcceptConnection(WaitEventSet *wset)
{
	pgsocket	sockfd;

	/* Server is not waiting */
	SpinLockAcquire(&gpuServConnState->lock);
	gpuServConnState->num_waiting_servers++;
	SpinLockRelease(&gpuServConnState->lock);

	PG_TRY();
	{
		do {
			sockfd = __gpuservAcceptConnection(wset);
			CHECK_FOR_INTERRUPTS();
		} while (sockfd == PGINVALID_SOCKET && !gpuServGotSignal);
	}
	PG_CATCH();
	{
		/* Server break for waiting */
		SpinLockAcquire(&gpuServConnState->lock);
		gpuServConnState->num_waiting_servers--;
		SpinLockRelease(&gpuServConnState->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Server break for waiting */
	SpinLockAcquire(&gpuServConnState->lock);
	gpuServConnState->num_waiting_servers--;
	SpinLockRelease(&gpuServConnState->lock);

	return sockfd;
}

/*
 * SIGTERM handler
 */
static void
gpuserv_got_sigterm(SIGNAL_ARGS)
{
	GpuServGotSignal = SIGTERM;
}

/*
 * gpuserv_session_main - it processes a session once established
 */
static void
gpuserv_session_main(pgsocket sockfd)
{
	GpuServCommand		cmd;

	while (gpuServGotSignal == 0)
	{
		// TODO: send back ready tasks to the backend

		CHECK_FOR_INTERRUPTS();
		if (!gpuserv_recv_command(sockfd, &cmd))
			continue;

		switch (cmd.command)
		{
			case GPUSERV_CMD_CLOSE:
				return;		/* EOF detection, close the session */

			default:
				elog(ERROR, "unexpected GpuServ command: %d", cmd.command);
				break;
		}
	}
}

/*
 * pgstrom_cuda_serv_main - entrypoint of the CUDA server process
 */
static void
gpuserv_main(Datum __server_id)
{
	cl_int		server_id = DatumGetUint32(__server_id);
	cl_int		device_id;
	pgsocket	sockfd;
	WaitEventSet *wset;
	CUresult	rc;

	/* I am a GPU server process */
	gpu_server_process = true;
	pqsignal(SIGTERM, gpuserv_got_sigterm);
	BackgroundWorkerUnblockSignals();

	/* Init CUDA runtime */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	device_id = devAttrs[server_id % numDevAttrs].DEV_ID;
	rc = cuDeviceGet(&cuda_device, device_id);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

	/* see the comments in WaitLatchOrSocket() */
	wset = WaitEventSet(TopMemoryContext, 3);
	AddWaitEventToSet(wset, WL_LATCH_SET, PGINVALID_SOCKET, MyLatch, NULL);
	AddWaitEventToSet(wset, WL_POSTMASTER_DEATH, PGINVALID_SOCKET, NULL, NULL);
	AddWaitEventToSet(wset, WL_SOCKET_READABLE, gpu_server_sock, NULL, NULL);

	/* memory context per session duration */
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "GPU Server per Session",
												 ALLOCSET_DEFAULT_MINSIZE,
												 ALLOCSET_DEFAULT_INITSIZE,
												 ALLOCSET_DEFAULT_MAXSIZE);
	PG_TRY();
	{
		while (!gpuServGotSignal)
		{
			sockfd = gpuservAcceptConnection(wset);
			if (sockfd == PGINVALID_SOCKET)
				break;

			gpuservSessionMain(sockfd);
			closeFileDesc(sockfd);

			// clean up orphan resources
			MemoryContextReset(CurrentMemoryContext);
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
	PG_ENT_TRY();

	/* detach shared resources if any */
	if (currentSharedGpuContext)
		PutSharedGpuContext(currentSharedGpuContext);
}

/*
 * pgstrom_startup_gpu_server
 */
static void
pgstrom_startup_gpu_server(void)
{
	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	gpuServConnState = ShmemInitStruct("gpuServConnState",
									   sizeof(GpuServConnState), &found);
	Assert(!found);
	memset(gpuServConnState, 0, sizeof(GpuServConnState));
	SpinLockInit(&gpuServConnState->lock);
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
	}
	RegisterBackgroundWorker(&worker);

	/* request for the static shared memory */
	RequestAddinShmemSpace(sizeof(GpuServConnState));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_server;
}
