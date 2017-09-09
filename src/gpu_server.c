/*
 * gpu_server.c
 *
 * Routines of GPU/CUDA intermediation server
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "access/twophase.h"
#include "port.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"
#include "cuda_dynpara.h"

#include <pthread.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

/*
 * GpuServProcState - per process state of GPU server
 */
typedef struct GpuServProcState
{
	PGPROC		   *server_pgproc;			/* reference to PGPROC */
	pg_atomic_uint32 num_server_gpu_tasks;	/* sum of pending + running +
											 * completed tasks */
} GpuServProcState;

/*
 * GpuServState - global state of GPU server
 */
typedef struct GpuServState
{
	pg_atomic_uint32 rr_count;			/* seed for round-robin distribution */
	slock_t			lock;				/* lock of conn_pending */
	dlist_head		conn_pending_list;	/* list of GpuServConn */
	GpuServProcState sv_procs[FLEXIBLE_ARRAY_MEMBER];
} GpuServState;

/*
 * GpuServConn - credential information when backend opens a new connection
 */
typedef struct GpuServConn
{
	dlist_node		chain;
	PGPROC		   *client;
	SharedGpuContext *shgcon;
} GpuServConn;

/*
 * GpuServCommand - Token of request for GPU server
 */
#define GPUSERV_CMD_TASK			0x102
#define GPUSERV_CMD_ERROR			0x103
#define GPUSERV_CMD_GPU_MEMFREE		0x110
#define GPUSERV_CMD_IOMAP_MEMFREE	0x111

typedef struct GpuServCommand
{
	cl_uint		command;	/* one of the GPUSERV_CMD_* */
	cl_uint		length;		/* length of the command */
	union {
		GpuTask	   *gtask;	/* TASK or ERROR */
		CUdeviceptr	devptr;	/* (GPU|IOMAP)_MEMFREE */
	};
	/* above fields are common to any message type */
	struct {
		cl_int	elevel;
		cl_int	sqlerrcode;
		cl_uint	filename_offset;
		cl_int	lineno;
		cl_uint	funcname_offset;
		cl_uint	message_offset;
		cl_int	caused_by_myself;
		char	buffer[FLEXIBLE_ARRAY_MEMBER];
	} error;
} GpuServCommand;

/*
 * static/public variables
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static struct sockaddr_un *gpuServSockAddr;		/* const */
static GpuServState	   *gpuServState = NULL;	/* shmem */
static GpuServConn	   *gpuServConn = NULL;		/* shmem */
static int				GpuServerMinNWorkers;	/* GUC */
static int				GpuServerCommTimeout;	/* GUC */
static volatile bool	gpuserv_got_sigterm = false;
static int				gpuserv_id = -1;
static pgsocket			gpuserv_server_sockfd = PGINVALID_SOCKET;
static int				gpuserv_epoll_fd = -1;
static int				gpuserv_event_fd = -1;
/* common CUDA resources */
int						gpuserv_cuda_dindex = -1;
CUdevice				gpuserv_cuda_device = NULL;
CUcontext				gpuserv_cuda_context = NULL;
__thread CUevent		gpuserv_cuda_event = NULL;

#define CUDA_MODULE_CACHE_NSLOTS	200
static dlist_head		gpuserv_cuda_module_cache[CUDA_MODULE_CACHE_NSLOTS];
static slock_t			gpuserv_cuda_module_lock;
/* GPU server session info */
static slock_t			session_tasks_lock;
static dlist_head		session_pending_tasks;
static dlist_head		session_completed_tasks;
static pg_atomic_uint32	session_num_clients;
static pg_atomic_uint32	gpuserv_worker_npolls;
static pg_atomic_uint32	gpuserv_worker_nthreads;
__thread bool			gpuserv_is_worker_context = false;
__thread sigjmp_buf	   *gpuserv_worker_exception_stack = NULL;

/*
 * static functions
 */
static void gpuserv_wakeup_workers(void);
static void gpuservSendCommand(GpuContext *gcontext, GpuServCommand *cmd);

/* SIGTERM handler */
static void
gpuservSigtermHandler(SIGNAL_ARGS)
{
	int		save_errno = errno;

	gpuserv_got_sigterm = true;

	pg_memory_barrier();

	SetLatch(MyLatch);

	errno = save_errno;
}

bool
gpuservGotSigterm(void)
{
	Assert(IsGpuServerProcess());
	pg_memory_barrier();
	return gpuserv_got_sigterm;
}

/*
 * gpuservOnExitCleanup - remove UNIX domain socket on shutdown of
 * the postmaster process.
 */
static void
gpuservOnExitCleanup(int code, Datum arg)
{
	if (IsGpuServerProcess())
	{
		const char *sockpath = gpuServSockAddr[gpuserv_id].sun_path;

		if (unlink(sockpath) != 0 && errno != ENOENT)
			elog(LOG, "failed on unlink('%s'): %m", sockpath);
	}
}

/*
 * wlog - an alternative of elog() in GPU server worker context
 */
static struct
{
	slock_t		lock;
	pthread_t	thread;
	int			elevel;
	const char *filename;
	int			lineno;
	const char *funcname;
	char		message[WORKER_ERROR_MESSAGE_MAXLEN];
} worker_exception_data;

void
worker_error(const char *funcname,
			 const char *filename, int lineno,
			 const char *fmt, ...)
{
	va_list		va_args;
	char		__fmt[2048];

	snprintf(__fmt, sizeof(__fmt), "ERROR: (%s:%d) %s\n",
			 filename, lineno, fmt);
	/* output log to stderr */
	va_start(va_args, fmt);
	vfprintf(stderr, __fmt, va_args);
	
	SpinLockAcquire(&worker_exception_data.lock);
	if (worker_exception_data.elevel == 0)
	{
		worker_exception_data.thread	= pthread_self();
		worker_exception_data.elevel	= ERROR;
		worker_exception_data.filename	= filename;
		worker_exception_data.lineno	= lineno;
		worker_exception_data.funcname	= funcname;
		snprintf(worker_exception_data.message,
				 sizeof(worker_exception_data.message),
				 fmt, va_args);
	}
	SpinLockRelease(&worker_exception_data.lock);
	va_end(va_args);

	Assert(gpuserv_worker_exception_stack != NULL);
	siglongjmp(*gpuserv_worker_exception_stack, 1);
}

/*
 * IsGpuServerProcess
 *
 * zero : current process is backend
 * negative: current process is GPU server with multi-threading
 * positive: current process is GPU server without multi-threading
 */
int
IsGpuServerProcess(void)
{
	if (gpuserv_id < 0)
		return 0;
	else if (!gpuserv_is_worker_context)
		return 1;
	else
		return -1;
}

/*
 * (Inc|Dec|Get)NumberOfGpuServerTasks
 */
static inline void
IncNumberOfGpuServerTasks(void)
{
	pg_atomic_uint32   *p_count;
	uint32				oldval __attribute__((unused));

	Assert(gpuserv_id >= 0 && gpuserv_id < numDevAttrs);
	p_count = &gpuServState->sv_procs[gpuserv_id].num_server_gpu_tasks;
	oldval = pg_atomic_fetch_add_u32(p_count, 1);
	Assert(oldval >= 0);
}

static inline void
DecNumberOfGpuServerTasks(void)
{
	pg_atomic_uint32   *p_count;
	uint32				oldval __attribute__((unused));

	Assert(gpuserv_id >= 0 && gpuserv_id < numDevAttrs);
	p_count = &gpuServState->sv_procs[gpuserv_id].num_server_gpu_tasks;
	oldval = pg_atomic_fetch_sub_u32(p_count, 1);
	Assert(oldval > 0);
}

uint32
GetNumberOfGpuServerTasks(int server_id)
{
	pg_atomic_uint32   *p_count;

	Assert(server_id >= 0 && server_id < numDevAttrs);
	p_count = &gpuServState->sv_procs[server_id].num_server_gpu_tasks;
	return pg_atomic_read_u32(p_count);
}

/*
 * lock/unlock_per_data_socket - serializer of socket access
 */
#define NUM_DATA_SOCKET_LOCKS	1024
static slock_t		per_data_socket_locks[NUM_DATA_SOCKET_LOCKS];

static inline void
lock_per_data_socket(pgsocket sockfd)
{
	int		index = (int)sockfd / NUM_DATA_SOCKET_LOCKS;

	SpinLockAcquire(&per_data_socket_locks[index]);
}

static inline void
unlock_per_data_socket(pgsocket sockfd)
{
	int		index = (int)sockfd / NUM_DATA_SOCKET_LOCKS;

	SpinLockRelease(&per_data_socket_locks[index]);
}

/*
 * lookupGpuModuleCache
 */
typedef struct GpuModuleCache
{
	dlist_node		chain;
	cl_int			refcnt;
	ProgramId		program_id;
	CUmodule		cuda_module;
} GpuModuleCache;

static GpuModuleCache *
lookupGpuModuleCache(ProgramId program_id)
{
	dlist_iter	iter;
	int			index;
	CUmodule	cuda_module;
	CUresult	rc;
	GpuModuleCache *gmod_cache;

	/* Is the cuda module already loaded? */
	index = program_id % CUDA_MODULE_CACHE_NSLOTS;
	SpinLockAcquire(&gpuserv_cuda_module_lock);
	dlist_foreach (iter, &gpuserv_cuda_module_cache[index])
	{
		gmod_cache = dlist_container(GpuModuleCache, chain, iter.cur);

		if (gmod_cache->program_id == program_id)
		{
			gmod_cache->refcnt++;
			SpinLockRelease(&gpuserv_cuda_module_lock);
			return gmod_cache;
		}
	}
	SpinLockRelease(&gpuserv_cuda_module_lock);

	/*
	 * NOTE: Pay attention pgstrom_load_cuda_program() can block until
	 * completion of the GPU program build, if nobody picked up the program
	 * entry yet, even if timeout == 0.
	 * In this case, second or later threads return immediately with NULL;
	 * which means NVRTC works in progress on the program entry, thus,
	 * worker needs to release this task then handle another tasks.
	 *
	 * Only the first victim thread contributes to build the program entry,
	 * and construct GpuModuleCache here.
	 */
	cuda_module = pgstrom_load_cuda_program(program_id);
	if (!cuda_module)
		return NULL;
	
	SpinLockAcquire(&gpuserv_cuda_module_lock);
	dlist_foreach (iter, &gpuserv_cuda_module_cache[index])
	{
		gmod_cache = dlist_container(GpuModuleCache, chain, iter.cur);
		/*
		 * Even if this thread builds the program entry, concurrent worker
		 * might already build a GpuModuleCache here.
		 */
		if (gmod_cache->program_id == program_id)
		{
			gmod_cache->refcnt++;
			SpinLockRelease(&gpuserv_cuda_module_lock);
			rc = cuModuleUnload(cuda_module);
			if (rc != CUDA_SUCCESS)
				wnotice("failed on cuModuleUnload: %s", errorText(rc));
			return gmod_cache;
		}
	}
	gmod_cache = malloc(sizeof(GpuModuleCache));
	if (!gmod_cache)
	{
		SpinLockRelease(&gpuserv_cuda_module_lock);
		werror("out of memory");
	}
	gmod_cache->refcnt = 2;
	gmod_cache->program_id = program_id;
	gmod_cache->cuda_module = cuda_module;

	dlist_push_head(&gpuserv_cuda_module_cache[index], &gmod_cache->chain);
	SpinLockRelease(&gpuserv_cuda_module_lock);

	return gmod_cache;
}

static void
putGpuModuleCacheNoLock(GpuModuleCache *gmod_cache)
{
	Assert(gmod_cache->refcnt > 0);
	if (--gmod_cache->refcnt == 0)
	{
		CUresult	rc;

		if (gmod_cache->chain.prev || gmod_cache->chain.next)
			dlist_delete(&gmod_cache->chain);
		rc = cuModuleUnload(gmod_cache->cuda_module);
		if (rc != CUDA_SUCCESS)
			wnotice("failed on cuModuleUnload: %s",
					errorText(rc));
		free(gmod_cache);
	}
}

static inline void
putGpuModuleCache(GpuModuleCache *gmod_cache)
{
	SpinLockAcquire(&gpuserv_cuda_module_lock);
	putGpuModuleCacheNoLock(gmod_cache);
	SpinLockRelease(&gpuserv_cuda_module_lock);
}

static void
cleanupGpuModuleCache(void)
{
	GpuModuleCache *gmod_cache;
	dlist_node	   *dnode;
	cl_int			i;

	SpinLockAcquire(&gpuserv_cuda_module_lock);
	for (i=0; i < CUDA_MODULE_CACHE_NSLOTS; i++)
	{
		while (!dlist_is_empty(&gpuserv_cuda_module_cache[i]))
		{
			dnode = dlist_pop_head_node(&gpuserv_cuda_module_cache[i]);
			gmod_cache = dlist_container(GpuModuleCache, chain, dnode);
			memset(&gmod_cache->chain, 0, sizeof(dlist_node));

			putGpuModuleCacheNoLock(gmod_cache);
		}
	}
	SpinLockRelease(&gpuserv_cuda_module_lock);
}

/*
 * ReportErrorForBackend - it shall be called on the error handler.
 * It send back a error command to the backend immediately.
 */
static void
ReportErrorForBackend(GpuContext *gcontext)
{
	GpuServCommand *cmd;
	size_t			required;
	size_t			offset = 0;
	char			buffer[sizeof(worker_exception_data.message)];

	SpinLockAcquire(&worker_exception_data.lock);
	required = (offsetof(GpuServCommand, error.buffer) +
				MAXALIGN(strlen(worker_exception_data.filename) + 1) +
				MAXALIGN(strlen(worker_exception_data.funcname) + 1) +
				MAXALIGN(strlen(worker_exception_data.message) + 1));
	cmd = (GpuServCommand *)buffer;
	cmd->command            = GPUSERV_CMD_ERROR;
	cmd->length             = required;
	cmd->gtask				= NULL;
	cmd->error.elevel		= worker_exception_data.elevel;
	cmd->error.sqlerrcode	= ERRCODE_INTERNAL_ERROR;	/* tentative */

	cmd->error.filename_offset = offset;
	strcpy(cmd->error.buffer + offset, worker_exception_data.filename);
	offset += MAXALIGN(strlen(worker_exception_data.filename) + 1);

	cmd->error.lineno		= worker_exception_data.lineno;

	cmd->error.funcname_offset = offset;
	strcpy(cmd->error.buffer + offset, worker_exception_data.funcname);
	offset += MAXALIGN(strlen(worker_exception_data.funcname) + 1);

	cmd->error.message_offset = offset;
	strcpy(cmd->error.buffer + offset, worker_exception_data.message);
	offset += MAXALIGN(strlen(worker_exception_data.message) + 1);

	cmd->error.caused_by_myself = pthread_equal(worker_exception_data.thread,
												pthread_self());
	SpinLockRelease(&worker_exception_data.lock);

	/* Urgent return to the backend */
	gpuservSendCommand(gcontext, cmd);
}

/*
 * gpuservClenupGpuContext
 *
 * Extra cleanup stuff if refcnt of GpuContext reached zero.
 */
void
gpuservClenupGpuContext(GpuContext *gcontext)
{
	uint32		expected = 0;

	/* remove data socket from the epoll_fd */
	if (pg_atomic_compare_exchange_u32(&gcontext->is_unlinked,
									   &expected, 1))
	{
		if (epoll_ctl(gpuserv_epoll_fd,
					  EPOLL_CTL_DEL,
					  gcontext->sockfd, NULL) < 0)
			wnotice("failed on epoll_ctl(EPOLL_CTL_DEL): %m");
	}
	/* unload CUDA modules if no concurrent sessions */
	if (pg_atomic_sub_fetch_u32(&session_num_clients, 1) == 0)
		cleanupGpuModuleCache();
}

/*
 * gpuservTryToWakeUp - wakes up a GPU server process
 */
void
gpuservTryToWakeUp(void)
{
	PGPROC	   *server_pgproc;
	uint32		k, i;

	k = i = pg_atomic_fetch_add_u32(&gpuServState->rr_count, 1);
	do {
		server_pgproc = gpuServState->sv_procs[k % numDevAttrs].server_pgproc;
		if (server_pgproc)
		{
			SetLatch(&server_pgproc->procLatch);
			break;
		}
		k = pg_atomic_fetch_add_u32(&gpuServState->rr_count, 1);
	} while (k % numDevAttrs != i % numDevAttrs);
}

/*
 * gpuservSendCommand - an internal low-level interface
 */
static void
gpuservSendCommand(GpuContext *gcontext, GpuServCommand *cmd)
{
	struct msghdr	msg;
	struct iovec    iov;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	ssize_t			retval;

	Assert(cmd->command == GPUSERV_CMD_TASK ||
		   cmd->command == GPUSERV_CMD_ERROR ||
		   cmd->command == GPUSERV_CMD_GPU_MEMFREE ||
		   cmd->command == GPUSERV_CMD_IOMAP_MEMFREE);

	memset(&msg, 0, sizeof(struct msghdr));
	memset(&iov, 0, sizeof(iov));

	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	iov.iov_base = cmd;
	iov.iov_len = cmd->length;

	/* Is a file-descriptor of the backend delivered to the server? */
	if (!IsGpuServerProcess() &&
		cmd->command == GPUSERV_CMD_TASK &&
		cmd->gtask->file_desc >= 0)
	{
		struct cmsghdr *cmsg;

		msg.msg_control = cmsgbuf;
		msg.msg_controllen = sizeof(cmsgbuf);

		cmsg = CMSG_FIRSTHDR(&msg);
		cmsg->cmsg_level = SOL_SOCKET;
		cmsg->cmsg_type = SCM_RIGHTS;
		cmsg->cmsg_len = CMSG_LEN(sizeof(int));
		((int *)CMSG_DATA(cmsg))[0] = cmd->gtask->file_desc;
	}
	retval = sendmsg(gcontext->sockfd, &msg, MSG_DONTWAIT);
	if (retval < 0)
		werror("failed on sendmsg(2): (sockfd=%d) %m", gcontext->sockfd);
	else if (retval == 0)
		werror("no bytes sent using sengmsg(2): %m");
	else if (retval != cmd->length)
		werror("incomplete size of message sent: %zu of %zu",
			   retval, (size_t)cmd->length);
}

/*
 * gpuservSendGpuTask - enqueue a GpuTask to the socket
 */
void
gpuservSendGpuTask(GpuContext *gcontext, GpuTask *gtask)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	GpuServCommand	cmd;

	if (IsGpuServerProcess())
	{
		SpinLockAcquire(&shgcon->lock);
		shgcon->num_async_tasks--;
		SpinLockRelease(&shgcon->lock);
	}
	cmd.command = GPUSERV_CMD_TASK;
    cmd.length = offsetof(GpuServCommand, error);
	cmd.gtask = gtask;
	gpuservSendCommand(gcontext, &cmd);

	if (!IsGpuServerProcess())
	{
		SpinLockAcquire(&shgcon->lock);
		shgcon->num_async_tasks++;
		SpinLockRelease(&shgcon->lock);
	}
}

/*
 * gpuservSendGpuMemFree
 */
void
gpuservSendGpuMemFree(GpuContext *gcontext, CUdeviceptr devptr)
{
	GpuServCommand	cmd;

	cmd.command = GPUSERV_CMD_GPU_MEMFREE;
	cmd.length = offsetof(GpuServCommand, error);
	cmd.devptr = devptr;
	gpuservSendCommand(gcontext, &cmd);
}

/*
 * gpuservSendIOMapMemFree
 */
void
gpuservSendIOMapMemFree(GpuContext *gcontext, CUdeviceptr devptr)
{
	GpuServCommand	cmd;

	cmd.command = GPUSERV_CMD_IOMAP_MEMFREE;
	cmd.length = offsetof(GpuServCommand, error);
	cmd.devptr = devptr;
	gpuservSendCommand(gcontext, &cmd);
}

/*
 * gpuservOpenConnection - open a unix domain socket from the backend
 * (it may fail if no available GPU server)
 */
void
gpuservOpenConnection(GpuContext *gcontext)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	GpuServConn	   *gs_conn;
	pgsocket		sockfd = PGINVALID_SOCKET;
	cl_int			dindex;
	long			timeout = 30000;	/* 30s */
	struct timeval	tv1, tv2;

	Assert(!IsGpuServerProcess());
	Assert(gcontext->sockfd == PGINVALID_SOCKET);
	/* Setup GpuServConn of the current context */
	gs_conn = &gpuServConn[MyProc->pgprocno];
	Assert(!gs_conn->chain.prev && !gs_conn->chain.next &&
		   !gs_conn->client && !gs_conn->shgcon);
	gs_conn->client = MyProc;
	gs_conn->shgcon = shgcon;
	SpinLockAcquire(&gpuServState->lock);
	dlist_push_tail(&gpuServState->conn_pending_list, &gs_conn->chain);
	SpinLockRelease(&gpuServState->lock);

	dindex = pg_atomic_fetch_add_u32(&gpuServState->rr_count, 1) % numDevAttrs;
	gettimeofday(&tv1, NULL);
	PG_TRY();
	{
		/*
		 * Open the connection to GPU server
		 */
		sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (sockfd < 0)
			elog(ERROR, "failed on socket(2): %m");

		for (;;)
		{
			CHECK_FOR_INTERRUPTS();

			if (connect(sockfd,
						(struct sockaddr *)&gpuServSockAddr[dindex],
						sizeof(struct sockaddr_un)) == 0)
				break;

			if (errno != EINTR)
				elog(ERROR, "failed on connect(2): %m");

			gettimeofday(&tv2, NULL);
			timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
						(tv1.tv_sec * 1000 + tv2.tv_usec / 1000));
			tv1 = tv2;
			if (timeout < 0)
				elog(ERROR, "timeout on connect(2)");
		}

		/*
		 * Wait until GPU server confirmed the connection
		 */
		for (;;)
		{
			int		ev;

			CHECK_FOR_INTERRUPTS();

			ResetLatch(MyLatch);

			if (shgcon->server)
				break;

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   timeout);
			if (ev & WL_POSTMASTER_DEATH)
				ereport(FATAL,
						(errcode(ERRCODE_ADMIN_SHUTDOWN),
						 errmsg("Urgent termination by postmaster dead")));
			/* timeout? */
			gettimeofday(&tv2, NULL);
			timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
						(tv1.tv_sec * 1000 + tv2.tv_usec / 1000));
			tv1 = tv2;
			if (timeout < 0)
				elog(ERROR, "timeout on connection establishment");
		}
		elog(DEBUG2, "connect socket %d to GPU server %d", sockfd, gpuserv_id);
		gcontext->gpuserv_id = dindex;
		gcontext->sockfd = sockfd;
	}
	PG_CATCH();
	{
		SpinLockAcquire(&gpuServState->lock);
		if (gs_conn->chain.prev || gs_conn->chain.next)
			dlist_delete(&gs_conn->chain);
		SpinLockRelease(&gpuServState->lock);
		memset(gs_conn, 0, sizeof(GpuServConn));
		/* also close the socket */
		if (sockfd >= 0)
			close(sockfd);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * gpuservAcceptConnection - accept a new client connection
 */
static void
gpuservAcceptConnection(void)
{
	pgsocket		sockfd = PGINVALID_SOCKET;
	struct ucred	peer_cred;
	socklen_t		so_len = sizeof(peer_cred);
	dlist_mutable_iter iter;
	PGPROC		   *client = NULL;
	SharedGpuContext *shgcon = NULL;


	Assert(IsGpuServerProcess());
	sockfd = accept(gpuserv_server_sockfd, NULL, NULL);
	if (sockfd < 0)
	{
		if (errno == EAGAIN || errno == EWOULDBLOCK)
			return;		/* no more pending connections */
		werror("failed on accept(2): %m");
	}

	if (getsockopt(sockfd, SOL_SOCKET, SO_PEERCRED,
				   &peer_cred, &so_len) != 0 ||
		so_len != sizeof(peer_cred))
	{
		close(sockfd);
		werror("failed on getsockopt(SO_PEERCRED): %m");
	}

	/* Try to lookup the connection pending backend */
	SpinLockAcquire(&gpuServState->lock);
	dlist_foreach_modify(iter, &gpuServState->conn_pending_list)
	{
		GpuServConn *gs_conn = dlist_container(GpuServConn, chain, iter.cur);

		/* Bug? storange connection pending client */
		if (!gs_conn->client || !gs_conn->shgcon)
		{
			dlist_delete(&gs_conn->chain);
			memset(gs_conn, 0, sizeof(GpuServConn));
			wnotice("Bug? strange pending client connection (%p %p)",
					gs_conn->client, gs_conn->shgcon);
			continue;
		}

		if (gs_conn->client->pid == peer_cred.pid)
		{
			/* ok, found */
			dlist_delete(&gs_conn->chain);
			client = gs_conn->client;
			shgcon = gs_conn->shgcon;
			memset(gs_conn, 0, sizeof(GpuServConn));
			break;
		}
	}
	SpinLockRelease(&gpuServState->lock);
	if (!shgcon)
	{
		close(sockfd);
		wnotice("pending connection (pid: %u) not found", peer_cred.pid);
		return;
	}

	STROM_TRY();
	{
		/* attach GPU context */
		AttachGpuContext(sockfd, shgcon, gpuserv_epoll_fd);
	}
	STROM_CATCH();
	{
		SpinLockAcquire(&shgcon->lock);
		shgcon->server = (void *)(~0UL);
		SpinLockRelease(&shgcon->lock);

		SetLatch(&client->procLatch);
		STROM_RE_THROW();
	}
	STROM_END_TRY();
	/* wake up the backend process */
	SetLatch(&client->procLatch);
}

/*
 * gputask_pushto_pending_list - push a GpuTask to the pending list with
 * delay (if any).
 */
static inline void
gputask_pushto_pending_list(GpuTask *gtask, unsigned long delay)
{
	dlist_iter		iter;
	struct timeval	tv;

	if (delay == 0)
		memset(&gtask->tv_wakeup, 0, sizeof(struct timeval));
	else
	{
		gettimeofday(&tv, NULL);

		tv.tv_usec += delay;
		if (tv.tv_usec >= 1000000)
		{
			tv.tv_sec += tv.tv_usec / 1000000;
			tv.tv_usec = tv.tv_usec % 1000000;
		}
		gtask->tv_wakeup = tv;
	}

	SpinLockAcquire(&session_tasks_lock);
	if (gtask->chain.prev != NULL || gtask->chain.next != NULL)
		dlist_delete(&gtask->chain);
	dlist_reverse_foreach(iter, &session_pending_tasks)
	{
		GpuTask *curr = dlist_container(GpuTask, chain, iter.cur);

		if (curr->tv_wakeup.tv_sec < gtask->tv_wakeup.tv_sec ||
			(curr->tv_wakeup.tv_sec == gtask->tv_wakeup.tv_sec &&
			 curr->tv_wakeup.tv_usec <= gtask->tv_wakeup.tv_usec))
		{
			dlist_insert_after(&curr->chain, &gtask->chain);
			goto found;
		}
	}
	dlist_push_head(&session_pending_tasks, &gtask->chain);
found:
	SpinLockRelease(&session_tasks_lock);
}

/*
 * gpuservRecvCommands - returns number of the commands received
 */
static int
gpuservRecvCommands(GpuContext *gcontext, bool *p_peer_sock_closed)
{
	struct msghdr	msg;
	struct iovec	iov;
	struct cmsghdr *cmsg;
	char			cmsgbuf[CMSG_SPACE(sizeof(int))];
	char			__cmd[offsetof(GpuServCommand, error)];
	GpuServCommand *cmd;
	int				num_received = 0;
	ssize_t			retval;

	/* socket already closed? */
	if (gcontext->sockfd == PGINVALID_SOCKET)
	{
		wnotice("peer_sock_closed by invalid socket");
		*p_peer_sock_closed = true;
		return 0;
	}

	for (;;)
	{
		int		peer_fdesc = -1;

		memset(&msg, 0, sizeof(msg));
		memset(&iov, 0, sizeof(iov));
		iov.iov_base = __cmd;
		iov.iov_len  = offsetof(GpuServCommand, error);
		msg.msg_iov = &iov;
		msg.msg_iovlen = 1;
		msg.msg_control = cmsgbuf;
		msg.msg_controllen = sizeof(cmsgbuf);

		retval = recvmsg(gcontext->sockfd, &msg, MSG_DONTWAIT);
		if (retval < 0)
		{
			if (errno == EAGAIN || errno == EWOULDBLOCK)
			{
				/* no messages arrived yet */
				break;
			}
			if (errno == ECONNRESET)
			{
				/* peer socket was closed */
				*p_peer_sock_closed = true;
				wnotice("peer_sock_closed by ECONNRESET");
				break;
			}
			werror("failed on recvmsg(2): %m");
		}
		else if (retval == 0)
		{
			/* likely, peer socket was closed */
			*p_peer_sock_closed = true;
			break;
		}

		/* pick up peer file-desc, if delivered */
		if ((cmsg = CMSG_FIRSTHDR(&msg)) != NULL)
		{
			if (!IsGpuServerProcess())
				wfatal("Bug? Only GPU server can receive FD");
			if (cmsg->cmsg_level != SOL_SOCKET ||
				cmsg->cmsg_type != SCM_RIGHTS)
				wfatal("Bug? unexpected cmsghdr");
			if ((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int) > 1)
				wfatal("Bug? two or more FDs delivered at once");
			if (CMSG_NXTHDR(&msg, cmsg) != NULL)
				wfatal("Bug? two or more cmsghdr at once");
			peer_fdesc = ((int *)CMSG_DATA(cmsg))[0];
		}

		cmd = (GpuServCommand *) __cmd;
		if (cmd->command == GPUSERV_CMD_TASK)
		{
			struct timeval	tv __attribute__((unused));
			GpuTask		   *gtask;

			Assert(cmd->length == offsetof(GpuServCommand, error));

			gtask = cmd->gtask;
			Assert(dmaBufferValidatePtr(gtask));
			Assert(!gtask->chain.prev && !gtask->chain.next);
			Assert(!gtask->gcontext);
			Assert(!gtask->tv_wakeup.tv_sec && !gtask->tv_wakeup.tv_usec);
			Assert(gtask->peer_fdesc = -1);

			if (IsGpuServerProcess())
			{
				/* local file_desc must be delivered */
				Assert(gtask->file_desc < 0 || peer_fdesc >= 0);
				/* increment refcnt by GpuTask */
				gtask->gcontext = GetGpuContext(gcontext);
				/* attach peer file-descriptor, if any */
				if (peer_fdesc >= 0)
					gtask->peer_fdesc = peer_fdesc;
				gputask_pushto_pending_list(gtask, 0);
				IncNumberOfGpuServerTasks();
			}
			else
			{
				GpuTaskState	   *gts = gtask->gts;

				if (gts->cb_ready_task)
					gts->cb_ready_task(gts, gtask);
				dlist_push_tail(&gts->ready_tasks, &gtask->chain);
				gts->num_ready_tasks++;
			}
			num_received++;
		}
		else if (cmd->command == GPUSERV_CMD_ERROR)
		{
			char	   *temp = palloc(cmd->length);
			const char *filename;
			const char *funcname;
			const char *message;

			if (IsGpuServerProcess())
				wfatal("Bug? Only GPU server can deliver ERROR");

			/* retrive the error message body */
			memcpy(temp, cmd, offsetof(GpuServCommand, error));
			retval = recv(gcontext->sockfd,
						  temp + offsetof(GpuServCommand, error),
						  cmd->length - offsetof(GpuServCommand, error),
						  MSG_DONTWAIT);
			if (retval < 0)
				elog(ERROR, "failed on recv(2): %m");
			if (retval != cmd->length - offsetof(GpuServCommand, error))
				elog(ERROR, "Bug? error message corruption");
			cmd = (GpuServCommand *) temp;
			filename = cmd->error.buffer + cmd->error.filename_offset;
			funcname = cmd->error.buffer + cmd->error.funcname_offset;
			message  = cmd->error.buffer + cmd->error.message_offset;

			if (errstart(Max(cmd->error.elevel, ERROR),
						 filename,
						 cmd->error.lineno,
						 funcname,
						 TEXTDOMAIN))
				errfinish(errcode(cmd->error.sqlerrcode),
						  errmsg("%s", message));
		}
		else if (cmd->command == GPUSERV_CMD_GPU_MEMFREE)
		{
			CUresult	rc;

			if (!IsGpuServerProcess())
				wfatal("Bug? Only GPU server can handle GPU_MEMFREE");
			rc = gpuMemFree(gcontext, cmd->devptr);
			if (rc != CUDA_SUCCESS)
				wnotice("failed on gpuMemFree with GPU_MEMFREE: %s",
						errorText(rc));
			num_received++;
		}
		else if (cmd->command == GPUSERV_CMD_IOMAP_MEMFREE)
		{
			CUresult	rc;

			if (!IsGpuServerProcess())
				wfatal("Bug? Only GPU server can handle IOMAP_MEMFREE");
			rc = gpuMemFree(gcontext, cmd->devptr);
			if (rc != CUDA_SUCCESS)
				wnotice("failed on gpuMemFree with IOMAP_MEMFREE: %s",
						errorText(rc));
			num_received++;
		}
		else
			wfatal("Bug? unknown GPUSERV_CMD_* tag: %d", cmd->command);
	}
	return num_received;
}

/*
 * gpuservRecvGpuTasks - picks up GpuTasks from the socket
 *
 * NOTE: We expect backend processes calls this function, because GPU server
 * don't takes individual timeout.
 */
bool
gpuservRecvGpuTasks(GpuContext *gcontext, long timeout)
{
	struct timeval	tv1, tv2;
	bool			retval = false;

	if (gcontext->sockfd == PGINVALID_SOCKET)
		return false;

	Assert(!IsGpuServerProcess());

	/* default timeout if negative */
	if (timeout < 0)
		timeout = GpuServerCommTimeout;

	gettimeofday(&tv1, NULL);
	do {
		bool	peer_sock_closed = false;
		int		ev;

		CHECK_FOR_INTERRUPTS();
		ResetLatch(MyLatch);

		if (gpuservRecvCommands(gcontext, &peer_sock_closed) > 0)
		{
			retval = true;
			break;
		}

		if (peer_sock_closed)
		{
			Assert(gcontext->sockfd != PGINVALID_SOCKET);
			if (close(gcontext->sockfd) != 0)
				elog(WARNING, "failed on close(%d) socket: %m",
					 gcontext->sockfd);
			else
				elog(NOTICE, "sockfd=%d closed", gcontext->sockfd);
			gcontext->sockfd = PGINVALID_SOCKET;
			break;
		}

		if (timeout == 0)
			break;

		ev = WaitLatchOrSocket(MyLatch,
							   WL_LATCH_SET |
							   WL_SOCKET_READABLE |
							   (timeout >= 0 ? WL_TIMEOUT : 0) |
							   WL_POSTMASTER_DEATH,
							   gcontext->sockfd,
							   timeout);
		if (ev & WL_POSTMASTER_DEATH)
			ereport(FATAL,
					(errcode(ERRCODE_ADMIN_SHUTDOWN),
					 errmsg("Urgent termination by postmaster dead")));
		if (ev & WL_TIMEOUT)
			elog(ERROR, "GPU server response timeout...");
		if (ev & WL_LATCH_SET)
			break;	/* something happen */

		/* elsewhere wake up by WL_LATCH_SET or WL_SOCKET_READABLE */
		gettimeofday(&tv2, NULL);
		if (timeout > 0)
		{
			timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
						(tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
			tv1 = tv2;
			if (timeout < 0)
				timeout = 0;
		}
	} while (timeout != 0);

	return retval;
}

/*
 * gpuserv_try_run_pending_task
 */
static void
gpuserv_try_run_pending_task(long *p_timeout)
{
	dlist_node	   *dnode;
	GpuTask		   *gtask;
	GpuContext	   *gcontext;
	struct timeval	tv;
	long			delay;

	SpinLockAcquire(&session_tasks_lock);
	if (dlist_is_empty(&session_pending_tasks))
	{
		SpinLockRelease(&session_tasks_lock);
		return;
	}
	dnode = dlist_pop_head_node(&session_pending_tasks);
	gtask = dlist_container(GpuTask, chain, dnode);
	/* check task's delay, if any */
	if (gtask->tv_wakeup.tv_sec != 0 || gtask->tv_wakeup.tv_usec != 0)
	{
		gettimeofday(&tv, NULL);
		if (tv.tv_sec < gtask->tv_wakeup.tv_sec ||
			(tv.tv_sec == gtask->tv_wakeup.tv_sec &&
			 tv.tv_usec < gtask->tv_wakeup.tv_usec))
		{
			delay = ((gtask->tv_wakeup.tv_sec - tv.tv_sec) * 1000 +
					 (gtask->tv_wakeup.tv_usec - tv.tv_usec) / 1000);
			/* it is not a time to run yet! */
			dlist_push_head(&session_pending_tasks, &gtask->chain);
			SpinLockRelease(&session_tasks_lock);

			*p_timeout = Min(*p_timeout, delay);
			return;
		}
		memset(&gtask->tv_wakeup, 0, sizeof(struct timeval));
	}
	memset(&gtask->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&session_tasks_lock);
	/*
	 * XXX
	 */
	gpuserv_wakeup_workers();
#ifdef PGSTROM_DEBUG
	gettimeofday(&tv, NULL);
	gtask->send_delay = (1000000 * tv.tv_sec + tv.tv_usec) -
		(1000000 * gtask->tv_timestamp.tv_sec + gtask->tv_timestamp.tv_usec);
#endif
	gcontext = gtask->gcontext;
	STROM_TRY();
	{
		SharedGpuContext *shgcon = gcontext->shgcon;
		GpuModuleCache *gmod_cache = NULL;
		cl_int			retval = -1;

		/* Skip GpuTask execution if GpuContext is under termination */
		if (pg_atomic_read_u32(&shgcon->in_termination) != 0)
			goto skip_process_task;

		gmod_cache = lookupGpuModuleCache(gtask->program_id);
		if (!gmod_cache)
		{
			/*
			 * When lookupGpuModuleCache() -> pgstrom_load_cuda_program()
			 * returns NULL, it means the required progrem is still under 
			 * run-time build process, so GpuTask has to wait until its
			 * completion for launch.
			 * 150ms delay shall be added for the 
			 */
			gputask_pushto_pending_list(gtask, 150000);
		}
		else
		{
			struct timeval	tv __attribute__((unused));
#ifdef PGSTROM_DEBUG
			gettimeofday(&tv, NULL);
			gtask->kstart_delay = ((1000000 * tv.tv_sec + tv.tv_usec) -
								   (1000000 * gtask->tv_timestamp.tv_sec +
											  gtask->tv_timestamp.tv_usec));
#endif
			/*
			 * pgstromProcessGpuTask() returns the following status:
			 *
			 *  0 : GpuTask is successfully queued to the stream.
			 *      It's now running, and then callback will inform its
			 *      completion later.
			 *  1 : Unable to launch GpuTask due to lack of GPU resource.
			 *      It shall be released later, so, back to the pending
			 *      list again.
			 * -1 : GpuTask handler wants to release GpuTask immediately,
			 *      without message back. So, it shall be released and
			 *      noticed to the backend which may wait for response.
			 */
			retval = pgstromProcessGpuTask(gtask, gmod_cache->cuda_module);
			putGpuModuleCache(gmod_cache);
			if (pg_atomic_read_u32(&shgcon->in_termination) != 0)
				retval = -1;
		skip_process_task:
			if (retval == 0)
			{
				/* send the GpuTask to the backend */
				int		peer_fdesc = gtask->peer_fdesc;

				gtask->gcontext = NULL;
				gtask->peer_fdesc = -1;
				gpuservSendGpuTask(gcontext, gtask);
				DecNumberOfGpuServerTasks();
				if (peer_fdesc >= 0)
					close(peer_fdesc);
				PutGpuContext(gcontext);
			}
			else if (retval > 0)
			{
				long	delay = Max(retval - 1, 0);

				/*
				 * Request for re-execution of GpuTask by out-of-memory
				 * in usual. If retval is larger than 1, it means GpuTask
				 * needs some delay until next launch.
				 */
				gputask_pushto_pending_list(gtask, delay);
			}
			else
			{
				/*
				 * GpuTask wants to release this task without message-back.
				 * So, we have to decrement num_async_tasks and set latch
				 * of the backend process (it may wait for the last task).
				 */
				int			peer_fdesc = gtask->peer_fdesc;
				PGPROC	   *backend;

				Assert(retval < 0);

				pgstromReleaseGpuTask(gtask);

				SpinLockAcquire(&shgcon->lock);
				shgcon->num_async_tasks--;
				backend = shgcon->backend;
				SpinLockRelease(&shgcon->lock);
				DecNumberOfGpuServerTasks();
				if (backend)
					SetLatch(&backend->procLatch);
				if (peer_fdesc >= 0)
					close(peer_fdesc);
				PutGpuContext(gcontext);
			}
		}
	}
	STROM_CATCH();
	{
		ReportErrorForBackend(gcontext);
		STROM_RE_THROW();
	}
	STROM_END_TRY();

	*p_timeout = 0;
}

/*
 * gpuservPushGpuTask - attach a GpuTask to the queue by GPU server itself
 */
void
gpuservPushGpuTask(GpuContext *gcontext, GpuTask *gtask)
{
	SharedGpuContext *shgcon = gcontext->shgcon;

	if (!IsGpuServerProcess())
		elog(FATAL, "Bug? %s is called out of GPU server's context",
			__FUNCTION__);
	SpinLockAcquire(&shgcon->lock);
	shgcon->num_async_tasks++;
	SpinLockRelease(&shgcon->lock);
	IncNumberOfGpuServerTasks();
	/* increment refcnt by GpuTask */
	gtask->gcontext = GetGpuContext(gcontext);
	/* TODO: How to handle peer_fdesc if any? */
	Assert(gtask->peer_fdesc < 0);

	SpinLockAcquire(&session_tasks_lock);
	dlist_push_tail(&session_pending_tasks, &gtask->chain);
	SpinLockRelease(&session_tasks_lock);
}

/*
 * gpuservCompleteGpuTask - A routine for CUDA callback to register GpuTask
 * on the completed list.
 */
void
gpuservCompleteGpuTask(GpuTask *gtask, bool is_urgent)
{
	struct timeval tv __attribute__((unused));
	uint64		count = 1;

	SpinLockAcquire(&session_tasks_lock);
	dlist_delete(&gtask->chain);
	if (is_urgent)
		dlist_push_head(&session_completed_tasks, &gtask->chain);
	else
		dlist_push_tail(&session_completed_tasks, &gtask->chain);
	SpinLockRelease(&session_tasks_lock);
#ifdef PGSTROM_DEBUG
	gettimeofday(&tv, NULL);
	gtask->kfinish_delay = ((1000000 * tv.tv_sec + tv.tv_usec) -
							(1000000 * gtask->tv_timestamp.tv_sec +
									   gtask->tv_timestamp.tv_usec));
#endif
	write(gpuserv_event_fd, &count, sizeof(uint64));
}

/*
 * gpuservWorkerMain
 */
static void
gpuservWorkerMain(void)
{
	bool		read_event_fd = false;

	for (;;)
	{
		struct epoll_event ep_event;
		int			nevents;
		uint32		npolls;
		long		timeout = 15000;	/* 15sec timeout */
		int			__errno;

		if (gpuserv_got_sigterm)
			return;
		/* reset event fd, if waked up by eventfd */
		if (read_event_fd)
		{
			uint64	temp;

			if (read(gpuserv_event_fd, &temp, sizeof(temp)) < 0 &&
				errno != EAGAIN)
				wlog("failed on read(gpuserv_event_fd): %m");
			read_event_fd = false;
		}

		if (pgstrom_try_build_cuda_program())
			timeout = 0;
		gpuserv_try_run_pending_task(&timeout);

		if (gpuserv_got_sigterm)
			return;

		pg_atomic_add_fetch_u32(&gpuserv_worker_npolls, 1);
		nevents = epoll_wait(gpuserv_epoll_fd,
							 &ep_event, 1,
							 timeout);		/* may be non-blocking */
		npolls = pg_atomic_sub_fetch_u32(&gpuserv_worker_npolls, 1);
		__errno = errno;		/* SetLatch overwrite errno... */
		if (npolls < GpuServerMinNWorkers)
			SetLatch(MyLatch);				/* launch workers on demand */

		if (nevents > 0)
		{
			pgsocket		sockfd = ep_event.data.fd;
			uint32_t		events = ep_event.events;
			GpuContext	   *gcontext;

			Assert(nevents == 1);
			if (sockfd == PGINVALID_SOCKET)
			{
				/* event around event fd, just wake up */
				if (events & ~EPOLLIN)
					werror("eventfd reported %08x", events);
				read_event_fd = true;
			}
			else if (sockfd == gpuserv_server_sockfd)
			{
				/* event around server socket */
				if (events & EPOLLIN)
					gpuservAcceptConnection();
				else
					werror("server socket reported %08x", events);
			}
			else if ((gcontext = GetGpuContextBySockfd(sockfd)) != NULL)
			{
				bool		peer_sock_closed = false;
				uint32		expected = 0;

				if (events & EPOLLIN)
					gpuservRecvCommands(gcontext, &peer_sock_closed);
				if (events & ~EPOLLIN)
				{
					wnotice("client socket %d was closed by event %08x",
							sockfd, events);
					peer_sock_closed = true;
				}

				if (peer_sock_closed &&
					pg_atomic_compare_exchange_u32(&gcontext->is_unlinked,
												   &expected, 1))
				{
					Assert(expected == 0);
					/*
					 * Concurrent threads may get 'peer_sock_closed' state
					 * simultitaneously. GpuContext must be unlinked
					 * exactly once per connection, so we use atomic
					 * compare-exchange operation to control the link
					 * status.
					 */
					PutGpuContext(gcontext);

					/* remove data socket from epoll_fd */
					if (epoll_ctl(gpuserv_epoll_fd,
								  EPOLL_CTL_DEL,
								  gcontext->sockfd, NULL) < 0)
						wnotice("failed on epoll_ctl(EPOLL_CTL_DEL): %m");
				}
				PutGpuContext(gcontext);
			}
			else
			{
				wnotice("epoll_wait(2) reported event %08x on socket %d but might be closed already",
						events, sockfd);
			}
		}
		else if (nevents == 0 && timeout > 0)
		{
			/*
			 * Because worker could not receive any incoming event during
			 * valid timeout, we assume GPU server is now relaxing.
			 * Unless number of worker threads are not less than threshold,
			 * worker can exit to release resources.
			 */
			if (npolls >= GpuServerMinNWorkers)
				break;
		}
		else if (nevents < 0 && __errno != EINTR)
			werror("failed on epoll_wait(2): %s", strerror(__errno));
	}
}

static void *
gpuservWorkerEntryPoint(void *__private)
{
	uint32		nthreads;
	CUresult	rc;

	gpuserv_is_worker_context = true;
	nthreads = pg_atomic_add_fetch_u32(&gpuserv_worker_nthreads, 1);
	Assert(nthreads > 0);
	STROM_TRY();
	{
		rc = cuCtxSetCurrent(gpuserv_cuda_context);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuCtxSetCurrent: %s", errorText(rc));

		rc = cuEventCreate(&gpuserv_cuda_event, (CU_EVENT_BLOCKING_SYNC |
												 CU_EVENT_DISABLE_TIMING));
		if (rc != CUDA_SUCCESS)
			werror("failed on cuEventCreate: %s", errorText(rc));

		gpuservWorkerMain();
	}
	STROM_CATCH();
	{
		/*
		 * Suggest primary thread of the background worker to wake up
		 * any other worker threads and terminate them.
		 */
		gpuservSigtermHandler(0);
	}
	STROM_END_TRY();
	nthreads = pg_atomic_fetch_sub_u32(&gpuserv_worker_nthreads, 1);
	Assert(nthreads > 0);
	gpuserv_is_worker_context = false;

	return NULL;
}

/*
 * gpuserv_wakeup_workers
 */
static void
gpuserv_wakeup_workers(void)
{
	uint64		ev_count = 1;

	if (write(gpuserv_event_fd, &ev_count, sizeof(ev_count)) < 0)
	{
		fprintf(stderr, "Bug? could not write on gpuserv_event_fd_one: %m\n");
		proc_exit(1);	/* could not continue any more */
	}
}

/*
 * gpuservEventLoop
 */
static void
gpuservEventLoop(void)
{
	WaitEventSet   *wait_events;
	long			timeout = 5000;		/* wake up per 5 seconds */

	memset(&worker_exception_data, 0, sizeof(worker_exception_data));
	SpinLockInit(&worker_exception_data.lock);
	gpuserv_worker_exception_stack = (void *)(~0UL);

	/*
	 * Setup wait events
	 */
	wait_events = CreateWaitEventSet(TopMemoryContext, 2);
	AddWaitEventToSet(wait_events, WL_LATCH_SET,
					  PGINVALID_SOCKET,
					  (Latch *) MyLatch, NULL);
	AddWaitEventToSet(wait_events, WL_POSTMASTER_DEATH,
					  PGINVALID_SOCKET,
					  NULL, NULL);
	/*
	 * Event loop
	 */
	elog(LOG, "PG-Strom GPU Server is now ready on GPU-%d %s",
		 devAttrs[gpuserv_cuda_dindex].DEV_ID,
		 devAttrs[gpuserv_cuda_dindex].DEV_NAME);

	for (;;)
	{
		WaitEvent	ev;
		int			nevents;
		int			npolls;

		ResetLatch(MyLatch);
		gpuserv_wakeup_workers();
		if (gpuserv_got_sigterm)
			break;

		nevents = WaitEventSetWait(wait_events,
								   timeout,
								   &ev, 1);
		if (nevents > 0)
		{
			Assert(nevents == 1);
			if (ev.events & WL_POSTMASTER_DEATH)
				ereport(FATAL,
						(errcode(ERRCODE_ADMIN_SHUTDOWN),
						 errmsg("Urgent termination by postmaster dead")));
		}
		/* startup worker threads on demand */
		npolls = pg_atomic_fetch_add_u32(&gpuserv_worker_npolls, 1);
		while (npolls < GpuServerMinNWorkers)
		{
			pthread_t	thread;

			if (pg_atomic_read_u32(&gpuserv_worker_nthreads) > pgstrom_max_async_tasks)
				break;

			/* launch a worker thread */
			if ((errno = pthread_create(&thread,
										NULL,
										gpuservWorkerEntryPoint,
										NULL)) != 0)
			{
				if (errno != EAGAIN)
					elog(ERROR, "failed on pthread_create: %m");
				elog(LOG, "failed on pthread_create: %m");
				break;
			}
			if ((errno = pthread_detach(thread)) != 0)
				elog(FATAL, "failed on pthread_detach: %m");
			npolls++;
		}
		pg_atomic_fetch_sub_u32(&gpuserv_worker_npolls, 1);

#if 0
		/*
		 * Try to release free device memory if GPU server is relaxed.
		 */
		if (GetNumberOfGpuServerTasks(gpuserv_id) == 0)
			gpuMemReclaim();
#endif
	}
	elog(LOG, "PG-Strom GPU Server on GPU-%d %s is terminated",
		 devAttrs[gpuserv_cuda_dindex].DEV_ID,
		 devAttrs[gpuserv_cuda_dindex].DEV_NAME);	
}

/*
 * gpuservBgWorkerMain - entrypoint of the CUDA server process
 */
static void
gpuservBgWorkerMain(Datum __server_id)
{
	GpuServProcState *sv_proc;
	CUresult		rc;
	cl_int			i;
	struct epoll_event ep_event;

	/* I am a GPU server process */
	gpuserv_id = DatumGetInt32(__server_id);
	Assert(gpuserv_id >= 0 && gpuserv_id < numDevAttrs);
	sv_proc = &gpuServState->sv_procs[gpuserv_id];
	sv_proc->server_pgproc = MyProc;
	pg_atomic_init_u32(&sv_proc->num_server_gpu_tasks, 0);
	pqsignal(SIGTERM, gpuservSigtermHandler);
	BackgroundWorkerUnblockSignals();

	/* cleanup of UNIX domain socket */
	on_proc_exit(gpuservOnExitCleanup, 0);
	/* memory context per session duration */
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "GPU Server");
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "GPU Server Session",
												 ALLOCSET_DEFAULT_MINSIZE,
												 ALLOCSET_DEFAULT_INITSIZE,
												 ALLOCSET_DEFAULT_MAXSIZE);
	/* init session info */
	SpinLockInit(&session_tasks_lock);
	dlist_init(&session_pending_tasks);
	dlist_init(&session_completed_tasks);
	pg_atomic_init_u32(&session_num_clients, 0);
	pg_atomic_init_u32(&gpuserv_worker_nthreads, 0);
	pg_atomic_init_u32(&gpuserv_worker_npolls, 0);

	/*
	 * Inter threads communications stuff
	 */
	gpuserv_epoll_fd = epoll_create(10);
	if (gpuserv_epoll_fd < 0)
		elog(ERROR, "failed on epoll_create(2): %m");

	gpuserv_event_fd = eventfd(0, EFD_NONBLOCK);
	if (gpuserv_event_fd < 0)
		elog(ERROR, "failed on eventfd(2): %m");
	ep_event.events = EPOLLIN;
	ep_event.data.fd = PGINVALID_SOCKET;	/* identifier of event fd */
	if (epoll_ctl(gpuserv_epoll_fd,
				  EPOLL_CTL_ADD,
				  gpuserv_event_fd,
				  &ep_event) < 0)
		elog(ERROR, "failed on epoll_ctl(2): %m");

	/*
	 * Open unix domain server socket
	 */
	gpuserv_server_sockfd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
	if (gpuserv_server_sockfd < 0)
		elog(ERROR, "failed on socket(AF_UNIX, SOCK_STREAM, 0): %m");

	if (bind(gpuserv_server_sockfd,
			 (struct sockaddr *)&gpuServSockAddr[gpuserv_id],
			 sizeof(struct sockaddr_un)) != 0)
		elog(ERROR, "failed on bind('%s'): %m",
			 gpuServSockAddr[gpuserv_id].sun_path);

	if (listen(gpuserv_server_sockfd, 16) != 0)
		elog(ERROR, "failed on listen(2): %m");

	ep_event.events = EPOLLIN;
	ep_event.data.fd = gpuserv_server_sockfd;
	if (epoll_ctl(gpuserv_epoll_fd,
				  EPOLL_CTL_ADD,
				  gpuserv_server_sockfd,
				  &ep_event) < 0)
		elog(ERROR, "failed on epoll_ctl(2): %m");

	/*
	 * Init CUDA driver APIs stuff
	 */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	gpuserv_cuda_dindex = gpuserv_id;
	rc = cuDeviceGet(&gpuserv_cuda_device,
					 devAttrs[gpuserv_cuda_dindex].DEV_ID);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&gpuserv_cuda_context,
					 CU_CTX_SCHED_AUTO,
					 //CU_CTX_SCHED_BLOCKING_SYNC,
					 gpuserv_cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxCreate: %s", errorText(rc));
	for (i=0; i < CUDA_MODULE_CACHE_NSLOTS; i++)
		dlist_init(&gpuserv_cuda_module_cache[i]);
	SpinLockInit(&gpuserv_cuda_module_lock);
	for (i=0; i < NUM_DATA_SOCKET_LOCKS; i++)
		SpinLockInit(&per_data_socket_locks[i]);

	PG_TRY();
	{
		gpuservEventLoop();
		elog(ERROR, "GPU Server-%d [%s] is terminating normally",
			 gpuserv_id, devAttrs[gpuserv_id].DEV_NAME);
	}
	PG_CATCH();
	{
		/* terminate any active worker threads */
		gpuserv_got_sigterm = true;
		pg_memory_barrier();
		gpuserv_wakeup_workers();

		/*
		 * Destroy the CUDA context to force bailout of long running
		 * GPU kernels; like PL/CUDA's one, regardless of the status
		 * of pending/running tasks.
		 *
		 * NOTE: We should never force to detach SharedGpuContext prior
		 * to destroy of CUDA context, because asynchronous tasks are
		 * still in-progress, thus, it may touch DMA buffer allocated
		 * on the shared memory segment. Once PutSharedGpuContext() is
		 * called, here is no guarantee the shared memory segment is
		 * assigned to the GpuTask. Eventually, it may lead memory
		 * corruption hard to find out.
		 */
		rc = cuCtxDestroy(gpuserv_cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(FATAL, "failed on cuCtxDestroy: %s", errorText(rc));
		gpuserv_cuda_context = NULL;

		/* ensure termination of the worker threads */
		while (pg_atomic_read_u32(&gpuserv_worker_nthreads) > 0)
			pg_usleep(100);
		gpuserv_worker_exception_stack = NULL;

		/* GPU server is terminating... */
		gpuServState->sv_procs[gpuserv_id].server_pgproc = NULL;
		pg_memory_barrier();

		/*
		 * Shared portion of GpuContext has to be detached regardless of
		 * the reference counter of local portion, because the orphan
		 * SharedGpuContext will lead memory leak of the shared DMA buffer
		 * segments.
		 */
		ForcePutAllGpuContext();

		PG_RE_THROW();
	}
	PG_END_TRY();
	elog(FATAL, "Bug? GpuServer has no path to exit normally");
}

/*
 * pgstrom_startup_gpu_server
 */
static void
pgstrom_startup_gpu_server(void)
{
	uint32		numBackends = MaxConnections + max_worker_processes + 100;
	Size		required;
	bool		found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* request for the static shared memory */
	required = offsetof(GpuServState, sv_procs[numDevAttrs]);
	gpuServState = ShmemInitStruct("gpuServState", required, &found);
	Assert(!found);
	memset(gpuServState, 0, required);
	pg_atomic_init_u32(&gpuServState->rr_count, 0);
	SpinLockInit(&gpuServState->lock);
	dlist_init(&gpuServState->conn_pending_list);

	required = sizeof(GpuServConn) * numBackends;
	gpuServConn = ShmemInitStruct("gpuServConn", required, &found);
	Assert(!found);
	memset(gpuServConn, 0, required);
}

/*
 * pgstrom_init_gpu_server
 */
void
pgstrom_init_gpu_server(void)
{
	cl_uint		i, numBackends;

	/*
	 * Minimum number of worker threads which poll new incomng events
	 */
	DefineCustomIntVariable("pg_strom.gpuserv_min_nworkers",
							"minimum number of GPU server worker threads",
							NULL,
							&GpuServerMinNWorkers,
							4,
							2,
							256,
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
	 * Setup pathname of the UNIX domain sockets, for each GPU servers
	 */
	gpuServSockAddr = malloc(sizeof(struct sockaddr_un) * numDevAttrs);
	if (!gpuServSockAddr)
		elog(ERROR, "out of memory");

	for (i=0; i < numDevAttrs; i++)
	{
		struct sockaddr_un *sockaddr = &gpuServSockAddr[i];
		struct stat			stbuf;
		cl_long				suffix;

		suffix = (long) getpid();
		for (;;)
		{
			gpuServSockAddr[i].sun_family = AF_UNIX;
			snprintf(gpuServSockAddr[i].sun_path,
					 sizeof(gpuServSockAddr[i].sun_path),
					 ".pg_strom.gpuserv.sock.%ld.%d", suffix, i);
			if (stat(gpuServSockAddr[i].sun_path, &stbuf) != 0)
			{
				if (errno == ENOENT)
					break;	/* OK */
				elog(ERROR, "pathname '%s' may be strange: %m",
					 gpuServSockAddr[i].sun_path);
			}
			elog(LOG, "pathname '%s' is already in use: %m",
				 sockaddr->sun_path);
			suffix++;
		}
	}

	/*
	 * Launch background worker processes for GPU/CUDA servers
	 */
	for (i=0; i < numDevAttrs; i++)
	{
		BackgroundWorker	worker;

		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "GPU Server-%d [%s]", i, devAttrs[i].DEV_NAME);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = 1;
		worker.bgw_main = gpuservBgWorkerMain;
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* request for the static shared memory */
	numBackends = MaxConnections + max_worker_processes + 100;
	RequestAddinShmemSpace(STROMALIGN(offsetof(GpuServState,
											   sv_procs[numDevAttrs])) +
						   STROMALIGN(sizeof(GpuServConn) * numBackends));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_server;
}

/* ----------------------------------------------------------------
 *
 * Service routines for handlers working on GPU server context
 *
 * ----------------------------------------------------------------
 */
