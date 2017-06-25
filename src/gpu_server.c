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
#include <cudaProfiler.h>

typedef struct GpuServConn
{
	dlist_node		chain;
	PGPROC		   *client;
	SharedGpuContext *shgcon;
} GpuServConn;

typedef struct GpuServState
{
	pg_atomic_uint32 rr_count;			/* seed for round-robin distribution */
	pg_atomic_uint32 cuda_profiler; 	/* GUC: pg_strom.cuda_profiler */
	slock_t			lock;				/* lock of conn_pending */
	dlist_head		conn_pending_list;	/* list of GpuServConn */
	PGPROC		   *serv_procs[FLEXIBLE_ARRAY_MEMBER];
} GpuServState;

/*
 * GpuServCommand - Token of request for GPU server
 */
#define GPUSERV_CMD_TASK		0x102
#define GPUSERV_CMD_ERROR		0x103

typedef struct GpuServCommand
{
	cl_uint		command;	/* one of the GPUSERV_CMD_* */
	cl_uint		length;		/* length of the command */
	GpuTask_v2 *gtask;		/* reference to DMA buffer */
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
static int				GpuServerNumWorkers;	/* GUC */
static int				GpuServerCommTimeout;	/* GUC */
static volatile bool	gpuserv_got_sigterm = false;
static int				gpuserv_id = -1;
static pgsocket			gpuserv_server_sockfd = PGINVALID_SOCKET;
static int				gpuserv_epoll_fd = -1;
static int				gpuserv_event_fd = -1;
static pthread_t	   *gpuworker_threads = NULL;
/* common CUDA resources */
int						gpuserv_cuda_dindex = -1;
CUdevice				gpuserv_cuda_device = NULL;
CUcontext				gpuserv_cuda_context = NULL;
static pg_atomic_uint32	gpuserv_rr_cuda_stream;
static int				gpuserv_num_cuda_stream;
static CUstream		   *gpuserv_cuda_stream;
#define CUDA_MODULE_CACHE_NSLOTS	200
static dlist_head		gpuserv_cuda_module_cache[CUDA_MODULE_CACHE_NSLOTS];
static slock_t			gpuserv_cuda_module_lock;
static int32			gpuserv_cuda_profiler = 0;
static char			   *gpuserv_cuda_profiler_config_file = NULL;
static char			   *gpuserv_cuda_profiler_log_file = NULL;
/* GPU server session info */
static slock_t			session_tasks_lock;
static dlist_head		session_pending_tasks;
static dlist_head		session_running_tasks;
static dlist_head		session_completed_tasks;
static pg_atomic_uint32	session_num_clients;

//HOGE


/*
 * static functions
 */
static void gpuservSendCommand(GpuContext_v2 *gcontext, GpuServCommand *cmd);
static void	pgstrom_cuda_profiler_init(void);
static void	pgstrom_cuda_profiler_update(void);

/* SIGTERM handler */
static void
gpuservGotSigterm(SIGNAL_ARGS)
{
	int		save_errno = errno;

	gpuserv_got_sigterm = true;

	pg_memory_barrier();

	SetLatch(MyLatch);

	errno = save_errno;
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
	char		message[256 * 1024];	/* up to 256KB */
} worker_exception_data;
static __thread sigjmp_buf *worker_exception_stack = NULL;

#define WORKER_TRY() \
	do { \
		sigjmp_buf *save_exception_stack = worker_exception_stack; \
		sigjmp_buf	local_sigjmp_buf; \
		if (sigsetjmp(local_sigjmp_buf, 0) == 0) \
		{ \
			worker_exception_stack = &local_sigjmp_buf

#define WORKER_CATCH() \
		} \
		else \
		{ \
			worker_exception_stack = save_exception_stack

#define WORKER_END_TRY() \
		} \
		worker_exception_stack = save_exception_stack; \
	} while(0)

#define WORKER_RE_THROW()	siglongjmp(*worker_exception_stack, 1)

#define WORKER_CHECK_FOR_INTERRUPTS()			\
	do {										\
		if (gpuserv_got_sigterm)				\
			wlog(ERROR, "Got SIGTERM");			\
	} while(0)

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

	Assert(worker_exception_stack != NULL);
	siglongjmp(*worker_exception_stack, 1);
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
	else if (worker_exception_stack != 0)
		return -1;
	else
		return 1;
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
	/* Program was not loaded to the current CUDA context yet */
	gmod_cache = malloc(sizeof(GpuModuleCache));
	if (!gmod_cache)
	{
		SpinLockRelease(&gpuserv_cuda_module_lock);
		werror("out of memory");
	}
	gmod_cache->refcnt = 2;
	gmod_cache->program_id = program_id;
	gmod_cache->cuda_module = pgstrom_load_cuda_program(program_id, 0);
	if (!gmod_cache->cuda_module)
	{
		SpinLockRelease(&gpuserv_cuda_module_lock);
		free(gmod_cache);
		return NULL;
	}
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
ReportErrorForBackend(GpuTask_v2 *gtask)
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
	gpuservSendCommand(gtask->gcontext, cmd);
}

/*
 * gpuservPutGpuContext
 *
 * Put a GpuContext for a particular backend, and takes some extra stuff
 * to update the session info.
 */
static void
gpuservPutGpuContext(GpuContext_v2 *gcontext)
{
	pgsocket	sockfd = gcontext->sockfd;

	if (PutGpuContext(gcontext))
	{
		/* remove data socket from the epoll_fd */
		if (epoll_ctl(gpuserv_epoll_fd,
					  EPOLL_CTL_DEL,
					  sockfd, NULL) < 0)
			wnotice("failed on epoll_ctl(EPOLL_CTL_DEL): %m");

		/* close the data socket */
		if (close(sockfd) < 0)
			wnotice("failed on close(sockfd): %m");

		/* unload CUDA modules if no concurrent sessions */
		if (pg_atomic_sub_fetch_u32(&session_num_clients, 1) == 0)
			cleanupGpuModuleCache();
	}
}

/*
 * gpuservTryToWakeUp - wakes up a GPU server process
 */
void
gpuservTryToWakeUp(void)
{
	PGPROC	   *serv_proc;
	uint32		k, i;

	k = i = pg_atomic_fetch_add_u32(&gpuServState->rr_count, 1);
	do {
		serv_proc = gpuServState->serv_procs[k % numDevAttrs];
		if (serv_proc)
		{
			SetLatch(&serv_proc->procLatch);
			break;
		}
		k = pg_atomic_fetch_add_u32(&gpuServState->rr_count, 1);
	} while (k % numDevAttrs != i % numDevAttrs);
}

/*
 * gpuservSendCommand - an internal low-level interface
 */
static void
gpuservSendCommand(GpuContext_v2 *gcontext, GpuServCommand *cmd)
{
	struct msghdr	msg;
	struct iovec    iov;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	ssize_t			retval;

	Assert(cmd->command == GPUSERV_CMD_TASK ||
		   cmd->command == GPUSERV_CMD_ERROR);

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
		werror("failed on sendmsg(2): %m");
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
gpuservSendGpuTask(GpuContext_v2 *gcontext, GpuTask_v2 *gtask)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	GpuServCommand	cmd;

	/* update num_async_tasks */
	SpinLockAcquire(&shgcon->lock);
	if (IsGpuServerProcess())
		shgcon->num_async_tasks--;
	else
		shgcon->num_async_tasks++;
	SpinLockRelease(&shgcon->lock);

	cmd.command = GPUSERV_CMD_TASK;
    cmd.length = offsetof(GpuServCommand, error);
	cmd.gtask = gtask;
	gpuservSendCommand(gcontext, &cmd);
}

/*
 * gpuservOpenConnection - open a unix domain socket from the backend
 * (it may fail if no available GPU server)
 */
void
gpuservOpenConnection(GpuContext_v2 *gcontext)
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
				elog(ERROR, "timeout on connection establish");
		}
		elog(DEBUG2, "connect socket %d to GPU server %d", sockfd, gpuserv_id);
		gcontext->sockfd = sockfd;
	}
	PG_CATCH();
	{
		SpinLockAcquire(&gpuServState->lock);
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
			return;		/* already accept(2) by someone? */
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

	WORKER_TRY();
	{
		/* attach GPU context */
		AttachGpuContext(sockfd, shgcon, gpuserv_epoll_fd);
	}
	WORKER_CATCH();
	{
		SpinLockAcquire(&shgcon->lock);
		shgcon->server = (void *)(~0UL);
		SpinLockRelease(&shgcon->lock);

		SetLatch(&client->procLatch);
		WORKER_RE_THROW();
	}
	WORKER_END_TRY();
	/* wake up the backend process */
	SetLatch(&client->procLatch);
}

/*
 * gpuservRecvCommands - returns number of the commands received
 */
static int
gpuservRecvCommands(GpuContext_v2 *gcontext, bool *p_peer_sock_closed)
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
		*p_peer_sock_closed = true;
		return false;
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
			GpuTask_v2 *gtask;

			Assert(cmd->length == offsetof(GpuServCommand, error));

			gtask = cmd->gtask;
			Assert(dmaBufferValidatePtr(gtask));
			Assert(!gtask->chain.prev && !gtask->chain.next);
			Assert(!gtask->gcontext);
			Assert(!gtask->cuda_stream);
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
				SpinLockAcquire(&session_tasks_lock);
				dlist_push_tail(&session_pending_tasks, &gtask->chain);
				SpinLockRelease(&session_tasks_lock);
			}
			else
			{
				GpuTaskState_v2	   *gts = gtask->gts;

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
		else
			elog(ERROR, "Bug? unknown GPUSERV_CMD_* tag: %d", cmd->command);
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
gpuservRecvGpuTasks(GpuContext_v2 *gcontext, long timeout)
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
static bool
gpuserv_try_run_pending_task(void)
{
	dlist_node	   *dnode;
	GpuTask_v2	   *gtask;

	SpinLockAcquire(&session_tasks_lock);
	if (dlist_is_empty(&session_pending_tasks))
	{
		SpinLockRelease(&session_tasks_lock);
		return false;
	}
	dnode = dlist_pop_head_node(&session_pending_tasks);
	gtask = dlist_container(GpuTask_v2, chain, dnode);
	dlist_push_tail(&session_running_tasks, &gtask->chain);
	SpinLockRelease(&session_tasks_lock);

	WORKER_TRY();
	{
		GpuModuleCache *gmod_cache;
		CUstream		cuda_stream;
		struct timeval	tv;
		cl_int			i, retval;

		gmod_cache = lookupGpuModuleCache(gtask->program_id);
		if (!gmod_cache)
		{
			/* add 250ms wait until next launch */
			gettimeofday(&tv, NULL);
			tv.tv_usec += 250000;
			if (tv.tv_usec >= 1000000)
			{
				tv.tv_usec -= tv.tv_usec;
				tv.tv_sec++;
			}

			/*
			 * When pgstrom_load_cuda_program() returns NULL (it means
			 * build of the supplied program is in-progress), caller
			 * is registered to the waiting processes list.
			 * So, we can expect MyLatch will be set when CUDA program
			 * build gets completed.
			 */
			SpinLockAcquire(&session_tasks_lock);
			dlist_delete(&gtask->chain);
			dlist_push_tail(&session_pending_tasks, &gtask->chain);
			SpinLockRelease(&session_tasks_lock);
		}
		else
		{
			gtask->gmod_cache = gmod_cache;

			/* Pick up a CUDA stream */
			i = pg_atomic_fetch_add_u32(&gpuserv_rr_cuda_stream, 1);
			cuda_stream = gpuserv_cuda_stream[i % gpuserv_num_cuda_stream];

			/*
			 * pgstromProcessGpuTask() returns the following status:
			 *
			 *  0 : GpuTask is successfully queued to the stream.
			 *      It's now running, and then callback will inform its
			 *      completion later.
			 *  1 : Unable to launch GpuTask due to lack of GPU resource.
			 *      It shall be released later, so, back to the pending
			 *      list again.
			 * -1 : GpuTask handler managed its status by itself.
			 *      So, gputasks.c does not need to deal with any more.
			 */
			retval = pgstromProcessGpuTask(gtask,
										   gmod_cache->cuda_module,
										   cuda_stream);
			if (retval > 0)
			{
				/* release GPU module once */
				putGpuModuleCache(gmod_cache);
				gtask->gmod_cache = NULL;

				/* add 400ms wait until next launch */
				gettimeofday(&tv, NULL);
				tv.tv_usec += 400000;
				if (tv.tv_usec >= 1000000)
				{
					tv.tv_usec -= tv.tv_usec;
					tv.tv_sec++;
				}

				SpinLockAcquire(&session_tasks_lock);
				dlist_delete(&gtask->chain);
				dlist_push_tail(&session_pending_tasks, &gtask->chain);
				SpinLockRelease(&session_tasks_lock);
			}
		}
	}
	WORKER_CATCH();
	{
		ReportErrorForBackend(gtask);
		WORKER_RE_THROW();
	}
	WORKER_END_TRY();

	return true;
}


/*
 * gpuserv_try_run_completed_task
 */
static bool
gpuserv_try_run_completed_task(void)
{
	dlist_node	   *dnode;
	GpuTask_v2	   *gtask;

	SpinLockAcquire(&session_tasks_lock);
	if (dlist_is_empty(&session_completed_tasks))
	{
		SpinLockRelease(&session_tasks_lock);
		return false;
	}
	dnode = dlist_pop_head_node(&session_completed_tasks);
	gtask = dlist_container(GpuTask_v2, chain, dnode);
	memset(&gtask->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&session_tasks_lock);

	WORKER_TRY();
	{
		GpuContext_v2  *gcontext = gtask->gcontext;
		GpuModuleCache *gmod_cache = gtask->gmod_cache;
		int				peer_fdesc = gtask->peer_fdesc;
		int				retval;

		/* release CUDA module */
		putGpuModuleCache(gmod_cache);
		gtask->gmod_cache = NULL;

		/*
		 * pgstromCompleteGpuTask returns the following status:
		 *
		 *  0: GpuTask is successfully completed. So, let's return to
		 *     the backend process over the socket.
		 *  1: GpuTask needs to retry GPU kernel execution. So, let's
		 *     attach pending list again.
		 * -1: GpuTask wants to be released here. So, task shall be
		 *     removed and num_async_tasks shall be decremented.
		 */
		retval = pgstromCompleteGpuTask(gtask);
		if (retval == 0)
		{
			gtask->gcontext = NULL;
			gtask->peer_fdesc = -1;
			gpuservSendGpuTask(gcontext, gtask);
			if (peer_fdesc >= 0)
				close(peer_fdesc);
			gpuservPutGpuContext(gcontext);
		}
		else if (retval > 0 && GpuContextIsEstablished(gcontext))
		{
			/*
			 * GpuTask wants to execute GPU kernel again, so attach
			 * this task on the pending list again, as long as the
			 * backend is still valid.
			 */
			SpinLockAcquire(&session_tasks_lock);
			dlist_push_head(&session_pending_tasks, &gtask->chain);
			SpinLockRelease(&session_tasks_lock);
		}
		else
		{
			/*
			 * GpuTask wants to release this task without message-back.
			 * So, we have to decrement num_async_tasks and set latch
			 * of the backend process (it may wait for the last task).
			 *
			 * NOTE: We may not have this option because performance
			 * counter data needs to be written back to the backend
			 * process. Omit of message-back will give us small
			 * optimization benefit, but it makes performance counter
			 * code complicated...
			 */
			SharedGpuContext   *shgcon = gcontext->shgcon;
			PGPROC			   *backend;

			pgstromReleaseGpuTask(gtask);

			SpinLockAcquire(&shgcon->lock);
			shgcon->num_async_tasks--;
			backend = shgcon->backend;
			SpinLockRelease(&shgcon->lock);

			if (backend)
				SetLatch(&backend->procLatch);

			if (peer_fdesc >= 0)
				close(peer_fdesc);
			gpuservPutGpuContext(gcontext);
		}
	}
	WORKER_CATCH();
	{
		ReportErrorForBackend(gtask);
		WORKER_RE_THROW();
	}
	WORKER_END_TRY();

	return true;
}

/*
 * gpuservPushGpuTask - attach a GpuTask to the queue by GPU server itself
 */
void
gpuservPushGpuTask(GpuContext_v2 *gcontext, GpuTask_v2 *gtask)
{
	SharedGpuContext *shgcon = gcontext->shgcon;

	if (!IsGpuServerProcess())
		elog(FATAL, "Bug? %s is called out of GPU server's context",
			__FUNCTION__);
	SpinLockAcquire(&shgcon->lock);
	shgcon->num_async_tasks++;
	SpinLockRelease(&shgcon->lock);
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
gpuservCompleteGpuTask(GpuTask_v2 *gtask, bool is_urgent)
{
	SpinLockAcquire(&session_tasks_lock);
	dlist_delete(&gtask->chain);
	if (is_urgent)
		dlist_push_head(&session_completed_tasks, &gtask->chain);
	else
		dlist_push_tail(&session_completed_tasks, &gtask->chain);
	SpinLockRelease(&session_tasks_lock);

	SetLatch(MyLatch);
}

/*
 * gpuservWorkerMain
 */
static void *
gpuservWorkerMain(void *__private)
{
	Datum	worker_id __attribute__((unused)) = PointerGetDatum(__private);

	wnotice("GPU worker %ld began working", worker_id);

	WORKER_TRY();
	{
		for (;;)
		{
			struct epoll_event	ep_event;
			int			nevents;
			uint64		ev_count;
			long		timeout = 5000;	/* 5.0s interval */

			if (gpuserv_got_sigterm)
				break;
			/* reset event fd */
			if (read(gpuserv_event_fd, &ev_count, sizeof(ev_count)) < 0 &&
				errno != EAGAIN)
				wlog("failed on read(gpuserv_event_fd): %m");
			if (pgstrom_try_build_cuda_program())
				timeout = 0;
			if (gpuserv_try_run_completed_task())
				timeout = 0;
			if (gpuserv_try_run_pending_task())
				timeout = 0;

			nevents = epoll_wait(gpuserv_epoll_fd,
								 &ep_event, 1,
								 timeout);	/* may be non-blocking poll */
			if (nevents > 0)
			{
				Assert(nevents == 1);
				if (ep_event.data.ptr == NULL)
				{
					/* event around event fd, just wake up */
					if (ep_event.events & ~EPOLLIN)
						werror("eventfd reported %08x", ep_event.events);
				}
				else if (ep_event.data.ptr == (void *)(~0UL))
				{
					/* event around server socket */
					if (ep_event.events & EPOLLIN)
						gpuservAcceptConnection();
					else
						werror("server socket reported %08x", ep_event.events);
				}
				else
				{
					/* event around client socket */
					GpuContext_v2  *gcontext = ep_event.data.ptr;
					bool			peer_sock_closed;

					lock_per_data_socket(gcontext->sockfd);
					if (ep_event.events & EPOLLIN)
						gpuservRecvCommands(gcontext, &peer_sock_closed);
					else
						peer_sock_closed = true;

					if (peer_sock_closed)
					{
						PutGpuContext(gcontext);
						if (epoll_ctl(gpuserv_epoll_fd,
									  EPOLL_CTL_DEL,
									  gcontext->sockfd, NULL) < 0)
						{
							unlock_per_data_socket(gcontext->sockfd);
							werror("failed on epoll_ctl(EPOLL_CTL_DEL): %m");
						}
					}
					unlock_per_data_socket(gcontext->sockfd);
				}
			}
			else if (nevents < 0 && errno != EINTR)
				werror("failed on epoll_wait(2): %m");
		}
			
	}
	WORKER_CATCH();
	{
		/*
		 * Suggest primary thread of the background worker to wake up
		 * any other worker threads and terminate them.
		 */
		gpuservGotSigterm(0);
	}
	WORKER_END_TRY();

	wnotice("GPU worker %ld is terminating", worker_id);

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
 * gpuserv_terminate_workers - terminates all the running workers
 */
static void
gpuserv_terminate_workers(int nworkers)
{
	int			i, errcode;

	Assert(nworkers >= 0 && nworkers <= GpuServerNumWorkers);
	Assert(worker_exception_stack == (void *)(~0UL));
	/* ensure termination of the worker threads */
	gpuserv_got_sigterm = true;
	pg_memory_barrier();

	gpuserv_wakeup_workers();

	/* synchronization of worker threads termination */
	for (i=0; i < nworkers; i++)
	{
		pthread_t	thread = gpuworker_threads[i];

		errcode = pthread_join(thread, NULL);
		if (errcode != 0 && errcode != ESRCH)
			elog(LOG, "failed on pthread_join(worker: %d): %s",
				 i, strerror(errcode));
		wnotice("worker-%d pthread_join() done", i);
	}
	worker_exception_stack = NULL;
}

/*
 * gpuservEventLoop
 */
static void
gpuservEventLoop(void)
{
	WaitEventSet *wait_events;
	long		timeout = -1;
	Datum		i;

	memset(&worker_exception_data, 0, sizeof(worker_exception_data));
	SpinLockInit(&worker_exception_data.lock);

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
	 * Launch worker threads
	 */
	gpuworker_threads = calloc(GpuServerNumWorkers, sizeof(pthread_t));
	if (!gpuworker_threads)
		elog(ERROR, "out of memory");

	worker_exception_stack = (void *)(~0UL);
	for (i=0; i < GpuServerNumWorkers; i++)
	{
		int		code = pthread_create(gpuworker_threads + i, NULL,
									  gpuservWorkerMain,
									  DatumGetPointer(i));
		if (code != 0)
		{
			gpuserv_terminate_workers(i);
			elog(ERROR, "failed on pthread_create: %s", strerror(code));
		}
	}

	/*
	 * Event loop
	 */
	elog(LOG, "PG-Strom GPU Server is now ready on GPU-%d %s",
		 devAttrs[gpuserv_cuda_dindex].DEV_ID,
		 devAttrs[gpuserv_cuda_dindex].DEV_NAME);
	PG_TRY();
	{
		for (;;)
		{
			WaitEvent	ev;
			int			nevents;

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
			/* refresh profiler's status */
			pgstrom_cuda_profiler_update();
		}
	}
	PG_CATCH();
	{
		gpuserv_terminate_workers(GpuServerNumWorkers);
		PG_RE_THROW();
	}
	PG_END_TRY();
	gpuserv_terminate_workers(GpuServerNumWorkers);

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
	CUresult		rc;
	cl_int			i;
	struct epoll_event ep_event;

	/* I am a GPU server process */
	gpuserv_id = DatumGetInt32(__server_id);
	Assert(gpuserv_id >= 0 && gpuserv_id < numDevAttrs);
	gpuServState->serv_procs[gpuserv_id] = MyProc;
	pqsignal(SIGTERM, gpuservGotSigterm);
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
	dlist_init(&session_running_tasks);
	dlist_init(&session_completed_tasks);
	pg_atomic_init_u32(&session_num_clients, 0);

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
	ep_event.data.ptr = NULL;		/* identifier of event fd */
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

	ep_event.events = EPOLLIN | EPOLLET;
	ep_event.data.ptr = (void *)(~0UL);		/* identifier of server socket */
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
					 gpuserv_cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxCreate: %s", errorText(rc));

	pg_atomic_init_u32(&gpuserv_rr_cuda_stream, 0);
	gpuserv_num_cuda_stream = 5 * GpuServerNumWorkers;
	gpuserv_cuda_stream = malloc(sizeof(CUstream) * gpuserv_num_cuda_stream);
	if (!gpuserv_cuda_stream)
		elog(FATAL, "out of memory");
	for (i=0; i < gpuserv_num_cuda_stream; i++)
	{
		rc = cuStreamCreate(gpuserv_cuda_stream + i, CU_STREAM_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(FATAL, "failed on cuStreamCreate: %s", errorText(rc));
	}
	for (i=0; i < CUDA_MODULE_CACHE_NSLOTS; i++)
		dlist_init(&gpuserv_cuda_module_cache[i]);
	SpinLockInit(&gpuserv_cuda_module_lock);
	for (i=0; i < NUM_DATA_SOCKET_LOCKS; i++)
		SpinLockInit(&per_data_socket_locks[i]);
	pgstrom_cuda_profiler_init();

	PG_TRY();
	{
		gpuservEventLoop();
		elog(ERROR, "GPU Server-%d [%s] is terminating normally",
			 gpuserv_id, devAttrs[gpuserv_id].DEV_NAME);
	}
	PG_CATCH();
	{
		/* At this point, all the worker threads should be terminated */
		Assert(IsGpuServerProcess() > 0);

		/* GPU server is terminating... */
		gpuServState->serv_procs[gpuserv_id] = NULL;

		/*
		 * Destroy the CUDA context not to wake up the callback functions
		 * any more, regardless of the status of asynchronous GpuTasks.
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
 * pgstrom_cuda_profiler_init (per GPU server)
 */
static void
pgstrom_cuda_profiler_init(void)
{
	char		output_path[MAXPGPATH];
	int			c, i, j;
	CUresult	rc;

	Assert(IsGpuServerProcess());
	for (i=0, j=0; (c = gpuserv_cuda_profiler_log_file[i]) != '\0'; i++)
	{
		if (c != '%')
			output_path[j++] = c;
		else
			j += snprintf(output_path + j, MAXPGPATH - j,
						  "%d", gpuserv_cuda_dindex);
	}
	output_path[i] = 0;

	rc = cuProfilerInitialize(gpuserv_cuda_profiler_config_file,
							  output_path,
							  CU_OUT_CSV);
	if (rc != CUDA_SUCCESS)
	{
		if (rc != CUDA_ERROR_PROFILER_DISABLED)
			elog(FATAL, "failed on cuProfilerInitialize: %s", errorText(rc));
		else
		{
			gpuserv_cuda_profiler = -1;
			elog(LOG, "CUDA Profiler is disabled");
		}
	}
}

/*
 * Update CUDA Profiler support to the latest state
 */
static void
pgstrom_cuda_profiler_update(void)
{
	CUresult	rc;

	Assert(IsGpuServerProcess());

	if (gpuserv_cuda_profiler < 0)
		return;
	else if (gpuserv_cuda_profiler == 0)
	{
		if (pg_atomic_read_u32(&gpuServState->cuda_profiler) != 0)
		{
			rc = cuProfilerStart();
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuProfilerStart: %s", errorText(rc));
			gpuserv_cuda_profiler = 1;
		}
	}
	else
	{
		if (pg_atomic_read_u32(&gpuServState->cuda_profiler) == 0)
		{
			rc = cuProfilerStop();
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuProfilerStop: %s", errorText(rc));
			gpuserv_cuda_profiler = 1;
		}
	}
}

/*
 * pgstrom_cuda_profiler_guc_check
 */
static bool
pgstrom_cuda_profiler_guc_check(bool *newval, void **extra, GucSource source)
{
	int		cuda_profiler;

	if (gpuServState)
	{
		cuda_profiler = (int)pg_atomic_read_u32(&gpuServState->cuda_profiler);

		if (cuda_profiler >= 0)
			return true;
	}
	elog(ERROR, "CUDA Profiler is not initialized");
}

/*
 * pgstrom_cuda_profiler_guc_assign
 */
static void
pgstrom_cuda_profiler_guc_assign(bool new_config, void *extra)
{
	uint32		oldval = (new_config ? 0 : 1);
	uint32		newval = (new_config ? 1 : 0);
	int			i;

	Assert(gpuServState);
	if (pg_atomic_compare_exchange_u32(&gpuServState->cuda_profiler,
									   &oldval, newval))
	{
		/* wakeup all the active servers */
		SpinLockAcquire(&gpuServState->lock);
		for (i=0; i < numDevAttrs; i++)
		{
			PGPROC *serv = gpuServState->serv_procs[i];

			SetLatch(&serv->procLatch);
		}
		SpinLockRelease(&gpuServState->lock);
	}
}

/*
 * pgstrom_cuda_profiler_guc_show
 */
static const char *
pgstrom_cuda_profiler_guc_show(void)
{
	int		cuda_profiler;

	if (gpuServState)
	{
		cuda_profiler = (int)pg_atomic_read_u32(&gpuServState->cuda_profiler);

		if (cuda_profiler > 0)
			return "on";
		else if (cuda_profiler == 0)
			return "off";
	}
	return "disabled";
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
	static bool	__pgstrom_cuda_profiler;	/* dummy */

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* request for the static shared memory */
	required = offsetof(GpuServState, serv_procs[numDevAttrs]);
	gpuServState = ShmemInitStruct("gpuServState", required, &found);
	Assert(!found);
	memset(gpuServState, 0, required);
	SpinLockInit(&gpuServState->lock);
	dlist_init(&gpuServState->conn_pending_list);

	required = sizeof(GpuServConn) * numBackends;
	gpuServConn = ShmemInitStruct("gpuServConn", required, &found);
	Assert(!found);
	memset(gpuServConn, 0, required);

	/*
	 * Support for CUDA visual profiler
	 */
	DefineCustomBoolVariable("pg_strom.cuda_profiler",
							 "start/stop CUDA visual profiler",
							 NULL,
							 &__pgstrom_cuda_profiler,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 pgstrom_cuda_profiler_guc_check,
							 pgstrom_cuda_profiler_guc_assign,
							 pgstrom_cuda_profiler_guc_show);
}

/*
 * pgstrom_init_gpu_server
 */
void
pgstrom_init_gpu_server(void)
{
	char		path[MAXPGPATH];
	char		config_file[MAXPGPATH];
	cl_uint		i, numBackends;

	/*
	 * Number of the worker threads per device
	 */
	DefineCustomIntVariable("pg_strom.gpuserv_nworkers",
							"number of GPU server worker threads",
							NULL,
							&GpuServerNumWorkers,
							4,
							1,
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

	get_share_path(my_exec_path, path);
	snprintf(config_file, MAXPGPATH,
			 "%s/extension/cuda_profiler.ini", path);
	DefineCustomStringVariable("pg_strom.cuda_profiler_config_file",
							   "filename of CUDA profiler config file",
							   NULL,
							   &gpuserv_cuda_profiler_config_file,
							   config_file,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE |
							   GUC_SUPERUSER_ONLY,
							   NULL, NULL, NULL);
	
	DefineCustomStringVariable("pg_strom.cuda_profiler_log_file",
							   "filename of CUDA profiler log file",
							   NULL,
							   &gpuserv_cuda_profiler_log_file,
							   "cuda_profile.%.log",
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE |
							   GUC_SUPERUSER_ONLY,
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
											   serv_procs[numDevAttrs])) +
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

/*
 * optimal_workgroup_size - calculates the optimal block size
 * according to the function and device attributes
 */
static size_t __dynamic_shmem_per_block;
static size_t __dynamic_shmem_per_thread;

static size_t
blocksize_to_shmemsize_helper(int blocksize)
{
	return (__dynamic_shmem_per_block +
			__dynamic_shmem_per_thread * (size_t)blocksize);
}

void
optimal_workgroup_size(size_t *p_grid_size,
					   size_t *p_block_size,
					   CUfunction function,
					   CUdevice device,
					   size_t nitems,
					   size_t dynamic_shmem_per_block,
					   size_t dynamic_shmem_per_thread)
{
	cl_int		min_grid_sz;
	cl_int		max_block_sz;
	cl_int		warpSize;
	CUresult	rc;

	rc = cuDeviceGetAttribute(&warpSize,
							  CU_DEVICE_ATTRIBUTE_WARP_SIZE,
							  device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	__dynamic_shmem_per_block = dynamic_shmem_per_block;
	__dynamic_shmem_per_thread = dynamic_shmem_per_thread;
	rc = cuOccupancyMaxPotentialBlockSize(&min_grid_sz,
										  &max_block_sz,
										  function,
										  blocksize_to_shmemsize_helper,
										  0,
										  Min((size_t)nitems,
											  (size_t)INT_MAX));
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuOccupancyMaxPotentialBlockSize: %s",
			 errorText(rc));

	if ((size_t)max_block_sz * (size_t)INT_MAX < nitems)
		elog(ERROR, "to large nitems (%zu) to launch kernel (blockSz=%d)",
			 nitems, max_block_sz);

	*p_block_size = (size_t)max_block_sz;
	*p_grid_size  = (nitems + (size_t)max_block_sz - 1) / (size_t)max_block_sz;
}

/*
 * largest_workgroup_size - calculate the block size maximum available
 */
void
largest_workgroup_size(size_t *p_grid_size,
					   size_t *p_block_size,
					   CUfunction function,
					   CUdevice device,
					   size_t nitems,
					   size_t dynamic_shmem_per_block,
					   size_t dynamic_shmem_per_thread)
{
	cl_int		warpSize;
	cl_int		maxBlockSize;
	cl_int		staticShmemSize;
	cl_int		maxShmemSize;
	cl_int		shmemSizeTotal;
	CUresult	rc;

	/* get max number of thread per block on this kernel function */
	rc = cuFuncGetAttribute(&maxBlockSize,
							CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
							function);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

	/* get statically allocated shared memory */
	rc = cuFuncGetAttribute(&staticShmemSize,
							CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
							function);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

	/* get device warp size */
	rc = cuDeviceGetAttribute(&warpSize,
							  CU_DEVICE_ATTRIBUTE_WARP_SIZE,
							  device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	/* get device limit of thread/block ratio */
	rc = cuDeviceGetAttribute(&maxShmemSize,
							  CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK,
							  device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	/* only shared memory consumption is what we have to control */
	shmemSizeTotal = (staticShmemSize +
					  dynamic_shmem_per_block +
					  dynamic_shmem_per_thread * maxBlockSize);
	if (shmemSizeTotal > maxShmemSize)
	{
		if (dynamic_shmem_per_thread > 0 &&
			staticShmemSize +
			dynamic_shmem_per_block +
			dynamic_shmem_per_thread * warpSize <= maxShmemSize)
		{
			maxBlockSize = (maxShmemSize -
							staticShmemSize -
							dynamic_shmem_per_block)/dynamic_shmem_per_thread;
			maxBlockSize = (maxBlockSize / warpSize) * warpSize;
		}
		else
			elog(ERROR,
				 "too large fixed amount of shared memory consumption: "
				 "static: %d, dynamic-per-block: %zu, dynamic-per-thread: %zu",
				 staticShmemSize,
				 dynamic_shmem_per_block,
				 dynamic_shmem_per_thread);
	}

	if ((size_t)maxBlockSize * (size_t)INT_MAX < nitems)
		elog(ERROR, "to large nitems (%zu) to launch kernel (blockSz=%d)",
			 nitems, maxBlockSize);

	*p_block_size = (size_t)maxBlockSize;
	*p_grid_size = (nitems + maxBlockSize - 1) / maxBlockSize;
}
