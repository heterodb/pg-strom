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

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <cudaProfiler.h>

typedef struct GpuServProc
{
	dlist_node		chain;		/* link to serv_procs_list */
	cl_uint			gpuserv_id;	/* identifier of this GPU server */
	cl_uint			num_clients;/* number of the clients */
	PGPROC		   *pgproc;		/* reference to the server's PGPROC */
	BackendId		backend_id;	/* backend which is going to connect to */
	cl_uint			gcontext_id;/* gcontext which is going to be assigned to */
	/* per gpu-server state counter */
	pg_atomic_uint64	gpu_mem_usage;	/* amount of GPU RAM by the server */
	pg_atomic_uint64	gpu_task_count;	/* count of GPU tasks in this server */
} GpuServProc;

#define GPUSERV_IS_ACTIVE(gsproc)								\
	((gsproc)->pgproc && (gsproc)->backend_id != InvalidBackendId)

typedef struct GpuServState
{
	pg_atomic_uint32 rr_count;	/* seed for round-robin distribution */
	pg_atomic_uint32 cuda_profiler; /* GUC: pg_strom.cuda_profiler */
	slock_t			lock;
	dlist_head	   *serv_procs_list;	/* for each device */
	GpuServProc		serv_procs[FLEXIBLE_ARRAY_MEMBER];
} GpuServState;

#define WORDNUM(x)	((x) / BITS_PER_BITMAPWORD)
#define BITNUM(x)	((x) % BITS_PER_BITMAPWORD)

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
		char	buffer[FLEXIBLE_ARRAY_MEMBER];
	} error;
} GpuServCommand;

/*
 * static/public variables
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static struct sockaddr_un *gpuServSockAddr;		/* const */
static GpuServState	   *gpuServState = NULL;	/* shmem */
static GpuServProc	   *gpuServProc = NULL;		/* shmem */
static int				numGpuServers;			/* GUC */
static int				GpuServerCommTimeout;	/* GUC */
static int				pgstrom_max_async_tasks_per_device;	/* GUC */
static bool				gpuserv_got_sigterm = false;
static int				gpuserv_id = -1;
static pgsocket			gpuserv_server_sockfd = PGINVALID_SOCKET;
int						gpuserv_cuda_dindex = -1;
CUdevice				gpuserv_cuda_device = NULL;
CUcontext				gpuserv_cuda_context = NULL;
static int32			gpuserv_cuda_profiler = 0;
static char			   *gpuserv_cuda_profiler_config_file = NULL;
static char			   *gpuserv_cuda_profiler_log_file = NULL;
/* GPU server session info */
static slock_t			session_tasks_lock;
static dlist_head		session_pending_tasks;
static dlist_head		session_running_tasks;
static dlist_head		session_completed_tasks;
static cl_int			session_num_clients = 0;
static WaitEvent	   *session_events = NULL;
static WaitEventSet	   *session_event_set = NULL;

/*
 * static functions
 */
static bool gpuservSendCommand(GpuContext_v2 *gcontext,
							   GpuServCommand *cmd, long timeout);
static void	pgstrom_cuda_profiler_init(void);
static void	pgstrom_cuda_profiler_update(void);

/* SIGTERM handler */
static void
gpuservGotSigterm(SIGNAL_ARGS)
{
	int		save_errno = errno;

	gpuserv_got_sigterm = true;

	SetLatch(MyLatch);

	errno = save_errno;
}

/* signal checks for GpuServer */
#define GPUSERV_CHECK_FOR_INTERRUPTS()									\
	do {																\
		CHECK_FOR_INTERRUPTS();											\
		if (gpuserv_got_sigterm)										\
		{																\
			Assert(IsGpuServerProcess());								\
			elog(FATAL, "GPU/CUDA Server [%d] (GPU-%d %s) was terminated", \
				 gpuserv_id,											\
				 devAttrs[gpuserv_cuda_dindex].DEV_ID,					\
				 devAttrs[gpuserv_cuda_dindex].DEV_NAME);				\
		}																\
	} while(0)

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
 * IsGpuServerProcess - returns true, if current process is gpu server
 */
bool
IsGpuServerProcess(void)
{
	return (bool)(gpuserv_id >= 0);
}

/*
 * numGpuServerProcesses - returns number of GPU server processes
 */
int
numGpuServerProcesses(void)
{
	return numGpuServers;
}

/*
 * lookupCudaModuleCache - find a CUDA module loaded in the current context
 */

/* CUDA module lookup cache */
#define CUDA_MODULES_SLOTSIZE		200
static dlist_head		cuda_modules_slot[CUDA_MODULES_SLOTSIZE];

typedef struct
{
	dlist_node		chain;
	ProgramId		program_id;
	CUmodule		cuda_module;
} cudaModuleCache;

static CUmodule
lookupCudaModuleCache(ProgramId program_id)
{
	dlist_iter	iter;
	int			index;
	cudaModuleCache *mod_cache;

	/* Is the cuda module already loaded? */
	index = program_id % CUDA_MODULES_SLOTSIZE;
	dlist_foreach (iter, &cuda_modules_slot[index])
	{
		mod_cache = dlist_container(cudaModuleCache, chain, iter.cur);

		if (mod_cache->program_id == program_id)
			return mod_cache->cuda_module;
	}

	/* Program was not loaded to the current context yet */
	mod_cache = palloc0(sizeof(cudaModuleCache));
	mod_cache->program_id = program_id;
	mod_cache->cuda_module = pgstrom_load_cuda_program(program_id, 0);
	if (!mod_cache->cuda_module)
	{
		pfree(mod_cache);
		return NULL;
	}
	dlist_push_head(&cuda_modules_slot[index], &mod_cache->chain);

	return mod_cache->cuda_module;
}

/*
 * ReportErrorForBackend - it shall be called on the error handler.
 * It send back a error command to the backend immediately.
 */
static void
ReportErrorForBackend(GpuContext_v2 *gcontext, MemoryContext memcxt)
{
	MemoryContext	oldcxt;
	ErrorData	   *errdata;
	GpuServCommand *cmd;
	Size			required;
	cl_uint			offset = 0;
	long			timeout = 5000;		/* 5.0sec */

	/*
	 * Prior to the ereport() to exit the server once, we deliver
	 * the error detail to the backend to cause an ereport on the
	 * backend side also.
	 */
	oldcxt = MemoryContextSwitchTo(memcxt);
	errdata = CopyErrorData();

	required = (offsetof(GpuServCommand, error.buffer) +
				MAXALIGN(strlen(errdata->filename) + 1) +
				MAXALIGN(strlen(errdata->funcname) + 1) +
				MAXALIGN(strlen(errdata->message) + 1));
	cmd = palloc(required);
	cmd->command            = GPUSERV_CMD_ERROR;
	cmd->length             = required;
	cmd->gtask				= NULL;
	cmd->error.elevel		= errdata->elevel;
	cmd->error.sqlerrcode	= errdata->sqlerrcode;

	cmd->error.filename_offset = offset;
	strcpy(cmd->error.buffer + offset, errdata->filename);
	offset += MAXALIGN(strlen(errdata->filename) + 1);

	cmd->error.lineno		= errdata->lineno;

	cmd->error.funcname_offset = offset;
	strcpy(cmd->error.buffer + offset, errdata->funcname);
	offset += MAXALIGN(strlen(errdata->funcname) + 1);

	cmd->error.message_offset = offset;
	strcpy(cmd->error.buffer + offset, errdata->message);
	offset += MAXALIGN(strlen(errdata->message) + 1);

	/* Urgent return to the backend */
	gpuservSendCommand(gcontext, cmd, timeout);
	MemoryContextSwitchTo(oldcxt);
}

/*
 * reconstructSessionEventSet - refresh session_event_set according to the
 * latest session_events.
 */
static void
reconstructSessionEventSet(void)
{
	int		i;

	/* release previous one if any */
	if (session_event_set)
		FreeWaitEventSet(session_event_set);
	/* reconstruct a new one */
	session_event_set = CreateWaitEventSet(TopMemoryContext,
										   MaxBackends + 3);
	AddWaitEventToSet(session_event_set,
					  WL_POSTMASTER_DEATH,
					  PGINVALID_SOCKET, NULL, NULL);
	AddWaitEventToSet(session_event_set,
					  WL_LATCH_SET,
					  PGINVALID_SOCKET, MyLatch, NULL);
	AddWaitEventToSet(session_event_set,
					  WL_SOCKET_READABLE,
					  gpuserv_server_sockfd, NULL, NULL);
	for (i=0; i < session_num_clients; i++)
	{
		AddWaitEventToSet(session_event_set,
						  session_events[i].events,
						  session_events[i].fd,
						  NULL,		/* no latch */
						  session_events[i].user_data);
	}
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
	if (PutGpuContext(gcontext))
	{
		cl_int		i;

		/* update the shared scoreboard */
		SpinLockAcquire(&gpuServState->lock);
		gpuServProc->num_clients--;
		SpinLockRelease(&gpuServState->lock);

		/*
		 * Remove the closed session from the WaitEventSet
		 *
		 * NOTE: It should NOT be removed when a connection gets closed
		 * immediately, because asynchronous tasks may be still running
		 * when peer backend closed the UNIX domain socket.
		 * If this asynchronous task raised an error after the connection
		 * closed, try...catch block forces to detach SharedGpuContext
		 * not to leak shared memory region. However, if we would remove
		 * the WaitEvent immediately, error handler cannot reference the
		 * SharedGpuContext, then it leads memory leak!
		 */
		for (i=0; i < session_num_clients; i++)
		{
			if (session_events[i].user_data != (void *)gcontext)
				continue;

			if (i + 1 < session_num_clients)
				memmove(session_events + i,
						session_events + i + 1,
						sizeof(WaitEvent) * (session_num_clients - (i + 1)));
			session_num_clients--;

			reconstructSessionEventSet();

			/*
			 * Inactivates this GPU server process - If supplied GpuContext
			 * is the last session of the backends group, this GPU server
			 * will become able to accept connection from any other unrelated
			 * sessions again.
			 */
			if (session_num_clients == 0)
			{
				bool	unload_cuda_modules = true;

				SpinLockAcquire(&gpuServState->lock);
				if (gpuServProc->backend_id != InvalidBackendId)
				{
					/*
					 * NOTE: Oh, it is an extreme corver case. Last session
					 * was closed just moment before, however, a new sibling
					 * backend trying to accept(2).
					 * Next WaitEventSetWait() will return immediately because
					 * the backend already issued connect(2) system call. So,
					 * we don't need to inactivate this GPU server.
					 */
					unload_cuda_modules = false;
				}
				else
				{
					dlist_head	   *dhead;

					gpuServProc->backend_id = InvalidBackendId;
					gpuServProc->gcontext_id = INVALID_GPU_CONTEXT_ID;

					dhead = &gpuServState->serv_procs_list[gpuserv_cuda_dindex];
					dlist_delete(&gpuServProc->chain);
					dlist_push_tail(dhead, &gpuServProc->chain);
				}
				SpinLockRelease(&gpuServState->lock);

				/* Unload all the CUDA modules used by this session */
				if (unload_cuda_modules)
				{
					cudaModuleCache	   *cache;
                    dlist_node		   *dnode;
                    CUresult			rc;

					for (i=0; i < CUDA_MODULES_SLOTSIZE; i++)
					{
						while (!dlist_is_empty(&cuda_modules_slot[i]))
						{
							dnode = dlist_pop_head_node(&cuda_modules_slot[i]);
							cache = dlist_container(cudaModuleCache,
													chain, dnode);
							rc = cuModuleUnload(cache->cuda_module);
							if (rc != CUDA_SUCCESS)
								elog(WARNING, "failed on cuModuleUnload: %s",
									 errorText(rc));
						}
					}
				}
			}
			return;
		}
		elog(FATAL, "Bug? GPU server misses the GpuContext");
	}
}

/*
 * gpuservProcessPendingTasks
 */
static void
gpuservProcessPendingTasks(bool *p_has_pending, bool *p_has_completed)
{
	MemoryContext	memcxt = CurrentMemoryContext;
	dlist_node	   *dnode;
	GpuTask_v2	   *gtask;
	GpuContext_v2  *gcontext;
	bool			process_continue = true;

	SpinLockAcquire(&session_tasks_lock);
	while (process_continue &&
		   !dlist_is_empty(&session_pending_tasks))
	{
		CUmodule	cuda_module;
		CUstream	cuda_stream;
		CUresult	rc;
		cl_int		retval;

		dnode = dlist_pop_head_node(&session_pending_tasks);
		gtask = dlist_container(GpuTask_v2, chain, dnode);
		dlist_push_tail(&session_running_tasks, &gtask->chain);
		SpinLockRelease(&session_tasks_lock);

		GPUSERV_CHECK_FOR_INTERRUPTS();

		gcontext = gtask->gcontext;
		PG_TRY();
		{
			cuda_module = lookupCudaModuleCache(gtask->program_id);
			if (!cuda_module)
			{
				/*
				 * When pgstrom_load_cuda_program() returns NULL (it means
				 * build of the supplied program is in-progress), caller
				 * is registered to the waiting processes list.
				 * So, we can expect MyLatch will be set when CUDA program
				 * build gets completed.
				 */
				SpinLockAcquire(&session_tasks_lock);
				dlist_delete(&gtask->chain);
				dlist_push_head(&session_pending_tasks, &gtask->chain);
				process_continue = false;
			}
			else
			{
				if (gtask->cuda_stream)
					cuda_stream = gtask->cuda_stream;
				else
				{
					rc = cuStreamCreate(&cuda_stream, CU_STREAM_DEFAULT);
					if (rc != CUDA_SUCCESS)
						elog(ERROR, "failed on cuStreamCreate: %s",
							 errorText(rc));
				}
				gtask->cuda_stream = cuda_stream;

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
											   cuda_module,
											   cuda_stream);
				SpinLockAcquire(&session_tasks_lock);
				if (retval)
				{
					dlist_delete(&gtask->chain);
					dlist_push_head(&session_pending_tasks, &gtask->chain);
					process_continue = false;
				}
			}
		}
		PG_CATCH();
		{
			ReportErrorForBackend(gcontext, memcxt);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	*p_has_pending = !dlist_is_empty(&session_pending_tasks);
	*p_has_completed = !dlist_is_empty(&session_completed_tasks);
	SpinLockRelease(&session_tasks_lock);
}

/*
 * gpuservFlushOutCompletedTasks
 */
static void
gpuservFlushOutCompletedTasks(void)
{
	MemoryContext	memcxt = CurrentMemoryContext;
	dlist_node	   *dnode;
	GpuTask_v2	   *gtask;
	GpuContext_v2  *gcontext;
	int				peer_fdesc;
	int				retval;
	CUresult		rc;

	SpinLockAcquire(&session_tasks_lock);
	while (!dlist_is_empty(&session_completed_tasks))
	{
		dnode = dlist_pop_head_node(&session_completed_tasks);
		gtask = dlist_container(GpuTask_v2, chain, dnode);
		memset(&gtask->chain, 0, sizeof(dlist_node));
		SpinLockRelease(&session_tasks_lock);

		GPUSERV_CHECK_FOR_INTERRUPTS();

		gcontext = gtask->gcontext;
		peer_fdesc = gtask->peer_fdesc;
		PG_TRY();
		{
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

			/* release of CUDA stream is caller's job */
			rc = cuStreamDestroy(gtask->cuda_stream);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuStreamDestroy: %s", errorText(rc));
			gtask->cuda_stream = NULL;

			if (retval == 0)
			{
				gtask->gcontext = NULL;
				gtask->peer_fdesc = -1;
				if (!gpuservSendGpuTask(gcontext, gtask))
					elog(ERROR, "failed on gpuservSendGpuTask");
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
				pg_atomic_sub_fetch_u64(shgcon->gpu_task_count, 1);
				backend = shgcon->backend;
				SpinLockRelease(&shgcon->lock);

				if (backend)
					SetLatch(&backend->procLatch);

				if (peer_fdesc >= 0)
					close(peer_fdesc);
				gpuservPutGpuContext(gcontext);
			}
		}
		PG_CATCH();
		{
			ReportErrorForBackend(gcontext, memcxt);
			PG_RE_THROW();
		}
		PG_END_TRY();

		SpinLockAcquire(&session_tasks_lock);
	}
	SpinLockRelease(&session_tasks_lock);
}

/*
 * gpuservTryToWakeUp - wakes up a (likely inactive) GPU server process
 */
void
gpuservTryToWakeUp(void)
{
	int			i, j, k;
	int			only_inactive = 1;

	k = pg_atomic_fetch_add_u32(&gpuServState->rr_count, 1);
	SpinLockAcquire(&gpuServState->lock);
	do {
		/*
		 * wake up inactive server process first, then try to wake up anybody
		 * if no inactive server process found.
		 */
		for (i=0; i < numDevAttrs; i++)
		{
			dlist_iter		iter;
			GpuServProc	   *victim = NULL;

			j = (i + k) % numDevAttrs;

			dlist_foreach(iter, &gpuServState->serv_procs_list[j])
			{
				GpuServProc	   *curr = dlist_container(GpuServProc,
													   chain, iter.cur);
				if (only_inactive)
				{
					if (curr->num_clients == 0 &&
						curr->backend_id == InvalidBackendId)
					{
						victim = curr;
						break;
					}
				}
				else if (!victim ||
						 (curr->num_clients +
						  (curr->backend_id != InvalidBackendId ? 1 : 0)) <
						 (victim->num_clients +
						  (victim->backend_id != InvalidBackendId ? 1 : 0)))
				{
					victim = curr;
				}
			}
			/* OK, wake up this victim GPU server */
			if (victim)
			{
				SetLatch(&victim->pgproc->procLatch);
				SpinLockRelease(&gpuServState->lock);
				return;
			}
		}
	} while (--only_inactive >= 0);
	SpinLockRelease(&gpuServState->lock);

	/*
	 * Hmm... we cannot wake up any GPU server processes. However, it means
	 * all processes are now working actively, thus, they eventually pick up
	 * lazy tasks, no problem.
	 */
}

/*
 * gpuservSendCommand - an internal low-level interface
 */
static bool
gpuservSendCommand(GpuContext_v2 *gcontext, GpuServCommand *cmd, long timeout)
{
	struct msghdr	msg;
	struct iovec    iov;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	ssize_t			retval;
	int				ev;
	struct timeval	tv1, tv2;

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

	gettimeofday(&tv1, NULL);
	for (;;)
	{
		GPUSERV_CHECK_FOR_INTERRUPTS();
		ResetLatch(MyLatch);

		ev = WaitLatchOrSocket(MyLatch,
							   WL_SOCKET_WRITEABLE |
							   WL_POSTMASTER_DEATH |
							   (timeout < 0 ? 0 : WL_TIMEOUT),
							   gcontext->sockfd,
							   timeout);
		/* urgent bailout if postmaster is dead */
		if (ev & WL_POSTMASTER_DEATH)
			ereport(FATAL,
					(errcode(ERRCODE_ADMIN_SHUTDOWN),
					 errmsg("Urgent termination by postmaster dead")));
		/* Is the socket writable? */
		if (ev & WL_SOCKET_WRITEABLE)
		{
			retval = sendmsg(gcontext->sockfd, &msg, 0);
			if (retval < 0)
				elog(ERROR, "failed on sendmsg(2): %m");
			else if (retval == 0)
				elog(ERROR, "no bytes sent using sengmsg(2): %m");
			else if (retval != cmd->length)
				elog(ERROR, "incomplete size of message sent: %zu of %zu",
					 retval, (size_t)cmd->length);
			return true;		/* success to send */
		}
		/* check timeout? */
		gettimeofday(&tv2, NULL);
		if (timeout >= 0)
		{
			timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
						(tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
			if (timeout <= 0)
				break;	/* give up to send a command */
		}
		tv1 = tv2;
	}
	elog(ERROR, "failed on sendmsg(2) by timeout on %s side",
		 IsGpuServerProcess() ? "server" : "backend");
}

/*
 * gpuservOpenConnection - open a unix domain socket from the backend
 * (it may fail if no available GPU server)
 */
void
gpuservOpenConnection(GpuContext_v2 *gcontext)
{
	GpuServProc	   *serv_proc = NULL;
	pgsocket		sockfd = PGINVALID_SOCKET;
	cl_int			gpuserv_id;
	cl_int			dindex;
	cl_int			dindex_first;
	long			timeout = 30000;	/* 30s */
	struct timeval	tv1, tv2;

	Assert(!IsGpuServerProcess());
	Assert(gcontext->sockfd == PGINVALID_SOCKET);

	gettimeofday(&tv1, NULL);
	/*
	 * Look up a proper GPU server, according to the policy below
	 *
	 * 1. Determine the device to use
	 * 2. Find an inactive server in the target device
	 * 3. If not found, try to find out an inactive server on the other
	 *    target device.
	 * 4. If not found, try to connect to the active server that has least
	 *    number of clients right now.
	 */
	dindex = pg_atomic_fetch_add_u32(&gpuServState->rr_count, 1) % numDevAttrs;
	dindex_first = dindex;

	SpinLockAcquire(&gpuServState->lock);
	do {
		dlist_iter		iter;

		dlist_foreach(iter, &gpuServState->serv_procs_list[dindex])
		{
			GpuServProc	   *curr = dlist_container(GpuServProc,
												   chain, iter.cur);
			if (curr->num_clients == 0 &&
				curr->backend_id == InvalidBackendId)
			{
				serv_proc = curr;
				break;
			}
		}
		/* try to move other GPU device if no inactive GPU server now */
		dindex = (dindex + 1) % numDevAttrs;
	} while (serv_proc == NULL && dindex != dindex_first);

	if (!serv_proc)
	{
		/*
		 * we have no inactive server now, let's connect to an active server
		 * but with least number of clients.
		 */
		dlist_iter		iter;
		GpuServProc	   *temp = NULL;

		dlist_foreach(iter, &gpuServState->serv_procs_list[dindex_first])
		{
			GpuServProc	   *curr = dlist_container(GpuServProc,
												   chain, iter.cur);
			if (!temp ||
				(temp->num_clients +
				 (temp->backend_id != InvalidBackendId ? 1 : 0)) >
				(curr->num_clients +
				 (curr->backend_id != InvalidBackendId ? 1 : 0)))
				temp = curr;
		}

		/* All the GPU server is now dead? */
		if (!temp)
		{
			SpinLockRelease(&gpuServState->lock);
			elog(ERROR, "No available GPU servers");
		}
		serv_proc = temp;
	}
	gpuserv_id = serv_proc->gpuserv_id;

	/*
	 * If and when connect(2) by other sibling backend is in-progress,
	 * we have to wait for a short moment to get the slot.
	 */
	while (serv_proc->backend_id != InvalidBackendId)
	{
		SpinLockRelease(&gpuServState->lock);

		gettimeofday(&tv2, NULL);
		timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
					(tv1.tv_sec * 1000 + tv2.tv_usec / 1000));
		tv1 = tv2;
		if (timeout < 0)
			elog(ERROR, "open connection by other siblings took too long");
		CHECK_FOR_INTERRUPTS();
		pg_usleep(5000L);	/* 5ms */
		SpinLockAcquire(&gpuServState->lock);
	}
	serv_proc->backend_id = MyBackendId;
	serv_proc->gcontext_id = gcontext->shgcon->context_id;
	SpinLockRelease(&gpuServState->lock);

	PG_TRY();
	{
		/*
		 * Open the connection
		 */
		sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (sockfd < 0)
			elog(ERROR, "failed on socket(2): %m");

		for (;;)
		{
			if (connect(sockfd,
						(struct sockaddr *)&gpuServSockAddr[gpuserv_id],
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
			CHECK_FOR_INTERRUPTS();
		}

		/*
		 * wait for server's accept(2)
		 */
		for (;;)
		{
			int		ev;

			CHECK_FOR_INTERRUPTS();

			ResetLatch(MyLatch);

			if (gcontext->shgcon->server)
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
			if (ev & WL_LATCH_SET)
			{
				if (gcontext->shgcon->server)
					break;
			}

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
		/* release the right for connection, if caller still hold */
		SpinLockAcquire(&gpuServState->lock);
		if (serv_proc->backend_id == MyBackendId)
		{
			serv_proc->backend_id = InvalidBackendId;
			serv_proc->gcontext_id = INVALID_GPU_CONTEXT_ID;
		}
		SpinLockRelease(&gpuServState->lock);

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
	GpuContext_v2  *gcontext = NULL;
	cl_uint			gcontext_id;
	BackendId		backend_id;
	pgsocket		sockfd = PGINVALID_SOCKET;

	Assert(IsGpuServerProcess());

	for (;;)
	{
		sockfd = accept(gpuserv_server_sockfd, NULL, NULL);
		if (sockfd >= 0)
			break;
		if (errno != EINTR)
			elog(ERROR, "failed on accept(2): %m");
		GPUSERV_CHECK_FOR_INTERRUPTS();
	}

	/*
	 * Note of the protocol to open a new session:
	 * 1. Backend looks up a proper GPU server. The "proper" means GPU
	 *    server is either inactive or attached to the leader backend of
	 *    the bgworker which is going to connect.
	 * 2. Backend put its backend-id (itself and its leader; can be same)
	 *    and context-id on gpuServProc of the target GPU server under
	 *    the lock.
	 * 3. Backend calls connect(2) to the GPU server, and then GPU server
	 *    will wake up to accept the connection.
	 * 4. GPU server picks up backend_id and context_id from gpuServProc,
	 *    then reset them, but backend_leader_id is kept. It allows other
	 *    sibling bgworker trying to open a new connection, but prevent
	 *    connection from unrelated backend.
	 * 5. GPU server attach the supplied GPU context with itself, then
	 *    set latch of the backend to wake it up.
	 *
	 * Above is the hand-shaking process between backend and GPU server.
	 *
	 * If and when a bgworker found a proper GPU server but it is still
	 * under the hand-shaking process, the backend has to wait for a short
	 * time.
	 */
	SpinLockAcquire(&gpuServState->lock);
	backend_id  = gpuServProc->backend_id;
	gcontext_id = gpuServProc->gcontext_id;
	gpuServProc->backend_id  = InvalidBackendId;
	gpuServProc->gcontext_id = INVALID_GPU_CONTEXT_ID;
	/* any error on client side before the accept(2)? */
	if (backend_id == InvalidBackendId)
	{
		SpinLockRelease(&gpuServState->lock);
		close(sockfd);
		return;
	}
	gpuServProc->num_clients++;
	SpinLockRelease(&gpuServState->lock);

	PG_TRY();
	{
		PGPROC *backend;
		int		i = session_num_clients;

		/* attach connection to a new GpuContext */
		gcontext = AttachGpuContext(sockfd,
									gcontext_id,
									backend_id,
									devAttrs[gpuserv_cuda_dindex].DEV_ID,
									&gpuServProc->gpu_mem_usage,
									&gpuServProc->gpu_task_count);
		/* expand session's WaitEventSet */
		Assert(i < MaxBackends);
		session_events[i].pos = session_num_clients;
		session_events[i].events = WL_SOCKET_READABLE;
		session_events[i].fd = sockfd;
		session_events[i].user_data = gcontext;
		session_num_clients++;

		AddWaitEventToSet(session_event_set,
						  session_events[i].events,
						  session_events[i].fd,
						  NULL,		/* no latch */
						  session_events[i].user_data);
		/* wake up the backend */
		backend = gcontext->shgcon->backend;
		SetLatch(&backend->procLatch);
		elog(DEBUG2, "GPU server[%d] (pid=%u) accept connection from pid=%u",
			 gpuserv_id, MyProcPid, backend->pid);

	}
	PG_CATCH();
	{
		SpinLockAcquire(&gpuServState->lock);
		gpuServProc->num_clients--;
		SpinLockRelease(&gpuServState->lock);
		/*
		 * Note that the client socket shall be released automatically
		 * once it is attached on the GpuContext.
		 */
		if (!gcontext)
			close(sockfd);
		PG_RE_THROW();
	}
	PG_END_TRY();
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
	int				recvmsg_flags = MSG_DONTWAIT;
	int				peer_fdesc = -1;
	ssize_t			retval;

	/* socket already closed? */
	if (gcontext->sockfd == PGINVALID_SOCKET)
	{
		*p_peer_sock_closed = true;
		return false;
	}

	PG_TRY();
	{
		for (;;)
		{
			peer_fdesc = -1;

			memset(&msg, 0, sizeof(msg));
			memset(&iov, 0, sizeof(iov));
			iov.iov_base = __cmd;
			iov.iov_len  = offsetof(GpuServCommand, error);
			msg.msg_iov = &iov;
			msg.msg_iovlen = 1;
			msg.msg_control = cmsgbuf;
			msg.msg_controllen = sizeof(cmsgbuf);

			retval = recvmsg(gcontext->sockfd, &msg, recvmsg_flags);
			if (retval < 0)
			{
				if (errno == EAGAIN || errno == EWOULDBLOCK)
				{
					/* no messages arrived yes */
					break;
				}
				if (errno == ECONNRESET)
				{
					/* peer socket was closed */
					*p_peer_sock_closed = true;
					break;
				}
				elog(ERROR, "failed on recvmsg(2) %d: %m", errno);
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
					elog(FATAL, "Bug? Only GPU server can receive FD");
				if (cmsg->cmsg_level != SOL_SOCKET ||
					cmsg->cmsg_type != SCM_RIGHTS)
					elog(FATAL, "Bug? unexpected cmsghdr");
				if ((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int) > 1)
					elog(FATAL, "Bug? two or more FDs delivered at once");
				if (CMSG_NXTHDR(&msg, cmsg) != NULL)
					elog(FATAL, "Bug? two or more cmsghdr at once");

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
					{
						gtask->peer_fdesc = peer_fdesc;
						peer_fdesc = -1;
					}
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
					elog(FATAL, "Bug? Only GPU server can deliver ERROR");

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
				elog(ERROR, "Bug? unknown GPUSERV_CMD_* tag: %d",
					 cmd->command);
		}
	}
	PG_CATCH();
	{
		if (peer_fdesc >= 0)
			close(peer_fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return num_received;
}

/*
 * gpuservSendGpuTask - enqueue a GpuTask to the socket
 */
bool
gpuservSendGpuTask(GpuContext_v2 *gcontext, GpuTask_v2 *gtask)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	GpuServCommand	cmd;
	long			timeout = 10000;		/* 5.0sec; usually enough */
	bool			result;

	/* update num_async_tasks */
	SpinLockAcquire(&shgcon->lock);
	if (IsGpuServerProcess())
	{
		shgcon->num_async_tasks--;
		pg_atomic_sub_fetch_u64(shgcon->gpu_task_count, 1);
	}
	else
		shgcon->num_async_tasks++;
	SpinLockRelease(&shgcon->lock);

	cmd.command = GPUSERV_CMD_TASK;
	cmd.length = offsetof(GpuServCommand, error);
	cmd.gtask = gtask;
	result = gpuservSendCommand(gcontext, &cmd, timeout);

	return result;
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
 * gpuserv_session_main - it processes a session once established
 */
static void
gpuserv_session_main(void)
{
	for (;;)
	{
		bool		try_build_cuda_program = false;
		bool		has_pending;
		bool		has_completed;
		int			nr_events = 0;
		WaitEvent	event;

		GPUSERV_CHECK_FOR_INTERRUPTS();
		ResetLatch(MyLatch);

		do {
			/*
			 * Flush out completed tasks (it also releases GPU resource),
			 * then process pending tasks. If any tasks get completed
			 * during task processing, retry this process again.
			 */
			gpuservFlushOutCompletedTasks();

			/*
			 * Process pending tasks if any. Task may want to retry its
			 * processing callback due to out of the resources. Usually,
			 * device resources shall not be releases unless completion
			 * callback.
			 * So, if we have any completed tasks, these tasks shall be
			 * called first. Elsewhere, we move to the synchronous event
			 * wait.
			 *
			 * Even if we have completed tasks, we have to process socket
			 * event with higher priority; to avoid timeout at connect(2)
			 * or sendmsg(2). So, if we found any events without context
			 * blocking, move to the event processing then completed tasks
			 * shall be processed on the next loop.
			 *
			 * We also need to pay attention for the out-of-resource case
			 * by the concurrent GPU server's job. Synchronization of latch
			 * event does not help anything in this case, because we have
			 * no running tasks at this moment, thus, all we can do is
			 * just polling the moment of device resource gets available.
			 * So, we will have blocking event wait with very small timeout
			 * if session_pending_tasks list is not empty.
			 * After the short sleep, other concurrent GPU server may release
			 * relevant GPU resources.
			 */
			gpuservProcessPendingTasks(&has_pending, &has_completed);
			if (has_completed)
			{
				nr_events = WaitEventSetWait(session_event_set,
											 0,
											 &event, 1);
			}
		} while (has_completed && nr_events == 0);

		/*
		 * NOTE: We may get event notification during task processing and
		 * completion callback. In this case, we should not take a side-job
		 * (run-time compile of GPU code), and no need to wait for another
		 * events.
		 */
		if (nr_events == 0)
		{
			/* sleep 10ms if any pending tasks here */
			long	timeout = (has_pending ? 10 : GpuServerCommTimeout);

			/* picks up pending code compile, if no running tasks exist */
			SpinLockAcquire(&session_tasks_lock);
			if (dlist_is_empty(&session_pending_tasks) &&
				dlist_is_empty(&session_running_tasks))
				try_build_cuda_program = true;
			SpinLockRelease(&session_tasks_lock);
			if (try_build_cuda_program)
				pgstrom_try_build_cuda_program();

			/* try to wait for the next event */
			nr_events = WaitEventSetWait(session_event_set,
										 timeout,
										 &event, 1);
		}

		if (nr_events > 0)
		{
			if (event.events & WL_POSTMASTER_DEATH)
				ereport(FATAL,
						(errcode(ERRCODE_ADMIN_SHUTDOWN),
						 errmsg("Urgent termination due to postmaster dead")));

			if (event.events & WL_SOCKET_READABLE)
			{
				if (!event.user_data)
				{
					gpuservAcceptConnection();
				}
				else
				{
					GpuContext_v2  *gcontext = event.user_data;
					bool			peer_sock_closed = false;

					gpuservRecvCommands(gcontext, &peer_sock_closed);
					if (peer_sock_closed)
					{
						if (gcontext->sockfd != PGINVALID_SOCKET)
						{
							if (close(gcontext->sockfd) != 0)
								elog(WARNING, "failed on close(%d) socket: %m",
									 gcontext->sockfd);
							gcontext->sockfd = PGINVALID_SOCKET;
						}
						gpuservPutGpuContext(gcontext);
					}
				}
			}
			pgstrom_cuda_profiler_update();
		}
	}
}

/*
 * pgstrom_bgworker_main - entrypoint of the CUDA server process
 */
static void
gpuserv_bgworker_main(Datum __server_id)
{
	CUresult		rc;
	cl_int			i;
	struct timeval	timeout;

	/* I am a GPU server process */
	gpuserv_id = DatumGetInt32(__server_id);
	Assert(gpuserv_id >= 0 && gpuserv_id < numGpuServers);
	gpuServProc = &gpuServState->serv_procs[gpuserv_id];
	pqsignal(SIGTERM, gpuservGotSigterm);
	BackgroundWorkerUnblockSignals();

	/* cleanup of UNIX domain socket */
	on_proc_exit(gpuservOnExitCleanup, 0);

	/* Open the server socket */
	gpuserv_server_sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (gpuserv_server_sockfd < 0)
		elog(ERROR, "failed on socket(AF_UNIX, SOCK_STREAM, 0): %m");

	if (bind(gpuserv_server_sockfd,
			 (struct sockaddr *)&gpuServSockAddr[gpuserv_id],
			 sizeof(struct sockaddr_un)) != 0)
		elog(ERROR, "failed on bind('%s'): %m",
			 gpuServSockAddr[gpuserv_id].sun_path);

	if (listen(gpuserv_server_sockfd, numGpuServers) != 0)
		elog(ERROR, "failed on listen(2): %m");

	/* assign reasonably short timeout for accept(2) */
	timeout.tv_sec = 0;
	timeout.tv_usec = 400 * 1000;	/* 400ms */
	if (setsockopt(gpuserv_server_sockfd,
				   SOL_SOCKET, SO_RCVTIMEO,
				   &timeout, sizeof(timeout)) != 0)
		elog(ERROR, "failed on setsockopt(2): %m");

	/* Init CUDA runtime */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	gpuserv_cuda_dindex = gpuserv_id % numDevAttrs;
	rc = cuDeviceGet(&gpuserv_cuda_device,
					 devAttrs[gpuserv_cuda_dindex].DEV_ID);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&gpuserv_cuda_context,
					 CU_CTX_SCHED_AUTO,
					 gpuserv_cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxCreate: %s", errorText(rc));

	for (i=0; i < CUDA_MODULES_SLOTSIZE; i++)
		dlist_init(&cuda_modules_slot[i]);

	pgstrom_cuda_profiler_init();

	/* memory context per session duration */
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "GPU Server");
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "GPU Server per session",
												 ALLOCSET_DEFAULT_MINSIZE,
												 ALLOCSET_DEFAULT_INITSIZE,
												 ALLOCSET_DEFAULT_MAXSIZE);
	/* init session status */
	SpinLockInit(&session_tasks_lock);
	dlist_init(&session_pending_tasks);
	dlist_init(&session_running_tasks);
	dlist_init(&session_completed_tasks);
	session_events = MemoryContextAllocZero(TopMemoryContext,
											sizeof(WaitEvent) * MaxBackends);
	reconstructSessionEventSet();

	/* register myself on the shared GpuServProc structure */
	SpinLockAcquire(&gpuServState->lock);
	gpuServProc->gpuserv_id = gpuserv_id;
	gpuServProc->num_clients = 0;
	gpuServProc->pgproc = MyProc;
	gpuServProc->backend_id = InvalidBackendId;
	gpuServProc->gcontext_id = INVALID_GPU_CONTEXT_ID;
	pg_atomic_init_u64(&gpuServProc->gpu_mem_usage, 0);
	pg_atomic_init_u64(&gpuServProc->gpu_task_count, 0);
	dlist_push_head(&gpuServState->serv_procs_list[gpuserv_cuda_dindex],
					&gpuServProc->chain);
	SpinLockRelease(&gpuServState->lock);

	elog(LOG, "PG-Strom GPU/CUDA Server [%d] is now ready on GPU-%d %s",
		 gpuserv_id,
		 devAttrs[gpuserv_cuda_dindex].DEV_ID,
		 devAttrs[gpuserv_cuda_dindex].DEV_NAME);

	PG_TRY();
	{
		gpuserv_session_main();
	}
	PG_CATCH();
	{
		/*
		 * An exception happen during the GpuTask execution.
		 * This background worker will exit once, thus no need to release
		 * individual local resources, but has to revert shared resources
		 * not to have incorrect status.
		 */

		/* this bgworker process goes to die */
		SpinLockAcquire(&gpuServState->lock);
		gpuServProc->pgproc = NULL;
		gpuServProc->backend_id = InvalidBackendId;
		gpuServProc->gcontext_id = INVALID_GPU_CONTEXT_ID;
		dlist_delete(&gpuServProc->chain);
		memset(&gpuServProc->chain, 0, sizeof(dlist_node));
		SpinLockRelease(&gpuServState->lock);

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
		for (i=0; i < session_num_clients; i++)
		{
			GpuContext_v2  *gcontext = session_events[i].user_data;

			ForcePutGpuContext(gcontext);
		}
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
	dlist_iter	iter;
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
			dlist_foreach(iter, &gpuServState->serv_procs_list[i])
			{
				GpuServProc	   *serv = dlist_container(GpuServProc,
													   chain, iter.cur);
				SetLatch(&serv->pgproc->procLatch);
			}
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
 * gpuservShmemRequired - required size of static shared memory
 */
static inline Size
gpuservShmemRequired(void)
{
	Size	required;

	required = (MAXALIGN(offsetof(GpuServState, serv_procs[numGpuServers])) +
				/* serv_procs_list */
				MAXALIGN(sizeof(dlist_head) * numDevAttrs));
	return required;
}

/*
 * pgstrom_startup_gpu_server
 */
static void
pgstrom_startup_gpu_server(void)
{
	Size		required = gpuservShmemRequired();
	int			i;
	char	   *pos;
	bool		found;
	static bool	__pgstrom_cuda_profiler;	/* dummy */

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* request for the static shared memory */
	gpuServState = ShmemInitStruct("gpuServState", required, &found);
	Assert(!found);

	memset(gpuServState, 0, required);
	SpinLockInit(&gpuServState->lock);
	pos = ((char *)gpuServState +
		   MAXALIGN(offsetof(GpuServState, serv_procs[numGpuServers])));
	/* serv_procs_list */
	gpuServState->serv_procs_list = (dlist_head *) pos;
	pos += MAXALIGN(sizeof(dlist_head) * numDevAttrs);
	for (i=0; i < numDevAttrs; i++)
		dlist_init(&gpuServState->serv_procs_list[i]);

	Assert((char *)gpuServState + required == pos);

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
	Size		refGpuMemSz = SIZE_MAX;
	Size		refCpuMemSz = SIZE_MAX;
	long		sysconf_pagesize;	/* _SC_PAGESIZE */
	long		sysconf_phys_pages;	/* _SC_PHYS_PAGES */
	cl_uint		i, nr_async;
	char		path[MAXPGPATH];
	char		config_file[MAXPGPATH];

	/*
	 * Maximum number of GPU servers we can use concurrently.
	 * (it is equivalent to the number of CUDA contexts)
	 */
	DefineCustomIntVariable("pg_strom.num_gpu_servers",
							"number of GPU/CUDA intermediation servers",
							NULL,
							&numGpuServers,
							2,
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
	 * pg_strom.max_async_tasks_per_device
	 *
	 * Number of maximum asynchronous tasks per GPU device.
	 * Its default value is adjusted based on:
	 * - amount of GPU device memory
	 * - amount of CPU host memory
	 * - number of GPU servers
	 */
	for (i=0; i < numDevAttrs; i++)
		refGpuMemSz = Min(refGpuMemSz, devAttrs[i].DEV_TOTAL_MEMSZ);
	if (refGpuMemSz >= (16UL<<30))		/* 1/3 of > 16GB portion */
		refGpuMemSz = (refGpuMemSz - (16UL<<30)) / 3 + (11UL<<30);
	else if (refGpuMemSz >= (10UL<<30))	/* 1/2 of > 10GB portion */
		refGpuMemSz = (refGpuMemSz - (10UL<<30)) / 2 + (8UL<<30);
	else if (refGpuMemSz >= (4UL<<30))	/* 2/3 of > 4GB portion */
		refGpuMemSz = (refGpuMemSz - (4UL<<30)) * 2 / 3 + (4UL<<30);

	sysconf_pagesize = sysconf(_SC_PAGESIZE);
	if (sysconf_pagesize < 0)
		elog(ERROR, "failed on sysconf(_SC_PAGESIZE): %m");
	sysconf_phys_pages = sysconf(_SC_PHYS_PAGES);
	if (sysconf_phys_pages < 0)
		elog(ERROR, "failed on sysconf(_SC_PHYS_PAGES): %m");
	refCpuMemSz = (sysconf_pagesize * sysconf_phys_pages -
				   (Size)NBuffers * (Size)BLCKSZ) / 3;
	nr_async = (Min(refGpuMemSz / (2 * pgstrom_chunk_size()),
					refCpuMemSz / (4 * pgstrom_chunk_size())) +
				4 * numGpuServers);

	DefineCustomIntVariable("pg_strom.max_async_tasks_per_device",
					"Soft limit for number of concurrent tasks per GPU device",
							NULL,
							&pgstrom_max_async_tasks_per_device,
							nr_async,
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/*
	 * Setup pathname of the UNIX domain sockets, for each GPU servers
	 */
	gpuServSockAddr = malloc(sizeof(struct sockaddr_un) * numGpuServers);
	if (!gpuServSockAddr)
		elog(ERROR, "out of memory");

	for (i=0; i < numGpuServers; i++)
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
	for (i=0; i < numGpuServers; i++)
	{
		BackgroundWorker	worker;

		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "PG-Strom GPU/CUDA Server [%d]", i);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = 1;
		worker.bgw_main = gpuserv_bgworker_main;
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* request for the static shared memory */
	RequestAddinShmemSpace(gpuservShmemRequired());
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
