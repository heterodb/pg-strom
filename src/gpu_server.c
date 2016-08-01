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
#include "access/twophase.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"
#include "cuda_dynpara.h"

#include <sys/socket.h>
#include <sys/un.h>

typedef struct GpuServSocket
{
	pgsocket		sockfd;
	struct sockaddr_un sockadd;
} GpuServSocket;

typedef struct GpuServProc
{
	dlist_node		chain;		/* link to serv_procs_list */
	BackendId		backend_leader_id;	/* backend id of the leader */
	BackendId		backend_id;	/* backend which is going to connect to */
	cl_uint			gcontext_id;/* gcontext which is going to be assigned to */
	PGPROC		   *pgproc;		/* reference to the server's PGPROC */
} GpuServProc;

#define GPUSERV_IS_ACTIVE(gsproc)								\
	((gsproc)->pgproc && (gsproc)->backend_id != InvalidBackendId)

typedef struct GpuServState
{
	pg_atomic_uint32 rr_count;	/* seed for round-robin distribution */
	slock_t			lock;
	Bitmapset	  **gpumem_waiters;		/* for each device */
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
	union {
		struct {
			GpuTask_v2 *gtask;	/* to be located on the DMA buffer */
		} task;
		struct {
			cl_int		elevel;
			cl_int		sqlerrcode;
			cl_uint		filename_offset;
			cl_int		lineno;
			cl_uint		funcname_offset;
			cl_uint		message_offset;
			char		buffer[FLEXIBLE_ARRAY_MEMBER];
		} error;
	} u;
} GpuServCommand;

/*
 * public variables
 */
SharedGpuContext	   *currentSharedGpuContext = NULL;

/*
 * static/public variables
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static GpuServSocket   *gpuServSocket = NULL;	/* const */
static GpuServState	   *gpuServState = NULL;	/* shmem */
static GpuServProc	   *gpuServProc = NULL;		/* shmem */
static int				numGpuServers;			/* GUC */
static int				GpuServerCommTimeout;	/* GUC */
static bool				gpuserv_got_sigterm = false;
static int				gpuserv_id = -1;
static int				gpuserv_dindex = -1;
static pgsocket			gpuserv_server_sockfd = PGINVALID_SOCKET;
CUdevice				gpuserv_cuda_device = NULL;
CUcontext				gpuserv_cuda_context = NULL;
/* GPU server session info */
static slock_t			session_tasks_lock;
static dlist_head		session_pending_tasks;
static dlist_head		session_running_tasks;
static dlist_head		session_completed_tasks;
static cl_int			session_num_clients;
static GpuContext_v2   *session_gcontexts = NULL;
static WaitEventSet	   *session_event_set = NULL;

/*
 * static functions
 */
static bool gpuservSendCommand(GpuContext_v2 *gcontext,
							   GpuServCommand *cmd, long timeout);
static bool gpuservRecvCommand(pgsocket sockfd,
							   GpuServCommand *cmd, long timeout);

/* SIGTERM handler */
static void
gpuservGotSigterm(SIGNAL_ARGS)
{
	int		save_errno = errno;

	gpuserv_got_sigterm = true;

	SetLatch(MyLatch);

	errno = save_errno;
}

/*
 * IsGpuServerProcess - returns true, if current process is gpu server
 */
bool
IsGpuServerProcess(void)
{
	return (bool)(gpuserv_id);
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
ReportErrorForBackend(GpuTask_v2 *gtask,
					  GpuContext_v2 *gcontext,
					  MemoryContext memcxt)
{
	GpuServCommand	cmd;
	MemoryContext	oldcxt;
	ErrorData	   *errdata;
	Size			required;
	char		   *buf;
	cl_uint			offset = 0;
	long			timeout = 5000;		/* 5.0sec */
	struct timeval	tv1, tv2;

	/*
	 * Prior to the ereport() to exit the server once, we deliver
	 * the error detail to the backend to cause an ereport on the
	 * backend side also.
	 */
	oldcxt = MemoryContextSwitchTo(memcxt);
	errdata = CopyErrorData();

	required = (offsetof(GpuServCommand, u.error.buffer) +
				MAXALIGN(strlen(errdata->filename) + 1) +
				MAXALIGN(strlen(errdata->funcname) + 1) +
				MAXALIGN(strlen(errdata->message) + 1));
	cmd = palloc(required);
	cmd->command            = GPUSERV_CMD_ERROR;
	cmd->length             = required;
	cmd->u.error.elevel     = errdata->elevel;
	cmd->u.error.sqlerrcode = errdata->sqlerrcode;

	cmd->u.error.filename_offset = offset;
	strcpy(cmd->u.error.buffer + offset, errdata->filename);
	offset += MAXALIGN(strlen(errdata->filename) + 1);

	cmd->u.error.lineno     = errdata->lineno;

	cmd->u.error.funcname_offset = offset;
	strcpy(cmd->u.error.buffer + offset, errdata->funcname);
	offset += MAXALIGN(strlen(errdata->funcname) + 1);

	cmd->u.error.message_offset = offset;
	strcpy(cmd->u.error.buffer + offset, errdata->message);
	offset += MAXALIGN(strlen(errdata->message) + 1);

	/*
	 * Urgent return to the backend
	 */
	gettimeofday(&tv1, NULL);
	for (;;)
	{
		CHECK_FOR_INTERRUPTS();
		ResetLatch(MyLatch);

		if (gpuservSendCommand(gcontext, cmd, timeout))
			break;



		if (gpuservSendCommand(gpuserv_gpu_context, &cmd, timeout))
			break;

		/* adjust timeout */
		gettimeofday(&tv2, NULL);
		if (timeout >= 0)
		{
			timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
						(tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
			if (timeout <= 0)
				break;	/* give up to ereport, but logged on server side */
		}
		tv1 = tv2;
	}

	MemoryContextSwitchTo(oldcxt);
}

/*
 * gpuservProcessPendingTasks
 */
static void
gpuservProcessPendingTasks(void)
{
	SharedGpuContext *shgcon = gpuserv_gpu_context->shgcon;
	MemoryContext	memcxt = CurrentMemoryContext;
	dlist_node	   *dnode;
	GpuTask_v2	   *gtask;

	SpinLockAcquire(&session_tasks_lock);
	while (!dlist_is_empty(&session_pending_tasks))
	{
		CUmodule	cuda_module;
		CUstream	cuda_stream;
		CUresult	rc;
		cl_int		retval;

		dnode = dlist_pop_head_node(&session_pending_tasks);
		gtask = dlist_container(GpuTask_v2, chain, dnode);
		SpinLockRelease(&session_tasks_lock);

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
				retval = 1;
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
				/*
				 * pgstromProcessGpuTask() may return false, if GpuTask
				 * cannot acquire enough amount of GPU memory. In this
				 * case, we will retry the task after the free of GPU
				 * memory.
				 */
				if (retval > 0)
				{
					Bitmapset  *waiters;

					SpinLockAcquire(&gpuServState->lock);
					waiters = gpuServState->gpumem_waiters[shgcon->device_id];
					waiters->words[WORDNUM(MyProc->pgprocno)]
						|= (1 << BITNUM(MyProc->pgprocno));
					SpinLockRelease(&gpuServState->lock);
				}
			}
		}
		PG_CATCH();
		{
			ReportErrorForBackend(gtask, memcxt);
			PG_RE_THROW();
		}
		PG_END_TRY();

		/*
		 * When we give up to kick a GpuTask due to lack of GPU memory,
		 * we once break processing the pending tasks, even if we still
		 * have multiple tasks in the pending list.
		 * Process shall be waken up by gpuMemFree().
		 */
		SpinLockAcquire(&session_tasks_lock);
		if (retval == 0)
			dlist_push_tail(&session_running_tasks, &gtask->chain);
		else if (retval > 0)
		{
			dlist_push_head(&session_pending_tasks, &gtask->chain);
			break;
		}
	}
	SpinLockRelease(&session_tasks_lock);
}

/*
 * flushout_completed_tasks
 */
static void
gpuservFlushOutCompletedTasks(void)
{
	MemoryContext	memcxt = CurrentMemoryContext;
	dlist_node	   *dnode;
	GpuTask_v2	   *gtask;
	int				retval;
	CUresult		rc;

	SpinLockAcquire(&session_tasks_lock);
	while (!dlist_is_empty(&session_completed_tasks))
	{
		dnode = dlist_pop_head_node(&session_completed_tasks);
		gtask = dlist_container(GpuTask_v2, chain, dnode);
		memset(&gtask->chain, 0, sizeof(dlist_node));
		SpinLockRelease(&session_tasks_lock);

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
				if (!gpuservSendGpuTask(gpuserv_gpu_context, gtask))
					elog(ERROR, "failed on gpuservSendGpuTask");
			}
			else if (retval > 0)
			{
				/*
				 * GpuTask wants to execute GPU kernel again, so attach it
				 * on the pending list again.
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
				 */
				SharedGpuContext *shgcon = gpuserv_gpu_context->shgcon;

				pgstromReleaseGpuTask(gtask);

				SpinLockAcquire(&shgcon->lock);
				shgcon->num_async_tasks--;
				SpinLockRelease(&shgcon->lock);

				SetLatch(&shgcon->backend->procLatch);
			}
		}
		PG_CATCH();
		{
			ReportErrorForBackend(gtask, memcxt);
			PG_RE_THROW();
		}
		PG_END_TRY();

		SpinLockAcquire(&session_tasks_lock);
	}
	SpinLockRelease(&session_tasks_lock);
}

/*
 * gpuservHandleLazyJobs
 */
void
gpuservHandleLazyJobs(bool flushout_completed, bool process_pending)
{
	/* Exit, if SIGTERM was delivered */
	if (gpuserv_got_sigterm)
		ereport(FATAL,
				(errcode(ERRCODE_ADMIN_SHUTDOWN),
				 errmsg("Terminating PG-Strom GPU/CUDA Server[%d]",
						gpuserv_id)));
	/* flush out completed tasks, if any */
	if (flushout_completed)
		gpuservFlushOutCompletedTasks();
	/* process pending tasks, if any */
	if (process_pending)
		gpuservProcessPendingTasks();
	/* flush out completed tasks again, if any */
	if (flushout_completed)
		gpuservFlushOutCompletedTasks();
	/* build a new CUDA program, if any */
	pgstrom_try_build_cuda_program();
}

/*
 * gpuservWakeUpProc - wakes up a sleeping GPU server process
 */
void
gpuservWakeUpProc(void)
{
	dlist_head	   *dhead;
	dlist_node	   *dnode;
	GpuServProc	   *serv_proc;
	PGPROC		   *pgproc;
	int				i, j;

	SpinLockAcquire(&gpuServState->lock);
	/* wake up inactive server process first */
	for (i=0; i < numDevAttrs; i++)
	{
		dhead = &gpuServState->serv_procs_list[j];
		if (dlist_is_empty(dhead))
			continue;
		for (dnode = dlist_tail_node(dhead);
			 dnode != NULL;
			 dnode = (dlist_has_prev(dhead, dnode)
					  ? dlist_prev_node(dhead, dnode) : NULL))
		{
			serv_proc = dlist_container(GpuServProc, chain, dnode);
			/* likely, a dead GPU server process */
			if (!serv_proc->pgproc)
				continue;
			/* an active GPU server process */
			if (serv_proc->backend_id != InvalidBackendId)
				break;
			/* skip inactive process latches are already set */
			pg_memory_barrier();
			if (pgproc->procLatch.is_set)
				continue;
			/* OK, wake up an inactive GPU server process */
			SetLatch(&serv_proc->pgproc->procLatch);

			SpinLockRelease(&gpuServState->lock);
			return;
		}
	}

	/* wake up active server process, if no inactive process exists */
	for (i=0; i < numDevAttrs; i++)
	{
		dhead = &gpuServState->serv_procs_list[j];
		if (dlist_is_empty(dhead))
			continue;
		for (dnode = dlist_head_node(dhead);
			 dnode != NULL;
			 dnode = (dlist_has_next(dhead, dnode)
					  ? dlist_next_node(dhead, dnode) : NULL))
		{
			serv_proc = dlist_container(GpuServProc, chain, dnode);
			/* likely, a dead GPU server process */
			if (!serv_proc->pgproc)
				continue;
			/* an inactive GPU server process we already tried above */
			if (serv_proc->backend_id == InvalidBackendId)
				break;
			/* skip inactive process latches are already set */
			pg_memory_barrier();
			if (pgproc->procLatch.is_set)
				continue;
			/* OK, wake up an inactive GPU server process */
			SetLatch(&serv_proc->pgproc->procLatch);

			SpinLockRelease(&gpuServState->lock);
			return;
		}
	}
	SpinLockRelease(&gpuServState->lock);

	/*
	 * Hmm... we cannot wake up any GPU server processes. However, it means
	 * all processes are now working actively, thus, they eventually pick up
	 * lazy tasks, no problem.
	 */
}

/*
 * notifierGpuMemFree - wake up processes that wait for free GPU RAM
 */
void
notifierGpuMemFree(cl_int device_id)
{
	Bitmapset  *waiters;
	PGPROC	   *pgproc;
	int			prev = -1;

	Assert(device_id >= 0 && device_id < numDevAttrs);

	SpinLockAcquire(&gpuServState->lock);
	waiters = gpuServState->gpumem_waiters[device_id];
	while ((prev = bms_next_member(waiters, prev)) >= 0)
	{
		Assert(prev < ProcGlobal->allProcCount);
		pgproc = &ProcGlobal->allProcs[prev];
		SetLatch(&pgproc->procLatch);
	}
	memset(waiters->words, 0, sizeof(bitmapword) * waiters->nwords);
	SpinLockRelease(&gpuServState->lock);
}




/*
 * gpuservSendCommand - an internal low-level interface
 */
static void
gpuservSendCommand(GpuContext_v2 *gcontext, GpuServCommand *cmd, long timeout)
{
	struct msghdr	msg;
	struct iovec	iov[2];
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	ssize_t			retval;
	int				ev;

	memset(&msg, 0, sizeof(struct msghdr));
	memset(iov, 0, sizeof(iov));
	if (cmd->command == GPUSERV_CMD_TASK)
	{
		GpuTask_v2 *gtask = cmd->u.task.gtask;

		Assert(dmaBufferValidatePtr(gtask));

		msg.msg_iov = iov;
		msg.msg_iovlen = 1;
		iov[0].iov_base = cmd;
		iov[0].iov_len = (offsetof(GpuServCommand, u.task) +
						  sizeof(cmd->u.task));
		/* Is a file-descriptor attached on the command? */
		if (!IsGpuServerProcess() && gtask->file_desc >= 0)
		{
			struct cmsghdr *cmsg;

			msg.msg_control = cmsgbuf;
			msg.msg_controllen = sizeof(cmsgbuf);

			cmsg = CMSG_FIRSTHDR(&msg);
			cmsg->cmsg_level = SOL_SOCKET;
			cmsg->cmsg_type = SCM_RIGHTS;
			cmsg->cmsg_len = CMSG_LEN(sizeof(int));
			((int *)CMSG_DATA(cmsg))[0] = gtask->file_desc;
		}
		Assert(cmd->length == iov[0].iov_len);
	}
	else if (cmd->command == GPUSERV_CMD_ERROR)
	{
		Assert(IsGpuServerProcess());

		msg.msg_iov = iov;
		msg.msg_iovlen = 2;
		iov[0].iov_base = cmd;
		iov[0].iov_len = offsetof(GpuServCommand, u.error.buffer);
		iov[1].iov_base = cmd->u.error.buffer;
		iov[1].iov_len = cmd->length - iov[0].iov_len;
	}
	else
		elog(ERROR, "Bug? unexpected GPUSERV_CMD_* tag: %d", cmd->command);

	gettimeofday(&tv1, NULL);
	for (;;)
	{
		CHECK_FOR_INTERRUPTS();
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
			else if (retval != iov[0].iov_len + iov[1].iov_len)
				elog(ERROR, "incomplete size of message sent: %zu of %zu",
					 retval, iov[0].iov_len, iov[1].iov_len);
			return;		/* success to send */
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
	elog(ERROR, "failed on sendmsg(2) by timeout");
}

/*
 * gpuservSendGpuTask - enqueue a GpuTask to the socket
 */
void
gpuservSendGpuTask(GpuContext_v2 *gcontext, GpuTask_v2 *gtask)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	GpuServCommand	cmd;
	long			timeout = 5000;		/* 5.0sec; usually enough */

	cmd.command = GPUSERV_CMD_TASK;
	cmd.length = offsetof(GpuServCommand, u.task) + sizeof(cmd->u.task);
	cmd.u.task.gtask = gtask;

	gpuservSendCommand(gcontext, &cmd, timeout);

	/* update num_async_tasks */
	SpinLockAcquire(&shgcon->lock);
	if (IsGpuServerProcess())
		shgcon->num_async_tasks--;
	else
		shgcon->num_async_tasks++;
	SpinLockRelease(&shgcon->lock);
}

/*
 * gpuserv_open_connection - open a unix domain socket from the backend
 * (it may fail if no available GPU server)
 */
bool
gpuservOpenConnection(GpuContext_v2 *gcontext)
{
	GpuServProc	   *serv_proc;
	pgsocket		sockfd = PGINVALID_SOCKET;
	BackendId		MyBackendLeaderId;
	cl_int			dindex;
	cl_int			dindex_first;
	cl_long			timeout = 400;	/* up to 400ms for connect(2) */
	struct timeval	tv1, tv2;

	Assert(!IsGpuServerProcess());
	Assert(gcontext->sockfd == PGINVALID_SOCKET);

	gettimeofday(&tv1, NULL);

	/* determine the device we use */
	dindex = pg_atomic_fetch_add_u32(&gpuServState->rr_count) % numDevAttrs;
	dindex_first = dindex;

	MyBackendLeaderId = (ParallelMasterBackendId == InvalidBackendId
						 ? MyBackendId
						 : ParallelMasterBackendId);
	/* look up a proper GPU server */
	SpinLockAcquire(&gpuServState->lock);
retry_lookup:
	serv_proc = NULL;
	for (;;)
	{
		dlist_iter		iter;

		dlist_foreach(iter, &gpuServState->serv_procs_list[dindex])
		{
			GpuServProc	   *curr = dlist_container(GpuServProc,
												   chain, iter.cur);
			if (curr->backend_leader_id == InvalidBackendId)
				serv_proc = curr;	/* candidate of inactive server */
			else if (curr->backend_leader_id == MyBackendLeaderId)
			{
				serv_proc = curr;
				break;
			}
		}

		if (serv_proc)
			break;

		/* try to connect other device if no available GPU server */
		dindex = (dindex + 1) % numDevAttrs;
		if (dindex == dindex_first)
		{
			SpinLockRelease(&gpuServState->lock);
			elog(NOTICE, "no available GPU server");
			return false;
		}
	}

	/*
	 * If connect(2) - accept(2) by other sibling backend is in-progress,
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
		{
			elog(NOTICE, "time out for connection to GPU server");
			return false;
		}
		pg_usleep(5000L);	/* 5ms */
		CHECK_FOR_INTERRUPTS();
		SpinLockAcquire(&gpuServState->lock);
		/*
		 * A corner case - the other sibling backend closed the session
		 * soon, then GPU server might be attached to unrelated backend
		 * during the short sleep. In this case, we retry lookup.
		 */
		if (serv_proc->backend_leader_id != InvalidBackendId &&
			serv_proc->backend_leader_id != MyBackendLeaderId)
		{
			dindex = dindex_first;
			goto retry_lookup;
		}
	}
	gpuserv_id = serv_proc->gpuserv_id;
	serv_proc->backend_leader_id = MyBackendLeaderId;
	serv_proc->backend_id = MyBackendId;
	serv_proc->context_id = gcontext->shgcon->context_id;
	SpinLockRelease(&gpuServState->lock);

	/*
	 * open the connection
	 */
	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");

	for (;;)
	{
		if (connect(sockfd,
					(struct sockaddr *)&gpuServSocket[gpuserv_id].sockaddr,
					sizeof(struct sockaddr_un)) == 0)
			break;

		if (errno != EINTR)
		{
			close(sockfd);
			elog(ERROR, "failed on connect(2): %m");
		}
		gettimeofday(&tv2, NULL);
		timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
					(tv1.tv_sec * 1000 + tv2.tv_usec / 1000));
		tv1 = tv2;
		if (timeout < 0)
		{
			close(sockfd);
			elog(NOTICE, "time out for connection to GPU server");
			return false;
		}
	}

	/*
	 * wait for server's accept(2)
	 */
	for (;;)
	{
		ResetLatch(MyLatch);

		rc = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_TIMEOUT |
					   WL_POSTMASTER_DEATH,
					   timeout);
		if (rc & WL_POSTMASTER_DEATH)
			ereport(FATAL,
					(errcode(ERRCODE_ADMIN_SHUTDOWN),
					 errmsg("Urgent termination by postmaster dead")));
		if (rc & WL_LATCH_SET)
			break;

		close(sockfd);
		elog(NOTICE, "timeout for accept(2) by the GPU server");
		return false;
	}

	/*
	 * check status of the connection
	 */
	if (!gcontext->shgcon->server)
	{
		close(sockfd);
		elog(ERROR, "Bug? server's PGPROC was not set correctly");
	}

	gcontext->sockfd;
	return true;		/* OK, connection established */
}

/*
 * gpuservAcceptConnection - accept a new client connection
 */
static void
gpuservAcceptConnection(void)
{
	GpuServCommand	cmd;
	GpuContext_v2  *gcontext = NULL;
	cl_uint			gcontext_id;
	BackendId		backend_id;
	pgsocket		sockfd = PGINVALID_SOCKET;

	Assert(IsGpuServerProcess());

	for (;;)
	{
		sockfd = accept(gpuserv_server_sockfd);
		if (sockfd >= 0)
			break;
		if (errno != EINTR)
			elog(ERROR, "failed on accept(2): %m");
		CHECK_FOR_INTERRUPTS();
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
	/* move to the list head because of its activeness */
	dlist_move_head(&gpuServState->serv_procs_list[gpuserv_dindex],
					&gpuServProc->chain);
	SpinLockRelease(&gpuServState->lock);

	PG_TRY();
	{
		/* expand session_gcontexts */
		session_gcontexts = repalloc(session_gcontexts,
									 sizeof(GpuContext_v2 *) *
									 (session_num_clients + 1));
		/* expand session_event_set on demand */
		if (session_event_set->nevents >= session_event_set->nevents_space)
		{
			WaitEventSet *new_event_set;
			int		new_nevents_space = 2 * ession_event_set->nevents_space;

			new_event_set = CreateWaitEventSet(TopMemoryContext,
											   new_nevents_space);
			for (i=0; i < session_event_set->nevents; i++)
			{
				WaitEvent  *ev = &session_event_set->events[i];

				AddWaitEventToSet(new_event_set,
								  ev->events,
								  ev->fd,
								  i == 0 ? MyLatch : NULL,
								  ev->user_data);
			}
			FreeWaitEventSet(session_event_set);
			session_event_set = new_event_set;
		}

		/* attach connection to a new GpuContext */
		gcontext = AttachGpuContext(sockfd,
									gcontext_id,
									backend_id,
									devAttrs[gpuserv_dindex].DEV_ID);
		session_gcontexts[session_num_clients++] = gcontext;
		AddWaitEventToSet(session_event_set,
						  WL_SOCKET_READABLE,
						  sockfd,
						  NULL,
						  gcontext);

		/* wake up the backend */
		SetLatch(gcontext->shgcon->backend);
	}
	PG_CATCH();
	{
		/* Note that client socket shall be released automatically once
		 * it is attached to GpuContext.
		 */
		if (!gcontext)
			close(sockfd);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * gpuservCloseConnection
 */
static void
gpuservCloseConnection(GpuContext_v2 *gcontext)
{

	Assert(IsGpuServerProcess());

	/* remove this session from the WaitEventSet */
	for (i=0; i < session_event_set->nevents; i++)
	{
		WaitEvent  *event = &session_event_set->events[i];

		if (event->user_data != gcontext)
			continue;

		if (i + 1 < session_event_set->nevents)
			memmove(session_event_set->events + i,
					session_event_set->events + i + 1,
					sizeof(WaitEvent) *
					(session_event_set->nevents - (i + 1)));
		session_event_set->nevents--;
		Assert(session_event_set->nevents > 0);

		/* The first WaitEvent should not be changed */
		event = &session_event_set->events[0];
		if (event->user_data != NULL ||
			event->fd != PGINVALID_SOCKET ||
			event->events != (WL_POSTMASTER_DEATH |
							  WL_LATCH_SET |
							  WL_SOCKET_READABLE))
			elog(ERROR, "Bug? primary WaitEvent was destroyed");

		/*
		 * Close the session.
		 *
		 * PutGpuContext() will also close the socket, however, peer socket
		 * is already closed at this point. If any asynchronous tasks are
		 * still running, it is waste of time to wait for ready of socket
		 * writable. Thus, we close the socket immediately.
		 * The completed tasks shall be reclaimed, simply.
		 */
		if (gcontext->sockfd != PGINVALID_SOCKET)
		{
			if (close(gcontext->sockfd) != 0)
				elog(WARNING, "failed on close(%d) socket: %m",
					 gcontext->sockfd);
			gcontext->sockfd = PGINVALID_SOCKET;
		}
		PutGpuContext(gcontext);

		/*
		 * Inactivates this GPU server process
		 *
		 * If supplied GpuContext is the last session of the group of backends,
		 * this GPU server can accept connection from any other unrelated
		 * sessions again.
		 */
		if (session_event_set->nevents == 1)
		{
			SpinLockAcquire(&gpuServState->lock);
			if (gpuServProc->backend_id != InvalidBackendId)
			{
				/*
				 * NOTE: Oh, it is an extreme corver case. Last session was
				 * closed just moment before, however, a new sibling backend
				 * trying to accept(2).
				 * Next WaitEventSetWait() will return immediately because
				 * the backend already issued connect(2) system call. So, we
				 * don't need to inactivate this GPU server.
				 */
			}
			else
			{
				gpuServProc->backend_leader_id = InvalidBackendId;
				gpuServProc->backend_id = InvalidBackendId;
				gpuServProc->gcontext_id = INVALID_GPU_CONTEXT_ID;
				dlist_delete(&gpuServProc->chain);
				dlist_push_tail(&gpuServState->serv_procs_list[gpuserv_dindex],
								&gpuServProc->chain);
			}
			SpinLockRelease(&gpuServState->lock);
		}
		return;
	}
	elog(FATAL, "Bug? GPU server misses GpuContext");
}

/*
 * gpuservRecvCommands
 */
static bool
gpuservRecvCommands(GpuContext_v2 *gcontext)
{
	pgsocket		sockfd = gcontext->sockfd;
	struct msghdr	msg;
	struct iovec	iov;
	struct cmsghdr *cmsg;
	unsigned char	msgbuf[2 * sizeof(GpuServCommand)];
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	int				peer_fdesc = -1;
	bool			result = false;
	ssize_t			retval;

	/* fetch messages from the socket */
	memset(&msg, 0, sizeof(msg));
	memset(&iov, 0, sizeof(iov));
	iov.iov_base = msgbuf;
	iov.iov_len = sizeof(GpuServCommand);
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	msg.msg_control = cmsgbuf;
	msg.msg_controllen = sizeof(cmsgbuf);

	retval = recvmsg(sockfd, &msg, 0);
	if (retval < 0)
		elog(ERROR, "failed on recvmsg(2): %m");

	/* pick up peer fdesc, if any */
	if ((cmsg = CMSG_FIRSTHDR(&msg)) != NULL)
	{
		/* Only GPU server can receive SCM_RIGHTS message */
		Assert(IsGpuServerProcess());

		if (cmsg->cmsg_level != SOL_SOCKET ||
			cmsg->cmsg_type != SCM_RIGHTS)
			elog(FATAL, "unexpected cmsghdr {cmsg_level=%d cmsg_type=%d}",
				 cmsg->cmsg_level, cmsg->cmsg_type);
		/* needs to exit once then restart server */
		if ((cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int) > 1)
			elog(FATAL, "we cannot handle two or more FDs at once");
		if (CMSG_NXTHDR(&msg, cmsg) != NULL)
			elog(FATAL, "we cannot handle two or more cmsghdr at once");

		peer_fdesc = ((int *)CMSG_DATA(cmsg))[0];
	}

	/*
	 * It is likely EOF; peer socket is closed.
	 */
	if (retval == 0 || msg.msg_iovlen == 0)
		return false;

	PG_TRY();
	{
		GpuServCommand *cmd = (GpuServCommand *)msgbuf;
		GpuTask_v2	   *gtask;
		Size			unitsz;

		while (retval > 0)
		{
			if (retval < offsetof(GpuServCommand, u))
				elog(ERROR, "Bug? unexpected message format");

			if (cmd->command == GPUSERV_CMD_TASK)
			{
				unitsz = offsetof(offsetof(GpuServCommand, u) +
								  sizeof(cmd->u.task));
				if (retval < unitsz)
					elog(ERROR, "Bug? short GPUSERV_CMD_TASK message");

				gtask = cmd->u.task.gtask;
				Assert(dmaBufferValidatePtr(gtask));
				if (peer_fdesc < 0)
					gtask->peer_fdesc = -1;
				else
				{
					Assert(gtask->file_desc >= 0);
					gtask->peer_fdesc = peer_fdesc;
				}
				Assert(!gtask->chain.prev && !gtask->chain.next);
				Assert(!gtask->cuda_stream);

				if (IsGpuServerProcess())
				{
					// we need an api to just in refcnt
					gtask->gcontext = GetGpuContext(gcontext)

					SpinLockAcquire();
					dlist_push_tail(&session_pending_tasks, &gtask->chain);
					SpinLockRelease();
				}
				else
				{
					GpuTaskState   *gts = gtask->gts;

					dlist_push_tail(&gts->ready_tasks, &gtask->chain);
					gts->num_ready_tasks++;
				}
			}
			else if (cmd->command == GPUSERV_CMD_ERROR)
			{
				const char *filename;
				const char *funcname;
				const char *message;
				const char *buf;

				unitsz = offsetof(offsetof(GpuServCommand, u) +
								  sizeof(cmd->u.error));
				if (retval < unitsz)
					elog(ERROR, "Bug? short GPUSERV_CMD_ERROR message");
				if (peer_fdesc >= 0)
					elog(ERROR, "Bug? only GPUSERV_CMD_TASK can deliver FDs");
				if (IsGpuServerProcess())
					elog(ERROR, "Bug? only GPU server can send ERROR");
				if (cmd->u.error.buffer_external)
				{
					buf = cmd->u.error.buffer_external;
					Assert(cmd->u.error.buffer_usage == 0);
				}
				else
				{
					buf = cmd->u.error.buffer_inline;
				}
				filename = buf + cmd->u.error.filename_offset;
				funcname = buf + cmd->u.error.funcname_offset;
				message  = buf + cmd->u.error.message_offset;

				if (errstart(Max(__cmd.u.error.elevel, ERROR),
							 filename,
							 cmd->u.error.lineno,
							 funcname,
							 TEXTDOMAIN))
					errfinish(errcode(cmd->u.error.sqlerrcode),
							  errmsg("%s", message));
			}
			else
				elog(ERROR, "Bug? unexpected GPUSERV_CMD_* command");

			cmd = (GpuServCommand *)((char *)cmd + unitsz);
			retval -= unitsz;
			if (peer_fdesc >= 0 && retval > 0)
				elog(ERROR, "Peer-fd is delivered with multiple commands");
		}
	}
	PG_CATCH();
	{
		if (peer_fdesc >= 0)
			close(peer_fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return true;	/* one or more commands were received */
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
		CHECK_FOR_INTERRUPTS();

		ResetLatch();

		// flush out if completed tasks

		// enqueu pendinf tasks

		// flush out again if completed tasks

		// picks up pending code compile, if no running tasks

		rc = WaitEventSetWait(session_event_set,
							  (long)GpuServerCommTimeout,
							  &event, 1);
		if (retval > 0)
		{
			if (event.events & WL_POSTMASTER_DEATH)
				elog(FATAL,
					 (errcode(ERRCODE_ADMIN_SHUTDOWN),
					  errmsg("Urgent termination due to postmaster dead")));

			if (event.events & WL_SOCKET_READABLE)
			{
				if (!event.user_data)
					gpuservAcceptConnection();
				else
				{
					GpuContext_v2  *gcontext = event.user_data;

					if (!gpuservRecvCommands(gcontext))
						gpuservCloseConnection(gcontext);
				}
			}
		}
	}


#if 0
	/* Unload all the CUDA modules;  */ --> once deactivated;
	for (i=0; i < CUDA_MODULES_SLOTSIZE; i++)
	{
		cudaModuleCache *cache;
		dlist_node *dnode;
		CUresult	rc;

		while (!dlist_is_empty(&cuda_modules_slot[i]))
		{
			dnode = dlist_pop_head_node(&cuda_modules_slot[i]);
			cache = dlist_container(cudaModuleCache, chain, dnode);

			rc = cuModuleUnload(cache->cuda_module);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuModuleUnload: %s", errorText(rc));
		}
	}
#endif
}

/*
 * pgstrom_bgworker_main - entrypoint of the CUDA server process
 */
static void
gpuserv_bgworker_main(Datum __server_id)
{
	cl_int		device_id;
	cl_int		i;
	CUresult	rc;

	/* I am a GPU server process */
	gpuserv_id = DatumGetInt32(__server_id);
	Assert(gpuserv_id >= 0 && gpuserv_id < numGpuServers);
	gpuServProc = &gpuServState->serv_procs[gpuserv_id];
	gpuserv_server_sockfd = gpuServSocket[gpuserv_id].sockfd;
	pqsignal(SIGTERM, gpuservGotSigterm);
	BackgroundWorkerUnblockSignals();

	/* Init CUDA runtime */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	gpuserv_dindex = gpuserv_id % numDevAttrs;
	rc = cuDeviceGet(&gpuserv_cuda_device,
					 devAttrs[gpuserv_dindex].DEV_ID);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&gpuserv_cuda_context,
					 CU_CTX_SCHED_AUTO,
					 gpuserv_cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxCreate: %s", errorText(rc));

	for (i=0; i < CUDA_MODULES_SLOTSIZE; i++)
		dlist_init(&cuda_modules_slot[i]);

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
	session_event_set = CreateWaitEventSet(TopMemoryContext, 30);
	AddWaitEventToSet(session_event_set,
					  WL_POSTMASTER_DEATH |
					  WL_LATCH_SET |
					  WL_SOCKET_READABLE,
					  gpuserv_server_sockfd,
					  MyLatch, NULL);
	session_num_clients = 0;

	/* register myself on the shared GpuServProc structure */
	SpinLockAcquire(&gpuServState->lock);
	gpuServProc->backend_leader_id = InvalidBackendId;
	gpuServProc->backend_id = InvalidBackendId;
	gpuServProc->gcontext_id = INVALID_GPU_CONTEXT_ID;
	gpuServProc->pgproc = MyProc;
	dlist_push_tail(&gpuServState->serv_procs_list[dindex],
					&gpuServProc->chain);
	SpinLockRelease(&gpuServState->lock);

	elog(LOG, "PG-Strom GPU/CUDA Server [%d] is now ready on GPU-%d %s",
		 gpuserv_id, devAttrs[dindex].DEV_ID, devAttrs[dindex].DEV_NAME);

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
		gpuServProc->backend_id = InvalidBackendId;
		gpuServProc->connect_in_progress = false;
		gpuServProc->pgproc = NULL;
		dlist_delete(&gpuServProc->chain);
		memset(&gpuServProc->chain, 0, sizeof(dlist_node));
		SpinLockRelease(&gpuServState->lock);

		/*
		 * Destroy the CUDA context not to wake up the callback functions
		 * any more, regardless of the status of asynchronous GpuTasks.
		 */
		rc = cuCtxDestroy(gpuserv_cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));

		/*
		 * Shared portion of GpuContext has to be detached regardless of
		 * the reference counter of local portion, because the orphan
		 * SharedGpuContext will lead memory leak of the shared DMA buffer
		 * segments.
		 */
		for (i=0; i < session_num_clients; i++)
		{
			GpuContext_v2  *gcontext = gpuserv_gcontexts[i];

			PutSharedGpuContext(gcontext->shgcon);
		}
		PG_RE_THROW();
	}
	PG_END_TRY();

	elog(FATAL, "Bug? GpuServer has no path to exit normally");
}

/*
 * gpuserv_on_postmaster_exit - remove UNIX domain socket on shutdown of
 * the postmaster process.
 */
static void
gpuserv_on_postmaster_exit(int code, Datum arg)
{
	int		i;

	if (MyProcPid != PostmasterPid)
		return;

	for (i=0; i < numGpuServers; i++)
	{
		if (close(gpuServSocket[i].sockfd) != 0)
			elog(WARNING, "failed on close(2): %m");
		if (unlink(gpuServSocket[i].sockaddr.sun_path) != 0)
			elog(WARNING, "failed on unlink('%s'): %m",
				 gpuServSocket[i].sockaddr.sun_path);
	}
}

/*
 * totalProcs - see the logic in InitProcGlobal
 */
static inline uint32
totalProcs(void)
{
	return MaxBackends + NUM_AUXILIARY_PROCS + max_prepared_xacts;
}

/*
 * gpuservShmemRequired - required size of static shared memory
 */
static inline Size
gpuservShmemRequired(void)
{
	int		nwords;
	Size	required;

    nwords = (totalProcs() + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
    required = (MAXALIGN(offsetof(GpuServState, serv_procs[numGpuServers])) +
				/* inactive_servers */
				MAXALIGN(offsetof(Bitmapset, words[nwords])) +
				/* gpumem_waiters */
                MAXALIGN(sizeof(Bitmapset *) * numDevAttrs) +
                MAXALIGN(offsetof(Bitmapset, words[nwords])) * numDevAttrs +
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
	int			i, nwords;
	char	   *pos;
	bool		found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* request for the static shared memory */
	gpuServState = ShmemInitStruct("gpuServState", required, &found);
	Assert(!found);

	nwords = (totalProcs() + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	memset(gpuServState, 0, required);
	SpinLockInit(&gpuServState->lock);
	pos = ((char *)gpuServState +
		   MAXALIGN(offsetof(GpuServState, serv_procs[numGpuServers])));
	/* inactive_servers */
	gpuServState->inactive_servers = (Bitmapset *)pos;
	pos += MAXALIGN(offsetof(Bitmapset, words[nwords]));
	/* gpumem_waiters */
	gpuServState->gpumem_waiters = (Bitmapset **)pos;
	pos += MAXALIGN(sizeof(Bitmapset *) * numDevAttrs);

	for (i=0; i < numDevAttrs; i++)
	{
		gpuServState->gpumem_waiters[i] = (Bitmapset *) pos;
		pos += MAXALIGN(offsetof(Bitmapset, words[nwords]));
	}
	/* serv_procs_list */
	gpuServState->serv_procs_list = (dlist_head *) pos;
	pos += MAXALIGN(sizeof(dlist_head) * numDevAttrs);
	for (i=0; i < numDevAttrs; i++)
		dlist_init(&gpuServState->serv_procs_list[i]);

	Assert((char *)gpuServState + required == pos);
}

/*
 * pgstrom_init_gpu_server
 */
void
pgstrom_init_gpu_server(void)
{
	Size			required;
	cl_int			nwords;
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
	 * Setup UNIX domain sockets for listen/accept, for each CUDA devices
	 */
	gpuServSocket = malloc(sizeof(GpuServSocket) * numGpuServers);
	if (!gpuServSocket)
		elog(ERROR, "out of memory");

	for (i=0; i < numGpuServers; i++)
	{
		struct sockaddr_un *sockaddr = &gpuServSocket[i].sockaddr;
		pgsocket	sockfd;
		cl_long		suffix;

		sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
		if (sockfd < 0)
			elog(ERROR, "failed on socket(AF_UNIX, SOCK_STREAM, 0): %m");
		gpuServSocket[i].sockfd = sockfd;

		suffix = (long) getpid();
		for (;;)
		{
			sockaddr->sun_family = AF_UNIX;
			snprintf(sockaddr->sun_path, sizeof(sockaddr->sun_path),
					 ".pg_strom.gpuserv.sock.%u.%d", suffix, i);
			if (bind(sockfd, (struct sockaddr *)sockaddr,
					 sizeof(struct sockaddr_un)) == 0)
				break;
			else if (errno == EADDRINUSE)
			{
				elog(LOG, "UNIX domain socket \"%s\" is already in use: %m",
					 sockaddr->sun_path);
				suffix++;
			}
			else
				elog(ERROR, "failed on bind('%s'): %m", sockaddr->sun_path);
		}

		/* listen(2) */
		if (listen(sockfd, numGpuServers) != 0)
			elog(ERROR, "failed on listen(2): %m");

		/* assign reasonably short timeout for accept(2) */
		timeout.tv_sec = 0;
		timeout.tv_usec = 400 * 1000;	/* 400ms */
		if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO,
					   &timeout, sizeof(timeout)) != 0)
			elog(ERROR, "failed on setsockopt(2): %m");
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
		worker.bgw_restart_time = 4;
		worker.bgw_main = gpuserv_bgworker_main;
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* request for the static shared memory */
	RequestAddinShmemSpace(gpuservShmemRequired());
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_server;
	/* cleanup of UNIX domain socket */
	on_proc_exit(gpuserv_on_postmaster_exit, 0);
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
