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

#define GPUSERV_CMD_OPEN		0x101
#define GPUSERV_CMD_TASK		0x102
#define GPUSERV_CMD_ERROR		0x103
#define GPUSERV_CMD_CLOSE		0x104

typedef struct GpuServProc
{
	dlist_node		chain;
	dlist_node		devmem_chain;
	cl_int			device_id;
	PGPROC		   *pgproc;
} GpuServProc;

typedef struct GpuServState
{
	slock_t			lock;
	cl_int			num_accept_servs;
	cl_int			num_pending_conn;
	Bitmapset	  **gpumem_waiters;		/* for each devices */
	dlist_head		serv_procs_list;
	GpuServProc		serv_procs[FLEXIBLE_ARRAY_MEMBER];
} GpuServState;

#define WORDNUM(x)	((x) / BITS_PER_BITMAPWORD)
#define BITNUM(x)	((x) % BITS_PER_BITMAPWORD)

typedef struct GpuServCommand
{
	cl_int		command;	/* one of the GPUSERV_CMD_* */
	union {
		struct {
			cl_uint		context_id;
			BackendId	backend_id;
		} open;
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
			cl_uint		buffer_usage;
			char	   *buffer_external;
			char		buffer_inline[200];
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
static GpuServState	   *gpuServState = NULL;
static struct sockaddr_un gpuserv_addr;
static pgsocket			gpu_server_sock = PGINVALID_SOCKET;
static int				gpu_server_id = -1;
static bool				gpu_server_got_sigterm = false;
static int				numGpuServers;			/* GUC */
static int				GpuServerCommTimeout;	/* GUC */
static int				gpuserv_device_id = -1;
GpuContext_v2		   *gpuserv_gpu_context = NULL;
CUdevice				gpuserv_cuda_device = NULL;
CUcontext				gpuserv_cuda_context = NULL;
/*
 * per session state - these list might be updated by the callbacl of CUDA,
 * thus, it must be touched under the lock
 */
static slock_t			session_tasks_lock;
static dlist_head		session_pending_tasks;
static dlist_head		session_running_tasks;
static dlist_head		session_completed_tasks;

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

	gpu_server_got_sigterm = true;

	SetLatch(MyLatch);

	errno = save_errno;
}

/*
 * IsGpuServerProcess - returns true, if current process is gpu server
 */
bool
IsGpuServerProcess(void)
{
	return (bool)(gpu_server_id >= 0);
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
ReportErrorForBackend(GpuTask_v2 *gtask, MemoryContext memcxt)
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

	required = (MAXALIGN(strlen(errdata->filename) + 1) +
				MAXALIGN(strlen(errdata->funcname) + 1) +
				MAXALIGN(strlen(errdata->message) + 1));
	if (required <= sizeof(cmd.u.error.buffer_inline))
	{
		buf = cmd.u.error.buffer_inline;
		cmd.u.error.buffer_external = NULL;
	}
	else
	{
		buf = dmaBufferAlloc(gpuserv_gpu_context, required);
		cmd.u.error.buffer_external = buf;
	}
	cmd.command                 = GPUSERV_CMD_ERROR;
	cmd.u.error.elevel          = errdata->elevel;
	cmd.u.error.sqlerrcode      = errdata->sqlerrcode;

	cmd.u.error.filename_offset = offset;
	strcpy(buf + offset, errdata->filename);
	offset += MAXALIGN(strlen(errdata->filename) + 1);

	cmd.u.error.lineno          = errdata->lineno;

	cmd.u.error.funcname_offset = offset;
	strcpy(buf + offset, errdata->funcname);
	offset += MAXALIGN(strlen(errdata->funcname) + 1);

	cmd.u.error.message_offset = offset;
	strcpy(buf + offset, errdata->message);
	offset += MAXALIGN(strlen(errdata->message) + 1);

	if (!cmd.u.error.buffer_external)
		cmd.u.error.buffer_usage = offset;
	else
		cmd.u.error.buffer_usage = 0;	/* no count for external buffer */

	/*
	 * Urgent return to the backend
	 */
	gettimeofday(&tv1, NULL);
	for (;;)
	{
		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();
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
	if (gpu_server_got_sigterm)
		ereport(FATAL,
				(errcode(ERRCODE_ADMIN_SHUTDOWN),
				 errmsg("Terminating PG-Strom GPU/CUDA Server[%d]",
						gpu_server_id)));
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
 * gpuservWakeUpProcesses - wakes up sleeping GPU server processes
 */
void
gpuservWakeUpProcesses(cl_uint max_procs)
{
	dlist_iter		iter;
	GpuServProc	   *serv_proc;
	PGPROC		   *pgproc;

	if (max_procs == 0)
		return;
	SpinLockAcquire(&gpuServState->lock);
	dlist_foreach(iter, &gpuServState->serv_procs_list)
	{
		serv_proc = dlist_container(GpuServProc, chain, iter.cur);
		Assert(serv_proc->pgproc != NULL);
		pgproc = serv_proc->pgproc;
		/* skip latches already set */
		pg_memory_barrier();
		if (pgproc->procLatch.is_set)
			continue;
		/* wake up a server process */
		SetLatch(&pgproc->procLatch);
		if (--max_procs == 0)
			break;
	}
	SpinLockRelease(&gpuServState->lock);
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
static bool
gpuservSendCommand(GpuContext_v2 *gcontext, GpuServCommand *cmd, long timeout)
{
	struct msghdr	msg;
	struct iovec	iov;
	unsigned char	cmsgbuf[CMSG_SPACE(sizeof(int))];
	ssize_t			retval;
	int				ev;

	memset(&msg, 0, sizeof(struct msghdr));
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;
	iov.iov_base = cmd;
	if (cmd->command == GPUSERV_CMD_TASK)
	{
		GpuTask_v2 *gtask = cmd->u.task.gtask;

		Assert(dmaBufferValidatePtr(gtask));
		/* Is the file-descriptor attached on the message? */
		if (!IsGpuServerProcess() && gtask->file_desc >= 0)
		{
			struct cmsghdr *cmsg;

			Assert(!IsGpuServerProcess());
			msg.msg_control = cmsgbuf;
			msg.msg_controllen = sizeof(cmsgbuf);
			cmsg = CMSG_FIRSTHDR(&msg);
			cmsg->cmsg_level = SOL_SOCKET;
			cmsg->cmsg_type = SCM_RIGHTS;
			cmsg->cmsg_len = CMSG_LEN(sizeof(int));
			((int *)CMSG_DATA(cmsg))[0] = gtask->file_desc;
		}
		iov.iov_len = offsetof(GpuServCommand, u.task) + sizeof(cmd->u.task);
	}
	else if (cmd->command == GPUSERV_CMD_OPEN)
	{
		iov.iov_len = offsetof(GpuServCommand, u.open) + sizeof(cmd->u.open);
	}
	else if (cmd->command == GPUSERV_CMD_ERROR)
	{
		iov.iov_len = (offsetof(GpuServCommand, u.error.buffer_inline) +
					   cmd->u.error.buffer_usage);
		Assert(iov.iov_len < sizeof(GpuServCommand));
		Assert(IsGpuServerProcess());
	}
	else
		elog(ERROR, "Bug? unexpected GPUSERV_CMD_* tag: %d", cmd->command);

	ev = WaitLatchOrSocket(MyLatch,
						   WL_LATCH_SET |
						   WL_POSTMASTER_DEATH |
						   WL_SOCKET_WRITEABLE |
						   (timeout < 0 ? 0 : WL_TIMEOUT),
						   gcontext->sockfd,
						   timeout);
	/* urgent bailout if postmaster is dead */
	if (ev & WL_POSTMASTER_DEATH)
		ereport(FATAL,
				(errcode(ERRCODE_ADMIN_SHUTDOWN),
				 errmsg("Urgent termination by unexpected postmaster dead")));
	/* Is the socket writable? */
	if ((ev & WL_SOCKET_WRITEABLE) == 0)
		return false;

	retval = sendmsg(gcontext->sockfd, &msg, 0);
	if (retval < 0)
		elog(ERROR, "failed on sendmsg(2): %m");
	else if (retval == 0)
		elog(ERROR, "no bytes sent using sendmsg(2): %m");
	else if (retval != iov.iov_len)
		elog(ERROR, "incorrect size of message sent: %zu but %zu expected",
			 retval, iov.iov_len);

	return true;
}

/*
 * gpuservSendGpuTask - enqueue a GpuTask to the socket
 */
bool
gpuservSendGpuTask(GpuContext_v2 *gcontext, GpuTask_v2 *gtask)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	GpuServCommand	cmd;
	long			timeout = 5000;		/* 5.0sec; usually enough */
	struct timeval	tv1, tv2;

	cmd.command = GPUSERV_CMD_TASK;
	cmd.u.task.gtask = gtask;

	gettimeofday(&tv1, NULL);
	for (;;)
	{
		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();
		if (IsGpuServerProcess())
			gpuservHandleLazyJobs(false, false);

		if (gpuservSendCommand(gcontext, &cmd, timeout))
		{
			SpinLockAcquire(&shgcon->lock);
			if (IsGpuServerProcess())
				shgcon->num_async_tasks--;
			else
				shgcon->num_async_tasks++;
			SpinLockRelease(&shgcon->lock);
			break;
		}

		/* adjust timeout */
	   	gettimeofday(&tv2, NULL);
		if (timeout >= 0)
		{
			timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
						(tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
			if (timeout <= 0)
				return false;
		}
		tv1 = tv2;
	}
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
	int				peer_fdesc = -1;
	int				ev;
	ssize_t			retval;

	ev = WaitLatchOrSocket(MyLatch,
						   WL_LATCH_SET |
						   WL_POSTMASTER_DEATH |
						   WL_SOCKET_READABLE |
						   (timeout < 0 ? 0 : WL_TIMEOUT),
						   sockfd,
						   timeout);
	/* urgent bailout if postmaster is dead */
	if (ev & WL_POSTMASTER_DEATH)
		ereport(FATAL,
				(errcode(ERRCODE_ADMIN_SHUTDOWN),
				 errmsg("Urgent termination by unexpected postmaster dead")));

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

		/* On EOL, we will return a dummy CLOSE command */
		if (retval == 0 || msg.msg_iovlen == 0)
		{
			/* Of course, no peer file-desc shall not be moved */
			Assert(peer_fdesc < 0);
			cmd->command = GPUSERV_CMD_CLOSE;
			elog(LOG, "no bytes received, likely connection closed");
		}
		else
		{
			if (retval < offsetof(GpuServCommand, u))
				elog(ERROR, "Bug? unexpected message format");

			cmd->command = __cmd.command;
			if (__cmd.command == GPUSERV_CMD_TASK)
			{
				GpuTask_v2 *gtask;

				if (retval != (offsetof(GpuServCommand, u) +
							   sizeof(__cmd.u.task)))
					elog(ERROR, "GPUSERV_CMD_TASK has unexpected format");

				gtask = __cmd.u.task.gtask;
				Assert(dmaBufferValidatePtr(gtask));
				if (peer_fdesc < 0)
					gtask->peer_fdesc = -1;
				else
				{
					Assert(gtask->file_desc >= 0);
					gtask->peer_fdesc = peer_fdesc;
				}
				Assert(!gtask->cuda_stream);
				cmd->u.task.gtask = gtask;
			}
			else if (__cmd.command == GPUSERV_CMD_OPEN)
			{
				if (retval != (offsetof(GpuServCommand, u) +
							   sizeof(__cmd.u.open)))
					elog(ERROR, "GPUSERV_CMD_OPEN has unexpected format");

				if (peer_fdesc >= 0)
					elog(ERROR, "Bug? only GPUSERV_CMD_TASK can deliver FD");

				cmd->u.open.context_id = __cmd.u.open.context_id;
				cmd->u.open.backend_id = __cmd.u.open.backend_id;
			}
			else if (__cmd.command == GPUSERV_CMD_ERROR)
			{
				/*
				 * Raise an error, if GPUSERV_CMD_ERROR was received
				 */
				const char *filename;
				const char *funcname;
				const char *message;
				const char *buf;

				if (retval != offsetof(GpuServCommand,
									   u.error.buffer_inline))
					elog(ERROR, "GPUSERV_CMD_ERROR has unexpected format");

				if (peer_fdesc >= 0)
					elog(ERROR, "Bug? only GPUSERV_CMD_TASK can deliver FD");

				if (__cmd.u.error.buffer_external)
				{
					buf = __cmd.u.error.buffer_external;
					Assert(__cmd.u.error.buffer_usage == 0);
				}
				else
				{
					buf = __cmd.u.error.buffer_external;
					if (retval < (offsetof(GpuServCommand,
										   u.error.buffer_inline) +
								  __cmd.u.error.buffer_usage))
						elog(ERROR, "Message of GPUSERV_CMD_ERROR is missing");
				}
				filename = buf + __cmd.u.error.filename_offset;
				funcname = buf + __cmd.u.error.funcname_offset;
				message  = buf + __cmd.u.error.message_offset;

				if (errstart(Max(__cmd.u.error.elevel, ERROR),
							 filename,
							 __cmd.u.error.lineno,
							 funcname,
							 TEXTDOMAIN))
					errfinish(errcode(__cmd.u.error.sqlerrcode),
							  errmsg("%s", message));
				pg_unreachable();
			}
			else
				elog(ERROR, "Bug? unexpected GPUSERV_CMD_* command");
		}
	}
	PG_CATCH();
	{
		if (peer_fdesc >= 0)
			close(peer_fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return true;
}

/*
 * gpuservRecvGpuTask - fetch a GpuTask from the socket
 */
GpuTask_v2 *
gpuservRecvGpuTaskTimeout(GpuContext_v2 *gcontext, long timeout)
{
	GpuServCommand	cmd;
	GpuTask_v2	   *gtask = NULL;
	struct timeval	tv1, tv2;

	gettimeofday(&tv1, NULL);
	for (;;)
	{
		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();
		if (IsGpuServerProcess())
			gpuservHandleLazyJobs(true, true);

		if (gpuservRecvCommand(gcontext->sockfd, &cmd, timeout))
			break;		/* OK, successfully received a message */

		gettimeofday(&tv2, NULL);
		if (timeout >= 0)
		{
			timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
						(tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
			if (timeout <= 0)
				return NULL;
		}
		tv1 = tv2;
	}

	if (cmd.command == GPUSERV_CMD_TASK)
	{
		/*
		 * Once peer_fd is tracked by the GpuContext, we can raise error
		 * with no explicit file-descriptor handling because GpuContext's
		 * resource tracker will clean up.
		 */
		gtask = cmd.u.task.gtask;
		if (gtask->peer_fdesc >= 0)
		{
			PG_TRY();
			{
				trackFileDesc(gcontext, gtask->peer_fdesc);
			}
			PG_CATCH();
			{
				close(gtask->peer_fdesc);
				PG_RE_THROW();
			}
			PG_END_TRY();
		}
	}
	else if (cmd.command != GPUSERV_CMD_CLOSE)
		elog(ERROR, "Bug? unexpected GpuServCommand %d", cmd.command);

	return gtask;
}

GpuTask_v2 *
gpuservRecvGpuTask(GpuContext_v2 *gcontext)
{
	return gpuservRecvGpuTaskTimeout(gcontext, GpuServerCommTimeout);
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
	SpinLockAcquire(&gpuServState->lock);
	if (gpuServState->num_pending_conn >= gpuServState->num_accept_servs)
	{
		SpinLockRelease(&gpuServState->lock);
		return false;
	}
	gpuServState->num_pending_conn++;
	SpinLockRelease(&gpuServState->lock);

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
			CHECK_FOR_INTERRUPTS();
		}
		gcontext->sockfd = sockfd;

		/*
		 * NOTE: Exchange message with very short timeout to handle a corner
		 * case when no available GPU server, even though num_pending_conn is
		 * less than num_accept_servs on above checks.
		 * Once a GPU server process got a signal during accept(2), it will
		 * break accept(2) a new connection then decrease the num_accept_servs
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
			struct timeval	tv1, tv2;

			/*
			 * Once GPU server successfully processed the OPEN command,
			 * it shall set backend's latch with a reasonable delay.
			 */
			ResetLatch(MyLatch);

			/* Send OPEN command, with 100ms timeout */
			cmd.command = GPUSERV_CMD_OPEN;
			cmd.u.open.context_id = gcontext->shgcon->context_id;
			cmd.u.open.backend_id = MyProc->backendId;
			if (gpuservSendCommand(gcontext, &cmd, 100))
			{
				long	timeout = 100;	/* 100ms */

				gettimeofday(&tv1, NULL);
				for (;;)
				{
					ResetLatch(MyLatch);

					ev = WaitLatch(MyLatch,
								   WL_LATCH_SET |
								   WL_TIMEOUT |
								   WL_POSTMASTER_DEATH,
								   timeout);
					/* urgent bailout if postmaster is dead */
					if (ev & WL_POSTMASTER_DEATH)
						ereport(FATAL,
								(errcode(ERRCODE_ADMIN_SHUTDOWN),
							errmsg("Urgent termination by postmaster dead")));
					if (ev & WL_LATCH_SET)
					{
						SharedGpuContext   *shgcon = gcontext->shgcon;

						SpinLockAcquire(&shgcon->lock);
						/* OK, session is successfully open */
						if (shgcon->server != NULL)
						{
							SpinLockRelease(&shgcon->lock);
							break;
						}
						SpinLockRelease(&shgcon->lock);
					}
					if (ev & WL_TIMEOUT)
						timeout = 0;
					else
					{
						gettimeofday(&tv2, NULL);

						timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
									(tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
					}

					if (timeout <= 0)
					{
						/* ...revert it... */
						gcontext->sockfd = PGINVALID_SOCKET;
						close(sockfd);
						break;
					}
				}
			}
			else
			{
				/* ...revert it... */
				gcontext->sockfd = PGINVALID_SOCKET;
				close(sockfd);
			}
		}
	}
	PG_CATCH();
	{
		/* close the socket if opened */
	   	if (sockfd >= 0)
			close(sockfd);
		/* revert the number of pending clients */
		SpinLockAcquire(&gpuServState->lock);
		gpuServState->num_pending_conn--;
		SpinLockRelease(&gpuServState->lock);
	}
	PG_END_TRY();

	/* revert the number of pending clients */
	SpinLockAcquire(&gpuServState->lock);
	gpuServState->num_pending_conn--;
	SpinLockRelease(&gpuServState->lock);

	return (gcontext->sockfd != PGINVALID_SOCKET ? true : false);
}

/*
 * gpuservAcceptConnection - accept a client connection
 */
static GpuContext_v2 *
gpuservAcceptConnection(void)
{
	GpuServCommand	cmd;
	GpuContext_v2  *gcontext = NULL;
	pgsocket		sockfd = PGINVALID_SOCKET;

	Assert(IsGpuServerProcess());

	/* server is now waiting */
	SpinLockAcquire(&gpuServState->lock);
	gpuServState->num_accept_servs++;
	SpinLockRelease(&gpuServState->lock);

	PG_TRY();
	{
		for (;;)
		{
			ResetLatch(MyLatch);
			sockfd = accept(gpu_server_sock, NULL, NULL);
			if (sockfd >= 0)
				break;
			if (errno != EINTR && errno != EAGAIN)
				elog(ERROR, "failed on accept(2): %m");
			CHECK_FOR_INTERRUPTS();
			gpuservHandleLazyJobs(false, false);
		}
	}
	PG_CATCH();
	{
		/* server now stopped waiting */
		SpinLockAcquire(&gpuServState->lock);
		gpuServState->num_accept_servs--;
		SpinLockRelease(&gpuServState->lock);

		PG_RE_THROW();
	}
	PG_END_TRY();

	/* server now stopped waiting */
	SpinLockAcquire(&gpuServState->lock);
	gpuServState->num_accept_servs--;
	SpinLockRelease(&gpuServState->lock);

	if (sockfd < 0)
		return NULL;

	PG_TRY();
	{
		/* receive OPEN command (timeout=100ms) */
		if (gpuservRecvCommand(sockfd, &cmd, 100) &&
			cmd.command == GPUSERV_CMD_OPEN)
		{
			gcontext = AttachGpuContext(sockfd,
										cmd.u.open.context_id,
										cmd.u.open.backend_id,
										gpuserv_device_id);
		}
	}
	PG_CATCH();
	{
		close(sockfd);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return gcontext;
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
gpuservSessionMain(void)
{
	GpuTask_v2 *gtask;
	int			i, ev;

	while ((gtask = gpuservRecvGpuTask(gpuserv_gpu_context)) != NULL)
	{
		SpinLockAcquire(&session_tasks_lock);
		dlist_push_tail(&session_pending_tasks, &gtask->chain);
		SpinLockRelease(&session_tasks_lock);
	}

	/* Got EOL, and then process all the pending tasks */
	for (;;)
	{
		ResetLatch(MyLatch);

		CHECK_FOR_INTERRUPTS();
		gpuservHandleLazyJobs(true, true);

		SpinLockAcquire(&session_tasks_lock);
		if (dlist_is_empty(&session_pending_tasks) &&
			dlist_is_empty(&session_running_tasks) &&
			dlist_is_empty(&session_completed_tasks))
		{
			SpinLockRelease(&session_tasks_lock);
			break;
		}
		SpinLockRelease(&session_tasks_lock);

		ev = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_POSTMASTER_DEATH,
					   0);
		if (ev & WL_POSTMASTER_DEATH)
			ereport(FATAL,
					(errcode(ERRCODE_ADMIN_SHUTDOWN),
					 errmsg("Urgent termination by postmaster dead")));
	}

	/* Unload all the CUDA modules */
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
}

/*
 * pgstrom_cuda_serv_main - entrypoint of the CUDA server process
 */
static void
gpuserv_main(Datum __server_id)
{
	GpuServProc *serv_proc;
	cl_int		dindex;
	cl_int		i;
	CUresult	rc;

	/* I am a GPU server process */
	gpu_server_id = DatumGetInt32(__server_id);
	Assert(gpu_server_id >= 0 && gpu_server_id < numGpuServers);
	serv_proc = &gpuServState->serv_procs[gpu_server_id];
	pqsignal(SIGTERM, gpuservGotSigterm);
	BackgroundWorkerUnblockSignals();

	/* Init CUDA runtime */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuInit(0): %s", errorText(rc));

	dindex = gpu_server_id % numDevAttrs;
	gpuserv_device_id = devAttrs[dindex].DEV_ID;
	rc = cuDeviceGet(&gpuserv_cuda_device, gpuserv_device_id);
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
	elog(LOG, "PG-Strom GPU/CUDA Server [%d] is now ready on GPU-%d %s",
		 gpu_server_id, devAttrs[dindex].DEV_ID, devAttrs[dindex].DEV_NAME);

	/* ready to handle async tasks */
	SpinLockAcquire(&gpuServState->lock);
	serv_proc->pgproc = MyProc;
	dlist_push_head(&gpuServState->serv_procs_list, &serv_proc->chain);
    SpinLockRelease(&gpuServState->lock);

	PG_TRY();
	{
		for (;;)
		{
			SpinLockInit(&session_tasks_lock);
			dlist_init(&session_pending_tasks);
			dlist_init(&session_running_tasks);
			dlist_init(&session_completed_tasks);
			gpuserv_gpu_context = gpuservAcceptConnection();
			if (gpuserv_gpu_context)
			{
				/* move to the tail for lower priority of async tasks */
				SpinLockAcquire(&gpuServState->lock);
				dlist_delete(&serv_proc->chain);
				dlist_push_tail(&gpuServState->serv_procs_list,
								&serv_proc->chain);
				SpinLockRelease(&gpuServState->lock);

				gpuservSessionMain();

				PutGpuContext(gpuserv_gpu_context);
				MemoryContextReset(CurrentMemoryContext);

				/* move to the tail for higher priority of async tasks */
				SpinLockAcquire(&gpuServState->lock);
				dlist_delete(&serv_proc->chain);
				dlist_push_head(&gpuServState->serv_procs_list,
								&serv_proc->chain);
				SpinLockRelease(&gpuServState->lock);
			}
		}
	}
	PG_CATCH();
	{
		/* This server process can take async jobs no longer */
		SpinLockAcquire(&gpuServState->lock);
		dlist_delete(&serv_proc->chain);
		memset(serv_proc, 0, sizeof(GpuServProc));
		SpinLockRelease(&gpuServState->lock);

		/*
		 * NOTE: ereport() eventually kills the background worker process.
		 * It also releases any CUDA resources privately held by this worker,
		 * so, we don't need to reclaim these objects here.
		 * SharedGpuContext is exception. Unless putting the SharedGpuContext
		 * we hold, nobody will release its shared resources.
		 */
		if (currentSharedGpuContext)
			PutSharedGpuContext(currentSharedGpuContext);
		PG_RE_THROW();
	}
	PG_END_TRY();
	/* this server process can take async tasks no longer */
	SpinLockAcquire(&gpuServState->lock);
	dlist_delete(&serv_proc->chain);
	memset(serv_proc, 0, sizeof(GpuServProc));
	SpinLockRelease(&gpuServState->lock);

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
 * totalProcs - see the logic in InitProcGlobal
 */
static inline uint32
totalProcs(void)
{
	return MaxBackends + NUM_AUXILIARY_PROCS + max_prepared_xacts;
}

/*
 * pgstrom_startup_gpu_server
 */
static void
pgstrom_startup_gpu_server(void)
{
	Size		required;
	int			i, nwords;
	Bitmapset  *bms;
	bool		found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	nwords = (totalProcs() + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	required = (MAXALIGN(offsetof(GpuServState, serv_procs[numGpuServers])) +
				MAXALIGN(sizeof(Bitmapset *) * numDevAttrs) +
				MAXALIGN(offsetof(Bitmapset, words[nwords])) * numDevAttrs);
	gpuServState = ShmemInitStruct("gpuServState", required, &found);
	Assert(!found);

	memset(gpuServState, 0, required);
	SpinLockInit(&gpuServState->lock);
	dlist_init(&gpuServState->serv_procs_list);

	gpuServState->gpumem_waiters = (Bitmapset **)
		((char *)gpuServState + MAXALIGN(offsetof(GpuServState,
												  serv_procs[numGpuServers])));
	bms = (Bitmapset *)((char *)gpuServState->gpumem_waiters +
						MAXALIGN(sizeof(Bitmapset *) * numDevAttrs));
	for (i=0; i < numDevAttrs; i++)
	{
		gpuServState->gpumem_waiters[i] = bms;
		bms = (Bitmapset *)((char *)bms +
							MAXALIGN(offsetof(Bitmapset, words[nwords])));
	}
	Assert((char *)gpuServState + required == (char *)bms);
}

/*
 * pgstrom_init_gpu_server
 */
void
pgstrom_init_gpu_server(void)
{
	Size			required;
	cl_int			nwords;
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

	/* check signal for each 400ms during accept(2) */
	timeout.tv_sec = 0;
	timeout.tv_usec = 400 * 1000;	/* 400ms */
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
	nwords = (totalProcs() + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	required = (MAXALIGN(offsetof(GpuServState, serv_procs[numGpuServers])) +
				MAXALIGN(sizeof(Bitmapset *) * numDevAttrs) +
				MAXALIGN(offsetof(Bitmapset, words[nwords])) * numDevAttrs);
	RequestAddinShmemSpace(required);
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
										  nitems);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuOccupancyMaxPotentialBlockSize: %s",
			 errorText(rc));

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

	*p_block_size = (size_t)maxBlockSize;
	*p_grid_size = (nitems + maxBlockSize - 1) / maxBlockSize;
}
