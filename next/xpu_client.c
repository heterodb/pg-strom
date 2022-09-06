/*
 * gpu_client.c
 *
 * Backend side routines to connect GPU service.
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

static dlist_head	xpu_connections_list;

/*
 * Worker thread to receive response messages
 */
static void *
__xpuConnectSessionWorkerAlloc(void *__priv, size_t sz)
{
	return malloc(sz);
}

static void
__xpuConnectSessionWorkerAttach(void *__priv, XpuCommand *xcmd)
{
	XpuConnection *conn = __priv;

	xcmd->priv = conn;
	pthreadMutexLock(&conn->mutex);
	Assert(conn->num_running_cmds > 0);
	dlist_push_tail(&conn->ready_cmds_list, &xcmd->chain);
	conn->num_running_cmds--;
	conn->num_ready_cmds++;
	pthreadMutexUnlock(&conn->mutex);
	SetLatch(MyLatch);
}

static void *
__xpuConnectSessionWorker(void *__priv)
{
	XpuConnection *conn = __priv;

#define __fprintf(filp,fmt,...)										\
	fprintf((filp), "[%s; %s:%d] " fmt "\n",						\
			conn->devname, __FILE_NAME__, __LINE__, ##__VA_ARGS__)

	for (;;)
	{
		struct pollfd pfd;
		int		sockfd;
		int		nevents;

		sockfd = conn->sockfd;
		if (sockfd < 0)
			break;
		pfd.fd = sockfd;
		pfd.events = POLLIN;
		pfd.revents = 0;
		nevents = poll(&pfd, 1, -1);
		if (nevents < 0)
		{
			if (errno == EINTR)
				continue;
			__fprintf(stderr, "failed on poll(2): %m");
			break;
		}
		else if (nevents > 0)
		{
			Assert(nevents == 1);
			if (pfd.revents & ~POLLIN)
			{
				/* peer socket closed */
				pthreadMutexLock(&conn->mutex);
				conn->terminated = 1;
				pthreadMutexUnlock(&conn->mutex);
				__fprintf(stderr, "peer socket closed.");
				return NULL;
			}
			else if (pfd.revents & POLLIN)
			{
				if (pgstrom_receive_xpu_command(conn->sockfd,
												__xpuConnectSessionWorkerAlloc,
												__xpuConnectSessionWorkerAttach,
												conn, conn->devname) < 0)
					break;
			}
		}
	}
	pthreadMutexLock(&conn->mutex);
	conn->terminated = -1;
	pthreadMutexUnlock(&conn->mutex);

	return NULL;
#undef __fprintf
}

/*
 * xpuClientSendCommand
 */
void
xpuClientSendCommand(XpuConnection *conn, const XpuCommand *xcmd)
{
	int		sockfd = conn->sockfd;
	ssize_t	nbytes;

	pthreadMutexLock(&conn->mutex);
	conn->num_running_cmds++;
	pthreadMutexUnlock(&conn->mutex);

	nbytes = __writeFile(sockfd, xcmd, xcmd->length);
	if (nbytes != xcmd->length)
		elog(ERROR, "unable to send XPU command to GPU service (%zd of %lu): %m",
			 nbytes, xcmd->length);
}

/*
 * xpuClientGetResponse
 */
XpuCommand *
xpuClientGetResponse(XpuConnection *conn, long timeout)
{
	XpuCommand *xcmd = NULL;
	dlist_node *dnode;
	TimestampTz	ts_expired = INT64_MAX;
	TimestampTz	ts_curr;
	int			ev, flags = WL_LATCH_SET | WL_POSTMASTER_DEATH;

	if (timeout > 0)
	{
		flags |= WL_TIMEOUT;
		ts_expired = GetCurrentTimestamp() / 1000L + timeout;
	}

	for (;;)
	{
		pthreadMutexLock(&conn->mutex);
		if (conn->terminated < 0)
		{
			pthreadMutexUnlock(&conn->mutex);
			elog(ERROR, "%s: connection terminated", conn->devname);
		}
		if (!dlist_is_empty(&conn->ready_cmds_list))
		{
			dnode = dlist_pop_head_node(&conn->ready_cmds_list);
			xcmd = dlist_container(XpuCommand, chain, dnode);
			dlist_push_tail(&conn->active_cmds_list, &xcmd->chain);
			conn->num_ready_cmds--;
			pthreadMutexUnlock(&conn->mutex);
			return xcmd;
		}
		/* if no running tasks, it makes no sense to wait for */
		if (conn->num_running_cmds == 0 || conn->terminated != 0)
			timeout = 0;
		pthreadMutexUnlock(&conn->mutex);
		if (timeout == 0)
			break;
		/* wait for response */
		ev = WaitLatch(MyLatch, flags, timeout,
					   PG_WAIT_EXTENSION);
		if (ev & WL_POSTMASTER_DEATH)
			elog(FATAL, "unexpected postmaster dead");
		if (ev & WL_TIMEOUT)
			break;
		if (flags & WL_TIMEOUT)
		{
			ts_curr = GetCurrentTimestamp() / 1000L;
			if (ts_expired <= ts_curr)
				break;
			timeout = ts_expired - ts_curr;
		}
		CHECK_FOR_INTERRUPTS();
	}
	return NULL;
}

/*
 * xpuClientPutResponse
 */
void
xpuClientPutResponse(XpuCommand *xcmd)
{
	XpuConnection  *conn = xcmd->priv;
	
	pthreadMutexLock(&conn->mutex);
	dlist_delete(&xcmd->chain);
	pthreadMutexUnlock(&conn->mutex);
	free(xcmd);
}

/*
 * __xpuClientInitSession
 */
static XpuConnection *
__xpuClientInitSession(XpuConnection *conn, const XpuCommand *session)
{
	XpuCommand *resp;

	Assert(session->tag == XpuCommandTag__OpenSession);
	xpuClientSendCommand(conn, session);
	resp = xpuClientGetResponse(conn, -1);
	if (resp->tag != XpuCommandTag__Success)
		elog(ERROR, "%s:OpenSession failed - %s (%s:%d %s)",
			 conn->devname,
			 resp->u.error.message,
			 resp->u.error.filename,
			 resp->u.error.lineno,
			 resp->u.error.funcname);
	xpuClientPutResponse(resp);

	return conn;
}

/*
 * gpuClientOpenSession
 */
static int
__gpuClientChooseDevice(const Bitmapset *gpuset)
{
	static bool		rr_initialized = false;
	static uint32	rr_counter = 0;

	if (!rr_initialized)
	{
		rr_counter = (uint32)getpid();
		rr_initialized = true;
	}

	if (!bms_is_empty(gpuset))
	{
		int		num = bms_num_members(gpuset);
		int	   *dindex = alloca(sizeof(int) * num);
		int		i, k;

		for (i=0, k=bms_next_member(gpuset, -1);
			 k >= 0;
			 i++, k=bms_next_member(gpuset, k))
		{
			dindex[i] = k;
		}
		Assert(i == num);
		return dindex[rr_counter++ % num];
	}
	/* a simple round-robin if no GPUs preference */
	return (rr_counter++ % numGpuDevAttrs);
}

XpuConnection *
gpuClientOpenSession(const Bitmapset *gpuset,
					 const XpuCommand *session)
{
	struct sockaddr_un addr;
	pgsocket	sockfd;
	XpuConnection *conn = NULL;

	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");
	PG_TRY();
	{
		int		cuda_dindex = __gpuClientChooseDevice(gpuset);
		int		errcode;

		addr.sun_family = AF_UNIX;
		snprintf(addr.sun_path, sizeof(addr.sun_path),
				 ".pg_strom.%u.gpu%u.sock",
				 PostmasterPid, cuda_dindex);
		if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
			elog(ERROR, "failed on connect('%s'): %m", addr.sun_path);

		conn = calloc(1, sizeof(XpuConnection));
		if (!conn)
			elog(ERROR, "out of memory");
		snprintf(conn->devname, 32, "GPU-%d", cuda_dindex);
		conn->sockfd = sockfd;
		conn->resowner = CurrentResourceOwner;
		pthreadMutexInit(&conn->mutex);
		conn->num_running_cmds = 0;
		conn->num_ready_cmds = 0;
		dlist_init(&conn->ready_cmds_list);
		dlist_init(&conn->active_cmds_list);
		if ((errcode = pthread_create(&conn->worker, NULL,
									  __xpuConnectSessionWorker, conn)) != 0)
		{
			free(conn);
			elog(ERROR, "failed on pthread_create: %s", strerror(errcode));
		}
		dlist_push_tail(&xpu_connections_list, &conn->chain);
	}
	PG_CATCH();
	{
		close(sockfd);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return __xpuClientInitSession(conn, session);
}

/*
 * xpuClientCloseSession
 */
void
xpuClientCloseSession(XpuConnection *conn)
{
	XpuCommand *xcmd;
	dlist_node *dnode;

	/* ensure termination of worker thread */
	close(conn->sockfd);
	conn->sockfd = -1;
	pg_memory_barrier();
	pthread_kill(conn->worker, SIGPOLL);
	pthread_join(conn->worker, NULL);

	while (!dlist_is_empty(&conn->ready_cmds_list))
	{
		dnode = dlist_pop_head_node(&conn->ready_cmds_list);
		xcmd = dlist_container(XpuCommand, chain, dnode);
		free(xcmd);
	}
	while (!dlist_is_empty(&conn->active_cmds_list))
	{
		dnode = dlist_pop_head_node(&conn->active_cmds_list);
		xcmd = dlist_container(XpuCommand, chain, dnode);
		free(xcmd);
	}
	dlist_delete(&conn->chain);
	free(conn);
}

/*
 * xpuclientCleanupConnections
 */
static void
xpuclientCleanupConnections(ResourceReleasePhase phase,
							bool isCommit,
							bool isTopLevel,
							void *arg)
{
	dlist_mutable_iter	iter;

	if (phase != RESOURCE_RELEASE_BEFORE_LOCKS)
		return;

	dlist_foreach_modify(iter, &xpu_connections_list)
	{
		XpuConnection  *conn = dlist_container(XpuConnection,
											   chain, iter.cur);
		if (conn->resowner == CurrentResourceOwner)
		{
			if (isCommit)
				elog(LOG, "Bug? GPU connection %d is not closed on ExecEnd",
					 conn->sockfd);
			xpuClientCloseSession(conn);
		}
	}
}

void
pgstrom_init_xpu_client(void)
{
	dlist_init(&xpu_connections_list);
	RegisterResourceReleaseCallback(xpuclientCleanupConnections, NULL);
}
