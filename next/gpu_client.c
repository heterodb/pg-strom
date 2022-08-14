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

static dlist_head	gpu_connections_list;

struct GpuConnection
{
	dlist_node		chain;	/* link to gpuserv_connection_slots */
	int				cuda_dindex;
	volatile pgsocket sockfd;
	ResourceOwner	resowner;
	pthread_t		worker;
	pthread_mutex_t	mutex;
	int				num_running_tasks;
	dlist_head		ready_commands_list;
	dlist_head		active_commands_list;
	int				terminated;	/* positive: normal exit
								 * negative: exit with error */
	char			errmsg[200];
};

/*
 * Worker thread to receive response messages
 */
static void *
__gpuConnectSessionWorkerAlloc(void *__priv, size_t sz)
{
	return malloc(sz);
}

static void
__gpuConnectSessionWorkerAttach(void *__priv, XpuCommand *xcmd)
{
	GpuConnection *conn = __priv;

	xcmd->priv = conn;
	pthreadMutexLock(&conn->mutex);
	Assert(conn->num_running_tasks > 0);
	dlist_push_tail(&conn->ready_commands_list, &xcmd->chain);
	conn->num_running_tasks--;
	pthreadMutexUnlock(&conn->mutex);
	SetLatch(MyLatch);
}

static void *
__gpuConnectSessionWorker(void *__priv)
{
	GpuConnection *conn = __priv;
	char		errmsg[200];

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
			snprintf(errmsg, sizeof(errmsg),
					 "failed on poll(2): %m");
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
				return NULL;
			}
			else if (pfd.revents & POLLIN)
			{
				if (pgstrom_receive_xpu_command(conn->sockfd,
												__gpuConnectSessionWorkerAlloc,
												__gpuConnectSessionWorkerAttach,
												conn,
												__FUNCTION__,
												errmsg, sizeof(errmsg)) < 0)
					break;
			}
		}
	}
	pthreadMutexLock(&conn->mutex);
	conn->terminated = -1;
	strcpy(conn->errmsg, errmsg);
	pthreadMutexUnlock(&conn->mutex);
	return NULL;
}

/*
 * gpuClientSendCommand
 */
void
gpuClientSendCommand(GpuConnection *conn, XpuCommand *xcmd)
{
	int		sockfd = conn->sockfd;
	ssize_t	nbytes;

	pthreadMutexLock(&conn->mutex);
	conn->num_running_tasks++;
	pthreadMutexUnlock(&conn->mutex);

	nbytes = __writeFile(sockfd, xcmd, xcmd->length);
	if (nbytes != xcmd->length)
		elog(ERROR, "unable to send XPU command to GPU service (%zd of %u): %m",
			 nbytes, xcmd->length);
}

/*
 * gpuClientGetResponse
 */
XpuCommand *
gpuClientGetResponse(GpuConnection *conn, long timeout)
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
			elog(ERROR, "GPU%d) %s", conn->cuda_dindex, conn->errmsg);
		}
		if (!dlist_is_empty(&conn->ready_commands_list))
		{
			dnode = dlist_pop_head_node(&conn->ready_commands_list);
			xcmd = dlist_container(XpuCommand, chain, dnode);
			dlist_push_tail(&conn->active_commands_list, &xcmd->chain);
			pthreadMutexUnlock(&conn->mutex);
			return xcmd;
		}
		/* if no running tasks, it makes no sense to wait for */
		if (conn->num_running_tasks == 0 || conn->terminated != 0)
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
 * gpuClientPutResponse
 */
void
gpuClientPutResponse(XpuCommand *xcmd)
{
	GpuConnection  *conn = xcmd->priv;
	
	pthreadMutexLock(&conn->mutex);
	dlist_delete(&xcmd->chain);
	pthreadMutexUnlock(&conn->mutex);
	free(xcmd);
}

/*
 * __gpuClientInitSession
 */
static GpuConnection *
__gpuClientInitSession(GpuConnection *conn,
					   const kern_session_info *session)
{
	XpuCommand *resp;

	gpuClientSendCommand(conn, (XpuCommand *)session);
	resp = gpuClientGetResponse(conn, -1);
	if (resp->tag != XpuCommandTag__Success)
		elog(ERROR, "GPU:OpenSession failed - %s", resp->data);
	gpuClientPutResponse(resp);

	return conn;
}

/*
 * __gpuClientChooseDevice
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

/*
 * gpuClientOpenSession
 */
GpuConnection *
gpuClientOpenSession(const Bitmapset *gpuset,
					 const kern_session_info *session)
{
	int				cuda_dindex = __gpuClientChooseDevice(gpuset);
	GpuConnection  *conn;
	struct sockaddr_un addr;
	pgsocket		sockfd;

	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");

	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path),
			 ".pg_strom.%u.gpu%u.sock",
			 PostmasterPid, cuda_dindex);
	if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
	{
		int		__errno = errno;

		close(sockfd);
		elog(ERROR, "failed on connect('%s'): %s",
			 addr.sun_path, strerror(__errno));
	}
	/* remember the connection */
	conn = calloc(1, sizeof(GpuConnection));
	if (!conn)
	{
		close(sockfd);
		elog(ERROR, "out of memory");
	}
	conn->cuda_dindex = cuda_dindex;
	conn->sockfd = sockfd;
	conn->resowner = CurrentResourceOwner;
	pthreadMutexInit(&conn->mutex);
	dlist_init(&conn->ready_commands_list);
	dlist_init(&conn->active_commands_list);
	if ((errno = pthread_create(&conn->worker, NULL,
								__gpuConnectSessionWorker, conn)) != 0)
	{
		free(conn);
		close(sockfd);
		elog(ERROR, "failed on pthread_create: %m");
	}
	dlist_push_tail(&gpu_connections_list, &conn->chain);

	return __gpuClientInitSession(conn, session);
}

/*
 * gpuClientCloseSession
 */
void
gpuClientCloseSession(GpuConnection *conn)
{
	XpuCommand *xcmd;
	dlist_node *dnode;

	/* ensure termination of worker thread */
	close(conn->sockfd);
	conn->sockfd = -1;
	pg_memory_barrier();
	pthread_kill(conn->worker, SIGPOLL);
	pthread_join(conn->worker, NULL);

	while (!dlist_is_empty(&conn->ready_commands_list))
	{
		dnode = dlist_pop_head_node(&conn->ready_commands_list);
		xcmd = dlist_container(XpuCommand, chain, dnode);
		free(xcmd);
	}
	while (!dlist_is_empty(&conn->active_commands_list))
	{
		dnode = dlist_pop_head_node(&conn->active_commands_list);
		xcmd = dlist_container(XpuCommand, chain, dnode);
		free(xcmd);
	}
	dlist_delete(&conn->chain);
	free(conn);
}

/*
 * gpuclientCleanupConnections
 */
static void
gpuclientCleanupConnections(ResourceReleasePhase phase,
						  bool isCommit,
						  bool isTopLevel,
						  void *arg)
{
	dlist_mutable_iter	iter;

	if (phase != RESOURCE_RELEASE_BEFORE_LOCKS)
		return;

	dlist_foreach_modify(iter, &gpu_connections_list)
	{
		GpuConnection  *conn = dlist_container(GpuConnection,
											   chain, iter.cur);
		if (conn->resowner == CurrentResourceOwner)
		{
			if (isCommit)
				elog(LOG, "Bug? GPU connection %d is not closed on ExecEnd",
					 conn->sockfd);
			gpuClientCloseSession(conn);
		}
	}
}

void
pgstrom_init_gpu_client(void)
{
	dlist_init(&gpu_connections_list);
	RegisterResourceReleaseCallback(gpuclientCleanupConnections, NULL);
}
