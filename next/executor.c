/*
 * executor.c
 *
 * Common routines related to query execution phase
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/*
 * XpuConnection
 */
struct XpuConnection
{
	dlist_node		chain;	/* link to gpuserv_connection_slots */
	char			devname[32];
	volatile pgsocket sockfd;
	volatile int	terminated;		/* positive: normal exit
									 * negative: exit by errors */
	ResourceOwner	resowner;
	pthread_t		worker;
	pthread_mutex_t	mutex;
	int				num_running_cmds;
	int				num_ready_cmds;
	dlist_head		ready_cmds_list;	/* ready, but not fetched yet  */
	dlist_head		active_cmds_list;	/* currently in-use */
	kern_errorbuf	errorbuf;
};

/* see xact.c */
extern int				nParallelCurrentXids;
extern TransactionId   *ParallelCurrentXids;

/* static variables */
static dlist_head		xpu_connections_list;

/*
 * xpuConnectReceiveCommands
 */
int
xpuConnectReceiveCommands(pgsocket sockfd,
						  void *(*alloc_f)(void *priv, size_t sz),
						  void  (*attach_f)(void *priv, XpuCommand *xcmd),
						  void *priv,
						  const char *error_label)
{
	char		buffer_local[2 * BLCKSZ];
	char	   *buffer;
	size_t		bufsz, offset;
	ssize_t		nbytes;
	int			recv_flags;
	int			count = 0;
	XpuCommand *curr = NULL;

#define __fprintf(filp,fmt,...)										\
	fprintf((filp), "[%s; %s:%d] " fmt "\n",						\
			error_label, __FILE_NAME__, __LINE__, ##__VA_ARGS__)
	
restart:
	buffer = buffer_local;
	bufsz  = sizeof(buffer_local);
	offset = 0;
	recv_flags = MSG_DONTWAIT;
	curr   = NULL;

	for (;;)
	{
		nbytes = recv(sockfd, buffer + offset, bufsz - offset, recv_flags);
		if (nbytes < 0)
		{
			if (errno == EINTR)
				continue;
			if (errno == EAGAIN || errno == EWOULDBLOCK)
			{
				/*
				 * If we are under the read of a XpuCommand fraction,
				 * we have to wait for completion of the XpuCommand.
				 * (Its peer side should send the entire command very
				 * soon.)
				 * Elsewhere, we have no queued XpuCommand right now.
				 */
				if (!curr && offset == 0)
					return count;
				/* next recv(2) shall be blocking call */
				recv_flags = 0;
				continue;
			}
			__fprintf(stderr, "failed on recv(%d, %p, %ld, %d): %m",
					  sockfd, buffer + offset, bufsz - offset, recv_flags);
			return -1;
		}
		else if (nbytes == 0)
		{
			/* end of the stream */
			if (curr || offset > 0)
			{
				__fprintf(stderr, "connection closed during XpuCommand read");
				return -1;
			}
			return count;
		}

		offset += nbytes;
		if (!curr)
		{
			XpuCommand *temp, *xcmd;
		next:
			if (offset < offsetof(XpuCommand, u))
			{
				if (buffer != buffer_local)
				{
					memmove(buffer_local, buffer, offset);
					buffer = buffer_local;
					bufsz  = sizeof(buffer_local);
				}
				recv_flags = 0;		/* next recv(2) shall be blockable */
				continue;
			}
			temp = (XpuCommand *)buffer;
			if (temp->length <= offset)
			{
				assert(temp->magic == XpuCommandMagicNumber);
				xcmd = alloc_f(priv, temp->length);
				if (!xcmd)
				{
					__fprintf(stderr, "out of memory (sz=%lu): %m", temp->length);
					return -1;
				}
				memcpy(xcmd, temp, temp->length);
				attach_f(priv, xcmd);
				count++;

				if (temp->length == offset)
					goto restart;
				/* read remained portion, if any */
				buffer += temp->length;
				offset -= temp->length;
				goto next;
			}
			else
			{
				curr = alloc_f(priv, temp->length);
				if (!curr)
				{
					__fprintf(stderr, "out of memory (sz=%lu): %m", temp->length);
					return -1;
				}
				memcpy(curr, temp, offset);
				buffer = (char *)curr;
				bufsz  = temp->length;
				recv_flags = 0;		/* blocking enabled */
			}
		}
		else if (offset >= curr->length)
		{
			assert(curr->magic == XpuCommandMagicNumber);
			assert(curr->length == offset);
			attach_f(priv, curr);
			count++;
			goto restart;
		}
	}
	__fprintf(stderr, "Bug? should not break this loop");
	return -1;
#undef __fprintf
}

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
				__fprintf(stderr, "peer socket closed.");
				pthreadMutexLock(&conn->mutex);
				conn->terminated = 1;
				SetLatch(MyLatch);
				pthreadMutexUnlock(&conn->mutex);
				return NULL;
			}
			else if (pfd.revents & POLLIN)
			{
				if (xpuConnectReceiveCommands(conn->sockfd,
											  __xpuConnectSessionWorkerAlloc,
											  __xpuConnectSessionWorkerAttach,
											  conn, conn->devname) < 0)
					break;
			}
		}
	}
	pthreadMutexLock(&conn->mutex);
	conn->terminated = -1;
	SetLatch(MyLatch);
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

/* ----------------------------------------------------------------
 *
 * Routines to build session-information
 *
 * ----------------------------------------------------------------
 */
static uint32_t
__build_session_xact_id_vector(StringInfo buf)
{
	uint32_t	offset = 0;

	if (nParallelCurrentXids > 0)
	{
		uint32_t	sz = VARHDRSZ + sizeof(TransactionId) * nParallelCurrentXids;
		uint32_t	temp;

		SET_VARSIZE(&temp, sz);
		offset = __appendBinaryStringInfo(buf, &temp, sizeof(uint32_t));
		appendBinaryStringInfo(buf, (char *)ParallelCurrentXids,
							   sizeof(TransactionId) * nParallelCurrentXids);
	}
	return offset;
}

static uint32_t
__build_session_timezone(StringInfo buf)
{
	uint32_t	offset = 0;

	if (session_timezone)
	{
		offset = __appendBinaryStringInfo(buf, session_timezone,
										  sizeof(struct pg_tz));
	}
	return offset;
}

static uint32_t
__build_session_encode(StringInfo buf)
{
	xpu_encode_info encode;

	memset(&encode, 0, sizeof(xpu_encode_info));
	strncpy(encode.encname,
			GetDatabaseEncodingName(),
			sizeof(encode.encname));
	encode.enc_maxlen = pg_database_encoding_max_length();
	encode.enc_mblen = NULL;

	return __appendBinaryStringInfo(buf, &encode, sizeof(xpu_encode_info));
}

const XpuCommand *
pgstromBuildSessionInfo(PlanState *ps,
						List *used_params,
						uint32_t kcxt_extra_bufsz,
						uint32_t kcxt_kvars_nslots,
						const bytea *xpucode_scan_quals,
						const bytea *xpucode_scan_projs)
{
	ExprContext	   *econtext = ps->ps_ExprContext;
	ParamListInfo	param_info = econtext->ecxt_param_list_info;
	uint32_t		nparams = (param_info ? param_info->numParams : 0);
	uint32_t		session_sz = offsetof(kernSessionInfo, poffset[nparams]);
	StringInfoData	buf;
	XpuCommand	   *xcmd;
	kernSessionInfo *session;

	initStringInfo(&buf);
	__appendZeroStringInfo(&buf, session_sz);
	session = alloca(session_sz);
	memset(session, 0, session_sz);

	/* put XPU code */
	if (xpucode_scan_quals)
	{
		session->xpucode_scan_quals =
			__appendBinaryStringInfo(&buf, xpucode_scan_quals,
									 VARSIZE(xpucode_scan_quals));
	}
	if (xpucode_scan_projs)
	{
		session->xpucode_scan_projs =
			__appendBinaryStringInfo(&buf, xpucode_scan_projs,
									 VARSIZE(xpucode_scan_projs));
	}

	/* put executor parameters */
	if (param_info)
	{
		ListCell   *lc;

		session->nparams = nparams;
		foreach (lc, used_params)
		{
			Param  *param = lfirst(lc);
			Datum	param_value;
			bool	param_isnull;
			uint32_t	offset;

			if (param->paramkind == PARAM_EXEC)
			{
				/* See ExecEvalParamExec */
				ParamExecData  *prm = &(econtext->ecxt_param_exec_vals[param->paramid]);

				if (prm->execPlan)
				{
					/* Parameter not evaluated yet, so go do it */
					ExecSetParamPlan(prm->execPlan, econtext);
					/* ExecSetParamPlan should have processed this param... */
					Assert(prm->execPlan == NULL);
				}
				param_isnull = prm->isnull;
				param_value  = prm->value;
			}
			else if (param->paramkind == PARAM_EXTERN)
			{
				/* See ExecEvalParamExtern */
				ParamExternData *prm, prmData;

				if (param_info->paramFetch != NULL)
					prm = param_info->paramFetch(param_info,
												 param->paramid,
												 false, &prmData);
				else
					prm = &param_info->params[param->paramid - 1];
				if (!OidIsValid(prm->ptype))
					elog(ERROR, "no value found for parameter %d", param->paramid);
				if (prm->ptype != param->paramtype)
					elog(ERROR, "type of parameter %d (%s) does not match that when preparing the plan (%s)",
						 param->paramid,
						 format_type_be(prm->ptype),
						 format_type_be(param->paramtype));
				param_isnull = prm->isnull;
				param_value  = prm->value;
			}
			else
			{
				elog(ERROR, "Bug? unexpected parameter kind: %d",
					 (int)param->paramkind);
			}

			if (param_isnull)
				offset = 0;
			else
			{
				int16	typlen;
				bool	typbyval;

				get_typlenbyval(param->paramtype, &typlen, &typbyval);
				if (typbyval)
				{
					offset = __appendBinaryStringInfo(&buf,
													  (char *)&param_value,
													  typlen);
				}
				else if (typlen > 0)
				{
					offset = __appendBinaryStringInfo(&buf,
													  DatumGetPointer(param_value),
													  typlen);
				}
				else if (typlen == -1)
				{
					struct varlena *temp = PG_DETOAST_DATUM(param_value);

					offset = __appendBinaryStringInfo(&buf,
													  DatumGetPointer(temp),
													  VARSIZE(temp));
					if (param_value != PointerGetDatum(temp))
						pfree(temp);
				}
				else
				{
					elog(ERROR, "Not a supported data type for kernel parameter: %s",
						 format_type_be(param->paramtype));
				}
			}
			Assert(param->paramid >= 0 && param->paramid < nparams);
			session->poffset[param->paramid] = offset;
		}
	}
	/* other database session information */
	session->kcxt_extra_bufsz = kcxt_extra_bufsz;
	session->kcxt_kvars_nslots = kcxt_kvars_nslots;
	session->xactStartTimestamp = GetCurrentTransactionStartTimestamp();
	session->xact_id_array = __build_session_xact_id_vector(&buf);
	session->session_timezone = __build_session_timezone(&buf);
	session->session_encode = __build_session_encode(&buf);
	memcpy(buf.data, session, session_sz);

	/* setup XpuCommand */
	xcmd = palloc(offsetof(XpuCommand, u.session) + buf.len);
	memset(xcmd, 0, offsetof(XpuCommand, u.session));
	xcmd->magic = XpuCommandMagicNumber;
	xcmd->tag = XpuCommandTag__OpenSession;
	xcmd->length = offsetof(XpuCommand, u.session) + buf.len;
	memcpy(&xcmd->u.session, buf.data, buf.len);
	pfree(buf.data);

	return xcmd;
}

/*
 * __pickupNextXpuCommand
 *
 * MEMO: caller must hold 'conn->mutex'
 */
static XpuCommand *
__pickupNextXpuCommand(XpuConnection *conn)
{
	XpuCommand	   *xcmd;
	dlist_node	   *dnode;

	Assert(conn->num_ready_cmds > 0);
	dnode = dlist_pop_head_node(&conn->ready_cmds_list);
	xcmd = dlist_container(XpuCommand, chain, dnode);
	dlist_push_tail(&conn->active_cmds_list, &xcmd->chain);
	conn->num_ready_cmds--;

	return xcmd;
}

static XpuCommand *
__waitAndFetchNextXpuCommand(pgstromTaskState *pts, bool try_final_callback)
{
	XpuConnection  *conn = pts->conn;
	XpuCommand	   *xcmd;
	int				ev;

	pthreadMutexLock(&conn->mutex);
	for (;;)
	{
		/* device error checks */
		if (conn->errorbuf.errcode != ERRCODE_STROM_SUCCESS)
		{
			pthreadMutexUnlock(&conn->mutex);
			ereport(ERROR,
					(errcode(conn->errorbuf.errcode),
					 errmsg("%s:%d  %s",
							conn->errorbuf.filename,
							conn->errorbuf.lineno,
							conn->errorbuf.message),
					 errhint("device at %s, function at %s",
							 conn->devname,
							 conn->errorbuf.funcname)));

		}
		if (!dlist_is_empty(&conn->ready_cmds_list))
		{
			/* ok, ready commands we have */
			break;
		}
		else if (conn->num_running_cmds > 0)
		{
			/* wait for the running commands */
			pthreadMutexUnlock(&conn->mutex);
		}
		else
		{
			pthreadMutexUnlock(&conn->mutex);
			if (!try_final_callback)
				return NULL;
			if (!pts->cb_final_chunk || pts->final_done)
				return NULL;
			xcmd = pts->cb_final_chunk(pts);
			if (!xcmd)
				return NULL;
			xpuClientSendCommand(conn, xcmd);
		}
		ev = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_TIMEOUT |
					   WL_POSTMASTER_DEATH,
					   1000L,
					   PG_WAIT_EXTENSION);
		if (ev & WL_POSTMASTER_DEATH)
			ereport(FATAL,
					(errcode(ERRCODE_ADMIN_SHUTDOWN),
					 errmsg("Unexpected Postmaster dead")));
		pthreadMutexLock(&conn->mutex);
	}
	xcmd = __pickupNextXpuCommand(conn);
	pthreadMutexUnlock(&conn->mutex);

	return xcmd;
}

static XpuCommand *
__fetchNextXpuCommand(pgstromTaskState *pts)
{
	XpuConnection  *conn = pts->conn;
	XpuCommand	   *xcmd;
	int				ev;

	while (!pts->scan_done)
	{
		pthreadMutexLock(&conn->mutex);
		/* device error checks */
		if (conn->errorbuf.errcode != ERRCODE_STROM_SUCCESS)
		{
			pthreadMutexUnlock(&conn->mutex);
			ereport(ERROR,
					(errcode(conn->errorbuf.errcode),
					 errmsg("%s:%d  %s",
							conn->errorbuf.filename,
							conn->errorbuf.lineno,
							conn->errorbuf.message),
					 errhint("device at %s, function at %s",
							 conn->devname,
							 conn->errorbuf.funcname)));
		}

		if ((conn->num_running_cmds +
			 conn->num_ready_cmds) < pgstrom_max_async_tasks &&
			(dlist_is_empty(&conn->ready_cmds_list) ||
			 conn->num_running_cmds < pgstrom_max_async_tasks / 2))
		{
			/*
			 * xPU service still has margin to enqueue new commands.
			 * If we have no ready commands or number of running commands
			 * are less than pg_strom.max_async_tasks/2, we try to load
			 * the next chunk and enqueue this command.
			 */
			pthreadMutexUnlock(&conn->mutex);
			xcmd = pts->cb_next_chunk(pts);
			if (!xcmd)
			{
				/* end of scan */
				pts->scan_done = true;
				break;
			}
			xpuClientSendCommand(conn, xcmd);
		}
		else if (!dlist_is_empty(&conn->ready_cmds_list))
		{
			xcmd = __pickupNextXpuCommand(conn);
			pthreadMutexUnlock(&conn->mutex);

			return xcmd;
		}
		else if (conn->num_running_cmds > 0)
		{
			/*
			 * This block means we already runs enough number of concurrent
			 * tasks, but none of them are already finished.
			 * So, let's wait for the response.
			 */
			ResetLatch(MyLatch);
			pthreadMutexUnlock(&conn->mutex);

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   1000L,
						   PG_WAIT_EXTENSION);
			if (ev & WL_POSTMASTER_DEATH)
				ereport(FATAL,
						(errcode(ERRCODE_ADMIN_SHUTDOWN),
						 errmsg("Unexpected Postmaster dead")));
		}
		else
		{
			/*
			 * Unfortunately, we touched the threshold. Take a short wait
			 */
			pthreadMutexUnlock(&conn->mutex);
			pg_usleep(20000L);		/* 20ms */
		}
	}
	return __waitAndFetchNextXpuCommand(pts, true);
}

/*
 * pgstromExecTaskState
 */
TupleTableSlot *
pgstromExecTaskState(pgstromTaskState *pts)
{
	TupleTableSlot *slot = NULL;

	while (!pts->curr_resp || !(slot = pts->cb_next_tuple(pts)))
	{
		if (pts->curr_resp)
			free(pts->curr_resp);
		pts->curr_resp = __fetchNextXpuCommand(pts);
		if (!pts->curr_resp)
			return NULL;
		pts->curr_index = 0;
	}
	return slot;
}

/*
 * __xpuClientOpenSession
 */
static void
__xpuClientOpenSession(pgstromTaskState *pts,
					   const XpuCommand *session,
					   pgsocket sockfd,
					   const char *devname)
{
	XpuConnection  *conn;
	XpuCommand	   *resp;
	int				rv;

	Assert(!pts->conn);
	conn = calloc(1, sizeof(XpuConnection));
	if (!conn)
	{
		close(sockfd);
		elog(ERROR, "out of memory");
	}
	strncpy(conn->devname, devname, 32);
	conn->sockfd = sockfd;
	conn->resowner = CurrentResourceOwner;
	conn->worker = pthread_self();	/* to be over-written by worker's-id */
	pthreadMutexInit(&conn->mutex);
	conn->num_running_cmds = 0;
	conn->num_ready_cmds = 0;
	dlist_init(&conn->ready_cmds_list);
	dlist_init(&conn->active_cmds_list);
	dlist_push_tail(&xpu_connections_list, &conn->chain);
	pts->conn = conn;

	/*
	 * Ok, sockfd and conn shall be automatically released on ereport()
	 * after that.
	 */
	if ((rv = pthread_create(&conn->worker, NULL,
							 __xpuConnectSessionWorker, conn)) != 0)
		elog(ERROR, "failed on pthread_create: %s", strerror(rv));

	/*
	 * Initialize the new session
	 */
	Assert(session->tag == XpuCommandTag__OpenSession);
	xpuClientSendCommand(conn, session);
	resp = __waitAndFetchNextXpuCommand(pts, false);
	if (!resp)
		elog(ERROR, "Bug? %s:OpenSession response is missing", conn->devname);
	if (resp->tag != XpuCommandTag__Success)
		elog(ERROR, "%s:OpenSession failed - %s (%s:%d %s)",
			 conn->devname,
			 resp->u.error.message,
			 resp->u.error.filename,
			 resp->u.error.lineno,
			 resp->u.error.funcname);
	xpuClientPutResponse(resp);
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

void
gpuClientOpenSession(pgstromTaskState *pts,
					 const Bitmapset *gpuset,
					 const XpuCommand *session)
{
	struct sockaddr_un addr;
	pgsocket	sockfd;
	int			cuda_dindex = __gpuClientChooseDevice(gpuset);
	char		namebuf[32];

	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path),
			 ".pg_strom.%u.gpu%u.sock",
			 PostmasterPid, cuda_dindex);
	if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
	{
		close(sockfd);
		elog(ERROR, "failed on connect('%s'): %m", addr.sun_path);
	}
	snprintf(namebuf, sizeof(namebuf), "GPU-%d", cuda_dindex);

	__xpuClientOpenSession(pts, session, sockfd, namebuf);
}

/*
 * pgstrom_init_executor
 */
void
pgstrom_init_executor(void)
{
	dlist_init(&xpu_connections_list);
	RegisterResourceReleaseCallback(xpuclientCleanupConnections, NULL);
}
