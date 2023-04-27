/*
 * executor.c
 *
 * Common routines related to query execution phase
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
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
	int				dev_index;		/* cuda_dindex or dpu_endpoint_id */
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
static bool				pgstrom_use_debug_code;		/* GUC */

/*
 * Worker thread to receive response messages
 */
static void *
__xpuConnectAllocCommand(void *__priv, size_t sz)
{
	return malloc(sz);
}

static void
__xpuConnectAttachCommand(void *__priv, XpuCommand *xcmd)
{
	XpuConnection *conn = __priv;

	xcmd->priv = conn;
	pthreadMutexLock(&conn->mutex);
	Assert(conn->num_running_cmds > 0);
	conn->num_running_cmds--;
	if (xcmd->tag == XpuCommandTag__Error)
	{
		if (conn->errorbuf.errcode == ERRCODE_STROM_SUCCESS)
		{
			Assert(xcmd->u.error.errcode != ERRCODE_STROM_SUCCESS);
			memcpy(&conn->errorbuf, &xcmd->u.error, sizeof(kern_errorbuf));
		}
		free(xcmd);
	}
	else
	{
		Assert(xcmd->tag == XpuCommandTag__Success ||
			   xcmd->tag == XpuCommandTag__SuccessAndRightOuter ||
			   xcmd->tag == XpuCommandTag__CPUFallback);
		dlist_push_tail(&conn->ready_cmds_list, &xcmd->chain);
		conn->num_ready_cmds++;
	}
	SetLatch(MyLatch);
	pthreadMutexUnlock(&conn->mutex);
}
TEMPLATE_XPU_CONNECT_RECEIVE_COMMANDS(__xpuConnect)

static void *
__xpuConnectSessionWorker(void *__priv)
{
	XpuConnection *conn = __priv;

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
			fprintf(stderr, "[%s; %s:%d] failed on poll(2): %m\n",
					conn->devname, __FILE_NAME__, __LINE__);
			break;
		}
		else if (nevents > 0)
		{
			Assert(nevents == 1);
			if (pfd.revents & ~POLLIN)
			{
				pthreadMutexLock(&conn->mutex);
				conn->terminated = 1;
				SetLatch(MyLatch);
				pthreadMutexUnlock(&conn->mutex);
				return NULL;
			}
			else if (pfd.revents & POLLIN)
			{
				if (__xpuConnectReceiveCommands(conn->sockfd,
												conn,
												conn->devname) < 0)
					break;
			}
		}
	}
	pthreadMutexLock(&conn->mutex);
	conn->terminated = -1;
	SetLatch(MyLatch);
	pthreadMutexUnlock(&conn->mutex);

	return NULL;
}

/*
 * xpuClientSendCommand
 */
void
xpuClientSendCommand(XpuConnection *conn, const XpuCommand *xcmd)
{
	int			sockfd = conn->sockfd;
	const char *buf = (const char *)xcmd;
	size_t		len = xcmd->length;
	ssize_t		nbytes;

	pthreadMutexLock(&conn->mutex);
	conn->num_running_cmds++;
	pthreadMutexUnlock(&conn->mutex);

	while (len > 0)
	{
		nbytes = write(sockfd, buf, len);
		if (nbytes > 0)
		{
			buf += nbytes;
			len -= nbytes;
		}
		else if (nbytes == 0)
			elog(ERROR, "unable to send xPU command to the service");
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			elog(ERROR, "failed on write(2): %m");
	}
}

/*
 * xpuClientSendCommandIOV
 */
static void
xpuClientSendCommandIOV(XpuConnection *conn, struct iovec *iov, int iovcnt)
{
	int			sockfd = conn->sockfd;
	ssize_t		nbytes;

	Assert(iovcnt > 0);
	pthreadMutexLock(&conn->mutex);
	conn->num_running_cmds++;
	pthreadMutexUnlock(&conn->mutex);

	while (iovcnt > 0)
	{
		nbytes = writev(sockfd, iov, iovcnt);
		if (nbytes > 0)
		{
			do {
				if (iov->iov_len <= nbytes)
				{
					nbytes -= iov->iov_len;
					iov++;
					iovcnt--;
				}
				else
				{
					iov->iov_base = (char *)iov->iov_base + nbytes;
					iov->iov_len -= nbytes;
					break;
				}
			} while (iovcnt > 0 && nbytes > 0);
		}
		else if (nbytes == 0)
			elog(ERROR, "unable to send xPU command to the service");
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			elog(ERROR, "failed on writev(2): %m");
	}
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
static void
__build_session_param_info(pgstromTaskState *pts,
						   kern_session_info *session,
						   StringInfo buf)
{
	pgstromPlanInfo *pp_info = pts->pp_info;
	ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
	ParamListInfo	param_info = econtext->ecxt_param_list_info;
	ListCell	   *lc;

	Assert(param_info != NULL);
	session->nparams = param_info->numParams;
	foreach (lc, pp_info->used_params)
	{
		Param	   *param = lfirst(lc);
		Datum		param_value;
		bool		param_isnull;
		uint32_t	offset;

		Assert(param->paramid >= 0 &&
			   param->paramid < session->nparams);
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
				offset = __appendBinaryStringInfo(buf,
												  (char *)&param_value,
												  typlen);
			}
			else if (typlen > 0)
			{
				offset = __appendBinaryStringInfo(buf,
												  DatumGetPointer(param_value),
												  typlen);
			}
			else if (typlen == -1)
			{
				struct varlena *temp = PG_DETOAST_DATUM(param_value);

				offset = __appendBinaryStringInfo(buf,
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
		Assert(param->paramid >= 0 && param->paramid < session->nparams);
		session->poffset[param->paramid] = offset;
	}
}

static uint32_t
__build_session_xact_state(StringInfo buf)
{
	Size		bufsz;
	char	   *buffer;

	bufsz = EstimateTransactionStateSpace();
	buffer = alloca(bufsz);
	SerializeTransactionState(bufsz, buffer);

	return __appendBinaryStringInfo(buf, buffer, bufsz);
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
pgstromBuildSessionInfo(pgstromTaskState *pts,
						uint32_t join_inner_handle,
						TupleDesc groupby_tdesc_final)
{
	pgstromSharedState *ps_state = pts->ps_state;
	pgstromPlanInfo *pp_info = pts->pp_info;
	ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
	ParamListInfo	param_info = econtext->ecxt_param_list_info;
	uint32_t		nparams = (param_info ? param_info->numParams : 0);
	uint32_t		kvars_nbytes;
	uint32_t		kvars_nslots;
	uint32_t		session_sz;
	kern_session_info *session;
	ListCell	   *lc;
	XpuCommand	   *xcmd;
	StringInfoData	buf;
	bytea		   *xpucode;

	initStringInfo(&buf);
	session_sz = offsetof(kern_session_info, poffset[nparams]);
	session = alloca(session_sz);
	memset(session, 0, session_sz);
	__appendZeroStringInfo(&buf, session_sz);
	if (param_info)
		__build_session_param_info(pts, session, &buf);
	if (pp_info->kexp_scan_kvars_load)
	{
		xpucode = pp_info->kexp_scan_kvars_load;
		session->xpucode_scan_load_vars =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_scan_quals)
	{
		xpucode = pp_info->kexp_scan_quals;
		session->xpucode_scan_quals =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_join_kvars_load_packed)
	{
		xpucode = pp_info->kexp_join_kvars_load_packed;
		session->xpucode_join_load_vars_packed =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_join_quals_packed)
	{
		xpucode = pp_info->kexp_join_quals_packed;
		session->xpucode_join_quals_packed =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_hash_keys_packed)
	{
		xpucode = pp_info->kexp_hash_keys_packed;
		session->xpucode_hash_values_packed =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_gist_quals_packed)
	{
		xpucode = pp_info->kexp_gist_quals_packed;
		session->xpucode_gist_quals_packed =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_projection)
	{
		xpucode = pp_info->kexp_projection;
		session->xpucode_projection =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_groupby_keyhash)
	{
		xpucode = pp_info->kexp_groupby_keyhash;
		session->xpucode_groupby_keyhash =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_groupby_keyload)
	{
		xpucode = pp_info->kexp_groupby_keyload;
		session->xpucode_groupby_keyload =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_groupby_keycomp)
	{
		xpucode = pp_info->kexp_groupby_keycomp;
		session->xpucode_groupby_keycomp =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_groupby_actions)
	{
		xpucode = pp_info->kexp_groupby_actions;
		session->xpucode_groupby_actions =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (groupby_tdesc_final)
	{
		size_t		sz = estimate_kern_data_store(groupby_tdesc_final);
		kern_data_store *kds_temp = (kern_data_store *)alloca(sz);
		char		format = KDS_FORMAT_ROW;
		uint32_t	hash_nslots = 0;
		size_t		kds_length = (4UL << 20);	/* 4MB */

		if (pp_info->kexp_groupby_keyhash &&
			pp_info->kexp_groupby_keyload &&
			pp_info->kexp_groupby_keycomp)
		{
			format = KDS_FORMAT_HASH;
			hash_nslots = 20000; //to be estimated using num_groups
			kds_length = (1UL << 30);			/* 1GB */
		}
		setup_kern_data_store(kds_temp, groupby_tdesc_final, kds_length, format);
		kds_temp->hash_nslots = hash_nslots;
		session->groupby_kds_final = __appendBinaryStringInfo(&buf, kds_temp, sz);
	}
	/* other database session information */
	kvars_nslots = list_length(pp_info->kvars_depth);
	Assert(kvars_nslots == list_length(pp_info->kvars_resno) &&
		   kvars_nslots == list_length(pp_info->kvars_types));
	kvars_nbytes = (sizeof(kern_variable) * kvars_nslots +
					sizeof(int)           * kvars_nslots);
	foreach (lc, pp_info->kvars_types)
	{
		Oid		type_oid = lfirst_oid(lc);
		devtype_info *dtype;

		if (OidIsValid(type_oid) &&
			(dtype = pgstrom_devtype_lookup(type_oid)) != NULL)
		{
			kvars_nbytes = TYPEALIGN(dtype->type_alignof, kvars_nbytes);
			kvars_nbytes += dtype->type_sizeof;
		}
	}
	kvars_nbytes = MAXALIGN(kvars_nbytes);
	session->query_plan_id = ps_state->query_plan_id;
	session->kcxt_extra_bufsz = pp_info->extra_bufsz;
	session->kcxt_kvars_nslots = kvars_nslots;
	session->kcxt_kvars_nbytes = kvars_nbytes;
	session->xpucode_use_debug_code = pgstrom_use_debug_code;
	session->xactStartTimestamp = GetCurrentTransactionStartTimestamp();
	session->session_xact_state = __build_session_xact_state(&buf);
	session->session_timezone = __build_session_timezone(&buf);
	session->session_encode = __build_session_encode(&buf);
	session->pgsql_port_number = PostPortNumber;
	session->pgsql_plan_node_id = pts->css.ss.ps.plan->plan_node_id;
	session->join_inner_handle = join_inner_handle;
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
 * pgstromTaskStateBeginScan
 */
static bool
pgstromTaskStateBeginScan(pgstromTaskState *pts)
{
	pgstromSharedState *ps_state = pts->ps_state;
	XpuConnection  *conn = pts->conn;
	uint32_t		curval, newval;

	Assert(conn != NULL);
	curval = pg_atomic_read_u32(&ps_state->scan_task_control);
	do {
		if ((curval & 1) != 0)
			return false;
		newval = curval + 2;
	} while (!pg_atomic_compare_exchange_u32(&ps_state->scan_task_control,
											 &curval, newval));
	SpinLockAcquire(pts->rjoin_control_lock);
	pts->rjoin_control_array[conn->dev_index]++;
	SpinLockRelease(pts->rjoin_control_lock);

	return true;
}

/*
 * pgstromTaskStateEndScan
 */
static bool
pgstromTaskStateEndScan(pgstromTaskState *pts, kern_final_task *kfin)
{
	pgstromSharedState *ps_state = pts->ps_state;
	XpuConnection  *conn = pts->conn;
	uint32_t		curval, newval;

	Assert(conn != NULL);
	memset(kfin, 0, sizeof(kern_final_task));
	curval = pg_atomic_read_u32(&ps_state->scan_task_control);
	do {
		Assert(curval >= 2);
		newval = ((curval - 2) | 1);
	} while (!pg_atomic_compare_exchange_u32(&ps_state->scan_task_control,
											 &curval, newval));
	if (newval == 1)
		kfin->final_plan_node = true;
	SpinLockAcquire(pts->rjoin_control_lock);
	Assert(pts->rjoin_control_array[conn->dev_index] > 0);
	if (--pts->rjoin_control_array[conn->dev_index] == 0)
	{
		bool	final_all_devices = true;

		kfin->final_this_device = true;
		for (int i=0; pts->rjoin_control_array[i] >= 0; i++)
		{
			if (pts->rjoin_control_array[i] > 0)
			{
				final_all_devices = false;
				break;
			}
		}
		kfin->final_all_devices = final_all_devices;
	}
	SpinLockRelease(pts->rjoin_control_lock);

	return (kfin->final_plan_node   |
			kfin->final_this_device |
			kfin->final_all_devices);
}

/*
 * pgstromTaskStateResetScan
 */
static void
pgstromTaskStateResetScan(pgstromTaskState *pts)
{
	pgstromSharedState *ps_state = pts->ps_state;

	pg_atomic_write_u32(&ps_state->scan_task_control, 0);
	pts->rjoin_control_lock = &ps_state->__rjoin_control_lock;
	pts->rjoin_control_array = (int *)
		((char *)ps_state + MAXALIGN(offsetof(pgstromSharedState,
											  inners[pts->num_rels])));
	for (int i=0; pts->rjoin_control_array[i] >= 0; i++)
		pts->rjoin_control_array[i] = 0;
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
	struct iovec	xcmd_iov[10];
	int				xcmd_iovcnt;
	int				ev;

	pthreadMutexLock(&conn->mutex);
	for (;;)
	{
		ResetLatch(MyLatch);

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
			if (!pts->final_done)
			{
				kern_final_task	kfin;

				pts->final_done = true;
				if (try_final_callback &&
					pgstromTaskStateEndScan(pts, &kfin) &&
					pts->cb_final_chunk != NULL)
				{
					xcmd = pts->cb_final_chunk(pts, &kfin, xcmd_iov, &xcmd_iovcnt);
					if (xcmd)
					{
						xpuClientSendCommandIOV(conn, xcmd_iov, xcmd_iovcnt);
						continue;
					}
				}
			}
			return NULL;
		}
		CHECK_FOR_INTERRUPTS();

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
	struct iovec	xcmd_iov[10];
	int				xcmd_iovcnt;
	int				ev;

	while (!pts->scan_done)
	{
		CHECK_FOR_INTERRUPTS();

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
			xcmd = pts->cb_next_chunk(pts, xcmd_iov, &xcmd_iovcnt);
			if (!xcmd)
			{
				Assert(pts->scan_done);
				break;
			}
			xpuClientSendCommandIOV(conn, xcmd_iov, xcmd_iovcnt);
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

static XpuCommand *
pgstromScanChunkGpuCache(pgstromTaskState *pts,
						 struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	elog(ERROR, "not implemented yet");
	return NULL;
}

/*
 * pgstromScanNextTuple
 */
static TupleTableSlot *
pgstromScanNextTuple(pgstromTaskState *pts)
{
	TupleTableSlot *slot = pts->css.ss.ss_ScanTupleSlot;

	for (;;)
	{
		kern_data_store *kds = pts->curr_kds;
		int64_t		index = pts->curr_index++;

		if (index < kds->nitems)
		{
			kern_tupitem   *tupitem = KDS_GET_TUPITEM(kds, index);

			pts->curr_htup.t_len = tupitem->t_len;
			pts->curr_htup.t_data = &tupitem->htup;
			return ExecStoreHeapTuple(&pts->curr_htup, slot, false);
		}
		if (++pts->curr_chunk < pts->curr_resp->u.results.chunks_nitems)
		{
			pts->curr_kds = (kern_data_store *)((char *)kds + kds->length);
			pts->curr_index = 0;
			continue;
		}
		return NULL;
	}
}

/*
 * pgstromExecFinalChunk
 */
static XpuCommand *
pgstromExecFinalChunk(pgstromTaskState *pts,
					  kern_final_task *kfin,
					  struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	XpuCommand	   *xcmd;

	pts->xcmd_buf.len = offsetof(XpuCommand, u.fin.data);
	enlargeStringInfo(&pts->xcmd_buf, 0);

	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	memset(xcmd, 0, sizeof(XpuCommand));
	xcmd->magic  = XpuCommandMagicNumber;
	xcmd->tag    = XpuCommandTag__XpuTaskFinal;
	xcmd->length = offsetof(XpuCommand, u.fin.data);
	memcpy(&xcmd->u.fin, kfin, sizeof(kern_final_task));

	xcmd_iov[0].iov_base = xcmd;
	xcmd_iov[0].iov_len  = offsetof(XpuCommand, u.fin.data);
	*xcmd_iovcnt = 1;

	return xcmd;
}

/*
 * fixup_inner_varnode
 *
 * Any var-nodes are rewritten at setrefs.c to indicate a particular item
 * on the cscan->custom_scan_tlist. However, inner expression must reference
 * the inner relation, so we need to fix up it again.
 */
typedef struct
{
	CustomScan *cscan;
	Plan	   *inner_plan;
} fixup_inner_varnode_context;

static Node *
__fixup_inner_varnode_walker(Node *node, void *data)
{
	fixup_inner_varnode_context *con = data;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var		   *var = (Var *)node;
		List	   *tlist_dev = con->cscan->custom_scan_tlist;
		TargetEntry *tle;

		Assert(var->varno == INDEX_VAR &&
			   var->varattno >= 1 &&
			   var->varattno <= list_length(tlist_dev));
		tle = list_nth(tlist_dev, var->varattno - 1);
		return (Node *)makeVar(INNER_VAR,
							   tle->resorigcol,
							   var->vartype,
							   var->vartypmod,
							   var->varcollid,
							   0);
	}
	return expression_tree_mutator(node, __fixup_inner_varnode_walker, con);
}

static Node *
fixup_inner_varnode(Node *expr, CustomScan *cscan, Plan *inner_plan)
{
	fixup_inner_varnode_context con;

	memset(&con, 0, sizeof(con));
	con.cscan = cscan;
	con.inner_plan = inner_plan;

	return __fixup_inner_varnode_walker(expr, &con);
}

/*
 * fixup_fallback_varnode
 */
typedef struct
{
	CustomScan *cscan;
	List	   *kvars_slot_list;
} fixup_fallback_varnode_context;

static Node *
__fixup_fallback_varnode_walker(Node *node, void *data)
{
	fixup_fallback_varnode_context *con = data;
	ListCell   *lc;
	int			slot_id = 0;

	if (!node)
		return NULL;
	foreach (lc, con->kvars_slot_list)
	{
		Node   *curr = lfirst(lc);

		if (equal(node, curr))
		{
			return (Node *)makeVar(INDEX_VAR,
								   slot_id+1,
								   exprType(node),
								   exprTypmod(node),
								   exprCollation(node), 0);
		}
		slot_id++;
	}
	if (IsA(node, Var))
	{
		List   *custom_scan_tlist = con->cscan->custom_scan_tlist;
		Var	   *var = (Var *)node;

		if (var->varno == INDEX_VAR &&
			var->varattno > 0 &&
			var->varattno <= list_length(custom_scan_tlist))
		{
			TargetEntry *tle = list_nth(custom_scan_tlist, var->varattno-1);

			return __fixup_fallback_varnode_walker((Node *)tle->expr, data);
		}
		elog(ERROR, "Bug? kvars_slot_list lookup failed: (%s)", nodeToString(node));
	}
	return expression_tree_mutator(node, __fixup_fallback_varnode_walker, data);
}

static Node *
fixup_fallback_varnode(Node *node, CustomScan *cscan, List *kvars_slot_list)
{
	fixup_fallback_varnode_context con;

	memset(&con, 0, sizeof(con));
	con.cscan = cscan;
	con.kvars_slot_list = kvars_slot_list;

	return __fixup_fallback_varnode_walker(node, &con);
}

/*
 * __setupTaskStateRequestBuffer
 */
static void
__setupTaskStateRequestBuffer(pgstromTaskState *pts,
							  TupleDesc tdesc_src,
							  TupleDesc tdesc_dst,
							  char format)
{
	XpuCommand	   *xcmd;
	kern_data_store *kds;
	size_t			bufsz;
	size_t			off;

	initStringInfo(&pts->xcmd_buf);
	bufsz = MAXALIGN(offsetof(XpuCommand, u.task.data));
	if (tdesc_src)
		bufsz += estimate_kern_data_store(tdesc_src);
	if (tdesc_dst)
		bufsz += estimate_kern_data_store(tdesc_dst);
	enlargeStringInfo(&pts->xcmd_buf, bufsz);

	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	memset(xcmd, 0, offsetof(XpuCommand, u.task.data));
	xcmd->magic = XpuCommandMagicNumber;
	xcmd->tag   = XpuCommandTag__XpuTaskExec;
	xcmd->length = bufsz;

	off = offsetof(XpuCommand, u.task.data);
	if (tdesc_dst)
	{
		xcmd->u.task.kds_dst_offset = off;
		kds  = (kern_data_store *)((char *)xcmd + off);
		off += setup_kern_data_store(kds, tdesc_dst, 0, KDS_FORMAT_ROW);
	}
	if (tdesc_src)
	{
		xcmd->u.task.kds_src_offset = off;
		kds  = (kern_data_store *)((char *)xcmd + off);
		off += setup_kern_data_store(kds, tdesc_src, 0, format);
	}
	pts->xcmd_buf.len = off;
}

/*
 * __execInitTaskStateCpuFallback
 */
static List *
__execInitTaskStateCpuFallback(pgstromTaskState *pts)
{
	CustomScan *cscan = (CustomScan *)pts->css.ss.ps.plan;
	pgstromPlanInfo	*pp_info = pts->pp_info;
	Relation	rel = pts->css.ss.ss_currentRelation;
	TupleDesc	fallback_tdesc = RelationGetDescr(rel);
	List	   *fallback_tlist = NIL;
	List	   *kvars_slot_list = NIL;
	List	   *base_quals;
	ListCell   *cell;

	/*
	 * Setup CPU fallback infrastructure for base relation scan (depth-0)
	 */
	base_quals = (List *)
		fixup_varnode_to_origin((Node *)pp_info->scan_quals,
								cscan->custom_scan_tlist);
	pts->base_quals = ExecInitQual(base_quals, &pts->css.ss.ps);
	foreach (cell, cscan->custom_scan_tlist)
	{
		TargetEntry *tle = lfirst(cell);

		if (!tle->resjunk)
			fallback_tlist = lappend(fallback_tlist, tle);
	}
	pts->base_slot = MakeSingleTupleTableSlot(RelationGetDescr(rel),
											  table_slot_callbacks(rel));
	if (pts->num_rels > 0)
	{
		TupleDesc	relation_tdesc = RelationGetDescr(rel);
		List	   *fallback_tlist_fixed = NIL;
		int			nslots = list_length(pp_info->kvars_depth);
		int			slot_id = 0;
		ListCell   *lc1, *lc2;
		
		fallback_tdesc = CreateTemplateTupleDesc(nslots);
		forboth (lc1, pp_info->kvars_depth,
				 lc2, pp_info->kvars_resno)
		{
			int		__depth = lfirst_int(lc1);
			int		__resno = lfirst_int(lc2);

			if (__depth < 0)
			{
				Const  *con = makeNullConst(INT4OID, -1, InvalidOid);

				TupleDescInitEntry(fallback_tdesc,
								   slot_id+1,
								   "...null.value...",
								   INT4OID,
								   -1,
								   InvalidOid);
				kvars_slot_list = lappend(kvars_slot_list, con);
			}
			else if (__depth == 0)
			{
				Form_pg_attribute attr = TupleDescAttr(relation_tdesc, __resno-1);
				Var	   *var;

				Assert(__resno > 0 && __resno <= relation_tdesc->natts);
				TupleDescInitEntry(fallback_tdesc,
								   slot_id+1,
								   NameStr(attr->attname),
								   attr->atttypid,
								   attr->atttypmod,
								   attr->attndims);
				var = makeVar(cscan->scan.scanrelid,
							  attr->attnum,
							  attr->atttypid,
							  attr->atttypmod,
							  attr->attcollation, 0);
				kvars_slot_list = lappend(kvars_slot_list, var);
			}
			else
			{
				Plan	   *i_plan = list_nth(cscan->custom_plans, __depth-1);
				TargetEntry *tle = list_nth(i_plan->targetlist, __resno-1);
				Oid			type_oid;
				int32		type_mod;

				Assert(__depth > 0 && __depth <= pts->num_rels);
				Assert(__resno > 0 && __resno <= list_length(i_plan->targetlist) &&
					   __resno == tle->resno);
				type_oid = exprType((Node *)tle->expr);
				type_mod = exprTypmod((Node *)tle->expr);
				TupleDescInitEntry(fallback_tdesc,
								   slot_id+1,
								   tle->resname,
								   type_oid,
								   type_mod,
								   type_is_array(type_oid) ? 1 : 0);
				kvars_slot_list = lappend(kvars_slot_list, tle->expr);
			}
			slot_id++;
		}
		pts->fallback_slot = MakeSingleTupleTableSlot(fallback_tdesc,
													  &TTSOpsVirtual);
		foreach (lc1, fallback_tlist)
		{
			TargetEntry *tle_old = lfirst(lc1);
			TargetEntry *tle_new = flatCopyTargetEntry(tle_old);

			tle_new->expr = (Expr *)
				fixup_fallback_varnode((Node *)tle_old->expr,
									   cscan, kvars_slot_list);
			fallback_tlist_fixed = lappend(fallback_tlist_fixed, tle_new);
		}
		fallback_tlist = fallback_tlist_fixed;
	}
	/*
	 * Setup CPU fallback Projection
	 */
	pts->fallback_proj =
		ExecBuildProjectionInfo(fallback_tlist,
								pts->css.ss.ps.ps_ExprContext,
								pts->css.ss.ss_ScanTupleSlot,
								&pts->css.ss.ps,
								fallback_tdesc);
	return kvars_slot_list;
}

/*
 * pgstromExecInitTaskState
 */
void
pgstromExecInitTaskState(CustomScanState *node, EState *estate, int eflags)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;
	pgstromPlanInfo	*pp_info = pts->pp_info;
	CustomScan *cscan = (CustomScan *)pts->css.ss.ps.plan;
	Relation	rel = pts->css.ss.ss_currentRelation;
	TupleDesc	tupdesc_src = RelationGetDescr(rel);
	TupleDesc	tupdesc_dst;
	int			depth_index = 0;
	bool		has_right_outer = false;
	ListCell   *lc;
	List	   *kvars_slot_list;

	/* sanity checks */
	Assert(rel != NULL &&
		   outerPlanState(node) == NULL &&
		   innerPlanState(node) == NULL &&
		   pp_info->num_rels == list_length(cscan->custom_plans) &&
		   pts->num_rels == list_length(cscan->custom_plans));
	/*
	 * PG-Strom supports:
	 * - regular relation with 'heap' access method
	 * - foreign-table with 'arrow_fdw' driver
	 */
	if (RelationGetForm(rel)->relkind == RELKIND_RELATION ||
		RelationGetForm(rel)->relkind == RELKIND_MATVIEW)
	{
		SMgrRelation smgr = RelationGetSmgr(rel);
		Oid			am_oid = RelationGetForm(rel)->relam;
		const char *kds_pathname = relpath(smgr->smgr_rnode, MAIN_FORKNUM);

		if (am_oid != HEAP_TABLE_AM_OID)
			elog(ERROR, "PG-Strom does not support table access method: %s",
				 get_am_name(am_oid));

		/* setup BRIN-index if any */
		pgstromBrinIndexExecBegin(pts,
								  pp_info->brin_index_oid,
								  pp_info->brin_index_conds,
								  pp_info->brin_index_quals);
		if ((pts->task_kind & DEVKIND__NVIDIA_GPU) != 0)
			pts->optimal_gpus = GetOptimalGpuForRelation(rel);
		if ((pts->task_kind & DEVKIND__NVIDIA_DPU) != 0)
			pts->ds_entry = GetOptimalDpuForRelation(rel, &kds_pathname);
		pts->kds_pathname = kds_pathname;
	}
	else if (RelationGetForm(rel)->relkind == RELKIND_FOREIGN_TABLE)
	{
		if (!pgstromArrowFdwExecInit(pts,
									 pp_info->scan_quals,
									 pp_info->outer_refs))
			elog(ERROR, "Bug? only arrow_fdw is supported in PG-Strom");
	}
	else
	{
		elog(ERROR, "Bug? PG-Strom does not support relation type of '%s'",
			 RelationGetRelationName(rel));
	}

	/*
	 * Re-initialization of scan tuple-descriptor and projection-info,
	 * because commit 1a8a4e5cde2b7755e11bde2ea7897bd650622d3e of
	 * PostgreSQL makes to assign result of ExecTypeFromTL() instead
	 * of ExecCleanTypeFromTL; that leads incorrect projection.
	 * So, we try to remove junk attributes from the scan-descriptor.
	 *
	 * And, device projection returns a tuple in heap-format, so we
	 * prefer TTSOpsHeapTuple, instead of the TTSOpsVirtual.
	 */
	tupdesc_dst = ExecCleanTypeFromTL(cscan->custom_scan_tlist);
	ExecInitScanTupleSlot(estate, &pts->css.ss, tupdesc_dst,
						  &TTSOpsHeapTuple);
	ExecAssignScanProjectionInfoWithVarno(&pts->css.ss, INDEX_VAR);

	/*
	 * Init resources for CPU fallbacks
	 */
	kvars_slot_list = __execInitTaskStateCpuFallback(pts);

	/*
	 * init inner relations
	 */
	foreach (lc, cscan->custom_plans)
	{
		pgstromTaskInnerState *istate = &pts->inners[depth_index];
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[depth_index];
		Plan	   *plan = lfirst(lc);
		PlanState  *ps = ExecInitNode(plan, estate, eflags);
		List	   *join_quals;
		List	   *other_quals;
		ListCell   *cell;
		ExprState  *es;
		devtype_info *dtype;

		istate->ps = ps;
		istate->econtext = CreateExprContext(estate);
		istate->depth = depth_index + 1;
		istate->join_type = pp_inner->join_type;
		join_quals = (List *)
			fixup_fallback_varnode((Node *)pp_inner->join_quals,
								   cscan, kvars_slot_list);
		istate->join_quals = ExecInitQual(join_quals, &pts->css.ss.ps);
		other_quals = (List *)
			fixup_fallback_varnode((Node *)pp_inner->other_quals,
								   cscan, kvars_slot_list);
		istate->other_quals = ExecInitQual(other_quals, &pts->css.ss.ps);
		if (pp_inner->join_type == JOIN_FULL ||
			pp_inner->join_type == JOIN_RIGHT)
			has_right_outer = true;

		foreach (cell, pp_inner->hash_outer_keys)
		{
			Node	   *outer_key = (Node *)lfirst(cell);

			outer_key = fixup_fallback_varnode(outer_key, cscan, kvars_slot_list);
			es = ExecInitExpr((Expr *)outer_key, &pts->css.ss.ps);
			dtype = pgstrom_devtype_lookup(exprType((Node *)es->expr));
			if (!dtype)
				elog(ERROR, "failed on lookup device type of %s",
					 nodeToString(es->expr));
			istate->hash_outer_keys = lappend(istate->hash_outer_keys, es);
			istate->hash_outer_dtypes = lappend(istate->hash_outer_dtypes, dtype);
		}
		/* inner hash-keys references the result of inner-slot */
		foreach (cell, pp_inner->hash_inner_keys)
		{
			Node	   *inner_key = (Node *)lfirst(cell);

			inner_key = fixup_inner_varnode(inner_key, cscan, plan);
			es = ExecInitExpr((Expr *)inner_key, &pts->css.ss.ps);
			dtype = pgstrom_devtype_lookup(exprType((Node *)es->expr));
			if (!dtype)
				elog(ERROR, "failed on lookup device type of %s",
					 nodeToString(es->expr));
			istate->hash_inner_keys = lappend(istate->hash_inner_keys, es);
			istate->hash_inner_dtypes = lappend(istate->hash_inner_dtypes, dtype);
		}

		if (OidIsValid(pp_inner->gist_index_oid))
		{
			istate->gist_irel = index_open(pp_inner->gist_index_oid,
										   AccessShareLock);
			istate->gist_clause = ExecInitExpr((Expr *)pp_inner->gist_clause,
											   &pts->css.ss.ps);
		}
		pts->css.custom_ps = lappend(pts->css.custom_ps, ps);
		depth_index++;
	}
	Assert(depth_index == pts->num_rels);
	
	/*
	 * Setup request buffer
	 */
	if (pts->arrow_state)		/* Apache Arrow */
	{
		pts->cb_next_chunk = pgstromScanChunkArrowFdw;
		pts->cb_next_tuple = pgstromScanNextTuple;
	    __setupTaskStateRequestBuffer(pts,
									  NULL,
									  tupdesc_dst,
									  KDS_FORMAT_ARROW);
	}
	else if (pts->gcache_state)		/* GPU-Cache */
	{
		pts->cb_next_chunk = pgstromScanChunkGpuCache;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  NULL,
									  tupdesc_dst,
									  KDS_FORMAT_COLUMN);
	}
	else if (!bms_is_empty(pts->optimal_gpus) ||	/* GPU-Direct SQL */
			 pts->ds_entry)							/* DPU Storage */
	{
		pts->cb_next_chunk = pgstromRelScanChunkDirect;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  tupdesc_src,
									  tupdesc_dst,
									  KDS_FORMAT_BLOCK);
	}
	else						/* Slow normal heap storage */
	{
		pts->cb_next_chunk = pgstromRelScanChunkNormal;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  tupdesc_src,
									  tupdesc_dst,
									  KDS_FORMAT_ROW);
	}

	/*
	 * workload specific callback routines
	 */
	if ((pts->task_kind & DEVTASK__SCAN) != 0)
	{
		pts->cb_cpu_fallback = ExecFallbackCpuScan;
	}
	else if ((pts->task_kind & DEVTASK__JOIN) != 0)
	{
		if (has_right_outer)
			pts->cb_final_chunk = pgstromExecFinalChunk;
		pts->cb_cpu_fallback = ExecFallbackCpuJoin;
	}
	else if ((pts->task_kind & DEVTASK__PREAGG) != 0)
	{
		pts->cb_final_chunk = pgstromExecFinalChunk;
		pts->cb_cpu_fallback = ExecFallbackCpuPreAgg;
	}
	else
		elog(ERROR, "Bug? unknown DEVTASK");
	/* other fields init */
	pts->curr_vm_buffer = InvalidBuffer;
}

/*
 * pgstromExecScanAccess
 */
static TupleTableSlot *
pgstromExecScanAccess(pgstromTaskState *pts)
{
	TupleTableSlot *slot;
	XpuCommand	   *resp;

	slot = pgstromFetchFallbackTuple(pts);
	if (slot)
		return slot;
	
	while (!pts->curr_resp || !(slot = pts->cb_next_tuple(pts)))
	{
	next_chunks:
		if (pts->curr_resp)
			xpuClientPutResponse(pts->curr_resp);
		pts->curr_resp = __fetchNextXpuCommand(pts);
		if (!pts->curr_resp)
			return pgstromFetchFallbackTuple(pts);

		resp = pts->curr_resp;
		if (resp->tag == XpuCommandTag__Success ||
			resp->tag == XpuCommandTag__SuccessAndRightOuter)
		{
			if (resp->u.results.chunks_nitems == 0)
				goto next_chunks;
			pts->curr_kds = (kern_data_store *)
				((char *)resp + resp->u.results.chunks_offset);
			pts->curr_chunk = 0;
			pts->curr_index = 0;
			if (resp->tag == XpuCommandTag__SuccessAndRightOuter)
				elog(ERROR, "RIGHT OUTER is not ready");
		}
		else
		{
			Assert(resp->tag == XpuCommandTag__CPUFallback);
			//run CPU fallback
			//attach alternative KDS
			elog(ERROR, "CPU fallback is not ready");
		}
	}
	return slot;
}

/*
 * pgstromExecScanReCheck
 */
static bool
pgstromExecScanReCheck(pgstromTaskState *pts, TupleTableSlot *epq_slot)
{
	/*
	 * NOTE: Only immutable operators/functions are executable
	 * on the GPU devices, so its decision will never changed.
	 */
	return true;
}

/*
 * __pgstromExecTaskOpenConnection
 */
static bool
__pgstromExecTaskOpenConnection(pgstromTaskState *pts)
{
	const XpuCommand *session;
	uint32_t	inner_handle = 0;
	TupleDesc	tupdesc_kds_final = NULL;

	/* attach pgstromSharedState, if none */
	if (!pts->ps_state)
		pgstromSharedStateInitDSM(&pts->css, NULL, NULL);
	/* preload inner buffer, if any */
	if (pts->num_rels > 0)
	{
		inner_handle = GpuJoinInnerPreload(pts);
		if (inner_handle == 0)
			return false;
	}
	/* XPU-PreAgg needs tupdesc of kds_final */
	if ((pts->task_kind & DEVTASK__PREAGG) != 0)
	{
		tupdesc_kds_final = pts->css.ss.ps.scandesc;
	}
	/* build the session information */
	session = pgstromBuildSessionInfo(pts, inner_handle, tupdesc_kds_final);

	if ((pts->task_kind & DEVKIND__NVIDIA_GPU) != 0)
	{
		gpuClientOpenSession(pts, session);
	}
	else if ((pts->task_kind & DEVKIND__NVIDIA_DPU) != 0)
	{
		Assert(pts->ds_entry != NULL);
		DpuClientOpenSession(pts, session);
	}
	else
	{
		elog(ERROR, "Bug? unknown PG-Strom task kind: %08x", pts->task_kind);
	}
	/* update the scan/join control variables */
	if (!pgstromTaskStateBeginScan(pts))
		return false;
	
	return true;
}

/*
 * pgstromExecTaskState
 */
TupleTableSlot *
pgstromExecTaskState(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;

	if (!pts->conn)
	{
		if (!__pgstromExecTaskOpenConnection(pts))
			return NULL;
		Assert(pts->conn);
	}
	return ExecScan(&pts->css.ss,
					(ExecScanAccessMtd) pgstromExecScanAccess,
					(ExecScanRecheckMtd) pgstromExecScanReCheck);
}

/*
 * pgstromExecEndTaskState
 */
void
pgstromExecEndTaskState(CustomScanState *node)
{
	pgstromTaskState   *pts = (pgstromTaskState *)node;
	pgstromSharedState *ps_state = pts->ps_state;
	ListCell   *lc;

	if (pts->curr_vm_buffer != InvalidBuffer)
		ReleaseBuffer(pts->curr_vm_buffer);
	if (pts->conn)
		xpuClientCloseSession(pts->conn);
	if (pts->br_state)
		pgstromBrinIndexExecEnd(pts);
	if (pts->arrow_state)
		pgstromArrowFdwExecEnd(pts->arrow_state);
	if (pts->base_slot)
		ExecDropSingleTupleTableSlot(pts->base_slot);
	if (pts->css.ss.ss_currentScanDesc)
		table_endscan(pts->css.ss.ss_currentScanDesc);
	if (pts->h_kmrels)
		__munmapShmem(pts->h_kmrels);
	if (!IsParallelWorker())
	{
		if (ps_state && ps_state->preload_shmem_handle != 0)
			__shmemDrop(ps_state->preload_shmem_handle);
	}
	foreach (lc, pts->css.custom_ps)
		ExecEndNode((PlanState *) lfirst(lc));
}

/*
 * pgstromExecResetTaskState
 */
void
pgstromExecResetTaskState(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	if (pts->conn)
	{
		xpuClientCloseSession(pts->conn);
		pts->conn = NULL;
	}
	pgstromTaskStateResetScan(pts);
	if (pts->br_state)
		pgstromBrinIndexExecReset(pts);
	if (pts->arrow_state)
		pgstromArrowFdwExecReset(pts->arrow_state);
}

/*
 * pgstromSharedStateEstimateDSM
 */
Size
pgstromSharedStateEstimateDSM(CustomScanState *node,
							  ParallelContext *pcxt)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;
	Relation	relation = node->ss.ss_currentRelation;
	EState	   *estate   = node->ss.ps.state;
	Snapshot	snapshot = estate->es_snapshot;
	int			num_rels = list_length(node->custom_ps);
	int			num_devs = 0;
	Size		len = 0;

	if (pts->br_state)
		len += pgstromBrinIndexEstimateDSM(pts);
	len += MAXALIGN(offsetof(pgstromSharedState, inners[num_rels]));

	if ((pts->task_kind & DEVKIND__NVIDIA_GPU) != 0)
		num_devs = numGpuDevAttrs;
	else if ((pts->task_kind & DEVKIND__NVIDIA_DPU) != 0)
		num_devs = DpuStorageEntryCount();
	len += MAXALIGN(sizeof(int) * (num_devs+1));

	if (!pts->arrow_state)
		len += table_parallelscan_estimate(relation, snapshot);
	return MAXALIGN(len);
}

/*
 * pgstromSharedStateInitDSM
 */
void
pgstromSharedStateInitDSM(CustomScanState *node,
						  ParallelContext *pcxt,
						  void *coordinate)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;
	Relation	relation = node->ss.ss_currentRelation;
	EState	   *estate   = node->ss.ps.state;
	Snapshot	snapshot = estate->es_snapshot;
	int			num_rels = list_length(node->custom_ps);
	int			num_devs = 0;
	size_t		dsm_length = offsetof(pgstromSharedState, inners[num_rels]);
	char	   *dsm_addr = coordinate;
	pgstromSharedState *ps_state;
	TableScanDesc scan = NULL;

	Assert(!IsBackgroundWorker);
	if ((pts->task_kind & DEVKIND__NVIDIA_GPU) != 0)
		num_devs = numGpuDevAttrs;
	else if ((pts->task_kind & DEVKIND__NVIDIA_DPU) != 0)
		num_devs = DpuStorageEntryCount();

	if (pts->br_state)
		dsm_addr += pgstromBrinIndexInitDSM(pts, dsm_addr);
	Assert(!pts->css.ss.ss_currentScanDesc);
	if (dsm_addr)
	{
		ps_state = (pgstromSharedState *) dsm_addr;
		memset(ps_state, 0, dsm_length);
		ps_state->ss_handle = dsm_segment_handle(pcxt->seg);
		ps_state->ss_length = dsm_length;
		dsm_addr += MAXALIGN(dsm_length);

		/* control variables for scan/rjoin */
		pg_atomic_init_u32(&ps_state->scan_task_control, 0);
		SpinLockInit(&ps_state->__rjoin_control_lock);
		pts->rjoin_control_lock = &ps_state->__rjoin_control_lock;
		pts->rjoin_control_array = (int *)dsm_addr;
		memset(pts->rjoin_control_array, 0, sizeof(int) * (num_devs+1));
		pts->rjoin_control_array[num_devs] = -1;	/* terminator */
		dsm_addr += MAXALIGN(sizeof(int) * (num_devs+1));

		/* parallel scan descriptor */
		if (pts->arrow_state)
			pgstromArrowFdwInitDSM(pts->arrow_state, ps_state);
		else
		{
			ParallelTableScanDesc pdesc = (ParallelTableScanDesc) dsm_addr;

			table_parallelscan_initialize(relation, pdesc, snapshot);
			scan = table_beginscan_parallel(relation, pdesc);
		}
	}
	else
	{
		ps_state = MemoryContextAllocZero(estate->es_query_cxt, dsm_length);
		ps_state->ss_handle = DSM_HANDLE_INVALID;
		ps_state->ss_length = dsm_length;

		/* control variables for scan/rjoin */
		pg_atomic_init_u32(&ps_state->scan_task_control, 0);
		SpinLockInit(&ps_state->__rjoin_control_lock);
		pts->rjoin_control_lock = &ps_state->__rjoin_control_lock;
		pts->rjoin_control_array = (int *)
			MemoryContextAllocZero(estate->es_query_cxt,
								   sizeof(int) * (num_devs+1));
		pts->rjoin_control_array[num_devs] = -1;	/* terminator */

		/* scan descriptor */
		if (pts->arrow_state)
			pgstromArrowFdwInitDSM(pts->arrow_state, ps_state);
		else
			scan = table_beginscan(relation, estate->es_snapshot, 0, NULL);
	}
	ps_state->query_plan_id = ((uint64_t)MyProcPid) << 32 |
		(uint64_t)pts->css.ss.ps.plan->plan_node_id;	
	ps_state->num_rels = num_rels;
	ConditionVariableInit(&ps_state->preload_cond);
	SpinLockInit(&ps_state->preload_mutex);
	if (num_rels > 0)
		ps_state->preload_shmem_handle = __shmemCreate(pts->ds_entry);
	pts->ps_state = ps_state;
	pts->css.ss.ss_currentScanDesc = scan;
}

/*
 * pgstromSharedStateAttachDSM
 */
void
pgstromSharedStateAttachDSM(CustomScanState *node,
							shm_toc *toc,
							void *coordinate)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;
	char	   *dsm_addr = coordinate;
	int			num_rels = list_length(pts->css.custom_ps);
	int			num_devs = 0;

	if ((pts->task_kind & DEVKIND__NVIDIA_GPU) != 0)
		num_devs = numGpuDevAttrs;
	else if ((pts->task_kind & DEVKIND__NVIDIA_DPU) != 0)
		num_devs = DpuStorageEntryCount();

	if (pts->br_state)
		dsm_addr += pgstromBrinIndexAttachDSM(pts, dsm_addr);
	pts->ps_state = (pgstromSharedState *)dsm_addr;
	Assert(pts->ps_state->num_rels == num_rels);
	dsm_addr += MAXALIGN(offsetof(pgstromSharedState, inners[num_rels]));

	pts->rjoin_control_lock = &pts->ps_state->__rjoin_control_lock;
	pts->rjoin_control_array = (int *)dsm_addr;
	dsm_addr += MAXALIGN(sizeof(int) * (num_devs+1));

	if (pts->arrow_state)
		pgstromArrowFdwAttachDSM(pts->arrow_state, pts->ps_state);
	else
	{
		Relation	relation = pts->css.ss.ss_currentRelation;
		ParallelTableScanDesc pdesc = (ParallelTableScanDesc) dsm_addr;

		pts->css.ss.ss_currentScanDesc = table_beginscan_parallel(relation, pdesc);
	}
}

/*
 * pgstromSharedStateShutdownDSM
 */
void
pgstromSharedStateShutdownDSM(CustomScanState *node)
{
	pgstromTaskState   *pts = (pgstromTaskState *) node;
	pgstromSharedState *src_state = pts->ps_state;
	pgstromSharedState *dst_state;
	EState			   *estate = node->ss.ps.state;

	if (pts->br_state)
		pgstromBrinIndexShutdownDSM(pts);
	if (pts->arrow_state)
		pgstromArrowFdwShutdown(pts->arrow_state);
	if (src_state)
	{
		size_t	sz = offsetof(pgstromSharedState,
							  inners[src_state->num_rels]);
		dst_state = MemoryContextAllocZero(estate->es_query_cxt, sz);
		memcpy(dst_state, src_state, sz);
		pts->ps_state = dst_state;
	}
}

/*
 * pgstromExplainTaskState
 */
void
pgstromExplainTaskState(CustomScanState *node,
						List *ancestors,
						ExplainState *es)
{
	pgstromTaskState   *pts = (pgstromTaskState *)node;
	pgstromPlanInfo	   *pp_info = pts->pp_info;
	CustomScan		   *cscan = (CustomScan *)node->ss.ps.plan;
	List			   *dcontext;
	StringInfoData		buf;
	ListCell		   *lc;
	const char		   *xpu_label;
	char				label[100];
	char			   *str;
	double				ntuples;

	/* setup deparse context */
	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										ancestors);
	if ((pts->task_kind & DEVKIND__NVIDIA_GPU) != 0)
		xpu_label = "GPU";
	else if ((pts->task_kind & DEVKIND__NVIDIA_DPU) != 0)
		xpu_label = "DPU";
	else
		xpu_label = "???";

	/* xPU Projection */
	initStringInfo(&buf);
	foreach (lc, cscan->custom_scan_tlist)
	{
		TargetEntry *tle = lfirst(lc);

		if (tle->resjunk)
			continue;
		str = deparse_expression((Node *)tle->expr, dcontext, false, true);
		if (buf.len > 0)
			appendStringInfoString(&buf, ", ");
		appendStringInfoString(&buf, str);
	}
	snprintf(label, sizeof(label),
			 "%s Projection", xpu_label);
	ExplainPropertyText(label, buf.data, es);

	/* xPU Scan Quals */
	if (pp_info->scan_quals)
	{
		List   *scan_quals = pp_info->scan_quals;
		Expr   *expr;

		resetStringInfo(&buf);
		if (list_length(scan_quals) > 1)
			expr = make_andclause(scan_quals);
		else
			expr = linitial(scan_quals);
		str = deparse_expression((Node *)expr, dcontext, false, true);
		appendStringInfo(&buf, "%s [rows: %.0f -> %.0f]",
						 str,
						 pp_info->scan_tuples,
						 pp_info->scan_rows);
		snprintf(label, sizeof(label), "%s Scan Quals", xpu_label);
		ExplainPropertyText(label, buf.data, es);
	}

	/* xPU JOIN */
	ntuples = pp_info->scan_rows;
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];

		if (pp_inner->join_quals != NIL || pp_inner->other_quals != NIL)
		{
			const char *join_label;

			resetStringInfo(&buf);
			foreach (lc, pp_inner->join_quals)
			{
				Node   *expr = lfirst(lc);

				str = deparse_expression(expr, dcontext, false, true);
				if (buf.len > 0)
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, str);
			}
			if (pp_inner->other_quals != NIL)
			{
				foreach (lc, pp_inner->other_quals)
				{
					Node   *expr = lfirst(lc);

					str = deparse_expression(expr, dcontext, false, true);
					if (buf.len > 0)
						appendStringInfoString(&buf, ", ");
					appendStringInfo(&buf, "[%s]", str);
				}
			}
			appendStringInfo(&buf, " ... [nrows: %.0f -> %.0f]",
							 ntuples, pp_inner->join_nrows);
			switch (pp_inner->join_type)
			{
				case JOIN_INNER: join_label = "Join"; break;
				case JOIN_LEFT:  join_label = "Left Outer Join"; break;
				case JOIN_RIGHT: join_label = "Right Outer Join"; break;
				case JOIN_FULL:  join_label = "Full Outer Join"; break;
				case JOIN_SEMI:  join_label = "Semi Join"; break;
				case JOIN_ANTI:  join_label = "Anti Join"; break;
				default:         join_label = "??? Join"; break;
			}
			snprintf(label, sizeof(label),
					 "%s %s Quals [%d]", xpu_label, join_label, i+1);
			ExplainPropertyText(label, buf.data, es);
		}
		ntuples = pp_inner->join_nrows;

		if (pp_inner->hash_outer_keys != NIL)
		{
			resetStringInfo(&buf);
			foreach (lc, pp_inner->hash_outer_keys)
			{
				Node   *expr = lfirst(lc);

				str = deparse_expression(expr, dcontext, true, true);
				if (buf.len > 0)
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, str);
			}
			snprintf(label, sizeof(label),
					 "%s Outer Hash [%d]", xpu_label, i+1);
			ExplainPropertyText(label, buf.data, es);
		}
		if (pp_inner->hash_inner_keys != NIL)
		{
			resetStringInfo(&buf);
			foreach (lc, pp_inner->hash_inner_keys)
			{
				Node   *expr = lfirst(lc);

				str = deparse_expression(expr, dcontext, true, true);
				if (buf.len > 0)
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, str);
			}
			snprintf(label, sizeof(label),
					 "%s Inner Hash [%d]", xpu_label, i+1);
			ExplainPropertyText(label, buf.data, es);
		}
		if (pp_inner->gist_clause)
		{
			char   *idxname = get_rel_name(pp_inner->gist_index_oid);
			char   *colname = get_attname(pp_inner->gist_index_oid,
										  pp_inner->gist_index_col, false);
			resetStringInfo(&buf);

			str = deparse_expression(pp_inner->gist_clause,
									 dcontext, false, true);
			appendStringInfoString(&buf, str);
			if (idxname && colname)
				appendStringInfo(&buf, " on %s (%s)", idxname, colname);
			snprintf(label, sizeof(label),
					 "%s GiST Join [%d]", xpu_label, i+1);
			ExplainPropertyText(label, buf.data, es);
		}
	}

	/*
	 * Storage related info
	 */
	if (pts->arrow_state)
	{
		pgstromArrowFdwExplain(pts->arrow_state,
							   pts->css.ss.ss_currentRelation,
							   es, dcontext);
	}
	else if (pts->gcache_state)
	{
		/* GPU-Cache */
	}
	else if (!bms_is_empty(pts->optimal_gpus))
	{
		/* GPU-Direct */
		resetStringInfo(&buf);
		if (!es->analyze)
		{
			bool	is_first = true;
			int		k;

			appendStringInfo(&buf, "enabled (");
			for (k = bms_next_member(pts->optimal_gpus, -1);
				 k >= 0;
				 k = bms_next_member(pts->optimal_gpus, k))
			{
				if (!is_first)
					appendStringInfo(&buf, ", ");
				appendStringInfo(&buf, "GPU-%d", k);
				is_first = false;
			}
			appendStringInfo(&buf, ")");
		}
		else
		{
			pgstromSharedState *ps_state = pts->ps_state;
			XpuConnection  *conn = pts->conn;
			uint32			count;

			count = pg_atomic_read_u32(&ps_state->heap_direct_nblocks);
			appendStringInfo(&buf, "enabled (%s; direct=%u", conn->devname, count);
			count = pg_atomic_read_u32(&ps_state->heap_normal_nblocks);
			if (count > 0)
				appendStringInfo(&buf, ", buffer=%u", count);
			count = pg_atomic_read_u32(&ps_state->heap_fallback_nblocks);
			if (count > 0)
				appendStringInfo(&buf, ", fallback=%u", count);
			appendStringInfo(&buf, ")");
		}
		ExplainPropertyText("GPU-Direct SQL", buf.data, es);
	}
	else if (pts->ds_entry)
	{
		/* DPU-Entry */
		explainDpuStorageEntry(pts->ds_entry, es);
	}
	else
	{
		/* Normal Heap Storage */
	}
	/* State of BRIN-index */
	if (pts->br_state)
		pgstromBrinIndexExplain(pts, dcontext, es);

	/*
	 * Dump the XPU code (only if verbose)
	 */
	if (es->verbose)
	{
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Scan VarLoads OpCode",
								pp_info->kexp_scan_kvars_load);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Scan Quals OpCode",
								pp_info->kexp_scan_quals);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Join VarLoads OpCode",
								pp_info->kexp_join_kvars_load_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Join Quals OpCode",
								pp_info->kexp_join_quals_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Join HashValue OpCode",
								pp_info->kexp_hash_keys_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"GiST-Index Join OpCode",
								pp_info->kexp_gist_quals_packed);
		snprintf(label, sizeof(label),
				 "%s Projection OpCode", xpu_label);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								label,
								pp_info->kexp_projection);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Group-By KeyHash OpCode",
								pp_info->kexp_groupby_keyhash);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Group-By KeyLoad OpCode",
								pp_info->kexp_groupby_keyload);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Group-By KeyComp OpCode",
								pp_info->kexp_groupby_keycomp);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Partial Aggregation OpCode",
								pp_info->kexp_groupby_actions);
	}
	pfree(buf.data);
}

/*
 * __xpuClientOpenSession
 */
void
__xpuClientOpenSession(pgstromTaskState *pts,
					   const XpuCommand *session,
					   pgsocket sockfd,
					   const char *devname,
					   int dev_index)
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
	conn->dev_index = dev_index;
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
 * pgstrom_init_executor
 */
void
pgstrom_init_executor(void)
{
	DefineCustomBoolVariable("pg_strom.use_debug_code",
							 "Use debug-mode enabled device code",
							 NULL,
							 &pgstrom_use_debug_code,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE | GUC_SUPERUSER_ONLY,
							 NULL, NULL, NULL);
	dlist_init(&xpu_connections_list);
	RegisterResourceReleaseCallback(xpuclientCleanupConnections, NULL);
}
