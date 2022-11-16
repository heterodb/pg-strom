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
	uint32_t		session_sz = offsetof(kern_session_info, poffset[nparams]);
	StringInfoData	buf;
	XpuCommand	   *xcmd;
	kern_session_info *session;

	initStringInfo(&buf);
	__appendZeroStringInfo(&buf, session_sz);
	session = alloca(session_sz);
	memset(session, 0, session_sz);

	/* put XPU code */
	if (xpucode_scan_quals)
	{
		session->xpucode_scan_quals =
			__appendBinaryStringInfo(&buf,
									 VARDATA_ANY(xpucode_scan_quals),
									 VARSIZE_ANY_EXHDR(xpucode_scan_quals));
	}
	if (xpucode_scan_projs)
	{
		session->xpucode_scan_projs =
			__appendBinaryStringInfo(&buf,
									 VARDATA_ANY(xpucode_scan_projs),
									 VARSIZE_ANY_EXHDR(xpucode_scan_projs));
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
	session->xpucode_use_debug_code = pgstrom_use_debug_code;
	session->xactStartTimestamp = GetCurrentTransactionStartTimestamp();
	session->session_xact_state = __build_session_xact_state(&buf);
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
			if (!try_final_callback)
				return NULL;
			if (!pts->cb_final_chunk || pts->final_done)
				return NULL;
			xcmd = pts->cb_final_chunk(pts, xcmd_iov, &xcmd_iovcnt);
			if (!xcmd)
				return NULL;
			xpuClientSendCommandIOV(conn, xcmd_iov, xcmd_iovcnt);
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
pgstromScanChunkArrowFdw(pgstromTaskState *pts,
						 struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	elog(ERROR, "not implemented yet");
	return NULL;
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

	if (pts->fallback_store)
	{
		if (tuplestore_gettupleslot(pts->fallback_store,
									true,	/* forward scan */
									false,	/* no copy */
									slot))
			return slot;
		/* no more fallback tuples */
		tuplestore_end(pts->fallback_store);
		pts->fallback_store = NULL;
	}

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
	bufsz = MAXALIGN(offsetof(XpuCommand, u.scan.data));
	if (tdesc_src)
		bufsz += estimate_kern_data_store(tdesc_src);
	if (tdesc_dst)
		bufsz += estimate_kern_data_store(tdesc_dst);
	enlargeStringInfo(&pts->xcmd_buf, bufsz);

	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	memset(xcmd, 0, offsetof(XpuCommand, u.scan.data));
	xcmd->magic  = XpuCommandMagicNumber;
	xcmd->tag    = XpuCommandTag__XpuScanExec;
	xcmd->length = bufsz;

	off = offsetof(XpuCommand, u.scan.data);
	if (tdesc_dst)
	{
		xcmd->u.scan.kds_dst_offset = off;
		kds  = (kern_data_store *)((char *)xcmd + off);
		off += setup_kern_data_store(kds, tdesc_dst, 0, KDS_FORMAT_ROW);
	}
	if (tdesc_src)
	{
		xcmd->u.scan.kds_src_offset = off;
		kds  = (kern_data_store *)((char *)xcmd + off);
		off += setup_kern_data_store(kds, tdesc_src, 0, format);
	}
	pts->xcmd_buf.len = off;
}

/*
 * pgstromExecInitTaskState
 */
void
pgstromExecInitTaskState(pgstromTaskState *pts,
						 List *outer_dev_quals)
{
	EState		   *estate = pts->css.ss.ps.state;
	CustomScan	   *cscan = (CustomScan *)pts->css.ss.ps.plan;
	Relation		rel = pts->css.ss.ss_currentRelation;
	TupleDesc		tupdesc_src = RelationGetDescr(rel);
	TupleDesc		tupdesc_dst;
	List		   *tlist_dev = NIL;
	ListCell	   *lc;
//	ArrowFdwState  *af_state = NULL;

	/*
	 * PG-Strom supports:
	 * - regular relation with 'heap' access method
	 * - foreign-table with 'arrow_fdw' driver
	 */
	if (RelationGetForm(rel)->relkind == RELKIND_RELATION ||
		RelationGetForm(rel)->relkind == RELKIND_MATVIEW)
	{
		Oid		am_oid = RelationGetForm(rel)->relam;

		if (am_oid != HEAP_TABLE_AM_OID)
			elog(ERROR, "PG-Strom does not support table access method: %s",
				 get_am_name(am_oid));
	}
	else if (RelationGetForm(rel)->relkind == RELKIND_FOREIGN_TABLE)
	{
		//call pgstromArrowFdwExecBegin here
		elog(ERROR, "Arrow support is not implemented yet");
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
	outer_dev_quals = (List *)
		fixup_varnode_to_origin((Node *)outer_dev_quals,
								cscan->custom_scan_tlist);
	pts->base_quals = ExecInitQual(outer_dev_quals, &pts->css.ss.ps);
	foreach (lc, cscan->custom_scan_tlist)
	{
		TargetEntry *tle = lfirst(lc);

		if (!tle->resjunk)
			tlist_dev = lappend(tlist_dev, tle);
	}
	pts->base_slot = MakeSingleTupleTableSlot(RelationGetDescr(rel),
											  table_slot_callbacks(rel));
	pts->base_proj = ExecBuildProjectionInfo(tlist_dev,
											 pts->css.ss.ps.ps_ExprContext,
											 pts->base_slot,
											 &pts->css.ss.ps,
											 RelationGetDescr(rel));
	/*
	 * Setup request buffer
	 */
	if (pts->af_state)			/* Apache Arrow */
	{
		pts->cb_next_chunk = pgstromScanChunkArrowFdw;
		pts->cb_next_tuple = pgstromScanNextTuple;
		 __setupTaskStateRequestBuffer(pts,
									   tupdesc_src,
									   tupdesc_dst,
									   KDS_FORMAT_ARROW);
	}
	else if (pts->gc_state)		/* GPU-Cache */
	{
		pts->cb_next_chunk = pgstromScanChunkGpuCache;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  NULL,
									  tupdesc_dst,
									  KDS_FORMAT_COLUMN);
	}
	else if (pts->gd_state)		/* GPU-Direct SQL */
	{
		pts->cb_next_chunk = pgstromRelScanChunkDirect;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  tupdesc_src,
									  tupdesc_dst,
									  KDS_FORMAT_BLOCK);
	}
	else if (pts->ds_entry)		/* DPU Storage */
	{
		pts->cb_next_chunk = pgstromRelScanChunkDirect;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  tupdesc_src,
									  tupdesc_dst,
									  KDS_FORMAT_BLOCK);
	}
	else
	{
		pts->cb_next_chunk = pgstromRelScanChunkNormal;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  tupdesc_src,
									  tupdesc_dst,
									  KDS_FORMAT_ROW);
	}
	/* other fields init */
	pts->curr_vm_buffer = InvalidBuffer;
}

/*
 * pgstromExecTaskState
 */
TupleTableSlot *
pgstromExecTaskState(pgstromTaskState *pts)
{
	TupleTableSlot *slot = NULL;
	XpuCommand	   *resp;

	while (!pts->curr_resp || !(slot = pts->cb_next_tuple(pts)))
	{
	next_chunks:
		if (pts->curr_resp)
			xpuClientPutResponse(pts->curr_resp);
		pts->curr_resp = __fetchNextXpuCommand(pts);
		if (!pts->curr_resp)
			return NULL;
		resp = pts->curr_resp;
		if (resp->tag == XpuCommandTag__Success)
		{
			if (resp->u.results.chunks_nitems == 0)
				goto next_chunks;
			pts->curr_kds = (kern_data_store *)
				((char *)resp + resp->u.results.chunks_offset);
			pts->curr_chunk = 0;
			pts->curr_index = 0;
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
 * pgstromExecEndTaskState
 */
void
pgstromExecEndTaskState(pgstromTaskState *pts)
{
	if (pts->curr_vm_buffer != InvalidBuffer)
		ReleaseBuffer(pts->curr_vm_buffer);
	if (pts->conn)
		xpuClientCloseSession(pts->conn);
	if (pts->br_state)
		pgstromBrinIndexExecEnd(pts);
	if (pts->base_slot)
		ExecDropSingleTupleTableSlot(pts->base_slot);
	if (pts->css.ss.ss_currentScanDesc)
		table_endscan(pts->css.ss.ss_currentScanDesc);
}

/*
 * pgstromExecResetTaskState
 */
void
pgstromExecResetTaskState(pgstromTaskState *pts)
{
	if (pts->conn)
	{
		xpuClientCloseSession(pts->conn);
		pts->conn = NULL;
	}
	if (pts->br_state)
		pgstromBrinIndexExecReset(pts);
}

/*
 * __xpuClientOpenSession
 */
void
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
