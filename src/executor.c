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
#include "cuda_common.h"

/*
 * XpuConnection
 */
struct XpuConnection
{
	dlist_node		chain;	/* link to gpuserv_connection_slots */
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
	if (xcmd->tag == XpuCommandTag__Error)
	{
		conn->num_running_cmds--;
		if (conn->errorbuf.errcode == ERRCODE_STROM_SUCCESS)
		{
			Assert(xcmd->u.error.errcode != ERRCODE_STROM_SUCCESS);
			memcpy(&conn->errorbuf, &xcmd->u.error, sizeof(kern_errorbuf));
		}
		free(xcmd);
	}
	else
	{
		if (xcmd->tag == XpuCommandTag__Success)
			conn->num_running_cmds--;
		else
		{
			/*
			 * NOTE: XpuCommandTag__SuccessHalfWay is used to return
			 * partial results of the request, but GPU-Service still
			 * continues the task execution, so we should not
			 * decrement 'num_running_cmds'.
			 */
			Assert(xcmd->tag == XpuCommandTag__SuccessHalfWay);
			xcmd->tag = XpuCommandTag__Success;
		}
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
		int		num_ready_cmds;

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
			fprintf(stderr, "[%s:%d] failed on poll(2): %m\n",
					__FILE_NAME__, __LINE__);
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
				/* check # of pending (ready) tasks */
				pthreadMutexLock(&conn->mutex);
				num_ready_cmds = conn->num_ready_cmds;
				pthreadMutexUnlock(&conn->mutex);

				if (num_ready_cmds >= pgstrom_max_async_tasks())
				{
					/*
					 * In case when the backend-side cannot handle the
					 * pending results at this moment, session worker
					 * temporary stops reading results from GPU service
					 * to stop dry running.
					 * Since poll(2) is level trigger, once pending tasks
					 * are processed, session worker continue to read.
					 */
					pg_usleep(2000L);
				}
				else if (__xpuConnectReceiveCommands(conn->sockfd,
													 conn) < 0)
				{
					break;
				}
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

				offset = __appendBinaryStringInfo(buf, temp, VARSIZE(temp));
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

/*
 * __build_session_kvars_defs
 */
static int
__count_session_kvars_defs_subfields(codegen_kvar_defitem *kvdef)
{
	int		count = list_length(kvdef->kv_subfields);
	ListCell *lc;

	foreach (lc, kvdef->kv_subfields)
	{
		codegen_kvar_defitem *__kvdef = lfirst(lc);

		count += __count_session_kvars_defs_subfields(__kvdef);
	}
	return count;
}

static int
__setup_session_kvars_defs_array(kern_varslot_desc *vslot_desc_root,
								 kern_varslot_desc *vslot_desc_base,
								 List *kvars_deflist)
{
	kern_varslot_desc *vs_desc = vslot_desc_base;
	int			nitems = list_length(kvars_deflist);
	int			count;
	ListCell   *lc;

	vslot_desc_base += nitems;
	foreach (lc, kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);

		vs_desc->vs_type_code = kvdef->kv_type_code;
		vs_desc->vs_typbyval  = kvdef->kv_typbyval;
        vs_desc->vs_typalign  = kvdef->kv_typalign;
        vs_desc->vs_typlen    = kvdef->kv_typlen;
		vs_desc->vs_typmod    = exprTypmod((Node *)kvdef->kv_expr);
		vs_desc->vs_offset    = kvdef->kv_offset;

		if (kvdef->kv_subfields != NIL)
		{
			count = __setup_session_kvars_defs_array(vslot_desc_root,
													 vslot_desc_base,
													 kvdef->kv_subfields);
			vs_desc->idx_subfield = (vslot_desc_base - vslot_desc_root);
			vs_desc->num_subfield = count;

			vslot_desc_base += count;
			nitems += count;
		}
		vs_desc++;
	}
	return nitems;
}

static void
__build_session_kvars_defs(pgstromTaskState *pts,
						   kern_session_info *session,
						   StringInfo buf)
{
	pgstromPlanInfo *pp_info = pts->pp_info;
	kern_varslot_desc *kvars_defs;
	uint32_t	nitems = list_length(pp_info->kvars_deflist);
	uint32_t	nrooms = nitems;
	uint32_t	sz;
	ListCell   *lc;

	foreach (lc, pp_info->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);

		nrooms += __count_session_kvars_defs_subfields(kvdef);
	}
	sz = sizeof(kern_varslot_desc) * nrooms;
	
	kvars_defs = alloca(sz);
	memset(kvars_defs, 0, sz);
	__setup_session_kvars_defs_array(kvars_defs,
									 kvars_defs,
									 pp_info->kvars_deflist);
	session->kcxt_kvars_nrooms = nrooms;
	session->kcxt_kvars_nslots = nitems;
	session->kcxt_kvars_defs = __appendBinaryStringInfo(buf, kvars_defs, sz);
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

static void
__build_session_lconvert(kern_session_info *session)
{
	struct lconv *lconvert = PGLC_localeconv();

	/* see comments about frac_digits in cash_in() */
	session->session_currency_frac_digits = lconvert->frac_digits;
}

const XpuCommand *
pgstromBuildSessionInfo(pgstromTaskState *pts,
						uint32_t join_inner_handle,
						TupleDesc kds_final_tdesc)
{
	pgstromSharedState *ps_state = pts->ps_state;
	pgstromPlanInfo *pp_info = pts->pp_info;
	ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
	ParamListInfo	param_info = econtext->ecxt_param_list_info;
	uint32_t		nparams = (param_info ? param_info->numParams : 0);
	uint32_t		session_sz;
	kern_session_info *session;
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
	if (pp_info->kvars_deflist != NIL)
		__build_session_kvars_defs(pts, session, &buf);
	if (pp_info->kexp_load_vars_packed)
	{
		xpucode = pp_info->kexp_load_vars_packed;
		session->xpucode_load_vars_packed =
			__appendBinaryStringInfo(&buf,
									 VARDATA(xpucode),
									 VARSIZE(xpucode) - VARHDRSZ);
	}
	if (pp_info->kexp_move_vars_packed)
	{
		xpucode = pp_info->kexp_move_vars_packed;
		session->xpucode_move_vars_packed =
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
	if (pp_info->kexp_gist_evals_packed)
	{
		xpucode = pp_info->kexp_gist_evals_packed;
		session->xpucode_gist_evals_packed =
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
	if (kds_final_tdesc)
	{
		size_t		head_sz = estimate_kern_data_store(kds_final_tdesc);
		kern_data_store *kds_temp = (kern_data_store *)alloca(head_sz);
		char		format = KDS_FORMAT_ROW;
		uint32_t	hash_nslots = 0;
		size_t		kds_length = (4UL << 20);	/* 4MB */

		if ((pts->xpu_task_flags & DEVTASK__PINNED_HASH_RESULTS) != 0)
		{
			double	final_nrows = pp_info->final_nrows;
			size_t	unit_sz;

			format = KDS_FORMAT_HASH;
			hash_nslots = KDS_GET_HASHSLOT_WIDTH(final_nrows);
			unit_sz = offsetof(kern_hashitem, t.htup) +
				MAXALIGN(offsetof(HeapTupleHeaderData, t_bits) +
						 BITMAPLEN(kds_final_tdesc->natts)) +
				pts->css.ss.ps.plan->plan_width;

			/* kds_final->length with margin */
			kds_length = (head_sz +
						  sizeof(uint64_t) * hash_nslots +	/* hash-slots */
						  sizeof(uint64_t) * 2 * hash_nslots +	/* row-index */
						  unit_sz * 2 * hash_nslots);
			if (kds_length < (1UL<<30))
				kds_length = (1UL<<30);		/* 1GB at least */
			else
				kds_length = TYPEALIGN(1024, kds_length);	/* alignment */
		}
		setup_kern_data_store(kds_temp, kds_final_tdesc, kds_length, format);
		kds_temp->hash_nslots = hash_nslots;
		session->groupby_kds_final = __appendBinaryStringInfo(&buf, kds_temp, head_sz);
		session->groupby_prepfn_bufsz = pp_info->groupby_prepfn_bufsz;
		session->groupby_ngroups_estimation = pts->css.ss.ps.plan->plan_rows;
	}
	/* CPU fallback related */
	if (pgstrom_cpu_fallback_elevel < ERROR)
	{
		TupleDesc	scan_desc = pts->css.ss.ps.scandesc;
		size_t		sz = estimate_kern_data_store(scan_desc);
		kern_data_store *kds_head = (kern_data_store *)alloca(sz);
		size_t		kds_length = PGSTROM_CHUNK_SIZE;
		int			nitems = ((VARSIZE(pts->kern_fallback_desc) -
							   VARHDRSZ) / sizeof(kern_fallback_desc));

		setup_kern_data_store(kds_head,
							  scan_desc,
							  kds_length,
							  KDS_FORMAT_FALLBACK);
		session->fallback_kds_head = __appendBinaryStringInfo(&buf, kds_head, sz);
		session->fallback_desc_defs =
			__appendBinaryStringInfo(&buf, VARDATA(pts->kern_fallback_desc),
									 sizeof(kern_fallback_desc) * nitems);
		session->fallback_desc_nitems = nitems;
	}
	/* other database session information */
	session->query_plan_id = ps_state->query_plan_id;
	session->xpu_task_flags = pts->xpu_task_flags;
	session->optimal_gpus = pts->optimal_gpus;
	session->kcxt_kvecs_bufsz = pp_info->kvecs_bufsz;
	session->kcxt_kvecs_ndims = pp_info->kvecs_ndims;
	session->kcxt_extra_bufsz = pp_info->extra_bufsz;
	session->cuda_stack_size  = pp_info->cuda_stack_size;
	session->xpu_task_flags = pts->xpu_task_flags;
	session->hostEpochTimestamp = SetEpochTimestamp();
	session->xactStartTimestamp = GetCurrentTransactionStartTimestamp();
	session->session_xact_state = __build_session_xact_state(&buf);
	session->session_timezone = __build_session_timezone(&buf);
	session->session_encode = __build_session_encode(&buf);
	__build_session_lconvert(session);
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
	HeapScanDesc	h_scan = (HeapScanDesc)pts->css.ss.ss_currentScanDesc;
	XpuConnection  *conn = pts->conn;
	uint32_t		curval, newval;
	uint32_t		block_nums = 0;
	uint32_t		block_start = 0;

	/* init heap-scan position, if any */
	if (h_scan)
	{
		block_nums	= h_scan->rs_nblocks;
		block_start	= h_scan->rs_startblock;
	}
	ps_state->scan_block_nums  = block_nums;
	ps_state->scan_block_start = block_start;

	/* update the parallel_task_control */
	Assert(conn != NULL);
	curval = pg_atomic_read_u32(&ps_state->parallel_task_control);
	do {
		if ((curval & 1) != 0)
			return false;
		newval = curval + 2;
	} while (!pg_atomic_compare_exchange_u32(&ps_state->parallel_task_control,
											 &curval, newval));
	return true;
}

/*
 * pgstromTaskStateEndScan
 */
static bool
pgstromTaskStateEndScan(pgstromTaskState *pts)
{
	pgstromSharedState *ps_state = pts->ps_state;
	XpuConnection  *conn = pts->conn;
	uint32_t		curval, newval;

	Assert(conn != NULL);
	curval = pg_atomic_read_u32(&ps_state->parallel_task_control);
	do {
		Assert(curval >= 2);
		newval = ((curval - 2) | 1);
	} while (!pg_atomic_compare_exchange_u32(&ps_state->parallel_task_control,
											 &curval, newval));
	return (newval == 1);
}

/*
 * __updateStatsXpuCommand
 */
static void
__updateStatsXpuCommand(pgstromTaskState *pts, const XpuCommand *xcmd)
{
	if (xcmd->tag == XpuCommandTag__Success)
	{
		pgstromSharedState *ps_state = pts->ps_state;
		int		n_rels = Min(pts->num_rels, xcmd->u.results.num_rels);

		pg_atomic_fetch_add_u64(&ps_state->npages_direct_read,
								xcmd->u.results.npages_direct_read);
		pg_atomic_fetch_add_u64(&ps_state->npages_vfs_read,
								xcmd->u.results.npages_vfs_read);
		pg_atomic_fetch_add_u64(&ps_state->source_ntuples_raw,
								xcmd->u.results.nitems_raw);
		pg_atomic_fetch_add_u64(&ps_state->source_ntuples_in,
								xcmd->u.results.nitems_in);
		for (int i=0; i < n_rels; i++)
		{
			pg_atomic_fetch_add_u64(&ps_state->inners[i].stats_gist,
									xcmd->u.results.stats[i].nitems_gist);
			pg_atomic_fetch_add_u64(&ps_state->inners[i].stats_join,
									xcmd->u.results.stats[i].nitems_out);
		}
		pg_atomic_fetch_add_u64(&ps_state->result_ntuples, xcmd->u.results.nitems_out);
		if (xcmd->u.results.final_plan_task)
		{
			pg_atomic_fetch_add_u64(&ps_state->final_nitems,
									xcmd->u.results.final_nitems);
			pg_atomic_fetch_add_u64(&ps_state->final_usage,
									xcmd->u.results.final_usage);
			pg_atomic_fetch_add_u64(&ps_state->final_total,
									xcmd->u.results.final_total);
		}
	}
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
							conn->errorbuf.message)));

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
				pts->final_done = true;
				if (try_final_callback &&
					pgstromTaskStateEndScan(pts) &&
					pts->cb_final_chunk != NULL)
				{
					xcmd = pts->cb_final_chunk(pts, xcmd_iov, &xcmd_iovcnt);
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
	__updateStatsXpuCommand(pts, xcmd);
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
	int				max_async_tasks = pgstrom_max_async_tasks();

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
							conn->errorbuf.message)));
		}

		if ((conn->num_running_cmds + conn->num_ready_cmds) < max_async_tasks &&
			(dlist_is_empty(&conn->ready_cmds_list) ||
			 conn->num_running_cmds < max_async_tasks / 2))
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
			__updateStatsXpuCommand(pts, xcmd);
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
 * CPU Fallback Routines
 */
static inline int
tryExecCpuFallbackChunks(pgstromTaskState *pts)
{
	int		nchunks = pts->curr_resp->u.results.chunks_nitems;

	while (pts->curr_chunk < nchunks)
	{
		kern_data_store *kds = pts->curr_kds;

		if (kds->format == KDS_FORMAT_FALLBACK)
		{
			execCpuFallbackOneChunk(pts);
			/* move to the next chunk */
			pts->curr_kds = (kern_data_store *)
				((char *)kds + kds->length);
			pts->curr_chunk++;
			pts->curr_index = 0;
		}
		else
		{
			break;
		}
	}
	return (nchunks - pts->curr_chunk);
}

/*
 * pgstromScanNextTuple
 */
static TupleTableSlot *
pgstromScanNextTuple(pgstromTaskState *pts)
{
	TupleTableSlot *slot = pts->css.ss.ss_ScanTupleSlot;

	do {
		kern_data_store *kds = pts->curr_kds;
		int64_t		index = pts->curr_index++;

		if (index < kds->nitems)
		{
			kern_tupitem   *tupitem = KDS_GET_TUPITEM(kds, index);

			pts->curr_htup.t_len = tupitem->t_len;
			memcpy(&pts->curr_htup.t_self,
				   &tupitem->htup.t_ctid,
				   sizeof(ItemPointerData));
			pts->curr_htup.t_data = &tupitem->htup;
			return ExecStoreHeapTuple(&pts->curr_htup, slot, false);
		}
		if (++pts->curr_chunk >= pts->curr_resp->u.results.chunks_nitems)
			break;
		pts->curr_kds = (kern_data_store *)
			((char *)kds + kds->length);
		pts->curr_index = 0;
	} while (tryExecCpuFallbackChunks(pts) > 0);

	return NULL;
}

/*
 * pgstromExecFinalChunk
 */
static XpuCommand *
pgstromExecFinalChunk(pgstromTaskState *pts,
					  struct iovec *xcmd_iov, int *xcmd_iovcnt)
{
	XpuCommand	   *xcmd;

	pts->xcmd_buf.len = offsetof(XpuCommand, u);
	enlargeStringInfo(&pts->xcmd_buf, 0);

	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	memset(xcmd, 0, sizeof(XpuCommand));
	xcmd->magic  = XpuCommandMagicNumber;
	xcmd->tag    = XpuCommandTag__XpuTaskFinal;
	xcmd->length = offsetof(XpuCommand, u);

	xcmd_iov[0].iov_base = xcmd;
	xcmd_iov[0].iov_len  = offsetof(XpuCommand, u);
	*xcmd_iovcnt = 1;

	return xcmd;
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
	if (pts->gcache_desc)
		bufsz += MAXALIGN(sizeof(GpuCacheIdent));
	if (tdesc_src)
		bufsz += estimate_kern_data_store(tdesc_src);
	if (tdesc_dst)
		bufsz += estimate_kern_data_store(tdesc_dst);
	enlargeStringInfo(&pts->xcmd_buf, bufsz);

	xcmd = (XpuCommand *)pts->xcmd_buf.data;
	memset(xcmd, 0, offsetof(XpuCommand, u.task.data));
	off = offsetof(XpuCommand, u.task.data);
	if (pts->gcache_desc)
	{
		const GpuCacheIdent *ident = getGpuCacheDescIdent(pts->gcache_desc);

		memcpy((char *)xcmd + off, ident, sizeof(GpuCacheIdent));
		off += MAXALIGN(sizeof(GpuCacheIdent));
	}
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
	Assert(off <= bufsz);
	xcmd->magic = XpuCommandMagicNumber;
	xcmd->tag   = (!pts->gcache_desc
				   ? XpuCommandTag__XpuTaskExec
				   : XpuCommandTag__XpuTaskExecGpuCache);
	xcmd->length = off;
	pts->xcmd_buf.len = off;
}

/*
 * __fixup_fallback_projection
 */
static Node *
__fixup_fallback_projection(Node *node, void *__data)
{
	List	   *custom_scan_tlist = __data;
	ListCell   *lc;

	if (!node)
		return NULL;
	foreach (lc, custom_scan_tlist)
	{
		TargetEntry *tle = lfirst(lc);

		if (tle->resorigtbl != (Oid)UINT_MAX &&
			equal(tle->expr, node))
		{
			return (Node *)makeVar(INDEX_VAR,
								   tle->resno,
								   exprType(node),
								   exprTypmod(node),
								   exprCollation(node),
								   0);
		}
	}

	if (IsA(node, Var))
	{
		return (Node *)makeNullConst(exprType(node),
									 exprTypmod(node),
									 exprCollation(node));
	}
	return expression_tree_mutator(node, __fixup_fallback_projection, __data);
}

/*
 * fallback_varload_mapping
 */
static int
__compare_fallback_desc_by_dst_resno(const void *__a, const void *__b)
{
	const kern_fallback_desc *a = __a;
	const kern_fallback_desc *b = __b;

	Assert(a->fb_dst_resno > 0 && b->fb_dst_resno > 0);
	if (a->fb_dst_resno < b->fb_dst_resno)
		return -1;
	if (a->fb_dst_resno > b->fb_dst_resno)
		return 1;
	return 0;
}

static int
__compare_fallback_desc_by_src_depth_resno(const void *__a, const void *__b)
{
	const kern_fallback_desc *a = __a;
	const kern_fallback_desc *b = __b;

	if (a->fb_src_depth < b->fb_src_depth)
		return -1;
	if (a->fb_src_depth > b->fb_src_depth)
		return  1;
	if (a->fb_src_resno < b->fb_src_resno)
		return -1;
	if (a->fb_src_resno > b->fb_src_resno)
		return  1;
	return 0;
}

/*
 * __execInitTaskStateCpuFallback
 */
static void
__execInitTaskStateCpuFallback(pgstromTaskState *pts)
{
	pgstromPlanInfo *pp_info = pts->pp_info;
	CustomScan *cscan = (CustomScan *)pts->css.ss.ps.plan;
	Relation	rel = pts->css.ss.ss_currentRelation;
	List	   *fallback_proj = NIL;
	ListCell   *lc1, *lc2;
	int			nrooms = list_length(cscan->custom_scan_tlist);
	int			nitems = 0;
	int			last_depth = -1;
	List	   *src_list = NIL;
	List	   *dst_list = NIL;
	bool		compatible = true;
	bytea	   *vl_temp;
	kern_fallback_desc *__fb_desc_array;

	/*
	 * WHERE-clause
	 */
	pts->base_quals = ExecInitQual(pp_info->scan_quals, &pts->css.ss.ps);
	pts->base_slot = MakeSingleTupleTableSlot(RelationGetDescr(rel),
											  table_slot_callbacks(rel));
	/*
	 * CPU-Projection
	 */
	__fb_desc_array = alloca(sizeof(kern_fallback_desc) * nrooms);
	foreach (lc1, cscan->custom_scan_tlist)
	{
		TargetEntry *tle = lfirst(lc1);
		ExprState  *state = NULL;

		if (tle->resorigtbl >= 0 &&
			tle->resorigtbl <= pts->num_rels)
		{
			kern_fallback_desc *fb_desc = &__fb_desc_array[nitems++];

			fb_desc->fb_src_depth = tle->resorigtbl;
			fb_desc->fb_src_resno = tle->resorigcol;
			fb_desc->fb_dst_resno = tle->resno;
			fb_desc->fb_max_depth = pts->num_rels + 1;
			fb_desc->fb_slot_id   = -1;
			fb_desc->fb_kvec_offset = -1;

			foreach (lc2, pp_info->kvars_deflist)
			{
				codegen_kvar_defitem *kvdef = lfirst(lc2);

				if (tle->resorigtbl == kvdef->kv_depth &&
					tle->resorigcol == kvdef->kv_resno)
				{
					fb_desc->fb_max_depth   = kvdef->kv_maxref;
					fb_desc->fb_slot_id     = kvdef->kv_slot_id;
					fb_desc->fb_kvec_offset = kvdef->kv_offset;
					break;
				}
			}
			fallback_proj = lappend(fallback_proj, NULL);
		}
		else
		{
			Node	   *expr;

			Assert(tle->resorigtbl == (Oid)UINT_MAX);
			expr = __fixup_fallback_projection((Node *)tle->expr,
											   cscan->custom_scan_tlist);
			state = ExecInitExpr((Expr *)expr, &pts->css.ss.ps);
			compatible = false;
			fallback_proj = lappend(fallback_proj, state);
		}
	}
	if (!compatible)
		pts->fallback_proj = fallback_proj;
	/* session->fallback_desc_defs */
	qsort(__fb_desc_array, nitems,
		  sizeof(kern_fallback_desc),
		  __compare_fallback_desc_by_dst_resno);
	vl_temp = palloc(VARHDRSZ + sizeof(kern_fallback_desc) * nitems);
	SET_VARSIZE(vl_temp, VARHDRSZ + sizeof(kern_fallback_desc) * nitems);
	memcpy(VARDATA(vl_temp), __fb_desc_array,
		   sizeof(kern_fallback_desc) * nitems);
	pts->kern_fallback_desc = vl_temp;

	/* fallback var-loads */
	qsort(__fb_desc_array, nitems,
		  sizeof(kern_fallback_desc),
		  __compare_fallback_desc_by_src_depth_resno);

	for (int i=0; i <= nitems; i++)
	{
		kern_fallback_desc *fb_desc = &__fb_desc_array[i];

		if (i == nitems || fb_desc->fb_src_depth != last_depth)
		{
			if (last_depth == 0)
			{
				pts->fallback_load_src = src_list;
				pts->fallback_load_dst = dst_list;
			}
			else if (last_depth > 0 &&
					 last_depth <= pts->num_rels)
			{
				pts->inners[last_depth-1].inner_load_src = src_list;
				pts->inners[last_depth-1].inner_load_dst = dst_list;
			}
			src_list = NIL;
			dst_list = NIL;
			if (i == nitems)
				break;
		}
		last_depth = fb_desc->fb_src_depth;
		src_list = lappend_int(src_list, fb_desc->fb_src_resno);
		dst_list = lappend_int(dst_list, fb_desc->fb_dst_resno);
	}
	Assert(src_list == NIL && dst_list == NIL);
}

/*
 * pgstromCreateTaskState
 */
Node *
pgstromCreateTaskState(CustomScan *cscan,
					   const CustomExecMethods *methods)
{
	pgstromPlanInfo *pp_info = deform_pgstrom_plan_info(cscan);
	pgstromTaskState *pts;
	int		num_rels = list_length(cscan->custom_plans);

	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = methods;
#if PG_VERSION_NUM >= 160000
	pts->css.slotOps = &TTSOpsHeapTuple;
#endif
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert(pp_info->num_rels == num_rels);
	pts->num_scan_repeats = 1;
	pts->num_rels = num_rels;
	pts->curr_tbm = palloc0(offsetof(TBMIterateResult, offsets) +
							sizeof(OffsetNumber) * MaxHeapTuplesPerPage);
	pts->curr_repeat_id = -1;

	return (Node *)pts;
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
	ListCell   *lc;

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
		const char *kds_pathname = smgr_relpath(smgr, MAIN_FORKNUM);

		if (am_oid != HEAP_TABLE_AM_OID)
			elog(ERROR, "PG-Strom does not support table access method: %s",
				 get_am_name(am_oid));
		/* Is GPU-Cache available? */
		pts->gcache_desc = pgstromGpuCacheExecInit(pts);
		if (pts->gcache_desc)
			pts->xpu_task_flags |= DEVTASK__USED_GPUCACHE;
		else
		{
			/* setup BRIN-index if any */
			pgstromBrinIndexExecBegin(pts,
									  pp_info->brin_index_oid,
									  pp_info->brin_index_conds,
									  pp_info->brin_index_quals);
			if ((pts->xpu_task_flags & DEVKIND__NVIDIA_GPU) != 0)
			{
				pts->optimal_gpus = GetOptimalGpuForRelation(rel);
				if (pts->optimal_gpus)
				{
					/*
					 * If particular GPUs are optimal, we can use
					 * GPU-Direct SQL for the table scan.
					 */
					pts->xpu_task_flags |= DEVTASK__USED_GPUDIRECT;
				}
				else
				{
					pts->optimal_gpus = GetSystemAvailableGpus();
				}
			}
			if ((pts->xpu_task_flags & DEVKIND__NVIDIA_DPU) != 0)
				pts->ds_entry = GetOptimalDpuForRelation(rel, &kds_pathname);
		}
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

	/* TupleDesc according to GpuProjection */
	tupdesc_dst = pts->css.ss.ps.scandesc;
#if PG_VERSION_NUM < 160000
	/*
	 * PG16 adds CustomScanState::slotOps to initialize scan-tuple-slot
	 * with the specified tuple-slot-ops.
	 * GPU projection returns tuples in heap-format, so we prefer
	 * TTSOpsHeapTuple, instead of the TTSOpsVirtual.
	 */
	ExecInitScanTupleSlot(estate, &pts->css.ss, tupdesc_dst,
						  &TTSOpsHeapTuple);
	ExecAssignScanProjectionInfoWithVarno(&pts->css.ss, INDEX_VAR);
#endif
	/*
	 * Initialize the CPU Fallback stuff
	 */
	__execInitTaskStateCpuFallback(pts);
	
	/*
	 * init inner relations
	 */
	depth_index = 0;
	foreach (lc, cscan->custom_plans)
	{
		pgstromTaskInnerState *istate = &pts->inners[depth_index];
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[depth_index];
		Plan	   *plan = lfirst(lc);
		ListCell   *cell;

		istate->ps = ExecInitNode(plan, estate, eflags);
		istate->econtext = CreateExprContext(estate);
		istate->depth = depth_index + 1;
		istate->join_type = pp_inner->join_type;
		istate->join_quals = ExecInitQual(pp_inner->join_quals,
										  &pts->css.ss.ps);
		istate->other_quals = ExecInitQual(pp_inner->other_quals,
										   &pts->css.ss.ps);

		foreach (cell, pp_inner->hash_outer_keys)
		{
			Node	   *outer_key = (Node *)lfirst(cell);
			ExprState  *es;
			devtype_info *dtype;

			dtype = pgstrom_devtype_lookup(exprType(outer_key));
			if (!dtype)
				elog(ERROR, "failed on lookup device type of %s",
					 nodeToString(outer_key));
			es = ExecInitExpr((Expr *)outer_key, &pts->css.ss.ps);
			istate->hash_outer_keys = lappend(istate->hash_outer_keys, es);
			istate->hash_outer_funcs = lappend(istate->hash_outer_funcs,
											   dtype->type_hashfunc);
		}
		/* inner hash-keys references the result of inner-slot */
		foreach (cell, pp_inner->hash_inner_keys)
		{
			Node	   *inner_key = (Node *)lfirst(cell);
			ExprState  *es;
			devtype_info *dtype;

			dtype = pgstrom_devtype_lookup(exprType(inner_key));
			if (!dtype)
				elog(ERROR, "failed on lookup device type of %s",
					 nodeToString(inner_key));
			es = ExecInitExpr((Expr *)inner_key, &pts->css.ss.ps);
			istate->hash_inner_keys = lappend(istate->hash_inner_keys, es);
			istate->hash_inner_funcs = lappend(istate->hash_inner_funcs,
											   dtype->type_hashfunc);
		}
		/* gist-index initialization */
		if (OidIsValid(pp_inner->gist_index_oid))
		{
			istate->gist_irel = index_open(pp_inner->gist_index_oid,
										   AccessShareLock);
			istate->gist_clause = ExecInitExpr((Expr *)pp_inner->gist_clause,
											   &pts->css.ss.ps);
			istate->gist_ctid_resno = pp_inner->gist_ctid_resno;
		}
		/* require the pinned results if GpuScan/Join results may large */
		if (pp_inner->inner_pinned_buffer)
		{
			pgstromTaskState   *i_pts = (pgstromTaskState *)istate->ps;

			Assert(pgstrom_is_gpuscan_state(istate->ps) ||
				   pgstrom_is_gpujoin_state(istate->ps));
			if (pp_inner->hash_inner_keys != NIL &&
				pp_inner->hash_outer_keys != NIL)
				i_pts->xpu_task_flags |= DEVTASK__PINNED_HASH_RESULTS;
			else
				i_pts->xpu_task_flags |= DEVTASK__PINNED_ROW_RESULTS;

			istate->inner_pinned_buffer = true;
		}
		pts->css.custom_ps = lappend(pts->css.custom_ps, istate->ps);
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
	else if (pts->gcache_desc)		/* GPU-Cache */
	{
		pts->cb_next_chunk = pgstromScanChunkGpuCache;
		pts->cb_next_tuple = pgstromScanNextTuple;
		__setupTaskStateRequestBuffer(pts,
									  NULL,
									  tupdesc_dst,
									  KDS_FORMAT_COLUMN);
	}
	else if ((pts->xpu_task_flags & DEVTASK__USED_GPUDIRECT) != 0 ||
			 pts->ds_entry)
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
	if ((pts->xpu_task_flags & DEVTASK__SCAN) != 0)
	{
		pts->cb_final_chunk = pgstromExecFinalChunk;
	}
	else if ((pts->xpu_task_flags & DEVTASK__JOIN) != 0)
	{
		pts->cb_final_chunk = pgstromExecFinalChunk;
	}
	else if ((pts->xpu_task_flags & DEVTASK__PREAGG) != 0)
	{
		pts->cb_final_chunk = pgstromExecFinalChunk;
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
		if (resp->tag == XpuCommandTag__Success)
		{
			if (resp->u.results.final_plan_task)
			{
				ExecFallbackCpuJoinOuterJoinMap(pts, resp);
				ExecFallbackCpuJoinRightOuter(pts);
			}
			if (resp->u.results.chunks_nitems == 0)
				goto next_chunks;
			pts->curr_kds = (kern_data_store *)
				((char *)resp + resp->u.results.chunks_offset);
			pts->curr_chunk = 0;
			pts->curr_index = 0;
			if (tryExecCpuFallbackChunks(pts) == 0)
				goto next_chunks;
		}
		else
		{
			elog(ERROR, "unknown response tag: %u", resp->tag);
		}
	}
	slot_getallattrs(slot);
	return slot;
}

/*
 * pgstromExecScanReCheck
 */
static TupleTableSlot *
pgstromExecScanReCheck(pgstromTaskState *pts, EPQState *epqstate)
{
	Index		scanrelid = ((Scan *)pts->css.ss.ps.plan)->scanrelid;

	/* see ExecScanFetch */
	if (scanrelid == 0)
	{
		elog(ERROR, "Bug? CustomScan(%s) has scanrelid==0",
			 pts->css.methods->CustomName);
	}
	else if (epqstate->relsubs_done[scanrelid-1])
	{
		return NULL;
	}
	else if (epqstate->relsubs_slot[scanrelid-1])
	{
		TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
		TupleTableSlot *epq_slot = epqstate->relsubs_slot[scanrelid-1];

		Assert(epqstate->relsubs_rowmark[scanrelid - 1] == NULL);
		/* Mark to remember that we shouldn't return it again */
		epqstate->relsubs_done[scanrelid - 1] = true;

		/* Return empty slot if we haven't got a test tuple */
		if (TupIsNull(epq_slot))
			ExecClearTuple(scan_slot);
		else
		{
			HeapTuple	epq_tuple;
			size_t		__fallback_nitems = pts->fallback_nitems;
			size_t		__fallback_usage  = pts->fallback_usage;
			bool		should_free;
#if 0
			slot_getallattrs(epq_slot);
			for (int j=0; j < epq_slot->tts_nvalid; j++)
			{
				elog(INFO, "epq_slot[%d] isnull=%s values=0x%lx", j,
					 epq_slot->tts_isnull[j] ? "true" : "false",
					 epq_slot->tts_values[j]);
			}
#endif
			epq_tuple = ExecFetchSlotHeapTuple(epq_slot, false,
											   &should_free);
			execCpuFallbackBaseTuple(pts, epq_tuple);
			if (pts->fallback_tuples != NULL &&
				pts->fallback_buffer != NULL &&
				pts->fallback_nitems > __fallback_nitems &&
				pts->fallback_usage  > __fallback_usage)
			{
				HeapTupleData	htup;
				kern_tupitem   *titem = (kern_tupitem *)
					(pts->fallback_buffer +
					 pts->fallback_tuples[pts->fallback_index]);

				htup.t_len = titem->t_len;
				htup.t_data = &titem->htup;
				scan_slot = pts->css.ss.ss_ScanTupleSlot;
				ExecForceStoreHeapTuple(&htup, scan_slot, false);
			}
			else
			{
				ExecClearTuple(scan_slot);
			}
			/* release fallback tuple & buffer */
			if (should_free)
				pfree(epq_tuple);
			pts->fallback_nitems = __fallback_nitems;
			pts->fallback_usage  = __fallback_usage;
		}
		return scan_slot;
	}
	else if (epqstate->relsubs_rowmark[scanrelid-1])
	{
		elog(ERROR, "RowMark on CustomScan(%s) is not implemented yet",
			 pts->css.methods->CustomName);
	}
	return pgstromExecScanAccess(pts);
}

/*
 * __pgstromExecTaskOpenConnection
 */
static bool
__pgstromExecTaskOpenConnection(pgstromTaskState *pts)
{
	const XpuCommand *session;
	uint32_t	inner_handle = 0;
	TupleDesc	kds_final_tupdesc = NULL;

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
	if ((pts->xpu_task_flags & (DEVTASK__PINNED_HASH_RESULTS |
								DEVTASK__PINNED_ROW_RESULTS)) != 0)
	{
		CustomScan *cscan = (CustomScan *)pts->css.ss.ps.plan;
		ListCell   *lc;
		int			nvalids = 0;

		/*
		 * MEMO: 'scandesc' often contains junk fields, used only
		 * for EXPLAIN output, thus GpuPreAgg results shall not have
		 * any valid values for the junk, and these fields in kds_final
		 * are waste of space (not only colmeta array, it affects the
		 * length of BITMAPLEN(kds->ncols) and may expand the starting
		 * point of t_hoff for all the tuples.
		 */
		kds_final_tupdesc = CreateTupleDescCopy(pts->css.ss.ps.scandesc);
		foreach (lc, cscan->custom_scan_tlist)
		{
			TargetEntry *tle = lfirst(lc);

			if (!tle->resjunk)
				nvalids = tle->resno;
		}
		Assert(nvalids <= kds_final_tupdesc->natts);
		kds_final_tupdesc->natts = nvalids;
	}
	/* build the session information */
	session = pgstromBuildSessionInfo(pts, inner_handle, kds_final_tupdesc);
	if ((pts->xpu_task_flags & DEVKIND__NVIDIA_GPU) != 0)
	{
		gpuClientOpenSession(pts, session);
		GpuJoinInnerPreloadAfterWorks(pts);
	}
	else if ((pts->xpu_task_flags & DEVKIND__NVIDIA_DPU) != 0)
	{
		Assert(pts->ds_entry != NULL);
		DpuClientOpenSession(pts, session);
	}
	else
	{
		elog(ERROR, "Bug? unknown PG-Strom task kind: %08x", pts->xpu_task_flags);
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
	EState		   *estate = pts->css.ss.ps.state;
	ExprState	   *host_quals = node->ss.ps.qual;
	ExprContext	   *econtext = node->ss.ps.ps_ExprContext;
	ProjectionInfo *proj_info = node->ss.ps.ps_ProjInfo;
	TupleTableSlot *slot;

	if (!pts->conn)
	{
		if (!__pgstromExecTaskOpenConnection(pts))
			return NULL;
		Assert(pts->conn);
	}

	/*
	 * see, ExecScan() - it assumes CustomScan with scanrelid > 0 returns
	 * tuples identical with table definition, thus these tuples are not
	 * suitable for input of ExecProjection().
	 */
	if (estate->es_epq_active)
	{
		slot = pgstromExecScanReCheck(pts, estate->es_epq_active);
		if (TupIsNull(slot))
			return NULL;
		ResetExprContext(econtext);
		econtext->ecxt_scantuple = slot;
		if (proj_info)
			return ExecProject(proj_info);
		return slot;
	}

	for (;;)
	{
		slot = pgstromExecScanAccess(pts);
		if (TupIsNull(slot))
			break;
		/* check whether the current tuple satisfies the qual-clause */
		ResetExprContext(econtext);
		econtext->ecxt_scantuple = slot;

		if (!host_quals || ExecQual(host_quals, econtext))
		{
			if (proj_info)
				return ExecProject(proj_info);
			return slot;
		}
		InstrCountFiltered1(pts, 1);
	}
	return NULL;
}

/*
 * execInnerPreLoadPinnedOneDepth
 *
 * It runs the supplied pgstromTaskState to build inner-pinned-buffer
 * on the device memory. It shall be reused as a part of GpuJoin inner
 * buffer, so no need to handle its results on the CPU side.
 */
void
execInnerPreLoadPinnedOneDepth(pgstromTaskState *pts,
							   pg_atomic_uint64 *p_inner_nitems,
							   pg_atomic_uint64 *p_inner_usage,
							   pg_atomic_uint64 *p_inner_total,
							   uint64_t *p_inner_buffer_id)
{
	XpuCommand *resp;
	uint64_t	ival;

	if (pts->css.ss.ps.instrument)
		InstrStartNode(pts->css.ss.ps.instrument);

	if (!pts->conn)
	{
		if (!__pgstromExecTaskOpenConnection(pts))
			goto skip;
		Assert(pts->conn);
	}

	for (;;)
	{
		resp = __fetchNextXpuCommand(pts);
		if (!resp)
			break;
		if (resp->tag == XpuCommandTag__Success)
		{
			if (resp->u.results.ojmap_offset != 0 ||
				resp->u.results.chunks_nitems != 0)
				elog(ERROR, "GPU Service returned valid contents, but should be pinned buffer");
		}
		else
		{
			elog(ERROR, "unexpected response tag: %u", resp->tag);
		}
	}
	ival = pg_atomic_read_u64(&pts->ps_state->final_nitems);
	pg_atomic_write_u64(p_inner_nitems, ival);
	ival = pg_atomic_read_u64(&pts->ps_state->final_usage);
	pg_atomic_write_u64(p_inner_usage,  ival);
	ival = pg_atomic_read_u64(&pts->ps_state->final_total);
	pg_atomic_write_u64(p_inner_total,  ival);
skip:
	*p_inner_buffer_id    = pts->ps_state->query_plan_id;

	if (pts->css.ss.ps.instrument)
		InstrStopNode(pts->css.ss.ps.instrument, -1.0);
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
	if (pts->gcache_desc)
		pgstromGpuCacheExecEnd(pts);
	if (pts->arrow_state)
		pgstromArrowFdwExecEnd(pts->arrow_state);
	if (pts->base_slot)
		ExecDropSingleTupleTableSlot(pts->base_slot);
	if (pts->css.ss.ss_currentScanDesc)
		table_endscan(pts->css.ss.ss_currentScanDesc);
	for (int i=0; i < pts->num_rels; i++)
	{
		pgstromTaskInnerState *istate = &pts->inners[i];

		if (istate->gist_irel)
			index_close(istate->gist_irel, AccessShareLock);
	}
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
	pgstromSharedState *ps_state = pts->ps_state;
	Relation	rel = node->ss.ss_currentRelation;
	TableScanDesc scan = node->ss.ss_currentScanDesc;

	if (pts->conn)
	{
		xpuClientCloseSession(pts->conn);
		pts->conn = NULL;
	}
	/* reset status */
	if (pts->br_state)
		pgstromBrinIndexExecReset(pts);
	if (pts->arrow_state)
		pgstromArrowFdwExecReset(pts->arrow_state);
	if (!scan->rs_parallel)
		table_rescan(scan, NULL);
	else
		table_parallelscan_reinitialize(rel, scan->rs_parallel);
	if (ps_state)
		pg_atomic_write_u64(&ps_state->scan_block_count, 0);
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
	Size		len = 0;

	if (pts->br_state)
		len += pgstromBrinIndexEstimateDSM(pts);
	len += MAXALIGN(offsetof(pgstromSharedState, inners[num_rels]));

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
	size_t		dsm_length = offsetof(pgstromSharedState, inners[num_rels]);
	char	   *dsm_addr = coordinate;
	pgstromSharedState *ps_state;
	TableScanDesc scan = NULL;

	Assert(!AmBackgroundWorkerProcess());

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

		/* parallel scan descriptor */
		if (pts->gcache_desc)
			pgstromGpuCacheInitDSM(pts, ps_state);
		if (pts->arrow_state)
			pgstromArrowFdwInitDSM(pts->arrow_state, ps_state);
		else
		{
			ParallelTableScanDesc pdesc = (ParallelTableScanDesc) dsm_addr;

			table_parallelscan_initialize(relation, pdesc, snapshot);
			scan = table_beginscan_parallel(relation, pdesc);
			ps_state->parallel_scan_desc_offset = ((char *)pdesc - (char *)ps_state);
		}
	}
	else
	{
		ps_state = MemoryContextAllocZero(estate->es_query_cxt, dsm_length);
		ps_state->ss_handle = DSM_HANDLE_INVALID;
		ps_state->ss_length = dsm_length;

		/* scan descriptor */
		if (pts->gcache_desc)
			pgstromGpuCacheInitDSM(pts, ps_state);
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
	pgstromSharedState *ps_state;
	char	   *dsm_addr = coordinate;
	int			num_rels = list_length(pts->css.custom_ps);

	if (pts->br_state)
		dsm_addr += pgstromBrinIndexAttachDSM(pts, dsm_addr);
	pts->ps_state = ps_state = (pgstromSharedState *)dsm_addr;
	Assert(ps_state->num_rels == num_rels);
	dsm_addr += MAXALIGN(offsetof(pgstromSharedState, inners[num_rels]));

	if (pts->gcache_desc)
		pgstromGpuCacheAttachDSM(pts, pts->ps_state);
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
	if (pts->gcache_desc)
		pgstromGpuCacheShutdown(pts);
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
 * pgstromGpuDirectExplain
 */
static void
pgstromGpuDirectExplain(pgstromTaskState *pts,
						ExplainState *es, List *dcontext)
{
	pgstromSharedState *ps_state = pts->ps_state;
	StringInfoData		buf;

	initStringInfo(&buf);
	if ((pts->xpu_task_flags & DEVTASK__USED_GPUDIRECT) == 0)
	{
		appendStringInfo(&buf, "disabled");
	}
	else
	{
		int		head = -1;
		int		base;
		uint64	count;

		appendStringInfo(&buf, "enabled (N=%d,", numGpuDevAttrs);
		base = buf.len;
		for (int k=0; k <= numGpuDevAttrs; k++)
		{
			if (k < numGpuDevAttrs &&
				(pts->optimal_gpus & (1UL<<k)) != 0)
			{
				if (head < 0)
				{
					appendStringInfo(&buf, "%s%d",
									 buf.len == base ? "GPU" : ",", k);
					head = k;
				}
			}
			else if (head >= 0)
			{
				if (head + 2 == k)
					appendStringInfo(&buf, ",%d", k-1);
				else if (head + 2 > k)
					appendStringInfo(&buf, "-%d", k-1);
				head = -1;
			}
		}
		if (es->analyze && ps_state)
		{
			base = buf.len;

			count = pg_atomic_read_u64(&ps_state->npages_buffer_read);
			if (count)
				appendStringInfo(&buf, "%sbuffer=%lu",
								 (buf.len > base ? ", " : "; "),
								 count / PAGES_PER_BLOCK);
			count = pg_atomic_read_u64(&ps_state->npages_vfs_read);
			if (count)
				appendStringInfo(&buf, "%svfs=%lu",
								 (buf.len > base ? ", " : "; "),
								 count / PAGES_PER_BLOCK);
			count = pg_atomic_read_u64(&ps_state->npages_direct_read);
			if (count)
				appendStringInfo(&buf, "%sdirect=%lu",
								 (buf.len > base ? ", " : "; "),
								 count / PAGES_PER_BLOCK);
			count = pg_atomic_read_u64(&ps_state->source_ntuples_raw);
			appendStringInfo(&buf, "%sntuples=%lu",
							 (buf.len > base ? ", " : "; "),
							 count);
		}
		appendStringInfo(&buf, ")");
	}
	if (!pgstrom_regression_test_mode)
		ExplainPropertyText("GPU-Direct SQL", buf.data, es);

	pfree(buf.data);
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
	pgstromSharedState *ps_state = pts->ps_state;
	pgstromPlanInfo	   *pp_info = pts->pp_info;
	CustomScan		   *cscan = (CustomScan *)node->ss.ps.plan;
	bool				verbose = (cscan->custom_plans != NIL);
	List			   *dcontext;
	StringInfoData		buf;
	ListCell		   *lc;
	const char		   *xpu_label;
	char				label[100];
	char			   *str;
	double				ntuples;
	uint64_t			stat_ntuples = 0;

	/* setup deparse context */
	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										ancestors);
	if ((pts->xpu_task_flags & DEVKIND__NVIDIA_GPU) != 0)
		xpu_label = "GPU";
	else if ((pts->xpu_task_flags & DEVKIND__NVIDIA_DPU) != 0)
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
		str = deparse_expression((Node *)tle->expr, dcontext, verbose, true);
		if (buf.len > 0)
			appendStringInfoString(&buf, ", ");
		appendStringInfoString(&buf, str);
	}
	snprintf(label, sizeof(label),
			 "%s Projection", xpu_label);
	ExplainPropertyText(label, buf.data, es);

	/* Pinned Inner Buffer */
	if ((pts->xpu_task_flags & (DEVTASK__PINNED_HASH_RESULTS |
								DEVTASK__PINNED_ROW_RESULTS)) != 0 &&
		(pts->xpu_task_flags & DEVTASK__PREAGG) == 0)
	{
		resetStringInfo(&buf);
		if (!es->analyze)
		{
			appendStringInfoString(&buf, "enabled");
		}
		else
		{
			uint64_t	final_nitems = pg_atomic_read_u64(&ps_state->final_nitems);
			uint64_t	final_usage  = pg_atomic_read_u64(&ps_state->final_usage);
			uint64_t	final_total  = pg_atomic_read_u64(&ps_state->final_total);
			
			appendStringInfo(&buf, "nitems: %lu, usage: %s, total: %s",
							 final_nitems,
							 format_bytesz(final_usage),
							 format_bytesz(final_total));
		}
		snprintf(label, sizeof(label),
				 "%s Pinned Buffer", xpu_label);
		ExplainPropertyText(label, buf.data, es);
	}

	/* xPU Scan Quals */
	if (ps_state)
		stat_ntuples = pg_atomic_read_u64(&ps_state->source_ntuples_in);
	if (pp_info->scan_quals)
	{
		List   *scan_quals = pp_info->scan_quals;
		Expr   *expr;

		resetStringInfo(&buf);
		if (list_length(scan_quals) > 1)
			expr = make_andclause(scan_quals);
		else
			expr = linitial(scan_quals);
		str = deparse_expression((Node *)expr, dcontext, verbose, true);
		appendStringInfoString(&buf, str);
		if (!es->analyze)
		{
			appendStringInfo(&buf, " [rows: %.0f -> %.0f]",
							 pp_info->scan_tuples,
							 pp_info->scan_nrows);
		}
		else
		{
			uint64_t		prev_ntuples = 0;

			if (ps_state)
				prev_ntuples = pg_atomic_read_u64(&ps_state->source_ntuples_raw);
			appendStringInfo(&buf, " [plan: %.0f -> %.0f, exec: %lu -> %lu]",
							 pp_info->scan_tuples,
							 pp_info->scan_nrows,
							 prev_ntuples,
							 stat_ntuples);
		}
		snprintf(label, sizeof(label), "%s Scan Quals", xpu_label);
		ExplainPropertyText(label, buf.data, es);
	}

	/* xPU JOIN */
	ntuples = pp_info->scan_nrows;
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];

		if (pp_inner->join_quals != NIL ||
			pp_inner->other_quals != NIL)
		{
			const char *join_label;

			resetStringInfo(&buf);
			foreach (lc, pp_inner->join_quals)
			{
				Node   *expr = lfirst(lc);

				str = deparse_expression(expr, dcontext, verbose, true);
				if (buf.len > 0)
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, str);
			}
			if (pp_inner->other_quals != NIL)
			{
				foreach (lc, pp_inner->other_quals)
				{
					Node   *expr = lfirst(lc);

					str = deparse_expression(expr, dcontext, verbose, true);
					if (buf.len > 0)
						appendStringInfoString(&buf, ", ");
					appendStringInfo(&buf, "[%s]", str);
				}
			}
			if (!es->analyze || !ps_state)
			{
				appendStringInfo(&buf, " ... [nrows: %.0f -> %.0f]",
								 ntuples, pp_inner->join_nrows);
			}
			else
			{
				uint64_t	next_ntuples;

				next_ntuples = pg_atomic_read_u64(&ps_state->inners[i].stats_join);
				appendStringInfo(&buf, " ... [plan: %.0f -> %.0f, exec: %lu -> %lu]",
								 ntuples, pp_inner->join_nrows,
								 stat_ntuples,
								 next_ntuples);
				stat_ntuples = next_ntuples;
			}
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

				str = deparse_expression(expr, dcontext, verbose, true);
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

				str = deparse_expression(expr, dcontext, verbose, true);
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

			str = deparse_expression((Node *)pp_inner->gist_clause,
									 dcontext, verbose, true);
			appendStringInfoString(&buf, str);
			if (idxname && colname)
				appendStringInfo(&buf, " on %s (%s)", idxname, colname);
			if (es->analyze && ps_state)
			{
				appendStringInfo(&buf, " [fetched: %lu]",
								 pg_atomic_read_u64(&ps_state->inners[i].stats_gist));
			}
			snprintf(label, sizeof(label),
					 "%s GiST Join [%d]", xpu_label, i+1);
			ExplainPropertyText(label, buf.data, es);
		}
	}
	if (pp_info->sibling_param_id >= 0)
		ExplainPropertyInteger("Inner Siblings-Id", NULL,
							   pp_info->sibling_param_id, es);
	/*
	 * CPU Fallback
	 */
	if (es->analyze && ps_state)
	{
		uint64_t   *fallback_nitems = alloca(sizeof(uint64_t) * (pts->num_rels + 1));
		bool		fallback_exists;

		fallback_nitems[0] = pg_atomic_read_u64(&ps_state->fallback_nitems);
		fallback_exists = (fallback_nitems[0] > 0);
		for (int i=1; i <= pts->num_rels; i++)
		{
			fallback_nitems[i] = pg_atomic_read_u64(&ps_state->inners[i-1].fallback_nitems);
			if (fallback_nitems[i] > 0)
				fallback_exists = true;
		}
		if (fallback_exists)
		{
			resetStringInfo(&buf);
			for (int i=0; i <= pts->num_rels; i++)
			{
				if (i > 0)
					appendStringInfo(&buf, ", ");
				appendStringInfo(&buf, "depth[%d]=%lu", i, fallback_nitems[i]);
			}
			ExplainPropertyText("Fallback-stats", buf.data, es);
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
		pgstromGpuDirectExplain(pts, es, dcontext);
	}
	else if (pts->gcache_desc)
	{
		/* GPU-Cache */
		pgstromGpuCacheExplain(pts, es, dcontext);
	}
	else if ((pts->xpu_task_flags & DEVTASK__USED_GPUDIRECT) != 0)
	{
		/* GPU-Direct */
		pgstromGpuDirectExplain(pts, es, dcontext);
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
		pgstrom_explain_kvars_slot(&pts->css, es, dcontext);
		pgstrom_explain_kvecs_buffer(&pts->css, es, dcontext);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"LoadVars OpCode",
								pp_info->kexp_load_vars_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"MoveVars OpCode",
								pp_info->kexp_move_vars_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Scan Quals OpCode",
								pp_info->kexp_scan_quals);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Join Quals OpCode",
								pp_info->kexp_join_quals_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Join HashValue OpCode",
								pp_info->kexp_hash_keys_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"GiST-Index Join OpCode",
								pp_info->kexp_gist_evals_packed);
		pgstrom_explain_xpucode(&pts->css, es, dcontext,
								"Projection OpCode",
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
		pgstrom_explain_fallback_desc(pts, es, dcontext);
		if (pp_info->groupby_prepfn_bufsz > 0)
			ExplainPropertyInteger("Partial Function BufSz", NULL,
								   pp_info->groupby_prepfn_bufsz, es);
		if (pp_info->cuda_stack_size > 0)
			ExplainPropertyInteger("CUDA Stack Size", NULL,
								   pp_info->cuda_stack_size, es);
	}
	pfree(buf.data);
}

/*
 * __xpuClientOpenSession
 */
void
__xpuClientOpenSession(pgstromTaskState *pts,
					   const XpuCommand *session,
					   pgsocket sockfd)
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
		elog(ERROR, "Bug? OpenSession response is missing");
	if (resp->tag != XpuCommandTag__Success)
		elog(ERROR, "OpenSession failed - %s (%s:%d %s)",
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
	dlist_init(&xpu_connections_list);
	RegisterResourceReleaseCallback(xpuclientCleanupConnections, NULL);
}
