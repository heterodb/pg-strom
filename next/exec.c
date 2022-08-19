/*
 * exec.c
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

/* see xact.c */
extern int				nParallelCurrentXids;
extern TransactionId   *ParallelCurrentXids;

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

const kern_session_info *
pgstrom_build_session_info(PlanState *ps,
						   List *used_params,
						   uint32_t kcxt_extra_bufsz,
						   uint32_t kcxt_kvars_nslots,
						   const bytea *xpucode_scan_quals,
						   const bytea *xpucode_scan_projs)
{
	ExprContext	   *econtext = ps->ps_ExprContext;
	ParamListInfo	param_info = econtext->ecxt_param_list_info;
	uint32_t		nparams = (param_info ? param_info->numParams : 0);
	uint32_t		offset;
	StringInfoData	buf;
	kern_session_info *session;
	static uint32_t	__magic = XpuCommandMagicNumber;

	offset = offsetof(kern_session_info, poffset[nparams]);
	session = alloca(offset);
	memset(session, 0, offset);

	initStringInfo(&buf);
	enlargeStringInfo(&buf, offset);
	memset(buf.data, 0, offset);
	buf.len = offset;

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
				offset -= offsetof(XpuCommand, data);
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
	__appendBinaryStringInfo(&buf, &__magic, sizeof(uint32_t));

	session->tag = XpuCommandTag__OpenSession;
	session->length = buf.len;
	memcpy(buf.data, session, offsetof(kern_session_info, poffset[nparams]));

	return (const kern_session_info *)buf.data;
}

/*
 * pgstrom_receive_xpu_command
 */
int
pgstrom_receive_xpu_command(pgsocket sockfd,
							void *(*alloc_f)(void *priv, size_t sz),
							void  (*attach_f)(void *priv, XpuCommand *xcmd),
							void *priv,
							const char *errmsg_label,
							char *errmsg_buffer,
							size_t errmsg_bufsz)
{
	char		buffer_local[2 * BLCKSZ];
	char	   *buffer;
	size_t		bufsz, offset;
	ssize_t		nbytes;
	int			recv_flags;
	int			count = 0;
	XpuCommand *curr = NULL;

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
			snprintf(errmsg_buffer, errmsg_bufsz,
					 "%s: failed on recv(%d, %p, %ld, %d): %m",
					 errmsg_label,
					 sockfd, buffer + offset, bufsz - offset, recv_flags);
			return -1;
		}
		else if (nbytes == 0)
		{
			/* end of the stream */
			if (curr || offset > 0)
			{
				snprintf(errmsg_buffer, errmsg_bufsz,
						 "%s: connection closed during XpuCommand read",
						 errmsg_label);
				return -1;
			}
			return count;
		}

		offset += nbytes;
		if (!curr)
		{
			XpuCommand *temp, *xcmd;
		next:
			if (offset < offsetof(XpuCommand, data))
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
				XPU_COMMAND_SANITY_CHECK(temp);
				xcmd = alloc_f(priv, temp->length);
				if (!xcmd)
				{
					snprintf(errmsg_buffer, errmsg_bufsz,
							 "%s: out of memory (sz=%u): %m",
							 errmsg_label, temp->length);
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
					snprintf(errmsg_buffer, errmsg_bufsz,
							 "%s: out of memory (sz=%u): %m",
							 errmsg_label, temp->length);
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
			assert(offset == curr->length);
			XPU_COMMAND_SANITY_CHECK(curr);
			attach_f(priv, curr);
			count++;
			goto restart;
		}
	}
	snprintf(errmsg_buffer, errmsg_bufsz,
			 "%s: Bug? should not break this loop",
			 errmsg_label);
	return -1;
}
