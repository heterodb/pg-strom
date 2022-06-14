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
static Datum			__zero = 0;

static void
__build_session_xact_id_vector(StringInfo buf)
{
	uint32_t	sz = VARHDRSZ + sizeof(TransactionId) * nParallelCurrentXids;
	uint32_t	base;
	uint32_t	temp;

	if (buf->len != MAXALIGN(buf->len))
	{
		appendBinaryStringInfo(buf, (char *)&__zero,
							   MAXALIGN(buf->len) - buf->len);
	}
	base = buf->len;

	if (nParallelCurrentXids > 0)
	{
		SET_VARSIZE(&temp, sz);
		appendBinaryStringInfo(buf, (char *)&temp, VARHDRSZ);
		appendBinaryStringInfo(buf, (char *)ParallelCurrentXids,
							   sizeof(TransactionId) * nParallelCurrentXids);
		((kern_session_info *)buf->data)->xact_id_array = base;
	}
}

static void
__build_session_timezone(StringInfo buf)
{
	uint32_t	base;

	if (!session_timezone)
		return;
	if (buf->len != MAXALIGN(buf->len))
	{
		appendBinaryStringInfo(buf, (char *)&__zero,
							   MAXALIGN(buf->len) - buf->len);
	}
	base = buf->len;
	appendBinaryStringInfo(buf, (char *)session_timezone, sizeof(struct pg_tz));
	((kern_session_info *)buf->data)->session_timezone = base;
}

static void
__build_session_encode(StringInfo buf)
{
	xpu_encode_info encode;
	uint32_t	base;

	if (buf->len != MAXALIGN(buf->len))
	{
		appendBinaryStringInfo(buf, (char *)&__zero,
							   MAXALIGN(buf->len) - buf->len);
	}
	base = buf->len;
	
	strncpy(encode.encname,
			GetDatabaseEncodingName(),
			sizeof(encode.encname));
	encode.enc_maxlen = pg_database_encoding_max_length();
	encode.enc_mblen = NULL;
	appendBinaryStringInfo(buf, (char *)&encode, sizeof(xpu_encode_info));

	((kern_session_info *)buf->data)->session_encode = base;
}

kern_session_info *
pgstrom_build_session_info(PlanState *ps, List *used_params,
						   uint32_t num_cached_kvars,
						   uint32_t kcxt_extra_bufsz)
{
	ExprContext	   *econtext = ps->ps_ExprContext;
	ParamListInfo	param_info = econtext->ecxt_param_list_info;
	uint32_t		nparams = (param_info ? param_info->numParams : 0);
	uint32_t		offset;
	StringInfoData	buf;
	kern_session_info *session;

	initStringInfo(&buf);
	offset = MAXALIGN(offsetof(kern_session_info, poffset[nparams]));
	enlargeStringInfo(&buf, offset);
	memset(buf.data, 0, offset);
	buf.len = offset;

	/* Put executor parameters */
	if (param_info)
	{
		ListCell   *lc;

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
			session = (kern_session_info *)buf.data;
			if (param_isnull)
				session->poffset[param->paramid] = 0;
			else
			{
				int16	typlen;
				bool	typbyval;

				if (buf.len != MAXALIGN(buf.len))
				{
					appendBinaryStringInfo(&buf, (char *)&__zero,
										   MAXALIGN(buf.len) - buf.len);
				}
				session->poffset[param->paramid] = buf.len;
				get_typlenbyval(param->paramtype, &typlen, &typbyval);
				if (typbyval)
				{
					appendBinaryStringInfo(&buf,
										   (char *)&param_value,
										   typlen);
				}
				else if (typlen > 0)
				{
					appendBinaryStringInfo(&buf,
										   DatumGetPointer(param_value),
										   typlen);
				}
				else if (typlen == -1)
				{
					struct varlena *temp = PG_DETOAST_DATUM(param_value);

					appendBinaryStringInfo(&buf,
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
		}
	}
	/* other database session information */
	session = (kern_session_info *)buf.data;
	session->num_cached_kvars = num_cached_kvars;
	session->kcxt_extra_bufsz = kcxt_extra_bufsz;
	session->xactStartTimestamp = GetCurrentTransactionStartTimestamp();
	__build_session_xact_id_vector(&buf);
	__build_session_timezone(&buf);
	__build_session_encode(&buf);

	session = (kern_session_info *)buf.data;
	session->length = buf.len;

	return session;
}
