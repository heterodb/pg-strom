/*
 * datastore.c
 *
 * Routines to manage data store; row-store, column-store, toast-buffer,
 * and param-buffer.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "pg_strom.h"

/*
 * calculate_param_buffer_size
 *
 * It calculates the total length of param buffer that can hold all
 * the Const / Param values.
 */
static Size
calculate_param_buffer_size(List *used_params, ExprContext *econtext)
{
	ListCell   *cell;
	Size		length;

	/* length of offset tables */
	length = STROMALIGN(sizeof(cl_uint) * list_length(used_params));

	/* add length of every values */
	foreach (cell, used_params)
	{
		Node   *node = lfirst(cell);

		if (IsA(node, Const))
		{
			Const  *con = (Const *) node;

			if (!con->constisnull)
			{
				if (con->constlen > 0)
					length += STROMALIGN(con->constlen);
				else
					length += STROMALIGN(VARSIZE(con->constvalue));
			}
		}
		else if (IsA(node, Param))
		{
			ParamListInfo param_info = econtext->ecxt_param_list_info;
			Param  *param = (Param *) node;

			if (param_info &&
				param->paramid > 0 && param->paramid <= param_info->numParams)
			{
				ParamExternData	*prm = &param_info->params[param->paramid - 1];

				/* give hook a chance in case parameter is dynamic */
				if (!OidIsValid(prm->ptype) && param_info->paramFetch != NULL)
					(*param_info->paramFetch) (param_info, param->paramid);
				if (!OidIsValid(prm->ptype))
					continue;

				/* safety check in case hook did something unexpected */
				if (prm->ptype != param->paramtype)
					ereport(ERROR,
							(errcode(ERRCODE_DATATYPE_MISMATCH),
							 errmsg("type of parameter %d (%s) does not match that when preparing the plan (%s)",
									param->paramid,
									format_type_be(prm->ptype),
									format_type_be(param->paramtype))));
				if (!prm->isnull)
				{
					int		typlen = get_typlen(prm->ptype);

					if (typlen == 0)
						elog(ERROR, "cache lookup failed for type %u",
							 prm->ptype);
					if (typlen > 0)
						length += STROMALIGN(typlen);
					else
						length += STROMALIGN(VARSIZE(prm->value));
				}
			}
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(node));
	}
	return length;
}

/*
 * pgstrom_create_param_buffer
 *
 * It construct a param-buffer on the shared memory segment, according to
 * the supplied Const/Param list. Its initial reference counter is 1, so
 * this buffer can be released using pgstrom_put_param_buffer().
 */
pgstrom_parambuf *
pgstrom_create_param_buffer(shmem_context *shm_context,
							List *used_params,
							ExprContext *econtext)
{
	pgstrom_parambuf *parambuf;
	ListCell   *cell;
	Size		length;
	Size		offset;
	int			index;

	/* no constant/params; an obvious case */
	if (used_params == NIL)
		return NULL;
	length = calculate_param_buffer_size(used_params, econtext);

	parambuf = pgstrom_shmem_alloc(shm_context, length);
	if (!parambuf)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of shared memory")));

	parambuf->mtag.type = StromMsg_ParamBuf;
	parambuf->mtag.length = length;
	SpinLockInit(&parambuf->lock);
	parambuf->refcnt = 1;

	index = 0;
	offset = STROMALIGN(sizeof(cl_uint) * list_length(used_params));
	foreach (cell, used_params)
	{
		Node   *node = lfirst(cell);

		if (IsA(node, Const))
		{
			Const  *con = (Const *) node;

			if (!con->constisnull)
				parambuf->kern.poffset[index] = 0;	/* null */
			else
			{
				parambuf->kern.poffset[index] = offset;
				if (con->constlen > 0)
				{
					memcpy((char *)parambuf + offset,
						   &con->constvalue,
						   con->constlen);
					offset += STROMALIGN(con->constlen);
				}
				else
				{
					memcpy((char *)parambuf + offset,
						   DatumGetPointer(con->constvalue),
						   VARSIZE(con->constvalue));
					offset += STROMALIGN(VARSIZE(con->constvalue));
				}
			}
		}
		else if (IsA(node, Param))
		{
			ParamListInfo param_info = econtext->ecxt_param_list_info;
			Param  *param = (Param *) node;

			if (param_info &&
                param->paramid > 0 && param->paramid <= param_info->numParams)
			{
				ParamExternData *prm = &param_info->params[param->paramid - 1];

				/* dynamic param is already set on the 1st stage */
				if (OidIsValid(prm->ptype))
				{
					int		typlen = get_typlen(prm->ptype);

					if (typlen == 0)
						elog(ERROR, "cache lookup failed for type %u",
							 prm->ptype);

					parambuf->kern.poffset[index] = offset;
					if (typlen > 0)
					{
						memcpy((char *)parambuf + offset,
							   &prm->value,
							   typlen);
						offset += STROMALIGN(typlen);
					}
					else
					{
						memcpy((char *)parambuf + offset,
							   DatumGetPointer(prm->value),
							   VARSIZE(prm->value));
						offset += STROMALIGN(VARSIZE(prm->value));
					}
				}
				else
					parambuf->kern.poffset[index] = 0;	/* null */
			}
			else
				parambuf->kern.poffset[index] = 0;	/* null */
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(node));
		Assert(offset <= length);
		index++;
	}
	Assert(offset == length);
	parambuf->kern.nparams = index;

	return parambuf;
}

/*
 * Increment reference counter
 */
void
pgstrom_get_param_buffer(pgstrom_parambuf *parambuf)
{
	SpinLockAcquire(&parambuf->lock);
	parambuf->refcnt++;
	SpinLockFree(&parambuf->lock);
}

/*
 * Decrement reference counter and release param-buffer if nobody references.
 */
void
pgstrom_put_param_buffer(pgstrom_parambuf *parambuf)
{
	bool	do_release = false;

	SpinLockAcquire(&parambuf->lock);
	if (--parambuf->refcnt == 0)
		do_release = true;
	SpinLockFree(&parambuf->lock);
	if (do_release)
		pgstrom_shmem_free(parambuf);
}
