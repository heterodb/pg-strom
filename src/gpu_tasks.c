/*
 * gputasks.c
 *
 * Routines to manage GpuTaskState/GpuTask state machine.
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "pg_strom.h"

/*
 * construct_kern_parambuf
 *
 * It construct a kernel parameter buffer to deliver Const/Param nodes.
 */
static kern_parambuf *
construct_kern_parambuf(List *used_params, ExprContext *econtext,
						List *custom_scan_tlist)
{
	StringInfoData	str;
	kern_parambuf  *kparams;
	char		padding[STROMALIGN_LEN];
	ListCell   *cell;
	Size		offset;
	int			index = 0;
	int			nparams = list_length(used_params);

	memset(padding, 0, sizeof(padding));

	/* seek to the head of variable length field */
	offset = STROMALIGN(offsetof(kern_parambuf, poffset[nparams]));
	initStringInfo(&str);
	enlargeStringInfo(&str, offset);
	memset(str.data, 0, offset);
	str.len = offset;
	/* walks on the Para/Const list */
	foreach (cell, used_params)
	{
		Node   *node = lfirst(cell);
		bool	nested_custom_scan_tlist = false;

	retry_custom_scan_tlist:
		if (IsA(node, Const))
		{
			Const  *con = (Const *) node;

			kparams = (kern_parambuf *)str.data;
			if (con->constisnull)
				kparams->poffset[index] = 0;	/* null */
			else if (con->constbyval)
			{
				Assert(con->constlen > 0);
				kparams->poffset[index] = str.len;
				appendBinaryStringInfo(&str,
									   (char *)&con->constvalue,
									   con->constlen);
			}
			else
			{
				kparams->poffset[index] = str.len;
				if (con->constlen > 0)
					appendBinaryStringInfo(&str,
										   DatumGetPointer(con->constvalue),
										   con->constlen);
				else
					appendBinaryStringInfo(&str,
                                           DatumGetPointer(con->constvalue),
                                           VARSIZE(con->constvalue));
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

				kparams = (kern_parambuf *)str.data;
				if (!OidIsValid(prm->ptype))
				{
					elog(INFO, "debug: Param has no particular data type");
					kparams->poffset[index++] = 0;	/* null */
					continue;
				}
				/* safety check in case hook did something unexpected */
				if (prm->ptype != param->paramtype)
					ereport(ERROR,
							(errcode(ERRCODE_DATATYPE_MISMATCH),
							 errmsg("type of parameter %d (%s) does not match "
									"that when preparing the plan (%s)",
									param->paramid,
									format_type_be(prm->ptype),
									format_type_be(param->paramtype))));
				if (prm->isnull)
					kparams->poffset[index] = 0;	/* null */
				else
				{
					int16	typlen;
					bool	typbyval;

					get_typlenbyval(prm->ptype, &typlen, &typbyval);
					if (typbyval)
					{
						appendBinaryStringInfo(&str,
											   (char *)&prm->value,
											   typlen);
					}
					else if (typlen > 0)
					{
						appendBinaryStringInfo(&str,
											   DatumGetPointer(prm->value),
											   typlen);
					}
					else
					{
						appendBinaryStringInfo(&str,
											   DatumGetPointer(prm->value),
											   VARSIZE(prm->value));
					}
				}
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_UNDEFINED_OBJECT),
						 errmsg("no value found for parameter %d",
								param->paramid)));
			}
		}
		else if (!nested_custom_scan_tlist &&
				 IsA(node, Var) &&
				 custom_scan_tlist != NIL &&
				 ((Var *)node)->varno == INDEX_VAR &&
				 ((Var *)node)->varattno <= list_length(custom_scan_tlist))
		{
			/*
			 * NOTE: setrefs.c often replaces the Const/Param expressions on
			 * the @used_params, if custom_scan_tlist has an identical TLE.
			 * So, if expression is a references to the custom_scan_tlist,
			 * we try to solve the underlying value, then retry.
			 */
			AttrNumber		varattno = ((Var *)node)->varattno;
			TargetEntry	   *tle = list_nth(custom_scan_tlist, varattno - 1);

			node = (Node *)tle->expr;

			nested_custom_scan_tlist = true;
			goto retry_custom_scan_tlist;
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(node));

		/* alignment */
		if (STROMALIGN(str.len) != str.len)
			appendBinaryStringInfo(&str, padding,
								   STROMALIGN(str.len) - str.len);
		index++;
	}
	Assert(STROMALIGN(str.len) == str.len);
	kparams = (kern_parambuf *)str.data;
	kparams->hostptr = (hostptr_t) &kparams->hostptr;
	kparams->xactStartTimestamp = GetCurrentTransactionStartTimestamp();
	kparams->length = str.len;
	kparams->nparams = nparams;

	return kparams;
}

/*
 * pgstromInitGpuTaskState
 */
void
pgstromInitGpuTaskState(GpuTaskState *gts,
						GpuContext *gcontext,
						GpuTaskKind task_kind,
						List *ccache_refs_list,
						List *used_params,
						EState *estate)
{
	Relation		relation = gts->css.ss.ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(relation);
	ExprContext	   *econtext = gts->css.ss.ps.ps_ExprContext;
	CustomScan	   *cscan = (CustomScan *)(gts->css.ss.ps.plan);
	Relids			ccache_refs = NULL;
	ListCell	   *lc;

	Assert(gts->gcontext == gcontext);
	gts->task_kind = task_kind;
	gts->program_id = INVALID_PROGRAM_ID;	/* to be set later */
	gts->kern_params = construct_kern_parambuf(used_params, econtext,
											   cscan->custom_scan_tlist);
	if (relation && RelationCanUseColumnarCache(relation))
	{
		foreach (lc, ccache_refs_list)
		{
			int		i, anum = lfirst_int(lc);

			if (anum == InvalidAttrNumber)
			{
				for (i=0; i < tupdesc->natts; i++)
				{
					Form_pg_attribute	attr = tupdesc->attrs[i];

					if (attr->attisdropped)
						continue;
					anum = attr->attnum - FirstLowInvalidHeapAttributeNumber;
					ccache_refs = bms_add_member(ccache_refs, anum);
				}
			}
			else if (anum < 0)
			{
				anum += (tupdesc->natts -
						 (1 + FirstLowInvalidHeapAttributeNumber));
				ccache_refs = bms_add_member(ccache_refs, anum);
			}
			else
			{
				ccache_refs = bms_add_member(ccache_refs, anum-1);
			}
		}
		/*
		 * Non-NULL ccache_refs also means the relation can have columnar-
		 * cache, but no columns are referenced in the query like:
		 *   SELECT count(*) FROM tbl;
		 */
		if (!ccache_refs)
		{
			ccache_refs = palloc0(offsetof(Bitmapset, words[1]));
			ccache_refs->nwords = 1;
		}
	}
#if 0
	if (!ccache_refs)
		elog(INFO, "ccache_refs = NULL");
	else
	{
		int		i, j;

		for (i = bms_first_member(ccache_refs);
			 i >= 0;
			 i = bms_next_member(ccache_refs, i))
		{
			j = i + FirstLowInvalidHeapAttributeNumber;
			elog(INFO, "ccache_refs: [%s].[%s]",
				 RelationGetRelationName(relation),
				 get_attname(RelationGetRelid(relation), j));
		}
	}
#endif
	gts->ccache_refs = ccache_refs;
	gts->scan_done = false;

	gts->outer_bulk_exec = false;
	InstrInit(&gts->outer_instrument, estate->es_instrument);
	gts->scan_overflow = NULL;
	gts->outer_pds_suspend = NULL;
	gts->nvme_sstate = NULL;

	/*
	 * NOTE: initialization of HeapScanDesc was moved to the first try of
	 * ExecGpuXXX() call to support CPU parallel. A local HeapScanDesc shall
	 * be setup only when it is not responsible to partial read.
	 */

	/* callbacks shall be set by the caller */
	dlist_init(&gts->ready_tasks);
	gts->num_ready_tasks = 0;

	/* co-operation with CPU parallel (setup by DSM init handler) */
	gts->pcxt = NULL;
}

/*
 * fetch_next_gputask
 */
GpuTask *
fetch_next_gputask(GpuTaskState *gts)
{
	GpuContext	   *gcontext = gts->gcontext;
	GpuTask		   *gtask;
	dlist_node	   *dnode;
	cl_int			local_num_running_tasks;
	cl_int			global_num_running_tasks;
	cl_int			ev;

	CHECK_FOR_GPUCONTEXT(gcontext);

	pthreadMutexLock(gcontext->mutex);
	while (!gts->scan_done)
	{
		ResetLatch(MyLatch);
		local_num_running_tasks = (gts->num_ready_tasks +
								   gts->num_running_tasks);
		global_num_running_tasks =
			pg_atomic_read_u32(gcontext->global_num_running_tasks);
		if ((local_num_running_tasks < local_max_async_tasks &&
			 global_num_running_tasks < global_max_async_tasks) ||
			(dlist_is_empty(&gts->ready_tasks) &&
			 gts->num_running_tasks == 0))
		{
			pthreadMutexUnlock(gcontext->mutex);
			gtask = gts->cb_next_task(gts);
			pthreadMutexLock(gcontext->mutex);
			if (!gtask)
			{
				gts->scan_done = true;
				break;
			}
			dlist_push_tail(&gcontext->pending_tasks, &gtask->chain);
			gts->num_running_tasks++;
			pg_atomic_add_fetch_u32(gcontext->global_num_running_tasks, 1);
			pthreadCondSignal(gcontext->cond);
		}
		else if (!dlist_is_empty(&gts->ready_tasks))
		{
			/*
			 * Even though we touched either local or global limitation of
			 * the number of concurrent tasks, GTS already has ready tasks,
			 * so pick them up instead of wait.
			 */
			pthreadMutexUnlock(gcontext->mutex);
			goto pickup_gputask;
		}
		else if (gts->num_running_tasks > 0)
		{
			/*
			 * Even though a few GpuTasks are running, but nobody gets
			 * completed yet. Try to wait for completion to 
			 */
			pthreadMutexUnlock(gcontext->mutex);

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   500L
#if PG_VERSION_NUM >= 100000
						   ,PG_WAIT_EXTENSION
#endif
				);
			if (ev & WL_POSTMASTER_DEATH)
				ereport(FATAL,
						(errcode(ERRCODE_ADMIN_SHUTDOWN),
						 errmsg("Unexpected Postmaster dead")));
			CHECK_FOR_GPUCONTEXT(gcontext);

			pthreadMutexLock(gcontext->mutex);
		}
		else
		{
			pthreadMutexUnlock(gcontext->mutex);
			/*
			 * Sadly, we touched a threshold. Taks a short break.
			 */
			pg_usleep(20000L);	/* wait for 20msec */

			CHECK_FOR_GPUCONTEXT(gcontext);
			pthreadMutexLock(gcontext->mutex);
		}
	}
	pthreadMutexUnlock(gcontext->mutex);

	/*
	 * Once we exit the above loop, either a completed task was returned,
	 * or relation scan has already done thus wait for synchronously.
	 */
	Assert(gts->scan_done);
	pthreadMutexLock(gcontext->mutex);
retry:
	ResetLatch(MyLatch);
	while (dlist_is_empty(&gts->ready_tasks))
	{
		Assert(gts->num_running_tasks >= 0);
		if (gts->num_running_tasks == 0)
		{
			pthreadMutexUnlock(gcontext->mutex);

			CHECK_FOR_GPUCONTEXT(gcontext);

			if (gts->cb_terminator_task)
			{
				cl_bool		is_ready = false;

				gtask = gts->cb_terminator_task(gts, &is_ready);
				pthreadMutexLock(gcontext->mutex);
				if (gtask)
				{
					if (is_ready)
					{
						dlist_push_tail(&gts->ready_tasks,
										&gtask->chain);
						gts->num_ready_tasks++;
					}
					else
					{
						dlist_push_tail(&gcontext->pending_tasks,
										&gtask->chain);
						gts->num_running_tasks++;
						pg_atomic_add_fetch_u32(gcontext->global_num_running_tasks, 1);
						pthreadCondSignal(gcontext->cond);
					}
					goto retry;
				}
				pthreadMutexUnlock(gcontext->mutex);
			}
			return NULL;
		}
		pthreadMutexUnlock(gcontext->mutex);

		CHECK_FOR_GPUCONTEXT(gcontext);

		ev = WaitLatch(MyLatch,
					   WL_LATCH_SET |
					   WL_TIMEOUT |
					   WL_POSTMASTER_DEATH,
					   500L
#if PG_VERSION_NUM >= 100000
					   ,PG_WAIT_EXTENSION
#endif
			);
		if (ev & WL_POSTMASTER_DEATH)
			ereport(FATAL,
					(errcode(ERRCODE_ADMIN_SHUTDOWN),
					 errmsg("Unexpected Postmaster dead")));

		pthreadMutexLock(gcontext->mutex);
		ResetLatch(MyLatch);
	}
	pthreadMutexUnlock(gcontext->mutex);
pickup_gputask:
	/* OK, pick up GpuTask from the head */
	Assert(gts->num_ready_tasks > 0);
	dnode = dlist_pop_head_node(&gts->ready_tasks);
	gtask = dlist_container(GpuTask, chain, dnode);
	gts->num_ready_tasks--;

	return gtask;
}

/*
 * pgstromExecGpuTaskState
 */
TupleTableSlot *
pgstromExecGpuTaskState(GpuTaskState *gts)
{
	TupleTableSlot *slot = gts->css.ss.ss_ScanTupleSlot;

	while (!gts->curr_task || !(slot = gts->cb_next_tuple(gts)))
	{
		GpuTask	   *gtask = gts->curr_task;

		/* release the current GpuTask object that was already scanned */
		if (gtask)
		{
			gts->cb_release_task(gtask);
			gts->curr_task = NULL;
			gts->curr_index = 0;
			gts->curr_lp_index = 0;
		}
		/* reload next chunk to be scanned */
		gtask = fetch_next_gputask(gts);
		if (!gtask)
			break;
		if (gtask->cpu_fallback)
			gts->num_cpu_fallbacks++;
		gts->curr_task = gtask;
		gts->curr_index = 0;
		gts->curr_lp_index = 0;
		/* notify a new task is assigned */
		if (gts->cb_switch_task)
			gts->cb_switch_task(gts, gtask);
	}
	return slot;
}

/*
 * pgstromBulkExecGpuTaskState
 */
pgstrom_data_store *
pgstromBulkExecGpuTaskState(GpuTaskState *gts, size_t chunk_size)
{
	pgstrom_data_store *pds_dst = NULL;
	TupleTableSlot	   *slot;

	/* GTS should not have neither host qualifier nor projection */
	Assert(!gts->css.ss.ps.qual);
	Assert(!gts->css.ss.ps.ps_ProjInfo);

	do {
		GpuTask	   *gtask = gts->curr_task;

		/* Reload next GpuTask to be scanned, if needed */
		if (!gtask)
		{
			gtask = fetch_next_gputask(gts);
			if (!gtask)
				break;	/* end of the scan */
			gts->curr_task = gtask;
			gts->curr_index = 0;
			gts->curr_lp_index = 0;
			if (gts->cb_switch_task)
				gts->cb_switch_task(gts, gtask);
		}
		Assert(gtask != NULL);

		while ((slot = gts->cb_next_tuple(gts)) != NULL)
		{
			/*
			 * Creation of the destination store on demand.
			 */
			if (!pds_dst)
			{
				pds_dst = PDS_create_row(gts->gcontext,
										 slot->tts_tupleDescriptor,
										 chunk_size);
			}

			/*
			 * Move rows from the source data-store to the destination store
			 * until:
			 *  The destination store still has space.
			 *  The source store still has unread rows.
			 */
			if (!PDS_insert_tuple(pds_dst, slot))
			{
				/* Rewind the source PDS, if destination gets filled up */
				Assert(gts->curr_index > 0 && gts->curr_lp_index == 0);
				gts->curr_index--;

				/*
				 * At least one tuple can be stored, unless the supplied
				 * chunk_size is not too small.
				 */
				if (pds_dst->kds.nitems == 0)
				{
					HeapTuple	tuple = ExecFetchSlotTuple(slot);
					elog(ERROR,
						 "Bug? Too short chunk_size (%zu) for tuple (len=%u)",
						 chunk_size, tuple->t_len);
				}
				return pds_dst;
			}
		}

		/*
		 * All the rows in pds_src are already fetched,
		 * so current GpuTask shall be detached.
		 */
		gts->cb_release_task(gtask);
		gts->curr_task = NULL;
		gts->curr_index = 0;
		gts->curr_lp_index = 0;
	} while (true);

	return pds_dst;
}

/*
 * pgstromRescanGpuTaskState
 */
void
pgstromRescanGpuTaskState(GpuTaskState *gts)
{
	/*
	 * release all the unprocessed tasks
	 */
	while (!dlist_is_empty(&gts->ready_tasks))
	{
		dlist_node *dnode = dlist_pop_head_node(&gts->ready_tasks);
		GpuTask	   *gtask = dlist_container(GpuTask, chain, dnode);
		gts->num_ready_tasks--;
		Assert(gts->num_ready_tasks >= 0);
		gts->cb_release_task(gtask);
	}
}

/*
 * pgstromReleaseGpuTaskState
 */
void
pgstromReleaseGpuTaskState(GpuTaskState *gts)
{
	/*
	 * release any unprocessed tasks
	 */
	while (!dlist_is_empty(&gts->ready_tasks))
	{
		dlist_node *dnode = dlist_pop_head_node(&gts->ready_tasks);
		GpuTask	   *gtask = dlist_container(GpuTask, chain, dnode);
		gts->num_ready_tasks--;
		Assert(gts->num_ready_tasks >= 0);
		gts->cb_release_task(gtask);
	}
	/* cleanup per-query PDS-scan state, if any */
	PDS_end_heapscan_state(gts);
	InstrEndLoop(&gts->outer_instrument);
	/* release scan-desc if any */
	if (gts->css.ss.ss_currentScanDesc)
		heap_endscan(gts->css.ss.ss_currentScanDesc);
	/* unreference CUDA program */
	if (gts->program_id != INVALID_PROGRAM_ID)
		pgstrom_put_cuda_program(gts->gcontext, gts->program_id);
	/* unreference GpuContext */
	PutGpuContext(gts->gcontext);
}

/*
 * pgstromExplainGpuTaskState
 */
void
pgstromExplainGpuTaskState(GpuTaskState *gts, ExplainState *es)
{
	/* outer-bulk-exec? */
	if (gts->outer_bulk_exec)
		ExplainPropertyText("Outer Bulk Exec", "enabled", es);
	else if (es->format != EXPLAIN_FORMAT_TEXT)
		ExplainPropertyText("Outer Bulk Exec", "disabled", es);

	/* NVMe-Strom support */
	if (gts->nvme_sstate ||
		(!es->analyze &&
		 gts->css.ss.ss_currentRelation &&
		 RelationWillUseNvmeStrom(gts->css.ss.ss_currentRelation, NULL)))
		ExplainPropertyText("NVMe-Strom", "enabled", es);
	else if (es->format != EXPLAIN_FORMAT_TEXT)
		ExplainPropertyText("NVMe-Strom", "disabled", es);

	/* Number of CPU fallbacks, if any */
	if (es->analyze && gts->num_cpu_fallbacks > 0)
		ExplainPropertyLong("CPU fallbacks", gts->num_cpu_fallbacks, es);

	/* Source path of the GPU kernel */
	if (es->verbose &&
		gts->program_id != INVALID_PROGRAM_ID &&
		pgstrom_debug_kernel_source)
	{
		const char *cuda_source = pgstrom_cuda_source_file(gts->program_id);
		const char *cuda_binary = pgstrom_cuda_binary_file(gts->program_id);

		if (cuda_source)
			ExplainPropertyText("Kernel Source", cuda_source, es);
		if (cuda_binary)
			ExplainPropertyText("Kernel Binary", cuda_binary, es);
	}
}

/*
 * pgstromInitGpuTask
 */
void
pgstromInitGpuTask(GpuTaskState *gts, GpuTask *gtask)
{
	gtask->task_kind    = gts->task_kind;
	gtask->program_id   = gts->program_id;
	gtask->gts          = gts;
	gtask->cpu_fallback = false;
}


/* ------------------------------------------------------------ *
 *   Misc routines to support EXPLAIN command
 * ------------------------------------------------------------ */
/*
 * pgstromExplainOuterScan
 */
void
pgstromExplainOuterScan(GpuTaskState *gts,
						List *deparse_context,
						List *ancestors,
						ExplainState *es,
						Expr *outer_quals,
						Cost outer_startup_cost,
						Cost outer_total_cost,
						double outer_plan_rows,
						int outer_plan_width)
{
	Plan		   *plannode = gts->css.ss.ps.plan;
	Index			scanrelid = ((Scan *) plannode)->scanrelid;
	Instrumentation *instrument = &gts->outer_instrument;
	RangeTblEntry  *rte;
	const char	   *refname;
	const char	   *relname;
	const char	   *nspname = NULL;
	StringInfoData	str;

	/* Does this GpuTaskState has outer simple scan? */
	if (scanrelid == 0)
		return;

	/*
	 * See the logic in ExplainTargetRel()
	 */
	rte = rt_fetch(scanrelid, es->rtable);
	Assert(rte->rtekind == RTE_RELATION);
	refname = (char *) list_nth(es->rtable_names, scanrelid - 1);
	if (!refname)
		refname = rte->eref->aliasname;
	relname = get_rel_name(rte->relid);
	if (es->verbose)
		nspname = get_namespace_name(get_rel_namespace(rte->relid));

	initStringInfo(&str);
	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		if (nspname != NULL)
			appendStringInfo(&str, "%s.%s",
							 quote_identifier(nspname),
							 quote_identifier(relname));
		else if (relname)
			appendStringInfo(&str, "%s",
							 quote_identifier(relname));
		if (!relname || strcmp(refname, relname) != 0)
		{
			if (str.len > 0)
				appendStringInfoChar(&str, ' ');
			appendStringInfo(&str, "%s", refname);
		}
	}
	else
	{
		ExplainPropertyText("Outer Scan Relation", relname, es);
		if (nspname)
			ExplainPropertyText("Outer Scan Schema", nspname, es);
		ExplainPropertyText("Outer Scan Alias", refname, es);
	}

	if (es->costs)
	{
		if (es->format == EXPLAIN_FORMAT_TEXT)
			appendStringInfo(&str, "  (cost=%.2f..%.2f rows=%.0f width=%d)",
							 outer_startup_cost,
							 outer_total_cost,
							 outer_plan_rows,
							 outer_plan_width);
		else
		{
			ExplainPropertyFloat("Outer Startup Cost",
								 outer_startup_cost, 2, es);
			ExplainPropertyFloat("Outer Total Cost", outer_total_cost, 2, es);
			ExplainPropertyFloat("Outer Plan Rows", outer_plan_rows, 0, es);
			ExplainPropertyInteger("Outer Plan Width", outer_plan_width, es);
		}
	}

	/*
	 * We have to forcibly clean up the instrumentation state because we
	 * haven't done ExecutorEnd yet.  This is pretty grotty ...
	 * See the comment in ExplainNode()
	 */
	InstrEndLoop(instrument);

	if (es->analyze && instrument->nloops > 0)
	{
		double	nloops = instrument->nloops;
		double	startup_sec = 1000.0 * instrument->startup / nloops;
		double	total_sec = 1000.0 * instrument->total / nloops;
		double	rows = instrument->ntuples / nloops;

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			if (es->timing)
				appendStringInfo(
					&str,
					" (actual time=%.3f..%.3f rows=%.0f loops=%.0f)",
					startup_sec, total_sec, rows, nloops);
			else
				appendStringInfo(
					&str,
					" (actual rows=%.0f loops=%.0f)",
					rows, nloops);
		}
		else
		{
			if (es->timing)
			{
				ExplainPropertyFloat("Outer Actual Startup Time",
									 startup_sec, 3, es);
				ExplainPropertyFloat("Outer Actual Total Time",
									 total_sec, 3, es);
			}
			ExplainPropertyFloat("Outer Actual Rows", rows, 0, es);
			ExplainPropertyFloat("Outer Actual Loops", nloops, 0, es);
		}
	}
	else if (es->analyze)
	{
		if (es->format == EXPLAIN_FORMAT_TEXT)
			appendStringInfoString(&str, " (never executed)");
		else
		{
			if (es->timing)
			{
				ExplainPropertyFloat("Outer Actual Startup Time", 0.0, 3, es);
				ExplainPropertyFloat("Outer Actual Total Time", 0.0, 3, es);
			}
			ExplainPropertyFloat("Outer Actual Rows", 0.0, 0, es);
			ExplainPropertyFloat("Outer Actual Loops", 0.0, 0, es);
		}
	}
	if (es->format == EXPLAIN_FORMAT_TEXT)
		ExplainPropertyText("Outer Scan", str.data, es);

	if (outer_quals)
	{
		char   *temp = deparse_expression((Node *)outer_quals,
										  deparse_context,
										  es->verbose, false);
		ExplainPropertyText("Outer Scan Filter", temp, es);

		if (gts->outer_instrument.nfiltered1 > 0.0)
			ExplainPropertyFloat("Rows Removed by Outer Scan Filter",
								 gts->outer_instrument.nfiltered1 /
								 gts->outer_instrument.nloops,
								 0, es);
	}
}

/*
 * pgstrom_init_gputasks
 */
void
pgstrom_init_gputasks(void)
{
	/* nothing to do */
}
