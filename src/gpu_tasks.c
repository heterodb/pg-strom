/*
 * gputasks.c
 *
 * Routines to manage GpuTaskState/GpuTask state machine.
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#include "postgres.h"
#include "access/xact.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "pg_strom.h"

/*
 * static functions
 */
static void pgstrom_collect_perfmon_master(GpuTaskState_v2 *gts);
static void pgstrom_collect_perfmon_worker(GpuTaskState_v2 *gts);
static void pgstrom_explain_perfmon(GpuTaskState_v2 *gts, ExplainState *es);


/*
 * construct_kern_parambuf
 *
 * It construct a kernel parameter buffer to deliver Const/Param nodes.
 *
 * TODO: make this function static once we move all logics v2.0 based
 */
kern_parambuf *
construct_kern_parambuf(List *used_params, ExprContext *econtext)
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
pgstromInitGpuTaskState(GpuTaskState_v2 *gts,
						GpuContext_v2 *gcontext,
						GpuTaskKind task_kind,
						List *used_params,
						EState *estate)
{
	ExprContext	   *econtext = gts->css.ss.ps.ps_ExprContext;

	gts->gcontext = gcontext;
	gts->task_kind = task_kind;
	gts->program_id = INVALID_PROGRAM_ID;	/* to be set later */
	gts->revision = 1;
	gts->kern_params = construct_kern_parambuf(used_params, econtext);
	gts->scan_done = false;
	gts->row_format = false;

	gts->outer_bulk_exec = false;
	InstrInit(&gts->outer_instrument, estate->es_instrument);
	gts->scan_overflow = NULL;

	/*
	 * NOTE: initialization of HeapScanDesc was moved to the first try of
	 * ExecGpuXXX() call to support CPU parallel. A local HeapScanDesc shall
	 * be setup only when it is not responsible to partial read.
	 */

	/* callbacks shall be set by the caller */
	dlist_init(&gts->ready_tasks);
	gts->num_ready_tasks = 0;

	/* performance monitor */
	memset(&gts->pfm, 0, sizeof(pgstrom_perfmon));
	gts->pfm.enabled = pgstrom_perfmon_enabled;
	gts->pfm.prime_in_gpucontext = (gcontext->refcnt == 1);
	gts->pfm.task_kind = task_kind;
	SpinLockInit(&gts->pfm.lock);
	gts->pfm_master = NULL;		/* setup by DSM init handler */
}

/*
 * fetch_next_gputask
 *
 *
 */
static GpuTask_v2 *
fetch_next_gputask(GpuTaskState_v2 *gts)
{
	GpuContext_v2	   *gcontext = gts->gcontext;
	SharedGpuContext   *shgcon = gcontext->shgcon;
	GpuTask_v2		   *gtask;
	dlist_node		   *dnode;
	cl_uint				num_ready_tasks;
	cl_uint				num_recv_tasks = 0;
	cl_uint				num_sent_tasks = 0;

	/*
	 * If no server connection is established, GpuTask cannot be processed
	 * by GPU devices. All we can do is CPU fallback instead of the GPU
	 * processing.
	 */
	if (gcontext->sockfd == PGINVALID_SOCKET)
	{
		gtask = gts->cb_next_task(gts);
		if (gtask)
			gtask->cpu_fallback = true;
		return gtask;
	}

retry_scan:
	CHECK_FOR_INTERRUPTS();

	/*
	 * Fetch all the tasks already processed by the server side, if any.
	 * It is non-blocking operations, so we never wait for tasks currently
	 * running.
	 */
	num_ready_tasks = gts->num_ready_tasks;
	while (gpuservRecvGpuTasks(gcontext, 0));
	num_recv_tasks = gts->num_ready_tasks - num_ready_tasks;

	/*
	 * Fetch and send GpuTasks - We already fetched several tasks that were
	 * processed on the GPU server side. Unless GTS does not have many ready
	 * tasks, we try to push similar amount of tasks not to starvate GPU
	 * server, but one task at least.
	 * In addition to the above fixed-number of norm, we try to make advance
	 * the underlying scan until new asynchronous task is arrived, but as
	 * long as the number of asynchronous tasks does not touch the threshold.
	 */
	while (!gts->scan_done)
	{
		/* Do we already have enough processed GpuTask? */
		if (gts->num_ready_tasks > pgstrom_max_async_tasks / 2)
			break;

		/* Fixed number of norm to scan */
		if (num_sent_tasks <= num_recv_tasks)
		{
			gtask = gts->cb_next_task(gts);
			if (!gtask)
			{
				gts->scan_done = true;
				elog(DEBUG2, "scan done (%s)",
					 gts->css.methods->CustomName);
				break;
			}

			if (!gpuservSendGpuTask(gcontext, gtask))
				elog(ERROR, "failed to send GpuTask to GPU server");
			num_sent_tasks++;
		}
		else
		{
			/*
			 * Make advance the underlying scan until arrive of the next
			 * asynchronous task
			 */
			SpinLockAcquire(&shgcon->lock);
			if (shgcon->num_async_tasks < pgstrom_max_async_tasks)
			{
				SpinLockRelease(&shgcon->lock);

				num_ready_tasks = gts->num_ready_tasks;
				if (gpuservRecvGpuTasks(gcontext, 0))
				{
					if (gts->num_ready_tasks > num_ready_tasks)
						break;
				}

				gtask = gts->cb_next_task(gts);
				if (!gtask)
				{
					gts->scan_done = true;
					elog(DEBUG2, "scan done (%s)",
						 gts->css.methods->CustomName);
					break;
				}

				if (!gpuservSendGpuTask(gcontext, gtask))
					elog(ERROR, "failed to send GpuTask to GPU server");
			}
			else
			{
				SpinLockRelease(&shgcon->lock);
				break;
			}
		}
	}

retry_fetch:
	/*
	 * In case when GTS has no ready tasks yet, we try to wait the response
	 * synchronously.
	 */
	while (dlist_is_empty(&gts->ready_tasks))
	{
		Assert(gts->num_ready_tasks == 0);

		SpinLockAcquire(&shgcon->lock);
		if (shgcon->num_async_tasks == 0)
		{
			SpinLockRelease(&shgcon->lock);
			/*
			 * If we have no ready tasks, no asynchronous tasks and no further
			 * chunks to scan, it means this GTS gets end of the scan.
			 * If GTS may read further blocks, it needs to retry scan. (We
			 * might give up scan because of larger number of async tasks)
			 */
			if (gts->scan_done)
				return NULL;
			goto retry_scan;
		}
		SpinLockRelease(&shgcon->lock);

		/*
		 * If we have any asynchronous tasks, try synchronous receive to
		 * get next task.
		 */
		if (!gpuservRecvGpuTasks(gcontext, -1))
			elog(ERROR, "GPU server response timeout...");
	}

	/* OK, pick up GpuTask from the head */
	Assert(gts->num_ready_tasks > 0);
	dnode = dlist_pop_head_node(&gts->ready_tasks);
	gtask = dlist_container(GpuTask_v2, chain, dnode);
	gts->num_ready_tasks--;

	/*
	 * Discard GpuTask if revision number mismatch. ExecRescan() rewind the
	 * scan status then restart scan with new parameters. It means all the
	 * results of asynchronous tasks shall be discarded.
	 * To avoid synchronization here, all the GpuTask has a revision number
	 * copied from the GTS when it is constructed. It matched, the GpuTask
	 * is launched under the current GTS state.
	 */
	if (gtask->revision != gts->revision)
	{
		pgstromReleaseGpuTask(gtask);
		goto retry_fetch;
	}
	return gtask;
}

/*
 * pgstromExecGpuTaskState
 */
TupleTableSlot *
pgstromExecGpuTaskState(GpuTaskState_v2 *gts)
{
	TupleTableSlot *slot = gts->css.ss.ss_ScanTupleSlot;

	while (!gts->curr_task || !(slot = gts->cb_next_tuple(gts)))
	{
		GpuTask_v2	   *gtask = gts->curr_task;

		/* release the current GpuTask object that was already scanned */
		if (gtask)
		{
			pgstromReleaseGpuTask(gtask);
			gts->curr_task = NULL;
			gts->curr_index = 0;
			gts->curr_lp_index = 0;
		}
		/* reload next chunk to be scanned */
		gtask = fetch_next_gputask(gts);
		if (!gtask)
			break;
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
pgstromBulkExecGpuTaskState(GpuTaskState_v2 *gts, size_t chunk_size)
{
	pgstrom_data_store *pds_dst = NULL;
	TupleTableSlot	   *slot;

	/* GTS should not have neither host qualifier nor projection */
	Assert(gts->css.ss.ps.qual == NIL);
	Assert(gts->css.ss.ps.ps_ProjInfo == NULL);

	do {
		GpuTask_v2	   *gtask = gts->curr_task;

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
		pgstromReleaseGpuTask(gtask);
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
pgstromRescanGpuTaskState(GpuTaskState_v2 *gts)
{
	/*
	 * Once revision number of GTS is changed, any asynchronous GpuTasks
	 * are discarded when 
	 *
	 */
	gts->revision++;
}

/*
 * pgstromReleaseGpuTaskState
 */
void
pgstromReleaseGpuTaskState(GpuTaskState_v2 *gts)
{
	/*
	 * collect perfmon statistics if parallel worker
	 *
	 * NOTE: ExplainCustomScan() shall be called prior to EndCustomScan() of
	 * the leader process. Thus, collection of perfmon count makes sense only
	 * when it is parallel worker context.
	 * In addition, ExecEndGather() releases parallel context and DSM segment
	 * prior to EndCustomScan(), so any reference to @pfm_master on the leader
	 * process context will make SEGV.
	 */
	if (ParallelMasterBackendId != InvalidBackendId)
		pgstrom_collect_perfmon_worker(gts);
	else if (gts->pfm_master)
	{
		/* shared performance counter is no longer needed */
		dmaBufferFree(gts->pfm_master);
		gts->pfm_master = NULL;
	}
	/* cleanup per-query PDS-scan state, if any */
	PDS_end_heapscan_state(gts);
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
pgstromExplainGpuTaskState(GpuTaskState_v2 *gts, ExplainState *es)
{
	/*
	 * Extra features if any
	 */
	if (es->verbose)
	{
		char	temp[256];
		int		ofs = 0;

		/* run per-chunk-execution? */
		if (gts->outer_bulk_exec)
			ofs += snprintf(temp+ofs, sizeof(temp) - ofs,
							"%souter-bulk-exec",
							ofs > 0 ? ", " : "");
		/* per-chunk-execution support? */
		if (gts->cb_bulk_exec != NULL)
			ofs += snprintf(temp+ofs, sizeof(temp) - ofs,
							"%sbulk-exec-support",
							ofs > 0 ? ", " : "");
		/* preferable result format */
		ofs += snprintf(temp+ofs, sizeof(temp) - ofs, "%s%s-format",
						ofs > 0 ? ", " : "",
						gts->row_format ? "row" : "slot");
		/* availability of NVMe-Strom */
		if (gts->nvme_sstate)
			ofs += snprintf(temp+ofs, sizeof(temp) - ofs, "%snvme-strom",
							ofs > 0 ? ", " : "");
		if (ofs > 0)
			ExplainPropertyText("Extra", temp, es);
	}

	/*
	 * Show source path of the GPU kernel
	 */
	if (es->verbose &&
		gts->program_id != INVALID_PROGRAM_ID &&
		pgstrom_debug_kernel_source)
	{
		const char *cuda_source = pgstrom_cuda_source_file(gts->program_id);

		ExplainPropertyText("Kernel Source", cuda_source, es);
	}

	/*
	 * Show performance information
	 */
	if (es->analyze && gts->pfm.enabled)
		pgstrom_explain_perfmon(gts, es);
}

/*
 * pgstromInitGpuTask
 */
void
pgstromInitGpuTask(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{
	gtask->task_kind    = gts->task_kind;
	gtask->program_id   = gts->program_id;
	gtask->gts          = gts;
	gtask->revision     = gts->revision;
	gtask->file_desc    = -1;
	gtask->row_format   = gts->row_format;
	gtask->cpu_fallback = false;
	gtask->perfmon      = gts->pfm.enabled;
	gtask->peer_fdesc   = -1;
	gtask->cuda_stream  = NULL;
}

/*
 * pgstromProcessGpuTask - processing handler of GpuTask
 */
int
pgstromProcessGpuTask(GpuTask_v2 *gtask,
					  CUmodule cuda_module,
					  CUstream cuda_stream)
{
	int		retval;

	Assert(IsGpuServerProcess());

	switch (gtask->task_kind)
	{
		case GpuTaskKind_GpuScan:
			retval = gpuscan_process_task(gtask, cuda_module, cuda_stream);
			break;
		case GpuTaskKind_GpuJoin:
			retval = gpujoin_process_task(gtask, cuda_module, cuda_stream);
			break;
		case GpuTaskKind_GpuPreAgg:
			retval = gpupreagg_process_task(gtask, cuda_module, cuda_stream);
			break;
		default:
			elog(ERROR, "Unknown GpuTask kind: %d", gtask->task_kind);
			break;
	}
	return retval;
}

/*
 * pgstromCompleteGpuTask - completion handler of GpuTask
 */
int
pgstromCompleteGpuTask(GpuTask_v2 *gtask)
{
	int		retval;

	Assert(IsGpuServerProcess());

	switch (gtask->task_kind)
	{
		case GpuTaskKind_GpuScan:
			retval = gpuscan_complete_task(gtask);
			break;
		case GpuTaskKind_GpuJoin:
			retval = gpujoin_complete_task(gtask);
			break;
		case GpuTaskKind_GpuPreAgg:
			retval = gpupreagg_complete_task(gtask);
			break;
		default:
			elog(ERROR, "Unknown GpuTask kind: %d", (int)gtask->task_kind);
			break;
	}
	return retval;
}

/*
 * pgstromReleaseGpuTask - release of GpuTask
 */
void
pgstromReleaseGpuTask(GpuTask_v2 *gtask)
{
	switch (gtask->task_kind)
	{
		case GpuTaskKind_GpuScan:
			gpuscan_release_task(gtask);
			break;
		case GpuTaskKind_GpuJoin:
			gpujoin_release_task(gtask);
			break;
		case GpuTaskKind_GpuPreAgg:
			gpupreagg_release_task(gtask);
			break;
		default:
			elog(ERROR, "Unknown GpuTask kind: %d", (int)gtask->task_kind);
			break;
	}
}

/*
 * errorText - string form of the error code
 */
const char *
errorText(int errcode)
{
	static __thread char buffer[512];
	const char	   *error_val;
	const char	   *error_str;

	switch (errcode)
	{
		case StromError_Success:
			return "Suceess";
		case StromError_CpuReCheck:
			return "CPU ReCheck";
		case StromError_CudaInternal:
			return "CUDA Internal Error";
		case StromError_OutOfMemory:
			return "Out of memory";
		case StromError_OutOfSharedMemory:
			return "Out of shared memory";
		case StromError_OutOfKernelArgs:
			return "Out of kernel argument buffer";
		case StromError_InvalidValue:
			return "Invalid Value";
		case StromError_DataStoreCorruption:
			return "Data store corruption";
		case StromError_DataStoreNoSpace:
			return "Data store no space";
		case StromError_DataStoreOutOfRange:
			return "Data store out of range";
		case StromError_SanityCheckViolation:
			return "Sanity check violation";

		/*
		 * CUDA Runtime Error - we don't want to link entire CUDA runtime
		 * for error code handling only.
		 */
#define RT_ERROR(ERRCODE, ERRNAME)								\
			case (StromError_CudaDevRunTimeBase + (ERRCODE)):			\
				return "CUDA Runtime Error " #ERRCODE " - " #ERRNAME
			RT_ERROR(1, MissingConfiguration);
			RT_ERROR(2, MemoryAllocation);
			RT_ERROR(3, InitializationError);
			RT_ERROR(4, LaunchFailure);
			RT_ERROR(5, PriorLaunchFailure);
			RT_ERROR(6, LaunchTimeout);
			RT_ERROR(7, LaunchOutOfResources);
			RT_ERROR(8, InvalidDeviceFunction);
			RT_ERROR(9, InvalidConfiguration);
			RT_ERROR(10, InvalidDevice);
			RT_ERROR(11, InvalidValue);
			RT_ERROR(12, InvalidPitchValue);
			RT_ERROR(13, InvalidSymbol);
			RT_ERROR(14, MapBufferObjectFailed);
			RT_ERROR(15, UnmapBufferObjectFailed);
			RT_ERROR(16, InvalidHostPointer);
			RT_ERROR(17, InvalidDevicePointer);
			RT_ERROR(18, InvalidTexture);
			RT_ERROR(19, InvalidTextureBinding);
			RT_ERROR(20, InvalidChannelDescriptor);
			RT_ERROR(21, InvalidMemcpyDirection);
			RT_ERROR(22, AddressOfConstant);
			RT_ERROR(23, TextureFetchFailed);
			RT_ERROR(24, TextureNotBound);
			RT_ERROR(25, SynchronizationError);
			RT_ERROR(26, InvalidFilterSetting);
			RT_ERROR(27, InvalidNormSetting);
			RT_ERROR(28, MixedDeviceExecution);
			RT_ERROR(29, CudartUnloading);
			RT_ERROR(30, Unknown);
			RT_ERROR(31, NotYetImplemented);
			RT_ERROR(32, MemoryValueTooLarge);
			RT_ERROR(33, InvalidResourceHandle);
			RT_ERROR(34, NotReady);
			RT_ERROR(35, InsufficientDriver);
			RT_ERROR(36, SetOnActiveProcess);
			RT_ERROR(37, InvalidSurface);
			RT_ERROR(38, NoDevice);
			RT_ERROR(39, ECCUncorrectable);
			RT_ERROR(40, SharedObjectSymbolNotFound);
			RT_ERROR(41, SharedObjectInitFailed);
			RT_ERROR(42, UnsupportedLimit);
			RT_ERROR(43, DuplicateVariableName);
			RT_ERROR(44, DuplicateTextureName);
			RT_ERROR(45, DuplicateSurfaceName);
			RT_ERROR(46, DevicesUnavailable);
			RT_ERROR(47, InvalidKernelImage);
			RT_ERROR(48, NoKernelImageForDevice);
			RT_ERROR(49, IncompatibleDriverContext);
			RT_ERROR(50, PeerAccessAlreadyEnabled);
			RT_ERROR(51, PeerAccessNotEnabled);
			RT_ERROR(54, DeviceAlreadyInUse);
			RT_ERROR(55, ProfilerDisabled);
			RT_ERROR(56, ProfilerNotInitialized);
			RT_ERROR(57, ProfilerAlreadyStarted);
			RT_ERROR(58, ProfilerAlreadyStopped);
			RT_ERROR(59, Assert);
			RT_ERROR(60, TooManyPeers);
			RT_ERROR(61, HostMemoryAlreadyRegistered);
			RT_ERROR(62, HostMemoryNotRegistered);
			RT_ERROR(63, OperatingSystem);
			RT_ERROR(64, PeerAccessUnsupported);
			RT_ERROR(65, LaunchMaxDepthExceeded);
			RT_ERROR(66, LaunchFileScopedTex);
			RT_ERROR(67, LaunchFileScopedSurf);
			RT_ERROR(68, SyncDepthExceeded);
			RT_ERROR(69, LaunchPendingCountExceeded);
			RT_ERROR(70, NotPermitted);
			RT_ERROR(71, NotSupported);
			RT_ERROR(72, HardwareStackError);
			RT_ERROR(73, IllegalInstruction);
			RT_ERROR(74, MisalignedAddress);
			RT_ERROR(75, InvalidAddressSpace);
			RT_ERROR(76, InvalidPc);
			RT_ERROR(77, IllegalAddress);
			RT_ERROR(78, InvalidPtx);
			RT_ERROR(79, InvalidGraphicsContext);
			RT_ERROR(127, StartupFailure);
#undef RT_ERROR

		default:
			if (errcode <= CUDA_ERROR_UNKNOWN)
			{
				/* Likely CUDA driver error */
				if (cuGetErrorName(errcode, &error_val) == CUDA_SUCCESS &&
					cuGetErrorString(errcode, &error_str) == CUDA_SUCCESS)
					snprintf(buffer, sizeof(buffer), "%s - %s",
							 error_val, error_str);
				else
					snprintf(buffer, sizeof(buffer), "%d - unknown", errcode);
			}
			else if (errcode >= StromError_CudaDevRunTimeBase)
			{
				/* Or, unknown CUDA runtime error */
				snprintf(buffer, sizeof(buffer),
						 "CUDA Runtime Error %d - unknown",
						 errcode - StromError_CudaDevRunTimeBase);
			}
			else
			{
				/* ??? Unknown PG-Strom error??? */
				snprintf(buffer, sizeof(buffer),
						 "Unexpected Error: %d", errcode);
			}
	}
	return buffer;
}

/*
 * errorTextKernel - string form of the kern_errorbuf
 */
const char *
errorTextKernel(kern_errorbuf *kerror)
{
	static __thread char buffer[1024];
	const char *kernel_name;

#define KERN_ENTRY(KERNEL)						\
	case StromKernel_##KERNEL: kernel_name = #KERNEL; break

	switch (kerror->kernel)
	{
		KERN_ENTRY(HostPGStrom);
		KERN_ENTRY(CudaRuntime);
		KERN_ENTRY(NVMeStrom);
		KERN_ENTRY(gpuscan_exec_quals_block);
		KERN_ENTRY(gpuscan_exec_quals_row);
		KERN_ENTRY(gpuscan_projection_row);
		KERN_ENTRY(gpuscan_projection_slot);
		KERN_ENTRY(gpuscan_main);
		KERN_ENTRY(gpujoin_exec_outerscan);
		KERN_ENTRY(gpujoin_exec_nestloop);
		KERN_ENTRY(gpujoin_exec_hashjoin);
		KERN_ENTRY(gpujoin_outer_nestloop);
		KERN_ENTRY(gpujoin_outer_hashjoin);
		KERN_ENTRY(gpujoin_projection_row);
		KERN_ENTRY(gpujoin_projection_slot);
		KERN_ENTRY(gpujoin_count_rows_dist);
		KERN_ENTRY(gpujoin_main);
		KERN_ENTRY(gpupreagg_preparation);
		KERN_ENTRY(gpupreagg_local_reduction);
		KERN_ENTRY(gpupreagg_global_reduction);
		KERN_ENTRY(gpupreagg_nogroup_reduction);
		KERN_ENTRY(gpupreagg_final_preparation);
		KERN_ENTRY(gpupreagg_final_reduction);
		KERN_ENTRY(gpupreagg_fixup_varlena);
		KERN_ENTRY(gpupreagg_main);
		KERN_ENTRY(gpusort_projection);
		KERN_ENTRY(gpusort_bitonic_local);
		KERN_ENTRY(gpusort_bitonic_step);
		KERN_ENTRY(gpusort_bitonic_merge);
		KERN_ENTRY(gpusort_fixup_pointers);
		KERN_ENTRY(gpusort_main);
		KERN_ENTRY(plcuda_prep_kernel);
		KERN_ENTRY(plcuda_main_kernel);
		KERN_ENTRY(plcuda_post_kernel);
		default:
			kernel_name = "unknown kernel";
			break;
	}
#undef KERN_ENTRY
	snprintf(buffer, sizeof(buffer), "%s:%d %s",
			 kernel_name, kerror->lineno,
			 errorText(kerror->errcode));
	return buffer;
}

/* ------------------------------------------------------------ *
 *   Misc routines to support EXPLAIN command
 * ------------------------------------------------------------ */
#if 0
void
pgstrom_explain_expression(List *expr_list, const char *qlabel,
						   PlanState *planstate, List *deparse_context,
						   List *ancestors, ExplainState *es,
						   bool force_prefix, bool convert_to_and)
{
	bool        useprefix = (force_prefix | es->verbose);
	char       *exprstr;

	/* No work if empty expression list */
	if (expr_list == NIL)
		return;

	/* Deparse the expression */
	/* List shall be replaced by explicit AND, if needed */
	exprstr = deparse_expression(convert_to_and
								 ? (Node *) make_ands_explicit(expr_list)
								 : (Node *) expr_list,
								 deparse_context,
								 useprefix,
								 false);
	/* And add to es->str */
	ExplainPropertyText(qlabel, exprstr, es);
}
#endif

/*
 * pgstromExplainOuterBulkExec
 *
 * A utility routine to explain status of the outer-scan execution.
 *
 * TODO: add information to explain number of rows filtered
 */
void
pgstromExplainOuterBulkExec(GpuTaskState_v2 *gts,
							List *deparse_context,
							List *ancestors,
							ExplainState *es)
{
	Plan		   *plannode = gts->css.ss.ps.plan;
	Index			scanrelid = ((Scan *) plannode)->scanrelid;
	StringInfoData	str;

	/* Does this GpuTaskState has outer simple scan? */
	if (scanrelid == 0)
		return;

	/* Is it EXPLAIN ANALYZE? */
	if (!es->analyze)
		return;

	/*
	 * We have to forcibly clean up the instrumentation state because we
	 * haven't done ExecutorEnd yet.  This is pretty grotty ...
	 * See the comment in ExplainNode()
	 */
	InstrEndLoop(&gts->outer_instrument);

	/*
	 * See the logic in ExplainTargetRel()
	 */
	initStringInfo(&str);
	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		RangeTblEntry  *rte = rt_fetch(scanrelid, es->rtable);
		char		   *refname;
		char		   *relname;

		refname = (char *) list_nth(es->rtable_names, scanrelid - 1);
		if (refname == NULL)
			refname = rte->eref->aliasname;
		relname = get_rel_name(rte->relid);
		if (es->verbose)
		{
			char	   *nspname
				= get_namespace_name(get_rel_namespace(rte->relid));

			appendStringInfo(&str, "%s.%s",
							 quote_identifier(nspname),
							 quote_identifier(relname));
		}
		else if (relname != NULL)
			appendStringInfo(&str, "%s", quote_identifier(relname));
		if (strcmp(relname, refname) != 0)
			appendStringInfo(&str, " %s", quote_identifier(refname));
	}

	if (gts->outer_instrument.nloops > 0)
	{
		Instrumentation *instrument = &gts->outer_instrument;
		double		nloops = instrument->nloops;
		double		startup_sec = 1000.0 * instrument->startup / nloops;
		double		total_sec = 1000.0 * instrument->total / nloops;
		double		rows = instrument->ntuples / nloops;

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
	else
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

	/*
	 * Logic copied from show_buffer_usage()
	 */
	if (es->buffers)
	{
		BufferUsage *usage = &gts->outer_instrument.bufusage;

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			bool	has_shared = (usage->shared_blks_hit > 0 ||
								  usage->shared_blks_read > 0 ||
								  usage->shared_blks_dirtied > 0 ||
								  usage->shared_blks_written > 0);
			bool	has_local = (usage->local_blks_hit > 0 ||
								 usage->local_blks_read > 0 ||
								 usage->local_blks_dirtied > 0 ||
							   	 usage->local_blks_written > 0);
			bool	has_temp = (usage->temp_blks_read > 0 ||
								usage->temp_blks_written > 0);
			bool	has_timing = (!INSTR_TIME_IS_ZERO(usage->blk_read_time) ||
								  !INSTR_TIME_IS_ZERO(usage->blk_write_time));

			/* Show only positive counter values. */
			if (has_shared || has_local || has_temp)
			{
				appendStringInfoChar(&str, '\n');
				appendStringInfoSpaces(&str, es->indent * 2 + 12);
				appendStringInfoString(&str, "buffers:");

				if (has_shared)
				{
					appendStringInfoString(&str, " shared");
					if (usage->shared_blks_hit > 0)
						appendStringInfo(&str, " hit=%ld",
										 usage->shared_blks_hit);
					if (usage->shared_blks_read > 0)
						appendStringInfo(&str, " read=%ld",
										 usage->shared_blks_read);
					if (usage->shared_blks_dirtied > 0)
						appendStringInfo(&str, " dirtied=%ld",
										 usage->shared_blks_dirtied);
					if (usage->shared_blks_written > 0)
						appendStringInfo(&str, " written=%ld",
										 usage->shared_blks_written);
					if (has_local || has_temp)
						appendStringInfoChar(&str, ',');
				}
				if (has_local)
				{
					appendStringInfoString(&str, " local");
					if (usage->local_blks_hit > 0)
						appendStringInfo(&str, " hit=%ld",
										 usage->local_blks_hit);
					if (usage->local_blks_read > 0)
						appendStringInfo(&str, " read=%ld",
										 usage->local_blks_read);
					if (usage->local_blks_dirtied > 0)
						appendStringInfo(&str, " dirtied=%ld",
										 usage->local_blks_dirtied);
					if (usage->local_blks_written > 0)
						appendStringInfo(&str, " written=%ld",
										 usage->local_blks_written);
					if (has_temp)
						appendStringInfoChar(&str, ',');
				}
				if (has_temp)
				{
					appendStringInfoString(&str, " temp");
					if (usage->temp_blks_read > 0)
						appendStringInfo(&str, " read=%ld",
										 usage->temp_blks_read);
					if (usage->temp_blks_written > 0)
						appendStringInfo(&str, " written=%ld",
										 usage->temp_blks_written);
				}
			}

			/* As above, show only positive counter values. */
			if (has_timing)
			{
				if (has_shared || has_local || has_temp)
					appendStringInfo(&str, ", ");
				appendStringInfoString(&str, "I/O Timings:");
				if (!INSTR_TIME_IS_ZERO(usage->blk_read_time))
					appendStringInfo(&str, " read=%0.3f",
							INSTR_TIME_GET_MILLISEC(usage->blk_read_time));
				if (!INSTR_TIME_IS_ZERO(usage->blk_write_time))
					appendStringInfo(&str, " write=%0.3f",
							INSTR_TIME_GET_MILLISEC(usage->blk_write_time));
			}
		}
		else
		{
			double		time_value;
			ExplainPropertyLong("Outer Shared Hit Blocks",
								usage->shared_blks_hit, es);
			ExplainPropertyLong("Outer Shared Read Blocks",
								usage->shared_blks_read, es);
			ExplainPropertyLong("Outer Shared Dirtied Blocks",
								usage->shared_blks_dirtied, es);
			ExplainPropertyLong("Outer Shared Written Blocks",
								usage->shared_blks_written, es);
			ExplainPropertyLong("Outer Local Hit Blocks",
								usage->local_blks_hit, es);
			ExplainPropertyLong("Outer Local Read Blocks",
								usage->local_blks_read, es);
			ExplainPropertyLong("Outer Local Dirtied Blocks",
								usage->local_blks_dirtied, es);
			ExplainPropertyLong("Outer Local Written Blocks",
								usage->local_blks_written, es);
			ExplainPropertyLong("Outer Temp Read Blocks",
								usage->temp_blks_read, es);
			ExplainPropertyLong("Outer Temp Written Blocks",
								usage->temp_blks_written, es);
			time_value = INSTR_TIME_GET_MILLISEC(usage->blk_read_time);
			ExplainPropertyFloat("Outer I/O Read Time", time_value, 3, es);
			time_value = INSTR_TIME_GET_MILLISEC(usage->blk_write_time);
			ExplainPropertyFloat("Outer I/O Write Time", time_value, 3, es);
		}
	}

	if (es->format == EXPLAIN_FORMAT_TEXT)
		ExplainPropertyText("Outer Scan", str.data, es);
}

#if 0
void
show_scan_qual(List *qual, const char *qlabel,
               PlanState *planstate, List *ancestors,
               ExplainState *es)
{
	bool        useprefix;
	Node	   *node;
	List       *context;
	char       *exprstr;

	useprefix = (IsA(planstate->plan, SubqueryScan) || es->verbose);

	/* No work if empty qual */
	if (qual == NIL)
		return;

	/* Convert AND list to explicit AND */
	node = (Node *) make_ands_explicit(qual);

	/* Set up deparsing context */
	context = set_deparse_context_planstate(es->deparse_cxt,
											(Node *) planstate,
											ancestors);
	/* Deparse the expression */
	exprstr = deparse_expression(node, context, useprefix, false);

	/* And add to es->str */
	ExplainPropertyText(qlabel, exprstr, es);
}
#endif
#if 0
/*
 * If it's EXPLAIN ANALYZE, show instrumentation information for a plan node
 *
 * "which" identifies which instrumentation counter to print
 */
void
show_instrumentation_count(const char *qlabel, int which,
						   PlanState *planstate, ExplainState *es)
{
	double		nfiltered;
	double		nloops;

	if (!es->analyze || !planstate->instrument)
		return;

	if (which == 2)
		nfiltered = planstate->instrument->nfiltered2;
	else
		nfiltered = planstate->instrument->nfiltered1;
	nloops = planstate->instrument->nloops;

	/* In text mode, suppress zero counts; they're not interesting enough */
	if (nfiltered > 0 || es->format != EXPLAIN_FORMAT_TEXT)
	{
		if (nloops > 0)
			ExplainPropertyFloat(qlabel, nfiltered / nloops, 0, es);
		else
			ExplainPropertyFloat(qlabel, 0.0, 0, es);
	}
}
#endif



/*
 * pgstrom_collect_perfmon - collect perfmon information
 */
static inline void
__pgstrom_collect_perfmon(pgstrom_perfmon *pfm_dst, pgstrom_perfmon *pfm_src)
{
#define PFM_ADD(FIELD)			pfm_dst->FIELD += pfm_src->FIELD
	PFM_ADD(num_dmabuf_alloc);
	PFM_ADD(num_dmabuf_free);
	PFM_ADD(num_gpumem_alloc);
	PFM_ADD(num_gpumem_free);
	PFM_ADD(num_iomapped_alloc);
	PFM_ADD(num_iomapped_free);
	PFM_ADD(tv_dmabuf_alloc);
	PFM_ADD(tv_dmabuf_free);
	PFM_ADD(tv_gpumem_alloc);
	PFM_ADD(tv_gpumem_free);
	PFM_ADD(tv_iomapped_alloc);
	PFM_ADD(tv_iomapped_free);
	PFM_ADD(size_dmabuf_total);
	PFM_ADD(size_gpumem_total);
	PFM_ADD(size_iomapped_total);

	/* build cuda program */
	// no idea how to track...

	/* time for i/o stuff */
	PFM_ADD(time_inner_load);
	PFM_ADD(time_outer_load);
	PFM_ADD(time_materialize);

	/* dma data transfer */
	PFM_ADD(num_dma_send);
	PFM_ADD(num_dma_recv);
	PFM_ADD(bytes_dma_send);
	PFM_ADD(bytes_dma_recv);
	PFM_ADD(time_dma_send);
	PFM_ADD(time_dma_recv);

	/* common GPU tasks */
	PFM_ADD(num_tasks);
	PFM_ADD(time_launch_cuda);
	PFM_ADD(time_sync_tasks);

	/* GpuScan */
	PFM_ADD(gscan.num_kern_main);
	PFM_ADD(gscan.tv_kern_main);
	PFM_ADD(gscan.tv_kern_exec_quals);
	PFM_ADD(gscan.tv_kern_projection);

	/* GpuJoin */
	PFM_ADD(gjoin.num_kern_main);
	PFM_ADD(gjoin.num_kern_outer_scan);
	PFM_ADD(gjoin.num_kern_exec_nestloop);
	PFM_ADD(gjoin.num_kern_exec_hashjoin);
	PFM_ADD(gjoin.num_kern_outer_nestloop);
	PFM_ADD(gjoin.num_kern_outer_hashjoin);
	PFM_ADD(gjoin.num_kern_projection);
	PFM_ADD(gjoin.num_kern_rows_dist);
	PFM_ADD(gjoin.num_global_retry);
	PFM_ADD(gjoin.num_major_retry);
	PFM_ADD(gjoin.num_minor_retry);
	PFM_ADD(gjoin.tv_kern_main);
	PFM_ADD(gjoin.tv_kern_outer_scan);
	PFM_ADD(gjoin.tv_kern_exec_nestloop);
	PFM_ADD(gjoin.tv_kern_exec_hashjoin);
	PFM_ADD(gjoin.tv_kern_outer_nestloop);
	PFM_ADD(gjoin.tv_kern_outer_hashjoin);
	PFM_ADD(gjoin.tv_kern_projection);
	PFM_ADD(gjoin.tv_kern_rows_dist);
	PFM_ADD(gjoin.num_inner_dma_send);
	PFM_ADD(gjoin.bytes_inner_dma_send);
	PFM_ADD(gjoin.tv_inner_dma_send);

	/* GpuPreAgg */
	PFM_ADD(gpreagg.num_kern_main);
	PFM_ADD(gpreagg.num_kern_prep);
	PFM_ADD(gpreagg.num_kern_nogrp);
	PFM_ADD(gpreagg.num_kern_lagg);
	PFM_ADD(gpreagg.num_kern_gagg);
	PFM_ADD(gpreagg.num_kern_fagg);
	PFM_ADD(gpreagg.num_kern_fixvar);
	PFM_ADD(gpreagg.tv_kern_main);
	PFM_ADD(gpreagg.tv_kern_prep);
	PFM_ADD(gpreagg.tv_kern_nogrp);
	PFM_ADD(gpreagg.tv_kern_lagg);
	PFM_ADD(gpreagg.tv_kern_gagg);
	PFM_ADD(gpreagg.tv_kern_fagg);
	PFM_ADD(gpreagg.tv_kern_fixvar);

	/* GpuSort */
	PFM_ADD(gsort.num_kern_proj);
	PFM_ADD(gsort.num_kern_main);
	PFM_ADD(gsort.num_kern_lsort);
	PFM_ADD(gsort.num_kern_ssort);
	PFM_ADD(gsort.num_kern_msort);
	PFM_ADD(gsort.num_kern_fixvar);
	PFM_ADD(gsort.tv_kern_proj);
	PFM_ADD(gsort.tv_kern_main);
	PFM_ADD(gsort.tv_kern_lsort);
	PFM_ADD(gsort.tv_kern_ssort);
	PFM_ADD(gsort.tv_kern_msort);
	PFM_ADD(gsort.tv_kern_fixvar);
	PFM_ADD(gsort.tv_cpu_sort);
#undef PFM_ADD
}

static inline void
pgstrom_collect_perfmon_shared_gcontext(GpuTaskState_v2 *gts)
{
#define PFM_MOVE(FIELD)	gts->pfm.FIELD = shgcon->pfm.FIELD
	SharedGpuContext   *shgcon = gts->gcontext->shgcon;

	PFM_MOVE(num_dmabuf_alloc);
	PFM_MOVE(num_dmabuf_free);
	PFM_MOVE(num_gpumem_alloc);
	PFM_MOVE(num_gpumem_free);
	PFM_MOVE(num_iomapped_alloc);
	PFM_MOVE(num_iomapped_free);
	PFM_MOVE(tv_dmabuf_alloc);
	PFM_MOVE(tv_dmabuf_free);
	PFM_MOVE(tv_gpumem_alloc);
	PFM_MOVE(tv_gpumem_free);
	PFM_MOVE(tv_iomapped_alloc);
	PFM_MOVE(tv_iomapped_free);
	PFM_MOVE(size_dmabuf_total);
	PFM_MOVE(size_gpumem_total);
	PFM_MOVE(size_iomapped_total);
#undef PFM_MOVE
}

static void
pgstrom_collect_perfmon_master(GpuTaskState_v2 *gts)
{
	Assert(ParallelMasterBackendId == InvalidBackendId);
	if (!gts->pfm.enabled)
		return;
	pgstrom_collect_perfmon_shared_gcontext(gts);
	if (gts->pfm_master)
	{
		SpinLockAcquire(&gts->pfm_master->lock);
		__pgstrom_collect_perfmon(&gts->pfm, gts->pfm_master);
		SpinLockRelease(&gts->pfm_master->lock);
	}
}

static void
pgstrom_collect_perfmon_worker(GpuTaskState_v2 *gts)
{
	Assert(ParallelMasterBackendId != InvalidBackendId);
	if (!gts->pfm.enabled)
		return;
	pgstrom_collect_perfmon_shared_gcontext(gts);
	if (gts->pfm_master)
	{
		SpinLockAcquire(&gts->pfm_master->lock);
		__pgstrom_collect_perfmon(gts->pfm_master, &gts->pfm);
		SpinLockRelease(&gts->pfm_master->lock);
	}
}

/*
 * pgstrom_explain_perfmon - common routine to explain performance info
 */
static void
pgstrom_explain_perfmon(GpuTaskState_v2 *gts, ExplainState *es)
{
	pgstrom_perfmon	   *pfm = &gts->pfm;
	char				buf[1024];

	if (!pfm->enabled)
		return;
	pgstrom_collect_perfmon_master(gts);

	/* common performance statistics */
	ExplainPropertyInteger("Number of tasks", pfm->num_tasks, es);

#define EXPLAIN_KERNEL_PERFMON(label,num_field,tv_field)		\
	do {														\
		if (pfm->num_field > 0)									\
		{														\
			snprintf(buf, sizeof(buf),							\
					 "total: %s, avg: %s, count: %u",			\
					 format_millisec(pfm->tv_field),			\
					 format_millisec(pfm->tv_field /			\
									 (double)pfm->num_field),	\
					 pfm->num_field);							\
			ExplainPropertyText(label, buf, es);				\
		}														\
	} while(0)

	switch (pfm->task_kind)
	{
		case GpuTaskKind_GpuScan:
			EXPLAIN_KERNEL_PERFMON("gpuscan_main()",
								   gscan.num_kern_main,
								   gscan.tv_kern_main);
			EXPLAIN_KERNEL_PERFMON(" - gpuscan_exec_quals",
								   gscan.num_kern_main,
								   gscan.tv_kern_exec_quals);
			EXPLAIN_KERNEL_PERFMON(" - gpuscan_projection",
								   gscan.num_kern_main,
								   gscan.tv_kern_projection);
			break;

		case GpuTaskKind_GpuJoin:
			EXPLAIN_KERNEL_PERFMON("gpujoin_main()",
								   gjoin.num_kern_main,
								   gjoin.tv_kern_main);
			EXPLAIN_KERNEL_PERFMON(" - gpujoin_exec_outerscan",
								   gjoin.num_kern_outer_scan,
								   gjoin.tv_kern_outer_scan);
			EXPLAIN_KERNEL_PERFMON(" - gpujoin_exec_nestloop",
								   gjoin.num_kern_exec_nestloop,
								   gjoin.tv_kern_exec_nestloop);
			EXPLAIN_KERNEL_PERFMON(" - gpujoin_exec_hashjoin",
								   gjoin.num_kern_exec_hashjoin,
								   gjoin.tv_kern_exec_hashjoin);
			EXPLAIN_KERNEL_PERFMON(" - gpujoin_outer_nestloop",
								   gjoin.num_kern_outer_nestloop,
								   gjoin.tv_kern_outer_nestloop);
			EXPLAIN_KERNEL_PERFMON(" - gpujoin_outer_hashjoin",
								   gjoin.num_kern_outer_hashjoin,
								   gjoin.tv_kern_outer_hashjoin);
			EXPLAIN_KERNEL_PERFMON(" - gpujoin_projection",
								   gjoin.num_kern_projection,
								   gjoin.tv_kern_projection);
			EXPLAIN_KERNEL_PERFMON(" - gpujoin_count_rows_dist",
								   gjoin.num_kern_rows_dist,
								   gjoin.tv_kern_rows_dist);
			if (pfm->gjoin.num_global_retry > 0 ||
				pfm->gjoin.num_major_retry > 0 ||
				pfm->gjoin.num_minor_retry > 0)
			{
				snprintf(buf, sizeof(buf), "global: %u, major: %u, minor: %u",
						 pfm->gjoin.num_global_retry,
						 pfm->gjoin.num_major_retry,
						 pfm->gjoin.num_minor_retry);
				ExplainPropertyText("Retry Loops", buf, es);
			}
			break;

		case GpuTaskKind_GpuPreAgg:
			EXPLAIN_KERNEL_PERFMON("gpupreagg_main()",
								   gpreagg.num_kern_main,
								   gpreagg.tv_kern_main);
			EXPLAIN_KERNEL_PERFMON(" - gpupreagg_preparation()",
								   gpreagg.num_kern_prep,
								   gpreagg.tv_kern_prep);
			EXPLAIN_KERNEL_PERFMON(" - gpupreagg_nogroup_reduction()",
								   gpreagg.num_kern_nogrp,
								   gpreagg.tv_kern_nogrp);
			EXPLAIN_KERNEL_PERFMON(" - gpupreagg_local_reduction()",
								   gpreagg.num_kern_lagg,
								   gpreagg.tv_kern_lagg);
			EXPLAIN_KERNEL_PERFMON(" - gpupreagg_global_reduction()",
								   gpreagg.num_kern_gagg,
								   gpreagg.tv_kern_gagg);
			EXPLAIN_KERNEL_PERFMON(" - gpupreagg_final_reduction()",
								   gpreagg.num_kern_fagg,
								   gpreagg.tv_kern_fagg);
			EXPLAIN_KERNEL_PERFMON(" - gpupreagg_fixup_varlena()",
								   gpreagg.num_kern_fixvar,
								   gpreagg.tv_kern_fixvar);
			break;

		case GpuTaskKind_GpuSort:
			EXPLAIN_KERNEL_PERFMON("gpusort_projection()",
								   gsort.num_kern_proj,
								   gsort.tv_kern_proj);
			EXPLAIN_KERNEL_PERFMON("gpusort_main()",
								   gsort.num_kern_main,
								   gsort.tv_kern_main);
			EXPLAIN_KERNEL_PERFMON(" - gpusort_bitonic_local()",
								   gsort.num_kern_lsort,
								   gsort.tv_kern_lsort);
			EXPLAIN_KERNEL_PERFMON(" - gpusort_bitonic_step()",
								   gsort.num_kern_ssort,
								   gsort.tv_kern_ssort);
			EXPLAIN_KERNEL_PERFMON(" - gpusort_bitonic_merge()",
								   gsort.num_kern_msort,
								   gsort.tv_kern_msort);
			EXPLAIN_KERNEL_PERFMON(" - gpusort_fixup_pointers()",
								   gsort.num_kern_fixvar,
								   gsort.tv_kern_fixvar);
			snprintf(buf, sizeof(buf), "total: %s",
					 format_millisec(pfm->gsort.tv_cpu_sort));
			ExplainPropertyText("CPU merge sort", buf, es);
			break;

		default:
			elog(ERROR, "unexpected GpuTaskKind: %d", (int)pfm->task_kind);
			break;
	}
#undef EXPLAIN_KERNEL_PERFMON

	/* Time of I/O stuff */
	if (pfm->task_kind == GpuTaskKind_GpuJoin)
	{
		snprintf(buf, sizeof(buf), "%s",
				 format_millisec(pfm->time_inner_load));
		ExplainPropertyText("Time of inner load", buf, es);
		snprintf(buf, sizeof(buf), "%s",
				 format_millisec(pfm->time_outer_load));
		ExplainPropertyText("Time of outer load", buf, es);
	}
	else
	{
		snprintf(buf, sizeof(buf), "%s",
				 format_millisec(pfm->time_outer_load));
		ExplainPropertyText("Time of load", buf, es);
	}

	snprintf(buf, sizeof(buf), "%s",
			 format_millisec(pfm->time_materialize));
	ExplainPropertyText("Time of materialize", buf, es);

	/* DMA Send/Recv performance */
	if (pfm->num_dma_send > 0)
	{
		Size	band = (Size)((double)pfm->bytes_dma_send *
							  1000.0 / pfm->time_dma_send);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 format_bytesz(band),
				 format_bytesz((double)pfm->bytes_dma_send),
				 format_millisec(pfm->time_dma_send),
				 pfm->num_dma_send);
		ExplainPropertyText("DMA send", buf, es);
	}

	if (pfm->num_dma_recv > 0)
	{
		Size	band = (Size)((double)pfm->bytes_dma_recv *
							  1000.0 / pfm->time_dma_recv);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 format_bytesz(band),
				 format_bytesz((double)pfm->bytes_dma_recv),
				 format_millisec(pfm->time_dma_recv),
				 pfm->num_dma_recv);
		ExplainPropertyText("DMA recv", buf, es);
	}

	/* Time to build CUDA code */
	if (pfm->tv_build_start.tv_sec > 0 &&
		pfm->tv_build_end.tv_sec > 0 &&
		(pfm->tv_build_start.tv_sec < pfm->tv_build_end.tv_sec ||
		 (pfm->tv_build_start.tv_sec == pfm->tv_build_end.tv_sec &&
		  pfm->tv_build_start.tv_usec < pfm->tv_build_end.tv_usec)))
	{
		cl_double	tv_cuda_build = PERFMON_TIMEVAL_DIFF(pfm->tv_build_start,
														 pfm->tv_build_end);
		snprintf(buf, sizeof(buf), "%s", format_millisec(tv_cuda_build));
		ExplainPropertyText("Build CUDA Program", buf, es);
	}

	/* Host/Device Memory Allocation (only prime node) */
	if (pfm->prime_in_gpucontext)
	{
		if (pfm->num_dmabuf_alloc > 0 ||
			pfm->num_dmabuf_free > 0)
		{
			if (pfm->num_dmabuf_alloc == pfm->num_dmabuf_free)
			{
				snprintf(buf, sizeof(buf),
						 "total: %s, count: %u, time: %s",
						 format_bytesz(pfm->size_dmabuf_total),
						 pfm->num_dmabuf_alloc,
						 format_millisec(pfm->tv_dmabuf_alloc +
										 pfm->tv_dmabuf_free));
			}
			else
			{
				snprintf(buf, sizeof(buf),
						 "total: %s, alloc (count: %u, time: %s) "
						 "free (count: %u, time: %s)",
						 format_bytesz(pfm->size_dmabuf_total),
						 pfm->num_dmabuf_alloc,
						 format_millisec(pfm->tv_dmabuf_alloc),
						 pfm->num_dmabuf_free,
						 format_millisec(pfm->tv_dmabuf_free));
			}
			ExplainPropertyText("DMA Buffer", buf, es);
		}

		if (pfm->num_gpumem_alloc > 0 ||
			pfm->num_gpumem_free > 0)
		{
			if (pfm->num_gpumem_alloc == pfm->num_gpumem_free)
			{
				snprintf(buf, sizeof(buf),
						 "total: %s, count: %u, time: %s",
						 format_bytesz(pfm->size_gpumem_total),
						 pfm->num_gpumem_alloc,
						 format_millisec(pfm->tv_gpumem_alloc +
										 pfm->tv_gpumem_free));
			}
			else
			{
				snprintf(buf, sizeof(buf),
						 "total: %s, alloc (count: %u, time: %s) "
						 "free (count: %u, time: %s)",
						 format_bytesz(pfm->size_gpumem_total),
						 pfm->num_gpumem_alloc,
						 format_millisec(pfm->tv_gpumem_alloc),
						 pfm->num_gpumem_free,
						 format_millisec(pfm->tv_gpumem_free));
			}
			ExplainPropertyText("GPU Memory", buf, es);
		}

		if (pfm->num_iomapped_alloc > 0 ||
			pfm->num_iomapped_free > 0)
		{
			if (pfm->num_iomapped_alloc == pfm->num_iomapped_free)
			{
				snprintf(buf, sizeof(buf),
						 "total: %s, count: %u, time: %s",
						 format_bytesz(pfm->size_iomapped_total),
						 pfm->num_iomapped_alloc,
						 format_millisec(pfm->tv_iomapped_alloc +
										 pfm->tv_iomapped_free));
			}
			else
			{
				snprintf(buf, sizeof(buf),
						 "total: %s, alloc (count: %u, time: %s) "
						 "free (count: %u, time: %s)",
						 format_bytesz(pfm->size_iomapped_total),
						 pfm->num_iomapped_alloc,
						 format_millisec(pfm->tv_iomapped_alloc),
						 pfm->num_iomapped_free,
						 format_millisec(pfm->tv_iomapped_free));
			}
			ExplainPropertyText("I/O Mapped Memory", buf, es);
		}
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






