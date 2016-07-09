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
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "pg_strom.h"


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
							 errmsg("type of parameter %d (%s) does not match that when preparing the plan (%s)",
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
						ProgramId program_id,
						List *used_params,
						EState *estate)
{
	ExprContext	   *econtext = gts->css.ss.ps.ps_ExprContext;

	gts->gcontext = gcontext;
	gts->task_kind = task_kind;
	gts->program_id = program_id;
	gts->revision = 1;
	gts->kern_params = construct_kern_parambuf(used_params, econtext);
	gts->scan_done = false;
	gts->row_format = false;

	gts->outer_bulk_exec = false;
	InstrInit(&gts->outer_instrument, estate->es_instrument);
	if (gts->css.ss.ss_currentRelation)
	{
		Relation		scan_rel = gts->css.ss.ss_currentRelation;
		HeapScanDesc	scan_desc = heap_beginscan(scan_rel,
												   estate->es_snapshot,
												   0, NULL);
		gts->css.ss.ss_currentScanDesc = scan_desc;
	}
	gts->scan_overflow = NULL;

	/* callbacks shall be set by the caller */

	dlist_init(&gts->ready_tasks);
	gts->num_running_tasks = 0;
	gts->num_ready_tasks = 0;

	/* performance monitor */
	memset(&gts->pfm, 0, sizeof(pgstrom_perfmon));
	gts->pfm.enabled = pgstrom_perfmon_enabled;
	gts->pfm.prime_in_gpucontext = (gcontext->refcnt == 1);
	// FIXME: pfm should has GpuTaskKind instead of extra_flags in v2.0
	switch (task_kind)
	{
		case GpuTaskKind_GpuScan:
			gts->pfm.extra_flags = DEVKERNEL_NEEDS_GPUSCAN;
			break;
		case GpuTaskKind_GpuJoin:
			gts->pfm.extra_flags = DEVKERNEL_NEEDS_GPUJOIN;
			break;
		case GpuTaskKind_GpuPreAgg:
			gts->pfm.extra_flags = DEVKERNEL_NEEDS_GPUPREAGG;
			break;
		case GpuTaskKind_GpuSort:
			gts->pfm.extra_flags = DEVKERNEL_NEEDS_GPUSORT;
			break;
		default:
			elog(ERROR, "unexpected GpuTaskKind: %d", (int)task_kind);
	}
}

/*
 * fetch_next_gputask
 *
 *
 */
static GpuTask_v2 *
fetch_next_gputask(GpuTaskState_v2 *gts)
{
	GpuContext_v2  *gcontext = gts->gcontext;
	GpuTask_v2	   *gtask;

	/*
	 * If no server connection is established, GpuTask cannot be processed
	 * by GPU devices. All we can do is CPU fallback instead of the GPU
	 * processing.
	 */
	if (gcontext->sockfd == PGINVALID_SOCKET)
	{
		gtask = gts->cb_next_task(gts);
		gtask->cpu_fallback = true;
		return gtask;
	}


	// pop all the tasks already returned, if any

	if (!gts->scan_done)
	{
		// read untils next response is backed, or reached to max_async


	}

	// wait for remote task if empty


	// poll the response from the server
	// if any back it
	// prior to this, push some asynchronous tasks 



	/*
	 * Error handling - See definition of ereport(). What we have to report
	 * is an error on the server side where it actually happen.
	 */
	if (gtask->error.errcode != 0)
	{
		if (errstart(ERROR,
					 gtask->error.filename,
					 gtask->error.lineno,
					 gtask->error.filename,
					 TEXTDOMAIN))
			errfinish(errcode(gtask->error.errcode),
					  errmsg("%s", gtask->error.message));
		pg_unreachable();
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
			gts->cb_task_release(gtask);
			gts->curr_task = NULL;
			gts->curr_index = 0;
		}
		/* reload next chunk to be scanned */
		gtask = fetch_next_gputask(gts);
		if (!gtask)
			break;
		gts->curr_task = gtask;
		gts->curr_index = 0;
		/* notify a new task is assigned */
		if (gts->cb_switch_task)
			gts->cb_switch_task(gts, gtask);
	}
	return slot;
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
		KERN_ENTRY(CudaRuntime);
		KERN_ENTRY(gpuscan_exec_quals);
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

/*
 * pgstrom_init_gputasks
 */
void
pgstrom_init_gputasks(void)
{

}






