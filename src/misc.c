/*
 * misc.c
 *
 * miscellaneous and uncategorized routines but usefull for multiple subsystems
 * of PG-Strom.
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
 * make_flat_ands_expr - similar to make_ands_explicit but it pulls up
 * underlying and-clause
 */
Expr *
make_flat_ands_explicit(List *andclauses)
{
	List	   *args = NIL;
	ListCell   *lc;

	if (andclauses == NIL)
		return (Expr *) makeBoolConst(true, false);
	else if (list_length(andclauses) == 1)
		return (Expr *) linitial(andclauses);

	foreach (lc, andclauses)
	{
		Expr   *expr = lfirst(lc);

		Assert(exprType((Node *)expr) == BOOLOID);
		if (IsA(expr, BoolExpr) &&
			((BoolExpr *)expr)->boolop == AND_EXPR)
			args = list_concat(args, ((BoolExpr *) expr)->args);
		else
			args = lappend(args, expr);
	}
	Assert(list_length(args) > 1);
	return make_andclause(args);
}






#if PG_VERSION_NUM < 100000
/*
 * compute_parallel_worker at optimizer/path/allpaths.c
 * was newly added at PG10.x
 *
 * Compute the number of parallel workers that should be used to scan a
 * relation.  We compute the parallel workers based on the size of the heap to
 * be scanned and the size of the index to be scanned, then choose a minimum
 * of those.
 *
 * "heap_pages" is the number of pages from the table that we expect to scan,
 *  or -1 if we don't expect to scan any.
 *
 * "index_pages" is the number of pages from the index that we expect to scan,
 *  or -1 if we don't expect to scan any.
 */
int
compute_parallel_worker(RelOptInfo *rel, double heap_pages, double index_pages)
{
	int			parallel_workers = 0;

	/*
	 * If the user has set the parallel_workers reloption, use that; otherwise
	 * select a default number of workers.
	 */
	if (rel->rel_parallel_workers != -1)
		parallel_workers = rel->rel_parallel_workers;
	else
	{
		/*
		 * If the number of pages being scanned is insufficient to justify a
		 * parallel scan, just return zero ... unless it's an inheritance
		 * child. In that case, we want to generate a parallel path here
		 * anyway.  It might not be worthwhile just for this relation, but
		 * when combined with all of its inheritance siblings it may well pay
		 * off.
		 */
		if (rel->reloptkind == RELOPT_BASEREL &&
			(heap_pages >= 0 && heap_pages < min_parallel_relation_size))
			return 0;

		if (heap_pages >= 0)
		{
			int			heap_parallel_threshold;
			int			heap_parallel_workers = 1;

			/*
			 * Select the number of workers based on the log of the size of
			 * the relation.  This probably needs to be a good deal more
			 * sophisticated, but we need something here for now.  Note that
			 * the upper limit of the min_parallel_relation_size GUC is
			 * chosen to prevent overflow here.
			 */
			heap_parallel_threshold = Max(min_parallel_relation_size, 1);
			while (heap_pages >= (BlockNumber) (heap_parallel_threshold * 3))
			{
				heap_parallel_workers++;
				heap_parallel_threshold *= 3;
				if (heap_parallel_threshold > INT_MAX / 3)
					break;		/* avoid overflow */
			}

			parallel_workers = heap_parallel_workers;
		}
		/*
		 * NOTE: PG9.6 does not pay attention for # of index pages
		 * for decision of parallel execution.
		 */
	}

	/*
	 * In no case use more than max_parallel_workers_per_gather workers.
	 */
	parallel_workers = Min(parallel_workers, max_parallel_workers_per_gather);

	return parallel_workers;
}
#endif		/* < PG10 */











/*
 * errorText - string form of the error code
 */
const char *
errorText(int errcode)
{
	static __thread char buffer[800];
	const char	   *label;

	switch (errcode)
	{
		case StromError_Success:
			label = "Suceess";
			break;
		case StromError_Suspend:
			label = "GPU Suspend";
			break;
		case StromError_CpuReCheck:
			label = "CPU ReCheck";
			break;
		case StromError_InvalidValue:
			label = "Invalid Value";
			break;
		case StromError_DataStoreNoSpace:
			label = "Data store no space";
			break;
		case StromError_WrongCodeGeneration:
			label = "Wrong code generation";
			break;
		case StromError_OutOfMemory:
			label = "Out of Memory";
			break;

		/*
		 * CUDA Runtime Error - we don't want to link entire CUDA runtime
		 * for error code handling only.
		 */
#define RT_ERROR(ERRCODE, ERRNAME)										\
			case (StromError_CudaDevRunTimeBase + (ERRCODE)):			\
				label = "CUDA Runtime Error " #ERRCODE " - " #ERRNAME;	\
				break;
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
				const char *error_val;
				const char *error_str;

				/* Likely CUDA driver error */
				if (cuGetErrorName(errcode, &error_val) == CUDA_SUCCESS &&
					cuGetErrorString(errcode, &error_str) == CUDA_SUCCESS)
					snprintf(buffer, sizeof(buffer), "%s - %s",
							 error_val, error_str);
				else
					snprintf(buffer, sizeof(buffer), "%d - unknown",
							 errcode);
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
						 "Unexpected Error: %d",
						 errcode);
			}
			return buffer;
	}
	return label;
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
		KERN_ENTRY(gpujoin_main);
		KERN_ENTRY(gpujoin_right_outer);
		KERN_ENTRY(gpupreagg_setup_row);
		KERN_ENTRY(gpupreagg_setup_block);
		KERN_ENTRY(gpupreagg_nogroup_reduction);
		KERN_ENTRY(gpupreagg_groupby_reduction);
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
