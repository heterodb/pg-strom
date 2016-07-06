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


#include "pg_strom.h"



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






