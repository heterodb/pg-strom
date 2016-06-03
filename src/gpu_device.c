/*
 * gpu_device.c
 *
 * Routines to collect GPU device information.
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
#include "funcapi.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include "gpu_device.h"

/* public variable declaration */
DevAttributes  *devAttrs = NULL;
cl_int			numDevAttrs = 0;

/* catalog of device attributes */
enum DevAttrKind {
	DEVATTRKIND__BOOL,
	DEVATTRKIND__INT,
	DEVATTRKIND__SIZE,
	DEVATTRKIND__KHZ,
	DEVATTRKIND__COMP_MODE,
	DEVATTRKIND__BITS,
};

static struct {
	enum CUdevice_attribute	attr_id;
	enum DevAttrKind attr_kind;
	size_t		attr_offset;
	const char *attr_desc;
} DevAttrCatalog[] = {
#define _DEVATTR(ATTLABEL,ATTKIND,ATTDESC)				\
	{ CU_DEVICE_ATTRIBUTE_##ATTLABEL,					\
	  DEVATTRKIND__##ATTKIND,							\
	  offsetof(struct DevAttributesData, ATTLABEL),		\
	  (ATTDESC) }
	_DEVATTR(MAX_THREADS_PER_BLOCK, INT,
			 "Max number of threads per block"),
	_DEVATTR(MAX_BLOCK_DIM_X, INT,
			 "Max block dimension X"),
	_DEVATTR(MAX_BLOCK_DIM_Y, INT,
			 "Max block dimension Y"),
	_DEVATTR(MAX_BLOCK_DIM_Z, INT,
			 "Max block dimension Z"),
	_DEVATTR(MAX_GRID_DIM_X, INT,
			 "Max grid dimension X"),
	_DEVATTR(MAX_GRID_DIM_Y, INT,
			 "Max grid dimension Y"),
	_DEVATTR(MAX_GRID_DIM_Z, INT,
			 "Max grid dimension Z"),
	_DEVATTR(MAX_SHARED_MEMORY_PER_BLOCK, SIZE,
			 "Max shared memory available per block"),
	_DEVATTR(TOTAL_CONSTANT_MEMORY, SIZE,
			 "Constant memory available on device"),
	_DEVATTR(WARP_SIZE, INT,
			 "Warp size in threads"),
	_DEVATTR(MAX_PITCH, SIZE,
			 "Max pitch in bytes allowed by memory copies"),
	_DEVATTR(MAX_REGISTERS_PER_BLOCK, INT,
			 "Max number of 32-bit registers per block"),
	_DEVATTR(CLOCK_RATE, KHZ,
			 "Typical clock frequency in kHz"),
	_DEVATTR(TEXTURE_ALIGNMENT, BOOL,
			 "Alignment requirement for textures"),
	_DEVATTR(MULTIPROCESSOR_COUNT, INT,
			 "Number of multiprocessors on device"),
	_DEVATTR(KERNEL_EXEC_TIMEOUT, BOOL,
			 "Whether there is a run time limit on kernels"),
	_DEVATTR(INTEGRATED, BOOL,
			 "Device is integrated with host memory"),
	_DEVATTR(CAN_MAP_HOST_MEMORY, BOOL,
			 "Device can map host memory into CUDA address space"),
	_DEVATTR(COMPUTE_MODE, COMP_MODE,
			 "Device compute mode"),
	_DEVATTR(MAXIMUM_TEXTURE1D_WIDTH, INT,
			 "Max 1D texture width"),
	_DEVATTR(MAXIMUM_TEXTURE2D_WIDTH, INT,
			 "Max 2D texture width"),
	_DEVATTR(MAXIMUM_TEXTURE2D_HEIGHT, INT,
			 "Max 2D texture height"),
	_DEVATTR(MAXIMUM_TEXTURE3D_WIDTH, INT,
			 "Max 3D texture width"),
	_DEVATTR(MAXIMUM_TEXTURE3D_HEIGHT, INT,
			 "Max 3D texture height"),
	_DEVATTR(MAXIMUM_TEXTURE3D_DEPTH, INT,
			 "Max 3D texture depth"),
	_DEVATTR(MAXIMUM_TEXTURE2D_LAYERED_WIDTH, INT,
			 "Max 2D layered texture width"),
	_DEVATTR(MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, INT,
			 "Max 2D layered texture height"),
	_DEVATTR(MAXIMUM_TEXTURE2D_LAYERED_LAYERS, INT,
			 "Max layers in a 2D layered texture"),
	_DEVATTR(SURFACE_ALIGNMENT, INT,
			 "Alignment requirement for surfaces"),
	_DEVATTR(CONCURRENT_KERNELS, BOOL,
			 "Device supports concurrent kernel execution"),
	_DEVATTR(ECC_ENABLED, BOOL,
			 "Device has ECC support enabled"),
	_DEVATTR(PCI_BUS_ID, INT,
			 "PCI bus ID of the device"),
	_DEVATTR(PCI_DEVICE_ID, INT,
			 "PCI device ID of the device"),
	_DEVATTR(TCC_DRIVER, BOOL,
			 "Device is using TCC driver model"),
	_DEVATTR(MEMORY_CLOCK_RATE, KHZ,
			 "Peak memory clock frequency in kHz"),
	_DEVATTR(GLOBAL_MEMORY_BUS_WIDTH, BITS,
			 "Global memory bus width in bits"),
	_DEVATTR(L2_CACHE_SIZE, SIZE,
			 "Size of L2 cache in bytes"),
	_DEVATTR(MAX_THREADS_PER_MULTIPROCESSOR, INT,
			 "Max resident threads per multiprocessor"),
	_DEVATTR(ASYNC_ENGINE_COUNT, INT,
			 "Number of asynchronous engines"),
	_DEVATTR(UNIFIED_ADDRESSING, BOOL,
			 "Device shares a unified address space with the host"),
	_DEVATTR(MAXIMUM_TEXTURE1D_LAYERED_WIDTH, INT,
			 "Max 1D layered texture width"),
	_DEVATTR(MAXIMUM_TEXTURE1D_LAYERED_LAYERS, INT,
			 "Max layers in a 1D layered texture"),
	_DEVATTR(MAXIMUM_TEXTURE2D_GATHER_WIDTH, INT,
			 "Max 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER"),
	_DEVATTR(MAXIMUM_TEXTURE2D_GATHER_HEIGHT, INT,
			"Max 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER"),
	_DEVATTR(MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, INT,
			 "Alternate maximum 3D texture width"),
	_DEVATTR(MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, INT,
			 "Alternate maximum 3D texture height"),
	_DEVATTR(MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, INT,
			 "Alternate maximum 3D texture depth"),
	_DEVATTR(PCI_DOMAIN_ID, INT,
			 "PCI domain ID of the device"),
	_DEVATTR(TEXTURE_PITCH_ALIGNMENT, INT,
			 "Pitch alignment requirement for textures"),
	_DEVATTR(MAXIMUM_TEXTURECUBEMAP_WIDTH, INT,
			 "Max cubemap texture width/height"),
	_DEVATTR(MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, INT,
			 "Max cubemap layered texture width/height"),
	_DEVATTR(MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, INT,
			 "Max layers in a cubemap layered texture"),
	_DEVATTR(MAXIMUM_SURFACE1D_WIDTH, INT,
			 "Max 1D surface width"),
	_DEVATTR(MAXIMUM_SURFACE2D_WIDTH, INT,
			 "Max 2D surface width"),
	_DEVATTR(MAXIMUM_SURFACE2D_HEIGHT, INT,
			 "Max 2D surface height"),
	_DEVATTR(MAXIMUM_SURFACE3D_WIDTH, INT,
			 "Max 3D surface width"),
	_DEVATTR(MAXIMUM_SURFACE3D_HEIGHT, INT,
			 "Max 3D surface height"),
	_DEVATTR(MAXIMUM_SURFACE3D_DEPTH, INT,
			 "Max 3D surface depth"),
	_DEVATTR(MAXIMUM_SURFACE1D_LAYERED_WIDTH, INT,
			 "Max 1D layered surface width"),
	_DEVATTR(MAXIMUM_SURFACE1D_LAYERED_LAYERS, INT,
			 "Max layers in a 1D layered surface"),
	_DEVATTR(MAXIMUM_SURFACE2D_LAYERED_WIDTH, INT,
			 "Max 2D layered surface width"),
	_DEVATTR(MAXIMUM_SURFACE2D_LAYERED_HEIGHT, INT,
			 "Max 2D layered surface height"),
	_DEVATTR(MAXIMUM_SURFACE2D_LAYERED_LAYERS, INT,
			 "Max layers in a 2D layered surface"),
	_DEVATTR(MAXIMUM_SURFACECUBEMAP_WIDTH, INT,
			 "Max cubemap surface width"),
	_DEVATTR(MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, INT,
			 "Max cubemap layered surface width"),
	_DEVATTR(MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, INT,
			 "Max layers in a cubemap layered surface"),
	_DEVATTR(MAXIMUM_TEXTURE1D_LINEAR_WIDTH, INT,
			 "Max 1D linear texture width"),
	_DEVATTR(MAXIMUM_TEXTURE2D_LINEAR_WIDTH, INT,
			 "Max 2D linear texture width"),
	_DEVATTR(MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, INT,
			 "Max 2D linear texture height"),
	_DEVATTR(MAXIMUM_TEXTURE2D_LINEAR_PITCH, INT,
			 "Max 2D linear texture pitch in bytes"),
	_DEVATTR(MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, INT,
			 "Max mipmapped 2D texture width"),
	_DEVATTR(MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, INT,
			 "Max mipmapped 2D texture height"),
	_DEVATTR(COMPUTE_CAPABILITY_MAJOR, INT,
			 "Major compute capability number"),
	_DEVATTR(COMPUTE_CAPABILITY_MINOR, INT,
			 "Minor compute capability number"),
	_DEVATTR(MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, INT,
			 "Max mipmapped 1D texture width"),
	_DEVATTR(STREAM_PRIORITIES_SUPPORTED, BOOL,
			 "Device supports stream priorities"),
	_DEVATTR(GLOBAL_L1_CACHE_SUPPORTED, BOOL,
			 "Device supports caching globals in L1"),
	_DEVATTR(LOCAL_L1_CACHE_SUPPORTED, BOOL,
			 "Device supports caching locals in L1"),
	_DEVATTR(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, SIZE,
			 "Max shared memory per multiprocessor"),
	_DEVATTR(MAX_REGISTERS_PER_MULTIPROCESSOR, INT,
			"Max number of 32bit registers per multiprocessor"),
	_DEVATTR(MANAGED_MEMORY, BOOL,
			 "Device can allocate managed memory"),
	_DEVATTR(MULTI_GPU_BOARD, BOOL,
			 "Device is on a multi-GPU board"),
	_DEVATTR(MULTI_GPU_BOARD_GROUP_ID, INT,
			 "Unique ID within a multi-GPU board"),
#undef _DEVATTR	
};

/*
 * collect_gpu_device_attributes
 */
static cl_int
collect_gpu_device_attributes(void)
{
	StringInfoData buf;
	CUdevice	dev;
	CUresult	rc;
	int			count;
	int			i, j, k;

	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "PG-Strom: failed on cuDeviceGetCount: %s", errorText(rc));
	if (count == 0)
		return 0;	/* no device found */

	devAttrs = malloc(sizeof(DevAttributes) * count);
	if (!devAttrs)
		elog(ERROR, "out of memory");

	initStringInfo(&buf);
	for (i=0, j=0; i < count; i++)
	{
		DevAttributes  *dattrs = &devAttrs[j];

		memset(dattrs, 0, sizeof(DevAttributesData));
		dattrs->DEV_ID = i;

		rc = cuDeviceGet(&dev, dattrs->DEV_ID);
		if (rc != CUDA_SUCCESS)
		{
			elog(LOG, "failed on cuDeviceGet, for device %d: %s",
				 i, errorText(rc));
			continue;
		}

		/* fill up DevAttributeData */
		rc = cuDeviceGetName(dattrs->DEV_NAME, sizeof(dattrs->DEV_NAME), dev);
		if (rc != CUDA_SUCCESS)
		{
			elog(LOG, "failed on cuDeviceGetName, for device %d: %s",
				 i, errorText(rc));
			continue;
		}

		rc = cuDeviceTotalMem(&dattrs->DEV_TOTAL_MEMSZ, dev);
		if (rc != CUDA_SUCCESS)
		{
			elog(LOG, "failed on cuDeviceTotalMem, for device %s: %s",
				 dattrs->DEV_NAME, errorText(rc));
			continue;
		}

		for (k=0; k < lengthof(DevAttrCatalog); k++)
		{
			rc = cuDeviceGetAttribute((int *)((char *)dattrs +
											  DevAttrCatalog[k].attr_offset),
									  DevAttrCatalog[k].attr_id, dev);
			if (rc != CUDA_SUCCESS)
			{
				elog(LOG, "failed on cuDeviceGetAttribute on %s: %s",
					 dattrs->DEV_NAME, errorText(rc));
				continue;
			}
		}

		if (dattrs->COMPUTE_CAPABILITY_MAJOR < 3 ||
			(dattrs->COMPUTE_CAPABILITY_MAJOR == 3 &&
			 dattrs->COMPUTE_CAPABILITY_MINOR < 5))
		{
			elog(LOG, "PG-Strom: GPU%d %s - capability %d.%d is not supported",
				 dattrs->DEV_ID,
				 dattrs->COMPUTE_CAPABILITY_MAJOR,
				 dattrs->COMPUTE_CAPABILITY_MINOR);
			continue;
		}

		/*
		 * Number of CUDA cores (determined by CC)
		 */
		if (dattrs->COMPUTE_CAPABILITY_MAJOR == 1)
			dattrs->CORES_PER_MPU = 8;
		else if (dattrs->COMPUTE_CAPABILITY_MAJOR == 2)
		{
			if (dattrs->COMPUTE_CAPABILITY_MINOR == 0)
				dattrs->CORES_PER_MPU = 32;
			else if (dattrs->COMPUTE_CAPABILITY_MINOR == 1)
				dattrs->CORES_PER_MPU = 48;
			else
				dattrs->CORES_PER_MPU = -1;
		}
		else if (dattrs->COMPUTE_CAPABILITY_MAJOR == 3)
			dattrs->CORES_PER_MPU = 192;
		else if (dattrs->COMPUTE_CAPABILITY_MAJOR == 5)
			dattrs->CORES_PER_MPU = 128;
		else if (dattrs->COMPUTE_CAPABILITY_MAJOR == 6)
		{
			if (dattrs->COMPUTE_CAPABILITY_MINOR == 0)
				dattrs->CORES_PER_MPU = 64;
			else if (dattrs->COMPUTE_CAPABILITY_MINOR == 1)
				dattrs->CORES_PER_MPU = 128;
			else
				dattrs->CORES_PER_MPU = -1;
		}
		else
			dattrs->CORES_PER_MPU = -1;		/* unknows */

		/* Log brief CUDA device properties */
		resetStringInfo(&buf);
		appendStringInfo(&buf, "GPU%d %s (", dattrs->DEV_ID, dattrs->DEV_NAME);
		if (dattrs->CORES_PER_MPU > 0)
			appendStringInfo(&buf, "%d CUDA cores",
							 dattrs->CORES_PER_MPU *
							 dattrs->MULTIPROCESSOR_COUNT);
		else
			appendStringInfo(&buf, "%d SMXs",
							 dattrs->MULTIPROCESSOR_COUNT);
		appendStringInfo(&buf, "), %dMHz, L2 %dkB",
						 dattrs->CLOCK_RATE / 1000,
						 dattrs->L2_CACHE_SIZE >> 10);
		if (dattrs->DEV_TOTAL_MEMSZ > (4UL << 30))
			appendStringInfo(&buf, ", RAM %.2fGB",
							 ((double)dattrs->DEV_TOTAL_MEMSZ /
							  (double)(1UL << 20)));
		else
			appendStringInfo(&buf, ", RAM %zuMB",
							 dattrs->DEV_TOTAL_MEMSZ >> 20);
		if (dattrs->MEMORY_CLOCK_RATE > (1UL << 20))
			appendStringInfo(&buf, " (%dbits, %.2fGHz)",
							 dattrs->GLOBAL_MEMORY_BUS_WIDTH,
							 ((double)dattrs->MEMORY_CLOCK_RATE /
							  (double)(1UL << 20)));
		else
			appendStringInfo(&buf, " (%dbits, %dMHz)",
							 dattrs->GLOBAL_MEMORY_BUS_WIDTH,
							 dattrs->MEMORY_CLOCK_RATE >> 10);
		appendStringInfo(&buf, ", capability %d.%d",
						 dattrs->COMPUTE_CAPABILITY_MAJOR,
						 dattrs->COMPUTE_CAPABILITY_MINOR);
		elog(LOG, "PG-Strom: %s", buf.data);
		j++;
	}
	pfree(buf.data);
	return j;	/* number of supported devices */
}

/*
 * pgstrom_init_gpu_device
 */
void
pgstrom_init_gpu_device(void)
{
	static char	   *cuda_visible_devices = NULL;
	CUresult		rc;

	/*
	 * Set CUDA_VISIBLE_DEVICES environment variable prior to CUDA
	 * initialization
	 */
	DefineCustomStringVariable("pg_strom.cuda_visible_devices",
							   "CUDA_VISIBLE_DEVICES environment variables",
							   NULL,
							   &cuda_visible_devices,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	if (cuda_visible_devices)
	{
		if (setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices, 1) != 0)
			elog(ERROR, "failed to set CUDA_VISIBLE_DEVICES");
	}

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "PG-Strom: failed on cuInit: %s", errorText(rc));

	numDevAttrs = collect_gpu_device_attributes();
	if (numDevAttrs == 0)
		elog(ERROR, "PG-Strom: no supported GPU devices found");
}

Datum
pgstrom_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	int				dindex;
	int				aindex;
	const char	   *att_name;
	const char	   *att_value;
	Datum			values[3];
	bool			isnull[3];
	HeapTuple		tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(3, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "attribute",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = 0;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	dindex = fncxt->call_cntr / (lengthof(DevAttrCatalog) + 3);
	aindex = fncxt->call_cntr % (lengthof(DevAttrCatalog) + 3);

	if (dindex >= numDevAttrs)
		SRF_RETURN_DONE(fncxt);
	dattrs = &devAttrs[dindex];

	if (aindex == 0)
	{
		att_name = "GPU Device ID";
		att_velue = psprintf("%d", dattrs->DEV_ID);
	}
	else if (aindex == 1)
	{
		att_name = "GPU Device Name";
		att_value = dattrs->DEV_NAME;
	}
	else if (aindex == 2)
	{
		att_name = "GPU Total RAM Size";
		att_value = format_bytesz(dattrs->DEV_TOTAL_MEMSZ);
	}
	else
	{
		int		i = aindex - 3;
		int		value = *((int *)((char *)dattrs +
								  DevAttrCatalog[i].attr_offset));

		att_name = DevAttrCatalog[i].attr_desc;
		switch (DevAttrCatalog[i].attkind)
		{
			case DEVATTRKIND__BOOL:
				att_value = psprintf("%s", value != 0 ? "True" : "False");
				break;
			case DEVATTRKIND__INT:
				att_value = psprintf("%d", value);
				break;
			case DEVATTRKIND__SIZE:
				att_value = format_bytesz((Size) value);
				break;
			case DEVATTRKIND__KHZ:
				if (value > 2 * 1000 * 1000)	/* more than 2.0GHz */
					att_value = psprintf("%.2f", (double) value / 1000000.0);
				else if (value > 2 * 1000)		/* more than 40MHz */
					att_value = psprintf("%d MHz", value / 1000);
				else
					att_value = psprintf("%d kHz", value);
				break;
			case DEVATTRKIND__COMP_MODE:
				if (value)
				{
					case CU_COMPUTEMODE_DEFAULT:
						att_value = "Default";
						break;
					case CU_COMPUTEMODE_EXCLUSIVE:
						att_value = "Exclusive";
						break;
					case CU_COMPUTEMODE_PROHIBITED:
						att_value = "Prohibited";
						break;
					case CU_COMPUTEMODE_EXCLUSIVE_PROCESS:
						att_value = "Exclusive Process";
						break;
					default:
						att_value = "Unknown";
						break;
				}
				break;
			case DEVATTRKIND__BITS:
				att_value = psprintf("%d bits", value);
				break;
			default:
				elog(ERROR, "Bug? unknown DevAttrKind: %d",
					 (int)DevAttrCatalog[i].attkind);
		}
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dindex);
	values[1] = CStringGetTextDatum(att_name);
	values[2] = CStringGetTextDatum(att_value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_device_info);
