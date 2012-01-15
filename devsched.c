/*
 * devsched.c
 *
 * Routines to make a plan to schedule computing resources being
 * optimized according to device properties.
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "pg_strom.h"

/*
 * PgStromDeviceInfo
 *
 * Data structure to store properties of devices. Every fields are initialized
 * at server starting up time; except for device and context.
 *
 */
typedef struct {
	CUdevice	device;
	CUcontext	context;
	char		dev_name[256];
	int			dev_major;
	int			dev_minor;
	int			dev_proc_nums;
	int			dev_proc_warp_sz;
	int			dev_proc_clock;
	size_t		dev_global_mem_sz;
	int			dev_global_mem_width;
	int			dev_global_mem_clock;
	int			dev_shared_mem_sz;
	int			dev_l2_cache_sz;
	int			dev_const_mem_sz;
	int			dev_max_block_dim_x;
	int			dev_max_block_dim_y;
	int			dev_max_block_dim_z;
	int			dev_max_grid_dim_x;
	int			dev_max_grid_dim_y;
	int			dev_max_grid_dim_z;
	int			dev_max_threads_per_proc;
	int			dev_max_regs_per_block;
	int			dev_integrated;
	int			dev_unified_addr;
	int			dev_can_map_hostmem;
	int			dev_concurrent_kernel;
	int			dev_concurrent_memcpy;
	int			dev_pci_busid;
	int			dev_pci_deviceid;
} PgStromDeviceInfo;

static bool					pgstrom_cuda_initialized = false;
static int					pgstrom_num_devices = 0;
static PgStromDeviceInfo   *pgstrom_device_info_data = NULL;

Datum
pgstrom_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	   *funcctx;
	PgStromDeviceInfo  *devinfo;
	StringInfoData		str;
	uint32		devindex;
	uint32		property;
	HeapTuple	tuple;
	Datum		values[3];
	bool		isnull[3];

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(3, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "devid",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "name",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "value",
						   TEXTOID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->user_fctx = NULL;

		MemoryContextSwitchTo(oldcxt);
	}
	funcctx = SRF_PERCALL_SETUP();

	devindex = PG_GETARG_UINT32(0);
	if (devindex == 0)
	{
		devindex = funcctx->call_cntr / 22;
		property = funcctx->call_cntr % 22;

		if (devindex >= pgstrom_num_devices)
			SRF_RETURN_DONE(funcctx);
	}
	else
	{
		if (devindex >= pgstrom_num_devices)
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("GPU device %d does not exist", devindex)));

		if (funcctx->call_cntr >= 22)
			SRF_RETURN_DONE(funcctx);
		property = funcctx->call_cntr;
	}

	devinfo = &pgstrom_device_info_data[devindex];
	initStringInfo(&str);

	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(devindex);

	switch (property)
	{
		case 0:
			values[1] = CStringGetTextDatum("name");
			appendStringInfo(&str, "%s", devinfo->dev_name);
			break;
		case 1:
			values[1] = CStringGetTextDatum("capability");
			appendStringInfo(&str, "%d.%d",
							 devinfo->dev_major, devinfo->dev_minor);
			break;
		case 2:
			values[1] = CStringGetTextDatum("num of procs");
			appendStringInfo(&str, "%d", devinfo->dev_proc_nums);
			break;
		case 3:
			values[1] = CStringGetTextDatum("wrap per proc");
			appendStringInfo(&str, "%d", devinfo->dev_proc_warp_sz);
			break;
		case 4:
			values[1] = CStringGetTextDatum("clock of proc");
			appendStringInfo(&str, "%d MHz", devinfo->dev_proc_clock / 1000);
			break;
		case 5:
			values[1] = CStringGetTextDatum("global mem size");
			appendStringInfo(&str, "%lu MB",
							 devinfo->dev_global_mem_sz / (1024 * 1024));
			break;
		case 6:
			values[1] = CStringGetTextDatum("global mem width");
			appendStringInfo(&str, "%d bits", devinfo->dev_global_mem_width);
			break;
		case 7:
			values[1] = CStringGetTextDatum("global mem clock");
			appendStringInfo(&str, "%d MHz",
							 devinfo->dev_global_mem_clock / 1000);
			break;
		case 8:
			values[1] = CStringGetTextDatum("shared mem size");
			appendStringInfo(&str, "%d KB", devinfo->dev_shared_mem_sz / 1024);
			break;
		case 9:
			values[1] = CStringGetTextDatum("L2 cache size");
			appendStringInfo(&str, "%d KB", devinfo->dev_l2_cache_sz / 1024);
			break;
		case 10:
			values[1] = CStringGetTextDatum("const mem size");
			appendStringInfo(&str, "%d KB", devinfo->dev_const_mem_sz / 1024);
			break;
		case 11:
			values[1] = CStringGetTextDatum("max block size");
			appendStringInfo(&str, "{%d, %d, %d}",
							 devinfo->dev_max_block_dim_x,
							 devinfo->dev_max_block_dim_y,
							 devinfo->dev_max_block_dim_z);
			break;
		case 12:
			values[1] = CStringGetTextDatum("max grid size");
			appendStringInfo(&str, "{%d, %d, %d}",
							 devinfo->dev_max_grid_dim_x,
							 devinfo->dev_max_grid_dim_y,
							 devinfo->dev_max_grid_dim_z);
			break;
		case 13:
			values[1] = CStringGetTextDatum("max threads per proc");
			appendStringInfo(&str, "%d", devinfo->dev_max_threads_per_proc);
			break;
		case 14:
			values[1] = CStringGetTextDatum("max registers per block");
			appendStringInfo(&str, "%d", devinfo->dev_max_regs_per_block);
			break;
		case 15:
			values[1] = CStringGetTextDatum("integrated memory");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_integrated ? "yes" : "no"));
			break;
		case 16:
			values[1] = CStringGetTextDatum("unified address");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_unified_addr ? "yes" : "no"));
			break;
		case 17:
			values[1] = CStringGetTextDatum("map host memory");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_can_map_hostmem ? "yes" : "no"));
			break;
		case 18:
			values[1] = CStringGetTextDatum("concurrent kernel");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_concurrent_kernel ? "yes" : "no"));
			break;
		case 19:
			values[1] = CStringGetTextDatum("concurrent memcpy");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_concurrent_memcpy ? "yes" : "no"));
			break;
		case 20:
			values[1] = CStringGetTextDatum("pci bus-id");
			appendStringInfo(&str, "%d", devinfo->dev_pci_busid);
			break;
		case 21:
			values[1] = CStringGetTextDatum("pci device-id");
			appendStringInfo(&str, "%d", devinfo->dev_pci_deviceid);
			break;
		default:
			elog(ERROR, "unexpected property : %d", property);
			break;
	}
	values[2] = CStringGetTextDatum(str.data);

	tuple = heap_form_tuple(funcctx->tuple_desc, values, isnull);

	pfree(str.data);
	SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_device_info);

int
pgstrom_get_num_devices(void)
{
	return pgstrom_num_devices;
}

void
pgstrom_set_device_context(int dev_index)
{
	CUresult	ret;

	if (!pgstrom_cuda_initialized)
	{
		cuInit(0);
		pgstrom_cuda_initialized = true;
	}

	Assert(dev_index < pgstrom_num_devices);
	if (!pgstrom_device_info_data[dev_index].context)
	{
		ret = cuCtxCreate(&pgstrom_device_info_data[dev_index].context,
						  0,
						  pgstrom_device_info_data[dev_index].device);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to create device context: %s",
				 cuda_error_to_string(ret));
	}
	ret = cuCtxSetCurrent(pgstrom_device_info_data[dev_index].context);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to switch device context: %s",
			 cuda_error_to_string(ret));
}

/*
 * pgstrom_devsched_init
 *
 * This routine collects properties of GPU devices being used to computing
 * schedule at server starting up time.
 * Note that cuInit(0) has to be called at the backend processes again to
 * avoid CUDA_ERROR_NOT_INITIALIZED errors.
 */
void
pgstrom_devsched_init(void)
{
	CUresult	ret;
	int			i, j;

	/*
	 * Initialize CUDA APIs
	 */
	ret = cuInit(0);
	if (ret != CUDA_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("cuda: failed to initialized APIs : %s",
						cuda_error_to_string(ret))));

	/*
	 * Collect properties of installed devices
	 */
	ret = cuDeviceGetCount(&pgstrom_num_devices);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to get number of devices : %s",
			 cuda_error_to_string(ret));

	pgstrom_device_info_data
		= MemoryContextAllocZero(TopMemoryContext,
								 sizeof(PgStromDeviceInfo) *
								 pgstrom_num_devices);
	for (i=0; i < pgstrom_num_devices; i++)
	{
		PgStromDeviceInfo  *devinfo;
		static struct {
			size_t				offset;
			CUdevice_attribute	attribute;
		} device_attrs[] = {
			{ offsetof(PgStromDeviceInfo, dev_proc_nums),
			  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT },
			{ offsetof(PgStromDeviceInfo, dev_proc_warp_sz),
			  CU_DEVICE_ATTRIBUTE_WARP_SIZE },
			{ offsetof(PgStromDeviceInfo, dev_proc_clock),
			  CU_DEVICE_ATTRIBUTE_CLOCK_RATE },
			{ offsetof(PgStromDeviceInfo, dev_global_mem_width),
			  CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH },
			{ offsetof(PgStromDeviceInfo, dev_global_mem_clock),
			  CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE },
			{ offsetof(PgStromDeviceInfo, dev_shared_mem_sz),
			  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK },
			{ offsetof(PgStromDeviceInfo, dev_l2_cache_sz),
			  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE },
			{ offsetof(PgStromDeviceInfo, dev_const_mem_sz),
			  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY },
			{ offsetof(PgStromDeviceInfo, dev_max_block_dim_x),
			  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X },
			{ offsetof(PgStromDeviceInfo, dev_max_block_dim_y),
			  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y },
			{ offsetof(PgStromDeviceInfo, dev_max_block_dim_z),
			  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z },
			{ offsetof(PgStromDeviceInfo, dev_max_grid_dim_x),
			  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X },
			{ offsetof(PgStromDeviceInfo, dev_max_grid_dim_y),
			  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y },
			{ offsetof(PgStromDeviceInfo, dev_max_grid_dim_z),
			  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z },
			{ offsetof(PgStromDeviceInfo, dev_max_regs_per_block),
			  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK },
			{ offsetof(PgStromDeviceInfo, dev_max_threads_per_proc),
			  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR },
			{ offsetof(PgStromDeviceInfo, dev_integrated),
			  CU_DEVICE_ATTRIBUTE_INTEGRATED },
			{ offsetof(PgStromDeviceInfo, dev_unified_addr),
			  CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING },
			{ offsetof(PgStromDeviceInfo, dev_can_map_hostmem),
			  CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY },
			{ offsetof(PgStromDeviceInfo, dev_concurrent_kernel),
			  CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS },
			{ offsetof(PgStromDeviceInfo, dev_concurrent_memcpy),
			  CU_DEVICE_ATTRIBUTE_GPU_OVERLAP},
			{ offsetof(PgStromDeviceInfo, dev_pci_busid),
			  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID },
			{ offsetof(PgStromDeviceInfo, dev_pci_deviceid),
			  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID },
		};

		devinfo = &pgstrom_device_info_data[i];

		if ((ret = cuDeviceGetName(devinfo->dev_name,
								   sizeof(devinfo->dev_name),
								   devinfo->device)) ||
			(ret = cuDeviceComputeCapability(&devinfo->dev_major,
											 &devinfo->dev_minor,
											 devinfo->device)) ||
			(ret = cuDeviceTotalMem(&devinfo->dev_global_mem_sz,
									devinfo->device)))
			elog(ERROR, "cuda: failed to get attribute of GPU device : %s",
				 cuda_error_to_string(ret));
		for (j=0; j < lengthof(device_attrs); j++)
		{
			ret = cuDeviceGetAttribute((int *)((uintptr_t) devinfo +
											   device_attrs[j].offset),
									   device_attrs[j].attribute,
									   devinfo->device);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "cuda: failed to get attribute of GPU device : %s",
					 cuda_error_to_string(ret));
		}

		/*
		 * Logs detected device properties
		 */
		elog(LOG, "PG-Strom: GPU device[%d] %s; capability v%d.%d, "
			 "%d of streaming processor units (%d wraps per unit, %dMHz), "
			 "%luMB of global memory (%d bits, %dMHz)",
			 i, devinfo->dev_name, devinfo->dev_major, devinfo->dev_minor,
			 devinfo->dev_proc_nums, devinfo->dev_proc_warp_sz,
			 devinfo->dev_proc_clock / 1000,
			 devinfo->dev_global_mem_sz / (1024 * 1024),
			 devinfo->dev_global_mem_width,
			 devinfo->dev_global_mem_clock / 1000);
	}
}
