/*
 * devinfo.c
 *
 * Collect properties of OpenCL processing units
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "utils/memutils.h"
#include "pg_strom.h"

typedef struct {
	cl_platform_id	pf_id;
	char			pf_profile[24];
	char			pf_version[80];
	char			pf_name[80];
	char			pf_vendor[80];
	char			pf_extensions[256];
} PgStromPlatformInfo;

typedef struct {
	PgStromPlatformInfo		   *pf_info;
	cl_device_id				dev_id;
	cl_uint						dev_address_bits;
	cl_bool						dev_available;
	cl_bool						dev_compiler_available;
	cl_device_fp_config			dev_double_fp_config;
	cl_bool						dev_endian_little;
	cl_bool						dev_error_correction_support;
	cl_device_exec_capabilities	dev_execution_capabilities;
	cl_ulong					dev_global_mem_cache_size;
	cl_device_mem_cache_type	dev_global_mem_cache_type;
	cl_uint						dev_global_mem_cacheline_size;
	cl_ulong					dev_global_mem_size;
	cl_ulong					dev_local_mem_size;
	cl_device_local_mem_type	dev_local_mem_type;
	cl_uint						dev_max_clock_frequency;
	cl_uint						dev_max_compute_units;
	cl_uint						dev_max_constant_args;
	cl_ulong					dev_max_constant_buffer_size;
	cl_ulong					dev_max_mem_alloc_size;
	size_t						dev_max_parameter_size;
	size_t						dev_max_work_group_size;
	size_t						dev_max_work_item_dimensions;
	size_t						dev_max_work_item_sizes[3];
	char						dev_name[256];
	char						dev_version[256];
	char						dev_profile[24];
} PgStromDeviceInfo;

List   *pgstrom_device_info_list = NIL;

void
pgstrom_device_info_init(void)
{
	PgStromPlatformInfo	*pf_info;
	PgStromDeviceInfo *dev_info;
	cl_platform_id	platform_ids[32];
	cl_device_id	device_ids[64];
	cl_uint			num_platforms;
	cl_uint			num_devices;
	cl_int			ret, pi, di;
	MemoryContext	oldctx;

	oldctx = MemoryContextSwitchTo(TopMemoryContext);

	ret = clGetPlatformIDs(lengthof(platform_ids),
						   platform_ids, &num_platforms);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: filed to get number of platforms");

	for (pi=0; pi < num_platforms; pi++)
	{
		pf_info = palloc(sizeof(PgStromPlatformInfo));
		pf_info->pf_id = platform_ids[pi];
		if (clGetPlatformInfo(pf_info->pf_id,
							  CL_PLATFORM_PROFILE,
							  sizeof(pf_info->pf_profile),
							  pf_info->pf_profile,NULL) != CL_SUCCESS ||
			clGetPlatformInfo(pf_info->pf_id,
							  CL_PLATFORM_VERSION,
							  sizeof(pf_info->pf_version),
							  pf_info->pf_version, NULL) != CL_SUCCESS ||
			clGetPlatformInfo(pf_info->pf_id,
							  CL_PLATFORM_NAME,
							  sizeof(pf_info->pf_name),
                              pf_info->pf_name, NULL) != CL_SUCCESS ||
			clGetPlatformInfo(pf_info->pf_id,
							  CL_PLATFORM_VENDOR,
							  sizeof(pf_info->pf_vendor),
							  pf_info->pf_vendor, NULL) != CL_SUCCESS ||
			clGetPlatformInfo(pf_info->pf_id,
							  CL_PLATFORM_EXTENSIONS,
							  sizeof(pf_info->pf_extensions),
							  pf_info->pf_extensions, NULL) != CL_SUCCESS)
			elog(ERROR, "OpenCL: failed to get properties of platforms");

		ret = clGetDeviceIDs(platform_ids[pi],
							 CL_DEVICE_TYPE_DEFAULT,
							 lengthof(device_ids),
							 device_ids, &num_devices);
		if (ret != CL_SUCCESS)
			elog(ERROR, "OpenCL: filed to get number of devices");

		for (di=0; di < num_devices; di++)
		{
			dev_info = palloc(sizeof(PgStromDeviceInfo));
			dev_info->pf_info = pf_info;
			dev_info->dev_id = device_ids[di];

			if (clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_ADDRESS_BITS,
								sizeof(dev_info->dev_address_bits),
								&dev_info->dev_address_bits,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_AVAILABLE,
								sizeof(dev_info->dev_available),
								&dev_info->dev_available,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_COMPILER_AVAILABLE,
								sizeof(dev_info->dev_compiler_available),
								&dev_info->dev_compiler_available,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_DOUBLE_FP_CONFIG,
								sizeof(dev_info->dev_double_fp_config),
								&dev_info->dev_double_fp_config,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_ENDIAN_LITTLE,
								sizeof(dev_info->dev_endian_little),
								&dev_info->dev_endian_little,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_ERROR_CORRECTION_SUPPORT,
								sizeof(dev_info->dev_error_correction_support),
								&dev_info->dev_error_correction_support,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_EXECUTION_CAPABILITIES,
								sizeof(dev_info->dev_execution_capabilities),
								&dev_info->dev_execution_capabilities,
								NULL) != CL_SUCCESS ||
#if 0
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_EXTENSIONS,
								sizeof(dev_info->dev_extensions),
								dev_info->dev_extensions,
								NULL) != CL_SUCCESS ||
#endif
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
								sizeof(dev_info->dev_global_mem_cache_size),
								&dev_info->dev_global_mem_cache_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
								sizeof(dev_info->dev_global_mem_cache_type),
								&dev_info->dev_global_mem_cache_type,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
							sizeof(dev_info->dev_global_mem_cacheline_size),
								&dev_info->dev_global_mem_cacheline_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_GLOBAL_MEM_SIZE,
								sizeof(dev_info->dev_global_mem_size),
								&dev_info->dev_global_mem_size,
								NULL) != CL_SUCCESS ||
#if 0
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_HALF_FP_CONFIG,
								sizeof(dev_info->dev_half_fp_config),
								&dev_info->dev_half_fp_config,
								NULL != CL_SUCCESS) ||
#endif
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_LOCAL_MEM_SIZE,
								sizeof(dev_info->dev_local_mem_size),
								&dev_info->dev_local_mem_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_LOCAL_MEM_TYPE,
								sizeof(dev_info->dev_local_mem_type),
								&dev_info->dev_local_mem_type,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_CLOCK_FREQUENCY,
								sizeof(dev_info->dev_max_clock_frequency),
								&dev_info->dev_max_clock_frequency,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_COMPUTE_UNITS,
								sizeof(dev_info->dev_max_compute_units),
								&dev_info->dev_max_compute_units,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_CONSTANT_ARGS,
								sizeof(dev_info->dev_max_constant_args),
								&dev_info->dev_max_constant_args,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
								sizeof(dev_info->dev_max_constant_buffer_size),
								&dev_info->dev_max_constant_buffer_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_MEM_ALLOC_SIZE,
								sizeof(dev_info->dev_max_mem_alloc_size),
								&dev_info->dev_max_mem_alloc_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_PARAMETER_SIZE,
								sizeof(dev_info->dev_max_parameter_size),
								&dev_info->dev_max_parameter_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_WORK_GROUP_SIZE,
								sizeof(dev_info->dev_max_work_group_size),
								&dev_info->dev_max_work_group_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
								sizeof(dev_info->dev_max_work_item_dimensions),
								&dev_info->dev_max_work_item_dimensions,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_WORK_ITEM_SIZES,
								sizeof(dev_info->dev_max_work_item_sizes),
								dev_info->dev_max_work_item_sizes,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_NAME,
								sizeof(dev_info->dev_name),
								dev_info->dev_name,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_VERSION,
								sizeof(dev_info->dev_version),
								&dev_info->dev_version,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_PROFILE,
								sizeof(dev_info->dev_profile),
								dev_info->dev_profile,
								NULL) != CL_SUCCESS)
				elog(ERROR, "OpenCL: failed to get properties of device");
			/*
			 * Print properties of the device into log
			 */
			elog(LOG,
				 "pg_strom: device %s (%s), %u of units (%uMHz), "
				 "%luMB device memory, %luKB of cache memory%s",
				 dev_info->dev_name,
				 dev_info->dev_version,
				 dev_info->dev_max_compute_units,
				 dev_info->dev_max_clock_frequency,
				 dev_info->dev_global_mem_size / (1024 * 1024),
				 dev_info->dev_global_mem_cache_size / 1024,
				 (dev_info->dev_compiler_available ?
				  ", run-time compile supported" : ""));

			pgstrom_device_info_list
				= lappend(pgstrom_device_info_list, dev_info);
		}
	}
	MemoryContextSwitchTo(oldctx);
}
