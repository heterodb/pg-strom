/*
 * opencl_devinfo.c
 *
 * Routines to collect properties of OpenCL devices
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "nodes/pg_list.h"
#include "storage/barrier.h"
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"

/* GUC variables */
static int opencl_platform_index;

/* OpenCL resources for quick reference */
#define MAX_NUM_DEVICES		128

#define OPENCL_DEVINFO_SHM_LENGTH	(64 * 1024)	/* usually sufficient */
static struct {
	cl_uint			num_devices;
	pgstrom_platform_info  *pl_info;
	pgstrom_device_info	   *dev_info[FLEXIBLE_ARRAY_MEMBER];
} *opencl_devinfo_shm_values = NULL;

/* shmem call chain */
static shmem_startup_hook_type shmem_startup_hook_next;

/* quick references */
cl_platform_id		opencl_platform_id = NULL;
cl_context			opencl_context;
cl_device_type		opencl_device_types = 0;
cl_uint				opencl_num_devices;
cl_device_id		opencl_devices[MAX_NUM_DEVICES];
cl_command_queue	opencl_cmdq[MAX_NUM_DEVICES];
static List		   *opencl_valid_devices = NIL;

/*
 * Registration of OpenCL device info.
 */
#define CLPF_PARAM(param,field,is_cstring)								\
	{ (param), sizeof(((pgstrom_platform_info *) NULL)->field),			\
	  offsetof(pgstrom_platform_info, field), (is_cstring) }
#define CLDEV_PARAM(param,field,is_cstring)								\
	{ (param), sizeof(((pgstrom_device_info *) NULL)->field),			\
	  offsetof(pgstrom_device_info, field), (is_cstring) }

static pgstrom_device_info *
collect_opencl_device_info(cl_device_id device_id)
{
	pgstrom_device_info *dev_info;
	Size		offset = 0;
	Size		buflen = 10240;
	cl_int		i, rc;
	int			major, minor;
	static struct {
		cl_uint		param;
		size_t		size;
		size_t		offset;
		bool		is_cstring;
	} catalog[] = {
		CLDEV_PARAM(CL_DEVICE_ADDRESS_BITS,
					dev_address_bits, false),
		CLDEV_PARAM(CL_DEVICE_AVAILABLE,
					dev_available, false),
		CLDEV_PARAM(CL_DEVICE_COMPILER_AVAILABLE,
					dev_compiler_available, false),
/*
 * XXX - Bug? CUDA6.5 does not define this label
 *		CLDEV_PARAM(CL_DEVICE_DOUBLE_FP_CONFIG,
 *		dev_double_fp_config, false),
 */
		CLDEV_PARAM(CL_DEVICE_ENDIAN_LITTLE,
					dev_endian_little, false),
		CLDEV_PARAM(CL_DEVICE_ERROR_CORRECTION_SUPPORT,
					dev_error_correction_support, false),
		CLDEV_PARAM(CL_DEVICE_EXECUTION_CAPABILITIES,
					dev_execution_capabilities, false),
		CLDEV_PARAM(CL_DEVICE_EXTENSIONS,
					dev_device_extensions, true),
		CLDEV_PARAM(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
					dev_global_mem_cache_size, false),
		CLDEV_PARAM(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
					dev_global_mem_cache_type, false),
		CLDEV_PARAM(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
					dev_global_mem_cacheline_size, false),
		CLDEV_PARAM(CL_DEVICE_GLOBAL_MEM_SIZE,
					dev_global_mem_size, false),
		CLDEV_PARAM(CL_DEVICE_HOST_UNIFIED_MEMORY,
					dev_host_unified_memory, false),
		CLDEV_PARAM(CL_DEVICE_LOCAL_MEM_SIZE,
					dev_local_mem_size, false),
		CLDEV_PARAM(CL_DEVICE_LOCAL_MEM_TYPE,
					dev_local_mem_type, false),
		CLDEV_PARAM(CL_DEVICE_MAX_CLOCK_FREQUENCY,
					dev_max_clock_frequency, false),
		CLDEV_PARAM(CL_DEVICE_MAX_COMPUTE_UNITS,
					dev_max_compute_units, false),
		CLDEV_PARAM(CL_DEVICE_MAX_CONSTANT_ARGS,
					dev_max_constant_args, false),
		CLDEV_PARAM(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
					dev_max_constant_buffer_size, false),
		CLDEV_PARAM(CL_DEVICE_MAX_MEM_ALLOC_SIZE,
					dev_max_mem_alloc_size, false),
		CLDEV_PARAM(CL_DEVICE_MAX_PARAMETER_SIZE,
					dev_max_parameter_size, false),
		CLDEV_PARAM(CL_DEVICE_MAX_SAMPLERS,
					dev_max_samplers, false),
		CLDEV_PARAM(CL_DEVICE_MAX_WORK_GROUP_SIZE,
					dev_max_work_group_size, false),
		CLDEV_PARAM(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
					dev_max_work_item_dimensions, false),
		CLDEV_PARAM(CL_DEVICE_MAX_WORK_ITEM_SIZES,
					dev_max_work_item_sizes, false),
		CLDEV_PARAM(CL_DEVICE_MEM_BASE_ADDR_ALIGN,
					dev_mem_base_addr_align, false),
		CLDEV_PARAM(CL_DEVICE_NAME,
					dev_name, true),
		CLDEV_PARAM(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
					dev_native_vector_width_char, false),
		CLDEV_PARAM(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
					dev_native_vector_width_short, false),
		CLDEV_PARAM(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
					dev_native_vector_width_int, false),
		CLDEV_PARAM(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
					dev_native_vector_width_long, false),
		CLDEV_PARAM(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
					dev_native_vector_width_float, false),
		CLDEV_PARAM(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
					dev_native_vector_width_double, false),
		CLDEV_PARAM(CL_DEVICE_OPENCL_C_VERSION,
					dev_opencl_c_version, true),
		CLDEV_PARAM(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
					dev_preferred_vector_width_char, false),
		CLDEV_PARAM(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
					dev_preferred_vector_width_short, false),
		CLDEV_PARAM(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
					dev_preferred_vector_width_int, false),
		CLDEV_PARAM(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
					dev_preferred_vector_width_long, false),
		CLDEV_PARAM(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
					dev_preferred_vector_width_float, false),
		CLDEV_PARAM(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
					dev_preferred_vector_width_double, false),
		CLDEV_PARAM(CL_DEVICE_PROFILE,
					dev_profile, true),
		CLDEV_PARAM(CL_DEVICE_PROFILING_TIMER_RESOLUTION,
					dev_profiling_timer_resolution, false),
		CLDEV_PARAM(CL_DEVICE_QUEUE_PROPERTIES,
					dev_queue_properties, false),
		CLDEV_PARAM(CL_DEVICE_SINGLE_FP_CONFIG,
					dev_single_fp_config, false),
		CLDEV_PARAM(CL_DEVICE_TYPE,
					dev_type, false),
		CLDEV_PARAM(CL_DEVICE_VENDOR,
					dev_vendor, true),
		CLDEV_PARAM(CL_DEVICE_VENDOR_ID,
					dev_vendor_id, false),
		CLDEV_PARAM(CL_DEVICE_VERSION,
					dev_version, true),
		CLDEV_PARAM(CL_DRIVER_VERSION,
					driver_version, true)
	};

	dev_info = palloc(offsetof(pgstrom_device_info, buffer[buflen]));
	memset(dev_info, 0, sizeof(pgstrom_device_info));

	for (i=0; i < lengthof(catalog); i++)
	{
		size_t	param_size;
		size_t	param_retsz;
		char   *param_addr;

		if (!catalog[i].is_cstring)
		{
			param_size = catalog[i].size;
			param_addr = (char *)dev_info + catalog[i].offset;
		}
		else
		{
			Assert(catalog[i].size == sizeof(char *));
			param_size = buflen - offset;
			param_addr = &dev_info->buffer[offset];
		}

		rc = clGetDeviceInfo(device_id,
							 catalog[i].param,
							 param_size,
							 param_addr,
							 &param_retsz);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clGetDeviceInfo (param=%d, %s)",
				 catalog[i].param, opencl_strerror(rc));
		Assert(param_size == param_retsz || catalog[i].is_cstring);

		if (catalog[i].is_cstring)
		{
			param_addr[param_retsz] = '\0';
			*((char **)((char *)dev_info + catalog[i].offset)) = param_addr;
			offset += MAXALIGN(param_retsz);
		}
	}
	dev_info->buflen = offset;

	/*
	 * Check device capability is enough to run PG-Strom
	 */
	if (strcmp(dev_info->dev_profile, "FULL_PROFILE") != 0)
	{
		elog(LOG, "Profile of OpenCL device \"%s\" is \"%s\", skipped",
			 dev_info->dev_name, dev_info->dev_profile);
		goto out_clean;
	}
	if ((dev_info->dev_type & (CL_DEVICE_TYPE_CPU |
							   CL_DEVICE_TYPE_GPU |
							   CL_DEVICE_TYPE_ACCELERATOR)) == 0)
	{
		elog(LOG, "Only CPU, GPU or Accelerator are supported, skipped");
		goto out_clean;
	}
	if (!strstr(dev_info->dev_device_extensions, "cl_khr_fp64"))
	{
		elog(LOG, "OpenCL device has to support cl_khr_fp64 extension");
		goto out_clean;
	}
	if (!strstr(dev_info->dev_device_extensions,
				"cl_khr_byte_addressable_store"))
	{
		elog(LOG, "OpenCL device has to support cl_khr_byte_addressable_store extension");
		goto out_clean;
	}
	if (!dev_info->dev_available)
	{
		elog(LOG, "OpenCL device \"%s\" is not available, skipped",
			 dev_info->dev_name);
		goto out_clean;
	}
	if (!dev_info->dev_compiler_available)
	{
		elog(LOG, "OpenCL compiler of device \"%s\" is not available, skipped",
			 dev_info->dev_name);
		goto out_clean;
	}
	if (!dev_info->dev_endian_little)
	{
		elog(LOG, "OpenCL device \"%s\" has big endian, not supported",
			 dev_info->dev_name);
		goto out_clean;
	}
	if (sscanf(dev_info->dev_opencl_c_version, "OpenCL C %d.%d ",
			   &major, &minor) != 2 ||
		major < 1 || (major == 1 && minor < 1))
	{
		elog(LOG, "OpenCL C version of \"%s\"is too old \"%s\", skipped",
			 dev_info->dev_name, dev_info->dev_opencl_c_version);
		goto out_clean;
	}

	if (dev_info->dev_max_work_item_dimensions != 3)
	{
		elog(LOG, "OpenCL device \"%s\" has work item dimensions larger than 3, skipped",
			dev_info->dev_name);
		goto out_clean;
	}
	return dev_info;

out_clean:
	pfree(dev_info);
	return NULL;
}

static pgstrom_platform_info *
collect_opencl_platform_info(cl_platform_id platform_id)
{
	pgstrom_platform_info *pl_info;
	Size		offset = 0;
	Size		buflen = 10240;
	cl_int		i, rc;
	int			major, minor;
	static struct {
		cl_uint		param;
		size_t		size;
		size_t		offset;
		bool		is_cstring;
	} catalog[] = {
		CLPF_PARAM(CL_PLATFORM_PROFILE, pl_profile, true),
        CLPF_PARAM(CL_PLATFORM_VERSION, pl_version, true),
        CLPF_PARAM(CL_PLATFORM_NAME, pl_name, true),
        CLPF_PARAM(CL_PLATFORM_VENDOR, pl_vendor, true),
        CLPF_PARAM(CL_PLATFORM_EXTENSIONS, pl_extensions, true),
	};

	pl_info = palloc(offsetof(pgstrom_platform_info, buffer[buflen]));
	memset(pl_info, 0, sizeof(pgstrom_platform_info));

	/* collect platform properties */
	for (i=0; i < lengthof(catalog); i++)
	{
		size_t	param_size;
		size_t	param_retsz;
		char   *param_addr;

		if (!catalog[i].is_cstring)
		{
			param_size = catalog[i].size;
			param_addr = (char *)pl_info + catalog[i].offset;
		}
		else
		{
			Assert(catalog[i].size == sizeof(char *));
			param_size = buflen - offset;
			param_addr = &pl_info->buffer[offset];
		}

		rc = clGetPlatformInfo(platform_id,
							   catalog[i].param,
							   param_size,
							   param_addr,
							   &param_retsz);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clGetPlatformInfo (param=%d, %s)",
				 catalog[i].param, opencl_strerror(rc));
		Assert(param_size == param_retsz || catalog[i].is_cstring);

		if (catalog[i].is_cstring)
		{
			param_addr[param_retsz] = '\0';
			*((char **)((char *)pl_info + catalog[i].offset)) = param_addr;
			offset += MAXALIGN(param_retsz);
		}
	}
	pl_info->buflen = offset;

	if (strcmp(pl_info->pl_profile, "FULL_PROFILE") != 0)
	{
		elog(LOG, "Profile of OpenCL driver \"%s\" is \"%s\", skipped",
			 pl_info->pl_name, pl_info->pl_profile);
		goto out_clean;
	}

	if (sscanf(pl_info->pl_version, "OpenCL %d.%d ", &major, &minor) != 2 ||
		major < 1 || (major == 1 && minor < 1))
	{
		elog(LOG, "OpenCL version of \"%s\" is too old \"%s\", skipped",
			 pl_info->pl_name, pl_info->pl_version);
		goto out_clean;
	}
	return pl_info;

out_clean:
	pfree(pl_info);
	return NULL;
}

/*
 * pgstrom_opencl_device_info
 *
 * It dumps all the properties of OpenCL devices in user visible form.
 */
static char *
fp_config_to_cstring(cl_device_fp_config fp_conf)
{
	char	buf[256];
	int		ofs = 0;

	buf[0] = '\0';
	if ((fp_conf & CL_FP_DENORM) != 0)
		ofs += sprintf(buf + ofs, "%sDenorm", ofs > 0 ? ", " : "");
	if ((fp_conf & CL_FP_INF_NAN) != 0)
		ofs += sprintf(buf + ofs, "%sInf_NaN", ofs > 0 ? ", " : "");
	if ((fp_conf & CL_FP_ROUND_TO_NEAREST) != 0)
		ofs += sprintf(buf + ofs, "%sR-nearest", ofs > 0 ? ", " : "");
	if ((fp_conf & CL_FP_ROUND_TO_ZERO) != 0)
		ofs += sprintf(buf + ofs, "%sR-zero", ofs > 0 ? ", " : "");
	if ((fp_conf & CL_FP_ROUND_TO_INF) != 0)
		ofs += sprintf(buf + ofs, "%sR-inf", ofs > 0 ? ", " : "");
	if ((fp_conf & CL_FP_FMA) != 0)
		ofs += sprintf(buf + ofs, "%sFMA", ofs > 0 ? ", " : "");
	if ((fp_conf & CL_FP_SOFT_FLOAT) != 0)
		ofs += sprintf(buf + ofs, "%ssoft-float", ofs > 0 ? ", " : "");

	return pstrdup(buf);
}

static char *
memsize_to_cstring(Size memsize)
{
	if (memsize > 1UL << 43)
		return psprintf("%luTB", memsize >> 40);
	else if (memsize > 1UL << 33)
		return psprintf("%luGB", memsize >> 30);
	else if (memsize > 1UL << 23)
		return psprintf("%luMB", memsize >> 20);
	else if (memsize > 1UL << 13)
		return psprintf("%luKB", memsize >> 10);
	return psprintf("%lu", memsize);
}

Datum
pgstrom_opencl_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	*fncxt;
	Datum		values[4];
	bool		isnull[4];
	HeapTuple	tuple;
	uint32		dindex;
	uint32		pindex;
	const pgstrom_device_info *dinfo;
	const char *key;
	const char *value;
	char		buf[256];
	int			ofs = 0;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(4, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "dnum",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "pnum",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "property",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = 0;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	dindex = fncxt->call_cntr / 55;
	pindex = fncxt->call_cntr % 55;

	if (dindex == opencl_devinfo_shm_values->num_devices)
		SRF_RETURN_DONE(fncxt);

	dinfo = opencl_devinfo_shm_values->dev_info[dindex];
	Assert(dinfo != NULL);

	switch (pindex)
	{
		case 0:
			key = "platform index";
			value = psprintf("%u", dinfo->pl_info->pl_index);
			break;
		case 1:
			key = "platform profile";
			value = dinfo->pl_info->pl_profile;
			break;
		case 2:
			key = "platform version";
			value = dinfo->pl_info->pl_version;
			break;
		case 3:
			key = "platform name";
			value = dinfo->pl_info->pl_name;
			break;
		case 4:
			key = "platform vendor";
			value = dinfo->pl_info->pl_vendor;
			break;
		case 5:
			key = "platform extensions";
			value = dinfo->pl_info->pl_extensions;
			break;
		case 6:
			key = "address bits";
			value = psprintf("%u", dinfo->dev_address_bits);
			break;
		case 7:
			key = "device available";
			value = dinfo->dev_available ? "yes" : "no";
			break;
		case 8:
			key = "compiler available";
			value = dinfo->dev_compiler_available ? "yes" : "no";
			break;
		case 9:
			key = "double fp config";
			value = fp_config_to_cstring(dinfo->dev_double_fp_config);
			break;
		case 10:
			key = "little endian";
			value = dinfo->dev_endian_little ? "yes" : "no";
			break;
		case 11:
			key = "error correction support";
			value = dinfo->dev_error_correction_support ? "yes" : "no";
			break;
		case 12:
			key = "execution capabilities";
			if (dinfo->dev_execution_capabilities & CL_EXEC_KERNEL)
				ofs += sprintf(buf + ofs, "OpenCL");
			if (dinfo->dev_execution_capabilities & CL_EXEC_NATIVE_KERNEL)
				ofs += sprintf(buf + ofs, "%sNative", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 13:
			key = "device extensions";
			value = dinfo->dev_device_extensions;
			break;
		case 14:
			key = "global mem cache size";
			value = psprintf("%lu", dinfo->dev_global_mem_cache_size);
			break;
		case 15:
			key = "global mem cache type";
			switch (dinfo->dev_global_mem_cache_type)
			{
				case CL_NONE:
					value = "none";
					break;
				case CL_READ_ONLY_CACHE:
					value = "read only";
					break;
				case CL_READ_WRITE_CACHE:
					value = "read write";
					break;
				default:
					value = "???";
					break;
			}
			break;
		case 16:
			key = "global mem cacheline size";
			value = memsize_to_cstring(dinfo->dev_global_mem_cacheline_size);
			break;
		case 17:
			key = "global mem size";
			value = memsize_to_cstring(dinfo->dev_global_mem_size);
			break;
		case 18:
			key = "host unified memory";
			value = dinfo->dev_host_unified_memory ? "yes" : "no";
			break;
		case 19:
			key = "local mem size";
			value = memsize_to_cstring(dinfo->dev_local_mem_size);
			break;
		case 20:
			key = "local mem type";
			switch (dinfo->dev_local_mem_type)
			{
				case CL_LOCAL:
					value = "local";
					break;
				case CL_GLOBAL:
					value = "global";
					break;
				case CL_NONE:
					value = "none";
					break;
				default:
					value = "???";
					break;
			}
			break;
		case 21:
			key = "max clock frequency";
			value = psprintf("%u", dinfo->dev_max_clock_frequency);
			break;
		case 22:
			key = "max compute units";
			value = psprintf("%u", dinfo->dev_max_compute_units);
			break;
		case 23:
			key = "max constant args";
			value = psprintf("%u", dinfo->dev_max_constant_args);
			break;
		case 24:
			key = "max constant buffer size";
			value = memsize_to_cstring(dinfo->dev_max_constant_buffer_size);
			break;
		case 25:
			key = "max mem alloc size";
			value = memsize_to_cstring(dinfo->dev_max_mem_alloc_size);
			break;
		case 26:
			key = "max parameter size";
			value = psprintf("%lu", dinfo->dev_max_parameter_size);
			break;
		case 27:
			key = "max samplers";
			value = psprintf("%u", dinfo->dev_max_samplers);
			break;
		case 28:
			key = "max work group size";
			value = psprintf("%zu", dinfo->dev_max_work_group_size);
			break;
		case 29:
			key = "max work group dimensions";
			value = psprintf("%u", dinfo->dev_max_work_item_dimensions);
			break;
		case 30:
			key = "max work item sizes";
			value = psprintf("{%zu, %zu, %zu}",
							 dinfo->dev_max_work_item_sizes[0],
							 dinfo->dev_max_work_item_sizes[1],
							 dinfo->dev_max_work_item_sizes[2]);
			break;
		case 31:
			key = "mem base address align";
			value = psprintf("%u", dinfo->dev_mem_base_addr_align);
			break;
		case 32:
			key = "device name";
			value = dinfo->dev_name;
			break;
		case 33:
			key = "native vector width (char)";
			value = psprintf("%u", dinfo->dev_native_vector_width_char);
			break;
		case 34:
			key = "native vector width (short)";
			value = psprintf("%u", dinfo->dev_native_vector_width_short);
			break;
		case 35:
			key = "native vector width (int)";
			value = psprintf("%u", dinfo->dev_native_vector_width_int);
			break;
		case 36:
			key = "native vector width (long)";
			value = psprintf("%u", dinfo->dev_native_vector_width_long);
			break;
		case 37:
			key = "native vector width (float)";
			value = psprintf("%u", dinfo->dev_native_vector_width_float);
			break;
		case 38:
			key = "native vector width (double)";
			value = psprintf("%u", dinfo->dev_native_vector_width_double);
			break;
		case 39:
			key = "opencl c version";
			value = dinfo->dev_opencl_c_version;
			break;
		case 40:
			key = "preferred vector width (char)";
			value = psprintf("%u", dinfo->dev_preferred_vector_width_char);
			break;
		case 41:
			key = "preferred vector width (short)";
			value = psprintf("%u", dinfo->dev_preferred_vector_width_short);
			break;
		case 42:
			key = "preferred vector width (int)";
			value = psprintf("%u", dinfo->dev_preferred_vector_width_int);
			break;
		case 43:
			key = "preferred vector width (long)";
			value = psprintf("%u", dinfo->dev_preferred_vector_width_long);
			break;
		case 44:
			key = "preferred vector width (float)";
			value = psprintf("%u", dinfo->dev_preferred_vector_width_float);
			break;
		case 45:
			key = "preferred vector width (double)";
			value = psprintf("%u", dinfo->dev_preferred_vector_width_double);
			break;
		case 46:
			key = "device profile";
			value = dinfo->dev_profile;
			break;
		case 47:
			key = "profiling timer resolution";
			value = psprintf("%zu", dinfo->dev_profiling_timer_resolution);
			break;
		case 48:
			key = "command queue properties";
			if (dinfo->dev_queue_properties &
				CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
				ofs += sprintf(buf+ofs, "%sout-of-order", ofs > 0 ? ", " : "");
			if (dinfo->dev_queue_properties & CL_QUEUE_PROFILING_ENABLE)
				ofs += sprintf(buf+ofs, "%sprofiling", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 49:
			key = "single fp config";
			value = fp_config_to_cstring(dinfo->dev_single_fp_config);
			break;
		case 50:
			key = "device type";
			if (dinfo->dev_type & CL_DEVICE_TYPE_CPU)
				ofs += sprintf(buf, "%scpu", ofs > 0 ? ", " : "");
			if (dinfo->dev_type & CL_DEVICE_TYPE_GPU)
				ofs += sprintf(buf, "%sgpu", ofs > 0 ? ", " : "");
			if (dinfo->dev_type & CL_DEVICE_TYPE_ACCELERATOR)
				ofs += sprintf(buf, "%saccelerator", ofs > 0 ? ", " : "");
			if (dinfo->dev_type & CL_DEVICE_TYPE_DEFAULT)
				ofs += sprintf(buf, "%sdefault", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 51:
			key = "device vendor";
			value = dinfo->dev_vendor;
			break;
		case 52:
			key = "device vendor id";
			value = psprintf("%u", dinfo->dev_vendor_id);
			break;
		case 53:
			key = "device version";
			value = dinfo->dev_version;
			break;
		case 54:
			key = "driver version";
			value = dinfo->driver_version;
			break;
		default:
			elog(ERROR, "unexpected property index");
			break;
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dindex);
	values[1] = Int32GetDatum(pindex);
	values[2] = CStringGetTextDatum(key);
	values[3] = CStringGetTextDatum(value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_opencl_device_info);

/*
 * interface to access platform/device info
 */
int
pgstrom_get_device_nums(void)
{
	return opencl_devinfo_shm_values->num_devices;
}

const pgstrom_device_info *
pgstrom_get_device_info(unsigned int index)
{
	if (index < opencl_devinfo_shm_values->num_devices)
		return opencl_devinfo_shm_values->dev_info[index];
	return NULL;
}

/*
 * copy platform/device info into shared memory
 */
#define PLINFO_POINTER_SHIFT(pinfo, field)							\
	do {															\
		opencl_devinfo_shm_values->pl_info->field					\
			+= ((uintptr_t)opencl_devinfo_shm_values->pl_info -		\
				(uintptr_t)(pinfo));								\
	} while(0)

#define DEVINFO_POINTER_SHIFT(dinfo, i, field)						\
	do {															\
		opencl_devinfo_shm_values->dev_info[i]->field				\
			+= ((uintptr_t)opencl_devinfo_shm_values->dev_info[i] -	\
				(uintptr_t)(dinfo));								\
	} while(0)

static void
disclose_opencl_device_info(List *devinfo_list)
{
	pgstrom_platform_info  *pl_info = NULL;
	pgstrom_device_info	   *dev_info;
	cl_uint		num_devices;
	ListCell   *cell;
	Size		length;
	Size		offset;
	cl_uint		i = 0;

	num_devices = list_length(devinfo_list);

	length = ((Size)(opencl_devinfo_shm_values->dev_info + num_devices) -
			  (Size)(opencl_devinfo_shm_values));
	offset = length;
	foreach (cell, devinfo_list)
	{
		dev_info = lfirst(cell);

		if (cell == list_head(devinfo_list))
		{
			pl_info = dev_info->pl_info;
			length += MAXALIGN(offsetof(pgstrom_platform_info,
										buffer[pl_info->buflen]));
		}
		length += MAXALIGN(offsetof(pgstrom_device_info,
									buffer[dev_info->buflen]));
	}
	if (length >= OPENCL_DEVINFO_SHM_LENGTH)
		elog(ERROR, "usage of pgstrom_platform/device_info too large: %lu",
			 length);

	/* shows selected platform */
	Assert(pl_info != NULL);
	elog(LOG, "PG-Strom: Platform \"%s (%s)\" was installed",
		 pl_info->pl_name, pl_info->pl_version);

	/* copy platform/device info */
	foreach (cell, devinfo_list)
	{
		dev_info = lfirst(cell);
		elog(LOG, "PG-Strom: Device \"%s\" was installed", dev_info->dev_name);

		if (cell == list_head(devinfo_list))
		{
			pl_info = dev_info->pl_info;

			length = offsetof(pgstrom_platform_info,
							  buffer[pl_info->buflen]);
			memcpy((char *)opencl_devinfo_shm_values + offset,
				   pl_info, length);
			opencl_devinfo_shm_values->pl_info = (pgstrom_platform_info *)
				((char *)opencl_devinfo_shm_values + offset);
			offset += MAXALIGN(length);
			PLINFO_POINTER_SHIFT(pl_info, pl_profile);
			PLINFO_POINTER_SHIFT(pl_info, pl_version);
			PLINFO_POINTER_SHIFT(pl_info, pl_name);
			PLINFO_POINTER_SHIFT(pl_info, pl_vendor);
			PLINFO_POINTER_SHIFT(pl_info, pl_extensions);
		}
		length = offsetof(pgstrom_device_info,
						  buffer[dev_info->buflen]);
		memcpy((char *)opencl_devinfo_shm_values + offset,
			   dev_info, length);
		opencl_devinfo_shm_values->dev_info[i] = (pgstrom_device_info *)
			((char *)opencl_devinfo_shm_values + offset);
		offset += MAXALIGN(length);
		opencl_devinfo_shm_values->dev_info[i]->pl_info
			= opencl_devinfo_shm_values->pl_info;
		DEVINFO_POINTER_SHIFT(dev_info, i, dev_device_extensions);
		DEVINFO_POINTER_SHIFT(dev_info, i, dev_name);
		DEVINFO_POINTER_SHIFT(dev_info, i, dev_opencl_c_version);
		DEVINFO_POINTER_SHIFT(dev_info, i, dev_profile);
		DEVINFO_POINTER_SHIFT(dev_info, i, dev_vendor);
		DEVINFO_POINTER_SHIFT(dev_info, i, dev_version);
		DEVINFO_POINTER_SHIFT(dev_info, i, driver_version);
		i++;
	}
	pg_memory_barrier();
	opencl_devinfo_shm_values->num_devices = num_devices;
}

/*
 * Routines to get device properties.
 */
void
construct_opencl_device_info(void)
{
	cl_platform_id	platforms[32];
	cl_device_id	devices[MAX_NUM_DEVICES];
	cl_uint			n_platform;
	cl_uint			n_devices;
	pgstrom_platform_info *pl_info;
	pgstrom_device_info	*dev_info;
	cl_int			i, j, k, rc;
	long			score_max = -1;
	List		   *result = NIL;
	ListCell	   *lc;

	rc = clGetPlatformIDs(lengthof(platforms),
						  platforms,
						  &n_platform);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clGetPlatformIDs failed (%s)", opencl_strerror(rc));
	if (n_platform == 0)
		elog(ERROR, "No OpenCL platforms available");

	for (i=0; i < n_platform; i++)
	{
		long		score = 0;
		List	   *temp = NIL;
		List	   *cleanup = NIL;

		pl_info = collect_opencl_platform_info(platforms[i]);
		pl_info->pl_index = i;

		elog(LOG, "PG-Strom: [%d] OpenCL Platform: %s", i, pl_info->pl_name);
		if (opencl_platform_index >= 0 && opencl_platform_index != i)
		{
			pfree(pl_info);
			continue;
		}

		/* Get list of device ids */
		rc = clGetDeviceIDs(platforms[i],
							opencl_device_types,
							lengthof(devices),
							devices,
							&n_devices);
		if (rc != CL_SUCCESS && rc != CL_DEVICE_NOT_FOUND)
			elog(ERROR, "clGetDeviceIDs failed (%s)", opencl_strerror(rc));

		/* any devices available on this platform? */
		if (rc == CL_DEVICE_NOT_FOUND || n_devices == 0)
		{
			if (i == opencl_platform_index)
				elog(ERROR, "no device available on platform \"%s\"",
					 pl_info->pl_name);
			pfree(pl_info);
			continue;
		}

		for (j=0, k=0; j < n_devices; j++)
		{
			dev_info = collect_opencl_device_info(devices[j]);
			if (!dev_info)
				continue;
			dev_info->pl_info = pl_info;
			dev_info->dev_index = k;

			elog(LOG, "PG-Strom: (%d:%d) Device %s (%uMHz x %uunits, %luMB)",
				 i, j,
				 dev_info->dev_name,
				 dev_info->dev_max_clock_frequency,
				 dev_info->dev_max_compute_units,
				 dev_info->dev_global_mem_size >> 20);

			if (opencl_valid_devices)
			{
				foreach (lc, opencl_valid_devices)
				{
					if (lfirst_int(lc) == j)
						break;
				}
				if (!lc)
				{
					pfree(dev_info);
					continue;
				}
			}

			/* rough estimation about computing power */
			if ((dev_info->dev_type & CL_DEVICE_TYPE_GPU) != 0)
				score += 32 * (dev_info->dev_max_compute_units *
							   dev_info->dev_max_clock_frequency);
			else
				score += (dev_info->dev_max_compute_units *
						  dev_info->dev_max_clock_frequency);

			temp = lappend(temp, dev_info);
			devices[k] = devices[j];
			k++;
		}

		if (k > 0 &&
			(opencl_platform_index == i ||
			 (opencl_platform_index < 0 && score > score_max)))
		{
			opencl_platform_id = platforms[i];
			opencl_num_devices = k;
			memcpy(opencl_devices, devices, sizeof(cl_device_id) * k);
			score_max = score;
			cleanup = result;
			result = temp;
		}
		else
			cleanup = temp;

		if (cleanup != NIL)
		{
			pl_info = ((pgstrom_device_info *) linitial(cleanup))->pl_info;
			pfree(pl_info);
			list_free_deep(cleanup);
		}
	}

	/* show platform name if auto-selection */
	if (!opencl_platform_id)
		elog(ERROR,
			 "PG-Strom: No OpenCL device available. "
			 "Please check \"pg_strom.opencl_platform\" parameter");

	/* OK, let's put device/platform information on shared memory */
	disclose_opencl_device_info(result);
}

static void
pgstrom_startup_opencl_devinfo(void)
{
	bool	found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* reserved area for opencl device info */
	opencl_devinfo_shm_values = ShmemInitStruct("opencl_devinfo_shm_values",
												OPENCL_DEVINFO_SHM_LENGTH,
												&found);
	Assert(!found);
	memset(opencl_devinfo_shm_values, 0, sizeof(*opencl_devinfo_shm_values));
}

void
pgstrom_init_opencl_devinfo(void)
{
	static char *devnums_strings;
	static char	*devtype_strings;
	char		*token;
	char		*pos;

	/* selection of opencl platform */
	DefineCustomIntVariable("pg_strom.opencl_platform",
							"selection of OpenCL platform to be used",
							NULL,
							&opencl_platform_index,
							-1,		/* auto selection */
							-1,
							INT_MAX,
							PGC_POSTMASTER,
                            GUC_NOT_IN_SAMPLE,
                            NULL, NULL, NULL);
	/* selection of opencl devices */
	DefineCustomStringVariable("pg_strom.opencl_devices",
							   "selection of OpenCL devices to be used",
							   NULL,
							   &devnums_strings,
							   "any",
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	token = strtok_r(devnums_strings, ", ", &pos);
	while (token != NULL)
	{
		int		code;

		if (strcmp(token, "any") == 0)
		{
			opencl_valid_devices = NIL;
			break;
		}
		code = atoi(token);
		if (code < 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("invalid pg_strom.opencl_devices option: %s",
							token),
					 errhint("must be 'any' or non-negative integer")));

		opencl_valid_devices = lappend_int(opencl_valid_devices, code);
		token = strtok_r(NULL, ", ", &pos);
	}

	/* selection of opencl device types */
	DefineCustomStringVariable("pg_strom.opencl_device_types",
							   "OpenCL device filter based on device types",
							   NULL,
							   &devtype_strings,
							   "gpu,accelerator",
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	token = strtok_r(devtype_strings, ", ", &pos);
	while (token != NULL)
	{
		if (strcmp(token, "cpu") == 0)
			opencl_device_types |= CL_DEVICE_TYPE_CPU;
		else if (strcmp(token, "gpu") == 0)
			opencl_device_types |= CL_DEVICE_TYPE_GPU;
		else if (strcmp(token, "accelerator") == 0 ||
				 strcmp(token, "mic") == 0)
			opencl_device_types |= CL_DEVICE_TYPE_ACCELERATOR;
		else if (strcmp(token, "any") == 0)
			opencl_device_types |= CL_DEVICE_TYPE_ALL;
		else
			ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("invalid pgstrom.opencl_device_types option: %s",
							pos)));
		token = strtok_r(NULL, ", ", &pos);
	}
	if (opencl_device_types == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("no valid opencl device types are specified")));

	/* shared memory request to store device info */
	RequestAddinShmemSpace(OPENCL_DEVINFO_SHM_LENGTH);
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_opencl_devinfo;
}

/*
 * clserv_compute_workgroup_size
 *
 * It computes an optimal workgroup size for the supplied kernel object.
 * Workgroup size is usually restricted by resource consumption (like
 * registers, local memory, ...). Alignment is also significant because
 * some reduction logic assumes workgroup size is power-of-two larger
 * than or equal to 32 (= width of cl_uint).
 * It depends on the context whether larger workgroup-size is better,
 * or smaller. So, caller shall give a hint of "large_is_better".
 * Local memory is consumed by two different types of variables; one
 * is static variables described in the kernel function, the other
 * one is dynamic one being supplied as kernel arguments.
 * Right now, we assume 1KB is a fair estimation for the consumption
 * by static local variables, even though we ask OpenCL how much local
 * memory is consumed by this kernel. In case when we want to apply
 * same global/local workgroup size for multiple kernels (see gpupreagg.c),
 * it enables to reduce number of workgroup size estimation.
 */
#define MINIMUM_LOCALMEM_CONSUMPTION	1024
#define MINIMUM_WORKGROUP_UNITSZ			(sizeof(cl_uint) * BITS_PER_BYTE)

bool
clserv_compute_workgroup_size(size_t *p_gwork_sz,
							  size_t *p_lwork_sz,
							  cl_kernel kernel,
							  int dev_index,
							  bool larger_is_better,
							  size_t num_threads,
							  size_t local_memsz_per_thread)
{
	const pgstrom_device_info *devinfo;
	cl_device_id kdevice;
	size_t		max_workgroup_sz;
	size_t		unitsz;
	size_t		local_usage;
	size_t		adjusted;
	size_t		lwork_sz;
	cl_int		rc;

	Assert(pgstrom_i_am_clserv);

	kdevice = opencl_devices[dev_index];

	/*
	 * Get a suggested maximum workgroup size by rum-time.
	 * It does not pay attention about dynamically allocated local
	 * memory, so we need additional estimation for local memory
	 * to avoid over consumption.
	 */
	rc = clGetKernelWorkGroupInfo(kernel,
								  kdevice,
								  CL_KERNEL_WORK_GROUP_SIZE,
								  sizeof(size_t),
								  &max_workgroup_sz,
								  NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clGetKernelWorkGroupInfo: %s",
				   opencl_strerror(rc));
		return false;
	}
	/* we expect max_workgroup_sz is power of two */
	if ((max_workgroup_sz & (max_workgroup_sz - 1)) != 0)
		max_workgroup_sz = (1UL << (get_next_log2(max_workgroup_sz + 1) - 1));

	/*
	 * Get a preferred unit size of workgroup to be launched.
	 * It's a performance hint. PG-Strom expects workgroup size is
	 * multiplexer of MINIMUM_WORKGROUP_UNITSZ, so it has to be
	 * adjusted if needed.
	 */
	rc = clGetKernelWorkGroupInfo(kernel,
								  kdevice,
								  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
								  sizeof(size_t),
								  &unitsz,
								  NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clGetKernelWorkGroupInfo: %s",
				   opencl_strerror(rc));
		return false;
	}
	/* we have no idea if runtime told a unitsz not power of two */
	Assert((unitsz & (unitsz - 1)) == 0);

	/*
	 * We may need to adjust maximum workgroup size according to
	 * the consumption of local memory usage.
     */
	rc = clGetKernelWorkGroupInfo(kernel,
								  kdevice,
								  CL_KERNEL_LOCAL_MEM_SIZE,
								  sizeof(local_usage),
								  &local_usage,
								  NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clGetKernelWorkGroupInfo: %s",
				   opencl_strerror(rc));
		return false;
	}
	local_usage = Max(MINIMUM_LOCALMEM_CONSUMPTION, local_usage);

	devinfo = pgstrom_get_device_info(dev_index);
	adjusted = (devinfo->dev_local_mem_size -
				local_usage) / local_memsz_per_thread;
	if (adjusted < max_workgroup_sz)
		max_workgroup_sz = (1UL << (get_next_log2(adjusted + 1) - 1));

	/*
	 * Determine local workgroup size according to the policy
	 */
	if (larger_is_better)
		lwork_sz = max_workgroup_sz;
	else
		lwork_sz = Min(unitsz, max_workgroup_sz);
	Assert((lwork_sz & (lwork_sz - 1)) == 0);

	*p_lwork_sz = lwork_sz;
	*p_gwork_sz = TYPEALIGN(lwork_sz, num_threads);

	/*
	 * TODO: needs to put optimal workgroup size for each
	 * GPU models. Kernel execution time is affected so much.
	 */
	return true;
}
