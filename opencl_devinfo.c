/*
 * opencl_devinfo.c
 *
 * Routines to collect properties of OpenCL devices
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "nodes/pg_list.h"
#include "utils/builtins.h"
#include "pg_strom.h"

/*
 * Registration of OpenCL device info.
 */
#define CLPF_PARAM(param,field,is_cstring)								\
	{ (param), sizeof(((pgstrom_device_info *) NULL)->field),			\
			offsetof(pgstrom_device_info, field), true, (is_cstring) }
#define CLDEV_PARAM(param,field,is_cstring)								\
	{ (param), sizeof(((pgstrom_device_info *) NULL)->field),			\
			offsetof(pgstrom_device_info, field), false, (is_cstring)}

static pgstrom_device_info *
init_opencl_device_info(cl_platform_id platform, cl_device_id device)
{
	pgstrom_device_info *devinfo;
	Size		offset = 0;
	Size		buflen = 10240;
	cl_int		i, rc;
	static struct {
		cl_uint		param;
		size_t		size;
		size_t		offset;
		bool		is_platform;
		bool		is_cstring;
	} catalog[] = {
		CLPF_PARAM(CL_PLATFORM_PROFILE, pl_profile, true),
		CLPF_PARAM(CL_PLATFORM_VERSION, pl_version, true),
		CLPF_PARAM(CL_PLATFORM_NAME, pl_name, true),
		CLPF_PARAM(CL_PLATFORM_VENDOR, pl_vendor, true),
		CLPF_PARAM(CL_PLATFORM_EXTENSIONS, pl_extensions, true),
		CLDEV_PARAM(CL_DEVICE_ADDRESS_BITS,
					dev_address_bits, false),
		CLDEV_PARAM(CL_DEVICE_AVAILABLE,
					dev_available, false),
		CLDEV_PARAM(CL_DEVICE_BUILT_IN_KERNELS,
					dev_built_in_kernels, true),
		CLDEV_PARAM(CL_DEVICE_COMPILER_AVAILABLE,
					dev_compiler_available, false),
		CLDEV_PARAM(CL_DEVICE_DOUBLE_FP_CONFIG,
					dev_double_fp_config, false),
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
		CLDEV_PARAM(CL_DEVICE_LINKER_AVAILABLE,
					dev_linker_available, false),
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
		CLDEV_PARAM(CL_DEVICE_PRINTF_BUFFER_SIZE,
					dev_printf_buffer_size, false),
		CLDEV_PARAM(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
					dev_preferred_interop_user_sync, false),
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

	devinfo = palloc(offsetof(pgstrom_device_info, buffer[buflen]));
	memset(devinfo, 0, sizeof(pgstrom_device_info));

	for (i=0; i < lengthof(catalog); i++)
	{
		size_t	param_size;
		size_t	param_retsz;
		void   *param_addr;

		if (!catalog[i].is_cstring)
		{
			param_size = catalog[i].size;
			param_addr = (char *)devinfo + catalog[i].offset;
		}
		else
		{
			Assert(catalog[i].size == sizeof(char *));
			param_size = buflen - offset;
			param_addr = &devinfo->buffer[offset];
		}

		if (catalog[i].is_platform)
		{
			rc = clGetPlatformInfo(platform,
								   catalog[i].param,
								   param_size,
								   param_addr,
								   &param_retsz);
			if (rc != CL_SUCCESS)
				elog(ERROR, "failed on clGetPlatformInfo (param=%d, %s)",
					 catalog[i].param, opencl_strerror(rc));
			Assert(param_size == param_retsz || catalog[i].is_cstring);
		}
		else
		{
			rc = clGetDeviceInfo(device,
								 catalog[i].param,
								 param_size,
								 param_addr,
								 &param_retsz);
			if (rc != CL_SUCCESS)
				elog(ERROR, "failed on clGetDeviceInfo (param=%d, %s)",
					 catalog[i].param, opencl_strerror(rc));
			Assert(param_size == param_retsz || catalog[i].is_cstring);
		}

		if (catalog[i].is_cstring)
		{
			((char *)param_addr)[param_retsz] = '\0';
			offset += MAXALIGN(param_retsz);
		}
	}
	devinfo->buflen = offset;

	/*
	 * Check whether the detected device has enough capability we expect
	 */
	if (strcmp(devinfo->pl_profile, "FULL_PROFILE") != 0)
	{
		elog(LOG, "Profile of OpenCL driver \"%s\" is \"%s\", skipped",
			 devinfo->pl_name, devinfo->pl_profile);
		goto out_clean;
	}
	if (!strstr(devinfo->pl_extensions, "cl_khr_icd"))
	{
		elog(LOG, "OpenCL driver \"%s\" does not support \"cl_khr_icd\" extension to control multiple drivers (extensions: %s), skipped",
			 devinfo->pl_name, devinfo->pl_extensions);
		goto out_clean;
	}
	if (strcmp(devinfo->dev_profile, "FULL_PROFILE") != 0)
	{
		elog(LOG, "Profile of OpenCL device \"%s\" is \"%s\", skipped",
			 devinfo->dev_name, devinfo->dev_profile);
		goto out_clean;
	}
	if ((devinfo->dev_type & (CL_DEVICE_TYPE_CPU |
							  CL_DEVICE_TYPE_GPU |
							  CL_DEVICE_TYPE_ACCELERATOR)) == 0)
	{
		elog(LOG, "Only CPU, GPU or Accelerator are supported, skipped");
		goto out_clean;
	}
	if (!devinfo->dev_available)
	{
		elog(LOG, "OpenCL device \"%s\" is not available, skipped",
			 devinfo->dev_name);
		goto out_clean;
	}
	if (!devinfo->dev_compiler_available)
	{
		elog(LOG, "OpenCL compiler of device \"%s\" is not available, skipped",
			 devinfo->dev_name);
		goto out_clean;
	}
#ifdef WORDS_BIGENDIAN
	if (devinfo->dev_endian_little)
	{
		elog(LOG, "OpenCL device \"%s\" has little endian, unlike host",
			 devinfo->dev_name);
		goto out_clean;
	}
#else
	if (!devinfo->dev_endian_little)
	{
		elog(LOG, "OpenCL device \"%s\" has big endian, unlike host",
			 devinfo->dev_name);
		goto out_clean;
	}
#endif
	if (devinfo->dev_max_work_item_dimensions != 3)
	{
		elog(LOG, "OpenCL device \"%s\" has work item dimensions larger than 3, skipped",
			devinfo->dev_name);
		goto out_clean;
	}
	return devinfo;

out_clean:
	pfree(devinfo);
	return NULL;
}

static List *
init_opencl_platform_info(cl_platform_id platform)
{
	pgstrom_device_info *devinfo;
	cl_device_id	devices[128];
	cl_uint			n_device;
	cl_int			i, rc;
	List		   *result = NIL;

	rc = clGetDeviceIDs(platform,
						CL_DEVICE_TYPE_DEFAULT,
						lengthof(devices),
						devices,
						&n_device);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clGetDeviceIDs failed (%s)", opencl_strerror(rc));

	for (i=0; i < n_device; i++)
	{
		devinfo = init_opencl_device_info(platform, devices[i]);

		if (devinfo)
			result = lappend(result, devinfo);
	}
	return result;
}

/*
 * pgstrom_init_opencl_device_info
 *
 * It gathers properties of OpenCL devices. It should be called once, by
 * the worker process that manages OpenCL interactions.
 *
 */
void
pgstrom_init_opencl_device_info(void)
{
	cl_platform_id	platforms[32];
	cl_uint			n_platform;
	cl_int			i, rc;
	List		   *result = NIL;

	rc = clGetPlatformIDs(lengthof(platforms),
						  platforms,
						  &n_platform);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clGetPlatformIDs failed (%s)", opencl_strerror(rc));

	for (i=0; i < n_platform; i++)
	{
		List   *temp
			= init_opencl_platform_info(platforms[i]);

		result = list_concat(result, temp);
	}
	pgstrom_register_device_info(result);

	list_free_deep(result);
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

Datum
pgstrom_opencl_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	*fncxt;
	Datum		values[3];
	bool		isnull[3];
	HeapTuple	tuple;
	uint32		dindex;
	uint32		pindex;
	pgstrom_device_info *devinfo;
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

		tupdesc = CreateTemplateTupleDesc(3, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "index",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "property",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = 0;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	dindex = fncxt->call_cntr / 100;
	pindex = fncxt->call_cntr % 100;

	if (dindex == pgstrom_get_device_nums())
		SRF_RETURN_DONE(fncxt);

	devinfo = pgstrom_get_device_info(dindex);
	Assert(devinfo != NULL);

	switch (pindex)
	{
		case 0:
			key = "platform index";
			value = psprintf("%u", devinfo->pl_index);
			break;
		case 1:
			key = "platform profile";
			value = devinfo->pl_profile;
			break;
		case 2:
			key = "platform version";
			value = devinfo->pl_version;
			break;
		case 3:
			key = "platform name";
			value = devinfo->pl_name;
			break;
		case 4:
			key = "platform vendor";
			value = devinfo->pl_vendor;
			break;
		case 5:
			key = "platform extensions";
			value = devinfo->pl_extensions;
			break;
		case 6:
			key = "address bits";
			value = psprintf("%u", devinfo->dev_address_bits);
			break;
		case 7:
			key = "device available";
			value = devinfo->dev_available ? "yes" : "no";
			break;
		case 8:
			key = "built in kernels";
			value = devinfo->dev_built_in_kernels;
			break;
		case 9:
			key = "compiler available";
			value = devinfo->dev_compiler_available ? "yes" : "no";
			break;
		case 10:
			key = "double fp config";
			value = fp_config_to_cstring(devinfo->dev_double_fp_config);
			break;
		case 11:
			key = "little endian";
			value = devinfo->dev_endian_little ? "yes" : "no";
			break;
		case 12:
			key = "error correction support";
			value = devinfo->dev_error_correction_support ? "yes" : "no";
			break;
		case 13:
			key = "execution capabilities";
			if (devinfo->dev_execution_capabilities & CL_EXEC_KERNEL)
				ofs += sprintf(buf + ofs, "OpenCL");
			if (devinfo->dev_execution_capabilities & CL_EXEC_NATIVE_KERNEL)
				ofs += sprintf(buf + ofs, "%sNative", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 14:
			key = "device extensions";
			value = devinfo->dev_device_extensions;
			break;
		case 15:
			key = "global mem cache size";
			value = psprintf("%lu", devinfo->dev_global_mem_cache_size);
			break;
		case 16:
			key = "global mem cache type";
			switch (devinfo->dev_global_mem_cache_type)
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
		case 17:
			key = "global mem cacheline size";
			value = psprintf("%u", devinfo->dev_global_mem_cacheline_size);
			break;
		case 18:
			key = "global mem size";
			value = psprintf("%lu", devinfo->dev_global_mem_size);
			break;
		case 19:
			key = "host unified memory";
			value = devinfo->dev_host_unified_memory ? "yes" : "no";
			break;
		case 20:
			key = "linker available";
			value = devinfo->dev_linker_available ? "yes" : "no";
			break;
		case 21:
			key = "local mem size";
			value = psprintf("%lu", devinfo->dev_local_mem_size);
			break;
		case 22:
			key = "local mem type";
			switch (devinfo->dev_local_mem_type)
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
		case 23:
			key = "max clock frequency";
			value = psprintf("%u", devinfo->dev_max_clock_frequency);
			break;
		case 24:
			key = "max compute units";
			value = psprintf("%u", devinfo->dev_max_compute_units);
			break;
		case 25:
			key = "max constant args";
			value = psprintf("%u", devinfo->dev_max_constant_args);
			break;
		case 26:
			key = "max constant buffer size";
			value = psprintf("%lu", devinfo->dev_max_constant_buffer_size);
			break;
		case 27:
			key = "max mem alloc size";
			value = psprintf("%lu", devinfo->dev_max_mem_alloc_size);
			break;
		case 28:
			key = "max parameter size";
			value = psprintf("%lu", devinfo->dev_max_parameter_size);
			break;
		case 29:
			key = "max samplers";
			value = psprintf("%u", devinfo->dev_max_samplers);
			break;
		case 30:
			key = "max work group size";
			value = psprintf("%zu", devinfo->dev_max_work_group_size);
			break;
		case 31:
			key = "max work group dimensions";
			value = psprintf("%u", devinfo->dev_max_work_item_dimensions);
			break;
		case 32:
			key = "max work item sizes";
			value = psprintf("{%zu, %zu, %zu}",
							 devinfo->dev_max_work_item_sizes[0],
							 devinfo->dev_max_work_item_sizes[1],
							 devinfo->dev_max_work_item_sizes[2]);
			break;
		case 33:
			key = "mem base address align";
			value = psprintf("%u", devinfo->dev_mem_base_addr_align);
			break;
		case 34:
			key = "device name";
			value = devinfo->dev_name;
			break;
		case 35:
			key = "native vector width (char)";
			value = psprintf("%u", devinfo->dev_native_vector_width_char);
			break;
		case 36:
			key = "native vector width (short)";
			value = psprintf("%u", devinfo->dev_native_vector_width_short);
			break;
		case 37:
			key = "native vector width (int)";
			value = psprintf("%u", devinfo->dev_native_vector_width_int);
			break;
		case 38:
			key = "native vector width (long)";
			value = psprintf("%u", devinfo->dev_native_vector_width_long);
			break;
		case 39:
			key = "native vector width (float)";
			value = psprintf("%u", devinfo->dev_native_vector_width_float);
			break;
		case 40:
			key = "native vector width (double)";
			value = psprintf("%u", devinfo->dev_native_vector_width_double);
			break;
		case 41:
			key = "opencl c version";
			value = devinfo->dev_opencl_c_version;
			break;
		case 42:
			key = "preferred vector width (char)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_char);
			break;
		case 43:
			key = "preferred vector width (short)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_short);
			break;
		case 44:
			key = "preferred vector width (int)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_int);
			break;
		case 45:
			key = "preferred vector width (long)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_long);
			break;
		case 46:
			key = "preferred vector width (float)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_float);
			break;
		case 47:
			key = "preferred vector width (double)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_double);
			break;
		case 48:
			key = "printf buffer size";
			value = psprintf("%zu", devinfo->dev_printf_buffer_size);
			break;
		case 49:
			key = "preferred interop user sync";
			value = devinfo->dev_preferred_interop_user_sync ? "yes" : "no";
			break;
		case 50:
			key = "device profile";
			value = devinfo->dev_profile;
			break;
		case 51:
			key = "profiling timer resolution";
			value = psprintf("%zu", devinfo->dev_profiling_timer_resolution);
			break;
		case 52:
			key = "command queue properties";
			if (devinfo->dev_queue_properties &
				CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
				ofs += sprintf(buf, "%sout of order", ofs > 0 ? ", " : "");
			if (devinfo->dev_queue_properties & CL_QUEUE_PROFILING_ENABLE)
				ofs += sprintf(buf, "%sprofiling", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 53:
			key = "single fp config";
			value = fp_config_to_cstring(devinfo->dev_single_fp_config);
			break;
		case 54:
			key = "device type";
			if (devinfo->dev_type & CL_DEVICE_TYPE_CPU)
				ofs += sprintf(buf, "%scpu", ofs > 0 ? ", " : "");
			if (devinfo->dev_type & CL_DEVICE_TYPE_GPU)
				ofs += sprintf(buf, "%sgpu", ofs > 0 ? ", " : "");
			if (devinfo->dev_type & CL_DEVICE_TYPE_ACCELERATOR)
				ofs += sprintf(buf, "%saccelerator", ofs > 0 ? ", " : "");
			if (devinfo->dev_type & CL_DEVICE_TYPE_DEFAULT)
				ofs += sprintf(buf, "%sdefault", ofs > 0 ? ", " : "");
			if (devinfo->dev_type & CL_DEVICE_TYPE_CUSTOM)
				ofs += sprintf(buf, "%scustom", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 55:
			key = "device vendor";
			value = devinfo->dev_vendor;
			break;
		case 56:
			key = "device vendor id";
			value = psprintf("%u", devinfo->dev_vendor_id);
			break;
		case 57:
			key = "device version";
			value = devinfo->dev_version;
			break;
		case 58:
			key = "driver version";
			value = devinfo->driver_version;
			break;
		default:
			elog(ERROR, "unexpected property index");
			break;
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dindex);
	values[1] = CStringGetTextDatum(key);
	values[2] = CStringGetTextDatum(value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_opencl_device_info);
