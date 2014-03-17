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
	{ (param), sizeof(((pgstrom_platform_info *) NULL)->field),			\
	  offsetof(pgstrom_platform_info, field), (is_cstring) }
#define CLDEV_PARAM(param,field,is_cstring)								\
	{ (param), sizeof(((pgstrom_device_info *) NULL)->field),			\
	  offsetof(pgstrom_device_info, field), (is_cstring) }

static pgstrom_device_info *
init_opencl_device_info(pgstrom_platform_info *pl_info, cl_device_id device_id)
{
	pgstrom_device_info *dev_info;
	Size		offset = 0;
	Size		buflen = 10240;
	cl_int		i, rc;
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
#ifdef WORDS_BIGENDIAN
	if (dev_info->dev_endian_little)
	{
		elog(LOG, "OpenCL device \"%s\" has little endian, unlike host",
			 dev_info->dev_name);
		goto out_clean;
	}
#else
	if (!dev_info->dev_endian_little)
	{
		elog(LOG, "OpenCL device \"%s\" has big endian, unlike host",
			 dev_info->dev_name);
		goto out_clean;
	}
#endif
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
	dev_info->pl_info = pl_info;
	dev_info->device_id = device_id;

	return dev_info;

out_clean:
	pfree(dev_info);
	return NULL;
}

static List *
init_opencl_platform_info(cl_platform_id platform_id)
{
	pgstrom_platform_info *pl_info;
	cl_device_id device_ids[128];
	cl_uint		n_device;
	Size		offset = 0;
	Size		buflen = 10240;
	cl_int		i, rc;
	int			major, minor;
	List	   *result = NIL;
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
			param_addr = &devinfo->buffer[offset];
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
	pl_info->platform_id = platform_id;

	if (strcmp(pl_info->pl_name, "FULL_PROFILE") != 0)
	{
		elog(LOG, "Profile of OpenCL driver \"%s\" is \"%s\", skipped",
			 pl_info->pl_name, pl_info->pl_profile);
		goto out_clean;
	}

	if (sscanf(pl_info->pl_version, "OpenCL %d.%d ", &major, &minor) != 2 ||
		major < 1 || (major == 1 && minor < 1))
	{
		elog(LOG, "OpenCL version of \"%s\" is too old \"%s\", skipped",
			 pl_info->pl_version);
		goto out_clean;
	}

	rc = clGetDeviceIDs(platform_id,
						CL_DEVICE_TYPE_DEFAULT,
						lengthof(device_ids),
						device_ids,
						&n_device);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clGetDeviceIDs failed (%s)", opencl_strerror(rc));

	for (i=0; i < n_device; i++)
	{
		pgstrom_device_info	   *devinfo
			= init_opencl_device_info(pl_info, device_ids[i]);

		if (devinfo)
			result = lappend(result, devinfo);
	}
	if (result != NIL)
		return result;

out_clean:
	pfree(pl_info);
	return NIL;
}

/*
 * pgstrom_collect_opencl_device_info
 *
 * It collects properties of all the OpenCL devices. It shall be called once
 * by the OpenCL management worker process, prior to any other backends.
 */
List *
pgstrom_collect_opencl_device_info(int pl_index)
{
	cl_platform_id	platforms[32];
	cl_uint			n_platform;
	cl_int			i, rc;
	int				score_max = -1;
	List		   *result = NIL;
	ListCell	   *cell;

	rc = clGetPlatformIDs(lengthof(platforms),
						  platforms,
						  &n_platform);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clGetPlatformIDs failed (%s)", opencl_strerror(rc));

	for (i=0; i < n_platform; i++)
	{
		pgstrom_platform_info  *pl_info;
		pgstrom_device_info	   *dev_info;
		List	   *temp;
		int			score = 0;

		temp = init_opencl_platform_info(platforms[i]);
		if (temp == NIL)
			continue;

		dev_info = linitial(temp);
		pl_info = dev_info->pl_info;
		
		elog(LOG, "PG-Strom: [%d] OpenCL Platform - \"%s\"", i, pl_info->pl_name);
		if (pl_index < 0)
		{
			int		score = 0;

			foreach (cell, temp)
			{
				dev_info = lfirst(cell);

				score += (dev_info->dev_max_compute_units *
						  dev_info->dev_max_clock_frequency *
						  (dev_info->dev_type & CL_DEVICE_TYPE_GPU != 0 ? 32 : 1));
			}
			if (score > score_max)
			{
				score_max = score;
				result = temp;
			}
		}
		else if (pl_index == i)
			result = temp;

		/* shows device properties */
		foreach (cell, temp)
		{
			elog(LOG, "PG-Strom: %c device %s (%uMHz x %uunits, %luMB)"
				 cell != llast(temp) ? '+' : '`',
				 devinfo->dev_name,
                 devinfo->dev_max_clock_frequency,
                 devinfo->dev_max_compute_units,
                 devinfo->dev_global_mem_size >> 20);
		}
	}

	if (result != NIL)
	{
		cl_device_id   *devices = alloca(list_length(result));
		cl_context		context;
		cl_context_properties properties[2];

		/*
		 * Create an OpenCL context
		 */
		i = 0;
		foreach (cell, result)
		{
			pgstrom_device_info	   *dev_info = lfirst(cell);
			devices[i++] = dev_info->device_id;
		}
		context = clCreateContext(i, devices, NULL, NULL, &rc);
		if (rc != CL_SUCCESS)
			elog(ERROR, "clCreateContext failed: %s", opencl_strerror(rc));

		/*
		 * Create an OpenCL command queue for each device
		 */
		foreach (cell, result)
		{
			pgstrom_device_info	   *dev_info = lfirst(cell);
			pgstrom_platform_info  *pl_info = dev_info->pl_info;

			pl_info->context = context;
			dev_info->cmdq =
				clCreateCommandQueue(context,
									 dev_info->device_id,
									 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
									 CL_QUEUE_PROFILING_ENABLE,
									 &rc);
			if (rc != CL_SUCCESS)
				elog(ERROR, "clCreateCommandQueue failed on \"%s\": %s",
					 dev_info->dev_name, opencl_strerror(rc));
		}
	}
	return result;
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
	Datum		values[4];
	bool		isnull[4];
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
			key = "compiler available";
			value = devinfo->dev_compiler_available ? "yes" : "no";
			break;
		case 9:
			key = "double fp config";
			value = fp_config_to_cstring(devinfo->dev_double_fp_config);
			break;
		case 10:
			key = "little endian";
			value = devinfo->dev_endian_little ? "yes" : "no";
			break;
		case 11:
			key = "error correction support";
			value = devinfo->dev_error_correction_support ? "yes" : "no";
			break;
		case 12:
			key = "execution capabilities";
			if (devinfo->dev_execution_capabilities & CL_EXEC_KERNEL)
				ofs += sprintf(buf + ofs, "OpenCL");
			if (devinfo->dev_execution_capabilities & CL_EXEC_NATIVE_KERNEL)
				ofs += sprintf(buf + ofs, "%sNative", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 13:
			key = "device extensions";
			value = devinfo->dev_device_extensions;
			break;
		case 14:
			key = "global mem cache size";
			value = psprintf("%lu", devinfo->dev_global_mem_cache_size);
			break;
		case 15:
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
		case 16:
			key = "global mem cacheline size";
			value = psprintf("%u", devinfo->dev_global_mem_cacheline_size);
			break;
		case 17:
			key = "global mem size";
			value = psprintf("%lu", devinfo->dev_global_mem_size);
			break;
		case 18:
			key = "host unified memory";
			value = devinfo->dev_host_unified_memory ? "yes" : "no";
			break;
		case 19:
			key = "local mem size";
			value = psprintf("%lu", devinfo->dev_local_mem_size);
			break;
		case 20:
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
		case 21:
			key = "max clock frequency";
			value = psprintf("%u", devinfo->dev_max_clock_frequency);
			break;
		case 22:
			key = "max compute units";
			value = psprintf("%u", devinfo->dev_max_compute_units);
			break;
		case 23:
			key = "max constant args";
			value = psprintf("%u", devinfo->dev_max_constant_args);
			break;
		case 24:
			key = "max constant buffer size";
			value = psprintf("%lu", devinfo->dev_max_constant_buffer_size);
			break;
		case 25:
			key = "max mem alloc size";
			value = psprintf("%lu", devinfo->dev_max_mem_alloc_size);
			break;
		case 26:
			key = "max parameter size";
			value = psprintf("%lu", devinfo->dev_max_parameter_size);
			break;
		case 27:
			key = "max samplers";
			value = psprintf("%u", devinfo->dev_max_samplers);
			break;
		case 28:
			key = "max work group size";
			value = psprintf("%zu", devinfo->dev_max_work_group_size);
			break;
		case 29:
			key = "max work group dimensions";
			value = psprintf("%u", devinfo->dev_max_work_item_dimensions);
			break;
		case 30:
			key = "max work item sizes";
			value = psprintf("{%zu, %zu, %zu}",
							 devinfo->dev_max_work_item_sizes[0],
							 devinfo->dev_max_work_item_sizes[1],
							 devinfo->dev_max_work_item_sizes[2]);
			break;
		case 31:
			key = "mem base address align";
			value = psprintf("%u", devinfo->dev_mem_base_addr_align);
			break;
		case 32:
			key = "device name";
			value = devinfo->dev_name;
			break;
		case 33:
			key = "native vector width (char)";
			value = psprintf("%u", devinfo->dev_native_vector_width_char);
			break;
		case 34:
			key = "native vector width (short)";
			value = psprintf("%u", devinfo->dev_native_vector_width_short);
			break;
		case 35:
			key = "native vector width (int)";
			value = psprintf("%u", devinfo->dev_native_vector_width_int);
			break;
		case 36:
			key = "native vector width (long)";
			value = psprintf("%u", devinfo->dev_native_vector_width_long);
			break;
		case 37:
			key = "native vector width (float)";
			value = psprintf("%u", devinfo->dev_native_vector_width_float);
			break;
		case 38:
			key = "native vector width (double)";
			value = psprintf("%u", devinfo->dev_native_vector_width_double);
			break;
		case 39:
			key = "opencl c version";
			value = devinfo->dev_opencl_c_version;
			break;
		case 40:
			key = "preferred vector width (char)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_char);
			break;
		case 41:
			key = "preferred vector width (short)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_short);
			break;
		case 42:
			key = "preferred vector width (int)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_int);
			break;
		case 43:
			key = "preferred vector width (long)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_long);
			break;
		case 44:
			key = "preferred vector width (float)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_float);
			break;
		case 45:
			key = "preferred vector width (double)";
			value = psprintf("%u", devinfo->dev_preferred_vector_width_double);
			break;
		case 46:
			key = "device profile";
			value = devinfo->dev_profile;
			break;
		case 47:
			key = "profiling timer resolution";
			value = psprintf("%zu", devinfo->dev_profiling_timer_resolution);
			break;
		case 48:
			key = "command queue properties";
			if (devinfo->dev_queue_properties &
				CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
				ofs += sprintf(buf, "%sout of order", ofs > 0 ? ", " : "");
			if (devinfo->dev_queue_properties & CL_QUEUE_PROFILING_ENABLE)
				ofs += sprintf(buf, "%sprofiling", ofs > 0 ? ", " : "");
			value = buf;
			break;
		case 49:
			key = "single fp config";
			value = fp_config_to_cstring(devinfo->dev_single_fp_config);
			break;
		case 50:
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
		case 51:
			key = "device vendor";
			value = devinfo->dev_vendor;
			break;
		case 52:
			key = "device vendor id";
			value = psprintf("%u", devinfo->dev_vendor_id);
			break;
		case 53:
			key = "device version";
			value = devinfo->dev_version;
			break;
		case 54:
			key = "driver version";
			value = devinfo->driver_version;
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
