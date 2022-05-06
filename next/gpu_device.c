/*
 * gpu_device.c
 *
 * Routines to collect GPU device information.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* variable declarations */
GpuDevAttributes *gpuDevAttrs = NULL;
int			numGpuDevAttrs = 0;

/* catalog of device attributes */
typedef enum {
	DEVATTRKIND__INT,
	DEVATTRKIND__BYTES,
	DEVATTRKIND__KB,
	DEVATTRKIND__KHZ,
	DEVATTRKIND__COMPUTEMODE,
	DEVATTRKIND__BOOL,
	DEVATTRKIND__BITS,
} DevAttrKind;

static struct {
	CUdevice_attribute	attr_id;
	DevAttrKind	attr_kind;
	size_t		attr_offset;
	const char *attr_desc;
} GpuDevAttrCatalog[] = {
#define DEV_ATTR(LABEL,KIND,a,DESC)				\
	{ CU_DEVICE_ATTRIBUTE_##LABEL,				\
	  DEVATTRKIND__##KIND,						\
	  offsetof(struct GpuDevAttributes, LABEL),	\
	  DESC },
#include "gpu_devattrs.h"
#undef DEV_ATTR
};

/* declaration */
Datum pgstrom_gpu_device_info(PG_FUNCTION_ARGS);

/* static variables */
static bool		gpudirect_driver_is_initialized = false;
static bool		__pgstrom_gpudirect_enabled;	/* GUC */
static int		__pgstrom_gpudirect_threshold;	/* GUC */
static int		devBaselineMaxThreadsPerBlock = INT_MAX;

/*
 * pgstrom_gpudirect_enabled
 */
bool
pgstrom_gpudirect_enabled(void)
{
	return __pgstrom_gpudirect_enabled;
}

/*
 * pgstrom_gpudirect_enabled_checker
 */
static bool
pgstrom_gpudirect_enabled_checker(bool *p_newval, void **extra, GucSource source)
{
	bool	newval = *p_newval;

	if (newval && !gpudirect_driver_is_initialized)
		elog(ERROR, "cannot enable GPUDirectSQL without driver module loaded");
	return true;
}

/*
 * pgstrom_gpudirect_threshold
 */
Size
pgstrom_gpudirect_threshold(void)
{
	return (Size)__pgstrom_gpudirect_threshold << 10;
}

/*
 * pgstrom_collect_gpu_device
 */
static bool
pgstrom_collect_gpu_device(void)
{
	StringInfoData str;
	const char *cmdline = (CMD_GPUINFO_PATH " -md");
	char		linebuf[2048];
	FILE	   *filp;
	char	   *tok_attr;
	char	   *tok_val;
	char	   *pos;
	char	   *cuda_runtime_version = NULL;
	char	   *nvidia_driver_version = NULL;
	int			num_devices = -1;	/* total num of GPUs; incl legacy models */
	int			i, cuda_dindex;

	Assert(numGpuDevAttrs == 0);
	filp = OpenPipeStream(cmdline, PG_BINARY_R);
	if (!filp)
		return false;

	initStringInfo(&str);
	while (fgets(linebuf, sizeof(linebuf), filp) != NULL)
	{
		/* trim '\n' on the tail */
		pos = linebuf + strlen(linebuf);
		while (pos > linebuf && isspace(*--pos))
			*pos = '\0';
		/* empty line? */
		if (linebuf[0] == '\0')
			continue;

		tok_attr = strchr(linebuf, ':');
		if (!tok_attr)
			elog(ERROR, "unexpected gpuinfo -md format");
		*tok_attr++ = '\0';

		tok_val = strchr(tok_attr, '=');
		if (!tok_val)
			elog(ERROR, "incorrect gpuinfo -md format");
		*tok_val++ = '\0';

		if (strcmp(linebuf, "PLATFORM") == 0)
		{
			if (strcmp(tok_attr, "CUDA_RUNTIME_VERSION") == 0)
				cuda_runtime_version = pstrdup(tok_val);
			else if (strcmp(tok_attr, "NVIDIA_DRIVER_VERSION") == 0)
				nvidia_driver_version = pstrdup(tok_val);
			else if (strcmp(tok_attr, "NUMBER_OF_DEVICES") == 0)
			{
				num_devices = atoi(tok_val);
				if (num_devices < 0)
					elog(ERROR, "NUMBER_OF_DEVICES is not correct");
			}
			else
				elog(ERROR, "unknown PLATFORM attribute");
		}
		else if (strncmp(linebuf, "DEVICE", 6) == 0)
		{
			int		dindex = atoi(linebuf + 6);

			if (!gpuDevAttrs)
			{
				if (!cuda_runtime_version ||
					!nvidia_driver_version ||
					num_devices < 0)
					elog(ERROR, "incorrect gpuinfo -md format");
				Assert(num_devices > 0);
				gpuDevAttrs = calloc(num_devices, sizeof(GpuDevAttributes));
				if (!gpuDevAttrs)
					elog(ERROR, "out of memory");
			}

			if (dindex < 0 || dindex >= num_devices)
				elog(ERROR, "device index out of range");

#define DEV_ATTR(LABEL,a,b,c)						\
			else if (strcmp(tok_attr, #LABEL) == 0)	\
				gpuDevAttrs[dindex].LABEL = atoi(tok_val);

			if (strcmp(tok_attr, "DEVICE_ID") == 0)
			{
				gpuDevAttrs[dindex].DEV_ID = atoi(tok_val);
			}
			else if (strcmp(tok_attr, "DEVICE_NAME") == 0)
			{
				strncpy(gpuDevAttrs[dindex].DEV_NAME, tok_val,
						sizeof(gpuDevAttrs[dindex].DEV_NAME));
			}
			else if (strcmp(tok_attr, "DEVICE_BRAND") == 0)
			{
				strncpy(gpuDevAttrs[dindex].DEV_BRAND, tok_val,
						sizeof(gpuDevAttrs[dindex].DEV_BRAND));
			}
			else if (strcmp(tok_attr, "DEVICE_UUID") == 0)
			{
				strncpy(gpuDevAttrs[dindex].DEV_UUID, tok_val,
						sizeof(gpuDevAttrs[dindex].DEV_UUID));
			}
			else if (strcmp(tok_attr, "GLOBAL_MEMORY_SIZE") == 0)
				gpuDevAttrs[dindex].DEV_TOTAL_MEMSZ = atol(tok_val);
			else if (strcmp(tok_attr, "PCI_BAR1_MEMORY_SIZE") == 0)
				gpuDevAttrs[dindex].DEV_BAR1_MEMSZ = atol(tok_val);
#include "gpu_devattrs.h"
			else
				elog(ERROR, "incorrect gpuinfo -md format");
#undef DEV_ATTR
		}
		else
			elog(ERROR, "unexpected gpuinfo -md input:\n%s", linebuf);
	}
	ClosePipeStream(filp);

	for (i=0, cuda_dindex=0; i < num_devices; i++)
	{
		GpuDevAttributes  *dattrs = &gpuDevAttrs[i];
		char		path[MAXPGPATH];
		char		linebuf[2048];
		FILE	   *filp;

		/* Recommend to use Pascal or later */
		if (dattrs->COMPUTE_CAPABILITY_MAJOR < 6)
		{
			elog(LOG, "PG-Strom: GPU%d %s - CC %d.%d is not supported",
				 dattrs->DEV_ID,
				 dattrs->DEV_NAME,
				 dattrs->COMPUTE_CAPABILITY_MAJOR,
				 dattrs->COMPUTE_CAPABILITY_MINOR);
			continue;
		}

		/* Update the baseline device capability */
		devBaselineMaxThreadsPerBlock = Min(devBaselineMaxThreadsPerBlock,
											dattrs->MAX_THREADS_PER_BLOCK);

		/*
		 * Only Tesla or Quadro which have PCI Bar1 more than 256MB
		 * supports GPUDirectSQL
		 */
		dattrs->DEV_SUPPORT_GPUDIRECTSQL = false;
		if (dattrs->DEV_BAR1_MEMSZ > (256UL << 20))
		{
#if CUDA_VERSION < 11030
			if (strcmp(dattrs->DEV_BRAND, "TESLA") == 0 ||
				strcmp(dattrs->DEV_BRAND, "QUADRO") == 0 ||
				strcmp(dattrs->DEV_BRAND, "NVIDIA") == 0)
				dattrs->DEV_SUPPORT_GPUDIRECTSQL = true;
#else
			if (dattrs->GPU_DIRECT_RDMA_SUPPORTED)
				dattrs->DEV_SUPPORT_GPUDIRECTSQL = true;
#endif
		}

		/*
		 * read the numa node-id from the sysfs entry
		 *
		 * Note that we assume device function-id is 0, because it is
		 * uncertain whether MULTI_GPU_BOARD_GROUP_ID is an adequate value
		 * to query, and these sibling devices obviously belongs to same
		 * numa-node, even if function-id is not identical.
		 */
		snprintf(path, sizeof(path),
				 "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
				 dattrs->PCI_DOMAIN_ID,
				 dattrs->PCI_BUS_ID,
				 dattrs->PCI_DEVICE_ID);
		filp = fopen(path, "r");
		if (!filp)
			dattrs->NUMA_NODE_ID = -1;              /* unknown */
		else
		{
			if (!fgets(linebuf, sizeof(linebuf), filp))
				dattrs->NUMA_NODE_ID = -1;      /* unknown */
			else
				dattrs->NUMA_NODE_ID = atoi(linebuf);
			fclose(filp);
		}

		/* Log brief CUDA device properties */
		resetStringInfo(&str);
		appendStringInfo(&str, "GPU%d %s (%d SMs; %dMHz, L2 %dkB)",
						 dattrs->DEV_ID, dattrs->DEV_NAME,
						 dattrs->MULTIPROCESSOR_COUNT,
						 dattrs->CLOCK_RATE / 1000,
						 dattrs->L2_CACHE_SIZE >> 10);
		if (dattrs->DEV_TOTAL_MEMSZ > (4UL << 30))
			appendStringInfo(&str, ", RAM %.2fGB",
							 ((double)dattrs->DEV_TOTAL_MEMSZ /
							  (double)(1UL << 30)));
		else
			appendStringInfo(&str, ", RAM %zuMB",
							 dattrs->DEV_TOTAL_MEMSZ >> 20);
		if (dattrs->MEMORY_CLOCK_RATE > (1UL << 20))
			appendStringInfo(&str, " (%dbits, %.2fGHz)",
							 dattrs->GLOBAL_MEMORY_BUS_WIDTH,
							 ((double)dattrs->MEMORY_CLOCK_RATE /
							  (double)(1UL << 20)));
		else
			appendStringInfo(&str, " (%dbits, %dMHz)",
							 dattrs->GLOBAL_MEMORY_BUS_WIDTH,
							 dattrs->MEMORY_CLOCK_RATE >> 10);

		if (dattrs->DEV_BAR1_MEMSZ > (1UL << 30))
			appendStringInfo(&str, ", PCI-E Bar1 %luGB",
							 dattrs->DEV_BAR1_MEMSZ >> 30);
		else if (dattrs->DEV_BAR1_MEMSZ > (1UL << 20))
			appendStringInfo(&str, ", PCI-E Bar1 %luMB",
							 dattrs->DEV_BAR1_MEMSZ >> 30);

		appendStringInfo(&str, ", CC %d.%d",
						 dattrs->COMPUTE_CAPABILITY_MAJOR,
						 dattrs->COMPUTE_CAPABILITY_MINOR);
		elog(LOG, "PG-Strom: %s", str.data);

		if (i != cuda_dindex)
			memcpy(&gpuDevAttrs[cuda_dindex],
				   &gpuDevAttrs[i], sizeof(GpuDevAttributes));
		cuda_dindex++;
	}

	if (num_devices > 0)
	{
		if (cuda_dindex == 0)
			elog(ERROR, "PG-Strom: no supported GPU devices found");
		numGpuDevAttrs = cuda_dindex;
		return true;
	}
	return false;
}

/*
 * pgstrom_init_gpu_device
 */
bool
pgstrom_init_gpu_device(void)
{
	static char	*cuda_visible_devices = NULL;
	bool		default_gpudirect_enabled = false;
	size_t		default_threshold = 0;
	size_t		shared_buffer_size = (size_t)NBuffers * (size_t)BLCKSZ;
	int			i;

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
	/* collect device attributes using gpuinfo command */
	if (!pgstrom_collect_gpu_device())
		return false;

	/* pgstrom.gpudirect_enabled */
	if (gpuDirectInitDriver() == 0)
	{
		for (i=0; i < numGpuDevAttrs; i++)
		{
			if (gpuDevAttrs[i].DEV_SUPPORT_GPUDIRECTSQL)
				default_gpudirect_enabled = true;
		}
		gpudirect_driver_is_initialized = true;
	}
	DefineCustomBoolVariable("pg_strom.gpudirect_enabled",
							 "enables GPUDirect SQL",
							 NULL,
							 &__pgstrom_gpudirect_enabled,
							 default_gpudirect_enabled,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 pgstrom_gpudirect_enabled_checker, NULL, NULL);

	/*
	 * MEMO: Threshold of table's physical size to use NVMe-Strom:
	 *   ((System RAM size) -
	 *    (shared_buffer size)) * 0.5 + (shared_buffer size)
	 *
	 * If table size is enough large to issue real i/o, NVMe-Strom will
	 * make advantage by higher i/o performance.
	 */
	if (PAGE_SIZE * PHYS_PAGES > shared_buffer_size / 2)
		default_threshold = (PAGE_SIZE * PHYS_PAGES - shared_buffer_size / 2);
	default_threshold += shared_buffer_size;

	DefineCustomIntVariable("pg_strom.gpudirect_threshold",
							"Tablesize threshold to use GPUDirect SQL",
							NULL,
							&__pgstrom_gpudirect_threshold,
							default_threshold >> 10,
							262144,	/* 256MB */
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	return true;
}

/*
 * optimal_workgroup_size - calculates the optimal block size
 * according to the function and device attributes
 */
static __thread size_t __dynamic_shmem_per_block;
static __thread size_t __dynamic_shmem_per_thread;

static size_t
blocksize_to_shmemsize_helper(int blocksize)
{
	return (__dynamic_shmem_per_block +
			__dynamic_shmem_per_thread * (size_t)blocksize);
}

/*
 * gpuOccupancyMaxPotentialBlockSize
 */
static CUresult
gpuOccupancyMaxPotentialBlockSize(int *p_min_grid_sz,
								  int *p_max_block_sz,
								  CUfunction kern_function,
								  size_t dynamic_shmem_per_block,
								  size_t dynamic_shmem_per_thread)
{
	int32		min_grid_sz;
	int32		max_block_sz;
	CUresult	rc;

	if (dynamic_shmem_per_thread > 0)
	{
		__dynamic_shmem_per_block = dynamic_shmem_per_block;
		__dynamic_shmem_per_thread = dynamic_shmem_per_thread;
		rc = cuOccupancyMaxPotentialBlockSize(&min_grid_sz,
											  &max_block_sz,
											  kern_function,
											  blocksize_to_shmemsize_helper,
											  0,
											  0);
	}
	else
	{
		rc = cuOccupancyMaxPotentialBlockSize(&min_grid_sz,
											  &max_block_sz,
											  kern_function,
											  0,
											  dynamic_shmem_per_block,
											  0);
	}
	if (p_min_grid_sz)
		*p_min_grid_sz = min_grid_sz;
	if (p_max_block_sz)
		*p_max_block_sz = max_block_sz;
	return rc;
}

#if 0
CUresult
gpuOptimalBlockSize(int *p_grid_sz,
					int *p_block_sz,
					CUfunction kern_function,
					CUdevice cuda_device,
					size_t dynamic_shmem_per_block,
					size_t dynamic_shmem_per_thread)
{
	int			mp_count;
	int			min_grid_sz;
	int			max_block_sz;
	int			max_multiplicity;
	size_t		dynamic_shmem_sz;
	CUresult	rc;

	rc = cuDeviceGetAttribute(&mp_count,
							  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
							  cuda_device);
	if (rc != CUDA_SUCCESS)
		return rc;

	rc = gpuOccupancyMaxPotentialBlockSize(&min_grid_sz,
										   &max_block_sz,
										   kern_function,
										   dynamic_shmem_per_block,
										   dynamic_shmem_per_thread);
	if (rc != CUDA_SUCCESS)
		return rc;

	dynamic_shmem_sz = (dynamic_shmem_per_block +
						dynamic_shmem_per_thread * max_block_sz);
	rc = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_multiplicity,
													 kern_function,
													 max_block_sz,
													 dynamic_shmem_sz);
	if (rc != CUDA_SUCCESS)
		return rc;

	*p_grid_sz = Min(GPUKERNEL_MAX_SM_MULTIPLICITY,
					 max_multiplicity) * mp_count;
	*p_block_sz = max_block_sz;

	return CUDA_SUCCESS;
}
#endif

CUresult
gpuOptimalBlockSize(int *p_grid_sz,
					int *p_block_sz,
					CUfunction kern_function,
					/* memo: in old version, cuda_device was given instead */
					int cuda_dindex,
					size_t dynamic_shmem_per_block,
					size_t dynamic_shmem_per_thread)
{
	int			mp_count = gpuDevAttrs[cuda_dindex].MULTIPROCESSOR_COUNT;
	int			min_grid_sz;
	int			max_block_sz;
	int			max_multiplicity;
	size_t		dynamic_shmem_sz;
	CUresult	rc;

	rc = gpuOccupancyMaxPotentialBlockSize(&min_grid_sz,
										   &max_block_sz,
										   kern_function,
										   dynamic_shmem_per_block,
										   dynamic_shmem_per_thread);
	if (rc != CUDA_SUCCESS)
		return rc;

	dynamic_shmem_sz = (dynamic_shmem_per_block +
						dynamic_shmem_per_thread * max_block_sz);
	rc = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_multiplicity,
													 kern_function,
													 max_block_sz,
													 dynamic_shmem_sz);
	if (rc != CUDA_SUCCESS)
		return rc;

	*p_grid_sz = Min(GPUKERNEL_MAX_SM_MULTIPLICITY,
					 max_multiplicity) * mp_count;
	*p_block_sz = max_block_sz;

	return CUDA_SUCCESS;
}

/*
 * pgstrom_gpu_device_info - SQL function to dump device info
 */
PG_FUNCTION_INFO_V1(pgstrom_gpu_device_info);
Datum
pgstrom_gpu_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	GpuDevAttributes *dattrs;
	int			dindex;
	int			aindex;
	const char *att_name;
	const char *att_value;
	Datum		values[4];
	bool		isnull[4];
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(4);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "gpu_id",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "att_num",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "att_key",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "att_value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = 0;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	dindex = fncxt->call_cntr / (lengthof(GpuDevAttrCatalog) + 5);
	aindex = fncxt->call_cntr % (lengthof(GpuDevAttrCatalog) + 5);

	if (dindex >= numGpuDevAttrs)
		SRF_RETURN_DONE(fncxt);
	dattrs = &gpuDevAttrs[dindex];

	if (aindex == 0)
	{
		att_name = "GPU Device Name";
		att_value = dattrs->DEV_NAME;
	}
	else if (aindex == 1)
	{
		att_name = "GPU Device Brand";
		att_value = dattrs->DEV_BRAND;
	}
	else if (aindex == 2)
	{
		att_name = "GPU Device UUID";
		att_value = dattrs->DEV_UUID;
	}
	else if (aindex == 3)
	{
		att_name = "GPU Total RAM Size";
		att_value = format_bytesz(dattrs->DEV_TOTAL_MEMSZ);
	}
	else if (aindex == 4)
	{
		att_name = "GPU PCI Bar1 Size";
		att_value = format_bytesz(dattrs->DEV_BAR1_MEMSZ);
	}
	else
	{
		int		i = aindex - 5;
		int		value = *((int *)((char *)dattrs +
								  GpuDevAttrCatalog[i].attr_offset));

		att_name = GpuDevAttrCatalog[i].attr_desc;
		switch (GpuDevAttrCatalog[i].attr_kind)
		{
			case DEVATTRKIND__INT:
				att_value = psprintf("%d", value);
				break;
			case DEVATTRKIND__BYTES:
				att_value = format_bytesz((size_t)value);
				break;
			case DEVATTRKIND__KB:
				att_value = format_bytesz((size_t)value * 1024);
				break;
			case DEVATTRKIND__KHZ:
				if (value > 4000000)
					att_value = psprintf("%.2f GHz", (double)value/1000000.0);
				else if (value > 4000)
					att_value = psprintf("%d MHz", value / 1000);
				else
					att_value = psprintf("%d kHz", value);
				break;
			case DEVATTRKIND__COMPUTEMODE:
				switch (value)
				{
					case CU_COMPUTEMODE_DEFAULT:
						att_value = "Default";
						break;
#if CUDA_VERSION < 8000
					case CU_COMPUTEMODE_EXCLUSIVE:
						att_value = "Exclusive";
						break;
#endif
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
			case DEVATTRKIND__BOOL:
				att_value = psprintf("%s", value != 0 ? "True" : "False");
				break;
			case DEVATTRKIND__BITS:
				att_value = psprintf("%dbits", value);
				break;
			default:
				elog(ERROR, "Bug? unknown DevAttrKind: %d",
					 (int)GpuDevAttrCatalog[i].attr_kind);
		}
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dattrs->DEV_ID);
	values[1] = Int32GetDatum(aindex);
	values[2] = CStringGetTextDatum(att_name);
	values[3] = CStringGetTextDatum(att_value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
