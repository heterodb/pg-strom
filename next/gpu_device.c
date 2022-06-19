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
 * collectGpuDevAttrs
 */
static void
__collectGpuDevAttrs(GpuDevAttributes *dattrs, CUdevice cuda_device)
{
	CUresult	rc;
	char		path[1024];
	char		linebuf[1024];
	FILE	   *filp;
	struct stat	stat_buf;

	rc = cuDeviceGetName(dattrs->DEV_NAME, sizeof(dattrs->DEV_NAME), cuda_device);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceGetName: %s", cuStrError(rc));
	rc = cuDeviceGetUuid((CUuuid *)dattrs->DEV_UUID, cuda_device);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceGetUuid: %s", cuStrError(rc));
	rc = cuDeviceTotalMem(&dattrs->DEV_TOTAL_MEMSZ, cuda_device);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceTotalMem: %s", cuStrError(rc));
#define DEV_ATTR(LABEL,a,b,c)										\
	rc = cuDeviceGetAttribute(&dattrs->LABEL,						\
							  CU_DEVICE_ATTRIBUTE_##LABEL,			\
							  cuda_device);							\
	if (rc != CUDA_SUCCESS)											\
		__FATAL("failed on cuDeviceGetAttribute(" #LABEL "): %s",	\
				cuStrError(rc));
#include "gpu_devattrs.h"
#undef DEV_ATTR
	/*
	 * Some other fields to be fetched from Sysfs
	 */
	snprintf(path, sizeof(path),
			 "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
			 dattrs->PCI_DOMAIN_ID,
			 dattrs->PCI_BUS_ID,
			 dattrs->PCI_DEVICE_ID);
	filp = fopen(path, "r");
	if (!filp)
		dattrs->NUMA_NODE_ID = -1;	/* unknown */
	else
	{
		if (!fgets(linebuf, sizeof(linebuf), filp))
			dattrs->NUMA_NODE_ID = -1;	/* unknown */
		else
			dattrs->NUMA_NODE_ID = atoi(linebuf);
		fclose(filp);
	}

	snprintf(path, sizeof(path),
			 "/sys/bus/pci/devices/%04x:%02x:%02x.0/resource1",
			 dattrs->PCI_DOMAIN_ID,
			 dattrs->PCI_BUS_ID,
			 dattrs->PCI_DEVICE_ID);
	if (stat(path, &stat_buf) == 0)
		dattrs->DEV_BAR1_MEMSZ = stat_buf.st_size;
	else
		dattrs->DEV_BAR1_MEMSZ = 0;		/* unknown */

	/*
	 * GPU-Direct SQL is supported?
	 */
	if (dattrs->GPU_DIRECT_RDMA_SUPPORTED)
	{
		if (dattrs->DEV_BAR1_MEMSZ == 0 /* unknown */ ||
			dattrs->DEV_BAR1_MEMSZ > (256UL << 20))
			dattrs->DEV_SUPPORT_GPUDIRECTSQL = true;
	}
}

static int
collectGpuDevAttrs(int fdesc)
{
	GpuDevAttributes dattrs;
	CUdevice	cuda_device;
	CUresult	rc;
	int			i, nr_gpus;

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuInit: %s", cuStrError(rc));
	rc = cuDeviceGetCount(&nr_gpus);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceGetCount: %s", cuStrError(rc));

	for (i=0; i < nr_gpus; i++)
	{
		ssize_t		offset, nbytes;

		rc = cuDeviceGet(&cuda_device, i);
		if (rc != CUDA_SUCCESS)
			__FATAL("failed on cuDeviceGet: %s", cuStrError(rc));
		memset(&dattrs, 0, sizeof(GpuDevAttributes));
		dattrs.DEV_ID = i;
		__collectGpuDevAttrs(&dattrs, cuda_device);

		for (offset=0; offset < sizeof(GpuDevAttributes); offset += nbytes)
		{
			nbytes = write(fdesc, ((char *)&dattrs) + offset,
						   sizeof(GpuDevAttributes) - offset);
			if (nbytes == 0)
				break;
			if (nbytes < 0)
				__FATAL("failed on write(pipefd): %m");
		}
	}
	fprintf(stderr, "\nhogehoge\n\n");
	return 0;
}

/*
 * receiveGpuDevAttrs
 */
static void
receiveGpuDevAttrs(int fdesc)
{
	GpuDevAttributes *__devAttrs = NULL;
	GpuDevAttributes dattrs_saved;
	int			nitems = 0;
	int			nrooms = 0;
	bool		is_saved = false;

	for (;;)
	{
		GpuDevAttributes dtemp;
		ssize_t		nbytes;

		nbytes = __readFile(fdesc, &dtemp, sizeof(GpuDevAttributes));
		if (nbytes == 0)
			break;	/* end */
		if (nbytes != sizeof(GpuDevAttributes))
			elog(ERROR, "failed on collect GPU device attributes");
		if (dtemp.COMPUTE_CAPABILITY_MAJOR < 6)
		{
			elog(LOG, "PG-Strom: GPU%d %s - CC %d.%d is not supported",
				 dtemp.DEV_ID,
				 dtemp.DEV_NAME,
				 dtemp.COMPUTE_CAPABILITY_MAJOR,
				 dtemp.COMPUTE_CAPABILITY_MINOR);
			continue;
		}
		if (heterodbValidateDevice(dtemp.DEV_ID,
								   dtemp.DEV_NAME,
								   dtemp.DEV_UUID))
		{
			if (nitems >= nrooms)
			{
				nrooms += 10;
				__devAttrs = realloc(__devAttrs, sizeof(GpuDevAttributes) * nrooms);
				if (!__devAttrs)
					elog(ERROR, "out of memory");
			}
			memcpy(&__devAttrs[nitems++], &dtemp, sizeof(GpuDevAttributes));
		}
		else if (!is_saved)
		{
			memcpy(&dattrs_saved, &dtemp, sizeof(GpuDevAttributes));
			is_saved = true;
		}
	}

	if (nitems == 0 && is_saved)
	{
		__devAttrs = malloc(sizeof(GpuDevAttributes));
		if (!__devAttrs)
			elog(ERROR, "out of memory");
		memcpy(&__devAttrs[nitems++], &dattrs_saved, sizeof(GpuDevAttributes));
	}
	numGpuDevAttrs = nitems;
	gpuDevAttrs = __devAttrs;
}

/*
 * pgstrom_collect_gpu_devices
 */
static void
pgstrom_collect_gpu_devices(void)
{
	int		i, pipefd[2];
	pid_t	child;
	StringInfoData buf;

	if (pipe(pipefd) != 0)
		elog(ERROR, "failed on pipe(2): %m");
	child = fork();
	if (child == 0)
	{
		close(pipefd[0]);
		_exit(collectGpuDevAttrs(pipefd[1]));
	}
	else if (child > 0)
	{
		int		status;

		close(pipefd[1]);
		PG_TRY();
		{
			receiveGpuDevAttrs(pipefd[0]);
		}
		PG_CATCH();
		{
			/* cleanup */
			kill(child, SIGKILL);
			close(pipefd[0]);
			PG_RE_THROW();
		}
		PG_END_TRY();
		close(pipefd[0]);

		while (waitpid(child, &status, 1) < 0)
		{
			if (errno != EINTR)
			{
				kill(child, SIGKILL);
				elog(ERROR, "failed on waitpid: %m");
			}
		}
		if (WEXITSTATUS(status) != 0)
			elog(ERROR, "GPU device attribute collector exited with %d",
				 WEXITSTATUS(status));
	}
	else
	{
		close(pipefd[0]);
		close(pipefd[1]);
		elog(ERROR, "failed on fork(2): %m");
	}
	initStringInfo(&buf);
	for (i=0; i < numGpuDevAttrs; i++)
	{
		GpuDevAttributes *dattrs = &gpuDevAttrs[i];

		resetStringInfo(&buf);
		appendStringInfo(&buf, "GPU%d %s (%d SMs; %dMHz, L2 %dkB)",
						 dattrs->DEV_ID, dattrs->DEV_NAME,
						 dattrs->MULTIPROCESSOR_COUNT,
						 dattrs->CLOCK_RATE / 1000,
						 dattrs->L2_CACHE_SIZE >> 10);
		if (dattrs->DEV_TOTAL_MEMSZ > (4UL << 30))
			appendStringInfo(&buf, ", RAM %.2fGB",
							 ((double)dattrs->DEV_TOTAL_MEMSZ /
							  (double)(1UL << 30)));
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
		if (dattrs->DEV_BAR1_MEMSZ > (1UL << 30))
			appendStringInfo(&buf, ", PCI-E Bar1 %luGB",
							 dattrs->DEV_BAR1_MEMSZ >> 30);
		else if (dattrs->DEV_BAR1_MEMSZ > (1UL << 20))
			appendStringInfo(&buf, ", PCI-E Bar1 %luMB",
							 dattrs->DEV_BAR1_MEMSZ >> 30);
		appendStringInfo(&buf, ", CC %d.%d",
						 dattrs->COMPUTE_CAPABILITY_MAJOR,
						 dattrs->COMPUTE_CAPABILITY_MINOR);
        elog(LOG, "PG-Strom: %s", buf.data);
	}
	pfree(buf.data);
}

/*
 * pgstrom_init_gpu_device
 */
bool
pgstrom_init_gpu_device(void)
{
	static char	*cuda_visible_devices = NULL;
	bool		default_gpudirect_enabled = false;
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
	/* collect device attributes using child process */
	pgstrom_collect_gpu_devices();
	if (numGpuDevAttrs == 0)
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

	DefineCustomIntVariable("pg_strom.gpudirect_threshold",
							"Tablesize threshold to use GPUDirect SQL",
							NULL,
							&__pgstrom_gpudirect_threshold,
							5242880,	/* 5GB */
							262144,		/* 256MB */
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
	int			i, val;
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
	switch (aindex)
	{
		case 0:
			att_name = "GPU Device Name";
			att_value = dattrs->DEV_NAME;
			break;
		case 1:
			att_name = "GPU Device ID";
			att_value = psprintf("%d", dattrs->DEV_ID);
			break;
		case 2:
			att_name = "GPU Device UUID";
			att_value = dattrs->DEV_UUID;
			break;
		case 3:
			att_name = "GPU Total RAM Size";
			att_value = format_bytesz(dattrs->DEV_TOTAL_MEMSZ);
			break;
		case 4:
			att_name = "GPU PCI Bar1 Size";
			att_value = format_bytesz(dattrs->DEV_BAR1_MEMSZ);
			break;
		case 5:
			att_name = "GPU NUMA Node Id";
			att_value = psprintf("%d", dattrs->NUMA_NODE_ID);
			break;
		default:
			i = aindex - 6;
			val = *((int *)((char *)dattrs +
							GpuDevAttrCatalog[i].attr_offset));
			att_name = GpuDevAttrCatalog[i].attr_desc;
			switch (GpuDevAttrCatalog[i].attr_kind)
			{
				case DEVATTRKIND__INT:
					att_value = psprintf("%d", val);
					break;
				case DEVATTRKIND__BYTES:
					att_value = format_bytesz((size_t)val);
					break;
				case DEVATTRKIND__KB:
					att_value = format_bytesz((size_t)val * 1024);
					break;
				case DEVATTRKIND__KHZ:
					if (val > 4000000)
						att_value = psprintf("%.2f GHz", (double)val/1000000.0);
					else if (val > 4000)
						att_value = psprintf("%d MHz", val / 1000);
					else
						att_value = psprintf("%d kHz", val);
					break;
				case DEVATTRKIND__COMPUTEMODE:
					switch (val)
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
					att_value = psprintf("%s", val != 0 ? "True" : "False");
					break;
				case DEVATTRKIND__BITS:
					att_value = psprintf("%dbits", val);
					break;
				default:
					elog(ERROR, "Bug? unknown DevAttrKind: %d",
						 (int)GpuDevAttrCatalog[i].attr_kind);
			}
			break;
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dattrs->DEV_ID);
	values[1] = Int32GetDatum(aindex);
	values[2] = CStringGetTextDatum(att_name);
	values[3] = CStringGetTextDatum(att_value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
