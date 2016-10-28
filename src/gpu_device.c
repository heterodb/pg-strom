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
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"

/* scoreboard for resource management */
typedef struct GpuScoreBoard
{
	uint64				mem_total;	/* copy of DEV_TOTAL_MEMSZ */
	uint64				mem_least;	/* initial value of @mem_usage */
	pg_atomic_uint64	mem_usage;	/* current usage of device memory */
} GpuScoreBoard;

/* variable declarations */
static shmem_startup_hook_type	shmem_startup_hook_next = NULL;
static GpuScoreBoard		   *gpuScoreBoard = NULL;
static int						minGpuMemoryPreserved;
DevAttributes				   *devAttrs = NULL;
cl_int							numDevAttrs = 0;
cl_ulong						devComputeCapability = UINT_MAX;

/*
 * gpu_scoreboard_mem_alloc
 */
bool
gpu_scoreboard_mem_alloc(size_t nbytes)
{
	GpuScoreBoard  *gscore = &gpuScoreBoard[gpuserv_cuda_dindex];
	size_t			new_value;

	Assert(IsGpuServerProcess());
	Assert(gpuserv_cuda_dindex < numDevAttrs);
	new_value = pg_atomic_add_fetch_u64(&gscore->mem_usage, nbytes);
	if (new_value <= gscore->mem_total)
		return true;
	/* revert it */
	pg_atomic_sub_fetch_u64(&gscore->mem_usage, nbytes);
	return false;
}

/*
 * gpu_scoreboard_mem_free
 */
void
gpu_scoreboard_mem_free(size_t nbytes)
{
	GpuScoreBoard  *gscore = &gpuScoreBoard[gpuserv_cuda_dindex];
	size_t			new_value;

	Assert(IsGpuServerProcess());
	Assert(gpuserv_cuda_dindex < numDevAttrs);
	new_value = pg_atomic_sub_fetch_u64(&gscore->mem_usage, nbytes);
	Assert(new_value >= gscore->mem_least ||	/* free more than alloc? */
		   new_value <  (size_t)(1UL << 60));	/* underflow? */
}

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
} DevAttrCatalog[] = {
#define DEV_ATTR(LABEL,KIND,a,DESC)				\
	{ CU_DEVICE_ATTRIBUTE_##LABEL,				\
	  DEVATTRKIND__##KIND,						\
	  offsetof(struct DevAttributes, LABEL),	\
	  DESC },
#include "device_attrs.h"
#undef DEV_ATTR
};

/*
 * pgstrom_collect_gpu_device
 */
static void
pgstrom_collect_gpu_device(void)
{
	StringInfoData str;
	char	   *cmdline;
	char		linebuf[2048];
	FILE	   *filp;
	char	   *tok_attr;
	char	   *tok_val;
	char	   *pos;
	char	   *cuda_runtime_version = NULL;
	char	   *nvidia_driver_version = NULL;
	int			num_devices = -1;	/* total num of GPUs; incl legacy models */
	int			i, j;

	initStringInfo(&str);

	cmdline = psprintf("%s -md", CMD_GPUINFO_PATH);
	filp = OpenPipeStream(cmdline, PG_BINARY_R);

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

			if (!devAttrs)
			{
				if (!cuda_runtime_version ||
					!nvidia_driver_version ||
					num_devices < 0)
					elog(ERROR, "incorrect gpuinfo -md format");
				Assert(num_devices > 0);
				devAttrs = MemoryContextAllocZero(TopMemoryContext,
												  sizeof(DevAttributes) *
												  num_devices);
			}

			if (dindex < 0 || dindex >= num_devices)
				elog(ERROR, "device index out of range");

#define DEV_ATTR(LABEL,a,b,c)						\
			else if (strcmp(tok_attr, #LABEL) == 0)	\
				devAttrs[dindex].LABEL = atoi(tok_val);

			if (strcmp(tok_attr, "DEVICE_ID") == 0)
			{
				if (dindex != atoi(tok_val))
					elog(ERROR, "incorrect gpuinfo -md format");
				devAttrs[dindex].DEV_ID = dindex;
			}
			else if (strcmp(tok_attr, "DEVICE_NAME") == 0)
			{
				strncpy(devAttrs[dindex].DEV_NAME, tok_val,
						sizeof(devAttrs[dindex].DEV_NAME));
			}
			else if (strcmp(tok_attr, "GLOBAL_MEMORY_SIZE") == 0)
				devAttrs[dindex].DEV_TOTAL_MEMSZ = atol(tok_val);
#include "device_attrs.h"
			else
				elog(ERROR, "incorrect gpuinfo -md format");
#undef DEV_ATTR
		}
		else
			elog(ERROR, "unexpected gpuinfo -md input:\n%s", linebuf);
	}
	ClosePipeStream(filp);

	for (i=0, j=0; i < num_devices; i++)
	{
		DevAttributes  *dattrs = &devAttrs[i];
		int				compute_capability;

		/* Is it supported CC? */
		if (dattrs->COMPUTE_CAPABILITY_MAJOR < 3 ||
			(dattrs->COMPUTE_CAPABILITY_MAJOR == 3 &&
			 dattrs->COMPUTE_CAPABILITY_MINOR < 5))
		{
			elog(LOG, "PG-Strom: GPU%d %s - CC %d.%d is not supported",
				 dattrs->DEV_ID,
				 dattrs->DEV_NAME,
				 dattrs->COMPUTE_CAPABILITY_MAJOR,
				 dattrs->COMPUTE_CAPABILITY_MINOR);
			continue;
		}

		/* Update baseline CC for code build */
		compute_capability = (dattrs->COMPUTE_CAPABILITY_MAJOR * 10 +
							  dattrs->COMPUTE_CAPABILITY_MINOR);
		devComputeCapability = Min(devComputeCapability,
								   compute_capability);

		/* Determine CORES_PER_MPU by CC */
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
			else
				dattrs->CORES_PER_MPU = 128;
		}
		else
			dattrs->CORES_PER_MPU = 0;	/* unknown */

		/* Log brief CUDA device properties */
		resetStringInfo(&str);
		appendStringInfo(&str, "GPU%d %s (",
						 dattrs->DEV_ID, dattrs->DEV_NAME);
		if (dattrs->CORES_PER_MPU > 0)
			appendStringInfo(&str, "%d CUDA cores",
							 dattrs->CORES_PER_MPU *
							 dattrs->MULTIPROCESSOR_COUNT);
		else
			appendStringInfo(&str, "%d SMs",
							 dattrs->MULTIPROCESSOR_COUNT);
		appendStringInfo(&str, "; %dMHz, L2 %dkB)",
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
		appendStringInfo(&str, ", CC %d.%d",
						 dattrs->COMPUTE_CAPABILITY_MAJOR,
						 dattrs->COMPUTE_CAPABILITY_MINOR);
		elog(LOG, "PG-Strom: %s", str.data);

		if (i != j)
			memcpy(&devAttrs[j], &devAttrs[i], sizeof(DevAttributes));

		j++;
	}
	Assert(j <= num_devices);
	numDevAttrs = j;
	if (numDevAttrs == 0)
		elog(ERROR, "PG-Strom: no supported GPU devices found");
}

/*
 * pgstrom_startup_gpu_device
 */
static void
pgstrom_startup_gpu_device(void)
{
	int		i;
	bool	found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	gpuScoreBoard = ShmemInitStruct("gpuScoreBoard",
									sizeof(GpuScoreBoard) * numDevAttrs,
									&found);
	Assert(!found);

	for (i=0; i < numDevAttrs; i++)
	{
		GpuScoreBoard  *gscore = &gpuScoreBoard[i];

		gscore->mem_total = devAttrs[i].DEV_TOTAL_MEMSZ;
		gscore->mem_least = (((size_t)minGpuMemoryPreserved << 10) +
							 gpuMemSizeIOMap());
		pg_atomic_init_u64(&gscore->mem_usage, gscore->mem_least);
	}
}

/*
 * pgstrom_init_gpu_device
 */
void
pgstrom_init_gpu_device(void)
{
	static char	   *cuda_visible_devices = NULL;

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

	/*
	 * Minimum amount of GPU device memory to be preserved for CUDA platform
	 * usage. In case of device memory starvation, CUDA APIs may randomly
	 * fail with OUT_OF_MEMORY error.
	 * If and when most of device memory is consumed, it is likely stuck of
	 * GPU task which is submited but not executed, so short wait is a good
	 * idea to cool down.
	 */
	DefineCustomIntVariable("pg_strom.min_gpu_memory_preserved",
							"minimum amount of GPU device memory preserved",
							NULL,
							&minGpuMemoryPreserved,
							81920,	/* 80MB */
							0,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* collect device properties by gpuinfo command */
	pgstrom_collect_gpu_device();

	/* require shared memory for score-board */
	RequestAddinShmemSpace(MAXALIGN(sizeof(gpuScoreBoard) * numDevAttrs));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_device;
}

Datum
pgstrom_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	DevAttributes  *dattrs;
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
		att_value = psprintf("%d", dattrs->DEV_ID);
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
		switch (DevAttrCatalog[i].attr_kind)
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
					 (int)DevAttrCatalog[i].attr_kind);
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
