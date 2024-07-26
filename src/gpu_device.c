/*
 * gpu_device.c
 *
 * Routines to collect GPU device information.
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"

/* variable declarations */
GpuDevAttributes *gpuDevAttrs = NULL;
int				numGpuDevAttrs = 0;
double			pgstrom_gpu_setup_cost;			/* GUC */
double			pgstrom_gpu_tuple_cost;			/* GUC */
double			pgstrom_gpu_operator_cost;		/* GUC */
double			pgstrom_gpu_direct_seq_page_cost; /* GUC */
static bool		pgstrom_gpudirect_enabled;			/* GUC */
static int		__pgstrom_gpudirect_threshold_kb;	/* GUC */
#define pgstrom_gpudirect_threshold		((size_t)__pgstrom_gpudirect_threshold_kb << 10)
static char	   *pgstrom_gpu_selection_policy = "optimal";

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
	size_t		attr_offset;
	const char *attr_label;
	const char *attr_desc;
} GpuDevAttrCatalog[] = {
#define DEV_ATTR(LABEL,DESC)					\
	{ CU_DEVICE_ATTRIBUTE_##LABEL,				\
	  offsetof(struct GpuDevAttributes, LABEL),	\
	  #LABEL, DESC },
#include "gpu_devattrs.h"
#undef DEV_ATTR
};

static const char *
sysfs_read_line(const char *path)
{
	static char	buffer[2048];
	int			fdesc;
	ssize_t		off, sz;
	char	   *pos;

	fdesc = open(path, O_RDONLY);
	if (fdesc < 0)
		return NULL;
	off = 0;
	for (;;)
	{
		sz = read(fdesc, buffer+off, sizeof(buffer)-1-off);
		if (sz > 0)
			off += sz;
		else if (sz == 0)
			break;
		else if (errno != EINTR)
		{
			close(fdesc);
			return NULL;
		}
	}
	close(fdesc);
	buffer[sz] = '\0';
	pos = strchr(buffer, '\n');
	if (pos)
		*pos = '\0';
	return __trim(buffer);
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
	CUuuid		uuid;
	int			x, y, z;
	const char *str;
	struct stat	stat_buf;

	str = sysfs_read_line("/sys/module/nvidia/version");
	if (str && sscanf(str, "%u.%u.%u", &x, &y, &z) == 3)
		dattrs->NVIDIA_KMOD_VERSION = x * 100000 + y * 100 + z;
	str = sysfs_read_line("/sys/module/nvidia_fs/version");
	if (str && sscanf(str, "%u.%u.%u", &x, &y, &z) == 3)
		dattrs->NVIDIA_FS_KMOD_VERSION = x * 100000 + y * 100 + z;
	rc = cuDriverGetVersion(&dattrs->CUDA_DRIVER_VERSION);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDriverGetVersion: %s", cuStrError(rc));
	rc = cuDeviceGetName(dattrs->DEV_NAME, sizeof(dattrs->DEV_NAME), cuda_device);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceGetName: %s", cuStrError(rc));
	rc = cuDeviceGetUuid(&uuid, cuda_device);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceGetUuid: %s", cuStrError(rc));
	sprintf(dattrs->DEV_UUID,
			"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
			(unsigned char)uuid.bytes[0],
			(unsigned char)uuid.bytes[1],
			(unsigned char)uuid.bytes[2],
			(unsigned char)uuid.bytes[3],
			(unsigned char)uuid.bytes[4],
			(unsigned char)uuid.bytes[5],
			(unsigned char)uuid.bytes[6],
			(unsigned char)uuid.bytes[7],
			(unsigned char)uuid.bytes[8],
			(unsigned char)uuid.bytes[9],
			(unsigned char)uuid.bytes[10],
			(unsigned char)uuid.bytes[11],
			(unsigned char)uuid.bytes[12],
			(unsigned char)uuid.bytes[13],
			(unsigned char)uuid.bytes[14],
			(unsigned char)uuid.bytes[15]);
	rc = cuDeviceTotalMem(&dattrs->DEV_TOTAL_MEMSZ, cuda_device);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceTotalMem: %s", cuStrError(rc));
#define DEV_ATTR(LABEL,DESC)										\
	rc = cuDeviceGetAttribute(&dattrs->LABEL,						\
							  CU_DEVICE_ATTRIBUTE_##LABEL,			\
							  cuda_device);							\
	if (CU_DEVICE_ATTRIBUTE_##LABEL > CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED &&	\
		rc == CUDA_ERROR_INVALID_VALUE)								\
		dattrs->LABEL = DEV_ATTR__UNKNOWN;							\
	else if (rc != CUDA_SUCCESS)									\
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
	return 0;
}

/*
 * receiveGpuDevAttrs
 */
static void
receiveGpuDevAttrs(int fdesc)
{
	static GpuDevAttributes devNotValidated;
	GpuDevAttributes *devAttrs = NULL;
	int			dindex = 0;
	int			nitems = 0;
	int			nrooms = 0;
	int			num_not_validated = 0;

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
		dindex = heterodbValidateDevice(dtemp.DEV_NAME,
										dtemp.DEV_UUID);
		if (dindex >= 0)
		{
			while (dindex >= nrooms)
			{
				GpuDevAttributes *__devAttrs;
				int		__nrooms = nrooms + 10;

				__devAttrs = calloc(__nrooms, sizeof(GpuDevAttributes));
				if (!__devAttrs)
					elog(ERROR, "out of memory");
				if (devAttrs)
				{
					memcpy(__devAttrs, devAttrs,
						   sizeof(GpuDevAttributes) * nrooms);
					free(devAttrs);
				}
				devAttrs = __devAttrs;
				nrooms = __nrooms;
			}
			memcpy(&devAttrs[dindex], &dtemp, sizeof(GpuDevAttributes));
			nitems = Max(nitems, dindex+1);
		}
		else if (num_not_validated++ == 0)
		{
			memcpy(&devNotValidated, &dtemp, sizeof(GpuDevAttributes));
		}
	}

	if (devAttrs)
	{
		numGpuDevAttrs = nitems;
		gpuDevAttrs = devAttrs;
	}
	else if (num_not_validated > 0)
	{
		numGpuDevAttrs = 1;
		gpuDevAttrs = &devNotValidated;
	}
	else
	{
		numGpuDevAttrs = 0;
		gpuDevAttrs = NULL;
	}
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

		while (waitpid(child, &status, 0) < 0)
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
		if (i == 0)
		{
			appendStringInfo(&buf, "PG-Strom binary built for CUDA %u.%u",
							 (CUDA_VERSION / 1000),
							 (CUDA_VERSION % 1000) / 10);
			appendStringInfo(&buf, " (CUDA runtime %u.%u",
							 (dattrs->CUDA_DRIVER_VERSION / 1000),
							 (dattrs->CUDA_DRIVER_VERSION % 1000) / 10);
			if (dattrs->NVIDIA_KMOD_VERSION != 0)
				appendStringInfo(&buf, ", nvidia kmod: %u.%u.%u",
								 (dattrs->NVIDIA_KMOD_VERSION / 100000),
								 (dattrs->NVIDIA_KMOD_VERSION % 100000) / 100,
								 (dattrs->NVIDIA_KMOD_VERSION % 100));
			if (dattrs->NVIDIA_FS_KMOD_VERSION != 0)
				appendStringInfo(&buf, ", nvidia-fs kmod: %u.%u.%u",
								 (dattrs->NVIDIA_FS_KMOD_VERSION / 100000),
								 (dattrs->NVIDIA_FS_KMOD_VERSION % 100000) / 100,
								 (dattrs->NVIDIA_FS_KMOD_VERSION % 100));
			appendStringInfo(&buf, ")");
			elog(LOG, "%s", buf.data);

			if (CUDA_VERSION < dattrs->CUDA_DRIVER_VERSION)
				elog(WARNING, "The CUDA version where this PG-Strom module binary was built for (%u.%u) is newer than the CUDA runtime version on this platform (%u.%u). It may lead unexpected behavior, and upgrade of CUDA toolkit is recommended.",
					 (CUDA_VERSION / 1000),
					 (CUDA_VERSION % 1000) / 10,
					 (dattrs->CUDA_DRIVER_VERSION / 1000),
					 (dattrs->CUDA_DRIVER_VERSION % 1000) / 10);

			resetStringInfo(&buf);
		}
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
 * pgstrom_gpu_operator_ratio
 */
double
pgstrom_gpu_operator_ratio(void)
{
	if (cpu_operator_cost > 0.0)
	{
		return pgstrom_gpu_operator_cost / cpu_operator_cost;
	}
	return (pgstrom_gpu_operator_cost == 0.0 ? 1.0 : disable_cost);
}

/*
 * optimal-gpus cache
 */
static HTAB	   *filesystem_optimal_gpu_htable = NULL;
static HTAB	   *tablespace_optimal_gpu_htable = NULL;

typedef struct
{
	dev_t		file_dev;	/* stat_buf.st_dev */
	ino_t		file_ino;	/* stat_buf.st_ino */
	struct timespec file_ctime; /* stat_buf.st_ctim */
	int64_t		optimal_gpus;
} filesystem_optimal_gpu_entry;

typedef struct
{
	Oid			tablespace_oid;
	int64_t		optimal_gpus;
} tablespace_optimal_gpu_entry;

static void
tablespace_optimal_gpu_cache_callback(Datum arg, int cacheid, uint32 hashvalue)
{
	/* invalidate all the cached status */
	if (filesystem_optimal_gpu_htable)
	{
		hash_destroy(filesystem_optimal_gpu_htable);
		filesystem_optimal_gpu_htable = NULL;
	}
	if (tablespace_optimal_gpu_htable)
	{
		hash_destroy(tablespace_optimal_gpu_htable);
		tablespace_optimal_gpu_htable = NULL;
	}
}

static bool
pgstrom_gpu_selection_policy_check_callback(char **newval, void **extra,
											GucSource source)
{
	const char *policy = *newval;

	if (strcmp(policy, "optimal") == 0 ||
		strcmp(policy, "numa") == 0 ||
		strcmp(policy, "system") == 0)
		return true;

	return false;
}

static void
pgstrom_gpu_selection_policy_assign_callback(const char *newval, void *extra)
{
	tablespace_optimal_gpu_cache_callback(0, 0, 0);
}

/*
 * GetOptimalGpuForFile
 */
static int64_t
__GetOptimalGpuForFile(const char *pathname)
{
	filesystem_optimal_gpu_entry *hentry;
	struct stat stat_buf;
	bool		found;

	if (!filesystem_optimal_gpu_htable)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = offsetof(filesystem_optimal_gpu_entry,
								file_ino) + sizeof(ino_t);
		hctl.entrysize = sizeof(filesystem_optimal_gpu_entry);
		hctl.hcxt = CacheMemoryContext;
		filesystem_optimal_gpu_htable
			= hash_create("FilesystemOptimalGpus", 1024, &hctl,
						  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}

	if (stat(pathname, &stat_buf) != 0)
	{
		elog(WARNING, "failed on stat('%s'): %m", pathname);
		return 0UL;
	}
	hentry = (filesystem_optimal_gpu_entry *)
		hash_search(filesystem_optimal_gpu_htable,
					&stat_buf,
					HASH_ENTER,
					&found);
	if (!found || (stat_buf.st_ctim.tv_sec > hentry->file_ctime.tv_sec ||
				   (stat_buf.st_ctim.tv_sec == hentry->file_ctime.tv_sec &&
					stat_buf.st_ctim.tv_nsec > hentry->file_ctime.tv_nsec)))
	{
		const char *policy = pgstrom_gpu_selection_policy;

		Assert(hentry->file_dev == stat_buf.st_dev &&
			   hentry->file_ino == stat_buf.st_ino);
		memcpy(&hentry->file_ctime, &stat_buf.st_ctim, sizeof(struct timespec));
		hentry->optimal_gpus = heterodbGetOptimalGpus(pathname, policy);
	}
	return hentry->optimal_gpus;
}

const Bitmapset *
GetOptimalGpuForFile(const char *pathname)
{
	int64_t		optimal_gpus = __GetOptimalGpuForFile(pathname);
	Bitmapset  *bms = NULL;

	for (int k=0; optimal_gpus != 0; k++)
	{
		if ((optimal_gpus & (1UL<<k)) != 0)
		{
			bms = bms_add_member(bms, k);
			optimal_gpus &= ~(1UL<<k);
		}
	}
	return bms;
}

/*
 * GetOptimalGpuForTablespace
 */
static const Bitmapset *
GetOptimalGpuForTablespace(Oid tablespace_oid)
{
    tablespace_optimal_gpu_entry *hentry;
	Bitmapset  *bms = NULL;
	bool        found;

    if (!pgstrom_gpudirect_enabled)
		return NULL;

	if (!OidIsValid(tablespace_oid))
		tablespace_oid = MyDatabaseTableSpace;

	if (!tablespace_optimal_gpu_htable)
	{
		HASHCTL     hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(tablespace_optimal_gpu_entry);
		hctl.hcxt = CacheMemoryContext;
		tablespace_optimal_gpu_htable
			= hash_create("TablespaceOptimalGpus", 128, &hctl,
						  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
    }

	hentry = (tablespace_optimal_gpu_entry *)
		hash_search(tablespace_optimal_gpu_htable,
					&tablespace_oid,
					HASH_ENTER,
					&found);
	if (!found)
	{
		char	   *path;

		Assert(hentry->tablespace_oid == tablespace_oid);
		PG_TRY();
		{
			path = GetDatabasePath(MyDatabaseId, tablespace_oid);
			hentry->optimal_gpus = __GetOptimalGpuForFile(path);
		}
		PG_CATCH();
		{
			hash_search(tablespace_optimal_gpu_htable,
						&tablespace_oid,
						HASH_REMOVE,
						NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	if (hentry->optimal_gpus != 0)
	{
		int64_t		optimal_gpus = hentry->optimal_gpus;

		for (int k=0; optimal_gpus != 0; k++)
		{
			if ((optimal_gpus & (1UL<<k)) != 0)
			{
				bms = bms_add_member(bms, k);
				optimal_gpus &= ~(1UL<<k);
			}
		}
	}
	return bms;
}

/*
 * GetOptimalGpuForRelation
 */
const Bitmapset *
GetOptimalGpuForRelation(Relation relation)
{
	Oid		tablespace_oid;

	/* only heap relation */
	Assert(RelationGetForm(relation)->relam == HEAP_TABLE_AM_OID);
	tablespace_oid = RelationGetForm(relation)->reltablespace;

	return GetOptimalGpuForTablespace(tablespace_oid);
}

/*
 * GetOptimalGpuForBaseRel - checks wthere the relation can use GPU-Direct SQL.
 * If possible, it returns bitmap of the optimal GPUs.
 */
const Bitmapset *
GetOptimalGpuForBaseRel(PlannerInfo *root, RelOptInfo *baserel)
{
	const Bitmapset *optimal_gpus;
	double		total_sz;

	if (!pgstrom_gpudirect_enabled)
		return NULL;
	if (baseRelIsArrowFdw(baserel))
	{
		if (pgstrom_gpudirect_enabled)
			return GetOptimalGpusForArrowFdw(root, baserel);
		return NULL;
	}
	total_sz = (size_t)baserel->pages * (size_t)BLCKSZ;
	if (total_sz < pgstrom_gpudirect_threshold)
		return NULL;	/* table is too small */

	optimal_gpus = GetOptimalGpuForTablespace(baserel->reltablespace);
	if (!bms_is_empty(optimal_gpus))
	{
		RangeTblEntry *rte = root->simple_rte_array[baserel->relid];
		char	relpersistence = get_rel_persistence(rte->relid);

		/* temporary table is not supported by GPU-Direct SQL */
		if (relpersistence != RELPERSISTENCE_PERMANENT &&
			relpersistence != RELPERSISTENCE_UNLOGGED)
			optimal_gpus = NULL;
	}
	return optimal_gpus;
}

/*
 * __fetchJsonField/Element - NULL aware thin-wrapper
 */
static Datum
__fetchJsonField(Datum json, const char *field)
{
	LOCAL_FCINFO(fcinfo, 2);
	Datum	datum;

	InitFunctionCallInfoData(*fcinfo, NULL, 2, InvalidOid, NULL, NULL);

	fcinfo->args[0].value = json;
	fcinfo->args[0].isnull = false;
	fcinfo->args[1].value = CStringGetTextDatum(field);
	fcinfo->args[1].isnull = false;

	datum = json_object_field(fcinfo);
	if (fcinfo->isnull)
		return 0UL;
	Assert(datum != 0UL);
	return datum;
}

static char *
__fetchJsonFieldText(Datum json, const char *field)
{
	LOCAL_FCINFO(fcinfo, 2);
	Datum	datum;

	InitFunctionCallInfoData(*fcinfo, NULL, 2, InvalidOid, NULL, NULL);

	fcinfo->args[0].value = json;
	fcinfo->args[0].isnull = false;
	fcinfo->args[1].value = CStringGetTextDatum(field);
	fcinfo->args[1].isnull = false;

	datum = json_object_field_text(fcinfo);
	if (fcinfo->isnull)
		return NULL;
	return TextDatumGetCString(datum);
}

static Datum
__fetchJsonElement(Datum json, int index)
{
	LOCAL_FCINFO(fcinfo, 2);
	Datum	datum;

	InitFunctionCallInfoData(*fcinfo, NULL, 2, InvalidOid, NULL, NULL);

	fcinfo->args[0].value = json;
	fcinfo->args[0].isnull = false;
	fcinfo->args[1].value = Int32GetDatum(index);
	fcinfo->args[1].isnull = false;

	datum = json_array_element(fcinfo);
	if (fcinfo->isnull)
		return 0UL;
	Assert(datum != 0UL);
	return datum;
}

static char *
__fetchJsonFieldOptimalGpus(Datum json)
{
	char   *s = __fetchJsonFieldText(json, "optimal_gpus");
	int64_t	optimal_gpus = (s ? atol(s) : 0);
	char	buf[1024];
	size_t	off = 0;

	if (optimal_gpus == 0)
		return "<no GPUs>";
	for (int k=0; optimal_gpus != 0; k++)
	{
		if ((optimal_gpus & (1UL<<k)) != 0)
		{
			if (off > 0)
				buf[off++] = ',';
			off += sprintf(buf+off, "GPU%d", k);
		}
		optimal_gpus &= ~(1UL<<k);
	}
	return pstrdup(buf);
}

/*
 * pgstrom_print_gpu_properties
 */
static void
pgstrom_print_gpu_properties(const char *manual_config)
{
	const char *json_cstring = heterodbInitOptimalGpus(manual_config);

	if (json_cstring)
	{
		Datum	json;
		Datum	gpus_array;
		Datum	disk_array;

		PG_TRY();
		{
			json = DirectFunctionCall1(json_in, PointerGetDatum(json_cstring));
			gpus_array = __fetchJsonField(json, "gpus");
			if (gpus_array != 0UL)
			{
				Datum	gpu;

				for (int i=0; (gpu = __fetchJsonElement(gpus_array, i)) != 0UL; i++)
				{
					char   *dindex = __fetchJsonFieldText(gpu, "dindex");
					char   *name = __fetchJsonFieldText(gpu, "name");
					char   *uuid = __fetchJsonFieldText(gpu, "uuid");
					char   *pcie = __fetchJsonFieldText(gpu, "pcie");

					elog(LOG, "[%s] GPU%s (%s; %s)",
						 pcie ? pcie : "????:??:??.?",
						 dindex ? dindex : "??",
						 name ? name : "unknown GPU",
						 uuid ? uuid : "unknown UUID");
				}
			}

			disk_array = __fetchJsonField(json, "disk");
			if (disk_array != 0UL)
			{
				Datum	disk;

				for (int i=0; (disk = __fetchJsonElement(disk_array, i)) != 0UL; i++)
				{
					char   *type;

					type = __fetchJsonFieldText(disk, "type");
					if (!type)
						continue;
					if (strcmp(type, "nvme") == 0)
					{
						char   *name = __fetchJsonFieldText(disk, "name");
						char   *model = __fetchJsonFieldText(disk, "model");
						char   *pcie = __fetchJsonFieldText(disk, "pcie");
						char   *dist = __fetchJsonFieldText(disk, "distance");
						char   *optimal_gpus = __fetchJsonFieldOptimalGpus(disk);

						elog(LOG, "[%s] %s (%s) --> %s [dist=%s]",
							 pcie ? pcie : "????:??:??.?",
							 name ? name : "nvme??",
							 model ? model : "unknown nvme",
							 optimal_gpus,
							 dist ? dist : "???");
					}
					else if (strcmp(type, "hca") == 0)
					{
						char   *name = __fetchJsonFieldText(disk, "name");
						char   *hca_type = __fetchJsonFieldText(disk, "hca_type");
						char   *pcie = __fetchJsonFieldText(disk, "pcie");
						char   *dist = __fetchJsonFieldText(disk, "distance");
						char   *optimal_gpus = __fetchJsonFieldOptimalGpus(disk);

						elog(LOG, "[%s] %s (%s) --> %s [dist=%s]",
							 pcie ? pcie : "????:??:??.?",
                             name ? name : "???",
							 hca_type ? hca_type : "???",
							 optimal_gpus,
							 dist ? dist : "???");
					}
				}
			}
		}
		PG_CATCH();
		{
			FlushErrorState();
			elog(LOG, "GPU-NVME Properties: %s", json_cstring);
		}
		PG_END_TRY();
	}
}

/*
 * pgstrom_init_gpu_options - init GUC options related to GPUs
 */
static void
pgstrom_init_gpu_options(void)
{
	bool	has_gpudirectsql = gpuDirectIsAvailable();

	/* cost factor for GPU setup */
	DefineCustomRealVariable("pg_strom.gpu_setup_cost",
							 "Cost to setup GPU device to run",
							 NULL,
							 &pgstrom_gpu_setup_cost,
							 100 * DEFAULT_SEQ_PAGE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for each Gpu task */
	DefineCustomRealVariable("pg_strom.gpu_tuple_cost",
							 "Default cost to transfer GPU<->Host per tuple",
							 NULL,
							 &pgstrom_gpu_tuple_cost,
							 DEFAULT_CPU_TUPLE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for GPU operator */
	DefineCustomRealVariable("pg_strom.gpu_operator_cost",
							 "Cost of processing each operators by GPU",
							 NULL,
							 &pgstrom_gpu_operator_cost,
							 DEFAULT_CPU_OPERATOR_COST / 16.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for GPU-Direct SQL */
	DefineCustomRealVariable("pg_strom.gpu_direct_seq_page_cost",
							 "Cost for sequential page read by GPU-Direct SQL",
							 NULL,
							 &pgstrom_gpu_direct_seq_page_cost,
							 DEFAULT_SEQ_PAGE_COST / 4.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* on/off GPU-Direct SQL */
	DefineCustomBoolVariable("pg_strom.gpudirect_enabled",
							 "enables GPUDirect SQL",
							 NULL,
							 &pgstrom_gpudirect_enabled,
							 (has_gpudirectsql ? true : false),
							 (has_gpudirectsql ? PGC_SUSET : PGC_POSTMASTER),
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* table size threshold for GPU-Direct SQL */
	DefineCustomIntVariable("pg_strom.gpudirect_threshold",
							"table-size threshold to use GPU-Direct SQL",
							NULL,
							&__pgstrom_gpudirect_threshold_kb,
							2097152,	/* 2GB */
							0,
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
}

/*
 * pgstrom_init_gpu_device
 */
bool
pgstrom_init_gpu_device(void)
{
	static char	*cuda_visible_devices = NULL;

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
	if (numGpuDevAttrs > 0)
	{
		static char *pgstrom_manual_optimal_gpus = NULL; /* GUC */

		pgstrom_init_gpu_options();
		/*
		 * pg_strom.manual_optimal_xpus
		 *
		 * config := <token>[,<token> ...]
		 * token  := <path>=<xpus>
		 * path   := (<absolute dir>|<nvmeX>)
		 * gpus   := <gpuX>[:<gpuX>...]
		 *
		 * e.g) /mnt/data_1=gpu0,/mnt/data_2=gpu1:gpu2,nvme3=gpu3,/mnt/data_2/extra=gpu0
		 */
		DefineCustomStringVariable("pg_strom.manual_optimal_gpus",
								   "manual configuration of optimal GPUs",
								   NULL,
								   &pgstrom_manual_optimal_gpus,
								   NULL,
								   PGC_POSTMASTER,
								   GUC_NOT_IN_SAMPLE,
								   NULL, NULL, NULL);
		/*
		 * pg_strom.gpu_selection_policy
		 */
		DefineCustomStringVariable("pg_strom.gpu_selection_policy",
								   "GPU selection policy - one of 'optimal', 'numa' or 'system'",
								   NULL,
								   &pgstrom_gpu_selection_policy,
								   "optimal",
								   PGC_SUSET,
								   GUC_NOT_IN_SAMPLE,
								   pgstrom_gpu_selection_policy_check_callback,
								   pgstrom_gpu_selection_policy_assign_callback,
								   NULL);
		/* tablespace cache */
		tablespace_optimal_gpu_htable = NULL;
		CacheRegisterSyscacheCallback(TABLESPACEOID,
									  tablespace_optimal_gpu_cache_callback,
									  (Datum) 0);
		/* print hardware configuration */
		pgstrom_print_gpu_properties(pgstrom_manual_optimal_gpus);
		return true;
	}
	return false;
}

/*
 * gpuClientOpenSession
 */
static int
__gpuClientChooseDevice(pgstromTaskState *pts)
{
	const Bitmapset	*optimal_gpus = pts->optimal_gpus;
	static bool		rr_initialized = false;
	static uint32	rr_counter = 0;
	uint32_t		cuda_dindex;

	if (!rr_initialized)
	{
		rr_counter = (uint32)getpid();
		rr_initialized = true;
	}

	if (!bms_is_empty(optimal_gpus))
	{
		int		num = bms_num_members(optimal_gpus);
		int	   *__dindex = alloca(sizeof(int) * num);
		int		i, k;

		for (i=0, k=bms_next_member(optimal_gpus, -1);
			 k >= 0;
			 i++, k=bms_next_member(optimal_gpus, k))
		{
			__dindex[i] = k;
		}
		Assert(i == num);
		cuda_dindex = __dindex[rr_counter++ % num];
	}
	else
	{
		/* a simple round-robin if no GPUs preference */
		cuda_dindex = (rr_counter++ % numGpuDevAttrs);
	}

	/*
	 * In case when pinned device buffer is used, parallel workers must
	 * connect to the identical device because its final buffer shall be
	 * massively updated and passed to the next task (inner buffer of JOIN).
	 */
	if ((pts->xpu_task_flags & (DEVTASK__PINNED_HASH_RESULTS |
								DEVTASK__PINNED_ROW_RESULTS)) != 0 &&
		(pts->xpu_task_flags & DEVTASK__PREAGG) == 0)
	{
		pgstromSharedState *ps_state = pts->ps_state;
		uint32_t	expected = UINT_MAX;

		if (!pg_atomic_compare_exchange_u32(&ps_state->device_selection_hint,
											&expected,
											cuda_dindex))
		{
			cuda_dindex = expected;
			if (cuda_dindex >= numGpuDevAttrs ||
				(!bms_is_empty(optimal_gpus) &&
				 !bms_is_member(cuda_dindex, optimal_gpus)))
				elog(ERROR, "Bug? 'device_selection_hint' suggest GPU%u, but out of range", cuda_dindex);
		}
	}
	return cuda_dindex;
}

void
gpuClientOpenSession(pgstromTaskState *pts,
					 const XpuCommand *session)
{
	struct sockaddr_un addr;
	pgsocket	sockfd;
	int			cuda_dindex = __gpuClientChooseDevice(pts);
	char		namebuf[32];

	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path),
			 ".pg_strom.%u.gpu%u.sock",
			 PostmasterPid, cuda_dindex);
	if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
	{
		close(sockfd);
		elog(ERROR, "failed on connect('%s'): %m", addr.sun_path);
	}
	snprintf(namebuf, sizeof(namebuf), "GPU-%d", cuda_dindex);

	__xpuClientOpenSession(pts, session, sockfd, namebuf, cuda_dindex);
}

/*
 * optimal_workgroup_size - calculates the optimal block size
 * according to the function and device attributes
 */
CUresult
gpuOptimalBlockSize(int *p_grid_sz,
					int *p_block_sz,
					CUfunction kern_function,
					unsigned int dynamic_shmem_per_block)
{
	CUresult	rc;

	rc = cuOccupancyMaxPotentialBlockSize(p_grid_sz,
										  p_block_sz,
										  kern_function,
										  NULL,
										  dynamic_shmem_per_block,
										  0);
	if (rc == CUDA_SUCCESS)
	{
		if (*p_block_sz > CUDA_MAXTHREADS_PER_BLOCK)
			*p_block_sz = CUDA_MAXTHREADS_PER_BLOCK;
	}
	return rc;
}

/*
 * pgstrom_gpu_device_info - SQL function to dump device info
 */
PG_FUNCTION_INFO_V1(pgstrom_gpu_device_info);
PUBLIC_FUNCTION(Datum)
pgstrom_gpu_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	GpuDevAttributes *dattrs;
	int			dindex;
	int			aindex;
	int			i, val;
	const char *att_name;
	const char *att_value;
	const char *att_desc;
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
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "att_name",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "att_value",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "att_desc",
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
			att_name = "DEV_NAME";
			att_desc = "GPU Device Name";
			att_value = dattrs->DEV_NAME;
			break;
		case 1:
			att_name = "DEV_ID";
			att_desc = "GPU Device ID";
			att_value = psprintf("%d", dattrs->DEV_ID);
			break;
		case 2:
			att_name = "DEV_UUID";
			att_desc = "GPU Device UUID";
			att_value = dattrs->DEV_UUID;
			break;
		case 3:
			att_name = "DEV_TOTAL_MEMSZ";
			att_desc = "GPU Total RAM Size";
			att_value = format_bytesz(dattrs->DEV_TOTAL_MEMSZ);
			break;
		case 4:
			att_name = "DEV_BAR1_MEMSZ";
			att_desc = "GPU PCI Bar1 Size";
			att_value = format_bytesz(dattrs->DEV_BAR1_MEMSZ);
			break;
		case 5:
			att_name = "NUMA_NODE_ID";
			att_desc = "GPU NUMA Node Id";
			att_value = psprintf("%d", dattrs->NUMA_NODE_ID);
			break;
		default:
			i = aindex - 6;
			val = *((int *)((char *)dattrs +
							GpuDevAttrCatalog[i].attr_offset));
			att_name = GpuDevAttrCatalog[i].attr_label;
			att_desc = GpuDevAttrCatalog[i].attr_desc;
			switch (GpuDevAttrCatalog[i].attr_id)
			{
				case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
				case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
				case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
				case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
				case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
					/* bytes */
					att_value = format_bytesz((size_t)val);
					break;

				case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
				case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
					/* clock */
					if (val > 4000000)
						att_value = psprintf("%.2f GHz", (double)val/1000000.0);
					else if (val > 4000)
						att_value = psprintf("%d MHz", val / 1000);
					else
						att_value = psprintf("%d kHz", val);
					break;

				case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
					/* bits */
					att_value = psprintf("%s", val != 0 ? "True" : "False");
					break;

				case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
					/* compute mode */
					switch (val)
					{
						case CU_COMPUTEMODE_DEFAULT:
							att_value = "Default";
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

				default:
					if (val != DEV_ATTR__UNKNOWN)
						att_value = psprintf("%d", val);
					else
						att_value = NULL;
					break;
			}
			break;
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dattrs->DEV_ID);
	values[1] = CStringGetTextDatum(att_name);
	if (att_value)
		values[2] = CStringGetTextDatum(att_value);
	else
		isnull[2] = true;
	if (att_desc)
		values[3] = CStringGetTextDatum(att_desc);
	else
		isnull[3] = true;

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
