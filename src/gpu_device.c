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
int			numGpuDevAttrs = 0;
double		pgstrom_gpu_setup_cost;			/* GUC */
double		pgstrom_gpu_tuple_cost;			/* GUC */
double		pgstrom_gpu_operator_cost;		/* GUC */
double		pgstrom_gpu_direct_seq_page_cost; /* GUC */
char	   *pgstrom_cuda_toolkit_basedir = CUDA_TOOLKIT_BASEDIR; /* GUC */
const char *pgstrom_fatbin_image_filename = "/dev/null";
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
	rc = cuDeviceGetUuid((CUuuid *)dattrs->DEV_UUID, cuda_device);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuDeviceGetUuid: %s", cuStrError(rc));
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
 * __setup_gpu_fatbin_filename
 */
#define PGSTROM_FATBIN_DIR		".pgstrom_fatbin"

static void
__appendTextFromFile(StringInfo buf, const char *filename, const char *suffix)
{
	char	path[MAXPGPATH];
	int		fdesc;

	snprintf(path, MAXPGPATH,
			 PGSHAREDIR "/pg_strom/%s%s", filename, suffix ? suffix : "");
	fdesc = open(path, O_RDONLY);
	if (fdesc < 0)
		elog(ERROR, "could not open '%s': %m", path);
	PG_TRY();
	{
		struct stat st_buf;
		off_t		remained;
		ssize_t		nbytes;

		if (fstat(fdesc, &st_buf) != 0)
			elog(ERROR, "failed on fstat('%s'): %m", path);
		remained = st_buf.st_size;

		enlargeStringInfo(buf, remained);
		while (remained > 0)
		{
			nbytes = read(fdesc, buf->data + buf->len, remained);
			if (nbytes < 0)
			{
				if (errno != EINTR)
					elog(ERROR, "failed on read('%s'): %m", path);
			}
			else if (nbytes == 0)
			{
				elog(ERROR, "unable to read '%s' by the EOF", path);
			}
			else
			{
				Assert(nbytes <= remained);
				buf->len += nbytes;
				remained -= nbytes;
			}
		}
	}
	PG_CATCH();
	{
		close(fdesc);
		PG_RE_THROW();
	}
	PG_END_TRY();
	close(fdesc);
}

static char *
__setup_gpu_fatbin_filename(void)
{
	int			cuda_version = -1;
	char		hexsum[33];		/* 128bit hash */
	char	   *namebuf;
	char	   *tok, *pos;
	const char *errstr;
	ResourceOwner resowner_saved;
	ResourceOwner resowner_dummy;
	StringInfoData buf;

	for (int i=0; i < numGpuDevAttrs; i++)
	{
		if (i == 0)
			cuda_version = gpuDevAttrs[i].CUDA_DRIVER_VERSION;
		else if (cuda_version != gpuDevAttrs[i].CUDA_DRIVER_VERSION)
			elog(ERROR, "Bug? CUDA Driver version mismatch between devices");
	}
	namebuf = alloca(Max(sizeof(CUDA_CORE_HEADERS),
						 sizeof(CUDA_CORE_FILES)) + 1);
	initStringInfo(&buf);
	/* CUDA_CORE_HEADERS */
	strcpy(namebuf, CUDA_CORE_HEADERS);
	for (tok = strtok_r(namebuf, " ", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL, " ", &pos))
	{
		__appendTextFromFile(&buf, tok, NULL);
	}

	/* CUDA_CORE_SRCS */
	strcpy(namebuf, CUDA_CORE_FILES);
	for (tok = strtok_r(namebuf, " ", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL, " ", &pos))
	{
		__appendTextFromFile(&buf, tok, ".cu");
	}
	/*
	 * Calculation of MD5SUM. Note that pg_md5_hash internally use
	 * ResourceOwner to track openSSL memory, however, we may not
	 * have the CurrentResourceOwner during startup.
	 */
	resowner_saved = CurrentResourceOwner;
	resowner_dummy = ResourceOwnerCreate(NULL, "MD5SUM Dummy");
	PG_TRY();
	{
		CurrentResourceOwner = resowner_dummy;
		if (!pg_md5_hash(buf.data, buf.len, hexsum, &errstr))
			elog(ERROR, "could not compute MD5 hash: %s", errstr);
	}
	PG_CATCH();
	{
		CurrentResourceOwner = resowner_saved;
		ResourceOwnerRelease(resowner_dummy,
							 RESOURCE_RELEASE_BEFORE_LOCKS,
							 false,
							 false);
		ResourceOwnerDelete(resowner_dummy);
		PG_RE_THROW();
	}
	PG_END_TRY();
	CurrentResourceOwner = resowner_saved;
	ResourceOwnerRelease(resowner_dummy,
						 RESOURCE_RELEASE_BEFORE_LOCKS,
						 true,
						 false);
	ResourceOwnerDelete(resowner_dummy);

	return psprintf("pgstrom-gpucode-V%06d-%s.fatbin",
					cuda_version, hexsum);
}

/*
 * __gpu_archtecture_label
 */
static const char *
__gpu_archtecture_label(int major_cc, int minor_cc)
{
	int		cuda_arch = major_cc * 100 + minor_cc;

	switch (cuda_arch)
	{
		case 600:	/* Tesla P100 */
			return "sm_60";
		case 601:	/* Tesla P40 */
			return "sm_61";
		case 700:	/* Tesla V100 */
			return "sm_70";
		case 705:	/* Tesla T4 */
			return "sm_75";
		case 800:	/* NVIDIA A100 */
			return "sm_80";
		case 806:	/* NVIDIA A40 */
			return "sm_86";
		case 809:	/* NVIDIA L40 */
			return "sm_89";
		case 900:	/* NVIDIA H100 */
			return "sm_90";
		default:
			elog(ERROR, "unsupported compute capability (%d.%d)",
				 major_cc, minor_cc);
	}
	return NULL;
}

/*
 * __validate_gpu_fatbin_file
 */
static bool
__validate_gpu_fatbin_file(const char *fatbin_dir, const char *fatbin_file)
{
	StringInfoData cmd;
	StringInfoData buf;
	FILE	   *filp;
	char	   *temp;
	bool		retval = false;

	initStringInfo(&cmd);
	initStringInfo(&buf);

	appendStringInfo(&buf, "%s/%s", fatbin_dir, fatbin_file);
	if (access(buf.data, R_OK) != 0)
		return false;
	/* Pick up supported SM from the fatbin file */
	appendStringInfo(&cmd,
					 "%s/bin/cuobjdump '%s/%s'"
					 " | grep '^arch '"
					 " | awk '{print $3}'",
					 pgstrom_cuda_toolkit_basedir,
					 fatbin_dir, fatbin_file);
	filp = OpenPipeStream(cmd.data, "r");
	if (!filp)
	{
		elog(LOG, "unable to run [%s]: %m", cmd.data);
		goto out;
	}

	resetStringInfo(&buf);
	for (;;)
	{
		ssize_t	nbytes;

		enlargeStringInfo(&buf, 512);
		nbytes = fread(buf.data + buf.len, 1, 512, filp);

		if (nbytes < 0)
		{
			if (errno != EINTR)
			{
				elog(LOG, "unable to read from pipe:[%s]: %m", cmd.data);
				goto out;
			}
		}
		else if (nbytes == 0)
		{
			if (feof(filp))
				break;
			elog(LOG, "unable to read from pipe:[%s]: %m", cmd.data);
			goto out;
		}
		else
		{
			buf.len += nbytes;
		}
	}
	ClosePipeStream(filp);

	temp = alloca(buf.len + 1);
	for (int i=0; i < numGpuDevAttrs; i++)
	{
		char	   *tok, *pos;
		const char *label;

		label = __gpu_archtecture_label(gpuDevAttrs[i].COMPUTE_CAPABILITY_MAJOR,
										gpuDevAttrs[i].COMPUTE_CAPABILITY_MINOR);

		memcpy(temp, buf.data, buf.len+1);
		for (tok = strtok_r(temp, " \n\r", &pos);
			 tok != NULL;
			 tok = strtok_r(NULL, " \n\r", &pos))
		{
			if (strcmp(label, tok) == 0)
				break;	/* ok, supported */
		}
		if (!tok)
		{
			elog(LOG, "GPU%d '%s' CC%d.%d is not supported at '%s'",
				 i, gpuDevAttrs[i].DEV_NAME,
				 gpuDevAttrs[i].COMPUTE_CAPABILITY_MAJOR,
				 gpuDevAttrs[i].COMPUTE_CAPABILITY_MINOR,
				 fatbin_file);
			goto out;
		}
	}
	/* ok, this fatbin is validated */
	retval = true;
out:
	pfree(cmd.data);
	pfree(buf.data);
	return retval;
}

/*
 * __rebuild_gpu_fatbin_file
 */
static void
__rebuild_gpu_fatbin_file(const char *fatbin_dir,
						  const char *fatbin_file)
{
	StringInfoData cmd;
	char	workdir[200];
	char   *namebuf;
	char   *tok, *pos;
	int		count;
	int		status;

	strcpy(workdir, "/tmp/.pgstrom_fatbin_build_XXXXXX");
	if (!mkdtemp(workdir))
		elog(ERROR, "unable to create work directory for fatbin rebuild");

	elog(LOG, "PG-Strom fatbin image is not valid now, so rebuild in progress...");
	
	namebuf = alloca(sizeof(CUDA_CORE_FILES) + 1);
	strcpy(namebuf, CUDA_CORE_FILES);

	initStringInfo(&cmd);
	appendStringInfo(&cmd, "cd '%s' && (", workdir);
	for (tok = strtok_r(namebuf, " ", &pos), count=0;
		 tok != NULL;
		 tok = strtok_r(NULL,    " ", &pos), count++)
	{
		if (count > 0)
			appendStringInfo(&cmd, " & ");
		appendStringInfo(&cmd,
						 " /bin/sh -x -c '%s/bin/nvcc"
						 " --maxrregcount=%d"
						 " --source-in-ptx -lineinfo"
						 " -I. -I%s "
						 " -DHAVE_FLOAT2 "
						 " -arch=native --threads 4"
						 " --device-c"
						 " -o %s.o"
						 " %s/pg_strom/%s.cu' >& %s.log",
						 pgstrom_cuda_toolkit_basedir,
						 CUDA_MAXREGCOUNT,
						 PGINCLUDEDIR,
						 tok,
						 PGSHAREDIR, tok, tok);
	}
	appendStringInfo(&cmd,
					 ") && wait;"
					 " /bin/sh -x -c '%s/bin/nvcc"
					 " -Xnvlink --suppress-stack-size-warning"
					 " -arch=native --threads 4"
					 " --device-link --fatbin"
					 " -o '%s'",
					 pgstrom_cuda_toolkit_basedir,
					 fatbin_file);
	strcpy(namebuf, CUDA_CORE_FILES);
	for (tok = strtok_r(namebuf, " ", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL,    " ", &pos))
	{
		appendStringInfo(&cmd, " %s.o", tok);
	}
	appendStringInfo(&cmd, "' >& %s.log", fatbin_file);

	status = system(cmd.data);
	if (status != 0)
		elog(ERROR, "failed on the build process at [%s]", workdir);

	/* validation of the fatbin file */
	if (!__validate_gpu_fatbin_file(workdir, fatbin_file))
		elog(ERROR, "failed on validation of the rebuilt fatbin at [%s]", workdir);

	/* installation of the rebuilt fatbin */
	resetStringInfo(&cmd);
	appendStringInfo(&cmd,
					 "mkdir -p '%s'; "
					 "install -m 0644 %s/%s '%s'",
					 fatbin_dir,
					 workdir, fatbin_file, fatbin_dir);
	strcpy(namebuf, CUDA_CORE_FILES);
	for (tok = strtok_r(namebuf, " ", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL,    " ", &pos))
	{
		appendStringInfo(&cmd, "; cat %s/%s.log >> %s/%s.log",
						 workdir, tok,
						 PGSTROM_FATBIN_DIR, fatbin_file);
	}
	appendStringInfo(&cmd, "; cat %s/%s.log >> %s/%s.log",
					 workdir, fatbin_file,
					 PGSTROM_FATBIN_DIR, fatbin_file);

	status = system(cmd.data);
	if (status != 0)
		elog(ERROR, "failed on shell command: %s", cmd.data);
}

/*
 * pgstrom_setup_gpu_fatbin
 */
static void
pgstrom_setup_gpu_fatbin(void)
{
	const char *fatbin_file = __setup_gpu_fatbin_filename();
	const char *fatbin_dir = PGSHAREDIR "/pg_strom";
	char	   *path;

	if (!__validate_gpu_fatbin_file(fatbin_dir,
									fatbin_file))
	{
		fatbin_dir = PGSTROM_FATBIN_DIR;
		if (!__validate_gpu_fatbin_file(fatbin_dir,
										fatbin_file))
		{
			__rebuild_gpu_fatbin_file(fatbin_dir,
									  fatbin_file);
		}
	}
	path = alloca(strlen(fatbin_dir) +
				  strlen(fatbin_file) + 100);
	sprintf(path, "%s/%s", fatbin_dir, fatbin_file);
	pgstrom_fatbin_image_filename = strdup(path);
	if (!pgstrom_fatbin_image_filename)
		elog(ERROR, "out of memory");
	elog(LOG, "PG-Strom fatbin image is ready: %s", fatbin_file);
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
 * pgstrom_init_gpu_options - init GUC options related to GPUs
 */
static void
pgstrom_init_gpu_options(void)
{
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
		DefineCustomStringVariable("pg_strom.cuda_toolkit_basedir",
								   "CUDA Toolkit installation directory",
								   NULL,
								   &pgstrom_cuda_toolkit_basedir,
								   CUDA_TOOLKIT_BASEDIR,
								   PGC_POSTMASTER,
								   GUC_NOT_IN_SAMPLE,
								   NULL, NULL, NULL);
		pgstrom_setup_gpu_fatbin();
		pgstrom_init_gpu_options();
		return true;
	}
	return false;
}

/*
 * gpuClientOpenSession
 */
static int
__gpuClientChooseDevice(const Bitmapset *gpuset)
{
	static bool		rr_initialized = false;
	static uint32	rr_counter = 0;

	if (!rr_initialized)
	{
		rr_counter = (uint32)getpid();
		rr_initialized = true;
	}

	if (!bms_is_empty(gpuset))
	{
		int		num = bms_num_members(gpuset);
		int	   *dindex = alloca(sizeof(int) * num);
		int		i, k;

		for (i=0, k=bms_next_member(gpuset, -1);
			 k >= 0;
			 i++, k=bms_next_member(gpuset, k))
		{
			dindex[i] = k;
		}
		Assert(i == num);
		return dindex[rr_counter++ % num];
	}
	/* a simple round-robin if no GPUs preference */
	return (rr_counter++ % numGpuDevAttrs);
}

void
gpuClientOpenSession(pgstromTaskState *pts,
					 const XpuCommand *session)
{
	struct sockaddr_un addr;
	pgsocket	sockfd;
	int			cuda_dindex = __gpuClientChooseDevice(pts->optimal_gpus);
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
	return cuOccupancyMaxPotentialBlockSize(p_grid_sz,
											p_block_sz,
											kern_function,
											NULL,
											dynamic_shmem_per_block,
											0);
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
			att_value = psprintf("GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-"
								 "%02x%02x-%02x%02x%02x%02x%02x%02x",
								 (uint8_t)dattrs->DEV_UUID[0],
								 (uint8_t)dattrs->DEV_UUID[1],
								 (uint8_t)dattrs->DEV_UUID[2],
								 (uint8_t)dattrs->DEV_UUID[3],
								 (uint8_t)dattrs->DEV_UUID[4],
								 (uint8_t)dattrs->DEV_UUID[5],
								 (uint8_t)dattrs->DEV_UUID[6],
								 (uint8_t)dattrs->DEV_UUID[7],
								 (uint8_t)dattrs->DEV_UUID[8],
								 (uint8_t)dattrs->DEV_UUID[9],
								 (uint8_t)dattrs->DEV_UUID[10],
								 (uint8_t)dattrs->DEV_UUID[11],
								 (uint8_t)dattrs->DEV_UUID[12],
								 (uint8_t)dattrs->DEV_UUID[13],
								 (uint8_t)dattrs->DEV_UUID[14],
								 (uint8_t)dattrs->DEV_UUID[15]);
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
