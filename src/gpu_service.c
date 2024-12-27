/*
 * gpu_service.c
 *
 * A background worker process that handles any interactions with GPU
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"
#include <cudaProfiler.h>
/*
 * gpuContext / gpuMemory
 */
typedef struct
{
	struct gpuContext *gcontext;	/* gpuContext that owns this pool */
	pthread_mutex_t	lock;
	bool			is_managed;	/* true, if managed memory pool */
	size_t			total_sz;	/* total pool size */
	size_t			hard_limit;
	size_t			keep_limit;
	dlist_head		segment_list;
} gpuMemoryPool;

struct gpuContext
{
	dlist_node		chain;
	int				serv_fd;		/* for accept(2) */
	int				cuda_dindex;
	gpumask_t		cuda_dmask;		/* = (1UL<<cuda_dindex) */
	CUdevice		cuda_device;
	CUcontext		cuda_context;
	CUmodule		cuda_module;
	HTAB		   *cuda_type_htab;
	HTAB		   *cuda_func_htab;
	CUfunction		cufn_kern_gpumain;
	CUfunction		cufn_prep_gistindex;
	CUfunction		cufn_merge_outer_join_map;
	CUfunction		cufn_kbuf_partitioning;
	CUfunction		cufn_kbuf_reconstruction;
	CUfunction		cufn_gpucache_apply_redo;
	CUfunction		cufn_gpucache_compaction;
	int				gpumain_shmem_sz_limit;
	xpu_encode_info *cuda_encode_catalog;
	gpuMemoryPool	pool_raw;
	gpuMemoryPool	pool_managed;
	bool			cuda_profiler_started;
	volatile uint32_t cuda_stack_limit;	/* current configuration */
	pthread_mutex_t	cuda_setlimit_lock;
	/* GPU workers */
	pthread_mutex_t	worker_lock;
	dlist_head		worker_list;
	/* XPU commands */
	pthread_cond_t	cond;
	pthread_mutex_t	lock;
	dlist_head		command_list;
	pg_atomic_uint32 num_commands;
};

struct gpuMemChunk;

struct gpuClient
{
	struct gpuContext *gcontext;/* per-device status */
	dlist_node		chain;		/* gcontext->client_list */
	gpumask_t		optimal_gpus; /* candidate GPUs for scheduling */
	uint32_t		xpu_task_flags; /* copy from session->xpu_task_flags */
	kern_session_info *h_session; /* session info in host memory */
	struct gpuQueryBuffer *gq_buf; /* per query join/preagg device buffer */
	pg_atomic_uint32 refcnt;	/* odd number, if error status */
	pthread_mutex_t	mutex;		/* mutex to write the socket */
	int				sockfd;		/* connection to PG backend */
	pthread_t		worker;		/* receiver thread */
	struct gpuMemChunk *__session[1];	/* per device session info */
};

#define GPUSERV_WORKER_KIND__GPUTASK		't'
#define GPUSERV_WORKER_KIND__GPUCACHE		'c'

typedef struct
{
	dlist_node		chain;
	gpuContext	   *gcontext;
	pthread_t		worker;
	char			kind;	/* one of GPUSERV_WORKER_KIND__* */
	volatile bool	termination;
} gpuWorker;

/*
 * GPU service shared GPU variable
 */
typedef struct
{
	volatile bool		gpuserv_ready_accept;
	/*
	 * For worker startup delay (issue #811), max_async_tasks records
	 * the timestamp when configuration was updated.
	 * Lower 10bit : number of worker threads per GPU device.
	 * Upper 53bit : timestamp in msec precision from the epoch.
	 *               (If timestamp==0, it didn't updated yet)
	 */
#define MAX_ASYNC_TASKS_DELAY	4000	/* 4.0sec */
#define MAX_ASYNC_TASKS_BITS	10
#define MAX_ASYNC_TASKS_MASK	((1UL<<MAX_ASYNC_TASKS_BITS)-1)
	pg_atomic_uint64	max_async_tasks;
	pg_atomic_uint32	gpuserv_debug_output;
} gpuServSharedState;

/*
 * variables
 */
static __thread gpuContext		*GpuWorkerCurrentContext = NULL;
#define MY_DINDEX_PER_THREAD	(GpuWorkerCurrentContext->cuda_dindex)
#define MY_DEVICE_PER_THREAD	(GpuWorkerCurrentContext->cuda_device)
#define MY_CONTEXT_PER_THREAD	(GpuWorkerCurrentContext->cuda_context)
#define MY_STREAM_PER_THREAD	CU_STREAM_PER_THREAD 
static volatile int			gpuserv_bgworker_got_signal = 0;
static gpuContext		   *gpuserv_gpucontext_array;
static dlist_head			gpuserv_gpucontext_list;
static int					gpuserv_epoll_fdesc = -1;
static int					gpuserv_listen_sockfd = -1;
static int					gpuserv_logger_pipefd[2];
static FILE				   *gpuserv_logger_filp;
static StringInfoData		gpuserv_logger_buffer;
static pthread_mutex_t		gpuserv_client_lock;
static dlist_head			gpuserv_client_list;
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static gpuServSharedState *gpuserv_shared_state = NULL;
static int			__pgstrom_max_async_tasks_dummy;
static int			__pgstrom_cuda_stack_limit_kb;
static bool			__gpuserv_debug_output_dummy;
static char		   *pgstrom_cuda_toolkit_basedir = CUDA_TOOLKIT_BASEDIR; /* GUC */
static const char  *pgstrom_fatbin_image_filename = "/dev/null";

static void
gpuservLoggerReport(const char *fmt, ...)	pg_attribute_printf(1, 2);

#define __gsLogCxt(gcontext,fmt,...)								\
	do {															\
		if (!gcontext)												\
			gpuservLoggerReport("GPU-Serv|LOG|%s|%d|%s|" fmt "\n",	\
								__basename(__FILE__),				\
								__LINE__,							\
								__FUNCTION__,						\
								##__VA_ARGS__);						\
		else														\
			gpuservLoggerReport("GPU%d|LOG|%s|%d|%s|" fmt "\n",		\
								gcontext->cuda_dindex,				\
								__basename(__FILE__),				\
								__LINE__,							\
								__FUNCTION__,						\
								##__VA_ARGS__);						\
	} while(0)

#define __gsLog(fmt, ...)									\
	__gsLogCxt(GpuWorkerCurrentContext,fmt,##__VA_ARGS__)

#define __gsDebug(fmt, ...)										\
	do {														\
		if (gpuserv_shared_state &&								\
			pg_atomic_read_u32(&gpuserv_shared_state->gpuserv_debug_output) != 0) \
			__gsLog(fmt, ##__VA_ARGS__);						\
	} while(0)

#define __gsDebugExtra(fmt, ...)										\
	do {																\
		if (gpuserv_shared_state &&										\
			pg_atomic_read_u32(&gpuserv_shared_state->gpuserv_debug_output) != 0) \
		{																\
			const char *filename;										\
			uint32_t	lineno;											\
			int			errcode;										\
			char		buffer[2048];									\
																		\
			errcode = heterodbExtraGetError(&filename,					\
											&lineno,					\
											NULL,						\
											buffer,						\
											sizeof(buffer));			\
			if (errcode == 0)											\
				__gsLog("heterodb-extra: " fmt, ##__VA_ARGS__);			\
			else														\
				__gsLog("heterodb-extra: " fmt " [%s] (code=%d, %s:%d)", \
						##__VA_ARGS__, buffer, errcode,					\
						__basename(filename), lineno);					\
		}																\
	} while(0)

static void
gpuserv_debug_output_assign(bool newval, void *extra)
{
	uint32_t	ival = (newval ? 1 : 0);

	if (gpuserv_shared_state)
		pg_atomic_write_u32(&gpuserv_shared_state->gpuserv_debug_output, ival);
	else
		__gpuserv_debug_output_dummy = ival;
}

static const char *
gpuserv_debug_output_show(void)
{
	if (gpuserv_shared_state)
	{
		if (pg_atomic_read_u32(&gpuserv_shared_state->gpuserv_debug_output) != 0)
			return "on";
		else
			return "off";
	}
	return (__gpuserv_debug_output_dummy ? "on" : "off");
}

static void
pgstrom_max_async_tasks_assign(int newval, void *extra)
{
	if (!gpuserv_shared_state)
		__pgstrom_max_async_tasks_dummy = newval;
	else
	{
		struct timeval ts;
		uint64_t	conf_val;
		uint64_t	curr_val;

		conf_val = pg_atomic_read_u64(&gpuserv_shared_state->max_async_tasks);
		do {
			if ((conf_val & MAX_ASYNC_TASKS_MASK) == newval)
				break;		/* nothing to do */
			gettimeofday(&ts, NULL);
			curr_val = ((ts.tv_sec  * 1000L +
						 ts.tv_usec / 1000L) << MAX_ASYNC_TASKS_BITS) | (uint64_t)newval;
		} while (!pg_atomic_compare_exchange_u64(&gpuserv_shared_state->max_async_tasks,
												 &conf_val,
												 curr_val));
	}
}

int
pgstrom_max_async_tasks(void)
{
	uint64_t	curr_val;

	if (gpuserv_shared_state)
	{
		curr_val = pg_atomic_read_u64(&gpuserv_shared_state->max_async_tasks);
		return (curr_val & MAX_ASYNC_TASKS_MASK);
	}
	return __pgstrom_max_async_tasks_dummy;
}

static const char *
pgstrom_max_async_tasks_show(void)
{
	return psprintf("%u", pgstrom_max_async_tasks());
}

/*
 * gpuserv_ready_accept
 */
bool
gpuserv_ready_accept(void)
{
	return (gpuserv_shared_state &&
			gpuserv_shared_state->gpuserv_ready_accept);
}

/*
 * gpuservLoggerOpen
 */
static void
gpuservLoggerOpen(void)
{
	struct epoll_event ev;

	if (pipe(gpuserv_logger_pipefd) != 0)
		elog(ERROR, "failed on pipe(2): %m");
	memset(&ev, 0, sizeof(ev));
	ev.events = EPOLLIN;
	ev.data.fd = gpuserv_logger_pipefd[0];
	if (epoll_ctl(gpuserv_epoll_fdesc,
				  EPOLL_CTL_ADD,
				  gpuserv_logger_pipefd[0], &ev) != 0)
		elog(ERROR, "failed on epoll_ctl(2): %m");
	gpuserv_logger_filp = fdopen(gpuserv_logger_pipefd[1], "ab");

	initStringInfo(&gpuserv_logger_buffer);
}

/*
 * gpuservLoggerDispatch
 */
static void
gpuservLoggerDispatch(void)
{
	char   *line, *tail;
	size_t	remained;

	Assert(!GpuWorkerCurrentContext);
	/* read from pipe */
	for (;;)
	{
		size_t	unitsz = 2048;
		ssize_t	nbytes;

		enlargeStringInfo(&gpuserv_logger_buffer, unitsz);

		nbytes = read(gpuserv_logger_pipefd[0],
					  gpuserv_logger_buffer.data + gpuserv_logger_buffer.len,
					  unitsz);
		if (nbytes < 0)
		{
			if (errno != EINTR)
				elog(ERROR, "failed on read(gpuserv_logger_pipefd): %m");
		}
		else
		{
			gpuserv_logger_buffer.len += nbytes;
			if (nbytes < unitsz)
				break;
		}
	}
	gpuserv_logger_buffer.data[gpuserv_logger_buffer.len] = '\0';

	for (line = gpuserv_logger_buffer.data;
		 (tail = strchr(line, '\n')) != NULL;
		 line = tail + 1)
	{
		char	   *tok, *saveptr;
		char	   *domain		= "GPU?";
		char	   *filename	= "unknown file";
		int			lineno		= -1;
		char	   *funcname	= "unknown function";
		char	   *message		= "unknown message";
		int			elevel		= LOG;

		*tail = '\0';
		line = __trim(line);
		if (line[0] == '\0')
			continue;

		tok = strtok_r(line, "|", &saveptr);
		if (tok)
			domain = __trim(tok);
		tok = strtok_r(NULL, "|", &saveptr);
		if (tok)
		{
			char   *elabel = __trim(tok);

			if (strcmp(elabel, "DEBUG") == 0)
				elevel = DEBUG1;
			else if (strcmp(elabel, "NOTICE") == 0)
				elevel = NOTICE;
			else if (strcmp(elabel, "WARNING") == 0)
				elevel = WARNING;
			else
				elevel = LOG;
		}
		tok = strtok_r(NULL, "|", &saveptr);
		if (tok)
			filename = __trim(tok);
		tok = strtok_r(NULL, "|", &saveptr);
		if (tok)
			lineno = atoi(__trim(tok));
		tok = strtok_r(NULL, "|", &saveptr);
		if (tok)
		{
			funcname = __trim(tok);
			message  = saveptr;
		}

		if (errstart(elevel, domain))
		{
			errmsg("%s: %s", domain, message);
			errfinish(filename, lineno, funcname);
		}
	}
	remained = strlen(line);
	if (remained > 0)
	{
		memmove(gpuserv_logger_buffer.data, line, remained);
		gpuserv_logger_buffer.data[remained] = '\0';
		gpuserv_logger_buffer.len = remained;
	}
	else
	{
		gpuserv_logger_buffer.len = 0;
	}
}

/*
 * gpuservLoggerReport
 */
static void
gpuservLoggerReport(const char *fmt, ...)
{
	va_list		ap;

	va_start(ap, fmt);
	vfprintf(gpuserv_logger_filp, fmt, ap);
	va_end(ap);
	fflush(gpuserv_logger_filp);
}

/*
 * cuStrError
 */
const char *
cuStrError(CUresult rc)
{
	static __thread char buffer[300];
	const char	   *err_name;

	/* is it cufile error? */
	if ((int)rc > CUFILEOP_BASE_ERR)
		return cufileop_status_error((CUfileOpError)rc);
	if (cuGetErrorName(rc, &err_name) == CUDA_SUCCESS)
		return err_name;
	snprintf(buffer, sizeof(buffer), "Unknown CUDA Error (%d)", (int)rc);
	return buffer;
}

/*
 * gpuContextSwitchTo
 */
static gpuContext *
__gpuContextSwitchTo(gpuContext *gcontext_new,
					 const char *filename, int lineno)
{
	gpuContext *gcontext_old = GpuWorkerCurrentContext;
	CUresult	rc;

	rc = cuCtxSetCurrent(gcontext_new != NULL
						 ? gcontext_new->cuda_context
						 : NULL);
	if (rc != CUDA_SUCCESS)
		__FATAL("%s:%d - failed on gpuContextSwitchTo: %s",
				filename, lineno, cuStrError(rc));
	GpuWorkerCurrentContext = gcontext_new;
	pg_memory_barrier();
	return gcontext_old;
}
#define gpuContextSwitchTo(gcontext)			\
	__gpuContextSwitchTo((gcontext),__FILE__,__LINE__)

/*
 * THREAD_GPU_CONTEXT_VALIDATION_CHECK
 */
static void
THREAD_GPU_CONTEXT_VALIDATION_CHECK(gpuContext *gcontext)
{
#if 1
	CUcontext	cuda_context;
	CUresult	rc;

	assert(gcontext >= gpuserv_gpucontext_array &&
		   gcontext <  gpuserv_gpucontext_array + numGpuDevAttrs &&
		   gcontext == GpuWorkerCurrentContext);
	rc = cuCtxGetCurrent(&cuda_context);
	if (rc != CUDA_SUCCESS)
		__FATAL("failed on cuCtxGetCurrent: %s", cuStrError(rc));
	assert(gcontext->cuda_context == cuda_context);
#endif
}

/* ----------------------------------------------------------------
 *
 * Open/Close the server socket
 *
 * ----------------------------------------------------------------
 */
static void
gpuservOpenServerSocket(void)
{
	struct sockaddr_un	addr;
	struct epoll_event	ev;

	gpuserv_listen_sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (gpuserv_listen_sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");

	snprintf(addr.sun_path, sizeof(addr.sun_path),
			 ".pg_strom.%u.gpuserv.sock",
			 PostmasterPid);
	addr.sun_family = AF_UNIX;
	if (bind(gpuserv_listen_sockfd,
			 (struct sockaddr *) &addr, sizeof(addr)) != 0)
		elog(ERROR, "failed on bind('%s'): %m", addr.sun_path);
	if (listen(gpuserv_listen_sockfd, 32) != 0)
		elog(ERROR, "failed on listen(2): %m");
	/* register to epoll-fd */
	memset(&ev, 0, sizeof(ev));
	ev.events = EPOLLIN;
	ev.data.fd = gpuserv_listen_sockfd;
	if (epoll_ctl(gpuserv_epoll_fdesc,
				  EPOLL_CTL_ADD,
				  gpuserv_listen_sockfd, &ev) != 0)
		elog(ERROR, "failed on epoll_ctl(2): %m");
}

static void
gpuservCloseServerSocket(void)
{
	dlist_iter	iter;

	if (gpuserv_listen_sockfd >= 0)
		close(gpuserv_listen_sockfd);

	/* terminate client monitor threads */
	pthreadMutexLock(&gpuserv_client_lock);
	dlist_foreach (iter, &gpuserv_client_list)
	{
		gpuClient  *gclient = dlist_container(gpuClient,
											  chain, iter.cur);
		if (close(gclient->sockfd) == 0)
			gclient->sockfd = -1;
	}
	pthreadMutexUnlock(&gpuserv_client_lock);
}

/* ----------------------------------------------------------------
 *
 * GPU Fatbin Builder
 *
 * ----------------------------------------------------------------
 */

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

	strcpy(workdir, ".pgstrom_fatbin_build_XXXXXX");
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
						 " -DCUDA_MAXTHREADS_PER_BLOCK=%u "
						 " -arch=native --threads 4"
						 " --device-c"
						 " -o %s.o"
						 " %s/pg_strom/%s.cu' > %s.log 2>&1",
						 pgstrom_cuda_toolkit_basedir,
						 CUDA_MAXREGCOUNT,
						 PGINCLUDEDIR,
						 CUDA_MAXTHREADS_PER_BLOCK,
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
	appendStringInfo(&cmd, "' > %s.log 2>&1", fatbin_file);

	elog(LOG, "rebuild fatbin command: %s", cmd.data);
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

	/* cleanup working directory */
	resetStringInfo(&cmd);
	appendStringInfo(&cmd, "rm -rf '%s'", workdir);
	status = system(cmd.data);
	if (status != 0)
		elog(ERROR, "failed on shell command: %s", cmd.data);
}

static void
gpuservSetupFatbin(void)
{
	const char *fatbin_file = __setup_gpu_fatbin_filename();
	const char *fatbin_dir  = PGSHAREDIR "/pg_strom";
	char	   *path;
#ifdef WITH_FATBIN
	if (!__validate_gpu_fatbin_file(fatbin_dir,
									fatbin_file))
#endif
	{
		fatbin_dir = PGSTROM_FATBIN_DIR;
		if (!__validate_gpu_fatbin_file(fatbin_dir,
										fatbin_file))
		{
			MemoryContext	curctx = CurrentMemoryContext;

			PG_TRY();
			{
				__rebuild_gpu_fatbin_file(fatbin_dir,
										  fatbin_file);
			}
			PG_CATCH();
			{
				ErrorData  *edata;

				MemoryContextSwitchTo(curctx);
				edata = CopyErrorData();
				FlushErrorState();

				elog(LOG, "[%s:%d] GPU code build error: %s",
					 edata->filename,
					 edata->lineno,
					 edata->message);
				/*
				 * We shall not restart again, until source code
				 * problems are fixed.
				 */
				proc_exit(0);
			}
			PG_END_TRY();
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

/* ----------------------------------------------------------------
 *
 * GPU Memory Allocator
 *
 * ----------------------------------------------------------------
 */
static int		pgstrom_gpu_mempool_segment_sz_kb;	/* GUC */
static double	pgstrom_gpu_mempool_max_ratio;		/* GUC */
static double	pgstrom_gpu_mempool_min_ratio;		/* GUC */
static int		pgstrom_gpu_mempool_release_delay;	/* GUC */
struct gpuMemorySegment
{
	dlist_node		chain;
	gpuMemoryPool  *pool;			/* memory pool that owns this segment */
	size_t			segment_sz;
	size_t			active_sz;		/* == 0 can be released */
	CUdeviceptr		devptr;
	unsigned long	iomap_handle;	/* for legacy nvme_strom */
	dlist_head		free_chunks;	/* list of free chunks */
	dlist_head		addr_chunks;	/* list of ordered chunks */
	struct timeval	tval;
};
typedef struct gpuMemorySegment gpuMemorySegment;

struct gpuMemChunk
{
	dlist_node	free_chain;
	dlist_node	addr_chain;
	gpuMemorySegment *mseg;
	CUdeviceptr	__base;		/* base pointer of the segment */
	size_t		__offset;	/* offset from the base */
	size_t		__length;	/* length of the chunk */
	CUdeviceptr	m_devptr;	/* __base + __offset */
};
typedef struct gpuMemChunk		gpuMemChunk;

static gpuMemChunk *
__gpuMemAllocFromSegment(gpuMemoryPool *pool,
						 gpuMemorySegment *mseg,
						 size_t bytesize)
{
	gpuMemChunk	   *chunk;
	gpuMemChunk	   *buddy;
	dlist_iter		iter;

	dlist_foreach(iter, &mseg->free_chunks)
	{
		chunk = dlist_container(gpuMemChunk, free_chain, iter.cur);

		if (bytesize <= chunk->__length)
		{
			size_t	surplus = chunk->__length - bytesize;

			/* try to split, if free chunk is enough large (>4MB) */
			if (surplus > (4UL << 20))
			{
				buddy = calloc(1, sizeof(gpuMemChunk));
				if (!buddy)
				{
					__gsDebug("out of memory");
					return NULL;	/* out of memory */
				}
				chunk->__length -= surplus;
				chunk->m_devptr = (chunk->__base + chunk->__offset);

				buddy->mseg   = mseg;
				buddy->__base = mseg->devptr;
				buddy->__offset = chunk->__offset + chunk->__length;
				buddy->__length = surplus;
				buddy->m_devptr = (buddy->__base + buddy->__offset);
				dlist_insert_after(&chunk->free_chain, &buddy->free_chain);
				dlist_insert_after(&chunk->addr_chain, &buddy->addr_chain);
			}
			/* mark it as an active chunk */
			dlist_delete(&chunk->free_chain);
			memset(&chunk->free_chain, 0, sizeof(dlist_node));
			mseg->active_sz += chunk->__length;

			/* update the LRU ordered segment list and timestamp */
			gettimeofday(&mseg->tval, NULL);
			dlist_move_head(&pool->segment_list, &mseg->chain);

			return chunk;
		}
	}
	return NULL;
}

static gpuMemorySegment *
__gpuMemAllocNewSegment(gpuMemoryPool *pool, size_t segment_sz)
{
	gpuMemorySegment *mseg = calloc(1, sizeof(gpuMemorySegment));
	gpuMemChunk	   *chunk = calloc(1, sizeof(gpuMemChunk));
	gpuContext	   *gcontext_saved = NULL;
	CUresult		rc;

	if (!mseg || !chunk)
		goto error_0;
	mseg->pool = pool;
	mseg->segment_sz = segment_sz;
	mseg->active_sz = 0;
	dlist_init(&mseg->free_chunks);
	dlist_init(&mseg->addr_chunks);

	gcontext_saved = gpuContextSwitchTo(pool->gcontext);
	if (pool->is_managed)
	{
		rc = cuMemAllocManaged(&mseg->devptr, mseg->segment_sz,
							   CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
		{
			__gsDebug("failed on cuMemAllocManaged(sz=%lu): %s",
					  mseg->segment_sz, cuStrError(rc));
			goto error_1;
		}
		memset((void *)mseg->devptr, 0, mseg->segment_sz);
	}
	else
	{
		rc = cuMemAlloc(&mseg->devptr, mseg->segment_sz);
		if (rc != CUDA_SUCCESS)
		{
			__gsDebug("failed on cuMemAlloc(sz=%lu): %s",
					  mseg->segment_sz, cuStrError(rc));
			goto error_1;
		}
		if (!gpuDirectMapGpuMemory(mseg->devptr,
								   mseg->segment_sz,
								   &mseg->iomap_handle))
		{
			__gsDebugExtra("failed on gpuDirectMapGpuMemory");
			goto error_1;
		}
	}
	gpuContextSwitchTo(gcontext_saved);

	chunk->mseg   = mseg;
	chunk->__base = mseg->devptr;
	chunk->__offset = 0;
	chunk->__length = segment_sz;
	chunk->m_devptr = (chunk->__base + chunk->__offset);
	dlist_push_head(&mseg->free_chunks, &chunk->free_chain);
	dlist_push_head(&mseg->addr_chunks, &chunk->addr_chain);

	dlist_push_head(&pool->segment_list, &mseg->chain);
	pool->total_sz += segment_sz;

	return mseg;
error_1:
	if (mseg->devptr)
		cuMemFree(mseg->devptr);
	if (gcontext_saved)
		gpuContextSwitchTo(gcontext_saved);
error_0:
	if (mseg)
		free(mseg);
	if (chunk)
		free(chunk);
	return NULL;
}

static gpuMemChunk *
__gpuMemAllocCommon(gpuMemoryPool *pool, size_t bytesize)
{
	dlist_iter	iter;
	size_t		segment_sz;
	gpuMemChunk *chunk = NULL;

	bytesize = PAGE_ALIGN(bytesize);
	pthreadMutexLock(&pool->lock);
	dlist_foreach(iter, &pool->segment_list)
	{
		gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
												 chain, iter.cur);
		if (mseg->active_sz + bytesize <= mseg->segment_sz)
		{
			chunk = __gpuMemAllocFromSegment(pool, mseg, bytesize);
			if (chunk)
				goto out_unlock;
		}
	}
	segment_sz = ((size_t)pgstrom_gpu_mempool_segment_sz_kb << 10);
	if (segment_sz < bytesize)
		segment_sz = bytesize;
	/*
	 * total consumption of mapped GPU memory must be less than
	 * the hard-limit of the memory pool.
	 */
	if (pool->is_managed ||
		pool->total_sz + segment_sz <= pool->hard_limit)
	{
		gpuMemorySegment *mseg = __gpuMemAllocNewSegment(pool, segment_sz);

		if (mseg)
			chunk = __gpuMemAllocFromSegment(pool, mseg, bytesize);
	}
	else
	{
		__gsDebug("Raw memory pool exceeds the hard limit (%zu + %zu of %zu) ",
				  pool->total_sz, segment_sz, pool->hard_limit);
	}
out_unlock:	
	pthreadMutexUnlock(&pool->lock);

	return (chunk ? chunk : NULL);
}

static gpuMemChunk *
gpuMemAlloc(size_t bytesize)
{
	return __gpuMemAllocCommon(&GpuWorkerCurrentContext->pool_raw, bytesize);
}

static gpuMemChunk *
gpuMemAllocManaged(size_t bytesize)
{
	return __gpuMemAllocCommon(&GpuWorkerCurrentContext->pool_managed, bytesize);
}

static void
gpuMemFree(gpuMemChunk *chunk)
{
	gpuMemoryPool  *pool;
	gpuMemorySegment *mseg;
	gpuMemChunk	*buddy;
	dlist_node	*dnode;

	Assert(!chunk->free_chain.prev && !chunk->free_chain.next);
	mseg = chunk->mseg;
	pool = mseg->pool;

	pthreadMutexLock(&pool->lock);
	/* revert this chunk state to 'free' */
	mseg->active_sz -= chunk->__length;
	dlist_push_head(&mseg->free_chunks,
					&chunk->free_chain);

	/* try merge if next chunk is also free */
	if (dlist_has_next(&mseg->addr_chunks,
					   &chunk->addr_chain))
	{
		dnode = dlist_next_node(&mseg->addr_chunks,
								&chunk->addr_chain);
		buddy = dlist_container(gpuMemChunk,
								addr_chain, dnode);
		if (buddy->free_chain.prev && buddy->addr_chain.next)
		{
			Assert(chunk->__offset +
				   chunk->__length == buddy->__offset);
			dlist_delete(&buddy->free_chain);
			dlist_delete(&buddy->addr_chain);
			chunk->__length += buddy->__length;
			free(buddy);
		}
	}
	/* try merge if prev chunk is also free */
	if (dlist_has_prev(&mseg->addr_chunks,
					   &chunk->addr_chain))
	{
		dnode = dlist_prev_node(&mseg->addr_chunks,
								&chunk->addr_chain);
		buddy = dlist_container(gpuMemChunk,
								addr_chain, dnode);
		/* merge if prev chunk is also free */
		if (buddy->free_chain.prev && buddy->addr_chain.next)
		{
			Assert(buddy->__offset +
				   buddy->__length == chunk->__offset);
			dlist_delete(&chunk->free_chain);
			dlist_delete(&chunk->addr_chain);
			buddy->__length += chunk->__length;
			free(chunk);
		}
	}
	/* update the LRU ordered segment list and timestamp */
	gettimeofday(&mseg->tval, NULL);
	dlist_move_head(&pool->segment_list, &mseg->chain);
	pthreadMutexUnlock(&pool->lock);
}

/*
 * gpuMemoryPoolMaintenance
 */
static void
__gpuMemoryPoolMaintenanceTask(gpuContext *gcontext, gpuMemoryPool *pool)
{
	dlist_iter		iter;
	struct timeval	tval;
	int64			tdiff;
	CUresult		rc;

	if (!pthreadMutexTryLock(&pool->lock))
		return;
	if (pool->total_sz > pool->keep_limit)
	{
		gettimeofday(&tval, NULL);
		dlist_reverse_foreach(iter, &pool->segment_list)
		{
			gpuMemorySegment *mseg = dlist_container(gpuMemorySegment,
													 chain, iter.cur);
			/* still in active? */
			if (mseg->active_sz != 0)
				continue;

			/* enough time to release is elapsed? */
			tdiff = ((tval.tv_sec  - mseg->tval.tv_sec)  * 1000 +
					 (tval.tv_usec - mseg->tval.tv_usec) / 1000);
			if (tdiff < pgstrom_gpu_mempool_release_delay)
				continue;

			/* ok, this segment should be released */
			if (!pool->is_managed &&
				!gpuDirectUnmapGpuMemory(mseg->devptr,
										 mseg->iomap_handle))
				__FATAL("failed on gpuDirectUnmapGpuMemory");
			rc = cuMemFree(mseg->devptr);
			if (rc != CUDA_SUCCESS)
				__FATAL("failed on cuMemFree: %s", cuStrError(rc));
			/* detach segment */
			dlist_delete(&mseg->chain);
			while (!dlist_is_empty(&mseg->addr_chunks))
			{
				dlist_node	   *dnode = dlist_pop_head_node(&mseg->addr_chunks);
				gpuMemChunk	   *chunk = dlist_container(gpuMemChunk,
														addr_chain, dnode);
				Assert(chunk->free_chain.prev &&
					   chunk->free_chain.next);
				free(chunk);
			}
			__gsDebug("GPU-%d: i/o mapped device memory %lu bytes released",
					  gcontext->cuda_dindex, mseg->segment_sz);
			Assert(pool->total_sz >= mseg->segment_sz);
			pool->total_sz -= mseg->segment_sz;
			free(mseg);
			break;
		}
	}
	pthreadMutexUnlock(&pool->lock);
}

static void
gpuMemoryPoolMaintenance(gpuContext *gcontext)
{
	__gpuMemoryPoolMaintenanceTask(gcontext, &gcontext->pool_raw);
	__gpuMemoryPoolMaintenanceTask(gcontext, &gcontext->pool_managed);
}


static void
gpuMemoryPoolInit(gpuContext *gcontext,
				  bool is_managed,
				  size_t dev_total_memsz)
{
	gpuMemoryPool *pool = (!is_managed
						   ? &gcontext->pool_raw
						   : &gcontext->pool_managed);
	pool->gcontext = gcontext;
	pthreadMutexInit(&pool->lock);
	pool->is_managed = is_managed;
	pool->total_sz = 0;
	pool->hard_limit = pgstrom_gpu_mempool_max_ratio * (double)dev_total_memsz;
	pool->keep_limit = pgstrom_gpu_mempool_min_ratio * (double)dev_total_memsz;
	dlist_init(&pool->segment_list);
}

/*
 * gpuMemoryPrefetchKDS
 */
static CUresult
gpuMemoryPrefetchKDS(kern_data_store *kds,
					 CUdevice dst_device)
{
	CUresult	rc;

	if (kds->format == KDS_FORMAT_ROW ||
		kds->format == KDS_FORMAT_HASH)
	{
		rc = cuMemPrefetchAsync((CUdeviceptr)kds,
								(KDS_HEAD_LENGTH(kds) +
								 sizeof(uint64_t) * kds->hash_nslots +
								 sizeof(uint64_t) * kds->nitems),
								dst_device,
								MY_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			return rc;

		rc = cuMemPrefetchAsync((CUdeviceptr)kds
								+ kds->length
								- kds->usage,
								kds->usage,
								dst_device,
								MY_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			return rc;
	}
	else
	{
		rc = cuMemPrefetchAsync((CUdeviceptr)kds,
								kds->length,
								dst_device,
								MY_STREAM_PER_THREAD);
	}
	return rc;
}

/* ----------------------------------------------------------------
 *
 * gpuClientELog
 *
 * ----------------------------------------------------------------
 */
static void
__gpuClientWriteBack(gpuClient *gclient,
					 struct iovec *iov, int iovcnt);
static void
__gpuClientELog(gpuClient *gclient,
				int errcode,
				const char *filename, int lineno,
				const char *funcname,
				const char *fmt, ...)	pg_attribute_printf(6,7);

#define gpuClientELog(gclient,fmt,...)						\
	__gpuClientELog((gclient), ERRCODE_DEVICE_ERROR,		\
					__FILE__, __LINE__, __FUNCTION__,		\
					(fmt), ##__VA_ARGS__)
#define gpuClientFatal(gclient,fmt,...)						\
	__gpuClientELog((gclient), ERRCODE_DEVICE_FATAL,		\
					__FILE__, __LINE__, __FUNCTION__,		\
					(fmt), ##__VA_ARGS__)

static void
__gpuClientELogRaw(gpuClient *gclient, kern_errorbuf *errorbuf)
{
	XpuCommand		resp;
	struct iovec	iov;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Error;
	resp.length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	memcpy(&resp.u.error, errorbuf, sizeof(kern_errorbuf));

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);
}

static void
__gpuClientELog(gpuClient *gclient,
				int errcode,
				const char *filename, int lineno,
				const char *funcname,
				const char *fmt, ...)
{
	XpuCommand		resp;
	va_list			ap;
	struct iovec	iov;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Error;
	resp.length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	resp.u.error.errcode = errcode,
	resp.u.error.lineno = lineno;
	strncpy(resp.u.error.filename, __basename(filename),
			KERN_ERRORBUF_FILENAME_LEN);
	strncpy(resp.u.error.funcname, funcname,
			KERN_ERRORBUF_FUNCNAME_LEN);

	va_start(ap, fmt);
	vsnprintf(resp.u.error.message, KERN_ERRORBUF_MESSAGE_LEN, fmt, ap);
	va_end(ap);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);

	/* unable to continue GPU service, so try to restart */
	if (errcode == ERRCODE_DEVICE_FATAL)
	{
		__gsDebug("(%s:%d, %s) GPU fatal - %s\n",
				  resp.u.error.filename,
				  resp.u.error.lineno,
				  resp.u.error.funcname,
				  resp.u.error.message);
		gpuserv_bgworker_got_signal |= (1 << SIGHUP);
		pg_memory_barrier();
		SetLatch(MyLatch);
	}
}

static void
gpuClientELogByExtraModule(gpuClient *gclient)
{
	int				errcode;
	const char	   *filename;
	unsigned int	lineno;
	const char	   *funcname;
	char			buffer[2000];

	errcode = heterodbExtraGetError(&filename,
									&lineno,
									&funcname,
									buffer, sizeof(buffer));
	if (errcode == 0)
		gpuClientELog(gclient, "Bug? %s is called but no error status", __FUNCTION__);
	else
		__gpuClientELog(gclient,
						errcode,
						filename, lineno,
						funcname,
						"extra-module: %s", buffer);
}

/* ----------------------------------------------------------------
 *
 * Session buffer support routines
 *
 * This buffer is used by GpuJoin's inner buffer and GpuPreAgg.
 * It is kept until session end, and can be shared by multiple sessions.
 *
 * ----------------------------------------------------------------
 */
struct gpuQueryBuffer
{
	dlist_node		chain;
	volatile int	refcnt;
	volatile int	phase;			/*  0: not initialized,
									 *  1: buffer is ready,
									 * -1: error, during buffer setup */
	uint64_t		buffer_id;		/* unique buffer id */
	CUdeviceptr		m_kmrels;		/* GpuJoin inner buffer (device) */
	void		   *h_kmrels;		/* GpuJoin inner buffer (host) */
	size_t			kmrels_sz;		/* GpuJoin inner buffer size */
	pthread_rwlock_t m_kds_rwlock;	/* RWLock for the final/fallback buffer */
	int				gpumem_nitems;
	int				gpumem_nrooms;
	CUdeviceptr	   *gpumem_devptrs;
	struct {
		CUdeviceptr	m_kds_final;		/* final buffer (device) */
		size_t		m_kds_final_length;	/* length of final buffer */
		CUdeviceptr	m_kds_fallback;		/* CPU-fallback buffer */
		uint32_t	m_kds_fallback_revision;	/* revision of CPU-fallback buffer */
	} gpus[1];		/* per-device final/fallback buffer */
};
typedef struct gpuQueryBuffer		gpuQueryBuffer;

#define GQBUF_KDS_FINAL(gq_buf)					\
	((kern_data_store *)(gq_buf)->gpus[MY_DINDEX_PER_THREAD].m_kds_final)
#define GQBUF_KDS_FINAL_LENGTH(gq_buf)			\
	((gq_buf)->gpus[MY_DINDEX_PER_THREAD].m_kds_final_length)
#define GQBUF_KDS_FALLBACK(gq_buf)				\
	((kern_data_store *)(gq_buf)->gpus[MY_DINDEX_PER_THREAD].m_kds_fallback)
#define GQBUF_KDS_FALLBACK_REVISION(gq_buf)		\
	((gq_buf)->gpus[MY_DINDEX_PER_THREAD].m_kds_fallback_revision)

#define GPU_QUERY_BUFFER_NSLOTS		320
static dlist_head		gpu_query_buffer_hslot[GPU_QUERY_BUFFER_NSLOTS];
static pthread_mutex_t	gpu_query_buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t	gpu_query_buffer_cond = PTHREAD_COND_INITIALIZER;

static gpuQueryBuffer *__getGpuQueryBuffer(uint64_t buffer_id, bool may_create);

static bool
__enlargeGpuQueryBuffer(gpuQueryBuffer *gq_buf)
{
	if (gq_buf->gpumem_nitems >= gq_buf->gpumem_nrooms)
	{
		CUdeviceptr *__devptrs;
		int		__nrooms = (2 * gq_buf->gpumem_nrooms + 20);

		__devptrs = realloc(gq_buf->gpumem_devptrs,
							sizeof(CUdeviceptr) * __nrooms);
		if (!__devptrs)
			return false;
		gq_buf->gpumem_devptrs = __devptrs;
		gq_buf->gpumem_nrooms  = __nrooms;
    }
	return true;
}

static CUresult
allocGpuQueryBuffer(gpuQueryBuffer *gq_buf,
					CUdeviceptr *p_devptr, size_t bytesize)
{
	CUdeviceptr	m_devptr;
	CUresult	rc;

	if (!__enlargeGpuQueryBuffer(gq_buf))
		return CUDA_ERROR_OUT_OF_MEMORY;
	/* allocation */
	rc = cuMemAllocManaged(&m_devptr, bytesize,
						   CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		return rc;
	gq_buf->gpumem_devptrs[gq_buf->gpumem_nitems++] = m_devptr;
	*p_devptr = m_devptr;

	__gsLog("Query buffer allocation at %p (sz=%s)",
			(void *)m_devptr, format_bytesz(bytesize));

	return CUDA_SUCCESS;
}

static bool
releaseGpuQueryBufferOne(gpuQueryBuffer *gq_buf,
						 CUdeviceptr m_devptr)
{
	for (int i=0; i < gq_buf->gpumem_nitems; i++)
	{
		CUresult	rc;

		if (gq_buf->gpumem_devptrs[i] == m_devptr)
		{
			gq_buf->gpumem_devptrs[i] = 0;
			rc = cuMemFree(m_devptr);
			if (rc != CUDA_SUCCESS)
				__FATAL("failed on cuMemFree(%p): %s",
						(void *)m_devptr, cuStrError(rc));
			__gsLog("Query buffer release one at %p",
					(void *)m_devptr);
			return true;
		}
	}
	return false;
}


static void
releaseGpuQueryBufferAll(gpuQueryBuffer *gq_buf)
{
	for (int i=0; i < gq_buf->gpumem_nitems; i++)
	{
		CUdeviceptr	m_devptr = gq_buf->gpumem_devptrs[i];
		CUresult	rc;

		if (m_devptr != 0UL)
		{
			rc = cuMemFree(m_devptr);
			if (rc != CUDA_SUCCESS)
				__FATAL("failed on cuMemFree(%p): %s",
						(void *)m_devptr, cuStrError(rc));
			__gsLog("Query buffer release all at %p",
					(void *)m_devptr);
		}
	}
	gq_buf->gpumem_nitems = 0;
}

static void
__putGpuQueryBufferNoLock(gpuQueryBuffer *gq_buf)
{
	Assert(gq_buf->refcnt > 0);
	if (--gq_buf->refcnt == 0)
	{
		releaseGpuQueryBufferAll(gq_buf);
		if (gq_buf->h_kmrels)
		{
			if (munmap(gq_buf->h_kmrels,
					   gq_buf->kmrels_sz) != 0)
				__gsDebug("failed on munmap: %m");
		}
		dlist_delete(&gq_buf->chain);
		free(gq_buf);
	}
}

static void
putGpuQueryBuffer(gpuQueryBuffer *gq_buf)
{
	pthreadMutexLock(&gpu_query_buffer_mutex);
	__putGpuQueryBufferNoLock(gq_buf);
	pthreadMutexUnlock(&gpu_query_buffer_mutex);
}

static bool
__setupGpuQueryJoinGiSTIndexBuffer(gpuClient *gclient,
								   gpuQueryBuffer *gq_buf)
{
	gpuContext *gcontext = GpuWorkerCurrentContext;
	kern_multirels *h_kmrels = gq_buf->h_kmrels;
	CUresult	rc;
	int			grid_sz;
	int			block_sz;
	void	   *kern_args[10];
	bool		has_gist = false;

	for (int depth=1; depth <= h_kmrels->num_rels; depth++)
	{
		if (h_kmrels->chunks[depth-1].gist_offset == 0)
			continue;

		Assert(gcontext && gcontext->cufn_prep_gistindex);
		rc = gpuOptimalBlockSize(&grid_sz,
								 &block_sz,
								 gcontext->cufn_prep_gistindex, 0);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on gpuOptimalBlockSize: %s",
						  cuStrError(rc));
			return false;
		}
		kern_args[0] = &gq_buf->m_kmrels;
		kern_args[1] = &depth;
		rc = cuLaunchKernel(gcontext->cufn_prep_gistindex,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							0,
							MY_STREAM_PER_THREAD,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on cuLaunchKernel: %s",
						  cuStrError(rc));
			return false;
		}
		has_gist = true;
	}

	if (has_gist)
	{
		rc = cuStreamSynchronize(MY_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on cuStreamSynchronize: %s",
						  cuStrError(rc));
			return false;
		}
	}
	return true;
}

static gpuContext *
__lookupOneRandomGpuContext(gpumask_t candidate_gpus)
{
	gpuContext *gcontexts[sizeof(gpumask_t) * BITS_PER_BYTE];
	int		nitems = 0;
	int		index;

	for (int dindex=0; candidate_gpus != 0 && dindex < numGpuDevAttrs; dindex++)
	{
		gpumask_t	__mask = (1UL << dindex);

		if ((candidate_gpus & __mask) != 0)
		{
			gcontexts[nitems++] = &gpuserv_gpucontext_array[dindex];
			candidate_gpus &= ~__mask;
		}
	}
	if (nitems == 0)
		return NULL;
	index = floor((double)nitems * drand48());
	assert(index >= 0 && index < nitems);
	return gcontexts[index];
}

static bool
__setupGpuJoinPinnedInnerBufferPartitioned(gpuClient *gclient,
										   gpuQueryBuffer *gq_buf,
										   gpuQueryBuffer *gq_src,
										   kern_data_store *kds_head,
										   kern_buffer_partitions *kbuf_parts)
{
	gpuContext *gcontext_saved = GpuWorkerCurrentContext;
	uint32_t	hash_divisor = kbuf_parts->hash_divisor;
	CUresult	rc;
	struct timeval	tv1, tv2, tv3, tv4;

	/* allocation of kds_in for each partition */
	for (int i=0; i < hash_divisor; i++)
	{
		CUdeviceptr	m_kds_in;

		rc = allocGpuQueryBuffer(gq_buf, &m_kds_in, kds_head->length);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on allocGpuQueryBuffer(sz=%lu): %s",
						  kds_head->length, cuStrError(rc));
			return false;
		}
		memcpy((void *)m_kds_in, kds_head, KDS_HEAD_LENGTH(kds_head));
		kbuf_parts->parts[i].kds_in = (kern_data_store *)m_kds_in;
	}
	/* partitioning for each GPU device */
	gettimeofday(&tv1, NULL);
	for (int i=0; i < numGpuDevAttrs; i++)
	{
		CUdeviceptr	m_kds_src = gq_src->gpus[i].m_kds_final;
		gpuContext *gcontext = &gpuserv_gpucontext_array[i];
		gpuContext *gcontext_prev;
		int			grid_sz;
		int			block_sz;
		void	   *kern_args[5];

		if (m_kds_src == 0UL)
			continue;
		gcontext_prev = gpuContextSwitchTo(gcontext);

		/* launch buffer reconstruction kernel function for each source */
		rc = gpuOptimalBlockSize(&grid_sz,
								 &block_sz,
								 gcontext->cufn_kbuf_partitioning,
								 sizeof(uint64_t) * hash_divisor);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on gpuOptimalBlockSize: %s",
						  cuStrError(rc));
			goto error;
		}
		kern_args[0] = &kbuf_parts;
		kern_args[1] = &m_kds_src;
		rc = cuLaunchKernel(gcontext->cufn_kbuf_partitioning,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							sizeof(uint64_t) * hash_divisor,
							MY_STREAM_PER_THREAD,
							kern_args,
							0);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on cuLaunchKernel: %s",
						  cuStrError(rc));
			goto error;
		}
		rc = cuStreamSynchronize(MY_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on cuStreamSynchronize: %s",
						  cuStrError(rc));
			goto error;
		}
		gpuContextSwitchTo(gcontext_prev);

		if (!releaseGpuQueryBufferOne(gq_src, m_kds_src))
		{
			gpuClientELog(gclient, "unable to release source of pinned-inner buffer");
			goto error;
		}
		gq_src->gpus[i].m_kds_final = 0UL;
		gq_src->gpus[i].m_kds_final_length = 0UL;
	}
	/* setup memory attribute for each partition */
	gettimeofday(&tv2, NULL);
	for (int i=hash_divisor-1; i >= 0; i--)
	{
		kern_data_store *kds_in = kbuf_parts->parts[i].kds_in;
		gpumask_t	available_gpus = kbuf_parts->parts[i].available_gpus;

		/*
		 * migrate the inner-buffer partitions to be used in the 2nd
		 * or later repeats into the host-memory
		 */
		if (i > numGpuDevAttrs)
		{
			rc = gpuMemoryPrefetchKDS(kds_in, CU_DEVICE_CPU);
			if (rc != CUDA_SUCCESS)
			{
				gpuClientELog(gclient, "failed on gpuMemoryPrefetchKDS: %s",
							  cuStrError(rc));
				return false;
			}
			continue;
		}
		/*
		 * migrate the inner-buffer partitions to the primary GPU device.
		 */
		for (int k=0; k < numGpuDevAttrs; k++)
		{
			gpuContext *gcontext = &gpuserv_gpucontext_array[k];
			gpuContext *gcontext_prev;

			if ((available_gpus & (1UL << k)) != 0)
			{
				gcontext_prev = gpuContextSwitchTo(gcontext);
				/* Prefetch the inner-buffer */
				rc = gpuMemoryPrefetchKDS(kds_in, gcontext->cuda_device);
				if (rc != CUDA_SUCCESS)
				{
					gpuClientELog(gclient, "failed on gpuMemoryPrefetchKDS: %s",
								  cuStrError(rc));
				}
				gpuContextSwitchTo(gcontext_prev);
				break;
			}
		}
	}
	gettimeofday(&tv3, NULL);

	/*
	 * Change the memory policy, and distribute inner-buffers to the
	 * secondary devices.
	 */
	for (int i=0; i < hash_divisor; i++)
	{
		kern_data_store *kds_in = kbuf_parts->parts[i].kds_in;
		gpumask_t	available_gpus = kbuf_parts->parts[i].available_gpus;
		bool		meet_first = false;

		/* inner-buffer partition is not distributed to multiple GPUs */
		if (get_bitcount(available_gpus) <= 1)
			continue;

		for (int k=0; k < numGpuDevAttrs; k++)
		{
			gpuContext *gcontext = &gpuserv_gpucontext_array[k];
			gpuContext *gcontext_prev;

			if ((available_gpus & gcontext->cuda_dmask) == 0)
				continue;

			gcontext_prev = gpuContextSwitchTo(gcontext);
			if (!meet_first)
			{
				rc = cuMemAdvise((CUdeviceptr)kds_in,
								 (KDS_HEAD_LENGTH(kds_in) +
								  sizeof(uint64_t) * kds_in->hash_nslots +
								  sizeof(uint64_t) * kds_in->nitems),
								 CU_MEM_ADVISE_SET_READ_MOSTLY, -1);
				if (rc != CUDA_SUCCESS)
				{
					gpuContextSwitchTo(gcontext_prev);
					gpuClientELog(gclient, "failed on cuMemAdvise(SET_READ_MOSTLY): %s",
								  cuStrError(rc));
					return false;
				}
				rc = cuMemAdvise((CUdeviceptr)kds_in
								 + kds_in->length
								 - kds_in->usage,
								 kds_in->usage,
								 CU_MEM_ADVISE_SET_READ_MOSTLY, -1);
				if (rc != CUDA_SUCCESS)
				{
					gpuContextSwitchTo(gcontext_prev);
					gpuClientELog(gclient, "failed on cuMemAdvise(SET_READ_MOSTLY): %s",
								  cuStrError(rc));
					return false;
				}
				meet_first = true;
			}
			else
			{
				/* Prefetch the inner-buffer */
				rc = gpuMemoryPrefetchKDS(kds_in, gcontext->cuda_device);
				if (rc != CUDA_SUCCESS)
				{
					gpuContextSwitchTo(gcontext_prev);
					gpuClientELog(gclient, "failed on gpuMemoryPrefetchKDS: %s",
								  cuStrError(rc));
					return false;
				}
			}
			gpuContextSwitchTo(gcontext_prev);
		}
	}
	gettimeofday(&tv4, NULL);

	/* print partition information */
	for (int i=0; i < hash_divisor; i++)
	{
		kern_data_store	*kds_in = kbuf_parts->parts[i].kds_in;
		size_t		consumed
			= (KDS_HEAD_LENGTH(kds_in)
			   + sizeof(uint64_t) * kds_in->hash_nslots
			   + sizeof(uint64_t) * kds_in->nitems
			   + kds_in->usage);
		__gsLog("inner-buffer partition (%u of %d, nitems=%u, nslots=%u, usage=%s, consumed=%s)",
				i, hash_divisor,
				kds_in->nitems,
				kds_in->hash_nslots,
				format_bytesz(kds_in->usage),
				format_bytesz(consumed));
	}
	__gsLog("inner-buffer partitioning (rebuild: %s, prefetch: %s, set-policy: %s)",
			format_millisec((double)((tv2.tv_sec  - tv1.tv_sec)  * 1000 +
									 (tv2.tv_usec - tv1.tv_usec) / 1000)),
			format_millisec((double)((tv3.tv_sec  - tv2.tv_sec)  * 1000 +
									 (tv3.tv_usec - tv2.tv_usec) / 1000)),
			format_millisec((double)((tv4.tv_sec  - tv3.tv_sec)  * 1000 +
									 (tv4.tv_usec - tv3.tv_usec) / 1000)));
	return true;

error:
	gpuContextSwitchTo(gcontext_saved);
	return false;
}

static bool
__setupGpuJoinPinnedInnerBufferReconstruct(gpuClient *gclient,
										   gpuQueryBuffer *gq_buf,
										   gpuQueryBuffer *gq_src,
										   kern_data_store *kds_head,
										   kern_multirels *m_kmrels,
										   int depth)
{
	CUdeviceptr	m_kds_in;
	CUresult	rc;
	int			gpu_count = 0;
	struct timeval tv1, tv2;

	gettimeofday(&tv1, NULL);
	rc = allocGpuQueryBuffer(gq_buf, &m_kds_in, kds_head->length);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on allocGpuQueryBuffer(sz=%lu): %s",
					  kds_head->length, cuStrError(rc));
		return false;
	}
	memcpy((void *)m_kds_in, kds_head, KDS_HEAD_LENGTH(kds_head));

	/*
	 * Move kds_final to kds_in for each GPU devices
	 */
	for (int i=0; i < numGpuDevAttrs; i++)
	{
		CUdeviceptr	m_kds_src = gq_src->gpus[i].m_kds_final;
		gpuContext *gcontext = &gpuserv_gpucontext_array[i];
		gpuContext *gcontext_prev;
		int			grid_sz;
		int			block_sz;
		void	   *kern_args[4];

		if (m_kds_src == 0UL)
			continue;

		gcontext_prev = gpuContextSwitchTo(gcontext);

		/* launch buffer reconstruction kernel function */
		rc = gpuOptimalBlockSize(&grid_sz,
								 &block_sz,
								 gcontext->cufn_kbuf_reconstruction, 0);
		if (rc != CUDA_SUCCESS)
		{
			gpuContextSwitchTo(gcontext_prev);
			gpuClientELog(gclient, "failed on gpuOptimalBlockSize: %s",
						  cuStrError(rc));
			return false;
		}
		/* runs-reconstruction kernel */
		kern_args[0] = &m_kds_in;
		kern_args[1] = &m_kds_src;
		rc = cuLaunchKernel(gcontext->cufn_kbuf_reconstruction,
							grid_sz, 1, 1,
							block_sz, 1, 1,
							0,
							MY_STREAM_PER_THREAD,
							kern_args,
							0);
		if (rc != CUDA_SUCCESS)
		{
			gpuContextSwitchTo(gcontext_prev);
			gpuClientELog(gclient, "failed on cuLaunchKernel: %s",
						  cuStrError(rc));
			return false;
		}
		/* wait for completion */
		rc = cuStreamSynchronize(MY_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuContextSwitchTo(gcontext_prev);
			gpuClientELog(gclient, "failed on cuStreamSynchronize: %s",
						  cuStrError(rc));
			return false;
		}
		gpuContextSwitchTo(gcontext_prev);

		/* release m_kds_src */
		if (!releaseGpuQueryBufferOne(gq_src, m_kds_src))
		{
			gpuClientELog(gclient, "unable to release source of pinned-inner buffer");
			return false;
		}
		gq_src->gpus[i].m_kds_final = 0UL;
		gq_src->gpus[i].m_kds_final_length = 0UL;
		gpu_count++;
	}
	/* assign read-only attribute */
	rc = cuMemAdvise(m_kds_in, kds_head->length,
					 CU_MEM_ADVISE_SET_READ_MOSTLY, -1);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on cuMemAdvise(SET_READ_MOSTLY): %s",
					  cuStrError(rc));
		return false;
	}
	gettimeofday(&tv2, NULL);
	m_kmrels->chunks[depth-1].kds_in = (kern_data_store *)m_kds_in;

	__gsLog("inner-buffer reconstruction (%u GPUs, nitems=%u, nslots=%u, usage=%s, consumption=%s): %s",
			gpu_count,
			((kern_data_store *)m_kds_in)->nitems,
			((kern_data_store *)m_kds_in)->hash_nslots,
			format_bytesz(((kern_data_store *)m_kds_in)->usage),
			format_bytesz(kds_head->length),
			format_millisec((double)((tv2.tv_sec  - tv1.tv_sec)  * 1000 +
									 (tv2.tv_usec - tv1.tv_usec) / 1000)));
	return true;
}

static bool
__setupGpuJoinPinnedInnerBufferZeroCopy(gpuClient *gclient,
										gpuQueryBuffer *gq_buf,
										gpuQueryBuffer *gq_src,
										kern_data_store *kds_src,
										kern_multirels *m_kmrels,
										int depth)
{
	if (!__enlargeGpuQueryBuffer(gq_buf))
		return CUDA_ERROR_OUT_OF_MEMORY;
	/* move the ownership of kds_src to gq_buf */
	for (int i=0; i < gq_src->gpumem_nitems; i++)
	{
		if (gq_src->gpumem_devptrs[i] == (CUdeviceptr)kds_src)
		{
			size_t	consumed;

			gq_src->gpumem_devptrs[i] = 0UL;
			gq_buf->gpumem_devptrs[gq_buf->gpumem_nitems++] = (CUdeviceptr)kds_src;
			m_kmrels->chunks[depth-1].kds_in = kds_src;
			consumed = (KDS_HEAD_LENGTH(kds_src) +
						sizeof(uint64_t) * kds_src->nitems +
						sizeof(uint64_t) * kds_src->hash_nslots +
						kds_src->usage);
			__gsLog("pinned inner buffer zero-copy (total: nitems=%u, usage=%s, consumption=%s of %s allocated)",
					kds_src->nitems,
					format_bytesz(kds_src->usage),
					format_bytesz(consumed),
					format_bytesz(kds_src->length));
			return true;
		}
	}
	gpuClientELog(gclient, "unable to obtain GPU Query Buffer");
	return false;
}

static bool
__setupGpuJoinPinnedInnerBufferCommon(gpuClient *gclient,
									  gpuQueryBuffer *gq_buf,
									  kern_multirels *m_kmrels, int depth)
{
	uint64_t			buffer_id = m_kmrels->chunks[depth-1].buffer_id;
	kern_buffer_partitions *kbuf_parts
		= KERN_MULTIRELS_PARTITION_DESC(m_kmrels, depth);
	gpuQueryBuffer	   *gq_src;
	kern_data_store	   *kds_src = NULL;
	kern_data_store	   *kds_head;
	int					kds_nchunks = 0;
	uint64_t			kds_nitems = 0;
	uint64_t			kds_usage = 0;
	uint64_t			kds_nslots = 0;
	uint64_t			kds_length;
	bool				retval = false;

	/* lookup the result buffer in the previous level */
	gq_src = __getGpuQueryBuffer(buffer_id, false);
	if (!gq_src)
	{
		gpuClientELog(gclient, "pinned inner buffer[%d] was not found", depth);
		return false;
	}
	pthreadRWLockWriteLock(&gq_src->m_kds_rwlock);

	/*
	 * Setup kds_head of the new inner-buffer, if necessary
	 */
	for (int i=0; i < numGpuDevAttrs; i++)
	{
		kern_data_store *__kds = (kern_data_store *)gq_src->gpus[i].m_kds_final;

		if (__kds)
		{
			if (kds_src == NULL)
				kds_src = __kds;
			kds_nchunks++;
			kds_nitems += __kds->nitems;
			kds_usage  += __kds->usage;
		}
	}

	if (!kds_src)
	{
		gpuClientELog(gclient, "No pinned inner buffer[%d] was found", depth);
		pthreadRWLockUnlock(&gq_src->m_kds_rwlock);
		return false;
	}
	kds_head = alloca(KDS_HEAD_LENGTH(kds_src));
	kds_nslots = KDS_GET_HASHSLOT_WIDTH(kds_nitems);

	memcpy(kds_head, kds_src, KDS_HEAD_LENGTH(kds_src));
	kds_length = KDS_HEAD_LENGTH(kds_head)
		+ sizeof(uint64_t) * kds_nslots		/* hash-nslots */
		+ sizeof(uint64_t) * kds_nitems		/* row-index */
		+ (1UL<<30) + kds_usage;			/* tuples (with 1GB margin) */
	kds_head->length = kds_length;
	kds_head->usage  = 0;
	kds_head->nitems = 0;
	kds_head->hash_nslots = kds_nslots;

	if (kbuf_parts)
	{
		/* with inner-buffer partitioning */
		if (!__setupGpuJoinPinnedInnerBufferPartitioned(gclient,
														gq_buf,
														gq_src,
														kds_head,
														kbuf_parts))
			goto error;
		m_kmrels->chunks[depth-1].kbuf_parts = kbuf_parts;
	}
	else if (kds_nchunks > 1)
	{
		/* with inner-buffer reconstruction */
		if (!__setupGpuJoinPinnedInnerBufferReconstruct(gclient,
														gq_buf,
														gq_src,
														kds_head,
														m_kmrels,
														depth))
			goto error;
	}
	else
	{
		if (!__setupGpuJoinPinnedInnerBufferZeroCopy(gclient,
													 gq_buf,
													 gq_src,
													 kds_src,
													 m_kmrels,
													 depth))
			goto error;
	}
	retval = true;
error:
	pthreadRWLockUnlock(&gq_src->m_kds_rwlock);
	putGpuQueryBuffer(gq_src);
	return retval;
}

static bool
__setupGpuQueryJoinInnerBuffer(gpuClient *gclient,
							   gpuQueryBuffer *gq_buf,
							   uint32_t kmrels_handle)
{
	kern_multirels *h_kmrels = NULL;
	kern_multirels *m_kmrels = NULL;
	CUresult	rc;
	int			fdesc;
	struct stat	stat_buf;
	char		namebuf[100];
	size_t		mmap_sz = 0;

	if (kmrels_handle == 0)
		return true;

	snprintf(namebuf, sizeof(namebuf),
			 ".pgstrom_shmbuf_%u_%d",
			 PostPortNumber, kmrels_handle);
	fdesc = shm_open(namebuf, O_RDWR, 0600);
	if (fdesc < 0)
	{
		gpuClientELog(gclient, "failed on shm_open('%s'): %m", namebuf);
		goto error;
	}
	if (fstat(fdesc, &stat_buf) != 0)
	{
		gpuClientELog(gclient, "failed on fstat('%s'): %m", namebuf);
		goto error;
	}
	mmap_sz = PAGE_ALIGN(stat_buf.st_size);

	h_kmrels = mmap(NULL, mmap_sz,
					PROT_READ | PROT_WRITE,
					MAP_SHARED,
					fdesc, 0);
	if (h_kmrels == MAP_FAILED)
	{
		gpuClientELog(gclient, "failed on mmap('%s', %zu): %m",
					  namebuf, mmap_sz);
		goto error;
	}
	/*
	 * MEMO: The outer-join-map locates on the tail of h_kmrels,
	 * and the host-buffer has only one set of the OJMap, however,
	 * it shoud be prepared for each device because concurrent
	 * writes to OJMap by multiple GPUs will cause huge slashing.
	 * So, we setup mutiple OJMap on the device memory for each
	 * GPUs, then consolidate them in the final request handler.
	 */
	rc = allocGpuQueryBuffer(gq_buf,
							 (CUdeviceptr *)&m_kmrels,
							 mmap_sz +
							 h_kmrels->ojmap_sz * (numGpuDevAttrs-1));
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on cuMemAllocManaged: %s",
					  cuStrError(rc));
		goto error;
	}
	memcpy(m_kmrels, h_kmrels, mmap_sz);
	for (int depth=1; depth <= m_kmrels->num_rels; depth++)
	{
		if (m_kmrels->chunks[depth-1].pinned_buffer)
		{
			if (!__setupGpuJoinPinnedInnerBufferCommon(gclient,
													   gq_buf,
													   m_kmrels, depth))
				goto error;
		}
		else
		{
			uint64_t	kds_offset = m_kmrels->chunks[depth-1].kds_offset;

			assert(kds_offset > 0 && kds_offset < m_kmrels->length);
			m_kmrels->chunks[depth-1].kds_in = (kern_data_store *)
				((char *)m_kmrels + kds_offset);
		}
	}
	gq_buf->m_kmrels = (CUdeviceptr)m_kmrels;
	gq_buf->h_kmrels = h_kmrels;
	gq_buf->kmrels_sz = mmap_sz;

	/* preparation of GiST-index buffer, if any */
	if (!__setupGpuQueryJoinGiSTIndexBuffer(gclient, gq_buf))
		goto error;

	/*
	 * MEMO: kern_multirels deploys the outer-join map (that is only read-writable
	 * portion on the m_kmrels buffer) at tail of the m_kmrels buffer.
	 * So, we expect the CUDA unified memory system distributes the read-only
	 * portion for each devices on the demand.
	 * cuMemAdvise() allows to give driver a hint for this purpose.
	 */
	assert(h_kmrels->ojmap_sz < h_kmrels->length);
	rc = cuMemAdvise((CUdeviceptr)m_kmrels,
					 h_kmrels->length - h_kmrels->ojmap_sz,
					 CU_MEM_ADVISE_SET_READ_MOSTLY,
					 -1);	/* 4th argument should be ignored */
	if (rc != CUDA_SUCCESS)
		__gsLog("failed on cuMemAdvise (%p-%p; CU_MEM_ADVISE_SET_READ_MOSTLY): %s",
				(char *)m_kmrels,
				(char *)m_kmrels + (h_kmrels->length - h_kmrels->ojmap_sz),
				cuStrError(rc));
	/* OK */
	close(fdesc);
	return true;

error:
	releaseGpuQueryBufferAll(gq_buf);
	if (h_kmrels != NULL &&
		h_kmrels != MAP_FAILED)
		munmap(h_kmrels, mmap_sz);
	if (fdesc >= 0)
		close(fdesc);
	return false;
}

static bool
__setupGpuQueryGroupByBuffer(gpuClient *gclient,
							 gpuQueryBuffer *gq_buf,
							 kern_data_store *kds_final_head)
{
	dlist_iter	iter;

	if (!kds_final_head)
		return true;	/* nothing to do */

	Assert(KDS_HEAD_LENGTH(kds_final_head) <= kds_final_head->length);
	dlist_foreach (iter, &gpuserv_gpucontext_list)
	{
		gpuContext *__gcontext = dlist_container(gpuContext, chain, iter.cur);
		gpuContext *__gcontext_prev;
		int			__dindex = __gcontext->cuda_dindex;
		CUdeviceptr	m_kds_final;
		CUresult	rc;

		if ((gclient->optimal_gpus & __gcontext->cuda_dmask) == 0)
			continue;

		__gcontext_prev = gpuContextSwitchTo(__gcontext);
		rc = allocGpuQueryBuffer(gq_buf,
								 &m_kds_final,
								 kds_final_head->length);
		if (rc != CUDA_SUCCESS)
		{
			gpuContextSwitchTo(__gcontext_prev);
			gpuClientELog(gclient, "failed on cuMemAllocManaged(%zu): %s",
						  kds_final_head->length, cuStrError(rc));
			releaseGpuQueryBufferAll(gq_buf);
			return false;
		}
		memcpy((void *)m_kds_final,
			   kds_final_head,
			   KDS_HEAD_LENGTH(kds_final_head));
		/* prefetch to the device */
		(void)cuMemPrefetchAsync(m_kds_final,
								 KDS_HEAD_LENGTH(kds_final_head),
								 __gcontext->cuda_device,
								 MY_STREAM_PER_THREAD);
		gq_buf->gpus[__dindex].m_kds_final = m_kds_final;
		gq_buf->gpus[__dindex].m_kds_final_length = kds_final_head->length;
		gpuContextSwitchTo(__gcontext_prev);
	}
	return true;
}

/*
 * __expandGpuQueryGroupByBuffer
 */
static bool
__expandGpuQueryGroupByBuffer(gpuClient *gclient,
							  gpuQueryBuffer *gq_buf,
							  size_t kds_length_last)
{
	pthreadRWLockWriteLock(&gq_buf->m_kds_rwlock);
	if (GQBUF_KDS_FINAL_LENGTH(gq_buf) == kds_length_last)
	{
		kern_data_store *kds_old = GQBUF_KDS_FINAL(gq_buf);
		kern_data_store *kds_new;
		CUdeviceptr		m_devptr;
		CUresult		rc;
		size_t			sz, length;
		struct timeval	tv1, tv2;

		gettimeofday(&tv1, NULL);
		assert(kds_old->length == kds_length_last);
		length = kds_old->length + Max(kds_old->length, 1UL<<30);
		rc = cuMemAllocManaged(&m_devptr, length,
							   CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
			gpuClientELog(gclient, "failed on cuMemAllocManaged(%lu): %s",
						  length, cuStrError(rc));
			return false;
		}
		kds_new = (kern_data_store *)m_devptr;

		/* early half */
		sz = (KDS_HEAD_LENGTH(kds_old) +
			  sizeof(uint64_t) * (kds_old->nitems +
								  kds_old->hash_nslots));
		rc = cuMemcpyDtoD((CUdeviceptr)kds_new,
						  (CUdeviceptr)kds_old, sz);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
			cuMemFree(m_devptr);
			gpuClientELog(gclient, "failed on cuMemcpyDtoD: %s", cuStrError(rc));
			return false;
		}
		kds_new->length = length;

		/* later falf */
		sz = kds_old->usage;
		rc = cuMemcpyDtoD((CUdeviceptr)kds_new + kds_new->length - sz,
						  (CUdeviceptr)kds_old + kds_old->length - sz, sz);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
			cuMemFree(m_devptr);
			gpuClientELog(gclient, "failed on cuMemcpyDtoD: %s", cuStrError(rc));
			return false;
		}
		/* swap them */
		gettimeofday(&tv2, NULL);
		for (int i=0; i < gq_buf->gpumem_nitems; i++)
		{
			if (gq_buf->gpumem_devptrs[i] == (CUdeviceptr)kds_old)
			{
				gq_buf->gpumem_devptrs[i] = (CUdeviceptr)kds_new;
				goto found;
			}
		}
		__FATAL("Bug? kds_final for GPU%d is missing", MY_DINDEX_PER_THREAD);
	found:
		__gsDebug("kds_final expand: %lu => %lu [%.3fs]",
				  kds_old->length, kds_new->length,
				  (double)((tv2.tv_sec  - tv1.tv_sec)   * 1000000 +
						   (tv2.tv_usec - tv1.tv_usec)) / 1000000.0);
		cuMemFree((CUdeviceptr)kds_old);
		gq_buf->gpus[MY_DINDEX_PER_THREAD].m_kds_final = m_devptr;
		gq_buf->gpus[MY_DINDEX_PER_THREAD].m_kds_final_length = length;
	}
	pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);

	return true;
}

/*
 * __allocateGpuFallbackBuffer
 */
static bool
__allocateGpuFallbackBuffer(gpuQueryBuffer *gq_buf,
							kern_session_info *session)
{
	pthreadRWLockWriteLock(&gq_buf->m_kds_rwlock);
	if (GQBUF_KDS_FALLBACK(gq_buf) == NULL)
	{
		/* allocation of a new one */
		kern_data_store *kds_head;
		CUdeviceptr	m_devptr;
		CUresult	rc;

		assert(session->fallback_kds_head != 0);
		kds_head = (kern_data_store *)
			((char *)session + session->fallback_kds_head);
		rc = allocGpuQueryBuffer(gq_buf, &m_devptr, kds_head->length);
		if (rc != CUDA_SUCCESS)
		{
			pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
			return false;
		}
		memcpy((void *)m_devptr, kds_head, KDS_HEAD_LENGTH(kds_head));
		gq_buf->gpus[MY_DINDEX_PER_THREAD].m_kds_fallback = m_devptr;
		gq_buf->gpus[MY_DINDEX_PER_THREAD].m_kds_fallback_revision = 1;
		__gsDebug("kds_fallback allocated: %lu", kds_head->length);
	}
	pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
	return true;
}

static gpuQueryBuffer *
__getGpuQueryBuffer(uint64_t buffer_id, bool may_create)
{
	gpuQueryBuffer *gq_buf;
	dlist_iter		iter;
	int				hindex;

	/* lookup the hash-table */
	hindex = hash_bytes((unsigned char *)&buffer_id,
						sizeof(buffer_id)) % GPU_QUERY_BUFFER_NSLOTS;
	pthreadMutexLock(&gpu_query_buffer_mutex);
	dlist_foreach(iter, &gpu_query_buffer_hslot[hindex])
	{
		gq_buf = dlist_container(gpuQueryBuffer, chain, iter.cur);
		if (gq_buf->buffer_id == buffer_id)
		{
			gq_buf->refcnt++;

			/* wait for initial setup by other thread */
			while (gq_buf->phase == 0)
			{
				pthreadCondWait(&gpu_query_buffer_cond,
								&gpu_query_buffer_mutex);
			}
			/* check build status */
			if (gq_buf->phase < 0)
			{
				__putGpuQueryBufferNoLock(gq_buf);
				gq_buf = NULL;		/* failed on build */
			}
			goto found;
		}
	}
	/* not found, so create a new one */
	if (may_create)
	{
		gq_buf = calloc(1, offsetof(gpuQueryBuffer,
									gpus[numGpuDevAttrs]));
		if (!gq_buf)
			goto found;		/* out of memory */
		gq_buf->refcnt = 1;
		gq_buf->phase  = 0;	/* not initialized yet */
		gq_buf->buffer_id = buffer_id;
		pthreadRWLockInit(&gq_buf->m_kds_rwlock);
		dlist_push_tail(&gpu_query_buffer_hslot[hindex], &gq_buf->chain);
	}
	else
	{
		gq_buf = NULL;
	}
found:
	pthreadMutexUnlock(&gpu_query_buffer_mutex);
	return gq_buf;
}

static gpuQueryBuffer *
getGpuQueryBuffer(gpuClient *gclient,
				  uint64_t buffer_id,
				  uint32_t kmrels_handle,
				  kern_data_store *kds_final_head)
{
	gpuQueryBuffer *gq_buf = __getGpuQueryBuffer(buffer_id, true);

	if (gq_buf && gq_buf->phase == 0)
	{
		/* needs buffer initialization */
		if (__setupGpuQueryJoinInnerBuffer(gclient, gq_buf, kmrels_handle) &&
			__setupGpuQueryGroupByBuffer(gclient, gq_buf, kds_final_head))
		{
			/* ok, buffer is now ready */
			pthreadMutexLock(&gpu_query_buffer_mutex);
			gq_buf->phase = 1;		/* buffer is now ready */
			pthreadCondBroadcast(&gpu_query_buffer_cond);
			pthreadMutexUnlock(&gpu_query_buffer_mutex);
		}
		else
		{
			/* unable to setup the buffer */
			pthreadMutexLock(&gpu_query_buffer_mutex);
			gq_buf->phase = -1;			/* buffer unavailable */
			__putGpuQueryBufferNoLock(gq_buf);
			pthreadCondBroadcast(&gpu_query_buffer_cond);
			pthreadMutexUnlock(&gpu_query_buffer_mutex);

			gq_buf = NULL;
		}
	}
	return gq_buf;
}

/*
 * gpuServiceGoingTerminate
 */
bool
gpuServiceGoingTerminate(void)
{
	return (gpuserv_bgworker_got_signal != 0);
}

/*
 * gpuClientPut
 */
static void
gpuClientPut(gpuClient *gclient, bool exit_monitor_thread)
{
	int		cnt = (exit_monitor_thread ? 1 : 2);
	int		val;

	if ((val = pg_atomic_sub_fetch_u32(&gclient->refcnt, cnt)) == 0)
	{
		pthreadMutexLock(&gpuserv_client_lock);
		dlist_delete(&gclient->chain);
		pthreadMutexUnlock(&gpuserv_client_lock);

		if (gclient->sockfd >= 0)
			close(gclient->sockfd);
		if (gclient->gq_buf)
			putGpuQueryBuffer(gclient->gq_buf);
		for (int k=0; k < numGpuDevAttrs; k++)
		{
			gpuMemChunk *chunk = gclient->__session[k];

			if (chunk)
				gpuMemFree(chunk);
		}
		if (gclient->h_session)
			free(gclient->h_session);
		free(gclient);
	}
}

/*
 * gpuClientWriteBack
 */
static void
__gpuClientWriteBack(gpuClient *gclient, struct iovec *iov, int iovcnt)
{
	pthreadMutexLock(&gclient->mutex);
	if (gclient->sockfd >= 0)
	{
		ssize_t		nbytes;

		while (iovcnt > 0)
		{
			nbytes = writev(gclient->sockfd, iov, iovcnt);
			if (nbytes > 0)
			{
				do {
					if (iov->iov_len <= nbytes)
					{
						nbytes -= iov->iov_len;
						iov++;
						iovcnt--;
					}
					else
					{
						iov->iov_base = (char *)iov->iov_base + nbytes;
						iov->iov_len -= nbytes;
						break;
					}
				} while (iovcnt > 0 && nbytes > 0);
			}
			else if (errno != EINTR)
			{
				/*
				 * Peer socket is closed? Anyway, it looks we cannot continue
				 * to send back the message any more. So, clean up this gpuClient.
				 */
				pg_atomic_fetch_and_u32(&gclient->refcnt, ~1U);
				close(gclient->sockfd);
				gclient->sockfd = -1;
				break;
			}
		}
	}
	pthreadMutexUnlock(&gclient->mutex);
}

static int
__gpuClientWriteBackOneChunk(gpuClient *gclient,
							 struct iovec *iov_array,
							 kern_data_store *kds)
{
	struct iovec   *iov;
	int				iovcnt = 0;
	size_t			sz, head_sz;

	if (kds->format == KDS_FORMAT_HASH)
	{
		assert(kds->hash_nslots > 0);
		sz = KDS_HEAD_LENGTH(kds);
		iov = &iov_array[iovcnt++];
		iov->iov_base = kds;
		iov->iov_len  = sz;

		sz = sizeof(uint64_t) * kds->nitems;
		if (sz > 0)
		{
			iov = &iov_array[iovcnt++];
			iov->iov_base = KDS_GET_ROWINDEX(kds);
			iov->iov_len  = sz;
		}
		/*
		 * MEMO: When GPU Projection or similar fill up KDS-ROW/HASH,
		 * it often increase kds->usage too much if no space left.
		 * In this case, results are not written of course, however,
		 * kds->usage is not reliable. So, we cut down the usege by
		 * the tail of row-offset array.
		 */
		head_sz = (KDS_HEAD_LENGTH(kds) +
				   sizeof(uint64_t) * (kds->hash_nslots +
									   kds->nitems));
		sz = Min(kds->usage, kds->length - head_sz);
		if (sz > 0)
		{
			 iov = &iov_array[iovcnt++];
			 iov->iov_base = (char *)kds + kds->length - sz;
			 iov->iov_len  = sz;
		}
		/* fixup kds */
		kds->format = KDS_FORMAT_ROW;
		kds->hash_nslots = 0;
		kds->length = (KDS_HEAD_LENGTH(kds) +
					   sizeof(uint64_t) * kds->nitems +
					   sz);
	}
	else
	{
		assert(kds->format == KDS_FORMAT_ROW ||
			   kds->format == KDS_FORMAT_FALLBACK);
		assert(kds->hash_nslots == 0);
		head_sz = (KDS_HEAD_LENGTH(kds) +
				   MAXALIGN(sizeof(uint64_t) * kds->nitems));
		/* see the comment above */
		if (head_sz + kds->usage >= kds->length)
		{
			iov = &iov_array[iovcnt++];
			iov->iov_base = kds;
			iov->iov_len  = kds->length;
		}
		else
		{
			iov = &iov_array[iovcnt++];
			iov->iov_base = kds;
			iov->iov_len  = head_sz;

			if (kds->usage > 0)
			{
				iov = &iov_array[iovcnt++];
				iov->iov_base = (char *)kds + kds->length - kds->usage;
				iov->iov_len  = kds->usage;
			}
			/* fixup kds */
			kds->length = head_sz + kds->usage;
		}
	}
	return iovcnt;
}

static void
gpuClientWriteBackPartial(gpuClient  *gclient,
						  kern_data_store *kds)
{
	struct iovec	iov_array[10];
	struct iovec   *iov;
	int				iovcnt = 0;
	XpuCommand		resp;
	size_t			resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats));
	size_t			__kds_length = kds->length;
	char			__kds_format = kds->format;
	uint32_t		__kds_hash_nslots = kds->hash_nslots;

	memset(&resp, 0, sizeof(resp));
	resp.magic  = XpuCommandMagicNumber;
	resp.tag    = XpuCommandTag__SuccessHalfWay;
	resp.u.results.chunks_nitems = 1;
	resp.u.results.chunks_offset = resp_sz;

	iov = &iov_array[iovcnt++];
	iov->iov_base = &resp;
	iov->iov_len = resp_sz;
	iovcnt += __gpuClientWriteBackOneChunk(gclient, iov_array+iovcnt, kds);
	for (int i=1; i < iovcnt; i++)
		resp_sz += iov_array[i].iov_len;
	resp.length = resp_sz;

	__gpuClientWriteBack(gclient, iov_array, iovcnt);
	/* reset chunk */
	kds->length = __kds_length;
	kds->format = __kds_format;
	kds->hash_nslots = __kds_hash_nslots;
	kds->nitems = 0;
	kds->usage  = 0;
	if (kds->format == KDS_FORMAT_HASH)
		memset(KDS_GET_HASHSLOT_BASE(kds), 0, sizeof(uint64_t) * kds->hash_nslots);
}

static void
gpuClientWriteBackNormal(gpuClient  *gclient,
						 kern_exec_results *kern_stats,
						 kern_data_store *kds)
{
	struct iovec	iov_array[10];
	struct iovec   *iov;
	int				iovcnt = 0;
	XpuCommand	   *resp;
	size_t			resp_sz;

	resp_sz = MAXALIGN(offsetof(XpuCommand,
								u.results.stats[kern_stats->num_rels]));
	resp = alloca(resp_sz);
	memset(resp, 0, resp_sz);
	resp->magic  = XpuCommandMagicNumber;
	resp->tag    = XpuCommandTag__Success;
	memcpy(&resp->u.results, kern_stats,
		   offsetof(kern_exec_results, stats[kern_stats->num_rels]));
	/* sanity checks */
	assert(resp->u.results.chunks_offset == 0 &&
		   resp->u.results.chunks_nitems == 0 &&
		   resp->u.results.ojmap_offset == 0 &&
		   resp->u.results.ojmap_length == 0 &&
		   resp->u.results.final_plan_task == 0 &&
		   resp->u.results.final_nitems == 0 &&
		   resp->u.results.final_usage  == 0 &&
		   resp->u.results.final_total  == 0);
	if (kds)
	{
		resp->u.results.chunks_nitems = 1;
		resp->u.results.chunks_offset = resp_sz;
	}
	iov = &iov_array[iovcnt++];
	iov->iov_base = resp;
	iov->iov_len  = resp_sz;
	if (kds)
		iovcnt += __gpuClientWriteBackOneChunk(gclient, iov_array+iovcnt, kds);
	for (int i=1; i < iovcnt; i++)
		resp_sz += iov_array[i].iov_len;
	resp->length = resp_sz;

	__gpuClientWriteBack(gclient, iov_array, iovcnt);
}


/* ----------------------------------------------------------------
 *
 * gpuservMonitorClient
 *
 * ----------------------------------------------------------------
 */
static bool	gpuservHandleOpenSession(gpuClient *gclient,
									 XpuCommand *xcmd);

typedef struct
{
	gpuContext	   *gcontext;
	gpuMemChunk	   *chunk;
	XpuCommand		xcmd;
} gpuServXpuCommandPacked;

static void *
__gpuServiceAllocCommand(void *__priv, size_t sz)
{
	gpuClient	   *gclient = (gpuClient *)__priv;
	gpuMemChunk	   *chunk;
	gpuServXpuCommandPacked *packed;

	sz += offsetof(gpuServXpuCommandPacked, xcmd);
	if (gclient->optimal_gpus == 0)
	{
		packed = calloc(1, sz);
		if (!packed)
		{
			gpuClientELog(gclient, "out of memory");
			return NULL;
		}
		packed->gcontext = NULL;	/* be released by free(3) */
		packed->chunk = NULL;		/* be released by free(3) */
	}
	else
	{
		/*
		 * Once the target GPUs are determined by OpenSession command,
		 * we choose one of the candidate GPU here.
		 */
		gpuContext *gcontext = NULL;
		uint32_t	gcontext_count = 0;
		dlist_iter	iter;

		dlist_foreach(iter, &gpuserv_gpucontext_list)
		{
			gpuContext *__gcontext = dlist_container(gpuContext, chain, iter.cur);
			int64_t		__mask = (1UL << __gcontext->cuda_dindex);
			uint32_t	__count;

			if ((gclient->optimal_gpus & __mask) == 0)
				continue;
			__count = pg_atomic_read_u32(&__gcontext->num_commands);
			if (!gcontext || __count < gcontext_count)
			{
				gcontext        = __gcontext;
				gcontext_count = __count;
			}
		}
		if (!gcontext)
		{
			gpuClientELog(gclient, "No GPUs are available (optimal_gpus=%08lx)",
						  gclient->optimal_gpus);
			return NULL;
		}
		chunk = __gpuMemAllocCommon(&gcontext->pool_managed, sz);
		if (!chunk)
		{
			gpuClientELog(gclient, "out of managed memory");
			return NULL;
		}
		packed = (gpuServXpuCommandPacked *)chunk->m_devptr;
		packed->gcontext = gcontext;
		packed->chunk = chunk;
	}
	return &packed->xcmd;
}

static void
__gpuServiceFreeCommand(XpuCommand *xcmd)
{
	gpuServXpuCommandPacked *packed = (gpuServXpuCommandPacked *)
		((char *)xcmd - offsetof(gpuServXpuCommandPacked, xcmd));
	if (!packed->chunk)
		free(packed);		/* Xcmd of OpenSession */
	else
		gpuMemFree(packed->chunk);	/* other commands */
}

static void
__gpuServiceAttachCommand(void *__priv, XpuCommand *xcmd)
{
	gpuServXpuCommandPacked *packed = (gpuServXpuCommandPacked *)
		((char *)xcmd - offsetof(gpuServXpuCommandPacked, xcmd));
	gpuClient  *gclient = (gpuClient *)__priv;

	pg_atomic_fetch_add_u32(&gclient->refcnt, 2);
	xcmd->priv = gclient;
	if (!packed->gcontext)
	{
		Assert(gclient->optimal_gpus == 0);
		if (xcmd->tag == XpuCommandTag__OpenSession)
		{
			gpuservHandleOpenSession(gclient, xcmd);
		}
		else
		{
			gpuClientELog(gclient, "XPU command '%d' before OpenSession", xcmd->tag);
		}
		__gpuServiceFreeCommand(xcmd);
		gpuClientPut(gclient, false);
	}
	else
	{
		gpuContext *gcontext = packed->gcontext;

		pthreadMutexLock(&gcontext->lock);
		dlist_push_tail(&gcontext->command_list, &xcmd->chain);
		pg_atomic_fetch_add_u32(&gcontext->num_commands, 1);
		pthreadMutexUnlock(&gcontext->lock);
		pthreadCondSignal(&gcontext->cond);
	}
}
TEMPLATE_XPU_CONNECT_RECEIVE_COMMANDS(__gpuService)

/*
 * expandCudaStackLimit
 */
static bool
expandCudaStackLimit(gpuClient *gclient,
					 gpuContext *gcontext,
					 kern_session_info *session)
{
	uint32_t	cuda_stack_size = session->cuda_stack_size;
	uint32_t	cuda_stack_limit;
	CUresult	rc;

	if (cuda_stack_size > gcontext->cuda_stack_limit)
	{
		cuda_stack_limit = TYPEALIGN(1024, cuda_stack_size);
		if (cuda_stack_limit > (__pgstrom_cuda_stack_limit_kb << 10))
		{
			gpuClientELog(gclient, "CUDA stack size %u is larger than the configured limitation %ukB by pg_strom.cuda_stack_limit",
						  cuda_stack_size, __pgstrom_cuda_stack_limit_kb);
			return false;
		}

		pthreadMutexLock(&gcontext->cuda_setlimit_lock);
		if (cuda_stack_size > gcontext->cuda_stack_limit)
		{
			gpuContext *gcontext_prev = gpuContextSwitchTo(gcontext);

			rc = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, cuda_stack_limit);
			if (rc != CUDA_SUCCESS)
			{
				pthreadMutexUnlock(&gcontext->cuda_setlimit_lock);
				gpuClientELog(gclient, "failed on cuCtxSetLimit: %s",
							  cuStrError(rc));
				return false;
			}
			gpuContextSwitchTo(gcontext_prev);

			__gsLogCxt(gcontext, "CUDA stack size expanded %u -> %u bytes",
					   gcontext->cuda_stack_limit,
					   cuda_stack_limit);
			gcontext->cuda_stack_limit = cuda_stack_limit;
		}
		pthreadMutexUnlock(&gcontext->cuda_setlimit_lock);
	}
	return true;
}

/*
 * gpuservHandleOpenSession
 */
static bool
__lookupDeviceTypeOper(gpuClient *gclient,
					   gpuContext *gcontext,
					   const xpu_datum_operators **p_expr_ops,
					   TypeOpCode type_code)
{
	xpu_type_catalog_entry *xpu_type;

	xpu_type = hash_search(gcontext->cuda_type_htab,
						   &type_code,
						   HASH_FIND, NULL);
	if (!xpu_type)
	{
		gpuClientELog(gclient, "device type pointer for opcode:%u not found.",
					  (int)type_code);
		return false;
	}
	*p_expr_ops = xpu_type->type_ops;
	return true;
}

static bool
__lookupDeviceFuncDptr(gpuClient *gclient,
					   gpuContext *gcontext,
					   xpu_function_t *p_func_dptr,
					   FuncOpCode func_code)
{
	xpu_function_catalog_entry *xpu_func;

	xpu_func = hash_search(gcontext->cuda_func_htab,
						   &func_code,
						   HASH_FIND, NULL);
	if (!xpu_func)
	{
		gpuClientELog(gclient, "device function pointer for opcode:%u not found.",
					  (int)func_code);
		return false;
	}
	*p_func_dptr = xpu_func->func_dptr;
	return true;
}

static bool
__resolveDevicePointersWalker(gpuClient *gclient,
							  gpuContext *gcontext,
							  kern_expression *kexp)
{
	kern_expression *karg;
	int		i;

	if (!__lookupDeviceFuncDptr(gclient,
								gcontext,
								&kexp->fn_dptr,
								kexp->opcode))
		return false;

	if (!__lookupDeviceTypeOper(gclient,
								gcontext,
								&kexp->expr_ops,
								kexp->exptype))
		return false;

	/* some special cases */
	switch (kexp->opcode)
	{
		case FuncOpCode__CaseWhenExpr:
			if (kexp->u.casewhen.case_comp)
			{
				karg = (kern_expression *)
					((char *)kexp + kexp->u.casewhen.case_comp);
				if (!__KEXP_IS_VALID(kexp,karg))
					goto corruption;
				if (!__resolveDevicePointersWalker(gclient,
												   gcontext, karg))
					return false;
			}
			if (kexp->u.casewhen.case_else)
			{
				karg = (kern_expression *)
					((char *)kexp + kexp->u.casewhen.case_else);
				if (!__KEXP_IS_VALID(kexp,karg))
					goto corruption;
				if (!__resolveDevicePointersWalker(gclient,
												   gcontext, karg))
					return false;
			}
			break;
		case FuncOpCode__Projection:
			if (kexp->u.proj.hash)
			{
				karg = (kern_expression *)
					((char *)kexp + kexp->u.proj.hash);
				if (!__KEXP_IS_VALID(kexp,karg))
					goto corruption;
				if (!__resolveDevicePointersWalker(gclient,
												   gcontext, karg))
					return false;
			}
			break;

		default:
			break;
	}
	/* walk on the arguments */
	for (i=0, karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		if (!__KEXP_IS_VALID(kexp,karg))
			goto corruption;
		if (!__resolveDevicePointersWalker(gclient,
										   gcontext, karg))
			return false;
	}
	return true;

corruption:
	gpuClientELog(gclient, "XPU code corruption at kexp (%d)", kexp->opcode);
	return false;
}

static bool
__resolveDevicePointers(gpuClient *gclient,
						gpuContext *gcontext,
						kern_session_info *session)
{
	kern_varslot_desc *kvslot_desc = SESSION_KVARS_SLOT_DESC(session);
	xpu_encode_info *encode = SESSION_ENCODE(session);
	kern_expression *__kexp[20];
	int		nitems = 0;

	__kexp[nitems++] = SESSION_KEXP_LOAD_VARS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_MOVE_VARS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_SCAN_QUALS(session);
	__kexp[nitems++] = SESSION_KEXP_JOIN_QUALS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_HASH_VALUE(session, -1);
	__kexp[nitems++] = SESSION_KEXP_GIST_EVALS(session, -1);
	__kexp[nitems++] = SESSION_KEXP_PROJECTION(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYHASH(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYLOAD(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_KEYCOMP(session);
	__kexp[nitems++] = SESSION_KEXP_GROUPBY_ACTIONS(session);

	for (int i=0; i < nitems; i++)
	{
		if (__kexp[i] && !__resolveDevicePointersWalker(gclient,
														gcontext,
														__kexp[i]))
			return false;
	}

	/* fixup kern_varslot_desc also */
	for (int i=0; i < session->kcxt_kvars_nslots; i++)
	{
		if (!__lookupDeviceTypeOper(gclient,
									gcontext,
									&kvslot_desc[i].vs_ops,
									kvslot_desc[i].vs_type_code))
			return false;
	}

	/* encoding catalog */
	if (encode)
	{
		xpu_encode_info *catalog = gcontext->cuda_encode_catalog;

		for (int i=0; ; i++)
		{
			if (!catalog[i].enc_mblen || catalog[i].enc_maxlen < 1)
			{
				gpuClientELog(gclient, "encode [%s] was not found.",
							  encode->encname);
				return false;
			}
			if (strcmp(encode->encname, catalog[i].encname) == 0)
			{
				encode->enc_maxlen = catalog[i].enc_maxlen;
				encode->enc_mblen  = catalog[i].enc_mblen;
				break;
			}
		}
	}
	return true;
}

static bool
gpuservHandleOpenSession(gpuClient *gclient, XpuCommand *xcmd)
{
	kern_session_info *session = &xcmd->u.session;
	size_t			session_sz = xcmd->length - offsetof(XpuCommand, u.session);
	gpuContext	   *gcontext_home;
	gpuContext	   *gcontext_prev;
	dlist_iter		iter;
	kern_data_store *kds_final_head = NULL;
	XpuCommand		resp;
	struct iovec	iov;

	if (gclient->optimal_gpus != 0)
	{
		gpuClientELog(gclient, "OpenSession is called twice");
		return false;
	}
	gcontext_home = __lookupOneRandomGpuContext(session->optimal_gpus);
	if (!gcontext_home)
	{
		gpuClientELog(gclient, "GPU-client must have at least one schedulable GPUs: %08lx", session->optimal_gpus);
		return false;
	}
	gcontext_prev = gpuContextSwitchTo(gcontext_home);

	gclient->optimal_gpus   = session->optimal_gpus;
	gclient->xpu_task_flags = session->xpu_task_flags;
	gclient->h_session = malloc(session_sz);
	if (!gclient->h_session)
	{
		gpuClientELog(gclient, "out of memory");
		goto error;
	}
	memcpy(gclient->h_session, session, session_sz);

	dlist_foreach (iter, &gpuserv_gpucontext_list)
	{
		gpuContext *__gcontext = dlist_container(gpuContext, chain, iter.cur);
		uint32_t	__cuda_dindex = __gcontext->cuda_dindex;
		gpuMemChunk *__chunk;
		kern_session_info *__session;

		if ((gclient->optimal_gpus & __gcontext->cuda_dmask) == 0)
			continue;
		__chunk = __gpuMemAllocCommon(&__gcontext->pool_managed, session_sz);
		if (!__chunk)
		{
			gpuClientELog(gclient, "out of managed memory");
			goto error;
		}
		gclient->__session[__cuda_dindex] = __chunk;
		__session = (kern_session_info *)__chunk->m_devptr;
		memcpy(__session, session, session_sz);
		/* expand CUDA thread stack limit on demand */
		if (!expandCudaStackLimit(gclient,
								  __gcontext,
								  __session))
			goto error;
		/* resolve device pointers */
		if (!__resolveDevicePointers(gclient,
									 __gcontext,
									 __session))
			goto error;
	}
	/* setup per-cliend GPU buffer */
	if (session->groupby_kds_final != 0)
	{
		kds_final_head = (kern_data_store *)
			((char *)session + session->groupby_kds_final);
	}
	gclient->gq_buf = getGpuQueryBuffer(gclient,
										session->query_plan_id,
										session->join_inner_handle,
										kds_final_head);
	if (!gclient->gq_buf)
		goto error;

	/* success status */
	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Success;
	resp.length = offsetof(XpuCommand, u.results.stats);

	iov.iov_base = &resp;
	iov.iov_len  = resp.length;
	__gpuClientWriteBack(gclient, &iov, 1);

	return true;

error:
	gpuContextSwitchTo(gcontext_prev);
	return false;
}

/* ----------------------------------------------------------------
 *
 * gpuservLoadKdsXXXX - Load data chunks using GPU-Direct SQL
 *
 * ----------------------------------------------------------------
 */
static gpuMemChunk *
__gpuservLoadKdsCommon(gpuClient *gclient,
					   kern_data_store *kds,
					   size_t base_offset,
					   const char *pathname,
					   strom_io_vector *kds_iovec,
					   uint32_t *p_npages_direct_read,
					   uint32_t *p_npages_vfs_read)
{
	gpuMemChunk *chunk;
	CUresult	rc;
	off_t		off = PAGE_ALIGN(base_offset);
	size_t		gap = off - base_offset;

	chunk = gpuMemAlloc(gap + kds->length);
	if (!chunk)
	{
		chunk = gpuMemAllocManaged(gap + kds->length);
		if (!chunk)
		{
			gpuClientELog(gclient, "failed on gpuMemAlloc(%zu+%zu)",
						  gap, kds->length);
			return NULL;
		}
	}
	chunk->m_devptr = chunk->__base + chunk->__offset + gap;

	rc = cuMemcpyHtoD(chunk->m_devptr, kds, base_offset);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on cuMemcpyHtoD: %s", cuStrError(rc));
		goto error;
	}
	if (!gpuDirectFileReadIOV(pathname,
							  chunk->__base,
							  chunk->__offset + off,
							  chunk->mseg->iomap_handle,
							  kds_iovec,
							  p_npages_direct_read,
							  p_npages_vfs_read))
	{
		gpuClientELogByExtraModule(gclient);
		goto error;
	}
	return chunk;

error:
	gpuMemFree(chunk);
	return NULL;
}

/*
 * gpuservLoadKdsBlock
 *
 * fill up KDS_FORMAT_BLOCK using GPU-Direct
 */
static gpuMemChunk *
gpuservLoadKdsBlock(gpuClient *gclient,
					kern_data_store *kds,
					const char *pathname,
					strom_io_vector *kds_iovec,
					uint32_t *p_npages_direct_read,
					uint32_t *p_npages_vfs_read)
{
	size_t		base_offset;

	Assert(kds->format == KDS_FORMAT_BLOCK);
	base_offset = kds->block_offset + kds->block_nloaded * BLCKSZ;
	return __gpuservLoadKdsCommon(gclient,
								  kds,
								  base_offset,
								  pathname,
								  kds_iovec,
								  p_npages_direct_read,
								  p_npages_vfs_read);
}

/*
 * gpuservLoadKdsArrow
 *
 * fill up KDS_FORMAT_ARROW using GPU-Direct
 */
static gpuMemChunk *
gpuservLoadKdsArrow(gpuClient *gclient,
					kern_data_store *kds,
					const char *pathname,
					strom_io_vector *kds_iovec,
					uint32_t *p_npages_direct_read,
					uint32_t *p_npages_vfs_read)
{
	size_t		base_offset;

	Assert(kds->format == KDS_FORMAT_ARROW);
	base_offset = KDS_HEAD_LENGTH(kds) + kds->arrow_virtual_usage;
	return __gpuservLoadKdsCommon(gclient,
								  kds,
								  base_offset,
								  pathname,
								  kds_iovec,
								  p_npages_direct_read,
								  p_npages_vfs_read);
}

/* ----------------------------------------------------------------
 *
 * gpuservHandleGpuTaskExec
 *
 * ----------------------------------------------------------------
 */
static unsigned int
__expand_gpupreagg_prepfunc_buffer(kern_session_info *session,
								   int grid_sz, int block_sz,
								   unsigned int __shmem_dynamic_sz,
								   unsigned int shmem_dynamic_limit,
								   unsigned int *p_groupby_prepfn_bufsz,
								   unsigned int *p_groupby_prepfn_nbufs)
{
	unsigned int	shmem_dynamic_sz = TYPEALIGN(1024, __shmem_dynamic_sz);

	if (session->groupby_prepfn_bufsz == 0 || shmem_dynamic_sz >= shmem_dynamic_limit)
		goto no_prepfunc_buffer;

	if (session->xpucode_groupby_keyhash &&
		session->xpucode_groupby_keyload &&
		session->xpucode_groupby_keycomp)
	{
		/* GROUP-BY */
		int		num_buffers = 2 * session->groupby_ngroups_estimation + 100;
		int		prepfn_usage = session->groupby_prepfn_bufsz * num_buffers;

		if (shmem_dynamic_sz + prepfn_usage > shmem_dynamic_limit)
		{
			/* adjust num_buffers, if too large */
			num_buffers = (shmem_dynamic_limit -
						   shmem_dynamic_sz) / session->groupby_prepfn_bufsz;
			prepfn_usage = session->groupby_prepfn_bufsz * num_buffers;
		}
		Assert(shmem_dynamic_sz + prepfn_usage <= shmem_dynamic_limit);
		if (num_buffers >= 32)
		{
			*p_groupby_prepfn_bufsz = session->groupby_prepfn_bufsz;
			*p_groupby_prepfn_nbufs = num_buffers;
			return shmem_dynamic_sz + prepfn_usage;
		}
	}
	else if (session->xpucode_groupby_actions)
	{
		/* NO-GROUPS */
		if (shmem_dynamic_sz + session->groupby_prepfn_bufsz <= shmem_dynamic_limit)
		{
			*p_groupby_prepfn_bufsz = session->groupby_prepfn_bufsz;
			*p_groupby_prepfn_nbufs = 1;
			return shmem_dynamic_sz + session->groupby_prepfn_bufsz;
		}
	}
no_prepfunc_buffer:
	*p_groupby_prepfn_bufsz = 0;
	*p_groupby_prepfn_nbufs = 0;
	return __shmem_dynamic_sz;		/* unaligned original size */
}

/*
 * __gpuservLaunchGpuTaskExecKernel
 */
static bool
__gpuservLaunchGpuTaskExecKernel(gpuContext *gcontext,
								 gpuClient *gclient,
								 CUdeviceptr m_kds_src,
								 CUdeviceptr m_kds_extra,
								 CUdeviceptr m_kmrels,
								 kern_data_store *kds_dst,
								 int part_divisor,
								 int part_reminder,
								 int right_outer_depth,
								 kern_exec_results *kern_stats)
{
	kern_session_info *session = gclient->h_session;
	gpuQueryBuffer  *gq_buf = gclient->gq_buf;
	kern_gputask	*kgtask = NULL;
	kern_data_store *kds_fallback = NULL;
	void		   *gc_lmap = NULL;
	gpuMemChunk	   *t_chunk = NULL;		/* for kern_gputask */
	CUdeviceptr		m_session;
	CUresult		rc;
	int				num_inner_rels = kern_stats->num_rels;
	int				grid_sz;
	int				block_sz;
	unsigned int	shmem_dynamic_sz;
	unsigned int	groupby_prepfn_bufsz = 0;
	unsigned int	groupby_prepfn_nbufs = 0;
	size_t			kds_final_length = 0;
	uint32_t		kds_fallback_revision = 0;
	uint32_t		last_kernel_errcode = ERRCODE_STROM_SUCCESS;
	size_t			sz;
	void		   *kern_args[10];
	bool			status = false;

	/* sanity checks */
	Assert(gcontext == GpuWorkerCurrentContext);

	/*
	 * calculation of dynamic shared memory size, and grid/block size.
	 */
	shmem_dynamic_sz = __KERN_WARP_CONTEXT_BASESZ(session->kcxt_kvecs_ndims);
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 gcontext->cufn_kern_gpumain,
							 shmem_dynamic_sz);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on gpuOptimalBlockSize: %s",
					   cuStrError(rc));
		goto bailout;
	}
	//TODO: adjust grid_sz for better concurrency
//	block_sz = 128;
//	grid_sz = 1;

	/* allocation of extra shared memory for GpuPreAgg (if any) */
	shmem_dynamic_sz =
		__expand_gpupreagg_prepfunc_buffer(session,
										   grid_sz, block_sz,
										   shmem_dynamic_sz,
										   gcontext->gpumain_shmem_sz_limit,
										   &groupby_prepfn_bufsz,
										   &groupby_prepfn_nbufs);
	/*
	 * Allocation of the control structure
	 */
	sz = KERN_GPUTASK_LENGTH(session->kcxt_kvecs_ndims,
							 session->kcxt_kvecs_bufsz,
							 grid_sz, block_sz);
	t_chunk = gpuMemAllocManaged(sz);
	if (!t_chunk)
	{
		gpuClientFatal(gclient, "failed on gpuMemAllocManaged: %lu", sz);
		goto bailout;
	}
	kgtask = (kern_gputask *)t_chunk->m_devptr;
	memset(kgtask, 0, offsetof(kern_gputask, stats[num_inner_rels]));
	kgtask->grid_sz      = grid_sz;
	kgtask->block_sz     = block_sz;
	kgtask->kvars_nslots = session->kcxt_kvars_nslots;
	kgtask->kvecs_bufsz  = session->kcxt_kvecs_bufsz;
	kgtask->kvecs_ndims  = session->kcxt_kvecs_ndims;
	kgtask->n_rels       = num_inner_rels;
	kgtask->groupby_prepfn_bufsz = groupby_prepfn_bufsz;
	kgtask->groupby_prepfn_nbufs = groupby_prepfn_nbufs;
	kgtask->cuda_dindex        = MY_DINDEX_PER_THREAD;
	kgtask->cuda_stack_limit   = GpuWorkerCurrentContext->cuda_stack_limit;
	kgtask->partition_divisor  = part_divisor;
	kgtask->partition_reminder = part_reminder;
	kgtask->right_outer_depth  = right_outer_depth;

	/*
	 * Allocation of the destination buffer
	 */
resume_kernel:
	if (last_kernel_errcode == ERRCODE_SUSPEND_FALLBACK)
	{
		if (kds_fallback)
		{
			pthreadRWLockWriteLock(&gq_buf->m_kds_rwlock);
			if (GQBUF_KDS_FALLBACK_REVISION(gq_buf) == kds_fallback_revision)
			{
				gpuClientWriteBackPartial(gclient, kds_fallback);
				GQBUF_KDS_FALLBACK_REVISION(gq_buf)++;
			}
			pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
		}
		else if (!__allocateGpuFallbackBuffer(gq_buf, session))
		{
			gpuClientFatal(gclient, "unable to allocate GPU-Fallback buffer");
			goto bailout;
		}
	}

	if (GQBUF_KDS_FINAL(gq_buf))
	{
		/* needs to expand the final buffer? */
		if (last_kernel_errcode == ERRCODE_SUSPEND_NO_SPACE)
		{
			if (!__expandGpuQueryGroupByBuffer(gclient, gq_buf, kds_final_length))
				goto bailout;
		}
		pthreadRWLockReadLock(&gq_buf->m_kds_rwlock);
		kds_dst = GQBUF_KDS_FINAL(gq_buf);
		kds_final_length = GQBUF_KDS_FINAL_LENGTH(gq_buf);
		kds_fallback = GQBUF_KDS_FALLBACK(gq_buf);
		kds_fallback_revision = GQBUF_KDS_FALLBACK_REVISION(gq_buf);
	}
	else
	{
		Assert(kds_dst != NULL);

		if (last_kernel_errcode == ERRCODE_SUSPEND_NO_SPACE)
		{
			gpuClientWriteBackPartial(gclient, kds_dst);
			assert(kds_dst->nitems == 0 && kds_dst->usage == 0);
		}
		else if (last_kernel_errcode != ERRCODE_STROM_SUCCESS)
		{
			/*
			 * When GPU kernel is suspended by CPU fallback,
			 * we don't need to touch the kds_dst.
			 */
			Assert(last_kernel_errcode == ERRCODE_SUSPEND_FALLBACK);
		}
		pthreadRWLockReadLock(&gq_buf->m_kds_rwlock);
		kds_fallback = GQBUF_KDS_FALLBACK(gq_buf);
		kds_fallback_revision = GQBUF_KDS_FALLBACK_REVISION(gq_buf);
	}

	/*
	 * Launch kernel
	 */
	m_session = gclient->__session[MY_DINDEX_PER_THREAD]->m_devptr;
	kern_args[0] = &m_session;
	kern_args[1] = &kgtask;
	kern_args[2] = &m_kmrels;
	kern_args[3] = &m_kds_src;
	kern_args[4] = &m_kds_extra;
	kern_args[5] = &kds_dst;
	kern_args[6] = &kds_fallback;

	rc = cuLaunchKernel(gcontext->cufn_kern_gpumain,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						shmem_dynamic_sz,
						MY_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
		gpuClientFatal(gclient, "failed on cuLaunchKernel: %s", cuStrError(rc));
		goto bailout;
	}

	rc = cuStreamSynchronize(MY_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);
		gpuClientFatal(gclient, "failed on cuStreamSynchronize: %s", cuStrError(rc));
		goto bailout;
	}
	/* unlock kds_final/kds_fallback buffer */
	pthreadRWLockUnlock(&gq_buf->m_kds_rwlock);

	/* status check */
	if (kgtask->kerror.errcode == ERRCODE_STROM_SUCCESS)
	{
		kern_stats->nitems_raw += kgtask->nitems_raw;
		kern_stats->nitems_in  += kgtask->nitems_in;
		kern_stats->nitems_out += kgtask->nitems_out;
		for (int j=0; j < kgtask->n_rels; j++)
		{
			kern_stats->stats[j].nitems_roj  += kgtask->stats[j].nitems_roj;
			kern_stats->stats[j].nitems_gist += kgtask->stats[j].nitems_gist;
			kern_stats->stats[j].nitems_out  += kgtask->stats[j].nitems_out;
		}
		/* success in kernel execution */
		status = true;
	}
	else if (kgtask->kerror.errcode == ERRCODE_SUSPEND_NO_SPACE ||
			 (kgtask->kerror.errcode == ERRCODE_SUSPEND_FALLBACK &&
			  session->fallback_kds_head > 0))
	{
		if (gpuServiceGoingTerminate())
		{
			gpuClientFatal(gclient, "GpuService is going to terminate during kernel suspend/resume");
			goto bailout;
		}
		__gsDebug("suspend / resume by %s at %s:%d (%s)",
				  kgtask->kerror.errcode == ERRCODE_SUSPEND_NO_SPACE
				  ? "Buffer no space"
				  : "CPU fallback",
				  kgtask->kerror.filename,
				  kgtask->kerror.lineno,
				  kgtask->kerror.message);
		last_kernel_errcode = kgtask->kerror.errcode;
		kgtask->kerror.errcode = 0;
		kgtask->resume_context = true;
		goto resume_kernel;
	}
	else
	{
		/* send back error status */
		__gpuClientELogRaw(gclient, &kgtask->kerror);
	}
bailout:
	if (t_chunk)
		gpuMemFree(t_chunk);
	if (gc_lmap)
		gpuCachePutDeviceBuffer(gc_lmap);
	return status;
}

static void
gpuservHandleGpuTaskExec(gpuContext *gcontext,
						 gpuClient *gclient,
						 XpuCommand *xcmd)
{
	gpuQueryBuffer  *gq_buf = gclient->gq_buf;
	const char		*kds_src_pathname = NULL;
	strom_io_vector *kds_src_iovec = NULL;
	kern_data_store	*kds_src = NULL;
	kern_data_store *kds_dst = NULL;
	CUdeviceptr		m_kds_src = 0UL;
	CUdeviceptr		m_kds_extra = 0UL;
	CUdeviceptr		m_kmrels = 0UL;
	CUresult		rc;
	uint32_t		num_inner_rels = 0;
	uint32_t		npages_direct_read = 0;
	uint32_t		npages_vfs_read = 0;
	kern_exec_results *kern_stats;			/* for statistics */
	void		   *gc_lmap = NULL;
	gpuMemChunk	   *t_chunk = NULL;			/* for kgtask */
	gpuMemChunk	   *s_chunk = NULL;			/* for kds_src */
	gpuMemChunk	   *d_chunk = NULL;			/* for kds_dst */
	int				part_nitems = 0;		/* for inner-buffer partitions */
	gpuContext	  **part_gcontexts = NULL;	/* for inner-buffer partitions */
	uint32_t	   *part_reminders = NULL;	/* for inner-buffer partitions */
	int				part_divisor = -1;		/* for inner-buffer partitions */
	int				curr_reminder = -1;		/* for inner-buffer partitions */
	size_t			sz;

	THREAD_GPU_CONTEXT_VALIDATION_CHECK(gcontext);
	if (xcmd->u.task.kds_src_pathname)
		kds_src_pathname = (char *)xcmd + xcmd->u.task.kds_src_pathname;
	if (xcmd->u.task.kds_src_iovec)
		kds_src_iovec = (strom_io_vector *)((char *)xcmd + xcmd->u.task.kds_src_iovec);
	if (xcmd->u.task.kds_src_offset)
		kds_src = (kern_data_store *)((char *)xcmd + xcmd->u.task.kds_src_offset);
	if (!kds_src)
	{
		const GpuCacheIdent *ident = (GpuCacheIdent *)xcmd->u.task.data;
		char		errbuf[120];

		Assert(xcmd->tag == XpuCommandTag__XpuTaskExecGpuCache);
		gc_lmap = gpuCacheGetDeviceBuffer(ident,
										  &m_kds_src,
										  &m_kds_extra,
										  errbuf, sizeof(errbuf));
		if (!gc_lmap)
		{
			gpuClientELog(gclient, "no GpuCache (dat=%u,rel=%u,sig=%09lx) found - %s",
						  ident->database_oid,
						  ident->table_oid,
						  ident->signature,
						  errbuf);
			return;
		}
	}
	else if (kds_src->format == KDS_FORMAT_ROW)
	{
		rc = gpuMemoryPrefetchKDS(kds_src, MY_DEVICE_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientELog(gclient, "failed on gpuMemoryPrefetchKDS: %s",
						  cuStrError(rc));
			return;
		}
		m_kds_src = (CUdeviceptr)kds_src;
	}
	else if (kds_src->format == KDS_FORMAT_BLOCK)
	{
		if (kds_src_pathname && kds_src_iovec)
		{
			s_chunk = gpuservLoadKdsBlock(gclient,
										  kds_src,
										  kds_src_pathname,
										  kds_src_iovec,
										  &npages_direct_read,
										  &npages_vfs_read);
			if (!s_chunk)
				return;
			m_kds_src = s_chunk->m_devptr;
		}
		else
		{
			Assert(kds_src->block_nloaded == kds_src->nitems);
			rc = gpuMemoryPrefetchKDS(kds_src, MY_DEVICE_PER_THREAD);
			if (rc != CUDA_SUCCESS)
			{
				gpuClientELog(gclient, "failed on gpuMemoryPrefetchKDS: %s",
							  cuStrError(rc));
				return;
			}
			m_kds_src = (CUdeviceptr)kds_src;
		}
	}
	else if (kds_src->format == KDS_FORMAT_ARROW)
	{
		if (kds_src_iovec->nr_chunks == 0)
		{
			rc = gpuMemoryPrefetchKDS(kds_src, MY_DEVICE_PER_THREAD);
			if (rc != CUDA_SUCCESS)
			{
				gpuClientELog(gclient, "failed on gpuMemoryPrefetchKDS: %s",
							  cuStrError(rc));
				return;
			}
			m_kds_src = (CUdeviceptr)kds_src;
		}
		else
		{
			if (!kds_src_pathname)
			{
				gpuClientELog(gclient, "GpuScan: arrow file is missing");
				return;
			}
			s_chunk = gpuservLoadKdsArrow(gclient,
										  kds_src,
										  kds_src_pathname,
										  kds_src_iovec,
										  &npages_direct_read,
										  &npages_vfs_read);
			if (!s_chunk)
				return;
			m_kds_src = s_chunk->m_devptr;
		}
	}
	else
	{
		gpuClientELog(gclient, "unknown GpuScan Source format (%c)",
					  kds_src->format);
		return;
	}

	/*
	 * Allocation of destination buffer if necessary
	 */
	if (!GQBUF_KDS_FINAL(gq_buf))
	{
		const kern_data_store *kds_dst_head = SESSION_KDS_DST_HEAD(gclient->h_session);

		sz = (KDS_HEAD_LENGTH(kds_dst_head) +
			  PGSTROM_CHUNK_SIZE);
		d_chunk = gpuMemAllocManaged(sz);
		if (!d_chunk)
		{
			gpuClientFatal(gclient, "failed on gpuMemAllocManaged(%lu)", sz);
			goto bailout;
		}
		kds_dst = (kern_data_store *)d_chunk->m_devptr;
		memcpy(kds_dst, kds_dst_head, KDS_HEAD_LENGTH(kds_dst_head));
		kds_dst->length = sz;
	}

	/*
	 * Build GPU kernel execution plan, if pinned inner-buffer is
	 * partitioned to multiple GPUs.
	 */
	if (gq_buf && gq_buf->h_kmrels)
	{
		kern_multirels *h_kmrels = (kern_multirels *)gq_buf->h_kmrels;
		kern_buffer_partitions *kbuf_parts
			= KERN_MULTIRELS_PARTITION_DESC(h_kmrels, -1);

		if (kbuf_parts)
		{
			int32_t		repeat_id = xcmd->u.task.scan_repeat_id;
			int32_t		start = repeat_id * numGpuDevAttrs;
			int32_t		end = Min(start + numGpuDevAttrs,
								  kbuf_parts->hash_divisor);
			part_divisor = kbuf_parts->hash_divisor;
			part_gcontexts = alloca(sizeof(gpuContext *) * part_divisor);
			part_reminders = alloca(sizeof(uint32_t) * part_divisor);

			assert(start < end);
			for (int k=start; k < end; k++)
			{
				gpumask_t	part_mask = kbuf_parts->parts[k].available_gpus;

				if ((part_mask & gcontext->cuda_dmask) != 0)
				{
					curr_reminder = k;
				}
				else
				{
					gpuContext *__gcontext = __lookupOneRandomGpuContext(part_mask);

					Assert(__gcontext != NULL);
					part_gcontexts[part_nitems] = __gcontext;
					part_reminders[part_nitems] = k;
					part_nitems++;
				}
			}
			assert(curr_reminder >= 0);
		}
		m_kmrels = gq_buf->m_kmrels;
		num_inner_rels = h_kmrels->num_rels;
	}
	/* statistics buffer */
	sz = offsetof(kern_exec_results, stats[num_inner_rels]);
	kern_stats = alloca(sz);
	memset(kern_stats, 0, sz);
	kern_stats->num_rels = num_inner_rels;
	kern_stats->npages_direct_read = npages_direct_read;
	kern_stats->npages_vfs_read = npages_vfs_read;

	/* kick GPU kernel function */
	if (__gpuservLaunchGpuTaskExecKernel(gcontext,
										 gclient,
										 m_kds_src,
										 m_kds_extra,
										 m_kmrels,
										 kds_dst,
										 part_divisor,
										 curr_reminder,
										 0,
										 kern_stats))
	{
		/* for each inner-buffer partitions if any */
		gpuContext *__gcontext_prev = GpuWorkerCurrentContext;
		int		k;

		for (k=0; k < part_nitems; k++)
		{
			uint32_t	__reminder = part_reminders[k];
			gpuContext *__gcontext = part_gcontexts[k];
			gpuMemChunk *n_chunk = NULL;
			CUdeviceptr	m_kds_src_dup = m_kds_src;

			/* switch gpuContext */
			gpuContextSwitchTo(__gcontext);

			/* copy kds_src if raw device memory */
			if (s_chunk != NULL)
			{
				assert(s_chunk->mseg->pool->gcontext == __gcontext_prev);
				assert(m_kds_extra == 0UL);
				n_chunk = gpuMemAlloc(s_chunk->__length);
				if (!n_chunk)
				{
					gpuClientFatal(gclient, "failed on gpuMemAlloc: %s",
								   cuStrError(rc));
					break;
				}
				rc = cuMemcpyPeerAsync(n_chunk->m_devptr,
									   __gcontext->cuda_context,
									   m_kds_src,
									   __gcontext_prev->cuda_context,
									   kds_src->length,
									   MY_STREAM_PER_THREAD);
				if (rc != CUDA_SUCCESS)
				{
					gpuMemFree(n_chunk);
					gpuClientFatal(gclient, "failed on cuMemcpyPeerAsync: %s",
								   cuStrError(rc));
					break;
				}
				m_kds_src_dup = n_chunk->m_devptr;
			}
			/* launch GPU Kernel for other inner buffer partitions */
			if (!__gpuservLaunchGpuTaskExecKernel(__gcontext,
												  gclient,
												  m_kds_src_dup,
												  m_kds_extra,
												  m_kmrels,
												  kds_dst,
												  part_divisor,
												  __reminder,
												  0,
												  kern_stats))
			{
				if (n_chunk != NULL)
					gpuMemFree(n_chunk);
				break;
			}
			if (n_chunk != NULL)
				gpuMemFree(n_chunk);
		}
		gpuContextSwitchTo(__gcontext_prev);
		/* returns the success status, if OK */
		if (k == part_nitems)
			gpuClientWriteBackNormal(gclient, kern_stats, kds_dst);
	}
bailout:
	THREAD_GPU_CONTEXT_VALIDATION_CHECK(gcontext);
	/* release buffers */
	if (t_chunk)
		gpuMemFree(t_chunk);
	if (s_chunk)
		gpuMemFree(s_chunk);
	if (d_chunk)
		gpuMemFree(d_chunk);
    if (gc_lmap)
		gpuCachePutDeviceBuffer(gc_lmap);
}

/* ------------------------------------------------------------
 *
 * gpuservGpuCacheManager - GpuCache worker
 *
 * ------------------------------------------------------------
 */
static void *
gpuservGpuCacheManager(void *__arg)
{
	gpuWorker  *gworker = (gpuWorker *)__arg;
	gpuContext *gcontext = gworker->gcontext;

	gpuContextSwitchTo(gcontext);

	__gsDebug("GPU-%d GpuCache manager thread launched.",
			  MY_DINDEX_PER_THREAD);
	
	gpucacheManagerEventLoop(gcontext->cuda_dindex,
							 gcontext->cuda_context,
							 gcontext->cufn_gpucache_apply_redo,
							 gcontext->cufn_gpucache_compaction);

	/* delete gpuWorker from the gpuContext */
	pthreadMutexLock(&gcontext->worker_lock);
	dlist_delete(&gworker->chain);
	pthreadMutexUnlock(&gcontext->worker_lock);
	free(gworker);

	__gsDebug("GPU-%d GpuCache manager terminated.",
			  MY_DINDEX_PER_THREAD);
	return NULL;
}

/* ----------------------------------------------------------------
 *
 * gpuservHandleRightOuterJoin
 *
 * ----------------------------------------------------------------
 */
static bool
__execMergeRightOuterJoinMap(gpuClient *gclient,
							 gpuContext *dst_gcontext,
							 gpuContext *src_gcontext,
							 kern_multirels *d_kmrels,
							 int depth)
{
	gpuContext *old_gcontext = NULL;
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(d_kmrels, depth);
	uint32_t	nitems_in = (kds_in->nitems + sizeof(uint32_t)-1) / sizeof(uint32_t);
	bool	   *src_ojmap;
	bool	   *dst_ojmap;
	int			grid_sz;
	int			block_sz;
	void	   *kern_args[4];
	CUresult	rc;
	bool		retval = false;

	if (dst_gcontext != GpuWorkerCurrentContext)
		old_gcontext = gpuContextSwitchTo(dst_gcontext);

	dst_ojmap = KERN_MULTIRELS_GPU_OUTER_JOIN_MAP(d_kmrels, depth,
												  dst_gcontext->cuda_dindex);
	src_ojmap = KERN_MULTIRELS_GPU_OUTER_JOIN_MAP(d_kmrels, depth,
												  src_gcontext->cuda_dindex);
	Assert(dst_ojmap != NULL && src_ojmap != NULL);
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 dst_gcontext->cufn_merge_outer_join_map, 0);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on gpuOptimalBlockSize: %s",
					  cuStrError(rc));
		goto bailout;
	}
	grid_sz = Min(grid_sz, (nitems_in + block_sz - 1) / block_sz);

	kern_args[0] = &dst_ojmap;
	kern_args[1] = &src_ojmap;
	kern_args[2] = &nitems_in;
	rc = cuLaunchKernel(dst_gcontext->cufn_merge_outer_join_map,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						MY_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientELog(gclient, "failed on cuLaunchKernel(grid_sz=%d, block_sz=%d): %s",
					  grid_sz, block_sz, cuStrError(rc));
		goto bailout;
	}
	retval = true;
bailout:
	if (old_gcontext)
		gpuContextSwitchTo(old_gcontext);
	return retval;
}

static bool
gpuservHandleRightOuterJoin(gpuClient *gclient,
							gpuContext *gcontext,
							kern_exec_results *kern_stats)
{
	gpuQueryBuffer	   *gq_buf = gclient->gq_buf;
	kern_multirels	   *d_kmrels = (kern_multirels *)gq_buf->m_kmrels;
	kern_data_store	   *kds_dst = GQBUF_KDS_FINAL(gq_buf);
	gpuMemChunk		   *d_chunk = NULL;
	bool				retval = false;

	for (int depth=1; depth <= d_kmrels->num_rels; depth++)
	{
		kern_buffer_partitions *kbuf_parts;

		if (!d_kmrels->chunks[depth-1].right_outer)
			continue;

		if (!kds_dst)
		{
			const kern_data_store *kds_head;
			size_t		sz;

			Assert(!GQBUF_KDS_FINAL(gq_buf));
			kds_head = SESSION_KDS_DST_HEAD(gclient->h_session);
			sz = KDS_HEAD_LENGTH(kds_head) + PGSTROM_CHUNK_SIZE;

			d_chunk = gpuMemAllocManaged(sz);
			if (!d_chunk)
			{
				gpuClientELog(gclient, "failed on gpuMemAllocManaged(%lu)", sz);
				goto bailout;
			}
			kds_dst = (kern_data_store *)d_chunk->m_devptr;
			memcpy(kds_dst, kds_head, KDS_HEAD_LENGTH(kds_head));
			kds_dst->length = sz;
		}

		kbuf_parts = d_kmrels->chunks[depth-1].kbuf_parts;
		if (!kbuf_parts)
		{
			/*
			 * merge the outer-join-map to the currend device.
			 */
			for (int dindex=0; dindex < numGpuDevAttrs; dindex++)
			{
				gpuContext *gcontext_src = &gpuserv_gpucontext_array[dindex];

				if (dindex == gcontext->cuda_dindex)
					continue;
				if (!__execMergeRightOuterJoinMap(gclient,
												  gcontext,
												  gcontext_src,
												  d_kmrels,
												  depth))
					goto bailout;
			}
			/*
			 * Runs the right-outer-join from this depth
			 * and, continue to return partial results.
			 * kds_dst should be allocated here?
			 */
			if (!__gpuservLaunchGpuTaskExecKernel(gcontext,
												  gclient,
												  0UL,
												  0UL,
												  gq_buf->m_kmrels,
												  kds_dst,
												  0, 0,
												  depth,
												  kern_stats))
				goto bailout;
		}
		else if (kbuf_parts->hash_divisor <= numGpuDevAttrs)
		{
			uint32_t	hash_divisor = kbuf_parts->hash_divisor;
			gpuContext **gcontext_parts;

			gcontext_parts = alloca(sizeof(gpuContext *) * hash_divisor);
			memset(gcontext_parts, 0, sizeof(gpuContext *) * hash_divisor);
			/*
			 * merge the outer-join-map to the device that is responsible
			 * to the partition. (If exact 1-partition 1-GPU mapping, it
			 * does not run outer-join-map merging.
			 */
			for (int dindex=0; dindex < numGpuDevAttrs; dindex++)
			{
				gpuContext *gcontext_curr = &gpuserv_gpucontext_array[dindex];
				gpumask_t	cuda_dmask = (1UL<<dindex);
				bool		ojmap_merged = false;

				for (int k=0; k < hash_divisor; k++)
				{
					if ((kbuf_parts->parts[k].available_gpus & cuda_dmask) != 0)
					{
						if (gcontext_parts[k])
						{
							if (!__execMergeRightOuterJoinMap(gclient,
															  gcontext_parts[k],
															  gcontext_curr,
															  d_kmrels,
															  depth))
								goto bailout;
						}
						else
						{
							gcontext_parts[k] = gcontext_curr;
						}
						ojmap_merged = true;
					}
				}
				if (!ojmap_merged)
				{
					gpuClientELog(gclient, "Unable to merge OUTER-JOIN-MAP at GPU-%d",
								  dindex);
					goto bailout;
				}
			}
			/*
			 * Runs the right-outer-join for each pinned-inner-buffer
			 * partitions on the responsible device.
			 */
			for (int k=0; k < hash_divisor; k++)
			{
				if (!__gpuservLaunchGpuTaskExecKernel(gcontext_parts[k],
													  gclient,
													  0UL,
													  0UL,
													  gq_buf->m_kmrels,
													  kds_dst,
													  hash_divisor,
													  k,
													  depth,
													  kern_stats))
					goto bailout;
			}
		}
		else
		{
			/*
			 * MEMO: If number of pinned inner buffer partitions are
			 * larger than the number of GPU devices, we have to send
			 * a special request to run RIGHT-OUTER-JOIN on the partition
			 * at the end of OUTER-SCAN, prior to switch the repeat_id.
			 * On the other hands, we have no reliable way to run the
			 * special request command on the timing.
			 * It is a future To-Do item.
			 */
			gpuClientELog(gclient, "RIGHT/FULL OUTER JOIN performing on the pinned inner buffer partitioned to multiple chunks more than the number of GPU devices are not implemented right now.");
			goto bailout;
        }
	}

	/*
	 * If any results, send back to the client as a partial response
	 */
	if (!GQBUF_KDS_FINAL(gq_buf) &&
		kds_dst && kds_dst->nitems > 0)
	{
		gpuClientWriteBackPartial(gclient, kds_dst);
	}
	retval = true;
bailout:
   if (d_chunk)
       gpuMemFree(d_chunk);
  return retval;
}

/* ----------------------------------------------------------------
 *
 * gpuservHandleGpuTaskFinal
 *
 * ----------------------------------------------------------------
 */
static void
gpuservHandleGpuTaskFinal(gpuContext *gcontext,
						  gpuClient *gclient,
						  XpuCommand *xcmd)
{
	gpuQueryBuffer *gq_buf = gclient->gq_buf;
	kern_multirels *h_kmrels = (kern_multirels *)gq_buf->h_kmrels;
	struct iovec   *iov_array;
	struct iovec   *iov;
	int				iovcnt = 0;
	int				iovmax = (6 * numGpuDevAttrs + 10);
	int				num_rels = (h_kmrels ? h_kmrels->num_rels : 0);
	XpuCommand	   *resp;
	size_t			resp_sz = MAXALIGN(offsetof(XpuCommand,
												u.results.stats[num_rels]));
	/* setup final response message */
	resp = alloca(resp_sz);
	memset(resp, 0, resp_sz);
	resp->magic = XpuCommandMagicNumber;
	resp->tag   = XpuCommandTag__Success;
	resp->u.results.chunks_nitems = 0;
	resp->u.results.chunks_offset = resp_sz;
	resp->u.results.num_rels = num_rels;

	iov_array = alloca(sizeof(struct iovec) * iovmax);
	iov = &iov_array[iovcnt++];
	iov->iov_base = resp;
	iov->iov_len  = resp_sz;

	/*
	 * Is the outer-join-map written back to the host buffer?
	 */
	if (SESSION_SUPPORTS_CPU_FALLBACK(gclient->h_session))
	{
		kern_multirels *d_kmrels = (kern_multirels *)gq_buf->m_kmrels;

		/* Merge RIGHT-OUTER-JOIN Map to the shared host buffer */
		if (h_kmrels && d_kmrels)
		{
			for (int dindex=0; dindex < numGpuDevAttrs; dindex++)
			{
				if ((gclient->optimal_gpus & (1UL<<dindex)) == 0)
					continue;
				for (int depth=1; depth <= d_kmrels->num_rels; depth++)
				{
					kern_data_store *kds = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth);
					bool   *d_ojmap;
					bool   *h_ojmap;

					d_ojmap = KERN_MULTIRELS_GPU_OUTER_JOIN_MAP(d_kmrels, depth, dindex);
					h_ojmap = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth);
					if (d_ojmap && h_ojmap)
					{
						for (uint32_t j=0; j < kds->nitems; j++)
							h_ojmap[j] |= d_ojmap[j];
					}
				}
				pg_memory_barrier();
			}
			/* kick CPU fallback for RIGHT-OUTER-JOIN */
			resp->u.results.right_outer_join = true;
		}

		/*
		 * Is the CPU-Fallback buffer must be written back?
		 */
		for (int __dindex=0; __dindex < numGpuDevAttrs; __dindex++)
		{
			kern_data_store *kds_fallback = (kern_data_store *)
				gq_buf->gpus[__dindex].m_kds_fallback;

			if (kds_fallback && kds_fallback->nitems > 0)
			{
				iovcnt += __gpuClientWriteBackOneChunk(gclient,
													   iov_array+iovcnt,
													   kds_fallback);
				resp->u.results.chunks_nitems++;
			}
		}
	}
	else
	{
		/*
		 * If we have no CPU fallback events in the past and future (during RIGHT OUTER
		 * JOIN handling), we can run RIGHT OUTER JOIN on the GPU device without data
		 * migration.
		 */
		if (!gpuservHandleRightOuterJoin(gclient, gcontext, &resp->u.results))
			return;
		/* this code path should never have CPU-Fallback buffer */
		for (int __dindex=0; __dindex < numGpuDevAttrs; __dindex++)
			Assert(gq_buf->gpus[__dindex].m_kds_fallback == 0UL);
	}

	/*
	 * Is the GpuPreAgg final buffer written back?
	 */
	for (int __dindex=0; __dindex < numGpuDevAttrs; __dindex++)
	{
		kern_data_store *kds_final = (kern_data_store *)
			gq_buf->gpus[__dindex].m_kds_final;

		if (kds_final && kds_final->nitems > 0)
		{
			if ((gclient->xpu_task_flags & DEVTASK__PREAGG) != 0)
			{
				iovcnt += __gpuClientWriteBackOneChunk(gclient,
													   iov_array+iovcnt,
													   kds_final);
				resp->u.results.chunks_nitems++;
			}
			resp->u.results.final_nitems += kds_final->nitems;
			resp->u.results.final_usage  += kds_final->usage;
			resp->u.results.final_total  += KDS_HEAD_LENGTH(kds_final)
				+ sizeof(uint64_t) * (kds_final->hash_nslots +
									  kds_final->nitems)
				+ kds_final->usage;
		}
	}
	resp->u.results.final_plan_task = true;

	for (int i=1; i < iovcnt; i++)
		resp_sz += iov_array[i].iov_len;
	resp->length = resp_sz;

	assert(iovcnt <= iovmax);
	__gpuClientWriteBack(gclient, iov_array, iovcnt);
}

/*
 * gpuservGpuWorkerMain -- actual worker
 */
static void *
gpuservGpuWorkerMain(void *__arg)
{
	gpuWorker  *gworker = (gpuWorker *)__arg;
	gpuContext *gcontext = gworker->gcontext;
	gpuClient  *gclient;

	/* set primary working context */
	gpuContextSwitchTo(gcontext);

	__gsDebug("GPU-%d worker thread launched", MY_DINDEX_PER_THREAD);
	
	pthreadMutexLock(&gcontext->lock);
	while (!gpuServiceGoingTerminate() && !gworker->termination)
	{
		XpuCommand *xcmd;
		dlist_node *dnode;
		uint32_t	count	__attribute__((unused));

		THREAD_GPU_CONTEXT_VALIDATION_CHECK(gcontext);
		if (!dlist_is_empty(&gcontext->command_list))
		{
			dnode = dlist_pop_head_node(&gcontext->command_list);
			xcmd = dlist_container(XpuCommand, chain, dnode);
			pthreadMutexUnlock(&gcontext->lock);

			gclient = xcmd->priv;
			/*
			 * MEMO: If the least bit of gclient->refcnt is not set,
			 * it means the gpu-client connection is no longer available.
			 * (already closed, or error detected.)
			 */
			if ((pg_atomic_read_u32(&gclient->refcnt) & 1) == 1)
			{
				switch (xcmd->tag)
				{
					case XpuCommandTag__XpuTaskExec:
					case XpuCommandTag__XpuTaskExecGpuCache:
						gpuservHandleGpuTaskExec(gcontext, gclient, xcmd);
						break;
					case XpuCommandTag__XpuTaskFinal:
						gpuservHandleGpuTaskFinal(gcontext, gclient, xcmd);
						break;
					default:
						gpuClientELog(gclient, "unknown XPU command (%d)",
									  (int)xcmd->tag);
						break;
				}
			}
			__gpuServiceFreeCommand(xcmd);
			gpuClientPut(gclient, false);
			pthreadMutexLock(&gcontext->lock);
			count = pg_atomic_fetch_sub_u32(&gcontext->num_commands, 1);
			Assert(count > 0);
		}
		else if (!pthreadCondWaitTimeout(&gcontext->cond,
										 &gcontext->lock,
										 5000))
		{
			pthreadMutexUnlock(&gcontext->lock);
			/* maintenance works */
			gpuMemoryPoolMaintenance(gcontext);
			pthreadMutexLock(&gcontext->lock);
		}
	}
	pthreadMutexUnlock(&gcontext->lock);

	/* detach from the gpuContext */
	pthreadMutexLock(&gcontext->worker_lock);
	dlist_delete(&gworker->chain);
	pthreadMutexUnlock(&gcontext->worker_lock);
	free(gworker);

	__gsDebug("GPU-%d worker thread launched", MY_DINDEX_PER_THREAD);

	return NULL;
}

static void *
gpuservMonitorClient(void *__priv)
{
	gpuClient  *gclient = __priv;
	gpuContext *gcontext;
	pgsocket	sockfd = gclient->sockfd;
	CUresult	rc;

	if (dlist_is_empty(&gpuserv_gpucontext_list))
	{
		__gsLog("No GPU context is available");
		goto out;
	}
	gcontext = dlist_container(gpuContext, chain,
							   dlist_head_node(&gpuserv_gpucontext_list));
	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		__gsDebug("failed on cuCtxSetCurrent: %s",
				  cuStrError(rc));
		goto out;
	}
	
	for (;;)
	{
		struct pollfd  pfd;
		int		nevents;

		pfd.fd = sockfd;
		pfd.events = POLLIN;
		pfd.revents = 0;
		nevents = poll(&pfd, 1, -1);
		if (nevents < 0)
		{
			if (errno == EINTR)
				continue;
			__gsDebug("failed on poll(2): %m");
			break;
		}
		if (nevents == 0)
			continue;
		Assert(nevents == 1);
		if (pfd.revents == POLLIN)
		{
			if (__gpuServiceReceiveCommands(sockfd, gclient) < 0)
				break;
		}
		else if (pfd.revents & ~POLLIN)
		{
			__gsDebug("peer socket closed.");
			break;
		}
	}
out:
	gpuClientPut(gclient, true);
	return NULL;
}

/*
 * gpuservAcceptClient
 */
static void
gpuservAcceptClient(void)
{
	pthread_attr_t th_attr;
	gpuClient  *gclient;
	pgsocket	sockfd;
	int			errcode;

	sockfd = accept(gpuserv_listen_sockfd, NULL, NULL);
	if (sockfd < 0)
	{
		elog(LOG, "GPU-Service: could not accept new connection: %m");
		pg_usleep(10000L);		/* wait 10ms */
		return;
	}

	gclient = calloc(1, offsetof(gpuClient, __session[numGpuDevAttrs]));
	if (!gclient)
	{
		elog(LOG, "out of memory");
		close(sockfd);
		return;
	}
	pg_atomic_init_u32(&gclient->refcnt, 1);
	pthreadMutexInit(&gclient->mutex);
	gclient->sockfd = sockfd;

	/* launch workers */
	if (pthread_attr_init(&th_attr) != 0)
		__FATAL("failed on pthread_attr_init");
	if (pthread_attr_setdetachstate(&th_attr, PTHREAD_CREATE_DETACHED) != 0)
		__FATAL("failed on pthread_attr_setdetachstate");

	if ((errcode = pthread_create(&gclient->worker,
								  &th_attr,
								  gpuservMonitorClient,
								  gclient)) != 0)
	{
		elog(LOG, "failed on pthread_create: %s", strerror(errcode));
		close(sockfd);
		free(gclient);
		return;
	}
	pthreadMutexLock(&gpuserv_client_lock);
	dlist_push_tail(&gpuserv_client_list, &gclient->chain);
	pthreadMutexUnlock(&gpuserv_client_lock);
}

/*
 * __setupDevTypeLinkageTable
 */
static HTAB *
__setupDevTypeLinkageTable(CUmodule cuda_module)
{
	xpu_type_catalog_entry *xpu_types_catalog;
	const char *symbol = "builtin_xpu_types_catalog";
	HASHCTL		hctl;
	HTAB	   *htab = NULL;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;
	int			i;
	
	/* build device type table */
	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(TypeOpCode);
	hctl.entrysize = sizeof(xpu_type_catalog_entry);
	hctl.hcxt = TopMemoryContext;
	htab = hash_create("CUDA device type hash table",
					   512,
					   &hctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	rc = cuModuleGetGlobal(&dptr, &nbytes, cuda_module, symbol);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal('%s'): %s",
			 symbol, cuStrError(rc));

	xpu_types_catalog = alloca(nbytes);
	rc = cuMemcpyDtoH(xpu_types_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));
	for (i=0; xpu_types_catalog[i].type_opcode != TypeOpCode__Invalid; i++)
	{
		TypeOpCode	type_opcode = xpu_types_catalog[i].type_opcode;
		xpu_type_catalog_entry *entry;
		bool		found;

		entry = hash_search(htab, &type_opcode, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? duplicated TypeOpCode: %u", (uint32_t)type_opcode);
		Assert(entry->type_opcode == type_opcode);
		entry->type_ops = xpu_types_catalog[i].type_ops;
	}
	return htab;
}

/*
 * __setupDevFuncLinkageTable
 */
static HTAB *
__setupDevFuncLinkageTable(CUmodule cuda_module)
{
	xpu_function_catalog_entry *xpu_funcs_catalog;
	const char *symbol = "builtin_xpu_functions_catalog";
	HASHCTL		hctl;
	HTAB	   *htab;
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;
	int			i;

	memset(&hctl, 0, sizeof(HASHCTL));
	hctl.keysize = sizeof(FuncOpCode);
	hctl.entrysize = sizeof(xpu_function_catalog_entry);
	hctl.hcxt = TopMemoryContext;
	htab = hash_create("CUDA device function hash table",
					   1024,
					   &hctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	rc = cuModuleGetGlobal(&dptr, &nbytes, cuda_module, symbol);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal('%s'): %s",
			 symbol, cuStrError(rc));
	xpu_funcs_catalog = alloca(nbytes);
	rc = cuMemcpyDtoH(xpu_funcs_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));
	for (i=0; xpu_funcs_catalog[i].func_opcode != FuncOpCode__Invalid; i++)
	{
		FuncOpCode	func_opcode = xpu_funcs_catalog[i].func_opcode;
		xpu_function_catalog_entry *entry;
		bool		found;

		entry = hash_search(htab, &func_opcode, HASH_ENTER, &found);
		if (found)
			elog(ERROR, "Bug? duplicated FuncOpCode: %u", (uint32_t)func_opcode);
		Assert(entry->func_opcode == func_opcode);
		entry->func_dptr = xpu_funcs_catalog[i].func_dptr;
	}
	return htab;
}

/*
 * __setupDevEncodeLinkageCatalog
 */
static xpu_encode_info *
__setupDevEncodeLinkageCatalog(CUmodule cuda_module)
{
	xpu_encode_info *xpu_encode_catalog;
	const char *symbol = "xpu_encode_catalog";
	CUdeviceptr	dptr;
	CUresult	rc;
	size_t		nbytes;

	rc = cuModuleGetGlobal(&dptr, &nbytes, cuda_module, symbol);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetGlobal('%s'): %s",
			 symbol, cuStrError(rc));
	xpu_encode_catalog = MemoryContextAlloc(TopMemoryContext, nbytes);
	rc = cuMemcpyDtoH(xpu_encode_catalog, dptr, nbytes);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoH: %s", cuStrError(rc));

	return xpu_encode_catalog;
}

/*
 * __setupGpuKernelFunctionsAndParams
 */
static void
__setupGpuKernelFunctionsAndParams(gpuContext *gcontext)
{
	GpuDevAttributes *dattrs = &gpuDevAttrs[gcontext->cuda_dindex];
	CUmodule	cuda_module = gcontext->cuda_module;
	CUfunction	cuda_function;
	CUresult	rc;
	int			shmem_sz_static;
	int			shmem_sz_dynamic;
	const char *func_name;

	Assert(gcontext->cuda_dindex < numGpuDevAttrs);
	/* ------ kern_gpujoin_main ------ */
	func_name = "kern_gpujoin_main";
	rc = cuModuleGetFunction(&cuda_function, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction('%s'): %s",
			 func_name, cuStrError(rc));
	gcontext->cufn_kern_gpumain = cuda_function;

	rc = cuFuncGetAttribute(&shmem_sz_static,
							CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
							cuda_function);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuFuncGetAttribute(SHARED_SIZE_BYTES): %s",
			 cuStrError(rc));
	shmem_sz_dynamic = (dattrs->MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
						- TYPEALIGN(1024, shmem_sz_static)
						- 8192);	/* margin for L1-cache */
	rc = cuFuncSetAttribute(cuda_function,
							CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
							shmem_sz_dynamic);
	if (rc != CUDA_SUCCESS)
        elog(ERROR, "failed on cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, %d): %s",
			 shmem_sz_dynamic, cuStrError(rc));
	gcontext->gpumain_shmem_sz_limit = shmem_sz_dynamic;

	/* ------ gpujoin_prep_gistindex ------ */
	func_name = "gpujoin_prep_gistindex";
    rc = cuModuleGetFunction(&cuda_function, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction('%s'): %s",
			 func_name, cuStrError(rc));
	gcontext->cufn_prep_gistindex = cuda_function;
	/* ------ gpujoin_merge_outer_join_map ------ */
	func_name = "gpujoin_merge_outer_join_map";
	rc = cuModuleGetFunction(&cuda_function, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction('%s'): %s",
			 func_name, cuStrError(rc));
	gcontext->cufn_merge_outer_join_map = cuda_function;
	/* ------ kern_buffer_partitioning ------ */
	func_name = "kern_buffer_partitioning";
	rc = cuModuleGetFunction(&cuda_function, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction('%s'): %s",
			 func_name, cuStrError(rc));
	gcontext->cufn_kbuf_partitioning = cuda_function;
	/* ------ kern_buffer_reconstruction ------ */
	func_name = "kern_buffer_reconstruction";
	rc = cuModuleGetFunction(&cuda_function, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction('%s'): %s",
			 func_name, cuStrError(rc));
	gcontext->cufn_kbuf_reconstruction = cuda_function;
	/* ------ kern_gpucache_apply_redo ------ */
	func_name = "kern_gpucache_apply_redo";
	rc = cuModuleGetFunction(&cuda_function, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction('%s'): %s",
			 func_name, cuStrError(rc));
	gcontext->cufn_gpucache_apply_redo = cuda_function;
	/* ------ kern_gpucache_compaction ------ */
	func_name = "kern_gpucache_compaction";
	rc = cuModuleGetFunction(&cuda_function, cuda_module, func_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction('%s'): %s",
			 func_name, cuStrError(rc));
	gcontext->cufn_gpucache_compaction = cuda_function;
}

/*
 * gpuservSetupGpuModule
 */
static void
gpuservSetupGpuModule(gpuContext *gcontext)
{
	CUmodule	cuda_module;
	CUresult	rc;

	rc = cuModuleLoad(&cuda_module, pgstrom_fatbin_image_filename);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleLoad('%s'): %s",
			 pgstrom_fatbin_image_filename,
			 cuStrError(rc));
	/* setup XPU linkage hash tables */
	gcontext->cuda_type_htab = __setupDevTypeLinkageTable(cuda_module);
	gcontext->cuda_func_htab = __setupDevFuncLinkageTable(cuda_module);
	gcontext->cuda_encode_catalog = __setupDevEncodeLinkageCatalog(cuda_module);
	gcontext->cuda_module = cuda_module;

	/* lookup CUDA functions and setup parameters */
	__setupGpuKernelFunctionsAndParams(gcontext);
}

/*
 * __gpuContextAdjustWorkers
 */
static void
__gpuContextAdjustWorkersOne(gpuContext *gcontext, uint32_t nworkers)
{
	pthread_attr_t th_attr;
	bool		has_gpucache = false;
	bool		needs_wakeup = false;
	uint32_t	count = 0;
	int			nr_startup = 0;
	int			nr_terminate = 0;
	dlist_iter	__iter;

	pthreadMutexLock(&gcontext->worker_lock);
	dlist_foreach(__iter, &gcontext->worker_list)
	{
		gpuWorker *gworker = dlist_container(gpuWorker, chain, __iter.cur);

		if (gworker->kind == GPUSERV_WORKER_KIND__GPUCACHE)
		{
			if (!gworker->termination)
				has_gpucache = true;
		}
		else if (count < nworkers)
		{
			if (!gworker->termination)
				count++;
			else
				needs_wakeup = true;
		}
		else
		{
			gworker->termination = true;
			needs_wakeup = true;
			nr_terminate++;
		}
	}
	pthreadMutexUnlock(&gcontext->worker_lock);
	if (needs_wakeup)
		pthreadCondBroadcast(&gcontext->cond);
	if (count >= nworkers && has_gpucache)
		goto out;

	/* launch workers */
	if (pthread_attr_init(&th_attr) != 0)
		__FATAL("failed on pthread_attr_init");
	if (pthread_attr_setdetachstate(&th_attr, PTHREAD_CREATE_DETACHED) != 0)
		__FATAL("failed on pthread_attr_setdetachstate");

	while (count < nworkers)
	{
		gpuWorker  *gworker = calloc(1, sizeof(gpuWorker));

		if (!gworker)
		{
			elog(LOG, "out of memory");
			break;
		}
		gworker->gcontext = gcontext;
		gworker->kind = GPUSERV_WORKER_KIND__GPUTASK;
		if ((errno = pthread_create(&gworker->worker,
									&th_attr,
									gpuservGpuWorkerMain,
									gworker)) != 0)
		{
			elog(LOG, "failed on pthread_create: %m");
			free(gworker);
			break;
		}
		pthreadMutexLock(&gcontext->worker_lock);
		dlist_push_tail(&gcontext->worker_list, &gworker->chain);
		pthreadMutexUnlock(&gcontext->worker_lock);
		count++;
		nr_startup += 2;
	}
	if (!has_gpucache)
	{
		gpuWorker  *gworker = calloc(1, sizeof(gpuWorker));

		if (!gworker)
		{
			elog(LOG, "out of memory");
			return;
		}
		gworker->gcontext = gcontext;
		gworker->kind = GPUSERV_WORKER_KIND__GPUCACHE;
		if ((errno = pthread_create(&gworker->worker,
									&th_attr,
									gpuservGpuCacheManager,
									gworker)) != 0)
		{
			elog(LOG, "failed on pthread_create: %m");
			free(gworker);
			return;
		}
		pthreadMutexLock(&gcontext->worker_lock);
		dlist_push_tail(&gcontext->worker_list, &gworker->chain);
		pthreadMutexUnlock(&gcontext->worker_lock);
		nr_startup += 3;
	}
out:
	if (nr_startup > 0 || nr_terminate > 0)
	{
		elog(LOG, "GPU%d workers - %d startup%s, %d terminate",
			 gcontext->cuda_dindex,
			 nr_startup >> 1,
			 (nr_startup & 1) ? " (with GpuCacheManager)" : "",
			 nr_terminate);
	}
}

static void
__gpuContextAdjustWorkers(void)
{
	uint64_t	conf_val;

	conf_val = pg_atomic_read_u64(&gpuserv_shared_state->max_async_tasks);
	if ((conf_val & ~MAX_ASYNC_TASKS_MASK) != 0)
	{
		uint64_t	conf_ts = (conf_val >> MAX_ASYNC_TASKS_BITS);
		uint64_t	curr_ts;
		struct timeval ts;

		gettimeofday(&ts, NULL);
		curr_ts = (ts.tv_sec * 1000L + ts.tv_usec / 1000L);
		if (curr_ts >= conf_ts + MAX_ASYNC_TASKS_DELAY)
		{
			uint32_t	nworkers = (conf_val & MAX_ASYNC_TASKS_MASK);
			dlist_iter	iter;

			dlist_foreach(iter, &gpuserv_gpucontext_list)
			{
				gpuContext *gcontext = dlist_container(gpuContext,
													   chain, iter.cur);
				__gpuContextAdjustWorkersOne(gcontext, nworkers);
			}
			pg_atomic_compare_exchange_u64(&gpuserv_shared_state->max_async_tasks,
										   &conf_val, (uint64_t)nworkers);
		}
	}
}

/*
 * gpuservSetupGpuContext
 */
static gpuContext *
gpuservSetupGpuContext(int cuda_dindex)
{
	GpuDevAttributes *dattrs = &gpuDevAttrs[cuda_dindex];
	gpuContext *gcontext = &gpuserv_gpucontext_array[cuda_dindex];
	CUresult	rc;

	/* gpuContext initialization */
	gcontext->serv_fd = -1;
	gcontext->cuda_dindex = cuda_dindex;
	gcontext->cuda_dmask = (1UL << cuda_dindex);
	pthreadMutexInit(&gcontext->cuda_setlimit_lock);
	pthreadMutexInit(&gcontext->worker_lock);
	dlist_init(&gcontext->worker_list);

	pthreadCondInit(&gcontext->cond);
	pthreadMutexInit(&gcontext->lock);
	dlist_init(&gcontext->command_list);
	pg_atomic_init_u32(&gcontext->num_commands, 0);

	/* Setup raw CUDA context */
	rc = cuDeviceGet(&gcontext->cuda_device, dattrs->DEV_ID);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGet: %s", cuStrError(rc));

	rc = cuDevicePrimaryCtxRetain(&gcontext->cuda_context,
								  gcontext->cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDevicePrimaryCtxRetain: %s", cuStrError(rc));

	rc = cuDevicePrimaryCtxSetFlags(gcontext->cuda_device,
									CU_CTX_SCHED_BLOCKING_SYNC);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDevicePrimaryCtxSetFlags: %s", cuStrError(rc));

	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxSetCurrent: %s", cuStrError(rc));

	gpuservSetupGpuModule(gcontext);
	gpuMemoryPoolInit(gcontext, false, dattrs->DEV_TOTAL_MEMSZ);
	gpuMemoryPoolInit(gcontext, true, dattrs->DEV_TOTAL_MEMSZ);
	/* enable kernel profiling if captured */
	if (getenv("NSYS_PROFILING_SESSION_ID") != NULL)
	{
		rc = cuProfilerStart();
		if (rc != CUDA_SUCCESS)
			elog(LOG, "failed on cuProfilerStart: %s", cuStrError(rc));
		else
			gcontext->cuda_profiler_started = true;
	}
	return gcontext;
}

/*
 * gpuservCleanupGpuContext
 */
static void
gpuservCleanupGpuContext(gpuContext *gcontext)
{
	/*
	 * Wake up all the worker threads, and terminate them.
	 */
	gpuserv_bgworker_got_signal |= (1 << SIGHUP);
	pg_memory_barrier();
	for (;;)
	{
		pthreadCondBroadcast(&gcontext->cond);
		gpucacheManagerWakeUp(gcontext->cuda_dindex);

		pthreadMutexLock(&gcontext->worker_lock);
		if (dlist_is_empty(&gcontext->worker_list))
		{
			pthreadMutexUnlock(&gcontext->worker_lock);
			break;
		}
		pthreadMutexUnlock(&gcontext->worker_lock);
		/* wait 2ms */
		pg_usleep(2000L);
	}

	/* Stop CUDA Profiler */
	if (gcontext->cuda_profiler_started)
	{
		CUresult	rc;

		rc = cuProfilerStop();
		if (rc != CUDA_SUCCESS)
			elog(LOG, "failed on cuProfilerStop: %s", cuStrError(rc));
	}
}

/*
 * SIGTERM/SIGHUP handler
 */
static void
gpuservBgWorkerSignal(SIGNAL_ARGS)
{
	int		saved_errno = errno;

	gpuserv_bgworker_got_signal |= (1 << postgres_signal_arg);

	pg_memory_barrier();

	SetLatch(MyLatch);

	errno = saved_errno;
}

/*
 * gpuservClenupListenSocket
 */
static void
gpuservCleanupOnProcExit(int code, Datum arg)
{
	/* cleanup the UNIX domain socket, if exist */
	struct stat	stat_buf;
	char		path[MAXPGPATH];

	snprintf(path, sizeof(path),
			 ".pg_strom.%u.gpuserv.sock", PostmasterPid);
	if (stat(path, &stat_buf) == 0 &&
		(stat_buf.st_mode & S_IFMT) == S_IFSOCK)
	{
		if (unlink(path) < 0)
			elog(LOG, "failed on unlink('%s'): %m", path);
	}
}

/*
 * gpuservBgWorkerMain
 */
PUBLIC_FUNCTION(void)
gpuservBgWorkerMain(Datum arg)
{
	CUresult	rc;
	int			dindex;

	pqsignal(SIGTERM, gpuservBgWorkerSignal);	/* terminate GpuServ */
	pqsignal(SIGHUP,  gpuservBgWorkerSignal);	/* restart GpuServ */
	BackgroundWorkerUnblockSignals();

	/* Registration of resource cleanup handler */
	dlist_init(&gpuserv_gpucontext_list);
	before_shmem_exit(gpuservCleanupOnProcExit, 0);

	/* Open epoll descriptor */
	gpuserv_epoll_fdesc = epoll_create(30);
	if (gpuserv_epoll_fdesc < 0)
		elog(ERROR, "failed on epoll_create: %m");
	/* Open logger pipe for worker threads */
	gpuservLoggerOpen();
	/* Build the fatbin binary image on demand */
	gpuservSetupFatbin();
	/* Open the server socket */
	gpuservOpenServerSocket();
	/* Init GPU Context for each devices */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", cuStrError(rc));
	/* Misc init */
	srand48(time(NULL) ^ getpid());
	PG_TRY();
	{
		gpuserv_gpucontext_array = calloc(numGpuDevAttrs,
										  sizeof(gpuContext));
		if (!gpuserv_gpucontext_array)
			elog(FATAL, "out of memory");
		for (dindex=0; dindex < numGpuDevAttrs; dindex++)
		{
			gpuContext *gcontext = gpuservSetupGpuContext(dindex);
			dlist_push_tail(&gpuserv_gpucontext_list, &gcontext->chain);
		}
		/* ready to accept connection from the PostgreSQL backend */
		gpuserv_shared_state->gpuserv_ready_accept = true;

		if (!gpuDirectOpenDriver())
			heterodbExtraEreport(true);
		while (!gpuServiceGoingTerminate())
		{
			struct epoll_event	ep_ev;
			int		status;

			if (!PostmasterIsAlive())
				elog(FATAL, "unexpected postmaster dead");
			CHECK_FOR_INTERRUPTS();
			/* launch/eliminate worker threads */
			__gpuContextAdjustWorkers();

			status = epoll_wait(gpuserv_epoll_fdesc, &ep_ev, 1, 4000);
			if (status < 0)
			{
				if (errno != EINTR)
				{
					elog(LOG, "failed on epoll_wait: %m");
					break;
				}
			}
			else if (status > 0)
			{
				/* errors on server socker? */
				if ((ep_ev.events & ~EPOLLIN) != 0)
					break;
				/* any connection pending? */
				if ((ep_ev.events & EPOLLIN) != 0)
				{
					if (ep_ev.data.fd == gpuserv_listen_sockfd)
						gpuservAcceptClient();
					else
						gpuservLoggerDispatch();
				}
			}
		}
	}
	PG_CATCH();
	{
		gpuserv_shared_state->gpuserv_ready_accept = false;
		gpuservCloseServerSocket();
		while (!dlist_is_empty(&gpuserv_gpucontext_list))
		{
			dlist_node *dnode = dlist_pop_head_node(&gpuserv_gpucontext_list);
			gpuContext *gcontext = dlist_container(gpuContext, chain, dnode);
			gpuservCleanupGpuContext(gcontext);
		}
		if (!gpuDirectCloseDriver())
			heterodbExtraEreport(false);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* cleanup */
	gpuserv_shared_state->gpuserv_ready_accept = false;
	gpuservCloseServerSocket();
	while (!dlist_is_empty(&gpuserv_gpucontext_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gpuserv_gpucontext_list);
		gpuContext *gcontext = dlist_container(gpuContext, chain, dnode);
		gpuservCleanupGpuContext(gcontext);
	}
	if (!gpuDirectCloseDriver())
		heterodbExtraEreport(false);

	/*
	 * If it received only SIGHUP (no SIGTERM), try to restart rather than
	 * shutdown.
	 */
	if (gpuserv_bgworker_got_signal == (1 << SIGHUP))
		proc_exit(1);
}

/*
 * pgstrom_request_executor
 */
static void
pgstrom_request_executor(void)
{
	if (shmem_request_next)
		(*shmem_request_next)();
	RequestAddinShmemSpace(MAXALIGN(sizeof(gpuServSharedState)));
}

/*
 * pgstrom_startup_executor
 */
static void
pgstrom_startup_executor(void)
{
	bool    found;

	if (shmem_startup_next)
		(*shmem_startup_next)();
	gpuserv_shared_state = ShmemInitStruct("gpuServSharedState",
										   MAXALIGN(sizeof(gpuServSharedState)),
										   &found);
	memset(gpuserv_shared_state, 0, sizeof(gpuServSharedState));
	pg_atomic_init_u64(&gpuserv_shared_state->max_async_tasks,
					   __pgstrom_max_async_tasks_dummy | (1UL<<MAX_ASYNC_TASKS_BITS));
	pg_atomic_init_u32(&gpuserv_shared_state->gpuserv_debug_output,
					   __gpuserv_debug_output_dummy);
}

/*
 * pgstrom_init_gpu_service
 */
void
pgstrom_init_gpu_service(void)
{
	BackgroundWorker worker;

	Assert(numGpuDevAttrs > 0);
	DefineCustomIntVariable("pg_strom.gpu_mempool_segment_sz",
							"Segment size of GPU memory pool",
							NULL,
							&pgstrom_gpu_mempool_segment_sz_kb,
							1048576,	/* 1GB */
							262144,		/* 256MB */
							16777216,	/* 16GB */
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.gpu_mempool_max_ratio",
							 "GPU memory pool: maximum usable ratio for memory pool (only mapped memory)",
							 NULL,
							 &pgstrom_gpu_mempool_max_ratio,
							 0.50,		/* 50% */
							 0.20,		/* 20% */
							 0.80,		/* 80% */
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomRealVariable("pg_strom.gpu_mempool_min_ratio",
							 "GPU memory pool: minimum preserved ratio memory pool (both mapped/managed memory)",
							 NULL,
							 &pgstrom_gpu_mempool_min_ratio,
							 0.05,		/*  5% */
							 0.0,		/*  0% */
							 pgstrom_gpu_mempool_max_ratio,
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpu_mempool_release_delay",
							"GPU memory pool: time to release device memory segment after the last chunk is released",
							NULL,
							&pgstrom_gpu_mempool_release_delay,
							5000,		/* 5sec */
							1,
							INT_MAX,
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MS | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.max_async_tasks",
							"Limit of concurrent xPU task execution",
							NULL,
							&__pgstrom_max_async_tasks_dummy,
							16,
							1,
							256,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_SUPERUSER_ONLY,
							NULL,
							pgstrom_max_async_tasks_assign,
							pgstrom_max_async_tasks_show);
	DefineCustomStringVariable("pg_strom.cuda_toolkit_basedir",
							   "CUDA Toolkit installation directory",
							   NULL,
							   &pgstrom_cuda_toolkit_basedir,
							   CUDA_TOOLKIT_BASEDIR,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.cuda_stack_limit",
							"Limit of adaptive cuda stack size per thread",
							NULL,
							&__pgstrom_cuda_stack_limit_kb,
							32,		/* 32kB */
							4,		/* 4kB */
							2048,	/* 2MB */
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.gpuserv_debug_output",
							 "enables to generate debug message of GPU service",
							 NULL,
							 &__gpuserv_debug_output_dummy,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE | GUC_SUPERUSER_ONLY,
							 NULL,
							 gpuserv_debug_output_assign,
							 gpuserv_debug_output_show);
	for (int i=0; i < GPU_QUERY_BUFFER_NSLOTS; i++)
		dlist_init(&gpu_query_buffer_hslot[i]);

	memset(&worker, 0, sizeof(BackgroundWorker));
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = 5;
	snprintf(worker.bgw_name, BGW_MAXLEN, "PG-Strom GPU Service");
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_strom");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "gpuservBgWorkerMain");
	worker.bgw_main_arg = 0;
	RegisterBackgroundWorker(&worker);
	/* shared memory setup */
	shmem_request_next = shmem_request_hook;
	shmem_request_hook = pgstrom_request_executor;
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_executor;
}
