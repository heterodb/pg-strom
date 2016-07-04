/*
 * cuda_program.c
 *
 * Routines for just-in-time comple cuda code
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
#include "access/twophase.h"
#include "access/xact.h"
#include "catalog/catalog.h"
#include "catalog/pg_tablespace.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include <nvrtc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "pg_strom.h"
#include "cuda_money.h"
#include "cuda_timelib.h"
#include "cuda_textlib.h"

typedef struct
{
	dlist_node		pgid_chain;
	dlist_node		hash_chain;
	dlist_node		lru_chain;
	/* fields below are never updated once entry is constructed */
	ProgramId		program_id;
	Oid				database_oid;	/* for async build */
	Oid				user_oid;		/* for async build */
	pg_crc32		crc;			/* hash value by extra_flags */
	int				extra_flags;	/*             + kern_define */
	char		   *kern_define;	/*             + kern_source */
	char		   *kern_source;
	/* fields above are never updated once entry is constructed */
	cl_int			refcnt;
	Bitmapset	   *waiting_backends;
	char		   *bin_image;		/* may be CUDA_PROGRAM_BUILD_FAILURE */
	size_t			bin_length;
	char		   *error_msg;
	char		   *extra_buf;		/* alloc pointer for bin_image/error_msg */
	struct timeval	tv_build_end;	/* timestamp when build end */
	char			data[FLEXIBLE_ARRAY_MEMBER];
} program_cache_entry;

#define PGCACHE_MIN_ERRORMSG_BUFSIZE	256
#define PGCACHE_ERRORMSG_LEN(entry)		\
	(dmaBufferSize(entry) -				\
	 ((uintptr_t)(entry)->error_msg -	\
	  (uintptr_t)(entry)))

#define CUDA_PROGRAM_BUILD_FAILURE			((void *)(~0UL))

#define PGCACHE_HASH_SIZE	2048
#define WORDNUM(x)			((x) / BITS_PER_BITMAPWORD)
#define BITNUM(x)			((x) % BITS_PER_BITMAPWORD)

typedef struct
{
	volatile slock_t lock;
	Size		program_cache_usage;
	ProgramId	last_program_id;
	dlist_head	pgid_slots[PGCACHE_HASH_SIZE];
	dlist_head	hash_slots[PGCACHE_HASH_SIZE];
	dlist_head	lru_list;
} program_cache_head;

/* ---- GUC variables ---- */
static Size		program_cache_size;

/* ---- static variables ---- */
static shmem_startup_hook_type shmem_startup_next;
static program_cache_head *pgcache_head = NULL;

/*
 * pgstrom_wakeup_backends
 *
 * wake up the backends that may be blocked for kernel build.
 * we expects caller already hold pgcache_head->lock
 */
static void
pgstrom_wakeup_backends(Bitmapset *waiting_backends)
{
	struct PGPROC  *proc;
	int				idx = -1;

	while ((idx = bms_next_member(waiting_backends, idx)) >= 0)
	{
		Assert(idx < ProcGlobal->allProcCount);
		proc = &ProcGlobal->allProcs[idx];
		SetLatch(&proc->procLatch);
	}
}

/*
 * lookup_cuda_program_entry_nolock - lookup a program_cache_entry by the
 * program_id under the lock
 */
static inline program_cache_entry *
lookup_cuda_program_entry_nolock(ProgramId program_id)
{
	program_cache_entry *entry;
	dlist_iter	iter;
	int			pindex = program_id % PGCACHE_HASH_SIZE;

	dlist_foreach (iter, &pgcache_head->pgid_slots[pindex])
	{
		entry = dlist_container(program_cache_entry, pgid_chain, iter.cur);
		if (entry->program_id == program_id)
			return entry;
	}
	return NULL;
}

/*
 * get_cuda_program_entry_nolock
 */
static void
get_cuda_program_entry_nolock(program_cache_entry *entry)
{
	Assert(entry->refcnt > 0);
	entry->refcnt++;
}

#if 0
/*
 * get_cuda_program_entry
 */
static void
get_cuda_program_entry(program_cache_entry *entry)
{
	SpinLockAcquire(&pgcache_head->lock);
	get_cuda_program_entry_nolock(entry);
	SpinLockRelease(&pgcache_head->lock);
}
#endif

/*
 * put_cuda_program_entry_nolock
 */
static void
put_cuda_program_entry_nolock(program_cache_entry *entry)
{
	if (--entry->refcnt == 0)
	{
		/*
		 * NOTE: unless either reclaim_cuda_program_entry() or
		 * __build_cuda_program() don't detach entry from active
		 * entries hash, it never goes to refcnt == 0.
		 */
		Assert(!entry->pgid_chain.next && !entry->pgid_chain.prev);
		Assert(!entry->hash_chain.next && !entry->hash_chain.prev);
		Assert(!entry->lru_chain.next && !entry->lru_chain.prev);
		pgcache_head->program_cache_usage -= dmaBufferChunkSize(entry);
		elog(INFO, "dmaBufferFree(%zu)", dmaBufferSize(entry));
		if (entry->bin_image != NULL &&
			entry->bin_image != CUDA_PROGRAM_BUILD_FAILURE)
			dmaBufferFree(entry->bin_image);
		dmaBufferFree(entry);
	}
}

/*
 * put_cuda_program_entry
 */
static void
put_cuda_program_entry(program_cache_entry *entry)
{
	SpinLockAcquire(&pgcache_head->lock);
	put_cuda_program_entry_nolock(entry);
	SpinLockRelease(&pgcache_head->lock);
}

/*
 * reclaim_cuda_program_entry
 *
 * it tries to reclaim the shared memory if highly memory presure.
 */
static void
reclaim_cuda_program_entry(void)
{
	program_cache_entry *entry;

	while (pgcache_head->program_cache_usage > program_cache_size &&
		   !dlist_is_empty(&pgcache_head->lru_list))
	{
		entry = dlist_container(program_cache_entry, lru_chain,
								dlist_tail_node(&pgcache_head->lru_list));
		/* remove from the list not to be reclaimed again */
		dlist_delete(&entry->hash_chain);
		dlist_delete(&entry->lru_chain);
		memset(&entry->hash_chain, 0, sizeof(dlist_node));
		memset(&entry->lru_chain, 0, sizeof(dlist_node));
		put_cuda_program_entry_nolock(entry);
	}
}

/*
 * construct_flat_cuda_source
 *
 * It makes a flat cstring kernel source.
 */
static char *
construct_flat_cuda_source(const char *kern_source,
						   const char *kern_define, uint32 extra_flags)
{
	StringInfoData		source;

	initStringInfo(&source);
	appendStringInfo(&source,
					 "#include <cuda_device_runtime_api.h>\n"
					 "\n"
					 "#define HOSTPTRLEN %u\n"
					 "#define DEVICEPTRLEN %lu\n"
					 "#define BLCKSZ %u\n"
					 "#define MAXIMUM_ALIGNOF %u\n"
					 "\n",
					 SIZEOF_VOID_P,
					 sizeof(CUdeviceptr),
					 BLCKSZ,
					 MAXIMUM_ALIGNOF);
#ifdef PGSTROM_DEBUG
	appendStringInfo(&source,
					 "#define PGSTROM_DEBUG %d\n", PGSTROM_DEBUG);
#endif

	/* disable C++ feature */
	appendStringInfo(&source,
					 "#ifdef __cplusplus\n"
					 "extern \"C\" {\n"
					 "#endif	/* __cplusplus */\n");
	/* Declaration of device type oids */
	pgstrom_codegen_typeoid_declarations(&source);

	/* Common PG-Strom device routine */
	appendStringInfoString(&source, pgstrom_cuda_common_code);

	/* Per session definition if any */
	appendStringInfoString(&source, kern_define);

	/* PG-Strom CUDA device code libraries */

	/* cuda dynpara.h */
	if (extra_flags & DEVKERNEL_NEEDS_DYNPARA)
		appendStringInfoString(&source, pgstrom_cuda_dynpara_code);
	/* cuda mathlib.h */
	if (extra_flags & DEVKERNEL_NEEDS_MATHLIB)
		appendStringInfoString(&source, pgstrom_cuda_mathlib_code);
	/* cuda timelib.h */
	if (extra_flags & DEVKERNEL_NEEDS_TIMELIB)
		appendStringInfoString(&source, pgstrom_cuda_timelib_code);
	/* cuda textlib.h */
	if (extra_flags & DEVKERNEL_NEEDS_TEXTLIB)
		appendStringInfoString(&source, pgstrom_cuda_textlib_code);
	/* cuda numeric.h */
	if (extra_flags & DEVKERNEL_NEEDS_NUMERIC)
		appendStringInfoString(&source, pgstrom_cuda_numeric_code);
	/* cuda money.h */
	if (extra_flags & DEVKERNEL_NEEDS_MONEY)
		appendStringInfoString(&source, pgstrom_cuda_money_code);
	/* cuda matrix.h */
	if (extra_flags & DEVKERNEL_NEEDS_MATRIX)
		appendStringInfoString(&source, pgstrom_cuda_matrix_code);

	/* Main logic of each GPU tasks */

	/* GpuScan */
	if (extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
		appendStringInfoString(&source, pgstrom_cuda_gpuscan_code);
	/* GpuHashJoin */
	if (extra_flags & DEVKERNEL_NEEDS_GPUJOIN)
		appendStringInfoString(&source, pgstrom_cuda_gpujoin_code);
	/* GpuPreAgg */
	if (extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
		appendStringInfoString(&source, pgstrom_cuda_gpupreagg_code);
	/* GpuSort */
	if (extra_flags & DEVKERNEL_NEEDS_GPUSORT)
		appendStringInfoString(&source, pgstrom_cuda_gpusort_code);
	/* PL/CUDA functions */
	if (extra_flags & DEVKERNEL_NEEDS_PLCUDA)
		appendStringInfoString(&source, pgstrom_cuda_plcuda_code);

	/* Source code generated on the fly */
	appendStringInfoString(&source, kern_source);

	/* Source code to fix up undefined type/functions */
	appendStringInfoString(&source, pgstrom_cuda_terminal_code);

	/* disable C++ feature */
	appendStringInfo(&source,
					 "#ifdef __cplusplus\n"
					 "}\n"
					 "#endif /* __cplusplus */\n");
	return source.data;
}

/*
 * link_cuda_libraries - links CUDA libraries with the supplied PTX binary
 */
static void
link_cuda_libraries(char *ptx_image, size_t ptx_length, cl_uint extra_flags,
					void **p_bin_image, size_t *p_bin_length)
{
	GpuContext	   *gcontext;
	CUlinkState		lstate;
	CUresult		rc;
	CUjit_option	jit_options[10];
	void		   *jit_option_values[10];
	int				jit_index = 0;
	void		   *bin_image;
	size_t			bin_length;
	char			pathname[MAXPGPATH];

	/* at least one library has to be specified */
	Assert((extra_flags & DEVKERNEL_NEEDS_DYNPARA) != 0);

	/*
	 * NOTE: cuLinkXXXX() APIs works under a particular CUDA context,
	 * so we pick up one of the available GPU device contexts. In case
	 * of background worker, we also have to initialize CUDA library,
	 * pgstrom_get_gpucontext() does it internally.
	 */
	gcontext = pgstrom_get_gpucontext();
	rc = cuCtxPushCurrent(gcontext->gpu[0].cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent");

	/*
	 * JIT Options
	 *
	 * NOTE: Even though CU_JIT_TARGET expects CU_TARGET_COMPUTE_XX is
	 * supplied, it is actually defined as (10 * <major capability> +
	 * <minor capability>), thus it is equivalent to the definition
	 * of pgstrom_baseline_cuda_capability.
	 */
	jit_options[jit_index] = CU_JIT_TARGET;
	jit_option_values[jit_index] = (void *)pgstrom_baseline_cuda_capability();
	jit_index++;

#ifdef PGSTROM_DEBUG
	jit_options[jit_index] = CU_JIT_GENERATE_DEBUG_INFO;
	jit_option_values[jit_index] = (void *)1UL;
	jit_index++;

	jit_options[jit_index] = CU_JIT_GENERATE_LINE_INFO;
	jit_option_values[jit_index] = (void *)1UL;
	jit_index++;
#endif

	/* makes a linkage object */
	rc = cuLinkCreate(jit_index, jit_options, jit_option_values, &lstate);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkCreate: %s", errorText(rc));

	/* add the base PTX image */
	rc = cuLinkAddData(lstate, CU_JIT_INPUT_PTX, ptx_image, ptx_length,
					   "pg-strom", 0, NULL, NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkAddData: %s", errorText(rc));

	/* libcudart.a, if any */
	if (extra_flags & DEVKERNEL_NEEDS_DYNPARA)
	{
		snprintf(pathname, sizeof(pathname), "%s/libcudadevrt.a",
				 CUDA_LIBRARY_PATH);
		rc = cuLinkAddFile(lstate, CU_JIT_INPUT_LIBRARY, pathname,
						   0, NULL, NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLinkAddFile(\"%s\"): %s",
				 pathname, errorText(rc));
	}

	/* do the linkage */
	rc = cuLinkComplete(lstate, &bin_image, &bin_length);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLinkComplete: %s", errorText(rc));

	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
	pgstrom_put_gpucontext(gcontext);

	*p_bin_image = bin_image;
	*p_bin_length = bin_length;
}

/*
 * writeout_cuda_source_file
 *
 * It makes a temporary file to write-out cuda source.
 */
static const char *
writeout_cuda_source_file(char *cuda_source)
{
	static long	sourceFileCounter = 0;
	static char	tempfilepath[MAXPGPATH];
	char		tempdirpath[MAXPGPATH];
	Oid			tablespace_oid;
	File		filp;

	tablespace_oid = (OidIsValid(MyDatabaseTableSpace)
					  ? MyDatabaseTableSpace
					  : DEFAULTTABLESPACE_OID);
	if (tablespace_oid == DEFAULTTABLESPACE_OID ||
		tablespace_oid == GLOBALTABLESPACE_OID)
	{
		/* The default tablespace is {datadir}/base */
		snprintf(tempdirpath, sizeof(tempdirpath), "base/%s",
				 PG_TEMP_FILES_DIR);
	}
	else
	{
		/* All other tablespaces are accessed via symlinks */
		snprintf(tempdirpath, sizeof(tempdirpath), "pg_tblspc/%u/%s/%s",
				 tablespace_oid,
				 TABLESPACE_VERSION_DIRECTORY,
				 PG_TEMP_FILES_DIR);
	}

	/*
	 * Generate a tempfile name that should be unique within the current
	 * database instance.
	 */
	snprintf(tempfilepath, sizeof(tempfilepath), "%s/%s/%s_strom_%d.%ld.gpu",
			 DataDir, tempdirpath, PG_TEMP_FILE_PREFIX, MyProcPid,
			 sourceFileCounter++);

	/*
	 * Open the file.  Note: we don't use O_EXCL, in case there is an orphaned
	 * temp file that can be reused.
	 */
	filp = PathNameOpenFile(tempfilepath,
							O_RDWR | O_CREAT | O_TRUNC | PG_BINARY,
							0600);
	if (filp <= 0)
	{
		/*
		 * We might need to create the tablespace's tempfile directory, if no
		 * one has yet done so.
		 *
		 * Don't check for error from mkdir; it could fail if someone else
		 * just did the same thing.  If it doesn't work then we'll bomb out on
		 * the second create attempt, instead.
		 */
		mkdir(tempdirpath, S_IRWXU);

		filp = PathNameOpenFile(tempfilepath,
								O_RDWR | O_CREAT | O_TRUNC | PG_BINARY,
								0600);
		if (filp <= 0)
			elog(ERROR, "could not create temporary file \"%s\": %m",
				 tempfilepath);
	}

	FileWrite(filp, cuda_source, strlen(cuda_source));

	FileClose(filp);

	return tempfilepath;
}

const char *
pgstrom_cuda_source_file(GpuTaskState *gts)
{
	char   *cuda_source = construct_flat_cuda_source(gts->kern_source,
													 gts->kern_define,
													 gts->extra_flags);
	return writeout_cuda_source_file(cuda_source);
}

static bool
__build_cuda_program(program_cache_entry *entry)
{
	char		   *source;
	const char	   *source_pathname = NULL;
	nvrtcProgram	program;
	nvrtcResult		rc;
	const char	   *options[10];
	int				opt_index = 0;
	void		   *bin_image;
	size_t			bin_length;
	char		   *build_log;
	size_t			length;
	Size			required;
	char		   *extra_buf;
	char		   *bin_image_new;
	char		   *error_msg_new;
	bool			build_success = true;

	/*
	 * Make a nvrtcProgram object
	 */
	source = construct_flat_cuda_source(entry->kern_source,
										entry->kern_define,
										entry->extra_flags);
	rc = nvrtcCreateProgram(&program,
							source,
							"pg_strom",
							0,
							NULL,
							NULL);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcCreateProgram: %s",
			 nvrtcGetErrorString(rc));

	/*
	 * Put command line options
	 */
	options[opt_index++] = "-I " CUDA_INCLUDE_PATH;
	options[opt_index++] =
		psprintf("--gpu-architecture=compute_%lu",
				 pgstrom_baseline_cuda_capability());
#ifdef PGSTROM_DEBUG
	options[opt_index++] = "--device-debug";
	options[opt_index++] = "--generate-line-info";
#endif
	options[opt_index++] = "--use_fast_math";
	/* library linkage needs relocatable PTX */
	if (entry->extra_flags & DEVKERNEL_NEEDS_DYNPARA)
		options[opt_index++] = "--relocatable-device-code=true";

	/*
	 * Kick runtime compiler
	 */
	rc = nvrtcCompileProgram(program, opt_index, options);
	if (rc != NVRTC_SUCCESS)
	{
		if (rc == NVRTC_ERROR_COMPILATION)
			build_success = false;
		else
			elog(ERROR, "failed on nvrtcCompileProgram: %s",
				 nvrtcGetErrorString(rc));
	}

	/*
	 * Save the source file, if required or build failure
	 */
	if (!build_success)
		source_pathname = writeout_cuda_source_file(source);

	/*
	 * Read PTX Binary
	 */
	if (build_success)
	{
		char	   *ptx_image;
		size_t		ptx_length;

		rc = nvrtcGetPTXSize(program, &ptx_length);
		if (rc != NVRTC_SUCCESS)
			elog(ERROR, "failed on nvrtcGetPTXSize: %s",
				 nvrtcGetErrorString(rc));
		ptx_image = palloc(ptx_length + 1);

		rc = nvrtcGetPTX(program, ptx_image);
		if (rc != NVRTC_SUCCESS)
			elog(ERROR, "failed on nvrtcGetPTX: %s",
				 nvrtcGetErrorString(rc));
		ptx_image[ptx_length++] = '\0';	/* may not be necessary */

		/*
		 * Link the required run-time libraries, if any
		 */
		if (entry->extra_flags & DEVKERNEL_NEEDS_DYNPARA)
		{
			link_cuda_libraries(ptx_image, ptx_length,
								entry->extra_flags,
								&bin_image, &bin_length);
			pfree(ptx_image);
		}
		else
		{
			bin_image = ptx_image;
			bin_length = ptx_length;
		}
	}
	else
	{
		bin_image = NULL;
		bin_length = 0;
	}

	/*
	 * Read Log Output
	 */
	rc = nvrtcGetProgramLogSize(program, &length);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcGetProgramLogSize: %s",
			 nvrtcGetErrorString(rc));
	build_log = palloc(length + 1);

	rc = nvrtcGetProgramLog(program, build_log);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcGetProgramLog: %s",
			 nvrtcGetErrorString(rc));
	build_log[length] = '\0';	/* may not be necessary? */

	/*
	 * Attach bin_image/build_log to the existing entry
	 */
	required = (bin_image ? MAXALIGN(bin_length) : 0) +
		MAXALIGN(strlen(build_log)) +
		PGCACHE_MIN_ERRORMSG_BUFSIZE;	/* margin for error message */
	extra_buf = dmaBufferAlloc(MasterGpuContext(), required);

	if (bin_image)
	{
		bin_image_new = extra_buf;
		error_msg_new = extra_buf + MAXALIGN(bin_length);
		memcpy(bin_image_new, bin_image, bin_length);
	}
	else
	{
		bin_image_new = CUDA_PROGRAM_BUILD_FAILURE;
		error_msg_new = extra_buf;
	}

	if (source_pathname)
		snprintf(error_msg_new, required - MAXALIGN(bin_length),
				 "build %s:\n%s\nsource: %s",
				 bin_image ? "success" : "failure",
				 build_log, source_pathname);
	else
		snprintf(error_msg_new, required - MAXALIGN(bin_length),
				 "build %s:\n%s",
                 bin_image ? "success" : "failure",
				 build_log);

	/* update the program entry */
	SpinLockAcquire(&pgcache_head->lock);
	Assert(entry->bin_image == NULL);
	entry->bin_image = bin_image_new;
	entry->error_msg = error_msg_new;
	entry->extra_buf = extra_buf;		/* to be released later */
	pgcache_head->program_cache_usage += dmaBufferChunkSize(extra_buf);
	/* wake up backends which wait for build */
	pgstrom_wakeup_backends(entry->waiting_backends);
	entry->waiting_backends = NULL;
	/* reclaim the older buffer if overconsumption */
	dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);
	if (pgcache_head->program_cache_usage > program_cache_size)
		reclaim_cuda_program_entry();
	SpinLockRelease(&pgcache_head->lock);

	/* release nvrtcProgram object */
	rc = nvrtcDestroyProgram(&program);
	if (rc != NVRTC_SUCCESS)
   		elog(WARNING, "failed on nvrtcDestroyProgram: %s",
			 nvrtcGetErrorString(rc));

	return build_success;
}	

static bool
pgstrom_build_cuda_program(program_cache_entry *entry)
{
	MemoryContext memcxt = CurrentMemoryContext;
	MemoryContext oldcxt;
	bool		retval;

	PG_TRY();
	{
		retval = __build_cuda_program(entry);
	}
	PG_CATCH();
	{
		ErrorData  *errdata;

		oldcxt = MemoryContextSwitchTo(memcxt);
		errdata = CopyErrorData();

		SpinLockAcquire(&pgcache_head->lock);
		if (!entry->bin_image)
		{
			snprintf(entry->error_msg, PGCACHE_ERRORMSG_LEN(entry),
					 "(%s:%d, %s) %s",
					 errdata->filename,
					 errdata->lineno,
					 errdata->funcname,
					 errdata->message);
			entry->bin_image = CUDA_PROGRAM_BUILD_FAILURE;
		}
		pgstrom_wakeup_backends(entry->waiting_backends);
		entry->waiting_backends = NULL;
		SpinLockRelease(&pgcache_head->lock);
		MemoryContextSwitchTo(oldcxt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return retval;
}

static void
build_cuda_program_async_main(Datum bgw_arg)
{
	ProgramId	program_id = (ProgramId) bgw_arg;
	program_cache_entry *entry;

	BackgroundWorkerUnblockSignals();
	/* Fetch database_oid/user_oid */
	SpinLockAcquire(&pgcache_head->lock);
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(WARNING, "Bug? CUDA Program ID=%lu has already gone", program_id);
		return;
	}
	get_cuda_program_entry_nolock(entry);
	SpinLockRelease(&pgcache_head->lock);

	/* Set up a memory context and resource owner. */
	Assert(CurrentResourceOwner == NULL);
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "CUDA Async Builder");
	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "CUDA Async Builder",
												 ALLOCSET_DEFAULT_MINSIZE,
												 ALLOCSET_DEFAULT_INITSIZE,
												 ALLOCSET_DEFAULT_MAXSIZE);
	/* Restore the database connection. */
    BackgroundWorkerInitializeConnectionByOid(entry->database_oid,
											  entry->user_oid);
	/* Start dummy transaction */
	StartTransactionCommand();
	PG_TRY();
	{
		pgstrom_build_cuda_program(entry);
	}
	PG_CATCH();
	{
		put_cuda_program_entry(entry);
		PG_RE_THROW();
	}
	PG_END_TRY();
	CommitTransactionCommand();
	/* Unpin the entry */
	put_cuda_program_entry(entry);
}

/*
 * pgstrom_create_cuda_program
 *
 * It makes a new GPU program cache entry, or acquires an existing entry if
 * equivalent one is already exists.
 */
ProgramId
pgstrom_create_cuda_program(GpuContext_v2 *gcontext,
							const char *kern_source,
							const char *kern_define,
							cl_uint extra_flags,
							bool try_async_build)
{
	program_cache_entry	*entry;
	ProgramId	program_id;
	Size		kern_source_len = strlen(kern_source);
	Size		kern_define_len = strlen(kern_define);
	Size		required;
	Size		usage;
	int			nwords;
	int			pindex;
	int			hindex;
	dlist_iter	iter;
	pg_crc32	crc;
	BackgroundWorker worker;

	/* makes a hash value */
	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &extra_flags, sizeof(int32));
	COMP_LEGACY_CRC32(crc, kern_source, kern_source_len);
	COMP_LEGACY_CRC32(crc, kern_define, kern_define_len);
	FIN_LEGACY_CRC32(crc);

	hindex = crc % PGCACHE_HASH_SIZE;
	SpinLockAcquire(&pgcache_head->lock);
	dlist_foreach (iter, &pgcache_head->hash_slots[hindex])
	{
		entry = dlist_container(program_cache_entry, hash_chain, iter.cur);

		if (entry->crc == crc &&
			entry->extra_flags == extra_flags &&
			strcmp(entry->kern_source, kern_source) == 0 &&
			strcmp(entry->kern_define, kern_define) == 0)
		{
			/* Move this entry to the head of LRU list */
			dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);

			/* Raise an syntax error if already failed on build */
			if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
			{
				SpinLockRelease(&pgcache_head->lock);
				elog(ERROR, "%s", entry->error_msg);
			}
			/*
			 * Kernel build is still in-progress, so register myself a list
			 * of wakeup backend processes.
			 */
			if (!entry->bin_image)
			{
				Bitmapset  *waiting_backends = entry->waiting_backends;
				waiting_backends->words[WORDNUM(MyProc->pgprocno)]
					|= (1 << BITNUM(MyProc->pgprocno));
			}
			/*
			 * OK, this CUDA program already exist (even though it may be
			 * under the asynchronous build in-progress)
			 */
			program_id = entry->program_id;
			get_cuda_program_entry_nolock(entry);
			SpinLockRelease(&pgcache_head->lock);

			/*
			 * Also, we have to track this program entry locally, to release
			 * it normally when transaction is closed abnormally.
			 */
			PG_TRY();
			{
				// FIXME: It shall be always tracked in the future version
				if (gcontext)
					trackCudaProgram(gcontext, program_id);
			}
			PG_CATCH();
			{
				pgstrom_put_cuda_program(NULL, program_id);
				PG_RE_THROW();
			}
			PG_END_TRY();

			return program_id;
		}
	}

	/*
	 * Not found on the existing program cache.
	 * So, create a new entry then kick NVRTC
	 */
	PG_TRY();
	{
		required = offsetof(program_cache_entry, data[0]);
		nwords = (ProcGlobal->allProcCount +
				  BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
		required += MAXALIGN(offsetof(Bitmapset, words[nwords]));
		required += MAXALIGN(kern_source_len + 1);
		required += MAXALIGN(kern_define_len + 1);
		required += PGCACHE_MIN_ERRORMSG_BUFSIZE;
		usage = 0;

		entry = dmaBufferAlloc(MasterGpuContext(), required);
		elog(INFO, "%d: dmaBufferAlloc(%zu)", __LINE__, required);

		/* find out a unique program_id */
	retry_program_id:
		program_id = ++pgcache_head->last_program_id;
		pindex = program_id % PGCACHE_HASH_SIZE;
		dlist_foreach (iter, &pgcache_head->pgid_slots[pindex])
		{
			program_cache_entry	   *temp
				= dlist_container(program_cache_entry, pgid_chain, iter.cur);
			if (temp->program_id == program_id)
				goto retry_program_id;
		}
		entry->program_id = program_id;
		/* session info for asynchronous build */
		entry->database_oid = MyDatabaseId;
		entry->user_oid = GetUserId();

		/* device kernel source */
		entry->crc = crc;
		entry->extra_flags = extra_flags;

		entry->kern_define = (char *)(entry->data + usage);
		memcpy(entry->kern_define, kern_define, kern_define_len + 1);
		usage += MAXALIGN(kern_define_len + 1);

		entry->kern_source = (char *)(entry->data + usage);
		memcpy(entry->kern_source, kern_source, kern_source_len + 1);
		usage += MAXALIGN(kern_source_len + 1);

		/* reference count */
		entry->refcnt = 2;

		/* bitmap for waiting backends */
		entry->waiting_backends = (Bitmapset *)(entry->data + usage);
		entry->waiting_backends->nwords = nwords;
		memset(entry->waiting_backends->words, 0, sizeof(bitmapword) * nwords);
		usage += MAXALIGN(offsetof(Bitmapset, words[nwords]));

		/* no cuda binary at this moment */
		entry->bin_image = NULL;
		entry->bin_length = 0;
		/* remaining are for error message */
		entry->error_msg = (char *)(entry->data + usage);
		/* no extra buffer at this moment */
		entry->extra_buf = NULL;

		/* at least, caller is waiting for build */
		entry->waiting_backends = entry->waiting_backends;
		entry->waiting_backends->words[WORDNUM(MyProc->pgprocno)]
			|= (1 << BITNUM(MyProc->pgprocno));

		/* add an entry for program build in-progress */
		dlist_push_head(&pgcache_head->pgid_slots[pindex],
						&entry->pgid_chain);
		dlist_push_head(&pgcache_head->hash_slots[hindex],
						&entry->hash_chain);
		dlist_push_head(&pgcache_head->lru_list,
						&entry->lru_chain);
		pgcache_head->program_cache_usage += dmaBufferChunkSize(entry);
		if (pgcache_head->program_cache_usage > program_cache_size)
			reclaim_cuda_program_entry();
	}
	PG_CATCH();
	{
		SpinLockRelease(&pgcache_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&pgcache_head->lock);

	/*
	 * track this new entry locally
	 */
	PG_TRY();
	{
		// FIXME: It shall be always tracked in the future version
		if (gcontext)
			trackCudaProgram(gcontext, program_id);
	}
	PG_CATCH();
	{
		pgstrom_put_cuda_program(NULL, program_id);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/*
	 * Kick a dynamic background worker to build
	 */
	if (try_async_build)
	{
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "NVRTC launcher - program_id = %lu, crc %08x",
				 program_id, crc);
		worker.bgw_flags = (BGWORKER_SHMEM_ACCESS |
							BGWORKER_BACKEND_DATABASE_CONNECTION);
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = BGW_NEVER_RESTART;
		worker.bgw_main = build_cuda_program_async_main;
		worker.bgw_main_arg = Int64GetDatum(program_id);
		if (RegisterDynamicBackgroundWorker(&worker, NULL))
			return program_id;
	}
	/* synchronous build, or fallback if failed to launch BGworker */
	if (!pgstrom_build_cuda_program(entry))
		elog(ERROR, "%s", entry->error_msg);

	return entry->program_id;
}

/*
 * pgstrom_get_cuda_program
 *
 * acquire an existing GPU program entry
 */
void
pgstrom_get_cuda_program(GpuContext_v2 *gcontext, ProgramId program_id)
{
	program_cache_entry *entry;

	SpinLockAcquire(&pgcache_head->lock);
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "ProgramId=%lu not found", program_id);
	}
	Assert(entry->refcnt > 0);
	entry->refcnt++;
	SpinLockRelease(&pgcache_head->lock);

	PG_TRY();
	{
		trackCudaProgram(gcontext, program_id);
	}
	PG_CATCH();
	{
		pgstrom_put_cuda_program(NULL, program_id);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * pgstrom_put_cuda_program
 *
 * release an existing GPU program entry
 */
void
pgstrom_put_cuda_program(GpuContext_v2 *gcontext, ProgramId program_id)
{
	program_cache_entry *entry;

	/*
	 * untrack this program entry locally.
	 *
	 * Note that this function can be called with gcontext==NULL, when
	 * caller controls the state of tracking; like error handling during
	 * the tracking or untracking.
	 */
	if (gcontext)
		untrackCudaProgram(gcontext, program_id);

	SpinLockAcquire(&pgcache_head->lock);
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "ProgramId=%lu not found", program_id);
	}
	put_cuda_program_entry_nolock(entry);
   	SpinLockRelease(&pgcache_head->lock);
}

/*
 * pgstrom_wait_cuda_program
 *
 * wait for asynchronous build completion of the CUDA program.
 * timeout == 0 means non-blocked status check.
 */
bool
pgstrom_wait_cuda_program(ProgramId program_id, long timeout)
{
	program_cache_entry *entry;
	struct timeval	tv_last;
	struct timeval	tv_curr;
	int				ev;
	bool			retval = false;

	gettimeofday(&tv_last, NULL);
	for (;;)
	{
		ResetLatch(MyLatch);

		SpinLockAcquire(&pgcache_head->lock);
		entry = lookup_cuda_program_entry_nolock(program_id);
		if (!entry)
		{
			SpinLockRelease(&pgcache_head->lock);
			elog(ERROR, "CUDA Program ID=%lu was not found", program_id);
		}

		if (entry->bin_image == NULL)
		{
			/* program build is in-progress */
			if (timeout == 0)
				break;
			/* elsewhere, wait for completion of async build */
			SpinLockRelease(&pgcache_head->lock);

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   timeout);
			/* emergency bailout if postmaster is dead */
			if (ev & WL_POSTMASTER_DEATH)
				proc_exit(1);
			/* check status again, but never wait any more */
			if (ev & WL_TIMEOUT)
				timeout = 0;
			else
			{
				/*
				 * elsewhere, someone set a latch; likely, asynchronous build
				 * gets completed. Check it again. If it is false signal, we
				 * try to sleep with shorter timeout.
				 */
				gettimeofday(&tv_curr, NULL);

				timeout -= (tv_curr.tv_sec * 1000 + tv_curr.tv_usec / 1000 -
							tv_last.tv_sec * 1000 + tv_curr.tv_usec / 1000);
				timeout = Max(timeout, 0);

				tv_last = tv_curr;
			}
		}
		else
		{
			/* program is already built (or failure to build) */
			if (entry->bin_image != CUDA_PROGRAM_BUILD_FAILURE)
				retval = true;
			break;
		}
	}
	SpinLockRelease(&pgcache_head->lock);

	return retval;
}










#if 1
/* The legacy interface; to be revised at v9.6 support */
static CUmodule *
__load_cuda_program(GpuContext *gcontext, ProgramId program_id)
{
	program_cache_entry *entry;
	int				i, num_context;
	char		   *bin_image;
	CUresult		rc;
	CUmodule	   *cuda_modules = NULL;

	SpinLockAcquire(&pgcache_head->lock);
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "Bug? CUDA Program=%lu was not found", program_id);
	}
	bin_image = entry->bin_image;

	/* Is this program has build error? */
	if (bin_image == CUDA_PROGRAM_BUILD_FAILURE)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "%s", entry->error_msg);
	}
	/* Is kernel build still in-progress? */
	if (!bin_image)
	{
		Bitmapset  *waiting_backends = entry->waiting_backends;
		waiting_backends->words[WORDNUM(MyProc->pgprocno)]
			|= (1 << BITNUM(MyProc->pgprocno));
		SpinLockRelease(&pgcache_head->lock);
		return NULL;
	}
	SpinLockRelease(&pgcache_head->lock);

	/*
	 * OK, it seems to me the kernel gets successfully built
	 * Note that 'bin_image' shall not be changed once CUDA program
	 * entry is built, unless its reference counter gets zero.
	 * Because pgstrom_create_cuda_program() pinned the entry already,
	 * we can reference the bin_image safely.
	 */

	/* Let's load this module for each context */
	num_context = gcontext->num_context;
	PG_TRY();
	{
		cuda_modules = MemoryContextAllocZero(gcontext->memcxt,
											  sizeof(CUmodule) *
											  num_context);
		for (i=0; i < num_context; i++)
		{
			rc = cuCtxPushCurrent(gcontext->gpu[i].cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent (%s)", errorText(rc));

			rc = cuModuleLoadData(&cuda_modules[i], bin_image);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuModuleLoadData (%s)\n",
					 errorText(rc));

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPopCurrent (%s)", errorText(rc));
			}
	}
	PG_CATCH();
	{
		while (cuda_modules && i > 0)
		{
			rc = cuModuleUnload(cuda_modules[--i]);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuModuleUnload (%s)", errorText(rc));
		}
		/* Legacy interface does not pin the entry for long time */
		pgstrom_put_cuda_program(NULL, program_id);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Legacy interface does not pin the entry for long time */
	pgstrom_put_cuda_program(NULL, program_id);

	return cuda_modules;
}

/*
 * pgstrom_load_cuda_program
 *
 * a legacy interface to assign cuda_modules on the GpuTaskState
 */
bool
pgstrom_load_cuda_program(GpuTaskState *gts, bool is_preload)
{
	ProgramId		program_id;
	CUmodule	   *cuda_modules;

	program_id = pgstrom_create_cuda_program(NULL,
											 gts->kern_source,
											 gts->kern_define,
											 gts->extra_flags,
											 true);
	cuda_modules = __load_cuda_program(gts->gcontext, program_id);
	if (cuda_modules != NULL)
	{
		gts->cuda_modules = cuda_modules;
		return true;
	}
	return false;
}

/*
 * plcuda_load_cuda_program
 *
 * It builds the program synchronously and load CUmodule.
 * Used for pl/cuda functions
 */
CUmodule *
plcuda_load_cuda_program(GpuContext *gcontext,
						 const char *kern_source,
						 cl_uint extra_flags)
{
	ProgramId		program_id;
	char		   *kern_define
		= pgstrom_build_session_info(extra_flags, NULL);
	program_id =  pgstrom_create_cuda_program(NULL,
											  kern_source,
											  kern_define,
											  extra_flags,
											  false);
	/*
	 * FIXME: we need to pay attention if __load_cuda_program() returns NULL;
	 * when concurrent session already launched asynchronous build, but not
	 * completed yet.
	 */
	return __load_cuda_program(gcontext, program_id);
}
#endif

/*
 * construct_kern_parambuf
 *
 * It construct a kernel parameter buffer to deliver Const/Param nodes.
 */
static kern_parambuf *
construct_kern_parambuf(List *used_params, ExprContext *econtext)
{
	StringInfoData	str;
	kern_parambuf  *kparams;
	char		padding[STROMALIGN_LEN];
	ListCell   *cell;
	Size		offset;
	int			index = 0;
	int			nparams = list_length(used_params);

	memset(padding, 0, sizeof(padding));

	/* seek to the head of variable length field */
	offset = STROMALIGN(offsetof(kern_parambuf, poffset[nparams]));
	initStringInfo(&str);
	enlargeStringInfo(&str, offset);
	memset(str.data, 0, offset);
	str.len = offset;
	/* walks on the Para/Const list */
	foreach (cell, used_params)
	{
		Node   *node = lfirst(cell);

		if (IsA(node, Const))
		{
			Const  *con = (Const *) node;

			kparams = (kern_parambuf *)str.data;
			if (con->constisnull)
				kparams->poffset[index] = 0;	/* null */
			else if (con->constbyval)
			{
				Assert(con->constlen > 0);
				kparams->poffset[index] = str.len;
				appendBinaryStringInfo(&str,
									   (char *)&con->constvalue,
									   con->constlen);
			}
			else
			{
				kparams->poffset[index] = str.len;
				if (con->constlen > 0)
					appendBinaryStringInfo(&str,
										   DatumGetPointer(con->constvalue),
										   con->constlen);
				else
					appendBinaryStringInfo(&str,
                                           DatumGetPointer(con->constvalue),
                                           VARSIZE(con->constvalue));
			}
		}
		else if (IsA(node, Param))
		{
			ParamListInfo param_info = econtext->ecxt_param_list_info;
			Param  *param = (Param *) node;

			if (param_info &&
				param->paramid > 0 && param->paramid <= param_info->numParams)
			{
				ParamExternData	*prm = &param_info->params[param->paramid - 1];

				/* give hook a chance in case parameter is dynamic */
				if (!OidIsValid(prm->ptype) && param_info->paramFetch != NULL)
					(*param_info->paramFetch) (param_info, param->paramid);

				kparams = (kern_parambuf *)str.data;
				if (!OidIsValid(prm->ptype))
				{
					elog(INFO, "debug: Param has no particular data type");
					kparams->poffset[index++] = 0;	/* null */
					continue;
				}
				/* safety check in case hook did something unexpected */
				if (prm->ptype != param->paramtype)
					ereport(ERROR,
							(errcode(ERRCODE_DATATYPE_MISMATCH),
							 errmsg("type of parameter %d (%s) does not match that when preparing the plan (%s)",
									param->paramid,
									format_type_be(prm->ptype),
									format_type_be(param->paramtype))));
				if (prm->isnull)
					kparams->poffset[index] = 0;	/* null */
				else
				{
					int16	typlen;
					bool	typbyval;

					get_typlenbyval(prm->ptype, &typlen, &typbyval);
					if (typbyval)
					{
						appendBinaryStringInfo(&str,
											   (char *)&prm->value,
											   typlen);
					}
					else if (typlen > 0)
					{
						appendBinaryStringInfo(&str,
											   DatumGetPointer(prm->value),
											   typlen);
					}
					else
					{
						appendBinaryStringInfo(&str,
											   DatumGetPointer(prm->value),
											   VARSIZE(prm->value));
					}
				}
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_UNDEFINED_OBJECT),
						 errmsg("no value found for parameter %d",
								param->paramid)));
			}
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(node));

		/* alignment */
		if (STROMALIGN(str.len) != str.len)
			appendBinaryStringInfo(&str, padding,
								   STROMALIGN(str.len) - str.len);
		index++;
	}
	Assert(STROMALIGN(str.len) == str.len);
	kparams = (kern_parambuf *)str.data;
	kparams->hostptr = (hostptr_t) &kparams->hostptr;
	kparams->xactStartTimestamp = GetCurrentTransactionStartTimestamp();
	kparams->length = str.len;
	kparams->nparams = nparams;

	return kparams;
}

/*
 * pgstrom_build_session_info
 *
 * it build a session specific code. if extra_flags contains a particular
 * custom_scan node related GPU routine, GpuTaskState must be provided.
 */
char *
pgstrom_build_session_info(cl_uint extra_flags,
						   GpuTaskState *gts)
{
	StringInfoData	buf;

	if ((extra_flags & (DEVKERNEL_NEEDS_TIMELIB |
						DEVKERNEL_NEEDS_MONEY   |
						DEVKERNEL_NEEDS_TEXTLIB |
						DEVKERNEL_NEEDS_GPUSCAN |
						DEVKERNEL_NEEDS_GPUJOIN |
						DEVKERNEL_NEEDS_GPUSORT)) == 0)
		return "";	/* no session specific code */

	Assert(gts != NULL || (extra_flags & (DEVKERNEL_NEEDS_GPUSCAN |
										  DEVKERNEL_NEEDS_GPUJOIN |
										  DEVKERNEL_NEEDS_GPUSORT)) == 0);
	initStringInfo(&buf);

	/* put timezone info */
	if ((extra_flags & DEVKERNEL_NEEDS_TIMELIB) != 0)
		assign_timelib_session_info(&buf);
	/* put currency info */
	if ((extra_flags & DEVKERNEL_NEEDS_MONEY) != 0)
		assign_moneylib_session_info(&buf);
	/* put text/string info */
	if ((extra_flags & DEVKERNEL_NEEDS_TEXTLIB) != 0)
		assign_textlib_session_info(&buf);

	/* enables device projection? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUSCAN) != 0)
		assign_gpuscan_session_info(&buf, gts);
	/* enables device projection? */
//	if ((extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
//		assign_gpujoin_session_info(&buf, gts);
	/* enables device projection? */
//	if ((extra_flags & DEVKERNEL_NEEDS_GPUSORT) != 0)
//		assign_gpusort_session_info(&buf, gts);

	return buf.data;
}

/*
 * pgstrom_assign_cuda_program
 *
 * It assigns kernel_source and extra_flags on the given GpuTaskState.
 * Also, construct per-session specific definition according to the
 * extra_flags.
 */
void
pgstrom_assign_cuda_program(GpuTaskState *gts,
							List *used_params,
							const char *kern_source,
							int extra_flags)
{
	ExprContext	   *econtext = gts->css.ss.ps.ps_ExprContext;
	const char	   *kern_define
		= pgstrom_build_session_info(extra_flags, gts);

	gts->kern_params = construct_kern_parambuf(used_params, econtext);
	gts->kern_source = kern_source;
	gts->kern_define = kern_define;
	gts->extra_flags = extra_flags;
}

static void
pgstrom_startup_cuda_program(void)
{
	bool		found;
	int			i;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	pgcache_head = ShmemInitStruct("PG-Strom Program Cache",
								   sizeof(program_cache_head),
								   &found);
	if (found)
		elog(ERROR, "Bug? shared memory for program cache already exists");

	/* initialize program cache header */
	memset(pgcache_head, 0, sizeof(program_cache_head));
	SpinLockInit(&pgcache_head->lock);
	for (i=0; i < PGCACHE_HASH_SIZE; i++)
	{
		dlist_init(&pgcache_head->pgid_slots[i]);
		dlist_init(&pgcache_head->hash_slots[i]);
	}
	dlist_init(&pgcache_head->lru_list);
}

void
pgstrom_init_cuda_program(void)
{
	static int	__program_cache_size;
	int			major;
	int			minor;
	nvrtcResult	rc;

	/*
	 * allocation of shared memory segment size
	 */
	DefineCustomIntVariable("pg_strom.program_cache_size",
							"size of shared program cache",
							NULL,
							&__program_cache_size,
							48 * 1024,		/* 48MB */
							16 * 1024,		/* 16MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	program_cache_size = (Size)__program_cache_size * 1024L;

	/*
	 * Init CUDA run-time compiler library
	 */
	rc = nvrtcVersion(&major, &minor);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcVersion: %s", nvrtcGetErrorString(rc));
	elog(LOG, "NVRTC - CUDA Runtime Compilation vertion %d.%d",
		 major, minor);

	/* allocation of static shared memory */
	RequestAddinShmemSpace(sizeof(program_cache_head));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_program;
}
