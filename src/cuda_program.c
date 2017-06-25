/*
 * cuda_program.c
 *
 * Routines for just-in-time comple cuda code
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "cuda_misc.h"
#include "cuda_timelib.h"
#include "cuda_textlib.h"

typedef struct
{
	dlist_node		pgid_chain;
	dlist_node		hash_chain;
	dlist_node		lru_chain;
	dlist_node		build_chain;
	char		   *extra_buf;		/* extra memory for build/link */
	/* fields below are never updated once entry is constructed */
	ProgramId		program_id;
	pg_crc32		crc;			/* hash value by extra_flags */
	int				extra_flags;	/*             + kern_define */
	char		   *kern_define;	/*             + kern_source */
	char		   *kern_source;
	/* fields above are never updated once entry is constructed */
	cl_int			refcnt;
	Bitmapset	   *waiting_backends;	/* DEPRECATED */
	char		   *bin_image;		/* may be CUDA_PROGRAM_BUILD_FAILURE */
	size_t			bin_length;
	char		   *error_msg;
	int				error_code;
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
	dlist_head	build_list;		/* build pending list */
} program_cache_head;

/* ---- GUC variables ---- */
static Size		program_cache_size;

/* ---- static variables ---- */
static shmem_startup_hook_type shmem_startup_next;
static program_cache_head *pgcache_head = NULL;

static void	   *curand_wrapper_lib = NULL;
static size_t	curand_wrapper_libsz;

#if 1
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
	memset(waiting_backends->words, 0,
		   sizeof(bitmapword) * waiting_backends->nwords);
}
#endif

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
		 * build_cuda_program() don't detach entry from active
		 * entries hash, it never goes to refcnt == 0.
		 */
		Assert(!entry->pgid_chain.next && !entry->pgid_chain.prev);
		Assert(!entry->hash_chain.next && !entry->hash_chain.prev);
		Assert(!entry->lru_chain.next && !entry->lru_chain.prev);
		pgcache_head->program_cache_usage -= dmaBufferChunkSize(entry);
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
construct_flat_cuda_source(uint32 extra_flags,
						   const char *kern_define,
						   const char *kern_source)
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
	/* Common PG-Strom device routine */
	appendStringInfoString(&source, pgstrom_cuda_common_code);

	/* Per session definition if any */
	appendStringInfoString(&source, kern_define);

	/* PG-Strom CUDA device code libraries */

	/* cuRand library */
	if ((extra_flags & DEVKERNEL_NEEDS_CURAND) == DEVKERNEL_NEEDS_CURAND)
		appendStringInfoString(&source, pgstrom_cuda_curand_code);
	/* cuBlas library */
	if ((extra_flags & DEVKERNEL_NEEDS_CUBLAS) == DEVKERNEL_NEEDS_CUBLAS)
		appendStringInfoString(&source, pgstrom_cuda_cublas_code);
	/* cuda dynpara.h */
	if ((extra_flags & DEVKERNEL_NEEDS_DYNPARA) == DEVKERNEL_NEEDS_DYNPARA)
		appendStringInfoString(&source, pgstrom_cuda_dynpara_code);
	/* cuda mathlib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_MATHLIB) == DEVKERNEL_NEEDS_MATHLIB)
		appendStringInfoString(&source, pgstrom_cuda_mathlib_code);
	/* cuda timelib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_TIMELIB) == DEVKERNEL_NEEDS_TIMELIB)
		appendStringInfoString(&source, pgstrom_cuda_timelib_code);
	/* cuda textlib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_TEXTLIB) == DEVKERNEL_NEEDS_TEXTLIB)
		appendStringInfoString(&source, pgstrom_cuda_textlib_code);
	/* cuda numeric.h */
	if ((extra_flags & DEVKERNEL_NEEDS_NUMERIC) == DEVKERNEL_NEEDS_NUMERIC)
		appendStringInfoString(&source, pgstrom_cuda_numeric_code);
	/* cuda money.h */
	if ((extra_flags & DEVKERNEL_NEEDS_MISC) == DEVKERNEL_NEEDS_MISC)
		appendStringInfoString(&source, pgstrom_cuda_misc_code);
	/* cuda matrix.h */
	if ((extra_flags & DEVKERNEL_NEEDS_MATRIX) == DEVKERNEL_NEEDS_MATRIX)
		appendStringInfoString(&source, pgstrom_cuda_matrix_code);
	/* pg_anytype_t declaration */
	appendStringInfoString(
		&source,
		"typedef union {\n"
		"    pg_varlena_t     varlena_v;\n"
		"    pg_bool_t        bool_v;\n"
		"    pg_int2_t        int2_v;\n"
		"    pg_int4_t        int4_v;\n"
		"    pg_int8_t        int8_v;\n"
		"    pg_float4_t      float4_v;\n"
		"    pg_float8_t      float8_v;\n"
		"#ifdef CUDA_NUMERIC_H\n"
		"    pg_numeric_t     numeric_v;\n"
		"#endif\n"
		"#ifdef CUDA_MISC_H\n"
		"    pg_money_t       money_v;\n"
		"#endif\n"
		"#ifdef CUDA_TIMELIB_H\n"
		"    pg_date_t        date_v;\n"
		"    pg_time_t        time_v;\n"
		"    pg_timestamp_t   timestamp_v;\n"
		"    pg_timestamptz_t timestamptz_v;\n"
		"#endif\n"
		"#ifdef CUDA_TEXTLIB_H\n"
		"    pg_bpchar_t      bpchar_v;\n"
		"    pg_text_t        text_v;\n"
		"    pg_varchar_t     varchar_v;\n"
		"#endif\n"
		"  } pg_anytype_t;\n\n");

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
#ifdef NOT_USED
	/* GpuSort */
	if (extra_flags & DEVKERNEL_NEEDS_GPUSORT)
		appendStringInfoString(&source, pgstrom_cuda_gpusort_code);
#endif
	/* PL/CUDA functions */
	if (extra_flags & DEVKERNEL_NEEDS_PLCUDA)
		appendStringInfoString(&source, pgstrom_cuda_plcuda_code);

	/* Source code generated on the fly */
	appendStringInfoString(&source, kern_source);

	/* Source code to fix up undefined type/functions */
	appendStringInfoString(&source, pgstrom_cuda_terminal_code);

	return source.data;
}

/*
 * link_cuda_libraries - links CUDA libraries with the supplied PTX binary
 */
static void
link_cuda_libraries(char *ptx_image, size_t ptx_length, cl_uint extra_flags,
					void **p_bin_image, size_t *p_bin_length)
{
	CUlinkState		lstate;
	CUresult		rc;
	CUjit_option	jit_options[10];
	void		   *jit_option_values[10];
	int				jit_index = 0;
	void		   *bin_image;
	size_t			bin_length;
	char			pathname[MAXPGPATH];

	/* at least one library has to be specified */
	Assert((extra_flags & DEVKERNEL_NEEDS_LINKAGE) != 0);

	/*
	 * NOTE: cuLinkXXXX() APIs works under a particular CUDA context,
	 * so it must be processed by the process which has a valid CUDA
	 * context; that is GPU intermediation server.
	 */

	/*
	 * JIT Options
	 *
	 * NOTE: Even though CU_JIT_TARGET expects CU_TARGET_COMPUTE_XX is
	 * supplied, it is actually defined as (10 * <major capability> +
	 * <minor capability>), thus it is equivalent to the definition
	 * of pgstrom_baseline_cuda_capability.
	 */
	jit_options[jit_index] = CU_JIT_TARGET;
	jit_option_values[jit_index] = (void *)devComputeCapability;
	jit_index++;

#ifdef PGSTROM_DEBUG
	jit_options[jit_index] = CU_JIT_GENERATE_DEBUG_INFO;
	jit_option_values[jit_index] = (void *)1UL;
	jit_index++;

	jit_options[jit_index] = CU_JIT_GENERATE_LINE_INFO;
	jit_option_values[jit_index] = (void *)1UL;
	jit_index++;
#endif
#if 1
	/* how much effective? */
	jit_options[jit_index] = CU_JIT_CACHE_MODE;
	jit_option_values[jit_index] = (void *)CU_JIT_CACHE_OPTION_CA;
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
	if ((extra_flags & DEVKERNEL_NEEDS_DYNPARA) == DEVKERNEL_NEEDS_DYNPARA)
	{
		snprintf(pathname, sizeof(pathname), "%s/libcudadevrt.a",
				 CUDA_LIBRARY_PATH);
		rc = cuLinkAddFile(lstate, CU_JIT_INPUT_LIBRARY, pathname,
						   0, NULL, NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLinkAddFile(\"%s\"): %s",
				 pathname, errorText(rc));
	}
	/* curand is accessed via wrapper library */
	if ((extra_flags & DEVKERNEL_NEEDS_CURAND) == DEVKERNEL_NEEDS_CURAND)
	{
		rc = cuLinkAddData(lstate, CU_JIT_INPUT_OBJECT,
						   curand_wrapper_lib,
						   curand_wrapper_libsz,
						   "curand",
						   0, NULL, NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLinkAddData: %s", errorText(rc));
	}

	/* libcublas_device.a, if any */
	if ((extra_flags & DEVKERNEL_NEEDS_CUBLAS) == DEVKERNEL_NEEDS_CUBLAS)
	{
		snprintf(pathname, sizeof(pathname), "%s/libcublas_device.a",
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

/*
 * pgstrom_cuda_source_file - write out a CUDA program to temp-file
 */
const char *
pgstrom_cuda_source_file(ProgramId program_id)
{
	program_cache_entry *entry;
	char	   *source;

	SpinLockAcquire(&pgcache_head->lock);
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "ProgramId=%lu not found", program_id);
	}
	get_cuda_program_entry_nolock(entry);
	SpinLockRelease(&pgcache_head->lock);

	source = construct_flat_cuda_source(entry->extra_flags,
										entry->kern_define,
										entry->kern_source);

	put_cuda_program_entry(entry);

	return writeout_cuda_source_file(source);
}

/*
 * build_cuda_program - an interface to run synchronous build process
 */
static void
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
	source = construct_flat_cuda_source(entry->extra_flags,
										entry->kern_define,
										entry->kern_source);
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
//  NOTE: we may need to add GCC default include path when we link
//        actual cuRAND library....
//	options[opt_index++] = "-I /usr/lib/gcc/x86_64-redhat-linux/4.8.5/include";
	options[opt_index++] =
		psprintf("--gpu-architecture=compute_%lu",
				 devComputeCapability);
#ifdef PGSTROM_DEBUG
	options[opt_index++] = "--device-debug";
	options[opt_index++] = "--generate-line-info";
#endif
	options[opt_index++] = "--use_fast_math";
//	options[opt_index++] = "--device-as-default-execution-space";
	/* library linkage needs relocatable PTX */
	if (entry->extra_flags & DEVKERNEL_NEEDS_LINKAGE)
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
		if (entry->extra_flags & DEVKERNEL_NEEDS_LINKAGE)
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
#if 1
	/* wake up backends which wait for build */
	pgstrom_wakeup_backends(entry->waiting_backends);
#endif
	/*
	 * Reclaim the older buffer if overconsumption. Also note that this
	 * entry might be already reclaimed by others.
	 */
	if (entry->lru_chain.prev && entry->lru_chain.next)
	{
		dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);
		if (pgcache_head->program_cache_usage > program_cache_size)
			reclaim_cuda_program_entry();
	}
	SpinLockRelease(&pgcache_head->lock);

	/* release nvrtcProgram object */
	rc = nvrtcDestroyProgram(&program);
	if (rc != NVRTC_SUCCESS)
   		elog(WARNING, "failed on nvrtcDestroyProgram: %s",
			 nvrtcGetErrorString(rc));
}	

static void
build_cuda_program(program_cache_entry *entry)
{
	static MemoryContext memcxt = NULL;
	MemoryContext	oldcxt;
	bool			wakeup_done = false;

	/* memory context during the code build (to be reused) */
	if (!memcxt)
	{
		memcxt = AllocSetContextCreate(CurrentMemoryContext,
									   "CUDA program build context",
									   ALLOCSET_DEFAULT_MINSIZE,
									   ALLOCSET_DEFAULT_INITSIZE,
									   ALLOCSET_DEFAULT_MAXSIZE);
	}

	oldcxt = MemoryContextSwitchTo(memcxt);
	PG_TRY();
	{
		__build_cuda_program(entry);
	}
	PG_CATCH();
	{
		ErrorData  *errdata;

		MemoryContextSwitchTo(memcxt);
		errdata = CopyErrorData();

		/* put build error log and wakeup other processes */
		SpinLockAcquire(&pgcache_head->lock);
		if (!entry->bin_image)
		{
			entry->error_code = errdata->sqlerrcode;
			snprintf(entry->error_msg, PGCACHE_ERRORMSG_LEN(entry),
					 "(%s:%d, %s) %s",
					 errdata->filename,
					 errdata->lineno,
					 errdata->funcname,
					 errdata->message);
			entry->bin_image = CUDA_PROGRAM_BUILD_FAILURE;
		}
		/* wake up processes which wait for this program */
		pgstrom_wakeup_backends(entry->waiting_backends);
		wakeup_done = true;
		SpinLockRelease(&pgcache_head->lock);
		/* revert the error status */
		FreeErrorData(errdata);
		FlushErrorState();
	}
	PG_END_TRY();

	/* reset memory context */
	MemoryContextSwitchTo(oldcxt);
	MemoryContextReset(memcxt);

	/* wake up processes which wait for this program */
	if (!wakeup_done)
	{
		SpinLockAcquire(&pgcache_head->lock);
		pgstrom_wakeup_backends(entry->waiting_backends);
		SpinLockRelease(&pgcache_head->lock);
	}
}

/*
 * pgstrom_try_build_cuda_program
 *
 * It picks up a program entry that is not built yet, if any, then runs
 * NVRTC and linker to construct an executable binary.
 * Because linker process needs a valid CUDA context, only GPU server
 * can call this function.
 */
bool
pgstrom_try_build_cuda_program(void)
{
	dlist_node	   *dnode;
	program_cache_entry *entry;

	/* Is there any pending CUDA program? */
	SpinLockAcquire(&pgcache_head->lock);
	if (dlist_is_empty(&pgcache_head->build_list))
	{
		SpinLockRelease(&pgcache_head->lock);
		return false;		/* no programs were built */
	}
	dnode = dlist_pop_head_node(&pgcache_head->build_list);
	entry = dlist_container(program_cache_entry, build_chain, dnode);

	/*
	 * !bin_image && build_chain==0 means build is in-progress,
	 * so it can block concurrent program build any more
	 */
	memset(&entry->build_chain, 0, sizeof(dlist_node));
	Assert(!entry->bin_image);	/* must be build in-progress */
	get_cuda_program_entry_nolock(entry);
	SpinLockRelease(&pgcache_head->lock);

	build_cuda_program(entry);

	put_cuda_program_entry(entry);

	return true;
}

/*
 * wait_for_build_cuda_program
 */
static void
wait_for_build_cuda_program(ProgramId program_id)
{
	for (;;)
	{
		program_cache_entry *entry;
		int		events;

		CHECK_FOR_INTERRUPTS();
		ResetLatch(MyLatch);
		SpinLockAcquire(&pgcache_head->lock);
		entry = lookup_cuda_program_entry_nolock(program_id);
		if (!entry)
		{
			SpinLockRelease(&pgcache_head->lock);
			elog(ERROR, "ProgramId=%lu is missing", program_id);
		}
		else if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
		{
			SpinLockRelease(&pgcache_head->lock);
			ereport(ERROR,
					(errcode(entry->error_code),
					 errmsg("%s", entry->error_msg)));
		}
		else if (entry->bin_image != NULL)
		{
			SpinLockRelease(&pgcache_head->lock);
			break;		/* ok, it is successfully built */
		}
		SpinLockRelease(&pgcache_head->lock);

		events = WaitLatch(MyLatch,
						   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
						   300 * 1000);	/* 5min */
		if (events & WL_POSTMASTER_DEATH)
			ereport(FATAL,
					(errcode(ERRCODE_ADMIN_SHUTDOWN),
					 errmsg("Urgent termination by postmaster dead")));
		if (events & WL_TIMEOUT)
			elog(ERROR, "CUDA code build timeout");
	}
}

/*
 * pgstrom_create_cuda_program
 *
 * It makes a new GPU program cache entry, or acquires an existing entry if
 * equivalent one is already exists.
 */
ProgramId
pgstrom_create_cuda_program(GpuContext_v2 *gcontext,
							cl_uint extra_flags,
							const char *kern_source,
							const char *kern_define,
							bool wait_for_build)
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

	/* Only backend server can create CUDA program entry */
	Assert(!IsGpuServerProcess());

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
				ereport(ERROR,
						(errcode(entry->error_code),
						 errmsg("%s", entry->error_msg)));
			}

			/*
			 * Kernel build is still in-progress, so register myself
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
				trackCudaProgram(gcontext, program_id);
			}
			PG_CATCH();
			{
				pgstrom_put_cuda_program(NULL, program_id);
				PG_RE_THROW();
			}
			PG_END_TRY();

			/* Wait for completion of the GPU code build by NVRTC. */
			if (wait_for_build)
				wait_for_build_cuda_program(program_id);
			return program_id;
		}
	}

	/*
	 * Not found on the existing program cache.
	 * So, create a new entry then kick NVRTC
	 */
	PG_TRY();
	{
		uint32		totalProcs;		/* see InitProcGlobal() */

		totalProcs = MaxBackends + NUM_AUXILIARY_PROCS + max_prepared_xacts;
		required = offsetof(program_cache_entry, data[0]);
		nwords = (totalProcs + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
		required += MAXALIGN(offsetof(Bitmapset, words[nwords]));
		required += MAXALIGN(kern_source_len + 1);
		required += MAXALIGN(kern_define_len + 1);
		required += PGCACHE_MIN_ERRORMSG_BUFSIZE;
		usage = 0;

		entry = dmaBufferAlloc(MasterGpuContext(), required);

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
		usage += MAXALIGN(offsetof(Bitmapset, words[nwords]));;

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
		dlist_push_head(&pgcache_head->build_list,
						&entry->build_chain);
		pgcache_head->program_cache_usage += dmaBufferChunkSize(entry);
		if (pgcache_head->program_cache_usage > program_cache_size)
			reclaim_cuda_program_entry();

		/* try to wake up a GPU server process (likely inactive) */
		gpuservTryToWakeUp();
	}
	PG_CATCH();
	{
		SpinLockRelease(&pgcache_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&pgcache_head->lock);

	/*
	 * Also, track this new entry locally
	 */
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

	if (wait_for_build)
		wait_for_build_cuda_program(program_id);

	return program_id;
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
 * pgstrom_build_session_info
 *
 * it build a session specific code. if extra_flags contains a particular
 * custom_scan node related GPU routine, GpuTaskState must be provided.
 */
char *
pgstrom_build_session_info(cl_uint extra_flags,
						   GpuTaskState_v2 *gts)
{
	StringInfoData	buf;

	initStringInfo(&buf);

	/* OID declaration of types */
	pgstrom_codegen_typeoid_declarations(&buf);
	/* put timezone info */
	if ((extra_flags & DEVKERNEL_NEEDS_TIMELIB) != 0)
		assign_timelib_session_info(&buf);
	/* put currency info */
	if ((extra_flags & DEVKERNEL_NEEDS_MISC) != 0)
		assign_misclib_session_info(&buf);
	/* put text/string info */
	if ((extra_flags & DEVKERNEL_NEEDS_TEXTLIB) != 0)
		assign_textlib_session_info(&buf);

	/* enables device projection? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUSCAN) != 0)
		assign_gpuscan_session_info(&buf, gts);
	/* enables device projection? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
		assign_gpujoin_session_info(&buf, gts);
	/* enables outer-quals evaluation? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUPREAGG) != 0)
		assign_gpupreagg_session_info(&buf, gts);
#ifdef NOT_USED
	/* enables device projection? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUSORT) != 0)
		assign_gpusort_session_info(&buf, gts);
#endif
	return buf.data;
}

/*
 * pgstrom_load_cuda_program
 *
 *
 */
CUmodule
pgstrom_load_cuda_program(ProgramId program_id, long timeout)
{
	program_cache_entry *entry;
	CUmodule	cuda_module;
	CUresult	rc;
	char	   *bin_image;

	Assert(IsGpuServerProcess());

	SpinLockAcquire(&pgcache_head->lock);
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "CUDA Program ID=%lu was not found", program_id);
	}

	if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
	{
		const char *error_msg = entry->error_msg;

		SpinLockRelease(&pgcache_head->lock);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_FUNCTION_DEFINITION),
				 errmsg("CUDA Program ID=%lu build failed:\n%s",
						program_id, error_msg)));
	}
	get_cuda_program_entry_nolock(entry);

	if (!entry->bin_image)
	{
		if (entry->build_chain.prev || entry->build_chain.next)
		{
			/*
			 * It looks to me nobody picked up this CUDA program for build,
			 * so we try to build by ourself, synchronously.
			 */
			SpinLockRelease(&pgcache_head->lock);

			build_cuda_program(entry);

			SpinLockAcquire(&pgcache_head->lock);
			Assert(entry->bin_image != NULL);
			if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
			{
				const char *error_msg = entry->error_msg;

				put_cuda_program_entry_nolock(entry);
				SpinLockRelease(&pgcache_head->lock);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_FUNCTION_DEFINITION),
						 errmsg("CUDA Program (id=%lu) build failed:\n%s",
								program_id, error_msg)));
			}
		}
		else
		{
			struct timeval	tv1, tv2;
			int				ev;

			/*
			 * It looks to me somebody else already picks up this CUDA
			 * program, and then kicked build process but not finished.
			 */
			gettimeofday(&tv1, NULL);
			for (;;)
			{
				/* register myself on the waiter list */
				ResetLatch(MyLatch);
				entry->waiting_backends->words[WORDNUM(MyProc->pgprocno)]
					|= (1 << BITNUM(MyProc->pgprocno));

				if (timeout == 0)
				{
					put_cuda_program_entry_nolock(entry);
					SpinLockRelease(&pgcache_head->lock);
					return NULL;
				}
				SpinLockRelease(&pgcache_head->lock);

				ev = WaitLatch(MyLatch,
							   WL_LATCH_SET |
							   WL_POSTMASTER_DEATH |
							   (timeout < 0 ? 0 : WL_TIMEOUT),
							   timeout);
				/* emergency bailout if postmaster is dead */
				if (ev & WL_POSTMASTER_DEATH)
					ereport(FATAL,
							(errcode(ERRCODE_ADMIN_SHUTDOWN),
							 errmsg("Urgent termination by postmaster dead")));
				if (ev & WL_TIMEOUT)
					timeout = 0;	/* no wait any more */
				else
				{
					gettimeofday(&tv2, NULL);
					if (timeout >= 0)
					{
						timeout -= ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
									(tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
						timeout = Max(timeout, 0);
						tv1 = tv2;
					}
				}

				/* check status of the entry->bin_image */
				SpinLockAcquire(&pgcache_head->lock);
				if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
				{
					const char *error_msg = entry->error_msg;

					put_cuda_program_entry_nolock(entry);
					SpinLockRelease(&pgcache_head->lock);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_FUNCTION_DEFINITION),
							 errmsg("CUDA Program (id=%lu) build failed:\n%s",
									program_id, error_msg)));
				}
				else if (entry->bin_image)
					break;
			}
		}
	}
	bin_image = entry->bin_image;
	SpinLockRelease(&pgcache_head->lock);

	rc = cuModuleLoadData(&cuda_module, bin_image);
	if (rc != CUDA_SUCCESS)
	{
		put_cuda_program_entry(entry);
		elog(ERROR, "failed on cuModuleLoadData: %s", errorText(rc));
	}
	return cuda_module;
}

static void
build_wrapper_libraries(const void *wrapper_source,
						void **p_wrapper_lib,
						size_t *p_wrapper_libsz)
{
	char   *src_fname = NULL;
	char   *lib_fname = NULL;
	int		fdesc = -1;
	int		status;
	char	spath[128];
	char	lpath[128];
	char	cmd[MAXPGPATH];
	void   *wrapper_lib = NULL;
	ssize_t	rv, source_len = strlen(wrapper_source);
	struct stat st_buf;

	PG_TRY();
	{
		/* Write source */
		strcpy(spath, P_tmpdir "/XXXXXX.cu");
		fdesc = mkstemps(spath, 3);
		if (fdesc < 0)
			elog(ERROR, "failed on mkstemps('%s') : %m", src_fname);
		src_fname = spath;

		rv = write(fdesc, wrapper_source, source_len);
		if (rv != source_len)
			elog(ERROR, "failed on write(2) on '%s': %m", src_fname);
		close(fdesc);
		fdesc = -1;

		/* Run NVCC */
		snprintf(lpath, sizeof(lpath),
				 "%s.sm_%lu.o",
				 spath, devComputeCapability);
		lib_fname = lpath;
		snprintf(cmd, sizeof(cmd),
				 CUDA_BINARY_PATH "/nvcc "
				 " --relocatable-device-code=true"
				 " --gpu-architecture=sm_%lu"
				 " -DPGSTROM_BUILD_WRAPPER"
				 " --device-c %s -o %s",
				 devComputeCapability,
				 src_fname,
				 lib_fname);
		status = system(cmd);
		if (status < 0 || WEXITSTATUS(status) != 0)
			elog(ERROR, "failed on nvcc (%s)", cmd);

		/* Read library */
		fdesc = open(lib_fname, O_RDONLY);
		if (fdesc < 0)
			elog(ERROR, "failed to open \"%s\": %m", lib_fname);
		if (fstat(fdesc, &st_buf) != 0)
			elog(ERROR, "failed on fstat(\"%s\") : %m", lib_fname);

		wrapper_lib = malloc(st_buf.st_size);
		if (!wrapper_lib)
			elog(ERROR, "out of memory");
		rv = read(fdesc, wrapper_lib, st_buf.st_size);
		if (rv != st_buf.st_size)
			elog(ERROR, "failed on read(\"%s\") : %m", lib_fname);
		close(fdesc);
		fdesc = -1;
	}
	PG_CATCH();
	{
		if (wrapper_lib)
			free(wrapper_lib);
		if (fdesc >= 0)
			close(fdesc);
		if (src_fname)
			unlink(src_fname);
		if (lib_fname)
			unlink(lib_fname);
		PG_RE_THROW();
	}
	PG_END_TRY();

	unlink(src_fname);
	unlink(lib_fname);

	*p_wrapper_lib = wrapper_lib;
	*p_wrapper_libsz = st_buf.st_size;
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
	dlist_init(&pgcache_head->build_list);
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

	/* build wrapper library objects */
	build_wrapper_libraries(pgstrom_cuda_curand_code,
							&curand_wrapper_lib,
							&curand_wrapper_libsz);
}
