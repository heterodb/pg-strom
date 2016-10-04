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
	dlist_node		hash_chain;
	dlist_node		lru_chain;
	int				shift;	/* block class of this entry */
	int				refcnt;	/* 0 means free entry */
	pg_crc32		crc;	/* hash value by extra_flags + kern_source */
	struct timeval	tv_build_end;	/* timestamp when build end */
	Bitmapset	   *waiting_backends;
	Oid				database_oid;
	Oid				user_oid;
	int				extra_flags;
	char		   *kern_define;
	char		   *kern_source;
	char		   *bin_image;
	size_t			bin_length;
	char		   *error_msg;
	char			data[FLEXIBLE_ARRAY_MEMBER];
} program_cache_entry;

#define PGCACHE_ACTIVE_ENTRY(entry)				\
	((entry)->lru_chain.prev && (entry)->lru_chain.next)
#define PGCACHE_FREE_ENTRY(entry)				\
	(!(entry)->lru_chain.prev && !(entry)->lru_chain.next)
#define PGCACHE_MAGIC					0xabadcafe
#define PGCACHE_MAGIC_CODE(entry)				\
	*((cl_uint *)((char *)(entry) + (1UL << (entry)->shift) - sizeof(cl_uint)))
#define PGCACHE_CHECK_ACTIVE(entry)				\
	Assert(PGCACHE_ACTIVE_ENTRY(entry) &&		\
		   (entry)->refcnt > 0 &&				\
		   PGCACHE_MAGIC_CODE(entry) == PGCACHE_MAGIC)
#define PGCACHE_CHECK_FREE(entry)				\
	Assert(PGCACHE_FREE_ENTRY(entry) &&			\
		   (entry)->refcnt == 0 &&				\
		   PGCACHE_MAGIC_CODE(entry) == PGCACHE_MAGIC)
#define PGCACHE_MIN_ERRORMSG_BUFSIZE	256
#define PGCACHE_ERRORMSG_LEN(entry)				\
	((uintptr_t)(entry) +						\
	 (1UL << (entry)->shift) -					\
	 sizeof(uint) -								\
	 (uintptr_t)(entry)->error_msg)
#define CUDA_PROGRAM_BUILD_FAILURE			((void *)(~0UL))

#define PGCACHE_MIN_BITS		10		/* 1KB */
#define PGCACHE_MAX_BITS		24		/* 16MB */	
#define PGCACHE_HASH_SIZE		1024

#define WORDNUM(x)		((x) / BITS_PER_BITMAPWORD)
#define BITNUM(x)		((x) % BITS_PER_BITMAPWORD)

typedef struct
{
	volatile slock_t lock;
	dlist_head	free_list[PGCACHE_MAX_BITS + 1];
	dlist_head	active_list[PGCACHE_HASH_SIZE];
	dlist_head	lru_list;
	program_cache_entry *entry_begin;	/* start address of entries */
	program_cache_entry *entry_end;		/* end address of entries */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} program_cache_head;

/* ---- GUC variables ---- */
static Size		program_cache_size;
static bool		pgstrom_enable_cuda_coredump;

/* ---- static variables ---- */
static shmem_startup_hook_type shmem_startup_next;
static program_cache_head *pgcache_head = NULL;

/* ---- static functions ---- */
static program_cache_entry *pgstrom_program_cache_alloc(Size required);
static void pgstrom_program_cache_free(program_cache_entry *entry);

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
	bitmapword		bitmap;
	int				i, j;

	for (i=0; i < waiting_backends->nwords; i++)
	{
		bitmap = waiting_backends->words[i];
		if (!bitmap)
			continue;

		for (j=0; j < BITS_PER_BITMAPWORD; j++)
		{
			if ((bitmap & (1 << j)) == 0)
				continue;
			Assert(i * BITS_PER_BITMAPWORD + j < ProcGlobal->allProcCount);
			proc = &ProcGlobal->allProcs[i * BITS_PER_BITMAPWORD + j];
			SetLatch(&proc->procLatch);
		}
	}
}

/*
 * pgstrom_program_cache_reclaim
 *
 * it tries to reclaim the shared memory if highly memory presure.
 */
static bool
pgstrom_program_cache_reclaim(int shift_min)
{
	program_cache_entry *entry;

	while (!dlist_is_empty(&pgcache_head->lru_list))
	{
		dlist_node *dnode = dlist_tail_node(&pgcache_head->lru_list);
		int			shift;

		entry = dlist_container(program_cache_entry, lru_chain, dnode);
		PGCACHE_CHECK_ACTIVE(entry);
		/* remove from the list not to be reclaimed again */
		dlist_delete(&entry->hash_chain);
		dlist_delete(&entry->lru_chain);
		memset(&entry->hash_chain, 0, sizeof(dlist_node));
		memset(&entry->lru_chain, 0, sizeof(dlist_node));

		if (--entry->refcnt == 0)
		{
			pgstrom_program_cache_free(entry);

			/* check whether the required size is allocatable */
			for (shift = shift_min; shift <= PGCACHE_MAX_BITS; shift++)
			{
				if (!dlist_is_empty(&pgcache_head->free_list[shift]))
					return true;
			}
		}
	}
	return false;
}

/*
 * pgstrom_program_cache_*
 *
 * a simple buddy memory allocation on the shared memory segment.
 */
static bool
pgstrom_program_cache_split(int shift)
{
	program_cache_entry *entry;
	dlist_node	   *dnode;

	Assert(shift > PGCACHE_MIN_BITS && shift <= PGCACHE_MAX_BITS);
	if (dlist_is_empty(&pgcache_head->free_list[shift]))
	{
		if (shift == PGCACHE_MAX_BITS ||
			!pgstrom_program_cache_split(shift + 1))
			return false;
	}
	Assert(!dlist_is_empty(&pgcache_head->free_list[shift]));

	dnode = dlist_pop_head_node(&pgcache_head->free_list[shift]);

	entry = dlist_container(program_cache_entry, hash_chain, dnode);
	Assert(entry->shift == shift);
	Assert((((uintptr_t)entry - (uintptr_t)pgcache_head->entry_begin)
			& ((1UL << shift) - 1)) == 0);
	shift--;

	/* earlier half */
	memset(entry, 0, offsetof(program_cache_entry, data[0]));
	entry->shift = shift;
	entry->refcnt = 0;
	PGCACHE_MAGIC_CODE(entry) = PGCACHE_MAGIC;
	dlist_push_tail(&pgcache_head->free_list[shift], &entry->hash_chain);

	/* later half */
	entry = (program_cache_entry *)((char *)entry + (1UL << shift));
	memset(entry, 0, offsetof(program_cache_entry, data[0]));
	entry->shift = shift;
	entry->refcnt = 0;
	PGCACHE_MAGIC_CODE(entry) = PGCACHE_MAGIC;
	dlist_push_tail(&pgcache_head->free_list[shift], &entry->hash_chain);

	return true;
}

static program_cache_entry *
pgstrom_program_cache_alloc(Size required)
{
	program_cache_entry *entry;
	dlist_node *dnode;
	Size		total_size;
	int			shift;

	total_size = offsetof(program_cache_entry, data[0])
		+ MAXALIGN(required)
		+ PGCACHE_MIN_ERRORMSG_BUFSIZE
		+ sizeof(cl_uint);

	/* required size too large? */
	if (total_size > (1UL << PGCACHE_MAX_BITS))
		return NULL;

	shift = get_next_log2(total_size);
	if (shift < PGCACHE_MIN_BITS)
		shift = PGCACHE_MIN_BITS;

	if (dlist_is_empty(&pgcache_head->free_list[shift]))
	{
		/*
		 * If no entries are free in the suitable class,
		 * we try to split larger blocks first, then try
		 * to reclaim entries according to LRU.
		 * If both of them make no sense, we give up!
		 */
		while (!pgstrom_program_cache_split(shift + 1))
		{
			if (!pgstrom_program_cache_reclaim(shift))
				return NULL;
		}
	}
	Assert(!dlist_is_empty(&pgcache_head->free_list[shift]));

	dnode = dlist_pop_head_node(&pgcache_head->free_list[shift]);
	entry = dlist_container(program_cache_entry, hash_chain, dnode);
	Assert(entry->shift == shift);

	memset(entry, 0, sizeof(program_cache_entry));
	entry->shift = shift;
	entry->refcnt = 1;
	PGCACHE_MAGIC_CODE(entry) = PGCACHE_MAGIC;

	return entry;
}

static void
pgstrom_program_cache_free(program_cache_entry *entry)
{
	int			shift = entry->shift;
	Size		offset;

	Assert(entry->refcnt == 0);
	Assert(!entry->hash_chain.next && !entry->hash_chain.prev);
	Assert(!entry->lru_chain.next && !entry->lru_chain.prev);

	offset = (uintptr_t)entry - (uintptr_t)pgcache_head->entry_begin;
	Assert((offset & ((1UL << shift) - 1)) == 0);

	/* try to merge buddy entry, if it is also free */
	while (shift < PGCACHE_MAX_BITS)
	{
		program_cache_entry *buddy;

		offset = (uintptr_t) entry - (uintptr_t)pgcache_head->entry_begin;
		if ((offset & (1UL << shift)) == 0)
			buddy = (program_cache_entry *)((char *)entry + (1UL << shift));
		else
			buddy = (program_cache_entry *)((char *)entry - (1UL << shift));

		if (buddy >= pgcache_head->entry_end ||		/* out of range? */
			buddy->shift != shift ||				/* same size? */
			PGCACHE_ACTIVE_ENTRY(buddy))			/* and free entry? */
			break;
#ifdef NOT_USED
		/* Sanity check - buddy should be in free list */
		do {
			dlist_head	   *free_list = pgcache_head->free_list + shift;
			dlist_iter		iter;
			bool			found = false;

			dlist_foreach (iter, free_list)
			{
				program_cache_entry *temp
					= dlist_container(program_cache_entry,
									  hash_chain, iter.cur);
				if (temp == buddy)
				{
					found = true;
					break;
				}
			}
			Assert(found);
		} while(0);
#endif
		/* OK, chunk and buddy can be merged */
		PGCACHE_CHECK_FREE(buddy);
		dlist_delete(&buddy->hash_chain);	/* remove from free_list */
		memset(&buddy->hash_chain, 0, sizeof(dlist_node));
		if (buddy < entry)
			entry = buddy;
		entry->shift = ++shift;
		PGCACHE_MAGIC_CODE(entry) = PGCACHE_MAGIC;
	}
	PGCACHE_CHECK_FREE(entry);
	dlist_push_head(&pgcache_head->free_list[shift], &entry->hash_chain);
}

static void
pgstrom_put_cuda_program(program_cache_entry *entry)
{
	SpinLockAcquire(&pgcache_head->lock);
	if (--entry->refcnt == 0)
	{
		/*
		 * NOTE: unless either pgstrom_program_cache_reclaim() or
		 * __build_cuda_program() don't detach entry from active
		 * entries hash, it never goes to refcnt == 0.
		 */
		Assert(!entry->hash_chain.next && !entry->hash_chain.prev);
		Assert(!entry->lru_chain.next && !entry->lru_chain.prev);
		pgstrom_program_cache_free(entry);
	}
	SpinLockRelease(&pgcache_head->lock);
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

static void
__build_cuda_program(program_cache_entry *old_entry)
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
	Size			usage;
	int				hindex;
	bool			build_failure = false;
	program_cache_entry *new_entry;

	/*
	 * Make a nvrtcProgram object
	 */
	source = construct_flat_cuda_source(old_entry->kern_source,
										old_entry->kern_define,
										old_entry->extra_flags);
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
	if (old_entry->extra_flags & DEVKERNEL_NEEDS_DYNPARA)
		options[opt_index++] = "--relocatable-device-code=true";

	/*
	 * Kick runtime compiler
	 */
	rc = nvrtcCompileProgram(program, opt_index, options);
	if (rc != NVRTC_SUCCESS)
	{
		if (rc == NVRTC_ERROR_COMPILATION)
			build_failure = true;
		else
			elog(ERROR, "failed on nvrtcCompileProgram: %s",
				 nvrtcGetErrorString(rc));
	}

	/*
	 * Save the source file, if required or build failure
	 */
	if (build_failure)
		source_pathname = writeout_cuda_source_file(source);

	/*
	 * Read PTX Binary
	 */
	if (build_failure)
	{
		bin_image = NULL;
		bin_length = 0;
	}
	else
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
		if (old_entry->extra_flags & DEVKERNEL_NEEDS_DYNPARA)
		{
			link_cuda_libraries(ptx_image, ptx_length,
								old_entry->extra_flags,
								&bin_image, &bin_length);
			pfree(ptx_image);
		}
		else
		{
			bin_image = ptx_image;
			bin_length = ptx_length;
		}
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
	 * Make a new entry, instead of the old one
	 */
	required = MAXALIGN(strlen(old_entry->kern_source) + 1);
	required += MAXALIGN(strlen(old_entry->kern_define) + 1);
	if (bin_image)
		required += MAXALIGN(bin_length);
	required += MAXALIGN(strlen(build_log) + 1);
	required += 512;	/* margin for error message */

	SpinLockAcquire(&pgcache_head->lock);
	new_entry = pgstrom_program_cache_alloc(required);
	if (!new_entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "out of shared memory");
	}
	usage = 0;
	new_entry->crc = old_entry->crc;
	new_entry->waiting_backends = NULL;		/* no need to set latch */
	new_entry->database_oid = old_entry->database_oid;
	new_entry->user_oid = old_entry->user_oid;
	new_entry->extra_flags = old_entry->extra_flags;

	new_entry->kern_source = new_entry->data + usage;
	length = strlen(old_entry->kern_source);
	memcpy(new_entry->kern_source,
		   old_entry->kern_source, length + 1);
	usage += MAXALIGN(length + 1);

	new_entry->kern_define = new_entry->data + usage;
	length = strlen(old_entry->kern_define);
	memcpy(new_entry->kern_define,
		   old_entry->kern_define, length + 1);
	usage += MAXALIGN(length + 1);

	if (!bin_image)
	{
		new_entry->bin_image = CUDA_PROGRAM_BUILD_FAILURE;
		new_entry->bin_length = 0;
	}
	else
	{
		new_entry->bin_image = new_entry->data + usage;
		new_entry->bin_length = bin_length;
		memcpy(new_entry->bin_image, bin_image, bin_length);
		usage += MAXALIGN(bin_length);
	}
	new_entry->error_msg = new_entry->data + usage;
	length = PGCACHE_ERRORMSG_LEN(new_entry);
	if (source_pathname)
		snprintf(new_entry->error_msg, length,
				 "build: %s\n%s\nsource: %s\n",
				 !bin_image ? "failed" : "success",
				 build_log,
				 source_pathname);
	else
		snprintf(new_entry->error_msg, length,
				 "build: %s\n%s",
				 !bin_image ? "failed" : "success",
				 build_log);

	/* record timestamp of the build end */
	gettimeofday(&new_entry->tv_build_end, NULL);

	/*
	 * Add new_entry to the hash slot
	 */
	hindex = new_entry->crc % PGCACHE_HASH_SIZE;
	dlist_push_head(&pgcache_head->active_list[hindex],
					&new_entry->hash_chain);
	dlist_push_head(&pgcache_head->lru_list,
					&new_entry->lru_chain);

	/*
	 * Waking up blocking tasks, and detach old_entry from
	 * the hash/lru list to ensure nobody will grab it.
	 */
	pgstrom_wakeup_backends(old_entry->waiting_backends);

	/*
	 * NOTE: In case of buils success or NVRTC_ERROR_COMPILATION, old_entry
	 * shall not be used no longer. The new_entry is already entered, then
	 * we detach old_entry instead. pgstrom_put_cuda_program() will release
	 * shared memory segment.
	 */
	dlist_delete(&old_entry->hash_chain);
	dlist_delete(&old_entry->lru_chain);
	memset(&old_entry->hash_chain, 0, sizeof(dlist_node));
	memset(&old_entry->lru_chain, 0, sizeof(dlist_node));
	SpinLockRelease(&pgcache_head->lock);

	pgstrom_put_cuda_program(old_entry);
}

static void
pgstrom_build_cuda_program(program_cache_entry *entry)
{
	MemoryContext	memcxt = CurrentMemoryContext;
	MemoryContext	oldcxt;

	Assert(entry->bin_image == NULL);

	PG_TRY();
	{
		__build_cuda_program(entry);
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
		SpinLockRelease(&pgcache_head->lock);
		MemoryContextSwitchTo(oldcxt);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

static void
pgstrom_build_cuda_program_bgw_main(Datum cuda_program)
{
	program_cache_entry *entry = (program_cache_entry *) cuda_program;

	/*
	 * TODO: It is more preferable to use ParallelContext on v2.0
	 */
	BackgroundWorkerUnblockSignals();
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
	pgstrom_build_cuda_program(entry);
	CommitTransactionCommand();
}

static CUmodule *
__pgstrom_load_cuda_program(GpuContext *gcontext,
							cl_uint extra_flags,
							const char *kern_source,
							const char *kern_define,
							bool is_preload,
							bool with_async_build,
							struct timeval *tv_build_start,
							struct timeval *tv_build_end)
{
	program_cache_entry	*entry;
	Size			kern_source_len = strlen(kern_source);
	Size			kern_define_len = strlen(kern_define);
	Size			required;
	Size			usage;
	int				nwords;
	int				hindex;
	dlist_iter		iter;
	pg_crc32		crc;
	CUresult		rc;
	CUmodule	   *cuda_modules = NULL;
	int				i, num_context;
	BackgroundWorker worker;

	/* makes a hash value */
	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &extra_flags, sizeof(int32));
	COMP_LEGACY_CRC32(crc, kern_source, kern_source_len);
	FIN_LEGACY_CRC32(crc);

retry:
	hindex = crc % PGCACHE_HASH_SIZE;
	SpinLockAcquire(&pgcache_head->lock);
	dlist_foreach (iter, &pgcache_head->active_list[hindex])
	{
		entry = dlist_container(program_cache_entry, hash_chain, iter.cur);

		if (entry->crc == crc &&
			entry->extra_flags == extra_flags &&
			strcmp(entry->kern_source, kern_source) == 0 &&
			strcmp(entry->kern_define, kern_define) == 0)
		{
			/* Move this entry to the head of LRU list */
			dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);

			/* This kernel build already lead an error */
			if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
			{
				SpinLockRelease(&pgcache_head->lock);
				if (!is_preload)
					elog(ERROR, "%s", entry->error_msg);
				return NULL;
			}
			/* Kernel build is still in-progress */
			if (!entry->bin_image)
			{
				Bitmapset  *waiting_backends = entry->waiting_backends;
				waiting_backends->words[WORDNUM(MyProc->pgprocno)]
					|= (1 << BITNUM(MyProc->pgprocno));
				SpinLockRelease(&pgcache_head->lock);

				/*
				 * NOTE: current timestamp is an alternative of the timestamp
				 * when build start, if somebody concurrent already kicked
				 * the same kernel.
				 */
				if (tv_build_start && tv_build_start->tv_sec == 0)
					gettimeofday(tv_build_start, NULL);
				return NULL;
			}
			/* OK, this kernel is already built */
			Assert(entry->refcnt > 0);
			entry->refcnt++;

			if (tv_build_end && tv_build_end->tv_sec == 0)
				*tv_build_end = entry->tv_build_end;

			SpinLockRelease(&pgcache_head->lock);

			/*
			 * Let's load this module for each context
			 */
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
						elog(ERROR, "failed on cuCtxPushCurrent (%s)",
							 errorText(rc));

					rc = cuModuleLoadData(&cuda_modules[i],
										  entry->bin_image);
					if (rc != CUDA_SUCCESS)
						elog(ERROR, "failed on cuModuleLoadData (%s)\n",
							 errorText(rc));
				}
				rc = cuCtxPopCurrent(NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuCtxPopCurrent (%s)",
						 errorText(rc));
			}
			PG_CATCH();
			{
				while (cuda_modules && i > 0)
				{
					rc = cuModuleUnload(cuda_modules[--i]);
					if (rc != CUDA_SUCCESS)
						elog(WARNING, "failed on cuModuleUnload (%s)",
							 errorText(rc));
				}
				pgstrom_put_cuda_program(entry);
				PG_RE_THROW();
			}
			PG_END_TRY();
			pgstrom_put_cuda_program(entry);

			return cuda_modules;
		}
	}

	/*
	 * Not found on the existing cache.
	 * So, create a new one then kick NVRTC
	 */
	if (tv_build_start && tv_build_start->tv_sec == 0)
		gettimeofday(tv_build_start, NULL);

	required = offsetof(program_cache_entry, data[0]);
	nwords = (ProcGlobal->allProcCount +
			  BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	required += MAXALIGN(offsetof(Bitmapset, words[nwords]));
	required += MAXALIGN(kern_source_len + 1);
	required += MAXALIGN(kern_define_len + 1);
	required += 512;	/* margin for error message */
	usage = 0;

	entry = pgstrom_program_cache_alloc(required);
	entry->crc = crc;
	memset(&entry->tv_build_end, 0, sizeof(struct timeval));
	/* bitmap for waiting backends */
	entry->waiting_backends = (Bitmapset *) entry->data;
	entry->waiting_backends->nwords = nwords;
	memset(entry->waiting_backends->words, 0, sizeof(bitmapword) * nwords);
	usage += MAXALIGN(offsetof(Bitmapset, words[nwords]));
	/* session info who tries to build the program */
	entry->database_oid = MyDatabaseId;
	entry->user_oid = GetUserId();
	/* device kernel source */
	entry->extra_flags = extra_flags;
	entry->kern_source = (char *)(entry->data + usage);
	memcpy(entry->kern_source, kern_source, kern_source_len + 1);
	usage += MAXALIGN(kern_source_len + 1);
	entry->kern_define = (char *)(entry->data + usage);
	memcpy(entry->kern_define, kern_define, kern_define_len + 1);
	usage += MAXALIGN(kern_define_len + 1);
	/* no cuda binary yet */
	entry->bin_image = NULL;
	entry->bin_length = 0;
	/* remaining are for error message */
	entry->error_msg = (char *)(entry->data + usage);

	/* at least, caller is waiting for build */
	entry->waiting_backends = entry->waiting_backends;
	entry->waiting_backends->words[WORDNUM(MyProc->pgprocno)]
		|= (1 << BITNUM(MyProc->pgprocno));

	/* to be acquired by program builder */
	dlist_push_head(&pgcache_head->active_list[hindex], &entry->hash_chain);
	dlist_push_head(&pgcache_head->lru_list, &entry->lru_chain);

	/* Kick a dynamic background worker to build */
	if (with_async_build)
	{
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "nvcc launcher - crc %08x", crc);
		worker.bgw_flags = (BGWORKER_SHMEM_ACCESS |
							BGWORKER_BACKEND_DATABASE_CONNECTION);
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = BGW_NEVER_RESTART;
		worker.bgw_main = pgstrom_build_cuda_program_bgw_main;
		worker.bgw_main_arg = PointerGetDatum(entry);

		if (RegisterDynamicBackgroundWorker(&worker, NULL))
		{
			SpinLockRelease(&pgcache_head->lock);
			return NULL;	/* now bgworker building the device kernel */
		}
		else if (is_preload)
		{
			/*
			 * Revert the new program_cache_entry if no background worker is
			 * available but kernel build as a preload.
			 * @entry->bin_image == NULL means somebody is still in-progress
			 * of the code build, thus other concurrent tasks will wait for
			 * completion. Unless caller does not take this job, we cannot
			 * leave the CUDA program entry.
			 */
			Assert(entry->refcnt == 1);
			dlist_delete(&entry->hash_chain);
			dlist_delete(&entry->lru_chain);
			memset(&entry->hash_chain, 0, sizeof(dlist_node));
			memset(&entry->lru_chain, 0, sizeof(dlist_node));

			pgstrom_program_cache_free(entry);

			SpinLockRelease(&pgcache_head->lock);
			return NULL;
		}
		elog(LOG, "failed to launch async NVRTC build, try sync mode");
	}
	SpinLockRelease(&pgcache_head->lock);
	/* build the device kernel synchronously */
	pgstrom_build_cuda_program(entry);
	goto retry;
}

bool
pgstrom_load_cuda_program(GpuTaskState *gts, bool is_preload)
{
	CUmodule	   *cuda_modules;

	Assert(!gts->cuda_modules);

	cuda_modules = __pgstrom_load_cuda_program(gts->gcontext,
											   gts->extra_flags,
											   gts->kern_source,
											   gts->kern_define,
											   is_preload, true,
											   &gts->pfm.tv_build_start,
											   &gts->pfm.tv_build_end);
	if (cuda_modules)
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
	const char	   *kern_define
		= pgstrom_build_session_info(NULL, kern_source, extra_flags);
	return __pgstrom_load_cuda_program(gcontext,
									   extra_flags,
									   kern_source,
									   kern_define,
									   false, false,
									   NULL, NULL);
}

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
				{
					void   *vl_val = PG_DETOAST_DATUM(con->constvalue);

					appendBinaryStringInfo(&str, vl_val, VARSIZE(vl_val));
				}
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
						void   *vl_val = PG_DETOAST_DATUM(prm->value);

						appendBinaryStringInfo(&str, vl_val, VARSIZE(vl_val));
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
pgstrom_build_session_info(GpuTaskState *gts,
						   const char *kern_source,
						   cl_uint extra_flags)
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
	if ((extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
		assign_gpujoin_session_info(&buf, gts);
	/* enables device projection? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUSORT) != 0)
		assign_gpusort_session_info(&buf, gts);

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
		= pgstrom_build_session_info(gts, kern_source, extra_flags);

	gts->kern_params = construct_kern_parambuf(used_params, econtext);
	gts->kern_source = kern_source;
	gts->kern_define = kern_define;
	gts->extra_flags = extra_flags;
}

/*
 * pgstrom_program_info
 *
 * A SQL function to dump cached CUDA programs
 */
typedef struct
{
	int64		addr;
	int64		length;
	bool		active;
	const char *status;
	int32		crc32;
	int32		flags;
	text	   *kern_define;
	text	   *kern_source;
	bytea	   *kern_binary;
	text	   *error_msg;
	text	   *backends;
} program_info;

static List *
__collect_program_info(void)
{
	program_cache_entry *entry = pgcache_head->entry_begin;
	List	   *results = NIL;

	while (entry < pgcache_head->entry_end)
	{
		program_info   *pinfo = palloc0(sizeof(program_info));

		pinfo->addr = (int64) entry;
		pinfo->length = (1UL << entry->shift);
		pinfo->active = PGCACHE_ACTIVE_ENTRY(entry);
		if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
			pinfo->status = "Build Failed";
		else if (!entry->bin_image)
			pinfo->status = "In Progress";
		else
			pinfo->status = "Ready";
		pinfo->crc32 = entry->crc;
		pinfo->flags = entry->extra_flags;
		if (entry->kern_define)
			pinfo->kern_define = cstring_to_text(entry->kern_define);
		if (entry->kern_source)
			pinfo->kern_source = cstring_to_text(entry->kern_source);
		if (entry->bin_image != NULL &&
			entry->bin_image != CUDA_PROGRAM_BUILD_FAILURE)
			pinfo->kern_binary = (bytea *)
				cstring_to_text_with_len(entry->bin_image,
										 entry->bin_length);
		if (entry->error_msg)
			pinfo->error_msg = cstring_to_text(entry->error_msg);
		if (entry->waiting_backends)
		{
			StringInfoData	buf;
			struct PGPROC  *proc;
			int				i = -1;

			initStringInfo(&buf);
			while ((i = bms_next_member(entry->waiting_backends, i)) >= 0)
			{
				Assert(i < ProcGlobal->allProcCount);
				proc = &ProcGlobal->allProcs[i];

				if (buf.len > 0)
					appendStringInfo(&buf, ", ");
				appendStringInfo(&buf, "%d (pid: %u)",
								 proc->backendId, proc->pid);
			}
			if (buf.len > 0)
				pinfo->backends = cstring_to_text(buf.data);
			pfree(buf.data);
		}
		results = lappend(results, pinfo);

		/* next entry */
		entry = (program_cache_entry *)((char *)entry + (1UL << entry->shift));
	}
	return results;
}

static List *
collect_program_info(void)
{
	List   *results;

	SpinLockAcquire(&pgcache_head->lock);
	PG_TRY();
	{
		results = __collect_program_info();
	}
	PG_CATCH();
	{
		SpinLockRelease(&pgcache_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&pgcache_head->lock);

	return results;
}

Datum
pgstrom_program_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *fncxt;
	program_info   *pinfo;
	List		   *pinfo_list;
	Datum			values[11];
	bool			isnull[11];
	HeapTuple		tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(11, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "addr",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "length",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "active",
						   BOOLOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "status",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "crc32",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "flags",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 7, "kern_define",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 8, "kern_source",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 9, "kern_binary",
						   BYTEAOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 10, "error_msg",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 11, "backends",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);
		fncxt->user_fctx = collect_program_info();

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	/* fetch the first entry */
	pinfo_list = fncxt->user_fctx;
	if (pinfo_list == NIL)
		SRF_RETURN_DONE(fncxt);
	pinfo = linitial(pinfo_list);
	fncxt->user_fctx = list_delete_first(pinfo_list);

	/* make a heap-tuple */
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int64GetDatum(pinfo->addr);
	values[1] = Int64GetDatum(pinfo->length);
	values[2] = BoolGetDatum(pinfo->active);
	if (!pinfo->active)
	{
		isnull[3] = true;
		isnull[4] = true;
		isnull[5] = true;
		isnull[6] = true;
		isnull[7] = true;
		isnull[8] = true;
		isnull[9] = true;
		isnull[10] = true;
	}
	else
	{
		values[3] = CStringGetTextDatum(pinfo->status);
		values[4] = Int32GetDatum(pinfo->crc32);
		values[5] = Int32GetDatum(pinfo->flags);
		if (!pinfo->kern_define)
			isnull[6] = true;
		else
			values[6] = PointerGetDatum(pinfo->kern_define);
		if (!pinfo->kern_source)
			isnull[7] = true;
		else
			values[7] = PointerGetDatum(pinfo->kern_source);
		if (!pinfo->kern_binary)
			isnull[8] = true;
		else
			values[8] = PointerGetDatum(pinfo->kern_binary);
		if (!pinfo->error_msg)
			isnull[9] = true;
		else
			values[9] = PointerGetDatum(pinfo->error_msg);
		if (!pinfo->backends)
			isnull[10] = true;
		else
			values[10] = PointerGetDatum(pinfo->backends);
	}
	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_program_info);

static void
pgstrom_startup_cuda_program(void)
{
	program_cache_entry *entry;
	bool		found;
	int			i;
	int			shift;
	char	   *curr_addr;
	char	   *end_addr;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	pgcache_head = ShmemInitStruct("PG-Strom program cache",
								   program_cache_size, &found);
	if (found)
		elog(ERROR, "Bug? shared memory for program cache already exists");

	/* initialize program cache header */
	memset(pgcache_head, 0, sizeof(program_cache_head));
	SpinLockInit(&pgcache_head->lock);
	for (i=0; i < PGCACHE_MAX_BITS; i++)
		dlist_init(&pgcache_head->free_list[i]);
	for (i=0; i < PGCACHE_HASH_SIZE; i++)
		dlist_init(&pgcache_head->active_list[i]);
	dlist_init(&pgcache_head->lru_list);
	pgcache_head->entry_begin = (program_cache_entry *)
		BUFFERALIGN(pgcache_head->data);

	/* makes free entries */
	curr_addr = (char *) pgcache_head->entry_begin;
	end_addr = ((char *) pgcache_head) + program_cache_size;
	shift = PGCACHE_MAX_BITS;
	while (shift >= PGCACHE_MIN_BITS)
	{
		if (curr_addr + (1UL << shift) > end_addr)
		{
			shift--;
			continue;
		}
		entry = (program_cache_entry *) curr_addr;
		memset(entry, 0, sizeof(program_cache_entry));
		entry->shift = shift;
		entry->refcnt = 0;
		dlist_push_tail(&pgcache_head->free_list[shift], &entry->hash_chain);

		curr_addr += (1UL << shift);
	}
	pgcache_head->entry_end = (program_cache_entry *)curr_addr;
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
	 * turn on/off cuda coredump feature
	 */
	DefineCustomBoolVariable("pg_strom.debug_cuda_coredump",
							 "Turn on/off GPU coredump feature",
							 NULL,
							 &pgstrom_enable_cuda_coredump,
							 false,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	if (pgstrom_enable_cuda_coredump)
	{
		elog(WARNING,
			 "pg_strom.debug.cuda_coredump = on is danger configuration. "
			 "It may lead random/unexpected system crash.");

		if (setenv("CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION", "0", 1) != 0 ||
			setenv("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "1", 1) != 0)
			elog(ERROR, "failed on set environment variable for core dump");
	}

	/*
	 * Init CUDA run-time compiler library
	 */
	rc = nvrtcVersion(&major, &minor);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcVersion: %s", nvrtcGetErrorString(rc));
	elog(LOG, "NVRTC - CUDA Runtime Compilation vertion %d.%d",
		 major, minor);

	/* allocation of static shared memory */
	RequestAddinShmemSpace(program_cache_size);
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_program;
}
