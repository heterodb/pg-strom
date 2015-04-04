/*
 * cuda_program.c
 *
 * Routines for just-in-time comple cuda code
 * ----
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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
#include "common/pg_crc.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include <nvrtc.h>
#include <sys/stat.h>
#include <sys/types.h>

typedef struct
{
	dlist_node	hash_chain;
	dlist_node	lru_chain;
	int			shift;	/* block class of this entry */
	int			refcnt;	/* 0 means free entry */
	pg_crc32	crc;	/* hash value by extra_flags + kern_source */
	bool		retain_cuda_program;
	Bitmapset  *waiting_backends;
	int			extra_flags;
	char	   *kern_source;
	char	   *ptx_image;
	char	   *error_msg;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} program_cache_entry;

#define PGCACHE_ACTIVE_ENTRY(entry)		((entry)->refcnt == 0)
#define PGCACHE_FREE_ENTRY(entry)		((entry)->refcnt > 0)
#define PGCACHE_MAGIC					0xabadcafe
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
static char	   *pgstrom_nvcc_path;
static bool		debug_optimize_cuda_program;
static bool		debug_retain_cuda_program;

/* ---- static variables ---- */
static shmem_startup_hook_type shmem_startup_next;
static program_cache_head *pgcache_head;

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
	entry->shift = shift;
	entry->refcnt = 0;
	dlist_push_tail(&pgcache_head->free_list[shift], &entry->hash_chain);

	/* later half */
	entry = (program_cache_entry *)((char *)entry + (1UL << shift));
	entry->shift = shift;
	entry->refcnt = 0;
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
		if (!pgstrom_program_cache_split(shift + 1) &&
			!pgstrom_program_cache_reclaim(shift))
			return NULL;
	}
	Assert(!dlist_is_empty(&pgcache_head->free_list[shift]));

	dnode = dlist_pop_head_node(&pgcache_head->free_list[shift]);
	entry = dlist_container(program_cache_entry, hash_chain, dnode);
	Assert(entry->shift == shift);

	memset(entry, 0, sizeof(program_cache_entry));
	entry->shift = shift;
	entry->refcnt = 1;
	*((uint *)((char *)entry + (1UL << shift) - sizeof(uint))) = PGCACHE_MAGIC;

	return entry;
}

static void
pgstrom_program_cache_free(program_cache_entry *entry)
{
	int			shift = entry->shift;
	Size		offset;

	Assert(entry->refcnt == 0);

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

		if (buddy >= pgcache_head->entry_end ||	/* out of range? */
			buddy->shift != shift ||			/* same size? */
			PGCACHE_ACTIVE_ENTRY(buddy))		/* and free entry? */
			break;
		/* OK, chunk and buddy can be merged */

		dlist_delete(&buddy->hash_chain);
		if (buddy < entry)
			entry = buddy;
		entry->shift = ++shift;
	}
	dlist_push_head(&pgcache_head->free_list[shift], &entry->hash_chain);
}

static void
pgstrom_put_cuda_program(program_cache_entry *entry)
{
	SpinLockAcquire(&pgcache_head->lock);
	if (--entry->refcnt == 0)
	{
		/*
		 * NOTE: unless pgstrom_program_cache_reclaim() detach
		 * entries, it never goes to refcnt == 0.
		 */
		Assert(!entry->hash_chain.next && !entry->hash_chain.prev);
		Assert(!entry->lru_chain.next && !entry->lru_chain.prev);
		pgstrom_program_cache_free(entry);
	}
	SpinLockRelease(&pgcache_head->lock);
}

static char *
pgstrom_write_cuda_program(program_cache_entry *entry)
{
	StringInfoData		source;

	initStringInfo(&source);
	appendStringInfo(&source,
					 "#define HOSTPTRLEN %u\n"
					 "#define DEVICEPTRLEN %lu\n"
					 "#define BLCKSZ %u\n"
					 "#define MAXIMUM_ALIGNOF %u\n"
					 "\n",
					 SIZEOF_VOID_P,
					 sizeof(CUdeviceptr),
					 BLCKSZ,
					 MAXIMUM_ALIGNOF);

	/* disable C++ feature */
	appendStringInfo(&source,
					 "#ifdef __cplusplus\n"
					 "extern \"C\" {\n"
					 "#endif	/* __cplusplus */\n");
	/* Common PG-Strom device routine */
	appendStringInfoString(&source, pgstrom_cuda_common_code);

	/* PG-Strom CUDA device code libraries */

	/* cuda mathlib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_MATHLIB)
		appendStringInfoString(&source, pgstrom_cuda_mathlib_code);
	/* cuda timelib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_TIMELIB)
		appendStringInfoString(&source, pgstrom_cuda_timelib_code);
	/* cuda textlib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_TEXTLIB)
		appendStringInfoString(&source, pgstrom_cuda_textlib_code);
	/* cuda numeric.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_NUMERIC)
		appendStringInfoString(&source, pgstrom_cuda_numeric_code);

	/* Main logic of each GPU tasks */

	/* GpuScan */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
		appendStringInfoString(&source, pgstrom_cuda_gpuscan_code);
	/* GpuHashJoin */
	if (entry->extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
		appendStringInfoString(&source, pgstrom_cuda_hashjoin_code);
	/* GpuPreAgg */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
		appendStringInfoString(&source, pgstrom_cuda_gpupreagg_code);
	/* GpuSort */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUSORT)
		appendStringInfoString(&source, pgstrom_cuda_gpusort_code);

	/* Source code generated on the fly */
	appendStringInfoString(&source, entry->kern_source);

	/* disable C++ feature */
	appendStringInfo(&source,
					 "#ifdef __cplusplus\n"
					 "}\n"
					 "#endif /* __cplusplus */\n");
	return source.data;
}

static void
__build_cuda_program(program_cache_entry *old_entry)
{
	char		   *source;
	char		   *source_pathname = NULL;
	nvrtcProgram	program;
	nvrtcResult		rc;
	const char	   *options[10];
	int				opt_index = 0;
	char		   *ptx_image;
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
	source = pgstrom_write_cuda_program(old_entry);
	rc = nvrtcCreateProgram(&program,
							source,
							"autogen",
							0,
							NULL,
							NULL);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcCreateProgram: %s",
			 nvrtcGetErrorString(rc));

	/*
	 * Put command line options
	 */
	options[opt_index++] =
		psprintf("--gpu-architecture=compute_%u",
				 pgstrom_baseline_cuda_capability());
#ifdef PGSTROM_DEBUG
	options[opt_index++] = "--device-debug";
	options[opt_index++] = "--generate-line-info";
#endif
	options[opt_index++] = "--use_fast_math";

	/*
	 * Save the source file, if needed
	 */
	if (old_entry->retain_cuda_program)
	{
		char	pathname[128];
		int		fdesc;

		strcpy(pathname, "/tmp/pg_strom_XXXXXX.gpu");
		fdesc = mkstemps(pathname, 4);
		if (fdesc >= 0)
		{
			write(fdesc, source, strlen(source));
			close(fdesc);
			source_pathname = pstrdup(pathname);
		}
	}

	/*
	 * Kick runtime compiler
	 */
	rc = nvrtcCompileProgram(program, opt_index, options);
	if (rc != NVRTC_SUCCESS)
	{
		if (rc != NVRTC_ERROR_COMPILATION)
			elog(ERROR, "failed on nvrtcCompileProgram: %s",
				 nvrtcGetErrorString(rc));
		else
			build_failure = true;
	}

	/*
	 * Read PTX Binary
	 */
	if (build_failure)
		ptx_image = NULL;
	else
	{
		rc = nvrtcGetPTXSize(program, &length);
		if (rc != NVRTC_SUCCESS)
			elog(ERROR, "failed on nvrtcGetPTXSize: %s",
				 nvrtcGetErrorString(rc));
		ptx_image = palloc(length + 1);

		rc = nvrtcGetPTX(program, ptx_image);
		if (rc != NVRTC_SUCCESS)
			elog(ERROR, "failed on nvrtcGetPTX: %s",
				 nvrtcGetErrorString(rc));
		ptx_image[length] = '\0';	/* may not be necessary */
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
	if (ptx_image)
		required += MAXALIGN(strlen(ptx_image) + 1);
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
	new_entry->retain_cuda_program = false;
	new_entry->waiting_backends = NULL;		/* no need to set latch */
	new_entry->extra_flags = old_entry->extra_flags;
	new_entry->kern_source = new_entry->data + usage;
	length = strlen(old_entry->kern_source);
	memcpy(new_entry->kern_source,
		   old_entry->kern_source, length + 1);
	usage += MAXALIGN(length + 1);

	if (!ptx_image)
		new_entry->ptx_image = CUDA_PROGRAM_BUILD_FAILURE;
	else
	{
		new_entry->ptx_image = new_entry->data + usage;
		length = strlen(ptx_image);
		memcpy(new_entry->ptx_image, ptx_image, length + 1);
		usage += MAXALIGN(length + 1);
	}
	new_entry->error_msg = new_entry->data + usage;
	length = PGCACHE_ERRORMSG_LEN(new_entry);
	if (source_pathname)
		snprintf(new_entry->error_msg, length,
				 "build: %s\n%s\nsource: %s\n",
				 !ptx_image ? "failed" : "success",
				 build_log,
 				 source_pathname);
	else
		snprintf(new_entry->error_msg, length,
				 "build: %s\n%s",
				 !ptx_image ? "failed" : "success",
				 build_log);
	/*
	 * Add new_entry to the hash slot
	 */
	hindex = new_entry->crc % PGCACHE_HASH_SIZE;
	dlist_push_head(&pgcache_head->active_list[hindex],
					&new_entry->hash_chain);
	dlist_push_head(&pgcache_head->lru_list,
					&new_entry->lru_chain);

	/*
	 * Also, waking up blocking tasks and drop old_entry
	 * which shall not be referenced no longer.
	 */
	pgstrom_wakeup_backends(old_entry->waiting_backends);

	dlist_delete(&old_entry->hash_chain);
	dlist_delete(&old_entry->lru_chain);
	memset(&old_entry->hash_chain, 0, sizeof(dlist_node));
	memset(&old_entry->lru_chain, 0, sizeof(dlist_node));
	if (--old_entry->refcnt == 0)
		pgstrom_program_cache_free(old_entry);

	SpinLockRelease(&pgcache_head->lock);
}



static void
pgstrom_build_cuda_program(Datum cuda_program)
{
	program_cache_entry *entry = (program_cache_entry *) cuda_program;
	MemoryContext	memcxt = CurrentMemoryContext;
	MemoryContext	oldcxt;

	Assert(entry->ptx_image == NULL);

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
		if (!entry->ptx_image)
		{
			snprintf(entry->error_msg, PGCACHE_ERRORMSG_LEN(entry),
					 "(%s:%d, %s) %s",
					 errdata->filename,
					 errdata->lineno,
					 errdata->funcname,
					 errdata->message);
			entry->ptx_image = CUDA_PROGRAM_BUILD_FAILURE;
		}
		pgstrom_wakeup_backends(entry->waiting_backends);
		SpinLockRelease(&pgcache_head->lock);
		pgstrom_put_cuda_program(entry);
		MemoryContextSwitchTo(oldcxt);
		PG_RE_THROW();
	}
	PG_END_TRY();
	pgstrom_put_cuda_program(entry);
}

static bool
__pgstrom_load_cuda_program(GpuTaskState *gts, bool is_preload)
{
	program_cache_entry	*entry;
	GpuContext	   *gcontext = gts->gcontext;
	cl_uint			extra_flags = gts->extra_flags;
	const char	   *kern_source = gts->kern_source;
	Size			kern_source_len = strlen(kern_source);
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

	/* Is optimization available? */
	if (!debug_optimize_cuda_program)
		extra_flags |= DEVKERNEL_DISABLE_OPTIMIZE;

	/* makes a hash value */
	INIT_CRC32C(crc);
	COMP_CRC32C(crc, &extra_flags, sizeof(int32));
	COMP_CRC32C(crc, kern_source, kern_source_len);
	FIN_CRC32C(crc);

retry:
	hindex = crc % PGCACHE_HASH_SIZE;
	SpinLockAcquire(&pgcache_head->lock);
	dlist_foreach (iter, &pgcache_head->active_list[hindex])
	{
		entry = dlist_container(program_cache_entry, hash_chain, iter.cur);

		if (entry->crc == crc &&
			entry->extra_flags == extra_flags &&
			strcmp(entry->kern_source, kern_source) == 0)
		{
			/* Move this entry to the head of LRU list */
			dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);

			/* This kernel build already lead an error */
			if (entry->ptx_image == CUDA_PROGRAM_BUILD_FAILURE)
			{
				SpinLockRelease(&pgcache_head->lock);
				if (!is_preload)
					elog(ERROR, "%s", entry->error_msg);
				return false;
			}
			/* Kernel build is still in-progress */
			if (!entry->ptx_image)
			{
				Bitmapset  *waiting_backends = entry->waiting_backends;
				waiting_backends->words[WORDNUM(MyProc->pgprocno)]
					|= (1 << BITNUM(MyProc->pgprocno));
				SpinLockRelease(&pgcache_head->lock);
				return false;
			}
			/* OK, this kernel is already built */
			entry->refcnt++;
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
										  entry->ptx_image);
					if (rc != CUDA_SUCCESS)
						elog(ERROR, "failed on cuModuleLoadData (%s)\n",
							 errorText(rc));
				}
				rc = cuCtxPopCurrent(NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuCtxPopCurrent (%s)",
						 errorText(rc));
				gts->cuda_modules = cuda_modules;
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
			return true;
		}
	}
	/* Not found on the existing cache */
	required = offsetof(program_cache_entry, data[0]);
	nwords = (ProcGlobal->allProcCount +
			  BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	required += MAXALIGN(offsetof(Bitmapset, words[nwords]));
	required += MAXALIGN(kern_source_len + 1);
	required += 512;	/* margin for error message */
	usage = 0;

	entry = pgstrom_program_cache_alloc(required);
	entry->crc = crc;
	entry->retain_cuda_program = debug_retain_cuda_program;
	/* bitmap for waiting backends */
	entry->waiting_backends = (Bitmapset *) entry->data;
	entry->waiting_backends->nwords = nwords;
	memset(entry->waiting_backends->words, 0, sizeof(bitmapword) * nwords);
	usage += MAXALIGN(offsetof(Bitmapset, words[nwords]));
	/* device kernel source */
	entry->extra_flags = extra_flags;
	entry->kern_source = (char *)(entry->data + usage);
	memcpy(entry->kern_source, kern_source, kern_source_len + 1);
	usage += MAXALIGN(kern_source_len + 1);
	/* no cuda binary yet */
	entry->ptx_image = NULL;
	/* remaining are for error message */
	entry->error_msg = (char *)(entry->data + usage);

	/* at least, caller is waiting for build */
	entry->waiting_backends = entry->waiting_backends;
	entry->waiting_backends->words[WORDNUM(MyProc->pgprocno)]
		|= (1 << BITNUM(MyProc->pgprocno));

	/* to be acquired by program builder */
	entry->refcnt++;
	dlist_push_head(&pgcache_head->active_list[hindex], &entry->hash_chain);
	dlist_push_head(&pgcache_head->lru_list, &entry->lru_chain);

	/* Kick a dynamic background worker to build */
	snprintf(worker.bgw_name, sizeof(worker.bgw_name),
			 "nvcc launcher - crc %08x", crc);
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	worker.bgw_restart_time = BGW_NEVER_RESTART;
	worker.bgw_main = pgstrom_build_cuda_program;
	worker.bgw_main_arg = PointerGetDatum(entry);

	if (!RegisterDynamicBackgroundWorker(&worker, NULL))
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(LOG, "failed to launch nvcc asynchronous mode, try synchronous");
		pgstrom_build_cuda_program(PointerGetDatum(entry));
		goto retry;
	}
	SpinLockRelease(&pgcache_head->lock);

	return false;	/* now build the device kernel */
}

bool
pgstrom_load_cuda_program(GpuTaskState *gts)
{
	return __pgstrom_load_cuda_program(gts, false);
}

void
pgstrom_preload_cuda_program(GpuTaskState *gts)
{
	__pgstrom_load_cuda_program(gts, true);
}

Datum
pgstrom_program_info(PG_FUNCTION_ARGS)
{
	elog(ERROR, "not implemented yet");
	PG_RETURN_NULL();
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

	DefineCustomStringVariable("pg_strom.nvcc_path",
							   "path to nvcc (NVIDIA CUDA Compiler)",
							   NULL,
							   &pgstrom_nvcc_path,
							   "/usr/local/cuda/bin/nvcc",
							   PGC_SIGHUP,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);

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

	DefineCustomBoolVariable("pg_strom.debug_optimize_program",
							 "enabled optimization on program build",
							 NULL,
							 &debug_optimize_cuda_program,
							 true,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.debug_retain_program",
							 "retain CUDA program file generated on the fly",
							 NULL,
							 &debug_retain_cuda_program,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* CUDA online compiler */
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
