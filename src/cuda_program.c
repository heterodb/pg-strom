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
#include "utils/pg_crc.h"
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
	Bitmapset  *waiting_backends;
	int			extra_flags;
	char	   *kern_define;
	char	   *kern_source;
	char	   *ptx_image;
	char	   *error_msg;
	char		data[FLEXIBLE_ARRAY_MEMBER];
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

	/* Per session definition if any */
	appendStringInfoString(&source, kern_define);

	/* PG-Strom CUDA device code libraries */

	/* cuda mathlib.h */
	if (extra_flags & DEVFUNC_NEEDS_MATHLIB)
		appendStringInfoString(&source, pgstrom_cuda_mathlib_code);
	/* cuda timelib.h */
	if (extra_flags & DEVFUNC_NEEDS_TIMELIB)
		appendStringInfoString(&source, pgstrom_cuda_timelib_code);
	/* cuda textlib.h */
	if (extra_flags & DEVFUNC_NEEDS_TEXTLIB)
		appendStringInfoString(&source, pgstrom_cuda_textlib_code);
	/* cuda numeric.h */
	if (extra_flags & DEVFUNC_NEEDS_NUMERIC)
		appendStringInfoString(&source, pgstrom_cuda_numeric_code);

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

	/* Source code generated on the fly */
	appendStringInfoString(&source, kern_source);

	/* disable C++ feature */
	appendStringInfo(&source,
					 "#ifdef __cplusplus\n"
					 "}\n"
					 "#endif /* __cplusplus */\n");
	return source.data;
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
	options[opt_index++] =
		psprintf("--gpu-architecture=compute_%u",
				 pgstrom_baseline_cuda_capability());
#ifdef PGSTROM_DEBUG
	options[opt_index++] = "--device-debug";
	options[opt_index++] = "--generate-line-info";
#endif
	options[opt_index++] = "--use_fast_math";

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
	required += MAXALIGN(strlen(old_entry->kern_define) + 1);
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
	new_entry->waiting_backends = NULL;		/* no need to set latch */
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
	 * Waking up blocking tasks, and detach old_entry from
	 * the hash/lru list to ensure nobody will grab it.
	 */
	pgstrom_wakeup_backends(old_entry->waiting_backends);

	dlist_delete(&old_entry->hash_chain);
	dlist_delete(&old_entry->lru_chain);
	memset(&old_entry->hash_chain, 0, sizeof(dlist_node));
	memset(&old_entry->lru_chain, 0, sizeof(dlist_node));

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
	const char	   *kern_define = gts->kern_define;
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
			Assert(entry->refcnt > 0);
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
	required += MAXALIGN(kern_define_len + 1);
	required += 512;	/* margin for error message */
	usage = 0;

	entry = pgstrom_program_cache_alloc(required);
	entry->crc = crc;
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
	entry->kern_define = (char *)(entry->data + usage);
	memcpy(entry->kern_define, kern_define, kern_define_len + 1);
	usage += MAXALIGN(kern_define_len + 1);
	/* no cuda binary yet */
	entry->ptx_image = NULL;
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

/*
 * assign_timelib_session_info
 *
 * It construct per-session information around timelib.h.
 */
static void
assign_timelib_session_info(StringInfo buf)
{
	appendStringInfo(
		buf,
		"/* ================================================\n"
		" * session information for cuda_timelib.h\n"
		" * ================================================ */\n");



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
							const char *kern_source,
							int extra_flags)
{
	const char	   *kern_define;
	StringInfoData	buf;

	if ((extra_flags & (DEVFUNC_NEEDS_TIMELIB)) != 0)
	{
		initStringInfo(&buf);

		/* put timezone info */
		assign_timelib_session_info(&buf);

		kern_define = buf.data;
	}
	else
		kern_define = "";	/* no session specific code */

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
	text	   *ptx_image;
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
		if (entry->ptx_image == CUDA_PROGRAM_BUILD_FAILURE)
			pinfo->status = "Build Failed";
		else if (!entry->ptx_image)
			pinfo->status = "In Progress";
		else
			pinfo->status = "Ready";
		pinfo->crc32 = entry->crc;
		pinfo->flags = entry->extra_flags;
		if (entry->kern_define)
			pinfo->kern_define = cstring_to_text(entry->kern_define);
		if (entry->kern_source)
			pinfo->kern_source = cstring_to_text(entry->kern_source);
		if (entry->ptx_image)
			pinfo->ptx_image = cstring_to_text(entry->ptx_image);
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
	Datum			values[10];
	bool			isnull[10];
	HeapTuple		tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(10, false);
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
		TupleDescInitEntry(tupdesc, (AttrNumber) 9, "ptx_image",
						   TEXTOID, -1, 0);
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
		if (!pinfo->ptx_image)
			isnull[8] = true;
		else
			values[8] = PointerGetDatum(pinfo->ptx_image);
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
