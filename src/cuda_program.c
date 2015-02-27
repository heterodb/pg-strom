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
	Size		kern_source_len;
	char	   *cuda_binary;
	Size		cuda_binary_len;
	char	   *error_msg;
	Size		error_msg_len;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} program_cache_entry;

#define PGCACHE_ACTIVE_ENTRY(entry)		((entry)->refcnt == 0)
#define PGCACHE_FREE_ENTRY(entry)		((entry)->refcnt > 0)
#define PGCACHE_MAGIC					0xabadcafe
#define PGCACHE_MIN_ERRORMSG_BUFSIZE	256
#define PGCACHE_ERRORMSG_LEN(entry)									\
	((uintptr_t)(entry) + (1UL << (entry)->shift) - sizeof(uint) -	\
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
static int		itemid_offset_shift;
static int		itemid_flags_shift;
static int		itemid_length_shift;

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

static void
pgstrom_write_cuda_program(int fdesc, program_cache_entry *entry,
						   const char *pathname)
{
	static size_t	common_code_length = 0;
	size_t			nbytes;

	/*
	 * Common PG-Strom device routine
	 */
	if (!common_code_length)
		common_code_length = strlen(pgstrom_cuda_common_code);
	nbytes = write(fdesc,
				   pgstrom_cuda_common_code,
				   common_code_length);
	if (nbytes != common_code_length)
		elog(ERROR, "could not write to file \"%s\": %m", pathname);

	/*
	 * Supplemental CUDA libraries
	 */
	/* cuda mathlib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_MATHLIB)
	{
		static size_t	mathlib_code_length = 0;

		if (!mathlib_code_length)
			mathlib_code_length = strlen(pgstrom_cuda_mathlib_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_mathlib_code,
					   mathlib_code_length);
		if (nbytes != mathlib_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* cuda timelib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_TIMELIB)
	{
		static size_t	timelib_code_length = 0;

		if (!timelib_code_length)
			timelib_code_length = strlen(pgstrom_cuda_timelib_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_timelib_code,
					   timelib_code_length);
		if (nbytes != timelib_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* cuda textlib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_TEXTLIB)
	{
		static size_t	textlib_code_length = 0;

		if (!textlib_code_length)
			textlib_code_length = strlen(pgstrom_cuda_textlib_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_textlib_code,
					   textlib_code_length);
		if (nbytes != textlib_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* cuda numeric.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_NUMERIC)
	{
		static size_t  numeric_code_length = 0;

		if (!numeric_code_length)
			numeric_code_length = strlen(pgstrom_cuda_numeric_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_numeric_code,
					   numeric_code_length);
		if (nbytes != numeric_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/*
	 * main logic of each GPU tasks (scan, sort, join and aggregate)
	 */
	/* gpuscan */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
	{
		static size_t	gpuscan_code_length = 0;

		if (!gpuscan_code_length)
			gpuscan_code_length = strlen(pgstrom_cuda_gpuscan_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_gpuscan_code,
					   gpuscan_code_length);
		if (nbytes != gpuscan_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* gpuhashjoin */
	if (entry->extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
	{
		static size_t	hashjoin_code_length = 0;

		if (!hashjoin_code_length)
			hashjoin_code_length = strlen(pgstrom_cuda_hashjoin_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_hashjoin_code,
					   hashjoin_code_length);
		if (nbytes != hashjoin_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* gpupreagg */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
	{
		static size_t	gpupreagg_code_length = 0;

		if (!gpupreagg_code_length)
			gpupreagg_code_length = strlen(pgstrom_cuda_gpupreagg_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_gpupreagg_code,
					   gpupreagg_code_length);
		if (nbytes != gpupreagg_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* gpusort */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUSORT)
	{
		static size_t	gpusort_code_length = 0;

		if (!gpusort_code_length)
			gpusort_code_length = strlen(pgstrom_cuda_gpusort_code);
		nbytes = write(fdesc,
					   pgstrom_cuda_gpusort_code,
					   gpusort_code_length);
		if (nbytes != gpusort_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* source code generated on the fly */
	nbytes = write(fdesc,
				   entry->kern_source,
				   entry->kern_source_len);
	if (nbytes != entry->kern_source_len)
		elog(ERROR, "could not write to file \"%s\": %m", pathname);
}

static void
__build_cuda_program(program_cache_entry *old_entry)
{
	const char *source_pathname;
	char		basename[MAXPGPATH];
	char		pathname[MAXPGPATH];
	StringInfoData cmdline;
	int			fdesc;
	ssize_t		filp_unitsz = 4096;
	ssize_t		nbytes;
	char	   *cuda_binary = NULL;
	Size		cuda_binary_len = 0;
	char	   *build_log = NULL;
	Size		build_log_len = 0;
	Size		required;
	Size		usage;
	int			hindex;
	const char *opt_optimize = "";
	int			rc;
	StringInfoData buf;
	program_cache_entry *new_entry;

	/*
	 * Write out the source program
	 */
	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "#define CUDA_DEVICE_CODE\n"
					 "#define HOSTPTRLEN %u\n"
					 "#define DEVICEPTRLEN %lu\n"
					 "#define BLCKSZ %u\n"
					 "#define ITEMID_OFFSET_SHIFT %u\n"
					 "#define ITEMID_FLAGS_SHIFT %u\n"
					 "#define ITEMID_LENGTH_SHIFT %u\n"
					 "#define MAXIMUM_ALIGNOF %u\n"
					 "\n"
					 "#include \"cuda_runtime.h\"\n"
					 "#include \"crt/device_runtime.h\"\n"
					 "\n",
					 SIZEOF_VOID_P,
					 sizeof(CUdeviceptr),
					 BLCKSZ,
					 itemid_offset_shift,
					 itemid_flags_shift,
					 itemid_length_shift,
					 MAXIMUM_ALIGNOF);

	fdesc = pgstrom_open_tempfile(".gpu", &source_pathname);
	nbytes = write(fdesc, buf.data, buf.len);
	if (nbytes != buf.len)
		elog(ERROR, "could not write to file \"%s\": %m", source_pathname);
	pgstrom_write_cuda_program(fdesc, old_entry, source_pathname);
	CloseTransientFile(fdesc);
	strncpy(basename, source_pathname, sizeof(basename));
	basename[strlen(source_pathname) - 4] = '\0';

	/*
	 * Makes a command line to be kicked
	 */
	if ((old_entry->extra_flags & DEVKERNEL_DISABLE_OPTIMIZE) != 0)
		opt_optimize = " -Xptxas '-O0'";

	initStringInfo(&cmdline);
	appendStringInfo(
		&cmdline,
		"env LANG=C %s %s --ptx -o %s.ptx %s"
#ifdef PGSTROM_DEBUG
		" -G -Werror cross-execution-space-call"
#endif
		" >& %s.log",
		pgstrom_nvcc_path,
		source_pathname,
		basename,
		opt_optimize,
		basename);

	/*
	 * Run nvcc compiler
	 */
	rc = system(cmdline.data);

	/*
	 * Read binary file (if any)
	 */
	if (rc == 0)
	{
		snprintf(pathname, sizeof(pathname), "%s.ptx", basename);
		fdesc = OpenTransientFile(pathname, O_RDONLY | PG_BINARY, 0);
		if (fdesc >= 0)
		{
			initStringInfo(&buf);
			do {
				enlargeStringInfo(&buf, filp_unitsz);
				nbytes = read(fdesc, buf.data + buf.len, filp_unitsz);
				if (nbytes < 0)
					elog(ERROR, "could not read from \"%s\": %m", pathname);
				buf.len += nbytes;
			} while (nbytes == filp_unitsz);
			CloseTransientFile(fdesc);

			cuda_binary = buf.data;
			cuda_binary_len = buf.len;
		}
	}

	/*
	 * Read build-log file (if any)
	 */
	snprintf(pathname, sizeof(pathname), "%s.log", basename);
	fdesc = OpenTransientFile(pathname, O_RDONLY | PG_BINARY, 0);
	if (fdesc >= 0)
	{
		initStringInfo(&buf);
		do {
			enlargeStringInfo(&buf, filp_unitsz);
			nbytes = read(fdesc, buf.data + buf.len, filp_unitsz);
			if (nbytes < 0)
				elog(ERROR, "could not read from \"%s\": %m", pathname);
			buf.len += nbytes;
		} while (nbytes == filp_unitsz);
		CloseTransientFile(fdesc);

		build_log = buf.data;
		build_log_len = buf.len;
	}

	/*
	 * Make a new entry, instead of the old one
	 */
	required = MAXALIGN(old_entry->kern_source_len + 1);
	if (cuda_binary)
		required += MAXALIGN(cuda_binary_len + 1);
	required += MAXALIGN(strlen(cmdline.data) + 1);
	required += MAXALIGN(build_log_len + 1);

	SpinLockAcquire(&pgcache_head->lock);
	new_entry = pgstrom_program_cache_alloc(required);
	if (!new_entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "out of shared memory");
	}
	usage = 0;
	new_entry->crc = old_entry->crc;
	new_entry->retain_cuda_program = old_entry->retain_cuda_program;
	new_entry->waiting_backends = NULL;		/* no need to set latch */
	new_entry->extra_flags = old_entry->extra_flags;
	new_entry->kern_source = new_entry->data + usage;
	memcpy(new_entry->kern_source,
		   old_entry->kern_source,
		   old_entry->kern_source_len + 1);
	new_entry->kern_source_len = old_entry->kern_source_len;
	usage += MAXALIGN(new_entry->kern_source_len + 1);

	if (!cuda_binary)
	{
		new_entry->cuda_binary = CUDA_PROGRAM_BUILD_FAILURE;
		new_entry->cuda_binary_len = 0;
	}
	else
	{
		new_entry->cuda_binary = new_entry->data + usage;
		new_entry->cuda_binary_len = cuda_binary_len;
		memcpy(new_entry->cuda_binary,
			   cuda_binary,
			   cuda_binary_len + 1);
		usage += MAXALIGN(cuda_binary_len + 1);
	}

	new_entry->error_msg = new_entry->data + usage;
	new_entry->error_msg_len = PGCACHE_ERRORMSG_LEN(new_entry);
	snprintf(new_entry->error_msg,
			 new_entry->error_msg_len,
			 "cuda kernel build: %s\n"
			 "command: %s\n%s",
			 !cuda_binary ? "failed" : "success",
			 cmdline.data,
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

	/*
	 * Remove temporary files (or retain for debug)
	 */
	if (cuda_binary)
	{
		snprintf(pathname, sizeof(pathname), "%s.ptx", basename);
		if (unlink(pathname) != 0)
			elog(WARNING, "could not cleanup \"%s\" : %m", pathname);
	}

	if (build_log)
	{
		snprintf(pathname, sizeof(pathname), "%s.log", basename);
		if (unlink(pathname) != 0)
			elog(WARNING, "could not cleanup \"%s\" : %m", pathname);
	}

	if (!old_entry->retain_cuda_program)
	{
		if (unlink(source_pathname) != 0)
			elog(WARNING, "could not cleanup \"%s\" : %m", source_pathname);
	}
}



static void
pgstrom_build_cuda_program(Datum cuda_program)
{
	program_cache_entry *entry = (program_cache_entry *) cuda_program;
	MemoryContext	memcxt = CurrentMemoryContext;
	MemoryContext	oldcxt;

	Assert(entry->cuda_binary == NULL);

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
		if (!entry->cuda_binary)
		{
			snprintf(entry->error_msg, entry->error_msg_len,
					 "(%s:%d, %s) %s",
					 errdata->filename,
					 errdata->lineno,
					 errdata->funcname,
					 errdata->message);
			entry->cuda_binary = CUDA_PROGRAM_BUILD_FAILURE;
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

bool
pgstrom_load_cuda_program(GpuTaskState *gts)
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
			entry->kern_source_len == kern_source_len &&
			strcmp(entry->kern_source, kern_source) == 0)
		{
			/* Move this entry to the head of LRU list */
			dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);

			/* This kernel build already lead an error */
			if (entry->cuda_binary == CUDA_PROGRAM_BUILD_FAILURE)
			{
				SpinLockRelease(&pgcache_head->lock);
				elog(ERROR, "%s", entry->error_msg);
			}
			/* Kernel build is still in-progress */
			if (!entry->cuda_binary)
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
					rc = cuCtxSetCurrent(gcontext->cuda_context[i]);
					if (rc != CUDA_SUCCESS)
						elog(ERROR, "failed on cuCtxSetCurrent (%s)",
							 errorText(rc));

					rc = cuModuleLoadData(&cuda_modules[i],
										  entry->cuda_binary);
					if (rc != CUDA_SUCCESS)
						elog(ERROR, "failed on cuModuleLoadData (%s)",
							 errorText(rc));
				}
				rc = cuCtxSetCurrent(NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuCtxSetCurrent (%s)",
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
	entry->kern_source_len = kern_source_len;
	usage += MAXALIGN(kern_source_len + 1);
	/* no cuda binary yet */
	entry->cuda_binary = NULL;
    entry->cuda_binary_len = 0;
	/* remaining are for error message */
	entry->error_msg = (char *)(entry->data + usage);
	entry->error_msg_len = PGCACHE_ERRORMSG_LEN(entry);

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
	ItemIdData	item_id;
	uint		code;

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

	/*
	 * NOTE: Here is no C standard for bitfield layout (thus, OpenCL does not
	 * support bitfields), so we need to tell run-time compiler exact layout
	 * of the ItemIdData structure.
	 */
	Assert(sizeof(item_id) == sizeof(code));
	memset(&item_id, 0, sizeof(ItemIdData));
	item_id.lp_off = 1;
	memcpy(&code, &item_id, sizeof(ItemIdData));
	for (itemid_offset_shift = 0;
		 ((code >> itemid_offset_shift) & 0x0001) == 0;
		 itemid_offset_shift++);

	memset(&item_id, 0, sizeof(ItemIdData));
	item_id.lp_flags = 1;
	memcpy(&code, &item_id, sizeof(ItemIdData));
	for (itemid_flags_shift = 0;
		 ((code >> itemid_flags_shift) & 0x0001) == 0;
		 itemid_flags_shift++);

	memset(&item_id, 0, sizeof(ItemIdData));
	item_id.lp_len = 1;
	memcpy(&code, &item_id, sizeof(ItemIdData));
	for (itemid_length_shift = 0;
		 ((code >> itemid_length_shift) & 0x0001) == 0;
		 itemid_length_shift++);

	/* allocation of static shared memory */
	RequestAddinShmemSpace(program_cache_size);
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_program;
}
