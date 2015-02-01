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
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/guc.h"
#include "utils/pg_crc.h"
#include "pg_strom.h"
#include <sys/stat.h>
#include <sys/types.h>

typedef struct
{
	dlist_node	chain;
	dlist_node	lru_chain;
	int			shift;	/* block class of this entry */
	int			refcnt;	/* 0 means free entry */
	pg_crc32	crc;	/* hash value by extra_flags + cuda_source */
	int			extra_flags;
	char	   *cuda_source;
	Size		cuda_source_len;
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

typedef struct
{
	volatile slock_t lock;
	dlist_head	free_list[PGCACHE_MAX_BITS + 1];
	dlist_head	active_list[PGCACHE_HASH_SIZE];
	dlist_head	lru_list;
	program_cache_entry *entry_begin;	/* start address of entries */
	program_cache_entry *entry_end;		/* end address of entries */
	Bitmapset	waiting_backends;	/* flexible length */
} program_cache_head;

#define TOTAL_PROCS				\
	(MaxBackends + NUM_AUXILIARY_PROCS + max_prepared_xacts)

/* ---- static variables ---- */
static shmem_startup_hook_type shmem_startup_next;
static Size		program_cache_size;
static char	   *pgstrom_nvcc_path;
static program_cache_head *pgcache_head;
static int		itemid_offset_shift;
static int		itemid_flags_shift;
static int		itemid_length_shift;

/* ---- static functions ---- */
static program_cache_entry *pgstrom_program_cache_alloc(Size required);
static void pgstrom_program_cache_free(program_cache_entry *entry);

/*****/
static void
pgstrom_wakeup_backends(void)
{
	struct PGPROC  *proc;
	bitmapword		bitmap;
	int				i, j;

	SpinLockAcquire(&pgcache_head->lock);
	for (i=0; i < pgcache_head->waiting_backends.nwords; i++)
	{
		bitmap = pgcache_head->waiting_backends.words[i];
		if (!bitmap)
			continue;

		for (j=0; j < BITS_PER_BITMAPWORD; j++)
		{
			if ((bitmap & (1 << j)) == 0)
				continue;
			Assert(i * BITS_PER_BITMAPWORD + j < TOTAL_PROCS);
			proc = &ProcGlobal->allProcs[i * BITS_PER_BITMAPWORD + j];
			SetLatch(&proc->procLatch);
		}
		pgcache_head->waiting_backends.words[i] = 0;
	}
	SpinLockRelease(&pgcache_head->lock);
}

/*****/
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
		dlist_delete(&entry->chain);
		dlist_delete(&entry->lru_chain);
		memset(&entry->chain, 0, sizeof(dlist_node));
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

	entry = dlist_container(program_cache_entry, chain, dnode);
	Assert(entry->shift == shift);
	Assert((((uintptr_t)entry - (uintptr_t)pgcache_head->entry_begin)
			& ((1UL << shift) - 1)) == 0);
	shift--;

	/* earlier half */
	entry->shift = shift;
	entry->refcnt = 0;
	dlist_push_tail(&pgcache_head->free_list[shift], &entry->chain);

	/* later half */
	entry = (program_cache_entry *)((char *)entry + (1UL << shift));
	entry->shift = shift;
	entry->refcnt = 0;
	dlist_push_tail(&pgcache_head->free_list[shift], &entry->chain);

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
	entry = dlist_container(program_cache_entry, chain, dnode);
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
	Assert(!entry->chain.next && !entry->chain.prev);
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

		if (buddy >= pgcache_head->entry_end ||	/* out of range? */
			buddy->shift != shift ||			/* same size? */
			PGCACHE_ACTIVE_ENTRY(buddy))		/* and free entry? */
			break;
		/* OK, chunk and buddy can be merged */

		dlist_delete(&buddy->chain);
		if (buddy < entry)
			entry = buddy;
		entry->shift = ++shift;
	}
	dlist_push_head(&pgcache_head->free_list[shift], &entry->chain);
}

static void
pgstrom_write_cuda_program(FILE *filp, program_cache_entry *entry,
						   const char *pathname)
{
	static size_t	common_code_length = 0;
	size_t			nbytes;

	/*
	 * Common PG-Strom device routine
	 */
	if (!common_code_length)
		common_code_length = strlen(pgstrom_cuda_common_code);
	nbytes = fwrite(pgstrom_cuda_common_code,
					common_code_length, 1, filp);
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
		nbytes = fwrite(pgstrom_cuda_mathlib_code,
						mathlib_code_length, 1, filp);
		if (nbytes != mathlib_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* cuda timelib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_TIMELIB)
	{
		static size_t	timelib_code_length = 0;

		if (!timelib_code_length)
			timelib_code_length = strlen(pgstrom_cuda_timelib_code);
		nbytes = fwrite(pgstrom_cuda_timelib_code,
						timelib_code_length, 1, filp);
		if (nbytes != timelib_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* cuda textlib.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_TEXTLIB)
	{
		static size_t	textlib_code_length = 0;

		if (!textlib_code_length)
			textlib_code_length = strlen(pgstrom_cuda_textlib_code);
		nbytes = fwrite(pgstrom_cuda_textlib_code,
						textlib_code_length, 1, filp);
		if (nbytes != textlib_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* cuda numeric.h */
	if (entry->extra_flags & DEVFUNC_NEEDS_NUMERIC)
	{
		static size_t  numeric_code_length = 0;

		if (!numeric_code_length)
			numeric_code_length = strlen(pgstrom_cuda_numeric_code);
		nbytes = fwrite(pgstrom_cuda_numeric_code,
						numeric_code_length, 1, filp);
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
		nbytes = fwrite(pgstrom_cuda_gpuscan_code,
						gpuscan_code_length, 1, filp);
		if (nbytes != gpuscan_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* gpuhashjoin */
	if (entry->extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
	{
		static size_t	hashjoin_code_length = 0;

		if (!hashjoin_code_length)
			hashjoin_code_length = strlen(pgstrom_cuda_hashjoin_code);
		nbytes = fwrite(pgstrom_cuda_hashjoin_code,
						hashjoin_code_length, 1, filp);
		if (nbytes != hashjoin_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* gpupreagg */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
	{
		static size_t	gpupreagg_code_length = 0;

		if (!gpupreagg_code_length)
			gpupreagg_code_length = strlen(pgstrom_cuda_gpupreagg_code);
		nbytes = fwrite(pgstrom_cuda_gpupreagg_code,
						gpupreagg_code_length, 1, filp);
		if (nbytes != gpupreagg_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* gpusort */
	if (entry->extra_flags & DEVKERNEL_NEEDS_GPUSORT)
	{
		static size_t	gpusort_code_length = 0;

		if (!gpusort_code_length)
			gpusort_code_length = strlen(pgstrom_cuda_gpusort_code);
		nbytes = fwrite(pgstrom_cuda_gpusort_code,
						gpusort_code_length, 1, filp);
		if (nbytes != gpusort_code_length)
			elog(ERROR, "could not write to file \"%s\": %m", pathname);
	}
	/* source code generated on the fly */
	nbytes = fwrite(entry->cuda_source,
					entry->cuda_source_len, 1, filp);
	if (nbytes != entry->cuda_source_len)
		elog(ERROR, "could not write to file \"%s\": %m", pathname);



}

static void
__build_cuda_program(program_cache_entry *old_entry)
{
	char		basename[MAXPGPATH];
	char		pathname[MAXPGPATH];
	StringInfoData cmdline;
	uint		uniq_id;
	FILE	   *filp;
	size_t		filp_unitsz = 2048;
	size_t		nbytes;
	char	   *cuda_binary = NULL;
	Size		cuda_binary_len = 0;
	char	   *build_log = NULL;
	Size		build_log_len = 0;
	Size		required;
	int			hindex;
	int			rc;
	StringInfoData buf;
	program_cache_entry *new_entry;

	/* make a basename of the tempfiles */
	uniq_id = ((uintptr_t)old_entry -
			   (uintptr_t)pgcache_head->entry_begin) >> PGCACHE_MIN_BITS;
	snprintf(basename, sizeof(basename), "%s/code_%08x",
			 PGSTROM_TEMP_DIR, uniq_id);

	snprintf(pathname, sizeof(pathname), "%s.gpu", basename);
	filp = AllocateFile(pathname, PG_BINARY_W);
	if (!filp)
	{
		/*
		 * NOTE: we don't check error code because we have no reasonable
		 * way to prevent concurrent mkdir(3). If actually it failed,
		 * the AllocateFile() will fail eventually.
		 */
		mkdir(PGSTROM_TEMP_DIR, S_IRWXU);

		filp = AllocateFile(pathname, PG_BINARY_W);
		if (!filp)
			elog(ERROR, "could not create source file \"%s\": %m", pathname);
	}
	/* Write out the source code */
	pgstrom_write_cuda_program(filp, old_entry, pathname);
	/* OK, done */
	FreeFile(filp);

	/*
	 * Makes a command line to be kicked
	 */
	initStringInfo(&cmdline);
	appendStringInfo(
		&cmdline,
		"%s %s.gpu --ptx "
		" -DFLEXIBLE_ARRAY_MEMBER"
#ifdef PGSTROM_DEBUG
		" -Werror -G"
#endif
		" -DCUDA_DEVICE_CODE"
		" -DHOSTPTRLEN=%u -DBLCKSZ=%u"
		" -DITEMID_OFFSET_SHIFT=%u"
		" -DITEMID_FLAGS_SHIFT=%u"
		" -DITEMID_LENGTH_SHIFT=%u"
		" -DMAXIMUM_ALIGNOF=%u"
		" 2>%s.log",
		pgstrom_nvcc_path,
		basename,
		SIZEOF_VOID_P, BLCKSZ,
		itemid_offset_shift,
		itemid_flags_shift,
		itemid_length_shift,
		MAXIMUM_ALIGNOF,
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
		filp = AllocateFile(pathname, PG_BINARY_R);
		if (!filp)
			elog(ERROR, "could not open \"%s\": %m", pathname);

		initStringInfo(&buf);
		do {
			enlargeStringInfo(&buf, filp_unitsz);
			nbytes = fread(buf.data + buf.len, 1, filp_unitsz, filp);
			if (nbytes < filp_unitsz && ferror(filp) != 0)
				elog(ERROR, "could not read from \"%s\": %m", pathname);
		} while (nbytes == filp_unitsz);

		cuda_binary = buf.data;
		cuda_binary_len = buf.len;

		FreeFile(filp);
	}

	/*
	 * Read build-log file (if any)
	 */
	snprintf(pathname, sizeof(pathname), "%s.log", basename);
	filp = AllocateFile(pathname, PG_BINARY_R);
	if (!filp)
		elog(ERROR, "could not open \"%s\": %m", pathname);

	initStringInfo(&buf);
	do {
		enlargeStringInfo(&buf, filp_unitsz);
		nbytes = fread(buf.data + buf.len, 1, filp_unitsz, filp);
		if (nbytes < filp_unitsz && ferror(filp) != 0)
			elog(ERROR, "could not read from \"%s\": %m", pathname);
	} while (nbytes == filp_unitsz);

	build_log = buf.data;
	build_log_len = buf.len;

	FreeFile(filp);

	/*
	 * Make a new entry, instead of the old one
	 */
	required = MAXALIGN(old_entry->cuda_source_len + 1);
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

	new_entry->crc = old_entry->crc;
	new_entry->extra_flags = old_entry->extra_flags;
	new_entry->cuda_source = new_entry->data;
	memcpy(new_entry->cuda_source,
		   old_entry->cuda_source,
		   old_entry->cuda_source_len + 1);
	new_entry->cuda_source_len = old_entry->cuda_source_len;

	if (!cuda_binary)
	{
		new_entry->cuda_binary = CUDA_PROGRAM_BUILD_FAILURE;
		new_entry->error_msg = (new_entry->cuda_source +
								MAXALIGN(new_entry->cuda_source_len + 1));
		new_entry->error_msg_len = PGCACHE_ERRORMSG_LEN(new_entry);
		snprintf(new_entry->error_msg,
				 new_entry->error_msg_len,
				 "cuda kernel build: failed\n%s\%s",
				 cmdline.data, build_log);
	}
	else
	{
		new_entry->cuda_binary = (new_entry->cuda_source +
								  MAXALIGN(new_entry->cuda_source_len + 1));
		new_entry->cuda_binary_len = cuda_binary_len;
		new_entry->error_msg = (new_entry->cuda_binary +
								MAXALIGN(new_entry->cuda_binary_len + 1));
		new_entry->error_msg_len = PGCACHE_ERRORMSG_LEN(new_entry);
		snprintf(new_entry->error_msg,
                 new_entry->error_msg_len,
				 "cuda kernel build: success\n%s\n%s",
				 cmdline.data, build_log);
	}

	/*
	 * Add new_entry to the hash slot
	 */
	hindex = new_entry->crc % PGCACHE_HASH_SIZE;
	dlist_push_head(&pgcache_head->active_list[hindex], &new_entry->chain);
	dlist_push_head(&pgcache_head->lru_list, &new_entry->lru_chain);

	/*
	 * Also, old_entry shall not be referenced no longer
	 */
	dlist_delete(&old_entry->chain);
	dlist_delete(&old_entry->lru_chain);
	memset(&old_entry->chain, 0, sizeof(dlist_node));
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
}



static void
pgstrom_build_cuda_program(Datum cuda_program)
{
	program_cache_entry *entry = (program_cache_entry *) cuda_program;

	Assert(entry->cuda_binary == NULL);

	PG_TRY();
	{
		__build_cuda_program(entry);
	}
	PG_CATCH();
	{
		ErrorData  *errdata = CopyErrorData();

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
		SpinLockRelease(&pgcache_head->lock);
		pgstrom_put_cuda_program(entry);
		pgstrom_wakeup_backends();
		PG_RE_THROW();
	}
	PG_END_TRY();
	pgstrom_put_cuda_program(entry);
	pgstrom_wakeup_backends();
}

bool
pgstrom_get_cuda_program(GpuTaskState *gts,
						 const char *source, int32 extra_flags)
{
	program_cache_entry	*entry;
	Size		source_len = strlen(source);
	Size		required;
	int			hindex;
	dlist_iter	iter;
	pg_crc32	crc;
	CUresult	rc;
	BackgroundWorker worker;

	/* makes a hash value */
	INIT_CRC32C(crc);
	COMP_CRC32C(crc, &extra_flags, sizeof(int32));
	COMP_CRC32C(crc, source, source_len);
	FIN_CRC32C(crc);

retry:
	hindex = crc % PGCACHE_HASH_SIZE;
	SpinLockAcquire(&pgcache_head->lock);
	dlist_foreach (iter, &pgcache_head->active_list[hindex])
	{
		entry = dlist_container(program_cache_entry, chain, iter.cur);

		if (entry->crc == crc &&
			entry->extra_flags == extra_flags &&
			entry->cuda_source_len == source_len &&
			strcmp(entry->cuda_source, source) == 0)
		{
			/* This kernel build already lead an error */
			if (entry->cuda_binary == CUDA_PROGRAM_BUILD_FAILURE)
			{
				SpinLockRelease(&pgcache_head->lock);
				elog(ERROR, "%s", entry->error_msg);
			}
			/* Still kernel build is in progress */
			if (!entry->cuda_binary)
			{
				SpinLockRelease(&pgcache_head->lock);
				return false;
			}
			/* OK, this kernel is already built */
			entry->refcnt++;
			dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);
			SpinLockRelease(&pgcache_head->lock);

			/* Also, load it on the private space */
			rc = cuModuleLoadData(&gts->cuda_module, entry->cuda_binary);
			if (rc != CUDA_SUCCESS)
			{
				SpinLockAcquire(&pgcache_head->lock);
				Assert(entry->refcnt > 0);
				if (--entry->refcnt == 0)
					pgstrom_program_cache_free(entry);
				SpinLockRelease(&pgcache_head->lock);

				elog(ERROR, "failed on cuModuleLoadData (%s)",
					 cuda_strerror(rc));
			}
			return true;
		}
	}
	/* Not found on the existing cache */
	required = offsetof(program_cache_entry, data[source_len + 1]);
	entry = pgstrom_program_cache_alloc(required);
	entry->crc = crc;
	entry->extra_flags = extra_flags;
	strcpy(entry->data, source);
	entry->cuda_source = entry->data;
	entry->cuda_source_len = source_len;
	entry->cuda_binary = NULL;
	entry->cuda_binary_len = 0;
	/* remaining area for error message */
	entry->error_msg = entry->data + MAXALIGN(source_len + 1);
	entry->error_msg_len = PGCACHE_ERRORMSG_LEN(entry);

	/* to be acquired by program builder */
	entry->refcnt++;
	dlist_push_head(&pgcache_head->active_list[hindex], &entry->chain);
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

void
pgstrom_put_cuda_program(GpuTaskState *gts)
{
	const program_cache_entry *pc_entry = gts->cuda_program;
	CUresult		rc;

	if (!pc_entry)
	{
		Assert(!gts->cuda_module);
		return;
	}

	SpinLockAcquire(&pgcache_head->lock);
	Assert(pc_entry->refcnt > 0);
	if (--pc_entry->refcnt == 0)
		pgstrom_program_cache_free(pc_entry);
	SpinLockRelease(&pgcache_head->lock);

	if (pc_entry->cuda_module)
	{
		rc = cuModuleUnload(pc_entry->cuda_module);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuModuleUnload : %s", cuda_strerror(rc));
	}
}

static void
pgstrom_startup_cuda_program(void)
{
	program_cache_entry *entry;
	bool		found;
	int			i, nwords;
	int			shift;
	char	   *curr_addr;
	char	   *end_addr;
	Bitmapset  *tempset;

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

	nwords = (TOTAL_PROCS + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;
	tempset = &pgcache_head->waiting_backends;
	tempset->nwords = nwords;
	memset(tempset->words, 0, sizeof(bitmapword) * nwords);
	pgcache_head->entry_begin =
		(program_cache_entry *) BUFFERALIGN(tempset->words + nwords);

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
		dlist_push_tail(&pgcache_head->free_list[shift], &entry->chain);

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
