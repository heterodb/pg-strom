/*
 * cuda_program.c
 *
 * Routines for just-in-time comple cuda code
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#include "pg_strom.h"
#include "mb/pg_wchar.h"
#include "access/xact.h"
#include "pgtime.h"
#include "utils/pg_locale.h"

typedef struct
{
	cl_int			magic;
	cl_int			mclass;
	dlist_node		free_chain;		/* zero clear, if active chunk */
	/* -- fields below are valid only if active chunks -- */
	dlist_node		pgid_chain;
	dlist_node		hash_chain;
	dlist_node		lru_chain;
	dlist_node		build_chain;
	cl_int			refcnt;
	/* fields below are never updated once entry is constructed */
	ProgramId		program_id;
	pg_crc32		crc;			/* hash value by extra_flags */
	int				target_cc;		/*             + target_cc   */
	cl_uint			extra_flags;	/*             + kern_define */
	char		   *kern_define;	/*             + kern_source */
	size_t			kern_deflen;
	char		   *kern_source;
	size_t			kern_srclen;
	cl_uint			varlena_bufsz;
	pg_crc32		ptx_crc;
	char		   *ptx_image;		/* may be CUDA_PROGRAM_BUILD_FAILURE */
	size_t			ptx_length;
	char		   *error_msg;
	int				error_code;
	char			data[FLEXIBLE_ARRAY_MEMBER];
} program_cache_entry;

#define PGCACHE_MIN_ERRORMSG_BUFSIZE	256

#define PGCACHE_CHUNKSZ_MAX_BIT		34		/* 16GB */
#define PGCACHE_CHUNKSZ_MIN_BIT		9		/* 512B */
#define PGCACHE_CHUNKSZ_MAX			(1UL << PGCACHE_CHUNKSZ_MAX_BIT)
#define PGCACHE_CHUNKSZ_MIN			(1UL << PGCACHE_CHUNKSZ_MIN_BIT)
#define PGCACHE_CHUNK_MAGIC			0xdeadbeaf

#define CUDA_PROGRAM_BUILD_FAILURE			((void *)(~0UL))

#define PGCACHE_HASH_SIZE	960
#define WORDNUM(x)			((x) / BITS_PER_BITMAPWORD)
#define BITNUM(x)			((x) % BITS_PER_BITMAPWORD)

typedef struct
{
	volatile slock_t lock;
	ProgramId	last_program_id;
	dlist_head	pgid_slots[PGCACHE_HASH_SIZE];
	dlist_head	hash_slots[PGCACHE_HASH_SIZE];
	dlist_head	lru_list;
	dlist_head	build_list;		/* build pending list */
	dlist_head	addr_list;
	dlist_head	free_list[PGCACHE_CHUNKSZ_MAX_BIT + 1];
	char		base[FLEXIBLE_ARRAY_MEMBER];
} program_cache_head;

typedef struct
{
	pg_atomic_uint32	num_active_builders;
	struct {
		Latch *latch;
	} builders[FLEXIBLE_ARRAY_MEMBER];
} program_builder_state;

/* ---- GUC variables ---- */
static int		program_cache_size_kb;
static int		num_program_builders;
static bool		pgstrom_debug_jit_compile_options;

/* ---- static variables ---- */
static shmem_startup_hook_type shmem_startup_next;
static program_cache_head *pgcache_head = NULL;
static program_builder_state *pgbuilder_state = NULL;
static bool		cuda_program_builder_got_signal = false;

/* ---- forward declarations ---- */
static void put_cuda_program_entry_nolock(program_cache_entry *entry);
void cudaProgramBuilderMain(Datum arg);
static void cudaProgramBuilderWakeUp(bool error_if_no_builders);

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
 * split_cuda_program_entry_nolock
 */
static bool
split_cuda_program_entry_nolock(int mclass)
{
	program_cache_entry *entry1;
	program_cache_entry *entry2;
	dlist_node	   *dnode;

	if (mclass > PGCACHE_CHUNKSZ_MAX_BIT)
		return false;
	Assert(mclass > PGCACHE_CHUNKSZ_MIN_BIT);
	if (dlist_is_empty(&pgcache_head->free_list[mclass]))
	{
		if (!split_cuda_program_entry_nolock(mclass + 1))
			return false;
	}
	Assert(!dlist_is_empty(&pgcache_head->free_list[mclass]));
	dnode = dlist_pop_head_node(&pgcache_head->free_list[mclass]);
	entry1 = dlist_container(program_cache_entry, free_chain, dnode);
	Assert(entry1->magic == PGCACHE_CHUNK_MAGIC &&
		   entry1->mclass == mclass);
	entry2 = (program_cache_entry *)((char *)entry1 + (1UL << (mclass - 1)));
	memset(entry2, 0, offsetof(program_cache_entry, data));

	mclass--;
	entry1->magic = PGCACHE_CHUNK_MAGIC;
	entry1->mclass = mclass;
	entry2->magic = PGCACHE_CHUNK_MAGIC;
	entry2->mclass = mclass;
	dlist_push_head(&pgcache_head->free_list[mclass], &entry1->free_chain);
	dlist_push_head(&pgcache_head->free_list[mclass], &entry2->free_chain);

	return true;
}

/*
 * reclaim_cuda_program_entry_nolock(int mclass)
 */
static bool
reclaim_cuda_program_entry_nolock(int mclass)
{
	program_cache_entry *entry;
	dlist_node	   *dnode;

	if (dlist_is_empty(&pgcache_head->lru_list))
		return false;
	dnode = dlist_tail_node(&pgcache_head->lru_list);

	while (dlist_is_empty(&pgcache_head->free_list[mclass]) &&
		   !split_cuda_program_entry_nolock(mclass + 1))
	{
	prev_entry:
		entry = dlist_container(program_cache_entry, lru_chain, dnode);
		if (entry->refcnt > 0)
		{
			if (dlist_has_prev(&pgcache_head->lru_list, dnode))
			{
				dnode = dlist_prev_node(&pgcache_head->lru_list, dnode);
				goto prev_entry;
			}
			else
				return false;
		}
		dlist_delete(&entry->pgid_chain);
		dlist_delete(&entry->hash_chain);
		dlist_delete(&entry->lru_chain);
		memset(&entry->pgid_chain, 0, sizeof(dlist_node));
		memset(&entry->hash_chain, 0, sizeof(dlist_node));
		memset(&entry->lru_chain, 0, sizeof(dlist_node));

		put_cuda_program_entry_nolock(entry);
	}
	return true;
}

/*
 * create_cuda_program_entry_nolock
 */
static program_cache_entry *
create_cuda_program_entry_nolock(size_t variable_length)
{
	program_cache_entry *entry;
	dlist_node *dnode;
	size_t		required;
	cl_int		mclass;

	required = offsetof(program_cache_entry, data) + variable_length;
	mclass = get_next_log2(required);
	if (mclass < PGCACHE_CHUNKSZ_MIN_BIT)
		mclass = PGCACHE_CHUNKSZ_MIN_BIT;
	if (mclass > PGCACHE_CHUNKSZ_MAX_BIT)
		return NULL;	/* invalid length */

	if (dlist_is_empty(&pgcache_head->free_list[mclass]))
	{
		if (!split_cuda_program_entry_nolock(mclass + 1))
		{
			/* out of memory!, reclaim recently unused ones */
			if (!reclaim_cuda_program_entry_nolock(mclass))
				return NULL;	/* Oops! */
		}
	}
	Assert(!dlist_is_empty(&pgcache_head->free_list[mclass]));
	dnode = dlist_pop_head_node(&pgcache_head->free_list[mclass]);
	entry = dlist_container(program_cache_entry, free_chain, dnode);
	Assert(entry->magic == PGCACHE_CHUNK_MAGIC &&
		   entry->mclass == mclass);
	memset(&entry->free_chain, 0, sizeof(dlist_node));

	return entry;
}

/*
 * get_cuda_program_entry_nolock
 */
static void
get_cuda_program_entry_nolock(program_cache_entry *entry)
{
	Assert(entry->refcnt > 0);
	entry->refcnt++;
	if (entry->lru_chain.prev && entry->lru_chain.next)
		dlist_move_head(&pgcache_head->lru_list,
						&entry->lru_chain);
}

/*
 * put_cuda_program_entry_nolock
 */
static void
put_cuda_program_entry_nolock(program_cache_entry *entry)
{
	program_cache_entry *buddy;
	size_t		limit = (size_t)program_cache_size_kb << 10;
	size_t		offset;

	if (--entry->refcnt > 0)
		return;

	/*
	 * NOTE: unless program_cache_entry was not detached, refcnt shall
	 * never be zero, so we confirm the entry is already detached.
	 */
	Assert(!entry->pgid_chain.next && !entry->pgid_chain.prev);
	Assert(!entry->hash_chain.next && !entry->hash_chain.prev);
	Assert(!entry->lru_chain.next && !entry->lru_chain.prev);

	while (entry->mclass < PGCACHE_CHUNKSZ_MAX_BIT)
	{
		offset = ((uintptr_t)entry - (uintptr_t)pgcache_head->base);
		if ((offset & (1UL << entry->mclass)) == 0)
		{
			buddy = (program_cache_entry *)
				((char *)entry + (1UL << entry->mclass));
			if ((uintptr_t)buddy >= (uintptr_t)pgcache_head->base + limit)
				break;		/* out of the range */
			Assert(buddy->magic == PGCACHE_CHUNK_MAGIC);
			if (!buddy->free_chain.prev ||		/* active? */
				!buddy->free_chain.next ||		/* active? */
				buddy->mclass != entry->mclass)	/* same size? */
				break;
			Assert(!buddy->pgid_chain.next && !buddy->pgid_chain.prev);
			Assert(!buddy->hash_chain.next && !buddy->hash_chain.prev);
			Assert(!buddy->lru_chain.next  && !buddy->lru_chain.prev);

			dlist_delete(&buddy->free_chain);
			entry->mclass++;
		}
		else
		{
			buddy = (program_cache_entry *)
				((char *)entry - (1UL << entry->mclass));
			if ((uintptr_t)buddy < (uintptr_t)pgcache_head->base)
				break;		/* out of the range */
			Assert(buddy->magic == PGCACHE_CHUNK_MAGIC);
			if (!buddy->free_chain.prev ||		/* active? */
				!buddy->free_chain.next ||		/* active? */
				buddy->mclass != entry->mclass)	/* same size? */
				break;
			Assert(!buddy->pgid_chain.next && !buddy->pgid_chain.prev);
			Assert(!buddy->hash_chain.next && !buddy->hash_chain.prev);
			Assert(!buddy->lru_chain.next  && !buddy->lru_chain.prev);
			dlist_delete(&buddy->free_chain);
			entry = buddy;
			entry->mclass++;
		}
	}
	/* back to the free list */
	dlist_push_head(&pgcache_head->free_list[entry->mclass],
					&entry->free_chain);
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
 * construct_flat_cuda_source
 */
static char *
construct_flat_cuda_source(cl_uint extra_flags,
						   cl_uint varlena_bufsz,
						   const char *kern_define,
						   const char *kern_source)
{
	size_t		ofs = 0;
	size_t		len = strlen(kern_define) + strlen(kern_source) + 25000;
	char	   *source;

	source = malloc(len);
	if (!source)
		return NULL;

	ofs += snprintf(source + ofs, len - ofs,
					"#include <cuda_device_runtime_api.h>\n"
					"#define KERN_CONTEXT_VARLENA_BUFSZ %u\n",
					Max(varlena_bufsz, 1));
	/* Enables Debug build? */
	if ((extra_flags & DEVKERNEL_BUILD_DEBUG_INFO) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#define PGSTROM_KERNEL_DEBUG 1\n");
	/* Common PG-Strom device routine */
	ofs += snprintf(source + ofs, len - ofs,
					"#include \"cuda_common.h\"\n");
	/* Per session definition if any */
	ofs += snprintf(source + ofs, len - ofs,
					"\n%s\n", kern_define);

	/*
	 * PG-Strom CUDA device code libraries
	 */
	/* cuda_jsonlib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_JSONLIB) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_jsonlib.h\"\n");
	/* cuda_misclib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_MISCLIB) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_misclib.h\"\n");
	/* cuda_rangetypes.h */
	if ((extra_flags & DEVKERNEL_NEEDS_RANGETYPE) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_rangetype.h\"\n");
	/* cuda_primitive.h */
	if ((extra_flags & DEVKERNEL_NEEDS_PRIMITIVE) != 0)
		ofs += snprintf(source + ofs, len - ofs,
                        "#include \"cuda_primitive.h\"\n");
	/* GpuScan */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUSCAN) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_gpuscan.h\"\n");
	/* GpuJoin */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_gpujoin.h\"\n");
	/* GpuPreAgg */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUPREAGG) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_gpupreagg.h\"\n");
	/* GpuSort */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUSORT) != 0)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_gpusort.h\"\n");
	/* Generated from SQL */
	ofs += snprintf(source + ofs, len - ofs, "\n%s\n", kern_source);

	return source;
}

/*
 * link_cuda_libraries - links CUDA libraries with the supplied PTX binary
 */
static char *
link_cuda_libraries(char *ptx_image,
					size_t ptx_length,
					cl_uint extra_flags)
{
	CUlinkState		lstate;
	CUresult		rc;
	CUjit_option	jit_options[16];
	void		   *jit_option_values[16];
	int				jit_index = 0;
	void		   *temp;
	void		   *bin_image;
	size_t			bin_length;
	const char	   *lib_suffix = "fatbin";
	char			pathname[MAXPGPATH];
	char			log_buffer[16384];

	/*
	 * NOTE: cuLinkXXXX() APIs works under a particular CUDA context,
	 * so it must be processed by the process which has a valid CUDA
	 * context; that is GPU intermediation server.
	 */

	/*
	 * JIT Options
	 */
	/* Limit max number of registers per threads for ABI compatibility */
	jit_options[jit_index] = CU_JIT_MAX_REGISTERS;
	jit_option_values[jit_index] = (void *)CUDA_MAXREGCOUNT;
	jit_index++;

	/* Get optimal binary to the current context */
	jit_options[jit_index] = CU_JIT_TARGET_FROM_CUCONTEXT;
	jit_option_values[jit_index] = NULL;
	jit_index++;

	/* Compile with L1 cache enabled */
	jit_options[jit_index] = CU_JIT_CACHE_MODE;
	jit_option_values[jit_index] = (void *)CU_JIT_CACHE_OPTION_CA;
	jit_index++;

	/* Enables debug options if required */
	if ((extra_flags & DEVKERNEL_BUILD_DEBUG_INFO) != 0)
	{
		jit_options[jit_index] = CU_JIT_GENERATE_DEBUG_INFO;
		jit_option_values[jit_index] = (void *)1UL;
		jit_index++;

		jit_options[jit_index] = CU_JIT_GENERATE_LINE_INFO;
		jit_option_values[jit_index] = (void *)1UL;
		jit_index++;
#if 0
		jit_options[jit_index] = CU_JIT_OPTIMIZATION_LEVEL;
		jit_option_values[jit_index] = (void *)0;
		jit_index++;
#endif
		/* link libraries with debug options */
		lib_suffix = "gfatbin";
	}
	/* Link log buffer */
	jit_options[jit_index] = CU_JIT_ERROR_LOG_BUFFER;
	jit_option_values[jit_index] = (void *)log_buffer;
	jit_index++;

	jit_options[jit_index] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
	jit_option_values[jit_index] = (void *)sizeof(log_buffer);
	jit_index++;

	/* makes a linkage object */
	rc = cuLinkCreate(jit_index, jit_options, jit_option_values, &lstate);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLinkCreate: %s", errorText(rc));

	STROM_TRY();
	{
		static struct {
			const char *libname;
			cl_int		libflags;
		} catalog[] = {
			{ "cuda_common", 0 },
			{ "cuda_numeric", 0 },
			{ "cuda_primitive", DEVKERNEL_NEEDS_PRIMITIVE },
			{ "cuda_textlib",   DEVKERNEL_NEEDS_TEXTLIB },
			{ "cuda_timelib",   DEVKERNEL_NEEDS_TIMELIB },
			{ "cuda_misclib",   DEVKERNEL_NEEDS_MISCLIB },
			{ "cuda_jsonlib",   DEVKERNEL_NEEDS_JSONLIB },
			{ "cuda_gpuscan",   DEVKERNEL_NEEDS_GPUSCAN },
			{ "cuda_gpujoin",   DEVKERNEL_NEEDS_GPUJOIN },
			{ "cuda_gpupreagg", DEVKERNEL_NEEDS_GPUPREAGG },
			{ "cuda_gpusort",   DEVKERNEL_NEEDS_GPUSORT },
			{ NULL, 0 },
		};
		cl_int		i;

		/* add the base PTX image */
		rc = cuLinkAddData(lstate, CU_JIT_INPUT_PTX,
						   ptx_image, ptx_length,
						   "pg-strom", 0, NULL, NULL);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuLinkAddData: %s", errorText(rc));

		/* other libraries */
		for (i=0; catalog[i].libname != NULL; i++)
		{
			if ((extra_flags & catalog[i].libflags) == catalog[i].libflags)
			{
				snprintf(pathname, sizeof(pathname),
						 PGSHAREDIR "/pg_strom/%s.%s",
						 catalog[i].libname, lib_suffix);
				rc = cuLinkAddFile(lstate, CU_JIT_INPUT_FATBINARY,
								   pathname, 0, NULL, NULL);
				if (rc != CUDA_SUCCESS)
					werror("failed on cuLinkAddFile(\"%s\"): %s",
						   pathname, errorText(rc));
			}
		}

		/* do the linkage */
		rc = cuLinkComplete(lstate, &temp, &bin_length);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuLinkComplete: %s\nLog: %s",
				   errorText(rc), log_buffer);

		/*
		 * copy onto the result buffer; because bin_image is owned by
		 * CUlinkState, so cuLinkDestroy also releases the bin_image.
		 */
		bin_image = malloc(bin_length);
		if (!bin_image)
			werror("out of memory");
		memcpy(bin_image, temp, bin_length);
	}
	STROM_CATCH();
	{
		rc = cuLinkDestroy(lstate);
		if (rc != CUDA_SUCCESS)
			wnotice("failed on cuLinkDestroy: %s", errorText(rc));
		STROM_RE_THROW();
	}
	STROM_END_TRY();

	rc = cuLinkDestroy(lstate);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLinkDestroy: %s", errorText(rc));

	return bin_image;
}


/*
 * writeout_temporary_file
 *
 * It makes a temporary file to write-out cuda source.
 */
static void
writeout_temporary_file(char *tempfile, const char *suffix,
						const char *source, size_t length)
{
	static pg_atomic_uint64 sourceFileCounter = {0};
	char		tempdir[MAXPGPATH];
	FILE	   *filp;

	/*
	 * Generate a tempfile name that should be unique within the current
	 * database instance.
	 */
	snprintf(tempdir, MAXPGPATH, "%s/%s",
			 DataDir,
			 PG_TEMP_FILES_DIR);
	snprintf(tempfile, MAXPGPATH, "%s/%s_strom_%d.%ld.%s",
			 tempdir,
			 PG_TEMP_FILE_PREFIX,
			 MyProcPid,
			 pg_atomic_fetch_add_u64(&sourceFileCounter, 1),
			 suffix);
	/* Open the temporary file */
	filp = fopen(tempfile, "w+b");
	if (!filp)
	{
		mkdir(tempdir, S_IRWXU);

		filp = fopen(tempfile, "w+b");
		if (!filp)
		{
			snprintf(tempfile, MAXPGPATH,
					 "!!unable open temporary file!! (%m)");
			return;
		}
	}
	fputs(source, filp);
	fclose(filp);
}

/*
 * pgstrom_cuda_source_string
 *
 * NOTE: construct_flat_cuda_source() returns cstring which is allocated
 * with malloc(3), so caller must ensure to release this buffer.
 */
char *
pgstrom_cuda_source_string(ProgramId program_id)
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
										entry->varlena_bufsz,
										entry->kern_define,
										entry->kern_source);
	put_cuda_program_entry(entry);
	if (!source)
		elog(ERROR, "out of memory");

	return source;
}

/*
 * pgstrom_cuda_source_file - write out a CUDA source program to temporary file
 */
const char *
pgstrom_cuda_source_file(ProgramId program_id)
{
	char	tempfilepath[MAXPGPATH];
	char   *source;

	source = pgstrom_cuda_source_string(program_id);
	writeout_temporary_file(tempfilepath, "gpu",
							source, strlen(source));
	free(source);

	return pstrdup(tempfilepath);
}

/*
 * pgstrom_cuda_binary_file - write out a CUDA PTX binary to temporary file
 */
const char *
pgstrom_cuda_binary_file(ProgramId program_id)
{
	program_cache_entry *entry;
	char		tempfilepath[MAXPGPATH];

	SpinLockAcquire(&pgcache_head->lock);
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "ProgramId=%lu not found", program_id);
	}
	else if (!entry->ptx_image ||
			 entry->ptx_image == CUDA_PROGRAM_BUILD_FAILURE)
	{
		SpinLockRelease(&pgcache_head->lock);
		return NULL;
	}
	get_cuda_program_entry_nolock(entry);
	SpinLockRelease(&pgcache_head->lock);

	writeout_temporary_file(tempfilepath, "ptx",
							entry->ptx_image, entry->ptx_length);
	put_cuda_program_entry(entry);

	return pstrdup(tempfilepath);
}

/*
 * build_cuda_program - an interface to run synchronous build process
 */
static program_cache_entry *
build_cuda_program(program_cache_entry *src_entry)
{
	program_cache_entry *bin_entry = NULL;
	nvrtcProgram	program = NULL;
	nvrtcResult		rc;
	char		   *source = NULL;
	char			tempfile[MAXPGPATH];
	const char	   *options[16];
	int				opt_index = 0;
	char		   *ptx_image = NULL;
	size_t			ptx_length = 0;
	char		   *build_log = NULL;
	size_t			log_length;
	int				pindex;
	int				hindex;
	size_t			offset;
	size_t			length;

	Assert(!src_entry->build_chain.prev && !src_entry->build_chain.next);

	/* Make a nvrtcProgram object */
	source = construct_flat_cuda_source(src_entry->extra_flags,
										src_entry->varlena_bufsz,
										src_entry->kern_define,
										src_entry->kern_source);
	if (!source)
		werror("out of memory");

	STROM_TRY();
	{
		char	gpu_arch_option[256];

		rc = nvrtcCreateProgram(&program,
								source,
								"pg-strom",
								0,
								NULL,
								NULL);
		if (rc != NVRTC_SUCCESS)
			werror("failed on nvrtcCreateProgram: %s",
				   nvrtcGetErrorString(rc));
		/*
		 * Put command line options
		 */
		options[opt_index++] = "-D__" CPU_ARCH "__=1";
		options[opt_index++] = "-I " CUDA_INCLUDE_PATH;
		options[opt_index++] = "-I " PGSHAREDIR "/pg_strom";
		options[opt_index++] = "-I " PGSERV_INCLUDEDIR;
		snprintf(gpu_arch_option, sizeof(gpu_arch_option),
				 "--gpu-architecture=compute_%u", src_entry->target_cc);
		options[opt_index++] = gpu_arch_option;
		if ((src_entry->extra_flags & DEVKERNEL_BUILD_DEBUG_INFO) != 0)
		{
			options[opt_index++] = "--device-debug";
			options[opt_index++] = "--generate-line-info";
		}
		options[opt_index++] = "--use_fast_math";
		/* library linkage needs relocatable PTX */
		options[opt_index++] = "--device-c";
		/* enables c++11 template features */
		options[opt_index++] = "--std=c++11";

		/*
		 * Kick runtime compiler
		 */
		rc = nvrtcCompileProgram(program, opt_index, options);
		if (rc == NVRTC_ERROR_COMPILATION)
		{
			writeout_temporary_file(tempfile, "gpu",
									source, strlen(source));
		}
		else if (rc != NVRTC_SUCCESS)
		{
			werror("failed on nvrtcCompileProgram: %s",
				   nvrtcGetErrorString(rc));
		}
		else
		{
			/*
			 * Read PTX Binary
			 */
			rc = nvrtcGetPTXSize(program, &ptx_length);
			if (rc != NVRTC_SUCCESS)
				werror("failed on nvrtcGetPTXSize: %s",
					   nvrtcGetErrorString(rc));
			ptx_image = malloc(ptx_length + 1);
			if (!ptx_image)
				werror("out of memory");

			rc = nvrtcGetPTX(program, ptx_image);
			if (rc != NVRTC_SUCCESS)
				werror("failed on nvrtcGetPTX: %s",
					   nvrtcGetErrorString(rc));
			ptx_image[ptx_length++] = '\0';
		}

		/*
		 * Read Log Output
		 */
		rc = nvrtcGetProgramLogSize(program, &log_length);
		if (rc != NVRTC_SUCCESS)
			werror("failed on nvrtcGetProgramLogSize: %s",
				   nvrtcGetErrorString(rc));
		build_log = malloc(log_length + 1);
		if (!build_log)
			werror("out of memory");

		rc = nvrtcGetProgramLog(program, build_log);
		if (rc != NVRTC_SUCCESS)
			werror("failed on nvrtcGetProgramLog: %s",
				   nvrtcGetErrorString(rc));
		build_log[log_length] = '\0';	/* may not be necessary? */

		/* release nvrtcProgram object */
		rc = nvrtcDestroyProgram(&program);
		if (rc != NVRTC_SUCCESS)
			werror("failed on nvrtcDestroyProgram: %s",
				   nvrtcGetErrorString(rc));

		/*
		 * Allocation of a new entry, to keep ptx_image/build_log
		 */
		length = (MAXALIGN(src_entry->kern_deflen + 1) +
				  MAXALIGN(src_entry->kern_srclen + 1) +
				  MAXALIGN(ptx_length + 1) +
				  MAXALIGN(log_length + 1) +
				  PGCACHE_MIN_ERRORMSG_BUFSIZE);
		SpinLockAcquire(&pgcache_head->lock);
		bin_entry = create_cuda_program_entry_nolock(length);
		if (!bin_entry)
		{
			SpinLockRelease(&pgcache_head->lock);
			werror("out of CUDA program cache");
		}
		/*
		 * OK, replace the src_entry by the bin_entry
		 */
		offset = 0;
		bin_entry->program_id		= src_entry->program_id;
		bin_entry->crc				= src_entry->crc;
		bin_entry->target_cc        = src_entry->target_cc;
		bin_entry->extra_flags		= src_entry->extra_flags;
		bin_entry->kern_deflen		= src_entry->kern_deflen;
		bin_entry->kern_define		= bin_entry->data + offset;
		strcpy(bin_entry->kern_define, src_entry->kern_define);
		offset += MAXALIGN(bin_entry->kern_deflen + 1);

		bin_entry->kern_srclen		= src_entry->kern_srclen;
		bin_entry->kern_source		= bin_entry->data + offset;
		strcpy(bin_entry->kern_source, src_entry->kern_source);
		offset += MAXALIGN(bin_entry->kern_srclen + 1);
		bin_entry->varlena_bufsz	= src_entry->varlena_bufsz;

		if (!ptx_image)
			bin_entry->ptx_image	= CUDA_PROGRAM_BUILD_FAILURE;
		else
		{
			pg_crc32	ptx_crc;

			bin_entry->ptx_image	= bin_entry->data + offset;
			bin_entry->ptx_length	= ptx_length;
			memcpy(bin_entry->ptx_image, ptx_image, ptx_length);
			offset += MAXALIGN(ptx_length);

			INIT_LEGACY_CRC32(ptx_crc);
			COMP_LEGACY_CRC32(ptx_crc, ptx_image, ptx_length);
			FIN_LEGACY_CRC32(ptx_crc);
			bin_entry->ptx_crc		= ptx_crc;
		}

		bin_entry->error_msg		= bin_entry->data + offset;
		if (!ptx_image)
			snprintf(bin_entry->error_msg, length - offset,
					 "build failure:\n%s\nsource: %s",
					 build_log, tempfile);
		else
			snprintf(bin_entry->error_msg, length - offset,
					 "build success:\n%s\n",
					 build_log);
		/* OK, bin_entry was built */
		pindex = bin_entry->program_id % PGCACHE_HASH_SIZE;
		hindex = bin_entry->crc % PGCACHE_HASH_SIZE;
		dlist_push_head(&pgcache_head->pgid_slots[pindex],
						&bin_entry->pgid_chain);
		dlist_push_head(&pgcache_head->hash_slots[hindex],
						&bin_entry->hash_chain);
		dlist_push_head(&pgcache_head->lru_list,
						&bin_entry->lru_chain);
		memset(&bin_entry->build_chain, 0, sizeof(dlist_node));
		bin_entry->refcnt = src_entry->refcnt;
		/* release src_entry */
		src_entry->refcnt = 0;
		dlist_delete(&src_entry->pgid_chain);
		dlist_delete(&src_entry->hash_chain);
		dlist_delete(&src_entry->lru_chain);
		memset(&src_entry->pgid_chain, 0, sizeof(dlist_node));
		memset(&src_entry->hash_chain, 0, sizeof(dlist_node));
		memset(&src_entry->lru_chain, 0, sizeof(dlist_node));
		put_cuda_program_entry_nolock(src_entry);
		SpinLockRelease(&pgcache_head->lock);
	}
	STROM_CATCH();
	{
		if (build_log)
			free(build_log);
		if (ptx_image)
			free(ptx_image);
		if (program)
		{
			rc = nvrtcDestroyProgram(&program);
			if (rc != NVRTC_SUCCESS)
				wnotice("failed on nvrtcDestroyProgram: %s",
						nvrtcGetErrorString(rc));
		}
		if (source)
			free(source);
		STROM_RE_THROW();
	}
	STROM_END_TRY();
	if (build_log)
		free(build_log);
	if (ptx_image)
		free(ptx_image);
	free(source);

	return bin_entry;
}

/*
 * pgstrom_create_cuda_program
 *
 * It makes a new GPU program cache entry, or acquires an existing entry if
 * equivalent one is already exists.
 */
ProgramId
__pgstrom_create_cuda_program(GpuContext *gcontext,
							  cl_uint extra_flags,
							  cl_uint varlena_bufsz,
							  const char *kern_source,
							  const char *kern_define,
							  bool wait_for_build,
							  bool explain_only,
							  const char *filename, int lineno)
{
	program_cache_entry	*entry;
	ProgramId	program_id;
	Size		kern_srclen = strlen(kern_source);
	Size		kern_deflen = strlen(kern_define);
	Size		length;
	Size		usage = 0;
	int			dindex = gcontext->cuda_dindex;
	int			hindex;
	cl_int		target_cc;
	dlist_iter	iter;
	pg_crc32	crc;

	/* build with debug option? */
	if (pgstrom_debug_jit_compile_options)
		extra_flags |= DEVKERNEL_BUILD_DEBUG_INFO;
	/* target binary to build */
	Assert(dindex >= 0 && dindex < numDevAttrs);
	target_cc = (devAttrs[dindex].COMPUTE_CAPABILITY_MAJOR * 10 +
				 devAttrs[dindex].COMPUTE_CAPABILITY_MINOR);
	/* makes a hash value */
	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &target_cc, sizeof(cl_int));
	COMP_LEGACY_CRC32(crc, &extra_flags, sizeof(int32));
	COMP_LEGACY_CRC32(crc, kern_source, kern_srclen);
	COMP_LEGACY_CRC32(crc, kern_define, kern_deflen);
	FIN_LEGACY_CRC32(crc);

	hindex = crc % PGCACHE_HASH_SIZE;
	SpinLockAcquire(&pgcache_head->lock);
	dlist_foreach (iter, &pgcache_head->hash_slots[hindex])
	{
		bool		kick_builders = false;

		entry = dlist_container(program_cache_entry, hash_chain, iter.cur);

		if (entry->crc == crc &&
			entry->target_cc == target_cc &&
			entry->extra_flags == extra_flags &&
			strcmp(entry->kern_source, kern_source) == 0 &&
			strcmp(entry->kern_define, kern_define) == 0 &&
			entry->varlena_bufsz >= varlena_bufsz)
		{
			program_id = entry->program_id;
			get_cuda_program_entry_nolock(entry);
			/* Move this entry to the head of LRU list */
			dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);
		retry_checks:
			if (entry->ptx_image != NULL || !wait_for_build)
			{
				if (!trackCudaProgram(gcontext, program_id,
									  filename, lineno))
				{
					put_cuda_program_entry_nolock(entry);
					SpinLockRelease(&pgcache_head->lock);
					elog(ERROR, "out of memory");
				}

				/* Raise syntax error if already failed on build */
				if (!explain_only &&
					entry->ptx_image == CUDA_PROGRAM_BUILD_FAILURE)
				{
					SpinLockRelease(&pgcache_head->lock);
					ereport(ERROR,
							(errcode(entry->error_code),
							 errmsg("%s", entry->error_msg)));
				}

				/* Kick program builders if not built yet */
				if (!entry->ptx_image &&
					(entry->build_chain.prev != NULL ||
					 entry->build_chain.next != NULL))
					kick_builders = true;
				SpinLockRelease(&pgcache_head->lock);
				if (kick_builders)
					cudaProgramBuilderWakeUp(!explain_only);
                return program_id;
			}
			else
			{
				kick_builders = (entry->build_chain.prev != NULL ||
								 entry->build_chain.next != NULL);

				SpinLockRelease(&pgcache_head->lock);

				/*
				 * Run-time compilation of this GPU kernel is still in-progress
				 * or pending to build, and caller wants to synchronize the
				 * completion of the code build
				 */
				if (kick_builders)
					cudaProgramBuilderWakeUp(true);

				pg_usleep(50000);	/* short sleep (50ms) */

				CHECK_FOR_GPUCONTEXT(gcontext);

				/* check again */
				SpinLockAcquire(&pgcache_head->lock);
				entry = lookup_cuda_program_entry_nolock(program_id);
				Assert(entry != NULL);
				goto retry_checks;
			}
		}
	}

	/*
	 * Not found on the existing program cache.
	 * So, create a new entry then kick NVRTC
	 */
	length = (MAXALIGN(kern_srclen + 1) +
			  MAXALIGN(kern_deflen + 1) +
			  PGCACHE_MIN_ERRORMSG_BUFSIZE);
	entry = create_cuda_program_entry_nolock(length);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		werror("out of shared memory");
	}

	/* find out a unique program_id */
	do {
		program_id = ++pgcache_head->last_program_id;
	} while (lookup_cuda_program_entry_nolock(program_id) != NULL);

	entry->program_id  = program_id;
	entry->crc         = crc;
	entry->target_cc   = target_cc;
	entry->extra_flags = extra_flags;
	entry->kern_define = (char *)(entry->data + usage);
	entry->kern_deflen = kern_deflen;
	memcpy(entry->kern_define, kern_define, kern_deflen + 1);
	usage += MAXALIGN(kern_deflen + 1);

	entry->kern_source = (char *)(entry->data + usage);
	entry->kern_srclen = kern_srclen;
	memcpy(entry->kern_source, kern_source, kern_srclen + 1);
	usage += MAXALIGN(kern_srclen + 1);

	/*
	 * An extra margin on the @varlena_bufsz might be valuable to avoid
	 * unnecessary program rebuild if program contains device functions
	 * that can return varlena datum. Because @varlena_bufsz estimation
	 * can be affected by small changes in query;
	 * e.g, substring(X from 0 for 3) will make different value from
	 * the substring(X from 1 for 4), but code itself shall not be
	 * changed. So, extra margin will help the case.
	 */
	if (varlena_bufsz == 0)
		entry->varlena_bufsz = 0;
	else
		entry->varlena_bufsz = MAXALIGN(varlena_bufsz + 36);

	/* no cuda binary at this moment */
	entry->ptx_image = NULL;
	entry->ptx_length = 0;
	/* remaining are for error message */
	entry->error_msg = (char *)(entry->data + usage);

	/* add an entry for program build in-progress */
	dlist_push_head(&pgcache_head->pgid_slots[program_id % PGCACHE_HASH_SIZE],
					&entry->pgid_chain);
	dlist_push_head(&pgcache_head->hash_slots[hindex],
					&entry->hash_chain);
	dlist_push_head(&pgcache_head->lru_list,
					&entry->lru_chain);
	dlist_push_head(&pgcache_head->build_list,
					&entry->build_chain);
	entry->refcnt = 2;	/* reference count (entry itself and owner) */

	/* track this program entry by GpuContext */
	if (!trackCudaProgram(gcontext, program_id,
						  filename, lineno))
	{
		put_cuda_program_entry_nolock(entry);
		SpinLockRelease(&pgcache_head->lock);
		elog(ERROR, "out of memory");
	}
	SpinLockRelease(&pgcache_head->lock);

	/*
	 * Start asynchronous code build with NVRTC
	 */
	cudaProgramBuilderWakeUp(wait_for_build);

	/* wait for completion of build, if needed */
	while (wait_for_build)
	{
		bool	kick_builders = false;

		SpinLockAcquire(&pgcache_head->lock);
		entry = lookup_cuda_program_entry_nolock(program_id);
		if (!entry)
		{
			SpinLockRelease(&pgcache_head->lock);
			elog(ERROR, "Bug? ProgramId=%lu has gone.", program_id);
		}
		else if (entry->ptx_image == CUDA_PROGRAM_BUILD_FAILURE)
		{
			SpinLockRelease(&pgcache_head->lock);
			ereport(ERROR,
					(errcode(entry->error_code),
					 errmsg("%s", entry->error_msg)));
		}
		else if (entry->ptx_image)
		{
			SpinLockRelease(&pgcache_head->lock);
			break;
		}
		else if (entry->build_chain.prev != NULL ||
				 entry->build_chain.next != NULL)
		{
			kick_builders = true;
		}
		/* NVRTC on this GPU program is still in-progress */
		SpinLockRelease(&pgcache_head->lock);

		if (kick_builders)
			cudaProgramBuilderWakeUp(true);

		CHECK_FOR_GPUCONTEXT(gcontext);

		pg_usleep(50000L);		/* short sleep */
	}
	return program_id;
}

/*
 * pgstrom_put_cuda_program
 *
 * release an existing GPU program entry
 */
void
pgstrom_put_cuda_program(GpuContext *gcontext, ProgramId program_id)
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
 * session info for libgputext
 */
static void
assign_textlib_session_info(StringInfo buf)
{
	appendStringInfoString(
		buf,
		"/* ================================================\n"
		" * session information for device text library\n"
		" * ================================================ */\n");
	/*
	 * Max character bytes in this encoding
	 */
	appendStringInfo(
		buf,
		"DEVICE_FUNCTION(cl_int)\n"
		"pg_database_encoding_max_length(void)\n"
		"{\n"
		"  return %d;\n"
		"}\n\n", pg_database_encoding_max_length());

	/*
	 * Makes encoding aware character length function
	 */
	appendStringInfoString(
		buf,
		"DEVICE_FUNCTION(cl_int)\n"
		"pg_wchar_mblen(const char *str)\n"
		"{\n");

	switch (GetDatabaseEncoding())
	{
		case PG_EUC_JP:		/* logic in pg_euc_mblen() */
		case PG_EUC_KR:
		case PG_EUC_TW:		/* logic in pg_euctw_mblen(), but identical */
		case PG_EUC_JIS_2004:
		case PG_JOHAB:		/* logic in pg_johab_mblen(), but identical */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c == 0x8e)\n"
				"    return 2;\n"
				"  else if (c == 0x8f)\n"
				"    return 3;\n"
				"  else if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_EUC_CN:		/* logic in pg_euccn_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_UTF8:		/* logic in pg_utf_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if ((c & 0x80) == 0)\n"
				"    return 1;\n"
				"  else if ((c & 0xe0) == 0xc0)\n"
				"    return 2;\n"
				"  else if ((c & 0xf0) == 0xe0)\n"
				"    return 3;\n"
				"  else if ((c & 0xf8) == 0xf0)\n"
				"    return 4;\n"
				"#ifdef NOT_USED\n"
				"  else if ((c & 0xfc) == 0xf8)\n"
				"    return 5;\n"
				"  else if ((c & 0xfe) == 0xfc)\n"
				"    return 6;\n"
				"#endif\n"
				"  return 1;\n");
			break;

		case PG_MULE_INTERNAL:	/* logic in pg_mule_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c >= 0x81 && c <= 0x8d)\n"
				"    return 2;\n"
				"  else if (c == 0x9a || c == 0x9b)\n"
				"    return 3;\n"
				"  else if (c >= 0x90 && c <= 0x99)\n"
				"    return 2;\n"
				"  else if (c == 0x9c || c == 0x9d)\n"
				"    return 4;\n"
				"  return 1;\n");
			break;

		case PG_SJIS:	/* logic in pg_sjis_mblen */
		case PG_SHIFT_JIS_2004:
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c >= 0xa1 && c <= 0xdf)\n"
				"    return 1;	/* 1byte kana? */\n"
				"  else if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_BIG5:	/* logic in pg_big5_mblen */
		case PG_GBK:	/* logic in pg_gbk_mblen, but identical */
		case PG_UHC:	/* logic in pg_uhc_mblen, but identical */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_GB18030:/* logic in pg_gb18030_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c1 = *((const cl_uchar *)str);\n"
				"  cl_uchar c2;\n"
				"  if ((c & 0x80) == 0)\n"
				"    return 1; /* ASCII */\n"
				"  c2 = *((const cl_uchar *)(str + 1));\n"
				"  if (c2 >= 0x30 && c2 <= 0x39)\n"
				"    return 4;\n"
				"  return 2;\n");
			break;

		default:	/* encoding with maxlen==1 */
			if (pg_database_encoding_max_length() != 1)
				elog(ERROR, "Bug? unsupported database encoding: %s",
					 GetDatabaseEncodingName());
			appendStringInfoString(
				buf,
				"  return 1;\n");
			break;
	}
	appendStringInfoString(
		buf,
		"}\n\n");
}

/*
 * session info for libgputime.a
 */

/*
 * assign_timelib_session_info
 *
 * It constructs per-session information around cuda_timelib.h.
 * At this moment, items below has to be informed.
 * - session_timezone information
 */

/* copied from src/timezone/tzfile.h */
#define TZ_MAX_TIMES	2000
#define TZ_MAX_TYPES	256		/* Limited by what (uchar)'s can hold */
#define TZ_MAX_CHARS	50		/* Maximum number of abbreviation characters */
#define TZ_MAX_LEAPS	50		/* Maximum number of leap second corrections */

/* copied from src/timezone/pgtz.h */
#define BIGGEST(a, b)	(((a) > (b)) ? (a) : (b))

struct ttinfo
{								/* time type information */
	int32		tt_gmtoff;		/* UTC offset in seconds */
	bool		tt_isdst;		/* used to set tm_isdst */
	int			tt_abbrind;		/* abbreviation list index */
	bool		tt_ttisstd;		/* TRUE if transition is std time */
	bool		tt_ttisgmt;		/* TRUE if transition is UTC */
};

struct lsinfo
{                               /* leap second information */
    pg_time_t   ls_trans;       /* transition time */
    long        ls_corr;        /* correction to apply */
};

struct state
{
	cl_int		leapcnt;
	cl_int		timecnt;
	cl_int		typecnt;
	cl_int		charcnt;
	bool		goback;
	bool		goahead;
	/* NOTE: pg_time_t has different meaning in GPU kernel */
    pg_time_t	ats[TZ_MAX_TIMES];
	unsigned char types[TZ_MAX_TIMES];
	struct ttinfo ttis[TZ_MAX_TYPES];
	cl_char		chars[BIGGEST(BIGGEST(TZ_MAX_CHARS + 1, 3 /* sizeof gmt */ ),
							  (2 * (TZ_STRLEN_MAX + 1)))];
	struct lsinfo lsis[TZ_MAX_LEAPS];
	/*
	 * The time type to use for early times or if no transitions. It is always
	 * zero for recent tzdb releases. It might be nonzero for data from tzdb
	 * 2018e or earlier.
	 */
	int			defaulttype;
};

struct pg_tz
{
	/* TZname contains the canonically-cased name of the timezone */
	char			TZname[TZ_STRLEN_MAX + 1];
	struct state	state;
};

STATIC_INLINE(void)
assign_timelib_session_info(StringInfo buf)
{
	const struct state *sp;
	int			i;

	appendStringInfo(
		buf,
		"/* ================================================\n"
		" * session information for device time library\n"
		" * ================================================ */\n");

	/* put session timezone info */
	sp = &session_timezone->state;

	/* ats[] */
	appendStringInfo(
		buf,
		"static __device__ cl_long __session_timezone_state_ats[] =\n");
	appendStringInfo(buf, "  {\n");
	if (sp->timecnt == 0)
		appendStringInfo(buf, " 0");
	else
	{
		for (i=0; i < sp->timecnt; i++)
		{
			appendStringInfo(buf, "    %10ld,\n", sp->ats[i]);
		}
	}
	appendStringInfo(buf, " };\n");

	/* types[] */
	appendStringInfo(
		buf,
		"static __device__ cl_uchar __session_timezone_state_types[] =\n"
		"  {");
	if (sp->timecnt == 0)
		appendStringInfo(buf, " 0");
	else
	{
		for (i=0; i < sp->timecnt; i++)
		{
			appendStringInfo(buf, " %d,", sp->types[i]);
		}
	}
	appendStringInfo(buf, "  };\n");

	/* ttis[] */
	appendStringInfo(
		buf,
		"static __device__ tz_ttinfo __session_timezone_state_ttis[] = {\n");
	if (sp->typecnt == 0)
		appendStringInfo(buf, "  { 0, 0, 0, 0, 0 },\n");
	else
	{
		for (i=0; i < sp->typecnt; i++)
		{
			appendStringInfo(
				buf,
				"  { %d, %s, %d, %s, %s },\n",
				sp->ttis[i].tt_gmtoff,
				sp->ttis[i].tt_isdst   ? "true" : "false",
				sp->ttis[i].tt_abbrind,
				sp->ttis[i].tt_ttisstd ? "true" : "false",
				sp->ttis[i].tt_ttisgmt ? "true" : "false");
		}
	}
	appendStringInfo(buf, "};\n");

	/* lsis[] */
	appendStringInfo(
		buf,
		"static __device__ tz_lsinfo __session_timezone_state_lsis[] = {\n");
	if (sp->leapcnt == 0)
		appendStringInfo(buf, "  { 0, 0 },\n");
	else
	{
		for (i=0; i < sp->leapcnt; i++)
		{
			appendStringInfo(
				buf,
				"  { %ld, %ld },\n",
				sp->lsis[i].ls_trans,
				sp->lsis[i].ls_corr);
		}
	}
	appendStringInfo(buf, "};\n");
	/* session_timezone_state */
	appendStringInfo(
        buf,
		"__device__ const tz_state session_timezone_state =\n"
		"{\n"
		"    %d,    /* leapcnt */\n"
		"    %d,    /* timecnt */\n"
		"    %d,    /* typecnt */\n"
		"    %d,    /* charcnt */\n"
		"    %s,    /* goback */\n"
		"    %s,    /* goahead */\n"
		"    __session_timezone_state_ats,   /* ats[] */\n"
		"    __session_timezone_state_types, /* types[] */\n"
		"    __session_timezone_state_ttis,  /* ttis[] */\n"
		"    __session_timezone_state_lsis,  /* lsis[] */\n"
		"    %d,    /* defaulttype */\n"
		"};\n",
		sp->leapcnt,
		sp->timecnt,
		sp->typecnt,
		sp->charcnt,
		sp->goback  ? "true" : "false",
		sp->goahead ? "true" : "false",
		sp->defaulttype);

	/* SetEpochTimestamp() */
	appendStringInfo(
		buf,
		"DEVICE_FUNCTION(Timestamp)\n"
		"SetEpochTimestamp(void)\n"
		"{\n"
		"  return %ldLL;\n"
		"}\n",
		SetEpochTimestamp());

	appendStringInfoChar(buf, '\n');
}

static void
assign_misclib_session_info(StringInfo buf)
{
	struct lconv *lconvert = PGLC_localeconv();
	cl_int		fpoint;
	cl_long		scale;
	cl_int		i;

	/* see comments about frac_digits in cash_in() */
	fpoint = lconvert->frac_digits;
	if (fpoint < 0 || fpoint > 10)
		fpoint = 2;

	/* compute required scale factor */
	scale = 1;
	for (i=0; i < fpoint; i++)
		scale *= 10;

	appendStringInfo(
		buf,
		"/* ================================================\n"
		" * session information for cuda_misc.h\n"
		" * ================================================ */\n"
		"\n"
		"DEVICE_FUNCTION(cl_long)\n"
		"PGLC_CURRENCY_SCALE(void)\n"
		"{\n"
		"  return %ld;\n"
		"}\n",
		scale);
}

/*
 * pgstrom_build_session_info
 *
 * it build a session specific code. if extra_flags contains a particular
 * custom_scan node related GPU routine, GpuTaskState must be provided.
 */
void
pgstrom_build_session_info(StringInfo buf,
						   GpuTaskState *gts,
						   cl_uint extra_flags)
{
	if ((extra_flags & DEVKERNEL_NEEDS_TIMELIB) != 0)
		assign_timelib_session_info(buf);
	if ((extra_flags & DEVKERNEL_NEEDS_TEXTLIB) != 0)
		assign_textlib_session_info(buf);
	if ((extra_flags & DEVKERNEL_NEEDS_MISCLIB) != 0)
		assign_misclib_session_info(buf);

	if ((extra_flags & DEVKERNEL_NEEDS_GPUSCAN) != 0)
		assign_gpuscan_session_info(buf, gts);
	if ((extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
		assign_gpujoin_session_info(buf, gts);
	if ((extra_flags & DEVKERNEL_NEEDS_GPUPREAGG) != 0)
		assign_gpupreagg_session_info(buf, gts);
}

/*
 * pgstrom_load_cuda_program
 */
CUmodule
pgstrom_load_cuda_program(ProgramId program_id)
{
	program_cache_entry *entry = NULL;
	CUmodule	cuda_module;
	CUresult	rc;
	size_t		stack_sz, lvalue;
	char	   *bin_image;

	SpinLockAcquire(&pgcache_head->lock);
retry_checks:
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		werror("CUDA Program ID=%lu was not found", program_id);
	}
	if (entry->ptx_image == CUDA_PROGRAM_BUILD_FAILURE)
	{
		SpinLockRelease(&pgcache_head->lock);
		werror("CUDA program build failure (id=%lu):\n%s",
			   (long)program_id, entry->error_msg);
	}
	else if (entry->ptx_image)
	{
		get_cuda_program_entry_nolock(entry);
		SpinLockRelease(&pgcache_head->lock);

		/*
		 * NOTE: Stack size of GPU thread is configured to 1024 in default,
		 * however, not a small varlena buffer needs more stack configuration.
		 * In addition, build with debug option also requires more stack than
		 * optimized kernel.
		 */
		stack_sz = 1800 + MAXALIGN(entry->varlena_bufsz);
		if ((entry->extra_flags & DEVKERNEL_BUILD_DEBUG_INFO) != 0)
			stack_sz += 4096;
#ifdef USE_ASSERT_CHECKING
		{
			pg_crc32	__ptx_crc;

			INIT_LEGACY_CRC32(__ptx_crc);
			COMP_LEGACY_CRC32(__ptx_crc,
							  entry->ptx_image,
							  entry->ptx_length);
			FIN_LEGACY_CRC32(__ptx_crc);
			Assert(__ptx_crc == entry->ptx_crc);
		}
#endif /* USE_ASSERT_CHECKING */
		STROM_TRY();
		{
			bin_image = link_cuda_libraries(entry->ptx_image,
											entry->ptx_length,
											entry->extra_flags);
		}
		STROM_CATCH();
		{
			put_cuda_program_entry(entry);
			STROM_RE_THROW();
		}
		STROM_END_TRY();
		put_cuda_program_entry(entry);
	}
	else if (entry->build_chain.prev || entry->build_chain.next)
	{
		/*
		 * Nobody picked up this CUDA program for build yet, so we
		 * try to build the program by ourselves, but synchronously.
		 */
		dlist_delete(&entry->build_chain);
		memset(&entry->build_chain, 0, sizeof(dlist_node));
		get_cuda_program_entry_nolock(entry);
		SpinLockRelease(&pgcache_head->lock);
		STROM_TRY();
		{
			entry = build_cuda_program(entry);
		}
		STROM_CATCH();
		{
			put_cuda_program_entry(entry);
			STROM_RE_THROW();
		}
		STROM_END_TRY();
		if (!GpuWorkerCurrentContext)
			CHECK_FOR_INTERRUPTS();
		else
			CHECK_WORKER_TERMINATION();
		SpinLockAcquire(&pgcache_head->lock);
		put_cuda_program_entry_nolock(entry);
		goto retry_checks;
	}
	else
	{
		/* NVRTC is still in-progress */
		SpinLockRelease(&pgcache_head->lock);
		if (!GpuWorkerCurrentContext)
			CHECK_FOR_INTERRUPTS();
		else
			CHECK_WORKER_TERMINATION();
		pg_usleep(50000L);
		SpinLockAcquire(&pgcache_head->lock);
		goto retry_checks;
	}
	rc = cuModuleLoadData(&cuda_module, bin_image);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleLoadData: %s", errorText(rc));

	rc = cuCtxGetLimit(&lvalue, CU_LIMIT_STACK_SIZE);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuCtxGetLimit: %s", errorText(rc));
	if (stack_sz > lvalue)
	{
		rc = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, stack_sz);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuCtxSetLimit: %s", errorText(rc));
	}
	return cuda_module;
}

/*
 * cudaProgramBuilderSigTerm
 */
static void
cudaProgramBuilderSigTerm(SIGNAL_ARGS)
{
	int		saved_errno = errno;

	cuda_program_builder_got_signal = true;

	pg_memory_barrier();

	SetLatch(MyLatch);

	errno = saved_errno;
}

/*
 * cudaProgramBuilderMain
 */
void
cudaProgramBuilderMain(Datum arg)
{
	int			builder_id = DatumGetInt32(arg);
	int			major;
	int			minor;
	nvrtcResult	rc;

	pqsignal(SIGTERM, cudaProgramBuilderSigTerm);
	BackgroundWorkerUnblockSignals();

	/* Init CUDA run-time compiler library */
	rc = nvrtcVersion(&major, &minor);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcVersion: %d", (int)rc);
	elog(LOG, "CUDA Program Builder-%d with NVRTC version %d.%d",
		 builder_id, major, minor);

	/*
	 * Event Loop
	 */
	pgbuilder_state->builders[builder_id].latch = MyLatch;
	pg_atomic_fetch_add_u32(&pgbuilder_state->num_active_builders, 1);
	PG_TRY();
	{
		while (!cuda_program_builder_got_signal)
		{
			program_cache_entry *entry;
			dlist_node *dnode;
			int			ev;

			/* Is there any pending CUDA program? */
			SpinLockAcquire(&pgcache_head->lock);
			if (dlist_is_empty(&pgcache_head->build_list))
			{
				SpinLockRelease(&pgcache_head->lock);

				ev = WaitLatch(MyLatch,
							   WL_LATCH_SET |
							   WL_TIMEOUT |
							   WL_POSTMASTER_DEATH,
							   1000L,
							   PG_WAIT_EXTENSION);
				ResetLatch(MyLatch);
				if (ev & WL_POSTMASTER_DEATH)
					elog(FATAL, "unexpected postmaster dead");
				CHECK_FOR_INTERRUPTS();
				continue;
			}
			dnode = dlist_pop_head_node(&pgcache_head->build_list);
			entry = dlist_container(program_cache_entry, build_chain, dnode);

			/*
			 * !ptx_image && build_chain==0 means program compilation is
			 * in-progress. So, it avoid duplication of the program build.
			 */
			memset(&entry->build_chain, 0, sizeof(dlist_node));
			Assert(!entry->ptx_image);	/* must be build in-progress */
			get_cuda_program_entry_nolock(entry);
			SpinLockRelease(&pgcache_head->lock);

			PG_TRY();
			{
				entry = build_cuda_program(entry);
			}
			PG_CATCH();
			{
				/*
				 * Unlike CUDA_PROGRAM_BUILD_FAILURE case, exceptions are
				 * often raised by resource starvation, or other reasons,
				 * thus, CUDA program entry must be backed to the build
				 * pending list, to be picked up by other workers.
				 */
				SpinLockAcquire(&pgcache_head->lock);
				dlist_push_tail(&pgcache_head->build_list,
								&entry->build_chain);
				put_cuda_program_entry_nolock(entry);
				SpinLockRelease(&pgcache_head->lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			put_cuda_program_entry(entry);
		}
	}
	PG_CATCH();
	{
		pg_atomic_fetch_sub_u32(&pgbuilder_state->num_active_builders, 1);
		pgbuilder_state->builders[builder_id].latch = NULL;
		PG_RE_THROW();
	}
	PG_END_TRY();
	pg_atomic_fetch_sub_u32(&pgbuilder_state->num_active_builders, 1);
	pgbuilder_state->builders[builder_id].latch = NULL;
}

static void
cudaProgramBuilderWakeUp(bool error_if_no_builders)
{
	int		i, count = 0;

	for (i=0; i < num_program_builders; i++)
	{
		Latch *latch = pgbuilder_state->builders[i].latch;

		if (latch)
		{
			SetLatch(latch);
			count++;
		}
	}

	if (error_if_no_builders && count == 0)
		elog(ERROR, "PG-Strom: no active CUDA C program builder");
}

#if 0
/*
 * XXXX - PL/CUDA was re-designed to use CUDA runtime,
 * so we no longer need own wrapper library.
 */
static void
build_wrapper_libraries(const char *wrapper_filename,
						void **p_wrapper_lib,
						size_t *p_wrapper_libsz)
{
	char   *src_fname = NULL;
	char   *lib_fname = NULL;
	int		fdesc = -1;
	int		status;
	char	buffer[1024];
	char	spath[128];
	char	lpath[128];
	char	cmd[MAXPGPATH];
	void   *wrapper_lib = NULL;
	ssize_t	rv, buffer_len;
	struct stat st_buf;

	PG_TRY();
	{
		/* write out source */
		strcpy(spath, P_tmpdir "/XXXXXX.cu");
		fdesc = mkstemps(spath, 3);
		if (fdesc < 0)
			elog(ERROR, "failed on mkstemps('%s') : %m", src_fname);
		src_fname = spath;

		buffer_len = snprintf(buffer, sizeof(buffer),
							  "#include <%s>\n",
							  wrapper_filename);
		rv = write(fdesc, buffer, buffer_len);
		if (rv != buffer_len)
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
				 " -I " PGSHAREDIR "/extension"
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
#endif

static void
pgstrom_startup_cuda_program(void)
{
	bool		found;
	int			i, mclass;
	size_t		offset;
	size_t		length;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	pgcache_head = ShmemInitStruct("PG-Strom Program Cache",
								   offsetof(program_cache_head, base) +
								   ((size_t)program_cache_size_kb << 10),
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
	dlist_init(&pgcache_head->addr_list);
	for (i=0; i <= PGCACHE_CHUNKSZ_MAX_BIT; i++)
		dlist_init(&pgcache_head->free_list[i]);

	length = ((size_t)program_cache_size_kb << 10);
	offset = 0;
	mclass = PGCACHE_CHUNKSZ_MAX_BIT;
	while (mclass >= PGCACHE_CHUNKSZ_MIN_BIT)
	{
		program_cache_entry *entry;

		if (offset + (1UL << mclass) >= length)
		{
			mclass--;
			continue;
		}
		entry = (program_cache_entry *)(pgcache_head->base + offset);
		memset(entry, 0, offsetof(program_cache_entry, data));
		entry->magic = PGCACHE_CHUNK_MAGIC;
		entry->mclass = mclass;
		dlist_push_tail(&pgcache_head->free_list[mclass],
						&entry->free_chain);
		offset += (1UL << mclass);
	}

	/* initialize program builder state */
	length = offsetof(program_builder_state,
					  builders[num_program_builders]);
	pgbuilder_state = ShmemInitStruct("PG-Strom Program Builders State",
									  length, &found);
	if (found)
		elog(ERROR, "Bug? shared memory for program builders already exists");
	memset(pgbuilder_state, 0, length);
}

void
pgstrom_init_cuda_program(void)
{
	int			i;

	/*
	 * allocation of shared memory segment size
	 */
	DefineCustomIntVariable("pg_strom.program_cache_size",
							"size of shared program cache",
							NULL,
							&program_cache_size_kb,
							256 * 1024,		/* 256MB */
							16 * 1024,		/* 16MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);

	/*
	 * number of worker process to build CUDA program
	 */
	DefineCustomIntVariable("pg_strom.num_program_builders",
							"number of workers to build CUDA C programs",
							NULL,
							&num_program_builders,
							2,
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/*
	 * Enables debug option on GPU kernel build
	 */
	DefineCustomBoolVariable("pg_strom.debug_jit_compile_options",
							 "Enables debug options on GPU kernel build",
							 NULL,
							 &pgstrom_debug_jit_compile_options,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE | GUC_SUPERUSER_ONLY,
							 NULL, NULL, NULL);

	/* allocation of static shared memory */
	RequestAddinShmemSpace(offsetof(program_cache_head, base) +
						   ((size_t)program_cache_size_kb << 10));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_program;

	/* register CUDA C program builders */
	for (i=0; i < num_program_builders; i++)
	{
		BackgroundWorker worker;

		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "PG-Strom Program Builder-%d", i);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
		worker.bgw_start_time = BgWorkerStart_PostmasterStart;
		worker.bgw_restart_time = 1;

		snprintf(worker.bgw_library_name,
				 BGW_MAXLEN, "pg_strom");
		snprintf(worker.bgw_function_name,
				 BGW_MAXLEN, "cudaProgramBuilderMain");
		worker.bgw_main_arg = Int32GetDatum(i);
		RegisterBackgroundWorker(&worker);
	}
}
