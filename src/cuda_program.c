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
#include "pg_strom.h"
#include "cuda_misc.h"
#include "cuda_timelib.h"
#include "cuda_textlib.h"

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
	int				extra_flags;	/*             + kern_define */
	char		   *kern_define;	/*             + kern_source */
	size_t			kern_deflen;
	char		   *kern_source;
	size_t			kern_srclen;
	pg_crc32		bin_crc;
	char		   *bin_image;		/* may be CUDA_PROGRAM_BUILD_FAILURE */
	size_t			bin_length;
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

/* ---- GUC variables ---- */
static int		program_cache_size_kb;

/* ---- static variables ---- */
static shmem_startup_hook_type shmem_startup_next;
static program_cache_head *pgcache_head = NULL;
#define PGSTROM_CUDA(x)	\
	static const char *pgstrom_cuda_##x##_pathname = NULL;
#include "cuda_filelist"
#undef PGSTROM_CUDA
static void	   *curand_wrapper_lib = NULL;
static size_t	curand_wrapper_libsz;

/* ---- forward declarations ---- */
static void put_cuda_program_entry_nolock(program_cache_entry *entry);


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
construct_flat_cuda_source(uint32 extra_flags,
						   const char *kern_define,
						   const char *kern_source)
{
	size_t		ofs = 0;
	size_t		len = strlen(kern_define) + strlen(kern_source) + 20000;
	char	   *source;
	const char *pg_anytype;

	source = malloc(len);
	if (!source)
		return NULL;

	ofs += snprintf(source + ofs, len - ofs,
					"#include <cuda_device_runtime_api.h>\n"
					"\n"
					"#define HOSTPTRLEN %u\n"
					"#define DEVICEPTRLEN %lu\n"
					"#define BLCKSZ %u\n"
					"#define MAXIMUM_ALIGNOF %u\n"
					"#define MAXIMUM_ALIGNOF_SHIFT %u\n"
#ifdef PGSTROM_DEBUG
					"#define PGSTROM_DEBUG 1\n"
#endif
					"\n",
					SIZEOF_VOID_P,
					sizeof(CUdeviceptr),
					BLCKSZ,
					MAXIMUM_ALIGNOF,
					MAXIMUM_ALIGNOF_SHIFT);
	/* Common PG-Strom device routine */
	ofs += snprintf(source + ofs, len - ofs,
					"#include \"cuda_common.h\"\n");
	/* Per session definition if any */
	ofs += snprintf(source + ofs, len - ofs,
					"%s\n", kern_define);

	/*
	 * PG-Strom CUDA device code libraries
	 */
	/* cuRand library */
	if ((extra_flags & DEVKERNEL_NEEDS_CURAND) == DEVKERNEL_NEEDS_CURAND)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_curand.h\"\n");
	/* cuBlas library */
	if ((extra_flags & DEVKERNEL_NEEDS_CUBLAS) == DEVKERNEL_NEEDS_CUBLAS)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_cublas.h\"\n");
	/* cuda dynpara.h */
	if ((extra_flags & DEVKERNEL_NEEDS_DYNPARA) == DEVKERNEL_NEEDS_DYNPARA)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_dynpara.h\"\n");
	/* cuda mathlib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_MATHLIB) == DEVKERNEL_NEEDS_MATHLIB)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_mathlib.h\"\n");
	/* cuda timelib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_TIMELIB) == DEVKERNEL_NEEDS_TIMELIB)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_timelib.h\"\n");
	/* cuda textlib.h */
	if ((extra_flags & DEVKERNEL_NEEDS_TEXTLIB) == DEVKERNEL_NEEDS_TEXTLIB)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_textlib.h\"\n");
	/* cuda numeric.h */
	if ((extra_flags & DEVKERNEL_NEEDS_NUMERIC) == DEVKERNEL_NEEDS_NUMERIC)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_numeric.h\"\n");
	/* cuda money.h */
	if ((extra_flags & DEVKERNEL_NEEDS_MISC) == DEVKERNEL_NEEDS_MISC)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_misc.h\"\n");
	/* cuda matrix.h */
	if ((extra_flags & DEVKERNEL_NEEDS_MATRIX) == DEVKERNEL_NEEDS_MATRIX)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_matrix.h\"\n");
	/* pg_anytype_t declaration */
	pg_anytype =
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
		"    pg_uuid_t        uuid_v;\n"
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
		"  } pg_anytype_t;\n\n";
	ofs += snprintf(source + ofs, len - ofs, "%s\n", pg_anytype);

	/* Main logic of each GPU tasks */

	/* GpuScan */
	if (extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_gpuscan.h\"\n");
	/* GpuHashJoin */
	if (extra_flags & DEVKERNEL_NEEDS_GPUJOIN)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_gpujoin.h\"\n");
	/* GpuPreAgg */
	if (extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_gpupreagg.h\"\n");
	/* PL/CUDA functions */
	if (extra_flags & DEVKERNEL_NEEDS_PLCUDA)
		ofs += snprintf(source + ofs, len - ofs,
						"#include \"cuda_plcuda.h\"\n");
	/* automatically generated portion */
	ofs += snprintf(source + ofs, len - ofs, "%s\n", kern_source);
	/* code to be included at the last */
	ofs += snprintf(source + ofs, len - ofs,
					"#include \"cuda_terminal.h\"\n");
	return source;
}

/*
 * link_cuda_libraries - links CUDA libraries with the supplied PTX binary
 */
static void
link_cuda_libraries(char *ptx_image, size_t ptx_length,
					cl_uint extra_flags,
					void **p_bin_image, size_t *p_bin_length)
{
	CUlinkState		lstate;
	CUresult		rc;
	CUjit_option	jit_options[10];
	void		   *jit_option_values[10];
	int				jit_index = 0;
	void		   *temp;
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
		werror("failed on cuLinkCreate: %s", errorText(rc));

	STROM_TRY();
	{
		/* add the base PTX image */
		rc = cuLinkAddData(lstate, CU_JIT_INPUT_PTX,
						   ptx_image, ptx_length,
						   "pg-strom", 0, NULL, NULL);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuLinkAddData: %s", errorText(rc));

		/* libcudart.a, if any */
		if ((extra_flags & DEVKERNEL_NEEDS_DYNPARA) == DEVKERNEL_NEEDS_DYNPARA)
		{
			snprintf(pathname, sizeof(pathname), "%s/libcudadevrt.a",
					 CUDA_LIBRARY_PATH);
			rc = cuLinkAddFile(lstate, CU_JIT_INPUT_LIBRARY, pathname,
							   0, NULL, NULL);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuLinkAddFile(\"%s\"): %s",
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
				werror("failed on cuLinkAddData: %s", errorText(rc));
		}

		/* libcublas_device.a, if any */
		if ((extra_flags & DEVKERNEL_NEEDS_CUBLAS) == DEVKERNEL_NEEDS_CUBLAS)
		{
			snprintf(pathname, sizeof(pathname), "%s/libcublas_device.a",
					 CUDA_LIBRARY_PATH);
			rc = cuLinkAddFile(lstate, CU_JIT_INPUT_LIBRARY, pathname,
							   0, NULL, NULL);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuLinkAddFile(\"%s\"): %s",
					   pathname, errorText(rc));
		}
		/* do the linkage */
		rc = cuLinkComplete(lstate, &temp, &bin_length);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuLinkComplete: %s", errorText(rc));

		/*
		 * copy onto the result buffer; because bin_image is owned by
		 * CUlinkState, so cuLinkDestroy also releases the bin_image.
		 */
		bin_image = malloc(bin_length);
		if (!bin_image)
			werror("out of memory");
		memcpy(bin_image, temp, bin_length);

		*p_bin_image = bin_image;
		*p_bin_length = bin_length;
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
}

/*
 * writeout_cuda_source_file
 *
 * It makes a temporary file to write-out cuda source.
 */
static void
writeout_cuda_source_file(char *tempfile, const char *source)
{
	static pg_atomic_uint64 sourceFileCounter = {0};
	FILE	   *filp;

	/*
	 * Generate a tempfile name that should be unique within the current
	 * database instance.
	 */
	snprintf(tempfile, MAXPGPATH,
			 "%s/base/%s/%s_strom_%d.%ld.gpu",
			 DataDir, PG_TEMP_FILES_DIR,
			 PG_TEMP_FILE_PREFIX,
			 MyProcPid,
			 pg_atomic_fetch_add_u64(&sourceFileCounter, 1));
	/*
	 * Open the file.  Note: we don't use O_EXCL, in case there is
	 * an orphaned temp file that can be reused.
	 */
	filp = fopen(tempfile, "w+b");
	fputs(source, filp);
	fclose(filp);
}

/*
 * pgstrom_cuda_source_file - write out a CUDA program to temp-file
 */
const char *
pgstrom_cuda_source_file(ProgramId program_id)
{
	program_cache_entry *entry;
	char	   *source;
	char		tempfilepath[MAXPGPATH];

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
	if (!source)
		elog(ERROR, "out of memory");
	writeout_cuda_source_file(tempfilepath, source);

	free(source);

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
	const char	   *options[10];
	int				opt_index = 0;
	void		   *ptx_image = NULL;
	size_t			ptx_length = 0;
	void		   *bin_image = NULL;
	size_t			bin_length = 0;
	char		   *build_log = NULL;
	size_t			log_length;
	int				pindex;
	int				hindex;
	size_t			offset;
	size_t			length;
	char			tempfilepath[MAXPGPATH];

	Assert(!src_entry->build_chain.prev && !src_entry->build_chain.next);

	/* Make a nvrtcProgram object */
	source = construct_flat_cuda_source(src_entry->extra_flags,
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
		 *
		 * MEMO: (23-Oct-2017) It looks to me "--device-debug" leads
		 * CUDA_ERROR_ILLEGAL_INSTRUCTION error on execution.
		 * So, as a workaround, we removed this option here.
		 */
		snprintf(gpu_arch_option, sizeof(gpu_arch_option),
				 "--gpu-architecture=compute_%lu", devComputeCapability);
		options[opt_index++] = "-I " CUDA_INCLUDE_PATH;
		options[opt_index++] = "-I " PGSHAREDIR "/extension";
		snprintf(gpu_arch_option, sizeof(gpu_arch_option),
                 "--gpu-architecture=compute_%lu", devComputeCapability);
		options[opt_index++] = gpu_arch_option;
		options[opt_index++] = "--generate-line-info";
		options[opt_index++] = "--use_fast_math";
//		options[opt_index++] = "--device-as-default-execution-space";
		/* library linkage needs relocatable PTX */
		if (src_entry->extra_flags & DEVKERNEL_NEEDS_LINKAGE)
			options[opt_index++] = "--relocatable-device-code=true";

		/*
		 * Kick runtime compiler
		 */
		rc = nvrtcCompileProgram(program, opt_index, options);
		if (rc == NVRTC_ERROR_COMPILATION)
		{
			writeout_cuda_source_file(tempfile, source);
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
			//ptx_image[ptx_length++] = '\0';

			/*
			 * Link with run-time libraries, if any
			 */
			if (src_entry->extra_flags & DEVKERNEL_NEEDS_LINKAGE)
			{
				link_cuda_libraries(ptx_image, ptx_length,
									src_entry->extra_flags,
									&bin_image, &bin_length);
				free(ptx_image);
			}
			else
			{
				bin_image = ptx_image;
				bin_length = ptx_length;
			}
			ptx_image = NULL;
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
		 * Allocation of a new entry, to keep bin_image/build_log
		 */
		length = (MAXALIGN(src_entry->kern_deflen + 1) +
				  MAXALIGN(src_entry->kern_srclen + 1) +
				  MAXALIGN(bin_length + 1) +
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
		bin_entry->extra_flags		= src_entry->extra_flags;
		bin_entry->kern_deflen		= src_entry->kern_deflen;
		bin_entry->kern_define		= bin_entry->data + offset;
		strcpy(bin_entry->kern_define, src_entry->kern_define);
		offset += MAXALIGN(bin_entry->kern_deflen + 1);

		bin_entry->kern_srclen		= src_entry->kern_srclen;
		bin_entry->kern_source		= bin_entry->data + offset;
		strcpy(bin_entry->kern_source, src_entry->kern_source);
		offset += MAXALIGN(bin_entry->kern_srclen + 1);

		if (!bin_image)
			bin_entry->bin_image	= CUDA_PROGRAM_BUILD_FAILURE;
		else
		{
			pg_crc32	bin_crc;

			bin_entry->bin_image	= bin_entry->data + offset;
			bin_entry->bin_length	= bin_length;
			memcpy(bin_entry->bin_image, bin_image, bin_length);
			offset += MAXALIGN(bin_length + 1);

			INIT_LEGACY_CRC32(bin_crc);
			COMP_LEGACY_CRC32(bin_crc, bin_image, bin_length);
			FIN_LEGACY_CRC32(bin_crc);
			bin_entry->bin_crc		= bin_crc;
		}

		bin_entry->error_msg		= bin_entry->data + offset;
		if (!bin_image)
			snprintf(bin_entry->error_msg, length - offset,
					 "build failure:\n%s\nsource: %s",
					 build_log, tempfilepath);
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
		if (bin_image)
			free(bin_image);
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

	return bin_entry;
}

/*
 * pgstrom_try_build_cuda_program
 *
 * It picks up a program entry that is not built yet, if any, then runs
 * NVRTC and linker to construct an executable binary.
 * Because linker process needs a valid CUDA context, only GpuContext
 * worker can call this function.
 */
bool
pgstrom_try_build_cuda_program(void)
{
	dlist_node	   *dnode;
	program_cache_entry *entry;

	Assert(GpuWorkerCurrentContext != NULL);

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

	/*
	 * This thread will focus on the program build, so some other
	 * worker needs to process the pending tasks.
	 */
	pthreadCondSignal(GpuWorkerCurrentContext->cond);

	STROM_TRY();
	{
		entry = build_cuda_program(entry);
	}
	STROM_CATCH();
	{
		/*
		 * Unlike CUDA_PROGRAM_BUILD_FAILURE case, exceptions are usually
		 * raised by resource starvation, thus, restart of GPU server may
		 * be able to build the GPU program on the next trial.
		 * So, CUDA program entry is backed to the build pending list.
		 */
		SpinLockAcquire(&pgcache_head->lock);
		dlist_push_tail(&pgcache_head->build_list, &entry->build_chain);
		put_cuda_program_entry_nolock(entry);
		SpinLockRelease(&pgcache_head->lock);

		STROM_RE_THROW();
	}
	STROM_END_TRY();

	put_cuda_program_entry(entry);

	return true;
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
							  const char *kern_source,
							  const char *kern_define,
							  bool wait_for_build,
							  const char *filename, int lineno)
{
	program_cache_entry	*entry;
	ProgramId	program_id;
	Size		kern_srclen = strlen(kern_source);
	Size		kern_deflen = strlen(kern_define);
	Size		length;
	Size		usage = 0;
	int			hindex;
	dlist_iter	iter;
	pg_crc32	crc;

	/*
	 * sanity check - even though concurrent and unrelated worker may pick up
	 * the CUDA program for build, here is no guarantee. For synchronous code
	 * build, the callers' context should be capable to run NVRTC.
	 */
	if (wait_for_build && !gcontext->cuda_context)
		elog(ERROR, "Bug? unable to kick synchronous code build with inactive GpuContext");

	/* makes a hash value */
	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &extra_flags, sizeof(int32));
	COMP_LEGACY_CRC32(crc, kern_source, kern_srclen);
	COMP_LEGACY_CRC32(crc, kern_define, kern_deflen);
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
			program_id = entry->program_id;
			get_cuda_program_entry_nolock(entry);
			/* Move this entry to the head of LRU list */
			dlist_move_head(&pgcache_head->lru_list, &entry->lru_chain);
		retry_checks:
			/* Raise an syntax error if already failed on build */
			if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
			{
				put_cuda_program_entry_nolock(entry);
				SpinLockRelease(&pgcache_head->lock);
				ereport(ERROR,
						(errcode(entry->error_code),
						 errmsg("%s", entry->error_msg)));
			}
			else if (entry->bin_image || !wait_for_build)
			{
				if (!trackCudaProgram(gcontext, program_id,
									  filename, lineno))
				{
					put_cuda_program_entry_nolock(entry);
					SpinLockRelease(&pgcache_head->lock);
					elog(ERROR, "out of memory");
				}
				SpinLockRelease(&pgcache_head->lock);
				return program_id;
			}
			else
			{
				/*
				 * NVRTC on this GPU kernel is still in-progress, and caller
				 * wants to synchronize the completion of the build.
				 */
				SpinLockRelease(&pgcache_head->lock);

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

	entry->program_id = program_id;
	entry->crc = crc;
	entry->extra_flags = extra_flags;

	entry->kern_define = (char *)(entry->data + usage);
	entry->kern_deflen = kern_deflen;
	memcpy(entry->kern_define, kern_define, kern_deflen + 1);
	usage += MAXALIGN(kern_deflen + 1);

	entry->kern_source = (char *)(entry->data + usage);
	entry->kern_srclen = kern_srclen;
	memcpy(entry->kern_source, kern_source, kern_srclen + 1);
	usage += MAXALIGN(kern_srclen + 1);

	/* no cuda binary at this moment */
	entry->bin_image = NULL;
	entry->bin_length = 0;
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
	pthreadCondSignal(gcontext->cond);

	/* wait for completion of build, if needed */
	while (wait_for_build)
	{
		SpinLockAcquire(&pgcache_head->lock);
		entry = lookup_cuda_program_entry_nolock(program_id);
		if (!entry)
		{
			SpinLockRelease(&pgcache_head->lock);
			elog(ERROR, "Bug? ProgramId=%lu has gone.", program_id);
		}
		else if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
		{
			SpinLockRelease(&pgcache_head->lock);
			ereport(ERROR,
					(errcode(entry->error_code),
					 errmsg("%s", entry->error_msg)));
		}
		else if (entry->bin_image)
		{
			SpinLockRelease(&pgcache_head->lock);
			break;
		}
		/* NVRTC on this GPU program is still in-progress */
		SpinLockRelease(&pgcache_head->lock);

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
	/* OID declaration of types */
	pgstrom_codegen_typeoid_declarations(buf);
	/* put timezone info */
	if ((extra_flags & DEVKERNEL_NEEDS_TIMELIB) != 0)
		assign_timelib_session_info(buf);
	/* put currency info */
	if ((extra_flags & DEVKERNEL_NEEDS_MISC) != 0)
		assign_misclib_session_info(buf);
	/* put text/string info */
	if ((extra_flags & DEVKERNEL_NEEDS_TEXTLIB) != 0)
		assign_textlib_session_info(buf);

	/* enables device projection? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUSCAN) != 0)
		assign_gpuscan_session_info(buf, gts);
	/* enables device projection? */
	if ((extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
		assign_gpujoin_session_info(buf, gts);
	/* enables outer-quals evaluation? */
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
	CUmodule		cuda_module;
	CUresult		rc;
	char		   *bin_image;

	SpinLockAcquire(&pgcache_head->lock);
retry_checks:
	entry = lookup_cuda_program_entry_nolock(program_id);
	if (!entry)
	{
		SpinLockRelease(&pgcache_head->lock);
		werror("CUDA Program ID=%lu was not found", program_id);
	}
	if (entry->bin_image == CUDA_PROGRAM_BUILD_FAILURE)
	{
		SpinLockRelease(&pgcache_head->lock);
		werror("CUDA program build failure (id=%lu):\n%s",
			   (long)program_id, entry->error_msg);
	}
	else if (entry->bin_image)
	{
		pg_crc32	bin_crc		__attribute__((unused));

		get_cuda_program_entry_nolock(entry);
		bin_image = entry->bin_image;
#ifdef PGSTROM_DEBUG
		INIT_LEGACY_CRC32(bin_crc);
		COMP_LEGACY_CRC32(bin_crc, entry->bin_image, entry->bin_length);
		FIN_LEGACY_CRC32(bin_crc);
		if (bin_crc != entry->bin_crc)
		{
			pg_crc32	__bin_crc = entry->bin_crc;
			SpinLockRelease(&pgcache_head->lock);
			werror("CUDA program binary corrupted (%08x / %08x)",
				   __bin_crc, bin_crc);
		}
#endif
		SpinLockRelease(&pgcache_head->lock);
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
		//TODO: bailout if worker shall be terminated
		SpinLockAcquire(&pgcache_head->lock);
		put_cuda_program_entry_nolock(entry);
		goto retry_checks;
	}
	else
	{
		/* NVRTC is still in-progress */
		SpinLockRelease(&pgcache_head->lock);
		//TODO: bailout if worker shall be terminated
		pg_usleep(50000L);
		SpinLockAcquire(&pgcache_head->lock);
		goto retry_checks;
	}

	rc = cuModuleLoadData(&cuda_module, bin_image);
	put_cuda_program_entry(entry);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleLoadData: %s", errorText(rc));
	return cuda_module;
}

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
}

void
pgstrom_init_cuda_program(void)
{
	int			major;
	int			minor;
	nvrtcResult	rc;

	/*
	 * allocation of shared memory segment size
	 */
	DefineCustomIntVariable("pg_strom.program_cache_size",
							"size of shared program cache",
							NULL,
							&program_cache_size_kb,
							512 * 1024,		/* 512MB */
							16 * 1024,		/* 16MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);

	/*
	 * Init CUDA run-time compiler library
	 */
	rc = nvrtcVersion(&major, &minor);
	if (rc != NVRTC_SUCCESS)
		elog(ERROR, "failed on nvrtcVersion: %s", nvrtcGetErrorString(rc));
	elog(LOG, "NVRTC - CUDA Runtime Compilation vertion %d.%d",
		 major, minor);

	/* setup cuda_xxxx.h file pathname */
#define PGSTROM_CUDA(x) \
	pgstrom_cuda_##x##_pathname = PGSHAREDIR "/extension/cuda_" #x ".h";
#include "cuda_filelist"
#undef PGSTROM_CUDA

	/* allocation of static shared memory */
	RequestAddinShmemSpace(offsetof(program_cache_head, base) +
						   ((size_t)program_cache_size_kb << 10));
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_program;

	/* build wrapper library objects */
	build_wrapper_libraries("cuda_curand.h",
							&curand_wrapper_lib,
							&curand_wrapper_libsz);
}
