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
#include "storage/ipc.h"
#include "storage/shmem.h"
#include "pg_strom.h"

static shmem_startup_hook_type shmem_startup_next;
static Size		program_cache_size;




static void
pgstrom_startup_cuda_program(void)
{
	if (shmem_startup_next)
		(*shmem_startup_next)();

	program_cache_head = ShmemInitStruct("PG-Strom program cache",
										 program_cache_size, &found);
	if (found)
		elog(ERROR, "Bug? shared memory for program cache already exists");

}

void
pgstrom_init_cuda_program(void)
{
	static int	__program_cache_size;

	DefineCustomIntVariable("pg_strom.program_cache_size",
							"size of shared program cache",
							NULL,
							&__program_cache_size,
							128 * 1024,		/* 128MB */
							16 * 1024,		/* 16MB */
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							NULL, NULL, NULL);
	program_cache_size = (Size)__program_cache_size * 1024L;

	/* allocation of static shared memory */
	RequestAddinShmemSpace(program_cache_size);
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_cuda_program;
}
