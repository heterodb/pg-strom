/*
 * debug.c
 *
 * functions for debugging
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "nodes/pg_list.h"
#include "pg_strom.h"

Datum
pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	Size	size = PG_GETARG_INT64(0);
	void   *address;

	address = pgstrom_shmem_alloc(TopShmemContext, size);

	PG_RETURN_INT64((Size) address);
#else
	elog(ERROR, "%s is not implemented for production release");

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_alloc_func);

Datum
pgstrom_shmem_free_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	void		   *address = (void *) PG_GETARG_INT64(0);

	pgstrom_shmem_free(address);

	PG_RETURN_BOOL(true);
#else
	elog(ERROR, "%s is not implemented for production release");

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_free_func);

