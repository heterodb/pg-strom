/*
 * openmp_serv.c
 *
 * The background computing engine stick on the OpenMP infrastructure.
 * In addition, it also provide catalog of supported type and functions.
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "pg_strom.h"

void
pgstrom_cpu_enqueue_chunk(ChunkBuffer *chunk)
{
	elog(ERROR, "OpenMP is not supported");
}

void
pgstrom_cpu_startup(void *shmptr, Size shmsize)
{
	/* do nothing */
}

void
pgstrom_cpu_init(void)
{
	/* do nothing */
}

CpuTypeInfo *
pgstrom_cpu_type_lookup(Oid type_oid)
{
	/* CPU computing server is not supported */
	return NULL;
}

CpuFuncInfo *
pgstrom_cpu_func_lookup(Oid func_oid)
{
	/* CPU computing server is not supported */
	return NULL;
}

int
pgstrom_cpu_command_string(Oid ftableOid, int cmds[],
						   char *buf, size_t buflen)
{
	elog(ERROR, "CPU compuing server is not supported");

	return -1;	/* being compiler quiet */
}
