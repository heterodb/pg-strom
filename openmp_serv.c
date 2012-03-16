/*
 * openmp_serv.c
 *
 * Routines of computing engine component based on OpenMP.
 *
 * --
 * Copyright 2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "pg_strom.h"

void
pgstrom_openmp_enqueue_chunk(ChunkBuffer *chunk)
{
	elog(ERROR, "OpenMP is not supported in this package");
}
