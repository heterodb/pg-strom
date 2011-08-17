/*
 * pg_boost.c
 *
 * Entrypoint of the pg_boost module
 *
 * Copyright 2011 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "postgres.h"
#include "pg_boost.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

void	PG_init(void);

/*
 * Entrypoint of the pg_boost module
 */
void
PG_init(void)
{
	/* Create own shared memory segment */
	shmseg_init();
}
