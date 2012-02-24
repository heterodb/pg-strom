/*
 * main.c
 *
 * Entrypoint of the PG-Strom module
 * --
 * Copyright 2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * Local declarations
 */
void		_PG_init(void);



void
_PG_init(void)
{
	/*
	 * PG-Strom has to be loaded using shared_preload_libraries option
	 */
	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
		errmsg("PG-Strom must be loaded via shared_preload_libraries")));

	/* initialize shared memory segment */
	pgstrom_shmseg_init();
}
