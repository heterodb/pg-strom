/*
 * pg_boost.c
 *
 * Entrypoint of the pg_boost module
 *
 * Copyright 2011 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "pg_boost.h"

PG_MODULE_MAGIC;

void	_PG_init(void);

/*
 * Global variables
 */
int		guc_segment_size;
bool	guc_use_hugetlb;;
char   *guc_temp_dir;

/*
 * Entrypoint of the pg_boost module
 */
void
_PG_init(void)
{
	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
		errmsg("pg_boost must be loaded via shared_preload_libraries")));

	/* Create own shared memory segment */
	shmseg_init();
}
