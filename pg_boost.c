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
#include "utils/guc.h"
#include "pg_boost.h"

PG_MODULE_MAGIC;

void	_PG_init(void);

/*
 * Global variables
 */
int		guc_segment_size;
bool	guc_with_hugetlb;
char   *guc_unbuffered_dir;

static void
pg_boost_guc_init(void)
{
	DefineCustomIntVariable("pg_boost.segment_size",
							"Size of shared memory segment in MB",
                            NULL,
							&guc_segment_size,
							128,			/* 128MB */
							32,				/*  32MB */
							2048 * 1024,	/*   2TB */
							PGC_SIGHUP,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	DefineCustomBoolVariable("pg_boost.with_hugetlb",
							 "True, if HugeTlb on shared memory segment",
							 NULL,
							 &guc_with_hugetlb,
							 false,
							 PGC_SIGHUP,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	DefineCustomStringVariable("pg_boost.unbuffered_dir",
							   "Path to the directory of unbuffered data",
							   NULL,
							   &guc_unbuffered_dir,
							   "/dev/shm/pg_boost",
							   PGC_SIGHUP,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
}


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

	/* Get GUC configurations */
	pg_boost_guc_init();



	/* Create own shared memory segment */
	shmmgr_init();
}
