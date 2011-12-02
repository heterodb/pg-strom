/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#ifndef PG_STROM_H
#define PG_STROM_H

/*
 * XXX - Some of naming scheme conflicts between PostgreSQL and
 * CUDA, so we make declarations of PostgreSQL side invisible in
 * the case when npp.h was included.
 */
#ifdef POSTGRES_H
#include "commands/explain.h"
#include "fmgr.h"
#include "foreign/fdwapi.h"
#endif
#include <driver_types.h>

#define PGSTROM_SCHEMA_NAME		"pg_strom"

#define PGSTROM_CHUNK_SIZE		(2400 * (BLCKSZ / 8192))

/*
 * utilcmds.c
 */
#ifdef POSTGRES_H
extern void	pgstrom_utilcmds_init(void);

#endif
/*
 * blkload.c
 */
#ifdef POSTGRES_H
extern Datum pgstrom_fdw_handler(PG_FUNCTION_ARGS);


#endif

/*
 * plan.c
 */
#ifdef POSTGRES_H
extern FdwPlan *
pgstrom_plan_foreign_scan(Oid foreignTblOid,
						  PlannerInfo *root, RelOptInfo *baserel);
extern void
pgstrom_explain_foreign_scan(ForeignScanState *node, ExplainState *es);
#endif

/*
 * scan.c
 */
#ifdef POSTGRES_H
extern void
pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags);
extern TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss);
extern void
pgboost_rescan_foreign_scan(ForeignScanState *fss);
extern void
pgboost_end_foreign_scan(ForeignScanState *fss);
#endif

/*
 * pg_strom.c
 */
#ifdef POSTGRES_H
extern FdwRoutine pgstromFdwHandlerData;
extern Datum pgstrom_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgstrom_fdw_validator(PG_FUNCTION_ARGS);
#endif

/*
 * cuda.c
 */
extern const char *
pgcuda_get_error_string(cudaError_t error);
extern cudaError_t
pgcuda_get_device_count(int *count);
extern cudaError_t
pgcuda_get_device_properties(struct cudaDeviceProp *prop, int device);

#endif	/* PG_STROM_H */
