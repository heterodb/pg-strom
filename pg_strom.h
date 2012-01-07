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

#include "catalog/indexing.h"
#include "commands/explain.h"
#include "fmgr.h"
#include "foreign/fdwapi.h"
#include "utils/memutils.h"
#include <cuda.h>

#define PGSTROM_SCHEMA_NAME		"pg_strom"

#define PGSTROM_THREADS_PER_BLOCK	32

/*
 * utilcmds.c
 */
typedef struct {
	Relation	base_rel;
	Relation	rowid_rel;
	Relation	rowid_idx;
	Relation   *cs_rel;
	Relation   *cs_idx;
	Oid			rowid_seqid;
} RelationSetData;
typedef RelationSetData *RelationSet;

extern RelationSet	pgstrom_open_relation_set(Relation base_rel,
											  LOCKMODE lockmode,
											  bool with_index);
extern void	pgstrom_close_relation_set(RelationSet relset,
									   LOCKMODE lockmode);
extern void	pgstrom_utilcmds_init(void);

/*
 * blkload.c
 */
extern Datum pgstrom_data_load(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_clear(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_compaction(PG_FUNCTION_ARGS);

/*
 * plan.c
 */
extern FdwPlan *pgstrom_plan_foreign_scan(Oid foreignTblOid,
										  PlannerInfo *root,
										  RelOptInfo *baserel);
extern void		pgstrom_explain_foreign_scan(ForeignScanState *node,
											 ExplainState *es);

/*
 * scan.c
 */
extern void	pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags);
extern TupleTableSlot *pgstrom_iterate_foreign_scan(ForeignScanState *fss);
extern void	pgstrom_rescan_foreign_scan(ForeignScanState *fss);
extern void	pgstrom_end_foreign_scan(ForeignScanState *fss);
extern List *pgstrom_scan_debug_info(List *debug_info_list);
extern void pgstrom_scan_init(void);

/*
 * devinfo.c
 */
typedef struct {
	Oid		type_oid;
	char   *type_ident;
	char   *type_source;
	char   *type_varref;
	uint32	type_flags;			/* set of DEVINFO_FLAGS_* */
} PgStromDevTypeInfo;

typedef struct {
	Oid		func_oid;
	char	func_kind;			/* see the comments in devinfo.c */
	char   *func_ident;			/* identifier of device function */
	char   *func_source;		/* definition of device function, if needed */
	uint32	func_flags;			/* set of DEVINFO_FLAGS_* */
	int16	func_nargs;			/* copy from pg_proc.pronargs */
	Oid		func_argtypes[0];	/* copy from pg_proc.proargtypes */
} PgStromDevFuncInfo;

#define DEVINFO_FLAGS_DOUBLE_FP				0x0001
#define DEVINFO_FLAGS_INC_MATHFUNC_H		0x0002

extern void pgstrom_devtype_format(StringInfo str,
								   Oid type_oid, Datum value);
extern PgStromDevTypeInfo *pgstrom_devtype_lookup(Oid type_oid);
extern PgStromDevFuncInfo *pgstrom_devfunc_lookup(Oid func_oid);
extern void pgstrom_set_device_context(int dev_index);
extern int	pgstrom_get_num_devices(void);
extern void pgstrom_devinfo_init(void);
extern const char *cuda_error_to_string(CUresult result);

/*
 * nvcc.c
 */
extern void *pgstrom_nvcc_kernel_build(const char *kernel_source);
extern void pgstrom_nvcc_init(void);

/*
 * pg_strom.c
 */
extern FdwRoutine pgstromFdwHandlerData;
extern Datum pgstrom_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgstrom_fdw_validator(PG_FUNCTION_ARGS);
extern Datum pgstrom_debug_info(PG_FUNCTION_ARGS);

#endif	/* PG_STROM_H */
