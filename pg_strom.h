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
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define PGSTROM_SCHEMA_NAME		"pg_strom"

#define PGSTROM_CHUNK_SIZE		(2400 * (BLCKSZ / 8192))
#define PGSTROM_UNIT_SIZE(attr)								\
	(((attr)->attlen > 0 ?									\
	  (PGSTROM_CHUNK_SIZE / (attr)->attlen) :				\
	  (PGSTROM_CHUNK_SIZE / 100)) & ~(BITS_PER_BYTE - 1))

/*
 * utilcmds.c
 */
extern void
pgstrom_utilcmds_init(void);

/*
 * blkload.c
 */
extern Datum pgstrom_data_load(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_clear(PG_FUNCTION_ARGS);
extern Datum pgstrom_data_compaction(PG_FUNCTION_ARGS);

/*
 * plan.c
 */
extern FdwPlan *
pgstrom_plan_foreign_scan(Oid foreignTblOid,
			  PlannerInfo *root, RelOptInfo *baserel);

/*
 * scan.c
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

extern RelationSet
pgstrom_open_relation_set(Relation base_rel,
						  LOCKMODE lockmode, bool with_index);
extern void
pgstrom_close_relation_set(RelationSet relset, LOCKMODE lockmode);

extern void
pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags);
extern TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss);
extern void
pgboost_rescan_foreign_scan(ForeignScanState *fss);
extern void
pgboost_end_foreign_scan(ForeignScanState *fss);
extern void
pgstrom_explain_foreign_scan(ForeignScanState *node, ExplainState *es);

/*
 * devinfo.c
 */
typedef struct {
	Oid		type_oid;
	char   *type_ident;
	char   *type_source;
} PgStromDevTypeInfo;

typedef struct {
	Oid		func_oid;
	char	func_kind;			/* 'c', 'b', 'l', 'r' or 'f' */
	char   *func_ident;
	char   *func_source;
	int16	func_nargs;			/* copy from pg_proc.pronargs */
	Oid		func_argtypes[0];	/* copy from pg_proc.proargtypes */
} PgStromDevFuncInfo;

typedef struct {
	Oid		cast_source;
	Oid		cast_target;
	char   *func_ident;
	char   *func_source;
} PgStromDevCastInfo;

typedef struct {
	cl_platform_id				pf_id;
	cl_device_id				dev_id;
	cl_bool						dev_compiler_available;
	cl_device_fp_config			dev_double_fp_config;
	cl_ulong					dev_global_mem_cache_size;
	cl_device_mem_cache_type	dev_global_mem_cache_type;
	cl_ulong					dev_global_mem_size;
	cl_ulong					dev_local_mem_size;
	cl_device_local_mem_type	dev_local_mem_type;
	cl_uint						dev_max_clock_frequency;
	cl_uint						dev_max_compute_units;
	cl_uint						dev_max_constant_args;
	cl_ulong					dev_max_constant_buffer_size;
	cl_ulong					dev_max_mem_alloc_size;
	size_t						dev_max_parameter_size;
	size_t						dev_max_work_group_size;
	size_t						dev_max_work_item_dimensions;
	size_t						dev_max_work_item_sizes[3];
	char						dev_name[256];
	char						dev_version[256];
	char						dev_profile[24];
} PgStromDeviceInfo;

extern cl_uint				pgstrom_num_devices;
extern cl_device_id		   *pgstrom_device_id;
extern PgStromDeviceInfo  **pgstrom_device_info;
extern cl_context			pgstrom_device_context;

extern void pgstrom_devtype_format(StringInfo str,
								   Oid type_oid, Datum value);
extern PgStromDevTypeInfo *pgstrom_devtype_lookup(Oid type_oid);
extern PgStromDevFuncInfo *pgstrom_devfunc_lookup(Oid func_oid);
extern PgStromDevCastInfo *pgstrom_devcast_lookup(Oid source_typeid,
												  Oid target_typeid);
extern void pgstrom_devinfo_init(void);
extern const char *opencl_error_to_string(cl_int errcode);

/*
 * pg_strom.c
 */
extern FdwRoutine pgstromFdwHandlerData;
extern Datum pgstrom_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgstrom_fdw_validator(PG_FUNCTION_ARGS);

#endif	/* PG_STROM_H */
