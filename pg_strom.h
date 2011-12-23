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
extern void
pgstrom_explain_foreign_scan(ForeignScanState *node, ExplainState *es);

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

/*
 * devinfo.c
 */
extern List *pgstrom_device_into_list;
extern void pgstrom_device_info_init(void);

/*
 * catalog.c
 */
/*
 * data types supported by pg_strom
 */
typedef struct {
	Oid			type_oid;
	const char *type_ident;
	const char *type_source;
} PgStromTypeCatalog;

typedef struct {
	/* oid of pg_proc catalog */
	Oid			func_oid;

	/*
	 * 'c' : function by device const
	 * 'l' : function by device left-operator
	 * 'r' : function by device right-operator
	 * 'b' : function by device both-operator
	 * 'f' : function by device function
	 */
	char		func_kind;

	/* number of arguments */
	int16		func_nargs;

	/* identifier of device function */
	const char *func_ident;

	/* declaration of device function, or NULL */
	const char *func_source;

	/*
	 * this argmap allows to switch order of arguments.
	 * e.g) SQL func(a,b,c) -> DEV func(c,a,b)
	 *      func_argmap should be {3,1,2}
	 */
	int16		func_argmap[0];
} PgStromFuncCatalog;

typedef struct {
	Oid			oper_oid;
	char		oper_kind;
	int16		oper_nargs;
	const char *oper_ident;
	const char *oper_source;
	int16		oper_argmap[2];
} PgStromOperCatalog;

typedef struct {
	Oid			cast_source;
	Oid			cast_target;
	const char *cast_func;
} PgStromCastCatalog;

extern PgStromTypeCatalog *pgstrom_type_catalog_lookup(Oid type_oid);
extern PgStromFuncCatalog *pgstrom_func_catalog_lookup(Oid func_oid);
extern PgStromCastCatalog *pgstrom_cast_catalog_lookup(Oid source_typeid,
													   Oid target_typeid);
extern PgStromOperCatalog *pgstrom_oper_catalog_lookup(Oid oper_oid);
extern void				   pgstrom_catalog_init(void);

/*
 * pg_strom.c
 */
extern FdwRoutine pgstromFdwHandlerData;
extern Datum pgstrom_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgstrom_fdw_validator(PG_FUNCTION_ARGS);

#endif	/* PG_STROM_H */
