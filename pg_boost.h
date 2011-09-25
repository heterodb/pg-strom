/*
 * pg_boost.h - Header file of pg_boost module
 *
 * Copyright (c) 2011 KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 */
#ifndef PG_BOOST_H
#define PG_BOOST_H
#include "fmgr.h"

typedef struct
{
	Oid			typeid;		/* OID of pg_type */
	int32		length;		/* length of this vector */
	/*
	 * XXX - negative length means it is a const value, that actually has
	 * one element, however, operators presume it is abs(length) vector
	 * with the const value for each elements.
	 */
	union {
		int8			v_bool[0];		/* BOOLOID */
		int16			v_int2[0];		/* INT2OID */
		int32			v_int4[0];		/* INT4OID */
		int64			v_int8[0];		/* INT8OID */
		float			v_float4[0];	/* FLOAT4OID */
		double			v_float8[0];	/* FLOAT8OID */
		DateADT			v_date[0];		/* DATEOID */
		TimeADT			v_time[0];		/* TIMEOID */
		Timestamp		v_timestamp[0];	/* TIMESTAMPOID */
		Oid				v_oid[0];		/* OIDOID, CIDOID, XIDOID */
		ItemPointerData	v_tid[0];		/* TIDOID */
		void		   *v_varlena[0];	/* TEXTOID, ... */
	} v;
} vector_t;

typedef struct
{
	BlockNumber		blkno_begin;
	BlockNumber		blkno_end;
	List		   *buffer_list;
	int				num_tuples;
	int				num_valid;
	vector_t	   *valid;
	vector_t	  **nulls;
	vector_t	  **attrs;
} scan_chunk_t;

typedef struct
{
	HeapScanDesc	scan;
	List		   *vquals;
	Bitmapset	   *referenced;
	AttrNumber		min_attr;
	AttrNumber		max_attr;
	AttrNumber	   *map_attr;
	int32			chunk_size;
	List		   *chunk_list;
	scan_chunk_t   *chunk_curr;
	int32			index_curr;
} scan_state_t;

/*
 * plan.c
 */
extern void
pgboost_extract_plan_info(scan_state_t *scan_state,
                          List *plan_info);
extern FdwPlan *
pgboost_plan_foreign_scan(Oid foreignTableId,
						  PlannerInfo *root,
						  RelOptInfo *baserel);
extern void
pgboost_explain_foreign_scan(ForeignScanState *node,
							 ExplainState *es);

/*
 * scan.c
 */
extern void
pgboost_begin_foreign_scan(ForeignScanState *node, int eflags);
extern TupleTableSlot *
pgboost_iterate_foreign_scan(ForeignScanState *node);
extern void
pgboost_rescan_foreign_scan(ForeignScanState *node);
extern void
pgboost_end_foreign_scan(ForeignScanState *node);

/*
 * vector.c
 */
extern vector_t *
pgboost_vector_alloc(Oid typeId, int32 length);
extern void
pgboost_vector_free(vector_t *vector);
extern Datum
pgboost_vector_getval(vector_t *vec, int index);
extern void
pgboost_vector_setval(vector_t *vec, int index, Datum datum);
/*
 * SQL Functions and module entrypoints
 */
extern Datum pgboost_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pgboost_fdw_validator(PG_FUNCTION_ARGS);
extern void _PG_init(void);

#endif	/* PG_BOOST_H */
