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
	HeapScanDesc	scan;
	BlockNumber		blkno_begin;
	BlockNumber		blkno_end;
	int				num_tuples;
	int				num_valid;
	Datum		   *validmap;
	Datum		  **nullmap;
	void		  **columns;
} PgBoostScanChunk;

typedef struct
{
	HeapScanDesc		scandesc;
	List			   *scanchunk_list;
	PgBoostScanChunk   *scanchunk_current;
} PgBoostScanState;

typedef struct
{
	Oid					relationId;

	/*
	 * We should put vectorized qualifier here
	 */
} PgBoostPlanDesc;

/*
 * plan.c
 */
extern FdwPlan *PgBoostPlanForeignScan(Oid foreignTableId,
									   PlannerInfo *root,
									   RelOptInfo *baserel);
extern void		PgBoostExplainForeignScan(ForeignScanState *node,
										  ExplainState *es);
extern List	   *PgBoostPlanDescPack(PgBoostPlanDesc *plandesc);
extern PgBoostPlanDesc *PgBoostPlanDescUnpack(List *packed);

/*
 * scan.c
 */
extern void		PgBoostBeginForeignScan(ForeignScanState *node, int eflags);
extern TupleTableSlot  *PgBoostIterateForeignScan(ForeignScanState *node);
extern void		PgBoostReScanForeignScan(ForeignScanState *node);
extern void		PgBoostEndForeignScan(ForeignScanState *node);

/*
 * SQL Functions and module entrypoints
 */
extern Datum pg_boost_fdw_handler(PG_FUNCTION_ARGS);
extern Datum pg_boost_fdw_validator(PG_FUNCTION_ARGS);
extern void _PG_init(void);

#endif	/* PG_BOOST_H */
