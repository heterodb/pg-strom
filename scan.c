/*
 * scan.c
 *
 * routines to scan shared memory buffer
 *
 * Copyright (C) 2011 - 2012 KaiGai Kohei <kaigai@kaigai.gr.jp>
 */
#include "postgres.h"
#include "pg_boost.h"







static void
pgboost_fetch_next_chunk(PgBoostScanState *scan_state)
{
	/*
	 * (1) allocate a new pgboost_scan_chunk_t 
	 * (2) read blkno_begin to blkno_end
	 * (3) setup columns array
	 */


}

void
PgBoostBeginForeignScan(ForeignScanState *fss, int eflags)
{
	// FIXME: fdw_private must be copied by copy_object
	PgBoostPlanDesc	   *plan_desc = ((FdwPlan *)fss->ss.ps.plan)->fdw_private;
	PgBoostScanState   *scan_state;
	EState			   *estate = fss->ss.ps.state;
	Relation			relation;

	/*
	 * TODO:
	 * Permission check on the master relation here
	 */

	scanState = palloc(sizeof(PgBoostScanState));

	relation = ExecOpenScanRelation(fss->ss.ps.state,
									plan_desc->relationId);
	scan_state->scandesc = heap_beginscan(relation,
										  estate->es_snapshot,
										  0, NULL);
	scan_state->scanchunk_list = NIL;
	scan_state->scanchunk_current = NULL;

	// FIXME: need to check both relations are compatible
	ExecAssignScanType(node, RelationGetDescr(currentRelation));
}

TupleTableSlot*
PgBoostIterateForeignScan(ForeignScanState *node)
{
	PgBoostScanState;
	if (not_buffer_scan_yet)
	{
		do_buffer_scan();

		exec_vectorized_qual();
	}

	pop_first_matched_tuple;

}

void
PgBoostEndForeignScan(ForeignScanState *node)
{


}
