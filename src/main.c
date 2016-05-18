/*
 * main.c
 *
 * Entrypoint of PG-Strom extension, and misc uncategolized functions.
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "postgres.h"
#include "access/hash.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/planner.h"
#include "parser/parsetree.h"
#include "storage/ipc.h"
#include "storage/pg_shmem.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/ruleutils.h"
#include <float.h>
#include <limits.h>
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * Misc static variables
 */
static planner_hook_type	planner_hook_next;

/*
 * miscellaneous GUC parameters
 */
bool		pgstrom_enabled;
bool		pgstrom_perfmon_enabled;
static bool	pgstrom_debug_kernel_source;
bool		pgstrom_bulkexec_enabled;
bool		pgstrom_cpu_fallback_enabled;
int			pgstrom_max_async_tasks;
double		pgstrom_num_threads_margin;
double		pgstrom_chunk_size_margin;

/* cost factors */
double		pgstrom_gpu_setup_cost;
double		pgstrom_gpu_dma_cost;
double		pgstrom_gpu_operator_cost;
double		pgstrom_gpu_tuple_cost;

static void
pgstrom_init_misc_guc(void)
{
	/* turn on/off PG-Strom feature */
	DefineCustomBoolVariable("pg_strom.enabled",
							 "Enables the planner's use of PG-Strom",
							 NULL,
							 &pgstrom_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off performance monitor on EXPLAIN ANALYZE */
	DefineCustomBoolVariable("pg_strom.perfmon",
							 "Enables the performance monitor of PG-Strom",
							 NULL,
							 &pgstrom_perfmon_enabled,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off bulkload feature to exchange PG-Strom nodes */
	DefineCustomBoolVariable("pg_strom.bulkexec",
							 "Enables the bulk-execution mode of PG-Strom",
							 NULL,
							 &pgstrom_bulkexec_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off CPU fallback if GPU could not execute the query */
	DefineCustomBoolVariable("pg_strom.cpu_fallback",
							 "Enables CPU fallback if GPU is ",
							 NULL,
							 &pgstrom_cpu_fallback_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off cuda kernel source saving */
	DefineCustomBoolVariable("pg_strom.debug_kernel_source",
							 "Turn on/off to display the kernel source path",
							 NULL,
							 &pgstrom_debug_kernel_source,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* maximum number of GpuTask can concurrently executed */
	DefineCustomIntVariable("pg_strom.max_async_tasks",
							"max number of GPU tasks to be run asynchronously",
							NULL,
							&pgstrom_max_async_tasks,
							32,
							4,
							INT_MAX,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* margin of number of CUDA threads */
	DefineCustomRealVariable("pg_strom.num_threads_margin",
							 "margin of number of CUDA threads if not predictable exactly",
							 NULL,
							 &pgstrom_num_threads_margin,
							 1.10,
							 1.00,	/* 0% margin - strict estimation */
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/**/
	DefineCustomRealVariable("pg_strom.chunk_size_margin",
							 "margin of chunk size if not predictable exactly",
							 NULL,
							 &pgstrom_chunk_size_margin,
							 1.25,
							 1.00,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for Gpu setup */
	DefineCustomRealVariable("pg_strom.gpu_setup_cost",
							 "Cost to setup GPU device to run",
							 NULL,
							 &pgstrom_gpu_setup_cost,
							 4000 * DEFAULT_SEQ_PAGE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for each Gpu task */
	DefineCustomRealVariable("pg_strom.gpu_dma_cost",
							 "Cost to send/recv data via DMA",
							 NULL,
							 &pgstrom_gpu_dma_cost,
							 10 * DEFAULT_SEQ_PAGE_COST,
							 0,
							 DBL_MAX,
                             PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);
	/* cost factor for Gpu operator */
	DefineCustomRealVariable("pg_strom.gpu_operator_cost",
							 "Cost of processing each operators by GPU",
							 NULL,
							 &pgstrom_gpu_operator_cost,
							 DEFAULT_CPU_OPERATOR_COST / 32.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor to process tuples in Gpu */
	DefineCustomRealVariable("pg_strom.gpu_tuple_cost",
							 "Cost of processing each tuple for GPU",
							 NULL,
							 &pgstrom_gpu_tuple_cost,
							 DEFAULT_CPU_TUPLE_COST / 32.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}

/*
 * pgstrom_recursive_grafter
 *
 * It tries to inject GpuPreAgg and GpuSort (these are not "officially"
 * supported by planner) on the pre-built plan tree.
 */
static void
pgstrom_recursive_grafter(PlannedStmt *pstmt, Plan *parent, Plan **p_curr_plan)
{
	Plan	   *plan = *p_curr_plan;
	ListCell   *lc;

	Assert(plan != NULL);

	switch (nodeTag(plan))
	{
		case T_Agg:
			/*
			 * Try to inject GpuPreAgg plan if cost of the aggregate plan
			 * is enough expensive to justify preprocess by GPU.
			 */
			pgstrom_try_insert_gpupreagg(pstmt, (Agg *) plan);
			break;

		case T_SubqueryScan:
			{
				SubqueryScan   *subquery = (SubqueryScan *) plan;
				Plan		  **p_subplan = &subquery->subplan;
				pgstrom_recursive_grafter(pstmt, plan, p_subplan);
			}
			break;
		case T_ModifyTable:
			{
				ModifyTable *mtplan = (ModifyTable *) plan;

				foreach (lc, mtplan->plans)
				{
					Plan  **p_subplan = (Plan **) &lfirst(lc);
					pgstrom_recursive_grafter(pstmt, plan, p_subplan);
				}
			}
			break;
		case T_Append:
			{
				Append *aplan = (Append *) plan;

				foreach (lc, aplan->appendplans)
				{
					Plan  **p_subplan = (Plan **) &lfirst(lc);
					pgstrom_recursive_grafter(pstmt, plan, p_subplan);
				}
			}
			break;
		case T_MergeAppend:
			{
				MergeAppend *maplan = (MergeAppend *) plan;

				foreach (lc, maplan->mergeplans)
				{
					Plan  **p_subplan = (Plan **) &lfirst(lc);
					pgstrom_recursive_grafter(pstmt, plan, p_subplan);
				}
			}
			break;
		case T_BitmapAnd:
			{
				BitmapAnd  *baplan = (BitmapAnd *) plan;

				foreach (lc, baplan->bitmapplans)
				{
					Plan  **p_subplan = (Plan **) &lfirst(lc);
					pgstrom_recursive_grafter(pstmt, plan, p_subplan);
				}
			}
			break;
		case T_BitmapOr:
			{
				BitmapOr   *boplan = (BitmapOr *) plan;

				foreach (lc, boplan->bitmapplans)
				{
					Plan  **p_subplan = (Plan **) &lfirst(lc);
					pgstrom_recursive_grafter(pstmt, plan, p_subplan);
				}
			}
			break;
		case T_CustomScan:
			{
				CustomScan *cscan = (CustomScan *) plan;

				foreach (lc, cscan->custom_plans)
				{
					Plan  **p_subplan = (Plan **) &lfirst(lc);
					pgstrom_recursive_grafter(pstmt, plan, p_subplan);
				}
			}
			break;
		default:
			/* nothing to do, keep existgin one */
			break;
	}

	/* also walk down left and right child plan sub-tree, if any */
	if (plan->lefttree)
		pgstrom_recursive_grafter(pstmt, plan, &plan->lefttree);
	if (plan->righttree)
		pgstrom_recursive_grafter(pstmt, plan, &plan->righttree);

	switch (nodeTag(plan))
	{
		case T_Sort:
			/*
			 * Heuristically, we should avoid to replace Sort-node just
			 * below the Limit-node, because Limit-node informs Sort-node
			 * minimum required number of rows then Sort-node takes special
			 * optimization. It is not easy to win with GpuSort...
			 */
			if (parent && IsA(parent, Limit))
				break;

			/*
			 * Try to replace Sort node by GpuSort node if cost of
			 * the alternative plan is enough reasonable to replace.
			 */
			pgstrom_try_insert_gpusort(pstmt, p_curr_plan);
			break;

		case T_CustomScan:
			if (pgstrom_plan_is_gpuscan(plan))
				pgstrom_post_planner_gpuscan(pstmt, p_curr_plan);
			else if (pgstrom_plan_is_gpujoin(plan))
				pgstrom_post_planner_gpujoin(pstmt, p_curr_plan);
			break;

		default:
			/* nothing to do, keep existing one */
			break;
	}
}

/*
 * pgstrom_planner_entrypoint
 *
 * It overrides the planner_hook for two purposes.
 * 1. To inject GpuPreAgg and GpuSort on the PlannedStmt once built.
 *    (Note that it is not a usual way to inject paths, so we have
 *     to be careful to inject it)
 */
static PlannedStmt *
pgstrom_planner_entrypoint(Query *parse,
						   int cursorOptions,
						   ParamListInfo boundParams)
{
	PlannedStmt	*result;

	if (planner_hook_next)
		result = planner_hook_next(parse, cursorOptions, boundParams);
	else
		result = standard_planner(parse, cursorOptions, boundParams);

	if (pgstrom_enabled)
	{
		ListCell   *cell;

		Assert(result->planTree != NULL);
		pgstrom_recursive_grafter(result, NULL, &result->planTree);

		foreach (cell, result->subplans)
		{
			Plan  **p_subplan = (Plan **) &cell->data.ptr_value;
			pgstrom_recursive_grafter(result, NULL, p_subplan);
		}
	}
	return result;
}

/*
 * _PG_init
 *
 * Main entrypoint of PG-Strom. It shall be invoked only once when postmaster
 * process is starting up, then it calls other sub-systems to initialize for
 * each ones.
 */
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

	/* dump version number */
	elog(LOG, "PG-Strom version %s built for PostgreSQL %s",
		 PGSTROM_VERSION, PG_MAJORVERSION);

	/* initialization of CUDA related stuff */
	pgstrom_init_cuda_control();
	pgstrom_init_cuda_program();
	/* initialization of data store support */
	pgstrom_init_datastore();

	/* registration of custom-scan providers */
	pgstrom_init_gpuscan();
	pgstrom_init_gpujoin();
	pgstrom_init_gpupreagg();
	pgstrom_init_gpusort();

	/* miscellaneous initializations */
	pgstrom_init_misc_guc();
	pgstrom_init_codegen();
	pgstrom_init_plcuda();

	/* overall planner hook registration */
	planner_hook_next = planner_hook;
	planner_hook = pgstrom_planner_entrypoint;
}

/* ------------------------------------------------------------
 *
 * Misc routines to support EXPLAIN command
 *
 * ------------------------------------------------------------
 */
void
pgstrom_explain_expression(List *expr_list, const char *qlabel,
						   PlanState *planstate, List *deparse_context,
						   List *ancestors, ExplainState *es,
						   bool force_prefix, bool convert_to_and)
{
	bool        useprefix;
	char       *exprstr;

	useprefix = (force_prefix || es->verbose);

	/* No work if empty expression list */
	if (expr_list == NIL)
		return;

	/* Deparse the expression */
	/* List shall be replaced by explicit AND, if needed */
	exprstr = deparse_expression(convert_to_and
								 ? (Node *) make_ands_explicit(expr_list)
								 : (Node *) expr_list,
								 deparse_context,
								 useprefix,
								 false);
	/* And add to es->str */
	ExplainPropertyText(qlabel, exprstr, es);
}

void
pgstrom_explain_outer_bulkexec(GpuTaskState *gts,
							   List *deparse_context,
							   List *ancestors,
							   ExplainState *es)
{
	Plan		   *plannode = gts->css.ss.ps.plan;
	Index			scanrelid = ((Scan *) plannode)->scanrelid;
	StringInfoData	str;

	/* Does this GpuTaskState has outer simple scan? */
	if (scanrelid == 0)
		return;

	/* Is it EXPLAIN ANALYZE? */
	if (!es->analyze)
		return;

	/*
	 * We have to forcibly clean up the instrumentation state because we
	 * haven't done ExecutorEnd yet.  This is pretty grotty ...
	 * See the comment in ExplainNode()
	 */
	InstrEndLoop(&gts->outer_instrument);

	/*
	 * See the logic in ExplainTargetRel()
	 */
	initStringInfo(&str);
	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		RangeTblEntry  *rte = rt_fetch(scanrelid, es->rtable);
		char		   *refname;
		char		   *relname;

		refname = (char *) list_nth(es->rtable_names, scanrelid - 1);
		if (refname == NULL)
			refname = rte->eref->aliasname;
		relname = get_rel_name(rte->relid);
		if (es->verbose)
		{
			char	   *nspname
				= get_namespace_name(get_rel_namespace(rte->relid));

			appendStringInfo(&str, "%s.%s",
							 quote_identifier(nspname),
							 quote_identifier(relname));
		}
		else if (relname != NULL)
			appendStringInfo(&str, "%s", quote_identifier(relname));
		if (strcmp(relname, refname) != 0)
			appendStringInfo(&str, " %s", quote_identifier(refname));
	}

	if (gts->outer_instrument.nloops > 0)
	{
		Instrumentation *instrument = &gts->outer_instrument;
		double		nloops = instrument->nloops;
		double		startup_sec = 1000.0 * instrument->startup / nloops;
		double		total_sec = 1000.0 * instrument->total / nloops;
		double		rows = instrument->ntuples / nloops;

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			if (es->timing)
				appendStringInfo(
					&str,
					" (actual time=%.3f..%.3f rows=%.0f loops=%.0f)",
					startup_sec, total_sec, rows, nloops);
		else
			appendStringInfo(
				&str,
				" (actual rows=%.0f loops=%.0f)",
				rows, nloops);
		}
		else
		{
			if (es->timing)
			{
				ExplainPropertyFloat("Outer Actual Startup Time",
									 startup_sec, 3, es);
				ExplainPropertyFloat("Outer Actual Total Time",
									 total_sec, 3, es);
			}
			ExplainPropertyFloat("Outer Actual Rows", rows, 0, es);
			ExplainPropertyFloat("Outer Actual Loops", nloops, 0, es);
		}
	}
	else
	{
		if (es->format == EXPLAIN_FORMAT_TEXT)
			appendStringInfoString(&str, " (never executed)");
		else
		{
			if (es->timing)
			{
				ExplainPropertyFloat("Outer Actual Startup Time", 0.0, 3, es);
				ExplainPropertyFloat("Outer Actual Total Time", 0.0, 3, es);
			}
			ExplainPropertyFloat("Outer Actual Rows", 0.0, 0, es);
			ExplainPropertyFloat("Outer Actual Loops", 0.0, 0, es);
		}
	}

	/*
	 * Logic copied from show_buffer_usage()
	 */
	if (es->buffers)
	{
		BufferUsage *usage = &gts->outer_instrument.bufusage;

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			bool	has_shared = (usage->shared_blks_hit > 0 ||
								  usage->shared_blks_read > 0 ||
								  usage->shared_blks_dirtied > 0 ||
								  usage->shared_blks_written > 0);
			bool	has_local = (usage->local_blks_hit > 0 ||
								 usage->local_blks_read > 0 ||
								 usage->local_blks_dirtied > 0 ||
							   	 usage->local_blks_written > 0);
			bool	has_temp = (usage->temp_blks_read > 0 ||
								usage->temp_blks_written > 0);
			bool	has_timing = (!INSTR_TIME_IS_ZERO(usage->blk_read_time) ||
								  !INSTR_TIME_IS_ZERO(usage->blk_write_time));

			/* Show only positive counter values. */
			if (has_shared || has_local || has_temp)
			{
				appendStringInfoChar(&str, '\n');
				appendStringInfoSpaces(&str, es->indent * 2 + 12);
				appendStringInfoString(&str, "buffers:");

				if (has_shared)
				{
					appendStringInfoString(&str, " shared");
					if (usage->shared_blks_hit > 0)
						appendStringInfo(&str, " hit=%ld",
										 usage->shared_blks_hit);
					if (usage->shared_blks_read > 0)
						appendStringInfo(&str, " read=%ld",
										 usage->shared_blks_read);
					if (usage->shared_blks_dirtied > 0)
						appendStringInfo(&str, " dirtied=%ld",
										 usage->shared_blks_dirtied);
					if (usage->shared_blks_written > 0)
						appendStringInfo(&str, " written=%ld",
										 usage->shared_blks_written);
					if (has_local || has_temp)
						appendStringInfoChar(&str, ',');
				}
				if (has_local)
				{
					appendStringInfoString(&str, " local");
					if (usage->local_blks_hit > 0)
						appendStringInfo(&str, " hit=%ld",
										 usage->local_blks_hit);
					if (usage->local_blks_read > 0)
						appendStringInfo(&str, " read=%ld",
										 usage->local_blks_read);
					if (usage->local_blks_dirtied > 0)
						appendStringInfo(&str, " dirtied=%ld",
										 usage->local_blks_dirtied);
					if (usage->local_blks_written > 0)
						appendStringInfo(&str, " written=%ld",
										 usage->local_blks_written);
					if (has_temp)
						appendStringInfoChar(&str, ',');
				}
				if (has_temp)
				{
					appendStringInfoString(&str, " temp");
					if (usage->temp_blks_read > 0)
						appendStringInfo(&str, " read=%ld",
										 usage->temp_blks_read);
					if (usage->temp_blks_written > 0)
						appendStringInfo(&str, " written=%ld",
										 usage->temp_blks_written);
				}
			}

			/* As above, show only positive counter values. */
			if (has_timing)
			{
				if (has_shared || has_local || has_temp)
					appendStringInfo(&str, ", ");
				appendStringInfoString(&str, "I/O Timings:");
				if (!INSTR_TIME_IS_ZERO(usage->blk_read_time))
					appendStringInfo(&str, " read=%0.3f",
							INSTR_TIME_GET_MILLISEC(usage->blk_read_time));
				if (!INSTR_TIME_IS_ZERO(usage->blk_write_time))
					appendStringInfo(&str, " write=%0.3f",
							INSTR_TIME_GET_MILLISEC(usage->blk_write_time));
			}
		}
		else
		{
			double		time_value;
			ExplainPropertyLong("Outer Shared Hit Blocks",
								usage->shared_blks_hit, es);
			ExplainPropertyLong("Outer Shared Read Blocks",
								usage->shared_blks_read, es);
			ExplainPropertyLong("Outer Shared Dirtied Blocks",
								usage->shared_blks_dirtied, es);
			ExplainPropertyLong("Outer Shared Written Blocks",
								usage->shared_blks_written, es);
			ExplainPropertyLong("Outer Local Hit Blocks",
								usage->local_blks_hit, es);
			ExplainPropertyLong("Outer Local Read Blocks",
								usage->local_blks_read, es);
			ExplainPropertyLong("Outer Local Dirtied Blocks",
								usage->local_blks_dirtied, es);
			ExplainPropertyLong("Outer Local Written Blocks",
								usage->local_blks_written, es);
			ExplainPropertyLong("Outer Temp Read Blocks",
								usage->temp_blks_read, es);
			ExplainPropertyLong("Outer Temp Written Blocks",
								usage->temp_blks_written, es);
			time_value = INSTR_TIME_GET_MILLISEC(usage->blk_read_time);
			ExplainPropertyFloat("Outer I/O Read Time", time_value, 3, es);
			time_value = INSTR_TIME_GET_MILLISEC(usage->blk_write_time);
			ExplainPropertyFloat("Outer I/O Write Time", time_value, 3, es);
		}
	}

	if (es->format == EXPLAIN_FORMAT_TEXT)
		ExplainPropertyText("Outer Scan", str.data, es);
}

void
show_scan_qual(List *qual, const char *qlabel,
               PlanState *planstate, List *ancestors,
               ExplainState *es)
{
	bool        useprefix;
	Node	   *node;
	List       *context;
	char       *exprstr;

	useprefix = (IsA(planstate->plan, SubqueryScan) || es->verbose);

	/* No work if empty qual */
	if (qual == NIL)
		return;

	/* Convert AND list to explicit AND */
	node = (Node *) make_ands_explicit(qual);

	/* Set up deparsing context */
	context = set_deparse_context_planstate(es->deparse_cxt,
											(Node *) planstate,
											ancestors);
	/* Deparse the expression */
	exprstr = deparse_expression(node, context, useprefix, false);

	/* And add to es->str */
	ExplainPropertyText(qlabel, exprstr, es);
}

/*
 * If it's EXPLAIN ANALYZE, show instrumentation information for a plan node
 *
 * "which" identifies which instrumentation counter to print
 */
void
show_instrumentation_count(const char *qlabel, int which,
						   PlanState *planstate, ExplainState *es)
{
	double		nfiltered;
	double		nloops;

	if (!es->analyze || !planstate->instrument)
		return;

	if (which == 2)
		nfiltered = planstate->instrument->nfiltered2;
	else
		nfiltered = planstate->instrument->nfiltered1;
	nloops = planstate->instrument->nloops;

	/* In text mode, suppress zero counts; they're not interesting enough */
	if (nfiltered > 0 || es->format != EXPLAIN_FORMAT_TEXT)
	{
		if (nloops > 0)
			ExplainPropertyFloat(qlabel, nfiltered / nloops, 0, es);
		else
			ExplainPropertyFloat(qlabel, 0.0, 0, es);
	}
}

void
pgstrom_init_perfmon(GpuTaskState *gts)
{
	GpuContext	   *gcontext = gts->gcontext;

	memset(&gts->pfm, 0, sizeof(pgstrom_perfmon));
	gts->pfm.enabled = pgstrom_perfmon_enabled;
	gts->pfm.prime_in_gpucontext = (gcontext && gcontext->refcnt == 1);
	gts->pfm.extra_flags = gts->extra_flags;
}

static void
pgstrom_explain_perfmon(GpuTaskState *gts, ExplainState *es)
{
	pgstrom_perfmon	   *pfm = &gts->pfm;
	char				buf[1024];

	if (!pfm->enabled)
		return;

	/* common performance statistics */
	ExplainPropertyInteger("Number of tasks", pfm->num_tasks, es);

#define EXPLAIN_KERNEL_PERFMON(label,num_field,tv_field)		\
	do {														\
		if (pfm->num_field > 0)									\
		{														\
			snprintf(buf, sizeof(buf),							\
					 "total: %s, avg: %s, count: %u",			\
					 format_millisec(pfm->tv_field),			\
					 format_millisec(pfm->tv_field /			\
									 (double)pfm->num_field),	\
					 pfm->num_field);							\
			ExplainPropertyText(label, buf, es);				\
		}														\
	} while(0)

	/* GpuScan: kernel execution */
	if ((pfm->extra_flags & DEVKERNEL_NEEDS_GPUSCAN) != 0)
	{
		EXPLAIN_KERNEL_PERFMON("gpuscan_exec_quals",
							   gscan.num_kern_exec_quals,
							   gscan.tv_kern_exec_quals);
		EXPLAIN_KERNEL_PERFMON("gpuscan_projection",
							   gscan.num_kern_projection,
							   gscan.tv_kern_projection);
	}

	/* GpuJoin: kernel execution */
	if ((pfm->extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
	{
		EXPLAIN_KERNEL_PERFMON("gpujoin_main()",
							   gjoin.num_kern_main,
							   gjoin.tv_kern_main);
		EXPLAIN_KERNEL_PERFMON(" - gpujoin_exec_outerscan",
							   gjoin.num_kern_outer_scan,
							   gjoin.tv_kern_outer_scan);
		EXPLAIN_KERNEL_PERFMON(" - gpujoin_exec_nestloop",
							   gjoin.num_kern_exec_nestloop,
							   gjoin.tv_kern_exec_nestloop);
		EXPLAIN_KERNEL_PERFMON(" - gpujoin_exec_hashjoin",
							   gjoin.num_kern_exec_hashjoin,
							   gjoin.tv_kern_exec_hashjoin);
		EXPLAIN_KERNEL_PERFMON(" - gpujoin_outer_nestloop",
							   gjoin.num_kern_outer_nestloop,
							   gjoin.tv_kern_outer_nestloop);
		EXPLAIN_KERNEL_PERFMON(" - gpujoin_outer_hashjoin",
							   gjoin.num_kern_outer_hashjoin,
							   gjoin.tv_kern_outer_hashjoin);
		EXPLAIN_KERNEL_PERFMON(" - gpujoin_projection",
							   gjoin.num_kern_projection,
							   gjoin.tv_kern_projection);
		EXPLAIN_KERNEL_PERFMON(" - gpujoin_count_rows_dist",
							   gjoin.num_kern_rows_dist,
							   gjoin.tv_kern_rows_dist);
		if (pfm->gjoin.num_global_retry > 0 ||
			pfm->gjoin.num_major_retry > 0 ||
			pfm->gjoin.num_minor_retry > 0)
		{
			snprintf(buf, sizeof(buf), "global: %u, major: %u, minor: %u",
					 pfm->gjoin.num_global_retry,
					 pfm->gjoin.num_major_retry,
					 pfm->gjoin.num_minor_retry);
			ExplainPropertyText("Retry Loops", buf, es);
		}
	}

	/* GpuPreAgg: kernel execution */
	if ((pfm->extra_flags & DEVKERNEL_NEEDS_GPUPREAGG) != 0)
	{
		EXPLAIN_KERNEL_PERFMON("gpupreagg_main()",
							   gpreagg.num_kern_main,
							   gpreagg.tv_kern_main);
		EXPLAIN_KERNEL_PERFMON(" - gpupreagg_preparation()",
							   gpreagg.num_kern_prep,
							   gpreagg.tv_kern_prep);
		EXPLAIN_KERNEL_PERFMON(" - gpupreagg_nogroup_reduction()",
							   gpreagg.num_kern_nogrp,
							   gpreagg.tv_kern_nogrp);
		EXPLAIN_KERNEL_PERFMON(" - gpupreagg_local_reduction()",
							   gpreagg.num_kern_lagg,
							   gpreagg.tv_kern_lagg);
		EXPLAIN_KERNEL_PERFMON(" - gpupreagg_global_reduction()",
							   gpreagg.num_kern_gagg,
							   gpreagg.tv_kern_gagg);
		EXPLAIN_KERNEL_PERFMON(" - gpupreagg_final_reduction()",
							   gpreagg.num_kern_fagg,
							   gpreagg.tv_kern_fagg);
		EXPLAIN_KERNEL_PERFMON(" - gpupreagg_fixup_varlena()",
							   gpreagg.num_kern_fixvar,
							   gpreagg.tv_kern_fixvar);
	}

	/* GpuSort: kernel execution */
	if ((pfm->extra_flags & DEVKERNEL_NEEDS_GPUSORT) != 0)
	{
		EXPLAIN_KERNEL_PERFMON("gpusort_projection()",
							   gsort.num_kern_proj,
							   gsort.tv_kern_proj);
		EXPLAIN_KERNEL_PERFMON("gpusort_main()",
							   gsort.num_kern_main,
							   gsort.tv_kern_main);
		EXPLAIN_KERNEL_PERFMON(" - gpusort_bitonic_local()",
							   gsort.num_kern_lsort,
							   gsort.tv_kern_lsort);
		EXPLAIN_KERNEL_PERFMON(" - gpusort_bitonic_step()",
							   gsort.num_kern_ssort,
							   gsort.tv_kern_ssort);
		EXPLAIN_KERNEL_PERFMON(" - gpusort_bitonic_merge()",
							   gsort.num_kern_msort,
							   gsort.tv_kern_msort);
		EXPLAIN_KERNEL_PERFMON(" - gpusort_fixup_pointers()",
							   gsort.num_kern_fixvar,
							   gsort.tv_kern_fixvar);
		snprintf(buf, sizeof(buf), "total: %s",
				 format_millisec(pfm->gsort.tv_cpu_sort));
		ExplainPropertyText("CPU merge sort", buf, es);
	}

#undef EXPLAIN_KERNEL_PERFMON
	/* Time of I/O stuff */
	if ((pfm->extra_flags & DEVKERNEL_NEEDS_GPUJOIN) != 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 format_millisec(pfm->time_inner_load));
		ExplainPropertyText("Time of inner load", buf, es);
		snprintf(buf, sizeof(buf), "%s",
				 format_millisec(pfm->time_outer_load));
		ExplainPropertyText("Time of outer load", buf, es);
	}
	else
	{
		snprintf(buf, sizeof(buf), "%s",
				 format_millisec(pfm->time_outer_load));
		ExplainPropertyText("Time of load", buf, es);
	}

	snprintf(buf, sizeof(buf), "%s",
			 format_millisec(pfm->time_materialize));
	ExplainPropertyText("Time of materialize", buf, es);

	/* DMA Send/Recv performance */
	if (pfm->num_dma_send > 0)
	{
		Size	band = (Size)((double)pfm->bytes_dma_send *
							  1000.0 / pfm->time_dma_send);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 format_bytesz(band),
				 format_bytesz((double)pfm->bytes_dma_send),
				 format_millisec(pfm->time_dma_send),
				 pfm->num_dma_send);
		ExplainPropertyText("DMA send", buf, es);
	}

	if (pfm->num_dma_recv > 0)
	{
		Size	band = (Size)((double)pfm->bytes_dma_recv *
							  1000.0 / pfm->time_dma_recv);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 format_bytesz(band),
				 format_bytesz((double)pfm->bytes_dma_recv),
				 format_millisec(pfm->time_dma_recv),
				 pfm->num_dma_recv);
		ExplainPropertyText("DMA recv", buf, es);
	}

	/* Time to build CUDA code */
	if (pfm->tv_build_start.tv_sec > 0 &&
		pfm->tv_build_end.tv_sec > 0 &&
		(pfm->tv_build_start.tv_sec < pfm->tv_build_end.tv_sec ||
		 (pfm->tv_build_start.tv_sec == pfm->tv_build_end.tv_sec &&
		  pfm->tv_build_start.tv_usec < pfm->tv_build_end.tv_usec)))
	{
		cl_double	tv_cuda_build = PFMON_TIMEVAL_DIFF(&pfm->tv_build_start,
													   &pfm->tv_build_end);
		snprintf(buf, sizeof(buf), "%s", format_millisec(tv_cuda_build));
		ExplainPropertyText("Build CUDA Program", buf, es);
	}

	/* Host/Device Memory Allocation (only prime node) */
	if (pfm->prime_in_gpucontext)
	{
		GpuContext *gcontext = gts->gcontext;
		cl_int		num_host_malloc = *gcontext->p_num_host_malloc;
		cl_int		num_host_mfree = *gcontext->p_num_host_mfree;
		cl_int		num_dev_malloc = gcontext->num_dev_malloc;
		cl_int		num_dev_mfree = gcontext->num_dev_mfree;
		cl_double	tv_host_malloc =
			PFMON_TIMEVAL_AS_FLOAT(gcontext->p_tv_host_malloc);
		cl_double	tv_host_mfree =
			PFMON_TIMEVAL_AS_FLOAT(gcontext->p_tv_host_mfree);
		cl_double	tv_dev_malloc =
			PFMON_TIMEVAL_AS_FLOAT(&gcontext->tv_dev_malloc);
		cl_double	tv_dev_mfree =
			PFMON_TIMEVAL_AS_FLOAT(&gcontext->tv_dev_mfree);

		snprintf(buf, sizeof(buf),
				 "alloc (count: %u, time: %s), free (count: %u, time: %s)",
				 num_host_malloc, format_millisec(tv_host_malloc),
				 num_host_mfree, format_millisec(tv_host_mfree));
		ExplainPropertyText("CUDA host memory", buf, es);

		snprintf(buf, sizeof(buf),
				 "alloc (count: %u, time: %s), free (count: %u, time: %s)",
				 num_dev_malloc, format_millisec(tv_dev_malloc),
				 num_dev_mfree, format_millisec(tv_dev_mfree));
		ExplainPropertyText("CUDA device memory", buf, es);
	}
}

/*
 * pgstrom_explain_gputaskstate
 *
 * common additional explain output for all the GpuTaskState nodes
 */
void
pgstrom_explain_gputaskstate(GpuTaskState *gts, ExplainState *es)
{
	/*
	 * Extra features if any
	 */
	if (es->verbose)
	{
		char	temp[256];
		int		ofs = 0;

		/* run per-chunk-execution? */
		if (gts->outer_bulk_exec)
			ofs += snprintf(temp+ofs, sizeof(temp) - ofs,
							"%souter-bulk-exec",
							ofs > 0 ? ", " : "");
		/* per-chunk-execution support? */
		if (gts->cb_bulk_exec != NULL)
			ofs += snprintf(temp+ofs, sizeof(temp) - ofs,
							"%sbulk-exec-support",
							ofs > 0 ? ", " : "");
		/* preferable result format */
		if (gts->be_row_format)
			ofs += snprintf(temp+ofs, sizeof(temp) - ofs, "%srow-format",
							ofs > 0 ? ", " : "");
		if (ofs > 0)
			ExplainPropertyText("Extra", temp, es);
	}

	/*
	 * Show source path of the GPU kernel
	 */
	if (es->verbose &&
		gts->kern_source != NULL &&
		pgstrom_debug_kernel_source)
	{
		const char *cuda_source = pgstrom_cuda_source_file(gts);

		ExplainPropertyText("Kernel Source", cuda_source, es);
	}

	/*
	 * Show performance information
	 */
	if (es->analyze && gts->pfm.enabled)
		pgstrom_explain_perfmon(gts, es);
}
