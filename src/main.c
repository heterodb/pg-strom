/*
 * main.c
 *
 * Entrypoint of PG-Strom extension, and misc uncategolized functions.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "fmgr.h"
#include "miscadmin.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "storage/ipc.h"
#include "storage/pg_shmem.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/ruleutils.h"
#include <float.h>
#include <limits.h>
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * miscellaneous GUC parameters
 */
static bool	guc_pgstrom_enabled;
static bool guc_pgstrom_enabled_global;
bool	pgstrom_perfmon_enabled;
bool	pgstrom_debug_bulkload_enabled;
bool	pgstrom_debug_kernel_source;
int		pgstrom_max_async_chunks;
int		pgstrom_min_async_chunks;

/* cost factors */
double	pgstrom_gpu_setup_cost;
double	pgstrom_gpu_operator_cost;
double	pgstrom_gpu_tuple_cost;

/* buffer usage */
double	pgstrom_row_population_max;
double	pgstrom_row_population_margin;

/*
 * global_guc_values - segment for global GUC variables
 *
 * pg_strom.enabled_global turns on/off PG-Strom functionality of
 * all the concurrent backend, for test/benchmark purpose mainly.
 */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static struct {
	slock_t		lock;
	bool		pgstrom_enabled_global;
} *global_guc_values = NULL;

/*
 * wrapper of pg_strom.enabled and pg_strom.enabled_global configuration
 */
bool
pgstrom_enabled(void)
{
	bool	rc = false;

	if (guc_pgstrom_enabled)
	{
		SpinLockAcquire(&global_guc_values->lock);
		rc = global_guc_values->pgstrom_enabled_global;
		SpinLockRelease(&global_guc_values->lock);
	}
	return rc;
}

/*
 * assign callback of pg_strom.enabled_global
 */
static void
pg_strom_enabled_global_assign(bool newval, void *extra)
{
	/*
	 * NOTE: we cannot save state on the shared memory segment
	 * during _PG_init() because it is not initialized yet.
	 * So, we once save the "pg_strom.enabled_global" state on
	 * a private variable, then initialize the shared state
	 * using this private variable. Once shared memory segment
	 * is allocated, we shall reference the shared memory side.
	 *
	 * Also note that some worker process may detach shared-
	 * memory segment on starting-up, so we also need to check
	 * the shared memory segment is still valid. PG-Strom works
	 * only backend process with valid shared memory segment.
	 * So, here is no actual problem even if dummy behavior.
	 */
	if (!UsedShmemSegAddr || !global_guc_values)
		guc_pgstrom_enabled_global = newval;
	else
	{
		SpinLockAcquire(&global_guc_values->lock);
		global_guc_values->pgstrom_enabled_global = newval;
		SpinLockRelease(&global_guc_values->lock);
	}
}

/*
 * show callback of pg_strom.enabled_global
 */
static const char *
pg_strom_enabled_global_show(void)
{
	bool	state;

	if (!UsedShmemSegAddr || !global_guc_values)
		state = guc_pgstrom_enabled_global;	/* private variable! */
	else
	{
		SpinLockAcquire(&global_guc_values->lock);
		state = global_guc_values->pgstrom_enabled_global;
		SpinLockRelease(&global_guc_values->lock);
	}
	return state ? "on" : "off";
}

static void
pgstrom_init_misc_guc(void)
{
	/* turn on/off PG-Strom feature */
	DefineCustomBoolVariable("pg_strom.enabled",
							 "Enables the planner's use of PG-Strom",
							 NULL,
							 &guc_pgstrom_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off PG-Strom feature on all the instance */
	DefineCustomBoolVariable("pg_strom.enabled_global",
							 "Enables the planner's use of PG-Strom in global",
							 NULL,
							 &guc_pgstrom_enabled_global,
							 true,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE |
							 GUC_SUPERUSER_ONLY,
							 NULL,
							 pg_strom_enabled_global_assign,
							 pg_strom_enabled_global_show);
	/* turn on/off performance monitor on EXPLAIN ANALYZE */
	DefineCustomBoolVariable("pg_strom.perfmon",
							 "Enables the performance monitor of PG-Strom",
							 NULL,
							 &pgstrom_perfmon_enabled,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.debug_bulkload_enabled",
							 "Enables the bulk-loading mode of PG-Strom",
							 NULL,
							 &pgstrom_debug_bulkload_enabled,
							 true,
							 PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.debug_kernel_source",
							 "Enables to show kernel source on EXPLAIN",
							 NULL,
							 &pgstrom_debug_kernel_source,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.min_async_chunks",
							"least number of chunks to be run asynchronously",
							NULL,
							&pgstrom_min_async_chunks,
							2,
							2,
							INT_MAX,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.max_async_chunks",
							"max number of chunk to be run asynchronously",
							NULL,
							&pgstrom_max_async_chunks,
							32,
							pgstrom_min_async_chunks + 1,
							INT_MAX,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	if (pgstrom_max_async_chunks <= pgstrom_min_async_chunks)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("\"pg_strom.max_async_chunks\" must be larger than \"pg_strom.min_async_chunks\"")));

	DefineCustomRealVariable("gpu_setup_cost",
							 "Cost to setup GPU device to run",
							 NULL,
							 &pgstrom_gpu_setup_cost,
							 500 * DEFAULT_SEQ_PAGE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	DefineCustomRealVariable("gpu_operator_cost",
							 "Cost of processing each operators by GPU",
							 NULL,
							 &pgstrom_gpu_operator_cost,
							 DEFAULT_CPU_OPERATOR_COST / 100.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	DefineCustomRealVariable("gpu_tuple_cost",
							 "Cost of processing each tuple for GPU",
							 NULL,
							 &pgstrom_gpu_tuple_cost,
							 DEFAULT_CPU_TUPLE_COST / 32,
							 0,
							 DBL_MAX,
                             PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);

	DefineCustomRealVariable("pg_strom.row_population_max",
							 "hard limit of row population ratio",
							 NULL,
							 &pgstrom_row_population_max,
							 12.0,
							 0,
							 DBL_MAX,
                             PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);

	DefineCustomRealVariable("pg_strom.row_population_margin",
							 "safety margin if row will populate",
							 NULL,
							 &pgstrom_row_population_margin,
							 0.25,
							 0.0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}

/*
 * pgstrom_startup_global_guc
 *
 * allocation of shared memory for global guc
 */
static void
pgstrom_startup_global_guc(void)
{
	bool	found;
	char   *result = NULL;
	char   *varname = NULL;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	global_guc_values = ShmemInitStruct("pg_strom: global_guc",
										MAXALIGN(sizeof(*global_guc_values)),
										&found);
	Assert(!found);

	/* segment initialization */
	memset(global_guc_values, 0, MAXALIGN(sizeof(*global_guc_values)));
	SpinLockInit(&global_guc_values->lock);
	global_guc_values->pgstrom_enabled_global = guc_pgstrom_enabled_global;
}

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

	/* initialization of CUDA related stuff */
	pgstrom_init_cuda_control();
	pgstrom_init_cuda_program();
	/* initialization of data store support */
	pgstrom_init_datastore();

	/* registration of custom-scan providers */
	pgstrom_init_gpuscan();
	pgstrom_init_gpuhashjoin();
	pgstrom_init_gpupreagg();
	pgstrom_init_gpusort();

	/* miscellaneous initializations */
	pgstrom_init_misc_guc();
	pgstrom_init_codegen();
	pgstrom_init_grafter();

	/* allocation of shared memory */
	RequestAddinShmemSpace(MAXALIGN(sizeof(*global_guc_values)));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_global_guc;
}

/* ------------------------------------------------------------
 *
 * Routines copied from core PostgreSQL implementation
 *
 * ------------------------------------------------------------
 */
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
pgstrom_explain_custom_flags(CustomScanState *css, ExplainState *es)
{
	StringInfoData	str;

	if (!es->verbose)
		return;

	initStringInfo(&str);
	if ((css->flags & CUSTOMPATH_PREFERE_ROW_FORMAT) != 0)
		appendStringInfo(&str, "likely-heap-tuple");
	else
		appendStringInfo(&str, "likely-tuple-slot");

	if ((css->flags & CUSTOMPATH_SUPPORT_BULKLOAD) != 0)
		appendStringInfo(&str, ", bulkload-supported");

	ExplainPropertyText("Features", str.data, es);

	pfree(str.data);
}

void
pgstrom_explain_kernel_source(GpuTaskState *gts, ExplainState *es)
{
	const char	   *kern_source = gts->kern_source;
	int				extra_flags = gts->extra_flags;
	StringInfoData	str;

	if (!kern_source || !es->verbose || !pgstrom_debug_kernel_source)
		return;

	initStringInfo(&str);
	/*
	 * In case of EXPLAIN command context, we show the built-in logics
	 * like a usual #include preprocessor command.
	 * Practically, clCreateProgramWithSource() accepts multiple cstrings
	 * as if external files are included.
	 */
	appendStringInfo(&str, "#include \"cuda_common.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
		appendStringInfo(&str, "#include \"cuda_gpuscan.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
		appendStringInfo(&str, "#include \"cuda_hashjoin.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
		appendStringInfo(&str, "#include \"cuda_gpupreagg.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_GPUSORT)
		appendStringInfo(&str, "#include \"cuda_gpusort.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_MATHLIB)
		appendStringInfo(&str, "#include \"cuda_mathlib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_TIMELIB)
		appendStringInfo(&str, "#include \"cuda_timelib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_TEXTLIB)
		appendStringInfo(&str, "#include \"cuda_textlib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_NUMERIC)
		appendStringInfo(&str, "#include \"cuda_numeric.h\"\n");
	appendStringInfo(&str, "\n%s", kern_source);

	ExplainPropertyText("Kernel Source", str.data, es);

	pfree(str.data);
}

void
pgstrom_accum_perfmon(pgstrom_perfmon *accum, const pgstrom_perfmon *pfm)
{
	if (!accum->enabled)
		return;

	accum->num_samples++;
	accum->time_inner_load		+= pfm->time_inner_load;
	accum->time_outer_load		+= pfm->time_outer_load;
	accum->time_materialize		+= pfm->time_materialize;
	accum->time_launch_cuda		+= pfm->time_launch_cuda;
	accum->time_sync_tasks		+= pfm->time_sync_tasks;
	accum->num_dma_send			+= pfm->num_dma_send;
	accum->num_dma_recv			+= pfm->num_dma_recv;
	accum->bytes_dma_send		+= pfm->bytes_dma_send;
	accum->bytes_dma_recv		+= pfm->bytes_dma_recv;
	accum->time_dma_send		+= pfm->time_dma_send;
	accum->time_dma_recv		+= pfm->time_dma_recv;
	/* in case of gpuscan */
	accum->num_kern_qual		+= pfm->num_kern_qual;
	accum->time_kern_qual		+= pfm->time_kern_qual;
	/* in case of gpuhashjoin */
	accum->num_kern_join		+= pfm->num_kern_join;
	accum->num_kern_proj		+= pfm->num_kern_proj;
	accum->time_kern_join		+= pfm->time_kern_join;
	accum->time_kern_proj		+= pfm->time_kern_proj;
	/* in case of gpupreagg */
	accum->num_kern_prep		+= pfm->num_kern_prep;
	accum->num_kern_lagg		+= pfm->num_kern_lagg;
	accum->num_kern_gagg		+= pfm->num_kern_gagg;
	accum->num_kern_nogrp		+= pfm->num_kern_nogrp;
	accum->time_kern_prep		+= pfm->time_kern_prep;
	accum->time_kern_lagg		+= pfm->time_kern_lagg;
	accum->time_kern_gagg		+= pfm->time_kern_gagg;
	accum->time_kern_nogrp		+= pfm->time_kern_nogrp;
	/* in case of gpusort */
	accum->num_prep_sort		+= pfm->num_prep_sort;
	accum->num_gpu_sort			+= pfm->num_gpu_sort;
	accum->num_cpu_sort			+= pfm->num_cpu_sort;
	accum->time_prep_sort		+= pfm->time_prep_sort;
	accum->time_gpu_sort		+= pfm->time_gpu_sort;
	accum->time_cpu_sort		+= pfm->time_cpu_sort;
	accum->time_cpu_sort_real	+= pfm->time_cpu_sort_real;
	accum->time_cpu_sort_min	= Min(accum->time_cpu_sort_min,
									  pfm->time_cpu_sort);
	accum->time_cpu_sort_max	= Max(accum->time_cpu_sort_min,
									  pfm->time_cpu_sort);
	accum->time_bgw_sync		+= pfm->time_bgw_sync;
	/* for debug usage */
	accum->time_debug1			+= pfm->time_debug1;
	accum->time_debug2			+= pfm->time_debug2;
	accum->time_debug3			+= pfm->time_debug3;
	accum->time_debug4			+= pfm->time_debug4;
}

static char *
bytesz_unitary_format(double nbytes)
{
	if (nbytes > (double)(1UL << 43))
		return psprintf("%.2fTB", nbytes / (double)(1UL << 40));
	else if (nbytes > (double)(1UL << 33))
		return psprintf("%.2fGB", nbytes / (double)(1UL << 30));
	else if (nbytes > (double)(1UL << 23))
		return psprintf("%.2fMB", nbytes / (double)(1UL << 20));
	else if (nbytes > (double)(1UL << 13))
		return psprintf("%.2fKB", nbytes / (double)(1UL << 10));
	return psprintf("%uB", (unsigned int)nbytes);
}

static char *
milliseconds_unitary_format(double milliseconds)
{
	if (milliseconds > 300000.0)	/* more then 5min */
		return psprintf("%.2fmin", milliseconds / 60000.0);
	else if (milliseconds > 8000.0)	/* more than 8sec */
		return psprintf("%.2fsec", milliseconds / 1000.0);
	return psprintf("%.2fms", milliseconds);
}

void
pgstrom_explain_perfmon(pgstrom_perfmon *pfm, ExplainState *es)
{
	char		buf[256];

	if (!pfm->enabled || pfm->num_samples == 0)
		return;

	/* common performance statistics */
	ExplainPropertyInteger("number of tasks", pfm->num_samples, es);

	if (pfm->time_inner_load > 0.0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 milliseconds_unitary_format(pfm->time_inner_load));
		ExplainPropertyText("total time for inner load", buf, es);

		snprintf(buf, sizeof(buf), "%s",
				 milliseconds_unitary_format(pfm->time_outer_load));
		ExplainPropertyText("total time for outer load", buf, es);
	}
	else
	{
		snprintf(buf, sizeof(buf), "%s",
				 milliseconds_unitary_format(pfm->time_outer_load));
		ExplainPropertyText("total time to load", buf, es);
	}

	if (pfm->time_materialize > 0.0)
	{
		snprintf(buf, sizeof(buf), "%s",
                 milliseconds_unitary_format(pfm->time_materialize));
		ExplainPropertyText("total time to materialize", buf, es);
	}

	if (pfm->time_launch_cuda > 0.0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 milliseconds_unitary_format(pfm->time_launch_cuda));
		ExplainPropertyText("total time to CUDA commands", buf, es);
	}

	if (pfm->time_sync_tasks > 0.0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 milliseconds_unitary_format(pfm->time_sync_tasks));
		ExplainPropertyText("total time to synchronize", buf, es);
	}

	if (pfm->num_dma_send > 0)
	{
		double	band =
			((double)pfm->bytes_dma_send * 1000.0 / pfm->time_dma_send);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 bytesz_unitary_format(band),
				 bytesz_unitary_format((double)pfm->bytes_dma_send),
				 milliseconds_unitary_format(pfm->time_dma_send),
				 pfm->num_dma_send);
		ExplainPropertyText("DMA send", buf, es);
	}

	if (pfm->num_dma_recv > 0)
	{
		double	band =
			((double)pfm->bytes_dma_recv * 1000.0 / pfm->time_dma_recv);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 bytesz_unitary_format(band),
				 bytesz_unitary_format((double)pfm->bytes_dma_recv),
				 milliseconds_unitary_format(pfm->time_dma_recv),
				 pfm->num_dma_recv);
		ExplainPropertyText("DMA recv", buf, es);
	}

	/* in case of gpuscan */
	if (pfm->num_kern_qual > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format(pfm->time_kern_qual),
                 milliseconds_unitary_format(pfm->time_kern_qual /
											 (double)pfm->num_kern_qual),
                 pfm->num_kern_qual);
		ExplainPropertyText("Qual kernel exec", buf, es);
	}
	/* in case of gpuhashjoin */
	if (pfm->num_kern_join > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format(pfm->time_kern_join),
                 milliseconds_unitary_format(pfm->time_kern_join /
											 (double)pfm->num_kern_join),
                 pfm->num_kern_join);
		ExplainPropertyText("Hash-join main kernel", buf, es);
	}

	if (pfm->num_kern_proj > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format(pfm->time_kern_proj),
                 milliseconds_unitary_format(pfm->time_kern_proj /
											 (double)pfm->num_kern_proj),
                 pfm->num_kern_proj);
		ExplainPropertyText("Hash-join projection", buf, es);
	}
	/* in case of gpupreagg */
	if (pfm->num_kern_prep > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format(pfm->time_kern_prep),
				 milliseconds_unitary_format(pfm->time_kern_prep /
											 (double)pfm->num_kern_prep),
				 pfm->num_kern_prep);
        ExplainPropertyText("Aggregate preparation", buf, es);
	}

	if (pfm->num_kern_lagg > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format((double)pfm->time_kern_lagg),
				 milliseconds_unitary_format((double)pfm->time_kern_lagg /
											 (double)pfm->num_kern_lagg),
				 pfm->num_kern_lagg);
        ExplainPropertyText("Local reduction kernel", buf, es);
	}

	if (pfm->num_kern_gagg > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format((double)pfm->time_kern_gagg),
				 milliseconds_unitary_format((double)pfm->time_kern_gagg /
											 (double)pfm->num_kern_gagg),
				 pfm->num_kern_gagg);
        ExplainPropertyText("Global reduction kernel", buf, es);
	}

	if (pfm->num_kern_nogrp > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format((double)pfm->time_kern_nogrp),
				 milliseconds_unitary_format((double)pfm->time_kern_nogrp /
											 (double)pfm->num_kern_nogrp),
				 pfm->num_kern_nogrp);
        ExplainPropertyText("NoGroup reduction kernel", buf, es);
	}

	/* in case of gpusort */
	if (pfm->num_prep_sort > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format(pfm->time_prep_sort),
				 milliseconds_unitary_format(pfm->time_prep_sort /
											 (double)pfm->num_gpu_sort),
				 pfm->num_gpu_sort);
		ExplainPropertyText("GPU sort prep", buf, es);
	}

	if (pfm->num_gpu_sort > 0)
	{
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format(pfm->time_gpu_sort),
				 milliseconds_unitary_format(pfm->time_gpu_sort /
											 (double)pfm->num_gpu_sort),
				 pfm->num_gpu_sort);
		ExplainPropertyText("GPU sort exec", buf, es);
	}

	if (pfm->num_cpu_sort > 0)
	{
		cl_double	overhead;

		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 milliseconds_unitary_format(pfm->time_cpu_sort),
				 milliseconds_unitary_format(pfm->time_cpu_sort /
											 (double)pfm->num_cpu_sort),
				 pfm->num_cpu_sort);
		ExplainPropertyText("CPU sort exec", buf, es);

		overhead = pfm->time_cpu_sort_real - pfm->time_cpu_sort;
		snprintf(buf, sizeof(buf), "overhead: %s, sync: %s",
				 milliseconds_unitary_format(overhead),
				 milliseconds_unitary_format(pfm->time_bgw_sync));
		ExplainPropertyText("BGWorker", buf, es);
	}

	/* for debugging if any */
	if (pfm->time_debug1 > 0.0)
	{
		snprintf(buf, sizeof(buf), "debug1: %s",
				 milliseconds_unitary_format(pfm->time_debug1));
		ExplainPropertyText("debug-1", buf, es);
	}
	if (pfm->time_debug2 > 0.0)
	{
		snprintf(buf, sizeof(buf), "debug2: %s",
				 milliseconds_unitary_format(pfm->time_debug2));
		ExplainPropertyText("debug-2", buf, es);
	}
	if (pfm->time_debug3 > 0.0)
	{
		snprintf(buf, sizeof(buf), "debug3: %s",
				 milliseconds_unitary_format(pfm->time_debug3));
		ExplainPropertyText("debug-3", buf, es);
	}
	if (pfm->time_debug4 > 0.0)
	{
		snprintf(buf, sizeof(buf), "debug4: %s",
				 milliseconds_unitary_format(pfm->time_debug4));
		ExplainPropertyText("debug-4", buf, es);
	}
}

/*
 * XXX - copied from outfuncs.c
 *
 * _outToken
 *	  Convert an ordinary string (eg, an identifier) into a form that
 *	  will be decoded back to a plain token by read.c's functions.
 *
 *	  If a null or empty string is given, it is encoded as "<>".
 */
void
_outToken(StringInfo str, const char *s)
{
	if (s == NULL || *s == '\0')
	{
		appendStringInfoString(str, "<>");
		return;
	}

	/*
	 * Look for characters or patterns that are treated specially by read.c
	 * (either in pg_strtok() or in nodeRead()), and therefore need a
	 * protective backslash.
	 */
	/* These characters only need to be quoted at the start of the string */
	if (*s == '<' ||
		*s == '\"' ||
		isdigit((unsigned char) *s) ||
		((*s == '+' || *s == '-') &&
		 (isdigit((unsigned char) s[1]) || s[1] == '.')))
		appendStringInfoChar(str, '\\');
	while (*s)
	{
		/* These chars must be backslashed anywhere in the string */
		if (*s == ' ' || *s == '\n' || *s == '\t' ||
			*s == '(' || *s == ')' || *s == '{' || *s == '}' ||
			*s == '\\')
			appendStringInfoChar(str, '\\');
		appendStringInfoChar(str, *s++);
	}
}

/*
 * formBitmapset / deformBitmapset
 *
 * It translate a Bitmapset to/from copyObject available form.
 * Logic was fully copied from outfunc.c and readfunc.c.
 *
 * Note: the output format is "(b int int ...)", similar to an integer List.
 */
Value *
formBitmapset(const Bitmapset *bms)
{
	StringInfoData str;
	int		i;

	initStringInfo(&str);
	appendStringInfo(&str, "b:");
	for (i=0; i < bms->nwords; i++)
	{
		if (i > 0)
			appendStringInfoChar(&str, ',');
		appendStringInfo(&str, "%08x", bms->words[i]);
	}
	return makeString(str.data);
}

Bitmapset *
deformBitmapset(const Value *value)
{
	Bitmapset  *result;
	char	   *temp = strVal(value);
	char	   *token;
	char	   *delim;
	int			nwords;

	if (!temp)
		elog(ERROR, "incomplete Bitmapset structure");
	if (strncmp(temp, "b:", 2) != 0)
		elog(ERROR, "unrecognized Bitmapset format \"%s\"", temp);
	if (temp[3] == '\0')
		return NULL;	/* NULL bitmap */

	token = temp = pstrdup(temp + 2);
	nwords = strlen(temp) / (BITS_PER_BITMAPWORD / 4);
	result = palloc0(offsetof(Bitmapset, words[nwords]));

	do {
		bitmapword	x;

		delim = strchr(token, ',');
		if (delim)
			*delim = '\0';
		if (sscanf(token, "%8x", &x) != 1)
			elog(ERROR, "unrecognized Bitmapset token \"%s\"", token);
		Assert(result->nwords < nwords);
		result->words[result->nwords++] = x;
		token = delim + 1;
	} while (delim != NULL);

	pfree(temp);

	return result;
}
