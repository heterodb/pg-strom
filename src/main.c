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
bool	pgstrom_show_device_kernel;
int		pgstrom_max_async_chunks;
int		pgstrom_min_async_chunks;

/* cost factors */
double	pgstrom_gpu_setup_cost;
double	pgstrom_gpu_operator_cost;
double	pgstrom_gpu_tuple_cost;

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
	SpinLockAcquire(&global_guc_values->lock);
	global_guc_values->pgstrom_enabled_global = newval;
	SpinLockRelease(&global_guc_values->lock);
}

/*
 * show callback of pg_strom.enabled_global
 */
static const char *
pg_strom_enabled_global_show(void)
{
	bool	rc;

	SpinLockAcquire(&global_guc_values->lock);
	rc = global_guc_values->pgstrom_enabled_global;
	SpinLockRelease(&global_guc_values->lock);

	return rc ? "on" : "off";
}

static void
pgstrom_init_misc_guc(void)
{
	/* GUC variables according to the device information */
	DefineCustomBoolVariable("pg_strom.enabled",
							 "Enables the planner's use of PG-Strom",
							 NULL,
							 &guc_pgstrom_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
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
	DefineCustomBoolVariable("pg_strom.show_device_kernel",
							 "Enables to show device kernel on EXPLAIN",
							 NULL,
							 &pgstrom_show_device_kernel,
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
							3,
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

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	global_guc_values = ShmemInitStruct("pg_strom: global_guc",
										MAXALIGN(sizeof(*global_guc_values)),
										&found);
	Assert(!found);

	/* segment initialization */
	memset(global_guc_values, 0, MAXALIGN(sizeof(*global_guc_values)));
	SpinLockInit(&global_guc_values->lock);

	/* add pg_strom.enabled_global parameter */
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

	/* load OpenCL runtime and initialize entrypoints */
	pgstrom_init_opencl_entry();

	/* initialization of device info on postmaster stage */
	pgstrom_init_opencl_devinfo();
	pgstrom_init_opencl_devprog();

	/* initialization of message queue on postmaster stage */
	pgstrom_init_mqueue();

	/* initialization of resource tracking subsystem */
	pgstrom_init_restrack();

	/* initialize shared memory segment and memory context stuff */
	pgstrom_init_shmem();

	/* registration of OpenCL background worker process */
	pgstrom_init_opencl_server();

	/* initialization of data-store */
	pgstrom_init_datastore();

	/* registration of custom-plan providers */
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

/*
 * pgstrom_strerror
 *
 * translation from StromError_* to human readable form
 */
const char *
pgstrom_strerror(cl_int errcode)
{
	static char		unknown_buf[256];

	if (errcode < 0)
		return opencl_strerror(errcode);

	switch (errcode)
	{
		case StromError_Success:
			return "Success";
		case StromError_RowFiltered:
			return "Row is filtered";
		case StromError_CpuReCheck:
			return "To be re-checked by CPU";
		case StromError_ServerNotReady:
			return "OpenCL server is not ready";
		case StromError_BadRequestMessage:
			return "Request message is bad";
		case StromError_OpenCLInternal:
			return "OpenCL internal error";
		case StromError_OutOfSharedMemory:
			return "out of shared memory";
		case StromError_OutOfMemory:
			return "out of host memory";
		case StromError_DataStoreCorruption:
			return "data store is corrupted";
		case StromError_DataStoreNoSpace:
			return "data store has no space";
		case StromError_DataStoreOutOfRange:
			return "out of range in data store";
		case StromError_SanityCheckViolation:
			return "sanity check violation";
		default:
			snprintf(unknown_buf, sizeof(unknown_buf),
					 "undefined strom error (code: %d)", errcode);
			break;
	}
	return unknown_buf;
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
	context = deparse_context_for_planstate((Node *) planstate,
											ancestors,
											es->rtable,
											es->rtable_names);

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
show_device_kernel(Datum dprog_key, ExplainState *es)
{
	StringInfoData	str;
	const char *kernel_source;
	int32		extra_flags;

	if (!dprog_key || !es->verbose || !pgstrom_show_device_kernel)
		return;

	kernel_source = pgstrom_get_devprog_kernel_source(dprog_key);
	extra_flags = pgstrom_get_devprog_extra_flags(dprog_key);

	initStringInfo(&str);
	/*
	 * In case of EXPLAIN command context, we show the built-in logics
	 * like a usual #include preprocessor command.
	 * Practically, clCreateProgramWithSource() accepts multiple cstrings
	 * as if external files are included.
	 */
	appendStringInfo(&str, "#include \"opencl_common.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_GPUSCAN)
		appendStringInfo(&str, "#include \"opencl_gpuscan.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
		appendStringInfo(&str, "#include \"opencl_hashjoin.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_GPUPREAGG)
		appendStringInfo(&str, "#include \"opencl_gpupreagg.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_MATHLIB)
		appendStringInfo(&str, "#include \"opencl_mathlib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_TIMELIB)
		appendStringInfo(&str, "#include \"opencl_timelib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_TEXTLIB)
		appendStringInfo(&str, "#include \"opencl_textlib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_NUMERIC)
		appendStringInfo(&str, "#include \"opencl_numeric.h\"\n");
	appendStringInfo(&str, "\n%s", kernel_source);

	ExplainPropertyText("Kernel Source", str.data, es);

	pfree(str.data);
}

void
pgstrom_perfmon_add(pgstrom_perfmon *pfm_sum, pgstrom_perfmon *pfm_item)
{
	if (!pfm_sum->enabled)
		return;

	pfm_sum->num_samples++;
	pfm_sum->time_inner_load	+= pfm_item->time_inner_load;
	pfm_sum->time_outer_load	+= pfm_item->time_outer_load;
	pfm_sum->time_materialize	+= pfm_item->time_materialize;
	pfm_sum->time_in_sendq		+= pfm_item->time_in_sendq;
	pfm_sum->time_in_recvq		+= pfm_item->time_in_recvq;
	pfm_sum->time_kern_build = Max(pfm_sum->time_kern_build,
								   pfm_item->time_kern_build);
	pfm_sum->num_dma_send		+= pfm_item->num_dma_send;
	pfm_sum->num_dma_recv		+= pfm_item->num_dma_recv;
	pfm_sum->bytes_dma_send		+= pfm_item->bytes_dma_send;
	pfm_sum->bytes_dma_recv		+= pfm_item->bytes_dma_recv;
	pfm_sum->time_dma_send		+= pfm_item->time_dma_send;
	pfm_sum->time_dma_recv		+= pfm_item->time_dma_recv;
	pfm_sum->num_kern_exec		+= pfm_item->num_kern_exec;
	pfm_sum->time_kern_exec		+= pfm_item->time_kern_exec;
	/* for gpuhashjoin */
	pfm_sum->num_kern_proj		+= pfm_item->num_kern_proj;
	pfm_sum->time_kern_proj		+= pfm_item->time_kern_proj;
	/* for gpupreagg */
	pfm_sum->num_kern_prep		+= pfm_item->num_kern_prep;
	pfm_sum->num_kern_sort		+= pfm_item->num_kern_sort;
	pfm_sum->time_kern_prep		+= pfm_item->time_kern_prep;
	pfm_sum->time_kern_sort		+= pfm_item->time_kern_sort;
	/* for debugging */
	pfm_sum->time_debug1		+= pfm_item->time_debug1;
	pfm_sum->time_debug2		+= pfm_item->time_debug2;
	pfm_sum->time_debug3		+= pfm_item->time_debug3;
	pfm_sum->time_debug4		+= pfm_item->time_debug4;
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
usecond_unitary_format(double usecond)
{
	if (usecond > 300.0 * 1000.0 * 1000.0)
		return psprintf("%.2fmin", usecond / (60.0 * 1000.0 * 1000.0));
	else if (usecond > 8000.0 * 1000.0)
		return psprintf("%.2fsec", usecond / (1000.0 * 1000.0));
	else if (usecond > 8000.0)
		return psprintf("%.2fms", usecond / 1000.0);
	return psprintf("%uus", (unsigned int)usecond);
}

void
pgstrom_perfmon_explain(pgstrom_perfmon *pfm, ExplainState *es)
{
	bool		multi_kernel = false;
	char		buf[256];

	if (!pfm->enabled || pfm->num_samples == 0)
		return;

	/* common performance statistics */
	ExplainPropertyInteger("number of requests", pfm->num_samples, es);

	if (pfm->time_inner_load > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_inner_load));
		ExplainPropertyText("total time for inner load", buf, es);

		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_outer_load));
		ExplainPropertyText("total time for outer load", buf, es);
	}
	else
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_outer_load));
		ExplainPropertyText("total time to load", buf, es);
	}

	if (pfm->time_materialize > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
                 usecond_unitary_format((double)pfm->time_materialize));
		ExplainPropertyText("total time to materialize", buf, es);
	}

	if (pfm->num_samples > 0 && (pfm->time_in_sendq > 0 ||
								 pfm->time_in_recvq > 0))
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_in_sendq /
										(double)pfm->num_samples));
		ExplainPropertyText("average time in send-mq", buf, es);

		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_in_recvq /
										(double)pfm->num_samples));
		ExplainPropertyText("average time in recv-mq", buf, es);
	}

	if (pfm->time_kern_build > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_kern_build));
		ExplainPropertyText("max time to build kernel", buf, es);
	}

	if (pfm->num_dma_send > 0)
	{
		double	band = (((double)pfm->bytes_dma_send * 1000000.0)
						/ (double)pfm->time_dma_send);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 bytesz_unitary_format(band),
				 bytesz_unitary_format((double)pfm->bytes_dma_send),
				 usecond_unitary_format((double)pfm->time_dma_send),
				 pfm->num_dma_send);
		ExplainPropertyText("DMA send", buf, es);
	}

	if (pfm->num_dma_recv > 0)
	{
		double	band = (((double)pfm->bytes_dma_recv * 1000000.0)
                        / (double)pfm->time_dma_recv);
		snprintf(buf, sizeof(buf),
				 "%s/sec, len: %s, time: %s, count: %u",
				 bytesz_unitary_format(band),
                 bytesz_unitary_format((double)pfm->bytes_dma_recv),
                 usecond_unitary_format((double)pfm->time_dma_recv),
				 pfm->num_dma_recv);
		ExplainPropertyText("DMA recv", buf, es);
	}

	/* only gpupreagg */
	if (pfm->num_kern_prep > 0)
	{
		multi_kernel = true;
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 usecond_unitary_format((double)pfm->time_kern_prep),
				 usecond_unitary_format((double)pfm->time_kern_prep /
										(double)pfm->num_kern_prep),
				 pfm->num_kern_prep);
        ExplainPropertyText("prep kernel exec", buf, es);
	}

	/* only gpupreagg */
	if (pfm->num_kern_sort > 0)
	{
		multi_kernel = true;
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 usecond_unitary_format((double)pfm->time_kern_sort),
				 usecond_unitary_format((double)pfm->time_kern_sort /
										(double)pfm->num_kern_sort),
				 pfm->num_kern_sort);
        ExplainPropertyText("sort kernel exec", buf, es);
	}

	/* only gpuhashjoin */
	if (pfm->num_kern_proj > 0)
	{
		multi_kernel = true;
		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 usecond_unitary_format((double)pfm->time_kern_proj),
				 usecond_unitary_format((double)pfm->time_kern_proj /
										(double)pfm->num_kern_proj),
				 pfm->num_kern_exec);
		ExplainPropertyText("proj kernel exec", buf, es);
	}

	if (pfm->num_kern_exec > 0)
	{
		const char	   *label =
			(multi_kernel ? "main kernel exec" : "kernel exec");

		snprintf(buf, sizeof(buf), "total: %s, avg: %s, count: %u",
				 usecond_unitary_format((double)pfm->time_kern_exec),
				 usecond_unitary_format((double)pfm->time_kern_exec /
										(double)pfm->num_kern_exec),
				 pfm->num_kern_exec);
		ExplainPropertyText(label, buf, es);
	}
	/* for debugging if any */
	if (pfm->time_debug1 > 0)
	{
		snprintf(buf, sizeof(buf), "debug1: %s",
				 usecond_unitary_format((double)pfm->time_debug1));
		ExplainPropertyText("debug-1", buf, es);
	}
	if (pfm->time_debug2 > 0)
	{
		snprintf(buf, sizeof(buf), "debug2: %s",
				 usecond_unitary_format((double)pfm->time_debug2));
		ExplainPropertyText("debug-2", buf, es);
	}
	if (pfm->time_debug3 > 0)
	{
		snprintf(buf, sizeof(buf), "debug3: %s",
				 usecond_unitary_format((double)pfm->time_debug3));
		ExplainPropertyText("debug-3", buf, es);
	}
	if (pfm->time_debug4 > 0)
	{
		snprintf(buf, sizeof(buf), "debug4: %s",
				 usecond_unitary_format((double)pfm->time_debug4));
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
