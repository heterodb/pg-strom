/*
 * main.c
 *
 * Entrypoint of PG-Strom extension, and misc uncategolized functions.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include <float.h>
#include <limits.h>
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * miscellaneous GUC parameters
 */
bool	pgstrom_enabled;
bool	pgstrom_perfmon_enabled;
int		pgstrom_max_async_chunks;
int		pgstrom_min_async_chunks;
int		pgstrom_max_inline_varlena;

/* cost factors */
double	pgstrom_gpu_setup_cost;
double	pgstrom_gpu_operator_cost;
double	pgstrom_gpu_tuple_cost;

static void
pgstrom_init_misc_guc(void)
{
	/* GUC variables according to the device information */
	DefineCustomBoolVariable("pg_strom.enabled",
							 "Enables the planner's use of PG-Strom",
							 NULL,
							 &pgstrom_enabled,
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
	DefineCustomIntVariable("pg_strom.max_inline_varlena",
							"max length to inline varlena variables",
							NULL,
							&pgstrom_max_inline_varlena,
							64,
							0,
							128,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomRealVariable("gpu_setup_cost",
							 "Cost to setup GPU device to run",
							 NULL,
							 &pgstrom_gpu_setup_cost,
							 50 * DEFAULT_SEQ_PAGE_COST,
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

	/* registration of custom-plan providers */
	pgstrom_init_gpuscan();
	//pgstrom_init_gpusort();
	pgstrom_init_gpuhashjoin();

	/* initialization of tcache & registration of columnizer */
	pgstrom_init_tcache();

	/* miscellaneous initializations */
	pgstrom_init_misc_guc();
	pgstrom_init_codegen();
	pgstrom_init_grafter();
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
			return "success";
		case StromError_RowFiltered:
			return "row is filtered";
		case StromError_RowReCheck:
			return "row should be rechecked";
		case StromError_ServerNotReady:
			return "OpenCL server is not ready";
		case StromError_BadRequestMessage:
			return "request message is bad";
		case StromError_OpenCLInternal:
			return "OpenCL internal error";
		case StromError_OutOfSharedMemory:
			return "out of shared memory";
		case StromError_OutOfMemory:
			return "out of host memory";
		case StromError_DivisionByZero:
			return "division by zero";
		case StromError_DataStoreCorruption:
			return "row/column store is corrupted";
		case StromError_DataStoreNoSpace:
			return "row/column store has no space";
		case StromError_DataStoreOutOfRange:
			return "out of range in row/column store";
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

	if (!dprog_key || !es->verbose)
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
	if (extra_flags & DEVKERNEL_NEEDS_GPUSORT)
		appendStringInfo(&str, "#include \"opencl_gpusort.h\"\n");
	if (extra_flags & DEVKERNEL_NEEDS_HASHJOIN)
		appendStringInfo(&str, "#include \"opencl_hashjoin.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_TIMELIB)
		appendStringInfo(&str, "#include \"opencl_timelib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_TEXTLIB)
		appendStringInfo(&str, "#include \"opencl_textlib.h\"\n");
	if (extra_flags & DEVFUNC_NEEDS_NUMERICLIB)
		appendStringInfo(&str, "#include \"opencl_numericlib.h\"\n");
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
	pfm_sum->time_to_load	+= pfm_item->time_to_load;
	pfm_sum->time_in_sendq	+= pfm_item->time_in_sendq;
	pfm_sum->time_kern_build
		= Max(pfm_sum->time_kern_build,
			  pfm_item->time_kern_build);
	pfm_sum->bytes_dma_send += pfm_item->bytes_dma_send;
	pfm_sum->bytes_dma_recv += pfm_item->bytes_dma_recv;
	pfm_sum->num_dma_send	+= pfm_item->num_dma_send;
	pfm_sum->num_dma_recv	+= pfm_item->num_dma_recv;
	pfm_sum->time_dma_send	+= pfm_item->time_dma_send;
	pfm_sum->num_prep_exec	+= pfm_item->num_prep_exec;
	pfm_sum->num_kern_exec	+= pfm_item->num_kern_exec;
	pfm_sum->time_prep_exec	+= pfm_item->time_prep_exec;
	pfm_sum->time_kern_exec	+= pfm_item->time_kern_exec;
	pfm_sum->time_dma_recv	+= pfm_item->time_dma_recv;
	pfm_sum->time_in_recvq	+= pfm_item->time_in_recvq;
	pfm_sum->time_post_exec	+= pfm_item->time_post_exec;
	pfm_sum->time_move_slot	+= pfm_item->time_move_slot;
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
	char	buf[256];

	if (!pfm->enabled || pfm->num_samples == 0)
		return;

	/* common performance statistics */
	ExplainPropertyInteger("number of requests", pfm->num_samples, es);

	if (pfm->time_to_load_inner > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_to_load_inner));
		ExplainPropertyText("total time to inner load", buf, es);

		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_to_load));
		ExplainPropertyText("total time to outer load", buf, es);
	}
	else
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_to_load));
		ExplainPropertyText("total time to load", buf, es);
	}

	if (pfm->time_tcache_build > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_tcache_build));
		ExplainPropertyText("time to build tcache", buf, es);
	}

	if (pfm->time_kern_build > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_kern_build));
		ExplainPropertyText("max time to build kernel", buf, es);
	}

	if (pfm->num_samples > 0 && pfm->time_in_sendq > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_in_sendq /
										(double)pfm->num_samples));
		ExplainPropertyText("average time in send-mq", buf, es);
	}

	if (pfm->num_dma_send > 0)
	{
		snprintf(buf, sizeof(buf),
				 "total length %s, total time %s, avg time %s",
				 bytesz_unitary_format((double)pfm->bytes_dma_send),
				 usecond_unitary_format((double)pfm->time_dma_send),
				 usecond_unitary_format((double)pfm->time_dma_send /
										(double)pfm->num_dma_send));
		ExplainPropertyText("DMA send", buf, es);
	}

	if (pfm->num_prep_exec > 0)
	{
		snprintf(buf, sizeof(buf), "total %s, avg %s",
				 usecond_unitary_format((double)pfm->time_prep_exec),
				 usecond_unitary_format((double)pfm->time_prep_exec /
										(double)pfm->num_prep_exec));
        ExplainPropertyText("prep kernel exec time", buf, es);
	}

	if (pfm->num_kern_exec > 0)
	{
		snprintf(buf, sizeof(buf), "total %s, avg %s",
				 usecond_unitary_format((double)pfm->time_kern_exec),
				 usecond_unitary_format((double)pfm->time_kern_exec /
										(double)pfm->num_kern_exec));
		ExplainPropertyText("main kernel exec time", buf, es);
	}

	if (pfm->num_dma_recv > 0)
	{
		snprintf(buf, sizeof(buf),
				 "total length %s, total time %s, avg time %s",
				 bytesz_unitary_format((double)pfm->bytes_dma_recv),
				 usecond_unitary_format((double)pfm->time_dma_recv),
				 usecond_unitary_format((double)pfm->time_dma_recv /
										(double)pfm->num_dma_recv));
		ExplainPropertyText("DMA recv", buf, es);
	}

	if (pfm->num_dma_send > 0 || pfm->num_dma_recv > 0)
	{
		double	band;
		double	unitsz;
		size_t	ofs = 0;

		if (pfm->num_dma_send > 0)
		{
			band = (((double)pfm->bytes_dma_send * 1000000.0)
					/ (double)pfm->time_dma_send);
			unitsz = ((double)pfm->bytes_dma_send /
					  (double)pfm->num_dma_send);
			ofs += snprintf(buf + ofs, sizeof(buf) - ofs,
							"%ssend %s/sec unitsz %s",
							ofs > 0 ? ", " : "",
							bytesz_unitary_format(band),
							bytesz_unitary_format(unitsz));
		}
		if (pfm->num_dma_recv > 0)
		{
			band = (((double)pfm->bytes_dma_recv * 1000000.0)
					/ (double)pfm->time_dma_recv);
			unitsz = ((double)pfm->bytes_dma_recv /
					  (double)pfm->num_dma_recv);
			ofs += snprintf(buf + ofs, sizeof(buf) - ofs,
							"%srecv %s/sec unitsz %s",
							ofs > 0 ? ", " : "",
							bytesz_unitary_format(band),
							bytesz_unitary_format(unitsz));
		}
		ExplainPropertyText("DMA band", buf, es);
	}

	if (pfm->num_samples > 0 && pfm->time_in_recvq > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_in_recvq /
										(double)pfm->num_samples));
		ExplainPropertyText("average time in recv-mq", buf, es);
	}

	if (pfm->time_post_exec > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_post_exec));
		ExplainPropertyText("time for post device exec", buf, es);
	}

	if (pfm->time_move_slot > 0)
	{
		snprintf(buf, sizeof(buf), "%s",
				 usecond_unitary_format((double)pfm->time_move_slot));
		ExplainPropertyText("time to transform column to row", buf, es);
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
 * _outBitmapset -
 *	   converts a bitmap set of integers
 *
 * Note: the output format is "(b int int ...)", similar to an integer List.
 */
void
_outBitmapset(StringInfo str, const Bitmapset *bms)
{
	Bitmapset  *tmpset;
	int			x;

	appendStringInfoChar(str, '(');
	appendStringInfoChar(str, 'b');
	tmpset = bms_copy(bms);
	while ((x = bms_first_member(tmpset)) >= 0)
		appendStringInfo(str, " %d", x);
	bms_free(tmpset);
	appendStringInfoChar(str, ')');
}
