/*
 * multirels.c
 *
 * Inner relations loader for GpuJoin
 * ----
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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
#include "pg_strom.h"



/*
 * MultiRelsInfo - state object of CustomScan(MultiRels)
 */
typedef struct
{





} MultiRelsInfo;

/*
 * MultiRelsState - state object of the executor
 */
typedef struct
{
	CustomScanState	css;
	GpuContext	   *gcontext;
	int				depth;

} MultiRelsState;



/* static variables */
static CustomScanMethods	multirels_plan_methods;
static PGStromExecMethods	multirels_exec_methods;





/*
 * pgstrom_init_multirels
 *
 * entrypoint of this custom-scan provider
 */
void
pgstrom_init_multirels(void)
{
	/* setup plan methods */
	multirels_plan_methods.CustomName			= "MultiRels";
	multirels_plan_methods.CreateCustomScanState
		= multirels_create_scan_state;
	multirels_plan_methods.TextOutCustomScan	= NULL;

	/* setup exec methods */
	multirels_exec_methods.c.CustomName			= "MultiRels";
	multirels_exec_methods.c.BeginCustomScan	= multirels_begin;
	multirels_exec_methods.c.ExecCustomScan		= multirels_exec;
	multirels_exec_methods.c.EndCustomScan		= multirels_end;
	multirels_exec_methods.c.ReScanCustomScan	= multirels_rescan;
	multirels_exec_methods.c.MarkPosCustomScan	= NULL;
	multirels_exec_methods.c.RestrPosCustomScan	= NULL;
	multirels_exec_methods.c.ExplainCustomScan	= multirels_explain;
	multirels_exec_methods.ExecCustomBulk		= multirels_exec_bulk;
}
