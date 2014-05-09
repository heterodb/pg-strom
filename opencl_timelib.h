/*
 * opencl_timelib.h
 *
 * Collection of date/time functions for OpenCL devices
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_TIMELIB_H
#define OPENCL_TIMELIB_H
#ifdef OPENCL_DEVICE_CODE

	{ "date", 1, {TIMESTAMPOID}, "F:timestamp_date", NULL },
	{ "date", 1, {TIMESTAMPTZOID}, "F:timestamptz_date", NULL },
	{ "time", 1, {TIMESTAMPOID}, "F:timestamp_time", NULL },
	{ "time", 1, {TIMESTAMPTZOID}, "F:timestamptz_time", NULL },
	{ "timestamp", 1, {TIMESTAMPTZOID}, "F:timestamptz_timestamp", NULL },
	{ "timestamp", 1, {DATEOID}, "F:date_timestamp", NULL },
	{ "timestamptz", 1, {TIMESTAMPOID}, "F:timestamp_timestamptz", NULL },
	{ "timestamptz", 1, {DATEOID}, "F:date_timestamptz", NULL },
	/* timedata operators */
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, "F:datetime_pl", NULL },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, "F:timedata_pl", NULL },
	{ "date_pli", 2, {DATEOID, INT4OID}, "F:date_pli", NULL },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, "F:integer_pl_date", NULL },
	{ "date_mii", 2, {DATEOID, INT4OID}, "F:date_mii", NULL },
	/* timedate comparison */

	{ "timestamp_eq", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:timestamp_eq", NULL},
	{ "timestamp_ne", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:timestamp_ne", NULL},
	{ "timestamp_lt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:timestamp_lt", NULL},
	{ "timestamp_le", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:timestamp_le", NULL},
	{ "timestamp_gt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:timestamp_gt", NULL},
	{ "timestamp_ge", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:timestamp_ge", NULL},
#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_TIMELIB_H */
