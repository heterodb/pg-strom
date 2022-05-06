/*
 * xpu_timelib.c
 *
 * Collection of the primitive Date/Time type support on xPU(GPU/DPU/SPU)
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"

/*
 * Misc definitions copied from PostgreSQL
 */
#ifndef DATATYPE_TIMESTAMP_H
#define DAYS_PER_YEAR		365.25	/* assumes leap year every four years */
#define MONTHS_PER_YEAR		12
#define DAYS_PER_MONTH		30		/* assumes exactly 30 days per month */
#define HOURS_PER_DAY		24		/* assume no daylight savings time changes */

#define SECS_PER_YEAR		(36525 * 864)	/* avoid floating-point computation */
#define SECS_PER_DAY		86400
#define SECS_PER_HOUR		3600
#define SECS_PER_MINUTE		60
#define MINS_PER_HOUR		60

#define USECS_PER_DAY		86400000000L
#define USECS_PER_HOUR       3600000000L
#define USECS_PER_MINUTE       60000000L
#define USECS_PER_SEC           1000000L

#define DT_NOBEGIN			(-0x7fffffffffffffffL - 1)
#define DT_NOEND			(0x7fffffffffffffffL)
#endif	/* DATATYPE_TIMESTAMP_H */

/*
 * Common ref/store functions for:
 *  date/time/timetz/timestamp/timestamptz/interval
 */

PGSTROM_SIMPLE_BASETYPE_TEMPLATE(date, DateADT);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(time, TimeADT);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(timetz, TimeTzADT);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(timestamp, Timestamp);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(timestamptz, TimestampTz);

STATIC_FUNCTION(bool)
sql_interval_datum_ref(kern_context *kcxt,
					   sql_datum_t *__result,
					   void *addr)
{
	sql_interval_t *result = (sql_interval_t *)__result;

	memset(result, 0, sizeof(sql_interval_t));
	if (!addr)
		result->isnull = true;
	else
		memcpy(&result->value, addr, sizeof(Interval));
	result->ops = &sql_interval_ops;
	return true;
}

STATIC_FUNCTION(bool)
arrow_interval_datum_ref(kern_context *kcxt,
						 sql_datum_t *__result,
						 kern_data_store *kds,
						 kern_colmeta *cmeta,
						 uint32_t rowidx)
{
	sql_interval_t *result = (sql_interval_t *)__result;
	uint32_t   *ival;

	memset(result, 0, sizeof(sql_interval_t));
	switch (cmeta->attopts.interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			ival = (uint32_t *)
				KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, rowidx,
										   sizeof(uint32_t));
			if (!ival)
				result->isnull = true;
			else
				result->value.month = *ival;
			break;
		case ArrowIntervalUnit__Day_Time:
			ival = (uint32_t *)
				KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, rowidx,
										   2 * sizeof(uint32_t));
			if (!ival)
				result->isnull = true;
			else
			{
				result->value.day = ival[0];
				result->value.time = ival[1];
			}
			break;
		default:
			STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
						  "unknown unit-size of Arrow::Interval");
			return false;
	}
	result->ops = &sql_interval_ops;
	return true;
}

STATIC_FUNCTION(int)
sql_interval_datum_store(kern_context *kcxt,
						 char *buffer,
						 sql_datum_t *__arg)
{
	sql_interval_t *arg = (sql_interval_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, &arg->value, sizeof(Interval));
	return sizeof(Interval);
}

STATIC_FUNCTION(bool)
sql_interval_datum_hash(kern_context *kcxt,
						uint32_t *p_hash,
						sql_datum_t *__arg)
{
	sql_interval_t *arg = (sql_interval_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(Interval));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(interval);
