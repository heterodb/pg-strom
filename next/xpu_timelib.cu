/*
 * xpu_timelib.cu
 *
 * Collection of the primitive Date/Time type support for both of GPU and DPU
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
#ifndef DATE_H
#define MAX_TIME_PRECISION	6

#define DATEVAL_NOBEGIN		((DateADT)(-0x7fffffff - 1))
#define DATEVAL_NOEND       ((DateADT)  0x7fffffff)

#define DATE_NOBEGIN(j)		((j) = DATEVAL_NOBEGIN)
#define DATE_IS_NOBEGIN(j)	((j) == DATEVAL_NOBEGIN)
#define DATE_NOEND(j)		((j) = DATEVAL_NOEND)
#define DATE_IS_NOEND(j)	((j) == DATEVAL_NOEND)
#define DATE_NOT_FINITE(j)	(DATE_IS_NOBEGIN(j) || DATE_IS_NOEND(j))
#endif	/* DATE_H */

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

/* -Inf and +Inf */
#define DT_NOBEGIN			LLONG_MIN
#define DT_NOEND			LLONG_MAX

#define TIMESTAMP_NOBEGIN(j)		do {(j) = DT_NOBEGIN;} while (0)
#define TIMESTAMP_IS_NOBEGIN(j)		((j) == DT_NOBEGIN)
#define TIMESTAMP_NOEND(j)			do {(j) = DT_NOEND;} while (0)
#define TIMESTAMP_IS_NOEND(j)		((j) == DT_NOEND)
#define TIMESTAMP_NOT_FINITE(j)		(TIMESTAMP_IS_NOBEGIN(j) ||	\
									 TIMESTAMP_IS_NOEND(j))

/* Julian date support */
#define JULIAN_MINYEAR		(-4713)
#define JULIAN_MINMONTH		(11)
#define JULIAN_MINDAY		(24)
#define JULIAN_MAXYEAR		(5874898)
#define JULIAN_MAXMONTH		(6)
#define JULIAN_MAXDAY		(3)

#define IS_VALID_JULIAN(y,m,d)								  \
	(((y) > JULIAN_MINYEAR ||								  \
	  ((y) == JULIAN_MINYEAR && ((m) >= JULIAN_MINMONTH))) && \
	 ((y) < JULIAN_MAXYEAR ||								  \
	  ((y) == JULIAN_MAXYEAR && ((m) < JULIAN_MAXMONTH))))

/* Julian-date equivalents of Day 0 in Unix and Postgres reckoning */
#define UNIX_EPOCH_JDATE		2440588 /* == date2j(1970, 1, 1) */
#define POSTGRES_EPOCH_JDATE	2451545 /* == date2j(2000, 1, 1) */

/* First allowed date, and first disallowed date, in Julian-date form */
#define DATETIME_MIN_JULIAN		(0)
#define DATE_END_JULIAN			(2147483494)	/* == date2j(JULIAN_MAXYEAR, 1, 1) */
#define TIMESTAMP_END_JULIAN	(109203528)		/* == date2j(294277, 1, 1) */

/* Timestamp limits */
#define MIN_TIMESTAMP		(-211813488000000000LL)
/* == (DATETIME_MIN_JULIAN - POSTGRES_EPOCH_JDATE) * USECS_PER_DAY */
#define END_TIMESTAMP		(9223371331200000000LL)
/* == (TIMESTAMP_END_JULIAN - POSTGRES_EPOCH_JDATE) * USECS_PER_DAY */

/* Range-check a date (given in Postgres, not Julian, numbering) */
#define IS_VALID_DATE(d) \
	((DATETIME_MIN_JULIAN - POSTGRES_EPOCH_JDATE) <= (d) && \
	 (d) < (DATE_END_JULIAN - POSTGRES_EPOCH_JDATE))
#define IS_VALID_TIMESTAMP(t)  (MIN_TIMESTAMP <= (t) && (t) < END_TIMESTAMP)

#endif	/* DATATYPE_TIMESTAMP_H */

/* from timezone/private.h */
#ifndef PRIVATE_H
#define YEARSPERREPEAT	400		/* years before a Gregorian repeat */
#define AVGSECSPERYEAR	31556952L
#define SECSPERREPEAT	((pg_time_t)YEARSPERREPEAT * (pg_time_t)AVGSECSPERYEAR)

#define SECSPERMIN		60
#define MINSPERHOUR		60
#define HOURSPERDAY		24
#define DAYSPERWEEK		7
#define DAYSPERNYEAR	365
#define DAYSPERLYEAR	366
#define SECSPERHOUR		(SECSPERMIN * MINSPERHOUR)
#define SECSPERDAY		((long) SECSPERHOUR * HOURSPERDAY)
#define MONSPERYEAR		12

#define TM_SUNDAY		0
#define TM_MONDAY		1
#define TM_TUESDAY		2
#define TM_WEDNESDAY	3
#define TM_THURSDAY		4
#define TM_FRIDAY		5
#define TM_SATURDAY		6

#define TM_JANUARY		0
#define TM_FEBRUARY		1
#define TM_MARCH		2
#define TM_APRIL		3
#define TM_MAY			4
#define TM_JUNE			5
#define TM_JULY			6
#define TM_AUGUST		7
#define TM_SEPTEMBER	8
#define TM_OCTOBER		9
#define TM_NOVEMBER		10
#define TM_DECEMBER		11

#define TM_YEAR_BASE    1900

#define EPOCH_YEAR		1970
#define EPOCH_WDAY		TM_THURSDAY

#define isleap(y) (((y) % 4) == 0 && (((y) % 100) != 0 || ((y) % 400) == 0))
#endif	/* PRIVATE_H */

/*
 * xpu_date_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_date_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const void *addr)
{
	xpu_date_t *result = (xpu_date_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value = *((const DateADT *)addr);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_date_arrow_move(kern_context *kcxt,
					char *buffer,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	if (cmeta->attopts.common.tag != ArrowType__Date)
	{
		STROM_ELOG(kcxt, "Value is not convertible to date");
		return -1;
	}
	if (!addr)
		return 0;
	if (buffer)
	{
		DateADT	   *value = (DateADT *)buffer;

		switch (cmeta->attopts.date.unit)
		{
			case ArrowDateUnit__Day:
				assert(len == sizeof(uint32_t));
				*value = *((uint32_t *)addr)
					- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
				break;
			case ArrowDateUnit__MilliSecond:
				assert(len == sizeof(uint64_t));
				*value = *((uint64_t *)addr) / 1000
					- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
				break;
			default:
				STROM_ELOG(kcxt, "unknown unit size of Arrow::Date");
				return -1;
		}
	}
	return sizeof(DateADT);
}

STATIC_FUNCTION(bool)
xpu_date_arrow_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	xpu_date_t *result = (xpu_date_t *)__result;
	int		sz;

	sz = xpu_date_arrow_move(kcxt, (char *)&result->value,
							 cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_date_datum_store(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_date_t *arg = (const xpu_date_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		*((DateADT *)buffer) = arg->value;
	return sizeof(DateADT);
}

STATIC_FUNCTION(bool)
xpu_date_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					const xpu_datum_t *__arg)
{
	const xpu_date_t *arg = (const xpu_date_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(DateADT));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(date, true, 4, sizeof(DateADT));

/*
 * xpu_time_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_time_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const void *addr)
{
	xpu_time_t *result = (xpu_time_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value = *((const TimeADT *)addr);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_time_arrow_move(kern_context *kcxt,
					char *buffer,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	if (cmeta->attopts.common.tag != ArrowType__Time)
	{
		STROM_ELOG(kcxt, "value is not convertible to time");
		return -1;
	}
	if (!addr)
		return 0;
	if (buffer)
	{
		TimeADT	   *value = (TimeADT *)buffer;

		switch (cmeta->attopts.time.unit)
		{
			case ArrowTimeUnit__Second:
				assert(len == sizeof(uint32_t));
				*value = *((uint32_t *)addr) * 1000000L;
				break;
			case ArrowTimeUnit__MilliSecond:
				assert(len == sizeof(uint32_t));
				*value = *((uint32_t *)addr) * 1000L;
				break;
			case ArrowTimeUnit__MicroSecond:
				assert(len == sizeof(uint64_t));
				*value = *((uint32_t *)addr);
				break;
			case ArrowTimeUnit__NanoSecond:
				assert(len == sizeof(uint64_t));
				*value = *((uint32_t *)addr) / 1000L;
				break;
			default:
				STROM_ELOG(kcxt, "unknown unit size of Arrow::Time");
				return -1;
		}
	}
	return sizeof(TimeADT);
}

STATIC_FUNCTION(bool)
xpu_time_arrow_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	xpu_time_t *result = (xpu_time_t *)__result;
	int		sz;

	sz = xpu_time_arrow_move(kcxt, (char *)&result->value,
							 cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_time_datum_store(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_time_t *arg = (const xpu_time_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		*((TimeADT *)buffer) = arg->value;
	return sizeof(TimeADT);
}

STATIC_FUNCTION(bool)
xpu_time_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					const xpu_datum_t *__arg)
{
	const xpu_time_t *arg = (const xpu_time_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(TimeADT));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(time, true, 8, sizeof(TimeADT));

/*
 * xpu_timetz_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_timetz_datum_ref(kern_context *kcxt,
					 xpu_datum_t *__result,
					 const void *addr)
{
	xpu_timetz_t *result = (xpu_timetz_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		memcpy(&result->value, addr, SizeOfTimeTzADT);
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_timetz_arrow_ref(kern_context *kcxt,
					 xpu_datum_t *result,
					 const kern_colmeta *cmeta,
					 const void *addr, int len)
{
	STROM_ELOG(kcxt, "PostgreSQL 'timetz' has no convertiblae type of Arrow");
	return false;
}

STATIC_FUNCTION(int)
xpu_timetz_arrow_move(kern_context *kcxt,
					  char *buffer,
					  const kern_colmeta *cmeta,
					  const void *addr, int len)
{
	STROM_ELOG(kcxt, "PostgreSQL 'timetz' has no convertiblae type of Arrow");
	return -1;
}

STATIC_FUNCTION(int)
xpu_timetz_datum_store(kern_context *kcxt,
					   char *buffer,
					   const xpu_datum_t *__arg)
{
	const xpu_timetz_t *arg = (const xpu_timetz_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, &arg->value, SizeOfTimeTzADT);
	return SizeOfTimeTzADT;
}

STATIC_FUNCTION(bool)
xpu_timetz_datum_hash(kern_context *kcxt,
					  uint32_t *p_hash,
					  const xpu_datum_t *__arg)
{
	const xpu_timetz_t *arg = (const xpu_timetz_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, SizeOfTimeTzADT);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(timetz, false, 8, SizeOfTimeTzADT);

/*
 * xpu_timestamp_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_timestamp_datum_ref(kern_context *kcxt,
						xpu_datum_t *__result,
						const void *addr)
{
	xpu_timestamp_t *result = (xpu_timestamp_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value = *((const Timestamp *)addr);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_timestamp_arrow_move(kern_context *kcxt,
						 char *buffer,
						 const kern_colmeta *cmeta,
						 const void *addr, int len)
{
	if (cmeta->attopts.common.tag != ArrowType__Timestamp)
	{
		STROM_ELOG(kcxt, "Value is not convertible to timestamp");
		return -1;
	}
	if (!addr)
		return 0;
	if (buffer)
	{
		uint64_t	tval;

		assert(len == sizeof(uint64_t));
		switch (cmeta->attopts.timestamp.unit)
		{
			case ArrowTimeUnit__Second:
				tval = *((uint64_t *)addr) * 1000000L;
				break;
			case ArrowTimeUnit__MilliSecond:
				tval = *((uint64_t *)addr) * 1000L;
				break;
			case ArrowTimeUnit__MicroSecond:
				tval = *((uint64_t *)addr);
				break;
			case ArrowTimeUnit__NanoSecond:
				tval = *((uint64_t *)addr) / 1000L;
				break;
			default:
				STROM_ELOG(kcxt, "unknown unit size of Arrow::Timestamp");
				return -1;
		}
		*((Timestamp *)buffer) = tval - (POSTGRES_EPOCH_JDATE -
										 UNIX_EPOCH_JDATE) * USECS_PER_DAY;
	}
	return sizeof(Timestamp);
}

STATIC_FUNCTION(bool)
xpu_timestamp_arrow_ref(kern_context *kcxt,
						xpu_datum_t *__result,
						const kern_colmeta *cmeta,
						const void *addr, int len)
{
	xpu_timestamp_t *result = (xpu_timestamp_t *)__result;
	int		sz;

	sz = xpu_timestamp_arrow_move(kcxt, (char *)&result->value,
								  cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_timestamp_datum_store(kern_context *kcxt,
						  char *buffer,
						  const xpu_datum_t *__arg)
{
	const xpu_timestamp_t *arg = (const xpu_timestamp_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		*((Timestamp *)buffer) = arg->value;
	return sizeof(Timestamp);
}

STATIC_FUNCTION(bool)
xpu_timestamp_datum_hash(kern_context *kcxt,
						 uint32_t *p_hash,
						 const xpu_datum_t *__arg)
{
	const xpu_timestamp_t *arg = (const xpu_timestamp_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(Timestamp));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(timestamp, true, 8, sizeof(Timestamp));

/*
 * xpu_timestamptz_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_timestamptz_datum_ref(kern_context *kcxt,
						  xpu_datum_t *__result,
						  const void *addr)
{
	xpu_timestamptz_t *result = (xpu_timestamptz_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		memcpy(&result->value, addr, sizeof(TimestampTz));
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_timestamptz_arrow_move(kern_context *kcxt,
						   char *buffer,
						   const kern_colmeta *cmeta,
						   const void *addr, int len)
{
	if (cmeta->attopts.common.tag != ArrowType__Timestamp)
	{
		STROM_ELOG(kcxt, "Value is not convertible to timestamp");
		return -1;
	}
	if (!addr)
		return 0;
	if (buffer)
	{
		uint64_t	tval;

		assert(len == sizeof(uint64_t));
		switch (cmeta->attopts.timestamp.unit)
		{
			case ArrowTimeUnit__Second:
				tval = *((uint64_t *)addr) * 1000000L;
				break;
			case ArrowTimeUnit__MilliSecond:
				tval = *((uint64_t *)addr) * 1000L;
				break;
			case ArrowTimeUnit__MicroSecond:
				tval = *((uint64_t *)addr);
				break;
			case ArrowTimeUnit__NanoSecond:
				tval = *((uint64_t *)addr) / 1000L;
				break;
			default:
				STROM_ELOG(kcxt, "unknown unit size of Arrow::Timestamp");
				return -1;
		}
		*((TimestampTz *)buffer) = tval - (POSTGRES_EPOCH_JDATE -
										   UNIX_EPOCH_JDATE) * USECS_PER_DAY;
	}
	return sizeof(TimestampTz);
}

STATIC_FUNCTION(bool)
xpu_timestamptz_arrow_ref(kern_context *kcxt,
						  xpu_datum_t *__result,
						  const kern_colmeta *cmeta,
						  const void *addr, int len)
{
    xpu_timestamptz_t *result = (xpu_timestamptz_t *)__result;
	int		sz;

	sz = xpu_timestamptz_arrow_move(kcxt, (char *)&result->value,
									cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_timestamptz_datum_store(kern_context *kcxt,
							char *buffer,
							const xpu_datum_t *__arg)
{
	const xpu_timestamptz_t *arg = (const xpu_timestamptz_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, &arg->value, sizeof(TimestampTz));
	return sizeof(TimestampTz);
}

STATIC_FUNCTION(bool)
xpu_timestamptz_datum_hash(kern_context *kcxt,
						   uint32_t *p_hash,
						   const xpu_datum_t *__arg)
{
	const xpu_timestamptz_t *arg = (const xpu_timestamptz_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(TimestampTz));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(timestamptz, true, 8, sizeof(TimestampTz));

/*
 * xpu_interval_t device type handlers
 */

/* definition copied from datetime.c */
STATIC_DATA const int day_tab[2][13] =
{
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0}
};

/* definition copied from pgtime.h */
STATIC_DATA const int mon_lengths[2][MONSPERYEAR] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

STATIC_DATA const int year_lengths[2] = {
    DAYSPERNYEAR, DAYSPERLYEAR
};

STATIC_FUNCTION(bool)
xpu_interval_datum_ref(kern_context *kcxt,
					   xpu_datum_t *__result,
					   const void *addr)
{
	xpu_interval_t *result = (xpu_interval_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		memcpy(&result->value, addr, sizeof(Interval));
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_interval_arrow_move(kern_context *kcxt,
						char *buffer,
						const kern_colmeta *cmeta,
						const void *addr, int len)
{
	if (cmeta->attopts.common.tag != ArrowType__Interval)
	{
		STROM_ELOG(kcxt, "value is not convertible to interval");
		return -1;
	}
	if (!addr)
		return 0;
	if (buffer)
	{
		uint32_t   *ival = (uint32_t *)addr;
		Interval   *value = (Interval *)buffer;

		switch (cmeta->attopts.interval.unit)
		{
			case ArrowIntervalUnit__Year_Month:
				value->month = *ival;
				value->day   = 0;
				value->time  = 0;
				break;
			case ArrowIntervalUnit__Day_Time:
				value->month = 0;
				value->day   = ival[0];
				value->time  = ival[1];
				break;
			default:
				STROM_ELOG(kcxt, "unknown unit-size of Arrow::Interval");
				return -1;
		}
	}
	return sizeof(Interval);
}

STATIC_FUNCTION(bool)
xpu_interval_arrow_ref(kern_context *kcxt,
					   xpu_datum_t *__result,
					   const kern_colmeta *cmeta,
					   const void *addr, int len)
{
	xpu_interval_t *result = (xpu_interval_t *)__result;
	int		sz;

	sz = xpu_interval_arrow_move(kcxt, (char *)&result->value,
								 cmeta, addr, len);
	if (sz < 0)
		return false;
	result->isnull = (sz == 0);
	return true;
}

STATIC_FUNCTION(int)
xpu_interval_datum_store(kern_context *kcxt,
						 char *buffer,
						 const xpu_datum_t *__arg)
{
	const xpu_interval_t *arg = (const xpu_interval_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		memcpy(buffer, &arg->value, sizeof(Interval));
	return sizeof(Interval);
}

STATIC_FUNCTION(bool)
xpu_interval_datum_hash(kern_context *kcxt,
						uint32_t *p_hash,
						const xpu_datum_t *__arg)
{
	const xpu_interval_t *arg = (const xpu_interval_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(Interval));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(interval, false, 8, sizeof(Interval));


STATIC_FUNCTION(int)
date2j(int y, int m, int d)
{
	int		julian;
	int		century;

	if (m > 2)
	{
		m += 1;
		y += 4800;
	}
	else
	{
		m += 13;
		y += 4799;
	}
	century = y / 100;
	julian = y * 365 - 32167;
	julian += y / 4 - century + century / 4;
	julian += 7834 * m / 256 + d;

	return julian;
}

STATIC_FUNCTION(void)
j2date(int jd, int *year, int *month, int *day)
{
	unsigned int julian;
	unsigned int quad;
	unsigned int extra;
	int		y;

	julian = jd;
	julian += 32044;
	quad = julian / 146097;
	extra = (julian - quad * 146097) * 4 + 3;
	julian += 60 + quad * 3 + extra / 146097;
	quad = julian / 1461;
	julian -= quad * 1461;
	y = julian * 4 / 1461;
	julian = ((y != 0) ? ((julian + 305) % 365) : ((julian + 306) % 366)) + 123;
	y += quad * 4;
	*year = y - 4800;
	quad = julian * 2141 / 65536;
	*day = julian - 7834 * quad / 256;
	*month = (quad + 10) % MONTHS_PER_YEAR + 1;
}

INLINE_FUNCTION(int)
j2day(int date)
{
	date += 1;
	date %= 7;
	/* Cope if division truncates towards zero, as it probably does */
	if (date < 0)
		date += 7;
	return date;
}	/* j2day() */

INLINE_FUNCTION(void)
dt2time(Timestamp jd, int *hour, int *min, int *sec, int *fsec)
{
	TimeOffset	time = jd;

	*hour = time / USECS_PER_HOUR;
	time -= (*hour) * USECS_PER_HOUR;
	*min = time / USECS_PER_MINUTE;
	time -= (*min) * USECS_PER_MINUTE;
	*sec = time / USECS_PER_SEC;
	*fsec = time - (*sec * USECS_PER_SEC);
}

INLINE_FUNCTION(TimeOffset)
time2t(const int hour, const int min, const int sec, const fsec_t fsec)
{
	return (hour * 3600 + min  * 60 + sec) * USECS_PER_SEC + fsec;
}

STATIC_FUNCTION(int)
pg_next_dst_boundary(const pg_time_t *timep,
					 long int *before_gmtoff,
					 int *before_isdst,
					 pg_time_t *boundary,
					 long int *after_gmtoff,
					 int *after_isdst,
					 const pg_tz *tz)
{
	const struct pg_tz_ttinfo *ttisp;
    const pg_time_t t = *timep;
	int			i, j;

	if (tz->state.timecnt == 0)
    {
        /* non-DST zone, use lowest-numbered standard type */
        i = 0;
        while (tz->state.ttis[i].tt_isdst)
		{
            if (++i >= tz->state.typecnt)
            {
                i = 0;
                break;
            }
		}
        ttisp = &tz->state.ttis[i];
        *before_gmtoff = ttisp->tt_utoff;
        *before_isdst = ttisp->tt_isdst;
        return 0;
    }
	if ((tz->state.goback  && t < tz->state.ats[0]) ||
		(tz->state.goahead && t > tz->state.ats[tz->state.timecnt - 1]))
    {
		/* For values outside the transition table, extrapolate */
		pg_time_t	newt = t;
		pg_time_t	seconds;
		pg_time_t	tcycles;
		int64_t		icycles;
		int			result;

		if (t < tz->state.ats[0])
			seconds = tz->state.ats[0] - t;
		else
			seconds = t - tz->state.ats[tz->state.timecnt - 1];
		--seconds;
		tcycles = seconds / YEARSPERREPEAT / AVGSECSPERYEAR;
		++tcycles;
		icycles = tcycles;
		if (tcycles - icycles >= 1 || icycles - tcycles >= 1)
			return -1;
		seconds = icycles;
		seconds *= YEARSPERREPEAT;
		seconds *= AVGSECSPERYEAR;
		if (t < tz->state.ats[0])
			newt += seconds;
		else
			newt -= seconds;
		if (newt < tz->state.ats[0] ||
			newt > tz->state.ats[tz->state.timecnt - 1])
			return -1;		/* "cannot happen" */

		result = pg_next_dst_boundary(&newt, before_gmtoff,
									  before_isdst,
									  boundary,
									  after_gmtoff,
									  after_isdst,
									  tz);
		if (t < tz->state.ats[0])
			*boundary -= seconds;
		else
			*boundary += seconds;
		return result;
	}
	if (t >= tz->state.ats[tz->state.timecnt - 1])
	{
		/* No known transition > t, so use last known segment's type */
		i = tz->state.types[tz->state.timecnt - 1];
		ttisp = &tz->state.ttis[i];
		*before_gmtoff = ttisp->tt_utoff;
		*before_isdst = ttisp->tt_isdst;
		return 0;
	}
	if (t < tz->state.ats[0])
	{
		/* For "before", use lowest-numbered standard type */
		i = 0;
		while (tz->state.ttis[i].tt_isdst)
		{
			if (++i >= tz->state.typecnt)
			{
				i = 0;
				break;
			}
		}
		ttisp = &tz->state.ttis[i];
		*before_gmtoff = ttisp->tt_utoff;
		*before_isdst = ttisp->tt_isdst;
		*boundary = tz->state.ats[0];
		/* And for "after", use the first segment's type */
		i = tz->state.types[0];
		ttisp = &tz->state.ttis[i];
		*after_gmtoff = ttisp->tt_utoff;
		*after_isdst = ttisp->tt_isdst;
		return 1;
	}
	/* Else search to find the boundary following t */
	{
		int			lo = 1;
		int			hi = tz->state.timecnt - 1;

		while (lo < hi)
		{
			int		mid = (lo + hi) >> 1;

			if (t < tz->state.ats[mid])
				hi = mid;
			else
				lo = mid + 1;
		}
		i = lo;
	}
	j = tz->state.types[i - 1];
	ttisp = &tz->state.ttis[j];
	*before_gmtoff = ttisp->tt_utoff;
	*before_isdst = ttisp->tt_isdst;
	*boundary = tz->state.ats[i];
	j = tz->state.types[i];
	ttisp = &tz->state.ttis[j];
	*after_gmtoff = ttisp->tt_utoff;
	*after_isdst = ttisp->tt_isdst;
	return 1;
}

STATIC_FUNCTION(int)
DetermineTimeZoneOffset(struct pg_tm *tm, const pg_tz *tzp)
{
	pg_time_t	t;
	pg_time_t  *tp = &t;
	int			date, sec;
    pg_time_t	day,
				mytime,
				prevtime,
				boundary,
				beforetime,
				aftertime;
    long int	before_gmtoff,
				after_gmtoff;
    int			before_isdst,
				after_isdst;
    int			res;

	/*
	 * First, generate the pg_time_t value corresponding to the given
	 * y/m/d/h/m/s taken as GMT time.  If this overflows, punt and decide the
	 * timezone is GMT.  (For a valid Julian date, integer overflow should be
	 * impossible with 64-bit pg_time_t, but let's check for safety.)
	 */
	if (!IS_VALID_JULIAN(tm->tm_year, tm->tm_mon, tm->tm_mday))
		goto overflow;
	date = date2j(tm->tm_year, tm->tm_mon, tm->tm_mday) - UNIX_EPOCH_JDATE;

	day = ((pg_time_t) date) * SECS_PER_DAY;
	if (day / SECS_PER_DAY != date)
		goto overflow;
	sec = tm->tm_sec + (tm->tm_min + tm->tm_hour * MINS_PER_HOUR) * SECS_PER_MINUTE;
	mytime = day + sec;
	/* since sec >= 0, overflow could only be from +day to -mytime */
	if (mytime < 0 && day > 0)
		goto overflow;

	/*
	 * Find the DST time boundary just before or following the target time. We
	 * assume that all zones have GMT offsets less than 24 hours, and that DST
	 * boundaries can't be closer together than 48 hours, so backing up 24
	 * hours and finding the "next" boundary will work.
	 */
	prevtime = mytime - SECS_PER_DAY;
	if (mytime < 0 && prevtime > 0)
		goto overflow;

	res = pg_next_dst_boundary(&prevtime,
							   &before_gmtoff, &before_isdst,
							   &boundary,
							   &after_gmtoff, &after_isdst,
							   tzp);
	if (res < 0)
		goto overflow;		/* failure? */

	if (res == 0)
	{
		/* Non-DST zone, life is simple */
		tm->tm_isdst = before_isdst;
		*tp = mytime - before_gmtoff;
		return -(int) before_gmtoff;
	}

	/*
	 * Form the candidate pg_time_t values with local-time adjustment
	 */
	beforetime = mytime - before_gmtoff;
	if ((before_gmtoff > 0 &&
		 mytime < 0 && beforetime > 0) ||
		(before_gmtoff <= 0 &&
		 mytime > 0 && beforetime < 0))
		goto overflow;
	aftertime = mytime - after_gmtoff;
	if ((after_gmtoff > 0 &&
		 mytime < 0 && aftertime > 0) ||
		(after_gmtoff <= 0 &&
		 mytime > 0 && aftertime < 0))
		goto overflow;

	/*
	 * If both before or both after the boundary time, we know what to do. The
	 * boundary time itself is considered to be after the transition, which
	 * means we can accept aftertime == boundary in the second case.
	 */
	if (beforetime < boundary && aftertime < boundary)
	{
		tm->tm_isdst = before_isdst;
		*tp = beforetime;
		return -(int) before_gmtoff;
	}
	if (beforetime > boundary && aftertime >= boundary)
	{
		tm->tm_isdst = after_isdst;
		*tp = aftertime;
		return -(int) after_gmtoff;
	}

	/*
	 * It's an invalid or ambiguous time due to timezone transition.  In a
	 * spring-forward transition, prefer the "before" interpretation; in a
	 * fall-back transition, prefer "after".  (We used to define and implement
	 * this test as "prefer the standard-time interpretation", but that rule
	 * does not help to resolve the behavior when both times are reported as
	 * standard time; which does happen, eg Europe/Moscow in Oct 2014.  Also,
	 * in some zones such as Europe/Dublin, there is widespread confusion
	 * about which time offset is "standard" time, so it's fortunate that our
	 * behavior doesn't depend on that.)
	 */
	if (beforetime > aftertime)
	{
		tm->tm_isdst = before_isdst;
		*tp = beforetime;
		return -(int) before_gmtoff;
	}
	tm->tm_isdst = after_isdst;
	*tp = aftertime;
	return -(int) after_gmtoff;

overflow:
	/* Given date is out of range, so assume UTC */
	tm->tm_isdst = 0;
	*tp = 0;
	return 0;
}

INLINE_FUNCTION(bool)
increment_overflow(int *ip, int x)
{
	int64_t		ival = *ip;

	ival += x;
	if (ival < INT_MIN || ival > INT_MAX)
		return false;
	*ip = ival;
	return true;
}

INLINE_FUNCTION(int)
leaps_thru_end_of_nonneg(int y)
{
    return y / 4 - y / 100 + y / 400;
}

INLINE_FUNCTION(int)
leaps_thru_end_of(int y)
{
	if (y < 0)
		return -1 - leaps_thru_end_of_nonneg(-1 - y);
	return leaps_thru_end_of_nonneg(y);
}

STATIC_FUNCTION(bool)
__timesub(struct pg_tm *tm, pg_time_t t, int32_t offset, const pg_tz *tz)
{
	const struct pg_tz_lsinfo *lp;
	pg_time_t	tdays;
	int			idays;		/* unsigned would be so 2003 */
	int64_t		rem;
	int64_t		corr;
	const int  *ip;
	bool		hit;
	int			i, y;

	corr = 0;
	hit = false;
	i = tz->state.leapcnt;
	while (--i >= 0)
	{
		lp = &tz->state.lsis[i];
		if (t >= lp->ls_trans)
		{
			corr = lp->ls_corr;
			hit = (t == lp->ls_trans && (i == 0 ? 0 : lp[-1].ls_corr) < corr);
			break;
		}
	}
	y = EPOCH_YEAR;
	tdays = t / SECSPERDAY;
	rem = t % SECSPERDAY;
	while (tdays < 0 || tdays >= year_lengths[isleap(y)])
	{
		pg_time_t	tdelta;
		int			newy;
		int			idelta;
		int			leapdays;

		tdelta = tdays / DAYSPERLYEAR;
		if (tdelta < INT_MIN || tdelta > INT_MAX)
			return false;
		idelta = tdelta;
		if (idelta == 0)
			idelta = (tdays < 0) ? -1 : 1;
		newy = y;
		if (increment_overflow(&newy, idelta))
			return false;
		leapdays = (leaps_thru_end_of(newy - 1) -
					leaps_thru_end_of(y - 1));
		tdays -= ((pg_time_t) newy - y) * DAYSPERNYEAR;
		tdays -= leapdays;
		y = newy;
	}
	/*
	 * Given the range, we can now fearlessly cast...
	 */
	idays = tdays;
	rem += offset - corr;
	while (rem < 0)
	{
		rem += SECSPERDAY;
		--idays;
	}
	while (rem >= SECSPERDAY)
	{
		rem -= SECSPERDAY;
		++idays;
	}
	while (idays < 0)
	{
		if (increment_overflow(&y, -1))
			return false;
		idays += year_lengths[isleap(y)];
	}
	while (idays >= year_lengths[isleap(y)])
	{
		idays -= year_lengths[isleap(y)];
		if (increment_overflow(&y, 1))
			return false;
	}
	tm->tm_year = y;
	if (increment_overflow(&tm->tm_year, -TM_YEAR_BASE))
		return false;
	tm->tm_yday = idays;

	/*
	 * The "extra" mods below avoid overflow problems.
	 */
	tm->tm_wday = EPOCH_WDAY +
		((y - EPOCH_YEAR) % DAYSPERWEEK) *
		(DAYSPERNYEAR % DAYSPERWEEK) +
		leaps_thru_end_of(y - 1) -
		leaps_thru_end_of(EPOCH_YEAR - 1) +
		idays;
	tm->tm_wday %= DAYSPERWEEK;
	if (tm->tm_wday < 0)
		tm->tm_wday += DAYSPERWEEK;
	tm->tm_hour = (int) (rem / SECSPERHOUR);
	rem %= SECSPERHOUR;
	tm->tm_min = (int) (rem / SECSPERMIN);

	/*
	 * A positive leap second requires a special representation. This uses
	 * "... ??:59:60" et seq.
	 */
	tm->tm_sec = (int) (rem % SECSPERMIN) + hit;
	ip = mon_lengths[isleap(y)];
	for (tm->tm_mon = 0; idays >= ip[tm->tm_mon]; ++(tm->tm_mon))
		idays -= ip[tm->tm_mon];
    tm->tm_mday = (int) (idays + 1);
    tm->tm_isdst = 0;
    tm->tm_gmtoff = offset;

	return true;
}

STATIC_FUNCTION(bool)
pg_localtime(struct pg_tm *tx, pg_time_t t, const pg_tz *tz)
{
	const struct pg_tz_ttinfo *ttisp;
	bool		rv;
	int         i;

	assert(tz != NULL);
	if ((tz->state.goback && t < tz->state.ats[0]) ||
        (tz->state.goahead && t > tz->state.ats[tz->state.timecnt - 1]))
    {
		pg_time_t	newt = t;
		pg_time_t	seconds;
		pg_time_t	years;

		if (t < tz->state.ats[0])
			seconds = tz->state.ats[0] - t;
		else
			seconds = t - tz->state.ats[tz->state.timecnt - 1];
		--seconds;
		years = (seconds / SECSPERREPEAT + 1) * YEARSPERREPEAT;
		seconds = years * AVGSECSPERYEAR;
		if (t < tz->state.ats[0])
			newt += seconds;
		else
			newt -= seconds;
		if (newt < tz->state.ats[0] ||
			newt > tz->state.ats[tz->state.timecnt - 1])
			return false;		/* cannot happen */
		rv = pg_localtime(tx, newt, tz);
		if (rv)
		{
			int64_t		newy = tx->tm_year;

			if (t < tz->state.ats[0])
				newy -= years;
			else
				newy += years;
			if (!(INT_MIN <= newy && newy <= INT_MAX))
				return false;
			tx->tm_year = newy;
		}
		return rv;
	}

	if (tz->state.timecnt == 0 || t < tz->state.ats[0])
	{
		i = tz->state.defaulttype;
	}
	else
	{
		int		lo = 1;
        int		hi = tz->state.timecnt;

		while (lo < hi)
		{
			int		mid = (lo + hi) >> 1;

			if (t < tz->state.ats[mid])
				hi = mid;
			else
				lo = mid + 1;
		}
		i = (int) tz->state.types[lo - 1];
    }
	ttisp = &tz->state.ttis[i];

	/*
	 * To get (wrong) behavior that's compatible with System V Release 2.0
	 * you'd replace the statement below with t += ttisp->tt_utoff;
	 * timesub(&t, 0L, sp, tmp);
	 */
    rv = __timesub(tx, t, ttisp->tt_utoff, tz);
	if (rv)
		tx->tm_isdst = ttisp->tt_isdst;
	tx->tm_zone = NULL;
	return rv;
}

STATIC_FUNCTION(bool)
timestamp2tm(Timestamp dt, struct pg_tm *tm, fsec_t *fsec, const pg_tz *tz_info)
{
	Timestamp	date;
	Timestamp	time;
	pg_time_t	utime;

	/* dt -> date + time */
	time = dt;
	date = time / USECS_PER_DAY;
	if (date != 0)
		time -= date * USECS_PER_DAY;
	if (time < 0)
	{
		time += USECS_PER_DAY;
		date -= 1;
	}

	/* add offset to go from J2000 back to standard Julian date */
	date += POSTGRES_EPOCH_JDATE;

	/* Julian day routine does not work for negative Julian days */
	if (date < 0 || date > (Timestamp) INT_MAX)
		return false;

	j2date((int)date, &tm->tm_year, &tm->tm_mon, &tm->tm_mday);
	dt2time(time, &tm->tm_hour, &tm->tm_min, &tm->tm_sec, fsec);

	/* Does is need TZ conversion? */
	if (tz_info)
	{
		dt = (dt - *fsec) / USECS_PER_SEC +
			(POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * SECS_PER_DAY;
		utime = (pg_time_t) dt;
		if ((Timestamp) utime == dt)
		{
			struct pg_tm	tx;

			pg_localtime(&tx, utime, tz_info);
			tm->tm_year   = tx.tm_year + 1900;
			tm->tm_mon    = tx.tm_mon + 1;
			tm->tm_mday   = tx.tm_mday;
			tm->tm_hour   = tx.tm_hour;
			tm->tm_min    = tx.tm_min;
			tm->tm_sec    = tx.tm_sec;
			tm->tm_isdst  = tx.tm_isdst;
			tm->tm_gmtoff = tx.tm_gmtoff;
			tm->tm_zone   = NULL;
			//*tzp = -tm->tm_gmtoff;
			return true;
		}
	}
	/* elsewhere, no TZ conversion possible */
	tm->tm_isdst = -1;
	tm->tm_gmtoff = 0;
	tm->tm_zone = NULL;
	return true;
}

STATIC_FUNCTION(bool)
tm2timestamp(Timestamp *result, const struct pg_tm *tm, fsec_t fsec,
			 const pg_tz *tz_info)
{
	TimeOffset	date;
	TimeOffset	time;
	Timestamp	ts;

	/* Prevent overflow in Julian-day routines */
	if (!IS_VALID_JULIAN(tm->tm_year, tm->tm_mon, tm->tm_mday))
		return false;
	date = date2j(tm->tm_year, tm->tm_mon, tm->tm_mday) - POSTGRES_EPOCH_JDATE;
	time = time2t(tm->tm_hour, tm->tm_min, tm->tm_sec, fsec);

	ts = date * USECS_PER_DAY + time;
	/* check for major overflow */
	if ((ts - time) / USECS_PER_DAY != date)
		return false;
	if ((ts < 0 && date > 0) || (ts > 0 && date < -1))
		return false;
	/* TZ conversion, if any */
	if (tz_info)
		ts -= (tm->tm_gmtoff * USECS_PER_SEC);
	/* final range check */
	if (!IS_VALID_TIMESTAMP(ts))
		return false;
	*result = ts;
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_date_to_timestamp(XPU_PGFUNCTION_ARGS)
{
	xpu_timestamp_t *result = (xpu_timestamp_t *)__result;
	xpu_date_t	datum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, date));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
		return false;
	result->isnull = datum.isnull;
	if (!datum.isnull)
	{
		if (DATE_IS_NOBEGIN(datum.value))
			TIMESTAMP_NOBEGIN(result->value);
		else if (DATE_IS_NOEND(datum.value))
			TIMESTAMP_NOEND(result->value);
		else if (datum.value < (TIMESTAMP_END_JULIAN - POSTGRES_EPOCH_JDATE))
			result->value = datum.value * USECS_PER_DAY;
		else
		{
			STROM_ELOG(kcxt, "date out of range for timestamp");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_date_to_timestamptz(XPU_PGFUNCTION_ARGS)
{
	xpu_timestamptz_t *result = (xpu_timestamptz_t *)__result;
	xpu_date_t	datum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, date));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
		return false;
	result->isnull = datum.isnull;
	if (!datum.isnull)
	{
		if (DATE_IS_NOBEGIN(datum.value))
			TIMESTAMP_NOBEGIN(result->value);
		else if (DATE_IS_NOEND(datum.value))
			TIMESTAMP_NOEND(result->value);
		else if (datum.value >= (TIMESTAMP_END_JULIAN - POSTGRES_EPOCH_JDATE))
		{
			STROM_ELOG(kcxt, "date out of range for timestamp");
			return false;
		}
		else
		{
			TimestampTz		tval;
			struct pg_tm	tt;
			int				tz;

			memset(&tt, 0, sizeof(tt));
			j2date(datum.value + POSTGRES_EPOCH_JDATE,
				   &tt.tm_year, &tt.tm_mon, &tt.tm_mday);
			tz = DetermineTimeZoneOffset(&tt, SESSION_TIMEZONE(kcxt->session));
			tval = datum.value * USECS_PER_DAY + tz * USECS_PER_SEC;

			if (!IS_VALID_TIMESTAMP(tval))
			{
				STROM_ELOG(kcxt, "date out of range for timestamp");
				return false;
			}
			result->value = tval;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamptz_to_timestamp(XPU_PGFUNCTION_ARGS)
{
	xpu_timestamp_t	   *result = (xpu_timestamp_t *)__result;
	xpu_timestamptz_t	datum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, timestamptz));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
		return false;
	result->isnull = datum.isnull;
	if (!datum.isnull)
	{
		struct pg_tm	tm;
		fsec_t			fsec;

		if (TIMESTAMP_NOT_FINITE(datum.value))
			result->value = datum.value;
		else if (!timestamp2tm(datum.value, &tm, &fsec, NULL) ||
				 !tm2timestamp(&result->value, &tm, fsec, NULL))
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamp_to_timestamptz(XPU_PGFUNCTION_ARGS)
{
	xpu_timestamptz_t  *result = (xpu_timestamptz_t *)__result;
	xpu_timestamp_t		datum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, timestamp));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
		return false;
	result->isnull = datum.isnull;
	if (!datum.isnull)
	{
		struct pg_tm	tm;
		Timestamp		ts = datum.value;
		fsec_t			fsec;

		if (TIMESTAMP_NOT_FINITE(datum.value))
			result->value = datum.value;
		else if (!timestamp2tm(datum.value, &tm, &fsec,
							   SESSION_TIMEZONE(kcxt->session)))
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
		ts -= tm.tm_gmtoff * USECS_PER_SEC;
		if (!IS_VALID_TIMESTAMP(ts))
		{
			if (ts < MIN_TIMESTAMP)
				ts = DT_NOBEGIN;
			else
				ts = DT_NOEND;
		}
		result->value = ts;
	}
	return true;
}


PG_SIMPLE_COMPARE_TEMPLATE(date_,date,date,DateADT)
PG_SIMPLE_COMPARE_TEMPLATE(time_,time,time,TimeADT)
PG_SIMPLE_COMPARE_TEMPLATE(timestamp_,timestamp,timestamp,Timestamp)
PG_SIMPLE_COMPARE_TEMPLATE(timestamptz_,timestamptz,timestamptz,TimestampTz)

#define PG_TIMETZ_COMPARE_TEMPLATE(NAME,OPER)							\
	PUBLIC_FUNCTION(bool)												\
	pgfn_timetz_##NAME(XPU_PGFUNCTION_ARGS)								\
	{																	\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;				\
		xpu_timetz_t	datum_a;										\
		xpu_timetz_t	datum_b;										\
		TimeOffset		t1, t2;											\
		int				comp;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg,timetz));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, timetz));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		result->isnull = (datum_a.isnull | datum_b.isnull);				\
		if (!result->isnull)											\
		{																\
			t1 = datum_a.value.time + datum_a.value.zone * USECS_PER_SEC; \
			t2 = datum_b.value.time + datum_b.value.zone * USECS_PER_SEC; \
																		\
			if (t1 > t2)												\
				comp = 1;												\
			else if (t1 < t2)											\
				comp = -1;												\
			else if (datum_a.value.zone > datum_b.value.zone)			\
				comp = 1;												\
			else if (datum_a.value.zone < datum_b.value.zone)			\
				comp = -1;												\
			else														\
				comp = 0;												\
																		\
			result->value = (comp OPER 0);								\
		}																\
		return true;													\
	}
PG_TIMETZ_COMPARE_TEMPLATE(eq, ==)
PG_TIMETZ_COMPARE_TEMPLATE(ne, !=)
PG_TIMETZ_COMPARE_TEMPLATE(lt, <)
PG_TIMETZ_COMPARE_TEMPLATE(le, <=)
PG_TIMETZ_COMPARE_TEMPLATE(gt, >)
PG_TIMETZ_COMPARE_TEMPLATE(ge, >=)

INLINE_FUNCTION(int)
__compare_date_timestamp(DateADT a, Timestamp b)
{
	Timestamp	ts;

	if (DATE_IS_NOBEGIN(a))
		ts = DT_NOBEGIN;
	else if (DATE_IS_NOEND(a) || a >= (TIMESTAMP_END_JULIAN -
									   POSTGRES_EPOCH_JDATE))
		ts = DT_NOEND;
	else
		ts = a * USECS_PER_DAY;

	if (ts < b)
		return -1;
	if (ts > b)
		return 1;
	return 0;
}

#define PG_DATE_TIMESTAMP_COMPARE_TEMPLATE(NAME,OPER)					\
	PUBLIC_FUNCTION(bool)												\
	pgfn_date_##NAME##_timestamp(XPU_PGFUNCTION_ARGS)					\
	{																	\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;				\
		xpu_date_t		datum_a;										\
		xpu_timestamp_t	datum_b;										\
		int				comp;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg,date));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, timestamp));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		result->isnull = (datum_a.isnull | datum_b.isnull);				\
		if (!result->isnull)											\
		{																\
			comp = __compare_date_timestamp(datum_a.value,				\
											datum_b.value);				\
			result->value = (comp OPER 0);								\
		}																\
		return true;													\
	}
PG_DATE_TIMESTAMP_COMPARE_TEMPLATE(eq, ==);
PG_DATE_TIMESTAMP_COMPARE_TEMPLATE(ne, !=);
PG_DATE_TIMESTAMP_COMPARE_TEMPLATE(lt, <);
PG_DATE_TIMESTAMP_COMPARE_TEMPLATE(le, <=);
PG_DATE_TIMESTAMP_COMPARE_TEMPLATE(gt, >);
PG_DATE_TIMESTAMP_COMPARE_TEMPLATE(ge, >=);

#define PG_TIMESTAMP_DATE_COMPARE_TEMPLATE(NAME,OPER)					\
	PUBLIC_FUNCTION(bool)												\
	pgfn_timestamp_##NAME##_date(XPU_PGFUNCTION_ARGS)					\
	{																	\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;				\
		xpu_timestamp_t	datum_a;										\
		xpu_date_t		datum_b;										\
		int				comp;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg, timestamp));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, date));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		result->isnull = (datum_a.isnull | datum_b.isnull);				\
		if (!result->isnull)											\
		{																\
			comp = __compare_date_timestamp(datum_b.value,				\
											datum_a.value);				\
			result->value = (0 OPER comp);								\
		}																\
		return true;													\
	}
PG_TIMESTAMP_DATE_COMPARE_TEMPLATE(eq, ==)
PG_TIMESTAMP_DATE_COMPARE_TEMPLATE(ne, !=)
PG_TIMESTAMP_DATE_COMPARE_TEMPLATE(lt, <)
PG_TIMESTAMP_DATE_COMPARE_TEMPLATE(le, <=)
PG_TIMESTAMP_DATE_COMPARE_TEMPLATE(gt, >)
PG_TIMESTAMP_DATE_COMPARE_TEMPLATE(ge, >=)

INLINE_FUNCTION(int)
__compare_date_timestamptz(DateADT a, TimestampTz b, const pg_tz *tz_info)
{
	Timestamp		ts;
	struct pg_tm	tm;
	int				tz;

	if (DATE_IS_NOBEGIN(a))
		ts = DT_NOBEGIN;
	else if (DATE_IS_NOEND(a) || a >= (TIMESTAMP_END_JULIAN -
									   POSTGRES_EPOCH_JDATE))
		ts = DT_NOEND;
	else
	{
		memset(&tm, 0, sizeof(tm));
		j2date(a + POSTGRES_EPOCH_JDATE,
			   &tm.tm_year,
			   &tm.tm_mon,
			   &tm.tm_mday);
		tz = DetermineTimeZoneOffset(&tm, tz_info);
		ts = a * USECS_PER_DAY + tz * USECS_PER_SEC;

		if (!IS_VALID_TIMESTAMP(ts))
		{
			if (ts < MIN_TIMESTAMP)
				ts = DT_NOBEGIN;
			else
				ts = DT_NOEND;
		}
	}
	/* timestamptz_cmp_internal */
	if (ts < b)
		return -1;
	if (ts > b)
		return 1;
	return 0;
}

#define PG_DATE_TIMESTAMPTZ_COMPARE_TEMPLATE(NAME,OPER)					\
	PUBLIC_FUNCTION(bool)												\
	pgfn_date_##NAME##_timestamptz(XPU_PGFUNCTION_ARGS)					\
	{																	\
		xpu_bool_t *result = (xpu_bool_t *)__result;					\
		xpu_date_t	datum_a;											\
		xpu_timestamptz_t datum_b;										\
		int			comp;												\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg, date));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, timestamptz));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		result->isnull = (datum_a.isnull | datum_b.isnull);				\
		if (!result->isnull)											\
		{																\
			pg_tz  *tz_info = SESSION_TIMEZONE(kcxt->session);			\
			comp = __compare_date_timestamptz(datum_a.value,			\
											  datum_b.value,			\
											  tz_info);					\
			result->value = (comp OPER 0);								\
		}																\
		return true;													\
	}
PG_DATE_TIMESTAMPTZ_COMPARE_TEMPLATE(eq, ==);
PG_DATE_TIMESTAMPTZ_COMPARE_TEMPLATE(ne, !=);
PG_DATE_TIMESTAMPTZ_COMPARE_TEMPLATE(lt, <);
PG_DATE_TIMESTAMPTZ_COMPARE_TEMPLATE(le, <=);
PG_DATE_TIMESTAMPTZ_COMPARE_TEMPLATE(gt, >);
PG_DATE_TIMESTAMPTZ_COMPARE_TEMPLATE(ge, >=);

#define PG_TIMESTAMPTZ_DATE_COMPARE_TEMPLATE(NAME,OPER)					\
	PUBLIC_FUNCTION(bool)												\
	pgfn_timestamptz_##NAME##_date(XPU_PGFUNCTION_ARGS)					\
	{																	\
		xpu_bool_t *result = (xpu_bool_t *)__result;					\
		xpu_timestamptz_t datum_a;										\
		xpu_date_t	datum_b;											\
		int			comp;												\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg, timestamptz));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		KEXP_IS_VALID(karg, date);										\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		result->isnull = (datum_a.isnull | datum_b.isnull);				\
		if (!result->isnull)											\
		{																\
			pg_tz  *tz_info = SESSION_TIMEZONE(kcxt->session);			\
			comp = __compare_date_timestamptz(datum_b.value,			\
											  datum_a.value,			\
											  tz_info);					\
			result->value = (0 OPER comp);								\
		}																\
		return true;													\
	}
PG_TIMESTAMPTZ_DATE_COMPARE_TEMPLATE(eq, ==);
PG_TIMESTAMPTZ_DATE_COMPARE_TEMPLATE(ne, !=);
PG_TIMESTAMPTZ_DATE_COMPARE_TEMPLATE(lt, <);
PG_TIMESTAMPTZ_DATE_COMPARE_TEMPLATE(le, <=);
PG_TIMESTAMPTZ_DATE_COMPARE_TEMPLATE(gt, >);
PG_TIMESTAMPTZ_DATE_COMPARE_TEMPLATE(ge, >=);

INLINE_FUNCTION(int)
__compare_timestamp_timestamptz(Timestamp a,
								TimestampTz b,
								const pg_tz *tz_info)
{
	TimestampTz		ts;
	struct pg_tm	tm;
	fsec_t			fsec;
	int				tz;

	if (TIMESTAMP_NOT_FINITE(a))
		ts = a;
	else if (!timestamp2tm(a, &tm, &fsec, NULL))
		ts = (a < 0 ? DT_NOBEGIN : DT_NOEND);
	else
	{
		tz = DetermineTimeZoneOffset(&tm, tz_info);
		ts = a * USECS_PER_SEC + tz * USECS_PER_SEC;
		if (ts < MIN_TIMESTAMP)
			ts = DT_NOBEGIN;
		else if (ts > END_TIMESTAMP)
			ts = DT_NOEND;
	}
	if (ts < b)
		return -1;
	if (ts > b)
		return 1;
	return 0;
}

#define PG_TIMESTAMP_TIMESTAMPTZ_COMPARE_TEMPLATE(NAME,OPER)			\
	PUBLIC_FUNCTION(bool)                                               \
	pgfn_timestamp_##NAME##_timestamptz(XPU_PGFUNCTION_ARGS)			\
	{                                                                   \
		xpu_bool_t *result = (xpu_bool_t *)__result;                    \
        xpu_timestamp_t		datum_a;									\
        xpu_timestamptz_t	datum_b;									\
        int			comp;												\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg, timestamp));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))                \
			return false;                                               \
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, timestamptz));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;                                               \
                                                                        \
		result->isnull = (datum_a.isnull | datum_b.isnull);             \
        if (!result->isnull)                                            \
        {                                                               \
			const pg_tz	*tz_info = SESSION_TIMEZONE(kcxt->session);		\
            comp = __compare_timestamp_timestamptz(datum_b.value,		\
												   datum_a.value,		\
												   tz_info);			\
			result->value = (comp OPER 0);                              \
        }                                                               \
        return true;                                                    \
    }
PG_TIMESTAMP_TIMESTAMPTZ_COMPARE_TEMPLATE(eq, ==)
PG_TIMESTAMP_TIMESTAMPTZ_COMPARE_TEMPLATE(ne, !=)
PG_TIMESTAMP_TIMESTAMPTZ_COMPARE_TEMPLATE(lt, <)
PG_TIMESTAMP_TIMESTAMPTZ_COMPARE_TEMPLATE(le, <=)
PG_TIMESTAMP_TIMESTAMPTZ_COMPARE_TEMPLATE(gt, >)
PG_TIMESTAMP_TIMESTAMPTZ_COMPARE_TEMPLATE(ge, >=)

#define PG_TIMESTAMPTZ_TIMESTAMP_COMPARE_TEMPLATE(NAME,OPER)			\
	PUBLIC_FUNCTION(bool)                                               \
	pgfn_timestamptz_##NAME##_timestamp(XPU_PGFUNCTION_ARGS)			\
	{                                                                   \
		xpu_bool_t *result = (xpu_bool_t *)__result;                    \
        xpu_timestamp_t		datum_a;									\
        xpu_timestamptz_t	datum_b;									\
        int			comp;												\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg, timestamp));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))                \
			return false;                                               \
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, timestamptz));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))                \
			return false;                                               \
                                                                        \
		result->isnull = (datum_a.isnull | datum_b.isnull);             \
        if (!result->isnull)                                            \
        {                                                               \
			const pg_tz *tz_info = SESSION_TIMEZONE(kcxt->session);		\
			comp = __compare_timestamp_timestamptz(datum_b.value,		\
												   datum_a.value,		\
												   tz_info);			\
			result->value = (0 OPER comp);                              \
        }                                                               \
        return true;                                                    \
    }
PG_TIMESTAMPTZ_TIMESTAMP_COMPARE_TEMPLATE(eq, ==)
PG_TIMESTAMPTZ_TIMESTAMP_COMPARE_TEMPLATE(ne, !=)
PG_TIMESTAMPTZ_TIMESTAMP_COMPARE_TEMPLATE(lt, <)
PG_TIMESTAMPTZ_TIMESTAMP_COMPARE_TEMPLATE(le, <=)
PG_TIMESTAMPTZ_TIMESTAMP_COMPARE_TEMPLATE(gt, >)
PG_TIMESTAMPTZ_TIMESTAMP_COMPARE_TEMPLATE(ge, >=)

INLINE_FUNCTION(int128_t)
interval_cmp_value(const Interval *ival)
{
	int128_t	span;
	int64_t		days;

	span = ival->time % USECS_PER_DAY;
	days = ival->time / USECS_PER_DAY;
	days += ival->month * 30;
	days += ival->day;

	span += (int128_t)days * USECS_PER_DAY;

	return span;
}

#define PG_INTERVAL_COMPARE_TEMPLATE(NAME,OPER)							\
	PUBLIC_FUNCTION(bool)                                               \
	pgfn_interval_##NAME(XPU_PGFUNCTION_ARGS)							\
	{                                                                   \
		xpu_bool_t *result = (xpu_bool_t *)__result;                    \
        xpu_interval_t	datum_a;										\
        xpu_interval_t	datum_b;										\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 &&									\
			   KEXP_IS_VALID(karg, interval));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))                \
			return false;                                               \
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, interval));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))                \
			return false;                                               \
                                                                        \
		result->isnull = (datum_a.isnull | datum_b.isnull);             \
		if (!result->isnull)                                            \
		{                                                               \
            int128_t	aval = interval_cmp_value(&datum_a.value);		\
            int128_t	bval = interval_cmp_value(&datum_b.value);		\
																		\
			result->value = (aval OPER bval);							\
        }                                                               \
        return true;                                                    \
    }
PG_INTERVAL_COMPARE_TEMPLATE(eq, ==)
PG_INTERVAL_COMPARE_TEMPLATE(ne, !=)
PG_INTERVAL_COMPARE_TEMPLATE(lt, <)
PG_INTERVAL_COMPARE_TEMPLATE(le, <=)
PG_INTERVAL_COMPARE_TEMPLATE(gt, >)
PG_INTERVAL_COMPARE_TEMPLATE(ge, >=)
