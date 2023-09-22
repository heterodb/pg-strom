/*
 * xpu_timelib.cu
 *
 * Collection of the primitive Date/Time type support for both of GPU and DPU
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"

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
				   int vclass,
				   const kern_variable *kvar)
{
	xpu_date_t *result = (xpu_date_t *)__result;

	result->expr_ops = &xpu_date_ops;
	if (vclass == KVAR_CLASS__INLINE)
		result->value = kvar->i32;
	else if (vclass >= sizeof(DateADT))
		result->value = *((const DateADT *)kvar->ptr);
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device date data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_date_datum_store(kern_context *kcxt,
					 const xpu_datum_t *__arg,
					 int *p_vclass,
					 kern_variable *p_kvar)
{
	const xpu_date_t *arg = (const xpu_date_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		p_kvar->u32 = arg->value;
		*p_vclass = KVAR_CLASS__INLINE;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_date_datum_write(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_date_t *arg = (const xpu_date_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
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

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(DateADT));
	return true;
}

STATIC_FUNCTION(bool)
xpu_date_datum_comp(kern_context *kcxt,
					int *p_comp,
					const xpu_datum_t *__a,
					const xpu_datum_t *__b)
{
	const xpu_date_t *a = (const xpu_date_t *)__a;
	const xpu_date_t *b = (const xpu_date_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	if (a->value > b->value)
		*p_comp = 1;
	else if (a->value < b->value)
		*p_comp = -1;
	else
		*p_comp = 0;
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(date, true, 4, sizeof(DateADT));

/*
 * xpu_time_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_time_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   int vclass,
				   const kern_variable *kvar)
{
	xpu_time_t *result = (xpu_time_t *)__result;

	result->expr_ops = &xpu_time_ops;
	if (vclass == KVAR_CLASS__INLINE)
		result->value = kvar->i64;
	else if (vclass >= sizeof(TimeADT))
		result->value = *((const TimeADT *)kvar->ptr);
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device time data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_time_datum_store(kern_context *kcxt,
					 const xpu_datum_t *__arg,
					 int *p_vclass,
					 kern_variable *p_kvar)
{
	const xpu_time_t *arg = (const xpu_time_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		p_kvar->u64 = arg->value;
		*p_vclass = KVAR_CLASS__INLINE;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_time_datum_write(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_time_t *arg = (const xpu_time_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
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

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(TimeADT));
	return true;
}

STATIC_FUNCTION(bool)
xpu_time_datum_comp(kern_context *kcxt,
					int *p_comp,
					const xpu_datum_t *__a,
					const xpu_datum_t *__b)
{
	const xpu_time_t *a = (const xpu_time_t *)__a;
	const xpu_time_t *b = (const xpu_time_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	if (a->value > b->value)
		*p_comp = 1;
	else if (a->value < b->value)
		*p_comp = -1;
	else
		*p_comp = 0;
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(time, true, 8, sizeof(TimeADT));

/*
 * xpu_timetz_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_timetz_datum_ref(kern_context *kcxt,
					 xpu_datum_t *__result,
					 int vclass,
					 const kern_variable *kvar)
{
	xpu_timetz_t *result = (xpu_timetz_t *)__result;

	result->expr_ops = &xpu_timetz_ops;
	if (vclass >= SizeOfTimeTzADT)
		memcpy(&result->value, kvar->ptr, SizeOfTimeTzADT);
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device timetz data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_timetz_datum_store(kern_context *kcxt,
					   const xpu_datum_t *__arg,
					   int *p_vclass,
					   kern_variable *p_kvar)
{
	xpu_timetz_t *arg = (xpu_timetz_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
    else
	{
		TimeTzADT  *buf = (TimeTzADT *)kcxt_alloc(kcxt, SizeOfTimeTzADT);

		if (!buf)
			return false;
		buf->time = arg->value.time;
		buf->zone = arg->value.zone;
		p_kvar->ptr = buf;
		*p_vclass = SizeOfTimeTzADT;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_timetz_datum_write(kern_context *kcxt,
					   char *buffer,
					   const xpu_datum_t *__arg)
{
	const xpu_timetz_t *arg = (const xpu_timetz_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (buffer)
		*((TimeTzADT *)buffer) = arg->value;
	return sizeof(TimeTzADT);
}

STATIC_FUNCTION(bool)
xpu_timetz_datum_hash(kern_context *kcxt,
					  uint32_t *p_hash,
					  const xpu_datum_t *__arg)
{
	const xpu_timetz_t *arg = (const xpu_timetz_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, SizeOfTimeTzADT);
	return true;
}

STATIC_FUNCTION(bool)
xpu_timetz_datum_comp(kern_context *kcxt,
					  int *p_comp,
					  const xpu_datum_t *__a,
					const xpu_datum_t *__b)
{
	STROM_ELOG(kcxt, "timetz has no compare handler");
	return false;
}
PGSTROM_SQLTYPE_OPERATORS(timetz, false, 8, SizeOfTimeTzADT);

/*
 * xpu_timestamp_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_timestamp_datum_ref(kern_context *kcxt,
						xpu_datum_t *__result,
						int vclass,
						const kern_variable *kvar)
{
	xpu_timestamp_t *result = (xpu_timestamp_t *)__result;

	result->expr_ops = &xpu_timestamp_ops;
	if (vclass == KVAR_CLASS__INLINE)
		result->value = kvar->i64;
	else if (vclass >= sizeof(Timestamp))
		result->value = *((const Timestamp *)kvar->ptr);
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device timestamp data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_timestamp_datum_store(kern_context *kcxt,
						  const xpu_datum_t *__arg,
						  int *p_vclass,
						  kern_variable *p_kvar)
{
	const xpu_timestamp_t *arg = (const xpu_timestamp_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		p_kvar->i64 = arg->value;
		*p_vclass = KVAR_CLASS__INLINE;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_timestamp_datum_write(kern_context *kcxt,
						  char *buffer,
						  const xpu_datum_t *__arg)
{
	const xpu_timestamp_t *arg = (const xpu_timestamp_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
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

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(Timestamp));
	return true;
}

STATIC_FUNCTION(bool)
xpu_timestamp_datum_comp(kern_context *kcxt,
						 int *p_comp,
						 const xpu_datum_t *__a,
						 const xpu_datum_t *__b)
{
	const xpu_timestamp_t *a = (const xpu_timestamp_t *)__a;
	const xpu_timestamp_t *b = (const xpu_timestamp_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	if (a->value > b->value)
		*p_comp = 1;
	else if (a->value < b->value)
		*p_comp = -1;
	else
		*p_comp = 0;
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(timestamp, true, 8, sizeof(Timestamp));

/*
 * xpu_timestamptz_t device type handlers
 */
STATIC_FUNCTION(bool)
xpu_timestamptz_datum_ref(kern_context *kcxt,
						  xpu_datum_t *__result,
						  int vclass,
						  const kern_variable *kvar)
{
	xpu_timestamptz_t *result = (xpu_timestamptz_t *)__result;

	result->expr_ops = &xpu_timestamptz_ops;
	if (vclass == KVAR_CLASS__INLINE)
		result->value = kvar->i64;
	else if (vclass >= sizeof(Timestamp))
		result->value = *((const TimestampTz *)kvar->ptr);
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device timestamp data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_timestamptz_datum_store(kern_context *kcxt,
							const xpu_datum_t *__arg,
							int *p_vclass,
							kern_variable *p_kvar)
{
	const xpu_timestamptz_t *arg = (const xpu_timestamptz_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		p_kvar->i64 = arg->value;
		*p_vclass = KVAR_CLASS__INLINE;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_timestamptz_datum_write(kern_context *kcxt,
							char *buffer,
							const xpu_datum_t *__arg)
{
	const xpu_timestamptz_t *arg = (const xpu_timestamptz_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (buffer)
		*((TimestampTz *)buffer) = arg->value;
	return sizeof(TimestampTz);
}

STATIC_FUNCTION(bool)
xpu_timestamptz_datum_hash(kern_context *kcxt,
						   uint32_t *p_hash,
						   const xpu_datum_t *__arg)
{
	const xpu_timestamptz_t *arg = (const xpu_timestamptz_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(TimestampTz));
	return true;
}

STATIC_FUNCTION(bool)
xpu_timestamptz_datum_comp(kern_context *kcxt,
						   int *p_comp,
						   const xpu_datum_t *__a,
						   const xpu_datum_t *__b)
{
	const xpu_timestamptz_t *a = (const xpu_timestamptz_t *)__a;
	const xpu_timestamptz_t *b = (const xpu_timestamptz_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	if (a->value > b->value)
		*p_comp = 1;
	else if (a->value < b->value)
		*p_comp = -1;
	else
		*p_comp = 0;
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
					   int vclass,
					   const kern_variable *kvar)
{
	xpu_interval_t *result = (xpu_interval_t *)__result;

	result->expr_ops = &xpu_interval_ops;
	if (vclass >= sizeof(Interval))
	{
		result->expr_ops = &xpu_interval_ops;
		memcpy(&result->value, kvar->ptr, sizeof(Interval));
	}
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device interval data type.");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_interval_datum_store(kern_context *kcxt,
						 const xpu_datum_t *__arg,
						 int *p_vclass,
						 kern_variable *p_kvar)
{
	xpu_interval_t *arg = (xpu_interval_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_vclass = KVAR_CLASS__NULL;
	else
	{
		Interval   *buf = (Interval *)kcxt_alloc(kcxt, sizeof(Interval));

		if (!buf)
			return false;
		buf->time  = arg->value.time;
		buf->day   = arg->value.day;
		buf->month = arg->value.month;
		p_kvar->ptr = buf;
		*p_vclass = sizeof(Interval);
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_interval_datum_write(kern_context *kcxt,
						 char *buffer,
						 const xpu_datum_t *__arg)
{
	const xpu_interval_t *arg = (const xpu_interval_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
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

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(Interval));
	return true;
}

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

STATIC_FUNCTION(bool)
xpu_interval_datum_comp(kern_context *kcxt,
						 int *p_comp,
						 const xpu_datum_t *__a,
						 const xpu_datum_t *__b)
{
	const xpu_interval_t *a = (const xpu_interval_t *)__a;
	const xpu_interval_t *b = (const xpu_interval_t *)__b;
	int128_t	aval;
	int128_t	bval;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	aval = interval_cmp_value(&a->value);
	bval = interval_cmp_value(&b->value);
	*p_comp = (aval - bval);
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

STATIC_FUNCTION(int)
date2isoweek(int year, int mon, int mday)
{
	double	result;
    int		day0;
	int		day4;
	int		dayn;

	/* current day */
	dayn = date2j(year, mon, mday);

	/* fourth day of current year */
	day4 = date2j(year, 1, 4);

	/* day0 == offset to first day of week (Monday) */
	day0 = j2day(day4 - 1);

	/*
	 * We need the first week containing a Thursday, otherwise this day falls
	 * into the previous year for purposes of counting weeks
	 */
	if (dayn < day4 - day0)
	{
		day4 = date2j(year - 1, 1, 4);

		/* day0 == offset to first day of week (Monday) */
		day0 = j2day(day4 - 1);
	}
	result = (dayn - (day4 - day0)) / 7 + 1;

	/*
	 * Sometimes the last few days in a year will fall into the first week of
	 * the next year, so check for this.
	 */
	if (result >= 52)
	{
		day4 = date2j(year + 1, 1, 4);

		/* day0 == offset to first day of week (Monday) */
		day0 = j2day(day4 - 1);

		if (dayn >= day4 - day0)
			result = (dayn - (day4 - day0)) / 7 + 1;
	}
	return (int) result;
}

STATIC_FUNCTION(int)
date2isoyear(int year, int mon, int mday)
{
	double	result;
    int		day0;
    int		day4;
    int		dayn;

	/* current day */
	dayn = date2j(year, mon, mday);

	/* fourth day of current year */
	day4 = date2j(year, 1, 4);

	/* day0 == offset to first day of week (Monday) */
	day0 = j2day(day4 - 1);

	/*
	 * We need the first week containing a Thursday, otherwise this day falls
	 * into the previous year for purposes of counting weeks
	 */
	if (dayn < day4 - day0)
	{
		day4 = date2j(year - 1, 1, 4);

		/* day0 == offset to first day of week (Monday) */
		day0 = j2day(day4 - 1);

		year--;
	}
	result = (dayn - (day4 - day0)) / 7 + 1;

	/*
	 * Sometimes the last few days in a year will fall into the first week of
	 * the next year, so check for this.
	 */
	if (result >= 52)
	{
		day4 = date2j(year + 1, 1, 4);

		/* day0 == offset to first day of week (Monday) */
		day0 = j2day(day4 - 1);

		if (dayn >= day4 - day0)
			year++;
	}
	return year;
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
		return true;
	*ip = ival;
	return false;
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

STATIC_FUNCTION(void)
time2tm(TimeADT t, struct pg_tm *tm, fsec_t *fsec)
{
	tm->tm_hour = t / USECS_PER_HOUR;
	t -= tm->tm_hour * USECS_PER_HOUR;
	tm->tm_min = t / USECS_PER_MINUTE;
	t -= tm->tm_min * USECS_PER_MINUTE;
	tm->tm_sec = t / USECS_PER_SEC;
	t -= tm->tm_sec * USECS_PER_SEC;
	*fsec = t;
}

STATIC_FUNCTION(void)
timetz2tm(const TimeTzADT *time, struct pg_tm *tm, fsec_t *fsec, int *tzp)
{
	TimeOffset  trem = time->time;

	tm->tm_hour = trem / USECS_PER_HOUR;
	trem -= tm->tm_hour * USECS_PER_HOUR;
	tm->tm_min = trem / USECS_PER_MINUTE;
	trem -= tm->tm_min * USECS_PER_MINUTE;
	tm->tm_sec = trem / USECS_PER_SEC;
	*fsec = trem - tm->tm_sec * USECS_PER_SEC;

	if (tzp != NULL)
		*tzp = time->zone;
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
	KEXP_PROCESS_ARGS1(timestamp, date, datum);

	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_timestamp_ops;

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
	KEXP_PROCESS_ARGS1(timestamptz, date, datum);

	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_timestamptz_ops;

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
	KEXP_PROCESS_ARGS1(timestamp, timestamptz, datum);

	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		const pg_tz	   *tz_info = SESSION_TIMEZONE(kcxt->session);
		struct pg_tm	tm;
		fsec_t			fsec;

		result->expr_ops = &xpu_timestamp_ops;
		if (TIMESTAMP_NOT_FINITE(datum.value))
			result->value = datum.value;
		else if (!timestamp2tm(datum.value, &tm, &fsec, tz_info) ||
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
	KEXP_PROCESS_ARGS1(timestamptz, timestamp, datum);

	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		const pg_tz	   *tz_info = SESSION_TIMEZONE(kcxt->session);
		struct pg_tm	tm;
		Timestamp		ts = datum.value;
		fsec_t			fsec;

		result->expr_ops = &xpu_timestamptz_ops;
		if (TIMESTAMP_NOT_FINITE(datum.value))
			result->value = datum.value;
		else if (!timestamp2tm(datum.value, &tm, &fsec, tz_info))
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

PUBLIC_FUNCTION(bool)
pgfn_timestamp_date(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(date, timestamp, datum);

	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		struct pg_tm	tm;
		fsec_t			fsec;

		result->expr_ops = &xpu_date_ops;
		if (TIMESTAMP_IS_NOBEGIN(datum.value))
			DATE_NOBEGIN(result->value);
		else if (TIMESTAMP_IS_NOEND(datum.value))
			DATE_NOEND(result->value);
		else if (timestamp2tm(datum.value, &tm, &fsec, NULL))
		{
			result->value = date2j(tm.tm_year,
								   tm.tm_mon,
								   tm.tm_mday) - POSTGRES_EPOCH_JDATE;
		}
		else
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamptz_date(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(date, timestamptz, datum);

	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		const pg_tz	   *tz_info = SESSION_TIMEZONE(kcxt->session);
		struct pg_tm	tm;
		fsec_t			fsec;

		result->expr_ops = &xpu_date_ops;
		if (TIMESTAMP_IS_NOBEGIN(datum.value))
			DATE_NOBEGIN(result->value);
		else if (TIMESTAMP_IS_NOEND(datum.value))
			DATE_NOEND(result->value);
		else if (timestamp2tm(datum.value, &tm, &fsec, tz_info))
			result->value = date2j(tm.tm_year,
								   tm.tm_mon,
								   tm.tm_mday) - POSTGRES_EPOCH_JDATE;
		else
		{
			STROM_ELOG(kcxt, "timestamp with timezone out of range");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timetz_time(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(time, timetz, datum);

	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_time_ops;
		result->value = datum.value.time;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamp_time(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(time, timestamp, datum);

	if (XPU_DATUM_ISNULL(&datum) || TIMESTAMP_NOT_FINITE(datum.value))
		result->expr_ops = NULL;
	else
	{
		struct pg_tm	tm;
		fsec_t			fsec;

		if (timestamp2tm(datum.value, &tm, &fsec, NULL))
		{
			result->expr_ops = &xpu_time_ops;
			result->value = ((((tm.tm_hour * MINS_PER_HOUR
								+ tm.tm_min) * SECS_PER_MINUTE)
								+ tm.tm_sec) * USECS_PER_SEC) + fsec;
		}
		else
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamptz_time(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(time, timestamptz, datum);

	if (XPU_DATUM_ISNULL(&datum) || TIMESTAMP_NOT_FINITE(datum.value))
		result->expr_ops = NULL;
	else
	{
		const pg_tz	   *tz_info = SESSION_TIMEZONE(kcxt->session);
		struct pg_tm	tm;
		fsec_t			fsec;
		
		if (timestamp2tm(datum.value, &tm, &fsec, tz_info))
		{
			result->expr_ops = &xpu_time_ops;
			result->value = ((((tm.tm_hour * MINS_PER_HOUR
								+ tm.tm_min) * SECS_PER_MINUTE)
								+ tm.tm_sec) * USECS_PER_SEC) + fsec;
		}
		else
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
	}
	return true;
}

PG_SIMPLE_COMPARE_TEMPLATE(date_,date,date,)
PG_SIMPLE_COMPARE_TEMPLATE(time_,time,time,)
PG_SIMPLE_COMPARE_TEMPLATE(timestamp_,timestamp,timestamp,)
PG_SIMPLE_COMPARE_TEMPLATE(timestamptz_,timestamptz,timestamptz,)

INLINE_FUNCTION(int)
__compare_timetz(kern_context *kcxt, const TimeTzADT &lval, const TimeTzADT &rval)
{
	TimeOffset	t1 = lval.time + lval.zone * USECS_PER_SEC;
	TimeOffset	t2 = rval.time + rval.zone * USECS_PER_SEC;

	if (t1 < t2)
		return -1;
	if (t1 > t2)
		return 1;
	if (lval.zone < rval.zone)
		return -1;
	if (lval.zone > rval.zone)
		return 1;
	return 0;
}
PG_FLEXIBLE1_COMPARE_TEMPLATE(timetz, __compare_timetz)

INLINE_FUNCTION(int)
__compare_date_timestamp(kern_context *kcxt,
						 DateADT lval, Timestamp rval)
{
	Timestamp	ts;

	if (DATE_IS_NOBEGIN(lval))
		ts = DT_NOBEGIN;
	else if (DATE_IS_NOEND(lval) || lval >= (TIMESTAMP_END_JULIAN -
											 POSTGRES_EPOCH_JDATE))
		ts = DT_NOEND;
	else
		ts = lval * USECS_PER_DAY;

	if (ts < rval)
		return -1;
	if (ts > rval)
		return 1;
	return 0;
}
INLINE_FUNCTION(int)
__compare_timestamp_date(kern_context *kcxt,
						 Timestamp lval, DateADT rval)
{
	return -__compare_date_timestamp(kcxt, rval, lval);
}
PG_FLEXIBLE2_COMPARE_TEMPLATE(date, timestamp, __compare_date_timestamp)
PG_FLEXIBLE2_COMPARE_TEMPLATE(timestamp, date, __compare_timestamp_date)

INLINE_FUNCTION(int)
__compare_date_timestamptz(kern_context *kcxt, DateADT a, TimestampTz b)
{
	const pg_tz	   *tz_info = SESSION_TIMEZONE(kcxt->session);
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
INLINE_FUNCTION(int)
__compare_timestamptz_date(kern_context *kcxt, DateADT a, TimestampTz b)
{
	return __compare_date_timestamptz(kcxt, b, a);
}
PG_FLEXIBLE2_COMPARE_TEMPLATE(date, timestamptz, __compare_date_timestamptz)
PG_FLEXIBLE2_COMPARE_TEMPLATE(timestamptz, date, __compare_timestamptz_date)

INLINE_FUNCTION(int)
__compare_timestamp_timestamptz(kern_context *kcxt,
								Timestamp a,
								TimestampTz b)
{
	const pg_tz	   *tz_info = SESSION_TIMEZONE(kcxt->session);
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
INLINE_FUNCTION(int)
__compare_timestamptz_timestamp(kern_context *kcxt,
								TimestampTz a,
								Timestamp b)	
{
	return -__compare_timestamp_timestamptz(kcxt, b, a);
}
PG_FLEXIBLE2_COMPARE_TEMPLATE(timestamp, timestamptz, __compare_timestamp_timestamptz)
PG_FLEXIBLE2_COMPARE_TEMPLATE(timestamptz, timestamp, __compare_timestamptz_timestamp)

INLINE_FUNCTION(int)	
__compare_interval(kern_context *kcxt, const Interval &lval, const Interval &rval)
{
	int128_t	lcmp = interval_cmp_value(&lval);
	int128_t	rcmp = interval_cmp_value(&rval);

	return (lcmp - rcmp);
}
PG_FLEXIBLE1_COMPARE_TEMPLATE(interval, __compare_interval)

/* ----------------------------------------------------------------
 *
 * Date and Time operator functions
 *
 * ----------------------------------------------------------------
 */
PUBLIC_FUNCTION(bool)
pgfn_date_pli(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(date, date, dval, int4, ival);

	if (XPU_DATUM_ISNULL(&dval) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_date_ops;
		if (DATE_NOT_FINITE(dval.value))
			result->value = dval.value;		/* can't change infinity */
		else
			result->value = dval.value + ival.value;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_date_mii(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(date, date, dval, int4, ival);

	if (XPU_DATUM_ISNULL(&dval) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_date_ops;
		if (DATE_NOT_FINITE(dval.value))
			result->value = dval.value;		/* can't change infinity */
		else
			result->value = dval.value - ival.value;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_date_mi(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(int4, date, dval1, date, dval2);

	if (XPU_DATUM_ISNULL(&dval1) || XPU_DATUM_ISNULL(&dval2))
		result->expr_ops = NULL;
	else if (!DATE_NOT_FINITE(dval1.value) &&
			 !DATE_NOT_FINITE(dval2.value))
	{
		result->expr_ops = &xpu_int4_ops;
		result->value = (int32_t)(dval1.value - dval2.value);
	}
	else
	{
		STROM_ELOG(kcxt, "cannot subtract infinite dates");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
__pg_datetime_pl(kern_context *kcxt,
				 xpu_timestamp_t *result,
				 const xpu_date_t *dval,
				 const xpu_time_t *tval)
{
	if (XPU_DATUM_ISNULL(dval) || XPU_DATUM_ISNULL(tval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_timestamp_ops;

		/* transform date -> timestamp */
		if (DATE_IS_NOBEGIN(dval->value))
			TIMESTAMP_NOBEGIN(result->value);
		else if (DATE_IS_NOEND(dval->value))
			TIMESTAMP_NOEND(result->value);
		else if (dval->value < (TIMESTAMP_END_JULIAN - POSTGRES_EPOCH_JDATE))
		{
			/* timestamp += time */
			result->value = (dval->value * USECS_PER_DAY + tval->value);
		}
		else
		{
			STROM_ELOG(kcxt, "date out of range for timestamp");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_datetime_pl(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timestamp, date, dval, time, tval);
	return __pg_datetime_pl(kcxt, result, &dval, &tval);
}

PUBLIC_FUNCTION(bool)
pgfn_integer_pl_date(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(date, int4, ival, date, dval);

	if (XPU_DATUM_ISNULL(&ival) || XPU_DATUM_ISNULL(&dval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_date_ops;
		if (DATE_NOT_FINITE(dval.value))
			result->value = dval.value;		/* can't change infinity */
		else
			result->value = dval.value + ival.value;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timedate_pl(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timestamp, time, tval, date, dval);
	return __pg_datetime_pl(kcxt, result, &dval, &tval);
}

PUBLIC_FUNCTION(bool)
pgfn_time_mi_time(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(interval, time, tval1, time, tval2);
	
	if (XPU_DATUM_ISNULL(&tval1) || XPU_DATUM_ISNULL(&tval2))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_interval_ops;
		result->value.time  = tval1.value - tval2.value;
		result->value.day   = 0;
		result->value.month = 0;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamp_mi(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(interval, timestamp, tval1, timestamp, tval2);

	if (XPU_DATUM_ISNULL(&tval1) || XPU_DATUM_ISNULL(&tval2))
        result->expr_ops = NULL;
	else if (TIMESTAMP_NOT_FINITE(tval1.value) ||
			 TIMESTAMP_NOT_FINITE(tval2.value))
	{
		STROM_ELOG(kcxt, "cannot subtract infinite timestamps");
		return false;
	}
	else
	{
		TimeOffset		t;

		result->expr_ops = &xpu_interval_ops;
		result->value.month = 0;
		result->value.day   = 0;
		/*----------
		 *  This is wrong, but removing it breaks a lot of regression tests.
		 *  For example:
		 *
		 *  test=> SET timezone = 'EST5EDT';
		 *  test=> SELECT
		 *  test-> ('2005-10-30 13:22:00-05'::timestamptz -
		 *  test(>  '2005-10-29 13:22:00-04'::timestamptz);
		 *  ?column?
		 *  ----------------
		 *   1 day 01:00:00
		 *   (1 row)
		 *
		 *  so adding that to the first timestamp gets:
		 *
		 *   test=> SELECT
		 *   test-> ('2005-10-29 13:22:00-04'::timestamptz +
		 *   test(> ('2005-10-30 13:22:00-05'::timestamptz -
		 *   test(>  '2005-10-29 13:22:00-04'::timestamptz)) at time zone 'EST';
		 *      timezone
		 *  --------------------
		 *  2005-10-30 14:22:00
		 *  (1 row)
		 *----------
		 *
		 * See, the original at interval_justify_hours()
		 */
		t = tval1.value - tval2.value;
		if (t >= USECS_PER_DAY)
		{
			TimeOffset	d = t / USECS_PER_DAY;
			result->value.day += d;
			t -= d * USECS_PER_DAY;
		}
		result->value.time = t;

		if (result->value.day > 0 && result->value.time < 0)
		{
			result->value.time += USECS_PER_DAY;
			result->value.day--;
		}
		else if (result->value.day < 0 && result->value.time > 0)
		{
			result->value.time -= USECS_PER_DAY;
			result->value.day++;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_time_pl_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(time, time, tval, interval, ival);

	if (XPU_DATUM_ISNULL(&tval) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else
	{
		TimeADT		t;

		t = tval.value + ival.value.time;
		t -= (t / USECS_PER_DAY) * USECS_PER_DAY;
		if (t < 0)
			t += USECS_PER_DAY;
		result->expr_ops = &xpu_time_ops;
		result->value = t;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_time_mi_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(time, time, tval, interval, ival);

	if (XPU_DATUM_ISNULL(&tval) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else
	{
		TimeADT		t;

		t = tval.value - ival.value.time;
		t -= (t / USECS_PER_DAY) * USECS_PER_DAY;
		if (t < 0)
			t += USECS_PER_DAY;
		result->expr_ops = &xpu_time_ops;
		result->value = t;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timetz_pl_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timetz, timetz, tval, interval, ival);

	if (XPU_DATUM_ISNULL(&tval) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else
	{
		TimeADT		t;

		t = tval.value.time + ival.value.time;
		t -= (t / USECS_PER_DAY) * USECS_PER_DAY;
		if (t < 0)
			t += USECS_PER_DAY;
		result->expr_ops = &xpu_time_ops;
		result->value.time = t;
		result->value.zone = tval.value.zone;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timetz_mi_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timetz, timetz, tval, interval, ival);

	if (XPU_DATUM_ISNULL(&tval) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else
	{
		TimeADT		t;

		t = tval.value.time - ival.value.time;
		t -= (t / USECS_PER_DAY) * USECS_PER_DAY;
		if (t < 0)
			t += USECS_PER_DAY;
		result->expr_ops = &xpu_time_ops;
		result->value.time = t;
		result->value.zone = tval.value.zone;
	}
	return true;
}

STATIC_FUNCTION(bool)
__pg_timestamp_pl_interval(kern_context *kcxt,
						   xpu_timestamp_t *result,
						   const xpu_timestamp_t *tval,
						   const xpu_interval_t *ival)
{
	if (XPU_DATUM_ISNULL(tval) || XPU_DATUM_ISNULL(ival))
		result->expr_ops = NULL;
	else if (TIMESTAMP_NOT_FINITE(tval->value))
	{
		result->expr_ops = &xpu_timestamp_ops;
		result->value = tval->value;
	}
	else
	{
		Timestamp	ts = tval->value;

		if (ival->value.month != 0)
		{
			struct pg_tm	tm;
			fsec_t			fsec;

			if (!timestamp2tm(ts, &tm, &fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
			tm.tm_mon += ival->value.month;
			if (tm.tm_mon > MONTHS_PER_YEAR)
			{
				tm.tm_year += (tm.tm_mon - 1) / MONTHS_PER_YEAR;
				tm.tm_mon  = (tm.tm_mon - 1) % MONTHS_PER_YEAR + 1;
			}
			else if (tm.tm_mon < 1)
			{
				tm.tm_year += tm.tm_mon / MONTHS_PER_YEAR - 1;
				tm.tm_mon  = tm.tm_mon % MONTHS_PER_YEAR + MONTHS_PER_YEAR;
			}
			/* adjust for end of month boundary problems... */
			if (tm.tm_mday > day_tab[isleap(tm.tm_year)][tm.tm_mon - 1])
				tm.tm_mday = day_tab[isleap(tm.tm_year)][tm.tm_mon - 1];
			if (!tm2timestamp(&ts, &tm, fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
		}

		if (ival->value.day != 0)
		{
			struct pg_tm	tm;
			fsec_t			fsec;
			int				julian;

			if (!timestamp2tm(ts, &tm, &fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
			/* add days by converting to and from Julian */
            julian = date2j(tm.tm_year, tm.tm_mon, tm.tm_mday) + ival->value.day;
			j2date(julian, &tm.tm_year, &tm.tm_mon, &tm.tm_mday);
			if (!tm2timestamp(&ts, &tm, fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
		}
		ts += ival->value.time;
		if (!IS_VALID_TIMESTAMP(ts))
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
		result->expr_ops = &xpu_timestamp_ops;
		result->value    = ts;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamp_pl_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timestamp, timestamp, tval, interval, ival);
	return __pg_timestamp_pl_interval(kcxt, result, &tval, &ival);
}

PUBLIC_FUNCTION(bool)
pgfn_timestamp_mi_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timestamp, timestamp, tval, interval, ival);
	ival.value.month = -ival.value.month;
	ival.value.day   = -ival.value.day;
	ival.value.time  = -ival.value.time;
	return __pg_timestamp_pl_interval(kcxt, result, &tval, &ival);
}

STATIC_FUNCTION(bool)
__pg_timestamptz_pl_interval(kern_context *kcxt,
							 xpu_timestamptz_t *result,
							 const xpu_timestamptz_t *tval,
							 const xpu_interval_t *ival)
{
	if (XPU_DATUM_ISNULL(tval) || XPU_DATUM_ISNULL(ival))
		result->expr_ops = NULL;
	else if (TIMESTAMP_NOT_FINITE(tval->value))
	{
		result->expr_ops = &xpu_timestamp_ops;
		result->value = tval->value;
	}
	else
	{
		Timestamp	ts = tval->value;
		const pg_tz *tz_info = SESSION_TIMEZONE(kcxt->session);
		int			tz;

		if (ival->value.month != 0)
		{
			struct pg_tm tm;
			fsec_t	fsec;

			if (!timestamp2tm(ts, &tm, &fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
			tm.tm_mon += ival->value.month;
			if (tm.tm_mon > MONTHS_PER_YEAR)
			{
				tm.tm_year += (tm.tm_mon - 1) / MONTHS_PER_YEAR;
				tm.tm_mon  = (tm.tm_mon - 1) % MONTHS_PER_YEAR + 1;
			}
			else if (tm.tm_mon < 1)
			{
				tm.tm_year += tm.tm_mon / MONTHS_PER_YEAR - 1;
				tm.tm_mon  = tm.tm_mon % MONTHS_PER_YEAR + MONTHS_PER_YEAR;
			}
			/* adjust for end of month boundary problems... */
			if (tm.tm_mday > day_tab[isleap(tm.tm_year)][tm.tm_mon - 1])
				tm.tm_mday = day_tab[isleap(tm.tm_year)][tm.tm_mon - 1];

			tz = DetermineTimeZoneOffset(&tm, tz_info);
			if (!tm2timestamp(&ts, &tm, fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
			ts += tz * USECS_PER_SEC;
		}

		if (ival->value.day != 0)
		{
			struct pg_tm tm;
			fsec_t	fsec;
			int		julian;

			if (!timestamp2tm(ts, &tm, &fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
            /* add days by converting to and from Julian */
            julian = date2j(tm.tm_year, tm.tm_mon, tm.tm_mday) + ival->value.day;
            j2date(julian, &tm.tm_year, &tm.tm_mon, &tm.tm_mday);

            tz = DetermineTimeZoneOffset(&tm, tz_info);
			if (!tm2timestamp(&ts, &tm, fsec, NULL))
			{
				STROM_ELOG(kcxt, "timestamp out of range");
				return false;
			}
			ts += tz * USECS_PER_SEC;
		}
		ts += ival->value.time;
		if (!IS_VALID_TIMESTAMP(ts))
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
		result->expr_ops = &xpu_timestamp_ops;
		result->value    = ts;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_timestamptz_pl_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timestamptz, timestamptz, tval, interval, ival);
	return __pg_timestamptz_pl_interval(kcxt, result, &tval, &ival);
}

PUBLIC_FUNCTION(bool)
pgfn_timestamptz_mi_interval(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(timestamptz, timestamptz, tval, interval, ival);
	ival.value.month = -ival.value.month;
	ival.value.day   = -ival.value.day;
	ival.value.time  = -ival.value.time;
	return __pg_timestamptz_pl_interval(kcxt, result, &tval, &ival);
}

PUBLIC_FUNCTION(bool)
pgfn_interval_um(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(interval, interval, ival);

	if (XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else
	{
		if (ival.value.month   == -1 ||
			ival.value.day     == -1 ||
			result->value.time == -1L)
		{
			STROM_ELOG(kcxt, "interval out of range");
			return false;
		}
		result->value.month = -ival.value.month;
		result->value.day   = -ival.value.day;
		result->value.time  = -ival.value.time;
		result->expr_ops    = &xpu_interval_ops;
	}
	return true;
}

#define SAMESIGN(a,b)		(((a) < 0) == ((b) < 0))

PUBLIC_FUNCTION(bool)
pgfn_interval_pl(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(interval, interval, ival1, interval, ival2);

	if (XPU_DATUM_ISNULL(&ival1) || XPU_DATUM_ISNULL(&ival2))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_interval_ops;
		result->value.month = ival1.value.month + ival2.value.month;
		result->value.day   = ival1.value.day   + ival2.value.day;
		result->value.time  = ival1.value.time  + ival2.value.time;
		if ((SAMESIGN(ival1.value.month, ival2.value.month) &&
			 !SAMESIGN(result->value.month, ival1.value.month)) ||
			(SAMESIGN(ival1.value.day, ival2.value.day) &&
			 !SAMESIGN(result->value.day, ival1.value.day)) ||
			(SAMESIGN(ival1.value.time, ival2.value.time) &&
			 !SAMESIGN(result->value.time, ival1.value.time)))
		{
			STROM_ELOG(kcxt, "interval out of range");
			return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_interval_mi(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(interval, interval, ival1, interval, ival2);

	if (XPU_DATUM_ISNULL(&ival1) || XPU_DATUM_ISNULL(&ival2))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_interval_ops;
		result->value.month = ival1.value.month - ival2.value.month;
		result->value.day   = ival1.value.day   - ival2.value.day;
		result->value.time  = ival1.value.time  - ival2.value.time;
		if ((SAMESIGN(ival1.value.month, ival2.value.month) &&
			 !SAMESIGN(result->value.month, ival1.value.month)) ||
			(SAMESIGN(ival1.value.day, ival2.value.day) &&
			 !SAMESIGN(result->value.day, ival1.value.day)) ||
			(SAMESIGN(ival1.value.time, ival2.value.time) &&
			 !SAMESIGN(result->value.time, ival1.value.time)))
		{
			STROM_ELOG(kcxt, "interval out of range");
			return false;
		}
	}
}

#undef SAMESIGN

/* ----------------------------------------------------------------
 *
 * OVERLAP() function
 *
 * ----------------------------------------------------------------
 */
#define OVERLAPS_TEMPLATE(TYPE,COMP_FN)								\
	STATIC_FUNCTION(bool)											\
	__pg_overlaps_##TYPE(kern_context *kcxt,						\
						 xpu_bool_t *result,						\
						 const xpu_##TYPE##_t *ts1,					\
						 const xpu_##TYPE##_t *te1,					\
						 const xpu_##TYPE##_t *ts2,					\
						 const xpu_##TYPE##_t *te2)					\
	{																\
		int		comp;												\
																	\
		if (XPU_DATUM_ISNULL(ts1))									\
		{															\
			if (XPU_DATUM_ISNULL(te1))								\
			{														\
				result->expr_ops = NULL;							\
				return true;										\
			}														\
			ts1 = te1;												\
		}															\
		else if (!XPU_DATUM_ISNULL(te1))							\
		{															\
			if (COMP_FN(kcxt, ts1->value, te1->value) > 0)			\
			{														\
				const xpu_##TYPE##_t *tmp = ts1;					\
				ts1 = te1;											\
				te1 = tmp;											\
			}														\
		}															\
																	\
		if (XPU_DATUM_ISNULL(ts2))									\
		{															\
			if (XPU_DATUM_ISNULL(te2))								\
			{														\
				result->expr_ops = NULL;							\
				return true;										\
			}														\
			ts2 = te2;												\
		}															\
		else if (!XPU_DATUM_ISNULL(te2))							\
		{															\
			if (COMP_FN(kcxt, ts2->value, te2->value) > 0)			\
			{														\
				const xpu_##TYPE##_t *tmp = ts2;					\
				ts2 = te2;											\
				te2 = tmp;											\
			}														\
		}															\
																	\
		comp = COMP_FN(kcxt, ts1->value, ts2->value);				\
		if (comp > 0)												\
		{															\
			if (XPU_DATUM_ISNULL(te2))								\
			{														\
				result->expr_ops = NULL;							\
			}														\
			else if (COMP_FN(kcxt, ts1->value, te2->value) < 0)		\
			{														\
				result->expr_ops = &xpu_bool_ops;					\
				result->value = true;								\
			}														\
			else if (XPU_DATUM_ISNULL(te1))							\
			{														\
				result->expr_ops = NULL;							\
			}														\
			else													\
			{														\
				result->expr_ops = &xpu_bool_ops;					\
				result->value = false;								\
			}														\
		}															\
		else if (comp < 0)											\
		{															\
			if (XPU_DATUM_ISNULL(te1))								\
			{														\
				result->expr_ops = NULL;							\
			}														\
			else if (COMP_FN(kcxt, ts2->value, te1->value) < 0)		\
			{														\
				result->expr_ops = &xpu_bool_ops;					\
				result->value = true;								\
			}														\
			else if (XPU_DATUM_ISNULL(te2))							\
			{														\
				result->expr_ops = NULL;							\
			}														\
			else													\
			{														\
				result->expr_ops = &xpu_bool_ops;					\
				result->value = false;								\
			}														\
		}															\
		else														\
		{															\
			if (XPU_DATUM_ISNULL(te1) || XPU_DATUM_ISNULL(te2))		\
			{														\
				result->expr_ops = NULL;							\
			}														\
			else													\
			{														\
				result->expr_ops = &xpu_bool_ops;					\
				result->value = true;								\
			}														\
		}															\
		return true;												\
	}

#define __SIMPLE_COMP(a,b,c)		((b) - (c))
OVERLAPS_TEMPLATE(time, __SIMPLE_COMP)
OVERLAPS_TEMPLATE(timetz, __compare_timetz)
OVERLAPS_TEMPLATE(timestamp, __SIMPLE_COMP)
OVERLAPS_TEMPLATE(timestamptz, __SIMPLE_COMP)

PUBLIC_FUNCTION(bool)
pgfn_overlaps_time(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS4(bool,
					   time, arg1,
					   time, arg2,
					   time, arg3,
					   time, arg4);
	return __pg_overlaps_time(kcxt, result, &arg1, &arg2, &arg3, &arg4);
}

PUBLIC_FUNCTION(bool)
pgfn_overlaps_timetz(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS4(bool,
					   timetz, arg1,
					   timetz, arg2,
					   timetz, arg3,
					   timetz, arg4);
	return __pg_overlaps_timetz(kcxt, result, &arg1, &arg2, &arg3, &arg4);
}

PUBLIC_FUNCTION(bool)
pgfn_overlaps_timestamp(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS4(bool,
					   timestamp, arg1,
					   timestamp, arg2,
					   timestamp, arg3,
					   timestamp, arg4);
	return __pg_overlaps_timestamp(kcxt, result, &arg1, &arg2, &arg3, &arg4);
}

PUBLIC_FUNCTION(bool)
pgfn_overlaps_timestamptz(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS4(bool,
					   timestamptz, arg1,
					   timestamptz, arg2,
					   timestamptz, arg3,
					   timestamptz, arg4);
	return __pg_overlaps_timestamptz(kcxt, result, &arg1, &arg2, &arg3, &arg4);
}

/* ----------------------------------------------------------------
 *
 * EXTRACT(text,date/time) function
 *
 * ----------------------------------------------------------------
 */
/* 'type' field definition */
#define RESERV			0
#define MONTH			1
#define YEAR			2
#define DAY				3
#define JULIAN			4
#define TZ				5	/* fixed-offset timezone abbreviation */
#define DTZ				6	/* fixed-offset timezone abbrev, DST */
#define DYNTZ			7	/* dynamic timezone abbreviation */
#define IGNORE_DTF		8
#define AMPM			9
#define HOUR			10
#define MINUTE			11
#define SECOND			12
#define MILLISECOND		13
#define MICROSECOND		14
#define DOY				15
#define DOW				16
#define UNITS			17
#define ADBC			18
/* these are only for relative dates */
#define AGO				19
#define ABS_BEFORE		20
#define ABS_AFTER		21
/* generic fields to help with parsing */
#define ISODATE			22
#define ISOTIME			23
/* these are only for parsing intervals */
#define WEEK			24
#define DECADE			25
#define CENTURY			26
#define MILLENNIUM		27
/* hack for parsing two-word timezone specs "MET DST" etc */
#define DTZMOD			28		/* "DST" as a separate word */
#define UNKNOWN_FIELD	31		/* reserved for unrecognized string values */

/* 'token' field definition */
#define DTK_NUMBER		0
#define DTK_STRING		1
#define DTK_DATE		2
#define DTK_TIME		3
#define DTK_TZ			4
#define DTK_AGO			5
#define DTK_SPECIAL		6
#define DTK_INVALID		7
#define DTK_CURRENT		8
#define DTK_EARLY		9
#define DTK_LATE		10
#define DTK_EPOCH		11
#define DTK_NOW			12
#define DTK_YESTERDAY	13
#define DTK_TODAY		14
#define DTK_TOMORROW	15
#define DTK_ZULU		16
#define DTK_DELTA		17
#define DTK_SECOND		18
#define DTK_MINUTE		19
#define DTK_HOUR		20
#define DTK_DAY			21
#define DTK_WEEK		22
#define DTK_MONTH		23
#define DTK_QUARTER		24
#define DTK_YEAR		25
#define DTK_DECADE		26
#define DTK_CENTURY		27
#define DTK_MILLENNIUM	28
#define DTK_MILLISEC	29
#define DTK_MICROSEC	30
#define DTK_JULIAN		31
#define DTK_DOW			32
#define DTK_DOY			33
#define DTK_TZ_HOUR		34
#define DTK_TZ_MINUTE	35
#define DTK_ISOYEAR		36
#define DTK_ISODOW		37

/* keep this struct small; it gets used a lot */
typedef struct
{
	const char *token;		/* always NUL-terminated */
	char		type;		/* see field type codes above */
	int			value;		/* meaning depends on type */
} datetkn;
#define TOKMAXLEN		10

static __device__ const datetkn deltatktbl[] = {
	/* token, type, value */
	{"@",		IGNORE_DTF, 0},		/* postgres relative prefix */
	{"ago",		AGO, 0},			/* "ago" indicates negative time offset */
	{"c",		UNITS, DTK_CENTURY},	/* "century" relative */
	{"cent",	UNITS, DTK_CENTURY},	/* "century" relative */
	{"centuries", UNITS, DTK_CENTURY},	/* "centuries" relative */
	{"century",	UNITS, DTK_CENTURY},	/* "century" relative */
	{"d",		UNITS, DTK_DAY},		/* "day" relative */
	{"day",		UNITS, DTK_DAY},		/* "day" relative */
	{"days",	UNITS, DTK_DAY},		/* "days" relative */
	{"dec",		UNITS, DTK_DECADE},		/* "decade" relative */
	{"decade",	UNITS, DTK_DECADE},		/* "decade" relative */
	{"decades",	UNITS, DTK_DECADE},		/* "decades" relative */
	{"decs",	UNITS, DTK_DECADE},		/* "decades" relative */
	{"h",		UNITS, DTK_HOUR},		/* "hour" relative */
	{"hour",	UNITS, DTK_HOUR},		/* "hour" relative */
	{"hours",	UNITS, DTK_HOUR},		/* "hours" relative */
	{"hr",		UNITS, DTK_HOUR},		/* "hour" relative */
	{"hrs",		UNITS, DTK_HOUR},		/* "hours" relative */
	{"invalid",	RESERV, DTK_INVALID},		/* reserved for invalid time */
	{"m",		UNITS, DTK_MINUTE},			/* "minute" relative */
	{"microsecon", UNITS, DTK_MICROSEC},	/* "microsecond" relative */
	{"mil",		UNITS, DTK_MILLENNIUM},		/* "millennium" relative */
	{"millennia", UNITS, DTK_MILLENNIUM},	/* "millennia" relative */
	{"millennium", UNITS, DTK_MILLENNIUM},	/* "millennium" relative */
	{"millisecon", UNITS, DTK_MILLISEC},/* relative */
	{"mils",	UNITS, DTK_MILLENNIUM},	/* "millennia" relative */
	{"min",		UNITS, DTK_MINUTE},		/* "minute" relative */
	{"mins",	UNITS, DTK_MINUTE},		/* "minutes" relative */
	{"minute",	UNITS, DTK_MINUTE},		/* "minute" relative */
	{"minutes",	UNITS, DTK_MINUTE},		/* "minutes" relative */
	{"mon",		UNITS, DTK_MONTH},		/* "months" relative */
	{"mons",	UNITS, DTK_MONTH},		/* "months" relative */
	{"month",	UNITS, DTK_MONTH},		/* "month" relative */
	{"months",	UNITS, DTK_MONTH},
	{"ms",		UNITS, DTK_MILLISEC},
	{"msec",	UNITS, DTK_MILLISEC},
	{"msecond",	UNITS, DTK_MILLISEC},
	{"mseconds", UNITS, DTK_MILLISEC},
	{"msecs",	UNITS, DTK_MILLISEC},
	{"qtr",		UNITS, DTK_QUARTER},	/* "quarter" relative */
	{"quarter",	UNITS, DTK_QUARTER},	/* "quarter" relative */
	{"s",		UNITS, DTK_SECOND},
	{"sec",		UNITS, DTK_SECOND},
	{"second",	UNITS, DTK_SECOND},
	{"seconds", UNITS, DTK_SECOND},
	{"secs",	UNITS, DTK_SECOND},
	{"timezone", UNITS, DTK_TZ},		/* "timezone" time offset */
	{"timezone_h", UNITS, DTK_TZ_HOUR},	/* timezone hour units */
	{"timezone_m", UNITS, DTK_TZ_MINUTE}, /* timezone minutes units */
	{"undefined", RESERV, DTK_INVALID},	/* pre-v6.1 invalid time */
	{"us",		UNITS, DTK_MICROSEC},	/* "microsecond" relative */
	{"usec",	UNITS, DTK_MICROSEC},	/* "microsecond" relative */
	{"usecond",	UNITS, DTK_MICROSEC},	/* "microsecond" relative */
	{"useconds", UNITS, DTK_MICROSEC},	/* "microseconds" relative */
	{"usecs",	UNITS, DTK_MICROSEC},	/* "microseconds" relative */
	{"w",		UNITS, DTK_WEEK},		/* "week" relative */
	{"week",	UNITS, DTK_WEEK},		/* "week" relative */
	{"weeks",	UNITS, DTK_WEEK},		/* "weeks" relative */
	{"y",		UNITS, DTK_YEAR},		/* "year" relative */
	{"year",	UNITS, DTK_YEAR},		/* "year" relative */
	{"years",	UNITS, DTK_YEAR},		/* "years" relative */
	{"yr",		UNITS, DTK_YEAR},		/* "year" relative */
	{"yrs",		UNITS, DTK_YEAR},		/* "years" relative */
};

/* misc definitions */
#define AM      0
#define PM      1
#define HR24    2

#define AD      0
#define BC      1

static __device__ const datetkn datetktbl[] = {
    /* token, type, value */
	{"-infinity",	RESERV, DTK_EARLY},
	{"ad",			ADBC, AD},           /* "ad" for years > 0 */
	{"allballs",	RESERV, DTK_ZULU},     /* 00:00:00 */
	{"am",			AMPM, AM},
	{"apr",			MONTH, 4},
	{"april",		MONTH, 4},
	{"at",			IGNORE_DTF, 0},      /* "at" (throwaway) */
	{"aug",			MONTH, 8},
	{"august",		MONTH, 8},
	{"bc",			ADBC, BC},           /* "bc" for years <= 0 */
	{"current",		RESERV, DTK_CURRENT},    /* "current" is always now */
	{"d",			UNITS, DTK_DAY},      /* "day of month" for ISO input */
	{"dec",			MONTH, 12},
	{"december",	MONTH, 12},
	{"dow",			UNITS, DTK_DOW},    /* day of week */
	{"doy",			UNITS, DTK_DOY},    /* day of year */
	{"dst",			DTZMOD, SECS_PER_HOUR},
	{"epoch",		RESERV, DTK_EPOCH}, /* "epoch" reserved for system epoch time */
	{"feb",			MONTH, 2},
	{"february",	MONTH, 2},
	{"fri",			DOW, 5},
	{"friday",		DOW, 5},
	{"h",			UNITS, DTK_HOUR},	/* "hour" */
	{"infinity",	RESERV, DTK_LATE},
	{"invalid",		RESERV, DTK_INVALID},
	{"isodow",		UNITS, DTK_ISODOW},
	{"isoyear",		UNITS, DTK_ISOYEAR},
	{"j",			UNITS, DTK_JULIAN},
	{"jan",			MONTH, 1},
	{"january",		MONTH, 1},
	{"jd",			UNITS, DTK_JULIAN},
	{"jul",			MONTH, 7},
	{"julian",		UNITS, DTK_JULIAN},
	{"july",		MONTH, 7},
	{"jun",			MONTH, 6},
	{"june",		MONTH, 6},
	{"m",			UNITS, DTK_MONTH},    /* "month" for ISO input */
	{"mar",			MONTH, 3},
	{"march",		MONTH, 3},
	{"may",			MONTH, 5},
	{"mm",			UNITS, DTK_MINUTE},  /* "minute" for ISO input */
	{"mon",			DOW, 1},
	{"monday",		DOW, 1},
	{"nov",			MONTH, 11},
	{"november",	MONTH, 11},
	{"now",			RESERV, DTK_NOW},     /* current transaction time */
	{"oct",			MONTH, 10},
	{"october",		MONTH, 10},
	{"on",			IGNORE_DTF, 0},      /* "on" (throwaway) */
	{"pm",			AMPM, PM},
	{"s",			UNITS, DTK_SECOND},   /* "seconds" for ISO input */
	{"sat",			DOW, 6},
	{"saturday",	DOW, 6},
	{"sep",			MONTH, 9},
	{"sept",		MONTH, 9},
	{"september",	MONTH, 9},
	{"sun",			DOW, 0},
	{"sunday",		DOW, 0},
	{"t",			ISOTIME, DTK_TIME},   /* Filler for ISO time fields */
	{"thu",			DOW, 4},
	{"thur",		DOW, 4},
	{"thurs",		DOW, 4},
	{"thursday",	DOW, 4},
	{"today",		RESERV, DTK_TODAY}, /* midnight */
	{"tomorrow",	RESERV, DTK_TOMORROW},   /* tomorrow midnight */
	{"tue",			DOW, 2},
	{"tues",		DOW, 2},
	{"tuesday",		DOW, 2},
	{"undefined",	RESERV, DTK_INVALID}, /* pre-v6.1 invalid time */
	{"wed",			DOW, 3},
	{"wednesday",	DOW, 3},
	{"weds",		DOW, 3},
	{"y",			UNITS, DTK_YEAR},     /* "year" for ISO input */
	{"yesterday",	RESERV, DTK_YESTERDAY}  /* yesterday midnight */
};

/*
 * datebsearch - search the keyword array
 */
STATIC_FUNCTION(const datetkn *)
datebsearch(const char *key, const datetkn *datetkntbl, int nitems)
{
	const datetkn  *base = datetkntbl;
	const datetkn  *last = datetkntbl + nitems - 1;
	const datetkn  *position;
	int				comp;

	while (last >= base)
	{
		position = base + ((last - base) >> 1);
		comp = (int) key[0] - (int) position->token[0];
		if (comp == 0)
		{
			comp = __strcmp(key, position->token);
			if (comp == 0)
				return position;
		}
		if (comp < 0)
			last = position - 1;
		else
			base = position + 1;
	}
	return NULL;
}

STATIC_FUNCTION(bool)
extract_decode_unit(kern_context *kcxt,
					const xpu_text_t *key,
					int *p_type, int *p_value)
{
	const datetkn *dtoken;
	char	namebuf[24];
	int		i;

	assert(key->length >= 0);
	for (i=0; i < key->length; i++)
	{
		char	ch = key->value[i];

		if (ch >= 'A' && ch <= 'Z')
			ch += ('a' - 'A');
		namebuf[i] = ch;
	}
	namebuf[i] = '\0';

	/* DecodeUnits() / DecodeSpecial() */
	dtoken = datebsearch(namebuf, deltatktbl, lengthof(deltatktbl));
    if (!dtoken)
	{
        dtoken = datebsearch(namebuf, datetktbl, lengthof(datetktbl));
		if (!dtoken)
		{
			STROM_ELOG(kcxt, "not a recognized unit name");
			return false;
		}
	}
	*p_type  = dtoken->type;
	*p_value = dtoken->value;
	return true;
}

STATIC_FUNCTION(bool)
NonFiniteTimestampTzPart(kern_context *kcxt,
						 xpu_numeric_t *result,
						 int type, int unit, bool is_negative)
{
	if (type != UNITS && type != RESERV)
	{
		STROM_ELOG(kcxt, "not recognized timestamp units");
		return false;
	}

	memset(result, 0, sizeof(xpu_numeric_t));
	switch (unit)
	{
        /* Oscillating units */
        case DTK_MICROSEC:
        case DTK_MILLISEC:
        case DTK_SECOND:
        case DTK_MINUTE:
        case DTK_HOUR:
        case DTK_DAY:
        case DTK_MONTH:
        case DTK_QUARTER:
        case DTK_WEEK:
        case DTK_DOW:
        case DTK_ISODOW:
        case DTK_DOY:
        case DTK_TZ:
        case DTK_TZ_MINUTE:
        case DTK_TZ_HOUR:
			break;
			/* Monotonically-increasing units */
		case DTK_YEAR:
        case DTK_DECADE:
        case DTK_CENTURY:
        case DTK_MILLENNIUM:
        case DTK_JULIAN:
        case DTK_ISOYEAR:
        case DTK_EPOCH:
			result->kind = (is_negative
							? XPU_NUMERIC_KIND__NEG_INF
							: XPU_NUMERIC_KIND__POS_INF);
			break;
		default:
			STROM_ELOG(kcxt, "unsupported timestamp unit");
			return false;
	}
	result->expr_ops = &xpu_numeric_ops;
	return true;
}

STATIC_FUNCTION(bool)
__pg_extract_timestamp_common(kern_context *kcxt,
							  xpu_numeric_t *result,
							  int type, int value,
							  Timestamp ts,
							  const pg_tz *tz_info)
{
	if (TIMESTAMP_NOT_FINITE(ts))
		return NonFiniteTimestampTzPart(kcxt, result, type, value,
										TIMESTAMP_IS_NOBEGIN(ts));
	if (type == UNITS)
	{
		struct pg_tm tm;
		fsec_t		fsec;

		if (!timestamp2tm(ts, &tm, &fsec, tz_info))
		{
			STROM_ELOG(kcxt, "timestamp out of range");
			return false;
		}
		memset(result, 0, sizeof(xpu_numeric_t));

		switch (value)
		{
			case DTK_MICROSEC:
				result->value = tm.tm_sec * 1000000 + fsec;
				result->weight = 6;
				break;
			case DTK_MILLISEC:
				result->value = tm.tm_sec * 1000 + fsec / 1000.0;
				result->weight = 3;
				break;
			case DTK_SECOND:
				result->value = tm.tm_sec;
				break;
			case DTK_MINUTE:
				result->value = tm.tm_min;
				break;
			case DTK_HOUR:
				result->value = tm.tm_hour;
				break;
			case DTK_DAY:
				result->value = tm.tm_mday;
				break;
			case DTK_MONTH:
				result->value = tm.tm_mon;
				break;
			case DTK_QUARTER:
				result->value = (tm.tm_mon - 1) / 3 + 1;
				break;
			case DTK_WEEK:
				result->value = date2isoweek(tm.tm_year,
											 tm.tm_mon,
											 tm.tm_mday);
				break;
			case DTK_YEAR:
				/* there is no year 0, just 1 BC and 1 AD */
				if (tm.tm_year > 0)
					result->value = tm.tm_year;
				else
					result->value = tm.tm_year - 1;
				break;
			case DTK_DECADE:
				if (tm.tm_year >= 0)
					result->value = tm.tm_year / 10;
				else
					result->value = -((8 - (tm.tm_year - 1)) / 10);
				break;
			case DTK_CENTURY:
				if (tm.tm_year > 0)
					result->value = (tm.tm_year + 99) / 100;
				else
					result->value = -((99 - (tm.tm_year - 1)) / 100);
				break;
			case DTK_MILLENNIUM:
				if (tm.tm_year > 0)
					result->value = (tm.tm_year + 999) / 1000;
				else
					result->value = -((999 - (tm.tm_year - 1)) / 1000);
				break;
			case DTK_JULIAN:
				result->value = date2j(tm.tm_year,
									   tm.tm_mon,
									   tm.tm_mday) * 1000000L +
					(double)(tm.tm_hour * SECS_PER_HOUR +
							 tm.tm_min  * SECS_PER_MINUTE +
							 tm.tm_sec) * 1000000.0 / (double)SECS_PER_DAY;
				result->weight = 6;
				break;
			case DTK_ISOYEAR:
				result->value = date2isoyear(tm.tm_year,
											 tm.tm_mon,
											 tm.tm_mday);
				break;
			case DTK_DOW:
			case DTK_ISODOW:
				result->value = j2day(date2j(tm.tm_year,
											 tm.tm_mon,
											 tm.tm_mday));
				if (value == DTK_ISODOW && result->value == 0)
					result->value = 7;
				break;
			case DTK_DOY:
				result->value = (date2j(tm.tm_year,
										tm.tm_mon,
										tm.tm_mday) -
								 date2j(tm.tm_year, 1, 1) + 1);
				break;
			case DTK_TZ:
			case DTK_TZ_MINUTE:
			case DTK_TZ_HOUR:
			default:
				STROM_ELOG(kcxt, "unsupported unit of timestamp");
				return false;
		}
		result->expr_ops = &xpu_numeric_ops;
	}
	else if (type == RESERV)
	{
		memset(result, 0, sizeof(xpu_numeric_t));
		if (value == DTK_EPOCH)
		{
			Timestamp	epoch = kcxt->session->hostEpochTimestamp;

			result->value = (ts - epoch);
			result->weight = 6;
		}
		else
		{
			STROM_ELOG(kcxt, "unsupported unit of timestamp");
			return false;
		}
		result->expr_ops = &xpu_numeric_ops;
	}
	else
	{
		STROM_ELOG(kcxt, "not recognized unit of timestamp");
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_extract_timestamp(XPU_PGFUNCTION_ARGS)
{
	int		type, value;
	KEXP_PROCESS_ARGS2(numeric, text, key, timestamp, tval);

	if (XPU_DATUM_ISNULL(&key) || XPU_DATUM_ISNULL(&tval))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &key) ||
			 !extract_decode_unit(kcxt, &key, &type, &value) ||
			 !__pg_extract_timestamp_common(kcxt, result,
											type, value,
											tval.value, NULL))
	{
		assert(kcxt->errcode != 0);
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_extract_timestamptz(XPU_PGFUNCTION_ARGS)
{
	int		type, value;
	KEXP_PROCESS_ARGS2(numeric, text, key, timestamptz, tval);

	if (XPU_DATUM_ISNULL(&key) || XPU_DATUM_ISNULL(&tval))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &key) ||
			 !extract_decode_unit(kcxt, &key, &type, &value) ||
			 !__pg_extract_timestamp_common(kcxt, result,
											type, value, tval.value,
											SESSION_TIMEZONE(kcxt->session)))
	{
		assert(kcxt->errcode != 0);
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_extract_date(XPU_PGFUNCTION_ARGS)
{
	int		type, value;
	KEXP_PROCESS_ARGS2(numeric, text, key, date, dval);

	if (XPU_DATUM_ISNULL(&key) || XPU_DATUM_ISNULL(&dval))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &key) ||
			 !extract_decode_unit(kcxt, &key, &type, &value))
	{
		assert(kcxt->errcode != 0);
		return false;
	}
	else if (DATE_NOT_FINITE(dval.value) && (type == UNITS || type == RESERV))
	{
		memset(result, 0, sizeof(xpu_numeric_t));
		switch (value)
		{
			/* Oscillating units */
			case DTK_DAY:
			case DTK_MONTH:
			case DTK_QUARTER:
			case DTK_WEEK:
			case DTK_DOW:
			case DTK_ISODOW:
			case DTK_DOY:
				break;
			/* Monotonically-increasing units */
			case DTK_YEAR:
			case DTK_DECADE:
			case DTK_CENTURY:
			case DTK_MILLENNIUM:
			case DTK_JULIAN:
			case DTK_ISOYEAR:
			case DTK_EPOCH:
				if (DATE_IS_NOBEGIN(dval.value))
					result->kind = XPU_NUMERIC_KIND__NEG_INF;
				else
					result->kind = XPU_NUMERIC_KIND__POS_INF;
				result->expr_ops = &xpu_numeric_ops;
				break;
			default:
				STROM_ELOG(kcxt, "not a supported unit of date");
				return false;
		}
	}
	else if (type == UNITS)
	{
		int		year, mon, mday;

		j2date(dval.value + POSTGRES_EPOCH_JDATE, &year, &mon, &mday);
		memset(result, 0, sizeof(xpu_numeric_t));
		switch (value)
		{
			case DTK_DAY:
				result->value = mday;
				break;
			case DTK_MONTH:
				result->value = mon;
				break;
			case DTK_QUARTER:
				result->value = (mon - 1) / 3 + 1;
				break;
			case DTK_WEEK:
				result->value = date2isoweek(year, mon, mday);
				break;
			case DTK_YEAR:
				result->value = (year > 0 ? year : year - 1);
				break;
			case DTK_DECADE:
				if (year >= 0)
					result->value = year / 10;
				else
					result->value = -((8 - (year - 1)) / 10);
				break;
			case DTK_CENTURY:
				/* see comments in timestamp_part */
				if (year > 0)
					result->value = (year + 99) / 100;
				else
					result->value = -((99 - (year - 1)) / 100);
				break;
			case DTK_MILLENNIUM:
				/* see comments in timestamp_part */
				if (year > 0)
					result->value = (year + 999) / 1000;
				else
					result->value = -((999 - (year - 1)) / 1000);
				break;
			case DTK_JULIAN:
                result->value = dval.value + POSTGRES_EPOCH_JDATE;
                break;
			case DTK_ISOYEAR:
                result->value = date2isoyear(year, mon, mday);
                /* Adjust BC years */
				if (result->value <= 0)
					result->value -= 1;
				break;
			case DTK_DOW:
			case DTK_ISODOW:
				result->value = j2day(dval.value + POSTGRES_EPOCH_JDATE);
				if (value == DTK_ISODOW && result->value == 0)
					result->value = 7;
                break;
			case DTK_DOY:
                result->value = date2j(year, mon, mday) - date2j(year, 1, 1) + 1;
                break;
            default:
				STROM_ELOG(kcxt, "not a not supported for date");
				return false;
		}
		result->expr_ops = &xpu_numeric_ops;
	}
	else if (type == RESERV && value == DTK_EPOCH)
	{
		result->value = ((int64_t)dval.value
						 + POSTGRES_EPOCH_JDATE 
						 - UNIX_EPOCH_JDATE) * SECS_PER_DAY;
		result->expr_ops = &xpu_numeric_ops;
	}
	else
	{
		STROM_ELOG(kcxt, "not a recognized unit of time");
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_extract_time(XPU_PGFUNCTION_ARGS)
{
	int		type, value;
	KEXP_PROCESS_ARGS2(numeric, text, key, time, tval);

	if (XPU_DATUM_ISNULL(&key) || XPU_DATUM_ISNULL(&tval))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &key) ||
			 !extract_decode_unit(kcxt, &key, &type, &value))
	{
		assert(kcxt->errcode != 0);
		return false;
	}
	else if (type == UNITS)
	{
		struct pg_tm tm;
		fsec_t		fsec;

		memset(result, 0, sizeof(xpu_numeric_t));
		time2tm(tval.value, &tm, &fsec);
		switch (value)
		{
			case DTK_MICROSEC:
				result->value = tm.tm_sec * 1000000L + fsec;
				break;
			case DTK_MILLISEC:
				result->value = tm.tm_sec * 1000000L + fsec;
				result->weight = 3;
				break;
			case DTK_SECOND:
				result->value = tm.tm_sec * 1000000L + fsec;
				result->weight = 6;
				break;
			case DTK_MINUTE:
				result->value = tm.tm_min;
				break;
			case DTK_HOUR:
				result->value = tm.tm_hour;
				break;
			default:
				STROM_ELOG(kcxt, "unsupported unit of time");
				return false;
		}
		result->expr_ops = &xpu_numeric_ops;
	}
	else if (type == RESERV && value == DTK_EPOCH)
	{
		memset(result, 0, sizeof(xpu_numeric_t));
		result->value = tval.value / 1000000;
		result->expr_ops = &xpu_numeric_ops;
	}
	else
	{
		STROM_ELOG(kcxt, "not a recognized unit of time");
		return false;
	}
	return true;
}

	
PUBLIC_FUNCTION(bool)
pgfn_extract_timetz(XPU_PGFUNCTION_ARGS)
{
	int		type, value;
	KEXP_PROCESS_ARGS2(numeric, text, key, timetz, tval);

	if (XPU_DATUM_ISNULL(&key) || XPU_DATUM_ISNULL(&tval))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &key) ||
			 !extract_decode_unit(kcxt, &key, &type, &value))
	{
		assert(kcxt->errcode != 0);
		return false;
	}
	else if (type == UNITS)
	{
		struct pg_tm tm;
		fsec_t		fsec;
		int			tz;

		memset(result, 0, sizeof(xpu_numeric_t));
		timetz2tm(&tval.value, &tm, &fsec, &tz);
		switch (value)
		{
			case DTK_TZ:
				result->value = -tz;
                break;
            case DTK_TZ_MINUTE:
                result->value = (-tz / SECS_PER_MINUTE) % MINS_PER_HOUR;
                break;
			case DTK_TZ_HOUR:
                result->value  = -tz / SECS_PER_HOUR;
                break;
			case DTK_MICROSEC:
                result->value = tm.tm_sec * 1000000L + fsec;
                break;
            case DTK_MILLISEC:
				result->value = tm.tm_sec * 1000000L + fsec;
				result->weight = 3;
				break;
            case DTK_SECOND:
				result->value = tm.tm_sec * 1000000L + fsec;
				result->weight = 6;
				break;
            case DTK_MINUTE:
                result->value = tm.tm_min;
                break;
			case DTK_HOUR:
                result->value = tm.tm_hour;
                break;
			default:
				STROM_ELOG(kcxt, "unsupported unit of time");
				return false;
		}
		result->expr_ops = &xpu_numeric_ops;
	}
	else if (type == RESERV && value == DTK_EPOCH)
	{
		memset(result, 0, sizeof(xpu_numeric_ops));
		result->value = tval.value.time / 1000000L + tval.value.zone;
		result->expr_ops = &xpu_numeric_ops;
	}
	else
	{
		STROM_ELOG(kcxt, "not a recognized unit of time");
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_extract_interval(XPU_PGFUNCTION_ARGS)
{
	int		type, value;
	KEXP_PROCESS_ARGS2(numeric, text, key, interval, ival);

	if (XPU_DATUM_ISNULL(&key) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &key) ||
			 !extract_decode_unit(kcxt, &key, &type, &value))
	{
		assert(kcxt->errcode != 0);
		return false;
	}
	else if (type == UNITS)
	{
		/* see interval2itm */
		TimeOffset	time;
		TimeOffset	tfrac;
		int64_t		tm_year, tm_mon, tm_mday;
		int64_t		tm_hour, tm_min, tm_sec, tm_usec;

		tm_year = ival.value.month / MONTHS_PER_YEAR;
		tm_mon  = ival.value.month % MONTHS_PER_YEAR;
		tm_mday = ival.value.day;

		time    = ival.value.time;
		tfrac   = time / USECS_PER_HOUR;
		time   -= tfrac * USECS_PER_HOUR;
		tm_hour = tfrac;
		tfrac   = time / USECS_PER_MINUTE;
		time   -= tfrac * USECS_PER_MINUTE;
		tm_min  = tfrac;
		tfrac   = time / USECS_PER_SEC;
		time   -= tfrac * USECS_PER_SEC;
		tm_sec  = tfrac;
		tm_usec = time;

		switch (value)
		{
			case DTK_MICROSEC:
				result->value = tm_sec * 1000000L + tm_usec;
				break;
			case DTK_MILLISEC:
				result->value = tm_sec * 1000000L + tm_usec;
				result->weight = 3;
				break;
            case DTK_SECOND:
				result->value = tm_sec * 1000000L + tm_usec;
				result->weight = 6;
				break;
            case DTK_MINUTE:
                result->value = tm_min;
                break;
			case DTK_HOUR:
                result->value = tm_hour;
                break;
            case DTK_DAY:
                result->value = tm_mday;
                break;
            case DTK_MONTH:
                result->value = tm_mon;
                break;
            case DTK_QUARTER:
                result->value = (tm_mon / 3) + 1;
                break;
            case DTK_YEAR:
                result->value = tm_year;
                break;
			case DTK_DECADE:
				/* caution: C division may have negative remainder */
				result->value = tm_year / 10;
				break;
			case DTK_CENTURY:
                /* caution: C division may have negative remainder */
                result->value = tm_year / 100;
                break;
			case DTK_MILLENNIUM:
				/* caution: C division may have negative remainder */
				result->value = tm_year / 1000;
				break;
			default:
				STROM_ELOG(kcxt, "not a supported for interval type");
				return false;
		}
	}
	else if (type == RESERV && value == DTK_EPOCH)
	{
		memset(result, 0, sizeof(xpu_numeric_t));
		result->value =
			(((int64_t)ival.value.month / MONTHS_PER_YEAR) * SECS_PER_YEAR +
			 ((int64_t)ival.value.month % MONTHS_PER_YEAR) * SECS_PER_MONTH +
			 (int64_t)ival.value.day * SECS_PER_DAY) * 1000000L +
			(int64_t)ival.value.time;
		result->weight = 6;
		result->expr_ops = &xpu_numeric_ops;
	}
    else
	{
		STROM_ELOG(kcxt, "not a recognized unit for interval");
		return false;
	}
	return true;
}
