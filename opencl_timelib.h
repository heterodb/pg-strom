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

/* definitions copied from date.h */
typedef cl_int		DateADT;
typedef cl_long		TimeADT;

#define DATEVAL_NOBEGIN		((cl_int)(-0x7fffffff - 1))
#define DATEVAL_NOEND		((cl_int)  0x7fffffff)

#define DATE_NOBEGIN(j)		((j) = DATEVAL_NOBEGIN)
#define DATE_IS_NOBEGIN(j)	((j) == DATEVAL_NOBEGIN)
#define DATE_NOEND(j)		((j) = DATEVAL_NOEND)
#define DATE_IS_NOEND(j)	((j) == DATEVAL_NOEND)
#define DATE_NOT_FINITE(j)	(DATE_IS_NOBEGIN(j) || DATE_IS_NOEND(j))

/* definitions copied from timestamp.h */
typedef cl_long	Timestamp;
typedef cl_long	TimestampTz;
typedef cl_long	TimeOffset;
typedef cl_int	fsec_t;		/* fractional seconds (in microseconds) */

#define DAYS_PER_YEAR	365.25	/* assumes leap year every four years */
#define MONTHS_PER_YEAR	12
#define DAYS_PER_MONTH	30		/* assumes exactly 30 days per month */
#define HOURS_PER_DAY	24		/* assume no daylight savings time changes */

#define SECS_PER_YEAR	(36525 * 864)   /* avoid floating-point computation */
#define SECS_PER_DAY	86400
#define SECS_PER_HOUR	3600
#define SECS_PER_MINUTE	60
#define MINS_PER_HOUR	60

#define USECS_PER_DAY		INT64CONST(86400000000)
#define USECS_PER_HOUR		INT64CONST(3600000000)
#define USECS_PER_MINUTE	INT64CONST(60000000)
#define USECS_PER_SEC		INT64CONST(1000000)

#define DT_NOBEGIN		(-INT64CONST(0x7fffffffffffffff) - 1)
#define DT_NOEND		(INT64CONST(0x7fffffffffffffff))

#define TIMESTAMP_NOBEGIN(j) \
	do {(j) = DT_NOBEGIN;} while (0)
#define TIMESTAMP_IS_NOBEGIN(j) ((j) == DT_NOBEGIN)

#define TIMESTAMP_NOEND(j) \
	do {(j) = DT_NOEND;} while (0)

#define TIMESTAMP_IS_NOEND(j) \
	((j) == DT_NOEND)

#define TIMESTAMP_NOT_FINITE(j) \
	(TIMESTAMP_IS_NOBEGIN(j) || TIMESTAMP_IS_NOEND(j))

/* Julian-date equivalents of Day 0 in Unix and Postgres reckoning */
#define UNIX_EPOCH_JDATE		2440588	/* == date2j(1970, 1, 1) */
#define POSTGRES_EPOCH_JDATE	2451545	/* == date2j(2000, 1, 1) */

/* definition copied from datetime.h */
#define TMODULO(t,q,u) \
	do {			   \
		(q) = ((t) / (u));			  \
		if ((q) != 0) (t) -= ((q) * (u));		\
	} while(0)

/* definition copied from pgtime.h */
struct pg_tm
{
	cl_int		tm_sec;
	cl_int		tm_min;
	cl_int		tm_hour;
	cl_int		tm_mday;
	cl_int		tm_mon;		/* origin 0, not 1 */
	cl_int		tm_year;	/* relative to 1900 */
	cl_int		tm_wday;
	cl_int		tm_yday;
	cl_int		tm_isdst;
	cl_long		tm_gmtoff;
	// const char *tm_zone;	not supported yet
};

#ifndef PG_DATE_TYPE_DEFINED
#define PG_DATE_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(date,DateADT)
#endif

#ifndef PG_TIME_TYPE_DEFINED
#define PG_TIME_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(time,TimeADT)
#endif

#ifndef PG_TIMESTAMP_TYPE_DEFINED
#define PG_TIMESTAMP_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(timestamp,Timestamp)
#endif

#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4,cl_int);
#endif

/*
 * Support routines
 */
static inline cl_int
date2j(cl_int y, cl_int m, cl_int d)
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

static inline void
j2date(cl_int jd, int *year, int *month, int *day)
{
	cl_uint		julian;
	cl_uint		quad;
	cl_uint		extra;
	cl_int		y;

	julian = jd;
	julian += 32044;
	quad = julian / 146097;
	extra = (julian - quad * 146097) * 4 + 3;
	julian += 60 + quad * 3 + extra / 146097;
	quad = julian / 1461;
	julian -= quad * 1461;
	y = julian * 4 / 1461;
	julian = ((y != 0) ? ((julian + 305) % 365) : ((julian + 306) % 366))
		+ 123;
	y += quad * 4;
	*year = y - 4800;
	quad = julian * 2141 / 65536;
	*day = julian - 7834 * quad / 256;
	*month = (quad + 10) % MONTHS_PER_YEAR + 1;
}

static inline void
dt2time(Timestamp jd, cl_int *hour, cl_int *min, cl_int *sec, cl_int *fsec)
{
	TimeOffset  time;

	time = jd;

	*hour = time / USECS_PER_HOUR;
	time -= (*hour) * USECS_PER_HOUR;
	*min = time / USECS_PER_MINUTE;
	time -= (*min) * USECS_PER_MINUTE;
	*sec = time / USECS_PER_SEC;
	*fsec = time - (*sec * USECS_PER_SEC);
}

/* simplified version; no timezone support now */
static cl_bool
timestamp2tm(Timestamp dt, struct pg_tm *tm, fsec_t *fsec)
{
	cl_long		date;	/* Timestamp in original */
	cl_long		time;	/* Timestamp in original */
	cl_long		utime;	/* pg_time_t in original */

	time = dt;
	TMODULO(time, date, USECS_PER_DAY);

	if (time < INT64CONST(0))
	{
		time += USECS_PER_DAY;
		date -= 1;
	}
	/* add offset to go from J2000 back to standard Julian date */
	date += POSTGRES_EPOCH_JDATE;

	/* Julian day routine does not work for negative Julian days */
	if (date < 0 || date > (Timestamp) INT_MAX)
		return false;

	j2date((cl_int) date, &tm->tm_year, &tm->tm_mon, &tm->tm_mday);
	dt2time(time, &tm->tm_hour, &tm->tm_min, &tm->tm_sec, fsec);

	/* Done if no TZ conversion wanted */
	tm->tm_isdst = -1;
	tm->tm_gmtoff = 0;
	//tm->tm_zone = NULL;

	return true;
}

/* ---------------------------------------------------------------
 *
 * Type cast functions
 *
 * --------------------------------------------------------------- */
static pg_date_t
pgfn_timestamp_date(__private cl_int *errcode, pg_timestamp_t arg1)
{
	pg_date_t		result;
	struct pg_tm	tm;
	fsec_t			fsec;

	if (arg1.isnull)
		result.isnull = true;
	else if (TIMESTAMP_IS_NOBEGIN(arg1.value))
	{
		result.isnull = false;
		DATE_NOBEGIN(result.value);
	}
	else if (TIMESTAMP_IS_NOEND(arg1.value))
	{
		result.isnull = false;
		DATE_NOEND(result.value);
	}
	else if (!timestamp2tm(arg1.value, &tm, &fsec))
	{
		result.isnull = true;
		STROM_SET_ERROR(errcode, StromError_RowReCheck);
	}
	else
	{
		result.value = (date2j(tm.tm_year, tm.tm_mon, tm.tm_mday)
						- POSTGRES_EPOCH_JDATE);
	}
	return result;
}


static pg_time_t
pgfn_timestamp_time(__private cl_int *errcode, pg_timestamp_t arg1)
{
	pg_time_t		result;
	struct pg_tm	tm;
	fsec_t			fsec;

	if (arg1.isnull)
		result.isnull = true;
	else if (TIMESTAMP_NOT_FINITE(arg1.value))
		result.isnull = true;
	else if (!timestamp2tm(arg1.value, &tm, &fsec))
	{
		result.isnull = true;
		STROM_SET_ERROR(errcode, StromError_RowReCheck);
	}
	else
	{
		result.isnull = false;
		result.value =
			((((tm.tm_hour * MINS_PER_HOUR
				+ tm.tm_min) * SECS_PER_MINUTE)
			    + tm.tm_sec) * USECS_PER_SEC)
			    + fsec;
	}
	return result;
}

static pg_timestamp_t
pgfn_date_timestamp(__private cl_int *errcode, pg_date_t arg1)
{
	pg_timestamp_t	result;

	if (arg1.isnull)
	{
		result.isnull = true;
	}
	else if (DATE_IS_NOBEGIN(arg1.value))
	{
		result.isnull = false;
		TIMESTAMP_NOBEGIN(result.value);
	}
	else if (DATE_IS_NOEND(arg1.value))
	{
		result.isnull = false;
		TIMESTAMP_NOEND(result.value);
	}
	else
	{
		/* date is days since 2000, timestamp is microseconds since same... */
		result.isnull = false;
		result.value = arg1.value * USECS_PER_DAY;
		/* Date's range is wider than timestamp's, so check for overflow */
		if (result.value / USECS_PER_DAY != arg1.value)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_RowReCheck);
		}
	}
	return result;
}

/*
 * Time/Date operators
 */
static pg_date_t
pgfn_date_pli(__private cl_int *errcode, pg_date_t arg1, pg_int4_t arg2)
{
	pg_date_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		if (DATE_NOT_FINITE(arg1.value))
			result.value = arg1.value;	/* can't change infinity */
		else
			result.value = arg1.value + arg2.value;
	}
	return result;
}

static pg_date_t
pgfn_date_mii(__private cl_int *errcode, pg_date_t arg1, pg_int4_t arg2)
{
	pg_date_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		if (DATE_NOT_FINITE(arg1.value))
			result.value = arg1.value;	/* can't change infinity */
		else
			result.value = arg1.value - arg2.value;
	}
	return result;
}

static pg_int4_t
pgfn_date_mi(__private cl_int *errcode, pg_date_t arg1, pg_date_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else if (DATE_NOT_FINITE(arg1.value) || DATE_NOT_FINITE(arg2.value))
	{
		result.isnull = true;
		STROM_SET_ERROR(errcode, StromError_RowReCheck);
	}
	else
	{
		result.isnull = false;
		result.value = (cl_int)(arg1.value - arg2.value);
	}
	return result;
}

static pg_timestamp_t
pgfn_datetime_pl(__private cl_int *errcode, pg_date_t arg1, pg_time_t arg2)
{
	pg_timestamp_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result = pgfn_date_timestamp(errcode, arg1);
		if (!TIMESTAMP_NOT_FINITE(result.value))
			result.value += arg2.value;
	}
	return result;
}

static pg_date_t
pgfn_integer_pl_date(__private cl_int *errcode, pg_int4_t arg1, pg_date_t arg2)
{
	return pgfn_date_pli(errcode, arg2, arg1);
}

static pg_timestamp_t
pgfn_timedata_pl(__private cl_int *errcode, pg_time_t arg1, pg_date_t arg2)
{
	return pgfn_datetime_pl(errcode, arg2, arg1);
}

/*
 * Date comparison
 */
static pg_bool_t
pgfn_date_eq_timestamp(__private cl_int *errcode,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(errcode, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value == arg2.value);
	}
	return result;
}

static pg_bool_t
pgfn_date_ne_timestamp(__private cl_int *errcode,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(errcode, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value != arg2.value);
	}
	return result;
}

static pg_bool_t
pgfn_date_lt_timestamp(__private cl_int *errcode,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(errcode, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value < arg2.value);
	}
	return result;
}

static pg_bool_t
pgfn_date_le_timestamp(__private cl_int *errcode,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(errcode, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value <= arg2.value);
	}
	return result;
}

static pg_bool_t
pgfn_date_gt_timestamp(__private cl_int *errcode,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(errcode, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value > arg2.value);
	}
	return result;
}

static pg_bool_t
pgfn_date_ge_timestamp(__private cl_int *errcode,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(errcode, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value >= arg2.value);
	}
	return result;
}

static pg_int4_t
date_cmp_timestamp(__private cl_int *errcode,
				   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_int4_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(errcode, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		if (dt1.value > arg2.value)
			result.value = 1;
		else if (dt1.value < arg2.value)
			result.value = -1;
		else
			result.value = 0;
	}
	return result;
}

/*
 * Timestamp comparison
 */
static pg_bool_t
pgfn_timestamp_eq_date(__private cl_int *errcode,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(errcode, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value == dt2.value);
	}
	return result;
}

static pg_bool_t
pgfn_timestamp_ne_date(__private cl_int *errcode,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(errcode, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value != dt2.value);
	}
	return result;
}

static pg_bool_t
pgfn_timestamp_lt_date(__private cl_int *errcode,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(errcode, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value < dt2.value);
	}
	return result;
}

static pg_bool_t
pgfn_timestamp_le_date(__private cl_int *errcode,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(errcode, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value <= dt2.value);
	}
	return result;
}

static pg_bool_t
pgfn_timestamp_gt_date(__private cl_int *errcode,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(errcode, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value > dt2.value);
	}
	return result;
}

static pg_bool_t
pgfn_timestamp_ge_date(__private cl_int *errcode,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(errcode, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value >= dt2.value);
	}
	return result;
}

static pg_int4_t
pgfn_timestamp_cmp_date(__private cl_int *errcode,
						pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_int4_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(errcode, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		if (arg1.value > dt2.value)
			result.value = 1;
		else if (arg1.value < dt2.value)
			result.value = -1;
		else
			result.value = 0;
	}
	return result;
}

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_TIMELIB_H */
