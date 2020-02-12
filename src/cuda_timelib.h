/*
 * cuda_timelib.h
 *
 * Collection of date/time functions for CUDA GPU devices
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#ifndef CUDA_TIMELIB_H
#define CUDA_TIMELIB_H
#ifdef __CUDACC__
/* definitions copied from date.h */
typedef cl_int		DateADT;
typedef cl_long		TimeADT;
typedef struct
{
	TimeADT			time;	/* all time units other than months and years */
	cl_int			zone;	/* numeric time zone, in seconds */
} TimeTzADT;
typedef cl_long		TimeOffset;

#define MAX_TIME_PRECISION	6

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
typedef struct
{
	TimeOffset	time;	/* all time units other than days, months and years */
	cl_int		day;	/* days, after time for alignment */
	cl_int		month;	/* months and years, after time for alignment */
} Interval;

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

/* import from timezone/tzfile.h */
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

/* import from timezone/private.h */
#define YEARSPERREPEAT	400		/* years before a Gregorian repeat */
#define AVGSECSPERYEAR	31556952L
#define SECSPERREPEAT	((cl_long)YEARSPERREPEAT * (cl_long)AVGSECSPERYEAR)

/* import from include/datatype/timestamp.h */
#define JULIAN_MINYEAR (-4713)
#define JULIAN_MINMONTH (11)
#define JULIAN_MINDAY (24)
#define JULIAN_MAXYEAR (5874898)

#define IS_VALID_JULIAN(y,m,d) \
	(((y) > JULIAN_MINYEAR \
	  || ((y) == JULIAN_MINYEAR && \
		  ((m) > JULIAN_MINMONTH \
		   || ((m) == JULIAN_MINMONTH && (d) >= JULIAN_MINDAY)))) \
	 && (y) < JULIAN_MAXYEAR)

/* date/timestamp limits and range-checks */
#define DATETIME_MIN_JULIAN (0)
#define DATE_END_JULIAN (2147483494)	/* == date2j(JULIAN_MAXYEAR, 1, 1) */
#define TIMESTAMP_END_JULIAN (109203528) /* == date2j(294277, 1, 1) */
#define MIN_TIMESTAMP   INT64CONST(-211813488000000000)
#define END_TIMESTAMP   INT64CONST(9223371331200000000)
/* Range-check a data / timestamp */
#define IS_VALID_DATE(d) \
	((DATETIME_MIN_JULIAN - POSTGRES_EPOCH_JDATE) <= (d) && \
	 (d) < (DATE_END_JULIAN - POSTGRES_EPOCH_JDATE))
#define IS_VALID_TIMESTAMP(t)  (MIN_TIMESTAMP <= (t) && (t) < END_TIMESTAMP)

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
#define UNIX_EPOCH_JDATE		2440588		/* == date2j(1970, 1, 1) */
#define POSTGRES_EPOCH_JDATE	2451545		/* == date2j(2000, 1, 1) */

#ifndef SAMESIGN
#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))
#endif

/* definition copied from datetime.h */
#define FMODULO(t,q,u)											\
	do {														\
		(q) = (((t) < 0) ? ceil((t) / (u)) : floor((t) / (u))); \
		if ((q) != 0) (t) -= rint((q) * (u));					\
	} while(0)

#define TMODULO(t,q,u) \
	do {			   \
		(q) = ((t) / (u));			  \
		if ((q) != 0) (t) -= ((q) * (u));		\
	} while(0)

STATIC_INLINE(void)
interval_cmp_value(const Interval interval, cl_long *days, cl_long *fraction)
{
	*fraction = interval.time % USECS_PER_DAY;
	*days = (interval.time / USECS_PER_DAY +
			 interval.month * 30L +
			 interval.day);
}

/*
 * to be defined by session information
 */
DEVICE_FUNCTION(Timestamp) SetEpochTimestamp(void);
typedef struct {
	cl_long		ls_trans; /* pg_time_t in original */
	cl_long		ls_corr;
} tz_lsinfo;
typedef struct {
	cl_int		tt_gmtoff;
	cl_bool		tt_isdst;
	cl_int		tt_abbrind;
	cl_bool		tt_ttisstd;
	cl_bool		tt_ttisgmt;
} tz_ttinfo;
typedef struct {
	cl_int		leapcnt;
	cl_int		timecnt;
	cl_int		typecnt;
	cl_int		charcnt;
	cl_bool		goback;
	cl_bool		goahead;
	cl_long	   *ats;
	cl_uchar   *types;
	tz_ttinfo  *ttis;
	/* GPU kernel does not use chars[] */
	tz_lsinfo  *lsis;
	cl_int		defaulttype;
} tz_state;

extern const __device__ tz_state session_timezone_state;

#ifndef PG_DATE_TYPE_DEFINED
#define PG_DATE_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(date,DateADT,)
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(date,DateADT)
STROMCL_EXTERNAL_ARROW_TEMPLATE(date)
STROMCL_SIMPLE_COMPARE_TEMPLATE(date_,date,date,DateADT)
#endif /* PG_DATE_TYPE_DEFINED */
#ifndef PG_TIME_TYPE_DEFINED
#define PG_TIME_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(time,TimeADT,)
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(time,TimeADT)
STROMCL_EXTERNAL_ARROW_TEMPLATE(time)
STROMCL_SIMPLE_COMPARE_TEMPLATE(time_,time,time,TimeADT)
#endif /* PG_TIME_TYPE_DEFINED */
#ifndef PG_TIMETZ_TYPE_DEFINED
#define PG_TIMETZ_TYPE_DEFINED
STROMCL_INDIRECT_TYPE_TEMPLATE(timetz,TimeTzADT)
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(timetz,TimeTzADT)
STROMCL_UNSUPPORTED_ARROW_TEMPLATE(timetz)
#endif /* PG_TIMETZ_TYPE_DEFINED */

#ifndef PG_TIMESTAMP_TYPE_DEFINED
#define PG_TIMESTAMP_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(timestamp,Timestamp,)
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(timestamp,Timestamp)
STROMCL_EXTERNAL_ARROW_TEMPLATE(timestamp)
STROMCL_SIMPLE_COMPARE_TEMPLATE(timestamp_,
								timestamp,
								timestamp,
								Timestamp)
#endif /* PG_TIMESTAMP_TYPE_DEFINED */
#ifndef PG_TIMESTAMPTZ_TYPE_DEFINED
#define PG_TIMESTAMPTZ_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(timestamptz,TimestampTz,)
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(timestamptz,TimestampTz)
STROMCL_EXTERNAL_ARROW_TEMPLATE(timestamptz)
STROMCL_SIMPLE_COMPARE_TEMPLATE(timestamptz_,
								timestamptz,
								timestamptz,
								TimestampTz)
#endif /* PG_TIMESTAMPTZ_TYPE_DEFINED */
#ifndef PG_INTERVAL_TYPE_DEFINED
#define PG_INTERVAL_TYPE_DEFINED
STROMCL_INDIRECT_TYPE_TEMPLATE(interval,Interval)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(interval)
STROMCL_EXTERNAL_ARROW_TEMPLATE(interval)
#endif /* PG_INTERVAL_TYPE_DEFINED */

/* ---------------------------------------------------------------
 *
 * Type cast functions
 *
 * --------------------------------------------------------------- */
DEVICE_FUNCTION(pg_date_t)
pgfn_timestamp_date(kern_context *kcxt, pg_timestamp_t arg1);
DEVICE_FUNCTION(pg_date_t)
pgfn_timestamptz_date(kern_context *kcxt, pg_timestamptz_t arg1);
DEVICE_FUNCTION(pg_time_t)
pgfn_timetz_time(kern_context *kcxt, pg_timetz_t arg1);
DEVICE_FUNCTION(pg_time_t)
pgfn_timestamp_time(kern_context *kcxt, pg_timestamp_t arg1);
DEVICE_FUNCTION(pg_time_t)
pgfn_timestamptz_time(kern_context *kcxt, pg_timestamptz_t arg1);
DEVICE_FUNCTION(pg_timetz_t)
pgfn_time_timetz(kern_context *kcxt, pg_time_t arg1);
DEVICE_FUNCTION(pg_timetz_t)
pgfn_timestamptz_timetz(kern_context *kcxt, pg_timestamptz_t arg1);
#ifdef NOT_USED
DEVICE_FUNCTION(pg_timetz_t)
pgfn_timetz_scale(kern_context *kcxt, pg_timetz_t arg1, pg_int4_t arg2);
#endif
DEVICE_FUNCTION(pg_timestamp_t)
pgfn_date_timestamp(kern_context *kcxt, pg_date_t arg1);
DEVICE_FUNCTION(pg_timestamp_t)
pgfn_timestamptz_timestamp(kern_context *kcxt, pg_timestamptz_t arg1);
DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_date_timestamptz(kern_context *kcxt, pg_date_t arg1);
DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_timestamp_timestamptz(kern_context *kcxt, pg_timestamp_t arg1);

/*
 * Simple comparison
 */
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_cmp(kern_context *kcxt, pg_date_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_time_cmp(kern_context *kcxt, pg_time_t arg1, pg_time_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt,
				  pg_timestamp_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt,
				  pg_timestamptz_t arg1, pg_timestamptz_t arg2);

/*
 * Time/Date operators
 */
DEVICE_FUNCTION(pg_date_t)
pgfn_date_pli(kern_context *kcxt, pg_date_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_date_t)
pgfn_date_mii(kern_context *kcxt, pg_date_t arg1, pg_int4_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_date_mi(kern_context *kcxt, pg_date_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_timestamp_t)
pgfn_datetime_pl(kern_context *kcxt, pg_date_t arg1, pg_time_t arg2);
DEVICE_FUNCTION(pg_date_t)
pgfn_integer_pl_date(kern_context *kcxt, pg_int4_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_timestamp_t)
pgfn_timedate_pl(kern_context *kcxt, pg_time_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_interval_t)
pgfn_time_mi_time(kern_context *kcxt, pg_time_t arg1, pg_time_t arg2);
DEVICE_FUNCTION(pg_interval_t)
pgfn_timestamp_mi(kern_context *kcxt,
				  pg_timestamp_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_timetz_t)
pgfn_timetz_pl_interval(kern_context *kcxt,
						pg_timetz_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_timetz_t)
pgfn_timetz_mi_interval(kern_context *kcxt,
						pg_timetz_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_timestamptz_pl_interval(kern_context *kcxt,
							 pg_timestamptz_t arg1,
							 pg_interval_t arg2);
DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_timestamptz_mi_interval(kern_context *kcxt,
							 pg_timestamptz_t arg1,
							 pg_interval_t arg2);
DEVICE_FUNCTION(pg_interval_t)
pgfn_interval_um(kern_context *kcxt, pg_interval_t arg1);
DEVICE_FUNCTION(pg_interval_t)
pgfn_interval_pl(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_interval_t)
pgfn_interval_mi(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_datetimetz_timestamptz(kern_context *kcxt,
							pg_date_t arg1, pg_timetz_t arg2);
/*
 * Date comparison
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_eq_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_ne_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_lt_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_le_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_gt_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_ge_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_date_cmp_timestamp(kern_context *kcxt,
						pg_date_t arg1, pg_timestamp_t arg2);

/*
 * Comparison between timetz
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_timetz_eq(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timetz_ne(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timetz_lt(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timetz_le(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timetz_ge(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timetz_gt(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2);
/*
 * Timestamp comparison
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_eq_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_ne_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_lt_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_le_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_gt_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_ge_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_timestamp_cmp_date(kern_context *kcxt,
						pg_timestamp_t arg1, pg_date_t arg2);
/*
 * Comparison between date and timestamptz
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_lt_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_le_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_eq_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_ge_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_gt_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_date_ne_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2);
/*
 * Comparison between timestamptz and date
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_lt_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_le_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_eq_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ge_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_gt_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ne_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2);
/*
 * Comparison between timestamp and timestamptz
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_lt_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_le_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_eq_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_ge_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_gt_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamp_ne_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2);
/*
 * Comparison between timestamptz and timestamp
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_lt_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_le_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_eq_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ge_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_gt_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ne_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2);
/*
 * Comparison between pg_interval_t
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_interval_eq(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_interval_ne(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_interval_lt(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_interval_le(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_interval_ge(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_interval_gt(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2);
/*
 * current date time function
 */
DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_now(kern_context *kcxt);
/*
 * overlaps() SQL functions
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_overlaps_time(kern_context *kcxt,
				   pg_time_t arg1, pg_time_t arg2,
				   pg_time_t arg3, pg_time_t arg4);
DEVICE_FUNCTION(pg_bool_t)
pgfn_overlaps_timetz(kern_context *kcxt,
					 pg_timetz_t arg1, pg_timetz_t arg2,
					 pg_timetz_t arg3, pg_timetz_t arg4);
DEVICE_FUNCTION(pg_bool_t)
pgfn_overlaps_timestamp(kern_context *kcxt,
						pg_timestamp_t arg1, pg_timestamp_t arg2,
						pg_timestamp_t arg3, pg_timestamp_t arg4);
DEVICE_FUNCTION(pg_bool_t)
pgfn_overlaps_timestamptz(kern_context *kcxt,
						  pg_timestamptz_t arg1, pg_timestamptz_t arg2,
						  pg_timestamptz_t arg3, pg_timestamptz_t arg4);
/*
 * EXTRACT()
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_timestamp(kern_context *kcxt,
                       pg_text_t arg1, pg_timestamp_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_timestamptz(kern_context *kcxt,
                         pg_text_t arg1, pg_timestamptz_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_interval(kern_context *kcxt, pg_text_t arg1, pg_interval_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_timetz(kern_context *kcxt, pg_text_t arg1, pg_timetz_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_time(kern_context *kcxt, pg_text_t arg1, pg_time_t arg2);

#endif	/* __CUDACC__ */
#endif	/* CUDA_TIMELIB_H */
