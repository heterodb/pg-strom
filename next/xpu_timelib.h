/*
 * xpu_timelib.h
 *
 * Date and time related definitions for both of GPU and DPU
 * --
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_TIMELIB_H
#define XPU_TIMELIB_H

#ifndef DATE_H
typedef int32_t         DateADT;
typedef int64_t         TimeADT;
typedef struct
{
    TimeADT     time;   /* all time units other than months and years */
    int32_t     zone;   /* numeric time zone, in seconds */
} TimeTzADT;
#define SizeOfTimeTzADT	(offsetof(TimeTzADT, zone) + sizeof(int32_t))

typedef int64_t			pg_time_t;

struct pg_tm
{
	int			tm_sec;
	int			tm_min;
	int			tm_hour;
	int			tm_mday;
	int			tm_mon;		/* origin 0, not 1 */
	int			tm_year;	/* relative to 1900 */
	int			tm_wday;
	int			tm_yday;
	int			tm_isdst;
	int64_t		tm_gmtoff;
	const char *tm_zone;	/* always NULL, on the device side */
};

#define MAX_TIME_PRECISION	6

#define DATEVAL_NOBEGIN		((DateADT)(-0x7fffffff - 1))
#define DATEVAL_NOEND       ((DateADT)  0x7fffffff)

#define DATE_NOBEGIN(j)		((j) = DATEVAL_NOBEGIN)
#define DATE_IS_NOBEGIN(j)	((j) == DATEVAL_NOBEGIN)
#define DATE_NOEND(j)		((j) = DATEVAL_NOEND)
#define DATE_IS_NOEND(j)	((j) == DATEVAL_NOEND)
#define DATE_NOT_FINITE(j)	(DATE_IS_NOBEGIN(j) || DATE_IS_NOEND(j))
#endif /* DATE_H */

#ifndef DATATYPE_TIMESTAMP_H
typedef int64_t		Timestamp;
typedef int64_t		TimestampTz;
typedef int64_t		TimeOffset;
typedef int32_t		fsec_t;		/* fractional seconds (in microseconds) */

typedef struct
{
    TimeOffset  time;   /* all time units other than days, months and years */
    int32_t     day;    /* days, after time for alignment */
    int32_t     month;  /* months and years, after time for alignment */
} Interval;

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
#endif /* DATATYPE_TIMESTAMP_H */

PGSTROM_SQLTYPE_SIMPLE_DECLARATION(date, DateADT);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(time, TimeADT);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(timetz, TimeTzADT);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(timestamp, Timestamp);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(timestamptz, TimestampTz);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(interval, Interval);

PUBLIC_FUNCTION(int)
xpu_date_arrow_datum_store(kern_context *kcxt,
						   void *buffer,
						   const kern_colmeta *cmeta,
						   const void *addr, int len);






/*
 * session_timezone (src/timezone/pgtz.h - pg_tz definition)
 */
#define TZ_MAX_TIMES	2000
#define TZ_MAX_TYPES	256
#define TZ_MAX_LEAPS	50
#ifndef TZ_STRLEN_MAX
#define TZ_STRLEN_MAX	255
#endif

struct pg_tz_ttinfo
{								/* time type information */
	int32_t		tt_utoff;		/* UT offset in seconds */
	bool		tt_isdst;		/* used to set tm_isdst */
	int32_t		tt_desigidx;	/* abbreviation list index */
	bool		tt_ttisstd;		/* transition is std time */
	bool		tt_ttisut;		/* transition is UT */
};

struct pg_tz_lsinfo
{								/* leap second information */
	pg_time_t	ls_trans;		/* transition time */
	int64_t		ls_corr;		/* correction to apply */
};

struct pg_tz_state
{
    int32_t		leapcnt;
	int32_t		timecnt;
	int32_t		typecnt;
	int32_t		charcnt;
	bool		goback;
	bool		goahead;
	int64_t		ats[TZ_MAX_TIMES];
	uint8_t		types[TZ_MAX_TIMES];
	struct pg_tz_ttinfo ttis[TZ_MAX_TYPES];
	char		chars[2 * (TZ_STRLEN_MAX + 1)];
	struct pg_tz_lsinfo lsis[TZ_MAX_LEAPS];
	/*
	 * The time type to use for early times or if no transitions. It is always
	 * zero for recent tzdb releases. It might be nonzero for data from tzdb
	 * 2018e or earlier.
	 */
	int			defaulttype;
};

struct pg_tz
{
	/* TZname contains the canonically-cased name of the timezone */
	char		TZname[TZ_STRLEN_MAX + 1];
	struct pg_tz_state state;
};
typedef struct pg_tz	pg_tz;


EXTERN_FUNCTION(int)
xpu_interval_write_heap(kern_context *kcxt,
						char *buffer,
						const xpu_datum_t *arg);

#endif  /* XPU_TIMELIB_H */
