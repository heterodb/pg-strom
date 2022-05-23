/*
 * xpu_timelib.h
 *
 * Date and time related definitions for xPU(GPU/DPU/SPU).
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
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
#endif /* DATATYPE_TIMESTAMP_H */

PGSTROM_SQLTYPE_SIMPLE_DECLARATION(date, DateADT);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(time, TimeADT);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(timetz, TimeTzADT);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(timestamp, Timestamp);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(timestamptz, TimestampTz);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(interval, Interval);

/*
 * session_timezone
 */
struct xpu_ttinfo
{
	int			tt_utoff;		/* UT offset in seconds */
	bool		tt_isdst;		/* used to set tm_isdst */
	int			tt_desigidx;	/* abbreviation list index */
	bool		tt_ttisstd;		/* transition is std time */
	bool		tt_ttisut;		/* transition is UT */
};
typedef struct xpu_ttinfo	xpu_ttinfo;

struct xpu_lsinfo
{
	int64_t		ls_trans;		/* transition time */
	int64_t		ls_corr;		/* correction to apply */
};
typedef struct xpu_lsinfo	xpu_lsinfo;

struct xpu_tz_info
{
	char		   *tzname;
	int				leapcnt;
	int				timecnt;
	int				typecnt;
	int				charcnt;
	bool			goback;
	bool			goahead;
	int64_t		   *ats;
	unsigned char  *types;
	xpu_ttinfo	   *ttis;
	char		   *chars;
	xpu_lsinfo	   *lsis;
	int				defaulttype;
};
typedef struct xpu_tz_info	xpu_tz_info;
//TODO: encode/decode the structure



#endif  /* XPU_TIMELIB_H */
