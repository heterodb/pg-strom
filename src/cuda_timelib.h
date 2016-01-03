/*
 * cuda_timelib.h
 *
 * Collection of date/time functions for OpenCL devices
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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

#ifndef SAMESIGN
#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))
#endif

/* definition copied from datetime.h */
#define TMODULO(t,q,u) \
	do {			   \
		(q) = ((t) / (u));			  \
		if ((q) != 0) (t) -= ((q) * (u));		\
	} while(0)

/* definition copied from datetime.c */
static const int day_tab[2][13] =
{
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0}
};

/* definition copied from pgtime.h */
static const int mon_lengths[2][MONSPERYEAR] = {
	{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
	{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

static const int year_lengths[2] = {
	DAYSPERNYEAR, DAYSPERLYEAR
};

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
STATIC_INLINE(Datum)
pg_date_to_datum(DateADT value)
{
	return pg_int4_to_datum(value);
}
#endif

#ifndef PG_TIME_TYPE_DEFINED
#define PG_TIME_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(time,TimeADT)
STATIC_INLINE(Datum)
pg_time_to_datum(TimeADT value)
{
	return pg_int8_to_datum(value);
}
#endif

#ifndef PG_TIMETZ_TYPE_DEFINED
#define PG_TIMETZ_TYPE_DEFINED
STROMCL_INDIRECT_TYPE_TEMPLATE(timetz,TimeTzADT)
#endif

#ifndef PG_TIMESTAMP_TYPE_DEFINED
#define PG_TIMESTAMP_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(timestamp,Timestamp)
STATIC_INLINE(Datum)
pg_timestamp_to_datum(Timestamp value)
{
	return pg_int8_to_datum(value);
}
#endif

#ifndef PG_TIMESTAMPTZ_TYPE_DEFINED
#define PG_TIMESTAMPTZ_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(timestamptz,TimestampTz)
STATIC_INLINE(Datum)
pg_timestamptz_to_datum(TimestampTz value)
{
	return pg_int8_to_datum(value);
}
#endif

#ifndef PG_INTERVAL_TYPE_DEFINED
#define PG_INTERVAL_TYPE_DEFINED
STROMCL_INDIRECT_TYPE_TEMPLATE(interval,Interval)
#endif

/*
 * Support routines
 */
STATIC_INLINE(cl_int)
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

STATIC_INLINE(void)
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

STATIC_INLINE(void)
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

STATIC_INLINE(Timestamp)
dt2local(Timestamp dt, int tz)
{
    return dt -= (tz * USECS_PER_SEC);
}

STATIC_INLINE(TimeOffset)
time2t(const int hour, const int min, const int sec, const fsec_t fsec)
{
    return (((((hour * MINS_PER_HOUR) + min) * SECS_PER_MINUTE) + sec) *
			USECS_PER_SEC) + fsec;
}

STATIC_INLINE(int)
time2tm(TimeADT time, struct pg_tm *tm, fsec_t *fsec)
{
    tm->tm_hour = time / USECS_PER_HOUR;
    time -= tm->tm_hour * USECS_PER_HOUR;
    tm->tm_min = time / USECS_PER_MINUTE;
    time -= tm->tm_min * USECS_PER_MINUTE;
    tm->tm_sec = time / USECS_PER_SEC;
    time -= tm->tm_sec * USECS_PER_SEC;
    *fsec = time;

	return 0;
}

STATIC_INLINE(int)
tm2timetz(struct pg_tm * tm, fsec_t fsec, int tz, TimeTzADT *result)
{
    result->time = ((((tm->tm_hour * MINS_PER_HOUR + tm->tm_min) 
					  * SECS_PER_MINUTE) + tm->tm_sec) * USECS_PER_SEC) + fsec;
    result->zone = tz;

	return 0;
}

STATIC_INLINE(int)
increment_overflow(int *number, int delta)
{
    int         number0;

    number0 = *number;
    *number += delta;
    return (*number < number0) != (delta < 0);
}

STATIC_INLINE(int)
leaps_thru_end_of_no_recursive(const int y)
{
	assert(y >= 0);
	return y / 4 - y / 100 + y / 400;
}

STATIC_INLINE(int)
leaps_thru_end_of(const int y)
{
    return (y >= 0) ? (y / 4 - y / 100 + y / 400) :
        -(leaps_thru_end_of_no_recursive(-(y + 1)) + 1);
}

STATIC_INLINE(struct pg_tm *)
timesub(const cl_long *timep,	/* pg_time_t in original */
		long offset, const tz_state * sp, struct pg_tm * tmp)
{
	const tz_lsinfo *lp;
	cl_long		tdays;			/* pg_time_t in original */
	int			idays;			/* unsigned would be so 2003 */
	long		rem;
	int			y;
	const int  *ip;
	long		corr;
	int			hit;
	int			i;

	corr = 0;
	hit = 0;
	i = sp->leapcnt;
	while (--i >= 0)
	{
		lp = &sp->lsis[i];
		if (*timep >= lp->ls_trans)
		{
			if (*timep == lp->ls_trans)
			{
				hit = ((i == 0 && lp->ls_corr > 0) ||
					   lp->ls_corr > sp->lsis[i - 1].ls_corr);
				if (hit)
					while (i > 0 &&
						   sp->lsis[i].ls_trans ==
						   sp->lsis[i - 1].ls_trans + 1 &&
						   sp->lsis[i].ls_corr ==
						   sp->lsis[i - 1].ls_corr + 1)
					{
						++hit;
						--i;
					}
			}
			corr = lp->ls_corr;
			break;
		}
	}
	y = EPOCH_YEAR;
	tdays = *timep / SECSPERDAY;
	rem = *timep - tdays * SECSPERDAY;
	while (tdays < 0 || tdays >= year_lengths[isleap(y)])
	{
		int			newy;
		cl_long		tdelta;		/* pg_time_t in original */
		int			idelta;
		int			leapdays;

		tdelta = tdays / DAYSPERLYEAR;
		idelta = tdelta;
		if (tdelta - idelta >= 1 || idelta - tdelta >= 1)
			return NULL;
		if (idelta == 0)
			idelta = (tdays < 0) ? -1 : 1;
		newy = y;
		if (increment_overflow(&newy, idelta))
			return NULL;
		leapdays = leaps_thru_end_of(newy - 1) -
			leaps_thru_end_of(y - 1);
		tdays -= ((cl_long) newy - y) * DAYSPERNYEAR;
		tdays -= leapdays;
		y = newy;
	}
	{
		long		seconds;

		seconds = tdays * SECSPERDAY + 0.5;
		tdays = seconds / SECSPERDAY;
		rem += seconds - tdays * SECSPERDAY;
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
			return NULL;
		idays += year_lengths[isleap(y)];
	}
	while (idays >= year_lengths[isleap(y)])
	{
		idays -= year_lengths[isleap(y)];
		if (increment_overflow(&y, 1))
			return NULL;
	}
	tmp->tm_year = y;
	if (increment_overflow(&tmp->tm_year, -TM_YEAR_BASE))
		return NULL;
	tmp->tm_yday = idays;

	/*
	 * The "extra" mods below avoid overflow problems.
	 */
	tmp->tm_wday = EPOCH_WDAY +
		((y - EPOCH_YEAR) % DAYSPERWEEK) *
		(DAYSPERNYEAR % DAYSPERWEEK) +
		leaps_thru_end_of(y - 1) -
		leaps_thru_end_of(EPOCH_YEAR - 1) +
		idays;
	tmp->tm_wday %= DAYSPERWEEK;
	if (tmp->tm_wday < 0)
		tmp->tm_wday += DAYSPERWEEK;
	tmp->tm_hour = (int) (rem / SECSPERHOUR);
	rem %= SECSPERHOUR;
	tmp->tm_min = (int) (rem / SECSPERMIN);

	/*
	 * A positive leap second requires a special representation. This uses
	 * "... ??:59:60" et seq.
	 */
	tmp->tm_sec = (int) (rem % SECSPERMIN) + hit;
	ip = mon_lengths[isleap(y)];
	for (tmp->tm_mon = 0; idays >= ip[tmp->tm_mon]; ++(tmp->tm_mon))
		idays -= ip[tmp->tm_mon];
	tmp->tm_mday = (int) (idays + 1);
	tmp->tm_isdst = 0;
	tmp->tm_gmtoff = offset;
	return tmp;
}

STATIC_INLINE(struct pg_tm *)
localsub_no_recursive(const cl_long *timep,	/* pg_time_t in original */
					  long offset,
					  struct pg_tm * tmp,
					  const tz_state *sp)	/* const pg_tz *tz in original */
{
	const tz_ttinfo *ttisp;
	int			i;
	struct pg_tm *result;
	const cl_long t = *timep;	/* pg_time_t in original */

#if 0
	if ((sp->goback && t < sp->ats[0]) ||
		(sp->goahead && t > sp->ats[sp->timecnt - 1]))
	{
		cl_long	newt = t;	/* pg_time_t in original */
		cl_long	seconds;	/* pg_time_t in original */
		cl_long	tcycles;	/* pg_time_t in original */
		cl_long	icycles;	/* pg_time_t in original */

		if (t < sp->ats[0])
			seconds = sp->ats[0] - t;
		else
			seconds = t - sp->ats[sp->timecnt - 1];
		--seconds;
		tcycles = seconds / YEARSPERREPEAT / AVGSECSPERYEAR;
		++tcycles;
		icycles = tcycles;
		if (tcycles - icycles >= 1 || icycles - tcycles >= 1)
			return NULL;
		seconds = icycles;
		seconds *= YEARSPERREPEAT;
		seconds *= AVGSECSPERYEAR;
		if (t < sp->ats[0])
			newt += seconds;
		else
			newt -= seconds;
		if (newt < sp->ats[0] ||
			newt > sp->ats[sp->timecnt - 1])
			return NULL;		/* "cannot happen" */
		result = localsub(&newt, offset, tmp, tz);
		if (result == tmp)
		{
			cl_long newy;		/* pg_time_t in original */

			newy = tmp->tm_year;
			if (t < sp->ats[0])
				newy -= icycles * YEARSPERREPEAT;
			else
				newy += icycles * YEARSPERREPEAT;
			tmp->tm_year = newy;
			if (tmp->tm_year != newy)
				return NULL;
		}
		return result;
	}
#else
	/* Don't recursive call in this function. */
	assert(!((sp->goback && t < sp->ats[0]) ||
			 (sp->goahead && t > sp->ats[sp->timecnt - 1])));
#endif

	if (sp->timecnt == 0 || t < sp->ats[0])
	{
		i = 0;
		while (sp->ttis[i].tt_isdst)
			if (++i >= sp->typecnt)
			{
				i = 0;
				break;
			}
	}
	else
	{
		int			lo = 1;
		int			hi = sp->timecnt;

		while (lo < hi)
		{
			int			mid = (lo + hi) >> 1;

			if (t < sp->ats[mid])
				hi = mid;
			else
				lo = mid + 1;
		}
		i = (int) sp->types[lo - 1];
	}
	ttisp = &sp->ttis[i];

	result = timesub(&t, ttisp->tt_gmtoff, sp, tmp);
	tmp->tm_isdst = ttisp->tt_isdst;

	return result;
}

STATIC_INLINE(struct pg_tm *)
localsub(const cl_long *timep, 	/* pg_time_t in original */
		 long offset,
		 struct pg_tm * tmp,
		 const tz_state *sp	    /* const pg_tz *tz in original*/ )
{
	const tz_ttinfo *ttisp;
	int			i;
	struct pg_tm *result;
	const cl_long t = *timep;	/* pg_time_t in original */

	if ((sp->goback && t < sp->ats[0]) ||
		(sp->goahead && t > sp->ats[sp->timecnt - 1]))
	{
		cl_long	newt = t;		/* pg_time_t in original */
		cl_long	seconds;		/* pg_time_t in original */
		cl_long	tcycles;		/* pg_time_t in original */
		cl_long	icycles;

		if (t < sp->ats[0])
			seconds = sp->ats[0] - t;
		else
			seconds = t - sp->ats[sp->timecnt - 1];
		--seconds;
		tcycles = seconds / YEARSPERREPEAT / AVGSECSPERYEAR;
		++tcycles;
		icycles = tcycles;
		if (tcycles - icycles >= 1 || icycles - tcycles >= 1)
			return NULL;
		seconds = icycles;
		seconds *= YEARSPERREPEAT;
		seconds *= AVGSECSPERYEAR;
		if (t < sp->ats[0])
			newt += seconds;
		else
			newt -= seconds;
		if (newt < sp->ats[0] ||
			newt > sp->ats[sp->timecnt - 1])
			return NULL;		/* "cannot happen" */
		result = localsub_no_recursive(&newt, offset, tmp, sp);
		if (result == tmp)
		{
			cl_long	newy;		/* pg_time_t in original */

			newy = tmp->tm_year;
			if (t < sp->ats[0])
				newy -= icycles * YEARSPERREPEAT;
			else
				newy += icycles * YEARSPERREPEAT;
			tmp->tm_year = newy;
			if (tmp->tm_year != newy)
				return NULL;
		}
		return result;
	}
	if (sp->timecnt == 0 || t < sp->ats[0])
	{
		i = 0;
		while (sp->ttis[i].tt_isdst)
			if (++i >= sp->typecnt)
			{
				i = 0;
				break;
			}
	}
	else
	{
		int			lo = 1;
		int			hi = sp->timecnt;

		while (lo < hi)
		{
			int			mid = (lo + hi) >> 1;

			if (t < sp->ats[mid])
				hi = mid;
			else
				lo = mid + 1;
		}
		i = (int) sp->types[lo - 1];
	}
	ttisp = &sp->ttis[i];

	result = timesub(&t, ttisp->tt_gmtoff, sp, tmp);
	tmp->tm_isdst = ttisp->tt_isdst;

	return result;
}

STATIC_INLINE(struct pg_tm *)
pg_localtime(const cl_long *timep,	/* pg_time_t in original */
			 struct pg_tm *tm,
			 const tz_state *sp)	/* const pg_tz *tz in original */
{
	/*
	 * pg_localtime() returns tm if success. NULL, elsewhere.
	 */
	return localsub(timep, 0L, tm, sp);
}

STATIC_INLINE(int)
pg_next_dst_boundary_no_recursive(
	const cl_long *timep,		/* pg_time_t in original */
	long int *before_gmtoff,
	int *before_isdst,
	cl_long *boundary,			/* pg_time_t in original */
	long int *after_gmtoff,
	int *after_isdst,
	const tz_state *sp)			/* const pg_tz *tz in original */
{
	const tz_ttinfo *ttisp;
	int			i;
	int			j;
	const cl_long t = *timep;	/* pg_time_t in original */

	if (sp->timecnt == 0)
	{
		/* non-DST zone, use lowest-numbered standard type */
		i = 0;
		while (sp->ttis[i].tt_isdst)
			if (++i >= sp->typecnt)
			{
				i = 0;
				break;
			}
		ttisp = &sp->ttis[i];
		*before_gmtoff = ttisp->tt_gmtoff;
		*before_isdst = ttisp->tt_isdst;
		return 0;
	}
#if 0
	if ((sp->goback && t < sp->ats[0]) ||
		(sp->goahead && t > sp->ats[sp->timecnt - 1]))
	{
		/* For values outside the transition table, extrapolate */
		cl_long	newt = t;	/* pg_time_t in original */
		cl_long	seconds;	/* pg_time_t in original */
		cl_long	tcycles;	/* pg_time_t in original */
		cl_long	icycles;
		int		result;

		if (t < sp->ats[0])
			seconds = sp->ats[0] - t;
		else
			seconds = t - sp->ats[sp->timecnt - 1];
		--seconds;
		tcycles = seconds / YEARSPERREPEAT / AVGSECSPERYEAR;
		++tcycles;
		icycles = tcycles;
		if (tcycles - icycles >= 1 || icycles - tcycles >= 1)
			return -1;
		seconds = icycles;
		seconds *= YEARSPERREPEAT;
		seconds *= AVGSECSPERYEAR;
		if (t < sp->ats[0])
			newt += seconds;
		else
			newt -= seconds;
		if (newt < sp->ats[0] ||
			newt > sp->ats[sp->timecnt - 1])
			return -1;			/* "cannot happen" */

		result = pg_next_dst_boundary(&newt, before_gmtoff,
									  before_isdst,
									  boundary,
									  after_gmtoff,
									  after_isdst,
									  tz);
		if (t < sp->ats[0])
			*boundary -= seconds;
		else
			*boundary += seconds;
		return result;
	}
#else
	/* Don't recursive call in this function. */
	assert(!((sp->goback && t < sp->ats[0]) ||
			 (sp->goahead && t > sp->ats[sp->timecnt - 1])));
#endif

	if (t >= sp->ats[sp->timecnt - 1])
	{
		/* No known transition > t, so use last known segment's type */
		i = sp->types[sp->timecnt - 1];
		ttisp = &sp->ttis[i];
		*before_gmtoff = ttisp->tt_gmtoff;
		*before_isdst = ttisp->tt_isdst;
		return 0;
	}
	if (t < sp->ats[0])
	{
		/* For "before", use lowest-numbered standard type */
		i = 0;
		while (sp->ttis[i].tt_isdst)
			if (++i >= sp->typecnt)
			{
				i = 0;
				break;
			}
		ttisp = &sp->ttis[i];
		*before_gmtoff = ttisp->tt_gmtoff;
		*before_isdst = ttisp->tt_isdst;
		*boundary = sp->ats[0];
		/* And for "after", use the first segment's type */
		i = sp->types[0];
		ttisp = &sp->ttis[i];
		*after_gmtoff = ttisp->tt_gmtoff;
		*after_isdst = ttisp->tt_isdst;
		return 1;
	}
	/* Else search to find the boundary following t */
	{
		int			lo = 1;
		int			hi = sp->timecnt - 1;

		while (lo < hi)
		{
			int			mid = (lo + hi) >> 1;

			if (t < sp->ats[mid])
				hi = mid;
			else
				lo = mid + 1;
		}
		i = lo;
	}
	j = sp->types[i - 1];
	ttisp = &sp->ttis[j];
	*before_gmtoff = ttisp->tt_gmtoff;
	*before_isdst = ttisp->tt_isdst;
	*boundary = sp->ats[i];
	j = sp->types[i];
	ttisp = &sp->ttis[j];
	*after_gmtoff = ttisp->tt_gmtoff;
	*after_isdst = ttisp->tt_isdst;
	return 1;
}

STATIC_INLINE(int)
pg_next_dst_boundary(const cl_long *timep,	/* pg_time_t in original */
					 long int *before_gmtoff,
					 int *before_isdst,
					 cl_long *boundary,		/* pg_time_t in original */
					 long int *after_gmtoff,
					 int *after_isdst,
					 const tz_state *sp)	/* const pg_tz *tz in original */
{
	const tz_ttinfo *ttisp;
	int			i;
	int			j;
	const cl_long t = *timep;	/* pg_time_t in original */

	if (sp->timecnt == 0)
	{
		/* non-DST zone, use lowest-numbered standard type */
		i = 0;
		while (sp->ttis[i].tt_isdst)
			if (++i >= sp->typecnt)
			{
				i = 0;
				break;
			}
		ttisp = &sp->ttis[i];
		*before_gmtoff = ttisp->tt_gmtoff;
		*before_isdst = ttisp->tt_isdst;
		return 0;
	}
	if ((sp->goback && t < sp->ats[0]) ||
		(sp->goahead && t > sp->ats[sp->timecnt - 1]))
	{
		/* For values outside the transition table, extrapolate */
		cl_long	newt = t;	/* pg_time_t in original */
		cl_long	seconds;	/* pg_time_t in original */
		cl_long	tcycles;	/* pg_time_t in original */
		cl_long	icycles;
		int		result;

		if (t < sp->ats[0])
			seconds = sp->ats[0] - t;
		else
			seconds = t - sp->ats[sp->timecnt - 1];
		--seconds;
		tcycles = seconds / YEARSPERREPEAT / AVGSECSPERYEAR;
		++tcycles;
		icycles = tcycles;
		if (tcycles - icycles >= 1 || icycles - tcycles >= 1)
			return -1;
		seconds = icycles;
		seconds *= YEARSPERREPEAT;
		seconds *= AVGSECSPERYEAR;
		if (t < sp->ats[0])
			newt += seconds;
		else
			newt -= seconds;
		if (newt < sp->ats[0] ||
			newt > sp->ats[sp->timecnt - 1])
			return -1;			/* "cannot happen" */

		result = pg_next_dst_boundary_no_recursive(&newt, before_gmtoff,
												   before_isdst,
												   boundary,
												   after_gmtoff,
												   after_isdst,
												   sp);
		if (t < sp->ats[0])
			*boundary -= seconds;
		else
			*boundary += seconds;
		return result;
	}

	if (t >= sp->ats[sp->timecnt - 1])
	{
		/* No known transition > t, so use last known segment's type */
		i = sp->types[sp->timecnt - 1];
		ttisp = &sp->ttis[i];
		*before_gmtoff = ttisp->tt_gmtoff;
		*before_isdst = ttisp->tt_isdst;
		return 0;
	}
	if (t < sp->ats[0])
	{
		/* For "before", use lowest-numbered standard type */
		i = 0;
		while (sp->ttis[i].tt_isdst)
			if (++i >= sp->typecnt)
			{
				i = 0;
				break;
			}
		ttisp = &sp->ttis[i];
		*before_gmtoff = ttisp->tt_gmtoff;
		*before_isdst = ttisp->tt_isdst;
		*boundary = sp->ats[0];
		/* And for "after", use the first segment's type */
		i = sp->types[0];
		ttisp = &sp->ttis[i];
		*after_gmtoff = ttisp->tt_gmtoff;
		*after_isdst = ttisp->tt_isdst;
		return 1;
	}
	/* Else search to find the boundary following t */
	{
		int			lo = 1;
		int			hi = sp->timecnt - 1;

		while (lo < hi)
		{
			int			mid = (lo + hi) >> 1;

			if (t < sp->ats[mid])
				hi = mid;
			else
				lo = mid + 1;
		}
		i = lo;
	}
	j = sp->types[i - 1];
	ttisp = &sp->ttis[j];
	*before_gmtoff = ttisp->tt_gmtoff;
	*before_isdst = ttisp->tt_isdst;
	*boundary = sp->ats[i];
	j = sp->types[i];
	ttisp = &sp->ttis[j];
	*after_gmtoff = ttisp->tt_gmtoff;
	*after_isdst = ttisp->tt_isdst;
	return 1;
}

STATIC_INLINE(cl_int)
DetermineTimeZoneOffset(struct pg_tm *tm,
						const tz_state *sp)	/* pg_tz *tzp in original */
{
	cl_long t;					/* pg_time_t in original */

	cl_long	*tp = &t;			/* pg_time_t in original */

	int		date, sec;
	cl_long	day, mytime, boundary; /* pg_time_t in original */
	cl_long	prevtime, beforetime, aftertime; /* pg_time_t in original */
	long	before_gmtoff,after_gmtoff;
	int		before_isdst, after_isdst;
	int		res;

	/*
	 * First, generate the pg_time_t value corresponding to the given
	 * y/m/d/h/m/s taken as GMT time.  If this overflows, punt and decide the
	 * timezone is GMT.  (For a valid Julian date, integer overflow should be
	 * impossible with 64-bit pg_time_t, but let's check for safety.)
	 */
	if (!IS_VALID_JULIAN(tm->tm_year, tm->tm_mon, tm->tm_mday))
		goto overflow;
	date = date2j(tm->tm_year, tm->tm_mon, tm->tm_mday) - UNIX_EPOCH_JDATE;

	day = ((cl_long) date) * SECS_PER_DAY;
	if (day / SECS_PER_DAY != date)
		goto overflow;
	sec = (tm->tm_sec +
		   (tm->tm_min + tm->tm_hour * MINS_PER_HOUR) * SECS_PER_MINUTE);
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
							   sp);
	if (res < 0)
		goto overflow;			/* failure? */

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
	 * standard time; which does happen, eg Europe/Moscow in Oct 2014.)
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

/* simplified version; no timezone support now */
STATIC_INLINE(cl_bool)
timestamp2tm(Timestamp dt, int *tzp, struct pg_tm *tm, fsec_t *fsec,
			 const tz_state *sp)	/* pg_tz *attimezone in original */
{
	cl_long		date;	/* Timestamp in original */
	cl_long		time;	/* Timestamp in original */
	cl_long		utime;	/* pg_time_t in original */

	/* Use session timezone if caller asks for default */
    if (sp == NULL)
        sp = &session_timezone_state;

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
	if (tzp == NULL) {
		tm->tm_isdst = -1;
		tm->tm_gmtoff = 0;

		return true;
	}

	/*
     * If the time falls within the range of pg_time_t, use pg_localtime() to
     * rotate to the local time zone.
     *
     * First, convert to an integral timestamp, avoiding possibly
     * platform-specific roundoff-in-wrong-direction errors, and adjust to
     * Unix epoch.  Then see if we can convert to pg_time_t without loss. This
     * coding avoids hardwiring any assumptions about the width of pg_time_t,
     * so it should behave sanely on machines without int64.
     */
    dt = (dt - *fsec) / USECS_PER_SEC +
        (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * SECS_PER_DAY;

    utime = (cl_long) dt;
    if ((Timestamp) utime == dt)
    {
        struct pg_tm tx;

		pg_localtime(&utime, &tx, sp);

        tm->tm_year = tx.tm_year + 1900;
        tm->tm_mon = tx.tm_mon + 1;
        tm->tm_mday = tx.tm_mday;
        tm->tm_hour = tx.tm_hour;
        tm->tm_min = tx.tm_min;
        tm->tm_sec = tx.tm_sec;
        tm->tm_isdst = tx.tm_isdst;
        tm->tm_gmtoff = tx.tm_gmtoff;
        *tzp = -tm->tm_gmtoff;
    }
    else
    {
        /*
         * When out of range of pg_time_t, treat as GMT
         */
        *tzp = 0;
        /* Mark this as *no* time zone available */
        tm->tm_isdst = -1;
        tm->tm_gmtoff = 0;
    }

	return true;
}

STATIC_INLINE(cl_bool)
tm2timestamp(struct pg_tm * tm, fsec_t fsec, int *tzp, Timestamp *result)
{
    TimeOffset  date;
    TimeOffset  time;

    /* Julian day routines are not correct for negative Julian days */
    if (!IS_VALID_JULIAN(tm->tm_year, tm->tm_mon, tm->tm_mday))
    {
        *result = 0;
		/* keep compiler quiet */
        return false;
    }

    date = date2j(tm->tm_year, tm->tm_mon, tm->tm_mday) - POSTGRES_EPOCH_JDATE;
    time = time2t(tm->tm_hour, tm->tm_min, tm->tm_sec, fsec);

    *result = date * USECS_PER_DAY + time;
    /* check for major overflow */
    if ((*result - time) / USECS_PER_DAY != date)
    {
        *result = 0;		/* keep compiler quiet */
        return false;
    }
    /* check for just-barely overflow (okay except time-of-day wraps) */
    /* caution: we want to allow 1999-12-31 24:00:00 */
    if ((*result < 0 && date > 0) ||
        (*result > 0 && date < -1))
    {
        *result = 0;
		/* keep compiler quiet */
        return false;
    }
	
    if (tzp != NULL)
        *result = dt2local(*result, -(*tzp));

    return true;
}

/*
 * date2timestamptz
 *
 * It translates pg_date_t to pg_timestamptz_t based on the session
 * timezone information (session_timezone_state)
 */
STATIC_FUNCTION(pg_timestamptz_t)
date2timestamptz(kern_context *kcxt, pg_date_t arg)
{
	pg_timestamptz_t	result;
	struct pg_tm	tm;
	int				tz;

	if (arg.isnull)
	{
		result.isnull = true;
	}
	else if (DATE_IS_NOBEGIN(arg.value))
	{
		result.isnull = false;
		TIMESTAMP_NOBEGIN(result.value);
	}
	else if (DATE_IS_NOEND(arg.value))
	{
		result.isnull = false;
		TIMESTAMP_NOEND(result.value);
	}
	else
	{
        j2date(arg.value + POSTGRES_EPOCH_JDATE,
			   &tm.tm_year, &tm.tm_mon, &tm.tm_mday);
        tm.tm_hour = 0;
        tm.tm_min = 0;
        tm.tm_sec = 0;
        tz = DetermineTimeZoneOffset(&tm, &session_timezone_state);

		result.isnull = false;
		result.value = arg.value * USECS_PER_DAY + tz * USECS_PER_SEC;
        /* Date's range is wider than timestamp's, so check for overflow */
        if ((result.value - tz * USECS_PER_SEC) / USECS_PER_DAY != arg.value)
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return result;
}

/*
 * timestamp2timestamptz
 *
 * It translates pg_timestamp_t to pg_timestamptz_t based on the session
 * timezone information (session_timezone_state)
 */
STATIC_FUNCTION(pg_timestamptz_t)
timestamp2timestamptz(kern_context *kcxt, pg_timestamp_t arg)
{
	pg_timestamptz_t	result;
	struct pg_tm		tm;
	fsec_t				fsec;
	int					tz;

	if (arg.isnull)
	{
		result.isnull = true;
	}
    else if (TIMESTAMP_NOT_FINITE(arg.value))
	{
		result.isnull = false;
        result.value  = arg.value;
	}
	else if (!timestamp2tm(arg.value, NULL, &tm, &fsec, NULL))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	else
	{
        tz = DetermineTimeZoneOffset(&tm, &session_timezone_state);
		if (!tm2timestamp(&tm, fsec, &tz, &result.value))
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
		else
		{
			result.isnull = false;
		}
	}

	return result;
}

/*
 * GetCurrentDateTime()
 *
 * Get the transaction start time ("now()") broken down as a struct pg_tm.
 */
STATIC_INLINE(void)
GetCurrentDateTime(kern_context *kcxt, struct pg_tm * tm)
{
	int		tz;
	fsec_t	fsec;

	// TimestampTz dt = GetCurrentTransactionStartTimestamp();
	TimestampTz dt = kcxt->kparams->xactStartTimestamp;

	timestamp2tm(dt, &tz, tm, &fsec, NULL);
    /* Note: don't pass NULL tzp to timestamp2tm; affects behavior */
}

#ifdef NOT_USED
/* AdjustTimeForTypmod()
 * Force the precision of the time value to a specified value.
 * Uses *exactly* the same code as in AdjustTimestampForTypemod()
 * but we make a separate copy because those types do not
 * have a fundamental tie together but rather a coincidence of
 * implementation. - thomas
 */
static const cl_long TimeScales[MAX_TIME_PRECISION + 1] =
{
	INT64CONST(1000000),
	INT64CONST(100000),
	INT64CONST(10000),
	INT64CONST(1000),
	INT64CONST(100),
	INT64CONST(10),
	INT64CONST(1)
};

static const cl_long TimeOffsets[MAX_TIME_PRECISION + 1] =
{
	INT64CONST(500000),
	INT64CONST(50000),
	INT64CONST(5000),
	INT64CONST(500),
	INT64CONST(50),
	INT64CONST(5),
	INT64CONST(0)
};

STATIC_INLINE(void)
AdjustTimeForTypmod(TimeADT *time, cl_int typmod)
{
	if (typmod >= 0 && typmod <= MAX_TIME_PRECISION)
	{
		/*
		 * Note: this round-to-nearest code is not completely consistent about
		 * rounding values that are exactly halfway between integral values.
		 * On most platforms, rint() will implement round-to-nearest-even, but
		 * the integer code always rounds up (away from zero).  Is it worth
		 * trying to be consistent?
		 */
		if (*time >= 0LL)
			*time = ((*time + TimeOffsets[typmod]) / TimeScales[typmod]) *
				TimeScales[typmod];
		else
			*time = -((((-*time) + TimeOffsets[typmod]) / TimeScales[typmod]) *
					  TimeScales[typmod]);
	}
}
#endif

STATIC_INLINE(Interval)
interval_justify_hours(Interval span)
{
	Interval	result;
	TimeOffset	wholeday;

	result.month = span.month;
	result.day	 = span.day;
	result.time	 = span.time;

	TMODULO(result.time, wholeday, USECS_PER_DAY);
	result.day += wholeday;	/* could overflow... */

	if (result.day > 0 && result.time < 0)
	{
		result.time += USECS_PER_DAY;
		result.day--;
	}
	else if (result.day < 0 && result.time > 0)
	{
		result.time -= USECS_PER_DAY;
		result.day++;
	}

	return result;
}

/* ---------------------------------------------------------------
 *
 * Type cast functions
 *
 * --------------------------------------------------------------- */
STATIC_FUNCTION(pg_date_t)
pgfn_timestamp_date(kern_context *kcxt, pg_timestamp_t arg1)
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
	else if (!timestamp2tm(arg1.value, NULL, &tm, &fsec, NULL))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	else
	{
		result.isnull = false;
		result.value  = (date2j(tm.tm_year, tm.tm_mon, tm.tm_mday)
						 - POSTGRES_EPOCH_JDATE);
	}
	return result;
}

/*
 * Data cast functions related to timezonetz
 */
STATIC_FUNCTION(pg_date_t)
pgfn_timestamptz_date(kern_context *kcxt, pg_timestamptz_t arg1)
{
	pg_date_t		result;
	struct pg_tm	tm;
	fsec_t			fsec;
	int				tz;

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
	else if (!timestamp2tm(arg1.value, &tz, &tm, &fsec, NULL))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	else
	{
		result.isnull = false;
		result.value  = (date2j(tm.tm_year, tm.tm_mon, tm.tm_mday)
						 - POSTGRES_EPOCH_JDATE);
	}
	return result;
}

STATIC_FUNCTION(pg_time_t)
pgfn_timetz_time(kern_context *kcxt, pg_timetz_t arg1)
{
	pg_time_t result;

	if (arg1.isnull) 
		result.isnull = true;
	else {
		result.isnull = false;
		result.value = arg1.value.time;
	}

	return result;
}

STATIC_FUNCTION(pg_time_t)
pgfn_timestamp_time(kern_context *kcxt, pg_timestamp_t arg1)
{
	pg_time_t		result;
	struct pg_tm	tm;
	fsec_t			fsec;

	if (arg1.isnull)
		result.isnull = true;
	else if (TIMESTAMP_NOT_FINITE(arg1.value))
		result.isnull = true;
	else if (!timestamp2tm(arg1.value, NULL, &tm, &fsec, NULL))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
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

STATIC_FUNCTION(pg_time_t)
pgfn_timestamptz_time(kern_context *kcxt, pg_timestamptz_t arg1)
{
	pg_time_t		result;
	struct pg_tm	tm;
	fsec_t			fsec;
	int				tz;

	if (arg1.isnull)
		result.isnull = true;
	else if (TIMESTAMP_NOT_FINITE(arg1.value))
		result.isnull = true;
	else if (!timestamp2tm(arg1.value, &tz, &tm, &fsec, NULL))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
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

STATIC_FUNCTION(pg_timetz_t)
pgfn_time_timetz(kern_context *kcxt, pg_time_t arg1)
{
	pg_timetz_t		result;
	struct pg_tm	tm;
	fsec_t			fsec;
	int				tz;


	if (arg1.isnull)
		result.isnull = true;
	else
	{
		GetCurrentDateTime(kcxt, &tm);
		time2tm(arg1.value, &tm, &fsec);
		tz = DetermineTimeZoneOffset(&tm, &session_timezone_state);

		result.isnull     = false;
		result.value.time = arg1.value;
		result.value.zone = tz;
	}

	return result;
}

STATIC_FUNCTION(pg_timetz_t)
pgfn_timestamptz_timetz(kern_context *kcxt, pg_timestamptz_t arg1)
{
	pg_timetz_t		result;
	struct pg_tm	tm;
	int				tz;
	fsec_t			fsec;


	if (arg1.isnull)
		result.isnull = true;
	else if (TIMESTAMP_NOT_FINITE(arg1.value))
		result.isnull = true;
	else if (timestamp2tm(arg1.value, &tz, &tm, &fsec, NULL) != 0)
	{
		// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
	}
	else
	{
		tm2timetz(&tm, fsec, tz, &(result.value));
		result.isnull = false;
	}

	return result;
}

#ifdef NOT_USED
STATIC_FUNCTION(pg_timetz_t)
pgfn_timetz_scale(kern_context *kcxt, pg_timetz_t arg1, pg_int4_t arg2)
{
	pg_timetz_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull     = false;
		result.value.time = arg1.value.time;
		result.value.zone = arg1.value.zone;

		AdjustTimeForTypmod(&(result.value.time), arg2.value);
	}

	return result;
}
#endif

STATIC_FUNCTION(pg_timestamp_t)
pgfn_date_timestamp(kern_context *kcxt, pg_date_t arg1)
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
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_timestamp_t)
pgfn_timestamptz_timestamp(kern_context *kcxt, pg_timestamptz_t arg1)
{
	pg_timestamp_t	result;
	struct pg_tm	tm;
	fsec_t			fsec;
	int				tz;

	if (arg1.isnull)
	{
		result.isnull = true;
	}
	else if (TIMESTAMP_NOT_FINITE(arg1.value))
	{
		result.isnull = false;
        result.value  = arg1.value;
	}
	else if (!timestamp2tm(arg1.value, &tz, &tm, &fsec, NULL))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	else if (!tm2timestamp(&tm, fsec, NULL, &result.value))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	else
	{
		result.isnull = false;
	}

	return result;
}

STATIC_FUNCTION(pg_timestamptz_t)
pgfn_date_timestamptz(kern_context *kcxt, pg_date_t arg1)
{
	return date2timestamptz(kcxt, arg1);
}

STATIC_FUNCTION(pg_timestamptz_t)
pgfn_timestamp_timestamptz(kern_context *kcxt, pg_timestamp_t arg1)
{
	return timestamp2timestamptz(kcxt, arg1);
}

/*
 * Time/Date operators
 */
STATIC_FUNCTION(pg_date_t)
pgfn_date_pli(kern_context *kcxt, pg_date_t arg1, pg_int4_t arg2)
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

STATIC_FUNCTION(pg_date_t)
pgfn_date_mii(kern_context *kcxt, pg_date_t arg1, pg_int4_t arg2)
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

STATIC_FUNCTION(pg_int4_t)
pgfn_date_mi(kern_context *kcxt, pg_date_t arg1, pg_date_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else if (DATE_NOT_FINITE(arg1.value) || DATE_NOT_FINITE(arg2.value))
	{
		result.isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	else
	{
		result.isnull = false;
		result.value = (cl_int)(arg1.value - arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_timestamp_t)
pgfn_datetime_pl(kern_context *kcxt, pg_date_t arg1, pg_time_t arg2)
{
	pg_timestamp_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result = pgfn_date_timestamp(kcxt, arg1);
		if (!TIMESTAMP_NOT_FINITE(result.value))
			result.value += arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_date_t)
pgfn_integer_pl_date(kern_context *kcxt, pg_int4_t arg1, pg_date_t arg2)
{
	return pgfn_date_pli(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_timestamp_t)
pgfn_timedate_pl(kern_context *kcxt, pg_time_t arg1, pg_date_t arg2)
{
	return pgfn_datetime_pl(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_interval_t)
pgfn_time_mi_time(kern_context *kcxt, pg_time_t arg1, pg_time_t arg2)
{
	pg_interval_t result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else 
	{
		result.isnull      = false;
		result.value.month = 0;
		result.value.day   = 0;
		result.value.time  = arg1.value - arg2.value;
	}

	return result;
}

STATIC_FUNCTION(pg_interval_t)
pgfn_timestamp_mi(kern_context *kcxt, pg_timestamp_t arg1, pg_timestamp_t arg2)
{
	pg_interval_t result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else if (TIMESTAMP_NOT_FINITE(arg1.value) || 
			 TIMESTAMP_NOT_FINITE(arg2.value))
	{
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
	}
	else 
	{
		result.isnull      = false;
		result.value.month = 0;
		result.value.day   = 0;
		result.value.time  = arg1.value - arg2.value;

		/*----------
		 *	This is wrong, but removing it breaks a lot of regression tests.
		 *	For example:
		 *
		 *	test=> SET timezone = 'EST5EDT';
		 *	test=> SELECT
		 *	test-> ('2005-10-30 13:22:00-05'::timestamptz -
		 *	test(>	'2005-10-29 13:22:00-04'::timestamptz);
		 *	?column?
		 *	----------------
		 *	 1 day 01:00:00
		 *	 (1 row)
		 *
		 *	so adding that to the first timestamp gets:
		 *
		 *	test=> SELECT
		 *	test-> ('2005-10-29 13:22:00-04'::timestamptz +
		 *	test(> ('2005-10-30 13:22:00-05'::timestamptz -
		 *	test(>  '2005-10-29 13:22:00-04'::timestamptz)) at time zone 'EST';
		 *		timezone
		 *	--------------------
		 *	2005-10-30 14:22:00
		 *	(1 row)
		 *----------
		 */
		result.value = interval_justify_hours(result.value);
	}

	return result;
}

STATIC_FUNCTION(pg_timetz_t)
pgfn_timetz_pl_interval(kern_context *kcxt,
						pg_timetz_t arg1, pg_interval_t arg2)
{
    pg_timetz_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value.time = arg1.value.time + arg2.value.time;
		result.value.time -= result.value.time / USECS_PER_DAY * USECS_PER_DAY;
		if (result.value.time < INT64CONST(0))
			result.value.time += USECS_PER_DAY;

		result.value.zone = arg1.value.zone;
	}

	return result;
}

STATIC_FUNCTION(pg_timetz_t)
pgfn_timetz_mi_interval(kern_context *kcxt, pg_timetz_t arg1, pg_interval_t arg2)
{
	arg2.value.time = - arg2.value.time;

	return pgfn_timetz_pl_interval(kcxt, arg1, arg2);
}

STATIC_FUNCTION(pg_timestamptz_t)
pgfn_timestamptz_pl_interval(kern_context *kcxt,
							 pg_timestamptz_t arg1,
							 pg_interval_t arg2)
{
	pg_timestamptz_t	result;

	if (arg1.isnull || arg2.isnull)
	{
		result.isnull = true;
		return result;
	}

	if (TIMESTAMP_NOT_FINITE(arg1.value))
		result = arg1;
	else
	{
		if (arg2.value.month != 0)
		{
			struct pg_tm tm;
			fsec_t fsec;

			if (timestamp2tm(arg1.value, NULL, &tm, &fsec, NULL) != 0)
			{
				// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
				result.isnull = true;
				return result;
			}

			tm.tm_mon += arg2.value.month;
			if (tm.tm_mon > MONTHS_PER_YEAR)
			{
				tm.tm_year +=  (tm.tm_mon - 1) / MONTHS_PER_YEAR;
				tm.tm_mon   = ((tm.tm_mon - 1) % MONTHS_PER_YEAR) + 1;
			}
			else if (tm.tm_mon < 1)
			{
				tm.tm_year += tm.tm_mon / MONTHS_PER_YEAR - 1;
				tm.tm_mon   = tm.tm_mon % MONTHS_PER_YEAR + MONTHS_PER_YEAR;
			}

			/* adjust for end of month boundary problems... */
			if (tm.tm_mday > day_tab[isleap(tm.tm_year)][tm.tm_mon - 1])
				tm.tm_mday = (day_tab[isleap(tm.tm_year)][tm.tm_mon - 1]);

			if (tm2timestamp(&tm, fsec, NULL, &arg1.value) != 0)
			{
				// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
				result.isnull = true;
				return result;
			}
		}

		if (arg2.value.day != 0)
		{
			struct pg_tm tm;
			fsec_t fsec;
			int julian;

			if (timestamp2tm(arg1.value, NULL, &tm, &fsec, NULL) != 0)
			{
				// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
				result.isnull = true;
				return result;
			}

			/* Add days by converting to and from julian */
			julian = date2j(tm.tm_year, tm.tm_mon, tm.tm_mday) + arg2.value.day;
			j2date(julian, &tm.tm_year, &tm.tm_mon, &tm.tm_mday);

			if (tm2timestamp(&tm, fsec, NULL, &arg1.value) != 0)
			{
				// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE;
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
				result.isnull = true;
				return result;
			}
		}
		
		result.isnull = false;
		result.value  = arg1.value + arg2.value.time;
	}

	return result;
}

STATIC_FUNCTION(pg_timestamptz_t)
pgfn_timestamptz_mi_interval(kern_context *kcxt,
							 pg_timestamptz_t arg1,
							 pg_interval_t arg2)
{
	arg2.value.month = - arg2.value.month;
	arg2.value.day   = - arg2.value.day;
	arg2.value.time  = - arg2.value.time;

	return pgfn_timestamptz_pl_interval(kcxt, arg1, arg2);
}

STATIC_FUNCTION(pg_interval_t)
pgfn_interval_um(kern_context *kcxt, pg_interval_t arg1)
{
	pg_interval_t result;

	if (arg1.isnull)
	{
		result.isnull = true;
		return result;
	}

	result.value.time = - arg1.value.time;
	/* overflow check copied from int4um */
	if (arg1.value.time != 0 && SAMESIGN(result.value.time, arg1.value.time))
	{
		// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	result.value.day = - arg1.value.day;
	if (arg1.value.day != 0 && SAMESIGN(result.value.day, arg1.value.day))
	{
		// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	result.value.month = - arg1.value.month;
	if (arg1.value.month != 0 &&
		SAMESIGN(result.value.month, arg1.value.month))
	{
		// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	result.isnull = false;

	return result;
}

STATIC_FUNCTION(pg_interval_t)
pgfn_interval_pl(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_interval_t result;

	if (arg1.isnull || arg2.isnull)
	{
		result.isnull = true;
		return result;
	}
	
	result.value.month = arg1.value.month + arg2.value.month;
	/* overflow check copied from int4pl */
	if (SAMESIGN(arg1.value.month, arg2.value.month) &&
		!SAMESIGN(result.value.month, arg1.value.month))
	{
		// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	result.value.day = arg1.value.day + arg2.value.day;
	if (SAMESIGN(arg1.value.day, arg2.value.day) &&
		!SAMESIGN(result.value.day, arg1.value.day))
	{
		// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	result.value.time = arg1.value.time + arg2.value.time;
	if (SAMESIGN(arg1.value.time, arg2.value.time) &&
		!SAMESIGN(result.value.time, arg1.value.time))
	{
		// ERRCODE_DATETIME_VALUE_OUT_OF_RANGE
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	result.isnull = false;

	return result;
}

STATIC_FUNCTION(pg_interval_t)
pgfn_interval_mi(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	arg2.value.time  = - arg2.value.time;
	arg2.value.day   = - arg2.value.day;
	arg2.value.month = - arg2.value.month;

	return pgfn_interval_pl(kcxt, arg1, arg2);
}

STATIC_FUNCTION(pg_timestamptz_t)
pgfn_datetimetz_timestamptz(kern_context *kcxt, pg_date_t arg1, pg_timetz_t arg2)
{
    pg_timestamptz_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;

		if (DATE_IS_NOBEGIN(arg1.value))
			TIMESTAMP_NOBEGIN(result.value);
		else if (DATE_IS_NOEND(arg1.value))
			TIMESTAMP_NOEND(result.value);
		else
			result.value = arg1.value * USECS_PER_DAY 
				+ arg2.value.time + arg2.value.zone * USECS_PER_SEC;
	}

	return result;
}

/*
 * Date comparison
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_date_eq_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(kcxt, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value == arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_ne_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(kcxt, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value != arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_lt_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(kcxt, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value < arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_le_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(kcxt, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value <= arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_gt_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(kcxt, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value > arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_ge_timestamp(kern_context *kcxt,
					   pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(kcxt, arg1);

	if (dt1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(dt1.value >= arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_date_cmp_timestamp(kern_context *kcxt,
						pg_date_t arg1, pg_timestamp_t arg2)
{
	pg_int4_t		result;
	pg_timestamp_t	dt1 = pgfn_date_timestamp(kcxt, arg1);

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
 * Comparison between timetz
 */
STATIC_INLINE(cl_int)
timetz_cmp_internal(TimeTzADT arg1, TimeTzADT arg2)
{
	TimeOffset	t1 = arg1.time + arg1.zone * USECS_PER_SEC;
	TimeOffset	t2 = arg2.time + arg2.zone * USECS_PER_SEC;

	cl_int	result;

	if (t1 > t2)
		result = 1;
	else if (t1 < t2)
		result = -1;
	/*
	 * If same GMT time, sort by timezone; we only want to say that two
	 * timetz's are equal if both the time and zone parts are equal.
	 */
	else if (arg1.zone > arg2.zone)
		result = 1;
	else if (arg1.zone < arg2.zone)
		result = -1;
	else 
		result = 0;

    return result;	
}

STATIC_INLINE(cl_bool)
timetz_eq_internal(TimeTzADT arg1, TimeTzADT arg2)
{
	return (cl_bool)(timetz_cmp_internal(arg1, arg2) == 0);
}

STATIC_INLINE(cl_bool)
timetz_ne_internal(TimeTzADT arg1, TimeTzADT arg2)
{
	return (cl_bool)(timetz_cmp_internal(arg1, arg2) != 0);
}

STATIC_INLINE(cl_bool)
timetz_lt_internal(TimeTzADT arg1, TimeTzADT arg2)
{
	return (cl_bool)(timetz_cmp_internal(arg1, arg2) < 0);
}

STATIC_INLINE(cl_bool)
timetz_le_internal(TimeTzADT arg1, TimeTzADT arg2)
{
	return (cl_bool)(timetz_cmp_internal(arg1, arg2) <= 0);
}

STATIC_INLINE(cl_bool)
timetz_ge_internal(TimeTzADT arg1, TimeTzADT arg2)
{
	return (cl_bool)(timetz_cmp_internal(arg1, arg2) >= 0);
}

STATIC_INLINE(cl_bool)
timetz_gt_internal(TimeTzADT arg1, TimeTzADT arg2)
{
	return (cl_bool)(timetz_cmp_internal(arg1, arg2) > 0);
}


STATIC_FUNCTION(pg_bool_t)
pgfn_timetz_eq(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
{
	pg_bool_t	result;
	

	if (arg1.isnull || arg2.isnull) 
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = timetz_eq_internal(arg1.value, arg2.value);
	}	

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timetz_ne(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
{
	pg_bool_t	result;
	

	if (arg1.isnull || arg2.isnull) 
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = timetz_ne_internal(arg1.value, arg2.value);
	}	

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timetz_lt(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
{
	pg_bool_t	result;
	

	if (arg1.isnull || arg2.isnull) 
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = timetz_lt_internal(arg1.value, arg2.value);
	}	

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timetz_le(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
{
	pg_bool_t	result;
	

	if (arg1.isnull || arg2.isnull) 
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = timetz_le_internal(arg1.value, arg2.value);
	}	

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timetz_ge(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
{
	pg_bool_t	result;
	

	if (arg1.isnull || arg2.isnull) 
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = timetz_ge_internal(arg1.value, arg2.value);
	}	

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timetz_gt(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
{
	pg_bool_t	result;
	

	if (arg1.isnull || arg2.isnull) 
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = timetz_gt_internal(arg1.value, arg2.value);
	}	

	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_timetz_cmp(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
{
	pg_int4_t	result;


	if (arg1.isnull || arg2.isnull) 
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value  = timetz_cmp_internal(arg1.value, arg2.value);
	}

    return result;	
}

/*
 * Timestamp comparison
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_eq_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(kcxt, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value == dt2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_ne_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(kcxt, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value != dt2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_lt_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(kcxt, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value < dt2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_le_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(kcxt, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value <= dt2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_gt_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(kcxt, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value > dt2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_ge_date(kern_context *kcxt,
					   pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_bool_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(kcxt, arg2);

	if (arg1.isnull || dt2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(arg1.value >= dt2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_timestamp_cmp_date(kern_context *kcxt,
						pg_timestamp_t arg1, pg_date_t arg2)
{
	pg_int4_t		result;
	pg_timestamp_t	dt2 = pgfn_date_timestamp(kcxt, arg2);

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

/*
 * Comparison between date and timestamptz
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_date_lt_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = date2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value < arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_le_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = date2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value <= arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_eq_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = date2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value == arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_ge_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = date2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value >= arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_gt_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = date2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value > arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_date_ne_timestamptz(kern_context *kcxt,
						 pg_date_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = date2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value != arg2.value);
	}
	return result;
}

/*
 * Comparison between timestamptz and date
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_lt_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_gt_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_le_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_ge_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_eq_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_eq_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_ge_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_le_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_gt_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_lt_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_ne_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_ne_timestamptz(kcxt, arg2, arg1);
}

/*
 * Comparison between timestamp and timestamptz
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_lt_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = timestamp2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value < arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_le_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = timestamp2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value <= arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_eq_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = timestamp2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value == arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_ge_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = timestamp2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value >= arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_gt_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = timestamp2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value > arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamp_ne_timestamptz(kern_context *kcxt,
							  pg_timestamp_t arg1, pg_timestamptz_t arg2)
{
	pg_timestamptz_t temp;
	pg_bool_t	result;

	temp = timestamp2timestamptz(kcxt, arg1);
	if (temp.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_bool)(temp.value != arg2.value);
	}
	return result;
}

/*
 * Comparison between timestamptz and timestamp
 */
STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_lt_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_gt_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_le_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_ge_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_eq_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_eq_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_ge_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_le_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_gt_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_lt_timestamptz(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_timestamptz_ne_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_ne_timestamptz(kcxt, arg2, arg1);
}

/*
 * Comparison between pg_interval_t
 */
STATIC_INLINE(TimeOffset)
interval_cmp_value(const Interval interval)
{
	TimeOffset	span;

	span = interval.time;
	span += interval.month * INT64CONST(30) * USECS_PER_DAY;
	span += interval.day * INT64CONST(24) * USECS_PER_HOUR;

	return span;
}

STATIC_INLINE(cl_int)
interval_cmp_internal(Interval arg1, Interval arg2)
{
	TimeOffset	span1 = interval_cmp_value(arg1);
	TimeOffset	span2 = interval_cmp_value(arg2);

	return ((span1 < span2) ? -1 : (span1 > span2) ? 1 : 0);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_interval_eq(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value =
			(cl_bool)(interval_cmp_internal(arg1.value, arg2.value) == 0);
	}

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_interval_ne(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value =
			(cl_bool)(interval_cmp_internal(arg1.value, arg2.value) != 0);
	}

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_interval_lt(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value =
			(cl_bool)(interval_cmp_internal(arg1.value, arg2.value) < 0);
	}

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_interval_le(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value =
			(cl_bool)(interval_cmp_internal(arg1.value, arg2.value) <= 0);
	}

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_interval_ge(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value =
			(cl_bool)(interval_cmp_internal(arg1.value, arg2.value) >= 0);
	}

	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_interval_gt(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value =
			(cl_bool)(interval_cmp_internal(arg1.value, arg2.value) > 0);
	}

	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_interval_cmp(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = interval_cmp_internal(arg1.value, arg2.value);
	}

	return result;
}

/*
 * current date time function
 */
STATIC_FUNCTION(pg_timestamptz_t)
pgfn_now(kern_context *kcxt)
{
	pg_timestamptz_t	result;
	kern_parambuf	   *kparams = kcxt->kparams;

	result.value  = kparams->xactStartTimestamp;
	result.isnull = false;

	return result;
}

/*
 * overlaps() SQL functions
 *
 * NOTE: Even though overlaps() has more variations, inline_function()
 * preliminary break down combination with type-cast.
 */
#if 0
STATIC_INLINE(pg_bool_t)
overlaps_cl_long(kern_context *kcxt,
				 cl_long ts1, cl_bool ts1_isnull,
				 cl_long te1, cl_bool te1_isnull,
				 cl_long ts2, cl_bool ts2_isnull,
				 cl_long te2, cl_bool te2_isnull)
{
	pg_bool_t rc;

	/*
	 * If both endpoints of interval 1 are null, the result is null (unknown).
	 * If just one endpoint is null, take ts1 as the non-null one. Otherwise,
	 * take ts1 as the lesser endpoint.
	 */
	if (ts1_isnull)
	{
		if (te1_isnull)
		{
			rc.isnull = true;
			return rc;
		}
		/* swap null for non-null */
		ts1        = te1;
		ts1_isnull = false;
		te1_isnull = true;
	}
	else if (!te1_isnull)
	{
		if (ts1 > te1)
		{
			cl_long tmp = ts1;
			ts1 = te1;
			te1 = tmp;
		}
	}

	/* Likewise for interval 2. */
	if (ts2_isnull)
	{
		if (te2_isnull)
		{
			rc.isnull = true;
			return rc;
		}
		/* swap null for non-null */
		ts2        = te2;
		ts2_isnull = false;
		te2_isnull = true;
	}
	else if (!te2_isnull)
	{
		if (ts2 > te2)
		{
			cl_long tmp = ts2;
			ts2 = te2;
			te2 = tmp;
		}
	}

	/*
	 * At this point neither ts1 nor ts2 is null, so we can consider three
	 * cases: ts1 > ts2, ts1 < ts2, ts1 = ts2
	 */
	if (ts1 > ts2)
	{
		/*
		 * This case is ts1 < te2 OR te1 < te2, which may look redundant but
		 * in the presence of nulls it's not quite completely so.
		 */
		if (te2_isnull)
		{
			rc.isnull = true;
			return rc;
		}
		if (ts1 < te2)
		{
			rc.isnull = false;
			rc.value  = true;
			return rc;
		}
		if (te1_isnull)
		{
			rc.isnull = true;
			return rc;
		}

		/*
		 * If te1 is not null then we had ts1 <= te1 above, and we just found
		 * ts1 >= te2, hence te1 >= te2.
		 */
		rc.isnull = false;
		rc.value  = false;
		return rc;
	}
	else if (ts1 < ts2)
	{
		/* This case is ts2 < te1 OR te2 < te1 */
		if (te1_isnull)
		{
			rc.isnull = true;
			return rc;
		}
		if (ts2 < te1)
		{
			rc.isnull = false;
			rc.value  = true;
			return rc;
		}
		if (te2_isnull)
		{
			rc.isnull = true;
			return rc;
		}

		/*
		 * If te2 is not null then we had ts2 <= te2 above, and we just found
		 * ts2 >= te1, hence te2 >= te1.
		 */
		rc.isnull = false;
		rc.value  = false;
		return rc;
	}
	else
	{
		/*
		 * For ts1 = ts2 the spec says te1 <> te2 OR te1 = te2, which is a
		 * rather silly way of saying "true if both are nonnull, else null".
		 */
		if (te1_isnull || te2_isnull)
		{
			rc.isnull = true;
			return rc;
		}
		rc.isnull = false;
		rc.value  = true;
		return rc;
	}
}
#endif

#define OVERLAPS(type,cmp_gt_ops,cmp_lt_ops)				\
	STATIC_INLINE(pg_bool_t)								\
	overlaps_##type(kern_context *kcxt,						\
					type ts1, bool ts1_isnull,				\
					type te1, bool te1_isnull,				\
					type ts2, bool ts2_isnull,				\
					type te4, bool te4_isnull)				\
	{														\
		pg_bool_t result;									\
															\
		if (ts1_isnull)										\
		{													\
			if (te1_isnull)									\
			{												\
				result.isnull = true;						\
				return result;								\
			}												\
			ts1 = te1;										\
			ts1_isnull = false;								\
			te1_isnull = true;								\
		}													\
		else if (! te1_isnull)								\
		{													\
			if (cmp_gt_ops(ts1, te1)) 						\
			{												\
				type tmp = ts1;								\
				ts1 = te1;									\
				te1 = tmp;									\
			}												\
		}													\
															\
		if (ts2_isnull)										\
		{													\
			if (te4_isnull)									\
			{												\
				result.isnull = true;						\
				return result;								\
			}												\
			ts2 = te4;										\
			ts2_isnull = false;								\
			te4_isnull = true;								\
		}													\
		else if (! te4_isnull)								\
		{													\
			if (cmp_gt_ops(ts2, te4))						\
			{												\
				type tmp = ts2;								\
				ts2 = te4;									\
				te4 = tmp;									\
			}												\
		}													\
															\
		if (cmp_gt_ops(ts1, ts2))							\
		{													\
			if (te4_isnull)									\
			{												\
				result.isnull = true;						\
				return result;								\
			}												\
			if (cmp_lt_ops(ts1, te4))						\
			{												\
				result.isnull = false;						\
				result.value  = true;						\
				return result;								\
			}												\
			if (te1_isnull)									\
			{												\
				result.isnull = true;						\
				return result;								\
			}												\
															\
			result.isnull = false;							\
			result.value  = false;							\
			return result;									\
		}													\
		else												\
		{													\
			if (cmp_lt_ops(ts1, ts2))						\
			{												\
				if (te1_isnull)								\
				{											\
					result.isnull = true;					\
					return result;							\
				}											\
				if (cmp_lt_ops(ts2, te1))					\
				{											\
					result.isnull = false;					\
					result.value  = true;					\
					return result;							\
				}											\
				if (te4_isnull)								\
				{											\
					result.isnull = true;					\
					return result;							\
				}											\
				result.isnull = false;						\
				result.value  = false;						\
				return result;								\
			}												\
			else											\
			{												\
				if (te1_isnull || te4_isnull)				\
				{											\
					result.isnull = true;					\
					return result;							\
				}											\
				result.isnull = false;						\
				result.value  = true;						\
				return result;								\
			}												\
		}													\
	}

#define COMPARE_GT_OPS(x,y)	((x) > (y))
#define COMPARE_LT_OPS(x,y)	((x) < (y))

OVERLAPS(cl_long,COMPARE_GT_OPS,COMPARE_LT_OPS)
OVERLAPS(TimeTzADT,timetz_gt_internal,timetz_lt_internal)


STATIC_FUNCTION(pg_bool_t)
pgfn_overlaps_time(kern_context *kcxt,
				   pg_time_t arg1, pg_time_t arg2,
				   pg_time_t arg3, pg_time_t arg4)
{
  return overlaps_cl_long(kcxt,
						  arg1.value, arg1.isnull,
						  arg2.value, arg2.isnull,
						  arg3.value, arg3.isnull,
						  arg4.value, arg4.isnull);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_overlaps_timetz(kern_context *kcxt,
					 pg_timetz_t arg1, pg_timetz_t arg2,
					 pg_timetz_t arg3, pg_timetz_t arg4)
{
	return overlaps_TimeTzADT(kcxt,
							  arg1.value, arg1.isnull,
							  arg2.value, arg2.isnull,
							  arg3.value, arg3.isnull,
							  arg4.value, arg4.isnull);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_overlaps_timestamp(kern_context *kcxt,
						pg_timestamp_t arg1, pg_timestamp_t arg2,
						pg_timestamp_t arg3, pg_timestamp_t arg4)
{
  return overlaps_cl_long(kcxt,
						  arg1.value, arg1.isnull,
						  arg2.value, arg2.isnull,
						  arg3.value, arg3.isnull,
						  arg4.value, arg4.isnull);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_overlaps_timestamptz(kern_context *kcxt,
						  pg_timestamptz_t arg1, pg_timestamptz_t arg2,
						  pg_timestamptz_t arg3, pg_timestamptz_t arg4)
{
  return overlaps_cl_long(kcxt,
						  arg1.value, arg1.isnull,
						  arg2.value, arg2.isnull,
						  arg3.value, arg3.isnull,
						  arg4.value, arg4.isnull);
}

#else	/* __CUDACC__ */

#include "access/xact.h"
#include "pgtime.h"

/*
 * assign_timelib_session_info
 *
 * It constructs per-session information around cuda_timelib.h.
 * At this moment, items below has to be informed.
 * - session_timezone information
 */

/* copied from src/timezone/tzfile.h */
#define TZ_MAX_TIMES	1200
#define TZ_MAX_TYPES	256		/* Limited by what (uchar)'s can hold */
#define TZ_MAX_CHARS	50		/* Maximum number of abbreviation characters */
#define TZ_MAX_LEAPS	50		/* Maximum number of leap second corrections */

/* copied from src/timezone/pgtz.h */
#define BIGGEST(a, b)	(((a) > (b)) ? (a) : (b))

struct ttinfo
{								/* time type information */
	cl_long		tt_gmtoff;		/* UTC offset in seconds */
	cl_int		tt_isdst;		/* used to set tm_isdst */
	cl_int		tt_abbrind;		/* abbreviation list index */
	cl_int		tt_ttisstd;		/* TRUE if transition is std time */
	cl_int		tt_ttisgmt;		/* TRUE if transition is UTC */
};

struct lsinfo
{                               /* leap second information */
    pg_time_t   ls_trans;       /* transition time */
    long        ls_corr;        /* correction to apply */
};

struct state
{
	cl_int		leapcnt;
	cl_int		timecnt;
	cl_int		typecnt;
	cl_int		charcnt;
	cl_int		goback;
	cl_int		goahead;
	/* NOTE: pg_time_t has different meaning in GPU kernel */
    cl_long		ats[TZ_MAX_TIMES];
	cl_uchar	types[TZ_MAX_TIMES];
	struct ttinfo ttis[TZ_MAX_TYPES];
	cl_char		chars[BIGGEST(BIGGEST(TZ_MAX_CHARS + 1, 3 /* sizeof gmt */ ),
							  (2 * (TZ_STRLEN_MAX + 1)))];
	struct lsinfo lsis[TZ_MAX_LEAPS];
};

struct pg_tz
{
	/* TZname contains the canonically-cased name of the timezone */
	char			TZname[TZ_STRLEN_MAX + 1];
	struct state	state;
};

STATIC_INLINE(void)
assign_timelib_session_info(StringInfo buf)
{
	const struct state *sp;
	int			i;

	appendStringInfo(
		buf,
		"#ifdef __CUDACC__\n"
		"/* ================================================\n"
		" * session information for cuda_timelib.h\n"
		" * ================================================ */\n");

	sp = &session_timezone->state;
	/*
	 * type declaration of tz_state structure
	 */
	appendStringInfo(
		buf,
		"typedef struct {\n"
		"        cl_long     ls_trans; /* pg_time_t in original */ \n"
		"        long        ls_corr;\n"
		"} tz_lsinfo;\n"
		"\n"
		"typedef struct {\n"
		"        cl_long     tt_gmtoff;\n"
		"        cl_int      tt_isdst;\n"
		"        cl_int      tt_abbrind;\n"
		"        cl_int      tt_ttisstd;\n"
		"        cl_int      tt_ttisgmt;\n"
		"} tz_ttinfo;\n"
		"\n"
		"typedef struct {\n"
		"    cl_int          leapcnt;\n"
		"    cl_int          timecnt;\n"
		"    cl_int          typecnt;\n"
		"    cl_int          charcnt;\n"
		"    cl_int          goback;\n"
		"    cl_int          goahead;\n"
		"    cl_long         ats[%d];\n"
		"    cl_uchar        types[%d];\n"
		"    tz_ttinfo       ttis[%d];\n"
		"    /* GPU kernel does not use chars[] */\n"
		"    tz_lsinfo       lsis[%d];\n"
		"} tz_state;\n"
		"\n",
		sp->timecnt <= 0 ? 1 : sp->timecnt,
		sp->timecnt <= 0 ? 1 : sp->timecnt,
		sp->typecnt,
		sp->leapcnt <= 0 ? 1 : sp->leapcnt
		);
	/*
	 * initializer of session_timezone_state
	 */
	appendStringInfo(
        buf,
		"static const tz_state session_timezone_state =\n"
		"{\n"
		"    %d,    /* leapcnt */\n"
		"    %d,    /* timecnt */\n"
		"    %d,    /* typecnt */\n"
		"    %d,    /* charcnt */\n"
		"    %d,    /* goback */\n"
		"    %d,    /* goahead */\n"
		"    {      /* ats[] */\n",
		sp->leapcnt,
		sp->timecnt,
		sp->typecnt,
		sp->charcnt,
		sp->goback,
		sp->goahead);

	if (sp->timecnt <= 0)
		appendStringInfo(buf, "        0\n");
	else 
	{
		for (i=0; i < sp->timecnt; i++)
		{
			appendStringInfo(
				buf,
				"        %ld,\n",
				sp->ats[i]);
		}
	}

	appendStringInfo(
		buf,
		"    },\n"
		"    {    /* types[] */");

	if (sp->timecnt <= 0)
		appendStringInfo(buf, "\n        0");
	else 
	{
		for (i=0; i < sp->timecnt; i++)
		{
			if ((i % 12) == 0)
				appendStringInfo(buf, "\n       ");
			appendStringInfo(buf, " %d,", sp->types[i]);
		}
	}

	appendStringInfo(
		buf,
		"\n"
		"    },\n"
		"    {    /* ttis[] */\n");

	for (i=0; i < sp->typecnt; i++)
	{
		appendStringInfo(
			buf,
			"        { %ld, %d, %d, %d, %d },\n",
			sp->ttis[i].tt_gmtoff,
			sp->ttis[i].tt_isdst,
			sp->ttis[i].tt_abbrind,
			sp->ttis[i].tt_ttisstd,
			sp->ttis[i].tt_ttisgmt);
	}

	appendStringInfo(
		buf,
		"    },\n"
		"    {    /* lsis[] */\n");

	if (sp->leapcnt <= 0)
		appendStringInfo(buf, "        { 0, 0 }\n");
	else 
	{
		for (i=0; i < sp->leapcnt; i++)
		{
			appendStringInfo(
				buf,
				"        { %ld, %ld },\n",
				sp->lsis[i].ls_trans,
				sp->lsis[i].ls_corr);
		}
	}

	appendStringInfo(
		buf,
		"    },\n"
		"};\n");

	appendStringInfo(
		buf,
		"#endif /* __CUDACC__ */\n");
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_TIMELIB_H */
