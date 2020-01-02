/*
 * libgputime.cu
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
#include "cuda_common.h"

/* definition copied from datetime.c */
static __device__ const int day_tab[2][13] =
{
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0}
};

/* definition copied from pgtime.h */
static __device__ const int mon_lengths[2][MONSPERYEAR] = {
	{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
	{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

static __device__ const int year_lengths[2] = {
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

/*
 * Support routines
 */
STATIC_FUNCTION(cl_int)
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

STATIC_FUNCTION(void)
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

STATIC_FUNCTION(cl_int)
j2day(cl_int date)
{
	date += 1;
	date %= 7;
	/* Cope if division truncates towards zero, as it probably does */
	if (date < 0)
		date += 7;

	return date;
}	/* j2day() */

STATIC_FUNCTION(void)
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

STATIC_FUNCTION(int)
date2isoweek(int year, int mon, int mday)
{
	cl_double	result;
	cl_int		day0;
	cl_int		day4;
	cl_int		dayn;

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
	cl_double	result;
	cl_int		day0;
	cl_int		day4;
	cl_int		dayn;

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

STATIC_FUNCTION(Timestamp)
dt2local(Timestamp dt, int tz)
{
    return dt -= (tz * USECS_PER_SEC);
}

STATIC_FUNCTION(TimeOffset)
time2t(const int hour, const int min, const int sec, const fsec_t fsec)
{
    return (((((hour * MINS_PER_HOUR) + min) * SECS_PER_MINUTE) + sec) *
			USECS_PER_SEC) + fsec;
}

STATIC_FUNCTION(int)
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

STATIC_FUNCTION(int)
interval2tm(Interval span, struct pg_tm * tm, fsec_t *fsec)
{
    TimeOffset  time;
    TimeOffset  tfrac;

    tm->tm_year = span.month / MONTHS_PER_YEAR;
    tm->tm_mon = span.month % MONTHS_PER_YEAR;
    tm->tm_mday = span.day;
    time = span.time;

    tfrac = time / USECS_PER_HOUR;
    time -= tfrac * USECS_PER_HOUR;
    tm->tm_hour = tfrac;
    if (!SAMESIGN(tm->tm_hour, tfrac))
		return -1;
    tfrac = time / USECS_PER_MINUTE;
    time -= tfrac * USECS_PER_MINUTE;
    tm->tm_min = tfrac;
    tfrac = time / USECS_PER_SEC;
    *fsec = time - (tfrac * USECS_PER_SEC);
    tm->tm_sec = tfrac;

    return 0;
}

STATIC_FUNCTION(int)
tm2timetz(struct pg_tm * tm, fsec_t fsec, int tz, TimeTzADT *result)
{
    result->time = ((((tm->tm_hour * MINS_PER_HOUR + tm->tm_min) 
					  * SECS_PER_MINUTE) + tm->tm_sec) * USECS_PER_SEC) + fsec;
    result->zone = tz;

	return 0;
}

STATIC_FUNCTION(int)
increment_overflow(int *number, int delta)
{
    int         number0;

    number0 = *number;
    *number += delta;
    return (*number < number0) != (delta < 0);
}

STATIC_FUNCTION(int)
leaps_thru_end_of_no_recursive(const int y)
{
	assert(y >= 0);
	return y / 4 - y / 100 + y / 400;
}

STATIC_FUNCTION(int)
leaps_thru_end_of(const int y)
{
    return (y >= 0) ? (y / 4 - y / 100 + y / 400) :
        -(leaps_thru_end_of_no_recursive(-(y + 1)) + 1);
}

STATIC_FUNCTION(struct pg_tm *)
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

STATIC_FUNCTION(struct pg_tm *)
localsub(const tz_state *sp,
		 const cl_long *timep, 	/* pg_time_t in original */
		 struct pg_tm *tmp)
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
		cl_long	years;			/* pg_time_t in original */

		if (t < sp->ats[0])
			seconds = sp->ats[0] - t;
		else
			seconds = t - sp->ats[sp->timecnt - 1];
		--seconds;
		years = (seconds / SECSPERREPEAT + 1) * YEARSPERREPEAT;
		seconds = years * AVGSECSPERYEAR;
		if (t < sp->ats[0])
			newt += seconds;
		else
			newt -= seconds;
		if (newt < sp->ats[0] ||
			newt > sp->ats[sp->timecnt - 1])
			return NULL;		/* "cannot happen" */
		result = localsub(sp, &newt, tmp);
		if (result)
		{
			cl_long		newy;

			newy = result->tm_year;
			if (t < sp->ats[0])
				newy -= years;
			else
				newy += years;
			if (!(INT_MIN <= newy && newy <= INT_MAX))
				return NULL;
			result->tm_year = newy;
		}
		return result;
	}
	if (sp->timecnt == 0 || t < sp->ats[0])
	{
		i = sp->defaulttype;
	}
	else
	{
		int		lo = 1;
		int		hi = sp->timecnt;

		while (lo < hi)
		{
			int		mid = (lo + hi) >> 1;

			if (t < sp->ats[mid])
				hi = mid;
			else
				lo = mid + 1;
		}
		i = (int) sp->types[lo - 1];
	}
	ttisp = &sp->ttis[i];

	/*
	 * To get (wrong) behavior that's compatible with System V Release 2.0
	 * you'd replace the statement below with t += ttisp->tt_gmtoff;
	 * timesub(&t, 0L, sp, tmp);
	 */
	result = timesub(&t, ttisp->tt_gmtoff, sp, tmp);
	if (result)
	{
		result->tm_isdst = ttisp->tt_isdst;
//		result->tm_zone = (char *) &sp->chars[ttisp->tt_abbrind];
	}
	return result;
}

STATIC_FUNCTION(struct pg_tm *)
pg_localtime(const cl_long *timep,	/* pg_time_t in original */
			 struct pg_tm *tm,
			 const tz_state *sp)	/* const pg_tz *tz in original */
{
	/*
	 * pg_localtime() returns tm if success. NULL, elsewhere.
	 */
	return localsub(sp, timep, tm);
}

STATIC_FUNCTION(int)
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

		result = pg_next_dst_boundary(&newt,
									  before_gmtoff,
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

STATIC_FUNCTION(cl_int)
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
STATIC_FUNCTION(cl_bool)
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

STATIC_FUNCTION(cl_bool)
tm2timestamp(struct pg_tm * tm, fsec_t fsec, int *tzp, Timestamp *result)
{
    TimeOffset  date;
    TimeOffset  time;

    /* Julian day routines are not correct for negative Julian days */
    if (!IS_VALID_JULIAN(tm->tm_year, tm->tm_mon, tm->tm_mday))
    {
        *result = 0;		/* keep compiler quiet */
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
        *result = 0;		/* keep compiler quiet */
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
	else if (arg.value >= (TIMESTAMP_END_JULIAN - POSTGRES_EPOCH_JDATE))
	{
		result.isnull = false;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "date out of range for timestamp");
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

		if (!IS_VALID_TIMESTAMP(result.value))
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
						  "date out of range for timestamp");
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
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp out of range");
	}
	else
	{
        tz = DetermineTimeZoneOffset(&tm, &session_timezone_state);
		if (!tm2timestamp(&tm, fsec, &tz, &result.value))
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
						  "timestamp out of range");
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
STATIC_FUNCTION(void)
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

STATIC_FUNCTION(void)
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

STATIC_FUNCTION(Interval)
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
DEVICE_FUNCTION(pg_date_t)
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
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp out of range");
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
DEVICE_FUNCTION(pg_date_t)
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
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp with timezone out of range");
	}
	else
	{
		result.isnull = false;
		result.value  = (date2j(tm.tm_year, tm.tm_mon, tm.tm_mday)
						 - POSTGRES_EPOCH_JDATE);
	}
	return result;
}

DEVICE_FUNCTION(pg_time_t)
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

DEVICE_FUNCTION(pg_time_t)
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
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp out of range");
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

DEVICE_FUNCTION(pg_time_t)
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
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp with timezone out of range");
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

DEVICE_FUNCTION(pg_timetz_t)
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

DEVICE_FUNCTION(pg_timetz_t)
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
	else if (!timestamp2tm(arg1.value, &tz, &tm, &fsec, NULL) != 0)
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp with timezone out of range");
	}
	else
	{
		tm2timetz(&tm, fsec, tz, &(result.value));
		result.isnull = false;
	}

	return result;
}

#ifdef NOT_USED
DEVICE_FUNCTION(pg_timetz_t)
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

DEVICE_FUNCTION(pg_timestamp_t)
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
	else if (arg1.value >= (TIMESTAMP_END_JULIAN - POSTGRES_EPOCH_JDATE))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "date out of range for timestamp");
	}
	else
	{
		/* date is days since 2000, timestamp is microseconds since same... */
		result.isnull = false;
		result.value = arg1.value * USECS_PER_DAY;
	}
	return result;
}

DEVICE_FUNCTION(pg_timestamp_t)
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
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp out of range");
	}
	else if (!tm2timestamp(&tm, fsec, NULL, &result.value))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "timestamp out of range");
	}
	else
	{
		result.isnull = false;
	}

	return result;
}

DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_date_timestamptz(kern_context *kcxt, pg_date_t arg1)
{
	return date2timestamptz(kcxt, arg1);
}

DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_timestamp_timestamptz(kern_context *kcxt, pg_timestamp_t arg1)
{
	return timestamp2timestamptz(kcxt, arg1);
}

/*
 * Simple comparison
 */
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_cmp(kern_context *kcxt, pg_date_t arg1, pg_date_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value < arg2.value)
			result.value = 1;
		else if (arg1.value > arg2.value)
			result.value = -1;
		else
			result.value = 0;
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_time_cmp(kern_context *kcxt, pg_time_t arg1, pg_time_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value < arg2.value)
			result.value = 1;
		else if (arg1.value > arg2.value)
			result.value = -1;
		else
			result.value = 0;
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt,
				  pg_timestamp_t arg1, pg_timestamp_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value < arg2.value)
			result.value = 1;
		else if (arg1.value > arg2.value)
			result.value = -1;
		else
			result.value = 0;
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt,
				  pg_timestamptz_t arg1, pg_timestamptz_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value < arg2.value)
			result.value = 1;
		else if (arg1.value > arg2.value)
			result.value = -1;
		else
			result.value = 0;
	}
	return result;
}

/*
 * Time/Date operators
 */
DEVICE_FUNCTION(pg_date_t)
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

DEVICE_FUNCTION(pg_date_t)
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

DEVICE_FUNCTION(pg_int4_t)
pgfn_date_mi(kern_context *kcxt, pg_date_t arg1, pg_date_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else if (DATE_NOT_FINITE(arg1.value) || DATE_NOT_FINITE(arg2.value))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "cannot subtract infinite dates");
	}
	else
	{
		result.isnull = false;
		result.value = (cl_int)(arg1.value - arg2.value);
	}
	return result;
}

DEVICE_FUNCTION(pg_timestamp_t)
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

DEVICE_FUNCTION(pg_date_t)
pgfn_integer_pl_date(kern_context *kcxt, pg_int4_t arg1, pg_date_t arg2)
{
	return pgfn_date_pli(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_timestamp_t)
pgfn_timedate_pl(kern_context *kcxt, pg_time_t arg1, pg_date_t arg2)
{
	return pgfn_datetime_pl(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_interval_t)
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

DEVICE_FUNCTION(pg_interval_t)
pgfn_timestamp_mi(kern_context *kcxt, pg_timestamp_t arg1, pg_timestamp_t arg2)
{
	pg_interval_t result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else if (TIMESTAMP_NOT_FINITE(arg1.value) || 
			 TIMESTAMP_NOT_FINITE(arg2.value))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "cannot subtract infinite timestamps");
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

DEVICE_FUNCTION(pg_timetz_t)
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

DEVICE_FUNCTION(pg_timetz_t)
pgfn_timetz_mi_interval(kern_context *kcxt, pg_timetz_t arg1, pg_interval_t arg2)
{
	arg2.value.time = - arg2.value.time;

	return pgfn_timetz_pl_interval(kcxt, arg1, arg2);
}

DEVICE_FUNCTION(pg_timestamptz_t)
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
			fsec_t	fsec;
			int		tz;

			if (!timestamp2tm(arg1.value, &tz, &tm, &fsec, NULL))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
							  "timestamp with timezone out of range");
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

			tz = DetermineTimeZoneOffset(&tm, &session_timezone_state);
			if (tm2timestamp(&tm, fsec, &tz, &arg1.value) != 0)
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
							  "timestamp with timezone out of range");
				return result;
			}
		}

		if (arg2.value.day != 0)
		{
			struct pg_tm tm;
			fsec_t	fsec;
			int		julian;
			int		tz;

			if (!timestamp2tm(arg1.value, &tz, &tm, &fsec, NULL))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
							  "timestamp with timezone out of range");
				return result;
			}

			/* Add days by converting to and from julian */
			julian = (date2j(tm.tm_year, tm.tm_mon, tm.tm_mday) +
					  arg2.value.day);
			j2date(julian, &tm.tm_year, &tm.tm_mon, &tm.tm_mday);

			tz = DetermineTimeZoneOffset(&tm, &session_timezone_state);
			if (!tm2timestamp(&tm, fsec, &tz, &arg1.value))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
							  "timestamp with timezone out of range");
				return result;
			}
		}
		
		result.isnull = false;
		result.value  = arg1.value + arg2.value.time;
	}

	return result;
}

DEVICE_FUNCTION(pg_timestamptz_t)
pgfn_timestamptz_mi_interval(kern_context *kcxt,
							 pg_timestamptz_t arg1,
							 pg_interval_t arg2)
{
	arg2.value.month = - arg2.value.month;
	arg2.value.day   = - arg2.value.day;
	arg2.value.time  = - arg2.value.time;

	return pgfn_timestamptz_pl_interval(kcxt, arg1, arg2);
}

DEVICE_FUNCTION(pg_interval_t)
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
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "interval out of range");
		return result;
	}
	result.value.day = - arg1.value.day;
	if (arg1.value.day != 0 && SAMESIGN(result.value.day, arg1.value.day))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "interval out of range");
		return result;
	}
	result.value.month = - arg1.value.month;
	if (arg1.value.month != 0 &&
		SAMESIGN(result.value.month, arg1.value.month))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "interval out of range");
		return result;
	}
	result.isnull = false;

	return result;
}

DEVICE_FUNCTION(pg_interval_t)
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
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "interval out of range");
		return result;
	}
	result.value.day = arg1.value.day + arg2.value.day;
	if (SAMESIGN(arg1.value.day, arg2.value.day) &&
		!SAMESIGN(result.value.day, arg1.value.day))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "interval out of range");
		return result;
	}
	result.value.time = arg1.value.time + arg2.value.time;
	if (SAMESIGN(arg1.value.time, arg2.value.time) &&
		!SAMESIGN(result.value.time, arg1.value.time))
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
					  "interval out of range");
		return result;
	}
	result.isnull = false;

	return result;
}

DEVICE_FUNCTION(pg_interval_t)
pgfn_interval_mi(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
{
	arg2.value.time  = - arg2.value.time;
	arg2.value.day   = - arg2.value.day;
	arg2.value.month = - arg2.value.month;

	return pgfn_interval_pl(kcxt, arg1, arg2);
}

DEVICE_FUNCTION(pg_timestamptz_t)
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
DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_int4_t)
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
STATIC_FUNCTION(cl_int)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_timetz_t arg1, pg_timetz_t arg2)
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
DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_int4_t)
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
DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_lt_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_gt_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_le_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_ge_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_eq_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_eq_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ge_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_le_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_gt_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_lt_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ne_date(kern_context *kcxt,
						 pg_timestamptz_t arg1, pg_date_t arg2)
{
	return pgfn_date_ne_timestamptz(kcxt, arg2, arg1);
}

/*
 * Comparison between timestamp and timestamptz
 */
DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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
DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_lt_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_gt_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_le_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_ge_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_eq_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_eq_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ge_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_le_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_gt_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_lt_timestamptz(kcxt, arg2, arg1);
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_timestamptz_ne_timestamp(kern_context *kcxt,
							  pg_timestamptz_t arg1, pg_timestamp_t arg2)
{
	return pgfn_timestamp_ne_timestamptz(kcxt, arg2, arg1);
}

/*
 * Comparison between pg_interval_t
 */
STATIC_FUNCTION(cl_int)
interval_cmp_internal(Interval arg1, Interval arg2)
{
	cl_long		days1, days2;
	cl_long		frac1, frac2;

	interval_cmp_value(arg1, &days1, &frac1);
	interval_cmp_value(arg2, &days2, &frac2);

	if (days1 < days2)
		return -1;
	else if (days1 > days2)
		return  1;
	else if (frac1 < frac2)
		return -1;
	else if (frac1 > frac2)
		return  1;
	return 0;
}

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_interval_t arg1, pg_interval_t arg2)
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
DEVICE_FUNCTION(pg_timestamptz_t)
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


DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

DEVICE_FUNCTION(pg_bool_t)
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

/*
 * extract() SQL functions support
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
	cl_char		type;		/* see field type codes above */
	cl_int		value;		/* meaning depends on type */
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
 *
 */
STATIC_INLINE(const datetkn *)
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
			comp = __strncmp(key, position->token, TOKMAXLEN);
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

STATIC_FUNCTION(cl_bool)
extract_decode_unit(const char *s, cl_int slen,
					cl_int *p_type, cl_int *p_value)
{
	char		key[20];
	const datetkn *dtoken;
	int			i;

	if (slen >= 20)
		return false;
	/* convert to the lower case string */
	for (i=0; i < slen; i++)
	{
		cl_uchar	ch = (cl_uchar) s[i];

		if (ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
#if 0
		//TODO: ctype.h functions support on the device code
		else if (enc_is_single_byte && IS_HIGHBIT_SET(ch) && isupper(ch))
			ch = tolower(ch);
#endif
		key[i] = ch;
	}
	key[i] = '\0';

	/* DecodeUnits() / DecodeSpecial() */
	dtoken = datebsearch(key, deltatktbl, lengthof(deltatktbl));
	if (!dtoken)
		dtoken = datebsearch(key, datetktbl, lengthof(datetktbl));
	if (dtoken)
	{
		*p_type  = dtoken->type;
		*p_value = dtoken->value;
		return true;
	}
	return false;
}


STATIC_FUNCTION(pg_float8_t)
NonFiniteTimestampTzPart(kern_context *kcxt,
						 int type, int unit, bool is_negative)
{
	pg_float8_t	result;

	if (type != UNITS && type != RESERV)
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
					  "not recognized timestamp units");
		return result;
	}

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
			result.isnull = false;
			result.value = 0.0;
			break;
			/* Monotonically-increasing units */
		case DTK_YEAR:
		case DTK_DECADE:
		case DTK_CENTURY:
		case DTK_MILLENNIUM:
		case DTK_JULIAN:
		case DTK_ISOYEAR:
		case DTK_EPOCH:
			result.isnull = false;
			result.value = (is_negative ? -DBL_INFINITY : DBL_INFINITY);
			break;

		default:
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_FEATURE_NOT_SUPPORTED,
						  "unsupported timestamp unit");
			break;
	}
	return result;
}

/*
 * date_part(text,timestamp) - timestamp_part
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_timestamp(kern_context *kcxt,
					   pg_text_t arg1, pg_timestamp_t arg2)
{
	pg_float8_t	result;
	char	   *s;
	cl_int		slen;
	cl_int		type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (pg_varlena_datum_extract(kcxt, arg1, &s, &slen) &&
		extract_decode_unit(s, slen, &type, &val))
	{
		if (TIMESTAMP_NOT_FINITE(arg2.value))
			return NonFiniteTimestampTzPart(kcxt, type, val,
											TIMESTAMP_IS_NOBEGIN(arg2.value));
		if (type == UNITS)
		{
			fsec_t		fsec;
			struct pg_tm  tm;

			if (!timestamp2tm(arg2.value, NULL, &tm, &fsec, NULL))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
							  "timestamp out of range");
			}
			else
			{
				switch (val)
				{
					case DTK_MICROSEC:
						result.value = tm.tm_sec * 1000000.0 + fsec;
						return result;

					case DTK_MILLISEC:
						result.value = tm.tm_sec * 1000.0 + fsec / 1000.0;
						return result;

					case DTK_SECOND:
						result.value = tm.tm_sec + fsec / 1000000.0;
						return result;

					case DTK_MINUTE:
						result.value = tm.tm_min;
						return result;

					case DTK_HOUR:
						result.value = tm.tm_hour;
						return result;

					case DTK_DAY:
						result.value = tm.tm_mday;
						return result;

					case DTK_MONTH:
						result.value = tm.tm_mon;
						return result;

					case DTK_QUARTER:
						result.value = (tm.tm_mon - 1) / 3 + 1;
						return result;

					case DTK_WEEK:
						result.value = (double) date2isoweek(tm.tm_year,
															 tm.tm_mon,
															 tm.tm_mday);
						return result;

					case DTK_YEAR:
						/* there is no year 0, just 1 BC and 1 AD */
						if (tm.tm_year > 0)
							result.value = tm.tm_year;
						else
							result.value = tm.tm_year - 1;
						return result;

					case DTK_DECADE:
						if (tm.tm_year >= 0)
							result.value = tm.tm_year / 10;
						else
							result.value = -((8 - (tm.tm_year - 1)) / 10);
						return result;

					case DTK_CENTURY:
						if (tm.tm_year > 0)
							result.value = (tm.tm_year + 99) / 100;
						else
							result.value = -((99 - (tm.tm_year - 1)) / 100);
						return result;

					case DTK_MILLENNIUM:
						if (tm.tm_year > 0)
							result.value = (tm.tm_year + 999) / 1000;
						else
							result.value = -((999 - (tm.tm_year - 1)) / 1000);
						return result;

					case DTK_JULIAN:
						result.value = date2j(tm.tm_year,
											  tm.tm_mon,
											  tm.tm_mday) +
							(tm.tm_hour * MINS_PER_HOUR * SECS_PER_MINUTE +
							 tm.tm_min * SECS_PER_MINUTE +
							 tm.tm_sec +
							 fsec / 1000000.0) / (double)SECS_PER_DAY;
						return result;

					case DTK_ISOYEAR:
						result.value = date2isoyear(tm.tm_year,
													tm.tm_mon,
													tm.tm_mday);
						return result;

					case DTK_DOW:
					case DTK_ISODOW:
						result.value = j2day(date2j(tm.tm_year,
													tm.tm_mon,
													tm.tm_mday));
						if (val == DTK_ISODOW && result.value == 0)
							result.value = 7;
						return result;

					case DTK_DOY:
						result.value = (date2j(tm.tm_year,
											   tm.tm_mon,
											   tm.tm_mday) -
										date2j(tm.tm_year, 1, 1) + 1);
						return result;

					case DTK_TZ:
					case DTK_TZ_MINUTE:
					case DTK_TZ_HOUR:
					default:
						result.isnull = true;
						STROM_EREPORT(kcxt, ERRCODE_FEATURE_NOT_SUPPORTED,
									  "unsupported unit of timestamp");
						break;
				}
			}
			return result;
		}
		else if (type == RESERV)
		{
			if (val == DTK_EPOCH)
			{
				Timestamp	epoch = SetEpochTimestamp();
				/* try to avoid precision loss in subtraction */
				if (arg2.value < (LONG_MAX + epoch))
					result.value = (arg2.value - epoch) / 1000000.0;
				else
					result.value = ((double)arg2.value - epoch) / 1000000.0;
			}
			else
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_FEATURE_NOT_SUPPORTED,
							  "unsupported unit of timestamp");
			}
			return result;
		}
	}
	result.isnull = true;
	STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
				  "not recognized unit of timestamp");
	return result;
}

/*
 * date_part(text,timestamp with time zone) - timestamptz_part
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_timestamptz(kern_context *kcxt,
						 pg_text_t arg1, pg_timestamptz_t arg2)
{
	pg_float8_t	result;
	char	   *s;
	cl_int		slen;
	cl_int		type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (pg_varlena_datum_extract(kcxt, arg1, &s, &slen) &&
		extract_decode_unit(s, slen, &type, &val))
	{
		if (TIMESTAMP_NOT_FINITE(arg2.value))
			return NonFiniteTimestampTzPart(kcxt, type, val,
											TIMESTAMP_IS_NOBEGIN(arg2.value));
		if (type == UNITS)
		{
			fsec_t		fsec;
			int			tz;
			struct pg_tm  tm;
			double		dummy;

			if (!timestamp2tm(arg2.value, &tz, &tm, &fsec, NULL))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_DATETIME_VALUE_OUT_OF_RANGE,
							  "timestamp with timezone out of range");
			}
			else
			{
				switch (val)
				{
					case DTK_TZ:
						result.value = -tz;
						return result;

					case DTK_TZ_MINUTE:
						result.value = (cl_double)(-tz) / MINS_PER_HOUR;
						FMODULO(result.value, dummy, (double) MINS_PER_HOUR);
						return result;

					case DTK_TZ_HOUR:
						dummy = -tz;
						FMODULO(dummy, result.value, (double) SECS_PER_HOUR);
						return result;

					case DTK_MICROSEC:
						result.value = tm.tm_sec * 1000000.0 + fsec;
						return result;

					case DTK_MILLISEC:
						result.value = tm.tm_sec * 1000.0 + fsec / 1000.0;
						return result;

					case DTK_SECOND:
						result.value = tm.tm_sec + fsec / 1000000.0;
						return result;

					case DTK_MINUTE:
						result.value = tm.tm_min;
						return result;

					case DTK_HOUR:
						result.value = tm.tm_hour;
						return result;

					case DTK_DAY:
						result.value = tm.tm_mday;
						return result;

					case DTK_MONTH:
						result.value = tm.tm_mon;
						return result;

					case DTK_QUARTER:
						result.value = (tm.tm_mon - 1) / 3 + 1;
						return result;

					case DTK_WEEK:
						result.value = (cl_double) date2isoweek(tm.tm_year,
																tm.tm_mon,
																tm.tm_mday);
						return result;

					case DTK_YEAR:
						if (tm.tm_year > 0)
							result.value = tm.tm_year;
						else
							/* there is no year 0, just 1 BC and 1 AD */
							result.value = tm.tm_year - 1;
						return result;

					case DTK_DECADE:
						/* see comments in timestamp_part */
						if (tm.tm_year > 0)
							result.value = tm.tm_year / 10;
						else
							result.value = -((8 - (tm.tm_year - 1)) / 10);
						return result;

					case DTK_CENTURY:
						/* see comments in timestamp_part */
						if (tm.tm_year > 0)
							result.value = (tm.tm_year + 99) / 100;
						else
							result.value = -((99 - (tm.tm_year - 1)) / 100);
						return result;

					case DTK_MILLENNIUM:
						/* see comments in timestamp_part */
						if (tm.tm_year > 0)
							result.value = (tm.tm_year + 999) / 1000;
						else
							result.value = -((999 - (tm.tm_year - 1)) / 1000);
						return result;

					case DTK_JULIAN:
						result.value = (cl_double)date2j(tm.tm_year,
														 tm.tm_mon,
														 tm.tm_mday) +
							(tm.tm_hour * MINS_PER_HOUR * SECS_PER_MINUTE +
							 tm.tm_min * SECS_PER_MINUTE +
							 tm.tm_sec +
							 fsec / 1000000.0) / (double) SECS_PER_MINUTE;
						return result;

					case DTK_ISOYEAR:
						result.value = date2isoyear(tm.tm_year,
													tm.tm_mon,
													tm.tm_mday);
						return result;

					case DTK_DOW:
					case DTK_ISODOW:
						result.value = j2day(date2j(tm.tm_year,
													tm.tm_mon,
													tm.tm_mday));
						if (val == DTK_ISODOW && result.value == 0)
							result.value = 7;
						return result;

					case DTK_DOY:
						result.value = (date2j(tm.tm_year,
											   tm.tm_mon,
											   tm.tm_mday) -
										date2j(tm.tm_year, 1, 1) + 1);
						return result;

					default:
						result.isnull = true;
						STROM_EREPORT(kcxt, ERRCODE_FEATURE_NOT_SUPPORTED,
									  "unsupported unit of timestamp with timezone");
						break;
				}
			}
			return result;
		}
		else if (type == RESERV)
		{
			if (val == DTK_EPOCH)
			{
				TimestampTz	epoch = SetEpochTimestamp();
				/* try to avoid precision loss in subtraction */
				if (arg2.value < (LONG_MAX + epoch))
					result.value = (arg2.value - epoch) / 1000000.0;
				else
					result.value = ((double)arg2.value - epoch) / 1000000.0;
			}
			else
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_FEATURE_NOT_SUPPORTED,
							  "unsupported unit of timestamp with timezone");
			}
			return result;
		}
	}
	result.isnull = true;
	STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
				  "not recognized unit of timestamp");
	return result;
}

/*
 * date_part(text,interval) - interval_part
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_interval(kern_context *kcxt, pg_text_t arg1, pg_interval_t arg2)
{
	pg_float8_t	result;
	char	   *s;
	cl_int		slen;
	cl_int		type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (pg_varlena_datum_extract(kcxt, arg1, &s, &slen) &&
		extract_decode_unit(s, slen, &type, &val))
	{
		if (type == UNITS)
		{
			fsec_t		fsec;
			struct pg_tm tm;

			if (interval2tm(arg2.value, &tm, &fsec) == 0)
			{
				switch (val)
				{
					case DTK_MICROSEC:
						result.value = tm.tm_sec * 1000000.0 + fsec;
						return result;
					case DTK_MILLISEC:
						result.value = tm.tm_sec * 1000.0 + fsec / 1000.0;
						return result;
					case DTK_SECOND:
						result.value = tm.tm_sec + fsec / 1000000.0;
						return result;
					case DTK_MINUTE:
						result.value = tm.tm_min;
						return result;
					case DTK_HOUR:
						result.value = tm.tm_hour;
						return result;
					case DTK_DAY:
						result.value = tm.tm_mday;
						return result;
					case DTK_MONTH:
						result.value = tm.tm_mon;
						return result;
					case DTK_QUARTER:
						result.value = (tm.tm_mon / 3) + 1;
						return result;
					case DTK_YEAR:
						result.value = tm.tm_year;
						return result;
					case DTK_DECADE:
						/* caution: C division may have negative remainder */
						result.value = tm.tm_year / 10;
						return result;
					case DTK_CENTURY:
						/* caution: C division may have negative remainder */
						result.value = tm.tm_year / 100;
						return result;
					case DTK_MILLENNIUM:
						/* caution: C division may have negative remainder */
						result.value = tm.tm_year / 1000;
						return result;
					default:
						result.isnull = true;
						STROM_EREPORT(kcxt, ERRCODE_FEATURE_NOT_SUPPORTED,
									  "unsupported unit of interval");
						break;
				}
			}
			else
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_INTERNAL_ERROR,
							  "could not convert interval to tm");
			}
		}
		else if (type == RESERV && val == DTK_EPOCH)
		{
			result.value = arg2.value.time / 1000000.0
				+ ((double)(DAYS_PER_YEAR * SECS_PER_DAY) *
				   (arg2.value.month / MONTHS_PER_YEAR))
				+ ((double)(DAYS_PER_MONTH * SECS_PER_DAY) *
				   (arg2.value.month % MONTHS_PER_YEAR))
				+ ((double) SECS_PER_DAY) * arg2.value.day;
			return result;
		}
	}
	result.isnull = true;
	STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
				  "unrecognized parameter of interval");
	return result;
}

/*
 * date_part(text,time with time zone) - timetz_part
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_timetz(kern_context *kcxt, pg_text_t arg1, pg_timetz_t arg2)
{
	pg_float8_t	result;
	char	   *s;
	cl_int		slen;
	cl_int		type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (pg_varlena_datum_extract(kcxt, arg1, &s, &slen) &&
		extract_decode_unit(s, slen, &type, &val))
	{
		if (type == UNITS)
		{
			fsec_t		fsec;
			struct pg_tm tm;

			time2tm(arg2.value.time, &tm, &fsec);
			switch (val)
			{
				case DTK_MICROSEC:
					result.value = tm.tm_sec * 1000000.0 + fsec;
					return result;

				case DTK_MILLISEC:
					result.value = tm.tm_sec * 1000.0 + fsec / 1000.0;
					return result;

				case DTK_SECOND:
					result.value = tm.tm_sec + fsec / 1000000.0;
					return result;

				case DTK_MINUTE:
					result.value = tm.tm_min;
					return result;

				case DTK_HOUR:
					result.value = tm.tm_hour;
					return result;

				case DTK_TZ:
				case DTK_TZ_MINUTE:
				case DTK_TZ_HOUR:
				case DTK_DAY:
				case DTK_MONTH:
				case DTK_QUARTER:
				case DTK_YEAR:
				case DTK_DECADE:
				case DTK_CENTURY:
				case DTK_MILLENNIUM:
				case DTK_ISOYEAR:
				default:
					result.isnull = true;
					STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
								  "unsupported unit of time");
					break;
			}
		}
		else if (type == RESERV && val == DTK_EPOCH)
		{
			result.value = arg2.value.time / 1000000.0 + arg2.value.zone;
			return result;
		}
	}
	result.isnull = true;
	STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
				  "not a recognized unit of time");
	return result;
}

/*
 * date_part(text,time) - time_part
 */
DEVICE_FUNCTION(pg_float8_t)
pgfn_extract_time(kern_context *kcxt, pg_text_t arg1, pg_time_t arg2)
{
	pg_float8_t	result;
	char	   *s;
	cl_int		slen;
	cl_int		type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (pg_varlena_datum_extract(kcxt, arg1, &s, &slen) &&
		extract_decode_unit(s, slen, &type, &val))
	{
		if (type == UNITS)
		{
			fsec_t		fsec;
			struct pg_tm tm;

			time2tm(arg2.value, &tm, &fsec);
			switch (val)
			{
				case DTK_MICROSEC:
					result.value = tm.tm_sec * 1000000.0 + fsec;
					return result;

				case DTK_MILLISEC:
					result.value = tm.tm_sec * 1000.0 + fsec / 1000.0;
					return result;

				case DTK_SECOND:
					result.value = tm.tm_sec + fsec / 1000000.0;
					return result;

				case DTK_MINUTE:
					result.value = tm.tm_min;
					return result;

				case DTK_HOUR:
					result.value = tm.tm_hour;
					return result;

				case DTK_TZ:
				case DTK_TZ_MINUTE:
				case DTK_TZ_HOUR:
				case DTK_DAY:
				case DTK_MONTH:
				case DTK_QUARTER:
				case DTK_YEAR:
				case DTK_DECADE:
				case DTK_CENTURY:
				case DTK_MILLENNIUM:
				case DTK_ISOYEAR:
				default:
					result.isnull = true;
					STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
								  "unsupported unit of time");
					break;
			}
		}
		else if (type == RESERV && val == DTK_EPOCH)
		{
			result.value = arg2.value / 1000000.0;
			return result;
		}
	}
	result.isnull = true;
    STROM_EREPORT(kcxt, ERRCODE_INVALID_PARAMETER_VALUE,
                  "not a recognized unit of time");
	return result;
}
