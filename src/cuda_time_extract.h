/*
 * cuda_time_extract.h
 *
 * Routines to support EXTRACT() for CUDA GPU devices
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#ifndef CUDA_TIME_EXTRACT_H
#define CUDA_TIME_EXTRACT_H
#ifdef __CUDACC__

/*
 * extract() SQL functions
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
//	cl_char		token[11];	/* always NUL-terminated */
	cl_char		type;		/* see field type codes above */
	cl_int		value;		/* meaning depends on type */
} datetkn;

static const datetkn deltatktbl[] = {
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

static const datetkn datetktbl[] = {
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
			comp = strcmp(key, position->token);
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
extract_decode_unit(struct varlena *units, cl_int *p_type, cl_int *p_value)
{
	const char *s = VARDATA_ANY(units);
	cl_int		slen = VARSIZE_ANY_EXHDR(units);
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
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
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
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			break;
	}
	return result;
}

/*
 * date_part(text,timestamp) - timestamp_part
 */
STATIC_FUNCTION(pg_float8_t)
pgfn_extract_timestamp(kern_context *kcxt,
					   pg_text_t arg1, pg_timestamp_t arg2)
{
	pg_float8_t	result;
	cl_int		type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (extract_decode_unit(arg1.value, &type, &val))
	{
		if (TIMESTAMP_NOT_FINITE(arg2.value))
			return NonFiniteTimestampTzPart(kcxt, type, val,
											TIMESTAMP_IS_NOBEGIN(arg2.value));
		if (type == UNITS)
		{
			fsec_t		fsec;
			int			tz;
			struct pg_tm  tm;

			if (timestamp2tm(arg2.value, &tz, &tm, &fsec, NULL))
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
						break;
				}
			}
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
				return result;
			}
		}
	}
	STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	result.isnull = true;
	return result;
}

/*
 * date_part(text,timestamp with time zone) - timestamptz_part
 */
STATIC_FUNCTION(pg_float8_t)
pgfn_extract_timestamptz(kern_context *kcxt,
						 pg_text_t arg1, pg_timestamptz_t arg2)
{
	pg_float8_t	result;
	cl_int		type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (extract_decode_unit(arg1.value, &type, &val))
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

			if (timestamp2tm(arg2.value, &tz, &tm, &fsec, NULL))
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
						result.value = -tz;
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
						break;
				}
			}
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
				return result;
			}
		}
	}
	STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	result.isnull = true;
	return result;
}

/*
 * date_part(text,interval) - interval_part
 */
STATIC_FUNCTION(pg_float8_t)
pgfn_extract_interval(kern_context *kcxt, pg_text_t arg1, pg_interval_t arg2)
{
	pg_float8_t	result;
	int			type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (extract_decode_unit(arg1.value, &type, &val))
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
						break;
				}
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
	STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	result.isnull = true;
	return result;
}

/*
 * date_part(text,time with time zone) - timetz_part
 */
STATIC_FUNCTION(pg_float8_t)
pgfn_extract_timetz(kern_context *kcxt, pg_text_t arg1, pg_timetz_t arg2)
{
	pg_float8_t	result;
	int			type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (extract_decode_unit(arg1.value, &type, &val))
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
					break;
			}
		}
		else if (type == RESERV && val == DTK_EPOCH)
		{
			result.value = arg2.value.time / 1000000.0 + arg2.value.zone;
			return result;
		}
	}
	/* elsewhere, it is an error */
	STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	result.isnull = true;
	return result;
}

/*
 * date_part(text,time) - time_part
 */
STATIC_FUNCTION(pg_float8_t)
pgfn_extract_time(kern_context *kcxt, pg_text_t arg1, pg_time_t arg2)
{
	pg_float8_t	result;
	int			type, val;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;

	if (extract_decode_unit(arg1.value, &type, &val))
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
					break;
			}
		}
		else if (type == RESERV && val == DTK_EPOCH)
		{
			result.value = arg2.value / 1000000.0;
			return result;
		}
	}
	STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	result.isnull = true;
	return result;
}

#endif /* __CUDACC__ */
#endif /* CUDA_TIME_EXTRACT_H */
