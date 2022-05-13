/*
 * pg_utils.h
 *
 * Inline routines of misc utility purposes
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef PG_UTILS_H
#define PG_UTILS_H


/*
 * transformation from align character into width
 */
static inline int
typealign_get_width(char type_align)
{
	switch (type_align)
	{
		case 'c':
			return 1;
		case 's':
			return ALIGNOF_SHORT;
		case 'i':
			return ALIGNOF_INT;
		case 'd':
			return ALIGNOF_DOUBLE;
		default:
			elog(ERROR, "unexpected type alignment: %c", type_align);
	}
	return -1;  /* be compiler quiet */
}

/*
 * get_next_log2
 *
 * It returns N of the least 2^N value that is larger than or equal to
 * the supplied value.
 */
static inline int
get_next_log2(Size size)
{
	int		shift = 0;

	if (size == 0 || size == 1)
		return 0;
	size--;
#ifdef __GNUC__
	shift = sizeof(Size) * BITS_PER_BYTE - __builtin_clzl(size);
#else
#if SIZEOF_VOID_P == 8
	if ((size & 0xffffffff00000000UL) != 0)
	{
		size >>= 32;
		shift += 32;
	}
#endif
	if ((size & 0xffff0000UL) != 0)
	{
		size >>= 16;
		shift += 16;
	}
	if ((size & 0x0000ff00UL) != 0)
	{
		size >>= 8;
		shift += 8;
	}
	if ((size & 0x000000f0UL) != 0)
	{
		size >>= 4;
		shift += 4;
	}
	if ((size & 0x0000000cUL) != 0)
	{
		size >>= 2;
		shift += 2;
	}
	if ((size & 0x00000002UL) != 0)
	{
		size >>= 1;
		shift += 1;
	}
	if ((size & 0x00000001UL) != 0)
		shift += 1;
#endif  /* !__GNUC__ */
	return shift;
}

/*
 * __trim - remove whitespace at the head/tail of cstring
 */
static inline char *
__trim(char *token)
{
	char   *tail = token + strlen(token) - 1;

	while (*token == ' ' || *token == '\t')
		token++;
	while (tail >= token && (*tail == ' ' || *tail == '\t'))
		*tail-- = '\0';
	return token;
}

/* lappend on the specified memory-context */
static inline List *
lappend_cxt(MemoryContext memcxt, List *list, void *datum)
{
	MemoryContext oldcxt = MemoryContextSwitchTo(memcxt);
	List   *r;

	r = lappend(list, datum);
	MemoryContextSwitchTo(oldcxt);

	return r;
}

/*
 * formater of numeric/bytesz/millisec
 */
static inline char *
format_numeric(int64 value)
{
	if (value > 8000000000000L   || value < -8000000000000L)
		return psprintf("%.2fT", (double)value / 1000000000000.0);
	else if (value > 8000000000L || value < -8000000000L)
		return psprintf("%.2fG", (double)value / 1000000000.0);
	else if (value > 8000000L    || value < -8000000L)
		return psprintf("%.2fM", (double)value / 1000000.0);
	else if (value > 8000L       || value < -8000L)
		return psprintf("%.2fK", (double)value / 1000.0);
	else
		return psprintf("%ld", value);
}

static inline char *
format_bytesz(size_t nbytes)
{
	if (nbytes > (1UL<<43))
		return psprintf("%.2fTB", (double)nbytes / (double)(1UL<<40));
	else if (nbytes > (1UL<<33))
		return psprintf("%.2fGB", (double)nbytes / (double)(1UL<<30));
	else if (nbytes > (1UL<<23))
		return psprintf("%.2fMB", (double)nbytes / (double)(1UL<<20));
	else if (nbytes > (1UL<<13))
		return psprintf("%.2fKB", (double)nbytes / (double)(1UL<<10));
	return psprintf("%uB", (unsigned int)nbytes);
}

static inline char *
format_millisec(double milliseconds)
{
	if (milliseconds > 300000.0)    /* more then 5min */
		return psprintf("%.2fmin", milliseconds / 60000.0);
	else if (milliseconds > 8000.0) /* more than 8sec */
		return psprintf("%.2fsec", milliseconds / 1000.0);
	return psprintf("%.2fms", milliseconds);
}
#endif	/* PG_UTILS_H */

