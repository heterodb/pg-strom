/*
 * xpu_textlib.h
 *
 * Misc definitions for text routines for both of GPU and DPU
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_TEXTLIB_H
#define XPU_TEXTLIB_H

#ifndef PG_BYTEAOID
#define PG_BYTEAOID		17
#endif
#ifndef PG_TEXTOID
#define PG_TEXTOID		25
#endif
PGSTROM_SQLTYPE_VARLENA_DECLARATION(bytea);
PGSTROM_SQLTYPE_VARLENA_DECLARATION(text);
PGSTROM_SQLTYPE_VARLENA_DECLARATION(bpchar);

/*
 * Database Encoding Info
 */
struct xpu_encode_info {
	char	encname[16];
	int		enc_maxlen;
	int	  (*enc_mblen)(const char *s);
};
typedef struct xpu_encode_info	xpu_encode_info;

EXTERN_DATA(xpu_encode_info, xpu_encode_catalog[]);

/*
 * validation checkers
 */
INLINE_FUNCTION(bool)
xpu_bpchar_is_valid(kern_context *kcxt, const xpu_bpchar_t *arg)
{
	if (arg->length < 0)
	{
		SUSPEND_FALLBACK(kcxt, "bpchar datum is compressed or external");
		return false;
	}
	return true;
}

INLINE_FUNCTION(bool)
xpu_text_is_valid(kern_context *kcxt, const xpu_text_t *arg)
{
	if (arg->length < 0)
	{
		SUSPEND_FALLBACK(kcxt, "text datum is compressed or external");
		return false;
	}
	return true;
}

INLINE_FUNCTION(bool)
xpu_bytea_is_valid(kern_context *kcxt, const xpu_bytea_t *arg)
{
	if (arg->length < 0)
	{
		SUSPEND_FALLBACK(kcxt, "bytea datum is compressed or external");
		return false;
	}
	return true;
}

/*
 * Functions in string.h
 */
#define __UPPER(x)		(((x) >= 'a' && (x) <= 'z') ? 'A' + ((x) - 'a') : (x))
#define __LOWER(x)		(((x) >= 'A' && (x) <= 'Z') ? 'a' + ((X) - 'A') : (x))

INLINE_FUNCTION(int)
__strcmp(const char *s1, const char *s2)
{
	for (;;)
	{
		char	c1 = *s1++;
		char	c2 = *s2++;
		int		diff = c1 - c2;

		if (c1 == '\0' || diff != 0)
			return diff;
	}
}

INLINE_FUNCTION(int)
__strncmp(const char *s1, const char *s2, int n)
{
	while (n > 0)
	{
		char	c1 = (unsigned char) *s1++;
		char	c2 = (unsigned char) *s2++;

		if (c1 == '\0' || c1 != c2)
			return c1 - c2;
		n--;
	}
	return 0;
}

INLINE_FUNCTION(int)
__strcasecmp(const char *s1, const char *s2)
{
	for (;;)
	{
		char	c1 = *s1++;
		char	c2 = *s2++;
		int		diff = __UPPER(c1) - __UPPER(c2);

		if (c1 == '\0' || diff != 0)
			return diff;
	}
}

INLINE_FUNCTION(int)
__strncasecmp(const char *s1, const char *s2, int n)
{
	int		diff;

	while (n > 0)
	{
		char	c1 = *s1++;
		char	c2 = *s2++;

		diff = __UPPER(c1) - __UPPER(c2);
		if (c1 == '\0' || diff != 0)
			return diff;
		n--;
	} while (diff == 0);

	return diff;
}

/*
 * functions in ctype.h
 */
INLINE_FUNCTION(bool)
__isspace(int c)
{
	return (c == ' ' || c == '\t' || c == '\n' || c == '\r');
}

INLINE_FUNCTION(bool)
__isupper(int c)
{
	return (c >= 'A' && c <= 'Z');
}

INLINE_FUNCTION(bool)
__islower(int c)
{
	return (c >= 'a' && c <= 'z');
}

INLINE_FUNCTION(bool)
__isdigit(int c)
{
	return (c >= '0' && c <= '9');
}

INLINE_FUNCTION(bool)
__isxdigit(int c)
{
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

#endif  /* XPU_TEXTLIB_H */
