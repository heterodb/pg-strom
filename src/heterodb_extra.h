/*
 * heterodb_extra.h
 *
 * Definitions of HeteroDB Extra Package
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef HETERODB_EXTRA_H
#include <stdint.h>

#ifndef offsetof
#define offsetof(type, field)   ((long) &((type *)0)->field)
#endif

/* commercial license validation */
typedef struct
{
	uint32_t	version;		/* =2 */
	time_t		timestamp;
	const char *serial_nr;
	uint32_t	issued_at;		/* YYYYMMDD */
	uint32_t	expired_at;		/* YYYYMMDD */
	const char *licensee_org;
	const char *licensee_name;
	const char *licensee_mail;
	const char *description;
	uint32_t	nr_gpus;
	const char *gpu_uuid[1];	/* variable length */
} heterodb_license_info_v2;

typedef union
{
	uint32_t	version;
	heterodb_license_info_v2 v2;
} heterodb_license_info;

/* license query */
extern heterodb_license_info *heterodb_license_reload(FILE *out);

#ifndef HETERODB_EXTRA_VERSION
#include <dlfcn.h>

static void	   *heterodb_extra_handle = NULL;

heterodb_license_info *(*p__heterodb_license_reload)(FILE *out) = NULL;
static inline const heterodb_license_info *
heterodbLicenseReload(FILE *out)
{
	if (!p__heterodb_license_reload)
		return NULL;
	return p__heterodb_license_reload(out);
}

static inline int
heterodbExtraInit(void)
{
	int		__errno;

	if (heterodb_extra_handle)
		return 0;		/* already loaded */

	heterodb_extra_handle = dlopen("heterodb_extra.so",
								   RTLD_NOW | RTLD_LOCAL);
	if (!heterodb_extra_handle)
	{
		heterodb_extra_handle = dlopen("/usr/lib64/heterodb_extra.so",
									   RTLD_NOW | RTLD_LOCAL);
		if (!heterodb_extra_handle)
			return ENOENT;
	}
	p__heterodb_license_reload = dlsym(heterodb_extra_handle,
									   "heterodb_license_reload");
	if (!p__heterodb_license_reload)
		goto error;
	return 0;

error:
	__errno = (errno ? errno : -1);

	p__heterodb_license_reload = NULL;
	dlclose(heterodb_extra_handle);
	heterodb_extra_handle = NULL;
	return __errno;
}
#endif	/* HETERODB_EXTRA_VERSION */
#endif	/* HETERODB_EXTRA_H */
