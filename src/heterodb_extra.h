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
#include <dlfcn.h>

#ifndef offsetof
#define offsetof(type, field)   ((long) &((type *)0)->field)
#endif

/* commercial license validation */
typedef struct
{
	uint32_t	version;
	const char *serial_nr;
	uint32_t	issued_at;		/* YYYYMMDD */
	uint32_t	expired_at;		/* YYYYMMDD */
	const char *licensee_org;
	const char *licensee_name;
	const char *licensee_mail;
	const char *description;
	uint32_t	nr_gpus;
	const char *gpu_uuid[1];
} heterodb_license_info_v2;

typedef union
{
	uint32_t	version;
	heterodb_license_info_v2 v2;
} heterodb_license_info;

static inline heterodb_license_info *
heterodb_license_parse_version2(char *license)
{
	heterodb_license_info_v2 linfo;
	char   *key, *val, *pos;
	int		year, mon, day;
	int		__mdays[] = {31,29,30,30,31,30,31,31,30,31,30,31};
	int		count = 0;
	int		limit = 16;
	char  **gpu_uuid = alloca(sizeof(char *) * limit);
	int		i, extra = 0;
	heterodb_license_info *res;

	memset(&linfo, 0, sizeof(heterodb_license_info_v2));
	for (key = strtok_r(license, "\n", &pos);
		 key != NULL;
		 key = strtok_r(NULL, "\n", &pos))
	{
		val = strchr(key, '=');
		if (!val)
			return NULL;
		*val++ = '\0';

		if (strcmp(key, "VERSION") == 0)
		{
			if (linfo.version != 0)
				return NULL;
			linfo.version = atoi(val);
		}
		else if (strcmp(key, "SERIAL_NR") == 0)
		{
			if (linfo.serial_nr)
				return NULL;
			linfo.serial_nr = val;
			extra += strlen(val) + 1;
		}
		else if (strcmp(key, "ISSUED_AT") == 0)
		{
			if (linfo.issued_at != 0)
				return NULL;
			if (sscanf(val, "%d-%d-%d", &year, &mon, &day) != 3)
				return NULL;
			if (year < 2000 || year >= 3000 ||
				mon  < 1    || mon  >  12 ||
				day  < 1    || day  >  __mdays[mon-1])
				return NULL;	/* invalid YYYY-MM-DD */
			linfo.issued_at = 10000 * year + 100 * mon + day;
		}
		else if (strcmp(key, "EXPIRED_AT") == 0)
		{
			if (linfo.expired_at != 0)
				return NULL;
			if (sscanf(val, "%d-%d-%d", &year, &mon, &day) != 3)
				return NULL;
			if (year < 2000 || year >= 3000 ||
				mon  < 1    || mon  >  12 ||
				day  < 1    || day  >  __mdays[mon-1])
				return NULL;	/* invalid YYYY-MM-DD */
            linfo.expired_at = 10000 * year + 100 * mon + day;
		}
		else if (strcmp(key, "LICENSEE_ORG") == 0)
		{
			if (linfo.licensee_org)
				return NULL;
			linfo.licensee_org = val;
			extra += strlen(val) + 1;
		}
		else if (strcmp(key, "LICENSEE_NAME") == 0)
		{
			if (linfo.licensee_name)
				return NULL;
			linfo.licensee_name = val;
			extra += strlen(val) + 1;
		}
		else if (strcmp(key, "LICENSEE_MAIL") == 0)
		{
			if (linfo.licensee_mail)
				return NULL;
			linfo.licensee_mail = val;
			extra += strlen(val) + 1;
		}
		else if (strcmp(key, "DESCRIPTION") == 0)
		{
			if (linfo.description)
				return NULL;
			linfo.description = val;
			extra += strlen(val) + 1;
		}
		else if (strcmp(key, "NR_GPUS") == 0)
		{
			if (linfo.nr_gpus != 0)
				return NULL;
			linfo.nr_gpus = atoi(val);
		}
		else if (strcmp(key, "GPU_UUID") == 0)
		{
			if (count == limit)
			{
				char  **tmp_uuid = alloca(sizeof(char *) * 2 * limit);

				memcpy(tmp_uuid, gpu_uuid, sizeof(char *) * count);
				gpu_uuid = tmp_uuid;
				limit *= 2;
			}
			gpu_uuid[count++] = val;
			extra += strlen(val) + 1;
		}
		else
		{
			return NULL;	/* unknown KEY=VAL */
		}
	}

	if (linfo.version != 2 ||
		linfo.serial_nr == NULL ||
		linfo.issued_at == 0 ||
		linfo.expired_at == 0 ||
		linfo.nr_gpus != count)
		return NULL;		/* not a valid license info v2 */

	res = calloc(1, offsetof(heterodb_license_info_v2,
							 gpu_uuid[count]) + extra);
	if (!res)
		return NULL;		/* out of memory */
	pos = (char *)&res->v2.gpu_uuid[count];

	res->v2.version = linfo.version;

	res->v2.serial_nr = pos;
	strcpy(pos, linfo.serial_nr);
	pos += strlen(pos) + 1;

	res->v2.issued_at = linfo.issued_at;
	res->v2.expired_at = linfo.expired_at;

	if (linfo.licensee_org)
	{
		res->v2.licensee_org = pos;
		strcpy(pos, linfo.licensee_org);
		pos += strlen(pos) + 1;
	}

	if (linfo.licensee_name)
	{
		res->v2.licensee_name = pos;
		strcpy(pos, linfo.licensee_name);
		pos += strlen(pos) + 1;
	}

	if (linfo.licensee_mail)
	{
		res->v2.licensee_mail = pos;
		strcpy(pos, linfo.licensee_mail);
		pos += strlen(pos) + 1;
	}

	if (linfo.description)
	{
		res->v2.description = pos;
		strcpy(pos, linfo.description);
		pos += strlen(pos) + 1;
	}

	res->v2.nr_gpus = linfo.nr_gpus;
	for (i=0; i < count; i++)
	{
		res->v2.gpu_uuid[i] = pos;
		strcpy(pos, gpu_uuid[i]);
		pos += strlen(pos) + 1;
	}
	fprintf(stderr, "base = %p, extra = %u, pos = %p %p\n",
			(char *)&res->v2.gpu_uuid[count], extra, pos, (char *)&res->v2.gpu_uuid[count] + extra);
	assert((char *)&res->v2.gpu_uuid[count] + extra == pos);

	return res;
}

static inline heterodb_license_info *
heterodb_license_reload(void)
{
	char	 *(*license_reload)(void) = NULL;
	void	   *handle;
	char	   *license = NULL;
	heterodb_license_info *result;

	/* fetch the latest license info */
	handle = dlopen(HETERODB_EXTRA_LIBNAME, RTLD_NOW | RTLD_LOCAL);
	if (!handle)
		return NULL;
    license_reload = dlsym(handle, "heterodb_license_reload");
	if (license_reload)
		license = license_reload();
	dlclose(handle);
	if (!license)
		return NULL;
	/* parse the plain license text */
	if (strncmp(license, "VERSION=2\n", 10) == 0)
		result = heterodb_license_parse_version2(license);
	else
		result = NULL;
	free(license);

	return result;
}

#endif	/* HETERODB_EXTRA_H */
