/*
 * extra.c
 *
 * Stuff related to invoke HeteroDB's commercial features.
 * ----
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
#include "pg_strom.h"
#include "heterodb_extra.h"

/* SQL function declarations */
Datum pgstrom_license_query(PG_FUNCTION_ARGS);

/*
 * heterodb_license_query
 */
static void
heterodb_license_print_v2(StringInfo str, const heterodb_license_info_v2 *linfo)
{
	appendStringInfo(str, "{ \"version\" : %d",
					 linfo->version);
	if (linfo->serial_nr)
		appendStringInfo(str, ", \"serial_nr\" : \"%s\"",
						 linfo->serial_nr);
	appendStringInfo(str, ", \"issued_at\" : \"%04d-%02d-%02d\"",
					 (linfo->issued_at / 10000),
					 (linfo->issued_at / 100) % 100,
					 (linfo->issued_at % 100));
	appendStringInfo(str, ", \"expired_at\" : \"%04d-%02d-%02d\"",
					 (linfo->expired_at / 10000),
					 (linfo->expired_at / 100) % 100,
					 (linfo->expired_at % 100));
	if (linfo->licensee_org)
		appendStringInfo(str, ", \"licensee_org\" : \"%s\"",
						 linfo->licensee_org);
	if (linfo->licensee_name)
		appendStringInfo(str, ", \"licensee_name\" : \"%s\"",
						 linfo->licensee_name);
	if (linfo->licensee_mail)
		appendStringInfo(str, ", \"licensee_mail\" : \"%s\"",
						 linfo->licensee_mail);
	if (linfo->description)
		appendStringInfo(str, ", \"description\" : \"%s\"",
						 linfo->description);
	if (linfo->nr_gpus > 0)
	{
		int		i;

		appendStringInfo(str, ", \"gpus\" : [");
		for (i=0; i < linfo->nr_gpus; i++)
		{
			appendStringInfo(str, "%s{ \"uuid\" : \"%s\" }",
							 i > 0 ? ", " : " ",
							 linfo->gpu_uuid[i]);
		}
		appendStringInfo(str, " ]");
	}
	appendStringInfo(str, "}");
}

static char *
heterodb_license_query(void)
{
	const heterodb_license_info *linfo = heterodbLicenseReload(NULL);
	StringInfoData str;

	if (!linfo)
		return NULL;

	initStringInfo(&str);

	if (linfo->version == 2)
		heterodb_license_print_v2(&str, &linfo->v2);
	else
		elog(ERROR, "unknown license version: %d", linfo->version);
	return str.data;
}

Datum
pgstrom_license_query(PG_FUNCTION_ARGS)
{
	char   *license;

	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 (errmsg("only superuser can query commercial license"))));
	license = heterodb_license_query();
	if (!license)
		PG_RETURN_NULL();
	PG_RETURN_POINTER(DirectFunctionCall1(json_in, PointerGetDatum(license)));
}
PG_FUNCTION_INFO_V1(pgstrom_license_query);

/*
 * pgstrom_init_extra
 */
void
pgstrom_init_extra(void)
{
	if (heterodbExtraInit() == 0)
	{
		char   *license = heterodb_license_query();

		if (license)
		{
			elog(LOG, "HeteroDB License: %s", license);
			pfree(license);
		}
	}
}
