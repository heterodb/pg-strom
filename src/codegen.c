/*
 * codegen.c
 *
 * Routines for CUDA code generator
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_numeric.h"
#include "cuda_postgis.h"

static MemoryContext	devinfo_memcxt;
static dlist_head	devtype_info_slot[128];
static dlist_head	devfunc_info_slot[1024];
static dlist_head	devcast_info_slot[48];
static dlist_head	devindex_info_slot[48];

static cl_uint generic_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_int1_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_int2_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_int4_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_int8_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_float2_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_float4_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_float8_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_numeric_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_interval_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_bpchar_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_inet_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_jsonb_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_range_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_geometry_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_box2df_devtype_hashfunc(devtype_info *dtype, Datum datum);

/* callback to handle special cases of device cast */
static int	devcast_text2numeric_callback(codegen_context *context,
										  StringInfo body,
										  devcast_info *dcast,
										  CoerceViaIO *node);
/* error report */
#define __ELog(fmt, ...)								\
	ereport(ERROR,										\
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),	\
			 errmsg((fmt), ##__VA_ARGS__)))

/* known extension name */
#define PGSTROM		"pg_strom"
#define POSTGIS3	"postgis"

/*
 * Catalog of data types supported by device code
 *
 * naming convension of types:
 *   pg_<type_name>_t
 */

/*
 * MEMO: PG10 does not have OID definitions below
 */
#ifndef INT8RANGEOID
#define INT8RANGEOID	3926
#endif
#ifndef TSRANGEOID
#define TSRANGEOID		3908
#endif
#ifndef TSTZRANGEOID
#define TSTZRANGEOID	3910
#endif
#ifndef DATERANGEOID
#define DATERANGEOID	3912
#endif

static struct {
	const char	   *type_extension;
	const char	   *type_name;
	Oid				type_oid_fixed;	/* can be InvalidOid if not build-in */
	const char	   *type_oid_label;
	cl_uint			type_flags;		/* library to declare this type */
	cl_uint			extra_sz;		/* required size to store internal form */
	devtype_hashfunc_type hash_func;
} devtype_catalog[] = {
	/*
	 * Primitive datatypes
	 */
	{ NULL, "bool", BOOLOID, "BOOLOID",
	  0, 0, generic_devtype_hashfunc
	},
	{ PGSTROM, "int1", INT1OID, "INT1OID",
	  0, 0, pg_int1_devtype_hashfunc
	},
	{ NULL, "int2", INT2OID, "INT2OID",
	  0, 0, pg_int2_devtype_hashfunc
	},
	{ NULL, "int4", INT4OID, "INT4OID",
	  0, 0, pg_int4_devtype_hashfunc
	},
	{ NULL, "int8", INT8OID, "INT8OID",
	  0, 0, pg_int8_devtype_hashfunc
	},
	/* XXX - float2 is not a built-in data type */
	{ PGSTROM, "float2", FLOAT2OID, "FLOAT2OID",
	  0, 0, pg_float2_devtype_hashfunc
	},
	{ NULL, "float4", FLOAT4OID, "FLOAT4OID",
	  0, 0, pg_float4_devtype_hashfunc
	},
	{ NULL, "float8", FLOAT8OID, "FLOAT8OID",
	  0, 0, pg_float8_devtype_hashfunc
	},
	/*
	 * Misc data types
	 */
	{ NULL, "money", CASHOID, "CASHOID",
	  DEVKERNEL_NEEDS_MISCLIB, 0,
	  generic_devtype_hashfunc
	},
	{ NULL, "uuid", UUIDOID, "UUIDOID",
	  DEVKERNEL_NEEDS_MISCLIB, UUID_LEN,
	  generic_devtype_hashfunc
	},
	{ NULL, "macaddr", MACADDROID, "MACADDROID",
	  DEVKERNEL_NEEDS_MISCLIB, sizeof(macaddr),
	  generic_devtype_hashfunc
	},
	{ NULL, "inet", INETOID, "INETOID",
	  DEVKERNEL_NEEDS_MISCLIB, sizeof(inet),
	  pg_inet_devtype_hashfunc
	},
	{ NULL, "cidr", CIDROID, "CIDROID",
	  DEVKERNEL_NEEDS_MISCLIB, sizeof(inet),
	  pg_inet_devtype_hashfunc
	},
	/*
	 * Date and time datatypes
	 */
	{ NULL, "date", DATEOID, "DATEOID",
	  DEVKERNEL_NEEDS_TIMELIB, 0,
	  generic_devtype_hashfunc
	},
	{ NULL, "time", TIMEOID, "TIMEOID",
	  DEVKERNEL_NEEDS_TIMELIB, 0,
	  generic_devtype_hashfunc
	},
	{ NULL, "timetz", TIMETZOID, "TIMETZOID",
	  DEVKERNEL_NEEDS_TIMELIB, sizeof(TimeTzADT),
	  generic_devtype_hashfunc
	},
	{ NULL, "timestamp", TIMESTAMPOID, "TIMESTAMPOID",
	  DEVKERNEL_NEEDS_TIMELIB, 0,
	  generic_devtype_hashfunc
	},
	{ NULL, "timestamptz", TIMESTAMPTZOID, "TIMESTAMPTZOID",
	  DEVKERNEL_NEEDS_TIMELIB, 0,
	  generic_devtype_hashfunc
	},
	{ NULL, "interval", INTERVALOID, "INTERVALOID",
	  DEVKERNEL_NEEDS_TIMELIB, sizeof(Interval),
	  pg_interval_devtype_hashfunc
	},
	/*
	 * variable length datatypes
	 */
	{ NULL, "bpchar", BPCHAROID, "BPCHAROID",
	  DEVKERNEL_NEEDS_TEXTLIB, 0,
	  pg_bpchar_devtype_hashfunc
	},
	{ NULL, "varchar", VARCHAROID, "VARCHAROID",
	  DEVKERNEL_NEEDS_TEXTLIB, 0,
	  generic_devtype_hashfunc
	},
	{ NULL, "numeric", NUMERICOID, "NUMERICOID",
	  0, sizeof(struct NumericData),
	  pg_numeric_devtype_hashfunc
	},
	{ NULL, "bytea", BYTEAOID, "BYTEAOID",
	  0, sizeof(pg_varlena_t),
	  generic_devtype_hashfunc
	},
	{ NULL, "text", TEXTOID, "TEXTOID",
	  DEVKERNEL_NEEDS_TEXTLIB, sizeof(pg_varlena_t),
	  generic_devtype_hashfunc
	},
	{ NULL, "jsonb", JSONBOID, "JSONBOID",
	  DEVKERNEL_NEEDS_JSONLIB,
	  /* see comment at vlbuf_estimate_jsonb() */
	  TOAST_TUPLE_THRESHOLD,
	  pg_jsonb_devtype_hashfunc
	},
	/*
	 * range types
	 */
	{ NULL, "int4range", INT4RANGEOID, "INT4RANGEOID",
	  DEVKERNEL_NEEDS_RANGETYPE,
	  sizeof(RangeType) + 2 * sizeof(cl_int) + 1,
	  pg_range_devtype_hashfunc
	},
	{ NULL, "int8range", INT8RANGEOID, "INT8RANGEOID",
	  DEVKERNEL_NEEDS_RANGETYPE,
	  sizeof(RangeType) + 2 * sizeof(cl_long) + 1,
	  pg_range_devtype_hashfunc
	},
	{ NULL, "tsrange", TSRANGEOID, "TSRANGEOID",
	  DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
	  sizeof(RangeType) + 2 * sizeof(Timestamp) + 1,
	  pg_range_devtype_hashfunc
	},
	{ NULL, "tstzrange", TSTZRANGEOID, "TSTZRANGEOID",
	  DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
	  sizeof(RangeType) + 2 * sizeof(TimestampTz) + 1,
	  pg_range_devtype_hashfunc
	},
	{ NULL, "daterange", DATERANGEOID, "DATERANGEOID",
	  DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
	  sizeof(RangeType) + 2 * sizeof(DateADT) + 1,
	  pg_range_devtype_hashfunc
	},
	/*
	 * PostGIS types
	 */
	{ POSTGIS3, "geometry", InvalidOid, "GEOMETRYOID",
	  DEVKERNEL_NEEDS_POSTGIS,
	  sizeof(pg_geometry_t),
	  pg_geometry_devtype_hashfunc
	},
	{ POSTGIS3, "box2df", InvalidOid, "BOX2DFOID",
	  DEVKERNEL_NEEDS_POSTGIS,
	  sizeof(pg_box2df_t),
	  pg_box2df_devtype_hashfunc
	}
};

static const char *
get_extension_name_by_object(Oid class_id, Oid object_id)
{
	Relation	rel;
	ScanKeyData	skeys[2];
	SysScanDesc	scan;
	HeapTuple	htup;
	const char *ext_name = NULL;

	ScanKeyInit(&skeys[0],
				Anum_pg_depend_classid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(class_id));
	ScanKeyInit(&skeys[1],
				Anum_pg_depend_objid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(object_id));

	rel = table_open(DependRelationId, AccessShareLock);
	scan = systable_beginscan(rel, DependDependerIndexId, true,
							  NULL, 2, skeys);
	while (HeapTupleIsValid(htup = systable_getnext(scan)))
	{
		Form_pg_depend	dep = (Form_pg_depend) GETSTRUCT(htup);
		const char	   *__ext_name;

		if (dep->refclassid == ExtensionRelationId &&
			dep->deptype == DEPENDENCY_EXTENSION)
		{
			__ext_name = get_extension_name(dep->refobjid);
			if (__ext_name)
				ext_name = quote_identifier(__ext_name);
			break;
		}
	}
	systable_endscan(scan);
	table_close(rel, AccessShareLock);

	return ext_name;
}

static void
append_string_devtype_identifier(StringInfo buf, Oid type_oid)
{
	HeapTuple		htup;
	Form_pg_type	type_form;
	char		   *nsp_name;

	htup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	type_form = (Form_pg_type) GETSTRUCT(htup);

	nsp_name = get_namespace_name(type_form->typnamespace);
	if (!nsp_name)
		elog(ERROR, "cache lookup failed for namespace %u", type_form->typnamespace);
	appendStringInfo(buf, "%s.%s",
					 quote_identifier(nsp_name),
					 quote_identifier(NameStr(type_form->typname)));
	ReleaseSysCache(htup);
}

/*
 * build_extra_devtype_info
 *
 * it queries the extra device type support
 */
static devtype_info *
build_extra_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	StringInfoData	ident;
	devtype_info	__dtype;
	devtype_info   *dtype = NULL;
	int				i;

	/* setup arguments */
	initStringInfo(&ident);
	append_string_devtype_identifier(&ident, tcache->type_id);

	memset(&__dtype, 0, sizeof(devtype_info));
	__dtype.type_extension = ext_name;
	__dtype.type_oid = tcache->type_id;
	__dtype.type_flags = 0;
	__dtype.type_length = tcache->typlen;
	__dtype.type_align = typealign_get_width(tcache->typalign);
	__dtype.type_byval = tcache->typbyval;
	__dtype.type_name = NULL;	/* callback must set the device type name */
	__dtype.extra_sz = 0;
	__dtype.hash_func = NULL;
	__dtype.type_eqfunc = get_opcode(tcache->eq_opr);
	__dtype.type_cmpfunc = tcache->cmp_proc;
	
	for (i=0; i < pgstrom_num_users_extra; i++)
	{
		pgstromUsersExtraDescriptor *extra = &pgstrom_users_extra_desc[i];

		if (extra->lookup_extra_devtype &&
			extra->lookup_extra_devtype(ident.data, &__dtype))
		{
			MemoryContext	oldcxt;

			/* must be still base type */
			Assert(__dtype.type_element == NULL &&
				   __dtype.comp_nfields == 0);
			if (!__dtype.type_name)
			{
				elog(DEBUG2, "Extra module didn't set device type name for '%s'",
					 format_type_be(tcache->type_id));
				continue;
			}
			oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
			dtype = pmemdup(&__dtype, offsetof(devtype_info, comp_subtypes[0]));
			if (__dtype.type_extension)
				dtype->type_extension = pstrdup(__dtype.type_extension);
			dtype->type_name = pstrdup(__dtype.type_name);
			dtype->type_flags |= extra->extra_flags;
			MemoryContextSwitchTo(oldcxt);
			break;
		}
	}
	pfree(ident.data);
	return dtype;
}

static devtype_info *
build_basic_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	HeapTuple		htup;
	Form_pg_type	type_form;
	const char	   *type_name;
	devtype_info   *entry = NULL;
	int				i;

	htup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(tcache->type_id));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for type %u", tcache->type_id);
	type_form = (Form_pg_type) GETSTRUCT(htup);
	type_name = NameStr(type_form->typname);

	for (i=0; i < lengthof(devtype_catalog); i++)
	{
		const char *__ext_name = devtype_catalog[i].type_extension;
		const char *__type_name = devtype_catalog[i].type_name;

		if (ext_name)
		{
			if (!__ext_name || strcmp(ext_name, __ext_name) != 0)
				continue;
		}
		else
		{
			if (__ext_name || type_form->typnamespace != PG_CATALOG_NAMESPACE)
				continue;
		}

		if (strcmp(type_name, __type_name) == 0)
		{
			MemoryContext	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

			entry = palloc0(offsetof(devtype_info, comp_subtypes[0]));
			if (ext_name)
				entry->type_extension = pstrdup(ext_name);
			entry->type_oid = tcache->type_id;
			entry->type_flags = devtype_catalog[i].type_flags;
			entry->type_length = tcache->typlen;
			entry->type_align = typealign_get_width(tcache->typalign);
			entry->type_byval = tcache->typbyval;
			entry->type_name = devtype_catalog[i].type_name; /* const */
			entry->extra_sz = devtype_catalog[i].extra_sz;
			entry->hash_func = devtype_catalog[i].hash_func;
			/* type equality functions */
			entry->type_eqfunc = get_opcode(tcache->eq_opr);
			entry->type_cmpfunc = tcache->cmp_proc;

			MemoryContextSwitchTo(oldcxt);
			break;
		}
	}
	if (!entry && pgstrom_num_users_extra > 0)
		entry = build_extra_devtype_info(tcache, ext_name);
	ReleaseSysCache(htup);

	return entry;
}

static devtype_info *
build_array_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	devtype_info   *element;
	devtype_info   *entry;
	Oid				typelem;
	MemoryContext	oldcxt;

	typelem = get_element_type(tcache->type_id);
	Assert(OidIsValid(typelem) && tcache->typlen == -1);
	element = pgstrom_devtype_lookup(typelem);
	if (!element)
		return NULL;

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	entry = palloc0(offsetof(devtype_info, comp_subtypes[0]));
	if (ext_name)
		entry->type_extension = pstrdup(ext_name);
	entry->type_oid = tcache->type_id;
	entry->type_flags = element->type_flags;
	entry->type_length = tcache->typlen;
	entry->type_align = typealign_get_width(tcache->typalign);
	entry->type_byval = tcache->typbyval;
	entry->type_name = "array";
	entry->extra_sz = sizeof(pg_array_t);
	entry->hash_func = generic_devtype_hashfunc;
	entry->type_element = element;
	MemoryContextSwitchTo(oldcxt);

	return entry;
}

static devtype_info *
build_composite_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	Oid				type_relid = tcache->typrelid;
	int				j, nfields = get_relnatts(type_relid);
	devtype_info  **subtypes = alloca(sizeof(devtype_info *) * nfields);
	devtype_info   *entry;
	cl_uint			extra_flags = 0;
	size_t			extra_sz;
	MemoryContext	oldcxt;

	extra_sz = (MAXALIGN(sizeof(Datum) * nfields) +
				MAXALIGN(sizeof(bool) * nfields));
	for (j=0; j < nfields; j++)
	{
		HeapTuple		tup;
		Oid				atttypid;
		devtype_info   *dtype;

		tup = SearchSysCache2(ATTNUM,
							  ObjectIdGetDatum(type_relid),
							  Int16GetDatum(j+1));
		if (!HeapTupleIsValid(tup))
			return NULL;
		atttypid = ((Form_pg_attribute) GETSTRUCT(tup))->atttypid;
		ReleaseSysCache(tup);

		dtype = pgstrom_devtype_lookup(atttypid);
		if (!dtype)
			return NULL;
		subtypes[j] = dtype;

		extra_flags |= dtype->type_flags;
		extra_sz    += MAXALIGN(dtype->extra_sz);
	}

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	entry = palloc0(offsetof(devtype_info, comp_subtypes[nfields]));
	if (ext_name)
		entry->type_extension = pstrdup(ext_name);
	entry->type_oid = tcache->type_id;
	entry->type_flags = extra_flags;
	entry->type_length = tcache->typlen;
	entry->type_align = typealign_get_width(tcache->typalign);
	entry->type_byval = tcache->typbyval;
	entry->type_name = "composite";
	entry->extra_sz = extra_sz;
	entry->comp_nfields = nfields;
	memcpy(entry->comp_subtypes, subtypes,
		   sizeof(devtype_info *) * nfields);
	MemoryContextSwitchTo(oldcxt);

	return entry;
}

devtype_info *
pgstrom_devtype_lookup(Oid type_oid)
{
	TypeCacheEntry *tcache;
	devtype_info   *dtype;
	uint32			hashvalue;
	uint32			hindex;
	size_t			sz;
	dlist_iter		iter;
	const char	   *ext_name;

	/* lookup dtype that is already built */
	hashvalue = GetSysCacheHashValue(TYPEOID, ObjectIdGetDatum(type_oid), 0, 0, 0);
	hindex = hashvalue % lengthof(devtype_info_slot);
	dlist_foreach(iter, &devtype_info_slot[hindex])
	{
		dtype = dlist_container(devtype_info, chain, iter.cur);

		if (dtype->type_oid == type_oid)
		{
			if (dtype->type_is_negative)
				return NULL;
			return dtype;
		}
	}
	/* try to build devtype_info entry */
	ext_name = get_extension_name_by_object(TypeRelationId, type_oid);
	tcache = lookup_type_cache(type_oid,
							   TYPECACHE_EQ_OPR |
							   TYPECACHE_CMP_PROC);
	if (OidIsValid(tcache->typrelid))
	{
		/* composite type */
		dtype = build_composite_devtype_info(tcache, ext_name);
	}
	else if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		/* array type */
		dtype = build_array_devtype_info(tcache, ext_name);
	}
	else
	{
		/* base or extra type */
		dtype = build_basic_devtype_info(tcache, ext_name);
	}
	
	/* makes a negative entry, if not in the catalog */
	if (!dtype)
	{
		sz = offsetof(devtype_info, comp_subtypes[0]);
		dtype = MemoryContextAllocZero(devinfo_memcxt, sz);
		dtype->type_oid = type_oid;
		dtype->type_is_negative = true;
	}
	dtype->hashvalue = hashvalue;
	dlist_push_head(&devtype_info_slot[hindex], &dtype->chain);

	if (dtype->type_is_negative)
		return NULL;
	return dtype;
}

devtype_info *
pgstrom_devtype_lookup_and_track(Oid type_oid, codegen_context *context)
{
	devtype_info   *dtype = pgstrom_devtype_lookup(type_oid);

	if (dtype)
		context->extra_flags |= dtype->type_flags;

	return dtype;
}

static devtype_info *
pgstrom_devtype_lookup_by_name(const char *type_ident)
{
	char	   *type_name = NULL;
	char	   *ext_name = NULL;
	const char *__ext_name;
	Oid			type_oid = InvalidOid;
	Relation	rel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	htup;

	type_name = alloca(strlen(type_ident) + 1);
	strcpy(type_name, type_ident);
	ext_name = strchr(type_name, '@');
	if (ext_name)
		*ext_name++ = '\0';

	htup = SearchSysCache2(TYPENAMENSP,
						   CStringGetDatum(type_name),
						   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (HeapTupleIsValid(htup))
	{
		type_oid = PgTypeTupleGetOid(htup);
		__ext_name = get_extension_name_by_object(TypeRelationId, type_oid);
		if (ext_name)
		{
			if (!__ext_name || strcmp(ext_name, __ext_name) != 0)
				type_oid = InvalidOid;
		}
		else if (__ext_name != NULL)
			type_oid = InvalidOid;
		ReleaseSysCache(htup);
	}

	if (!OidIsValid(type_oid))
	{
		rel = table_open(TypeRelationId, AccessShareLock);

		ScanKeyInit(&skey,
					Anum_pg_type_typname,
					BTEqualStrategyNumber, F_NAMEEQ,
					CStringGetDatum(type_name));
		sscan = systable_beginscan(rel, TypeNameNspIndexId,
								   true, NULL, 1, &skey);
		do {
			htup = systable_getnext(sscan);
			if (!HeapTupleIsValid(htup))
				break;
			type_oid = PgTypeTupleGetOid(htup);
			__ext_name = get_extension_name_by_object(TypeRelationId, type_oid);
			if (ext_name)
			{
				if (!__ext_name || strcmp(ext_name, __ext_name) != 0)
					type_oid = InvalidOid;
			}
			else if (__ext_name != NULL)
				type_oid = InvalidOid;
		} while (!OidIsValid(type_oid));

		systable_endscan(sscan);
		table_close(rel, AccessShareLock);
	}

	if (OidIsValid(type_oid))
		return pgstrom_devtype_lookup(type_oid);
	return NULL;
}

/* code for extra device types */
size_t
pgstrom_codegen_extra_devtypes(char *buf, size_t bufsz, uint32 extra_flags)
{
	size_t		off = 0;
	int			i;

	/* only extra device types */
	extra_flags &= DEVKERNEL_USERS_EXTRA_MASK;

	for (i=0; i < pgstrom_num_users_extra; i++)
	{
		pgstromUsersExtraDescriptor *ex_desc = &pgstrom_users_extra_desc[i];

		if ((ex_desc->extra_flags & extra_flags) == ex_desc->extra_flags)
		{
			off += snprintf(buf + off, bufsz - off,
							"#include \"%s.h\"\n",
							ex_desc->extra_name);
		}
	}
	/* array type support */
	off += snprintf(
		buf + off, bufsz - off,
		"\n"
		"DEVICE_FUNCTION(cl_uint)\n"
		"pg_extras_array_from_arrow(kern_context *kcxt,\n"
		"                           char *dest,\n"
		"                           kern_colmeta *smeta,\n"
		"                           char *base,\n"
		"                           cl_uint start,\n"
		"                           cl_uint end)\n"
		"{\n");
	if (pgstrom_num_users_extra > 0)
	{
		off += snprintf(
			buf + off, bufsz - off,
			"  switch (smeta->atttypid)\n"
			"  {\n");
		for (i=0; i < lengthof(devtype_info_slot); i++)
		{
			dlist_iter		iter;
			devtype_info   *dtype;

			dlist_foreach(iter, &devtype_info_slot[i])
			{
				dtype = dlist_container(devtype_info, chain, iter.cur);
				if ((dtype->type_flags & extra_flags) == 0)
					continue;
				off += snprintf(
					buf + off, bufsz - off,
					"  case %u:\n"
					"    return pg_%s_array_from_arrow(kcxt, dest,\n"
					"                                  smeta, base,\n"
					"                                  start, end);\n",
					dtype->type_oid,
					dtype->type_name);
			}
		}
		off += snprintf(
			buf + off, bufsz - off,
			"  default:\n"
			"    break;\n"
			"  }\n");
	}
	off += snprintf(
		buf + off, bufsz - off,
		"  return 0;\n"
		"}\n");

	/* composite type support */
	off += snprintf(
		buf + off, bufsz - off,
		"\n"
		"DEVICE_FUNCTION(cl_bool)\n"
		"pg_extras_composite_from_arrow(kern_context *kcxt,\n"
		"                               kern_colmeta *smeta,\n"
		"                               char *base,\n"
		"                               cl_uint rowidx,\n"
		"                               cl_char *p_dclass,\n"
		"                               Datum *p_datum)\n"
		"{\n");

	if (pgstrom_num_users_extra > 0)
	{
		off += snprintf(
			buf + off, bufsz - off,
			"  switch (smeta->atttypid)\n"
			"  {\n");
		for (i=0; i < lengthof(devtype_info_slot); i++)
		{
			dlist_iter		iter;
			devtype_info   *dtype;

			dlist_foreach(iter, &devtype_info_slot[i])
			{
				dtype = dlist_container(devtype_info, chain, iter.cur);
				if ((dtype->type_flags & extra_flags) == 0)
					continue;

				off += snprintf(
					buf + off, bufsz - off,
					"  case %u: {\n"
					"    pg_%s_t temp;\n"
					"    pg_datum_fetch_arrow(kcxt, temp, smeta, base, rowidx);\n"
					"    pg_datum_store(kcxt, temp, p_dclass, p_datum);\n"
					"    return true;\n"
					"  }\n",
					dtype->type_oid,
					dtype->type_name);
			}
		}
		off += snprintf(
			buf + off, bufsz - off,
			"  default:\n"
			"    break;\n"
			"  }\n");
	}
	off += snprintf(
		buf + off, bufsz - off,
		"  return false;\n"
		"}\n");
	return off;
}

/*
 * Device type specific hash-functions
 *
 * Some device types have internal representation, like numeric, which shall
 * be used to GpuHashJoin for join-key hashing.
 */
static cl_uint
generic_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	if (dtype->type_byval)
		return hash_any((unsigned char *)&datum, dtype->type_length);
	if (dtype->type_length > 0)
		return hash_any((unsigned char *)DatumGetPointer(datum),
						dtype->type_length);
	Assert(dtype->type_length == -1);
	return hash_any((cl_uchar *)VARDATA_ANY(datum),
					VARSIZE_ANY_EXHDR(datum));
}

static cl_uint
pg_int1_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	cl_int		ival = DatumGetChar(datum);

	return hash_any((cl_uchar *)&ival, sizeof(cl_char));
}

static cl_uint
pg_int2_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	cl_int		ival = DatumGetInt16(datum);

	return hash_any((cl_uchar *)&ival, sizeof(cl_short));
}

static cl_uint
pg_int4_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	cl_int		ival = DatumGetInt32(datum);

	return hash_any((cl_uchar *)&ival, sizeof(cl_int));
}

static cl_uint
pg_int8_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	cl_long		ival = DatumGetInt64(datum);
	cl_uint		lo = (ival & 0xffffffffL);
	cl_uint		hi = (ival >> 32);

	lo ^= (ival >= 0 ? hi : ~hi);

	return hash_any((cl_uchar *)&lo, sizeof(cl_int));
}

extern Datum	pgstrom_float2_to_float8(PG_FUNCTION_ARGS);
static cl_uint
pg_float2_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	Datum		v = DirectFunctionCall1(pgstrom_float2_to_float8, datum);
	cl_double	fval = DatumGetFloat8(v);

	if (fval == 0.0)
		return 0;
	return hash_any((cl_uchar *)&fval, sizeof(cl_double));
}

static cl_uint
pg_float4_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	cl_double	fval = DatumGetFloat4(datum);

	if (fval == 0.0)
		return 0;
	return hash_any((cl_uchar *)&fval, sizeof(cl_double));
}

static cl_uint
pg_float8_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	cl_double	fval = DatumGetFloat8(datum);

	if (fval == 0.0)
		return 0;
	return hash_any((cl_uchar *)&fval, sizeof(cl_double));
}

static cl_uint
pg_numeric_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	kern_context	dummy;
	pg_numeric_t	temp;

	memset(&dummy, 0, sizeof(dummy));
	/*
	 * MEMO: If NUMERIC value is out of range, we may not be able to
	 * execute GpuJoin in the kernel space for all the outer chunks.
	 * Is it still valuable to run on GPU kernel?
	 */
	temp = pg_numeric_from_varlena(&dummy, (struct varlena *)
								   DatumGetPointer(datum));
	if (dummy.errcode != ERRCODE_STROM_SUCCESS)
		elog(ERROR, "failed on hash calculation of device numeric: %s",
			 DatumGetCString(DirectFunctionCall1(numeric_out, datum)));

	return hash_any((cl_uchar *)&temp.value,
					offsetof(pg_numeric_t, weight) + sizeof(cl_short));
}

static cl_uint
pg_interval_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	Interval   *interval = DatumGetIntervalP(datum);
	cl_long		frac;
	cl_long		days;

	frac = interval->time % USECS_PER_DAY;
	days = (interval->time / USECS_PER_DAY +
			interval->month * 30L +
			interval->day);
	days ^= frac;

	return hash_any((cl_uchar *)&days, sizeof(cl_long));
}

static cl_uint
pg_bpchar_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	char   *s = VARDATA_ANY(datum);
	int		i, len = VARSIZE_ANY_EXHDR(datum);

	Assert(dtype->type_oid == BPCHAROID);
	/*
	 * whitespace is the tail end of CHAR(n) data shall be ignored
	 * when we calculate hash-value, to match same text exactly.
	 */
	for (i = len - 1; i >= 0 && s[i] == ' '; i--)
		;
	return hash_any((unsigned char *)VARDATA_ANY(datum), i+1);
}

static cl_uint
pg_inet_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	inet_struct *is = (inet_struct *) VARDATA_ANY(datum);

	Assert(dtype->type_oid == INETOID ||
		   dtype->type_oid == CIDROID);
	if (is->family == PGSQL_AF_INET)
		return hash_any((cl_uchar *)is, offsetof(inet_struct, ipaddr[4]));
	else if (is->family == PGSQL_AF_INET6)
		return hash_any((cl_uchar *)is, offsetof(inet_struct, ipaddr[16]));

	elog(ERROR, "unexpected address family: %d", is->family);
	return ~0U;
}

static cl_uint
__jsonb_devtype_hashfunc(devtype_info *dtype, JsonbContainer *jc)
{
	cl_uint		hash = 0;
	cl_uint		j, nitems = JsonContainerSize(jc);
	char	   *base = NULL;
	char	   *data;
	cl_uint		datalen;

	if (!JsonContainerIsScalar(jc))
	{
		if (JsonContainerIsObject(jc))
		{
			base = (char *)(jc->children + 2 * nitems);
			hash ^= JB_FOBJECT;
		}
		else
		{
			base = (char *)(jc->children + nitems);
			hash ^= JB_FARRAY;
		}
	}

	for (j=0; j < nitems; j++)
	{
		cl_uint		index = j;
		cl_uint		temp;
		JEntry		entry;

		/* hash value for key */
		if (JsonContainerIsObject(jc))
		{
			entry = jc->children[index];
			if (!JBE_ISSTRING(entry))
				elog(ERROR, "jsonb key value is not STRING");
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			temp = hash_any((cl_uchar *)data, datalen);
			hash = ((hash << 1) | (hash >> 31)) ^ temp;

			index += nitems;
		}
		/* hash value for element */
		entry = jc->children[index];
		if (JBE_ISNULL(entry))
			temp = 0x01;
		else if (JBE_ISSTRING(entry))
		{
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			temp = hash_any((cl_uchar *)data, datalen);
		}
		else if (JBE_ISNUMERIC(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));
			temp = pg_numeric_devtype_hashfunc(NULL, PointerGetDatum(data));
		}
		else if (JBE_ISBOOL_TRUE(entry))
			temp = 0x02;
		else if (JBE_ISBOOL_FALSE(entry))
			temp = 0x04;
		else if (JBE_ISCONTAINER(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));
			temp = __jsonb_devtype_hashfunc(dtype, (JsonbContainer *)data);
		}
        else
			elog(ERROR, "Unexpected jsonb entry (%08x)", entry);
		hash = ((hash << 1) | (hash >> 31)) ^ temp;
	}
	return hash;
}

static cl_uint
pg_jsonb_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	JsonbContainer *jc = (JsonbContainer *) VARDATA_ANY(datum);

	return __jsonb_devtype_hashfunc(dtype, jc);
}

static cl_uint
pg_range_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	RangeType	   *r = DatumGetRangeTypeP(datum);
	cl_uchar		flags = *((char *)r + VARSIZE_ANY(r) - 1);
	cl_uchar	   *pos = (cl_uchar *)(r + 1);
	struct {
		Datum		l_val;
		Datum		u_val;
		cl_uchar	flags;
	} temp;
	int32			ival32;

	if (RANGE_HAS_LBOUND(flags))
	{
		switch (RangeTypeGetOid(r))
		{
			case INT4RANGEOID:
			case DATERANGEOID:
				memcpy(&ival32, pos, sizeof(cl_int));
				temp.l_val = (cl_long)ival32;
				pos += sizeof(cl_int);
				break;
			case INT8RANGEOID:
			case TSRANGEOID:
			case TSTZRANGEOID:
				memcpy(&temp.l_val, pos, sizeof(cl_long));
				pos += sizeof(cl_long);
				break;
			default:
				elog(ERROR, "unexpected range type: %s",
					 format_type_be(RangeTypeGetOid(r)));
		}
	}
	if (RANGE_HAS_UBOUND(flags))
	{
		switch (RangeTypeGetOid(r))
		{
			case INT4RANGEOID:
			case DATERANGEOID:
				memcpy(&ival32, pos, sizeof(cl_int));
				temp.l_val = (cl_long)ival32;
				pos += sizeof(cl_int);
				break;
			case INT8RANGEOID:
			case TSRANGEOID:
			case TSTZRANGEOID:
				memcpy(&temp.l_val, pos, sizeof(cl_long));
				pos += sizeof(cl_long);
				break;
			default:
				elog(ERROR, "unexpected range type: %s",
					 format_type_be(RangeTypeGetOid(r)));
		}
	}
	temp.flags = flags;

	return hash_any((unsigned char *)&temp,
					2*sizeof(Datum)+sizeof(cl_uchar));
}

static cl_uint
pg_geometry_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	return 0; //TODO
}

static cl_uint
pg_box2df_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	return 0; //TODO
}

/*
 * varlena buffer estimation handler
 */
static int
vlbuf_estimate_textcat(codegen_context *context,
					   devfunc_info *dfunc,
					   Expr **args, int *vl_width)
{
	int		i, nargs = list_length(dfunc->func_args);
	int		maxlen = 0;

	for (i=0; i < nargs; i++)
	{
		if (vl_width[i] < 0)
			__ELog("unable to estimate result size of textcat");
		maxlen += vl_width[i];
	}
	/* it consumes varlena buffer on run-time */
	context->extra_bufsz += MAXALIGN(maxlen + VARHDRSZ);

	return maxlen;
}

static int
vlbuf_estimate_substring(codegen_context *context,
						 devfunc_info *dfunc,
						 Expr **args, int *vl_width)
{
	if (list_length(dfunc->func_args) > 2 &&
		IsA(args[2], Const))
	{
		Const  *con = (Const *)args[2];

		Assert(con->consttype == INT4OID);
		if (con->constisnull)
			return 0;
		return Max(DatumGetInt32(con->constvalue), 0);
	}
	return vl_width[0];
}

static int
vlbuf_estimate_jsonb(codegen_context *context,
					 devfunc_info *dfunc,
					 Expr **args, int *vl_width)
{
	context->extra_bufsz += MAXALIGN(TOAST_TUPLE_THRESHOLD);
	/*
	 * We usually have no information about jsonb object length preliminary,
	 * however, plain varlena must be less than the threshold of toasting.
	 * If user altered storage option of jsonb column to 'main', it may be
	 * increased to BLCKSZ, but unusual.
	 */
	return TOAST_TUPLE_THRESHOLD;
}

static int
vlbuf_estimate__st_makepoint(codegen_context *context,
							 devfunc_info *dfunc,
							 Expr **args, int *vl_width)
{
	int		nargs = list_length(dfunc->func_args);

	context->extra_bufsz += MAXALIGN(sizeof(double) * 2 * nargs);

	return -1;
}

static int
vlbuf_estimate__st_relate(codegen_context *context,
						  devfunc_info *dfunc,
						  Expr **args, int *vl_width)
{
	context->extra_bufsz += MAXALIGN(VARHDRSZ + 9);

	return VARHDRSZ + 9;
}

static int
vlbuf_estimate__st_expand(codegen_context *context,
						  devfunc_info *dfunc,
						  Expr **args, int *vl_width)
{
	context->extra_bufsz += MAXALIGN(4 * sizeof(cl_float) +	/* bounding-box */
									   2 * sizeof(cl_uint) +	/* nitems + padding */
									   10 * sizeof(double));	/* polygon rawdata */
	return -1;		/* not a normal varlena */
}

/*
 * Catalog of functions supported by device code
 *
 * naming convension of functions:
 *   pgfn_<func_name>(...)
 *
 * func_template is a set of characters based on the rules below:
 *
 * [<attributes>/]f:<extra>
 *
 * attributes:
 * 'L' : this function is locale aware, thus, available only if simple
 *       collation configuration (none, and C-locale).
 * 'C' : this function uses its special callback to estimate the result
 *       width of varlena-buffer.
 * 'p' : this function needs cuda_primitive.h
 * 's' : this function needs cuda_textlib.h
 * 't' : this function needs cuda_timelib.h
 * 'j' : this function needs cuda_jsonlib.h
 * 'm' : this function needs cuda_misclib.h
 * 'r' : this function needs cuda_rangetype.h
 * 'g' : this function needs cuda_postgis.h
 *
 * class character:
 * 'r' : right operator that takes an argument (deprecated)
 * 'l' : left operator that takes an argument (deprecated)
 * 'b' : both operator that takes two arguments (deprecated)
 * 'f' : this function is implemented as device function.
 *     ==> extra is the function name being declared somewhere
 */
#define DEVFUNC_MAX_NARGS	4

typedef struct devfunc_catalog_t {
	const char *func_extension;	/* NULL, if built-in functions */
	const char *func_signature;
	int			func_devcost;	/* relative cost to run on device */
	const char *func_template;	/* a template string if simple function */
	devfunc_result_sz_type devfunc_result_sz;
} devfunc_catalog_t;

static devfunc_catalog_t devfunc_common_catalog[] = {
	/* Type cast functions */
	{ NULL,    "bool bool(int4)",    1, "f:to_bool" },

	{ PGSTROM, "int1 int1(int2)",    1, "f:to_int1" },
	{ PGSTROM, "int1 int1(int4)",    1, "f:to_int1" },
	{ PGSTROM, "int1 int1(int8)",    1, "f:to_int1" },
	{ PGSTROM, "int1 int1(float2)",  1, "f:to_int1" },
	{ PGSTROM, "int1 int1(float4)",  1, "f:to_int1" },
	{ PGSTROM, "int1 int1(float8)",  1, "f:to_int1" },
	
	{ PGSTROM, "int2 int2(int1)",    1, "f:to_int2" },
	{ NULL,    "int2 int2(int4)",    1, "f:to_int2" },
	{ NULL,    "int2 int2(int8)",    1, "f:to_int2" },
	{ PGSTROM, "int2 int2(float2)",  1, "f:to_int2" },
	{ NULL,    "int2 int2(float4)",  1, "f:to_int2" },
	{ NULL,    "int2 int2(float8)",  1, "f:to_int2" },

	{ NULL,	   "int4 int4(bool)",    1, "f:to_int4" },
	{ PGSTROM, "int4 int4(int1)",    1, "f:to_int4" },
	{ NULL,    "int4 int4(int2)",    1, "f:to_int4" },
	{ NULL,    "int4 int4(int8)",    1, "f:to_int4" },
	{ PGSTROM, "int4 int4(float2)",  1, "f:to_int4" },
	{ NULL,    "int4 int4(float4)",  1, "f:to_int4" },
	{ NULL,    "int4 int4(float8)",  1, "f:to_int4" },

	{ PGSTROM, "int8 int8(int1)",    1, "f:to_int8" },
	{ NULL,    "int8 int8(int2)",    1, "f:to_int8" },
	{ NULL,    "int8 int8(int4)",    1, "f:to_int8" },
	{ PGSTROM, "int8 int8(float2)",  1, "f:to_int8" },
	{ NULL,    "int8 int8(float4)",  1, "f:to_int8" },
	{ NULL,    "int8 int8(float8)",  1, "f:to_int8" },

	{ PGSTROM, "float2 float2(int1)",   1, "f:to_float2" },
	{ PGSTROM, "float2 float2(int2)",   1, "f:to_float2" },
	{ PGSTROM, "float2 float2(int4)",   1, "f:to_float2" },
	{ PGSTROM, "float2 float2(int8)",   1, "f:to_float2" },
	{ PGSTROM, "float2 float2(float4)", 1, "f:to_float2" },
	{ PGSTROM, "float2 float2(float8)", 1, "f:to_float2" },
	
	{ PGSTROM, "float4 float4(int1)",   1, "f:to_float4" },
	{ NULL,    "float4 float4(int2)",   1, "f:to_float4" },
	{ NULL,    "float4 float4(int4)",   1, "f:to_float4" },
	{ NULL,    "float4 float4(int8)",   1, "f:to_float4" },
	{ PGSTROM, "float4 float4(float2)", 1, "f:to_float4" },
	{ NULL,    "float4 float4(float8)", 1, "f:to_float4" },

	{ PGSTROM, "float8 float8(int1)",   1, "f:to_float8" },
	{ NULL,    "float8 float8(int2)",   1, "f:to_float8" },
	{ NULL,    "float8 float8(int4)",   1, "f:to_float8" },
	{ NULL,    "float8 float8(int8)",   1, "f:to_float8" },
	{ PGSTROM, "float8 float8(float2)", 1, "f:to_float8" },
	{ NULL,    "float8 float8(float4)", 1, "f:to_float8" },

	/* '+' : add operators */
	{ PGSTROM, "int1 int1pl(int1,int1)",  1, "p/f:int1pl" },
	{ PGSTROM, "int2 int12pl(int1,int2)", 1, "p/f:int12pl" },
	{ PGSTROM, "int4 int14pl(int1,int4)", 1, "p/f:int14pl" },
	{ PGSTROM, "int8 int18pl(int1,int8)", 1, "p/f:int18pl" },
	{ PGSTROM, "int2 int21pl(int2,int1)", 1, "p/f:int21pl" },
	{ NULL,    "int2 int2pl(int2,int2)",  1, "p/f:int2pl" },
	{ NULL,    "int4 int24pl(int2,int4)", 1, "p/f:int24pl" },
	{ NULL,    "int8 int28pl(int2,int8)", 1, "p/f:int28pl" },
	{ PGSTROM, "int4 int41pl(int4,int1)", 1, "p/f:int41pl" },
	{ NULL,    "int4 int42pl(int4,int2)", 1, "p/f:int42pl" },
	{ NULL,    "int4 int4pl(int4,int4)",  1, "p/f:int4pl" },
	{ NULL,    "int8 int48pl(int4,int8)", 1, "p/f:int48pl" },
	{ PGSTROM, "int8 int81pl(int8,int1)", 1, "p/f:int81pl" },
	{ NULL,    "int8 int82pl(int8,int2)", 1, "p/f:int82pl" },
	{ NULL,    "int8 int84pl(int8,int4)", 1, "p/f:int84pl" },
	{ NULL,    "int8 int8pl(int8,int8)",  1, "p/f:int8pl" },
	{ PGSTROM, "float4 float2_pl(float2,float2)",  1, "p/f:float2pl" },
	{ PGSTROM, "float4 float24_pl(float2,float4)", 1, "p/f:float24pl" },
	{ PGSTROM, "float8 float28_pl(float2,float8)", 1, "p/f:float28pl" },
	{ PGSTROM, "float4 float42_pl(float4,float2)", 1, "p/f:float42pl" },
	{ NULL,    "float4 float4pl(float4,float4)",   1, "p/f:float4pl" },
	{ NULL,    "float8 float48pl(float4,float8)",  1, "p/f:float48pl" },
	{ PGSTROM, "float8 float82_pl(float8,float2)", 1, "p/f:float82pl" },
	{ NULL,    "float8 float84pl(float8,float4)",  1, "p/f:float84pl" },
	{ NULL,    "float8 float8pl(float8,float8)",   1, "p/f:float8pl" },

	/* '-' : subtract operators */
	{ PGSTROM, "int1 int1mi(int1,int1)",  1, "p/f:int1mi" },
	{ PGSTROM, "int2 int12mi(int1,int2)", 1, "p/f:int12mi" },
	{ PGSTROM, "int4 int14mi(int1,int4)", 1, "p/f:int14mi" },
	{ PGSTROM, "int8 int18mi(int1,int8)", 1, "p/f:int18mi" },
	{ PGSTROM, "int2 int21mi(int2,int1)", 1, "p/f:int21mi" },
	{ NULL,    "int2 int2mi(int2,int2)",  1, "p/f:int2mi" },
	{ NULL,    "int4 int24mi(int2,int4)", 1, "p/f:int24mi" },
	{ NULL,    "int8 int28mi(int2,int8)", 1, "p/f:int28mi" },
	{ PGSTROM, "int4 int41mi(int4,int1)", 1, "p/f:int41mi" },
	{ NULL,    "int4 int42mi(int4,int2)", 1, "p/f:int42mi" },
	{ NULL,    "int4 int4mi(int4,int4)",  1, "p/f:int4mi" },
	{ NULL,    "int8 int48mi(int4,int8)", 1, "p/f:int48mi" },
	{ PGSTROM, "int8 int81mi(int8,int1)", 1, "p/f:int81mi" },
	{ NULL,    "int8 int82mi(int8,int2)", 1, "p/f:int82mi" },
	{ NULL,    "int8 int84mi(int8,int4)", 1, "p/f:int84mi" },
	{ NULL,    "int8 int8mi(int8,int8)",  1, "p/f:int8mi" },
	{ PGSTROM, "float4 float2_mi(float2,float2)",  1, "p/f:float2mi" },
	{ PGSTROM, "float4 float24_mi(float2,float4)", 1, "p/f:float24mi" },
	{ PGSTROM, "float8 float28_mi(float2,float8)", 1, "p/f:float28mi" },
	{ PGSTROM, "float4 float42_mi(float4,float2)", 1, "p/f:float42mi" },
	{ NULL,    "float4 float4mi(float4,float4)",   1, "p/f:float4mi" },
	{ NULL,    "float8 float48mi(float4,float8)",  1, "p/f:float48mi" },
	{ PGSTROM, "float8 float82_mi(float8,float2)", 1, "p/f:float82mi" },
	{ NULL,    "float8 float84mi(float8,float4)",  1, "p/f:float84mi" },
	{ NULL,    "float8 float8mi(float8,float8)",   1, "p/f:float8mi" },

	/* '*' : mutiply operators */
	{ PGSTROM, "int1 int1mul(int1,int1)",  2, "p/f:int1mul" },
	{ PGSTROM, "int2 int12mul(int1,int2)", 2, "p/f:int12mul" },
	{ PGSTROM, "int4 int14mul(int1,int4)", 2, "p/f:int14mul" },
	{ PGSTROM, "int8 int18mul(int1,int8)", 2, "p/f:int18mul" },
	{ PGSTROM, "int2 int21mul(int2,int1)", 2, "p/f:int21mul" },
	{ NULL,    "int2 int2mul(int2,int2)",  2, "p/f:int2mul" },
	{ NULL,    "int4 int24mul(int2,int4)", 2, "p/f:int24mul" },
	{ NULL,    "int8 int28mul(int2,int8)", 2, "p/f:int28mul" },
	{ PGSTROM, "int4 int41mul(int4,int1)", 2, "p/f:int41mul" },
	{ NULL,    "int4 int42mul(int4,int2)", 2, "p/f:int42mul" },
	{ NULL,    "int4 int4mul(int4,int4)",  2, "p/f:int4mul" },
	{ NULL,    "int8 int48mul(int4,int8)", 2, "p/f:int48mul" },
	{ PGSTROM, "int8 int81mul(int8,int1)", 2, "p/f:int81mul" },
	{ NULL,    "int8 int82mul(int8,int2)", 2, "p/f:int82mul" },
	{ NULL,    "int8 int84mul(int8,int4)", 2, "p/f:int84mul" },
	{ NULL,    "int8 int8mul(int8,int8)",  2, "p/f:int8mul" },
	{ PGSTROM, "float4 float2_mul(float2,float2)",  2, "p/f:float2mul" },
	{ PGSTROM, "float4 float24_mul(float2,float4)", 2, "p/f:float24mul" },
	{ PGSTROM, "float8 float28_mul(float2,float8)", 2, "p/f:float28mul" },
	{ PGSTROM, "float4 float42_mul(float4,float2)", 2, "p/f:float42mul" },
	{ NULL,    "float4 float4mul(float4,float4)",   2, "p/f:float4mul" },
	{ NULL,    "float8 float48mul(float4,float8)",  2, "p/f:float48mul" },
	{ PGSTROM, "float8 float82_mul(float8,float2)", 2, "p/f:float82mul" },
	{ NULL,    "float8 float84mul(float8,float4)",  2, "p/f:float84mul" },
	{ NULL,    "float8 float8mul(float8,float8)",   2, "p/f:float8mul" },

	/* '/' : divide operators */
	{ PGSTROM, "int1 int1div(int1,int1)",  2, "p/f:int1div" },
	{ PGSTROM, "int2 int12div(int1,int2)", 2, "p/f:int12div" },
	{ PGSTROM, "int4 int14div(int1,int4)", 2, "p/f:int14div" },
	{ PGSTROM, "int8 int18div(int1,int8)", 2, "p/f:int18div" },
	{ PGSTROM, "int2 int21div(int2,int1)", 2, "p/f:int21div" },
	{ NULL,    "int2 int2div(int2,int2)",  2, "p/f:int2div" },
	{ NULL,    "int4 int24div(int2,int4)", 2, "p/f:int24div" },
	{ NULL,    "int8 int28div(int2,int8)", 2, "p/f:int28div" },
	{ PGSTROM, "int4 int41div(int4,int1)", 2, "p/f:int41div" },
	{ NULL,    "int4 int42div(int4,int2)", 2, "p/f:int42div" },
	{ NULL,    "int4 int4div(int4,int4)",  2, "p/f:int4div" },
	{ NULL,    "int8 int48div(int4,int8)", 2, "p/f:int48div" },
	{ PGSTROM, "int2 int81div(int8,int1)", 2, "p/f:int81div" },
	{ NULL,    "int8 int82div(int8,int2)", 2, "p/f:int82div" },
	{ NULL,    "int8 int84div(int8,int4)", 2, "p/f:int84div" },
	{ NULL,    "int8 int8div(int8,int8)",  2, "p/f:int8div" },
	{ PGSTROM, "float4 float2_div(float2,float2)",  2, "p/f:float2div" },
	{ PGSTROM, "float4 float24_div(float2,float4)", 2, "p/f:float24div" },
	{ PGSTROM, "float8 float28_div(float2,float8)", 2, "p/f:float28div" },
	{ PGSTROM, "float4 float42_div(float4,float2)", 2, "p/f:float42div" },
	{ NULL,    "float4 float4div(float4,float4)",   2, "p/f:float4div" },
	{ NULL,    "float8 float48div(float4,float8)",  2, "p/f:float48div" },
	{ PGSTROM, "float8 float82_div(float8,float2)", 2, "p/f:float82div" },
	{ NULL,    "float8 float84div(float8,float4)",  2, "p/f:float84div" },
	{ NULL,    "float8 float8div(float8,float8)",   2, "p/f:float8div" },

	/* '%' : reminder operators */
	{ PGSTROM,	"int1 int1mod(int1,int1)", 2, "p/f:int1mod" },
	{ NULL,		"int2 int2mod(int2,int2)", 2, "p/f:int2mod" },
	{ NULL,		"int4 int4mod(int4,int4)", 2, "p/f:int4mod" },
	{ NULL,		"int8 int8mod(int8,int8)", 2, "p/f:int8mod" },

	/* '+' : unary plus operators */
	{ PGSTROM, "int1 int1up(int1)",       1, "p/f:int1up" },
	{ NULL,    "int2 int2up(int2)",       1, "p/f:int2up" },
	{ NULL,    "int4 int4up(int4)",       1, "p/f:int4up" },
	{ NULL,    "int8 int8up(int8)",       1, "p/f:int8up" },
	{ PGSTROM, "float2 float2_up(float2)",1, "p/f:float2up" },
	{ NULL,    "float4 float4up(float4)", 1, "p/f:float4up" },
	{ NULL,    "float8 float8up(float8)", 1, "p/f:float8up" },

	/* '-' : unary minus operators */
	{ PGSTROM, "int1 int1um(int1)",       1, "p/f:int1um" },
	{ NULL,    "int2 int2um(int2)",       1, "p/f:int2um" },
	{ NULL,    "int4 int4um(int4)",       1, "p/f:int4um" },
	{ NULL,    "int8 int8um(int8)",       1, "p/f:int8um" },
	{ PGSTROM, "float2 float2_um(float2)",1, "p/f:float2um" },
	{ NULL,    "float4 float4um(float4)", 1, "p/f:float4um" },
	{ NULL,    "float8 float8um(float8)", 1, "p/f:float8um" },
	
	/* '@' : absolute value operators */
	{ PGSTROM, "int1 int1abs(int1)", 1, "p/f:int1abs" },
	{ NULL,    "int2 int2abs(int2)", 1, "p/f:int2abs" },
	{ NULL,    "int4 int4abs(int4)", 1, "p/f:int4abs" },
	{ NULL,    "int8 int8abs(int8)", 1, "p/f:int8abs" },
	{ PGSTROM, "float2 float2abs(float2)", 1, "p/f:float2abs" },
	{ NULL,    "float4 float4abs(float4)", 1, "p/f:float4abs" },
	{ NULL,    "float8 float8abs(float8)", 1, "p/f:float8abs" },

	/* '=' : equal operators */
	{ NULL,    "bool booleq(bool,bool)",  1, "f:booleq" },
	{ PGSTROM, "bool int1eq(int1,int1)",  1, "f:int1eq" },
	{ PGSTROM, "bool int12eq(int1,int2)", 1, "f:int12eq" },
	{ PGSTROM, "bool int14eq(int1,int4)", 1, "f:int14eq" },
	{ PGSTROM, "bool int18eq(int1,int8)", 1, "f:int18eq" },
	{ PGSTROM, "bool int21eq(int2,int1)", 1, "f:int21eq" },
	{ NULL,    "bool int2eq(int2,int2)",  1, "f:int2eq" },
	{ NULL,    "bool int24eq(int2,int4)", 1, "f:int24eq" },
	{ NULL,    "bool int28eq(int2,int8)", 1, "f:int28eq" },
	{ PGSTROM, "bool int41eq(int4,int1)", 1, "f:int41eq" },
	{ NULL,    "bool int42eq(int4,int2)", 1, "f:int42eq" },
	{ NULL,    "bool int4eq(int4,int4)",  1, "f:int4eq" },
	{ NULL,    "bool int48eq(int4,int8)", 1, "f:int48eq" },
	{ PGSTROM, "bool int81eq(int8,int1)", 1, "f:int81eq" },
	{ NULL,    "bool int82eq(int8,int2)", 1, "f:int82eq" },
	{ NULL,    "bool int84eq(int8,int4)", 1, "f:int84eq" },
	{ NULL,    "bool int8eq(int8,int8)",  1, "f:int8eq" },
	{ PGSTROM, "bool float2_eq(float2,float2)",  1, "f:float2eq" },
	{ PGSTROM, "bool float24_eq(float2,float4)", 1, "f:float24eq" },
	{ PGSTROM, "bool float28_eq(float2,float8)", 1, "f:float28eq" },
	{ PGSTROM, "bool float42_eq(float4,float2)", 1, "f:float42eq" },
	{ NULL,    "bool float4eq(float4,float4)",   1, "f:float4eq" },
	{ NULL,    "bool float48eq(float4,float8)",  1, "f:float48eq" },
	{ PGSTROM, "bool float82_eq(float8,float2)", 1, "f:float82eq" },
	{ NULL,    "bool float84eq(float8,float4)",  1, "f:float84eq" },
	{ NULL,    "bool float8eq(float8,float8)",   1, "f:float8eq" },

	/* '<>' : not equal operators */
	{ PGSTROM, "bool int1ne(int1,int1)",  1, "f:int1ne" },
	{ PGSTROM, "bool int12ne(int1,int2)", 1, "f:int12ne" },
	{ PGSTROM, "bool int14ne(int1,int4)", 1, "f:int14ne" },
	{ PGSTROM, "bool int18ne(int1,int8)", 1, "f:int18ne" },
	{ PGSTROM, "bool int21ne(int2,int1)", 1, "f:int21ne" },
	{ NULL,    "bool int2ne(int2,int2)",  1, "f:int2ne" },
	{ NULL,    "bool int24ne(int2,int4)", 1, "f:int24ne" },
	{ NULL,    "bool int28ne(int2,int8)", 1, "f:int28ne" },
	{ PGSTROM, "bool int41ne(int4,int1)", 1, "f:int41ne" },
	{ NULL,    "bool int42ne(int4,int2)", 1, "f:int42ne" },
	{ NULL,    "bool int4ne(int4,int4)",  1, "f:int4ne" },
	{ NULL,    "bool int48ne(int4,int8)", 1, "f:int48ne" },
	{ PGSTROM, "bool int81ne(int8,int1)", 1, "f:int81ne" },
	{ NULL,    "bool int82ne(int8,int2)", 1, "f:int82ne" },
	{ NULL,    "bool int84ne(int8,int4)", 1, "f:int84ne" },
	{ NULL,    "bool int8ne(int8,int8)",  1, "f:int8ne" },
	{ PGSTROM, "bool float2_ne(float2,float2)",  1, "f:float2ne" },
	{ PGSTROM, "bool float24_ne(float2,float4)", 1, "f:float24ne" },
	{ PGSTROM, "bool float28_ne(float2,float8)", 1, "f:float28ne" },
	{ PGSTROM, "bool float42_ne(float4,float2)", 1, "f:float42ne" },
	{ NULL,    "bool float4ne(float4,float4)",   1, "f:float4ne" },
	{ NULL,    "bool float48ne(float4,float8)",  1, "f:float48ne" },
	{ PGSTROM, "bool float82_ne(float8,float2)",  1, "f:float82ne" },
	{ NULL,    "bool float84ne(float8,float4)",  1, "f:float84ne" },
	{ NULL,    "bool float8ne(float8,float8)",   1, "f:float8ne" },

	/* '>' : greater than operators */
	{ PGSTROM, "bool int1gt(int1,int1)",  1, "f:int1gt" },
	{ PGSTROM, "bool int12gt(int1,int2)", 1, "f:int12gt" },
	{ PGSTROM, "bool int14gt(int1,int4)", 1, "f:int14gt" },
	{ PGSTROM, "bool int18gt(int1,int8)", 1, "f:int18gt" },
	{ PGSTROM, "bool int21gt(int2,int1)", 1, "f:int21gt" },
	{ NULL,    "bool int2gt(int2,int2)",  1, "f:int2gt" },
	{ NULL,    "bool int24gt(int2,int4)", 1, "f:int24gt" },
	{ NULL,    "bool int28gt(int2,int8)", 1, "f:int28gt" },
	{ PGSTROM, "bool int41gt(int4,int1)", 1, "f:int41gt" },
	{ NULL,    "bool int42gt(int4,int2)", 1, "f:int42gt" },
	{ NULL,    "bool int4gt(int4,int4)",  1, "f:int4gt" },
	{ NULL,    "bool int48gt(int4,int8)", 1, "f:int48gt" },
	{ PGSTROM, "bool int81gt(int8,int1)", 1, "f:int81gt" },
	{ NULL,    "bool int82gt(int8,int2)", 1, "f:int82gt" },
	{ NULL,    "bool int84gt(int8,int4)", 1, "f:int84gt" },
	{ NULL,    "bool int8gt(int8,int8)",  1, "f:int8gt" },
	{ PGSTROM, "bool float2_gt(float2,float2)",  1, "f:float2gt" },
	{ PGSTROM, "bool float24_gt(float2,float4)", 1, "f:float24gt" },
	{ PGSTROM, "bool float28_gt(float2,float8)", 1, "f:float28gt" },
	{ PGSTROM, "bool float42_gt(float4,float2)", 1, "f:float42gt" },
	{ NULL,    "bool float4gt(float4,float4)",   1, "f:float4gt" },
	{ NULL,    "bool float48gt(float4,float8)",  1, "f:float48gt" },
	{ PGSTROM, "bool float82_gt(float8,float2)", 1, "f:float82gt" },
	{ NULL,    "bool float84gt(float8,float4)",  1, "f:float84gt" },
	{ NULL,    "bool float8gt(float8,float8)",   1, "f:float8gt" },

	/* '<' : less than operators */
	{ PGSTROM, "bool int1lt(int1,int1)",  1, "f:int1lt" },
	{ PGSTROM, "bool int12lt(int1,int2)", 1, "f:int12lt" },
	{ PGSTROM, "bool int14lt(int1,int4)", 1, "f:int14lt" },
	{ PGSTROM, "bool int18lt(int1,int8)", 1, "f:int18lt" },
	{ PGSTROM, "bool int21lt(int2,int1)", 1, "f:int21lt" },
	{ NULL,    "bool int2lt(int2,int2)",  1, "f:int2lt" },
	{ NULL,    "bool int24lt(int2,int4)", 1, "f:int24lt" },
	{ NULL,    "bool int28lt(int2,int8)", 1, "f:int28lt" },
	{ PGSTROM, "bool int41lt(int4,int1)", 1, "f:int41lt" },
	{ NULL,    "bool int42lt(int4,int2)", 1, "f:int42lt" },
	{ NULL,    "bool int4lt(int4,int4)",  1, "f:int4lt" },
	{ NULL,    "bool int48lt(int4,int8)", 1, "f:int48lt" },
	{ PGSTROM, "bool int81lt(int8,int1)", 1, "f:int81lt" },
	{ NULL,    "bool int82lt(int8,int2)", 1, "f:int82lt" },
	{ NULL,    "bool int84lt(int8,int4)", 1, "f:int84lt" },
	{ NULL,    "bool int8lt(int8,int8)",  1, "f:int8lt" },
	{ PGSTROM, "bool float2_lt(float2,float2)",  1, "f:float2lt" },
	{ PGSTROM, "bool float24_lt(float2,float4)", 1, "f:float24lt" },
	{ PGSTROM, "bool float28_lt(float2,float8)", 1, "f:float28lt" },
	{ PGSTROM, "bool float42_lt(float4,float2)", 1, "f:float42lt" },
	{ NULL,    "bool float4lt(float4,float4)",   1, "f:float4lt" },
	{ NULL,    "bool float48lt(float4,float8)",  1, "f:float48lt" },
	{ PGSTROM, "bool float82_lt(float8,float2)", 1, "f:float82lt" },
	{ NULL,    "bool float84lt(float8,float4)",  1, "f:float84lt" },
	{ NULL,    "bool float8lt(float8,float8)",   1, "f:float8lt" },

	/* '>=' : relational greater-than or equal-to */
	{ PGSTROM, "bool int1ge(int1,int1)",  1, "f:int1ge" },
	{ PGSTROM, "bool int12ge(int1,int2)", 1, "f:int12ge" },
	{ PGSTROM, "bool int14ge(int1,int4)", 1, "f:int14ge" },
	{ PGSTROM, "bool int18ge(int1,int8)", 1, "f:int18ge" },
	{ PGSTROM, "bool int21ge(int2,int1)", 1, "f:int21ge" },
	{ NULL,    "bool int2ge(int2,int2)",  1, "f:int2ge" },
	{ NULL,    "bool int24ge(int2,int4)", 1, "f:int24ge" },
	{ NULL,    "bool int28ge(int2,int8)", 1, "f:int28ge" },
	{ PGSTROM, "bool int41ge(int4,int1)", 1, "f:int41ge" },
	{ NULL,    "bool int42ge(int4,int2)", 1, "f:int42ge" },
	{ NULL,    "bool int4ge(int4,int4)",  1, "f:int4ge" },
	{ NULL,    "bool int48ge(int4,int8)", 1, "f:int48ge" },
	{ PGSTROM, "bool int81ge(int8,int1)", 1, "f:int81ge" },
	{ NULL,    "bool int82ge(int8,int2)", 1, "f:int82ge" },
	{ NULL,    "bool int84ge(int8,int4)", 1, "f:int84ge" },
	{ NULL,    "bool int8ge(int8,int8)",  1, "f:int8ge" },
	{ PGSTROM, "bool float2_ge(float2,float2)",  1, "f:float2ge" },
	{ PGSTROM, "bool float24_ge(float2,float4)", 1, "f:float24ge" },
	{ PGSTROM, "bool float28_ge(float2,float8)", 1, "f:float28ge" },
	{ PGSTROM, "bool float42_ge(float4,float2)", 1, "f:float42ge" },
	{ NULL,    "bool float4ge(float4,float4)",   1, "f:float4ge" },
	{ NULL,    "bool float48ge(float4,float8)",  1, "f:float48ge" },
	{ PGSTROM, "bool float82_ge(float8,float2)", 1, "f:float82ge" },
	{ NULL,    "bool float84ge(float8,float4)",  1, "f:float84ge" },
	{ NULL,    "bool float8ge(float8,float8)",   1, "f:float8ge" },

	/* '<=' : relational greater-than or equal-to */
	{ PGSTROM, "bool int1le(int1,int1)",  1, "f:int1le" },
	{ PGSTROM, "bool int12le(int1,int2)", 1, "f:int12le" },
	{ PGSTROM, "bool int14le(int1,int4)", 1, "f:int14le" },
	{ PGSTROM, "bool int18le(int1,int8)", 1, "f:int18le" },
	{ PGSTROM, "bool int21le(int2,int1)", 1, "f:int21le" },
	{ NULL,    "bool int2le(int2,int2)",  1, "f:int2le" },
	{ NULL,    "bool int24le(int2,int4)", 1, "f:int24le" },
	{ NULL,    "bool int28le(int2,int8)", 1, "f:int28le" },
	{ PGSTROM, "bool int41le(int4,int1)", 1, "f:int41le" },
	{ NULL,    "bool int42le(int4,int2)", 1, "f:int42le" },
	{ NULL,    "bool int4le(int4,int4)",  1, "f:int4le" },
	{ NULL,    "bool int48le(int4,int8)", 1, "f:int48le" },
	{ PGSTROM, "bool int81le(int8,int1)", 1, "f:int81le" },
	{ NULL,    "bool int82le(int8,int2)", 1, "f:int82le" },
	{ NULL,    "bool int84le(int8,int4)", 1, "f:int84le" },
	{ NULL,    "bool int8le(int8,int8)",  1, "f:int8le" },
	{ PGSTROM, "bool float2_le(float2,float2)", 1, "f:float2le" },
	{ PGSTROM, "bool float24_le(float2,float4)", 1, "f:float24le" },
	{ PGSTROM, "bool float28_le(float2,float8)", 1, "f:float28le" },
	{ PGSTROM, "bool float42_le(float4,float2)", 2, "f:float42le" },
	{ NULL,    "bool float4le(float4,float4)",  1, "f:float4le" },
	{ NULL,    "bool float48le(float4,float8)", 1, "f:float48le" },
	{ PGSTROM, "bool float82_le(float8,float2)", 1, "f:float82le" },
	{ NULL,    "bool float84le(float8,float4)", 1, "f:float84le" },
	{ NULL,    "bool float8le(float8,float8)",  1, "f:float8le" },

	/* '&' : bitwise and */
	{ PGSTROM, "int1 int1and(int1,int1)", 1, "p/f:int1and" },
	{ NULL,    "int2 int2and(int2,int2)", 1, "p/f:int2and" },
	{ NULL,    "int4 int4and(int4,int4)", 1, "p/f:int4and" },
	{ NULL,    "int8 int8and(int8,int8)", 1, "p/f:int8and" },

	/* '|'  : bitwise or */
	{ PGSTROM, "int1 int1or(int1,int1)",  1, "p/f:int1or" },
	{ NULL,    "int2 int2or(int2,int2)",  1, "p/f:int2or" },
	{ NULL,    "int4 int4or(int4,int4)",  1, "p/f:int4or" },
	{ NULL,    "int8 int8or(int8,int8)",  1, "p/f:int8or" },

	/* '#'  : bitwise xor */
	{ PGSTROM, "int1 int1xor(int1,int1)", 1, "p/f:int1xor" },
	{ NULL,    "int2 int2xor(int2,int2)", 1, "p/f:int2xor" },
	{ NULL,    "int4 int4xor(int4,int4)", 1, "p/f:int4xor" },
	{ NULL,    "int8 int8xor(int8,int8)", 1, "p/f:int8xor" },

	/* '~'  : bitwise not operators */
	{ PGSTROM, "int1 int1not(int1)", 1, "p/f:int1not" },
	{ NULL,    "int2 int2not(int2)", 1, "p/f:int2not" },
	{ NULL,    "int4 int4not(int4)", 1, "p/f:int4not" },
	{ NULL,    "int8 int8not(int8)", 1, "p/f:int8not" },

	/* '>>' : right shift */
	{ PGSTROM, "int1 int1shr(int1,int4)", 1, "p/f:int1shr" },
	{ NULL,    "int2 int2shr(int2,int4)", 1, "p/f:int2shr" },
	{ NULL,    "int4 int4shr(int4,int4)", 1, "p/f:int4shr" },
	{ NULL,    "int8 int8shr(int8,int4)", 1, "p/f:int8shr" },

	/* '<<' : left shift */
	{ PGSTROM, "int1 int1shl(int1,int4)", 1, "p/f:int1shl" },
	{ NULL,    "int2 int2shl(int2,int4)", 1, "p/f:int2shl" },
	{ NULL,    "int4 int4shl(int4,int4)", 1, "p/f:int4shl" },
	{ NULL,    "int8 int8shl(int8,int4)", 1, "p/f:int8shl" },

	/* comparison functions */
	{ NULL,    "int4 btboolcmp(bool,bool)",  1, "p/f:type_compare" },
	{ PGSTROM, "int4 btint1cmp(int1,int1)",  1, "p/f:type_compare" },
	{ PGSTROM, "int4 btint12cmp(int1,int2)", 1, "p/f:type_compare" },
	{ PGSTROM, "int4 btint14cmp(int1,int4)", 1, "p/f:type_compare" },
	{ PGSTROM, "int4 btint18cmp(int1,int8)", 1, "p/f:type_compare" },
	{ PGSTROM, "int4 btint21cmp(int2,int1)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btint2cmp(int2,int2)",  1, "p/f:type_compare" },
	{ NULL,    "int4 btint24cmp(int2,int4)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btint28cmp(int2,int8)", 1, "p/f:type_compare" },
	{ PGSTROM, "int4 btint41cmp(int4,int1)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btint42cmp(int4,int2)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btint4cmp(int4,int4)",  1, "p/f:type_compare" },
	{ NULL,    "int4 btint48cmp(int4,int8)", 1, "p/f:type_compare" },
	{ PGSTROM, "int4 btint81cmp(int8,int1)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btint82cmp(int8,int2)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btint84cmp(int8,int4)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btint8cmp(int8,int8)",  1, "p/f:type_compare" },
	{ PGSTROM, "int4 float2_cmp(float2,float2)",   1, "f:type_compare" },
	{ PGSTROM, "int4 float24_cmp(float2,float4)",  1, "f:type_compare" },
	{ PGSTROM, "int4 float28_cmp(float2,float8)",  1, "f:type_compare" },
	{ PGSTROM, "int4 float42_cmp(float4,float2)",  1, "f:type_compare" },
	{ NULL,    "int4 btfloat4cmp(float4,float4)",  1, "p/f:type_compare" },
	{ NULL,    "int4 btfloat48cmp(float4,float8)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btfloat84cmp(float8,float4)", 1, "p/f:type_compare" },
	{ NULL,    "int4 btfloat8cmp(float8,float8)",  1, "p/f:type_compare" },
	{ PGSTROM, "int4 float82_cmp(float8,float2)",  1, "f:type_compare" },

	/* currency cast */
	{ NULL, "money money(numeric)", 1, "m/f:numeric_cash" },
	{ NULL, "money money(int4)",    1, "m/f:int4_cash" },
	{ NULL, "money money(int8)",    1, "m/f:int8_cash" },
	/* currency operators */
	{ NULL, "money cash_pl(money,money)",	        1, "m/f:cash_pl" },
	{ NULL, "money cash_mi(money,money)",           1, "m/f:cash_mi" },
	{ NULL, "float8 cash_div_cash(money,money)",    2, "m/f:cash_div_cash" },
	{ PGSTROM, "money cash_mul_int1(money,int1)",   2, "m/f:cash_mul_int1" },
	{ NULL,    "money cash_mul_int2(money,int2)",   2, "m/f:cash_mul_int2" },
	{ NULL,    "money cash_mul_int4(money,int4)",   2, "m/f:cash_mul_int4" },
	{ PGSTROM, "money cash_mul_flt2(money,float2)", 2, "m/f:cash_mul_flt2" },
	{ NULL,    "money cash_mul_flt4(money,float4)", 2, "m/f:cash_mul_flt4" },
	{ NULL,    "money cash_mul_flt8(money,float8)", 2, "m/f:cash_mul_flt8" },
	{ PGSTROM, "money cash_div_int1(money,int1)",   2, "m/f:cash_div_int1" },
	{ NULL,    "money cash_div_int2(money,int2)",   2, "m/f:cash_div_int2" },
	{ NULL,    "money cash_div_int4(money,int4)",   2, "m/f:cash_div_int4" },
	{ PGSTROM, "money cash_div_flt2(money,float2)", 2, "m/f:cash_div_flt2" },
	{ NULL,    "money cash_div_flt4(money,float4)", 2, "m/f:cash_div_flt4" },
	{ NULL,    "money cash_div_flt8(money,float8)", 2, "m/f:cash_div_flt8" },
	{ PGSTROM, "money int1_mul_cash(int1,money)",   2, "m/f:int1_mul_cash" },
	{ NULL,    "money int2_mul_cash(int2,money)",   2, "m/f:int2_mul_cash" },
	{ NULL,    "money int4_mul_cash(int4,money)",   2, "m/f:int4_mul_cash" },
	{ PGSTROM, "money flt2_mul_cash(float2,money)", 2, "m/f:flt2_mul_cash" },
	{ NULL,    "money flt4_mul_cash(float4,money)", 2, "m/f:flt4_mul_cash" },
	{ NULL,    "money flt8_mul_cash(float8,money)", 2, "m/f:flt8_mul_cash" },
	/* currency comparison */
	{ NULL, "int4 cash_cmp(money,money)", 1, "m/f:type_compare" },
	{ NULL, "bool cash_eq(money,money)",  1, "m/f:cash_eq" },
	{ NULL, "bool cash_ne(money,money)",  1, "m/f:cash_ne" },
	{ NULL, "bool cash_lt(money,money)",  1, "m/f:cash_lt" },
	{ NULL, "bool cash_le(money,money)",  1, "m/f:cash_le" },
	{ NULL, "bool cash_gt(money,money)",  1, "m/f:cash_gt" },
	{ NULL, "bool cash_ge(money,money)",  1, "m/f:cash_ge" },
	/* uuid comparison */
	{ NULL, "int4 uuid_cmp(uuid,uuid)",   5, "m/f:type_compare" },
	{ NULL, "bool uuid_eq(uuid,uuid)",    5, "m/f:uuid_eq" },
	{ NULL, "bool uuid_ne(uuid,uuid)",    5, "m/f:uuid_ne" },
	{ NULL, "bool uuid_lt(uuid,uuid)",    5, "m/f:uuid_lt" },
	{ NULL, "bool uuid_le(uuid,uuid)",    5, "m/f:uuid_le" },
	{ NULL, "bool uuid_gt(uuid,uuid)",    5, "m/f:uuid_gt" },
	{ NULL, "bool uuid_ge(uuid,uuid)",    5, "m/f:uuid_ge" },
	/* macaddr comparison */
	{ NULL, "int4 macaddr_cmp(macaddr,macaddr)", 5, "m/f:type_compare" },
	{ NULL, "bool macaddr_eq(macaddr,macaddr)",  5, "m/f:macaddr_eq" },
	{ NULL, "bool macaddr_ne(macaddr,macaddr)",  5, "m/f:macaddr_ne" },
	{ NULL, "bool macaddr_lt(macaddr,macaddr)",  5, "m/f:macaddr_lt" },
	{ NULL, "bool macaddr_le(macaddr,macaddr)",  5, "m/f:macaddr_le" },
	{ NULL, "bool macaddr_gt(macaddr,macaddr)",  5, "m/f:macaddr_gt" },
	{ NULL, "bool macaddr_ge(macaddr,macaddr)",  5, "m/f:macaddr_ge" },
	/* inet comparison */
	{ NULL, "int4 network_cmp(inet,inet)",    8, "m/f:type_compare" },
	{ NULL, "bool network_eq(inet,inet)",     8, "m/f:network_eq" },
	{ NULL, "bool network_ne(inet,inet)",     8, "m/f:network_ne" },
	{ NULL, "bool network_lt(inet,inet)",     8, "m/f:network_lt" },
	{ NULL, "bool network_le(inet,inet)",     8, "m/f:network_le" },
	{ NULL, "bool network_gt(inet,inet)",     8, "m/f:network_gt" },
	{ NULL, "bool network_ge(inet,inet)",     8, "m/f:network_ge" },
	{ NULL, "inet network_larger(inet,inet)", 8, "m/f:network_larger" },
	{ NULL, "inet network_smaller(inet,inet)",8, "m/f:network_smaller" },
	{ NULL, "bool network_sub(inet,inet)",    8, "m/f:network_sub" },
	{ NULL, "bool network_subeq(inet,inet)",  8, "m/f:network_subeq" },
	{ NULL, "bool network_sup(inet,inet)",    8, "m/f:network_sup" },
	{ NULL, "bool network_supeq(inet,inet)",  8, "m/f:network_supeq" },
	{ NULL, "bool network_overlap(inet,inet)",8, "m/f:network_overlap" },

	/*
     * Mathmatical functions
     */
	{ PGSTROM, "int1 abs(int1)",         1, "p/f:int1abs" },
	{ NULL,    "int2 abs(int2)",         1, "p/f:int2abs" },
	{ NULL,    "int4 abs(int4)",         1, "p/f:int4abs" },
	{ NULL,    "int8 abs(int8)",         1, "p/f:int8abs" },
	{ PGSTROM, "float2 abs(float2)",     1, "p/f:float2abs" },
	{ NULL,    "float4 abs(float4)",     1, "p/f:float4abs" },
	{ NULL,    "float8 abs(float8)",     1, "p/f:float8abs" },
	{ NULL,    "float8 cbrt(float8)",    1, "m/f:cbrt" },
	{ NULL,    "float8 dcbrt(float8)",   1, "m/f:cbrt" },
	{ NULL,    "float8 ceil(float8)",    1, "m/f:ceil" },
	{ NULL,    "float8 ceiling(float8)", 1, "m/f:ceil" },
	{ NULL,    "float8 exp(float8)",     5, "m/f:exp" },
	{ NULL,    "float8 dexp(float8)",    5, "m/f:exp" },
	{ NULL,    "float8 floor(float8)",   1, "m/f:floor" },
	{ NULL,    "float8 ln(float8)",      5, "m/f:ln" },
	{ NULL,    "float8 dlog1(float8)",   5, "m/f:ln" },
	{ NULL,    "float8 log(float8)",     5, "m/f:log10" },
	{ NULL,    "float8 dlog10(float8)",  5, "m/f:log10" },
	{ NULL,    "float8 pi()",            0, "m/f:dpi" },
	{ NULL,    "float8 power(float8,float8)", 5, "m/f:dpow" },
	{ NULL,    "float8 pow(float8,float8)",   5, "m/f:dpow" },
	{ NULL,    "float8 dpow(float8,float8)",  5, "m/f:dpow" },
	{ NULL,    "float8 round(float8)",   5, "m/f:round" },
	{ NULL,    "float8 dround(float8)",  5, "m/f:round" },
	{ NULL,    "float8 sign(float8)",    1, "m/f:sign" },
	{ NULL,    "float8 sqrt(float8)",    5, "m/f:dsqrt" },
	{ NULL,    "float8 dsqrt(float8)",   5, "m/f:dsqrt" },
	{ NULL,    "float8 trunc(float8)",   1, "m/f:trunc" },
	{ NULL,    "float8 dtrunc(float8)",  1, "m/f:trunc" },

	/*
	 * Trigonometric function
	 */
	{ NULL, "float8 degrees(float8)", 5, "m/f:degrees" },
	{ NULL, "float8 radians(float8)", 5, "m/f:radians" },
	{ NULL, "float8 acos(float8)",    5, "m/f:acos" },
	{ NULL, "float8 asin(float8)",    5, "m/f:asin" },
	{ NULL, "float8 atan(float8)",    5, "m/f:atan" },
	{ NULL, "float8 atan2(float8,float8)", 5, "m/f:atan2" },
	{ NULL, "float8 cos(float8)",     5, "m/f:cos" },
	{ NULL, "float8 cot(float8)",     5, "m/f:cot" },
	{ NULL, "float8 sin(float8)",     5, "m/f:sin" },
	{ NULL, "float8 tan(float8)",     5, "m/f:tan" },

	/*
	 * Numeric functions
	 * ------------------------- */
	/* Numeric type cast functions */
	{ PGSTROM, "int1 int1(numeric)",         8, "f:numeric_int1" },
	{ NULL,    "int2 int2(numeric)",         8, "f:numeric_int2" },
	{ NULL,    "int4 int4(numeric)",         8, "f:numeric_int4" },
	{ NULL,    "int8 int8(numeric)",         8, "f:numeric_int8" },
	{ PGSTROM, "float2 float2(numeric)",     8, "f:numeric_float2" },
	{ NULL,    "float4 float4(numeric)",     8, "f:numeric_float4" },
	{ NULL,    "float8 float8(numeric)",     8, "f:numeric_float8" },
	{ PGSTROM, "numeric numeric(int1)",      5, "f:int1_numeric" },
	{ NULL,    "numeric numeric(int2)",      5, "f:int2_numeric" },
	{ NULL,    "numeric numeric(int4)",      5, "f:int4_numeric" },
	{ NULL,    "numeric numeric(int8)",      5, "f:int8_numeric" },
	{ PGSTROM, "numeric numeric(float2)",    5, "f:float2_numeric" },
	{ NULL,    "numeric numeric(float4)",    5, "f:float4_numeric" },
	{ NULL,    "numeric numeric(float8)",    5, "f:float8_numeric" },
	/* Numeric operators */
	{ NULL, "numeric numeric_add(numeric,numeric)", 10, "f:numeric_add" },
	{ NULL, "numeric numeric_sub(numeric,numeric)", 10, "f:numeric_sub" },
	{ NULL, "numeric numeric_mul(numeric,numeric)", 10, "f:numeric_mul" },
	{ NULL, "numeric numeric_uplus(numeric)",       10, "f:numeric_uplus" },
	{ NULL, "numeric numeric_uminus(numeric)",      10, "f:numeric_uminus" },
	{ NULL, "numeric numeric_abs(numeric)",         10, "f:numeric_abs" },
	{ NULL, "numeric abs(numeric)",                 10, "f:numeric_abs" },
	/* Numeric comparison */
	{ NULL, "bool numeric_eq(numeric,numeric)",  8, "f:numeric_eq" },
	{ NULL, "bool numeric_ne(numeric,numeric)",  8, "f:numeric_ne" },
	{ NULL, "bool numeric_lt(numeric,numeric)",  8, "f:numeric_lt" },
	{ NULL, "bool numeric_le(numeric,numeric)",  8, "f:numeric_le" },
	{ NULL, "bool numeric_gt(numeric,numeric)",  8, "f:numeric_gt" },
	{ NULL, "bool numeric_ge(numeric,numeric)",  8, "f:numeric_ge" },
	{ NULL, "int4 numeric_cmp(numeric,numeric)", 8, "f:type_compare" },

	/*
	 * Date and time functions
	 * ------------------------------- */
	/* Type cast functions */
	{ NULL, "date date(timestamp)",     1, "t/f:timestamp_date" },
	{ NULL, "date date(timestamptz)",   1, "t/f:timestamptz_date" },
	{ NULL, "time time(timetz)",        1, "t/f:timetz_time" },
	{ NULL, "time time(timestamp)",     1, "t/f:timestamp_time" },
	{ NULL, "time time(timestamptz)",   1, "t/f:timestamptz_time" },
	{ NULL, "timetz timetz(time)",      1, "t/f:time_timetz" },
	{ NULL, "timetz timetz(timestamptz)", 1, "t/f:timestamptz_timetz" },
#ifdef NOT_USED
	{ NULL, "timetz timetz(timetz,int4)", 1, "t/f:timetz_scale" },
#endif
	{ NULL, "timestamp timestamp(date)",
	  1, "t/f:date_timestamp" },
	{ NULL, "timestamp timestamp(timestamptz)",
	  1, "t/f:timestamptz_timestamp" },
	{ NULL, "timestamptz timestamptz(date)",
	  1, "t/f:date_timestamptz" },
	{ NULL, "timestamptz timestamptz(timestamp)",
	  1, "t/f:timestamp_timestamptz" },
	/* timedata operators */
	{ NULL, "date date_pli(date,int4)", 1, "t/f:date_pli" },
	{ NULL, "date date_mii(date,int4)", 1, "t/f:date_mii" },
	{ NULL, "int4 date_mi(date,date)",  1, "t/f:date_mi" },
	{ NULL, "timestamp datetime_pl(date,time)", 2, "t/f:datetime_pl" },
	{ NULL, "date integer_pl_date(int4,date)",  2, "t/f:integer_pl_date" },
	{ NULL, "timestamp timedate_pl(time,date)", 2, "t/f:timedate_pl" },
	/* time - time => interval */
	{ NULL, "interval time_mi_time(time,time)",
	  2, "t/f:time_mi_time" },
	/* timestamp - timestamp => interval */
	{ NULL, "interval timestamp_mi(timestamp,timestamp)",
	  4, "t/f:timestamp_mi" },
	/* timetz +/- interval => timetz */
	{ NULL, "timetz timetz_pl_interval(timetz,interval)",
	  4, "t/f:timetz_pl_interval" },
	{ NULL, "timetz timetz_mi_interval(timetz,interval)",
	  4, "t/f:timetz_mi_interval" },
	/* timestamptz +/- interval => timestamptz */
	{ NULL, "timestamptz timestamptz_pl_interval(timestamptz,interval)",
	  4, "t/f:timestamptz_pl_interval" },
	{ NULL, "timestamptz timestamptz_mi_interval(timestamptz,interval)",
	  4, "t/f:timestamptz_mi_interval" },
	/* interval operators */
	{ NULL, "interval interval_um(interval)",          4, "t/f:interval_um" },
	{ NULL, "interval interval_pl(interval,interval)", 4, "t/f:interval_pl" },
	{ NULL, "interval interval_mi(interval,interval)", 4, "t/f:interval_mi" },
	/* date + timetz => timestamptz */
	{ NULL, "timestamptz datetimetz_pl(date,timetz)",
	  4, "t/f:datetimetz_timestamptz" },
	{ NULL, "timestamptz timestamptz(date,timetz)",
	  4, "t/f:datetimetz_timestamptz" },
	/* comparison between date */
	{ NULL, "bool date_eq(date,date)",  2, "t/f:date_eq" },
	{ NULL, "bool date_ne(date,date)",  2, "t/f:date_ne" },
	{ NULL, "bool date_lt(date,date)",  2, "t/f:date_lt"  },
	{ NULL, "bool date_le(date,date)",  2, "t/f:date_le" },
	{ NULL, "bool date_gt(date,date)",  2, "t/f:date_gt"  },
	{ NULL, "bool date_ge(date,date)",  2, "t/f:date_ge" },
	{ NULL, "int4 date_cmp(date,date)", 2, "t/f:type_compare" },
	/* comparison of date and timestamp */
	{ NULL, "bool date_eq_timestamp(date,timestamp)",
	  2, "t/f:date_eq_timestamp" },
	{ NULL, "bool date_ne_timestamp(date,timestamp)",
	  2, "t/f:date_ne_timestamp" },
	{ NULL, "bool date_lt_timestamp(date,timestamp)",
	  2, "t/f:date_lt_timestamp" },
	{ NULL, "bool date_le_timestamp(date,timestamp)",
	  2, "t/f:date_le_timestamp" },
	{ NULL, "bool date_gt_timestamp(date,timestamp)",
	  2, "t/f:date_gt_timestamp" },
	{ NULL, "bool date_ge_timestamp(date,timestamp)",
	  2, "t/f:date_ge_timestamp" },
	{ NULL, "int4 date_cmp_timestamp(date,timestamp)",
	  2, "t/f:date_cmp_timestamp" },
	/* comparison between time */
	{ NULL, "bool time_eq(time,time)", 2, "t/f:time_eq" },
	{ NULL, "bool time_ne(time,time)", 2, "t/f:time_ne" },
	{ NULL, "bool time_lt(time,time)", 2, "t/f:time_lt"  },
	{ NULL, "bool time_le(time,time)", 2, "t/f:time_le" },
	{ NULL, "bool time_gt(time,time)", 2, "t/f:time_gt"  },
	{ NULL, "bool time_ge(time,time)", 2, "t/f:time_ge" },
	{ NULL, "int4 time_cmp(time,time)",2, "t/f:type_compare" },
	/* comparison between timetz */
	{ NULL, "bool timetz_eq(timetz,timetz)", 1, "t/f:timetz_eq" },
	{ NULL, "bool timetz_ne(timetz,timetz)", 1, "t/f:timetz_ne" },
	{ NULL, "bool timetz_lt(timetz,timetz)", 1, "t/f:timetz_lt" },
	{ NULL, "bool timetz_le(timetz,timetz)", 1, "t/f:timetz_le" },
	{ NULL, "bool timetz_ge(timetz,timetz)", 1, "t/f:timetz_ge" },
	{ NULL, "bool timetz_gt(timetz,timetz)", 1, "t/f:timetz_gt" },
	{ NULL, "int4 timetz_cmp(timetz,timetz)",1, "t/f:timetz_cmp" },
	/* comparison between timestamp */
	{ NULL, "bool timestamp_eq(timestamp,timestamp)", 1, "t/f:timestamp_eq" },
	{ NULL, "bool timestamp_ne(timestamp,timestamp)", 1, "t/f:timestamp_ne" },
	{ NULL, "bool timestamp_lt(timestamp,timestamp)", 1, "t/f:timestamp_lt" },
	{ NULL, "bool timestamp_le(timestamp,timestamp)", 1, "t/f:timestamp_le" },
	{ NULL, "bool timestamp_gt(timestamp,timestamp)", 1, "t/f:timestamp_gt" },
	{ NULL, "bool timestamp_ge(timestamp,timestamp)", 1, "t/f:timestamp_ge" },
	{ NULL, "int4 timestamp_cmp(timestamp,timestamp)",1, "t/f:timestamp_cmp"},
	/* comparison of timestamp and date */
	{ NULL, "bool timestamp_eq_date(timestamp,date)",
	  3, "t/f:timestamp_eq_date" },
	{ NULL, "bool timestamp_ne_date(timestamp,date)",
	  3, "t/f:timestamp_ne_date" },
	{ NULL, "bool timestamp_lt_date(timestamp,date)",
	  3, "t/f:timestamp_lt_date" },
	{ NULL, "bool timestamp_le_date(timestamp,date)",
	  3, "t/f:timestamp_le_date" },
	{ NULL, "bool timestamp_gt_date(timestamp,date)",
	  3, "t/f:timestamp_gt_date" },
	{ NULL, "bool timestamp_ge_date(timestamp,date)",
	  3, "t/f:timestamp_ge_date" },
	{ NULL, "int4 timestamp_cmp_date(timestamp,date)",
	  3, "t/f:timestamp_cmp_date"},
	/* comparison between timestamptz */
	{ NULL, "bool timestamptz_eq(timestamptz,timestamptz)",
	  1, "t/f:timestamptz_eq" },
	{ NULL, "bool timestamptz_ne(timestamptz,timestamptz)",
	  1, "t/f:timestamptz_ne" },
	{ NULL, "bool timestamptz_lt(timestamptz,timestamptz)",
	  1, "t/f:timestamptz_lt" },
	{ NULL, "bool timestamptz_le(timestamptz,timestamptz)",
	  1, "t/f:timestamptz_le" },
	{ NULL, "bool timestamptz_gt(timestamptz,timestamptz)",
	  1, "t/f:timestamptz_gt" },
	{ NULL, "bool timestamptz_ge(timestamptz,timestamptz)",
	  1, "t/f:timestamptz_ge" },
	{ NULL, "int4 timestamptz_cmp(timestamptz,timestamptz)",
	  1, "t/f:type_compare" },
	/* comparison between date and timestamptz */
	{ NULL, "bool date_lt_timestamptz(date,timestamptz)",
	  3, "t/f:date_lt_timestamptz" },
	{ NULL, "bool date_le_timestamptz(date,timestamptz)",
	  3, "t/f:date_le_timestamptz" },
	{ NULL, "bool date_eq_timestamptz(date,timestamptz)",
	  3, "t/f:date_eq_timestamptz" },
	{ NULL, "bool date_ge_timestamptz(date,timestamptz)",
	  3, "t/f:date_ge_timestamptz" },
	{ NULL, "bool date_gt_timestamptz(date,timestamptz)",
	  3, "t/f:date_gt_timestamptz" },
	{ NULL, "bool date_ne_timestamptz(date,timestamptz)",
	  3, "t/f:date_ne_timestamptz" },
	/* comparison between timestamptz and date */
	{ NULL, "bool timestamptz_lt_date(timestamptz,date)",
	  3, "t/f:timestamptz_lt_date" },
	{ NULL, "bool timestamptz_le_date(timestamptz,date)",
	  3, "t/f:timestamptz_le_date" },
	{ NULL, "bool timestamptz_eq_date(timestamptz,date)",
	  3, "t/f:timestamptz_eq_date" },
	{ NULL, "bool timestamptz_ge_date(timestamptz,date)",
	  3, "t/f:timestamptz_ge_date" },
	{ NULL, "bool timestamptz_gt_date(timestamptz,date)",
	  3, "t/f:timestamptz_gt_date" },
	{ NULL, "bool timestamptz_ne_date(timestamptz,date)",
	  3, "t/f:timestamptz_ne_date" },
	/* comparison between timestamp and timestamptz  */
	{ NULL, "bool timestamp_lt_timestamptz(timestamp,timestamptz)",
	  2, "t/f:timestamp_lt_timestamptz" },
	{ NULL, "bool timestamp_le_timestamptz(timestamp,timestamptz)",
	  2, "t/f:timestamp_le_timestamptz" },
	{ NULL, "bool timestamp_eq_timestamptz(timestamp,timestamptz)",
	  2, "t/f:timestamp_eq_timestamptz" },
	{ NULL, "bool timestamp_ge_timestamptz(timestamp,timestamptz)",
	  2, "t/f:timestamp_ge_timestamptz" },
	{ NULL, "bool timestamp_gt_timestamptz(timestamp,timestamptz)",
	  2, "t/f:timestamp_gt_timestamptz" },
	{ NULL, "bool timestamp_ne_timestamptz(timestamp,timestamptz)",
	  2, "t/f:timestamp_ne_timestamptz" },
	/* comparison between timestamptz and timestamp  */
	{ NULL, "bool timestamptz_lt_timestamp(timestamptz,timestamp)",
	  2, "t/f:timestamptz_lt_timestamp" },
	{ NULL, "bool timestamptz_le_timestamp(timestamptz,timestamp)",
	  2, "t/f:timestamptz_le_timestamp" },
	{ NULL, "bool timestamptz_eq_timestamp(timestamptz,timestamp)",
	  2, "t/f:timestamptz_eq_timestamp" },
	{ NULL, "bool timestamptz_ge_timestamp(timestamptz,timestamp)",
	  2, "t/f:timestamptz_ge_timestamp" },
	{ NULL, "bool timestamptz_gt_timestamp(timestamptz,timestamp)",
	  2, "t/f:timestamptz_gt_timestamp" },
	{ NULL, "bool timestamptz_ne_timestamp(timestamptz,timestamp)",
	  2, "t/f:timestamptz_ne_timestamp" },
	/* comparison between intervals */
	{ NULL, "bool interval_eq(interval,interval)", 2, "t/f:interval_eq" },
	{ NULL, "bool interval_ne(interval,interval)", 2, "t/f:interval_ne" },
	{ NULL, "bool interval_lt(interval,interval)", 2, "t/f:interval_lt" },
	{ NULL, "bool interval_le(interval,interval)", 2, "t/f:interval_le" },
	{ NULL, "bool interval_ge(interval,interval)", 2, "t/f:interval_ge" },
	{ NULL, "bool interval_gt(interval,interval)", 2, "t/f:interval_gt" },
	{ NULL, "int4 interval_cmp(interval,interval)",2, "t/f:interval_cmp"},
	/* overlaps() */
	{ NULL, "bool overlaps(time,time,time,time)",
	  20, "t/f:overlaps_time" },
	{ NULL, "bool overlaps(timetz,timetz,timetz,timetz)",
	  20, "t/f:overlaps_timetz" },
	{ NULL, "bool overlaps(timestamp,timestamp,timestamp,timestamp)",
	  20, "t/f:overlaps_timestamp" },
	{ NULL, "bool overlaps(timestamptz,timestamptz,timestamptz,timestamptz)",
	  20, "t/f:overlaps_timestamptz" },
	/* extract() - PG14 changed to return numeric, not float8 */
	{ NULL, "float8 date_part(text,timestamp)",
	  100, "t/f:date_part_timestamp"},
	{ NULL, "float8 date_part(text,timestamptz)",
	  100, "t/f:date_part_timestamptz"},
	{ NULL, "float8 date_part(text,interval)",
	  100, "t/f:date_part_interval"},
	{ NULL, "float8 date_part(text,timetz)",
	  100, "t/f:date_part_timetz"},
	{ NULL, "float8 date_part(text,time)",
	  100, "t/f:date_part_time"},

	{ NULL, "numeric extract(text,timestamp)",
	  100, "t/f:extract_timestamp"},
	{ NULL, "numeric extract(text,timestamptz)",
	  100, "t/f:extract_timestamptz"},
	{ NULL, "numeric extract(text,time)",
	  100, "t/f:extract_time"},
	{ NULL, "numeric extract(text,timetz)",
	  100, "t/f:extract_timetz"},
	{ NULL, "numeric extract(text,interval)",
	  100, "t/f:extract_interval"},

	/* other time and data functions */
	{ NULL, "timestamptz now()", 1, "t/f:now" },

	/* macaddr functions */
	{ NULL, "macaddr trunc(macaddr)",               8, "m/f:macaddr_trunc" },
	{ NULL, "macaddr macaddr_not(macaddr)",         8, "m/f:macaddr_not" },
	{ NULL, "macaddr macaddr_and(macaddr,macaddr)", 8, "m/f:macaddr_and" },
	{ NULL, "macaddr macaddr_or(macaddr,macaddr)",  8, "m/f:macaddr_or" },

	/* inet/cidr functions */
	{ NULL, "iner set_masklen(inet,int4)", 8, "m/f:inet_set_masklen" },
	{ NULL, "cidr set_masklen(cidr,int4)", 8, "m/f:cidr_set_masklen" },
	{ NULL, "int4 family(inet)",           8, "m/f:inet_family" },
	{ NULL, "cidr network(inet)",          8, "m/f:network_network" },
	{ NULL, "inet netmask(inet)",          8, "m/f:inet_netmask" },
	{ NULL, "int4 masklen(inet)",          8, "m/f:inet_masklen" },
	{ NULL, "inet broadcast(inet)",        8, "m/f:inet_broadcast" },
	{ NULL, "iner hostmask(inet)",         8, "m/f:inet_hostmask" },
	{ NULL, "cidr cidr(iner)",             8, "m/f:inet_to_cidr" },
	{ NULL, "inet inetnot(inet)",          8, "m/f:inet_not" },
	{ NULL, "inet inetand(inet,inet)",     8, "m/f:inet_and" },
	{ NULL, "inet inetor(inet,inet)",      8, "m/f:inet_or" },
	{ NULL, "inet inetpl(inet,int8)",      8, "m/f:inetpl_int8" },
	{ NULL, "inet inetmi_int8(inet,int8)", 8, "m/f:inetmi_int8" },
	{ NULL, "int8 inetmi(inet,inet)",      8, "m/f:inetmi" },
	{ NULL, "bool inet_same_family(inet,inet)", 8, "m/f:inet_same_family" },
//	{ NULL, "inet inet_merge(inet,inet)",  8, "m/f:inet_merge" },

	/*
	 * Text functions
	 */
	{ NULL, "bool bpchareq(bpchar,bpchar)", 200, "s/f:bpchareq" },
	{ NULL, "bool bpcharne(bpchar,bpchar)", 200, "s/f:bpcharne" },
	{ NULL, "bool bpcharlt(bpchar,bpchar)", 200, "sL/f:bpcharlt" },
	{ NULL, "bool bpcharle(bpchar,bpchar)", 200, "sL/f:bpcharle" },
	{ NULL, "bool bpchargt(bpchar,bpchar)", 200, "sL/f:bpchargt" },
	{ NULL, "bool bpcharge(bpchar,bpchar)", 200, "sL/f:bpcharge" },
	{ NULL, "int4 bpcharcmp(bpchar,bpchar)",200, "sL/f:type_compare"},
	{ NULL, "int4 length(bpchar)",            2, "sL/f:bpcharlen"},
	{ NULL, "bool texteq(text,text)",       200, "s/f:texteq" },
	{ NULL, "bool textne(text,text)",       200, "s/f:textne" },
	{ NULL, "bool text_lt(text,text)",      200, "sL/f:text_lt" },
	{ NULL, "bool text_le(text,text)",      200, "sL/f:text_le" },
	{ NULL, "bool text_gt(text,text)",      200, "sL/f:text_gt" },
	{ NULL, "bool text_ge(text,text)",      200, "sL/f:text_ge" },
	{ NULL, "int4 bttextcmp(text,text)",    200, "sL/f:type_compare" },
	/* LIKE operators */
	{ NULL, "bool like(text,text)",           9999, "s/f:textlike" },
	{ NULL, "bool textlike(text,text)",       9999, "s/f:textlike" },
	{ NULL, "bool bpcharlike(bpchar,text)",   9999, "s/f:bpcharlike" },
	{ NULL, "bool notlike(text,text)",        9999, "s/f:textnlike" },
	{ NULL, "bool textnlike(text,text)",      9999, "s/f:textnlike" },
	{ NULL, "bool bpcharnlike(bpchar,text)",  9999, "s/f:bpcharnlike" },
	/* ILIKE operators */
	{ NULL, "bool texticlike(text,text)",     9999, "Ls/f:texticlike" },
	{ NULL, "bool bpchariclike(text,text)",   9999, "Ls/f:bpchariclike" },
	{ NULL, "bool texticnlike(text,text)",    9999, "Ls/f:texticnlike" },
	{ NULL, "bool bpcharicnlike(bpchar,text)",9999, "Ls/f:bpcharicnlike" },
	/* string operations */
	{ NULL, "int4 length(text)", 2, "s/f:textlen" },
	{ NULL, "text textcat(text,text)",
	  999, "Cs/f:textcat",
	  vlbuf_estimate_textcat
	},
	{ NULL, "text concat(text,text)",
	  999, "Cs/f:text_concat2",
	  vlbuf_estimate_textcat
	},
	{ NULL, "text concat(text,text,text)",
	  999, "Cs/f:text_concat3",
	  vlbuf_estimate_textcat
	},
	{ NULL, "text concat(text,text,text,text)",
	  999, "Cs/f:text_concat4",
	  vlbuf_estimate_textcat
	},
	{ NULL, "text substr(text,int4,int4)",
	  10, "Cs/f:text_substring",
	  vlbuf_estimate_substring
	},
	{ NULL, "text substring(text,int4,int4)",
	  10, "Cs/f:text_substring",
	  vlbuf_estimate_substring
	},
	{ NULL, "text substr(text,int4)",
	  10, "Cs/f:text_substring_nolen",
	  vlbuf_estimate_substring
	},
	{ NULL, "text substring(text,int4)",
	  10, "Cs/f:text_substring_nolen",
	  vlbuf_estimate_substring
	},
	/* jsonb operators */
	{ NULL, "jsonb jsonb_object_field(jsonb,text)",
	  1000, "jC/f:jsonb_object_field",
	  vlbuf_estimate_jsonb
	},
	{ NULL, "text jsonb_object_field_text(jsonb,text)",
	  1000, "jC/f:jsonb_object_field_text",
	  vlbuf_estimate_jsonb
	},
	{ NULL, "jsonb jsonb_array_element(jsonb,int4)",
	  1000, "jC/f:jsonb_array_element",
	  vlbuf_estimate_jsonb
	},
	{ NULL, "text jsonb_array_element_text(jsonb,int4)",
	  1000, "jC/f:jsonb_array_element_text",
	  vlbuf_estimate_jsonb
	},
	{ NULL, "bool jsonb_exists(jsonb,text)",
	  100, "j/f:jsonb_exists"
	},
	/*
	 * int4range operators
	 */
	{ NULL, "int4 lower(int4range)", 2, "r/f:int4range_lower" },
	{ NULL, "int4 upper(int4range)", 2, "r/f:int4range_upper" },
	{ NULL, "bool isempty(int4range)", 1, "r/f:int4range_isempty" },
	{ NULL, "bool lower_inc(int4range)", 1, "r/f:int4range_lower_inc" },
	{ NULL, "bool upper_inc(int4range)", 1, "r/f:int4range_upper_inc" },
	{ NULL, "bool lower_inf(int4range)", 1, "r/f:int4range_lower_inf" },
	{ NULL, "bool upper_inf(int4range)", 1, "r/f:int4range_upper_inf" },
	{ NULL, "bool range_eq(int4range,int4range)", 2, "r/f:int4range_eq" },
	{ NULL, "bool range_ne(int4range,int4range)", 2, "r/f:int4range_ne" },
	{ NULL, "bool range_lt(int4range,int4range)", 2, "r/f:int4range_lt" },
	{ NULL, "bool range_le(int4range,int4range)", 2, "r/f:int4range_le" },
	{ NULL, "bool range_gt(int4range,int4range)", 2, "r/f:int4range_gt" },
	{ NULL, "bool range_ge(int4range,int4range)", 2, "r/f:int4range_ge" },
	{ NULL, "int4 range_cmp(int4range,int4range)",2, "r/f:int4range_cmp"},
	{ NULL, "bool range_overlaps(int4range,int4range)",
	  4, "r/f:int4range_overlaps" },
	{ NULL, "bool range_contains_elem(int4range,int4)",
	  4, "r/f:int4range_contains_elem" },
	{ NULL, "bool range_contains(int4range,int4range)",
	  4, "r/f:int4range_contains" },
	{ NULL, "bool elem_contained_by_range(int4,int4range)",
	  4, "r/f:elem_contained_by_int4range" },
	{ NULL, "bool range_contained_by(int4range,int4range)",
	  4, "r/f:int4range_contained_by" },
	{ NULL, "bool range_adjacent(int4range,int4range)",
	  4, "r/f:int4range_adjacent" },
	{ NULL, "bool range_before(int4range,int4range)",
	  4, "r/f:int4range_before" },
	{ NULL, "bool range_after(int4range,int4range)",
	  4, "r/f:int4range_after" },
	{ NULL, "bool range_overleft(int4range,int4range)",
	  4, "r/f:int4range_overleft" },
	{ NULL, "bool range_overright(int4range,int4range)",
	  4, "r/f:int4range_overright" },
	{ NULL, "int4range range_union(int4range,int4range)",
	  4, "r/f:int4range_union" },
	{ NULL, "int4range range_merge(int4range,int4range)",
	  4, "r/f:int4range_merge" },
	{ NULL, "int4range range_intersect(int4range,int4range)",
	  4, "r/f:int4range_intersect" },
	{ NULL, "int4range range_minus(int4range,int4range)",
	  4, "r/f:int4range_minus" },
	/*
	 * int8range operators
	 */
	{ NULL, "int8 lower(int8range)", 2, "r/f:int8range_lower" },
	{ NULL, "int8 upper(int8range)", 2, "r/f:int8range_upper" },
	{ NULL, "bool isempty(int8range)", 1, "r/f:int8range_isempty" },
	{ NULL, "bool lower_inc(int8range)", 1, "r/f:int8range_lower_inc" },
	{ NULL, "bool upper_inc(int8range)", 1, "r/f:int8range_upper_inc" },
	{ NULL, "bool lower_inf(int8range)", 1, "r/f:int8range_lower_inf" },
	{ NULL, "bool upper_inf(int8range)", 1, "r/f:int8range_upper_inf" },
	{ NULL, "bool range_eq(int8range,int8range)", 2, "r/f:int8range_eq" },
	{ NULL, "bool range_ne(int8range,int8range)", 2, "r/f:int8range_ne" },
	{ NULL, "bool range_lt(int8range,int8range)", 2, "r/f:int8range_lt" },
	{ NULL, "bool range_le(int8range,int8range)", 2, "r/f:int8range_le" },
	{ NULL, "bool range_gt(int8range,int8range)", 2, "r/f:int8range_gt" },
	{ NULL, "bool range_ge(int8range,int8range)", 2, "r/f:int8range_ge" },
	{ NULL, "int4 range_cmp(int8range,int8range)",2, "r/f:int8range_cmp"},
	{ NULL, "bool range_overlaps(int8range,int8range)",
	  4, "r/f:int8range_overlaps" },
	{ NULL, "bool range_contains_elem(int8range,int8)",
	  4, "r/f:int8range_contains_elem" },
	{ NULL, "bool range_contains(int8range,int8range)",
	  4, "r/f:int8range_contains" },
	{ NULL, "bool elem_contained_by_range(int8,int8range)",
	  4, "r/f:elem_contained_by_int8range" },
	{ NULL, "bool range_contained_by(int8range,int8range)",
	  4, "r/f:int8range_contained_by" },
	{ NULL, "bool range_adjacent(int8range,int8range)",
	  4, "r/f:int8range_adjacent" },
	{ NULL, "bool range_before(int8range,int8range)",
	  4, "r/f:int8range_before" },
	{ NULL, "bool range_after(int8range,int8range)",
	  4, "r/f:int8range_after" },
	{ NULL, "bool range_overleft(int8range,int8range)",
	  4, "r/f:int8range_overleft" },
	{ NULL, "bool range_overright(int8range,int8range)",
	  4, "r/f:int8range_overright" },
	{ NULL, "int8range range_union(int8range,int8range)",
	  4, "r/f:int8range_union" },
	{ NULL, "int8range range_merge(int8range,int8range)",
	  4, "r/f:int8range_merge" },
	{ NULL, "int8range range_intersect(int8range,int8range)",
	  4, "r/f:int8range_intersect" },
	{ NULL, "int8range range_minus(int8range,int8range)",
	  4, "r/f:int8range_minus" },
	/*
	 * tsrange operators
	 */
	{ NULL, "timestamp lower(tsrange)", 2, "r/f:tsrange_lower" },
	{ NULL, "timestamp upper(tsrange)", 2, "r/f:tsrange_upper" },
	{ NULL, "bool isempty(tsrange)", 1, "r/f:tsrange_isempty" },
	{ NULL, "bool lower_inc(tsrange)", 1, "r/f:tsrange_lower_inc" },
	{ NULL, "bool upper_inc(tsrange)", 1, "r/f:tsrange_upper_inc" },
	{ NULL, "bool lower_inf(tsrange)", 1, "r/f:tsrange_lower_inf" },
	{ NULL, "bool upper_inf(tsrange)", 1, "r/f:tsrange_upper_inf" },
	{ NULL, "bool range_eq(tsrange,tsrange)", 2, "r/f:tsrange_eq" },
	{ NULL, "bool range_ne(tsrange,tsrange)", 2, "r/f:tsrange_ne" },
	{ NULL, "bool range_lt(tsrange,tsrange)", 2, "r/f:tsrange_lt" },
	{ NULL, "bool range_le(tsrange,tsrange)", 2, "r/f:tsrange_le" },
	{ NULL, "bool range_gt(tsrange,tsrange)", 2, "r/f:tsrange_gt" },
	{ NULL, "bool range_ge(tsrange,tsrange)", 2, "r/f:tsrange_ge" },
	{ NULL, "int4 range_cmp(tsrange,tsrange)",2, "r/f:tsrange_cmp"},
	{ NULL, "bool range_overlaps(tsrange,tsrange)",
	  4, "r/f:tsrange_overlaps" },
	{ NULL, "bool range_contains_elem(tsrange,timestamp)",
	  4, "r/f:tsrange_contains_elem" },
	{ NULL, "bool range_contains(tsrange,tsrange)",
	  4, "r/f:tsrange_contains" },
	{ NULL, "bool elem_contained_by_range(timestamp,tsrange)",
	  4, "r/f:elem_contained_by_tsrange" },
	{ NULL, "bool range_contained_by(tsrange,tsrange)",
	  4, "r/f:tsrange_contained_by" },
	{ NULL, "bool range_adjacent(tsrange,tsrange)",
	  4, "r/f:tsrange_adjacent" },
	{ NULL, "bool range_before(tsrange,tsrange)",
	  4, "r/f:tsrange_before" },
	{ NULL, "bool range_after(tsrange,tsrange)",
	  4, "r/f:tsrange_after" },
	{ NULL, "bool range_overleft(tsrange,tsrange)",
	  4, "r/f:tsrange_overleft" },
	{ NULL, "bool range_overright(tsrange,tsrange)",
	  4, "r/f:tsrange_overright" },
	{ NULL, "tsrange range_union(tsrange,tsrange)",
	  4, "r/f:tsrange_union" },
	{ NULL, "tsrange range_merge(tsrange,tsrange)",
	  4, "r/f:tsrange_merge" },
	{ NULL, "tsrange range_intersect(tsrange,tsrange)",
	  4, "r/f:tsrange_intersect" },
	{ NULL, "tsrange range_minus(tsrange,tsrange)",
	  4, "r/f:tsrange_minus" },
	/*
	 * tstzrange operators
	 */
	{ NULL, "timestamptz lower(tstzrange)", 2, "r/f:tstzrange_lower" },
	{ NULL, "timestamptz upper(tstzrange)", 2, "r/f:tstzrange_upper" },
	{ NULL, "bool isempty(tstzrange)", 1, "r/f:tstzrange_isempty" },
	{ NULL, "bool lower_inc(tstzrange)", 1, "r/f:tstzrange_lower_inc" },
	{ NULL, "bool upper_inc(tstzrange)", 1, "r/f:tstzrange_upper_inc" },
	{ NULL, "bool lower_inf(tstzrange)", 1, "r/f:tstzrange_lower_inf" },
	{ NULL, "bool upper_inf(tstzrange)", 1, "r/f:tstzrange_upper_inf" },
	{ NULL, "bool range_eq(tstzrange,tstzrange)", 2, "r/f:tstzrange_eq" },
	{ NULL, "bool range_ne(tstzrange,tstzrange)", 2, "r/f:tstzrange_ne" },
	{ NULL, "bool range_lt(tstzrange,tstzrange)", 2, "r/f:tstzrange_lt" },
	{ NULL, "bool range_le(tstzrange,tstzrange)", 2, "r/f:tstzrange_le" },
	{ NULL, "bool range_gt(tstzrange,tstzrange)", 2, "r/f:tstzrange_gt" },
	{ NULL, "bool range_ge(tstzrange,tstzrange)", 2, "r/f:tstzrange_ge" },
	{ NULL, "int4 range_cmp(tstzrange,tstzrange)",2, "r/f:tstzrange_cmp"},
	{ NULL, "bool range_overlaps(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_overlaps" },
	{ NULL, "bool range_contains_elem(tstzrange,timestamptz)",
	  4, "r/f:tstzrange_contains_elem" },
	{ NULL, "bool range_contains(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_contains" },
	{ NULL, "bool elem_contained_by_range(timestamptz,tstzrange)",
	  4, "r/f:elem_contained_by_tstzrange" },
	{ NULL, "bool range_contained_by(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_contained_by" },
	{ NULL, "bool range_adjacent(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_adjacent" },
	{ NULL, "bool range_before(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_before" },
	{ NULL, "bool range_after(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_after" },
	{ NULL, "bool range_overleft(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_overleft" },
	{ NULL, "bool range_overright(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_overright" },
	{ NULL, "tstzrange range_union(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_union" },
	{ NULL, "tstzrange range_merge(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_merge" },
	{ NULL, "tstzrange range_intersect(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_intersect" },
	{ NULL, "tstzrange range_minus(tstzrange,tstzrange)",
	  4, "r/f:tstzrange_minus" },
	/*
	 * daterange operators
	 */
	{ NULL, "date lower(daterange)", 2, "r/f:daterange_lower" },
	{ NULL, "date upper(daterange)", 2, "r/f:daterange_upper" },
	{ NULL, "bool isempty(daterange)", 1, "r/f:daterange_isempty" },
	{ NULL, "bool lower_inc(daterange)", 1, "r/f:daterange_lower_inc" },
	{ NULL, "bool upper_inc(daterange)", 1, "r/f:daterange_upper_inc" },
	{ NULL, "bool lower_inf(daterange)", 1, "r/f:daterange_lower_inf" },
	{ NULL, "bool upper_inf(daterange)", 1, "r/f:daterange_upper_inf" },
	{ NULL, "bool range_eq(daterange,daterange)", 2, "r/f:daterange_eq" },
	{ NULL, "bool range_ne(daterange,daterange)", 2, "r/f:daterange_ne" },
	{ NULL, "bool range_lt(daterange,daterange)", 2, "r/f:daterange_lt" },
	{ NULL, "bool range_le(daterange,daterange)", 2, "r/f:daterange_le" },
	{ NULL, "bool range_gt(daterange,daterange)", 2, "r/f:daterange_gt" },
	{ NULL, "bool range_ge(daterange,daterange)", 2, "r/f:daterange_ge" },
	{ NULL, "int4 range_cmp(daterange,daterange)",2, "r/f:daterange_cmp"},
	{ NULL, "bool range_overlaps(daterange,daterange)",
	  4, "r/f:daterange_overlaps" },
	{ NULL, "bool range_contains_elem(daterange,date)",
	  4, "r/f:daterange_contains_elem" },
	{ NULL, "bool range_contains(daterange,daterange)",
	  4, "r/f:daterange_contains" },
	{ NULL, "bool elem_contained_by_range(date,daterange)",
	  4, "r/f:elem_contained_by_daterange" },
	{ NULL, "bool range_contained_by(daterange,daterange)",
	  4, "r/f:daterange_contained_by" },
	{ NULL, "bool range_adjacent(daterange,daterange)",
	  4, "r/f:daterange_adjacent" },
	{ NULL, "bool range_before(daterange,daterange)",
	  4, "r/f:daterange_before" },
	{ NULL, "bool range_after(daterange,daterange)",
	  4, "r/f:daterange_after" },
	{ NULL, "bool range_overleft(daterange,daterange)",
	  4, "r/f:daterange_overleft" },
	{ NULL, "bool range_overright(daterange,daterange)",
	  4, "r/f:daterange_overright" },
	{ NULL, "daterange range_union(daterange,daterange)",
	  4, "r/f:daterange_union" },
	{ NULL, "daterange range_merge(daterange,daterange)",
	  4, "r/f:daterange_merge" },
	{ NULL, "daterange range_intersect(daterange,daterange)",
	  4, "r/f:daterange_intersect" },
	{ NULL, "daterange range_minus(daterange,daterange)",
	  4, "r/f:daterange_minus" },

	/*
	 * PostGIS functions
	 */
	{ POSTGIS3, "geometry st_setsrid(geometry,int4)",
	  1, "g/f:st_setsrid" },
	{ POSTGIS3, "geometry st_point(float8,float8)",
	  10, "gC/f:st_makepoint2",
	  vlbuf_estimate__st_makepoint },
	{ POSTGIS3, "geometry st_makepoint(float8,float8)",
	  10, "gC/f:st_makepoint2",
	  vlbuf_estimate__st_makepoint },
	{ POSTGIS3, "geometry st_makepoint(float8,float8,float8)",
	  10, "gC/f:st_makepoint3",
	  vlbuf_estimate__st_makepoint },
	{ POSTGIS3, "geometry st_makepoint(float8,float8,float8,float8)",
	  10, "gC/f:st_makepoint4",
	  vlbuf_estimate__st_makepoint },
	{ POSTGIS3, "float8 st_distance(geometry,geometry)",
	  50, "g/f:st_distance" },
	{ POSTGIS3, "bool st_dwithin(geometry,geometry,float8)",
	  50, "g/f:st_dwithin" },
	{ POSTGIS3, "int4 st_linecrossingdirection(geometry,geometry)",
	  50, "g/f:st_linecrossingdirection" },
	{ POSTGIS3, "text st_relate(geometry,geometry)",
	  999, "g/f:st_relate",
	  vlbuf_estimate__st_relate },
	{ POSTGIS3, "bool st_contains(geometry,geometry)",
	  999, "g/f:st_contains" },
	{ POSTGIS3, "bool st_crosses(geometry,geometry)",
	  999, "g/f:st_crosses" },
	{ POSTGIS3, "bool geometry_overlaps(geometry,geometry)",
	  10, "g/f:geometry_overlaps" },
	{ POSTGIS3, "bool overlaps_2d(box2df,geometry)",
	  10, "g/f:box2df_geometry_overlaps" },
	{ POSTGIS3, "bool geometry_contains(geometry,geometry)",
	  10, "g/f:geometry_contains" },
	{ POSTGIS3, "bool contains_2d(box2df,geometry)",
	  10, "g/f:box2df_geometry_contains" },
	{ POSTGIS3, "bool geometry_within(geometry,geometry)",
	  10, "g/f:geometry_within" },
	{ POSTGIS3, "bool is_contained_2d(box2df,geometry)",
	  10, "g/f:box2df_geometry_within" },
	{ POSTGIS3, "geometry st_expand(geometry,float8)",
	  20, "gC/f:st_expand",
	  vlbuf_estimate__st_expand },
	/*
	 * GpuPreAgg COUNT(distinct KEY) support
	 */
	{ PGSTROM, "int8 hll_hash(int1)",        1, "f:hll_hash_int1" },
	{ PGSTROM, "int8 hll_hash(int2)",        1, "f:hll_hash_int2" },
	{ PGSTROM, "int8 hll_hash(int4)",        1, "f:hll_hash_int4" },
	{ PGSTROM, "int8 hll_hash(int8)",        1, "f:hll_hash_int8" },
	{ PGSTROM, "int8 hll_hash(numeric)",     1, "f:hll_hash_numeric" },
	{ PGSTROM, "int8 hll_hash(date)",        1, "t/f:hll_hash_date" },
	{ PGSTROM, "int8 hll_hash(time)",        1, "t/f:hll_hash_time" },
	{ PGSTROM, "int8 hll_hash(timetz)",      1, "t/f:hll_hash_timetz" },
	{ PGSTROM, "int8 hll_hash(timestamp)",   1, "t/f:hll_hash_timestamp" },
	{ PGSTROM, "int8 hll_hash(timestamptz)", 1, "t/f:hll_hash_timestamptz" },
	{ PGSTROM, "int8 hll_hash(bpchar)",      1, "s/f:hll_hash_bpchar" },
	{ PGSTROM, "int8 hll_hash(text)",        1, "s/f:hll_hash_text" },
	{ PGSTROM, "int8 hll_hash(uuid)",        1, "m/f:hll_hash_uuid"}
};

/* default of dfunc->dfunc_varlena_sz if not specified */
static int
devfunc_generic_result_sz(codegen_context *context,
						  devfunc_info *dfunc,
						  Expr **args, int *vl_width)
{
	devtype_info   *rtype = dfunc->func_rettype;

	if (rtype->type_length > 0)
		return rtype->type_length;
	else if (rtype->type_length == -1)
		return type_maximum_size(rtype->type_oid, -1);
	elog(ERROR, "unexpected type length: %d", rtype->type_length);
}

static devfunc_info *
__construct_devfunc_info(const char *func_extension,
						 HeapTuple protup,
						 devtype_info *dfunc_rettype,
						 int dfunc_nargs,
						 devtype_info **dfunc_argtypes,
						 Oid dfunc_collid,
						 int func_devcost,
						 const char *func_template,
						 devfunc_result_sz_type devfunc_result_sz)
{
	Form_pg_proc    proc = (Form_pg_proc) GETSTRUCT(protup);
	MemoryContext	oldcxt;
	devfunc_info   *dfunc = NULL;
	List		   *dfunc_args = NIL;
	const char	   *pos;
	const char	   *end;
	int32			flags = 0;
	int				j;
	bool			has_collation = false;
	bool			has_callbacks = false;

	/* fetch attribute */
	end = strchr(func_template, '/');
	if (end)
	{
		for (pos = func_template; pos < end; pos++)
		{
			switch (*pos)
			{
				case 'L':
					has_collation = true;
					break;
				case 'C':
					has_callbacks = true;
					break;
				case 'p':
					flags |= DEVKERNEL_NEEDS_PRIMITIVE;
					break;
				case 's':
					flags |= DEVKERNEL_NEEDS_TEXTLIB;
					break;
				case 't':
					flags |= DEVKERNEL_NEEDS_TIMELIB;
					break;
				case 'j':
					flags |= DEVKERNEL_NEEDS_JSONLIB;
					break;
				case 'm':
					flags |= DEVKERNEL_NEEDS_MISCLIB;
					break;
				case 'r':
					flags |= DEVKERNEL_NEEDS_RANGETYPE;
					break;
				case 'g':
					flags |= DEVKERNEL_NEEDS_POSTGIS;
					break;
				default:
					elog(NOTICE,
						 "Bug? unkwnon devfunc property: %c",
						 *pos);
					break;
			}
		}
		func_template = end + 1;
	}
	if (strncmp(func_template, "f:", 2) != 0)
	{
		elog(NOTICE, "Bug? unknown device function template: '%s'",
			 func_template);
		return NULL;
	}
	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	for (j=0; j < dfunc_nargs; j++)
		dfunc_args = lappend(dfunc_args, dfunc_argtypes[j]);

	dfunc = palloc0(sizeof(devfunc_info));
	if (func_extension)
		dfunc->func_extension = pstrdup(func_extension);
	dfunc->func_oid = PgProcTupleGetOid(protup);
	if (has_collation)
	{
		if (OidIsValid(dfunc_collid) && !lc_collate_is_c(dfunc_collid))
			dfunc->func_is_negative = true;
		dfunc->func_collid = dfunc_collid;
	}
	dfunc->func_is_strict = proc->proisstrict;
	dfunc->func_flags = flags;
	dfunc->func_args = dfunc_args;
	dfunc->func_rettype = dfunc_rettype;
	dfunc->func_sqlname = pstrdup(NameStr(proc->proname));
	dfunc->func_devname = func_template + 2;	/* const cstring */
	dfunc->func_devcost = func_devcost;
	dfunc->devfunc_result_sz = (has_callbacks
								? devfunc_result_sz
								: devfunc_generic_result_sz);
	/* other fields shall be assigned on the caller side */
	MemoryContextSwitchTo(oldcxt);

	return dfunc;
}

static devfunc_info *
pgstrom_devfunc_construct_fuzzy(const char *func_extension,
								HeapTuple protup,
								devtype_info *dfunc_rettype,
								int dfunc_nargs,
								devtype_info **dfunc_argtypes,
								Oid dfunc_collid,
								int fuzzy_index_head,
								int fuzzy_index_tail)
{
	Form_pg_proc proc = (Form_pg_proc) GETSTRUCT(protup);
	char	buffer[512];
	int		i, j;

	Assert(fuzzy_index_head >= 0 &&
		   fuzzy_index_head <= fuzzy_index_tail &&
		   fuzzy_index_tail <  lengthof(devfunc_common_catalog));
	for (i = fuzzy_index_head; i <= fuzzy_index_tail; i++)
	{
		devfunc_catalog_t *procat = devfunc_common_catalog + i;
		devtype_info *dtype;
		char	   *tok;
		char	   *pos;

		if (func_extension)
		{
			if (!procat->func_extension ||
				strcmp(procat->func_extension, func_extension) != 0)
				continue;
		}
		else
		{
			if (procat->func_extension)
				continue;
		}
		strncpy(buffer, procat->func_signature, sizeof(buffer));
		pos = strchr(buffer, ' ');
		if (!pos)
			continue;
		*pos++ = '\0';

		/* check the function name */
		tok = pos;
		pos = strchr(pos, '(');
		if (!pos)
			continue;
		*pos++ = '\0';
		if (strcmp(tok, NameStr(proc->proname)) != 0)
			continue;

		/* check the argument types */
		for (j=0; j < dfunc_nargs; j++)
		{
			tok = pos;
			pos = strchr(pos, (j < dfunc_nargs - 1 ? ',' : ')'));
			if (!pos)
				break;		/* not match */
			*pos++ = '\0';

			dtype = pgstrom_devtype_lookup_by_name(tok);
			if (!dtype)
				break;		/* not match */
			if (dtype->type_oid != dfunc_argtypes[j]->type_oid &&
				!pgstrom_devtype_can_relabel(dfunc_argtypes[j]->type_oid,
											 dtype->type_oid))
				break;		/* not match */
		}
		if (j < dfunc_nargs)
			continue;
		/* check the result type */
		dtype = pgstrom_devtype_lookup_by_name(buffer);
		if (!dtype)
			continue;
		if (dtype->type_oid != dfunc_rettype->type_oid &&
			!pgstrom_devtype_can_relabel(dtype->type_oid,
										 dfunc_rettype->type_oid))
			continue;

		/* Ok, found the fuzzy entry */
		return __construct_devfunc_info(func_extension,
										protup,
										dfunc_rettype,
										dfunc_nargs,
										dfunc_argtypes,
										dfunc_collid,
										procat->func_devcost,
										procat->func_template,
										procat->devfunc_result_sz);
	}
	/* not found */
	return NULL;
}

static devfunc_info *
build_extra_devfunc_info(const char *func_extension,
						 HeapTuple protup,
						 devtype_info *dfunc_rettype,
						 int dfunc_nargs,
						 devtype_info **dfunc_argtypes,
						 Oid dfunc_collid)
{
	Form_pg_proc	proc_form = (Form_pg_proc) GETSTRUCT(protup);
	StringInfoData	ident;
	devfunc_info	__dfunc;
	devfunc_info   *dfunc = NULL;
	List		   *dfunc_args = NIL;
	const char	   *nsp_name;
	int				i;

	/* setup devfunc identifier */
	initStringInfo(&ident);
	append_string_devtype_identifier(&ident, dfunc_rettype->type_oid);
	nsp_name = get_namespace_name(proc_form->pronamespace);
	appendStringInfo(&ident, " %s.%s(",
					 quote_identifier(nsp_name),
					 quote_identifier(NameStr(proc_form->proname)));
	for (i=0; i < dfunc_nargs; i++)
	{
		devtype_info   *dtype = dfunc_argtypes[i];

		if (i > 0)
			appendStringInfoChar(&ident, ',');
		append_string_devtype_identifier(&ident, dtype->type_oid);
		dfunc_args = lappend(dfunc_args, dtype);
	}
	appendStringInfoChar(&ident, ')');

	memset(&__dfunc, 0, sizeof(devfunc_info));
	__dfunc.func_extension = func_extension;
	__dfunc.func_oid = PgProcTupleGetOid(protup);
	__dfunc.hashvalue = GetSysCacheHashValue(PROCOID, __dfunc.func_oid, 0, 0, 0);
	__dfunc.func_collid = dfunc_collid;
	__dfunc.func_is_strict = proc_form->proisstrict;
	__dfunc.func_args = dfunc_args;
	__dfunc.func_rettype = dfunc_rettype;
	__dfunc.func_sqlname = NameStr(proc_form->proname);
	__dfunc.func_devname = NULL;		/* callback must set */
	__dfunc.func_devcost = 0;			/* callback must set */
	__dfunc.devfunc_result_sz = NULL;	/* callback must set, if any */

	for (i=0; i < pgstrom_num_users_extra; i++)
	{
		pgstromUsersExtraDescriptor *extra = &pgstrom_users_extra_desc[i];

		if (extra->lookup_extra_devfunc &&
			extra->lookup_extra_devfunc(ident.data, &__dfunc))
		{
			MemoryContext	oldcxt;

			/* must be */
			if (!__dfunc.func_devname)
			{
				elog(DEBUG2, "Extra module didn't set device function name for %s",
					 format_procedure(__dfunc.func_oid));
				continue;
			}
			oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
			dfunc = palloc0(sizeof(devfunc_info));
			dfunc->func_extension = pstrdup(__dfunc.func_extension);
			dfunc->func_oid = __dfunc.func_oid;
			dfunc->func_collid = __dfunc.func_collid;
			dfunc->func_is_negative = __dfunc.func_is_negative;
			dfunc->func_is_strict = __dfunc.func_is_strict;
			dfunc->func_flags = __dfunc.func_flags;
			dfunc->func_args = list_copy(__dfunc.func_args);
			dfunc->func_rettype = __dfunc.func_rettype;
			dfunc->func_sqlname = pstrdup(__dfunc.func_sqlname);
			dfunc->func_devname = pstrdup(__dfunc.func_devname);
			if (__dfunc.devfunc_result_sz)
				dfunc->devfunc_result_sz = __dfunc.devfunc_result_sz;
			else
				dfunc->devfunc_result_sz = devfunc_generic_result_sz;
			MemoryContextSwitchTo(oldcxt);
			break;
		}
	}
	pfree(ident.data);
	return dfunc;
}

static devfunc_info *
pgstrom_devfunc_construct(HeapTuple protup,
						  Oid func_rettype,
						  oidvector *func_argtypes,
						  Oid func_collid)
{
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	const char	   *func_extension;
	StringInfoData	sig;
	devtype_info   *dtype;
	devtype_info   *dfunc_rettype;
	devtype_info  **dfunc_argtypes;
	devfunc_info   *dfunc = NULL;
	int				fuzzy_index_head = -1;
	int				fuzzy_index_tail = -1;
	int				i;

	/* extension name */
	func_extension = get_extension_name_by_object(ProcedureRelationId,
												  PgProcTupleGetOid(protup));
	/* make a signature string */
	initStringInfo(&sig);
	dfunc_rettype = pgstrom_devtype_lookup(func_rettype);
	if (!dfunc_rettype)
		goto not_found;
	appendStringInfo(&sig, "%s %s(",
					 dfunc_rettype->type_name,
					 NameStr(proc->proname));

	dfunc_argtypes = alloca(sizeof(devtype_info *) * func_argtypes->dim1);
	for (i=0; i < func_argtypes->dim1; i++)
	{
		dtype = pgstrom_devtype_lookup(func_argtypes->values[i]);
		if (!dtype)
			goto not_found;
		if (i > 0)
			appendStringInfoChar(&sig, ',');
		appendStringInfo(&sig, "%s", dtype->type_name);
		dfunc_argtypes[i] = dtype;
	}
	appendStringInfoChar(&sig, ')');

	for (i=0; i < lengthof(devfunc_common_catalog); i++)
	{
		devfunc_catalog_t  *procat = &devfunc_common_catalog[i];

		if (func_extension)
		{
			if (!procat->func_extension ||
				strcmp(procat->func_extension, func_extension) != 0)
				continue;
		}
		else if (procat->func_extension)
			continue;

		if (strcmp(procat->func_signature, sig.data) == 0)
		{
			dfunc = __construct_devfunc_info(func_extension,
											 protup,
											 dfunc_rettype,
											 func_argtypes->dim1,
											 dfunc_argtypes,
											 func_collid,
											 procat->func_devcost,
											 procat->func_template,
											 procat->devfunc_result_sz);
			break;
		}
		else
		{
			/*
			 * In case when function name is identical, but argument list
			 * does not match exactly. (
			 */
			const char *sname = strchr(procat->func_signature, ' ');
			const char *pname = NameStr(proc->proname);

			if (sname)
			{
				sname++;
				while (*sname != '\0' &&
					   *pname != '\0' &&
					   *sname == *pname)
				{
					sname++;
					pname++;
				}
				if (*sname == '(' && *pname == '\0')
				{
					if (fuzzy_index_head < 0)
						fuzzy_index_head = i;
					fuzzy_index_tail = i;
				}
			}
		}
	}
	/* try invocation with implicit type relabel */
	if (!dfunc && fuzzy_index_head >= 0)
	{
		dfunc = pgstrom_devfunc_construct_fuzzy(func_extension,
												protup,
												dfunc_rettype,
												func_argtypes->dim1,
												dfunc_argtypes,
												func_collid,
												fuzzy_index_head,
												fuzzy_index_tail);
	}
	/* extra device function, if any */
	if (!dfunc && pgstrom_num_users_extra > 0)
	{
		dfunc = build_extra_devfunc_info(func_extension,
										 protup,
										 dfunc_rettype,
										 func_argtypes->dim1,
										 dfunc_argtypes,
										 func_collid);
	}
not_found:
	pfree(sig.data);
	return dfunc;
}

static devfunc_info *
__pgstrom_devfunc_lookup(HeapTuple protup,
						 Oid func_rettype,
						 oidvector *func_argtypes,
						 Oid func_collid)
{
	Oid				func_oid = PgProcTupleGetOid(protup);
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	devfunc_info   *dfunc;
	devtype_info   *dtype;
	ListCell	   *lc;
	cl_uint			hashvalue;
	int				j, hindex;
	dlist_iter		iter;
	bool			consider_relabel = false;

	hashvalue = GetSysCacheHashValue(PROCOID, func_oid, 0, 0, 0);
	hindex = hashvalue % lengthof(devfunc_info_slot);
retry:
	dlist_foreach (iter, &devfunc_info_slot[hindex])
	{
		dfunc = dlist_container(devfunc_info, chain, iter.cur);
		if (dfunc->func_oid != func_oid)
			continue;
		if (OidIsValid(dfunc->func_collid) &&
			dfunc->func_collid != func_collid)
			continue;

		dtype = dfunc->func_rettype;
		if (dtype->type_oid != func_rettype &&
			(!consider_relabel ||
			 !pgstrom_devtype_can_relabel(dtype->type_oid, func_rettype)))
			continue;

		if (list_length(dfunc->func_args) == func_argtypes->dim1)
		{
			j = 0;
			foreach (lc, dfunc->func_args)
			{
				dtype = lfirst(lc);
				if (dtype->type_oid != func_argtypes->values[j] &&
					(!consider_relabel ||
					 !pgstrom_devtype_can_relabel(func_argtypes->values[j],
												  dtype->type_oid)))
					break;		/* not match */
				j++;
			}
			if (!lc)
			{
				if (dfunc->func_is_negative)
					return NULL;
				return dfunc;
			}
		}
	}
	if (!consider_relabel)
	{
		consider_relabel = true;
		goto retry;
	}

	/* Not cached, construct a new entry of the device function */
	dfunc = pgstrom_devfunc_construct(protup,
									  func_rettype,
									  func_argtypes,
									  func_collid);
	/* Not found, so this function should be a nagative entry */
	if (!dfunc)
	{
		MemoryContext	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

		/* dummy devtype_info just for oid checks */
		dfunc = palloc0(sizeof(devfunc_info));
		dfunc->func_oid = func_oid;
		dfunc->func_is_negative = true;
		for (j=0; j < func_argtypes->dim1; j++)
		{
			dtype = palloc0(sizeof(devtype_info));
			dtype->type_oid = func_argtypes->values[j];
			dfunc->func_args = lappend(dfunc->func_args, dtype);
		}
		dtype = palloc0(sizeof(devtype_info));
		dtype->type_oid = func_rettype;
		dfunc->func_rettype = dtype;
		dfunc->func_sqlname = pstrdup(NameStr(proc->proname));

		MemoryContextSwitchTo(oldcxt);
	}
	dfunc->hashvalue = hashvalue;
	dlist_push_head(&devfunc_info_slot[hindex], &dfunc->chain);
	if (dfunc->func_is_negative)
		return NULL;
	return dfunc;
}

devfunc_info *
pgstrom_devfunc_lookup(Oid func_oid,
					   Oid func_rettype,
					   List *func_args,	/* list of expressions */
					   Oid func_collid)
{
	devfunc_info   *result = NULL;
	HeapTuple		tup;

	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	PG_TRY();
	{
		int			func_nargs = list_length(func_args);
		oidvector  *func_argtypes;
		int			i = 0;
		ListCell   *lc;

		func_argtypes = alloca(offsetof(oidvector, values[func_nargs]));
		func_argtypes->ndim = 1;
		func_argtypes->dataoffset = 0;
		func_argtypes->elemtype = OIDOID;
		func_argtypes->dim1 = func_nargs;
		func_argtypes->lbound1 = 0;
		foreach (lc, func_args)
		{
			Oid		type_oid = exprType((Node *)lfirst(lc));

			func_argtypes->values[i++] = type_oid;
		}
		SET_VARSIZE(func_argtypes, offsetof(oidvector, values[func_nargs]));

		result = __pgstrom_devfunc_lookup(tup,
										  func_rettype,
										  func_argtypes,
										  func_collid);
	}
	PG_CATCH();
	{
		ReleaseSysCache(tup);
		PG_RE_THROW();
	}
	PG_END_TRY();
	ReleaseSysCache(tup);

	return result;
}

devfunc_info *
pgstrom_devfunc_lookup_type_equal(devtype_info *dtype, Oid type_collid)
{
	devfunc_info *result = NULL;
	char		buffer[offsetof(oidvector, values[2])];
	oidvector  *func_argtypes = (oidvector *)buffer;
	HeapTuple	tup;
	Form_pg_proc proc	__attribute__((unused));

	if (!OidIsValid(dtype->type_eqfunc))
		return NULL;
	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(dtype->type_eqfunc));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u",
			 dtype->type_eqfunc);
	PG_TRY();
	{
		proc = (Form_pg_proc) GETSTRUCT(tup);
		Assert(proc->pronargs == 2);
		Assert(proc->prorettype == BOOLOID);

		memset(func_argtypes, 0, offsetof(oidvector, values[2]));
		func_argtypes->ndim = 1;
		func_argtypes->dataoffset = 0;
		func_argtypes->elemtype = OIDOID;
		func_argtypes->dim1 = 2;
		func_argtypes->lbound1 = 0;
		func_argtypes->values[0] = dtype->type_oid;
		func_argtypes->values[1] = dtype->type_oid;
		SET_VARSIZE(func_argtypes, offsetof(oidvector, values[2]));

		result = __pgstrom_devfunc_lookup(tup,
										  BOOLOID,
										  func_argtypes,
										  type_collid);
	}
	PG_CATCH();
	{
		ReleaseSysCache(tup);
		PG_RE_THROW();
	}
	PG_END_TRY();
	ReleaseSysCache(tup);

	return result;
}

devfunc_info *
pgstrom_devfunc_lookup_type_compare(devtype_info *dtype, Oid type_collid)
{
	devfunc_info *result = NULL;
	char		buffer[offsetof(oidvector, values[2])];
	oidvector  *func_argtypes = (oidvector *)buffer;
	HeapTuple	tup;
	Form_pg_proc proc	__attribute__((unused));

	if (!OidIsValid(dtype->type_cmpfunc))
		return NULL;
	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(dtype->type_cmpfunc));
	if (!HeapTupleIsValid(tup))
        elog(ERROR, "cache lookup failed for function %u",
			 dtype->type_cmpfunc);
	PG_TRY();
	{
		proc = (Form_pg_proc) GETSTRUCT(tup);
		Assert(proc->pronargs == 2);
		Assert(proc->prorettype == INT4OID);

		memset(func_argtypes, 0, offsetof(oidvector, values[2]));
		func_argtypes->ndim = 1;
		func_argtypes->dataoffset = 0;
		func_argtypes->elemtype = OIDOID;
		func_argtypes->dim1 = 2;
		func_argtypes->lbound1 = 0;
		func_argtypes->values[0] = dtype->type_oid;
		func_argtypes->values[1] = dtype->type_oid;
		SET_VARSIZE(func_argtypes, offsetof(oidvector, values[2]));

		result = __pgstrom_devfunc_lookup(tup,
										  INT4OID,
										  func_argtypes,
										  type_collid);
	}
	PG_CATCH();
	{
		ReleaseSysCache(tup);
		PG_RE_THROW();
	}
	PG_END_TRY();
	ReleaseSysCache(tup);

	return result;
}

void
pgstrom_devfunc_track(codegen_context *context, devfunc_info *dfunc)
{
	devtype_info   *dtype = dfunc->func_rettype;
	ListCell	   *lc;

	/* track device function */
	context->extra_flags |= (dfunc->func_flags | dtype->type_flags);
	foreach (lc, dfunc->func_args)
	{
		dtype = (devtype_info *) lfirst(lc);
		context->extra_flags |= dtype->type_flags;
	}
}

/*
 * Device cast support
 *
 * In some cases, a function can be called with different argument types or
 * result type from its declaration, if these types are binary compatible.
 * PostgreSQL does not have any infrastructure to check data types, it relies
 * on the caller which shall give correct data types, and binary-compatible
 * types will work without any problems.
 * On the other hands, CUDA C++ has strict type checks for function invocation,
 * so we need to inject a thin type cast device function even if they are
 * binary compatible.
 * The thin device function has the following naming convention:
 *
 *   STATIC_INLINE(DESTTYPE) to_DESTTYPE(kcxt, SOURCETYPE)
 *
 * We have no SQL function on host side because the above device function 
 * reflects binary-compatible type cast. If cast is COERCION_METHOD_FUNCTION,
 * SQL function shall be explicitly used.
 *
 * In case of COERCION_METHOD_INOUT, expression tree have CoerceViaIO; that
 * involves a pair of heavy operation (cstring-out/in). Usually, it is not
 * supported on the device code except for small number of exceptions.
 * dcast_coerceviaio_callback allows to inject special case handling to run
 * the job of CoerceViaIO.
 */
static struct {
	Oid			src_type_oid;
	Oid			dst_type_oid;
	bool		has_domain_checks;
	devcast_coerceviaio_callback_f dcast_coerceviaio_callback;
} devcast_catalog[] = {
	/* text, varchar, bpchar */
	{ TEXTOID,    BPCHAROID,  false, NULL },
	{ TEXTOID,    VARCHAROID, false, NULL },
	{ VARCHAROID, TEXTOID,    false, NULL },
	{ VARCHAROID, BPCHAROID,  false, NULL },
	/* cidr -> inet, but no reverse type cast */
	{ CIDROID,    INETOID,    false, NULL },
	/* text -> (intX/floatX/numeric), including (jsonb->>'key') reference */
	{ TEXTOID,    BOOLOID,    false, devcast_text2numeric_callback },
	{ TEXTOID,    INT2OID,    false, devcast_text2numeric_callback },
	{ TEXTOID,    INT4OID,    false, devcast_text2numeric_callback },
	{ TEXTOID,    INT8OID,    false, devcast_text2numeric_callback },
	{ TEXTOID,    FLOAT4OID,  false, devcast_text2numeric_callback },
	{ TEXTOID,    FLOAT8OID,  false, devcast_text2numeric_callback },
	{ TEXTOID,    NUMERICOID, false, devcast_text2numeric_callback },
};

static devcast_info *
build_devcast_info(Oid src_type_oid, Oid dst_type_oid)
{
	devcast_info *dcast = NULL;
	devtype_info *dtype_s = NULL;
	devtype_info *dtype_d = NULL;
	int		i;

	dtype_s = pgstrom_devtype_lookup(src_type_oid);
	if (!dtype_s)
		goto not_found;
	dtype_d = pgstrom_devtype_lookup(dst_type_oid);
	if (!dtype_d)
		goto not_found;

	for (i=0; i < lengthof(devcast_catalog); i++)
	{
		if (dtype_s->type_oid == devcast_catalog[i].src_type_oid &&
			dtype_d->type_oid == devcast_catalog[i].dst_type_oid)
		{
			dcast = MemoryContextAllocZero(devinfo_memcxt,
										   sizeof(devcast_info));
			dcast->src_type = dtype_s;
			dcast->dst_type = dtype_d;
			dcast->has_domain_checks = devcast_catalog[i].has_domain_checks;
			dcast->dcast_coerceviaio_callback
				= devcast_catalog[i].dcast_coerceviaio_callback;
			break;
		}
	}
	/* extra type cast */
	if (!dcast)
	{
		StringInfoData	src_ident;
		StringInfoData	dst_ident;
		devcast_info	__dcast;

		initStringInfo(&src_ident);
		initStringInfo(&dst_ident);
		append_string_devtype_identifier(&src_ident, dtype_s->type_oid);
		append_string_devtype_identifier(&dst_ident, dtype_d->type_oid);

		memset(&__dcast, 0, sizeof(devcast_info));
		__dcast.src_type = dtype_s;
		__dcast.dst_type = dtype_d;
		__dcast.has_domain_checks = false;	/* extra module must set, if any */

		for (i=0; i < pgstrom_num_users_extra; i++)
		{
			pgstromUsersExtraDescriptor *extra = &pgstrom_users_extra_desc[i];

			if (extra->lookup_extra_devcast &&
				extra->lookup_extra_devcast(src_ident.data,
											dst_ident.data,
											&__dcast))
			{
				MemoryContext	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

				dcast = pmemdup(&__dcast, sizeof(devcast_info));

				MemoryContextSwitchTo(oldcxt);
				break;
			}
		}
		pfree(src_ident.data);
		pfree(dst_ident.data);
	}
not_found:
	/* negative entry */
	if (!dcast)
	{
		MemoryContext	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

		if (!dtype_s)
		{
			dtype_s = palloc0(sizeof(devtype_info));
            dtype_s->type_oid = src_type_oid;
		}
		if (!dtype_d)
		{
			dtype_d = palloc0(sizeof(devtype_info));
			dtype_d->type_oid = dst_type_oid;
		}
		dcast = palloc0(sizeof(devcast_info));
		dcast->src_type = dtype_s;
		dcast->dst_type = dtype_d;
		dcast->cast_is_negative = true;
		MemoryContextSwitchTo(oldcxt);
	}
	/* sanity checks */
	if (dcast->has_domain_checks &&
		dcast->dcast_coerceviaio_callback != NULL)
		__ELog("Bug? type cast %s -> %s with domain checks must be binary compatible",
			   format_type_be(dcast->src_type->type_oid),
			   format_type_be(dcast->dst_type->type_oid));
	return dcast;
}

devcast_info *
pgstrom_devcast_lookup(Oid src_type_oid, Oid dst_type_oid)
{
	uint32		hashvalue;
	int			hindex;
	devcast_info *dcast;
	dlist_iter	iter;

	hashvalue = GetSysCacheHashValue(CASTSOURCETARGET,
									 src_type_oid,
									 dst_type_oid,
									 0, 0);
	hindex = hashvalue % lengthof(devcast_info_slot);
	dlist_foreach (iter, &devcast_info_slot[hindex])
	{
		dcast = dlist_container(devcast_info, chain, iter.cur);
		if (dcast->src_type->type_oid == src_type_oid &&
			dcast->dst_type->type_oid == dst_type_oid)
		{
			if (dcast->cast_is_negative)
				return NULL;
			return dcast;
		}
	}
	/* create a new one */
	dcast = build_devcast_info(src_type_oid, dst_type_oid);
	dcast->hashvalue = hashvalue;
	dlist_push_head(&devcast_info_slot[hindex], &dcast->chain);
	if (dcast->cast_is_negative)
		return NULL;
	return dcast;
}

bool
pgstrom_devtype_can_relabel(Oid src_type_oid,
							Oid dst_type_oid)
{
	devcast_info *dcast;

	dcast = pgstrom_devcast_lookup(src_type_oid, dst_type_oid);
	if (dcast && dcast->dcast_coerceviaio_callback == NULL)
		return true;

	return false;
}

/*
 * Device index support
 *
 * devide index handler must be declared as:
 *
 * DEVICE_FUNCTION(cl_bool)
 * pgindex_<dev_index_fname>(kern_context *cxt,
 *                           PageHeaderData *i_page,
 *                           <left device type> arg1,
 *                           <right device type> arg2);
 */
static struct {
	const char	   *extname;
	const char	   *signature;
	const char	   *index_kind;
	int				opstrategy;
	const char	   *index_fname;
	const char	   *ivar_typname;
	const char	   *iarg_typname;
} devindex_catalog[] = {
	/* geometry overlap operator */
	{ POSTGIS3, "geometry && geometry",
	  "gist", RTOverlapStrategyNumber,
	  "gist_geometry_overlap",
	  "box2df@postgis",
	  "geometry@postgis",
	},
	{ POSTGIS3, "box2df && geometry",
	  "gist", RTOverlapStrategyNumber,
	  "gist_geometry_overlap",
	  "box2df@postgis",
	  "geometry@postgis",
	},
	{ POSTGIS3, "geometry && box2df",
	  "gist", RTOverlapStrategyNumber,
	  "gist_box2df_overlap",
	  "box2df@postgis",
	  "box2df@postgis",
	},
	{ POSTGIS3, "box2df && box2df",
	  "gist", RTOverlapStrategyNumber,
	  "gist_box2df_overlap",
	  "box2df@postgis",
	  "box2df@postgis",
	},
	/* geometry contains operator */
	{ POSTGIS3, "geometry ~ geometry",
	  "gist", RTContainsStrategyNumber,
	  "gist_geometry_contains",
	  "box2df@postgis",
	  "geometry@postgis",
	},
	{ POSTGIS3, "box2df ~ geometry",
	  "gist", RTContainsStrategyNumber,
	  "gist_geometry_contains",
	  "box2df@postgis",
	  "geometry@postgis",
	},
	{ POSTGIS3, "geometry ~ box2df",
	  "gist", RTContainsStrategyNumber,
	  "gist_box2df_contains",
	  "box2df@postgis",
	  "box2df@postgis",
	},
	{ POSTGIS3, "box2df ~ box2df",
	  "gist", RTContainsStrategyNumber,
	  "gist_box2df_contains",
	  "box2df@postgis",
	  "box2df@postgis",
	},
	/* geometry contained operator */
	{ POSTGIS3, "geometry @ geometry",
	  "gist", RTContainedByStrategyNumber,
	  "gist_geometry_contained",
	  "box2df@postgis",
	  "geometry@postgis",
	},
	{ POSTGIS3, "box2df @ geometry",
	  "gist", RTContainedByStrategyNumber,
	  "gist_geometry_contained",
	  "box2df@postgis",
	  "geometry@postgis",
	},
	{ POSTGIS3, "geometry @ box2df",
	  "gist", RTContainedByStrategyNumber,
	  "gist_box2df_contained",
     "box2df@postgis",
     "box2df@postgis",
	},
	{ POSTGIS3, "box2df @ box2df",
	  "gist", RTContainedByStrategyNumber,
	  "gist_box2df_contained",
	  "box2df@postgis",
	  "box2df@postgis",
	},
};

devindex_info *
pgstrom_devindex_lookup(Oid opcode, Oid opfamily)
{
	devindex_info  *dindex = NULL;
	uint32			hashvalue;
	uint32			hindex;
	HeapTuple		htup;
	Form_pg_amop	amop;
	dlist_iter		iter;
	const char	   *extname;
	char			signature[3*NAMEDATALEN + 100];
	int				i;

	hashvalue = GetSysCacheHashValue(AMOPOPID,
									 ObjectIdGetDatum(opcode),
									 CharGetDatum(AMOP_SEARCH),
									 ObjectIdGetDatum(opfamily), 0);
	hindex = hashvalue % lengthof(devindex_info_slot);
	dlist_foreach(iter, &devindex_info_slot[hindex])
	{
		dindex = dlist_container(devindex_info, chain, iter.cur);
		if (dindex->opcode == opcode &&
			dindex->opfamily == opfamily)
			goto found;
	}

	extname = get_extension_name_by_object(OperatorRelationId, opcode);
	htup = SearchSysCache3(AMOPOPID,
						   ObjectIdGetDatum(opcode),
						   CharGetDatum(AMOP_SEARCH),
						   ObjectIdGetDatum(opfamily));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "operator %u is not a member of opfamily %u",
			 opcode, opfamily);
	amop = (Form_pg_amop) GETSTRUCT(htup);
	snprintf(signature, sizeof(signature), "%s %s %s",
			 get_type_name(amop->amoplefttype, false),
			 get_opname(opcode),
			 get_type_name(amop->amoprighttype, false));

	dindex = NULL;
	for (i=0; i < lengthof(devindex_catalog); i++)
	{
		const char *__extname = devindex_catalog[i].extname;
		const char *__signature = devindex_catalog[i].signature;
		const char *__ivar_typname = devindex_catalog[i].ivar_typname;
		const char *__iarg_typname = devindex_catalog[i].iarg_typname;
		devtype_info   *ivar_dtype;
		devtype_info   *iarg_dtype;

		if (__extname)
		{
			if (!extname || strcmp(__extname, extname) != 0)
				continue;
		}
		else if (extname != NULL)
			continue;

		if (strcmp(__signature, signature) != 0)
			continue;

		ivar_dtype = pgstrom_devtype_lookup_by_name(__ivar_typname);
		if (!ivar_dtype)
			continue;
		iarg_dtype = pgstrom_devtype_lookup_by_name(__iarg_typname);
		if (!iarg_dtype)
			continue;

		dindex = MemoryContextAllocZero(devinfo_memcxt, sizeof(devindex_info));
		dindex->oper_extension = extname;
		dindex->opcode = opcode;
		dindex->opfamily = opfamily;
		dindex->opstrategy = amop->amopstrategy;
		dindex->index_kind = devindex_catalog[i].index_kind;
		dindex->index_fname = devindex_catalog[i].index_fname;
		dindex->ivar_dtype = ivar_dtype;
		dindex->iarg_dtype = iarg_dtype;
		break;
	}

	//TODO: call extra module

	/* not supported, add negative entry */
	if (!dindex)
	{
		dindex = MemoryContextAllocZero(devinfo_memcxt, sizeof(devindex_info));
		dindex->oper_extension = extname;
		dindex->opcode = opcode;
		dindex->opfamily = opfamily;
		dindex->opstrategy = amop->amopstrategy;
		dindex->index_is_negative = true;
	}
	ReleaseSysCache(htup);

	dindex->hashvalue = hashvalue;
	dlist_push_head(&devindex_info_slot[hindex], &dindex->chain);
found:
	if (dindex->index_is_negative)
		return NULL;
	return dindex;
}

/*
 * codegen_expression_walker - main logic of run-time code generator
 */
static void codegen_expression_walker(codegen_context *context,
									  StringInfo body,
									  Node *node, int *p_varlena_sz);

static Node *__codegen_current_node = NULL;
static void
__appendStringInfo(StringInfo str, const char *fmt,...)
	pg_attribute_printf(2, 3);

static void
__appendStringInfo(StringInfo str, const char *fmt,...)
{
	int		save_errno = errno;

	if (!str)
		return;
	for (;;)
	{
		va_list		va_args;
		int			needed;

		errno = save_errno;
		va_start(va_args, fmt);
		needed = appendStringInfoVA(str, fmt, va_args);
		va_end(va_args);

		if (needed == 0)
			break;
		enlargeStringInfo(str, needed);
	}
}

static inline void
__appendStringInfoChar(StringInfo str, char c)
{
	if (str)
		appendStringInfoChar(str, c);
}

static int
codegen_const_expression(codegen_context *context,
						 StringInfo body,
						 Const *con)
{
	devtype_info *dtype;
	cl_int		index;
	cl_int		width;

	dtype = pgstrom_devtype_lookup_and_track(con->consttype, context);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(con->consttype));
	context->used_params = lappend(context->used_params,
								   copyObject(con));
	index = list_length(context->used_params) - 1;

	__appendStringInfo(body,
					   "pg_%s_param(kcxt,%d)",
					   dtype->type_name, index);
	if (con->constisnull)
		width = 0;
	else if (con->constlen > 0)
		width = con->constlen;
	else if (con->constlen == -1)
		width = VARSIZE_ANY_EXHDR(con->constvalue);
	else
		elog(ERROR, "unexpected type length: %d", con->constlen);
	return width;
}

static int
codegen_param_expression(codegen_context *context,
						 StringInfo body,
						 Param *param)
{
	devtype_info   *dtype;
	ListCell	   *lc;
	int				index = 0;
	int				width;

	if (param->paramkind != PARAM_EXTERN)
		__ELog("ParamKind is not PARAM_EXTERN: %d",
			   (int)param->paramkind);

	dtype = pgstrom_devtype_lookup_and_track(param->paramtype, context);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(param->paramtype));

	foreach (lc, context->used_params)
	{
		if (equal(param, lfirst(lc)))
			goto found;
		index++;
	}
	context->used_params = lappend(context->used_params,
								   copyObject(param));
	index = list_length(context->used_params) - 1;

found:
	__appendStringInfo(body,
					   "pg_%s_param(kcxt,%d)",
					   dtype->type_name, index);
	if (dtype->type_length > 0)
		width = dtype->type_length;
	else if (dtype->type_length == -1)
		width = type_maximum_size(param->paramtype,
								  param->paramtypmod) - VARHDRSZ;
	else
		elog(ERROR, "unexpected type length: %d", dtype->type_length);

	return width;
}

static int
codegen_varnode_expression(codegen_context *context,
						   StringInfo body, Var *var)
{
	AttrNumber		varattno = var->varattno;
	devtype_info   *dtype;
	ListCell	   *lc;
	int				width;

	dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(var->vartype));
	/*
	 * NOTE: Expression tree at the path-construction time can contain
	 * references to other tables; which can be eventually replaced by 
	 * replace_nestloop_params(). So, this Var-node shall not be visible
	 * when we generate the device code.
	 * We may be able to handle the check well, however, we simply
	 * prohibit the Var-node which references out of the current scope
	 * of the relations.
	 *
	 * If var->varno == INDEX_VAR, it is obvious that caller is
	 * responsible to build custom_scan_tlist with adequate source.
	 */
	if (context->baserel && !IS_SPECIAL_VARNO(var->varno))
	{
		RelOptInfo *baserel = context->baserel;

		if (!bms_is_member(var->varno, baserel->relids))
			elog(ERROR, "Var (varno=%d) referred out of expected range %s",
				 var->varno, bms_to_cstring(baserel->relids));
	}

	/*
	 * Fixup varattno when pseudo-scan tlist exists, because varattno
	 * shall be adjusted on setrefs.c, so we have to adjust variable
	 * name according to the expected attribute number is kernel-
	 * source shall be constructed prior to setrefs.c / subselect.c
	 */
	if (context->pseudo_tlist != NIL)
	{
		foreach (lc, context->pseudo_tlist)
		{
			TargetEntry *tle = lfirst(lc);
			Var			*ptv = (Var *) tle->expr;

			if (!IsA(tle->expr, Var) ||
				ptv->varno != var->varno ||
				ptv->varattno != var->varattno ||
				ptv->varlevelsup != var->varlevelsup)
				continue;

			varattno = tle->resno;
			break;
		}
		if (!lc)
			elog(ERROR, "failed on map Var (%s) on ps_tlist: %s",
				 nodeToString(var),
				 nodeToString(context->pseudo_tlist));
	}
	if (varattno < 0)
		__appendStringInfo(body, "KVAR_S%u", -varattno);
	else
		__appendStringInfo(body, "KVAR_%u",   varattno);
	if (!list_member(context->used_vars, var))
		context->used_vars = lappend(context->used_vars,
									 copyObject(var));
	if (dtype->type_length >= 0)
		width = dtype->type_length;
	else
		width = type_maximum_size(var->vartype,
								  var->vartypmod) - VARHDRSZ;
	return width;
}

static int
codegen_function_expression(codegen_context *context,
							StringInfo body,
							devfunc_info *dfunc, List *args)
{
	ListCell *lc1, *lc2;
	Expr  **fn_args = alloca(sizeof(Expr *) * list_length(args));
	int	   *vl_width = alloca(sizeof(int) * list_length(args));
	int		index = 0;

	__appendStringInfo(body,
					   "pgfn_%s(kcxt",
					   dfunc->func_devname);
	forboth (lc1, dfunc->func_args,
			 lc2, args)
	{
		devtype_info *dtype = lfirst(lc1);
		Node   *expr = lfirst(lc2);
		Oid		expr_type_oid = exprType(expr);

		__appendStringInfo(body, ", ");

		if (dtype->type_oid == expr_type_oid)
			codegen_expression_walker(context, body, expr, &vl_width[index]);
		else if (pgstrom_devtype_can_relabel(expr_type_oid,
											 dtype->type_oid))
		{
			/*
			 * NOTE: PostgreSQL may pass binary compatible arguments
			 * without explicit RelabelType, like varchar(N) values
			 * onto text arguments.
			 * It is quite right implementation from the PostgreSQL
			 * function invocation API, however, unable to describe
			 * the relevant device code, because CUDA C++ has strict
			 * type checks. So, we have to inject an explicit type
			 * relabel in this case.
			 */
			__appendStringInfo(body, "to_%s(", dtype->type_name);
			codegen_expression_walker(context, body, expr, &vl_width[index]);
			__appendStringInfoChar(body, ')');
		}
		else
		{
			__ELog("Bug? unsupported implicit type cast (%s)->(%s)",
				   format_type_be(expr_type_oid),
				   format_type_be(dtype->type_oid));
		}
		fn_args[index++] = (Expr *)expr;
	}
	__appendStringInfoChar(body, ')');
	/* estimation of function result width */
	return dfunc->devfunc_result_sz(context, dfunc, fn_args, vl_width);
}

static int
codegen_nulltest_expression(codegen_context *context,
							StringInfo body,
							NullTest *nulltest)
{
	devtype_info *dtype;
	Oid		typeoid = exprType((Node *)nulltest->arg);

	if (nulltest->argisrow)
		__ELog("NullTest towards RECORD data");

	dtype = pgstrom_devtype_lookup_and_track(typeoid, context);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(typeoid));
	switch (nulltest->nulltesttype)
	{
		case IS_NULL:
			__appendStringInfo(body, "PG_ISNULL");
			break;
		case IS_NOT_NULL:
			__appendStringInfo(body, "PG_ISNOTNULL");
			break;
		default:
			elog(ERROR, "unknown NullTestType: %d",
				 (int)nulltest->nulltesttype);
	}
	__appendStringInfo(body, "(kcxt, ");
	codegen_expression_walker(context, body, (Node *) nulltest->arg, NULL);
	__appendStringInfoChar(body, ')');
	context->devcost += 1;

	return sizeof(cl_bool);
}

static int
codegen_booleantest_expression(codegen_context *context,
							   StringInfo body,
							   BooleanTest *booltest)
{
	const char	   *func_name;

	if (exprType((Node *)booltest->arg) != BOOLOID)
		elog(ERROR, "argument type of BooleanTest is not bool");

	/* choose one of built-in functions */
	switch (booltest->booltesttype)
	{
		case IS_TRUE:
			func_name = "bool_is_true";
			break;
		case IS_NOT_TRUE:
			func_name = "bool_is_not_true";
			break;
		case IS_FALSE:
			func_name = "bool_is_false";
			break;
		case IS_NOT_FALSE:
			func_name = "bool_is_not_false";
			break;
		case IS_UNKNOWN:
			func_name = "bool_is_unknown";
			break;
		case IS_NOT_UNKNOWN:
			func_name = "bool_is_not_unknown";
			break;
		default:
			elog(ERROR, "unknown BoolTestType: %d",
				 (int)booltest->booltesttype);
			break;
	}
	__appendStringInfo(body, "pgfn_%s(kcxt, ", func_name);
	codegen_expression_walker(context, body,
							  (Node *) booltest->arg, NULL);
	__appendStringInfoChar(body, ')');
	context->devcost += 1;

	return sizeof(cl_bool);
}

static int
codegen_bool_expression(codegen_context *context,
						StringInfo body, BoolExpr *b)
{
	Node	   *node;

	if (b->boolop == NOT_EXPR)
	{
		Assert(list_length(b->args) == 1);
		node = linitial(b->args);

		__appendStringInfo(body, "NOT(");
		codegen_expression_walker(context, body, node, NULL);
		__appendStringInfoChar(body, ')');
	}
	else if (b->boolop == AND_EXPR ||
			 b->boolop == OR_EXPR)
	{
		StringInfoData temp;
		List	   *used_vars_saved;
		ListCell   *lc;

		initStringInfo(&temp);

		used_vars_saved = context->used_vars;
		context->used_vars = NIL;
		foreach (lc, b->args)
		{
			Node	   *node = lfirst(lc);

			if (lc != list_head(b->args))
				__appendStringInfo(&temp, "  has_null |= status.isnull;\n");
			__appendStringInfo(&temp,
							   "  status = ");
			codegen_expression_walker(context, &temp, node, NULL);
			__appendStringInfo(&temp, ";\n"
							   "  if (PG_BOOL_%s(status))\n"
							   "    return status;\n",
							   (b->boolop == AND_EXPR ? "ISFALSE" : "ISTRUE"));
		}
		context->decl_count++;
		__appendStringInfo(
			&context->decl,
			"DEVICE_INLINE(pg_bool_t)\n"
			"__exprBoolOp_%u(kern_context *kcxt",
			context->decl_count);
		__appendStringInfo(body,
						   "__exprBoolOp_%u(kcxt", context->decl_count);
		foreach (lc, context->used_vars)
		{
			devtype_info *dtype;
			Var	   *var = lfirst(lc);

			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				__ELog("type %s is not device supported",
					   format_type_be(var->vartype));
			__appendStringInfo(
				&context->decl,
				", pg_%s_t &",
				dtype->type_name);
			codegen_expression_walker(context,
									  &context->decl,
									  (Node *)var, NULL);
			__appendStringInfo(body, ", ");
			codegen_expression_walker(context, body, (Node *)var, NULL);

			if (!list_member(used_vars_saved, var))
				used_vars_saved = lappend(used_vars_saved, var);
		}
		__appendStringInfo(
			&context->decl,
			")\n"
			"{\n"
			"  pg_bool_t status __attribute__((unused));\n"
			"  cl_bool   has_null = false;\n"
			"\n"
			"%s"
			"  status.isnull |= has_null;\n"
			"  return status;\n"
			"}\n\n",
			temp.data);
		__appendStringInfo(body, ")");
		context->used_vars = used_vars_saved;

		pfree(temp.data);
	}
	else
	{
		elog(ERROR, "unknown BoolExprType: %d", (int) b->boolop);
	}
	context->devcost += list_length(b->args);
	return sizeof(cl_bool);
}

static int
codegen_coalesce_expression(codegen_context *context,
							StringInfo body,
							CoalesceExpr *coalesce)
{
	devtype_info   *dtype;
	StringInfoData	temp;
	List		   *used_vars_saved;
	ListCell	   *lc;
	int				maxlen = 0;

	initStringInfo(&temp);
	dtype = pgstrom_devtype_lookup(coalesce->coalescetype);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(coalesce->coalescetype));

	used_vars_saved = context->used_vars;
	context->used_vars = NIL;
	foreach (lc, coalesce->args)
	{
		Node   *expr = lfirst(lc);
		Oid		type_oid = exprType(expr);
		int		width;

		if (dtype->type_oid != type_oid)
			__ELog("device type mismatch in COALESCE: %s / %s",
				   format_type_be(dtype->type_oid),
				   format_type_be(type_oid));
		__appendStringInfo(&temp,
						   "  retval = ");
		codegen_expression_walker(context, &temp, expr, &width);
		__appendStringInfo(&temp,
						   ";\n"
						   "  if (!retval.isnull)\n"
						   "    return retval;\n");
		if (width < 0)
			maxlen = -1;
		else if (maxlen >= 0)
			maxlen = Max(maxlen, width);
		context->devcost += 1;
	}

	context->decl_count++;
	__appendStringInfo(
		&context->decl,
		"DEVICE_INLINE(pg_%s_t)\n"
		"__exprCoalesce_%u(kern_context *kcxt",
		dtype->type_name,
		context->decl_count);
	__appendStringInfo(
		body,
		"__exprCoalesce_%u(kcxt",
		context->decl_count);

	foreach (lc, context->used_vars)
	{
		devtype_info *__dtype;
		Var	   *var = lfirst(lc);

		__dtype = pgstrom_devtype_lookup(var->vartype);
		if (!__dtype)
			__ELog("type %s is not device supported",
				   format_type_be(var->vartype));
		__appendStringInfo(
			&context->decl,
			", pg_%s_t &",
			__dtype->type_name);
		codegen_expression_walker(context,
								  &context->decl,
								  (Node *)var, NULL);
		__appendStringInfo(body, ", ");
		codegen_expression_walker(context, body, (Node *)var, NULL);

		if (!list_member(used_vars_saved, var))
			used_vars_saved = lappend(used_vars_saved, var);
	}
	__appendStringInfo(
		&context->decl,
		")\n"
		"{\n"
		"  pg_%s_t retval __attribute__((unused));\n"
		"\n"
		"  retval.isnull = true;\n"
		"%s"
		"  return retval;\n"
		"}\n\n",
		dtype->type_name,
		temp.data);
	__appendStringInfo(body, ")");
	context->used_vars = used_vars_saved;

	pfree(temp.data);

	return maxlen;
}

static int
codegen_minmax_expression(codegen_context *context,
						  StringInfo body,
						  MinMaxExpr *minmax)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	List		   *used_vars_saved;
	ListCell	   *lc;
	StringInfoData	temp;
	int				maxlen = 0;

	dtype = pgstrom_devtype_lookup(minmax->minmaxtype);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(minmax->minmaxtype));
	context->extra_flags |= dtype->type_flags;

	dfunc = pgstrom_devfunc_lookup_type_compare(dtype, minmax->inputcollid);
	if (!dfunc)
		__ELog("device type %s has no comparison operator",
			   format_type_be(minmax->minmaxtype));
	context->extra_flags |= dfunc->func_flags;
	
	initStringInfo(&temp);
	used_vars_saved = context->used_vars;
	context->used_vars = NIL;
	foreach (lc, minmax->args)
	{
		Node   *expr = lfirst(lc);
		Oid		type_oid = exprType(expr);
		int		width;

		if (dtype->type_oid != type_oid)
			__ELog("device type mismatch in LEAST/GREATEST: %s / %s",
				   format_type_be(dtype->type_oid),
				   format_type_be(exprType(expr)));
		if (lc == list_head(minmax->args))
			__appendStringInfo(&temp, "  r = ");
		else
			__appendStringInfo(&temp, "  x = ");
		codegen_expression_walker(context, &temp, expr, &width);
		__appendStringInfo(&temp, ";\n");

		if (lc != list_head(minmax->args))
		{
			__appendStringInfo(
				&temp,
				"  if (r.isnull)\n"
				"    r = x;\n"
				"  else if (!x.isnull && PG_%s_THAN(pgfn_%s(kcxt, x, r)))\n"
				"    r = x;\n",
				minmax->op == IS_GREATEST ? "GREATER" : "LESS",
				dfunc->func_devname);
		}
		if (width < 0)
			maxlen = -1;
		else if (maxlen >= 0)
			maxlen = Max(maxlen, width);
		context->devcost += 1;
	}

	context->decl_count++;
    __appendStringInfo(
		&context->decl,
		"DEVICE_INLINE(pg_%s_t)\n"
        "__exprMinMax_%u(kern_context *kcxt",
        dtype->type_name,
        context->decl_count);
    __appendStringInfo(
        body,
        "__exprMinMax_%u(kcxt",
        context->decl_count);

    foreach (lc, context->used_vars)
    {
        devtype_info *__dtype;
        Var    *var = lfirst(lc);

        __dtype = pgstrom_devtype_lookup(var->vartype);
        if (!__dtype)
            __ELog("type %s is not device supported",
                   format_type_be(var->vartype));
        __appendStringInfo(
            &context->decl,
            ", pg_%s_t &",
            __dtype->type_name);
		codegen_expression_walker(context,
								  &context->decl,
								  (Node *)var, NULL);
        __appendStringInfo(body, ", ");
        codegen_expression_walker(context, body, (Node *)var, NULL);

        if (!list_member(used_vars_saved, var))
            used_vars_saved = lappend(used_vars_saved, var);
    }
    __appendStringInfo(
        &context->decl,
        ")\n"
        "{\n"
        "  pg_%s_t r, x __attribute__((unused));\n"
		"  pg_int4_t cmp __attribute__((unused));\n"
        "\n"
        "%s"
        "  return r;\n"
        "}\n\n",
		dtype->type_name,
		temp.data);
	__appendStringInfo(body, ")");
	context->used_vars = used_vars_saved;

	pfree(temp.data);

	return maxlen;
}

static int
codegen_relabel_expression(codegen_context *context,
						   StringInfo body,
						   RelabelType *relabel)
{
	devtype_info *dtype;
	Oid		stype_oid = exprType((Node *)relabel->arg);
	int		width;

	dtype = pgstrom_devtype_lookup_and_track(stype_oid, context);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(stype_oid));

	dtype = pgstrom_devtype_lookup_and_track(relabel->resulttype, context);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(relabel->resulttype));
	if (!pgstrom_devtype_can_relabel(stype_oid, dtype->type_oid))
		__ELog("type %s->%s cannot be relabeled on device",
			   format_type_be(stype_oid),
			   format_type_be(relabel->resulttype));

	__appendStringInfo(body, "to_%s(", dtype->type_name);
	codegen_expression_walker(context, body, (Node *)relabel->arg, &width);
	__appendStringInfoChar(body, ')');

	return width;
}

static int
codegen_coerceviaio_expression(codegen_context *context,
							   StringInfo body,
							   CoerceViaIO *coerce)
{
	devcast_info   *dcast;
	Oid		stype_oid = exprType((Node *)coerce->arg);
	Oid		dtype_oid = coerce->resulttype;

	dcast = pgstrom_devcast_lookup(stype_oid, dtype_oid);
	if (!dcast || dcast->dcast_coerceviaio_callback == NULL)
		__ELog("no device support of coerceviaio (%s -> %s)",
			   format_type_be(stype_oid),
			   format_type_be(dtype_oid));
	context->devcost += 8;		/* just a rough estimation */

	return dcast->dcast_coerceviaio_callback(context, body, dcast, coerce);
}

static int
codegen_coercetodomain_expression(codegen_context *context,
								  StringInfo body,
								  CoerceToDomain *coerce_d)
{
	devcast_info *dcast;
	Oid		stype_oid = exprType((Node *)coerce_d->arg);
	Oid		dtype_oid = coerce_d->resulttype;
	int		width;

	dcast = pgstrom_devcast_lookup(stype_oid, dtype_oid);
	if (!dcast || dcast->dcast_coerceviaio_callback != NULL)
		__ELog("type cast (%s -> %s) is not binary compatible",
			   format_type_be(stype_oid),
			   format_type_be(dtype_oid));
	if (!dcast->has_domain_checks)
		__ELog("type cast (%s -> %s) has no domain constraint",
			   format_type_be(stype_oid),
			   format_type_be(dtype_oid));
	__appendStringInfo(body, "to_%s_domain(kcxt,",
					   dcast->dst_type->type_name);
	codegen_expression_walker(context, body, (Node *)coerce_d->arg, &width);
	__appendStringInfoChar(body, ')');

	return width;
}

static int
codegen_casewhen_expression(codegen_context *context,
							StringInfo body,
							CaseExpr *caseexpr)
{
	devtype_info   *rtype;	/* result type */
	devtype_info   *dtype;
	StringInfoData	temp;
	Node		   *defresult;
	List		   *used_vars_saved;
	ListCell	   *lc;
	Oid				type_oid;
	int				width, maxlen = 0;

	/* check result type */
	rtype = pgstrom_devtype_lookup(caseexpr->casetype);
	if (!rtype)
		__ELog("type %s is not device supported",
			   format_type_be(caseexpr->casetype));
	if (caseexpr->defresult)
		defresult = (Node *)caseexpr->defresult;
	else
	{
		defresult = (Node *)makeConst(rtype->type_oid,
									  -1,
									  InvalidOid,
									  rtype->type_length,
									  0UL,
									  true,     /* NULL */
									  rtype->type_byval);
	}

	initStringInfo(&temp);
	used_vars_saved = context->used_vars;
	context->used_vars = NIL;
	if (caseexpr->arg)
	{
		/* type compare function internally used */
		type_oid = exprType((Node *) caseexpr->arg);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			__ELog("type %s is not device supported",
				   format_type_be(type_oid));

		__appendStringInfo(&temp, "  pg_%s_t CARG = ", dtype->type_name);
		codegen_expression_walker(context, &temp,
								  (Node *)caseexpr->arg, NULL);
		__appendStringInfo(&temp, ";\n\n");
	}

	foreach (lc, caseexpr->args)
	{
		CaseWhen   *casewhen = (CaseWhen *) lfirst(lc);
		Expr	   *expr = casewhen->expr;

		Assert(IsA(casewhen, CaseWhen) &&
			   exprType((Node *)expr) == BOOLOID &&
			   exprType((Node *)casewhen->result) == rtype->type_oid);
		__appendStringInfo(&temp, "  if (EVAL(");
		codegen_expression_walker(context, &temp, (Node *)expr, NULL);
		__appendStringInfo(&temp, "))\n" "    return ");
		codegen_expression_walker(context, &temp,
								  (Node *)casewhen->result,
								  &width);
		__appendStringInfo(&temp, ";\n");
		if (width < 0)
			maxlen = -1;
		else if (maxlen >= 0)
			maxlen = Max(maxlen, width);
		context->devcost += 1;
	}
	__appendStringInfo(&temp, "  return ");
	codegen_expression_walker(context, &temp, defresult, NULL);
	__appendStringInfo(&temp, ";\n");

	context->decl_count++;
	__appendStringInfo(
		&context->decl,
		"DEVICE_INLINE(pg_%s_t)\n"
		"__exprCaseWhen_%u(kern_context *kcxt",
		rtype->type_name,
		context->decl_count);
	__appendStringInfo(
		body,
		"__exprCaseWhen_%u(kcxt",
		context->decl_count);

	foreach (lc, context->used_vars)
	{
		devtype_info *__dtype;
		Var    *var = lfirst(lc);

		__dtype = pgstrom_devtype_lookup(var->vartype);
		if (!__dtype)
			__ELog("type %s is not device supported",
				   format_type_be(var->vartype));
		__appendStringInfo(
			&context->decl,
			", pg_%s_t &",
			__dtype->type_name);
		codegen_expression_walker(context,
								  &context->decl,
								  (Node *)var, NULL);
		__appendStringInfo(body, ", ");
		codegen_expression_walker(context, body, (Node *)var, NULL);

		if (!list_member(used_vars_saved, var))
			used_vars_saved = lappend(used_vars_saved, var);
	}
	__appendStringInfo(
		&context->decl,
		")\n"
		"{\n"
        "%s"
        "}\n\n",
		temp.data);
	__appendStringInfo(body, ")");
	context->used_vars = used_vars_saved;

	pfree(temp.data);

	return maxlen;
}

static int
codegen_casetest_expression(codegen_context *context,
							StringInfo body,
							CaseTestExpr *ctest)
{
	__appendStringInfo(body, "CARG");
	return 0;
}

static int
codegen_scalar_array_op_expression(codegen_context *context,
								   StringInfo body,
								   ScalarArrayOpExpr *opexpr)
{
	devfunc_info *dfunc;
	devtype_info *dtype_s;
	devtype_info *dtype_a;
	devtype_info *dtype_e;
	Node	   *node_s;
	Node	   *node_a;
	HeapTuple	fn_tup;
	oidvector  *fn_argtypes = alloca(offsetof(oidvector, values[2]));

	Assert(list_length(opexpr->args) == 2);
	node_s = linitial(opexpr->args);
	node_a = lsecond(opexpr->args);
	dtype_s = pgstrom_devtype_lookup_and_track(exprType(node_s), context);
	if (!dtype_s)
		__ELog("type %s is not device supported",
			   format_type_be(exprType(node_s)));
	dtype_a = pgstrom_devtype_lookup_and_track(exprType(node_a), context);
	if (!dtype_a)
		__ELog("type %s is not device supported",
			   format_type_be(exprType(node_a)));
	dtype_e = dtype_a->type_element;
	if (!dtype_e)
		__ELog("type %s is not an array data type",
			   format_type_be(exprType(node_a)));

	/* lookup operator function */
	fn_tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(opexpr->opfuncid));
	if (!HeapTupleIsValid(fn_tup))
		elog(ERROR, "cache lookup failed for function %u", opexpr->opfuncid);
	PG_TRY();
	{
		memset(fn_argtypes, 0, offsetof(oidvector, values[2]));
		fn_argtypes->ndim = 1;
		fn_argtypes->dataoffset = 0;
		fn_argtypes->elemtype = OIDOID;
		fn_argtypes->dim1 = 2;
		fn_argtypes->lbound1 = 0;
		fn_argtypes->values[0] = dtype_s->type_oid;
		fn_argtypes->values[1] = dtype_e->type_oid;
		SET_VARSIZE(fn_argtypes, offsetof(oidvector, values[2]));

		dfunc = __pgstrom_devfunc_lookup(fn_tup,
										 BOOLOID,
										 fn_argtypes,
										 opexpr->inputcollid);
		if (!dfunc)
			__ELog("function %s is not device supported",
				   format_procedure(opexpr->opfuncid));
		pgstrom_devfunc_track(context, dfunc);
	}
	PG_CATCH();
	{
		ReleaseSysCache(fn_tup);
		PG_RE_THROW();
	}
	PG_END_TRY();
	ReleaseSysCache(fn_tup);

	__appendStringInfo(body,
					   "PG_SCALAR_ARRAY_OP(kcxt, pgfn_%s, ",
					   dfunc->func_devname);
	codegen_expression_walker(context, body, node_s, NULL);
	__appendStringInfo(body, ", ");
	codegen_expression_walker(context, body, node_a, NULL);
	__appendStringInfo(body, ", %s, %d, %d)",
					   opexpr->useOr ? "true" : "false",
					   dtype_e->type_length,
					   dtype_e->type_align);
	/*
	 * Cost for PG_SCALAR_ARRAY_OP - It repeats on number of invocation
	 * of the operator function for each array elements. Tentatively,
	 * we assume one array has 32 elements in average.
	 */
	context->devcost += 32 * dfunc->func_devcost;

	return sizeof(cl_bool);
}

static void
codegen_expression_walker(codegen_context *context,
						  StringInfo body,
						  Node *node, int *p_width)
{
	devfunc_info   *dfunc;
	int				width = 0;
	Node		   *__codegen_saved_node;

	if (node == NULL)
		return;
	/* save the current node for error message */
	__codegen_saved_node   = __codegen_current_node;
	__codegen_current_node = node;

	switch (nodeTag(node))
	{
		case T_Const:
			width = codegen_const_expression(context, body, (Const *) node);
			break;

		case T_Param:
			width = codegen_param_expression(context, body, (Param *) node);
			break;

		case T_Var:
			width = codegen_varnode_expression(context, body, (Var *) node);
			break;

		case T_FuncExpr:
			{
				FuncExpr   *func = (FuncExpr *) node;

				dfunc = pgstrom_devfunc_lookup(func->funcid,
											   func->funcresulttype,
											   func->args,
											   func->inputcollid);
				if (!dfunc)
					__ELog("function %s is not device supported",
						   format_procedure(func->funcid));
				pgstrom_devfunc_track(context, dfunc);
				width = codegen_function_expression(context,
													body,
													dfunc,
													func->args);
				context->devcost += dfunc->func_devcost;
			}
			break;

		case T_OpExpr:
		case T_DistinctExpr:
			{
				OpExpr	   *op = (OpExpr *) node;
				Oid			func_oid = get_opcode(op->opno);

				dfunc = pgstrom_devfunc_lookup(func_oid,
											   op->opresulttype,
											   op->args,
											   op->inputcollid);
				if (!dfunc)
					__ELog("function %s is not device supported",
						   format_procedure(func_oid));
				pgstrom_devfunc_track(context, dfunc);
				width = codegen_function_expression(context,
													body,
													dfunc,
													op->args);
				context->devcost += dfunc->func_devcost;
			}
			break;

		case T_NullTest:
			width = codegen_nulltest_expression(context,
												body,
												(NullTest *) node);
			break;

		case T_BooleanTest:
			width = codegen_booleantest_expression(context,
												   body,
												   (BooleanTest *) node);
			break;

		case T_BoolExpr:
			width = codegen_bool_expression(context,
											body,
											(BoolExpr *) node);
			break;

		case T_CoalesceExpr:
			width = codegen_coalesce_expression(context,
												body,
												(CoalesceExpr *) node);
			break;

		case T_MinMaxExpr:
			width = codegen_minmax_expression(context,
											  body,
											  (MinMaxExpr *) node);
			break;

		case T_RelabelType:
			width = codegen_relabel_expression(context,
											   body,
											   (RelabelType *) node);
			break;

		case T_CoerceViaIO:
			width = codegen_coerceviaio_expression(context,
												   body,
												   (CoerceViaIO *) node);
			break;

		case T_CoerceToDomain:
			width = codegen_coercetodomain_expression(context,
													  body,
													  (CoerceToDomain *) node);
			break;

		case T_CaseExpr:
			width = codegen_casewhen_expression(context,
												body,
												(CaseExpr *) node);
			break;

		case T_CaseTestExpr:
			width = codegen_casetest_expression(context,
												body,
												(CaseTestExpr *) node);
			break;

		case T_ScalarArrayOpExpr:
			width = codegen_scalar_array_op_expression(context,
													   body,
												(ScalarArrayOpExpr *) node);
			break;
		default:
			__ELog("Bug? unsupported expression: %s", nodeToString(node));
			break;
	}
	if (p_width)
		*p_width = width;
	/* restore */
	__codegen_current_node = __codegen_saved_node;
}

char *
pgstrom_codegen_expression(Node *expr, codegen_context *context)
{
	StringInfoData	body;
	devtype_info   *dtype;

	initStringInfo(&body);
	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = (Node *)linitial((List *)expr);
		else
			expr = (Node *)make_andclause((List *)expr);
	}

	PG_TRY();
	{
		codegen_expression_walker(context, &body, expr, NULL);
	}
	PG_CATCH();
	{
		errdetail("problematic expression: %s", nodeToString(expr));
		PG_RE_THROW();
	}
	PG_END_TRY();

	/*
	 * Even if expression itself needs no varlena extra buffer, projection
	 * code may require the buffer to construct a temporary datum.
	 * E.g) Numeric datum is encoded to 128bit at the GPU kernel, however,
	 * projection needs to decode to varlena again.
	 */
	dtype = pgstrom_devtype_lookup(exprType((Node *) expr));
	if (dtype)
		context->extra_bufsz += MAXALIGN(dtype->extra_sz);

	return body.data;
}

/*
 * pgstrom_union_type_declarations
 *
 * put declaration of a union type which contains all the types in type_oid_list,
 * as follows. OID of device types should be unique, must not duplicated.
 *
 *   union {
 *     pg_bool_t   bool_v;
 *     pg_text_t   text_v;
 *        :
 *   } NAME;
 */
void
pgstrom_union_type_declarations(StringInfo buf,
								const char *name,
								List *type_oid_list)
{
	ListCell	   *lc;
	devtype_info   *dtype;
	bool			meet_array_v = false;

	if (type_oid_list == NIL)
		return;
	appendStringInfo(buf, "  union {\n");
	foreach (lc, type_oid_list)
	{
		Oid		type_oid = lfirst_oid(lc);

		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			__ELog("failed to lookup device type: %u", type_oid);
		/*
		 * All the array types have same device type name (pg_array_t)
		 * regardless of the element type. So, we have to avoid duplication
		 * of the field name in union, by special handling.
		 */
		if (dtype->type_element)
		{
			if (meet_array_v)
				continue;
			meet_array_v = true;
		}
		appendStringInfo(buf,
						 "    pg_%s_t %s_v;\n",
						 dtype->type_name,
						 dtype->type_name);
	}
	appendStringInfo(buf, "  } %s __attribute__((unused));\n", name);
}

/*
 * __pgstrom_device_expression
 *
 * It shows a quick decision whether the provided expression tree is
 * available to run on CUDA device, or not.
 */
bool
__pgstrom_device_expression(PlannerInfo *root,
							RelOptInfo *baserel,
							Expr *expr,
							int *p_devcost, int *p_extra_sz,
							const char *filename, int lineno)
{
	MemoryContext memcxt = CurrentMemoryContext;
	codegen_context con;
	int			dummy = 0;
	bool		result = true;

	if (!expr)
		return false;
	pgstrom_init_codegen_context(&con, root, baserel);
	PG_TRY();
	{
		if (IsA(expr, List))
		{
			List	   *exprsList = (List *)expr;
			ListCell   *lc;

			foreach (lc, exprsList)
			{
				Node   *node = (Node *)lfirst(lc);

				codegen_expression_walker(&con, NULL, node, &dummy);
			}
		}
		else
		{
			codegen_expression_walker(&con, NULL, (Node *)expr, &dummy);
		}
	}
	PG_CATCH();
	{
		ErrorData	   *edata;

		MemoryContextSwitchTo(memcxt);
		edata = CopyErrorData();
		if (edata->sqlerrcode != ERRCODE_FEATURE_NOT_SUPPORTED)
			PG_RE_THROW();

		FlushErrorState();

		ereport(DEBUG2,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s:%d %s, at %s:%d",
						filename, lineno,
						edata->message,
						edata->filename, edata->lineno),
				 errdetail("expression: %s",
						   nodeToString(__codegen_current_node))));
		__codegen_current_node = NULL;
		FreeErrorData(edata);
		result = false;
	}
	PG_END_TRY();

	if (result)
	{
		if (con.extra_bufsz > KERN_CONTEXT_VARLENA_BUFSZ_LIMIT)
		{
			elog(DEBUG2, "Expression consumes too much buffer (%u): %s",
				 con.extra_bufsz, nodeToString(expr));
			return false;
		}
		Assert(con.devcost >= 0);
		if (p_devcost)
			*p_devcost = con.devcost;
		if (p_extra_sz)
			*p_extra_sz = con.extra_bufsz;
	}
	return result;
}

/*
 * devcast_text2numeric_callback
 * ------
 * Special case handling of text->numeric values, including the case of
 * jsonb key references.
 */
static int
devcast_text2numeric_callback(codegen_context *context,
							  StringInfo body,
							  devcast_info *dcast,
							  CoerceViaIO *node)
{
	devtype_info   *dtype = dcast->dst_type;
	Expr		   *arg = node->arg;
	Oid				func_oid = InvalidOid;
	List		   *func_args = NIL;
	char			dfunc_name[100];
	int				width;
	ListCell	   *lc;

	/* check special case if jsonb key reference */
	if (IsA(arg, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *)arg;

		func_oid  = func->funcid;
		func_args = func->args;
	}
	else if (IsA(arg, OpExpr) || IsA(arg, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *)arg;

		func_oid  = get_opcode(op->opno);
		func_args = op->args;
	}
	else
		__ELog("Not supported CoerceViaIO with jsonb key reference");

	switch (func_oid)
	{
		case F_JSONB_OBJECT_FIELD_TEXT:
			snprintf(dfunc_name, sizeof(dfunc_name),
					 "jsonb_object_field_as_%s", dtype->type_name);
			break;
		case F_JSONB_ARRAY_ELEMENT_TEXT:
			snprintf(dfunc_name, sizeof(dfunc_name),
					 "jsonb_array_element_as_%s", dtype->type_name);
			break;
		default:
			__ELog("Not supported CoerceViaIO with jsonb key reference");
	}
	context->extra_flags |= DEVKERNEL_NEEDS_JSONLIB;
	__appendStringInfo(body, "pgfn_%s(kcxt", dfunc_name);
	foreach (lc, func_args)
	{
		Node   *expr = lfirst(lc);
		int		dummy;

		__appendStringInfo(body, ", ");
		codegen_expression_walker(context, body, expr, &dummy);
	}
	__appendStringInfoChar(body, ')');
	if (dtype->type_length > 0)
		width = dtype->type_length;
	else if (dtype->type_length == -1)
		width = -1;		/* we don't know max length of a jsonb field */
	else
		elog(ERROR, "unexpected type length: %d", dtype->type_length);

	return width;
}

static void
devtype_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	dlist_mutable_iter iter;
	int		hindex;

	Assert(cacheid == TYPEOID);
	if (hashvalue == 0)
	{
		for (hindex=0; hindex < lengthof(devtype_info_slot); hindex++)
			dlist_init(&devtype_info_slot[hindex]);
		return;
	}

	hindex = hashvalue % lengthof(devtype_info_slot);
	dlist_foreach_modify (iter, &devtype_info_slot[hindex])
	{
		devtype_info *dtype = dlist_container(devtype_info,
											  chain, iter.cur);
		if (dtype->hashvalue == hashvalue)
		{
			dlist_delete(&dtype->chain);
			memset(&dtype->chain, 0, sizeof(dlist_node));
		}
	}
}

static void
devfunc_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	dlist_mutable_iter iter;
	int		hindex;

	Assert(cacheid == PROCOID);
	if (hashvalue == 0)
	{
		for (hindex=0; hindex < lengthof(devfunc_info_slot); hindex++)
			dlist_init(&devfunc_info_slot[hindex]);
		return;
	}

	hindex = hashvalue % lengthof(devfunc_info_slot);
	dlist_foreach_modify (iter, &devfunc_info_slot[hindex])
	{
		devfunc_info *dfunc = dlist_container(devfunc_info,
											  chain, iter.cur);
		if (dfunc->hashvalue == hashvalue)
		{
			dlist_delete(&dfunc->chain);
			memset(&dfunc->chain, 0, sizeof(dlist_node));
		}
	}
}

static void
devcast_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	dlist_mutable_iter iter;
	int		hindex;

	Assert(cacheid == CASTSOURCETARGET);
	if (hashvalue == 0)
	{
		for (hindex=0; hindex < lengthof(devcast_info_slot); hindex++)
			dlist_init(&devcast_info_slot[hindex]);
		return;
	}

	hindex = hashvalue % lengthof(devcast_info_slot);
	dlist_foreach_modify (iter, &devcast_info_slot[hindex])
	{
		devcast_info *dcast = dlist_container(devcast_info,
											  chain, iter.cur);
		if (dcast->hashvalue == hashvalue)
		{
			dlist_delete(&dcast->chain);
			memset(&dcast->chain, 0, sizeof(dlist_node));
		}
	}
}

static void
devindex_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	dlist_mutable_iter iter;
	int		hindex;

	Assert(cacheid == AMOPOPID);
	if (hashvalue == 0)
	{
		for (hindex=0; hindex < lengthof(devindex_info_slot); hindex++)
			dlist_init(&devindex_info_slot[hindex]);
		return;
	}

	hindex = hashvalue % lengthof(devcast_info_slot);
	dlist_foreach_modify (iter, &devcast_info_slot[hindex])
	{
		devindex_info *dindex = dlist_container(devindex_info,
												chain, iter.cur);
		if (dindex->hashvalue == hashvalue)
		{
			dlist_delete(&dindex->chain);
			memset(&dindex->chain, 0, sizeof(dlist_node));
		}
	}
}

void
pgstrom_init_codegen_context(codegen_context *context,
							 PlannerInfo *root,
							 RelOptInfo *baserel)
{
	memset(context, 0, sizeof(codegen_context));
	initStringInfo(&context->decl);
	context->root = root;
	context->baserel = baserel;
}

void
pgstrom_init_codegen(void)
{
	int		i;

	for (i=0; i < lengthof(devtype_info_slot); i++)
		dlist_init(&devtype_info_slot[i]);
	for (i=0; i < lengthof(devfunc_info_slot); i++)
		dlist_init(&devfunc_info_slot[i]);
	for (i=0; i < lengthof(devcast_info_slot); i++)
		dlist_init(&devcast_info_slot[i]);
	for (i=0; i < lengthof(devindex_info_slot); i++)
		dlist_init(&devindex_info_slot[i]);

	devinfo_memcxt = AllocSetContextCreate(CacheMemoryContext,
										   "device type/func info cache",
										   ALLOCSET_DEFAULT_SIZES);
	CacheRegisterSyscacheCallback(PROCOID, devfunc_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(TYPEOID, devtype_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(CASTSOURCETARGET, devcast_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(AMOPOPID, devindex_cache_invalidator, 0);
}
