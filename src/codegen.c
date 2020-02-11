/*
 * codegen.c
 *
 * Routines for CUDA code generator
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
#include "cuda_numeric.h"

static MemoryContext	devinfo_memcxt;
static dlist_head	devtype_info_slot[128];
static dlist_head	devfunc_info_slot[1024];
static dlist_head	devcast_info_slot[48];

static cl_uint generic_devtype_hashfunc(devtype_info *dtype, Datum datum);
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

/* callback to handle special cases of device cast */
static int	devcast_text2numeric_callback(codegen_context *context,
										  devcast_info *dcast,
										  CoerceViaIO *node);
/* error report */
#define __ELog(fmt, ...)								\
	ereport(ERROR,										\
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),	\
			 errmsg((fmt), ##__VA_ARGS__)))

/*
 * Catalog of data types supported by device code
 *
 * naming convension of types:
 *   pg_<type_name>_t
 */
#define DEVTYPE_DECL(type_name,type_oid_label,					\
					 min_const,max_const,zero_const,			\
					 type_flags,extra_sz,hash_func)				\
	{ "pg_catalog", type_name, type_oid_label,					\
	  min_const, max_const, zero_const, type_flags, extra_sz, hash_func }

/* XXX - These types have no constant definition at catalog/pg_type.h */
#define INT8RANGEOID	3926
#define NUMRANGEOID		3906
#define TSRANGEOID		3908
#define TSTZRANGEOID	3910
#define DATERANEGOID	3912

static struct {
	const char	   *type_schema;
	const char	   *type_name;
	const char	   *type_oid_label;
	const char	   *max_const;
	const char	   *min_const;
	const char	   *zero_const;
	cl_uint			type_flags;		/* library to declare this type */
	cl_uint			extra_sz;		/* required size to store internal form */
	devtype_hashfunc_type hash_func;
} devtype_catalog[] = {
	/*
	 * Primitive datatypes
	 */
	DEVTYPE_DECL("bool",   "BOOLOID",
				 NULL, NULL, "false",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("int2",   "INT2OID",
				 "SHRT_MAX", "SHRT_MIN", "0",
				 0, 0, pg_int2_devtype_hashfunc),
	DEVTYPE_DECL("int4",   "INT4OID",
				 "INT_MAX", "INT_MIN", "0",
				 0, 0, pg_int4_devtype_hashfunc),
	DEVTYPE_DECL("int8",   "INT8OID",
				 "LONG_MAX", "LONG_MIN", "0",
				 0, 0, pg_int8_devtype_hashfunc),
	/* XXX - float2 is not a built-in data type */
	DEVTYPE_DECL("float2", "FLOAT2OID",
				 "__half_as_short(HALF_MAX)",
				 "__half_as_short(-HALF_MAX)",
				 "__half_as_short(0.0)",
				 0, 0, pg_float2_devtype_hashfunc),
	DEVTYPE_DECL("float4", "FLOAT4OID",
				 "__float_as_int(FLT_MAX)",
				 "__float_as_int(-FLT_MAX)",
				 "__float_as_int(0.0)",
				 0, 0, pg_float4_devtype_hashfunc),
	DEVTYPE_DECL("float8", "FLOAT8OID",
				 "__double_as_longlong(DBL_MAX)",
				 "__double_as_longlong(-DBL_MAX)",
				 "__double_as_longlong(0.0)",
				 0, 0, pg_float8_devtype_hashfunc),
	/*
	 * Misc data types
	 */
	DEVTYPE_DECL("money",  "CASHOID",
				 "LONG_MAX", "LONG_MIN", "0",
				 DEVKERNEL_NEEDS_MISCLIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("uuid",   "UUIDOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISCLIB, UUID_LEN,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("macaddr", "MACADDROID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISCLIB, sizeof(macaddr),
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("inet",   "INETOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISCLIB, sizeof(inet),
				 pg_inet_devtype_hashfunc),
	DEVTYPE_DECL("cidr",   "CIDROID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISCLIB, sizeof(inet),
				 pg_inet_devtype_hashfunc),
	/*
	 * Date and time datatypes
	 */
	DEVTYPE_DECL("date", "DATEOID",
				 "INT_MAX", "INT_MIN", "0",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("time", "TIMEOID",
				 "LONG_MAX", "LONG_MIN", "0",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timetz", "TIMETZOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB,
				 sizeof(TimeTzADT),
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timestamp", "TIMESTAMPOID",
				 "LONG_MAX", "LONG_MIN", "0",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timestamptz", "TIMESTAMPTZOID",
				 "LONG_MAX", "LONG_MIN", "0",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("interval", "INTERVALOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB,
				 sizeof(Interval),
				 pg_interval_devtype_hashfunc),
	/*
	 * variable length datatypes
	 */
	DEVTYPE_DECL("bpchar",  "BPCHAROID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TEXTLIB, 0,
				 pg_bpchar_devtype_hashfunc),
	DEVTYPE_DECL("varchar", "VARCHAROID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TEXTLIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("numeric", "NUMERICOID",
				 NULL, NULL, NULL,
				 0, sizeof(struct NumericData),
				 pg_numeric_devtype_hashfunc),
	DEVTYPE_DECL("bytea",   "BYTEAOID",
				 NULL, NULL, NULL,
				 0,
				 sizeof(pg_varlena_t),
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("text",    "TEXTOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TEXTLIB,
				 sizeof(pg_varlena_t),
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("jsonb",   "JSONBOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_JSONLIB,
				 /* see comment at vlbuf_estimate_jsonb() */
				 TOAST_TUPLE_THRESHOLD,
				 pg_jsonb_devtype_hashfunc),
	/*
	 * range types
	 */
	DEVTYPE_DECL("int4range",  "INT4RANGEOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(cl_int) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("int8range",  "INT8RANGEOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(cl_long) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("tsrange",    "TSRANGEOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(Timestamp) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("tstzrange",  "TSTZRANGEOID",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(TimestampTz) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("daterange",  "DATERANGEOID",
				 NULL, NULL, NULL,
                 DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(DateADT) + 1,
				 pg_range_devtype_hashfunc),
};

static devtype_info *
build_basic_devtype_info(TypeCacheEntry *tcache)
{
	int		i;

	for (i=0; i < lengthof(devtype_catalog); i++)
	{
		const char *nsp_name = devtype_catalog[i].type_schema;
		const char *typ_name = devtype_catalog[i].type_name;
		Oid			nsp_oid;
		Oid			typ_oid;

		nsp_oid = get_namespace_oid(nsp_name, true);
		if (!OidIsValid(nsp_oid))
			continue;
		typ_oid = get_type_oid(typ_name, nsp_oid, true);
		if (typ_oid == tcache->type_id)
		{
			devtype_info   *entry
				= MemoryContextAllocZero(devinfo_memcxt,
										 offsetof(devtype_info,
												  comp_subtypes[0]));
			entry->hashvalue = GetSysCacheHashValue(TYPEOID, typ_oid, 0, 0, 0);
			entry->type_oid = typ_oid;
			entry->type_flags = devtype_catalog[i].type_flags;
			entry->type_length = tcache->typlen;
			entry->type_align = typealign_get_width(tcache->typalign);
			entry->type_byval = tcache->typbyval;
			entry->type_name = devtype_catalog[i].type_name; /* const */
			entry->max_const = devtype_catalog[i].max_const;
			entry->min_const = devtype_catalog[i].min_const;
			entry->zero_const = devtype_catalog[i].zero_const;
			entry->extra_sz = devtype_catalog[i].extra_sz;
			entry->hash_func = devtype_catalog[i].hash_func;
			/* type equality functions */
			entry->type_eqfunc = get_opcode(tcache->eq_opr);
			entry->type_cmpfunc = tcache->cmp_proc;

			return entry;
		}
	}
	return NULL;
}

static devtype_info *
build_array_devtype_info(TypeCacheEntry *tcache)
{
	devtype_info *element;
	devtype_info *entry;
	Oid			typelem = get_element_type(tcache->type_id);

	Assert(OidIsValid(typelem) && tcache->typlen == -1);
	element = pgstrom_devtype_lookup(typelem);
	if (!element)
		return NULL;
	entry = MemoryContextAllocZero(devinfo_memcxt,
								   offsetof(devtype_info,
											comp_subtypes[0]));
	entry->hashvalue = GetSysCacheHashValue(TYPEOID,
											tcache->type_id, 0, 0, 0);
	entry->type_oid = tcache->type_id;
	entry->type_flags = element->type_flags;
	entry->type_length = tcache->typlen;
	entry->type_align = typealign_get_width(tcache->typalign);
	entry->type_byval = tcache->typbyval;
	entry->type_name = "array";
	entry->max_const = NULL;
	entry->min_const = NULL;
	entry->zero_const = NULL;
	entry->extra_sz = sizeof(pg_array_t);
	entry->hash_func = generic_devtype_hashfunc;
	entry->type_element = element;

	return entry;
}

static devtype_info *
build_composite_devtype_info(TypeCacheEntry *tcache)
{
	Oid				type_relid = tcache->typrelid;
	int				j, nfields = get_relnatts(type_relid);
	devtype_info  **subtypes = alloca(sizeof(devtype_info *) * nfields);
	devtype_info   *entry;
	cl_uint			extra_flags = 0;
	size_t			extra_sz;

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
	entry = MemoryContextAllocZero(devinfo_memcxt,
								   offsetof(devtype_info,
											comp_subtypes[nfields]));
	entry->hashvalue = GetSysCacheHashValue(TYPEOID,
											tcache->type_id, 0, 0, 0);
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
	return entry;
}

devtype_info *
pgstrom_devtype_lookup(Oid type_oid)
{
	TypeCacheEntry *tcache;
	devtype_info   *dtype;
	int				hindex;
	dlist_iter		iter;

	/* lookup dtype that is already built */
	hindex = hash_uint32(type_oid) % lengthof(devtype_info_slot);
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
	tcache = lookup_type_cache(type_oid,
							   TYPECACHE_EQ_OPR |
							   TYPECACHE_CMP_PROC);
	if (OidIsValid(tcache->typrelid))
		dtype = build_composite_devtype_info(tcache);
	else
	{
		Oid		typelem = get_element_type(tcache->type_id);

		if (OidIsValid(typelem) && tcache->typlen == -1)
			dtype = build_array_devtype_info(tcache);
		else
			dtype = build_basic_devtype_info(tcache);
	}
	/* makes a negative entry, if not in the catalog */
	if (!dtype)
	{
		dtype = MemoryContextAllocZero(devinfo_memcxt,
									   offsetof(devtype_info,
												comp_subtypes[0]));
		dtype->hashvalue = GetSysCacheHashValue(TYPEOID, type_oid, 0, 0, 0);
		dtype->type_oid = type_oid;
		dtype->type_is_negative = true;
	}
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

/* dump all the type_oid declaration */
void
pgstrom_codegen_typeoid_declarations(StringInfo source)
{
	int		i;

	for (i=0; i < lengthof(devtype_catalog); i++)
	{
		const char *nsp_name = devtype_catalog[i].type_schema;
		const char *typ_name = devtype_catalog[i].type_name;
		const char *oid_label = devtype_catalog[i].type_oid_label;
		Oid			nsp_oid;
		Oid			typ_oid;

		nsp_oid = get_namespace_oid(nsp_name, true);
		if (!OidIsValid(nsp_oid))
			continue;

		typ_oid = get_type_oid(typ_name, nsp_oid, true);
		if (!OidIsValid(typ_oid))
			continue;

		appendStringInfo(source, "#define PG_%s %u\n", oid_label, typ_oid);
	}
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
pg_int2_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	cl_int		ival = DatumGetInt16(datum);

	return hash_any((cl_uchar *)&ival, sizeof(cl_int));
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
	context->varlena_bufsz += MAXALIGN(maxlen + VARHDRSZ);

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
	/*
	 * We usually have no information about jsonb object length preliminary,
	 * however, plain varlena must be less than the threshold of toasting.
	 * If user altered storage option of jsonb column to 'main', it may be
	 * increased to BLCKSZ, but unusual.
	 */
	return TOAST_TUPLE_THRESHOLD;
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
	const char *func_name;
	int			func_nargs;
	Oid			func_argtypes[DEVFUNC_MAX_NARGS];
	int			func_devcost;	/* relative cost to run on device */
	const char *func_template;	/* a template string if simple function */
	devfunc_result_sz_type devfunc_result_sz;
} devfunc_catalog_t;

static devfunc_catalog_t devfunc_common_catalog[] = {
	/* Type cast functions */
	{ "bool", 1, {INT4OID},     1, "f:to_bool" },

	{ "int2", 1, {INT4OID},     1, "f:to_int2" },
	{ "int2", 1, {INT8OID},     1, "f:to_int2" },
	{ "int2", 1, {FLOAT4OID},   1, "f:to_int2" },
	{ "int2", 1, {FLOAT8OID},   1, "f:to_int2" },

	{ "int4", 1, {BOOLOID},     1, "f:to_int4" },
	{ "int4", 1, {INT2OID},     1, "f:to_int4" },
	{ "int4", 1, {INT8OID},     1, "f:to_int4" },
	{ "int4", 1, {FLOAT4OID},   1, "f:to_int4" },
	{ "int4", 1, {FLOAT8OID},   1, "f:to_int4" },

	{ "int8", 1, {INT2OID},     1, "f:to_int8" },
	{ "int8", 1, {INT4OID},     1, "f:to_int8" },
	{ "int8", 1, {FLOAT4OID},   1, "f:to_int8" },
	{ "int8", 1, {FLOAT8OID},   1, "f:to_int8" },

	{ "float4", 1, {INT2OID},   1, "f:to_float4" },
	{ "float4", 1, {INT4OID},   1, "f:to_float4" },
	{ "float4", 1, {INT8OID},   1, "f:to_float4" },
	{ "float4", 1, {FLOAT8OID}, 1, "f:to_float4" },

	{ "float8", 1, {INT2OID},   1, "f:to_float8" },
	{ "float8", 1, {INT4OID},   1, "f:to_float8" },
	{ "float8", 1, {INT8OID},   1, "f:to_float8" },
	{ "float8", 1, {FLOAT4OID}, 1, "f:to_float8" },

	/* '+' : add operators */
	{ "int2pl",  2, {INT2OID, INT2OID}, 1, "p/f:int2pl" },
	{ "int24pl", 2, {INT2OID, INT4OID}, 1, "p/f:int24pl" },
	{ "int28pl", 2, {INT2OID, INT8OID}, 1, "p/f:int28pl" },
	{ "int42pl", 2, {INT4OID, INT2OID}, 1, "p/f:int42pl" },
	{ "int4pl",  2, {INT4OID, INT4OID}, 1, "p/f:int4pl" },
	{ "int48pl", 2, {INT4OID, INT8OID}, 1, "p/f:int48pl" },
	{ "int82pl", 2, {INT8OID, INT2OID}, 1, "p/f:int82pl" },
	{ "int84pl", 2, {INT8OID, INT4OID}, 1, "p/f:int84pl" },
	{ "int8pl",  2, {INT8OID, INT8OID}, 1, "p/f:int8pl" },
	{ "float4pl",  2, {FLOAT4OID, FLOAT4OID}, 1, "p/f:float4pl" },
	{ "float48pl", 2, {FLOAT4OID, FLOAT8OID}, 1, "p/f:float48pl" },
	{ "float84pl", 2, {FLOAT8OID, FLOAT4OID}, 1, "p/f:float84pl" },
	{ "float8pl",  2, {FLOAT8OID, FLOAT8OID}, 1, "p/f:float8pl" },

	/* '-' : subtract operators */
	{ "int2mi",  2, {INT2OID, INT2OID}, 1, "p/f:int2mi" },
	{ "int24mi", 2, {INT2OID, INT4OID}, 1, "p/f:int24mi" },
	{ "int28mi", 2, {INT2OID, INT8OID}, 1, "p/f:int28mi" },
	{ "int42mi", 2, {INT4OID, INT2OID}, 1, "p/f:int42mi" },
	{ "int4mi",  2, {INT4OID, INT4OID}, 1, "p/f:int4mi" },
	{ "int48mi", 2, {INT4OID, INT8OID}, 1, "p/f:int48mi" },
	{ "int82mi", 2, {INT8OID, INT2OID}, 1, "p/f:int82mi" },
	{ "int84mi", 2, {INT8OID, INT4OID}, 1, "p/f:int84mi" },
	{ "int8mi",  2, {INT8OID, INT8OID}, 1, "p/f:int8mi" },
	{ "float4mi",  2, {FLOAT4OID, FLOAT4OID}, 1, "p/f:float4mi" },
	{ "float48mi", 2, {FLOAT4OID, FLOAT8OID}, 1, "p/f:float48mi" },
	{ "float84mi", 2, {FLOAT8OID, FLOAT4OID}, 1, "p/f:float84mi" },
	{ "float8mi",  2, {FLOAT8OID, FLOAT8OID}, 1, "p/f:float8mi" },

	/* '*' : mutiply operators */
	{ "int2mul",  2, {INT2OID, INT2OID}, 2, "p/f:int2mul" },
	{ "int24mul", 2, {INT2OID, INT4OID}, 2, "p/f:int24mul" },
	{ "int28mul", 2, {INT2OID, INT8OID}, 2, "p/f:int28mul" },
	{ "int42mul", 2, {INT4OID, INT2OID}, 2, "p/f:int42mul" },
	{ "int4mul",  2, {INT4OID, INT4OID}, 2, "p/f:int4mul" },
	{ "int48mul", 2, {INT4OID, INT8OID}, 2, "p/f:int48mul" },
	{ "int82mul", 2, {INT8OID, INT2OID}, 2, "p/f:int82mul" },
	{ "int84mul", 2, {INT8OID, INT4OID}, 2, "p/f:int84mul" },
	{ "int8mul",  2, {INT8OID, INT8OID}, 2, "p/f:int8mul" },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, 2, "p/f:float4mul" },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, 2, "p/f:float48mul" },
	{ "float84mul", 2, {FLOAT8OID, FLOAT4OID}, 2, "p/f:float84mul" },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, 2, "p/f:float8mul" },

	/* '/' : divide operators */
	{ "int2div",  2, {INT2OID, INT2OID}, 2, "p/f:int2div" },
	{ "int24div", 2, {INT2OID, INT4OID}, 2, "p/f:int24div" },
	{ "int28div", 2, {INT2OID, INT8OID}, 2, "p/f:int28div" },
	{ "int42div", 2, {INT4OID, INT2OID}, 2, "p/f:int42div" },
	{ "int4div",  2, {INT4OID, INT4OID}, 2, "p/f:int4div" },
	{ "int48div", 2, {INT4OID, INT8OID}, 2, "p/f:int48div" },
	{ "int82div", 2, {INT8OID, INT2OID}, 2, "p/f:int82div" },
	{ "int84div", 2, {INT8OID, INT4OID}, 2, "p/f:int84div" },
	{ "int8div",  2, {INT8OID, INT8OID}, 2, "p/f:int8div" },
	{ "float4div",  2, {FLOAT4OID, FLOAT4OID}, 2, "p/f:float4div" },
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, 2, "p/f:float48div" },
	{ "float84div", 2, {FLOAT8OID, FLOAT4OID}, 2, "p/f:float84div" },
	{ "float8div",  2, {FLOAT8OID, FLOAT8OID}, 2, "p/f:float8div" },

	/* '%' : reminder operators */
	{ "int2mod", 2, {INT2OID, INT2OID}, 2, "p/f:int2mod" },
	{ "int4mod", 2, {INT4OID, INT4OID}, 2, "p/f:int4mod" },
	{ "int8mod", 2, {INT8OID, INT8OID}, 2, "p/f:int8mod" },

	/* '+' : unary plus operators */
	{ "int2up", 1, {INT2OID},      1, "p/f:int2up" },
	{ "int4up", 1, {INT4OID},      1, "p/f:int4up" },
	{ "int8up", 1, {INT8OID},      1, "p/f:int8up" },
	{ "float4up", 1, {FLOAT4OID},  1, "p/f:float4up" },
	{ "float8up", 1, {FLOAT8OID},  1, "p/f:float8up" },

	/* '-' : unary minus operators */
	{ "int2um", 1, {INT2OID},      1, "p/f:int2um" },
	{ "int4um", 1, {INT4OID},      1, "p/f:int4um" },
	{ "int8um", 1, {INT8OID},      1, "p/f:int8um" },
	{ "float4um", 1, {FLOAT4OID},  1, "p/f:float4um" },
	{ "float8um", 1, {FLOAT8OID},  1, "p/f:float8um" },

	/* '@' : absolute value operators */
	{ "int2abs", 1, {INT2OID},     1, "p/f:int2abs" },
	{ "int4abs", 1, {INT4OID},     1, "p/f:int4abs" },
	{ "int8abs", 1, {INT8OID},     1, "p/f:int8abs" },
	{ "float4abs", 1, {FLOAT4OID}, 1, "p/f:float4abs" },
	{ "float8abs", 1, {FLOAT8OID}, 1, "p/f:float8abs" },

	/* '=' : equal operators */
	{ "booleq",  2, {BOOLOID, BOOLOID}, 1, "f:booleq" },
	{ "int2eq",  2, {INT2OID, INT2OID}, 1, "f:int2eq" },
	{ "int24eq", 2, {INT2OID, INT4OID}, 1, "f:int24eq" },
	{ "int28eq", 2, {INT2OID, INT8OID}, 1, "f:int28eq" },
	{ "int42eq", 2, {INT4OID, INT2OID}, 1, "f:int42eq" },
	{ "int4eq",  2, {INT4OID, INT4OID}, 1, "f:int4eq" },
	{ "int48eq", 2, {INT4OID, INT8OID}, 1, "f:int48eq" },
	{ "int82eq", 2, {INT8OID, INT2OID}, 1, "f:int82eq" },
	{ "int84eq", 2, {INT8OID, INT4OID}, 1, "f:int84eq" },
	{ "int8eq",  2, {INT8OID, INT8OID}, 1, "f:int8eq" },
	{ "float4eq",  2, {FLOAT4OID, FLOAT4OID}, 1, "f:float4eq" },
	{ "float48eq", 2, {FLOAT4OID, FLOAT8OID}, 1, "f:float48eq" },
	{ "float84eq", 2, {FLOAT8OID, FLOAT4OID}, 1, "f:float84eq" },
	{ "float8eq",  2, {FLOAT8OID, FLOAT8OID}, 1, "f:float8eq" },

	/* '<>' : not equal operators */
	{ "int2ne",  2, {INT2OID, INT2OID}, 1, "f:int2ne" },
	{ "int24ne", 2, {INT2OID, INT4OID}, 1, "f:int24ne" },
	{ "int28ne", 2, {INT2OID, INT8OID}, 1, "f:int28ne" },
	{ "int42ne", 2, {INT4OID, INT2OID}, 1, "f:int42ne" },
	{ "int4ne",  2, {INT4OID, INT4OID}, 1, "f:int4ne" },
	{ "int48ne", 2, {INT4OID, INT8OID}, 1, "f:int48ne" },
	{ "int82ne", 2, {INT8OID, INT2OID}, 1, "f:int82ne" },
	{ "int84ne", 2, {INT8OID, INT4OID}, 1, "f:int84ne" },
	{ "int8ne",  2, {INT8OID, INT8OID}, 1, "f:int8ne" },
	{ "float4ne",  2, {FLOAT4OID, FLOAT4OID}, 1, "f:float4ne" },
	{ "float48ne", 2, {FLOAT4OID, FLOAT8OID}, 1, "f:float48ne" },
	{ "float84ne", 2, {FLOAT8OID, FLOAT4OID}, 1, "f:float84ne" },
	{ "float8ne",  2, {FLOAT8OID, FLOAT8OID}, 1, "f:float8ne" },

	/* '>' : greater than operators */
	{ "int2gt",  2, {INT2OID, INT2OID}, 1, "f:int2gt" },
	{ "int24gt", 2, {INT2OID, INT4OID}, 1, "f:int24gt" },
	{ "int28gt", 2, {INT2OID, INT8OID}, 1, "f:int28gt" },
	{ "int42gt", 2, {INT4OID, INT2OID}, 1, "f:int42gt" },
	{ "int4gt",  2, {INT4OID, INT4OID}, 1, "f:int4gt" },
	{ "int48gt", 2, {INT4OID, INT8OID}, 1, "f:int48gt" },
	{ "int82gt", 2, {INT8OID, INT2OID}, 1, "f:int82gt" },
	{ "int84gt", 2, {INT8OID, INT4OID}, 1, "f:int84gt" },
	{ "int8gt",  2, {INT8OID, INT8OID}, 1, "f:int8gt" },
	{ "float4gt",  2, {FLOAT4OID, FLOAT4OID}, 1, "f:float4gt" },
	{ "float48gt", 2, {FLOAT4OID, FLOAT8OID}, 1, "f:float48gt" },
	{ "float84gt", 2, {FLOAT8OID, FLOAT4OID}, 1, "f:float84gt" },
	{ "float8gt",  2, {FLOAT8OID, FLOAT8OID}, 1, "f:float8gt" },

	/* '<' : less than operators */
	{ "int2lt",  2, {INT2OID, INT2OID}, 1, "f:int2lt" },
	{ "int24lt", 2, {INT2OID, INT4OID}, 1, "f:int24lt" },
	{ "int28lt", 2, {INT2OID, INT8OID}, 1, "f:int28lt" },
	{ "int42lt", 2, {INT4OID, INT2OID}, 1, "f:int42lt" },
	{ "int4lt",  2, {INT4OID, INT4OID}, 1, "f:int4lt" },
	{ "int48lt", 2, {INT4OID, INT8OID}, 1, "f:int48lt" },
	{ "int82lt", 2, {INT8OID, INT2OID}, 1, "f:int82lt" },
	{ "int84lt", 2, {INT8OID, INT4OID}, 1, "f:int84lt" },
	{ "int8lt",  2, {INT8OID, INT8OID}, 1, "f:int8lt" },
	{ "float4lt",  2, {FLOAT4OID, FLOAT4OID}, 1, "f:float4lt" },
	{ "float48lt", 2, {FLOAT4OID, FLOAT8OID}, 1, "f:float48lt" },
	{ "float84lt", 2, {FLOAT8OID, FLOAT4OID}, 1, "f:float84lt" },
	{ "float8lt",  2, {FLOAT8OID, FLOAT8OID}, 1, "f:float8lt" },

	/* '>=' : relational greater-than or equal-to */
	{ "int2ge",  2, {INT2OID, INT2OID}, 1, "f:int2ge" },
	{ "int24ge", 2, {INT2OID, INT4OID}, 1, "f:int24ge" },
	{ "int28ge", 2, {INT2OID, INT8OID}, 1, "f:int28ge" },
	{ "int42ge", 2, {INT4OID, INT2OID}, 1, "f:int42ge" },
	{ "int4ge",  2, {INT4OID, INT4OID}, 1, "f:int4ge" },
	{ "int48ge", 2, {INT4OID, INT8OID}, 1, "f:int48ge" },
	{ "int82ge", 2, {INT8OID, INT2OID}, 1, "f:int82ge" },
	{ "int84ge", 2, {INT8OID, INT4OID}, 1, "f:int84ge" },
	{ "int8ge",  2, {INT8OID, INT8OID}, 1, "f:int8ge" },
	{ "float4ge",  2, {FLOAT4OID, FLOAT4OID}, 1, "f:float4ge" },
	{ "float48ge", 2, {FLOAT4OID, FLOAT8OID}, 1, "f:float48ge" },
	{ "float84ge", 2, {FLOAT8OID, FLOAT4OID}, 1, "f:float84ge" },
	{ "float8ge",  2, {FLOAT8OID, FLOAT8OID}, 1, "f:float8ge" },

	/* '<=' : relational greater-than or equal-to */
	{ "int2le",  2, {INT2OID, INT2OID}, 1, "f:int2le" },
	{ "int24le", 2, {INT2OID, INT4OID}, 1, "f:int24le" },
	{ "int28le", 2, {INT2OID, INT8OID}, 1, "f:int28le" },
	{ "int42le", 2, {INT4OID, INT2OID}, 1, "f:int42le" },
	{ "int4le",  2, {INT4OID, INT4OID}, 1, "f:int4le" },
	{ "int48le", 2, {INT4OID, INT8OID}, 1, "f:int48le" },
	{ "int82le", 2, {INT8OID, INT2OID}, 1, "f:int82le" },
	{ "int84le", 2, {INT8OID, INT4OID}, 1, "f:int84le" },
	{ "int8le",  2, {INT8OID, INT8OID}, 1, "f:int8le" },
	{ "float4le",  2, {FLOAT4OID, FLOAT4OID}, 1, "f:float4le" },
	{ "float48le", 2, {FLOAT4OID, FLOAT8OID}, 1, "f:float48le" },
	{ "float84le", 2, {FLOAT8OID, FLOAT4OID}, 1, "f:float84le" },
	{ "float8le",  2, {FLOAT8OID, FLOAT8OID}, 1, "f:float8le" },

	/* '&' : bitwise and */
	{ "int2and", 2, {INT2OID, INT2OID}, 1, "p/f:int2and" },
	{ "int4and", 2, {INT4OID, INT4OID}, 1, "p/f:int4and" },
	{ "int8and", 2, {INT8OID, INT8OID}, 1, "p/f:int8and" },

	/* '|'  : bitwise or */
	{ "int2or", 2, {INT2OID, INT2OID}, 1, "p/f:int2or" },
	{ "int4or", 2, {INT4OID, INT4OID}, 1, "p/f:int4or" },
	{ "int8or", 2, {INT8OID, INT8OID}, 1, "p/f:int8or" },

	/* '#'  : bitwise xor */
	{ "int2xor", 2, {INT2OID, INT2OID}, 1, "p/f:int2xor" },
	{ "int4xor", 2, {INT4OID, INT4OID}, 1, "p/f:int4xor" },
	{ "int8xor", 2, {INT8OID, INT8OID}, 1, "p/f:int8xor" },

	/* '~'  : bitwise not operators */
	{ "int2not", 1, {INT2OID}, 1, "p/f:int2not" },
	{ "int4not", 1, {INT4OID}, 1, "p/f:int4not" },
	{ "int8not", 1, {INT8OID}, 1, "p/f:int8not" },

	/* '>>' : right shift */
	{ "int2shr", 2, {INT2OID, INT4OID}, 1, "p/f:int2shr" },
	{ "int4shr", 2, {INT4OID, INT4OID}, 1, "p/f:int4shr" },
	{ "int8shr", 2, {INT8OID, INT4OID}, 1, "p/f:int8shr" },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID, INT4OID}, 1, "p/f:int2shl" },
	{ "int4shl", 2, {INT4OID, INT4OID}, 1, "p/f:int4shl" },
	{ "int8shl", 2, {INT8OID, INT4OID}, 1, "p/f:int8shl" },

	/* comparison functions */
	{ "btboolcmp",  2, {BOOLOID, BOOLOID}, 1, "p/f:type_compare" },
	{ "btint2cmp",  2, {INT2OID, INT2OID}, 1, "p/f:type_compare" },
	{ "btint24cmp", 2, {INT2OID, INT4OID}, 1, "p/f:type_compare" },
	{ "btint28cmp", 2, {INT2OID, INT8OID}, 1, "p/f:type_compare" },
	{ "btint42cmp", 2, {INT4OID, INT2OID}, 1, "p/f:type_compare" },
	{ "btint4cmp",  2, {INT4OID, INT4OID}, 1, "p/f:type_compare" },
	{ "btint48cmp", 2, {INT4OID, INT8OID}, 1, "p/f:type_compare" },
	{ "btint82cmp", 2, {INT8OID, INT2OID}, 1, "p/f:type_compare" },
	{ "btint84cmp", 2, {INT8OID, INT4OID}, 1, "p/f:type_compare" },
	{ "btint8cmp",  2, {INT8OID, INT8OID}, 1, "p/f:type_compare" },
	{ "btfloat4cmp",  2, {FLOAT4OID, FLOAT4OID}, 1, "p/f:type_compare" },
	{ "btfloat48cmp", 2, {FLOAT4OID, FLOAT8OID}, 1, "p/f:type_compare" },
	{ "btfloat84cmp", 2, {FLOAT8OID, FLOAT4OID}, 1, "p/f:type_compare" },
	{ "btfloat8cmp",  2, {FLOAT8OID, FLOAT8OID}, 1, "p/f:type_compare" },

	/* currency cast */
	{ "money",			1, {NUMERICOID},		1, "m/f:numeric_cash" },
	{ "money",			1, {INT4OID},			1, "m/f:int4_cash" },
	{ "money",			1, {INT8OID},			1, "m/f:int8_cash" },
	/* currency operators */
	{ "cash_pl",		2, {CASHOID, CASHOID},	1, "m/f:cash_pl" },
	{ "cash_mi",		2, {CASHOID, CASHOID},	1, "m/f:cash_mi" },
	{ "cash_div_cash",	2, {CASHOID, CASHOID},	2, "m/f:cash_div_cash" },
	{ "cash_mul_int2",	2, {CASHOID, INT2OID},	2, "m/f:cash_mul_int2" },
	{ "cash_mul_int4",	2, {CASHOID, INT4OID},	2, "m/f:cash_mul_int4" },
	{ "cash_mul_flt4",	2, {CASHOID, FLOAT4OID},2, "m/f:cash_mul_flt4" },
	{ "cash_mul_flt8",	2, {CASHOID, FLOAT8OID},2, "m/f:cash_mul_flt8" },
	{ "cash_div_int2",	2, {CASHOID, INT2OID},	2, "m/f:cash_div_int2" },
	{ "cash_div_int4",	2, {CASHOID, INT4OID},	2, "m/f:cash_div_int4" },
	{ "cash_div_flt4",	2, {CASHOID, FLOAT4OID},2, "m/f:cash_div_flt4" },
	{ "cash_div_flt8",	2, {CASHOID, FLOAT8OID},2, "m/f:cash_div_flt8" },
	{ "int2_mul_cash",	2, {INT2OID, CASHOID},	2, "m/f:int2_mul_cash" },
	{ "int4_mul_cash",	2, {INT4OID, CASHOID},	2, "m/f:int4_mul_cash" },
	{ "flt4_mul_cash",	2, {FLOAT4OID, CASHOID},2, "m/f:flt4_mul_cash" },
	{ "flt8_mul_cash",	2, {FLOAT8OID, CASHOID},2, "m/f:flt8_mul_cash" },
	/* currency comparison */
	{ "cash_cmp",		2, {CASHOID, CASHOID},	1, "m/f:type_compare" },
	{ "cash_eq",		2, {CASHOID, CASHOID},	1, "m/f:cash_eq" },
	{ "cash_ne",		2, {CASHOID, CASHOID},	1, "m/f:cash_ne" },
	{ "cash_lt",		2, {CASHOID, CASHOID},	1, "m/f:cash_lt" },
	{ "cash_le",		2, {CASHOID, CASHOID},	1, "m/f:cash_le" },
	{ "cash_gt",		2, {CASHOID, CASHOID},	1, "m/f:cash_gt" },
	{ "cash_ge",		2, {CASHOID, CASHOID},	1, "m/f:cash_ge" },
	/* uuid comparison */
	{ "uuid_cmp",		2, {UUIDOID, UUIDOID},	5, "m/f:type_compare" },
	{ "uuid_eq",		2, {UUIDOID, UUIDOID},	5, "m/f:uuid_eq" },
	{ "uuid_ne",		2, {UUIDOID, UUIDOID},	5, "m/f:uuid_ne" },
	{ "uuid_lt",		2, {UUIDOID, UUIDOID},	5, "m/f:uuid_lt" },
	{ "uuid_le",		2, {UUIDOID, UUIDOID},	5, "m/f:uuid_le" },
	{ "uuid_gt",		2, {UUIDOID, UUIDOID},	5, "m/f:uuid_gt" },
	{ "uuid_ge",		2, {UUIDOID, UUIDOID},	5, "m/f:uuid_ge" },
	/* macaddr comparison */
	{ "macaddr_cmp",    2, {MACADDROID,MACADDROID}, 5, "m/f:type_compare" },
	{ "macaddr_eq",     2, {MACADDROID,MACADDROID}, 5, "m/f:macaddr_eq" },
	{ "macaddr_ne",     2, {MACADDROID,MACADDROID}, 5, "m/f:macaddr_ne" },
	{ "macaddr_lt",     2, {MACADDROID,MACADDROID}, 5, "m/f:macaddr_lt" },
	{ "macaddr_le",     2, {MACADDROID,MACADDROID}, 5, "m/f:macaddr_le" },
	{ "macaddr_gt",     2, {MACADDROID,MACADDROID}, 5, "m/f:macaddr_gt" },
	{ "macaddr_ge",     2, {MACADDROID,MACADDROID}, 5, "m/f:macaddr_ge" },
	/* inet comparison */
	{ "network_cmp",    2, {INETOID,INETOID}, 8, "m/f:type_compare" },
	{ "network_eq",     2, {INETOID,INETOID}, 8, "m/f:network_eq" },
	{ "network_ne",     2, {INETOID,INETOID}, 8, "m/f:network_ne" },
	{ "network_lt",     2, {INETOID,INETOID}, 8, "m/f:network_lt" },
	{ "network_le",     2, {INETOID,INETOID}, 8, "m/f:network_le" },
	{ "network_gt",     2, {INETOID,INETOID}, 8, "m/f:network_gt" },
	{ "network_ge",     2, {INETOID,INETOID}, 8, "m/f:network_ge" },
	{ "network_larger", 2, {INETOID,INETOID}, 8, "m/f:network_larger" },
	{ "network_smaller",2, {INETOID,INETOID}, 8, "m/f:network_smaller" },
	{ "network_sub",    2, {INETOID,INETOID}, 8, "m/f:network_sub" },
	{ "network_subeq",  2, {INETOID,INETOID}, 8, "m/f:network_subeq" },
	{ "network_sup",    2, {INETOID,INETOID}, 8, "m/f:network_sup" },
	{ "network_supeq",  2, {INETOID,INETOID}, 8, "m/f:network_supeq" },
	{ "network_overlap",2, {INETOID,INETOID}, 8, "m/f:network_overlap" },

	/*
     * Mathmatical functions
     */
	{ "abs",     1, {INT2OID},   1, "p/f:int2abs" },
	{ "abs",     1, {INT4OID},   1, "p/f:int4abs" },
	{ "abs",     1, {INT8OID},   1, "p/f:int8abs" },
	{ "abs",     1, {FLOAT4OID}, 1, "p/f:float4abs" },
	{ "abs",     1, {FLOAT8OID}, 1, "p/f:float8abs" },
	{ "cbrt",    1, {FLOAT8OID}, 1, "m/f:cbrt" },
	{ "dcbrt",   1, {FLOAT8OID}, 1, "m/f:cbrt" },
	{ "ceil",    1, {FLOAT8OID}, 1, "m/f:ceil" },
	{ "ceiling", 1, {FLOAT8OID}, 1, "m/f:ceil" },
	{ "exp",     1, {FLOAT8OID}, 5, "m/f:exp" },
	{ "dexp",    1, {FLOAT8OID}, 5, "m/f:exp" },
	{ "floor",   1, {FLOAT8OID}, 1, "m/f:floor" },
	{ "ln",      1, {FLOAT8OID}, 5, "m/f:ln" },
	{ "dlog1",   1, {FLOAT8OID}, 5, "m/f:ln" },
	{ "log",     1, {FLOAT8OID}, 5, "m/f:log10" },
	{ "dlog10",  1, {FLOAT8OID}, 5, "m/f:log10" },
	{ "pi",      0, {}, 0, "m/f:dpi" },
	{ "power",   2, {FLOAT8OID, FLOAT8OID}, 5, "m/f:dpow" },
	{ "pow",     2, {FLOAT8OID, FLOAT8OID}, 5, "m/f:dpow" },
	{ "dpow",    2, {FLOAT8OID, FLOAT8OID}, 5, "m/f:dpow" },
	{ "round",   1, {FLOAT8OID}, 5, "m/f:round" },
	{ "dround",  1, {FLOAT8OID}, 5, "m/f:round" },
	{ "sign",    1, {FLOAT8OID}, 1, "m/f:sign" },
	{ "sqrt",    1, {FLOAT8OID}, 5, "m/f:dsqrt" },
	{ "dsqrt",   1, {FLOAT8OID}, 5, "m/f:dsqrt" },
	{ "trunc",   1, {FLOAT8OID}, 1, "m/f:trunc" },
	{ "dtrunc",  1, {FLOAT8OID}, 1, "m/f:trunc" },

	/*
     * Trigonometric function
     */
	{ "degrees", 1, {FLOAT8OID}, 5, "m/f:degrees" },
	{ "radians", 1, {FLOAT8OID}, 5, "m/f:radians" },
	{ "acos",    1, {FLOAT8OID}, 5, "m/f:acos" },
	{ "asin",    1, {FLOAT8OID}, 5, "m/f:asin" },
	{ "atan",    1, {FLOAT8OID}, 5, "m/f:atan" },
	{ "atan2",   2, {FLOAT8OID, FLOAT8OID}, 5, "m/f:atan2" },
	{ "cos",     1, {FLOAT8OID}, 5, "m/f:cos" },
	{ "cot",     1, {FLOAT8OID}, 5, "m/f:cot" },
	{ "sin",     1, {FLOAT8OID}, 5, "m/f:sin" },
	{ "tan",     1, {FLOAT8OID}, 5, "m/f:tan" },

	/*
	 * Numeric functions
	 * ------------------------- */
	/* Numeric type cast functions */
	{ "int2",    1, {NUMERICOID}, 8, "f:numeric_int2" },
	{ "int4",    1, {NUMERICOID}, 8, "f:numeric_int4" },
	{ "int8",    1, {NUMERICOID}, 8, "f:numeric_int8" },
	{ "float4",  1, {NUMERICOID}, 8, "f:numeric_float4" },
	{ "float8",  1, {NUMERICOID}, 8, "f:numeric_float8" },
	{ "numeric", 1, {INT2OID},    5, "f:int2_numeric" },
	{ "numeric", 1, {INT4OID},    5, "f:int4_numeric" },
	{ "numeric", 1, {INT8OID},    5, "f:int8_numeric" },
	{ "numeric", 1, {FLOAT4OID},  5, "f:float4_numeric" },
	{ "numeric", 1, {FLOAT8OID},  5, "f:float8_numeric" },
	/* Numeric operators */
	{ "numeric_add", 2, {NUMERICOID, NUMERICOID}, 10, "f:numeric_add" },
	{ "numeric_sub", 2, {NUMERICOID, NUMERICOID}, 10, "f:numeric_sub" },
	{ "numeric_mul", 2, {NUMERICOID, NUMERICOID}, 10, "f:numeric_mul" },
	{ "numeric_uplus",  1, {NUMERICOID}, 10, "f:numeric_uplus" },
	{ "numeric_uminus", 1, {NUMERICOID}, 10, "f:numeric_uminus" },
	{ "numeric_abs",    1, {NUMERICOID}, 10, "f:numeric_abs" },
	{ "abs",            1, {NUMERICOID}, 10, "f:numeric_abs" },
	/* Numeric comparison */
	{ "numeric_eq", 2, {NUMERICOID, NUMERICOID},  8, "f:numeric_eq" },
	{ "numeric_ne", 2, {NUMERICOID, NUMERICOID},  8, "f:numeric_ne" },
	{ "numeric_lt", 2, {NUMERICOID, NUMERICOID},  8, "f:numeric_lt" },
	{ "numeric_le", 2, {NUMERICOID, NUMERICOID},  8, "f:numeric_le" },
	{ "numeric_gt", 2, {NUMERICOID, NUMERICOID},  8, "f:numeric_gt" },
	{ "numeric_ge", 2, {NUMERICOID, NUMERICOID},  8, "f:numeric_ge" },
	{ "numeric_cmp", 2,{NUMERICOID, NUMERICOID},  8, "f:type_compare" },

	/*
	 * Date and time functions
	 * ------------------------------- */
	/* Type cast functions */
	{ "date", 1, {TIMESTAMPOID},     1, "t/f:timestamp_date" },
	{ "date", 1, {TIMESTAMPTZOID},   1, "t/f:timestamptz_date" },
	{ "time", 1, {TIMETZOID},        1, "t/f:timetz_time" },
	{ "time", 1, {TIMESTAMPOID},     1, "t/f:timestamp_time" },
	{ "time", 1, {TIMESTAMPTZOID},   1, "t/f:timestamptz_time" },
	{ "timetz", 1, {TIMEOID},        1, "t/f:time_timetz" },
	{ "timetz", 1, {TIMESTAMPTZOID}, 1, "t/f:timestamptz_timetz" },
#ifdef NOT_USED
	{ "timetz", 2, {TIMETZOID, INT4OID}, 1, "t/f:timetz_scale" },
#endif
	{ "timestamp", 1, {DATEOID},        1, "t/f:date_timestamp" },
	{ "timestamp", 1, {TIMESTAMPTZOID}, 1, "t/f:timestamptz_timestamp" },
	{ "timestamptz", 1, {DATEOID},      1, "t/f:date_timestamptz" },
	{ "timestamptz", 1, {TIMESTAMPOID}, 1, "t/f:timestamp_timestamptz" },
	/* timedata operators */
	{ "date_pli", 2, {DATEOID, INT4OID}, 1, "t/f:date_pli" },
	{ "date_mii", 2, {DATEOID, INT4OID}, 1, "t/f:date_mii" },
	{ "date_mi", 2, {DATEOID, DATEOID},  1, "t/f:date_mi" },
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, 2, "t/f:datetime_pl" },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, 2, "t/f:integer_pl_date" },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, 2, "t/f:timedate_pl" },
	/* time - time => interval */
	{ "time_mi_time", 2, {TIMEOID, TIMEOID}, 2, "t/f:time_mi_time" },
	/* timestamp - timestamp => interval */
	{ "timestamp_mi",
	  2, {TIMESTAMPOID, TIMESTAMPOID},
	  4, "t/f:timestamp_mi"
	},
	/* timetz +/- interval => timetz */
	{ "timetz_pl_interval",
	  2, {TIMETZOID, INTERVALOID},
	  4, "t/f:timetz_pl_interval"
	},
	{ "timetz_mi_interval",
	  2, {TIMETZOID, INTERVALOID},
	  4, "t/f:timetz_mi_interval"
	},
	/* timestamptz +/- interval => timestamptz */
	{ "timestamptz_pl_interval",
	  2, {TIMESTAMPTZOID, INTERVALOID},
	  4, "t/f:timestamptz_pl_interval"
	},
	{ "timestamptz_mi_interval",
	  2, {TIMESTAMPTZOID, INTERVALOID},
	  4, "t/f:timestamptz_mi_interval"
	},
	/* interval operators */
	{ "interval_um", 1, {INTERVALOID}, 4, "t/f:interval_um" },
	{ "interval_pl", 2, {INTERVALOID, INTERVALOID}, 4, "t/f:interval_pl" },
	{ "interval_mi", 2, {INTERVALOID, INTERVALOID}, 4, "t/f:interval_mi" },
	/* date + timetz => timestamptz */
	{ "datetimetz_pl",
	  2, {DATEOID, TIMETZOID},
	  4, "t/f:datetimetz_timestamptz"
	},
	{ "timestamptz",
	  2, {DATEOID, TIMETZOID},
	  4, "t/f:datetimetz_timestamptz"
	},
	/* comparison between date */
	{ "date_eq", 2, {DATEOID, DATEOID}, 2, "t/f:date_eq" },
	{ "date_ne", 2, {DATEOID, DATEOID}, 2, "t/f:date_ne" },
	{ "date_lt", 2, {DATEOID, DATEOID}, 2, "t/f:date_lt"  },
	{ "date_le", 2, {DATEOID, DATEOID}, 2, "t/f:date_le" },
	{ "date_gt", 2, {DATEOID, DATEOID}, 2, "t/f:date_gt"  },
	{ "date_ge", 2, {DATEOID, DATEOID}, 2, "t/f:date_ge" },
	{ "date_cmp",2, {DATEOID, DATEOID}, 2, "t/f:type_compare" },
	/* comparison of date and timestamp */
	{ "date_eq_timestamp",
	  2, {DATEOID, TIMESTAMPOID},
	  2, "t/f:date_eq_timestamp"
	},
	{ "date_ne_timestamp",
	  2, {DATEOID, TIMESTAMPOID},
	  2, "t/f:date_ne_timestamp"
	},
	{ "date_lt_timestamp",
	  2, {DATEOID, TIMESTAMPOID},
	  2, "t/f:date_lt_timestamp"
	},
	{ "date_le_timestamp",
	  2, {DATEOID, TIMESTAMPOID},
	  2, "t/f:date_le_timestamp"
	},
	{ "date_gt_timestamp",
	  2, {DATEOID, TIMESTAMPOID},
	  2, "t/f:date_gt_timestamp"
	},
	{ "date_ge_timestamp",
	  2, {DATEOID, TIMESTAMPOID},
	  2, "t/f:date_ge_timestamp"
	},
	{ "date_cmp_timestamp",
	  2, {DATEOID, TIMESTAMPOID},
	  2, "t/f:date_cmp_timestamp"
	},
	/* comparison between time */
	{ "time_eq", 2, {TIMEOID, TIMEOID}, 2, "t/f:time_eq" },
	{ "time_ne", 2, {TIMEOID, TIMEOID}, 2, "t/f:time_ne" },
	{ "time_lt", 2, {TIMEOID, TIMEOID}, 2, "t/f:time_lt"  },
	{ "time_le", 2, {TIMEOID, TIMEOID}, 2, "t/f:time_le" },
	{ "time_gt", 2, {TIMEOID, TIMEOID}, 2, "t/f:time_gt"  },
	{ "time_ge", 2, {TIMEOID, TIMEOID}, 2, "t/f:time_ge" },
	{ "time_cmp",2, {TIMEOID, TIMEOID}, 2, "t/f:type_compare" },
	/* comparison between timetz */
	{ "timetz_eq", 2, {TIMETZOID, TIMETZOID}, 1, "t/f:timetz_eq" },
	{ "timetz_ne", 2, {TIMETZOID, TIMETZOID}, 1, "t/f:timetz_ne" },
	{ "timetz_lt", 2, {TIMETZOID, TIMETZOID}, 1, "t/f:timetz_lt" },
	{ "timetz_le", 2, {TIMETZOID, TIMETZOID}, 1, "t/f:timetz_le" },
	{ "timetz_ge", 2, {TIMETZOID, TIMETZOID}, 1, "t/f:timetz_ge" },
	{ "timetz_gt", 2, {TIMETZOID, TIMETZOID}, 1, "t/f:timetz_gt" },
	{ "timetz_cmp",2, {TIMETZOID, TIMETZOID}, 1, "t/f:timetz_cmp" },
	/* comparison between timestamp */
	{ "timestamp_eq", 2, {TIMESTAMPOID, TIMESTAMPOID}, 1, "t/f:timestamp_eq" },
	{ "timestamp_ne", 2, {TIMESTAMPOID, TIMESTAMPOID}, 1, "t/f:timestamp_ne" },
	{ "timestamp_lt", 2, {TIMESTAMPOID, TIMESTAMPOID}, 1, "t/f:timestamp_lt" },
	{ "timestamp_le", 2, {TIMESTAMPOID, TIMESTAMPOID}, 1, "t/f:timestamp_le" },
	{ "timestamp_gt", 2, {TIMESTAMPOID, TIMESTAMPOID}, 1, "t/f:timestamp_gt" },
	{ "timestamp_ge", 2, {TIMESTAMPOID, TIMESTAMPOID}, 1, "t/f:timestamp_ge" },
	{ "timestamp_cmp", 2, {TIMESTAMPOID, TIMESTAMPOID},1, "t/f:timestamp_cmp"},
	/* comparison of timestamp and date */
	{ "timestamp_eq_date",
	  2, {TIMESTAMPOID, DATEOID},
	  3, "t/f:timestamp_eq_date"
	},
	{ "timestamp_ne_date",
	  2, {TIMESTAMPOID, DATEOID},
	  3, "t/f:timestamp_ne_date"
	},
	{ "timestamp_lt_date",
	  2, {TIMESTAMPOID, DATEOID},
	  3, "t/f:timestamp_lt_date"
	},
	{ "timestamp_le_date",
	  2, {TIMESTAMPOID, DATEOID},
	  3, "t/f:timestamp_le_date"
	},
	{ "timestamp_gt_date",
	  2, {TIMESTAMPOID, DATEOID},
	  3, "t/f:timestamp_gt_date"
	},
	{ "timestamp_ge_date",
	  2, {TIMESTAMPOID, DATEOID},
	  3, "t/f:timestamp_ge_date"
	},
	{ "timestamp_cmp_date",
	  2, {TIMESTAMPOID, DATEOID},
	  3, "t/f:timestamp_cmp_date"
	},
	/* comparison between timestamptz */
	{ "timestamptz_eq",
	  2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, "t/f:timestamptz_eq"
	},
	{ "timestamptz_ne",
	  2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, "t/f:timestamptz_ne"
	},
	{ "timestamptz_lt",
	  2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, "t/f:timestamptz_lt"
	},
	{ "timestamptz_le",
	  2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, "t/f:timestamptz_le"
	},
	{ "timestamptz_gt",
	  2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, "t/f:timestamptz_gt"
	},
	{ "timestamptz_ge",
	  2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, "t/f:timestamptz_ge"
	},
	{ "timestamptz_cmp",
	  2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, "t/f:type_compare"
	},
	/* comparison between date and timestamptz */
	{ "date_lt_timestamptz",
	  2, {DATEOID, TIMESTAMPTZOID},
	  3, "t/f:date_lt_timestamptz"
	},
	{ "date_le_timestamptz",
	  2, {DATEOID, TIMESTAMPTZOID},
	  3, "t/f:date_le_timestamptz"
	},
	{ "date_eq_timestamptz",
	  2, {DATEOID, TIMESTAMPTZOID},
	  3, "t/f:date_eq_timestamptz"
	},
	{ "date_ge_timestamptz",
	  2, {DATEOID, TIMESTAMPTZOID},
	  3, "t/f:date_ge_timestamptz"
	},
	{ "date_gt_timestamptz",
	  2, {DATEOID, TIMESTAMPTZOID},
	  3, "t/f:date_gt_timestamptz"
	},
	{ "date_ne_timestamptz",
	  2, {DATEOID, TIMESTAMPTZOID},
	  3, "t/f:date_ne_timestamptz"
	},

	/* comparison between timestamptz and date */
	{ "timestamptz_lt_date",
	  2, {TIMESTAMPTZOID, DATEOID},
	  3, "t/f:timestamptz_lt_date"
	},
	{ "timestamptz_le_date",
	  2, {TIMESTAMPTZOID, DATEOID},
	  3, "t/f:timestamptz_le_date"
	},
	{ "timestamptz_eq_date",
	  2, {TIMESTAMPTZOID, DATEOID},
	  3, "t/f:timestamptz_eq_date"
	},
	{ "timestamptz_ge_date",
	  2, {TIMESTAMPTZOID, DATEOID},
	  3, "t/f:timestamptz_ge_date"
	},
	{ "timestamptz_gt_date",
	  2, {TIMESTAMPTZOID, DATEOID},
	  3, "t/f:timestamptz_gt_date"
	},
	{ "timestamptz_ne_date",
	  2, {TIMESTAMPTZOID, DATEOID},
	  3, "t/f:timestamptz_ne_date"
	},
	/* comparison between timestamp and timestamptz  */
	{ "timestamp_lt_timestamptz",
	  2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, "t/f:timestamp_lt_timestamptz"
	},
	{ "timestamp_le_timestamptz",
	  2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, "t/f:timestamp_le_timestamptz"
	},
	{ "timestamp_eq_timestamptz",
	  2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, "t/f:timestamp_eq_timestamptz"
	},
	{ "timestamp_ge_timestamptz",
	  2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, "t/f:timestamp_ge_timestamptz"
	},
	{ "timestamp_gt_timestamptz",
	  2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, "t/f:timestamp_gt_timestamptz"
	},
	{ "timestamp_ne_timestamptz",
	  2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, "t/f:timestamp_ne_timestamptz"
	},
	/* comparison between timestamptz and timestamp  */
	{ "timestamptz_lt_timestamp",
	  2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, "t/f:timestamptz_lt_timestamp"
	},
	{ "timestamptz_le_timestamp",
	  2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, "t/f:timestamptz_le_timestamp"
	},
	{ "timestamptz_eq_timestamp",
	  2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, "t/f:timestamptz_eq_timestamp"
	},
	{ "timestamptz_ge_timestamp",
	  2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, "t/f:timestamptz_ge_timestamp"
	},
	{ "timestamptz_gt_timestamp",
	  2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, "t/f:timestamptz_gt_timestamp"
	},
	{ "timestamptz_ne_timestamp",
	  2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, "t/f:timestamptz_ne_timestamp"
	},
	/* comparison between intervals */
	{ "interval_eq", 2, {INTERVALOID, INTERVALOID}, 2, "t/f:interval_eq" },
	{ "interval_ne", 2, {INTERVALOID, INTERVALOID}, 2, "t/f:interval_ne" },
	{ "interval_lt", 2, {INTERVALOID, INTERVALOID}, 2, "t/f:interval_lt" },
	{ "interval_le", 2, {INTERVALOID, INTERVALOID}, 2, "t/f:interval_le" },
	{ "interval_ge", 2, {INTERVALOID, INTERVALOID}, 2, "t/f:interval_ge" },
	{ "interval_gt", 2, {INTERVALOID, INTERVALOID}, 2, "t/f:interval_gt" },
	{ "interval_cmp",2, {INTERVALOID, INTERVALOID}, 2, "t/f:interval_cmp" },

	/* overlaps() */
	{ "overlaps",
	  4, {TIMEOID, TIMEOID, TIMEOID, TIMEOID},
	  20, "t/f:overlaps_time"
	},
	{ "overlaps",
	  4, {TIMETZOID, TIMETZOID, TIMETZOID, TIMETZOID},
	  20, "t/f:overlaps_timetz"
	},
	{ "overlaps",
	  4, {TIMESTAMPOID, TIMESTAMPOID, TIMESTAMPOID, TIMESTAMPOID},
	  20, "t/f:overlaps_timestamp"
	},
	{ "overlaps",
	  4, {TIMESTAMPTZOID, TIMESTAMPTZOID, TIMESTAMPTZOID, TIMESTAMPTZOID},
	  20, "t/f:overlaps_timestamptz"
	},
	/* extract() */
	{ "date_part", 2, {TEXTOID,TIMESTAMPOID},  100, "t/f:extract_timestamp"},
	{ "date_part", 2, {TEXTOID,TIMESTAMPTZOID},100, "t/f:extract_timestamptz"},
	{ "date_part", 2, {TEXTOID,INTERVALOID},   100, "t/f:extract_interval"},
	{ "date_part", 2, {TEXTOID,TIMETZOID},     100, "t/f:extract_timetz"},
	{ "date_part", 2, {TEXTOID,TIMEOID},       100, "t/f:extract_time"},

	/* other time and data functions */
	{ "now", 0, {}, 1, "t/f:now" },

	/* macaddr functions */
	{ "trunc",       1, {MACADDROID},		8, "m/f:macaddr_trunc" },
	{ "macaddr_not", 1, {MACADDROID},		8, "m/f:macaddr_not" },
	{ "macaddr_and", 2, {MACADDROID,MACADDROID}, 8, "m/f:macaddr_and" },
	{ "macaddr_or",  2, {MACADDROID,MACADDROID}, 8, "m/f:macaddr_or" },

	/* inet/cidr functions */
	{ "set_masklen", 2, {INETOID,INT4OID},	8, "m/f:inet_set_masklen" },
	{ "set_masklen", 2, {CIDROID,INT4OID},	8, "m/f:cidr_set_masklen" },
	{ "family",      1, {INETOID},			8, "m/f:inet_family" },
	{ "network",     1, {INETOID},			8, "m/f:network_network" },
	{ "netmask",     1, {INETOID},			8, "m/f:inet_netmask" },
	{ "masklen",     1, {INETOID},			8, "m/f:inet_masklen" },
	{ "broadcast",   1, {INETOID},			8, "m/f:inet_broadcast" },
	{ "hostmask",    1, {INETOID},			8, "m/f:inet_hostmask" },
	{ "cidr",        1, {INETOID},			8, "m/f:inet_to_cidr" },
	{ "inetnot",     1, {INETOID},			8, "m/f:inet_not" },
	{ "inetand",     2, {INETOID,INETOID},	8, "m/f:inet_and" },
	{ "inetor",      2, {INETOID,INETOID},	8, "m/f:inet_or" },
	{ "inetpl",      2, {INETOID,INT8OID},	8, "m/f:inetpl_int8" },
	{ "inetmi_int8", 2, {INETOID,INT8OID},	8, "m/f:inetmi_int8" },
	{ "inetmi",      2, {INETOID,INETOID},	8, "m/f:inetmi" },
	{ "inet_same_family", 2, {INETOID,INETOID}, 8, "m/f:inet_same_family" },
	{ "inet_merge",  2, {INETOID,INETOID},	8, "m/f:inet_merge" },

	/*
	 * Text functions
	 */
	{ "bpchareq",  2, {BPCHAROID,BPCHAROID},  200, "s/f:bpchareq" },
	{ "bpcharne",  2, {BPCHAROID,BPCHAROID},  200, "s/f:bpcharne" },
	{ "bpcharlt",  2, {BPCHAROID,BPCHAROID},  200, "sL/f:bpcharlt" },
	{ "bpcharle",  2, {BPCHAROID,BPCHAROID},  200, "sL/f:bpcharle" },
	{ "bpchargt",  2, {BPCHAROID,BPCHAROID},  200, "sL/f:bpchargt" },
	{ "bpcharge",  2, {BPCHAROID,BPCHAROID},  200, "sL/f:bpcharge" },
	{ "bpcharcmp", 2, {BPCHAROID, BPCHAROID}, 200, "sL/f:type_compare"},
	{ "length",    1, {BPCHAROID},              2, "sL/f:bpcharlen"},
	{ "texteq",    2, {TEXTOID, TEXTOID},     200, "s/f:texteq" },
	{ "textne",    2, {TEXTOID, TEXTOID},     200, "s/f:textne" },
	{ "text_lt",   2, {TEXTOID, TEXTOID},     200, "sL/f:text_lt" },
	{ "text_le",   2, {TEXTOID, TEXTOID},     200, "sL/f:text_le" },
	{ "text_gt",   2, {TEXTOID, TEXTOID},     200, "sL/f:text_gt" },
	{ "text_ge",   2, {TEXTOID, TEXTOID},     200, "sL/f:text_ge" },
	{ "bttextcmp", 2, {TEXTOID, TEXTOID},     200, "sL/f:type_compare" },
	/* LIKE operators */
	{ "like",        2, {TEXTOID, TEXTOID},    9999, "s/f:textlike" },
	{ "textlike",    2, {TEXTOID, TEXTOID},    9999, "s/f:textlike" },
	{ "bpcharlike",  2, {BPCHAROID, TEXTOID},  9999, "s/f:bpcharlike" },
	{ "notlike",     2, {TEXTOID, TEXTOID},    9999, "s/f:textnlike" },
	{ "textnlike",   2, {TEXTOID, TEXTOID},    9999, "s/f:textnlike" },
	{ "bpcharnlike", 2, {BPCHAROID, TEXTOID},  9999, "s/f:bpcharnlike" },
	/* ILIKE operators */
	{ "texticlike",    2, {TEXTOID, TEXTOID},  9999, "Ls/f:texticlike" },
	{ "bpchariclike",  2, {TEXTOID, TEXTOID},  9999, "Ls/f:bpchariclike" },
	{ "texticnlike",   2, {TEXTOID, TEXTOID},  9999, "Ls/f:texticnlike" },
	{ "bpcharicnlike", 2, {BPCHAROID, TEXTOID},9999, "Ls/f:bpcharicnlike" },
	/* string operations */
	{ "length",		1, {TEXTOID},                 2, "s/f:textlen" },
	{ "textcat",	2, {TEXTOID,TEXTOID},
	  999, "Cs/f:textcat",
	  vlbuf_estimate_textcat
	},
	{ "concat",     2, {TEXTOID,TEXTOID},
	  999, "Cs/f:text_concat2",
	  vlbuf_estimate_textcat
	},
	{ "concat",     3, {TEXTOID,TEXTOID,TEXTOID},
	  999, "Cs/f:text_concat3",
	  vlbuf_estimate_textcat
	},
	{ "concat",     4, {TEXTOID,TEXTOID,TEXTOID,TEXTOID},
	  999, "Cs/f:text_concat4",
	  vlbuf_estimate_textcat
	},
	{ "substr",		3, {TEXTOID,INT4OID,INT4OID},
	  10, "Cs/f:text_substring",
	  vlbuf_estimate_substring },
	{ "substring",	3, {TEXTOID,INT4OID,INT4OID},
	  10, "Cs/f:text_substring",
	  vlbuf_estimate_substring },
	{ "substr",		2, {TEXTOID,INT4OID},
	  10, "Cs/f:text_substring_nolen",
	  vlbuf_estimate_substring },
	{ "substring",	2, {TEXTOID,INT4OID},
	  10, "Cs/f:text_substring_nolen",
	  vlbuf_estimate_substring },
	/* jsonb operators */
	{ "jsonb_object_field",       2, {JSONBOID,TEXTOID},
	  1000, "jC/f:jsonb_object_field",
	  vlbuf_estimate_jsonb
	},
	{ "jsonb_object_field_text",  2, {JSONBOID,TEXTOID},
	  1000, "jC/f:jsonb_object_field_text",
	  vlbuf_estimate_jsonb
	},
	{ "jsonb_array_element",      2, {JSONBOID,INT4OID},
	  1000, "jC/f:jsonb_array_element",
	  vlbuf_estimate_jsonb
	},
	{ "jsonb_array_element_text", 2, {JSONBOID,INT4OID},
	  1000, "jC/f:jsonb_array_element_text",
	  vlbuf_estimate_jsonb
	},
	{ "jsonb_exists",             1, {JSONBOID,TEXTOID},
	  100, "j/f:jsonb_exists"
	},
};

/*
 * device function catalog for extra SQL functions
 */
typedef struct devfunc_extra_catalog_t {
	const char *func_rettype;
	const char *func_signature;
	int			func_devcost;
	const char *func_template;
	devfunc_result_sz_type devfunc_result_sz;
} devfunc_extra_catalog_t;

#define BOOL    "boolean"
#define INT2	"smallint"
#define INT4	"integer"
#define INT8	"bigint"
#define FLOAT2	"float2"
#define FLOAT4	"real"
#define FLOAT8	"double precision"
#define NUMERIC	"numeric"

static devfunc_extra_catalog_t devfunc_extra_catalog[] = {
	/* float2 - type cast functions */
	{ FLOAT4,  "pgstrom.float4("FLOAT2")",  2, "f:to_float4" },
	{ FLOAT8,  "pgstrom.float8("FLOAT2")",  2, "f:to_float8" },
	{ INT2,    "pgstrom.int2("FLOAT2")",    2, "f:to_int2" },
	{ INT4,    "pgstrom.int4("FLOAT2")",    2, "f:to_int4" },
	{ INT8,    "pgstrom.int8("FLOAT2")",    2, "f:to_int8" },
	{ NUMERIC, "pgstrom.numeric("FLOAT2")", 2, "f:float2_numeric" },
	{ FLOAT2,  "pgstrom.float2("FLOAT4")",  2, "f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("FLOAT8")",  2, "f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT2")",    2, "f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT4")",    2, "f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT8")",    2, "f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("NUMERIC")", 2, "f:numeric_float2" },
	/* float2 - type comparison functions */
	{ BOOL,    "pgstrom.float2_eq("FLOAT2","FLOAT2")",  2, "f:float2eq" },
	{ BOOL,    "pgstrom.float2_ne("FLOAT2","FLOAT2")",  2, "f:float2ne" },
	{ BOOL,    "pgstrom.float2_lt("FLOAT2","FLOAT2")",  2, "f:float2lt" },
	{ BOOL,    "pgstrom.float2_le("FLOAT2","FLOAT2")",  2, "f:float2le" },
	{ BOOL,    "pgstrom.float2_gt("FLOAT2","FLOAT2")",  2, "f:float2gt" },
	{ BOOL,    "pgstrom.float2_ge("FLOAT2","FLOAT2")",  2, "f:float2ge" },
	{ BOOL,    "pgstrom.float2_cmp("FLOAT2","FLOAT2")", 2, "f:type_compare" },

	{ BOOL,    "pgstrom.float42_eq("FLOAT4","FLOAT2")", 2, "f:float42eq" },
	{ BOOL,    "pgstrom.float42_ne("FLOAT4","FLOAT2")", 2, "f:float42ne" },
	{ BOOL,    "pgstrom.float42_lt("FLOAT4","FLOAT2")", 2, "f:float42lt" },
	{ BOOL,    "pgstrom.float42_le("FLOAT4","FLOAT2")", 2, "f:float42le" },
	{ BOOL,    "pgstrom.float42_gt("FLOAT4","FLOAT2")", 2, "f:float42gt" },
	{ BOOL,    "pgstrom.float42_ge("FLOAT4","FLOAT2")", 2, "f:float42ge" },
	{ BOOL,    "pgstrom.float42_cmp("FLOAT4","FLOAT2")",2, "f:type_compare" },

	{ BOOL,    "pgstrom.float82_eq("FLOAT8","FLOAT2")", 2, "f:float82eq" },
	{ BOOL,    "pgstrom.float82_ne("FLOAT8","FLOAT2")", 2, "f:float82ne" },
	{ BOOL,    "pgstrom.float82_lt("FLOAT8","FLOAT2")", 2, "f:float82lt" },
	{ BOOL,    "pgstrom.float82_le("FLOAT8","FLOAT2")", 2, "f:float82le" },
	{ BOOL,    "pgstrom.float82_gt("FLOAT8","FLOAT2")", 2, "f:float82gt" },
	{ BOOL,    "pgstrom.float82_ge("FLOAT8","FLOAT2")", 2, "f:float82ge" },
	{ BOOL,    "pgstrom.float82_cmp("FLOAT8","FLOAT2")",2, "f:type_compare" },

	{ BOOL,    "pgstrom.float24_eq("FLOAT2","FLOAT4")", 2, "f:float24eq" },
	{ BOOL,    "pgstrom.float24_ne("FLOAT2","FLOAT4")", 2, "f:float24ne" },
	{ BOOL,    "pgstrom.float24_lt("FLOAT2","FLOAT4")", 2, "f:float24lt" },
	{ BOOL,    "pgstrom.float24_le("FLOAT2","FLOAT4")", 2, "f:float24le" },
	{ BOOL,    "pgstrom.float24_gt("FLOAT2","FLOAT4")", 2, "f:float24gt" },
	{ BOOL,    "pgstrom.float24_ge("FLOAT2","FLOAT4")", 2, "f:float24ge" },
	{ BOOL,    "pgstrom.float24_cmp("FLOAT2","FLOAT4")",2, "f:type_compare" },

	{ BOOL,    "pgstrom.float28_eq("FLOAT2","FLOAT8")", 2, "f:float28eq" },
	{ BOOL,    "pgstrom.float28_ne("FLOAT2","FLOAT8")", 2, "f:float28ne" },
	{ BOOL,    "pgstrom.float28_lt("FLOAT2","FLOAT8")", 2, "f:float28lt" },
	{ BOOL,    "pgstrom.float28_le("FLOAT2","FLOAT8")", 2, "f:float28le" },
	{ BOOL,    "pgstrom.float28_gt("FLOAT2","FLOAT8")", 2, "f:float28gt" },
	{ BOOL,    "pgstrom.float28_ge("FLOAT2","FLOAT8")", 2, "f:float28ge" },
	{ BOOL,    "pgstrom.float28_cmp("FLOAT2","FLOAT8")",2, "f:type_compare" },

	/* float2 - unary operator */
	{ FLOAT2,  "pgstrom.float2_up("FLOAT2")",	2, "p/f:float2up" },
	{ FLOAT2,  "pgstrom.float2_um("FLOAT2")",	2, "p/f:float2um" },
	{ FLOAT2,  "abs("FLOAT2")",					2, "p/f:float2abs" },

	/* float2 - arithmetic operators */
	{ FLOAT4, "pgstrom.float2_pl("FLOAT2","FLOAT2")",   2, "p/f:float2pl" },
	{ FLOAT4, "pgstrom.float2_mi("FLOAT2","FLOAT2")",   2, "p/f:float2mi" },
	{ FLOAT4, "pgstrom.float2_mul("FLOAT2","FLOAT2")",  3, "p/f:float2mul" },
	{ FLOAT4, "pgstrom.float2_div("FLOAT2","FLOAT2")",  3, "p/f:float2div" },
	{ FLOAT4, "pgstrom.float24_pl("FLOAT2","FLOAT4")",  2, "p/f:float24pl" },
	{ FLOAT4, "pgstrom.float24_mi("FLOAT2","FLOAT4")",  2, "p/f:float24mi" },
	{ FLOAT4, "pgstrom.float24_mul("FLOAT2","FLOAT4")", 3, "p/f:float24mul" },
	{ FLOAT4, "pgstrom.float24_div("FLOAT2","FLOAT4")", 3, "p/f:float24div" },
	{ FLOAT8, "pgstrom.float28_pl("FLOAT2","FLOAT8")",  2, "p/f:float28pl" },
	{ FLOAT8, "pgstrom.float28_mi("FLOAT2","FLOAT8")",  2, "p/f:float28mi" },
	{ FLOAT8, "pgstrom.float28_mul("FLOAT2","FLOAT8")", 3, "p/f:float28mul" },
	{ FLOAT8, "pgstrom.float28_div("FLOAT2","FLOAT8")", 3, "p/f:float28div" },
	{ FLOAT4, "pgstrom.float42_pl("FLOAT4","FLOAT2")",  2, "p/f:float42pl" },
	{ FLOAT4, "pgstrom.float42_mi("FLOAT4","FLOAT2")",  2, "p/f:float42mi" },
	{ FLOAT4, "pgstrom.float42_mul("FLOAT4","FLOAT2")", 3, "p/f:float42mul" },
	{ FLOAT4, "pgstrom.float42_div("FLOAT4","FLOAT2")", 3, "p/f:float42div" },
	{ FLOAT8, "pgstrom.float82_pl("FLOAT8","FLOAT2")",  2, "p/f:float82pl" },
	{ FLOAT8, "pgstrom.float82_mi("FLOAT8","FLOAT2")",  2, "p/f:float82mi" },
	{ FLOAT8, "pgstrom.float82_mul("FLOAT8","FLOAT2")", 3, "p/f:float82mul" },
	{ FLOAT8, "pgstrom.float82_div("FLOAT8","FLOAT2")", 3, "p/f:float82div" },
	{ "money", "pgstrom.cash_mul_flt2(money,"FLOAT2")",
	  3, "m/f:cash_mul_flt2" },
	{ "money", "pgstrom.flt2_mul_cash("FLOAT2",money)",
	  3, "m/f:flt2_mul_cash" },
	{ "money", "pgstrom.cash_div_flt2(money,"FLOAT2")",
	  3, "m/f:cash_div_flt2" },

	/* int4range operators */
	{ INT4, "lower(int4range)",		2, "r/f:int4range_lower" },
	{ INT4, "upper(int4range)",		2, "r/f:int4range_upper" },
	{ BOOL, "isempty(int4range)",	1, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(int4range)",	1, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int4range)",	1, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int4range)",	1, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int4range)",	1, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(int4range,int4range)",
	  2, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(int4range,int4range)",
	  2, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(int4range,int4range)",
	  2, "r/f:generic_range_lt" },
	{ BOOL, "range_le(int4range,int4range)",
	  2, "r/f:generic_range_le" },
	{ BOOL, "range_gt(int4range,int4range)",
	  2, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(int4range,int4range)",
	  2, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(int4range,int4range)",
	  2, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(int4range,int4range)",
	  4, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int4range,"INT4")",
	  4, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(int4range,int4range)",
	  4, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT4",int4range)",
	  4, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int4range,int4range)",
	  4, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int4range,int4range)",
	  4, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(int4range,int4range)",
	  4, "r/f:generic_range_before" },
	{ BOOL, "range_after(int4range,int4range)",
	  4, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(int4range,int4range)",
	  4, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(int4range,int4range)",
	  4, "r/f:generic_range_overleft" },
	{ "int4range", "range_union(int4range,int4range)",
	  4, "r/f:generic_range_union" },
	{ "int4range", "range_merge(int4range,int4range)",
	  4, "r/f:generic_range_merge" },
	{ "int4range", "range_intersect(int4range,int4range)",
	  4, "r/f:generic_range_intersect" },
	{ "int4range", "range_minus(int4range,int4range)",
	  4, "r/f:generic_range_minus" },

	/* int8range operators */
	{ INT8, "lower(int8range)",		2, "r/f:int8range_lower" },
	{ INT8, "upper(int8range)",		2, "r/f:int8range_upper" },
	{ BOOL, "isempty(int8range)",	1, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(int8range)",	1, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int8range)",	1, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int8range)",	1, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int8range)",	1, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(int8range,int8range)",
	  2, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(int8range,int8range)",
	  2, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(int8range,int8range)",
	  2, "r/f:generic_range_lt" },
	{ BOOL, "range_le(int8range,int8range)",
	  2, "r/f:generic_range_le" },
	{ BOOL, "range_gt(int8range,int8range)",
	  2, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(int8range,int8range)",
	  2, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(int8range,int8range)",
	  2, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(int8range,int8range)",
	  4, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int8range,"INT8")",
	  4, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(int8range,int8range)",
	  4, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT8",int8range)",
	  4, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int8range,int8range)",
	  4, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int8range,int8range)",
	  4, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(int8range,int8range)",
	  4, "r/f:generic_range_before" },
	{ BOOL, "range_after(int8range,int8range)",
	  4, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(int8range,int8range)",
	  4, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(int8range,int8range)",
	  4, "r/f:generic_range_overleft" },
	{ "int8range", "range_union(int8range,int8range)",
	  4, "r/f:generic_range_union" },
	{ "int8range", "range_merge(int8range,int8range)",
	  4, "r/f:generic_range_merge" },
	{ "int8range", "range_intersect(int8range,int8range)",
	  4, "r/f:generic_range_intersect" },
	{ "int8range", "range_minus(int8range,int8range)",
	  4, "r/f:generic_range_minus" },

	/* tsrange operators */
	{ "timestamp", "lower(tsrange)",	2, "r/f:tsrange_lower" },
	{ "timestamp", "upper(tsrange)",	2, "r/f:tsrange_upper" },
	{ BOOL, "isempty(tsrange)",		1, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(tsrange)",	1, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tsrange)",	1, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tsrange)",	1, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tsrange)",	1, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(tsrange,tsrange)",  2, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(tsrange,tsrange)",  2, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(tsrange,tsrange)",  2, "r/f:generic_range_lt" },
	{ BOOL, "range_le(tsrange,tsrange)",  2, "r/f:generic_range_le" },
	{ BOOL, "range_gt(tsrange,tsrange)",  2, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(tsrange,tsrange)",  2, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(tsrange,tsrange)", 2, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(tsrange,tsrange)",
	  4, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tsrange,timestamp)",
	  4, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(tsrange,tsrange)",
	  4, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamp,tsrange)",
	  4, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tsrange,tsrange)",
	  4, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tsrange,tsrange)",
	  4, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(tsrange,tsrange)",
	  4, "r/f:generic_range_before" },
	{ BOOL, "range_after(tsrange,tsrange)",
	  4, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(tsrange,tsrange)",
	  4, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(tsrange,tsrange)",
	  4, "r/f:generic_range_overleft" },
	{ "tsrange", "range_union(tsrange,tsrange)",
	  4, "r/f:generic_range_union" },
	{ "tsrange", "range_merge(tsrange,tsrange)",
	  4, "r/f:generic_range_merge" },
	{ "tsrange", "range_intersect(tsrange,tsrange)",
	  4, "r/f:generic_range_intersect" },
	{ "tsrange", "range_minus(tsrange,tsrange)",
	  4, "r/f:generic_range_minus" },

	/* tstzrange operators */
	{ "timestamptz", "lower(tstzrange)",
	  2, "r/f:tstzrange_lower" },
	{ "timestamptz", "upper(tstzrange)",
	  2, "r/f:tstzrange_upper" },
	{ BOOL, "isempty(tstzrange)",
	  1, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(tstzrange)",
	  1, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tstzrange)",
	  1, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tstzrange)",
	  1, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tstzrange)",
	  1, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(tstzrange,tstzrange)",
	  2, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(tstzrange,tstzrange)",
	  2, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(tstzrange,tstzrange)",
	  2, "r/f:generic_range_lt" },
	{ BOOL, "range_le(tstzrange,tstzrange)",
	  2, "r/f:generic_range_le" },
	{ BOOL, "range_gt(tstzrange,tstzrange)",
	  2, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(tstzrange,tstzrange)",
	  2, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(tstzrange,tstzrange)",
	  2, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(tstzrange,tstzrange)",
	  4, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tstzrange,timestamptz)",
	  4, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(tstzrange,tstzrange)",
	  4, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamptz,tstzrange)",
	  4, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tstzrange,tstzrange)",
	  4, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tstzrange,tstzrange)",
	  4, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(tstzrange,tstzrange)",
	  4, "r/f:generic_range_before" },
	{ BOOL, "range_after(tstzrange,tstzrange)",
	  4, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(tstzrange,tstzrange)",
	  4, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(tstzrange,tstzrange)",
	  4, "r/f:generic_range_overleft" },
	{ "tstzrange", "range_union(tstzrange,tstzrange)",
	  4, "r/f:generic_range_union" },
	{ "tstzrange", "range_merge(tstzrange,tstzrange)",
	  4, "r/f:generic_range_merge" },
	{ "tstzrange", "range_intersect(tstzrange,tstzrange)",
	  4, "r/f:generic_range_intersect" },
	{ "tstzrange", "range_minus(tstzrange,tstzrange)",
	  4, "r/f:generic_range_minus" },

	/* daterange operators */
	{ "date", "lower(daterange)",	2, "r/f:daterange_lower" },
	{ "date", "upper(daterange)",	2, "r/f:daterange_upper" },
	{ BOOL, "isempty(daterange)",	1, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(daterange)",	1, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(daterange)",	1, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(daterange)",	1, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(daterange)",	1, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(daterange,daterange)",
	  4, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(daterange,daterange)",
	  4, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(daterange,daterange)",
	  4, "r/f:generic_range_lt" },
	{ BOOL, "range_le(daterange,daterange)",
	  4, "r/f:generic_range_le" },
	{ BOOL, "range_gt(daterange,daterange)",
	  4, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(daterange,daterange)",
	  4, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(daterange,daterange)",
	  4, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(daterange,daterange)",
	  4, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(daterange,date)",
	  4, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(daterange,daterange)",
	  4, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(date,daterange)",
	  4, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(daterange,daterange)",
	  4, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(daterange,daterange)",
	  4, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(daterange,daterange)",
	  4, "r/f:generic_range_before" },
	{ BOOL, "range_after(daterange,daterange)",
	  4, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(daterange,daterange)",
	  4, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(daterange,daterange)",
	  4, "r/f:generic_range_overleft" },
	{ "daterange", "range_union(daterange,daterange)",
	  4, "r/f:generic_range_union" },
	{ "daterange", "range_merge(daterange,daterange)",
	  4, "r/f:generic_range_merge" },
	{ "daterange", "range_intersect(daterange,daterange)",
	  4, "r/f:generic_range_intersect" },
	{ "daterange", "range_minus(daterange,daterange)",
	  4, "r/f:generic_range_minus" },

	/* type re-interpretation */
	{ INT8,   "as_int8("FLOAT8")", 1, "p/f:as_int8" },
	{ INT4,   "as_int4("FLOAT4")", 1, "p/f:as_int4" },
	{ INT2,   "as_int2("FLOAT2")", 1, "p/f:as_int2" },
	{ FLOAT8, "as_float8("INT8")", 1, "p/f:as_float8" },
	{ FLOAT4, "as_float4("INT4")", 1, "p/f:as_float4" },
	{ FLOAT2, "as_float2("INT2")", 1, "p/f:as_float2" },
};

#undef BOOL
#undef INT2
#undef INT4
#undef INT8
#undef FLOAT2
#undef FLOAT4
#undef FLOAT8
#undef NUMERIC

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
__construct_devfunc_info(HeapTuple protup,
						 Oid func_collid,
						 int func_nargs, Oid *func_argtypes,
						 int func_devcost,
						 const char *func_template,
						 devfunc_result_sz_type devfunc_result_sz)
{
	Form_pg_proc    proc = (Form_pg_proc) GETSTRUCT(protup);
	MemoryContext	oldcxt;
	devfunc_info   *dfunc = NULL;
	devtype_info   *dtype;
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
	for (j=0; j < func_nargs; j++)
	{
		dtype = pgstrom_devtype_lookup(func_argtypes[j]);
		if (!dtype)
			goto negative;
		dfunc_args = lappend(dfunc_args, dtype);
	}
	dtype = pgstrom_devtype_lookup(proc->prorettype);
	if (!dtype)
		goto negative;

	dfunc = palloc0(sizeof(devfunc_info));
	dfunc->func_oid = PgProcTupleGetOid(protup);
	if (has_collation)
	{
		if (OidIsValid(func_collid) && !lc_collate_is_c(func_collid))
			dfunc->func_is_negative = true;
		dfunc->func_collid = func_collid;
	}
	dfunc->func_is_strict = proc->proisstrict;
	dfunc->func_flags = flags;
	dfunc->func_args = dfunc_args;
	dfunc->func_rettype = dtype;
	dfunc->func_sqlname = pstrdup(NameStr(proc->proname));
	dfunc->func_devname = func_template + 2;	/* const cstring */
	dfunc->func_devcost = func_devcost;
	dfunc->devfunc_result_sz = (has_callbacks
								? devfunc_result_sz
								: devfunc_generic_result_sz);
	/* other fields shall be assigned on the caller side */
negative:
	MemoryContextSwitchTo(oldcxt);

	return dfunc;
}

static devfunc_info *
pgstrom_devfunc_construct_core(HeapTuple protup,
							   oidvector *func_argtypes,
							   Oid func_collid,
							   bool consider_relabel)
{
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	int				i, j;

	for (i=0; i < lengthof(devfunc_common_catalog); i++)
	{
		devfunc_catalog_t  *procat = devfunc_common_catalog + i;

		if (strcmp(procat->func_name, NameStr(proc->proname)) != 0 ||
			procat->func_nargs != func_argtypes->dim1)
			continue;

		for (j=0; j < procat->func_nargs; j++)
		{
			if (procat->func_argtypes[j] != func_argtypes->values[j])
			{
				if (!consider_relabel ||
					!pgstrom_devtype_can_relabel(func_argtypes->values[j],
												 procat->func_argtypes[j]))
					break;
			}
		}
		if (j == procat->func_nargs)
			return __construct_devfunc_info(protup,
											func_collid,
											procat->func_nargs,
											procat->func_argtypes,
											procat->func_devcost,
											procat->func_template,
											procat->devfunc_result_sz);
	}
	return NULL;	/* not found, to be negative entry */
}

/*
 * pgstrom_devfunc_construct_extra
 *
 * FIXME: Right now, we have no way to describe device function catalog
 * that accepts binary compatible arguments; like varchar(N) values on
 * text argument. Thus, FuncExpr must have exactly identical argument
 * list towards the device function definition.
 * So, this function has no 'consider_relabel' argument
 */
static devfunc_info *
pgstrom_devfunc_construct_extra(HeapTuple protup,
								oidvector *func_argtypes,
								Oid func_collid)
{
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	StringInfoData	sig;
	devfunc_info   *result = NULL;
	char		   *rettype;
	char		   *temp;
	int				i;

	/* make a signature string */
	initStringInfo(&sig);
	if (proc->pronamespace != PG_CATALOG_NAMESPACE)
	{
		temp = get_namespace_name(proc->pronamespace);
		appendStringInfo(&sig, "%s.", quote_identifier(temp));
		pfree(temp);
	}

	appendStringInfo(&sig, "%s(", quote_identifier(NameStr(proc->proname)));
	for (i=0; i < func_argtypes->dim1; i++)
	{
		Oid		argtype = func_argtypes->values[i];

		if (i > 0)
			appendStringInfoChar(&sig, ',');
		temp = format_type_be_qualified(argtype);
		if (strncmp(temp, "pg_catalog.", 11) == 0)
			appendStringInfo(&sig, "%s", temp + 11);
		else
			appendStringInfo(&sig, "%s", temp);
		pfree(temp);
	}
	appendStringInfoChar(&sig, ')');

	temp = format_type_be_qualified(proc->prorettype);
	if (strncmp(temp, "pg_catalog.", 11) == 0)
		rettype = temp + 11;
	else
		rettype = temp;

	for (i=0; i < lengthof(devfunc_extra_catalog); i++)
	{
		devfunc_extra_catalog_t  *procat = devfunc_extra_catalog + i;

		if (strcmp(procat->func_signature, sig.data) == 0 &&
			strcmp(procat->func_rettype, rettype) == 0)
		{
			result = __construct_devfunc_info(protup,
											  func_collid,
											  func_argtypes->dim1,
											  func_argtypes->values,
											  procat->func_devcost,
											  procat->func_template,
											  procat->devfunc_result_sz);
			goto found;
		}
	}
	elog(DEBUG2, "no extra function found for sig=[%s] rettype=[%s]",
		 sig.data, rettype);
found:
	pfree(sig.data);
	pfree(temp);
	return result;
}

static devfunc_info *
__pgstrom_devfunc_lookup_or_create(HeapTuple protup,
								   Oid func_rettype,
								   oidvector *func_argtypes,
								   Oid func_collid,
								   bool consider_relabel)
{
	Oid				func_oid = PgProcTupleGetOid(protup);
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	devfunc_info   *dfunc;
	devtype_info   *dtype;
	ListCell	   *lc;
	cl_uint			hashvalue;
	int				j, hindex;
	dlist_iter		iter;

	hashvalue = GetSysCacheHashValue(PROCOID, func_oid, 0, 0, 0);
	hindex = hashvalue % lengthof(devfunc_info_slot);
	dlist_foreach (iter, &devfunc_info_slot[hindex])
	{
		dfunc = dlist_container(devfunc_info, chain, iter.cur);

		if (dfunc->func_oid == func_oid &&
			list_length(dfunc->func_args) == func_argtypes->dim1 &&
			(!OidIsValid(dfunc->func_collid) ||
			 dfunc->func_collid == func_collid))
		{
			j = 0;
			foreach (lc, dfunc->func_args)
			{
				Oid		arg_type_oid = func_argtypes->values[j++];

				dtype = lfirst(lc);
				if (dtype->type_oid != arg_type_oid)
				{
					if (!consider_relabel ||
						!pgstrom_devtype_can_relabel(arg_type_oid,
													 dtype->type_oid))
						break;		/* not match */
				}
			}

			if (!lc)
			{
				if (dfunc->func_is_negative)
					return NULL;
				dtype = dfunc->func_rettype;
				if (dtype->type_oid != func_rettype &&
					!pgstrom_devtype_can_relabel(dtype->type_oid,
												 func_rettype))
					return NULL;

				return dfunc;	/* Ok, found */
			}
		}
	}

	/*
	 * Not cached, construct a new entry of the device function
	 */
	if (proc->prorettype != func_rettype &&
		!pgstrom_devtype_can_relabel(proc->prorettype, func_rettype))
	{
		elog(NOTICE, "Bug? function result type is not compatible");
		dfunc = NULL;
	}
	else if (proc->pronamespace == PG_CATALOG_NAMESPACE)
	{
		dfunc = pgstrom_devfunc_construct_core(protup,
											   func_argtypes,
											   func_collid,
											   consider_relabel);
	}
	else
	{
		dfunc = pgstrom_devfunc_construct_extra(protup,
												func_argtypes,
												func_collid);
	}

	/*
	 * Not found, so this function should be a nagative entry
	 */
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

static devfunc_info *
__pgstrom_devfunc_lookup(HeapTuple protup,
						 Oid func_rettype,
						 oidvector *func_argtypes,
						 Oid func_collid)
{
	devfunc_info   *dfunc;

	dfunc = __pgstrom_devfunc_lookup_or_create(protup,
											   func_rettype,
											   func_argtypes,
											   func_collid,
											   false);
	if (!dfunc)
		dfunc = __pgstrom_devfunc_lookup_or_create(protup,
												   func_rettype,
												   func_argtypes,
												   func_collid,
												   true);
	return dfunc;
}

static devfunc_info *
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
	devcast_coerceviaio_callback_f dcast_coerceviaio_callback;
} devcast_catalog[] = {
	/* text, varchar, bpchar */
	{ TEXTOID,    BPCHAROID,  NULL },
	{ TEXTOID,    VARCHAROID, NULL },
	{ VARCHAROID, TEXTOID,    NULL },
	{ VARCHAROID, BPCHAROID,  NULL },
	/* cidr -> inet, but no reverse type cast */
	{ CIDROID,    INETOID,    NULL },
	/* text -> (intX/floatX/numeric), including (jsonb->>'key') reference */
	{ TEXTOID,    BOOLOID,    devcast_text2numeric_callback },
	{ TEXTOID,    INT2OID,    devcast_text2numeric_callback },
	{ TEXTOID,    INT4OID,    devcast_text2numeric_callback },
	{ TEXTOID,    INT8OID,    devcast_text2numeric_callback },
	{ TEXTOID,    FLOAT4OID,  devcast_text2numeric_callback },
	{ TEXTOID,    FLOAT8OID,  devcast_text2numeric_callback },
	{ TEXTOID,    NUMERICOID, devcast_text2numeric_callback },
};

static devcast_info *
build_devcast_info(Oid src_type_oid, Oid dst_type_oid)
{
	int			i;

	for (i=0; i < lengthof(devcast_catalog); i++)
	{
		devcast_coerceviaio_callback_f dcast_callback;
		devtype_info   *dtype;
		devcast_info   *dcast;
		HeapTuple		tup;
		char			method;

		if (src_type_oid != devcast_catalog[i].src_type_oid ||
			dst_type_oid != devcast_catalog[i].dst_type_oid)
			continue;
		dcast_callback = devcast_catalog[i].dcast_coerceviaio_callback;

		tup = SearchSysCache2(CASTSOURCETARGET,
							  ObjectIdGetDatum(src_type_oid),
							  ObjectIdGetDatum(dst_type_oid));
		if (!HeapTupleIsValid(tup))
		{
			if (!dcast_callback)
				elog(ERROR, "Bug? type cast (%s -> %s) has wrong catalog item",
					 format_type_be(src_type_oid),
					 format_type_be(dst_type_oid));
			/*
			 * No pg_cast entry, so we assume this conversion is processed by
			 * the cstring in/out function.
			 */
			method = COERCION_METHOD_INOUT;
		}
		else
		{
			method = ((Form_pg_cast) GETSTRUCT(tup))->castmethod;
			ReleaseSysCache(tup);

			if (!((method == COERCION_METHOD_BINARY && !dcast_callback) ||
				  (method == COERCION_METHOD_INOUT && dcast_callback)))
				elog(ERROR, "Bug? type cast (%s -> %s) has wrong catalog item",
					 format_type_be(src_type_oid),
					 format_type_be(dst_type_oid));
		}
		dcast = MemoryContextAllocZero(devinfo_memcxt,
									   sizeof(devcast_info));
		dcast->hashvalue = GetSysCacheHashValue(CASTSOURCETARGET,
												src_type_oid,
												dst_type_oid, 0, 0);
		/* source */
		dtype = pgstrom_devtype_lookup(src_type_oid);
		if (!dtype)
			__ELog("Bug? type '%s' is not supported on device",
				   format_type_be(src_type_oid));
		dcast->src_type = dtype;
		/* destination */
		dtype = pgstrom_devtype_lookup(dst_type_oid);
		if (!dtype)
			__ELog("Bug? type '%s' is not supported on device",
				   format_type_be(dst_type_oid));
		dcast->dst_type = dtype;
		/* method && callback */
		dcast->castmethod = method;
		dcast->dcast_coerceviaio_callback = dcast_callback;

		return dcast;
	}
	return NULL;
}

devcast_info *
pgstrom_devcast_lookup(Oid src_type_oid,
					   Oid dst_type_oid,
					   char castmethod)
{

	int			hindex;
	devcast_info *dcast;
	dlist_iter	iter;

	hindex = GetSysCacheHashValue(CASTSOURCETARGET,
								  src_type_oid,
								  dst_type_oid,
								  0, 0) % lengthof(devcast_info_slot);
	dlist_foreach (iter, &devcast_info_slot[hindex])
	{
		dcast = dlist_container(devcast_info, chain, iter.cur);
		if (dcast->src_type->type_oid == src_type_oid &&
            dcast->dst_type->type_oid == dst_type_oid)
		{
			if (dcast->castmethod == castmethod)
				return dcast;
			return NULL;
		}
	}

	dcast = build_devcast_info(src_type_oid, dst_type_oid);
	if (dcast)
	{
		hindex = dcast->hashvalue % lengthof(devcast_info_slot);
		dlist_push_head(&devcast_info_slot[hindex], &dcast->chain);

		if (dcast->castmethod == castmethod)
			return dcast;
	}
	return NULL;
}

bool
pgstrom_devtype_can_relabel(Oid src_type_oid,
							Oid dst_type_oid)
{
	devcast_info *dcast = pgstrom_devcast_lookup(src_type_oid,
												 dst_type_oid,
												 COERCION_METHOD_BINARY);
	if (dcast)
	{
		Assert(!dcast->dcast_coerceviaio_callback);
		return true;
	}
	return false;
}

/*
 * codegen_expression_walker - main logic of run-time code generator
 */
static void codegen_expression_walker(codegen_context *context,
									  Node *node, int *p_varlena_sz);

static Node *__codegen_current_node = NULL;

#define __appendStringInfo(str,fmt,...)						\
	do {													\
		if ((str)->data)									\
			appendStringInfo((str),(fmt), ##__VA_ARGS__);	\
	} while(0)
#define __appendStringInfoChar(str,c)			\
	do {										\
		if ((str)->data)						\
			appendStringInfoChar((str),(c));	\
	} while(0)

static int
codegen_const_expression(codegen_context *context,
						 Const *con)
{
	cl_int		index;
	cl_int		width;

	if (!pgstrom_devtype_lookup_and_track(con->consttype, context))
		__ELog("type %s is not device supported",
			   format_type_be(con->consttype));

	context->used_params = lappend(context->used_params,
								   copyObject(con));
	index = list_length(context->used_params) - 1;
	__appendStringInfo(&context->str,
					   "KPARAM_%u", index);
	context->param_refs = bms_add_member(context->param_refs, index);
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
		{
			__appendStringInfo(&context->str,
							   "KPARAM_%u", index);
			context->param_refs = bms_add_member(context->param_refs, index);
			goto out;
		}
		index++;
	}
	context->used_params = lappend(context->used_params,
								   copyObject(param));
	index = list_length(context->used_params) - 1;
	__appendStringInfo(&context->str,
					   "KPARAM_%u", index);
	context->param_refs = bms_add_member(context->param_refs, index);
out:
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
codegen_varnode_expression(codegen_context *context, Var *var)
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
		__appendStringInfo(&context->str, "%s_S%u",
						   context->var_label,
						   -varattno);
	else
		__appendStringInfo(&context->str, "%s_%u",
						   context->var_label,
						   varattno);
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
							devfunc_info *dfunc, List *args)
{
	ListCell   *lc1, *lc2;
	Expr	   *fn_args[DEVFUNC_MAX_NARGS];
	int			vl_width[DEVFUNC_MAX_NARGS];
	int			index = 0;

	__appendStringInfo(&context->str,
					   "pgfn_%s(kcxt",
					   dfunc->func_devname);
	forboth (lc1, dfunc->func_args,
			 lc2, args)
	{
		devtype_info *dtype = lfirst(lc1);
		Node   *expr = lfirst(lc2);
		Oid		expr_type_oid = exprType(expr);

		__appendStringInfo(&context->str, ", ");

		if (dtype->type_oid == expr_type_oid)
			codegen_expression_walker(context, expr, &vl_width[index]);
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
			__appendStringInfo(&context->str, "to_%s(", dtype->type_name);
			codegen_expression_walker(context, expr, &vl_width[index]);
			__appendStringInfoChar(&context->str, ')');
		}
		else
		{
			__ELog("Bug? unsupported implicit type cast (%s)->(%s)",
				   format_type_be(expr_type_oid),
				   format_type_be(dtype->type_oid));
		}
		fn_args[index++] = (Expr *)expr;
	}
	__appendStringInfoChar(&context->str, ')');
	/* estimation of function result width */
	return dfunc->devfunc_result_sz(context, dfunc, fn_args, vl_width);
}

static int
codegen_nulltest_expression(codegen_context *context,
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
			__appendStringInfo(&context->str, "PG_ISNULL");
			break;
		case IS_NOT_NULL:
			__appendStringInfo(&context->str, "PG_ISNOTNULL");
			break;
		default:
			elog(ERROR, "unknown NullTestType: %d",
				 (int)nulltest->nulltesttype);
	}
	__appendStringInfo(&context->str, "(kcxt, ");
	codegen_expression_walker(context, (Node *) nulltest->arg, NULL);
	__appendStringInfoChar(&context->str, ')');
	context->devcost += 1;

	return sizeof(cl_bool);
}

static int
codegen_booleantest_expression(codegen_context *context,
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
	__appendStringInfo(&context->str, "pgfn_%s(kcxt, ", func_name);
	codegen_expression_walker(context, (Node *) booltest->arg, NULL);
	__appendStringInfoChar(&context->str, ')');
	context->devcost += 1;

	return sizeof(cl_bool);
}

static int
codegen_bool_expression(codegen_context *context, BoolExpr *b)
{
	Node	   *node;
	ListCell   *lc;
	int			varno;

	switch (b->boolop)
	{
		case NOT_EXPR:
			Assert(list_length(b->args) == 1);
			node = linitial(b->args);

			__appendStringInfo(&context->str, "NOT(");
			codegen_expression_walker(context, node, NULL);
			__appendStringInfoChar(&context->str, ')');
			break;
		case AND_EXPR:
		case OR_EXPR:
			Assert(list_length(b->args) > 1);
			varno = ++context->decl_count;
			__appendStringInfo(
				&context->decl_temp,
				"  pg_bool_t __temp%d __attribute__((unused));\n"
				"  cl_bool   __anynull%d __attribute__((unused)) = false;\n",
				varno, varno);

			foreach (lc, b->args)
			{
				Node   *node = lfirst(lc);

				Assert(exprType(node) == BOOLOID);
				__appendStringInfo(
					&context->str, "%s(__temp%d, __anynull%d, ",
					b->boolop == AND_EXPR ? "AND" : "OR",
					varno, varno);
				codegen_expression_walker(context, node, NULL);
				__appendStringInfo(&context->str, ", ");
			}
			__appendStringInfo(
				&context->str,
				"PG_BOOL(__anynull%d, %s)",
				varno, b->boolop == AND_EXPR ? "true" : "false");
			foreach (lc, b->args)
				__appendStringInfo(&context->str, ")");
			break;
		default:
			elog(ERROR, "unknown BoolExprType: %d", (int) b->boolop);
			break;
	}
	context->devcost += list_length(b->args);
	return sizeof(cl_bool);
}

static int
codegen_coalesce_expression(codegen_context *context,
							CoalesceExpr *coalesce)
{
	devtype_info   *dtype;
	ListCell	   *lc;
	int				temp_nr;
	int				maxlen = 0;

	dtype = pgstrom_devtype_lookup(coalesce->coalescetype);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(coalesce->coalescetype));
	temp_nr = ++context->decl_count;
	__appendStringInfo(
		&context->decl_temp,
		"  pg_%s_t __temp%d __attribute__((unused));\n",
		dtype->type_name, temp_nr);

	foreach (lc, coalesce->args)
	{
		Node   *expr = (Node *)lfirst(lc);
		Oid		type_oid = exprType(expr);
		int		width;

		if (dtype->type_oid != type_oid)
			__ELog("device type mismatch in COALESCE: %s / %s",
				   format_type_be(dtype->type_oid),
				   format_type_be(type_oid));
		if (lc->next != NULL)
		{
			__appendStringInfo(&context->str, "((__temp%d = ", temp_nr);
			codegen_expression_walker(context, expr, &width);
			__appendStringInfo(&context->str,
							   ").isnull == false ? __temp%d : ", temp_nr);
		}
		else
		{
			/* last item */
			__appendStringInfo(&context->str, "(__temp%d = ", temp_nr);
			codegen_expression_walker(context, expr, &width);
		}
		if (width < 0)
			maxlen = -1;
		else if (maxlen >= 0)
			maxlen = Max(maxlen, width);
		context->devcost += 1;
	}
	foreach (lc, coalesce->args)
		__appendStringInfo(&context->str, ")");

	return maxlen;
}

static int
codegen_minmax_expression(codegen_context *context,
						  MinMaxExpr *minmax)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	ListCell	   *lc;
	int				maxlen = 0;

	dtype = pgstrom_devtype_lookup(minmax->minmaxtype);
	if (!dtype)
		__ELog("type %s is not device supported",
			   format_type_be(minmax->minmaxtype));

	dfunc = pgstrom_devfunc_lookup_type_compare(dtype, minmax->inputcollid);
	if (!dfunc)
		__ELog("device type %s has no comparison operator",
			   format_type_be(minmax->minmaxtype));
	switch (minmax->op)
	{
		case IS_GREATEST:
			__appendStringInfo(&context->str, "PG_GREATEST(kcxt");
			break;
		case IS_LEAST:
			__appendStringInfo(&context->str, "PG_LEAST(kcxt");
			break;
		default:
			elog(ERROR, "unknown MinMaxOp: %d", (int)minmax->op);
			break;
	}

	foreach (lc, minmax->args)
	{
		Node   *expr = lfirst(lc);
		Oid		type_oid = exprType(expr);
		int		width;

		if (dtype->type_oid != type_oid)
			__ELog("device type mismatch in LEAST/GREATEST: %s / %s",
				   format_type_be(dtype->type_oid),
				   format_type_be(exprType(expr)));
		__appendStringInfo(&context->str, ", ");
		codegen_expression_walker(context, expr, &width);
		if (width < 0)
			maxlen = -1;
		else if (maxlen >= 0)
			maxlen = Max(maxlen, width);
		context->devcost += 1;
	}
	__appendStringInfoChar(&context->str, ')');

	return maxlen;
}

static int
codegen_relabel_expression(codegen_context *context,
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

	__appendStringInfo(&context->str, "to_%s(", dtype->type_name);
	codegen_expression_walker(context, (Node *)relabel->arg, &width);
	__appendStringInfoChar(&context->str, ')');

	return width;
}

static int
codegen_coerceviaio_expression(codegen_context *context,
							   CoerceViaIO *coerce)
{
	devcast_info   *dcast;
	Oid		stype_oid = exprType((Node *)coerce->arg);
	Oid		dtype_oid = coerce->resulttype;

	dcast = pgstrom_devcast_lookup(stype_oid,
								   dtype_oid,
								   COERCION_METHOD_INOUT);
	if (!dcast)
		__ELog("type cast (%s -> %s) is not device supported",
			   format_type_be(stype_oid),
			   format_type_be(dtype_oid));

	if (!dcast->dcast_coerceviaio_callback)
		__ELog("no device cast support on %s -> %s",
			   format_type_be(dcast->src_type->type_oid),
			   format_type_be(dcast->dst_type->type_oid));
	context->devcost += 8;		/* just a rough estimation */

	return dcast->dcast_coerceviaio_callback(context, dcast, coerce);
}

static int
codegen_casewhen_expression(codegen_context *context,
							CaseExpr *caseexpr)
{
	devtype_info   *rtype;	/* result type */
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	Node		   *defresult;
	ListCell	   *cell;
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

	if (caseexpr->arg)
	{
		int			temp_nr;
		int			count = 1;
		/* type compare function internally used */
		type_oid = exprType((Node *) caseexpr->arg);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			__ELog("type %s is not device supported",
				   format_type_be(type_oid));
		dfunc = pgstrom_devfunc_lookup_type_compare(dtype, InvalidOid);
		if (!dfunc)
			__ELog("type %s has no device executable compare-operator",
				   format_type_be(type_oid));
		pgstrom_devfunc_track(context, dfunc);

		temp_nr = ++context->decl_count;
		__appendStringInfo(
			&context->decl_temp,
			"  pg_%s_t __temp%d __attribute__((unused));\n",
			dtype->type_name, temp_nr);

		__appendStringInfo(
			&context->str,
			"(__temp%d = ", temp_nr);
		codegen_expression_walker(context,
								  (Node *)caseexpr->arg,
								  NULL);
		__appendStringInfo(&context->str, ", ");

		foreach (cell, caseexpr->args)
		{
			CaseWhen   *casewhen = (CaseWhen *) lfirst(cell);
			OpExpr	   *op_expr;
			Node	   *test_val;

			Assert(IsA(casewhen, CaseWhen));
			op_expr = (OpExpr *) casewhen->expr;
			if (!IsA(op_expr, OpExpr) ||
				op_expr->opresulttype != BOOLOID ||
                list_length(op_expr->args) != 2)
				elog(ERROR, "Bug? unexpected expression at CASE ... WHEN");

			if (IsA(linitial(op_expr->args), CaseTestExpr) &&
				!IsA(lsecond(op_expr->args), CaseTestExpr))
				test_val = lsecond(op_expr->args);
			else if (!IsA(linitial(op_expr->args), CaseTestExpr) &&
					 IsA(lsecond(op_expr->args), CaseTestExpr))
				test_val = linitial(op_expr->args);
			else
				elog(ERROR, "Bug? CaseTestExpr has unexpected arguments");

			__appendStringInfo(&context->str,
							   "(pgfn_type_equal(kcxt, __temp%d, ", temp_nr);
			codegen_expression_walker(context, test_val, NULL);
			__appendStringInfo(&context->str, ") ? (");
			codegen_expression_walker(context,
									  (Node *)casewhen->result,
									  &width);
			__appendStringInfo(&context->str, ") : ");
			if (width < 0)
				maxlen = -1;
			else if (maxlen >= 0)
				maxlen = Max(maxlen, width);
			context->devcost += 1;
			count++;
		}
		/* default value or NULL */
		codegen_expression_walker(context, defresult, &width);
		if (width < 0)
			maxlen = -1;
		else if (maxlen >= 0)
			maxlen = Max(maxlen, width);
		context->devcost += 1;

		foreach (cell, caseexpr->args)
			__appendStringInfo(&context->str, ")");
		__appendStringInfo(&context->str, ")");
	}
	else
	{
		foreach (cell, caseexpr->args)
		{
			CaseWhen   *casewhen = (CaseWhen *) lfirst(cell);

			Assert(exprType((Node *)casewhen->expr) == BOOLOID);
			Assert(exprType((Node *)casewhen->result) == rtype->type_oid);
			__appendStringInfo(&context->str, "EVAL(");
			codegen_expression_walker(context,
									  (Node *)casewhen->expr, NULL);
			__appendStringInfo(&context->str, ") ? (");
			codegen_expression_walker(context, (Node *)casewhen->result,
									  &width);
			if (width < 0)
				maxlen = -1;
			else if (maxlen >= 0)
				maxlen = Max(maxlen, width);
			__appendStringInfo(&context->str, ") : (");
		}
		codegen_expression_walker(context, defresult, &width);
		if (width < 0)
			maxlen = -1;
		else if (width >= 0)
			maxlen = Max(maxlen, width);
		context->devcost += 1;

		foreach (cell, caseexpr->args)
			__appendStringInfoChar(&context->str, ')');
	}
	return maxlen;
}

static int
codegen_scalar_array_op_expression(codegen_context *context,
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

	__appendStringInfo(&context->str,
					   "PG_SCALAR_ARRAY_OP(kcxt, pgfn_%s, ",
					   dfunc->func_devname);
	codegen_expression_walker(context, node_s, NULL);
	__appendStringInfo(&context->str, ", ");
	codegen_expression_walker(context, node_a, NULL);
	__appendStringInfo(&context->str, ", %s, %d, %d)",
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
			width = codegen_const_expression(context, (Const *) node);
			break;

		case T_Param:
			width = codegen_param_expression(context, (Param *) node);
			break;

		case T_Var:
			width = codegen_varnode_expression(context, (Var *) node);
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
													dfunc,
													op->args);
				context->devcost += dfunc->func_devcost;
			}
			break;

		case T_NullTest:
			width = codegen_nulltest_expression(context,
												(NullTest *) node);
			break;

		case T_BooleanTest:
			width = codegen_booleantest_expression(context,
												   (BooleanTest *) node);
			break;

		case T_BoolExpr:
			width = codegen_bool_expression(context,
											(BoolExpr *) node);
			break;

		case T_CoalesceExpr:
			width = codegen_coalesce_expression(context,
												(CoalesceExpr *) node);
			break;

		case T_MinMaxExpr:
			width = codegen_minmax_expression(context,
											  (MinMaxExpr *) node);
			break;

		case T_RelabelType:
			width = codegen_relabel_expression(context,
											   (RelabelType *) node);
			break;

		case T_CoerceViaIO:
			width = codegen_coerceviaio_expression(context,
												   (CoerceViaIO *) node);
			break;

		case T_CaseExpr:
			width = codegen_casewhen_expression(context,
												(CaseExpr *) node);
			break;

		case T_ScalarArrayOpExpr:
			width = codegen_scalar_array_op_expression(context,
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
	devtype_info   *dtype;

	if (!context->str.data)
		initStringInfo(&context->str);
	else
		resetStringInfo(&context->str);

	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = (Node *)linitial((List *)expr);
		else
			expr = (Node *)make_andclause((List *)expr);
	}

	PG_TRY();
	{
		codegen_expression_walker(context, expr, NULL);
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
		context->varlena_bufsz += MAXALIGN(dtype->extra_sz);

	return context->str.data;
}

/*
 * pgstrom_codegen_param_declarations
 */
void
pgstrom_codegen_param_declarations(StringInfo buf, codegen_context *context)
{
	ListCell	   *cell;
	devtype_info   *dtype;
	int				index = 0;

	foreach (cell, context->used_params)
	{
		Node	   *node = lfirst(cell);

		if (!bms_is_member(index, context->param_refs))
			goto lnext;

		if (IsA(node, Const))
		{
			Const  *con = (Const *)node;

			dtype = pgstrom_devtype_lookup(con->consttype);
			if (!dtype)
				__ELog("failed to lookup device type: %u",
					   con->consttype);

			appendStringInfo(
				buf,
				"  pg_%s_t KPARAM_%u = pg_%s_param(kcxt,%d);\n",
				dtype->type_name, index, dtype->type_name, index);
		}
		else if (IsA(node, Param))
		{
			Param  *param = (Param *)node;

			dtype = pgstrom_devtype_lookup(param->paramtype);
			if (!dtype)
				__ELog("failed to lookup device type: %u",
					   param->paramtype);

			appendStringInfo(
				buf,
				"  pg_%s_t KPARAM_%u = pg_%s_param(kcxt,%d);\n",
				dtype->type_name, index, dtype->type_name, index);
		}
		else
			elog(ERROR, "Bug? unexpected node: %s", nodeToString(node));
	lnext:
		index++;
	}
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
	appendStringInfo(buf, "  } %s;\n", name);
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
	Assert(!con.str.data);
	PG_TRY();
	{
		if (IsA(expr, List))
		{
			List	   *exprsList = (List *)expr;
			ListCell   *lc;

			foreach (lc, exprsList)
			{
				Node   *node = (Node *)lfirst(lc);

				codegen_expression_walker(&con, node, &dummy);
			}
		}
		else
		{
			codegen_expression_walker(&con, (Node *)expr, &dummy);
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
		if (con.varlena_bufsz > KERN_CONTEXT_VARLENA_BUFSZ_LIMIT)
		{
			elog(DEBUG2, "Expression consumes too much buffer (%u): %s",
				 con.varlena_bufsz, nodeToString(expr));
			return false;
		}
		Assert(con.devcost >= 0);
		if (p_devcost)
			*p_devcost = con.devcost;
		if (p_extra_sz)
			*p_extra_sz = con.varlena_bufsz;
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
	__appendStringInfo(&context->str,
					   "pgfn_%s(kcxt",
					   dfunc_name);
	foreach (lc, func_args)
	{
		Node   *expr = lfirst(lc);
		int		dummy;

		__appendStringInfo(&context->str, ", ");
		codegen_expression_walker(context, expr, &dummy);
	}
	__appendStringInfoChar(&context->str, ')');
	if (dtype->type_length > 0)
		width = dtype->type_length;
	else if (dtype->type_length == -1)
		width = -1;		/* we don't know max length of a jsonb field */
	else
		elog(ERROR, "unexpected type length: %d", dtype->type_length);

	return width;
}

#undef __appendStringInfo
#undef __appendStringInfoChar

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

void
pgstrom_init_codegen_context(codegen_context *context,
							 PlannerInfo *root,
							 RelOptInfo *baserel)
{
	memset(context, 0, sizeof(codegen_context));
	initStringInfo(&context->decl_temp);
	context->root = root;
	context->baserel = baserel;
	context->var_label = "KVAR";
	context->kds_label = "kds";
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

	devinfo_memcxt = AllocSetContextCreate(CacheMemoryContext,
										   "device type/func info cache",
										   ALLOCSET_DEFAULT_SIZES);
	CacheRegisterSyscacheCallback(PROCOID, devfunc_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(TYPEOID, devtype_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(CASTSOURCETARGET,
								  devcast_cache_invalidator, 0);
}
