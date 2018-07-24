/*
 * codegen.c
 *
 * Routines for CUDA code generator
 * ----
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
#include "pg_strom.h"
#include "cuda_numeric.h"

static MemoryContext	devinfo_memcxt;
static bool		devtype_info_is_built;
static List	   *devtype_info_slot[128];
static List	   *devfunc_info_slot[1024];
bool			pgstrom_enable_numeric_type;	/* GUC */

static pg_crc32 generic_devtype_hashfunc(devtype_info *dtype,
										 pg_crc32 hash,
										 Datum datum, bool isnull);
static pg_crc32 pg_numeric_devtype_hashfunc(devtype_info *dtype,
											pg_crc32 hash,
											Datum datum, bool isnull);
static pg_crc32 pg_bpchar_devtype_hashfunc(devtype_info *dtype,
										   pg_crc32 hash,
										   Datum datum, bool isnull);
static pg_crc32 pg_inet_devtype_hashfunc(devtype_info *dtype,
										 pg_crc32 hash,
										 Datum datum, bool isnull);
static pg_crc32 pg_range_devtype_hashfunc(devtype_info *dtype,
										  pg_crc32 hash,
										  Datum datum, bool isnull);

/*
 * Catalog of data types supported by device code
 *
 * naming convension of types:
 *   pg_<type_name>_t
 */
#define DEVTYPE_DECL(type_name,type_oid_label,type_base,		\
					 min_const,max_const,zero_const,			\
					 type_flags,extra_sz,hash_func)				\
	{ "pg_catalog", type_name, type_oid_label, type_base,		\
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
	const char	   *type_base;
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
	DEVTYPE_DECL("bool",   "BOOLOID",   "cl_bool",
				 NULL, NULL, "false",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("int2",   "INT2OID",   "cl_short",
				 "SET_2_BYTES(SHRT_MAX)",
				 "SET_2_BYTES(SHRT_MIN)",
				 "SET_2_BYTES(0)",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("int4",   "INT4OID",   "cl_int",
				 "SET_4_BYTES(INT_MAX)",
				 "SET_4_BYTES(INT_MIN)",
				 "SET_4_BYTES(0)",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("int8",   "INT8OID",   "cl_long",
				 "SET_8_BYTES(LONG_MAX)",
				 "SET_8_BYTES(LONG_MIN)",
				 "SET_8_BYTES(0)",
				 0, 0, generic_devtype_hashfunc),
	/* XXX - float2 is not a built-in data type */
	DEVTYPE_DECL("float2", "FLOAT2OID", "cl_half",
				 "SET_2_BYTES(__half_as_short(HALF_MAX))",
				 "SET_2_BYTES(__half_as_short(-HALF_MAX))",
				 "SET_2_BYTES(__half_as_short(0.0))",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("float4", "FLOAT4OID", "cl_float",
				 "SET_4_BYTES(__float_as_int(FLT_MAX))",
				 "SET_4_BYTES(__float_as_int(-FLT_MAX))",
				 "SET_4_BYTES(__float_as_int(0.0))",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("float8", "FLOAT8OID", "cl_double",
				 "SET_8_BYTES(__double_as_longlong(DBL_MAX))",
				 "SET_8_BYTES(__double_as_longlong(-DBL_MAX))",
				 "SET_8_BYTES(__double_as_longlong(0.0))",
				 0, 0, generic_devtype_hashfunc),
	/*
	 * Misc data types
	 */
	DEVTYPE_DECL("money",  "CASHOID",   "cl_long",
				 "SET_8_BYTES(LONG_MAX)",
				 "SET_8_BYTES(LONG_MIN)",
				 "SET_8_BYTES(0)",
				 DEVKERNEL_NEEDS_MISC, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("uuid",   "UUIDOID",   "pg_uuid_t",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISC, UUID_LEN,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("macaddr", "MACADDROID", "macaddr",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISC, sizeof(macaddr),
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("inet",   "INETOID",   "inet_struct",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISC, sizeof(inet),
				 pg_inet_devtype_hashfunc),
	DEVTYPE_DECL("cidr",   "CIDROID",   "inet_struct",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISC, sizeof(inet),
				 pg_inet_devtype_hashfunc),
	/*
	 * Date and time datatypes
	 */
	DEVTYPE_DECL("date", "DATEOID", "DateADT",
				 "SET_4_BYTES(INT_MAX)",
				 "SET_4_BYTES(INT_MIN)",
				 "SET_4_BYTES(0)",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("time", "TIMEOID", "TimeADT",
				 "SET_8_BYTES(LONG_MAX)",
				 "SET_8_BYTES(LONG_MIN)",
				 "SET_8_BYTES(0)",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timetz", "TIMETZOID", "TimeTzADT",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB, sizeof(TimeTzADT),
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timestamp", "TIMESTAMPOID","Timestamp",
				 "SET_8_BYTES(LONG_MAX)",
				 "SET_8_BYTES(LONG_MIN)",
				 "SET_8_BYTES(0)",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timestamptz", "TIMESTAMPTZOID", "TimestampTz",
				 "SET_8_BYTES(LONG_MAX)",
				 "SET_8_BYTES(LONG_MIN)",
				 "SET_8_BYTES(0)",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("interval", "INTERVALOID", "Interval",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB, sizeof(Interval),
				 generic_devtype_hashfunc),
	/*
	 * variable length datatypes
	 */
	DEVTYPE_DECL("bpchar",  "BPCHAROID",  "varlena *",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TEXTLIB, 0,
				 pg_bpchar_devtype_hashfunc),
	DEVTYPE_DECL("varchar", "VARCHAROID", "varlena *",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TEXTLIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("numeric", "NUMERICOID", "cl_ulong",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_NUMERIC,
				 VARHDRSZ + sizeof(union NumericChoice),
				 pg_numeric_devtype_hashfunc),
	DEVTYPE_DECL("bytea",   "BYTEAOID",   "varlena *",
				 NULL, NULL, NULL,
				 0, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("text",    "TEXTOID",    "varlena *",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TEXTLIB, 0,
				 generic_devtype_hashfunc),
	/*
	 * range types
	 */
	DEVTYPE_DECL("int4range",  "INT4RANGEOID",  "__int4range",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(cl_int) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("int8range",  "INT8RANGEOID",  "__int8range",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(cl_long) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("tsrange",    "TSRANGEOID",    "__tsrange",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(Timestamp) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("tstzrange",  "TSTZRANGEOID",  "__tstzrange",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(TimestampTz) + 1,
				 pg_range_devtype_hashfunc),
	DEVTYPE_DECL("daterange",  "DATERANGEOID",  "__daterange",
				 NULL, NULL, NULL,
                 DEVKERNEL_NEEDS_TIMELIB | DEVKERNEL_NEEDS_RANGETYPE,
				 sizeof(RangeType) + 2 * sizeof(DateADT) + 1,
				 pg_range_devtype_hashfunc),
};

static devtype_info *
build_devtype_info_entry(Oid type_oid,
						 int32 type_flags,
						 const char *type_basename,
						 const char *max_const,
						 const char *min_const,
						 const char *zero_const,
						 cl_uint extra_sz,
						 devtype_hashfunc_type hash_func,
						 devtype_info *element)
{
	HeapTuple		tuple;
	Form_pg_type	type_form;
	TypeCacheEntry *tcache;
	devtype_info   *entry;
	cl_int			hindex;
	MemoryContext	oldcxt;

	tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tuple))
		return NULL;
	type_form = (Form_pg_type) GETSTRUCT(tuple);

	/* Don't register if array type is not true array type */
	if (element && (type_form->typelem != element->type_oid ||
					type_form->typlen >= 0))
	{
		ReleaseSysCache(tuple);
		return NULL;
	}

	tcache = lookup_type_cache(type_oid,
							   TYPECACHE_EQ_OPR |
							   TYPECACHE_CMP_PROC);
	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	entry = palloc0(sizeof(devtype_info));
	entry->type_oid = type_oid;
	entry->type_flags = type_flags;
	entry->type_length = type_form->typlen;
	entry->type_align = typealign_get_width(type_form->typalign);
	entry->type_byval = type_form->typbyval;
	if (!element)
		entry->type_name = pstrdup(NameStr(type_form->typname));
	else
		entry->type_name = pstrdup("array");
	entry->type_base = pstrdup(type_basename);
	entry->max_const = max_const;	/* may be NULL */
	entry->min_const = min_const;	/* may be NULL */
	entry->zero_const = zero_const;	/* may be NULL */
	entry->extra_sz = extra_sz;
	entry->hash_func = hash_func;
	/* type equality function */
	entry->type_eqfunc = get_opcode(tcache->eq_opr);
	entry->type_cmpfunc = tcache->cmp_proc;
	MemoryContextSwitchTo(oldcxt);

	if (!element)
		entry->type_array = build_devtype_info_entry(type_form->typarray,
													 type_flags |
													 DEVKERNEL_NEEDS_MATRIX,
													 "varlena *",
													 NULL,
													 NULL,
													 NULL,
													 0,
													 generic_devtype_hashfunc,
													 entry);
	else
		entry->type_element = element;

	ReleaseSysCache(tuple);

	/* add to the hash slot */
	hindex = (hash_uint32((uint32) entry->type_oid)
			  % lengthof(devtype_info_slot));
	devtype_info_slot[hindex] = lappend_cxt(devinfo_memcxt,
											devtype_info_slot[hindex],
											entry);
	return entry;
}

static void
build_devtype_info(void)
{
	int		i;

	Assert(!devtype_info_is_built);

	for (i=0; i < lengthof(devtype_catalog); i++)
	{
		const char *nsp_name = devtype_catalog[i].type_schema;
		const char *typ_name = devtype_catalog[i].type_name;
		Oid			nsp_oid;
		Oid			typ_oid;

		nsp_oid = GetSysCacheOid1(NAMESPACENAME, CStringGetDatum(nsp_name));
		if (!OidIsValid(nsp_oid))
			continue;

		typ_oid = GetSysCacheOid2(TYPENAMENSP,
								  CStringGetDatum(typ_name),
								  ObjectIdGetDatum(nsp_oid));
		if (!OidIsValid(typ_oid))
			continue;

		(void) build_devtype_info_entry(typ_oid,
										devtype_catalog[i].type_flags,
										devtype_catalog[i].type_base,
										devtype_catalog[i].max_const,
										devtype_catalog[i].min_const,
										devtype_catalog[i].zero_const,
										devtype_catalog[i].extra_sz,
										devtype_catalog[i].hash_func,
										NULL);
	}
	devtype_info_is_built = true;
}

devtype_info *
pgstrom_devtype_lookup(Oid type_oid)
{
	ListCell	   *cell;
	int				hindex;

	if (!devtype_info_is_built)
		build_devtype_info();

	/*
	 * Numeric data type with large digits tend to cause CPU fallback.
	 * It may cause performance slowdown or random fault. So, we give
	 * an option to disable only numeric values.
	 */
	if (type_oid == NUMERICOID && !pgstrom_enable_numeric_type)
		return NULL;

	hindex = hash_uint32((uint32) type_oid) % lengthof(devtype_info_slot);

	foreach (cell, devtype_info_slot[hindex])
	{
		devtype_info   *entry = lfirst(cell);

		if (entry->type_oid == type_oid)
			return entry;
	}
	return NULL;
}

static void
pgstrom_devtype_track(codegen_context *context, devtype_info *dtype)
{
	ListCell   *lc;

	context->extra_flags |= dtype->type_flags;
	foreach (lc, context->type_defs)
	{
		Oid		type_oid = intVal(lfirst(lc));

		if (type_oid == dtype->type_oid)
			return;
	}
	context->type_defs = lappend(context->type_defs,
								 makeInteger(dtype->type_oid));
}

devtype_info *
pgstrom_devtype_lookup_and_track(Oid type_oid, codegen_context *context)
{
	devtype_info   *dtype = pgstrom_devtype_lookup(type_oid);

	if (!dtype)
		return NULL;
	pgstrom_devtype_track(context, dtype);

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

		nsp_oid = GetSysCacheOid1(NAMESPACENAME,
								  CStringGetDatum(nsp_name));
		if (!OidIsValid(nsp_oid))
			continue;

		typ_oid = GetSysCacheOid2(TYPENAMENSP,
								  CStringGetDatum(typ_name),
								  ObjectIdGetDatum(nsp_oid));
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
static pg_crc32
generic_devtype_hashfunc(devtype_info *dtype,
						 pg_crc32 hash,
						 Datum datum, bool isnull)
{
	if (isnull)
		return hash;
	else if (dtype->type_byval)
		COMP_LEGACY_CRC32(hash, &datum, dtype->type_length);
	else if (dtype->type_length > 0)
		COMP_LEGACY_CRC32(hash, DatumGetPointer(datum), dtype->type_length);
	else
		COMP_LEGACY_CRC32(hash, VARDATA_ANY(datum), VARSIZE_ANY_EXHDR(datum));
	return hash;
}

static pg_crc32
pg_numeric_devtype_hashfunc(devtype_info *dtype,
							pg_crc32 hash,
							Datum datum, bool isnull)
{
	Assert(dtype->type_oid == NUMERICOID);
	if (!isnull)
	{
		kern_context	dummy;
		pg_numeric_t	temp;

		memset(&dummy, 0, sizeof(kern_context));
		/*
		 * FIXME: If NUMERIC value is out of range, we may not be able to
		 * execute GpuJoin in the kernel space for all the outer chunks.
		 * Is it still valuable to run on GPU kernel?
		 */
		temp = pg_numeric_from_varlena(&dummy, (struct varlena *)
									   DatumGetPointer(datum));
		if (dummy.e.errcode != StromError_Success)
			elog(ERROR, "failed on hash calculation of device numeric: %s",
				 DatumGetCString(DirectFunctionCall1(numeric_out, datum)));
		COMP_LEGACY_CRC32(hash, &temp.value, sizeof(temp.value));
	}
	return hash;
}

static pg_crc32
pg_bpchar_devtype_hashfunc(devtype_info *dtype,
						   pg_crc32 hash,
						   Datum datum, bool isnull)
{
	/*
	 * whitespace is the tail end of CHAR(n) data shall be ignored
	 * when we calculate hash-value, to match same text exactly.
	 */
	Assert(dtype->type_oid == BPCHAROID);
	if (!isnull)
	{
		char   *s = VARDATA_ANY(datum);
		int		i, len = VARSIZE_ANY_EXHDR(datum);

		for (i = len - 1; i >= 0 && s[i] == ' '; i--)
			;
		COMP_LEGACY_CRC32(hash, VARDATA_ANY(datum), i+1);
	}
	return hash;
}

static pg_crc32
pg_inet_devtype_hashfunc(devtype_info *dtype,
						 pg_crc32 hash,
						 Datum datum, bool isnull)
{
	Assert(dtype->type_oid == INETOID ||
		   dtype->type_oid == CIDROID);
	if (!isnull)
	{
		inet_struct *is = (inet_struct *) VARDATA_ANY(datum);

		if (is->family == PGSQL_AF_INET)
			COMP_LEGACY_CRC32(hash, is, offsetof(inet_struct, ipaddr[4]));
		else if (is->family == PGSQL_AF_INET6)
			COMP_LEGACY_CRC32(hash, is, offsetof(inet_struct, ipaddr[16]));
		else
			elog(ERROR, "unexpected address family: %d", is->family);
	}
	return hash;
}

static pg_crc32
pg_range_devtype_hashfunc(devtype_info *dtype,
						  pg_crc32 hash,
						  Datum datum, bool isnull)
{
	if (!isnull)
	{
		RangeType  *r = DatumGetRangeTypeP(datum);
		char	   *pos = (char *)(r + 1);
		char		flags = *((char *)r + VARSIZE(r) - 1);

		if (RANGE_HAS_LBOUND(flags))
		{
			COMP_LEGACY_CRC32(hash, pos, dtype->type_length);
			pos += TYPEALIGN(dtype->type_align,
							 dtype->type_length);
		}
		if (RANGE_HAS_UBOUND(flags))
		{
			COMP_LEGACY_CRC32(hash, pos, dtype->type_length);
		}
		COMP_LEGACY_CRC32(hash, &flags, sizeof(char));
	}
	return hash;
}

/*
 * Catalog of functions supported by device code
 *
 * naming convension of functions:
 *   pgfn_<func_name>(...)
 *
 * As PostgreSQL allows function overloading, OpenCL also allows it; we can
 * define multiple functions with same name but different argument types,
 * so we can assume PostgreSQL's function name can be a unique identifier
 * in the OpenCL world.
 * This convension is same if we use built-in PG-Strom functions on OpenCL.
 * All the built-in function shall be defined according to the above naming
 * convension.
 * One thing we need to pay attention is namespace of SQL functions.
 * Right now, we support only built-in functions installed in pg_catalog
 * namespace, so we don't put special qualification here.
 *
 * func_template is a set of characters based on the rules below:
 *
 * [<attributes>/](c|r|l|b|f|F):<extra>
 *
 * attributes:
 * 'c' : this function is locale aware, thus, available only if simple
 *       collation configuration (none, and C-locale).
 * 'p' : this function needs cuda_primitive.h
 * 'm' : this function needs cuda_mathlib.h
 * 'n' : this function needs cuda_numeric.h
 * 's' : this function needs cuda_textlib.h
 * 't' : this function needs cuda_timelib.h
 * 'y' : this function needs cuda_misc.h
 * 'r' : this function needs cuda_rangetype.h
 * 'E' : this function needs cuda_time_extract.h
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
	const char *func_template;	/* a template string if simple function */
} devfunc_catalog_t;

static devfunc_catalog_t devfunc_common_catalog[] = {
	/* Type cast functions */
	{ "bool", 1, {INT4OID},   "m/f:int4_bool" },

	{ "int2", 1, {INT4OID},   "p/f:to_int2" },
	{ "int2", 1, {INT8OID},   "p/f:to_int2" },
	{ "int2", 1, {FLOAT4OID}, "p/f:to_int2" },
	{ "int2", 1, {FLOAT8OID}, "p/f:to_int2" },

	{ "int4", 1, {BOOLOID},   "p/f:to_int4" },
	{ "int4", 1, {INT2OID},   "p/f:to_int4" },
	{ "int4", 1, {INT8OID},   "p/f:to_int4" },
	{ "int4", 1, {FLOAT4OID}, "p/f:to_int4" },
	{ "int4", 1, {FLOAT8OID}, "p/f:to_int4" },

	{ "int8", 1, {INT2OID},   "p/f:to_int8" },
	{ "int8", 1, {INT4OID},   "p/f:to_int8" },
	{ "int8", 1, {FLOAT4OID}, "p/f:to_int8" },
	{ "int8", 1, {FLOAT8OID}, "p/f:to_int8" },

	{ "float4", 1, {INT2OID},   "p/f:to_float4" },
	{ "float4", 1, {INT4OID},   "p/f:to_float4" },
	{ "float4", 1, {INT8OID},   "p/f:to_float4" },
	{ "float4", 1, {FLOAT8OID}, "p/f:to_float4" },

	{ "float8", 1, {INT2OID},   "p/f:to_float8" },
	{ "float8", 1, {INT4OID},   "p/f:to_float8" },
	{ "float8", 1, {INT8OID},   "p/f:to_float8" },
	{ "float8", 1, {FLOAT4OID}, "p/f:to_float8" },

	/* '+' : add operators */
	{ "int2pl",  2, {INT2OID, INT2OID}, "m/f:int2pl" },
	{ "int24pl", 2, {INT2OID, INT4OID}, "m/f:int24pl" },
	{ "int28pl", 2, {INT2OID, INT8OID}, "m/f:int28pl" },
	{ "int42pl", 2, {INT4OID, INT2OID}, "m/f:int42pl" },
	{ "int4pl",  2, {INT4OID, INT4OID}, "m/f:int4pl" },
	{ "int48pl", 2, {INT4OID, INT8OID}, "m/f:int48pl" },
	{ "int82pl", 2, {INT8OID, INT2OID}, "m/f:int82pl" },
	{ "int84pl", 2, {INT8OID, INT4OID}, "m/f:int84pl" },
	{ "int8pl",  2, {INT8OID, INT8OID}, "m/f:int8pl" },
	{ "float4pl",  2, {FLOAT4OID, FLOAT4OID}, "m/f:float4pl" },
	{ "float48pl", 2, {FLOAT4OID, FLOAT8OID}, "m/f:float48pl" },
	{ "float84pl", 2, {FLOAT8OID, FLOAT4OID}, "m/f:float84pl" },
	{ "float8pl",  2, {FLOAT8OID, FLOAT8OID}, "m/f:float8pl" },

	/* '-' : subtract operators */
	{ "int2mi",  2, {INT2OID, INT2OID}, "m/f:int2mi" },
	{ "int24mi", 2, {INT2OID, INT4OID}, "m/f:int24mi" },
	{ "int28mi", 2, {INT2OID, INT8OID}, "m/f:int28mi" },
	{ "int42mi", 2, {INT4OID, INT2OID}, "m/f:int42mi" },
	{ "int4mi",  2, {INT4OID, INT4OID}, "m/f:int4mi" },
	{ "int48mi", 2, {INT4OID, INT8OID}, "m/f:int48mi" },
	{ "int82mi", 2, {INT8OID, INT2OID}, "m/f:int82mi" },
	{ "int84mi", 2, {INT8OID, INT4OID}, "m/f:int84mi" },
	{ "int8mi",  2, {INT8OID, INT8OID}, "m/f:int8mi" },
	{ "float4mi",  2, {FLOAT4OID, FLOAT4OID}, "m/f:float4mi" },
	{ "float48mi", 2, {FLOAT4OID, FLOAT8OID}, "m/f:float48mi" },
	{ "float84mi", 2, {FLOAT8OID, FLOAT4OID}, "m/f:float84mi" },
	{ "float8mi",  2, {FLOAT8OID, FLOAT8OID}, "m/f:float8mi" },

	/* '*' : mutiply operators */
	{ "int2mul",  2, {INT2OID, INT2OID}, "m/f:int2mul" },
	{ "int24mul", 2, {INT2OID, INT4OID}, "m/f:int24mul" },
	{ "int28mul", 2, {INT2OID, INT8OID}, "m/f:int28mul" },
	{ "int42mul", 2, {INT4OID, INT2OID}, "m/f:int42mul" },
	{ "int4mul",  2, {INT4OID, INT4OID}, "m/f:int4mul" },
	{ "int48mul", 2, {INT4OID, INT8OID}, "m/f:int48mul" },
	{ "int82mul", 2, {INT8OID, INT2OID}, "m/f:int82mul" },
	{ "int84mul", 2, {INT8OID, INT4OID}, "m/f:int84mul" },
	{ "int8mul",  2, {INT8OID, INT8OID}, "m/f:int8mul" },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, "m/f:float4mul" },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, "m/f:float48mul" },
	{ "float84mul", 2, {FLOAT8OID, FLOAT4OID}, "m/f:float84mul" },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, "m/f:float8mul" },

	/* '/' : divide operators */
	{ "int2div",  2, {INT2OID, INT2OID}, "m/f:int2div" },
	{ "int24div", 2, {INT2OID, INT4OID}, "m/f:int24div" },
	{ "int28div", 2, {INT2OID, INT8OID}, "m/f:int28div" },
	{ "int42div", 2, {INT4OID, INT2OID}, "m/f:int42div" },
	{ "int4div",  2, {INT4OID, INT4OID}, "m/f:int4div" },
	{ "int48div", 2, {INT4OID, INT8OID}, "m/f:int48div" },
	{ "int82div", 2, {INT8OID, INT2OID}, "m/f:int82div" },
	{ "int84div", 2, {INT8OID, INT4OID}, "m/f:int84div" },
	{ "int8div",  2, {INT8OID, INT8OID}, "m/f:int8div" },
	{ "float4div",  2, {FLOAT4OID, FLOAT4OID}, "m/f:float4div" },
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, "m/f:float48div" },
	{ "float84div", 2, {FLOAT8OID, FLOAT4OID}, "m/f:float84div" },
	{ "float8div",  2, {FLOAT8OID, FLOAT8OID}, "m/f:float8div" },

	/* '%' : reminder operators */
	{ "int2mod", 2, {INT2OID, INT2OID}, "m/f:int2mod" },
	{ "int4mod", 2, {INT4OID, INT4OID}, "m/f:int4mod" },
	{ "int8mod", 2, {INT8OID, INT8OID}, "m/f:int8mod" },

	/* '+' : unary plus operators */
	{ "int2up", 1, {INT2OID},      "p/f:int2up" },
	{ "int4up", 1, {INT4OID},      "p/f:int4up" },
	{ "int8up", 1, {INT8OID},      "p/f:int8up" },
	{ "float4up", 1, {FLOAT4OID},  "p/f:float4up" },
	{ "float8up", 1, {FLOAT8OID},  "p/f:float8up" },

	/* '-' : unary minus operators */
	{ "int2um", 1, {INT2OID},      "p/f:int2um" },
	{ "int4um", 1, {INT4OID},      "p/f:int4um" },
	{ "int8um", 1, {INT8OID},      "p/f:int8um" },
	{ "float4um", 1, {FLOAT4OID},  "p/f:float4um" },
	{ "float8um", 1, {FLOAT8OID},  "p/f:float8um" },

	/* '@' : absolute value operators */
	{ "int2abs", 1, {INT2OID},     "p/f:int2abs" },
	{ "int4abs", 1, {INT4OID},     "p/f:int4abs" },
	{ "int8abs", 1, {INT8OID},     "p/f:int8abs" },
	{ "float4abs", 1, {FLOAT4OID}, "p/f:float4abs" },
	{ "float8abs", 1, {FLOAT8OID}, "p/f:float8abs" },

	/* '=' : equal operators */
	{ "booleq",  2, {BOOLOID, BOOLOID}, "p/f:booleq" },
	{ "int2eq",  2, {INT2OID, INT2OID}, "p/f:int2eq" },
	{ "int24eq", 2, {INT2OID, INT4OID}, "p/f:int24eq" },
	{ "int28eq", 2, {INT2OID, INT8OID}, "p/f:int28eq" },
	{ "int42eq", 2, {INT4OID, INT2OID}, "p/f:int42eq" },
	{ "int4eq",  2, {INT4OID, INT4OID}, "p/f:int4eq" },
	{ "int48eq", 2, {INT4OID, INT8OID}, "p/f:int48eq" },
	{ "int82eq", 2, {INT8OID, INT2OID}, "p/f:int82eq" },
	{ "int84eq", 2, {INT8OID, INT4OID}, "p/f:int84eq" },
	{ "int8eq",  2, {INT8OID, INT8OID}, "p/f:int8eq" },
	{ "float4eq",  2, {FLOAT4OID, FLOAT4OID}, "p/f:float4eq" },
	{ "float48eq", 2, {FLOAT4OID, FLOAT8OID}, "p/f:float48eq" },
	{ "float84eq", 2, {FLOAT8OID, FLOAT4OID}, "p/f:float84eq" },
	{ "float8eq",  2, {FLOAT8OID, FLOAT8OID}, "p/f:float8eq" },

	/* '<>' : not equal operators */
	{ "int2ne",  2, {INT2OID, INT2OID}, "p/f:int2ne" },
	{ "int24ne", 2, {INT2OID, INT4OID}, "p/f:int24ne" },
	{ "int28ne", 2, {INT2OID, INT8OID}, "p/f:int28ne" },
	{ "int42ne", 2, {INT4OID, INT2OID}, "p/f:int42ne" },
	{ "int4ne",  2, {INT4OID, INT4OID}, "p/f:int4ne" },
	{ "int48ne", 2, {INT4OID, INT8OID}, "p/f:int48ne" },
	{ "int82ne", 2, {INT8OID, INT2OID}, "p/f:int82ne" },
	{ "int84ne", 2, {INT8OID, INT4OID}, "p/f:int84ne" },
	{ "int8ne",  2, {INT8OID, INT8OID}, "p/f:int8ne" },
	{ "float4ne",  2, {FLOAT4OID, FLOAT4OID}, "p/f:float4ne" },
	{ "float48ne", 2, {FLOAT4OID, FLOAT8OID}, "p/f:float48ne" },
	{ "float84ne", 2, {FLOAT8OID, FLOAT4OID}, "p/f:float84ne" },
	{ "float8ne",  2, {FLOAT8OID, FLOAT8OID}, "p/f:float8ne" },

	/* '>' : greater than operators */
	{ "int2gt",  2, {INT2OID, INT2OID}, "p/f:int2gt" },
	{ "int24gt", 2, {INT2OID, INT4OID}, "p/f:int24gt" },
	{ "int28gt", 2, {INT2OID, INT8OID}, "p/f:int28gt" },
	{ "int42gt", 2, {INT4OID, INT2OID}, "p/f:int42gt" },
	{ "int4gt",  2, {INT4OID, INT4OID}, "p/f:int4gt" },
	{ "int48gt", 2, {INT4OID, INT8OID}, "p/f:int48gt" },
	{ "int82gt", 2, {INT8OID, INT2OID}, "p/f:int82gt" },
	{ "int84gt", 2, {INT8OID, INT4OID}, "p/f:int84gt" },
	{ "int8gt",  2, {INT8OID, INT8OID}, "p/f:int8gt" },
	{ "float4gt",  2, {FLOAT4OID, FLOAT4OID}, "p/f:float4gt" },
	{ "float48gt", 2, {FLOAT4OID, FLOAT8OID}, "p/f:float48gt" },
	{ "float84gt", 2, {FLOAT8OID, FLOAT4OID}, "p/f:float84gt" },
	{ "float8gt",  2, {FLOAT8OID, FLOAT8OID}, "p/f:float8gt" },

	/* '<' : less than operators */
	{ "int2lt",  2, {INT2OID, INT2OID}, "p/f:int2lt" },
	{ "int24lt", 2, {INT2OID, INT4OID}, "p/f:int24lt" },
	{ "int28lt", 2, {INT2OID, INT8OID}, "p/f:int28lt" },
	{ "int42lt", 2, {INT4OID, INT2OID}, "p/f:int42lt" },
	{ "int4lt",  2, {INT4OID, INT4OID}, "p/f:int4lt" },
	{ "int48lt", 2, {INT4OID, INT8OID}, "p/f:int48lt" },
	{ "int82lt", 2, {INT8OID, INT2OID}, "p/f:int82lt" },
	{ "int84lt", 2, {INT8OID, INT4OID}, "p/f:int84lt" },
	{ "int8lt",  2, {INT8OID, INT8OID}, "p/f:int8lt" },
	{ "float4lt",  2, {FLOAT4OID, FLOAT4OID}, "p/f:float4lt" },
	{ "float48lt", 2, {FLOAT4OID, FLOAT8OID}, "p/f:float48lt" },
	{ "float84lt", 2, {FLOAT8OID, FLOAT4OID}, "p/f:float84lt" },
	{ "float8lt",  2, {FLOAT8OID, FLOAT8OID}, "p/f:float8lt" },

	/* '>=' : relational greater-than or equal-to */
	{ "int2ge",  2, {INT2OID, INT2OID}, "p/f:int2ge" },
	{ "int24ge", 2, {INT2OID, INT4OID}, "p/f:int24ge" },
	{ "int28ge", 2, {INT2OID, INT8OID}, "p/f:int28ge" },
	{ "int42ge", 2, {INT4OID, INT2OID}, "p/f:int42ge" },
	{ "int4ge",  2, {INT4OID, INT4OID}, "p/f:int4ge" },
	{ "int48ge", 2, {INT4OID, INT8OID}, "p/f:int48ge" },
	{ "int82ge", 2, {INT8OID, INT2OID}, "p/f:int82ge" },
	{ "int84ge", 2, {INT8OID, INT4OID}, "p/f:int84ge" },
	{ "int8ge",  2, {INT8OID, INT8OID}, "p/f:int8ge" },
	{ "float4ge",  2, {FLOAT4OID, FLOAT4OID}, "p/f:float4ge" },
	{ "float48ge", 2, {FLOAT4OID, FLOAT8OID}, "p/f:float48ge" },
	{ "float84ge", 2, {FLOAT8OID, FLOAT4OID}, "p/f:float84ge" },
	{ "float8ge",  2, {FLOAT8OID, FLOAT8OID}, "p/f:float8ge" },

	/* '<=' : relational greater-than or equal-to */
	{ "int2le",  2, {INT2OID, INT2OID}, "p/f:int2le" },
	{ "int24le", 2, {INT2OID, INT4OID}, "p/f:int24le" },
	{ "int28le", 2, {INT2OID, INT8OID}, "p/f:int28le" },
	{ "int42le", 2, {INT4OID, INT2OID}, "p/f:int42le" },
	{ "int4le",  2, {INT4OID, INT4OID}, "p/f:int4le" },
	{ "int48le", 2, {INT4OID, INT8OID}, "p/f:int48le" },
	{ "int82le", 2, {INT8OID, INT2OID}, "p/f:int82le" },
	{ "int84le", 2, {INT8OID, INT4OID}, "p/f:int84le" },
	{ "int8le",  2, {INT8OID, INT8OID}, "p/f:int8le" },
	{ "float4le",  2, {FLOAT4OID, FLOAT4OID}, "p/f:float4le" },
	{ "float48le", 2, {FLOAT4OID, FLOAT8OID}, "p/f:float48le" },
	{ "float84le", 2, {FLOAT8OID, FLOAT4OID}, "p/f:float84le" },
	{ "float8le",  2, {FLOAT8OID, FLOAT8OID}, "p/f:float8le" },

	/* '&' : bitwise and */
	{ "int2and", 2, {INT2OID, INT2OID}, "p/f:int2and" },
	{ "int4and", 2, {INT4OID, INT4OID}, "p/f:int4and" },
	{ "int8and", 2, {INT8OID, INT8OID}, "p/f:int8and" },

	/* '|'  : bitwise or */
	{ "int2or", 2, {INT2OID, INT2OID}, "p/f:int2or" },
	{ "int4or", 2, {INT4OID, INT4OID}, "p/f:int4or" },
	{ "int8or", 2, {INT8OID, INT8OID}, "p/f:int8or" },

	/* '#'  : bitwise xor */
	{ "int2xor", 2, {INT2OID, INT2OID}, "p/f:int2xor" },
	{ "int4xor", 2, {INT4OID, INT4OID}, "p/f:int4xor" },
	{ "int8xor", 2, {INT8OID, INT8OID}, "p/f:int8xor" },

	/* '~'  : bitwise not operators */
	{ "int2not", 1, {INT2OID}, "p/f:int2not" },
	{ "int4not", 1, {INT4OID}, "p/f:int4not" },
	{ "int8not", 1, {INT8OID}, "p/f:int8not" },

	/* '>>' : right shift */
	{ "int2shr", 2, {INT2OID, INT4OID}, "p/f:int2shr" },
	{ "int4shr", 2, {INT4OID, INT4OID}, "p/f:int4shr" },
	{ "int8shr", 2, {INT8OID, INT4OID}, "p/f:int8shr" },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID, INT4OID}, "p/f:int2shl" },
	{ "int4shl", 2, {INT4OID, INT4OID}, "p/f:int4shl" },
	{ "int8shl", 2, {INT8OID, INT4OID}, "p/f:int8shl" },

	/* comparison functions */
	{ "btboolcmp",  2, {BOOLOID, BOOLOID}, "p/f:type_compare" },
	{ "btint2cmp",  2, {INT2OID, INT2OID}, "p/f:type_compare" },
	{ "btint24cmp", 2, {INT2OID, INT4OID}, "p/f:type_compare" },
	{ "btint28cmp", 2, {INT2OID, INT8OID}, "p/f:type_compare" },
	{ "btint42cmp", 2, {INT4OID, INT2OID}, "p/f:type_compare" },
	{ "btint4cmp",  2, {INT4OID, INT4OID}, "p/f:type_compare" },
	{ "btint48cmp", 2, {INT4OID, INT8OID}, "p/f:type_compare" },
	{ "btint82cmp", 2, {INT8OID, INT2OID}, "p/f:type_compare" },
	{ "btint84cmp", 2, {INT8OID, INT4OID}, "p/f:type_compare" },
	{ "btint8cmp",  2, {INT8OID, INT8OID}, "p/f:type_compare" },
	{ "btfloat4cmp",  2, {FLOAT4OID, FLOAT4OID}, "p/f:type_compare" },
	{ "btfloat48cmp", 2, {FLOAT4OID, FLOAT8OID}, "p/f:type_compare" },
	{ "btfloat84cmp", 2, {FLOAT8OID, FLOAT4OID}, "p/f:type_compare" },
	{ "btfloat8cmp",  2, {FLOAT8OID, FLOAT8OID}, "p/f:type_compare" },

	/* currency cast */
	{ "money",			1, {NUMERICOID},			"y/f:numeric_cash" },
	{ "money",			1, {INT4OID},				"y/f:int4_cash" },
	{ "money",			1, {INT8OID},				"y/f:int8_cash" },
	/* currency operators */
	{ "cash_pl",		2, {CASHOID, CASHOID},		"y/f:cash_pl" },
	{ "cash_mi",		2, {CASHOID, CASHOID},		"y/f:cash_mi" },
	{ "cash_div_cash",	2, {CASHOID, CASHOID},		"y/f:cash_div_cash" },
	{ "cash_mul_int2",	2, {CASHOID, INT2OID},		"y/f:cash_mul_int2" },
	{ "cash_mul_int4",	2, {CASHOID, INT4OID},		"y/f:cash_mul_int4" },
	{ "cash_mul_flt4",	2, {CASHOID, FLOAT4OID},	"y/f:cash_mul_flt4" },
	{ "cash_mul_flt8",	2, {CASHOID, FLOAT8OID},	"y/f:cash_mul_flt8" },
	{ "cash_div_int2",	2, {CASHOID, INT2OID},		"y/f:cash_div_int2" },
	{ "cash_div_int4",	2, {CASHOID, INT4OID},		"y/f:cash_div_int4" },
	{ "cash_div_flt4",	2, {CASHOID, FLOAT4OID},	"y/f:cash_div_flt4" },
	{ "cash_div_flt8",	2, {CASHOID, FLOAT8OID},	"y/f:cash_div_flt8" },
	{ "int2_mul_cash",	2, {INT2OID, CASHOID},		"y/f:int2_mul_cash" },
	{ "int4_mul_cash",	2, {INT4OID, CASHOID},		"y/f:int4_mul_cash" },
	{ "flt4_mul_cash",	2, {FLOAT4OID, CASHOID},	"y/f:flt4_mul_cash" },
	{ "flt8_mul_cash",	2, {FLOAT8OID, CASHOID},	"y/f:flt8_mul_cash" },
	/* currency comparison */
	{ "cash_cmp",		2, {CASHOID, CASHOID},		"y/f:type_compare" },
	{ "cash_eq",		2, {CASHOID, CASHOID},		"y/f:cash_eq" },
	{ "cash_ne",		2, {CASHOID, CASHOID},		"y/f:cash_ne" },
	{ "cash_lt",		2, {CASHOID, CASHOID},		"y/f:cash_lt" },
	{ "cash_le",		2, {CASHOID, CASHOID},		"y/f:cash_le" },
	{ "cash_gt",		2, {CASHOID, CASHOID},		"y/f:cash_gt" },
	{ "cash_ge",		2, {CASHOID, CASHOID},		"y/f:cash_ge" },
	/* uuid comparison */
	{ "uuid_cmp",		2, {UUIDOID, UUIDOID},		"y/f:type_compare" },
	{ "uuid_eq",		2, {UUIDOID, UUIDOID},		"y/f:uuid_eq" },
	{ "uuid_ne",		2, {UUIDOID, UUIDOID},		"y/f:uuid_ne" },
	{ "uuid_lt",		2, {UUIDOID, UUIDOID},		"y/f:uuid_lt" },
	{ "uuid_le",		2, {UUIDOID, UUIDOID},		"y/f:uuid_le" },
	{ "uuid_gt",		2, {UUIDOID, UUIDOID},		"y/f:uuid_gt" },
	{ "uuid_ge",		2, {UUIDOID, UUIDOID},		"y/f:uuid_ge" },
	/* macaddr comparison */
	{ "macaddr_cmp",    2, {MACADDROID,MACADDROID}, "y/f:type_compare" },
	{ "macaddr_eq",     2, {MACADDROID,MACADDROID}, "y/f:macaddr_eq" },
	{ "macaddr_ne",     2, {MACADDROID,MACADDROID}, "y/f:macaddr_ne" },
	{ "macaddr_lt",     2, {MACADDROID,MACADDROID}, "y/f:macaddr_lt" },
	{ "macaddr_le",     2, {MACADDROID,MACADDROID}, "y/f:macaddr_le" },
	{ "macaddr_gt",     2, {MACADDROID,MACADDROID}, "y/f:macaddr_gt" },
	{ "macaddr_ge",     2, {MACADDROID,MACADDROID}, "y/f:macaddr_ge" },
	/* inet comparison */
	{ "network_cmp",    2, {INETOID,INETOID},       "y/f:type_compare" },
	{ "network_eq",     2, {INETOID,INETOID},       "y/f:network_eq" },
	{ "network_ne",     2, {INETOID,INETOID},       "y/f:network_ne" },
	{ "network_lt",     2, {INETOID,INETOID},       "y/f:network_lt" },
	{ "network_le",     2, {INETOID,INETOID},       "y/f:network_le" },
	{ "network_gt",     2, {INETOID,INETOID},       "y/f:network_gt" },
	{ "network_ge",     2, {INETOID,INETOID},       "y/f:network_ge" },
	{ "network_larger", 2, {INETOID,INETOID},       "y/f:network_larger" },
	{ "network_smaller", 2, {INETOID,INETOID},      "y/f:network_smaller" },
	{ "network_sub",    2, {INETOID,INETOID},       "y/f:network_sub" },
	{ "network_subeq",  2, {INETOID,INETOID},       "y/f:network_subeq" },
	{ "network_sup",    2, {INETOID,INETOID},       "y/f:network_sup" },
	{ "network_supeq",  2, {INETOID,INETOID},       "y/f:network_supeq" },
	{ "network_overlap",2, {INETOID,INETOID},       "y/f:network_overlap" },

	/*
     * Mathmatical functions
     */
	{ "abs", 1, {INT2OID}, "p/f:int2abs" },
	{ "abs", 1, {INT4OID}, "p/f:int4abs" },
	{ "abs", 1, {INT8OID}, "p/f:int8abs" },
	{ "abs", 1, {FLOAT4OID}, "p/f:float4abs" },
	{ "abs", 1, {FLOAT8OID}, "p/f:float8abs" },
	{ "cbrt",  1, {FLOAT8OID}, "m/f:cbrt" },
	{ "dcbrt", 1, {FLOAT8OID}, "m/f:cbrt" },
	{ "ceil", 1, {FLOAT8OID}, "m/f:ceil" },
	{ "ceiling", 1, {FLOAT8OID}, "m/f:ceil" },
	{ "exp", 1, {FLOAT8OID}, "m/f:exp" },
	{ "dexp", 1, {FLOAT8OID}, "m/f:exp" },
	{ "floor", 1, {FLOAT8OID}, "m/f:floor" },
	{ "ln", 1, {FLOAT8OID}, "m/f:ln" },
	{ "dlog1", 1, {FLOAT8OID}, "m/f:ln" },
	{ "log", 1, {FLOAT8OID}, "m/f:log10" },
	{ "dlog10", 1, {FLOAT8OID}, "m/f:log10" },
	{ "pi", 0, {}, "m/f:dpi" },
	{ "power", 2, {FLOAT8OID, FLOAT8OID}, "m/f:dpow" },
	{ "pow", 2, {FLOAT8OID, FLOAT8OID}, "m/f:dpow" },
	{ "dpow", 2, {FLOAT8OID, FLOAT8OID}, "m/f:dpow" },
	{ "round", 1, {FLOAT8OID}, "m/f:round" },
	{ "dround", 1, {FLOAT8OID}, "m/f:round" },
	{ "sign", 1, {FLOAT8OID}, "m/f:sign" },
	{ "sqrt", 1, {FLOAT8OID}, "m/f:dsqrt" },
	{ "dsqrt", 1, {FLOAT8OID}, "m/f:dsqrt" },
	{ "trunc", 1, {FLOAT8OID}, "m/f:trunc" },
	{ "dtrunc", 1, {FLOAT8OID}, "m/f:trunc" },

	/*
     * Trigonometric function
     */
	{ "degrees", 1, {FLOAT8OID}, "m/f:degrees" },
	{ "radians", 1, {FLOAT8OID}, "m/f:radians" },
	{ "acos",    1, {FLOAT8OID}, "m/f:acos" },
	{ "asin",    1, {FLOAT8OID}, "m/f:asin" },
	{ "atan",    1, {FLOAT8OID}, "m/f:atan" },
	{ "atan2",   2, {FLOAT8OID, FLOAT8OID}, "m/f:atan2" },
	{ "cos",     1, {FLOAT8OID}, "m/f:cos" },
	{ "cot",     1, {FLOAT8OID}, "m/f:cot" },
	{ "sin",     1, {FLOAT8OID}, "m/f:sin" },
	{ "tan",     1, {FLOAT8OID}, "m/f:tan" },

	/*
	 * Numeric functions
	 * ------------------------- */
	/* Numeric type cast functions */
	{ "int2",    1, {NUMERICOID}, "n/f:numeric_int2" },
	{ "int4",    1, {NUMERICOID}, "n/f:numeric_int4" },
	{ "int8",    1, {NUMERICOID}, "n/f:numeric_int8" },
	{ "float4",  1, {NUMERICOID}, "n/f:numeric_float4" },
	{ "float8",  1, {NUMERICOID}, "n/f:numeric_float8" },
	{ "numeric", 1, {INT2OID},    "n/f:int2_numeric" },
	{ "numeric", 1, {INT4OID},    "n/f:int4_numeric" },
	{ "numeric", 1, {INT8OID},    "n/f:int8_numeric" },
	{ "numeric", 1, {FLOAT4OID},  "n/f:float4_numeric" },
	{ "numeric", 1, {FLOAT8OID},  "n/f:float8_numeric" },
	/* Numeric operators */
	{ "numeric_add", 2, {NUMERICOID, NUMERICOID}, "n/f:numeric_add" },
	{ "numeric_sub", 2, {NUMERICOID, NUMERICOID}, "n/f:numeric_sub" },
	{ "numeric_mul", 2, {NUMERICOID, NUMERICOID}, "n/f:numeric_mul" },
	{ "numeric_uplus",  1, {NUMERICOID}, "n/f:numeric_uplus" },
	{ "numeric_uminus", 1, {NUMERICOID}, "n/f:numeric_uminus" },
	{ "numeric_abs",    1, {NUMERICOID}, "n/f:numeric_abs" },
	{ "abs",            1, {NUMERICOID}, "n/f:numeric_abs" },
	/* Numeric comparison */
	{ "numeric_eq", 2, {NUMERICOID, NUMERICOID},  "n/f:numeric_eq" },
	{ "numeric_ne", 2, {NUMERICOID, NUMERICOID},  "n/f:numeric_ne" },
	{ "numeric_lt", 2, {NUMERICOID, NUMERICOID},  "n/f:numeric_lt" },
	{ "numeric_le", 2, {NUMERICOID, NUMERICOID},  "n/f:numeric_le" },
	{ "numeric_gt", 2, {NUMERICOID, NUMERICOID},  "n/f:numeric_gt" },
	{ "numeric_ge", 2, {NUMERICOID, NUMERICOID},  "n/f:numeric_ge" },
	{ "numeric_cmp", 2, {NUMERICOID, NUMERICOID}, "n/f:type_compare" },

	/*
	 * Date and time functions
	 * ------------------------------- */
	/* Type cast functions */
	{ "date", 1, {TIMESTAMPOID}, "t/f:timestamp_date" },
	{ "date", 1, {TIMESTAMPTZOID}, "t/f:timestamptz_date" },
	{ "time", 1, {TIMETZOID}, "t/f:timetz_time" },
	{ "time", 1, {TIMESTAMPOID}, "t/f:timestamp_time" },
	{ "time", 1, {TIMESTAMPTZOID}, "t/f:timestamptz_time" },
	{ "timetz", 1, {TIMEOID}, "t/f:time_timetz" },
	{ "timetz", 1, {TIMESTAMPTZOID}, "t/f:timestamptz_timetz" },
#ifdef NOT_USED
	{ "timetz", 2, {TIMETZOID, INT4OID}, "t/f:timetz_scale" },
#endif
	{ "timestamp", 1, {DATEOID}, "t/f:date_timestamp" },
	{ "timestamp", 1, {TIMESTAMPTZOID}, "t/f:timestamptz_timestamp" },
	{ "timestamptz", 1, {DATEOID}, "t/f:date_timestamptz" },
	{ "timestamptz", 1, {TIMESTAMPOID}, "t/f:timestamp_timestamptz" },
	/* timedata operators */
	{ "date_pli", 2, {DATEOID, INT4OID}, "t/f:date_pli" },
	{ "date_mii", 2, {DATEOID, INT4OID}, "t/f:date_mii" },
	{ "date_mi", 2, {DATEOID, DATEOID}, "t/f:date_mi" },
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, "t/f:datetime_pl" },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, "t/f:integer_pl_date" },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, "t/f:timedate_pl" },
	/* time - time => interval */
	{ "time_mi_time", 2, {TIMEOID, TIMEOID}, "t/f:time_mi_time" },
	/* timestamp - timestamp => interval */
	{ "timestamp_mi", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_mi" },
	/* timetz +/- interval => timetz */
	{ "timetz_pl_interval", 2, {TIMETZOID, INTERVALOID},
	  "t/f:timetz_pl_interval" },
	{ "timetz_mi_interval", 2, {TIMETZOID, INTERVALOID},
	  "t/f:timetz_mi_interval" },
	/* timestamptz +/- interval => timestamptz */
	{ "timestamptz_pl_interval", 2, {TIMESTAMPTZOID, INTERVALOID},
	  "t/f:timestamptz_pl_interval" },
	{ "timestamptz_mi_interval", 2, {TIMESTAMPTZOID, INTERVALOID},
	  "t/f:timestamptz_mi_interval" },
	/* interval operators */
	{ "interval_um", 1, {INTERVALOID}, "t/f:interval_um" },
	{ "interval_pl", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_pl" },
	{ "interval_mi", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_mi" },
	/* date + timetz => timestamptz */
	{ "datetimetz_pl", 2, {DATEOID, TIMETZOID}, "t/f:datetimetz_timestamptz" },
	{ "timestamptz", 2, {DATEOID, TIMETZOID}, "t/f:datetimetz_timestamptz" },
	/* comparison between date */
	{ "date_eq", 2, {DATEOID, DATEOID}, "t/f:date_eq" },
	{ "date_ne", 2, {DATEOID, DATEOID}, "t/f:date_ne" },
	{ "date_lt", 2, {DATEOID, DATEOID}, "t/f:date_lt"  },
	{ "date_le", 2, {DATEOID, DATEOID}, "t/f:date_le" },
	{ "date_gt", 2, {DATEOID, DATEOID}, "t/f:date_gt"  },
	{ "date_ge", 2, {DATEOID, DATEOID}, "t/f:date_ge" },
	{ "date_cmp", 2, {DATEOID, DATEOID}, "t/f:type_compare" },
	/* comparison of date and timestamp */
	{ "date_eq_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/f:date_eq_timestamp" },
	{ "date_ne_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/f:date_ne_timestamp" },
	{ "date_lt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/f:date_lt_timestamp" },
	{ "date_le_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/f:date_le_timestamp" },
	{ "date_gt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/f:date_gt_timestamp" },
	{ "date_ge_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/f:date_ge_timestamp" },
	{ "date_cmp_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/f:date_cmp_timestamp" },
	/* comparison between time */
	{ "time_eq", 2, {TIMEOID, TIMEOID}, "t/f:time_eq" },
	{ "time_ne", 2, {TIMEOID, TIMEOID}, "t/f:time_ne" },
	{ "time_lt", 2, {TIMEOID, TIMEOID}, "t/f:time_lt"  },
	{ "time_le", 2, {TIMEOID, TIMEOID}, "t/f:time_le" },
	{ "time_gt", 2, {TIMEOID, TIMEOID}, "t/f:time_gt"  },
	{ "time_ge", 2, {TIMEOID, TIMEOID}, "t/f:time_ge" },
	{ "time_cmp", 2, {TIMEOID, TIMEOID}, "t/f:type_compare" },
	/* comparison between timetz */
	{ "timetz_eq", 2, {TIMETZOID, TIMETZOID}, "t/f:timetz_eq" },
	{ "timetz_ne", 2, {TIMETZOID, TIMETZOID}, "t/f:timetz_ne" },
	{ "timetz_lt", 2, {TIMETZOID, TIMETZOID}, "t/f:timetz_lt" },
	{ "timetz_le", 2, {TIMETZOID, TIMETZOID}, "t/f:timetz_le" },
	{ "timetz_ge", 2, {TIMETZOID, TIMETZOID}, "t/f:timetz_ge" },
	{ "timetz_gt", 2, {TIMETZOID, TIMETZOID}, "t/f:timetz_gt" },
	{ "timetz_cmp", 2, {TIMETZOID, TIMETZOID}, "t/f:timetz_cmp" },
	/* comparison between timestamp */
	{ "timestamp_eq", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_eq" },
	{ "timestamp_ne", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_ne" },
	{ "timestamp_lt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_lt"  },
	{ "timestamp_le", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_le" },
	{ "timestamp_gt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_gt"  },
	{ "timestamp_ge", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_ge" },
	{ "timestamp_cmp", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/f:timestamp_cmp" },
	/* comparison of timestamp and date */
	{ "timestamp_eq_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/f:timestamp_eq_date" },
	{ "timestamp_ne_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/f:timestamp_ne_date" },
	{ "timestamp_lt_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/f:timestamp_lt_date" },
	{ "timestamp_le_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/f:timestamp_le_date" },
	{ "timestamp_gt_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/f:timestamp_gt_date" },
	{ "timestamp_ge_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/f:timestamp_ge_date" },
	{ "timestamp_cmp_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/f:timestamp_cmp_date"},
	/* comparison between timestamptz */
	{ "timestamptz_eq", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/f:timestamptz_eq" },
	{ "timestamptz_ne", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/f:timestamptz_ne" },
	{ "timestamptz_lt", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/f:timestamptz_lt" },
	{ "timestamptz_le", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/f:timestamptz_le" },
	{ "timestamptz_gt", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/f:timestamptz_gt" },
	{ "timestamptz_ge", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/f:timestamptz_ge" },
	{ "timestamptz_cmp", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, 
	  "t/f:type_compare" },

	/* comparison between date and timestamptz */
	{ "date_lt_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/f:date_lt_timestamptz" },
	{ "date_le_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/f:date_le_timestamptz" },
	{ "date_eq_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/f:date_eq_timestamptz" },
	{ "date_ge_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/f:date_ge_timestamptz" },
	{ "date_gt_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/f:date_gt_timestamptz" },
	{ "date_ne_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/f:date_ne_timestamptz" },

	/* comparison between timestamptz and date */
	{ "timestamptz_lt_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/f:timestamptz_lt_date" },
	{ "timestamptz_le_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/f:timestamptz_le_date" },
	{ "timestamptz_eq_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/f:timestamptz_eq_date" },
	{ "timestamptz_ge_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/f:timestamptz_ge_date" },
	{ "timestamptz_gt_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/f:timestamptz_gt_date" },
	{ "timestamptz_ne_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/f:timestamptz_ne_date" },

	/* comparison between timestamp and timestamptz  */
	{ "timestamp_lt_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/f:timestamp_lt_timestamptz" },
	{ "timestamp_le_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/f:timestamp_le_timestamptz" },
	{ "timestamp_eq_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/f:timestamp_eq_timestamptz" },
	{ "timestamp_ge_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/f:timestamp_ge_timestamptz" },
	{ "timestamp_gt_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/f:timestamp_gt_timestamptz" },
	{ "timestamp_ne_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/f:timestamp_ne_timestamptz" },

	/* comparison between timestamptz and timestamp  */
	{ "timestamptz_lt_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/f:timestamptz_lt_timestamp" },
	{ "timestamptz_le_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/f:timestamptz_le_timestamp" },
	{ "timestamptz_eq_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/f:timestamptz_eq_timestamp" },
	{ "timestamptz_ge_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/f:timestamptz_ge_timestamp" },
	{ "timestamptz_gt_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/f:timestamptz_gt_timestamp" },
	{ "timestamptz_ne_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/f:timestamptz_ne_timestamp" },

	/* comparison between intervals */
	{ "interval_eq", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_eq" },
	{ "interval_ne", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_ne" },
	{ "interval_lt", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_lt" },
	{ "interval_le", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_le" },
	{ "interval_ge", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_ge" },
	{ "interval_gt", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_gt" },
	{ "interval_cmp", 2, {INTERVALOID, INTERVALOID}, "t/f:interval_cmp" },

	/* overlaps() */
	{ "overlaps", 4, {TIMEOID, TIMEOID, TIMEOID, TIMEOID},
	  "t/f:overlaps_time" },
	{ "overlaps", 4, {TIMETZOID, TIMETZOID, TIMETZOID, TIMETZOID},
	  "t/f:overlaps_timetz" },
	{ "overlaps", 4, {TIMESTAMPOID, TIMESTAMPOID,
					  TIMESTAMPOID, TIMESTAMPOID},
	  "t/f:overlaps_timestamp" },
	{ "overlaps", 4, {TIMESTAMPTZOID, TIMESTAMPTZOID,
					  TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/f:overlaps_timestamptz" },

	/* extract() */
	{ "date_part", 2, {TEXTOID,TIMESTAMPOID},   "stE/f:extract_timestamp"},
	{ "date_part", 2, {TEXTOID,TIMESTAMPTZOID}, "stE/f:extract_timestamptz"},
	{ "date_part", 2, {TEXTOID,INTERVALOID},    "stE/f:extract_interval"},
	{ "date_part", 2, {TEXTOID,TIMETZOID},      "stE/f:extract_timetz"},
	{ "date_part", 2, {TEXTOID,TIMEOID},        "stE/f:extract_time"},

	/* other time and data functions */
	{ "now", 0, {}, "t/f:now" },

	/* macaddr functions */
	{ "trunc",       1, {MACADDROID},            "y/f:macaddr_trunc" },
	{ "macaddr_not", 1, {MACADDROID},            "y/f:macaddr_not" },
	{ "macaddr_and", 2, {MACADDROID,MACADDROID}, "y/f:macaddr_and" },
	{ "macaddr_or",  2, {MACADDROID,MACADDROID}, "y/f:macaddr_or" },

	/* inet/cidr functions */
	{ "set_masklen", 2, {INETOID,INT4OID},       "y/f:inet_set_masklen" },
	{ "set_masklen", 2, {CIDROID,INT4OID},       "y/f:cidr_set_masklen" },
	{ "family",      1, {INETOID},               "y/f:inet_family" },
	{ "network",     1, {INETOID},               "y/f:network_network" },
	{ "netmask",     1, {INETOID},               "y/f:inet_netmask" },
	{ "masklen",     1, {INETOID},               "y/f:inet_masklen" },
	{ "broadcast",   1, {INETOID},               "y/f:inet_broadcast" },
	{ "hostmask",    1, {INETOID},               "y/f:inet_hostmask" },
	{ "cidr",        1, {INETOID},               "y/f:inet_to_cidr" },
	{ "inetnot",     1, {INETOID},               "y/f:inet_not" },
	{ "inetand",     2, {INETOID,INETOID},       "y/f:inet_and" },
	{ "inetor",      2, {INETOID,INETOID},       "y/f:inet_or" },
	{ "inetpl",      2, {INETOID,INT8OID},       "y/f:inetpl_int8" },
	{ "inetmi_int8", 2, {INETOID,INT8OID},       "y/f:inetmi_int8" },
	{ "inetmi",      2, {INETOID,INETOID},       "y/f:inetmi" },
	{ "inet_same_family", 2, {INETOID,INETOID},  "y/f:inet_same_family" },
	{ "inet_merge",  2, {INETOID,INETOID},       "y/f:inet_merge" },

	/*
	 * Text functions
	 * ---------------------- */
	{ "bpchareq",  2, {BPCHAROID,BPCHAROID},  "s/f:bpchareq" },
	{ "bpcharne",  2, {BPCHAROID,BPCHAROID},  "s/f:bpcharne" },
	{ "bpcharlt",  2, {BPCHAROID,BPCHAROID},  "sc/f:bpcharlt" },
	{ "bpcharle",  2, {BPCHAROID,BPCHAROID},  "sc/f:bpcharle" },
	{ "bpchargt",  2, {BPCHAROID,BPCHAROID},  "sc/f:bpchargt" },
	{ "bpcharge",  2, {BPCHAROID,BPCHAROID},  "sc/f:bpcharge" },
	{ "bpcharcmp", 2, {BPCHAROID, BPCHAROID}, "sc/f:type_compare"},
	{ "length",    1, {BPCHAROID},            "sc/f:bpcharlen"},
	{ "texteq",    2, {TEXTOID, TEXTOID},     "s/f:texteq" },
	{ "textne",    2, {TEXTOID, TEXTOID},     "s/f:textne" },
	{ "text_lt",   2, {TEXTOID, TEXTOID},     "sc/f:text_lt" },
	{ "text_le",   2, {TEXTOID, TEXTOID},     "sc/f:text_le" },
	{ "text_gt",   2, {TEXTOID, TEXTOID},     "sc/f:text_gt" },
	{ "text_ge",   2, {TEXTOID, TEXTOID},     "sc/f:text_ge" },
	{ "bttextcmp", 2, {TEXTOID, TEXTOID},     "sc/f:type_compare" },
	{ "length",    1, {TEXTOID},              "sc/f:textlen" },
	/* LIKE operators */
	{ "like",        2, {TEXTOID, TEXTOID},   "s/f:textlike" },
	{ "textlike",    2, {TEXTOID, TEXTOID},   "s/f:textlike" },
	{ "bpcharlike",  2, {BPCHAROID, TEXTOID}, "s/f:textlike" },
	{ "notlike",     2, {TEXTOID, TEXTOID},   "s/f:textnlike" },
	{ "textnlike",   2, {TEXTOID, TEXTOID},   "s/f:textnlike" },
	{ "bpcharnlike", 2, {BPCHAROID, TEXTOID}, "s/f:textnlike" },
	/* ILIKE operators */
	{ "texticlike",    2, {TEXTOID, TEXTOID},   "sc/f:texticlike" },
	{ "bpchariclike",  2, {TEXTOID, TEXTOID},   "sc/f:texticlike" },
	{ "texticnlike",   2, {TEXTOID, TEXTOID},   "sc/f:texticnlike" },
	{ "bpcharicnlike", 2, {BPCHAROID, TEXTOID}, "sc/f:texticnlike" },
};

/*
 * device function catalog for extra SQL functions
 */
typedef struct devfunc_extra_catalog_t {
	const char *func_rettype;
	const char *func_signature;
	const char *func_template;
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
	{ FLOAT4,  "pgstrom.float4("FLOAT2")",  "p/f:to_float4" },
	{ FLOAT8,  "pgstrom.float8("FLOAT2")",  "p/f:to_float8" },
	{ INT2,    "pgstrom.int2("FLOAT2")",    "p/f:to_int2" },
	{ INT4,    "pgstrom.int4("FLOAT2")",    "p/f:to_int4" },
	{ INT8,    "pgstrom.int8("FLOAT2")",    "p/f:to_int8" },
	{ NUMERIC, "pgstrom.numeric("FLOAT2")", "n/f:float2_numeric" },
	{ FLOAT2,  "pgstrom.float2("FLOAT4")",  "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("FLOAT8")",  "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT2")",    "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT4")",    "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT8")",    "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("NUMERIC")", "n/f:numeric_float2" },
	/* float2 - type comparison functions */
	{ BOOL,    "pgstrom.float2_eq("FLOAT2","FLOAT2")",  "p/f:float2eq" },
	{ BOOL,    "pgstrom.float2_ne("FLOAT2","FLOAT2")",  "p/f:float2ne" },
	{ BOOL,    "pgstrom.float2_lt("FLOAT2","FLOAT2")",  "p/f:float2lt" },
	{ BOOL,    "pgstrom.float2_le("FLOAT2","FLOAT2")",  "p/f:float2le" },
	{ BOOL,    "pgstrom.float2_gt("FLOAT2","FLOAT2")",  "p/f:float2gt" },
	{ BOOL,    "pgstrom.float2_ge("FLOAT2","FLOAT2")",  "p/f:float2ge" },
	{ BOOL,    "pgstrom.float2_cmp("FLOAT2","FLOAT2")", "p/f:type_compare" },

	{ BOOL,    "pgstrom.float42_eq("FLOAT4","FLOAT2")", "p/f:float42eq" },
	{ BOOL,    "pgstrom.float42_ne("FLOAT4","FLOAT2")", "p/f:float42ne" },
	{ BOOL,    "pgstrom.float42_lt("FLOAT4","FLOAT2")", "p/f:float42lt" },
	{ BOOL,    "pgstrom.float42_le("FLOAT4","FLOAT2")", "p/f:float42le" },
	{ BOOL,    "pgstrom.float42_gt("FLOAT4","FLOAT2")", "p/f:float42gt" },
	{ BOOL,    "pgstrom.float42_ge("FLOAT4","FLOAT2")", "p/f:float42ge" },
	{ BOOL,    "pgstrom.float42_cmp("FLOAT4","FLOAT2")", "p/f:type_compare" },

	{ BOOL,    "pgstrom.float82_eq("FLOAT8","FLOAT2")", "p/f:float82eq" },
	{ BOOL,    "pgstrom.float82_ne("FLOAT8","FLOAT2")", "p/f:float82ne" },
	{ BOOL,    "pgstrom.float82_lt("FLOAT8","FLOAT2")", "p/f:float82lt" },
	{ BOOL,    "pgstrom.float82_le("FLOAT8","FLOAT2")", "p/f:float82le" },
	{ BOOL,    "pgstrom.float82_gt("FLOAT8","FLOAT2")", "p/f:float82gt" },
	{ BOOL,    "pgstrom.float82_ge("FLOAT8","FLOAT2")", "p/f:float82ge" },
	{ BOOL,    "pgstrom.float82_cmp("FLOAT8","FLOAT2")", "p/f:type_compare" },

	{ BOOL,    "pgstrom.float24_eq("FLOAT2","FLOAT4")", "p/f:float24eq" },
	{ BOOL,    "pgstrom.float24_ne("FLOAT2","FLOAT4")", "p/f:float24ne" },
	{ BOOL,    "pgstrom.float24_lt("FLOAT2","FLOAT4")", "p/f:float24lt" },
	{ BOOL,    "pgstrom.float24_le("FLOAT2","FLOAT4")", "p/f:float24le" },
	{ BOOL,    "pgstrom.float24_gt("FLOAT2","FLOAT4")", "p/f:float24gt" },
	{ BOOL,    "pgstrom.float24_ge("FLOAT2","FLOAT4")", "p/f:float24ge" },
	{ BOOL,    "pgstrom.float24_cmp("FLOAT2","FLOAT4")", "p/f:type_compare" },

	{ BOOL,    "pgstrom.float28_eq("FLOAT2","FLOAT8")", "p/f:float28eq" },
	{ BOOL,    "pgstrom.float28_ne("FLOAT2","FLOAT8")", "p/f:float28ne" },
	{ BOOL,    "pgstrom.float28_lt("FLOAT2","FLOAT8")", "p/f:float28lt" },
	{ BOOL,    "pgstrom.float28_le("FLOAT2","FLOAT8")", "p/f:float28le" },
	{ BOOL,    "pgstrom.float28_gt("FLOAT2","FLOAT8")", "p/f:float28gt" },
	{ BOOL,    "pgstrom.float28_ge("FLOAT2","FLOAT8")", "p/f:float28ge" },
	{ BOOL,    "pgstrom.float28_cmp("FLOAT2","FLOAT8")", "p/f:type_compare" },

	/* float2 - unary operator */
	{ FLOAT2,  "pgstrom.float2_up("FLOAT2")",  "p/f:float2up" },
	{ FLOAT2,  "pgstrom.float2_um("FLOAT2")",  "p/f:float2um" },
	{ FLOAT2,  "abs("FLOAT2")", "p/f:float2abs" },

	/* float2 - arithmetic operators */
	{ FLOAT4, "pgstrom.float2_pl("FLOAT2","FLOAT2")",   "m/f:float2pl" },
	{ FLOAT4, "pgstrom.float2_mi("FLOAT2","FLOAT2")",   "m/f:float2mi" },
	{ FLOAT4, "pgstrom.float2_mul("FLOAT2","FLOAT2")",  "m/f:float2mul" },
	{ FLOAT4, "pgstrom.float2_div("FLOAT2","FLOAT2")",  "m/f:float2div" },
	{ FLOAT4, "pgstrom.float24_pl("FLOAT2","FLOAT4")",  "m/f:float24pl" },
	{ FLOAT4, "pgstrom.float24_mi("FLOAT2","FLOAT4")",  "m/f:float24mi" },
	{ FLOAT4, "pgstrom.float24_mul("FLOAT2","FLOAT4")", "m/f:float24mul" },
	{ FLOAT4, "pgstrom.float24_div("FLOAT2","FLOAT4")", "m/f:float24div" },
	{ FLOAT8, "pgstrom.float28_pl("FLOAT2","FLOAT8")",  "m/f:float28pl" },
	{ FLOAT8, "pgstrom.float28_mi("FLOAT2","FLOAT8")",  "m/f:float28mi" },
	{ FLOAT8, "pgstrom.float28_mul("FLOAT2","FLOAT8")", "m/f:float28mul" },
	{ FLOAT8, "pgstrom.float28_div("FLOAT2","FLOAT8")", "m/f:float28div" },
	{ FLOAT4, "pgstrom.float42_pl("FLOAT4","FLOAT2")",  "m/f:float42pl" },
	{ FLOAT4, "pgstrom.float42_mi("FLOAT4","FLOAT2")",  "m/f:float42mi" },
	{ FLOAT4, "pgstrom.float42_mul("FLOAT4","FLOAT2")", "m/f:float42mul" },
	{ FLOAT4, "pgstrom.float42_div("FLOAT4","FLOAT2")", "m/f:float42div" },
	{ FLOAT8, "pgstrom.float82_pl("FLOAT8","FLOAT2")",  "m/f:float82pl" },
	{ FLOAT8, "pgstrom.float82_mi("FLOAT8","FLOAT2")",  "m/f:float82mi" },
	{ FLOAT8, "pgstrom.float82_mul("FLOAT8","FLOAT2")", "m/f:float82mul" },
	{ FLOAT8, "pgstrom.float82_div("FLOAT8","FLOAT2")", "m/f:float82div" },
	{ "money", "pgstrom.cash_mul_flt2(money,"FLOAT2")", "y/f:cash_mul_flt2" },
	{ "money", "pgstrom.flt2_mul_cash("FLOAT2",money)", "y/f:flt2_mul_cash" },
	{ "money", "pgstrom.cash_div_flt2(money,"FLOAT2")", "y/f:cash_div_flt2" },

	/* int4range operators */
	{ INT4, "lower(int4range)",               "r/f:int4range_lower" },
	{ INT4, "upper(int4range)",               "r/f:int4range_upper" },
	{ BOOL, "isempty(int4range)",             "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(int4range)",           "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int4range)",           "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int4range)",           "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int4range)",           "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(int4range,int4range)",  "r/f:generic_range_eq" },
	{ BOOL, "range_ne(int4range,int4range)",  "r/f:generic_range_ne" },
	{ BOOL, "range_lt(int4range,int4range)",  "r/f:generic_range_lt" },
	{ BOOL, "range_le(int4range,int4range)",  "r/f:generic_range_le" },
	{ BOOL, "range_gt(int4range,int4range)",  "r/f:generic_range_gt" },
	{ BOOL, "range_ge(int4range,int4range)",  "r/f:generic_range_ge" },
	{ INT4, "range_cmp(int4range,int4range)", "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(int4range,int4range)",
	  "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int4range,"INT4")",
	  "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(int4range,int4range)",
	  "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT4",int4range)",
	  "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int4range,int4range)",
	  "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int4range,int4range)",
	  "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(int4range,int4range)",
	  "r/f:generic_range_before" },
	{ BOOL, "range_after(int4range,int4range)",
	  "r/f:generic_range_after" },
	{ BOOL, "range_overleft(int4range,int4range)",
	  "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(int4range,int4range)",
	  "r/f:generic_range_overleft" },
	{ "int4range", "range_union(int4range,int4range)",
	  "r/f:generic_range_union" },
	{ "int4range", "range_merge(int4range,int4range)",
	  "r/f:generic_range_merge" },
	{ "int4range", "range_intersect(int4range,int4range)",
	  "r/f:generic_range_intersect" },
	{ "int4range", "range_minus(int4range,int4range)",
	  "r/f:generic_range_minus" },

	/* int8range operators */
	{ INT8, "lower(int8range)",               "r/f:int8range_lower" },
	{ INT8, "upper(int8range)",               "r/f:int8range_upper" },
	{ BOOL, "isempty(int8range)",             "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(int8range)",           "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int8range)",           "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int8range)",           "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int8range)",           "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(int8range,int8range)",  "r/f:generic_range_eq" },
	{ BOOL, "range_ne(int8range,int8range)",  "r/f:generic_range_ne" },
	{ BOOL, "range_lt(int8range,int8range)",  "r/f:generic_range_lt" },
	{ BOOL, "range_le(int8range,int8range)",  "r/f:generic_range_le" },
	{ BOOL, "range_gt(int8range,int8range)",  "r/f:generic_range_gt" },
	{ BOOL, "range_ge(int8range,int8range)",  "r/f:generic_range_ge" },
	{ INT4, "range_cmp(int8range,int8range)", "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(int8range,int8range)",
	  "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int8range,"INT8")",
	  "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(int8range,int8range)",
	  "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT8",int8range)",
	  "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int8range,int8range)",
	  "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int8range,int8range)",
	  "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(int8range,int8range)",
	  "r/f:generic_range_before" },
	{ BOOL, "range_after(int8range,int8range)",
	  "r/f:generic_range_after" },
	{ BOOL, "range_overleft(int8range,int8range)",
	  "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(int8range,int8range)",
	  "r/f:generic_range_overleft" },
	{ "int8range", "range_union(int8range,int8range)",
	  "r/f:generic_range_union" },
	{ "int8range", "range_merge(int8range,int8range)",
	  "r/f:generic_range_merge" },
	{ "int8range", "range_intersect(int8range,int8range)",
	  "r/f:generic_range_intersect" },
	{ "int8range", "range_minus(int8range,int8range)",
	  "r/f:generic_range_minus" },

	/* tsrange operators */
	{ "timestamp", "lower(tsrange)",      "r/f:tsrange_lower" },
	{ "timestamp", "upper(tsrange)",      "r/f:tsrange_upper" },
	{ BOOL, "isempty(tsrange)",           "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(tsrange)",         "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tsrange)",         "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tsrange)",         "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tsrange)",         "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(tsrange,tsrange)",  "r/f:generic_range_eq" },
	{ BOOL, "range_ne(tsrange,tsrange)",  "r/f:generic_range_ne" },
	{ BOOL, "range_lt(tsrange,tsrange)",  "r/f:generic_range_lt" },
	{ BOOL, "range_le(tsrange,tsrange)",  "r/f:generic_range_le" },
	{ BOOL, "range_gt(tsrange,tsrange)",  "r/f:generic_range_gt" },
	{ BOOL, "range_ge(tsrange,tsrange)",  "r/f:generic_range_ge" },
	{ INT4, "range_cmp(tsrange,tsrange)", "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(tsrange,tsrange)",
	  "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tsrange,timestamp)",
	  "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(tsrange,tsrange)",
	  "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamp,tsrange)",
	  "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tsrange,tsrange)",
	  "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tsrange,tsrange)",
	  "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(tsrange,tsrange)",
	  "r/f:generic_range_before" },
	{ BOOL, "range_after(tsrange,tsrange)",
	  "r/f:generic_range_after" },
	{ BOOL, "range_overleft(tsrange,tsrange)",
	  "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(tsrange,tsrange)",
	  "r/f:generic_range_overleft" },
	{ "tsrange", "range_union(tsrange,tsrange)",
	  "r/f:generic_range_union" },
	{ "tsrange", "range_merge(tsrange,tsrange)",
	  "r/f:generic_range_merge" },
	{ "tsrange", "range_intersect(tsrange,tsrange)",
	  "r/f:generic_range_intersect" },
	{ "tsrange", "range_minus(tsrange,tsrange)",
	  "r/f:generic_range_minus" },

	/* tstzrange operators */
	{ "timestamptz", "lower(tstzrange)",      "r/f:tstzrange_lower" },
	{ "timestamptz", "upper(tstzrange)",      "r/f:tstzrange_upper" },
	{ BOOL, "isempty(tstzrange)",             "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(tstzrange)",           "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tstzrange)",           "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tstzrange)",           "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tstzrange)",           "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(tstzrange,tstzrange)",  "r/f:generic_range_eq" },
	{ BOOL, "range_ne(tstzrange,tstzrange)",  "r/f:generic_range_ne" },
	{ BOOL, "range_lt(tstzrange,tstzrange)",  "r/f:generic_range_lt" },
	{ BOOL, "range_le(tstzrange,tstzrange)",  "r/f:generic_range_le" },
	{ BOOL, "range_gt(tstzrange,tstzrange)",  "r/f:generic_range_gt" },
	{ BOOL, "range_ge(tstzrange,tstzrange)",  "r/f:generic_range_ge" },
	{ INT4, "range_cmp(tstzrange,tstzrange)", "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(tstzrange,tstzrange)",
	  "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tstzrange,timestamptz)",
	  "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(tstzrange,tstzrange)",
	  "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamptz,tstzrange)",
	  "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tstzrange,tstzrange)",
	  "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tstzrange,tstzrange)",
	  "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(tstzrange,tstzrange)",
	  "r/f:generic_range_before" },
	{ BOOL, "range_after(tstzrange,tstzrange)",
	  "r/f:generic_range_after" },
	{ BOOL, "range_overleft(tstzrange,tstzrange)",
	  "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(tstzrange,tstzrange)",
	  "r/f:generic_range_overleft" },
	{ "tstzrange", "range_union(tstzrange,tstzrange)",
	  "r/f:generic_range_union" },
	{ "tstzrange", "range_merge(tstzrange,tstzrange)",
	  "r/f:generic_range_merge" },
	{ "tstzrange", "range_intersect(tstzrange,tstzrange)",
	  "r/f:generic_range_intersect" },
	{ "tstzrange", "range_minus(tstzrange,tstzrange)",
	  "r/f:generic_range_minus" },

	/* daterange operators */
	{ "date", "lower(daterange)",             "r/f:daterange_lower" },
	{ "date", "upper(daterange)",             "r/f:daterange_upper" },
	{ BOOL, "isempty(daterange)",             "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(daterange)",           "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(daterange)",           "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(daterange)",           "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(daterange)",           "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(daterange,daterange)",  "r/f:generic_range_eq" },
	{ BOOL, "range_ne(daterange,daterange)",  "r/f:generic_range_ne" },
	{ BOOL, "range_lt(daterange,daterange)",  "r/f:generic_range_lt" },
	{ BOOL, "range_le(daterange,daterange)",  "r/f:generic_range_le" },
	{ BOOL, "range_gt(daterange,daterange)",  "r/f:generic_range_gt" },
	{ BOOL, "range_ge(daterange,daterange)",  "r/f:generic_range_ge" },
	{ INT4, "range_cmp(daterange,daterange)", "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(daterange,daterange)",
	  "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(daterange,date)",
	  "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(daterange,daterange)",
	  "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(date,daterange)",
	  "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(daterange,daterange)",
	  "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(daterange,daterange)",
	  "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(daterange,daterange)",
	  "r/f:generic_range_before" },
	{ BOOL, "range_after(daterange,daterange)",
	  "r/f:generic_range_after" },
	{ BOOL, "range_overleft(daterange,daterange)",
	  "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(daterange,daterange)",
	  "r/f:generic_range_overleft" },
	{ "daterange", "range_union(daterange,daterange)",
	  "r/f:generic_range_union" },
	{ "daterange", "range_merge(daterange,daterange)",
	  "r/f:generic_range_merge" },
	{ "daterange", "range_intersect(daterange,daterange)",
	  "r/f:generic_range_intersect" },
	{ "daterange", "range_minus(daterange,daterange)",
	  "r/f:generic_range_minus" },

	/* type re-interpretation */
	{ INT8,   "as_int8("FLOAT8")", "p/f:as_int8" },
	{ INT4,   "as_int4("FLOAT4")", "p/f:as_int4" },
	{ INT2,   "as_int2("FLOAT2")", "p/f:as_int2" },
	{ FLOAT8, "as_float8("INT8")", "p/f:as_float8" },
	{ FLOAT4, "as_float4("INT4")", "p/f:as_float4" },
	{ FLOAT2, "as_float2("INT2")", "p/f:as_float2" },
};

#undef BOOL
#undef INT2
#undef INT4
#undef INT8
#undef FLOAT2
#undef FLOAT4
#undef FLOAT8
#undef NUMERIC

static bool
__construct_devfunc_info(devfunc_info *entry,
						 const char *template)
{
	const char *pos;
	const char *end;
	int32		flags = 0;
	bool		has_collation = false;

	/* fetch attribute */
	end = strchr(template, '/');
	if (end)
	{
		for (pos = template; pos < end; pos++)
		{
			switch (*pos)
			{
				case 'c':
					has_collation = true;
					break;
				case 'p':
					flags |= DEVKERNEL_NEEDS_PRIMITIVE;
					break;
				case 'n':
					flags |= DEVKERNEL_NEEDS_NUMERIC;
					break;
				case 'm':
					flags |= DEVKERNEL_NEEDS_MATHLIB;
					break;
				case 's':
					flags |= DEVKERNEL_NEEDS_TEXTLIB;
					break;
				case 't':
					flags |= DEVKERNEL_NEEDS_TIMELIB;
					break;
				case 'y':
					flags |= DEVKERNEL_NEEDS_MISC;
					break;
				case 'r':
					flags |= DEVKERNEL_NEEDS_RANGETYPE;
					break;
				case 'E':
					flags |= DEVKERNEL_NEEDS_TIME_EXTRACT;
					break;
				default:
					elog(NOTICE,
						 "Bug? unkwnon devfunc property: %c",
						 *pos);
					break;
			}
		}
		template = end + 1;
	}
	entry->func_flags = flags;

	/*
	 * If function is collation aware but not supported to run on GPU device,
	 * we have to give up to generate device code.
	 */
	if (!has_collation)
		entry->func_collid = InvalidOid;	/* clear default if any */
	else if (OidIsValid(entry->func_collid) &&
			 !lc_collate_is_c(entry->func_collid))
		return false;		/* unable to run on device */

	if (strncmp(template, "f:", 2) == 0)
		entry->func_devname = template + 2;
	else
	{
		elog(NOTICE, "Bug? unknown device function template: '%s'",
			 template);
		return false;
	}
	return true;
}

static bool
pgstrom_devfunc_construct_common(devfunc_info *entry)
{
	int		i, j;

	for (i=0; i < lengthof(devfunc_common_catalog); i++)
	{
		devfunc_catalog_t  *procat = devfunc_common_catalog + i;

		if (strcmp(procat->func_name, entry->func_sqlname) == 0 &&
			procat->func_nargs == list_length(entry->func_args))
		{
			ListCell   *lc;

			j = 0;
			foreach (lc, entry->func_args)
			{
				devtype_info   *dtype = lfirst(lc);

				if (dtype->type_oid != procat->func_argtypes[j++])
					break;
			}
			if (lc == NULL)
				return __construct_devfunc_info(entry, procat->func_template);
		}
	}
	return false;
}

static bool
pgstrom_devfunc_construct_extra(devfunc_info *entry, HeapTuple protup)
{
	Form_pg_proc proc = (Form_pg_proc) GETSTRUCT(protup);
	StringInfoData sig;
	ListCell   *lc;
	int			i;
	bool		result = false;
	char	   *func_rettype;
	char	   *temp;

	/* make a signature string */
	initStringInfo(&sig);
	if (proc->pronamespace != PG_CATALOG_NAMESPACE)
	{
		temp = get_namespace_name(proc->pronamespace);
		appendStringInfo(&sig, "%s.", quote_identifier(temp));
		pfree(temp);
	}

	appendStringInfo(&sig, "%s(", quote_identifier(NameStr(proc->proname)));
	foreach (lc, entry->func_args)
	{
		devtype_info   *dtype = lfirst(lc);

		if (lc != list_head(entry->func_args))
			appendStringInfoChar(&sig, ',');
		temp = format_type_be_qualified(dtype->type_oid);
		if (strncmp(temp, "pg_catalog.", 11) == 0)
			appendStringInfo(&sig, "%s", temp + 11);
		else
			appendStringInfo(&sig, "%s", temp);
		pfree(temp);
	}
	appendStringInfoChar(&sig, ')');

	temp = format_type_be_qualified(entry->func_rettype->type_oid);
	if (strncmp(temp, "pg_catalog.", 11) == 0)
		func_rettype = temp + 11;
	else
		func_rettype = temp;

	for (i=0; i < lengthof(devfunc_extra_catalog); i++)
	{
		devfunc_extra_catalog_t  *procat = devfunc_extra_catalog + i;

		if (strcmp(procat->func_signature, sig.data) == 0 &&
			strcmp(procat->func_rettype, func_rettype) == 0)
		{
			result = __construct_devfunc_info(entry, procat->func_template);
			goto found;
		}
	}
	elog(DEBUG2, "no extra function found for sig=[%s] rettype=[%s]",
		 sig.data, func_rettype);
found:
	pfree(sig.data);
	pfree(temp);
	return result;
}

static devfunc_info *
__pgstrom_devfunc_lookup_or_create(HeapTuple protup,
								   Oid func_rettype,
								   oidvector *func_argtypes,
								   Oid func_collid)
{
	Oid				func_oid = HeapTupleGetOid(protup);
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	devfunc_info   *entry;
	devtype_info   *dtype;
	List		   *func_args = NIL;
	ListCell	   *lc;
	ListCell	   *cell;
	int				i, j;

	i = func_oid % lengthof(devfunc_info_slot);
	foreach (lc, devfunc_info_slot[i])
	{
		entry = lfirst(lc);

		if (entry->func_oid == func_oid &&
			list_length(entry->func_args) == func_argtypes->dim1 &&
			(!OidIsValid(entry->func_collid) ||
			 entry->func_collid == func_collid))
		{
			j = 0;
			foreach (cell, entry->func_args)
			{
				dtype = lfirst(cell);
				if (dtype->type_oid != func_argtypes->values[j])
					break;
				j++;
			}
			if (!cell)
			{
				if (entry->func_is_negative)
					return NULL;
				return entry;
			}
		}
	}

	/*
	 * Not found, construct a new entry of the device function
	 */
	entry = MemoryContextAllocZero(devinfo_memcxt,
								   sizeof(devfunc_info));
	entry->func_oid = func_oid;
	entry->func_collid = func_collid;	/* may be cleared later */
	entry->func_is_strict = proc->proisstrict;
	Assert(proc->pronargs == func_argtypes->dim1);
	for (j=0; j < proc->pronargs; j++)
	{
		dtype = pgstrom_devtype_lookup(func_argtypes->values[j]);
		if (!dtype)
		{
			list_free(func_args);
			entry->func_is_negative = true;
			goto skip;
		}
		func_args = lappend_cxt(devinfo_memcxt, func_args, dtype);
	}

	dtype = pgstrom_devtype_lookup(func_rettype);
	if (!dtype)
	{
		list_free(func_args);
		entry->func_is_negative = true;
		goto skip;
	}
	entry->func_args = func_args;
	entry->func_rettype = dtype;
	entry->func_sqlname = pstrdup(NameStr(proc->proname));

	if (proc->pronamespace == PG_CATALOG_NAMESPACE
		/* for system default functions (pg_catalog) */
		? pgstrom_devfunc_construct_common(entry)
		/* other extra or polymorphic functions */
		: pgstrom_devfunc_construct_extra(entry, protup))
	{
		entry->func_is_negative = false;
	}
	else
	{
		/* oops, function has no entry */
		entry->func_is_negative = true;
	}
skip:
	devfunc_info_slot[i] = lappend_cxt(devinfo_memcxt,
									   devfunc_info_slot[i],
									   entry);
	if (entry->func_is_negative)
		return NULL;
	return entry;
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
											   func_collid);
	if (!dfunc)
	{
		/*
		 * NOTE: In some cases, function might be called with different
		 * argument types or result type from its definition, if both of
		 * the types are binary compatible.
		 * For example, type equality function of varchar(N) is texteq.
		 * It is legal and adequate in PostgreSQL, however, CUDA C++ code
		 * takes strict type checks, so we have to inject type relabel
		 * in this case.
		 */
		Form_pg_proc proc = (Form_pg_proc) GETSTRUCT(protup);
		oidvector  *proargtypes = &proc->proargtypes;
		HeapTuple	tuple;
		Oid			src_type;
		Oid			dst_type;
		char		castmethod;
		int			j;

		if (func_argtypes->dim1 != proargtypes->dim1)
			return NULL;
		for (j = 0; j < proargtypes->dim1; j++)
		{
			src_type = func_argtypes->values[j];
			dst_type = proargtypes->values[j];

			if (src_type == dst_type)
				continue;
			tuple = SearchSysCache2(CASTSOURCETARGET,
                                    ObjectIdGetDatum(src_type),
									ObjectIdGetDatum(dst_type));
			if (!HeapTupleIsValid(tuple))
			{
				elog(DEBUG2, "no type cast definition (%s->%s)",
					 format_type_be(src_type),
					 format_type_be(dst_type));
				return NULL;	/* no cast */
			}
			castmethod = ((Form_pg_cast) GETSTRUCT(tuple))->castmethod;
			ReleaseSysCache(tuple);

			/*
			 * It might be possible to inject device function to cast
			 * source type to the destination type. However, it should
			 * be already attached by the code PostgreSQL.
			 * Right now, we don't support it.
			 */
			if (castmethod != COERCION_METHOD_BINARY)
			{
				elog(DEBUG2, "not binary compatible type cast (%s->%s)",
					 format_type_be(src_type),
					 format_type_be(dst_type));
				return NULL;
			}
		}

		if (func_rettype != proc->prorettype)
		{
			tuple = SearchSysCache2(CASTSOURCETARGET,
									ObjectIdGetDatum(func_rettype),
									ObjectIdGetDatum(proc->prorettype));
			if (!HeapTupleIsValid(tuple))
			{
				elog(DEBUG2, "no type cast definition (%s->%s)",
					 format_type_be(func_rettype),
					 format_type_be(proc->prorettype));
				return NULL;    /* no cast */
			}
			castmethod = ((Form_pg_cast) GETSTRUCT(tuple))->castmethod;
			ReleaseSysCache(tuple);

			if (castmethod != COERCION_METHOD_BINARY)
			{
				elog(DEBUG2, "not binary compatible type cast (%s->%s)",
					 format_type_be(func_rettype),
					 format_type_be(proc->prorettype));
				return NULL;
			}
		}

		/*
		 * OK, it looks type-relabel allows to call the function
		 */
		dfunc = __pgstrom_devfunc_lookup_or_create(protup,
												   proc->prorettype,
												   proargtypes,
												   func_collid);
	}
	return dfunc;
}

static devfunc_info *
pgstrom_devfunc_lookup(Oid func_oid,
					   Oid func_rettype,
					   List *func_args,	/* list of expressions */
					   Oid func_collid)
{
	devfunc_info *result = NULL;
	char		buffer[offsetof(oidvector, values[DEVFUNC_MAX_NARGS])];
	oidvector  *func_argtypes = (oidvector *)buffer;
	HeapTuple	tup;
	Form_pg_proc proc;

	tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	proc = (Form_pg_proc) GETSTRUCT(tup);
	Assert(proc->pronargs == list_length(func_args));
	if (proc->pronargs <= DEVFUNC_MAX_NARGS)
	{
		size_t		len = offsetof(oidvector, values[proc->pronargs]);
		cl_uint		i = 0;
		ListCell   *lc;

		func_argtypes->ndim = 1;
		func_argtypes->dataoffset = 0;
		func_argtypes->elemtype = OIDOID;
		func_argtypes->dim1 = list_length(func_args);
		func_argtypes->lbound1 = 0;
		foreach (lc, func_args)
		{
			Oid		type_oid = exprType((Node *)lfirst(lc));

			func_argtypes->values[i++] = type_oid;
		}
		SET_VARSIZE(func_argtypes, len);

		result = __pgstrom_devfunc_lookup(tup,
										  func_rettype,
										  func_argtypes,
										  func_collid);
	}
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
		elog(ERROR, "cache lookup failed for function %u", dtype->type_eqfunc);
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
        elog(ERROR, "cache lookup failed for function %u", dtype->type_cmpfunc);
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
	ReleaseSysCache(tup);

	return result;
}

void
pgstrom_devfunc_track(codegen_context *context, devfunc_info *dfunc)
{
	ListCell	   *lc;

	/* track device function */
	context->extra_flags |= dfunc->func_flags;
	foreach (lc, context->func_defs)
	{
		devfunc_info   *dtemp = lfirst(lc);

		if (dfunc == dtemp)
			goto skip;
	}
	context->func_defs = lappend(context->func_defs, dfunc);
skip:
	/* track function arguments and result types also */
	pgstrom_devtype_track(context, dfunc->func_rettype);
	foreach (lc, dfunc->func_args)
		pgstrom_devtype_track(context, (devtype_info *) lfirst(lc));
}

/*
 * codegen_expression_walker - main logic of run-time code generator
 */
static void codegen_function_expression(devfunc_info *dfunc, List *args,
										codegen_context *context);

static void
codegen_expression_walker(Node *node, codegen_context *context)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	ListCell	   *cell;

	if (node == NULL)
		return;

	if (IsA(node, Const))
	{
		Const  *con = (Const *) node;
		cl_uint	index = 0;

		if (!pgstrom_devtype_lookup_and_track(con->consttype, context))
			elog(ERROR, "codegen: faied to lookup device type: %s",
				 format_type_be(con->consttype));

		context->used_params = lappend(context->used_params,
									   copyObject(node));
		index = list_length(context->used_params) - 1;
		appendStringInfo(&context->str, "KPARAM_%u", index);
		context->param_refs =
			bms_add_member(context->param_refs, index);
	}
	else if (IsA(node, Param))
	{
		Param  *param = (Param *) node;
		int		index = 0;

		if (param->paramkind != PARAM_EXTERN)
			elog(ERROR, "codegen: ParamKind is not PARAM_EXTERN: %d",
				 (int)param->paramkind);

		if (!pgstrom_devtype_lookup_and_track(param->paramtype, context))
			elog(ERROR, "codegen: faied to lookup device type: %s",
				 format_type_be(param->paramtype));

		foreach (cell, context->used_params)
		{
			if (equal(node, lfirst(cell)))
			{
				appendStringInfo(&context->str, "KPARAM_%u", index);
				context->param_refs =
					bms_add_member(context->param_refs, index);
				return;
			}
			index++;
		}
		context->used_params = lappend(context->used_params,
									   copyObject(node));
		index = list_length(context->used_params) - 1;
		appendStringInfo(&context->str, "KPARAM_%u", index);
		context->param_refs = bms_add_member(context->param_refs, index);
	}
	else if (IsA(node, Var))
	{
		Var			   *var = (Var *) node;
		AttrNumber		varattno = var->varattno;
		devtype_info   *dtype;
		ListCell	   *cell;

		dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
		if (!dtype)
			elog(ERROR, "codegen: faied to lookup device type: %s",
				 format_type_be(var->vartype));
		if (dtype->type_element)
			elog(ERROR, "codegen: array type referenced by Var: %s",
				 format_type_be(var->vartype));

		/* Fixup varattno when pseudo-scan tlist exists, because varattno
		 * shall be adjusted on setrefs.c, so we have to adjust variable
		 * name according to the expected attribute number is kernel-
		 * source shall be constructed prior to setrefs.c / subselect.c
		 */
		if (context->pseudo_tlist != NIL)
		{
			foreach (cell, context->pseudo_tlist)
			{
				TargetEntry *tle = lfirst(cell);
				Var	   *ptv = (Var *) tle->expr;

				if (!IsA(tle->expr, Var) ||
					ptv->varno != var->varno ||
					ptv->varattno != var->varattno ||
					ptv->varlevelsup != var->varlevelsup)
					continue;

				varattno = tle->resno;
				break;
			}
			Assert(cell != NULL);
			if (!cell)
				elog(ERROR, "codegen: failed to map Var (%s)on ps_tlist: %s",
					 nodeToString(var), nodeToString(context->pseudo_tlist));
		}

		if (varattno < 0)
			appendStringInfo(&context->str, "%s_S%u",
							 context->var_label,
							 -varattno);
		else
			appendStringInfo(&context->str, "%s_%u",
							 context->var_label,
							 varattno);
		context->used_vars = list_append_unique(context->used_vars,
												copyObject(node));
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) node;

		dfunc = pgstrom_devfunc_lookup(func->funcid,
									   func->funcresulttype,
									   func->args,
									   func->inputcollid);
		if (!dfunc)
			elog(ERROR, "codegen: failed to lookup device function: %s",
				 format_procedure(func->funcid));
		pgstrom_devfunc_track(context, dfunc);
		codegen_function_expression(dfunc, func->args, context);
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) node;

		dfunc = pgstrom_devfunc_lookup(get_opcode(op->opno),
									   op->opresulttype,
									   op->args,
									   op->inputcollid);
		if (!dfunc)
			elog(ERROR, "codegen: failed to lookup device function: %s",
				 format_procedure(dfunc->func_oid));
		pgstrom_devfunc_track(context, dfunc);
		codegen_function_expression(dfunc, op->args, context);
	}
	else if (IsA(node, NullTest))
	{
		NullTest   *nulltest = (NullTest *) node;
		Oid			typeoid = exprType((Node *)nulltest->arg);

		if (nulltest->argisrow)
			elog(ERROR, "codegen: NullTest towards RECORD data");

		dtype = pgstrom_devtype_lookup_and_track(typeoid, context);
		if (!dtype)
			elog(ERROR, "codegen: failed to lookup device type: %s",
				 format_type_be(typeoid));

		if (nulltest->nulltesttype == IS_NULL)
			appendStringInfo(&context->str, "PG_ISNULL");
		else if (nulltest->nulltesttype == IS_NOT_NULL)
			appendStringInfo(&context->str, "PG_ISNOTNULL");
		else
			elog(ERROR, "unrecognized nulltesttype: %d",
				 (int)nulltest->nulltesttype);

		appendStringInfo(&context->str, "(kcxt, ");
		codegen_expression_walker((Node *) nulltest->arg, context);
		appendStringInfoChar(&context->str, ')');
	}
	else if (IsA(node, BooleanTest))
	{
		BooleanTest	   *booltest = (BooleanTest *) node;
		const char	   *func_name;

		if (exprType((Node *)booltest->arg) != BOOLOID)
			elog(ERROR, "argument of BooleanTest is not bool");

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
				elog(ERROR, "unrecognized booltesttype: %d",
					 (int)booltest->booltesttype);
				break;
		}
		appendStringInfo(&context->str, "pgfn_%s(kcxt, ", func_name);
		codegen_expression_walker((Node *) booltest->arg, context);
		appendStringInfoChar(&context->str, ')');
	}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr   *b = (BoolExpr *) node;

		if (b->boolop == NOT_EXPR)
		{
			Assert(list_length(b->args) == 1);
			appendStringInfo(&context->str, "NOT(");
			codegen_expression_walker(linitial(b->args), context);
			appendStringInfoChar(&context->str, ')');
		}
		else if (b->boolop == AND_EXPR || b->boolop == OR_EXPR)
		{
			Assert(list_length(b->args) > 1);

			appendStringInfo(&context->str, "to_bool(");
			foreach (cell, b->args)
			{
				Assert(exprType(lfirst(cell)) == BOOLOID);
				if (cell != list_head(b->args))
				{
					if (b->boolop == AND_EXPR)
						appendStringInfo(&context->str, " && ");
					else
						appendStringInfo(&context->str, " || ");
				}
				appendStringInfo(&context->str, "EVAL(");
				codegen_expression_walker(lfirst(cell), context);
				appendStringInfoChar(&context->str, ')');
			}
			appendStringInfoChar(&context->str, ')');
		}
		else
			elog(ERROR, "unrecognized boolop: %d", (int) b->boolop);
	}
	else if (IsA(node, CoalesceExpr))
	{
		CoalesceExpr   *coalesce = (CoalesceExpr *) node;

		dtype = pgstrom_devtype_lookup(coalesce->coalescetype);
		if (!dtype)
			elog(ERROR, "codegen: unsupported device type in COALESCE: %s",
				 format_type_be(coalesce->coalescetype));

		appendStringInfo(&context->str, "PG_COALESCE(kcxt");
		foreach (cell, coalesce->args)
		{
			Node   *expr = (Node *)lfirst(cell);
			Oid		type_oid = exprType(expr);

			if (dtype->type_oid != type_oid)
				elog(ERROR, "device type mismatch in COALESCE: %s / %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(type_oid));
			appendStringInfo(&context->str, ", ");
			codegen_expression_walker(expr, context);
		}
		appendStringInfo(&context->str, ")");
	}
	else if (IsA(node, MinMaxExpr))
	{
		MinMaxExpr	   *minmax = (MinMaxExpr *) node;

		dtype = pgstrom_devtype_lookup(minmax->minmaxtype);
		if (!dtype)
			elog(ERROR, "unsupported device type in LEAST/GREATEST: %s",
				 format_type_be(minmax->minmaxtype));

		dfunc = pgstrom_devfunc_lookup_type_compare(dtype,
													minmax->inputcollid);
		if (!dfunc)
			elog(ERROR, "unsupported device type in LEAST/GREATEST: %s",
				 format_type_be(minmax->minmaxtype));

		if (minmax->op == IS_GREATEST)
			appendStringInfo(&context->str, "PG_GREATEST");
		else if (minmax->op == IS_LEAST)
			appendStringInfo(&context->str, "PG_LEAST");
		else
			elog(ERROR, "unknown operation at MinMaxExpr: %d",
                 (int)minmax->op);

		appendStringInfo(&context->str, "(kcxt");
		foreach (cell, minmax->args)
		{
			Node   *expr = lfirst(cell);
			Oid		type_oid = exprType(expr);

			if (dtype->type_oid != type_oid)
				elog(ERROR, "device type mismatch in LEAST/GREATEST: %s / %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(exprType(expr)));
			appendStringInfo(&context->str, ", ");
			codegen_expression_walker(expr, context);
		}
		appendStringInfoChar(&context->str, ')');
	}
	else if (IsA(node, RelabelType))
	{
		RelabelType *relabel = (RelabelType *) node;

		dtype = pgstrom_devtype_lookup_and_track(relabel->resulttype, context);
		if (!dtype)
			elog(ERROR, "codegen: failed to lookup device type: %s",
				 format_type_be(relabel->resulttype));
		appendStringInfo(&context->str, "to_%s(", dtype->type_name);
		codegen_expression_walker((Node *)relabel->arg, context);
		appendStringInfo(&context->str, ")");
	}
	else if (IsA(node, CaseExpr))
	{
		CaseExpr   *caseexpr = (CaseExpr *) node;
		ListCell   *cell;
		Oid			type_oid;

		if (caseexpr->arg)
		{
			appendStringInfo(
				&context->str,
				"PG_CASEWHEN_%s(kcxt,",
				caseexpr->defresult ? "ELSE" : "EXPR");

			/* type compare function internally used */
			type_oid = exprType((Node *) caseexpr->arg);
			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype)
				elog(ERROR, "codegen: failed to lookup device type: %s",
					 format_type_be(type_oid));
			dfunc = pgstrom_devfunc_lookup_type_compare(dtype, InvalidOid);
			if (!dfunc)
				elog(ERROR, "codegen: failed to lookup type compare func: %s",
					 format_type_be(type_oid));
			pgstrom_devfunc_track(context, dfunc);

			codegen_expression_walker((Node *) caseexpr->arg, context);
			if (caseexpr->defresult)
			{
				appendStringInfo(&context->str, ", ");
				codegen_expression_walker((Node *)caseexpr->defresult,
										  context);
			}
			foreach (cell, caseexpr->args)
			{
				CaseWhen   *casewhen = (CaseWhen *) lfirst(cell);
				OpExpr	   *op_expr = (OpExpr *) casewhen->expr;
				Node	   *test_val;

				Assert(IsA(casewhen, CaseWhen));
				if (!IsA(op_expr, OpExpr) ||
					op_expr->opresulttype != BOOLOID ||
					list_length(op_expr->args) != 2)
					elog(ERROR, "Bug? unexpected expression node at CASE ... WHEN");
				if (IsA(linitial(op_expr->args), CaseTestExpr) &&
					!IsA(lsecond(op_expr->args), CaseTestExpr))
					test_val = lsecond(op_expr->args);
				else if (!IsA(linitial(op_expr->args), CaseTestExpr) &&
						 IsA(lsecond(op_expr->args), CaseTestExpr))
					test_val = linitial(op_expr->args);
				else
					elog(ERROR, "Bug? CaseTestExpr is expected for either of OpExpr args");
				appendStringInfo(&context->str, ", ");
				codegen_expression_walker(test_val, context);
				appendStringInfo(&context->str, ", ");
				codegen_expression_walker((Node *)casewhen->result, context);
			}
			appendStringInfo(&context->str, ")");
		}
		else
		{
			foreach (cell, caseexpr->args)
			{
				CaseWhen   *casewhen = (CaseWhen *) lfirst(cell);

				Assert(exprType((Node *) casewhen->expr) == BOOLOID);
				Assert(exprType((Node *) casewhen->result) == caseexpr->casetype);
				appendStringInfo(&context->str, "EVAL(");
				codegen_expression_walker((Node *) casewhen->expr, context);
				appendStringInfo(&context->str, ") ? (");
				codegen_expression_walker((Node *) casewhen->result, context);
				appendStringInfo(&context->str, ") : (");
			}
			codegen_expression_walker((Node *) caseexpr->defresult, context);
			foreach (cell, caseexpr->args)
				appendStringInfo(&context->str, ")");
		}
	}
	else if (IsA(node, ScalarArrayOpExpr))
	{
		ScalarArrayOpExpr *opexpr = (ScalarArrayOpExpr *) node;
		Oid		func_oid = get_opcode(opexpr->opno);
		Node   *expr;

		dfunc = pgstrom_devfunc_lookup(func_oid,
									   get_func_rettype(func_oid),
									   opexpr->args,
									   opexpr->inputcollid);
		if (!dfunc)
			elog(ERROR, "codegen: failed to lookup device function: %s",
				 format_procedure(get_opcode(opexpr->opno)));
		pgstrom_devfunc_track(context, dfunc);
		Assert(dfunc->func_rettype->type_oid == BOOLOID &&
			   list_length(dfunc->func_args) == 2);

		appendStringInfo(&context->str, "PG_SCALAR_ARRAY_OP(kcxt, pgfn_%s, ",
						 dfunc->func_devname);
		expr = linitial(opexpr->args);
		codegen_expression_walker(expr, context);
		appendStringInfo(&context->str, ", ");
		expr = lsecond(opexpr->args);
		codegen_expression_walker(expr, context);
		/* type of array element */
		dtype = lsecond(dfunc->func_args);
		appendStringInfo(&context->str, ", %s, %d, %d)",
						 opexpr->useOr ? "true" : "false",
						 dtype->type_length,
						 dtype->type_align);
		context->extra_flags |= DEVKERNEL_NEEDS_MATRIX;
	}
	else
		elog(ERROR, "Bug? unsupported expression: %s", nodeToString(node));
}

static void
codegen_function_expression(devfunc_info *dfunc, List *args,
							codegen_context *context)
{
	ListCell   *lc1;
	ListCell   *lc2;

	appendStringInfo(&context->str,
					 "pgfn_%s(kcxt",
					 dfunc->func_devname);
	forboth (lc1, dfunc->func_args,
			 lc2, args)
	{
		devtype_info *dtype = lfirst(lc1);
		Node   *expr = lfirst(lc2);

		appendStringInfo(&context->str, ", ");
		if (dtype->type_oid == exprType(expr))
			codegen_expression_walker(expr, context);
		else
		{
			appendStringInfo(&context->str,
							 "to_%s(", dtype->type_name);
			codegen_expression_walker(expr, context);
			appendStringInfo(&context->str, ")");
		}
	}
	appendStringInfoChar(&context->str, ')');
}

char *
pgstrom_codegen_expression(Node *expr, codegen_context *context)
{
	codegen_context	walker_context;

	initStringInfo(&walker_context.str);
	walker_context.type_defs = list_copy(context->type_defs);
	walker_context.func_defs = list_copy(context->func_defs);
	walker_context.expr_defs = list_copy(context->expr_defs);
	walker_context.used_params = list_copy(context->used_params);
	walker_context.used_vars = list_copy(context->used_vars);
	walker_context.param_refs = bms_copy(context->param_refs);
	walker_context.var_label  = context->var_label;
	walker_context.kds_label  = context->kds_label;
	walker_context.kds_index_label = context->kds_index_label;
	walker_context.extra_flags = context->extra_flags;
	walker_context.pseudo_tlist = context->pseudo_tlist;

	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = (Node *)linitial((List *)expr);
		else
			expr = (Node *)make_andclause((List *)expr);
	}

	PG_TRY();
	{
		codegen_expression_walker(expr, &walker_context);
	}
	PG_CATCH();
	{
		errdetail("problematic expression: %s", nodeToString(expr));
		PG_RE_THROW();
	}
	PG_END_TRY();

	context->type_defs = walker_context.type_defs;
	context->func_defs = walker_context.func_defs;
	context->expr_defs = walker_context.expr_defs;
	context->used_params = walker_context.used_params;
	context->used_vars = walker_context.used_vars;
	context->param_refs = walker_context.param_refs;
	/* no need to write back xxx_label fields because read-only */
	context->extra_flags = walker_context.extra_flags;

	return walker_context.str.data;
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
		if (!bms_is_member(index, context->param_refs))
			goto lnext;

		if (IsA(lfirst(cell), Const))
		{
			Const  *con = lfirst(cell);

			dtype = pgstrom_devtype_lookup(con->consttype);
			if (!dtype)
				elog(ERROR, "failed to lookup device type: %u",
					 con->consttype);

			appendStringInfo(
				buf,
				"  pg_%s_t KPARAM_%u = pg_%s_param(kcxt,%d);\n",
				dtype->type_name, index, dtype->type_name, index);
		}
		else if (IsA(lfirst(cell), Param))
		{
			Param  *param = lfirst(cell);

			dtype = pgstrom_devtype_lookup(param->paramtype);
			if (!dtype)
				elog(ERROR, "failed to lookup device type: %u",
					 param->paramtype);

			appendStringInfo(
				buf,
				"  pg_%s_t KPARAM_%u = pg_%s_param(kcxt,%d);\n",
				dtype->type_name, index, dtype->type_name, index);
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(lfirst(cell)));
	lnext:
		index++;
	}
}

/*
 * pgstrom_device_expression
 *
 * It shows a quick decision whether the provided expression tree is
 * available to run on CUDA device, or not.
 */
bool
__pgstrom_device_expression(Expr *expr, const char *filename, int lineno)
{
	if (expr == NULL)
		return true;
	if (IsA(expr, List))
	{
		ListCell   *cell;

		foreach (cell, (List *) expr)
		{
			if (!__pgstrom_device_expression(lfirst(cell),filename,lineno))
				return false;
		}
		return true;
	}
	else if (IsA(expr, Const))
	{
		Const		   *con = (Const *) expr;

		/* supported types only */
		if (!pgstrom_devtype_lookup(con->consttype))
			goto unable_node;

		return true;
	}
	else if (IsA(expr, Param))
	{
		Param		   *param = (Param *) expr;

		/* only PARAM_EXTERN, right now */
		if (param->paramkind != PARAM_EXTERN)
			goto unable_node;

		/* supported types only */
		if (!pgstrom_devtype_lookup(param->paramtype))
			goto unable_node;

		return true;
	}
	else if (IsA(expr, Var))
	{
		Var			   *var = (Var *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(var->vartype);

		/*
		 * supported and scalar types only
		 *
		 * NOTE: We don't support array data type stored in relations,
		 * because it may have short varlena format (1-byte header), thus,
		 * we cannot guarantee alignment of packed datum in the array.
		 * Param or Const are individually untoasted on the parameter buffer,
		 * so its alignment is always 4-bytes, however, array datum in Var
		 * nodes have unpredictable alignment.
		 */
		if (!dtype || dtype->type_element)
			goto unable_node;

		return true;
	}
	else if (IsA(expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) expr;

		if (!pgstrom_devfunc_lookup(func->funcid,
									func->funcresulttype,
									func->args,
									func->inputcollid))
			goto unable_node;
		return __pgstrom_device_expression((Expr *) func->args,
										   filename, lineno);
	}
	else if (IsA(expr, OpExpr) || IsA(expr, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		if (!pgstrom_devfunc_lookup(get_opcode(op->opno),
									op->opresulttype,
									op->args,
									op->inputcollid))
			goto unable_node;
		return __pgstrom_device_expression((Expr *) op->args,
										   filename, lineno);
	}
	else if (IsA(expr, NullTest))
	{
		NullTest   *nulltest = (NullTest *) expr;

		if (nulltest->argisrow)
			goto unable_node;

		return __pgstrom_device_expression((Expr *) nulltest->arg,
										   filename, lineno);
	}
	else if (IsA(expr, BooleanTest))
	{
		BooleanTest	   *booltest = (BooleanTest *) expr;

		return __pgstrom_device_expression((Expr *) booltest->arg,
										   filename, lineno);
	}
	else if (IsA(expr, BoolExpr))
	{
		BoolExpr	   *boolexpr = (BoolExpr *) expr;

		Assert(boolexpr->boolop == AND_EXPR ||
			   boolexpr->boolop == OR_EXPR ||
			   boolexpr->boolop == NOT_EXPR);
		return __pgstrom_device_expression((Expr *) boolexpr->args,
										   filename, lineno);
	}
	else if (IsA(expr, CoalesceExpr))
	{
		CoalesceExpr   *coalesce = (CoalesceExpr *) expr;
		ListCell	   *cell;

		/* supported types only */
		if (!pgstrom_devtype_lookup(coalesce->coalescetype))
			goto unable_node;

		/* arguments also have to be same type (=device supported) */
		foreach (cell, coalesce->args)
		{
			Node   *expr = lfirst(cell);

			if (coalesce->coalescetype != exprType(expr))
				goto unable_node;
		}
		return __pgstrom_device_expression((Expr *) coalesce->args,
										   filename, lineno);
	}
	else if (IsA(expr, MinMaxExpr))
	{
		MinMaxExpr	   *minmax = (MinMaxExpr *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(minmax->minmaxtype);
		ListCell	   *cell;

		if (minmax->op != IS_GREATEST && minmax->op != IS_LEAST)
			return false;	/* unknown MinMax operation */

		/* supported types only */
		if (!dtype)
			goto unable_node;
		/* type compare function is required */
		if (!pgstrom_devfunc_lookup_type_compare(dtype, minmax->inputcollid))
			goto unable_node;

		/* arguments also have to be same type (=device supported) */
		foreach (cell, minmax->args)
		{
			Node   *expr = lfirst(cell);

			if (minmax->minmaxtype != exprType(expr))
				goto unable_node;
		}
		return __pgstrom_device_expression((Expr *) minmax->args,
										   filename, lineno);
	}
	else if (IsA(expr, RelabelType))
	{
		RelabelType	   *relabel = (RelabelType *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(relabel->resulttype);

		/* array->array relabel may be possible */
		if (!dtype)
			goto unable_node;

		return __pgstrom_device_expression((Expr *) relabel->arg,
										   filename, lineno);
	}
	else if (IsA(expr, CaseExpr))
	{
		CaseExpr   *caseexpr = (CaseExpr *) expr;
		ListCell   *cell;

		if (!pgstrom_devtype_lookup(caseexpr->casetype))
			goto unable_node;

		if (caseexpr->arg)
		{
			if (!__pgstrom_device_expression(caseexpr->arg,
											 filename, lineno))
				return false;
		}

		foreach (cell, caseexpr->args)
		{
			CaseWhen   *casewhen = lfirst(cell);

			Assert(IsA(casewhen, CaseWhen));
			if (exprType((Node *)casewhen->expr) != BOOLOID)
				goto unable_node;

			if (!__pgstrom_device_expression(casewhen->expr,
											 filename, lineno))
				return false;
			if (!__pgstrom_device_expression(casewhen->result,
											 filename, lineno))
				return false;
		}
		if (!__pgstrom_device_expression((Expr *)caseexpr->defresult,
										 filename, lineno))
			return false;
		return true;
	}
	else if (IsA(expr, CaseTestExpr))
	{
		CaseTestExpr   *casetest = (CaseTestExpr *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(casetest->typeId);

		if (!dtype)
			goto unable_node;

		return true;
	}
	else if (IsA(expr, ScalarArrayOpExpr))
	{
		ScalarArrayOpExpr  *opexpr = (ScalarArrayOpExpr *) expr;
		devtype_info	   *dtype;
		Oid					func_oid = get_opcode(opexpr->opno);
		
		if (!pgstrom_devfunc_lookup(func_oid,
									get_func_rettype(func_oid),
									opexpr->args,
									opexpr->inputcollid))
			goto unable_node;

		/* sanity checks */
		if (list_length(opexpr->args) != 2)
			goto unable_node;

		/* 1st argument must be scalar */
		dtype = pgstrom_devtype_lookup(exprType(linitial(opexpr->args)));
		if (!dtype || dtype->type_element)
			goto unable_node;

		/* 2nd argument must be array */
		dtype = pgstrom_devtype_lookup(exprType(lsecond(opexpr->args)));
		if (!dtype || dtype->type_array)
			goto unable_node;

		if (!__pgstrom_device_expression((Expr *) opexpr->args,
										 filename, lineno))
			return false;

		return true;
	}
unable_node:
	elog(DEBUG2, "Unable to run on device(%s:%d): %s",
		 basename(filename), lineno, nodeToString(expr));
	return false;
}

static void
codegen_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	MemoryContextReset(devinfo_memcxt);
	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));
	devtype_info_is_built = false;
}

static void
guc_assign_cache_invalidator(bool newval, void *extra)
{
	codegen_cache_invalidator(0, 0, 0);
}

void
pgstrom_init_codegen_context(codegen_context *context)
{
	memset(context, 0, sizeof(codegen_context));

	context->var_label = "KVAR";
	context->kds_label = "kds";
	context->kds_index_label = "kds_index";
}

void
pgstrom_init_codegen(void)
{
	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));
	devtype_info_is_built = false;

	/* create a memory context */
	devinfo_memcxt = AllocSetContextCreate(CacheMemoryContext,
										   "device type/func info cache",
										   ALLOCSET_DEFAULT_SIZES);
	CacheRegisterSyscacheCallback(PROCOID, codegen_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(TYPEOID, codegen_cache_invalidator, 0);

	/* pg_strom.enable_numeric_type */
    DefineCustomBoolVariable("pg_strom.enable_numeric_type",
							 "Turn on/off device numeric type support",
							 NULL,
							 &pgstrom_enable_numeric_type,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL,
							 guc_assign_cache_invalidator,
							 NULL);
}
