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

static pg_crc32 generic_devtype_hashfunc(devtype_info *dtype,
										 pg_crc32 hash,
										 Datum datum, bool isnull);
static pg_crc32 pg_numeric_devtype_hashfunc(devtype_info *dtype,
											pg_crc32 hash,
											Datum datum, bool isnull);
static pg_crc32 pg_bpchar_devtype_hashfunc(devtype_info *dtype,
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
				 "SHRT_MAX", "SHRT_MIN", "0",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("int4",   "INT4OID",   "cl_int",
				 "INT_MAX", "INT_MIN", "0",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("int8",   "INT8OID",   "cl_long",
				 "LONG_MAX", "LONG_MIN", "0",
				 0, 0, generic_devtype_hashfunc),
	/* XXX - float2 is not a built-in data type */
	DEVTYPE_DECL("float2", "FLOAT2OID", "cl_half",
				 "__half_as_short(HALF_MAX)",
				 "__half_as_short(-HALF_MAX)",
				 "__half_as_short(0.0)",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("float4", "FLOAT4OID", "cl_float",
				 "__float_as_int(FLT_MAX)",
				 "__float_as_int(-FLT_MAX)",
				 "__float_as_int(0.0)",
				 0, 0, generic_devtype_hashfunc),
	DEVTYPE_DECL("float8", "FLOAT8OID", "cl_double",
				 "__double_as_longlong(DBL_MAX)",
				 "__double_as_longlong(-DBL_MAX)",
				 "__double_as_longlong(0.0)",
				 0, 0, generic_devtype_hashfunc),
	/*
	 * Misc data types
	 */
	DEVTYPE_DECL("money",  "CASHOID",   "cl_long",
				 "LONG_MAX", "LONG_MIN", "0",
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
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("cidr",   "CIDROID",   "inet_struct",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISC, sizeof(inet),
				 generic_devtype_hashfunc),
	/*
	 * Date and time datatypes
	 */
	DEVTYPE_DECL("date", "DATEOID", "DateADT",
				 "INT_MAX", "INT_MIN", "0",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("time", "TIMEOID", "TimeADT",
				 "LONG_MAX", "LONG_MIN", "0",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timetz", "TIMETZOID", "TimeTzADT",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_TIMELIB, sizeof(TimeTzADT),
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timestamp", "TIMESTAMPOID","Timestamp",
				 "LONG_MAX", "LONG_MIN", "0",
				 DEVKERNEL_NEEDS_TIMELIB, 0,
				 generic_devtype_hashfunc),
	DEVTYPE_DECL("timestamptz", "TIMESTAMPTZOID", "TimestampTz",
				 "LONG_MAX", "LONG_MIN", "0",
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
	devtype_info_slot[hindex] = lappend(devtype_info_slot[hindex], entry);

	return entry;
}

static void
build_devtype_info(void)
{
	MemoryContext oldcxt;
	int		i;

	Assert(!devtype_info_is_built);

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
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
	MemoryContextSwitchTo(oldcxt);

	devtype_info_is_built = true;
}

devtype_info *
pgstrom_devtype_lookup(Oid type_oid)
{
	ListCell	   *cell;
	int				hindex;

	if (!devtype_info_is_built)
		build_devtype_info();

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
pg_range_devtype_hashfunc(devtype_info *dtype,
						  pg_crc32 hash,
						  Datum datum, bool isnull)
{
	if (!isnull)
	{
		RangeType  *r = DatumGetRangeType(datum);
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
 * 'a' : this function needs an alias, instead of SQL function name
 * 'c' : this function is locale aware, thus, available only if simple
 *       collation configuration (none, and C-locale).
 * 'm' : this function needs cuda_mathlib.h
 * 'n' : this function needs cuda_numeric.h
 * 's' : this function needs cuda_textlib.h
 * 't' : this function needs cuda_timelib.h
 * 'y' : this function needs cuda_misc.h
 * 'r' : this function needs cuda_rangetype.h
 *
 * class character:
 * 'c' : this function is type cast that takes an argument
 * 'r' : this function is right operator that takes an argument
 * 'l' : this function is left operator that takes an argument
 * 'b' : this function is both operator that takes two arguments
 *       type cast shall be added, if type length mismatch
 *     ==> extra is the operator character on CUDA
 * 'B' : almost equivalent to 'b', but type cast will not happen.
 * 'f' : this function utilizes built-in functions
 *     ==> extra is the built-in function name
 * 'F' : this function is externally declared.
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
	{ "bool", 1, {INT4OID},   "m/F:int4_bool" },

	{ "int2", 1, {INT4OID},   "a/c:" },
	{ "int2", 1, {INT8OID},   "a/c:" },
	{ "int2", 1, {FLOAT4OID}, "a/c:" },
	{ "int2", 1, {FLOAT8OID}, "a/c:" },

	{ "int4", 1, {BOOLOID},   "a/c:" },
	{ "int4", 1, {INT2OID},   "a/c:" },
	{ "int4", 1, {INT8OID},   "a/c:" },
	{ "int4", 1, {FLOAT4OID}, "a/c:" },
	{ "int4", 1, {FLOAT8OID}, "a/c:" },

	{ "int8", 1, {INT2OID},   "a/c:" },
	{ "int8", 1, {INT4OID},   "a/c:" },
	{ "int8", 1, {FLOAT4OID}, "a/c:" },
	{ "int8", 1, {FLOAT8OID}, "a/c:" },

	{ "float4", 1, {INT2OID},   "a/c:" },
	{ "float4", 1, {INT4OID},   "a/c:" },
	{ "float4", 1, {INT8OID},   "a/c:" },
	{ "float4", 1, {FLOAT8OID}, "a/c:" },

	{ "float8", 1, {INT2OID},   "a/c:" },
	{ "float8", 1, {INT4OID},   "a/c:" },
	{ "float8", 1, {INT8OID},   "a/c:" },
	{ "float8", 1, {FLOAT4OID}, "a/c:" },

	/* '+' : add operators */
	{ "int2pl",  2, {INT2OID, INT2OID}, "m/F:int2pl" },
	{ "int24pl", 2, {INT2OID, INT4OID}, "m/F:int24pl" },
	{ "int28pl", 2, {INT2OID, INT8OID}, "m/F:int28pl" },
	{ "int42pl", 2, {INT4OID, INT2OID}, "m/F:int42pl" },
	{ "int4pl",  2, {INT4OID, INT4OID}, "m/F:int4pl" },
	{ "int48pl", 2, {INT4OID, INT8OID}, "m/F:int48pl" },
	{ "int82pl", 2, {INT8OID, INT2OID}, "m/F:int82pl" },
	{ "int84pl", 2, {INT8OID, INT4OID}, "m/F:int84pl" },
	{ "int8pl",  2, {INT8OID, INT8OID}, "m/F:int8pl" },
	{ "float4pl",  2, {FLOAT4OID, FLOAT4OID}, "m/F:float4pl" },
	{ "float48pl", 2, {FLOAT4OID, FLOAT8OID}, "m/F:float48pl" },
	{ "float84pl", 2, {FLOAT8OID, FLOAT4OID}, "m/F:float84pl" },
	{ "float8pl",  2, {FLOAT8OID, FLOAT8OID}, "m/F:float8pl" },

	/* '-' : subtract operators */
	{ "int2mi",  2, {INT2OID, INT2OID}, "m/F:int2mi" },
	{ "int24mi", 2, {INT2OID, INT4OID}, "m/F:int24mi" },
	{ "int28mi", 2, {INT2OID, INT8OID}, "m/F:int28mi" },
	{ "int42mi", 2, {INT4OID, INT2OID}, "m/F:int42mi" },
	{ "int4mi",  2, {INT4OID, INT4OID}, "m/F:int4mi" },
	{ "int48mi", 2, {INT4OID, INT8OID}, "m/F:int48mi" },
	{ "int82mi", 2, {INT8OID, INT2OID}, "m/F:int82mi" },
	{ "int84mi", 2, {INT8OID, INT4OID}, "m/F:int84mi" },
	{ "int8mi",  2, {INT8OID, INT8OID}, "m/F:int8mi" },
	{ "float4mi",  2, {FLOAT4OID, FLOAT4OID}, "m/F:float4mi" },
	{ "float48mi", 2, {FLOAT4OID, FLOAT8OID}, "m/F:float48mi" },
	{ "float84mi", 2, {FLOAT8OID, FLOAT4OID}, "m/F:float84mi" },
	{ "float8mi",  2, {FLOAT8OID, FLOAT8OID}, "m/F:float8mi" },

	/* '*' : mutiply operators */
	{ "int2mul",  2, {INT2OID, INT2OID}, "m/F:int2mul" },
	{ "int24mul", 2, {INT2OID, INT4OID}, "m/F:int24mul" },
	{ "int28mul", 2, {INT2OID, INT8OID}, "m/F:int28mul" },
	{ "int42mul", 2, {INT4OID, INT2OID}, "m/F:int42mul" },
	{ "int4mul",  2, {INT4OID, INT4OID}, "m/F:int4mul" },
	{ "int48mul", 2, {INT4OID, INT8OID}, "m/F:int48mul" },
	{ "int82mul", 2, {INT8OID, INT2OID}, "m/F:int82mul" },
	{ "int84mul", 2, {INT8OID, INT4OID}, "m/F:int84mul" },
	{ "int8mul",  2, {INT8OID, INT8OID}, "m/F:int8mul" },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, "m/F:float4mul" },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, "m/F:float48mul" },
	{ "float84mul", 2, {FLOAT8OID, FLOAT4OID}, "m/F:float84mul" },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, "m/F:float8mul" },

	/* '/' : divide operators */
	{ "int2div",  2, {INT2OID, INT2OID}, "m/F:int2div" },
	{ "int24div", 2, {INT2OID, INT4OID}, "m/F:int24div" },
	{ "int28div", 2, {INT2OID, INT8OID}, "m/F:int28div" },
	{ "int42div", 2, {INT4OID, INT2OID}, "m/F:int42div" },
	{ "int4div",  2, {INT4OID, INT4OID}, "m/F:int4div" },
	{ "int48div", 2, {INT4OID, INT8OID}, "m/F:int48div" },
	{ "int82div", 2, {INT8OID, INT2OID}, "m/F:int82div" },
	{ "int84div", 2, {INT8OID, INT4OID}, "m/F:int84div" },
	{ "int8div",  2, {INT8OID, INT8OID}, "m/F:int8div" },
	{ "float4div",  2, {FLOAT4OID, FLOAT4OID}, "m/F:float4div" },
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, "m/F:float48div" },
	{ "float84div", 2, {FLOAT8OID, FLOAT4OID}, "m/F:float84div" },
	{ "float8div",  2, {FLOAT8OID, FLOAT8OID}, "m/F:float8div" },

	/* '%' : reminder operators */
	{ "int2mod", 2, {INT2OID, INT2OID}, "m/F:int2mod" },
	{ "int4mod", 2, {INT4OID, INT4OID}, "m/F:int4mod" },
	{ "int8mod", 2, {INT8OID, INT8OID}, "m/F:int8mod" },

	/* '+' : unary plus operators */
	{ "int2up", 1, {INT2OID}, "l:+" },
	{ "int4up", 1, {INT4OID}, "l:+" },
	{ "int8up", 1, {INT8OID}, "l:+" },
	{ "float4up", 1, {FLOAT4OID}, "l:+" },
	{ "float8up", 1, {FLOAT8OID}, "l:+" },

	/* '-' : unary minus operators */
	{ "int2um", 1, {INT2OID}, "l:-" },
	{ "int4um", 1, {INT4OID}, "l:-" },
	{ "int8um", 1, {INT8OID}, "l:-" },
	{ "float4um", 1, {FLOAT4OID}, "l:-" },
	{ "float8um", 1, {FLOAT8OID}, "l:-" },

	/* '@' : absolute value operators */
	{ "int2abs", 1, {INT2OID}, "f:abs" },
	{ "int4abs", 1, {INT4OID}, "f:abs" },
	{ "int8abs", 1, {INT8OID}, "f:abs" },
	{ "float4abs", 1, {FLOAT4OID}, "f:fabs" },
	{ "float8abs", 1, {FLOAT8OID}, "f:fabs" },

	/* '=' : equal operators */
	{ "int2eq",  2, {INT2OID, INT2OID}, "b:==" },
	{ "int24eq", 2, {INT2OID, INT4OID}, "b:==" },
	{ "int28eq", 2, {INT2OID, INT8OID}, "b:==" },
	{ "int42eq", 2, {INT4OID, INT2OID}, "b:==" },
	{ "int4eq",  2, {INT4OID, INT4OID}, "b:==" },
	{ "int48eq", 2, {INT4OID, INT8OID}, "b:==" },
	{ "int82eq", 2, {INT8OID, INT2OID}, "b:==" },
	{ "int84eq", 2, {INT8OID, INT4OID}, "b:==" },
	{ "int8eq",  2, {INT8OID, INT8OID}, "b:==" },
	{ "float4eq",  2, {FLOAT4OID, FLOAT4OID}, "b:==" },
	{ "float48eq", 2, {FLOAT4OID, FLOAT8OID}, "b:==" },
	{ "float84eq", 2, {FLOAT8OID, FLOAT4OID}, "b:==" },
	{ "float8eq",  2, {FLOAT8OID, FLOAT8OID}, "b:==" },

	/* '<>' : not equal operators */
	{ "int2ne",  2, {INT2OID, INT2OID}, "b:!=" },
	{ "int24ne", 2, {INT2OID, INT4OID}, "b:!=" },
	{ "int28ne", 2, {INT2OID, INT8OID}, "b:!=" },
	{ "int42ne", 2, {INT4OID, INT2OID}, "b:!=" },
	{ "int4ne",  2, {INT4OID, INT4OID}, "b:!=" },
	{ "int48ne", 2, {INT4OID, INT8OID}, "b:!=" },
	{ "int82ne", 2, {INT8OID, INT2OID}, "b:!=" },
	{ "int84ne", 2, {INT8OID, INT4OID}, "b:!=" },
	{ "int8ne",  2, {INT8OID, INT8OID}, "b:!=" },
	{ "float4ne",  2, {FLOAT4OID, FLOAT4OID}, "b:!=" },
	{ "float48ne", 2, {FLOAT4OID, FLOAT8OID}, "b:!=" },
	{ "float84ne", 2, {FLOAT8OID, FLOAT4OID}, "b:!=" },
	{ "float8ne",  2, {FLOAT8OID, FLOAT8OID}, "b:!=" },

	/* '>' : equal operators */
	{ "int2gt",  2, {INT2OID, INT2OID}, "b:>" },
	{ "int24gt", 2, {INT2OID, INT4OID}, "b:>" },
	{ "int28gt", 2, {INT2OID, INT8OID}, "b:>" },
	{ "int42gt", 2, {INT4OID, INT2OID}, "b:>" },
	{ "int4gt",  2, {INT4OID, INT4OID}, "b:>" },
	{ "int48gt", 2, {INT4OID, INT8OID}, "b:>" },
	{ "int82gt", 2, {INT8OID, INT2OID}, "b:>" },
	{ "int84gt", 2, {INT8OID, INT4OID}, "b:>" },
	{ "int8gt",  2, {INT8OID, INT8OID}, "b:>" },
	{ "float4gt",  2, {FLOAT4OID, FLOAT4OID}, "b:>" },
	{ "float48gt", 2, {FLOAT4OID, FLOAT8OID}, "b:>" },
	{ "float84gt", 2, {FLOAT8OID, FLOAT4OID}, "b:>" },
	{ "float8gt",  2, {FLOAT8OID, FLOAT8OID}, "b:>" },

	/* '<' : equal operators */
	{ "int2lt",  2, {INT2OID, INT2OID}, "b:<" },
	{ "int24lt", 2, {INT2OID, INT4OID}, "b:<" },
	{ "int28lt", 2, {INT2OID, INT8OID}, "b:<" },
	{ "int42lt", 2, {INT4OID, INT2OID}, "b:<" },
	{ "int4lt",  2, {INT4OID, INT4OID}, "b:<" },
	{ "int48lt", 2, {INT4OID, INT8OID}, "b:<" },
	{ "int82lt", 2, {INT8OID, INT2OID}, "b:<" },
	{ "int84lt", 2, {INT8OID, INT4OID}, "b:<" },
	{ "int8lt",  2, {INT8OID, INT8OID}, "b:<" },
	{ "float4lt",  2, {FLOAT4OID, FLOAT4OID}, "b:<" },
	{ "float48lt", 2, {FLOAT4OID, FLOAT8OID}, "b:<" },
	{ "float84lt", 2, {FLOAT8OID, FLOAT4OID}, "b:<" },
	{ "float8lt",  2, {FLOAT8OID, FLOAT8OID}, "b:<" },

	/* '>=' : relational greater-than or equal-to */
	{ "int2ge",  2, {INT2OID, INT2OID}, "b:>=" },
	{ "int24ge", 2, {INT2OID, INT4OID}, "b:>=" },
	{ "int28ge", 2, {INT2OID, INT8OID}, "b:>=" },
	{ "int42ge", 2, {INT4OID, INT2OID}, "b:>=" },
	{ "int4ge",  2, {INT4OID, INT4OID}, "b:>=" },
	{ "int48ge", 2, {INT4OID, INT8OID}, "b:>=" },
	{ "int82ge", 2, {INT8OID, INT2OID}, "b:>=" },
	{ "int84ge", 2, {INT8OID, INT4OID}, "b:>=" },
	{ "int8ge",  2, {INT8OID, INT8OID}, "b:>=" },
	{ "float4ge",  2, {FLOAT4OID, FLOAT4OID}, "b:>=" },
	{ "float48ge", 2, {FLOAT4OID, FLOAT8OID}, "b:>=" },
	{ "float84ge", 2, {FLOAT8OID, FLOAT4OID}, "b:>=" },
	{ "float8ge",  2, {FLOAT8OID, FLOAT8OID}, "b:>=" },

	/* '<=' : relational greater-than or equal-to */
	{ "int2le",  2, {INT2OID, INT2OID}, "b:<=" },
	{ "int24le", 2, {INT2OID, INT4OID}, "b:<=" },
	{ "int28le", 2, {INT2OID, INT8OID}, "b:<=" },
	{ "int42le", 2, {INT4OID, INT2OID}, "b:<=" },
	{ "int4le",  2, {INT4OID, INT4OID}, "b:<=" },
	{ "int48le", 2, {INT4OID, INT8OID}, "b:<=" },
	{ "int82le", 2, {INT8OID, INT2OID}, "b:<=" },
	{ "int84le", 2, {INT8OID, INT4OID}, "b:<=" },
	{ "int8le",  2, {INT8OID, INT8OID}, "b:<=" },
	{ "float4le",  2, {FLOAT4OID, FLOAT4OID}, "b:<=" },
	{ "float48le", 2, {FLOAT4OID, FLOAT8OID}, "b:<=" },
	{ "float84le", 2, {FLOAT8OID, FLOAT4OID}, "b:<=" },
	{ "float8le",  2, {FLOAT8OID, FLOAT8OID}, "b:<=" },

	/* '&' : bitwise and */
	{ "int2and", 2, {INT2OID, INT2OID}, "b:&" },
	{ "int4and", 2, {INT4OID, INT4OID}, "b:&" },
	{ "int8and", 2, {INT8OID, INT8OID}, "b:&" },

	/* '|'  : bitwise or */
	{ "int2or", 2, {INT2OID, INT2OID}, "b:|" },
	{ "int4or", 2, {INT4OID, INT4OID}, "b:|" },
	{ "int8or", 2, {INT8OID, INT8OID}, "b:|" },

	/* '#'  : bitwise xor */
	{ "int2xor", 2, {INT2OID, INT2OID}, "b:^" },
	{ "int4xor", 2, {INT4OID, INT4OID}, "b:^" },
	{ "int8xor", 2, {INT8OID, INT8OID}, "b:^" },

	/* '~'  : bitwise not operators */
	{ "int2not", 1, {INT2OID}, "b:~" },
	{ "int4not", 1, {INT4OID}, "b:~" },
	{ "int8not", 1, {INT8OID}, "b:~" },

	/* '>>' : right shift */
	{ "int2shr", 2, {INT2OID, INT4OID}, "B:>>" },
	{ "int4shr", 2, {INT4OID, INT4OID}, "B:>>" },
	{ "int8shr", 2, {INT8OID, INT4OID}, "B:>>" },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID, INT4OID}, "B:<<" },
	{ "int4shl", 2, {INT4OID, INT4OID}, "B:<<" },
	{ "int8shl", 2, {INT8OID, INT4OID}, "B:<<" },

	/* comparison functions */
	{ "btboolcmp",  2, {BOOLOID, BOOLOID}, "f:devfunc_int_comp" },
	{ "btint2cmp",  2, {INT2OID, INT2OID}, "f:devfunc_int_comp" },
	{ "btint24cmp", 2, {INT2OID, INT4OID}, "f:devfunc_int_comp" },
	{ "btint28cmp", 2, {INT2OID, INT8OID}, "f:devfunc_int_comp" },
	{ "btint42cmp", 2, {INT4OID, INT2OID}, "f:devfunc_int_comp" },
	{ "btint4cmp",  2, {INT4OID, INT4OID}, "f:devfunc_int_comp" },
	{ "btint48cmp", 2, {INT4OID, INT8OID}, "f:devfunc_int_comp" },
	{ "btint82cmp", 2, {INT8OID, INT2OID}, "f:devfunc_int_comp" },
	{ "btint84cmp", 2, {INT8OID, INT4OID}, "f:devfunc_int_comp" },
	{ "btint8cmp",  2, {INT8OID, INT8OID}, "f:devfunc_int_comp" },
	{ "btfloat4cmp",  2, {FLOAT4OID, FLOAT4OID}, "f:devfunc_float_comp" },
	{ "btfloat48cmp", 2, {FLOAT4OID, FLOAT8OID}, "f:devfunc_float_comp" },
	{ "btfloat84cmp", 2, {FLOAT8OID, FLOAT4OID}, "f:devfunc_float_comp" },
	{ "btfloat8cmp",  2, {FLOAT8OID, FLOAT8OID}, "f:devfunc_float_comp" },
	/* currency cast */
	{ "money",			1, {NUMERICOID},			"y/F:numeric_cash" },
	{ "money",			1, {INT4OID},				"y/F:int4_cash" },
	{ "money",			1, {INT8OID},				"y/F:int8_cash" },
	/* currency operators */
	{ "cash_pl",		2, {CASHOID, CASHOID},		"y/F:cash_pl" },
	{ "cash_mi",		2, {CASHOID, CASHOID},		"y/F:cash_mi" },
	{ "cash_div_cash",	2, {CASHOID, CASHOID},		"y/F:cash_div_cash" },
	{ "cash_mul_int2",	2, {CASHOID, INT2OID},		"y/F:cash_mul_int2" },
	{ "cash_mul_int4",	2, {CASHOID, INT4OID},		"y/F:cash_mul_int4" },
	{ "cash_mul_flt4",	2, {CASHOID, FLOAT4OID},	"y/F:cash_mul_flt4" },
	{ "cash_mul_flt8",	2, {CASHOID, FLOAT8OID},	"y/F:cash_mul_flt8" },
	{ "cash_div_int2",	2, {CASHOID, INT2OID},		"y/F:cash_div_int2" },
	{ "cash_div_int4",	2, {CASHOID, INT4OID},		"y/F:cash_div_int4" },
	{ "cash_div_flt4",	2, {CASHOID, FLOAT4OID},	"y/F:cash_div_flt4" },
	{ "cash_div_flt8",	2, {CASHOID, FLOAT8OID},	"y/F:cash_div_flt8" },
	{ "int2_mul_cash",	2, {INT2OID, CASHOID},		"y/F:int2_mul_cash" },
	{ "int4_mul_cash",	2, {INT4OID, CASHOID},		"y/F:int4_mul_cash" },
	{ "flt4_mul_cash",	2, {FLOAT4OID, CASHOID},	"y/F:flt4_mul_cash" },
	{ "flt8_mul_cash",	2, {FLOAT8OID, CASHOID},	"y/F:flt8_mul_cash" },
	/* currency comparison */
	{ "cash_cmp",		2, {CASHOID, CASHOID},		"y/F:cash_cmp" },
	{ "cash_eq",		2, {CASHOID, CASHOID},		"y/F:cash_eq" },
	{ "cash_ne",		2, {CASHOID, CASHOID},		"y/F:cash_ne" },
	{ "cash_lt",		2, {CASHOID, CASHOID},		"y/F:cash_lt" },
	{ "cash_le",		2, {CASHOID, CASHOID},		"y/F:cash_le" },
	{ "cash_gt",		2, {CASHOID, CASHOID},		"y/F:cash_gt" },
	{ "cash_ge",		2, {CASHOID, CASHOID},		"y/F:cash_ge" },
	/* uuid comparison */
	{ "uuid_cmp",		2, {UUIDOID, UUIDOID},		"y/F:uuid_cmp" },
	{ "uuid_eq",		2, {UUIDOID, UUIDOID},		"y/F:uuid_eq" },
	{ "uuid_ne",		2, {UUIDOID, UUIDOID},		"y/F:uuid_ne" },
	{ "uuid_lt",		2, {UUIDOID, UUIDOID},		"y/F:uuid_lt" },
	{ "uuid_le",		2, {UUIDOID, UUIDOID},		"y/F:uuid_le" },
	{ "uuid_gt",		2, {UUIDOID, UUIDOID},		"y/F:uuid_gt" },
	{ "uuid_ge",		2, {UUIDOID, UUIDOID},		"y/F:uuid_ge" },
	/* macaddr comparison */
	{ "macaddr_cmp",    2, {MACADDROID,MACADDROID}, "y/F:macaddr_cmp" },
	{ "macaddr_eq",     2, {MACADDROID,MACADDROID}, "y/F:macaddr_eq" },
	{ "macaddr_ne",     2, {MACADDROID,MACADDROID}, "y/F:macaddr_ne" },
	{ "macaddr_lt",     2, {MACADDROID,MACADDROID}, "y/F:macaddr_lt" },
	{ "macaddr_le",     2, {MACADDROID,MACADDROID}, "y/F:macaddr_le" },
	{ "macaddr_gt",     2, {MACADDROID,MACADDROID}, "y/F:macaddr_gt" },
	{ "macaddr_ge",     2, {MACADDROID,MACADDROID}, "y/F:macaddr_ge" },
	/* inet comparison */
	{ "network_cmp",    2, {INETOID,INETOID},       "y/F:network_cmp" },
	{ "network_eq",     2, {INETOID,INETOID},       "y/F:network_eq" },
	{ "network_ne",     2, {INETOID,INETOID},       "y/F:network_ne" },
	{ "network_lt",     2, {INETOID,INETOID},       "y/F:network_lt" },
	{ "network_le",     2, {INETOID,INETOID},       "y/F:network_le" },
	{ "network_gt",     2, {INETOID,INETOID},       "y/F:network_gt" },
	{ "network_ge",     2, {INETOID,INETOID},       "y/F:network_ge" },
	{ "network_larger", 2, {INETOID,INETOID},       "y/F:network_larger" },
	{ "network_smaller", 2, {INETOID,INETOID},      "y/F:network_smaller" },
	{ "network_sub",    2, {INETOID,INETOID},       "y/F:network_sub" },
	{ "network_subeq",  2, {INETOID,INETOID},       "y/F:network_subeq" },
	{ "network_sup",    2, {INETOID,INETOID},       "y/F:network_sup" },
	{ "network_supeq",  2, {INETOID,INETOID},       "y/F:network_supeq" },
	{ "network_overlap",2, {INETOID,INETOID},       "y/F:network_overlap" },

	/*
     * Mathmatical functions
     */
	{ "abs", 1, {INT2OID}, "a/f:abs" },
	{ "abs", 1, {INT4OID}, "a/f:abs" },
	{ "abs", 1, {INT8OID}, "a/f:abs" },
	{ "abs", 1, {FLOAT4OID}, "a/f:fabs" },
	{ "abs", 1, {FLOAT8OID}, "a/f:fabs" },
	{ "cbrt",  1, {FLOAT4OID}, "f:cbrt" },
	{ "dcbrt", 1, {FLOAT8OID}, "f:cbrt" },
	{ "ceil", 1, {FLOAT8OID}, "f:ceil" },
	{ "ceiling", 1, {FLOAT8OID}, "f:ceil" },
	{ "exp", 1, {FLOAT8OID}, "f:exp" },
	{ "dexp", 1, {FLOAT8OID}, "f:exp" },
	{ "floor", 1, {FLOAT8OID}, "f:dfloor" },
	{ "ln", 1, {FLOAT8OID}, "f:log" },
	{ "dlog1", 1, {FLOAT8OID}, "f:log" },
	{ "log", 1, {FLOAT8OID}, "f:log10" },
	{ "dlog10", 1, {FLOAT8OID}, "f:log10" },
	{ "pi", 0, {}, "m/F:dpi" },
	{ "power", 2, {FLOAT8OID, FLOAT8OID}, "m/F:dpow" },
	{ "pow", 2, {FLOAT8OID, FLOAT8OID}, "m/F:dpow" },
	{ "dpow", 2, {FLOAT8OID, FLOAT8OID}, "m/F:dpow" },
	{ "round", 1, {FLOAT8OID}, "f:round" },
	{ "dround", 1, {FLOAT8OID}, "f:round" },
	{ "sign", 1, {FLOAT8OID}, "f:sign" },
	{ "sqrt", 1, {FLOAT8OID}, "m/F:dsqrt" },
	{ "dsqrt", 1, {FLOAT8OID}, "m/F:dsqrt" },
	{ "trunc", 1, {FLOAT8OID}, "f:trunc" },
	{ "dtrunc", 1, {FLOAT8OID}, "f:trunc" },

	/*
     * Trigonometric function
     */
	{ "degrees", 1, {FLOAT8OID}, "f:degrees" },
	{ "radians", 1, {FLOAT8OID}, "f:radians" },
	{ "acos",    1, {FLOAT8OID}, "f:acos" },
	{ "asin",    1, {FLOAT8OID}, "f:asin" },
	{ "atan",    1, {FLOAT8OID}, "f:atan" },
	{ "atan2",   2, {FLOAT8OID, FLOAT8OID}, "f:atan2" },
	{ "cos",     1, {FLOAT8OID}, "f:cos" },
	{ "cot",     1, {FLOAT8OID}, "m/F:dcot" },
	{ "sin",     1, {FLOAT8OID}, "f:sin" },
	{ "tan",     1, {FLOAT8OID}, "f:tan" },

	/*
	 * Numeric functions
	 * ------------------------- */
	/* Numeric type cast functions */
	{ "int2",    1, {NUMERICOID}, "n/F:numeric_int2" },
	{ "int4",    1, {NUMERICOID}, "n/F:numeric_int4" },
	{ "int8",    1, {NUMERICOID}, "n/F:numeric_int8" },
	{ "float4",  1, {NUMERICOID}, "n/F:numeric_float4" },
	{ "float8",  1, {NUMERICOID}, "n/F:numeric_float8" },
	{ "numeric", 1, {INT2OID},    "n/F:int2_numeric" },
	{ "numeric", 1, {INT4OID},    "n/F:int4_numeric" },
	{ "numeric", 1, {INT8OID},    "n/F:int8_numeric" },
	{ "numeric", 1, {FLOAT4OID},  "n/F:float4_numeric" },
	{ "numeric", 1, {FLOAT8OID},  "n/F:float8_numeric" },
	/* Numeric operators */
	{ "numeric_add", 2, {NUMERICOID, NUMERICOID}, "n/F:numeric_add" },
	{ "numeric_sub", 2, {NUMERICOID, NUMERICOID}, "n/F:numeric_sub" },
	{ "numeric_mul", 2, {NUMERICOID, NUMERICOID}, "n/F:numeric_mul" },
	{ "numeric_uplus",  1, {NUMERICOID}, "n/F:numeric_uplus" },
	{ "numeric_uminus", 1, {NUMERICOID}, "n/F:numeric_uminus" },
	{ "numeric_abs",    1, {NUMERICOID}, "n/F:numeric_abs" },
	{ "abs",            1, {NUMERICOID}, "n/F:numeric_abs" },
	/* Numeric comparison */
	{ "numeric_eq", 2, {NUMERICOID, NUMERICOID},  "n/F:numeric_eq" },
	{ "numeric_ne", 2, {NUMERICOID, NUMERICOID},  "n/F:numeric_ne" },
	{ "numeric_lt", 2, {NUMERICOID, NUMERICOID},  "n/F:numeric_lt" },
	{ "numeric_le", 2, {NUMERICOID, NUMERICOID},  "n/F:numeric_le" },
	{ "numeric_gt", 2, {NUMERICOID, NUMERICOID},  "n/F:numeric_gt" },
	{ "numeric_ge", 2, {NUMERICOID, NUMERICOID},  "n/F:numeric_ge" },
	{ "numeric_cmp", 2, {NUMERICOID, NUMERICOID}, "n/F:numeric_cmp" },

	/*
	 * Date and time functions
	 * ------------------------------- */
	/* Type cast functions */
	{ "date", 1, {DATEOID}, "ta/c:" },
	{ "date", 1, {TIMESTAMPOID}, "t/F:timestamp_date" },
	{ "date", 1, {TIMESTAMPTZOID}, "t/F:timestamptz_date" },
	{ "time", 1, {TIMEOID}, "ta/c:" },
	{ "time", 1, {TIMETZOID}, "t/F:timetz_time" },
	{ "time", 1, {TIMESTAMPOID}, "t/F:timestamp_time" },
	{ "time", 1, {TIMESTAMPTZOID}, "t/F:timestamptz_time" },
	{ "timetz", 1, {TIMEOID}, "t/F:time_timetz" },
	{ "timetz", 1, {TIMESTAMPTZOID}, "t/F:timestamptz_timetz" },
#ifdef NOT_USED
	{ "timetz", 2, {TIMETZOID, INT4OID}, "t/F:timetz_scale" },
#endif
	{ "timestamp", 1, {DATEOID}, "t/F:date_timestamp" },
	{ "timestamp", 1, {TIMESTAMPOID}, "ta/c:" },
	{ "timestamp", 1, {TIMESTAMPTZOID}, "t/F:timestamptz_timestamp" },
	{ "timestamptz", 1, {DATEOID}, "t/F:date_timestamptz" },
	{ "timestamptz", 1, {TIMESTAMPOID}, "t/F:timestamp_timestamptz" },
	/* timedata operators */
	{ "date_pli", 2, {DATEOID, INT4OID}, "t/F:date_pli" },
	{ "date_mii", 2, {DATEOID, INT4OID}, "t/F:date_mii" },
	{ "date_mi", 2, {DATEOID, DATEOID}, "t/F:date_mi" },
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, "t/F:datetime_pl" },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, "t/F:integer_pl_date" },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, "t/F:timedate_pl" },
	/* time - time => interval */
	{ "time_mi_time", 2, {TIMEOID, TIMEOID}, "t/F:time_mi_time" },
	/* timestamp - timestamp => interval */
	{ "timestamp_mi", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/F:timestamp_mi" },
	/* timetz +/- interval => timetz */
	{ "timetz_pl_interval", 2, {TIMETZOID, INTERVALOID},
	  "t/F:timetz_pl_interval" },
	{ "timetz_mi_interval", 2, {TIMETZOID, INTERVALOID},
	  "t/F:timetz_mi_interval" },
	/* timestamptz +/- interval => timestamptz */
	{ "timestamptz_pl_interval", 2, {TIMESTAMPTZOID, INTERVALOID},
	  "t/F:timestamptz_pl_interval" },
	{ "timestamptz_mi_interval", 2, {TIMESTAMPTZOID, INTERVALOID},
	  "t/F:timestamptz_mi_interval" },
	/* interval operators */
	{ "interval_um", 1, {INTERVALOID}, "t/F:interval_um" },
	{ "interval_pl", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_pl" },
	{ "interval_mi", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_mi" },
	/* date + timetz => timestamptz */
	{ "datetimetz_pl", 2, {DATEOID, TIMETZOID}, "t/F:datetimetz_timestamptz" },
	{ "timestamptz", 2, {DATEOID, TIMETZOID}, "t/F:datetimetz_timestamptz" },
	/* comparison between date */
	{ "date_eq", 2, {DATEOID, DATEOID}, "t/b:==" },
	{ "date_ne", 2, {DATEOID, DATEOID}, "t/b:!=" },
	{ "date_lt", 2, {DATEOID, DATEOID}, "t/b:<"  },
	{ "date_le", 2, {DATEOID, DATEOID}, "t/b:<=" },
	{ "date_gt", 2, {DATEOID, DATEOID}, "t/b:>"  },
	{ "date_ge", 2, {DATEOID, DATEOID}, "t/b:>=" },
	{ "date_cmp", 2, {DATEOID, DATEOID}, "t/f:devfunc_int_comp" },
	/* comparison of date and timestamp */
	{ "date_eq_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/F:date_eq_timestamp" },
	{ "date_ne_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/F:date_ne_timestamp" },
	{ "date_lt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/F:date_lt_timestamp" },
	{ "date_le_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/F:date_le_timestamp" },
	{ "date_gt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/F:date_gt_timestamp" },
	{ "date_ge_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/F:date_ge_timestamp" },
	{ "date_cmp_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "t/F:date_cmp_timestamp" },
	/* comparison between time */
	{ "time_eq", 2, {TIMEOID, TIMEOID}, "t/b:==" },
	{ "time_ne", 2, {TIMEOID, TIMEOID}, "t/b:!=" },
	{ "time_lt", 2, {TIMEOID, TIMEOID}, "t/b:<"  },
	{ "time_le", 2, {TIMEOID, TIMEOID}, "t/b:<=" },
	{ "time_gt", 2, {TIMEOID, TIMEOID}, "t/b:>"  },
	{ "time_ge", 2, {TIMEOID, TIMEOID}, "t/b:>=" },
	{ "time_cmp", 2, {TIMEOID, TIMEOID}, "t/f:devfunc_int_comp" },
	/* comparison between timetz */
	{ "timetz_eq", 2, {TIMETZOID, TIMETZOID}, "t/F:timetz_eq" },
	{ "timetz_ne", 2, {TIMETZOID, TIMETZOID}, "t/F:timetz_ne" },
	{ "timetz_lt", 2, {TIMETZOID, TIMETZOID}, "t/F:timetz_lt" },
	{ "timetz_le", 2, {TIMETZOID, TIMETZOID}, "t/F:timetz_le" },
	{ "timetz_ge", 2, {TIMETZOID, TIMETZOID}, "t/F:timetz_ge" },
	{ "timetz_gt", 2, {TIMETZOID, TIMETZOID}, "t/F:timetz_gt" },
	{ "timetz_cmp", 2, {TIMETZOID, TIMETZOID}, "t/F:timetz_cmp" },
	/* comparison between timestamp */
	{ "timestamp_eq", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/b:==" },
	{ "timestamp_ne", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/b:!=" },
	{ "timestamp_lt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/b:<"  },
	{ "timestamp_le", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/b:<=" },
	{ "timestamp_gt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/b:>"  },
	{ "timestamp_ge", 2, {TIMESTAMPOID, TIMESTAMPOID}, "t/b:>=" },
	{ "timestamp_cmp", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  "t/f:devfunc_int_comp" },
	/* comparison of timestamp and date */
	{ "timestamp_eq_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/F:timestamp_eq_date" },
	{ "timestamp_ne_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/F:timestamp_ne_date" },
	{ "timestamp_lt_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/F:timestamp_lt_date" },
	{ "timestamp_le_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/F:timestamp_le_date" },
	{ "timestamp_gt_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/F:timestamp_gt_date" },
	{ "timestamp_ge_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/F:timestamp_ge_date" },
	{ "timestamp_cmp_date", 2, {TIMESTAMPOID, DATEOID},
	  "t/F:timestamp_cmp_date"},
	/* comparison between timestamptz */
	{ "timestamptz_eq", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, "t/b:==" },
	{ "timestamptz_ne", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, "t/b:!=" },
	{ "timestamptz_lt", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, "t/b:<" },
	{ "timestamptz_le", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, "t/b:<=" },
	{ "timestamptz_gt", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, "t/b:>" },
	{ "timestamptz_ge", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, "t/b:>=" },
	{ "timestamptz_cmp", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID}, 
	  "t/f:devfunc_int_comp" },

	/* comparison between date and timestamptz */
	{ "date_lt_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/F:date_lt_timestamptz" },
	{ "date_le_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/F:date_le_timestamptz" },
	{ "date_eq_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/F:date_eq_timestamptz" },
	{ "date_ge_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/F:date_ge_timestamptz" },
	{ "date_gt_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/F:date_gt_timestamptz" },
	{ "date_ne_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  "t/F:date_ne_timestamptz" },

	/* comparison between timestamptz and date */
	{ "timestamptz_lt_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/F:timestamptz_lt_date" },
	{ "timestamptz_le_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/F:timestamptz_le_date" },
	{ "timestamptz_eq_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/F:timestamptz_eq_date" },
	{ "timestamptz_ge_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/F:timestamptz_ge_date" },
	{ "timestamptz_gt_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/F:timestamptz_gt_date" },
	{ "timestamptz_ne_date", 2, {TIMESTAMPTZOID, DATEOID},
	  "t/F:timestamptz_ne_date" },

	/* comparison between timestamp and timestamptz  */
	{ "timestamp_lt_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/F:timestamp_lt_timestamptz" },
	{ "timestamp_le_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/F:timestamp_le_timestamptz" },
	{ "timestamp_eq_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/F:timestamp_eq_timestamptz" },
	{ "timestamp_ge_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/F:timestamp_ge_timestamptz" },
	{ "timestamp_gt_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/F:timestamp_gt_timestamptz" },
	{ "timestamp_ne_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  "t/F:timestamp_ne_timestamptz" },

	/* comparison between timestamptz and timestamp  */
	{ "timestamptz_lt_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/F:timestamptz_lt_timestamp" },
	{ "timestamptz_le_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/F:timestamptz_le_timestamp" },
	{ "timestamptz_eq_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/F:timestamptz_eq_timestamp" },
	{ "timestamptz_ge_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/F:timestamptz_ge_timestamp" },
	{ "timestamptz_gt_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/F:timestamptz_gt_timestamp" },
	{ "timestamptz_ne_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
      "t/F:timestamptz_ne_timestamp" },

	/* comparison between intervals */
	{ "interval_eq", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_eq" },
	{ "interval_ne", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_ne" },
	{ "interval_lt", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_lt" },
	{ "interval_le", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_le" },
	{ "interval_ge", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_ge" },
	{ "interval_gt", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_gt" },
	{ "interval_cmp", 2, {INTERVALOID, INTERVALOID}, "t/F:interval_cmp" },

	/* overlaps() */
	{ "overlaps", 4, {TIMEOID, TIMEOID, TIMEOID, TIMEOID},
	  "t/F:overlaps_time" },
	{ "overlaps", 4, {TIMETZOID, TIMETZOID, TIMETZOID, TIMETZOID},
	  "t/F:overlaps_timetz" },
	{ "overlaps", 4, {TIMESTAMPOID, TIMESTAMPOID,
					  TIMESTAMPOID, TIMESTAMPOID},
	  "t/F:overlaps_timestamp" },
	{ "overlaps", 4, {TIMESTAMPTZOID, TIMESTAMPTZOID,
					  TIMESTAMPTZOID, TIMESTAMPTZOID},
	  "t/F:overlaps_timestamptz" },

	/* extract() */
	{ "date_part", 2, {TEXTOID,TIMESTAMPOID},   "st/F:extract_timestamp"},
	{ "date_part", 2, {TEXTOID,TIMESTAMPTZOID}, "st/F:extract_timestamptz"},
	{ "date_part", 2, {TEXTOID,INTERVALOID},    "st/F:extract_interval"},
	{ "date_part", 2, {TEXTOID,TIMETZOID},      "st/F:extract_timetz"},
	{ "date_part", 2, {TEXTOID,TIMEOID},        "st/F:extract_time"},

	/* other time and data functions */
	{ "now", 0, {}, "t/F:now" },

	/* macaddr functions */
	{ "trunc",       1, {MACADDROID},            "y/F:macaddr_trunc" },
	{ "macaddr_not", 1, {MACADDROID},            "y/F:macaddr_not" },
	{ "macaddr_and", 2, {MACADDROID,MACADDROID}, "y/F:macaddr_and" },
	{ "macaddr_or",  2, {MACADDROID,MACADDROID}, "y/F:macaddr_or" },

	/* inet/cidr functions */
	{ "set_masklen", 2, {INETOID,INT4OID},       "y/F:inet_set_masklen" },
	{ "set_masklen", 2, {CIDROID,INT4OID},       "y/F:cidr_set_masklen" },
	{ "family",      1, {INETOID},               "y/F:inet_family" },
	{ "network",     1, {INETOID},               "y/F:network_network" },
	{ "netmask",     1, {INETOID},               "y/F:inet_netmask" },
	{ "masklen",     1, {INETOID},               "y/F:inet_masklen" },
	{ "broadcast",   1, {INETOID},               "y/F:inet_broadcast" },
	{ "hostmask",    1, {INETOID},               "y/F:inet_hostmask" },
	{ "cidr",        1, {INETOID},               "y/F:inet_to_cidr" },
	{ "inetnot",     1, {INETOID},               "y/F:inet_not" },
	{ "inetand",     2, {INETOID,INETOID},       "y/F:inet_and" },
	{ "inetor",      2, {INETOID,INETOID},       "y/F:inet_or" },
	{ "inetpl",      2, {INETOID,INT8OID},       "y/F:inetpl_int8" },
	{ "inetmi_int8", 2, {INETOID,INT8OID},       "y/F:inetmi_int8" },
	{ "inetmi",      2, {INETOID,INETOID},       "y/F:inetmi" },
	{ "inet_same_family", 2, {INETOID,INETOID},  "y/F:inet_same_family" },
	{ "inet_merge",  2, {INETOID,INETOID},       "y/F:inet_merge" },

	/*
	 * Text functions
	 * ---------------------- */
	{ "bpchareq",  2, {BPCHAROID,BPCHAROID},  "s/F:bpchareq" },
	{ "bpcharne",  2, {BPCHAROID,BPCHAROID},  "s/F:bpcharne" },
	{ "bpcharlt",  2, {BPCHAROID,BPCHAROID},  "sc/F:bpcharlt" },
	{ "bpcharle",  2, {BPCHAROID,BPCHAROID},  "sc/F:bpcharle" },
	{ "bpchargt",  2, {BPCHAROID,BPCHAROID},  "sc/F:bpchargt" },
	{ "bpcharge",  2, {BPCHAROID,BPCHAROID},  "sc/F:bpcharge" },
	{ "bpcharcmp", 2, {BPCHAROID, BPCHAROID}, "sc/F:bpcharcmp"},
	{ "length",    1, {BPCHAROID},            "sc/F:bpcharlen"},
	{ "texteq",    2, {TEXTOID, TEXTOID},     "s/F:texteq" },
	{ "textne",    2, {TEXTOID, TEXTOID},     "s/F:textne" },
	{ "text_lt",   2, {TEXTOID, TEXTOID},     "sc/F:text_lt" },
	{ "text_le",   2, {TEXTOID, TEXTOID},     "sc/F:text_le" },
	{ "text_gt",   2, {TEXTOID, TEXTOID},     "sc/F:text_gt" },
	{ "text_ge",   2, {TEXTOID, TEXTOID},     "sc/F:text_ge" },
	{ "bttextcmp", 2, {TEXTOID, TEXTOID},     "sc/F:text_cmp" },
	{ "length",    1, {TEXTOID},              "sc/F:textlen" },
	/* LIKE operators */
	{ "like",        2, {TEXTOID, TEXTOID},   "s/F:textlike" },
	{ "textlike",    2, {TEXTOID, TEXTOID},   "s/F:textlike" },
	{ "bpcharlike",  2, {BPCHAROID, TEXTOID}, "s/F:textlike" },
	{ "notlike",     2, {TEXTOID, TEXTOID},   "s/F:textnlike" },
	{ "textnlike",   2, {TEXTOID, TEXTOID},   "s/F:textnlike" },
	{ "bpcharnlike", 2, {BPCHAROID, TEXTOID}, "s/F:textnlike" },
	/* ILIKE operators */
	{ "texticlike",    2, {TEXTOID, TEXTOID},   "sc/F:texticlike" },
	{ "bpchariclike",  2, {TEXTOID, TEXTOID},   "sc/F:texticlike" },
	{ "texticnlike",   2, {TEXTOID, TEXTOID},   "sc/F:texticnlike" },
	{ "bpcharicnlike", 2, {BPCHAROID, TEXTOID}, "sc/F:texticnlike" },
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
#define FLOAT2	"pg_catalog.float2"
#define FLOAT4	"real"
#define FLOAT8	"double precision"
#define NUMERIC	"numeric"

static devfunc_extra_catalog_t devfunc_extra_catalog[] = {
	/* float2 - type cast functions */
	{ FLOAT4,  "pgstrom.float4("FLOAT2")",  "a/c:" },
	{ FLOAT8,  "pgstrom.float8("FLOAT2")",  "a/c:" },
	{ INT2,    "pgstrom.int2("FLOAT2")",    "a/c:" },
	{ INT4,    "pgstrom.int4("FLOAT2")",    "a/c:" },
	{ INT8,    "pgstrom.int8("FLOAT2")",    "a/c:" },
	{ NUMERIC, "pgstrom.numeric("FLOAT2")", "n/F:float2_numeric" },
	{ FLOAT2,  "pgstrom.float2("FLOAT4")",  "a/c:" },
	{ FLOAT2,  "pgstrom.float2("FLOAT8")",  "a/c:" },
	{ FLOAT2,  "pgstrom.float2("INT2")",    "a/c:" },
	{ FLOAT2,  "pgstrom.float2("INT4")",    "a/c:" },
	{ FLOAT2,  "pgstrom.float2("INT8")",    "a/c:" },
	{ FLOAT2,  "pgstrom.float2("NUMERIC")", "n/F:numeric_float2" },
	/* float2 - type comparison functions */
	{ BOOL,    "pgstrom.float2_eq("FLOAT2","FLOAT2")",  "b:==" },
	{ BOOL,    "pgstrom.float2_ne("FLOAT2","FLOAT2")",  "b:!=" },
	{ BOOL,    "pgstrom.float2_lt("FLOAT2","FLOAT2")",  "b:<" },
	{ BOOL,    "pgstrom.float2_le("FLOAT2","FLOAT2")",  "b:<=" },
	{ BOOL,    "pgstrom.float2_gt("FLOAT2","FLOAT2")",  "b:>" },
	{ BOOL,    "pgstrom.float2_ge("FLOAT2","FLOAT2")",  "b:>=" },
	{ BOOL,    "pgstrom.float2_larger("FLOAT2","FLOAT2")",  "f:Max" },
	{ BOOL,    "pgstrom.float2_smaller("FLOAT2","FLOAT2")", "f:Min" },

	{ BOOL,    "pgstrom.float42_eq("FLOAT4","FLOAT2")", "b:==" },
	{ BOOL,    "pgstrom.float42_ne("FLOAT4","FLOAT2")", "b:!=" },
	{ BOOL,    "pgstrom.float42_lt("FLOAT4","FLOAT2")", "b:<" },
	{ BOOL,    "pgstrom.float42_le("FLOAT4","FLOAT2")", "b:<=" },
	{ BOOL,    "pgstrom.float42_gt("FLOAT4","FLOAT2")", "b:>" },
	{ BOOL,    "pgstrom.float42_ge("FLOAT4","FLOAT2")", "b:>=" },

	{ BOOL,    "pgstrom.float82_eq("FLOAT8","FLOAT2")", "b:==" },
	{ BOOL,    "pgstrom.float82_ne("FLOAT8","FLOAT2")", "b:!=" },
	{ BOOL,    "pgstrom.float82_lt("FLOAT8","FLOAT2")", "b:<" },
	{ BOOL,    "pgstrom.float82_le("FLOAT8","FLOAT2")", "b:<=" },
	{ BOOL,    "pgstrom.float82_gt("FLOAT8","FLOAT2")", "b:>" },
	{ BOOL,    "pgstrom.float82_ge("FLOAT8","FLOAT2")", "b:>=" },

	{ BOOL,    "pgstrom.float24_eq("FLOAT2","FLOAT4")", "b:==" },
	{ BOOL,    "pgstrom.float24_ne("FLOAT2","FLOAT4")", "b:!=" },
	{ BOOL,    "pgstrom.float24_lt("FLOAT2","FLOAT4")", "b:<" },
	{ BOOL,    "pgstrom.float24_le("FLOAT2","FLOAT4")", "b:<=" },
	{ BOOL,    "pgstrom.float24_gt("FLOAT2","FLOAT4")", "b:>" },
	{ BOOL,    "pgstrom.float24_ge("FLOAT2","FLOAT4")", "b:>=" },

	{ BOOL,    "pgstrom.float28_eq("FLOAT2","FLOAT8")", "b:==" },
	{ BOOL,    "pgstrom.float28_ne("FLOAT2","FLOAT8")", "b:!=" },
	{ BOOL,    "pgstrom.float28_lt("FLOAT2","FLOAT8")", "b:<" },
	{ BOOL,    "pgstrom.float28_le("FLOAT2","FLOAT8")", "b:<=" },
	{ BOOL,    "pgstrom.float28_gt("FLOAT2","FLOAT8")", "b:>" },
	{ BOOL,    "pgstrom.float28_ge("FLOAT2","FLOAT8")", "b:>=" },

	/* float2 - unary operator */
	{ FLOAT2,  "pgstrom.float2_up("FLOAT2")",  "l:+" },
	{ FLOAT2,  "pgstrom.float2_um("FLOAT2")",  "l:-" },
	{ FLOAT2,  "pgstrom.float2_abs("FLOAT2")", "f:abs" },

	/* float2 - arithmetic operators */
	{ FLOAT4, "pgstrom.float2_pl("FLOAT2","FLOAT2")",   "m/F:float2pl" },
	{ FLOAT4, "pgstrom.float2_mi("FLOAT2","FLOAT2")",   "m/F:float2mi" },
	{ FLOAT4, "pgstrom.float2_mul("FLOAT2","FLOAT2")",  "m/F:float2mul" },
	{ FLOAT4, "pgstrom.float2_div("FLOAT2","FLOAT2")",  "m/F:float2div" },
	{ FLOAT4, "pgstrom.float24_pl("FLOAT2","FLOAT4")",  "m/F:float24pl" },
	{ FLOAT4, "pgstrom.float24_mi("FLOAT2","FLOAT4")",  "m/F:float24mi" },
	{ FLOAT4, "pgstrom.float24_mul("FLOAT2","FLOAT4")", "m/F:float24mul" },
	{ FLOAT4, "pgstrom.float24_div("FLOAT2","FLOAT4")", "m/F:float24div" },
	{ FLOAT8, "pgstrom.float28_pl("FLOAT2","FLOAT8")",  "m/F:float28pl" },
	{ FLOAT8, "pgstrom.float28_mi("FLOAT2","FLOAT8")",  "m/F:float28mi" },
	{ FLOAT8, "pgstrom.float28_mul("FLOAT2","FLOAT8")", "m/F:float28mul" },
	{ FLOAT8, "pgstrom.float28_div("FLOAT2","FLOAT8")", "m/F:float28div" },
	{ FLOAT4, "pgstrom.float42_pl("FLOAT4","FLOAT2")",  "m/F:float42pl" },
	{ FLOAT4, "pgstrom.float42_mi("FLOAT4","FLOAT2")",  "m/F:float42mi" },
	{ FLOAT4, "pgstrom.float42_mul("FLOAT4","FLOAT2")", "m/F:float42mul" },
	{ FLOAT4, "pgstrom.float42_div("FLOAT4","FLOAT2")", "m/F:float42div" },
	{ FLOAT8, "pgstrom.float82_pl("FLOAT8","FLOAT2")",  "m/F:float82pl" },
	{ FLOAT8, "pgstrom.float82_mi("FLOAT8","FLOAT2")",  "m/F:float82mi" },
	{ FLOAT8, "pgstrom.float82_mul("FLOAT8","FLOAT2")", "m/F:float82mul" },
	{ FLOAT8, "pgstrom.float82_div("FLOAT8","FLOAT2")", "m/F:float82div" },
	{ "money", "pgstrom.cash_mul_flt2(money,"FLOAT2")", "y:cash_mul_flt2" },
	{ "money", "pgstrom.flt2_mul_cash("FLOAT2",money)", "y:flt2_mul_cash" },
	{ "money", "pgstrom.cash_div_flt2(money,"FLOAT2")", "y:cash_div_flt2" },

	/* int4range operators */
	{ INT4, "lower(int4range)",               "r/F:int4range_lower" },
	{ INT4, "upper(int4range)",               "r/F:int4range_upper" },
	{ BOOL, "isempty(int4range)",             "r/F:generic_range_isempty" },
	{ BOOL, "lower_inc(int4range)",           "r/F:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int4range)",           "r/F:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int4range)",           "r/F:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int4range)",           "r/F:generic_range_upper_inf" },
	{ BOOL, "range_eq(int4range,int4range)",  "r/F:generic_range_eq" },
	{ BOOL, "range_ne(int4range,int4range)",  "r/F:generic_range_ne" },
	{ BOOL, "range_lt(int4range,int4range)",  "r/F:generic_range_lt" },
	{ BOOL, "range_le(int4range,int4range)",  "r/F:generic_range_le" },
	{ BOOL, "range_gt(int4range,int4range)",  "r/F:generic_range_gt" },
	{ BOOL, "range_ge(int4range,int4range)",  "r/F:generic_range_ge" },
	{ INT4, "range_cmp(int4range,int4range)", "r/F:generic_range_cmp" },
	{ BOOL, "range_overlaps(int4range,int4range)",
	  "r/F:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int4range,"INT4")",
	  "r/F:generic_range_contains_elem" },
	{ BOOL, "range_contains(int4range,int4range)",
	  "r/F:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT4",int4range)",
	  "r/F:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int4range,int4range)",
	  "r/F:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int4range,int4range)",
	  "r/F:generic_range_adjacent" },
	{ BOOL, "range_before(int4range,int4range)",
	  "r/F:generic_range_before" },
	{ BOOL, "range_after(int4range,int4range)",
	  "r/F:generic_range_after" },
	{ BOOL, "range_overleft(int4range,int4range)",
	  "r/F:generic_range_overleft" },
	{ BOOL, "range_overright(int4range,int4range)",
	  "r/F:generic_range_overleft" },
	{ "int4range", "range_union(int4range,int4range)",
	  "r/F:generic_range_union" },
	{ "int4range", "range_merge(int4range,int4range)",
	  "r/F:generic_range_merge" },
	{ "int4range", "range_intersect(int4range,int4range)",
	  "r/F:generic_range_intersect" },
	{ "int4range", "range_minus(int4range,int4range)",
	  "r/F:generic_range_minus" },

	/* int8range operators */
	{ INT8, "lower(int8range)",               "r/F:int8range_lower" },
	{ INT8, "upper(int8range)",               "r/F:int8range_upper" },
	{ BOOL, "isempty(int8range)",             "r/F:generic_range_isempty" },
	{ BOOL, "lower_inc(int8range)",           "r/F:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int8range)",           "r/F:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int8range)",           "r/F:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int8range)",           "r/F:generic_range_upper_inf" },
	{ BOOL, "range_eq(int8range,int8range)",  "r/F:generic_range_eq" },
	{ BOOL, "range_ne(int8range,int8range)",  "r/F:generic_range_ne" },
	{ BOOL, "range_lt(int8range,int8range)",  "r/F:generic_range_lt" },
	{ BOOL, "range_le(int8range,int8range)",  "r/F:generic_range_le" },
	{ BOOL, "range_gt(int8range,int8range)",  "r/F:generic_range_gt" },
	{ BOOL, "range_ge(int8range,int8range)",  "r/F:generic_range_ge" },
	{ INT4, "range_cmp(int8range,int8range)", "r/F:generic_range_cmp" },
	{ BOOL, "range_overlaps(int8range,int8range)",
	  "r/F:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int8range,"INT8")",
	  "r/F:generic_range_contains_elem" },
	{ BOOL, "range_contains(int8range,int8range)",
	  "r/F:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT8",int8range)",
	  "r/F:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int8range,int8range)",
	  "r/F:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int8range,int8range)",
	  "r/F:generic_range_adjacent" },
	{ BOOL, "range_before(int8range,int8range)",
	  "r/F:generic_range_before" },
	{ BOOL, "range_after(int8range,int8range)",
	  "r/F:generic_range_after" },
	{ BOOL, "range_overleft(int8range,int8range)",
	  "r/F:generic_range_overleft" },
	{ BOOL, "range_overright(int8range,int8range)",
	  "r/F:generic_range_overleft" },
	{ "int8range", "range_union(int8range,int8range)",
	  "r/F:generic_range_union" },
	{ "int8range", "range_merge(int8range,int8range)",
	  "r/F:generic_range_merge" },
	{ "int8range", "range_intersect(int8range,int8range)",
	  "r/F:generic_range_intersect" },
	{ "int8range", "range_minus(int8range,int8range)",
	  "r/F:generic_range_minus" },

	/* tsrange operators */
	{ "timestamp", "lower(tsrange)",      "r/F:tsrange_lower" },
	{ "timestamp", "upper(tsrange)",      "r/F:tsrange_upper" },
	{ BOOL, "isempty(tsrange)",           "r/F:generic_range_isempty" },
	{ BOOL, "lower_inc(tsrange)",         "r/F:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tsrange)",         "r/F:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tsrange)",         "r/F:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tsrange)",         "r/F:generic_range_upper_inf" },
	{ BOOL, "range_eq(tsrange,tsrange)",  "r/F:generic_range_eq" },
	{ BOOL, "range_ne(tsrange,tsrange)",  "r/F:generic_range_ne" },
	{ BOOL, "range_lt(tsrange,tsrange)",  "r/F:generic_range_lt" },
	{ BOOL, "range_le(tsrange,tsrange)",  "r/F:generic_range_le" },
	{ BOOL, "range_gt(tsrange,tsrange)",  "r/F:generic_range_gt" },
	{ BOOL, "range_ge(tsrange,tsrange)",  "r/F:generic_range_ge" },
	{ INT4, "range_cmp(tsrange,tsrange)", "r/F:generic_range_cmp" },
	{ BOOL, "range_overlaps(tsrange,tsrange)",
	  "r/F:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tsrange,timestamp)",
	  "r/F:generic_range_contains_elem" },
	{ BOOL, "range_contains(tsrange,tsrange)",
	  "r/F:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamp,tsrange)",
	  "r/F:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tsrange,tsrange)",
	  "r/F:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tsrange,tsrange)",
	  "r/F:generic_range_adjacent" },
	{ BOOL, "range_before(tsrange,tsrange)",
	  "r/F:generic_range_before" },
	{ BOOL, "range_after(tsrange,tsrange)",
	  "r/F:generic_range_after" },
	{ BOOL, "range_overleft(tsrange,tsrange)",
	  "r/F:generic_range_overleft" },
	{ BOOL, "range_overright(tsrange,tsrange)",
	  "r/F:generic_range_overleft" },
	{ "tsrange", "range_union(tsrange,tsrange)",
	  "r/F:generic_range_union" },
	{ "tsrange", "range_merge(tsrange,tsrange)",
	  "r/F:generic_range_merge" },
	{ "tsrange", "range_intersect(tsrange,tsrange)",
	  "r/F:generic_range_intersect" },
	{ "tsrange", "range_minus(tsrange,tsrange)",
	  "r/F:generic_range_minus" },

	/* tstzrange operators */
	{ "timestamptz", "lower(tstzrange)",      "r/F:tstzrange_lower" },
	{ "timestamptz", "upper(tstzrange)",      "r/F:tstzrange_upper" },
	{ BOOL, "isempty(tstzrange)",             "r/F:generic_range_isempty" },
	{ BOOL, "lower_inc(tstzrange)",           "r/F:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tstzrange)",           "r/F:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tstzrange)",           "r/F:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tstzrange)",           "r/F:generic_range_upper_inf" },
	{ BOOL, "range_eq(tstzrange,tstzrange)",  "r/F:generic_range_eq" },
	{ BOOL, "range_ne(tstzrange,tstzrange)",  "r/F:generic_range_ne" },
	{ BOOL, "range_lt(tstzrange,tstzrange)",  "r/F:generic_range_lt" },
	{ BOOL, "range_le(tstzrange,tstzrange)",  "r/F:generic_range_le" },
	{ BOOL, "range_gt(tstzrange,tstzrange)",  "r/F:generic_range_gt" },
	{ BOOL, "range_ge(tstzrange,tstzrange)",  "r/F:generic_range_ge" },
	{ INT4, "range_cmp(tstzrange,tstzrange)", "r/F:generic_range_cmp" },
	{ BOOL, "range_overlaps(tstzrange,tstzrange)",
	  "r/F:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tstzrange,timestamptz)",
	  "r/F:generic_range_contains_elem" },
	{ BOOL, "range_contains(tstzrange,tstzrange)",
	  "r/F:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamptz,tstzrange)",
	  "r/F:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tstzrange,tstzrange)",
	  "r/F:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tstzrange,tstzrange)",
	  "r/F:generic_range_adjacent" },
	{ BOOL, "range_before(tstzrange,tstzrange)",
	  "r/F:generic_range_before" },
	{ BOOL, "range_after(tstzrange,tstzrange)",
	  "r/F:generic_range_after" },
	{ BOOL, "range_overleft(tstzrange,tstzrange)",
	  "r/F:generic_range_overleft" },
	{ BOOL, "range_overright(tstzrange,tstzrange)",
	  "r/F:generic_range_overleft" },
	{ "tstzrange", "range_union(tstzrange,tstzrange)",
	  "r/F:generic_range_union" },
	{ "tstzrange", "range_merge(tstzrange,tstzrange)",
	  "r/F:generic_range_merge" },
	{ "tstzrange", "range_intersect(tstzrange,tstzrange)",
	  "r/F:generic_range_intersect" },
	{ "tstzrange", "range_minus(tstzrange,tstzrange)",
	  "r/F:generic_range_minus" },

	/* daterange operators */
	{ "date", "lower(daterange)",             "r/F:daterange_lower" },
	{ "date", "upper(daterange)",             "r/F:daterange_upper" },
	{ BOOL, "isempty(daterange)",             "r/F:generic_range_isempty" },
	{ BOOL, "lower_inc(daterange)",           "r/F:generic_range_lower_inc" },
	{ BOOL, "upper_inc(daterange)",           "r/F:generic_range_upper_inc" },
	{ BOOL, "lower_inf(daterange)",           "r/F:generic_range_lower_inf" },
	{ BOOL, "upper_inf(daterange)",           "r/F:generic_range_upper_inf" },
	{ BOOL, "range_eq(daterange,daterange)",  "r/F:generic_range_eq" },
	{ BOOL, "range_ne(daterange,daterange)",  "r/F:generic_range_ne" },
	{ BOOL, "range_lt(daterange,daterange)",  "r/F:generic_range_lt" },
	{ BOOL, "range_le(daterange,daterange)",  "r/F:generic_range_le" },
	{ BOOL, "range_gt(daterange,daterange)",  "r/F:generic_range_gt" },
	{ BOOL, "range_ge(daterange,daterange)",  "r/F:generic_range_ge" },
	{ INT4, "range_cmp(daterange,daterange)", "r/F:generic_range_cmp" },
	{ BOOL, "range_overlaps(daterange,daterange)",
	  "r/F:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(daterange,date)",
	  "r/F:generic_range_contains_elem" },
	{ BOOL, "range_contains(daterange,daterange)",
	  "r/F:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(date,daterange)",
	  "r/F:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(daterange,daterange)",
	  "r/F:generic_range_contained_by" },
	{ BOOL, "range_adjacent(daterange,daterange)",
	  "r/F:generic_range_adjacent" },
	{ BOOL, "range_before(daterange,daterange)",
	  "r/F:generic_range_before" },
	{ BOOL, "range_after(daterange,daterange)",
	  "r/F:generic_range_after" },
	{ BOOL, "range_overleft(daterange,daterange)",
	  "r/F:generic_range_overleft" },
	{ BOOL, "range_overright(daterange,daterange)",
	  "r/F:generic_range_overleft" },
	{ "daterange", "range_union(daterange,daterange)",
	  "r/F:generic_range_union" },
	{ "daterange", "range_merge(daterange,daterange)",
	  "r/F:generic_range_merge" },
	{ "daterange", "range_intersect(daterange,daterange)",
	  "r/F:generic_range_intersect" },
	{ "daterange", "range_minus(daterange,daterange)",
	  "r/F:generic_range_minus" },

	/* type re-interpretation */
	{ INT8,   "as_int8("FLOAT8")", "f:__double_as_longlong" },
	{ INT4,   "as_int4("FLOAT4")", "f:__float_as_int" },
	{ INT2,   "as_int2("FLOAT2")", "f:__half_as_short" },
	{ FLOAT8, "as_float8("INT8")", "f:__longlong_as_double" },
	{ FLOAT4, "as_float4("INT4")", "f:__int_as_float" },
	{ FLOAT2, "as_float2("INT2")", "f:__short_as_half" },
};

#undef BOOL
#undef INT2
#undef INT4
#undef INT8
#undef FLOAT2
#undef FLOAT4
#undef FLOAT8
#undef NUMERIC

static void
devfunc_setup_cast(devfunc_info *entry,
				   const char *extra, bool has_alias)
{
	devtype_info   *dtype = linitial(entry->func_args);

	Assert(list_length(entry->func_args) == 1);
	entry->func_devname = (!has_alias
						   ? entry->func_sqlname
						   : psprintf("%s_%s",
									  dtype->type_name,
									  entry->func_rettype->type_name));
	entry->func_decl
		= psprintf("STATIC_FUNCTION(pg_%s_t)\n"
				   "pgfn_%s(kern_context *kcxt, pg_%s_t arg)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    result.value  = (%s)arg.value;\n"
				   "    result.isnull = arg.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_devname,
				   dtype->type_name,
				   entry->func_rettype->type_name,
				   entry->func_rettype->type_base);
}

static void
devfunc_setup_oper_both(devfunc_info *entry,
						const char *extra, bool has_alias, bool add_type_cast)
{
	devtype_info   *dtype1 = linitial(entry->func_args);
	devtype_info   *dtype2 = lsecond(entry->func_args);
	const char	   *type_cast1 = "";
	const char	   *type_cast2 = "";
	char			temp[NAMEDATALEN+10];

	Assert(list_length(entry->func_args) == 2);
	entry->func_devname = (!has_alias
						   ? entry->func_sqlname
						   : psprintf("%s_%s_%s",
									  entry->func_sqlname,
									  dtype1->type_name,
									  dtype2->type_name));
	if (add_type_cast && dtype1->type_oid != dtype2->type_oid)
	{
		if (dtype1->type_length >= dtype2->type_length)
		{
			snprintf(temp, sizeof(temp), "(%s) ", dtype1->type_base);
			type_cast2 = temp;
		}
		else
		{
			snprintf(temp, sizeof(temp), "(%s) ", dtype2->type_base);
			type_cast1 = temp;
		}
	}

	entry->func_decl
		= psprintf("STATIC_FUNCTION(pg_%s_t)\n"
				   "pgfn_%s(kern_context *kcxt, pg_%s_t arg1, pg_%s_t arg2)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    result.value = (%s)(%sarg1.value %s %sarg2.value);\n"
				   "    result.isnull = arg1.isnull | arg2.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_devname,
				   dtype1->type_name,
				   dtype2->type_name,
				   entry->func_rettype->type_name,
				   entry->func_rettype->type_base,
				   type_cast1,
				   extra,
				   type_cast2);
}

static void
devfunc_setup_oper_either(devfunc_info *entry,
						  const char *left_extra,
						  const char *right_extra,
						  bool has_alias)
{
	devtype_info   *dtype = linitial(entry->func_args);

	Assert(list_length(entry->func_args) == 1);
	entry->func_devname = (!has_alias
						   ? entry->func_sqlname
						   : psprintf("%s_%s",
									  entry->func_sqlname,
									  dtype->type_name));
	entry->func_decl
		= psprintf("STATIC_FUNCTION(pg_%s_t)\n"
				   "pgfn_%s(kern_context *kcxt, pg_%s_t arg)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    result.value = (%s)(%sarg.value%s);\n"
				   "    result.isnull = arg.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_devname,
				   dtype->type_name,
				   entry->func_rettype->type_name,
				   entry->func_rettype->type_base,
				   !left_extra ? "" : left_extra,
				   !right_extra ? "" : right_extra);
}

static void
devfunc_setup_oper_left(devfunc_info *entry,
						const char *extra, bool has_alias)
{
	devfunc_setup_oper_either(entry, extra, NULL, has_alias);
}

static void
devfunc_setup_oper_right(devfunc_info *entry,
						 const char *extra, bool has_alias)
{
	devfunc_setup_oper_either(entry, NULL, extra, has_alias);
}

static void
devfunc_setup_func_decl(devfunc_info *entry,
						const char *extra, bool has_alias)
{
	StringInfoData	str;
	ListCell	   *cell;
	int				index;

	initStringInfo(&str);
	if (!has_alias)
		entry->func_devname = entry->func_sqlname;
	else
	{
		appendStringInfo(&str, "%s", entry->func_sqlname);
		foreach (cell, entry->func_args)
		{
			devtype_info   *dtype = lfirst(cell);

			appendStringInfo(&str, "_%s", dtype->type_name);
		}
		entry->func_devname = pstrdup(str.data);
	}

	/* declaration */
	resetStringInfo(&str);
	appendStringInfo(&str,
					 "STATIC_FUNCTION(pg_%s_t)\n"
					 "pgfn_%s(kern_context *kcxt",
					 entry->func_rettype->type_name,
					 entry->func_devname);
	index = 1;
	foreach (cell, entry->func_args)
	{
		devtype_info   *dtype = lfirst(cell);

		appendStringInfo(&str, ", pg_%s_t arg%d",
						 dtype->type_name,
						 index++);
	}
	appendStringInfo(&str, ")\n"
					 "{\n"
					 "    pg_%s_t result;\n"
					 "    result.isnull = ",
					 entry->func_rettype->type_name);
	if (entry->func_args == NIL)
		appendStringInfo(&str, "false");
	else
	{
		index = 1;
		foreach (cell, entry->func_args)
		{
			appendStringInfo(&str, "%sarg%d.isnull",
							 cell == list_head(entry->func_args) ? "" : " | ",
							 index++);
		}
	}
	appendStringInfo(&str, ";\n"
					 "    if (!result.isnull)\n"
					 "        result.value = (%s) %s(",
					 entry->func_rettype->type_base,
					 extra);
	index = 1;
	foreach (cell, entry->func_args)
	{
		appendStringInfo(&str, "%sarg%d.value",
						 cell == list_head(entry->func_args) ? "" : ", ",
						 index++);
	}
	appendStringInfo(&str, ");\n"
					 "    return result;\n"
					 "}\n");
	entry->func_decl = str.data;
}

static void
devfunc_setup_func_impl(devfunc_info *entry,
						const char *extra, bool has_alias)
{
	if (has_alias)
		elog(ERROR, "Bug? implimented device function should not have alias");
	entry->func_devname = extra;
}

static void
__construct_devfunc_info(devfunc_info *entry,
						 const char *template)
{
	const char *extra;
	const char *pos;
	const char *end;
	int32		flags = 0;
	bool		has_alias = false;
	bool		has_collation = false;

	/* fetch attribute */
	end = strchr(template, '/');
	if (end)
	{
		for (pos = template; pos < end; pos++)
		{
			switch (*pos)
			{
				case 'a':
					has_alias = true;
					break;
				case 'c':
					has_collation = true;
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
	else
	{
		if (OidIsValid(entry->func_collid) &&
			!lc_collate_is_c(entry->func_collid))
		{
			entry->func_is_negative = true;
			return;
		}
	}
	extra = template + 2;
	if (strncmp(template, "c:", 2) == 0)
		devfunc_setup_cast(entry, extra, has_alias);
	else if (strncmp(template, "b:", 2) == 0)
		devfunc_setup_oper_both(entry, extra, has_alias, true);
	else if (strncmp(template, "B:", 2) == 0)
		devfunc_setup_oper_both(entry, extra, has_alias, false);
	else if (strncmp(template, "l:", 2) == 0)
		devfunc_setup_oper_left(entry, extra, has_alias);
	else if (strncmp(template, "r:", 2) == 0)
		devfunc_setup_oper_right(entry, extra, has_alias);
	else if (strncmp(template, "f:", 2) == 0)
		devfunc_setup_func_decl(entry, extra, has_alias);
	else if (strncmp(template, "F:", 2) == 0)
		devfunc_setup_func_impl(entry, extra, has_alias);
	else
	{
		elog(NOTICE, "Bug? unknown device function template: '%s'",
			 template);
		entry->func_is_negative = true;
	}
}

static bool
pgstrom_devfunc_construct_common(devfunc_info *entry)
{
	int		i;

	for (i=0; i < lengthof(devfunc_common_catalog); i++)
	{
		devfunc_catalog_t  *procat = devfunc_common_catalog + i;

		if (strcmp(procat->func_name, entry->func_sqlname) == 0 &&
			procat->func_nargs == entry->func_argtypes->dim1 &&
			memcmp(procat->func_argtypes,
				   entry->func_argtypes->values,
				   sizeof(Oid) * procat->func_nargs) == 0)
		{
			__construct_devfunc_info(entry, procat->func_template);
			return true;
		}
	}
	return false;
}

static bool
pgstrom_devfunc_construct_extra(devfunc_info *entry, HeapTuple protup)
{
	Form_pg_proc proc = (Form_pg_proc) GETSTRUCT(protup);
	StringInfoData sig;
	oidvector  *func_argtypes = entry->func_argtypes;
	int			i, nargs = func_argtypes->dim1;
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

	for (i=0; i < nargs; i++)
	{
		Oid		type_oid = entry->func_argtypes->values[i];

		if (i > 0)
			appendStringInfoChar(&sig, ',');
		temp = format_type_be_qualified(type_oid);
		if (strncmp(temp, "pg_catalog.", 11) == 0)
			appendStringInfo(&sig, "%s", temp + 11);
		else
			appendStringInfo(&sig, "%s", temp);
		pfree(temp);
	}
	appendStringInfoChar(&sig, ')');

	temp = format_type_be_qualified(entry->func_rettype_oid);
	if (strncmp(temp, "pg_catalog.", 11) == 0)
		func_rettype = temp + 11;
	else
		func_rettype = temp;

	elog(INFO, "sig=[%s] ret=[%s]", sig.data, func_rettype);

	for (i=0; i < lengthof(devfunc_extra_catalog); i++)
	{
		devfunc_extra_catalog_t  *procat = devfunc_extra_catalog + i;

		if (strcmp(procat->func_signature, sig.data) == 0 &&
			strcmp(procat->func_rettype, func_rettype) == 0)
		{
			__construct_devfunc_info(entry, procat->func_template);
			result = true;
			break;
		}
	}
	pfree(sig.data);
	pfree(temp);
	return result;
}

static devfunc_info *
__pgstrom_devfunc_lookup(HeapTuple protup,
						 Oid func_rettype,
						 oidvector *func_argtypes,
						 Oid func_collid)
{
	Form_pg_proc	proc = (Form_pg_proc) GETSTRUCT(protup);
	Oid				func_oid = HeapTupleGetOid(protup);
	devfunc_info   *entry;
	devtype_info   *dtype;
	List		   *func_args = NIL;
	ListCell	   *lc;
	pg_crc32c		hash;
	int				i, j;
	MemoryContext	oldcxt;

	INIT_CRC32C(hash);
	COMP_CRC32C(hash, &func_oid, sizeof(Oid));
	COMP_CRC32C(hash, &func_rettype, sizeof(Oid));
	COMP_CRC32C(hash, func_argtypes->values,
				sizeof(Oid) * func_argtypes->dim1);
	FIN_CRC32C(hash);

	i = hash % lengthof(devfunc_info_slot);
	foreach (lc, devfunc_info_slot[i])
	{
		entry = lfirst(lc);

		if (entry->hash == hash &&
			entry->func_oid == func_oid &&
			entry->func_rettype_oid == func_rettype &&
			entry->func_argtypes->dim1 == func_argtypes->dim1 &&
			memcmp(entry->func_argtypes->values,
				   func_argtypes->values,
				   sizeof(Oid) * func_argtypes->dim1) == 0 &&
			(!OidIsValid(entry->func_collid) ||
			 entry->func_collid == func_collid))
		{
			if (entry->func_is_negative)
				return NULL;
			return entry;
		}
	}

	/*
	 * Not found, construct a new entry of the device function
	 */
	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	entry = palloc0(sizeof(devfunc_info));
	entry->hash = hash;
	entry->func_oid = func_oid;
	entry->func_rettype_oid = func_rettype;
	entry->func_argtypes = (oidvector *)
		pg_detoast_datum_copy((struct varlena *)func_argtypes);
	entry->func_collid = func_collid;	/* may be cleared later */
	entry->func_is_strict = proc->proisstrict;
	/* above is signature of SQL functions */
	for (j=0; j < proc->pronargs; j++)
	{
		dtype = pgstrom_devtype_lookup(func_argtypes->values[j]);
		if (!dtype)
		{
			list_free(func_args);
			entry->func_is_negative = true;
			goto skip;
		}
		func_args = lappend(func_args, dtype);
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

	elog(INFO, "funcid = %u rettype = %u arg1 = %u", func_oid, func_rettype, func_argtypes->values[0]);

	/* for system default functions (pg_catalog) */
	if (proc->pronamespace == PG_CATALOG_NAMESPACE &&
		pgstrom_devfunc_construct_common(entry))
		goto skip;
	/* other extra or polymorphic functions */
	if (pgstrom_devfunc_construct_extra(entry, protup))
		goto skip;
	/* for inline PL/CUDA functions */
	if (proc->prolang == get_language_oid("plcuda", true) &&
		pgstrom_devfunc_construct_plcuda(entry, protup))
		goto skip;
	/* oops, function has no entry */
	entry->func_is_negative = true;
skip:
	devfunc_info_slot[i] = lappend(devfunc_info_slot[i], entry);
	MemoryContextSwitchTo(oldcxt);

	if (entry->func_is_negative)
		return NULL;
	return entry;
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
		func_argtypes->dim1 = proc->pronargs;
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
	Form_pg_proc proc;

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
	Form_pg_proc proc;

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
static void codegen_coalesce_expression(CoalesceExpr *coalesce,
										codegen_context *context);
static void codegen_minmax_expression(MinMaxExpr *minmax,
									  codegen_context *context);
static void codegen_scalar_array_op_expression(ScalarArrayOpExpr *opexpr,
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

		appendStringInfo(&context->str,
						 "pgfn_%s(kcxt", dfunc->func_devname);

		foreach (cell, func->args)
		{
			appendStringInfo(&context->str, ", ");
			codegen_expression_walker(lfirst(cell), context);
		}
		appendStringInfoChar(&context->str, ')');
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

		appendStringInfo(&context->str,
						 "pgfn_%s(kcxt", dfunc->func_devname);

		foreach (cell, op->args)
		{
			appendStringInfo(&context->str, ", ");
			codegen_expression_walker(lfirst(cell), context);
		}
		appendStringInfoChar(&context->str, ')');
	}
	else if (IsA(node, NullTest))
	{
		NullTest   *nulltest = (NullTest *) node;
		Oid			typeoid = exprType((Node *)nulltest->arg);
		const char *func_name;

		if (nulltest->argisrow)
			elog(ERROR, "codegen: NullTest towards RECORD data");

		dtype = pgstrom_devtype_lookup_and_track(typeoid, context);
		if (!dtype)
			elog(ERROR, "codegen: failed to lookup device type: %s",
				 format_type_be(typeoid));

		switch (nulltest->nulltesttype)
		{
			case IS_NULL:
				func_name = "isnull";
				break;
			case IS_NOT_NULL:
				func_name = "isnotnull";
				break;
			default:
				elog(ERROR, "unrecognized nulltesttype: %d",
					 (int)nulltest->nulltesttype);
				break;
		}
		appendStringInfo(&context->str, "pgfn_%s_%s(kcxt, ",
						 dtype->type_name, func_name);
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
			appendStringInfo(&context->str, "(!");
			codegen_expression_walker(linitial(b->args), context);
			appendStringInfoChar(&context->str, ')');
		}
		else if (b->boolop == AND_EXPR || b->boolop == OR_EXPR)
		{
			Assert(list_length(b->args) > 1);

			appendStringInfoChar(&context->str, '(');
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
				codegen_expression_walker(lfirst(cell), context);
			}
			appendStringInfoChar(&context->str, ')');
		}
		else
			elog(ERROR, "unrecognized boolop: %d", (int) b->boolop);
	}
	else if (IsA(node, CoalesceExpr))
	{
		CoalesceExpr   *coalesce = (CoalesceExpr *) node;

		codegen_coalesce_expression(coalesce, context);
	}
	else if (IsA(node, MinMaxExpr))
	{
		MinMaxExpr	   *minmax = (MinMaxExpr *) node;

		codegen_minmax_expression(minmax, context);
	}
	else if (IsA(node, RelabelType))
	{
		RelabelType *relabel = (RelabelType *) node;
		/*
		 * RelabelType translates just label of data types. Both of types
		 * same binary form (and also PG-Strom kernel defines all varlena
		 * data types as alias of __global *varlena), so no need to do
		 * anything special.
		 */
		codegen_expression_walker((Node *)relabel->arg, context);
	}
	else if (IsA(node, CaseExpr))
	{
		CaseExpr   *caseexpr = (CaseExpr *) node;
		ListCell   *cell;

		foreach (cell, caseexpr->args)
		{
			CaseWhen   *casewhen = (CaseWhen *) lfirst(cell);

			Assert(IsA(casewhen, CaseWhen));
			if (caseexpr->arg)
			{
				devtype_info   *dtype;
				devfunc_info   *dfunc;
				Oid		expr_type = exprType((Node *)caseexpr->arg);
				Oid		colloid = caseexpr->casecollid;

				dtype = pgstrom_devtype_lookup_and_track(expr_type, context);
				if (!dtype)
					elog(ERROR, "codegen: failed to lookup device type: %s",
						 format_type_be(expr_type));
				dfunc = pgstrom_devfunc_lookup_type_equal(dtype, colloid);
				if (!dfunc)
					elog(ERROR,"codegen: failed to lookup device function: %s",
						 format_procedure_qualified(dtype->type_eqfunc));
				pgstrom_devfunc_track(context, dfunc);

				appendStringInfo(&context->str,
								 "EVAL(pgfn_%s(", dfunc->func_devname);
				codegen_expression_walker((Node *) caseexpr->arg, context);
				appendStringInfo(&context->str, ", ");
				codegen_expression_walker((Node *) casewhen->expr, context);
				appendStringInfo(&context->str, ") ? (");
				codegen_expression_walker((Node *) casewhen->result, context);
				appendStringInfo(&context->str, ") : (");
			}
			else
			{
				Assert(exprType((Node *) casewhen->expr) == BOOLOID);
				Assert(exprType((Node *) casewhen->result) == caseexpr->casetype);

				appendStringInfo(&context->str, "EVAL(");
				codegen_expression_walker((Node *) casewhen->expr, context);
				appendStringInfo(&context->str, ") ? (");
				codegen_expression_walker((Node *) casewhen->result, context);
				appendStringInfo(&context->str, ") : (");
			}
		}
		codegen_expression_walker((Node *) caseexpr->defresult, context);
		foreach (cell, caseexpr->args)
			appendStringInfo(&context->str, ")");
	}
	else if (IsA(node, ScalarArrayOpExpr))
	{
		ScalarArrayOpExpr  *opexpr = (ScalarArrayOpExpr *) node;

		codegen_scalar_array_op_expression(opexpr, context);
	}
	else
		elog(ERROR, "Bug? unsupported expression: %s", nodeToString(node));
}

/*
 * form_devexpr_info
 */
static List *
form_devexpr_info(devexpr_info *devexpr)
{
	devtype_info   *dtype;
	List		   *result = NIL;
	List		   *expr_args = NIL;
	ListCell	   *lc;

	result = lappend(result, makeInteger((long)devexpr->expr_tag));
	result = lappend(result, makeInteger((long)devexpr->expr_collid));
	foreach (lc, devexpr->expr_args)
	{
		dtype = lfirst(lc);
		expr_args = lappend(expr_args, makeInteger((long) dtype->type_oid));
	}
	result = lappend(result, expr_args);

	dtype = devexpr->expr_rettype;
	result = lappend(result, makeInteger((long) dtype->type_oid));
	result = lappend(result, makeInteger((long) devexpr->expr_extra1));
	result = lappend(result, makeInteger((long) devexpr->expr_extra2));
	result = lappend(result, makeString(pstrdup(devexpr->expr_name)));
	result = lappend(result, makeString(pstrdup(devexpr->expr_decl)));

	return result;
}

/*
 * deform_devexpr_info
 */
static void
deform_devexpr_info(devexpr_info *devexpr, List *contents)
{
	ListCell   *cell = list_head(contents);
	ListCell   *lc;

	memset(devexpr, 0, sizeof(devexpr_info));
	devexpr->expr_tag = intVal(lfirst(cell));

	cell = lnext(cell);
	devexpr->expr_collid = intVal(lfirst(cell));

    cell = lnext(cell);
	foreach (lc, (List *)lfirst(cell))
	{
		devtype_info   *dtype = pgstrom_devtype_lookup(intVal(lfirst(lc)));
		if (!dtype)
			elog(ERROR, "failed to lookup device type");
		devexpr->expr_args = lappend(devexpr->expr_args, dtype);
	}

	cell = lnext(cell);
	devexpr->expr_rettype = pgstrom_devtype_lookup(intVal(lfirst(cell)));
	if (!devexpr->expr_rettype)
		elog(ERROR, "failed to lookup device type");

	cell = lnext(cell);
	devexpr->expr_extra1 = (Datum)intVal(lfirst(cell));

	cell = lnext(cell);
	devexpr->expr_extra2 = (Datum)intVal(lfirst(cell));

	cell = lnext(cell);
	devexpr->expr_name = strVal(lfirst(cell));

	cell = lnext(cell);
	devexpr->expr_decl = strVal(lfirst(cell));

	Assert(lnext(cell) == NULL);
}

static void
codegen_coalesce_expression(CoalesceExpr *coalesce, codegen_context *context)
{
	devtype_info   *dtype;
	devexpr_info	devexpr;
	ListCell	   *cell;

	dtype = pgstrom_devtype_lookup(coalesce->coalescetype);
	if (!dtype)
		elog(ERROR, "codegen: unsupported device type in COALESCE: %s",
			 format_type_be(coalesce->coalescetype));

	/* find out identical predefined device COALESCE */
	foreach (cell, context->expr_defs)
	{
		deform_devexpr_info(&devexpr, (List *)lfirst(cell));

		if (devexpr.expr_tag == T_CoalesceExpr &&
			devexpr.expr_rettype->type_oid == coalesce->coalescetype &&
			devexpr.expr_collid == InvalidOid &&
			list_length(devexpr.expr_args) == list_length(coalesce->args))
			break;		/* ok, found */
	}

	/* if no predefined one, make a special expression device function */
	if (!cell)
	{
		StringInfoData decl;
		int		arg_index;

		memset(&devexpr, 0, sizeof(devexpr_info));
		devexpr.expr_tag = T_CoalesceExpr;
		devexpr.expr_collid = InvalidOid;	/* never collation aware */
		foreach (cell, coalesce->args)
		{
			Oid		type_oid = exprType((Node *)lfirst(cell));

			if (dtype->type_oid != type_oid)
				elog(ERROR, "device type mismatch in COALESCE: %s / %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(type_oid));

			devexpr.expr_args = lappend(devexpr.expr_args, dtype);
		}

		if (coalesce->coalescetype != dtype->type_oid)
			elog(ERROR, "device type mismatch in COALESCE: %s / %s",
				 format_type_be(dtype->type_oid),
				 format_type_be(coalesce->coalescetype));

		devexpr.expr_rettype = dtype;
		devexpr.expr_extra1 = 0;		/* no extra information */
		devexpr.expr_extra2 = 0;		/* no extra information */

		/* device function name */
		devexpr.expr_name = psprintf("%s_coalesce_%u",
									 dtype->type_name,
									 list_length(coalesce->args));
		/* device function body */
		initStringInfo(&decl);
		appendStringInfo(&decl,
						 "STATIC_INLINE(pg_%s_t)\n"
						 "pgfn_%s(kern_context *kcxt",
						 dtype->type_name,
						 devexpr.expr_name);
		arg_index = 1;
		foreach (cell, devexpr.expr_args)
		{
			dtype = lfirst(cell);

			appendStringInfo(&decl,
							 ", pg_%s_t arg%d",
							 dtype->type_name,
							 arg_index++);
		}
		appendStringInfo(&decl, ")\n{\n");

		arg_index = 1;
		foreach (cell, devexpr.expr_args)
		{
			appendStringInfo(
				&decl,
				"  if (!arg%d.isnull)\n"
				"    return arg%d;\n",
				arg_index,
				arg_index);
			arg_index++;
		}
		appendStringInfo(
			&decl,
			"\n"
			"  /* return NULL if any arguments are NULL */\n"
			"  memset(&arg1, 0, sizeof(arg1));\n"
			"  arg1.isnull = true;\n"
			"  return arg1;\n"
			"}\n");

		devexpr.expr_decl = decl.data;
		/* track this special expression */
		context->expr_defs = lappend(context->expr_defs,
									 form_devexpr_info(&devexpr));
	}

	/* write out this special expression */
	appendStringInfo(&context->str, "pgfn_%s(kcxt", devexpr.expr_name);
	foreach (cell, coalesce->args)
	{
		Node	   *expr = lfirst(cell);

		if (dtype->type_oid != exprType(expr))
			elog(ERROR, "codegen: device type mismatch in COALESCE: %s / %s",
				 format_type_be(dtype->type_oid),
				 format_type_be(exprType(expr)));

		appendStringInfo(&context->str, ", ");
		codegen_expression_walker(expr, context);
	}
	appendStringInfo(&context->str, ")");
}

static void
codegen_minmax_expression(MinMaxExpr *minmax, codegen_context *context)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	devexpr_info	devexpr;
	ListCell	   *cell;

	if (minmax->op != IS_GREATEST && minmax->op != IS_LEAST)
		elog(ERROR, "unknown operation at MinMaxExpr: %d",
			 (int)minmax->op);

	dtype = pgstrom_devtype_lookup(minmax->minmaxtype);
	if (!dtype)
		elog(ERROR, "unsupported device type in LEAST/GREATEST: %s",
			 format_type_be(minmax->minmaxtype));

	dfunc = pgstrom_devfunc_lookup_type_compare(dtype, minmax->inputcollid);
	if (!dfunc)
		elog(ERROR, "unsupported device function in LEAST/GREATEST: %s",
			 format_procedure_qualified(dtype->type_cmpfunc));
	pgstrom_devfunc_track(context, dfunc);

	/* find out identical predefined device LEAST/GREATEST */
	foreach (cell, context->expr_defs)
	{
		deform_devexpr_info(&devexpr, (List *)lfirst(cell));

		if (devexpr.expr_tag == T_MinMaxExpr &&
			devexpr.expr_rettype->type_oid == minmax->minmaxtype &&
			devexpr.expr_collid == minmax->inputcollid &&
			list_length(devexpr.expr_args) == list_length(minmax->args) &&
			devexpr.expr_extra1 == ObjectIdGetDatum(minmax->op))
			break;		/* ok, found */
	}

	/* if no predefined one, make a special expression device function */
	if (!cell)
	{
		StringInfoData decl;
		int		arg_index;

		memset(&devexpr, 0, sizeof(devexpr_info));
		devexpr.expr_tag = T_MinMaxExpr;
		devexpr.expr_collid = minmax->inputcollid;
		foreach (cell, minmax->args)
		{
			Node		   *expr = lfirst(cell);

			if (dtype->type_oid != exprType(expr))
				elog(ERROR, "device type mismatch in LEAST/GREATEST: %s / %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(exprType(expr)));

			devexpr.expr_args = lappend(devexpr.expr_args, dtype);
		}

		if (dtype->type_oid != minmax->minmaxtype)
			elog(ERROR, "device type mismatch in LEAST/GREATEST: %s / %s",
				 format_type_be(dtype->type_oid),
				 format_type_be(minmax->minmaxtype));

		devexpr.expr_rettype = dtype;
		devexpr.expr_extra1 = (Datum) minmax->op;
		devexpr.expr_name = psprintf("%s_%s_%u",
									 dtype->type_name,
									 minmax->op == IS_LEAST
									 ? "least"
									 : "greatest",
									 list_length(minmax->args));
		/* device function body */
		initStringInfo(&decl);
		appendStringInfo(&decl,
						 "STATIC_INLINE(pg_%s_t)\n"
						 "pgfn_%s(kern_context *kcxt",
						 devexpr.expr_rettype->type_name,
						 devexpr.expr_name);
		arg_index = 1;
		foreach (cell, devexpr.expr_args)
		{
			appendStringInfo(&decl, ", pg_%s_t arg%d",
							 dtype->type_name,
							 arg_index++);
		}
		appendStringInfo(&decl, ")\n"
						 "{\n"
						 "  pg_%s_t   result;\n"
						 "  pg_int4_t eval;\n"
						 "\n"
						 "  memset(&result, 0, sizeof(result));\n"
						 "  result.isnull = true;\n\n",
						 devexpr.expr_rettype->type_name);
		arg_index = 1;
		foreach (cell, devexpr.expr_args)
		{
			appendStringInfo(
				&decl,
				"  if (result.isnull)\n"
				"    result = arg%d;\n"
				"  else if (!arg%d.isnull)\n"
				"  {\n"
				"    eval = pgfn_%s(kcxt, result, arg%d);\n"
				"    if (!eval.isnull && eval.value %s 0)\n"
				"      result = arg%d;\n"
				"  }\n\n",
				arg_index,
				arg_index,
				dfunc->func_devname,
				arg_index,
				minmax->op == IS_LEAST ? ">" : "<",
				arg_index);
			arg_index++;
		}
		appendStringInfo(
			&decl,
			"  return result;\n"
			"}\n\n");

		devexpr.expr_decl = decl.data;
		/* track this special expression */
		context->expr_defs = lappend(context->expr_defs,
									 form_devexpr_info(&devexpr));
	}

	/* write out this special expression */
	appendStringInfo(&context->str, "pgfn_%s(kcxt", devexpr.expr_name);
	foreach (cell, minmax->args)
	{
		Node	   *expr = lfirst(cell);

		if (dtype->type_oid != exprType(expr))
			elog(ERROR, "device type mismatch in LEAST / GREATEST: %s / %s",
				 format_type_be(dtype->type_oid),
				 format_type_be(exprType(expr)));

		appendStringInfo(&context->str, ", ");
		codegen_expression_walker(expr, context);
    }
	appendStringInfo(&context->str, ")");
}

static void
codegen_scalar_array_op_expression(ScalarArrayOpExpr *opexpr,
								   codegen_context *context)
{
	devexpr_info	devexpr;
	devtype_info   *dtype1;
	devtype_info   *dtype2;
	devfunc_info   *dfunc;
	Oid				func_oid;
	Oid				type_oid;
	ListCell	   *cell;
	StringInfoData	decl;

	/* find out identical predefined device ScalarArrayOpExpr */
	foreach (cell, context->expr_defs)
	{
		deform_devexpr_info(&devexpr, (List *)lfirst(cell));

		if (devexpr.expr_tag == T_ScalarArrayOpExpr &&
			devexpr.expr_rettype->type_oid == BOOLOID &&
			devexpr.expr_collid == opexpr->inputcollid &&
			list_length(devexpr.expr_args) == 2 &&
			devexpr.expr_extra1 == ObjectIdGetDatum(opexpr->opno) &&
			devexpr.expr_extra2 == BoolGetDatum(opexpr->useOr))
			goto found;		/* OK, found a predefined one */
	}

	/* no predefined one, create a special expression function */
	memset(&devexpr, 0, sizeof(devexpr_info));
	devexpr.expr_tag = T_ScalarArrayOpExpr;
	devexpr.expr_collid = opexpr->inputcollid;

	func_oid = get_opcode(opexpr->opno);
	dfunc = pgstrom_devfunc_lookup(func_oid,
								   get_func_rettype(func_oid),
								   opexpr->args,
								   opexpr->inputcollid);
	if (!dfunc)
		elog(ERROR, "codegen: failed to lookup device function: %s",
			 format_procedure(get_opcode(opexpr->opno)));
	pgstrom_devfunc_track(context, dfunc);

	/* sanity checks */
	if (dfunc->func_rettype->type_oid != BOOLOID ||
		list_length(dfunc->func_args) != 2)
		elog(ERROR, "sanity check violation at ScalarArrayOp");

	type_oid = exprType(linitial(opexpr->args));
	dtype1 = pgstrom_devtype_lookup(type_oid);
	if (!dtype1 || dtype1->type_element)
		elog(ERROR, "codegen: failed to lookup device type, or array: %s",
			 format_type_be(type_oid));	/* 1st arg must be scalar */

	type_oid = exprType(lsecond(opexpr->args));
	dtype2 = pgstrom_devtype_lookup(type_oid);
	if (!dtype2 || !dtype2->type_element)
		elog(ERROR, "codegen: failed to lookup device type, or scalar: %s",
			 format_type_be(type_oid));	/* 2nd arg must be array */

	/* sanity checks */
	if (dfunc->func_rettype->type_oid != BOOLOID ||
		list_length(dfunc->func_args) != 2 ||
		((devtype_info *)linitial(dfunc->func_args))->type_oid
			!= dtype1->type_oid ||
		((devtype_info *)lsecond(dfunc->func_args))->type_oid
			!= dtype2->type_element->type_oid)
		elog(ERROR, "sanity check violation at ScalarArrayOp");

	devexpr.expr_args = list_make2(dtype1, dtype2);
	devexpr.expr_rettype = pgstrom_devtype_lookup(BOOLOID);
	if (!devexpr.expr_rettype)
		elog(ERROR, "codegen: failed to lookup device type: %s",
			 format_type_be(BOOLOID));
	devexpr.expr_extra1 = ObjectIdGetDatum(opexpr->opno);
	devexpr.expr_extra2 = BoolGetDatum(opexpr->useOr);
	/* device function name */
	devexpr.expr_name = psprintf("%s_%s_array",
								 dfunc->func_sqlname,
								 opexpr->useOr ? "any" : "all");
	/* device function declaration */
	initStringInfo(&decl);
	appendStringInfo(
		&decl,
		"STATIC_INLINE(pg_bool_t)\n"
		"pgfn_%s(kern_context *kcxt, pg_%s_t scalar, pg_array_t array)\n"
		"{\n"
		"  pg_bool_t  result;\n"
		"  pg_bool_t  rv;\n"
		"  cl_int     i, nitems;\n"
		"  char      *dataptr;\n"
		"  char      *bitmap;\n"
		"  int        bitmask;\n"
		"  pg_anytype_t temp  __attribute__((unused));\n",
		devexpr.expr_name,
		dtype1->type_name);

	appendStringInfo(
		&decl,
		"\n"
		"  /* NULL result to NULL array */\n"
		"  if (array.isnull)\n"
		"  {\n"
		"    result.isnull = true;\n"
		"    result.value  = false;\n"
		"    return result;\n"
		"  }\n\n");

	if (dfunc->func_is_strict)
	{
		appendStringInfo(
			&decl,
			"  /* Quick NULL return to NULL scalar and strict function */\n"
			"  if (scalar.isnull)\n"
			"  {\n"
			"    result.isnull = true;\n"
			"    result.value  = false;\n"
			"    return result;\n"
			"  }\n");
	}

	appendStringInfo(
        &decl,
		"  /* how much items in the array? */\n"
		"  nitems = ArrayGetNItems(kcxt, ARR_NDIM(array.value),\n"
		"                                ARR_DIMS(array.value));\n"
		"  if (nitems <= 0)\n"
		"  {\n"
		"    result.isnull = false;\n"
		"    result.value  = %s;\n"
		"  }\n\n",
		opexpr->useOr ? "false" : "true");

	appendStringInfo(
		&decl,
		"  /* loop over the array elements */\n"
		"  dataptr = ARR_DATA_PTR(array.value);\n"
		"  bitmap  = ARR_NULLBITMAP(array.value);\n"
		"  bitmask = 1;\n"
		"  result.isnull = false;\n"
		"  result.value  = %s;\n"
		"\n"
		"  for (i=0; i < nitems; i++)\n"
		"  {\n"
		"    if (bitmap && (*bitmap & bitmask) == 0)\n"
		"      temp.%s_v = pg_%s_datum_ref(kcxt,NULL,false);\n"
		"    else\n"
		"    {\n"
		"      temp.%s_v = pg_%s_datum_ref(kcxt,dataptr,false);\n"
		"      dataptr += %s;\n"
		"      dataptr = (char *) TYPEALIGN(%d, dataptr);\n"
		"    }\n\n",
		opexpr->useOr ? "false" : "true",
		dtype1->type_name, dtype1->type_name,
		dtype1->type_name, dtype1->type_name,
		dtype1->type_length < 0
		? "VARSIZE_ANY(dataptr)"
		: psprintf("%d", dtype1->type_length),
		dtype1->type_align);

	appendStringInfo(
		&decl,
		"    /* call for comparison function */\n"
		"    rv = pgfn_%s(kcxt, scalar, temp.%s_v);\n"
		"    if (rv.isnull)\n"
		"      result.isnull = true;\n"
		"    else if (%srv.value)\n"
		"    {\n"
		"      result.isnull = false;\n"
		"      result.value  = %s;\n"
		"      break;\n"
		"    }\n",
		dfunc->func_devname, dtype1->type_name,
		opexpr->useOr ? "" : "!",
		opexpr->useOr ? "true" : "false");

	appendStringInfo(
		&decl,
		"    /* advance bitmap pointer if any */\n"
		"    if (bitmap)\n"
		"    {\n"
		"      bitmask <<= 1;\n"
		"      if (bitmask == 0x0100)\n"
		"      {\n"
		"        bitmap++;\n"
		"        bitmask = 1;\n"
		"      }\n"
		"    }\n"
		"  }\n"
		"  return result;\n"
		"}\n");
	devexpr.expr_decl = decl.data;

	/* remember this special device function */
	context->expr_defs = lappend(context->expr_defs,
								 form_devexpr_info(&devexpr));

found:
	/* write out this special expression */
	appendStringInfo(&context->str, "pgfn_%s(kcxt", devexpr.expr_name);
	foreach (cell, opexpr->args)
	{
		Node   *expr = lfirst(cell);

		appendStringInfo(&context->str, ", ");
		codegen_expression_walker(expr, context);
	}
	appendStringInfo(&context->str, ")");
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
	codegen_expression_walker(expr, &walker_context);

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
 * pgstrom_codegen_func_declarations
 */
void
pgstrom_codegen_func_declarations(StringInfo buf, codegen_context *context)
{
	ListCell	   *lc;

	foreach (lc, context->func_defs)
	{
		devfunc_info *dfunc = lfirst(lc);

		if (dfunc->func_decl)
			appendStringInfo(buf, "%s\n", dfunc->func_decl);
	}
}

/*
 * pgstrom_codegen_expr_declarations
 */
void
pgstrom_codegen_expr_declarations(StringInfo buf, codegen_context *context)
{
	devexpr_info	devexpr;
	ListCell	   *lc;

	foreach (lc, context->expr_defs)
	{
		deform_devexpr_info(&devexpr, (List *) lfirst(lc));

		appendStringInfo(buf, "%s\n", devexpr.expr_decl);
	}
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
pgstrom_device_expression(Expr *expr)
{
	if (expr == NULL)
		return true;
	if (IsA(expr, List))
	{
		ListCell   *cell;

		foreach (cell, (List *) expr)
		{
			if (!pgstrom_device_expression(lfirst(cell)))
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
		return pgstrom_device_expression((Expr *) func->args);
	}
	else if (IsA(expr, OpExpr) || IsA(expr, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		if (!pgstrom_devfunc_lookup(get_opcode(op->opno),
									op->opresulttype,
									op->args,
									op->inputcollid))
			goto unable_node;
		return pgstrom_device_expression((Expr *) op->args);
	}
	else if (IsA(expr, NullTest))
	{
		NullTest   *nulltest = (NullTest *) expr;

		if (nulltest->argisrow)
			goto unable_node;

		return pgstrom_device_expression((Expr *) nulltest->arg);
	}
	else if (IsA(expr, BooleanTest))
	{
		BooleanTest	   *booltest = (BooleanTest *) expr;

		return pgstrom_device_expression((Expr *) booltest->arg);
	}
	else if (IsA(expr, BoolExpr))
	{
		BoolExpr	   *boolexpr = (BoolExpr *) expr;

		Assert(boolexpr->boolop == AND_EXPR ||
			   boolexpr->boolop == OR_EXPR ||
			   boolexpr->boolop == NOT_EXPR);
		return pgstrom_device_expression((Expr *) boolexpr->args);
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
		return pgstrom_device_expression((Expr *) coalesce->args);
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
		return pgstrom_device_expression((Expr *) minmax->args);
	}
	else if (IsA(expr, RelabelType))
	{
		RelabelType	   *relabel = (RelabelType *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(relabel->resulttype);

		/* array->array relabel may be possible */
		if (!dtype)
			goto unable_node;

		return pgstrom_device_expression((Expr *) relabel->arg);
	}
	else if (IsA(expr, CaseExpr))
	{
		CaseExpr   *caseexpr = (CaseExpr *) expr;
		ListCell   *cell;

		if (!pgstrom_devtype_lookup(caseexpr->casetype))
			goto unable_node;

		if (caseexpr->arg)
		{
			if (!pgstrom_device_expression(caseexpr->arg))
				return false;
		}

		foreach (cell, caseexpr->args)
		{
			CaseWhen   *casewhen = lfirst(cell);

			Assert(IsA(casewhen, CaseWhen));
			if (exprType((Node *)casewhen->expr) !=
				(caseexpr->arg ? exprType((Node *)caseexpr->arg) : BOOLOID))
				goto unable_node;

			if (!pgstrom_device_expression(casewhen->expr))
				return false;
			if (!pgstrom_device_expression(casewhen->result))
				return false;
		}
		if (!pgstrom_device_expression((Expr *)caseexpr->defresult))
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

		if (!pgstrom_device_expression((Expr *) opexpr->args))
			return false;

		return true;
	}
unable_node:
	elog(DEBUG2, "Unable to run on device: %s", nodeToString(expr));
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
										   ALLOCSET_DEFAULT_MINSIZE,
										   ALLOCSET_DEFAULT_INITSIZE,
										   ALLOCSET_DEFAULT_MAXSIZE);
	CacheRegisterSyscacheCallback(PROCOID, codegen_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(TYPEOID, codegen_cache_invalidator, 0);
}
