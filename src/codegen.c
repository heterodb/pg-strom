/*
 * codegen.c
 *
 * Routines for CUDA code generator
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
static List	   *devcast_info_slot[48];
bool			pgstrom_enable_numeric_type;	/* GUC */
static Oid		pgstrom_float2_typeoid = InvalidOid;

static void		build_devcast_info(void);

static cl_uint generic_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_numeric_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_bpchar_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_inet_devtype_hashfunc(devtype_info *dtype, Datum datum);
static cl_uint pg_range_devtype_hashfunc(devtype_info *dtype, Datum datum);

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
				 pg_inet_devtype_hashfunc),
	DEVTYPE_DECL("cidr",   "CIDROID",   "inet_struct",
				 NULL, NULL, NULL,
				 DEVKERNEL_NEEDS_MISC, sizeof(inet),
				 pg_inet_devtype_hashfunc),
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
				 sizeof(struct NumericData),
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
													 type_flags,
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
	/* also, device types cast */
	build_devcast_info();
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
	if (dummy.e.errcode != StromError_Success)
		elog(ERROR, "failed on hash calculation of device numeric: %s",
			 DatumGetCString(DirectFunctionCall1(numeric_out, datum)));

	return hash_any((cl_uchar *)&temp.value,
					offsetof(pg_numeric_t, precision) + sizeof(cl_short));
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
pg_range_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	RangeType  *r = DatumGetRangeTypeP(datum);
	cl_uchar	flags = *((char *)r + VARSIZE(r) - 1);
	cl_uchar   *pos = (cl_uchar *)(r + 1);
	cl_uchar	buf[sizeof(Datum) * 2 + sizeof(char)];
	int			len = 0;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	get_typlenbyvalalign(r->rangetypid, &typlen, &typbyval, &typalign);
	Assert(typlen > 0);		/* we support only fixed-length */
	if (RANGE_HAS_LBOUND(flags))
	{
		memcpy(buf + len, pos, typlen);
		len += typlen;
		pos += typlen;
	}
	if (RANGE_HAS_UBOUND(flags))
	{
		memcpy(buf + len, pos, typlen);
		len += typlen;
		pos += typlen;
	}
	buf[len++] = flags;

	return hash_any(buf, len);
}

/*
 * varlena buffer estimation handler
 */
static int
vlbuf_estimate_textcat(devfunc_info *dfunc, Expr **args, int *vl_width)
{
	int		i, maxlen = 0;

	for (i=0; i < 2; i++)
	{
		Expr   *str = args[i];
	retry:
		if (IsA(str, Const))
		{
			Const  *con = (Const *) str;

			if (!con->constisnull)
				maxlen += VARSIZE_ANY_EXHDR(con);
		}
		else
		{
			int		typmod = exprTypmod((Node *) str);

			if (typmod >= VARHDRSZ)
				maxlen += (typmod - VARHDRSZ);
			else if (IsA(str, FuncExpr) ||
					 IsA(str, OpExpr) ||
					 IsA(str, DistinctExpr))
				maxlen += vl_width[i];
			else if (IsA(str, RelabelType))
			{
				str = ((RelabelType *) str)->arg;
				goto retry;
			}
			else
			{
				/*
				 * Even though table statistics tell us 'average length' of
				 * the values, we have no information about 'maximum length'
				 * or 'standard diviation'. So, it may cause CPU recheck and
				 * performance slowdown, if we try textcat on the device.
				 * To avoid the risk, simply, we prohibit the operation.
				 */
				return -1;
			}
		}
	}
	return maxlen;
}

/*
static int
vlbuf_estimate_text_substr(devfunc_info *dfunc, Expr **args, int *vl_width)
{
	return 0;
}
*/

/*
 * pgstrom_get_float2_typeoid - FLOAT2OID
 */
Oid
pgstrom_get_float2_typeoid(void)
{
	if (!OidIsValid(pgstrom_float2_typeoid))
	{
		Oid		nsp_oid = get_namespace_oid(PGSTROM_SCHEMA_NAME, true);
		Oid		type_oid = GetSysCacheOid2(TYPENAMENSP,
										   PointerGetDatum("float2"),
										   ObjectIdGetDatum(nsp_oid));
		if (!OidIsValid(type_oid))
			elog(ERROR, "float2 is not defined at PostgreSQL");
		pgstrom_float2_typeoid = type_oid;
	}
	return pgstrom_float2_typeoid;
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
	int			func_devcost;	/* relative cost to run on device */
	int		  (*func_varlena_sz)(devfunc_info *dfunc,
								 Expr **args, int *vl_width);
	const char *func_template;	/* a template string if simple function */
} devfunc_catalog_t;

static devfunc_catalog_t devfunc_common_catalog[] = {
	/* Type cast functions */
	{ "bool", 1, {INT4OID},     1, NULL, "m/f:int4_bool" },

	{ "int2", 1, {INT4OID},     1, NULL, "p/f:to_int2" },
	{ "int2", 1, {INT8OID},     1, NULL, "p/f:to_int2" },
	{ "int2", 1, {FLOAT4OID},   1, NULL, "p/f:to_int2" },
	{ "int2", 1, {FLOAT8OID},   1, NULL, "p/f:to_int2" },

	{ "int4", 1, {BOOLOID},     1, NULL, "p/f:to_int4" },
	{ "int4", 1, {INT2OID},     1, NULL, "p/f:to_int4" },
	{ "int4", 1, {INT8OID},     1, NULL, "p/f:to_int4" },
	{ "int4", 1, {FLOAT4OID},   1, NULL, "p/f:to_int4" },
	{ "int4", 1, {FLOAT8OID},   1, NULL, "p/f:to_int4" },

	{ "int8", 1, {INT2OID},     1, NULL, "p/f:to_int8" },
	{ "int8", 1, {INT4OID},     1, NULL, "p/f:to_int8" },
	{ "int8", 1, {FLOAT4OID},   1, NULL, "p/f:to_int8" },
	{ "int8", 1, {FLOAT8OID},   1, NULL, "p/f:to_int8" },

	{ "float4", 1, {INT2OID},   1, NULL, "p/f:to_float4" },
	{ "float4", 1, {INT4OID},   1, NULL, "p/f:to_float4" },
	{ "float4", 1, {INT8OID},   1, NULL, "p/f:to_float4" },
	{ "float4", 1, {FLOAT8OID}, 1, NULL, "p/f:to_float4" },

	{ "float8", 1, {INT2OID},   1, NULL, "p/f:to_float8" },
	{ "float8", 1, {INT4OID},   1, NULL, "p/f:to_float8" },
	{ "float8", 1, {INT8OID},   1, NULL, "p/f:to_float8" },
	{ "float8", 1, {FLOAT4OID}, 1, NULL, "p/f:to_float8" },

	/* '+' : add operators */
	{ "int2pl",  2, {INT2OID, INT2OID}, 1, NULL, "m/f:int2pl" },
	{ "int24pl", 2, {INT2OID, INT4OID}, 1, NULL, "m/f:int24pl" },
	{ "int28pl", 2, {INT2OID, INT8OID}, 1, NULL, "m/f:int28pl" },
	{ "int42pl", 2, {INT4OID, INT2OID}, 1, NULL, "m/f:int42pl" },
	{ "int4pl",  2, {INT4OID, INT4OID}, 1, NULL, "m/f:int4pl" },
	{ "int48pl", 2, {INT4OID, INT8OID}, 1, NULL, "m/f:int48pl" },
	{ "int82pl", 2, {INT8OID, INT2OID}, 1, NULL, "m/f:int82pl" },
	{ "int84pl", 2, {INT8OID, INT4OID}, 1, NULL, "m/f:int84pl" },
	{ "int8pl",  2, {INT8OID, INT8OID}, 1, NULL, "m/f:int8pl" },
	{ "float4pl",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "m/f:float4pl" },
	{ "float48pl", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "m/f:float48pl" },
	{ "float84pl", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "m/f:float84pl" },
	{ "float8pl",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "m/f:float8pl" },

	/* '-' : subtract operators */
	{ "int2mi",  2, {INT2OID, INT2OID}, 1, NULL, "m/f:int2mi" },
	{ "int24mi", 2, {INT2OID, INT4OID}, 1, NULL, "m/f:int24mi" },
	{ "int28mi", 2, {INT2OID, INT8OID}, 1, NULL, "m/f:int28mi" },
	{ "int42mi", 2, {INT4OID, INT2OID}, 1, NULL, "m/f:int42mi" },
	{ "int4mi",  2, {INT4OID, INT4OID}, 1, NULL, "m/f:int4mi" },
	{ "int48mi", 2, {INT4OID, INT8OID}, 1, NULL, "m/f:int48mi" },
	{ "int82mi", 2, {INT8OID, INT2OID}, 1, NULL, "m/f:int82mi" },
	{ "int84mi", 2, {INT8OID, INT4OID}, 1, NULL, "m/f:int84mi" },
	{ "int8mi",  2, {INT8OID, INT8OID}, 1, NULL, "m/f:int8mi" },
	{ "float4mi",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "m/f:float4mi" },
	{ "float48mi", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "m/f:float48mi" },
	{ "float84mi", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "m/f:float84mi" },
	{ "float8mi",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "m/f:float8mi" },

	/* '*' : mutiply operators */
	{ "int2mul",  2, {INT2OID, INT2OID}, 2, NULL, "m/f:int2mul" },
	{ "int24mul", 2, {INT2OID, INT4OID}, 2, NULL, "m/f:int24mul" },
	{ "int28mul", 2, {INT2OID, INT8OID}, 2, NULL, "m/f:int28mul" },
	{ "int42mul", 2, {INT4OID, INT2OID}, 2, NULL, "m/f:int42mul" },
	{ "int4mul",  2, {INT4OID, INT4OID}, 2, NULL, "m/f:int4mul" },
	{ "int48mul", 2, {INT4OID, INT8OID}, 2, NULL, "m/f:int48mul" },
	{ "int82mul", 2, {INT8OID, INT2OID}, 2, NULL, "m/f:int82mul" },
	{ "int84mul", 2, {INT8OID, INT4OID}, 2, NULL, "m/f:int84mul" },
	{ "int8mul",  2, {INT8OID, INT8OID}, 2, NULL, "m/f:int8mul" },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, 2, NULL, "m/f:float4mul" },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, 2, NULL, "m/f:float48mul" },
	{ "float84mul", 2, {FLOAT8OID, FLOAT4OID}, 2, NULL, "m/f:float84mul" },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, 2, NULL, "m/f:float8mul" },

	/* '/' : divide operators */
	{ "int2div",  2, {INT2OID, INT2OID}, 2, NULL, "m/f:int2div" },
	{ "int24div", 2, {INT2OID, INT4OID}, 2, NULL, "m/f:int24div" },
	{ "int28div", 2, {INT2OID, INT8OID}, 2, NULL, "m/f:int28div" },
	{ "int42div", 2, {INT4OID, INT2OID}, 2, NULL, "m/f:int42div" },
	{ "int4div",  2, {INT4OID, INT4OID}, 2, NULL, "m/f:int4div" },
	{ "int48div", 2, {INT4OID, INT8OID}, 2, NULL, "m/f:int48div" },
	{ "int82div", 2, {INT8OID, INT2OID}, 2, NULL, "m/f:int82div" },
	{ "int84div", 2, {INT8OID, INT4OID}, 2, NULL, "m/f:int84div" },
	{ "int8div",  2, {INT8OID, INT8OID}, 2, NULL, "m/f:int8div" },
	{ "float4div",  2, {FLOAT4OID, FLOAT4OID}, 2, NULL, "m/f:float4div" },
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, 2, NULL, "m/f:float48div" },
	{ "float84div", 2, {FLOAT8OID, FLOAT4OID}, 2, NULL, "m/f:float84div" },
	{ "float8div",  2, {FLOAT8OID, FLOAT8OID}, 2, NULL, "m/f:float8div" },

	/* '%' : reminder operators */
	{ "int2mod", 2, {INT2OID, INT2OID}, 2, NULL, "m/f:int2mod" },
	{ "int4mod", 2, {INT4OID, INT4OID}, 2, NULL, "m/f:int4mod" },
	{ "int8mod", 2, {INT8OID, INT8OID}, 2, NULL, "m/f:int8mod" },

	/* '+' : unary plus operators */
	{ "int2up", 1, {INT2OID},      1, NULL, "p/f:int2up" },
	{ "int4up", 1, {INT4OID},      1, NULL, "p/f:int4up" },
	{ "int8up", 1, {INT8OID},      1, NULL, "p/f:int8up" },
	{ "float4up", 1, {FLOAT4OID},  1, NULL, "p/f:float4up" },
	{ "float8up", 1, {FLOAT8OID},  1, NULL, "p/f:float8up" },

	/* '-' : unary minus operators */
	{ "int2um", 1, {INT2OID},      1, NULL, "p/f:int2um" },
	{ "int4um", 1, {INT4OID},      1, NULL, "p/f:int4um" },
	{ "int8um", 1, {INT8OID},      1, NULL, "p/f:int8um" },
	{ "float4um", 1, {FLOAT4OID},  1, NULL, "p/f:float4um" },
	{ "float8um", 1, {FLOAT8OID},  1, NULL, "p/f:float8um" },

	/* '@' : absolute value operators */
	{ "int2abs", 1, {INT2OID},     1, NULL, "p/f:int2abs" },
	{ "int4abs", 1, {INT4OID},     1, NULL, "p/f:int4abs" },
	{ "int8abs", 1, {INT8OID},     1, NULL, "p/f:int8abs" },
	{ "float4abs", 1, {FLOAT4OID}, 1, NULL, "p/f:float4abs" },
	{ "float8abs", 1, {FLOAT8OID}, 1, NULL, "p/f:float8abs" },

	/* '=' : equal operators */
	{ "booleq",  2, {BOOLOID, BOOLOID}, 1, NULL, "p/f:booleq" },
	{ "int2eq",  2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2eq" },
	{ "int24eq", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int24eq" },
	{ "int28eq", 2, {INT2OID, INT8OID}, 1, NULL, "p/f:int28eq" },
	{ "int42eq", 2, {INT4OID, INT2OID}, 1, NULL, "p/f:int42eq" },
	{ "int4eq",  2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4eq" },
	{ "int48eq", 2, {INT4OID, INT8OID}, 1, NULL, "p/f:int48eq" },
	{ "int82eq", 2, {INT8OID, INT2OID}, 1, NULL, "p/f:int82eq" },
	{ "int84eq", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int84eq" },
	{ "int8eq",  2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8eq" },
	{ "float4eq",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "p/f:float4eq" },
	{ "float48eq", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "p/f:float48eq" },
	{ "float84eq", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "p/f:float84eq" },
	{ "float8eq",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "p/f:float8eq" },

	/* '<>' : not equal operators */
	{ "int2ne",  2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2ne" },
	{ "int24ne", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int24ne" },
	{ "int28ne", 2, {INT2OID, INT8OID}, 1, NULL, "p/f:int28ne" },
	{ "int42ne", 2, {INT4OID, INT2OID}, 1, NULL, "p/f:int42ne" },
	{ "int4ne",  2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4ne" },
	{ "int48ne", 2, {INT4OID, INT8OID}, 1, NULL, "p/f:int48ne" },
	{ "int82ne", 2, {INT8OID, INT2OID}, 1, NULL, "p/f:int82ne" },
	{ "int84ne", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int84ne" },
	{ "int8ne",  2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8ne" },
	{ "float4ne",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "p/f:float4ne" },
	{ "float48ne", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "p/f:float48ne" },
	{ "float84ne", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "p/f:float84ne" },
	{ "float8ne",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "p/f:float8ne" },

	/* '>' : greater than operators */
	{ "int2gt",  2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2gt" },
	{ "int24gt", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int24gt" },
	{ "int28gt", 2, {INT2OID, INT8OID}, 1, NULL, "p/f:int28gt" },
	{ "int42gt", 2, {INT4OID, INT2OID}, 1, NULL, "p/f:int42gt" },
	{ "int4gt",  2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4gt" },
	{ "int48gt", 2, {INT4OID, INT8OID}, 1, NULL, "p/f:int48gt" },
	{ "int82gt", 2, {INT8OID, INT2OID}, 1, NULL, "p/f:int82gt" },
	{ "int84gt", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int84gt" },
	{ "int8gt",  2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8gt" },
	{ "float4gt",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "p/f:float4gt" },
	{ "float48gt", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "p/f:float48gt" },
	{ "float84gt", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "p/f:float84gt" },
	{ "float8gt",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "p/f:float8gt" },

	/* '<' : less than operators */
	{ "int2lt",  2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2lt" },
	{ "int24lt", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int24lt" },
	{ "int28lt", 2, {INT2OID, INT8OID}, 1, NULL, "p/f:int28lt" },
	{ "int42lt", 2, {INT4OID, INT2OID}, 1, NULL, "p/f:int42lt" },
	{ "int4lt",  2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4lt" },
	{ "int48lt", 2, {INT4OID, INT8OID}, 1, NULL, "p/f:int48lt" },
	{ "int82lt", 2, {INT8OID, INT2OID}, 1, NULL, "p/f:int82lt" },
	{ "int84lt", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int84lt" },
	{ "int8lt",  2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8lt" },
	{ "float4lt",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "p/f:float4lt" },
	{ "float48lt", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "p/f:float48lt" },
	{ "float84lt", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "p/f:float84lt" },
	{ "float8lt",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "p/f:float8lt" },

	/* '>=' : relational greater-than or equal-to */
	{ "int2ge",  2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2ge" },
	{ "int24ge", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int24ge" },
	{ "int28ge", 2, {INT2OID, INT8OID}, 1, NULL, "p/f:int28ge" },
	{ "int42ge", 2, {INT4OID, INT2OID}, 1, NULL, "p/f:int42ge" },
	{ "int4ge",  2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4ge" },
	{ "int48ge", 2, {INT4OID, INT8OID}, 1, NULL, "p/f:int48ge" },
	{ "int82ge", 2, {INT8OID, INT2OID}, 1, NULL, "p/f:int82ge" },
	{ "int84ge", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int84ge" },
	{ "int8ge",  2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8ge" },
	{ "float4ge",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "p/f:float4ge" },
	{ "float48ge", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "p/f:float48ge" },
	{ "float84ge", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "p/f:float84ge" },
	{ "float8ge",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "p/f:float8ge" },

	/* '<=' : relational greater-than or equal-to */
	{ "int2le",  2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2le" },
	{ "int24le", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int24le" },
	{ "int28le", 2, {INT2OID, INT8OID}, 1, NULL, "p/f:int28le" },
	{ "int42le", 2, {INT4OID, INT2OID}, 1, NULL, "p/f:int42le" },
	{ "int4le",  2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4le" },
	{ "int48le", 2, {INT4OID, INT8OID}, 1, NULL, "p/f:int48le" },
	{ "int82le", 2, {INT8OID, INT2OID}, 1, NULL, "p/f:int82le" },
	{ "int84le", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int84le" },
	{ "int8le",  2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8le" },
	{ "float4le",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "p/f:float4le" },
	{ "float48le", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "p/f:float48le" },
	{ "float84le", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "p/f:float84le" },
	{ "float8le",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "p/f:float8le" },

	/* '&' : bitwise and */
	{ "int2and", 2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2and" },
	{ "int4and", 2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4and" },
	{ "int8and", 2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8and" },

	/* '|'  : bitwise or */
	{ "int2or", 2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2or" },
	{ "int4or", 2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4or" },
	{ "int8or", 2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8or" },

	/* '#'  : bitwise xor */
	{ "int2xor", 2, {INT2OID, INT2OID}, 1, NULL, "p/f:int2xor" },
	{ "int4xor", 2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4xor" },
	{ "int8xor", 2, {INT8OID, INT8OID}, 1, NULL, "p/f:int8xor" },

	/* '~'  : bitwise not operators */
	{ "int2not", 1, {INT2OID}, 1, NULL, "p/f:int2not" },
	{ "int4not", 1, {INT4OID}, 1, NULL, "p/f:int4not" },
	{ "int8not", 1, {INT8OID}, 1, NULL, "p/f:int8not" },

	/* '>>' : right shift */
	{ "int2shr", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int2shr" },
	{ "int4shr", 2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4shr" },
	{ "int8shr", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int8shr" },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:int2shl" },
	{ "int4shl", 2, {INT4OID, INT4OID}, 1, NULL, "p/f:int4shl" },
	{ "int8shl", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:int8shl" },

	/* comparison functions */
	{ "btboolcmp",  2, {BOOLOID, BOOLOID}, 1, NULL, "p/f:type_compare" },
	{ "btint2cmp",  2, {INT2OID, INT2OID}, 1, NULL, "p/f:type_compare" },
	{ "btint24cmp", 2, {INT2OID, INT4OID}, 1, NULL, "p/f:type_compare" },
	{ "btint28cmp", 2, {INT2OID, INT8OID}, 1, NULL, "p/f:type_compare" },
	{ "btint42cmp", 2, {INT4OID, INT2OID}, 1, NULL, "p/f:type_compare" },
	{ "btint4cmp",  2, {INT4OID, INT4OID}, 1, NULL, "p/f:type_compare" },
	{ "btint48cmp", 2, {INT4OID, INT8OID}, 1, NULL, "p/f:type_compare" },
	{ "btint82cmp", 2, {INT8OID, INT2OID}, 1, NULL, "p/f:type_compare" },
	{ "btint84cmp", 2, {INT8OID, INT4OID}, 1, NULL, "p/f:type_compare" },
	{ "btint8cmp",  2, {INT8OID, INT8OID}, 1, NULL, "p/f:type_compare" },
	{ "btfloat4cmp",  2, {FLOAT4OID, FLOAT4OID}, 1, NULL, "p/f:type_compare" },
	{ "btfloat48cmp", 2, {FLOAT4OID, FLOAT8OID}, 1, NULL, "p/f:type_compare" },
	{ "btfloat84cmp", 2, {FLOAT8OID, FLOAT4OID}, 1, NULL, "p/f:type_compare" },
	{ "btfloat8cmp",  2, {FLOAT8OID, FLOAT8OID}, 1, NULL, "p/f:type_compare" },

	/* currency cast */
	{ "money",			1, {NUMERICOID},		1, NULL, "y/f:numeric_cash" },
	{ "money",			1, {INT4OID},			1, NULL, "y/f:int4_cash" },
	{ "money",			1, {INT8OID},			1, NULL, "y/f:int8_cash" },
	/* currency operators */
	{ "cash_pl",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_pl" },
	{ "cash_mi",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_mi" },
	{ "cash_div_cash",	2, {CASHOID, CASHOID},	2, NULL, "y/f:cash_div_cash" },
	{ "cash_mul_int2",	2, {CASHOID, INT2OID},	2, NULL, "y/f:cash_mul_int2" },
	{ "cash_mul_int4",	2, {CASHOID, INT4OID},	2, NULL, "y/f:cash_mul_int4" },
	{ "cash_mul_flt4",	2, {CASHOID, FLOAT4OID},2, NULL, "y/f:cash_mul_flt4" },
	{ "cash_mul_flt8",	2, {CASHOID, FLOAT8OID},2, NULL, "y/f:cash_mul_flt8" },
	{ "cash_div_int2",	2, {CASHOID, INT2OID},	2, NULL, "y/f:cash_div_int2" },
	{ "cash_div_int4",	2, {CASHOID, INT4OID},	2, NULL, "y/f:cash_div_int4" },
	{ "cash_div_flt4",	2, {CASHOID, FLOAT4OID},2, NULL, "y/f:cash_div_flt4" },
	{ "cash_div_flt8",	2, {CASHOID, FLOAT8OID},2, NULL, "y/f:cash_div_flt8" },
	{ "int2_mul_cash",	2, {INT2OID, CASHOID},	2, NULL, "y/f:int2_mul_cash" },
	{ "int4_mul_cash",	2, {INT4OID, CASHOID},	2, NULL, "y/f:int4_mul_cash" },
	{ "flt4_mul_cash",	2, {FLOAT4OID, CASHOID},2, NULL, "y/f:flt4_mul_cash" },
	{ "flt8_mul_cash",	2, {FLOAT8OID, CASHOID},2, NULL, "y/f:flt8_mul_cash" },
	/* currency comparison */
	{ "cash_cmp",		2, {CASHOID, CASHOID},	1, NULL, "y/f:type_compare" },
	{ "cash_eq",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_eq" },
	{ "cash_ne",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_ne" },
	{ "cash_lt",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_lt" },
	{ "cash_le",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_le" },
	{ "cash_gt",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_gt" },
	{ "cash_ge",		2, {CASHOID, CASHOID},	1, NULL, "y/f:cash_ge" },
	/* uuid comparison */
	{ "uuid_cmp",		2, {UUIDOID, UUIDOID},	5, NULL, "y/f:type_compare" },
	{ "uuid_eq",		2, {UUIDOID, UUIDOID},	5, NULL, "y/f:uuid_eq" },
	{ "uuid_ne",		2, {UUIDOID, UUIDOID},	5, NULL, "y/f:uuid_ne" },
	{ "uuid_lt",		2, {UUIDOID, UUIDOID},	5, NULL, "y/f:uuid_lt" },
	{ "uuid_le",		2, {UUIDOID, UUIDOID},	5, NULL, "y/f:uuid_le" },
	{ "uuid_gt",		2, {UUIDOID, UUIDOID},	5, NULL, "y/f:uuid_gt" },
	{ "uuid_ge",		2, {UUIDOID, UUIDOID},	5, NULL, "y/f:uuid_ge" },
	/* macaddr comparison */
	{ "macaddr_cmp",    2, {MACADDROID,MACADDROID},
	  5, NULL, "y/f:type_compare" },
	{ "macaddr_eq",     2, {MACADDROID,MACADDROID},
	  5, NULL, "y/f:macaddr_eq" },
	{ "macaddr_ne",     2, {MACADDROID,MACADDROID},
	  5, NULL, "y/f:macaddr_ne" },
	{ "macaddr_lt",     2, {MACADDROID,MACADDROID},
	  5, NULL, "y/f:macaddr_lt" },
	{ "macaddr_le",     2, {MACADDROID,MACADDROID},
	  5, NULL, "y/f:macaddr_le" },
	{ "macaddr_gt",     2, {MACADDROID,MACADDROID},
	  5, NULL, "y/f:macaddr_gt" },
	{ "macaddr_ge",     2, {MACADDROID,MACADDROID},
	  5, NULL, "y/f:macaddr_ge" },
	/* inet comparison */
	{ "network_cmp",    2, {INETOID,INETOID},	8, NULL, "y/f:type_compare" },
	{ "network_eq",     2, {INETOID,INETOID},	8, NULL, "y/f:network_eq" },
	{ "network_ne",     2, {INETOID,INETOID},	8, NULL, "y/f:network_ne" },
	{ "network_lt",     2, {INETOID,INETOID},	8, NULL, "y/f:network_lt" },
	{ "network_le",     2, {INETOID,INETOID},	8, NULL, "y/f:network_le" },
	{ "network_gt",     2, {INETOID,INETOID},	8, NULL, "y/f:network_gt" },
	{ "network_ge",     2, {INETOID,INETOID},	8, NULL, "y/f:network_ge" },
	{ "network_larger", 2, {INETOID,INETOID},	8, NULL, "y/f:network_larger" },
	{ "network_smaller", 2, {INETOID,INETOID},	8, NULL, "y/f:network_smaller" },
	{ "network_sub",    2, {INETOID,INETOID},	8, NULL, "y/f:network_sub" },
	{ "network_subeq",  2, {INETOID,INETOID},	8, NULL, "y/f:network_subeq" },
	{ "network_sup",    2, {INETOID,INETOID},	8, NULL, "y/f:network_sup" },
	{ "network_supeq",  2, {INETOID,INETOID},	8, NULL, "y/f:network_supeq" },
	{ "network_overlap",2, {INETOID,INETOID},	8, NULL, "y/f:network_overlap" },

	/*
     * Mathmatical functions
     */
	{ "abs", 1, {INT2OID}, 1, NULL, "p/f:int2abs" },
	{ "abs", 1, {INT4OID}, 1, NULL, "p/f:int4abs" },
	{ "abs", 1, {INT8OID}, 1, NULL, "p/f:int8abs" },
	{ "abs", 1, {FLOAT4OID}, 1, NULL, "p/f:float4abs" },
	{ "abs", 1, {FLOAT8OID}, 1, NULL, "p/f:float8abs" },
	{ "cbrt",  1, {FLOAT8OID}, 1, NULL, "m/f:cbrt" },
	{ "dcbrt", 1, {FLOAT8OID}, 1, NULL, "m/f:cbrt" },
	{ "ceil", 1, {FLOAT8OID}, 1, NULL, "m/f:ceil" },
	{ "ceiling", 1, {FLOAT8OID}, 1, NULL, "m/f:ceil" },
	{ "exp", 1, {FLOAT8OID}, 5, NULL, "m/f:exp" },
	{ "dexp", 1, {FLOAT8OID}, 5, NULL, "m/f:exp" },
	{ "floor", 1, {FLOAT8OID}, 1, NULL, "m/f:floor" },
	{ "ln", 1, {FLOAT8OID}, 5, NULL, "m/f:ln" },
	{ "dlog1", 1, {FLOAT8OID}, 5, NULL, "m/f:ln" },
	{ "log", 1, {FLOAT8OID}, 5, NULL, "m/f:log10" },
	{ "dlog10", 1, {FLOAT8OID}, 5, NULL, "m/f:log10" },
	{ "pi", 0, {}, 0, NULL, "m/f:dpi" },
	{ "power", 2, {FLOAT8OID, FLOAT8OID}, 5, NULL, "m/f:dpow" },
	{ "pow", 2, {FLOAT8OID, FLOAT8OID}, 5, NULL,"m/f:dpow" },
	{ "dpow", 2, {FLOAT8OID, FLOAT8OID}, 5, NULL,"m/f:dpow" },
	{ "round", 1, {FLOAT8OID}, 5, NULL,"m/f:round" },
	{ "dround", 1, {FLOAT8OID}, 5, NULL,"m/f:round" },
	{ "sign", 1, {FLOAT8OID}, 1, NULL, "m/f:sign" },
	{ "sqrt", 1, {FLOAT8OID}, 5, NULL,"m/f:dsqrt" },
	{ "dsqrt", 1, {FLOAT8OID}, 5, NULL,"m/f:dsqrt" },
	{ "trunc", 1, {FLOAT8OID}, 1, NULL, "m/f:trunc" },
	{ "dtrunc", 1, {FLOAT8OID}, 1, NULL, "m/f:trunc" },

	/*
     * Trigonometric function
     */
	{ "degrees", 1, {FLOAT8OID}, 5, NULL, "m/f:degrees" },
	{ "radians", 1, {FLOAT8OID}, 5, NULL, "m/f:radians" },
	{ "acos",    1, {FLOAT8OID}, 5, NULL, "m/f:acos" },
	{ "asin",    1, {FLOAT8OID}, 5, NULL, "m/f:asin" },
	{ "atan",    1, {FLOAT8OID}, 5, NULL, "m/f:atan" },
	{ "atan2",   2, {FLOAT8OID, FLOAT8OID}, 5, NULL, "m/f:atan2" },
	{ "cos",     1, {FLOAT8OID}, 5, NULL, "m/f:cos" },
	{ "cot",     1, {FLOAT8OID}, 5, NULL, "m/f:cot" },
	{ "sin",     1, {FLOAT8OID}, 5, NULL, "m/f:sin" },
	{ "tan",     1, {FLOAT8OID}, 5, NULL, "m/f:tan" },

	/*
	 * Numeric functions
	 * ------------------------- */
	/* Numeric type cast functions */
	{ "int2",    1, {NUMERICOID}, 8, NULL, "n/f:numeric_int2" },
	{ "int4",    1, {NUMERICOID}, 8, NULL, "n/f:numeric_int4" },
	{ "int8",    1, {NUMERICOID}, 8, NULL, "n/f:numeric_int8" },
	{ "float4",  1, {NUMERICOID}, 8, NULL, "n/f:numeric_float4" },
	{ "float8",  1, {NUMERICOID}, 8, NULL, "n/f:numeric_float8" },
	{ "numeric", 1, {INT2OID},    5, NULL, "n/f:int2_numeric" },
	{ "numeric", 1, {INT4OID},    5, NULL, "n/f:int4_numeric" },
	{ "numeric", 1, {INT8OID},    5, NULL, "n/f:int8_numeric" },
	{ "numeric", 1, {FLOAT4OID},  5, NULL, "n/f:float4_numeric" },
	{ "numeric", 1, {FLOAT8OID},  5, NULL, "n/f:float8_numeric" },
	/* Numeric operators */
	{ "numeric_add", 2, {NUMERICOID, NUMERICOID},
	  10, NULL, "n/f:numeric_add" },
	{ "numeric_sub", 2, {NUMERICOID, NUMERICOID},
	  10, NULL, "n/f:numeric_sub" },
	{ "numeric_mul", 2, {NUMERICOID, NUMERICOID},
	  10, NULL, "n/f:numeric_mul" },
	{ "numeric_uplus",  1, {NUMERICOID}, 10, NULL, "n/f:numeric_uplus" },
	{ "numeric_uminus", 1, {NUMERICOID}, 10, NULL, "n/f:numeric_uminus" },
	{ "numeric_abs",    1, {NUMERICOID}, 10, NULL, "n/f:numeric_abs" },
	{ "abs",            1, {NUMERICOID}, 10, NULL, "n/f:numeric_abs" },
	/* Numeric comparison */
	{ "numeric_eq", 2, {NUMERICOID, NUMERICOID},  8, NULL, "n/f:numeric_eq" },
	{ "numeric_ne", 2, {NUMERICOID, NUMERICOID},  8, NULL, "n/f:numeric_ne" },
	{ "numeric_lt", 2, {NUMERICOID, NUMERICOID},  8, NULL, "n/f:numeric_lt" },
	{ "numeric_le", 2, {NUMERICOID, NUMERICOID},  8, NULL, "n/f:numeric_le" },
	{ "numeric_gt", 2, {NUMERICOID, NUMERICOID},  8, NULL, "n/f:numeric_gt" },
	{ "numeric_ge", 2, {NUMERICOID, NUMERICOID},  8, NULL, "n/f:numeric_ge" },
	{ "numeric_cmp", 2, {NUMERICOID, NUMERICOID}, 8, NULL, "n/f:type_compare" },

	/*
	 * Date and time functions
	 * ------------------------------- */
	/* Type cast functions */
	{ "date", 1, {TIMESTAMPOID},     1, NULL, "t/f:timestamp_date" },
	{ "date", 1, {TIMESTAMPTZOID},   1, NULL, "t/f:timestamptz_date" },
	{ "time", 1, {TIMETZOID},        1, NULL, "t/f:timetz_time" },
	{ "time", 1, {TIMESTAMPOID},     1, NULL, "t/f:timestamp_time" },
	{ "time", 1, {TIMESTAMPTZOID},   1, NULL, "t/f:timestamptz_time" },
	{ "timetz", 1, {TIMEOID},        1, NULL, "t/f:time_timetz" },
	{ "timetz", 1, {TIMESTAMPTZOID}, 1, NULL, "t/f:timestamptz_timetz" },
#ifdef NOT_USED
	{ "timetz", 2, {TIMETZOID, INT4OID}, 1, NULL, "t/f:timetz_scale" },
#endif
	{ "timestamp", 1, {DATEOID},        1, NULL, "t/f:date_timestamp" },
	{ "timestamp", 1, {TIMESTAMPTZOID}, 1, NULL, "t/f:timestamptz_timestamp" },
	{ "timestamptz", 1, {DATEOID},      1, NULL, "t/f:date_timestamptz" },
	{ "timestamptz", 1, {TIMESTAMPOID}, 1, NULL, "t/f:timestamp_timestamptz" },
	/* timedata operators */
	{ "date_pli", 2, {DATEOID, INT4OID}, 1, NULL, "t/f:date_pli" },
	{ "date_mii", 2, {DATEOID, INT4OID}, 1, NULL, "t/f:date_mii" },
	{ "date_mi", 2, {DATEOID, DATEOID},  1, NULL, "t/f:date_mi" },
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, 2, NULL, "t/f:datetime_pl" },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, 2, NULL, "t/f:integer_pl_date" },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, 2, NULL, "t/f:timedate_pl" },
	/* time - time => interval */
	{ "time_mi_time", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:time_mi_time" },
	/* timestamp - timestamp => interval */
	{ "timestamp_mi", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  4, NULL, "t/f:timestamp_mi" },
	/* timetz +/- interval => timetz */
	{ "timetz_pl_interval", 2, {TIMETZOID, INTERVALOID},
	  4, NULL, "t/f:timetz_pl_interval" },
	{ "timetz_mi_interval", 2, {TIMETZOID, INTERVALOID},
	  4, NULL, "t/f:timetz_mi_interval" },
	/* timestamptz +/- interval => timestamptz */
	{ "timestamptz_pl_interval", 2, {TIMESTAMPTZOID, INTERVALOID},
	  4, NULL, "t/f:timestamptz_pl_interval" },
	{ "timestamptz_mi_interval", 2, {TIMESTAMPTZOID, INTERVALOID},
	  4, NULL, "t/f:timestamptz_mi_interval" },
	/* interval operators */
	{ "interval_um", 1, {INTERVALOID}, 4, NULL, "t/f:interval_um" },
	{ "interval_pl", 2, {INTERVALOID, INTERVALOID},
	  4, NULL, "t/f:interval_pl" },
	{ "interval_mi", 2, {INTERVALOID, INTERVALOID},
	  4, NULL, "t/f:interval_mi" },
	/* date + timetz => timestamptz */
	{ "datetimetz_pl", 2, {DATEOID, TIMETZOID},
	  4, NULL, "t/f:datetimetz_timestamptz" },
	{ "timestamptz", 2, {DATEOID, TIMETZOID},
	  4, NULL, "t/f:datetimetz_timestamptz" },
	/* comparison between date */
	{ "date_eq", 2, {DATEOID, DATEOID}, 2, NULL, "t/f:date_eq" },
	{ "date_ne", 2, {DATEOID, DATEOID}, 2, NULL, "t/f:date_ne" },
	{ "date_lt", 2, {DATEOID, DATEOID}, 2, NULL, "t/f:date_lt"  },
	{ "date_le", 2, {DATEOID, DATEOID}, 2, NULL, "t/f:date_le" },
	{ "date_gt", 2, {DATEOID, DATEOID}, 2, NULL, "t/f:date_gt"  },
	{ "date_ge", 2, {DATEOID, DATEOID}, 2, NULL, "t/f:date_ge" },
	{ "date_cmp", 2, {DATEOID, DATEOID}, 2, NULL, "t/f:type_compare" },
	/* comparison of date and timestamp */
	{ "date_eq_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  2, NULL, "t/f:date_eq_timestamp" },
	{ "date_ne_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  2, NULL, "t/f:date_ne_timestamp" },
	{ "date_lt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  2, NULL, "t/f:date_lt_timestamp" },
	{ "date_le_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  2, NULL, "t/f:date_le_timestamp" },
	{ "date_gt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  2, NULL, "t/f:date_gt_timestamp" },
	{ "date_ge_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  2, NULL, "t/f:date_ge_timestamp" },
	{ "date_cmp_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  2, NULL, "t/f:date_cmp_timestamp" },
	/* comparison between time */
	{ "time_eq", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:time_eq" },
	{ "time_ne", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:time_ne" },
	{ "time_lt", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:time_lt"  },
	{ "time_le", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:time_le" },
	{ "time_gt", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:time_gt"  },
	{ "time_ge", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:time_ge" },
	{ "time_cmp", 2, {TIMEOID, TIMEOID}, 2, NULL, "t/f:type_compare" },
	/* comparison between timetz */
	{ "timetz_eq", 2, {TIMETZOID, TIMETZOID}, 1, NULL, "t/f:timetz_eq" },
	{ "timetz_ne", 2, {TIMETZOID, TIMETZOID}, 1, NULL, "t/f:timetz_ne" },
	{ "timetz_lt", 2, {TIMETZOID, TIMETZOID}, 1, NULL, "t/f:timetz_lt" },
	{ "timetz_le", 2, {TIMETZOID, TIMETZOID}, 1, NULL, "t/f:timetz_le" },
	{ "timetz_ge", 2, {TIMETZOID, TIMETZOID}, 1, NULL, "t/f:timetz_ge" },
	{ "timetz_gt", 2, {TIMETZOID, TIMETZOID}, 1, NULL, "t/f:timetz_gt" },
	{ "timetz_cmp", 2, {TIMETZOID, TIMETZOID}, 1, NULL, "t/f:timetz_cmp" },
	/* comparison between timestamp */
	{ "timestamp_eq", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  1, NULL, "t/f:timestamp_eq" },
	{ "timestamp_ne", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  1, NULL, "t/f:timestamp_ne" },
	{ "timestamp_lt", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  1, NULL, "t/f:timestamp_lt"  },
	{ "timestamp_le", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  1, NULL, "t/f:timestamp_le" },
	{ "timestamp_gt", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  1, NULL, "t/f:timestamp_gt"  },
	{ "timestamp_ge", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  1, NULL, "t/f:timestamp_ge" },
	{ "timestamp_cmp", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  1, NULL, "t/f:timestamp_cmp" },
	/* comparison of timestamp and date */
	{ "timestamp_eq_date", 2, {TIMESTAMPOID, DATEOID},
	  3, NULL, "t/f:timestamp_eq_date" },
	{ "timestamp_ne_date", 2, {TIMESTAMPOID, DATEOID},
	  3, NULL, "t/f:timestamp_ne_date" },
	{ "timestamp_lt_date", 2, {TIMESTAMPOID, DATEOID},
	  3, NULL, "t/f:timestamp_lt_date" },
	{ "timestamp_le_date", 2, {TIMESTAMPOID, DATEOID},
	  3, NULL, "t/f:timestamp_le_date" },
	{ "timestamp_gt_date", 2, {TIMESTAMPOID, DATEOID},
	  3, NULL, "t/f:timestamp_gt_date" },
	{ "timestamp_ge_date", 2, {TIMESTAMPOID, DATEOID},
	  3, NULL, "t/f:timestamp_ge_date" },
	{ "timestamp_cmp_date", 2, {TIMESTAMPOID, DATEOID},
	  3, NULL, "t/f:timestamp_cmp_date"},
	/* comparison between timestamptz */
	{ "timestamptz_eq", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, NULL, "t/f:timestamptz_eq" },
	{ "timestamptz_ne", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, NULL, "t/f:timestamptz_ne" },
	{ "timestamptz_lt", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, NULL, "t/f:timestamptz_lt" },
	{ "timestamptz_le", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, NULL, "t/f:timestamptz_le" },
	{ "timestamptz_gt", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, NULL, "t/f:timestamptz_gt" },
	{ "timestamptz_ge", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, NULL, "t/f:timestamptz_ge" },
	{ "timestamptz_cmp", 2, {TIMESTAMPTZOID, TIMESTAMPTZOID},
	  1, NULL, "t/f:type_compare" },

	/* comparison between date and timestamptz */
	{ "date_lt_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  3, NULL, "t/f:date_lt_timestamptz" },
	{ "date_le_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  3, NULL, "t/f:date_le_timestamptz" },
	{ "date_eq_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  3, NULL, "t/f:date_eq_timestamptz" },
	{ "date_ge_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  3, NULL, "t/f:date_ge_timestamptz" },
	{ "date_gt_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  3, NULL, "t/f:date_gt_timestamptz" },
	{ "date_ne_timestamptz", 2, {DATEOID, TIMESTAMPTZOID},
	  3, NULL, "t/f:date_ne_timestamptz" },

	/* comparison between timestamptz and date */
	{ "timestamptz_lt_date", 2, {TIMESTAMPTZOID, DATEOID},
	  3, NULL, "t/f:timestamptz_lt_date" },
	{ "timestamptz_le_date", 2, {TIMESTAMPTZOID, DATEOID},
	  3, NULL, "t/f:timestamptz_le_date" },
	{ "timestamptz_eq_date", 2, {TIMESTAMPTZOID, DATEOID},
	  3, NULL, "t/f:timestamptz_eq_date" },
	{ "timestamptz_ge_date", 2, {TIMESTAMPTZOID, DATEOID},
	  3, NULL, "t/f:timestamptz_ge_date" },
	{ "timestamptz_gt_date", 2, {TIMESTAMPTZOID, DATEOID},
	  3, NULL, "t/f:timestamptz_gt_date" },
	{ "timestamptz_ne_date", 2, {TIMESTAMPTZOID, DATEOID},
	  3, NULL, "t/f:timestamptz_ne_date" },

	/* comparison between timestamp and timestamptz  */
	{ "timestamp_lt_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, NULL, "t/f:timestamp_lt_timestamptz" },
	{ "timestamp_le_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, NULL, "t/f:timestamp_le_timestamptz" },
	{ "timestamp_eq_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, NULL, "t/f:timestamp_eq_timestamptz" },
	{ "timestamp_ge_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, NULL, "t/f:timestamp_ge_timestamptz" },
	{ "timestamp_gt_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, NULL, "t/f:timestamp_gt_timestamptz" },
	{ "timestamp_ne_timestamptz", 2, {TIMESTAMPOID, TIMESTAMPTZOID},
	  2, NULL, "t/f:timestamp_ne_timestamptz" },

	/* comparison between timestamptz and timestamp  */
	{ "timestamptz_lt_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, NULL, "t/f:timestamptz_lt_timestamp" },
	{ "timestamptz_le_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, NULL, "t/f:timestamptz_le_timestamp" },
	{ "timestamptz_eq_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, NULL, "t/f:timestamptz_eq_timestamp" },
	{ "timestamptz_ge_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, NULL, "t/f:timestamptz_ge_timestamp" },
	{ "timestamptz_gt_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, NULL, "t/f:timestamptz_gt_timestamp" },
	{ "timestamptz_ne_timestamp", 2, {TIMESTAMPTZOID, TIMESTAMPOID},
	  2, NULL, "t/f:timestamptz_ne_timestamp" },

	/* comparison between intervals */
	{ "interval_eq", 2, {INTERVALOID, INTERVALOID},
	  2, NULL, "t/f:interval_eq" },
	{ "interval_ne", 2, {INTERVALOID, INTERVALOID},
	  2, NULL, "t/f:interval_ne" },
	{ "interval_lt", 2, {INTERVALOID, INTERVALOID},
	  2, NULL, "t/f:interval_lt" },
	{ "interval_le", 2, {INTERVALOID, INTERVALOID},
	  2, NULL, "t/f:interval_le" },
	{ "interval_ge", 2, {INTERVALOID, INTERVALOID},
	  2, NULL, "t/f:interval_ge" },
	{ "interval_gt", 2, {INTERVALOID, INTERVALOID},
	  2, NULL, "t/f:interval_gt" },
	{ "interval_cmp", 2, {INTERVALOID, INTERVALOID},
	  2, NULL, "t/f:interval_cmp" },

	/* overlaps() */
	{ "overlaps", 4, {TIMEOID, TIMEOID, TIMEOID, TIMEOID},
	  20, NULL, "t/f:overlaps_time" },
	{ "overlaps", 4, {TIMETZOID, TIMETZOID, TIMETZOID, TIMETZOID},
	  20, NULL, "t/f:overlaps_timetz" },
	{ "overlaps", 4, {TIMESTAMPOID, TIMESTAMPOID,
					  TIMESTAMPOID, TIMESTAMPOID},
	  20, NULL, "t/f:overlaps_timestamp" },
	{ "overlaps", 4, {TIMESTAMPTZOID, TIMESTAMPTZOID,
					  TIMESTAMPTZOID, TIMESTAMPTZOID},
	  20, NULL, "t/f:overlaps_timestamptz" },

	/* extract() */
	{ "date_part", 2, {TEXTOID,TIMESTAMPOID},
	  100, NULL, "stE/f:extract_timestamp"},
	{ "date_part", 2, {TEXTOID,TIMESTAMPTZOID},
	  100, NULL, "stE/f:extract_timestamptz"},
	{ "date_part", 2, {TEXTOID,INTERVALOID},
	  100, NULL, "stE/f:extract_interval"},
	{ "date_part", 2, {TEXTOID,TIMETZOID},
      100, NULL, "stE/f:extract_timetz"},
	{ "date_part", 2, {TEXTOID,TIMEOID},
	  100, NULL, "stE/f:extract_time"},

	/* other time and data functions */
	{ "now", 0, {}, 1, NULL, "t/f:now" },

	/* macaddr functions */
	{ "trunc",       1, {MACADDROID},		8, NULL, "y/f:macaddr_trunc" },
	{ "macaddr_not", 1, {MACADDROID},		8, NULL, "y/f:macaddr_not" },
	{ "macaddr_and", 2, {MACADDROID,MACADDROID}, 8, NULL, "y/f:macaddr_and" },
	{ "macaddr_or",  2, {MACADDROID,MACADDROID}, 8, NULL, "y/f:macaddr_or" },

	/* inet/cidr functions */
	{ "set_masklen", 2, {INETOID,INT4OID},	8, NULL, "y/f:inet_set_masklen" },
	{ "set_masklen", 2, {CIDROID,INT4OID},	8, NULL, "y/f:cidr_set_masklen" },
	{ "family",      1, {INETOID},			8, NULL, "y/f:inet_family" },
	{ "network",     1, {INETOID},			8, NULL, "y/f:network_network" },
	{ "netmask",     1, {INETOID},			8, NULL, "y/f:inet_netmask" },
	{ "masklen",     1, {INETOID},			8, NULL, "y/f:inet_masklen" },
	{ "broadcast",   1, {INETOID},			8, NULL, "y/f:inet_broadcast" },
	{ "hostmask",    1, {INETOID},			8, NULL, "y/f:inet_hostmask" },
	{ "cidr",        1, {INETOID},			8, NULL, "y/f:inet_to_cidr" },
	{ "inetnot",     1, {INETOID},			8, NULL, "y/f:inet_not" },
	{ "inetand",     2, {INETOID,INETOID},	8, NULL, "y/f:inet_and" },
	{ "inetor",      2, {INETOID,INETOID},	8, NULL, "y/f:inet_or" },
	{ "inetpl",      2, {INETOID,INT8OID},	8, NULL, "y/f:inetpl_int8" },
	{ "inetmi_int8", 2, {INETOID,INT8OID},	8, NULL, "y/f:inetmi_int8" },
	{ "inetmi",      2, {INETOID,INETOID},	8, NULL, "y/f:inetmi" },
	{ "inet_same_family", 2, {INETOID,INETOID}, 8, NULL, "y/f:inet_same_family" },
	{ "inet_merge",  2, {INETOID,INETOID},	8, NULL, "y/f:inet_merge" },

	/*
	 * Text functions
	 */
	{ "bpchareq",  2, {BPCHAROID,BPCHAROID},  200, NULL, "s/f:bpchareq" },
	{ "bpcharne",  2, {BPCHAROID,BPCHAROID},  200, NULL, "s/f:bpcharne" },
	{ "bpcharlt",  2, {BPCHAROID,BPCHAROID},  200, NULL, "sc/f:bpcharlt" },
	{ "bpcharle",  2, {BPCHAROID,BPCHAROID},  200, NULL, "sc/f:bpcharle" },
	{ "bpchargt",  2, {BPCHAROID,BPCHAROID},  200, NULL, "sc/f:bpchargt" },
	{ "bpcharge",  2, {BPCHAROID,BPCHAROID},  200, NULL, "sc/f:bpcharge" },
	{ "bpcharcmp", 2, {BPCHAROID, BPCHAROID}, 200, NULL, "sc/f:type_compare"},
	{ "length",    1, {BPCHAROID},            2, NULL, "sc/f:bpcharlen"},
	{ "texteq",    2, {TEXTOID, TEXTOID},     200, NULL, "s/f:texteq" },
	{ "textne",    2, {TEXTOID, TEXTOID},     200, NULL, "s/f:textne" },
	{ "text_lt",   2, {TEXTOID, TEXTOID},     200, NULL, "sc/f:text_lt" },
	{ "text_le",   2, {TEXTOID, TEXTOID},     200, NULL, "sc/f:text_le" },
	{ "text_gt",   2, {TEXTOID, TEXTOID},     200, NULL, "sc/f:text_gt" },
	{ "text_ge",   2, {TEXTOID, TEXTOID},     200, NULL, "sc/f:text_ge" },
	{ "bttextcmp", 2, {TEXTOID, TEXTOID},     200, NULL, "sc/f:type_compare" },
	/* LIKE operators */
	{ "like",        2, {TEXTOID, TEXTOID},   9999, NULL, "s/f:textlike" },
	{ "textlike",    2, {TEXTOID, TEXTOID},   9999, NULL, "s/f:textlike" },
	{ "bpcharlike",  2, {BPCHAROID, TEXTOID}, 9999, NULL, "s/f:textlike" },
	{ "notlike",     2, {TEXTOID, TEXTOID},   9999, NULL, "s/f:textnlike" },
	{ "textnlike",   2, {TEXTOID, TEXTOID},   9999, NULL, "s/f:textnlike" },
	{ "bpcharnlike", 2, {BPCHAROID, TEXTOID}, 9999, NULL, "s/f:textnlike" },
	/* ILIKE operators */
	{ "texticlike",    2, {TEXTOID, TEXTOID}, 9999, NULL, "sc/f:texticlike" },
	{ "bpchariclike",  2, {TEXTOID, TEXTOID}, 9999, NULL, "sc/f:texticlike" },
	{ "texticnlike",   2, {TEXTOID, TEXTOID}, 9999, NULL, "sc/f:texticnlike" },
	{ "bpcharicnlike", 2, {BPCHAROID, TEXTOID},9999,NULL, "sc/f:texticnlike" },
	/* string operations */
	{ "length",		1, {TEXTOID},            2, NULL, "sc/f:textlen" },
	{ "textcat",	2, {TEXTOID,TEXTOID},
	  999, vlbuf_estimate_textcat, "s/f:textcat" },
//	{ "substring",	3, {TEXTOID,INT4OID,INT4OID},
//	  999, vlbuf_estimate_text_substr, "sc/f:text_substr" },
};

/*
 * device function catalog for extra SQL functions
 */
typedef struct devfunc_extra_catalog_t {
	const char *func_rettype;
	const char *func_signature;
	int			func_devcost;
	int		  (*func_varlena_sz)(devfunc_info *dfunc,
								 Expr **args, int *vl_width);
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
	{ FLOAT4,  "pgstrom.float4("FLOAT2")",  2, NULL, "p/f:to_float4" },
	{ FLOAT8,  "pgstrom.float8("FLOAT2")",  2, NULL, "p/f:to_float8" },
	{ INT2,    "pgstrom.int2("FLOAT2")",    2, NULL, "p/f:to_int2" },
	{ INT4,    "pgstrom.int4("FLOAT2")",    2, NULL, "p/f:to_int4" },
	{ INT8,    "pgstrom.int8("FLOAT2")",    2, NULL, "p/f:to_int8" },
	{ NUMERIC, "pgstrom.numeric("FLOAT2")", 2, NULL, "n/f:float2_numeric" },
	{ FLOAT2,  "pgstrom.float2("FLOAT4")",  2, NULL, "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("FLOAT8")",  2, NULL, "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT2")",    2, NULL, "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT4")",    2, NULL, "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("INT8")",    2, NULL, "p/f:to_float2" },
	{ FLOAT2,  "pgstrom.float2("NUMERIC")", 2, NULL, "n/f:numeric_float2" },
	/* float2 - type comparison functions */
	{ BOOL,    "pgstrom.float2_eq("FLOAT2","FLOAT2")",
	  2, NULL, "p/f:float2eq" },
	{ BOOL,    "pgstrom.float2_ne("FLOAT2","FLOAT2")",
	  2, NULL, "p/f:float2ne" },
	{ BOOL,    "pgstrom.float2_lt("FLOAT2","FLOAT2")",
	  2, NULL, "p/f:float2lt" },
	{ BOOL,    "pgstrom.float2_le("FLOAT2","FLOAT2")",
	  2, NULL, "p/f:float2le" },
	{ BOOL,    "pgstrom.float2_gt("FLOAT2","FLOAT2")",
	  2, NULL, "p/f:float2gt" },
	{ BOOL,    "pgstrom.float2_ge("FLOAT2","FLOAT2")",
	  2, NULL, "p/f:float2ge" },
	{ BOOL,    "pgstrom.float2_cmp("FLOAT2","FLOAT2")",
	  2, NULL, "p/f:type_compare" },

	{ BOOL,    "pgstrom.float42_eq("FLOAT4","FLOAT2")",
	  2, NULL, "p/f:float42eq" },
	{ BOOL,    "pgstrom.float42_ne("FLOAT4","FLOAT2")",
	  2, NULL, "p/f:float42ne" },
	{ BOOL,    "pgstrom.float42_lt("FLOAT4","FLOAT2")",
	  2, NULL, "p/f:float42lt" },
	{ BOOL,    "pgstrom.float42_le("FLOAT4","FLOAT2")",
	  2, NULL, "p/f:float42le" },
	{ BOOL,    "pgstrom.float42_gt("FLOAT4","FLOAT2")",
	  2, NULL, "p/f:float42gt" },
	{ BOOL,    "pgstrom.float42_ge("FLOAT4","FLOAT2")",
	  2, NULL, "p/f:float42ge" },
	{ BOOL,    "pgstrom.float42_cmp("FLOAT4","FLOAT2")",
	  2, NULL, "p/f:type_compare" },

	{ BOOL,    "pgstrom.float82_eq("FLOAT8","FLOAT2")",
	  2, NULL, "p/f:float82eq" },
	{ BOOL,    "pgstrom.float82_ne("FLOAT8","FLOAT2")",
	  2, NULL, "p/f:float82ne" },
	{ BOOL,    "pgstrom.float82_lt("FLOAT8","FLOAT2")",
	  2, NULL, "p/f:float82lt" },
	{ BOOL,    "pgstrom.float82_le("FLOAT8","FLOAT2")",
	  2, NULL, "p/f:float82le" },
	{ BOOL,    "pgstrom.float82_gt("FLOAT8","FLOAT2")",
	  2, NULL, "p/f:float82gt" },
	{ BOOL,    "pgstrom.float82_ge("FLOAT8","FLOAT2")",
	  2, NULL, "p/f:float82ge" },
	{ BOOL,    "pgstrom.float82_cmp("FLOAT8","FLOAT2")",
	  2, NULL, "p/f:type_compare" },

	{ BOOL,    "pgstrom.float24_eq("FLOAT2","FLOAT4")",
	  2, NULL, "p/f:float24eq" },
	{ BOOL,    "pgstrom.float24_ne("FLOAT2","FLOAT4")",
	  2, NULL, "p/f:float24ne" },
	{ BOOL,    "pgstrom.float24_lt("FLOAT2","FLOAT4")",
	  2, NULL, "p/f:float24lt" },
	{ BOOL,    "pgstrom.float24_le("FLOAT2","FLOAT4")",
	  2, NULL, "p/f:float24le" },
	{ BOOL,    "pgstrom.float24_gt("FLOAT2","FLOAT4")",
	  2, NULL, "p/f:float24gt" },
	{ BOOL,    "pgstrom.float24_ge("FLOAT2","FLOAT4")",
	  2, NULL, "p/f:float24ge" },
	{ BOOL,    "pgstrom.float24_cmp("FLOAT2","FLOAT4")",
	  2, NULL, "p/f:type_compare" },

	{ BOOL,    "pgstrom.float28_eq("FLOAT2","FLOAT8")",
	  2, NULL, "p/f:float28eq" },
	{ BOOL,    "pgstrom.float28_ne("FLOAT2","FLOAT8")",
	  2, NULL, "p/f:float28ne" },
	{ BOOL,    "pgstrom.float28_lt("FLOAT2","FLOAT8")",
	  2, NULL, "p/f:float28lt" },
	{ BOOL,    "pgstrom.float28_le("FLOAT2","FLOAT8")",
	  2, NULL, "p/f:float28le" },
	{ BOOL,    "pgstrom.float28_gt("FLOAT2","FLOAT8")",
	  2, NULL, "p/f:float28gt" },
	{ BOOL,    "pgstrom.float28_ge("FLOAT2","FLOAT8")",
	  2, NULL, "p/f:float28ge" },
	{ BOOL,    "pgstrom.float28_cmp("FLOAT2","FLOAT8")",
	  2, NULL, "p/f:type_compare" },

	/* float2 - unary operator */
	{ FLOAT2,  "pgstrom.float2_up("FLOAT2")",	2, NULL, "p/f:float2up" },
	{ FLOAT2,  "pgstrom.float2_um("FLOAT2")",	2, NULL, "p/f:float2um" },
	{ FLOAT2,  "abs("FLOAT2")",					2, NULL, "p/f:float2abs" },

	/* float2 - arithmetic operators */
	{ FLOAT4, "pgstrom.float2_pl("FLOAT2","FLOAT2")",
	  2, NULL, "m/f:float2pl" },
	{ FLOAT4, "pgstrom.float2_mi("FLOAT2","FLOAT2")",
	  2, NULL, "m/f:float2mi" },
	{ FLOAT4, "pgstrom.float2_mul("FLOAT2","FLOAT2")",
	  3, NULL, "m/f:float2mul" },
	{ FLOAT4, "pgstrom.float2_div("FLOAT2","FLOAT2")",
	  3, NULL, "m/f:float2div" },
	{ FLOAT4, "pgstrom.float24_pl("FLOAT2","FLOAT4")",
	  2, NULL, "m/f:float24pl" },
	{ FLOAT4, "pgstrom.float24_mi("FLOAT2","FLOAT4")",
	  2, NULL, "m/f:float24mi" },
	{ FLOAT4, "pgstrom.float24_mul("FLOAT2","FLOAT4")",
	  3, NULL, "m/f:float24mul" },
	{ FLOAT4, "pgstrom.float24_div("FLOAT2","FLOAT4")",
	  3, NULL, "m/f:float24div" },
	{ FLOAT8, "pgstrom.float28_pl("FLOAT2","FLOAT8")",
	  2, NULL, "m/f:float28pl" },
	{ FLOAT8, "pgstrom.float28_mi("FLOAT2","FLOAT8")",
	  2, NULL, "m/f:float28mi" },
	{ FLOAT8, "pgstrom.float28_mul("FLOAT2","FLOAT8")",
	  3, NULL, "m/f:float28mul" },
	{ FLOAT8, "pgstrom.float28_div("FLOAT2","FLOAT8")",
	  3, NULL, "m/f:float28div" },
	{ FLOAT4, "pgstrom.float42_pl("FLOAT4","FLOAT2")",
	  2, NULL, "m/f:float42pl" },
	{ FLOAT4, "pgstrom.float42_mi("FLOAT4","FLOAT2")",
	  2, NULL, "m/f:float42mi" },
	{ FLOAT4, "pgstrom.float42_mul("FLOAT4","FLOAT2")",
	  3, NULL, "m/f:float42mul" },
	{ FLOAT4, "pgstrom.float42_div("FLOAT4","FLOAT2")",
	  3, NULL, "m/f:float42div" },
	{ FLOAT8, "pgstrom.float82_pl("FLOAT8","FLOAT2")",
	  2, NULL, "m/f:float82pl" },
	{ FLOAT8, "pgstrom.float82_mi("FLOAT8","FLOAT2")",
	  2, NULL, "m/f:float82mi" },
	{ FLOAT8, "pgstrom.float82_mul("FLOAT8","FLOAT2")",
	  3, NULL, "m/f:float82mul" },
	{ FLOAT8, "pgstrom.float82_div("FLOAT8","FLOAT2")",
	  3, NULL, "m/f:float82div" },
	{ "money", "pgstrom.cash_mul_flt2(money,"FLOAT2")",
	  3, NULL, "y/f:cash_mul_flt2" },
	{ "money", "pgstrom.flt2_mul_cash("FLOAT2",money)",
	  3, NULL, "y/f:flt2_mul_cash" },
	{ "money", "pgstrom.cash_div_flt2(money,"FLOAT2")",
	  3, NULL, "y/f:cash_div_flt2" },

	/* int4range operators */
	{ INT4, "lower(int4range)",		2, NULL, "r/f:int4range_lower" },
	{ INT4, "upper(int4range)",		2, NULL, "r/f:int4range_upper" },
	{ BOOL, "isempty(int4range)",	1, NULL, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(int4range)",	1, NULL, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int4range)",	1, NULL, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int4range)",	1, NULL, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int4range)",	1, NULL, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(int4range,int4range)",
	  2, NULL, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(int4range,int4range)",
	  2, NULL, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(int4range,int4range)",
	  2, NULL, "r/f:generic_range_lt" },
	{ BOOL, "range_le(int4range,int4range)",
	  2, NULL, "r/f:generic_range_le" },
	{ BOOL, "range_gt(int4range,int4range)",
	  2, NULL, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(int4range,int4range)",
	  2, NULL, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(int4range,int4range)",
	  2, NULL, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(int4range,int4range)",
	  4, NULL, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int4range,"INT4")",
	  4, NULL, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(int4range,int4range)",
	  4, NULL, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT4",int4range)",
	  4, NULL, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int4range,int4range)",
	  4, NULL, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int4range,int4range)",
	  4, NULL, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(int4range,int4range)",
	  4, NULL, "r/f:generic_range_before" },
	{ BOOL, "range_after(int4range,int4range)",
	  4, NULL, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(int4range,int4range)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(int4range,int4range)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ "int4range", "range_union(int4range,int4range)",
	  4, NULL, "r/f:generic_range_union" },
	{ "int4range", "range_merge(int4range,int4range)",
	  4, NULL, "r/f:generic_range_merge" },
	{ "int4range", "range_intersect(int4range,int4range)",
	  4, NULL, "r/f:generic_range_intersect" },
	{ "int4range", "range_minus(int4range,int4range)",
	  4, NULL, "r/f:generic_range_minus" },

	/* int8range operators */
	{ INT8, "lower(int8range)",		2, NULL, "r/f:int8range_lower" },
	{ INT8, "upper(int8range)",		2, NULL, "r/f:int8range_upper" },
	{ BOOL, "isempty(int8range)",	1, NULL, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(int8range)",	1, NULL, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(int8range)",	1, NULL, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(int8range)",	1, NULL, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(int8range)",	1, NULL, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(int8range,int8range)",
	  2, NULL, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(int8range,int8range)",
	  2, NULL, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(int8range,int8range)",
	  2, NULL, "r/f:generic_range_lt" },
	{ BOOL, "range_le(int8range,int8range)",
	  2, NULL, "r/f:generic_range_le" },
	{ BOOL, "range_gt(int8range,int8range)",
	  2, NULL, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(int8range,int8range)",
	  2, NULL, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(int8range,int8range)",
	  2, NULL, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(int8range,int8range)",
	  4, NULL, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(int8range,"INT8")",
	  4, NULL, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(int8range,int8range)",
	  4, NULL, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range("INT8",int8range)",
	  4, NULL, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(int8range,int8range)",
	  4, NULL, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(int8range,int8range)",
	  4, NULL, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(int8range,int8range)",
	  4, NULL, "r/f:generic_range_before" },
	{ BOOL, "range_after(int8range,int8range)",
	  4, NULL, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(int8range,int8range)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(int8range,int8range)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ "int8range", "range_union(int8range,int8range)",
	  4, NULL, "r/f:generic_range_union" },
	{ "int8range", "range_merge(int8range,int8range)",
	  4, NULL, "r/f:generic_range_merge" },
	{ "int8range", "range_intersect(int8range,int8range)",
	  4, NULL, "r/f:generic_range_intersect" },
	{ "int8range", "range_minus(int8range,int8range)",
	  4, NULL, "r/f:generic_range_minus" },

	/* tsrange operators */
	{ "timestamp", "lower(tsrange)",	2, NULL, "r/f:tsrange_lower" },
	{ "timestamp", "upper(tsrange)",	2, NULL, "r/f:tsrange_upper" },
	{ BOOL, "isempty(tsrange)",		1, NULL, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(tsrange)",	1, NULL, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tsrange)",	1, NULL, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tsrange)",	1, NULL, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tsrange)",	1, NULL, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(tsrange,tsrange)",  2, NULL, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(tsrange,tsrange)",  2, NULL, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(tsrange,tsrange)",  2, NULL, "r/f:generic_range_lt" },
	{ BOOL, "range_le(tsrange,tsrange)",  2, NULL, "r/f:generic_range_le" },
	{ BOOL, "range_gt(tsrange,tsrange)",  2, NULL, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(tsrange,tsrange)",  2, NULL, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(tsrange,tsrange)", 2, NULL, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tsrange,timestamp)",
	  4, NULL, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamp,tsrange)",
	  4, NULL, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_before" },
	{ BOOL, "range_after(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ "tsrange", "range_union(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_union" },
	{ "tsrange", "range_merge(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_merge" },
	{ "tsrange", "range_intersect(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_intersect" },
	{ "tsrange", "range_minus(tsrange,tsrange)",
	  4, NULL, "r/f:generic_range_minus" },

	/* tstzrange operators */
	{ "timestamptz", "lower(tstzrange)",
	  2, NULL, "r/f:tstzrange_lower" },
	{ "timestamptz", "upper(tstzrange)",
	  2, NULL, "r/f:tstzrange_upper" },
	{ BOOL, "isempty(tstzrange)",
	  1, NULL, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(tstzrange)",
	  1, NULL, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(tstzrange)",
	  1, NULL, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(tstzrange)",
	  1, NULL, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(tstzrange)",
	  1, NULL, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(tstzrange,tstzrange)",
	  2, NULL, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(tstzrange,tstzrange)",
	  2, NULL, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(tstzrange,tstzrange)",
	  2, NULL, "r/f:generic_range_lt" },
	{ BOOL, "range_le(tstzrange,tstzrange)",
	  2, NULL, "r/f:generic_range_le" },
	{ BOOL, "range_gt(tstzrange,tstzrange)",
	  2, NULL, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(tstzrange,tstzrange)",
	  2, NULL, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(tstzrange,tstzrange)",
	  2, NULL, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(tstzrange,timestamptz)",
	  4, NULL, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(timestamptz,tstzrange)",
	  4, NULL, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_before" },
	{ BOOL, "range_after(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ "tstzrange", "range_union(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_union" },
	{ "tstzrange", "range_merge(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_merge" },
	{ "tstzrange", "range_intersect(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_intersect" },
	{ "tstzrange", "range_minus(tstzrange,tstzrange)",
	  4, NULL, "r/f:generic_range_minus" },

	/* daterange operators */
	{ "date", "lower(daterange)",	2, NULL, "r/f:daterange_lower" },
	{ "date", "upper(daterange)",	2, NULL, "r/f:daterange_upper" },
	{ BOOL, "isempty(daterange)",	1, NULL, "r/f:generic_range_isempty" },
	{ BOOL, "lower_inc(daterange)",	1, NULL, "r/f:generic_range_lower_inc" },
	{ BOOL, "upper_inc(daterange)",	1, NULL, "r/f:generic_range_upper_inc" },
	{ BOOL, "lower_inf(daterange)",	1, NULL, "r/f:generic_range_lower_inf" },
	{ BOOL, "upper_inf(daterange)",	1, NULL, "r/f:generic_range_upper_inf" },
	{ BOOL, "range_eq(daterange,daterange)",
	  4, NULL, "r/f:generic_range_eq" },
	{ BOOL, "range_ne(daterange,daterange)",
	  4, NULL, "r/f:generic_range_ne" },
	{ BOOL, "range_lt(daterange,daterange)",
	  4, NULL, "r/f:generic_range_lt" },
	{ BOOL, "range_le(daterange,daterange)",
	  4, NULL, "r/f:generic_range_le" },
	{ BOOL, "range_gt(daterange,daterange)",
	  4, NULL, "r/f:generic_range_gt" },
	{ BOOL, "range_ge(daterange,daterange)",
	  4, NULL, "r/f:generic_range_ge" },
	{ INT4, "range_cmp(daterange,daterange)",
	  4, NULL, "r/f:generic_range_cmp" },
	{ BOOL, "range_overlaps(daterange,daterange)",
	  4, NULL, "r/f:generic_range_overlaps" },
	{ BOOL, "range_contains_elem(daterange,date)",
	  4, NULL, "r/f:generic_range_contains_elem" },
	{ BOOL, "range_contains(daterange,daterange)",
	  4, NULL, "r/f:generic_range_contains" },
	{ BOOL, "elem_contained_by_range(date,daterange)",
	  4, NULL, "r/f:generic_elem_contained_by_range" },
	{ BOOL, "range_contained_by(daterange,daterange)",
	  4, NULL, "r/f:generic_range_contained_by" },
	{ BOOL, "range_adjacent(daterange,daterange)",
	  4, NULL, "r/f:generic_range_adjacent" },
	{ BOOL, "range_before(daterange,daterange)",
	  4, NULL, "r/f:generic_range_before" },
	{ BOOL, "range_after(daterange,daterange)",
	  4, NULL, "r/f:generic_range_after" },
	{ BOOL, "range_overleft(daterange,daterange)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ BOOL, "range_overright(daterange,daterange)",
	  4, NULL, "r/f:generic_range_overleft" },
	{ "daterange", "range_union(daterange,daterange)",
	  4, NULL, "r/f:generic_range_union" },
	{ "daterange", "range_merge(daterange,daterange)",
	  4, NULL, "r/f:generic_range_merge" },
	{ "daterange", "range_intersect(daterange,daterange)",
	  4, NULL, "r/f:generic_range_intersect" },
	{ "daterange", "range_minus(daterange,daterange)",
	  4, NULL, "r/f:generic_range_minus" },

	/* type re-interpretation */
	{ INT8,   "as_int8("FLOAT8")", 1, NULL, "p/f:as_int8" },
	{ INT4,   "as_int4("FLOAT4")", 1, NULL, "p/f:as_int4" },
	{ INT2,   "as_int2("FLOAT2")", 1, NULL, "p/f:as_int2" },
	{ FLOAT8, "as_float8("INT8")", 1, NULL, "p/f:as_float8" },
	{ FLOAT4, "as_float4("INT4")", 1, NULL, "p/f:as_float4" },
	{ FLOAT2, "as_float2("INT2")", 1, NULL, "p/f:as_float2" },
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

/* default of dfunc->func_varlena_sz if not specified */
static int
devfunc_varlena_size_zero(devfunc_info *dfunc, Expr **args, int *vl_width)
{
	return 0;
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
			{
				entry->func_devcost = procat->func_devcost;
				entry->func_varlena_sz = (procat->func_varlena_sz
										  ? procat->func_varlena_sz
										  : devfunc_varlena_size_zero);
				return __construct_devfunc_info(entry, procat->func_template);
			}
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
			entry->func_devcost = procat->func_devcost;
			if (procat->func_varlena_sz)
				entry->func_varlena_sz = procat->func_varlena_sz;
			else
				entry->func_varlena_sz = devfunc_varlena_size_zero;
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
		 *
		 * When we transform an expression tree, PostgreSQL injects
		 * RelabelType node, and its buffers implicit binary-compatible
		 * type cast, however, caller has to pay attention if it specifies
		 * a particular function by OID.
		 */
		Form_pg_proc proc = (Form_pg_proc) GETSTRUCT(protup);
		oidvector  *proargtypes = &proc->proargtypes;
		Oid			src_type;
		Oid			dst_type;
		int			j;

		if (func_argtypes->dim1 != proargtypes->dim1)
			return NULL;
		for (j = 0; j < proargtypes->dim1; j++)
		{
			src_type = func_argtypes->values[j];
			dst_type = proargtypes->values[j];

			if (src_type == dst_type)
				continue;
			/* have a to_DESTTYPE() device function? */
			if (!pgstrom_devcast_supported(src_type, dst_type))
			{
				elog(DEBUG2, "no type cast definition (%s->%s)",
					 format_type_be(src_type),
					 format_type_be(dst_type));
				return NULL;
			}
		}

		if (func_rettype != proc->prorettype)
		{
			/* have a to_DESTTYPE() device function? */
			if (!pgstrom_devcast_supported(func_rettype, proc->prorettype))
			{
				elog(DEBUG2, "not binary compatible type cast (%s->%s)",
					 format_type_be(func_rettype),
					 format_type_be(proc->prorettype));
				return NULL;
			}
		}
		/* OK, type-relabel allows to call the function */
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
 */
static struct {
	Oid			src_type_oid;
	Oid			dst_type_oid;
	cl_uint		extra_flags;
} devcast_catalog[] = {
	/* text, varchar, bpchar */
	{ TEXTOID,    BPCHAROID,  DEVKERNEL_NEEDS_TEXTLIB },
	{ TEXTOID,    VARCHAROID, DEVKERNEL_NEEDS_TEXTLIB },
	{ VARCHAROID, TEXTOID,    DEVKERNEL_NEEDS_TEXTLIB },
	{ VARCHAROID, BPCHAROID,  DEVKERNEL_NEEDS_TEXTLIB },
	/* cidr -> inet, but no reverse type cast */
	{ CIDROID,    INETOID,    DEVKERNEL_NEEDS_MISC },
};

static void
build_devcast_info(void)
{
	int			i;

	Assert(devtype_info_is_built);
	for (i=0; i < lengthof(devcast_catalog); i++)
	{
		Oid				src_type_oid = devcast_catalog[i].src_type_oid;
		Oid				dst_type_oid = devcast_catalog[i].dst_type_oid;
		cl_uint			extra_flags  = devcast_catalog[i].extra_flags;
		cl_int			nslots = lengthof(devcast_info_slot);
		cl_int			index;
		devtype_info   *dtype;
		devcast_info   *dcast;
		HeapTuple		tup;
		char			method;

		tup = SearchSysCache2(CASTSOURCETARGET,
							  ObjectIdGetDatum(src_type_oid),
							  ObjectIdGetDatum(dst_type_oid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "Bug? type cast %s --> %s is not defined",
				 format_type_be(src_type_oid),
				 format_type_be(dst_type_oid));
		method = ((Form_pg_cast) GETSTRUCT(tup))->castmethod;
		if (method != COERCION_METHOD_BINARY)
			elog(ERROR, "Bug? type cast %s --> %s is not binary compatible",
				 format_type_be(src_type_oid),
				 format_type_be(dst_type_oid));
		ReleaseSysCache(tup);

		dcast = MemoryContextAllocZero(devinfo_memcxt,
									   sizeof(devcast_info));
		/* source */
		dtype = pgstrom_devtype_lookup(src_type_oid);
		if (!dtype)
			elog(ERROR, "Bug? type '%s' is not supported on device",
				 format_type_be(src_type_oid));
		extra_flags |= dtype->type_flags;
		dcast->src_type = dtype;
		/* destination */
		dtype = pgstrom_devtype_lookup(dst_type_oid);
		if (!dtype)
			elog(ERROR, "Bug? type '%s' is not supported on device",
				 format_type_be(dst_type_oid));
		extra_flags |= dtype->type_flags;
		dcast->dst_type = dtype;
		/* extra flags */
		dcast->extra_flags = extra_flags;

		index = (hash_uint32((uint32) src_type_oid) ^
				 hash_uint32((uint32) dst_type_oid)) % nslots;
		devcast_info_slot[index] = lappend_cxt(devinfo_memcxt,
											   devcast_info_slot[index],
											   dcast);
	}
}

bool
pgstrom_devcast_supported(Oid src_type_oid,
						  Oid dst_type_oid)
{
	int			nslots = lengthof(devcast_info_slot);
	int			index;
	ListCell   *lc;

	index = (hash_uint32((uint32) src_type_oid) ^
			 hash_uint32((uint32) dst_type_oid)) % nslots;
	foreach (lc, devcast_info_slot[index])
	{
		devcast_info   *dcast = lfirst(lc);

		if (dcast->src_type->type_oid == src_type_oid &&
			dcast->dst_type->type_oid == dst_type_oid)
		{
			//Right now, device type inclusion also contains
			//type cast above
			//*p_extra_flags |= dcast->extra_flags;
			return true;
		}
	}
	return false;
}

/*
 * codegen_expression_walker - main logic of run-time code generator
 */
static int codegen_function_expression(codegen_context *context,
									   devfunc_info *dfunc, List *args);

static void
codegen_expression_walker(codegen_context *context,
						  Node *node, int *p_varlena_sz)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	ListCell	   *cell;
	int				varlena_sz = -1;

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
		if (con->constlen < 0 && !con->constisnull)
			varlena_sz = VARSIZE_ANY_EXHDR(con->constvalue);
		else
			varlena_sz = 0;
	}
	else if (IsA(node, Param))
	{
		Param  *param = (Param *) node;
		int		index = 0;

		if (param->paramkind != PARAM_EXTERN)
			elog(ERROR, "codegen: ParamKind is not PARAM_EXTERN: %d",
				 (int)param->paramkind);

		dtype = pgstrom_devtype_lookup_and_track(param->paramtype, context);
		if (!dtype)
			elog(ERROR, "codegen: faied to lookup device type: %s",
				 format_type_be(param->paramtype));
		if (dtype->type_length < 0)
			varlena_sz = -1;	/* unknown */
		else
			varlena_sz = 0;

		foreach (cell, context->used_params)
		{
			if (equal(node, lfirst(cell)))
			{
				appendStringInfo(&context->str, "KPARAM_%u", index);
				context->param_refs =
					bms_add_member(context->param_refs, index);
				goto out;
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
		if (dtype->type_length < 0)
		{
			PlannerInfo	   *root = context->root;
			RelOptInfo	   *rel;

			varlena_sz = -1;
			if (var->varno == INDEX_VAR)
			{
				if (var->varnoold < root->simple_rel_array_size)
				{
					rel = root->simple_rel_array[var->varnoold];
					if (var->varoattno >= rel->min_attr &&
						var->varoattno <= rel->max_attr)
						varlena_sz = rel->attr_widths[var->varoattno -
													  rel->min_attr];
				}
			}
			else if (var->varno < root->simple_rel_array_size)
			{
				rel = root->simple_rel_array[var->varno];
				if (var->varattno >= rel->min_attr &&
					var->varattno <= rel->max_attr)
					varlena_sz = rel->attr_widths[var->varattno -
												  rel->min_attr];
			}
		}
		else
			varlena_sz = 0;
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
		dtype = dfunc->func_rettype;
		pgstrom_devfunc_track(context, dfunc);
		varlena_sz = codegen_function_expression(context, dfunc, func->args);
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
		dtype = dfunc->func_rettype;
		pgstrom_devfunc_track(context, dfunc);
		varlena_sz = codegen_function_expression(context, dfunc, op->args);
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
		codegen_expression_walker(context, (Node *) nulltest->arg, NULL);
		appendStringInfoChar(&context->str, ')');
		varlena_sz = 0;
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
		codegen_expression_walker(context, (Node *) booltest->arg, NULL);
		appendStringInfoChar(&context->str, ')');
		varlena_sz = 0;
	}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr   *b = (BoolExpr *) node;

		if (b->boolop == NOT_EXPR)
		{
			Assert(list_length(b->args) == 1);
			appendStringInfo(&context->str, "NOT(");
			codegen_expression_walker(context, linitial(b->args), NULL);
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
				codegen_expression_walker(context, lfirst(cell), NULL);
				appendStringInfoChar(&context->str, ')');
			}
			appendStringInfoChar(&context->str, ')');
		}
		else
			elog(ERROR, "unrecognized boolop: %d", (int) b->boolop);
		varlena_sz = 0;
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
			int		width;

			if (dtype->type_oid != type_oid)
				elog(ERROR, "device type mismatch in COALESCE: %s / %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(type_oid));
			appendStringInfo(&context->str, ", ");
			codegen_expression_walker(context, expr, &width);
			varlena_sz = Max(varlena_sz, width);
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
			int		width;

			if (dtype->type_oid != type_oid)
				elog(ERROR, "device type mismatch in LEAST/GREATEST: %s / %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(exprType(expr)));
			appendStringInfo(&context->str, ", ");
			codegen_expression_walker(context, expr, &width);
			varlena_sz = Max(varlena_sz, width);
		}
		appendStringInfoChar(&context->str, ')');
	}
	else if (IsA(node, RelabelType))
	{
		RelabelType	   *relabel = (RelabelType *) node;
		Oid				stype_oid = exprType((Node *)relabel->arg);

		dtype = pgstrom_devtype_lookup_and_track(relabel->resulttype, context);
		if (!dtype)
			elog(ERROR, "codegen: failed to lookup device type: %s",
				 format_type_be(relabel->resulttype));
		if (!pgstrom_devcast_supported(stype_oid, dtype->type_oid))
			elog(ERROR, "codegen: failed to lookup device cast: %s->%s",
				 format_type_be(stype_oid),
				 format_type_be(relabel->resulttype));

		appendStringInfo(&context->str, "to_%s(", dtype->type_name);
		codegen_expression_walker(context, (Node *)relabel->arg, &varlena_sz);
		appendStringInfo(&context->str, ")");
	}
	else if (IsA(node, CaseExpr))
	{
		CaseExpr   *caseexpr = (CaseExpr *) node;
		ListCell   *cell;
		Oid			type_oid;
		int			width;

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

			/* walk on the expression */
			dtype = pgstrom_devtype_lookup(caseexpr->casetype);
			if (!dtype)
                elog(ERROR, "codegen: failed to lookup device type: %s",
                     format_type_be(caseexpr->casetype));
			codegen_expression_walker(context,
									  (Node *)caseexpr->arg,
									  NULL);
			if (caseexpr->defresult)
			{
				appendStringInfo(&context->str, ", ");
				codegen_expression_walker(context,
										  (Node *)caseexpr->defresult,
										  &width);
				if (width >= 0)
					varlena_sz = Max(varlena_sz, width);
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
				codegen_expression_walker(context, test_val, NULL);
				appendStringInfo(&context->str, ", ");
				codegen_expression_walker(context,
										  (Node *)casewhen->result,
										  &width);
				if (width >= 0)
					varlena_sz = Max(varlena_sz, width);
			}
			appendStringInfo(&context->str, ")");
		}
		else
		{
			dtype = pgstrom_devtype_lookup(caseexpr->casetype);
			if (!dtype)
				elog(ERROR, "codegen: failed to lookup device type: %s",
					 format_type_be(caseexpr->casetype));

			foreach (cell, caseexpr->args)
			{
				CaseWhen   *casewhen = (CaseWhen *) lfirst(cell);

				Assert(exprType((Node *)casewhen->expr) == BOOLOID);
				Assert(exprType((Node *)casewhen->result) == dtype->type_oid);
				appendStringInfo(&context->str, "EVAL(");
				codegen_expression_walker(context,
										  (Node *)casewhen->expr, NULL);
				appendStringInfo(&context->str, ") ? (");
				codegen_expression_walker(context, (Node *)casewhen->result,
										  &width);
				if (width >= 0)
					varlena_sz = Max(varlena_sz, width);
				appendStringInfo(&context->str, ") : (");
			}
			codegen_expression_walker(context,
									  (Node *)caseexpr->defresult,
									  &width);
			if (width >= 0)
				varlena_sz = Max(varlena_sz, width);
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
		codegen_expression_walker(context, expr, NULL);
		appendStringInfo(&context->str, ", ");
		expr = lsecond(opexpr->args);
		codegen_expression_walker(context, expr, NULL);
		/* type of array element */
		dtype = lsecond(dfunc->func_args);
		appendStringInfo(&context->str, ", %s, %d, %d)",
						 opexpr->useOr ? "true" : "false",
						 dtype->type_length,
						 dtype->type_align);
		varlena_sz = 0;
	}
	else
		elog(ERROR, "Bug? unsupported expression: %s", nodeToString(node));
out:
	if (p_varlena_sz)
		*p_varlena_sz = varlena_sz;
}

static int
codegen_function_expression(codegen_context *context,
							devfunc_info *dfunc, List *args)
{
	ListCell   *lc1;
	ListCell   *lc2;
	Expr	   *fn_args[DEVFUNC_MAX_NARGS];
	int			vl_width[DEVFUNC_MAX_NARGS];
	int			index = 0;
	int			varlena_sz;

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
			codegen_expression_walker(context, expr, &vl_width[index]);
		else
		{
			appendStringInfo(&context->str,
							 "to_%s(", dtype->type_name);
			codegen_expression_walker(context, expr, &vl_width[index]);
			appendStringInfo(&context->str, ")");
		}
		fn_args[index++] = (Expr *)expr;
	}
	appendStringInfoChar(&context->str, ')');

	varlena_sz = dfunc->func_varlena_sz(dfunc, fn_args, vl_width);
	if (varlena_sz > 0)
		context->varlena_bufsz += MAXALIGN(VARHDRSZ + varlena_sz);
	else if (varlena_sz < 0)
		elog(ERROR, "cannot run %s on device due to varlena buffer usage",
			 format_procedure(dfunc->func_oid));
	return varlena_sz;
}

char *
pgstrom_codegen_expression(Node *expr, codegen_context *context)
{
	codegen_context	walker_context;
	int			width;

	initStringInfo(&walker_context.str);
	walker_context.root = context->root;
	walker_context.type_defs = list_copy(context->type_defs);
	walker_context.func_defs = list_copy(context->func_defs);
	walker_context.expr_defs = list_copy(context->expr_defs);
	walker_context.used_params = list_copy(context->used_params);
	walker_context.used_vars = list_copy(context->used_vars);
	walker_context.param_refs = bms_copy(context->param_refs);
	walker_context.var_label  = context->var_label;
	walker_context.kds_label  = context->kds_label;
	walker_context.kds_index_label = context->kds_index_label;
	walker_context.pseudo_tlist = context->pseudo_tlist;
	walker_context.extra_flags = context->extra_flags;
	walker_context.varlena_bufsz = context->varlena_bufsz;

	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = (Node *)linitial((List *)expr);
		else
			expr = (Node *)make_andclause((List *)expr);
	}

	PG_TRY();
	{
		codegen_expression_walker(&walker_context, expr, &width);
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
	context->varlena_bufsz = walker_context.varlena_bufsz;
	/*
	 * Even if expression itself needs no varlena extra buffer, projection
	 * code may require the buffer to construct a temporary datum.
	 * E.g) Numeric datum is encoded to 64bit at the GPU kernel, however,
	 * projection needs to decode to varlena again.
	 */
	if (width == 0)
	{
		Oid				type_oid = exprType((Node *) expr);
		devtype_info   *dtype = pgstrom_devtype_lookup(type_oid);

		context->varlena_bufsz += MAXALIGN(dtype->extra_sz);
	}
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
 * device_expression_walker
 */
typedef struct {
	const char *filename;
	int			lineno;
	PlannerInfo *root;
	int			devcost;
	ssize_t		vl_usage;
} device_expression_walker_context;

static bool
device_expression_walker(device_expression_walker_context *con,
						 Expr *expr, int *p_varlena_sz)
{
	int			varlena_sz = -1;	/* estimated length, if varlena */

	if (!expr)
	{
		/* do nothing */
	}
	else if (IsA(expr, Const))
	{
		Const		   *con = (Const *) expr;

		/* supported types only */
		if (!pgstrom_devtype_lookup(con->consttype))
			goto unable_node;
		if (con->constlen < 0 && !con->constisnull)
			varlena_sz = VARSIZE_ANY_EXHDR(con->constvalue);
		else
			varlena_sz = 0;
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
		/* we have no hint, even if argument is varlena */
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
		if (dtype->type_length < 0)
		{
			PlannerInfo	   *root = con->root;
			RelOptInfo	   *rel;

			varlena_sz = -1;	/* unknown, if no table statistics */
			if (var->varno == INDEX_VAR)
			{
				if (var->varnoold < root->simple_rel_array_size)
				{
					rel = root->simple_rel_array[var->varnoold];
					if (var->varoattno >= rel->min_attr &&
						var->varoattno <= rel->max_attr)
						varlena_sz = rel->attr_widths[var->varoattno -
													  rel->min_attr];
				}
			}
			else if (var->varno < root->simple_rel_array_size)
			{
				rel = root->simple_rel_array[var->varno];
				if (var->varattno >= rel->min_attr &&
					var->varattno <= rel->max_attr)
					varlena_sz = rel->attr_widths[var->varattno -
												  rel->min_attr];
			}
		}
		else
			varlena_sz = 0;
	}
	else if (IsA(expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) expr;
		devfunc_info *dfunc;
		Expr	   *fn_args[DEVFUNC_MAX_NARGS];
		int			vl_width[DEVFUNC_MAX_NARGS];
		int			index = 0;
		ListCell   *lc;

		dfunc = pgstrom_devfunc_lookup(func->funcid,
									   func->funcresulttype,
									   func->args,
									   func->inputcollid);
		if (!dfunc)
			goto unable_node;
		foreach (lc, func->args)
		{
			fn_args[index] = (Expr *) lfirst(lc);
			if (!device_expression_walker(con, fn_args[index],
										  &vl_width[index]))
				return false;
			index++;
		}
		varlena_sz = dfunc->func_varlena_sz(dfunc, fn_args, vl_width);
		if (varlena_sz > 0)
			con->vl_usage += MAXALIGN(VARHDRSZ + varlena_sz);
		else if (varlena_sz < 0)
			goto unable_node;	/* don't want to run on device */
		con->devcost += dfunc->func_devcost;
	}
	else if (IsA(expr, OpExpr) || IsA(expr, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;
		devfunc_info *dfunc;
		Expr	   *fn_args[DEVFUNC_MAX_NARGS];
		int			vl_width[DEVFUNC_MAX_NARGS];
		int			index = 0;
		ListCell   *lc;

		dfunc = pgstrom_devfunc_lookup(get_opcode(op->opno),
									   op->opresulttype,
									   op->args,
									   op->inputcollid);
		if (!dfunc)
			goto unable_node;
		foreach (lc, op->args)
		{
			fn_args[index] = (Expr *) lfirst(lc);
			if (!device_expression_walker(con, fn_args[index],
										  &vl_width[index]))
				return false;
			index++;
		}
		varlena_sz = dfunc->func_varlena_sz(dfunc, fn_args, vl_width);
		if (varlena_sz > 0)
			con->vl_usage += MAXALIGN(VARHDRSZ + varlena_sz);
		else if (varlena_sz < 0)
			goto unable_node;	/* don't want to run on device */
		con->devcost += dfunc->func_devcost;
	}
	else if (IsA(expr, NullTest))
	{
		NullTest   *nulltest = (NullTest *) expr;

		if (nulltest->argisrow)
			goto unable_node;
		if (!device_expression_walker(con, nulltest->arg, NULL))
			return false;
		/* cost for PG_ISNULL or PG_ISNOTNULL */
		con->devcost += 1;
		varlena_sz = 0;
	}
	else if (IsA(expr, BooleanTest))
	{
		BooleanTest	   *booltest = (BooleanTest *) expr;

		if (!device_expression_walker(con, booltest->arg, NULL))
			return false;
		/* cost for pgfn_bool_is_xxxx */
		con->devcost += 1;
		varlena_sz = 0;
	}
	else if (IsA(expr, BoolExpr))
	{
		BoolExpr	   *boolexpr = (BoolExpr *) expr;
		ListCell	   *lc;

		Assert(boolexpr->boolop == AND_EXPR ||
			   boolexpr->boolop == OR_EXPR ||
			   boolexpr->boolop == NOT_EXPR);
		foreach (lc, boolexpr->args)
		{
			if (!device_expression_walker(con, (Expr *)lfirst(lc), NULL))
				return false;
		}
		/* cost for bool-check expression */
		con->devcost += list_length(boolexpr->args);
		varlena_sz = 0;
	}
	else if (IsA(expr, CoalesceExpr))
	{
		CoalesceExpr   *coalesce = (CoalesceExpr *) expr;
		ListCell	   *lc;

		/* supported types only */
		if (!pgstrom_devtype_lookup(coalesce->coalescetype))
			goto unable_node;

		/* arguments also have to be same type (=device supported) */
		foreach (lc, coalesce->args)
		{
			Expr   *expr = lfirst(lc);
			int		vl_width = -1;

			/* arguments must be same type (= device supported) */
			if (coalesce->coalescetype != exprType((Node *)expr))
				goto unable_node;

			if (!device_expression_walker(con, expr, &vl_width))
				return false;
			if (vl_width > 0)
				varlena_sz = Max(varlena_sz, vl_width);
		}
		/* cost for PG_COALESCE */
		con->devcost += list_length(coalesce->args);
	}
	else if (IsA(expr, MinMaxExpr))
	{
		MinMaxExpr	   *minmax = (MinMaxExpr *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(minmax->minmaxtype);
		ListCell	   *lc;

		if (minmax->op != IS_GREATEST && minmax->op != IS_LEAST)
			return false;	/* unknown MinMax operation */

		/* only supported types */
		if (!dtype)
			goto unable_node;
		/* type compare function is required */
		if (!pgstrom_devfunc_lookup_type_compare(dtype, minmax->inputcollid))
			goto unable_node;

		/* arguments also have to be same type (=device supported) */
		foreach (lc, minmax->args)
		{
			Node   *expr = lfirst(lc);
			int		vl_width = -1;

			if (minmax->minmaxtype != exprType(expr))
				goto unable_node;
			if (!device_expression_walker(con, lfirst(lc), &vl_width))
				return false;
			if (vl_width > 0)
				varlena_sz = Max(varlena_sz, vl_width);
		}
		/* cost for PG_GREATEST / PG_LEAST */
		con->devcost += list_length(minmax->args);
	}
	else if (IsA(expr, RelabelType))
	{
		RelabelType	   *relabel = (RelabelType *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(relabel->resulttype);
		devtype_info   *stype =
			pgstrom_devtype_lookup(exprType((Node *)relabel->arg));

		/* array->array relabel may be possible */
		if (!dtype ||
			!stype ||
			!pgstrom_devcast_supported(stype->type_oid,
									   dtype->type_oid))
			goto unable_node;
		if (!device_expression_walker(con, relabel->arg, &varlena_sz))
			return false;
	}
	else if (IsA(expr, CaseExpr))
	{
		CaseExpr   *caseexpr = (CaseExpr *) expr;
		ListCell   *lc;
		int			vl_width;

		if (!pgstrom_devtype_lookup(caseexpr->casetype))
			goto unable_node;
		if (caseexpr->arg)
		{
			Oid		casetypeid = exprType((Node *)caseexpr->arg);
			Oid		casecollid = caseexpr->casecollid;
			devtype_info *dtype;

			dtype = pgstrom_devtype_lookup(casetypeid);
			if (!dtype)
				goto unable_node;
			/* type comparison function */
			if (!pgstrom_devfunc_lookup_type_compare(dtype, casecollid))
				goto unable_node;
			if (!device_expression_walker(con, caseexpr->arg, NULL))
				return false;
		}

		foreach (lc, caseexpr->args)
		{
			CaseWhen   *casewhen = lfirst(lc);

			Assert(IsA(casewhen, CaseWhen));
			if (exprType((Node *)casewhen->expr) == BOOLOID)
				goto unable_node;
			if (!device_expression_walker(con, casewhen->expr, NULL))
				return false;
			if (!device_expression_walker(con, casewhen->result, &vl_width))
				return false;
			if (vl_width > 0)
				varlena_sz = Max(varlena_sz, vl_width);
		}
		if (!device_expression_walker(con, (Expr *)caseexpr->defresult,
									  &vl_width))
			return false;
		if (vl_width > 0)
			varlena_sz = Max(varlena_sz, vl_width);
		return true;
	}
	else if (IsA(expr, CaseTestExpr))
	{
		CaseTestExpr   *casetest = (CaseTestExpr *) expr;
		devtype_info   *dtype = pgstrom_devtype_lookup(casetest->typeId);

		if (!dtype)
			goto unable_node;
		/* cost per CASE WHEN item */
		con->devcost += 1;
		return true;
	}
	else if (IsA(expr, ScalarArrayOpExpr))
	{
		ScalarArrayOpExpr *opexpr = (ScalarArrayOpExpr *) expr;
		devfunc_info   *dfunc;
		devtype_info   *dtype;
		Oid				func_oid = get_opcode(opexpr->opno);

		dfunc = pgstrom_devfunc_lookup(func_oid,
									   get_func_rettype(func_oid),
									   opexpr->args,
									   opexpr->inputcollid);
		if (!dfunc)
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

		if (!device_expression_walker(con, (Expr *) opexpr->args, NULL))
			return false;
		/*
		 * cost for PG_SCALAR_ARRAY_OP - It repeats invocation of the operator
		 * function for each array elements. Tentatively, we assume an array
		 * has 32 elements in average.
		 */
		con->devcost += 32 * dfunc->func_devcost;
		varlena_sz = 0;
		return true;
	}
	else
	{
		/* elsewhere, not supported anyway */
		goto unable_node;
	}
	if (p_varlena_sz)
		*p_varlena_sz = varlena_sz;
	return true;

unable_node:
	elog(DEBUG2, "Unable to run on device(%s:%d): %s",
		 basename(con->filename), con->lineno, nodeToString(expr));
	return false;
}

/*
 * pgstrom_device_expression(_cost)
 *
 * It shows a quick decision whether the provided expression tree is
 * available to run on CUDA device, or not.
 */
bool
__pgstrom_device_expression(PlannerInfo *root, Expr *expr,
							int *p_devcost, int *p_extra_sz,
							const char *filename, int lineno)
{
	device_expression_walker_context con;
	ListCell   *lc;

	memset(&con, 0, sizeof(device_expression_walker_context));
	con.root     = root;
	con.filename = filename;
	con.lineno   = lineno;
	con.devcost  = 0.0;
	con.vl_usage = 0;

	if (!expr)
		return false;
	if (IsA(expr, List))
	{
		List   *exprList = (List *) expr;

		foreach (lc, exprList)
		{
			if (!device_expression_walker(&con, (Expr *)lfirst(lc) , NULL))
				return false;
		}
	}
	else
	{
		if (!device_expression_walker(&con, expr, NULL))
			return false;
	}
	if (con.vl_usage > KERN_CONTEXT_VARLENA_BUFSZ_LIMIT)
	{
		elog(DEBUG2, "Expression consumes too much varlena buffer (%zu): %s",
			 con.vl_usage, nodeToString(expr));
		return false;
	}
	Assert(con.devcost >= 0);
	if (p_devcost)
		*p_devcost = con.devcost;
	if (p_extra_sz)
		*p_extra_sz = con.vl_usage;
	return true;
}

static void
codegen_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	MemoryContextReset(devinfo_memcxt);
	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));
	memset(devcast_info_slot, 0, sizeof(devcast_info_slot));
	pgstrom_float2_typeoid = InvalidOid;
	devtype_info_is_built = false;
}

static void
guc_assign_cache_invalidator(bool newval, void *extra)
{
	codegen_cache_invalidator(0, 0, 0);
}

void
pgstrom_init_codegen_context(codegen_context *context,
							 PlannerInfo *root)
{
	memset(context, 0, sizeof(codegen_context));
	context->root = root;
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
	CacheRegisterSyscacheCallback(CASTSOURCETARGET, codegen_cache_invalidator, 0);

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
