/*
 * codegen.c
 *
 * Routines for OpenCL code generator
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#include "postgres.h"
#include "access/hash.h"
#include "access/htup_details.h"
#include "access/sysattr.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "nodes/nodeFuncs.h"
#include "nodes/pg_list.h"
#include "optimizer/clauses.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/inval.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/pg_locale.h"
#include "utils/syscache.h"
#include "utils/typcache.h"
#include "pg_strom.h"

static MemoryContext	devinfo_memcxt;
static bool		devtype_info_is_built;
static List	   *devtype_info_slot[128];
static List	   *devfunc_info_slot[1024];

/*
 * Catalog of data types supported by device code
 *
 * naming convension of types:
 *   pg_<type_name>_t
 */
#define DEVTYPE_DECL(type_name,type_oid,type_base,type_flags)	\
	{ "pg_catalog", type_name, type_oid, #type_oid, type_base, type_flags }

static struct {
	const char	   *type_schema;
	const char	   *type_name;
	Oid				type_oid_static;
	const char	   *type_oid_label;
	const char	   *type_base;
	int32			type_flags;		/* library to declare this type */
} devtype_catalog[] = {
	/* primitive datatypes */
	DEVTYPE_DECL("bool",   BOOLOID,   "cl_bool",   0),		/* bool */
	DEVTYPE_DECL("int2",   INT2OID,   "cl_short",  0),		/* smallint */
	DEVTYPE_DECL("int4",   INT4OID,   "cl_int",    0),		/* int */
	DEVTYPE_DECL("int8",   INT8OID,   "cl_long",   0),		/* bigint */
	DEVTYPE_DECL("float4", FLOAT4OID, "cl_float",  0),		/* real */
	DEVTYPE_DECL("float8", FLOAT8OID, "cl_double", 0),		/* float */
	DEVTYPE_DECL("money",  CASHOID,   "cl_long",
				 DEVKERNEL_NEEDS_MONEY),	/* money */
	/* date and time datatypes */
	DEVTYPE_DECL("date", DATEOID, "DateADT", DEVKERNEL_NEEDS_TIMELIB),
	DEVTYPE_DECL("time", TIMEOID, "TimeADT", DEVKERNEL_NEEDS_TIMELIB),
	DEVTYPE_DECL("timetz", TIMETZOID, "TimeTzADT", DEVKERNEL_NEEDS_TIMELIB),
	DEVTYPE_DECL("timestamp", TIMESTAMPOID,"Timestamp",
				 DEVKERNEL_NEEDS_TIMELIB),
	DEVTYPE_DECL("timestamptz", TIMESTAMPTZOID, "TimestampTz",
				 DEVKERNEL_NEEDS_TIMELIB),
	DEVTYPE_DECL("interval", INTERVALOID, "Interval",
				 DEVKERNEL_NEEDS_TIMELIB),
	/* variable length datatypes */
	DEVTYPE_DECL("bpchar",  BPCHAROID,  "varlena *", DEVKERNEL_NEEDS_TEXTLIB),
	DEVTYPE_DECL("varchar", VARCHAROID, "varlena *", DEVKERNEL_NEEDS_TEXTLIB),
	DEVTYPE_DECL("numeric", NUMERICOID, "cl_ulong",  DEVKERNEL_NEEDS_NUMERIC),
	DEVTYPE_DECL("bytea",   BYTEAOID,   "varlena *", 0),
	DEVTYPE_DECL("text",    TEXTOID,    "varlena *", DEVKERNEL_NEEDS_TEXTLIB),
	/* pl/cuda datatypes */
	{"pg_catalog", "matrix", InvalidOid, "MATRIXOID",
	 "varlena *", DEVKERNEL_NEEDS_MATRIX },
};

static devtype_info *
build_devtype_info_entry(Oid type_oid,
						 int32 type_flags,
						 const char *type_basename,
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

   	entry->type_eqfunc = get_opcode(tcache->eq_opr);
	entry->type_cmpfunc = tcache->cmp_proc;
	if (!element)
		entry->type_array = build_devtype_info_entry(type_form->typarray,
													 type_flags |
													 DEVKERNEL_NEEDS_MATRIX,
													 "varlena *",
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

	context->extra_flags |= dtype->type_flags;
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
 * 'y' : this function needs cuda_money.h
 *
 * class character:
 * 'c' : this function is type cast that takes an argument
 * 'r' : this function is right operator that takes an argument
 * 'l' : this function is left operator that takes an argument
 * 'b' : this function is both operator that takes two arguments
 *     ==> extra is the operator character on OpenCL
 * 'f' : this function utilizes built-in functions
 *     ==> extra is the built-in function name
 * 'F' : this function is externally declared.
 *     ==> extra is the function name being declared somewhere
 */
typedef struct devfunc_catalog_t {
	const char *func_name;
	int			func_nargs;
	Oid			func_argtypes[4];
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
	{ "int2shr", 2, {INT2OID, INT4OID}, "b:>>" },
	{ "int4shr", 2, {INT4OID, INT4OID}, "b:>>" },
	{ "int8shr", 2, {INT8OID, INT4OID}, "b:>>" },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID, INT4OID}, "b:<<" },
	{ "int4shl", 2, {INT4OID, INT4OID}, "b:<<" },
	{ "int8shl", 2, {INT8OID, INT4OID}, "b:<<" },

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

	/*
	 * Misc time and date functions
	 */
	{ "now", 0, {}, "t/F:now" },

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

static void
devfunc_setup_cast(devfunc_info *entry,
				   devfunc_catalog_t *procat,
				   const char *extra, bool has_alias)
{
	devtype_info   *dtype = linitial(entry->func_args);

	Assert(procat->func_nargs == 1);
	entry->func_sqlname = pstrdup(procat->func_name);
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
						devfunc_catalog_t *procat,
						const char *extra, bool has_alias)
{
	devtype_info   *dtype1 = linitial(entry->func_args);
	devtype_info   *dtype2 = lsecond(entry->func_args);

	Assert(procat->func_nargs == 2);
	entry->func_sqlname = pstrdup(procat->func_name);
	entry->func_devname = (!has_alias
						   ? entry->func_sqlname
						   : psprintf("%s_%s_%s",
									  entry->func_sqlname,
									  dtype1->type_name,
									  dtype2->type_name));
	entry->func_decl
		= psprintf("STATIC_FUNCTION(pg_%s_t)\n"
				   "pgfn_%s(kern_context *kcxt, pg_%s_t arg1, pg_%s_t arg2)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    result.value = (%s)(arg1.value %s arg2.value);\n"
				   "    result.isnull = arg1.isnull | arg2.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_devname,
				   dtype1->type_name,
				   dtype2->type_name,
				   entry->func_rettype->type_name,
				   entry->func_rettype->type_base,
				   extra);
}

static void
devfunc_setup_oper_either(devfunc_info *entry,
						  devfunc_catalog_t *procat,
						  const char *left_extra,
						  const char *right_extra,
						  bool has_alias)
{
	devtype_info   *dtype = linitial(entry->func_args);

	Assert(procat->func_nargs == 1);
	entry->func_sqlname = pstrdup(procat->func_name);
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
						devfunc_catalog_t *procat,
						const char *extra, bool has_alias)
{
	devfunc_setup_oper_either(entry, procat, extra, NULL, has_alias);
}

static void
devfunc_setup_oper_right(devfunc_info *entry,
						 devfunc_catalog_t *procat,
						 const char *extra, bool has_alias)
{
	devfunc_setup_oper_either(entry, procat, NULL, extra, has_alias);
}

static void
devfunc_setup_func_decl(devfunc_info *entry,
						devfunc_catalog_t *procat,
						const char *extra, bool has_alias)
{
	StringInfoData	str;
	ListCell	   *cell;
	int				index;

	initStringInfo(&str);

	entry->func_sqlname = pstrdup(procat->func_name);
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
						devfunc_catalog_t *procat,
						const char *extra, bool has_alias)
{
	entry->func_sqlname = pstrdup(procat->func_name);
	if (has_alias)
		elog(ERROR, "Bug? implimented device function should not have alias");
	entry->func_devname = extra;
}

static devfunc_info *
pgstrom_devfunc_construct(Oid func_oid,
						  Oid func_collid,
						  const char *func_name,
						  Oid func_namespace,
						  int func_nargs,
						  Oid func_argtypes[],
						  Oid func_rettype,
						  bool func_is_strict)
{
	devfunc_info   *entry;
	int				i, j;

	entry = palloc0(sizeof(devfunc_info));
	entry->func_oid = func_oid;
	entry->func_sqlname = pstrdup(func_name);
	entry->func_is_strict = func_is_strict;

	/*
	 * At this moment, only (part-of) built-in functions are supported
	 * to run on GPU devices.
	 */
	if (func_namespace != PG_CATALOG_NAMESPACE)
	{
		entry->func_is_negative = true;
		return entry;
	}

	for (i=0; i < lengthof(devfunc_common_catalog); i++)
	{
		devfunc_catalog_t  *procat = devfunc_common_catalog + i;

		if (strcmp(procat->func_name, func_name) == 0 &&
			procat->func_nargs == func_nargs &&
			memcmp(procat->func_argtypes, func_argtypes,
				   sizeof(Oid) * func_nargs) == 0)
		{
			const char *template = procat->func_template;
			const char *extra;
			const char *pos;
			const char *end;
			int32		flags = 0;
			bool		has_alias = false;
			bool		has_collation = false;

			entry->func_rettype = pgstrom_devtype_lookup(func_rettype);
			if (!entry->func_rettype)
				elog(ERROR, "Bug? unsupported device function result type");
			for (j=0; j < func_nargs; j++)
			{
				devtype_info   *dtype
					= pgstrom_devtype_lookup(func_argtypes[j]);
				if (!dtype)
					elog(ERROR, "Bug? unsupported device function arguments");
				entry->func_args = lappend(entry->func_args, dtype);
			}
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
							flags |= DEVKERNEL_NEEDS_MONEY;
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

			/* In case when function is collation aware but not supported
			 * to run on GPU device, we have to give up.
			 */
			if (!has_collation)
				entry->func_collid = InvalidOid;
			else
			{
				entry->func_collid = func_collid;
				if (OidIsValid(func_collid) && !lc_collate_is_c(func_collid))
				{
					entry->func_is_negative = true;
					return entry;
				}
			}

			extra = template + 2;
			if (strncmp(template, "c:", 2) == 0)
				devfunc_setup_cast(entry, procat, extra, has_alias);
			else if (strncmp(template, "b:", 2) == 0)
				devfunc_setup_oper_both(entry, procat, extra, has_alias);
			else if (strncmp(template, "l:", 2) == 0)
				devfunc_setup_oper_left(entry, procat, extra, has_alias);
			else if (strncmp(template, "r:", 2) == 0)
				devfunc_setup_oper_right(entry, procat, extra, has_alias);
			else if (strncmp(template, "f:", 2) == 0)
				devfunc_setup_func_decl(entry, procat, extra, has_alias);
			else if (strncmp(template, "F:", 2) == 0)
				devfunc_setup_func_impl(entry, procat, extra, has_alias);
			else
			{
				elog(NOTICE, "Bug? unknown device function template: '%s'",
					 template);
				entry->func_is_negative = true;
			}
			return entry;
		}
	}
	/* no entries on the device function catalog */
	entry->func_is_negative = true;
	return entry;
}

devfunc_info *
pgstrom_devfunc_lookup(Oid func_oid, Oid func_collid)
{
	devfunc_info   *entry;
	Form_pg_proc	proc;
	HeapTuple		tuple;
	ListCell	   *lc;
	int				index;
	MemoryContext	oldcxt;

	index = hash_uint32((uint32) func_oid) % lengthof(devfunc_info_slot);
	foreach (lc, devfunc_info_slot[index])
	{
		entry = lfirst(lc);

		if (entry->func_oid == func_oid &&
			(!OidIsValid(entry->func_oid) ||
			 entry->func_collid == func_collid))
		{
			if (entry->func_is_negative)
				return NULL;
			return entry;
		}
	}

	/*
	 * Not found, construct a new entry for device function
	 */
	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	Assert(HeapTupleIsValid(tuple));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	proc = (Form_pg_proc) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	entry = pgstrom_devfunc_construct(func_oid,
									  func_collid,
									  NameStr(proc->proname),
									  proc->pronamespace,
									  proc->pronargs,
									  proc->proargtypes.values,
									  proc->prorettype,
									  proc->proisstrict);
	devfunc_info_slot[index] = lappend(devfunc_info_slot[index], entry);
	MemoryContextSwitchTo(oldcxt);

	ReleaseSysCache(tuple);

	if (entry->func_is_negative)
		return NULL;
	return entry;
}

static void
pgstrom_devfunc_track(codegen_context *context, devfunc_info *dfunc)
{
	ListCell   *lc;
	union {
		struct {
			Oid		func_oid;
			Oid		func_collid;
		} f;
		long		packed;
	} uval;

	foreach (lc, context->func_defs)
	{
		uval.packed = intVal(lfirst(lc));

		if (uval.f.func_oid == dfunc->func_oid &&
			uval.f.func_collid == dfunc->func_collid)
			return;
	}
	uval.f.func_oid = dfunc->func_oid;
	uval.f.func_collid = dfunc->func_collid;
	context->func_defs = lappend(context->func_defs,
								 makeInteger(uval.packed));	
}

devfunc_info *
pgstrom_devfunc_lookup_and_track(Oid func_oid, Oid func_collid,
								 codegen_context *context)
{
	devfunc_info   *dfunc = pgstrom_devfunc_lookup(func_oid, func_collid);
	devtype_info   *dtype;
	ListCell	   *cell;

	if (!dfunc)
		return NULL;
	/* track device function */
	context->extra_flags |= dfunc->func_flags;
	pgstrom_devfunc_track(context, dfunc);

	/* track function arguments and result type also */
	dtype = dfunc->func_rettype;
	context->extra_flags |= dtype->type_flags;
	pgstrom_devtype_track(context, dtype);

	foreach (cell, dfunc->func_args)
	{
		dtype = lfirst(cell);
		context->extra_flags |= dtype->type_flags;
		pgstrom_devtype_track(context, dtype);
	}
	return dfunc;
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
		appendStringInfo(&context->str, "%s_%u",
						 context->var_label,
						 varattno);
		context->used_vars = list_append_unique(context->used_vars,
												copyObject(node));
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) node;

		dfunc = pgstrom_devfunc_lookup_and_track(func->funcid,
												 func->inputcollid,
												 context);
		if (!dfunc)
			elog(ERROR, "codegen: failed to lookup device function: %s",
				 format_procedure(func->funcid));

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
		Oid			op_funcid = get_opcode(op->opno);

		dfunc = pgstrom_devfunc_lookup_and_track(op_funcid,
												 op->inputcollid,
												 context);
		if (!dfunc)
			elog(ERROR, "codegen: failed to lookup device function: %s",
                 format_procedure(op_funcid));

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
				Oid				expr_type;

				expr_type = exprType((Node *)caseexpr->arg);
				dtype = pgstrom_devtype_lookup_and_track(expr_type, context);
				if (!dtype)
					elog(ERROR, "codegen: failed to lookup device type: %s",
						 format_type_be(expr_type));

				dfunc = pgstrom_devfunc_lookup_and_track(dtype->type_eqfunc,
														 caseexpr->casecollid,
														 context);
				if (!dfunc)
					elog(ERROR,"codegen: failed to lookup device function: %s",
						 format_procedure(dtype->type_eqfunc));

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
	if (!dtype || !OidIsValid(dtype->type_cmpfunc))
		elog(ERROR, "unsupported device type in LEAST/GREATEST: %s",
			 format_type_be(minmax->minmaxtype));

	dfunc = pgstrom_devfunc_lookup_and_track(dtype->type_cmpfunc,
											 minmax->inputcollid,
											 context);
	if (!dfunc)
		elog(ERROR, "unsupported device function in LEAST/GREATEST: %s",
			 format_procedure(dtype->type_cmpfunc));

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
	dfunc = pgstrom_devfunc_lookup_and_track(func_oid,
											 opexpr->inputcollid,
											 context);
	if (!dfunc)
		elog(ERROR, "codegen: failed to lookup device function: %s",
			 format_procedure(get_opcode(opexpr->opno)));

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
		"  int        bitmask;\n",
		devexpr.expr_name,
		dtype1->type_name);
	codegen_tempvar_declaration(&decl, "temp");

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
	devfunc_info   *dfunc;
	union {
		struct {
			Oid		func_oid;
			Oid		func_collid;
		} f;
		long		packed;
	} uval;

	foreach (lc, context->func_defs)
	{
		uval.packed = intVal(lfirst(lc));

		dfunc = pgstrom_devfunc_lookup(uval.f.func_oid,
									   uval.f.func_collid);
		if (!dfunc)
			elog(ERROR, "Failed to lookup device function: %u",
				 uval.f.func_oid);
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
 * pgstrom_codegen_var_declarations
 */
void
pgstrom_codegen_var_declarations(StringInfo buf, codegen_context *context)
{
	ListCell	   *cell;

	foreach (cell, context->used_vars)
	{
		Var			   *var = lfirst(cell);
		devtype_info   *dtype = pgstrom_devtype_lookup(var->vartype);

		if (!dtype)
			elog(ERROR, "failed to lookup device type: %u", var->vartype);

		appendStringInfo(
			buf,
			"  pg_%s_t %s_%u = pg_%s_vref(%s,kcxt,%u,%s);\n",
			dtype->type_name,
			context->var_label,
			var->varattno,
			dtype->type_name,
			context->kds_label,
			var->varattno - 1,
			context->kds_index_label);
	}
}

/*
 * codegen_tempvar_declarations - it declares a temporary variable
 * that can store any data type supported by PG-Strom.
 */
void
codegen_tempvar_declaration(StringInfo buf, const char *varname)
{
	appendStringInfo(
		buf,
		"  union {\n"
		"    pg_varlena_t     varlena_v;\n"
		"    pg_bool_t        bool_v;\n"
		"    pg_int2_t        int2_v;\n"
		"    pg_int4_t        int4_v;\n"
		"    pg_int8_t        int8_v;\n"
		"    pg_float4_t      float4_v;\n"
		"    pg_float8_t      float8_v;\n"
		"#ifdef CUDA_NUMERIC_H\n"
		"    pg_numeric_t     numeric_v;\n"
		"#endif\n"
		"#ifdef CUDA_MONEY_H\n"
		"    pg_money_t       money_v;\n"
		"#endif\n"
		"#ifdef CUDA_TIMELIB_H\n"
		"    pg_date_t        date_v;\n"
		"    pg_time_t        time_v;\n"
		"    pg_timestamp_t   timestamp_v;\n"
		"    pg_timestamptz_t timestamptz_v;\n"
		"#endif\n"
		"#ifdef CUDA_TEXTLIB_H\n"
		"    pg_bpchar_t      bpchar_v;\n"
		"    pg_text_t        text_v;\n"
		"    pg_varchar_t     varchar_v;\n"
		"#endif\n"
		"  } %s	__attribute__ ((unused));\n",
		varname);
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
									func->inputcollid))
			goto unable_node;

		return pgstrom_device_expression((Expr *) func->args);
	}
	else if (IsA(expr, OpExpr) || IsA(expr, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		if (!pgstrom_devfunc_lookup(get_opcode(op->opno),
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
		if (!OidIsValid(dtype->type_cmpfunc) ||
			!pgstrom_devfunc_lookup(dtype->type_cmpfunc,
									minmax->inputcollid))
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

		if (!pgstrom_devfunc_lookup(get_opcode(opexpr->opno),
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
