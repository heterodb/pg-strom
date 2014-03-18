/*
 * codegen_expr.c
 *
 * Routines for OpenCL code generator for expression nodes
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "class/pg_proc.h"
#include "class/pg_type.h"
#include "utils/inval.h"
#include "utils/memutils.h"
#include "pg_strom.h"

static MemoryContext	devinfo_memcxt;
static List	   *devtype_info_slot[128];
static List	   *devfunc_info_slot[1024];

typedef struct devtype_info {
	Oid			type_oid;
	uint32		type_flags;
	char	   *type_ident;
	char	   *type_declare;
} devtype_info;

#define DEVINFO_IS_NEGATIVE			0x0001
#define DEVTYPE_IS_VARLENA			0x0002
#define DEVFUNC_NEEDS_TIMELIB		0x0004
#define DEVFUNC_NEEDS_TEXTLIB		0x0008
#define DEVFUNC_NEEDS_NUMERICLIB	0x0010

/*
 * Catalog of data type supported on the device code
 */
static struct {
	Oid			type_oid;
	const char *type_basename;
} devtype_catalog[] = {
	/* basic datatypes */
	{ BOOLOID,			"cl_bool" },
	{ INT2OID,			"cl_short" },
	{ INT4OID,			"cl_int" },
	{ INT8OID,			"cl_long" },
	{ FLOAT4OID,		"cl_float" },
	{ FLOAT8OID,		"cl_double" },
	/* date and time datatypes */
	{ DATEOID,			"cl_int" },
	{ TIMEOID,			"cl_long" },
	{ TIMESTAMPOID,		"cl_long" },
	{ TIMESTAMPTZOID,	"cl_long" },

	/* variable length datatypes */
	{ BPCHAROID,		"varlena" },
	{ VARCHAROID,		"varlena" },
	{ NUMERICOID,		"varlena" },
	{ BYTEAOID,			"varlena" },
	{ TEXTOID,			"varlena" },
};

devtype_info *
pgstrom_devtype_lookup(Oid type_oid)
{
	devtype_info   *entry;
	ListCell	   *cell;
	HeapTuple		tuple;
	Form_pg_type	typeform;
	MemoryContext	oldcxt;
	int				i, hash;

	hash = hash_uint32((uint32) type_oid) % lengthof(devtype_info_slot);
	foreach (cell, devtype_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->type_oid == type_oid)
		{
			if (entry->flags & DEVINFO_IS_NEGATIVE)
				return NULL;
			return entry;
		}
	}

	/*
	 * Not found, insert a new entry
	 */
	tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typeform = (Form_pg_type) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

	entry = palloc0(sizeof(devtype_info));
	entry->type_oid = type_oid;
	if (typeform->typlen < 0)
		entry->type_flags |= DEVTYPE_IS_VARLENA;
	if (typeform->typnamespace != PG_CATALOG_NAMESPACE)
		entry->type_flags |= DEVINFO_IS_NEGATIVE;
	else
	{
		const char *typname;
		char	   *decl;

		for (i=0; i < lengthof(devtype_catalog); i++)
		{
			if (devtype_catalog[i].type_oid != type_oid)
				continue;

			typname = NameStr(typeform->typname);
			entry->type_ident = psprintf("pg_%s_t", typname);
			if (entry->type_flags & DEVTYPE_IS_VARLENA)
				decl = psprintf("STROMCL_SIMPLE_TYPE_TEMPLATE(%s,%s)",
								entry->type_ident,
								devtype_catalog[i].type_basename);
			else
				decl = psprintf("STROMCL_VRALENA_TYPE_TEMPLATE(%s)",
								entry->type_ident);
			entry->type_declare = decl;
			break;
		}
		if (i == lengthof(devtype_catalog))
			entry->type_flags |= DEVINFO_IS_NEGATIVE;
	}
	devtype_info_slot[hash] = lappend(devtype_info_slot[hash], entry);
	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	if (entry->type_flags & DEVINFO_IS_NEGATIVE)
		return NULL;
	return entry;
}


typedef struct devfunc_info {
	const char *func_ident;
	int32		func_flags;
	List	   *func_args;	/* list of devtype_info */
	devtype_info *func_rettype;
	const char *func_declare;
	const char *func_implementation;
} devfunc_info;

typedef struct {
	const char *func_name;
	int			func_nargs;
	Oid			func_argtypes[4];
	const char *func_template;	/* a template string if simple function */
	void	  (*func_callback)(devfunc_info *devfunc, Form_pg_proc proc);
} devfunc_catalog_t;

devfunc_catalog_t devfunc_common_catalog[] = {
	/* Type cast functions */
	{ "int2", 1, {INT4OID}, "c:", NULL },
	{ "int2", 1, {INT8OID}, "c:", NULL },
	{ "int2", 1, {FLOAT4OID}, "c:", NULL },
	{ "int2", 1, {FLOAT8OID}, "c:", NULL },

	{ "int4", 1, {BOOLOID}, "c:", NULL },
	{ "int4", 1, {INT2OID}, "c:", NULL },
	{ "int4", 1, {INT8OID}, "c:", NULL },
	{ "int4", 1, {FLOAT4OID}, "c:", NULL },
	{ "int4", 1, {FLOAT8OID}, "c:", NULL },

	{ "int8", 1, {INT2OID}, "c:", NULL },
	{ "int8", 1, {INT4OID}, "c:", NULL },
	{ "int8", 1, {FLOAT4OID}, "c:", NULL },
	{ "int8", 1, {FLOAT8OID}, "c:", NULL },

	{ "float4", 1, {INT2OID}, "c:", NULL },
	{ "float4", 1, {INT4OID}, "c:", NULL },
	{ "float4", 1, {INT8OID}, "c:", NULL },
	{ "float4", 1, {FLOAT8OID}, "c:", NULL },

	{ "float8", 1, {INT2OID}, "c:", NULL },
	{ "float8", 1, {INT4OID}, "c:", NULL },
	{ "float8", 1, {INT8OID}, "c:", NULL },
	{ "float8", 1, {FLOAT4OID}, "c:", NULL },

	/* '+' : add operators */
	{ "int2pl",  2, {INT2OID, INT2OID}, "b:+", NULL },
	{ "int24pl", 2, {INT2OID, INT4OID}, "b:+", NULL },
	{ "int28pl", 2, {INT2OID, INT8OID}, "b:+", NULL },
	{ "int42pl", 2, {INT4OID, INT2OID}, "b:+", NULL },
	{ "int4pl",  2, {INT4OID, INT4OID}, "b:+", NULL },
	{ "int48pl", 2, {INT4OID, INT8OID}, "b:+", NULL },
	{ "int82pl", 2, {INT8OID, INT2OID}, "b:+", NULL },
	{ "int84pl", 2, {INT8OID, INT4OID}, "b:+", NULL },
	{ "int8pl",  2, {INT8OID, INT8OID}, "b:+", NULL },
	{ "float4pl",  2, {FLOAT4OID, FLOAT4OID}, "b:+", NULL },
	{ "float48pl", 2, {FLOAT4OID, FLOAT8OID}, "b:+", NULL },
	{ "float84pl", 2, {FLOAT4OID, FLOAT4OID}, "b:+", NULL },
	{ "float8pl",  2, {FLOAT8OID, FLOAT8OID}, "b:+", NULL },

	/* '-' : subtract operators */
	{ "int2mi",  2, {INT2OID, INT2OID}, "b:-", NULL },
	{ "int24mi", 2, {INT2OID, INT4OID}, "b:-", NULL },
	{ "int28mi", 2, {INT2OID, INT8OID}, "b:-", NULL },
	{ "int42mi", 2, {INT4OID, INT2OID}, "b:-", NULL },
	{ "int4mi",  2, {INT4OID, INT4OID}, "b:-", NULL },
	{ "int48mi", 2, {INT4OID, INT8OID}, "b:-", NULL },
	{ "int82mi", 2, {INT8OID, INT2OID}, "b:-", NULL },
	{ "int84mi", 2, {INT8OID, INT4OID}, "b:-", NULL },
	{ "int8mi",  2, {INT8OID, INT8OID}, "b:-", NULL },
	{ "float4mi",  2, {FLOAT4OID, FLOAT4OID}, "b:-", NULL },
	{ "float48mi", 2, {FLOAT4OID, FLOAT8OID}, "b:-", NULL },
	{ "float84mi", 2, {FLOAT4OID, FLOAT4OID}, "b:-", NULL },
	{ "float8mi",  2, {FLOAT8OID, FLOAT8OID}, "b:-", NULL },

	/* '*' : mutiply operators */
	{ "int2mul",  2, {INT2OID, INT2OID}, "b:*", NULL },
	{ "int24mul", 2, {INT2OID, INT4OID}, "b:*", NULL },
	{ "int28mul", 2, {INT2OID, INT8OID}, "b:*", NULL },
	{ "int42mul", 2, {INT4OID, INT2OID}, "b:*", NULL },
	{ "int4mul",  2, {INT4OID, INT4OID}, "b:*", NULL },
	{ "int48mul", 2, {INT4OID, INT8OID}, "b:*", NULL },
	{ "int82mul", 2, {INT8OID, INT2OID}, "b:*", NULL },
	{ "int84mul", 2, {INT8OID, INT4OID}, "b:*", NULL },
	{ "int8mul",  2, {INT8OID, INT8OID}, "b:*", NULL },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, "b:*", NULL },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, "b:*", NULL },
	{ "float84mul", 2, {FLOAT4OID, FLOAT4OID}, "b:*", NULL },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, "b:*", NULL },

	/* '/' : divide operators */
	{ "int2div",  2, {INT2OID, INT2OID}, NULL, make_div_oper },
	{ "int24div", 2, {INT2OID, INT4OID}, NULL, make_div_oper },
	{ "int28div", 2, {INT2OID, INT8OID}, NULL, make_div_oper },
	{ "int42div", 2, {INT4OID, INT2OID}, NULL, make_div_oper },
	{ "int4div",  2, {INT4OID, INT4OID}, NULL, make_div_oper },
	{ "int48div", 2, {INT4OID, INT8OID}, NULL, make_div_oper },
	{ "int82div", 2, {INT8OID, INT2OID}, NULL, make_div_oper },
	{ "int84div", 2, {INT8OID, INT4OID}, NULL, make_div_oper },
	{ "int8div",  2, {INT8OID, INT8OID}, NULL, make_div_oper },
	{ "float4div",  2, {FLOAT4OID, FLOAT4OID}, NULL, make_div_oper },
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, NULL, make_div_oper },
	{ "float84div", 2, {FLOAT4OID, FLOAT4OID}, NULL, make_div_oper },
	{ "float8div",  2, {FLOAT8OID, FLOAT8OID}, NULL, make_div_oper },

	/* '%' : reminder operators */
	{ "int2mod", 2, {INT2OID, INT2OID}, "b:%", NULL },
	{ "int4mod", 2, {INT4OID, INT4OID}, "b:%", NULL },
	{ "int8mod", 2, {INT8OID, INT8OID}, "b:%", NULL },

	/* '+' : unary plus operators */
	{ "int2up", 1, {INT2OID}, "l:+", NULL },
	{ "int4up", 1, {INT4OID}, "l:+", NULL },
	{ "int8up", 1, {INT8OID}, "l:+", NULL },
	{ "float4up", 1, {FLOAT4OID}, "l:+", NULL },
	{ "float8up", 1, {FLOAT8OID}, "l:+", NULL },

	/* '-' : unary minus operators */
	{ "int2mi", 1, {INT2OID}, "l:-", NULL },
	{ "int4mi", 1, {INT4OID}, "l:-", NULL },
	{ "int8mi", 1, {INT8OID}, "l:-", NULL },
	{ "float4mi", 1, {FLOAT4OID}, "l:-", NULL },
	{ "float8mi", 1, {FLOAT8OID}, "l:-", NULL },

	/* '@' : absolute value operators */
	{ "int2abs", 1, {INT2OID}, "f:abs", NULL },
	{ "int4abs", 1, {INT4OID}, "f:abs", NULL },
	{ "int8abs", 1, {INT8OID}, "f:abs", NULL },
	{ "float4abs", 1, {FLOAT4OID}, "l:fabs", NULL },
	{ "float8abs", 1, {FLOAT8OID}, "l:fabs", NULL },

	/* '=' : equal operators */
	{ "int2eq",  2, {INT2OID, INT2OID}, "b:==", NULL },
	{ "int24eq", 2, {INT2OID, INT4OID}, "b:==", NULL },
	{ "int28eq", 2, {INT2OID, INT8OID}, "b:==", NULL },
	{ "int42eq", 2, {INT4OID, INT2OID}, "b:==", NULL },
	{ "int4eq",  2, {INT4OID, INT4OID}, "b:==", NULL },
	{ "int48eq", 2, {INT4OID, INT8OID}, "b:==", NULL },
	{ "int82eq", 2, {INT8OID, INT2OID}, "b:==", NULL },
	{ "int84eq", 2, {INT8OID, INT4OID}, "b:==", NULL },
	{ "int8eq",  2, {INT8OID, INT8OID}, "b:==", NULL },
	{ "float4eq",  2, {FLOAT4OID, FLOAT4OID}, "b:==", NULL },
	{ "float48eq", 2, {FLOAT4OID, FLOAT8OID}, "b:==", NULL },
	{ "float84eq", 2, {FLOAT4OID, FLOAT4OID}, "b:==", NULL },
	{ "float8eq",  2, {FLOAT8OID, FLOAT8OID}, "b:==", NULL },

	/* '<>' : not equal operators */
	{ "int2ne",  2, {INT2OID, INT2OID}, "b:!=", NULL },
	{ "int24ne", 2, {INT2OID, INT4OID}, "b:!=", NULL },
	{ "int28ne", 2, {INT2OID, INT8OID}, "b:!=", NULL },
	{ "int42ne", 2, {INT4OID, INT2OID}, "b:!=", NULL },
	{ "int4ne",  2, {INT4OID, INT4OID}, "b:!=", NULL },
	{ "int48ne", 2, {INT4OID, INT8OID}, "b:!=", NULL },
	{ "int82ne", 2, {INT8OID, INT2OID}, "b:!=", NULL },
	{ "int84ne", 2, {INT8OID, INT4OID}, "b:!=", NULL },
	{ "int8ne",  2, {INT8OID, INT8OID}, "b:!=", NULL },
	{ "float4ne",  2, {FLOAT4OID, FLOAT4OID}, "b:!=", NULL },
	{ "float48ne", 2, {FLOAT4OID, FLOAT8OID}, "b:!=", NULL },
	{ "float84ne", 2, {FLOAT4OID, FLOAT4OID}, "b:!=", NULL },
	{ "float8ne",  2, {FLOAT8OID, FLOAT8OID}, "b:!=", NULL },

	/* '>' : equal operators */
	{ "int2gt",  2, {INT2OID, INT2OID}, "b:>", NULL },
	{ "int24gt", 2, {INT2OID, INT4OID}, "b:>", NULL },
	{ "int28gt", 2, {INT2OID, INT8OID}, "b:>", NULL },
	{ "int42gt", 2, {INT4OID, INT2OID}, "b:>", NULL },
	{ "int4gt",  2, {INT4OID, INT4OID}, "b:>", NULL },
	{ "int48gt", 2, {INT4OID, INT8OID}, "b:>", NULL },
	{ "int82gt", 2, {INT8OID, INT2OID}, "b:>", NULL },
	{ "int84gt", 2, {INT8OID, INT4OID}, "b:>", NULL },
	{ "int8gt",  2, {INT8OID, INT8OID}, "b:>", NULL },
	{ "float4gt",  2, {FLOAT4OID, FLOAT4OID}, "b:>", NULL },
	{ "float48gt", 2, {FLOAT4OID, FLOAT8OID}, "b:>", NULL },
	{ "float84gt", 2, {FLOAT4OID, FLOAT4OID}, "b:>", NULL },
	{ "float8gt",  2, {FLOAT8OID, FLOAT8OID}, "b:>", NULL },

	/* '<' : equal operators */
	{ "int2lt",  2, {INT2OID, INT2OID}, "b:<", NULL },
	{ "int24lt", 2, {INT2OID, INT4OID}, "b:<", NULL },
	{ "int28lt", 2, {INT2OID, INT8OID}, "b:<", NULL },
	{ "int42lt", 2, {INT4OID, INT2OID}, "b:<", NULL },
	{ "int4lt",  2, {INT4OID, INT4OID}, "b:<", NULL },
	{ "int48lt", 2, {INT4OID, INT8OID}, "b:<", NULL },
	{ "int82lt", 2, {INT8OID, INT2OID}, "b:<", NULL },
	{ "int84lt", 2, {INT8OID, INT4OID}, "b:<", NULL },
	{ "int8lt",  2, {INT8OID, INT8OID}, "b:<", NULL },
	{ "float4lt",  2, {FLOAT4OID, FLOAT4OID}, "b:<", NULL },
	{ "float48lt", 2, {FLOAT4OID, FLOAT8OID}, "b:<", NULL },
	{ "float84lt", 2, {FLOAT4OID, FLOAT4OID}, "b:<", NULL },
	{ "float8lt",  2, {FLOAT8OID, FLOAT8OID}, "b:<", NULL },

	/* '>=' : relational greater-than or equal-to */
	{ "int2ge",  2, {INT2OID, INT2OID}, "b:>=", NULL },
	{ "int24ge", 2, {INT2OID, INT4OID}, "b:>=", NULL },
	{ "int28ge", 2, {INT2OID, INT8OID}, "b:>=", NULL },
	{ "int42ge", 2, {INT4OID, INT2OID}, "b:>=", NULL },
	{ "int4ge",  2, {INT4OID, INT4OID}, "b:>=", NULL },
	{ "int48ge", 2, {INT4OID, INT8OID}, "b:>=", NULL },
	{ "int82ge", 2, {INT8OID, INT2OID}, "b:>=", NULL },
	{ "int84ge", 2, {INT8OID, INT4OID}, "b:>=", NULL },
	{ "int8ge",  2, {INT8OID, INT8OID}, "b:>=", NULL },
	{ "float4ge",  2, {FLOAT4OID, FLOAT4OID}, "b:>=", NULL },
	{ "float48ge", 2, {FLOAT4OID, FLOAT8OID}, "b:>=", NULL },
	{ "float84ge", 2, {FLOAT4OID, FLOAT4OID}, "b:>=", NULL },
	{ "float8ge",  2, {FLOAT8OID, FLOAT8OID}, "b:>=", NULL },

	/* '<=' : relational greater-than or equal-to */
	{ "int2le",  2, {INT2OID, INT2OID}, "b:<=", NULL },
	{ "int24le", 2, {INT2OID, INT4OID}, "b:<=", NULL },
	{ "int28le", 2, {INT2OID, INT8OID}, "b:<=", NULL },
	{ "int42le", 2, {INT4OID, INT2OID}, "b:<=", NULL },
	{ "int4le",  2, {INT4OID, INT4OID}, "b:<=", NULL },
	{ "int48le", 2, {INT4OID, INT8OID}, "b:<=", NULL },
	{ "int82le", 2, {INT8OID, INT2OID}, "b:<=", NULL },
	{ "int84le", 2, {INT8OID, INT4OID}, "b:<=", NULL },
	{ "int8le",  2, {INT8OID, INT8OID}, "b:<=", NULL },
	{ "float4le",  2, {FLOAT4OID, FLOAT4OID}, "b:<=", NULL },
	{ "float48le", 2, {FLOAT4OID, FLOAT8OID}, "b:<=", NULL },
	{ "float84le", 2, {FLOAT4OID, FLOAT4OID}, "b:<=", NULL },
	{ "float8le",  2, {FLOAT8OID, FLOAT8OID}, "b:<=", NULL },

	/* '&' : bitwise and */
	{ "int2and", 2, {INT2OID, INT2OID}, "b:&", NULL },
	{ "int4and", 2, {INT4OID, INT4OID}, "b:&", NULL },
	{ "int8and", 2, {INT8OID, INT8OID}, "b:&", NULL },

	/* '|'  : bitwise or */
	{ "int2or", 2, {INT2OID, INT2OID}, "b:|", NULL },
	{ "int4or", 2, {INT4OID, INT4OID}, "b:|", NULL },
	{ "int8or", 2, {INT8OID, INT8OID}, "b:|", NULL },

	/* '#'  : bitwise xor */
	{ "int2xor", 2, {INT2OID, INT2OID}, "b:^", NULL },
	{ "int4xor", 2, {INT4OID, INT4OID}, "b:^", NULL },
	{ "int8xor", 2, {INT8OID, INT8OID}, "b:^", NULL },

	/* '~'  : bitwise not operators */
	{ "int2not", 1, {INT2OID}, "b:~", NULL },
	{ "int4not", 1, {INT4OID}, "b:~", NULL },
	{ "int8not", 1, {INT8OID}, "b:~", NULL },

	/* '>>' : right shift */
	{ "int2shr", 2, {INT2OID, INT4OID}, "b:>>", NULL },
	{ "int4shr", 2, {INT4OID, INT4OID}, "b:>>", NULL },
	{ "int8shr", 2, {INT8OID, INT4OID}, "b:>>", NULL },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID, INT4OID}, "b:<<", NULL },
	{ "int4shl", 2, {INT4OID, INT4OID}, "b:<<", NULL },
	{ "int8shl", 2, {INT8OID, INT4OID}, "b:<<", NULL },

	/*
     * Mathmatical functions
     */
	/*
     * Trigonometric function
     */

};

devfunc_catalog_t devfunc_numericlib_catalog[] = {
	/* Type cast functions */
	{ "int2",    1, {NUMERICOID}, "f:pg_numeric_int2",   NULL },
	{ "int4",    1, {NUMERICOID}, "f:pg_numeric_int4",   NULL },
	{ "int8",    1, {NUMERICOID}, "f:pg_numeric_int8",   NULL },
	{ "float4",  1, {NUMERICOID}, "f:pg_numeric_float4", NULL },
	{ "float8",  1, {NUMERICOID}, "f:pg_numeric_float8", NULL },
	{ "numeric", 1, {INT2OID},    "f:pg_int2_numeric",   NULL },
	{ "numeric", 1, {INT4OID},    "f:pg_int4_numeric",   NULL },
	{ "numeric", 1, {INT8OID},    "f:pg_int8_numeric",   NULL },
	{ "numeric", 1, {FLOAT4OID},  "f:pg_float4_numeric", NULL },
	{ "numeric", 1, {FLOAT8OID},  "f:pg_float8_numeric", NULL },
	/* numeric operators */
	{ "numeric_add", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_add", NULL },
	{ "numeric_sub", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_sub", NULL },
	{ "numeric_mul", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_mul", NULL },
	{ "numeric_div", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_div", NULL },
	{ "numeric_mod", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_mod", NULL },
	{ "numeric_power", 2,{NUMERICOID, NUMERICOID},"f:pg_numeric_power", NULL},
	{ "numeric_uplus",  1, {NUMERICOID}, "f:pg_numeric_uplus", NULL },
	{ "numeric_uminus", 1, {NUMERICOID}, "f:pg_numeric_uminus", NULL },
	{ "numeric_abs",    1, {NUMERICOID}, "f:pg_numeric_abs", NULL },
	{ "numeric_eq", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_add", NULL },
	{ "numeric_ne", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_add", NULL },
	{ "numeric_lt", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_add", NULL },
	{ "numeric_le", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_add", NULL },
	{ "numeric_gt", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_add", NULL },
	{ "numeric_ge", 2, {NUMERICOID, NUMERICOID}, "f:pg_numeric_add", NULL },
};

devfunc_catalog_t devfunc_timelib_catalog[] = {
	/* Type cast functions */
	{ "date", 1, {DATEOID}, "c:", NULL },
	{ "date", 1, {TIMESTAMPOID}, "f:pg_timestamp_date", NULL },
	{ "date", 1, {TIMESTAMPTZOID}, "f:pg_timestamptz_date", NULL },
	{ "time", 1, {TIMESTAMPOID}, "f:pg_timestamp_time", NULL },
	{ "time", 1, {TIMESTAMPTZOID}, "f:pg_timestamptz_time", NULL },
	{ "time", 1, {TIMEOID}, "c:", NULL },
	{ "timestamp", 1, {TIMESTAMPOID}, "c:", NULL },
	{ "timestamp", 1, {TIMESTAMPTZOID}, "f:pg_timestamptz_timestamp", NULL },
	{ "timestamp", 1, {DATEOID}, "f:pg_date_timestamp", NULL },
	{ "timestamptz", 1, {TIMESTAMPOID}, "f:pg_timestamp_timestamptz", NULL },
	{ "timestamptz", 1, {TIMESTAMPTZOID}, "c:", NULL },
	{ "timestamptz", 1, {DATEOID}, "f:pg_date_timestamptz", NULL },
	/* timedata operators */
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, "f:pg_datetime_pl", NULL },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, "f:pg_timedata_pl", NULL },
	{ "date_pli", 2, {DATEOID, INT4OID}, "f:pg_date_pli", NULL },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, "f:integer_pl_date", NULL },
	{ "date_mii", 2, {DATEOID, INT4OID}, "f:pg_date_mii", NULL },
};

devfunc_catalog_t devfunc_textlib_catalog[] = {
	/* Type cast functions */
	{   }


};

devfunc_info *
pgstrom_devfunc_lookup(Oid func_oid)
{}



static void
codegen_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	MemoryContextReset(devinfo_memcxt);
	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));
}

void
pgstrom_codegen_expr_init(void)
{
	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));

	/* create a memory context */
	devinfo_memcxt = AllocSetContextCreate(CacheMemoryContext,
										   "device type/func info cache",
										   ALLOCSET_DEFAULT_MINSIZE,
										   ALLOCSET_DEFAULT_INITSIZE,
										   ALLOCSET_DEFAULT_MAXSIZE);
	CacheRegisterSyscacheCallback(PROCOID, codegen_cache_invalidator, 0);
	CacheRegisterSyscacheCallback(TYPEOID, codegen_cache_invalidator, 0);
}
