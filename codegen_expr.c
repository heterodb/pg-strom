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
#include "access/hash.h"
#include "access/htup_details.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "nodes/nodeFuncs.h"
#include "nodes/pg_list.h"
#include "utils/inval.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"
#include "pg_strom.h"

static MemoryContext	devinfo_memcxt;
static List	   *devtype_info_slot[128];
static List	   *devfunc_info_slot[1024];

/*
 * Catalog of data types supported by device code
 */
static struct {
	Oid				type_oid;
	const char	   *type_base;
	bool			type_is_builtin;	/* true, if no need to redefine */
} devtype_catalog[] = {
	/* basic datatypes */
	{ BOOLOID,			"cl_bool",	true },
	{ INT2OID,			"cl_short",	false },
	{ INT4OID,			"cl_int",	false },
	{ INT8OID,			"cl_long",	false },
	{ FLOAT4OID,		"cl_float",	false },
	{ FLOAT8OID,		"cl_double",false },
	/* date and time datatypes */
	{ DATEOID,			"cl_int",	false },
	{ TIMEOID,			"cl_long",	false },
	{ TIMESTAMPOID,		"cl_long",	false },
	{ TIMESTAMPTZOID,	"cl_long",	false },
	/* variable length datatypes */
	{ BPCHAROID,		"varlena",	false },
	{ VARCHAROID,		"varlena",	false },
	{ NUMERICOID,		"varlena",	false },
	{ BYTEAOID,			"varlena",	false },
	{ TEXTOID,			"varlena",	false },
};

static void
make_devtype_is_null_fn(devtype_info *dtype)
{
	devfunc_info   *dfunc;

	dfunc = palloc0(sizeof(devfunc_info));
	dfunc->func_ident = psprintf("%s_is_null", dtype->type_ident);
	dfunc->func_args = list_make1(dtype);
	dfunc->func_rettype = pgstrom_devtype_lookup(BOOLOID);
	dfunc->func_decl =
		psprintf("static %s %s(%s arg)\n"
				 "{\n"
				 "  %s result;\n\n"
				 "  result.isnull = false;\n"
				 "  result.value = arg.isnull;\n"
				 "  return result;\n"
				 "}\n",
				 dfunc->func_rettype->type_ident,
				 dfunc->func_ident,
				 dtype->type_ident,
				 dfunc->func_rettype->type_ident);
	dtype->type_is_null_fn = dfunc;
}

static void
make_devtype_is_not_null_fn(devtype_info *dtype)
{
	devfunc_info   *dfunc;

	dfunc = palloc0(sizeof(devfunc_info));
	dfunc->func_ident = psprintf("%s_is_not_null", dtype->type_ident);
	dfunc->func_args = list_make1(dtype);
	dfunc->func_rettype = pgstrom_devtype_lookup(BOOLOID);
	dfunc->func_decl =
		psprintf("static %s %s(%s arg)\n"
				 "{\n"
				 "  %s result;\n\n"
				 "  result.isnull = false;\n"
				 "  result.value = !arg.isnull;\n"
				 "  return result;\n"
				 "}\n",
				 dfunc->func_rettype->type_ident,
				 dfunc->func_ident,
				 dtype->type_ident,
				 dfunc->func_rettype->type_ident);
	dtype->type_is_not_null_fn = dfunc;
}

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
			if (entry->type_flags & DEVINFO_IS_NEGATIVE)
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
			entry->type_base = pstrdup(devtype_catalog[i].type_base);
			if (entry->type_flags & DEVTYPE_IS_VARLENA)
				decl = psprintf("STROMCL_SIMPLE_TYPE_TEMPLATE(%s,%s)",
								entry->type_ident,
								devtype_catalog[i].type_base);
			else
				decl = psprintf("STROMCL_VRALENA_TYPE_TEMPLATE(%s)",
								entry->type_ident);
			entry->type_decl = decl;
			if (devtype_catalog[i].type_is_builtin)
				entry->type_flags |= DEVTYPE_IS_BUILTIN;
			break;
		}
		if (i == lengthof(devtype_catalog))
			entry->type_flags |= DEVINFO_IS_NEGATIVE;
	}
	devtype_info_slot[hash] = lappend(devtype_info_slot[hash], entry);

	/*
	 * Misc support functions associated with device type
	 */
	make_devtype_is_null_fn(entry);
	make_devtype_is_not_null_fn(entry);

	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	if (entry->type_flags & DEVINFO_IS_NEGATIVE)
		return NULL;
	return entry;
}

/*
 * Catalog of functions supported by device code
 */
typedef struct devfunc_catalog_t {
	const char *func_name;
	int			func_nargs;
	Oid			func_argtypes[4];
	const char *func_template;	/* a template string if simple function */
	void	  (*func_callback)(devfunc_info *dfunc,
							   struct devfunc_catalog_t *procat);
} devfunc_catalog_t;

static void devfunc_setup_div_oper(devfunc_info *entry,
								   devfunc_catalog_t *procat);

static devfunc_catalog_t devfunc_common_catalog[] = {
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
	{ "int2div",  2, {INT2OID, INT2OID}, "0", devfunc_setup_div_oper },
	{ "int24div", 2, {INT2OID, INT4OID}, "0", devfunc_setup_div_oper },
	{ "int28div", 2, {INT2OID, INT8OID}, "0", devfunc_setup_div_oper },
	{ "int42div", 2, {INT4OID, INT2OID}, "0", devfunc_setup_div_oper },
	{ "int4div",  2, {INT4OID, INT4OID}, "0", devfunc_setup_div_oper },
	{ "int48div", 2, {INT4OID, INT8OID}, "0", devfunc_setup_div_oper },
	{ "int82div", 2, {INT8OID, INT2OID}, "0", devfunc_setup_div_oper },
	{ "int84div", 2, {INT8OID, INT4OID}, "0", devfunc_setup_div_oper },
	{ "int8div",  2, {INT8OID, INT8OID}, "0", devfunc_setup_div_oper },
	{ "float4div",  2, {FLOAT4OID, FLOAT4OID}, "0.0", devfunc_setup_div_oper },
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, "0.0", devfunc_setup_div_oper },
	{ "float84div", 2, {FLOAT4OID, FLOAT4OID}, "0.0", devfunc_setup_div_oper },
	{ "float8div",  2, {FLOAT8OID, FLOAT8OID}, "0.0", devfunc_setup_div_oper },

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

static devfunc_catalog_t devfunc_numericlib_catalog[] = {
	/* Type cast functions */
	{ "int2",    1, {NUMERICOID}, "F:pg_numeric_int2",   NULL },
	{ "int4",    1, {NUMERICOID}, "F:pg_numeric_int4",   NULL },
	{ "int8",    1, {NUMERICOID}, "F:pg_numeric_int8",   NULL },
	{ "float4",  1, {NUMERICOID}, "F:pg_numeric_float4", NULL },
	{ "float8",  1, {NUMERICOID}, "F:pg_numeric_float8", NULL },
	/* numeric operators */
#if 0
	/*
	 * Right now, functions that return variable-length field are not
	 * supported.
	 */
	{ "numeric_add", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_add", NULL },
	{ "numeric_sub", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_sub", NULL },
	{ "numeric_mul", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_mul", NULL },
	{ "numeric_div", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_div", NULL },
	{ "numeric_mod", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_mod", NULL },
	{ "numeric_power", 2,{NUMERICOID, NUMERICOID},"F:pg_numeric_power", NULL},
	{ "numeric_uplus",  1, {NUMERICOID}, "F:pg_numeric_uplus", NULL },
	{ "numeric_uminus", 1, {NUMERICOID}, "F:pg_numeric_uminus", NULL },
	{ "numeric_abs",    1, {NUMERICOID}, "F:pg_numeric_abs", NULL },
#endif
	{ "numeric_eq", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_eq", NULL },
	{ "numeric_ne", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_ne", NULL },
	{ "numeric_lt", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_lt", NULL },
	{ "numeric_le", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_le", NULL },
	{ "numeric_gt", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_gt", NULL },
	{ "numeric_ge", 2, {NUMERICOID, NUMERICOID}, "F:pg_numeric_ge", NULL },
};

static devfunc_catalog_t devfunc_timelib_catalog[] = {
	/* Type cast functions */
	{ "date", 1, {DATEOID}, "c:", NULL },
	{ "date", 1, {TIMESTAMPOID}, "F:pg_timestamp_date", NULL },
	{ "date", 1, {TIMESTAMPTZOID}, "F:pg_timestamptz_date", NULL },
	{ "time", 1, {TIMESTAMPOID}, "F:pg_timestamp_time", NULL },
	{ "time", 1, {TIMESTAMPTZOID}, "F:timestamptz_time", NULL },
	{ "time", 1, {TIMEOID}, "c:", NULL },
	{ "timestamp", 1, {TIMESTAMPOID}, "c:", NULL },
	{ "timestamp", 1, {TIMESTAMPTZOID}, "F:pg_timestamptz_timestamp", NULL },
	{ "timestamp", 1, {DATEOID}, "F:pg_date_timestamp", NULL },
	{ "timestamptz", 1, {TIMESTAMPOID}, "F:pg_timestamp_timestamptz", NULL },
	{ "timestamptz", 1, {TIMESTAMPTZOID}, "c:", NULL },
	{ "timestamptz", 1, {DATEOID}, "F:pg_date_timestamptz", NULL },
	/* timedata operators */
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, "F:pg_datetime_pl", NULL },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, "F:pg_timedata_pl", NULL },
	{ "date_pli", 2, {DATEOID, INT4OID}, "F:pg_date_pli", NULL },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, "F:pg_integer_pl_date", NULL },
	{ "date_mii", 2, {DATEOID, INT4OID}, "F:pg_date_mii", NULL },
	/* timedate comparison */
	{ "date_eq", 2, {DATEOID, DATEOID}, "b:==", NULL },
	{ "date_ne", 2, {DATEOID, DATEOID}, "b:!=", NULL },
	{ "date_lt", 2, {DATEOID, DATEOID}, "b:<", NULL },
	{ "date_le", 2, {DATEOID, DATEOID}, "b:<=", NULL },
	{ "date_gt", 2, {DATEOID, DATEOID}, "b:>", NULL },
	{ "date_ge", 2, {DATEOID, DATEOID}, "b:>=", NULL },
	{ "time_eq", 2, {TIMEOID, TIMEOID}, "b:==", NULL },
	{ "time_ne", 2, {TIMEOID, TIMEOID}, "b:!=", NULL },
	{ "time_lt", 2, {TIMEOID, TIMEOID}, "b:<", NULL },
	{ "time_le", 2, {TIMEOID, TIMEOID}, "b:<=", NULL },
	{ "time_gt", 2, {TIMEOID, TIMEOID}, "b:>", NULL },
	{ "time_ge", 2, {TIMEOID, TIMEOID}, "b:>=", NULL },
	{ "timestamp_eq", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:pg_timestamp_eq", NULL },
	{ "timestamp_ne", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:pg_timestamp_ne", NULL },
	{ "timestamp_lt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:pg_timestamp_lt", NULL },
	{ "timestamp_le", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:pg_timestamp_le", NULL },
	{ "timestamp_gt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:pg_timestamp_gt", NULL },
	{ "timestamp_ge", 2, {TIMESTAMPOID, TIMESTAMPOID}, "F:pg_timestamp_ge", NULL },
};

static devfunc_catalog_t devfunc_textlib_catalog[] = {
	{ "bpchareq", 2, {BPCHAROID,BPCHAROID}, "F:pg_bpchareq", NULL },
	{ "bpcharne", 2, {BPCHAROID,BPCHAROID}, "F:pg_bpcharne", NULL },
	{ "bpcharlt", 2, {BPCHAROID,BPCHAROID}, "F:pg_bpcharlt", NULL },
	{ "bpcharle", 2, {BPCHAROID,BPCHAROID}, "F:pg_bpcharle", NULL },
	{ "bpchargt", 2, {BPCHAROID,BPCHAROID}, "F:pg_bpchargt", NULL },
	{ "bpcharge", 2, {BPCHAROID,BPCHAROID}, "F:pg_bpcharge", NULL },
	{ "texteq", 2, {TEXTOID, TEXTOID}, "F:pg_texteq", NULL  },
	{ "textne", 2, {TEXTOID, TEXTOID}, "F:pg_textne", NULL  },
	{ "textlt", 2, {TEXTOID, TEXTOID}, "F:pg_textlt", NULL  },
	{ "textle", 2, {TEXTOID, TEXTOID}, "F:pg_textle", NULL  },
	{ "textgt", 2, {TEXTOID, TEXTOID}, "F:pg_textgt", NULL  },
	{ "textge", 2, {TEXTOID, TEXTOID}, "F:pg_textge", NULL  },
};

static void
devfunc_setup_div_oper(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype1 = linitial(entry->func_args);
	devtype_info   *dtype2 = lsecond(entry->func_args);

	Assert(procat->func_nargs == 2);
	entry->func_ident = psprintf("pg_%s", procat->func_name);
	entry->func_decl
		= psprintf("static %s %s(%s arg1, %s arg2)\n"
				   "{\n"
				   "    %s result;\n"
				   "    if (arg2 == %s)\n"
				   "    {\n"
				   "        result.isnull = true;\n"
				   "        PG_ERRORSET(ERRCODE_DIVISION_BY_ZERO);\n"
				   "    }\n"
				   "    else\n"
				   "    {\n"
				   "        result.value = (%s)(arg1 / arg2);\n"
				   "        result.isnull = arg1.isnull | arg2.isnull;\n"
				   "    }\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_ident,
				   entry->func_ident,
				   dtype1->type_ident,
				   dtype2->type_ident,
				   entry->func_rettype->type_ident,
				   procat->func_template,	/* 0 or 0.0 */
				   entry->func_rettype->type_base);
}

static void
devfunc_setup_cast(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype = linitial(entry->func_args);

	Assert(procat->func_nargs == 1);
	entry->func_ident = psprintf("pg_%s", procat->func_name);
	entry->func_decl
		= psprintf("static %s %s(%s arg)\n"
				   "{\n"
				   "    %s result;\n"
				   "    result.value  = (%s)arg.value;\n"
				   "    result.isnull = arg.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_ident,
				   entry->func_ident,
				   dtype->type_ident,
				   entry->func_rettype->type_ident,
				   entry->func_rettype->type_base);
}

static void
devfunc_setup_oper_both(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype1 = linitial(entry->func_args);
	devtype_info   *dtype2 = lsecond(entry->func_args);

	Assert(procat->func_nargs == 2);
	entry->func_ident = psprintf("pg_%s", procat->func_name);
	entry->func_decl
		= psprintf("static %s %s(%s arg1, %s arg2)\n"
				   "{\n"
				   "    %s result;\n"
				   "    result.value = (%s)(arg1 %s arg2);\n"
				   "    result.isnull = arg1.isnull | arg2.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_ident,
				   entry->func_ident,
				   dtype1->type_ident,
				   dtype2->type_ident,
				   entry->func_rettype->type_ident,
				   entry->func_rettype->type_base,
				   procat->func_template + 2);
}

static void
devfunc_setup_oper_either(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype = linitial(entry->func_args);
	const char	   *templ = procat->func_template;

	Assert(procat->func_nargs == 1);
	entry->func_ident = psprintf("pg_%s", procat->func_name);
	entry->func_decl
		= psprintf("static %s %s(%s arg)\n"
				   "{\n"
				   "    %s result;\n"
				   "    result.value = (%s)(%sarg%s);\n"
				   "    result.isnull = arg.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_ident,
				   entry->func_ident,
				   dtype->type_ident,
				   entry->func_rettype->type_ident,
				   entry->func_rettype->type_base,
				   strncmp(templ, "l:", 2) == 0 ? templ + 2 : "",
				   strncmp(templ, "r:", 2) == 0 ? templ + 2 : "");
}

static void
devfunc_setup_func(devfunc_info *entry, devfunc_catalog_t *procat)
{
	StringInfoData	str;
	ListCell	   *cell;
	int				index;
	const char	   *templ = procat->func_template;

	entry->func_ident = psprintf("pg_%s", procat->func_name);
	/* declaration */
	initStringInfo(&str);
	appendStringInfo(&str, "static %s %s(",
					 entry->func_rettype->type_ident,
					 entry->func_ident);
	index = 1;
	foreach (cell, entry->func_args)
	{
		devtype_info   *dtype = lfirst(cell);

		appendStringInfo(&str, "%s%s arg%d",
						 cell == list_head(entry->func_args) ? "" : ", ",
						 dtype->type_ident,
						 index++);
	}
	appendStringInfo(&str, ")\n"
					 "{\n"
					 "    %s result;\n"
					 "    result.isnull = ",
					 entry->func_rettype->type_ident);
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
					 templ + 2);
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

static devfunc_info *
devfunc_setup_boolop(BoolExprType boolop, const char *fn_name, int fn_nargs)
{
	devfunc_info   *entry = palloc0(sizeof(devfunc_info));
	devtype_info   *dtype = pgstrom_devtype_lookup(BOOLOID);
	StringInfoData	str;
	int		i;

	initStringInfo(&str);

	for (i=0; i < fn_nargs; i++)
		entry->func_args = lappend(entry->func_args, dtype);
	entry->func_rettype = dtype;
	entry->func_ident = pstrdup(fn_name);
	appendStringInfo(&str, "static %s %s(", dtype->type_ident, fn_name);
	for (i=0; i < fn_nargs; i++)
		appendStringInfo(&str, "%s%s arg%u",
						 (i > 0 ? ", " : ""),
						 dtype->type_ident, i+1);
	appendStringInfo(&str, ")\n"
					 "{\n"
					 "  %s result;\n"
					 "  result.isnull = ",
					 dtype->type_ident);
	for (i=0; i < fn_nargs; i++)
		appendStringInfo(&str, "%sarg%u.isnull",
						 (i > 0 ? " | " : ""), i+1);
	appendStringInfo(&str, ";\n"
					 "  result.value = ");
	for (i=0; i < fn_nargs; i++)
	{
		if (boolop == AND_EXPR)
			appendStringInfo(&str, "%sarg%u.value", (i > 0 ? " & " : ""), i+1);
		else
			appendStringInfo(&str, "%sarg%u.value", (i > 0 ? " | " : ""), i+1);
	}
	appendStringInfo(&str, ";\n"
					 "  return result;\n"
					 "}\n");
	entry->func_decl = str.data;

	return entry;
}

static devfunc_info *
pgstrom_devfunc_lookup_by_name(const char *func_name,
							   Oid func_namespace,
							   int func_nargs,
							   Oid func_argtypes[],
							   Oid func_rettype)
{
	devfunc_info   *entry;
	ListCell	   *cell;
	MemoryContext	oldcxt;
	int32			flags = 0;
	int				i, j, k, hash;
	devfunc_catalog_t *procat = NULL;

	hash = (hash_any((void *)func_name, strlen(func_name)) ^
			hash_any((void *)func_argtypes, sizeof(Oid) * func_nargs))
		% lengthof(devfunc_info_slot);

	foreach (cell, devfunc_info_slot[hash])
	{
		entry = lfirst(cell);
		if (func_namespace == entry->func_namespace &&
			strcmp(func_name, entry->func_name) == 0 &&
			func_nargs == list_length(entry->func_args) &&
			memcmp(func_argtypes, entry->func_argtypes,
				   sizeof(Oid) * func_nargs) == 0)
		{
			Assert(entry->func_rettype->type_oid == func_rettype);
			if (entry->func_flags & DEVINFO_IS_NEGATIVE)
				return NULL;
			return entry;
		}
	}
	/* the function not found */

	/*
	 * We may have device-only functions that has no namespace.
	 * The caller has to be responsible to add these function entries
	 * into the cache.
	 */
	if (func_namespace == InvalidOid)
		return NULL;

	/* Elsewhere, let's walk on the function catalog */
	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

	entry = palloc0(sizeof(devfunc_info));
	entry->func_name = pstrdup(func_name);
	entry->func_namespace = func_namespace;
	entry->func_argtypes = palloc(sizeof(Oid) * func_nargs);
	memcpy(entry->func_argtypes, func_argtypes, sizeof(Oid) * func_nargs);

	if (func_namespace == PG_CATALOG_NAMESPACE)
	{
		static struct {
			devfunc_catalog_t *catalog;
			int		nitems;
			int		flags;
		} catalog_array[] = {
			{ devfunc_common_catalog,
			  lengthof(devfunc_common_catalog),
			  0 },
			{ devfunc_numericlib_catalog,
			  lengthof(devfunc_numericlib_catalog),
			  DEVFUNC_NEEDS_NUMERICLIB },
			{ devfunc_timelib_catalog,
			  lengthof(devfunc_timelib_catalog),
			  DEVFUNC_NEEDS_TIMELIB },
			{ devfunc_textlib_catalog,
			  lengthof(devfunc_textlib_catalog),
			  DEVFUNC_NEEDS_TEXTLIB },
		};

		for (i=0; i < lengthof(catalog_array); i++)
		{
			flags = catalog_array[i].flags;
			for (j=0; j < catalog_array[i].nitems; j++)
			{
				procat = &catalog_array[i].catalog[j];

				if (strcmp(procat->func_name, func_name) == 0 &&
					procat->func_nargs == func_nargs &&
					memcmp(procat->func_argtypes, func_argtypes,
						   sizeof(Oid) * func_nargs) == 0)
				{
					entry->func_flags = flags;
					entry->func_rettype = pgstrom_devtype_lookup(func_rettype);
					Assert(entry->func_rettype != NULL);

					for (k=0; k < func_nargs; k++)
					{
						devtype_info   *dtype
							= pgstrom_devtype_lookup(func_argtypes[i]);
						Assert(dtype != NULL);
						entry->func_args = lappend(entry->func_args, dtype);
					}

					if (procat->func_callback)
						procat->func_callback(entry, procat);
					else if (strncmp(procat->func_template, "c:", 2) == 0)
						devfunc_setup_cast(entry, procat);
					else if (strncmp(procat->func_template, "b:", 2) == 0)
						devfunc_setup_oper_both(entry, procat);
					else if (strncmp(procat->func_template, "l:", 2) == 0 ||
							 strncmp(procat->func_template, "r:", 2) == 0)
						devfunc_setup_oper_either(entry, procat);
					else if (strncmp(procat->func_template, "f:", 2) == 0)
						devfunc_setup_func(entry, procat);
					else if (strncmp(procat->func_template, "F:", 2) == 0)
						entry->func_ident = pstrdup(procat->func_template + 2);
					else
						entry->func_flags = DEVINFO_IS_NEGATIVE;

					goto out;
				}
			}
		}
	}
	entry->func_flags = DEVINFO_IS_NEGATIVE;
out:
	devfunc_info_slot[hash] = lappend(devfunc_info_slot[hash], entry);

	MemoryContextSwitchTo(oldcxt);

	if (entry->func_flags & DEVINFO_IS_NEGATIVE)
		return NULL;
	return entry;
}

devfunc_info *
pgstrom_devfunc_lookup(Oid func_oid)
{
	Form_pg_proc	proc;
	HeapTuple		tuple;
	devfunc_info   *dfunc;

	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	proc = (Form_pg_proc) GETSTRUCT(tuple);

	dfunc = pgstrom_devfunc_lookup_by_name(NameStr(proc->proname),
										   proc->pronamespace,
										   proc->pronargs,
										   proc->proargtypes.values,
										   proc->prorettype);
	ReleaseSysCache(tuple);

	return dfunc;
}

typedef struct
{
	StringInfoData	str;
	List	   *type_defs;		/* list of devtype_info */
	List	   *func_defs;		/* list of devfunc_info */
	List	   *used_params;	/* list of Const or Param nodes */
	List	   *used_vars;		/* list of Var nodes */
	int			incl_flags;
} codegen_walker_context;

static devtype_info *
devtype_lookup_and_track(Oid type_oid, codegen_walker_context *context)
{
	devtype_info   *dtype = pgstrom_devtype_lookup(type_oid);
	if (dtype)
		context->type_defs = list_append_unique(context->type_defs, dtype);
	return dtype;
}

static devfunc_info *
devfunc_lookup_and_track(Oid func_oid, codegen_walker_context *context)
{
	devfunc_info   *dfunc = pgstrom_devfunc_lookup(func_oid);
	if (dfunc)
	{
		context->func_defs = list_append_unique(context->func_defs, dfunc);
		context->incl_flags |= (dfunc->func_flags & DEVFUNC_INCL_FLAGS);
	}
	return dfunc;
}

static bool
codegen_expression_walker(Node *node, codegen_walker_context *context)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	ListCell	   *cell;

	if (node == NULL)
		return false;

	if (IsA(node, Const))
	{
		Const  *con = (Const *) node;
		int		index = 1;

		if (!devtype_lookup_and_track(con->consttype, context))
			return true;

		foreach (cell, context->used_params)
		{
			if (equal(node, lfirst(cell)))
			{
				appendStringInfo(&context->str, "KPARAM_%d", index);
				return false;
			}
			index++;
		}
		context->used_params = lappend(context->used_params,
									   copyObject(node));
		appendStringInfo(&context->str, "KPARAM_%d",
						 list_length(context->used_params));
		return false;
	}
	else if (IsA(node, Param))
	{
		Param  *param = (Param *) node;
		int		index = 1;

		if (!devtype_lookup_and_track(param->paramtype, context))
			return true;

		foreach (cell, context->used_params)
		{
			if (equal(node, lfirst(cell)))
			{
				appendStringInfo(&context->str, "KPARAM_%u", index);
				return false;
			}
			index++;
		}
		context->used_params = lappend(context->used_params,
									   copyObject(node));
		appendStringInfo(&context->str, "KPARAM_%u",
						 list_length(context->used_params));
		return false;
	}
	else if (IsA(node, Var))
	{
		Var	   *var = (Var *) node;
		int		index = 1;

		if (!devtype_lookup_and_track(var->vartype, context))
			return true;

		foreach (cell, context->used_vars)
		{
			if (equal(node, lfirst(cell)))
			{
				appendStringInfo(&context->str, "KVAR_%u", index);
				return false;
			}
			index++;
		}
		context->used_vars = lappend(context->used_vars,
									 copyObject(node));
		appendStringInfo(&context->str, "KVAR_%u", index);
		return false;
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) node;

		/* no collation support */
		if (OidIsValid(func->funccollid) || OidIsValid(func->inputcollid))
			return true;

		dfunc = devfunc_lookup_and_track(func->funcid, context);
		if (!func)
			return true;
		appendStringInfo(&context->str, "%s(", dfunc->func_ident);

		foreach (cell, func->args)
		{
			if (cell != list_head(func->args))
				appendStringInfo(&context->str, ", ");
			if (codegen_expression_walker(lfirst(cell), context))
				return true;
		}
		appendStringInfoChar(&context->str, ')');

		return false;
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) node;

		/* no collation support */
		if (OidIsValid(op->opcollid) || OidIsValid(op->inputcollid))
			return true;

		dfunc = devfunc_lookup_and_track(get_opcode(op->opno), context);
		if (!dfunc)
			return true;
		appendStringInfo(&context->str, "%s(", dfunc->func_ident);

		foreach (cell, op->args)
		{
			if (cell != list_head(op->args))
				appendStringInfo(&context->str, ", ");
			if (codegen_expression_walker(lfirst(cell), context))
				return true;
		}
		appendStringInfoChar(&context->str, ')');

		return false;
	}
	else if (IsA(node, NullTest))
	{
		NullTest   *nulltest = (NullTest *) node;
		const char *func_ident;

		if (nulltest->argisrow)
			return true;

		dtype = pgstrom_devtype_lookup(exprType((Node *)nulltest->arg));
		if (!dtype)
			return true;

		switch (nulltest->nulltesttype)
		{
			case IS_NULL:
				func_ident = dtype->type_is_null_fn->func_ident;
				break;
			case IS_NOT_NULL:
				func_ident = dtype->type_is_not_null_fn->func_ident;
				break;
			default:
				elog(ERROR, "unrecognized nulltesttype: %d",
					 (int)nulltest->nulltesttype);
				break;
		}
		appendStringInfo(&context->str, "%s(", func_ident);
		if (codegen_expression_walker((Node *) nulltest->arg, context))
			return true;
		appendStringInfoChar(&context->str, ')');

		return false;
	}
	else if (IsA(node, BooleanTest))
	{
		BooleanTest	   *booltest = (BooleanTest *) node;
		const char	   *func_ident;

		if (exprType((Node *)booltest->arg) != BOOLOID)
			elog(ERROR, "argument of BooleanTest is not bool");

		/* choose one of built-in functions */
		switch (booltest->booltesttype)
		{
			case IS_TRUE:
				func_ident = "pg_bool_is_true";
				break;
			case IS_NOT_TRUE:
				func_ident = "pg_bool_is_not_true";
				break;
			case IS_FALSE:
				func_ident = "pg_bool_is_false";
				break;
			case IS_NOT_FALSE:
				func_ident = "pg_bool_is_not_false";
				break;
			case IS_UNKNOWN:
				func_ident = "pg_bool_is_unknown";
				break;
			case IS_NOT_UNKNOWN:
				func_ident = "pg_bool_is_not_unknown";
				break;
			default:
				elog(ERROR, "unrecognized booltesttype: %d",
					 (int)booltest->booltesttype);
				break;
		}
		appendStringInfo(&context->str, "%s(", func_ident);
		if (codegen_expression_walker((Node *) booltest->arg, context))
			return true;
		appendStringInfoChar(&context->str, ')');
		return false;
	}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr   *b = (BoolExpr *) node;

		if (b->boolop == NOT_EXPR)
		{
			Assert(list_length(b->args) == 1);
			appendStringInfo(&context->str, "pg_boolop_not(");
			if (codegen_expression_walker(linitial(b->args), context))
				return true;
			appendStringInfoChar(&context->str, ')');
		}
		else if (b->boolop == AND_EXPR || b->boolop == OR_EXPR)
		{
			char	namebuf[NAMEDATALEN];
			int		nargs = list_length(b->args);
			Oid	   *argtypes = alloca(sizeof(Oid) * nargs);
			int		i;

			if (b->boolop == AND_EXPR)
				snprintf(namebuf, sizeof(namebuf), "pg_boolop_and_%u", nargs);
			else
				snprintf(namebuf, sizeof(namebuf), "pg_boolop_or_%u", nargs);

			for (i=0; i < nargs; i++)
				argtypes[i] = BOOLOID;

			/*
			 * AND/OR Expr is device only functions, so no catalog entries
			 * and needs to set up here.
			 */
			dfunc = pgstrom_devfunc_lookup_by_name(namebuf,
												   InvalidOid,
												   nargs,
												   argtypes,
												   BOOLOID);
			if (!dfunc)
				dfunc = devfunc_setup_boolop(b->boolop, namebuf, nargs);
			context->func_defs = list_append_unique(context->func_defs, dfunc);
			context->incl_flags |= (dfunc->func_flags & DEVFUNC_INCL_FLAGS);

			appendStringInfo(&context->str, "%s(", dfunc->func_ident);
			foreach (cell, b->args)
			{
				Assert(exprType(lfirst(cell)) == BOOLOID);
				if (cell != list_head(b->args))
					appendStringInfo(&context->str, ", ");
				if (codegen_expression_walker(lfirst(cell), context))
					return true;
			}
			appendStringInfoChar(&context->str, ')');
		}
		else
			elog(ERROR, "unrecognized boolop: %d", (int) b->boolop);
		return false;
	}
	return true;
}

char *
pgstrom_codegen_expression(Node *expr, codegen_context *context)
{
	codegen_walker_context	walker_context;

	initStringInfo(&walker_context.str);
	walker_context.type_defs = list_copy(context->type_defs);
	walker_context.func_defs = list_copy(context->func_defs);
	walker_context.used_params = list_copy(context->used_params);
	walker_context.used_vars = list_copy(context->used_vars);
	walker_context.incl_flags = context->incl_flags;

	if (codegen_expression_walker(expr, &walker_context))
		return NULL;

	context->type_defs = walker_context.type_defs;
	context->func_defs = walker_context.func_defs;
	context->used_params = walker_context.used_params;
	context->used_vars = walker_context.used_vars;
	context->incl_flags = walker_context.incl_flags;

	return walker_context.str.data;
}

char *
pgstrom_codegen_declarations(codegen_context *context)
{
	StringInfoData	str;
	ListCell	   *cell;

	initStringInfo(&str);
	appendStringInfo(&str, "#include \"opencl_common.h\"\n");
	if (context->incl_flags & DEVFUNC_NEEDS_TIMELIB)
		appendStringInfo(&str, "#include \"opencl_timelib.h\"\n");
	if (context->incl_flags & DEVFUNC_NEEDS_TEXTLIB)
		appendStringInfo(&str, "#include \"opencl_textlib.h\"\n");
	if (context->incl_flags & DEVFUNC_NEEDS_NUMERICLIB)
		appendStringInfo(&str, "#include \"opencl_numericlib.h\"\n");
	appendStringInfoChar(&str, '\n');

	foreach (cell, context->type_defs)
	{
		devtype_info   *dtype = lfirst(cell);

		if (dtype->type_flags & DEVTYPE_IS_VARLENA)
			appendStringInfo(&str, "STROMCL_VARLENA_TYPE_TEMPLATE(%s)\n",
							 dtype->type_ident);
		else
			appendStringInfo(&str, "STROMCL_SIMPLE_TYPE_TEMPLATE(%s,%s)\n",
							 dtype->type_ident, dtype->type_base);
	}
	appendStringInfoChar(&str, '\n');

	foreach (cell, context->func_defs)
	{
		devfunc_info   *dfunc = lfirst(cell);

		appendStringInfo(&str, "%s\n", dfunc->func_decl);
	}
	return str.data;
}

/*
 * codegen_available_expression
 *
 * It shows a quick decision whether the provided expression tree is
 * available to run on OpenCL device, or not.
 */
bool
pgstrom_codegen_available_expression(Expr *expr)
{
	if (expr == NULL)
		return true;
	if (IsA(expr, List))
	{
		ListCell   *cell;

		foreach (cell, (List *) expr)
		{
			if (!pgstrom_codegen_available_expression(lfirst(cell)))
				return false;
		}
		return true;
	}
	else if (IsA(expr, Const))
	{
		if (pgstrom_devtype_lookup(((Const *) expr)->consttype))
			return true;
		return false;
	}
	else if (IsA(expr, Param))
	{
		if (pgstrom_devtype_lookup(((Param *) expr)->paramtype))
			return true;
		return false;
	}
	else if (IsA(expr, Var))
	{
		if (pgstrom_devtype_lookup(((Var *) expr)->vartype))
			return true;
		return false;
	}
	else if (IsA(expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) expr;

		/* no collation support */
		if (OidIsValid(func->funccollid) || OidIsValid(func->inputcollid))
			return false;

		if (pgstrom_devfunc_lookup(func->funcid))
			return true;
		return pgstrom_codegen_available_expression((Expr *) func->args);
	}
	else if (IsA(expr, OpExpr) || IsA(expr, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		/* no collation support */
		if (OidIsValid(op->opcollid) || OidIsValid(op->inputcollid))
			return false;

		if (pgstrom_devfunc_lookup(get_opcode(op->opno)))
			return true;
		return pgstrom_codegen_available_expression((Expr *) op->args);
	}
	else if (IsA(expr, NullTest))
	{
		NullTest   *nulltest = (NullTest *) expr;

		if (nulltest->argisrow)
			return false;
		return pgstrom_codegen_available_expression((Expr *) nulltest->arg);
	}
	else if (IsA(expr, BooleanTest))
	{
		BooleanTest	   *booltest = (BooleanTest *) expr;

		return pgstrom_codegen_available_expression((Expr *) booltest->arg);
	}
	return false;
}




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
