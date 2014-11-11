/*
 * codegen.c
 *
 * Routines for OpenCL code generator
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "utils/fmgroids.h"
#include "utils/inval.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/pg_locale.h"
#include "utils/syscache.h"
#include "pg_strom.h"

static MemoryContext	devinfo_memcxt;
static List	   *devtype_info_slot[128];
static List	   *devfunc_info_slot[1024];

/*
 * Catalog of data types supported by device code
 *
 * naming convension of types:
 *   pg_<type_name>_t
 */
static struct {
	Oid				type_oid;
	const char	   *type_base;
	Oid				type_eqfunc;	/* function to check equality */
	Oid				type_cmpfunc;	/* function to compare two values */
} devtype_catalog[] = {
	/* basic datatypes */
	{ BOOLOID,			"cl_bool",		F_BOOLEQ,	F_BTBOOLCMP },
	{ INT2OID,			"cl_short",		F_INT2EQ,	F_BTINT2CMP },
	{ INT4OID,			"cl_int",		F_INT4EQ,	F_BTINT4CMP },
	{ INT8OID,			"cl_long",		F_INT8EQ,	F_BTINT8CMP },
	{ FLOAT4OID,		"cl_float",		F_FLOAT4EQ,	F_BTFLOAT4CMP },
	{ FLOAT8OID,		"cl_double",	F_FLOAT8EQ,	F_BTFLOAT8CMP },
	/* date and time datatypes */
	{ DATEOID,			"cl_int",		F_DATE_EQ,	F_DATE_CMP },
#ifdef HAVE_INT64_TIMESTAMP
	{ TIMEOID,			"cl_long",		F_TIME_EQ,	F_DATE_CMP },
	{ TIMESTAMPOID,		"cl_long",		F_TIMESTAMP_EQ, F_TIMESTAMP_CMP },
#endif
	/* variable length datatypes */
	{ BPCHAROID,		"varlena",		F_BPCHAREQ,	F_BPCHARCMP },
	{ VARCHAROID,		"varlena",		InvalidOid,	InvalidOid },
	{ NUMERICOID,		"varlena",		F_NUMERIC_EQ, F_NUMERIC_CMP },
	{ BYTEAOID,			"varlena",		F_BYTEAEQ,	F_BYTEACMP },
	{ TEXTOID,			"varlena",		F_TEXTEQ,	F_BTTEXTCMP },
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
		for (i=0; i < lengthof(devtype_catalog); i++)
		{
			if (devtype_catalog[i].type_oid != type_oid)
				continue;

			entry->type_length = typeform->typlen;
			entry->type_align = typealign_get_width(typeform->typalign);
			entry->type_name = pstrdup(NameStr(typeform->typname));
			entry->type_base = pstrdup(devtype_catalog[i].type_base);
			entry->type_eqfunc = devtype_catalog[i].type_eqfunc; 
			entry->type_cmpfunc = devtype_catalog[i].type_cmpfunc;
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

/*
 * collation checks - device does not understand text locale,
 * so all the locale we can handle is the binary comparable ones.
 * we have to ensure the collation is binary comparable.
 */
static bool
devtype_runnable_collation(Oid collation)
{
	/*
	 * Usually, core backend looks up a particular collation id,
	 * if it is varlena data type. (If failed to lookup, an error
	 * shall be reported). Elsewhere, it shall be a fixed-length
	 * datum that has no collation anyway.
	 */
	if (!OidIsValid(collation))
		return true;
	return lc_collate_is_c(collation);
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
 */
typedef struct devfunc_catalog_t {
	const char *func_name;
	int			func_nargs;
	Oid			func_argtypes[4];
	const char *func_template;	/* a template string if simple function */
	void	  (*func_callback)(devfunc_info *dfunc,
							   struct devfunc_catalog_t *procat);
} devfunc_catalog_t;

static void devfunc_setup_mul_oper(devfunc_info *entry,
								   devfunc_catalog_t *procat);
static void devfunc_setup_div_oper(devfunc_info *entry,
								   devfunc_catalog_t *procat);
static void devfunc_setup_const(devfunc_info *entry,
								devfunc_catalog_t *procat);
static void devfunc_setup_comp(devfunc_info *entry,
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
	{ "int2mul",  2, {INT2OID, INT2OID}, NULL, devfunc_setup_mul_oper },
	{ "int24mul", 2, {INT2OID, INT4OID}, NULL, devfunc_setup_mul_oper },
	{ "int28mul", 2, {INT2OID, INT8OID}, NULL, devfunc_setup_mul_oper },
	{ "int42mul", 2, {INT4OID, INT2OID}, NULL, devfunc_setup_mul_oper },
	{ "int4mul",  2, {INT4OID, INT4OID}, NULL, devfunc_setup_mul_oper },
	{ "int48mul", 2, {INT4OID, INT8OID}, NULL, devfunc_setup_mul_oper },
	{ "int82mul", 2, {INT8OID, INT2OID}, NULL, devfunc_setup_mul_oper },
	{ "int84mul", 2, {INT8OID, INT4OID}, NULL, devfunc_setup_mul_oper },
	{ "int8mul",  2, {INT8OID, INT8OID}, NULL, devfunc_setup_mul_oper },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, NULL, devfunc_setup_mul_oper },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, NULL, devfunc_setup_mul_oper },
	{ "float84mul", 2, {FLOAT4OID, FLOAT4OID}, NULL, devfunc_setup_mul_oper },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, NULL, devfunc_setup_mul_oper },

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

	/* comparison functions */
	{ "btboolcmp",  2, {BOOLOID, BOOLOID}, NULL, devfunc_setup_comp },
	{ "btint2cmp",  2, {INT2OID, INT2OID}, NULL, devfunc_setup_comp },
	{ "btint24cmp", 2, {INT2OID, INT4OID}, NULL, devfunc_setup_comp },
	{ "btint28cmp", 2, {INT2OID, INT8OID}, NULL, devfunc_setup_comp },
	{ "btint42cmp", 2, {INT4OID, INT2OID}, NULL, devfunc_setup_comp },
	{ "btint4cmp",  2, {INT4OID, INT4OID}, NULL, devfunc_setup_comp },
	{ "btint48cmp", 2, {INT4OID, INT8OID}, NULL, devfunc_setup_comp },
	{ "btint82cmp", 2, {INT8OID, INT2OID}, NULL, devfunc_setup_comp },
	{ "btint84cmp", 2, {INT8OID, INT4OID}, NULL, devfunc_setup_comp },
	{ "btint8cmp",  2, {INT8OID, INT8OID}, NULL, devfunc_setup_comp },
	{ "btfloat4cmp",  2, {FLOAT4OID, FLOAT4OID}, NULL, devfunc_setup_comp },
	{ "btfloat48cmp", 2, {FLOAT4OID, FLOAT8OID}, NULL, devfunc_setup_comp },
	{ "btfloat84cmp", 2, {FLOAT8OID, FLOAT4OID}, NULL, devfunc_setup_comp },
	{ "btfloat8cmp",  2, {FLOAT8OID, FLOAT8OID}, NULL, devfunc_setup_comp },

	/*
     * Mathmatical functions
     */
	{ "abs", 1, {INT2OID}, "f:abs", NULL },
	{ "abs", 1, {INT4OID}, "f:abs", NULL },
	{ "abs", 1, {INT8OID}, "f:abs", NULL },
	{ "abs", 1, {FLOAT4OID}, "f:fabs", NULL },
	{ "abs", 1, {FLOAT8OID}, "f:fabs", NULL },
	{ "cbrt",  1, {FLOAT4OID}, "f:cbrt", NULL },
	{ "dcbrt", 1, {FLOAT8OID}, "f:cbrt", NULL },
	{ "ceil", 1, {FLOAT8OID}, "f:ceil", NULL },
	{ "ceiling", 1, {FLOAT8OID}, "f:ceil", NULL },
	{ "exp", 1, {FLOAT8OID}, "f:exp", NULL },
	{ "dexp", 1, {FLOAT8OID}, "f:exp", NULL },
	{ "floor", 1, {FLOAT8OID}, "f:dfloor", NULL },
	{ "ln", 1, {FLOAT8OID}, "f:log", NULL },
	{ "dlog1", 1, {FLOAT8OID}, "f:log", NULL },
	{ "log", 1, {FLOAT8OID}, "f:log10", NULL },
	{ "dlog10", 1, {FLOAT8OID}, "f:log10", NULL },
	{ "pi", 0, {}, "f:M_PI", devfunc_setup_const },
	{ "power", 2, {FLOAT8OID, FLOAT8OID}, "f:pow" },
	{ "pow", 2, {FLOAT8OID, FLOAT8OID}, "f:pow" },
	{ "dpow", 2, {FLOAT8OID, FLOAT8OID}, "f:pow" },
	{ "round", 1, {FLOAT8OID}, "f:round", NULL },
	{ "dround", 1, {FLOAT8OID}, "f:round", NULL },
	{ "sign", 1, {FLOAT8OID}, "f:sign", NULL },
	{ "sqrt", 1, {FLOAT8OID}, "f:sqrt", NULL },
	{ "dsqrt", 1, {FLOAT8OID}, "f:sqrt", NULL },
	{ "trunc", 1, {FLOAT8OID}, "f:trunc", NULL },
	{ "dtrunc", 1, {FLOAT8OID}, "f:trunc", NULL },

	/*
     * Trigonometric function
     */
	{ "degrees", 1, {FLOAT4OID}, "f:degrees", NULL },
	{ "degrees", 1, {FLOAT8OID}, "f:degrees", NULL },
	{ "radians", 1, {FLOAT8OID}, "f:radians", NULL },
	{ "acos",    1, {FLOAT8OID}, "f:acos", NULL },
	{ "asin",    1, {FLOAT8OID}, "f:asin", NULL },
	{ "atan",    1, {FLOAT8OID}, "f:atan", NULL },
	{ "atan2",   2, {FLOAT8OID, FLOAT8OID}, "f:atan2", NULL },
	{ "cos",     1, {FLOAT8OID}, "f:cos", NULL },
	//{ "cot",     1, {FLOAT8OID}, "f:", NULL }, /* not supported in opencl */
	{ "sin",     1, {FLOAT8OID}, "f:sin", NULL },
	{ "tan",     1, {FLOAT8OID}, "f:tan", NULL },
};

static devfunc_catalog_t devfunc_numericlib_catalog[] = {
#if 0
	/* Type cast functions */
	{ "int2",    1, {NUMERICOID}, "F:numeric_int2",   NULL },
	{ "int4",    1, {NUMERICOID}, "F:numeric_int4",   NULL },
	{ "int8",    1, {NUMERICOID}, "F:numeric_int8",   NULL },
	{ "float4",  1, {NUMERICOID}, "F:numeric_float4", NULL },
	{ "float8",  1, {NUMERICOID}, "F:numeric_float8", NULL },
	{ "numeric", 1, {INT2OID},    "F:int2_numeric",   NULL },
	{ "numeric", 1, {INT4OID},    "F:int4_numeric",   NULL },
	{ "numeric", 1, {INT8OID},    "F:int8_numeric",   NULL },
	{ "numeric", 1, {FLOAT4OID},  "F:float4_numeric", NULL },
	{ "numeric", 1, {FLOAT8OID},  "F:float8_numeric", NULL },
	/* Numeric operators */
	{ "numeric_add", 2, {NUMERICOID, NUMERICOID}, "F:numeric_add", NULL },
	{ "numeric_sub", 2, {NUMERICOID, NUMERICOID}, "F:numeric_sub", NULL },
	{ "numeric_mul", 2, {NUMERICOID, NUMERICOID}, "F:numeric_mul", NULL },
	{ "numeric_uplus",  1, {NUMERICOID}, "F:numeric_uplus", NULL },
	{ "numeric_uminus", 1, {NUMERICOID}, "F:numeric_uminus", NULL },
	{ "numeric_abs",    1, {NUMERICOID}, "F:numeric_abs", NULL },
	/* Numeric comparison */
	{ "numeric_eq", 2, {NUMERICOID, NUMERICOID}, "F:numeric_eq", NULL },
	{ "numeric_ne", 2, {NUMERICOID, NUMERICOID}, "F:numeric_ne", NULL },
	{ "numeric_lt", 2, {NUMERICOID, NUMERICOID}, "F:numeric_lt", NULL },
	{ "numeric_le", 2, {NUMERICOID, NUMERICOID}, "F:numeric_le", NULL },
	{ "numeric_gt", 2, {NUMERICOID, NUMERICOID}, "F:numeric_gt", NULL },
	{ "numeric_ge", 2, {NUMERICOID, NUMERICOID}, "F:numeric_ge", NULL },
	{ "numeric_cmp", 2, {NUMERICOID, NUMERICOID}, "F:numeric_cmp", NULL },
#endif
};

static devfunc_catalog_t devfunc_timelib_catalog[] = {
	/* Type cast functions */
	{ "date", 1, {DATEOID}, "c:", NULL },
	{ "date", 1, {TIMESTAMPOID}, "F:timestamp_date", NULL },
	{ "time", 1, {TIMESTAMPOID}, "F:timestamp_time", NULL },
	{ "time", 1, {TIMEOID}, "c:", NULL },
	{ "timestamp", 1, {TIMESTAMPOID}, "c:", NULL },
	{ "timestamp", 1, {DATEOID}, "F:date_timestamp", NULL },
	/* timedata operators */
	{ "date_pli", 2, {DATEOID, INT4OID}, "F:date_pli", NULL },
	{ "date_mii", 2, {DATEOID, INT4OID}, "F:date_mii", NULL },
	{ "date_mi", 2, {DATEOID, DATEOID}, "F:date_mi", NULL },
	{ "datetime_pl", 2, {DATEOID, TIMEOID}, "F:datetime_pl", NULL },
	{ "integer_pl_date", 2, {INT4OID, DATEOID}, "F:integer_pl_date", NULL },
	//{ "time_mi_time", 2, {TIMEOID, TIMEOID}, "F:time_mi_time", NULL },
	{ "timedate_pl", 2, {TIMEOID, DATEOID}, "F:timedata_pl", NULL },
	//{ "timestamp_mi", 2, {TIMESTAMPOID, TIMESTAMPOID}, "", NULL },
	/* comparison between date */
	{ "date_eq", 2, {DATEOID, DATEOID}, "b:==", NULL },
	{ "date_ne", 2, {DATEOID, DATEOID}, "b:!=", NULL },
	{ "date_lt", 2, {DATEOID, DATEOID}, "b:<", NULL },
	{ "date_le", 2, {DATEOID, DATEOID}, "b:<=", NULL },
	{ "date_gt", 2, {DATEOID, DATEOID}, "b:>", NULL },
	{ "date_ge", 2, {DATEOID, DATEOID}, "b:>=", NULL },
	{ "date_cmp", 2, {DATEOID, DATEOID}, NULL, devfunc_setup_comp },
	/* comparison of date and timestamp */
	{ "date_eq_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "F:date_eq_timestamp", NULL },
	{ "date_ne_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "F:date_ne_timestamp", NULL },
	{ "date_lt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "F:date_lt_timestamp", NULL },
	{ "date_le_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "F:date_le_timestamp", NULL },
	{ "date_gt_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "F:date_gt_timestamp", NULL },
	{ "date_ge_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "F:date_ge_timestamp", NULL },
	{ "date_cmp_timestamp", 2, {DATEOID, TIMESTAMPOID},
	  "F:date_cmp_timestamp", NULL },
	/* comparion between time */
	{ "time_eq", 2, {TIMEOID, TIMEOID}, "b:==", NULL },
	{ "time_ne", 2, {TIMEOID, TIMEOID}, "b:!=", NULL },
	{ "time_lt", 2, {TIMEOID, TIMEOID}, "b:<", NULL },
	{ "time_le", 2, {TIMEOID, TIMEOID}, "b:<=", NULL },
	{ "time_gt", 2, {TIMEOID, TIMEOID}, "b:>", NULL },
	{ "time_ge", 2, {TIMEOID, TIMEOID}, "b:>=", NULL },
	{ "time_cmp", 2, {TIMEOID, TIMEOID}, NULL, devfunc_setup_comp },
	/* comparison between timestamp */
	{ "timestamp_eq", 2, {TIMESTAMPOID, TIMESTAMPOID}, "b:==", NULL},
	{ "timestamp_ne", 2, {TIMESTAMPOID, TIMESTAMPOID}, "b:!=", NULL},
	{ "timestamp_lt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "b:<", NULL},
	{ "timestamp_le", 2, {TIMESTAMPOID, TIMESTAMPOID}, "b:<=", NULL},
	{ "timestamp_gt", 2, {TIMESTAMPOID, TIMESTAMPOID}, "b:>", NULL},
	{ "timestamp_ge", 2, {TIMESTAMPOID, TIMESTAMPOID}, "b:>=", NULL},
	{ "timestamp_cmp", 2, {TIMESTAMPOID, TIMESTAMPOID},
	  NULL, devfunc_setup_comp },
	/* comparison of timestamp and date */
	{ "timestamp_eq_date", 2, {TIMESTAMPOID, DATEOID},
	  "F:timestamp_eq_date", NULL},
	{ "timestamp_ne_date", 2, {TIMESTAMPOID, DATEOID},
	  "F:timestamp_ne_date", NULL},
	{ "timestamp_lt_date", 2, {TIMESTAMPOID, DATEOID},
	  "F:timestamp_lt_date", NULL},
	{ "timestamp_le_date", 2, {TIMESTAMPOID, DATEOID},
	  "F:timestamp_le_date", NULL},
	{ "timestamp_gt_date", 2, {TIMESTAMPOID, DATEOID},
	  "F:timestamp_gt_date", NULL},
	{ "timestamp_ge_date", 2, {TIMESTAMPOID, DATEOID},
	  "F:timestamp_ge_date", NULL},
	{ "timestamp_cmp_date", 2, {TIMESTAMPOID, DATEOID},
	  "F:timestamp_cmp_date", NULL},
};

static devfunc_catalog_t devfunc_textlib_catalog[] = {
	{ "bpchareq", 2, {BPCHAROID,BPCHAROID}, "F:bpchareq", NULL },
	{ "bpcharne", 2, {BPCHAROID,BPCHAROID}, "F:bpcharne", NULL },
	{ "bpcharlt", 2, {BPCHAROID,BPCHAROID}, "F:bpcharlt", NULL },
	{ "bpcharle", 2, {BPCHAROID,BPCHAROID}, "F:bpcharle", NULL },
	{ "bpchargt", 2, {BPCHAROID,BPCHAROID}, "F:bpchargt", NULL },
	{ "bpcharge", 2, {BPCHAROID,BPCHAROID}, "F:bpcharge", NULL },
	{ "bpcharcmp", 2, {BPCHAROID, BPCHAROID}, "F:bpcharcmp", NULL },
	{ "texteq", 2, {TEXTOID, TEXTOID}, "F:texteq", NULL  },
	{ "textne", 2, {TEXTOID, TEXTOID}, "F:textne", NULL  },
	{ "text_lt", 2, {TEXTOID, TEXTOID}, "F:text_lt", NULL  },
	{ "text_le", 2, {TEXTOID, TEXTOID}, "F:text_le", NULL  },
	{ "text_gt", 2, {TEXTOID, TEXTOID}, "F:text_gt", NULL  },
	{ "text_ge", 2, {TEXTOID, TEXTOID}, "F:text_ge", NULL  },
	{ "bttextcmp",   2, {TEXTOID, TEXTOID}, "F:text_cmp", NULL },
};

static void
devfunc_setup_mul_oper(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype1 = linitial(entry->func_args);
	devtype_info   *dtype2 = lsecond(entry->func_args);
	StringInfoData	str;

	initStringInfo(&str);

	Assert(procat->func_nargs == 2);
	entry->func_name = pstrdup(procat->func_name);
	appendStringInfo(
		&str,
		"static pg_%s_t\n"
		"pgfn_%s(__private int *errcode, pg_%s_t arg1, pg_%s_t arg2)\n"
		"{\n"
		"  pg_%s_t result;\n"
		"\n"
		"  result.isnull = arg1.isnull | arg2.isnull;\n"
		"  if (!result.isnull)\n"
		"  {\n"
		"    result.value = (%s)(arg1.value * arg2.value);\n",
		entry->func_rettype->type_name,
		entry->func_name,
        dtype1->type_name,
        dtype2->type_name,
        entry->func_rettype->type_name,
		entry->func_rettype->type_base);

	/* overflow check depends on data types */
	if (strcmp(procat->func_name, "int2mul") == 0)
	{
		appendStringInfo(
			&str,
			"    if (result.value < SHRT_MIN || result.value > SHRT_MAX)\n"
			"      STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n");
	}
	else if (strcmp(procat->func_name, "int4mul") == 0 ||
			 strcmp(procat->func_name, "int8mul") == 0)
	{
		appendStringInfo(
			&str,
			"    if (arg2.value != 0 &&\n"
			"        ((arg2.value == -1 && arg1.value < 0 && result.value < 0) ||\n"
			"         (result.value / arg2.value != arg1.value)))\n"
			"      STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n");
	}
	else if (strcmp(procat->func_name, "int24mul") == 0 ||
			 strcmp(procat->func_name, "int28mul") == 0 ||
			 strcmp(procat->func_name, "int48mul") == 0)
	{
		appendStringInfo(
			&str,
			"    if (arg2.value != 0 &&\n"
			"        result.value / arg2.value != arg1.value)\n"
			"      STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n");
	}
	else if (strcmp(procat->func_name, "int42mul") == 0 ||
			 strcmp(procat->func_name, "int82mul") == 0 ||
			 strcmp(procat->func_name, "int84mul") == 0)
	{
		appendStringInfo(
			&str,
			"    if (arg1.value != 0\n"
			"        result.value / arg1.value != arg2.value)\n"
			"      STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n");
	}
	else if (strcmp(procat->func_name, "float4mul") == 0 ||
			 strcmp(procat->func_name, "float48mul") == 0 ||
			 strcmp(procat->func_name, "float48mul") == 0 ||
			 strcmp(procat->func_name, "float8mul") == 0)
	{
		appendStringInfo(
			&str,
			"    if ((isinf(result.value) &&\n"
			"         (!isinf(arg1.value) || !isinf(arg2.value))) ||\n"
			"        (result.value == 0 && arg1.value != 0 && arg2.value != 0))\n"
			"      STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n");
	}
	else
	{
		appendStringInfo(&str, "    /* no overflow checks */\n");
	}
	appendStringInfo(&str,
					 "  }\n"
					 "  return result;\n"
					 "}\n");
	entry->func_decl = str.data;
}

static void
devfunc_setup_div_oper(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype1 = linitial(entry->func_args);
	devtype_info   *dtype2 = lsecond(entry->func_args);

	Assert(procat->func_nargs == 2);
	entry->func_name = pstrdup(procat->func_name);
	entry->func_decl
		= psprintf("static pg_%s_t pgfn_%s"
				   "(__private int *errcode,"
				   " pg_%s_t arg1, pg_%s_t arg2)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    if (arg2.value == %s)\n"
				   "    {\n"
				   "        result.isnull = true;\n"
				   "        STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n"
				   "    }\n"
				   "    else\n"
				   "    {\n"
				   "        result.value = (%s)(arg1.value / arg2.value);\n"
				   "        result.isnull = arg1.isnull | arg2.isnull;\n"
				   "    }\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_name,
				   dtype1->type_name,
				   dtype2->type_name,
				   entry->func_rettype->type_name,
				   procat->func_template,	/* 0 or 0.0 */
				   entry->func_rettype->type_base);
}

static void
devfunc_setup_const(devfunc_info *entry, devfunc_catalog_t *procat)
{
	Assert(procat->func_nargs == 0);
	entry->func_name = pstrdup(procat->func_name);
	entry->func_decl
		= psprintf("static pg_%s_t pgfn_%s(__private int *errcode)\n"
				   "{\n"
				   "  pg_%s_t result;\n"
				   "  result.isnull = false;\n"
				   "  result.value = %s;\n"
				   "  return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_name,
				   entry->func_rettype->type_name,
				   procat->func_template);
}

/*
 * devfunc_setup_comp
 *
 * constructor of compare-function for binary comparable data types
 */
static void
devfunc_setup_comp(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype1;
	devtype_info   *dtype2;

	Assert(procat->func_nargs == 2);
	dtype1 = linitial(entry->func_args);
	dtype2 = lsecond(entry->func_args);

	entry->func_name = pstrdup(procat->func_name);
	entry->func_decl
		= psprintf("static pg_%s_t\n"
				   "pgfn_%s(__private int *errcode,\n"
				   "        pg_%s_t arg1, pg_%s_t arg2)\n"
                   "{\n"
                   "  pg_%s_t result;\n"
				   "  if (arg1.isnull || arg2.isnull)\n"
				   "    result.isnull = true;\n"
				   "  else\n"
				   "  {\n"
                   "    result.isnull = false;\n"
				   "    if (arg1.value > arg2.value)\n"
				   "      result.value = 1;\n"
				   "    else if (arg1.value < arg2.value)\n"
				   "      result.value = -1;\n"
				   "    else\n"
				   "      result.value = 0;\n"
				   "  }\n"
				   "  return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_name,
				   dtype1->type_name,
				   dtype2->type_name,
				   entry->func_rettype->type_name);
}

static void
devfunc_setup_cast(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype = linitial(entry->func_args);

	Assert(procat->func_nargs == 1);
	entry->func_name = pstrdup(procat->func_name);
	entry->func_decl
		= psprintf("static pg_%s_t pgfn_%s"
				   "(__private int *errcode, pg_%s_t arg)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    result.value  = (%s)arg.value;\n"
				   "    result.isnull = arg.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_name,
				   dtype->type_name,
				   entry->func_rettype->type_name,
				   entry->func_rettype->type_base);
}

static void
devfunc_setup_oper_both(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype1 = linitial(entry->func_args);
	devtype_info   *dtype2 = lsecond(entry->func_args);

	Assert(procat->func_nargs == 2);
	entry->func_name = pstrdup(procat->func_name);
	entry->func_decl
		= psprintf("static pg_%s_t pgfn_%s"
				   "(__private int *errcode, pg_%s_t arg1, pg_%s_t arg2)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    result.value = (%s)(arg1.value %s arg2.value);\n"
				   "    result.isnull = arg1.isnull | arg2.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_name,
				   dtype1->type_name,
				   dtype2->type_name,
				   entry->func_rettype->type_name,
				   entry->func_rettype->type_base,
				   procat->func_template + 2);
}

static void
devfunc_setup_oper_either(devfunc_info *entry, devfunc_catalog_t *procat)
{
	devtype_info   *dtype = linitial(entry->func_args);
	const char	   *templ = procat->func_template;

	Assert(procat->func_nargs == 1);
	entry->func_name = pstrdup(procat->func_name);
	entry->func_decl
		= psprintf("static pg_%s_t pgfn_%s"
				   "(__private int *errcode, pg_%s_t arg)\n"
				   "{\n"
				   "    pg_%s_t result;\n"
				   "    result.value = (%s)(%sarg%s);\n"
				   "    result.isnull = arg.isnull;\n"
				   "    return result;\n"
				   "}\n",
				   entry->func_rettype->type_name,
				   entry->func_name,
				   dtype->type_name,
				   entry->func_rettype->type_name,
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

	entry->func_name = pstrdup(procat->func_name);
	/* declaration */
	initStringInfo(&str);
	appendStringInfo(&str, "static pg_%s_t pgfn_%s(",
					 entry->func_rettype->type_name,
					 entry->func_name);
	appendStringInfo(&str, "__private int *errcode");
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
	entry->func_name = pstrdup(fn_name);
	appendStringInfo(&str, "static pg_%s_t pgfn_%s("
					 "__private int *errcode",
					 dtype->type_name, fn_name);
	for (i=0; i < fn_nargs; i++)
		appendStringInfo(&str, ", pg_%s_t arg%u",
						 dtype->type_name, i+1);
	appendStringInfo(&str, ")\n"
					 "{\n"
					 "  pg_%s_t result;\n"
					 "  result.isnull = ",
					 dtype->type_name);
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
			  DEVFUNC_NEEDS_NUMERIC },
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
							= pgstrom_devtype_lookup(func_argtypes[k]);
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
						entry->func_name = pstrdup(procat->func_template + 2);
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

devtype_info *
pgstrom_devtype_lookup_and_track(Oid type_oid, codegen_context *context)
{
	devtype_info   *dtype = pgstrom_devtype_lookup(type_oid);
	if (dtype)
		context->type_defs = list_append_unique_ptr(context->type_defs, dtype);
	return dtype;
}

static void
pgstrom_devfunc_track(devfunc_info *dfunc, codegen_context *context)
{
	ListCell   *cell;

	context->func_defs = list_append_unique_ptr(context->func_defs, dfunc);
	context->extra_flags |= (dfunc->func_flags & DEVFUNC_INCL_FLAGS);

	/* track function arguments and result type */
	context->type_defs = list_append_unique_ptr(context->type_defs,
												dfunc->func_rettype);
	foreach (cell, dfunc->func_args)
		context->type_defs = list_append_unique_ptr(context->type_defs,
													lfirst(cell));
}

devfunc_info *
pgstrom_devfunc_lookup_and_track(Oid func_oid, codegen_context *context)
{
	devfunc_info   *dfunc = pgstrom_devfunc_lookup(func_oid);

	if (dfunc)
		pgstrom_devfunc_track(dfunc, context);

	return dfunc;
}

static bool
codegen_expression_walker(Node *node, codegen_context *context)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	ListCell	   *cell;

	if (node == NULL)
		return true;

	if (IsA(node, Const))
	{
		Const  *con = (Const *) node;
		cl_uint	index = 0;

		if (!devtype_runnable_collation(con->constcollid) ||
			!pgstrom_devtype_lookup_and_track(con->consttype, context))
			return false;

		foreach (cell, context->used_params)
		{
			if (equal(node, lfirst(cell)))
			{
				appendStringInfo(&context->str, "KPARAM_%u", index);
				context->param_refs =
					bms_add_member(context->param_refs, index);
				return true;
			}
			index++;
		}
		context->used_params = lappend(context->used_params,
									   copyObject(node));
		index = list_length(context->used_params) - 1;
		appendStringInfo(&context->str, "KPARAM_%u", index);
		context->param_refs =
			bms_add_member(context->param_refs, index);
		return true;
	}
	else if (IsA(node, Param))
	{
		Param  *param = (Param *) node;
		int		index = 0;

		if (!devtype_runnable_collation(param->paramcollid) ||
			param->paramkind != PARAM_EXTERN ||
			!pgstrom_devtype_lookup_and_track(param->paramtype, context))
			return false;

		foreach (cell, context->used_params)
		{
			if (equal(node, lfirst(cell)))
			{
				appendStringInfo(&context->str, "KPARAM_%u", index);
				context->param_refs =
					bms_add_member(context->param_refs, index);
				return true;
			}
			index++;
		}
		context->used_params = lappend(context->used_params,
									   copyObject(node));
		index = list_length(context->used_params) - 1;
		appendStringInfo(&context->str, "KPARAM_%u", index);
		context->param_refs = bms_add_member(context->param_refs, index);
		return true;
	}
	else if (IsA(node, Var))
	{
		Var	   *var = (Var *) node;

		if (!devtype_runnable_collation(var->varcollid) ||
			!pgstrom_devtype_lookup_and_track(var->vartype, context))
			return false;

		appendStringInfo(&context->str, "%s_%u",
						 context->var_label,
						 var->varattno);
		context->used_vars = list_append_unique(context->used_vars,
												copyObject(node));
		return true;
	}
	else if (IsA(node, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) node;

		/* no collation support */
		if (!devtype_runnable_collation(func->funccollid) ||
			!devtype_runnable_collation(func->inputcollid))
			return false;

		dfunc = pgstrom_devfunc_lookup_and_track(func->funcid, context);
		if (!func)
			return false;
		appendStringInfo(&context->str,
						 "pgfn_%s(errcode", dfunc->func_name);

		foreach (cell, func->args)
		{
			appendStringInfo(&context->str, ", ");
			if (!codegen_expression_walker(lfirst(cell), context))
				return false;
		}
		appendStringInfoChar(&context->str, ')');

		return true;
	}
	else if (IsA(node, OpExpr) ||
			 IsA(node, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) node;

		/* no collation support */
		if (!devtype_runnable_collation(op->opcollid) ||
			!devtype_runnable_collation(op->inputcollid))
			return false;

		dfunc = pgstrom_devfunc_lookup_and_track(get_opcode(op->opno),
												 context);
		if (!dfunc)
			return false;
		appendStringInfo(&context->str,
						 "pgfn_%s(errcode", dfunc->func_name);

		foreach (cell, op->args)
		{
			appendStringInfo(&context->str, ", ");
			if (!codegen_expression_walker(lfirst(cell), context))
				return false;
		}
		appendStringInfoChar(&context->str, ')');

		return true;
	}
	else if (IsA(node, NullTest))
	{
		NullTest   *nulltest = (NullTest *) node;
		Oid			typeoid = exprType((Node *)nulltest->arg);
		const char *func_name;

		if (nulltest->argisrow)
			return false;

		dtype = pgstrom_devtype_lookup_and_track(typeoid, context);
		if (!dtype)
			return false;

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
		appendStringInfo(&context->str, "pgfn_%s_%s(errcode, ",
						 dtype->type_name, func_name);
		if (!codegen_expression_walker((Node *) nulltest->arg, context))
			return false;
		appendStringInfoChar(&context->str, ')');

		return true;
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
		appendStringInfo(&context->str, "pgfn_%s(errcode, ", func_name);
		if (!codegen_expression_walker((Node *) booltest->arg, context))
			return false;
		appendStringInfoChar(&context->str, ')');
		return true;
	}
	else if (IsA(node, BoolExpr))
	{
		BoolExpr   *b = (BoolExpr *) node;

		if (b->boolop == NOT_EXPR)
		{
			Assert(list_length(b->args) == 1);
			appendStringInfo(&context->str, "pg_boolop_not(errcode, ");
			if (!codegen_expression_walker(linitial(b->args), context))
				return false;
			appendStringInfoChar(&context->str, ')');
		}
		else if (b->boolop == AND_EXPR || b->boolop == OR_EXPR)
		{
			char	namebuf[NAMEDATALEN];
			int		nargs = list_length(b->args);
			Oid	   *argtypes = alloca(sizeof(Oid) * nargs);
			int		i;

			if (b->boolop == AND_EXPR)
				snprintf(namebuf, sizeof(namebuf), "boolop_and_%u", nargs);
			else
				snprintf(namebuf, sizeof(namebuf), "boolop_or_%u", nargs);

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
			context->func_defs = list_append_unique_ptr(context->func_defs,
														dfunc);
			context->extra_flags |= (dfunc->func_flags & DEVFUNC_INCL_FLAGS);

			appendStringInfo(&context->str, "pgfn_%s(errcode",
							 dfunc->func_name);
			foreach (cell, b->args)
			{
				Assert(exprType(lfirst(cell)) == BOOLOID);
				appendStringInfo(&context->str, ", ");
				if (!codegen_expression_walker(lfirst(cell), context))
					return false;
			}
			appendStringInfoChar(&context->str, ')');
		}
		else
			elog(ERROR, "unrecognized boolop: %d", (int) b->boolop);
		return true;
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
		return codegen_expression_walker((Node *)relabel->arg, context);
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
					return false;
				dfunc = pgstrom_devfunc_lookup_and_track(dtype->type_eqfunc,
														 context);
				if (!dfunc)
					return false;
				appendStringInfo(&context->str,
								 "EVAL(pgfn_%s(", dfunc->func_name);
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
		return true;
	}
	Assert(false);
	return false;
}

char *
pgstrom_codegen_expression(Node *expr, codegen_context *context)
{
	codegen_context	walker_context;

	initStringInfo(&walker_context.str);
	walker_context.type_defs = list_copy(context->type_defs);
	walker_context.func_defs = list_copy(context->func_defs);
	walker_context.used_params = list_copy(context->used_params);
	walker_context.used_vars = list_copy(context->used_vars);
	walker_context.param_refs = bms_copy(context->param_refs);
	walker_context.var_label  = context->var_label;
	walker_context.kds_label  = context->kds_label;
	walker_context.ktoast_label  = context->ktoast_label;
	walker_context.kds_index_label = context->kds_index_label;
	walker_context.extra_flags = context->extra_flags;

	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = (Node *)linitial((List *)expr);
		else
			expr = (Node *)make_andclause((List *)expr);
	}
	if (!codegen_expression_walker(expr, &walker_context))
		return NULL;

	context->type_defs = walker_context.type_defs;
	context->func_defs = walker_context.func_defs;
	context->used_params = walker_context.used_params;
	context->used_vars = walker_context.used_vars;
	context->param_refs = walker_context.param_refs;
	/* no need to write back xxx_label fields because read-only */
	context->extra_flags = walker_context.extra_flags;

	return walker_context.str.data;
}

/*
 * pgstrom_codegen_type_declarations
 */
char *
pgstrom_codegen_type_declarations(codegen_context *context)
{
	StringInfoData	str;
	ListCell	   *cell;

	initStringInfo(&str);
	foreach (cell, context->type_defs)
	{
		devtype_info   *dtype = lfirst(cell);
		int		i, n;

		/* prevent duplicated definition by built-in functions */
		appendStringInfo(&str, "#ifndef PG_");
		n = strlen(dtype->type_name);
		for (i=0; i < n; i++)
			appendStringInfoChar(&str, toupper(dtype->type_name[i]));
		appendStringInfo(&str, "_TYPE_DEFINED\n");

		if (dtype->type_flags & DEVTYPE_IS_VARLENA)
			appendStringInfo(&str, "STROMCL_VARLENA_TYPE_TEMPLATE(%s)\n",
							 dtype->type_name);
		else
			appendStringInfo(&str, "STROMCL_SIMPLE_TYPE_TEMPLATE(%s,%s)\n",
							 dtype->type_name, dtype->type_base);
		appendStringInfo(&str, "#endif\n");
	}
	appendStringInfoChar(&str, '\n');

	return str.data;
}

/*
 * pgstrom_codegen_func_declarations
 */
char *
pgstrom_codegen_func_declarations(codegen_context *context)
{
	StringInfoData	str;
	ListCell	   *cell;

	initStringInfo(&str);
	foreach (cell, context->func_defs)
	{
		devfunc_info   *dfunc = lfirst(cell);

		if (dfunc->func_decl)
			appendStringInfo(&str, "%s\n", dfunc->func_decl);
	}
	return str.data;
}

/*
 * pgstrom_codegen_param_declarations
 */
char *
pgstrom_codegen_param_declarations(codegen_context *context,
								   Bitmapset *param_refs)
{
	StringInfoData	str;
	ListCell	   *cell;
	devtype_info   *dtype;
	int				index = 0;

	initStringInfo(&str);
	foreach (cell, context->used_params)
	{
		if (!bms_is_member(index, param_refs))
			goto lnext;

		if (IsA(lfirst(cell), Const))
		{
			Const  *con = lfirst(cell);

			dtype = pgstrom_devtype_lookup(con->consttype);
			Assert(dtype != NULL);

			appendStringInfo(
				&str,
				"  pg_%s_t KPARAM_%u = pg_%s_param(kparams,errcode,%d);\n",
				dtype->type_name, index, dtype->type_name, index);
		}
		else if (IsA(lfirst(cell), Param))
		{
			Param  *param = lfirst(cell);

			dtype = pgstrom_devtype_lookup(param->paramtype);
			Assert(dtype != NULL);
			appendStringInfo(
				&str,
				"  pg_%s_t KPARAM_%u = pg_%s_param(kparams,errcode,%d);\n",
				dtype->type_name, index, dtype->type_name, index);
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(lfirst(cell)));
	lnext:
		index++;
	}
	return str.data;
}

/*
 * pgstrom_codegen_var_declarations
 */
char *
pgstrom_codegen_var_declarations(codegen_context *context)
{
	StringInfoData	str;
	ListCell	   *cell;

	initStringInfo(&str);
	foreach (cell, context->used_vars)
	{
		Var			   *var = lfirst(cell);
		devtype_info   *dtype = pgstrom_devtype_lookup(var->vartype);

		Assert(dtype != NULL);
		appendStringInfo(
			&str,
			"  pg_%s_t %s_%u = pg_%s_vref(%s,%s,errcode,%u,%s);\n",
			dtype->type_name,
			context->var_label,
			var->varattno,
			dtype->type_name,
			context->kds_label,
			context->ktoast_label,
			var->varattno - 1,
			context->kds_index_label);
	}
	return str.data;
}

/*
 * pgstrom_codegen_bulk_var_declarations
 *
 * It adds variable references on bulk loading. Note that this routine
 * assumes varattno points correct (pseudo) attribute number, so we may
 * needs to revise the design in the future version.
 * Right now, only GpuPreAgg can have bulk-loading from GpuHashJoin,
 * and it will be added all query-planning stuff was done.
 */
char *
pgstrom_codegen_bulk_var_declarations(codegen_context *context,
									  Plan *outer_plan,
									  Bitmapset *attr_refs)
{
	List		   *subplan_vars = NIL;
	ListCell	   *lc;
	StringInfoData	buf1;
	StringInfoData	buf2;
	devtype_info   *dtype;

	initStringInfo(&buf1);
	initStringInfo(&buf2);
	foreach (lc, outer_plan->targetlist)
	{
		TargetEntry *tle = lfirst(lc);
		int			x = tle->resno - FirstLowInvalidHeapAttributeNumber;
		Oid			type_oid;

		if (!bms_is_member(x, attr_refs))
			continue;

		type_oid = exprType((Node *) tle->expr);
		dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
		if (IsA(tle->expr, Var))
		{
			Var	   *outer_var = (Var *) tle->expr;

			Assert(outer_var->vartype == type_oid);
			appendStringInfo(
				&buf2,
				"  pg_%s_t %s_%u = pg_%s_vref(%s,%s,errcode,%u,%s);\n",
				dtype->type_name,
				context->var_label,
				tle->resno,
				dtype->type_name,
				context->kds_label,
				context->ktoast_label,
				outer_var->varattno - 1,
				context->kds_index_label);
		}
		else
		{
			const char *saved_var_label = context->var_label;
			List	   *saved_used_vars = context->used_vars;
			char	   *temp;

			context->var_label = "OVAR";
			context->used_vars = subplan_vars;

			temp = pgstrom_codegen_expression((Node *)tle->expr, context);
			appendStringInfo(&buf2, "  pg_%s_t %s_%u = %s;\n",
							 dtype->type_name,
							 saved_var_label,
							 tle->resno,
							 temp);
			subplan_vars = context->used_vars;
			context->var_label = saved_var_label;
			context->used_vars = saved_used_vars;
		}
	}

	foreach (lc, subplan_vars)
	{
		Var	   *var = lfirst(lc);

		dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
		appendStringInfo(
			&buf1,
			"  pg_%s_t OVAR_%u = pg_%s_vref(%s,%s,errcode,%u,%s);\n",
			dtype->type_name,
			var->varattno,
			dtype->type_name,
			context->kds_label,
			context->ktoast_label,
			var->varattno - 1,
			context->kds_index_label);
	}
	appendStringInfo(&buf1, "%s", buf2.data);
	pfree(buf2.data);

	return buf1.data;
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
		Const  *con = (Const *) expr;

		if (!devtype_runnable_collation(con->constcollid) ||
			!pgstrom_devtype_lookup(con->consttype))
			return false;
		return true;
	}
	else if (IsA(expr, Param))
	{
		Param  *param = (Param *) expr;

		if (!devtype_runnable_collation(param->paramcollid) ||
			param->paramkind != PARAM_EXTERN ||
			!pgstrom_devtype_lookup(param->paramtype))
			return false;
		return true;
	}
	else if (IsA(expr, Var))
	{
		Var	   *var = (Var *) expr;

		if (!devtype_runnable_collation(var->varcollid) ||
			!pgstrom_devtype_lookup(var->vartype))
			return false;
		return true;
	}
	else if (IsA(expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) expr;

		if (!devtype_runnable_collation(func->funccollid) ||
			!devtype_runnable_collation(func->inputcollid))
			return false;
		if (!pgstrom_devfunc_lookup(func->funcid))
			return false;
		return pgstrom_codegen_available_expression((Expr *) func->args);
	}
	else if (IsA(expr, OpExpr) || IsA(expr, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		if (!devtype_runnable_collation(op->opcollid) ||
			!devtype_runnable_collation(op->inputcollid))
			return false;
		if (!pgstrom_devfunc_lookup(get_opcode(op->opno)))
			return false;
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
	else if (IsA(expr, RelabelType))
	{
		RelabelType *relabel = (RelabelType *) expr;

		if (!devtype_runnable_collation(relabel->resultcollid))
			return false;
		if (!pgstrom_devtype_lookup(relabel->resulttype))
			return false;
		return pgstrom_codegen_available_expression((Expr *) relabel->arg);
	}
	else if (IsA(expr, CaseExpr))
	{
		CaseExpr   *caseexpr = (CaseExpr *) expr;
		ListCell   *cell;

		if (!devtype_runnable_collation(caseexpr->casecollid))
			return false;
		if (!pgstrom_devtype_lookup(caseexpr->casetype))
			return false;

		if (caseexpr->arg)
		{
			Oid				expr_type = exprType((Node *)caseexpr->arg);
			devtype_info   *dtype = pgstrom_devtype_lookup(expr_type);

			if (!OidIsValid(dtype->type_eqfunc) ||
				!pgstrom_devfunc_lookup(dtype->type_eqfunc))
				return false;
		}

		foreach (cell, caseexpr->args)
		{
			CaseWhen   *casewhen = lfirst(cell);

			Assert(IsA(casewhen, CaseWhen));
			if (exprType((Node *)casewhen->expr) !=
				(caseexpr->arg ? exprType((Node *)caseexpr->arg) : BOOLOID))
				return false;
			if (!pgstrom_codegen_available_expression(casewhen->expr))
				return false;
			if (!pgstrom_codegen_available_expression(casewhen->result))
				return false;
		}
		if (!pgstrom_codegen_available_expression((Expr *) caseexpr->defresult))
			return false;

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
pgstrom_init_codegen_context(codegen_context *context)
{
	memset(context, 0, sizeof(codegen_context));

	context->var_label = "KVAR";
	context->kds_label = "kds";
	context->ktoast_label = "ktoast";
	context->kds_index_label = "kds_index";
}

void
pgstrom_init_codegen(void)
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
