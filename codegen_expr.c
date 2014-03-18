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
	{ ABSTIMEOID,		"cl_int" },
	{ RELTIMEOID,		"cl_int" },
	{ TIMESTAMPOID,		"cl_long" },
	{ TIMESTAMPTZOID,	"cl_long" },

	/* variable length datatypes */
	{ CHAROID,			"varlena" },
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

static struct {
	const char *func_name;
	int			func_nargs;
	Oid			func_argtypes[4];
	const char *func_template;	/* a template string if simple function */
	void	  (*func_callback)(devfunc_info *devfunc, Form_pg_proc proc);
} devfunc_catalog[] = {
	/*
	 * Type case functions
	 */



	/* '+' : add operators */
	/* '-' : subtract operators */
	/* '*' : mutiply operators */
	/* '/' : divide operators */
	/* '%' : reminder operators */
	/* '+' : unary plus operators */
	/* '-' : unary minus operators */
	/* '@' : absolute value operators */
	/* '=' : equal operators */
	/* '<>' : not equal operators */
	/* '>' : equal operators */
	/* '<' : equal operators */
	/* '>=' : relational greater-than or equal-to */
	/* '<=' : relational greater-than or equal-to */
	/* '&' : bitwise and */
	/* '|'  : bitwise or */
	/* '#'  : bitwise xor */
	/* '~'  : bitwise not operators */
	/* '>>' : right shift */
	/* '<<' : left shift */
	/*
     * Mathmatical functions
     */
	/*
     * Trigonometric function
     */

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
