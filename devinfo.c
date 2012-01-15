/*
 * devinfo.c
 *
 * Routines to reference properties of GPU devices, and catalog of
 * device functions and types.
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "access/hash.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/syscache.h"
#include "pg_strom.h"

/* ------------------------------------------------------------
 *
 * Catalog of supported device types
 *
 * ------------------------------------------------------------
 */
#define DEVFUNC_VARREF_TEMPLATE(vtype)									\
	#vtype " varref_" #vtype "(unsigned char *errors, "					\
	"unsigned char bitmask, unsigned char isnull, " #vtype " value)\n"	\
	"{\n"																\
	"    if (bitmask & isnull)\n"										\
	"        *errors |= bitmask;\n"										\
	"    return value;\n"												\
	"}\n"

#define DEVFUNC_CONREF_INT(vtype)										\
	"#define conref_" #vtype "(conval)  ((" #vtype ")(conval))\n"
#define DEVFUNC_CONREF_FP32(vtype)										\
	"#define conref_" #vtype "(conval)  __int_as_float((int)(conval))\n"
#define DEVFUNC_CONREF_FP64(vtype)										\
	"#define conref_" #vtype "(conval)  __longlong_as_double(conval)\n"

static List		   *devtype_info_slot[512];

static struct {
	Oid		type_oid;
	char   *type_ident;
	char   *type_source;
	char   *type_varref;
	char   *type_conref;
	uint32	type_flags;
} device_type_catalog[] = {
	{ BOOLOID,		"char",		NULL,
	  DEVFUNC_VARREF_TEMPLATE(char),
	  DEVFUNC_CONREF_INT(char), 0 },
	{ INT2OID,		"short",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(short),
	  DEVFUNC_CONREF_INT(short), 0 },
	{ INT4OID,		"int",		NULL,
	  DEVFUNC_VARREF_TEMPLATE(int),
	  DEVFUNC_CONREF_INT(int), 0 },
	{ INT8OID,		"long",		NULL,
	  DEVFUNC_VARREF_TEMPLATE(long),
	  DEVFUNC_CONREF_INT(long), 0 },
	{ FLOAT4OID,	"float",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(float),
	  DEVFUNC_CONREF_FP32(float), 0 },
	{ FLOAT8OID,	"double",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(double),
	  DEVFUNC_CONREF_FP64(double),
	  DEVINFO_FLAGS_DOUBLE_FP },
};

PgStromDevTypeInfo *
pgstrom_devtype_lookup(Oid type_oid)
{
	PgStromDevTypeInfo *entry;
	HeapTuple		tuple;
	Form_pg_type	typeForm;
	MemoryContext	oldcxt;
	ListCell	   *cell;
	int				i, hash;

	hash = hash_uint32((uint32) type_oid) % lengthof(devtype_info_slot);
	foreach (cell, devtype_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->type_oid == type_oid)
			return (!entry->type_ident ? NULL : entry);
	}

	/*
	 * Not found, insert a new entry
	 */
	tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typeForm = (Form_pg_type) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

	entry = palloc0(sizeof(PgStromDevTypeInfo));
	entry->type_oid = type_oid;
	if (typeForm->typnamespace != PG_CATALOG_NAMESPACE)
		goto out;

	for (i=0; i < lengthof(device_type_catalog); i++)
	{
		if (device_type_catalog[i].type_oid == type_oid)
		{
			entry->type_ident  = device_type_catalog[i].type_ident;
			entry->type_source = device_type_catalog[i].type_source;
			entry->type_varref = device_type_catalog[i].type_varref;
			entry->type_conref = device_type_catalog[i].type_conref;
			entry->type_flags  = device_type_catalog[i].type_flags;
			break;
		}
	}
out:	
	devtype_info_slot[hash] = lappend(devtype_info_slot[hash], entry);
	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	return (!entry->type_ident ? NULL : entry);
}

/* ------------------------------------------------------------
 *
 * Catalog of supported device functions
 *
 * ------------------------------------------------------------
 */
#define DEVFUNC_INTxDIV_TEMPLATE(name,rtype,xtype,ytype)				\
	#rtype " " #name "(unsigned char *errors, unsigned char bitmap, "	\
	#xtype " x, " #ytype " y)\n"										\
	"{\n"																\
	"    " #rtype " result;\n"											\
	"    if (y == 0)\n"													\
	"        *errors |= bitmap;\n"										\
	"    result = x / y;\n"												\
	"    if (y == -1 && x < 0 && result <= 0)\n"						\
	"        *errors |= bitmap;\n"										\
	"    return result;\n"												\
	"}\n"

#define DEVFUNC_FPxDIV_TEMPLATE(name,rtype,xtype,ytype)					\
	#rtype " " #name "(unsigned char *errors, unsigned char bitmap, "	\
	#xtype " x, " #ytype " y)\n"										\
	"{\n"																\
	"    " #rtype " result;\n"											\
	"    if (y == 0)\n"													\
	"        *errors |= bitmap;\n"										\
	"    result = x / y;\n"												\
	"    if (isinf(result) && !isinf(x) && !isinf(y))\n"				\
	"        *errors |= bitmap;\n"										\
	"    return result;\n"												\
	"}\n"

#define DEVFUNC_INTxREMIND_TEMPLATE(name,rtype,xtype,ytype)			   \
	#rtype " " #name "(unsigned char *errors, unsigned char bitmap, "  \
	#xtype " x, " #ytype " y)\n"									   \
	"{\n"															   \
	"    if (y == 0)\n"												   \
	"        *errors |= bitmap;\n"									   \
	"    return x % y;\n"											   \
	"}\n"

#define DEVFUNC_FPSIGN_TEMPLATE(vtype)				\
	#vtype " sign_" #vtype "(" #vtype " value)\n"	\
	"{\n"											\
	"    if (value > 0.0)\n"						\
	"        return 1.0;\n"							\
	"    if (value < 0.0)\n"						\
	"        return -1.0;\n"						\
	"    return 0.0;\n"								\
	"}\n"

#define DEVFUNC_FPCOT_TEMPLATE(vtype)				\
	#vtype " cot_" #vtype "(" #vtype " value)\n"	\
	"{\n"											\
	"    return 1.0 / tan(value);\n"				\
	"}\n"

static List		   *devfunc_info_slot[1024];

static struct {
	/*
	 * Name and argument types of SQL function enables to identify
	 * a particular built-in function; The reason why we don't put
	 * F_XXX label here is some of functions does not have its own
	 * label.
	 */
	char   *func_name;
	int		func_nargs;
	Oid		func_argtypes[2];

	/*
	 * The func_kind is one of the following character:
	 *   'c', 'l', 'r', 'b', 'f' or 'F'.
	 *
	 * 'c' means SQL function is executable as a constant value on device.
	 *
	 * 'l' means SQL function is executable as a left-operator on device.
	 * It shall be extracted on the source code as:
	 *   <func_ident> <arg>  (e.g, -value)
	 * One special case is inline type-cast. It shall be described according
	 * to the manner of left-operator, such as: (double) value
	 *
	 * 'r' means SQL function is executable as a right-operator on device.
	 *
	 * 'b' means SQL function is executable as a both-operator on device.
	 * It shall be extracted on the source code as:
	 *   (<arg 1> <func_ident> <arg 2>)  (e.g, (v1 * v2))
	 *
	 * 'f' and 'F' means SQL function is also executable as device function
	 * on device. The only difference of them is whether it takes "errors"
	 * and "bitmask" for the first two arguments, or not.
	 * It is used to return an error status by self-defined functions, but
	 * nonsense towards built-in functions.
	 * The 'f' means a device function WITHOUT these two special arguments,
	 * and 'F' means one WITH these two special arguments
	 */
	char	func_kind;	/* 'f', 'l', 'r', 'b' or 'c' */

	/*
	 * identifier of function, operator or constant value
	 */
	char   *func_ident;

	/*
	 * declaration part of self-defined function, if exist.
	 */
	char   *func_source;

	/*
	 * set of DEVINFO_FLAGS_*
	 */
	uint32	func_flags;
} device_func_catalog[] = {
	/*
	 * cast of data types
	 *
	 * XXX - note that inline cast is writable using left-operator
	 * manner, like (uint2_t)X.
	 */
	{ "int2", 1, {INT4OID},   'l', "(short)", NULL, 0 },
	{ "int2", 1, {INT8OID},   'l', "(short)", NULL, 0 },
	{ "int2", 1, {FLOAT4OID}, 'l', "(short)", NULL, 0 },
	{ "int2", 1, {FLOAT8OID}, 'l', "(short)", NULL, 0 },
	{ "int4", 1, {INT2OID},   'l', "(int)", NULL, 0 },
	{ "int4", 1, {INT8OID},   'l', "(int)", NULL, 0 },
	{ "int4", 1, {FLOAT4OID}, 'l', "(int)", NULL, 0 },
	{ "int4", 1, {FLOAT8OID}, 'l', "(int)", NULL, 0 },
	{ "int8", 1, {INT2OID},   'l', "(long)", NULL, 0 },
	{ "int8", 1, {INT4OID},   'l', "(long)", NULL, 0 },
	{ "int8", 1, {FLOAT4OID}, 'l', "(long)", NULL, 0 },
	{ "int8", 1, {FLOAT8OID}, 'l', "(long)", NULL, 0 },
	{ "float4", 1, {INT2OID},   'l', "(float)",  NULL, 0 },
	{ "float4", 1, {INT4OID},   'l', "(float)",  NULL, 0 },
	{ "float4", 1, {INT8OID},   'l', "(float)",  NULL, 0 },
	{ "float4", 1, {FLOAT8OID}, 'l', "(float)",  NULL, 0 },
	{ "float8", 1, {INT2OID},   'l', "(double)",  NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8", 1, {INT4OID},   'l', "(double)",  NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8", 1, {INT8OID},   'l', "(double)",  NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8", 1, {FLOAT4OID}, 'l', "(double)",  NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '+'  : add operators */
	{ "int2pl",	 2, {INT2OID, INT2OID}, 'b', "+", NULL, 0 },
	{ "int24pl", 2, {INT2OID, INT4OID}, 'b', "+", NULL, 0 },
	{ "int28pl", 2, {INT2OID, INT8OID}, 'b', "+", NULL, 0 },
	{ "int42pl", 2, {INT4OID, INT2OID}, 'b', "+", NULL, 0 },
	{ "int4pl",  2, {INT4OID, INT4OID}, 'b', "+", NULL, 0 },
	{ "int48pl", 2, {INT4OID, INT8OID}, 'b', "+", NULL, 0 },
	{ "int82pl", 2, {INT8OID, INT2OID}, 'b', "+", NULL, 0 },
	{ "int84pl", 2, {INT8OID, INT4OID}, 'b', "+", NULL, 0 },
	{ "int8pl",  2, {INT8OID, INT8OID}, 'b', "+", NULL, 0 },
	{ "float4pl",  2, {FLOAT4OID, FLOAT4OID}, 'b', "+", NULL, 0 },
	{ "float48pl", 2, {FLOAT4OID, FLOAT8OID}, 'b', "+", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84pl", 2, {FLOAT8OID, FLOAT4OID}, 'b', "+", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8pl",  2, {FLOAT8OID, FLOAT8OID}, 'b', "+", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '-'  : subtract operators */
	{ "int2mi",	 2, {INT2OID, INT2OID}, 'b', "-", NULL, 0 },
	{ "int24mi", 2, {INT2OID, INT4OID}, 'b', "-", NULL, 0 },
	{ "int28mi", 2, {INT2OID, INT8OID}, 'b', "-", NULL, 0 },
	{ "int42mi", 2, {INT4OID, INT2OID}, 'b', "-", NULL, 0 },
	{ "int4mi",  2, {INT4OID, INT4OID}, 'b', "-", NULL, 0 },
	{ "int48mi", 2, {INT4OID, INT8OID}, 'b', "-", NULL, 0 },
	{ "int82mi", 2, {INT8OID, INT2OID}, 'b', "-", NULL, 0 },
	{ "int84mi", 2, {INT8OID, INT4OID}, 'b', "-", NULL, 0 },
	{ "int8mi",  2, {INT8OID, INT8OID}, 'b', "-", NULL, 0 },
	{ "float4mi",  2, {FLOAT4OID, FLOAT4OID}, 'b', "-", NULL, 0 },
	{ "float48mi", 2, {FLOAT4OID, FLOAT8OID}, 'b', "-", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84mi", 2, {FLOAT8OID, FLOAT4OID}, 'b', "-", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8mi",  2, {FLOAT8OID, FLOAT8OID}, 'b', "-", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '*'  : mutiply operators */
	{ "int2mul",  2, {INT2OID, INT2OID}, 'b', "*", NULL, 0 },
	{ "int24mul", 2, {INT2OID, INT4OID}, 'b', "*", NULL, 0 },
	{ "int28mul", 2, {INT2OID, INT8OID}, 'b', "*", NULL, 0 },
	{ "int42mul", 2, {INT4OID, INT2OID}, 'b', "*", NULL, 0 },
	{ "int4mul",  2, {INT4OID, INT4OID}, 'b', "*", NULL, 0 },
	{ "int48mul", 2, {INT4OID, INT8OID}, 'b', "*", NULL, 0 },
	{ "int82mul", 2, {INT8OID, INT2OID}, 'b', "*", NULL, 0 },
	{ "int84mul", 2, {INT8OID, INT4OID}, 'b', "*", NULL, 0 },
	{ "int8mul",  2, {INT8OID, INT8OID}, 'b', "*", NULL, 0 },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, 'b', "*", NULL, 0 },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, 'b', "*", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84mul", 2, {FLOAT8OID, FLOAT4OID}, 'b', "*", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, 'b', "*", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '/'  : divide operators */
	{ "int2div",  2, {INT2OID, INT2OID}, 'F', "int2div",
	  DEVFUNC_INTxDIV_TEMPLATE(int2div,  short, short, short), 0},
	{ "int24div", 2, {INT2OID, INT4OID}, 'F', "int24div",
	  DEVFUNC_INTxDIV_TEMPLATE(int24div, int, short, int), 0},
	{ "int28div", 2, {INT2OID, INT8OID}, 'F', "int28div",
	  DEVFUNC_INTxDIV_TEMPLATE(int28div, long, short, long), 0},
	{ "int42div", 2, {INT4OID, INT2OID}, 'F', "int42div",
	  DEVFUNC_INTxDIV_TEMPLATE(int42div,  int, int, short), 0},
	{ "int4div",  2, {INT4OID, INT4OID}, 'F', "int4div",
	  DEVFUNC_INTxDIV_TEMPLATE(int4div, int, int, int), 0},
	{ "int48div", 2, {INT4OID, INT8OID}, 'F', "int48div",
	  DEVFUNC_INTxDIV_TEMPLATE(int48div, long, int, long), 0},
	{ "int82div", 2, {INT8OID, INT2OID}, 'F', "int82div",
	  DEVFUNC_INTxDIV_TEMPLATE(int82div,  long, long, short), 0},
	{ "int84div", 2, {INT8OID, INT4OID}, 'F', "int84div",
	  DEVFUNC_INTxDIV_TEMPLATE(int84div, long, long, int), 0},
	{ "int8div",  2, {INT8OID, INT8OID}, 'F', "int8div",
	  DEVFUNC_INTxDIV_TEMPLATE(int8div, long, long, long), 0},
	{ "float4div", 2, {FLOAT4OID, FLOAT4OID},  'F', "float4div",
	  DEVFUNC_FPxDIV_TEMPLATE(float4div, float, float, float), 0},
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, 'F', "float48div",
	  DEVFUNC_FPxDIV_TEMPLATE(float48div, double, float, double),
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84div", 2, {FLOAT8OID, FLOAT4OID}, 'F', "float84div",
	  DEVFUNC_FPxDIV_TEMPLATE(float84div, double, double, float),
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8div", 2, {FLOAT8OID, FLOAT8OID},  'F', "float4div",
	  DEVFUNC_FPxDIV_TEMPLATE(float8div, double, double, double),
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '%'  : reminder operators */
	{ "int2mod", 2, {INT2OID, INT2OID}, 'F', "int2mod",
	  DEVFUNC_INTxREMIND_TEMPLATE(int2mod, short, short, short), 0 },
	{ "int4mod", 2, {INT4OID, INT4OID}, 'F', "int4mod",
	  DEVFUNC_INTxREMIND_TEMPLATE(int4mod, int, int, int), 0 },
	{ "int8mod", 2, {INT8OID, INT8OID}, 'F', "int8mod",
	  DEVFUNC_INTxREMIND_TEMPLATE(int8mod, long, long, long), 0 },

	/* '+'  : unary plus operators */
	{ "int2up", 1, {INT2OID}, 'l', "+", NULL, 0 },
	{ "int4up", 1, {INT4OID}, 'l', "+", NULL, 0 },
	{ "int8up", 1, {INT8OID}, 'l', "+", NULL, 0 },
	{ "float4up", 1, {FLOAT4OID}, 'l', "+", NULL, 0 },
	{ "float8up", 1, {FLOAT8OID}, 'l', "+", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '-'  : unary minus operators */
	{ "int2um", 1, {INT2OID}, 'l', "-", NULL, 0 },
	{ "int4um", 1, {INT4OID}, 'l', "-", NULL, 0 },
	{ "int8um", 1, {INT8OID}, 'l', "-", NULL, 0 },
	{ "float4um", 1, {FLOAT4OID}, 'l', "-", NULL, 0 },
	{ "float8um", 1, {FLOAT8OID}, 'l', "-", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '@'  : absolute value operators */
	{ "int2abs", 1, {INT2OID}, 'f', "abs", NULL, 0 },
	{ "int4abs", 1, {INT2OID}, 'f', "abs", NULL, 0 },
	{ "int8abs", 1, {INT2OID}, 'f', "abs", NULL, 0 },
	{ "float4abs", 1, {FLOAT4OID}, 'f', "fabs", NULL, 0 },
	{ "float8abs", 1, {FLOAT4OID}, 'f', "fabs", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '='  : equal operators */
	{ "int2eq",  2, {INT2OID,INT2OID}, 'b', "==", NULL, 0 },
	{ "int24eq", 2, {INT2OID,INT4OID}, 'b', "==", NULL, 0 },
	{ "int28eq", 2, {INT2OID,INT8OID}, 'b', "==", NULL, 0 },
	{ "int42eq", 2, {INT4OID,INT2OID}, 'b', "==", NULL, 0 },
	{ "int4eq",  2, {INT4OID,INT4OID}, 'b', "==", NULL, 0 },
	{ "int48eq", 2, {INT4OID,INT8OID}, 'b', "==", NULL, 0 },
	{ "int82eq", 2, {INT8OID,INT2OID}, 'b', "==", NULL, 0 },
	{ "int84eq", 2, {INT8OID,INT4OID}, 'b', "==", NULL, 0 },
	{ "int8eq" , 2, {INT8OID,INT8OID}, 'b', "==", NULL, 0 },
	{ "float4eq",  2, {FLOAT4OID, FLOAT4OID}, 'b', "==", NULL, 0 },
	{ "float48eq", 2, {FLOAT4OID, FLOAT8OID}, 'b', "==", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84eq", 2, {FLOAT8OID, FLOAT4OID}, 'b', "==", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8eq",  2, {FLOAT8OID, FLOAT8OID}, 'b', "==", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '<>' : not equal operators */
	{ "int2ne",  2, {INT2OID,INT2OID}, 'b', "!=", NULL, 0 },
	{ "int24ne", 2, {INT2OID,INT4OID}, 'b', "!=", NULL, 0 },
	{ "int28ne", 2, {INT2OID,INT8OID}, 'b', "!=", NULL, 0 },
	{ "int42ne", 2, {INT4OID,INT2OID}, 'b', "!=", NULL, 0 },
	{ "int4ne",  2, {INT4OID,INT4OID}, 'b', "!=", NULL, 0 },
	{ "int48ne", 2, {INT4OID,INT8OID}, 'b', "!=", NULL, 0 },
	{ "int82ne", 2, {INT8OID,INT2OID}, 'b', "!=", NULL, 0 },
	{ "int84ne", 2, {INT8OID,INT4OID}, 'b', "!=", NULL, 0 },
	{ "int8ne" , 2, {INT8OID,INT8OID}, 'b', "!=", NULL, 0 },
	{ "float4ne",  2, {FLOAT4OID, FLOAT4OID}, 'b', "!=", NULL, 0 },
	{ "float48ne", 2, {FLOAT4OID, FLOAT8OID}, 'b', "!=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84ne", 2, {FLOAT8OID, FLOAT4OID}, 'b', "!=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8ne",  2, {FLOAT8OID, FLOAT8OID}, 'b', "!=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '>'  : relational greater-than */
	{ "int2gt",  2, {INT2OID,INT2OID}, 'b', ">", NULL, 0 },
	{ "int24gt", 2, {INT2OID,INT4OID}, 'b', ">", NULL, 0 },
	{ "int28gt", 2, {INT2OID,INT8OID}, 'b', ">", NULL, 0 },
	{ "int42gt", 2, {INT4OID,INT2OID}, 'b', ">", NULL, 0 },
	{ "int4gt",  2, {INT4OID,INT4OID}, 'b', ">", NULL, 0 },
	{ "int48gt", 2, {INT4OID,INT8OID}, 'b', ">", NULL, 0 },
	{ "int82gt", 2, {INT8OID,INT2OID}, 'b', ">", NULL, 0 },
	{ "int84gt", 2, {INT8OID,INT4OID}, 'b', ">", NULL, 0 },
	{ "int8gt" , 2, {INT8OID,INT8OID}, 'b', ">", NULL, 0 },
	{ "float4gt",  2, {FLOAT4OID, FLOAT4OID}, 'b', ">", NULL, 0 },
	{ "float48gt", 2, {FLOAT4OID, FLOAT8OID}, 'b', ">", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84gt", 2, {FLOAT8OID, FLOAT4OID}, 'b', ">", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8gt",  2, {FLOAT8OID, FLOAT8OID}, 'b', ">", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '<'  : relational less-than */
	{ "int2lt",  2, {INT2OID,INT2OID}, 'b', "<", NULL, 0 },
	{ "int24lt", 2, {INT2OID,INT4OID}, 'b', "<", NULL, 0 },
	{ "int28lt", 2, {INT2OID,INT8OID}, 'b', "<", NULL, 0 },
	{ "int42lt", 2, {INT4OID,INT2OID}, 'b', "<", NULL, 0 },
	{ "int4lt",  2, {INT4OID,INT4OID}, 'b', "<", NULL, 0 },
	{ "int48lt", 2, {INT4OID,INT8OID}, 'b', "<", NULL, 0 },
	{ "int82lt", 2, {INT8OID,INT2OID}, 'b', "<", NULL, 0 },
	{ "int84lt", 2, {INT8OID,INT4OID}, 'b', "<", NULL, 0 },
	{ "int8lt" , 2, {INT8OID,INT8OID}, 'b', "<", NULL, 0 },
	{ "float4lt",  2, {FLOAT4OID, FLOAT4OID}, 'b', "<", NULL, 0 },
	{ "float48lt", 2, {FLOAT4OID, FLOAT8OID}, 'b', "<", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84lt", 2, {FLOAT8OID, FLOAT4OID}, 'b', "<", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8lt",  2, {FLOAT8OID, FLOAT8OID}, 'b', "<", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '>=' : relational greater-than or equal-to */
	{ "int2ge",  2, {INT2OID,INT2OID}, 'b', ">=", NULL, 0 },
	{ "int24ge", 2, {INT2OID,INT4OID}, 'b', ">=", NULL, 0 },
	{ "int28ge", 2, {INT2OID,INT8OID}, 'b', ">=", NULL, 0 },
	{ "int42ge", 2, {INT4OID,INT2OID}, 'b', ">=", NULL, 0 },
	{ "int4ge",  2, {INT4OID,INT4OID}, 'b', ">=", NULL, 0 },
	{ "int48ge", 2, {INT4OID,INT8OID}, 'b', ">=", NULL, 0 },
	{ "int82ge", 2, {INT8OID,INT2OID}, 'b', ">=", NULL, 0 },
	{ "int84ge", 2, {INT8OID,INT4OID}, 'b', ">=", NULL, 0 },
	{ "int8ge" , 2, {INT8OID,INT8OID}, 'b', ">=", NULL, 0 },
	{ "float4ge",  2, {FLOAT4OID, FLOAT4OID}, 'b', ">=", NULL, 0 },
	{ "float48ge", 2, {FLOAT4OID, FLOAT8OID}, 'b', ">=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84ge", 2, {FLOAT8OID, FLOAT4OID}, 'b', ">=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8ge",  2, {FLOAT8OID, FLOAT8OID}, 'b', ">=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '<=' : relational less-than or equal to */
	{ "int2le",  2, {INT2OID,INT2OID}, 'b', "<=", NULL, 0 },
	{ "int24le", 2, {INT2OID,INT4OID}, 'b', "<=", NULL, 0 },
	{ "int28le", 2, {INT2OID,INT8OID}, 'b', "<=", NULL, 0 },
	{ "int42le", 2, {INT4OID,INT2OID}, 'b', "<=", NULL, 0 },
	{ "int4le",  2, {INT4OID,INT4OID}, 'b', "<=", NULL, 0 },
	{ "int48le", 2, {INT4OID,INT8OID}, 'b', "<=", NULL, 0 },
	{ "int82le", 2, {INT8OID,INT2OID}, 'b', "<=", NULL, 0 },
	{ "int84le", 2, {INT8OID,INT4OID}, 'b', "<=", NULL, 0 },
	{ "int8le" , 2, {INT8OID,INT8OID}, 'b', "<=", NULL, 0 },
	{ "float4le",  2, {FLOAT4OID, FLOAT4OID}, 'b', "<=", NULL, 0 },
	{ "float48le", 2, {FLOAT4OID, FLOAT8OID}, 'b', "<=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float84le", 2, {FLOAT8OID, FLOAT4OID}, 'b', "<=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },
	{ "float8le",  2, {FLOAT8OID, FLOAT8OID}, 'b', "<=", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP },

	/* '&'  : bitwise and */
	{ "int2and", 2, {INT2OID, INT2OID}, 'b', "&", NULL, 0 },
	{ "int4and", 2, {INT4OID, INT4OID}, 'b', "&", NULL, 0 },
	{ "int8and", 2, {INT8OID, INT8OID}, 'b', "&", NULL, 0 },

	/* '|'  : bitwise or */
	{ "int2or", 2, {INT2OID, INT2OID}, 'b', "|", NULL, 0 },
	{ "int4or", 2, {INT4OID, INT4OID}, 'b', "|", NULL, 0 },
	{ "int8or", 2, {INT8OID, INT8OID}, 'b', "|", NULL, 0 },

	/* '#'  : bitwise xor */
	{ "int2xor", 2, {INT2OID, INT2OID}, 'b', "^", NULL, 0 },
	{ "int4xor", 2, {INT4OID, INT4OID}, 'b', "^", NULL, 0 },
	{ "int8xor", 2, {INT8OID, INT8OID}, 'b', "^", NULL, 0 },

	/* '~'  : bitwise not operators */
	{ "int2not", 1, {INT2OID}, 'l', "~", NULL, 0 },
	{ "int4not", 1, {INT4OID}, 'l', "~", NULL, 0 },
	{ "int8not", 1, {INT8OID}, 'l', "~", NULL, 0 },

	/* '>>' : right shift */
	{ "int2shr", 2, {INT2OID,INT4OID}, 'b', ">>", NULL, 0 },
	{ "int4shr", 2, {INT4OID,INT4OID}, 'b', ">>", NULL, 0 },
	{ "int4shr", 2, {INT8OID,INT4OID}, 'b', ">>", NULL, 0 },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID,INT4OID}, 'b', "<<", NULL, 0 },
	{ "int4shl", 2, {INT4OID,INT4OID}, 'b', "<<", NULL, 0 },
	{ "int4shl", 2, {INT8OID,INT4OID}, 'b', "<<", NULL, 0 },

	/*
	 * Mathmatical functions
	 */
	{ "cbrt",	1, {FLOAT8OID}, 'f', "cbrt",	NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "ceil",	1, {FLOAT8OID}, 'f', "ceil",	NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "exp",	1, {FLOAT8OID}, 'f', "exp",		NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "floor",	1, {FLOAT8OID}, 'f', "floor",	NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "ln",		1, {FLOAT8OID}, 'f', "log",		NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "log",	1, {FLOAT8OID}, 'f', "log10",	NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "pi",		0, {}, 'c',	"CUDART_PI", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },

	{ "power",	2, {FLOAT8OID,FLOAT8OID}, 'f',  "pow", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "pow",	2, {FLOAT8OID,FLOAT8OID}, 'f',  "pow", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "dpow",	2, {FLOAT8OID,FLOAT8OID}, 'f',  "pow", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },

	{ "round",	1, {FLOAT8OID}, 'f', "round",	NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "sign",	1, {FLOAT8OID}, 'f', "sign_double",
	  DEVFUNC_FPSIGN_TEMPLATE(double),
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "sqrt",	1, {FLOAT8OID}, 'f', "sqrt", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "dsqrt",	1, {FLOAT8OID}, 'f', "sqrt", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },

	{ "trunc",	1, {FLOAT8OID}, 'f', "trunc", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "dtrunc",	1, {FLOAT8OID}, 'f', "trunc", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },

	/*
	 * Trigonometric function
	 */
	{ "acos",	1, {FLOAT8OID}, 'f', "acos", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "asin",	1, {FLOAT8OID}, 'f', "asin", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "atan",	1, {FLOAT8OID}, 'f', "atan", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "atan2",	2, {FLOAT8OID,FLOAT8OID}, 'f', "atan2", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "cos",	1, {FLOAT8OID}, 'f', "cos", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "cot",	1, {FLOAT8OID}, 'f', "cot_double",
	  DEVFUNC_FPCOT_TEMPLATE(double),
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "sin",	1, {FLOAT8OID}, 'f', "sin", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
	{ "tan",	1, {FLOAT8OID}, 'f', "tan", NULL,
	  DEVINFO_FLAGS_DOUBLE_FP | DEVINFO_FLAGS_INC_MATHFUNC_H },
};

PgStromDevFuncInfo *
pgstrom_devfunc_lookup(Oid func_oid)
{
	PgStromDevFuncInfo *entry;
	HeapTuple		tuple;
	Form_pg_proc	procForm;
	MemoryContext	oldcxt;
	ListCell	   *cell;
	int				i, hash;

	hash = hash_uint32((uint32) func_oid) % lengthof(devfunc_info_slot);
	foreach (cell, devfunc_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->func_oid == func_oid)
			return (!entry->func_ident ? NULL : entry);
	}

	/*
	 * Not found, insert a new entry
	 */
	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	procForm = (Form_pg_proc) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

	entry = palloc0(sizeof(PgStromDevFuncInfo) +
					sizeof(Oid) * procForm->pronargs);
	entry->func_oid = func_oid;

	if (procForm->pronamespace != PG_CATALOG_NAMESPACE)
		goto out;

	if (!pgstrom_devtype_lookup(procForm->prorettype))
		goto out;
	for (i=0; i < procForm->pronargs; i++)
	{
		if (!pgstrom_devtype_lookup(procForm->proargtypes.values[i]))
			goto out;
	}

	for (i=0; i < lengthof(device_func_catalog); i++)
	{
		if (strcmp(NameStr(procForm->proname),
				   device_func_catalog[i].func_name) == 0 &&
			procForm->pronargs == device_func_catalog[i].func_nargs &&
			memcmp(procForm->proargtypes.values,
				   device_func_catalog[i].func_argtypes,
				   sizeof(Oid) * procForm->pronargs) == 0)
		{
			entry->func_kind   = device_func_catalog[i].func_kind;
			entry->func_ident  = device_func_catalog[i].func_ident;
			entry->func_source = device_func_catalog[i].func_source;
			entry->func_flags  = device_func_catalog[i].func_flags;
			entry->func_nargs  = device_func_catalog[i].func_nargs;
			memcpy(entry->func_argtypes,
				   device_func_catalog[i].func_argtypes,
				   sizeof(Oid) * procForm->pronargs);
			break;
		}
	}
out:
	devfunc_info_slot[hash] = lappend(devfunc_info_slot[hash], entry);
	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	return (!entry->func_ident ? NULL : entry);
}

/* ------------------------------------------------------------
 *
 * Routines to manage GPU device and CUDA context
 *
 * ------------------------------------------------------------
 */
static bool					pgstrom_cuda_initialized = false;
static int					pgstrom_num_devices = 0;
static PgStromDeviceInfo   *pgstrom_device_info_data = NULL;

int
pgstrom_get_num_devices(void)
{
	return pgstrom_num_devices;
}

const PgStromDeviceInfo *
pgstrom_get_device_info(int dev_index)
{
	if (dev_index < 0 || dev_index >= pgstrom_num_devices)
		elog(ERROR, "device index is out of range : %d", dev_index);

	return &pgstrom_device_info_data[dev_index];
}

void
pgstrom_set_device_context(int dev_index)
{
	CUresult	ret;

	if (!pgstrom_cuda_initialized)
	{
		cuInit(0);
		pgstrom_cuda_initialized = true;
	}

	Assert(dev_index < pgstrom_num_devices);
	if (!pgstrom_device_info_data[dev_index].context)
	{
		ret = cuCtxCreate(&pgstrom_device_info_data[dev_index].context,
						  0,
						  pgstrom_device_info_data[dev_index].device);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to create device context: %s",
				 cuda_error_to_string(ret));
	}
	ret = cuCtxSetCurrent(pgstrom_device_info_data[dev_index].context);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to switch device context: %s",
			 cuda_error_to_string(ret));
}

/*
 * pgstrom_device_info(int dev_index)
 *
 * This function shows properties of installed GPU devices.
 * If dev_index is 0, it shows all the device's one.
 */
Datum
pgstrom_device_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	   *funcctx;
	PgStromDeviceInfo  *devinfo;
	StringInfoData		str;
	uint32		devindex;
	uint32		property;
	HeapTuple	tuple;
	Datum		values[3];
	bool		isnull[3];

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(3, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "devid",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "name",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "value",
						   TEXTOID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->user_fctx = NULL;

		MemoryContextSwitchTo(oldcxt);
	}
	funcctx = SRF_PERCALL_SETUP();

	if (PG_GETARG_UINT32(0) == 0)
	{
		devindex = funcctx->call_cntr / 22;
		property = funcctx->call_cntr % 22;

		if (devindex >= pgstrom_num_devices)
			SRF_RETURN_DONE(funcctx);
	}
	else
	{
		devindex = PG_GETARG_UINT32(0) - 1;

		if (devindex >= pgstrom_num_devices)
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("GPU device %d does not exist", devindex)));

		if (funcctx->call_cntr >= 22)
			SRF_RETURN_DONE(funcctx);
		property = funcctx->call_cntr;
	}

	devinfo = &pgstrom_device_info_data[devindex];
	initStringInfo(&str);

	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(devindex + 1);

	switch (property)
	{
		case 0:
			values[1] = CStringGetTextDatum("name");
			appendStringInfo(&str, "%s", devinfo->dev_name);
			break;
		case 1:
			values[1] = CStringGetTextDatum("capability");
			appendStringInfo(&str, "%d.%d",
							 devinfo->dev_major, devinfo->dev_minor);
			break;
		case 2:
			values[1] = CStringGetTextDatum("num of procs");
			appendStringInfo(&str, "%d", devinfo->dev_proc_nums);
			break;
		case 3:
			values[1] = CStringGetTextDatum("wrap per proc");
			appendStringInfo(&str, "%d", devinfo->dev_proc_warp_sz);
			break;
		case 4:
			values[1] = CStringGetTextDatum("clock of proc");
			appendStringInfo(&str, "%d MHz", devinfo->dev_proc_clock / 1000);
			break;
		case 5:
			values[1] = CStringGetTextDatum("global mem size");
			appendStringInfo(&str, "%lu MB",
							 devinfo->dev_global_mem_sz / (1024 * 1024));
			break;
		case 6:
			values[1] = CStringGetTextDatum("global mem width");
			appendStringInfo(&str, "%d bits", devinfo->dev_global_mem_width);
			break;
		case 7:
			values[1] = CStringGetTextDatum("global mem clock");
			appendStringInfo(&str, "%d MHz",
							 devinfo->dev_global_mem_clock / 1000);
			break;
		case 8:
			values[1] = CStringGetTextDatum("shared mem size");
			appendStringInfo(&str, "%d KB", devinfo->dev_shared_mem_sz / 1024);
			break;
		case 9:
			values[1] = CStringGetTextDatum("L2 cache size");
			appendStringInfo(&str, "%d KB", devinfo->dev_l2_cache_sz / 1024);
			break;
		case 10:
			values[1] = CStringGetTextDatum("const mem size");
			appendStringInfo(&str, "%d KB", devinfo->dev_const_mem_sz / 1024);
			break;
		case 11:
			values[1] = CStringGetTextDatum("max block size");
			appendStringInfo(&str, "{%d, %d, %d}",
							 devinfo->dev_max_block_dim_x,
							 devinfo->dev_max_block_dim_y,
							 devinfo->dev_max_block_dim_z);
			break;
		case 12:
			values[1] = CStringGetTextDatum("max grid size");
			appendStringInfo(&str, "{%d, %d, %d}",
							 devinfo->dev_max_grid_dim_x,
							 devinfo->dev_max_grid_dim_y,
							 devinfo->dev_max_grid_dim_z);
			break;
		case 13:
			values[1] = CStringGetTextDatum("max threads per proc");
			appendStringInfo(&str, "%d", devinfo->dev_max_threads_per_proc);
			break;
		case 14:
			values[1] = CStringGetTextDatum("max registers per block");
			appendStringInfo(&str, "%d", devinfo->dev_max_regs_per_block);
			break;
		case 15:
			values[1] = CStringGetTextDatum("integrated memory");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_integrated ? "yes" : "no"));
			break;
		case 16:
			values[1] = CStringGetTextDatum("unified address");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_unified_addr ? "yes" : "no"));
			break;
		case 17:
			values[1] = CStringGetTextDatum("map host memory");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_can_map_hostmem ? "yes" : "no"));
			break;
		case 18:
			values[1] = CStringGetTextDatum("concurrent kernel");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_concurrent_kernel ? "yes" : "no"));
			break;
		case 19:
			values[1] = CStringGetTextDatum("concurrent memcpy");
			appendStringInfo(&str, "%s",
							 (devinfo->dev_concurrent_memcpy ? "yes" : "no"));
			break;
		case 20:
			values[1] = CStringGetTextDatum("pci bus-id");
			appendStringInfo(&str, "%d", devinfo->dev_pci_busid);
			break;
		case 21:
			values[1] = CStringGetTextDatum("pci device-id");
			appendStringInfo(&str, "%d", devinfo->dev_pci_deviceid);
			break;
		default:
			elog(ERROR, "unexpected property : %d", property);
			break;
	}
	values[2] = CStringGetTextDatum(str.data);

	tuple = heap_form_tuple(funcctx->tuple_desc, values, isnull);

	pfree(str.data);
	SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_device_info);

/*
 * pgstrom_devinfo_init
 *
 * This routine collects properties of GPU devices being used to computing
 * schedule at server starting up time.
 * Note that cuInit(0) has to be called at the backend processes again to
 * avoid CUDA_ERROR_NOT_INITIALIZED errors.
 */
void
pgstrom_devinfo_init(void)
{
	CUresult	ret;
	int			i, j;

	/*
	 * Zero clear the hash slots of device functions/types
	 */
	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));

	/*
	 * Initialize CUDA APIs
	 */
	ret = cuInit(0);
	if (ret != CUDA_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("cuda: failed to initialized APIs : %s",
						cuda_error_to_string(ret))));

	/*
	 * Collect properties of installed devices
	 */
	ret = cuDeviceGetCount(&pgstrom_num_devices);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to get number of devices : %s",
			 cuda_error_to_string(ret));

	pgstrom_device_info_data
		= MemoryContextAllocZero(TopMemoryContext,
								 sizeof(PgStromDeviceInfo) *
								 pgstrom_num_devices);
	for (i=0; i < pgstrom_num_devices; i++)
	{
		PgStromDeviceInfo  *devinfo;
		static struct {
			size_t				offset;
			CUdevice_attribute	attribute;
		} device_attrs[] = {
			{ offsetof(PgStromDeviceInfo, dev_proc_nums),
			  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT },
			{ offsetof(PgStromDeviceInfo, dev_proc_warp_sz),
			  CU_DEVICE_ATTRIBUTE_WARP_SIZE },
			{ offsetof(PgStromDeviceInfo, dev_proc_clock),
			  CU_DEVICE_ATTRIBUTE_CLOCK_RATE },
			{ offsetof(PgStromDeviceInfo, dev_global_mem_width),
			  CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH },
			{ offsetof(PgStromDeviceInfo, dev_global_mem_clock),
			  CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE },
			{ offsetof(PgStromDeviceInfo, dev_shared_mem_sz),
			  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK },
			{ offsetof(PgStromDeviceInfo, dev_l2_cache_sz),
			  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE },
			{ offsetof(PgStromDeviceInfo, dev_const_mem_sz),
			  CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY },
			{ offsetof(PgStromDeviceInfo, dev_max_block_dim_x),
			  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X },
			{ offsetof(PgStromDeviceInfo, dev_max_block_dim_y),
			  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y },
			{ offsetof(PgStromDeviceInfo, dev_max_block_dim_z),
			  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z },
			{ offsetof(PgStromDeviceInfo, dev_max_grid_dim_x),
			  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X },
			{ offsetof(PgStromDeviceInfo, dev_max_grid_dim_y),
			  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y },
			{ offsetof(PgStromDeviceInfo, dev_max_grid_dim_z),
			  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z },
			{ offsetof(PgStromDeviceInfo, dev_max_regs_per_block),
			  CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK },
			{ offsetof(PgStromDeviceInfo, dev_max_threads_per_proc),
			  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR },
			{ offsetof(PgStromDeviceInfo, dev_integrated),
			  CU_DEVICE_ATTRIBUTE_INTEGRATED },
			{ offsetof(PgStromDeviceInfo, dev_unified_addr),
			  CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING },
			{ offsetof(PgStromDeviceInfo, dev_can_map_hostmem),
			  CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY },
			{ offsetof(PgStromDeviceInfo, dev_concurrent_kernel),
			  CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS },
			{ offsetof(PgStromDeviceInfo, dev_concurrent_memcpy),
			  CU_DEVICE_ATTRIBUTE_GPU_OVERLAP},
			{ offsetof(PgStromDeviceInfo, dev_pci_busid),
			  CU_DEVICE_ATTRIBUTE_PCI_BUS_ID },
			{ offsetof(PgStromDeviceInfo, dev_pci_deviceid),
			  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID },
		};

		devinfo = &pgstrom_device_info_data[i];

		if ((ret = cuDeviceGetName(devinfo->dev_name,
								   sizeof(devinfo->dev_name),
								   devinfo->device)) ||
			(ret = cuDeviceComputeCapability(&devinfo->dev_major,
											 &devinfo->dev_minor,
											 devinfo->device)) ||
			(ret = cuDeviceTotalMem(&devinfo->dev_global_mem_sz,
									devinfo->device)))
			elog(ERROR, "cuda: failed to get attribute of GPU device : %s",
				 cuda_error_to_string(ret));
		for (j=0; j < lengthof(device_attrs); j++)
		{
			ret = cuDeviceGetAttribute((int *)((uintptr_t) devinfo +
											   device_attrs[j].offset),
									   device_attrs[j].attribute,
									   devinfo->device);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "cuda: failed to get attribute of GPU device : %s",
					 cuda_error_to_string(ret));
		}

		/*
		 * Logs detected device properties
		 */
		elog(LOG, "PG-Strom: GPU device[%d] %s; capability v%d.%d, "
			 "%d of streaming processor units (%d wraps per unit, %dMHz), "
			 "%luMB of global memory (%d bits, %dMHz)",
			 i, devinfo->dev_name, devinfo->dev_major, devinfo->dev_minor,
			 devinfo->dev_proc_nums, devinfo->dev_proc_warp_sz,
			 devinfo->dev_proc_clock / 1000,
			 devinfo->dev_global_mem_sz / (1024 * 1024),
			 devinfo->dev_global_mem_width,
			 devinfo->dev_global_mem_clock / 1000);
	}
}

/*
 * For coding convinient, translation from an error code to cstring.
 */
const char *
cuda_error_to_string(CUresult result)
{
	char	buf[256];

	switch (result)
	{
		case CUDA_SUCCESS:
			return "cuccess";
		case CUDA_ERROR_INVALID_VALUE:
			return "invalid value";
		case CUDA_ERROR_OUT_OF_MEMORY:
			return "out of memory";
		case CUDA_ERROR_NOT_INITIALIZED:
			return "not initialized";
		case CUDA_ERROR_DEINITIALIZED:
			return "deinitialized";
		case CUDA_ERROR_PROFILER_DISABLED:
			return "profiler disabled";
		case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
			return "profiler not initialized";
		case CUDA_ERROR_PROFILER_ALREADY_STARTED:
			return "profiler already started";
		case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
			return "profiler already stopped";
		case CUDA_ERROR_NO_DEVICE:
			return "no device";
		case CUDA_ERROR_INVALID_DEVICE:
			return "invalid device";
		case CUDA_ERROR_INVALID_IMAGE:
			return "invalid image";
		case CUDA_ERROR_INVALID_CONTEXT:
			return "invalid context";
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
			return "context already current";
		case CUDA_ERROR_MAP_FAILED:
			return "map failed";
		case CUDA_ERROR_UNMAP_FAILED:
			return "unmap failed";
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			return "array is mapped";
		case CUDA_ERROR_ALREADY_MAPPED:
			return "already mapped";
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			return "no binary for gpu";
		case CUDA_ERROR_ALREADY_ACQUIRED:
			return "already acquired";
		case CUDA_ERROR_NOT_MAPPED:
			return "not mapped";
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			return "not mapped as array";
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			return "not mapped as pointer";
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			return "ecc uncorrectable";
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			return "unsupported limit";
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
			return "context already in use";
		case CUDA_ERROR_INVALID_SOURCE:
			return "invalid source";
		case CUDA_ERROR_FILE_NOT_FOUND:
			return "file not found";
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			return "shared object symbol not found";
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			return "shared object init failed";
		case CUDA_ERROR_OPERATING_SYSTEM:
			return "operating system";
		case CUDA_ERROR_INVALID_HANDLE:
			return "invalid handle";
		case CUDA_ERROR_NOT_FOUND:
			return "not found";
		case CUDA_ERROR_NOT_READY:
			return "not ready";
		case CUDA_ERROR_LAUNCH_FAILED:
			return "launch failed";
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			return "launch out of resources";
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			return "launch timeout";
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			return "launch incompatible texturing";
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
			return "peer access already enabled";
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
			return "peer access not enabled";
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
			return "primary context active";
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:
			return "context is destroyed";
		default:
			break;
	}
	snprintf(buf, sizeof(buf), "cuda error code: %d", result);
	return pstrdup(buf);
}
