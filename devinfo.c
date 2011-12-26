/*
 * devinfo.c
 *
 * Collect properties of OpenCL processing units
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
#include "storage/ipc.h"
#include "utils/memutils.h"
#include "utils/syscache.h"
#include "pg_strom.h"

/*
 * Declarations
 */
#define PGSTROM_UNSUPPORTED_CAST_INTO	((void *)-1)
static List			   *devtype_info_slot[512];
static List			   *devfunc_info_slot[1024];
static List			   *devcast_info_slot[256];
static MemoryContext	devinfo_memcxt;

cl_uint					pgstrom_num_devices;
cl_device_id		   *pgstrom_device_id;
PgStromDeviceInfo	  **pgstrom_device_info;
cl_context				pgstrom_device_context = NULL;

/* ------------------------------------------------------------
 *
 * Catalog of supported device types
 *
 * ------------------------------------------------------------
 */
static struct {
	Oid		type_oid;
	char   *type_ident;
	char   *type_source;
} device_type_catalog[] = {
	{ BOOLOID,		"bool_t",	"typedef char  bool_t" },
	{ INT2OID,		"int2_t",	"typedef short int2_t" },
	{ INT4OID,		"int4_t",	"typedef int   int4_t" },
	{ INT8OID,		"int8_t",	"typedef long  int8_t" },
	{ FLOAT4OID,	"float",	NULL },
	{ FLOAT8OID,	"double",	NULL },
};

void
pgstrom_devtype_format(StringInfo str, Oid type_oid, Datum value)
{
	switch (type_oid)
	{
		case BOOLOID:
			appendStringInfo(str, "%d", DatumGetChar(value));
			break;
		case INT2OID:
			appendStringInfo(str, "%d", DatumGetInt16(value));
			break;
		case INT4OID:
			appendStringInfo(str, "%d", DatumGetInt32(value));
			break;
		case INT8OID:
			appendStringInfo(str, "%ld", DatumGetInt64(value));
			break;
		case FLOAT8OID:
		case FLOAT4OID:
			{
				double	num = (type_oid == FLOAT4OID ?
							   Float4GetDatum(value) :
							   Float8GetDatum(value));
				if (isnan(num))
					appendStringInfo(str, "NAN");
				else
					switch (isinf(num))
					{
						case 1:
							appendStringInfo(str, "INFINITY");
							break;
						case -1:
							appendStringInfo(str, "-INFINITY");
							break;
						default:
							appendStringInfo(str, "%.*g", FLT_DIG, num);
							break;
					}
			}
			break;
		default:
			elog(ERROR, "unexpected type value being formatted %u", type_oid);
			break;
	}
}

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

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

	entry = palloc0(sizeof(PgStromDevTypeInfo));
	entry->type_oid = type_oid;
	if (typeForm->typnamespace != PG_CATALOG_NAMESPACE)
		goto out;

	/* FLOAT8 is not available on device without FP64bit support */
	if (type_oid == FLOAT8OID)
	{
		for (i=0; i < pgstrom_num_devices; i++)
		{
			if (!pgstrom_device_info[i]->dev_double_fp_config)
				goto out;
		}
	}

	for (i=0; i < lengthof(device_type_catalog); i++)
	{
		if (device_type_catalog[i].type_oid == type_oid)
		{
			entry->type_ident  = device_type_catalog[i].type_ident;
			entry->type_source = device_type_catalog[i].type_source;
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
 * Catalog of supported device casts
 *
 * ------------------------------------------------------------
 */
static struct {
	Oid		cast_source;
	Oid		cast_target;
	char   *func_ident;
	char   *func_source;
} device_cast_by_func_catalog[] = {
};

static Oid	device_cast_by_inline_catalog[] = {
	BOOLOID,
	INT2OID,
	INT4OID,
	INT8OID,
	FLOAT4OID,
	FLOAT8OID,
};

PgStromDevCastInfo *
pgstrom_devcast_lookup(Oid source_typeid, Oid target_typeid)
{
	PgStromDevCastInfo *entry;
	MemoryContext		oldcxt;
	ListCell		   *cell;
	int					i, j, hash;

	hash = hash_uint32((uint32)(source_typeid ^ target_typeid))
		% lengthof(devcast_info_slot);
	foreach (cell, devcast_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->cast_source == source_typeid &&
			entry->cast_target == target_typeid)
		{
			if (entry->func_ident == PGSTROM_UNSUPPORTED_CAST_INTO)
				return NULL;
			return entry;
		}
	}

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

	entry = palloc0(sizeof(PgStromDevCastInfo));
	entry->cast_source = source_typeid;
	entry->cast_target = target_typeid;
	entry->func_ident  = PGSTROM_UNSUPPORTED_CAST_INTO;
	entry->func_source = NULL;

	if (!pgstrom_devtype_lookup(source_typeid) ||
		!pgstrom_devtype_lookup(target_typeid))
		goto out;

	/*
	 * Lookup cast by function catalog
	 */
	for (i=0; i < lengthof(device_cast_by_func_catalog); i++)
	{
		if (device_cast_by_func_catalog[i].cast_source == source_typeid &&
			device_cast_by_func_catalog[i].cast_target == target_typeid)
		{
			entry->func_ident  = device_cast_by_func_catalog[i].func_ident;
			entry->func_source = device_cast_by_func_catalog[i].func_source;
			goto out;
		}
	}

	/*
	 * Lookup cast by inline catalog
	 */
	for (i=0, j=0; i < lengthof(device_cast_by_inline_catalog); i++)
	{
		if (device_cast_by_inline_catalog[i] == source_typeid)
			j |= 1;
		if (device_cast_by_inline_catalog[i] == target_typeid)
			j |= 2;
		if (j == 3)
		{
			entry->func_ident = NULL;
			entry->func_source = NULL;
			break;
		}
	}
out:
	devcast_info_slot[hash]
		= lappend(devcast_info_slot[hash], entry);

	MemoryContextSwitchTo(oldcxt);

	if (entry->func_ident == PGSTROM_UNSUPPORTED_CAST_INTO)
		return NULL;
	return entry;
}

/* ------------------------------------------------------------
 *
 * Catalog of supported device functions
 *
 * ------------------------------------------------------------
 */
#define DEVFUNC_INTxDIV_TEMPLATE(name,rtype,xtype,ytype)			\
	#rtype " " #name "(int *error, " #xtype " x, " #ytype " y)\n"	\
	"{\n"															\
	"    " #rtype " rc;\n"											\
	"    if (y == 0)\n"												\
	"        *error |= DEVERR_DIVISION_BY_ZERO;\n"					\
	"    rc = x / y;\n"												\
	"    if (y == -1 && x < 0 && rc <= 0)\n"						\
	"        *error |= DEVERR_NUMERIC_VALUE_OUT_OF_RANGE\n"			\
	"    return rc;\n"												\
	"}\n"

#define DEVFUNC_FPxDIV_TEMPLATE(name,rtype,xtype,ytype)				\
	#rtype " " #name "(int *error, " #xtype " x, " #ytype " y)\n"	\
	"{\n"															\
	"    " #rtype " rc;\n"											\
	"    if (y == 0)\n"												\
	"        *error |= DEVERR_DIVISION_BY_ZERO;\n"					\
	"    rc = x / y;\n"												\
	"    if ((isinf(rc) && !isinf(x) && !isinf(y)) ||\n"			\
	"        (rc == 0.0 && x != 0.0))\n"							\
	"    if (y == -1 && x < 0 && rc <= 0)\n"						\
	"        *error |= DEVERR_VALUE_OUT_OF_RANGE\n"					\
	"    return rc;\n"												\
	"}\n"

#define DEVFUN_INTxREMIND_TEMPLATE(name,rtype,xtype,ytype)			\
	#rtype " " #name "(int *error, " #xtype " x, " #ytype " y)\n"	\
	"{\n"															\
	"    if (y == 0)\n"												\
	"        *error |= DEVERR_DIVISION_BY_ZERO;\n"					\
	"    return x % y;\n"											\
	"}\n"

static struct {
	char   *func_name;
	int		func_nargs;
	Oid		func_argtypes[2];
	char	func_kind;	/* 'f', 'l', 'r', 'b' or 'c' */
	char   *func_ident;
	char   *func_source;
} device_func_catalog[] = {
	/* '+'  : add operators */
	{ "int2pl",	 2, {INT2OID, INT2OID}, 'b', "+", NULL },
	{ "int24pl", 2, {INT2OID, INT4OID}, 'b', "+", NULL },
	{ "int28pl", 2, {INT2OID, INT8OID}, 'b', "+", NULL },
	{ "int42pl", 2, {INT4OID, INT2OID}, 'b', "+", NULL },
	{ "int4pl",  2, {INT4OID, INT4OID}, 'b', "+", NULL },
	{ "int48pl", 2, {INT4OID, INT8OID}, 'b', "+", NULL },
	{ "int82pl", 2, {INT8OID, INT2OID}, 'b', "+", NULL },
	{ "int84pl", 2, {INT8OID, INT4OID}, 'b', "+", NULL },
	{ "int8pl",  2, {INT8OID, INT8OID}, 'b', "+", NULL },
	{ "float4pl",  2, {FLOAT4OID, FLOAT4OID}, 'b', "+", NULL },
	{ "float48pl", 2, {FLOAT4OID, FLOAT8OID}, 'b', "+", NULL },
	{ "float84pl", 2, {FLOAT8OID, FLOAT4OID}, 'b', "+", NULL },
	{ "float8pl",  2, {FLOAT8OID, FLOAT8OID}, 'b', "+", NULL },

	/* '-'  : subtract operators */
	{ "int2mi",	 2, {INT2OID, INT2OID}, 'b', "-", NULL },
	{ "int24mi", 2, {INT2OID, INT4OID}, 'b', "-", NULL },
	{ "int28mi", 2, {INT2OID, INT8OID}, 'b', "-", NULL },
	{ "int42mi", 2, {INT4OID, INT2OID}, 'b', "-", NULL },
	{ "int4mi",  2, {INT4OID, INT4OID}, 'b', "-", NULL },
	{ "int48mi", 2, {INT4OID, INT8OID}, 'b', "-", NULL },
	{ "int82mi", 2, {INT8OID, INT2OID}, 'b', "-", NULL },
	{ "int84mi", 2, {INT8OID, INT4OID}, 'b', "-", NULL },
	{ "int8mi",  2, {INT8OID, INT8OID}, 'b', "-", NULL },
	{ "float4mi",  2, {FLOAT4OID, FLOAT4OID}, 'b', "-", NULL },
	{ "float48mi", 2, {FLOAT4OID, FLOAT8OID}, 'b', "-", NULL },
	{ "float84mi", 2, {FLOAT8OID, FLOAT4OID}, 'b', "-", NULL },
	{ "float8mi",  2, {FLOAT8OID, FLOAT8OID}, 'b', "-", NULL },

	/* '*'  : mutiply operators */
	{ "int2mul",  2, {INT2OID, INT2OID}, 'b', "*", NULL },
	{ "int24mul", 2, {INT2OID, INT4OID}, 'b', "*", NULL },
	{ "int28mul", 2, {INT2OID, INT8OID}, 'b', "*", NULL },
	{ "int42mul", 2, {INT4OID, INT2OID}, 'b', "*", NULL },
	{ "int4mul",  2, {INT4OID, INT4OID}, 'b', "*", NULL },
	{ "int48mul", 2, {INT4OID, INT8OID}, 'b', "*", NULL },
	{ "int82mul", 2, {INT8OID, INT2OID}, 'b', "*", NULL },
	{ "int84mul", 2, {INT8OID, INT4OID}, 'b', "*", NULL },
	{ "int8mul",  2, {INT8OID, INT8OID}, 'b', "*", NULL },
	{ "float4mul",  2, {FLOAT4OID, FLOAT4OID}, 'b', "*", NULL },
	{ "float48mul", 2, {FLOAT4OID, FLOAT8OID}, 'b', "*", NULL },
	{ "float84mul", 2, {FLOAT8OID, FLOAT4OID}, 'b', "*", NULL },
	{ "float8mul",  2, {FLOAT8OID, FLOAT8OID}, 'b', "*", NULL },

	/* '/'  : divide operators */
	{ "int2div",  2, {INT2OID, INT2OID}, 'f', "int2div",
	  DEVFUNC_INTxDIV_TEMPLATE(int2div,  int2_t, int2_t, int2_t) },
	{ "int24div", 2, {INT2OID, INT4OID}, 'f', "int24div",
	  DEVFUNC_INTxDIV_TEMPLATE(int24div, int4_t, int2_t, int4_t) },
	{ "int28div", 2, {INT2OID, INT8OID}, 'f', "int28div",
	  DEVFUNC_INTxDIV_TEMPLATE(int28div, int8_t, int2_t, int8_t) },
	{ "int42div", 2, {INT4OID, INT2OID}, 'f', "int42div",
	  DEVFUNC_INTxDIV_TEMPLATE(int42div,  int4_t, int4_t, int2_t) },
	{ "int4div",  2, {INT4OID, INT4OID}, 'f', "int4div",
	  DEVFUNC_INTxDIV_TEMPLATE(int4div, int4_t, int4_t, int4_t) },
	{ "int48div", 2, {INT4OID, INT8OID}, 'f', "int48div",
	  DEVFUNC_INTxDIV_TEMPLATE(int48div, int8_t, int4_t, int8_t) },
	{ "int82div", 2, {INT8OID, INT2OID}, 'f', "int82div",
	  DEVFUNC_INTxDIV_TEMPLATE(int82div,  int8_t, int8_t, int2_t) },
	{ "int84div", 2, {INT8OID, INT4OID}, 'f', "int84div",
	  DEVFUNC_INTxDIV_TEMPLATE(int84div, int8_t, int8_t, int4_t) },
	{ "int8div",  2, {INT8OID, INT8OID}, 'f', "int8div",
	  DEVFUNC_INTxDIV_TEMPLATE(int8div, int8_t, int8_t, int8_t) },
	{ "float4div", 2, {FLOAT4OID, FLOAT4OID}, 'f', "float4div",
	  DEVFUNC_FPxDIV_TEMPLATE(float4div, float, float, float) },
	{ "float48div", 2, {FLOAT4OID, FLOAT8OID}, 'f', "float4div",
	  DEVFUNC_FPxDIV_TEMPLATE(float48div, double, float, double) },
	{ "float84div", 2, {FLOAT8OID, FLOAT4OID}, 'f', "float4div",
	  DEVFUNC_FPxDIV_TEMPLATE(float84div, double, double, float) },
	{ "float8div", 2, {FLOAT8OID, FLOAT8OID}, 'f', "float4div",
	  DEVFUNC_FPxDIV_TEMPLATE(float8div, double, double, double) },

	/* '%'  : reminder operators */
	{ "int2mod", 2, {INT2OID, INT2OID}, 'f', "int2mod",
	  DEVFUN_INTxREMIND_TEMPLATE(int2mod, int2_t, int2_t, int2_t) },
	{ "int4mod", 2, {INT4OID, INT4OID}, 'f', "int4mod",
	  DEVFUN_INTxREMIND_TEMPLATE(int4mod, int4_t, int4_t, int4_t) },
	{ "int8mod", 2, {INT8OID, INT8OID}, 'f', "int8mod",
	  DEVFUN_INTxREMIND_TEMPLATE(int8mod, int8_t, int8_t, int8_t) },

	/* '+'  : unary plus operators */
	{ "int2up", 1, {INT2OID}, 'l', "+", NULL },
	{ "int4up", 1, {INT4OID}, 'l', "+", NULL },
	{ "int8up", 1, {INT8OID}, 'l', "+", NULL },
	{ "float4up", 1, {FLOAT4OID}, 'l', "+", NULL },
	{ "float8up", 1, {FLOAT8OID}, 'l', "+", NULL },

	/* '-'  : unary minus operators */
	{ "int2um", 1, {INT2OID}, 'l', "-", NULL },
	{ "int4um", 1, {INT4OID}, 'l', "-", NULL },
	{ "int8um", 1, {INT8OID}, 'l', "-", NULL },
	{ "float4um", 1, {FLOAT4OID}, 'l', "-", NULL },
	{ "float8um", 1, {FLOAT8OID}, 'l', "-", NULL },

	/* '@'  : absolute value operators */
	{ "int2abs", 1, {INT2OID}, 'f', "abs", NULL },
	{ "int4abs", 1, {INT2OID}, 'f', "abs", NULL },
	{ "int8abs", 1, {INT2OID}, 'f', "abs", NULL },
	{ "float4abs", 1, {FLOAT4OID}, 'f', "fabs", NULL },
	{ "float8abs", 1, {FLOAT4OID}, 'f', "fabs", NULL },

	/* '='  : equal operators */
	{ "int2eq",  2, {INT2OID,INT2OID}, 'b', "==", NULL },
	{ "int24eq", 2, {INT2OID,INT4OID}, 'b', "==", NULL },
	{ "int28eq", 2, {INT2OID,INT8OID}, 'b', "==", NULL },
	{ "int42eq", 2, {INT4OID,INT2OID}, 'b', "==", NULL },
	{ "int4eq",  2, {INT4OID,INT4OID}, 'b', "==", NULL },
	{ "int48eq", 2, {INT4OID,INT8OID}, 'b', "==", NULL },
	{ "int82eq", 2, {INT8OID,INT2OID}, 'b', "==", NULL },
	{ "int84eq", 2, {INT8OID,INT4OID}, 'b', "==", NULL },
	{ "int8eq" , 2, {INT8OID,INT8OID}, 'b', "==", NULL },
	{ "float4eq",  2, {FLOAT4OID, FLOAT4OID}, 'b', "==", NULL },
	{ "float48eq", 2, {FLOAT4OID, FLOAT8OID}, 'b', "==", NULL },
	{ "float84eq", 2, {FLOAT8OID, FLOAT4OID}, 'b', "==", NULL },
	{ "float8eq",  2, {FLOAT8OID, FLOAT8OID}, 'b', "==", NULL },

	/* '<>' : not equal operators */
	{ "int2ne",  2, {INT2OID,INT2OID}, 'b', "!=", NULL },
	{ "int24ne", 2, {INT2OID,INT4OID}, 'b', "!=", NULL },
	{ "int28ne", 2, {INT2OID,INT8OID}, 'b', "!=", NULL },
	{ "int42ne", 2, {INT4OID,INT2OID}, 'b', "!=", NULL },
	{ "int4ne",  2, {INT4OID,INT4OID}, 'b', "!=", NULL },
	{ "int48ne", 2, {INT4OID,INT8OID}, 'b', "!=", NULL },
	{ "int82ne", 2, {INT8OID,INT2OID}, 'b', "!=", NULL },
	{ "int84ne", 2, {INT8OID,INT4OID}, 'b', "!=", NULL },
	{ "int8ne" , 2, {INT8OID,INT8OID}, 'b', "!=", NULL },
	{ "float4ne",  2, {FLOAT4OID, FLOAT4OID}, 'b', "!=", NULL },
	{ "float48ne", 2, {FLOAT4OID, FLOAT8OID}, 'b', "!=", NULL },
	{ "float84ne", 2, {FLOAT8OID, FLOAT4OID}, 'b', "!=", NULL },
	{ "float8ne",  2, {FLOAT8OID, FLOAT8OID}, 'b', "!=", NULL },

	/* '>'  : relational greater-than */
	{ "int2gt",  2, {INT2OID,INT2OID}, 'b', ">", NULL },
	{ "int24gt", 2, {INT2OID,INT4OID}, 'b', ">", NULL },
	{ "int28gt", 2, {INT2OID,INT8OID}, 'b', ">", NULL },
	{ "int42gt", 2, {INT4OID,INT2OID}, 'b', ">", NULL },
	{ "int4gt",  2, {INT4OID,INT4OID}, 'b', ">", NULL },
	{ "int48gt", 2, {INT4OID,INT8OID}, 'b', ">", NULL },
	{ "int82gt", 2, {INT8OID,INT2OID}, 'b', ">", NULL },
	{ "int84gt", 2, {INT8OID,INT4OID}, 'b', ">", NULL },
	{ "int8gt" , 2, {INT8OID,INT8OID}, 'b', ">", NULL },
	{ "float4gt",  2, {FLOAT4OID, FLOAT4OID}, 'b', ">", NULL },
	{ "float48gt", 2, {FLOAT4OID, FLOAT8OID}, 'b', ">", NULL },
	{ "float84gt", 2, {FLOAT8OID, FLOAT4OID}, 'b', ">", NULL },
	{ "float8gt",  2, {FLOAT8OID, FLOAT8OID}, 'b', ">", NULL },

	/* '<'  : relational less-than */
	{ "int2lt",  2, {INT2OID,INT2OID}, 'b', "<", NULL },
	{ "int24lt", 2, {INT2OID,INT4OID}, 'b', "<", NULL },
	{ "int28lt", 2, {INT2OID,INT8OID}, 'b', "<", NULL },
	{ "int42lt", 2, {INT4OID,INT2OID}, 'b', "<", NULL },
	{ "int4lt",  2, {INT4OID,INT4OID}, 'b', "<", NULL },
	{ "int48lt", 2, {INT4OID,INT8OID}, 'b', "<", NULL },
	{ "int82lt", 2, {INT8OID,INT2OID}, 'b', "<", NULL },
	{ "int84lt", 2, {INT8OID,INT4OID}, 'b', "<", NULL },
	{ "int8lt" , 2, {INT8OID,INT8OID}, 'b', "<", NULL },
	{ "float4lt",  2, {FLOAT4OID, FLOAT4OID}, 'b', "<", NULL },
	{ "float48lt", 2, {FLOAT4OID, FLOAT8OID}, 'b', "<", NULL },
	{ "float84lt", 2, {FLOAT8OID, FLOAT4OID}, 'b', "<", NULL },
	{ "float8lt",  2, {FLOAT8OID, FLOAT8OID}, 'b', "<", NULL },

	/* '>=' : relational greater-than or equal-to */
	{ "int2ge",  2, {INT2OID,INT2OID}, 'b', ">=", NULL },
	{ "int24ge", 2, {INT2OID,INT4OID}, 'b', ">=", NULL },
	{ "int28ge", 2, {INT2OID,INT8OID}, 'b', ">=", NULL },
	{ "int42ge", 2, {INT4OID,INT2OID}, 'b', ">=", NULL },
	{ "int4ge",  2, {INT4OID,INT4OID}, 'b', ">=", NULL },
	{ "int48ge", 2, {INT4OID,INT8OID}, 'b', ">=", NULL },
	{ "int82ge", 2, {INT8OID,INT2OID}, 'b', ">=", NULL },
	{ "int84ge", 2, {INT8OID,INT4OID}, 'b', ">=", NULL },
	{ "int8ge" , 2, {INT8OID,INT8OID}, 'b', ">=", NULL },
	{ "float4ge",  2, {FLOAT4OID, FLOAT4OID}, 'b', ">=", NULL },
	{ "float48ge", 2, {FLOAT4OID, FLOAT8OID}, 'b', ">=", NULL },
	{ "float84ge", 2, {FLOAT8OID, FLOAT4OID}, 'b', ">=", NULL },
	{ "float8ge",  2, {FLOAT8OID, FLOAT8OID}, 'b', ">=", NULL },

	/* '<=' : relational less-than or equal to */
	{ "int2le",  2, {INT2OID,INT2OID}, 'b', "<=", NULL },
	{ "int24le", 2, {INT2OID,INT4OID}, 'b', "<=", NULL },
	{ "int28le", 2, {INT2OID,INT8OID}, 'b', "<=", NULL },
	{ "int42le", 2, {INT4OID,INT2OID}, 'b', "<=", NULL },
	{ "int4le",  2, {INT4OID,INT4OID}, 'b', "<=", NULL },
	{ "int48le", 2, {INT4OID,INT8OID}, 'b', "<=", NULL },
	{ "int82le", 2, {INT8OID,INT2OID}, 'b', "<=", NULL },
	{ "int84le", 2, {INT8OID,INT4OID}, 'b', "<=", NULL },
	{ "int8le" , 2, {INT8OID,INT8OID}, 'b', "<=", NULL },
	{ "float4le",  2, {FLOAT4OID, FLOAT4OID}, 'b', "<=", NULL },
	{ "float48le", 2, {FLOAT4OID, FLOAT8OID}, 'b', "<=", NULL },
	{ "float84le", 2, {FLOAT8OID, FLOAT4OID}, 'b', "<=", NULL },
	{ "float8le",  2, {FLOAT8OID, FLOAT8OID}, 'b', "<=", NULL },

	/* '&'  : bitwise and */
	{ "int2and", 2, {INT2OID, INT2OID}, 'b', "&", NULL },
	{ "int4and", 2, {INT4OID, INT4OID}, 'b', "&", NULL },
	{ "int8and", 2, {INT8OID, INT8OID}, 'b', "&", NULL },

	/* '|'  : bitwise or */
	{ "int2or", 2, {INT2OID, INT2OID}, 'b', "|", NULL },
	{ "int4or", 2, {INT4OID, INT4OID}, 'b', "|", NULL },
	{ "int8or", 2, {INT8OID, INT8OID}, 'b', "|", NULL },

	/* '#'  : bitwise xor */
	{ "int2xor", 2, {INT2OID, INT2OID}, 'b', "^", NULL },
	{ "int4xor", 2, {INT4OID, INT4OID}, 'b', "^", NULL },
	{ "int8xor", 2, {INT8OID, INT8OID}, 'b', "^", NULL },

	/* '~'  : bitwise not operators */
	{ "int2not", 1, {INT2OID}, 'l', "!", NULL },
	{ "int4not", 1, {INT4OID}, 'l', "!", NULL },
	{ "int8not", 1, {INT8OID}, 'l', "!", NULL },

	/* '>>' : right shift */
	{ "int2shr", 2, {INT2OID,INT4OID}, 'b', ">>", NULL },
	{ "int4shr", 2, {INT4OID,INT4OID}, 'b', ">>", NULL },
	{ "int4shr", 2, {INT8OID,INT4OID}, 'b', ">>", NULL },

	/* '<<' : left shift */
	{ "int2shl", 2, {INT2OID,INT4OID}, 'b', "<<", NULL },
	{ "int4shl", 2, {INT4OID,INT4OID}, 'b', "<<", NULL },
	{ "int4shl", 2, {INT8OID,INT4OID}, 'b', "<<", NULL },
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

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

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

/*
 * cleanup opencl resources on process exit time
 */
static void
pgstrom_devinfo_on_exit(int code, Datum arg)
{
	if (pgstrom_device_context)
		clReleaseContext(pgstrom_device_context);
}

void
pgstrom_devinfo_init(void)
{
	PgStromDeviceInfo  *dev_info;
	cl_platform_id		platform_ids[32];
	cl_device_id		device_ids[64];
	cl_uint				num_platforms;
	cl_uint				num_devices;
	cl_int				ret, pi, di;
	int					nitems = 10;
	MemoryContext		oldctx;

	devinfo_memcxt = AllocSetContextCreate(TopMemoryContext,
										   "pg_strom device info",
										   ALLOCSET_DEFAULT_MINSIZE,
										   ALLOCSET_DEFAULT_INITSIZE,
										   ALLOCSET_DEFAULT_MAXSIZE);
	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));
	memset(devcast_info_slot, 0, sizeof(devcast_info_slot));

	oldctx = MemoryContextSwitchTo(devinfo_memcxt);

	pgstrom_num_devices = 0;	
	pgstrom_device_id = palloc(sizeof(cl_device_id) * nitems);
	pgstrom_device_info = palloc(sizeof(PgStromDeviceInfo *) * nitems);

	ret = clGetPlatformIDs(lengthof(platform_ids),
						   platform_ids, &num_platforms);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: filed to get number of platforms");

	for (pi=0; pi < num_platforms; pi++)
	{
		ret = clGetDeviceIDs(platform_ids[pi],
							 CL_DEVICE_TYPE_DEFAULT,
							 lengthof(device_ids),
							 device_ids, &num_devices);
		if (ret != CL_SUCCESS)
			elog(ERROR, "OpenCL: filed to get number of devices");

		for (di=0; di < num_devices; di++)
		{
			cl_bool		dev_available;

			if (clGetDeviceInfo(device_ids[di],
								CL_DEVICE_AVAILABLE,
								sizeof(dev_available),
								&dev_available,
								NULL) != CL_SUCCESS)
				elog(ERROR, "OpenCL: failed to get properties of device");

			if (!dev_available)
				continue;


			dev_info = palloc0(sizeof(PgStromDeviceInfo));
			dev_info->pf_id = platform_ids[pi];
			dev_info->dev_id = device_ids[di];

			if (clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_COMPILER_AVAILABLE,
								sizeof(dev_info->dev_compiler_available),
								&dev_info->dev_compiler_available,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_DOUBLE_FP_CONFIG,
								sizeof(dev_info->dev_double_fp_config),
								&dev_info->dev_double_fp_config,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_EXECUTION_CAPABILITIES,
								sizeof(dev_info->dev_execution_capabilities),
								&dev_info->dev_execution_capabilities,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
								sizeof(dev_info->dev_global_mem_cache_type),
								&dev_info->dev_global_mem_cache_type,
								NULL) != CL_SUCCESS ||
				(dev_info->dev_global_mem_cache_type != CL_NONE &&
				 clGetDeviceInfo(dev_info->dev_id,
								 CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
								 sizeof(dev_info->dev_global_mem_cache_size),
								 &dev_info->dev_global_mem_cache_size,
								 NULL) != CL_SUCCESS) ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_GLOBAL_MEM_SIZE,
								sizeof(dev_info->dev_global_mem_size),
								&dev_info->dev_global_mem_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_LOCAL_MEM_SIZE,
								sizeof(dev_info->dev_local_mem_size),
								&dev_info->dev_local_mem_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_LOCAL_MEM_TYPE,
								sizeof(dev_info->dev_local_mem_type),
								&dev_info->dev_local_mem_type,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_CLOCK_FREQUENCY,
								sizeof(dev_info->dev_max_clock_frequency),
								&dev_info->dev_max_clock_frequency,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_COMPUTE_UNITS,
								sizeof(dev_info->dev_max_compute_units),
								&dev_info->dev_max_compute_units,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_CONSTANT_ARGS,
								sizeof(dev_info->dev_max_constant_args),
								&dev_info->dev_max_constant_args,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
								sizeof(dev_info->dev_max_constant_buffer_size),
								&dev_info->dev_max_constant_buffer_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_MEM_ALLOC_SIZE,
								sizeof(dev_info->dev_max_mem_alloc_size),
								&dev_info->dev_max_mem_alloc_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_PARAMETER_SIZE,
								sizeof(dev_info->dev_max_parameter_size),
								&dev_info->dev_max_parameter_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_WORK_GROUP_SIZE,
								sizeof(dev_info->dev_max_work_group_size),
								&dev_info->dev_max_work_group_size,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
								sizeof(dev_info->dev_max_work_item_dimensions),
								&dev_info->dev_max_work_item_dimensions,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_MAX_WORK_ITEM_SIZES,
								sizeof(dev_info->dev_max_work_item_sizes),
								dev_info->dev_max_work_item_sizes,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_NAME,
								sizeof(dev_info->dev_name),
								dev_info->dev_name,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_VERSION,
								sizeof(dev_info->dev_version),
								&dev_info->dev_version,
								NULL) != CL_SUCCESS ||
				clGetDeviceInfo(dev_info->dev_id,
								CL_DEVICE_PROFILE,
								sizeof(dev_info->dev_profile),
								dev_info->dev_profile,
								NULL) != CL_SUCCESS)
				elog(ERROR, "OpenCL: failed to get properties of device");

			/*
			 * Print properties of the device into log
			 */
			elog(LOG,
				 "pg_strom: device %s (%s), %u of compute units (%uMHz), "
				 "%luMB of device memory (%luKB cache), "
				 "%luKB of local memory%s%s",
				 dev_info->dev_name,
				 dev_info->dev_version,
				 dev_info->dev_max_compute_units,
				 dev_info->dev_max_clock_frequency,
				 dev_info->dev_global_mem_size / (1024 * 1024),
				 dev_info->dev_global_mem_cache_size / 1024,
				 dev_info->dev_local_mem_size / 1024,
				 (dev_info->dev_compiler_available ?
				  ", runtime compiler available" : ""),
				 (dev_info->dev_double_fp_config ?
				  ", 64bit-FP supported" : ""));

			if (pgstrom_num_devices == nitems)
			{
				cl_device_id       *id_temp
					= palloc(sizeof(cl_device_id) * (nitems + 10));
				PgStromDeviceInfo **info_temp
					= palloc(sizeof(PgStromDeviceInfo *) * (nitems + 10));

				memcpy(id_temp, pgstrom_device_id,
					   sizeof(cl_device_id) * nitems);
				memcpy(info_temp, pgstrom_device_info,
					   sizeof(PgStromDeviceInfo *) * nitems);

				pfree(pgstrom_device_id);
				pfree(pgstrom_device_info);

				pgstrom_device_id = id_temp;
				pgstrom_device_info = info_temp;

				nitems += 10;
			}
			pgstrom_device_id[pgstrom_num_devices] = device_ids[di];
			pgstrom_device_info[pgstrom_num_devices] = dev_info;
			pgstrom_num_devices++;
		}
	}
	MemoryContextSwitchTo(oldctx);

	/*
	 * Create an OpenCL context
	 */
	if (pgstrom_num_devices > 0)
	{
		pgstrom_device_context = clCreateContext(NULL,
												 pgstrom_num_devices,
												 pgstrom_device_id,
												 NULL, NULL, &ret);
		if (ret != CL_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("OpenCL failed to create execute context: %d", ret)));
		/* Clean up handler */
		on_proc_exit(pgstrom_devinfo_on_exit, 0);
    }
}
