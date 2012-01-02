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
#include "utils/syscache.h"
#include "pg_strom.h"

typedef struct {
	CUdevice	device;
	char		dev_name[256];
	int			dev_major;
	int			dev_minor;
	int			dev_proc_nums;
	int			dev_proc_warp_sz;
	int			dev_proc_clock;
	size_t		dev_global_mem_sz;
	int			dev_global_mem_width;
	int			dev_global_mem_clock;
	int			dev_global_mem_cache_sz;
	int			dev_shared_mem_sz;
} PgStromDeviceInfo;

/*
 * Declarations
 */
#define PGSTROM_UNSUPPORTED_CAST_INTO	((void *)-1)
static List		   *devtype_info_slot[512];
static List		   *devfunc_info_slot[1024];
static List		   *devcast_info_slot[256];

static int			pgstrom_num_devices;
static PgStromDeviceInfo  *pgstrom_device_info;
static CUcontext   *pgstrom_device_context;

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

static struct {
	Oid		type_oid;
	char   *type_ident;
	char   *type_source;
	char   *type_varref;
} device_type_catalog[] = {
	{ BOOLOID,		"bool_t",	"typedef char  bool_t",
	  DEVFUNC_VARREF_TEMPLATE(bool_t) },
	{ INT2OID,		"int2_t",	"typedef short int2_t",
	  DEVFUNC_VARREF_TEMPLATE(int2_t) },
	{ INT4OID,		"int4_t",	"typedef int   int4_t",
	  DEVFUNC_VARREF_TEMPLATE(int4_t) },
	{ INT8OID,		"int8_t",	"typedef long  int8_t",
	  DEVFUNC_VARREF_TEMPLATE(int8_t) },
	{ FLOAT4OID,	"float",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(float) },
	{ FLOAT8OID,	"double",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(double) },
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

	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

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

#define DEVFUN_INTxREMIND_TEMPLATE(name,rtype,xtype,ytype)			   \
	#rtype " " #name "(unsigned char *errors, unsigned char bitmap, "  \
	#xtype " x, " #ytype " y)\n"									   \
	"{\n"															   \
	"    if (y == 0)\n"												   \
	"        *errors |= bitmap;\n"									   \
	"    return x % y;\n"											   \
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
	{ "int2not", 1, {INT2OID}, 'l', "~", NULL },
	{ "int4not", 1, {INT4OID}, 'l', "~", NULL },
	{ "int8not", 1, {INT8OID}, 'l', "~", NULL },

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
 * Error code to string representation
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
		default:
			break;
	}
	snprintf(buf, sizeof(buf), "cuda error code: %d", result);
	return pstrdup(buf);
}

int
pgstrom_get_num_devices(void)
{
	return pgstrom_num_devices;
}

void
pgstrom_set_device_context(int dev_index)
{
	CUresult	ret;

	cuInit(0);

	Assert(dev_index < pgstrom_num_devices);
	if (!pgstrom_device_context[dev_index])
	{
		ret = cuCtxCreate(&pgstrom_device_context[dev_index],
						  0,
						  pgstrom_device_info[dev_index].device);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to create device context: %s",
				 cuda_error_to_string(ret));
	}
	ret = cuCtxSetCurrent(pgstrom_device_context[dev_index]);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to set device context: %s",
			 cuda_error_to_string(ret));
}

void
pgstrom_devinfo_init(void)
{
	PgStromDeviceInfo  *devinfo;
	CUresult	ret;
	int			i, j;

	memset(devtype_info_slot, 0, sizeof(devtype_info_slot));
	memset(devfunc_info_slot, 0, sizeof(devfunc_info_slot));
	memset(devcast_info_slot, 0, sizeof(devcast_info_slot));

	ret = cuInit(0);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to initialize driver API: %s",
			 cuda_error_to_string(ret));

	ret = cuDeviceGetCount(&pgstrom_num_devices);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to get number of devices: %s",
			 cuda_error_to_string(ret));

	pgstrom_device_info
		= MemoryContextAllocZero(TopMemoryContext,
								 sizeof(PgStromDeviceInfo) *
								 pgstrom_num_devices);
	pgstrom_device_context
		= MemoryContextAllocZero(TopMemoryContext,
								 sizeof(CUcontext) *
								 pgstrom_num_devices);

	for (i=0; i < pgstrom_num_devices; i++)
	{
		static struct {
			size_t				offset;
			CUdevice_attribute	attribute;
		} dev_attributes[] = {
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
			{ offsetof(PgStromDeviceInfo, dev_global_mem_cache_sz),
			  CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE },
			{ offsetof(PgStromDeviceInfo, dev_shared_mem_sz),
			  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK },
		};

		devinfo = &pgstrom_device_info[i];

		ret = cuDeviceGet(&devinfo->device, i);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to get handle of GPU device: %s",
				 cuda_error_to_string(ret));

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
		for (j=0; j < lengthof(dev_attributes); j++)
		{
			ret = cuDeviceGetAttribute((int *)((uintptr_t) devinfo +
											   dev_attributes[j].offset),
									   dev_attributes[j].attribute,
									   devinfo->device);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "cuda: failed to get attribute of GPU device : %s",
					 cuda_error_to_string(ret));
		}

		elog(LOG,
			 "pg_strom: %s (capability v%d.%d), "
			 "%d of processor units (%d of wraps/unit, %dMHz), "
			 "%luMB of global memory (%dbits, %dMHz, %dKB of L2 cache), "
			 "%dKB of shared memory",
			 devinfo->dev_name,
			 devinfo->dev_major,
			 devinfo->dev_minor,
			 devinfo->dev_proc_nums,
			 devinfo->dev_proc_warp_sz,
			 devinfo->dev_proc_clock / 1000,
			 devinfo->dev_global_mem_sz / (1024 * 1024),
			 devinfo->dev_global_mem_width,
			 devinfo->dev_global_mem_clock / 1000,
			 devinfo->dev_global_mem_cache_sz / 1024,
			 devinfo->dev_shared_mem_sz / 1024);
	}
}
