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
static List		   *devtype_info_slot[512];
static List		   *devfunc_info_slot[1024];

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
	uint32	type_flags;
} device_type_catalog[] = {
	{ BOOLOID,		"char",		NULL,
	  DEVFUNC_VARREF_TEMPLATE(char), 0 },
	{ INT2OID,		"short",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(short), 0 },
	{ INT4OID,		"int",		NULL,
	  DEVFUNC_VARREF_TEMPLATE(int), 0 },
	{ INT8OID,		"long",		NULL,
	  DEVFUNC_VARREF_TEMPLATE(long), 0 },
	{ FLOAT4OID,	"float",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(float), 0 },
	{ FLOAT8OID,	"double",	NULL,
	  DEVFUNC_VARREF_TEMPLATE(double),
	  DEVINFO_FLAGS_DOUBLE_FP },
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
							   DatumGetFloat4(value) :
							   DatumGetFloat8(value));
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
