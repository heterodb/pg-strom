/*
 * opencl_catalog.c
 *
 * Catalog of OpenCL supported type/functions
 * --
 * Copyright 2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
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
#include "utils/memutils.h"
#include "utils/syscache.h"
#include "pg_strom.h"
#include "opencl_catalog.h"

/* ------------------------------------------------------------ *
 *
 * Catalog of GPU Types
 *
 * ------------------------------------------------------------ */
#define GPUTYPE_FLAGS_X2REGS	0x0001	/* need x2 registers */
#define GPUTYPE_FLAGS_FP64		0x0002	/* need 64bits FP support  */

static struct {
	Oid		type_oid;
	int		type_flags;
	int32	type_varref;
	int32	type_conref;
} gpu_type_catalog[] = {
	{
		BOOLOID,
		0,
		GPUCMD_VARREF_BOOL,
		GPUCMD_CONREF_BOOL,
	},
	{
		INT2OID,
		0,
		GPUCMD_VARREF_INT2,
		GPUCMD_CONREF_INT2,
	},
	{
		INT4OID,
		0,
		GPUCMD_VARREF_INT4,
		GPUCMD_CONREF_INT4,
	},
	{
		INT8OID,
		GPUTYPE_FLAGS_X2REGS,
		GPUCMD_VARREF_INT8,
		GPUCMD_CONREF_INT8,
	},
	{
		FLOAT4OID,
		0,
		GPUCMD_VARREF_FLOAT4,
		GPUCMD_CONREF_FLOAT4,
	},
	{
		FLOAT8OID,
		GPUTYPE_FLAGS_X2REGS | GPUTYPE_FLAGS_FP64,
		GPUCMD_VARREF_FLOAT8,
		GPUCMD_CONREF_FLOAT8,
	},
};

static List	   *gpu_type_info_slot[128];
static bool		gpu_type_info_slot_init = false;

GpuTypeInfo *
pgstrom_gpu_type_lookup(Oid type_oid)
{
	GpuTypeInfo	   *entry;
	HeapTuple		tuple;
	Form_pg_type	typeform;
	MemoryContext	oldcxt;
	ListCell	   *cell;
	int				i, hash;

	if (!gpu_type_info_slot_init)
	{
		memset(gpu_type_info_slot, 0, sizeof(gpu_type_info_slot));
		gpu_type_info_slot_init = true;
	}

	hash = hash_uint32((uint32) type_oid) % lengthof(gpu_type_info_slot);
	foreach (cell, gpu_type_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->type_oid == type_oid)
		{
			/* supported type has _varref and _conref command */
			if (entry->type_varref != 0 && entry->type_conref != 0)
				return entry;
			return NULL;
		}
	}

	/*
	 * Not found, insert a new entry
	 */
	tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typeform = (Form_pg_type) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

	entry = palloc0(sizeof(GpuTypeInfo));
	entry->type_oid = type_oid;
	if (typeform->typnamespace == PG_CATALOG_NAMESPACE)
	{
		for (i=0; i < lengthof(gpu_type_catalog); i++)
		{
			if (gpu_type_catalog[i].type_oid == type_oid)
			{
				if (gpu_type_catalog[i].type_flags & GPUTYPE_FLAGS_X2REGS)
					entry->type_x2regs = true;
				if (gpu_type_catalog[i].type_flags & GPUTYPE_FLAGS_FP64)
					entry->type_fp64 = true;
				entry->type_varref = gpu_type_catalog[i].type_varref;
				entry->type_conref = gpu_type_catalog[i].type_conref;
				break;
			}
		}
	}
	gpu_type_info_slot[hash] = lappend(gpu_type_info_slot[hash], entry);
	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	if (entry->type_varref != 0 && entry->type_conref != 0)
		return entry;
	return NULL;
}

/* ------------------------------------------------------------ *
 *
 * Catalog of GPU Functions
 *
 * ------------------------------------------------------------ */
static struct {
	int32		func_cmd;
	const char *func_name;
	int16		func_nargs;
	Oid			func_argtypes[4];
} gpu_func_catalog[] = {
	/*
	 * Type Cast Functions
	 */
	{ GPUCMD_CAST_INT2_TO_INT4,		"int4",		1, {INT2OID} },
	{ GPUCMD_CAST_INT2_TO_INT8,		"int8",		1, {INT2OID} },
	{ GPUCMD_CAST_INT2_TO_FLOAT4,	"float4",	1, {INT2OID} },
	{ GPUCMD_CAST_INT2_TO_FLOAT8,	"float8",	1, {INT2OID} },

	{ GPUCMD_CAST_INT4_TO_INT2,		"int2",		1, {INT4OID} },
	{ GPUCMD_CAST_INT4_TO_INT8,		"int8",		1, {INT4OID} },
	{ GPUCMD_CAST_INT4_TO_FLOAT4,	"float4",	1, {INT4OID} },
	{ GPUCMD_CAST_INT4_TO_FLOAT8,	"float8",	1, {INT4OID} },

	{ GPUCMD_CAST_INT8_TO_INT2,		"int2",		1, {INT8OID} },
	{ GPUCMD_CAST_INT8_TO_INT4,		"int4",		1, {INT8OID} },
	{ GPUCMD_CAST_INT8_TO_FLOAT4,	"float4",	1, {INT8OID} },
	{ GPUCMD_CAST_INT8_TO_FLOAT8,	"float8",	1, {INT8OID} },

	{ GPUCMD_CAST_FLOAT4_TO_INT2,	"int2",		1, {FLOAT4OID} },
	{ GPUCMD_CAST_FLOAT4_TO_INT4,	"int4",		1, {FLOAT4OID} },
	{ GPUCMD_CAST_FLOAT4_TO_INT8,	"int8",		1, {FLOAT4OID} },
	{ GPUCMD_CAST_FLOAT4_TO_FLOAT8,	"float8",	1, {FLOAT4OID} },

	{ GPUCMD_CAST_FLOAT8_TO_INT2,	"int2",		1, {FLOAT8OID} },
	{ GPUCMD_CAST_FLOAT8_TO_INT4,	"int4",		1, {FLOAT8OID} },
	{ GPUCMD_CAST_FLOAT8_TO_INT8,	"int8",		1, {FLOAT8OID} },
	{ GPUCMD_CAST_FLOAT8_TO_FLOAT4,	"float4",	1, {FLOAT8OID} },
};

static List	   *gpu_func_info_slot[512];
static bool		gpu_func_info_slot_init = false;

GpuFuncInfo *
pgstrom_gpu_func_lookup(Oid func_oid)
{
	GpuFuncInfo	   *entry;
	HeapTuple		tuple;
	Form_pg_proc	procform;
	MemoryContext	oldcxt;
	ListCell	   *cell;
	int				i, hash;

	if (!gpu_func_info_slot_init)
	{
		memset(gpu_func_info_slot, 0, sizeof(gpu_func_info_slot));
		gpu_func_info_slot_init = true;
	}

	hash = hash_uint32((uint32) func_oid) % lengthof(gpu_func_info_slot);
	foreach (cell, gpu_func_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->func_oid == func_oid)
		{
			/* supported function has func_cmd */
			if (entry->func_cmd != 0)
				return entry;
			return NULL;
		}
	}

	/*
	 * Not found, insert a new entry
	 */
	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	procform = (Form_pg_proc) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

	entry = palloc0(sizeof(GpuFuncInfo) +
					sizeof(Oid) * procform->pronargs);
	entry->func_oid = func_oid;

	if (procform->pronamespace != PG_CATALOG_NAMESPACE)
		goto out;
	if (!pgstrom_gpu_type_lookup(procform->prorettype))
		goto out;
	for (i=0; i < procform->pronargs; i++)
	{
		if (!pgstrom_gpu_type_lookup(procform->proargtypes.values[i]))
			goto out;
	}

	for (i=0; i < lengthof(gpu_func_catalog); i++)
	{
		if (strcmp(NameStr(procform->proname),
				   gpu_func_catalog[i].func_name) == 0 &&
			procform->pronargs == gpu_func_catalog[i].func_nargs &&
			memcmp(procform->proargtypes.values,
				   gpu_func_catalog[i].func_argtypes,
				   sizeof(Oid) * procform->pronargs) == 0)
		{
			entry->func_cmd = gpu_func_catalog[i].func_cmd;
			entry->func_nargs = procform->pronargs;
			entry->func_rettype = procform->prorettype;
			memcpy(entry->func_argtypes,
				   procform->proargtypes.values,
				   sizeof(Oid) * procform->pronargs);
			break;
		}
	}
out:
	gpu_func_info_slot[hash] = lappend(gpu_func_info_slot[hash], entry);
	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	if (entry->func_cmd != 0)
		return entry;
	return NULL;
}
