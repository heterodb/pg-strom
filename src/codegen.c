/*
 * codegen.c
 *
 * Routines for xPU code generator
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* -------- static variables --------*/
#define DEVTYPE_INFO_NSLOTS		128
#define DEVFUNC_INFO_NSLOTS		1024
static MemoryContext	devinfo_memcxt = NULL;
static List	   *devtype_info_slot[DEVTYPE_INFO_NSLOTS];
static List	   *devtype_code_slot[DEVTYPE_INFO_NSLOTS];	/* by TypeOpCode */
static List	   *devfunc_info_slot[DEVFUNC_INFO_NSLOTS];
static List	   *devfunc_code_slot[DEVFUNC_INFO_NSLOTS];	/* by FuncOpCode */

#define TYPE_OPCODE(NAME,EXTENSION,FLAGS)			\
	static uint32_t devtype_##NAME##_hash(bool isnull, Datum value);
#include "xpu_opcodes.h"

#define TYPE_OPCODE(NAME,EXTENSION,FLAGS)			\
	{ EXTENSION, #NAME,	TypeOpCode__##NAME,			\
	  DEVKIND__ANY | (FLAGS),						\
	  devtype_##NAME##_hash,						\
	  sizeof(xpu_##NAME##_t),						\
	  __alignof__(xpu_##NAME##_t) },
static struct {
	const char	   *type_extension;
	const char	   *type_name;
	TypeOpCode		type_code;
	uint32_t		type_flags;
	devtype_hashfunc_f type_hashfunc;
	int				type_sizeof;
	int				type_alignof;
} devtype_catalog[] = {
#include "xpu_opcodes.h"
	/* alias device data types */
	{NULL, NULL, TypeOpCode__Invalid, 0, NULL, 0}
};

static struct {
	const char *type_name;
	const char *type_extension;
	const char *base_name;
	const char *base_extension;
} devtype_alias_catalog[] = {
#define TYPE_ALIAS(NAME,EXTENSION,BASE,BASE_EXTENSION)	\
	{#NAME, EXTENSION, #BASE, BASE_EXTENSION},
#include "xpu_opcodes.h"
	{NULL,NULL,NULL,NULL}
};

static const char *
get_extension_name_by_object(Oid class_id, Oid object_id)
{
	Oid		ext_oid = getExtensionOfObject(class_id, object_id);

	if (OidIsValid(ext_oid))
		return get_extension_name(ext_oid);
	return NULL;
}

static devtype_info *
build_basic_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	devtype_info   *dtype = NULL;
	HeapTuple		htup;
	Form_pg_type	__type;
	char			type_name[NAMEDATALEN+1];
	Oid				type_namespace;
	int				i;

	htup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(tcache->type_id));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for type %u", tcache->type_id);
    __type = (Form_pg_type) GETSTRUCT(htup);
	strcpy(type_name, NameStr(__type->typname));
	type_namespace = __type->typnamespace;
	ReleaseSysCache(htup);
	/* built-in types must be in pg_catalog */
	if (!ext_name && type_namespace != PG_CATALOG_NAMESPACE)
		return NULL;
	for (i=0; devtype_catalog[i].type_name != NULL; i++)
	{
		const char *__ext_name = devtype_catalog[i].type_extension;
		const char *__type_name = devtype_catalog[i].type_name;

		if ((ext_name
			 ? (__ext_name && strcmp(ext_name, __ext_name) == 0)
			 : (__ext_name == NULL)) &&
			strcmp(type_name, __type_name) == 0)
		{
			MemoryContext oldcxt = MemoryContextSwitchTo(devinfo_memcxt);

			dtype = palloc0(offsetof(devtype_info, comp_subtypes[0]));
			if (ext_name)
				dtype->type_extension = pstrdup(ext_name);
			dtype->type_code = devtype_catalog[i].type_code;
			dtype->type_oid = tcache->type_id;
			dtype->type_flags = devtype_catalog[i].type_flags;
			dtype->type_length = tcache->typlen;
			dtype->type_align = typealign_get_width(tcache->typalign);
			dtype->type_byval = tcache->typbyval;
			dtype->type_name = pstrdup(type_name);
			dtype->type_namespace = get_type_namespace(tcache->type_id);
			dtype->type_extension = (ext_name ? pstrdup(ext_name) : NULL);
			dtype->type_sizeof = devtype_catalog[i].type_sizeof;
			dtype->type_alignof = devtype_catalog[i].type_alignof;
			dtype->type_hashfunc = devtype_catalog[i].type_hashfunc;
			/* type equality functions */
			dtype->type_eqfunc = get_opcode(tcache->eq_opr);
			dtype->type_cmpfunc = tcache->cmp_proc;
			MemoryContextSwitchTo(oldcxt);

			return dtype;
		}
	}
	return NULL;		/* not found */
}

static devtype_info *
build_composite_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	TupleDesc		tupdesc = lookup_rowtype_tupdesc(tcache->type_id, -1);
	devtype_info  **subtypes = alloca(sizeof(devtype_info *) * tupdesc->natts);
	devtype_info   *dtype;
	MemoryContext	oldcxt;
	uint32_t		extra_flags = DEVKIND__ANY;
	int				j;

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		dtype = pgstrom_devtype_lookup(attr->atttypid);
		if (!dtype)
		{
			ReleaseTupleDesc(tupdesc);
			return NULL;
		}
		extra_flags &= dtype->type_flags;
		subtypes[j] = dtype;
	}
	ReleaseTupleDesc(tupdesc);

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	dtype = palloc0(offsetof(devtype_info,
							 comp_subtypes[tupdesc->natts]));
	if (ext_name)
		dtype->type_extension = pstrdup(ext_name);
	dtype->type_code = TypeOpCode__composite;
	dtype->type_oid = tcache->type_id;
	dtype->type_flags = extra_flags | DEVTYPE__USE_KVARS_SLOTBUF;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
	dtype->type_extension = NULL;
	dtype->type_name = "composite";
	dtype->type_namespace = get_type_namespace(tcache->type_id);
	dtype->type_sizeof = sizeof(xpu_composite_t);
	dtype->type_alignof = __alignof__(xpu_composite_t);
	dtype->type_hashfunc = NULL; //devtype_composite_hash;
	dtype->type_eqfunc = get_opcode(tcache->eq_opr);
	dtype->type_cmpfunc = tcache->cmp_proc;
	dtype->comp_nfields = tupdesc->natts;
	memcpy(dtype->comp_subtypes, subtypes,
		   sizeof(devtype_info *) * tupdesc->natts);
	MemoryContextSwitchTo(oldcxt);

	return dtype;
}

static devtype_info *
build_array_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	devtype_info   *elem;
	devtype_info   *dtype;
	MemoryContext	oldcxt;

	elem = pgstrom_devtype_lookup(tcache->typelem);
	if (!elem)
		return NULL;

	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	dtype = palloc0(offsetof(devtype_info, comp_subtypes[0]));
	if (ext_name)
		dtype->type_extension = pstrdup(ext_name);
	dtype->type_code = TypeOpCode__array;
	dtype->type_oid = tcache->type_id;
	dtype->type_flags = elem->type_flags | DEVTYPE__USE_KVARS_SLOTBUF;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
	dtype->type_name = "array";
	dtype->type_namespace = get_type_namespace(tcache->type_id);
	dtype->type_sizeof = sizeof(xpu_array_t);
	dtype->type_alignof = __alignof__(xpu_array_t);
	dtype->type_hashfunc = NULL; //devtype_array_hash;
	/* type equality functions */
	dtype->type_eqfunc = get_opcode(tcache->eq_opr);
	dtype->type_cmpfunc = tcache->cmp_proc;

	MemoryContextSwitchTo(oldcxt);

	return dtype;
}

static Oid
get_typeoid_by_name(const char *type_name, const char *type_extension)
{
	CatCList   *typelist;
	Oid			type_oid;

	typelist = SearchSysCacheList1(TYPENAMENSP,
								   CStringGetDatum(type_name));
	for (int i=0; i < typelist->n_members; i++)
	{
		HeapTuple		tuple = &typelist->members[i]->tuple;
		Form_pg_type	typeForm = (Form_pg_type) GETSTRUCT(tuple);
		const char	   *ext_name;

		type_oid = typeForm->oid;
		ext_name = get_extension_name_by_object(TypeRelationId, type_oid);
		if (type_extension)
		{
			if (ext_name && strcmp(type_extension, ext_name) == 0)
				goto found;
		}
		else
		{
			if (!ext_name && typeForm->typnamespace == PG_CATALOG_NAMESPACE)
				goto found;
		}
	}
	/* not found */
	type_oid = InvalidOid;
found:
	ReleaseCatCacheList(typelist);

	return type_oid;
}

static TypeCacheEntry *
__devtype_resolve_alias(TypeCacheEntry *tcache)
{
	const char	   *type_name;
	const char	   *ext_name;

	/* check alias list */
	type_name = get_type_name(tcache->type_id, false);
	ext_name = get_extension_name_by_object(TypeRelationId,
											tcache->type_id);
	for (int i=0; devtype_alias_catalog[i].type_name != NULL; i++)
	{
		Oid			__type_oid;
		const char *__type_name = devtype_alias_catalog[i].type_name;
		const char *__ext_name = devtype_alias_catalog[i].type_extension;

		if (strcmp(type_name, __type_name) != 0)
			continue;
		if (__ext_name != NULL)
		{
			if (!ext_name || strcmp(ext_name, __ext_name) != 0)
				continue;
		}
		else
		{
			Oid		namespace_oid = get_type_namespace(tcache->type_id);

			if (namespace_oid != PG_CATALOG_NAMESPACE)
				continue;
		}
		/* Hmm... it looks this type is alias of the base */
		__type_oid = get_typeoid_by_name(devtype_alias_catalog[i].base_name,
										 devtype_alias_catalog[i].base_extension);
		if (!OidIsValid(__type_oid))
			return NULL;
		tcache = lookup_type_cache(__type_oid,
								   TYPECACHE_EQ_OPR |
								   TYPECACHE_CMP_PROC);
		return __devtype_resolve_alias(tcache);
	}
	return tcache;
}

devtype_info *
pgstrom_devtype_lookup(Oid type_oid)
{
	devtype_info   *dtype;
	Datum			hash;
	uint32_t		index;
	ListCell	   *lc;
	const char	   *ext_name;
	TypeCacheEntry *tcache;

	hash = hash_any((unsigned char *)&type_oid, sizeof(Oid));
	index = hash % DEVTYPE_INFO_NSLOTS;
	foreach (lc, devtype_info_slot[index])
	{
		dtype = lfirst(lc);

		if (dtype->type_oid == type_oid)
		{
			Assert(dtype->hash == hash);
			goto found;
		}
	}
	/* try to build devtype_info entry */
	ext_name = get_extension_name_by_object(TypeRelationId, type_oid);
	tcache = lookup_type_cache(type_oid,
							   TYPECACHE_EQ_OPR |
							   TYPECACHE_CMP_PROC);
	/* if domain, move to the base type */
	while (tcache->nextDomain)
		tcache = tcache->nextDomain;
	/* if aliased device type, resolve this one */
	tcache = __devtype_resolve_alias(tcache);

	if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		/* array type */
		dtype = build_array_devtype_info(tcache, ext_name);
	}
	else if (tcache->typtype == TYPTYPE_COMPOSITE)
	{
		/* composite type */
		if (!OidIsValid(tcache->typrelid))
			elog(ERROR, "Bug? wrong composite definition at %s",
				 format_type_be(type_oid));
		dtype = build_composite_devtype_info(tcache, ext_name);
	}
	else if (tcache->typtype == TYPTYPE_BASE ||
			 tcache->typtype == TYPTYPE_RANGE)
	{
		/* base or range type */
		dtype = build_basic_devtype_info(tcache, ext_name);
	}
	else
	{
		/* not a supported type */
		dtype = NULL;
	}

	/* make a negative entry, if not device executable */
	if (!dtype)
	{
		dtype = MemoryContextAllocZero(devinfo_memcxt,
									   sizeof(devtype_info));
		dtype->type_is_negative = true;
	}
	dtype->type_oid = type_oid;
	dtype->hash = hash;
	devtype_info_slot[index] = lappend_cxt(devinfo_memcxt,
										   devtype_info_slot[index], dtype);
	if (!dtype->type_is_negative)
	{
		hash = hash_any((unsigned char *)&dtype->type_code, sizeof(TypeOpCode));
		index = hash % DEVTYPE_INFO_NSLOTS;
		devtype_code_slot[index] = lappend_cxt(devinfo_memcxt,
											   devtype_code_slot[index], dtype);
	}
found:
	if (dtype->type_is_negative)
		return NULL;
	return dtype;
}

/*
 * devtype_lookup_by_opcode
 */
static devtype_info *
devtype_lookup_by_opcode(TypeOpCode type_code)
{
	Datum		hash;
	uint32_t	index;
	ListCell   *lc;

	hash = hash_any((unsigned char *)&type_code, sizeof(TypeOpCode));
	index = hash % DEVTYPE_INFO_NSLOTS;
	foreach (lc, devtype_code_slot[index])
	{
		devtype_info *dtype = lfirst(lc);

		if (dtype->type_code == type_code)
			return dtype;
	}
	return NULL;
}

/*
 * Built-in device type hash functions
 */
static uint32_t
devtype_bool_hash(bool isnull, Datum value)
{
	bool	bval;

	if (isnull)
		return 0;
	bval = DatumGetBool(value) ? true : false;
	return hash_any((unsigned char *)&bval, sizeof(bool));
}

static inline uint32_t
__devtype_simple_hash(bool isnull, Datum value, int sz)
{
	if (isnull)
		return 0;
	return hash_any((unsigned char *)&value, sz);
}

static uint32_t
devtype_int1_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(int8_t));
}

static uint32_t
devtype_int2_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(int16_t));
}

static uint32_t
devtype_int4_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(int32_t));
}

static uint32_t
devtype_int8_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(int64_t));
}

static uint32_t
devtype_float2_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(float2_t));
}

static uint32_t
devtype_float4_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(float4_t));
}

static uint32_t
devtype_float8_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(float8_t));
}

static uint32_t
devtype_numeric_hash(bool isnull, Datum value)
{
	uint32_t	len;

	if (isnull)
		return 0;
	len = VARSIZE_ANY_EXHDR(value);
	if (len >= sizeof(uint16_t))
	{
		NumericChoice  *nc = (NumericChoice *)VARDATA_ANY(value);
		NumericDigit   *digits = NUMERIC_DIGITS(nc, nc->n_header);
		int				weight = NUMERIC_WEIGHT(nc, nc->n_header) + 1;
		int				i, ndigits = NUMERIC_NDIGITS(nc->n_header, len);
		int128_t		value = 0;

		for (i=0; i < ndigits; i++)
		{
			NumericDigit dig = digits[i];

			value = value * PG_NBASE + dig;
			if (value < 0)
				elog(ERROR, "numeric value is out of range");
		}
		if (NUMERIC_SIGN(nc->n_header) == NUMERIC_NEG)
			value = -value;
		weight = PG_DEC_DIGITS * (ndigits - weight);
		/* see, set_normalized_numeric */
		if (value == 0)
			weight = 0;
		else
		{
			while (value % 10 == 0)
			{
				value /= 10;
				weight--;
			}
		}
		return (hash_any((unsigned char *)&weight, sizeof(int16_t)) ^
				hash_any((unsigned char *)&value, sizeof(int128_t)));
	}
	elog(ERROR, "corrupted numeric header");
}

static uint32_t
devtype_bytea_hash(bool isnull, Datum value)
{
	if (isnull)
		return 0;
	return hash_any((unsigned char *)VARDATA_ANY(value), VARSIZE_ANY_EXHDR(value));
}

static uint32_t
devtype_text_hash(bool isnull, Datum value)
{
	if (isnull)
		return 0;
	return hash_any((unsigned char *)VARDATA_ANY(value), VARSIZE_ANY_EXHDR(value));
}

static uint32_t
devtype_bpchar_hash(bool isnull, Datum value)
{
	if (!isnull)
	{
		char   *s = VARDATA_ANY(value);
		int		sz = VARSIZE_ANY_EXHDR(value);

		sz = bpchartruelen(s, sz);
		return hash_any((unsigned char *)s, sz);
	}
	return 0;
}

static uint32_t
devtype_date_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(DateADT));
}

static uint32_t
devtype_time_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(TimeADT));
}

static uint32_t
devtype_timetz_hash(bool isnull, Datum value)
{
	if (!isnull)
	{
		TimeTzADT  *tmtz = DatumGetTimeTzADTP(value);

		return (hash_any((unsigned char *)&tmtz->time, sizeof(TimeADT)) ^
				hash_any((unsigned char *)&tmtz->zone, sizeof(int32_t)));
	}
	return 0;
}

static uint32_t
devtype_timestamp_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(Timestamp));
}

static uint32_t
devtype_timestamptz_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(TimestampTz));
}

static uint32_t
devtype_interval_hash(bool isnull, Datum value)
{
	if (!isnull)
	{
		Interval   *iv = DatumGetIntervalP(value);

		return hash_any((unsigned char *)iv, sizeof(Interval));
	}
	return 0;
}

static uint32_t
devtype_money_hash(bool isnull, Datum value)
{
	return __devtype_simple_hash(isnull, value, sizeof(int64_t));
}

static uint32_t
devtype_uuid_hash(bool isnull, Datum value)
{
	if (!isnull)
	{
		pg_uuid_t  *uuid = DatumGetUUIDP(value);

		return hash_any(uuid->data, UUID_LEN);
	}
	return 0;
}

static uint32_t
devtype_macaddr_hash(bool isnull, Datum value)
{
	if (!isnull)
	{
		macaddr	   *maddr = DatumGetMacaddrP(value);

		return hash_any((unsigned char *)maddr, sizeof(macaddr));
	}
	return 0;
}

static uint32_t
devtype_inet_hash(bool isnull, Datum value)
{
	if (!isnull)
	{
		inet	   *in = DatumGetInetP(value);
		int			sz;

		if (in->inet_data.family == PGSQL_AF_INET)
			sz = offsetof(inet_struct, ipaddr[4]);
		else if (in->inet_data.family == PGSQL_AF_INET6)
			sz = offsetof(inet_struct, ipaddr[16]);
		else
			elog(ERROR, "corrupted inet data");
		return hash_any((unsigned char *)&in->inet_data, sz);
	}
	return 0;
}

static uint32_t
__devtype_jsonb_hash(JsonbContainer *jc)
{
	uint32_t	hash = 0;
	uint32_t	nitems = JsonContainerSize(jc);
	char	   *base = NULL;
	char	   *data;
	uint32_t	datalen;

	if (!JsonContainerIsScalar(jc))
	{
		if (JsonContainerIsObject(jc))
		{
			base = (char *)(jc->children + 2 * nitems);
			hash ^= JB_FOBJECT;
		}
		else
		{
			base = (char *)(jc->children + nitems);
			hash ^= JB_FARRAY;
		}
	}

	for (int j=0; j < nitems; j++)
	{
		uint32_t	index = j;
		uint32_t	temp;
		JEntry		entry;

		/* hash value for key */
		if (JsonContainerIsObject(jc))
		{
			entry = jc->children[index];
			if (!JBE_ISSTRING(entry))
				elog(ERROR, "jsonb key value is not STRING");
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			temp = hash_any((unsigned char *)data, datalen);
			hash = ((hash << 1) | (hash >> 31)) ^ temp;

			index += nitems;
		}
		/* hash value for element */
		entry = jc->children[index];
		if (JBE_ISNULL(entry))
			temp = 0x01;
		else if (JBE_ISSTRING(entry))
		{
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			temp = hash_any((unsigned char *)data, datalen);
		}
		else if (JBE_ISNUMERIC(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));
			temp = devtype_numeric_hash(false, PointerGetDatum(data));
		}
		else if (JBE_ISBOOL_TRUE(entry))
			temp = 0x02;
		else if (JBE_ISBOOL_FALSE(entry))
			temp = 0x04;
		else if (JBE_ISCONTAINER(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));
			temp = __devtype_jsonb_hash((JsonbContainer *)data);
		}
		else
			elog(ERROR, "Unexpected jsonb entry (%08x)", entry);
        hash = ((hash << 1) | (hash >> 31)) ^ temp;
	}
	return hash;
}

static uint32_t
devtype_jsonb_hash(bool isnull, Datum value)
{
	if (!isnull)
	{
		JsonbContainer *jc = (JsonbContainer *) VARDATA_ANY(value);

		return __devtype_jsonb_hash(jc);
	}
	return 0;
}

static uint32_t
devtype_geometry_hash(bool isnull, Datum value)
{
	elog(ERROR, "geometry type has no device hash function");
}

static uint32_t
devtype_box2df_hash(bool isnull, Datum value)
{
	elog(ERROR, "box2df type has no device hash function");
}

/*
 * Built-in device functions/operators
 */
#define FUNC_OPCODE(SQLNAME,FN_ARGS,FN_FLAGS,DEVNAME,FUNC_COST,EXTENSION) \
	{ #SQLNAME, #FN_ARGS, FN_FLAGS, FuncOpCode__##DEVNAME, FUNC_COST, EXTENSION },
static struct {
	const char	   *func_name;
	const char	   *func_args;
	uint32_t		func_flags;
	FuncOpCode		func_code;
	int				func_cost;
	const char	   *func_extension;
} devfunc_catalog[] = {
#include "xpu_opcodes.h"
	{NULL,NULL,0,FuncOpCode__Invalid,0,NULL}
};

static devfunc_info *
pgstrom_devfunc_build(Oid func_oid, int func_nargs, Oid *func_argtypes)
{
	const char	   *fextension;
	const char	   *fname;
	Oid				fnamespace;
	Oid				frettype;
	StringInfoData	buf;
	devfunc_info   *dfunc = NULL;
	devtype_info   *dtype_rettype;
	devtype_info  **dtype_argtypes;
	MemoryContext	oldcxt;
	int				i, j, sz;

	initStringInfo(&buf);
	fname = get_func_name(func_oid);
	if (!fname)
		elog(ERROR, "cache lookup failed on procedure '%u'", func_oid);
	fnamespace = get_func_namespace(func_oid);
	frettype = get_func_rettype(func_oid);
	dtype_rettype = pgstrom_devtype_lookup(frettype);
	if (!dtype_rettype)
		goto bailout;
	dtype_argtypes = alloca(sizeof(devtype_info *) * func_nargs);
	for (j=0; j < func_nargs; j++)
	{
		dtype_argtypes[j] = pgstrom_devtype_lookup(func_argtypes[j]);
		if (!dtype_argtypes[j])
			goto bailout;
	}
	/* we expect built-in functions are in pg_catalog namespace */
	fextension = get_extension_name_by_object(ProcedureRelationId, func_oid);
	if (!fextension && fnamespace != PG_CATALOG_NAMESPACE)
		goto bailout;

	for (i=0; devfunc_catalog[i].func_name != NULL; i++)
	{
		const char *__extension = devfunc_catalog[i].func_extension;
		const char *__name = devfunc_catalog[i].func_name;
		char	   *tok, *saveptr;

		if (fextension != NULL
			? (__extension == NULL || strcmp(fextension, __extension) != 0)
			: (__extension != NULL))
			continue;
		if (strcmp(fname, __name) != 0)
			continue;

		resetStringInfo(&buf);
		appendStringInfoString(&buf, devfunc_catalog[i].func_args);
		for (tok = strtok_r(buf.data, "/", &saveptr), j=0;
			 tok != NULL && j < func_nargs;
			 tok = strtok_r(NULL, "/", &saveptr), j++)
		{
			devtype_info *dtype = dtype_argtypes[j];

			tok = __trim(tok);
			sz = strlen(tok);
			if (sz > 4 &&
				tok[0] == '_' && tok[1] == '_' &&
				tok[sz-1] == '_' && tok[sz-2] == '_')
			{
				/* __TYPE__ means variable length argument! */
				tok[sz-1] = '\0';
				if (strcmp(tok+2, dtype->type_name) != 0)
					break;
				/* must be the last argument set */
				tok = strtok_r(NULL, "/", &saveptr);
				if (tok)
					break;
				/* check whether the following arguments are identical */
				while (j < func_nargs)
				{
					if (dtype->type_oid != func_argtypes[j])
						break;
					j++;
				}
			}
			else
			{
				if (strcmp(tok, dtype->type_name) != 0)
					break;
			}
		}

		/* Ok, found an entry */
		if (!tok && j == func_nargs)
		{
			oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
			dfunc = palloc0(offsetof(devfunc_info,
									 func_argtypes[func_nargs]));
			dfunc->func_code = devfunc_catalog[i].func_code;
			if (fextension)
				dfunc->func_extension = pstrdup(fextension);
			dfunc->func_name = pstrdup(fname);
			dfunc->func_oid = func_oid;
			dfunc->func_rettype = dtype_rettype;
			dfunc->func_flags = devfunc_catalog[i].func_flags;
			dfunc->func_cost = devfunc_catalog[i].func_cost;
			dfunc->func_nargs = func_nargs;
			memcpy(dfunc->func_argtypes, dtype_argtypes,
				   sizeof(devtype_info *) * func_nargs);
			MemoryContextSwitchTo(oldcxt);
			break;
		}
	}
bailout:
	pfree(buf.data);
	return dfunc;
}

typedef struct {
	Oid		func_oid;
	int		func_nargs;
	Oid		func_argtypes[1];
} devfunc_cache_signature;

static devfunc_info *
__pgstrom_devfunc_lookup(Oid func_oid,
						 int func_nargs,
						 Oid *func_argtypes,
						 Oid func_collid)
{
	devfunc_cache_signature *signature;
	devtype_info   *dtype = NULL;
	devfunc_info   *dfunc = NULL;
	ListCell	   *lc;
	uint32_t		hash;
	int				i, j, sz;

	sz = offsetof(devfunc_cache_signature, func_argtypes[func_nargs]);
	signature = alloca(sz);
	memset(signature, 0, sz);
	signature->func_oid   = func_oid;
	signature->func_nargs = func_nargs;
	for (i=0; i < func_nargs; i++)
		signature->func_argtypes[i] = func_argtypes[i];
	hash = hash_any((unsigned char *)signature, sz);

	i = hash % DEVFUNC_INFO_NSLOTS;
	foreach (lc, devfunc_info_slot[i])
	{
		dfunc = lfirst(lc);
		if (dfunc->hash == hash &&
			dfunc->func_oid == func_oid &&
			dfunc->func_nargs == func_nargs)
		{
			for (j=0; j < func_nargs; j++)
			{
				dtype = dfunc->func_argtypes[j];
				if (dtype->type_oid != func_argtypes[j])
					break;
			}
			if (j == func_nargs)
				goto found;
		}
	}
	/* not found, build a new entry */
	dfunc = pgstrom_devfunc_build(func_oid, func_nargs, func_argtypes);
	if (!dfunc)
	{
		MemoryContext	oldcxt;

		oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
		dfunc =  palloc0(offsetof(devfunc_info, func_argtypes[func_nargs]));
		dfunc->func_oid = func_oid;
		dfunc->func_nargs = func_nargs;
		dfunc->func_is_negative = true;
		for (i=0; i < func_nargs; i++)
		{
			dtype = pgstrom_devtype_lookup(func_argtypes[i]);
			if (!dtype)
			{
				dtype = palloc0(sizeof(devtype_info));
				dtype->type_oid = func_argtypes[i];
				dtype->type_is_negative = true;
			}
			dfunc->func_argtypes[i] = dtype;
		}
		MemoryContextSwitchTo(oldcxt);
	}
	dfunc->hash = hash;
	devfunc_info_slot[i] = lappend_cxt(devinfo_memcxt,
									   devfunc_info_slot[i], dfunc);
	if (!dfunc->func_is_negative)
	{
		hash = hash_any((unsigned char *)&dfunc->func_code, sizeof(FuncOpCode));
		i = hash % DEVFUNC_INFO_NSLOTS;
		devfunc_code_slot[i] = lappend_cxt(devinfo_memcxt,
										   devfunc_code_slot[i], dfunc);
	}
found:
	if (dfunc->func_is_negative)
		return NULL;
	if (OidIsValid(func_collid) &&
		(dfunc->func_flags & DEVFUNC__LOCALE_AWARE) != 0)
	{
		/* see texteq, bpchareq */
		if (!lc_collate_is_c(func_collid))
		{
			pg_locale_t	mylocate = pg_newlocale_from_collation(func_collid);

			if (mylocate && !mylocate->deterministic)
				return NULL;	/* not supported */
		}
	}
	return dfunc;
}

devfunc_info *
pgstrom_devfunc_lookup(Oid func_oid,
					   List *func_args,
					   Oid func_collid)
{
	int			i, nargs = list_length(func_args);
	Oid		   *argtypes;
	ListCell   *lc;

	i = 0;
	argtypes = alloca(sizeof(Oid) * nargs);
	foreach (lc, func_args)
	{
		Node   *node = lfirst(lc);

		argtypes[i++] = exprType(node);
	}
	return __pgstrom_devfunc_lookup(func_oid, nargs, argtypes, func_collid);
}

static devfunc_info *
devfunc_lookup_by_opcode(FuncOpCode func_code)
{
	Datum		hash;
	uint32_t	index;
	ListCell   *lc;

	hash = hash_any((unsigned char *)&func_code, sizeof(FuncOpCode));
	index = hash % DEVFUNC_INFO_NSLOTS;
	foreach (lc, devfunc_code_slot[index])
	{
		devfunc_info *dfunc = lfirst(lc);

		if (dfunc->func_code == func_code)
			return dfunc;
	}
	return NULL;
}

/*
 * lookup special purpose devfuncs
 */
devfunc_info *
devtype_lookup_equal_func(devtype_info *dtype, Oid coll_id)
{
	if (OidIsValid(dtype->type_eqfunc))
	{
		Oid		argtypes[2];

		argtypes[0] = dtype->type_oid;
		argtypes[1] = dtype->type_oid;

		return __pgstrom_devfunc_lookup(dtype->type_eqfunc, 2, argtypes, coll_id);
	}
	return NULL;
}

devfunc_info *
devtype_lookup_compare_func(devtype_info *dtype, Oid coll_id)
{
	if (OidIsValid(dtype->type_cmpfunc))
	{
		Oid		argtypes[2];

		argtypes[0] = dtype->type_oid;
		argtypes[1] = dtype->type_oid;
		return __pgstrom_devfunc_lookup(dtype->type_cmpfunc, 2, argtypes, coll_id);
	}
	return NULL;
}

/* ----------------------------------------------------------------
 *
 * xPU pseudo code generator
 *
 * ----------------------------------------------------------------
 */
#define __Elog(fmt,...)													\
	do {																\
		ereport(context->elevel,										\
				(errcode(ERRCODE_INTERNAL_ERROR),						\
				 errmsg("(%s:%d) " fmt,	__FUNCTION__, __LINE__,			\
						##__VA_ARGS__),									\
				 errdetail("problematic expression: %s",				\
						   nodeToString(context->top_expr))));			\
		return -1;														\
	} while(0)

static int	codegen_expression_walker(codegen_context *context,
									  StringInfo buf, Expr *expr);

void
codegen_context_init(codegen_context *context, uint32_t xpu_required_flags)
{
	memset(context, 0, sizeof(codegen_context));
	context->elevel = ERROR;
	context->required_flags = (xpu_required_flags & DEVKIND__ANY);
}

static void
__appendKernExpMagicAndLength(StringInfo buf, int head_pos)
{
	static uint64_t __zero = 0;
	const kern_expression *kexp;
	int			padding = (INTALIGN(buf->len) - buf->len);
	uint32_t	magic;

	if (padding > 0)
		appendBinaryStringInfo(buf, (char *)&__zero, padding);
	kexp = (const kern_expression *)(buf->data + head_pos);
	magic = (KERN_EXPRESSION_MAGIC
			 ^ ((uint32_t)kexp->exptype << 6)
			 ^ ((uint32_t)kexp->opcode << 14));
	appendBinaryStringInfo(buf, (char *)&magic, sizeof(uint32_t));
	((kern_expression *)(buf->data + head_pos))->len = buf->len - head_pos;
}

static int
codegen_const_expression(codegen_context *context,
						 StringInfo buf, Const *con)
{
	devtype_info   *dtype;
	char			typtype;

	typtype = get_typtype(con->consttype);
	if (typtype != TYPTYPE_BASE &&
		typtype != TYPTYPE_ENUM &&
		typtype != TYPTYPE_RANGE &&
		typtype != TYPTYPE_DOMAIN)
		__Elog("unable to use type %s in Const expression (class: %c)",
			   format_type_be(con->consttype), typtype);

	dtype = pgstrom_devtype_lookup(con->consttype);
	if (!dtype)
		__Elog("type %s is not device supported",
			   format_type_be(con->consttype));
	if (buf)
	{
		kern_expression *kexp;
		int		pos, sz = 0;

		sz = offsetof(kern_expression, u.c.const_value);
		if (!con->constisnull)
		{
			if (con->constbyval)
				sz += con->constlen;
			else if (con->constlen == -1)
				sz += VARSIZE_ANY(con->constvalue);
			else
				elog(ERROR, "unsupported type length: %d", con->constlen);
		}
		kexp = alloca(sz);
		memset(kexp, 0, sz);
		kexp->exptype = dtype->type_code;
		kexp->expflags = context->kexp_flags;
		kexp->opcode = FuncOpCode__ConstExpr;
		kexp->u.c.const_type = con->consttype;
		kexp->u.c.const_isnull = con->constisnull;
		if (!con->constisnull)
		{
			if (con->constbyval)
				memcpy(kexp->u.c.const_value,
					   &con->constvalue,
					   con->constlen);
			else
				memcpy(kexp->u.c.const_value,
					   DatumGetPointer(con->constvalue),
					   VARSIZE_ANY(con->constvalue));
		}
		pos = __appendBinaryStringInfo(buf, kexp, sz);
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

static int
codegen_param_expression(codegen_context *context,
						 StringInfo buf, Param *param)
{
	kern_expression	kexp;
	devtype_info   *dtype;
	char			typtype;
	int				pos;

	if (param->paramkind != PARAM_EXTERN)
		__Elog("Only PARAM_EXTERN is supported on device: %d",
			   (int)param->paramkind);

	typtype = get_typtype(param->paramtype);
	if (typtype != TYPTYPE_BASE &&
		typtype != TYPTYPE_ENUM &&
		typtype != TYPTYPE_RANGE &&
		typtype != TYPTYPE_DOMAIN)
		__Elog("unable to use type %s in Const expression (class: %c)",
			   format_type_be(param->paramtype), typtype);

	dtype = pgstrom_devtype_lookup(param->paramtype);
	if (!dtype)
		__Elog("type %s is not device supported",
			   format_type_be(param->paramtype));
	if (buf)
	{
		memset(&kexp, 0, sizeof(kexp));
		kexp.opcode = FuncOpCode__ParamExpr;
		kexp.exptype = dtype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.u.p.param_id = param->paramid;
		pos = __appendBinaryStringInfo(buf, &kexp,
									   SizeOfKernExprParam);
		__appendKernExpMagicAndLength(buf, pos);
	}
	context->used_params = list_append_unique(context->used_params, param);

	return 0;
}

static int
codegen_var_expression(codegen_context *context,
					   StringInfo buf,
					   Expr *expr,
					   int kvar_slot_id)
{
	Oid		type_oid = exprType((Node *)expr);
	devtype_info *dtype;

	dtype = pgstrom_devtype_lookup(type_oid);
	if (!dtype)
		__Elog("type %s is not device supported", format_type_be(type_oid));

	if (buf)
	{
		kern_expression kexp;
		int			pos;

		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype = dtype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode = FuncOpCode__VarExpr;
		kexp.u.v.var_typlen   = dtype->type_length;
		kexp.u.v.var_typbyval = dtype->type_byval;
		kexp.u.v.var_typalign = dtype->type_align;
		kexp.u.v.var_slot_id  = kvar_slot_id;
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

static int
__codegen_func_expression(codegen_context *context,
						  StringInfo buf,
						  Oid func_oid,
						  List *func_args,
						  Oid func_collid)
{
	devfunc_info   *dfunc;
	devtype_info   *dtype;
	kern_expression	kexp;
	int				pos = -1;
	ListCell	   *lc;

	dfunc = pgstrom_devfunc_lookup(func_oid, func_args, func_collid);
	if (!dfunc ||
		(dfunc->func_flags & context->required_flags) != context->required_flags)
		__Elog("function %s is not supported on the target device",
			   format_procedure(func_oid));
	dtype = dfunc->func_rettype;
	context->device_cost += dfunc->func_cost;

	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = dtype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = dfunc->func_code;
	kexp.nr_args = list_length(func_args);
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	foreach (lc, func_args)
	{
		Expr   *arg = lfirst(lc);

		if (codegen_expression_walker(context, buf, arg) < 0)
			return -1;
	}
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

static int
codegen_func_expression(codegen_context *context,
						StringInfo buf, FuncExpr *func)
{
	return __codegen_func_expression(context,
									 buf,
									 func->funcid,
									 func->args,
									 func->funccollid);
}

static int
codegen_oper_expression(codegen_context *context,
						StringInfo buf, OpExpr *oper)
{
	return __codegen_func_expression(context,
									 buf,
									 get_opcode(oper->opno),
									 oper->args,
									 oper->opcollid);
}

static int
codegen_bool_expression(codegen_context *context,
						StringInfo buf, BoolExpr *b)
{
	kern_expression	kexp;
	int				pos = -1;
	ListCell	   *lc;

	memset(&kexp, 0, sizeof(kexp));
	switch (b->boolop)
	{
		case AND_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_And;
			kexp.nr_args = list_length(b->args);
			if (kexp.nr_args < 2)
				__Elog("BoolExpr(AND) must have 2 or more arguments");
			break;
		case OR_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_Or;
			kexp.nr_args = list_length(b->args);
			if (kexp.nr_args < 2)
				__Elog("BoolExpr(OR) must have 2 or more arguments");
			break;
		case NOT_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_Not;
			kexp.nr_args = list_length(b->args);
			if (kexp.nr_args != 1)
				__Elog("BoolExpr(OR) must not have multiple arguments");
			break;
		default:
			__Elog("BoolExpr has unknown bool operation (%d)", (int)b->boolop);
	}
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	foreach (lc, b->args)
	{
		Expr   *arg = lfirst(lc);

		if (codegen_expression_walker(context, buf, arg) < 0)
			return -1;
	}
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

static int
codegen_nulltest_expression(codegen_context *context,
							StringInfo buf, NullTest *nt)
{
	kern_expression	kexp;
	int				pos = -1;

	memset(&kexp, 0, sizeof(kexp));
	switch (nt->nulltesttype)
	{
		case IS_NULL:
			kexp.opcode = FuncOpCode__NullTestExpr_IsNull;
			break;
		case IS_NOT_NULL:
			kexp.opcode = FuncOpCode__NullTestExpr_IsNotNull;
			break;
		default:
			__Elog("NullTest has unknown NullTestType (%d)", (int)nt->nulltesttype);
	}
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.nr_args = 1;
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, buf, nt->arg) < 0)
		return -1;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

static int
codegen_booleantest_expression(codegen_context *context,
							   StringInfo buf, BooleanTest *bt)
{
	kern_expression	kexp;
	int				pos = -1;

	memset(&kexp, 0, sizeof(kexp));
	switch (bt->booltesttype)
	{
		case IS_TRUE:
			kexp.opcode = FuncOpCode__BoolTestExpr_IsTrue;
			break;
		case IS_NOT_TRUE:
			kexp.opcode = FuncOpCode__BoolTestExpr_IsNotTrue;
			break;
		case IS_FALSE:
			kexp.opcode = FuncOpCode__BoolTestExpr_IsFalse;
			break;
		case IS_NOT_FALSE:
			kexp.opcode = FuncOpCode__BoolTestExpr_IsNotFalse;
			break;
		case IS_UNKNOWN:
			kexp.opcode = FuncOpCode__BoolTestExpr_IsUnknown;
			break;
		case IS_NOT_UNKNOWN:
			kexp.opcode = FuncOpCode__BoolTestExpr_IsNotUnknown;
			break;
		default:
			__Elog("BooleanTest has unknown BoolTestType (%d)",
				   (int)bt->booltesttype);
	}
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.nr_args = 1;
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, buf, bt->arg) < 0)
		return -1;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

/*
 * Special case handling if (jsonb->>FIELD)::numeric is given, because it is
 * usually extracted as a text representation once then converted to numeric,
 * it is fundamentally waste of the memory in the xPU kernel space.
 * So, we add a special optimization for the numeric jsonb key references.
 */
static int
codegen_coerceviaio_expression(codegen_context *context,
							   StringInfo buf, CoerceViaIO *cvio)
{
	static struct {
		Oid			func_oid;
		const char *type_name;
		const char *type_extension;
		FuncOpCode	opcode;
	} jsonref_catalog[] = {
		/* JSONB->>KEY */
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "numeric", NULL,
		 FuncOpCode__jsonb_object_field_as_numeric},
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "int1",    "pg_strom",
		 FuncOpCode__jsonb_object_field_as_int1},
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "int2",    NULL,
		 FuncOpCode__jsonb_object_field_as_int2},
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "int4",    NULL,
		 FuncOpCode__jsonb_object_field_as_int4},
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "int8",    NULL,
		 FuncOpCode__jsonb_object_field_as_int8},
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "float2",  "pg_strom",
		 FuncOpCode__jsonb_object_field_as_float2},
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "float4",  NULL,
		 FuncOpCode__jsonb_object_field_as_float4},
		{F_JSONB_OBJECT_FIELD_TEXT,
		 "float8",  NULL,
		 FuncOpCode__jsonb_object_field_as_float8},
		/* JSONB->>NUM */
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "numeric", NULL,
		 FuncOpCode__jsonb_array_element_as_numeric},
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "int1",    "pg_strom",
		 FuncOpCode__jsonb_array_element_as_int1},
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "int2",    NULL,
		 FuncOpCode__jsonb_array_element_as_int2},
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "int4",    NULL,
		 FuncOpCode__jsonb_array_element_as_int4},
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "int8",    NULL,
		 FuncOpCode__jsonb_array_element_as_int8},
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "float2",  "pg_strom",
		 FuncOpCode__jsonb_array_element_as_float2},
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "float4",  NULL,
		 FuncOpCode__jsonb_array_element_as_float4},
		{F_JSONB_ARRAY_ELEMENT_TEXT,
		 "float8",  NULL,
		 FuncOpCode__jsonb_array_element_as_float8},
		{InvalidOid, NULL, NULL, FuncOpCode__Invalid},
	};
	Oid			func_oid = InvalidOid;
	List	   *func_args = NIL;
	devtype_info *dtype;
	kern_expression kexp;

	/* check special case if jsonb key reference */
	if (IsA(cvio->arg, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *)cvio->arg;

		func_oid  = func->funcid;
		func_args = func->args;
	}
	else if (IsA(cvio->arg, OpExpr) || IsA(cvio->arg, DistinctExpr))
	{
		OpExpr	   *op = (OpExpr *)cvio->arg;

		func_oid  = get_opcode(op->opno);
		func_args = op->args;
	}
	if (func_oid == F_JSONB_OBJECT_FIELD_TEXT)
	{
		/* sanity checks */
		if (list_length(func_args) != 2 ||
			exprType(linitial(func_args)) != JSONBOID ||
			exprType(lsecond(func_args))  != TEXTOID)
			__Elog("Not expected arguments of %s", format_procedure(func_oid));
	}
	else if (func_oid == F_JSONB_ARRAY_ELEMENT_TEXT)
	{
		/* sanity checks */
		if (list_length(func_args) != 2 ||
			exprType(linitial(func_args)) != JSONBOID ||
			exprType(lsecond(func_args))  != INT4OID)
			__Elog("Not expected arguments of %s", format_procedure(func_oid));
	}
	else
		__Elog("Not a supported CoerceViaIO: %s", nodeToString(cvio));

	dtype = pgstrom_devtype_lookup(cvio->resulttype);
	if (!dtype)
		__Elog("Not a supported CoerceViaIO: %s", nodeToString(cvio));

	memset(&kexp, 0, sizeof(kexp));
	for (int i=0; jsonref_catalog[i].type_name != NULL; i++)
	{
		if (func_oid == jsonref_catalog[i].func_oid &&
			strcmp(dtype->type_name, jsonref_catalog[i].type_name) == 0 &&
			(jsonref_catalog[i].type_extension != NULL
			 ? (dtype->type_extension != NULL &&
				strcmp(dtype->type_extension, jsonref_catalog[i].type_extension) == 0)
			 : (dtype->type_extension == NULL &&
				dtype->type_namespace == PG_CATALOG_NAMESPACE)))
		{
			ListCell   *lc;
			int			pos = -1;

			kexp.opcode = jsonref_catalog[i].opcode;
			kexp.exptype = dtype->type_code;
			kexp.expflags = context->kexp_flags;
			kexp.nr_args = 2;
			kexp.args_offset = SizeOfKernExpr(0);
			if (buf)
				pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
			foreach (lc, func_args)
			{
				Expr   *arg = lfirst(lc);

				if (codegen_expression_walker(context, buf, arg) < 0)
					return -1;
			}
			if (buf)
				__appendKernExpMagicAndLength(buf, pos);
			return 0;
		}
	}
	__Elog("Not a supported CoerceViaIO: %s", nodeToString(cvio));
	return -1;
}

/*
 * codegen_coalesce_expression
 */
static int
codegen_coalesce_expression(codegen_context *context,
							StringInfo buf, CoalesceExpr *cl)
{
	devtype_info   *dtype, *__dtype;
	kern_expression	kexp;
	int				pos = -1;
	ListCell	   *lc;

	dtype = pgstrom_devtype_lookup(cl->coalescetype);
	if (!dtype)
		__Elog("Coalesce with type '%s' is not supported",
			   format_type_be(cl->coalescetype));

	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = dtype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = FuncOpCode__CoalesceExpr;
	kexp.nr_args = list_length(cl->args);
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));

	foreach (lc, cl->args)
	{
		Expr	   *expr = lfirst(lc);
		Oid			type_oid = exprType((Node *)expr);

		__dtype = pgstrom_devtype_lookup(type_oid);
		if (!__dtype || dtype->type_code != __dtype->type_code)
			__Elog("Coalesce argument has incompatible type: %s",
				   nodeToString(cl));
		if (codegen_expression_walker(context, buf, expr) < 0)
			return -1;
	}
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

/*
 * codegen_minmax_expression
 */
static int
codegen_minmax_expression(codegen_context *context,
						  StringInfo buf, MinMaxExpr *mm)
{
	devtype_info   *dtype, *__dtype;
	kern_expression	kexp;
	int				pos = -1;
	ListCell	   *lc;

	dtype = pgstrom_devtype_lookup(mm->minmaxtype);
	if (!dtype || (dtype->type_flags & DEVTYPE__HAS_COMPARE) == 0)
		__Elog("Least/Greatest with type '%s' is not supported",
			   format_type_be(mm->minmaxtype));

	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = dtype->type_code;
	kexp.expflags = context->kexp_flags;
	if (mm->op == IS_GREATEST)
		kexp.opcode = FuncOpCode__GreatestExpr;
	else if (mm->op == IS_LEAST)
		kexp.opcode = FuncOpCode__LeastExpr;
	else
		__Elog("unknown MinMaxExpr operator: %s", nodeToString(mm));
	kexp.nr_args = list_length(mm->args);
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));

	foreach (lc, mm->args)
	{
		Expr	   *expr = lfirst(lc);
		Oid			type_oid = exprType((Node *)expr);

		__dtype = pgstrom_devtype_lookup(type_oid);
		if (!__dtype || dtype->type_code != __dtype->type_code)
			__Elog("Least/Greatest argument has incompatible type: %s",
				   nodeToString(mm));
		if (codegen_expression_walker(context, buf, expr) < 0)
			return -1;
	}
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

/*
 * codegen_relabel_expression
 */
static int
codegen_relabel_expression(codegen_context *context,
						   StringInfo buf, RelabelType *relabel)
{
	devtype_info *dtype;
	Oid			type_oid;
	TypeOpCode	type_code;

	dtype = pgstrom_devtype_lookup(relabel->resulttype);
	if (!dtype)
		__Elog("device type '%s' is not supported",
			   format_type_be(relabel->resulttype));
	type_code = dtype->type_code;

	type_oid = exprType((Node *)relabel->arg);
	dtype = pgstrom_devtype_lookup(type_oid);
	if (!dtype)
		__Elog("device type '%s' is not supported",
			   format_type_be(type_oid));
	if (dtype->type_code != type_code)
		__Elog("device type '%s' -> '%s' is not binary convertible",
			   format_type_be(relabel->resulttype),
			   format_type_be(type_oid));
	return codegen_expression_walker(context, buf, relabel->arg);
}

/*
 * codegen_casewhen_expression
 */
static int
codegen_casewhen_expression(codegen_context *context,
							StringInfo buf, CaseExpr *caseexpr)
{
	kern_expression	kexp;
	devtype_info   *rtype;
	devtype_info   *dtype;
	TypeOpCode		cond_type_code = TypeOpCode__bool;
	ListCell	   *lc;
	int				pos = -1;

	/* check result type */
	rtype = pgstrom_devtype_lookup(caseexpr->casetype);
	if (!rtype)
		__Elog("device type '%s' is not supported",
			   format_type_be(caseexpr->casetype));
	/* setup kexp */
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = rtype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = FuncOpCode__CaseWhenExpr;
	kexp.nr_args = 2 * list_length(caseexpr->args);
	kexp.args_offset = offsetof(kern_expression, u.casewhen.data);
	if (buf)
		pos = __appendZeroStringInfo(buf, kexp.args_offset);

	/* CASE expr WHEN */
	if (caseexpr->arg)
	{
		Oid		type_oid = exprType((Node *)caseexpr->arg);

		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			__Elog("device type '%s' is not supported",
				   format_type_be(type_oid));
		if ((dtype->type_flags & DEVTYPE__HAS_COMPARE) == 0)
			__Elog("device type '%s' has no comparison function",
				   dtype->type_name);
		cond_type_code = dtype->type_code;
	}

	/* WHEN ... THEN ... */
	foreach (lc, caseexpr->args)
	{
		CaseWhen   *casewhen = (CaseWhen *)lfirst(lc);
		Oid			type_oid;

		/* check case cond */
		type_oid = exprType((Node *)casewhen->expr);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			__Elog("device type '%s' is not supported",
				   format_type_be(type_oid));
		if (dtype->type_code != cond_type_code)
			__Elog("CaseWhen condition has unexpected data type: %s",
				   nodeToString(caseexpr));
		/* check case value */
		type_oid = exprType((Node *)casewhen->result);
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			__Elog("device type '%s' is not supported",
				   format_type_be(type_oid));
		if (dtype->type_code != rtype->type_code)
			__Elog("CaseWhen result has unexpected data type: %s",
				   nodeToString(caseexpr));
		/* write out CaseWhen */
		if (codegen_expression_walker(context, buf, casewhen->expr) < 0)
			return -1;
		if (codegen_expression_walker(context, buf, casewhen->result) < 0)
			return -1;
	}
	/* CASE <expression> */
	if (buf && caseexpr->arg)
	{
		kexp.u.casewhen.case_comp = (__appendZeroStringInfo(buf, 0) - pos);
		if (codegen_expression_walker(context, buf, caseexpr->arg) < 0)
			return -1;
	}
	/* ELSE <expression> */
	if (buf && caseexpr->defresult)
	{
		kexp.u.casewhen.case_else = (__appendZeroStringInfo(buf, 0) - pos);
		if (codegen_expression_walker(context, buf, caseexpr->defresult) < 0)
			return -1;
	}
	if (buf)
	{
		memcpy(buf->data + pos, &kexp, kexp.args_offset);
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

/*
 * is_expression_equals_tlist
 *
 * It checks whether the supplied expression exactly matches any entry of
 * the target-list. If found, it returns its depth and resno.
 */
static int
is_expression_equals_tlist(codegen_context *context, Expr *expr)
{
	ListCell   *lc1, *lc2;
	int			depth = 0;
	int			resno;
	int			slot_id;
	devtype_info *dtype = NULL;

	foreach (lc1, context->input_rels_tlist)
	{
		Node   *node = lfirst(lc1);

		if (IsA(node, Integer))
		{
			Index	varno = intVal(node);
			Var	   *var = (Var *)expr;

			if (IsA(var, Var) && var->varno == varno)
			{
				resno = var->varattno;
				dtype = pgstrom_devtype_lookup(var->vartype);
				goto found;
			}
		}
		else if (IsA(node, PathTarget))
		{
			PathTarget *reltarget = (PathTarget *)node;

			resno = 1;
			foreach (lc2, reltarget->exprs)
			{
				if (equal(expr, lfirst(lc2)))
				{
					dtype = pgstrom_devtype_lookup(exprType((Node *)expr));
					goto found;
				}
				resno++;
			}
		}
		else
		{
			elog(ERROR, "Bug? unexpected input_rels_tlist");
		}
		depth++;
	}
	return -1;		/* not found */

found:
	slot_id = 0;
	forboth (lc1, context->kvars_depth,
			 lc2, context->kvars_resno)
	{
		if (depth == lfirst_int(lc1) &&
			resno == lfirst_int(lc2))
		{
			return slot_id;
		}
		slot_id++;
	}
	context->kvars_depth = lappend_int(context->kvars_depth, depth);
	context->kvars_resno = lappend_int(context->kvars_resno, resno);
	if (dtype && (dtype->type_flags & DEVTYPE__USE_KVARS_SLOTBUF) != 0)
		context->kvars_types = lappend_oid(context->kvars_types, dtype->type_oid);
	else
		context->kvars_types = lappend_oid(context->kvars_types, InvalidOid);
	context->kvars_exprs = lappend(context->kvars_exprs, expr);

	return slot_id;
}

static int
codegen_expression_walker(codegen_context *context,
						  StringInfo buf, Expr *expr)
{
	int		slot_id;

	if (!expr)
		return 0;
	/* check simple var references */
	slot_id = is_expression_equals_tlist(context, expr);
	if (slot_id >= 0)
		return codegen_var_expression(context, buf, expr, slot_id);

	switch (nodeTag(expr))
	{
		case T_Const:
			return codegen_const_expression(context, buf, (Const *)expr);
		case T_Param:
			return codegen_param_expression(context, buf, (Param *)expr);
		case T_FuncExpr:
			return codegen_func_expression(context, buf, (FuncExpr *)expr);
		case T_OpExpr:
		case T_DistinctExpr:
			return codegen_oper_expression(context, buf, (OpExpr *)expr);
		case T_BoolExpr:
			return codegen_bool_expression(context, buf, (BoolExpr *)expr);
		case T_NullTest:
			return codegen_nulltest_expression(context, buf, (NullTest *)expr);
		case T_BooleanTest:
			return codegen_booleantest_expression(context, buf, (BooleanTest *)expr);
		case T_CoerceViaIO:
			return codegen_coerceviaio_expression(context, buf, (CoerceViaIO *)expr);
		case T_CoalesceExpr:
			return codegen_coalesce_expression(context, buf, (CoalesceExpr *)expr);
		case T_MinMaxExpr:
			return codegen_minmax_expression(context, buf, (MinMaxExpr *)expr);
		case T_RelabelType:
			return codegen_relabel_expression(context, buf, (RelabelType *)expr);
		case T_CaseExpr:
			return codegen_casewhen_expression(context, buf, (CaseExpr *)expr);
//		case T_CaseTestExpr:
		case T_CoerceToDomain:
		case T_ScalarArrayOpExpr:
		default:
			__Elog("not a supported expression type: %s", nodeToString(expr));
	}
	return -1;
}
#undef __Elog

/*
 * codegen_build_loadvars
 */
static int
kern_vars_defitem_comp(const void *__a, const void *__b)
{
	const kern_vars_defitem *a = __a;
	const kern_vars_defitem *b = __b;

	if (a->var_resno < b->var_resno)
		return -1;
	if (a->var_resno > b->var_resno)
		return 1;
	return 0;
}

static kern_expression *
__codegen_build_loadvars_one(codegen_context *context, int depth)
{
	kern_expression kexp;
	StringInfoData buf;
	int			slot_id = 0;
	int			nloads = 0;
	int			nslots = list_length(context->kvars_depth);
	uint32_t	kvars_offset;
	ListCell   *lc1, *lc2, *lc3;

	initStringInfo(&buf);
	buf.len = offsetof(kern_expression, u.load.kvars);
	kvars_offset = (sizeof(kern_variable) * nslots +
					sizeof(int)           * nslots);
	forthree (lc1, context->kvars_depth,
			  lc2, context->kvars_resno,
			  lc3, context->kvars_types)
	{
		kern_vars_defitem vitem;
		int		__depth = lfirst_int(lc1);
		int		__resno = lfirst_int(lc2);
		Oid		__type_oid = lfirst_oid(lc3);

		vitem.var_resno   = __resno;
		vitem.var_slot_id = slot_id++;
		vitem.var_slot_off = 0;
		if (OidIsValid(__type_oid))
		{
			devtype_info *dtype = pgstrom_devtype_lookup(__type_oid);

			Assert(dtype != NULL);
			kvars_offset = TYPEALIGN(dtype->type_alignof, kvars_offset);
			vitem.var_slot_off = kvars_offset;
			kvars_offset += dtype->type_sizeof;
		}
		if (__depth == depth)
		{
			appendBinaryStringInfo(&buf, (char *)&vitem,
								   sizeof(kern_vars_defitem));
			nloads++;
		}
	}
	if (nloads > 0)
	{
		qsort(buf.data + offsetof(kern_expression, u.load.kvars),
			  nloads,
			  sizeof(kern_vars_defitem),
			  kern_vars_defitem_comp);
	}
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype  = TypeOpCode__int4;
	kexp.expflags = context->kexp_flags;
	kexp.opcode   = FuncOpCode__LoadVars;
	kexp.args_offset = MAXALIGN(offsetof(kern_expression,
										 u.load.kvars[nloads]));
	kexp.u.load.depth = depth;
	kexp.u.load.nloads = nloads;
	memcpy(buf.data, &kexp, offsetof(kern_expression, u.load.kvars));
	__appendKernExpMagicAndLength(&buf, 0);

	return (kern_expression *)buf.data;
}

bytea *
codegen_build_scan_loadvars(codegen_context *context)
{
	kern_expression *kexp = __codegen_build_loadvars_one(context, 0);
	char	   *xpucode = NULL;

	if (kexp)
	{
		xpucode = palloc(VARHDRSZ + kexp->len);
		memcpy(xpucode + VARHDRSZ, kexp, kexp->len);
		SET_VARSIZE(xpucode, VARHDRSZ + kexp->len);
	}
	return (bytea *)xpucode;
}

bytea *
codegen_build_join_loadvars(codegen_context *context)
{
	kern_expression *kexp;
	StringInfoData buf;
	int			max_depth = -1;
	uint32_t	sz;
	char	   *result = NULL;
	ListCell   *lc;

	foreach (lc, context->kvars_depth)
	{
		int		depth = lfirst_int(lc);

		if (depth >= 0)
			max_depth = Max(max_depth, depth);
	}
	if (max_depth < 1)
		return NULL;
	sz = MAXALIGN(offsetof(kern_expression, u.pack.offset[max_depth+1]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype  = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode   = FuncOpCode__Packed;
	kexp->args_offset = sz;
	kexp->u.pack.npacked = max_depth;

	initStringInfo(&buf);
	buf.len = sz;
	for (int i=0; i < max_depth; i++)
	{
		kern_expression *karg = __codegen_build_loadvars_one(context, i+1);

		if (karg)
		{
			kexp->u.pack.offset[i]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
			kexp->nr_args++;
			pfree(karg);
		}
	}

	if (kexp->nr_args > 0)
	{
		memcpy(buf.data, kexp, sz);
		__appendKernExpMagicAndLength(&buf, 0);
		result = palloc(VARHDRSZ + buf.len);
		memcpy(result + VARHDRSZ, buf.data, buf.len);
		SET_VARSIZE(result, VARHDRSZ + buf.len);
	}
	pfree(buf.data);
	return (bytea *)result;
}

/*
 * codegen_build_scan_quals
 */
bytea *
codegen_build_scan_quals(codegen_context *context, List *dev_quals)
{
	StringInfoData buf;
	Expr	   *expr;
	char	   *result = NULL;

	Assert(context->elevel >= ERROR);
	if (dev_quals == NIL)
		return NULL;
	if (list_length(dev_quals) == 1)
		expr = linitial(dev_quals);
	else
		expr = make_andclause(dev_quals);

	initStringInfo(&buf);
	if (codegen_expression_walker(context, &buf, expr) == 0)
	{
		result = palloc(VARHDRSZ + buf.len);
		memcpy(result + VARHDRSZ, buf.data, buf.len);
		SET_VARSIZE(result, VARHDRSZ+buf.len);
	}
	pfree(buf.data);

	return (bytea *)result;
}

/*
 * __try_inject_projection_expression
 */
static int
__try_inject_projection_expression(codegen_context *context,
								   StringInfo buf,
								   Expr *expr,
								   bool write_kexp_if_exists,
								   bool *p_inject_new)
{
	ListCell   *lc1, *lc2;
	int			slot_id;

	/*
	 * When 'expr' is simple Var-reference on the input relations,
	 * we don't need to inject expression node here.
	 */
	slot_id = is_expression_equals_tlist(context, expr);
	if (slot_id >= 0)
	{
		if (write_kexp_if_exists)
			codegen_var_expression(context, buf, expr, slot_id);
		*p_inject_new = false;
		return slot_id;
	}

	/*
	 * Try to find out the expression which already has kvars-slot.
	 * If exists, we can reuse it.
	 */
	slot_id = 0;
	forboth (lc1, context->kvars_depth,
			 lc2, context->kvars_resno)
	{
		int		depth = lfirst_int(lc1);
		int		resno = lfirst_int(lc2);

		if (depth < 0 &&
			resno > 0 &&
			resno <= list_length(context->tlist_dev))
		{
			TargetEntry *tle = list_nth(context->tlist_dev, resno-1);

			if (equal(tle->expr, expr))
			{
				if (write_kexp_if_exists)
					codegen_var_expression(context, buf, expr, slot_id);
				*p_inject_new = false;
				return slot_id;
			}
		}
	}

	/*
	 * Try to assign a new kvars-slot, if 'expr' exists on tlist_dev.
	 */
	foreach (lc1, context->tlist_dev)
	{
		TargetEntry *tle = lfirst(lc1);

		if (equal(tle->expr, expr))
		{
			kern_expression kexp;
			devtype_info   *dtype;
			Oid				type_oid;
			int				pos;

			slot_id = list_length(context->kvars_depth);
			context->kvars_depth = lappend_int(context->kvars_depth, -1);
			context->kvars_resno = lappend_int(context->kvars_resno, tle->resno);
			context->kvars_types = lappend_oid(context->kvars_types, InvalidOid);

			type_oid = exprType((Node *)expr);
			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype)
				elog(ERROR, "type %s is not device supported",
					 format_type_be(type_oid));
			memset(&kexp, 0, sizeof(kexp));
			kexp.exptype = dtype->type_code;
			kexp.expflags = context->kexp_flags;
			kexp.opcode = FuncOpCode__SaveExpr;
			kexp.nr_args = 1;
			kexp.args_offset = MAXALIGN(offsetof(kern_expression,
												 u.save.data));
			kexp.u.save.slot_id = slot_id;
			pos = __appendBinaryStringInfo(buf, &kexp, kexp.args_offset);
			codegen_expression_walker(context, buf, expr);
			__appendKernExpMagicAndLength(buf, pos);

			*p_inject_new = true;

			return slot_id;
		}
	}
	return -1;		/* not found */
}

/*
 * codegen_build_projection
 */
bytea *
codegen_build_projection(codegen_context *context)
{
	kern_expression	*kexp;
	StringInfoData arg;
	StringInfoData buf;
	bool		meet_resjunk = false;
	ListCell   *lc;
	int			nexprs = 0;
	int			nattrs = 0;
	int			n, sz, pos;
	char	   *result;

	n = list_length(context->tlist_dev);
	sz = MAXALIGN(offsetof(kern_expression, u.pagg.desc[n]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);

	initStringInfo(&arg);
	foreach (lc, context->tlist_dev)
	{
		TargetEntry	*tle = lfirst(lc);
		int			slot_id;
		bool		inject_new;
		kern_projection_desc *desc;

		if (tle->resjunk)
		{
			meet_resjunk = true;
			continue;
		}
		else if (meet_resjunk)
			elog(ERROR, "Bug? a valid TLE after junk TLEs");

		slot_id = __try_inject_projection_expression(context,
													 &arg,
													 tle->expr,
													 false,
													 &inject_new);
		if (slot_id < 0)
			elog(ERROR, "Bug? expression is missing on tlist_dev: %s",
				 nodeToString(tle->expr));
		if (inject_new)
			nexprs++;

		desc = &kexp->u.proj.desc[nattrs++];
		desc->slot_id = slot_id;
	}
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Projection;
	kexp->nr_args = nexprs;
	kexp->args_offset = MAXALIGN(offsetof(kern_expression,
										  u.proj.desc[nattrs]));
	kexp->u.proj.nattrs = nattrs;
	initStringInfo(&buf);
	pos = __appendBinaryStringInfo(&buf, kexp, kexp->args_offset);
	if (nexprs > 0)
		__appendBinaryStringInfo(&buf, arg.data, arg.len);
	__appendKernExpMagicAndLength(&buf, pos);

	result = palloc(VARHDRSZ + buf.len);
	memcpy(result + VARHDRSZ, buf.data, buf.len);
	SET_VARSIZE(result, VARHDRSZ + buf.len);

	pfree(arg.data);
	pfree(buf.data);

	return (bytea *)result;
}

/*
 * __codegen_build_joinquals
 */
static kern_expression *
__codegen_build_joinquals(codegen_context *context,
						  List *join_quals,
						  List *other_quals)
{
	StringInfoData	buf;
	kern_expression	kexp;
	ListCell	   *lc;
	uint32_t		kexp_flags__saved;

	if (join_quals == NIL && other_quals == NIL)
		return NULL;

	initStringInfo(&buf);
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = FuncOpCode__JoinQuals;
	kexp.nr_args = list_length(join_quals) + list_length(other_quals);
	kexp.args_offset = SizeOfKernExpr(0);
	__appendBinaryStringInfo(&buf, &kexp, SizeOfKernExpr(0));

	foreach (lc, join_quals)
	{
		Expr   *qual = lfirst(lc);

		if (exprType((Node *)qual) != BOOLOID)
			elog(ERROR, "Bub? JOIN quals must be boolean");
		if (codegen_expression_walker(context, &buf, qual) < 0)
			return NULL;
	}

	kexp_flags__saved = context->kexp_flags;
	context->kexp_flags |= KEXP_FLAG__IS_PUSHED_DOWN;
	foreach (lc, other_quals)
	{
		Expr   *qual = lfirst(lc);

		if (exprType((Node *)qual) != BOOLOID)
			elog(ERROR, "Bub? JOIN quals must be boolean");
		if (codegen_expression_walker(context, &buf, qual) < 0)
			return NULL;
	}
	context->kexp_flags = kexp_flags__saved;
	__appendKernExpMagicAndLength(&buf, 0);

	return (kern_expression *)buf.data;
}

/*
 * codegen_build_packed_joinquals
 */
bytea *
codegen_build_packed_joinquals(codegen_context *context,
							   List *stacked_join_quals,
							   List *stacked_other_quals)
{
	kern_expression *kexp;
	StringInfoData buf;
	int			i, nrels;
	size_t		sz;
	ListCell   *lc1, *lc2;
	char	   *result = NULL;

	nrels = list_length(stacked_join_quals);
	sz = MAXALIGN(offsetof(kern_expression, u.pack.offset[nrels]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Packed;
	kexp->args_offset = sz;
	kexp->u.pack.npacked = nrels;

	initStringInfo(&buf);
	buf.len = sz;

	i = 0;
	forboth (lc1, stacked_join_quals,
			 lc2, stacked_other_quals)
	{
		List   *join_quals = lfirst(lc1);
		List   *other_quals = lfirst(lc2);
		kern_expression *karg;

		karg = __codegen_build_joinquals(context,
										 join_quals,
										 other_quals);
		if (karg)
		{
			kexp->u.pack.offset[i]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
			kexp->nr_args++;
			pfree(karg);
		}
		i++;
	}
	Assert(nrels == i);

	if (kexp->nr_args > 0)
	{
		memcpy(buf.data, kexp, sz);
		__appendKernExpMagicAndLength(&buf, 0);
		result = palloc(VARHDRSZ + buf.len);
		memcpy(result + VARHDRSZ, buf.data, buf.len);
		SET_VARSIZE(result, VARHDRSZ + buf.len);
	}
	pfree(buf.data);
	return (bytea *)result;
}

/*
 * codegen_build_packed_hashkeys
 */
static kern_expression *
__codegen_build_hash_value(codegen_context *context,
						   List *hash_keys)
{
	kern_expression *kexp;
	StringInfoData buf;
	size_t		sz = MAXALIGN(SizeOfKernExpr(0));
	ListCell   *lc;

	if (hash_keys == NIL)
		return NULL;

	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__HashValue;
	kexp->nr_args = list_length(hash_keys);
	kexp->args_offset = sz;

	initStringInfo(&buf);
	buf.len = sz;
	foreach (lc, hash_keys)
	{
		Expr   *expr = lfirst(lc);

		codegen_expression_walker(context, &buf, expr);
	}
	memcpy(buf.data, kexp, sz);
	__appendKernExpMagicAndLength(&buf, 0);

	return (kern_expression *)buf.data;
}

bytea *
codegen_build_packed_hashkeys(codegen_context *context,
							  List *stacked_hash_keys)
{
	kern_expression *kexp;
	StringInfoData buf;
	int			i, nrels;
	size_t		sz;
	ListCell   *lc;
	char	   *result = NULL;

	nrels = list_length(stacked_hash_keys);
	sz = MAXALIGN(offsetof(kern_expression, u.pack.offset[nrels]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Packed;
	kexp->args_offset = sz;
	kexp->u.pack.npacked = nrels;

	initStringInfo(&buf);
	buf.len = sz;

	i = 0;
	foreach (lc, stacked_hash_keys)
	{
		List   *hash_keys = lfirst(lc);
		kern_expression *karg;

		karg = __codegen_build_hash_value(context, hash_keys);
		if (karg)
		{
			kexp->u.pack.offset[i]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
			kexp->nr_args++;
		}
		i++;
	}
	Assert(i == nrels);

	if (kexp->nr_args > 0)
	{
		memcpy(buf.data, kexp, sz);
		__appendKernExpMagicAndLength(&buf, 0);
		result = palloc(VARHDRSZ + buf.len);
		memcpy(result + VARHDRSZ, buf.data, buf.len);
		SET_VARSIZE(result, VARHDRSZ + buf.len);
	}
	pfree(buf.data);

	return (bytea *)result;
}

/*
 * codegen_build_packed_gistevals
 */
static uint32_t
__codegen_build_one_gistquals(codegen_context *context,
							  StringInfo buf,
							  int   gist_depth,
							  Oid	gist_func_oid,
							  Oid	gist_index_oid,
							  int	gist_index_col,
							  Expr *gist_func_arg)
{
	kern_expression	kexp;
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	Oid				argtypes[2];
	uint32_t		gist_slot_id;
	uint32_t		off;
	uint32_t		__pos1;
	uint32_t		__pos2;

	/* device GiST evaluation operator */
	argtypes[0] = get_atttype(gist_index_oid,
							  gist_index_col + 1);
	argtypes[1] = exprType((Node *)gist_func_arg);
	dfunc = __pgstrom_devfunc_lookup(gist_func_oid,
									 2, argtypes,
									 InvalidOid);
	if (!dfunc)
		elog(ERROR, "Bug? cache lookup failed for device function: %u", gist_func_oid);

	/* allocation of kvars slot */
	gist_slot_id = list_length(context->kvars_depth);
	context->kvars_depth = lappend_int(context->kvars_depth,
									   gist_depth);
	context->kvars_resno = lappend_int(context->kvars_resno,
									   gist_index_col + 1);
	context->kvars_types = lappend_oid(context->kvars_types,
									   InvalidOid);
	context->kvars_exprs = lappend(context->kvars_exprs,	/* dummy */
								   makeNullConst(argtypes[0], -1, InvalidOid));
	/* setup GiSTEval expression */
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = dfunc->func_rettype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode   = FuncOpCode__GiSTEval;
	kexp.nr_args  = 1;
	kexp.args_offset = offsetof(kern_expression, u.gist.data);
	kexp.u.gist.gist_depth = gist_depth;
	kexp.u.gist.gist_oid = gist_index_oid;
	kexp.u.gist.ivar.var_resno = gist_index_col + 1;
	kexp.u.gist.ivar.var_slot_id = gist_slot_id;
	kexp.u.gist.ivar.var_slot_off = 0;
	off = __appendBinaryStringInfo(buf, &kexp, kexp.args_offset);

	/* setup binary operator to evaluate GiST index */
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype  = dfunc->func_rettype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode   = dfunc->func_code;
	kexp.nr_args  = 2;
	kexp.args_offset = offsetof(kern_expression, u.data);
	__pos1 = __appendBinaryStringInfo(buf, &kexp, kexp.args_offset);

	/* 1st argument - index reference */
	dtype = dfunc->func_argtypes[0];
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype  = dtype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode   = FuncOpCode__VarExpr;
	kexp.u.v.var_typlen   = dtype->type_length;
	kexp.u.v.var_typbyval = dtype->type_byval;
	kexp.u.v.var_typalign = dtype->type_align;
	kexp.u.v.var_slot_id = gist_slot_id;
	__pos2 = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExprVar);
	__appendKernExpMagicAndLength(buf, __pos2);

	/* 2nd argument - outer reference */
	codegen_expression_walker(context, buf, gist_func_arg);

	__appendKernExpMagicAndLength(buf, __pos1);
	__appendKernExpMagicAndLength(buf, off);

	return off;
}

void
codegen_build_packed_gistevals(codegen_context *context,
							   pgstromPlanInfo *pp_info)
{
	StringInfoData	buf;
	kern_expression	*kexp;
	int				gist_depth;
	size_t			head_sz;
	bytea		   *result = NULL;

	head_sz = MAXALIGN(offsetof(kern_expression,
								u.pack.offset[pp_info->num_rels]));
	kexp = alloca(head_sz);
	memset(kexp, 0, head_sz);
	kexp->exptype  = TypeOpCode__int4;
    kexp->expflags = context->kexp_flags;
    kexp->opcode   = FuncOpCode__Packed;
    kexp->args_offset = head_sz;
    kexp->u.pack.npacked = pp_info->num_rels;

	initStringInfo(&buf);
	buf.len = head_sz;
	gist_depth = pp_info->num_rels;
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];
		Expr	   *gist_clause = pp_inner->gist_clause;
		Expr	   *gist_func_arg = NULL;
		uint32_t	off;

		if (!gist_clause)
			continue;
		if (IsA(gist_clause, OpExpr))
		{
			OpExpr	   *op = (OpExpr *)gist_clause;

			if (op->opresulttype != BOOLOID ||
				list_length(op->args) != 2)
				elog(ERROR, "Bug? gist_clause is not binary operator: %s",
					 nodeToString(op));
			gist_func_arg = lsecond(op->args);
		}
		else if (IsA(gist_clause, FuncExpr))
		{
			FuncExpr   *func = (FuncExpr *)gist_clause;

			if (func->funcresulttype != BOOLOID ||
				list_length(func->args) != 2)
				elog(ERROR, "Bug? gist_clause is not binary function: %s",
					 nodeToString(func));
			gist_func_arg = lsecond(func->args);
		}
		else
			elog(ERROR, "Bug? gist_clause is neigher OpExpr nor FuncExpr: %s",
				 nodeToString(gist_clause));

		off = __codegen_build_one_gistquals(context,
											&buf,
											++gist_depth,
											pp_inner->gist_func_oid,
											pp_inner->gist_index_oid,
											pp_inner->gist_index_col,
											gist_func_arg);
		kexp->u.pack.offset[i] = off;
		kexp->nr_args++;
	}

	if (buf.len > head_sz)
	{
		memcpy(buf.data, kexp, head_sz);
		__appendKernExpMagicAndLength(&buf, 0);

		result = palloc(VARHDRSZ + buf.len);
		memcpy(result->vl_dat, buf.data, buf.len);
		SET_VARSIZE(result, VARHDRSZ + buf.len);
	}
	pfree(buf.data);
	pp_info->kexp_gist_evals_packed = result;
}

/*
 * __try_inject_groupby_expression
 */
static int
__try_inject_groupby_expression(codegen_context *context,
								StringInfo buf,
								Expr *expr,
								bool *p_found)
{
	int		slot_id;
	bool	found = false;

	slot_id = is_expression_equals_tlist(context, expr);
	if (slot_id >= 0)
	{
		found = true;
	}
	else
	{
		ListCell   *lc1, *lc2;

		slot_id = 0;
		forboth (lc1, context->kvars_depth,
				 lc2, context->kvars_resno)
		{
			int		depth = lfirst_int(lc1);
			int		resno = lfirst_int(lc2);

			if (depth >= 0)
				continue;
			if (resno > 0 &&
				resno <= list_length(context->tlist_dev))
			{
				TargetEntry *tle = list_nth(context->tlist_dev, resno-1);

				if (equal(expr, tle->expr))
				{
					found = true;
					break;
				}
			}
			slot_id++;
		}

		if (!found)
		{
			TargetEntry *tle;
			kern_expression kexp;
			StringInfoData temp;
			devtype_info *dtype;
			Oid			type_oid;

			/* inject expression */
			slot_id = list_length(context->kvars_depth);
			tle = makeTargetEntry(copyObject(expr),
								  list_length(context->tlist_dev) + 1,
								  psprintf("slot_%u", slot_id),
								  true);
			context->kvars_depth = lappend_int(context->kvars_depth, -1);
			context->kvars_resno = lappend_int(context->kvars_resno, tle->resno);
			context->kvars_types = lappend_int(context->kvars_types, InvalidOid);
			context->tlist_dev = lappend(context->tlist_dev, tle);

			/* SaveExpr */
			type_oid = exprType((Node *)expr);
			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype)
				elog(ERROR, "type %s is not device supported",
					 format_type_be(type_oid));

			initStringInfo(&temp);
			memset(&kexp, 0, sizeof(kexp));
			kexp.exptype = dtype->type_code;
			kexp.expflags = context->kexp_flags;
			kexp.opcode = FuncOpCode__SaveExpr;
			kexp.nr_args = 1;
			kexp.args_offset = MAXALIGN(offsetof(kern_expression,
												 u.save.data));
			kexp.u.save.slot_id = slot_id;
			__appendBinaryStringInfo(&temp, &kexp, kexp.args_offset);
			codegen_expression_walker(context, &temp, expr);
			__appendKernExpMagicAndLength(&temp, 0);
			__appendBinaryStringInfo(buf, temp.data, temp.len);
			pfree(temp.data);
		}
	}
	*p_found = found;

	return slot_id;
}

/*
 * __codegen_groupby_expression
 */
static int
__codegen_groupby_expression(codegen_context *context,
							 StringInfo buf,
							 Expr *expr)
{
	int		slot_id;
	bool	found;

	slot_id = __try_inject_groupby_expression(context, buf, expr, &found);
	if (found)
		codegen_var_expression(context, buf, expr, slot_id);
	return slot_id;
}

/*
 * codegen_build_groupby_keyhash
 */
static List *
codegen_build_groupby_keyhash(codegen_context *context,
                              pgstromPlanInfo *pp_info)
{
	StringInfoData buf;
	List		   *groupby_keys_input_slot = NIL;
	kern_expression	kexp;
	char		   *xpucode;
	ListCell	   *cell;

	Assert(pp_info->groupby_keys != NIL);

	/*
	 * Add variable slots to reference grouping-keys from the input and
	 * kds_final buffer.
	 */
	initStringInfo(&buf);
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = TypeOpCode__int4;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = FuncOpCode__HashValue;
	kexp.nr_args = list_length(pp_info->groupby_keys);
	kexp.args_offset = MAXALIGN(SizeOfKernExpr(0));
	__appendBinaryStringInfo(&buf, &kexp, kexp.args_offset);
	foreach (cell, pp_info->groupby_keys)
	{
		Expr   *key = lfirst(cell);
		int		slot_id = __codegen_groupby_expression(context, &buf, key);

		groupby_keys_input_slot = lappend_int(groupby_keys_input_slot, slot_id);
	}
	__appendKernExpMagicAndLength(&buf, 0);

	xpucode = palloc(VARHDRSZ + buf.len);
	memcpy(xpucode + VARHDRSZ, buf.data, buf.len);
	SET_VARSIZE(xpucode, VARHDRSZ + buf.len);
	pp_info->kexp_groupby_keyhash = (bytea *)xpucode;

	return groupby_keys_input_slot;
}

/*
 * codegen_build_groupby_keyload
 */
static List *
codegen_build_groupby_keyload(codegen_context *context,
							  pgstromPlanInfo *pp_info)
{
	kern_expression	*kexp;
	List	   *groupby_keys_final_slot = NIL;
	char	   *xpucode = NULL;
	ListCell   *lc1, *lc2;

	Assert(pp_info->groupby_keys != NIL);

	foreach (lc1, pp_info->groupby_keys)
	{
		Expr   *key = lfirst(lc1);

		foreach (lc2, context->tlist_dev)
		{
			TargetEntry *tle = lfirst(lc2);
			int		slot_id;

			if (tle->resjunk || !equal(key, tle->expr))
				continue;
			slot_id = list_length(context->kvars_depth);
			groupby_keys_final_slot = lappend_int(groupby_keys_final_slot, slot_id);
			context->kvars_depth = lappend_int(context->kvars_depth,
											   SPECIAL_DEPTH__PREAGG_FINAL);
			context->kvars_resno = lappend_int(context->kvars_resno, tle->resno);
			context->kvars_types = lappend_oid(context->kvars_types, InvalidOid);
			break;
		}
		if (!lc2)
			elog(ERROR, "Bug? group-by key is missing on the tlist_dev");
	}
	kexp = __codegen_build_loadvars_one(context, SPECIAL_DEPTH__PREAGG_FINAL);
	if (kexp)
	{
		xpucode = palloc(VARHDRSZ + kexp->len);
		memcpy(xpucode + VARHDRSZ, kexp, kexp->len);
		SET_VARSIZE(xpucode, VARHDRSZ + kexp->len);
		pfree(kexp);
	}
	pp_info->kexp_groupby_keyload = (bytea *)xpucode;

	return groupby_keys_final_slot;
}

/*
 * codegen_build_groupby_keycomp
 */
static void
codegen_build_groupby_keycomp(codegen_context *context,
							  pgstromPlanInfo *pp_info,
							  List *groupby_keys_input_slot,
							  List *groupby_keys_final_slot)
{
	StringInfoData	buf;
	kern_expression kexp;
	size_t			sz;
	char		   *xpucode;
	ListCell	   *lc1, *lc2, *lc3;

	Assert(pp_info->groupby_keys != NIL);

	initStringInfo(&buf);
	forthree (lc1, pp_info->groupby_keys,
			  lc2, groupby_keys_input_slot,
			  lc3, groupby_keys_final_slot)
	{
		Expr	   *key = lfirst(lc1);
		int			i_slot_id = lfirst_int(lc2);
		int			f_slot_id = lfirst_int(lc3);
		Oid			type_oid = exprType((Node *)key);
		Oid			coll_oid = exprCollation((Node *)key);
		int			pos, __pos;
		devtype_info *dtype;
		devfunc_info *dfunc;

		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "type %s is not device supported",
				 format_type_be(type_oid));
		dfunc = devtype_lookup_equal_func(dtype, coll_oid);
		if (!dfunc)
			elog(ERROR, "type %s has no device executable equal function",
				 format_type_be(type_oid));
		Assert(dfunc->func_rettype->type_code == TypeOpCode__bool &&
			   dfunc->func_nargs == 2 &&
			   dfunc->func_argtypes[0]->type_oid == type_oid &&
			   dfunc->func_argtypes[1]->type_oid == type_oid);
		memset(&kexp, 0, sizeof(kern_expression));
		kexp.exptype = dfunc->func_rettype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode  = dfunc->func_code;
		kexp.nr_args = 2;
		kexp.args_offset = SizeOfKernExpr(0);
		pos = __appendBinaryStringInfo(&buf, &kexp, kexp.args_offset);

		/* input variable */
		memset(&kexp, 0, sizeof(kern_expression));
		kexp.exptype = dtype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode = FuncOpCode__VarExpr;
		kexp.u.v.var_typlen = dtype->type_length;
		kexp.u.v.var_typbyval = dtype->type_byval;
		kexp.u.v.var_typalign = dtype->type_align;
		kexp.u.v.var_slot_id = i_slot_id;
		__pos = __appendBinaryStringInfo(&buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(&buf, __pos);	/* end of FuncExpr */

		/* final variable */
		memset(&kexp, 0, sizeof(kern_expression));
		kexp.exptype = dtype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode = FuncOpCode__VarExpr;
		kexp.u.v.var_typlen = dtype->type_length;
		kexp.u.v.var_typbyval = dtype->type_byval;
		kexp.u.v.var_typalign = dtype->type_align;
		kexp.u.v.var_slot_id = f_slot_id;
		__pos = __appendBinaryStringInfo(&buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(&buf, __pos);	/* end of VarExpr */

		__appendKernExpMagicAndLength(&buf, pos);	/* end of FuncExpr */
	}

	if (list_length(pp_info->groupby_keys) > 1)
	{
		kern_expression *payload = (kern_expression *)buf.data;
		int			payload_sz = buf.len;

		initStringInfo(&buf);
		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype = TypeOpCode__bool;
		kexp.expflags = context->kexp_flags;
		kexp.opcode = FuncOpCode__BoolExpr_And;
		kexp.nr_args = list_length(pp_info->groupby_keys);
		kexp.args_offset = SizeOfKernExpr(0);
		__appendBinaryStringInfo(&buf, &kexp, SizeOfKernExpr(0));
		__appendBinaryStringInfo(&buf, payload, payload_sz);
		__appendKernExpMagicAndLength(&buf, 0);
		pfree(payload);
	}
	sz = ((kern_expression *)buf.data)->len;

	xpucode = palloc(VARHDRSZ + sz);
	memcpy(xpucode + VARHDRSZ, buf.data, sz);
	SET_VARSIZE(xpucode, VARHDRSZ + sz);
	pfree(buf.data);

	pp_info->kexp_groupby_keycomp = (bytea *)xpucode;
}

/*
 * __codegen_build_groupby_actions
 */
static void
__codegen_build_groupby_actions(codegen_context *context,
								pgstromPlanInfo *pp_info)
{
	StringInfoData	buf;
	int			nattrs = list_length(pp_info->groupby_actions);
	int			nexprs = 0;
	int			index = 0;
	size_t		head_sz = MAXALIGN(offsetof(kern_expression, u.pagg.desc[nattrs]));
	char	   *xpucode;
	ListCell   *lc1, *lc2;
	kern_expression *kexp;

	kexp = alloca(head_sz);
	memset(kexp, 0, head_sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode = FuncOpCode__AggFuncs;
	kexp->nr_args = 0;
	kexp->args_offset = head_sz;
	kexp->u.pagg.nattrs = nattrs;

	initStringInfo(&buf);
	foreach (lc1, pp_info->groupby_actions)
	{
		/* MEMO: context->tlist_dev may be updated in the loop, so we cannot use
		 * forboth() macro here.
		 */
		TargetEntry *tle = list_nth(context->tlist_dev, index);
		int			action = lfirst_int(lc1);
		int			slot_id;
		bool		inject_new;
		kern_aggregate_desc *desc;

		Assert(!tle->resjunk);
		desc = &kexp->u.pagg.desc[index++];
		desc->action = action;
		if (action == KAGG_ACTION__VREF)
		{
			slot_id = __try_inject_projection_expression(context,
														 &buf,
														 tle->expr,
														 false,
														 &inject_new);
			if (slot_id < 0)
				elog(ERROR, "Bug? grouping-key is not on the kvars-slot");
			if (inject_new)
				nexprs++;
			desc->arg0_slot_id = slot_id;
		}
		else
		{
			FuncExpr   *func = (FuncExpr *)tle->expr;
			int			count = 0;

			Assert(IsA(func, FuncExpr) && list_length(func->args) <= 2);
			foreach (lc2, func->args)
			{
				Expr   *fn_arg = lfirst(lc2);

				slot_id = __try_inject_projection_expression(context,
															 &buf,
															 fn_arg,
															 false,
															 &inject_new);
				if (slot_id < 0)
					elog(ERROR, "Bug? partial-aggregate-function argument is missing");
				if (inject_new)
					nexprs++;
				if (count == 0)
					desc->arg0_slot_id = slot_id;
				else if (count == 1)
					desc->arg1_slot_id = slot_id;
				else
					elog(ERROR, "Bug? too much partial function arguments");
				count++;
			}
		}
	}
	Assert(index == nattrs);

	if (nexprs == 0)
	{
		Assert(buf.len == 0);
		__appendBinaryStringInfo(&buf, kexp, head_sz);
		__appendKernExpMagicAndLength(&buf, 0);
	}
	else
	{
		char   *payload = buf.data;
		size_t	payload_sz = buf.len;

		kexp->nr_args = nexprs;
		initStringInfo(&buf);
		__appendBinaryStringInfo(&buf, kexp, head_sz);
		__appendBinaryStringInfo(&buf, payload, payload_sz);
		__appendKernExpMagicAndLength(&buf, 0);
		pfree(payload);
	}
	xpucode = palloc(VARHDRSZ + buf.len);
	memcpy(xpucode + VARHDRSZ, buf.data, buf.len);
	SET_VARSIZE(xpucode, VARHDRSZ + buf.len);
	pfree(buf.data);

	pp_info->kexp_groupby_actions = (bytea *)xpucode;
}

/*
 * codegen_build_groupby_actions
 */
void
codegen_build_groupby_actions(codegen_context *context,
							  pgstromPlanInfo *pp_info)
{
	List   *groupby_keys_input_slot = NIL;
	List   *groupby_keys_final_slot = NIL;

	if (pp_info->groupby_keys != NIL)
	{
		groupby_keys_input_slot = codegen_build_groupby_keyhash(context, pp_info);
		groupby_keys_final_slot = codegen_build_groupby_keyload(context, pp_info);
		codegen_build_groupby_keycomp(context, pp_info,
									  groupby_keys_input_slot,
									  groupby_keys_final_slot);
	}
	__codegen_build_groupby_actions(context, pp_info);
}

/*
 * pgstrom_xpu_expression
 */
bool
pgstrom_xpu_expression(Expr *expr,
					   uint32_t required_xpu_flags,
					   List *input_rels_tlist,
					   int *p_devcost)
{
	codegen_context context;

	Assert((required_xpu_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU ||
		   (required_xpu_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU);
	memset(&context, 0, sizeof(context));
	context.elevel = DEBUG2;
	context.top_expr = expr;
	context.required_flags = (required_xpu_flags & DEVKIND__ANY);
	context.input_rels_tlist = input_rels_tlist;

	if (!expr)
		return false;
	if (IsA(expr, List))
	{
		List   *l = (List *)expr;

		if (list_length(l) == 1)
			expr = linitial(l);
		else
			expr = make_andclause(l);
	}
	if (codegen_expression_walker(&context, NULL, expr) < 0)
		return false;
	if (p_devcost)
		*p_devcost = context.device_cost;
	return true;
}

/*
 * pgstrom_gpu_expression
 *
 * checks whether the expression is executable on GPU devices.
 */
bool
pgstrom_gpu_expression(Expr *expr,
					   List *input_rels_tlist,
					   int *p_devcost)
{
	return pgstrom_xpu_expression(expr,
								  DEVKIND__NVIDIA_GPU,
								  input_rels_tlist,
								  p_devcost);
}

/*
 * pgstrom_dpu_expression
 *
 * checks whether the expression is executable on DPU devices.
 */
bool
pgstrom_dpu_expression(Expr *expr,
					   List *input_rels_tlist,
					   int *p_devcost)
{
	return pgstrom_xpu_expression(expr,
								  DEVKIND__NVIDIA_DPU,
								  input_rels_tlist,
								  p_devcost);
}

/*
 * pgstrom_xpucode_to_string
 *
 * transforms xPU code to human readable form.
 */
static void
__xpucode_to_cstring(StringInfo buf,
					 const kern_expression *kexp,
					 const CustomScanState *css,	/* optional */
					 ExplainState *es,				/* optional */
					 List *dcontext);				/* optionsl */

static void
__xpucode_const_cstring(StringInfo buf, const kern_expression *kexp)
{
	devtype_info   *dtype = devtype_lookup_by_opcode(kexp->exptype);

	if (kexp->u.c.const_isnull)
	{
		appendStringInfo(buf, "{Const(%s): value=NULL}", dtype->type_name);
	}
	else
	{
		int16	type_len;
		bool	type_byval;
		char	type_align;
		char	type_delim;
		Oid		type_ioparam;
		Oid		type_outfunc;
		Datum	datum = 0;
		Datum	label;

		get_type_io_data(kexp->u.c.const_type,
						 IOFunc_output,
						 &type_len,
						 &type_byval,
						 &type_align,
						 &type_delim,
						 &type_ioparam,
						 &type_outfunc);
		if (type_byval)
			memcpy(&datum, kexp->u.c.const_value, type_len);
		else
			datum = PointerGetDatum(kexp->u.c.const_value);
		label = OidFunctionCall1(type_outfunc, datum);
		appendStringInfo(buf, "{Const(%s): value='%s'}",
						 dtype->type_name,
						 DatumGetCString(label));
	}
}

static void
__xpucode_param_cstring(StringInfo buf, const kern_expression *kexp)
{
	devtype_info   *dtype = devtype_lookup_by_opcode(kexp->exptype);

	appendStringInfo(buf, "{Param(%s): param_id=%u}",
					 dtype->type_name,
					 kexp->u.p.param_id);
}

static void
__xpucode_var_cstring(StringInfo buf, const kern_expression *kexp)
{
	devtype_info   *dtype = devtype_lookup_by_opcode(kexp->exptype);
			
	appendStringInfo(buf, "{Var(%s): slot_id=%d}",
					 dtype->type_name,
					 kexp->u.v.var_slot_id);
}

static void
__xpucode_loadvars_cstring(StringInfo buf,
						   const kern_expression *kexp,
						   const CustomScanState *css,
						   ExplainState *es,
						   List *dcontext)
{
	bool	verbose = false;
	int		depth = kexp->u.load.depth;
	int		i;

	Assert(kexp->nr_args == 0);
	appendStringInfo(buf, "{LoadVars: depth=%d", depth);
	if (kexp->u.load.nloads > 0)
		appendStringInfo(buf, " kvars=[");

	if (css)
	{
		CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
		verbose = (cscan->custom_plans != NIL);
	}

	for (i=0; i < kexp->u.load.nloads; i++)
	{
		const kern_vars_defitem *vitem = &kexp->u.load.kvars[i];

		if (i > 0)
			appendStringInfo(buf, ", ");
		if (!css)
		{
			appendStringInfo(buf, "(slot_id=%u, resno=%d)",
							 vitem->var_slot_id,
							 vitem->var_resno);
		}
		else if (depth == 0)
		{
			CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
			TupleDesc	tupdesc = RelationGetDescr(css->ss.ss_currentRelation);
			const FormData_pg_attribute *attr;
			Var		   *kvar;

			if (vitem->var_resno == 0)
			{
				kvar = makeVar(cscan->scan.scanrelid,
							   0,
							   tupdesc->tdtypeid,
							   tupdesc->tdtypmod,
							   InvalidOid, 0);
			}
			else
			{
				if (vitem->var_resno > 0)
					attr = TupleDescAttr(tupdesc, vitem->var_resno - 1);
				else if (vitem->var_resno < 0)
					attr = SystemAttributeDefinition(vitem->var_resno);
				kvar = makeVar(cscan->scan.scanrelid,
							   attr->attnum,
							   attr->atttypid,
							   attr->atttypmod,
							   attr->attcollation, 0);
			}
			appendStringInfo(buf, "%u:%s",
							 vitem->var_slot_id,
							 deparse_expression((Node *)kvar,
												dcontext,
												verbose, false));
			pfree(kvar);
		}
		else if (depth < 0)
		{
			CustomScan *cscan = (CustomScan *)css->ss.ps.plan;

			if (vitem->var_resno >= 1 ||
				vitem->var_resno <= list_length(cscan->custom_scan_tlist))
			{
				TargetEntry *tle = list_nth(cscan->custom_scan_tlist,
											vitem->var_resno-1);
				appendStringInfo(buf, "%u:%s",
								 vitem->var_slot_id,
								 deparse_expression((Node *)tle->expr,
													dcontext,
													verbose, false));
			}
			else
			{
				appendStringInfo(buf, "var(slot_id=%u)", vitem->var_slot_id);
			}
		}
		else
		{
			CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
			Plan	   *plan;
			TargetEntry *tle;

			plan = list_nth(cscan->custom_plans, depth - 1);
			tle = list_nth(plan->targetlist, vitem->var_resno - 1);
			appendStringInfo(buf, "%u:%s",
							 vitem->var_slot_id,
							 deparse_expression((Node *)tle->expr,
												dcontext,
												verbose, false));
		}
	}
	if (kexp->u.load.nloads > 0)
		appendStringInfo(buf, "]");
}

static void
__xpucode_gisteval_cstring(StringInfo buf,
                           const kern_expression *kexp,
                           const CustomScanState *css,
                           ExplainState *es,
                           List *dcontext)
{
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);
	devtype_info   *dtype;
	Oid				type_oid;

	Assert(kexp->nr_args == 1 &&
		   kexp->exptype == karg->exptype);
	type_oid = get_atttype(kexp->u.gist.gist_oid,
						   kexp->u.gist.ivar.var_resno);
	dtype = pgstrom_devtype_lookup(type_oid);
	if (!dtype)
		elog(ERROR, "device type lookup failed for %u", type_oid);

	appendStringInfo(buf, "{GiSTEval(%s): ivar=<depth=%u, slot_id=%u, col=%s.%s(%s)>, arg=",
					 dtype->type_name,
					 kexp->u.gist.gist_depth,
					 kexp->u.gist.ivar.var_slot_id,
					 get_rel_name(kexp->u.gist.gist_oid),
					 get_attname(kexp->u.gist.gist_oid,
								 kexp->u.gist.ivar.var_resno, false),
					 dtype->type_name);
	__xpucode_to_cstring(buf, karg, css, es, dcontext);
	appendStringInfo(buf, "}");
}

static void
__xpucode_aggfuncs_cstring(StringInfo buf,
						   const kern_expression *kexp,
						   const CustomScanState *css,	/* optional */
						   ExplainState *es,			/* optional */
						   List *dcontext)
{
	appendStringInfo(buf, "{AggFuncs <");
	for (int j=0; j < kexp->u.pagg.nattrs; j++)
	{
		const kern_aggregate_desc *desc = &kexp->u.pagg.desc[j];

		if (j > 0)
			appendStringInfo(buf, ", ");
		switch (desc->action)
		{
			case KAGG_ACTION__VREF:
				appendStringInfo(buf, "vref[%d]", desc->arg0_slot_id);
				break;
			case KAGG_ACTION__NROWS_ANY:
				appendStringInfo(buf, "nrows[*]");
				break;
			case KAGG_ACTION__NROWS_COND:
				appendStringInfo(buf, "nrows[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PMIN_INT32:
				appendStringInfo(buf, "pmin::int32[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PMIN_INT64:
				appendStringInfo(buf, "pmin::int64[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PMIN_FP64:
				appendStringInfo(buf, "pmin::fp64[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PMAX_INT32:
				appendStringInfo(buf, "pmax::int32[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PMAX_INT64:
				appendStringInfo(buf, "pmax::int64[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PMAX_FP64:
				appendStringInfo(buf, "pmax::fp64[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PSUM_INT:
				appendStringInfo(buf, "psum::int[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PSUM_FP:
				appendStringInfo(buf, "psum::fp[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PAVG_INT:
				appendStringInfo(buf, "pavg::int[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__PAVG_FP:
				appendStringInfo(buf, "pavg::fp[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__STDDEV:
				appendStringInfo(buf, "stddev[%d]",
								 desc->arg0_slot_id);
				break;
			case KAGG_ACTION__COVAR:
				appendStringInfo(buf, "stddev[%d,%d]",
								 desc->arg0_slot_id,
								 desc->arg1_slot_id);
				break;
			default:
				appendStringInfo(buf, "unknown[%d,%d]",
								 desc->arg0_slot_id,
								 desc->arg1_slot_id);
				break;
		}
	}
	appendStringInfo(buf, ">");
}

static void
__xpucode_to_cstring(StringInfo buf,
					 const kern_expression *kexp,
					 const CustomScanState *css,	/* optional */
					 ExplainState *es,				/* optional */
					 List *dcontext)				/* optionsl */
{
	const kern_expression *karg;
	int		i, pos;

	switch (kexp->opcode)
	{
		case FuncOpCode__ConstExpr:
			__xpucode_const_cstring(buf, kexp);
			return;
		case FuncOpCode__ParamExpr:
			__xpucode_param_cstring(buf, kexp);
			return;
		case FuncOpCode__VarExpr:
			__xpucode_var_cstring(buf, kexp);
			return;
		case FuncOpCode__Projection:
			appendStringInfo(buf, "{Projection <");
			for (int j=0; j < kexp->u.proj.nattrs; j++)
			{
				const kern_projection_desc *desc = &kexp->u.proj.desc[j];
				if (j > 0)
					appendStringInfoChar(buf, ',');
				appendStringInfo(buf, "%d", desc->slot_id);
			}
			appendStringInfo(buf, ">");
			break;
		case FuncOpCode__LoadVars:
			__xpucode_loadvars_cstring(buf, kexp, css, es, dcontext);
			break;
		case FuncOpCode__GiSTEval:
			__xpucode_gisteval_cstring(buf, kexp, css, es, dcontext);
			break;
		case FuncOpCode__HashValue:
			appendStringInfo(buf, "{HashValue");
			break;
		case FuncOpCode__JoinQuals:
			appendStringInfo(buf, "{JoinQuals: ");
			for (i=0, karg=KEXP_FIRST_ARG(kexp);
				 i < kexp->nr_args;
				 i++, karg=KEXP_NEXT_ARG(karg))
			{
				if (!__KEXP_IS_VALID(kexp,karg))
					elog(ERROR, "XpuCode looks corrupted");
				appendStringInfo(buf, "%s ", i > 0 ? "," : "");
				if ((karg->expflags & KEXP_FLAG__IS_PUSHED_DOWN) != 0)
					appendStringInfoChar(buf, '<');
				__xpucode_to_cstring(buf, karg, css, es, dcontext);
				if ((karg->expflags & KEXP_FLAG__IS_PUSHED_DOWN) != 0)
					appendStringInfoChar(buf, '>');
			}
			appendStringInfo(buf, "}");
			return;
		case FuncOpCode__SaveExpr:
			appendStringInfo(buf, "{SaveExpr slot=%d:",
							 kexp->u.save.slot_id);
			break;
		case FuncOpCode__AggFuncs:
			__xpucode_aggfuncs_cstring(buf, kexp, css, es, dcontext);
			break;
		case FuncOpCode__Packed:
			appendStringInfo(buf, "{Packed");
			pos = buf->len;
			for (i=0; i < kexp->u.pack.npacked; i++)
			{
				karg = __PICKUP_PACKED_KEXP(kexp, i);
				if (!karg)
					continue;
				if (!__KEXP_IS_VALID(kexp,karg))
					elog(ERROR, "XpuCode looks corrupted");
				if (buf->len > pos)
					appendStringInfoChar(buf,',');
				appendStringInfo(buf, " items[%u]=", i);
				__xpucode_to_cstring(buf, karg, css, es, dcontext);
			}
			appendStringInfo(buf, "}");
			return;
		case FuncOpCode__BoolExpr_And:
			appendStringInfo(buf, "{Bool::AND");
			break;
		case FuncOpCode__BoolExpr_Or:
			appendStringInfo(buf, "{Bool::OR");
			break;
		case FuncOpCode__BoolExpr_Not:
			appendStringInfo(buf, "{Bool::NOT");
			break;
		case FuncOpCode__NullTestExpr_IsNull:
			appendStringInfo(buf, "{IsNull");
			break;
		case FuncOpCode__NullTestExpr_IsNotNull:
			appendStringInfo(buf, "{IsNotNull");
			break;
		case FuncOpCode__BoolTestExpr_IsTrue:
			appendStringInfo(buf, "{BoolTest::IsTrue");
			break;
		case FuncOpCode__BoolTestExpr_IsNotTrue:
			appendStringInfo(buf, "{BoolTest::IsNotTrue");
			break;
		case FuncOpCode__BoolTestExpr_IsFalse:
			appendStringInfo(buf, "{BoolTest::IsFalse");
			break;
		case FuncOpCode__BoolTestExpr_IsNotFalse:
			appendStringInfo(buf, "{BoolTest::IsNotFalse");
			break;
		case FuncOpCode__BoolTestExpr_IsUnknown:
			appendStringInfo(buf, "{BoolTest::IsUnknown");
			break;
		case FuncOpCode__BoolTestExpr_IsNotUnknown:
			appendStringInfo(buf, "{BoolTest::IsNotUnknown");
			break;
		case FuncOpCode__CoalesceExpr:
			appendStringInfo(buf, "{Coalesce");
			break;
		case FuncOpCode__LeastExpr:
			appendStringInfo(buf, "{Least");
			break;
		case FuncOpCode__GreatestExpr:
			appendStringInfo(buf, "{Greatest");
			break;
		case FuncOpCode__CaseWhenExpr:
			appendStringInfo(buf, "{Case:");
			Assert((kexp->nr_args % 2) == 0);
			if (kexp->u.casewhen.case_comp != 0)
			{
				karg = (kern_expression *)
					((char *)kexp + kexp->u.casewhen.case_comp);
				if (!__KEXP_IS_VALID(kexp,karg))
					elog(ERROR, "XpuCode looks corrupted");
				appendStringInfo(buf, " <key=");
				__xpucode_to_cstring(buf, karg, css, es, dcontext);
				appendStringInfo(buf, ">");
			}

			for (i = 0, karg = KEXP_FIRST_ARG(kexp);
				 i < kexp->nr_args;
				 i += 2, karg = KEXP_NEXT_ARG(karg))
			{
				if (!__KEXP_IS_VALID(kexp, karg))
					elog(ERROR, "XpuCode looks corrupted");
				appendStringInfo(buf, "%s <when=", (i > 0 ? "," : ""));
				__xpucode_to_cstring(buf, karg, css, es, dcontext);
				appendStringInfo(buf, ", then=");

				karg = KEXP_NEXT_ARG(karg);
				if (!__KEXP_IS_VALID(kexp, karg))
                    elog(ERROR, "XpuCode looks corrupted");
				__xpucode_to_cstring(buf, karg, css, es, dcontext);
				appendStringInfo(buf, ">");
			}

			if (kexp->u.casewhen.case_else != 0)
			{
				karg = (kern_expression *)
					((char *)kexp + kexp->u.casewhen.case_else);
				if (!__KEXP_IS_VALID(kexp, karg))
                    elog(ERROR, "XpuCode looks corrupted");
				appendStringInfo(buf, " <else=");
				__xpucode_to_cstring(buf, karg, css, es, dcontext);
				appendStringInfo(buf, ">");
			}
			appendStringInfo(buf, "}");
			return;

		default:
			{
				static struct {
					FuncOpCode	func_code;
					const char *func_name;
					const char *rettype_name;
				} devonly_funcs_catalog[] = {
#define DEVONLY_FUNC_OPCODE(RET_TYPE,DEV_NAME,a,b,c)	\
				{FuncOpCode__##DEV_NAME, #DEV_NAME, #RET_TYPE},
#include "xpu_opcodes.h"
					{FuncOpCode__Invalid,NULL,NULL}
				};
				devfunc_info *dfunc = devfunc_lookup_by_opcode(kexp->opcode);
				devtype_info *dtype;

				if (dfunc)
				{
					dtype = devtype_lookup_by_opcode(kexp->exptype);
					appendStringInfo(buf, "{Func::%s(%s)",
                                     dfunc->func_name,
                                     dtype->type_name);
				}
				else
				{
					const char *func_name = NULL;
					const char *rettype_name = NULL;

					for (int i=0; devonly_funcs_catalog[i].func_code; i++)
					{
						if (devonly_funcs_catalog[i].func_code == kexp->opcode)
						{
							func_name = devonly_funcs_catalog[i].func_name;
							rettype_name = devonly_funcs_catalog[i].rettype_name;
							break;
						}
					}
					if (!func_name || !rettype_name)
						elog(ERROR, "unknown device only function: %u", kexp->opcode);
					appendStringInfo(buf, "{DevOnlyFunc::%s(%s)",
									 func_name,
									 rettype_name);
				}
			}
			break;
	}
	if (kexp->nr_args > 0)
	{
		if (kexp->nr_args == 1)
			appendStringInfo(buf, " arg=");
		else
			appendStringInfo(buf, " args=[");

		for (i=0, karg=KEXP_FIRST_ARG(kexp);
			 i < kexp->nr_args;
			 i++, karg=KEXP_NEXT_ARG(karg))
		{
			if (!__KEXP_IS_VALID(kexp,karg))
				elog(ERROR, "XpuCode looks corrupted");
			if (i > 0)
				appendStringInfo(buf, ", ");
			__xpucode_to_cstring(buf, karg, css, es, dcontext);
		}
		if (kexp->nr_args > 1)
			appendStringInfoChar(buf, ']');
	}
	appendStringInfoChar(buf, '}');
}

void
pgstrom_explain_xpucode(const CustomScanState *css,
						ExplainState *es,
						List *dcontext,
						const char *label,
						bytea *xpucode)
{
	StringInfoData	buf;

	if (xpucode)
	{
		const kern_expression *kexp = (const kern_expression *)VARDATA(xpucode);

		initStringInfo(&buf);
		__xpucode_to_cstring(&buf, kexp, css, es, dcontext);
		ExplainPropertyText(label, buf.data, es);
		pfree(buf.data);
	}
}

char *
pgstrom_xpucode_to_string(bytea *xpu_code)
{
	StringInfoData buf;

	initStringInfo(&buf);
	__xpucode_to_cstring(&buf, (const kern_expression *)VARDATA(xpu_code),
						 NULL, NULL, NIL);

	return buf.data;
}

static void
pgstrom_devcache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	MemoryContextReset(devinfo_memcxt);
	memset(devtype_info_slot, 0, sizeof(List *) * DEVTYPE_INFO_NSLOTS);
	memset(devtype_code_slot, 0, sizeof(List *) * DEVTYPE_INFO_NSLOTS);
	memset(devfunc_info_slot, 0, sizeof(List *) * DEVFUNC_INFO_NSLOTS);
	memset(devfunc_code_slot, 0, sizeof(List *) * DEVFUNC_INFO_NSLOTS);
}

void
pgstrom_init_codegen(void)
{
	devinfo_memcxt = AllocSetContextCreate(CacheMemoryContext,
										   "device type/func info cache",
										   ALLOCSET_DEFAULT_SIZES);
	pgstrom_devcache_invalidator(0, 0, 0);
	CacheRegisterSyscacheCallback(TYPEOID, pgstrom_devcache_invalidator, 0);
	CacheRegisterSyscacheCallback(PROCOID, pgstrom_devcache_invalidator, 0);
}
