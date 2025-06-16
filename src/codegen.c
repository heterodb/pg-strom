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
static volatile int		devinfo_memcxt_generation = 0;
static List	   *devtype_info_slot[DEVTYPE_INFO_NSLOTS];
static List	   *devfunc_info_slot[DEVFUNC_INFO_NSLOTS];
static HTAB	   *devtype_rev_htable = NULL;		/* lookup by TypeOpCode */
static HTAB	   *devfunc_rev_htable = NULL;		/* lookup by FuncOpCode */

/* -------- static declarations -------- */
#define TYPE_OPCODE(NAME,EXTENSION,FLAGS)								\
	static uint32_t devtype_##NAME##_hash(bool isnull, Datum value);
#include "xpu_opcodes.h"

#define TYPE_OPCODE(NAME,EXTENSION,FLAGS)			\
	{ EXTENSION, #NAME,	TypeOpCode__##NAME,			\
	  DEVKIND__ANY | (FLAGS),						\
	  devtype_##NAME##_hash,						\
	  sizeof(xpu_##NAME##_t),						\
	  __alignof__(xpu_##NAME##_t),					\
	  sizeof(kvec_##NAME##_t) },
static struct {
	const char	   *type_extension;
	const char	   *type_name;
	TypeOpCode		type_code;
	uint32_t		type_flags;
	devtype_hashfunc_f type_hashfunc;
	int				type_sizeof;
	int				type_alignof;
	int				kvec_sizeof;
} devtype_catalog[] = {
#include "xpu_opcodes.h"
	/* alias device data types */
	{NULL, NULL, TypeOpCode__Invalid, 0, NULL, 0, 0}
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

static void	__appendKernExpMagicAndLength(StringInfo buf, int head_pos);
static int	codegen_expression_walker(codegen_context *context,
									  StringInfo buf,
									  int curr_depth,
									  Expr *expr);

static const char *
get_extension_name_by_object(Oid class_id, Oid object_id)
{
	Oid		ext_oid = getExtensionOfObject(class_id, object_id);

	if (OidIsValid(ext_oid))
		return get_extension_name(ext_oid);
	return NULL;
}

/*
 * type oid cache
 */
static Oid		__type_oid_cache_int1	= UINT_MAX;
Oid
get_int1_type_oid(bool missing_ok)
{
	if (__type_oid_cache_int1 == UINT_MAX)
	{
		const char *ext_name;
		Oid			type_oid;

		type_oid =  GetSysCacheOid2(TYPENAMENSP,
									Anum_pg_type_oid,
									CStringGetDatum("int1"),
									ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
		if (OidIsValid(type_oid))
		{
			ext_name = get_extension_name_by_object(TypeRelationId, type_oid);
			if (!ext_name || strcmp(ext_name, "pg_strom") != 0)
				type_oid = InvalidOid;
		}
		__type_oid_cache_int1 = type_oid;
	}
	if (!missing_ok && !OidIsValid(__type_oid_cache_int1))
		elog(ERROR, "type 'int1' is not installed");
	return __type_oid_cache_int1;
}

static Oid		__type_oid_cache_float2	= UINT_MAX;
Oid
get_float2_type_oid(bool missing_ok)
{
	if (__type_oid_cache_float2 == UINT_MAX)
	{
		const char *ext_name;
		Oid			type_oid;

		type_oid = GetSysCacheOid2(TYPENAMENSP,
								   Anum_pg_type_oid,
								   CStringGetDatum("float2"),
								   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
		if (OidIsValid(type_oid))
		{
			ext_name = get_extension_name_by_object(TypeRelationId, type_oid);
			if (!ext_name || strcmp(ext_name, "pg_strom") != 0)
				type_oid = InvalidOid;
		}
		__type_oid_cache_float2 = type_oid;
	}
	if (!missing_ok && !OidIsValid(__type_oid_cache_float2))
		elog(ERROR, "type 'float2' is not installed");
	return __type_oid_cache_float2;
}

static Oid		__type_oid_cache_cube	= UINT_MAX;
Oid
get_cube_type_oid(bool missing_ok)
{
	if (__type_oid_cache_cube == UINT_MAX)
	{
		Oid			type_oid = InvalidOid;
		CatCList   *typelist;

		typelist = SearchSysCacheList1(TYPENAMENSP,
									   CStringGetDatum("cube"));
		for (int i=0; i < typelist->n_members; i++)
		{
			HeapTuple	type_htup = &typelist->members[i]->tuple;
			Form_pg_type type_form = (Form_pg_type)GETSTRUCT(type_htup);
			const char *ext_name;

			ext_name = get_extension_name_by_object(TypeRelationId,
													type_form->oid);
			if (ext_name && strcmp(ext_name, "cube") == 0)
			{
				type_oid = type_form->oid;
				break;
			}
		}
		ReleaseCatCacheList(typelist);

		__type_oid_cache_cube = type_oid;
	}
	if (!missing_ok && !OidIsValid(__type_oid_cache_cube))
		elog(ERROR, "type 'cube' is not installed");
	return __type_oid_cache_cube;
}

/*
 * build_basic_devtype_info
 */
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
			dtype->kvec_sizeof = devtype_catalog[i].kvec_sizeof;
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

/*
 * build_composite_devtype_info
 */
static devtype_info *
build_composite_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	TupleDesc		tupdesc = lookup_rowtype_tupdesc(tcache->type_id, -1);
	devtype_info  **subtypes = alloca(sizeof(devtype_info *) * tupdesc->natts);
	devtype_info   *dtype;
	MemoryContext	oldcxt;
	uint32_t		subtype_flags = DEVKIND__ANY;

	for (int j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		dtype = pgstrom_devtype_lookup(attr->atttypid);
		if (!dtype)
		{
			ReleaseTupleDesc(tupdesc);
			return NULL;
		}
		subtype_flags &= dtype->type_flags;
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
	dtype->type_flags = subtype_flags;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
	dtype->type_extension = NULL;
	dtype->type_name = "composite";
	dtype->type_namespace = get_type_namespace(tcache->type_id);
	dtype->type_sizeof = sizeof(xpu_composite_t);
	dtype->type_alignof = __alignof__(xpu_composite_t);
	dtype->kvec_sizeof = sizeof(kvec_composite_t);
	dtype->type_hashfunc = NULL; //devtype_composite_hash;
	dtype->type_eqfunc = get_opcode(tcache->eq_opr);
	dtype->type_cmpfunc = tcache->cmp_proc;
	dtype->comp_nfields = tupdesc->natts;
	memcpy(dtype->comp_subtypes, subtypes,
		   sizeof(devtype_info *) * tupdesc->natts);
	MemoryContextSwitchTo(oldcxt);

	return dtype;
}

/*
 * build_array_devtype_info
 */
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
	dtype->type_flags = elem->type_flags;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
	dtype->type_name = "array";
	dtype->type_namespace = get_type_namespace(tcache->type_id);
	dtype->type_sizeof = sizeof(xpu_array_t);
	dtype->type_alignof = __alignof__(xpu_array_t);
	dtype->kvec_sizeof = sizeof(kvec_array_t);
	dtype->type_hashfunc = NULL; //devtype_array_hash;
	dtype->type_element = elem;
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
	int				__generation;

	if (!OidIsValid(type_oid))
		return NULL;	/* InvalidOid should never has device-type */

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
	/*
	 * Not found, try to build devtype_info entry
	 */
	do {
		__generation = devinfo_memcxt_generation;

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
	} while (__generation != devinfo_memcxt_generation);

	devtype_info_slot[index] = lappend_cxt(devinfo_memcxt,
										   devtype_info_slot[index], dtype);
found:
	return (dtype->type_is_negative ? NULL : dtype);
}

/*
 * devtype_get_name_by_opcode
 */
typedef struct {
	TypeOpCode	type_code;
	const char *type_name;
} devtype_reverse_entry;

static const char *
devtype_get_name_by_opcode(TypeOpCode type_code)
{
	devtype_reverse_entry *entry;
	bool		found;

	if (!devtype_rev_htable)
	{
		HASHCTL	hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(TypeOpCode);
		hctl.entrysize = sizeof(devtype_reverse_entry);
		hctl.hcxt = devinfo_memcxt;

		devtype_rev_htable = hash_create("devtype_rev_htable",
										 128,
										 &hctl,
										 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}

	entry = (devtype_reverse_entry *)
		hash_search(devtype_rev_htable, &type_code, HASH_ENTER, &found);
	if (!found)
	{
		switch (type_code)
		{
			case TypeOpCode__composite:
				entry->type_name = "composite";
				break;
			case TypeOpCode__array:
				entry->type_name = "array";
				break;
			case TypeOpCode__internal:
				entry->type_name = "internal";
				break;
			default:
				entry->type_name = NULL;
				for (int i=0; devtype_catalog[i].type_name != NULL; i++)
				{
					if (devtype_catalog[i].type_code == type_code)
					{
						entry->type_name = devtype_catalog[i].type_name;
						break;
					}
				}
				break;
		}
	}
	if (!entry->type_name)
		elog(ERROR, "device type opcode:%u not found", type_code);
	return entry->type_name;
}

/*
 * devtype_get_kvec_sizeof_by_opcode
 */
static uint32_t
devtype_get_kvec_sizeof_by_opcode(TypeOpCode type_code)
{
	switch (type_code)
	{
		case TypeOpCode__composite:
			return sizeof(kvec_composite_t);
		case TypeOpCode__array:
			return sizeof(kvec_array_t);
		case TypeOpCode__internal:
			return sizeof(kvec_internal_t);
		default:
			for (int i=0; devtype_catalog[i].type_name != NULL; i++)
			{
				if (devtype_catalog[i].type_code == type_code)
					return devtype_catalog[i].kvec_sizeof;
			}
			break;
	}
	elog(ERROR, "device type opcode:%u not found", type_code);
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
	uint32_t	hash;
	uint8_t		kind;
	int16_t		weight;
	int128_t	num;
	const char *emsg;

	if (isnull)
		return 0;
	emsg = __xpu_numeric_from_varlena(&kind,
									  &weight,
									  &num,
									  (varlena *)value);
	if (emsg)
		elog(ERROR, "%s", emsg);
	if (kind == XPU_NUMERIC_KIND__VALID)
		hash = (hash_any((unsigned char *)&weight, sizeof(int16_t)) ^
				hash_any((unsigned char *)&num, sizeof(int128_t)));
	else
		hash = hash_any((unsigned char *)&kind, sizeof(uint8_t));

	return hash;
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

static uint32_t
devtype_cube_hash(bool isnull, Datum value)
{
	if (isnull)
		return 0;
	return hash_any((unsigned char *)VARDATA_ANY(value), VARSIZE_ANY_EXHDR(value));
}

/*
 * Built-in device functions/operators
 */
#define FUNC_OPCODE(SQLNAME,FN_ARGS,FN_FLAGS,DEVNAME,FUNC_COST,EXTENSION) \
	{ #SQLNAME, #FN_ARGS, FN_FLAGS, FuncOpCode__##DEVNAME, FUNC_COST, EXTENSION },
#define FUNC_ALIAS(SQLNAME,FN_ARGS,FN_FLAGS,DEVNAME,FUNC_COST,EXTENSION) \
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
	int				__generation;

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
	do {
		__generation = devinfo_memcxt_generation;

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
	} while (__generation != devinfo_memcxt_generation);

	devfunc_info_slot[i] = lappend_cxt(devinfo_memcxt,
									   devfunc_info_slot[i], dfunc);
found:
	if (dfunc->func_is_negative)
		return NULL;
	if (OidIsValid(func_collid) &&
		(dfunc->func_flags & DEVFUNC__LOCALE_AWARE) != 0)
	{
		/* see texteq, bpchareq */
		if (!lc_collate_is_c(func_collid))
			return NULL;	/* not supported */
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

/*
 * devfunc_lookup_by_opcode - reverse lookup of device function
 */
typedef struct {
	FuncOpCode		opcode;
	bool			is_valid;
	bool			device_only;
	const char	   *dfunc_name;
} devfunc_reverse_entry;

static const char *
devfunc_get_name_by_opcode(FuncOpCode func_code, bool *p_device_only)
{
	devfunc_reverse_entry *entry;
	bool		found;

	if (!devfunc_rev_htable)
	{
		HASHCTL	hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(FuncOpCode);
		hctl.entrysize = sizeof(devfunc_reverse_entry);
		hctl.hcxt = devinfo_memcxt;

		devfunc_rev_htable = hash_create("devfunc_rev_htable",
										 256,
										 &hctl,
										 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}

	entry = (devfunc_reverse_entry *)
		hash_search(devfunc_rev_htable, &func_code, HASH_ENTER, &found);
	if (!found)
	{
		static struct {
			FuncOpCode	func_code;
			const char *func_name;
			const char *func_args;
			const char *rtype_name;
		} devonlyfunc_catalog[] = {
#define DEVONLY_FUNC_OPCODE(RET_TYPE,DEV_NAME,FUNC_ARGS,a,b)			\
			{FuncOpCode__##DEV_NAME, #DEV_NAME, #FUNC_ARGS, #RET_TYPE},
#include "xpu_opcodes.h"
			{FuncOpCode__Invalid,NULL,NULL}
		};

		for (int i=0; devfunc_catalog[i].func_code != FuncOpCode__Invalid; i++)
		{
			if (devfunc_catalog[i].func_code == func_code)
			{
				entry->is_valid    = true;
				entry->device_only = false;
				entry->dfunc_name  = devfunc_catalog[i].func_name;
				goto found;
			}
		}

		for (int i=0; devonlyfunc_catalog[i].func_code != FuncOpCode__Invalid; i++)
		{
			if (devonlyfunc_catalog[i].func_code == func_code)
			{
				entry->is_valid    = true;
				entry->device_only = true;
				entry->dfunc_name  = devonlyfunc_catalog[i].func_name;
				goto found;
			}
		}
		entry->is_valid   = false;
		entry->dfunc_name = NULL;
	}
found:
	if (entry->is_valid)
	{
		if (p_device_only)
			*p_device_only = entry->device_only;
		return entry->dfunc_name;
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

/*
 * __assign_codegen_kvar_defitem_type_params
 */
static bool
__assign_codegen_kvar_defitem_type_params(Oid kv_type_oid,
										  TypeOpCode *p_kv_type_code,
										  bool *p_kv_type_byval,
										  int8_t *p_kv_type_align,
										  int16_t *p_kv_type_length,
										  int32_t *p_kv_xdatum_sizeof,	/* optional */
										  int32_t *p_kv_kvec_sizeof,	/* optional */
										  bool allows_host_only_types)
{
	devtype_info   *dtype = pgstrom_devtype_lookup(kv_type_oid);

	if (dtype)
	{
		*p_kv_type_code = dtype->type_code;
		*p_kv_type_byval = dtype->type_byval;
		*p_kv_type_align = dtype->type_align;
		*p_kv_type_length = dtype->type_length;
		if (p_kv_xdatum_sizeof)
			*p_kv_xdatum_sizeof = dtype->type_sizeof;
		if (p_kv_kvec_sizeof)
			*p_kv_kvec_sizeof = dtype->kvec_sizeof;

		return true;
	}

	if (allows_host_only_types)
	{
		/* assign generic type if no device type defined */
		TypeOpCode	type_code;
		bool		typbyval;
		char		typalign;
		int16_t		typlen;
		int32_t		xdatum_sizeof;
		int32_t		kvec_sizeof;

		get_typlenbyvalalign(kv_type_oid, &typlen, &typbyval, &typalign);
		if (typbyval)
		{
			switch (typlen)
			{
				case 1:
					type_code   = TypeOpCode__int1;
					xdatum_sizeof = sizeof(xpu_int1_t);
					kvec_sizeof = sizeof(kvec_int1_t);
					break;
				case 2:
					type_code   = TypeOpCode__int2;
					xdatum_sizeof = sizeof(xpu_int2_t);
					kvec_sizeof = sizeof(kvec_int2_t);
					break;
                case 4:
					type_code   = TypeOpCode__int4;
					xdatum_sizeof = sizeof(xpu_int4_t);
					kvec_sizeof = sizeof(kvec_int4_t);
                    break;
                case 8:
					type_code   = TypeOpCode__int8;
					xdatum_sizeof = sizeof(xpu_int8_t);
					kvec_sizeof = sizeof(kvec_int8_t);
					break;
				default:
					elog(ERROR, "unexpected inline type length: %d", (int)typlen);
					break;
			}
		}
		else if (typlen > 0)
		{
			type_code   = TypeOpCode__internal;
			xdatum_sizeof = sizeof(xpu_internal_t);
			kvec_sizeof = sizeof(kvec_internal_t);
		}
		else if (typlen == -1)
		{
			type_code   = TypeOpCode__bytea;
			xdatum_sizeof = sizeof(xpu_bytea_t);
			kvec_sizeof = sizeof(kvec_bytea_t);
		}
		else
		{
			elog(ERROR, "unknown type length: %d", (int)typlen);
		}
		*p_kv_type_code = type_code;
		*p_kv_type_byval = typbyval;
		*p_kv_type_align = typealign_get_width(typalign);
		*p_kv_type_length = typlen;
		if (p_kv_xdatum_sizeof)
			*p_kv_xdatum_sizeof = xdatum_sizeof;
		if (p_kv_kvec_sizeof)
			*p_kv_kvec_sizeof = kvec_sizeof;

		return true;
	}
	/* not supported device type */
	return false;
}

/*
 * __assign_codegen_kvar_defitem_subfields
 */
static void
__assign_codegen_kvar_defitem_subfields(codegen_kvar_defitem *kvdef)
{
	codegen_kvar_defitem *__kvdef;
	TypeCacheEntry *tcache;

	tcache = lookup_type_cache(kvdef->kv_type_oid, TYPECACHE_TUPDESC);
	if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		/* array type element */
		__kvdef = palloc0(sizeof(codegen_kvar_defitem));
		__kvdef->kv_slot_id = -1;
		__kvdef->kv_depth = kvdef->kv_depth;
		__kvdef->kv_resno = kvdef->kv_resno;
		__kvdef->kv_maxref = kvdef->kv_maxref;
		__kvdef->kv_offset = -1;
		__kvdef->kv_type_oid = tcache->typelem;
		__assign_codegen_kvar_defitem_type_params(tcache->typelem,
												  &__kvdef->kv_type_code,
												  &__kvdef->kv_typbyval,
												  &__kvdef->kv_typalign,
												  &__kvdef->kv_typlen,
												  &__kvdef->kv_xdatum_sizeof,
												  &__kvdef->kv_kvec_sizeof,
												  true);
		__kvdef->kv_expr = (Expr *)makeNullConst(tcache->typelem, -1, InvalidOid);
		__assign_codegen_kvar_defitem_subfields(__kvdef);
		kvdef->kv_subfields = list_make1(__kvdef);
	}
	else if (tcache->tupDesc)
	{
		/* composite type element */
		TupleDesc	tdesc = tcache->tupDesc;

		for (int j=0; j < tdesc->natts; j++)
		{
			Form_pg_attribute attr = TupleDescAttr(tdesc, j);

			__kvdef = palloc0(sizeof(codegen_kvar_defitem));
			__kvdef->kv_slot_id = -1;
			__kvdef->kv_depth = kvdef->kv_depth;
			__kvdef->kv_resno = kvdef->kv_resno;
			__kvdef->kv_maxref = kvdef->kv_maxref;
			__kvdef->kv_offset = -1;
			__kvdef->kv_type_oid = attr->atttypid;
			__assign_codegen_kvar_defitem_type_params(attr->atttypid,
													  &__kvdef->kv_type_code,
													  &__kvdef->kv_typbyval,
													  &__kvdef->kv_typalign,
													  &__kvdef->kv_typlen,
													  &__kvdef->kv_xdatum_sizeof,
													  &__kvdef->kv_kvec_sizeof,
													  true);
			__kvdef->kv_expr = (Expr *)makeNullConst(attr->atttypid, -1, InvalidOid);
			__assign_codegen_kvar_defitem_subfields(__kvdef);
			kvdef->kv_subfields = lappend(kvdef->kv_subfields, __kvdef);
		}
	}
}

/*
 * equalVar - compares two Var nodes except for varnullingrels
 *
 * NOTE: Var-nodes in the reltarget of input-paths are not normalized
 * to this level of GpuJoin, so it may have different varnullingrels
 * even if they are identical Var-nodes. So, we should not use equal()
 * here to compare Var-nodes.
 */
static inline bool
equalVar(const void *__a, const void *__b)
{
	if (IsA(__a, Var) && IsA(__b, Var))
	{
		const Var  *a = __a;
		const Var  *b = __b;

		if (a->varno    == b->varno &&
			a->varattno == b->varattno)
		{
			Assert(a->vartype   == b->vartype &&
				   a->vartypmod == b->vartypmod &&
				   a->varcollid == b->varcollid);
			return true;
		}
	}
	return false;
}

/*
 * lookup_input_varnode_defitem
 */
static codegen_kvar_defitem *
lookup_input_varnode_defitem(codegen_context *context,
							 Var *var,
							 int curr_depth,
							 bool allows_host_only_types)
{
	codegen_kvar_defitem *kvdef;
	int			depth, resno;
	ListCell   *lc;
	Oid			kv_type_oid;
	TypeOpCode	kv_type_code;
	bool		kv_type_byval;
	int8_t		kv_type_align;
	int16_t		kv_type_length;
	int32_t		kv_xdatum_sizeof;
	int32_t		kv_kvec_sizeof;

	if (!IsA(var, Var))
		return NULL;

	if (var->varno == context->scan_relid)
	{
		depth = 0;
		resno = var->varattno;
		Assert(resno != InvalidAttrNumber);
		goto found;
	}

	for (depth = 1; depth <= context->num_rels; depth++)
	{
		PathTarget *target = context->pd[depth].inner_target;

		resno = 1;
		foreach (lc, target->exprs)
		{
			if (equalVar(var, lfirst(lc)))
				goto found;
			resno++;
		}
	}
	return NULL;	/* not found */
found:
	foreach (lc, context->kvars_deflist)
	{
		kvdef = lfirst(lc);

		if (kvdef->kv_depth == depth &&
			kvdef->kv_resno == resno)
		{
			Assert(equalVar(var, kvdef->kv_expr));
			kvdef->kv_maxref = Max(kvdef->kv_maxref, curr_depth);
			return kvdef;
		}
	}

	/* attach new one */
	kv_type_oid = exprType((Node *)var);
	if (!__assign_codegen_kvar_defitem_type_params(kv_type_oid,
												   &kv_type_code,
												   &kv_type_byval,
												   &kv_type_align,
												   &kv_type_length,
												   &kv_xdatum_sizeof,
												   &kv_kvec_sizeof,
												   allows_host_only_types))
	{
		return NULL;
	}
	kvdef = palloc0(sizeof(codegen_kvar_defitem));
	kvdef->kv_slot_id     = list_length(context->kvars_deflist);
	kvdef->kv_depth       = depth;
	kvdef->kv_resno       = resno;
	kvdef->kv_maxref      = curr_depth;
	kvdef->kv_offset      = context->kvecs_usage;
	kvdef->kv_type_oid    = kv_type_oid;
	kvdef->kv_type_code   = kv_type_code;
	kvdef->kv_typbyval    = kv_type_byval;
	kvdef->kv_typalign    = kv_type_align;
	kvdef->kv_typlen      = kv_type_length;
	kvdef->kv_xdatum_sizeof = kv_xdatum_sizeof;
	kvdef->kv_kvec_sizeof = kv_kvec_sizeof;
	kvdef->kv_expr        = (Expr *)var;
	__assign_codegen_kvar_defitem_subfields(kvdef);
	context->kvecs_usage += KVEC_ALIGN(kvdef->kv_kvec_sizeof);
	context->kvars_deflist = lappend(context->kvars_deflist, kvdef);

	return kvdef;
}

/*
 * __try_inject_temporary_expression
 */
static codegen_kvar_defitem *
__try_inject_temporary_expression(codegen_context *context,
								  StringInfo buf,
								  Expr *expr,
								  int curr_depth,
								  bool allows_host_only_types)
{
	codegen_kvar_defitem *kvdef;
	kern_expression kexp;
	ListCell   *lc;
	int			pos = -1;

	/*
	 * When 'expr' is simple Var-reference on the input relations,
	 * we don't need to inject expression node here.
	 */
	kvdef = lookup_input_varnode_defitem(context, (Var *)expr,
										 curr_depth,
										 allows_host_only_types);
	if (kvdef)
		goto found;

	/*
	 * Try to find out the expression which already injected to the
	 * last level kvec buffer. If exists, we can reuse it.
	 */
	foreach (lc, context->kvars_deflist)
	{
		kvdef = lfirst(lc);

		if (equal(expr, kvdef->kv_expr))
			goto found;
	}

	/*
	 * Try to allocate a new kvec-buffer, if an expression (not a simple
	 * var-reference) is required.
	 */
	kvdef = palloc0(sizeof(codegen_kvar_defitem));
	kvdef->kv_slot_id   = list_length(context->kvars_deflist);
	kvdef->kv_depth     = curr_depth;
	kvdef->kv_resno     = InvalidAttrNumber; /* no source */
	kvdef->kv_maxref    = curr_depth;
	kvdef->kv_offset    = -1;				/* never use vectorized buffer */
	kvdef->kv_type_oid  = exprType((Node *)expr);
	if (!__assign_codegen_kvar_defitem_type_params(kvdef->kv_type_oid,
												   &kvdef->kv_type_code,
												   &kvdef->kv_typbyval,
												   &kvdef->kv_typalign,
												   &kvdef->kv_typlen,
												   &kvdef->kv_xdatum_sizeof,
												   NULL,	/* no kvec-buffer */
												   false))
	{
		elog(ERROR, "type %s is not device supported",
			 format_type_be(kvdef->kv_type_oid));
	}
	kvdef->kv_expr      = expr;
	__assign_codegen_kvar_defitem_subfields(kvdef);
	context->kvars_deflist = lappend(context->kvars_deflist, kvdef);

	/*
	 * Setup SaveExpr expression
	 */
found:
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype  = kvdef->kv_type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode   = FuncOpCode__SaveExpr;
	kexp.nr_args  = 1;
	kexp.args_offset = MAXALIGN(offsetof(kern_expression,
										 u.save.data));
	kexp.u.save.sv_slot_id   = kvdef->kv_slot_id;
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, kexp.args_offset);
	if (codegen_expression_walker(context, buf, curr_depth, expr) < 0)
		return NULL;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return kvdef;
}

/* ----------------------------------------------------------------
 *
 * xPU pseudo code generator
 *
 * ----------------------------------------------------------------
 */
#define XPUCODE_STACK_USAGE_NORMAL		128		/* stack usage by normal function calls */
#define XPUCODE_STACK_USAGE_RECURSIVE	2048	/* stack usage by recursive function calls */

#define __Ereport(fmt,...)													\
	do {																\
		ereport(context->elevel,										\
				(errcode(ERRCODE_INTERNAL_ERROR),						\
				 errmsg("(%s:%d) " fmt,	__FUNCTION__, __LINE__,			\
						##__VA_ARGS__),									\
				 errdetail("problematic expression: %s",				\
						   nodeToString(context->top_expr))));			\
		return -1;														\
	} while(0)

codegen_context *
create_codegen_context(PlannerInfo *root,
					   CustomPath *cpath,
					   pgstromPlanInfo *pp_info)
{
	codegen_context *context;
	ListCell   *lc;
	int			depth = 1;

	/* allocation with possible max length */
	context = palloc0(offsetof(codegen_context, pd[pp_info->num_rels +
												   pp_info->num_rels + 2]));
	context->elevel = ERROR;
	context->root = root;
	context->xpu_task_flags = pp_info->xpu_task_flags;
	context->kvecs_ndims = pp_info->num_rels + 1;
	context->kvecs_usage = 0;
	context->scan_relid = pp_info->scan_relid;
	context->num_rels = pp_info->num_rels;
	Assert(pp_info->num_rels == list_length(cpath->custom_paths));
	foreach (lc, cpath->custom_paths)
	{
		Path   *ipath = lfirst(lc);

		context->pd[depth++].inner_target = ipath->pathtarget;
	}
	return context;
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
						 StringInfo buf, int curr_depth,
						 Const *con)
{
	devtype_info   *dtype;
	char			typtype;

	typtype = get_typtype(con->consttype);
	if (typtype != TYPTYPE_BASE &&
		typtype != TYPTYPE_ENUM &&
		typtype != TYPTYPE_RANGE &&
		typtype != TYPTYPE_DOMAIN)
		__Ereport("unable to use type %s in Const expression (class: %c)",
				  format_type_be(con->consttype), typtype);

	dtype = pgstrom_devtype_lookup(con->consttype);
	if (!dtype)
		__Ereport("type %s is not device supported",
				  format_type_be(con->consttype));
	if (buf)
	{
		kern_expression kexp;
		int		pos;

		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype = dtype->type_code;
        kexp.expflags = context->kexp_flags;
        kexp.opcode = FuncOpCode__ConstExpr;
        kexp.u.c.const_type = con->consttype;
        kexp.u.c.const_isnull = con->constisnull;
		pos = __appendBinaryStringInfo(buf, &kexp, offsetof(kern_expression,
															u.c.const_value));
		if (!con->constisnull)
		{
			if (con->constbyval)
				appendBinaryStringInfo(buf,
									   (char *)&con->constvalue,
									   con->constlen);
			else if (con->constlen > 0)
				appendBinaryStringInfo(buf,
									   DatumGetPointer(con->constvalue),
									   con->constlen);
			else if (con->constlen == -1)
				appendBinaryStringInfo(buf,
									   DatumGetPointer(con->constvalue),
									   VARSIZE_ANY(con->constvalue));
			else
				elog(ERROR, "unsupported type length: %d", con->constlen);
		}
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

static int
codegen_param_expression(codegen_context *context,
						 StringInfo buf, int curr_depth,
						 Param *param)
{
	devtype_info   *dtype;
	char			typtype;

	if (param->paramkind != PARAM_EXTERN)
		__Ereport("Only PARAM_EXTERN is supported on device: %d",
				  (int)param->paramkind);

	typtype = get_typtype(param->paramtype);
	if (typtype != TYPTYPE_BASE &&
		typtype != TYPTYPE_ENUM &&
		typtype != TYPTYPE_RANGE &&
		typtype != TYPTYPE_DOMAIN)
		__Ereport("unable to use type %s in Param expression (class: %c)",
				  format_type_be(param->paramtype), typtype);

	dtype = pgstrom_devtype_lookup(param->paramtype);
	if (!dtype)
		__Ereport("type %s is not device supported",
				  format_type_be(param->paramtype));
	if (buf)
	{
		kern_expression	kexp;
		int		pos;

		memset(&kexp, 0, sizeof(kexp));
		kexp.opcode = FuncOpCode__ParamExpr;
		kexp.exptype = dtype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.u.p.param_id = param->paramid;
		pos = __appendBinaryStringInfo(buf, &kexp, offsetof(kern_expression,
															u.p.__data));
		__appendKernExpMagicAndLength(buf, pos);
	}
	context->used_params = list_append_unique(context->used_params, param);

	return 0;
}

static int
codegen_var_expression(codegen_context *context,
					   StringInfo buf,
					   int curr_depth,
					   Var *var)
{
	codegen_kvar_defitem *kvdef;

	kvdef = lookup_input_varnode_defitem(context,
										 var,
										 curr_depth,
										 false);
	if (buf)
	{
		kern_expression kexp;
		int			pos;

		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype         = kvdef->kv_type_code;
		kexp.expflags        = context->kexp_flags;
		kexp.opcode          = FuncOpCode__VarExpr;
		kexp.u.v.var_slot_id = kvdef->kv_slot_id;
		/*
		 * NOTE: GPU uses kvec-buffer to save intermediation results
		 * per depth, because different core shall handle this result
		 * on the next depth, and private values are not visible to
		 * others.
		 */
		if ((context->xpu_task_flags & DEVKIND__NVIDIA_GPU) != 0 &&
			kvdef->kv_offset >= 0 &&
			kvdef->kv_depth != curr_depth)
			kexp.u.v.var_offset = kvdef->kv_offset;
		else
			kexp.u.v.var_offset = -1;
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

static int
__codegen_func_expression(codegen_context *context,
						  StringInfo buf,
						  int curr_depth,
						  Oid func_oid,
						  List *func_args,
						  Oid func_collid)
{
	devfunc_info   *dfunc;
	devtype_info   *dtype;
	kern_expression	kexp;
	int				pos = -1;
	uint32_t		stack_usage_saved = context->stack_usage;
	uint32_t		stack_usage_max = stack_usage_saved;
	ListCell	   *lc;

	dfunc = pgstrom_devfunc_lookup(func_oid, func_args, func_collid);
	if (!dfunc ||
		(dfunc->func_flags & context->xpu_task_flags & DEVKIND__ANY) == 0)
		__Ereport("function %s is not supported on the target device",
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

		if (codegen_expression_walker(context, buf, curr_depth, arg) < 0)
			return -1;
		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;
	if ((dfunc->func_flags & DEVFUNC__HAS_RECURSION) != 0)
		context->stack_usage += XPUCODE_STACK_USAGE_RECURSIVE;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

static int
codegen_func_expression(codegen_context *context,
						StringInfo buf, int curr_depth,
						FuncExpr *func)
{
	return __codegen_func_expression(context,
									 buf,
									 curr_depth,
									 func->funcid,
									 func->args,
									 func->inputcollid);
}

static int
codegen_oper_expression(codegen_context *context,
						StringInfo buf, int curr_depth,
						OpExpr *oper)
{
	return __codegen_func_expression(context,
									 buf,
									 curr_depth,
									 get_opcode(oper->opno),
									 oper->args,
									 oper->inputcollid);
}

static int
codegen_distinct_expression(codegen_context *context,
							StringInfo buf, int curr_depth,
							DistinctExpr *expr)
{
	kern_expression	kexp;
	int		pos = -1;

	Assert(exprType((Node *)expr) == BOOLOID);
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = FuncOpCode__DistinctFrom;
	kexp.nr_args = 1;
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	if (codegen_oper_expression(context, buf, curr_depth, expr) < 0)
		return -1;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

static int
codegen_bool_expression(codegen_context *context,
						StringInfo buf, int curr_depth,
						BoolExpr *b)
{
	kern_expression	kexp;
	uint32_t	stack_usage_saved = context->stack_usage;
	uint32_t	stack_usage_max = stack_usage_saved;
	int			pos = -1;
	ListCell   *lc;

	memset(&kexp, 0, sizeof(kexp));
	switch (b->boolop)
	{
		case AND_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_And;
			kexp.nr_args = list_length(b->args);
			if (kexp.nr_args < 2)
				__Ereport("BoolExpr(AND) must have 2 or more arguments");
			break;
		case OR_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_Or;
			kexp.nr_args = list_length(b->args);
			if (kexp.nr_args < 2)
				__Ereport("BoolExpr(OR) must have 2 or more arguments");
			break;
		case NOT_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_Not;
			kexp.nr_args = list_length(b->args);
			if (kexp.nr_args != 1)
				__Ereport("BoolExpr(OR) must not have multiple arguments");
			break;
		default:
			__Ereport("BoolExpr has unknown bool operation (%d)", (int)b->boolop);
	}
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	foreach (lc, b->args)
	{
		Expr   *arg = lfirst(lc);

		if (codegen_expression_walker(context, buf, curr_depth, arg) < 0)
			return -1;
		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

static int
codegen_nulltest_expression(codegen_context *context,
							StringInfo buf, int curr_depth,
							NullTest *nt)
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
			__Ereport("NullTest has unknown NullTestType (%d)", (int)nt->nulltesttype);
	}
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.nr_args = 1;
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, buf, curr_depth, nt->arg) < 0)
		return -1;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

static int
codegen_booleantest_expression(codegen_context *context,
							   StringInfo buf, int curr_depth,
							   BooleanTest *bt)
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
			__Ereport("BooleanTest has unknown BoolTestType (%d)",
					  (int)bt->booltesttype);
	}
	kexp.exptype = TypeOpCode__bool;
	kexp.expflags = context->kexp_flags;
	kexp.nr_args = 1;
	kexp.args_offset = SizeOfKernExpr(0);
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, buf, curr_depth, bt->arg) < 0)
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
__codegen_coerceviaio_jsonbref_expression(codegen_context *context,
										  StringInfo buf, int curr_depth,
										  CoerceViaIO *cvio)
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
	else
		return -1;

	switch (func_oid)
	{
		case F_JSONB_OBJECT_FIELD_TEXT:
			if (list_length(func_args) == 2 &&
				exprType(linitial(func_args)) == JSONBOID &&
				exprType(lsecond(func_args)) == TEXTOID)
				break;
			return -1;
		case F_JSONB_ARRAY_ELEMENT_TEXT:
			if (list_length(func_args) == 2 &&
				exprType(linitial(func_args)) == JSONBOID &&
				exprType(lsecond(func_args)) == INT4OID)
				break;
			return -1;
		default:
			return -1;
	}

	dtype = pgstrom_devtype_lookup(cvio->resulttype);
	if (!dtype)
		return -1;

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
			kern_expression kexp;
			uint32_t	stack_usage_saved = context->stack_usage;
			uint32_t	stack_usage_max = stack_usage_saved;
			ListCell   *lc;
			int			pos = -1;

			memset(&kexp, 0, sizeof(kexp));
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

				if (codegen_expression_walker(context, buf, curr_depth, arg) < 0)
				{
					if (buf)
						buf->len = pos;
					return -1;
				}
				stack_usage_max = Max(stack_usage_max, context->stack_usage);
				context->stack_usage = stack_usage_saved;
			}
			context->stack_usage = stack_usage_max;
			if (buf)
				__appendKernExpMagicAndLength(buf, pos);
			return 0;
		}
	}
	return -1;
}

static int
codegen_coerceviaio_expression(codegen_context *context,
							   StringInfo buf, int curr_depth,
							   CoerceViaIO *cvio)
{
	static struct
	{
		FuncOpCode	opcode;
		const char *source_name;
		const char *source_extension;
		const char *dest_name;
		const char *dest_extension;
	}	coerce_viaio_catalog[] = {
		{FuncOpCode__devcast_text_to_int1, "text", NULL, "int1", "pg_strom"},
		{FuncOpCode__devcast_text_to_int2, "text", NULL, "int2", NULL},
		{FuncOpCode__devcast_text_to_int4, "text", NULL, "int4", NULL},
		{FuncOpCode__devcast_text_to_int8, "text", NULL, "int8", NULL},
		{FuncOpCode__devcast_text_to_float2, "text", NULL, "float2", "pg_strom"},
		{FuncOpCode__devcast_text_to_float4, "text", NULL, "float4", NULL},
		{FuncOpCode__devcast_text_to_float8, "text", NULL, "float8", NULL},
		{FuncOpCode__devcast_text_to_numeric, "text", NULL, "numeric", NULL},
		{FuncOpCode__Invalid, NULL, NULL, NULL, NULL},
	};
	devtype_info   *stype;
	devtype_info   *dtype;

	if (__codegen_coerceviaio_jsonbref_expression(context, buf,
												  curr_depth,
												  cvio) == 0)
		return 0;		/* OK, cvio is JSONB field reference */

	dtype = pgstrom_devtype_lookup(cvio->resulttype);
	if (!dtype)
		__Ereport("Not a supported CoerceViaIO: %s", nodeToString(cvio));
	stype = pgstrom_devtype_lookup(exprType((Node *)cvio->arg));
	if (!stype)
		__Ereport("Not a supported CoerceViaIO: %s", nodeToString(cvio));

	for (int i=0; coerce_viaio_catalog[i].opcode != FuncOpCode__Invalid; i++)
	{
		if (strcmp(coerce_viaio_catalog[i].source_name, stype->type_name) == 0 &&
			((coerce_viaio_catalog[i].source_extension != NULL &&
			  stype->type_extension != NULL &&
			  strcmp(coerce_viaio_catalog[i].source_extension,
					 stype->type_extension) == 0) ||
			 (!coerce_viaio_catalog[i].source_extension && !stype->type_extension)) &&
			strcmp(coerce_viaio_catalog[i].dest_name, dtype->type_name) == 0 &&
			((coerce_viaio_catalog[i].dest_extension != NULL &&
			  dtype->type_extension != NULL &&
			  strcmp(coerce_viaio_catalog[i].dest_extension,
					 dtype->type_extension) == 0) ||
			 (!coerce_viaio_catalog[i].dest_extension && !dtype->type_extension)))
		{
			kern_expression kexp;
			int		pos = -1;

			memset(&kexp, 0, sizeof(kexp));
			kexp.opcode = coerce_viaio_catalog[i].opcode;
			kexp.exptype = dtype->type_code;
			kexp.expflags = context->kexp_flags;
			kexp.nr_args = 1;
			kexp.args_offset = SizeOfKernExpr(0);
			if (buf)
				pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
			if (codegen_expression_walker(context, buf, curr_depth, cvio->arg) < 0)
				return -1;
			if (buf)
				__appendKernExpMagicAndLength(buf, pos);
			return 0;
		}
	}
	__Ereport("Not a supported CoerceViaIO: %s", nodeToString(cvio));
}

/*
 * codegen_coalesce_expression
 */
static int
codegen_coalesce_expression(codegen_context *context,
							StringInfo buf, int curr_depth,
							CoalesceExpr *cl)
{
	devtype_info   *dtype, *__dtype;
	kern_expression	kexp;
	int				pos = -1;
	uint32_t		stack_usage_saved = context->stack_usage;
	uint32_t		stack_usage_max = stack_usage_saved;
	ListCell	   *lc;

	dtype = pgstrom_devtype_lookup(cl->coalescetype);
	if (!dtype)
		__Ereport("Coalesce with type '%s' is not supported",
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
			__Ereport("Coalesce argument has incompatible type: %s",
					  nodeToString(cl));
		if (codegen_expression_walker(context, buf, curr_depth, expr) < 0)
			return -1;
		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

/*
 * codegen_minmax_expression
 */
static int
codegen_minmax_expression(codegen_context *context,
						  StringInfo buf, int curr_depth,
						  MinMaxExpr *mm)
{
	devtype_info   *dtype, *__dtype;
	kern_expression	kexp;
	int				pos = -1;
	uint32_t		stack_usage_saved = context->stack_usage;
	uint32_t		stack_usage_max = stack_usage_saved;
	ListCell	   *lc;

	dtype = pgstrom_devtype_lookup(mm->minmaxtype);
	if (!dtype || (dtype->type_flags & DEVTYPE__HAS_COMPARE) == 0)
		__Ereport("Least/Greatest with type '%s' is not supported",
				  format_type_be(mm->minmaxtype));

	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = dtype->type_code;
	kexp.expflags = context->kexp_flags;
	if (mm->op == IS_GREATEST)
		kexp.opcode = FuncOpCode__GreatestExpr;
	else if (mm->op == IS_LEAST)
		kexp.opcode = FuncOpCode__LeastExpr;
	else
		__Ereport("unknown MinMaxExpr operator: %s", nodeToString(mm));
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
			__Ereport("Least/Greatest argument has incompatible type: %s",
					  nodeToString(mm));
		if (codegen_expression_walker(context, buf, curr_depth, expr) < 0)
			return -1;
		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;
	if (buf)
		__appendKernExpMagicAndLength(buf, pos);
	return 0;
}

/*
 * codegen_relabel_expression
 */
static int
codegen_relabel_expression(codegen_context *context,
						   StringInfo buf, int curr_depth,
						   RelabelType *relabel)
{
	devtype_info *dtype;
	Oid			type_oid;
	TypeOpCode	type_code;

	dtype = pgstrom_devtype_lookup(relabel->resulttype);
	if (!dtype)
		__Ereport("device type '%s' is not supported",
				  format_type_be(relabel->resulttype));
	type_code = dtype->type_code;

	type_oid = exprType((Node *)relabel->arg);
	dtype = pgstrom_devtype_lookup(type_oid);
	if (!dtype)
		__Ereport("device type '%s' is not supported",
				  format_type_be(type_oid));
	if (dtype->type_code != type_code)
		__Ereport("device type '%s' -> '%s' is not binary convertible",
				  format_type_be(type_oid),
				  format_type_be(relabel->resulttype));

	return codegen_expression_walker(context, buf, curr_depth, relabel->arg);
}

/*
 * codegen_casetest_expression
 */
static int		codegen_casetest_key_slot_id = -1;

static int
codegen_casetest_expression(codegen_context *context,
							StringInfo buf, int curr_depth,
							CaseTestExpr *casetest)
{
	codegen_kvar_defitem   *kvdef;

	if (codegen_casetest_key_slot_id < 0 ||
		codegen_casetest_key_slot_id >= list_length(context->kvars_deflist))
		__Ereport("Bug? CaseTestExpr is used out of CaseWhen");
	kvdef = list_nth(context->kvars_deflist, codegen_casetest_key_slot_id);

	if (buf)
	{
		kern_expression	kexp;
		int			pos;

		memset(&kexp, 0, sizeof(kern_expression));
		kexp.exptype = kvdef->kv_type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode   = FuncOpCode__VarExpr;
		kexp.u.v.var_slot_id = kvdef->kv_slot_id;
		if (kvdef->kv_offset >= 0 &&
			kvdef->kv_depth != curr_depth)
			kexp.u.v.var_offset = kvdef->kv_offset;
		else
			kexp.u.v.var_offset = -1;
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

/*
 * codegen_casewhen_expression
 */
static int
codegen_casewhen_expression(codegen_context *context,
							StringInfo buf, int curr_depth,
							CaseExpr *caseexpr)
{
	kern_expression	kexp;
	devtype_info *dtype;
	ListCell   *lc;
	uint32_t	stack_usage_saved = context->stack_usage;
	uint32_t	stack_usage_max = stack_usage_saved;
	int			pos = -1;
	int			saved_casetest_key_slot_id = codegen_casetest_key_slot_id;

	/* check result type */
	dtype = pgstrom_devtype_lookup(caseexpr->casetype);
	if (!dtype)
		__Ereport("device type '%s' is not supported",
				  format_type_be(caseexpr->casetype));
	/* setup kexp */
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = dtype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = FuncOpCode__CaseWhenExpr;
	kexp.nr_args = 2 * list_length(caseexpr->args);
	kexp.args_offset = offsetof(kern_expression, u.casewhen.data);
	if (buf)
		pos = __appendZeroStringInfo(buf, kexp.args_offset);

	/* CASE expr WHEN, if any */
	if (caseexpr->arg)
	{
		codegen_kvar_defitem *kvdef;

		if (buf)
			kexp.u.casewhen.case_comp = (__appendZeroStringInfo(buf, 0) - pos);

		kvdef = __try_inject_temporary_expression(context,
												  buf,
												  caseexpr->arg,
												  curr_depth,
												  false);
		if (!kvdef)
			return -1;
		/* enforce MAXALIGN */
		if (buf)
			kexp.args_offset = (__appendZeroStringInfo(buf, 0) - pos);

		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
		codegen_casetest_key_slot_id = kvdef->kv_slot_id;
	}

	PG_TRY();
	{
		/* WHEN ... THEN ... */
		foreach (lc, caseexpr->args)
		{
			CaseWhen   *casewhen = (CaseWhen *)lfirst(lc);

			/* write out CaseWhen */
			if (codegen_expression_walker(context, buf,
										  curr_depth,
										  casewhen->expr) < 0)
				return -1;
			stack_usage_max = Max(stack_usage_max, context->stack_usage);
			context->stack_usage = stack_usage_saved;

			if (codegen_expression_walker(context, buf,
										  curr_depth,
										  casewhen->result) < 0)
				return -1;
			stack_usage_max = Max(stack_usage_max, context->stack_usage);
			context->stack_usage = stack_usage_saved;
		}
		/* ELSE <expression> */
		if (caseexpr->defresult)
		{
			if (buf)
				kexp.u.casewhen.case_else = (__appendZeroStringInfo(buf, 0) - pos);

			if (codegen_expression_walker(context, buf,
										  curr_depth,
										  caseexpr->defresult) < 0)
				return -1;
			stack_usage_max = Max(stack_usage_max, context->stack_usage);
			context->stack_usage = stack_usage_saved;
		}
		context->stack_usage = stack_usage_max;
	}
	PG_CATCH();
	{
		codegen_casetest_key_slot_id = saved_casetest_key_slot_id;
		PG_RE_THROW();
	}
	PG_END_TRY();
	codegen_casetest_key_slot_id = saved_casetest_key_slot_id;

	if (buf)
	{
		memcpy(buf->data + pos, &kexp,
			   offsetof(kern_expression, u.casewhen.data));
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

/*
 * codegen_scalar_array_op_expression
 */
static int
codegen_scalar_array_op_expression(codegen_context *context,
								   StringInfo buf, int curr_depth,
								   ScalarArrayOpExpr *sa_op)
{
	Expr		   *expr_a, *expr_s;
	devtype_info   *dtype_a, *dtype_s, *dtype_e;
	devfunc_info   *dfunc;
	Oid				type_oid;
	Oid				func_oid;
	Oid				argtypes[2];
	int				pos = -1, __pos = -1;
	codegen_kvar_defitem *kvdef;
	kern_expression	kexp;

	if (list_length(sa_op->args) != 2)
	{
		__Ereport("ScalarArrayOpExpr is not binary operator, not supported");
		return -1;
	}
	expr_a = linitial(sa_op->args);
	type_oid = exprType((Node *)expr_a);
	dtype_a = pgstrom_devtype_lookup(type_oid);
	if (!dtype_a)
		__Ereport("type %s is not device supported", format_type_be(type_oid));

	expr_s = lsecond(sa_op->args);
	type_oid = exprType((Node *)expr_s);
	dtype_s = pgstrom_devtype_lookup(type_oid);
	if (!dtype_s)
		__Ereport("type %s is not device supported", format_type_be(type_oid));

	if (dtype_s->type_element == NULL &&
		dtype_a->type_element != NULL)
	{
		func_oid = get_opcode(sa_op->opno);
	}
	else if (dtype_s->type_element != NULL &&
			 dtype_a->type_element == NULL)
	{
		/* swap arguments */
		Expr		   *expr_temp  = expr_a;
		devtype_info   *dtype_temp = dtype_a;
		Oid				opcode;

		expr_a  = expr_s;
		dtype_a = dtype_s;
		expr_s  = expr_temp;
		dtype_s = dtype_temp;
		opcode = get_commutator(sa_op->opno);
		func_oid = get_opcode(opcode);
	}
	else
	{
		__Ereport("ScalarArrayOpExpr must be 'SCALAR = %s ARRAY' form",
				  sa_op->useOr ? "ANY" : "ALL");
	}
	dtype_e = dtype_a->type_element;
	argtypes[0] = dtype_s->type_oid;
	argtypes[1] = dtype_e->type_oid;
	dfunc = __pgstrom_devfunc_lookup(func_oid,
									 2, argtypes,
									 sa_op->inputcollid);
	if (!dfunc)
		__Ereport("function %s is not device supported",
				  format_procedure(func_oid));
	if (dfunc->func_rettype->type_oid != BOOLOID ||
		dfunc->func_nargs != 2)
		__Ereport("function %s is not a binary boolean function",
				  format_procedure(func_oid));
	/* allocation of kvar-slot for the temporary element variables */
	kvdef = palloc0(sizeof(codegen_kvar_defitem));
	kvdef->kv_slot_id = list_length(context->kvars_deflist);
	kvdef->kv_depth = -1;
	kvdef->kv_resno = InvalidAttrNumber;
	kvdef->kv_maxref = curr_depth;
	kvdef->kv_offset = -1;
	kvdef->kv_type_oid = dtype_e->type_oid;
	kvdef->kv_type_code = dtype_e->type_code;
	kvdef->kv_typbyval = dtype_e->type_byval;
	kvdef->kv_typalign = dtype_e->type_align;
	kvdef->kv_typlen = dtype_e->type_length;
	kvdef->kv_expr = (Expr *)makeNullConst(dtype_e->type_oid, -1, InvalidOid);
	__assign_codegen_kvar_defitem_subfields(kvdef);
	context->kvars_deflist = lappend(context->kvars_deflist, kvdef);

	/* allocation of saop kep */
	memset(&kexp, 0, sizeof(kexp));
	if (buf)
	{
		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype     = TypeOpCode__bool;
		kexp.expflags    = context->kexp_flags;
		kexp.opcode      = (sa_op->useOr
							? FuncOpCode__ScalarArrayOpAny
							: FuncOpCode__ScalarArrayOpAll);
		kexp.nr_args     = 2;
		kexp.args_offset = offsetof(kern_expression, u.saop.data);
		kexp.u.saop.elem_slot_id = kvdef->kv_slot_id;
		pos = __appendBinaryStringInfo(buf, &kexp, kexp.args_offset);
	}
	/* 1st arg - array-expression to be walked on */
	if (codegen_expression_walker(context, buf, curr_depth, expr_a) < 0)
		return -1;
	/* 2nd arg - comparator function */
	if (buf)
	{
		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype = dfunc->func_rettype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode = dfunc->func_code;
		kexp.nr_args = 2;
		kexp.args_offset = offsetof(kern_expression, u.data);
		__pos = __appendBinaryStringInfo(buf, &kexp, kexp.args_offset);
	}
	/* 1st arg (scalar expression) of the comparator function */
	if (codegen_expression_walker(context, buf, curr_depth, expr_s) < 0)
		return -1;
	/* 2nd arg (element reference) of the comparator function */
	if (buf)
	{
		int		elem_pos;

		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype  = dtype_e->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode   = FuncOpCode__VarExpr;
		kexp.nr_args  = 0;
		kexp.u.v.var_slot_id = kvdef->kv_slot_id;
		kexp.u.v.var_offset = -1;
		elem_pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(buf, elem_pos);
	}
	/* terminate the epression */
	if (buf)
	{
		__appendKernExpMagicAndLength(buf, __pos);
		__appendKernExpMagicAndLength(buf, pos);
	}
	return 0;
}

static int
codegen_expression_walker(codegen_context *context,
						  StringInfo buf, int curr_depth,
						  Expr *expr)
{
	if (!expr)
		return 0;

	context->stack_usage += XPUCODE_STACK_USAGE_NORMAL;
	switch (nodeTag(expr))
	{
		case T_Const:
			return codegen_const_expression(context, buf, curr_depth,
											(Const *)expr);
		case T_Param:
			return codegen_param_expression(context, buf, curr_depth,
											(Param *)expr);
		case T_Var:
			return codegen_var_expression(context, buf, curr_depth,
										  (Var *)expr);
		case T_FuncExpr:
			return codegen_func_expression(context, buf, curr_depth,
										   (FuncExpr *)expr);
		case T_OpExpr:
			return codegen_oper_expression(context, buf, curr_depth,
										   (OpExpr *)expr);
		case T_DistinctExpr:
			return codegen_distinct_expression(context, buf, curr_depth,
											   (DistinctExpr *)expr);
		case T_BoolExpr:
			return codegen_bool_expression(context, buf, curr_depth,
										   (BoolExpr *)expr);
		case T_NullTest:
			return codegen_nulltest_expression(context, buf, curr_depth,
											   (NullTest *)expr);
		case T_BooleanTest:
			return codegen_booleantest_expression(context, buf, curr_depth,
												  (BooleanTest *)expr);
		case T_CoerceViaIO:
			return codegen_coerceviaio_expression(context, buf, curr_depth,
												  (CoerceViaIO *)expr);
		case T_CoalesceExpr:
			return codegen_coalesce_expression(context, buf, curr_depth,
											   (CoalesceExpr *)expr);
		case T_MinMaxExpr:
			return codegen_minmax_expression(context, buf, curr_depth,
											 (MinMaxExpr *)expr);
		case T_RelabelType:
			return codegen_relabel_expression(context, buf, curr_depth,
											  (RelabelType *)expr);
		case T_CaseExpr:
			return codegen_casewhen_expression(context, buf, curr_depth,
											   (CaseExpr *)expr);
		case T_CaseTestExpr:
			return codegen_casetest_expression(context, buf, curr_depth,
											   (CaseTestExpr *)expr);
		case T_ScalarArrayOpExpr:
			return codegen_scalar_array_op_expression(context, buf, curr_depth,
													  (ScalarArrayOpExpr *)expr);
		case T_CoerceToDomain:
		default:
			__Ereport("not a supported expression type: %s", nodeToString(expr));
	}
	return -1;
}
#undef __Ereport

/*
 * codegen_build_loadvars
 */
static int
kern_varload_desc_comp(const void *__a, const void *__b)
{
	const kern_varload_desc *a = __a;
	const kern_varload_desc *b = __b;

	if (a->vl_resno < b->vl_resno)
		return -1;
	if (a->vl_resno > b->vl_resno)
		return 1;
	return 0;
}

static kern_expression *
__codegen_build_loadvars_one(codegen_context *context, int depth)
{
	kern_expression *kexp;
	StringInfoData buf;
	int			nrooms = list_length(context->kvars_deflist);
	int			nitems = 0;
	ListCell   *lc;

	kexp = alloca(offsetof(kern_expression, u.load.desc[nrooms+1]));
	memset(kexp, 0, offsetof(kern_expression, u.load.desc));
	foreach (lc, context->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);

		if (kvdef->kv_depth == depth)
		{
			kern_varload_desc  *vl_desc = &kexp->u.load.desc[nitems++];

			memset(vl_desc, 0, sizeof(kern_varload_desc));
			vl_desc->vl_resno     = kvdef->kv_resno;
			vl_desc->vl_slot_id   = kvdef->kv_slot_id;
		}
	}
	if (nitems == 0)
		return NULL;
	kexp->exptype  = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode   = FuncOpCode__LoadVars;
	kexp->args_offset = MAXALIGN(offsetof(kern_expression,
										  u.load.desc[nitems]));
	kexp->u.load.depth = depth;
	kexp->u.load.nitems = nitems;
	qsort(kexp->u.load.desc,
		  nitems,
		  sizeof(kern_varload_desc),
		  kern_varload_desc_comp);
	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *)kexp, kexp->args_offset);
	__appendKernExpMagicAndLength(&buf, 0);

	return (kern_expression *)buf.data;
}

void
codegen_build_packed_kvars_load(codegen_context *context, pgstromPlanInfo *pp_info)
{
	kern_expression *kexp;
	kern_expression *karg;
	StringInfoData buf;
	size_t		sz;

	sz = MAXALIGN(offsetof(kern_expression,
						   u.pack.offset[context->kvecs_ndims + 1]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype  = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode   = FuncOpCode__Packed;
	kexp->args_offset = sz;
	kexp->u.pack.npacked = context->kvecs_ndims;

	initStringInfo(&buf);
	buf.len = sz;
	for (int depth=0; depth <= context->kvecs_ndims; depth++)
	{
		karg = __codegen_build_loadvars_one(context, depth);
		if (karg)
		{
			kexp->u.pack.offset[depth]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
			pfree(karg);
			kexp->nr_args++;
		}
	}
	if (kexp->nr_args > 0)
	{
		bytea	   *xpucode;

		memcpy(buf.data, kexp, sz);
		__appendKernExpMagicAndLength(&buf, 0);

		xpucode = palloc(VARHDRSZ + buf.len);
		memcpy(xpucode->vl_dat, buf.data, buf.len);
		SET_VARSIZE(xpucode, VARHDRSZ + buf.len);

		pp_info->kexp_load_vars_packed = xpucode;
	}
	pfree(buf.data);
}

static kern_expression *
__codegen_build_movevars_one(codegen_context *context, int depth, int gist_depth)
{
	kern_expression *kexp;
	StringInfoData buf;
	int			nrooms = list_length(context->kvars_deflist);
	int			nitems = 0;
	size_t		sz = MAXALIGN(offsetof(kern_expression, u.move.desc[nrooms]));
	ListCell   *lc;

	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype  = TypeOpCode__int4;
    kexp->expflags = context->kexp_flags;
    kexp->opcode   = FuncOpCode__MoveVars;
	kexp->args_offset = sz;
	kexp->u.move.depth = depth;

	Assert(depth >= 0 && depth <= context->num_rels);
	foreach (lc, context->kvars_deflist)
	{
		const codegen_kvar_defitem *kvdef = lfirst(lc);

		/* temporary variables? */
		if (kvdef->kv_offset < 0)
			continue;

		if ((kvdef->kv_depth >= 0 &&
			 kvdef->kv_depth <= depth &&
			 kvdef->kv_maxref > depth) ||
			(kvdef->kv_depth == gist_depth &&
			 kvdef->kv_maxref == depth+1))
		{
			kern_varmove_desc  *vm_desc = &kexp->u.move.desc[nitems++];

			memset(vm_desc, 0, sizeof(kern_varmove_desc));
			vm_desc->vm_offset    = kvdef->kv_offset;
			vm_desc->vm_slot_id   = kvdef->kv_slot_id;
			vm_desc->vm_from_xdatum = (gist_depth < 0
									   ? kvdef->kv_depth == depth
									   : kvdef->kv_depth == gist_depth);
		}
	}
	if (nitems == 0)
		return NULL;
	kexp->u.move.nitems = nitems;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (const char *)kexp, sz);
	__appendKernExpMagicAndLength(&buf, 0);

	return (kern_expression *)buf.data;
}

void
codegen_build_packed_kvars_move(codegen_context *context, pgstromPlanInfo *pp_info)
{
	kern_expression *kexp;
	kern_expression *karg;
	StringInfoData buf;
	size_t		sz;
	int			nvalids = 0;
	int			gist_depth = context->num_rels + 1;

	/* Only NVIDIA-GPU has kvec-buffer for MoveVars across depth */
	if ((context->xpu_task_flags & DEVKIND__NVIDIA_GPU) == 0)
		return;

	sz = MAXALIGN(offsetof(kern_expression,
						   u.pack.offset[context->num_rels+1]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype  = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode   = FuncOpCode__Packed;
	kexp->args_offset = sz;
	kexp->u.pack.npacked = context->kvecs_ndims;

	initStringInfo(&buf);
	buf.len = sz;
	for (int depth=0; depth <= context->num_rels; depth++)
	{
		karg = __codegen_build_movevars_one(context, depth, -1);
		if (karg)
		{
			kexp->u.pack.offset[depth]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
			pfree(karg);
            nvalids++;
		}
		if (depth > 0 && pp_info->inners[depth-1].gist_clause != NULL)
		{
			karg = __codegen_build_movevars_one(context, depth-1, gist_depth);
			if (karg)
			{
				kexp->u.pack.offset[gist_depth]
					= __appendBinaryStringInfo(&buf, karg, karg->len);
				pfree(karg);
				nvalids++;
			}
			gist_depth++;
		}
	}

	if (nvalids > 0)
	{
		bytea	   *xpucode;

		memcpy(buf.data, kexp, sz);
		__appendKernExpMagicAndLength(&buf, 0);

		xpucode = palloc(VARHDRSZ + buf.len);
		memcpy(xpucode->vl_dat, buf.data, buf.len);
		SET_VARSIZE(xpucode, VARHDRSZ + buf.len);

		pp_info->kexp_move_vars_packed = xpucode;
	}
	pfree(buf.data);

	pp_info->kvecs_bufsz = KVEC_ALIGN(context->kvecs_usage);
	pp_info->kvecs_ndims = context->kvecs_ndims;
}

/*
 * codegen_build_scan_quals
 */
bytea *
codegen_build_scan_quals(codegen_context *context, List *dev_quals)
{
	StringInfoData buf;
	bytea	   *xpucode = NULL;
	Expr	   *expr;
	uint32_t	stack_usage_saved = context->stack_usage;
	int			saved_depth = context->curr_depth;

	Assert(context->elevel >= ERROR);
	if (dev_quals == NIL)
		return NULL;
	if (list_length(dev_quals) == 1)
		expr = linitial(dev_quals);
	else
		expr = make_andclause(dev_quals);

	initStringInfo(&buf);
	context->curr_depth = 0;
	context->stack_usage = 0;
	if (codegen_expression_walker(context, &buf, 0, expr) == 0)
	{
		xpucode = palloc(VARHDRSZ+buf.len);
		memcpy(xpucode->vl_dat, buf.data, buf.len);
		SET_VARSIZE(xpucode, VARHDRSZ+buf.len);
	}
	context->stack_usage = Max(stack_usage_saved, context->stack_usage);
	pfree(buf.data);
	context->curr_depth = saved_depth;

	return xpucode;
}

static codegen_kvar_defitem *
try_inject_projection_expression(codegen_context *context,
								 kern_expression *kexp_proj,
								 StringInfo buf,
								 Expr *expr)
{
	codegen_kvar_defitem *kvdef;
	int		pos = buf->len;

	kvdef = __try_inject_temporary_expression(context, buf, expr,
											  context->num_rels+1,
											  true);
	for (int i=0; i < kexp_proj->u.proj.nattrs; i++)
	{
		uint16_t	proj_slot_id = kexp_proj->u.proj.slot_id[i];

		/* revert expression if already loaded */
		if (kvdef->kv_slot_id == proj_slot_id)
		{
			buf->len = pos;
			goto bailout;
		}
	}
	kexp_proj->nr_args++;
bailout:
	return kvdef;
}

/*
 * __codegen_build_joinquals
 */
static kern_expression *
__codegen_build_joinquals(codegen_context *context,
						  List *join_quals,
						  List *other_quals,
						  int curr_depth)
{
	StringInfoData	buf;
	kern_expression	kexp;
	ListCell	   *lc;
	uint32_t		stack_usage_saved = context->stack_usage;
	uint32_t		stack_usage_max = stack_usage_saved;
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
		if (codegen_expression_walker(context, &buf, curr_depth, qual) < 0)
			return NULL;
		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}

	kexp_flags__saved = context->kexp_flags;
	context->kexp_flags |= KEXP_FLAG__IS_PUSHED_DOWN;
	foreach (lc, other_quals)
	{
		Expr   *qual = lfirst(lc);

		if (exprType((Node *)qual) != BOOLOID)
			elog(ERROR, "Bub? JOIN quals must be boolean");
		if (codegen_expression_walker(context, &buf, curr_depth, qual) < 0)
			return NULL;
		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;
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
	int			depth;
	int			nrels;
	size_t		sz;
	uint32_t	stack_usage_saved = context->stack_usage;
	uint32_t	stack_usage_max = stack_usage_saved;
	ListCell   *lc1, *lc2;
	char	   *result = NULL;

	nrels = list_length(stacked_join_quals);
	sz = MAXALIGN(offsetof(kern_expression, u.pack.offset[nrels+1]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Packed;
	kexp->args_offset = sz;
	kexp->u.pack.npacked = nrels + 1;

	initStringInfo(&buf);
	buf.len = sz;

	depth = 1;
	forboth (lc1, stacked_join_quals,
			 lc2, stacked_other_quals)
	{
		List   *join_quals = lfirst(lc1);
		List   *other_quals = lfirst(lc2);
		kern_expression *karg;

		karg = __codegen_build_joinquals(context,
										 join_quals,
										 other_quals,
										 depth);
		if (karg)
		{
			kexp->u.pack.offset[depth]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
			kexp->nr_args++;
			pfree(karg);
		}
		depth++;

		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;
	Assert(depth == nrels+1);

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
						   List *hash_keys,
						   int curr_depth)
{
	kern_expression *kexp;
	StringInfoData buf;
	size_t		sz = MAXALIGN(SizeOfKernExpr(0));
	uint32_t	stack_usage_saved = context->stack_usage;
	uint32_t	stack_usage_max = stack_usage_saved;
	ListCell   *lc;

	if (hash_keys == NIL)
		return NULL;

	kexp = alloca(sizeof(kern_expression));
	memset(kexp, 0, sizeof(kern_expression));
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

		codegen_expression_walker(context, &buf, curr_depth, expr);

		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;

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
	int			depth;
	int			nrels;
	size_t		sz;
	uint32_t	stack_usage_max = context->stack_usage;
	ListCell   *lc;
	char	   *result = NULL;

	nrels = list_length(stacked_hash_keys);
	sz = MAXALIGN(offsetof(kern_expression, u.pack.offset[nrels+1]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Packed;
	kexp->args_offset = sz;
	kexp->u.pack.npacked = nrels + 1;

	initStringInfo(&buf);
	buf.len = sz;

	depth = 1;
	foreach (lc, stacked_hash_keys)
	{
		kern_expression *karg;
		List   *hash_keys = lfirst(lc);

		context->stack_usage = 0;
		karg = __codegen_build_hash_value(context, hash_keys, depth);
		if (karg)
		{
			kexp->u.pack.offset[depth]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
			kexp->nr_args++;
		}
		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		depth++;
	}
	context->stack_usage = stack_usage_max;
	Assert(depth == nrels+1);

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
							  int   curr_depth,
							  Oid	gist_func_oid,
							  Oid	gist_index_oid,
							  int	gist_index_col,
							  Expr *gist_func_arg)
{
	codegen_kvar_defitem *kvdef;
	kern_expression	kexp;
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	Oid				argtypes[2];
	uint32_t		off;
	uint32_t		__pos1;
	uint32_t		__pos2;
	uint32_t		gist_depth;
	uint32_t		htup_slot_id;
	uint32_t		htup_offset;

	/* device GiST evaluation operator */
	argtypes[0] = get_atttype(gist_index_oid,
							  gist_index_col);
	argtypes[1] = exprType((Node *)gist_func_arg);
	dfunc = __pgstrom_devfunc_lookup(gist_func_oid,
									 2, argtypes,
									 InvalidOid);
	if (!dfunc)
		elog(ERROR, "Bug? cache lookup failed for device function: %u", gist_func_oid);
	dtype = dfunc->func_argtypes[0];
	gist_depth = context->kvecs_ndims++;
	/*
	 * allocation of the special pointer for reference of the index'ed
	 * heap-tuple (to be used in the post-processing)
	 */
	kvdef = palloc0(sizeof(codegen_kvar_defitem));
	kvdef->kv_slot_id   = list_length(context->kvars_deflist);
	kvdef->kv_depth     = gist_depth;
	kvdef->kv_resno     = InvalidAttrNumber;
	kvdef->kv_maxref    = curr_depth;
	kvdef->kv_offset    = context->kvecs_usage;
	kvdef->kv_type_oid  = INTERNALOID;
	kvdef->kv_type_code = TypeOpCode__internal;
	kvdef->kv_typbyval  = false;
	kvdef->kv_typalign  = sizeof(int64_t);
	kvdef->kv_typlen    = sizeof(CUdeviceptr);
	kvdef->kv_expr      = (Expr *)makeNullConst(INTERNALOID, -1, InvalidOid);
	__assign_codegen_kvar_defitem_subfields(kvdef);
	context->kvecs_usage += KVEC_ALIGN(sizeof(kvec_internal_t));
	context->kvars_deflist = lappend(context->kvars_deflist, kvdef);
	htup_slot_id = kvdef->kv_slot_id;
	htup_offset  = kvdef->kv_offset;

	/*
	 * allocation of the index-variable reference
	 */
	kvdef = palloc0(sizeof(codegen_kvar_defitem));
	kvdef->kv_slot_id   = list_length(context->kvars_deflist);
	kvdef->kv_depth     = gist_depth;
	kvdef->kv_resno     = gist_index_col;
	kvdef->kv_maxref    = -1;
	kvdef->kv_offset    = -1;
	kvdef->kv_type_oid  = dtype->type_oid;
	kvdef->kv_type_code = dtype->type_code;
	kvdef->kv_typbyval  = dtype->type_byval;
	kvdef->kv_typalign  = dtype->type_align;
	kvdef->kv_typlen    = dtype->type_length;
	kvdef->kv_expr      = (Expr *)makeNullConst(dtype->type_oid, -1, InvalidOid);
	__assign_codegen_kvar_defitem_subfields(kvdef);
	context->kvars_deflist = lappend(context->kvars_deflist, kvdef);

	/* setup GiSTEval expression */
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = dfunc->func_rettype->type_code;
	kexp.expflags = context->kexp_flags;
	kexp.opcode   = FuncOpCode__GiSTEval;
	kexp.nr_args  = 1;
	kexp.args_offset = offsetof(kern_expression, u.gist.data);
	kexp.u.gist.gist_oid     = gist_index_oid;
	kexp.u.gist.gist_depth   = kvdef->kv_depth;
	kexp.u.gist.htup_slot_id = htup_slot_id;
	kexp.u.gist.htup_offset  = htup_offset;
	kexp.u.gist.ivar_desc.vl_resno   = kvdef->kv_resno;
	kexp.u.gist.ivar_desc.vl_slot_id = kvdef->kv_slot_id;
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
	kexp.u.v.var_slot_id = kvdef->kv_slot_id;
	kexp.u.v.var_offset = -1;
	__pos2 = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExprVar);
	__appendKernExpMagicAndLength(buf, __pos2);

	/* 2nd argument - outer reference */
	//check 'curr_depth' is suitable here. gist_depth?
	codegen_expression_walker(context, buf, curr_depth, gist_func_arg);

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
	size_t			head_sz;
	uint32_t		stack_usage_saved = context->stack_usage;
	uint32_t		stack_usage_max = stack_usage_saved;
	bytea		   *result = NULL;

	head_sz = MAXALIGN(offsetof(kern_expression,
								u.pack.offset[pp_info->num_rels+1]));
	kexp = alloca(head_sz);
	memset(kexp, 0, head_sz);
	kexp->exptype  = TypeOpCode__int4;
    kexp->expflags = context->kexp_flags;
    kexp->opcode   = FuncOpCode__Packed;
    kexp->args_offset = head_sz;
    kexp->u.pack.npacked = pp_info->num_rels + 1;

	initStringInfo(&buf);
	buf.len = head_sz;
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
											i+1,
											pp_inner->gist_func_oid,
											pp_inner->gist_index_oid,
											pp_inner->gist_index_col,
											gist_func_arg);
		kexp->u.pack.offset[i+1] = off;
		kexp->nr_args++;

		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = stack_usage_saved;
	}
	context->stack_usage = stack_usage_max;

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
 * codegen_build_projection
 */
bytea *
codegen_build_projection(codegen_context *context,
						 List *proj_hash)
{
	kern_expression	*kexp;
	StringInfoData buf;
	bytea	   *xpucode;
	bool		meet_resjunk = false;
	int			nattrs = 0;
	int			sz;
	uint32_t	stack_usage_max = context->stack_usage;
	ListCell   *lc;

	/* count nattrs */
	foreach (lc, context->tlist_dev)
	{
		TargetEntry *tle = lfirst(lc);

		if (tle->resjunk)
		{
			meet_resjunk = true;
			continue;
		}
		else if (meet_resjunk)
			elog(ERROR, "Bug? a valid TLE after junk TLEs");
		else
			nattrs++;
	}
	sz = MAXALIGN(offsetof(kern_expression, u.proj.slot_id[nattrs]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);

	initStringInfo(&buf);
	buf.len = sz;
	foreach (lc, context->tlist_dev)
	{
		TargetEntry	*tle = lfirst(lc);
		codegen_kvar_defitem *kvdef;

		if (tle->resjunk)
			break;
		kvdef = try_inject_projection_expression(context,
												 kexp,
												 &buf,
												 tle->expr);
		kexp->u.proj.slot_id[kexp->u.proj.nattrs++] = kvdef->kv_slot_id;

		stack_usage_max = Max(stack_usage_max, context->stack_usage);
		context->stack_usage = 0;
	}
	context->stack_usage = stack_usage_max;

	/* hash-value (optional; for pinned inner buffer) */
	if (proj_hash != NIL)
	{
		const kern_expression  *khash
			= __codegen_build_hash_value(context,
										 proj_hash,
										 context->num_rels+1);
		if (khash)
			kexp->u.proj.hash = __appendBinaryStringInfo(&buf, khash, khash->len);
	}
	Assert(nattrs == kexp->u.proj.nattrs);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Projection;
	kexp->args_offset = sz;
	kexp->u.proj.nattrs = nattrs;
	memcpy(buf.data, kexp, sz);
	__appendKernExpMagicAndLength(&buf, 0);

	xpucode = palloc(VARHDRSZ + buf.len);
	memcpy(xpucode->vl_dat, buf.data, buf.len);
	SET_VARSIZE(xpucode, VARHDRSZ + buf.len);

	pfree(buf.data);

	return xpucode;
}

/*
 * codegen_build_groupby_keyhash
 */
static List *
codegen_build_groupby_keyhash(codegen_context *context,
                              pgstromPlanInfo *pp_info)
{
	StringInfoData buf;
	List		   *groupby_key_items = NIL;
	kern_expression	kexp;
	char		   *xpucode;
	ListCell	   *lc1, *lc2;

	/*
	 * Add variable slots to reference grouping-keys from the input and
	 * kds_final buffer.
	 */
	initStringInfo(&buf);
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = TypeOpCode__int4;
	kexp.expflags = context->kexp_flags;
	kexp.opcode = FuncOpCode__HashValue;
	kexp.nr_args = 0;
	kexp.args_offset = MAXALIGN(SizeOfKernExpr(0));
	buf.len = kexp.args_offset;

	forboth (lc1, context->tlist_dev,
			 lc2, pp_info->groupby_actions)
	{
		TargetEntry *tle = lfirst(lc1);
		int		action = lfirst_int(lc2);

		if (action == KAGG_ACTION__VREF)
		{
			codegen_kvar_defitem *kvdef;

			kvdef = __try_inject_temporary_expression(context, &buf, tle->expr,
													  context->num_rels+1,
													  false);
			groupby_key_items = lappend(groupby_key_items, kvdef);
			kexp.nr_args++;
		}
	}

	if (groupby_key_items != NIL)
	{
		memcpy(buf.data, &kexp, SizeOfKernExpr(0));
		__appendKernExpMagicAndLength(&buf, 0);

		xpucode = palloc(VARHDRSZ + buf.len);
		memcpy(xpucode + VARHDRSZ, buf.data, buf.len);
		SET_VARSIZE(xpucode, VARHDRSZ + buf.len);
		pp_info->kexp_groupby_keyhash = (bytea *)xpucode;
	}
	else
	{
		pp_info->kexp_groupby_keyhash = NULL;
	}
	pfree(buf.data);
	return groupby_key_items;
}

/*
 * codegen_build_groupby_keyload
 */
static List *
codegen_build_groupby_keyload(codegen_context *context,
							  pgstromPlanInfo *pp_info)
{
	ListCell   *lc1, *lc2;
	List	   *groupby_key_items = NIL;

	forboth (lc1, context->tlist_dev,
			 lc2, pp_info->groupby_actions)
	{
		TargetEntry *tle = lfirst(lc1);
		int		action = lfirst_int(lc2);

		if (!tle->resjunk && action == KAGG_ACTION__VREF)
		{
			codegen_kvar_defitem *kvdef;
			devtype_info *dtype;
			Oid		type_oid = exprType((Node *)tle->expr);

			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype)
				elog(ERROR, "type %s is not device supported",
					 format_type_be(type_oid));

			kvdef = palloc0(sizeof(codegen_kvar_defitem));
			kvdef->kv_slot_id   = list_length(context->kvars_deflist);
			kvdef->kv_depth     = SPECIAL_DEPTH__PREAGG_FINAL;
			kvdef->kv_resno     = tle->resno;
			kvdef->kv_maxref    = -1;
			kvdef->kv_offset    = -1;
			kvdef->kv_type_oid  = dtype->type_oid;
			kvdef->kv_type_code = dtype->type_code;
			kvdef->kv_typbyval  = dtype->type_byval;
			kvdef->kv_typalign  = dtype->type_align;
			kvdef->kv_typlen    = dtype->type_length;
			kvdef->kv_expr      = tle->expr;
			__assign_codegen_kvar_defitem_subfields(kvdef);
			context->kvars_deflist = lappend(context->kvars_deflist, kvdef);
			groupby_key_items = lappend(groupby_key_items, kvdef);
		}
	}

	if (groupby_key_items != NIL)
	{
		kern_expression	*kexp;
		char	   *xpucode = NULL;

		kexp = __codegen_build_loadvars_one(context, SPECIAL_DEPTH__PREAGG_FINAL);
		if (kexp)
		{
			xpucode = palloc(VARHDRSZ + kexp->len);
			memcpy(xpucode + VARHDRSZ, kexp, kexp->len);
			SET_VARSIZE(xpucode, VARHDRSZ + kexp->len);
			pfree(kexp);
		}
		pp_info->kexp_groupby_keyload = (bytea *)xpucode;
	}
	else
	{
		pp_info->kexp_groupby_keyload = NULL;
	}
	return groupby_key_items;
}

/*
 * codegen_build_groupby_keycomp
 */
static void
codegen_build_groupby_keycomp(codegen_context *context,
							  pgstromPlanInfo *pp_info,
							  List *groupby_keys_input,
							  List *groupby_keys_final)
{
	StringInfoData	buf;
	kern_expression kexp;
	size_t			sz;
	char		   *xpucode;
	ListCell	   *lc1, *lc2;

	Assert(pp_info->groupby_actions != NIL &&
		   list_length(groupby_keys_input) == list_length(groupby_keys_final));
	initStringInfo(&buf);

	forboth (lc1, groupby_keys_input,
			 lc2, groupby_keys_final)
	{
		codegen_kvar_defitem *kvdef_i = lfirst(lc1);
		codegen_kvar_defitem *kvdef_f = lfirst(lc2);
		Oid			type_oid = exprType((Node *)kvdef_i->kv_expr);
		Oid			coll_oid = exprCollation((Node *)kvdef_i->kv_expr);
		int			pos, __pos;
		devtype_info *dtype;
		devfunc_info *dfunc;

		Assert(type_oid == exprType((Node *)kvdef_f->kv_expr) &&
			   coll_oid == exprCollation((Node *)kvdef_f->kv_expr));
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
		kexp.u.v.var_slot_id = kvdef_i->kv_slot_id;
		kexp.u.v.var_offset = -1; //FIXME: how to handle SaveExpr
		__pos = __appendBinaryStringInfo(&buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(&buf, __pos);	/* end of FuncExpr */

		/* final variable */
		memset(&kexp, 0, sizeof(kern_expression));
		kexp.exptype = dtype->type_code;
		kexp.expflags = context->kexp_flags;
		kexp.opcode = FuncOpCode__VarExpr;
		kexp.u.v.var_slot_id = kvdef_f->kv_slot_id;
		kexp.u.v.var_offset = -1;
		__pos = __appendBinaryStringInfo(&buf, &kexp, SizeOfKernExprVar);
		__appendKernExpMagicAndLength(&buf, __pos);	/* end of VarExpr */

		__appendKernExpMagicAndLength(&buf, pos);	/* end of FuncExpr */
	}

	if (list_length(groupby_keys_input) > 1)
	{
		kern_expression *payload = (kern_expression *)buf.data;
		int		nr_args = list_length(groupby_keys_input);
		int		payload_sz = buf.len;

		initStringInfo(&buf);
		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype = TypeOpCode__bool;
		kexp.expflags = context->kexp_flags;
		kexp.opcode = FuncOpCode__BoolExpr_And;
		kexp.nr_args = nr_args;
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
 * try_inject_groupby_expression
 */
static codegen_kvar_defitem *
try_inject_groupby_expression(codegen_context *context,
							  kern_expression *kexp_pagg,
							  StringInfo buf,
							  Expr *expr)
{
	codegen_kvar_defitem *kvdef;
	int		pos = buf->len;

	Assert(kexp_pagg->opcode == FuncOpCode__AggFuncs);
	kvdef = __try_inject_temporary_expression(context, buf, expr,
											  context->num_rels+1,
											  false);
	for (int i=0; i < kexp_pagg->u.pagg.nattrs; i++)
	{
		const kern_aggregate_desc *desc = &kexp_pagg->u.pagg.desc[i];

		if (desc->action == KAGG_ACTION__COVAR)
		{
			if (desc->arg0_slot_id == kvdef->kv_slot_id ||
				desc->arg1_slot_id == kvdef->kv_slot_id)
			{
				buf->len = pos;
				goto bailout;
			}
		}
		else if (desc->action != KAGG_ACTION__NROWS_ANY)
		{
			if (desc->arg0_slot_id == kvdef->kv_slot_id)
			{
				buf->len = pos;
				goto bailout;
			}
		}
	}
	kexp_pagg->nr_args++;
bailout:
	return kvdef;
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
	size_t		head_sz = MAXALIGN(offsetof(kern_expression, u.pagg.desc[nattrs]));
	bytea	   *xpucode;
	ListCell   *lc1, *lc2, *lc3;
	kern_expression *kexp;

	kexp = alloca(head_sz);
	memset(kexp, 0, head_sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode = FuncOpCode__AggFuncs;
	kexp->nr_args = 0;
	kexp->args_offset = head_sz;

	initStringInfo(&buf);
	buf.len = head_sz;
	forthree (lc1, context->tlist_dev,
			  lc2, pp_info->groupby_actions,
			  lc3, pp_info->groupby_typmods)
	{
		TargetEntry *tle = lfirst(lc1);
		int		__action_flags = lfirst_int(lc2);
		int		action = (__action_flags & ~__KAGG_ACTION__USE_FILTER);
		bool	use_filter = ((__action_flags & __KAGG_ACTION__USE_FILTER) != 0);
		int		typmod = lfirst_int(lc3);
		kern_aggregate_desc *desc = &kexp->u.pagg.desc[kexp->u.pagg.nattrs];
		codegen_kvar_defitem *kvdef;

		Assert(!tle->resjunk);
		if (action == KAGG_ACTION__VREF ||
			action == KAGG_ACTION__VREF_NOKEY)
		{
			kvdef = try_inject_groupby_expression(context,
												  kexp,
												  &buf,
												  tle->expr);
			desc->action = KAGG_ACTION__VREF;
			desc->arg0_slot_id = -1;
			desc->arg1_slot_id = -1;
			desc->filter_slot_id = -1;
			desc->typmod = typmod;
			desc->arg0_slot_id = kvdef->kv_slot_id;
		}
		else
		{
			/* other KAGG_ACTION__* */
			FuncExpr   *func = (FuncExpr *)tle->expr;
			ListCell   *cell;

			Assert(IsA(func, FuncExpr) && list_length(func->args) <= (!use_filter ? 2 : 3));
			desc->action = action;
			desc->arg0_slot_id = -1;
			desc->arg1_slot_id = -1;
			desc->filter_slot_id = -1;
			desc->typmod = typmod;
			foreach (cell, func->args)
			{
				Expr   *fn_arg = lfirst(cell);

				kvdef = try_inject_groupby_expression(context,
													  kexp,
													  &buf,
													  fn_arg);
				if (use_filter && cell == list_tail(func->args))
				{
					desc->filter_slot_id = kvdef->kv_slot_id;
				}
				else if (cell == list_head(func->args))
				{
					desc->arg0_slot_id = kvdef->kv_slot_id;
				}
				else
				{
					Assert(action == KAGG_ACTION__COVAR);
					desc->arg1_slot_id = kvdef->kv_slot_id;
				}
			}
		}
		kexp->u.pagg.nattrs++;
	}
	memcpy(buf.data, kexp, head_sz);
	__appendKernExpMagicAndLength(&buf, 0);

	xpucode = palloc(VARHDRSZ + buf.len);
	memcpy(xpucode->vl_dat, buf.data, buf.len);
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
	List   *groupby_keys_input;
	List   *groupby_keys_final;

	Assert(pp_info->groupby_actions != NIL &&
		   list_length(pp_info->groupby_actions) <= list_length(context->tlist_dev));
	groupby_keys_input = codegen_build_groupby_keyhash(context, pp_info);
	groupby_keys_final = codegen_build_groupby_keyload(context, pp_info);
	if (groupby_keys_input != NIL &&
		groupby_keys_final != NIL)
	{
		codegen_build_groupby_keycomp(context, pp_info,
									  groupby_keys_input,
									  groupby_keys_final);
	}
	__codegen_build_groupby_actions(context, pp_info);
}

/*
 * codegen_build_gpusort_keydesc
 */
bytea *
codegen_build_gpusort_keydesc(codegen_context *context,
							  pgstromPlanInfo *pp_info)
{
	StringInfoData buf;
	int			i, nkeys = list_length(pp_info->gpusort_keys_expr);
	int			usage = 0;
	size_t		sz;
	ListCell   *lc1, *lc2;
	kern_expression *kexp;

	Assert(nkeys == list_length(pp_info->gpusort_keys_kind));
	if (nkeys == 0)
		return NULL;	/* quick bailout */
	initStringInfo(&buf);
	sz = VARHDRSZ + offsetof(kern_expression, u.sort.desc[nkeys]);
	enlargeStringInfo(&buf, sz);
	memset(buf.data, 0, sz);
	kexp = (kern_expression *)(buf.data + VARHDRSZ);
	kexp->exptype = TypeOpCode__int4;
	kexp->opcode = FuncOpCode__SortKeys;
	kexp->u.sort.nkeys = nkeys;
	buf.len = sz;

	i = 0;
	forboth (lc1, pp_info->gpusort_keys_expr,
			 lc2, pp_info->gpusort_keys_kind)
	{
		kern_sortkey_desc *keydesc = &kexp->u.sort.desc[i++];
		Expr	   *expr = lfirst(lc1);
		int			ival = lfirst_int(lc2);
		int			kind = (ival & KSORT_KEY_KIND__MASK);
		devtype_info *dtype;

		keydesc->kind = kind;
		keydesc->nulls_first = ((ival & KSORT_KEY_ATTR__NULLS_FIRST) != 0);
		keydesc->order_asc   = ((ival & KSORT_KEY_ATTR__ORDER_ASC)  != 0);
		if (kind == KSORT_KEY_KIND__VREF)
		{
			ListCell   *cell;
			bool		found = false;

			foreach (cell, context->tlist_dev)
			{
				TargetEntry *tle = lfirst(cell);

				if (tle->resjunk)
					continue;
				if (equal(expr, tle->expr))
				{
					keydesc->src_anum = tle->resno;
					found = true;
					break;
				}
			}
			if (!found)
				elog(ERROR, "Bug? GPU-SortKey (%s) is missing",
					 nodeToString((Node *)expr));
			keydesc->buf_offset = 0;
			dtype = pgstrom_devtype_lookup(exprType((Node *)expr));
			if (!dtype)
				elog(ERROR, "Bug? GPU-SortKey does not have device supported type: %s",
					 nodeToString((Node *)expr));
			keydesc->key_type_code = dtype->type_code;
		}
		else
		{
			FuncExpr   *func = (FuncExpr *)expr;

			if (!IsA(func, FuncExpr) || list_length(func->args) > 1)
				elog(ERROR, "Bug? GPU-SortKey is not unexpected expression: %s",
					 nodeToString(expr));
			kexp->u.sort.needs_finalization = true;
			if (func->args == 0)
				keydesc->src_anum = 0;
			else
			{
				ListCell   *cell;
				Expr	   *farg = linitial(func->args);
				bool		found = false;

				foreach (cell, context->tlist_dev)
				{
					TargetEntry *tle = lfirst(cell);

					if (tle->resjunk)
						continue;
					if (equal(farg, tle->expr))
					{
						keydesc->src_anum = tle->resno;
						found = true;
						break;
					}
				}
				if (!found)
					elog(ERROR, "Bug? GPU-SortKey (%s) is missing", nodeToString(expr));
			}
			switch (kind)
			{
				case KSORT_KEY_KIND__PMINMAX_INT64:
				case KSORT_KEY_KIND__PSUM_INT64:
					keydesc->buf_offset = 0;	/* no finalization */
					keydesc->key_type_code = TypeOpCode__int8;
					break;
				case KSORT_KEY_KIND__PMINMAX_FP64:
				case KSORT_KEY_KIND__PSUM_FP64:
					keydesc->buf_offset = 0;	/* no finalization */
					keydesc->key_type_code = TypeOpCode__float8;
					break;
				case KSORT_KEY_KIND__PSUM_NUMERIC:
					keydesc->buf_offset = 0;	/* no finalization */
					keydesc->key_type_code = TypeOpCode__numeric;
					break;
					/* finalization to fp64 */
				case KSORT_KEY_KIND__PAVG_INT64:
				case KSORT_KEY_KIND__PAVG_FP64:
				case KSORT_KEY_KIND__PAVG_NUMERIC:
					keydesc->buf_offset = usage;
					usage += (sizeof(bool) + sizeof(float8));
					keydesc->key_type_code = TypeOpCode__float8;
					break;
					/* finalization to fp64 */
				case KSORT_KEY_KIND__PVARIANCE_SAMP:
				case KSORT_KEY_KIND__PVARIANCE_POP:
				case KSORT_KEY_KIND__PCOVAR_CORR:
				case KSORT_KEY_KIND__PCOVAR_SAMP:
				case KSORT_KEY_KIND__PCOVAR_POP:
				case KSORT_KEY_KIND__PCOVAR_AVGX:
				case KSORT_KEY_KIND__PCOVAR_AVGY:
				case KSORT_KEY_KIND__PCOVAR_COUNT:
				case KSORT_KEY_KIND__PCOVAR_INTERCEPT:
				case KSORT_KEY_KIND__PCOVAR_REGR_R2:
				case KSORT_KEY_KIND__PCOVAR_REGR_SLOPE:
				case KSORT_KEY_KIND__PCOVAR_REGR_SXX:
				case KSORT_KEY_KIND__PCOVAR_REGR_SXY:
					keydesc->buf_offset = usage;
					usage += (sizeof(bool) + sizeof(float8));
					keydesc->key_type_code = TypeOpCode__float8;
					break;
				default:
					elog(ERROR, "Bug? unknown KSORT_KEY_KIND: %d", kind);
			}
		}
	}
	/* GPU-Sort + Window-Rank() functions, if any */
	Assert(pp_info->gpusort_limit_count == 0 ||
		   pp_info->window_rank_func == 0);		/* mutually exclusive */
	kexp->u.sort.window_rank_func		= pp_info->window_rank_func;
	kexp->u.sort.window_rank_limit		= pp_info->window_rank_limit;
	kexp->u.sort.window_partby_nkeys	= pp_info->window_partby_nkeys;
	kexp->u.sort.window_orderby_nkeys	= pp_info->window_orderby_nkeys;

	/* Put MAGIC */
	__appendKernExpMagicAndLength(&buf, VARHDRSZ);
	SET_VARSIZE(buf.data, buf.len);
	/*
	 * This FuncOpCode__SortKeys operation needs 'usage' bytes of margin
	 * after the kern_tupitem on the kds_final buffer for finalization.
	 * (used to calculate temporary value like average)
	 */
	pp_info->gpusort_htup_margin = usage;

	return (bytea *)buf.data;
}

/*
 * pgstrom_xpu_expression
 *
 * checks whether the expression is executable on GPU/DPU devices.
 */
bool
pgstrom_xpu_expression(Expr *expr,
					   uint32_t xpu_task_flags,
					   Index scan_relid,
					   List *inner_target_list,
					   int *p_devcost)
{
	codegen_context *context;
	int			num_rels = list_length(inner_target_list);
	int			sz = offsetof(codegen_context, pd[num_rels+2]);
	int			depth;
	ListCell   *lc;

	Assert((xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU ||
		   (xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU);
	context = alloca(sz);
	memset(context, 0, sz);
	context->elevel = DEBUG2;
	context->top_expr = expr;
	context->xpu_task_flags = xpu_task_flags;
	context->scan_relid = scan_relid;
	context->num_rels = num_rels;
	depth = 1;
	foreach (lc, inner_target_list)
		context->pd[depth++].inner_target = (PathTarget *)lfirst(lc);
	if (!expr || IsA(expr, List))
		return false;
	if (codegen_expression_walker(context, NULL, -1, expr) < 0)
		return false;
	if (p_devcost)
		*p_devcost = context->device_cost;
	return true;
}

/*
 * estimate_cuda_stack_size
 */
uint32_t
estimate_cuda_stack_size(codegen_context *context)
{
#define CUDA_ALLOCA_ALIGN	16
	uint32_t	extra_bufsz = Max(context->extra_bufsz, 512);
	uint32_t	stack_sz;
	int			kvars_nslots;
	ListCell   *lc;

	/* only NVIDIA-GPU needs stack configuration */
	if ((context->xpu_task_flags & DEVKIND__NVIDIA_GPU) == 0)
		return 0;
	/* minimum working area */
	stack_sz = 3200;
	/* kern_context */
	stack_sz += TYPEALIGN(CUDA_ALLOCA_ALIGN,
						  offsetof(kern_context, vlbuf) + extra_bufsz);
	/* kvars_slot[] */
	kvars_nslots = list_length(context->kvars_deflist);
	stack_sz += TYPEALIGN(CUDA_ALLOCA_ALIGN,
						  sizeof(xpu_datum_t *) * kvars_nslots);
	foreach (lc, context->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);

		stack_sz += TYPEALIGN(CUDA_ALLOCA_ALIGN,
							  kvdef->kv_xdatum_sizeof);
	}
	/* other expressions */
	stack_sz += context->stack_usage;
	return stack_sz;
#undef CUDA_ALLOCA_ALIGN
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

static const codegen_kvar_defitem *
__lookup_kvar_defitem_by_slot_id(const CustomScanState *css, int slot_id)
{
	pgstromTaskState *pts = (pgstromTaskState *)css;

	if (pts)
	{
		pgstromPlanInfo *pp_info = pts->pp_info;

		if (slot_id >= 0 &&
			slot_id < list_length(pp_info->kvars_deflist))
		{
			return (const codegen_kvar_defitem *)
				list_nth(pp_info->kvars_deflist, slot_id);
		}
	}
	return NULL;
}

static const char *
__get_expression_cstring(const CustomScanState *css,
						 List *dcontext,
						 int32_t slot_id)
{
	const codegen_kvar_defitem *kvdef;
	const char *label = "???";

	kvdef = __lookup_kvar_defitem_by_slot_id(css, slot_id);
	if (kvdef)
	{
		label = deparse_expression((Node *)kvdef->kv_expr,
								   dcontext,
								   false,
								   false);
	}
	return label;
}

static void
__xpucode_const_cstring(StringInfo buf, const kern_expression *kexp)
{
	const char *dname = devtype_get_name_by_opcode(kexp->exptype);

	if (kexp->u.c.const_isnull)
	{
		appendStringInfo(buf, "{Const(%s): value=NULL}", dname);
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
						 dname,
						 DatumGetCString(label));
	}
}

static void
__xpucode_param_cstring(StringInfo buf, const kern_expression *kexp)
{
	const char *dname = devtype_get_name_by_opcode(kexp->exptype);

	appendStringInfo(buf, "{Param(%s): param_id=%u}",
					 dname,
					 kexp->u.p.param_id);
}

static void
__xpucode_var_cstring(StringInfo buf,
					  const kern_expression *kexp,
					  const CustomScanState *css,	/* optional */
					  ExplainState *es,				/* optional */
					  List *dcontext)				/* optionsl */
{
	uint32_t	kvec_sz = devtype_get_kvec_sizeof_by_opcode(kexp->exptype);
	const char *dname = devtype_get_name_by_opcode(kexp->exptype);
	const char *label = __get_expression_cstring(css, dcontext,
												 kexp->u.v.var_slot_id);

	appendStringInfo(buf, "{Var(%s):", dname);
	if (kexp->u.v.var_offset < 0)
		appendStringInfo(buf, " slot=%u", kexp->u.v.var_slot_id);
	else
		appendStringInfo(buf, " kvec=0x%04x-%04x",
						 kexp->u.v.var_offset,
						 kexp->u.v.var_offset + kvec_sz);
	appendStringInfo(buf, ", expr='%s'}", label);
}

static void
__xpucode_loadvars_cstring(StringInfo buf,
						   const kern_expression *kexp,
						   const CustomScanState *css,
						   ExplainState *es,
						   List *dcontext)
{
	int		nitems = kexp->u.load.nitems;

	Assert(kexp->nr_args == 0);

	appendStringInfo(buf, "{LoadVars(depth=%d): ", kexp->u.load.depth);
	appendStringInfo(buf, "kvars=");
	if (kexp->u.load.nitems > 0)
		appendStringInfoChar(buf, '[');
	for (int i=0; i < nitems; i++)
	{
		const kern_varload_desc *vl_desc = &kexp->u.load.desc[i];
		const codegen_kvar_defitem *kvdef;
		const char	   *dname;
		const char	   *label;

		kvdef = __lookup_kvar_defitem_by_slot_id(css, vl_desc->vl_slot_id);
		if (!kvdef)
			elog(ERROR, "failed on kernel variable with slot_id=%d",
				vl_desc->vl_slot_id);
		dname = devtype_get_name_by_opcode(kvdef->kv_type_code);
		label = __get_expression_cstring(css, dcontext,
										 vl_desc->vl_slot_id);
		if (i > 0)
			appendStringInfo(buf, ", ");
		appendStringInfo(buf, "<slot=%d, type='%s' resno=%d(%s)>",
						 vl_desc->vl_slot_id,
						 dname,
						 vl_desc->vl_resno,
						 label);
	}
	if (kexp->u.load.nitems > 0)
		appendStringInfoChar(buf, ']');
}

static void
__xpucode_movevars_cstring(StringInfo buf,
						   const kern_expression *kexp,
						   const CustomScanState *css,
						   ExplainState *es,
						   List *dcontext)
{
	Assert(kexp->nr_args == 0);
	appendStringInfo(buf, "{MoveVars(depth=%d): items=[",
					 kexp->u.move.depth);

	for (int i=0; i < kexp->u.move.nitems; i++)
	{
		const kern_varmove_desc *vm_desc = &kexp->u.move.desc[i];
		const codegen_kvar_defitem *kvdef;
		const char *dname;
		const char *label;
		uint32_t	kvec_sz;

		kvdef = __lookup_kvar_defitem_by_slot_id(css, vm_desc->vm_slot_id);
		if (!kvdef)
			elog(ERROR, "failed on kernel variable with slot_id=%d",
				 vm_desc->vm_slot_id);
		dname = devtype_get_name_by_opcode(kvdef->kv_type_code);
		kvec_sz = devtype_get_kvec_sizeof_by_opcode(kvdef->kv_type_code);
		label = __get_expression_cstring(css, dcontext,
										 vm_desc->vm_slot_id);
		if (i > 0)
			appendStringInfo(buf, ", ");
		appendStringInfo(buf, "<");
		if (vm_desc->vm_from_xdatum)
			appendStringInfo(buf, "slot=%d, ", vm_desc->vm_slot_id);
		appendStringInfo(buf, "offset=0x%04x-%04x, type='%s', expr='%s'>",
						 vm_desc->vm_offset,
						 vm_desc->vm_offset + kvec_sz - 1,
						 dname,
						 label);
	}
	appendStringInfo(buf, "]}");
}

static void
__xpucode_gisteval_cstring(StringInfo buf,
                           const kern_expression *kexp,
                           const CustomScanState *css,
                           ExplainState *es,
                           List *dcontext)
{
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);
	const codegen_kvar_defitem *kvdef;
	const char	   *dname = devtype_get_name_by_opcode(kexp->exptype);

	Assert(kexp->nr_args == 1 &&
		   kexp->exptype == karg->exptype);

	appendStringInfo(buf, "{GiSTEval(%s): gist_depth=%d",
					 dname,
					 kexp->u.gist.gist_depth);

	kvdef = __lookup_kvar_defitem_by_slot_id(css, kexp->u.gist.ivar_desc.vl_slot_id);
	if (!kvdef)
		elog(ERROR, "failed on kernel variable with slot_id=%d",
			 kexp->u.gist.ivar_desc.vl_slot_id);
	appendStringInfo(buf, " <slot=%d, idxname%d='%s', type='%s'>",
					 kexp->u.gist.ivar_desc.vl_slot_id,
					 kexp->u.gist.ivar_desc.vl_resno,
					 get_attname(kexp->u.gist.gist_oid,
								 kexp->u.gist.ivar_desc.vl_resno, false),
					 devtype_get_name_by_opcode(kvdef->kv_type_code));
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
				appendStringInfo(buf, "vref[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__NROWS_ANY:
				appendStringInfo(buf, "nrows[*");
				break;
			case KAGG_ACTION__NROWS_COND:
				appendStringInfo(buf, "nrows[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PMIN_INT32:
				appendStringInfo(buf, "pmin::int32[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PMIN_INT64:
				appendStringInfo(buf, "pmin::int64[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PMIN_FP64:
				appendStringInfo(buf, "pmin::fp64[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PMAX_INT32:
				appendStringInfo(buf, "pmax::int32[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PMAX_INT64:
				appendStringInfo(buf, "pmax::int64[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PMAX_FP64:
				appendStringInfo(buf, "pmax::fp64[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PSUM_INT:
				appendStringInfo(buf, "psum::int[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PSUM_INT64:
				appendStringInfo(buf, "psum::int64[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PSUM_FP:
				appendStringInfo(buf, "psum::fp[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PSUM_NUMERIC:
				appendStringInfo(buf, "psum::numeric(%d)[slot=%d, expr='%s'",
								 __numeric_typmod_weight(desc->typmod),
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PAVG_INT:
				appendStringInfo(buf, "pavg::int[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PAVG_INT64:
				appendStringInfo(buf, "pavg::int64[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PAVG_FP:
				appendStringInfo(buf, "pavg::fp[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__PAVG_NUMERIC:
				appendStringInfo(buf, "pavg::numeric(%d)[slot=%d, expr='%s'",
								 __numeric_typmod_weight(desc->typmod),
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__STDDEV:
				appendStringInfo(buf, "stddev[slot=%d, expr='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id));
				break;
			case KAGG_ACTION__COVAR:
				appendStringInfo(buf, "covar[slotX=%d, exprX='%s', slotY=%d, exprY='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id),
								 desc->arg1_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg1_slot_id));
				break;
			default:
				appendStringInfo(buf, "unknown[slot0=%d, expr0='%s', slot1=%d, expr1='%s'",
								 desc->arg0_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg0_slot_id),
								 desc->arg1_slot_id,
								 __get_expression_cstring(css, dcontext,
														  desc->arg1_slot_id));
				break;
		}
		if (desc->filter_slot_id >= 0)
			appendStringInfo(buf, ", filter='%s; slot=%u'",
							 __get_expression_cstring(css, dcontext,
													  desc->filter_slot_id),
							 desc->filter_slot_id);
		appendStringInfo(buf, "]");
	}
	appendStringInfo(buf, ">");
}

static void
__xpucode_sortkeys_cstring(StringInfo buf,
						   const kern_expression *kexp,
						   const CustomScanState *css,	/* optional */
						   ExplainState *es,			/* optional */
						   List *dcontext)
{
	static const char *label[] = {
		"vref",				/* KSORT_KEY_KIND__VREF */
		"min/max[int64]",	/* KSORT_KEY_KIND__PMINMAX_INT64 */
		"min/max[fp64]",	/* KSORT_KEY_KIND__PMINMAX_FP64 */
		"sum[int64]",		/* KSORT_KEY_KIND__PSUM_INT64 */
		"sum[fp64]",		/* KSORT_KEY_KIND__PSUM_FP64 */
		"sum[numeric]",		/* KSORT_KEY_KIND__PSUM_NUMERIC */
		"avg[int64]",		/* KSORT_KEY_KIND__PAVG_INT64 */
		"avg[fp64]",		/* KSORT_KEY_KIND__PAVG_FP64 */
		"avg[numeric]",		/* KSORT_KEY_KIND__PAVG_NUMERIC */
		"var[samp]",		/* KSORT_KEY_KIND__PVARIANCE_SAMP */
		"var[pop]",			/* KSORT_KEY_KIND__PVARIANCE_POP */
		"corr",				/* KSORT_KEY_KIND__PCOVAR_CORR */
		"cov[samp]",		/* KSORT_KEY_KIND__PCOVAR_SAMP */
		"cov[pop]",			/* KSORT_KEY_KIND__PCOVAR_POP */
		"cov[avgx]",		/* KSORT_KEY_KIND__PCOVAR_AVGX */
		"cov[avgy]",		/* KSORT_KEY_KIND__PCOVAR_AVGY */
		"cov[count]",		/* KSORT_KEY_KIND__PCOVAR_COUNT */
		"cov[intercept]",	/* KSORT_KEY_KIND__PCOVAR_INTERCEPT */
		"regr[r2]",			/* KSORT_KEY_KIND__PCOVAR_REGR_R2 */
		"regr[slope]",		/* KSORT_KEY_KIND__PCOVAR_REGR_SLOPE */
		"regr[sxx]",		/* KSORT_KEY_KIND__PCOVAR_REGR_SXX */
		"regr[sxy]",		/* KSORT_KEY_KIND__PCOVAR_REGR_SXY */
		"regr[syy]",		/* KSORT_KEY_KIND__PCOVAR_REGR_SYY */
		NULL,
	};

	appendStringInfo(buf, "{SortKeys");
	for (int i=0; i < kexp->u.sort.nkeys; i++)
	{
		const kern_sortkey_desc *desc = &kexp->u.sort.desc[i];

		appendStringInfo(buf, "%s <", i==0 ? "" : ",");
		if (desc->kind < KSORT_KEY_KIND__NITEMS)
			appendStringInfoString(buf, label[desc->kind]);
		else
			appendStringInfo(buf, "unknown-%u", desc->kind);
		if (css)
		{
			CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
			const char *str;

			if (desc->src_anum > 0 &&
				desc->src_anum <= list_length(cscan->custom_scan_tlist))
			{
				TargetEntry *tle = list_nth(cscan->custom_scan_tlist,
											desc->src_anum - 1);
				str = deparse_expression((Node *)tle->expr,
										 dcontext,
										 false,
										 false);
				appendStringInfo(buf, "; key=%s", str);
			}
			else if (desc->src_anum != 0)
				appendStringInfo(buf, "; key=(out of range)");
		}
		appendStringInfo(buf, "[%s;%s]",
						 desc->nulls_first ? "NF" : "NL",
						 desc->order_asc ? "ASC" : "DESC");
	}
}

static void
__xpucode_projection_cstring(StringInfo buf,
							 const kern_expression *kexp,
							 const CustomScanState *css,
							 ExplainState *es,
							 List *dcontext)
{
	appendStringInfo(buf, "{Projection: layout=<");
	for (int j=0; j < kexp->u.proj.nattrs; j++)
	{
		uint16_t	proj_slot_id = kexp->u.proj.slot_id[j];

		if (j > 0)
			appendStringInfo(buf, ",");
		appendStringInfo(buf, "%d", proj_slot_id);
	}
	appendStringInfo(buf, ">");

	if (kexp->u.proj.hash != 0)
	{
		const kern_expression *khash = (const kern_expression *)
			((const char *)kexp + kexp->u.proj.hash);
		appendStringInfo(buf, ", Hash=");
		__xpucode_to_cstring(buf, khash, css, es, dcontext);
	}
}

static void
__xpucode_to_cstring(StringInfo buf,
					 const kern_expression *kexp,
					 const CustomScanState *css,	/* optional */
					 ExplainState *es,				/* optional */
					 List *dcontext)				/* optionsl */
{
	const kern_expression *karg;
	const codegen_kvar_defitem *kvdef;
	const char *dname;
	const char *label;
	bool		device_only;
	int			i, pos;

	switch (kexp->opcode)
	{
		case FuncOpCode__ConstExpr:
			__xpucode_const_cstring(buf, kexp);
			return;
		case FuncOpCode__ParamExpr:
			__xpucode_param_cstring(buf, kexp);
			return;
		case FuncOpCode__VarExpr:
			__xpucode_var_cstring(buf, kexp, css, es, dcontext);
			return;
		case FuncOpCode__Projection:
			__xpucode_projection_cstring(buf, kexp, css, es, dcontext);
			break;
		case FuncOpCode__LoadVars:
			__xpucode_loadvars_cstring(buf, kexp, css, es, dcontext);
			break;
		case FuncOpCode__MoveVars:
			__xpucode_movevars_cstring(buf, kexp, css, es, dcontext);
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
			dname = devtype_get_name_by_opcode(kexp->exptype);
			appendStringInfo(buf, "{SaveExpr: <slot=%d, type='%s'>",
							 kexp->u.save.sv_slot_id,
							 dname);
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
		case FuncOpCode__SortKeys:
			__xpucode_sortkeys_cstring(buf, kexp, css, es, dcontext);
			break;
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
		case FuncOpCode__DistinctFrom:
			appendStringInfo(buf, "{DistinctFrom");
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

		case FuncOpCode__ScalarArrayOpAny:
		case FuncOpCode__ScalarArrayOpAll:
			Assert(kexp->nr_args == 2);
			if (kexp->opcode == FuncOpCode__ScalarArrayOpAny)
				label = "Any";
			else
				label = "All";
			appendStringInfo(buf, "{ScalarArrayOp%s: elem=<slot=%d",
							 label,
							 kexp->u.saop.elem_slot_id);
			kvdef = __lookup_kvar_defitem_by_slot_id(css, kexp->u.saop.elem_slot_id);
			if (kvdef)
				appendStringInfo(buf, ", type='%s'",
								 devtype_get_name_by_opcode(kvdef->kv_type_code));
			appendStringInfoChar(buf, '>');
			break;

		default:
			dname = devfunc_get_name_by_opcode(kexp->opcode, &device_only);
			if (!dname)
				elog(ERROR, "unknown device function (code: %u)", kexp->opcode);
			if (!device_only)
				appendStringInfo(buf, "{Func(%s)::%s",
								 devtype_get_name_by_opcode(kexp->exptype),
								 dname);
			else
				appendStringInfo(buf, "{DevOnlyFunc::%s(%s)",
								 dname,
								 devtype_get_name_by_opcode(kexp->exptype));
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

/*
 * pgstrom_explain_kvars_slot
 */
static void
__explain_kvars_slot_subfield_types(StringInfo buf, List *subfields_list)
{
	ListCell   *lc;

	foreach (lc, subfields_list)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);
		const char *dname = devtype_get_name_by_opcode(kvdef->kv_type_code);

		if (lc != list_head(subfields_list))
			appendStringInfo(buf, ", ");
		appendStringInfo(buf, "%s", dname);
		if (kvdef->kv_subfields != NIL)
		{
			appendStringInfoChar(buf, '(');
			__explain_kvars_slot_subfield_types(buf, kvdef->kv_subfields);
			appendStringInfoChar(buf, ')');
		}
	}
}

void
pgstrom_explain_kvars_slot(const CustomScanState *css,
						   ExplainState *es,
						   List *dcontext)
{
	const pgstromTaskState *pts = (const pgstromTaskState *)css;
	pgstromPlanInfo *pp_info = pts->pp_info;
	StringInfoData	buf;
	CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
	ListCell   *lc;
	uint32_t	slot_id = 0;

	initStringInfo(&buf);
	foreach (lc, pp_info->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);
		const char *dname = devtype_get_name_by_opcode(kvdef->kv_type_code);

		if (lc != list_head(pp_info->kvars_deflist))
			appendStringInfo(&buf, ", ");
		appendStringInfo(&buf, "<slot=%d", slot_id);
#if 0
		appendStringInfo(&buf, ", depth=%d, resno=%d",
						 kvdef->kv_depth,
						 kvdef->kv_resno);
#endif
		appendStringInfo(&buf, ", type='%s", dname);
		if (kvdef->kv_type_code == TypeOpCode__internal)
			appendStringInfo(&buf, "[%d]", kvdef->kv_typlen);
		else if (kvdef->kv_subfields)
		{
			appendStringInfoChar(&buf, '(');
			__explain_kvars_slot_subfield_types(&buf, kvdef->kv_subfields);
			appendStringInfoChar(&buf, ')');
		}
		appendStringInfo(&buf, "', expr='%s'",
						 deparse_expression((Node *)kvdef->kv_expr,
											dcontext,
											(cscan->custom_plans != NIL),
											false));
		if (kvdef->kv_offset >= 0)
			appendStringInfo(&buf, ", kv_off=0x%04x",
							 kvdef->kv_offset);
		appendStringInfoString(&buf, ">");
		slot_id++;
	}
	ExplainPropertyText("KVars-Slot", buf.data, es);

	pfree(buf.data);
}

static int
__sort_kvar_defitem_by_kvecs_offset(const void *__a, const void *__b)
{
	const codegen_kvar_defitem *kvdef_a = *((const codegen_kvar_defitem **)__a);
	const codegen_kvar_defitem *kvdef_b = *((const codegen_kvar_defitem **)__b);

	if (kvdef_a->kv_offset < kvdef_b->kv_offset)
		return -1;
	if (kvdef_a->kv_offset > kvdef_b->kv_offset)
		return 1;
	return 0;
}

void
pgstrom_explain_kvecs_buffer(const CustomScanState *css,
							 ExplainState *es,
							 List *dcontext)
{
	const pgstromTaskState *pts = (const pgstromTaskState *)css;
	pgstromPlanInfo *pp_info = pts->pp_info;
	const codegen_kvar_defitem **kvdef_array;
	uint32_t	nrooms = list_length(pp_info->kvars_deflist);
	uint32_t	nitems = 0;
	ListCell   *lc;
	StringInfoData	buf;

	/* kvecs-buffer actually required? */
	if (pp_info->kvecs_bufsz == 0 ||
		pp_info->kvecs_ndims == 0)
		return;

	kvdef_array = alloca(sizeof(const codegen_kvar_defitem *) * nrooms);
	foreach (lc, pp_info->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);

		if (kvdef->kv_offset >= 0)
			kvdef_array[nitems++] = kvdef;
	}
	if (nitems == 0)
		return;		/* nothing to display */
	qsort(kvdef_array, nitems, sizeof(const codegen_kvar_defitem *),
		  __sort_kvar_defitem_by_kvecs_offset);

	initStringInfo(&buf);
	appendStringInfo(&buf, "nbytes: %u, ndims: %d",
					 pp_info->kvecs_bufsz,
					 pp_info->num_rels + 2);
	if (nitems > 1)
		appendStringInfo(&buf, ", items=[");
	else if (nitems > 0)
		appendStringInfo(&buf, ", item=");
	for (int i=0; i < nitems; i++)
	{
		const codegen_kvar_defitem *kvdef = kvdef_array[i];
		const char *dname = devtype_get_name_by_opcode(kvdef->kv_type_code);
		uint32_t	kvec_sz = devtype_get_kvec_sizeof_by_opcode(kvdef->kv_type_code);

		if (i > 0)
			appendStringInfo(&buf, ", ");
		appendStringInfo(&buf, "kvec%d=<0x%04x-%04x, type='%s', expr='%s'>",
						 i,
						 kvdef->kv_offset,
						 kvdef->kv_offset + kvec_sz - 1,
						 dname,
						 deparse_expression((Node *)kvdef->kv_expr,
											dcontext,
											false, false));
	}
	if (nitems > 1)
		appendStringInfoChar(&buf, ']');

	ExplainPropertyText("KVecs-Buffer", buf.data, es);

	pfree(buf.data);
}

/*
 * pgstrom_explain_fallback_desc
 */
void
pgstrom_explain_fallback_desc(pgstromTaskState *pts,
							  ExplainState *es,
							  List *dcontext)
{
	pgstromPlanInfo *pp_info = pts->pp_info;
	kern_fallback_desc *fb_desc_array;
	StringInfoData buf;
	int			fb_desc_nitems;

	if (!pts->kern_fallback_desc)
		return;
	initStringInfo(&buf);
	fb_desc_array = (kern_fallback_desc *)VARDATA(pts->kern_fallback_desc);
	fb_desc_nitems = (VARSIZE(pts->kern_fallback_desc) -
					  VARHDRSZ) / sizeof(kern_fallback_desc);
	appendStringInfo(&buf, "[");
	for (int i=0; i < fb_desc_nitems; i++)
	{
		kern_fallback_desc *fb_desc = &fb_desc_array[i];
		Expr	   *kv_expr = NULL;
		ListCell   *lc;

		foreach (lc, pp_info->kvars_deflist)
		{
			codegen_kvar_defitem *kvdef = lfirst(lc);

			if (fb_desc->fb_src_depth == kvdef->kv_depth &&
				fb_desc->fb_src_resno == kvdef->kv_resno)
			{
				kv_expr = kvdef->kv_expr;
				break;
			}
		}
		if (i > 0)
			appendStringInfo(&buf, ", ");
		if (!kv_expr)
			appendStringInfo(&buf, "<dest='%d', slot='%d', depth=%d:%d>",
							 fb_desc->fb_dst_resno-1,
							 fb_desc->fb_slot_id,
							 fb_desc->fb_src_depth,
							 fb_desc->fb_max_depth);
		else
			appendStringInfo(&buf, "<dest='%d', expr='%s', depth=%d:%d>",
							 fb_desc->fb_dst_resno-1,
							 deparse_expression((Node *)kv_expr,
												dcontext,
												false, false),
							 fb_desc->fb_src_depth,
							 fb_desc->fb_max_depth);
	}
	appendStringInfo(&buf, "]");
	if (buf.len > 0)
		ExplainPropertyText("Fallback-desc", buf.data, es);
    pfree(buf.data);
}

/*
 * pgstrom_explain_xpucode
 */
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
	devinfo_memcxt_generation++;

	if (!MemoryContextIsEmpty(devinfo_memcxt))
	{
		/*
		 * MEMO: invalidation callback can be invoked during devtype /
		 * devfunc cache build, therefore, it is not safe to reset the
		 * devinfo_memcxt immediately.
		 * So, its deletion is postponed to the timing when portal is
		 * dropped.
		 */
		if (PortalContext)
		{
			MemoryContext	devinfo_oldcxt = devinfo_memcxt;

			devinfo_memcxt = AllocSetContextCreate(CacheMemoryContext,
												   "device type/func info cache",
												   ALLOCSET_DEFAULT_SIZES);
			MemoryContextSetParent(devinfo_oldcxt, PortalContext);
		}
		else
		{
			/*
			 * When PortalContext == NULL, it is obviously not under plan
			 * constructing. So, we can reset the memory context now.
			 */
			MemoryContextReset(devinfo_memcxt);
		}
	}
	devtype_rev_htable = NULL;
	devfunc_rev_htable = NULL;
	memset(devtype_info_slot, 0, sizeof(List *) * DEVTYPE_INFO_NSLOTS);
	memset(devfunc_info_slot, 0, sizeof(List *) * DEVFUNC_INFO_NSLOTS);

	__type_oid_cache_int1	= UINT_MAX;
	__type_oid_cache_float2	= UINT_MAX;
	__type_oid_cache_cube	= UINT_MAX;
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
