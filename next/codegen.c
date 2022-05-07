/*
 * codegen.c
 *
 * Routines for xPU code generator
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* -------- static variables --------*/
#define DEVTYPE_INFO_NSLOTS		128
#define DEVFUNC_INFO_NSLOTS		1024
static MemoryContext	devinfo_memcxt = NULL;
static dlist_head		devtype_info_slot[DEVTYPE_INFO_NSLOTS];
static dlist_head		devfunc_info_slot[DEVFUNC_INFO_NSLOTS];

#define TYPE_OPCODE(NAME,OID,EXTENSION)			\
	static uint32_t devtype_##NAME##_hash(bool isnull, Datum value);
#include "xpu_opcodes.h"

#define TYPE_OPCODE(NAME,OID,EXTENSION)				\
	{EXTENSION, #NAME, DEVKERN__ANY, devtype_##NAME##_hash, NULL },
static struct {
	const char	   *type_extension;
	const char	   *type_name;
	uint32_t		type_flags;
	devtype_hashfunc_f type_hashfunc;
	const char	   *type_alias;
} devtype_catalog[] = {
#include "xpu_opcodes.h"
	/* alias device data types */
	{NULL, "varchar", DEVKERN__ANY, NULL, "text"},
	{NULL, "cidr", DEVKERN__ANY, NULL, "inet"},
	{NULL, NULL, 0, NULL, NULL}
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
pgstrom_devtype_lookup_by_name(const char *__type_name)
{
	char   *type_name = alloca(strlen(__type_name) + 1);
	char   *type_extension = NULL;
	Oid		type_oid = InvalidOid;

	strcpy(type_name, __type_name);
	type_extension = strchr(type_name, '@');
	if (type_extension)
	{
		Relation	rel;
		SysScanDesc	sdesc;
		ScanKeyData	skey;
		HeapTuple	tup;

		*type_extension++ = '\0';
		rel = table_open(TypeRelationId, AccessShareLock);
		ScanKeyInit(&skey,
					Anum_pg_type_typname,
					BTEqualStrategyNumber, F_NAMEEQ,
					CStringGetDatum(type_name));
		sdesc = systable_beginscan(rel, TypeNameNspIndexId,
								   true, NULL, 1, &skey);
		while (HeapTupleIsValid(tup = systable_getnext(sdesc)))
		{
			Datum	datum;
			bool	isnull;
			const char *extname;

			datum = heap_getattr(tup, Anum_pg_type_oid,
								 RelationGetDescr(rel),
								 &isnull);
			extname = get_extension_name_by_object(TypeRelationId,
												   DatumGetObjectId(datum));
			if (extname && strcmp(extname, type_extension) == 0)
			{
				type_oid = DatumGetObjectId(datum);
				break;
			}
		}
	}
	else
	{
		type_oid = GetSysCacheOid2(TYPENAMENSP,
								   Anum_pg_type_oid,
								   PointerGetDatum(type_name),
								   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	}
	if (!OidIsValid(type_oid))
		elog(ERROR, "catalog lookup failed for type '%s'", type_name);
	return pgstrom_devtype_lookup(type_oid);
}

static devtype_info *
build_basic_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	devtype_info   *dtype = NULL;
	HeapTuple		htup;
	Form_pg_type	__type;
	const char	   *type_name;
	Oid				type_namespace;
	int				i;

	htup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(tcache->type_id));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for type %u", tcache->type_id);
    __type = (Form_pg_type) GETSTRUCT(htup);
    type_name = NameStr(__type->typname);
	type_namespace = __type->typnamespace;
	
	for (i=0; devtype_catalog[i].type_name != NULL; i++)
	{
		const char	   *__ext_name = devtype_catalog[i].type_extension;
		const char	   *__type_name = devtype_catalog[i].type_name;
		const char	   *__type_alias = devtype_catalog[i].type_alias;
		MemoryContext	oldcxt;

		if (ext_name
			? (__ext_name && strcmp(ext_name, __ext_name) == 0)
			: (!__ext_name && type_namespace == PG_CATALOG_NAMESPACE) &&
			strcmp(type_name, __type_name) == 0)
		{
			oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
			dtype = palloc0(offsetof(devtype_info, comp_subtypes[0]));
			if (ext_name)
				dtype->type_extension = pstrdup(ext_name);
			dtype->type_oid = tcache->type_id;
			dtype->type_flags = devtype_catalog[i].type_flags;
			dtype->type_length = tcache->typlen;
			dtype->type_align = typealign_get_width(tcache->typalign);
			dtype->type_byval = tcache->typbyval;
			dtype->type_name = pstrdup(type_name);
			dtype->type_hashfunc = devtype_catalog[i].type_hashfunc;
			/* type equality functions */
			dtype->type_eqfunc = get_opcode(tcache->eq_opr);
			dtype->type_cmpfunc = tcache->cmp_proc;
			MemoryContextSwitchTo(oldcxt);
			/* type alias, if any */
			if (__type_alias)
			{
				dtype->type_alias = pgstrom_devtype_lookup_by_name(__type_alias);
				if (!dtype->type_alias)
					dtype->type_is_negative = true;
			}
			break;
		}
	}
	ReleaseSysCache(htup);

	return dtype;
}

static devtype_info *
build_composite_devtype_info(TypeCacheEntry *tcache, const char *ext_name)
{
	TupleDesc		tupdesc = lookup_rowtype_tupdesc(tcache->type_id, -1);
	devtype_info  **subtypes = alloca(sizeof(devtype_info *) * tupdesc->natts);
	devtype_info   *dtype;
	MemoryContext	oldcxt;
	uint32_t		extra_flags = DEVKERN__ANY;
	int				j;

	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		dtype = pgstrom_devtype_lookup(attr->atttypid);
		if (!dtype)
			return NULL;
		extra_flags &= dtype->type_flags;
		subtypes[j] = dtype;
	}
	oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
	dtype = palloc0(offsetof(devtype_info,
							 comp_subtypes[tupdesc->natts]));
	if (ext_name)
		dtype->type_extension = pstrdup(ext_name);
	dtype->type_oid = tcache->type_id;
    dtype->type_flags = extra_flags;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
    dtype->type_name = get_type_name(tcache->type_id, false);
	dtype->type_hashfunc = NULL; //devtype_composite_hash;
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
	dtype->type_oid = tcache->type_id;
	dtype->type_flags = elem->type_flags;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
	dtype->type_name = psprintf("%s[]", elem->type_name);
	dtype->type_hashfunc = NULL; //devtype_array_hash;
	/* type equality functions */
	dtype->type_eqfunc = get_opcode(tcache->eq_opr);
	dtype->type_cmpfunc = tcache->cmp_proc;

	MemoryContextSwitchTo(oldcxt);

	return dtype;
}

devtype_info *
pgstrom_devtype_lookup(Oid type_oid)
{
	devtype_info   *dtype;
	uint32_t		hash;
	uint32_t		index;
	dlist_iter		iter;
	const char	   *ext_name;
	TypeCacheEntry *tcache;

	hash = GetSysCacheHashValue(TYPEOID, ObjectIdGetDatum(type_oid), 0, 0, 0);
	index = hash % DEVTYPE_INFO_NSLOTS;
	dlist_foreach (iter, &devtype_info_slot[index])
	{
		dtype = dlist_container(devtype_info, chain, iter.cur);

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
	if (OidIsValid(tcache->typrelid))
	{
		/* composite type */
		dtype = build_composite_devtype_info(tcache, ext_name);
	}
	else if (OidIsValid(tcache->typelem) && tcache->typlen == -1)
	{
		/* array type */
		dtype = build_array_devtype_info(tcache, ext_name);
	}
	else
	{
		/* base type */
		dtype = build_basic_devtype_info(tcache, ext_name);
	}

	/* make a negative entry, if not device executable */
	if (!dtype)
	{
		dtype = MemoryContextAllocZero(devinfo_memcxt,
									   sizeof(devtype_info));
		dtype->type_oid = type_oid;
		dtype->type_is_negative = true;
	}
	dtype->hash = hash;
	dlist_push_head(&devtype_info_slot[index], &dtype->chain);
found:
	if (dtype->type_is_negative)
		return NULL;
	return dtype;
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
		NumericDigit   *digits = NUMERIC_DIGITS(nc);
		int				weight = NUMERIC_WEIGHT(nc) + 1;
		int				i, ndigits = NUMERIC_NDIGITS(nc, len);
		int128_t		value = 0;

		for (i=0; i < ndigits; i++)
		{
			NumericDigit dig = digits[i];

			value = value * PG_NBASE + dig;
			if (value < 0)
				elog(ERROR, "numeric value is out of range");
		}
		if (NUMERIC_SIGN(nc) == NUMERIC_NEG)
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

/*
 * Built-in device functions/operators
 */
#define FUNC_OPCODE(SQLNAME,FN_ARGS,FN_FLAGS,DEVNAME,EXTENSION)	\
	{ #SQLNAME, #FN_ARGS, FN_FLAGS, FuncOpCode__##DEVNAME, EXTENSION },
static struct {
	const char	   *func_name;
	const char	   *func_args;
	uint32_t		func_flags;
	FuncOpCode		func_opcode;
	const char	   *func_extension;
} devfunc_catalog[] = {
#include "xpu_opcodes.h"
	{NULL,NULL,0,FuncOpCode__Invalid,NULL}
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
			dfunc->func_code = devfunc_catalog[i].func_opcode;
			if (fextension)
				dfunc->func_extension = pstrdup(fextension);
			dfunc->func_name = pstrdup(fname);
			dfunc->func_oid = func_oid;
			dfunc->func_rettype = dtype_rettype;
			dfunc->func_flags = devfunc_catalog[i].func_flags;
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
	dlist_iter		iter;
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
	dlist_foreach (iter, &devfunc_info_slot[i])
	{
		dfunc = dlist_container(devfunc_info, chain, iter.cur);
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
	dlist_push_head(&devfunc_info_slot[i], &dfunc->chain);
found:
	if (dfunc->func_is_negative)
		return NULL;
	if (OidIsValid(func_collid) && !lc_collate_is_c(func_collid) &&
		(dfunc->func_flags & DEVFUNC__LOCALE_AWARE) != 0)
		return NULL;
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

/* ----------------------------------------------------------------
 *
 * xPU pseudo code generator
 *
 * ----------------------------------------------------------------
 */
typedef struct
{
	StringInfo	buf;
	List	   *used_params;
	List	   *used_vars;
	uint32_t	extra_flags;
	uint32_t	extra_bufsz;
	int			num_rels;
	List	   *rel_tlist[1];
} codegen_context;

#define __Elog(fmt,...)							\
	do {										\
		errmsg("%s:%d" fmt,						\
			   basename(__FILE__), __LINE__,	\
			   ##__VA_ARGS__);					\
		return -1;								\
	} while(0)

static int	codegen_expression_walker(codegen_context *context, Expr *expr);

static void
__enlargeStringInfoAligned(StringInfo buf, int needed)
{
	static uint64_t __zero = 0;

	if (MAXALIGN(buf->len) != buf->len)
		appendBinaryStringInfo(buf, (char *)&__zero,
							   MAXALIGN(buf->len) - buf->len);
	Assert(MAXALIGN(buf->len) == buf->len);
	enlargeStringInfo(buf, needed);
}

static int
codegen_const_expression(codegen_context *context, Const *con)
{
	StringInfo		buf = context->buf;
	devtype_info   *dtype;
	int				width;

	dtype = pgstrom_devtype_lookup(con->consttype);
	if (!dtype)
		__Elog("type %s is not device supported",
			   format_type_be(con->consttype));
	if (con->constisnull)
		width = 0;
	else if (con->constlen > 0)
		width = con->constlen;
	else if (con->constlen == -1)
		width = VARSIZE_ANY(con->constvalue);
	else
		__Elog("unexpected type length: %d", con->constlen);

	if (buf)
	{
		kern_expression *kexp;
		int			needed = SizeOfKernExprConst(width);

		__enlargeStringInfoAligned(buf, needed);
		kexp = (kern_expression *)(buf->data + buf->len);
		memset(kexp, 0, SizeOfKernExprConst(0));
		kexp->opcode = FuncOpCode__ConstExpr;
		kexp->nargs = 0;
		kexp->rettype = dtype->type_code;
		kexp->u.c.const_isnull = con->constisnull;
		if (con->constbyval)
			memcpy(kexp->u.c.const_value,
				   &con->constvalue, width);
		else
			memcpy(kexp->u.c.const_value,
				   DatumGetPointer(con->constvalue), width);
		SET_VARSIZE(kexp, needed);

		buf->len += needed;
	}
	return width;
}

static int
codegen_param_expression(codegen_context *context, Param *param)
{
	StringInfo		buf = context->buf;
	devtype_info   *dtype;
	int				width;

	if (param->paramkind != PARAM_EXTERN)
		__Elog("Only PARAM_EXTERN is supported on device: %d",
			   (int)param->paramkind);
	dtype = pgstrom_devtype_lookup(param->paramtype);
	if (!dtype)
		__Elog("type %s is not device supported",
			   format_type_be(param->paramtype));
	if (dtype->type_length > 0)
		width = dtype->type_length;
	else if (dtype->type_length == -1)
		width = type_maximum_size(param->paramtype,
								  param->paramtypmod);
	else
		__Elog("unexpected type length: %d", dtype->type_length);

	if (buf)
	{
		kern_expression *kexp;
		int			needed = SizeOfKernExprParam;

		__enlargeStringInfoAligned(buf, needed);
		kexp = (kern_expression *)(buf->data + buf->len);
		memset(kexp, 0, SizeOfKernExprParam);
		kexp->opcode = FuncOpCode__ParamExpr;
        kexp->nargs = 0;
		kexp->rettype = dtype->type_code;
		kexp->u.p.param_id = param->paramid;
		SET_VARSIZE(kexp, needed);

		buf->len += needed;
	}
	context->used_params = list_append_unique_int(context->used_params,
												  param->paramid);
	return width;
}

static int
codegen_var_expression(codegen_context *context, Var *var)
{
	StringInfo		buf = context->buf;
	devtype_info   *dtype;
	int				width;

	dtype = pgstrom_devtype_lookup(var->vartype);
	if (!dtype)
		__Elog("type %s is not device supported",
			   format_type_be(var->vartype));
	if (dtype->type_length > 0)
		width = dtype->type_length;
	else if (dtype->type_length == -1)
		width = type_maximum_size(var->vartype,
								  var->vartypmod);
	else
		__Elog("unexpected type length: %d", dtype->type_length);

	if (buf)
	{
		kern_expression *kexp;
		int			needed = SizeOfKernExprVar;

		__enlargeStringInfoAligned(buf, needed);
		kexp = (kern_expression *)(buf->data + buf->len);
		memset(kexp, 0, offsetof(kern_expression, u.v));
		kexp->opcode = FuncOpCode__VarExpr;
		kexp->nargs = 0;
		kexp->rettype = dtype->type_code;
		if (context->num_rels == 0)
		{
			kexp->u.v.var_depth = 0;
			kexp->u.v.var_resno = var->varattno;
		}
		else
		{
			ListCell *lc = NULL;
			int		depth = -1;
			int		resno = -1;

			for (depth=0; depth < context->num_rels; depth++)
			{
				List   *tlist = context->rel_tlist[depth];

				resno = 1;
				foreach (lc, tlist)
				{
					Var	   *curr = lfirst(lc);

					if (IsA(curr, Var) &&
						var->varno == curr->varno &&
						var->varattno == curr->varattno)
					{
						Assert(var->vartype == curr->vartype &&
							   var->vartypmod == curr->vartypmod &&
							   var->varcollid == curr->varcollid);
						goto found;
					}
					resno++;
				}
			}
			__Elog("var-node was not found at the targe-list: %s",
				   nodeToString(var));
		found:
			kexp->u.v.var_depth = depth;
			kexp->u.v.var_resno = resno;
		}
		SET_VARSIZE(kexp, needed);
	}
	context->used_vars = list_append_unique(context->used_vars, var);

	return width;
}

static int
__codegen_func_expression(codegen_context *context,
						  Oid func_oid, List *func_args, Oid func_collid)
{
	StringInfo		buf = context->buf;
	int				kexp_off = -1;
	devfunc_info   *dfunc;
	devtype_info   *dtype;
	ListCell	   *lc;
	int				width;

	dfunc = pgstrom_devfunc_lookup(func_oid, func_args, func_collid);
	if (!dfunc)
		__Elog("function %s is not supported",
			   format_procedure(func_oid));
	dtype = dfunc->func_rettype;
	if (dtype->type_length > 0)
		width = dtype->type_length;
	else if (dtype->type_length == -1)
		width = type_maximum_size(dtype->type_oid, -1);
	else
		__Elog("unexpected type length: %d", dtype->type_length);

	if (buf)
	{
		kern_expression *kexp;
		int			needed = SizeOfKernExpr(0);

		__enlargeStringInfoAligned(buf, needed);
		kexp_off = buf->len;
		kexp = (kern_expression *)(buf->data + buf->len);
		memset(kexp, 0, needed);
		kexp->opcode = dfunc->func_code;
		kexp->nargs = list_length(func_args);
		kexp->rettype = dtype->type_code;
		buf->len += needed;
	}
	foreach (lc, func_args)
	{
		if (codegen_expression_walker(context, (Expr *)lfirst(lc)) < 0)
			return -1;
	}
	if (buf)
		SET_VARSIZE(buf->data + kexp_off, buf->len - kexp_off);
	return width;
}

static int
codegen_func_expression(codegen_context *context, FuncExpr *func)
{
	return __codegen_func_expression(context,
									 func->funcid,
									 func->args,
									 func->funccollid);
}

static int
codegen_oper_expression(codegen_context *context, OpExpr *oper)
{
	return __codegen_func_expression(context,
									 get_opcode(oper->opno),
									 oper->args,
									 oper->opcollid);
}

static int
codegen_bool_expression(codegen_context *context, BoolExpr *b)
{
	StringInfo		buf = context->buf;
	int				kexp_off = -1;
	ListCell	   *lc;
	FuncOpCode		opcode;

	switch (b->boolop)
	{
		case AND_EXPR:
			if (list_length(b->args) < 2)
				__Elog("BoolExpr(AND) must have 2 or more arguments");
			opcode = FuncOpCode__BoolExpr_And;
			break;
		case OR_EXPR:
			if (list_length(b->args) < 2)
				__Elog("BoolExpr(OR) must have 2 or more arguments");
			opcode = FuncOpCode__BoolExpr_Or;
			break;
		case NOT_EXPR:
			if (list_length(b->args) != 1)
				__Elog("BoolExpr(OR) must not have multiple arguments");
			opcode = FuncOpCode__BoolExpr_Not;
			break;
		default:
			__Elog("BoolExpr has unknown bool operation (%d)", (int)b->boolop);
	}

	if (buf)
	{
		kern_expression *kexp;
		int			needed = SizeOfKernExpr(0);

		__enlargeStringInfoAligned(buf, needed);
		kexp_off = buf->len;
		kexp = (kern_expression *)(buf->data + kexp_off);
		memset(kexp, 0, needed);
		kexp->opcode = opcode;
		kexp->nargs = list_length(b->args);
		kexp->rettype = TypeOpCode__bool;
		buf->len += needed;
	}
	foreach (lc, b->args)
	{
		if (codegen_expression_walker(context, (Expr *)lfirst(lc)) < 0)
			return -1;
	}
	if (buf)
		SET_VARSIZE(buf->data + kexp_off, buf->len - kexp_off);
	return sizeof(bool);
}

static int
codegen_nulltest_expression(codegen_context *context, NullTest *nt)
{
	StringInfo		buf = context->buf;
	int				kexp_off = -1;
	FuncOpCode		opcode;

	switch (nt->nulltesttype)
	{
		case IS_NULL:
			opcode = FuncOpCode__NullTestExpr_IsNull;
			break;
		case IS_NOT_NULL:
			opcode = FuncOpCode__NullTestExpr_IsNotNull;
			break;
		default:
			__Elog("NullTest has unknown NullTestType (%d)", (int)nt->nulltesttype);
	}

	if (buf)
	{
		kern_expression *kexp;
		int			needed = SizeOfKernExpr(0);

		__enlargeStringInfoAligned(buf, needed);
		kexp_off = buf->len;
		kexp = (kern_expression *)(buf->data + kexp_off);
		memset(kexp, 0, needed);
		kexp->opcode = opcode;
		kexp->nargs = 1;
		kexp->rettype = TypeOpCode__bool;
		buf->len += needed;
	}
	if (codegen_expression_walker(context, nt->arg) < 0)
		return -1;
	if (buf)
		SET_VARSIZE(buf->data + kexp_off, buf->len - kexp_off);
	return sizeof(bool);
}

static int
codegen_booleantest_expression(codegen_context *context, BooleanTest *bt)
{
	StringInfo		buf = context->buf;
	int				kexp_off = -1;
	FuncOpCode		opcode;

	switch (bt->booltesttype)
	{
		case IS_TRUE:
			opcode = FuncOpCode__BoolTestExpr_IsTrue;
			break;
		case IS_NOT_TRUE:
			opcode = FuncOpCode__BoolTestExpr_IsNotTrue;
			break;
		case IS_FALSE:
			opcode = FuncOpCode__BoolTestExpr_IsFalse;
			break;
		case IS_NOT_FALSE:
			opcode = FuncOpCode__BoolTestExpr_IsNotFalse;
			break;
		case IS_UNKNOWN:
			opcode = FuncOpCode__BoolTestExpr_IsUnknown;
			break;
		case IS_NOT_UNKNOWN:
			opcode = FuncOpCode__BoolTestExpr_IsNotUnknown;
			break;
		default:
			__Elog("BooleanTest has unknown BoolTestType (%d)",
				   (int)bt->booltesttype);
	}

	if (buf)
	{
		kern_expression *kexp;
		int			needed = SizeOfKernExpr(0);

		__enlargeStringInfoAligned(buf, needed);
		kexp_off = buf->len;
		kexp = (kern_expression *)(buf->data + kexp_off);
		memset(kexp, 0, needed);
		kexp->opcode = opcode;
		kexp->nargs = 1;
		kexp->rettype = TypeOpCode__bool;
		buf->len += needed;
	}
	if (codegen_expression_walker(context, bt->arg) < 0)
		return -1;
	if (buf)
		SET_VARSIZE(buf->data + kexp_off, buf->len - kexp_off);
	return sizeof(bool);
}

static int
codegen_expression_walker(codegen_context *context, Expr *expr)
{
	if (!expr)
		return 0;
	switch (nodeTag(expr))
	{
		case T_Const:
			return codegen_const_expression(context, (Const *)expr);
		case T_Param:
			return codegen_param_expression(context, (Param *)expr);
		case T_Var:
			return codegen_var_expression(context, (Var *)expr);
		case T_FuncExpr:
			return codegen_func_expression(context, (FuncExpr *)expr);
		case T_OpExpr:
		case T_DistinctExpr:
			return codegen_oper_expression(context, (OpExpr *)expr);
		case T_BoolExpr:
			return codegen_bool_expression(context, (BoolExpr *)expr);
		case T_NullTest:
			return codegen_nulltest_expression(context, (NullTest *)expr);
		case T_BooleanTest:
			return codegen_booleantest_expression(context, (BooleanTest *)expr);
		case T_CoalesceExpr:
		case T_MinMaxExpr:
		case T_RelabelType:
		case T_CoerceViaIO:
		case T_CoerceToDomain:
		case T_CaseExpr:
		case T_CaseTestExpr:
		case T_ScalarArrayOpExpr:
		default:
			__Elog("not a supported expression type: %d", (int)nodeTag(expr));
	}
	return -1;
}
#undef __Elog

Const *
pgstrom_codegen_expression(Expr *expr,
						   List **p_used_params,
						   List **p_used_vars,
						   uint32_t *p_extra_flags,
						   uint32_t *p_extra_bufsz,
						   int num_rels,
						   List **rel_tlist)
{
	codegen_context	   *context;
	StringInfoData		buf;

	context = alloca(offsetof(codegen_context, rel_tlist[num_rels]));
	memset(context, 0, offsetof(codegen_context, rel_tlist));

	initStringInfo(&buf);
	context->buf = &buf;
	if (num_rels > 0)
		memcpy(context->rel_tlist, rel_tlist, sizeof(List *) * num_rels);
	context->num_rels = num_rels;

	if (IsA(expr, List) && list_length((List *)expr) > 1)
		expr = make_andclause((List *)expr);
	if (codegen_expression_walker(context, expr) < 0)
	{
		/* errmsg and errdetail shall be already set */
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errdetail("problematic expression: %s", nodeToString(expr))));
		return NULL;
	}
	if (p_used_params)
		*p_used_params = list_concat_unique_int(*p_used_params,
												context->used_params);
	if (p_used_vars)
		*p_used_vars = list_concat_unique(*p_used_vars,
										  context->used_vars);
	if (p_extra_flags)
		*p_extra_flags |= context->extra_flags;
	if (p_extra_bufsz)
		*p_extra_bufsz += context->extra_bufsz;

	return makeConst(BYTEAOID,
					 -1,
					 InvalidOid,
					 -1,
					 PointerGetDatum(buf.data),
					 false,
					 false);
}

static void
pgstrom_devcache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	int		i;

	MemoryContextReset(devinfo_memcxt);
	for (i=0; i < DEVTYPE_INFO_NSLOTS; i++)
		dlist_init(&devtype_info_slot[i]);
	for (i=0; i < DEVFUNC_INFO_NSLOTS; i++)
		dlist_init(&devfunc_info_slot[i]);
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
