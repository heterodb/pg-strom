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
static List	   *devtype_info_slot[DEVTYPE_INFO_NSLOTS];
static List	   *devtype_code_slot[DEVTYPE_INFO_NSLOTS];	/* by TypeOpCode */
static List	   *devfunc_info_slot[DEVFUNC_INFO_NSLOTS];
static List	   *devfunc_code_slot[DEVFUNC_INFO_NSLOTS];	/* by FuncOpCode */

#define TYPE_OPCODE(NAME,OID,EXTENSION)			\
	static uint32_t devtype_##NAME##_hash(bool isnull, Datum value);
#include "xpu_opcodes.h"

#define TYPE_OPCODE(NAME,OID,EXTENSION)		\
	{ EXTENSION, #NAME,						\
	  TypeOpCode__##NAME, DEVKERN__ANY,		\
	  devtype_##NAME##_hash, sizeof(xpu_##NAME##_t), InvalidOid},
static struct {
	const char	   *type_extension;
	const char	   *type_name;
	TypeOpCode		type_code;
	uint32_t		type_flags;
	devtype_hashfunc_f type_hashfunc;
	int				type_sizeof;
	Oid				type_alias;
} devtype_catalog[] = {
#include "xpu_opcodes.h"
	/* alias device data types */
	{NULL, "varchar", TypeOpCode__text, DEVKERN__ANY, devtype_text_hash, TEXTOID},
	{NULL, "cidr",    TypeOpCode__inet, DEVKERN__ANY, devtype_inet_hash, INETOID},
	{NULL, NULL, TypeOpCode__Invalid, 0, NULL, InvalidOid}
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
			MemoryContext oldcxt;
			Oid		__type_alias = devtype_catalog[i].type_alias;

			/* check feasibility of type alias */
			if (OidIsValid(__type_alias))
			{
				char		castmethod;

				htup = SearchSysCache2(CASTSOURCETARGET,
									   ObjectIdGetDatum(tcache->type_id),
									   ObjectIdGetDatum(__type_alias));
				if (!HeapTupleIsValid(htup))
					elog(ERROR, "binary type cast %s to %s is not defined",
						 format_type_be(tcache->type_id),
						 format_type_be(__type_alias));
				castmethod = ((Form_pg_cast)GETSTRUCT(htup))->castmethod;
				if (castmethod != COERCION_METHOD_BINARY)
					elog(ERROR, "type cast %s to %s is not binary compatible (%c)",
						 format_type_be(tcache->type_id),
						 format_type_be(__type_alias), castmethod);
				ReleaseSysCache(htup);
				/* use type name of the alias */
				htup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(__type_alias));
				if (!HeapTupleIsValid(htup))
					elog(ERROR, "cache lookup failed for type %u", __type_alias);
				__type = (Form_pg_type) GETSTRUCT(htup);
				strcpy(type_name, NameStr(__type->typname));
				ReleaseSysCache(htup);
			}
			oldcxt = MemoryContextSwitchTo(devinfo_memcxt);
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
			dtype->type_extension = (ext_name ? pstrdup(ext_name) : NULL);
			dtype->type_sizeof = devtype_catalog[i].type_sizeof;
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
	dtype->type_code = TypeOpCode__composite;
	dtype->type_oid = tcache->type_id;
	dtype->type_flags = extra_flags;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
	dtype->type_name = "composite";
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
	dtype->type_code = TypeOpCode__array;
	dtype->type_oid = tcache->type_id;
	dtype->type_flags = elem->type_flags;
	dtype->type_length = tcache->typlen;
	dtype->type_align = typealign_get_width(tcache->typalign);
	dtype->type_byval = tcache->typbyval;
	dtype->type_name = "array";
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
	uint32_t	required_flags;
	uint32_t	extra_flags;
	uint32_t	extra_bufsz;
	uint32_t	device_cost;
	List	   *kvars_depth;
	List	   *kvars_resno;
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

static int	codegen_expression_walker(codegen_context *context, Expr *expr,
									  bool is_projection_toplevel);
static int
codegen_const_expression(codegen_context *context, Const *con)
{
	devtype_info   *dtype;

	dtype = pgstrom_devtype_lookup(con->consttype);
	if (!dtype)
		__Elog("type %s is not device supported",
			   format_type_be(con->consttype));
	if (context->buf)
	{
		kern_expression *kexp;
		int		sz = offsetof(kern_expression, u.c.const_value);

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
		SET_VARSIZE(&kexp, sz);
		kexp->opcode = FuncOpCode__ConstExpr;
		kexp->rettype = dtype->type_code;
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
		__appendBinaryStringInfo(context->buf, kexp, sz);
	}
	return 0;
}

static int
codegen_param_expression(codegen_context *context, Param *param)
{
	kern_expression	kexp;
	devtype_info   *dtype;

	if (param->paramkind != PARAM_EXTERN)
		__Elog("Only PARAM_EXTERN is supported on device: %d",
			   (int)param->paramkind);
	dtype = pgstrom_devtype_lookup(param->paramtype);
	if (!dtype)
		__Elog("type %s is not device supported",
			   format_type_be(param->paramtype));
	if (context->buf)
	{
		memset(&kexp, 0, sizeof(kexp));
		SET_VARSIZE(&kexp, SizeOfKernExprParam);
		kexp.opcode = FuncOpCode__ParamExpr;
		kexp.rettype = dtype->type_code;
		kexp.u.p.param_id = param->paramid;
		__appendBinaryStringInfo(context->buf, &kexp,
								 SizeOfKernExprParam);
	}
	context->used_params = list_append_unique(context->used_params, param);

	return 0;
}

static int
__codegen_var_expression(codegen_context *context,
						 Oid type_oid,
						 bool is_projection_toplevel,
						 int kvar_depth,
						 int kvar_resno)
{
	devtype_info *dtype;
	ListCell   *lc1, *lc2;
	int			slot_id = 0;

	dtype = pgstrom_devtype_lookup(type_oid);
	if (!dtype && !is_projection_toplevel)
		__Elog("type %s is not device supported", format_type_be(type_oid));

	forboth (lc1, context->kvars_depth,
			 lc2, context->kvars_resno)
	{
		int		__depth = kvar_depth;
		int		__resno = kvar_resno;

		if (kvar_depth == __depth && kvar_resno == __resno)
			goto found;
		slot_id++;
	}
	context->kvars_depth = lappend_int(context->kvars_depth, kvar_depth);
	context->kvars_resno = lappend_int(context->kvars_resno, kvar_resno);
	Assert(slot_id == list_length(context->kvars_depth) &&
		   slot_id == list_length(context->kvars_resno));
found:
	if (context->buf)
	{
		kern_expression kexp;
		int16		typlen;
		bool		typbyval;
		char		typalign;

		get_typlenbyvalalign(type_oid, &typlen, &typbyval, &typalign);

		memset(&kexp, 0, sizeof(kexp));
		kexp.opcode = FuncOpCode__VarExpr;
		kexp.nargs = 0;
		kexp.rettype = (dtype ? dtype->type_code : TypeOpCode__unsupported);
		kexp.u.v.var_typlen = typlen;
		kexp.u.v.var_typbyval = typbyval;
		kexp.u.v.var_typalign = typealign_get_width(typalign);
		kexp.u.v.var_slot_id = slot_id;
		SET_VARSIZE(&kexp, SizeOfKernExprVar);
		__appendBinaryStringInfo(context->buf, &kexp, SizeOfKernExprVar);
	}
	return 0;
}

static int
codegen_var_expression(codegen_context *context, Var *var,
					   bool is_projection_toplevel)
{
	return __codegen_var_expression(context,
									var->vartype,
									is_projection_toplevel,
									0,				/* depth */
									var->varattno);	/* resno */
}

static int
__codegen_func_expression(codegen_context *context,
						  Oid func_oid, List *func_args, Oid func_collid)
{
	StringInfo		buf = context->buf;
	devfunc_info   *dfunc;
	devtype_info   *dtype;
	kern_expression	kexp;
	int				pos = -1;
	ListCell	   *lc;

	dfunc = pgstrom_devfunc_lookup(func_oid, func_args, func_collid);
	if (!dfunc)
		__Elog("function %s is not supported",
			   format_procedure(func_oid));
	dtype = dfunc->func_rettype;
	context->device_cost += dfunc->func_cost;

	memset(&kexp, 0, sizeof(kexp));
	kexp.opcode = dfunc->func_code;
	kexp.nargs = list_length(func_args);
	kexp.rettype = dtype->type_code;
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	foreach (lc, func_args)
	{
		Expr   *arg = lfirst(lc);

		if (codegen_expression_walker(context, arg, false) < 0)
			return -1;
	}
	if (buf)
		SET_VARSIZE(buf->data + pos, buf->len - pos);
	return 0;
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
	kern_expression	kexp;
	int				pos = -1;
	ListCell	   *lc;

	memset(&kexp, 0, sizeof(kexp));
	switch (b->boolop)
	{
		case AND_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_And;
			kexp.nargs = list_length(b->args);
			if (kexp.nargs < 2)
				__Elog("BoolExpr(AND) must have 2 or more arguments");
			break;
		case OR_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_Or;
			kexp.nargs = list_length(b->args);
			if (kexp.nargs < 2)
				__Elog("BoolExpr(OR) must have 2 or more arguments");
			break;
		case NOT_EXPR:
			kexp.opcode = FuncOpCode__BoolExpr_Not;
			kexp.nargs = list_length(b->args);
			if (kexp.nargs != 1)
				__Elog("BoolExpr(OR) must not have multiple arguments");
			break;
		default:
			__Elog("BoolExpr has unknown bool operation (%d)", (int)b->boolop);
	}
	kexp.rettype = TypeOpCode__bool;
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	foreach (lc, b->args)
	{
		Expr   *arg = lfirst(lc);
		if (codegen_expression_walker(context, arg, false) < 0)
			return -1;
	}
	if (buf)
		SET_VARSIZE(buf->data + pos, buf->len - pos);
	return 0;
}

static int
codegen_nulltest_expression(codegen_context *context, NullTest *nt)
{
	StringInfo		buf = context->buf;
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
	kexp.nargs = 1;
	kexp.rettype = TypeOpCode__bool;
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, nt->arg, false) < 0)
		return -1;
	if (buf)
		SET_VARSIZE(buf->data + pos, buf->len - pos);
	return 0;
}

static int
codegen_booleantest_expression(codegen_context *context, BooleanTest *bt)
{
	StringInfo		buf = context->buf;
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
	kexp.nargs = 1;
	kexp.rettype = TypeOpCode__bool;
	if (buf)
		pos = __appendBinaryStringInfo(buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, bt->arg, false) < 0)
		return -1;
	if (buf)
		SET_VARSIZE(buf->data + pos, buf->len - pos);
	return 0;
}

static int
codegen_expression_walker(codegen_context *context, Expr *expr,
						  bool is_projection_toplevel)
{
	int		depth;

	if (!expr)
		return 0;
	/*
	 * MEMO: If expression matches any of input target-entries,
	 * it shall be replaced to Var reference.
	 */
	for (depth=0; depth < context->num_rels; depth++)
	{
		List	   *tlist = context->rel_tlist[depth];
		ListCell   *lc;

		foreach (lc, tlist)
		{
			TargetEntry	   *tle = lfirst(lc);

			if (!tle->resjunk && equal(tle->expr, expr))
			{
				return __codegen_var_expression(context,
												exprType((Node *)expr),
												is_projection_toplevel,
												depth,
												tle->resno);
			}
		}
	}

	switch (nodeTag(expr))
	{
		case T_Const:
			return codegen_const_expression(context, (Const *)expr);
		case T_Param:
			return codegen_param_expression(context, (Param *)expr);
		case T_Var:
			return codegen_var_expression(context, (Var *)expr,
										  is_projection_toplevel);
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

static bytea *
attach_varloads_xpucode(codegen_context *context,
						kern_expression *kexp)
{
	StringInfoData	buf;
	kern_expression *vl;
	ListCell   *lc1, *lc2;
	size_t		sz;
	int			nloads = list_length(context->kvars_depth);
	int			i, j, k;

	sz = MAXALIGN(offsetof(kern_expression,
						   u.ld.kvars[nloads]));
	vl = alloca(sz);
	memset(vl, 0, sz);
	vl->opcode = FuncOpCode__LoadVars;
	if (kexp)
	{
		vl->nargs = 1;
		vl->rettype = kexp->rettype;
	}
	else
	{
		vl->nargs = 0;
		vl->rettype = TypeOpCode__bool;
	}
	vl->u.ld.nloads = nloads;
	i = 0;
	forboth (lc1, context->kvars_depth,
			 lc2, context->kvars_resno)
	{
		int		__depth = lfirst_int(lc1);
		int		__resno = lfirst_int(lc2);

		vl->u.ld.kvars[i].var_depth = __depth;
		vl->u.ld.kvars[i].var_resno = __resno;
		vl->u.ld.kvars[i].var_slot_id = i;
		i++;
	}
	/* sort by depth/resno */
	for (i=0; i < nloads; i++)
	{
		k = i;
		for (j=i+1; j < nloads; j++)
		{
			if (vl->u.ld.kvars[j].var_depth < vl->u.ld.kvars[k].var_depth ||
				(vl->u.ld.kvars[j].var_depth == vl->u.ld.kvars[k].var_depth &&
				 vl->u.ld.kvars[j].var_resno < vl->u.ld.kvars[k].var_resno))
				k = j;
		}
		if (i != k)
		{
			int16_t		__depth = vl->u.ld.kvars[i].var_depth;
			int16_t		__resno = vl->u.ld.kvars[i].var_resno;
			uint32_t	__slot_id = vl->u.ld.kvars[i].var_slot_id;

			vl->u.ld.kvars[i].var_depth   = vl->u.ld.kvars[k].var_depth;
			vl->u.ld.kvars[i].var_resno   = vl->u.ld.kvars[k].var_resno;
			vl->u.ld.kvars[i].var_slot_id = vl->u.ld.kvars[k].var_slot_id;
			vl->u.ld.kvars[k].var_depth   = __depth;
			vl->u.ld.kvars[k].var_resno   = __resno;
			vl->u.ld.kvars[k].var_slot_id = __slot_id;
		}
	}
	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (const char *)vl, sz);
	if (kexp)
		appendBinaryStringInfo(&buf, (const char *)kexp, VARSIZE(kexp));
	SET_VARSIZE(&buf, buf.len);

	return (bytea *)buf.data;
}

void
pgstrom_build_xpucode(bytea **p_xpucode,
					  Expr *expr,
					  int num_rels,
					  List **rel_tlist,
					  uint32_t *p_extra_flags,
					  uint32_t *p_extra_bufsz,
					  uint32_t *p_kvars_nslots,
					  List **p_used_params)
{
	codegen_context	   *context;
	StringInfoData		buf;

	context = alloca(offsetof(codegen_context, rel_tlist[num_rels]));
	memset(context, 0, offsetof(codegen_context, rel_tlist));

	initStringInfo(&buf);
	context->buf = &buf;
	if (p_extra_flags)
		context->extra_flags = *p_extra_flags;
	if (p_extra_bufsz)
		context->extra_bufsz = *p_extra_bufsz;
	if (p_used_params)
		context->used_params = *p_used_params;
	if (num_rels > 0)
		memcpy(context->rel_tlist, rel_tlist, sizeof(List *) * num_rels);
	context->num_rels = num_rels;

	if (IsA(expr, List) && list_length((List *)expr) > 1)
		expr = make_andclause((List *)expr);
	if (codegen_expression_walker(context, expr, false) < 0)
	{
		/* errmsg shall be already set */
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errdetail("problematic expression: %s", nodeToString(expr))));
	}
	SET_VARSIZE(buf.data, buf.len);
	/* attach VarLoads operation */
	*p_xpucode = attach_varloads_xpucode(context, (kern_expression *)buf.data);

	if (p_used_params)
		*p_used_params = context->used_params;
	if (p_extra_flags)
		*p_extra_flags = context->extra_flags;
	if (p_extra_bufsz)
		*p_extra_bufsz = context->extra_bufsz;
	if (p_kvars_nslots)
		*p_kvars_nslots = list_length(context->kvars_depth);
	pfree(buf.data);
}

/*
 * pgstrom_build_projection
 */
void
pgstrom_build_projection(bytea **p_xpucode_proj_prep,
						 bytea **p_xpucode_proj_exec,
						 List *tlist_dev,
						 int num_rels,
						 List **rel_tlist,
						 uint32_t *p_extra_flags,
						 uint32_t *p_extra_bufsz,
						 uint32_t *p_kvars_nslots,
						 List **p_used_params)
{
	codegen_context	   *context;
	StringInfoData		buf;
	kern_expression		kexp;
	ListCell		   *lc;

	context = alloca(offsetof(codegen_context, rel_tlist[num_rels]));
	memset(context, 0, offsetof(codegen_context, rel_tlist));
	initStringInfo(&buf);
	context->buf = &buf;
	if (p_extra_flags)
		context->extra_flags = *p_extra_flags;
	if (p_extra_bufsz)
		context->extra_bufsz = *p_extra_bufsz;
	if (p_used_params)
		context->used_params = *p_used_params;
	if (num_rels > 0)
		memcpy(context->rel_tlist, rel_tlist, sizeof(List *) * num_rels);
	context->num_rels = num_rels;

	/* FuncOp__Projection */
	memset(&kexp, 0, sizeof(kexp));
	kexp.opcode = FuncOpCode__Projection;
	kexp.nargs = list_length(tlist_dev);
	kexp.rettype = TypeOpCode__record;
	appendBinaryStringInfo(&buf, (char *)&kexp,
						   offsetof(kern_expression, u.data));
	foreach (lc, tlist_dev)
	{
		TargetEntry *tle = lfirst(lc);

		if (codegen_expression_walker(context, tle->expr, true) < 0)
		{
			/* errmsg shall be already set */
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errdetail("problematic expression: %s",
							   nodeToString(tle->expr))));
		}
	}
	SET_VARSIZE(buf.data, buf.len);
	*p_xpucode_proj_exec = (bytea *)buf.data;

	/* FuncOpCode__LoadVars */
	*p_xpucode_proj_prep = attach_varloads_xpucode(context, NULL);

	if (p_extra_flags)
		*p_extra_flags = context->extra_flags;
	if (p_extra_bufsz)
		*p_extra_bufsz = context->extra_bufsz;
	if (p_kvars_nslots)
		*p_kvars_nslots = list_length(context->kvars_depth);
	if (p_used_params)
		*p_used_params = context->used_params;
}

/*
 * pgstrom_gpu_expression
 *
 * checker whether the supplied expression is executable on GPU devices.
 */
bool
pgstrom_gpu_expression(Expr *expr,
					   int num_rels,
					   List **rel_tlist,
					   int *p_devcost)
{
	codegen_context	   *context;

	context = alloca(offsetof(codegen_context, rel_tlist[num_rels]));
	memset(&context, 0, offsetof(codegen_context, rel_tlist[num_rels]));
	if (num_rels > 0)
		memcpy(context->rel_tlist, rel_tlist, sizeof(List *) * num_rels);
	context->num_rels = num_rels;

	if (IsA(expr, List) && list_length((List *)expr) > 0)
		expr = make_andclause((List *)expr);
	if (codegen_expression_walker(context, expr, false) < 0)
		return false;
	if (p_devcost)
		*p_devcost = context->device_cost;
	return true;
}

/*
 * pgstrom_xpucode_to_string
 *
 * transforms xPU code to human readable form.
 */
static void
__xpucode_to_cstring(StringInfo buf, const kern_expression *kexp)
{
	devtype_info   *dtype = devtype_lookup_by_opcode(kexp->rettype);
	devfunc_info   *dfunc = devfunc_lookup_by_opcode(kexp->opcode);
	const char	   *label;
	const char	   *extra = NULL;
	const char	   *pos = kexp->u.data;
	StringInfoData	temp;
	int				i;

	switch (kexp->opcode)
	{
		case FuncOpCode__ConstExpr:
			if (kexp->u.c.const_isnull)
			{
				appendStringInfo(buf, "{Const(%s): value=null}",
								 !dtype ? "???" : dtype->type_name);
			}
			else
			{
				int16	type_len;
				bool	type_byval;
				char	type_align;
				char	type_delim;
				Oid		type_ioparam;
				Oid		type_outfunc;
				Datum	datum;
				Datum	label;

				get_type_io_data(kexp->u.c.const_type,
								 IOFunc_output,
								 &type_len,
								 &type_byval,
								 &type_align,
								 &type_delim,
								 &type_ioparam,
								 &type_outfunc);
				if (!type_byval)
					datum = PointerGetDatum(kexp->u.c.const_value);
				else
					memcpy(&datum, kexp->u.c.const_value, type_len);
				label = OidFunctionCall1(type_outfunc, datum);
				appendStringInfo(buf, "{Const: value='%s'}",
								 DatumGetCString(label));
			}
			return;

		case FuncOpCode__ParamExpr:
			appendStringInfo(buf, "{Param(%s): param_id=%u}",
							 !dtype ? "???" : dtype->type_name,
							 kexp->u.p.param_id);
			return;

		case FuncOpCode__VarExpr:
			appendStringInfo(buf, "{Var(%s): slot_id=%d}",
							 !dtype ? "???" : dtype->type_name,
							 kexp->u.v.var_slot_id);
			return;

		case FuncOpCode__BoolExpr_And:
			label = "Bool::AND";
			break;
		case FuncOpCode__BoolExpr_Or:
			label = "Bool::OR";
			break;
		case FuncOpCode__BoolExpr_Not:
			label = "Bool::NOT";
			break;
		case FuncOpCode__NullTestExpr_IsNull:
			label = "IsNull";
			break;
		case FuncOpCode__NullTestExpr_IsNotNull:
			label = "IsNotNull";
			break;
		case FuncOpCode__BoolTestExpr_IsTrue:
			label = "BoolTest::IsTrue";
			break;
		case FuncOpCode__BoolTestExpr_IsNotTrue:
			label = "BoolTest::IsNotTrue";
			break;
		case FuncOpCode__BoolTestExpr_IsFalse:
			label = "BoolTest::IsFalse";
			break;
		case FuncOpCode__BoolTestExpr_IsNotFalse:
			label = "BoolTest::IsNotFalse";
			break;
		case FuncOpCode__BoolTestExpr_IsUnknown:
			label = "BoolTest::IsUnknown";
			break;
		case FuncOpCode__BoolTestExpr_IsNotUnknown:
			label = "BoolTest::IsNotUnknown";
			break;
		case FuncOpCode__Projection:
			label = "Projection";
			break;
		case FuncOpCode__LoadVars:
			label = "LoadVars";
			initStringInfo(&temp);
			appendStringInfo(&temp, "[");
			for (i=0; i < kexp->u.ld.nloads; i++)
			{
				int16_t		__depth = kexp->u.ld.kvars[i].var_depth;
				int16_t		__resno = kexp->u.ld.kvars[i].var_resno;
				uint32_t	__slot_id = kexp->u.ld.kvars[i].var_slot_id;

				appendStringInfo(&temp, "%s(slot_id=%u, depth=%d, resno=%d)",
								 i > 0 ? ", " : "",
								 __slot_id,
								 __depth,
								 __resno);
			}
			appendStringInfo(&temp, "]");
			extra = temp.data;
			break;
		default:
			if (dfunc)
				label = psprintf("Func::%s", dfunc->func_name);
			else
				label = "Func::unknown";
	}
	appendStringInfo(buf, "{%s(%s) %sargs=[",
					 label,
					 !dtype ? "???" : dtype->type_name,
					 !extra ? "" : extra);
	for (i=0; i < kexp->nargs; i++)
	{
		const kern_expression *arg = (const kern_expression *)pos;

		if (VARSIZE(kexp) < (pos + VARSIZE(arg)) - (char *)kexp)
			elog(ERROR, "XpuCode looks corrupted: kexp (sz=%u, arg=[%lu...%lu])",
				 VARSIZE(kexp),
				 (pos - kexp->u.data),
				 (pos - kexp->u.data) + VARSIZE(arg));
		__xpucode_to_cstring(buf, arg);
		pos += MAXALIGN(VARSIZE(arg));
	}
	appendStringInfoString(buf, "]}");
}

char *
pgstrom_xpucode_to_string(bytea *xpu_code)
{
	StringInfoData buf;

	initStringInfo(&buf);
	__xpucode_to_cstring(&buf, (const kern_expression *)xpu_code);

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
