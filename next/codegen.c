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
	{NULL, "varchar", TypeOpCode__text, DEVKERN__ANY,
	 devtype_text_hash, sizeof(xpu_text_t), TEXTOID},
	{NULL, "cidr",    TypeOpCode__inet, DEVKERN__ANY,
	 devtype_inet_hash, sizeof(xpu_inet_t), INETOID},
	{NULL, NULL, TypeOpCode__Invalid, 0, NULL, 0, InvalidOid}
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
	int			elevel;
	Expr	   *top_expr;
	List	   *used_params;
	uint32_t	required_flags;
	uint32_t	extra_flags;
	uint32_t	extra_bufsz;
	uint32_t	device_cost;
	List	   *kvars_depth;
	List	   *kvars_resno;
	List	   *input_rels_tlist;
} codegen_context;

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

static int	codegen_expression_walker(codegen_context *context, Expr *expr);

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
		int			pos, sz = 0;

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
		pos = __appendBinaryStringInfo(context->buf, kexp, sz);
		__appendKernExpMagicAndLength(context->buf, pos);
	}
	return 0;
}

static int
codegen_param_expression(codegen_context *context, Param *param)
{
	kern_expression	kexp;
	devtype_info   *dtype;
	int				pos;

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
		kexp.opcode = FuncOpCode__ParamExpr;
		kexp.exptype = dtype->type_code;
		kexp.u.p.param_id = param->paramid;
		pos = __appendBinaryStringInfo(context->buf, &kexp,
									   SizeOfKernExprParam);
		__appendKernExpMagicAndLength(context->buf, pos);
	}
	context->used_params = list_append_unique(context->used_params, param);

	return 0;
}

static int
codegen_var_expression(codegen_context *context,
					   Expr *expr,
					   int kvar_depth,
					   int kvar_resno,
					   int kvar_slot_id)
{
	Oid		type_oid = exprType((Node *)expr);
	devtype_info *dtype;

	dtype = pgstrom_devtype_lookup(type_oid);
	if (!dtype)
		__Elog("type %s is not device supported", format_type_be(type_oid));

	if (context->buf)
	{
		kern_expression kexp;
		int16		typlen;
		bool		typbyval;
		char		typalign;
		int			pos;

		get_typlenbyvalalign(type_oid,
							 &typlen,
							 &typbyval,
							 &typalign);
		memset(&kexp, 0, sizeof(kexp));
		kexp.exptype = dtype->type_code;
		kexp.opcode = FuncOpCode__VarExpr;
		kexp.u.v.var_typlen = typlen;
		kexp.u.v.var_typbyval = typbyval;
		kexp.u.v.var_typalign = typealign_get_width(typalign);
		kexp.u.v.var_slot_id = kvar_slot_id;
		pos = __appendBinaryStringInfo(context->buf, &kexp,
									   SizeOfKernExprVar);
		__appendKernExpMagicAndLength(context->buf, pos);
	}
	return 0;
}

static int
__codegen_func_expression(codegen_context *context,
						  Oid func_oid, List *func_args, Oid func_collid)
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
	kexp.opcode = dfunc->func_code;
	kexp.nr_args = list_length(func_args);
	kexp.args_offset = SizeOfKernExpr(0);
	if (context->buf)
		pos = __appendBinaryStringInfo(context->buf, &kexp, SizeOfKernExpr(0));
	foreach (lc, func_args)
	{
		Expr   *arg = lfirst(lc);

		if (codegen_expression_walker(context, arg) < 0)
			return -1;
	}
	if (context->buf)
		__appendKernExpMagicAndLength(context->buf, pos);
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
	kexp.args_offset = SizeOfKernExpr(0);
	if (context->buf)
		pos = __appendBinaryStringInfo(context->buf, &kexp, SizeOfKernExpr(0));
	foreach (lc, b->args)
	{
		Expr   *arg = lfirst(lc);

		if (codegen_expression_walker(context, arg) < 0)
			return -1;
	}
	if (context->buf)
		__appendKernExpMagicAndLength(context->buf, pos);
	return 0;
}

static int
codegen_nulltest_expression(codegen_context *context, NullTest *nt)
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
	kexp.nr_args = 1;
	kexp.args_offset = SizeOfKernExpr(0);
	if (context->buf)
		pos = __appendBinaryStringInfo(context->buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, nt->arg) < 0)
		return -1;
	if (context->buf)
		__appendKernExpMagicAndLength(context->buf, pos);
	return 0;
}

static int
codegen_booleantest_expression(codegen_context *context, BooleanTest *bt)
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
	kexp.nr_args = 1;
	kexp.args_offset = SizeOfKernExpr(0);
	if (context->buf)
		pos = __appendBinaryStringInfo(context->buf, &kexp, SizeOfKernExpr(0));
	if (codegen_expression_walker(context, bt->arg) < 0)
		return -1;
	if (context->buf)
		__appendKernExpMagicAndLength(context->buf, pos);
	return 0;
}

/*
 * is_expression_equals_tlist
 *
 * It checks whether the supplied expression exactly matches any entry of
 * the target-list. If found, it returns its depth and resno.
 */
static int
is_expression_equals_tlist(codegen_context *context,
						   Expr *expr,
						   int *p_depth,
						   int *p_resno)
{
	ListCell   *lc1, *lc2;
	int			depth = 0;
	int			resno;
	int			slot_id;

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
				goto found;
				
				*p_depth = depth;
				*p_resno = var->varattno;
				return true;
			}
		}
		else if (IsA(node, List))
		{
			List   *tlist = (List *)node;

			foreach (lc2, tlist)
			{
				TargetEntry *tle = lfirst(lc2);

				if (tle->resjunk)
					continue;
				if (equal(tle->expr, expr))
				{
					resno = tle->resno;
					goto found;
					
					 *p_depth = depth;
					 *p_resno = tle->resno;
					 return true;
				}
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
	return slot_id;
}

static int
codegen_expression_walker(codegen_context *context, Expr *expr)
{
	int		__depth;
	int		__resno;
	int		__slot_id;

	if (!expr)
		return 0;
	/* check simple var references */
	if ((__slot_id = is_expression_equals_tlist(context,
												expr,
												&__depth,
												&__resno)) >= 0)
		return codegen_var_expression(context, expr, __depth, __resno, __slot_id);

	switch (nodeTag(expr))
	{
		case T_Const:
			return codegen_const_expression(context, (Const *)expr);
		case T_Param:
			return codegen_param_expression(context, (Param *)expr);
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

static int
kern_preload_vars_comp(const void *__a, const void *__b)
{
	const kern_preload_vars_item *a = __a;
	const kern_preload_vars_item *b = __b;

	if (a->var_depth < 0 || a->var_resno < 0)
		return 1;
	if (b->var_depth < 0 || b->var_resno < 0)
		return -1;
	if (a->var_depth < b->var_depth)
		return -1;
	if (a->var_depth > b->var_depth)
		return 1;
	if (a->var_resno < b->var_resno)
		return -1;
	if (a->var_resno > b->var_resno)
		return 1;
	return 0;
}

static bytea *
attach_varloads_xpucode(codegen_context *context,
						kern_expression *karg)
{
	StringInfoData	buf;
	kern_expression *kexp;
	ListCell   *lc1, *lc2;
	int			i, nloads = list_length(context->kvars_depth);

	kexp = alloca(MAXALIGN(offsetof(kern_expression,
									u.load.kvars[nloads])));
	kexp->exptype = karg->exptype;
	kexp->opcode  = FuncOpCode__LoadVars;
	kexp->nr_args = 1;

	i = 0;
	forboth (lc1, context->kvars_depth,
			 lc2, context->kvars_resno)
	{
		kexp->u.load.kvars[i].var_depth = lfirst_int(lc1);
		kexp->u.load.kvars[i].var_resno = lfirst_int(lc2);
		kexp->u.load.kvars[i].var_slot_id = i;
		i++;
	}
	pg_qsort(kexp->u.load.kvars, nloads,
			 sizeof(kern_preload_vars_item),
			 kern_preload_vars_comp);
	for (i=0; i < nloads; i++)
	{
		if (kexp->u.load.kvars[i].var_depth < 0 ||
			kexp->u.load.kvars[i].var_resno < 0)
			break;
	}
	kexp->u.load.nloads = nloads = i;
	kexp->args_offset = MAXALIGN(offsetof(kern_expression,
										  u.load.kvars[nloads]));
	initStringInfo(&buf);
	buf.len = VARHDRSZ;
	appendBinaryStringInfo(&buf, (const char *)kexp, kexp->args_offset);
	appendBinaryStringInfo(&buf, (const char *)karg, karg->len);
	__appendKernExpMagicAndLength(&buf, VARHDRSZ);

	SET_VARSIZE(buf.data, buf.len);

	return (bytea *)buf.data;
}

void
pgstrom_build_xpucode(bytea **p_xpucode,
					  Expr *expr,
					  List *input_rels_tlist,
					  uint32_t *p_extra_flags,
					  uint32_t *p_extra_bufsz,
					  uint32_t *p_kvars_nslots,
					  List **p_used_params)
{
	codegen_context	context;
	StringInfoData	buf;

	memset(&context, 0, sizeof(context));
	initStringInfo(&buf);
	context.elevel = ERROR;
	context.top_expr = expr;
	context.buf = &buf;
	if (p_extra_flags)
		context.extra_flags = *p_extra_flags;
	if (p_used_params)
		context.used_params = *p_used_params;
	context.input_rels_tlist = input_rels_tlist;

	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = linitial((List *)expr);
		else
			expr = make_andclause((List *)expr);
	}
	codegen_expression_walker(&context, expr);
	/* attach VarLoads operation prior to the kern_expression above */
	*p_xpucode = attach_varloads_xpucode(&context, (kern_expression *)buf.data);
	pfree(buf.data);

	if (p_used_params)
		*p_used_params = context.used_params;
	if (p_extra_flags)
		*p_extra_flags = context.extra_flags;
	if (p_extra_bufsz)
		*p_extra_bufsz = Max(*p_extra_bufsz, context.extra_bufsz);
	if (p_kvars_nslots)
		*p_kvars_nslots = Max(*p_kvars_nslots, list_length(context.kvars_depth));
}

/*
 * pgstrom_build_projection
 */
void
pgstrom_build_projection(bytea **p_xpucode_proj,
						 List *tlist_dev,
						 List *input_rels_tlist,
						 uint32_t *p_extra_flags,
						 uint32_t *p_extra_bufsz,
						 uint32_t *p_kvars_nslots,
						 List **p_used_params)
{
	codegen_context	context;
	StringInfoData	arg;
	StringInfoData	buf;
	List		   *proj_prep_slots = NIL;
	List		   *proj_prep_types = NIL;
	List		   *proj_dest_slots = NIL;
	List		   *proj_dest_types = NIL;
	ListCell	   *lc1, *lc2;
	kern_expression	*kexp;
	int				nexprs;
	int				nattrs;
	int				i, sz;
	bool			meet_resjunk = false;

	/* setup Projection arguments */
	initStringInfo(&arg);
	memset(&context, 0, sizeof(context));
	context.buf = &arg;
	context.elevel = ERROR;
	if (p_extra_flags)
		context.extra_flags = *p_extra_flags;
	if (p_used_params)
		context.used_params = *p_used_params;
	context.input_rels_tlist = input_rels_tlist;

	foreach (lc1, tlist_dev)
	{
		TargetEntry	*tle = lfirst(lc1);
		devtype_info *dtype;
		TypeOpCode __type_code = TypeOpCode__Invalid;
		int		__depth;
		int		__resno;
		int		__slot_id;

		if (tle->resjunk)
		{
			meet_resjunk = true;
			continue;
		}
		else if (meet_resjunk)
			elog(ERROR, "Bug? a valid TLE after junk TLEs");

		dtype = pgstrom_devtype_lookup(exprType((Node *)tle->expr));
		if (dtype)
			__type_code = dtype->type_code;

		if ((__slot_id = is_expression_equals_tlist(&context,
													tle->expr,
													&__depth,
													&__resno)) < 0)
		{
			/* not a simple var reference. run an expression on device. */
			context.top_expr = tle->expr;
			codegen_expression_walker(&context, tle->expr);

			/* reserve a slot_id */
			__slot_id = list_length(context.kvars_depth);
			Assert(__slot_id == list_length(context.kvars_resno));
			context.kvars_depth = lappend_int(context.kvars_depth, -1);
			context.kvars_resno = lappend_int(context.kvars_resno, -1);

			proj_prep_slots = lappend_int(proj_prep_slots, __slot_id);
			proj_prep_types = lappend_int(proj_prep_types, __type_code);
		}
		proj_dest_slots = lappend_int(proj_dest_slots, __slot_id);
		proj_dest_types = lappend_int(proj_dest_types, __type_code);
	}
	nexprs = list_length(proj_prep_slots);
	nattrs = list_length(proj_dest_slots);

	/* setup kern_expression for Projection */
	sz = MAXALIGN(offsetof(kern_expression,
						   u.proj.desc[nexprs + nattrs]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->opcode  = FuncOpCode__Projection;
	kexp->nr_args = nexprs;
	kexp->args_offset = sz;
	kexp->u.proj.nexprs = nexprs;
	kexp->u.proj.nattrs = nattrs;

	i = 0;
	forboth (lc1, proj_prep_slots,
			 lc2, proj_prep_types)
	{
		kern_projection_desc *desc = &kexp->u.proj.desc[i++];
		desc->slot_id   = lfirst_int(lc1);
		desc->slot_type = lfirst_int(lc2);
	}
	Assert(i == nexprs);
	forboth (lc1, proj_dest_slots,
			 lc2, proj_dest_types)
	{
		kern_projection_desc *desc = &kexp->u.proj.desc[i++];
		desc->slot_id   = lfirst_int(lc1);
		desc->slot_type = lfirst_int(lc2);
	}
	Assert(i == nexprs + nattrs);
	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (const char *)kexp, sz);
	appendBinaryStringInfo(&buf, arg.data, arg.len);
	__appendKernExpMagicAndLength(&buf, 0);
	pfree(arg.data);

	*p_xpucode_proj = attach_varloads_xpucode(&context, (kern_expression *)buf.data);
	pfree(buf.data);

	if (p_extra_flags)
		*p_extra_flags = context.extra_flags;
	if (p_extra_bufsz)
		*p_extra_bufsz = context.extra_bufsz;
	if (p_kvars_nslots)
		*p_kvars_nslots = nexprs + nattrs;
	if (p_used_params)
		*p_used_params = context.used_params;
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
	codegen_context	context;

	memset(&context, 0, sizeof(context));
	context.elevel = DEBUG2;
	context.top_expr = expr;
	context.required_flags = DEVKERN__NVIDIA_GPU;
	context.input_rels_tlist = input_rels_tlist;

	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = linitial((List *)expr);
		else
			expr = make_andclause((List *)expr);
	}
	if (codegen_expression_walker(&context, expr) < 0)
		return false;
	if (p_devcost)
		*p_devcost = context.device_cost;
	return true;
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
	codegen_context	context;

	memset(&context, 0, sizeof(context));
	context.elevel = DEBUG2;
	context.top_expr = expr;
	context.required_flags = DEVKERN__NVIDIA_DPU;
	context.input_rels_tlist = input_rels_tlist;

	if (IsA(expr, List))
	{
		if (list_length((List *)expr) == 1)
			expr = linitial((List *)expr);
		else
			expr = make_andclause((List *)expr);
	}
	if (codegen_expression_walker(&context, expr) < 0)
		return false;
	if (p_devcost)
		*p_devcost = context.device_cost;
	return true;
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
					 List *ancestors);				/* optionsl */

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
						   List *ancestors)
{
	bool	verbose = false;
	int		i;

	Assert(kexp->nr_args == 1);
	appendStringInfo(buf, "{LoadVars:");
	if (kexp->u.load.nloads > 0)
		appendStringInfo(buf, " kvars=[");

	if (css)
	{
		CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
		verbose = (cscan->custom_plans != NIL);
	}

	for (i=0; i < kexp->u.load.nloads; i++)
	{
		const kern_preload_vars_item *vitem = &kexp->u.load.kvars[i];

		if (i > 0)
			appendStringInfo(buf, ", ");
		if (!css)
		{
			appendStringInfo(buf, "(slot_id=%u, depth=%d, resno=%d)",
							 vitem->var_slot_id,
							 vitem->var_depth,
							 vitem->var_resno);
		}
		else if (vitem->var_depth == 0)
		{
			TupleDesc	tupdesc = RelationGetDescr(css->ss.ss_currentRelation);
			Form_pg_attribute attr = TupleDescAttr(tupdesc, vitem->var_resno - 1);
			CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
			List	   *dcontext;
			Var		   *kvar;

			dcontext = set_deparse_context_plan(es->deparse_cxt,
												(Plan *)cscan, ancestors);
			kvar = makeVar(cscan->scan.scanrelid,
						   attr->attnum,
						   attr->atttypid,
						   attr->atttypmod,
						   attr->attcollation, 0);
			appendStringInfo(buf, "%u:%s",
							 vitem->var_slot_id,
							 deparse_expression((Node *)kvar,
												dcontext,
												verbose, false));
			pfree(kvar);
		}
		else
		{
			CustomScan *cscan = (CustomScan *)css->ss.ps.plan;
			Plan	   *plan;
			List	   *dcontext;
			TargetEntry *tle;

			plan = list_nth(cscan->custom_plans, vitem->var_depth - 1);
			dcontext = set_deparse_context_plan(es->deparse_cxt,
												plan, ancestors);
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
__xpucode_projection_cstring(StringInfo buf,
							 const kern_expression *kexp,
							 const CustomScanState *css,	/* optional */
							 ExplainState *es,				/* optional */
							 List *ancestors)
{
	int		i, nexprs = kexp->u.proj.nexprs;

	appendStringInfo(buf, "{Projection record=(");
	for (i=0; i < kexp->u.proj.nattrs; i++)
	{
		const kern_projection_desc *desc = &kexp->u.proj.desc[nexprs + i];
		if (i > 0)
			appendStringInfoChar(buf, ',');
		appendStringInfo(buf, "%d", desc->slot_id);
	}
	appendStringInfo(buf, ")");

	if (kexp->nr_args > 0)
	{
		const kern_expression *karg;

		if (kexp->nr_args == 1)
			appendStringInfo(buf, " arg=");
		else
			appendStringInfo(buf, " args=[");
		for (i=0, karg = KEXP_FIRST_ARG(kexp);
			 i < kexp->nr_args;
			 i++, karg = KEXP_NEXT_ARG(karg))
		{
			const kern_projection_desc *desc = &kexp->u.proj.desc[i];

			if (!__KEXP_IS_VALID(kexp, karg))
				elog(ERROR, "XpuCode looks corrupted");
			if (i > 0)
				appendStringInfo(buf, ", ");
			appendStringInfo(buf, "%d:", desc->slot_id);
			__xpucode_to_cstring(buf, karg, css, es, ancestors);
		}
		if (kexp->nr_args > 1)
			appendStringInfoChar(buf, ']');
	}
	appendStringInfoChar(buf, '}');
}

static void
__xpucode_to_cstring(StringInfo buf,
					 const kern_expression *kexp,
					 const CustomScanState *css,	/* optional */
					 ExplainState *es,				/* optional */
					 List *ancestors)				/* optionsl */
{
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
		case FuncOpCode__LoadVars:
			__xpucode_loadvars_cstring(buf, kexp, css, es, ancestors);
			break;
		case FuncOpCode__Projection:
			__xpucode_projection_cstring(buf, kexp, css, es, ancestors);
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
		default:
			{
				devtype_info *dtype = devtype_lookup_by_opcode(kexp->exptype);
				devfunc_info *dfunc = devfunc_lookup_by_opcode(kexp->opcode);
			
				appendStringInfo(buf, "{Func::%s(%s)",
								 dfunc->func_name,
								 dtype->type_name);
			}
			break;
	}
	if (kexp->nr_args > 0)
	{
		const kern_expression *karg;
		int			i;

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
			__xpucode_to_cstring(buf, karg, css, es, ancestors);
		}
		if (kexp->nr_args > 1)
			appendStringInfoChar(buf, ']');
	}
	appendStringInfoChar(buf, '}');
}

void
pgstrom_explain_xpucode(StringInfo buf,
						bytea *xpu_code,
						const CustomScanState *css,
						ExplainState *es,
						List *ancestors)
{
	__xpucode_to_cstring(buf, (const kern_expression *)VARDATA(xpu_code),
						 css, es, ancestors);
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
