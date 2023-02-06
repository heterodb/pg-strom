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
	  TypeOpCode__##NAME, DEVKIND__ANY,		\
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
	{NULL, "varchar", TypeOpCode__text, DEVKIND__ANY,
	 devtype_text_hash, sizeof(xpu_text_t), TEXTOID},
	{NULL, "cidr",    TypeOpCode__inet, DEVKIND__ANY,
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
codegen_context_init(codegen_context *context, uint32_t devkind)
{
	memset(context, 0, sizeof(codegen_context));
	context->elevel = ERROR;
	context->required_flags = devkind;
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
	int				pos;

	if (param->paramkind != PARAM_EXTERN)
		__Elog("Only PARAM_EXTERN is supported on device: %d",
			   (int)param->paramkind);
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
					   int kvar_depth,
					   int kvar_resno,
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
		kexp.expflags = context->kexp_flags;
		kexp.opcode = FuncOpCode__VarExpr;
		kexp.u.v.var_typlen = typlen;
		kexp.u.v.var_typbyval = typbyval;
		kexp.u.v.var_typalign = typealign_get_width(typalign);
		kexp.u.v.var_slot_id = kvar_slot_id;
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
			}
		}
		else if (IsA(node, PathTarget))
		{
			PathTarget *reltarget = (PathTarget *)node;

			resno = 1;
			foreach (lc2, reltarget->exprs)
			{
				if (equal(expr, lfirst(lc2)))
					goto found;
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
	return slot_id;
}

static int
codegen_expression_walker(codegen_context *context,
						  StringInfo buf, Expr *expr)
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
		return codegen_var_expression(context, buf, expr,
									  __depth, __resno, __slot_id);

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
		case T_CoalesceExpr:
		case T_MinMaxExpr:
		case T_RelabelType:
		case T_CoerceViaIO:
		case T_CoerceToDomain:
		case T_CaseExpr:
		case T_CaseTestExpr:
		case T_ScalarArrayOpExpr:
		default:
			__Elog("not a supported expression type: %s", nodeToString(expr));
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
	kern_expression *kexp;
	StringInfoData buf;
	ListCell   *lc1, *lc2;
	int			i, nloads = list_length(context->kvars_depth);
	size_t		sz;

	sz = MAXALIGN(offsetof(kern_expression, u.load.kvars[nloads]));
	kexp = alloca(sz);
	kexp->exptype = karg->exptype;
	kexp->expflags = context->kexp_flags;
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
	kexp->args_offset = sz;

	initStringInfo(&buf);
	buf.len = VARHDRSZ;
	appendBinaryStringInfo(&buf, (const char *)kexp, kexp->args_offset);
	appendBinaryStringInfo(&buf, (const char *)karg, karg->len);
	__appendKernExpMagicAndLength(&buf, VARHDRSZ);
	SET_VARSIZE(buf.data, buf.len);

	/* summarize codegen_context */
	context->kvars_nslots = Max(context->kvars_nslots, nloads);
	context->kvars_depth = NIL;
    context->kvars_resno = NIL;

	return (bytea *)buf.data;
}

/*
 * codegen_build_loadvars
 */
static int
kern_preload_vars_item_comp(const void *__a, const void *__b)
{
	const kern_preload_vars_item *a = __a;
	const kern_preload_vars_item *b = __b;

	Assert(a->var_depth == b->var_depth);
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
	ListCell   *lc1, *lc2;

	initStringInfo(&buf);
	buf.len = offsetof(kern_expression, u.load.kvars);
	forboth (lc1, context->kvars_depth,
			 lc2, context->kvars_resno)
	{
		kern_preload_vars_item	vitem;

		vitem.var_depth = lfirst_int(lc1);
		vitem.var_resno = lfirst_int(lc2);
		vitem.var_slot_id = slot_id++;
		if (vitem.var_depth == depth)
		{
			appendBinaryStringInfo(&buf, (char *)&vitem,
								   sizeof(kern_preload_vars_item));
			nloads++;
		}
	}
	if (nloads == 0)
	{
		pfree(buf.data);
		return NULL;
	}
	qsort(buf.data + offsetof(kern_expression, u.load.kvars),
		  nloads,
		  sizeof(kern_preload_vars_item),
		  kern_preload_vars_item_comp);

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
	int			num_valid = 0;
	uint32_t	sz, pos;
	char	   *xpucode = NULL;
	ListCell   *lc;

	foreach (lc, context->kvars_depth)
	{
		int		depth = lfirst_int(lc);

		if (depth >= 0)
			max_depth = depth;
	}
	if (max_depth < 1)
		return NULL;
	sz = MAXALIGN(offsetof(kern_expression,
						   u.pack.subexp_offset[max_depth+1]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype  = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode   = FuncOpCode__Packed;
	kexp->nr_args  = max_depth + 1;
	kexp->args_offset = sz;

	initStringInfo(&buf);
	buf.len = sz;
	for (int d=1; d <= max_depth; d++)
	{
		kern_expression *karg = __codegen_build_loadvars_one(context, d);

		if (!karg)
			continue;
		pos = __appendBinaryStringInfo(&buf, karg, karg->len);
		kexp->u.pack.subexp_offset[d-1] = pos;
		pfree(karg);
		num_valid++;
	}
	__appendKernExpMagicAndLength(&buf, 0);
	if (num_valid == 0)
	{
		pfree(buf.data);
		return NULL;
	}
	memcpy(buf.data, kexp, sz);

	xpucode = palloc(VARHDRSZ + buf.len);
	memcpy(xpucode + VARHDRSZ, buf.data, buf.len);
	SET_VARSIZE(xpucode, VARHDRSZ + buf.len);
	pfree(buf.data);

	return (bytea *) xpucode;
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
 * codegen_build_projection
 */
bytea *
codegen_build_projection(codegen_context *context, List *tlist_dev)
{
	kern_expression	*kexp;
	StringInfoData arg;
	StringInfoData buf;
	List	   *proj_prep_slots = NIL;
	List	   *proj_prep_types = NIL;
	List	   *proj_dest_slots = NIL;
	List	   *proj_dest_types = NIL;
	bool		meet_resjunk = false;
	ListCell   *lc1, *lc2;
	int			nexprs;
	int			nattrs;
	int			i, sz;
	char	   *result;

	initStringInfo(&arg);
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

		if ((__slot_id = is_expression_equals_tlist(context,
													tle->expr,
													&__depth,
													&__resno)) < 0)
		{
			/* not a simple var reference. run an expression on device. */
			context->top_expr = tle->expr;
			if (codegen_expression_walker(context, &arg, tle->expr) < 0)
				return NULL;

			/* reserve a slot_id */
			__slot_id = list_length(context->kvars_depth);
			Assert(__slot_id == list_length(context->kvars_resno));
			context->kvars_depth = lappend_int(context->kvars_depth, -1);
			context->kvars_resno = lappend_int(context->kvars_resno, tle->resno);

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
	initStringInfo(&buf);
	buf.len = sz;
	enlargeStringInfo(&buf, 0);
	kexp = (kern_expression *)buf.data;
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
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
	appendBinaryStringInfo(&buf, arg.data, arg.len);
	__appendKernExpMagicAndLength(&buf, 0);
	pfree(arg.data);

	result = palloc(VARHDRSZ+buf.len);
	memcpy(result+VARHDRSZ, buf.data, buf.len);
	SET_VARSIZE(result, VARHDRSZ+buf.len);
	pfree(buf.data);

	return (bytea *)result;
}

/*
 * __codegen_build_joinquals
 */
static bytea *
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

	context->kvars_depth = context->kvars_resno = NIL;
	initStringInfo(&buf);
	memset(&kexp, 0, sizeof(kexp));
	kexp.exptype = TypeOpCode__int4;
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

	return attach_varloads_xpucode(context, (kern_expression *)buf.data);
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
	sz = MAXALIGN(offsetof(kern_expression,
						   u.pack.subexp_offset[nrels]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Packed;
	kexp->nr_args = nrels;
	kexp->args_offset = sz;

	initStringInfo(&buf);
	buf.len = sz;
	i = 0;
	forboth (lc1, stacked_join_quals,
			 lc2, stacked_other_quals)
	{
		List   *join_quals = lfirst(lc1);
		List   *other_quals = lfirst(lc2);
		bytea  *xpucode;

		xpucode = __codegen_build_joinquals(context,
											join_quals,
											other_quals);
		if (!xpucode)
			kexp->u.pack.subexp_offset[i++] = 0;
		else
		{
			kern_expression *karg = (kern_expression *)VARDATA(xpucode);
			
			Assert(VARSIZE(xpucode) == karg->len + VARHDRSZ);
			kexp->u.pack.subexp_offset[i++] =
				__appendBinaryStringInfo(&buf, karg, karg->len);
			pfree(xpucode);
		}
	}
	Assert(nrels == i);
	if (buf.len > sz)
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
	bytea	   *xpucode;
	ListCell   *lc;

	if (hash_keys == NIL)
		return NULL;

	context->kvars_depth = context->kvars_resno = NIL;
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

	xpucode = attach_varloads_xpucode(context, (kern_expression *)buf.data);
	pfree(buf.data);

	return (kern_expression *)VARDATA(xpucode);
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
	sz = MAXALIGN(offsetof(kern_expression,
						   u.pack.subexp_offset[nrels]));
	kexp = alloca(sz);
	memset(kexp, 0, sz);
	kexp->exptype = TypeOpCode__int4;
	kexp->expflags = context->kexp_flags;
	kexp->opcode  = FuncOpCode__Packed;
	kexp->nr_args = nrels;
	kexp->args_offset = sz;

	initStringInfo(&buf);
	buf.len = sz;
	i = 0;
	foreach (lc, stacked_hash_keys)
	{
		List   *hash_keys = lfirst(lc);

		if (hash_keys == NIL)
			kexp->u.pack.subexp_offset[i++] = 0;
		else
		{
			kern_expression *karg;

			Assert(IsA(hash_keys, List));
			karg = __codegen_build_hash_value(context, hash_keys);
			Assert(karg->exptype == TypeOpCode__int4 &&
				   karg->opcode  == FuncOpCode__HashValue);
			kexp->u.pack.subexp_offset[i++]
				= __appendBinaryStringInfo(&buf, karg, karg->len);
		}
	}
	if (buf.len > sz)
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
 * pgstrom_xpu_expression
 */
bool
pgstrom_xpu_expression(Expr *expr,
					   uint32_t devkind,
					   List *input_rels_tlist,
					   int *p_devcost)
{
	codegen_context context;

	Assert(devkind == DEVKIND__NVIDIA_GPU ||
		   devkind == DEVKIND__NVIDIA_DPU);
	memset(&context, 0, sizeof(context));
	context.elevel = DEBUG2;
	context.top_expr = expr;
	context.required_flags = devkind;
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
			Var		   *kvar;

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
			TargetEntry *tle;

			plan = list_nth(cscan->custom_plans, vitem->var_depth - 1);
			//hoge
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
							 List *dcontext)
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
			__xpucode_to_cstring(buf, karg, css, es, dcontext);
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
					 List *dcontext)				/* optionsl */
{
	const kern_expression *karg;
	int		i;

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
			__xpucode_projection_cstring(buf, kexp, css, es, dcontext);
			return;
		case FuncOpCode__LoadVars:
			__xpucode_loadvars_cstring(buf, kexp, css, es, dcontext);
			break;
		case FuncOpCode__HashValue:
			appendStringInfo(buf, "{HashValue");
			break;
		case FuncOpCode__JoinQuals:
			appendStringInfo(buf, "{JoinQuals");
			for (i=0, karg=KEXP_FIRST_ARG(kexp);
				 i < kexp->nr_args;
				 i++, karg=KEXP_NEXT_ARG(karg))
			{
				if (!__KEXP_IS_VALID(kexp,karg))
					elog(ERROR, "XpuCode looks corrupted");
				appendStringInfo(buf, "%s ", i > 0 ? "," : "");
				if ((karg->expflags & KEXP_FLAG__IS_PUSHED_DOWN) != 0)
					appendStringInfoString(buf, "[pushdown]");
				__xpucode_to_cstring(buf, karg, css, es, dcontext);
			}
			appendStringInfo(buf, "}");
			return;
		case FuncOpCode__Packed:
			appendStringInfo(buf, "{Packed");
			for (i=0, karg=KEXP_FIRST_ARG(kexp);
				 i < kexp->nr_args;
				 i++, karg=KEXP_NEXT_ARG(karg))
			{
				if (!__KEXP_IS_VALID(kexp,karg))
					elog(ERROR, "XpuCode looks corrupted");
				if (i > 0)
					appendStringInfoChar(buf, ',');
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

	if (es->verbose)
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
